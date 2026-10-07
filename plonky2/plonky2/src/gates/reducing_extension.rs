#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::ops::Range;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Computes `sum alpha^i c_i` for a vector `c_i` of `num_coeffs` elements of the extension field.
#[derive(Debug, Clone, Default)]
pub struct ReducingExtensionGate<const D: usize> {
    pub num_coeffs: usize,
}

#[cfg(feature = "std")]
fn reducing_extension_stack_scratch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var_os("LIGHTER_REDUCING_STACK").is_some_and(|value| value == "0")
    })
}

#[cfg(not(feature = "std"))]
fn reducing_extension_stack_scratch_enabled() -> bool {
    true
}

impl<const D: usize> ReducingExtensionGate<D> {
    pub const fn new(num_coeffs: usize) -> Self {
        Self { num_coeffs }
    }

    pub fn max_coeffs_len(num_wires: usize, num_routed_wires: usize) -> usize {
        // `3*D` routed wires are used for the output, alpha and old accumulator.
        // Need `num_coeffs*D` routed wires for coeffs, and `(num_coeffs-1)*D` wires for accumulators.
        ((num_routed_wires - 3 * D) / D).min((num_wires - 2 * D) / (D * 2))
    }

    pub(crate) const fn wires_output() -> Range<usize> {
        0..D
    }
    pub(crate) const fn wires_alpha() -> Range<usize> {
        D..2 * D
    }
    pub(crate) const fn wires_old_acc() -> Range<usize> {
        2 * D..3 * D
    }
    const START_COEFFS: usize = 3 * D;
    pub(crate) const fn wires_coeff(i: usize) -> Range<usize> {
        Self::START_COEFFS + i * D..Self::START_COEFFS + (i + 1) * D
    }
    const fn start_accs(&self) -> usize {
        Self::START_COEFFS + self.num_coeffs * D
    }
    const fn wires_accs(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_coeffs);
        if i == self.num_coeffs - 1 {
            // The last accumulator is the output.
            return Self::wires_output();
        }
        self.start_accs() + D * i..self.start_accs() + D * (i + 1)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ReducingExtensionGate<D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_coeffs)?;
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let num_coeffs = src.read_usize()?;
        Ok(Self::new(num_coeffs))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(Self::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            constraints.push(acc * alpha + coeffs[i] - accs[i]);
            acc = accs[i];
        }

        constraints
            .into_iter()
            .flat_map(|alg| alg.to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let alpha = vars.get_local_ext(Self::wires_alpha());
        let old_acc = vars.get_local_ext(Self::wires_old_acc());
        let coeffs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext(Self::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            yield_constr.many((acc * alpha + coeffs[i] - accs[i]).to_basefield_array());
            acc = accs[i];
        }
    }

    /// Contiguous-column fused evaluation: reads each wire as a contiguous
    /// `n`-point column, evaluates one accumulator step at a time and
    /// multiply-adds the filtered constraint rows straight into the shared
    /// buffer, avoiding the per-point strided writes and per-point `Vec`
    /// allocations of the default path.
    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= <Self as Gate<F, D>>::num_constraints(self) * n);

        let wires = vars_base.local_wires;
        let ext = |start: usize, p: usize| {
            let mut arr = [F::ZERO; D];
            for (d, a) in arr.iter_mut().enumerate() {
                *a = wires[(start + d) * n + p];
            }
            F::Extension::from_basefield_array(arr)
        };

        const STACK_POINTS: usize = 32;
        const STACK_BASE_VALUES: usize = 128;
        let use_stack = reducing_extension_stack_scratch_enabled()
            && n <= STACK_POINTS
            && D * n <= STACK_BASE_VALUES;
        let mut alpha_stack = [F::Extension::ZERO; STACK_POINTS];
        let mut acc_stack = [F::Extension::ZERO; STACK_POINTS];
        let mut scratch_stack = [F::ZERO; STACK_BASE_VALUES];
        let mut alpha_heap;
        let mut acc_heap;
        let mut scratch_heap;
        let (alphas, accs, scratch): (&mut [F::Extension], &mut [F::Extension], &mut [F]) =
            if use_stack {
                (
                    &mut alpha_stack[..n],
                    &mut acc_stack[..n],
                    &mut scratch_stack[..D * n],
                )
            } else {
                alpha_heap = vec![F::Extension::ZERO; n];
                acc_heap = vec![F::Extension::ZERO; n];
                scratch_heap = vec![F::ZERO; D * n];
                (&mut alpha_heap, &mut acc_heap, &mut scratch_heap)
            };
        for p in 0..n {
            alphas[p] = ext(Self::wires_alpha().start, p);
            accs[p] = ext(Self::wires_old_acc().start, p);
        }

        for i in 0..self.num_coeffs {
            let coeff_start = Self::wires_coeff(i).start;
            let acc_start = self.wires_accs(i).start;
            for p in 0..n {
                let next_acc = ext(acc_start, p);
                let constraint = accs[p] * alphas[p] + ext(coeff_start, p) - next_acc;
                let arr = constraint.to_basefield_array();
                for (d, a) in arr.iter().enumerate() {
                    scratch[d * n + p] = *a;
                }
                accs[p] = next_acc;
            }
            for d in 0..D {
                batch_multiply_add_inplace(
                    &mut combined_gate_constraints[(i * D + d) * n..][..n],
                    &scratch[d * n..][..n],
                    filters,
                );
            }
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(Self::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            let coeff = coeffs[i];
            let mut tmp = builder.mul_add_ext_algebra(acc, alpha, coeff);
            tmp = builder.sub_ext_algebra(tmp, accs[i]);
            constraints.push(tmp);
            acc = accs[i];
        }

        constraints
            .into_iter()
            .flat_map(|alg| alg.to_ext_target_array())
            .collect()
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            ReducingGenerator {
                row,
                gate: self.clone(),
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        2 * D + 2 * D * self.num_coeffs
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        D * self.num_coeffs
    }
}

#[derive(Debug, Default)]
pub struct ReducingGenerator<const D: usize> {
    row: usize,
    gate: ReducingExtensionGate<D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ReducingGenerator<D> {
    fn id(&self) -> String {
        "ReducingExtensionGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        ReducingExtensionGate::<D>::wires_alpha()
            .chain(ReducingExtensionGate::<D>::wires_old_acc())
            .chain((0..self.gate.num_coeffs).flat_map(ReducingExtensionGate::<D>::wires_coeff))
            .map(|i| Target::wire(self.row, i))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_extension = |range: Range<usize>| -> F::Extension {
            let t = ExtensionTarget::from_range(self.row, range);
            witness.get_extension_target(t)
        };

        let alpha = local_extension(ReducingExtensionGate::<D>::wires_alpha());
        let old_acc = local_extension(ReducingExtensionGate::<D>::wires_old_acc());
        let coeffs = (0..self.gate.num_coeffs)
            .map(|i| local_extension(ReducingExtensionGate::<D>::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.gate.num_coeffs)
            .map(|i| ExtensionTarget::from_range(self.row, self.gate.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut acc = old_acc;
        for i in 0..self.gate.num_coeffs {
            let computed_acc = acc * alpha + coeffs[i];
            out_buffer.set_extension_target(accs[i], computed_acc)?;
            acc = computed_acc;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        <ReducingExtensionGate<D> as Gate<F, D>>::serialize(&self.gate, dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = <ReducingExtensionGate<D> as Gate<F, D>>::deserialize(src, _common_data)?;
        Ok(Self { row, gate })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::reducing_extension::ReducingExtensionGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(ReducingExtensionGate::new(22));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(ReducingExtensionGate::new(22))
    }
}
