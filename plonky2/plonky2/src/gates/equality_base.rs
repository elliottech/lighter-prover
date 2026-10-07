#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};
extern crate alloc;
use alloc::string::ToString;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::{BoolTarget, Target};
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate specialized for Equality Checks
#[derive(Debug, Clone, Default)]
pub struct EqualityGate {
    /// Number of additions operations performed by an Equality gate.
    pub num_ops: usize,
}

impl EqualityGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }
    //Number of routed wires necessary for an operation
    const ROUTED_PER_OP: usize = 3;
    const NOT_ROUTED_PER_OP: usize = 3;
    const TOTAL_PER_OP: usize = Self::ROUTED_PER_OP + Self::NOT_ROUTED_PER_OP;
    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let routed_packed_count = config.num_routed_wires / Self::ROUTED_PER_OP;
        let unrouted_packed_count = config.num_wires / Self::TOTAL_PER_OP;
        if routed_packed_count < unrouted_packed_count {
            routed_packed_count
        } else {
            unrouted_packed_count
        }
    }

    pub(crate) const fn wire_ith_element_0(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i
    }
    pub(crate) const fn wire_ith_element_1(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 1
    }
    pub(crate) const fn wire_ith_output(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 2
    }

    pub(crate) const fn wire_ith_temporary(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < Self::NOT_ROUTED_PER_OP);
        Self::ROUTED_PER_OP * self.num_ops + i * Self::NOT_ROUTED_PER_OP + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for EqualityGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let const_0 = vars.local_constants[0]; //"one" value
        let mut constraints = Vec::with_capacity(self.num_ops * 4);

        for i in 0..self.num_ops {
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let equal = vars.local_wires[self.wire_ith_output(i)];
            let diff = vars.local_wires[self.wire_ith_temporary(i, 0)];
            let invdiff = vars.local_wires[self.wire_ith_temporary(i, 1)];
            let prod = vars.local_wires[self.wire_ith_temporary(i, 2)];
            constraints.push((x - y) - diff);
            constraints.push((diff * invdiff) - prod);
            constraints.push((prod * diff) - diff);
            constraints.push((const_0 - prod) - equal);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_ops * 4 * n);
        let wires = vars_base.local_wires;
        let col = |w: usize| &wires[w * n..][..n];
        // First (and only) constant column: the "one" value.
        let const_0 = &vars_base.local_constants[..n];

        // Batches are 32 points in this prover; keep the scratch row on the
        // stack and fall back to the heap only for oversized batches.
        let mut scratch_stack = [F::ZERO; 64];
        let mut scratch_heap;
        let scratch: &mut [F] = if n <= 64 {
            &mut scratch_stack[..n]
        } else {
            scratch_heap = vec![F::ZERO; n];
            &mut scratch_heap
        };
        let mut constraint_index = 0;
        // Mirrors the packed batch path constraint-for-constraint; each row
        // lands in `scratch` and is folded straight into the shared
        // accumulator instead of a materialized matrix.
        macro_rules! emit {
            () => {{
                let combined = &mut combined_gate_constraints
                    [constraint_index * n..(constraint_index + 1) * n];
                batch_multiply_add_inplace(combined, &scratch, filters);
                constraint_index += 1;
            }};
        }

        for i in 0..self.num_ops {
            let x = col(self.wire_ith_element_0(i));
            let y = col(self.wire_ith_element_1(i));
            let equal = col(self.wire_ith_output(i));
            let diff = col(self.wire_ith_temporary(i, 0));
            let invdiff = col(self.wire_ith_temporary(i, 1));
            let prod = col(self.wire_ith_temporary(i, 2));

            for p in 0..n {
                scratch[p] = (x[p] - y[p]) - diff[p];
            }
            emit!();
            for p in 0..n {
                scratch[p] = (diff[p] * invdiff[p]) - prod[p];
            }
            emit!();
            for p in 0..n {
                scratch[p] = (prod[p] * diff[p]) - diff[p];
            }
            emit!();
            for p in 0..n {
                scratch[p] = (const_0[p] - prod[p]) - equal[p];
            }
            emit!();
        }

        debug_assert_eq!(constraint_index, self.num_ops * 4);
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let mut constraints = Vec::with_capacity(self.num_ops * 4);

        for i in 0..self.num_ops {
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let equal = vars.local_wires[self.wire_ith_output(i)];
            let diff = vars.local_wires[self.wire_ith_temporary(i, 0)];
            let invdiff = vars.local_wires[self.wire_ith_temporary(i, 1)];
            let prod = vars.local_wires[self.wire_ith_temporary(i, 2)];

            constraints.push({
                let inner = builder.sub_extension(x, y);
                builder.sub_extension(inner, diff)
            });
            constraints.push(builder.mul_sub_extension(diff, invdiff, prod));
            constraints.push(builder.mul_sub_extension(prod, diff, diff));
            let inner = builder.sub_extension(const_0, prod);
            constraints.push(builder.sub_extension(inner, equal))
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let mut result: Vec<WitnessGeneratorRef<F, D>> = (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    EqualityBaseGenerator {
                        gate: self.clone(),
                        row,
                        const_0: local_constants[0],
                        i,
                    }
                    .adapter(),
                )
            })
            .collect();
        // One extra generator per row batches all of the row's `invdiff` inversions
        // once every slot's `diff` is known. See `EqualityRowInverseGenerator` for why
        // this is split from the per-slot generators rather than fused with them.
        result.push(WitnessGeneratorRef::new(
            EqualityRowInverseGenerator {
                gate: self.clone(),
                row,
            }
            .adapter(),
        ));
        result
    }

    /// This gate emits one generator per op *plus* one row-inverse generator, so the
    /// default implementation (`generators().len()`) would over-count by one. Both
    /// `find_slot`'s slot bookkeeping and the incomplete-row default-wiring pass in
    /// `try_build_with_options` must see the true op count; with the default, unused
    /// slot inputs stop being tied to zero and witness generation hangs.
    fn num_ops(&self) -> usize {
        self.num_ops
    }

    fn num_wires(&self) -> usize {
        self.num_ops * Self::TOTAL_PER_OP
    }

    fn num_constants(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 4
    }

    fn input_wires_defaults(&self, index: usize) -> Vec<(usize, F)> {
        Vec::from([
            (self.wire_ith_element_0(index), F::ZERO),
            (self.wire_ith_element_1(index), F::ZERO),
        ])
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for EqualityGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_0 = vars.local_constants[0];
        for i in 0..self.num_ops {
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let equal = vars.local_wires[self.wire_ith_output(i)];
            let diff = vars.local_wires[self.wire_ith_temporary(i, 0)];
            let invdiff = vars.local_wires[self.wire_ith_temporary(i, 1)];
            let prod = vars.local_wires[self.wire_ith_temporary(i, 2)];

            yield_constr.one((x - y) - diff);
            yield_constr.one((diff * invdiff) - prod);
            yield_constr.one((prod * diff) - diff);
            yield_constr.one((const_0 - prod) - equal);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EqualityBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub gate: EqualityGate,
    pub row: usize,
    pub const_0: F,
    pub i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for EqualityBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "EqualityBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            self.gate.wire_ith_element_0(self.i),
            self.gate.wire_ith_element_1(self.i),
        ]
        .iter()
        .map(|&i| Target::wire(self.row, i))
        .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let x = get_wire(self.gate.wire_ith_element_0(self.i));
        let y = get_wire(self.gate.wire_ith_element_1(self.i));
        let equal = Target::wire(self.row, self.gate.wire_ith_output(self.i));
        let diff = Target::wire(self.row, self.gate.wire_ith_temporary(self.i, 0));
        let prod = Target::wire(self.row, self.gate.wire_ith_temporary(self.i, 2));

        let prod_value = if x != y { F::ONE } else { F::ZERO };

        // `invdiff` is deliberately *not* filled here: computing `(x - y).inverse()` per
        // slot costs a full Fermat inversion for every unequal pair. The row-level
        // `EqualityRowInverseGenerator` fills every `invdiff` of this row with a single
        // shared batch inversion once all of the row's `diff` values are available.
        out_buffer.set_target(diff, x - y)?;
        out_buffer.set_bool_target(BoolTarget::new_unsafe(equal), x == y)?;
        out_buffer.set_target(prod, prod_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = EqualityGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            const_0,
            i,
        })
    }
}

/// Fills the `invdiff` temporary of every slot of one `EqualityGate` row using a single
/// shared field inversion (Montgomery's batch-inverse trick) instead of one full Fermat
/// inversion per unequal slot: 1 inversion + ~3 multiplications per unequal slot.
///
/// This is deliberately *split* from `EqualityBaseGenerator` instead of fused into one
/// whole-row generator. Real circuits copy-constrain an equality *output* into an
/// equality *input on the same row* (chained comparisons packed by `find_slot`), so a
/// fused generator depending on all of the row's input wires would depend on its own
/// output and deadlock. In the split design the per-slot generators fire exactly as
/// before (chained comparisons resolve in the old order), while this generator only
/// waits on the row's `diff` temporaries. `invdiff` lives on a non-routed wire, so it
/// cannot be copy-constrained and nothing outside the row's own constraints can depend
/// on it; deferring it to row completion therefore cannot create a cycle.
///
/// Value identity: field inverses are unique, so the batched computation produces the
/// same field element — and, with the Goldilocks implementation used here, the same
/// canonical bit pattern — as the per-slot `(x - y).inverse()` it replaces. Slots with
/// `diff == 0` keep `invdiff = 0` exactly as before.
#[derive(Clone, Debug, Default)]
pub struct EqualityRowInverseGenerator {
    pub gate: EqualityGate,
    pub row: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for EqualityRowInverseGenerator
{
    fn id(&self) -> String {
        "EqualityRowInverseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..self.gate.num_ops)
            .map(|i| Target::wire(self.row, self.gate.wire_ith_temporary(i, 0)))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let num_ops = self.gate.num_ops;
        let diffs: Vec<F> = (0..num_ops)
            .map(|i| witness.get_target(Target::wire(self.row, self.gate.wire_ith_temporary(i, 0))))
            .collect();

        // Montgomery's trick over the row's nonzero diffs. Forward pass: prefix products
        // of the nonzero diffs (zero diffs keep the running product unchanged and their
        // `invdiff` stays zero, as in the per-slot formula).
        let mut prefixes: Vec<F> = Vec::with_capacity(num_ops);
        let mut acc = F::ONE;
        let mut any_nonzero = false;
        for &diff in &diffs {
            prefixes.push(acc);
            if diff != F::ZERO {
                acc *= diff;
                any_nonzero = true;
            }
        }

        // One shared inversion for the whole row; skipped entirely on all-equal rows,
        // which also performed no inversions under the per-slot scheme.
        let mut inv_acc = if any_nonzero { acc.inverse() } else { F::ONE };

        // Backward pass: peel individual inverses off the shared inverse. Reuse
        // `prefixes` to hold each slot's `invdiff` value.
        for i in (0..num_ops).rev() {
            let diff = diffs[i];
            if diff != F::ZERO {
                prefixes[i] *= inv_acc;
                inv_acc *= diff;
            } else {
                prefixes[i] = F::ZERO;
            }
        }

        for (i, inv_value) in prefixes.into_iter().enumerate() {
            let invdiff = Target::wire(self.row, self.gate.wire_ith_temporary(i, 1));
            out_buffer.set_target(invdiff, inv_value)?;
        }
        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        <EqualityGate as Gate<F, D>>::serialize(&self.gate, dst, common_data)?;
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = <EqualityGate as Gate<F, D>>::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        Ok(Self { gate, row })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    #[allow(unused_imports)]
    use crate::field::types::Field64;
    use crate::field::types::{Field, PrimeField64, Sample};
    use crate::gates::equality_base::EqualityGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::target::{BoolTarget, Target};
    use crate::iop::witness::{PartialWitness, Witness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::util::serialization::{
        Buffer, DefaultGeneratorSerializer, WitnessGeneratorSerializer,
    };

    #[test]
    fn low_degree() {
        let gate = EqualityGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = EqualityGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn direct_filtered_accumulation_matches_materialized_batch() {
        const D: usize = 2;
        const N: usize = 11;
        type F = GoldilocksField;
        use crate::gates::gate::Gate;

        let gate = EqualityGate::new_from_config(&CircuitConfig::standard_recursion_config());
        let num_wires = <EqualityGate as Gate<F, D>>::num_wires(&gate);
        let num_constants = <EqualityGate as Gate<F, D>>::num_constants(&gate);
        let num_constraints = <EqualityGate as Gate<F, D>>::num_constraints(&gate);
        let wires = (0..num_wires * N)
            .map(|i| F::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let constants = (0..num_constants * N)
            .map(|i| F::from_canonical_usize(7 * i + 2))
            .collect::<Vec<_>>();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let vars = crate::plonk::vars::EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
        let filters = (0..N)
            .map(|i| F::from_canonical_usize(2 * i + 1))
            .collect::<Vec<_>>();
        let mut expected = vec![F::ZERO; num_constraints * N];
        let materialized = <EqualityGate as Gate<F, D>>::eval_unfiltered_base_batch(&gate, vars);
        for (acc, constraints) in expected
            .chunks_exact_mut(N)
            .zip(materialized.chunks_exact(N))
        {
            crate::field::batch_util::batch_multiply_add_inplace(acc, constraints, &filters);
        }
        let mut actual = vec![F::ZERO; expected.len()];
        <EqualityGate as Gate<F, D>>::eval_unfiltered_base_batch_accumulate(
            &gate,
            vars,
            &filters,
            &mut actual,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_succes() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        // Create targets for x and y
        let x = builder.add_virtual_target();
        let y = builder.add_virtual_target();

        // Instantiate your custom EqualityGate and get BoolTarget
        let gate = EqualityGate::new_from_config(&config);
        let ref_gate = gate.clone();
        let constants = [F::ONE];
        let (gate_row, i) = builder.find_slot(gate, &constants, &constants);

        let wire_x = Target::wire(gate_row, ref_gate.wire_ith_element_0(i));
        let wire_y = Target::wire(gate_row, ref_gate.wire_ith_element_1(i));
        let wire_equal = Target::wire(gate_row, ref_gate.wire_ith_output(i));

        builder.connect(x, wire_x);
        builder.connect(y, wire_y);

        let equal = BoolTarget::new_unsafe(wire_equal);

        // Optionally use equal in the circuit logic
        builder.assert_bool(equal);

        let circuit_data = builder.build::<C>();

        // Now set values for x and y such that x == y, so equal = 1
        let mut pw = PartialWitness::new();
        let value1 = F::from_canonical_u64(17);
        let value2 = F::from_canonical_u64(18);
        pw.set_target(x, value1)?;
        pw.set_target(y, value2)?;

        let proof = circuit_data.prove(pw)?;
        circuit_data.verify(proof)?;

        Ok(())
    }

    /// Differential test: run the full generator pipeline (per-slot generators + the
    /// row-inverse generator) over multiple equality rows and assert that every
    /// generated wire matches the old per-slot formulas — as field elements *and* as
    /// canonical u64 bit patterns.
    #[test]
    fn row_inverse_generator_matches_per_slot_values() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let gate = EqualityGate::new_from_config(&config);
        let ops_per_row = gate.num_ops;
        // One full row plus a partially-filled second row, so default-filled unused
        // slots go through the row-inverse path as well.
        let num_pairs = ops_per_row + 7;

        // Edge cases first (equal pairs, generic unequal, zero-vs-nonzero, p-1),
        // then random pairs.
        let mut pair_values: Vec<(F, F)> = vec![
            (F::ZERO, F::ZERO),
            (F::from_canonical_u64(17), F::from_canonical_u64(17)),
            (F::ZERO, F::from_canonical_u64(29)),
            (F::from_canonical_u64(29), F::ZERO),
            (F::NEG_ONE, F::ZERO),
            (F::NEG_ONE, F::ONE),
            (F::ONE, F::NEG_ONE),
            (F::NEG_ONE, F::NEG_ONE),
        ];
        while pair_values.len() < num_pairs {
            pair_values.push((F::rand(), F::rand()));
        }

        let constants = [F::ONE];
        let mut slot_rows = Vec::new();
        let mut inputs = Vec::new();
        for _ in 0..num_pairs {
            let x = builder.add_virtual_target();
            let y = builder.add_virtual_target();
            let (row, i) = builder.find_slot(gate.clone(), &constants, &constants);
            builder.connect(x, Target::wire(row, gate.wire_ith_element_0(i)));
            builder.connect(y, Target::wire(row, gate.wire_ith_element_1(i)));
            slot_rows.push(row);
            inputs.push((x, y));
        }
        let circuit_data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        for ((x, y), (vx, vy)) in inputs.iter().zip(&pair_values) {
            pw.set_target(*x, *vx)?;
            pw.set_target(*y, *vy)?;
        }
        let witness =
            generate_partial_witness(pw, &circuit_data.prover_only, &circuit_data.common)?;

        let mut rows = slot_rows;
        rows.sort_unstable();
        rows.dedup();
        assert!(rows.len() >= 2, "test should span multiple equality rows");

        let mut checked = 0;
        for &row in &rows {
            for i in 0..ops_per_row {
                let get = |column: usize| witness.get_target(Target::wire(row, column));
                let x = get(gate.wire_ith_element_0(i));
                let y = get(gate.wire_ith_element_1(i));
                let equal = get(gate.wire_ith_output(i));
                let diff = get(gate.wire_ith_temporary(i, 0));
                let invdiff = get(gate.wire_ith_temporary(i, 1));
                let prod = get(gate.wire_ith_temporary(i, 2));

                // The old per-slot formulas.
                let expected_diff = x - y;
                let expected_inv = if x != y { (x - y).inverse() } else { F::ZERO };
                let expected_prod = if x != y { F::ONE } else { F::ZERO };
                let expected_equal = if x == y { F::ONE } else { F::ZERO };

                assert_eq!(diff, expected_diff);
                assert_eq!(invdiff, expected_inv);
                assert_eq!(prod, expected_prod);
                assert_eq!(equal, expected_equal);
                assert_eq!(diff.to_canonical_u64(), expected_diff.to_canonical_u64());
                assert_eq!(invdiff.to_canonical_u64(), expected_inv.to_canonical_u64());
                assert_eq!(prod.to_canonical_u64(), expected_prod.to_canonical_u64());
                assert_eq!(equal.to_canonical_u64(), expected_equal.to_canonical_u64());

                if x != y {
                    assert_eq!(invdiff * diff, F::ONE);
                } else {
                    assert_eq!(invdiff, F::ZERO);
                }
                checked += 1;
            }
        }
        assert_eq!(checked, rows.len() * ops_per_row);

        Ok(())
    }

    /// End-to-end proof over multiple equality rows mixing equal and unequal slots,
    /// including same-row chaining (a comparison whose input is another comparison's
    /// output on the same row) — the pattern that deadlocks a fused whole-row
    /// generator and that the split design must survive.
    #[test]
    fn row_inverse_generator_proves_mixed_rows() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let gate = EqualityGate::new_from_config(&config);
        let num_pairs = gate.num_ops + 3;
        let constants = [F::ONE];

        let mut pw = PartialWitness::new();
        let mut prev_equal: Option<Target> = None;
        for k in 0..num_pairs {
            let x = builder.add_virtual_target();
            let (row, i) = builder.find_slot(gate.clone(), &constants, &constants);
            builder.connect(x, Target::wire(row, gate.wire_ith_element_0(i)));
            let wire_y = Target::wire(row, gate.wire_ith_element_1(i));
            // Chain every other comparison's second input to the previous
            // comparison's output, which usually lives on the same row.
            match prev_equal {
                Some(prev) if k % 2 == 1 => builder.connect(prev, wire_y),
                _ => {
                    let y = builder.add_virtual_target();
                    builder.connect(y, wire_y);
                    pw.set_target(y, F::from_canonical_u64((k % 3) as u64))?;
                }
            }
            pw.set_target(x, F::from_canonical_u64((k % 2) as u64))?;
            let equal = BoolTarget::new_unsafe(Target::wire(row, gate.wire_ith_output(i)));
            builder.assert_bool(equal);
            prev_equal = Some(equal.target);
        }

        let circuit_data = builder.build::<C>();
        let proof = circuit_data.prove(pw)?;
        circuit_data.verify(proof)
    }

    /// Serialization round-trip through the default generator serializer for both the
    /// per-slot generator and the row-inverse generator.
    #[test]
    fn equality_generators_serialization_roundtrip() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let x = builder.add_virtual_target();
        let y = builder.add_virtual_target();
        let gate = EqualityGate::new_from_config(&config);
        let constants = [F::ONE];
        let (row, i) = builder.find_slot(gate.clone(), &constants, &constants);
        builder.connect(x, Target::wire(row, gate.wire_ith_element_0(i)));
        builder.connect(y, Target::wire(row, gate.wire_ith_element_1(i)));
        let circuit_data = builder.build::<C>();

        let serializer = DefaultGeneratorSerializer::<C, D>::default();
        let mut seen_per_slot = false;
        let mut seen_row_inverse = false;
        for generator in &circuit_data.prover_only.generators {
            let id = generator.0.id();
            if id != "EqualityBaseGenerator" && id != "EqualityRowInverseGenerator" {
                continue;
            }
            seen_per_slot |= id == "EqualityBaseGenerator";
            seen_row_inverse |= id == "EqualityRowInverseGenerator";

            let mut bytes = Vec::new();
            serializer
                .write_generator(&mut bytes, generator, &circuit_data.common)
                .expect("write_generator failed");
            let mut buf = Buffer::new(&bytes);
            let restored = serializer
                .read_generator(&mut buf, &circuit_data.common)
                .expect("read_generator failed");
            assert_eq!(restored.0.id(), id);

            let mut bytes2 = Vec::new();
            serializer
                .write_generator(&mut bytes2, &restored, &circuit_data.common)
                .expect("write_generator after round-trip failed");
            assert_eq!(bytes, bytes2);
        }
        assert!(seen_per_slot, "no per-slot equality generator in circuit");
        assert!(
            seen_row_inverse,
            "no row-inverse equality generator in circuit"
        );
        Ok(())
    }
}
