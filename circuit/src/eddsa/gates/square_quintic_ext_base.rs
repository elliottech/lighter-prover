// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::gates::gate::{Gate, U32QuotientGate};
use plonky2::gates::packed_util::PackedEvaluableBase;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};

use crate::plonky2::util::serialization::{Buffer, IoResult, Read, Write};

#[derive(Debug, Clone, Default)]
pub struct QuinticSquaringGate {
    /// Number of Quintic Squarings performed by a Gate
    pub num_ops: usize,
}

impl QuinticSquaringGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }
    //Number of routed wires necessary for an operation
    const ROUTED_PER_OP: usize = 10;
    const NOT_ROUTED_PER_OP: usize = 10;
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

    pub(crate) const fn wire_ith_multiplicand_jth_limb(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + j
    }
    pub(crate) const fn wire_ith_output_jth_limb(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + 5 + j
    }
    pub(crate) const fn temporary_wire(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 10);
        Self::ROUTED_PER_OP * self.num_ops + i * Self::NOT_ROUTED_PER_OP + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for QuinticSquaringGate {
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
        let const_2 = F::Extension::from_basefield(F::from_canonical_u64(2));
        let const_3 = F::Extension::from_basefield(F::from_canonical_u64(3));
        let const_6 = F::Extension::from_basefield(F::from_canonical_u64(6));
        let mut constraints = Vec::with_capacity(self.num_ops * 15);

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let c = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();

            // Compute each output limb (copied from mul_quintic_ext structure)
            let extra = (0..10)
                .map(|j| vars.local_wires[self.temporary_wire(i, j)])
                .collect::<Vec<_>>();

            //c[0]
            constraints.push(a[0] * a[0] - extra[0]);
            constraints.push((const_6 * a[1] * a[4] + extra[0]) - extra[1]);
            constraints.push((const_6 * a[2] * a[3] + extra[1]) - c[0]);

            //c[1]
            constraints.push(const_3 * a[3] * a[3] - extra[2]);
            constraints.push((const_2 * a[0] * a[1] + extra[2]) - extra[3]);
            constraints.push((const_6 * a[2] * a[4] + extra[3]) - c[1]);

            //c[2]
            constraints.push(a[1] * a[1] - extra[4]);
            constraints.push((const_2 * a[0] * a[2] + extra[4]) - extra[5]);
            constraints.push((const_6 * a[3] * a[4] + extra[5]) - c[2]);

            //c[3]
            constraints.push((const_3 * a[4] * a[4]) - extra[6]);
            constraints.push((const_2 * a[0] * a[3] + extra[6]) - extra[7]);
            constraints.push((const_2 * a[1] * a[2] + extra[7]) - c[3]);

            //c[4]
            constraints.push(a[2] * a[2] - extra[8]);
            constraints.push((const_2 * a[0] * a[4] + extra[8]) - extra[9]);
            constraints.push((const_2 * a[1] * a[3] + extra[9]) - c[4]);
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
        assert!(combined_gate_constraints.len() >= <Self as Gate<F, D>>::num_constraints(self) * n);
        let wires = vars_base.local_wires;
        let col = |w: usize| &wires[w * n..][..n];
        let const_2 = F::from_canonical_u64(2);
        let const_3 = F::from_canonical_u64(3);
        let const_6 = F::from_canonical_u64(6);
        let mut chunks = combined_gate_constraints.chunks_exact_mut(n);

        for i in 0..self.num_ops {
            let a: [&[F]; 5] =
                core::array::from_fn(|j| col(self.wire_ith_multiplicand_jth_limb(i, j)));
            let c: [&[F]; 5] = core::array::from_fn(|j| col(self.wire_ith_output_jth_limb(i, j)));
            let extra: [&[F]; 10] = core::array::from_fn(|j| col(self.temporary_wire(i, j)));
            let outs: [&mut [F]; 15] = core::array::from_fn(|_| chunks.next().unwrap());

            for p in 0..n {
                let ap: [F; 5] = core::array::from_fn(|j| a[j][p]);
                let f = filters[p];
                // Identical expressions and order to `eval_unfiltered_base_packed`.
                outs[0][p] += f * (ap[0] * ap[0] - extra[0][p]);
                outs[1][p] += f * ((const_6 * ap[1] * ap[4] + extra[0][p]) - extra[1][p]);
                outs[2][p] += f * ((const_6 * ap[2] * ap[3] + extra[1][p]) - c[0][p]);

                outs[3][p] += f * (const_3 * ap[3] * ap[3] - extra[2][p]);
                outs[4][p] += f * ((const_2 * ap[0] * ap[1] + extra[2][p]) - extra[3][p]);
                outs[5][p] += f * ((const_6 * ap[2] * ap[4] + extra[3][p]) - c[1][p]);

                outs[6][p] += f * (ap[1] * ap[1] - extra[4][p]);
                outs[7][p] += f * ((const_2 * ap[0] * ap[2] + extra[4][p]) - extra[5][p]);
                outs[8][p] += f * ((const_6 * ap[3] * ap[4] + extra[5][p]) - c[2][p]);

                outs[9][p] += f * ((const_3 * ap[4] * ap[4]) - extra[6][p]);
                outs[10][p] += f * ((const_2 * ap[0] * ap[3] + extra[6][p]) - extra[7][p]);
                outs[11][p] += f * ((const_2 * ap[1] * ap[2] + extra[7][p]) - c[3][p]);

                outs[12][p] += f * (ap[2] * ap[2] - extra[8][p]);
                outs[13][p] += f * ((const_2 * ap[0] * ap[4] + extra[8][p]) - extra[9][p]);
                outs[14][p] += f * ((const_2 * ap[1] * ap[3] + extra[9][p]) - c[4][p]);
            }
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = F::from_canonical_u64(0);
        let const_1 = F::from_canonical_u64(1);
        let const_2 = F::from_canonical_u64(2);
        let const_3 = F::from_canonical_u64(3);
        let const_6 = F::from_canonical_u64(6);

        let mut constraints = Vec::with_capacity(self.num_ops * 24); // 24 intermediate constraints

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let out = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let extra = (0..10)
                .map(|j| vars.local_wires[self.temporary_wire(i, j)])
                .collect::<Vec<_>>();

            let [a0, a1, a2, a3, a4] = <[ExtensionTarget<D>; 5]>::try_from(a).unwrap();
            let [c0, c1, c2, c3, c4] = <[ExtensionTarget<D>; 5]>::try_from(out).unwrap();

            // --- c[0] ---
            let t0 = builder.mul_extension(a0, a0);
            constraints.push(builder.sub_extension(t0, extra[0]));
            let t1 = builder.arithmetic_extension(const_6, const_1, a1, a4, extra[0]);
            constraints.push(builder.sub_extension(t1, extra[1]));
            let t2 = builder.arithmetic_extension(const_6, const_1, a2, a3, extra[1]);
            constraints.push(builder.sub_extension(t2, c0));

            // --- c[1] ---
            let t4 = builder.arithmetic_extension(const_3, const_0, a3, a3, a3);
            constraints.push(builder.sub_extension(t4, extra[2]));
            let t6 = builder.arithmetic_extension(const_2, const_1, a0, a1, extra[2]);
            constraints.push(builder.sub_extension(t6, extra[3]));
            let t8 = builder.arithmetic_extension(const_6, const_1, a2, a4, extra[3]);
            constraints.push(builder.sub_extension(t8, c1));

            // --- c[2] ---
            let t9 = builder.mul_extension(a1, a1);
            constraints.push(builder.sub_extension(t9, extra[4]));
            let t11 = builder.arithmetic_extension(const_2, const_1, a0, a2, extra[4]);
            constraints.push(builder.sub_extension(t11, extra[5]));
            let t13 = builder.arithmetic_extension(const_6, const_1, a3, a4, extra[5]);
            constraints.push(builder.sub_extension(t13, c2));

            // --- c[3] ---
            let t15 = builder.arithmetic_extension(const_3, const_0, a4, a4, a4);
            constraints.push(builder.sub_extension(t15, extra[6]));
            let t17 = builder.arithmetic_extension(const_2, const_1, a0, a3, extra[6]);
            constraints.push(builder.sub_extension(t17, extra[7]));
            let t19 = builder.arithmetic_extension(const_2, const_1, a1, a2, extra[7]);
            constraints.push(builder.sub_extension(t19, c3));

            // --- c[4] ---
            let t20 = builder.mul_extension(a2, a2);
            constraints.push(builder.sub_extension(t20, extra[8]));
            let t22 = builder.arithmetic_extension(const_2, const_1, a0, a4, extra[8]);
            constraints.push(builder.sub_extension(t22, extra[9]));
            let t24 = builder.arithmetic_extension(const_2, const_1, a1, a3, extra[9]);
            constraints.push(builder.sub_extension(t24, c4));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    QuinticSquaringBaseGenerator {
                        gate: self.clone(),
                        row,
                        const_2: F::from_canonical_u64(2),
                        const_3: F::from_canonical_u64(3),
                        const_6: F::from_canonical_u64(6),
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * Self::TOTAL_PER_OP
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 15
    }

    fn u32_quotient_gate(&self) -> Option<U32QuotientGate> {
        Some(U32QuotientGate::QuinticSquaring {
            num_ops: self.num_ops,
        })
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for QuinticSquaringGate
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_2 = P::from(F::from_canonical_u64(2));
        let const_3 = P::from(F::from_canonical_u64(3));
        let const_6 = P::from(F::from_canonical_u64(6));

        for i in 0..self.num_ops {
            let a: [P; 5] = core::array::from_fn(|j| {
                vars.local_wires[self.wire_ith_multiplicand_jth_limb(i, j)]
            });
            let c: [P; 5] =
                core::array::from_fn(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)]);
            let extra: [P; 10] =
                core::array::from_fn(|j| vars.local_wires[self.temporary_wire(i, j)]);

            //c[0]
            yield_constr.one(a[0] * a[0] - extra[0]);
            yield_constr.one((const_6 * a[1] * a[4] + extra[0]) - extra[1]);
            yield_constr.one((const_6 * a[2] * a[3] + extra[1]) - c[0]);

            //c[1]
            yield_constr.one(const_3 * a[3] * a[3] - extra[2]);
            yield_constr.one((const_2 * a[0] * a[1] + extra[2]) - extra[3]);
            yield_constr.one((const_6 * a[2] * a[4] + extra[3]) - c[1]);

            //c[2]
            yield_constr.one(a[1] * a[1] - extra[4]);
            yield_constr.one((const_2 * a[0] * a[2] + extra[4]) - extra[5]);
            yield_constr.one((const_6 * a[3] * a[4] + extra[5]) - c[2]);

            //c[3]
            yield_constr.one((const_3 * a[4] * a[4]) - extra[6]);
            yield_constr.one((const_2 * a[0] * a[3] + extra[6]) - extra[7]);
            yield_constr.one((const_2 * a[1] * a[2] + extra[7]) - c[3]);

            //c[4]
            yield_constr.one(a[2] * a[2] - extra[8]);
            yield_constr.one((const_2 * a[0] * a[4] + extra[8]) - extra[9]);
            yield_constr.one((const_2 * a[1] * a[3] + extra[9]) - c[4]);
        }
    }
}

/// Computes the output limbs and the intermediate (`extra`) wire values for squaring
/// `a` in `F[u]/(u^5 - 3)`, matching the gate's constraint equations.
///
/// With the canonical gate constants (2, 3, 6) the small scalar multiples are computed
/// with addition chains, which yields the same field elements while performing one
/// modular multiplication per product instead of two or three. For any other constants
/// it falls back to the original fully-multiplying arithmetic.
fn quintic_square_wires<F: RichField>(
    a: &[F; 5],
    const_2: F,
    const_3: F,
    const_6: F,
) -> ([F; 5], [F; 10]) {
    let mut extra = [F::ZERO; 10];

    if const_2 == F::TWO
        && const_3 == F::from_canonical_u64(3)
        && const_6 == F::from_canonical_u64(6)
    {
        let two = |x: F| x + x;
        let three = |x: F| x + x + x;
        let six = |x: F| {
            let t = x + x + x;
            t + t
        };

        // c[0]
        extra[0] = a[0] * a[0];
        extra[1] = six(a[1] * a[4]) + extra[0];
        let c0 = six(a[2] * a[3]) + extra[1];

        // c[1]
        extra[2] = three(a[3] * a[3]);
        extra[3] = two(a[0] * a[1]) + extra[2];
        let c1 = six(a[2] * a[4]) + extra[3];

        // c[2]
        extra[4] = a[1] * a[1];
        extra[5] = two(a[0] * a[2]) + extra[4];
        let c2 = six(a[3] * a[4]) + extra[5];

        // c[3]
        extra[6] = three(a[4] * a[4]);
        extra[7] = two(a[0] * a[3]) + extra[6];
        let c3 = two(a[1] * a[2]) + extra[7];

        // c[4]
        extra[8] = a[2] * a[2];
        extra[9] = two(a[0] * a[4]) + extra[8];
        let c4 = two(a[1] * a[3]) + extra[9];

        return ([c0, c1, c2, c3, c4], extra);
    }

    // c[0]
    extra[0] = a[0] * a[0];
    extra[1] = const_6 * a[1] * a[4] + extra[0];
    let c0 = const_6 * a[2] * a[3] + extra[1];

    // c[1]
    extra[2] = const_3 * a[3] * a[3];
    extra[3] = const_2 * a[0] * a[1] + extra[2];
    let c1 = const_6 * a[2] * a[4] + extra[3];

    // c[2]
    extra[4] = a[1] * a[1];
    extra[5] = const_2 * a[0] * a[2] + extra[4];
    let c2 = const_6 * a[3] * a[4] + extra[5];

    // c[3]
    extra[6] = const_3 * a[4] * a[4];
    extra[7] = const_2 * a[0] * a[3] + extra[6];
    let c3 = const_2 * a[1] * a[2] + extra[7];

    // c[4]
    extra[8] = a[2] * a[2];
    extra[9] = const_2 * a[0] * a[4] + extra[8];
    let c4 = const_2 * a[1] * a[3] + extra[9];

    ([c0, c1, c2, c3, c4], extra)
}

#[derive(Clone, Debug, Default)]
pub struct QuinticSquaringBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: QuinticSquaringGate,
    row: usize,
    const_2: F,
    const_3: F,
    const_6: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for QuinticSquaringBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "QuinticSquaringBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 0),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 1),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 2),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 3),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 4),
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
        let a: [F; 5] = core::array::from_fn(|j| {
            witness.get_target(Target::wire(
                self.row,
                self.gate.wire_ith_multiplicand_jth_limb(self.i, j),
            ))
        });

        let (c, extra) = quintic_square_wires(&a, self.const_2, self.const_3, self.const_6);

        // Set outputs
        for j in 0..5 {
            out_buffer.set_target(
                Target::wire(self.row, self.gate.wire_ith_output_jth_limb(self.i, j)),
                c[j],
            )?;
        }

        // Set extra/intermediate wires
        for j in 0..10 {
            out_buffer.set_target(
                Target::wire(self.row, self.gate.temporary_wire(self.i, j)),
                extra[j],
            )?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_field(self.const_2)?;
        dst.write_field(self.const_3)?;
        dst.write_field(self.const_6)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = QuinticSquaringGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let const_2 = src.read_field()?;
        let const_3 = src.read_field()?;
        let const_6 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            const_2,
            const_3,
            const_6,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::eddsa::gates::square_quintic_ext_base::QuinticSquaringGate;
    use crate::plonky2::field::goldilocks_field::GoldilocksField;
    use crate::plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonky2::plonk::circuit_data::CircuitConfig;
    use crate::plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate =
            QuinticSquaringGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate =
            QuinticSquaringGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn square_generator_matches_reference() {
        use plonky2::field::types::{Field, PrimeField64};

        use super::quintic_square_wires;

        type F = GoldilocksField;

        // The original (pre-optimization) generator arithmetic, reconstructed
        // as the reference oracle.
        fn reference(a: &[F; 5], const_2: F, const_3: F, const_6: F) -> ([F; 5], [F; 10]) {
            let mut extra = [F::ZERO; 10];
            extra[0] = a[0] * a[0];
            extra[1] = const_6 * a[1] * a[4] + extra[0];
            let c0 = const_6 * a[2] * a[3] + extra[1];
            extra[2] = const_3 * a[3] * a[3];
            extra[3] = const_2 * a[0] * a[1] + extra[2];
            let c1 = const_6 * a[2] * a[4] + extra[3];
            extra[4] = a[1] * a[1];
            extra[5] = const_2 * a[0] * a[2] + extra[4];
            let c2 = const_6 * a[3] * a[4] + extra[5];
            extra[6] = const_3 * a[4] * a[4];
            extra[7] = const_2 * a[0] * a[3] + extra[6];
            let c3 = const_2 * a[1] * a[2] + extra[7];
            extra[8] = a[2] * a[2];
            extra[9] = const_2 * a[0] * a[4] + extra[8];
            let c4 = const_2 * a[1] * a[3] + extra[9];
            ([c0, c1, c2, c3, c4], extra)
        }

        let const_2 = F::from_canonical_u64(2);
        let const_3 = F::from_canonical_u64(3);
        let const_6 = F::from_canonical_u64(6);
        let check = |a: [F; 5]| {
            let (c_ref, extra_ref) = reference(&a, const_2, const_3, const_6);
            let (c_new, extra_new) = quintic_square_wires(&a, const_2, const_3, const_6);
            for j in 0..5 {
                assert_eq!(
                    c_new[j].to_canonical_u64(),
                    c_ref[j].to_canonical_u64(),
                    "c[{j}] mismatch for a={a:?}"
                );
            }
            for j in 0..10 {
                assert_eq!(
                    extra_new[j].to_canonical_u64(),
                    extra_ref[j].to_canonical_u64(),
                    "extra[{j}] mismatch for a={a:?}"
                );
            }
        };

        // Edge cases, including non-canonical representations.
        let p = 0xFFFF_FFFF_0000_0001u64;
        let specials = [0, 1, 2, 3, p - 2, p - 1, p, p + 1, u64::MAX];
        for &x in &specials {
            check([GoldilocksField(x); 5]);
            for j in 0..5 {
                let mut a = [GoldilocksField(0); 5];
                a[j] = GoldilocksField(x);
                check(a);
            }
        }

        // Randomized differential over the full u64 (non-canonical included) range.
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..100_000 {
            let a = core::array::from_fn(|_| GoldilocksField(next()));
            check(a);
        }
    }
}
