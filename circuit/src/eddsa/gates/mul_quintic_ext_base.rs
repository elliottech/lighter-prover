// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::any::Any;

use anyhow::Result;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
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
pub struct QuinticMultiplicationGate {
    /// Number of Quintic Multiplications performed by a Gate
    pub num_ops: usize,
}

impl QuinticMultiplicationGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }
    //Number of routed wires necessary for an operation
    const ROUTED_PER_OP: usize = 15;
    const TOTAL_PER_OP: usize = Self::ROUTED_PER_OP;
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

    pub(crate) const fn wire_ith_multiplicand_jth_limb_0(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + j
    }
    pub(crate) const fn wire_ith_multiplicand_jth_limb_1(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + 5 + j
    }
    pub(crate) const fn wire_ith_output_jth_limb(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + 10 + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for QuinticMultiplicationGate {
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
        let const_3 = F::Extension::from_basefield(F::from_canonical_u64(3));
        let mut constraints = Vec::with_capacity(self.num_ops * 25);

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb_0(i, j)])
                .collect::<Vec<_>>();
            let b = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb_1(i, j)])
                .collect::<Vec<_>>();
            let c = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();

            let mut d = [F::Extension::ZEROS; 9];
            for j in 0..5 {
                for k in 0..5 {
                    d[j + k] += a[j] * b[k];
                }
            }

            // Reduction u^5 = 3
            for k in 0..5 {
                let term = if k + 5 <= 8 {
                    d[k] + const_3 * d[k + 5]
                } else {
                    d[k]
                };
                constraints.push(term - c[k]);
            }
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
        let const_3 = F::from_canonical_u64(3);
        let mut chunks = combined_gate_constraints.chunks_exact_mut(n);

        for i in 0..self.num_ops {
            let a_cols: [&[F]; 5] =
                core::array::from_fn(|j| col(self.wire_ith_multiplicand_jth_limb_0(i, j)));
            let b_cols: [&[F]; 5] =
                core::array::from_fn(|j| col(self.wire_ith_multiplicand_jth_limb_1(i, j)));
            let c_cols: [&[F]; 5] =
                core::array::from_fn(|j| col(self.wire_ith_output_jth_limb(i, j)));
            let outs: [&mut [F]; 5] = core::array::from_fn(|_| chunks.next().unwrap());

            for p in 0..n {
                let a: [F; 5] = core::array::from_fn(|j| a_cols[j][p]);
                let b: [F; 5] = core::array::from_fn(|j| b_cols[j][p]);
                let limbs = quintic_mul_limbs(&a, &b, const_3);
                for k in 0..5 {
                    outs[k][p] += filters[p] * (limbs[k] - c_cols[k][p]);
                }
            }
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_1 = F::from_canonical_u64(1);
        let const_3 = F::from_canonical_u64(3);
        let mut constraints = Vec::with_capacity(self.num_ops * 25); // 24 intermediate constraints

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb_0(i, j)])
                .collect::<Vec<_>>();
            let b = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb_1(i, j)])
                .collect::<Vec<_>>();
            let out = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();

            let [a0, a1, a2, a3, a4] = <[ExtensionTarget<D>; 5]>::try_from(a).unwrap();
            let [b0, b1, b2, b3, b4] = <[ExtensionTarget<D>; 5]>::try_from(b).unwrap();
            let [c0, c1, c2, c3, c4] = <[ExtensionTarget<D>; 5]>::try_from(out).unwrap();

            // --- c0
            let t0 = builder.mul_extension(a4, b1);
            let t1 = builder.mul_add_extension(a3, b2, t0);
            let t2 = builder.mul_add_extension(a2, b3, t1);
            let t3 = builder.mul_add_extension(a1, b4, t2);
            let t4 = builder.arithmetic_extension(const_1, const_3, a0, b0, t3);
            constraints.push(builder.sub_extension(t4, c0));

            // --- c1
            let t5 = builder.mul_extension(a4, b2);
            let t6 = builder.mul_add_extension(a3, b3, t5);
            let t7 = builder.mul_add_extension(a2, b4, t6);
            let t8 = builder.arithmetic_extension(const_1, const_3, a1, b0, t7);
            let t9 = builder.mul_add_extension(a0, b1, t8);
            constraints.push(builder.sub_extension(t9, c1));

            // --- c2
            let t10 = builder.mul_extension(a4, b3);
            let t11 = builder.mul_add_extension(a3, b4, t10);
            let t12 = builder.arithmetic_extension(const_1, const_3, a2, b0, t11);
            let t13 = builder.mul_add_extension(a1, b1, t12);
            let t14 = builder.mul_add_extension(a0, b2, t13);
            constraints.push(builder.sub_extension(t14, c2));

            // --- c3
            let t15 = builder.mul_extension(a4, b4);
            let t16 = builder.arithmetic_extension(const_1, const_3, a3, b0, t15);
            let t17 = builder.mul_add_extension(a2, b1, t16);
            let t18 = builder.mul_add_extension(a1, b2, t17);
            let t19 = builder.mul_add_extension(a0, b3, t18);
            constraints.push(builder.sub_extension(t19, c3));

            // --- c4
            let t20 = builder.mul_extension(a4, b0);
            let t21 = builder.mul_add_extension(a3, b1, t20);
            let t22 = builder.mul_add_extension(a2, b2, t21);
            let t23 = builder.mul_add_extension(a1, b3, t22);
            let t24 = builder.mul_add_extension(a0, b4, t23);
            constraints.push(builder.sub_extension(t24, c4));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    QuinticMultiplicationBaseGenerator {
                        gate: self.clone(),
                        row,
                        const_3: F::from_canonical_u64(3),
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
        self.num_ops * 5
    }

    fn u32_quotient_gate(&self) -> Option<U32QuotientGate> {
        Some(U32QuotientGate::QuinticMultiplication {
            num_ops: self.num_ops,
        })
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for QuinticMultiplicationGate
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_3 = P::from(F::from_canonical_u64(3));

        for i in 0..self.num_ops {
            let a: [P; 5] = core::array::from_fn(|j| {
                vars.local_wires[self.wire_ith_multiplicand_jth_limb_0(i, j)]
            });
            let b: [P; 5] = core::array::from_fn(|j| {
                vars.local_wires[self.wire_ith_multiplicand_jth_limb_1(i, j)]
            });
            let c: [P; 5] =
                core::array::from_fn(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)]);

            let mut d = [P::ZEROS; 9];
            for j in 0..5 {
                for k in 0..5 {
                    d[j + k] += a[j] * b[k];
                }
            }

            // Reduction u^5 = 3
            for k in 0..5 {
                let term = if k + 5 < 9 {
                    d[k] + const_3 * d[k + 5]
                } else {
                    d[k]
                };
                yield_constr.one(term - c[k]);
            }
        }
    }
}

/// Computes the limbs of `a * b` in `F[u]/(u^5 - 3)`.
///
/// For `GoldilocksField` (with the canonical `const_3 == 3`) this dispatches to the
/// delayed-reduction `QuinticExtension` multiplication, which performs one modular
/// reduction per output limb; otherwise it falls back to the original fully-reduced
/// schoolbook multiplication. Both paths compute the same field elements.
fn quintic_mul_limbs<F: RichField>(a: &[F; 5], b: &[F; 5], const_3: F) -> [F; 5] {
    if const_3 == F::from_canonical_u64(3)
        && let (Some(ga), Some(gb)) = (
            (a as &dyn Any).downcast_ref::<[GoldilocksField; 5]>(),
            (b as &dyn Any).downcast_ref::<[GoldilocksField; 5]>(),
        )
    {
        let c = (QuinticExtension(*ga) * QuinticExtension(*gb)).0;
        return *(&c as &dyn Any).downcast_ref::<[F; 5]>().unwrap();
    }

    let mut d = [F::ZERO; 9];
    for j in 0..5 {
        for k in 0..5 {
            d[j + k] += a[j] * b[k];
        }
    }

    // Reduction by u^5 = 3:
    [
        d[0] + const_3 * d[5],
        d[1] + const_3 * d[6],
        d[2] + const_3 * d[7],
        d[3] + const_3 * d[8],
        d[4],
    ]
}

#[derive(Clone, Debug, Default)]
pub struct QuinticMultiplicationBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: QuinticMultiplicationGate,
    row: usize,
    const_3: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for QuinticMultiplicationBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "QuinticMultiplicationBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            self.gate.wire_ith_multiplicand_jth_limb_0(self.i, 0),
            self.gate.wire_ith_multiplicand_jth_limb_0(self.i, 1),
            self.gate.wire_ith_multiplicand_jth_limb_0(self.i, 2),
            self.gate.wire_ith_multiplicand_jth_limb_0(self.i, 3),
            self.gate.wire_ith_multiplicand_jth_limb_0(self.i, 4),
            self.gate.wire_ith_multiplicand_jth_limb_1(self.i, 0),
            self.gate.wire_ith_multiplicand_jth_limb_1(self.i, 1),
            self.gate.wire_ith_multiplicand_jth_limb_1(self.i, 2),
            self.gate.wire_ith_multiplicand_jth_limb_1(self.i, 3),
            self.gate.wire_ith_multiplicand_jth_limb_1(self.i, 4),
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
                self.gate.wire_ith_multiplicand_jth_limb_0(self.i, j),
            ))
        });

        let b: [F; 5] = core::array::from_fn(|j| {
            witness.get_target(Target::wire(
                self.row,
                self.gate.wire_ith_multiplicand_jth_limb_1(self.i, j),
            ))
        });

        let c = quintic_mul_limbs(&a, &b, self.const_3);

        for j in 0..5 {
            out_buffer.set_target(
                Target::wire(self.row, self.gate.wire_ith_output_jth_limb(self.i, j)),
                c[j],
            )?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_field(self.const_3)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = QuinticMultiplicationGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let const_3 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            const_3,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::eddsa::gates::mul_quintic_ext_base::QuinticMultiplicationGate;
    use crate::plonky2::field::goldilocks_field::GoldilocksField;
    use crate::plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonky2::plonk::circuit_data::CircuitConfig;
    use crate::plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate =
            QuinticMultiplicationGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate =
            QuinticMultiplicationGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn mul_generator_matches_reference() {
        use plonky2::field::types::{Field, PrimeField64};

        use super::quintic_mul_limbs;

        type F = GoldilocksField;

        // The original (pre-optimization) generator arithmetic, reconstructed
        // as the reference oracle.
        fn reference(a: &[F; 5], b: &[F; 5], const_3: F) -> [F; 5] {
            let mut d = [F::ZERO; 9];
            for j in 0..5 {
                for k in 0..5 {
                    d[j + k] += a[j] * b[k];
                }
            }
            [
                d[0] + const_3 * d[5],
                d[1] + const_3 * d[6],
                d[2] + const_3 * d[7],
                d[3] + const_3 * d[8],
                d[4],
            ]
        }

        let const_3 = F::from_canonical_u64(3);
        let check = |a: [F; 5], b: [F; 5]| {
            let expected = reference(&a, &b, const_3);
            let actual = quintic_mul_limbs(&a, &b, const_3);
            for j in 0..5 {
                assert_eq!(
                    actual[j].to_canonical_u64(),
                    expected[j].to_canonical_u64(),
                    "limb {j} mismatch for a={a:?} b={b:?}"
                );
            }
        };

        // Edge cases, including non-canonical representations.
        let p = 0xFFFF_FFFF_0000_0001u64;
        let specials = [0, 1, 2, 3, p - 2, p - 1, p, p + 1, u64::MAX];
        for &x in &specials {
            for &y in &specials {
                check([GoldilocksField(x); 5], [GoldilocksField(y); 5]);
            }
        }

        // Randomized differential over the full u64 (non-canonical included) range.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..100_000 {
            let a = core::array::from_fn(|_| GoldilocksField(next()));
            let b = core::array::from_fn(|_| GoldilocksField(next()));
            check(a, b);
        }
    }
}
