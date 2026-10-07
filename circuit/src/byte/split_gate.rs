// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::ops::Range;

use anyhow::Result;
use plonky2::field::batch_util::batch_multiply_add_inplace;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
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
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::types::config::{D, F};

/// A gate which can decompose a number into bytes
#[derive(Copy, Clone, Debug)]
pub struct ByteDecompositionGate {
    pub num_limbs: usize,
    pub num_ops: usize,
}

impl ByteDecompositionGate {
    pub(crate) const fn new(num_limbs: usize, num_ops: usize) -> Self {
        debug_assert!(num_limbs > 0);
        debug_assert!(num_ops > 0);
        Self { num_limbs, num_ops }
    }

    pub fn new_from_config(config: &CircuitConfig, num_limbs: usize) -> Self {
        let num_ops =
            (config.num_routed_wires / (1 + num_limbs)).min(config.num_wires / (1 + num_limbs * 5));
        debug_assert!(
            num_ops > 0,
            "Not enough wires to support {} limbs",
            num_limbs
        );

        let mut gate = Self { num_limbs, num_ops };

        while <ByteDecompositionGate as Gate<F, D>>::num_constraints(&gate) > 123
            && gate.num_ops > 1
        {
            // We need to reduce the number of constraints
            // by reducing the number of ops
            gate.num_ops -= 1;
        }

        gate
    }

    pub const fn i_th_sum(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        i * (1 + self.num_limbs)
    }

    /// Returns the range for the limbs
    pub const fn i_th_limbs(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_ops);
        let start = 1 + i * (1 + self.num_limbs);
        start..start + self.num_limbs
    }

    pub const fn i_th_aux_limbs(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_ops);
        let start = (1 + self.num_limbs) * self.num_ops + i * (4 * self.num_limbs);
        start..start + 4 * self.num_limbs
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ByteDecompositionGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_limbs)?;
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_limbs = src.read_usize()?;
        let num_ops = src.read_usize()?;
        Ok(Self { num_limbs, num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let limbs = vars.local_wires[self.i_th_aux_limbs(i)].to_vec();
            // Range check aux limbs
            limbs.iter().for_each(|&limb| {
                constraints.push(
                    (0..4)
                        .map(|i| limb - F::Extension::from_canonical_usize(i))
                        .product(),
                );
            });

            let bytes = vars.local_wires[self.i_th_limbs(i)].to_vec();

            // Constaint each limb
            limbs.chunks(4).enumerate().for_each(|(index, chunk)| {
                let sum = reduce_with_powers(chunk, F::Extension::from_canonical_usize(4));
                constraints.push(sum - bytes[index]);
            });

            // Constaint the sum
            let expected_sum = vars.local_wires[self.i_th_sum(i)];
            let sum = reduce_with_powers(&bytes, F::Extension::from_canonical_usize(256));
            constraints.push(sum - expected_sum);
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
        let n = vars_base.len();
        let wires = vars_base.local_wires;
        let three = F::from_canonical_usize(3);
        let four = F::from_canonical_usize(4);
        let base = F::from_canonical_usize(256);
        let mut res = vec![F::ZERO; n * <Self as Gate<F, D>>::num_constraints(self)];
        let mut chunks = res.chunks_exact_mut(n);

        for i in 0..self.num_ops {
            let aux = self.i_th_aux_limbs(i);
            // Range products per aux limb: x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3).
            for limb_wire in aux.clone() {
                let col = &wires[limb_wire * n..][..n];
                let out = chunks.next().unwrap();
                for p in 0..n {
                    let x = col[p];
                    let y = x * (x - three);
                    out[p] = y * (y + F::TWO);
                }
            }

            // Each byte equals its four aux limbs combined by powers of 4,
            // accumulated per point by Horner from the most significant limb.
            let bytes = self.i_th_limbs(i);
            for (byte_index, byte_wire) in bytes.clone().enumerate() {
                let chunk_start = aux.start + 4 * byte_index;
                let out = chunks.next().unwrap();
                out.copy_from_slice(&wires[(chunk_start + 3) * n..][..n]);
                for k in (0..3).rev() {
                    let limb = &wires[(chunk_start + k) * n..][..n];
                    for p in 0..n {
                        out[p] = out[p] * four + limb[p];
                    }
                }
                let byte_col = &wires[byte_wire * n..][..n];
                for p in 0..n {
                    out[p] -= byte_col[p];
                }
            }

            // The sum equals the bytes combined by powers of 256.
            let out = chunks.next().unwrap();
            out.copy_from_slice(&wires[(bytes.end - 1) * n..][..n]);
            for byte_wire in (bytes.start..bytes.end - 1).rev() {
                let col = &wires[byte_wire * n..][..n];
                for p in 0..n {
                    out[p] = out[p] * base + col[p];
                }
            }
            let sum_col = &wires[self.i_th_sum(i) * n..][..n];
            for p in 0..n {
                out[p] -= sum_col[p];
            }
        }
        res
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        let num_constraints = <Self as Gate<F, D>>::num_constraints(self);
        assert!(combined_gate_constraints.len() >= num_constraints * n);

        let wires = vars_base.local_wires;
        let three = F::from_canonical_usize(3);
        let four = F::from_canonical_usize(4);
        let base = F::from_canonical_usize(256);
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

        for i in 0..self.num_ops {
            let aux = self.i_th_aux_limbs(i);
            // Range products per aux limb: x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3).
            for limb_wire in aux.clone() {
                let col = &wires[limb_wire * n..][..n];
                for p in 0..n {
                    let x = col[p];
                    let y = x * (x - three);
                    scratch[p] = y * (y + F::TWO);
                }
                let combined = &mut combined_gate_constraints
                    [constraint_index * n..(constraint_index + 1) * n];
                batch_multiply_add_inplace(combined, scratch, filters);
                constraint_index += 1;
            }

            // Each byte equals its four aux limbs combined by powers of 4,
            // accumulated per point by Horner from the most significant limb.
            let bytes = self.i_th_limbs(i);
            for (byte_index, byte_wire) in bytes.clone().enumerate() {
                let chunk_start = aux.start + 4 * byte_index;
                scratch.copy_from_slice(&wires[(chunk_start + 3) * n..][..n]);
                for k in (0..3).rev() {
                    let limb = &wires[(chunk_start + k) * n..][..n];
                    for p in 0..n {
                        scratch[p] = scratch[p] * four + limb[p];
                    }
                }
                let byte_col = &wires[byte_wire * n..][..n];
                for p in 0..n {
                    scratch[p] -= byte_col[p];
                }
                let combined = &mut combined_gate_constraints
                    [constraint_index * n..(constraint_index + 1) * n];
                batch_multiply_add_inplace(combined, scratch, filters);
                constraint_index += 1;
            }

            // The sum equals the bytes combined by powers of 256.
            scratch.copy_from_slice(&wires[(bytes.end - 1) * n..][..n]);
            for byte_wire in (bytes.start..bytes.end - 1).rev() {
                let col = &wires[byte_wire * n..][..n];
                for p in 0..n {
                    scratch[p] = scratch[p] * base + col[p];
                }
            }
            let sum_col = &wires[self.i_th_sum(i) * n..][..n];
            for p in 0..n {
                scratch[p] -= sum_col[p];
            }
            let combined =
                &mut combined_gate_constraints[constraint_index * n..(constraint_index + 1) * n];
            batch_multiply_add_inplace(combined, scratch, filters);
            constraint_index += 1;
        }

        debug_assert_eq!(constraint_index, num_constraints);
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let _4 = builder.constant(F::from_canonical_usize(4));
        let _256 = builder.constant(F::from_canonical_usize(256));

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let limbs = vars.local_wires[self.i_th_aux_limbs(i)].to_vec();
            // Range check aux limbs
            limbs.iter().for_each(|&limb| {
                constraints.push({
                    let mut acc = builder.one_extension();
                    (0..4).for_each(|i| {
                        // We update our accumulator as:
                        // acc' = acc (x - i)
                        //      = acc x + (-i) acc
                        // Since -i is constant, we can do this in one arithmetic_extension call.
                        let neg_i = -F::from_canonical_usize(i);
                        acc = builder.arithmetic_extension(F::ONE, neg_i, acc, limb, acc)
                    });
                    acc
                })
            });

            let bytes = vars.local_wires[self.i_th_limbs(i)].to_vec();

            // Constaint each limb
            limbs.chunks(4).enumerate().for_each(|(index, chunk)| {
                let sum = reduce_with_powers_ext_circuit(builder, chunk, _4);
                constraints.push(builder.sub_extension(sum, bytes[index]));
            });

            // Constaint the sum
            let expected_sum = vars.local_wires[self.i_th_sum(i)];
            let sum = reduce_with_powers_ext_circuit(builder, &bytes, _256);
            constraints.push(builder.sub_extension(sum, expected_sum));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    ByteDecompositionGenerator {
                        row,
                        num_limbs: self.num_limbs,
                        num_ops: self.num_ops,
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    // 1 for the sum then `num_limbs` for the limbs.
    fn num_wires(&self) -> usize {
        (1 + self.num_limbs * 5) * self.num_ops
    }

    fn num_constants(&self) -> usize {
        0
    }

    // Bounded by the range-check
    fn degree(&self) -> usize {
        4
    }

    // 1 for checking the sum then `num_limbs` for range-checking the limbs.
    fn num_constraints(&self) -> usize {
        (1 + self.num_limbs * 5) * self.num_ops
    }

    fn u32_quotient_gate(&self) -> Option<U32QuotientGate> {
        Some(U32QuotientGate::ByteDecomposition {
            num_ops: self.num_ops,
            num_limbs: self.num_limbs,
        })
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for ByteDecompositionGate
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for i in 0..self.num_ops {
            let limbs = vars.local_wires.view(self.i_th_aux_limbs(i));
            // Range check aux limbs
            let constraints_iter = limbs.iter().map(|&limb| {
                (0..4)
                    .map(|i| limb - F::from_canonical_usize(i))
                    .product::<P>()
            });
            yield_constr.many(constraints_iter);

            let bytes = vars.local_wires.view(self.i_th_limbs(i));

            // Constaint each limb
            // Constaint each limb
            for j in 0..self.num_limbs {
                let chunk = limbs.view(j * 4..(j + 1) * 4);
                let sum = reduce_with_powers(chunk, F::from_canonical_usize(4));
                yield_constr.one(sum - bytes[j]);
            }
            // Constaint the sum
            let expected_sum = vars.local_wires[self.i_th_sum(i)];
            let sum = reduce_with_powers(bytes, F::from_canonical_usize(256));
            yield_constr.one(sum - expected_sum);
        }
    }
}

#[derive(Debug, Default)]
pub struct ByteDecompositionGenerator {
    row: usize,
    num_limbs: usize,
    num_ops: usize,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for ByteDecompositionGenerator
{
    fn id(&self) -> String {
        "ByteDecompositionGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(
            self.row,
            ByteDecompositionGate::new(self.num_limbs, self.num_ops).i_th_sum(self.i),
        )]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let dummy_gate = ByteDecompositionGate::new(self.num_limbs, self.num_ops);
        let sum_value = witness
            .get_target(Target::wire(self.row, dummy_gate.i_th_sum(self.i)))
            .to_canonical_u64();

        // Set bytes
        // Direct limb-decomposition loops: same limbs in the same order as the
        // previous `scan`/`collect` into temporary `Vec`s, minus the heap
        // allocations per generator execution. `i_th_limbs`/`i_th_aux_limbs`
        // are ranges of exactly `num_limbs`/`4 * num_limbs` columns, so the
        // pairing is exhaustive exactly as `zip_eq` required.
        let mut acc = sum_value;
        for i in dummy_gate.i_th_limbs(self.i) {
            let tmp = acc % 256_u64;
            acc /= 256_u64;
            out_buffer.set_target(Target::wire(self.row, i), F::from_canonical_u64(tmp))?;
        }

        // Set aux limbs
        let mut acc = sum_value;
        for i in dummy_gate.i_th_aux_limbs(self.i) {
            let tmp = acc % 4_u64;
            acc /= 4_u64;
            out_buffer.set_target(Target::wire(self.row, i), F::from_canonical_u64(tmp))?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.num_limbs)?;
        dst.write_usize(self.num_ops)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let num_limbs = src.read_usize()?;
        let num_ops = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            num_limbs,
            num_ops,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(ByteDecompositionGate::new(1, 1))
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(ByteDecompositionGate::new(1, 1))
    }

    // `test_eval_fns` only checks a batch of one point; compare the batched
    // path against per-point `eval_unfiltered` across a multi-point batch.
    #[test]
    fn base_batch_matches_eval_unfiltered_across_batch() {
        use plonky2::field::extension::FieldExtension;
        use plonky2::field::types::Field64;
        use plonky2::hash::hash_types::HashOut;
        use plonky2::plonk::vars::{EvaluationVars, EvaluationVarsBaseBatch};
        use rand::Rng;

        const D: usize = 2;
        type F = GoldilocksField;

        let mut rng = rand::thread_rng();
        for (num_limbs, num_ops) in [(1, 1), (4, 2), (8, 1)] {
            let gate = ByteDecompositionGate::new(num_limbs, num_ops);
            let n = 32;
            let num_wires = <ByteDecompositionGate as Gate<F, D>>::num_wires(&gate);
            let num_constraints = <ByteDecompositionGate as Gate<F, D>>::num_constraints(&gate);
            let wires_batch: Vec<F> = (0..num_wires * n)
                .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
                .collect();
            let public_inputs_hash = HashOut::<F>::ZERO;
            let vars_batch =
                EvaluationVarsBaseBatch::new(n, &[], &wires_batch, &public_inputs_hash);
            let batch_out = <ByteDecompositionGate as Gate<F, D>>::eval_unfiltered_base_batch(
                &gate, vars_batch,
            );
            assert_eq!(batch_out.len(), n * num_constraints);

            for p in 0..n {
                let wires_one: Vec<<F as Extendable<D>>::Extension> = (0..num_wires)
                    .map(|w| {
                        <<F as Extendable<D>>::Extension as FieldExtension<D>>::from_basefield(
                            wires_batch[w * n + p],
                        )
                    })
                    .collect();
                let vars_one = EvaluationVars::<F, D> {
                    local_constants: &[],
                    local_wires: &wires_one,
                    public_inputs_hash: &public_inputs_hash,
                };
                let expected = gate.eval_unfiltered(vars_one);
                for (j, expected_j) in expected.iter().enumerate() {
                    assert_eq!(
                        <<F as Extendable<D>>::Extension as FieldExtension<D>>::from_basefield(
                            batch_out[j * n + p]
                        ),
                        *expected_j,
                        "num_limbs {num_limbs}, num_ops {num_ops}, point {p}, constraint {j}"
                    );
                }
            }
        }
    }

    // The direct filtered accumulation override must produce bit-identical
    // values to materializing the batch then multiply-adding row by row.
    #[test]
    fn direct_filtered_accumulation_matches_materialized_batch() {
        use crate::gate_batch_testing::assert_direct_accumulation_matches_materialized_batch;

        for (num_limbs, num_ops) in [(1, 1), (4, 2), (8, 1)] {
            let gate = ByteDecompositionGate::new(num_limbs, num_ops);
            assert_direct_accumulation_matches_materialized_batch(&gate);
        }
    }
}
