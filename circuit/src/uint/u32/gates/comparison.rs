// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gates/comparison.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::{Field, Field64};
use plonky2::gates::gate::Gate;
use plonky2::gates::packed_util::PackedEvaluableBase;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::wire::Wire;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use plonky2::util::bits_u64;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::utils::ceil_div_usize;

/// A gate for checking that one value is less than or equal to another.
#[derive(Clone, Debug, Default)]
pub struct ComparisonGate<F: Field64 + Extendable<D>, const D: usize> {
    pub(crate) num_bits: usize,
    pub(crate) num_chunks: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> ComparisonGate<F, D> {
    pub fn new(num_bits: usize, num_chunks: usize) -> Self {
        debug_assert!(num_bits < bits_u64(F::ORDER));
        Self {
            num_bits,
            num_chunks,
            _phantom: PhantomData,
        }
    }

    pub fn chunk_bits(&self) -> usize {
        ceil_div_usize(self.num_bits, self.num_chunks)
    }

    pub fn wire_first_input(&self) -> usize {
        0
    }

    pub fn wire_second_input(&self) -> usize {
        1
    }

    pub fn wire_result_bool(&self) -> usize {
        2
    }

    pub fn wire_most_significant_diff(&self) -> usize {
        3
    }

    pub fn wire_first_chunk_val(&self, chunk: usize) -> usize {
        debug_assert!(chunk < self.num_chunks);
        4 + chunk
    }

    pub fn wire_second_chunk_val(&self, chunk: usize) -> usize {
        debug_assert!(chunk < self.num_chunks);
        4 + self.num_chunks + chunk
    }

    pub fn wire_equality_dummy(&self, chunk: usize) -> usize {
        debug_assert!(chunk < self.num_chunks);
        4 + 2 * self.num_chunks + chunk
    }

    pub fn wire_chunks_equal(&self, chunk: usize) -> usize {
        debug_assert!(chunk < self.num_chunks);
        4 + 3 * self.num_chunks + chunk
    }

    pub fn wire_intermediate_value(&self, chunk: usize) -> usize {
        debug_assert!(chunk < self.num_chunks);
        4 + 4 * self.num_chunks + chunk
    }

    /// The `bit_index`th bit of 2^n - 1 + most_significant_diff.
    pub fn wire_most_significant_diff_bit(&self, bit_index: usize) -> usize {
        4 + 5 * self.num_chunks + bit_index
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ComparisonGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_bits)?;
        dst.write_usize(self.num_chunks)?;
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_bits = src.read_usize()?;
        let num_chunks = src.read_usize()?;
        Ok(Self {
            num_bits,
            num_chunks,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        let first_input = vars.local_wires[self.wire_first_input()];
        let second_input = vars.local_wires[self.wire_second_input()];

        // Get chunks and assert that they match
        let first_chunks: Vec<F::Extension> = (0..self.num_chunks)
            .map(|i| vars.local_wires[self.wire_first_chunk_val(i)])
            .collect();
        let second_chunks: Vec<F::Extension> = (0..self.num_chunks)
            .map(|i| vars.local_wires[self.wire_second_chunk_val(i)])
            .collect();

        let first_chunks_combined = reduce_with_powers(
            &first_chunks,
            F::Extension::from_canonical_usize(1 << self.chunk_bits()),
        );
        let second_chunks_combined = reduce_with_powers(
            &second_chunks,
            F::Extension::from_canonical_usize(1 << self.chunk_bits()),
        );

        constraints.push(first_chunks_combined - first_input);
        constraints.push(second_chunks_combined - second_input);

        let chunk_size = 1 << self.chunk_bits();

        let mut most_significant_diff_so_far = F::Extension::ZERO;

        for i in 0..self.num_chunks {
            // Range-check the chunks to be less than `chunk_size`.
            let first_product: F::Extension = (0..chunk_size)
                .map(|x| first_chunks[i] - F::Extension::from_canonical_usize(x))
                .product();
            let second_product: F::Extension = (0..chunk_size)
                .map(|x| second_chunks[i] - F::Extension::from_canonical_usize(x))
                .product();
            constraints.push(first_product);
            constraints.push(second_product);

            let difference = second_chunks[i] - first_chunks[i];
            let equality_dummy = vars.local_wires[self.wire_equality_dummy(i)];
            let chunks_equal = vars.local_wires[self.wire_chunks_equal(i)];

            // Two constraints to assert that `chunks_equal` is valid.
            constraints.push(difference * equality_dummy - (F::Extension::ONE - chunks_equal));
            constraints.push(chunks_equal * difference);

            // Update `most_significant_diff_so_far`.
            let intermediate_value = vars.local_wires[self.wire_intermediate_value(i)];
            constraints.push(intermediate_value - chunks_equal * most_significant_diff_so_far);
            most_significant_diff_so_far =
                intermediate_value + (F::Extension::ONE - chunks_equal) * difference;
        }

        let most_significant_diff = vars.local_wires[self.wire_most_significant_diff()];
        constraints.push(most_significant_diff - most_significant_diff_so_far);

        let most_significant_diff_bits: Vec<F::Extension> = (0..self.chunk_bits() + 1)
            .map(|i| vars.local_wires[self.wire_most_significant_diff_bit(i)])
            .collect();

        // Range-check the bits.
        for &bit in &most_significant_diff_bits {
            constraints.push(bit * (F::Extension::ONE - bit));
        }

        let bits_combined = reduce_with_powers(&most_significant_diff_bits, F::Extension::TWO);
        let two_n = F::Extension::from_canonical_u64(1 << self.chunk_bits());
        constraints.push((two_n + most_significant_diff) - bits_combined);

        // Iff first <= second, the top (n + 1st) bit of (2^n + most_significant_diff) will be 1.
        let result_bool = vars.local_wires[self.wire_result_bool()];
        constraints.push(result_bool - most_significant_diff_bits[self.chunk_bits()]);

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
        let chunk_bits = self.chunk_bits();
        let chunk_base = F::from_canonical_usize(1 << chunk_bits);
        let chunk_size = 1usize << chunk_bits;
        let three = F::from_canonical_usize(3);
        let mut chunks_iter = combined_gate_constraints.chunks_exact_mut(n);
        // Batches are 32 points in this prover; keep the scratch rows on the
        // stack and fall back to the heap only for oversized batches.
        let mut scratch_stack = [F::ZERO; 64];
        let mut msd_stack = [F::ZERO; 64];
        let mut scratch_heap;
        let mut msd_heap;
        let (scratch, most_significant_diff_so_far): (&mut [F], &mut [F]) = if n <= 64 {
            (&mut scratch_stack[..n], &mut msd_stack[..n])
        } else {
            scratch_heap = vec![F::ZERO; n];
            msd_heap = vec![F::ZERO; n];
            (&mut scratch_heap, &mut msd_heap)
        };

        // combined chunks - input, for both inputs, accumulated per point by
        // Horner over the chunk columns from most to least significant. The
        // chunk-wire index is computed inline rather than collected, in the
        // same most-to-least-significant order as before.
        for second_input in [false, true] {
            let input_wire = if second_input {
                self.wire_second_input()
            } else {
                self.wire_first_input()
            };
            let chunk_wire = |i: usize| {
                if second_input {
                    self.wire_second_chunk_val(i)
                } else {
                    self.wire_first_chunk_val(i)
                }
            };
            let out = chunks_iter.next().unwrap();
            scratch.copy_from_slice(col(chunk_wire(self.num_chunks - 1)));
            for i in (0..self.num_chunks - 1).rev() {
                let chunk = col(chunk_wire(i));
                for p in 0..n {
                    scratch[p] = scratch[p] * chunk_base + chunk[p];
                }
            }
            let input = col(input_wire);
            for p in 0..n {
                out[p] += filters[p] * (scratch[p] - input[p]);
            }
        }

        for i in 0..self.num_chunks {
            let first = col(self.wire_first_chunk_val(i));
            let second = col(self.wire_second_chunk_val(i));
            for value in [first, second] {
                let out = chunks_iter.next().unwrap();
                match chunk_size {
                    4 => {
                        for p in 0..n {
                            let x = value[p];
                            let y = x * (x - three);
                            out[p] += filters[p] * (y * (y + F::TWO));
                        }
                    }
                    2 => {
                        for p in 0..n {
                            let x = value[p];
                            out[p] += filters[p] * (x * (x - F::ONE));
                        }
                    }
                    _ => {
                        for p in 0..n {
                            let x = value[p];
                            let mut product = x;
                            for k in 1..chunk_size {
                                product *= x - F::from_canonical_usize(k);
                            }
                            out[p] += filters[p] * product;
                        }
                    }
                }
            }

            let equality_dummy = col(self.wire_equality_dummy(i));
            let chunks_equal = col(self.wire_chunks_equal(i));
            let intermediate_value = col(self.wire_intermediate_value(i));

            let out = chunks_iter.next().unwrap();
            for p in 0..n {
                let difference = second[p] - first[p];
                out[p] +=
                    filters[p] * (difference * equality_dummy[p] - (F::ONE - chunks_equal[p]));
            }
            let out = chunks_iter.next().unwrap();
            for p in 0..n {
                out[p] += filters[p] * (chunks_equal[p] * (second[p] - first[p]));
            }
            let out = chunks_iter.next().unwrap();
            for p in 0..n {
                out[p] += filters[p]
                    * (intermediate_value[p] - chunks_equal[p] * most_significant_diff_so_far[p]);
                most_significant_diff_so_far[p] =
                    intermediate_value[p] + (F::ONE - chunks_equal[p]) * (second[p] - first[p]);
            }
        }

        let most_significant_diff = col(self.wire_most_significant_diff());
        let out = chunks_iter.next().unwrap();
        for p in 0..n {
            out[p] += filters[p] * (most_significant_diff[p] - most_significant_diff_so_far[p]);
        }

        for i in 0..chunk_bits + 1 {
            let bit = col(self.wire_most_significant_diff_bit(i));
            let out = chunks_iter.next().unwrap();
            for p in 0..n {
                out[p] += filters[p] * (bit[p] * (F::ONE - bit[p]));
            }
        }

        // (2^n + most_significant_diff) - bits_combined, Horner over bits.
        let two_n = F::from_canonical_u64(1 << chunk_bits);
        let out = chunks_iter.next().unwrap();
        scratch.copy_from_slice(col(self.wire_most_significant_diff_bit(chunk_bits)));
        for i in (0..chunk_bits).rev() {
            let bit = col(self.wire_most_significant_diff_bit(i));
            for p in 0..n {
                scratch[p] = scratch[p].double() + bit[p];
            }
        }
        for p in 0..n {
            out[p] += filters[p] * ((two_n + most_significant_diff[p]) - scratch[p]);
        }

        let result_bool = col(self.wire_result_bool());
        let top_bit = col(self.wire_most_significant_diff_bit(chunk_bits));
        let out = chunks_iter.next().unwrap();
        for p in 0..n {
            out[p] += filters[p] * (result_bool[p] - top_bit[p]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        let first_input = vars.local_wires[self.wire_first_input()];
        let second_input = vars.local_wires[self.wire_second_input()];

        // Get chunks and assert that they match
        let first_chunks: Vec<ExtensionTarget<D>> = (0..self.num_chunks)
            .map(|i| vars.local_wires[self.wire_first_chunk_val(i)])
            .collect();
        let second_chunks: Vec<ExtensionTarget<D>> = (0..self.num_chunks)
            .map(|i| vars.local_wires[self.wire_second_chunk_val(i)])
            .collect();

        let chunk_base = builder.constant(F::from_canonical_usize(1 << self.chunk_bits()));
        let first_chunks_combined =
            reduce_with_powers_ext_circuit(builder, &first_chunks, chunk_base);
        let second_chunks_combined =
            reduce_with_powers_ext_circuit(builder, &second_chunks, chunk_base);

        constraints.push(builder.sub_extension(first_chunks_combined, first_input));
        constraints.push(builder.sub_extension(second_chunks_combined, second_input));

        let chunk_size = 1 << self.chunk_bits();

        let mut most_significant_diff_so_far = builder.zero_extension();

        let one = builder.one_extension();
        // Find the chosen chunk.
        for i in 0..self.num_chunks {
            // Range-check the chunks to be less than `chunk_size`.
            let mut first_product = one;
            let mut second_product = one;
            for x in 0..chunk_size {
                let x_f = builder.constant_extension(F::Extension::from_canonical_usize(x));
                let first_diff = builder.sub_extension(first_chunks[i], x_f);
                let second_diff = builder.sub_extension(second_chunks[i], x_f);
                first_product = builder.mul_extension(first_product, first_diff);
                second_product = builder.mul_extension(second_product, second_diff);
            }
            constraints.push(first_product);
            constraints.push(second_product);

            let difference = builder.sub_extension(second_chunks[i], first_chunks[i]);
            let equality_dummy = vars.local_wires[self.wire_equality_dummy(i)];
            let chunks_equal = vars.local_wires[self.wire_chunks_equal(i)];

            // Two constraints to assert that `chunks_equal` is valid.
            let diff_times_equal = builder.mul_extension(difference, equality_dummy);
            let not_equal = builder.sub_extension(one, chunks_equal);
            constraints.push(builder.sub_extension(diff_times_equal, not_equal));
            constraints.push(builder.mul_extension(chunks_equal, difference));

            // Update `most_significant_diff_so_far`.
            let intermediate_value = vars.local_wires[self.wire_intermediate_value(i)];
            let old_diff = builder.mul_extension(chunks_equal, most_significant_diff_so_far);
            constraints.push(builder.sub_extension(intermediate_value, old_diff));

            let not_equal = builder.sub_extension(one, chunks_equal);
            let new_diff = builder.mul_extension(not_equal, difference);
            most_significant_diff_so_far = builder.add_extension(intermediate_value, new_diff);
        }

        let most_significant_diff = vars.local_wires[self.wire_most_significant_diff()];
        constraints
            .push(builder.sub_extension(most_significant_diff, most_significant_diff_so_far));

        let most_significant_diff_bits: Vec<ExtensionTarget<D>> = (0..self.chunk_bits() + 1)
            .map(|i| vars.local_wires[self.wire_most_significant_diff_bit(i)])
            .collect();

        // Range-check the bits.
        for &this_bit in &most_significant_diff_bits {
            let inverse = builder.sub_extension(one, this_bit);
            constraints.push(builder.mul_extension(this_bit, inverse));
        }

        let two = builder.two();
        let bits_combined =
            reduce_with_powers_ext_circuit(builder, &most_significant_diff_bits, two);
        let two_n =
            builder.constant_extension(F::Extension::from_canonical_u64(1 << self.chunk_bits()));
        let sum = builder.add_extension(two_n, most_significant_diff);
        constraints.push(builder.sub_extension(sum, bits_combined));

        // Iff first <= second, the top (n + 1st) bit of (2^n + most_significant_diff) will be 1.
        let result_bool = vars.local_wires[self.wire_result_bool()];
        constraints.push(
            builder.sub_extension(result_bool, most_significant_diff_bits[self.chunk_bits()]),
        );

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let cmp_gen = ComparisonGenerator::<F, D> {
            row,
            gate: self.clone(),
        };
        vec![WitnessGeneratorRef::new(cmp_gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        4 + 5 * self.num_chunks + (self.chunk_bits() + 1)
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1 << self.chunk_bits()
    }

    fn num_constraints(&self) -> usize {
        6 + 5 * self.num_chunks + self.chunk_bits()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for ComparisonGate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let first_input = vars.local_wires[self.wire_first_input()];
        let second_input = vars.local_wires[self.wire_second_input()];

        // Get chunks and assert that they match
        let first_chunks: Vec<_> = (0..self.num_chunks)
            .map(|i| vars.local_wires[self.wire_first_chunk_val(i)])
            .collect();
        let second_chunks: Vec<_> = (0..self.num_chunks)
            .map(|i| vars.local_wires[self.wire_second_chunk_val(i)])
            .collect();

        let first_chunks_combined = reduce_with_powers(
            &first_chunks,
            F::from_canonical_usize(1 << self.chunk_bits()),
        );
        let second_chunks_combined = reduce_with_powers(
            &second_chunks,
            F::from_canonical_usize(1 << self.chunk_bits()),
        );

        yield_constr.one(first_chunks_combined - first_input);
        yield_constr.one(second_chunks_combined - second_input);

        let chunk_size = 1 << self.chunk_bits();

        let mut most_significant_diff_so_far = P::ZEROS;

        for i in 0..self.num_chunks {
            // Range-check the chunks to be less than `chunk_size`.
            let first_product: P = (0..chunk_size)
                .map(|x| first_chunks[i] - F::from_canonical_usize(x))
                .product();
            let second_product: P = (0..chunk_size)
                .map(|x| second_chunks[i] - F::from_canonical_usize(x))
                .product();
            yield_constr.one(first_product);
            yield_constr.one(second_product);

            let difference = second_chunks[i] - first_chunks[i];
            let equality_dummy = vars.local_wires[self.wire_equality_dummy(i)];
            let chunks_equal = vars.local_wires[self.wire_chunks_equal(i)];

            // Two constraints to assert that `chunks_equal` is valid.
            yield_constr.one(difference * equality_dummy - (P::ONES - chunks_equal));
            yield_constr.one(chunks_equal * difference);

            // Update `most_significant_diff_so_far`.
            let intermediate_value = vars.local_wires[self.wire_intermediate_value(i)];
            yield_constr.one(intermediate_value - chunks_equal * most_significant_diff_so_far);
            most_significant_diff_so_far =
                intermediate_value + (P::ONES - chunks_equal) * difference;
        }

        let most_significant_diff = vars.local_wires[self.wire_most_significant_diff()];
        yield_constr.one(most_significant_diff - most_significant_diff_so_far);

        let most_significant_diff_bits: Vec<_> = (0..self.chunk_bits() + 1)
            .map(|i| vars.local_wires[self.wire_most_significant_diff_bit(i)])
            .collect();

        // Range-check the bits.
        for &bit in &most_significant_diff_bits {
            yield_constr.one(bit * (P::ONES - bit));
        }

        let bits_combined = reduce_with_powers(&most_significant_diff_bits, F::TWO);
        let two_n = F::from_canonical_u64(1 << self.chunk_bits());
        yield_constr.one((most_significant_diff + two_n) - bits_combined);

        // Iff first <= second, the top (n + 1st) bit of (2^n - 1 + most_significant_diff) will be 1.
        let result_bool = vars.local_wires[self.wire_result_bool()];
        yield_constr.one(result_bool - most_significant_diff_bits[self.chunk_bits()]);
    }
}

#[derive(Debug, Default)]
pub struct ComparisonGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    gate: ComparisonGate<F, D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for ComparisonGenerator<F, D>
{
    fn id(&self) -> String {
        "ComparisonGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        vec![
            local_target(self.gate.wire_first_input()),
            local_target(self.gate.wire_second_input()),
        ]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let get_local_wire = |column| witness.get_wire(local_wire(column));

        let first_input = get_local_wire(self.gate.wire_first_input());
        let second_input = get_local_wire(self.gate.wire_second_input());

        let first_input_u64 = first_input.to_canonical_u64();
        let second_input_u64 = second_input.to_canonical_u64();

        let result = F::from_canonical_usize((first_input_u64 <= second_input_u64) as usize);

        // Single allocation-free pass: chunks, equality wires, and the
        // running most-significant-diff are computed on the fly with the same
        // arithmetic as before, so every wire value is bit-identical.
        let chunk_size = 1u64 << self.gate.chunk_bits();
        let mut first_acc = first_input_u64;
        let mut second_acc = second_input_u64;
        // Batch the equality-dummy inversions: one field inversion for all
        // unequal chunks via Montgomery's trick instead of one ~72-mul
        // Fermat chain per unequal chunk. Field division is exact, so every
        // dummy value is bit-identical to `F::ONE / diff`.
        const MAX_CHUNKS: usize = 64;
        assert!(self.gate.num_chunks <= MAX_CHUNKS);
        let mut diffs = [F::ZERO; MAX_CHUNKS];
        let mut prefix = [F::ZERO; MAX_CHUNKS];
        let mut unequal_count = 0usize;
        {
            let mut first_scan = first_acc;
            let mut second_scan = second_acc;
            let mut running = F::ONE;
            for _ in 0..self.gate.num_chunks {
                let first_chunk = F::from_canonical_u64(first_scan % chunk_size);
                let second_chunk = F::from_canonical_u64(second_scan % chunk_size);
                first_scan /= chunk_size;
                second_scan /= chunk_size;
                if first_chunk != second_chunk {
                    let diff = second_chunk - first_chunk;
                    diffs[unequal_count] = diff;
                    prefix[unequal_count] = running;
                    running *= diff;
                    unequal_count += 1;
                }
            }
            if unequal_count > 0 {
                let mut inv_running = running.inverse();
                for k in (0..unequal_count).rev() {
                    // inverse of diffs[k] = inv(prod of all) * prod of others.
                    let inv_k = inv_running * prefix[k];
                    inv_running *= diffs[k];
                    diffs[k] = inv_k;
                }
            }
        }
        let mut next_unequal = 0usize;

        let mut most_significant_diff_so_far = F::ZERO;
        for i in 0..self.gate.num_chunks {
            let first_chunk = F::from_canonical_u64(first_acc % chunk_size);
            let second_chunk = F::from_canonical_u64(second_acc % chunk_size);
            first_acc /= chunk_size;
            second_acc /= chunk_size;

            let equal = first_chunk == second_chunk;
            let equality_dummy = if equal {
                F::ONE
            } else {
                let inv = diffs[next_unequal];
                next_unequal += 1;
                inv
            };
            let intermediate_value = if equal {
                most_significant_diff_so_far
            } else {
                F::ZERO
            };
            if !equal {
                most_significant_diff_so_far = second_chunk - first_chunk;
            }

            out_buffer.set_wire(local_wire(self.gate.wire_first_chunk_val(i)), first_chunk)?;
            out_buffer.set_wire(local_wire(self.gate.wire_second_chunk_val(i)), second_chunk)?;
            out_buffer.set_wire(local_wire(self.gate.wire_equality_dummy(i)), equality_dummy)?;
            out_buffer.set_wire(
                local_wire(self.gate.wire_chunks_equal(i)),
                F::from_bool(equal),
            )?;
            out_buffer.set_wire(
                local_wire(self.gate.wire_intermediate_value(i)),
                intermediate_value,
            )?;
        }
        let most_significant_diff = most_significant_diff_so_far;

        out_buffer.set_wire(local_wire(self.gate.wire_result_bool()), result)?;
        out_buffer.set_wire(
            local_wire(self.gate.wire_most_significant_diff()),
            most_significant_diff,
        )?;

        let two_n = F::from_canonical_usize(1 << self.gate.chunk_bits());
        let mut msd_acc = (two_n + most_significant_diff).to_canonical_u64();
        for i in 0..self.gate.chunk_bits() + 1 {
            out_buffer.set_wire(
                local_wire(self.gate.wire_most_significant_diff_bit(i)),
                F::from_canonical_u64(msd_acc % 2),
            )?;
            msd_acc /= 2;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        self.gate.serialize(dst, common_data)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = ComparisonGate::deserialize(src, common_data)?;
        Ok(Self { row, gate })
    }
}
