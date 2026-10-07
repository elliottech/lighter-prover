// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gates/arithmetic_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
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
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate to perform a basic mul-add on 16-bit values
/// ! We assume they are range-checked beforehand.
/// ! Outputs are range-checked
#[derive(Copy, Clone, Debug, Default)]
pub struct U16ArithmeticGate<F: RichField + Extendable<D>, const D: usize> {
    pub num_ops: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> U16ArithmeticGate<F, D> {
    pub fn new_from_config(config: &CircuitConfig) -> Self {
        let wires_per_op = Self::routed_wires_per_op() + Self::num_limbs();
        let num_ops = (config.num_wires / wires_per_op)
            .min(config.num_routed_wires / Self::routed_wires_per_op());

        Self {
            num_ops,
            _phantom: PhantomData,
        }
    }

    pub fn wire_ith_multiplicand_0(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        Self::routed_wires_per_op() * i
    }
    pub fn wire_ith_multiplicand_1(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        Self::routed_wires_per_op() * i + 1
    }
    pub fn wire_ith_addend(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        Self::routed_wires_per_op() * i + 2
    }

    pub fn wire_ith_output_low_half(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        Self::routed_wires_per_op() * i + 3
    }

    pub fn wire_ith_output_high_half(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        Self::routed_wires_per_op() * i + 4
    }

    pub fn limb_bits() -> usize {
        2
    }
    pub fn num_limbs() -> usize {
        16 / Self::limb_bits()
    }
    pub fn routed_wires_per_op() -> usize {
        5
    }
    pub fn wire_ith_output_jth_limb(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.num_ops);
        debug_assert!(j < Self::num_limbs());
        Self::routed_wires_per_op() * self.num_ops + Self::num_limbs() * i + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for U16ArithmeticGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self {
            num_ops,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[self.wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[self.wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[self.wire_ith_addend(i)];

            let computed_output = multiplicand_0 * multiplicand_1 + addend;

            let output_low = vars.local_wires[self.wire_ith_output_low_half(i)];
            let output_high = vars.local_wires[self.wire_ith_output_high_half(i)];

            let base = F::Extension::from_canonical_u64(1 << 16u64);
            let combined_output = output_high * base + output_low;
            // Check if multiplicand_0 * multiplicand_1 + addend = output_high * 2^16 + output_low
            constraints.push(combined_output - computed_output);

            // Check if output_high and output_low is really 16 bits. To do that we are expressing each number as sum of [`Self::limb_bits()`] limbs
            let mut combined_low_limbs = F::Extension::ZERO;
            let mut combined_high_limbs = F::Extension::ZERO;
            let midpoint = Self::num_limbs() / 2;
            let base = F::Extension::from_canonical_u64(1u64 << Self::limb_bits());
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let product = (0..max_limb)
                    .map(|x| this_limb - F::Extension::from_canonical_usize(x))
                    .product();
                constraints.push(product);

                if j < midpoint {
                    combined_low_limbs = base * combined_low_limbs + this_limb;
                } else {
                    combined_high_limbs = base * combined_high_limbs + this_limb;
                }
            }
            constraints.push(combined_low_limbs - output_low);
            constraints.push(combined_high_limbs - output_high);
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
        let three = F::from_canonical_usize(3);
        let base16 = F::from_canonical_u64(1 << 16u64);
        let midpoint = Self::num_limbs() / 2;
        // Each half-recomposition below is accumulated in a 128 bit integer and
        // reduced once, instead of once per limb. Raw representatives are at
        // most `2^64 - 1`, so a half of `midpoint` limbs in base
        // `2^limb_bits` sums to less than
        // `(2^64 - 1) * (limb_base^midpoint - 1) / (limb_base - 1)
        //    < 2^64 * limb_base^midpoint * limb_base / (limb_base - 1)`.
        // The assertion keeps `limb_base^midpoint <= 2^30`, which bounds the
        // accumulator by `2^64 * 2^30 * 4/3 < 2^96`: below `2^96` every
        // sub-`2^64` representative produced by the delayed reduction is
        // bit-identical to the per-limb Horner chain's, not merely congruent
        // to it (the high 32 bits of the 128 bit accumulator are zero, so
        // `from_noncanonical_u128` takes the same wrap-correction branch that
        // the multiply/add chain takes). With the production parameters
        // (`limb_bits = 2`, `num_limbs = 8`) the accumulator is below `2^71`.
        assert!(Self::limb_bits() * midpoint <= 30);
        let limb_base_u128 = 1u128 << Self::limb_bits();
        let mut chunks = combined_gate_constraints.chunks_exact_mut(n);
        // Batches are 32 points in this prover; keep the recomposition rows on
        // the stack and fall back to the heap only for oversized batches.
        let mut combined_low_stack = [0u128; 64];
        let mut combined_high_stack = [0u128; 64];
        let mut combined_low_heap;
        let mut combined_high_heap;
        let (mut combined_low, mut combined_high): (&mut [u128], &mut [u128]) = if n <= 64 {
            (&mut combined_low_stack[..n], &mut combined_high_stack[..n])
        } else {
            combined_low_heap = vec![0u128; n];
            combined_high_heap = vec![0u128; n];
            (&mut combined_low_heap, &mut combined_high_heap)
        };

        for i in 0..self.num_ops {
            let multiplicand_0 = &wires[self.wire_ith_multiplicand_0(i) * n..][..n];
            let multiplicand_1 = &wires[self.wire_ith_multiplicand_1(i) * n..][..n];
            let addend = &wires[self.wire_ith_addend(i) * n..][..n];
            let output_low = &wires[self.wire_ith_output_low_half(i) * n..][..n];
            let output_high = &wires[self.wire_ith_output_high_half(i) * n..][..n];

            let out = chunks.next().unwrap();
            for p in 0..n {
                let computed = multiplicand_0[p] * multiplicand_1[p] + addend[p];
                out[p] += filters[p] * (output_high[p] * base16 + output_low[p] - computed);
            }

            // Limb range products (base-4: x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3))
            // in the same descending order as `eval_unfiltered`, accumulating
            // the low/high recompositions along the way.
            combined_low.fill(0);
            combined_high.fill(0);
            for j in (0..Self::num_limbs()).rev() {
                let limb = &wires[self.wire_ith_output_jth_limb(i, j) * n..][..n];
                let out = chunks.next().unwrap();
                debug_assert_eq!(1 << Self::limb_bits(), 4);
                for p in 0..n {
                    let x = limb[p];
                    let y = x * (x - three);
                    out[p] += filters[p] * (y * (y + F::TWO));
                }
                let combined = if j < midpoint {
                    &mut combined_low
                } else {
                    &mut combined_high
                };
                for p in 0..n {
                    combined[p] =
                        combined[p] * limb_base_u128 + limb[p].to_noncanonical_u64() as u128;
                }
            }
            let out = chunks.next().unwrap();
            for p in 0..n {
                debug_assert!(combined_low[p] < 1u128 << 96);
                out[p] += filters[p] * (F::from_noncanonical_u128(combined_low[p]) - output_low[p]);
            }
            let out = chunks.next().unwrap();
            for p in 0..n {
                debug_assert!(combined_high[p] < 1u128 << 96);
                out[p] +=
                    filters[p] * (F::from_noncanonical_u128(combined_high[p]) - output_high[p]);
            }
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[self.wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[self.wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[self.wire_ith_addend(i)];

            let computed_output = builder.mul_add_extension(multiplicand_0, multiplicand_1, addend);

            let output_low = vars.local_wires[self.wire_ith_output_low_half(i)];
            let output_high = vars.local_wires[self.wire_ith_output_high_half(i)];
            let base: F::Extension = F::from_canonical_u64(1 << 16u64).into();
            let base_target = builder.constant_extension(base);

            // Check canonicity of combined_output = output_high * 2^16 + output_low
            let combined_output = builder.mul_add_extension(output_high, base_target, output_low);
            constraints.push(builder.sub_extension(combined_output, computed_output));

            let mut combined_low_limbs = builder.zero_extension();
            let mut combined_high_limbs = builder.zero_extension();
            let midpoint = Self::num_limbs() / 2;
            let base = builder
                .constant_extension(F::Extension::from_canonical_u64(1u64 << Self::limb_bits()));
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();

                let mut product = builder.one_extension();
                for x in 0..max_limb {
                    let x_target =
                        builder.constant_extension(F::Extension::from_canonical_usize(x));
                    let diff = builder.sub_extension(this_limb, x_target);
                    product = builder.mul_extension(product, diff);
                }
                constraints.push(product);

                if j < midpoint {
                    combined_low_limbs =
                        builder.mul_add_extension(base, combined_low_limbs, this_limb);
                } else {
                    combined_high_limbs =
                        builder.mul_add_extension(base, combined_high_limbs, this_limb);
                }
            }

            constraints.push(builder.sub_extension(combined_low_limbs, output_low));
            constraints.push(builder.sub_extension(combined_high_limbs, output_high));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                let g: WitnessGeneratorRef<F, D> = WitnessGeneratorRef::new(
                    U16ArithmeticGenerator {
                        gate: *self,
                        row,
                        i,
                        _phantom: PhantomData,
                    }
                    .adapter(),
                );
                g
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * (Self::routed_wires_per_op() + Self::num_limbs())
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1 << Self::limb_bits()
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * (4 + Self::num_limbs())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for U16ArithmeticGate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[self.wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[self.wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[self.wire_ith_addend(i)];

            let computed_output = multiplicand_0 * multiplicand_1 + addend;

            let output_low = vars.local_wires[self.wire_ith_output_low_half(i)];
            let output_high = vars.local_wires[self.wire_ith_output_high_half(i)];
            let base = P::from(F::from_canonical_u64(1 << 16u64));

            let combined_output = output_high * base + output_low;
            yield_constr.one(combined_output - computed_output);

            let mut combined_low_limbs = P::ZEROS;
            let mut combined_high_limbs = P::ZEROS;
            let midpoint = Self::num_limbs() / 2;
            let base = F::from_canonical_u64(1u64 << Self::limb_bits());
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let product = (0..max_limb)
                    .map(|x| this_limb - F::from_canonical_usize(x))
                    .product();
                yield_constr.one(product);

                if j < midpoint {
                    combined_low_limbs = combined_low_limbs * base + this_limb;
                } else {
                    combined_high_limbs = combined_high_limbs * base + this_limb;
                }
            }
            yield_constr.one(combined_low_limbs - output_low);
            yield_constr.one(combined_high_limbs - output_high);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct U16ArithmeticGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: U16ArithmeticGate<F, D>,
    row: usize,
    i: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for U16ArithmeticGenerator<F, D>
{
    fn id(&self) -> String {
        "U16ArithmeticGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        vec![
            local_target(self.gate.wire_ith_multiplicand_0(self.i)),
            local_target(self.gate.wire_ith_multiplicand_1(self.i)),
            local_target(self.gate.wire_ith_addend(self.i)),
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

        let multiplicand_0 = get_local_wire(self.gate.wire_ith_multiplicand_0(self.i));
        let multiplicand_1 = get_local_wire(self.gate.wire_ith_multiplicand_1(self.i));
        let addend = get_local_wire(self.gate.wire_ith_addend(self.i));

        let output = multiplicand_0 * multiplicand_1 + addend;
        let mut output_u64 = output.to_canonical_u64();

        let output_high_u64 = output_u64 >> 16;
        let output_low_u64 = output_u64 & ((1 << 16) - 1);

        let output_high = F::from_canonical_u64(output_high_u64);
        let output_low = F::from_canonical_u64(output_low_u64);

        let output_high_wire = local_wire(self.gate.wire_ith_output_high_half(self.i));
        let output_low_wire = local_wire(self.gate.wire_ith_output_low_half(self.i));

        out_buffer.set_wire(output_high_wire, output_high)?;
        out_buffer.set_wire(output_low_wire, output_low)?;

        let num_limbs = U16ArithmeticGate::<F, D>::num_limbs();
        let limb_base = 1 << U16ArithmeticGate::<F, D>::limb_bits();
        let output_limbs_u64 = std::iter::from_fn(move || {
            if output_u64 == 0 {
                None
            } else {
                let ret = output_u64 % limb_base;
                output_u64 /= limb_base;
                Some(ret)
            }
        })
        .take(num_limbs);
        let output_limbs_f = output_limbs_u64.map(F::from_canonical_u64);

        for (j, output_limb) in output_limbs_f.enumerate() {
            let wire = local_wire(self.gate.wire_ith_output_jth_limb(self.i, j));
            out_buffer.set_wire(wire, output_limb)?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = U16ArithmeticGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            i,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field64, PrimeField64};
    use plonky2::hash::hash_types::HashOut;

    use super::*;

    /// The pre-transformation accumulation path, kept compiled as the
    /// differential's reference: one modular reduction per limb inside each
    /// half recomposition.
    fn eval_accumulate_termwise_reference<F: RichField + Extendable<D>, const D: usize>(
        gate: &U16ArithmeticGate<F, D>,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        let wires = vars_base.local_wires;
        let three = F::from_canonical_usize(3);
        let limb_base = F::from_canonical_u64(1u64 << U16ArithmeticGate::<F, D>::limb_bits());
        let base16 = F::from_canonical_u64(1 << 16u64);
        let midpoint = U16ArithmeticGate::<F, D>::num_limbs() / 2;
        let mut chunks = combined_gate_constraints.chunks_exact_mut(n);
        let mut combined_low = vec![F::ZERO; n];
        let mut combined_high = vec![F::ZERO; n];

        for i in 0..gate.num_ops {
            let multiplicand_0 = &wires[gate.wire_ith_multiplicand_0(i) * n..][..n];
            let multiplicand_1 = &wires[gate.wire_ith_multiplicand_1(i) * n..][..n];
            let addend = &wires[gate.wire_ith_addend(i) * n..][..n];
            let output_low = &wires[gate.wire_ith_output_low_half(i) * n..][..n];
            let output_high = &wires[gate.wire_ith_output_high_half(i) * n..][..n];

            let out = chunks.next().unwrap();
            for p in 0..n {
                let computed = multiplicand_0[p] * multiplicand_1[p] + addend[p];
                out[p] += filters[p] * (output_high[p] * base16 + output_low[p] - computed);
            }

            combined_low.fill(F::ZERO);
            combined_high.fill(F::ZERO);
            for j in (0..U16ArithmeticGate::<F, D>::num_limbs()).rev() {
                let limb = &wires[gate.wire_ith_output_jth_limb(i, j) * n..][..n];
                let out = chunks.next().unwrap();
                for p in 0..n {
                    let x = limb[p];
                    let y = x * (x - three);
                    out[p] += filters[p] * (y * (y + F::TWO));
                }
                let combined = if j < midpoint {
                    &mut combined_low
                } else {
                    &mut combined_high
                };
                for p in 0..n {
                    combined[p] = combined[p] * limb_base + limb[p];
                }
            }
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] += filters[p] * (combined_low[p] - output_low[p]);
            }
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] += filters[p] * (combined_high[p] - output_high[p]);
            }
        }
    }

    /// Raw-limb differential between the delayed and the per-limb reduction at
    /// the production gate shape, over arbitrary `u64` representatives:
    /// canonical, non-canonical, and boundary. Limb wires are deliberately NOT
    /// restricted to the base-4 digit set — quotient-domain evaluations are
    /// not, so the bound is derived from `raw < 2^64` alone.
    #[test]
    fn delayed_limb_recomposition_matches_termwise_reduction_in_raw_limbs() {
        type F = GoldilocksField;
        const D: usize = 2;
        // CIRCUIT_CONFIG: num_wires 136, num_routed_wires 80 => num_ops 10.
        // The quotient walker evaluates 32 points per batch.
        const NUM_OPS: usize = 10;
        const N: usize = 32;

        let boundary: [u64; 15] = [
            0,
            1,
            2,
            3,
            (u32::MAX as u64) - 1,
            u32::MAX as u64,
            1u64 << 32,
            F::ORDER - 2,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            F::ORDER + 2,
            1u64 << 63,
            u64::MAX - 1,
            u64::MAX,
        ];

        let gate = U16ArithmeticGate::<F, D> {
            num_ops: NUM_OPS,
            _phantom: PhantomData,
        };
        let num_wires = <U16ArithmeticGate<F, D> as Gate<F, D>>::num_wires(&gate);
        let num_constraints = <U16ArithmeticGate<F, D> as Gate<F, D>>::num_constraints(&gate);
        let hash = HashOut::<F>::ZERO;

        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for trial in 0..64usize {
            // Trial 0 is a pure boundary sweep; every later trial mixes
            // boundary and uniform raw representatives.
            let mut draw = |k: usize| -> F {
                if trial == 0 {
                    GoldilocksField(boundary[k % boundary.len()])
                } else {
                    let r = next();
                    if r % 3 == 0 {
                        GoldilocksField(boundary[(r >> 2) as usize % boundary.len()])
                    } else {
                        GoldilocksField(r)
                    }
                }
            };

            let wires: Vec<F> = (0..num_wires * N).map(&mut draw).collect();
            let filters: Vec<F> = (0..N).map(&mut draw).collect();
            let seed: Vec<F> = (0..num_constraints * N).map(&mut draw).collect();

            let vars = EvaluationVarsBaseBatch::new(N, &[], &wires, &hash);

            let mut expected = seed.clone();
            eval_accumulate_termwise_reference(&gate, vars, &filters, &mut expected);

            let mut actual = seed;
            gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);

            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_noncanonical_u64(),
                    e.to_noncanonical_u64(),
                    "raw-limb mismatch at slot {index} (trial {trial})"
                );
            }
        }
    }
}
