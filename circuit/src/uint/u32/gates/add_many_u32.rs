// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gates/add_many_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::gates::gate::Gate;
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
};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::utils::ceil_div_usize;

const LOG2_MAX_NUM_ADDENDS: usize = 4;
pub const MAX_NUM_ADDENDS: usize = 16;

/// A gate to perform addition on `num_addends` different 32-bit values, plus a small carry
#[derive(Copy, Clone, Debug, Default)]
pub struct U32AddManyGate<F: RichField + Extendable<D>, const D: usize> {
    pub num_addends: usize,
    pub num_ops: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> U32AddManyGate<F, D> {
    pub fn new_from_config(config: &CircuitConfig, num_addends: usize) -> Self {
        debug_assert!(
            num_addends <= MAX_NUM_ADDENDS,
            "num_addends ({num_addends}) must be less than or equal to MAX_NUM_ADDENDS ({MAX_NUM_ADDENDS})"
        );
        let wires_per_op = (num_addends + 3) + Self::num_limbs();
        let routed_wires_per_op = num_addends + 3;
        let num_ops =
            (config.num_wires / wires_per_op).min(config.num_routed_wires / routed_wires_per_op);

        Self {
            num_addends,
            num_ops,
            _phantom: PhantomData,
        }
    }

    pub fn wire_ith_op_jth_addend(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.num_ops);
        debug_assert!(j < self.num_addends);
        (self.num_addends + 3) * i + j
    }
    pub fn wire_ith_carry(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        (self.num_addends + 3) * i + self.num_addends
    }

    pub fn wire_ith_output_result(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        (self.num_addends + 3) * i + self.num_addends + 1
    }
    pub fn wire_ith_output_carry(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        (self.num_addends + 3) * i + self.num_addends + 2
    }

    pub fn limb_bits() -> usize {
        2
    }
    pub fn num_result_limbs() -> usize {
        ceil_div_usize(32, Self::limb_bits())
    }
    pub fn num_carry_limbs() -> usize {
        ceil_div_usize(LOG2_MAX_NUM_ADDENDS, Self::limb_bits())
    }
    pub fn num_limbs() -> usize {
        Self::num_result_limbs() + Self::num_carry_limbs()
    }

    pub fn wire_ith_output_jth_limb(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.num_ops);
        debug_assert!(j < Self::num_limbs());
        (self.num_addends + 3) * self.num_ops + Self::num_limbs() * i + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for U32AddManyGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_addends)?;
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_addends = src.read_usize()?;
        let num_ops = src.read_usize()?;
        Ok(Self {
            num_addends,
            num_ops,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());
        for i in 0..self.num_ops {
            let addends: Vec<F::Extension> = (0..self.num_addends)
                .map(|j| vars.local_wires[self.wire_ith_op_jth_addend(i, j)])
                .collect();
            let carry = vars.local_wires[self.wire_ith_carry(i)];

            let computed_output = addends.iter().fold(F::Extension::ZERO, |x, &y| x + y) + carry;

            let output_result = vars.local_wires[self.wire_ith_output_result(i)];
            let output_carry = vars.local_wires[self.wire_ith_output_carry(i)];

            let base = F::Extension::from_canonical_u64(1 << 32u64);
            let combined_output = output_carry * base + output_result;

            constraints.push(combined_output - computed_output);

            let mut combined_result_limbs = F::Extension::ZERO;
            let mut combined_carry_limbs = F::Extension::ZERO;
            let base = F::Extension::from_canonical_u64(1u64 << Self::limb_bits());
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let product = (0..max_limb)
                    .map(|x| this_limb - F::Extension::from_canonical_usize(x))
                    .product();
                constraints.push(product);

                if j < Self::num_result_limbs() {
                    combined_result_limbs = base * combined_result_limbs + this_limb;
                } else {
                    combined_carry_limbs = base * combined_carry_limbs + this_limb;
                }
            }
            constraints.push(combined_result_limbs - output_result);
            constraints.push(combined_carry_limbs - output_carry);
        }

        constraints
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        assert_eq!(1 << Self::limb_bits(), 4);
        let n = vars_base.len();
        let wires = vars_base.local_wires;
        let three = F::from_canonical_usize(3);
        let base_limb = F::from_canonical_u64(1u64 << Self::limb_bits());
        let base32 = F::from_canonical_u64(1 << 32u64);
        let mut res = vec![F::ZERO; n * <Self as Gate<F, D>>::num_constraints(self)];
        let mut chunks = res.chunks_exact_mut(n);
        let mut combined_result = vec![F::ZERO; n];
        let mut combined_carry = vec![F::ZERO; n];

        for i in 0..self.num_ops {
            let output_result = &wires[self.wire_ith_output_result(i) * n..][..n];
            let output_carry = &wires[self.wire_ith_output_carry(i) * n..][..n];

            // output_carry * 2^32 + output_result - (sum of addends + carry).
            let out = chunks.next().unwrap();
            out.copy_from_slice(&wires[self.wire_ith_carry(i) * n..][..n]);
            for j in 0..self.num_addends {
                let addend = &wires[self.wire_ith_op_jth_addend(i, j) * n..][..n];
                for p in 0..n {
                    out[p] += addend[p];
                }
            }
            for p in 0..n {
                out[p] = output_carry[p] * base32 + output_result[p] - out[p];
            }

            // Limb range products (base-4: x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3))
            // in the same descending order as `eval_unfiltered`, accumulating
            // the result/carry recompositions along the way.
            combined_result.fill(F::ZERO);
            combined_carry.fill(F::ZERO);
            for j in (0..Self::num_limbs()).rev() {
                let limb = &wires[self.wire_ith_output_jth_limb(i, j) * n..][..n];
                let out = chunks.next().unwrap();
                for p in 0..n {
                    let x = limb[p];
                    let y = x * (x - three);
                    out[p] = y * (y + F::TWO);
                }
                let combined = if j < Self::num_result_limbs() {
                    &mut combined_result
                } else {
                    &mut combined_carry
                };
                for p in 0..n {
                    combined[p] = combined[p] * base_limb + limb[p];
                }
            }

            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = combined_result[p] - output_result[p];
            }
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = combined_carry[p] - output_carry[p];
            }
        }
        assert!(chunks.next().is_none());
        res
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        for i in 0..self.num_ops {
            let addends: Vec<F> = (0..self.num_addends)
                .map(|j| vars.local_wires[self.wire_ith_op_jth_addend(i, j)])
                .collect();
            let carry = vars.local_wires[self.wire_ith_carry(i)];

            let computed_output = addends.iter().fold(F::ZERO, |x, &y| x + y) + carry;

            let output_result = vars.local_wires[self.wire_ith_output_result(i)];
            let output_carry = vars.local_wires[self.wire_ith_output_carry(i)];

            let base = F::from_canonical_u64(1 << 32u64);
            let combined_output = output_carry * base + output_result;

            yield_constr.one(combined_output - computed_output);

            let mut combined_result_limbs = F::ZERO;
            let mut combined_carry_limbs = F::ZERO;
            let base = F::from_canonical_u64(1u64 << Self::limb_bits());
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let product = (0..max_limb)
                    .map(|x| this_limb - F::from_canonical_usize(x))
                    .product();
                yield_constr.one(product);

                if j < Self::num_result_limbs() {
                    combined_result_limbs = base * combined_result_limbs + this_limb;
                } else {
                    combined_carry_limbs = base * combined_carry_limbs + this_limb;
                }
            }
            yield_constr.one(combined_result_limbs - output_result);
            yield_constr.one(combined_carry_limbs - output_carry);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        for i in 0..self.num_ops {
            let addends: Vec<ExtensionTarget<D>> = (0..self.num_addends)
                .map(|j| vars.local_wires[self.wire_ith_op_jth_addend(i, j)])
                .collect();
            let carry = vars.local_wires[self.wire_ith_carry(i)];

            let mut computed_output = carry;
            for addend in addends {
                computed_output = builder.add_extension(computed_output, addend);
            }

            let output_result = vars.local_wires[self.wire_ith_output_result(i)];
            let output_carry = vars.local_wires[self.wire_ith_output_carry(i)];

            let base: F::Extension = F::from_canonical_u64(1 << 32u64).into();
            let base_target = builder.constant_extension(base);
            let combined_output =
                builder.mul_add_extension(output_carry, base_target, output_result);

            constraints.push(builder.sub_extension(combined_output, computed_output));

            let mut combined_result_limbs = builder.zero_extension();
            let mut combined_carry_limbs = builder.zero_extension();
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

                if j < Self::num_result_limbs() {
                    combined_result_limbs =
                        builder.mul_add_extension(base, combined_result_limbs, this_limb);
                } else {
                    combined_carry_limbs =
                        builder.mul_add_extension(base, combined_carry_limbs, this_limb);
                }
            }
            constraints.push(builder.sub_extension(combined_result_limbs, output_result));
            constraints.push(builder.sub_extension(combined_carry_limbs, output_carry));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    U32AddManyGenerator {
                        gate: *self,
                        row,
                        i,
                        _phantom: PhantomData,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        (self.num_addends + 3) * self.num_ops + Self::num_limbs() * self.num_ops
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1 << Self::limb_bits()
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * (3 + Self::num_limbs())
    }
}

#[derive(Clone, Debug, Default)]
pub struct U32AddManyGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: U32AddManyGate<F, D>,
    row: usize,
    i: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for U32AddManyGenerator<F, D>
{
    fn id(&self) -> String {
        "U32AddManyGenerator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = U32AddManyGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            i,
            _phantom: PhantomData,
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        (0..self.gate.num_addends)
            .map(|j| local_target(self.gate.wire_ith_op_jth_addend(self.i, j)))
            .chain([local_target(self.gate.wire_ith_carry(self.i))])
            .collect()
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

        let addends: Vec<_> = (0..self.gate.num_addends)
            .map(|j| get_local_wire(self.gate.wire_ith_op_jth_addend(self.i, j)))
            .collect();
        let carry = get_local_wire(self.gate.wire_ith_carry(self.i));

        let output = addends.iter().fold(F::ZERO, |x, &y| x + y) + carry;
        let output_u64 = output.to_canonical_u64();

        let output_carry_u64 = output_u64 >> 32;
        let output_result_u64 = output_u64 & ((1 << 32) - 1);

        let output_carry = F::from_canonical_u64(output_carry_u64);
        let output_result = F::from_canonical_u64(output_result_u64);

        let output_carry_wire = local_wire(self.gate.wire_ith_output_carry(self.i));
        let output_result_wire = local_wire(self.gate.wire_ith_output_result(self.i));

        out_buffer.set_wire(output_carry_wire, output_carry)?;
        out_buffer.set_wire(output_result_wire, output_result)?;

        let num_result_limbs = U32AddManyGate::<F, D>::num_result_limbs();
        let num_carry_limbs = U32AddManyGate::<F, D>::num_carry_limbs();
        let limb_base = 1 << U32AddManyGate::<F, D>::limb_bits();

        let split_to_limbs = |mut val, num| {
            std::iter::from_fn(move || {
                if num == 0 {
                    None
                } else {
                    let ret = val % limb_base;
                    val /= limb_base;
                    Some(F::from_canonical_u64(ret))
                }
            })
            .take(num)
            .collect::<Vec<_>>()
        };

        let result_limbs = split_to_limbs(output_result_u64, num_result_limbs);
        let carry_limbs = split_to_limbs(output_carry_u64, num_carry_limbs);

        for (j, limb) in result_limbs.into_iter().chain(carry_limbs).enumerate() {
            let wire = local_wire(self.gate.wire_ith_output_jth_limb(self.i, j));
            out_buffer.set_wire(wire, limb)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod batch_tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::*;
    use crate::gate_batch_testing::assert_base_batch_matches_eval_unfiltered;

    #[test]
    fn base_batch_matches_eval_unfiltered_across_batch() {
        for num_addends in [2, 5, 16] {
            let gate = U32AddManyGate::<GoldilocksField, 2>::new_from_config(
                &CircuitConfig::standard_recursion_config(),
                num_addends,
            );
            assert_base_batch_matches_eval_unfiltered(&gate);
        }
    }
}
