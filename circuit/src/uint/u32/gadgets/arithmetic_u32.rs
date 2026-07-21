// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gadgets/arithmetic_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use anyhow::Result;
use itertools::Itertools;
use num::{BigUint, FromPrimitive};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::builder::Builder;
use crate::builder::types::{U32AddManyOperation, U32ArithmeticOperation, U32SubtractionOperation};
use crate::uint::u32::gates::add_many_u32::{MAX_NUM_ADDENDS, U32AddManyGate};
use crate::uint::u32::gates::arithmetic_u32::U32ArithmeticGate;
use crate::uint::u32::gates::subtraction_u32::U32SubtractionGate;
use crate::uint::u32::serialization::{ReadU32, WriteU32};
use crate::uint::u32::witness::GeneratedValuesU32;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Default)]
pub struct U32Target(pub Target);

pub trait CircuitBuilderU32<F: RichField + Extendable<D>, const D: usize> {
    #[must_use]
    fn add_virtual_u32_target_unsafe(&mut self) -> U32Target;
    #[must_use]
    fn add_virtual_u32_targets_unsafe(&mut self, n: usize) -> Vec<U32Target>;

    #[must_use]
    fn add_virtual_u32_target_safe(&mut self) -> U32Target;
    #[must_use]
    fn add_virtual_u32_targets_safe(&mut self, n: usize) -> Vec<U32Target>;

    fn split_u32_to_4_bit_limbs_le(&mut self, val: U32Target) -> [Target; 8];

    /// Returns a U32Target for the value `c`, which is assumed to be at most 32 bits.
    fn constant_u32(&mut self, c: u32) -> U32Target;

    fn zero_u32(&mut self) -> U32Target;

    #[must_use]
    fn select_u32(&mut self, cond: BoolTarget, a: U32Target, b: U32Target) -> U32Target;
    #[must_use]
    fn select_arr_u32<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[U32Target; N],
        b: &[U32Target; N],
    ) -> [U32Target; N];

    fn one_u32(&mut self) -> U32Target;

    fn connect_u32(&mut self, x: U32Target, y: U32Target);

    #[must_use]
    fn is_equal_u32(&mut self, a: U32Target, b: U32Target) -> BoolTarget;
    fn conditional_assert_eq_u32(&mut self, cond: BoolTarget, a: U32Target, b: U32Target);

    fn assert_zero_u32(&mut self, x: U32Target);
    fn conditional_assert_zero_u32(&mut self, cond: BoolTarget, a: U32Target);

    #[must_use]
    fn is_zero_u32(&mut self, x: U32Target) -> BoolTarget;

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `U32ArithmeticGate`.
    fn arithmetic_u32_special_cases(
        &mut self,
        x: U32Target,
        y: U32Target,
        z: U32Target,
    ) -> Option<(U32Target, U32Target)>;

    // Returns x * y + z.
    fn mul_add_u32(&mut self, x: U32Target, y: U32Target, z: U32Target) -> (U32Target, U32Target);

    fn add_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target);

    fn add_u32_lo(&mut self, a: U32Target, b: U32Target) -> U32Target;

    fn add_many_u32(&mut self, to_add: &[U32Target]) -> (U32Target, U32Target);

    fn add_u32s_with_carry(
        &mut self,
        to_add: &[U32Target],
        carry: U32Target,
    ) -> (U32Target, U32Target);

    fn mul_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target);

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_u32(&mut self, x: U32Target, y: U32Target, borrow: U32Target) -> (U32Target, U32Target);

    fn split_u64_to_u32s_le(&mut self, val: Target) -> [U32Target; 2];
    fn split_u64_to_u32s_le_unsafe(&mut self, x: Target) -> [U32Target; 2];
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderU32<F, D> for Builder<F, D> {
    fn add_virtual_u32_target_unsafe(&mut self) -> U32Target {
        U32Target(self.add_virtual_target())
    }

    fn add_virtual_u32_targets_unsafe(&mut self, n: usize) -> Vec<U32Target> {
        self.add_virtual_targets(n)
            .into_iter()
            .map(U32Target)
            .collect()
    }

    fn add_virtual_u32_target_safe(&mut self) -> U32Target {
        let target = U32Target(self.add_virtual_target());
        self.register_range_check(target.0, 32);
        target
    }

    fn add_virtual_u32_targets_safe(&mut self, n: usize) -> Vec<U32Target> {
        let targets: Vec<U32Target> = self
            .add_virtual_targets(n)
            .into_iter()
            .map(U32Target)
            .collect();
        for target in &targets {
            self.register_range_check(target.0, 32);
        }
        targets
    }

    fn select_u32(&mut self, cond: BoolTarget, a: U32Target, b: U32Target) -> U32Target {
        U32Target(self.select(cond, a.0, b.0))
    }

    fn select_arr_u32<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[U32Target; N],
        b: &[U32Target; N],
    ) -> [U32Target; N] {
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| self.select_u32(cond, *a, *b))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn split_u64_to_u32s_le_unsafe(&mut self, x: Target) -> [U32Target; 2] {
        let low = self.add_virtual_u32_target_unsafe();
        let high = self.add_virtual_u32_target_unsafe();
        self.add_simple_generator(SplitToU32Generator {
            x,
            low,
            high,
            _phantom: PhantomData,
        });

        let lhs_32_bits_multiplier = self.constant(F::from_canonical_u64(1 << 32));
        let ls_limb_plus_ms_limb_times_32_bits =
            self.mul_add(high.0, lhs_32_bits_multiplier, low.0);
        self.connect(ls_limb_plus_ms_limb_times_32_bits, x);

        [low, high]
    }

    fn split_u64_to_u32s_le(&mut self, x: Target) -> [U32Target; 2] {
        if let Some(const_x) = self.builder.target_as_constant(x) {
            let mut big_x =
                self.constant_biguint(&BigUint::from_u64(const_x.to_canonical_u64()).unwrap());
            assert!(big_x.num_limbs() <= 2);
            big_x.limbs.resize_with(2, || self.zero_u32());

            return [big_x.limbs[0], big_x.limbs[1]];
        }

        if let Some(&result) = self.u32_split_cache.get(&x) {
            return result.into();
        }

        let [low, high] = self.split_u64_to_u32s_le_unsafe(x);

        self.register_range_check(low.0, 32);
        self.register_range_check(high.0, 32);

        // If high is 2^32 - 1, then low must be 0. Because 2^32(2^32 - 1) + 1 = 2^64 - 2^32 + 1 which is prime itself
        // If high is <= 2^32 - 2, then 2^32(2^32 - 2) + 2^32 - 1 = 2^64 - 2^32 - 1 which is smaller than the prime.
        let max = self.constant_u64((1u64 << 32) - 1);
        let is_high_max = self.is_equal(high.0, max);
        self.conditional_assert_zero_u32(is_high_max, low);

        self.u32_split_cache.insert(x, (low, high));
        [low, high]
    }

    fn split_u32_to_4_bit_limbs_le(&mut self, x: U32Target) -> [Target; 8] {
        let two_bit_limbs = self.split_le_base::<4>(x.0, 16);
        let four = self.constant(F::from_canonical_usize(4));
        let combined_limbs = two_bit_limbs
            .iter()
            .tuples()
            .map(|(&a, &b)| self.mul_add(b, four, a))
            .collect::<Vec<_>>();

        combined_limbs.try_into().unwrap()
    }

    fn constant_u32(&mut self, c: u32) -> U32Target {
        U32Target(self.constant(F::from_canonical_u32(c)))
    }

    fn zero_u32(&mut self) -> U32Target {
        U32Target(self.zero())
    }

    fn one_u32(&mut self) -> U32Target {
        U32Target(self.one())
    }

    fn connect_u32(&mut self, x: U32Target, y: U32Target) {
        self.connect(x.0, y.0)
    }

    fn assert_zero_u32(&mut self, x: U32Target) {
        self.assert_zero(x.0)
    }

    fn conditional_assert_zero_u32(&mut self, cond: BoolTarget, a: U32Target) {
        let zero = self.zero();
        self.conditional_assert_eq(cond, a.0, zero);
    }

    fn is_zero_u32(&mut self, x: U32Target) -> BoolTarget {
        let zero = self.zero_u32();
        self.is_equal_u32(zero, x)
    }

    fn is_equal_u32(&mut self, a: U32Target, b: U32Target) -> BoolTarget {
        self.is_equal(a.0, b.0)
    }

    fn conditional_assert_eq_u32(&mut self, cond: BoolTarget, a: U32Target, b: U32Target) {
        self.conditional_assert_eq(cond, a.0, b.0)
    }

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `U32ArithmeticGate`.
    fn arithmetic_u32_special_cases(
        &mut self,
        x: U32Target,
        y: U32Target,
        z: U32Target,
    ) -> Option<(U32Target, U32Target)> {
        let x_const = self.target_as_constant(x.0);
        let y_const = self.target_as_constant(y.0);
        let z_const = self.target_as_constant(z.0);

        // If both terms are constant, return their (constant) sum.
        let first_term_const = if let (Some(xx), Some(yy)) = (x_const, y_const) {
            Some(xx * yy)
        } else {
            None
        };

        if let (Some(a), Some(b)) = (first_term_const, z_const) {
            let sum = (a + b).to_canonical_u64();
            let (low, high) = (sum as u32, (sum >> 32) as u32);
            return Some((self.constant_u32(low), self.constant_u32(high)));
        }

        // If either multiplicand is the constant zero, the result is just `z`.
        if x_const == Some(F::ZERO) || y_const == Some(F::ZERO) {
            let zero = self.zero_u32();
            return Some((z, zero));
        }

        // If one multiplicand is the constant one and the addend is the constant zero,
        // the result is the other multiplicand.
        if z_const == Some(F::ZERO) {
            if x_const == Some(F::ONE) {
                let zero = self.zero_u32();
                return Some((y, zero));
            }
            if y_const == Some(F::ONE) {
                let zero = self.zero_u32();
                return Some((x, zero));
            }
        }

        None
    }

    // Returns x * y + z.
    fn mul_add_u32(&mut self, x: U32Target, y: U32Target, z: U32Target) -> (U32Target, U32Target) {
        if let Some(result) = self.arithmetic_u32_special_cases(x, y, z) {
            return result;
        }

        // See if we've already computed the same operation.
        let operation = U32ArithmeticOperation {
            multiplicand_0: x,
            multiplicand_1: y,
            addend: z,
        };
        if let Some(&result) = self.u32_arithmetic_results.get(&operation) {
            return result;
        }

        let gate = U32ArithmeticGate::<F, D>::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_multiplicand_0(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_multiplicand_1(copy)), y.0);
        self.connect(Target::wire(row, gate.wire_ith_addend(copy)), z.0);

        let output_low = U32Target(Target::wire(row, gate.wire_ith_output_low_half(copy)));
        let output_high = U32Target(Target::wire(row, gate.wire_ith_output_high_half(copy)));

        self.u32_arithmetic_results
            .insert(operation, (output_low, output_high));
        (output_low, output_high)
    }

    fn add_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target) {
        let one = self.one_u32();
        self.mul_add_u32(a, one, b)
    }

    fn add_u32_lo(&mut self, a: U32Target, b: U32Target) -> U32Target {
        let (result, _) = self.add_u32(a, b);
        result
    }

    fn add_many_u32(&mut self, to_add: &[U32Target]) -> (U32Target, U32Target) {
        match to_add.len() {
            0 => (self.zero_u32(), self.zero_u32()),
            1 => (to_add[0], self.zero_u32()),
            2 => self.add_u32(to_add[0], to_add[1]),
            _ => {
                let zero = self.zero_u32();
                self.add_u32s_with_carry(to_add, zero)
            }
        }
    }

    fn add_u32s_with_carry(
        &mut self,
        to_add: &[U32Target],
        carry: U32Target,
    ) -> (U32Target, U32Target) {
        if to_add.is_empty() {
            return (self.zero_u32(), carry);
        }
        if to_add.len() == 1 {
            return self.add_u32(to_add[0], carry);
        }

        // Drop addends that are the constant zero; they don't affect the sum.
        let filtered: Vec<U32Target> = to_add
            .iter()
            .copied()
            .filter(|t| self.target_as_constant(t.0) != Some(F::ZERO))
            .collect();
        if filtered.len() < to_add.len() {
            if filtered.is_empty() {
                let zero = self.zero_u32();
                return self.add_u32(zero, carry);
            }
            if filtered.len() == 1 {
                return self.add_u32(filtered[0], carry);
            }
            return self.add_u32s_with_carry(&filtered, carry);
        }

        // See if we've already computed the same operation.
        let operation = U32AddManyOperation {
            addends: to_add.to_vec(),
            carry,
        };
        if let Some(&result) = self.u32_add_many_results.get(&operation) {
            return result;
        }

        let num_addends = to_add.len();
        let mut expand_num_addends = to_add.len();
        let num_ops = U32AddManyGate::<F, D>::new_from_config(self.config(), num_addends).num_ops;
        while expand_num_addends < MAX_NUM_ADDENDS
            && U32AddManyGate::<F, D>::new_from_config(self.config(), expand_num_addends + 1)
                .num_ops
                == num_ops
        {
            expand_num_addends += 1;
        }

        let gate = U32AddManyGate::<F, D>::new_from_config(self.config(), expand_num_addends);
        let (row, copy) = self.find_slot(gate, &[F::from_canonical_usize(expand_num_addends)], &[]);
        let zero = self.zero();

        for j in 0..num_addends {
            self.connect(
                Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                to_add[j].0,
            );
        }
        // We add as many addends as possible to the gate while maintaining the same number of operations
        // The reason for this is to minimize the number of distinct custom gates
        for j in num_addends..expand_num_addends {
            self.connect(
                Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                zero,
            );
        }
        self.connect(Target::wire(row, gate.wire_ith_carry(copy)), carry.0);

        let output = U32Target(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_carry = U32Target(Target::wire(row, gate.wire_ith_output_carry(copy)));

        self.u32_add_many_results
            .insert(operation, (output, output_carry));
        (output, output_carry)
    }

    fn mul_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target) {
        let zero = self.zero_u32();
        self.mul_add_u32(a, b, zero)
    }

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_u32(&mut self, x: U32Target, y: U32Target, borrow: U32Target) -> (U32Target, U32Target) {
        // Subtracting the constant zero with no incoming borrow is the identity.
        if self.target_as_constant(y.0) == Some(F::ZERO)
            && self.target_as_constant(borrow.0) == Some(F::ZERO)
        {
            let zero = self.zero_u32();
            return (x, zero);
        }

        // See if we've already computed the same operation.
        let operation = U32SubtractionOperation { x, y, borrow };
        if let Some(&result) = self.u32_sub_results.get(&operation) {
            return result;
        }

        let gate = U32SubtractionGate::<F, D>::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_input_x(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_input_y(copy)), y.0);
        self.connect(
            Target::wire(row, gate.wire_ith_input_borrow(copy)),
            borrow.0,
        );

        let output_result = U32Target(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_borrow = U32Target(Target::wire(row, gate.wire_ith_output_borrow(copy)));

        self.u32_sub_results
            .insert(operation, (output_result, output_borrow));
        (output_result, output_borrow)
    }
}

#[derive(Debug, Default)]
pub struct SplitToU32Generator<F: RichField + Extendable<D>, const D: usize> {
    x: Target,
    low: U32Target,
    high: U32Target,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SplitToU32Generator<F, D>
{
    fn id(&self) -> String {
        "SplitToU32Generator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target_u32(self.low)?;
        dst.write_target_u32(self.high)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let x = src.read_target()?;
        let low = src.read_target_u32()?;
        let high = src.read_target_u32()?;
        Ok(Self {
            x,
            low,
            high,
            _phantom: PhantomData,
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.x]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let x = witness.get_target(self.x);
        let x_u64 = x.to_canonical_u64();
        let low = x_u64 as u32;
        let high = (x_u64 >> 32) as u32;

        out_buffer.set_u32_target(self.low, low)?;
        out_buffer.set_u32_target(self.high, high)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    #[allow(unused_imports)]
    use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use rand::Rng;
    use rand::rngs::OsRng;

    use super::*;
    use crate::types::config::{Builder, C, CIRCUIT_CONFIG, F};
    use crate::uint::u32::witness::WitnessU32;

    #[test]
    pub fn test_split_u64_to_u32s_le() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let x = builder.add_virtual_target();
        let [x_ls_32_bit, x_ms_32_bit] = builder.split_u64_to_u32s_le(x);

        let mut pw = PartialWitness::<F>::new();

        let x_value = rand::thread_rng().r#gen::<u64>() & 0xFFFFFFFFFFFFFFFE;
        let x_ls_32_bit_value = (x_value & 0xFFFF_FFFF) as u32;
        let x_ms_32_bit_value = (x_value >> 32) as u32;

        pw.set_target(x, F::from_canonical_u64(x_value))?;
        pw.set_u32_target(x_ls_32_bit, x_ls_32_bit_value)?;
        pw.set_u32_target(x_ms_32_bit, x_ms_32_bit_value)?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    pub fn test_add_many_u32s() -> Result<()> {
        type C = PoseidonGoldilocksConfig;

        const NUM_ADDENDS: usize = 15;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = Builder::new(config);

        let mut rng = OsRng;
        let mut to_add = Vec::new();
        let mut sum = 0u64;
        for _ in 0..NUM_ADDENDS {
            let x: u32 = rng.r#gen();
            sum += x as u64;
            to_add.push(builder.constant_u32(x));
        }
        let carry = builder.zero_u32();
        let (result_low, result_high) = builder.add_u32s_with_carry(&to_add, carry);
        let expected_low = builder.constant_u32((sum % (1 << 32)) as u32);
        let expected_high = builder.constant_u32((sum >> 32) as u32);

        builder.connect_u32(result_low, expected_low);
        builder.connect_u32(result_high, expected_high);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
