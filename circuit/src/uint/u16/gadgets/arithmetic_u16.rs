// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gadgets/arithmetic_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::builder::Builder;
use crate::builder::types::{U16AddManyOperation, U16ArithmeticOperation, U16SubtractionOperation};
use crate::uint::u16::gates::add_many_u16::U16AddManyGate;
use crate::uint::u16::gates::arithmetic_u16::U16ArithmeticGate;
use crate::uint::u16::gates::subtraction_u16::U16SubtractionGate;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Default)]
pub struct U16Target(pub Target);

pub trait CircuitBuilderU16<F: RichField + Extendable<D>, const D: usize> {
    #[must_use]
    fn add_virtual_u16_target_unsafe(&mut self) -> U16Target;
    #[must_use]
    fn add_virtual_u16_targets_unsafe(&mut self, n: usize) -> Vec<U16Target>;
    #[must_use]
    fn add_virtual_u16_target_safe(&mut self) -> U16Target;
    #[must_use]
    fn add_virtual_u16_targets_safe(&mut self, n: usize) -> Vec<U16Target>;

    /// Returns a U16Target for the value `c`, which is assumed to be at most 16 bits.
    fn constant_u16(&mut self, c: u16) -> U16Target;
    fn zero_u16(&mut self) -> U16Target;
    fn one_u16(&mut self) -> U16Target;

    #[must_use]
    fn select_u16(&mut self, cond: BoolTarget, a: U16Target, b: U16Target) -> U16Target;
    #[must_use]
    fn select_arr_u16<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[U16Target; N],
        b: &[U16Target; N],
    ) -> [U16Target; N];

    fn connect_u16(&mut self, x: U16Target, y: U16Target);

    #[must_use]
    fn is_equal_u16(&mut self, a: U16Target, b: U16Target) -> BoolTarget;
    fn conditional_assert_eq_u16(&mut self, cond: BoolTarget, a: U16Target, b: U16Target);

    #[must_use]
    fn is_zero_u16(&mut self, x: U16Target) -> BoolTarget;
    fn assert_zero_u16(&mut self, x: U16Target);
    fn conditional_assert_zero_u16(&mut self, cond: BoolTarget, a: U16Target);

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `U16ArithmeticGate`.
    fn arithmetic_u16_special_cases(
        &mut self,
        x: U16Target,
        y: U16Target,
        z: U16Target,
    ) -> Option<(U16Target, U16Target)>;

    // Returns x * y + z.
    fn mul_add_u16(&mut self, x: U16Target, y: U16Target, z: U16Target) -> (U16Target, U16Target);

    fn add_u16(&mut self, a: U16Target, b: U16Target) -> (U16Target, U16Target);

    fn add_u16_lo(&mut self, a: U16Target, b: U16Target) -> U16Target;

    fn add_many_u16(&mut self, to_add: &[U16Target]) -> (U16Target, U16Target);

    fn add_u16s_with_carry(
        &mut self,
        to_add: &[U16Target],
        carry: U16Target,
    ) -> (U16Target, U16Target);

    fn mul_u16(&mut self, a: U16Target, b: U16Target) -> (U16Target, U16Target);

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_u16(&mut self, x: U16Target, y: U16Target, borrow: U16Target) -> (U16Target, U16Target);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderU16<F, D> for Builder<F, D> {
    fn add_virtual_u16_target_unsafe(&mut self) -> U16Target {
        U16Target(self.add_virtual_target())
    }

    fn add_virtual_u16_targets_unsafe(&mut self, n: usize) -> Vec<U16Target> {
        self.add_virtual_targets(n)
            .into_iter()
            .map(U16Target)
            .collect()
    }

    fn add_virtual_u16_target_safe(&mut self) -> U16Target {
        let target = U16Target(self.add_virtual_target());
        self.register_range_check(target.0, 16);
        target
    }

    fn add_virtual_u16_targets_safe(&mut self, n: usize) -> Vec<U16Target> {
        let targets: Vec<U16Target> = self
            .add_virtual_targets(n)
            .into_iter()
            .map(U16Target)
            .collect();
        for target in &targets {
            self.register_range_check(target.0, 16);
        }
        targets
    }

    fn select_u16(&mut self, cond: BoolTarget, a: U16Target, b: U16Target) -> U16Target {
        U16Target(self.select(cond, a.0, b.0))
    }

    fn select_arr_u16<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[U16Target; N],
        b: &[U16Target; N],
    ) -> [U16Target; N] {
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| self.select_u16(cond, *a, *b))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Returns a U16Target for the value `c`, which is assumed to be at most 16 bits.
    fn constant_u16(&mut self, c: u16) -> U16Target {
        U16Target(self.constant(F::from_canonical_u16(c)))
    }

    fn zero_u16(&mut self) -> U16Target {
        U16Target(self.zero())
    }

    fn one_u16(&mut self) -> U16Target {
        U16Target(self.one())
    }

    fn connect_u16(&mut self, x: U16Target, y: U16Target) {
        self.connect(x.0, y.0)
    }

    fn assert_zero_u16(&mut self, x: U16Target) {
        self.assert_zero(x.0)
    }

    fn conditional_assert_zero_u16(&mut self, cond: BoolTarget, a: U16Target) {
        let zero = self.zero();
        self.conditional_assert_eq(cond, a.0, zero);
    }

    fn is_zero_u16(&mut self, x: U16Target) -> BoolTarget {
        let zero = self.zero_u16();
        self.is_equal_u16(zero, x)
    }

    fn is_equal_u16(&mut self, a: U16Target, b: U16Target) -> BoolTarget {
        self.is_equal(a.0, b.0)
    }

    fn conditional_assert_eq_u16(&mut self, cond: BoolTarget, a: U16Target, b: U16Target) {
        self.conditional_assert_eq(cond, a.0, b.0)
    }

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `U16ArithmeticGate`.
    fn arithmetic_u16_special_cases(
        &mut self,
        x: U16Target,
        y: U16Target,
        z: U16Target,
    ) -> Option<(U16Target, U16Target)> {
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
            let (low, high) = (sum as u16, (sum >> 16) as u16);
            return Some((self.constant_u16(low), self.constant_u16(high)));
        }

        // If either multiplicand is the constant zero, the result is just `z`.
        if x_const == Some(F::ZERO) || y_const == Some(F::ZERO) {
            let zero = self.zero_u16();
            return Some((z, zero));
        }

        // If one multiplicand is the constant one and the addend is the constant zero,
        // the result is the other multiplicand.
        if z_const == Some(F::ZERO) {
            if x_const == Some(F::ONE) {
                let zero = self.zero_u16();
                return Some((y, zero));
            }
            if y_const == Some(F::ONE) {
                let zero = self.zero_u16();
                return Some((x, zero));
            }
        }

        None
    }

    // Returns x * y + z.
    fn mul_add_u16(&mut self, x: U16Target, y: U16Target, z: U16Target) -> (U16Target, U16Target) {
        if let Some(result) = self.arithmetic_u16_special_cases(x, y, z) {
            return result;
        }

        // See if we've already computed the same operation.
        let operation = U16ArithmeticOperation {
            multiplicand_0: x,
            multiplicand_1: y,
            addend: z,
        };
        if let Some(&result) = self.u16_arithmetic_results.get(&operation) {
            return result;
        }

        let gate = U16ArithmeticGate::<F, D>::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_multiplicand_0(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_multiplicand_1(copy)), y.0);
        self.connect(Target::wire(row, gate.wire_ith_addend(copy)), z.0);

        let output_low = U16Target(Target::wire(row, gate.wire_ith_output_low_half(copy)));
        let output_high = U16Target(Target::wire(row, gate.wire_ith_output_high_half(copy)));

        self.u16_arithmetic_results
            .insert(operation, (output_low, output_high));
        (output_low, output_high)
    }

    fn add_u16(&mut self, a: U16Target, b: U16Target) -> (U16Target, U16Target) {
        let one = self.one_u16();
        self.mul_add_u16(a, one, b)
    }

    fn add_u16_lo(&mut self, a: U16Target, b: U16Target) -> U16Target {
        let (result, _) = self.add_u16(a, b);
        result
    }

    fn add_many_u16(&mut self, to_add: &[U16Target]) -> (U16Target, U16Target) {
        match to_add.len() {
            0 => (self.zero_u16(), self.zero_u16()),
            1 => (to_add[0], self.zero_u16()),
            2 => self.add_u16(to_add[0], to_add[1]),
            _ => {
                let zero = self.zero_u16();
                self.add_u16s_with_carry(to_add, zero)
            }
        }
    }

    fn add_u16s_with_carry(
        &mut self,
        to_add: &[U16Target],
        carry: U16Target,
    ) -> (U16Target, U16Target) {
        if to_add.is_empty() {
            return (self.zero_u16(), carry);
        }

        if to_add.len() == 1 {
            return self.add_u16(to_add[0], carry);
        }

        // Drop addends that are the constant zero; they don't affect the sum.
        let filtered: Vec<U16Target> = to_add
            .iter()
            .copied()
            .filter(|t| self.target_as_constant(t.0) != Some(F::ZERO))
            .collect();
        if filtered.len() < to_add.len() {
            if filtered.is_empty() {
                let zero = self.zero_u16();
                return self.add_u16(zero, carry);
            }
            if filtered.len() == 1 {
                return self.add_u16(filtered[0], carry);
            }
            return self.add_u16s_with_carry(&filtered, carry);
        }

        // See if we've already computed the same operation.
        let operation = U16AddManyOperation {
            addends: to_add.to_vec(),
            carry,
        };
        if let Some(&result) = self.u16_add_many_results.get(&operation) {
            return result;
        }

        let num_addends = to_add.len();
        let gate = U16AddManyGate::<F, D>::new_from_config(self.config(), num_addends);
        let (row, copy) = self.find_slot(gate, &[F::from_canonical_usize(num_addends)], &[]);

        for j in 0..num_addends {
            self.connect(
                Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                to_add[j].0,
            );
        }
        self.connect(Target::wire(row, gate.wire_ith_carry(copy)), carry.0);

        let output = U16Target(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_carry = U16Target(Target::wire(row, gate.wire_ith_output_carry(copy)));

        self.u16_add_many_results
            .insert(operation, (output, output_carry));
        (output, output_carry)
    }

    fn mul_u16(&mut self, a: U16Target, b: U16Target) -> (U16Target, U16Target) {
        let zero = self.zero_u16();
        self.mul_add_u16(a, b, zero)
    }

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_u16(&mut self, x: U16Target, y: U16Target, borrow: U16Target) -> (U16Target, U16Target) {
        // Subtracting the constant zero with no incoming borrow is the identity.
        if self.target_as_constant(y.0) == Some(F::ZERO)
            && self.target_as_constant(borrow.0) == Some(F::ZERO)
        {
            let zero = self.zero_u16();
            return (x, zero);
        }

        // See if we've already computed the same operation.
        let operation = U16SubtractionOperation { x, y, borrow };
        if let Some(&result) = self.u16_sub_results.get(&operation) {
            return result;
        }

        let gate = U16SubtractionGate::<F, D>::new_from_config(self.config());
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_input_x(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_input_y(copy)), y.0);
        self.connect(
            Target::wire(row, gate.wire_ith_input_borrow(copy)),
            borrow.0,
        );

        let output_result = U16Target(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_borrow = U16Target(Target::wire(row, gate.wire_ith_output_borrow(copy)));

        self.u16_sub_results
            .insert(operation, (output_result, output_borrow));
        (output_result, output_borrow)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    #[allow(unused_imports)]
    use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use rand::Rng;
    use rand::rngs::OsRng;

    use super::*;
    use crate::types::config::Builder;

    #[test]
    pub fn test_add_many_u16s() -> Result<()> {
        type C = PoseidonGoldilocksConfig;

        const NUM_ADDENDS: usize = 15;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = Builder::new(config);

        let mut rng = OsRng;
        let mut to_add = Vec::new();
        let mut sum = 0u32;
        for _ in 0..NUM_ADDENDS {
            let x: u16 = rng.r#gen();
            sum += x as u32;
            to_add.push(builder.constant_u16(x));
        }
        let carry = builder.zero_u16();
        let (result_low, result_high) = builder.add_u16s_with_carry(&to_add, carry);
        let expected_low = builder.constant_u16((sum % (1 << 16)) as u16);
        let expected_high = builder.constant_u16((sum >> 16) as u16);

        builder.connect_u16(result_low, expected_low);
        builder.connect_u16(result_high, expected_high);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
