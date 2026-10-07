// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigUint, One, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::types::{PrimeField, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;

use crate::bigint::biguint::BigUintTarget;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::uint::u16::gadgets::arithmetic_u16::{CircuitBuilderU16, U16Target};
use crate::uint::u16::witness::{GeneratedValuesU16, WitnessU16};
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::utils::ceil_div_usize;

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BigUintU16Target {
    pub limbs: Vec<U16Target>,
}

impl BigUintU16Target {
    pub fn num_limbs(&self) -> usize {
        self.limbs.len()
    }

    pub fn get_limb(&self, i: usize) -> U16Target {
        self.limbs[i]
    }

    pub fn bit_len(&self) -> usize {
        self.limbs.len() * 16
    }
}

pub trait CircuitBuilderBiguint16<F: RichField + Extendable<D>, const D: usize> {
    fn register_public_input_biguint_u16(&mut self, value: &BigUintU16Target);

    #[must_use]
    fn add_virtual_biguint_u16_target_safe(&mut self, num_limbs: usize) -> BigUintU16Target;
    #[must_use]
    fn add_virtual_biguint_u16_target_unsafe(&mut self, num_limbs: usize) -> BigUintU16Target;

    fn constant_biguint_u16(&mut self, value: &BigUint) -> BigUintU16Target;
    fn zero_biguint_u16(&mut self) -> BigUintU16Target;
    fn one_biguint_u16(&mut self) -> BigUintU16Target;

    #[must_use]
    fn is_zero_biguint_u16(&mut self, target: &BigUintU16Target) -> BoolTarget;

    fn connect_biguint_u16(&mut self, lhs: &BigUintU16Target, rhs: &BigUintU16Target);

    fn conditional_assert_not_zero_biguint_u16(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintU16Target,
    );
    fn conditional_assert_not_eq_biguint_u16(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    );

    /// Subtract two `BigUintTarget`s. We assume that the first is larger than the second.
    fn sub_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BigUintU16Target;

    fn try_sub_biguint_u16(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) -> (BigUintU16Target, U16Target);

    fn target_to_biguint_u16(&mut self, target: Target, num_limbs: usize) -> BigUintU16Target;

    fn biguint_u16_to_biguint(&mut self, target: &BigUintU16Target) -> BigUintTarget;
    fn biguint_u16_to_target(&mut self, target: &BigUintU16Target) -> Target;

    fn trim_biguint_u16(
        &mut self,
        a: &BigUintU16Target,
        final_num_limbs: usize,
    ) -> BigUintU16Target;

    fn try_trim_biguint_u16(
        &mut self,
        a: &BigUintU16Target,
        final_num_limbs: usize,
    ) -> (BoolTarget, BigUintU16Target);

    /// Add two `BigUintU16Target`s, with the size of the higher limb count
    fn add_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BigUintU16Target;

    /// Add two `BigUintU16Target`s, assuming that the result will fit in `num_limbs`. `num_limbs` should be at least the maximum of `a` and `b` limb counts.
    fn add_biguint_u16_non_carry(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
        num_limbs: usize,
    ) -> BigUintU16Target;

    fn mul_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BigUintU16Target;
    /// Multiply two `BigUintU16Target`s, assuming that the result will not overflow.
    fn mul_biguint_u16_non_carry(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
        num_limbs: usize,
    ) -> BigUintU16Target;

    /// Returns the bit representation of the integer in little-endian order.
    fn split_le_biguint_u16<const BITS: usize>(
        &mut self,
        target: &BigUintU16Target,
    ) -> [BoolTarget; BITS];
    /// Returns the sum of `bits[i] * 2^i`. This function does not validate that the bits are 0 or 1.
    /// Output of the `split_le_biguint_u16` can be safely used as input to this function.
    fn le_sum_biguint_u16(&mut self, bits: &[BoolTarget]) -> BigUintU16Target;

    fn assert_zero_biguint_u16(&mut self, target: &BigUintU16Target);
    #[must_use]
    fn is_equal_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BoolTarget;

    fn random_access_biguint_u16(
        &mut self,
        access_index: Target,
        v: Vec<BigUintU16Target>,
        limb_count: usize,
    ) -> BigUintU16Target;

    fn range_check_biguint_u16(&mut self, target: &BigUintU16Target, bit_count: usize);

    fn pad_biguints_u16(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) -> (BigUintU16Target, BigUintU16Target);
    #[must_use]
    fn select_biguint_u16(
        &mut self,
        cond: BoolTarget,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) -> BigUintU16Target;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBiguint16<F, D> for Builder<F, D> {
    fn register_public_input_biguint_u16(&mut self, value: &BigUintU16Target) {
        value.limbs.iter().for_each(|&target| {
            self.register_public_input(target.0);
        });
    }

    fn pad_biguints_u16(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) -> (BigUintU16Target, BigUintU16Target) {
        if a.num_limbs() > b.num_limbs() {
            let mut padded_b = b.clone();
            for _ in b.num_limbs()..a.num_limbs() {
                padded_b.limbs.push(self.zero_u16());
            }

            (a.clone(), padded_b)
        } else {
            let mut padded_a = a.clone();
            for _ in a.num_limbs()..b.num_limbs() {
                padded_a.limbs.push(self.zero_u16());
            }

            (padded_a, b.clone())
        }
    }

    fn select_biguint_u16(
        &mut self,
        cond: BoolTarget,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) -> BigUintU16Target {
        let (a_padded, b_padded) = self.pad_biguints_u16(a, b);
        BigUintU16Target {
            limbs: a_padded
                .limbs
                .iter()
                .zip(b_padded.limbs.iter())
                .map(|(a_limb, b_limb)| self.select_u16(cond, *a_limb, *b_limb))
                .collect(),
        }
    }

    fn constant_biguint_u16(&mut self, value: &BigUint) -> BigUintU16Target {
        BigUintU16Target {
            limbs: value
                .to_u32_digits()
                .iter()
                .flat_map(|_u32| [*_u32 as u16, (*_u32 >> 16) as u16])
                .map(|l| self.constant_u16(l))
                .collect(),
        }
    }

    fn zero_biguint_u16(&mut self) -> BigUintU16Target {
        self.constant_biguint_u16(&BigUint::ZERO)
    }

    fn one_biguint_u16(&mut self) -> BigUintU16Target {
        self.constant_biguint_u16(&BigUint::one())
    }

    fn connect_biguint_u16(&mut self, lhs: &BigUintU16Target, rhs: &BigUintU16Target) {
        let min_limbs = lhs.num_limbs().min(rhs.num_limbs());
        for i in 0..min_limbs {
            self.connect_u16(lhs.get_limb(i), rhs.get_limb(i));
        }

        for i in min_limbs..lhs.num_limbs() {
            self.assert_zero_u16(lhs.get_limb(i));
        }
        for i in min_limbs..rhs.num_limbs() {
            self.assert_zero_u16(rhs.get_limb(i));
        }
    }

    fn trim_biguint_u16(
        &mut self,
        a: &BigUintU16Target,
        final_num_limbs: usize,
    ) -> BigUintU16Target {
        assert!(
            a.limbs.len() >= final_num_limbs,
            "Cannot trim more limbs than existing"
        );

        for i in final_num_limbs..a.num_limbs() {
            self.assert_zero_u16(a.limbs[i]);
        }

        BigUintU16Target {
            limbs: a
                .limbs
                .iter()
                .take(final_num_limbs)
                .copied()
                .collect::<Vec<_>>(),
        }
    }

    fn try_trim_biguint_u16(
        &mut self,
        a: &BigUintU16Target,
        final_num_limbs: usize,
    ) -> (BoolTarget, BigUintU16Target) {
        assert!(
            a.limbs.len() >= final_num_limbs,
            "Cannot trim more limbs than existing"
        );

        let mut success = self._true();

        for i in final_num_limbs..a.num_limbs() {
            let empty_limb = self.is_zero_u16(a.limbs[i]);
            success = self.and(success, empty_limb);
        }

        (
            success,
            BigUintU16Target {
                limbs: a
                    .limbs
                    .iter()
                    .take(final_num_limbs)
                    .copied()
                    .collect::<Vec<_>>(),
            },
        )
    }

    fn conditional_assert_not_zero_biguint_u16(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintU16Target,
    ) {
        let is_zero = self.is_zero_biguint_u16(a);
        let res = self.and(is_enabled, is_zero);
        self.assert_false(res);
    }

    fn conditional_assert_not_eq_biguint_u16(
        &mut self,
        is_enabled: BoolTarget,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) {
        let eq = self.is_equal_biguint_u16(a, b);
        let res = self.and(is_enabled, eq);
        self.assert_false(res);
    }

    fn add_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BigUintU16Target {
        let num_limbs = a.num_limbs().max(b.num_limbs());

        let mut combined_limbs = vec![];
        let mut carry = self.zero_u16();
        for i in 0..num_limbs {
            let a_limb = if i < a.num_limbs() {
                a.limbs[i]
            } else {
                self.zero_u16()
            };
            let b_limb = if i < b.num_limbs() {
                b.limbs[i]
            } else {
                self.zero_u16()
            };

            let (new_limb, new_carry) = self.add_many_u16(&[carry, a_limb, b_limb]);
            carry = new_carry;
            combined_limbs.push(new_limb);
        }

        combined_limbs.push(carry);

        BigUintU16Target {
            limbs: combined_limbs,
        }
    }

    fn add_biguint_u16_non_carry(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
        num_limbs: usize,
    ) -> BigUintU16Target {
        let max_limbs = a.num_limbs().max(b.num_limbs());
        // Use trim if you want to get rid of extra limbs
        assert!(
            max_limbs <= num_limbs,
            "Sum of {} limb and {} limb big integers can't be expressed with {} limbs",
            a.num_limbs(),
            b.num_limbs(),
            num_limbs
        );

        let mut combined_limbs = Vec::<U16Target>::with_capacity(num_limbs);
        let mut carry = self.zero_u16();
        for i in 0..num_limbs {
            let a_limb = if i < a.num_limbs() {
                a.limbs[i]
            } else {
                self.zero_u16()
            };
            let b_limb = if i < b.num_limbs() {
                b.limbs[i]
            } else {
                self.zero_u16()
            };

            let (new_limb, new_carry) = self.add_many_u16(&[carry, a_limb, b_limb]);
            carry = new_carry;
            combined_limbs.push(new_limb);
        }

        self.assert_zero_u16(carry);

        BigUintU16Target {
            limbs: combined_limbs,
        }
    }

    fn assert_zero_biguint_u16(&mut self, target: &BigUintU16Target) {
        for &limb in &target.limbs {
            self.assert_zero_u16(limb);
        }
    }

    fn mul_biguint_u16_non_carry(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
        num_limbs: usize,
    ) -> BigUintU16Target {
        let max_limbs = a.num_limbs().max(b.num_limbs());
        assert!(
            max_limbs <= num_limbs,
            "Mul of {} limb and {} limb BigUintU16Targets can't be expressed with {} limbs",
            a.num_limbs(),
            b.num_limbs(),
            num_limbs
        );

        let total_limbs = a.limbs.len() + b.limbs.len();
        let num_limbs = num_limbs.min(total_limbs);

        let mut to_add = vec![vec![]; total_limbs];
        for i in 0..a.limbs.len() {
            for j in 0..b.limbs.len() {
                let (product, carry) = self.mul_u16(a.limbs[i], b.limbs[j]);
                to_add[i + j].push(product);
                to_add[i + j + 1].push(carry);
            }
        }

        let mut combined_limbs = vec![];
        let mut carry = self.zero_u16();
        for summands in &mut to_add {
            let (new_result, new_carry) = self.add_u16s_with_carry(summands, carry);
            combined_limbs.push(new_result);
            carry = new_carry;
        }

        self.assert_zero_u16(carry);

        for i in num_limbs..total_limbs {
            self.assert_zero_u16(combined_limbs[i]);
        }
        combined_limbs.resize(num_limbs, self.zero_u16());

        BigUintU16Target {
            limbs: combined_limbs,
        }
    }

    fn mul_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BigUintU16Target {
        let total_limbs = a.limbs.len() + b.limbs.len();

        let mut to_add = vec![vec![]; total_limbs];
        for i in 0..a.limbs.len() {
            for j in 0..b.limbs.len() {
                let (product, carry) = self.mul_u16(a.limbs[i], b.limbs[j]);
                to_add[i + j].push(product);
                to_add[i + j + 1].push(carry);
            }
        }

        let mut combined_limbs = vec![];
        let mut carry = self.zero_u16();
        for summands in &mut to_add {
            let (new_result, new_carry) = self.add_u16s_with_carry(summands, carry);
            combined_limbs.push(new_result);
            carry = new_carry;
        }
        combined_limbs.push(carry);

        BigUintU16Target {
            limbs: combined_limbs,
        }
    }

    fn is_zero_biguint_u16(&mut self, target: &BigUintU16Target) -> BoolTarget {
        let mut target_clone = target.clone();
        if target_clone.num_limbs() == 0 {
            target_clone.limbs.push(self.zero_u16());
        }

        let zero: Target = self.zero();
        let and_results = target_clone
            .limbs
            .iter()
            .map(|&l| self.is_equal(l.0, zero))
            .collect::<Vec<_>>();

        self.multi_and(&and_results)
    }

    fn is_equal_biguint_u16(
        &mut self,
        lhs: &BigUintU16Target,
        rhs: &BigUintU16Target,
    ) -> BoolTarget {
        let min_limbs = lhs.num_limbs().min(rhs.num_limbs());
        let mut accumulator: Target;
        accumulator = self.zero();
        let mut diff: Target;
        let zero = self.zero();

        //In order have overall equality, we must have equality on all limbs
        //We sum the square of the differences for each limb and check whether this is 0
        for i in 0..min_limbs {
            diff = self.sub(lhs.get_limb(i).0, rhs.get_limb(i).0);
            accumulator = self.mul_add(diff, diff, accumulator);
        }

        for i in min_limbs..lhs.num_limbs() {
            accumulator = self.add(accumulator, lhs.get_limb(i).0);
        }
        for i in min_limbs..rhs.num_limbs() {
            accumulator = self.add(accumulator, rhs.get_limb(i).0);
        }

        self.is_equal(accumulator, zero)
    }

    fn split_le_biguint_u16<const BITS: usize>(
        &mut self,
        target: &BigUintU16Target,
    ) -> [BoolTarget; BITS] {
        let mut bits = Vec::<BoolTarget>::with_capacity(BITS);

        let full_limbs = BITS / 16;
        let remaining_bits = BITS % 16;

        let limbs = target.limbs.iter().map(|&l| l.0).collect::<Vec<_>>();
        if BITS > target.bit_len() {
            // If the target is smaller than the number of bits we want to split it into, we pad it with zeros
            for i in 0..target.num_limbs() {
                let current_limb_bits = self.split_le(limbs[i], 16);
                bits.extend_from_slice(&current_limb_bits);
            }
            bits.resize(BITS, self._false());
        } else {
            for i in 0..full_limbs {
                let current_limb_bits = self.split_le(limbs[i], 16);
                bits.extend_from_slice(&current_limb_bits);
            }
            if remaining_bits > 0 {
                let mut current_limb_bits = self.split_le(limbs[full_limbs], 16);

                for i in remaining_bits..16 {
                    self.assert_false(current_limb_bits[i]);
                }
                current_limb_bits.resize(remaining_bits, self._false());
                bits.extend_from_slice(&current_limb_bits);
            }

            // If the target has larger limb size than the number of bits we want to split it into, we assert that the remaining limbs are zero
            let next_limb_index = full_limbs + (remaining_bits > 0) as usize;
            for i in next_limb_index..target.num_limbs() {
                self.assert_zero(limbs[i]);
            }
        }

        bits.try_into().unwrap()
    }

    fn le_sum_biguint_u16(&mut self, bits: &[BoolTarget]) -> BigUintU16Target {
        let len = bits.len();
        let full_limbs_count = len / 16;

        let mut limbs = vec![];
        for i in 0..full_limbs_count {
            limbs.push(U16Target(self.le_sum(bits[i * 16..(i + 1) * 16].iter())));
        }

        if !len.is_multiple_of(16) {
            limbs.push(U16Target(self.le_sum(bits[full_limbs_count * 16..].iter())));
        }

        BigUintU16Target { limbs }
    }

    fn random_access_biguint_u16(
        &mut self,
        access_index: Target,
        v: Vec<BigUintU16Target>,
        limb_count: usize,
    ) -> BigUintU16Target {
        let zero_u16 = self.zero_u16();
        BigUintU16Target {
            limbs: (0..limb_count)
                .map(|i| {
                    U16Target(
                        self.random_access(
                            access_index,
                            v.iter()
                                .map(|et| et.limbs.get(i).unwrap_or(&zero_u16).0)
                                .collect::<Vec<Target>>(),
                        ),
                    )
                })
                .collect(),
        }
    }

    fn range_check_biguint_u16(&mut self, target: &BigUintU16Target, bit_count: usize) {
        assert!(bit_count != 0, "can't range check with bit_count = 0");

        let limb = ceil_div_usize(bit_count, 16);

        for i in limb..target.num_limbs() {
            self.assert_zero_u16(target.get_limb(i));
        }

        if !bit_count.is_multiple_of(16) && limb <= target.num_limbs() {
            self.register_range_check(target.get_limb(limb - 1).0, bit_count % 16);
        }
    }

    fn target_to_biguint_u16(&mut self, target: Target, num_limbs: usize) -> BigUintU16Target {
        BigUintU16Target {
            limbs: self.split_u64_to_u16s_le(target, num_limbs),
        }
    }

    fn biguint_u16_to_biguint(&mut self, target: &BigUintU16Target) -> BigUintTarget {
        let mut limbs = target.limbs.clone();
        if !limbs.len().is_multiple_of(2) {
            limbs.push(self.zero_u16());
        }

        let multiplier = self.constant_u64(1 << 16);
        BigUintTarget {
            limbs: limbs
                .chunks(2)
                .map(|chunk| U32Target(self.mul_add(chunk[1].0, multiplier, chunk[0].0)))
                .collect(),
        }
    }

    fn biguint_u16_to_target(&mut self, target: &BigUintU16Target) -> Target {
        assert!(target.num_limbs() < 5, "too many limbs");

        let multiplier = self.constant_u64(1 << 16);
        let mut acc = target.limbs.last().unwrap_or(&self.zero_u16()).0;
        for limb in target.limbs.iter().rev().skip(1) {
            acc = self.mul_add(acc, multiplier, limb.0);
        }
        acc
    }

    fn add_virtual_biguint_u16_target_safe(&mut self, num_limbs: usize) -> BigUintU16Target {
        BigUintU16Target {
            limbs: self.add_virtual_u16_targets_safe(num_limbs),
        }
    }

    fn add_virtual_biguint_u16_target_unsafe(&mut self, num_limbs: usize) -> BigUintU16Target {
        BigUintU16Target {
            limbs: self.add_virtual_u16_targets_unsafe(num_limbs),
        }
    }

    fn sub_biguint_u16(&mut self, a: &BigUintU16Target, b: &BigUintU16Target) -> BigUintU16Target {
        let (result, borrow) = self.try_sub_biguint_u16(a, b);
        self.assert_zero_u16(borrow);
        result
    }

    fn try_sub_biguint_u16(
        &mut self,
        a: &BigUintU16Target,
        b: &BigUintU16Target,
    ) -> (BigUintU16Target, U16Target) {
        let (a, b) = self.pad_biguints_u16(a, b);
        let num_limbs = a.limbs.len();

        let mut result_limbs = vec![];

        let mut borrow = self.zero_u16();
        for i in 0..num_limbs {
            let (result, new_borrow) = self.sub_u16(a.limbs[i], b.limbs[i], borrow);
            result_limbs.push(result);
            borrow = new_borrow;
        }

        (
            BigUintU16Target {
                limbs: result_limbs,
            },
            borrow,
        )
    }
}

pub trait WitnessBigUintU16<F: PrimeField64>: Witness<F> {
    fn get_biguint_u16_target(&self, target: BigUintU16Target) -> BigUint;
    fn set_biguint_u16_target(&mut self, target: &BigUintU16Target, value: &BigUint) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> WitnessBigUintU16<F> for T {
    fn get_biguint_u16_target(&self, target: BigUintU16Target) -> BigUint {
        target
            .limbs
            .into_iter()
            .rev()
            .fold(BigUint::zero(), |acc, limb| {
                (acc << 16) + self.get_target(limb.0).to_canonical_biguint()
            })
    }

    fn set_biguint_u16_target(&mut self, target: &BigUintU16Target, value: &BigUint) -> Result<()> {
        let mut limbs = value
            .to_u32_digits()
            .iter()
            .flat_map(|_u32| [*_u32 as u16, (*_u32 >> 16) as u16])
            .collect::<Vec<_>>();
        assert!(target.num_limbs() >= limbs.len());
        limbs.resize(target.num_limbs(), 0);
        for i in 0..target.num_limbs() {
            self.set_u16_target(target.limbs[i], limbs[i])?;
        }

        Ok(())
    }
}

pub trait GeneratedValuesBigUintU16<F: PrimeField> {
    fn set_biguint_u16_target(&mut self, target: &BigUintU16Target, value: &BigUint) -> Result<()>;
}

impl<F: PrimeField> GeneratedValuesBigUintU16<F> for GeneratedValues<F> {
    fn set_biguint_u16_target(&mut self, target: &BigUintU16Target, value: &BigUint) -> Result<()> {
        let mut limbs = value
            .to_u32_digits()
            .iter()
            .flat_map(|_u32| [*_u32 as u16, (*_u32 >> 16) as u16])
            .collect::<Vec<_>>();
        assert!(target.num_limbs() >= limbs.len());
        limbs.resize(target.num_limbs(), 0);
        for i in 0..target.num_limbs() {
            self.set_u16_target(target.get_limb(i), limbs[i])?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use plonky2::iop::witness::PartialWitness;
    use rand::Rng;

    use super::*;
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    #[test]
    fn biguint_u16_add() {
        let mut rng = rand::thread_rng();
        let lhs = (0..2000)
            .map(|_| {
                let size = rng.gen_range(22..=48);
                let bytes: Vec<u8> = (0..size).map(|_| rng.r#gen()).collect();
                BigUint::from_bytes_be(&bytes)
            })
            .collect::<Vec<_>>();
        let rhs = (0..2000)
            .map(|_| {
                let size = rng.gen_range(22..=48);
                let bytes: Vec<u8> = (0..size).map(|_| rng.r#gen()).collect();
                BigUint::from_bytes_be(&bytes)
            })
            .collect::<Vec<_>>();

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            let add_out_circuit = builder.constant_biguint_u16(&(a + b));

            let a = builder.constant_biguint_u16(a);
            let b = builder.constant_biguint_u16(b);

            let calculated_sum = builder.add_biguint_u16(&a, &b);

            builder.connect_biguint_u16(&add_out_circuit, &calculated_sum);
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }

    #[test]
    fn biguint_u16_sub() {
        let mut rng = rand::thread_rng();
        let mut lhs = (0..2000)
            .map(|_| {
                let size = rng.gen_range(22..=48);
                let bytes: Vec<u8> = (0..size).map(|_| rng.r#gen()).collect();
                BigUint::from_bytes_be(&bytes)
            })
            .collect::<Vec<_>>();
        let mut rhs = (0..2000)
            .map(|_| {
                let size = rng.gen_range(22..=48);
                let bytes: Vec<u8> = (0..size).map(|_| rng.r#gen()).collect();
                BigUint::from_bytes_be(&bytes)
            })
            .collect::<Vec<_>>();
        for i in 0..2000 {
            if lhs[i] < rhs[i] {
                std::mem::swap(&mut lhs[i], &mut rhs[i]);
            }
        }

        rhs[0] = BigUint::ZERO;

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            let sub_out_circuit = builder.constant_biguint_u16(&(a - b));

            let a = builder.constant_biguint_u16(a);
            let b = builder.constant_biguint_u16(b);

            let calculated_sub = builder.sub_biguint_u16(&a, &b);

            builder.connect_biguint_u16(&sub_out_circuit, &calculated_sub);
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }
}
