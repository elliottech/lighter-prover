// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/nonnative/gadgets/biguint.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::array;

use anyhow::Result;
use itertools::Itertools;
use num::{BigUint, One, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::types::{PrimeField, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;

use super::bigint::{BigIntTarget, SignTarget};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
use crate::uint::u32::witness::{GeneratedValuesU32, WitnessU32};
use crate::utils::{CircuitBuilderUtils, ceil_div_usize};

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct BigUintTarget {
    pub limbs: Vec<U32Target>,
}

impl From<Vec<U32Target>> for BigUintTarget {
    fn from(limbs: Vec<U32Target>) -> Self {
        BigUintTarget { limbs }
    }
}

impl<const N: usize> From<&[U32Target; N]> for BigUintTarget {
    fn from(limbs: &[U32Target; N]) -> Self {
        BigUintTarget {
            limbs: limbs.iter().copied().collect(),
        }
    }
}

impl From<&[U32Target]> for BigUintTarget {
    fn from(limbs: &[U32Target]) -> Self {
        BigUintTarget {
            limbs: limbs.to_vec(),
        }
    }
}

/// Assumes Targets are already 32-bit integers.
impl From<&[Target]> for BigUintTarget {
    fn from(limbs: &[Target]) -> Self {
        BigUintTarget {
            limbs: limbs.iter().map(|arg0: &Target| U32Target(*arg0)).collect(),
        }
    }
}

impl From<U32Target> for BigUintTarget {
    fn from(limb: U32Target) -> Self {
        BigUintTarget { limbs: vec![limb] }
    }
}

impl BigUintTarget {
    pub fn num_limbs(&self) -> usize {
        self.limbs.len()
    }

    pub fn get_limb(&self, i: usize) -> U32Target {
        self.limbs[i]
    }

    pub fn bit_len(&self) -> usize {
        self.limbs.len() * 32
    }

    pub fn byte_len(&self) -> usize {
        self.limbs.len() * 4
    }

    /// This function assumes that the target is already 32 bit integer. Caller has to ensure that. (e.g. using `builder.range_check()`)
    pub fn from_unsafe(target: Target) -> Self {
        BigUintTarget {
            limbs: vec![U32Target(target)],
        }
    }
}

pub trait CircuitBuilderBiguint<F: RichField + Extendable<D>, const D: usize> {
    fn register_public_input_biguint(&mut self, value: &BigUintTarget);

    #[must_use]
    fn add_virtual_biguint_target_unsafe(&mut self, num_limbs: usize) -> BigUintTarget;
    #[must_use]
    fn add_virtual_biguint_target_safe(&mut self, num_limbs: usize) -> BigUintTarget;
    #[must_use]
    fn add_virtual_biguint_public_input_unsafe(&mut self, num_limbs: usize) -> BigUintTarget;
    #[must_use]
    fn add_virtual_biguint_public_input_safe(&mut self, num_limbs: usize) -> BigUintTarget;

    fn biguint_vector_diff(&mut self, new: &BigUintTarget, old: &BigUintTarget) -> BigUintTarget;
    fn biguint_vector_sum(
        &mut self,
        cond: BoolTarget,
        new: &BigUintTarget,
        old: &BigUintTarget,
    ) -> BigUintTarget;

    fn constant_biguint(&mut self, value: &BigUint) -> BigUintTarget;
    fn zero_biguint(&mut self) -> BigUintTarget;
    fn one_biguint(&mut self) -> BigUintTarget;

    #[must_use]
    fn is_zero_biguint(&mut self, target: &BigUintTarget) -> BoolTarget;

    fn connect_biguint(&mut self, lhs: &BigUintTarget, rhs: &BigUintTarget);

    #[must_use]
    fn select_biguint(
        &mut self,
        cond: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> BigUintTarget;

    fn pad_biguints(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> (BigUintTarget, BigUintTarget);
    fn trim_biguint(&mut self, a: &BigUintTarget, final_num_limbs: usize) -> BigUintTarget;

    fn try_trim_biguint(
        &mut self,
        a: &BigUintTarget,
        final_num_limbs: usize,
    ) -> (BoolTarget, BigUintTarget);

    /// Add two `BigUintTarget`s, with the size of the higher limb count
    fn add_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;

    /// Add two `BigUintTarget`s, assuming that the result will fit in `num_limbs`. `num_limbs` should be at least the maximum of `a` and `b` limb counts.
    fn add_biguint_non_carry(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigUintTarget;

    /// Add multiple `BigUintTarget`s, assuming that the result will fit in `num_limbs`. `num_limbs` should be at least the maximum of `a` and `b` limb counts.
    fn add_biguint_multiple(
        &mut self,
        targets: &[&BigUintTarget],
        num_limbs: usize,
    ) -> BigUintTarget;

    /// Subtract two `BigUintTarget`s. We assume that the first is larger than the second.
    fn sub_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    fn try_sub_biguint(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> (BigUintTarget, U32Target);

    fn mul_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget;
    /// Multiply two `BigUintTarget`s, assuming that the result will not overflow.
    fn mul_biguint_non_carry(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigUintTarget;

    fn mul_biguint_by_bool(&mut self, a: &BigUintTarget, b: BoolTarget) -> BigUintTarget;

    /// Returns x * y + z. This is no more efficient than mul-then-add; it's purely for convenience (only need to call one CircuitBuilder function).
    fn mul_add_biguint(
        &mut self,
        x: &BigUintTarget,
        y: &BigUintTarget,
        z: &BigUintTarget,
    ) -> BigUintTarget;

    /// Returns targets[0] * targets[1] * ... * targets[n-1].
    fn mul_many_biguint_non_carry(
        &mut self,
        targets: &[&BigUintTarget],
        num_limbs: usize,
    ) -> BigUintTarget;

    /// Returns the bit representation of the integer in little-endian order.
    fn split_le_biguint<const BITS: usize>(&mut self, target: &BigUintTarget)
    -> [BoolTarget; BITS];
    /// Returns the sum of `bits[i] * 2^i`. This function does not validate that the bits are 0 or 1.
    /// Output of the `split_le_biguint` can be safely used as input to this function.
    fn le_sum_biguint(&mut self, bits: &[BoolTarget]) -> BigUintTarget;
    fn le_sum_bytes_biguint(&mut self, bytes: &[U8Target]) -> BigUintTarget;
    fn biguint_from_bytes_be(&mut self, bytes: &[U8Target]) -> BigUintTarget;

    fn assert_zero_biguint(&mut self, target: &BigUintTarget);

    #[must_use]
    fn is_equal_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BoolTarget;

    /// Converts a `BigUintTarget` value to a `Target` if possible, otherwise fails.
    fn biguint_to_target_safe(&mut self, value: &BigUintTarget) -> Target;
    fn biguint_to_target_unsafe(&mut self, value: &BigUintTarget) -> Target;
    fn target_to_biguint(&mut self, target: Target) -> BigUintTarget;
    /// Converts a single limb target to a `BigUintTarget` without range checks, assuming the target is a valid 32-bit integer.
    fn target_to_biguint_single_limb_unsafe(&mut self, target: Target) -> BigUintTarget;

    fn random_access_biguint(
        &mut self,
        access_index: Target,
        v: Vec<BigUintTarget>,
        limb_count: usize,
    ) -> BigUintTarget;

    fn range_check_biguint(&mut self, target: &BigUintTarget, bit_count: usize);

    /// Returns -target as BigIntTarget. Assumes that target is not zero
    fn negative_biguint(&mut self, target: &BigUintTarget) -> BigIntTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBiguint<F, D> for Builder<F, D> {
    fn biguint_to_target_safe(&mut self, value: &BigUintTarget) -> Target {
        if value.num_limbs() == 0 {
            return self.zero();
        }

        if value.num_limbs() == 1 {
            return value.limbs.first().unwrap().0;
        }

        // Verify that all limbs except the first two are zero
        for limb in value.limbs.iter().skip(2) {
            self.assert_zero_u32(*limb);
        }

        let first_limb = value.limbs.first().unwrap().0;
        let second_limb = value.limbs.get(1).unwrap().0;

        let max = self.constant_u64((1u64 << 32) - 1);
        let is_high_max = self.is_equal(second_limb, max);
        self.conditional_assert_zero(is_high_max, first_limb);

        self.mul_const_add(F::from_canonical_u64(1 << 32), second_limb, first_limb)
    }

    fn biguint_to_target_unsafe(&mut self, value: &BigUintTarget) -> Target {
        if value.num_limbs() == 0 {
            return self.zero();
        }

        if value.num_limbs() == 1 {
            return value.limbs.first().unwrap().0;
        }

        let first_limb = value.limbs.first().unwrap().0;
        let second_limb = value.limbs.get(1).unwrap().0;

        self.mul_const_add(F::from_canonical_u64(1 << 32), second_limb, first_limb)
    }

    fn target_to_biguint(&mut self, target: Target) -> BigUintTarget {
        BigUintTarget {
            limbs: self.split_u64_to_u32s_le(target).to_vec(),
        }
    }

    fn register_public_input_biguint(&mut self, value: &BigUintTarget) {
        value.limbs.iter().for_each(|&target| {
            self.register_public_input(target.0);
        });
    }

    fn constant_biguint(&mut self, value: &BigUint) -> BigUintTarget {
        let limb_values = value.to_u32_digits();
        let limbs = limb_values.iter().map(|&l| self.constant_u32(l)).collect();

        BigUintTarget { limbs }
    }

    fn zero_biguint(&mut self) -> BigUintTarget {
        self.constant_biguint(&BigUint::ZERO)
    }

    fn one_biguint(&mut self) -> BigUintTarget {
        self.constant_biguint(&BigUint::one())
    }

    fn connect_biguint(&mut self, lhs: &BigUintTarget, rhs: &BigUintTarget) {
        let min_limbs = lhs.num_limbs().min(rhs.num_limbs());
        for i in 0..min_limbs {
            self.connect_u32(lhs.get_limb(i), rhs.get_limb(i));
        }

        for i in min_limbs..lhs.num_limbs() {
            self.assert_zero_u32(lhs.get_limb(i));
        }
        for i in min_limbs..rhs.num_limbs() {
            self.assert_zero_u32(rhs.get_limb(i));
        }
    }

    fn pad_biguints(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> (BigUintTarget, BigUintTarget) {
        if a.num_limbs() > b.num_limbs() {
            let mut padded_b = b.clone();
            for _ in b.num_limbs()..a.num_limbs() {
                padded_b.limbs.push(self.zero_u32());
            }

            (a.clone(), padded_b)
        } else {
            let mut padded_a = a.clone();
            for _ in a.num_limbs()..b.num_limbs() {
                padded_a.limbs.push(self.zero_u32());
            }

            (padded_a, b.clone())
        }
    }

    fn trim_biguint(&mut self, a: &BigUintTarget, final_num_limbs: usize) -> BigUintTarget {
        assert!(
            a.limbs.len() >= final_num_limbs,
            "Cannot trim more limbs than existing: {} > {}",
            final_num_limbs,
            a.limbs.len()
        );

        for i in final_num_limbs..a.num_limbs() {
            self.assert_zero_u32(a.limbs[i]);
        }

        BigUintTarget {
            limbs: a
                .limbs
                .iter()
                .take(final_num_limbs)
                .copied()
                .collect::<Vec<_>>(),
        }
    }

    fn try_trim_biguint(
        &mut self,
        a: &BigUintTarget,
        final_num_limbs: usize,
    ) -> (BoolTarget, BigUintTarget) {
        assert!(
            a.limbs.len() >= final_num_limbs,
            "Cannot trim more limbs than existing"
        );

        let mut success = self._true();

        for i in final_num_limbs..a.num_limbs() {
            let empty_limb = self.is_zero_u32(a.limbs[i]);
            success = self.and(success, empty_limb);
        }

        (
            success,
            BigUintTarget {
                limbs: a
                    .limbs
                    .iter()
                    .take(final_num_limbs)
                    .copied()
                    .collect::<Vec<_>>(),
            },
        )
    }

    fn add_virtual_biguint_target_unsafe(&mut self, num_limbs: usize) -> BigUintTarget {
        let limbs = self.add_virtual_u32_targets_unsafe(num_limbs);
        BigUintTarget { limbs }
    }

    fn add_virtual_biguint_target_safe(&mut self, num_limbs: usize) -> BigUintTarget {
        let limbs = self.add_virtual_u32_targets_safe(num_limbs);
        BigUintTarget { limbs }
    }

    fn add_virtual_biguint_public_input_unsafe(&mut self, num_limbs: usize) -> BigUintTarget {
        let big = self.add_virtual_biguint_target_unsafe(num_limbs);
        self.register_public_input_biguint(&big);
        big
    }

    fn add_virtual_biguint_public_input_safe(&mut self, num_limbs: usize) -> BigUintTarget {
        let big = self.add_virtual_biguint_target_safe(num_limbs);
        self.register_public_input_biguint(&big);
        big
    }

    fn select_biguint(
        &mut self,
        cond: BoolTarget,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> BigUintTarget {
        let (a_padded, b_padded) = self.pad_biguints(a, b);
        BigUintTarget {
            limbs: a_padded
                .limbs
                .iter()
                .zip(b_padded.limbs.iter())
                .map(|(a_limb, b_limb)| self.select_u32(cond, *a_limb, *b_limb))
                .collect(),
        }
    }

    fn add_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let num_limbs = a.num_limbs().max(b.num_limbs());

        let mut combined_limbs = vec![];
        let mut carry = self.zero_u32();
        for i in 0..num_limbs {
            let a_limb = if i < a.num_limbs() {
                a.limbs[i]
            } else {
                self.zero_u32()
            };
            let b_limb = if i < b.num_limbs() {
                b.limbs[i]
            } else {
                self.zero_u32()
            };

            let (new_limb, new_carry) = self.add_many_u32(&[carry, a_limb, b_limb]);
            carry = new_carry;
            combined_limbs.push(new_limb);
        }

        combined_limbs.push(carry);

        BigUintTarget {
            limbs: combined_limbs,
        }
    }

    fn add_biguint_non_carry(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigUintTarget {
        let max_limbs = a.num_limbs().max(b.num_limbs());
        // Use trim if you want to get rid of extra limbs
        assert!(
            max_limbs <= num_limbs,
            "Sum of {} limb and {} limb big integers can't be expressed with {} limbs",
            a.num_limbs(),
            b.num_limbs(),
            num_limbs
        );

        let mut combined_limbs = Vec::<U32Target>::with_capacity(num_limbs);
        let mut carry = self.zero_u32();
        for i in 0..num_limbs {
            let a_limb = if i < a.num_limbs() {
                a.limbs[i]
            } else {
                self.zero_u32()
            };
            let b_limb = if i < b.num_limbs() {
                b.limbs[i]
            } else {
                self.zero_u32()
            };

            let (new_limb, new_carry) = self.add_many_u32(&[carry, a_limb, b_limb]);
            carry = new_carry;
            combined_limbs.push(new_limb);
        }

        self.assert_zero_u32(carry);

        BigUintTarget {
            limbs: combined_limbs,
        }
    }

    fn add_biguint_multiple(
        &mut self,
        targets: &[&BigUintTarget],
        num_limbs: usize,
    ) -> BigUintTarget {
        // create vector of BigUintTargets to store partial sums
        let len = targets.len();
        if len == 1 {
            return targets[0].clone();
        }
        // create a vector to store the partial sums
        let mut partial_sums: Vec<BigUintTarget> = Vec::<BigUintTarget>::with_capacity(len);
        for i in 0..len {
            if i % 15 == 0 {
                // Get a slice with at most 15 elements
                let end = if i + 15 > len { len } else { i + 15 };
                let target = &targets[i..end];

                // Get maximum number of limbs from the slice
                let max_limbs = target.iter().map(|t| t.num_limbs()).max().unwrap();
                assert!(
                    max_limbs <= num_limbs,
                    "Maximum limb exceeds the number of limbs: {} > {}",
                    max_limbs,
                    num_limbs,
                );
                let mut combined_limbs = Vec::<U32Target>::with_capacity(num_limbs);
                let mut carry = self.zero_u32();
                for i in 0..num_limbs {
                    // If given limb exists in the element of the target slice, use it, otherwise use zero limb
                    let mut partial_limbs: [U32Target; 16] = array::from_fn(|j| {
                        target
                            .get(j)
                            .map(|t| t.limbs.get(i).copied().unwrap_or_else(|| self.zero_u32()))
                            .unwrap_or_else(|| self.zero_u32())
                    });
                    // prepend the carry to the beginning of the partial limbs array and call add_many_u32
                    partial_limbs[partial_limbs.len() - 1] = carry;

                    let (new_limb, new_carry) = self.add_many_u32(&partial_limbs);
                    carry = new_carry;
                    combined_limbs.push(new_limb);
                }
                self.assert_zero_u32(carry);
                partial_sums.push(BigUintTarget {
                    limbs: combined_limbs,
                });
            }
        }
        self.add_biguint_multiple(&partial_sums.iter().collect::<Vec<_>>(), num_limbs)
    }

    /// Fails if a < b
    fn sub_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let (result, borrow) = self.try_sub_biguint(a, b);
        self.assert_zero_u32(borrow);
        result
    }

    fn try_sub_biguint(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
    ) -> (BigUintTarget, U32Target) {
        let (a, b) = self.pad_biguints(a, b);
        let num_limbs = a.limbs.len();

        let mut result_limbs = vec![];

        let mut borrow = self.zero_u32();
        for i in 0..num_limbs {
            let (result, new_borrow) = self.sub_u32(a.limbs[i], b.limbs[i], borrow);
            result_limbs.push(result);
            borrow = new_borrow;
        }

        (
            BigUintTarget {
                limbs: result_limbs,
            },
            borrow,
        )
    }

    fn assert_zero_biguint(&mut self, target: &BigUintTarget) {
        for &limb in &target.limbs {
            self.assert_zero_u32(limb);
        }
    }

    fn mul_biguint_non_carry(
        &mut self,
        a: &BigUintTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigUintTarget {
        let max_limbs = a.num_limbs().max(b.num_limbs());
        assert!(
            max_limbs <= num_limbs,
            "Mul of {} limb and {} limb BigUintTargets can't be expressed with {} limbs",
            a.num_limbs(),
            b.num_limbs(),
            num_limbs
        );

        let total_limbs = a.limbs.len() + b.limbs.len();
        let num_limbs = num_limbs.min(total_limbs);

        let mut to_add = vec![vec![]; total_limbs];
        for i in 0..a.limbs.len() {
            for j in 0..b.limbs.len() {
                let (product, carry) = self.mul_u32(a.limbs[i], b.limbs[j]);
                to_add[i + j].push(product);
                to_add[i + j + 1].push(carry);
            }
        }

        let mut combined_limbs = vec![];
        let mut carry = self.zero_u32();
        for summands in &mut to_add {
            let (new_result, new_carry) = self.add_u32s_with_carry(summands, carry);
            combined_limbs.push(new_result);
            carry = new_carry;
        }

        self.assert_zero_u32(carry);

        for i in num_limbs..total_limbs {
            self.assert_zero_u32(combined_limbs[i]);
        }
        combined_limbs.resize(num_limbs, self.zero_u32());

        BigUintTarget {
            limbs: combined_limbs,
        }
    }

    fn mul_biguint(&mut self, a: &BigUintTarget, b: &BigUintTarget) -> BigUintTarget {
        let total_limbs = a.limbs.len() + b.limbs.len();

        let mut to_add = vec![vec![]; total_limbs];
        for i in 0..a.limbs.len() {
            for j in 0..b.limbs.len() {
                let (product, carry) = self.mul_u32(a.limbs[i], b.limbs[j]);
                to_add[i + j].push(product);
                to_add[i + j + 1].push(carry);
            }
        }

        let mut combined_limbs = vec![];
        let mut carry = self.zero_u32();
        for summands in &mut to_add {
            let (new_result, new_carry) = self.add_u32s_with_carry(summands, carry);
            combined_limbs.push(new_result);
            carry = new_carry;
        }
        combined_limbs.push(carry);

        BigUintTarget {
            limbs: combined_limbs,
        }
    }

    fn mul_biguint_by_bool(&mut self, a: &BigUintTarget, b: BoolTarget) -> BigUintTarget {
        let t = b.target;

        BigUintTarget {
            limbs: a
                .limbs
                .iter()
                .map(|&l| U32Target(self.mul(l.0, t)))
                .collect(),
        }
    }

    fn mul_add_biguint(
        &mut self,
        x: &BigUintTarget,
        y: &BigUintTarget,
        z: &BigUintTarget,
    ) -> BigUintTarget {
        let prod = self.mul_biguint(x, y);
        self.add_biguint(&prod, z)
    }

    fn mul_many_biguint_non_carry(
        &mut self,
        targets: &[&BigUintTarget],
        num_limbs: usize,
    ) -> BigUintTarget {
        let mut result = targets[0].clone();
        for i in 1..targets.len() {
            result = self.mul_biguint_non_carry(&result, targets[i], num_limbs);
        }
        result
    }

    fn is_zero_biguint(&mut self, target: &BigUintTarget) -> BoolTarget {
        let mut target_clone = target.clone();
        if target_clone.num_limbs() == 0 {
            target_clone.limbs.push(self.zero_u32());
        }

        let zero: Target = self.zero();
        let and_results = target_clone
            .limbs
            .iter()
            .map(|&l| self.is_equal(l.0, zero))
            .collect::<Vec<_>>();

        self.multi_and(&and_results)
    }

    fn is_equal_biguint(&mut self, lhs: &BigUintTarget, rhs: &BigUintTarget) -> BoolTarget {
        let min_limbs = lhs.num_limbs().min(rhs.num_limbs());
        let max_limbs = lhs.num_limbs().max(rhs.num_limbs());
        let mut and_results = Vec::<BoolTarget>::with_capacity(max_limbs);

        for i in 0..min_limbs {
            and_results.push(self.is_equal_u32(lhs.get_limb(i), rhs.get_limb(i)));
        }

        for i in min_limbs..lhs.num_limbs() {
            and_results.push(self.is_zero_u32(lhs.get_limb(i)));
        }
        for i in min_limbs..rhs.num_limbs() {
            and_results.push(self.is_zero_u32(rhs.get_limb(i)));
        }

        self.multi_and(&and_results)
    }

    fn split_le_biguint<const BITS: usize>(
        &mut self,
        target: &BigUintTarget,
    ) -> [BoolTarget; BITS] {
        let mut bits = Vec::<BoolTarget>::with_capacity(BITS);

        let full_limbs = BITS / 32;
        let remaining_bits = BITS % 32;

        let limbs = target.limbs.iter().map(|&l| l.0).collect::<Vec<_>>();
        if BITS > target.bit_len() {
            // If the target is smaller than the number of bits we want to split it into, we pad it with zeros
            for i in 0..target.num_limbs() {
                let current_limb_bits = self.split_le(limbs[i], 32);
                bits.extend_from_slice(&current_limb_bits);
            }
            bits.resize(BITS, self._false());
        } else {
            for i in 0..full_limbs {
                let current_limb_bits = self.split_le(limbs[i], 32);
                bits.extend_from_slice(&current_limb_bits);
            }
            if remaining_bits > 0 {
                let mut current_limb_bits = self.split_le(limbs[full_limbs], 32);

                for i in remaining_bits..32 {
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

    fn le_sum_biguint(&mut self, bits: &[BoolTarget]) -> BigUintTarget {
        let len = bits.len();
        let full_limbs_count = len / 32;

        let mut limbs = vec![];
        for i in 0..full_limbs_count {
            limbs.push(U32Target(self.le_sum(bits[i * 32..(i + 1) * 32].iter())));
        }

        if !len.is_multiple_of(32) {
            limbs.push(U32Target(self.le_sum(bits[full_limbs_count * 32..].iter())));
        }

        BigUintTarget { limbs }
    }

    fn le_sum_bytes_biguint(&mut self, bytes: &[U8Target]) -> BigUintTarget {
        let len = bytes.len();
        let full_limbs_count = len / 4;

        let mut limbs = vec![];
        for i in 0..full_limbs_count {
            limbs.push(U32Target(self.le_sum_bytes(&bytes[i * 4..(i + 1) * 4])));
        }

        if !len.is_multiple_of(4) {
            limbs.push(U32Target(self.le_sum_bytes(&bytes[full_limbs_count * 4..])));
        }

        BigUintTarget { limbs }
    }

    #[track_caller]
    fn range_check_biguint(&mut self, target: &BigUintTarget, bit_count: usize) {
        assert!(bit_count != 0, "can't range check with bit_count = 0");

        let limb = ceil_div_usize(bit_count, 32);

        for i in limb..target.num_limbs() {
            self.assert_zero_u32(target.get_limb(i));
        }

        if bit_count % 32 != 0 && limb <= target.num_limbs() {
            self.register_range_check(target.get_limb(limb - 1).0, bit_count % 32);
        }
    }

    fn negative_biguint(&mut self, target: &BigUintTarget) -> BigIntTarget {
        let is_target_zero = self.is_zero_biguint(target);
        let neg_one = self.neg_one();
        let zero = self.zero();
        let sign = self.select(is_target_zero, zero, neg_one);

        BigIntTarget {
            abs: target.clone(),
            sign: SignTarget::new_unsafe(sign),
        }
    }

    fn biguint_from_bytes_be(&mut self, bytes: &[U8Target]) -> BigUintTarget {
        let byte_len = bytes.len();
        if byte_len == 0 {
            return self.zero_biguint();
        }

        let limb_count = byte_len / 4 + (!byte_len.is_multiple_of(4)) as usize;

        let mut padded_bytes = vec![];
        padded_bytes.resize(limb_count * 4 - byte_len, self.zero_u8());
        padded_bytes.extend_from_slice(bytes);

        let eight_bit_shift_multiplier = self.constant(F::from_canonical_u16(256));

        BigUintTarget {
            limbs: padded_bytes
                .iter()
                .rev()
                .collect::<Vec<_>>()
                .chunks(4)
                .map(|four_bytes| {
                    let mut limb = self.zero();

                    for byte in four_bytes.iter().rev() {
                        limb = self.mul_add(limb, eight_bit_shift_multiplier, byte.0);
                    }

                    U32Target(limb)
                })
                .collect(),
        }
    }

    fn target_to_biguint_single_limb_unsafe(&mut self, target: Target) -> BigUintTarget {
        BigUintTarget {
            limbs: vec![U32Target(target)],
        }
    }

    fn random_access_biguint(
        &mut self,
        access_index: Target,
        v: Vec<BigUintTarget>,
        limb_count: usize,
    ) -> BigUintTarget {
        let zero_u32 = self.zero_u32();
        BigUintTarget {
            limbs: (0..limb_count)
                .map(|i| {
                    U32Target(
                        self.random_access(
                            access_index,
                            v.iter()
                                .map(|et| et.limbs.get(i).unwrap_or(&zero_u32).0)
                                .collect::<Vec<Target>>(),
                        ),
                    )
                })
                .collect(),
        }
    }

    fn biguint_vector_diff(&mut self, new: &BigUintTarget, old: &BigUintTarget) -> BigUintTarget {
        assert_eq!(new.num_limbs(), old.num_limbs());

        BigUintTarget {
            limbs: new
                .limbs
                .iter()
                .zip_eq(old.limbs.iter())
                .map(|(&x, &y)| U32Target(self.sub(x.0, y.0)))
                .collect(),
        }
    }

    fn biguint_vector_sum(
        &mut self,
        cond: BoolTarget,
        new: &BigUintTarget,
        old: &BigUintTarget,
    ) -> BigUintTarget {
        BigUintTarget {
            limbs: new
                .limbs
                .iter()
                .zip_eq(old.limbs.iter())
                .map(|(&x, &y)| U32Target(self.mul_add(cond.target, x.0, y.0)))
                .collect(),
        }
    }
}

pub trait WitnessBigUint<F: PrimeField64>: Witness<F> {
    fn get_biguint_target(&self, target: BigUintTarget) -> BigUint;
    fn set_biguint_target(&mut self, target: &BigUintTarget, value: &BigUint) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> WitnessBigUint<F> for T {
    fn get_biguint_target(&self, target: BigUintTarget) -> BigUint {
        target
            .limbs
            .into_iter()
            .rev()
            .fold(BigUint::zero(), |acc, limb| {
                (acc << 32) + self.get_target(limb.0).to_canonical_biguint()
            })
    }

    fn set_biguint_target(&mut self, target: &BigUintTarget, value: &BigUint) -> Result<()> {
        let mut limbs = value.to_u32_digits();
        assert!(target.num_limbs() >= limbs.len());
        limbs.resize(target.num_limbs(), 0);
        for i in 0..target.num_limbs() {
            self.set_u32_target(target.limbs[i], limbs[i])?;
        }

        Ok(())
    }
}

pub trait GeneratedValuesBigUint<F: PrimeField> {
    fn set_biguint_target(&mut self, target: &BigUintTarget, value: &BigUint) -> Result<()>;
}

impl<F: PrimeField> GeneratedValuesBigUint<F> for GeneratedValues<F> {
    fn set_biguint_target(&mut self, target: &BigUintTarget, value: &BigUint) -> Result<()> {
        let mut limbs = value.to_u32_digits();
        assert!(target.num_limbs() >= limbs.len());
        limbs.resize(target.num_limbs(), 0);
        for i in 0..target.num_limbs() {
            self.set_u32_target(target.get_limb(i), limbs[i])?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use num::FromPrimitive;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use rand::Rng;

    use super::*;
    use crate::types::config::{
        BIG_U32_LIMBS, BIG_U64_LIMBS, BIG_U160_LIMBS, C, CIRCUIT_CONFIG, D, F,
    };

    fn add_biguint_multiple_unoptimized(
        builder: &mut Builder<F, D>,
        targets: &[&BigUintTarget],
        num_limbs: usize,
    ) -> BigUintTarget {
        let mut sum = builder.zero_biguint();
        for target in targets {
            sum = builder.add_biguint_non_carry(&sum, target, num_limbs);
        }
        sum
    }

    #[test]
    fn target_to_biguint_conversions() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let a_32_bit_target = builder.add_virtual_target();
        let b_63_bit_target = builder.add_virtual_target();

        let a_32_bit_biguint_target = BigUintTarget::from_unsafe(a_32_bit_target);
        let b_63_bit_biguint_target = builder.target_to_biguint(b_63_bit_target);

        let mut pw = PartialWitness::<F>::new();

        let a_32_bit_value = rand::thread_rng().r#gen::<u32>();
        let b_63_bit_value = rand::thread_rng().r#gen::<u64>() & 0xFFFFFFFFFFFFFFFE;

        let a_32_bit_biguint = BigUint::from_u32(a_32_bit_value).unwrap();
        let b_63_bit_biguint = BigUint::from_u64(b_63_bit_value).unwrap();

        pw.set_target(a_32_bit_target, F::from_canonical_u32(a_32_bit_value))?;
        pw.set_target(b_63_bit_target, F::from_canonical_u64(b_63_bit_value))?;
        pw.set_biguint_target(&a_32_bit_biguint_target, &a_32_bit_biguint)?;
        pw.set_biguint_target(&b_63_bit_biguint_target, &b_63_bit_biguint)?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn biguint_to_target_conversions() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let a_32_bit_biguint = builder.add_virtual_biguint_target_unsafe(BIG_U32_LIMBS);
        let b_63_bit_biguint = builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS);

        let a_32_bit_target = builder.biguint_to_target_safe(&a_32_bit_biguint);
        let b_63_bit_target = builder.biguint_to_target_safe(&b_63_bit_biguint);

        let mut pw = PartialWitness::<F>::new();

        let a_value = rand::thread_rng().r#gen::<u32>();
        let a_biguint = BigUint::from(a_value);
        let b_value = rand::thread_rng().r#gen::<u64>() & 0x7FFFFFFFFFFFFFFF;
        let b_biguint = BigUint::from(b_value);

        pw.set_biguint_target(&a_32_bit_biguint, &a_biguint)?;
        pw.set_target(a_32_bit_target, F::from_canonical_u32(a_value))?;
        pw.set_biguint_target(&b_63_bit_biguint, &b_biguint)?;
        pw.set_target(b_63_bit_target, F::from_canonical_u64(b_value))?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn optimized_biguint_sum_is_correct() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let biguint_count = 120;
        let mut bigs = (0..biguint_count / 5)
            .map(|_| builder.constant_biguint(&BigUint::from_u128(u128::MAX).unwrap()))
            .collect::<Vec<_>>();
        bigs.extend_from_slice(
            (0..biguint_count - (biguint_count / 5))
                .map(|_| {
                    builder.constant_biguint(&BigUint::from_u128(rand::random::<u128>()).unwrap())
                })
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let sum1 = add_biguint_multiple_unoptimized(
            &mut builder,
            &bigs.iter().collect::<Vec<_>>(),
            BIG_U160_LIMBS,
        );
        let sum2 = builder.add_biguint_multiple(&bigs.iter().collect::<Vec<_>>(), BIG_U160_LIMBS);

        builder.connect_biguint(&sum1, &sum2);

        let data = builder.build::<C>();
        let proof = data.prove(PartialWitness::<F>::new()).unwrap();
        data.verify(proof)
    }

    #[test]
    fn le_sum_biguint() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let x_biguint_target = builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS);
        let x_bits_target = builder.split_le_biguint::<63>(&x_biguint_target);

        for i in 0..43 {
            builder.assert_true(x_bits_target[i]);
        }
        for i in 43..63 {
            builder.assert_false(x_bits_target[i]);
        }

        let data = builder.build::<C>();

        let mut pw = PartialWitness::<F>::new();

        let x = 2u64.pow(43) - 1;
        let x_biguint = BigUint::from(x);

        pw.set_biguint_target(&x_biguint_target, &x_biguint)?;

        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }

    #[test]
    fn le_sum_biguint_into_larger() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let x_biguint_target = builder.add_virtual_biguint_target_unsafe(BIG_U32_LIMBS);
        let x_bits_target = builder.split_le_biguint::<63>(&x_biguint_target);

        for i in 0..31 {
            builder.assert_true(x_bits_target[i]);
        }
        for i in 31..63 {
            builder.assert_false(x_bits_target[i]);
        }

        let data = builder.build::<C>();

        let mut pw = PartialWitness::<F>::new();

        let x = 2u64.pow(31) - 1;
        let x_biguint = BigUint::from(x);

        pw.set_biguint_target(&x_biguint_target, &x_biguint)?;

        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }

    #[test]
    #[should_panic(expected = "was set twice with different values")]
    fn le_sum_biguint_fail() {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let x_biguint_target = builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS);
        builder.split_le_biguint::<46>(&x_biguint_target);

        let data = builder.build::<C>();

        let mut pw = PartialWitness::<F>::new();

        let x = 2u64.pow(56) - 1;
        let x_biguint = BigUint::from(x);

        pw.set_biguint_target(&x_biguint_target, &x_biguint)
            .unwrap();

        let result = data.prove(pw);

        if let Err(e) = result {
            panic!("Proving Error: {:?}", e);
        }
    }

    #[test]
    fn range_check_biguint_test() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let mut pw = PartialWitness::<F>::new();

        let x_biguint_target = builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS);

        builder.range_check_biguint(&x_biguint_target, 16);
        builder.range_check_biguint(&x_biguint_target, 32);
        builder.range_check_biguint(&x_biguint_target, 64);
        builder.range_check_biguint(&x_biguint_target, 69);
        builder.range_check_biguint(&x_biguint_target, 96);

        builder.perform_registered_range_checks();

        let data = builder.build::<C>();

        let x_biguint = BigUint::from(45u64);
        pw.set_biguint_target(&x_biguint_target, &x_biguint)?;

        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }

    #[should_panic(expected = "Partition containing Wire")]
    #[test]
    fn range_check_biguint_fail_test() {
        // let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let x_biguint_target = builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS);
        builder.range_check_biguint(&x_biguint_target, 35);

        builder.perform_registered_range_checks();

        let data = builder.build::<C>();

        let mut pw = PartialWitness::<F>::new();
        pw.set_biguint_target(&x_biguint_target, &BigUint::from(1u64 << 35))
            .unwrap();

        data.verify(data.prove(pw).unwrap()).unwrap();
    }

    #[test]
    fn test_biguint_from_bytes_be() {
        let byte_count = rand::thread_rng().gen_range(54..124);
        let bytes = (0..byte_count)
            .map(|_| rand::thread_rng().r#gen::<u8>())
            .collect::<Vec<_>>();

        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let byte_targets = (0..byte_count)
            .map(|_| builder.add_virtual_u8_target_unsafe())
            .collect::<Vec<_>>();
        let x_biguint = builder.biguint_from_bytes_be(&byte_targets);
        let x_biguint_constant = builder.constant_biguint(&BigUint::from_bytes_be(&bytes));
        builder.connect_biguint(&x_biguint, &x_biguint_constant);

        let data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        for (i, &b) in bytes.iter().enumerate() {
            pw.set_target(byte_targets[i].0, F::from_canonical_u8(b))
                .unwrap();
        }

        data.verify(data.prove(pw).unwrap()).unwrap();
    }

    #[test]
    #[should_panic(expected = "was set twice with different values: 16 != 0")]
    fn range_check_biguint_fail2_test() {
        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let mut pw = PartialWitness::<F>::new();

        let x_biguint_target = builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS);

        builder.range_check_biguint(&x_biguint_target, 5);

        builder.perform_registered_range_checks();

        let data = builder.build::<C>();

        let x_biguint = BigUint::from(1u64 << 36);
        pw.set_biguint_target(&x_biguint_target, &x_biguint)
            .unwrap();

        let result = data.prove(pw);

        if let Err(e) = result {
            panic!("Proving Error: {:?}", e);
        }
    }
}
