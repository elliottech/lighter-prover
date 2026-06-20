// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use itertools::Itertools;
use num::bigint::Sign;
use num::{BigInt, Signed};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;

use super::biguint::{BigUintTarget, CircuitBuilderBiguint};
use super::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::biguint::WitnessBigUint;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::utils::CircuitBuilderUtils;

/// Wrapper around BigUintTarget to support signed operations
///
/// `sign`: `1`  &rarr; For positive values
///
/// `sign`: `-1` &rarr; For negative values
///
/// `sign`: `0` &rarr; For zero value
#[derive(Clone, Debug, Default)]
pub struct BigIntTarget {
    pub abs: BigUintTarget,
    pub sign: SignTarget,
}

/// Target that represents the Sign of the BigIntTarget, possible values are -1, 0 and 1
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[allow(clippy::manual_non_exhaustive)]
pub struct SignTarget {
    pub target: Target,
    /// This private field is here to force all instantiations to go through `new_unsafe`.
    _private: (),
}

impl SignTarget {
    pub fn new_unsafe(target: Target) -> SignTarget {
        SignTarget {
            target,
            _private: (),
        }
    }
}

pub trait CircuitBuilderBigInt<F: RichField + Extendable<D>, const D: usize> {
    fn register_public_input_bigint(&mut self, target: &BigIntTarget);

    fn euclidian_div_by_biguint(
        &mut self,
        a: &BigIntTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigIntTarget;

    #[must_use]
    fn add_virtual_sign_target_safe(&mut self) -> SignTarget;
    #[must_use]
    fn add_virtual_sign_target_unsafe(&mut self) -> SignTarget;

    #[must_use]
    fn add_virtual_bigint_target_safe(&mut self, num_limbs: usize) -> BigIntTarget;
    #[must_use]
    fn add_virtual_bigint_target_unsafe(&mut self, num_limbs: usize) -> BigIntTarget;

    #[must_use]
    fn add_virtual_bigint_public_input_safe(&mut self, num_limbs: usize) -> BigIntTarget;
    #[must_use]
    fn add_virtual_bigint_public_input_unsafe(&mut self, num_limbs: usize) -> BigIntTarget;

    fn connect_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget);

    fn assert_sign(&mut self, sign: SignTarget);
    #[must_use]
    fn is_sign_positive(&mut self, sign: SignTarget) -> BoolTarget;
    #[must_use]
    fn is_sign_negative(&mut self, sign: SignTarget) -> BoolTarget;
    fn negate_sign(&mut self, sign: SignTarget) -> SignTarget;
    #[must_use]
    fn is_equal_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BoolTarget;
    #[must_use]
    fn is_not_equal_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BoolTarget;
    #[must_use]
    fn is_zero_bigint(&mut self, a: &BigIntTarget) -> BoolTarget;

    #[must_use]
    fn select_bigint(
        &mut self,
        condition: BoolTarget,
        a: &BigIntTarget,
        b: &BigIntTarget,
    ) -> BigIntTarget;

    fn zero_bigint(&mut self) -> BigIntTarget;

    fn conditional_assert_zero_bigint(&mut self, is_enabled: BoolTarget, a: &BigIntTarget);

    fn signed_target_to_bigint(&mut self, target: SignedTarget) -> BigIntTarget;
    fn bigint_to_signed_target_unsafe(&mut self, target: &BigIntTarget) -> SignedTarget;

    fn biguint_to_bigint(&mut self, target: &BigUintTarget) -> BigIntTarget;

    fn random_access_bigint(
        &mut self,
        access_index: Target,
        v: &[BigIntTarget],
        limb_count: usize,
    ) -> BigIntTarget;

    /// Returns a.abs - b.abs as BigIntTarget
    fn abs_diff(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigIntTarget;
    /// Returns a.abs + b.abs as BigUintTarget using non-carry addition
    fn abs_sum(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigUintTarget;
    /// Returns a - b, using non-carry additions when necessary
    fn sub_bigint_non_carry(
        &mut self,
        a: &BigIntTarget,
        b: &BigIntTarget,
        num_limbs: usize,
    ) -> BigIntTarget;
    /// Return a + b, using non-carry additions when necessary
    fn add_bigint_non_carry(
        &mut self,
        a: &BigIntTarget,
        b: &BigIntTarget,
        num_limbs: usize,
    ) -> BigIntTarget;
    // Return a * b, where b is a BigUintTarget
    fn mul_bigint_with_biguint_non_carry(
        &mut self,
        a: &BigIntTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigIntTarget;
    // Return a * b, where b is a BoolTarget
    fn mul_bigint_by_bool(&mut self, a: &BigIntTarget, b: BoolTarget) -> BigIntTarget;

    fn sub_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigIntTarget;

    fn bigint_vector_diff(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigIntTarget;

    fn bigint_vector_sum(
        &mut self,
        cond: BoolTarget,
        a: &BigIntTarget,
        b: &BigIntTarget,
    ) -> BigIntTarget;

    fn trim_bigint(&mut self, a: &BigIntTarget, final_num_limbs: usize) -> BigIntTarget;

    fn neg_bigint(&mut self, a: &BigIntTarget) -> BigIntTarget;
    fn cmp_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> SignTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBigInt<F, D> for Builder<F, D> {
    fn register_public_input_bigint(&mut self, target: &BigIntTarget) {
        self.register_public_input(target.sign.target);
        self.register_public_input_biguint(&target.abs);
    }

    fn connect_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) {
        self.connect_biguint(&a.abs, &b.abs);
        self.connect(a.sign.target, b.sign.target);
    }

    fn is_sign_negative(&mut self, sign: SignTarget) -> BoolTarget {
        let neg_one = self.neg_one();
        self.is_equal(sign.target, neg_one)
    }

    fn negate_sign(&mut self, sign: SignTarget) -> SignTarget {
        let neg_one = self.neg_one();
        SignTarget::new_unsafe(self.mul(neg_one, sign.target))
    }

    fn is_sign_positive(&mut self, sign: SignTarget) -> BoolTarget {
        let one = self.one();
        self.is_equal(sign.target, one)
    }

    fn is_equal_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BoolTarget {
        let abs_eq = self.is_equal_biguint(&a.abs, &b.abs);
        let sign_eq = self.is_equal(a.sign.target, b.sign.target);
        self.and(abs_eq, sign_eq)
    }

    fn is_not_equal_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BoolTarget {
        let is_equal = self.is_equal_bigint(a, b);
        self.not(is_equal)
    }

    fn is_zero_bigint(&mut self, a: &BigIntTarget) -> BoolTarget {
        let is_sign_zero = self.is_zero(a.sign.target);
        let is_abs_zero = self.is_zero_biguint(&a.abs);
        self.and(is_sign_zero, is_abs_zero)
    }

    fn add_virtual_bigint_target_safe(&mut self, num_limbs: usize) -> BigIntTarget {
        let abs = self.add_virtual_biguint_target_safe(num_limbs);
        let sign = self.add_virtual_sign_target_safe();
        let abs_is_zero = self.is_zero_biguint(&abs);
        let sign_is_zero = self.is_zero(sign.target);
        self.connect(abs_is_zero.target, sign_is_zero.target);
        BigIntTarget { abs, sign }
    }

    fn add_virtual_bigint_target_unsafe(&mut self, num_limbs: usize) -> BigIntTarget {
        let abs = self.add_virtual_biguint_target_unsafe(num_limbs);
        let sign = self.add_virtual_sign_target_unsafe();
        BigIntTarget { abs, sign }
    }

    fn add_virtual_bigint_public_input_safe(&mut self, num_limbs: usize) -> BigIntTarget {
        let sign = self.add_virtual_sign_target_safe();
        self.register_public_input(sign.target);
        let abs = self.add_virtual_biguint_public_input_safe(num_limbs);
        BigIntTarget { abs, sign }
    }

    fn add_virtual_bigint_public_input_unsafe(&mut self, num_limbs: usize) -> BigIntTarget {
        let sign = self.add_virtual_sign_target_unsafe();
        self.register_public_input(sign.target);
        let abs = self.add_virtual_biguint_public_input_unsafe(num_limbs);
        BigIntTarget { abs, sign }
    }

    fn add_virtual_sign_target_safe(&mut self) -> SignTarget {
        let s = SignTarget::new_unsafe(self.add_virtual_target());
        self.assert_sign(s);
        s
    }

    fn add_virtual_sign_target_unsafe(&mut self) -> SignTarget {
        SignTarget::new_unsafe(self.add_virtual_target())
    }

    fn assert_sign(&mut self, sign: SignTarget) {
        let s_sq = self.mul(sign.target, sign.target);
        let s_cube = self.mul(s_sq, sign.target);

        self.connect(s_cube, sign.target);
    }

    fn zero_bigint(&mut self) -> BigIntTarget {
        let abs = self.zero_biguint();
        let sign = SignTarget::new_unsafe(self.zero());
        BigIntTarget { abs, sign }
    }

    fn abs_diff(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigIntTarget {
        let (diff_if_a_gt_b, borrow_a_b) = self.try_sub_biguint(&a.abs, &b.abs);
        let (diff_if_a_lte_b, borrow_b_a) = self.try_sub_biguint(&b.abs, &a.abs);

        BigIntTarget {
            abs: self.select_biguint(
                BoolTarget::new_unsafe(borrow_b_a.0),
                &diff_if_a_gt_b,
                &diff_if_a_lte_b,
            ),
            // borrow_a_b = 0, borrow_b_a = 0 -> 0
            // borrow_a_b = 1, borrow_b_a = 0 -> -1
            // borrow_a_b = 0, borrow_b_a = 1 -> 1
            sign: SignTarget::new_unsafe(self.sub(borrow_b_a.0, borrow_a_b.0)),
        }
    }

    fn abs_sum(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigUintTarget {
        self.add_biguint(&a.abs, &b.abs)
    }

    fn sub_bigint_non_carry(
        &mut self,
        a: &BigIntTarget,
        b: &BigIntTarget,
        num_limbs: usize,
    ) -> BigIntTarget {
        let one = self.one();

        let abs_diff = self.abs_diff(a, b);
        let abs_sum = self.abs_sum(a, b);

        let is_same_sign = self.is_equal(a.sign.target, b.sign.target);

        // Positive - Positive = abs_diff ( 3 - 5 = -2 )
        // Negative - Negative = -1 * abs_diff ( (-3) - (-5) = 2 while abs_diff is 3 - 5 = -2 )
        // Zero     - Zero     = abs_diff ( 0 - 0 = 0)

        // Positive - Negative = abs_sum ( 3 - (-5) = 8 )
        // Positive - Zero     = abs_sum ( 3 - 0 = 3)
        // Negative - Positive = -1 * abs_sum ( (-3) - 5 = -8 where abs_sum is 3 + 5 = 8)
        // Negative - Zero     = -1 * abs_sum ( -3 - 0 = -3; where abs_sum is 3)
        // Zero     - Negative = abs_sum ( 0 - (-3) = 3; where abs_sum is 3)
        // Zero     - Positive = -1 * abs_sum ( 0 - 3 = -3; where abs_sum is 3)

        let a_sign = a.sign.target;
        let b_sign = b.sign.target;

        let result_if_same_sign = BigIntTarget {
            abs: abs_diff.abs,
            // Revert the sign for Negative - Negative case
            sign: SignTarget::new_unsafe(self.mul(a_sign, abs_diff.sign.target)),
        };

        // 1 - a*a
        let one_minus_a_sign_sqr = self.arithmetic(F::NEG_ONE, F::ONE, a_sign, a_sign, one);
        // a - (1 - a*a)(b)
        let sign = self.arithmetic(F::NEG_ONE, F::ONE, one_minus_a_sign_sqr, b_sign, a_sign);

        let result_if_opposite_sign = BigIntTarget {
            abs: abs_sum,
            sign: SignTarget::new_unsafe(sign),
        };

        let result =
            self.select_bigint(is_same_sign, &result_if_same_sign, &result_if_opposite_sign);

        BigIntTarget {
            abs: self.trim_biguint(&result.abs, num_limbs),
            sign: result.sign,
        }
    }

    fn add_bigint_non_carry(
        &mut self,
        a: &BigIntTarget,
        b: &BigIntTarget,
        num_limbs: usize,
    ) -> BigIntTarget {
        let neg_one = self.neg_one();

        let neg_b = BigIntTarget {
            abs: b.abs.clone(),
            sign: SignTarget::new_unsafe(self.mul(neg_one, b.sign.target)),
        };

        // a - (-b) = a + b
        self.sub_bigint_non_carry(a, &neg_b, num_limbs)
    }

    fn mul_bigint_with_biguint_non_carry(
        &mut self,
        a: &BigIntTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigIntTarget {
        let abs = self.mul_biguint_non_carry(&a.abs, b, num_limbs);
        let is_b_zero = self.is_zero_biguint(b);
        let zero = self.zero();
        let sign = SignTarget::new_unsafe(self.select(is_b_zero, zero, a.sign.target));
        BigIntTarget { abs, sign }
    }

    fn mul_bigint_by_bool(&mut self, a: &BigIntTarget, b: BoolTarget) -> BigIntTarget {
        let abs = self.mul_biguint_by_bool(&a.abs, b);
        let sign = SignTarget::new_unsafe(self.mul(a.sign.target, b.target));
        BigIntTarget { abs, sign }
    }

    fn sub_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigIntTarget {
        let num_limbs = a.abs.num_limbs().max(b.abs.num_limbs()) + 1;
        self.sub_bigint_non_carry(a, b, num_limbs)
    }

    fn select_bigint(
        &mut self,
        condition: BoolTarget,
        a: &BigIntTarget,
        b: &BigIntTarget,
    ) -> BigIntTarget {
        let abs = self.select_biguint(condition, &a.abs, &b.abs);
        let sign = SignTarget::new_unsafe(self.select(condition, a.sign.target, b.sign.target));
        BigIntTarget { abs, sign }
    }

    fn conditional_assert_zero_bigint(&mut self, is_enabled: BoolTarget, a: &BigIntTarget) {
        self.conditional_assert_zero_biguint(is_enabled, &a.abs);
        self.conditional_assert_zero(is_enabled, a.sign.target);
    }

    fn signed_target_to_bigint(&mut self, target: SignedTarget) -> BigIntTarget {
        let (abs, sign) = self.abs(target);
        BigIntTarget {
            abs: self.target_to_biguint(abs),
            sign,
        }
    }

    /// Unsafe because it does not enforce target < 2^POSITIVE_THRESHOLD_BIT
    /// for SignedTarget. Caller must make sure target is in valid range.
    fn bigint_to_signed_target_unsafe(&mut self, target: &BigIntTarget) -> SignedTarget {
        let positive_result = self.biguint_to_target_safe(&target.abs);
        SignedTarget::new_unsafe(self.mul(positive_result, target.sign.target))
    }

    fn cmp_bigint(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> SignTarget {
        let zero = self.zero();
        let one = self.one();
        let neg_one = self.neg_one();

        let a_is_positive = self.is_sign_positive(a.sign);
        let b_is_positive = self.is_sign_positive(b.sign);

        let a_is_negative = self.is_sign_negative(a.sign);
        let b_is_negative = self.is_sign_negative(b.sign);

        // Positive - Positive -> a_abs_cmp_b_abs
        let mut result = self.cmp_biguint(&a.abs, &b.abs);

        // Positive - Zero     -> 1
        // Positive - Negative -> 1
        let b_not_pos = self.not(b_is_positive);
        let a_pos_b_not_pos = self.and(a_is_positive, b_not_pos);
        result = SignTarget::new_unsafe(self.select(a_pos_b_not_pos, one, result.target));

        // Zero - Positive -> -1
        // Zero - Zero     -> 0
        // Zero - Negative -> 1
        let a_is_zero = self.is_equal(a.sign.target, zero);
        let b_sign_flipped = self.mul(neg_one, b.sign.target);
        result = SignTarget::new_unsafe(self.select(a_is_zero, b_sign_flipped, result.target));

        // Negative - Positive -> -1
        // Negative - Zero     -> -1
        let b_not_neg = self.not(b_is_negative);
        let a_neg_b_not_neg = self.and(a_is_negative, b_not_neg);
        result = SignTarget::new_unsafe(self.select(a_neg_b_not_neg, neg_one, result.target));

        // Negative - Negative -> ^a_abs_cmp_b_abs
        let a_neg_b_neg = self.and(a_is_negative, b_is_negative);
        let a_cmp_b_flipped = self.mul(neg_one, result.target);
        SignTarget::new_unsafe(self.select(a_neg_b_neg, a_cmp_b_flipped, result.target))
    }

    fn biguint_to_bigint(&mut self, target: &BigUintTarget) -> BigIntTarget {
        let is_zero = self.is_zero_biguint(target);
        let one = self.one();
        BigIntTarget {
            abs: target.clone(),
            sign: SignTarget::new_unsafe(self.sub(one, is_zero.target)),
        }
    }

    fn random_access_bigint(
        &mut self,
        access_index: Target,
        v: &[BigIntTarget],
        limb_count: usize,
    ) -> BigIntTarget {
        BigIntTarget {
            abs: self.random_access_biguint(
                access_index,
                v.iter().map(|x| x.abs.clone()).collect(),
                limb_count,
            ),
            sign: SignTarget::new_unsafe(
                self.random_access(access_index, v.iter().map(|x| x.sign.target).collect()),
            ),
        }
    }

    fn bigint_vector_diff(&mut self, a: &BigIntTarget, b: &BigIntTarget) -> BigIntTarget {
        let mut limbs = vec![];
        a.abs
            .limbs
            .iter()
            .zip_eq(b.abs.limbs.iter())
            .for_each(|(&x, &y)| {
                limbs.push(U32Target(self.sub(x.0, y.0)));
            });

        BigIntTarget {
            abs: BigUintTarget { limbs },
            sign: SignTarget::new_unsafe(self.sub(a.sign.target, b.sign.target)),
        }
    }

    fn bigint_vector_sum(
        &mut self,
        cond: BoolTarget,
        a: &BigIntTarget,
        b: &BigIntTarget,
    ) -> BigIntTarget {
        let mut limbs = vec![];
        a.abs
            .limbs
            .iter()
            .zip_eq(b.abs.limbs.iter())
            .for_each(|(&x, &y)| {
                limbs.push(U32Target(self.mul_add(cond.target, x.0, y.0)));
            });

        BigIntTarget {
            abs: BigUintTarget { limbs },
            sign: SignTarget::new_unsafe(self.mul_add(cond.target, a.sign.target, b.sign.target)),
        }
    }

    fn trim_bigint(&mut self, a: &BigIntTarget, final_num_limbs: usize) -> BigIntTarget {
        BigIntTarget {
            abs: self.trim_biguint(&a.abs, final_num_limbs),
            sign: a.sign,
        }
    }

    fn neg_bigint(&mut self, a: &BigIntTarget) -> BigIntTarget {
        let neg_sign = SignTarget::new_unsafe(self.neg(a.sign.target));
        BigIntTarget {
            abs: a.abs.clone(),
            sign: neg_sign,
        }
    }

    fn euclidian_div_by_biguint(
        &mut self,
        a: &BigIntTarget,
        b: &BigUintTarget,
        num_limbs: usize,
    ) -> BigIntTarget {
        let (a_div, a_rem) = self.div_rem_biguint(&a.abs, b);
        let is_negative = self.is_sign_negative(a.sign);
        let rem_is_zero = self.is_zero_biguint(&a_rem);
        let need_to_add_one = self.and_not(is_negative, rem_is_zero);
        let big_need_to_add_one = self.target_to_biguint_single_limb_unsafe(need_to_add_one.target);
        let abs_result = self.add_biguint_non_carry(&a_div, &big_need_to_add_one, num_limbs);

        let zero = self.zero();
        let is_a_zero = self.is_zero_biguint(&abs_result);
        BigIntTarget {
            abs: abs_result,
            sign: SignTarget::new_unsafe(self.select(is_a_zero, zero, a.sign.target)),
        }
    }
}

pub trait WitnessBigInt<F: PrimeField64>: Witness<F> {
    fn get_bigint_target(&self, target: BigIntTarget) -> BigInt;
    fn set_bigint_target(&mut self, target: &BigIntTarget, value: &BigInt) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> WitnessBigInt<F> for T {
    fn get_bigint_target(&self, target: BigIntTarget) -> BigInt {
        let abs = self.get_biguint_target(target.abs);
        let sign = self.get_target(target.sign.target);

        let sign = if sign == F::ONE {
            Sign::Plus
        } else if sign == F::NEG_ONE {
            Sign::Minus
        } else if sign == F::ZERO {
            Sign::NoSign
        } else {
            panic!("Sign expected to be -1, 0 or 1 but got {:?}", sign)
        };
        BigInt::from_biguint(sign, abs)
    }

    fn set_bigint_target(&mut self, target: &BigIntTarget, value: &BigInt) -> Result<()> {
        let abs = value.abs().to_biguint().unwrap();
        let sign = match value.sign() {
            Sign::Plus => 1,
            Sign::Minus => -1,
            Sign::NoSign => 0,
        };

        self.set_biguint_target(&target.abs, &abs)?;
        self.set_target(target.sign.target, F::from_noncanonical_i64(sign))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    #[allow(unused_imports)]
    use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
    use num::FromPrimitive;
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    #[allow(unused_imports)]
    use crate::circuit_logger::CircuitBuilderLogging;
    use crate::types::config::{
        BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U128_LIMBS, C, CIRCUIT_CONFIG, F,
    };

    #[test]
    fn test_cmp_bigint() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let neg_one = builder.neg_one();
        let zero = builder.zero();
        let one = builder.one();

        let pos_big = builder.add_virtual_bigint_target_unsafe(BIG_U128_LIMBS);
        let pos_small = builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS);
        let neg_big = builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS);
        let neg_small = builder.add_virtual_bigint_target_unsafe(BIG_U128_LIMBS);
        let zero_bigint = builder.add_virtual_bigint_target_unsafe(BIG_U128_LIMBS);

        let pos_big_pos_small = builder.cmp_bigint(&pos_big, &pos_small);
        builder.connect(pos_big_pos_small.target, one);
        let pos_small_pos_big = builder.cmp_bigint(&pos_small, &pos_big);
        builder.connect(pos_small_pos_big.target, neg_one);
        let pos_big_pos_big = builder.cmp_bigint(&pos_big, &pos_big);
        builder.connect(pos_big_pos_big.target, zero);
        let pos_neg = builder.cmp_bigint(&pos_big, &neg_big);
        builder.connect(pos_neg.target, one);
        let pos_zero = builder.cmp_bigint(&pos_big, &zero_bigint);
        builder.connect(pos_zero.target, one);
        let zero_pos = builder.cmp_bigint(&zero_bigint, &pos_big);
        builder.connect(zero_pos.target, neg_one);
        let zero_zero = builder.cmp_bigint(&zero_bigint, &zero_bigint);
        builder.connect(zero_zero.target, zero);
        let zero_neg = builder.cmp_bigint(&zero_bigint, &neg_big);
        builder.connect(zero_neg.target, one);
        let neg_big_neg_small = builder.cmp_bigint(&neg_big, &neg_small);
        builder.connect(neg_big_neg_small.target, one);
        let neg_small_neg_big = builder.cmp_bigint(&neg_small, &neg_big);
        builder.connect(neg_small_neg_big.target, neg_one);
        let neg_big_neg_big = builder.cmp_bigint(&neg_big, &neg_big);
        builder.connect(neg_big_neg_big.target, zero);
        let neg_pos = builder.cmp_bigint(&neg_big, &pos_big);
        builder.connect(neg_pos.target, neg_one);
        let neg_zero = builder.cmp_bigint(&neg_big, &zero_bigint);
        builder.connect(neg_zero.target, neg_one);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();

        pw.set_bigint_target(&pos_big, &BigInt::from_i64(100i64).unwrap())?;
        pw.set_bigint_target(&pos_small, &BigInt::from_i64(10i64).unwrap())?;
        pw.set_bigint_target(&neg_big, &BigInt::from_i64(-10i64).unwrap())?;
        pw.set_bigint_target(&neg_small, &BigInt::from_i64(-100i64).unwrap())?;
        pw.set_bigint_target(&zero_bigint, &BigInt::from_i64(0i64).unwrap())?;

        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn test_abs_diff() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let mut pw = PartialWitness::<F>::new();

        let a0 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b0 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c00 = builder.abs_diff(&a0, &b0);
        let c01 = builder.abs_diff(&b0, &a0);

        let a0_value = 100i64;
        let b0_value = 90i64;

        pw.set_bigint_target(&a0, &BigInt::from_i64(a0_value).unwrap())?;
        pw.set_bigint_target(&b0, &BigInt::from_i64(b0_value).unwrap())?;
        pw.set_bigint_target(&c00, &BigInt::from_i64(10i64).unwrap())?;
        pw.set_bigint_target(&c01, &BigInt::from_i64(-10i64).unwrap())?;

        let a1 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b1 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c10 = builder.abs_diff(&a1, &b1);
        let c11 = builder.abs_diff(&b1, &a1);

        let a1_value = -100i64;
        let b1_value = -90i64;

        pw.set_bigint_target(&a1, &BigInt::from_i64(a1_value).unwrap())?;
        pw.set_bigint_target(&b1, &BigInt::from_i64(b1_value).unwrap())?;
        pw.set_bigint_target(&c10, &BigInt::from_i64(10i64).unwrap())?;
        pw.set_bigint_target(&c11, &BigInt::from_i64(-10i64).unwrap())?;

        let a2 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b2 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c20 = builder.abs_diff(&a2, &b2);
        let c21 = builder.abs_diff(&b2, &a2);

        let a2_value = 100i64;
        let b2_value = -90i64;

        pw.set_bigint_target(&a2, &BigInt::from_i64(a2_value).unwrap())?;
        pw.set_bigint_target(&b2, &BigInt::from_i64(b2_value).unwrap())?;
        pw.set_bigint_target(&c20, &BigInt::from_i64(10i64).unwrap())?;
        pw.set_bigint_target(&c21, &BigInt::from_i64(-10i64).unwrap())?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn test_sub_bigint() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let mut pw = PartialWitness::<F>::new();

        let a0 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b0 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c00 = builder.sub_bigint_non_carry(&a0, &b0, BIG_U64_LIMBS);
        let c01 = builder.sub_bigint_non_carry(&b0, &a0, BIG_U64_LIMBS);

        let a0_value = 100i64;
        let b0_value = 90i64;

        pw.set_bigint_target(&a0, &BigInt::from_i64(a0_value).unwrap())?;
        pw.set_bigint_target(&b0, &BigInt::from_i64(b0_value).unwrap())?;
        pw.set_bigint_target(&c00, &BigInt::from_i64(10i64).unwrap())?;
        pw.set_bigint_target(&c01, &BigInt::from_i64(-10i64).unwrap())?;

        let a1 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b1 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c10 = builder.sub_bigint_non_carry(&a1, &b1, BIG_U64_LIMBS);
        let c11 = builder.sub_bigint_non_carry(&b1, &a1, BIG_U64_LIMBS);

        let a1_value = -100i64;
        let b1_value = -90i64;

        pw.set_bigint_target(&a1, &BigInt::from_i64(a1_value).unwrap())?;
        pw.set_bigint_target(&b1, &BigInt::from_i64(b1_value).unwrap())?;
        pw.set_bigint_target(&c10, &BigInt::from_i64(-10i64).unwrap())?;
        pw.set_bigint_target(&c11, &BigInt::from_i64(10i64).unwrap())?;

        let a2 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b2 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c20 = builder.sub_bigint_non_carry(&a2, &b2, BIG_U64_LIMBS);
        let c21 = builder.sub_bigint_non_carry(&b2, &a2, BIG_U64_LIMBS);

        let a2_value = 100i64;
        let b2_value = -90i64;

        pw.set_bigint_target(&a2, &BigInt::from_i64(a2_value).unwrap())?;
        pw.set_bigint_target(&b2, &BigInt::from_i64(b2_value).unwrap())?;
        pw.set_bigint_target(&c20, &BigInt::from_i64(190i64).unwrap())?;
        pw.set_bigint_target(&c21, &BigInt::from_i64(-190i64).unwrap())?;

        let a3 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b3 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c30 = builder.sub_bigint_non_carry(&a3, &b3, BIG_U64_LIMBS);

        let a3_value = 100i64;
        let b3_value = 100i64;

        pw.set_bigint_target(&a3, &BigInt::from_i64(a3_value).unwrap())?;
        pw.set_bigint_target(&b3, &BigInt::from_i64(b3_value).unwrap())?;
        pw.set_bigint_target(&c30, &BigInt::from_i64(0i64).unwrap())?;

        let a4 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b4 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c40 = builder.sub_bigint_non_carry(&a4, &b4, BIG_U64_LIMBS);

        let a4_value = -100i64;
        let b4_value = -100i64;

        pw.set_bigint_target(&a4, &BigInt::from_i64(a4_value).unwrap())?;
        pw.set_bigint_target(&b4, &BigInt::from_i64(b4_value).unwrap())?;
        pw.set_bigint_target(&c40, &BigInt::from_i64(0i64).unwrap())?;

        let a5 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b5 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c50 = builder.sub_bigint_non_carry(&a5, &b5, BIG_U64_LIMBS);
        let c51 = builder.sub_bigint_non_carry(&b5, &a5, BIG_U64_LIMBS);

        let a5_value = 100i64;
        let b5_value = 0i64;

        pw.set_bigint_target(&a5, &BigInt::from_i64(a5_value).unwrap())?;
        pw.set_bigint_target(&b5, &BigInt::from_i64(b5_value).unwrap())?;
        pw.set_bigint_target(&c50, &BigInt::from_i64(100i64).unwrap())?;
        pw.set_bigint_target(&c51, &BigInt::from_i64(-100i64).unwrap())?;

        let a6 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let b6 = builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS);
        let c60 = builder.sub_bigint_non_carry(&a6, &b6, BIG_U64_LIMBS);
        let c61 = builder.sub_bigint_non_carry(&b6, &a6, BIG_U64_LIMBS);

        let a6_value = -100i64;
        let b6_value = 0i64;

        pw.set_bigint_target(&a6, &BigInt::from_i64(a6_value).unwrap())?;
        pw.set_bigint_target(&b6, &BigInt::from_i64(b6_value).unwrap())?;
        pw.set_bigint_target(&c60, &BigInt::from_i64(-100i64).unwrap())?;
        pw.set_bigint_target(&c61, &BigInt::from_i64(100i64).unwrap())?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }
}
