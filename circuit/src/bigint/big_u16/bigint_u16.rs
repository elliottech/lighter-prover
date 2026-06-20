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

use super::biguint_u16::{BigUintU16Target, CircuitBuilderBiguint16};
use crate::bigint::big_u16::biguint_u16::WitnessBigUintU16;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::builder::Builder;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::uint::u16::gadgets::arithmetic_u16::U16Target;
use crate::utils::CircuitBuilderUtils;

/// Wrapper around BigUintU16Target to support signed operations
///
/// `sign`: `1`  &rarr; For positive values
///
/// `sign`: `-1` &rarr; For negative values
///
/// `sign`: `0` &rarr; For zero value
#[derive(Clone, Debug, Default)]
pub struct BigIntU16Target {
    pub abs: BigUintU16Target,
    pub sign: SignTarget,
}

impl BigIntU16Target {
    /// Builds a BigIntU16Target from a vector of sign and limbs.
    /// Assumes element order as `register_public_input_bigint_u16`
    /// Does not range-check elements
    pub fn from_vec(limbs: &[Target]) -> Self {
        assert!(limbs.len() > 1);

        let sign = SignTarget::new_unsafe(limbs[0]);
        let abs = BigUintU16Target {
            limbs: limbs.iter().skip(1).map(|&x| U16Target(x)).collect(),
        };
        BigIntU16Target { abs, sign }
    }
}

pub trait CircuitBuilderBigIntU16<F: RichField + Extendable<D>, const D: usize> {
    fn register_public_input_bigint_u16(&mut self, value: &BigIntU16Target);

    #[must_use]
    fn add_virtual_bigint_u16_target_safe(&mut self, num_limbs: usize) -> BigIntU16Target;
    #[must_use]
    fn add_virtual_bigint_u16_target_unsafe(&mut self, num_limbs: usize) -> BigIntU16Target;

    fn connect_bigint_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target);

    #[must_use]
    fn is_equal_bigint_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BoolTarget;
    #[must_use]
    fn is_not_equal_bigint_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BoolTarget;
    #[must_use]
    fn is_zero_bigint_u16(&mut self, a: &BigIntU16Target) -> BoolTarget;
    #[must_use]
    fn select_bigint_u16(
        &mut self,
        condition: BoolTarget,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> BigIntU16Target;

    fn zero_bigint_u16(&mut self) -> BigIntU16Target;

    fn bigint_u16_to_bigint(&mut self, target: &BigIntU16Target) -> BigIntTarget;
    fn signed_target_to_bigint_u16(
        &mut self,
        target: SignedTarget,
        num_limbs: usize,
    ) -> BigIntU16Target;

    /// Returns a.abs - b.abs as BigIntU16Target
    fn abs_diff_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BigIntU16Target;

    fn abs_sum_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BigUintU16Target;
    /// Returns a - b, using non-carry additions when necessary
    fn sub_bigint_u16_non_carry(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
        num_limbs: usize,
    ) -> BigIntU16Target;
    /// Return a + b, using non-carry additions when necessary
    fn add_bigint_u16_non_carry(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
        num_limbs: usize,
    ) -> BigIntU16Target;

    /// Returns difference as BigIntU16 between two BigIntU16 numbers via calculating distance between limbs individually
    /// Returned number may not be a valid BigIntU16, for example it may overflow or have invalid sign
    fn bigint_u16_vector_diff(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> BigIntU16Target;
    /// Apply the difference between two BigIntU16 numbers calculated from [`CircuitBuilderBigIntU16::bigint_u16_vector_diff`] to another BigIntU16
    /// Usage example:
    ///   diff = x - y, z = w + diff = w + x - y. If w == y, then z == x
    fn bigint_u16_vector_sum(
        &mut self,
        cond: BoolTarget,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> BigIntU16Target;

    fn random_access_bigint_u16(
        &mut self,
        access_index: Target,
        v: Vec<BigIntU16Target>,
        limb_count: usize,
    ) -> BigIntU16Target;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBigIntU16<F, D> for Builder<F, D> {
    fn register_public_input_bigint_u16(&mut self, value: &BigIntU16Target) {
        self.register_public_input(value.sign.target);
        self.register_public_input_biguint_u16(&value.abs);
    }

    fn add_virtual_bigint_u16_target_safe(&mut self, num_limbs: usize) -> BigIntU16Target {
        let abs = self.add_virtual_biguint_u16_target_safe(num_limbs);
        let sign = self.add_virtual_sign_target_safe();
        let abs_is_zero = self.is_zero_biguint_u16(&abs);
        let sign_is_zero = self.is_zero(sign.target);
        self.connect(abs_is_zero.target, sign_is_zero.target);
        BigIntU16Target { abs, sign }
    }

    fn add_virtual_bigint_u16_target_unsafe(&mut self, num_limbs: usize) -> BigIntU16Target {
        BigIntU16Target {
            abs: self.add_virtual_biguint_u16_target_unsafe(num_limbs),
            sign: self.add_virtual_sign_target_unsafe(),
        }
    }

    fn connect_bigint_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) {
        self.connect_biguint_u16(&a.abs, &b.abs);
        self.connect(a.sign.target, b.sign.target);
    }

    fn is_equal_bigint_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BoolTarget {
        let abs_eq = self.is_equal_biguint_u16(&a.abs, &b.abs);
        let sign_eq = self.is_equal(a.sign.target, b.sign.target);
        self.and(abs_eq, sign_eq)
    }

    fn is_not_equal_bigint_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BoolTarget {
        let is_equal = self.is_equal_bigint_u16(a, b);
        self.not(is_equal)
    }

    fn select_bigint_u16(
        &mut self,
        condition: BoolTarget,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> BigIntU16Target {
        BigIntU16Target {
            abs: self.select_biguint_u16(condition, &a.abs, &b.abs),
            sign: SignTarget::new_unsafe(self.select(condition, a.sign.target, b.sign.target)),
        }
    }

    fn zero_bigint_u16(&mut self) -> BigIntU16Target {
        let abs = self.zero_biguint_u16();
        let sign = SignTarget::new_unsafe(self.zero());
        BigIntU16Target { abs, sign }
    }

    fn abs_diff_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BigIntU16Target {
        let (diff_if_a_gt_b, borrow_a_b) = self.try_sub_biguint_u16(&a.abs, &b.abs);
        let (diff_if_a_lte_b, borrow_b_a) = self.try_sub_biguint_u16(&b.abs, &a.abs);

        BigIntU16Target {
            abs: self.select_biguint_u16(
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

    fn abs_sum_u16(&mut self, a: &BigIntU16Target, b: &BigIntU16Target) -> BigUintU16Target {
        self.add_biguint_u16(&a.abs, &b.abs)
    }

    fn sub_bigint_u16_non_carry(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
        num_limbs: usize,
    ) -> BigIntU16Target {
        let one = self.one();

        let abs_diff = self.abs_diff_u16(a, b);
        let abs_sum = self.abs_sum_u16(a, b);

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

        let result_if_same_sign = BigIntU16Target {
            abs: abs_diff.abs,
            // Revert the sign for Negative - Negative case
            sign: SignTarget::new_unsafe(self.mul(a_sign, abs_diff.sign.target)),
        };

        // 1 - a*a
        let one_minus_a_sign_sqr = self.arithmetic(F::NEG_ONE, F::ONE, a_sign, a_sign, one);
        // a - (1 - a*a)(b)
        let sign = self.arithmetic(F::NEG_ONE, F::ONE, one_minus_a_sign_sqr, b_sign, a_sign);

        let result_if_opposite_sign = BigIntU16Target {
            abs: abs_sum,
            sign: SignTarget::new_unsafe(sign),
        };

        let result =
            self.select_bigint_u16(is_same_sign, &result_if_same_sign, &result_if_opposite_sign);

        BigIntU16Target {
            abs: self.trim_biguint_u16(&result.abs, num_limbs),
            sign: result.sign,
        }
    }

    fn add_bigint_u16_non_carry(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
        num_limbs: usize,
    ) -> BigIntU16Target {
        let neg_b = BigIntU16Target {
            abs: b.abs.clone(),
            sign: SignTarget::new_unsafe(self.neg(b.sign.target)),
        };
        // a - (-b) = a + b
        self.sub_bigint_u16_non_carry(a, &neg_b, num_limbs)
    }

    fn bigint_u16_to_bigint(&mut self, target: &BigIntU16Target) -> BigIntTarget {
        BigIntTarget {
            abs: self.biguint_u16_to_biguint(&target.abs),
            sign: target.sign,
        }
    }

    fn is_zero_bigint_u16(&mut self, a: &BigIntU16Target) -> BoolTarget {
        let is_sign_zero = self.is_zero(a.sign.target);
        let is_abs_zero = self.is_zero_biguint_u16(&a.abs);
        self.and(is_sign_zero, is_abs_zero)
    }

    fn signed_target_to_bigint_u16(
        &mut self,
        target: SignedTarget,
        num_limbs: usize,
    ) -> BigIntU16Target {
        let (abs, sign) = self.abs(target);
        BigIntU16Target {
            abs: self.target_to_biguint_u16(abs, num_limbs),
            sign,
        }
    }

    fn bigint_u16_vector_diff(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> BigIntU16Target {
        let mut limbs = vec![];
        a.abs
            .limbs
            .iter()
            .zip_eq(b.abs.limbs.iter())
            .for_each(|(&x, &y)| {
                limbs.push(U16Target(self.sub(x.0, y.0)));
            });

        BigIntU16Target {
            abs: BigUintU16Target { limbs },
            sign: SignTarget::new_unsafe(self.sub(a.sign.target, b.sign.target)),
        }
    }

    fn bigint_u16_vector_sum(
        &mut self,
        cond: BoolTarget,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> BigIntU16Target {
        let mut limbs = vec![];
        a.abs
            .limbs
            .iter()
            .zip_eq(b.abs.limbs.iter())
            .for_each(|(&x, &y)| {
                limbs.push(U16Target(self.mul_add(cond.target, x.0, y.0)));
            });

        BigIntU16Target {
            abs: BigUintU16Target { limbs },
            sign: SignTarget::new_unsafe(self.mul_add(cond.target, a.sign.target, b.sign.target)),
        }
    }

    fn random_access_bigint_u16(
        &mut self,
        access_index: Target,
        v: Vec<BigIntU16Target>,
        limb_count: usize,
    ) -> BigIntU16Target {
        BigIntU16Target {
            abs: self.random_access_biguint_u16(
                access_index,
                v.iter().map(|x| x.abs.clone()).collect(),
                limb_count,
            ),
            sign: SignTarget::new_unsafe(
                self.random_access(access_index, v.iter().map(|x| x.sign.target).collect()),
            ),
        }
    }
}

pub trait WitnessBigInt16<F: PrimeField64>: Witness<F> {
    fn get_bigint_u16_target(&self, target: BigIntU16Target) -> BigInt;
    fn set_bigint_u16_target(&mut self, target: &BigIntU16Target, value: &BigInt) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> WitnessBigInt16<F> for T {
    fn get_bigint_u16_target(&self, target: BigIntU16Target) -> BigInt {
        let abs = self.get_biguint_u16_target(target.abs);
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

    fn set_bigint_u16_target(&mut self, target: &BigIntU16Target, value: &BigInt) -> Result<()> {
        let abs = value.abs().to_biguint().unwrap();
        let sign = match value.sign() {
            Sign::Plus => 1,
            Sign::Minus => -1,
            Sign::NoSign => 0,
        };

        self.set_biguint_u16_target(&target.abs, &abs)?;
        self.set_target(target.sign.target, F::from_noncanonical_i64(sign))?;

        Ok(())
    }
}
