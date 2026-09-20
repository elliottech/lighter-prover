// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;

use crate::bigint::bigint::SignTarget;
use crate::builder::Builder;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::utils::CircuitBuilderUtils;

pub const POSITIVE_THRESHOLD_BIT: usize = 60;

/// A `Target` which is in range of `(-2^POSITIVE_THRESHOLD_BIT, 2^POSITIVE_THRESHOLD_BIT)` where negative values stored as
/// in the larger part of the field, ie. `(F::ORDER - 2^POSITIVE_THRESHOLD_BIT, F::ORDER)`
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[allow(clippy::manual_non_exhaustive)]
pub struct SignedTarget {
    pub target: Target,
    /// This private field is here to force all instantiations to go through `new_unsafe`.
    _private: (),
}

impl SignedTarget {
    pub fn new_unsafe(target: Target) -> SignedTarget {
        SignedTarget {
            target,
            _private: (),
        }
    }
}

pub trait CircuitBuilderSigned<F: RichField + Extendable<D>, const D: usize> {
    fn register_public_signed_target(&mut self, signed_target: SignedTarget);

    fn add_virtual_signed_target(&mut self) -> SignedTarget;

    fn connect_signed(&mut self, a: SignedTarget, b: SignedTarget);

    fn zero_signed(&mut self) -> SignedTarget;
    fn select_signed(&mut self, flag: BoolTarget, a: SignedTarget, b: SignedTarget)
    -> SignedTarget;

    fn is_non_negative(&mut self, signed_target: SignedTarget) -> BoolTarget;
    fn is_non_positive(&mut self, signed_target: SignedTarget) -> BoolTarget;
    fn is_negative(&mut self, signed_target: SignedTarget) -> BoolTarget;
    fn is_positive(&mut self, signed_target: SignedTarget) -> BoolTarget;
    fn sign(&mut self, signed_target: SignedTarget) -> SignTarget;

    /// Returns absolute value of the SignedTarget and its Sign
    fn abs(&mut self, signed_target: SignedTarget) -> (Target, SignTarget);

    fn add_signed(&mut self, a: SignedTarget, b: SignedTarget) -> SignedTarget;
    fn sub_signed(&mut self, a: SignedTarget, b: SignedTarget) -> SignedTarget;
    fn neg_signed(&mut self, a: SignedTarget) -> SignedTarget;

    fn range_check_signed(&mut self, signed_target: SignedTarget, bit_size: usize);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderSigned<F, D> for Builder<F, D> {
    fn register_public_signed_target(&mut self, signed_target: SignedTarget) {
        self.register_public_input(signed_target.target);
    }

    fn add_virtual_signed_target(&mut self) -> SignedTarget {
        SignedTarget::new_unsafe(self.add_virtual_target())
    }

    fn connect_signed(&mut self, a: SignedTarget, b: SignedTarget) {
        self.connect(a.target, b.target);
    }

    fn zero_signed(&mut self) -> SignedTarget {
        SignedTarget::new_unsafe(self.zero())
    }

    fn select_signed(
        &mut self,
        flag: BoolTarget,
        a: SignedTarget,
        b: SignedTarget,
    ) -> SignedTarget {
        SignedTarget::new_unsafe(self.select(flag, a.target, b.target))
    }

    fn sign(&mut self, signed_target: SignedTarget) -> SignTarget {
        let one = self.one();
        let two = self.two();

        /*
            (1 - is_zero) * (2 * is_lt - 1)

            is_zero is_lt    sign
            1       0        0
            1       1        0
            0       1        1
            0       0        -1
        */

        let threshold = self.constant(F::from_canonical_u64((1 << POSITIVE_THRESHOLD_BIT) - 1));
        let is_lte = self.is_lte(signed_target.target, threshold, 64);
        let rhs = self.mul_sub(is_lte.target, two, one);

        let is_not_zero = self.is_not_zero(signed_target.target);

        SignTarget::new_unsafe(self.mul(rhs, is_not_zero.target))
    }

    fn abs(&mut self, signed_target: SignedTarget) -> (Target, SignTarget) {
        let sign = self.sign(signed_target);
        let abs = self.mul(sign.target, signed_target.target);

        (abs, sign)
    }

    fn is_non_negative(&mut self, signed_target: SignedTarget) -> BoolTarget {
        let is_negative = self.is_negative(signed_target);
        self.not(is_negative) // this is free
    }

    fn is_non_positive(&mut self, signed_target: SignedTarget) -> BoolTarget {
        let is_positive = self.is_positive(signed_target);

        self.not(is_positive) // this is free
    }

    fn is_negative(&mut self, signed_target: SignedTarget) -> BoolTarget {
        let neg_one = self.neg_one();
        let sign = self.sign(signed_target);

        self.is_equal(sign.target, neg_one)
    }

    fn is_positive(&mut self, signed_target: SignedTarget) -> BoolTarget {
        let one = self.one();
        let sign = self.sign(signed_target);

        self.is_equal(sign.target, one)
    }

    fn add_signed(&mut self, a: SignedTarget, b: SignedTarget) -> SignedTarget {
        SignedTarget::new_unsafe(self.add(a.target, b.target))
    }

    fn sub_signed(&mut self, a: SignedTarget, b: SignedTarget) -> SignedTarget {
        SignedTarget::new_unsafe(self.sub(a.target, b.target))
    }

    fn neg_signed(&mut self, a: SignedTarget) -> SignedTarget {
        SignedTarget::new_unsafe(self.neg(a.target))
    }

    fn range_check_signed(&mut self, signed_target: SignedTarget, bit_size: usize) {
        let (abs, _) = self.abs(signed_target);

        self.register_range_check(abs, bit_size);
    }
}

pub trait WitnessSigned<F: PrimeField64>: Witness<F> {
    fn get_signed_target(&self, target: SignedTarget) -> i64;
    fn set_signed_target(&mut self, target: SignedTarget, value: i64) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> WitnessSigned<F> for T {
    fn get_signed_target(&self, target: SignedTarget) -> i64 {
        let target = self.get_target(target.target).to_canonical_u64();

        if target >= (1 << POSITIVE_THRESHOLD_BIT) {
            (target - F::ORDER) as i64
        } else if target == 0 {
            0
        } else {
            target as i64
        }
    }

    fn set_signed_target(&mut self, target: SignedTarget, value: i64) -> Result<()> {
        self.set_target(target.target, F::from_noncanonical_i64(value))
    }
}
