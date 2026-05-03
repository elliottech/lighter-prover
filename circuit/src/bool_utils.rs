// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::BoolTarget;

use crate::builder::Builder;
use crate::utils::CircuitBuilderUtils;

pub trait CircuitBuilderBoolUtils<F: RichField + Extendable<D>, const D: usize> {
    fn assert_true(&mut self, t: BoolTarget);
    fn assert_false(&mut self, t: BoolTarget);
    fn conditional_assert_true(&mut self, is_enabled: BoolTarget, a: BoolTarget);
    fn conditional_assert_false(&mut self, is_enabled: BoolTarget, a: BoolTarget);
    fn conditional_assert_bool(&mut self, is_enabled: BoolTarget, a: BoolTarget);

    fn select_bool(&mut self, flag: BoolTarget, a: BoolTarget, b: BoolTarget) -> BoolTarget;

    fn multi_and(&mut self, targets: &[BoolTarget]) -> BoolTarget;
    fn multi_or(&mut self, targets: &[BoolTarget]) -> BoolTarget;

    /// Calculate `a & !b`.
    fn and_not(&mut self, a: BoolTarget, b: BoolTarget) -> BoolTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderBoolUtils<F, D> for Builder<F, D> {
    fn assert_true(&mut self, t: BoolTarget) {
        self.assert_one(t.target)
    }

    fn assert_false(&mut self, t: BoolTarget) {
        self.assert_zero(t.target)
    }

    fn conditional_assert_true(&mut self, is_enabled: BoolTarget, a: BoolTarget) {
        self.conditional_assert_one(is_enabled, a.target);
    }

    fn conditional_assert_false(&mut self, is_enabled: BoolTarget, a: BoolTarget) {
        self.conditional_assert_zero(is_enabled, a.target);
    }

    fn select_bool(&mut self, flag: BoolTarget, a: BoolTarget, b: BoolTarget) -> BoolTarget {
        BoolTarget::new_unsafe(self.select(flag, a.target, b.target))
    }

    fn multi_and(&mut self, targets: &[BoolTarget]) -> BoolTarget {
        let mut result = targets[0];
        for i in 1..targets.len() {
            result = self.and(result, targets[i]);
        }
        result
    }

    fn multi_or(&mut self, targets: &[BoolTarget]) -> BoolTarget {
        let mut result = targets[0];
        for i in 1..targets.len() {
            result = self.or(result, targets[i]);
        }
        result
    }

    /// a & !b
    fn and_not(&mut self, a: BoolTarget, b: BoolTarget) -> BoolTarget {
        // a(1 - b) = a - ab
        BoolTarget::new_unsafe(self.arithmetic(F::NEG_ONE, F::ONE, a.target, b.target, a.target))
    }

    fn conditional_assert_bool(&mut self, is_enabled: BoolTarget, a: BoolTarget) {
        let a_sq_minus_a = self.builder.mul_sub(a.target, a.target, a.target);
        self.conditional_assert_zero(is_enabled, a_sq_minus_a);
    }
}
