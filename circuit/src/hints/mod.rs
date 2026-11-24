// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::comparison::{CircuitBuilderSubtractiveComparison, cmp_bit_size_bucket};
use crate::utils::CircuitBuilderUtils;

pub trait CircuitBuilderHints<F: RichField + Extendable<D>, const D: usize> {
    /// Returns `(quotient, remainder)` of `dividend / divisor`, or `(0, 0)` when `divisor` is 0.
    ///
    /// `quotient * divisor + remainder == dividend` is checked in the field, i.e. modulo
    /// p = 2^64 - 2^32 + 1, not over the integers. Given `quotient <= dividend` and
    /// `remainder < divisor`, a prover can satisfy it with a wrong quotient if and only if
    /// `(dividend + 1) * (divisor - 1) >= p`. Only call this when the operands are bounded below
    /// that (dividend bits + divisor bits <= 63 is always enough); otherwise use `div_rem_biguint`.
    ///
    /// `divisor_bits` only sizes the remainder range check.
    fn div_rem(
        &mut self,
        dividend: Target,
        divisor: Target,
        divisor_bits: usize,
    ) -> (Target, Target);
    /// Same as `div_rem`, but the division is only checked when `is_enabled` is true. The same
    /// bound on the operands is required in that case.
    fn conditional_div_rem(
        &mut self,
        is_enabled: BoolTarget,
        dividend: Target,
        divisor: Target,
        divisor_bits: usize,
    ) -> (Target, Target);
    fn ceil_div(&mut self, dividend: Target, divisor: Target, divisor_bits: usize) -> Target;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderHints<F, D> for Builder<F, D> {
    fn div_rem(
        &mut self,
        dividend: Target,
        divisor: Target,
        divisor_bits: usize,
    ) -> (Target, Target) {
        let quotient = self.add_virtual_target();
        let remainder = self.add_virtual_target();
        self.add_simple_generator(DivRemHintGenerator {
            dividend,
            divisor,
            quotient,
            remainder,
            _phantom: PhantomData,
        });

        let is_div_by_zero = self.is_zero(divisor);
        self.conditional_assert_zero(is_div_by_zero, quotient);
        self.conditional_assert_zero(is_div_by_zero, remainder);

        let remainder_plus_quotient_times_divisor = self.mul_add(quotient, divisor, remainder);
        let is_not_div_by_zero = self.not(is_div_by_zero);
        self.conditional_assert_eq(
            is_not_div_by_zero,
            remainder_plus_quotient_times_divisor,
            dividend,
        );
        self.conditional_assert_lte(is_not_div_by_zero, quotient, dividend, 64);

        self.register_range_check(remainder, cmp_bit_size_bucket(divisor_bits));
        self.conditional_assert_lt(is_not_div_by_zero, remainder, divisor, 64);

        (quotient, remainder)
    }

    #[track_caller]
    fn conditional_div_rem(
        &mut self,
        is_enabled: BoolTarget,
        dividend: Target,
        divisor: Target,
        divisor_bits: usize,
    ) -> (Target, Target) {
        let quotient = self.add_virtual_target();
        let remainder = self.add_virtual_target();
        self.add_simple_generator(DivRemHintGenerator {
            dividend,
            divisor,
            quotient,
            remainder,
            _phantom: PhantomData,
        });

        let is_div_by_zero = self.is_zero(divisor);
        self.conditional_assert_zero(is_div_by_zero, quotient);
        self.conditional_assert_zero(is_div_by_zero, remainder);

        let remainder_plus_quotient_times_divisor = self.mul_add(quotient, divisor, remainder);
        let enabled_and_not_div_by_zero = self.and_not(is_enabled, is_div_by_zero);
        self.conditional_assert_eq(
            enabled_and_not_div_by_zero,
            remainder_plus_quotient_times_divisor,
            dividend,
        );
        self.conditional_assert_lte(enabled_and_not_div_by_zero, quotient, dividend, 64);

        // divisor can underflow when div_rem is called from irrelevant places during execution.
        let remainder_for_range_check = self.mul_bool(is_enabled, remainder);
        self.register_range_check(remainder_for_range_check, cmp_bit_size_bucket(divisor_bits));
        self.conditional_assert_lt(enabled_and_not_div_by_zero, remainder, divisor, 64);

        (quotient, remainder)
    }

    fn ceil_div(&mut self, dividend: Target, divisor: Target, divisor_bits: usize) -> Target {
        let (quotient, remainder) = self.div_rem(dividend, divisor, divisor_bits);
        let one = self.one();
        let is_rem_zero = self.is_zero(remainder);
        let addend = self.sub(one, is_rem_zero.target);
        self.add(quotient, addend)
    }
}

#[derive(Debug, Default)]
pub struct DivRemHintGenerator<F: RichField + Extendable<D>, const D: usize> {
    dividend: Target,
    divisor: Target,

    quotient: Target,
    remainder: Target,

    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for DivRemHintGenerator<F, D>
{
    fn id(&self) -> String {
        "DivRemHintGenerator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.dividend)?;
        dst.write_target(self.divisor)?;
        dst.write_target(self.quotient)?;
        dst.write_target(self.remainder)?;

        IoResult::Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let dividend = src.read_target()?;
        let divisor = src.read_target()?;
        let quotient = src.read_target()?;
        let remainder = src.read_target()?;

        IoResult::Ok(Self {
            dividend,
            divisor,
            quotient,
            remainder,
            _phantom: PhantomData,
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.dividend, self.divisor]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let dividend = witness.get_target(self.dividend).to_canonical_u64();
        let divisor = witness.get_target(self.divisor).to_canonical_u64();

        if divisor == 0 {
            out_buffer.set_target(self.quotient, F::ZERO)?;
            out_buffer.set_target(self.remainder, F::ZERO)?;
            return Ok(());
        }

        out_buffer.set_target(self.quotient, F::from_canonical_u64(dividend / divisor))?;
        out_buffer.set_target(self.remainder, F::from_canonical_u64(dividend % divisor))?;

        Ok(())
    }
}
