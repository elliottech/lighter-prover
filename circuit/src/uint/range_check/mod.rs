// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

extern crate paste;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use crate::builder::Builder;

pub mod range_check_2_bit;
pub mod range_check_byte_decom;
pub use range_check_2_bit::{RangeCheck2BitGate, RangeCheck2BitGenerator};
pub use range_check_byte_decom::{RangeCheckGate, RangeCheckGenerator};

impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    #[track_caller]
    pub fn register_range_check(&mut self, val: Target, bit_size: usize) {
        if self.use_2bit_range_check {
            self.register_range_check_2_bit(val, bit_size);
        } else {
            self.register_range_check_byte_decom(val, bit_size);
        }
    }

    pub fn perform_registered_range_checks(&mut self) {
        if self.use_2bit_range_check {
            self.perform_registered_range_checks_2_bit();
        } else {
            self.perform_registered_range_checks_byte_decom();
        }
    }
}
