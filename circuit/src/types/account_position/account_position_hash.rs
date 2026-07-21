// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::Target;

use crate::types::account_position::AccountPositionTarget;
use crate::types::config::{BIG_U96_LIMBS, BIGU16_U64_LIMBS, Builder};
use crate::uint::u16::gadgets::arithmetic_u16::CircuitBuilderU16;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

impl AccountPositionTarget {
    pub fn append_position_hash_params(&self, builder: &mut Builder, elements: &mut Vec<Target>) {
        let mut limbs = self.allocated_margin.abs.limbs.clone();
        limbs.resize(BIG_U96_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }
        elements.extend_from_slice(&[
            self.allocated_margin.sign.target,
            self.margin_mode,
            self.entry_quote,
            self.initial_margin_fraction,
            self.total_order_count,
            self.total_position_tied_order_count,
            self.margin_set_flag,
        ]);
    }

    pub fn append_position_pub_data_hash_params(
        &self,
        builder: &mut Builder,
        elements: &mut Vec<Target>,
    ) {
        let mut limbs = self.last_funding_rate_prefix_sum.abs.limbs.clone();
        limbs.resize(BIGU16_U64_LIMBS, builder.zero_u16());
        for limb in limbs {
            elements.push(limb.0);
        }
        elements.push(self.last_funding_rate_prefix_sum.sign.target);

        let mut limbs = self.position.abs.limbs.clone();
        limbs.resize(BIGU16_U64_LIMBS, builder.zero_u16());
        for limb in limbs {
            elements.push(limb.0);
        }
        elements.push(self.position.sign.target);
    }
}
