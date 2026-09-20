// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::types::config::Builder;
use crate::types::constants::{MARKET_INDEX_BITS, MARKET_SLOT_STATUS_BITS};

pub const MARKET_DELTA_LIMBS: usize = 4;

/// Market lifecycle event published in the blob, packed as `slot | status << 12 | price << 14`, `pmi`,
/// `settlement_cap`, `quote_extension_multiplier`.
/// Status is Expired / Active / InSettlement; price is the default (refund) price while Active and the
/// settlement price once InSettlement, so an exit taken mid-settlement values the remaining positions correctly.
/// Cap and quote extension multiplier are fixed at creation and scale binary options payouts.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct MarketDeltaLeaf {
    #[serde(rename = "mi")]
    pub market_index: u16,
    #[serde(rename = "pmi")]
    pub public_market_index: i64,
    #[serde(rename = "s")]
    pub status: u8,
    #[serde(rename = "p")]
    pub price: u32,
    #[serde(rename = "sc")]
    pub settlement_cap: u32,
    #[serde(rename = "qem")]
    pub quote_extension_multiplier: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct MarketDeltaLeafTarget {
    pub market_index: Target,
    pub public_market_index: Target,
    pub status: Target,
    pub price: Target,
    pub settlement_cap: Target,
    pub quote_extension_multiplier: Target,
}

impl MarketDeltaLeafTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            market_index: builder.add_virtual_target(),
            public_market_index: builder.add_virtual_target(),
            status: builder.add_virtual_target(),
            price: builder.add_virtual_target(),
            settlement_cap: builder.add_virtual_target(),
            quote_extension_multiplier: builder.add_virtual_target(),
        }
    }

    pub fn pack(&self, builder: &mut Builder) -> [Target; MARKET_DELTA_LIMBS] {
        let status_shifter = builder.constant_u64(1 << MARKET_INDEX_BITS);
        let price_shifter =
            builder.constant_u64(1 << (MARKET_INDEX_BITS + MARKET_SLOT_STATUS_BITS));
        let limb = builder.mul_add(status_shifter, self.status, self.market_index);
        [
            builder.mul_add(price_shifter, self.price, limb),
            self.public_market_index,
            self.settlement_cap,
            self.quote_extension_multiplier,
        ]
    }
}

pub trait MarketDeltaLeafTargetWitness<F: RichField> {
    fn set_market_delta_leaf_target(
        &mut self,
        a: &MarketDeltaLeafTarget,
        b: &MarketDeltaLeaf,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: RichField> MarketDeltaLeafTargetWitness<F> for T {
    fn set_market_delta_leaf_target(
        &mut self,
        a: &MarketDeltaLeafTarget,
        b: &MarketDeltaLeaf,
    ) -> Result<()> {
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(
            a.public_market_index,
            F::from_noncanonical_i64(b.public_market_index),
        )?;
        self.set_target(a.status, F::from_canonical_u8(b.status))?;
        self.set_target(a.price, F::from_canonical_u32(b.price))?;
        self.set_target(a.settlement_cap, F::from_canonical_u32(b.settlement_cap))?;
        self.set_target(
            a.quote_extension_multiplier,
            F::from_noncanonical_i64(b.quote_extension_multiplier),
        )?;
        Ok(())
    }
}
