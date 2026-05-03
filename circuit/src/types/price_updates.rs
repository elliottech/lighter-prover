// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::deserializers;
use crate::types::constants::{MARGINED_ASSET_LIST_SIZE, POSITION_LIST_SIZE};

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
pub struct PriceUpdates {
    #[serde(rename = "i")]
    #[serde(deserialize_with = "deserializers::price_updates")]
    #[serde(default = "deserializers::default_price_updates")]
    pub index_price: [u32; POSITION_LIST_SIZE],

    #[serde(rename = "m")]
    #[serde(deserialize_with = "deserializers::price_updates")]
    #[serde(default = "deserializers::default_price_updates")]
    pub mark_price: [u32; POSITION_LIST_SIZE],

    #[serde(rename = "a")]
    #[serde(deserialize_with = "deserializers::asset_price_updates")]
    #[serde(default = "deserializers::default_asset_price_updates")]
    pub asset_index_price: [i64; MARGINED_ASSET_LIST_SIZE],
}

impl Default for PriceUpdates {
    fn default() -> Self {
        Self {
            index_price: [0; POSITION_LIST_SIZE],
            mark_price: [0; POSITION_LIST_SIZE],
            asset_index_price: [0; MARGINED_ASSET_LIST_SIZE],
        }
    }
}

#[derive(Debug)]
pub struct PriceUpdatesTarget {
    // 32 bits each
    pub index_price: [Target; POSITION_LIST_SIZE],
    pub mark_price: [Target; POSITION_LIST_SIZE],
    pub asset_index_price: [Target; MARGINED_ASSET_LIST_SIZE],
}

impl PriceUpdatesTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            index_price: builder
                .add_virtual_targets(POSITION_LIST_SIZE)
                .try_into()
                .unwrap(),
            mark_price: builder
                .add_virtual_targets(POSITION_LIST_SIZE)
                .try_into()
                .unwrap(),
            asset_index_price: builder
                .add_virtual_targets(MARGINED_ASSET_LIST_SIZE)
                .try_into()
                .unwrap(),
        }
    }
}

pub trait PriceUpdatesWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_price_updates_target(&mut self, t: &PriceUpdatesTarget, n: &PriceUpdates) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PriceUpdatesWitness<F> for T {
    fn set_price_updates_target(&mut self, t: &PriceUpdatesTarget, n: &PriceUpdates) -> Result<()> {
        for i in 0..POSITION_LIST_SIZE {
            self.set_target(t.index_price[i], F::from_canonical_u32(n.index_price[i]))?;
            self.set_target(t.mark_price[i], F::from_canonical_u32(n.mark_price[i]))?;
        }

        for i in 0..MARGINED_ASSET_LIST_SIZE {
            self.set_target(
                t.asset_index_price[i],
                F::from_canonical_u64(n.asset_index_price[i] as u64),
            )?;
        }

        Ok(())
    }
}
