// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::constants::*;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
#[serde(default)]
pub struct Market<F>
where
    F: RichField,
{
    #[serde(rename = "i")]
    pub market_index: u16, // Index is used only as hint to verify merkle proofs. It isn't included in the leaf hash

    #[serde(rename = "s", default)]
    pub status: u8,

    #[serde(rename = "mt")]
    pub market_type: u8,

    #[serde(rename = "ba")]
    pub base_asset_id: u16,

    #[serde(rename = "qa")]
    pub quote_asset_id: u16,

    #[serde(rename = "a")]
    pub ask_nonce: i64,

    #[serde(rename = "b")]
    pub bid_nonce: i64,

    #[serde(rename = "t")]
    pub taker_fee: u32,

    #[serde(rename = "m")]
    pub maker_fee: u32,

    #[serde(rename = "l")]
    pub liquidation_fee: u32,

    #[serde(rename = "sem")]
    pub size_extension_multiplier: i64,

    #[serde(rename = "qem")]
    pub quote_extension_multiplier: i64,

    #[serde(rename = "toc", default)]
    pub total_order_count: i64, // 48 bits

    #[serde(rename = "mba")]
    pub min_base_amount: u64,

    #[serde(rename = "ma")]
    pub min_quote_amount: u64,

    #[serde(rename = "oql")]
    pub order_quote_limit: i64,

    #[serde(rename = "r")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub order_book_root: HashOut<F>,
}

impl<F: RichField + Default> Default for Market<F> {
    fn default() -> Self {
        Self {
            market_index: 255,
            market_type: 0,
            base_asset_id: 0,
            quote_asset_id: 0,
            total_order_count: 0,
            size_extension_multiplier: 0,
            quote_extension_multiplier: 0,
            ask_nonce: 0,
            bid_nonce: 0,
            taker_fee: 0,
            maker_fee: 0,
            liquidation_fee: 0,
            min_base_amount: 0,
            min_quote_amount: 0,
            order_quote_limit: 0,
            order_book_root: HashOut::<F>::ZERO,
            status: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketTarget {
    pub market_index: Target, //  8 bits. Index is used only as hint to verify merkle proofs. It isn't included in the leaf hash

    pub status: Target,
    pub market_type: Target,
    pub base_asset_id: Target,              // 6 bits
    pub quote_asset_id: Target,             // 6 bits
    pub ask_nonce: Target,                  // 48 bits
    pub bid_nonce: Target,                  // 48 bits
    pub taker_fee: Target,                  // 20 bits
    pub maker_fee: Target,                  // 20 bits
    pub liquidation_fee: Target,            // 20 bits
    pub size_extension_multiplier: Target,  // 48 bits
    pub quote_extension_multiplier: Target, // 48 bits
    pub total_order_count: Target,          // 48 bits
    pub min_base_amount: Target,            // 48 bits
    pub min_quote_amount: Target,           // 48 bits
    pub order_quote_limit: Target,          // 48 bits
    pub order_book_root: HashOutTarget,

    // Helper
    pub perps_market_index: Target,
}

impl Default for MarketTarget {
    fn default() -> Self {
        Self {
            market_index: Target::default(),
            market_type: Target::default(),
            status: Target::default(),
            base_asset_id: Target::default(),
            quote_asset_id: Target::default(),
            ask_nonce: Target::default(),
            bid_nonce: Target::default(),
            taker_fee: Target::default(),
            total_order_count: Target::default(),
            maker_fee: Target::default(),
            liquidation_fee: Target::default(),
            size_extension_multiplier: Target::default(),
            quote_extension_multiplier: Target::default(),
            min_base_amount: Target::default(),
            min_quote_amount: Target::default(),
            order_quote_limit: Target::default(),
            order_book_root: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },
            perps_market_index: Target::default(),
        }
    }
}

impl MarketTarget {
    pub fn new(builder: &mut Builder) -> Self {
        let market_type = builder.add_virtual_target();
        let market_index = builder.add_virtual_target();

        let max_perps_market_index = builder.constant_usize(MAX_PERPS_MARKET_INDEX);
        let is_perps = builder.is_lte(market_index, max_perps_market_index, MARKET_INDEX_BITS);

        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);

        Self {
            market_index,
            market_type,
            status: builder.add_virtual_target(),
            base_asset_id: builder.add_virtual_target(),
            quote_asset_id: builder.add_virtual_target(),
            ask_nonce: builder.add_virtual_target(),
            bid_nonce: builder.add_virtual_target(),
            taker_fee: builder.add_virtual_target(),
            maker_fee: builder.add_virtual_target(),
            total_order_count: builder.add_virtual_target(),
            liquidation_fee: builder.add_virtual_target(),
            size_extension_multiplier: builder.add_virtual_target(),
            quote_extension_multiplier: builder.add_virtual_target(),
            min_base_amount: builder.add_virtual_target(),
            min_quote_amount: builder.add_virtual_target(),
            order_quote_limit: builder.add_virtual_target(),
            order_book_root: builder.add_virtual_hash(),

            perps_market_index: builder.select(is_perps, market_index, nil_market_index),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.market_index, &format!("{} market_index", tag));
        builder.println(
            self.perps_market_index,
            &format!("{} perps_market_index", tag),
        );
        builder.println(self.market_type, &format!("{} market_type", tag));
        builder.println(self.status, &format!("{} status", tag));
        builder.println(self.base_asset_id, &format!("{} base_asset_id", tag));
        builder.println(self.quote_asset_id, &format!("{} quote_asset_id", tag));
        builder.println(self.ask_nonce, &format!("{} ask_nonce", tag));
        builder.println(
            self.total_order_count,
            &format!("{} -- total_order_count", tag),
        );
        builder.println(self.bid_nonce, &format!("{} bid_nonce", tag));
        builder.println(self.taker_fee, &format!("{} taker_fee", tag));
        builder.println(self.maker_fee, &format!("{} maker_fee", tag));
        builder.println(self.liquidation_fee, &format!("{} liquidation_fee", tag));
        builder.println(
            self.size_extension_multiplier,
            &format!("{} size_extension_multiplier", tag),
        );
        builder.println(
            self.quote_extension_multiplier,
            &format!("{} quote_extension_multiplier", tag),
        );
        builder.println(self.min_base_amount, &format!("{} min_base_amount", tag));
        builder.println(self.min_quote_amount, &format!("{} min_quote_amount", tag));
        builder.println(
            self.order_quote_limit,
            &format!("{} order_quote_limit", tag),
        );
        builder.println_hash_out(&self.order_book_root, &format!("{} order_book_root", tag));
    }

    pub fn empty(
        builder: &mut Builder,
        market_index: Target,
        perps_market_index: Target,
        order_book_root: HashOutTarget,
    ) -> Self {
        Self {
            market_index,

            market_type: builder.zero(),
            status: builder.zero(),
            base_asset_id: builder.zero(),
            quote_asset_id: builder.zero(),
            total_order_count: builder.zero(),
            ask_nonce: builder.zero(),
            bid_nonce: builder.zero(),
            taker_fee: builder.zero(),
            maker_fee: builder.zero(),
            liquidation_fee: builder.zero(),
            size_extension_multiplier: builder.zero(),
            quote_extension_multiplier: builder.zero(),
            min_base_amount: builder.zero(),
            min_quote_amount: builder.zero(),
            order_quote_limit: builder.zero(),
            order_book_root,

            perps_market_index,
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // Adding following fields does not overflow Goldilocks, as long as
        // these fields are guaranteed by business logic to fit these sizes.
        let added = builder.add_many([
            self.ask_nonce,                  // 48 bits
            self.bid_nonce,                  // 48 bits
            self.taker_fee,                  // 32 bits
            self.maker_fee,                  // 32 bits
            self.liquidation_fee,            // 32 bits
            self.min_base_amount,            // 48 bits
            self.min_quote_amount,           // 48 bits
            self.status,                     // 8 bits
            self.order_quote_limit,          // 48 bits
            self.total_order_count,          // 48 bits
            self.market_type,                // 8 bits
            self.base_asset_id,              // 16 bits
            self.quote_asset_id,             // 16 bits
            self.size_extension_multiplier,  // 56 bits
            self.quote_extension_multiplier, // 56 bits
        ]);
        builder.is_zero(added)
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            self.market_type,
            self.status,
            self.base_asset_id,
            self.quote_asset_id,
            self.ask_nonce,
            self.bid_nonce,
            self.taker_fee,
            self.maker_fee,
            self.liquidation_fee,
            self.min_base_amount,
            self.min_quote_amount,
            self.order_quote_limit,
            self.total_order_count,
            self.size_extension_multiplier,
            self.quote_extension_multiplier,
            self.order_book_root.elements[0],
            self.order_book_root.elements[1],
            self.order_book_root.elements[2],
            self.order_book_root.elements[3],
        ]);

        let empty_hash = builder.zero_hash_out();
        let is_empty = self.is_empty(builder);

        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }
}

pub trait MarketTargetWitness<F: PrimeField64 + RichField> {
    fn set_market_target(&mut self, t: &MarketTarget, mi: &Market<F>) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + RichField> MarketTargetWitness<F> for T {
    fn set_market_target(&mut self, a: &MarketTarget, b: &Market<F>) -> Result<()> {
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;

        self.set_target(a.market_type, F::from_canonical_u8(b.market_type))?;
        self.set_target(a.status, F::from_canonical_u8(b.status))?;
        self.set_target(a.base_asset_id, F::from_canonical_u16(b.base_asset_id))?;
        self.set_target(a.quote_asset_id, F::from_canonical_u16(b.quote_asset_id))?;
        self.set_target(
            a.total_order_count,
            F::from_canonical_i64(b.total_order_count),
        )?;
        self.set_target(a.ask_nonce, F::from_canonical_i64(b.ask_nonce))?;
        self.set_target(a.bid_nonce, F::from_canonical_i64(b.bid_nonce))?;

        self.set_target(a.taker_fee, F::from_canonical_u32(b.taker_fee))?;
        self.set_target(a.maker_fee, F::from_canonical_u32(b.maker_fee))?;
        self.set_target(a.liquidation_fee, F::from_canonical_u32(b.liquidation_fee))?;

        self.set_target(
            a.size_extension_multiplier,
            F::from_canonical_i64(b.size_extension_multiplier),
        )?;
        self.set_target(
            a.quote_extension_multiplier,
            F::from_canonical_i64(b.quote_extension_multiplier),
        )?;

        self.set_target(a.min_base_amount, F::from_canonical_u64(b.min_base_amount))?;
        self.set_target(
            a.min_quote_amount,
            F::from_canonical_u64(b.min_quote_amount),
        )?;
        self.set_target(
            a.order_quote_limit,
            F::from_canonical_i64(b.order_quote_limit),
        )?;

        self.set_hash_target(a.order_book_root, b.order_book_root)?;

        Ok(())
    }
}

pub fn select_market(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &MarketTarget,
    b: &MarketTarget,
) -> MarketTarget {
    MarketTarget {
        market_index: builder.select(flag, a.market_index, b.market_index),
        perps_market_index: builder.select(flag, a.perps_market_index, b.perps_market_index),
        status: builder.select(flag, a.status, b.status),
        market_type: builder.select(flag, a.market_type, b.market_type),
        base_asset_id: builder.select(flag, a.base_asset_id, b.base_asset_id),
        quote_asset_id: builder.select(flag, a.quote_asset_id, b.quote_asset_id),
        total_order_count: builder.select(flag, a.total_order_count, b.total_order_count),
        ask_nonce: builder.select(flag, a.ask_nonce, b.ask_nonce),
        bid_nonce: builder.select(flag, a.bid_nonce, b.bid_nonce),
        taker_fee: builder.select(flag, a.taker_fee, b.taker_fee),
        maker_fee: builder.select(flag, a.maker_fee, b.maker_fee),
        liquidation_fee: builder.select(flag, a.liquidation_fee, b.liquidation_fee),
        size_extension_multiplier: builder.select(
            flag,
            a.size_extension_multiplier,
            b.size_extension_multiplier,
        ),
        quote_extension_multiplier: builder.select(
            flag,
            a.quote_extension_multiplier,
            b.quote_extension_multiplier,
        ),
        min_base_amount: builder.select(flag, a.min_base_amount, b.min_base_amount),
        min_quote_amount: builder.select(flag, a.min_quote_amount, b.min_quote_amount),
        order_quote_limit: builder.select(flag, a.order_quote_limit, b.order_quote_limit),
        order_book_root: builder.select_hash(flag, &a.order_book_root, &b.order_book_root),
    }
}

pub fn ensure_spot_market_index(builder: &mut Builder, is_enabled: BoolTarget, index: Target) {
    let min_spot_market_index = builder.constant_usize(MIN_SPOT_MARKET_INDEX);
    let max_spot_market_index = builder.constant_usize(MAX_SPOT_MARKET_INDEX);
    let invalid_range_assertions = [
        builder.is_gt(index, max_spot_market_index, 16),
        builder.is_lt(index, min_spot_market_index, 16),
    ];
    let is_invalid_market_index = builder.multi_or(&invalid_range_assertions);
    builder.conditional_assert_false(is_enabled, is_invalid_market_index);
}
