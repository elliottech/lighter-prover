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

    #[serde(rename = "pmi")]
    pub public_market_index: i64,

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

    #[serde(rename = "sts", default)]
    pub start_timestamp: i64,

    #[serde(rename = "ets", default)]
    pub end_timestamp: i64,

    #[serde(rename = "oi", default)]
    pub open_interest: i64,

    #[serde(rename = "oil", default)]
    pub open_interest_limit: i64,

    #[serde(rename = "o", default)]
    pub outcome: u8,

    #[serde(rename = "moai", default)]
    pub market_operator_account_index: i64,

    #[serde(rename = "sc", default)]
    pub settlement_cap: u32,

    #[serde(rename = "sty", default)]
    pub settlement_type: u8,

    #[serde(rename = "sp", default)]
    pub settlement_price: u32,

    #[serde(rename = "dp", default)]
    pub default_price: u32,

    #[serde(rename = "fz", default)]
    pub is_frozen: u8,

    #[serde(rename = "r")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub order_book_root: HashOut<F>,
}

impl<F: RichField + Default> Default for Market<F> {
    fn default() -> Self {
        Self {
            market_index: 255,
            public_market_index: NIL_PUBLIC_MARKET_INDEX,
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
            start_timestamp: 0,
            end_timestamp: 0,
            open_interest: 0,
            open_interest_limit: 0,
            outcome: 0,
            market_operator_account_index: NIL_ACCOUNT_INDEX,
            settlement_cap: 0,
            settlement_type: 0,
            settlement_price: 0,
            default_price: 0,
            is_frozen: 0,
            order_book_root: HashOut::<F>::ZERO,
            status: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketTarget {
    pub market_index: Target, //  8 bits. Index is used only as hint to verify merkle proofs. It isn't included in the leaf hash

    pub public_market_index: Target, // 48 bits

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

    pub start_timestamp: Target,               // 48 bits
    pub end_timestamp: Target,                 // 48 bits
    pub open_interest: Target,                 // 56 bits
    pub open_interest_limit: Target,           // 56 bits
    pub outcome: Target,                       // 8 bits
    pub market_operator_account_index: Target, // 48 bits
    pub settlement_cap: Target,                // 32 bits
    pub settlement_type: Target,               // 8 bits
    pub settlement_price: Target,              // 32 bits
    pub default_price: Target,                 // 32 bits
    pub is_frozen: Target,                     // 1 bit

    pub order_book_root: HashOutTarget,

    // Helpers: the market index when the slot is in the perps / binary options range, NIL_MARKET_INDEX otherwise
    pub perps_market_index: Target,
    pub binary_options_market_index: Target,
}

impl Default for MarketTarget {
    fn default() -> Self {
        Self {
            market_index: Target::default(),
            public_market_index: Target::default(),
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
            start_timestamp: Target::default(),
            end_timestamp: Target::default(),
            open_interest: Target::default(),
            open_interest_limit: Target::default(),
            outcome: Target::default(),
            market_operator_account_index: Target::default(),
            settlement_cap: Target::default(),
            settlement_type: Target::default(),
            settlement_price: Target::default(),
            default_price: Target::default(),
            is_frozen: Target::default(),
            order_book_root: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },
            perps_market_index: Target::default(),
            binary_options_market_index: Target::default(),
        }
    }
}

impl MarketTarget {
    pub fn new(builder: &mut Builder) -> Self {
        let market_type = builder.add_virtual_target();
        let market_index = builder.add_virtual_target();

        let max_perps_market_index = builder.constant_usize(MAX_PERPS_MARKET_INDEX);
        let is_perps = builder.is_lte(market_index, max_perps_market_index, MARKET_INDEX_BITS);
        let is_binary_options = is_binary_options_market_index(builder, market_index);

        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);

        Self {
            market_index,
            public_market_index: builder.add_virtual_target(),
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
            start_timestamp: builder.add_virtual_target(),
            end_timestamp: builder.add_virtual_target(),
            open_interest: builder.add_virtual_target(),
            open_interest_limit: builder.add_virtual_target(),
            outcome: builder.add_virtual_target(),
            market_operator_account_index: builder.add_virtual_target(),
            settlement_cap: builder.add_virtual_target(),
            settlement_type: builder.add_virtual_target(),
            settlement_price: builder.add_virtual_target(),
            default_price: builder.add_virtual_target(),
            is_frozen: builder.add_virtual_target(),
            order_book_root: builder.add_virtual_hash(),

            perps_market_index: builder.select(is_perps, market_index, nil_market_index),
            binary_options_market_index: builder.select(
                is_binary_options,
                market_index,
                nil_market_index,
            ),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.market_index, &format!("{} market_index", tag));
        builder.println(
            self.public_market_index,
            &format!("{} public_market_index", tag),
        );
        builder.println(
            self.perps_market_index,
            &format!("{} perps_market_index", tag),
        );
        builder.println(
            self.binary_options_market_index,
            &format!("{} binary_options_market_index", tag),
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
        builder.println(self.start_timestamp, &format!("{} start_timestamp", tag));
        builder.println(self.end_timestamp, &format!("{} end_timestamp", tag));
        builder.println(self.open_interest, &format!("{} open_interest", tag));
        builder.println(
            self.open_interest_limit,
            &format!("{} open_interest_limit", tag),
        );
        builder.println(self.outcome, &format!("{} outcome", tag));
        builder.println(
            self.market_operator_account_index,
            &format!("{} market_operator_account_index", tag),
        );
        builder.println(self.settlement_cap, &format!("{} settlement_cap", tag));
        builder.println(self.settlement_type, &format!("{} settlement_type", tag));
        builder.println(self.settlement_price, &format!("{} settlement_price", tag));
        builder.println(self.default_price, &format!("{} default_price", tag));
        builder.println(self.is_frozen, &format!("{} is_frozen", tag));
        builder.println_hash_out(&self.order_book_root, &format!("{} order_book_root", tag));
    }

    pub fn empty(
        builder: &mut Builder,
        market_index: Target,
        perps_market_index: Target,
        binary_options_market_index: Target,
        order_book_root: HashOutTarget,
    ) -> Self {
        Self {
            market_index,

            public_market_index: builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64),
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
            start_timestamp: builder.zero(),
            end_timestamp: builder.zero(),
            open_interest: builder.zero(),
            open_interest_limit: builder.zero(),
            outcome: builder.zero(),
            market_operator_account_index: builder.constant_i64(NIL_ACCOUNT_INDEX),
            settlement_cap: builder.zero(),
            settlement_type: builder.zero(),
            settlement_price: builder.zero(),
            default_price: builder.zero(),
            is_frozen: builder.zero(),
            order_book_root,

            perps_market_index,
            binary_options_market_index,
        }
    }

    /// The market a settled market leaves behind in its slot: empty except for the ask/bid
    /// nonces, which the next market created on the slot resumes so order indexes never repeat,
    /// and the slot's market operator, who can create the next market on the slot.
    pub fn settled(
        builder: &mut Builder,
        market: &MarketTarget,
        order_book_root: HashOutTarget,
    ) -> Self {
        Self {
            ask_nonce: market.ask_nonce,
            bid_nonce: market.bid_nonce,
            market_operator_account_index: market.market_operator_account_index,
            ..Self::empty(
                builder,
                market.market_index,
                market.perps_market_index,
                market.binary_options_market_index,
                order_book_root,
            )
        }
    }

    /// Nonces a market created on this slot starts from: the slot's nonces if it hosted a
    /// market before, the initial nonces otherwise.
    pub fn next_market_nonces(&self, builder: &mut Builder) -> (Target, Target) {
        let nonce_sum = builder.add(self.ask_nonce, self.bid_nonce);
        let never_hosted_market = builder.is_zero(nonce_sum);
        let first_ask_nonce = builder.constant_i64(FIRST_ASK_NONCE);
        let first_bid_nonce = builder.constant_i64(FIRST_BID_NONCE);
        (
            builder.select(never_hosted_market, first_ask_nonce, self.ask_nonce),
            builder.select(never_hosted_market, first_bid_nonce, self.bid_nonce),
        )
    }

    /// A slot never hosted a market and hashes to the nil leaf only while no market operator is
    /// assigned to it either; a slot with an operator is a non-empty leaf.
    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let nil_public_market_index = builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64);
        let nil_market_operator = builder.constant_i64(NIL_ACCOUNT_INDEX);
        let assertions = [
            builder.is_equal(self.public_market_index, nil_public_market_index),
            builder.is_zero(self.market_type),
            builder.is_zero(self.status),
            builder.is_zero(self.base_asset_id),
            builder.is_zero(self.quote_asset_id),
            builder.is_zero(self.ask_nonce),
            builder.is_zero(self.bid_nonce),
            builder.is_zero(self.taker_fee),
            builder.is_zero(self.maker_fee),
            builder.is_zero(self.liquidation_fee),
            builder.is_zero(self.min_base_amount),
            builder.is_zero(self.min_quote_amount),
            builder.is_zero(self.order_quote_limit),
            builder.is_zero(self.total_order_count),
            builder.is_zero(self.size_extension_multiplier),
            builder.is_zero(self.quote_extension_multiplier),
            builder.is_zero(self.start_timestamp),
            builder.is_zero(self.end_timestamp),
            builder.is_zero(self.open_interest),
            builder.is_zero(self.open_interest_limit),
            builder.is_zero(self.outcome),
            builder.is_equal(self.market_operator_account_index, nil_market_operator),
            builder.is_zero(self.settlement_cap),
            builder.is_zero(self.settlement_type),
            builder.is_zero(self.settlement_price),
            builder.is_zero(self.default_price),
            builder.is_zero(self.is_frozen),
        ];
        builder.multi_and(&assertions)
    }

    /// Price published in the market delta: the default (refund) price while Active, the settlement
    /// price while InSettlement and zero otherwise.
    pub fn pub_data_price(&self, builder: &mut Builder) -> Target {
        let is_active = builder.is_equal_constant(self.status, MARKET_STATUS_ACTIVE as u64);
        let is_in_settlement =
            builder.is_equal_constant(self.status, MARKET_STATUS_IN_SETTLEMENT as u64);

        let zero = builder.zero();
        let price = builder.select(is_in_settlement, self.settlement_price, zero);
        builder.select(is_active, self.default_price, price)
    }

    /// Leaf of the market pub data tree. Only binary options slots publish market data; the leaf
    /// is empty while every published field is zero, like every perps and spot slot.
    pub fn pub_data_hash(&self, builder: &mut Builder) -> HashOutTarget {
        let market_type = builder.constant_usize(MARKET_TYPE_BINARY_OPTIONS as usize);
        let price = self.pub_data_price(builder);
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            market_type,
            self.status,
            price,
            self.settlement_cap,
            self.quote_extension_multiplier,
        ]);

        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);
        let is_binary_options =
            builder.is_not_equal(self.binary_options_market_index, nil_market_index);
        let fields_sum = builder.add_many([
            self.status,
            price,
            self.settlement_cap,
            self.quote_extension_multiplier,
        ]);
        let is_empty = builder.is_zero(fields_sum);
        let is_non_empty = builder.and_not(is_binary_options, is_empty);

        let empty_hash = builder.zero_hash_out();
        builder.select_hash(is_non_empty, &non_empty_hash, &empty_hash)
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
            self.public_market_index,
            self.start_timestamp,
            self.end_timestamp,
            self.open_interest,
            self.open_interest_limit,
            self.outcome,
            self.market_operator_account_index,
            self.settlement_cap,
            self.settlement_type,
            self.settlement_price,
            self.default_price,
            self.is_frozen,
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
        self.set_target(
            a.public_market_index,
            F::from_canonical_i64(b.public_market_index),
        )?;

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

        self.set_target(a.start_timestamp, F::from_canonical_i64(b.start_timestamp))?;
        self.set_target(a.end_timestamp, F::from_canonical_i64(b.end_timestamp))?;
        self.set_target(a.open_interest, F::from_canonical_i64(b.open_interest))?;
        self.set_target(
            a.open_interest_limit,
            F::from_canonical_i64(b.open_interest_limit),
        )?;
        self.set_target(a.outcome, F::from_canonical_u8(b.outcome))?;
        self.set_target(
            a.market_operator_account_index,
            F::from_canonical_i64(b.market_operator_account_index),
        )?;
        self.set_target(a.settlement_cap, F::from_canonical_u32(b.settlement_cap))?;
        self.set_target(a.settlement_type, F::from_canonical_u8(b.settlement_type))?;
        self.set_target(
            a.settlement_price,
            F::from_canonical_u32(b.settlement_price),
        )?;
        self.set_target(a.default_price, F::from_canonical_u32(b.default_price))?;
        self.set_target(a.is_frozen, F::from_canonical_u8(b.is_frozen))?;

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
        public_market_index: builder.select(flag, a.public_market_index, b.public_market_index),
        perps_market_index: builder.select(flag, a.perps_market_index, b.perps_market_index),
        binary_options_market_index: builder.select(
            flag,
            a.binary_options_market_index,
            b.binary_options_market_index,
        ),
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
        start_timestamp: builder.select(flag, a.start_timestamp, b.start_timestamp),
        end_timestamp: builder.select(flag, a.end_timestamp, b.end_timestamp),
        open_interest: builder.select(flag, a.open_interest, b.open_interest),
        open_interest_limit: builder.select(flag, a.open_interest_limit, b.open_interest_limit),
        outcome: builder.select(flag, a.outcome, b.outcome),
        market_operator_account_index: builder.select(
            flag,
            a.market_operator_account_index,
            b.market_operator_account_index,
        ),
        settlement_cap: builder.select(flag, a.settlement_cap, b.settlement_cap),
        settlement_type: builder.select(flag, a.settlement_type, b.settlement_type),
        settlement_price: builder.select(flag, a.settlement_price, b.settlement_price),
        default_price: builder.select(flag, a.default_price, b.default_price),
        is_frozen: builder.select(flag, a.is_frozen, b.is_frozen),
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

fn is_binary_options_market_index(builder: &mut Builder, index: Target) -> BoolTarget {
    let min_index = builder.constant_usize(MIN_BINARY_OPTIONS_MARKET_INDEX);
    let max_index = builder.constant_usize(MAX_BINARY_OPTIONS_MARKET_INDEX);
    let invalid_range_assertions = [
        builder.is_gt(index, max_index, 16),
        builder.is_lt(index, min_index, 16),
    ];
    let is_invalid_market_index = builder.multi_or(&invalid_range_assertions);
    builder.not(is_invalid_market_index)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    fn settled_market_circuit(
        ask_nonce: i64,
        bid_nonce: i64,
        market_operator_account_index: i64,
        expect_empty: bool,
    ) -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let market = MarketTarget::new(&mut builder);
        let empty_order_book_root = builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT);
        let settled = MarketTarget::settled(&mut builder, &market, empty_order_book_root);

        // Settled market keeps only the nonces and the market operator; everything else is cleared.
        builder.connect(settled.ask_nonce, market.ask_nonce);
        builder.connect(settled.bid_nonce, market.bid_nonce);
        builder.connect(
            settled.market_operator_account_index,
            market.market_operator_account_index,
        );
        builder.assert_zero(settled.status);
        builder.assert_zero(settled.total_order_count);
        let nil_public_market_index = builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64);
        builder.connect(settled.public_market_index, nil_public_market_index);

        // It hashes to the nil leaf only when the slot never hosted a market.
        let is_empty = settled.is_empty(&mut builder);
        let hash = settled.hash(&mut builder);
        let is_zero_hash = builder.is_zero_hash_out(&hash);
        if expect_empty {
            builder.assert_true(is_empty);
            builder.assert_true(is_zero_hash);
        } else {
            builder.assert_false(is_empty);
            builder.assert_false(is_zero_hash);
        }

        // A market created on the slot resumes the nonces, or starts fresh on a slot that never
        // hosted a market (even if an operator is already assigned to it).
        let never_hosted_market = ask_nonce == 0 && bid_nonce == 0;
        let (next_ask_nonce, next_bid_nonce) = settled.next_market_nonces(&mut builder);
        let expected_ask_nonce = if never_hosted_market {
            builder.constant_i64(FIRST_ASK_NONCE)
        } else {
            market.ask_nonce
        };
        let expected_bid_nonce = if never_hosted_market {
            builder.constant_i64(FIRST_BID_NONCE)
        } else {
            market.bid_nonce
        };
        builder.connect(next_ask_nonce, expected_ask_nonce);
        builder.connect(next_bid_nonce, expected_bid_nonce);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();
        pw.set_market_target(
            &market,
            &Market::<F> {
                market_index: 7,
                public_market_index: 4096,
                market_type: MARKET_TYPE_PERPS as u8,
                status: MARKET_STATUS_ACTIVE,
                base_asset_id: 3,
                quote_asset_id: 1,
                ask_nonce,
                bid_nonce,
                market_operator_account_index,
                taker_fee: 100,
                maker_fee: 50,
                order_book_root: EMPTY_ORDER_BOOK_TREE_ROOT,
                ..Market::<F>::default()
            },
        )?;

        data.verify(data.prove(pw)?)
    }

    #[test]
    fn settled_market_keeps_nonces_and_is_not_empty() -> Result<()> {
        settled_market_circuit(FIRST_ASK_NONCE + 41, FIRST_BID_NONCE - 17, 12, false)
    }

    #[test]
    fn settled_market_with_operator_is_not_empty() -> Result<()> {
        settled_market_circuit(0, 0, 12, false)
    }

    #[test]
    fn settled_market_with_untouched_nonces_and_no_operator_is_empty() -> Result<()> {
        settled_market_circuit(0, 0, NIL_ACCOUNT_INDEX, true)
    }
}
