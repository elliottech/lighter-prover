// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use plonky2::plonk::config::Hasher;
use serde::Deserialize;

use super::config::{Builder, F};
use super::constants::*;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::byte::split::CircuitBuilderByteSplit;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::utils::{CircuitBuilderUtils, ceil_div_usize};

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Order {
    #[serde(rename = "kp")]
    pub key_price: i64, // 32 bits

    #[serde(rename = "kn")]
    pub key_nonce: i64, // 48 bits

    #[serde(rename = "ab")]
    pub ask_base_sum: i64,

    #[serde(rename = "aq")]
    pub ask_quote_sum: i64,

    #[serde(rename = "bb")]
    pub bid_base_sum: i64,

    #[serde(rename = "bq")]
    pub bid_quote_sum: i64,
}

impl Default for Order {
    fn default() -> Self {
        Order::empty(0, 0)
    }
}

impl Order {
    pub fn empty(price_index: i64, nonce_index: i64) -> Self {
        Order {
            key_price: price_index,
            key_nonce: nonce_index,

            ask_base_sum: 0,
            ask_quote_sum: 0,
            bid_base_sum: 0,
            bid_quote_sum: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ask_base_sum == 0
            && self.ask_quote_sum == 0
            && self.bid_base_sum == 0
            && self.bid_quote_sum == 0
    }

    pub fn hash(&self) -> HashOut<F> {
        if self.is_empty() {
            return HashOut::ZERO;
        }

        Poseidon2Hash::hash_no_pad(&[
            F::from_noncanonical_i64(self.ask_base_sum),
            F::from_noncanonical_i64(self.ask_quote_sum),
            F::from_noncanonical_i64(self.bid_base_sum),
            F::from_noncanonical_i64(self.bid_quote_sum),
        ])
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OrderTarget {
    pub price_index: Target,
    pub nonce_index: Target,

    pub ask_base_sum: Target,
    pub ask_quote_sum: Target,
    pub bid_base_sum: Target,
    pub bid_quote_sum: Target,
}

impl OrderTarget {
    pub fn new(builder: &mut Builder) -> Self {
        OrderTarget {
            price_index: builder.add_virtual_target(),
            nonce_index: builder.add_virtual_target(),

            ask_base_sum: builder.add_virtual_target(),
            ask_quote_sum: builder.add_virtual_target(),
            bid_base_sum: builder.add_virtual_target(),
            bid_quote_sum: builder.add_virtual_target(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.price_index, &format!("{} price_index", tag));
        builder.println(self.nonce_index, &format!("{} nonce_index", tag));
        builder.println(self.ask_base_sum, &format!("{} ask_base_sum", tag));
        builder.println(self.ask_quote_sum, &format!("{} ask_quote_sum", tag));
        builder.println(self.bid_base_sum, &format!("{} bid_base_sum", tag));
        builder.println(self.bid_quote_sum, &format!("{} bid_quote_sum", tag));
    }

    pub fn set_remaining_amount_conditional(
        &mut self,
        builder: &mut Builder,
        condition: BoolTarget,
        is_ask: BoolTarget,
        remaining_amount: Target,
    ) {
        let quote_sum = builder.mul(self.price_index, remaining_amount);

        let ask_condition = builder.and(condition, is_ask);
        self.ask_base_sum = builder.select(ask_condition, remaining_amount, self.ask_base_sum);
        self.ask_quote_sum = builder.select(ask_condition, quote_sum, self.ask_quote_sum);

        let is_bid = builder.not(is_ask);
        let bid_condition = builder.and(condition, is_bid);
        self.bid_base_sum = builder.select(bid_condition, remaining_amount, self.bid_base_sum);
        self.bid_quote_sum = builder.select(bid_condition, quote_sum, self.bid_quote_sum);
    }

    pub fn empty(builder: &mut Builder, price_index: Target, nonce_index: Target) -> Self {
        OrderTarget {
            price_index,
            nonce_index,

            ask_base_sum: builder.zero(),
            ask_quote_sum: builder.zero(),
            bid_base_sum: builder.zero(),
            bid_quote_sum: builder.zero(),
        }
    }

    pub fn is_ask(&self, builder: &mut Builder) -> BoolTarget {
        builder.is_zero(self.bid_base_sum)
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let is_ask_base_sum_empty = builder.is_zero(self.ask_base_sum);
        let is_ask_quote_sum_empty = builder.is_zero(self.ask_quote_sum);
        let is_bid_base_sum_empty = builder.is_zero(self.bid_base_sum);
        let is_bid_quote_sum_empty = builder.is_zero(self.bid_quote_sum);
        builder.multi_and(&[
            is_ask_base_sum_empty,
            is_ask_quote_sum_empty,
            is_bid_base_sum_empty,
            is_bid_quote_sum_empty,
        ])
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            self.ask_base_sum,
            self.ask_quote_sum,
            self.bid_base_sum,
            self.bid_quote_sum,
        ]);

        let empty_hash = builder.zero_hash_out();
        let is_empty = self.is_empty(builder);

        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    pub fn new_order_index(&self, builder: &mut Builder, market_index: Target) -> Target {
        let one = builder.one();
        let market_index_multiplier: Target =
            builder.constant(F::from_canonical_u64(1u64 << ORDER_NONCE_BITS));
        let market_index_plus_one = builder.add(market_index, one);

        builder.mul_add(
            market_index_plus_one,
            market_index_multiplier,
            self.nonce_index,
        )
    }
}

pub trait OrderTargetWitness<F: PrimeField64> {
    fn set_order_target(&mut self, t: &OrderTarget, mi: &Order) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> OrderTargetWitness<F> for T {
    fn set_order_target(&mut self, a: &OrderTarget, b: &Order) -> Result<()> {
        self.set_target(a.price_index, F::from_canonical_i64(b.key_price))?;
        self.set_target(a.nonce_index, F::from_canonical_i64(b.key_nonce))?;

        self.set_target(a.ask_base_sum, F::from_canonical_i64(b.ask_base_sum))?;
        self.set_target(a.ask_quote_sum, F::from_canonical_i64(b.ask_quote_sum))?;
        self.set_target(a.bid_base_sum, F::from_canonical_i64(b.bid_base_sum))?;
        self.set_target(a.bid_quote_sum, F::from_canonical_i64(b.bid_quote_sum))?;

        Ok(())
    }
}

pub fn select_order_target(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    a: &OrderTarget,
    b: &OrderTarget,
) -> OrderTarget {
    OrderTarget {
        price_index: builder.select(is_enabled, a.price_index, b.price_index),
        nonce_index: builder.select(is_enabled, a.nonce_index, b.nonce_index),

        ask_base_sum: builder.select(is_enabled, a.ask_base_sum, b.ask_base_sum),
        ask_quote_sum: builder.select(is_enabled, a.ask_quote_sum, b.ask_quote_sum),
        bid_base_sum: builder.select(is_enabled, a.bid_base_sum, b.bid_base_sum),
        bid_quote_sum: builder.select(is_enabled, a.bid_quote_sum, b.bid_quote_sum),
    }
}

pub fn get_market_index_and_order_nonce_from_order_index(
    builder: &mut Builder,
    order_index: Target,
) -> (Target, Target) {
    let one = builder.one();
    let bytes = builder.split_bytes(
        order_index,
        ceil_div_usize(ORDER_NONCE_BITS + MARKET_INDEX_BITS, 8),
    );

    let order_nonce = builder.le_sum_bytes(&bytes[0..ORDER_NONCE_BITS / 8]);
    let market_index_plus_one = builder.le_sum_bytes(&bytes[ORDER_NONCE_BITS / 8..]);
    let market_index = builder.sub(market_index_plus_one, one);

    (market_index, order_nonce)
}

pub fn get_order_index(builder: &mut Builder, market_index: Target, order_nonce: Target) -> Target {
    let one = builder.one();
    let market_index_lhs_multiplier =
        builder.constant(F::from_canonical_u64(1 << ORDER_NONCE_BITS));
    let market_index_plus_one = builder.add(market_index, one);
    builder.mul_add(
        market_index_plus_one,
        market_index_lhs_multiplier,
        order_nonce,
    )
}
