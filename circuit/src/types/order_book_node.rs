// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::{HashOut, HashOutTarget, NUM_HASH_OUT_ELTS};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use super::constants::ORDER_BOOK_MERKLE_LEVELS;
use crate::deserializers;
use crate::hash_utils::CircuitBuilderHashUtils;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(bound = "")]
pub struct OrderBookNode<F: Field> {
    #[serde(rename = "h", default)]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub sibling_child_hash: HashOut<F>,

    #[serde(rename = "ab", default)]
    pub ask_base_sum: i64, // 63 bits

    #[serde(rename = "aq", default)]
    pub ask_quote_sum: i64, // 63 bits

    #[serde(rename = "bb", default)]
    pub bid_base_sum: i64, // 63 bits

    #[serde(rename = "bq", default)]
    pub bid_quote_sum: i64, // 63 bits
}

impl<F: Field> OrderBookNode<F> {
    pub fn empty() -> Self {
        OrderBookNode {
            sibling_child_hash: HashOut::ZERO,
            ask_base_sum: 0,
            ask_quote_sum: 0,
            bid_base_sum: 0,
            bid_quote_sum: 0,
        }
    }

    pub fn internal_hash(&self) -> HashOut<F> {
        HashOut {
            elements: [
                F::from_noncanonical_i64(self.ask_base_sum),
                F::from_noncanonical_i64(self.ask_quote_sum),
                F::from_noncanonical_i64(self.bid_base_sum),
                F::from_noncanonical_i64(self.bid_quote_sum),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrderBookNodeTarget {
    pub sibling_child_hash: HashOutTarget,
    pub ask_base_sum: Target,  // 63 bits
    pub ask_quote_sum: Target, // 63 bits
    pub bid_base_sum: Target,  // 63 bits
    pub bid_quote_sum: Target, // 63 bits
}

impl Default for OrderBookNodeTarget {
    fn default() -> Self {
        OrderBookNodeTarget {
            sibling_child_hash: HashOutTarget::from([Target::default(); NUM_HASH_OUT_ELTS]),
            ask_base_sum: Target::default(),
            ask_quote_sum: Target::default(),
            bid_base_sum: Target::default(),
            bid_quote_sum: Target::default(),
        }
    }
}

impl OrderBookNodeTarget {
    pub fn new(builder: &mut Builder) -> Self {
        OrderBookNodeTarget {
            sibling_child_hash: builder.add_virtual_hash(),
            ask_base_sum: builder.add_virtual_target(),
            ask_quote_sum: builder.add_virtual_target(),
            bid_base_sum: builder.add_virtual_target(),
            bid_quote_sum: builder.add_virtual_target(),
        }
    }
    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            sibling_child_hash: builder.zero_hash_out(),
            ask_base_sum: builder.zero(),
            ask_quote_sum: builder.zero(),
            bid_base_sum: builder.zero(),
            bid_quote_sum: builder.zero(),
        }
    }

    pub fn internal_hash(&self) -> HashOutTarget {
        HashOutTarget {
            elements: [
                self.ask_base_sum,
                self.ask_quote_sum,
                self.bid_base_sum,
                self.bid_quote_sum,
            ],
        }
    }
}

pub trait OrderBookNodeTargetWitness<F: PrimeField64> {
    fn set_order_book_node_target(
        &mut self,
        t: &OrderBookNodeTarget,
        mi: &OrderBookNode<F>,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> OrderBookNodeTargetWitness<F> for T {
    fn set_order_book_node_target(
        &mut self,
        a: &OrderBookNodeTarget,
        b: &OrderBookNode<F>,
    ) -> Result<()> {
        self.set_hash_target(a.sibling_child_hash, b.sibling_child_hash)?;
        self.set_target(a.ask_base_sum, F::from_canonical_i64(b.ask_base_sum))?;
        self.set_target(a.ask_quote_sum, F::from_canonical_i64(b.ask_quote_sum))?;
        self.set_target(a.bid_base_sum, F::from_canonical_i64(b.bid_base_sum))?;
        self.set_target(a.bid_quote_sum, F::from_canonical_i64(b.bid_quote_sum))?;

        Ok(())
    }
}

pub fn select_order_book_path(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    b: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
) -> [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS] {
    let mut res = a.clone();
    for i in 0..ORDER_BOOK_MERKLE_LEVELS {
        res[i] = select_order_book_node(builder, flag, &a[i], &b[i]);
    }
    res
}

fn select_order_book_node(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &OrderBookNodeTarget,
    b: &OrderBookNodeTarget,
) -> OrderBookNodeTarget {
    OrderBookNodeTarget {
        sibling_child_hash: builder.select_hash(flag, &a.sibling_child_hash, &b.sibling_child_hash),
        ask_base_sum: builder.select(flag, a.ask_base_sum, b.ask_base_sum),
        ask_quote_sum: builder.select(flag, a.ask_quote_sum, b.ask_quote_sum),
        bid_base_sum: builder.select(flag, a.bid_base_sum, b.bid_base_sum),
        bid_quote_sum: builder.select(flag, a.bid_quote_sum, b.bid_quote_sum),
    }
}
