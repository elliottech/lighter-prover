// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigInt;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::{BIGU16_U64_LIMBS, Builder};
use crate::bigint::big_u16::bigint_u16::{
    BigIntU16Target, CircuitBuilderBigIntU16, WitnessBigInt16,
};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::constants::{MARKET_TYPE_BINARY_OPTIONS, NIL_PUBLIC_MARKET_INDEX};
use crate::utils::CircuitBuilderUtils;

/// A binary options position for a single binary options market slot (order book).
/// It is the leaf of the account market data tree, keyed by market index, and is
/// only part of the full account hash (never the pub data hash, delta trees or blob).
///
/// A position belongs to the market identified by `public_market_index`; a position of a
/// previous market hosted on the same slot (different public market index) is treated as empty.
#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "", default)]
pub struct BinaryOptionsPosition {
    #[serde(rename = "pmi", alias = "PublicMarketIndex")]
    pub public_market_index: i64,

    #[serde(rename = "s", alias = "Size")]
    pub size: i64, // Position size (+ is YES, - is NO)

    #[serde(rename = "eq", alias = "EntryQuote")]
    pub entry_quote: i64, // How much the user paid to be in the position, in USDC

    #[serde(rename = "toc", alias = "TotalOrderCount")]
    pub total_order_count: i64, // Resting orders of this account in the market
}

impl Default for BinaryOptionsPosition {
    fn default() -> Self {
        Self {
            public_market_index: NIL_PUBLIC_MARKET_INDEX,
            size: 0,
            entry_quote: 0,
            total_order_count: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BinaryOptionsPositionTarget {
    pub public_market_index: Target, // 48 bits
    pub size: BigIntU16Target,       // 56 bits
    pub entry_quote: Target,         // 56 bits
    pub total_order_count: Target,   // 16 bits
}

impl BinaryOptionsPositionTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            public_market_index: builder.add_virtual_target(),
            size: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
            entry_quote: builder.add_virtual_target(),
            total_order_count: builder.add_virtual_target(),
        }
    }

    pub fn empty(builder: &mut Builder, public_market_index: Target) -> Self {
        Self {
            public_market_index,
            size: builder.zero_bigint_u16(),
            entry_quote: builder.zero(),
            total_order_count: builder.zero(),
        }
    }

    /// Whether the position holds no size, entry quote or resting orders.
    pub fn has_no_position_data(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero_bigint_u16(&self.size),
            builder.is_zero(self.entry_quote),
            builder.is_zero(self.total_order_count),
        ];
        builder.multi_and(&assertions)
    }

    /// A position is empty when it has no position data and belongs to no market (nil public market index).
    fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let has_no_position_data = self.has_no_position_data(builder);
        let is_nil_public_market_index =
            builder.is_equal_constant(self.public_market_index, NIL_PUBLIC_MARKET_INDEX as u64);
        builder.and(has_no_position_data, is_nil_public_market_index)
    }

    /// Account market data tree leaf hash. Empty positions hash to the nil hash.
    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let market_type = builder.constant_u64(MARKET_TYPE_BINARY_OPTIONS);
        let mut elements = vec![market_type, self.public_market_index];
        elements.extend(self.size.abs.limbs.iter().map(|limb| limb.0));
        elements.extend([
            self.size.sign.target,
            self.entry_quote,
            self.total_order_count,
        ]);
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let is_empty = self.is_empty(builder);
        let empty_hash = builder.zero_hash_out();
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    /// Account market pub data tree leaf hash: only the size is published, a position without
    /// size hashes to the nil hash regardless of its resting orders.
    pub fn pub_data_hash(&self, builder: &mut Builder) -> HashOutTarget {
        let market_type = builder.constant_u64(MARKET_TYPE_BINARY_OPTIONS);
        let mut elements = vec![market_type];
        elements.extend(self.size.abs.limbs.iter().map(|limb| limb.0));
        elements.push(self.size.sign.target);
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let has_no_size = builder.is_zero_bigint_u16(&self.size);
        let empty_hash = builder.zero_hash_out();
        builder.select_hash(has_no_size, &empty_hash, &non_empty_hash)
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.public_market_index,
            &format!("{}: public_market_index", tag),
        );
        builder.println_bigint_u16(&self.size, &format!("{}: size", tag));
        builder.println(self.entry_quote, &format!("{}: entry_quote", tag));
        builder.println(
            self.total_order_count,
            &format!("{}: total_order_count", tag),
        );
    }
}

pub fn select_binary_options_position_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &BinaryOptionsPositionTarget,
    b: &BinaryOptionsPositionTarget,
) -> BinaryOptionsPositionTarget {
    BinaryOptionsPositionTarget {
        public_market_index: builder.select(flag, a.public_market_index, b.public_market_index),
        size: builder.select_bigint_u16(flag, &a.size, &b.size),
        entry_quote: builder.select(flag, a.entry_quote, b.entry_quote),
        total_order_count: builder.select(flag, a.total_order_count, b.total_order_count),
    }
}

pub trait BinaryOptionsPositionWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_binary_options_position_target(
        &mut self,
        t: &BinaryOptionsPositionTarget,
        p: &BinaryOptionsPosition,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> BinaryOptionsPositionWitness<F>
    for T
{
    fn set_binary_options_position_target(
        &mut self,
        t: &BinaryOptionsPositionTarget,
        p: &BinaryOptionsPosition,
    ) -> Result<()> {
        self.set_target(
            t.public_market_index,
            F::from_canonical_i64(p.public_market_index),
        )?;
        self.set_bigint_u16_target(&t.size, &BigInt::from(p.size))?;
        self.set_target(t.entry_quote, F::from_canonical_i64(p.entry_quote))?;
        self.set_target(
            t.total_order_count,
            F::from_canonical_i64(p.total_order_count),
        )?;
        Ok(())
    }
}
