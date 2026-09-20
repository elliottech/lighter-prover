// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use circuit::bigint::big_u16::bigint_u16::{
    BigIntU16Target, CircuitBuilderBigIntU16, WitnessBigInt16,
};
use circuit::bigint::big_u16::biguint_u16::CircuitBuilderBiguint16;
use circuit::bigint::bigint::CircuitBuilderBigInt;
use circuit::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use circuit::comparison::CircuitBuilderSubtractiveComparison;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::merkle_helpers::{conditional_verify_merkle_proof, market_index_to_merkle_path};
use circuit::poseidon2::Poseidon2Hash;
use circuit::types::config::{BIG_U128_LIMBS, BIGU16_U64_LIMBS, Builder, F};
use circuit::types::constants::{
    EXTENSION_MULTIPLIER_BITS, MARKET_MERKLE_LEVELS, MARKET_TYPE_BINARY_OPTIONS,
};
use circuit::utils::CircuitBuilderUtils;
use num::BigInt;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

pub const MARKET_STATUS_BITS: usize = 2;
pub const PRICE_BITS: usize = 32;

/// A binary options position of the exited account together with the published state of its
/// market. The size is a leaf of the account market pub data tree and the market fields are a
/// leaf of the market pub data tree, both keyed by the market slot.
#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "", default)]
pub struct PubdataBinaryOptionsPosition {
    #[serde(rename = "mi")]
    pub market_index: u16,
    #[serde(rename = "s")]
    pub size: i64,
    #[serde(rename = "st")]
    pub status: u8,
    #[serde(rename = "p")]
    pub price: u32,
    #[serde(rename = "sc")]
    pub settlement_cap: u32,
    #[serde(rename = "qem")]
    pub quote_extension_multiplier: i64,
    #[serde(rename = "mpampd")]
    #[serde(deserialize_with = "circuit::deserializers::market_tree_merkle_proof")]
    pub account_market_pub_data_merkle_proof: [HashOut<F>; MARKET_MERKLE_LEVELS],
    #[serde(rename = "mpmpd")]
    #[serde(deserialize_with = "circuit::deserializers::market_tree_merkle_proof")]
    pub market_pub_data_merkle_proof: [HashOut<F>; MARKET_MERKLE_LEVELS],
}

impl Default for PubdataBinaryOptionsPosition {
    fn default() -> Self {
        Self {
            market_index: 0,
            size: 0,
            status: 0,
            price: 0,
            settlement_cap: 0,
            quote_extension_multiplier: 0,
            account_market_pub_data_merkle_proof: [HashOut::<F>::ZERO; MARKET_MERKLE_LEVELS],
            market_pub_data_merkle_proof: [HashOut::<F>::ZERO; MARKET_MERKLE_LEVELS],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PubdataBinaryOptionsPositionTarget {
    pub market_index: Target,               // MARKET_MERKLE_LEVELS bits
    pub size: BigIntU16Target,              // 56 bits
    pub status: Target,                     // 2 bits
    pub price: Target,                      // 32 bits
    pub settlement_cap: Target,             // 32 bits
    pub quote_extension_multiplier: Target, // 56 bits
    pub account_market_pub_data_merkle_proof: [HashOutTarget; MARKET_MERKLE_LEVELS],
    pub market_pub_data_merkle_proof: [HashOutTarget; MARKET_MERKLE_LEVELS],
}

impl PubdataBinaryOptionsPositionTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            market_index: builder.add_virtual_target(),
            size: builder.add_virtual_bigint_u16_target_safe(BIGU16_U64_LIMBS),
            status: builder.add_virtual_target(),
            price: builder.add_virtual_target(),
            settlement_cap: builder.add_virtual_target(),
            quote_extension_multiplier: builder.add_virtual_target(),
            account_market_pub_data_merkle_proof: core::array::from_fn(|_| {
                builder.add_virtual_hash()
            }),
            market_pub_data_merkle_proof: core::array::from_fn(|_| builder.add_virtual_hash()),
        }
    }

    /// An entry without size is padding, its proofs and market fields are ignored.
    pub fn is_non_empty(&self, builder: &mut Builder) -> BoolTarget {
        let is_empty = builder.is_zero_bigint_u16(&self.size);
        builder.not(is_empty)
    }

    /// Account market pub data tree leaf, same preimage as `BinaryOptionsPositionTarget::pub_data_hash`.
    fn account_leaf_hash(&self, builder: &mut Builder) -> HashOutTarget {
        let market_type = builder.constant_u64(MARKET_TYPE_BINARY_OPTIONS);
        let mut elements = vec![market_type];
        elements.extend(self.size.abs.limbs.iter().map(|limb| limb.0));
        elements.push(self.size.sign.target);
        builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
    }

    /// Market pub data tree leaf, same preimage as `MarketTarget::pub_data_hash`.
    fn market_leaf_hash(&self, builder: &mut Builder) -> HashOutTarget {
        let market_type = builder.constant_u64(MARKET_TYPE_BINARY_OPTIONS);
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            market_type,
            self.status,
            self.price,
            self.settlement_cap,
            self.quote_extension_multiplier,
        ]);

        let fields_sum = builder.add_many([
            self.status,
            self.price,
            self.settlement_cap,
            self.quote_extension_multiplier,
        ]);
        let is_empty = builder.is_zero(fields_sum);
        let empty_hash = builder.zero_hash_out();
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    /// Proves the size against the account market pub data root and the market fields against the
    /// market pub data tree root for a non empty entry.
    pub fn verify(
        &self,
        builder: &mut Builder,
        account_market_pub_data_root: &HashOutTarget,
        market_pub_data_tree_root: &HashOutTarget,
    ) {
        let is_non_empty = self.is_non_empty(builder);
        let merkle_path = market_index_to_merkle_path(builder, self.market_index);

        let account_leaf_hash = self.account_leaf_hash(builder);
        conditional_verify_merkle_proof(
            builder,
            is_non_empty,
            account_market_pub_data_root,
            account_leaf_hash,
            self.account_market_pub_data_merkle_proof,
            merkle_path,
        );

        let market_leaf_hash = self.market_leaf_hash(builder);
        conditional_verify_merkle_proof(
            builder,
            is_non_empty,
            market_pub_data_tree_root,
            market_leaf_hash,
            self.market_pub_data_merkle_proof,
            merkle_path,
        );

        builder.register_range_check(self.status, MARKET_STATUS_BITS);
        builder.register_range_check(self.price, PRICE_BITS);
        builder.register_range_check(self.settlement_cap, PRICE_BITS);
        builder.register_range_check(self.quote_extension_multiplier, EXTENSION_MULTIPLIER_BITS);
        builder.conditional_assert_lte(is_non_empty, self.price, self.settlement_cap, PRICE_BITS);
    }

    /// Payout of the position in extended collateral: `|size| * price * qem` for a YES position and
    /// `|size| * (settlement_cap - price) * qem` for a NO position. The published price is the
    /// default price while the market is Active and the settlement price while InSettlement. An
    /// Expired market publishes no price and pays nothing.
    pub fn payout(&self, builder: &mut Builder) -> BigUintTarget {
        let is_non_empty = self.is_non_empty(builder);
        let is_settleable = builder.is_not_zero(self.status);
        let has_payout = builder.and(is_non_empty, is_settleable);
        let abs_size = builder.biguint_u16_to_target(&self.size.abs);
        let is_no = builder.is_sign_negative(self.size.sign);

        let no_price = builder.sub(self.settlement_cap, self.price);
        let price_per_share = builder.select(is_no, no_price, self.price);

        let abs_size_big = builder.target_to_biguint(abs_size);
        let price_per_share_big = builder.target_to_biguint(price_per_share);
        let quote_extension_multiplier_big =
            builder.target_to_biguint(self.quote_extension_multiplier);
        let payout = builder.mul_many_biguint_non_carry(
            &[
                &abs_size_big,
                &price_per_share_big,
                &quote_extension_multiplier_big,
            ],
            BIG_U128_LIMBS,
        );

        builder.mul_biguint_by_bool(&payout, has_payout)
    }
}

pub trait PubdataBinaryOptionsPositionTargetWitness {
    fn set_pubdata_binary_options_position_target(
        &mut self,
        target: &PubdataBinaryOptionsPositionTarget,
        witness: &PubdataBinaryOptionsPosition,
    ) -> Result<()>;
}

impl<T: Witness<F>> PubdataBinaryOptionsPositionTargetWitness for T {
    fn set_pubdata_binary_options_position_target(
        &mut self,
        target: &PubdataBinaryOptionsPositionTarget,
        witness: &PubdataBinaryOptionsPosition,
    ) -> Result<()> {
        self.set_target(
            target.market_index,
            F::from_canonical_u16(witness.market_index),
        )?;
        self.set_bigint_u16_target(&target.size, &BigInt::from(witness.size))?;
        self.set_target(target.status, F::from_canonical_u8(witness.status))?;
        self.set_target(target.price, F::from_canonical_u32(witness.price))?;
        self.set_target(
            target.settlement_cap,
            F::from_canonical_u32(witness.settlement_cap),
        )?;
        self.set_target(
            target.quote_extension_multiplier,
            F::from_canonical_i64(witness.quote_extension_multiplier),
        )?;
        for i in 0..MARKET_MERKLE_LEVELS {
            self.set_hash_target(
                target.account_market_pub_data_merkle_proof[i],
                witness.account_market_pub_data_merkle_proof[i],
            )?;
            self.set_hash_target(
                target.market_pub_data_merkle_proof[i],
                witness.market_pub_data_merkle_proof[i],
            )?;
        }
        Ok(())
    }
}
