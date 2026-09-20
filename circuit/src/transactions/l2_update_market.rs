// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

/// Updates the mutable parameters of an active binary options market on behalf of its market operator.
/// Settlement parameters, extension multipliers and the market identity are immutable.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UpdateMarketTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "mi")]
    pub market_index: i16,
    #[serde(rename = "pmi")]
    pub public_market_index: i64,
    #[serde(rename = "st")]
    pub start_timestamp: i64,
    #[serde(rename = "et")]
    pub end_timestamp: i64,
    #[serde(rename = "tk")]
    pub taker_fee: u32,
    #[serde(rename = "mf")]
    pub maker_fee: u32,
    #[serde(rename = "mba")]
    pub min_base_amount: i64,
    #[serde(rename = "mqa")]
    pub min_quote_amount: i64,
    #[serde(rename = "oql")]
    pub order_quote_limit: i64,
    #[serde(rename = "oil")]
    pub open_interest_limit: i64,
    #[serde(rename = "fz")]
    pub is_frozen: u8,
}

#[derive(Debug)]
pub struct L2UpdateMarketTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub market_index: Target,
    pub public_market_index: Target, // 48 bits
    pub start_timestamp: Target,
    pub end_timestamp: Target,
    pub taker_fee: Target,
    pub maker_fee: Target,
    pub min_base_amount: Target,
    pub min_quote_amount: Target,
    pub order_quote_limit: Target,
    pub open_interest_limit: Target,
    pub is_frozen: Target,

    // Output
    success: BoolTarget,
}

impl L2UpdateMarketTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            public_market_index: builder.add_virtual_target(),
            start_timestamp: builder.add_virtual_target(),
            end_timestamp: builder.add_virtual_target(),
            taker_fee: builder.add_virtual_target(),
            maker_fee: builder.add_virtual_target(),
            min_base_amount: builder.add_virtual_target(),
            min_quote_amount: builder.add_virtual_target(),
            order_quote_limit: builder.add_virtual_target(),
            open_interest_limit: builder.add_virtual_target(),
            is_frozen: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        builder.register_range_check(self.start_timestamp, TIMESTAMP_BITS);
        builder.register_range_check(self.end_timestamp, TIMESTAMP_BITS);
        builder.register_range_check(self.taker_fee, 24);
        builder.register_range_check(self.maker_fee, 24);
        builder.register_range_check(self.min_base_amount, ORDER_BASE_AMOUNT_BITS);
        builder.register_range_check(self.min_quote_amount, ORDER_QUOTE_SIZE_BITS);
        builder.register_range_check(self.order_quote_limit, ORDER_QUOTE_SIZE_BITS);
        builder.register_range_check(self.open_interest_limit, MARKET_OPEN_INTEREST_BITS);
    }
}

impl TxHash for L2UpdateMarketTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_MARKET)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.public_market_index,
            self.start_timestamp,
            self.end_timestamp,
            self.taker_fee,
            self.maker_fee,
            self.min_base_amount,
            self.min_quote_amount,
            self.order_quote_limit,
            self.open_interest_limit,
            self.is_frozen,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateMarketTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_market;
        self.success = is_enabled;

        self.register_range_checks(builder);

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );

        // The market is addressed by its binary options slot and its public market index
        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.market_index, nil_market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.binary_options_market_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.public_market_index,
            tx_state.market.public_market_index,
        );
        let nil_public_market_index = builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64);
        builder.conditional_assert_not_eq(
            is_enabled,
            self.public_market_index,
            nil_public_market_index,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_BINARY_OPTIONS,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.status,
            MARKET_STATUS_ACTIVE as u64,
        );

        // Only the market operator can update the market
        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.market.market_operator_account_index,
        );

        // Once trading has opened the trading window is frozen
        let is_market_started = builder.is_lte(
            tx_state.market.start_timestamp,
            tx_state.block_timestamp,
            TIMESTAMP_BITS,
        );
        let started_flag = builder.and(is_enabled, is_market_started);
        builder.conditional_assert_eq(
            started_flag,
            self.start_timestamp,
            tx_state.market.start_timestamp,
        );
        builder.conditional_assert_eq(
            started_flag,
            self.end_timestamp,
            tx_state.market.end_timestamp,
        );
        builder.conditional_assert_lt(
            is_enabled,
            self.start_timestamp,
            self.end_timestamp,
            TIMESTAMP_BITS,
        );

        let fee_tick = builder.constant(F::from_canonical_u64(FEE_TICK));
        builder.conditional_assert_lte(is_enabled, self.taker_fee, fee_tick, 24);
        builder.conditional_assert_lte(is_enabled, self.maker_fee, fee_tick, 24);

        builder.conditional_assert_not_zero(is_enabled, self.min_base_amount);
        builder.conditional_assert_not_zero(is_enabled, self.min_quote_amount);
        builder.conditional_assert_lte(
            is_enabled,
            self.min_quote_amount,
            self.order_quote_limit,
            ORDER_QUOTE_SIZE_BITS,
        );
        builder.conditional_assert_lte(
            is_enabled,
            self.order_quote_limit,
            self.open_interest_limit,
            MARKET_OPEN_INTEREST_BITS,
        );
        // The open interest limit scaled to collateral units must fit in an i64
        let open_interest_limit_big = builder.target_to_biguint(self.open_interest_limit);
        let quote_extension_multiplier_big =
            builder.target_to_biguint(tx_state.market.quote_extension_multiplier);
        let scaled_open_interest_limit =
            builder.mul_biguint(&open_interest_limit_big, &quote_extension_multiplier_big);
        let max_scaled_open_interest_limit = builder.constant_biguint(
            &(BigUint::from(i64::MAX as u64) * BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER)),
        );
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &scaled_open_interest_limit,
            &max_scaled_open_interest_limit,
        );

        builder.conditional_assert_bool(is_enabled, BoolTarget::new_unsafe(self.is_frozen));
    }
}

impl Apply for L2UpdateMarketTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let market = &mut tx_state.market;
        market.start_timestamp =
            builder.select(self.success, self.start_timestamp, market.start_timestamp);
        market.end_timestamp =
            builder.select(self.success, self.end_timestamp, market.end_timestamp);
        market.taker_fee = builder.select(self.success, self.taker_fee, market.taker_fee);
        market.maker_fee = builder.select(self.success, self.maker_fee, market.maker_fee);
        market.min_base_amount =
            builder.select(self.success, self.min_base_amount, market.min_base_amount);
        market.min_quote_amount =
            builder.select(self.success, self.min_quote_amount, market.min_quote_amount);
        market.order_quote_limit = builder.select(
            self.success,
            self.order_quote_limit,
            market.order_quote_limit,
        );
        market.open_interest_limit = builder.select(
            self.success,
            self.open_interest_limit,
            market.open_interest_limit,
        );
        market.is_frozen = builder.select(self.success, self.is_frozen, market.is_frozen);

        self.success
    }
}

pub trait L2UpdateMarketTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_market_tx_target(
        &mut self,
        a: &L2UpdateMarketTxTarget,
        b: &L2UpdateMarketTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateMarketTxTargetWitness<F> for T {
    fn set_l2_update_market_tx_target(
        &mut self,
        a: &L2UpdateMarketTxTarget,
        b: &L2UpdateMarketTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;
        self.set_target(
            a.public_market_index,
            F::from_canonical_i64(b.public_market_index),
        )?;
        self.set_target(a.start_timestamp, F::from_canonical_i64(b.start_timestamp))?;
        self.set_target(a.end_timestamp, F::from_canonical_i64(b.end_timestamp))?;
        self.set_target(a.taker_fee, F::from_canonical_u32(b.taker_fee))?;
        self.set_target(a.maker_fee, F::from_canonical_u32(b.maker_fee))?;
        self.set_target(a.min_base_amount, F::from_canonical_i64(b.min_base_amount))?;
        self.set_target(
            a.min_quote_amount,
            F::from_canonical_i64(b.min_quote_amount),
        )?;
        self.set_target(
            a.order_quote_limit,
            F::from_canonical_i64(b.order_quote_limit),
        )?;
        self.set_target(
            a.open_interest_limit,
            F::from_canonical_i64(b.open_interest_limit),
        )?;
        self.set_target(a.is_frozen, F::from_canonical_u8(b.is_frozen))?;

        Ok(())
    }
}
