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
use crate::hints::CircuitBuilderHints;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::market::{MarketTarget, select_market};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CreateMarketTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "mi")]
    pub market_index: i16,
    #[serde(rename = "mt")]
    pub market_type: u8,
    #[serde(rename = "ba")]
    pub base_asset_id: i16,
    #[serde(rename = "qa")]
    pub quote_asset_id: i16,
    #[serde(rename = "st")]
    pub start_timestamp: i64,
    #[serde(rename = "et")]
    pub end_timestamp: i64,

    #[serde(rename = "sem")]
    pub size_extension_multiplier: i64,
    #[serde(rename = "qem")]
    pub quote_extension_multiplier: i64,
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
    #[serde(rename = "sc")]
    pub settlement_cap: u32,
    #[serde(rename = "sty")]
    pub settlement_type: u8,
    #[serde(rename = "dp")]
    pub default_price: u32,
    #[serde(rename = "fz")]
    pub is_frozen: u8,
}

#[derive(Debug)]
pub struct L2CreateMarketTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,

    pub market_index: Target,
    pub market_type: Target,
    pub base_asset_id: Target,
    pub quote_asset_id: Target,
    pub start_timestamp: Target,
    pub end_timestamp: Target,

    pub size_extension_multiplier: Target,
    pub quote_extension_multiplier: Target,
    pub taker_fee: Target,
    pub maker_fee: Target,
    pub min_base_amount: Target,
    pub min_quote_amount: Target,
    pub order_quote_limit: Target,
    pub open_interest_limit: Target,
    pub settlement_cap: Target,
    pub settlement_type: Target,
    pub default_price: Target,
    pub is_frozen: Target,

    // Output
    success: BoolTarget,
}

impl L2CreateMarketTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),

            market_index: builder.add_virtual_target(),
            market_type: builder.add_virtual_target(),
            base_asset_id: builder.add_virtual_target(),
            quote_asset_id: builder.add_virtual_target(),
            start_timestamp: builder.add_virtual_target(),
            end_timestamp: builder.add_virtual_target(),

            size_extension_multiplier: builder.add_virtual_target(),
            quote_extension_multiplier: builder.add_virtual_target(),
            taker_fee: builder.add_virtual_target(),
            maker_fee: builder.add_virtual_target(),
            min_base_amount: builder.add_virtual_target(),
            min_quote_amount: builder.add_virtual_target(),
            order_quote_limit: builder.add_virtual_target(),
            open_interest_limit: builder.add_virtual_target(),
            settlement_cap: builder.add_virtual_target(),
            settlement_type: builder.add_virtual_target(),
            default_price: builder.add_virtual_target(),
            is_frozen: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        builder.register_range_check(self.start_timestamp, TIMESTAMP_BITS);
        builder.register_range_check(self.end_timestamp, TIMESTAMP_BITS);
        builder.register_range_check(
            self.size_extension_multiplier,
            ASSET_EXTENSION_MULTIPLIER_BITS,
        );
        builder.register_range_check(
            self.quote_extension_multiplier,
            ASSET_EXTENSION_MULTIPLIER_BITS,
        );
        builder.register_range_check(self.taker_fee, 24);
        builder.register_range_check(self.maker_fee, 24);
        builder.register_range_check(self.min_base_amount, ORDER_BASE_AMOUNT_BITS);
        builder.register_range_check(self.min_quote_amount, ORDER_QUOTE_SIZE_BITS);
        builder.register_range_check(self.order_quote_limit, ORDER_QUOTE_SIZE_BITS);
        builder.register_range_check(self.open_interest_limit, MARKET_OPEN_INTEREST_BITS);
        builder.register_range_check(self.settlement_cap, ORDER_PRICE_BITS);
        builder.register_range_check(self.default_price, ORDER_PRICE_BITS);
    }
}

impl TxHash for L2CreateMarketTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_MARKET)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.market_type,
            self.base_asset_id,
            self.quote_asset_id,
            self.start_timestamp,
            self.end_timestamp,
            self.size_extension_multiplier,
            self.quote_extension_multiplier,
            self.settlement_cap,
            self.settlement_type,
            self.default_price,
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

impl Verify for L2CreateMarketTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_create_market;
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

        // The market is created on a binary options slot
        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.market_index, nil_market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.binary_options_market_index,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            self.market_type,
            MARKET_TYPE_BINARY_OPTIONS,
        );

        // Only the operator assigned to the slot can create a market on it
        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.market.market_operator_account_index,
        );

        // Binary options markets are quoted (and settled) in USDC on both sides
        builder.conditional_assert_eq_constant(is_enabled, self.base_asset_id, USDC_ASSET_INDEX);
        builder.conditional_assert_eq_constant(is_enabled, self.quote_asset_id, USDC_ASSET_INDEX);
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
            USDC_ASSET_INDEX,
        );
        let is_usdc_asset_empty = tx_state.assets[USDC_BASE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_usdc_asset_empty);

        // The market slot must be expired and fully drained. A binary options market only reaches
        // the expired status once every position is settled and every resting order is cancelled,
        // so a reused slot never carries positions of a previous market.
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.status,
            MARKET_STATUS_EXPIRED as u64,
        );
        builder.conditional_assert_zero(is_enabled, tx_state.market.total_order_count);
        builder.conditional_assert_zero(is_enabled, tx_state.market.open_interest);

        // The public market index counter must still be in the assignable range, above the nil
        // sentinel and the reserved legacy range and not yet exhausted
        let nil_public_market_index = builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64);
        builder.conditional_assert_lt(
            is_enabled,
            nil_public_market_index,
            tx_state.next_public_market_index,
            56,
        );
        let exhausted_public_market_index =
            builder.constant_u64(MAX_PUBLIC_MARKET_INDEX as u64 + 1);
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.next_public_market_index,
            exhausted_public_market_index,
        );

        builder.conditional_assert_lt(
            is_enabled,
            self.start_timestamp,
            self.end_timestamp,
            TIMESTAMP_BITS,
        );

        builder.conditional_assert_not_zero(is_enabled, self.size_extension_multiplier);
        builder.conditional_assert_not_zero(is_enabled, self.quote_extension_multiplier);

        // settlement_cap * quote_extension_multiplier == size_extension_multiplier, so that a
        // share's full payout in quote ticks extends exactly to one extended base unit
        builder.conditional_assert_not_zero(is_enabled, self.settlement_cap);
        let settlement_cap_big = builder.target_to_biguint(self.settlement_cap);
        let quote_extension_multiplier_big =
            builder.target_to_biguint(self.quote_extension_multiplier);
        let extended_cap =
            builder.mul_biguint(&settlement_cap_big, &quote_extension_multiplier_big);
        let size_extension_multiplier_big =
            builder.target_to_biguint(self.size_extension_multiplier);
        builder.conditional_assert_eq_biguint(
            is_enabled,
            &extended_cap,
            &size_extension_multiplier_big,
        );

        builder.conditional_assert_bool(is_enabled, BoolTarget::new_unsafe(self.settlement_type));
        builder.conditional_assert_bool(is_enabled, BoolTarget::new_unsafe(self.is_frozen));
        builder.conditional_assert_lte(
            is_enabled,
            self.default_price,
            self.settlement_cap,
            ORDER_PRICE_BITS,
        );

        let fee_tick = builder.constant(F::from_canonical_u64(FEE_TICK));
        builder.conditional_assert_lte(is_enabled, self.taker_fee, fee_tick, 24);
        builder.conditional_assert_lte(is_enabled, self.maker_fee, fee_tick, 24);
        // Fees scale by quote_extension_multiplier / FEE_TICK, which must be exact
        let (_, quote_multiplier_fee_remainder) =
            builder.div_rem(self.quote_extension_multiplier, fee_tick, FEE_BITS);
        builder.conditional_assert_zero(is_enabled, quote_multiplier_fee_remainder);

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
    }
}

impl Apply for L2CreateMarketTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let nil_market_index = builder.constant_u64(NIL_MARKET_INDEX as u64);
        let assigned_public_market_index = tx_state.next_public_market_index;
        // A reused slot keeps its ask/bid nonces so order indexes of a settled market are never reissued
        let (ask_nonce, bid_nonce) = tx_state.market.next_market_nonces(builder);

        let market_after = MarketTarget {
            market_index: self.market_index,
            public_market_index: assigned_public_market_index,
            perps_market_index: nil_market_index,
            binary_options_market_index: self.market_index,

            status: builder.constant_from_u8(MARKET_STATUS_ACTIVE),
            market_type: self.market_type,
            base_asset_id: self.base_asset_id,
            quote_asset_id: self.quote_asset_id,

            ask_nonce,
            bid_nonce,

            taker_fee: self.taker_fee,
            maker_fee: self.maker_fee,
            liquidation_fee: builder.zero(),
            size_extension_multiplier: self.size_extension_multiplier,
            quote_extension_multiplier: self.quote_extension_multiplier,
            total_order_count: builder.zero(),
            min_base_amount: self.min_base_amount,
            min_quote_amount: self.min_quote_amount,
            order_quote_limit: self.order_quote_limit,

            start_timestamp: self.start_timestamp,
            end_timestamp: self.end_timestamp,
            open_interest: builder.zero(),
            open_interest_limit: self.open_interest_limit,
            outcome: builder.zero(),
            market_operator_account_index: tx_state.market.market_operator_account_index,
            settlement_cap: self.settlement_cap,
            settlement_type: self.settlement_type,
            settlement_price: builder.zero(),
            default_price: self.default_price,
            is_frozen: self.is_frozen,

            order_book_root: builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT),
        };
        tx_state.market = select_market(builder, self.success, &market_after, &tx_state.market);

        let next_public_market_index_after = builder.add_one(tx_state.next_public_market_index);
        tx_state.next_public_market_index = builder.select(
            self.success,
            next_public_market_index_after,
            assigned_public_market_index,
        );

        self.success
    }
}

pub trait L2CreateMarketTxTargetWitness<F: PrimeField64> {
    fn set_l2_create_market_tx_target(
        &mut self,
        a: &L2CreateMarketTxTarget,
        b: &L2CreateMarketTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CreateMarketTxTargetWitness<F> for T {
    fn set_l2_create_market_tx_target(
        &mut self,
        a: &L2CreateMarketTxTarget,
        b: &L2CreateMarketTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;
        self.set_target(a.market_type, F::from_canonical_u8(b.market_type))?;
        self.set_target(
            a.base_asset_id,
            F::from_canonical_i64(b.base_asset_id as i64),
        )?;
        self.set_target(
            a.quote_asset_id,
            F::from_canonical_i64(b.quote_asset_id as i64),
        )?;
        self.set_target(a.start_timestamp, F::from_canonical_i64(b.start_timestamp))?;
        self.set_target(a.end_timestamp, F::from_canonical_i64(b.end_timestamp))?;
        self.set_target(
            a.size_extension_multiplier,
            F::from_canonical_i64(b.size_extension_multiplier),
        )?;
        self.set_target(
            a.quote_extension_multiplier,
            F::from_canonical_i64(b.quote_extension_multiplier),
        )?;
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
        self.set_target(a.settlement_cap, F::from_canonical_u32(b.settlement_cap))?;
        self.set_target(a.settlement_type, F::from_canonical_u8(b.settlement_type))?;
        self.set_target(a.default_price, F::from_canonical_u32(b.default_price))?;
        self.set_target(a.is_frozen, F::from_canonical_u8(b.is_frozen))?;

        Ok(())
    }
}
