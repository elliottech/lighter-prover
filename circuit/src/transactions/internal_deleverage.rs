// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::apply_trade::{ApplyTradeParams, apply_perps_trade};
use crate::bigint::big_u16::CircuitBuilderBiguint16;
use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::liquidation::{BoolOrTarget, get_available_asset_balance, get_position_zero_quote};
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::tx_interface::{Apply, Verify};
use crate::types::account_position::AccountPositionTarget;
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct InternalDeleverageTx {
    #[serde(rename = "b")]
    pub bankrupt_account_index: i64,

    #[serde(rename = "d")]
    pub deleverager_account_index: i64,

    #[serde(rename = "m")]
    pub market_index: u16,

    #[serde(rename = "s")]
    pub size: i64,
}

impl Default for InternalDeleverageTx {
    fn default() -> Self {
        InternalDeleverageTx::empty()
    }
}

impl InternalDeleverageTx {
    pub fn empty() -> Self {
        Self {
            bankrupt_account_index: 0,
            deleverager_account_index: 0,
            market_index: 0,
            size: 0,
        }
    }
}

#[derive(Debug)]
pub struct InternalDeleverageTxTarget {
    pub bankrupt_account_index: Target,
    pub deleverager_account_index: Target,
    pub market_index: Target,
    pub size: Target,

    // helper
    pub is_adl: BoolTarget,
    pub quote: SignedTarget,

    // outputs
    pub success: BoolTarget,
}

impl InternalDeleverageTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        InternalDeleverageTxTarget {
            bankrupt_account_index: builder.add_virtual_target(),
            deleverager_account_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            size: builder.add_virtual_target(),

            // helper
            quote: SignedTarget::default(),
            is_adl: BoolTarget::default(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalDeleverageTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_deleverage;
        self.success = is_enabled;

        // Verify integrity with accounts data
        builder.conditional_assert_eq(
            is_enabled,
            self.bankrupt_account_index,
            tx_state.accounts[BANKRUPT_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.deleverager_account_index,
            tx_state.accounts[DELEVERAGER_ACCOUNT_ID].account_index,
        );

        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        // Size should be non-zero
        builder.conditional_assert_not_zero(is_enabled, self.size);
        builder.register_range_check(self.size, ORDER_SIZE_BITS);

        let active_market_status = builder.constant_from_u8(MARKET_STATUS_ACTIVE);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.market_details.status,
            active_market_status,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );

        // Verify register instruction type
        let execute_transaction = builder.constant_from_u8(EXECUTE_TRANSACTION);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            execute_transaction,
        );

        // Verify that bankrupt and deleverager accounts are not same
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[BANKRUPT_ACCOUNT_ID].account_index,
            tx_state.accounts[DELEVERAGER_ACCOUNT_ID].account_index,
        );

        // Verify account health
        let bankrupt_account_health = tx_state.risk_infos[BANKRUPT_ACCOUNT_ID]
            .current_risk_parameters
            .get_health(builder);
        let full_liquidation_status = builder.constant_from_u8(FULL_LIQUIDATION);
        let bankruptcy_status = builder.constant_from_u8(BANKRUPTCY);
        let is_bankrupt_account_health_full_liquidation =
            builder.is_equal(bankrupt_account_health, full_liquidation_status);
        let is_bankrupt_account_health_bankruptcy =
            builder.is_equal(bankrupt_account_health, bankruptcy_status);
        let is_bankrupt_account_health_correct = builder.or(
            is_bankrupt_account_health_bankruptcy,
            is_bankrupt_account_health_full_liquidation,
        );
        builder.conditional_assert_true(is_enabled, is_bankrupt_account_health_correct);

        // Bankrupt account should not have any open orders (unless bankrupt account is insurance fund)
        let cross_order_count = builder.sub(
            tx_state.accounts[BANKRUPT_ACCOUNT_ID].total_order_count,
            tx_state.accounts[BANKRUPT_ACCOUNT_ID].total_non_cross_order_count,
        );
        let isolated_order_count = tx_state.positions[BANKRUPT_ACCOUNT_ID].total_order_count;
        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
        let is_position_isolated = builder.is_equal(
            tx_state.positions[BANKRUPT_ACCOUNT_ID].margin_mode,
            isolated_margin_mode,
        );
        let order_count = builder.select(
            is_position_isolated,
            isolated_order_count,
            cross_order_count,
        );
        let is_order_count_zero = builder.is_zero(order_count);
        let is_bankrupt_account_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[BANKRUPT_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let should_be_true = builder.or(is_order_count_zero, is_bankrupt_account_insurance_fund);
        builder.conditional_assert_true(is_enabled, should_be_true);

        // Bankrupt account should have a position that is at least the size of the deleverage in absolute value
        let abs_bankrupt_position =
            builder.biguint_u16_to_target(&tx_state.positions[BANKRUPT_ACCOUNT_ID].position.abs);

        builder.conditional_assert_lte(is_enabled, self.size, abs_bankrupt_position, 64);

        let insurance_fund_operator_account_type =
            builder.constant_from_u8(INSURANCE_FUND_ACCOUNT_TYPE);
        self.is_adl = builder.is_not_equal(
            tx_state.accounts[DELEVERAGER_ACCOUNT_ID].account_type,
            insurance_fund_operator_account_type,
        );
        let is_enabled_and_is_adl = builder.and(is_enabled, self.is_adl);
        let is_not_adl = builder.not(self.is_adl);
        let is_enabled_and_is_not_adl = builder.and(is_enabled, is_not_adl);

        let frozen_public_pool = builder.constant_u64(FROZEN_PUBLIC_POOL as u64);
        builder.conditional_assert_not_eq(
            is_enabled_and_is_not_adl,
            tx_state.accounts[DELEVERAGER_ACCOUNT_ID]
                .public_pool_info
                .status,
            frozen_public_pool,
        );

        let bankrupt_position = tx_state.positions[BANKRUPT_ACCOUNT_ID].clone();
        let bankrupt_risk_info = tx_state.risk_infos[BANKRUPT_ACCOUNT_ID].clone();
        builder.conditional_assert_not_zero_biguint(
            is_enabled,
            &tx_state.risk_infos[TAKER_ACCOUNT_ID]
                .current_risk_parameters
                .maintenance_margin_requirement,
        );
        let big_deleverage_quote = get_position_zero_quote(
            builder,
            &bankrupt_position,
            &tx_state.market_details,
            &bankrupt_risk_info.current_risk_parameters,
            self.size,
        );

        let big_max_deleverage_quote =
            builder.constant_biguint(&BigUint::from(MAX_DELEVERAGE_QUOTE));
        let is_deleverage_quote_valid =
            builder.is_lte_biguint(&big_deleverage_quote.abs, &big_max_deleverage_quote);
        builder.conditional_assert_true(is_enabled, is_deleverage_quote_valid);

        // Safe call because big_deleverage_quote is a safe bigint calculated by get_position_zero_quote,
        // and it's is checked to be less than MAX_DELEVERAGE_QUOTE = 2^56 in bigint form, so it won't
        // exceed POSITIVE_THRESHOLD_BIT = 2^60 when converted to SignedTarget.
        self.quote = builder.bigint_to_signed_target_unsafe(&big_deleverage_quote);

        let is_quote_positive = builder.is_positive(self.quote);
        let is_enabled_and_is_quote_not_positive = builder.and_not(is_enabled, is_quote_positive);
        builder.conditional_assert_false(is_enabled_and_is_quote_not_positive, self.is_adl);

        // Deleverager account if from ADL queue, should have a position on the opposite side
        // and the position should be at least the size of the deleverage in absolute value
        let abs_deleverager_position =
            builder.biguint_u16_to_target(&tx_state.positions[DELEVERAGER_ACCOUNT_ID].position.abs);
        builder.conditional_assert_lte(
            is_enabled_and_is_adl,
            self.size,
            abs_deleverager_position,
            64,
        );

        builder.conditional_assert_not_eq(
            is_enabled_and_is_adl,
            tx_state.positions[BANKRUPT_ACCOUNT_ID].position.sign.target,
            tx_state.positions[DELEVERAGER_ACCOUNT_ID]
                .position
                .sign
                .target,
        );
    }
}

impl Apply for InternalDeleverageTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let bankrupt_position = tx_state.positions[BANKRUPT_ACCOUNT_ID].position.clone();
        let is_bankrupt_long = builder.is_sign_positive(bankrupt_position.sign);

        let apply_trade_params = ApplyTradeParams {
            market: &tx_state.market,
            market_details: &tx_state.market_details.clone(),
            is_taker_ask: is_bankrupt_long,
            trade_base: self.size,
            trade_quote: self.quote,
            taker_position: &tx_state.positions[BANKRUPT_ACCOUNT_ID].clone(),
            maker_position: &tx_state.positions[DELEVERAGER_ACCOUNT_ID].clone(),
            taker_risk_info: &tx_state.risk_infos[BANKRUPT_ACCOUNT_ID].clone(),
            maker_risk_info: &tx_state.risk_infos[DELEVERAGER_ACCOUNT_ID].clone(),
            taker_fee: builder.zero_signed(),
            maker_fee: builder.zero_signed(),
        };

        let (
            new_bankrupt_position,
            new_deleverager_position,
            new_bankrupt_risk_info,
            new_deleverager_risk_info,
            _fee_collateral_delta, // no fee
            new_open_interest,
            _bankrupt_position_sign_changed,
            deleverager_position_sign_changed,
            is_bankrupt_position_isolated,
            is_deleverager_position_isolated,
            bankrupt_margin_delta,
            deleverager_margin_delta,
        ) = apply_perps_trade(builder, self.success, &apply_trade_params);

        let _perps = builder.constant_u64(PRODUCT_TYPE_PERPS);
        let usdc_asset_index = builder.constant_u64(USDC_ASSET_INDEX);
        let bankrupt_available_cross_collateral = get_available_asset_balance(
            builder,
            _perps,
            usdc_asset_index,
            &tx_state.accounts[BANKRUPT_ACCOUNT_ID],
            &tx_state.account_assets[BANKRUPT_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[BANKRUPT_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[BANKRUPT_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.margined_asset[TX_ASSET_ID],
            &tx_state.account_margined_assets[BANKRUPT_ACCOUNT_ID][TX_ASSET_ID].balance,
            BoolOrTarget::True,
        );
        let is_bankrupt_has_enough_cross_collateral = {
            // new collateral = old collateral - margin_delta
            let collateral_gte_delta = builder.is_gte_biguint(
                &bankrupt_available_cross_collateral,
                &bankrupt_margin_delta.abs,
            );
            let is_delta_negative = builder.is_sign_negative(bankrupt_margin_delta.sign);

            // If delta is negative, the new collateral is increasing. Otherwise, we make sure that old collateral is greater than or equal to the margin delta.
            builder.or(collateral_gte_delta, is_delta_negative)
        };

        let deleverager_available_cross_collateral = get_available_asset_balance(
            builder,
            _perps,
            usdc_asset_index,
            &tx_state.accounts[DELEVERAGER_ACCOUNT_ID],
            &tx_state.account_assets[DELEVERAGER_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[DELEVERAGER_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[DELEVERAGER_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.margined_asset[TX_ASSET_ID],
            &tx_state.account_margined_assets[DELEVERAGER_ACCOUNT_ID][TX_ASSET_ID].balance,
            BoolOrTarget::True,
        );
        let is_deleverager_has_enough_cross_collateral = {
            // new collateral = old collateral - margin_delta
            let collateral_gte_delta = builder.is_gte_biguint(
                &deleverager_available_cross_collateral,
                &deleverager_margin_delta.abs,
            );
            let is_delta_negative = builder.is_sign_negative(deleverager_margin_delta.sign);

            // If delta is negative, the new collateral is increasing. Otherwise, we make sure that old collateral is greater than or equal to the margin delta.
            builder.or(collateral_gte_delta, is_delta_negative)
        };

        let new_bankrupt_position_is_valid = new_bankrupt_position.is_valid(builder);
        let new_deleverager_position_is_valid = new_deleverager_position.is_valid(builder);
        builder.conditional_assert_true(self.success, new_bankrupt_position_is_valid);
        builder.conditional_assert_true(self.success, new_deleverager_position_is_valid);

        let bankrupt_account_valid_risk_change = apply_trade_params
            .taker_risk_info
            .current_risk_parameters
            .is_valid_risk_change(builder, &new_bankrupt_risk_info.current_risk_parameters);
        builder.conditional_assert_true(self.success, bankrupt_account_valid_risk_change);
        builder.conditional_assert_true(self.success, is_bankrupt_has_enough_cross_collateral);

        let is_enabled_and_is_adl = builder.and(self.success, self.is_adl);
        builder.conditional_assert_true(
            is_enabled_and_is_adl,
            is_deleverager_has_enough_cross_collateral,
        );

        // Apply trade deltas
        tx_state.market_details.open_interest = builder.select(
            self.success,
            new_open_interest,
            tx_state.market_details.open_interest,
        );

        // Update positions in tx_state
        tx_state.positions[BANKRUPT_ACCOUNT_ID] = AccountPositionTarget::select_position(
            builder,
            self.success,
            &new_bankrupt_position,
            &tx_state.positions[BANKRUPT_ACCOUNT_ID],
        );
        tx_state.positions[DELEVERAGER_ACCOUNT_ID] = AccountPositionTarget::select_position(
            builder,
            self.success,
            &new_deleverager_position,
            &tx_state.positions[DELEVERAGER_ACCOUNT_ID],
        );

        // Update collaterals - Because deleverage can only happen in Perps markets, we don't need to handle asset balances for unified accounts
        let new_bankrupt_collateral = builder.select_bigint(
            is_bankrupt_position_isolated,
            &new_bankrupt_risk_info.cross_risk_parameters.usdc_collateral,
            &new_bankrupt_risk_info
                .current_risk_parameters
                .usdc_collateral,
        );
        let old_bankrupt_collateral = builder.select_bigint(
            is_bankrupt_position_isolated,
            &tx_state.risk_infos[BANKRUPT_ACCOUNT_ID]
                .cross_risk_parameters
                .usdc_collateral,
            &tx_state.risk_infos[BANKRUPT_ACCOUNT_ID]
                .current_risk_parameters
                .usdc_collateral,
        );
        let bankrupt_collateral_delta =
            builder.sub_bigint(&new_bankrupt_collateral, &old_bankrupt_collateral);

        tx_state.accounts[BANKRUPT_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &bankrupt_collateral_delta,
            &mut tx_state.strategies[BANKRUPT_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[BANKRUPT_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        let new_deleverager_collateral = builder.select_bigint(
            is_deleverager_position_isolated,
            &new_deleverager_risk_info
                .cross_risk_parameters
                .usdc_collateral,
            &new_deleverager_risk_info
                .current_risk_parameters
                .usdc_collateral,
        );
        let old_deleverager_collateral = builder.select_bigint(
            is_deleverager_position_isolated,
            &tx_state.risk_infos[DELEVERAGER_ACCOUNT_ID]
                .cross_risk_parameters
                .usdc_collateral,
            &tx_state.risk_infos[DELEVERAGER_ACCOUNT_ID]
                .current_risk_parameters
                .usdc_collateral,
        );
        let deleverager_collateral_delta =
            builder.sub_bigint(&new_deleverager_collateral, &old_deleverager_collateral);

        tx_state.accounts[DELEVERAGER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &deleverager_collateral_delta,
            &mut tx_state.strategies[DELEVERAGER_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[DELEVERAGER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        // If deleverager position sign changes, update the register to cancel reduce only orders
        let position_tied_order_count =
            tx_state.positions[DELEVERAGER_ACCOUNT_ID].total_position_tied_order_count;
        let non_zero_position_tied_order_count = builder.is_not_zero(position_tied_order_count);
        let cancel_maker_position_tied_orders_flag = builder.multi_and(&[
            self.success,
            non_zero_position_tied_order_count,
            deleverager_position_sign_changed,
        ]);

        let cancel_position_tied_account_orders =
            builder.constant_from_u8(CANCEL_POSITION_TIED_ACCOUNT_ORDERS);
        let cancel_position_tied_account_orders_instruction = BaseRegisterInfoTarget {
            instruction_type: cancel_position_tied_account_orders,
            market_index: self.market_index,
            account_index: self.deleverager_account_index,
            pending_size: position_tied_order_count,
            ..BaseRegisterInfoTarget::empty(builder)
        };
        tx_state.put_to_instruction_stack_unsafe(
            builder,
            cancel_maker_position_tied_orders_flag,
            &cancel_position_tied_account_orders_instruction,
            0,
        );

        self.success
    }
}

pub trait InternalDeleverageTxTargetWitness<F: PrimeField64> {
    fn set_internal_deleverage_tx_target(
        &mut self,
        a: &InternalDeleverageTxTarget,
        b: &InternalDeleverageTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalDeleverageTxTargetWitness<F> for T {
    fn set_internal_deleverage_tx_target(
        &mut self,
        a: &InternalDeleverageTxTarget,
        b: &InternalDeleverageTx,
    ) -> Result<()> {
        self.set_target(
            a.bankrupt_account_index,
            F::from_canonical_i64(b.bankrupt_account_index),
        )?;
        self.set_target(
            a.deleverager_account_index,
            F::from_canonical_i64(b.deleverager_account_index),
        )?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(a.size, F::from_canonical_i64(b.size))?;

        Ok(())
    }
}
