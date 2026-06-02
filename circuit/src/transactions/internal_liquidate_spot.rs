// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::liquidation::get_asset_zero_price;
use crate::tx_interface::{Apply, Verify};
use crate::types::asset::is_universal_asset;
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::order::get_order_index;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct InternalLiquidateSpotTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "asi")]
    pub asset_index: i16,
    #[serde(rename = "mi")]
    pub market_index: i16,
    #[serde(rename = "ba")]
    pub base_amount: i64,
}

#[derive(Debug)]
pub struct InternalLiquidateSpotTxTarget {
    pub account_index: Target,
    pub asset_index: Target,
    pub market_index: Target,
    pub base_amount: Target,

    // helper
    // outputs
    success: BoolTarget,
}

impl InternalLiquidateSpotTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        InternalLiquidateSpotTxTarget {
            account_index: builder.add_virtual_target(),
            asset_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            base_amount: builder.add_virtual_target(),

            // helper
            // outputs
            success: BoolTarget::default(),
        }
    }

    fn get_pending_order_register(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
    ) -> BaseRegisterInfoTarget {
        BaseRegisterInfoTarget {
            instruction_type: builder.constant_from_u8(INSERT_ORDER),

            market_index: self.market_index,
            account_index: self.account_index,

            pending_size: self.base_amount,
            pending_order_index: get_order_index(
                builder,
                tx_state.market.market_index,
                tx_state.market.ask_nonce,
            ),
            pending_client_order_index: builder.constant_i64(NIL_CLIENT_ORDER_INDEX),
            pending_initial_size: self.base_amount,
            pending_price: get_asset_zero_price(
                builder,
                &tx_state.market,
                &tx_state.margined_asset[TX_ASSET_ID],
            ),
            pending_nonce: tx_state.market.ask_nonce,
            pending_is_ask: builder._true(),
            pending_type: builder.constant_from_u8(LIQUIDATION_ORDER),
            pending_time_in_force: builder.constant_from_u8(IOC),

            pending_reduce_only: builder.zero(),
            pending_expiry: builder.zero(),

            generic_field_0: builder.zero(),

            pending_trigger_price: builder.zero(),
            pending_trigger_status: builder.zero(),
            pending_to_trigger_order_index0: builder.zero(),
            pending_to_trigger_order_index1: builder.zero(),
            pending_to_cancel_order_index0: builder.zero(),
            generic_field_1: builder.zero(),
            generic_field_2: builder.zero(),
            generic_field_3: builder.zero(),
        }
    }
}

impl Verify for InternalLiquidateSpotTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_liquidate_spot;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[TAKER_ACCOUNT_ID].account_index,
        );

        // No insurance funds type
        let insurance_fund_op_index = builder.constant_u64(INSURANCE_FUND_ACCOUNT_TYPE as u64);
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
            insurance_fund_op_index,
        );

        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            EXECUTE_TRANSACTION as u64,
        );

        builder.conditional_assert_eq(
            is_enabled,
            self.asset_index,
            tx_state.asset_indices[BASE_ASSET_ID],
        );
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.market.base_asset_id,
            tx_state.asset_indices[BASE_ASSET_ID],
        );
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.asset_indices[QUOTE_ASSET_ID],
            tx_state.market.quote_asset_id,
        );

        let is_base_asset_empty = tx_state.assets[BASE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_base_asset_empty);
        let is_base_margined_asset_empty = tx_state.margined_asset[BASE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_base_margined_asset_empty);
        let is_base_asset_universal =
            is_universal_asset(builder, tx_state.asset_indices[BASE_ASSET_ID]);
        builder.conditional_assert_false(is_enabled, is_base_asset_universal);

        let is_quote_asset_empty = tx_state.assets[QUOTE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_quote_asset_empty);
        let is_quote_margined_asset_empty =
            tx_state.margined_asset[QUOTE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_quote_margined_asset_empty);
        let is_quote_asset_universal =
            is_universal_asset(builder, tx_state.asset_indices[QUOTE_ASSET_ID]);
        builder.conditional_assert_true(is_enabled, is_quote_asset_universal);

        let ob_active_status = builder.constant(F::from_canonical_u8(MARKET_STATUS_ACTIVE));
        builder.conditional_assert_eq(is_enabled, tx_state.market.status, ob_active_status);
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_SPOT,
        );
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.market.ask_nonce,
            tx_state.market.bid_nonce,
        );

        let is_margin_balance_negative = builder.is_sign_negative(
            tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID]
                .balance
                .sign,
        );
        builder.conditional_assert_false(is_enabled, is_margin_balance_negative);
        let base_amount = builder.target_to_biguint(self.base_amount);
        builder.range_check_biguint(&base_amount, ORDER_BASE_AMOUNT_BITS);
        let multiplier = builder.target_to_biguint(tx_state.market.size_extension_multiplier);
        let base_amount_extended =
            builder.mul_biguint_non_carry(&base_amount, &multiplier, BIG_U96_LIMBS);
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &base_amount_extended,
            &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID]
                .balance
                .abs,
        );

        // Make sure account is unhealthy
        let is_in_liquidation = tx_state.risk_infos[TAKER_ACCOUNT_ID]
            .cross_risk_parameters
            .is_in_liquidation(builder);
        builder.conditional_assert_true(is_enabled, is_in_liquidation);
    }
}

impl Apply for InternalLiquidateSpotTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();

        let new_register = self.get_pending_order_register(builder, tx_state);
        tx_state.put_to_instruction_stack_unsafe(builder, self.success, &new_register, 0);

        tx_state.market.ask_nonce =
            builder.mul_add(one, self.success.target, tx_state.market.ask_nonce);

        tx_state.matching_engine_flag = builder.or(tx_state.matching_engine_flag, self.success);

        self.success
    }
}

pub trait InternalLiquidateSpotTxTargetWitness<F: PrimeField64> {
    fn set_internal_liquidate_spot_tx_target(
        &mut self,
        a: &InternalLiquidateSpotTxTarget,
        b: &InternalLiquidateSpotTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalLiquidateSpotTxTargetWitness<F> for T {
    fn set_internal_liquidate_spot_tx_target(
        &mut self,
        a: &InternalLiquidateSpotTxTarget,
        b: &InternalLiquidateSpotTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index as i64))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;
        self.set_target(a.base_amount, F::from_canonical_i64(b.base_amount))?;
        Ok(())
    }
}
