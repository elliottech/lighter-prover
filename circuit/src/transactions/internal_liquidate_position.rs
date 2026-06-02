// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigUint, FromPrimitive};
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::CircuitBuilderBiguint16;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::liquidation::get_position_zero_price;
use crate::matching_engine::get_next_order_nonce;
use crate::tx_interface::{Apply, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::order::get_order_index;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
pub struct InternalLiquidatePositionTx {
    #[serde(rename = "a")]
    pub account_index: i64,

    #[serde(rename = "m")]
    pub market_index: u16,

    #[serde(rename = "b")]
    pub base_amount: i64,
}

#[derive(Debug)]
pub struct InternalLiquidatePositionTxTarget {
    pub account_index: Target,
    pub market_index: Target,
    pub base_amount: Target,

    // Helpers
    is_ask: BoolTarget,
    zero_price: Target,

    // outputs
    success: BoolTarget,
}

impl InternalLiquidatePositionTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        InternalLiquidatePositionTxTarget {
            account_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            base_amount: builder.add_virtual_target(),

            // Helpers
            is_ask: BoolTarget::default(),
            zero_price: Default::default(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalLiquidatePositionTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_liquidate_position;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[TAKER_ACCOUNT_ID].account_index,
        );

        let active_market_status = builder.constant(F::from_canonical_u8(MARKET_STATUS_ACTIVE));
        builder.conditional_assert_eq(is_enabled, tx_state.market.status, active_market_status);

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );

        // We are going to insert an order to orderbook, so the market must not be full
        let is_order_book_full =
            builder.is_equal(tx_state.market.ask_nonce, tx_state.market.bid_nonce);
        builder.conditional_assert_false(is_enabled, is_order_book_full);

        // Insurance fund can't be partially liquidated
        let insurance_fund_account_type =
            builder.constant(F::from_canonical_u8(INSURANCE_FUND_ACCOUNT_TYPE));
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
            insurance_fund_account_type,
        );

        let execute_transaction = builder.constant_from_u8(EXECUTE_TRANSACTION);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            execute_transaction,
        );

        // BaseAmount - (Must fit in 48 bits) & (can't be zero)
        builder.register_range_check(self.base_amount, ORDER_BASE_AMOUNT_BITS);
        builder.conditional_assert_not_zero(is_enabled, self.base_amount);

        let abs_position =
            builder.biguint_u16_to_target(&tx_state.positions[TAKER_ACCOUNT_ID].position.abs);
        builder.conditional_assert_lte(is_enabled, self.base_amount, abs_position, 64);

        builder.conditional_assert_not_zero_biguint(
            is_enabled,
            &tx_state.risk_infos[TAKER_ACCOUNT_ID]
                .current_risk_parameters
                .maintenance_margin_requirement,
        );

        let partial_liquidation = builder.constant(F::from_canonical_u8(PARTIAL_LIQUIDATION));
        let taker_account_health = tx_state.risk_infos[TAKER_ACCOUNT_ID]
            .current_risk_parameters
            .get_health(builder);
        builder.conditional_assert_eq(is_enabled, taker_account_health, partial_liquidation);

        let one = builder.one();
        self.is_ask = builder.is_equal(
            tx_state.positions[TAKER_ACCOUNT_ID].position.sign.target,
            one,
        );
        let zero_price_big = get_position_zero_price(
            builder,
            &tx_state.positions[TAKER_ACCOUNT_ID],
            &tx_state.market_details,
            &tx_state.risk_infos[TAKER_ACCOUNT_ID].current_risk_parameters,
        );

        let base_amount_big = builder.target_to_biguint(self.base_amount);
        let quote_big = builder.mul_biguint(&base_amount_big, &zero_price_big);

        let max_quote_amount_big =
            builder.constant_biguint(&BigUint::from_u64(MAX_ORDER_QUOTE_AMOUNT).unwrap());
        builder.conditional_assert_lte_biguint(is_enabled, &quote_big, &max_quote_amount_big);

        self.zero_price = builder.biguint_to_target_unsafe(&zero_price_big);
    }
}

impl Apply for InternalLiquidatePositionTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();

        // Set the new register
        let order_nonce = get_next_order_nonce(builder, &tx_state.market, self.is_ask);
        let order_index = get_order_index(builder, tx_state.market.market_index, order_nonce);
        let new_register = BaseRegisterInfoTarget {
            instruction_type: builder.constant_from_u8(INSERT_ORDER),
            market_index: tx_state.market.market_index,
            account_index: self.account_index,
            pending_size: self.base_amount,
            pending_order_index: order_index,
            pending_client_order_index: builder
                .constant(F::from_canonical_i64(NIL_CLIENT_ORDER_INDEX)),
            pending_initial_size: self.base_amount,
            pending_price: self.zero_price,
            pending_nonce: order_nonce,
            pending_is_ask: self.is_ask,
            pending_type: builder.constant_from_u8(LIQUIDATION_ORDER),
            pending_time_in_force: builder.constant_from_u8(IOC),
            pending_reduce_only: one,
            pending_expiry: builder.constant(F::from_canonical_i64(NIL_ORDER_EXPIRY)),
            generic_field_0: builder.zero(),
            pending_trigger_price: builder.constant_i64(NIL_ORDER_TRIGGER_PRICE),
            pending_trigger_status: builder.constant_from_u8(TRIGGER_STATUS_NA),
            pending_to_trigger_order_index0: builder.zero(),
            pending_to_trigger_order_index1: builder.zero(),
            pending_to_cancel_order_index0: builder.zero(),
            generic_field_1: builder.zero(),
            generic_field_2: builder.zero(),
            generic_field_3: builder.zero(),
        };
        tx_state.put_to_instruction_stack_unsafe(builder, self.success, &new_register, 0);

        // Set the matching engine flag
        tx_state.matching_engine_flag = builder.or(tx_state.matching_engine_flag, self.success);
        // Set the update impact prices flag
        tx_state.update_impact_prices_flag =
            builder.or(tx_state.update_impact_prices_flag, self.success);

        // Set the new market
        let is_ask_and_success = builder.and(self.success, self.is_ask);
        let is_bid_and_success = builder.and_not(self.success, self.is_ask);

        tx_state.market.ask_nonce =
            builder.add(tx_state.market.ask_nonce, is_ask_and_success.target);
        tx_state.market.bid_nonce =
            builder.sub(tx_state.market.bid_nonce, is_bid_and_success.target);

        self.success
    }
}

pub trait InternalLiquidatePositionTxTargetWitness<F: PrimeField64> {
    fn set_internal_liquidate_position_tx_target(
        &mut self,
        a: &InternalLiquidatePositionTxTarget,
        b: &InternalLiquidatePositionTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalLiquidatePositionTxTargetWitness<F> for T {
    fn set_internal_liquidate_position_tx_target(
        &mut self,
        a: &InternalLiquidatePositionTxTarget,
        b: &InternalLiquidatePositionTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(a.base_amount, F::from_canonical_i64(b.base_amount))?;

        Ok(())
    }
}
