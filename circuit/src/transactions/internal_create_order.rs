// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::matching_engine::{
    decrement_order_count_in_place, get_next_order_nonce, is_not_valid_reduce_only_direction,
};
use crate::tx_interface::{Apply, Verify};
use crate::types::account_order::{AccountOrderTarget, select_account_order_target};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::order::{get_market_index_and_order_nonce_from_order_index, get_order_index};
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct InternalCreateOrderTx {
    #[serde(rename = "a")]
    pub account_index: i64,

    #[serde(rename = "o")]
    pub order_index: i64,

    #[serde(rename = "b")]
    pub base_amount: i64,

    #[serde(rename = "p")]
    pub price: u32,
}

impl Default for InternalCreateOrderTx {
    fn default() -> Self {
        InternalCreateOrderTx::empty()
    }
}

impl InternalCreateOrderTx {
    pub fn empty() -> Self {
        InternalCreateOrderTx {
            account_index: 0,
            order_index: 0,
            base_amount: 0,
            price: 0,
        }
    }
}

#[derive(Debug)]
pub struct InternalCreateOrderTxTarget {
    pub account_index: Target,
    pub order_index: Target,
    pub base_amount: Target,
    pub price: Target,

    // helpers
    is_execute_transaction: BoolTarget,
    is_trigger_child_order: BoolTarget,
    is_trigger_status_twap: BoolTarget,
    is_empty_order: BoolTarget,

    // outputs
    success: BoolTarget,
}

impl InternalCreateOrderTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        InternalCreateOrderTxTarget {
            account_index: builder.add_virtual_target(),
            order_index: builder.add_virtual_target(),
            base_amount: builder.add_virtual_target(),
            price: builder.add_virtual_target(),

            // helpers
            is_execute_transaction: BoolTarget::default(),
            is_trigger_child_order: BoolTarget::default(),
            is_trigger_status_twap: BoolTarget::default(),
            is_empty_order: BoolTarget::default(),

            // outputs
            success: BoolTarget::default(),
        }
    }

    fn get_register_for_twap_order(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
        next_order_nonce: Target,
    ) -> BaseRegisterInfoTarget {
        let (generic_field_1, generic_field_2, generic_field_3) = tx_state
            .account_order
            .get_register_generic_fields_from_order(builder);
        BaseRegisterInfoTarget {
            instruction_type: builder.constant_from_u8(INSERT_ORDER),
            market_index: tx_state.market.market_index,
            account_index: self.account_index,
            pending_size: self.base_amount,
            pending_order_index: get_order_index(
                builder,
                tx_state.market.market_index,
                next_order_nonce,
            ),
            pending_client_order_index: builder.constant_i64(NIL_CLIENT_ORDER_INDEX),
            pending_initial_size: self.base_amount,
            pending_price: self.price,
            pending_nonce: next_order_nonce,
            pending_is_ask: tx_state.account_order.is_ask,
            pending_type: builder.constant_from_u8(TWAP_SUB_ORDER),
            pending_time_in_force: builder.constant_from_u8(IOC),
            pending_reduce_only: tx_state.account_order.reduce_only,
            pending_expiry: builder.constant_i64(NIL_ORDER_EXPIRY),
            generic_field_0: builder.zero(),
            pending_trigger_price: builder.zero(),
            pending_trigger_status: builder.zero(),
            pending_to_trigger_order_index0: builder.zero(),
            pending_to_trigger_order_index1: builder.zero(),
            pending_to_cancel_order_index0: builder.zero(),
            generic_field_1,
            generic_field_2,
            generic_field_3,
        }
    }

    fn get_register_for_conditional_order(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
        next_order_nonce: Target,
    ) -> BaseRegisterInfoTarget {
        let position_tied_order_base_amount = tx_state.positions[TAKER_ACCOUNT_ID]
            .calculate_position_tied_order_base_amount(
                builder,
                tx_state.market_details.quote_multiplier,
                self.price,
                tx_state.market.order_quote_limit,
            );
        let base_amount_is_zero = builder.is_zero(self.base_amount);
        let base_size = builder.select(
            base_amount_is_zero,
            position_tied_order_base_amount,
            self.base_amount,
        );

        let trigger_status_na = builder.constant_from_u8(TRIGGER_STATUS_NA);

        let (generic_field_1, generic_field_2, generic_field_3) = tx_state
            .account_order
            .get_register_generic_fields_from_order(builder);

        BaseRegisterInfoTarget {
            instruction_type: builder.constant_from_u8(INSERT_ORDER),
            market_index: tx_state.market.market_index,
            account_index: self.account_index,
            pending_size: base_size,
            pending_order_index: tx_state.account_order.order_index,
            pending_client_order_index: tx_state.account_order.client_order_index,
            pending_initial_size: base_size,
            pending_price: tx_state.account_order.price,
            pending_nonce: next_order_nonce,
            pending_is_ask: tx_state.account_order.is_ask,
            pending_type: tx_state.account_order.order_type,
            pending_time_in_force: tx_state.account_order.time_in_force,
            pending_reduce_only: tx_state.account_order.reduce_only,
            pending_expiry: tx_state.account_order.expiry,
            generic_field_0: builder.zero(),
            pending_trigger_price: tx_state.account_order.trigger_price,
            pending_trigger_status: trigger_status_na,
            pending_to_trigger_order_index0: tx_state.account_order.to_trigger_order_index0,
            pending_to_trigger_order_index1: tx_state.account_order.to_trigger_order_index1,
            pending_to_cancel_order_index0: tx_state.account_order.to_cancel_order_index0,
            generic_field_1,
            generic_field_2,
            generic_field_3,
        }
    }

    fn is_triggered_conditional_order(
        &self,
        builder: &mut Builder,
        is_ask: BoolTarget,
        tx_state: &TxState,
        is_stop_loss_order: BoolTarget,
        is_stop_loss_limit_order: BoolTarget,
        is_take_profit_order: BoolTarget,
        is_take_profit_limit_order: BoolTarget,
    ) -> BoolTarget {
        let one = builder.one();
        let neg_one = builder.neg(one);

        let is_stop_loss_variation = builder.or(is_stop_loss_order, is_stop_loss_limit_order);
        let is_ask_stop_loss_variation = builder.and(is_ask, is_stop_loss_variation);
        let is_bid_stop_loss_variation = builder.and_not(is_stop_loss_variation, is_ask);
        let is_take_profit_variation = builder.or(is_take_profit_order, is_take_profit_limit_order);
        let is_ask_take_profit_variation = builder.and(is_ask, is_take_profit_variation);
        let is_bid_take_profit_variation = builder.and_not(is_take_profit_variation, is_ask);
        let mark_cmp_trigger_price = builder.cmp(
            tx_state.market_details.mark_price,
            tx_state.account_order.trigger_price,
            ORDER_PRICE_BITS,
        );
        let mark_gt_trigger_price = builder.is_equal(mark_cmp_trigger_price.target, one);
        let mark_lt_trigger_price = builder.is_equal(mark_cmp_trigger_price.target, neg_one);

        let is_not_triggered_with_gt =
            builder.or(is_ask_stop_loss_variation, is_bid_take_profit_variation);
        let is_not_triggered_with_gt = builder.and(is_not_triggered_with_gt, mark_gt_trigger_price);
        let is_not_triggered_with_lt =
            builder.or(is_ask_take_profit_variation, is_bid_stop_loss_variation);
        let is_not_triggered_with_lt = builder.and(is_not_triggered_with_lt, mark_lt_trigger_price);

        let is_pending_conditional_order =
            builder.or(is_not_triggered_with_gt, is_not_triggered_with_lt);

        builder.not(is_pending_conditional_order)
    }

    fn modify_order_count_in_cache_for_trigger_status(
        &self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        tx_state: &mut TxState,
        old_trigger_status: Target,
    ) {
        let trigger_status_changed =
            builder.is_not_equal(tx_state.account_order.trigger_status, old_trigger_status);
        let reduce_only_flag = builder.is_not_zero(tx_state.account_order.reduce_only);
        let is_enabled = builder.multi_and(&[is_enabled, trigger_status_changed, reduce_only_flag]);

        let trigger_status_parent_order = builder.constant_from_u8(TRIGGER_STATUS_PARENT_ORDER);
        let old_trigger_status_neq_parent_order =
            builder.is_not_equal(old_trigger_status, trigger_status_parent_order);
        let new_trigger_status_neq_parent_order = builder.is_not_equal(
            tx_state.account_order.trigger_status,
            trigger_status_parent_order,
        );

        let total_position_tied_order_count_delta = builder.sub(
            new_trigger_status_neq_parent_order.target,
            old_trigger_status_neq_parent_order.target,
        );

        tx_state.positions[TAKER_ACCOUNT_ID].total_position_tied_order_count = builder.mul_add(
            is_enabled.target,
            total_position_tied_order_count_delta,
            tx_state.positions[TAKER_ACCOUNT_ID].total_position_tied_order_count,
        );
    }

    fn get_register_for_to_cancel_order(
        &self,
        builder: &mut Builder,
        market_index: Target,
        to_cancel_order_index: Target,
    ) -> BaseRegisterInfoTarget {
        BaseRegisterInfoTarget {
            instruction_type: builder.constant_from_u8(CANCEL_SINGLE_ACCOUNT_ORDER),
            market_index,
            account_index: self.account_index,
            pending_order_index: to_cancel_order_index,
            ..BaseRegisterInfoTarget::empty(builder)
        }
    }
}

impl Verify for InternalCreateOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let one = builder.one();
        let neg_one = builder.neg_one();

        let is_enabled = tx_type.is_internal_create_order;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        builder.conditional_assert_eq(
            is_enabled,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.account_order.owner_account_index,
        );

        // Verify that given order index is not a client order index
        let min_order_index = builder.constant_i64(MIN_ORDER_INDEX);
        builder.conditional_assert_lte(is_enabled, min_order_index, self.order_index, 64);

        let (market_index, _) =
            get_market_index_and_order_nonce_from_order_index(builder, self.order_index);
        builder.conditional_assert_eq(is_enabled, market_index, tx_state.market.market_index);

        let active_market_status = builder.constant_from_u8(MARKET_STATUS_ACTIVE);
        builder.conditional_assert_eq(is_enabled, tx_state.market.status, active_market_status);

        // Spot only supports twap
        let is_spot = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_SPOT);
        self.is_trigger_status_twap = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_TWAP as u64,
        );
        let is_spot_but_not_twap = builder.and_not(is_spot, self.is_trigger_status_twap);
        builder.conditional_assert_false(is_enabled, is_spot_but_not_twap);

        builder.conditional_assert_eq(is_enabled, self.order_index, tx_state.account_order.index_0);

        let is_order_book_full =
            builder.is_equal(tx_state.market.ask_nonce, tx_state.market.bid_nonce);
        builder.conditional_assert_false(is_enabled, is_order_book_full);

        builder.register_range_check(self.base_amount, ORDER_SIZE_BITS);
        builder.register_range_check(self.price, ORDER_PRICE_BITS);

        // Register instruction types
        let execute_transaction = builder.constant_from_u8(EXECUTE_TRANSACTION);
        self.is_execute_transaction = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            execute_transaction,
        );
        let trigger_child_order = builder.constant_from_u8(TRIGGER_CHILD_ORDER);
        self.is_trigger_child_order = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            trigger_child_order,
        );
        let is_valid_instruction_type =
            builder.or(self.is_execute_transaction, self.is_trigger_child_order);
        builder.conditional_assert_true(is_enabled, is_valid_instruction_type);

        // Flags
        self.is_empty_order = tx_state.account_order.is_empty(builder);

        let is_stop_loss_order =
            builder.is_equal_constant(tx_state.account_order.order_type, STOP_LOSS_ORDER as u64);
        let is_stop_loss_limit_order = builder.is_equal_constant(
            tx_state.account_order.order_type,
            STOP_LOSS_LIMIT_ORDER as u64,
        );
        let is_take_profit_order =
            builder.is_equal_constant(tx_state.account_order.order_type, TAKE_PROFIT_ORDER as u64);
        let is_take_profit_limit_order = builder.is_equal_constant(
            tx_state.account_order.order_type,
            TAKE_PROFIT_LIMIT_ORDER as u64,
        );

        /* EXECUTE_TRANSACTION */
        {
            let execute_transaction_flag = builder.and(self.success, self.is_execute_transaction);

            // order should be present
            builder.conditional_assert_false(execute_transaction_flag, self.is_empty_order);

            let is_trigger_status_mark_price = builder.is_equal_constant(
                tx_state.account_order.trigger_status,
                TRIGGER_STATUS_MARK_PRICE as u64,
            );
            // TRIGGER_STATUS_MARK_PRICE
            {
                let trigger_status_mark_price_flag =
                    builder.and(execute_transaction_flag, is_trigger_status_mark_price);

                builder.conditional_assert_false(trigger_status_mark_price_flag, is_spot);

                let is_triggered_conditional_order = self.is_triggered_conditional_order(
                    builder,
                    tx_state.account_order.is_ask,
                    tx_state,
                    is_stop_loss_order,
                    is_stop_loss_limit_order,
                    is_take_profit_order,
                    is_take_profit_limit_order,
                );
                builder.conditional_assert_true(
                    trigger_status_mark_price_flag,
                    is_triggered_conditional_order,
                );
                builder.conditional_assert_eq(
                    trigger_status_mark_price_flag,
                    tx_state.account_order.remaining_base_amount,
                    self.base_amount,
                );
                builder.conditional_assert_eq(
                    trigger_status_mark_price_flag,
                    tx_state.account_order.price,
                    self.price,
                );
            }

            // TRIGGER_STATUS_TWAP
            {
                let is_ask_twap_order =
                    builder.and(tx_state.account_order.is_ask, self.is_trigger_status_twap);
                let is_bid_twap_order =
                    builder.and_not(self.is_trigger_status_twap, tx_state.account_order.is_ask);
                let account_order_price_cmp_tx_price =
                    builder.cmp(tx_state.account_order.price, self.price, ORDER_PRICE_BITS);
                let trigger_status_twap_flag =
                    builder.and(execute_transaction_flag, self.is_trigger_status_twap);
                // Check for twap orders:
                // - if ask(sell) order.price <= tx.price (-1 or 0)
                // - if bid(buy) order.price >= tx.price (0 or 1)
                // - !(txInfo.BaseAmount > order.RemainingBaseAmount && txInfo.BaseAmount > types.MinOrderBaseAmount)
                let is_ask_twap_flag = builder.and(trigger_status_twap_flag, is_ask_twap_order);
                builder.conditional_assert_not_eq(
                    is_ask_twap_flag,
                    account_order_price_cmp_tx_price.target,
                    one,
                );
                let is_bid_twap_flag = builder.and(trigger_status_twap_flag, is_bid_twap_order);
                builder.conditional_assert_not_eq(
                    is_bid_twap_flag,
                    account_order_price_cmp_tx_price.target,
                    neg_one,
                );
                let tx_base_amount_gt_order = builder.is_gt(
                    self.base_amount,
                    tx_state.account_order.remaining_base_amount,
                    ORDER_SIZE_BITS,
                );
                let tx_base_amount_not_zero = builder.is_not_zero(self.base_amount);
                let tx_base_amount_not_one = builder.is_not_equal(self.base_amount, one);
                let tx_base_amount_gt_min_order_base_amount =
                    builder.and(tx_base_amount_not_zero, tx_base_amount_not_one);
                let should_be_false = builder.and(
                    tx_base_amount_gt_order,
                    tx_base_amount_gt_min_order_base_amount,
                );
                builder.conditional_assert_false(trigger_status_twap_flag, should_be_false);
            }

            let is_trigger_status_twap_or_mark_price =
                builder.or(self.is_trigger_status_twap, is_trigger_status_mark_price);
            builder.conditional_assert_true(
                execute_transaction_flag,
                is_trigger_status_twap_or_mark_price,
            );
        }

        /* TRIGGER_CHILD_ORDER */
        {
            let trigger_child_order_flag = builder.and(is_enabled, self.is_trigger_child_order);

            builder.conditional_assert_false(trigger_child_order_flag, is_spot);

            let trigger_status_parent_order = builder.constant_from_u8(TRIGGER_STATUS_PARENT_ORDER);
            let is_trigger_status_not_parent_order = builder.is_not_equal(
                tx_state.account_order.trigger_status,
                trigger_status_parent_order,
            );
            let should_be_false =
                builder.and_not(is_trigger_status_not_parent_order, self.is_empty_order);
            builder.conditional_assert_false(trigger_child_order_flag, should_be_false);
            builder.conditional_assert_eq(
                trigger_child_order_flag,
                tx_state.register_stack[0].pending_size,
                self.base_amount,
            );
            builder.conditional_assert_eq(
                trigger_child_order_flag,
                tx_state.register_stack[0].account_index,
                self.account_index,
            );
            builder.conditional_assert_eq(
                trigger_child_order_flag,
                tx_state.register_stack[0].pending_order_index,
                self.order_index,
            );
            builder.conditional_assert_zero(trigger_child_order_flag, self.price);
        }
    }
}

impl Apply for InternalCreateOrderTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();
        let _false = builder._false();

        // Calculate invalid reduce only and expiry flags
        let is_ask = tx_state.account_order.is_ask;
        let is_invalid_reduce_only_direction = is_not_valid_reduce_only_direction(
            builder,
            tx_state.positions[TAKER_ACCOUNT_ID].position.sign,
            is_ask,
        );
        let is_invalid_reduce_only = builder.and(
            is_invalid_reduce_only_direction,
            BoolTarget::new_unsafe(tx_state.account_order.reduce_only),
        );
        let is_order_expired = builder.is_lte(
            tx_state.account_order.expiry,
            tx_state.block_timestamp,
            TIMESTAMP_BITS,
        );
        let should_trigger_dead_mans_switch = tx_state.accounts[OWNER_ACCOUNT_ID]
            .should_dms_be_triggered(builder, tx_state.block_timestamp);
        let cancel_order_flag = builder.multi_or(&[
            is_order_expired,
            should_trigger_dead_mans_switch,
            is_invalid_reduce_only,
        ]);
        let cancel_order_flag = builder.and(self.success, cancel_order_flag);

        /* TRIGGER_CHILD_ORDER */
        {
            let mut trigger_child_order_flag =
                builder.and(self.success, self.is_trigger_child_order);
            let pop_register_flag = trigger_child_order_flag;
            trigger_child_order_flag =
                builder.and_not(trigger_child_order_flag, self.is_empty_order);

            // Handle invalid reduce only or expiry - Decrement order count
            let cancel_order_flag = builder.and(trigger_child_order_flag, cancel_order_flag);
            decrement_order_count_in_place(
                builder,
                tx_state,
                TAKER_ACCOUNT_ID,
                cancel_order_flag,
                tx_state.account_order.trigger_status,
                tx_state.account_order.reduce_only,
            );

            // Modify Account Order
            let trigger_status_mark_price = builder.constant_from_u8(TRIGGER_STATUS_MARK_PRICE);
            let old_trigger_status = tx_state.account_order.trigger_status;
            tx_state.account_order.trigger_status = builder.select(
                trigger_child_order_flag,
                trigger_status_mark_price,
                tx_state.account_order.trigger_status,
            );
            tx_state.account_order.initial_base_amount = builder.select(
                trigger_child_order_flag,
                tx_state.register_stack[0].pending_size,
                tx_state.account_order.initial_base_amount,
            );
            tx_state.account_order.remaining_base_amount = builder.select(
                trigger_child_order_flag,
                tx_state.register_stack[0].pending_size,
                tx_state.account_order.remaining_base_amount,
            );

            // Handle invalid reduce only or expiry - Cancel account order
            let empty_account_order = AccountOrderTarget::empty(
                builder,
                tx_state.account_order.index_0,
                tx_state.account_order.index_1,
                tx_state.account_order.owner_account_index,
            );
            tx_state.account_order = select_account_order_target(
                builder,
                cancel_order_flag,
                &empty_account_order,
                &tx_state.account_order,
            );

            // Handle when reduce only and expiry are valid
            let modify_order_count = builder.and_not(trigger_child_order_flag, cancel_order_flag);
            self.modify_order_count_in_cache_for_trigger_status(
                builder,
                modify_order_count,
                tx_state,
                old_trigger_status,
            );

            // Pop Register
            tx_state
                .register_stack
                .pop_front(builder, pop_register_flag);
        }

        /* EXECUTE_TRANSACTION */
        {
            let mut execute_transaction_flag =
                builder.and(self.success, self.is_execute_transaction);

            // Handle invalid reduce only or expiry
            {
                let cancel_order_flag = builder.and(execute_transaction_flag, cancel_order_flag);
                decrement_order_count_in_place(
                    builder,
                    tx_state,
                    TAKER_ACCOUNT_ID,
                    cancel_order_flag,
                    tx_state.account_order.trigger_status,
                    tx_state.account_order.reduce_only,
                );
                let empty_account_order = AccountOrderTarget::empty(
                    builder,
                    tx_state.account_order.index_0,
                    tx_state.account_order.index_1,
                    tx_state.account_order.owner_account_index,
                );
                tx_state.account_order = select_account_order_target(
                    builder,
                    cancel_order_flag,
                    &empty_account_order,
                    &tx_state.account_order,
                );

                // Return
                execute_transaction_flag =
                    builder.select_bool(cancel_order_flag, _false, execute_transaction_flag);
            }

            let next_order_nonce = get_next_order_nonce(builder, &tx_state.market, is_ask);

            // Set new market
            {
                let ask_nonce_plus_one = builder.add(tx_state.market.ask_nonce, one);
                let bid_nonce_minus_one = builder.sub(tx_state.market.bid_nonce, one);
                let new_ask_nonce =
                    builder.select(is_ask, ask_nonce_plus_one, tx_state.market.ask_nonce);
                let new_bid_nonce =
                    builder.select(is_ask, tx_state.market.bid_nonce, bid_nonce_minus_one);
                tx_state.market.ask_nonce = builder.select(
                    execute_transaction_flag,
                    new_ask_nonce,
                    tx_state.market.ask_nonce,
                );
                tx_state.market.bid_nonce = builder.select(
                    execute_transaction_flag,
                    new_bid_nonce,
                    tx_state.market.bid_nonce,
                );
            }

            // TRIGGER_STATUS_TWAP
            {
                let twap_flag = builder.and(execute_transaction_flag, self.is_trigger_status_twap);

                let subtract_from_order = builder.mul(twap_flag.target, self.base_amount);
                tx_state.account_order.remaining_base_amount = builder.sub(
                    tx_state.account_order.remaining_base_amount,
                    subtract_from_order,
                );

                let is_remaining_base_amount_zero =
                    builder.is_zero(tx_state.account_order.remaining_base_amount);

                // Build the register
                let register_for_twap_order =
                    self.get_register_for_twap_order(builder, tx_state, next_order_nonce);
                tx_state.put_to_instruction_stack_unsafe(
                    builder,
                    twap_flag,
                    &register_for_twap_order,
                    0,
                );

                // Cancel twap order if filled
                let cancel_twap_order = builder.and(twap_flag, is_remaining_base_amount_zero);
                decrement_order_count_in_place(
                    builder,
                    tx_state,
                    TAKER_ACCOUNT_ID,
                    cancel_twap_order,
                    tx_state.account_order.trigger_status,
                    tx_state.account_order.reduce_only,
                );
                let empty_account_order = AccountOrderTarget::empty(
                    builder,
                    tx_state.account_order.index_0,
                    tx_state.account_order.index_1,
                    tx_state.account_order.owner_account_index,
                );
                tx_state.account_order = select_account_order_target(
                    builder,
                    cancel_twap_order,
                    &empty_account_order,
                    &tx_state.account_order,
                );
            }

            // TRIGGER_STATUS_MARK_PRICE
            {
                let conditional_flag =
                    builder.and_not(execute_transaction_flag, self.is_trigger_status_twap);

                // Set the register for twap or conditional order
                let register_for_conditional_order =
                    self.get_register_for_conditional_order(builder, tx_state, next_order_nonce);
                tx_state.put_to_instruction_stack_unsafe(
                    builder,
                    conditional_flag,
                    &register_for_conditional_order,
                    1,
                );

                let to_cancel_order_index0 = tx_state.account_order.to_cancel_order_index0;

                // Cancel conditional order
                decrement_order_count_in_place(
                    builder,
                    tx_state,
                    TAKER_ACCOUNT_ID,
                    conditional_flag,
                    tx_state.account_order.trigger_status,
                    tx_state.account_order.reduce_only,
                );
                let empty_account_order = AccountOrderTarget::empty(
                    builder,
                    tx_state.account_order.index_0,
                    tx_state.account_order.index_1,
                    tx_state.account_order.owner_account_index,
                );
                tx_state.account_order = select_account_order_target(
                    builder,
                    conditional_flag,
                    &empty_account_order,
                    &tx_state.account_order,
                );

                // Push to register if to_cancel_order_index_0 != 0
                let register_for_to_cancel_order = self.get_register_for_to_cancel_order(
                    builder,
                    tx_state.market.market_index,
                    to_cancel_order_index0,
                );
                let to_cancel_order_index_not_empty = builder.is_not_zero(to_cancel_order_index0);
                let push_to_register_stack_flag =
                    builder.and(conditional_flag, to_cancel_order_index_not_empty);
                tx_state.put_to_instruction_stack_unsafe(
                    builder,
                    push_to_register_stack_flag,
                    &register_for_to_cancel_order,
                    0,
                );
            }
        }

        let is_success_and_non_empty = builder.and_not(self.success, self.is_empty_order);
        tx_state.update_impact_prices_flag =
            builder.or(is_success_and_non_empty, tx_state.update_impact_prices_flag);

        self.success
    }
}

pub trait InternalCreateOrderTxTargetWitness<F: PrimeField64> {
    fn set_internal_create_order_tx_target(
        &mut self,
        a: &InternalCreateOrderTxTarget,
        b: &InternalCreateOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalCreateOrderTxTargetWitness<F> for T {
    fn set_internal_create_order_tx_target(
        &mut self,
        a: &InternalCreateOrderTxTarget,
        b: &InternalCreateOrderTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.order_index, F::from_canonical_i64(b.order_index))?;
        self.set_target(a.base_amount, F::from_canonical_i64(b.base_amount))?;
        self.set_target(a.price, F::from_canonical_u32(b.price))?;

        Ok(())
    }
}
