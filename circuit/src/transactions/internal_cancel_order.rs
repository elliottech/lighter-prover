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
    cancel_child_orders, decrement_locked_balance_for_order, decrement_order_count_in_place,
};
use crate::tx_interface::{Apply, Verify};
use crate::types::account_order::{AccountOrderTarget, select_account_order_target};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::market::{MarketTarget, select_market};
use crate::types::market_details::{MarketDetailsTarget, select_market_details};
use crate::types::order::{
    OrderTarget, get_market_index_and_order_nonce_from_order_index, select_order_target,
};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
pub struct InternalCancelOrderTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "oi")]
    pub order_index: i64,
}

#[derive(Debug)]
pub struct InternalCancelOrderTxTarget {
    pub account_index: Target,
    pub order_index: Target,

    // helpers
    is_empty_order: BoolTarget,
    is_cancel_all_kind: BoolTarget,
    is_register_set: BoolTarget,

    // outputs
    pub success: BoolTarget,
}

impl InternalCancelOrderTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        InternalCancelOrderTxTarget {
            account_index: builder.add_virtual_target(),
            order_index: builder.add_virtual_target(),

            // helpers
            is_empty_order: BoolTarget::default(),
            is_cancel_all_kind: BoolTarget::default(),
            is_register_set: BoolTarget::default(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalCancelOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_cancel_order;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        // Verify that given order index is not a client order index.
        let min_order_index = builder.constant_i64(MIN_ORDER_INDEX);
        builder.conditional_assert_lte(is_enabled, min_order_index, self.order_index, 64);

        // Verify that we load correct market
        let (market_index, _) =
            get_market_index_and_order_nonce_from_order_index(builder, self.order_index);
        builder.conditional_assert_eq(is_enabled, market_index, tx_state.market.market_index);

        // Verify that we load correct account order
        builder.conditional_assert_eq(is_enabled, self.order_index, tx_state.account_order.index_0);
        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.account_order.owner_account_index,
        );

        // Verify that we load correct order from orderbook if needed
        let is_trigger_status_na = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_NA as u64,
        );
        let is_trigger_state_na_and_enabled = builder.and(is_enabled, is_trigger_status_na);
        builder.conditional_assert_eq(
            is_trigger_state_na_and_enabled,
            tx_state.account_order.price,
            tx_state.order.price_index,
        );
        builder.conditional_assert_eq(
            is_trigger_state_na_and_enabled,
            tx_state.account_order.nonce,
            tx_state.order.nonce_index,
        );

        // Account order can be empty only if instruction type is CANCEL_SINGLE_ACCOUNT_ORDER
        let cancel_single_account_order = builder.constant_from_u8(CANCEL_SINGLE_ACCOUNT_ORDER);
        let is_cancel_single_account_order = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_single_account_order,
        );
        self.is_empty_order = tx_state.account_order.is_empty(builder);
        let order_is_empty_and_enabled = builder.and(is_enabled, self.is_empty_order);
        builder.conditional_assert_true(order_is_empty_and_enabled, is_cancel_single_account_order);

        // Verify the register instruction type.
        let cancel_all_account_orders = builder.constant_from_u8(CANCEL_ALL_ACCOUNT_ORDERS);
        let is_cancel_all_account_orders = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_all_account_orders,
        );
        let execute_transaction = builder.constant_from_u8(EXECUTE_TRANSACTION);
        let is_execute_transaction = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            execute_transaction,
        );
        let cancel_position_tied_account_orders =
            builder.constant_from_u8(CANCEL_POSITION_TIED_ACCOUNT_ORDERS);
        let is_cancel_position_tied_account_orders = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_position_tied_account_orders,
        );
        let cancel_all_cross_margin_orders =
            builder.constant_from_u8(CANCEL_ALL_CROSS_MARGIN_ORDERS);
        let is_cancel_all_cross_margin_orders = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_all_cross_margin_orders,
        );
        let cancel_all_isolated_margin_orders =
            builder.constant_from_u8(CANCEL_ALL_ISOLATED_MARGIN_ORDERS);
        let is_cancel_all_isolated_margin_orders = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_all_isolated_margin_orders,
        );
        let cancel_all_market_account_orders =
            builder.constant_from_u8(CANCEL_ALL_MARKET_ACCOUNT_ORDERS);
        let is_cancel_all_market_account_orders = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_all_market_account_orders,
        );
        self.is_cancel_all_kind = builder.multi_or(&[
            is_cancel_all_account_orders,
            is_cancel_position_tied_account_orders,
            is_cancel_all_cross_margin_orders,
            is_cancel_all_isolated_margin_orders,
            is_cancel_all_market_account_orders,
        ]);
        self.is_register_set =
            builder.multi_or(&[self.is_cancel_all_kind, is_cancel_single_account_order]);
        let is_correct_instruction_type =
            builder.multi_or(&[is_execute_transaction, self.is_register_set]);
        builder.conditional_assert_true(is_enabled, is_correct_instruction_type);

        // Verify EXECUTE_TRANSACTION mode.
        {
            let is_order_expired = builder.is_lte(
                tx_state.account_order.expiry,
                tx_state.block_timestamp,
                TIMESTAMP_BITS,
            );
            let market_expired_status = builder.constant_from_u8(MARKET_STATUS_EXPIRED);
            let is_market_expired = builder.is_equal(tx_state.market.status, market_expired_status);

            let is_valid_execute_transaction =
                builder.multi_or(&[is_order_expired, is_market_expired]);
            let is_enabled_and_execute_transaction =
                builder.and(is_enabled, is_execute_transaction);
            builder.conditional_assert_true(
                is_enabled_and_execute_transaction,
                is_valid_execute_transaction,
            );
        }

        // Verify common register data
        {
            let is_register_set_and_enabled = builder.and(self.is_register_set, is_enabled);
            builder.conditional_assert_eq(
                is_register_set_and_enabled,
                tx_state.register_stack[0].account_index,
                self.account_index,
            );
        }

        // Verify CANCEL_SINGLE_ACCOUNT_ORDER mode.
        {
            let is_enabled_and_cancel_single_account_order =
                builder.and(is_enabled, is_cancel_single_account_order);
            builder.conditional_assert_eq(
                is_enabled_and_cancel_single_account_order,
                tx_state.register_stack[0].pending_order_index,
                self.order_index,
            );
            builder.conditional_assert_eq(
                is_enabled_and_cancel_single_account_order,
                tx_state.register_stack[0].market_index,
                market_index,
            );
            // order should not have subsequent orders to trigger
            let to_trigger_order_index_0_is_empty =
                builder.is_zero(tx_state.account_order.to_trigger_order_index0);
            let to_trigger_order_index_1_is_empty =
                builder.is_zero(tx_state.account_order.to_trigger_order_index1);
            let trigger_order_indices_are_empty = builder.and(
                to_trigger_order_index_0_is_empty,
                to_trigger_order_index_1_is_empty,
            );
            builder.conditional_assert_true(
                is_enabled_and_cancel_single_account_order,
                trigger_order_indices_are_empty,
            );
        }

        // Verify CANCEL_POSITION_TIED_ACCOUNT_ORDERS mode
        {
            let is_enabled_and_cancel_position_tied_account_orders =
                builder.and(is_enabled, is_cancel_position_tied_account_orders);
            builder.conditional_assert_eq(
                is_enabled_and_cancel_position_tied_account_orders,
                tx_state.register_stack[0].market_index,
                market_index,
            );
            builder.conditional_assert_not_zero(
                is_enabled_and_cancel_position_tied_account_orders,
                tx_state.account_order.reduce_only,
            );
            let trigger_status_parent_order = builder.constant_from_u8(TRIGGER_STATUS_PARENT_ORDER);
            builder.conditional_assert_not_eq(
                is_enabled_and_cancel_position_tied_account_orders,
                trigger_status_parent_order,
                tx_state.account_order.trigger_status,
            );
        }

        // Verify CANCEL_ALL_ISOLATED_MARGIN_ORDERS mode
        {
            let is_enabled_and_cancel_all_isolated_margin_orders =
                builder.and(is_enabled, is_cancel_all_isolated_margin_orders);
            builder.conditional_assert_eq(
                is_enabled_and_cancel_all_isolated_margin_orders,
                tx_state.register_stack[0].market_index,
                market_index,
            );
            let margin_mode_isolated = builder.constant_usize(ISOLATED_MARGIN);
            builder.conditional_assert_eq(
                is_enabled_and_cancel_all_isolated_margin_orders,
                tx_state.positions[OWNER_ACCOUNT_ID].margin_mode,
                margin_mode_isolated,
            );
        }

        // Verify CANCEL_ALL_CROSS_MARGIN_ORDERS mode
        {
            let is_enabled_and_cancel_all_cross_margin_orders =
                builder.and(is_enabled, is_cancel_all_cross_margin_orders);
            let margin_mode_cross = builder.constant_usize(CROSS_MARGIN);
            builder.conditional_assert_eq(
                is_enabled_and_cancel_all_cross_margin_orders,
                tx_state.positions[OWNER_ACCOUNT_ID].margin_mode,
                margin_mode_cross,
            );
        }

        // Verify CANCEL_ALL_MARKET_ACCOUNT_ORDERS mode
        {
            let is_enabled_and_cancel_all_market =
                builder.and(is_enabled, is_cancel_all_market_account_orders);
            builder.conditional_assert_eq(
                is_enabled_and_cancel_all_market,
                tx_state.register_stack[0].market_index,
                market_index,
            );
        }

        self.success = is_enabled;
    }
}

impl Apply for InternalCancelOrderTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        // Update register and reduce pending size if we are cancelling some group of orders
        let update_register_flag = builder.and(self.success, self.is_cancel_all_kind);
        tx_state.register_stack[0].pending_size = builder.sub(
            tx_state.register_stack[0].pending_size,
            update_register_flag.target,
        );
        let pending_size_is_zero = builder.is_zero(tx_state.register_stack[0].pending_size);
        let pop_register_flag_1 = builder.and(update_register_flag, pending_size_is_zero);

        // Pop register if we are cancelling a single order
        let cancel_single_account_order = builder.constant_from_u8(CANCEL_SINGLE_ACCOUNT_ORDER);
        let is_cancel_single_account_order = builder.is_equal(
            tx_state.register_stack[0].instruction_type,
            cancel_single_account_order,
        );
        let pop_register_flag_2 = builder.and(self.success, is_cancel_single_account_order);

        // Pop register
        let pop_register_flag = builder.or(pop_register_flag_1, pop_register_flag_2);
        tx_state
            .register_stack
            .pop_front(builder, pop_register_flag);

        // Only update the state if order initially wasn't empty
        let update_state = builder.and_not(self.success, self.is_empty_order);

        // Update order counts
        decrement_order_count_in_place(
            builder,
            tx_state,
            OWNER_ACCOUNT_ID,
            update_state,
            tx_state.account_order.trigger_status,
            tx_state.account_order.reduce_only,
        );

        let is_spot_market =
            builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_SPOT);
        let is_limit_order =
            builder.is_equal_constant(tx_state.account_order.order_type, LIMIT_ORDER as u64);
        let decrement_locked_balance_flag =
            builder.multi_and(&[self.success, is_spot_market, is_limit_order]);
        decrement_locked_balance_for_order(
            builder,
            decrement_locked_balance_flag,
            &tx_state.account_order,
            &tx_state.market,
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID],
        );

        // Handle market expiration
        let market_expired_status = builder.constant_from_u8(MARKET_STATUS_EXPIRED);
        let is_market_expired = builder.is_equal(tx_state.market.status, market_expired_status);
        let is_market_has_no_order = builder.is_zero(tx_state.market.total_order_count);
        let is_market_has_no_position = builder.is_zero(tx_state.market_details.open_interest);
        let is_expired_market_is_empty_and_enabled = builder.multi_and(&[
            update_state,
            is_market_expired,
            is_market_has_no_order,
            is_market_has_no_position,
        ]);
        let empty_market_details = MarketDetailsTarget::empty(builder);
        let empty_order_book_tree_root = builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT);
        let empty_market = MarketTarget::empty(
            builder,
            tx_state.market.market_index,
            tx_state.market.perps_market_index,
            empty_order_book_tree_root,
        );
        tx_state.market_details = select_market_details(
            builder,
            is_expired_market_is_empty_and_enabled,
            &empty_market_details,
            &tx_state.market_details,
        );
        tx_state.market = select_market(
            builder,
            is_expired_market_is_empty_and_enabled,
            &empty_market,
            &tx_state.market,
        );

        // Trigger cancel child orders if instruction type != cancel all kind
        let cancel_child_orders_flag = builder.and_not(update_state, self.is_cancel_all_kind);
        cancel_child_orders(
            builder,
            cancel_child_orders_flag,
            tx_state,
            tx_state.market.market_index,
            tx_state.account_order.owner_account_index,
            tx_state.account_order.to_trigger_order_index0,
            tx_state.account_order.to_trigger_order_index1,
            1,
        );

        // Cancel order - account order leaf
        let empty_account_order = AccountOrderTarget::empty(
            builder,
            tx_state.account_order.index_0,
            tx_state.account_order.index_1,
            tx_state.account_order.owner_account_index,
        );
        tx_state.account_order = select_account_order_target(
            builder,
            update_state,
            &empty_account_order,
            &tx_state.account_order,
        );

        // Cancel order - order leaf. If order is already empty, this will be a no-op.
        let empty_order = OrderTarget::empty(
            builder,
            tx_state.order.price_index,
            tx_state.order.nonce_index,
        );
        tx_state.order = select_order_target(builder, update_state, &empty_order, &tx_state.order);

        // Set update_impact_prices_flag
        tx_state.update_impact_prices_flag =
            builder.or(update_state, tx_state.update_impact_prices_flag);

        self.success
    }
}

pub trait InternalCancelOrderTxTargetWitness<F: PrimeField64> {
    fn set_internal_cancel_order_tx_target(
        &mut self,
        a: &InternalCancelOrderTxTarget,
        b: &InternalCancelOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalCancelOrderTxTargetWitness<F> for T {
    fn set_internal_cancel_order_tx_target(
        &mut self,
        a: &InternalCancelOrderTxTarget,
        b: &InternalCancelOrderTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.order_index, F::from_canonical_i64(b.order_index))?;

        Ok(())
    }
}
