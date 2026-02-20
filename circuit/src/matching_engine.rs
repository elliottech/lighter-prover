// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module (spot-specific paths):
// 1. unified accounts skip spot balance updates for margin assets.
// 2. unified accounts apply spot margin-asset deltas to collateral.
// 3. spot trades are canceled when the account risk change is invalid.

use plonky2::field::types::Field;
use plonky2::iop::target::{BoolTarget, Target};

use crate::apply_trade::{
    ApplySpotTradeParams, ApplyTradeParams, apply_perps_trade, apply_spot_trade,
};
use crate::bigint::big_u16::CircuitBuilderBiguint16;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::hints::CircuitBuilderHints;
use crate::liquidation::{get_asset_balance_const, get_available_collateral};
use crate::order_book_tree_helpers::order_indexes_to_merkle_path;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::types::account::AccountTarget;
use crate::types::account_asset::AccountAssetTarget;
use crate::types::account_order::{AccountOrderTarget, select_account_order_target};
use crate::types::account_position::AccountPositionTarget;
use crate::types::config::{BIG_U96_LIMBS, BIG_U128_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::market::MarketTarget;
use crate::types::order::{
    OrderTarget, get_market_index_and_order_nonce_from_order_index, select_order_target,
};
use crate::types::order_book_node::OrderBookNodeTarget;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::risk_info::{RiskInfoTarget, RiskParametersTarget};
use crate::types::tx_state::TxState;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

pub fn get_order_book_path_delta(
    builder: &mut Builder,
    order_before: &OrderTarget,
    order_book_path_before: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    order_after: &OrderTarget,
) -> [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS] {
    let sibling_ask_base_amount = builder.sub(
        order_book_path_before[0].ask_base_sum,
        order_before.ask_base_sum,
    );
    let sibling_ask_quote_amount = builder.sub(
        order_book_path_before[0].ask_quote_sum,
        order_before.ask_quote_sum,
    );
    let sibling_bid_base_amount = builder.sub(
        order_book_path_before[0].bid_base_sum,
        order_before.bid_base_sum,
    );
    let sibling_bid_quote_amount = builder.sub(
        order_book_path_before[0].bid_quote_sum,
        order_before.bid_quote_sum,
    );

    let mut order_book_path_after_vec = vec![OrderBookNodeTarget {
        sibling_child_hash: order_book_path_before[0].sibling_child_hash,
        ask_base_sum: builder.add(order_after.ask_base_sum, sibling_ask_base_amount),
        ask_quote_sum: builder.add(order_after.ask_quote_sum, sibling_ask_quote_amount),
        bid_base_sum: builder.add(order_after.bid_base_sum, sibling_bid_base_amount),
        bid_quote_sum: builder.add(order_after.bid_quote_sum, sibling_bid_quote_amount),
    }];

    for i in 1..ORDER_BOOK_MERKLE_LEVELS {
        let sibling_ask_base_sum = builder.sub(
            order_book_path_before[i].ask_base_sum,
            order_book_path_before[i - 1].ask_base_sum,
        );
        let sibling_ask_quote_sum = builder.sub(
            order_book_path_before[i].ask_quote_sum,
            order_book_path_before[i - 1].ask_quote_sum,
        );
        let sibling_bid_base_sum = builder.sub(
            order_book_path_before[i].bid_base_sum,
            order_book_path_before[i - 1].bid_base_sum,
        );
        let sibling_bid_quote_sum = builder.sub(
            order_book_path_before[i].bid_quote_sum,
            order_book_path_before[i - 1].bid_quote_sum,
        );

        order_book_path_after_vec.push(OrderBookNodeTarget {
            sibling_child_hash: order_book_path_before[i].sibling_child_hash,
            ask_base_sum: builder.add(
                order_book_path_after_vec[i - 1].ask_base_sum,
                sibling_ask_base_sum,
            ),
            ask_quote_sum: builder.add(
                order_book_path_after_vec[i - 1].ask_quote_sum,
                sibling_ask_quote_sum,
            ),
            bid_base_sum: builder.add(
                order_book_path_after_vec[i - 1].bid_base_sum,
                sibling_bid_base_sum,
            ),
            bid_quote_sum: builder.add(
                order_book_path_after_vec[i - 1].bid_quote_sum,
                sibling_bid_quote_sum,
            ),
        });
    }

    builder.register_range_check(
        order_book_path_after_vec[ORDER_BOOK_MERKLE_LEVELS - 1].ask_base_sum,
        BASE_SUM_BITS,
    );
    builder.register_range_check(
        order_book_path_after_vec[ORDER_BOOK_MERKLE_LEVELS - 1].ask_quote_sum,
        QUOTE_SUM_BITS,
    );
    builder.register_range_check(
        order_book_path_after_vec[ORDER_BOOK_MERKLE_LEVELS - 1].bid_base_sum,
        BASE_SUM_BITS,
    );
    builder.register_range_check(
        order_book_path_after_vec[ORDER_BOOK_MERKLE_LEVELS - 1].bid_quote_sum,
        QUOTE_SUM_BITS,
    );

    order_book_path_after_vec.try_into().unwrap()
}

// For given order book tree path and order side, calculates the total size and quote of the orders that has strictly higher priority than the given order
pub fn get_quote(
    builder: &mut Builder,
    is_ask: BoolTarget,
    order_before: &OrderTarget,
    order_book_path: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    order_book_path_helper: &[BoolTarget; ORDER_BOOK_MERKLE_LEVELS],
) -> (Target, Target) {
    let mut size_sum = builder.zero();
    let mut quote_sum = builder.zero();

    let zero = builder.zero();
    // For each level above the leaf, calculate the size/quote values of the orders with higher priority for given side
    for i in 1..ORDER_BOOK_MERKLE_LEVELS {
        let ask_base_diff = builder.sub(
            order_book_path[i].ask_base_sum,
            order_book_path[i - 1].ask_base_sum,
        );
        let ask_quote_diff = builder.sub(
            order_book_path[i].ask_quote_sum,
            order_book_path[i - 1].ask_quote_sum,
        );
        let bid_base_diff = builder.sub(
            order_book_path[i].bid_base_sum,
            order_book_path[i - 1].bid_base_sum,
        );
        let bid_quote_diff = builder.sub(
            order_book_path[i].bid_quote_sum,
            order_book_path[i - 1].bid_quote_sum,
        );
        let ask_size = builder.select(order_book_path_helper[i], ask_base_diff, zero);
        let ask_quote = builder.select(order_book_path_helper[i], ask_quote_diff, zero);
        let bid_size = builder.select(order_book_path_helper[i], zero, bid_base_diff);
        let bid_quote = builder.select(order_book_path_helper[i], zero, bid_quote_diff);
        let side_adjusted_size = builder.select(is_ask, bid_size, ask_size);
        let side_adjusted_quote = builder.select(is_ask, bid_quote, ask_quote);
        size_sum = builder.add(size_sum, side_adjusted_size);
        quote_sum = builder.add(quote_sum, side_adjusted_quote);
    }

    let sibling_ask_base_diff =
        builder.sub(order_book_path[0].ask_base_sum, order_before.ask_base_sum);
    let sibling_ask_quote_diff =
        builder.sub(order_book_path[0].ask_quote_sum, order_before.ask_quote_sum);
    let sibling_bid_base_diff =
        builder.sub(order_book_path[0].bid_base_sum, order_before.bid_base_sum);
    let sibling_bid_quote_diff =
        builder.sub(order_book_path[0].bid_quote_sum, order_before.bid_quote_sum);
    let sibling_ask_size = builder.select(order_book_path_helper[0], sibling_ask_base_diff, zero);
    let sibling_ask_quote = builder.select(order_book_path_helper[0], sibling_ask_quote_diff, zero);
    let sibling_bid_size = builder.select(order_book_path_helper[0], zero, sibling_bid_base_diff);
    let sibling_bid_quote = builder.select(order_book_path_helper[0], zero, sibling_bid_quote_diff);

    let side_adjusted_sibling_size = builder.select(is_ask, sibling_bid_size, sibling_ask_size);
    let side_adjusted_sibling_quote = builder.select(is_ask, sibling_bid_quote, sibling_ask_quote);
    size_sum = builder.add(size_sum, side_adjusted_sibling_size);
    quote_sum = builder.add(quote_sum, side_adjusted_sibling_quote);

    (size_sum, quote_sum)
}

pub fn get_next_order_nonce(
    builder: &mut Builder,
    market: &MarketTarget,
    is_ask: BoolTarget,
) -> Target {
    builder.select(is_ask, market.ask_nonce, market.bid_nonce)
}

pub fn execute_matching(builder: &mut Builder, tx_state: &mut TxState, timestamp: Target) {
    let one = builder.one();
    let neg_one = builder.neg_one();
    let _false = builder._false();

    let is_perps = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_PERPS);
    let is_spot = builder.not(is_perps);

    let is_taker_ask = tx_state.register_stack[0].pending_is_ask;
    let is_taker_bid = builder.not(is_taker_ask);

    // Initialize order types
    let limit_order_type = builder.constant_from_u8(LIMIT_ORDER);
    let market_order_type = builder.constant_from_u8(MARKET_ORDER);
    let stop_loss_order_type = builder.constant_from_u8(STOP_LOSS_ORDER);
    let stop_loss_limit_order_type = builder.constant_from_u8(STOP_LOSS_LIMIT_ORDER);
    let take_profit_order_type = builder.constant_from_u8(TAKE_PROFIT_ORDER);
    let take_profit_limit_order_type = builder.constant_from_u8(TAKE_PROFIT_LIMIT_ORDER);
    let twap_sub_order_type = builder.constant_from_u8(TWAP_SUB_ORDER);
    let liquidation_order_type = builder.constant_from_u8(LIQUIDATION_ORDER);

    // Initialize NA trigger status
    let trigger_status_na = builder.constant_from_u8(TRIGGER_STATUS_NA);
    let is_pending_trigger_status_not_na = builder.is_not_equal(
        tx_state.register_stack[0].pending_trigger_status,
        trigger_status_na,
    );

    // Initialize order type flags
    let is_maker_limit_order =
        builder.is_equal(tx_state.account_order.order_type, limit_order_type);
    let is_limit_order =
        builder.is_equal(tx_state.register_stack[0].pending_type, limit_order_type);
    let is_market_order =
        builder.is_equal(tx_state.register_stack[0].pending_type, market_order_type);
    let is_stop_loss_order = builder.is_equal(
        tx_state.register_stack[0].pending_type,
        stop_loss_order_type,
    );
    let is_stop_loss_limit_order = builder.is_equal(
        tx_state.register_stack[0].pending_type,
        stop_loss_limit_order_type,
    );
    let is_take_profit_order = builder.is_equal(
        tx_state.register_stack[0].pending_type,
        take_profit_order_type,
    );
    let is_take_profit_limit_order = builder.is_equal(
        tx_state.register_stack[0].pending_type,
        take_profit_limit_order_type,
    );
    let is_twap_sub_order =
        builder.is_equal(tx_state.register_stack[0].pending_type, twap_sub_order_type);
    let is_liquidation_order = builder.is_equal(
        tx_state.register_stack[0].pending_type,
        liquidation_order_type,
    );

    let market_flag = builder.multi_or(&[
        is_market_order,
        is_stop_loss_order,
        is_take_profit_order,
        is_twap_sub_order,
    ]);
    let limit_flag = builder.multi_or(&[
        is_limit_order,
        is_liquidation_order,
        is_stop_loss_limit_order,
        is_take_profit_limit_order,
    ]);

    // Initialize time in force types
    let ioc = builder.constant_from_u8(IOC);
    let post_only = builder.constant_from_u8(POST_ONLY);

    // Initialize time in force flags
    let is_ioc = builder.is_equal(tx_state.register_stack[0].pending_time_in_force, ioc);
    let is_post_only =
        builder.is_equal(tx_state.register_stack[0].pending_time_in_force, post_only);

    let total_opposite_side_order_size = builder.select(
        is_taker_ask,
        tx_state.order_book_tree_path[ORDER_BOOK_MERKLE_LEVELS - 1].bid_base_sum,
        tx_state.order_book_tree_path[ORDER_BOOK_MERKLE_LEVELS - 1].ask_base_sum,
    );
    let is_opposite_side_empty = builder.is_zero(total_opposite_side_order_size);

    let taker_price_gt_maker_price = builder.is_gt(
        tx_state.register_stack[0].pending_price,
        tx_state.order.price_index,
        ORDER_PRICE_BITS,
    );
    let taker_price_eq_maker_price = builder.is_equal(
        tx_state.register_stack[0].pending_price,
        tx_state.order.price_index,
    );
    let taker_price_gte_maker_price =
        builder.or(taker_price_gt_maker_price, taker_price_eq_maker_price);
    let taker_price_lt_maker_price = builder.not(taker_price_gte_maker_price);

    let mut update_status_flags = tx_state.matching_engine_flag;
    let mut cancel_taker_order = builder._false();
    let mut cancel_maker_order = builder._false();
    let mut insert_taker_order = builder._false();

    // If pending trigger status is not NA, insert taker order
    {
        let flag = builder.and(update_status_flags, is_pending_trigger_status_not_na);
        insert_taker_order = builder.select_bool(flag, update_status_flags, insert_taker_order);
        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // 0. Handle the taker order invalid reduce only case
    let abs_taker_account_old_position =
        builder.biguint_u16_to_target(&tx_state.positions[TAKER_ACCOUNT_ID].position.abs);
    let taker_reduce_only = builder.is_equal(tx_state.register_stack[0].pending_reduce_only, one);
    {
        let is_not_valid_reduce_only_direction = is_not_valid_reduce_only_direction(
            builder,
            tx_state.positions[TAKER_ACCOUNT_ID].position.sign,
            is_taker_ask,
        );
        let flag = builder.multi_and(&[
            update_status_flags,
            is_perps,
            is_not_valid_reduce_only_direction,
            taker_reduce_only,
        ]);
        cancel_taker_order = builder.select_bool(flag, update_status_flags, cancel_taker_order);
        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    let order_leaf_is_empty = tx_state.order.is_empty(builder);

    // If order leaf is empty, it should belong to the taker order
    {
        let flag = builder.and(update_status_flags, order_leaf_is_empty);
        builder.conditional_assert_eq(
            flag,
            tx_state.order.price_index,
            tx_state.register_stack[0].pending_price,
        );
        builder.conditional_assert_eq(
            flag,
            tx_state.order.nonce_index,
            tx_state.register_stack[0].pending_nonce,
        );
        builder.conditional_assert_eq(
            flag,
            tx_state.account_order.owner_account_index,
            tx_state.register_stack[0].account_index,
        );
        builder.conditional_assert_eq(
            flag,
            tx_state.account_order.index_0,
            tx_state.register_stack[0].pending_order_index,
        );
        builder.conditional_assert_eq(
            flag,
            tx_state.account_order.index_1,
            tx_state.register_stack[0].pending_client_order_index,
        );
    }

    // Empty order book side for ioc order - cancel the taker order
    {
        let flag = builder.multi_and(&[update_status_flags, is_ioc, is_opposite_side_empty]);

        cancel_taker_order = builder.select_bool(flag, update_status_flags, cancel_taker_order);
        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // Assert that we have the best possible order from orderbook
    let (opposite_base_sum, _) = get_quote(
        builder,
        is_taker_ask,
        &tx_state.order,
        &tx_state.order_book_tree_path,
        &tx_state.order_path_helper,
    );
    let opposite_base_is_zero = builder.is_zero(opposite_base_sum);
    builder.conditional_assert_true(update_status_flags, opposite_base_is_zero);

    // Non crossing ioc - cancel the taker order
    {
        let flag = builder.multi_and(&[update_status_flags, is_ioc, order_leaf_is_empty]);

        cancel_taker_order = builder.select_bool(flag, update_status_flags, cancel_taker_order);
        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // Non crossing non ioc limit - put order to orderbook
    {
        let flag = builder.multi_and(&[update_status_flags, order_leaf_is_empty]);

        // Register should be a limit order
        builder.conditional_assert_true(flag, limit_flag);

        insert_taker_order = builder.select_bool(flag, update_status_flags, insert_taker_order);
        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // After this point, order is not empty
    // Account order and orderbook order should match
    {
        builder.conditional_assert_eq(
            update_status_flags,
            tx_state.order.price_index,
            tx_state.account_order.price,
        );
        builder.conditional_assert_eq(
            update_status_flags,
            tx_state.order.nonce_index,
            tx_state.account_order.nonce,
        );

        let (market_index, _) = get_market_index_and_order_nonce_from_order_index(
            builder,
            tx_state.account_order.index_0,
        );
        builder.conditional_assert_eq(
            update_status_flags,
            market_index,
            tx_state.market.market_index,
        );
    }

    let mut optimistic_trade_amount = builder.min(
        &[
            tx_state.account_order.remaining_base_amount,
            tx_state.register_stack[0].pending_size, // anything written to register is range-checked
        ],
        ORDER_SIZE_BITS,
    );

    let is_self_trade = builder.is_equal(
        tx_state.account_order.owner_account_index,
        tx_state.register_stack[0].account_index,
    );
    let is_maker_order_expired =
        builder.is_lte(tx_state.account_order.expiry, timestamp, TIMESTAMP_BITS);

    // Handle self trade case
    {
        // If it is a self trade;
        // - dead man's switch can not be triggered, since create order would have failed before setting the register.
        // - maker order can not be canceled due to health checks, since post self-trade account health or margin requirements do not change
        // Thus only case for maker to be canceled before executing the self trade is, order expiry

        // Handle expired order case
        {
            let order_expiry_flag =
                builder.multi_and(&[update_status_flags, is_self_trade, is_maker_order_expired]);
            cancel_maker_order =
                builder.select_bool(order_expiry_flag, update_status_flags, cancel_maker_order);

            update_status_flags =
                builder.select_bool(order_expiry_flag, _false, update_status_flags);
        }

        // Handle post-only taker case
        {
            let post_only_flag =
                builder.multi_and(&[update_status_flags, is_self_trade, is_post_only]);
            cancel_taker_order =
                builder.select_bool(post_only_flag, update_status_flags, cancel_taker_order);
            update_status_flags = builder.select_bool(post_only_flag, _false, update_status_flags);
        }

        let self_trade_flag = builder.and(update_status_flags, is_self_trade);

        let new_register_pending_size = builder.sub(
            tx_state.register_stack[0].pending_size,
            optimistic_trade_amount,
        );
        let new_order_remaining_size = builder.sub(
            tx_state.account_order.remaining_base_amount,
            optimistic_trade_amount,
        );
        tx_state.register_stack[0].pending_size = builder.select(
            self_trade_flag,
            new_register_pending_size,
            tx_state.register_stack[0].pending_size,
        );
        tx_state.account_order.remaining_base_amount = builder.select(
            self_trade_flag,
            new_order_remaining_size,
            tx_state.account_order.remaining_base_amount,
        );

        let decrement_locked_balance_flag =
            builder.multi_and(&[self_trade_flag, is_spot, is_maker_limit_order]);
        decrement_locked_balance_for_partial_order(
            builder,
            decrement_locked_balance_flag,
            &tx_state.market,
            tx_state.account_order.is_ask,
            optimistic_trade_amount,
            tx_state.account_order.price,
            &mut tx_state.account_assets[TAKER_ACCOUNT_ID],
        );
        tx_state.order.set_remaining_amount_conditional(
            builder,
            self_trade_flag,
            tx_state.account_order.is_ask,
            new_order_remaining_size,
        );

        // Taker filled
        {
            let is_register_pending_size_empty =
                builder.is_zero(tx_state.register_stack[0].pending_size);
            let self_trade_and_register_pending_size_empty =
                builder.and(self_trade_flag, is_register_pending_size_empty);
            cancel_taker_order = builder.select_bool(
                self_trade_and_register_pending_size_empty,
                update_status_flags,
                cancel_taker_order,
            );
        }

        // Maker filled
        {
            let is_order_remaining_size_empty =
                builder.is_zero(tx_state.account_order.remaining_base_amount);
            let self_trade_and_order_remaining_size_empty =
                builder.and(self_trade_flag, is_order_remaining_size_empty);
            cancel_maker_order = builder.select_bool(
                self_trade_and_order_remaining_size_empty,
                update_status_flags,
                cancel_maker_order,
            );
        }

        update_status_flags = builder.select_bool(self_trade_flag, _false, update_status_flags);
    }

    // Taker and maker are different accounts, verify if maker account in witness is consistent
    {
        builder.conditional_assert_eq(
            update_status_flags,
            tx_state.account_order.owner_account_index,
            tx_state.accounts[MAKER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_not_eq(
            update_status_flags,
            tx_state.accounts[TAKER_ACCOUNT_ID].account_index,
            tx_state.accounts[MAKER_ACCOUNT_ID].account_index,
        );
    }

    // Handle maker order being expired or dead mans switch time being passed case
    {
        let should_dms_be_triggered =
            tx_state.accounts[MAKER_ACCOUNT_ID].should_dms_be_triggered(builder, timestamp);

        let cancel_order = builder.or(should_dms_be_triggered, is_maker_order_expired);
        let flag = builder.and(update_status_flags, cancel_order);

        cancel_maker_order = builder.select_bool(flag, update_status_flags, cancel_maker_order);

        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // Cancel the taker order if it is a post only order
    {
        let flag = builder.and(update_status_flags, is_post_only);
        cancel_taker_order = builder.select_bool(flag, update_status_flags, cancel_taker_order);
        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // Handle maker order invalid reduce only case
    let is_maker_reduce_only = BoolTarget::new_unsafe(tx_state.account_order.reduce_only);
    {
        let is_not_valid_reduce_only_direction = is_not_valid_reduce_only_direction(
            builder,
            tx_state.positions[MAKER_ACCOUNT_ID].position.sign,
            tx_state.account_order.is_ask,
        );
        let flag = builder.multi_and(&[
            is_perps,
            update_status_flags,
            is_not_valid_reduce_only_direction,
            is_maker_reduce_only,
        ]);
        cancel_maker_order = builder.select_bool(flag, update_status_flags, cancel_maker_order);

        update_status_flags = builder.select_bool(flag, _false, update_status_flags);
    }

    // Compute trade base
    {
        let abs_maker_account_old_position =
            builder.biguint_u16_to_target(&tx_state.positions[MAKER_ACCOUNT_ID].position.abs);
        let abs_maker_position_lt_trade_amount =
            builder.is_lt(abs_maker_account_old_position, optimistic_trade_amount, 64);
        let reduce_trade_base_flag = builder.multi_and(&[
            update_status_flags,
            is_maker_reduce_only,
            abs_maker_position_lt_trade_amount,
        ]);
        optimistic_trade_amount = builder.select(
            reduce_trade_base_flag,
            abs_maker_account_old_position,
            optimistic_trade_amount,
        );

        let abs_taker_position_lt_trade_amount =
            builder.is_lt(abs_taker_account_old_position, optimistic_trade_amount, 64);
        let reduce_trade_base_flag = builder.multi_and(&[
            update_status_flags,
            taker_reduce_only,
            abs_taker_position_lt_trade_amount,
        ]);
        optimistic_trade_amount = builder.select(
            reduce_trade_base_flag,
            abs_taker_account_old_position,
            optimistic_trade_amount,
        );
    }

    // Adjust trade size using the slippage accumulator value (generic_field_0)
    let is_market_order_with_too_much_slippage = {
        let flag = builder.and(update_status_flags, market_flag);

        let ask_taker_with_slippage = builder.and(is_taker_ask, taker_price_gt_maker_price);
        let bid_taker_with_slippage = builder.and(is_taker_bid, taker_price_lt_maker_price);
        let is_slippage = builder.or(ask_taker_with_slippage, bid_taker_with_slippage);

        let taker_minus_maker_price = builder.sub(
            tx_state.register_stack[0].pending_price,
            tx_state.order.price_index,
        );
        let price_diff_multiplier = builder.select(ask_taker_with_slippage, one, neg_one);
        let price_diff = builder.mul(taker_minus_maker_price, price_diff_multiplier);

        let (mut allowed_trade_base, _) = builder.conditional_div_rem(
            is_slippage,
            tx_state.register_stack[0].generic_field_0,
            price_diff,
            ORDER_PRICE_BITS,
        );
        allowed_trade_base =
            builder.select(is_slippage, allowed_trade_base, optimistic_trade_amount);

        let is_allowed_trade_base_not_equal_to_optimistic_trade_amount =
            builder.is_not_equal(allowed_trade_base, optimistic_trade_amount);
        let new_optimistic_trade_amount_check = builder.min(
            &[allowed_trade_base, optimistic_trade_amount],
            ORDER_SIZE_BITS,
        );
        optimistic_trade_amount = builder.select(
            flag,
            new_optimistic_trade_amount_check,
            optimistic_trade_amount,
        );

        let is_trade_empty = builder.is_zero(optimistic_trade_amount);
        let empty_trade_flag = builder.and(flag, is_trade_empty);

        cancel_taker_order =
            builder.select_bool(empty_trade_flag, update_status_flags, cancel_taker_order);
        update_status_flags = builder.select_bool(empty_trade_flag, _false, update_status_flags);

        let optimistic_trade_base_eq_allowed_trade_base =
            builder.is_equal(optimistic_trade_amount, allowed_trade_base);
        let is_too_much_slippage = builder.multi_and(&[
            is_slippage,
            optimistic_trade_base_eq_allowed_trade_base,
            is_allowed_trade_base_not_equal_to_optimistic_trade_amount,
        ]);

        builder.select_bool(flag, is_too_much_slippage, _false)
    };

    // Both insurance fund and treasury can be the fee collector
    {
        let is_fee_collector_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[FEE_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_fee_collector_treasury = builder.is_equal_constant(
            tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            TREASURY_ACCOUNT_INDEX as u64,
        );
        let is_fee_collector_insurance_fund_or_treasury =
            builder.or(is_fee_collector_insurance_fund, is_fee_collector_treasury);
        builder.conditional_assert_true(
            update_status_flags,
            is_fee_collector_insurance_fund_or_treasury,
        );
    }

    let trade_base = optimistic_trade_amount;
    let quote_multiplier = builder.select(is_perps, tx_state.market_details.quote_multiplier, one);
    let trade_quote = SignedTarget::new_unsafe(builder.mul_many([
        trade_base,
        tx_state.order.price_index,
        quote_multiplier,
    ])); // Already verified that multiplication can fit NORMALIZED_QUOTE_BITS bits and can't be negative

    let apply_trade_params = ApplyTradeParams {
        market: &tx_state.market,
        market_details: &tx_state.market_details,
        is_taker_ask,
        trade_base,
        trade_quote,
        taker_position: &tx_state.positions[TAKER_ACCOUNT_ID],
        maker_position: &tx_state.positions[MAKER_ACCOUNT_ID],
        taker_risk_info: &tx_state.risk_infos[TAKER_ACCOUNT_ID],
        maker_risk_info: &tx_state.risk_infos[MAKER_ACCOUNT_ID],
        taker_fee: tx_state.taker_fee,
        maker_fee: tx_state.maker_fee,
    };

    let (
        new_taker_position,
        new_maker_position,
        new_taker_risk_info_perps,
        new_maker_risk_info_perps,
        fee_account_collateral_delta,
        new_open_interest,
        taker_position_sign_changed,
        maker_position_sign_changed,
        is_taker_position_isolated,
        is_maker_position_isolated,
        taker_margin_delta,
        maker_margin_delta,
    ) = apply_perps_trade(builder, update_status_flags, &apply_trade_params);

    is_valid_perps_trade(
        builder,
        &mut update_status_flags,
        tx_state,
        &new_taker_position,
        &new_taker_risk_info_perps,
        &taker_margin_delta,
        &new_maker_position,
        &new_maker_risk_info_perps,
        &maker_margin_delta,
        new_open_interest,
        &mut cancel_taker_order,
        &mut cancel_maker_order,
    );

    let apply_spot_trade_params = ApplySpotTradeParams {
        account_assets: &tx_state.account_assets,
        fee_account_is_taker: tx_state.fee_account_is_taker,
        fee_account_is_maker: tx_state.fee_account_is_maker,
    };
    let (
        taker_base_balance_delta,
        taker_quote_balance_delta,
        maker_base_balance_delta,
        maker_quote_balance_delta,
        fee_base_balance_delta,
        fee_quote_balance_delta,
        new_taker_risk_info_spot,
        new_maker_risk_info_spot,
    ) = apply_spot_trade(
        builder,
        update_status_flags,
        tx_state,
        &apply_trade_params,
        &apply_spot_trade_params,
    );

    is_valid_spot_trade(
        builder,
        &mut update_status_flags,
        tx_state,
        is_taker_ask,
        &taker_base_balance_delta,
        &taker_quote_balance_delta,
        &maker_base_balance_delta,
        &maker_quote_balance_delta,
        &new_taker_risk_info_spot,
        &new_maker_risk_info_spot,
        &mut cancel_taker_order,
        &mut cancel_maker_order,
    );

    let new_taker_risk_info = RiskInfoTarget {
        current_risk_parameters: RiskParametersTarget::select(
            builder,
            is_perps,
            &new_taker_risk_info_perps.current_risk_parameters,
            &new_taker_risk_info_spot.current_risk_parameters,
        ),
        cross_risk_parameters: RiskParametersTarget::select(
            builder,
            is_perps,
            &new_taker_risk_info_perps.cross_risk_parameters,
            &new_taker_risk_info_spot.cross_risk_parameters,
        ),
    };
    let new_maker_risk_info = RiskInfoTarget {
        current_risk_parameters: RiskParametersTarget::select(
            builder,
            is_perps,
            &new_maker_risk_info_perps.current_risk_parameters,
            &new_maker_risk_info_spot.current_risk_parameters,
        ),
        cross_risk_parameters: RiskParametersTarget::select(
            builder,
            is_perps,
            &new_maker_risk_info_perps.cross_risk_parameters,
            &new_maker_risk_info_spot.cross_risk_parameters,
        ),
    };

    // Verify maker and taker fee being valid
    {
        let not_liquidation_flag = builder.and_not(update_status_flags, is_liquidation_order);
        builder.range_check_signed(tx_state.taker_fee, 24); // 24 to use split_bytes cache
        builder.conditional_assert_lte_signed_special(
            not_liquidation_flag,
            tx_state.taker_fee,
            tx_state.market.taker_fee,
            FEE_BITS,
        );
        builder.range_check_signed(tx_state.maker_fee, 24); // 24 to use split_bytes cache
        builder.conditional_assert_lte_signed_special(
            update_status_flags,
            tx_state.maker_fee,
            tx_state.market.maker_fee,
            FEE_BITS,
        );
        let total_fee = builder.add_signed(tx_state.taker_fee, tx_state.maker_fee);
        let is_total_fee_non_negative = builder.is_non_negative(total_fee);
        builder.conditional_assert_true(update_status_flags, is_total_fee_non_negative);

        let liquidation_flag = builder.and(update_status_flags, is_liquidation_order);
        {
            let maker_price_signed = SignedTarget::new_unsafe(tx_state.order.price_index);
            let taker_price_signed =
                SignedTarget::new_unsafe(tx_state.register_stack[0].pending_price);
            let price_diff = builder.sub_signed(maker_price_signed, taker_price_signed);
            let (price_diff_abs, _) = builder.abs(price_diff);
            let fee_tick = builder.constant_u64(FEE_TICK);
            let price_diff_tick = builder.mul(price_diff_abs, fee_tick);
            let (price_diff_rate, _) = builder.conditional_div_rem(
                liquidation_flag,
                price_diff_tick,
                tx_state.order.price_index,
                ORDER_PRICE_BITS,
            ); // 52 bits
            let new_taker_fee =
                builder.min(&[tx_state.market.liquidation_fee, price_diff_rate], 64);

            builder.conditional_assert_lte_signed_special(
                liquidation_flag,
                tx_state.taker_fee,
                new_taker_fee,
                FEE_BITS,
            );
        }

        {
            let maker_fee_sign = builder.sign(tx_state.maker_fee);
            let maker_fee_is_not_negative = builder.is_not_equal(maker_fee_sign.target, neg_one);
            builder.conditional_assert_true(liquidation_flag, maker_fee_is_not_negative);
        }
    }

    // Apply trade to the state
    {
        let fee_account_is_taker = builder.and(update_status_flags, tx_state.fee_account_is_taker);
        let fee_account_is_maker = builder.and(update_status_flags, tx_state.fee_account_is_maker);

        // Update account assets for spot
        {
            let update_assets_flag = builder.and(update_status_flags, is_spot);

            AccountTarget::apply_asset_delta_const(
                builder,
                update_assets_flag,
                PRODUCT_TYPE_SPOT,
                &mut tx_state.accounts[TAKER_ACCOUNT_ID],
                Some(&mut tx_state.account_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID]),
                tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
                &taker_base_balance_delta,
                &mut tx_state.strategies[TAKER_ACCOUNT_ID],
            );

            AccountTarget::apply_asset_delta_const(
                builder,
                update_assets_flag,
                PRODUCT_TYPE_SPOT,
                &mut tx_state.accounts[TAKER_ACCOUNT_ID],
                Some(&mut tx_state.account_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID]),
                tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
                &taker_quote_balance_delta,
                &mut tx_state.strategies[TAKER_ACCOUNT_ID],
            );

            AccountTarget::apply_asset_delta_const(
                builder,
                update_assets_flag,
                PRODUCT_TYPE_SPOT,
                &mut tx_state.accounts[MAKER_ACCOUNT_ID],
                Some(&mut tx_state.account_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID]),
                tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
                &maker_base_balance_delta,
                &mut tx_state.strategies[MAKER_ACCOUNT_ID],
            );
            AccountTarget::apply_asset_delta_const(
                builder,
                update_assets_flag,
                PRODUCT_TYPE_SPOT,
                &mut tx_state.accounts[MAKER_ACCOUNT_ID],
                Some(&mut tx_state.account_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID]),
                tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
                &maker_quote_balance_delta,
                &mut tx_state.strategies[MAKER_ACCOUNT_ID],
            );

            AccountTarget::apply_asset_delta_const(
                builder,
                update_assets_flag,
                PRODUCT_TYPE_SPOT,
                &mut tx_state.accounts[FEE_ACCOUNT_ID],
                Some(&mut tx_state.account_assets[FEE_ACCOUNT_ID][BASE_ASSET_ID]),
                tx_state.is_asset_used_as_margin[FEE_ACCOUNT_ID][BASE_ASSET_ID],
                &fee_base_balance_delta,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
            );
            AccountTarget::apply_asset_delta_const(
                builder,
                update_assets_flag,
                PRODUCT_TYPE_SPOT,
                &mut tx_state.accounts[FEE_ACCOUNT_ID],
                Some(&mut tx_state.account_assets[FEE_ACCOUNT_ID][QUOTE_ASSET_ID]),
                tx_state.is_asset_used_as_margin[FEE_ACCOUNT_ID][QUOTE_ASSET_ID],
                &fee_quote_balance_delta,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
            );
        }

        // Update market, register, order leaf, account order leaf
        {
            tx_state.market_details.open_interest = builder.select(
                update_status_flags,
                new_open_interest,
                tx_state.market_details.open_interest,
            );

            let new_register_pending_size =
                builder.sub(tx_state.register_stack[0].pending_size, trade_base);
            let new_order_remaining_size =
                builder.sub(tx_state.account_order.remaining_base_amount, trade_base);
            tx_state.register_stack[0].pending_size = builder.select(
                update_status_flags,
                new_register_pending_size,
                tx_state.register_stack[0].pending_size,
            );
            let ask_taker_price_diff = builder.sub(
                tx_state.order.price_index,
                tx_state.register_stack[0].pending_price,
            );
            let bid_taker_price_diff = builder.sub(
                tx_state.register_stack[0].pending_price,
                tx_state.order.price_index,
            );
            let new_ask_taker_pending_generic_field_0 =
                builder.mul(ask_taker_price_diff, trade_base);
            let new_bid_taker_pending_generic_field_0 =
                builder.mul(bid_taker_price_diff, trade_base);
            let new_slippage_accumulator_delta = builder.select(
                is_taker_ask,
                new_ask_taker_pending_generic_field_0,
                new_bid_taker_pending_generic_field_0,
            );
            let new_market_order_generic_field_0 = builder.add(
                tx_state.register_stack[0].generic_field_0,
                new_slippage_accumulator_delta,
            );
            let new_generic_field_0 = builder.select(
                market_flag,
                new_market_order_generic_field_0,
                tx_state.register_stack[0].generic_field_0,
            );

            tx_state.register_stack[0].generic_field_0 = builder.select(
                update_status_flags,
                new_generic_field_0,
                tx_state.register_stack[0].generic_field_0,
            );

            tx_state.account_order.remaining_base_amount = builder.select(
                update_status_flags,
                new_order_remaining_size,
                tx_state.account_order.remaining_base_amount,
            );

            let decrement_locked_balance_flag =
                builder.multi_and(&[update_status_flags, is_spot, is_maker_limit_order]);
            decrement_locked_balance_for_partial_order(
                builder,
                decrement_locked_balance_flag,
                &tx_state.market,
                tx_state.account_order.is_ask,
                trade_base,
                tx_state.account_order.price,
                &mut tx_state.account_assets[MAKER_ACCOUNT_ID],
            );

            tx_state.order.set_remaining_amount_conditional(
                builder,
                update_status_flags,
                tx_state.account_order.is_ask,
                new_order_remaining_size,
            );
        }

        // Taker filled / too much slippage / reduce only cancel / liquidation stop
        {
            let is_register_pending_size_empty =
                builder.is_zero(tx_state.register_stack[0].pending_size);
            let is_taker_not_valid_reduce_only = is_not_valid_reduce_only_direction(
                builder,
                new_taker_position.position.sign,
                tx_state.register_stack[0].pending_is_ask,
            );
            let cancel_reduce_only_taker = // taker_reduce_only is enough to enforce is perps
                builder.and(taker_reduce_only, is_taker_not_valid_reduce_only);

            // Check if the account health is above MMR after a liquidation trade
            let is_not_in_liquidation = new_taker_risk_info
                .current_risk_parameters
                .is_not_in_liquidation(builder);
            let is_not_in_liquidation_and_is_liquidation_order =
                builder.and(is_not_in_liquidation, is_liquidation_order);
            let cancel_taker = builder.multi_or(&[
                is_register_pending_size_empty,
                cancel_reduce_only_taker,
                is_market_order_with_too_much_slippage,
                is_not_in_liquidation_and_is_liquidation_order,
            ]);
            let cancel_taker_flag = builder.and(update_status_flags, cancel_taker);
            cancel_taker_order =
                builder.select_bool(cancel_taker_flag, update_status_flags, cancel_taker_order);
        }

        // Maker filled // reduce only cancel
        {
            let is_order_remaining_size_empty =
                builder.is_zero(tx_state.account_order.remaining_base_amount);
            let is_maker_not_valid_reduce_only = is_not_valid_reduce_only_direction(
                builder,
                new_maker_position.position.sign,
                tx_state.account_order.is_ask,
            );
            let cancel_reduce_only_maker =
                builder.and(is_maker_reduce_only, is_maker_not_valid_reduce_only);
            let cancel_maker =
                builder.multi_or(&[is_order_remaining_size_empty, cancel_reduce_only_maker]);
            let cancel_maker_flag = builder.and(update_status_flags, cancel_maker);
            cancel_maker_order =
                builder.select_bool(cancel_maker_flag, update_status_flags, cancel_maker_order);
        }

        // Update positions
        {
            tx_state.positions[TAKER_ACCOUNT_ID] = AccountPositionTarget::select_position(
                builder,
                update_status_flags,
                &new_taker_position,
                &tx_state.positions[TAKER_ACCOUNT_ID],
            );
            tx_state.positions[MAKER_ACCOUNT_ID] = AccountPositionTarget::select_position(
                builder,
                update_status_flags,
                &new_maker_position,
                &tx_state.positions[MAKER_ACCOUNT_ID],
            );
        }

        // Update collaterals for perps
        {
            let flag = builder.and(update_status_flags, is_perps);

            // We are not handling unified accounts because perps trades always happen with collateral balance.
            // Update Taker
            let mut new_taker_collateral = builder.select_bigint(
                is_taker_position_isolated,
                &new_taker_risk_info.cross_risk_parameters.collateral,
                &new_taker_risk_info.current_risk_parameters.collateral,
            );
            // If taker and fee accounts are the same, add fee payment to taker's cross collateral too
            // We are using cross collateral here because this can only happen when taker and fee account is insurance fund and
            // it can't open isolated positions
            let new_taker_cross_collateral_with_funding = builder.add_bigint_non_carry(
                &new_taker_collateral,
                &fee_account_collateral_delta,
                BIG_U96_LIMBS,
            );
            new_taker_collateral = builder.select_bigint(
                fee_account_is_taker,
                &new_taker_cross_collateral_with_funding,
                &new_taker_collateral,
            );

            let old_taker_collateral = builder.select_bigint(
                is_taker_position_isolated,
                &tx_state.risk_infos[TAKER_ACCOUNT_ID]
                    .cross_risk_parameters
                    .collateral,
                &tx_state.risk_infos[TAKER_ACCOUNT_ID]
                    .current_risk_parameters
                    .collateral,
            );
            let taker_collateral_delta =
                builder.sub_bigint(&new_taker_collateral, &old_taker_collateral);

            tx_state.accounts[TAKER_ACCOUNT_ID].apply_collateral_delta(
                builder,
                flag,
                &taker_collateral_delta,
                &mut tx_state.strategies[TAKER_ACCOUNT_ID],
            );

            // Update Maker
            let mut new_maker_collateral = builder.select_bigint(
                is_maker_position_isolated,
                &new_maker_risk_info.cross_risk_parameters.collateral,
                &new_maker_risk_info.current_risk_parameters.collateral,
            );
            // If maker and fee accounts are the same, add fee payment to maker's cross collateral too
            // We are using cross collateral here because this can only happen when maker and fee account is insurance fund and
            // it can't open isolated positions
            let new_maker_cross_collateral_with_funding = builder.add_bigint_non_carry(
                &new_maker_collateral,
                &fee_account_collateral_delta,
                BIG_U96_LIMBS,
            );
            new_maker_collateral = builder.select_bigint(
                fee_account_is_maker,
                &new_maker_cross_collateral_with_funding,
                &new_maker_collateral,
            );

            let old_maker_collateral = builder.select_bigint(
                is_maker_position_isolated,
                &tx_state.risk_infos[MAKER_ACCOUNT_ID]
                    .cross_risk_parameters
                    .collateral,
                &tx_state.risk_infos[MAKER_ACCOUNT_ID]
                    .current_risk_parameters
                    .collateral,
            );
            let maker_collateral_delta =
                builder.sub_bigint(&new_maker_collateral, &old_maker_collateral);

            tx_state.accounts[MAKER_ACCOUNT_ID].apply_collateral_delta(
                builder,
                flag,
                &maker_collateral_delta,
                &mut tx_state.strategies[MAKER_ACCOUNT_ID],
            );

            // Update Fee account
            // Fee payments always applied to cross collateral
            tx_state.accounts[FEE_ACCOUNT_ID].apply_collateral_delta(
                builder,
                flag,
                &fee_account_collateral_delta,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
            );
        }
    }

    // Initialize empty order and account order
    let empty_order = OrderTarget::empty(
        builder,
        tx_state.order.price_index,
        tx_state.order.nonce_index,
    );
    let empty_account_order = AccountOrderTarget::empty(
        builder,
        tx_state.account_order.index_0,
        tx_state.account_order.index_1,
        tx_state.account_order.owner_account_index,
    );

    let market_index = tx_state.register_stack[0].market_index;
    let taker_account_index = tx_state.register_stack[0].account_index;
    let maker_account_index = tx_state.account_order.owner_account_index;

    let pop_register = builder.or(cancel_taker_order, insert_taker_order);
    let register_order = get_order_from_register(builder, &tx_state.register_stack[0]);
    let register_account_order = get_account_order_from_register(&tx_state.register_stack[0]);
    tx_state.register_stack.pop_front(builder, pop_register);

    // Cancel maker order if needed
    let cancel_self_trade_maker_order = builder.and(is_self_trade, cancel_maker_order);
    let cancel_non_self_trade_maker_order = builder.and_not(cancel_maker_order, is_self_trade);
    [
        (cancel_self_trade_maker_order, TAKER_ACCOUNT_ID),
        (cancel_non_self_trade_maker_order, MAKER_ACCOUNT_ID),
    ]
    .iter()
    .for_each(|(flag, account_id)| {
        decrement_order_count_in_place(
            builder,
            tx_state,
            *account_id,
            *flag,
            tx_state.account_order.trigger_status,
            tx_state.account_order.reduce_only,
        );

        let decrement_locked_balance_flag =
            builder.multi_and(&[*flag, is_spot, is_maker_limit_order]);
        decrement_locked_balance_for_order(
            builder,
            decrement_locked_balance_flag,
            &tx_state.account_order,
            &tx_state.market,
            &mut tx_state.account_assets[*account_id],
        );
    });

    let maker_child_order_index_0 = tx_state.account_order.to_trigger_order_index0;
    let maker_child_order_index_1 = tx_state.account_order.to_trigger_order_index1;
    let maker_filled_size = builder.sub(
        tx_state.account_order.initial_base_amount,
        tx_state.account_order.remaining_base_amount,
    );
    let is_maker_filled_size_zero = builder.is_zero(maker_filled_size);
    let is_maker_filled_size_non_zero = builder.not(is_maker_filled_size_zero);
    let trigger_maker_child_orders_flag =
        builder.and(is_maker_filled_size_non_zero, cancel_maker_order);
    let cancel_maker_child_orders_flag = builder.and(is_maker_filled_size_zero, cancel_maker_order);
    cancel_child_orders(
        builder,
        cancel_maker_child_orders_flag,
        tx_state,
        market_index,
        maker_account_index,
        maker_child_order_index_0,
        maker_child_order_index_1,
    );
    trigger_child_orders(
        builder,
        trigger_maker_child_orders_flag,
        tx_state,
        market_index,
        maker_account_index,
        maker_child_order_index_0,
        maker_child_order_index_1,
        maker_filled_size,
    );
    tx_state.account_order = select_account_order_target(
        builder,
        cancel_maker_order,
        &empty_account_order,
        &tx_state.account_order,
    );
    tx_state.order =
        select_order_target(builder, cancel_maker_order, &empty_order, &tx_state.order);

    // Cancel taker order if needed
    let taker_child_order_index_0 = register_account_order.to_trigger_order_index0;
    let taker_child_order_index_1 = register_account_order.to_trigger_order_index1;
    let taker_filled_size = builder.sub(
        register_account_order.initial_base_amount,
        register_account_order.remaining_base_amount,
    );
    let is_taker_filled_size_zero = builder.is_zero(taker_filled_size);
    let is_taker_filled_size_non_zero = builder.not(is_taker_filled_size_zero);
    let trigger_taker_child_orders_flag =
        builder.and(is_taker_filled_size_non_zero, cancel_taker_order);
    let cancel_taker_child_orders_flag = builder.and(is_taker_filled_size_zero, cancel_taker_order);
    cancel_child_orders(
        builder,
        cancel_taker_child_orders_flag,
        tx_state,
        market_index,
        taker_account_index,
        taker_child_order_index_0,
        taker_child_order_index_1,
    );
    trigger_child_orders(
        builder,
        trigger_taker_child_orders_flag,
        tx_state,
        market_index,
        taker_account_index,
        taker_child_order_index_0,
        taker_child_order_index_1,
        taker_filled_size,
    );

    // Insert taker order if needed
    let insert_taker_to_order_book =
        builder.and_not(insert_taker_order, is_pending_trigger_status_not_na);
    tx_state.order = select_order_target(
        builder,
        insert_taker_to_order_book,
        &register_order,
        &tx_state.order,
    );
    tx_state.account_order = select_account_order_target(
        builder,
        insert_taker_order,
        &register_account_order,
        &tx_state.account_order,
    );
    increment_order_count_in_place(
        builder,
        tx_state,
        insert_taker_order,
        register_account_order.trigger_status,
        register_account_order.reduce_only,
    );

    let increment_locked_balance_flag =
        builder.multi_and(&[insert_taker_to_order_book, is_spot, is_limit_order]);
    increment_locked_balance_for_order(
        builder,
        increment_locked_balance_flag,
        &tx_state.account_order,
        &tx_state.market,
        &mut tx_state.account_assets[TAKER_ACCOUNT_ID],
    );

    // Cancel all position tied orders for taker and maker if needed
    let taker_has_position_tied_orders =
        builder.is_not_zero(tx_state.positions[TAKER_ACCOUNT_ID].total_position_tied_order_count);
    let taker_cancel_position_tied_account_orders_flag = builder.multi_and(&[
        update_status_flags,
        taker_position_sign_changed,
        taker_has_position_tied_orders,
    ]);
    cancel_position_tied_account_orders(
        builder,
        taker_cancel_position_tied_account_orders_flag,
        tx_state,
        market_index,
        taker_account_index,
        tx_state.positions[TAKER_ACCOUNT_ID].total_position_tied_order_count,
    );

    let maker_has_position_tied_orders =
        builder.is_not_zero(tx_state.positions[MAKER_ACCOUNT_ID].total_position_tied_order_count);
    let maker_cancel_position_tied_account_orders_flag = builder.multi_and(&[
        update_status_flags,
        maker_position_sign_changed,
        maker_has_position_tied_orders,
    ]);
    cancel_position_tied_account_orders(
        builder,
        maker_cancel_position_tied_account_orders_flag,
        tx_state,
        market_index,
        maker_account_index,
        tx_state.positions[MAKER_ACCOUNT_ID].total_position_tied_order_count,
    );
}

fn is_valid_perps_trade(
    builder: &mut Builder,

    update_status_flags: &mut BoolTarget,
    tx_state: &TxState,

    new_taker_position: &AccountPositionTarget,
    new_taker_risk_info: &RiskInfoTarget,
    taker_margin_delta: &BigIntTarget,

    new_maker_position: &AccountPositionTarget,
    new_maker_risk_info: &RiskInfoTarget,
    maker_margin_delta: &BigIntTarget,

    new_open_interest: Target,

    cancel_taker_order: &mut BoolTarget,
    cancel_maker_order: &mut BoolTarget,
) {
    let is_perps = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_PERPS);
    let is_enabled = builder.and(*update_status_flags, is_perps);

    let new_taker_position_abs = builder.biguint_u16_to_biguint(&new_taker_position.position.abs);
    let old_taker_position_abs =
        builder.biguint_u16_to_biguint(&tx_state.positions[TAKER_ACCOUNT_ID].position.abs);
    let is_new_taker_position_gte =
        builder.is_gte_biguint(&new_taker_position_abs, &old_taker_position_abs);

    let old_taker_position_sign = tx_state.positions[TAKER_ACCOUNT_ID].position.sign.target;
    let new_taker_position_sign = new_taker_position.position.sign.target;
    let neg_taker_position_sign = builder.neg(old_taker_position_sign);
    let taker_position_side_flipped =
        builder.is_equal(neg_taker_position_sign, new_taker_position_sign);
    let is_position_increase_or_flip =
        builder.or(is_new_taker_position_gte, taker_position_side_flipped);

    let open_interest_notional_mult = builder.mul(
        tx_state.market_details.mark_price,
        tx_state.market_details.quote_multiplier,
    );
    let old_open_interest_notional = builder.mul(
        tx_state.market_details.open_interest,
        open_interest_notional_mult,
    );
    let new_open_interest_notional = builder.mul(new_open_interest, open_interest_notional_mult);
    let is_taker_insurance_fund = builder.is_equal_constant(
        tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
        INSURANCE_FUND_ACCOUNT_TYPE as u64,
    );
    let is_maker_insurance_fund = builder.is_equal_constant(
        tx_state.accounts[MAKER_ACCOUNT_ID].account_type,
        INSURANCE_FUND_ACCOUNT_TYPE as u64,
    );
    let is_insurance_fund_trade = builder.or(is_taker_insurance_fund, is_maker_insurance_fund);
    let is_not_insurance_fund_trade = builder.not(is_insurance_fund_trade);
    let is_market_open_interest_notional_full = builder.is_gt(
        old_open_interest_notional,
        tx_state.market_details.open_interest_limit,
        64,
    );
    let is_market_open_interest_full_and_is_taker_not_reduce = builder.and(
        is_market_open_interest_notional_full,
        is_position_increase_or_flip,
    );
    let is_market_open_interest_full_and_is_taker_not_reduce_and_not_insurance_fund_trade = builder
        .and(
            is_market_open_interest_full_and_is_taker_not_reduce,
            is_not_insurance_fund_trade,
        );

    let old_open_interest_notional_within_the_limit =
        builder.not(is_market_open_interest_notional_full);

    let new_open_interest_notional_gt_limit = builder.is_gt(
        new_open_interest_notional,
        tx_state.market_details.open_interest_limit,
        64,
    );

    let max_open_interest_notional =
        builder.constant(F::from_canonical_u64(MARKET_OPEN_INTEREST_NOTIONAL));

    let new_open_interest_notional_gt_max_limit =
        builder.is_gt(new_open_interest_notional, max_open_interest_notional, 64);

    let open_interest_notional_went_over_the_limit = builder.and(
        old_open_interest_notional_within_the_limit,
        new_open_interest_notional_gt_limit,
    );

    let mut open_interest_notional_went_over_the_limit_and_not_insurance_fund_trade = builder.and(
        open_interest_notional_went_over_the_limit,
        is_not_insurance_fund_trade,
    );

    let open_interest_limit = builder.constant_u64(MARKET_OPEN_INTEREST);

    let mut open_interest_went_over_the_limit =
        builder.is_gt(new_open_interest, open_interest_limit, 64);

    let mut cancel_taker = builder.and(
        is_market_open_interest_full_and_is_taker_not_reduce_and_not_insurance_fund_trade,
        is_enabled,
    );

    open_interest_notional_went_over_the_limit_and_not_insurance_fund_trade = builder.and(
        open_interest_notional_went_over_the_limit_and_not_insurance_fund_trade,
        is_enabled,
    );
    open_interest_went_over_the_limit = builder.and(open_interest_went_over_the_limit, is_enabled);

    cancel_taker = builder.or(
        cancel_taker,
        open_interest_notional_went_over_the_limit_and_not_insurance_fund_trade,
    );
    cancel_taker = builder.or(cancel_taker, open_interest_went_over_the_limit);
    cancel_taker = builder.or(cancel_taker, new_open_interest_notional_gt_max_limit);
    // Check if taker is health transition is valid and position is allowed, early return
    {
        // Check if position change is valid
        {
            let is_new_taker_position_valid = new_taker_position.is_valid(builder);
            let is_new_taker_position_invalid =
                builder.and_not(is_enabled, is_new_taker_position_valid);
            cancel_taker = builder.or(cancel_taker, is_new_taker_position_invalid);
        }
        // Check if risk change is valid
        {
            // current isolated or cross
            let is_taker_valid_risk_change = tx_state.risk_infos[TAKER_ACCOUNT_ID]
                .current_risk_parameters
                .is_valid_risk_change(builder, &new_taker_risk_info.current_risk_parameters);
            let is_taker_invalid_risk_change =
                builder.and_not(is_enabled, is_taker_valid_risk_change);
            cancel_taker = builder.or(cancel_taker, is_taker_invalid_risk_change);
        }
        {
            // cross collateral if position is isolated
            let taker_available_cross_collateral = get_available_collateral(
                builder,
                &tx_state.risk_infos[TAKER_ACCOUNT_ID].cross_risk_parameters,
            );
            let is_taker_has_enough_cross_collateral = {
                // new collateral = old collateral - margin_delta
                let collateral_gte_delta = builder
                    .is_gte_biguint(&taker_available_cross_collateral, &taker_margin_delta.abs);
                let is_delta_negative = builder.is_sign_negative(taker_margin_delta.sign);

                // If delta is negative, the new collateral is increasing. Otherwise, we make sure that old collateral is greater than or equal to the margin delta.
                builder.or(collateral_gte_delta, is_delta_negative)
            };

            let is_taker_invalid_risk_change =
                builder.and_not(is_enabled, is_taker_has_enough_cross_collateral);
            cancel_taker = builder.or(cancel_taker, is_taker_invalid_risk_change);
        }
        *cancel_taker_order = builder.select_bool(cancel_taker, is_enabled, *cancel_taker_order);
    }

    let mut cancel_maker = builder._false();
    // Check if maker is under initial margin and position is allowed, early return
    {
        // Check if position change is valid
        {
            let is_new_maker_position_valid = new_maker_position.is_valid(builder);
            let is_new_maker_position_invalid =
                builder.and_not(is_enabled, is_new_maker_position_valid);
            cancel_maker = builder.or(cancel_maker, is_new_maker_position_invalid);
        }
        // Check if risk change is valid
        {
            // current isolated or cross
            let is_maker_valid_risk_change = tx_state.risk_infos[MAKER_ACCOUNT_ID]
                .current_risk_parameters
                .is_valid_risk_change(builder, &new_maker_risk_info.current_risk_parameters);
            let is_maker_invalid_risk_change =
                builder.and_not(is_enabled, is_maker_valid_risk_change);
            cancel_maker = builder.or(cancel_maker, is_maker_invalid_risk_change);
        }
        {
            // cross collateral if position is isolated
            let maker_available_cross_collateral = get_available_collateral(
                builder,
                &tx_state.risk_infos[MAKER_ACCOUNT_ID].cross_risk_parameters,
            );
            let is_maker_has_enough_cross_collateral = {
                // new collateral = old collateral - margin_delta
                let collateral_gte_delta = builder
                    .is_gte_biguint(&maker_available_cross_collateral, &maker_margin_delta.abs);
                let is_delta_negative = builder.is_sign_negative(maker_margin_delta.sign);

                // If delta is negative, the new collateral is increasing. Otherwise, we make sure that old collateral is greater than or equal to the margin delta.
                builder.or(collateral_gte_delta, is_delta_negative)
            };
            let is_maker_invalid_risk_change =
                builder.and_not(is_enabled, is_maker_has_enough_cross_collateral);
            cancel_maker = builder.or(cancel_maker, is_maker_invalid_risk_change);
        }

        *cancel_maker_order = builder.select_bool(cancel_maker, is_enabled, *cancel_maker_order);
    }

    *update_status_flags = builder.and_not(*update_status_flags, *cancel_taker_order);
    *update_status_flags = builder.and_not(*update_status_flags, *cancel_maker_order);
}

fn is_valid_spot_trade(
    builder: &mut Builder,
    update_status_flags: &mut BoolTarget,
    tx_state: &mut TxState,
    is_taker_ask: BoolTarget,
    taker_base_balance_delta: &BigIntTarget,
    taker_quote_balance_delta: &BigIntTarget,
    maker_base_balance_delta: &BigIntTarget,
    maker_quote_balance_delta: &BigIntTarget,
    new_taker_risk_info: &RiskInfoTarget,
    new_maker_risk_info: &RiskInfoTarget,
    cancel_taker_order: &mut BoolTarget,
    cancel_maker_order: &mut BoolTarget,
) {
    let is_spot = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_SPOT);
    let is_enabled = builder.and(*update_status_flags, is_spot);

    let is_taker_unified = builder.is_equal_constant(
        tx_state.accounts[TAKER_ACCOUNT_ID].account_trading_mode,
        ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED as u64,
    );
    let is_maker_unified = builder.is_equal_constant(
        tx_state.accounts[MAKER_ACCOUNT_ID].account_trading_mode,
        ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED as u64,
    );

    let is_taker_base_asset_unified_margin = builder.multi_and(&[
        is_taker_unified,
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
        is_enabled,
    ]);
    let is_taker_quote_asset_unified_margin = builder.multi_and(&[
        is_taker_unified,
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        is_enabled,
    ]);
    let is_maker_base_asset_unified_margin = builder.multi_and(&[
        is_maker_unified,
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
        is_enabled,
    ]);
    let is_maker_quote_asset_unified_margin = builder.multi_and(&[
        is_maker_unified,
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        is_enabled,
    ]);

    let taker_base_balance = get_asset_balance_const(
        builder,
        PRODUCT_TYPE_SPOT,
        &tx_state.accounts[TAKER_ACCOUNT_ID],
        &tx_state.account_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
    );
    let new_taker_base_balance = builder.add_bigint_non_carry(
        &taker_base_balance,
        taker_base_balance_delta,
        BIG_U128_LIMBS,
    );
    let taker_quote_balance = get_asset_balance_const(
        builder,
        PRODUCT_TYPE_SPOT,
        &tx_state.accounts[TAKER_ACCOUNT_ID],
        &tx_state.account_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
    );
    let new_taker_quote_balance = builder.add_bigint_non_carry(
        &taker_quote_balance,
        taker_quote_balance_delta,
        BIG_U128_LIMBS,
    );

    let maker_base_balance = get_asset_balance_const(
        builder,
        PRODUCT_TYPE_SPOT,
        &tx_state.accounts[MAKER_ACCOUNT_ID],
        &tx_state.account_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
    );
    let new_maker_base_balance = builder.add_bigint_non_carry(
        &maker_base_balance,
        maker_base_balance_delta,
        BIG_U128_LIMBS,
    );
    let maker_quote_balance = get_asset_balance_const(
        builder,
        PRODUCT_TYPE_SPOT,
        &tx_state.accounts[MAKER_ACCOUNT_ID],
        &tx_state.account_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
    );
    let new_maker_quote_balance = builder.add_bigint_non_carry(
        &maker_quote_balance,
        maker_quote_balance_delta,
        BIG_U128_LIMBS,
    );

    let (mut valid_taker_base, _) = get_valid_asset_balance(
        builder,
        &new_taker_base_balance,
        is_taker_base_asset_unified_margin,
    );
    let (mut valid_taker_quote, _) = get_valid_asset_balance(
        builder,
        &new_taker_quote_balance,
        is_taker_quote_asset_unified_margin,
    );
    let (mut valid_maker_base, _) = get_valid_asset_balance(
        builder,
        &new_maker_base_balance,
        is_maker_base_asset_unified_margin,
    );
    let (mut valid_maker_quote, _) = get_valid_asset_balance(
        builder,
        &new_maker_quote_balance,
        is_maker_quote_asset_unified_margin,
    );

    let is_taker_bid = builder.not(is_taker_ask);

    let is_taker_collateral_negative = builder.is_sign_negative(
        new_taker_risk_info
            .cross_risk_parameters
            .collateral_with_funding
            .sign,
    );
    let is_taker_base_collateral_invalid = builder.multi_and(&[
        is_taker_base_asset_unified_margin,
        is_taker_ask,
        is_taker_collateral_negative,
    ]);
    valid_taker_base = builder.and_not(valid_taker_base, is_taker_base_collateral_invalid);
    let is_taker_quote_collateral_invalid = builder.multi_and(&[
        is_taker_quote_asset_unified_margin,
        is_taker_bid,
        is_taker_collateral_negative,
    ]);
    valid_taker_quote = builder.and_not(valid_taker_quote, is_taker_quote_collateral_invalid);

    let is_maker_collateral_negative = builder.is_sign_negative(
        new_maker_risk_info
            .cross_risk_parameters
            .collateral_with_funding
            .sign,
    );
    let is_maker_base_collateral_invalid = builder.multi_and(&[
        is_maker_base_asset_unified_margin,
        is_taker_bid,
        is_maker_collateral_negative,
    ]);
    valid_maker_base = builder.and_not(valid_maker_base, is_maker_base_collateral_invalid);
    let is_maker_quote_collateral_invalid = builder.multi_and(&[
        is_maker_quote_asset_unified_margin,
        is_taker_ask,
        is_maker_collateral_negative,
    ]);
    valid_maker_quote = builder.and_not(valid_maker_quote, is_maker_quote_collateral_invalid);

    let is_taker_using_margin_asset = builder.or(
        is_taker_base_asset_unified_margin,
        is_taker_quote_asset_unified_margin,
    );
    let is_taker_valid_risk_change = tx_state.risk_infos[TAKER_ACCOUNT_ID]
        .current_risk_parameters
        .is_valid_risk_change(builder, &new_taker_risk_info.current_risk_parameters);
    let is_taker_invalid_risk_change =
        builder.and_not(is_taker_using_margin_asset, is_taker_valid_risk_change);
    let is_taker_valid_risk_change_for_spot = builder.not(is_taker_invalid_risk_change);

    let is_maker_using_margin_asset = builder.or(
        is_maker_base_asset_unified_margin,
        is_maker_quote_asset_unified_margin,
    );
    let is_maker_valid_risk_change = tx_state.risk_infos[MAKER_ACCOUNT_ID]
        .current_risk_parameters
        .is_valid_risk_change(builder, &new_maker_risk_info.current_risk_parameters);
    let is_maker_invalid_risk_change =
        builder.and_not(is_maker_using_margin_asset, is_maker_valid_risk_change);
    let is_maker_valid_risk_change_for_spot = builder.not(is_maker_invalid_risk_change);

    let valid_taker_balances = builder.multi_and(&[
        valid_taker_base,
        valid_taker_quote,
        is_taker_valid_risk_change_for_spot,
    ]);
    let invalid_taker_balances = builder.and_not(is_enabled, valid_taker_balances);
    *cancel_taker_order = builder.or(invalid_taker_balances, *cancel_taker_order);
    *update_status_flags = builder.and_not(*update_status_flags, *cancel_taker_order);

    let valid_maker_balances = builder.multi_and(&[
        valid_maker_base,
        valid_maker_quote,
        is_maker_valid_risk_change_for_spot,
    ]);
    let invalid_maker_balances = builder.and_not(is_enabled, valid_maker_balances);
    *cancel_maker_order = builder.or(invalid_maker_balances, *cancel_maker_order);
    *update_status_flags = builder.and_not(*update_status_flags, *cancel_maker_order);
}

fn get_order_from_register(
    builder: &mut Builder,
    register: &BaseRegisterInfoTarget,
) -> OrderTarget {
    let zero = builder.zero();
    let quote = builder.mul(register.pending_size, register.pending_price);
    OrderTarget {
        price_index: register.pending_price,
        nonce_index: register.pending_nonce,

        ask_base_sum: builder.select(register.pending_is_ask, register.pending_size, zero),
        bid_base_sum: builder.select(register.pending_is_ask, zero, register.pending_size),
        ask_quote_sum: builder.select(register.pending_is_ask, quote, zero),
        bid_quote_sum: builder.select(register.pending_is_ask, zero, quote),
    }
}

fn get_account_order_from_register(register: &BaseRegisterInfoTarget) -> AccountOrderTarget {
    AccountOrderTarget {
        index_0: register.pending_order_index,
        index_1: register.pending_client_order_index,
        owner_account_index: register.account_index,

        order_index: register.pending_order_index,
        client_order_index: register.pending_client_order_index,

        initial_base_amount: register.pending_initial_size,
        price: register.pending_price,
        nonce: register.pending_nonce,
        remaining_base_amount: register.pending_size,
        is_ask: register.pending_is_ask,

        expiry: register.pending_expiry,
        time_in_force: register.pending_time_in_force,
        order_type: register.pending_type,
        reduce_only: register.pending_reduce_only,
        trigger_price: register.pending_trigger_price,

        trigger_status: register.pending_trigger_status,
        to_trigger_order_index0: register.pending_to_trigger_order_index0,
        to_trigger_order_index1: register.pending_to_trigger_order_index1,
        to_cancel_order_index0: register.pending_to_cancel_order_index0,
    }
}

pub fn increment_order_count_in_place(
    builder: &mut Builder,
    tx_state: &mut TxState,
    flag: BoolTarget,
    trigger_status: Target,
    reduce_only: Target,
) {
    tx_state.market.total_order_count = builder.add(tx_state.market.total_order_count, flag.target);

    tx_state.accounts[TAKER_ACCOUNT_ID].total_order_count = builder.add(
        tx_state.accounts[TAKER_ACCOUNT_ID].total_order_count,
        flag.target,
    );

    let is_spot = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_SPOT);
    let increment_flag = builder.and(is_spot, flag);
    tx_state.accounts[TAKER_ACCOUNT_ID].total_non_cross_order_count = builder.add(
        tx_state.accounts[TAKER_ACCOUNT_ID].total_non_cross_order_count,
        increment_flag.target,
    );

    let flag = builder.and_not(flag, is_spot); // Early return for spot

    let trigger_status_parent_order = builder.constant_from_u8(TRIGGER_STATUS_PARENT_ORDER);
    let is_not_trigger_status_parent_order =
        builder.is_not_equal(trigger_status, trigger_status_parent_order);
    let is_reduce_only = builder.is_not_zero(reduce_only);
    let position_tied_flag =
        builder.multi_and(&[flag, is_not_trigger_status_parent_order, is_reduce_only]);
    tx_state.positions[TAKER_ACCOUNT_ID].total_position_tied_order_count = builder.add(
        tx_state.positions[TAKER_ACCOUNT_ID].total_position_tied_order_count,
        position_tied_flag.target,
    );
    tx_state.positions[TAKER_ACCOUNT_ID].total_order_count = builder.add(
        tx_state.positions[TAKER_ACCOUNT_ID].total_order_count,
        flag.target,
    );

    let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
    let is_position_isolated = builder.is_equal(
        tx_state.positions[TAKER_ACCOUNT_ID].margin_mode,
        isolated_margin_mode,
    );
    let is_position_isolated_and_flag = builder.and(is_position_isolated, flag);
    tx_state.accounts[TAKER_ACCOUNT_ID].total_non_cross_order_count = builder.add(
        tx_state.accounts[TAKER_ACCOUNT_ID].total_non_cross_order_count,
        is_position_isolated_and_flag.target,
    );
}

pub fn decrement_order_count_in_place(
    builder: &mut Builder,
    tx_state: &mut TxState,
    account_slot: usize,
    flag: BoolTarget,
    trigger_status: Target,
    reduce_only: Target,
) {
    tx_state.market.total_order_count = builder.sub(tx_state.market.total_order_count, flag.target);

    tx_state.accounts[account_slot].total_order_count = builder.sub(
        tx_state.accounts[account_slot].total_order_count,
        flag.target,
    );

    let is_spot = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_SPOT);
    let decrement_flag = builder.and(is_spot, flag);
    tx_state.accounts[account_slot].total_non_cross_order_count = builder.sub(
        tx_state.accounts[account_slot].total_non_cross_order_count,
        decrement_flag.target,
    );

    let flag = builder.and_not(flag, is_spot); // Early return for spot

    let trigger_status_parent_order = builder.constant_from_u8(TRIGGER_STATUS_PARENT_ORDER);
    let is_not_trigger_status_parent_order =
        builder.is_not_equal(trigger_status, trigger_status_parent_order);
    let is_reduce_only = builder.is_not_zero(reduce_only);
    let position_tied_flag =
        builder.multi_and(&[flag, is_not_trigger_status_parent_order, is_reduce_only]);
    tx_state.positions[account_slot].total_position_tied_order_count = builder.sub(
        tx_state.positions[account_slot].total_position_tied_order_count,
        position_tied_flag.target,
    );

    tx_state.positions[account_slot].total_order_count = builder.sub(
        tx_state.positions[account_slot].total_order_count,
        flag.target,
    );

    let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
    let is_position_isolated = builder.is_equal(
        tx_state.positions[account_slot].margin_mode,
        isolated_margin_mode,
    );
    let is_position_isolated_and_flag = builder.and(is_position_isolated, flag);
    tx_state.accounts[account_slot].total_non_cross_order_count = builder.sub(
        tx_state.accounts[account_slot].total_non_cross_order_count,
        is_position_isolated_and_flag.target,
    );
}

pub fn get_locked_amount_and_ask_asset_index(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    market: &MarketTarget,
    base_amount: Target,
    price: Target,
    is_ask: BoolTarget,
) -> (BigUintTarget, Target) {
    let multiplier = {
        let ask_multiplier = builder.target_to_biguint(market.size_extension_multiplier);
        let bid_multiplier = {
            let quote_extension_multiplier_big =
                builder.target_to_biguint(market.quote_extension_multiplier);
            let price_big = builder.target_to_biguint_single_limb_unsafe(price);
            builder.mul_biguint(&price_big, &quote_extension_multiplier_big)
        };
        builder.select_biguint(is_ask, &ask_multiplier, &bid_multiplier)
    };
    let base_amount_big = builder.target_to_biguint(base_amount);

    let locked_amount = builder.mul_biguint(&base_amount_big, &multiplier);
    let (success, locked_amount) = builder.try_trim_biguint(&locked_amount, BIG_U96_LIMBS);
    builder.conditional_assert_true(is_enabled, success);

    (
        locked_amount,
        builder.select(is_ask, market.base_asset_id, market.quote_asset_id),
    )
}

pub fn increment_locked_balance_for_order(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    account_order: &AccountOrderTarget,
    market: &MarketTarget,
    account_assets: &mut [AccountAssetTarget; NB_ASSETS_PER_TX],
) {
    let (locked_amount, ask_asset_index) = get_locked_amount_and_ask_asset_index(
        builder,
        is_enabled,
        market,
        account_order.remaining_base_amount,
        account_order.price,
        account_order.is_ask,
    );

    let mut asset_found = builder._false();
    for asset in account_assets.iter_mut() {
        let new_locked_balance = builder.add_biguint(&asset.locked_balance, &locked_amount);
        let (success, new_locked_balance) =
            builder.try_trim_biguint(&new_locked_balance, BIG_U96_LIMBS);

        let is_asset_matched = builder.is_equal(asset.index_0, ask_asset_index);
        let flag = builder.and(is_enabled, is_asset_matched);
        let flag = builder.and_not(flag, asset_found);
        asset_found = builder.or(asset_found, is_asset_matched);

        builder.conditional_assert_true(flag, success);
        asset.locked_balance =
            builder.select_biguint(flag, &new_locked_balance, &asset.locked_balance);
    }
    builder.conditional_assert_true(is_enabled, asset_found);
}

fn decrement_locked_balance_for_partial_order(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    market: &MarketTarget,
    is_ask: BoolTarget,
    base_amount: Target,
    price: Target,
    account_assets: &mut [AccountAssetTarget; NB_ASSETS_PER_TX],
) {
    let (locked_amount, ask_asset_index) = get_locked_amount_and_ask_asset_index(
        builder,
        is_enabled,
        market,
        base_amount,
        price,
        is_ask,
    );
    let mut asset_found = builder._false();

    for asset in account_assets.iter_mut() {
        let (new_locked_balance, fail) =
            builder.try_sub_biguint(&asset.locked_balance, &locked_amount);

        let is_asset_matched = builder.is_equal(asset.index_0, ask_asset_index);
        let flag = builder.and(is_enabled, is_asset_matched);
        let flag = builder.and_not(flag, asset_found);
        asset_found = builder.or(asset_found, is_asset_matched);

        builder.conditional_assert_zero_u32(flag, fail);
        asset.locked_balance =
            builder.select_biguint(flag, &new_locked_balance, &asset.locked_balance);
    }
    builder.conditional_assert_true(is_enabled, asset_found);
}

pub fn decrement_locked_balance_for_order(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    account_order: &AccountOrderTarget,
    market: &MarketTarget,
    account_assets: &mut [AccountAssetTarget; NB_ASSETS_PER_TX],
) {
    let mut asset_found = builder._false();
    let (locked_amount, ask_asset_index) = get_locked_amount_and_ask_asset_index(
        builder,
        is_enabled,
        market,
        account_order.remaining_base_amount,
        account_order.price,
        account_order.is_ask,
    );
    for asset in account_assets.iter_mut() {
        let (new_locked_balance, fail) =
            builder.try_sub_biguint(&asset.locked_balance, &locked_amount);

        let is_asset_matched = builder.is_equal(asset.index_0, ask_asset_index);
        let flag = builder.and(is_enabled, is_asset_matched);
        let flag = builder.and_not(flag, asset_found);
        asset_found = builder.or(asset_found, is_asset_matched);

        builder.conditional_assert_zero_u32(flag, fail);
        asset.locked_balance =
            builder.select_biguint(flag, &new_locked_balance, &asset.locked_balance);
    }
    builder.conditional_assert_true(is_enabled, asset_found);
}

pub fn is_not_valid_reduce_only_direction(
    builder: &mut Builder,
    position_sign: SignTarget,
    is_ask: BoolTarget,
) -> BoolTarget {
    let positive_position = builder.is_sign_positive(position_sign);
    let is_ask_and_positive_position = builder.and(is_ask, positive_position);
    let negative_position = builder.is_sign_negative(position_sign);
    let is_bid_and_negative_position = builder.and_not(negative_position, is_ask);
    let is_valid_reduce_only_direction =
        builder.or(is_ask_and_positive_position, is_bid_and_negative_position);
    builder.not(is_valid_reduce_only_direction)
}

pub fn get_impact_prices(
    builder: &mut Builder,
    should_update_impact_price: BoolTarget,
    impact_ask_path: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    impact_ask_order: &OrderTarget,
    impact_bid_path: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    impact_bid_order: &OrderTarget,

    new_min_initial_margin_fraction: Target,
    old_quote_multiplier: Target,
) -> (Target, Target) {
    // Matching engine uses "base" amounts without ticks. USDC amount have ticks,
    // so we need to remove(by dividing) "Multiplier/Divider"

    let impact_usdc_amount_times_margin_tick = builder.constant(F::from_canonical_u64(
        MARGIN_TICK as u64 * IMPACT_USDC_AMOUNT,
    ));
    let (margin_tick_over_initial_margin, _) = builder.div_rem(
        impact_usdc_amount_times_margin_tick,
        new_min_initial_margin_fraction,
        MARGIN_FRACTION_BITS,
    );

    let (impact_notional_amount, _) = builder.div_rem(
        margin_tick_over_initial_margin,
        old_quote_multiplier,
        QUOTE_MULTIPLIER_BITS,
    );

    let _true = builder._true();
    let impact_ask_price = get_impact_price(
        builder,
        should_update_impact_price,
        impact_notional_amount,
        impact_ask_path,
        impact_ask_order,
        _true,
    );

    let _false = builder._false();
    let impact_bid_price = get_impact_price(
        builder,
        should_update_impact_price,
        impact_notional_amount,
        impact_bid_path,
        impact_bid_order,
        _false,
    );

    (impact_ask_price, impact_bid_price)
}

pub fn cancel_position_tied_account_orders(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    market_index: Target,
    owner_account_index: Target,
    position_tied_order_count: Target,
) {
    let cancel_position_tied_account_orders =
        builder.constant_from_u8(CANCEL_POSITION_TIED_ACCOUNT_ORDERS);
    let cancel_position_tied_account_orders_instruction = &BaseRegisterInfoTarget {
        instruction_type: cancel_position_tied_account_orders,
        market_index,
        account_index: owner_account_index,
        pending_size: position_tied_order_count,
        pending_order_index: builder.zero(),
        pending_client_order_index: builder.zero(),
        pending_price: builder.zero(),
        pending_nonce: builder.zero(),
        pending_is_ask: builder._false(),
        pending_initial_size: builder.zero(),
        pending_expiry: builder.zero(),
        pending_time_in_force: builder.zero(),
        pending_type: builder.zero(),
        pending_reduce_only: builder.zero(),
        generic_field_0: builder.zero(),
        pending_trigger_price: builder.zero(),
        pending_trigger_status: builder.zero(),
        pending_to_trigger_order_index0: builder.zero(),
        pending_to_trigger_order_index1: builder.zero(),
        pending_to_cancel_order_index0: builder.zero(),
    };
    tx_state.insert_to_instruction_stack(
        builder,
        is_enabled,
        cancel_position_tied_account_orders_instruction,
    );
}

pub fn trigger_child_orders(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    market_index: Target,
    owner_account_index: Target,
    child_order_index_0: Target,
    child_order_index_1: Target,
    pending_size: Target,
) {
    let trigger_child_order_0_instruction = get_trigger_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_0,
        pending_size,
    );
    let does_child_order_0_exist = builder.is_not_zero(child_order_index_0);
    let child_order_0_flag = builder.and(is_enabled, does_child_order_0_exist);
    tx_state.insert_to_instruction_stack(
        builder,
        child_order_0_flag,
        &trigger_child_order_0_instruction,
    );

    let trigger_child_order_1_instruction = get_trigger_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_1,
        pending_size,
    );
    let does_child_order_1_exist = builder.is_not_zero(child_order_index_1);
    let child_order_1_flag = builder.and(is_enabled, does_child_order_1_exist);
    tx_state.insert_to_instruction_stack(
        builder,
        child_order_1_flag,
        &trigger_child_order_1_instruction,
    );
}

fn get_trigger_child_order_instruction(
    builder: &mut Builder,
    market_index: Target,
    owner_account_index: Target,
    child_order_index: Target,
    pending_size: Target,
) -> BaseRegisterInfoTarget {
    let trigger_child_order = builder.constant_from_u8(TRIGGER_CHILD_ORDER);
    BaseRegisterInfoTarget {
        instruction_type: trigger_child_order,
        market_index,
        account_index: owner_account_index,
        pending_size,
        pending_order_index: child_order_index,
        pending_client_order_index: builder.zero(),
        pending_price: builder.zero(),
        pending_nonce: builder.zero(),
        pending_is_ask: builder._false(),
        pending_initial_size: builder.zero(),
        pending_expiry: builder.zero(),
        pending_time_in_force: builder.zero(),
        pending_type: builder.zero(),
        pending_reduce_only: builder.zero(),
        generic_field_0: builder.zero(),
        pending_trigger_price: builder.zero(),
        pending_trigger_status: builder.zero(),
        pending_to_trigger_order_index0: builder.zero(),
        pending_to_trigger_order_index1: builder.zero(),
        pending_to_cancel_order_index0: builder.zero(),
    }
}

pub fn cancel_child_orders(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    market_index: Target,
    owner_account_index: Target,
    child_order_index_0: Target,
    child_order_index_1: Target,
) {
    let cancel_child_order_0_instruction = get_cancel_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_0,
    );
    let does_child_order_0_exist = builder.is_not_zero(child_order_index_0);
    let child_order_0_flag = builder.and(is_enabled, does_child_order_0_exist);
    tx_state.insert_to_instruction_stack(
        builder,
        child_order_0_flag,
        &cancel_child_order_0_instruction,
    );

    let cancel_child_order_1_instruction = get_cancel_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_1,
    );
    let does_child_order_1_exist = builder.is_not_zero(child_order_index_1);
    let child_order_1_flag = builder.and(is_enabled, does_child_order_1_exist);
    tx_state.insert_to_instruction_stack(
        builder,
        child_order_1_flag,
        &cancel_child_order_1_instruction,
    );
}

fn get_cancel_child_order_instruction(
    builder: &mut Builder,
    market_index: Target,
    owner_account_index: Target,
    child_order_index: Target,
) -> BaseRegisterInfoTarget {
    BaseRegisterInfoTarget {
        instruction_type: builder.constant_from_u8(CANCEL_SINGLE_ACCOUNT_ORDER),
        market_index,
        account_index: owner_account_index,
        pending_order_index: child_order_index,
        ..BaseRegisterInfoTarget::empty(builder)
    }
}

fn get_impact_price(
    builder: &mut Builder,
    should_update_impact_price: BoolTarget,
    impact_notional_amount: Target,
    order_path: &[OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    order: &OrderTarget,
    is_ask: BoolTarget,
) -> Target {
    let zero = builder.zero();
    let is_bid = builder.not(is_ask);

    let order_merkle_helper =
        order_indexes_to_merkle_path(builder, order.price_index, order.nonce_index);
    let (orders_before_base_amount, orders_before_quote_amount) =
        get_quote(builder, is_bid, order, order_path, &order_merkle_helper);

    let leaf_quote_amount = builder.select(is_bid, order.bid_quote_sum, order.ask_quote_sum);
    let impact_path_quote_amount = builder.add(orders_before_quote_amount, leaf_quote_amount);

    let total_quote_amount = builder.select(
        is_bid,
        order_path[ORDER_BOOK_MERKLE_LEVELS - 1].bid_quote_sum,
        order_path[ORDER_BOOK_MERKLE_LEVELS - 1].ask_quote_sum,
    );

    let not_enough_liquidity = builder.is_gt(impact_notional_amount, total_quote_amount, 64);
    // Verify if given path points to the last order to iterate until impact notional amount.
    // orders_before_quote_amount should be stricly smaller than impact_notional_amount
    // and impact_path_quote_amount should be greater than or equal to impact_notional_amount
    let enough_liquidity = builder.not(not_enough_liquidity);
    let impact_path_checks_enabled = builder.and(should_update_impact_price, enough_liquidity);
    builder.conditional_assert_lt(
        impact_path_checks_enabled,
        orders_before_quote_amount,
        impact_notional_amount,
        64,
    );
    builder.conditional_assert_lte(
        impact_path_checks_enabled,
        impact_notional_amount,
        impact_path_quote_amount,
        64,
    );

    let remaining_quote_amount_for_leaf =
        builder.sub(impact_notional_amount, orders_before_quote_amount);
    let leaf_included_base_amount = builder.ceil_div(
        remaining_quote_amount_for_leaf,
        order.price_index,
        ORDER_PRICE_BITS,
    );
    let leaf_included_quote_amount = builder.mul(leaf_included_base_amount, order.price_index);

    let total_included_base_amount =
        builder.add(orders_before_base_amount, leaf_included_base_amount);
    let total_included_quote_amount =
        builder.add(orders_before_quote_amount, leaf_included_quote_amount);

    let (impact_price_div, _) = builder.div_rem(
        total_included_quote_amount,
        total_included_base_amount,
        ORDER_BASE_AMOUNT_BITS,
    );
    let impact_price_ceil_div = builder.ceil_div(
        total_included_quote_amount,
        total_included_base_amount,
        ORDER_BASE_AMOUNT_BITS,
    );

    let impact_price = builder.select(is_bid, impact_price_div, impact_price_ceil_div);

    builder.select(enough_liquidity, impact_price, zero)
}

// Returns trimmed balance and a boolean that indicates whether the original balance is valid (non-negative for spot only assets and fits in the trimmed size)
fn get_valid_asset_balance(
    builder: &mut Builder,
    balance: &BigIntTarget,
    is_asset_unified_margin: BoolTarget,
) -> (BoolTarget, BigIntTarget) {
    let (valid_balance_range, trimmed_balance) =
        builder.try_trim_biguint(&balance.abs, BIG_U96_LIMBS);

    let is_balance_negative = builder.is_sign_negative(balance.sign);
    let is_balance_non_negative = builder.not(is_balance_negative);
    let is_sign_valid = builder.or(is_asset_unified_margin, is_balance_non_negative);

    let is_valid = builder.and(is_sign_valid, valid_balance_range);

    (
        is_valid,
        BigIntTarget {
            abs: trimmed_balance,
            sign: balance.sign,
        },
    )
}
