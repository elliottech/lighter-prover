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
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::hints::CircuitBuilderHints;
use crate::liquidation::get_available_usdc_collateral;
use crate::order_book_tree_helpers::order_indexes_to_merkle_path;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::tx_attributes::is_integrator_fee_disabled;
use crate::types::account::AccountTarget;
use crate::types::account_asset::AccountAssetTarget;
use crate::types::account_order::{AccountOrderTarget, OrderFlags, select_account_order_target};
use crate::types::account_position::AccountPositionTarget;
use crate::types::asset::is_universal_asset;
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::margined_asset::MarginedAssetTarget;
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
    let zero = builder.zero();
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

    // Make a copy of attribute related variables
    let integrator_taker_fee_collector_index = tx_state.register_stack[0].generic_field_1;
    let is_integrator_taker_fee_disabled =
        is_integrator_fee_disabled(builder, integrator_taker_fee_collector_index);
    let taker_order_flags_value = builder.select(
        is_integrator_taker_fee_disabled,
        tx_state.register_stack[0].generic_field_2,
        zero,
    );
    let taker_order_flags = OrderFlags::from_target(builder, taker_order_flags_value);
    let integrator_taker_fee = builder.select(
        is_integrator_taker_fee_disabled,
        zero,
        tx_state.register_stack[0].generic_field_2,
    );
    let integrator_maker_fee_collector_index =
        tx_state.account_order.integrator_fee_collector_index;
    let is_integrator_maker_fee_disabled =
        is_integrator_fee_disabled(builder, integrator_maker_fee_collector_index);
    let integrator_maker_fee = builder.select(
        is_integrator_maker_fee_disabled,
        zero,
        tx_state.account_order.integrator_maker_fee,
    );

    let is_expire_maker_mode = builder.is_equal_constant(
        taker_order_flags.self_trade_behavior_mode,
        SELF_TRADE_BEHAVIOR_EXPIRE_MAKER,
    );
    let is_expire_taker_mode = builder.is_equal_constant(
        taker_order_flags.self_trade_behavior_mode,
        SELF_TRADE_BEHAVIOR_EXPIRE_TAKER,
    );
    let is_expire_both_mode = builder.is_equal_constant(
        taker_order_flags.self_trade_behavior_mode,
        SELF_TRADE_BEHAVIOR_EXPIRE_BOTH,
    );
    let is_reduce_mode = builder.is_equal_constant(
        taker_order_flags.self_trade_behavior_mode,
        SELF_TRADE_BEHAVIOR_REDUCE,
    );
    let is_maker_order_expired =
        builder.is_lte(tx_state.account_order.expiry, timestamp, TIMESTAMP_BITS);

    let is_account_index_equal = builder.is_equal(
        tx_state.account_order.owner_account_index,
        tx_state.register_stack[0].account_index,
    );

    // Handle self trade case where account indices match but there's integrator fee specified (reduce both sides)
    {
        let flag = builder.and_not(is_account_index_equal, is_integrator_taker_fee_disabled);

        // Order expiry first
        {
            let order_expiry_flag =
                builder.multi_and(&[update_status_flags, flag, is_maker_order_expired]);
            cancel_maker_order =
                builder.select_bool(order_expiry_flag, update_status_flags, cancel_maker_order);
            update_status_flags =
                builder.select_bool(order_expiry_flag, _false, update_status_flags);
        }

        apply_self_trade_reduce(
            builder,
            flag,
            is_post_only,
            is_spot,
            is_maker_limit_order,
            optimistic_trade_amount,
            &mut update_status_flags,
            &mut cancel_taker_order,
            &mut cancel_maker_order,
            tx_state,
        );
    }

    let is_self_trade_same_account_index = {
        let is_account_index_equality_mode =
            taker_order_flags.is_account_index_equality_mode(builder);
        builder.multi_and(&[is_account_index_equality_mode, is_account_index_equal])
    };
    // Handle self trade case for account index match
    {
        // Order expiry first
        {
            let order_expiry_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_account_index,
                is_maker_order_expired,
            ]);
            cancel_maker_order =
                builder.select_bool(order_expiry_flag, update_status_flags, cancel_maker_order);
            update_status_flags =
                builder.select_bool(order_expiry_flag, _false, update_status_flags);
        }

        // Expire maker mode
        {
            let expire_maker_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_account_index,
                is_expire_maker_mode,
            ]);
            cancel_maker_order =
                builder.select_bool(expire_maker_flag, update_status_flags, cancel_maker_order);
            update_status_flags =
                builder.select_bool(expire_maker_flag, _false, update_status_flags);
        }

        // Expire taker mode
        {
            let expire_taker_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_account_index,
                is_expire_taker_mode,
            ]);
            cancel_taker_order =
                builder.select_bool(expire_taker_flag, update_status_flags, cancel_taker_order);
            update_status_flags =
                builder.select_bool(expire_taker_flag, _false, update_status_flags);
        }

        // Expire both mode
        {
            let expire_both_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_account_index,
                is_expire_both_mode,
            ]);
            cancel_taker_order =
                builder.select_bool(expire_both_flag, update_status_flags, cancel_taker_order);
            cancel_maker_order =
                builder.select_bool(expire_both_flag, update_status_flags, cancel_maker_order);
            update_status_flags =
                builder.select_bool(expire_both_flag, _false, update_status_flags);
        }

        // Reduce mode - Reduce from both if taker is not post only
        {
            let reduce_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_account_index,
                is_reduce_mode,
            ]);
            apply_self_trade_reduce(
                builder,
                reduce_flag,
                is_post_only,
                is_spot,
                is_maker_limit_order,
                optimistic_trade_amount,
                &mut update_status_flags,
                &mut cancel_taker_order,
                &mut cancel_maker_order,
                tx_state,
            );
        }
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

    let is_self_trade_same_master_account_index = {
        let is_master_account_index_equality_mode =
            taker_order_flags.is_master_account_index_equality_mode();
        let is_master_account_index_equal = builder.is_equal(
            tx_state.accounts[MAKER_ACCOUNT_ID].master_account_index,
            tx_state.accounts[TAKER_ACCOUNT_ID].master_account_index,
        );
        builder.multi_and(&[
            is_master_account_index_equality_mode,
            is_master_account_index_equal,
            is_integrator_taker_fee_disabled,
        ])
    };
    // Handle self trade case for master account index match
    {
        // Expire maker mode
        {
            let expire_maker_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_master_account_index,
                is_expire_maker_mode,
            ]);
            cancel_maker_order =
                builder.select_bool(expire_maker_flag, update_status_flags, cancel_maker_order);
            update_status_flags =
                builder.select_bool(expire_maker_flag, _false, update_status_flags);
        }

        // Expire taker mode
        {
            let expire_taker_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_master_account_index,
                is_expire_taker_mode,
            ]);
            cancel_taker_order =
                builder.select_bool(expire_taker_flag, update_status_flags, cancel_taker_order);
            update_status_flags =
                builder.select_bool(expire_taker_flag, _false, update_status_flags);
        }

        // Expire both mode
        {
            let expire_both_flag = builder.multi_and(&[
                update_status_flags,
                is_self_trade_same_master_account_index,
                is_expire_both_mode,
            ]);
            cancel_taker_order =
                builder.select_bool(expire_both_flag, update_status_flags, cancel_taker_order);
            cancel_maker_order =
                builder.select_bool(expire_both_flag, update_status_flags, cancel_maker_order);
            update_status_flags =
                builder.select_bool(expire_both_flag, _false, update_status_flags);
        }

        // Expire maker and Reduce shouldn't happen at the same time, continue
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
        taker_fee: SignedTarget::new_unsafe(
            builder.add(tx_state.taker_fee.target, integrator_taker_fee),
        ),
        maker_fee: SignedTarget::new_unsafe(
            builder.add(tx_state.maker_fee.target, integrator_maker_fee),
        ),
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
        assets: &tx_state
            .assets
            .iter()
            .take(2)
            .cloned()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(), // Take first 2 assets
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
    ) = apply_spot_trade(
        builder,
        update_status_flags,
        &apply_trade_params,
        &apply_spot_trade_params,
    );

    let (
        total_supplied_amounts,
        taker_asset_balances,
        taker_margin_asset_balances,
        taker_strategy_balance,
        maker_asset_balances,
        maker_margin_asset_balances,
        maker_strategy_balance,
        new_taker_risk_info_spot,
        new_maker_risk_info_spot,
    ) = is_valid_spot_trade(
        builder,
        &mut update_status_flags,
        tx_state,
        &apply_trade_params,
        is_taker_ask,
        &taker_base_balance_delta,
        &taker_quote_balance_delta,
        &maker_base_balance_delta,
        &maker_quote_balance_delta,
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

            let liquidation_fee = builder.select(
                is_perps,
                tx_state.market.liquidation_fee,
                tx_state.margined_asset[BASE_ASSET_ID].liquidation_fee,
            );
            let new_taker_fee = builder.min(&[liquidation_fee, price_diff_rate], 64);

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
            // Fee account is maker or taker case is already handled in [`apply_spot_trade`], so here we just apply deltas
            let update_assets_flag = builder.and(update_status_flags, is_spot);
            let _spot = builder.constant_u64(PRODUCT_TYPE_SPOT);

            tx_state.account_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance = builder
                .select_biguint(
                    update_assets_flag,
                    &taker_asset_balances[BASE_ASSET_ID],
                    &tx_state.account_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
                );

            tx_state.account_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance = builder
                .select_biguint(
                    update_assets_flag,
                    &taker_asset_balances[QUOTE_ASSET_ID],
                    &tx_state.account_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
                );
            tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance = builder
                .select_bigint(
                    update_assets_flag,
                    &taker_margin_asset_balances[BASE_ASSET_ID],
                    &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
                );
            tx_state.account_margined_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance = builder
                .select_bigint(
                    update_assets_flag,
                    &taker_margin_asset_balances[QUOTE_ASSET_ID],
                    &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
                );
            tx_state.strategies[TAKER_ACCOUNT_ID] = builder.select_bigint(
                update_assets_flag,
                &taker_strategy_balance,
                &tx_state.strategies[TAKER_ACCOUNT_ID],
            );

            // Apply receiver changes
            tx_state.account_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance = builder
                .select_biguint(
                    update_assets_flag,
                    &maker_asset_balances[BASE_ASSET_ID],
                    &tx_state.account_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
                );
            tx_state.account_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance = builder
                .select_biguint(
                    update_assets_flag,
                    &maker_asset_balances[QUOTE_ASSET_ID],
                    &tx_state.account_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
                );
            tx_state.account_margined_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance = builder
                .select_bigint(
                    update_assets_flag,
                    &maker_margin_asset_balances[BASE_ASSET_ID],
                    &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
                );
            tx_state.account_margined_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance = builder
                .select_bigint(
                    update_assets_flag,
                    &maker_margin_asset_balances[QUOTE_ASSET_ID],
                    &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
                );
            tx_state.strategies[MAKER_ACCOUNT_ID] = builder.select_bigint(
                update_assets_flag,
                &maker_strategy_balance,
                &tx_state.strategies[MAKER_ACCOUNT_ID],
            );

            tx_state.margined_asset[BASE_ASSET_ID].total_supplied_amount = builder.select_biguint(
                update_assets_flag,
                &total_supplied_amounts[BASE_ASSET_ID],
                &tx_state.margined_asset[BASE_ASSET_ID].total_supplied_amount,
            );
            tx_state.margined_asset[QUOTE_ASSET_ID].total_supplied_amount = builder.select_biguint(
                update_assets_flag,
                &total_supplied_amounts[QUOTE_ASSET_ID],
                &tx_state.margined_asset[QUOTE_ASSET_ID].total_supplied_amount,
            );

            let is_fee_account_unified = tx_state.accounts[FEE_ACCOUNT_ID].is_unified_mode();
            let is_fee_account_insurance_fund = builder.is_equal_constant(
                tx_state.accounts[FEE_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            AccountTarget::apply_asset_delta(
                builder,
                update_assets_flag,
                _spot,
                tx_state.asset_indices[BASE_ASSET_ID],
                &mut tx_state.margined_asset[BASE_ASSET_ID],
                tx_state.is_asset_used_as_margin[FEE_ACCOUNT_ID][BASE_ASSET_ID],
                &fee_base_balance_delta,
                is_fee_account_unified,
                is_fee_account_insurance_fund,
                &mut tx_state.account_assets[FEE_ACCOUNT_ID][BASE_ASSET_ID].balance,
                &mut tx_state.account_margined_assets[FEE_ACCOUNT_ID][BASE_ASSET_ID].balance,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
                false,
            );
            AccountTarget::apply_asset_delta(
                builder,
                update_assets_flag,
                _spot,
                tx_state.asset_indices[QUOTE_ASSET_ID],
                &mut tx_state.margined_asset[QUOTE_ASSET_ID],
                tx_state.is_asset_used_as_margin[FEE_ACCOUNT_ID][QUOTE_ASSET_ID],
                &fee_quote_balance_delta,
                is_fee_account_unified,
                is_fee_account_insurance_fund,
                &mut tx_state.account_assets[FEE_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
                &mut tx_state.account_margined_assets[FEE_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
                false,
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
            let is_in_liquidation = new_taker_risk_info
                .current_risk_parameters
                .is_in_liquidation(builder);
            let is_not_in_liquidation_and_is_liquidation_order =
                builder.and_not(is_liquidation_order, is_in_liquidation);
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

        // Update margins for perps
        {
            let flag = builder.and(update_status_flags, is_perps);

            // We are not handling unified accounts because perps trades always happen with collateral balance.
            // Update Taker
            let mut new_taker_collateral = builder.select_bigint(
                is_taker_position_isolated,
                &new_taker_risk_info.cross_risk_parameters.usdc_collateral,
                &new_taker_risk_info.current_risk_parameters.usdc_collateral,
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
                    .usdc_collateral,
                &tx_state.risk_infos[TAKER_ACCOUNT_ID]
                    .current_risk_parameters
                    .usdc_collateral,
            );
            let taker_collateral_delta =
                builder.sub_bigint(&new_taker_collateral, &old_taker_collateral);

            tx_state.accounts[TAKER_ACCOUNT_ID].apply_collateral_delta(
                builder,
                flag,
                &taker_collateral_delta,
                &mut tx_state.strategies[TAKER_ACCOUNT_ID],
                &mut tx_state.account_margined_assets[TAKER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
            );

            // Update Maker
            let mut new_maker_collateral = builder.select_bigint(
                is_maker_position_isolated,
                &new_maker_risk_info.cross_risk_parameters.usdc_collateral,
                &new_maker_risk_info.current_risk_parameters.usdc_collateral,
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
                    .usdc_collateral,
                &tx_state.risk_infos[MAKER_ACCOUNT_ID]
                    .current_risk_parameters
                    .usdc_collateral,
            );
            let maker_collateral_delta =
                builder.sub_bigint(&new_maker_collateral, &old_maker_collateral);

            tx_state.accounts[MAKER_ACCOUNT_ID].apply_collateral_delta(
                builder,
                flag,
                &maker_collateral_delta,
                &mut tx_state.strategies[MAKER_ACCOUNT_ID],
                &mut tx_state.account_margined_assets[MAKER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
            );

            // Update Fee account
            // Fee payments always applied to cross collateral
            tx_state.accounts[FEE_ACCOUNT_ID].apply_collateral_delta(
                builder,
                flag,
                &fee_account_collateral_delta,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
                &mut tx_state.account_margined_assets[FEE_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
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
    let register_account_order =
        get_account_order_from_register(builder, &tx_state.register_stack[0]);
    tx_state.register_stack.pop_front(builder, pop_register);

    // Cancel maker order if needed
    let cancel_maker_order_from_first_account =
        builder.and(cancel_maker_order, is_account_index_equal);
    let cancel_maker_order_from_second_account =
        builder.and_not(cancel_maker_order, is_account_index_equal);
    [
        (cancel_maker_order_from_first_account, TAKER_ACCOUNT_ID),
        (cancel_maker_order_from_second_account, MAKER_ACCOUNT_ID),
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
        7,
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
        7,
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
    let trigger_taker_child_orders_flag =
        builder.and_not(cancel_taker_order, is_taker_filled_size_zero);
    let cancel_taker_child_orders_flag = builder.and(is_taker_filled_size_zero, cancel_taker_order);
    cancel_child_orders(
        builder,
        cancel_taker_child_orders_flag,
        tx_state,
        market_index,
        taker_account_index,
        taker_child_order_index_0,
        taker_child_order_index_1,
        5,
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
        5,
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
        3,
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
        2,
    );

    // Insert register for partner fees
    {
        let fee_account_exists = builder.not(tx_state.is_new_account[FEE_ACCOUNT_ID]);
        let trade_base_is_not_zero = builder.is_not_zero(trade_base);
        let flag = builder.multi_and(&[
            update_status_flags,
            trade_base_is_not_zero,
            fee_account_exists,
        ]);

        let usdc_asset_index = builder.constant_u64(USDC_ASSET_INDEX);
        let default_strategy_index = builder.constant_usize(DEFAULT_STRATEGY_INDEX);

        let trade_quote = trade_quote.target;

        let integrator_taker_fee_big =
            builder.target_to_biguint_single_limb_unsafe(integrator_taker_fee);
        let integrator_maker_fee_big =
            builder.target_to_biguint_single_limb_unsafe(integrator_maker_fee);

        let mut taker_fee_amount;
        let mut maker_fee_amount;

        let mut taker_asset_index;
        let mut maker_asset_index;

        let mut strategy_index;
        let mut route_type;

        // Perps
        {
            strategy_index = tx_state.market_details.strategy_index;
            route_type = builder.constant_u64(ROUTE_TYPE_PERPS);

            let usdc_to_collateral_multiplier =
                builder.constant_u32(USDC_TO_COLLATERAL_MULTIPLIER).0;
            let usdc_to_collateral_multiplier =
                builder.target_to_biguint_single_limb_unsafe(usdc_to_collateral_multiplier);

            let trade_quote = builder.mul(trade_quote, tx_state.market_details.quote_multiplier);
            let trade_quote_big = builder.target_to_biguint(trade_quote);

            let extended_taker_fee = builder.mul_biguint_non_carry(
                &integrator_taker_fee_big,
                &trade_quote_big,
                BIG_U96_LIMBS,
            );
            (taker_fee_amount, _) =
                builder.div_rem_biguint(&extended_taker_fee, &usdc_to_collateral_multiplier);
            taker_asset_index = usdc_asset_index;

            let extended_maker_fee = builder.mul_biguint_non_carry(
                &integrator_maker_fee_big,
                &trade_quote_big,
                BIG_U96_LIMBS,
            );
            (maker_fee_amount, _) =
                builder.div_rem_biguint(&extended_maker_fee, &usdc_to_collateral_multiplier);
            maker_asset_index = usdc_asset_index;
        }

        // Spot
        {
            let flag = builder.and(flag, is_spot);

            strategy_index = builder.select(flag, default_strategy_index, strategy_index);
            let route_type_spot = builder.constant_u64(ROUTE_TYPE_SPOT);
            route_type = builder.select(flag, route_type_spot, route_type);

            let fee_tick = builder.constant_u64(FEE_TICK);

            let base_fee_amount_big = {
                let (base_fee_multiplier, _) = builder.div_rem(
                    tx_state.market.size_extension_multiplier,
                    fee_tick,
                    FEE_BITS,
                );
                let base_fee_multiplier = builder.target_to_biguint(base_fee_multiplier);
                let trade_base_big = builder.target_to_biguint(trade_base);
                let trade_base_multiplied = builder.mul_biguint_non_carry(
                    &trade_base_big,
                    &base_fee_multiplier,
                    BIG_U96_LIMBS,
                );
                let base_fee_selected = builder.select_biguint(
                    is_taker_ask,
                    &integrator_maker_fee_big,
                    &integrator_taker_fee_big,
                );
                let extended_base_fee_amount = builder.mul_biguint_non_carry(
                    &trade_base_multiplied,
                    &base_fee_selected,
                    BIG_U96_LIMBS,
                );
                let (base_fee_amount_big, _) = builder.div_rem_biguint(
                    &extended_base_fee_amount,
                    &tx_state.assets[BASE_ASSET_ID].extension_multiplier,
                );

                base_fee_amount_big
            };

            let quote_fee_amount_big = {
                let (quote_fee_multiplier, _) = builder.div_rem(
                    tx_state.market.quote_extension_multiplier,
                    fee_tick,
                    FEE_BITS,
                );
                let quote_fee_multiplier = builder.target_to_biguint(quote_fee_multiplier);
                let trade_quote_big = builder.target_to_biguint(trade_quote);
                let trade_quote_multiplied = builder.mul_biguint_non_carry(
                    &trade_quote_big,
                    &quote_fee_multiplier,
                    BIG_U96_LIMBS,
                );
                let quote_fee_selected = builder.select_biguint(
                    is_taker_ask,
                    &integrator_taker_fee_big,
                    &integrator_maker_fee_big,
                );
                let extended_quote_fee_amount = builder.mul_biguint_non_carry(
                    &trade_quote_multiplied,
                    &quote_fee_selected,
                    BIG_U96_LIMBS,
                );
                let (quote_fee_amount_big, _) = builder.div_rem_biguint(
                    &extended_quote_fee_amount,
                    &tx_state.assets[QUOTE_ASSET_ID].extension_multiplier,
                );
                quote_fee_amount_big
            };

            let spot_taker_fee_amount =
                builder.select_biguint(is_taker_ask, &quote_fee_amount_big, &base_fee_amount_big);
            taker_fee_amount =
                builder.select_biguint(flag, &spot_taker_fee_amount, &taker_fee_amount);
            let spot_taker_asset_index = builder.select(
                is_taker_ask,
                tx_state.market.quote_asset_id,
                tx_state.market.base_asset_id,
            );
            taker_asset_index = builder.select(flag, spot_taker_asset_index, taker_asset_index);

            let spot_maker_fee_amount =
                builder.select_biguint(is_taker_ask, &base_fee_amount_big, &quote_fee_amount_big);
            maker_fee_amount =
                builder.select_biguint(flag, &spot_maker_fee_amount, &maker_fee_amount);
            let spot_maker_asset_index = builder.select(
                is_taker_ask,
                tx_state.market.base_asset_id,
                tx_state.market.quote_asset_id,
            );
            maker_asset_index = builder.select(flag, spot_maker_asset_index, maker_asset_index);
        }

        let max_integrator_fee_amount = builder.constant_usize(MAX_INTEGRATOR_FEE_AMOUNT);
        let taker_fee = builder.biguint_to_target_safe(&taker_fee_amount);
        let taker_fee = builder.min(
            &[taker_fee, max_integrator_fee_amount],
            MAX_INTEGRATOR_FEE_AMOUNT_BITS,
        );
        let maker_fee = builder.biguint_to_target_safe(&maker_fee_amount);
        let maker_fee = builder.min(
            &[maker_fee, max_integrator_fee_amount],
            MAX_INTEGRATOR_FEE_AMOUNT_BITS,
        );

        // Taker
        {
            let integrator_taker_fee_exists = builder.is_not_zero(taker_fee);
            let integrator_fee_collector_is_fee_collector = builder.is_equal(
                integrator_taker_fee_collector_index,
                tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            );
            let cond = builder.and_not(
                integrator_taker_fee_exists,
                integrator_fee_collector_is_fee_collector,
            );
            let flag = builder.and(flag, cond);

            let new_register = BaseRegisterInfoTarget {
                instruction_type: builder.constant_u64(TRANSFER_ASSET as u64),
                account_index: tx_state.accounts[FEE_ACCOUNT_ID].account_index,
                generic_field_0: integrator_taker_fee_collector_index,
                generic_field_2: taker_asset_index,
                generic_field_3: strategy_index,
                pending_type: route_type,
                pending_size: taker_fee,

                ..BaseRegisterInfoTarget::empty(builder)
            };
            tx_state.put_to_instruction_stack_unsafe(builder, flag, &new_register, 1);
        }

        // Maker
        {
            let integrator_maker_fee_exists = builder.is_not_zero(maker_fee);
            let integrator_fee_collector_is_fee_collector = builder.is_equal(
                integrator_maker_fee_collector_index,
                tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            );
            let cond = builder.and_not(
                integrator_maker_fee_exists,
                integrator_fee_collector_is_fee_collector,
            );
            let flag = builder.and(flag, cond);

            let new_register = BaseRegisterInfoTarget {
                instruction_type: builder.constant_u64(TRANSFER_ASSET as u64),
                account_index: tx_state.accounts[FEE_ACCOUNT_ID].account_index,
                generic_field_0: integrator_maker_fee_collector_index,
                generic_field_2: maker_asset_index,
                generic_field_3: strategy_index,
                pending_type: route_type,
                pending_size: maker_fee,

                ..BaseRegisterInfoTarget::empty(builder)
            };
            tx_state.put_to_instruction_stack_unsafe(builder, flag, &new_register, 0);
        }
    }
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
            let taker_available_cross_collateral = get_available_usdc_collateral(
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
            let maker_available_cross_collateral = get_available_usdc_collateral(
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
    tx_state: &TxState,
    input: &ApplyTradeParams,
    is_taker_ask: BoolTarget,
    taker_base_balance_delta: &BigIntTarget,
    taker_quote_balance_delta: &BigIntTarget,
    maker_base_balance_delta: &BigIntTarget,
    maker_quote_balance_delta: &BigIntTarget,
    cancel_taker_order: &mut BoolTarget,
    cancel_maker_order: &mut BoolTarget,
) -> (
    [BigUintTarget; 2], // Margined asset total supplied amounts
    [BigUintTarget; 2], // taker base and quote asset balance
    [BigIntTarget; 2],  // taker base and quote margined asset balance
    BigIntTarget,       // taker strategy
    [BigUintTarget; 2], // maker base and quote asset balance
    [BigIntTarget; 2],  // maker base and quote margined asset balance
    BigIntTarget,       // maker strategy
    RiskInfoTarget,     // new taker risk info
    RiskInfoTarget,     // new maker risk info
) {
    let _spot = builder.constant_u64(PRODUCT_TYPE_SPOT);
    let _perps = builder.constant_u64(PRODUCT_TYPE_PERPS);
    let is_spot = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_SPOT);
    let is_enabled = builder.and(*update_status_flags, is_spot);

    let is_taker_unified = tx_state.accounts[TAKER_ACCOUNT_ID].is_unified_mode();
    let is_maker_unified = tx_state.accounts[MAKER_ACCOUNT_ID].is_unified_mode();

    let is_base_asset_universal =
        is_universal_asset(builder, tx_state.asset_indices[BASE_ASSET_ID]);
    let is_quote_asset_universal =
        is_universal_asset(builder, tx_state.asset_indices[QUOTE_ASSET_ID]);

    let is_taker_bid = builder.not(is_taker_ask);

    let is_liquidation_order = builder.is_equal_constant(
        tx_state.register_stack[0].pending_type,
        LIQUIDATION_ORDER as u64,
    );
    let is_liquidation_order = builder.and(is_enabled, is_liquidation_order);
    let is_not_liquidation_order = builder.and_not(is_enabled, is_liquidation_order);

    let mut success;
    let mut valid_taker_ask;
    let mut valid_taker_bid;
    let mut valid_maker_ask;
    let mut valid_maker_bid;

    /************************************************************************************/
    /************************************************************************************/
    // Apply the sells first so that we can perform auto supply operations when buying
    // Select sold assets for maker and taker and apply them first.

    // Deltas
    let taker_ask_delta = builder.select_bigint(
        is_taker_ask,
        taker_base_balance_delta,
        taker_quote_balance_delta,
    );
    let taker_bid_delta = builder.select_bigint(
        is_taker_ask,
        taker_quote_balance_delta,
        taker_base_balance_delta,
    );
    let maker_ask_delta = builder.select_bigint(
        is_taker_ask,
        maker_quote_balance_delta,
        maker_base_balance_delta,
    );
    let maker_bid_delta = builder.select_bigint(
        is_taker_ask,
        maker_base_balance_delta,
        maker_quote_balance_delta,
    );
    // Asset indices
    let taker_ask_asset_index = builder.select(
        is_taker_ask,
        tx_state.market.base_asset_id,
        tx_state.market.quote_asset_id,
    );
    let taker_bid_asset_index = builder.select(
        is_taker_ask,
        tx_state.market.quote_asset_id,
        tx_state.market.base_asset_id,
    );
    let maker_ask_asset_index = taker_bid_asset_index;
    let maker_bid_asset_index = taker_ask_asset_index;
    // Account margined assets
    let mut taker_ask_maker_bid_margined_asset = MarginedAssetTarget::partial_select_for_spot_trade(
        builder,
        is_taker_ask,
        &tx_state.margined_asset[BASE_ASSET_ID],
        &tx_state.margined_asset[QUOTE_ASSET_ID],
    );
    let mut taker_bid_maker_ask_margined_asset = MarginedAssetTarget::partial_select_for_spot_trade(
        builder,
        is_taker_ask,
        &tx_state.margined_asset[QUOTE_ASSET_ID],
        &tx_state.margined_asset[BASE_ASSET_ID],
    );
    // Is asset used as margin
    let is_taker_ask_asset_used_as_margin = builder.select_bool(
        is_taker_ask,
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
    );
    let is_taker_bid_asset_used_as_margin = builder.select_bool(
        is_taker_ask,
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
    );
    let is_maker_ask_asset_used_as_margin = builder.select_bool(
        is_taker_ask,
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
    );
    let is_maker_bid_asset_used_as_margin = builder.select_bool(
        is_taker_ask,
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
        tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
    );
    // Spot balances
    let mut taker_ask_balance = builder.select_biguint(
        is_taker_ask,
        &tx_state.account_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
        &tx_state.account_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
    );
    let mut taker_bid_balance = builder.select_biguint(
        is_taker_ask,
        &tx_state.account_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
        &tx_state.account_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
    );
    let mut maker_ask_balance = builder.select_biguint(
        is_taker_ask,
        &tx_state.account_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
        &tx_state.account_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
    );
    let mut maker_bid_balance = builder.select_biguint(
        is_taker_ask,
        &tx_state.account_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
        &tx_state.account_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
    );
    // Margin balances
    let mut taker_ask_margined_balance = builder.select_bigint(
        is_taker_ask,
        &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
        &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
    );
    let mut taker_bid_margined_balance = builder.select_bigint(
        is_taker_ask,
        &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
        &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
    );
    let mut maker_ask_margined_balance = builder.select_bigint(
        is_taker_ask,
        &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
        &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
    );
    let mut maker_bid_margined_balance = builder.select_bigint(
        is_taker_ask,
        &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
        &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
    );

    /************************************************************************************/
    /************************************************************************************/
    // Apply ask and bid deltas

    let mut taker_strategy = tx_state.strategies[TAKER_ACCOUNT_ID].clone();
    let mut maker_strategy = tx_state.strategies[MAKER_ACCOUNT_ID].clone();
    {
        let is_maker_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[MAKER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_taker_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );

        // Taker - ask delta
        {
            // Apply raw delta for liquidation orders
            AccountTarget::apply_asset_delta_raw(
                builder,
                is_liquidation_order,
                _perps,
                taker_ask_asset_index,
                &mut taker_ask_maker_bid_margined_asset,
                &mut taker_ask_balance,
                &taker_ask_delta,
                &mut taker_ask_margined_balance,
                true,
            );
            // Apply delta for non-liquidation orders
            let is_taker_ask_spot_balance_non_negative = AccountTarget::apply_asset_delta(
                builder,
                is_not_liquidation_order,
                _spot,
                taker_ask_asset_index,
                &mut taker_ask_maker_bid_margined_asset,
                is_taker_ask_asset_used_as_margin,
                &taker_ask_delta,
                is_taker_unified,
                is_taker_insurance_fund,
                &mut taker_ask_balance,
                &mut taker_ask_margined_balance,
                &mut taker_strategy,
                true,
            );
            // Validity checks
            {
                valid_taker_ask = is_taker_ask_spot_balance_non_negative;
                (success, taker_ask_balance) =
                    builder.try_trim_biguint(&taker_ask_balance, BIG_U96_LIMBS);
                valid_taker_ask = builder.and(valid_taker_ask, success);
                (success, taker_ask_margined_balance.abs) =
                    builder.try_trim_biguint(&taker_ask_margined_balance.abs, BIG_U96_LIMBS);
                valid_taker_ask = builder.and(valid_taker_ask, success);
                let is_margin_balance_negative =
                    builder.is_sign_negative(taker_ask_margined_balance.sign);
                let is_asset_universal = is_universal_asset(builder, taker_ask_asset_index);
                let should_be_false =
                    builder.and_not(is_margin_balance_negative, is_asset_universal);
                valid_taker_ask = builder.and_not(valid_taker_ask, should_be_false);
            }
        }

        // Maker - ask delta
        {
            let is_maker_ask_spot_balance_non_negative = AccountTarget::apply_asset_delta(
                builder,
                is_enabled,
                _spot,
                maker_ask_asset_index,
                &mut taker_bid_maker_ask_margined_asset,
                is_maker_ask_asset_used_as_margin,
                &maker_ask_delta,
                is_maker_unified,
                is_maker_insurance_fund,
                &mut maker_ask_balance,
                &mut maker_ask_margined_balance,
                &mut maker_strategy,
                true,
            );
            // Validity checks
            {
                valid_maker_ask = is_maker_ask_spot_balance_non_negative;
                (success, maker_ask_balance) =
                    builder.try_trim_biguint(&maker_ask_balance, BIG_U96_LIMBS);
                valid_maker_ask = builder.and(valid_maker_ask, success);
                (success, maker_ask_margined_balance.abs) =
                    builder.try_trim_biguint(&maker_ask_margined_balance.abs, BIG_U96_LIMBS);
                valid_maker_ask = builder.and(valid_maker_ask, success);
                let is_margin_balance_negative =
                    builder.is_sign_negative(maker_ask_margined_balance.sign);
                let is_asset_universal = is_universal_asset(builder, maker_ask_asset_index);
                let should_be_false =
                    builder.and_not(is_margin_balance_negative, is_asset_universal);
                valid_maker_ask = builder.and_not(valid_maker_ask, should_be_false);
            }
        }

        // Taker - bid delta
        {
            // Apply raw delta for liquidation orders
            AccountTarget::apply_asset_delta_raw(
                builder,
                is_liquidation_order,
                _perps,
                taker_bid_asset_index,
                &mut taker_bid_maker_ask_margined_asset,
                &mut taker_bid_balance,
                &taker_bid_delta,
                &mut taker_bid_margined_balance,
                true,
            );
            // Apply delta for non-liquidation orders
            let is_taker_bid_spot_balance_non_negative = AccountTarget::apply_asset_delta(
                builder,
                is_not_liquidation_order,
                _spot,
                taker_bid_asset_index,
                &mut taker_bid_maker_ask_margined_asset,
                is_taker_bid_asset_used_as_margin,
                &taker_bid_delta,
                is_taker_unified,
                is_taker_insurance_fund,
                &mut taker_bid_balance,
                &mut taker_bid_margined_balance,
                &mut taker_strategy,
                true,
            );
            // Validity checks
            {
                valid_taker_bid = is_taker_bid_spot_balance_non_negative;
                (success, taker_bid_balance) =
                    builder.try_trim_biguint(&taker_bid_balance, BIG_U96_LIMBS);
                valid_taker_bid = builder.and(valid_taker_bid, success);
                (success, taker_bid_margined_balance.abs) =
                    builder.try_trim_biguint(&taker_bid_margined_balance.abs, BIG_U96_LIMBS);
                valid_taker_bid = builder.and(valid_taker_bid, success);
                let taker_is_margin_balance_negative =
                    builder.is_sign_negative(taker_bid_margined_balance.sign);
                let is_asset_universal = is_universal_asset(builder, taker_bid_asset_index);
                let should_be_false =
                    builder.and_not(taker_is_margin_balance_negative, is_asset_universal);
                valid_taker_bid = builder.and_not(valid_taker_bid, should_be_false);
            }
        }

        // Maker - bid delta
        {
            let is_maker_bid_spot_balance_non_negative = AccountTarget::apply_asset_delta(
                builder,
                is_enabled,
                _spot,
                maker_bid_asset_index,
                &mut taker_ask_maker_bid_margined_asset,
                is_maker_bid_asset_used_as_margin,
                &maker_bid_delta,
                is_maker_unified,
                is_maker_insurance_fund,
                &mut maker_bid_balance,
                &mut maker_bid_margined_balance,
                &mut maker_strategy,
                true,
            );
            // Validity checks
            valid_maker_bid = is_maker_bid_spot_balance_non_negative;
            (success, maker_bid_balance) =
                builder.try_trim_biguint(&maker_bid_balance, BIG_U96_LIMBS);
            valid_maker_bid = builder.and(valid_maker_bid, success);
            (success, maker_bid_margined_balance.abs) =
                builder.try_trim_biguint(&maker_bid_margined_balance.abs, BIG_U96_LIMBS);
            valid_maker_bid = builder.and(valid_maker_bid, success);
            let maker_is_margin_balance_negative =
                builder.is_sign_negative(maker_bid_margined_balance.sign);
            let is_asset_universal = is_universal_asset(builder, maker_bid_asset_index);
            let should_be_false =
                builder.and_not(maker_is_margin_balance_negative, is_asset_universal);
            valid_maker_bid = builder.and_not(valid_maker_bid, should_be_false);
        }
    }

    /************************************************************************************/
    /************************************************************************************/
    // Put mutated ask/bid parameters back as base/quote

    // Account margined assets - Only modified field is TSA
    let base_total_supplied_amount = builder.select_biguint(
        is_taker_ask,
        &taker_ask_maker_bid_margined_asset.total_supplied_amount,
        &taker_bid_maker_ask_margined_asset.total_supplied_amount,
    );
    let quote_total_supplied_amount = builder.select_biguint(
        is_taker_ask,
        &taker_bid_maker_ask_margined_asset.total_supplied_amount,
        &taker_ask_maker_bid_margined_asset.total_supplied_amount,
    );
    // Spot balances
    let taker_base_balance =
        builder.select_biguint(is_taker_ask, &taker_ask_balance, &taker_bid_balance);
    let taker_quote_balance =
        builder.select_biguint(is_taker_ask, &taker_bid_balance, &taker_ask_balance);
    let maker_base_balance =
        builder.select_biguint(is_taker_ask, &maker_bid_balance, &maker_ask_balance);
    let maker_quote_balance =
        builder.select_biguint(is_taker_ask, &maker_ask_balance, &maker_bid_balance);
    // Margined balances
    let taker_base_margin_balance = builder.select_bigint(
        is_taker_ask,
        &taker_ask_margined_balance,
        &taker_bid_margined_balance,
    );
    let taker_quote_margin_balance = builder.select_bigint(
        is_taker_ask,
        &taker_bid_margined_balance,
        &taker_ask_margined_balance,
    );
    let maker_base_margin_balance = builder.select_bigint(
        is_taker_ask,
        &maker_bid_margined_balance,
        &maker_ask_margined_balance,
    );
    let maker_quote_margin_balance = builder.select_bigint(
        is_taker_ask,
        &maker_ask_margined_balance,
        &maker_bid_margined_balance,
    );
    // Balance and margin balance validity parameters
    let mut valid_taker_base = builder.select_bool(is_taker_ask, valid_taker_ask, valid_taker_bid);
    let mut valid_taker_quote = builder.select_bool(is_taker_ask, valid_taker_bid, valid_taker_ask);
    let mut valid_maker_base = builder.select_bool(is_taker_ask, valid_maker_bid, valid_maker_ask);
    let mut valid_maker_quote = builder.select_bool(is_taker_ask, valid_maker_ask, valid_maker_bid);

    /************************************************************************************/
    /************************************************************************************/
    // Update risks if any asset is used as margin
    let new_taker_risk_info = {
        let mut new_taker_cross_risk_parameters =
            input.taker_risk_info.cross_risk_parameters.clone();
        let update_taker_risk_for_base = builder.and(
            is_enabled,
            tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][BASE_ASSET_ID],
        );
        new_taker_cross_risk_parameters.update_for_spot_trade(
            builder,
            update_taker_risk_for_base,
            tx_state.asset_indices[BASE_ASSET_ID],
            &tx_state.margined_asset[BASE_ASSET_ID],
            &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
            &taker_base_margin_balance,
        );
        let update_taker_risk_for_quote = builder.and(
            is_enabled,
            tx_state.is_asset_used_as_margin[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        );
        new_taker_cross_risk_parameters.update_for_spot_trade(
            builder,
            update_taker_risk_for_quote,
            tx_state.asset_indices[QUOTE_ASSET_ID],
            &tx_state.margined_asset[QUOTE_ASSET_ID],
            &tx_state.account_margined_assets[TAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
            &taker_quote_margin_balance,
        );
        RiskInfoTarget {
            cross_risk_parameters: new_taker_cross_risk_parameters.clone(),
            current_risk_parameters: new_taker_cross_risk_parameters,
        }
    };
    let new_maker_risk_info = {
        let mut new_maker_cross_risk_parameters =
            input.maker_risk_info.cross_risk_parameters.clone();
        let update_maker_risk_for_base = builder.and(
            is_enabled,
            tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][BASE_ASSET_ID],
        );
        new_maker_cross_risk_parameters.update_for_spot_trade(
            builder,
            update_maker_risk_for_base,
            tx_state.asset_indices[BASE_ASSET_ID],
            &tx_state.margined_asset[BASE_ASSET_ID],
            &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][BASE_ASSET_ID].balance,
            &maker_base_margin_balance,
        );
        let update_maker_risk_for_quote = builder.and(
            is_enabled,
            tx_state.is_asset_used_as_margin[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID],
        );
        new_maker_cross_risk_parameters.update_for_spot_trade(
            builder,
            update_maker_risk_for_quote,
            tx_state.asset_indices[QUOTE_ASSET_ID],
            &tx_state.margined_asset[QUOTE_ASSET_ID],
            &tx_state.account_margined_assets[MAKER_ACCOUNT_ID][QUOTE_ASSET_ID].balance,
            &maker_quote_margin_balance,
        );
        RiskInfoTarget {
            cross_risk_parameters: new_maker_cross_risk_parameters.clone(),
            current_risk_parameters: new_maker_cross_risk_parameters,
        }
    };

    /************************************************************************************/
    /************************************************************************************/
    // Perform validity checks and cancel taker/maker if necessary

    // Taker
    {
        let is_taker_collateral_negative = builder.is_sign_negative(
            new_taker_risk_info
                .cross_risk_parameters
                .usdc_collateral_with_funding
                .sign,
        );
        let is_taker_base_collateral_invalid = builder.multi_and(&[
            is_base_asset_universal,
            is_taker_ask,
            is_taker_collateral_negative,
        ]);
        valid_taker_base = builder.and_not(valid_taker_base, is_taker_base_collateral_invalid);
        let is_taker_quote_collateral_invalid = builder.multi_and(&[
            is_quote_asset_universal,
            is_taker_bid,
            is_taker_collateral_negative,
        ]);
        valid_taker_quote = builder.and_not(valid_taker_quote, is_taker_quote_collateral_invalid);

        let is_taker_valid_risk_change = tx_state.risk_infos[TAKER_ACCOUNT_ID]
            .current_risk_parameters
            .is_valid_risk_change(builder, &new_taker_risk_info.current_risk_parameters);
        let is_taker_invalid_risk_change =
            builder.and_not(is_taker_unified, is_taker_valid_risk_change);
        let is_taker_valid_risk_change_for_spot = builder.not(is_taker_invalid_risk_change);

        let valid_taker_balances = builder.multi_and(&[
            valid_taker_base,
            valid_taker_quote,
            is_taker_valid_risk_change_for_spot,
        ]);
        let invalid_taker_balances = builder.and_not(is_enabled, valid_taker_balances);
        *cancel_taker_order = builder.or(invalid_taker_balances, *cancel_taker_order);
        *update_status_flags = builder.and_not(*update_status_flags, *cancel_taker_order);
    }
    // Maker
    {
        let is_maker_collateral_negative = builder.is_sign_negative(
            new_maker_risk_info
                .cross_risk_parameters
                .usdc_collateral_with_funding
                .sign,
        );
        let is_maker_base_collateral_invalid = builder.multi_and(&[
            is_base_asset_universal,
            is_taker_bid,
            is_maker_collateral_negative,
        ]);
        valid_maker_base = builder.and_not(valid_maker_base, is_maker_base_collateral_invalid);
        let is_maker_quote_collateral_invalid = builder.multi_and(&[
            is_quote_asset_universal,
            is_taker_ask,
            is_maker_collateral_negative,
        ]);
        valid_maker_quote = builder.and_not(valid_maker_quote, is_maker_quote_collateral_invalid);

        let is_maker_valid_risk_change = tx_state.risk_infos[MAKER_ACCOUNT_ID]
            .current_risk_parameters
            .is_valid_risk_change(builder, &new_maker_risk_info.current_risk_parameters);
        let is_maker_invalid_risk_change =
            builder.and_not(is_maker_unified, is_maker_valid_risk_change);
        let is_maker_valid_risk_change_for_spot = builder.not(is_maker_invalid_risk_change);

        let valid_maker_balances = builder.multi_and(&[
            valid_maker_base,
            valid_maker_quote,
            is_maker_valid_risk_change_for_spot,
        ]);
        let invalid_maker_balances = builder.and_not(is_enabled, valid_maker_balances);
        *cancel_maker_order = builder.or(invalid_maker_balances, *cancel_maker_order);
        *update_status_flags = builder.and_not(*update_status_flags, *cancel_maker_order);
    }

    (
        [base_total_supplied_amount, quote_total_supplied_amount],
        [taker_base_balance, taker_quote_balance],
        [taker_base_margin_balance, taker_quote_margin_balance],
        taker_strategy,
        [maker_base_balance, maker_quote_balance],
        [maker_base_margin_balance, maker_quote_margin_balance],
        maker_strategy,
        new_taker_risk_info,
        new_maker_risk_info,
    )
}

fn apply_self_trade_reduce(
    builder: &mut Builder,
    flag: BoolTarget,
    is_post_only: BoolTarget,
    is_spot: BoolTarget,
    is_maker_limit_order: BoolTarget,
    optimistic_trade_amount: Target,
    update_status_flags: &mut BoolTarget,
    cancel_taker_order: &mut BoolTarget,
    cancel_maker_order: &mut BoolTarget,
    tx_state: &mut TxState,
) {
    let _false = builder._false();

    // Handle post-only taker case
    {
        let post_only_flag = builder.multi_and(&[*update_status_flags, flag, is_post_only]);
        *cancel_taker_order =
            builder.select_bool(post_only_flag, *update_status_flags, *cancel_taker_order);
        *update_status_flags = builder.select_bool(post_only_flag, _false, *update_status_flags);
    }

    let not_post_only_flag = builder.and(*update_status_flags, flag);

    let new_register_pending_size = builder.sub(
        tx_state.register_stack[0].pending_size,
        optimistic_trade_amount,
    );
    let new_order_remaining_size = builder.sub(
        tx_state.account_order.remaining_base_amount,
        optimistic_trade_amount,
    );
    tx_state.register_stack[0].pending_size = builder.select(
        not_post_only_flag,
        new_register_pending_size,
        tx_state.register_stack[0].pending_size,
    );
    tx_state.account_order.remaining_base_amount = builder.select(
        not_post_only_flag,
        new_order_remaining_size,
        tx_state.account_order.remaining_base_amount,
    );

    let decrement_locked_balance_flag =
        builder.multi_and(&[not_post_only_flag, is_spot, is_maker_limit_order]);
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
        not_post_only_flag,
        tx_state.account_order.is_ask,
        new_order_remaining_size,
    );

    // Taker filled
    {
        let is_register_pending_size_empty =
            builder.is_zero(tx_state.register_stack[0].pending_size);
        let self_trade_and_register_pending_size_empty =
            builder.and(not_post_only_flag, is_register_pending_size_empty);
        *cancel_taker_order = builder.select_bool(
            self_trade_and_register_pending_size_empty,
            *update_status_flags,
            *cancel_taker_order,
        );
    }

    // Maker filled
    {
        let is_order_remaining_size_empty =
            builder.is_zero(tx_state.account_order.remaining_base_amount);
        let self_trade_and_order_remaining_size_empty =
            builder.and(not_post_only_flag, is_order_remaining_size_empty);
        *cancel_maker_order = builder.select_bool(
            self_trade_and_order_remaining_size_empty,
            *update_status_flags,
            *cancel_maker_order,
        );
    }

    *update_status_flags = builder.select_bool(not_post_only_flag, _false, *update_status_flags);
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

fn get_account_order_from_register(
    builder: &mut Builder,
    register: &BaseRegisterInfoTarget,
) -> AccountOrderTarget {
    let (integrator_fee_collector_index, integrator_taker_fee, integrator_maker_fee, order_flags) =
        register.to_order_fields_from_generic_fields(builder);

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

        integrator_fee_collector_index,
        integrator_taker_fee,
        integrator_maker_fee,
        order_flags,
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

fn cancel_position_tied_account_orders(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    market_index: Target,
    owner_account_index: Target,
    position_tied_order_count: Target,
    register_index: usize,
) {
    let cancel_position_tied_account_orders =
        builder.constant_from_u8(CANCEL_POSITION_TIED_ACCOUNT_ORDERS);
    let cancel_position_tied_account_orders_instruction = &BaseRegisterInfoTarget {
        instruction_type: cancel_position_tied_account_orders,
        market_index,
        account_index: owner_account_index,
        pending_size: position_tied_order_count,

        ..BaseRegisterInfoTarget::empty(builder)
    };
    tx_state.put_to_instruction_stack_unsafe(
        builder,
        is_enabled,
        cancel_position_tied_account_orders_instruction,
        register_index,
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
    max_register_index: usize,
) {
    assert!(
        max_register_index >= 1,
        "max_register_index must be at least 1"
    );

    let trigger_child_order_0_instruction = get_trigger_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_0,
        pending_size,
    );
    let does_child_order_0_exist = builder.is_not_zero(child_order_index_0);
    let child_order_0_flag = builder.and(is_enabled, does_child_order_0_exist);
    tx_state.put_to_instruction_stack_unsafe(
        builder,
        child_order_0_flag,
        &trigger_child_order_0_instruction,
        max_register_index,
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
    tx_state.put_to_instruction_stack_unsafe(
        builder,
        child_order_1_flag,
        &trigger_child_order_1_instruction,
        max_register_index - 1,
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

        ..BaseRegisterInfoTarget::empty(builder)
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
    max_register_index: usize,
) {
    let cancel_child_order_0_instruction = get_cancel_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_0,
    );
    let does_child_order_0_exist = builder.is_not_zero(child_order_index_0);
    let child_order_0_flag = builder.and(is_enabled, does_child_order_0_exist);
    tx_state.put_to_instruction_stack_unsafe(
        builder,
        child_order_0_flag,
        &cancel_child_order_0_instruction,
        max_register_index,
    );

    let cancel_child_order_1_instruction = get_cancel_child_order_instruction(
        builder,
        market_index,
        owner_account_index,
        child_order_index_1,
    );
    let does_child_order_1_exist = builder.is_not_zero(child_order_index_1);
    let child_order_1_flag = builder.and(is_enabled, does_child_order_1_exist);
    tx_state.put_to_instruction_stack_unsafe(
        builder,
        child_order_1_flag,
        &cancel_child_order_1_instruction,
        max_register_index - 1,
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
