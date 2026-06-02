// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::matching_engine::{get_next_order_nonce, is_not_valid_reduce_only_direction};
use crate::poseidon2::Poseidon2Hash;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::account_order_type::AccountOrderTypes;
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::order::get_order_index;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TxOrder {
    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "oi")]
    pub client_order_index: i64, // 48 bits (user-assigned or 0)

    #[serde(rename = "t")]
    pub order_type: u8,

    #[serde(rename = "tf")]
    pub time_in_force: u8,

    #[serde(rename = "oe")]
    pub order_expiry: i64, // 48 bits

    #[serde(rename = "ba")]
    pub base_amount: i64, // 48 bits

    #[serde(rename = "p")]
    pub price: i64, // 32 bits

    #[serde(rename = "ia")]
    pub is_ask: u8,

    #[serde(rename = "r", default)]
    pub reduce_only: u8,

    #[serde(rename = "tp", default)]
    pub trigger_price: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CreateGroupedOrdersTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "gt")]
    pub grouping_type: u8,

    #[serde(rename = "o")]
    pub orders: [TxOrder; MAX_NB_GROUPED_ORDERS],
}

#[derive(Debug, Clone)]
pub struct TxOrderTarget {
    pub market_index: Target,       // 8 bits
    pub client_order_index: Target, // 48 bits
    pub base_amount: Target,        // 48 bits
    pub price: Target,              // 32 bits
    pub is_ask: BoolTarget,
    pub order_type: Target,
    pub time_in_force: Target,
    pub reduce_only: Target,
    pub trigger_price: Target, // 32 bits
    pub order_expiry: Target,  // 48 bits
}

#[derive(Debug)]
pub struct L2CreateGroupedOrdersTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub grouping_type: Target,
    pub orders: [TxOrderTarget; MAX_NB_GROUPED_ORDERS],

    // helpers
    pub market_index: Target,
    pub order_count: Target,
    pub base_amounts: [Target; MAX_NB_GROUPED_ORDERS],
    pub order_exists: [BoolTarget; MAX_NB_GROUPED_ORDERS],
    pub is_oco: BoolTarget,
    pub is_oto: BoolTarget,
    pub is_otoco: BoolTarget,

    // output
    pub success: BoolTarget,
}

impl L2CreateGroupedOrdersTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        let orders: [TxOrderTarget; 3] = array::from_fn(|_| TxOrderTarget {
            market_index: builder.add_virtual_target(),
            client_order_index: builder.add_virtual_target(),
            base_amount: builder.add_virtual_target(),
            price: builder.add_virtual_target(),
            is_ask: builder.add_virtual_bool_target_safe(),
            order_type: builder.add_virtual_target(),
            time_in_force: builder.add_virtual_target(),
            reduce_only: builder.add_virtual_target(),
            trigger_price: builder.add_virtual_target(),
            order_expiry: builder.add_virtual_target(),
        });
        L2CreateGroupedOrdersTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            grouping_type: builder.add_virtual_target(),
            orders,

            // helpers
            market_index: Target::default(),
            order_count: Target::default(),
            base_amounts: [Target::default(), Target::default(), Target::default()],
            order_exists: [
                BoolTarget::default(),
                BoolTarget::default(),
                BoolTarget::default(),
            ],
            is_oco: BoolTarget::default(),
            is_oto: BoolTarget::default(),
            is_otoco: BoolTarget::default(),

            // output
            success: BoolTarget::default(),
        }
    }

    fn verify_oco(
        &mut self,
        builder: &mut Builder,
        oco_flag: BoolTarget,
        order_types: &[AccountOrderTypes; MAX_NB_GROUPED_ORDERS],
    ) {
        // Both orders should have the same size
        builder.conditional_assert_eq(
            oco_flag,
            self.orders[0].base_amount,
            self.orders[1].base_amount,
        );

        // Orders should be in the same direction
        builder.conditional_assert_eq(
            oco_flag,
            self.orders[0].is_ask.target,
            self.orders[1].is_ask.target,
        );

        // Both orders should be reduce only
        builder.conditional_assert_eq(
            oco_flag,
            self.orders[0].reduce_only,
            self.orders[1].reduce_only,
        );

        // Both orders should have the same order expiry
        builder.conditional_assert_eq(
            oco_flag,
            self.orders[0].order_expiry,
            self.orders[1].order_expiry,
        );
        // Both orders should be conditional orders
        builder.conditional_assert_true(oco_flag, order_types[0].is_conditional_order);
        builder.conditional_assert_true(oco_flag, order_types[1].is_conditional_order);
        // Exactly one of the orders should be stop loss variant
        builder.conditional_assert_not_eq(
            oco_flag,
            order_types[0].is_stop_loss_variant.target,
            order_types[1].is_stop_loss_variant.target,
        );
    }

    fn verify_oto(
        &mut self,
        builder: &mut Builder,
        oto_flag: BoolTarget,
        order_types: &[AccountOrderTypes; MAX_NB_GROUPED_ORDERS],
    ) {
        let one = builder.one();

        // Child order should have nil size
        builder.conditional_assert_zero(oto_flag, self.orders[1].base_amount);

        // Orders should be in the opposite direction
        builder.conditional_assert_not_eq(
            oto_flag,
            self.orders[0].is_ask.target,
            self.orders[1].is_ask.target,
        );

        // Second order should be reduce only
        builder.conditional_assert_eq(oto_flag, self.orders[1].reduce_only, one);

        // Both orders should have the same order expiry if first one is not nil
        let is_parent_nil_expiry = builder.is_zero(self.orders[0].order_expiry);
        let are_expiries_equal =
            builder.is_equal(self.orders[0].order_expiry, self.orders[1].order_expiry);
        let valid_expiry = builder.or(is_parent_nil_expiry, are_expiries_equal);
        builder.conditional_assert_true(oto_flag, valid_expiry);

        // Parent order should be limit or market order
        let is_limit_or_market_order = builder.or(
            order_types[0].is_limit_order,
            order_types[0].is_market_order,
        );
        builder.conditional_assert_true(oto_flag, is_limit_or_market_order);

        // Child order should be conditional order
        builder.conditional_assert_true(oto_flag, order_types[1].is_conditional_order);
    }

    fn verify_otoco(
        &mut self,
        builder: &mut Builder,
        otoco_flag: BoolTarget,
        order_types: &[AccountOrderTypes; MAX_NB_GROUPED_ORDERS],
    ) {
        // Child orders should have nil size
        builder.conditional_assert_zero(otoco_flag, self.orders[1].base_amount);
        builder.conditional_assert_zero(otoco_flag, self.orders[2].base_amount);

        // Child orders should be in the opposite direction of parent
        builder.conditional_assert_not_eq(
            otoco_flag,
            self.orders[0].is_ask.target,
            self.orders[1].is_ask.target,
        );
        builder.conditional_assert_not_eq(
            otoco_flag,
            self.orders[0].is_ask.target,
            self.orders[2].is_ask.target,
        );

        // Both child orders should have the same order expiry
        builder.conditional_assert_eq(
            otoco_flag,
            self.orders[1].order_expiry,
            self.orders[2].order_expiry,
        );
        // Child and parent should have the same order expiry if parent is not nil
        let is_parent_nil_expiry = builder.is_zero(self.orders[0].order_expiry);
        let are_expiries_equal =
            builder.is_equal(self.orders[0].order_expiry, self.orders[1].order_expiry);
        let valid_expiry = builder.or(is_parent_nil_expiry, are_expiries_equal);
        builder.conditional_assert_true(otoco_flag, valid_expiry);

        // Parent order should be limit or market order
        let is_limit_or_market_order = builder.or(
            order_types[0].is_limit_order,
            order_types[0].is_market_order,
        );
        builder.conditional_assert_true(otoco_flag, is_limit_or_market_order);

        // Child orders should be conditional orders
        builder.conditional_assert_true(otoco_flag, order_types[1].is_conditional_order);
        builder.conditional_assert_true(otoco_flag, order_types[2].is_conditional_order);
        // Exactly one of the child orders should be stop loss variant
        builder.conditional_assert_not_eq(
            otoco_flag,
            order_types[1].is_stop_loss_variant.target,
            order_types[2].is_stop_loss_variant.target,
        );
    }
}

impl TxHash for L2CreateGroupedOrdersTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let two = builder.constant_from_u8(2);
        let three = builder.constant_from_u8(3);
        let otoco = builder.constant_from_u8(GROUPING_TYPE_ONE_TRIGGERS_A_ONE_CANCELS_THE_OTHER);
        let is_otoco = builder.is_equal(self.grouping_type, otoco);
        let num_orders = builder.select(is_otoco, three, two);

        let mut aggregated_order_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            self.orders[0].market_index,
            self.orders[0].client_order_index,
            self.orders[0].base_amount,
            self.orders[0].price,
            self.orders[0].is_ask.target,
            self.orders[0].order_type,
            self.orders[0].time_in_force,
            self.orders[0].reduce_only,
            self.orders[0].trigger_price,
            self.orders[0].order_expiry,
        ]);
        let mut flag = builder._true();
        for i in 1..MAX_NB_GROUPED_ORDERS {
            let _i = builder.constant_from_u8(i as u8);
            let is_equal = builder.is_equal(_i, num_orders);
            flag = builder.and_not(flag, is_equal);
            let order_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
                self.orders[i].market_index,
                self.orders[i].client_order_index,
                self.orders[i].base_amount,
                self.orders[i].price,
                self.orders[i].is_ask.target,
                self.orders[i].order_type,
                self.orders[i].time_in_force,
                self.orders[i].reduce_only,
                self.orders[i].trigger_price,
                self.orders[i].order_expiry,
            ]);
            let new_aggregated_order_hash =
                builder.hash_two_to_one(&aggregated_order_hash, &order_hash);
            aggregated_order_hash =
                builder.select_hash(flag, &new_aggregated_order_hash, &aggregated_order_hash);
        }

        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_GROUPED_ORDERS)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.grouping_type,
            aggregated_order_hash.elements[0],
            aggregated_order_hash.elements[1],
            aggregated_order_hash.elements[2],
            aggregated_order_hash.elements[3],
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CreateGroupedOrdersTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let one = builder.one();
        let two = builder.two();

        let ioc = builder.constant(F::from_canonical_u8(IOC));
        let gtt = builder.constant(F::from_canonical_u8(GTT));
        let post_only = builder.constant(F::from_canonical_u8(POST_ONLY));
        let nil_trigger_price = builder.constant(F::from_canonical_i64(NIL_ORDER_TRIGGER_PRICE));
        let nil_order_expiry = builder.constant(F::from_canonical_i64(NIL_ORDER_EXPIRY));

        let is_enabled = tx_type.is_l2_create_grouped_orders;
        self.success = is_enabled;

        self.market_index = self.orders[0].market_index;
        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );

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

        self.is_oto = builder.is_equal_constant(
            self.grouping_type,
            GROUPING_TYPE_ONE_TRIGGERS_THE_OTHER as u64,
        );
        let oto_flag = builder.and(is_enabled, self.is_oto);

        self.is_oco = builder.is_equal_constant(
            self.grouping_type,
            GROUPING_TYPE_ONE_CANCELS_THE_OTHER as u64,
        );
        let oco_flag = builder.and(is_enabled, self.is_oco);

        self.is_otoco = builder.is_equal_constant(
            self.grouping_type,
            GROUPING_TYPE_ONE_TRIGGERS_A_ONE_CANCELS_THE_OTHER as u64,
        );
        let otoco_flag = builder.and(is_enabled, self.is_otoco);

        let is_grouping_type_valid = builder.multi_or(&[self.is_oco, self.is_oto, self.is_otoco]);
        builder.conditional_assert_true(is_enabled, is_grouping_type_valid);

        self.order_count = builder.add(self.is_otoco.target, two);

        // Make sure given cloids are either nil or unique
        {
            let nil_cloid = builder.constant_i64(NIL_CLIENT_ORDER_INDEX);
            let _0_is_nil = builder.is_equal(self.orders[0].client_order_index, nil_cloid);
            let _1_is_nil = builder.is_equal(self.orders[1].client_order_index, nil_cloid);
            let _2_is_nil = builder.is_equal(self.orders[2].client_order_index, nil_cloid);
            let _0_eq_1 = builder.is_equal(
                self.orders[0].client_order_index,
                self.orders[1].client_order_index,
            );
            let _0_eq_2 = builder.is_equal(
                self.orders[0].client_order_index,
                self.orders[2].client_order_index,
            );
            let _1_eq_2 = builder.is_equal(
                self.orders[1].client_order_index,
                self.orders[2].client_order_index,
            );
            let _0_eq_any = builder.or(_0_eq_1, _0_eq_2);
            let _1_eq_any = builder.or(_1_eq_2, _0_eq_1);
            let _2_eq_any = builder.or(_0_eq_2, _1_eq_2);
            let should_all_be_false = [
                builder.and_not(_0_eq_any, _0_is_nil),
                builder.and_not(_1_eq_any, _1_is_nil),
                builder.and_not(_2_eq_any, _2_is_nil),
            ];
            let should_be_false = builder.multi_or(&should_all_be_false);
            builder.conditional_assert_false(is_enabled, should_be_false);
        }

        // First order should always fit into 48 bits, rest are equal to first order or zero
        builder.register_range_check(self.orders[0].base_amount, ORDER_SIZE_BITS);

        let order_types: [AccountOrderTypes; MAX_NB_GROUPED_ORDERS] =
            array::from_fn(|i| AccountOrderTypes::new(builder, self.orders[i].order_type));

        // Verify field ranges for each order
        let mut order_exists = builder._true();
        for i in 0..MAX_NB_GROUPED_ORDERS {
            let _i = builder.constant_from_u8(i as u8);
            let is_i_equal_to_order_count = builder.is_equal(_i, self.order_count);
            order_exists = builder.and_not(order_exists, is_i_equal_to_order_count);
            self.order_exists[i] = order_exists;
            let flag = builder.and(is_enabled, self.order_exists[i]);

            builder.conditional_assert_eq(flag, self.market_index, self.orders[i].market_index);

            // Assert reduce only is 0 or 1
            builder.assert_bool(BoolTarget::new_unsafe(self.orders[i].reduce_only));

            // ReduceOnly - If base amount is 0, reduce only must be 1
            let is_base_amount_zero = builder.is_zero(self.orders[i].base_amount);
            let flag = builder.and(flag, is_base_amount_zero);
            builder.conditional_assert_eq(flag, self.orders[i].reduce_only, one);

            // Price - (Must fit in 32 bits) & (shouldn't be zero if enabled)
            builder.conditional_assert_not_zero(flag, self.orders[i].price);
            builder.register_range_check(self.orders[i].price, ORDER_PRICE_BITS);

            // TimeInForce - Either IOC (0), GTT (1) or POST_ONLY (2)
            let is_ioc = builder.is_equal(self.orders[i].time_in_force, ioc);
            let is_gtt = builder.is_equal(self.orders[i].time_in_force, gtt);
            let is_post_only = builder.is_equal(self.orders[i].time_in_force, post_only);
            let is_time_in_force_valid = builder.multi_or(&[is_ioc, is_gtt, is_post_only]);
            builder.assert_true(is_time_in_force_valid);

            builder.assert_bool(self.orders[i].is_ask);
            builder.register_range_check(self.orders[i].order_expiry, TIMESTAMP_BITS);
            // order should not be already expired when created
            let is_order_expiry_nil =
                builder.is_equal(self.orders[i].order_expiry, nil_order_expiry);
            let is_order_expiry_in_future = builder.is_lt(
                tx_state.block_timestamp,
                self.orders[i].order_expiry,
                TIMESTAMP_BITS,
            );
            let is_order_expiry_valid = builder.or(is_order_expiry_nil, is_order_expiry_in_future);
            self.success = builder.and(self.success, is_order_expiry_valid);

            builder.register_range_check(self.orders[i].trigger_price, TRIGGER_PRICE_BITS);

            if i == 0 {
                /*****************************/
                /*  Limit Order Validations  */
                /*****************************/
                let is_enabled_and_limit_order = builder.and(flag, order_types[i].is_limit_order);
                // TriggerPrice must be nil for Limit Order
                builder.conditional_assert_eq(
                    is_enabled_and_limit_order,
                    self.orders[i].trigger_price,
                    nil_trigger_price,
                );
                // If Limit Order TimeInForce is IoC, then OrderExpiry must be nil
                let is_enabled_and_limit_order_and_ioc =
                    builder.and(is_enabled_and_limit_order, is_ioc);
                builder.conditional_assert_eq(
                    is_enabled_and_limit_order_and_ioc,
                    self.orders[i].order_expiry,
                    nil_order_expiry,
                );
                // If TimeInForce is GTT or PostOnly, then OrderExpiry must not be nil
                let is_enabled_and_limit_order_and_not_ioc =
                    builder.and_not(is_enabled_and_limit_order, is_ioc);
                builder.conditional_assert_not_eq(
                    is_enabled_and_limit_order_and_not_ioc,
                    self.orders[i].order_expiry,
                    nil_order_expiry,
                );

                /******************************/
                /*  Market Order Validations  */
                /******************************/
                // Market Order has to be IOC, no need for conditional since default is limit order
                builder.conditional_assert_true(
                    order_types[i].is_market_order, // We can omit the is_enabled check here because MARKET_ORDER=1 is not the default value
                    is_ioc,
                );
                // Market order expiry has to be nil(zero)
                builder.conditional_assert_eq(
                    order_types[i].is_market_order, // We can omit the is_enabled check here because MARKET_ORDER=1 is not the default value
                    self.orders[i].order_expiry,
                    nil_order_expiry,
                );
                // Market order trigger price has to be nil(zero)
                builder.conditional_assert_eq(
                    order_types[i].is_market_order, // We can omit the is_enabled check here because MARKET_ORDER=1 is not the default value
                    self.orders[i].trigger_price,
                    nil_trigger_price,
                );
            }
            /***********************************************/
            /*  StopLoss and TakeProfit Order Validations  */
            /***********************************************/
            let is_stop_loss_or_take_profit_order = builder.or(
                order_types[i].is_stop_loss_order,
                order_types[i].is_take_profit_order,
            );
            let is_enabled_and_stop_loss_or_take_profit_order =
                builder.and(flag, is_stop_loss_or_take_profit_order);
            // TimeInForce must be IOC for StopLoss and TakeProfit Market Orders
            builder.conditional_assert_eq(
                is_enabled_and_stop_loss_or_take_profit_order,
                self.orders[i].time_in_force,
                ioc,
            );
            // Trigger Price must not be nil for StopLoss and TakeProfit Market Orders
            builder.conditional_assert_not_eq(
                is_enabled_and_stop_loss_or_take_profit_order,
                self.orders[i].trigger_price,
                nil_trigger_price,
            );
            // OrderExpiry must not be nil for StopLoss and TakeProfit Market Orders
            builder.conditional_assert_not_eq(
                is_enabled_and_stop_loss_or_take_profit_order,
                self.orders[i].order_expiry,
                nil_order_expiry,
            );

            /*********************************************************/
            /*  StopLossLimit and TakeProfitLimit Order Validations  */
            /*********************************************************/
            let is_stop_loss_limit_or_take_profit_limit_order = builder.or(
                order_types[i].is_stop_loss_limit_order,
                order_types[i].is_take_profit_limit_order,
            );
            let is_enabled_and_stop_loss_limit_or_take_profit_limit_order =
                builder.and(flag, is_stop_loss_limit_or_take_profit_limit_order);
            // Trigger price must not be nil
            builder.conditional_assert_not_eq(
                is_enabled_and_stop_loss_limit_or_take_profit_limit_order,
                self.orders[i].trigger_price,
                nil_trigger_price,
            );
            // OrderExpiry must not be nil for StopLoss and TakeProfit Market Orders
            builder.conditional_assert_not_eq(
                is_enabled_and_stop_loss_limit_or_take_profit_limit_order,
                self.orders[i].order_expiry,
                nil_order_expiry,
            );
        }

        self.verify_oco(builder, oco_flag, &order_types);
        self.verify_oto(builder, oto_flag, &order_types);
        self.verify_otoco(builder, otoco_flag, &order_types);

        let ob_active_status = builder.constant(F::from_canonical_u8(MARKET_STATUS_ACTIVE));
        builder.conditional_assert_eq(is_enabled, tx_state.market_details.status, ob_active_status);
        builder.conditional_assert_not_zero(is_enabled, tx_state.market_details.index_price);
        builder.conditional_assert_not_zero(is_enabled, tx_state.market_details.mark_price);

        let perps_market_type = builder.constant_u64(MARKET_TYPE_PERPS);
        builder.conditional_assert_eq(is_enabled, tx_state.market.market_type, perps_market_type);

        let remaining_order_count =
            builder.sub(tx_state.market.bid_nonce, tx_state.market.ask_nonce);
        builder.conditional_assert_lte(
            is_enabled,
            self.order_count,
            remaining_order_count,
            ORDER_NONCE_BITS,
        );

        for i in 0..MAX_NB_GROUPED_ORDERS {
            self.base_amounts[i] = self.orders[i].base_amount;
        }
        let is_oto_or_otoco = builder.or(self.is_oto, self.is_otoco);
        let is_order_0_nil = builder.is_zero(self.base_amounts[0]);
        let position_tied_parent_order = builder.and(is_oto_or_otoco, is_order_0_nil);
        let position_tied_order_base_amount = tx_state.positions[OWNER_ACCOUNT_ID]
            .calculate_position_tied_order_base_amount(
                builder,
                tx_state.market_details.quote_multiplier,
                self.orders[0].price,
                tx_state.market.order_quote_limit,
            );
        self.base_amounts[0] = builder.select(
            position_tied_parent_order,
            position_tied_order_base_amount,
            self.base_amounts[0],
        );
        for i in 0..MAX_NB_GROUPED_ORDERS {
            let is_ioc = builder.is_equal(self.orders[i].time_in_force, ioc);
            let base_amount_to_check =
                builder.select(is_oto_or_otoco, self.base_amounts[0], self.base_amounts[i]);
            let is_base_amount_to_check_zero = builder.is_zero(base_amount_to_check);
            let is_valid_base_size_and_price = tx_state.is_valid_base_size_and_price(
                builder,
                base_amount_to_check,
                self.orders[i].price,
                order_types[i].is_twap_order,
                is_ioc,
            );
            let is_order_empty = builder.not(self.order_exists[i]);
            let is_valid_base_size_and_price_or_zero = builder.multi_or(&[
                is_valid_base_size_and_price,
                is_base_amount_to_check_zero,
                is_order_empty,
            ]);
            self.success = builder.and(self.success, is_valid_base_size_and_price_or_zero);

            self.success = builder.and(self.success, tx_state.is_cloid_unique[i]);
        }

        let invalid_reduce_only_direction = is_not_valid_reduce_only_direction(
            builder,
            tx_state.positions[OWNER_ACCOUNT_ID].position.sign,
            self.orders[0].is_ask,
        );
        let invalid_reduce_only_direction_check = builder.and(
            BoolTarget::new_unsafe(self.orders[0].reduce_only),
            invalid_reduce_only_direction,
        );
        self.success = builder.and_not(self.success, invalid_reduce_only_direction_check);
    }
}

impl Apply for L2CreateGroupedOrdersTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let nil_nonce = builder.constant_i64(NIL_ORDER_NONCE_INDEX);

        let mut order_instructions: [BaseRegisterInfoTarget; MAX_NB_GROUPED_ORDERS] =
            array::from_fn(|_| BaseRegisterInfoTarget::default());

        let trigger_status_na = builder.constant_from_u8(TRIGGER_STATUS_NA);
        let trigger_status_mark_price = builder.constant_from_u8(TRIGGER_STATUS_MARK_PRICE);
        let trigger_status_parent_order = builder.constant_from_u8(TRIGGER_STATUS_PARENT_ORDER);

        let (generic_field_1, generic_field_2, generic_field_3) =
            tx_state.attributes.get_register_generic_fields(builder);

        let execute_transaction = builder.constant_from_u8(EXECUTE_TRANSACTION);
        let insert_order = builder.constant_from_u8(INSERT_ORDER);
        let mut instruction_flag = self.success;
        let one = builder.one();
        for i in 0..MAX_NB_GROUPED_ORDERS {
            instruction_flag = builder.and(instruction_flag, self.order_exists[i]);

            // Update market nonce if order is to be inserted
            let order_nonce =
                get_next_order_nonce(builder, &tx_state.market, self.orders[i].is_ask);
            let order_index = get_order_index(builder, tx_state.market.market_index, order_nonce);
            let ask_nonce_plus_one = builder.add(tx_state.market.ask_nonce, one);
            let bid_nonce_minus_one = builder.sub(tx_state.market.bid_nonce, one);
            let new_ask_nonce = builder.select(
                self.orders[i].is_ask,
                ask_nonce_plus_one,
                tx_state.market.ask_nonce,
            );
            let new_bid_nonce = builder.select(
                self.orders[i].is_ask,
                tx_state.market.bid_nonce,
                bid_nonce_minus_one,
            );
            tx_state.market.ask_nonce =
                builder.select(instruction_flag, new_ask_nonce, tx_state.market.ask_nonce);
            tx_state.market.bid_nonce =
                builder.select(instruction_flag, new_bid_nonce, tx_state.market.bid_nonce);

            order_instructions[i] = BaseRegisterInfoTarget {
                instruction_type: builder.select(
                    instruction_flag,
                    insert_order,
                    execute_transaction,
                ),
                market_index: self.orders[i].market_index,
                account_index: self.account_index,

                pending_size: self.base_amounts[i],

                pending_order_index: order_index,
                pending_client_order_index: self.orders[i].client_order_index,
                pending_initial_size: self.base_amounts[i],
                pending_price: self.orders[i].price,
                pending_nonce: order_nonce,
                pending_is_ask: self.orders[i].is_ask,

                pending_type: self.orders[i].order_type,
                pending_time_in_force: self.orders[i].time_in_force,
                pending_reduce_only: self.orders[i].reduce_only,
                pending_expiry: self.orders[i].order_expiry,

                generic_field_0: builder.zero(),

                pending_trigger_price: self.orders[i].trigger_price,
                pending_trigger_status: trigger_status_na,
                pending_to_trigger_order_index0: builder.zero(),
                pending_to_trigger_order_index1: builder.zero(),
                pending_to_cancel_order_index0: builder.zero(),

                generic_field_1,
                generic_field_2,
                generic_field_3,
            };
        }

        // Handle OCO case
        {
            order_instructions[0].pending_trigger_status = builder.select(
                self.is_oco,
                trigger_status_mark_price,
                order_instructions[0].pending_trigger_status,
            );
            order_instructions[0].pending_nonce =
                builder.select(self.is_oco, nil_nonce, order_instructions[0].pending_nonce);
            order_instructions[0].pending_to_cancel_order_index0 = builder.select(
                self.is_oco,
                order_instructions[1].pending_order_index,
                order_instructions[0].pending_to_cancel_order_index0,
            );

            order_instructions[1].pending_trigger_status = builder.select(
                self.is_oco,
                trigger_status_mark_price,
                order_instructions[1].pending_trigger_status,
            );
            order_instructions[1].pending_nonce =
                builder.select(self.is_oco, nil_nonce, order_instructions[1].pending_nonce);
            order_instructions[1].pending_to_cancel_order_index0 = builder.select(
                self.is_oco,
                order_instructions[0].pending_order_index,
                order_instructions[1].pending_to_cancel_order_index0,
            );
        }

        // Handle OTO case
        {
            order_instructions[0].pending_trigger_status = builder.select(
                self.is_oto,
                trigger_status_na,
                order_instructions[0].pending_trigger_status,
            );
            order_instructions[0].pending_to_trigger_order_index0 = builder.select(
                self.is_oto,
                order_instructions[1].pending_order_index,
                order_instructions[0].pending_to_trigger_order_index0,
            );
            order_instructions[1].pending_trigger_status = builder.select(
                self.is_oto,
                trigger_status_parent_order,
                order_instructions[1].pending_trigger_status,
            );
            order_instructions[1].pending_nonce =
                builder.select(self.is_oto, nil_nonce, order_instructions[1].pending_nonce);
        }

        // Handle OTOCO case
        {
            order_instructions[0].pending_trigger_status = builder.select(
                self.is_otoco,
                trigger_status_na,
                order_instructions[0].pending_trigger_status,
            );
            order_instructions[0].pending_to_trigger_order_index0 = builder.select(
                self.is_otoco,
                order_instructions[1].pending_order_index,
                order_instructions[0].pending_to_trigger_order_index0,
            );
            order_instructions[0].pending_to_trigger_order_index1 = builder.select(
                self.is_otoco,
                order_instructions[2].pending_order_index,
                order_instructions[0].pending_to_trigger_order_index1,
            );
            order_instructions[1].pending_trigger_status = builder.select(
                self.is_otoco,
                trigger_status_parent_order,
                order_instructions[1].pending_trigger_status,
            );
            order_instructions[1].pending_nonce = builder.select(
                self.is_otoco,
                nil_nonce,
                order_instructions[1].pending_nonce,
            );
            order_instructions[1].pending_to_cancel_order_index0 = builder.select(
                self.is_otoco,
                order_instructions[2].pending_order_index,
                order_instructions[1].pending_to_cancel_order_index0,
            );
            order_instructions[2].pending_trigger_status = builder.select(
                self.is_otoco,
                trigger_status_parent_order,
                order_instructions[2].pending_trigger_status,
            );
            order_instructions[2].pending_nonce = builder.select(
                self.is_otoco,
                nil_nonce,
                order_instructions[2].pending_nonce,
            );
            order_instructions[2].pending_to_cancel_order_index0 = builder.select(
                self.is_otoco,
                order_instructions[1].pending_order_index,
                order_instructions[2].pending_to_cancel_order_index0,
            );
        }

        // Inserted registers won't have a hole in the middle.
        for i in 0..MAX_NB_GROUPED_ORDERS {
            let instruction_flag =
                builder.is_equal(order_instructions[i].instruction_type, insert_order);
            tx_state.put_to_instruction_stack_unsafe(
                builder,
                instruction_flag,
                &order_instructions[i],
                MAX_NB_GROUPED_ORDERS - i - 1,
            );
        }
        tx_state.matching_engine_flag = builder.or(tx_state.matching_engine_flag, self.success);

        self.success
    }
}

pub trait L2CreateGroupedOrdersTxTargetWitness<F: PrimeField64> {
    fn set_l2_create_grouped_orders_tx_target(
        &mut self,
        a: &L2CreateGroupedOrdersTxTarget,
        b: &L2CreateGroupedOrdersTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CreateGroupedOrdersTxTargetWitness<F> for T {
    fn set_l2_create_grouped_orders_tx_target(
        &mut self,
        a: &L2CreateGroupedOrdersTxTarget,
        b: &L2CreateGroupedOrdersTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.grouping_type, F::from_canonical_u8(b.grouping_type))?;
        for i in 0..MAX_NB_GROUPED_ORDERS {
            self.set_target(
                a.orders[i].market_index,
                F::from_canonical_u16(b.orders[i].market_index),
            )?;
            self.set_target(
                a.orders[i].client_order_index,
                F::from_canonical_i64(b.orders[i].client_order_index),
            )?;
            self.set_target(
                a.orders[i].base_amount,
                F::from_canonical_i64(b.orders[i].base_amount),
            )?;
            self.set_target(a.orders[i].price, F::from_canonical_i64(b.orders[i].price))?;
            self.set_target(
                a.orders[i].is_ask.target,
                F::from_canonical_u8(b.orders[i].is_ask),
            )?;
            self.set_target(
                a.orders[i].order_type,
                F::from_canonical_u8(b.orders[i].order_type),
            )?;
            self.set_target(
                a.orders[i].time_in_force,
                F::from_canonical_u8(b.orders[i].time_in_force),
            )?;
            self.set_target(
                a.orders[i].reduce_only,
                F::from_canonical_u8(b.orders[i].reduce_only),
            )?;
            self.set_target(
                a.orders[i].trigger_price,
                F::from_canonical_u32(b.orders[i].trigger_price),
            )?;
            self.set_target(
                a.orders[i].order_expiry,
                F::from_canonical_i64(b.orders[i].order_expiry),
            )?;
        }

        Ok(())
    }
}
