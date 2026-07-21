// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::matching_engine::{
    get_next_order_nonce, increment_order_count_in_place, is_not_valid_reduce_only_direction,
};
use crate::tx_attributes::{
    ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX, ATTR_INTEGRATOR_MAKER_FEE, ATTR_INTEGRATOR_TAKER_FEE,
    ATTR_SELF_TRADE_BEHAVIOR_MODE, ATTR_SELF_TRADE_EQUALITY_MODE, TxAttributesTarget,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::account_order::{AccountOrderTarget, OrderFlags, select_account_order_target};
use crate::types::account_order_type::AccountOrderTypes;
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::market_details::MarketFlags;
use crate::types::order::get_order_index;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CreateOrderTx {
    #[serde(rename = "ai")]
    pub account_index: i64, // 48 bits

    #[serde(rename = "ki")]
    pub api_key_index: u8,

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

#[derive(Debug, Clone)]
pub struct L2CreateOrderTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits

    pub market_index: Target, // 12 bits

    pub client_order_index: Target, // 48 bits

    pub base_amount: Target, // 48 bits
    pub price: Target,       // 32 bits
    pub is_ask: BoolTarget,

    pub order_type: Target,
    pub time_in_force: Target,
    pub reduce_only: Target,
    pub trigger_price: Target, // 32 bits
    pub order_expiry: Target,  // 48 bits

    // Helper
    trigger_status: Target,
    next_order_nonce: Target,
    next_order_index: Target,
    is_pending_order: BoolTarget,
    calculated_base_amount: Target,

    // Output
    success: BoolTarget,
    is_perps_market: BoolTarget,
}

impl L2CreateOrderTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2CreateOrderTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
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

            // Helper
            trigger_status: builder.zero(),
            next_order_nonce: builder.zero(),
            next_order_index: builder.zero(),
            is_pending_order: builder._false(),
            calculated_base_amount: builder.zero(),

            // output
            success: BoolTarget::default(),
            is_perps_market: BoolTarget::default(),
        }
    }

    fn get_in_progress_order_register(
        &self,
        builder: &mut Builder,
        tx_attributes: &TxAttributesTarget,
    ) -> BaseRegisterInfoTarget {
        let (generic_field_1, generic_field_2, generic_field_3) =
            tx_attributes.get_register_generic_fields(builder);

        BaseRegisterInfoTarget {
            instruction_type: builder.constant(F::from_canonical_u8(INSERT_ORDER)),

            market_index: self.market_index,
            account_index: self.account_index,

            pending_size: self.calculated_base_amount,
            pending_order_index: self.next_order_index,
            pending_client_order_index: self.client_order_index,
            pending_initial_size: self.calculated_base_amount,
            pending_price: self.price,
            pending_nonce: self.next_order_nonce,
            pending_is_ask: self.is_ask,

            pending_type: self.order_type,
            pending_time_in_force: self.time_in_force,
            pending_reduce_only: self.reduce_only,
            pending_expiry: self.order_expiry,

            generic_field_0: builder.zero(),

            pending_trigger_price: self.trigger_price,
            pending_trigger_status: self.trigger_status,
            pending_to_trigger_order_index0: builder.zero(),
            pending_to_trigger_order_index1: builder.zero(),
            pending_to_cancel_order_index0: builder.zero(),

            generic_field_1,
            generic_field_2,
            generic_field_3,
        }
    }

    fn get_pending_account_order(
        &self,
        builder: &mut Builder,
        tx_attributes: &TxAttributesTarget,
    ) -> AccountOrderTarget {
        AccountOrderTarget {
            index_0: self.next_order_index,
            index_1: self.client_order_index,

            order_index: self.next_order_index,
            client_order_index: self.client_order_index,

            owner_account_index: self.account_index,
            initial_base_amount: self.calculated_base_amount,
            price: self.price,
            nonce: builder.zero(),
            remaining_base_amount: self.calculated_base_amount,
            is_ask: self.is_ask,

            order_type: self.order_type,
            time_in_force: self.time_in_force,
            reduce_only: self.reduce_only,
            trigger_price: self.trigger_price,
            expiry: self.order_expiry,

            integrator_fee_collector_index: tx_attributes.get(ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX),
            integrator_taker_fee: tx_attributes.get(ATTR_INTEGRATOR_TAKER_FEE),
            integrator_maker_fee: tx_attributes.get(ATTR_INTEGRATOR_MAKER_FEE),
            order_flags: OrderFlags {
                self_trade_behavior_mode: tx_attributes.get(ATTR_SELF_TRADE_BEHAVIOR_MODE),
                self_trade_equality_mode: tx_attributes.get(ATTR_SELF_TRADE_EQUALITY_MODE),
            }
            .to_target(builder),

            trigger_status: self.trigger_status,
            to_trigger_order_index0: builder.zero(),
            to_trigger_order_index1: builder.zero(),
            to_cancel_order_index0: builder.zero(),
        }
    }

    fn register_range_checks(&self, builder: &mut Builder) {
        builder.register_range_check(self.base_amount, ORDER_BASE_AMOUNT_BITS);
        builder.register_range_check(self.trigger_price, ORDER_PRICE_BITS);
        builder.register_range_check(self.price, ORDER_PRICE_BITS);
        builder.register_range_check(self.order_expiry, TIMESTAMP_BITS);
        builder.assert_bool(BoolTarget::new_unsafe(self.reduce_only));
        builder.assert_bool(self.is_ask);
    }
}

impl TxHash for L2CreateOrderTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_ORDER)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.client_order_index,
            self.base_amount,
            self.price,
            self.is_ask.target,
            self.order_type,
            self.time_in_force,
            self.reduce_only,
            self.trigger_price,
            self.order_expiry,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CreateOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_create_order;
        self.success = is_enabled;

        self.register_range_checks(builder);

        let is_ioc = builder.is_equal_constant(self.time_in_force, IOC as u64);
        let is_gtt = builder.is_equal_constant(self.time_in_force, GTT as u64);
        let is_post_only = builder.is_equal_constant(self.time_in_force, POST_ONLY as u64);

        let is_nil_trigger_price =
            builder.is_equal_constant(self.trigger_price, NIL_ORDER_TRIGGER_PRICE as u64);
        let is_order_expiry_nil =
            builder.is_equal_constant(self.order_expiry, NIL_ORDER_EXPIRY as u64);

        self.is_perps_market =
            builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_PERPS);
        let is_spot_market = builder.not(self.is_perps_market);
        let treasury_account_index = builder.constant_usize(TREASURY_ACCOUNT_INDEX);
        let is_treasury = builder.is_equal(self.account_index, treasury_account_index);
        // Treasury can only create spot orders
        let treasury_flag = builder.and(is_enabled, is_treasury);
        builder.conditional_assert_true(treasury_flag, is_spot_market);

        self.next_order_nonce = get_next_order_nonce(builder, &tx_state.market, self.is_ask);
        self.next_order_index =
            get_order_index(builder, tx_state.market.market_index, self.next_order_nonce);

        /***********************/
        /*  State leaf checks  */
        /***********************/
        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
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

        let spot_flag = builder.and(is_enabled, is_spot_market);
        builder.conditional_assert_eq(
            spot_flag,
            tx_state.market.base_asset_id,
            tx_state.asset_indices[BASE_ASSET_ID],
        );
        builder.conditional_assert_eq(
            spot_flag,
            tx_state.market.quote_asset_id,
            tx_state.asset_indices[QUOTE_ASSET_ID],
        );
        let perps_flag = builder.and(is_enabled, self.is_perps_market);
        builder.conditional_assert_eq_constant(
            perps_flag,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        // Perps market margin-mode gating.
        {
            let market_flags =
                MarketFlags::from_target(builder, tx_state.market_details.market_flags);

            let is_taker_public_pool = builder.is_equal_constant(
                tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
                PUBLIC_POOL_ACCOUNT_TYPE as u64,
            );
            let is_taker_insurance_fund = builder.is_equal_constant(
                tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let flag = builder.and_not(perps_flag, is_taker_insurance_fund);

            let is_taker_isolated = builder.is_equal_constant(
                tx_state.positions[TAKER_ACCOUNT_ID].margin_mode,
                ISOLATED_MARGIN as u64,
            );
            let is_taker_cross = builder.is_equal_constant(
                tx_state.positions[TAKER_ACCOUNT_ID].margin_mode,
                CROSS_MARGIN as u64,
            );
            let is_margin_set = builder.is_equal_constant(
                tx_state.positions[TAKER_ACCOUNT_ID].margin_set_flag,
                MARGIN_SET as u64,
            );

            // Pools can't trade on isolated-only markets
            let pool_on_isolated_only =
                builder.and(is_taker_public_pool, market_flags.is_isolated_only());
            builder.conditional_assert_false(flag, pool_on_isolated_only);

            // On isolated-only: explicitly-cross position (MarginSetFlag=Set AND MarginMode=Cross) conflicts
            let explicitly_cross = builder.and(is_margin_set, is_taker_cross);
            let explicitly_cross_on_isolated =
                builder.and(market_flags.is_isolated_only(), explicitly_cross);
            builder.conditional_assert_false(flag, explicitly_cross_on_isolated);

            // On isolated-only without fallback: must already be isolated
            let isolated_no_fallback = builder.and_not(
                market_flags.is_isolated_only(),
                BoolTarget::new_unsafe(market_flags.default_margin_mode),
            );
            let should_be_false = builder.and_not(isolated_no_fallback, is_taker_isolated);
            builder.conditional_assert_false(flag, should_be_false);
        }

        // TimeInForce - Either IOC (0), GTT (1) or POST_ONLY (2)
        let is_time_in_force_valid = builder.multi_or(&[is_ioc, is_gtt, is_post_only]);
        builder.assert_true(is_time_in_force_valid);

        let order_type_target = AccountOrderTypes::new(builder, self.order_type);
        builder.assert_true(order_type_target.is_valid_l2_create_order);

        /*****************************/
        /*  Limit Order Validations  */
        /*****************************/
        let is_enabled_and_limit_order = builder.and(is_enabled, order_type_target.is_limit_order);
        // TriggerPrice must be nil for Limit Order
        builder.conditional_assert_true(is_enabled_and_limit_order, is_nil_trigger_price);
        // If Limit Order TimeInForce is IoC, then OrderExpiry must be nil
        let is_enabled_and_limit_order_and_ioc = builder.and(is_enabled_and_limit_order, is_ioc);
        builder.conditional_assert_true(is_enabled_and_limit_order_and_ioc, is_order_expiry_nil);
        // If TimeInForce is GTT or PostOnly, then OrderExpiry must not be nil
        let is_enabled_and_limit_order_and_not_ioc =
            builder.and_not(is_enabled_and_limit_order, is_ioc);
        builder
            .conditional_assert_false(is_enabled_and_limit_order_and_not_ioc, is_order_expiry_nil);

        /******************************/
        /*  Market Order Validations  */
        /******************************/
        // Market Order has to be IOC, no need for conditional since default is limit order
        builder.conditional_assert_true(
            order_type_target.is_market_order, // We can omit the is_enabled check here because MARKET_ORDER=1 is not the default value
            is_ioc,
        );
        // Market order expiry has to be nil(zero)
        builder.conditional_assert_true(
            order_type_target.is_market_order, // We can omit the is_enabled check here because MARKET_ORDER=1 is not the default value
            is_order_expiry_nil,
        );
        // Market order trigger price has to be nil(zero)
        builder.conditional_assert_true(
            order_type_target.is_market_order, // We can omit the is_enabled check here because MARKET_ORDER=1 is not the default value
            is_nil_trigger_price,
        );

        /***********************************************/
        /*  StopLoss and TakeProfit Order Validations  */
        /***********************************************/
        let is_stop_loss_or_take_profit_order = builder.or(
            order_type_target.is_stop_loss_order,
            order_type_target.is_take_profit_order,
        );
        let is_enabled_and_stop_loss_or_take_profit_order =
            builder.and(is_enabled, is_stop_loss_or_take_profit_order);
        // It must be a perp market
        builder.conditional_assert_true(
            is_enabled_and_stop_loss_or_take_profit_order,
            self.is_perps_market,
        );
        // TimeInForce must be IOC for StopLoss and TakeProfit Market Orders
        builder.conditional_assert_true(is_enabled_and_stop_loss_or_take_profit_order, is_ioc);
        // Trigger Price must not be nil for StopLoss and TakeProfit Market Orders
        builder.conditional_assert_false(
            is_enabled_and_stop_loss_or_take_profit_order,
            is_nil_trigger_price,
        );
        // OrderExpiry must not be nil for StopLoss and TakeProfit Market Orders
        builder.conditional_assert_false(
            is_enabled_and_stop_loss_or_take_profit_order,
            is_order_expiry_nil,
        );

        /*********************************************************/
        /*  StopLossLimit and TakeProfitLimit Order Validations  */
        /*********************************************************/
        let is_stop_loss_limit_or_take_profit_limit_order = builder.or(
            order_type_target.is_stop_loss_limit_order,
            order_type_target.is_take_profit_limit_order,
        );
        let is_enabled_and_stop_loss_limit_or_take_profit_limit_order =
            builder.and(is_enabled, is_stop_loss_limit_or_take_profit_limit_order);
        // It must be a perp market
        builder.conditional_assert_true(
            is_enabled_and_stop_loss_limit_or_take_profit_limit_order,
            self.is_perps_market,
        );
        // Trigger price must not be nil
        builder.conditional_assert_false(
            is_enabled_and_stop_loss_limit_or_take_profit_limit_order,
            is_nil_trigger_price,
        );
        // OrderExpiry must not be nil for StopLoss and TakeProfit Market Orders
        builder.conditional_assert_false(
            is_enabled_and_stop_loss_limit_or_take_profit_limit_order,
            is_order_expiry_nil,
        );

        /****************************/
        /*  TWAP Order Validations  */
        /****************************/
        let is_enabled_and_twap_order = builder.and(is_enabled, order_type_target.is_twap_order);
        // Time in force must be GTT
        builder.conditional_assert_true(is_enabled_and_twap_order, is_gtt);
        // Trigger price must be nil
        builder.conditional_assert_true(is_enabled_and_twap_order, is_nil_trigger_price);
        // Order expiry must not be nil
        builder.conditional_assert_false(is_enabled_and_twap_order, is_order_expiry_nil);

        /**********************************/
        /*  L2 Create Order Verification  */
        /**********************************/
        builder.conditional_assert_not_zero(is_enabled, self.price);

        // OB Status - Must be active
        let ob_active_status = builder.constant(F::from_canonical_u8(MARKET_STATUS_ACTIVE));
        builder.conditional_assert_eq(is_enabled, tx_state.market.status, ob_active_status);

        // Only allow order creation if market is not full, i.e. ask nonce < bid nonce, nonces are initially set so that ask nonce is smaller than bid nonce
        // since only the order creation can change one of the ask or bid nonces by exactly one, checking if orderBook.AskNonce != orderBook.BidNonce is enough
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.market.ask_nonce,
            tx_state.market.bid_nonce,
        );

        // Compute order trigger status
        let trigger_status_na = builder.constant_from_u8(TRIGGER_STATUS_NA);
        let trigger_status_mark_price = builder.constant_from_u8(TRIGGER_STATUS_MARK_PRICE);
        let trigger_status_twap = builder.constant_from_u8(TRIGGER_STATUS_TWAP);
        self.trigger_status = builder.select(
            order_type_target.is_twap_order,
            trigger_status_twap,
            trigger_status_na,
        );
        self.trigger_status = builder.select(
            order_type_target.is_conditional_order,
            trigger_status_mark_price,
            self.trigger_status,
        );

        // Client order id uniqueness check
        self.success = builder.and(self.success, tx_state.is_cloid_unique[0]);

        // Verify order base amounts
        self.calculated_base_amount = self.base_amount;

        let base_amount_is_zero = builder.is_zero(self.base_amount);
        let not_conditional = builder.not(order_type_target.is_conditional_order);
        let update_calculated_base_amount = builder.and(not_conditional, base_amount_is_zero);
        let position_tied_order_base_amount = tx_state.positions[TAKER_ACCOUNT_ID]
            .calculate_position_tied_order_base_amount(
                builder,
                tx_state.market_details.quote_multiplier,
                self.price,
                tx_state.market.order_quote_limit,
            );
        self.calculated_base_amount = builder.select(
            update_calculated_base_amount,
            position_tied_order_base_amount,
            self.calculated_base_amount,
        );
        let valid_base_size_and_price = tx_state.is_valid_base_size_and_price(
            builder,
            self.calculated_base_amount,
            self.price,
            order_type_target.is_twap_order,
            is_ioc,
        );
        let base_amount_not_zero = builder.not(base_amount_is_zero);
        let base_amount_check_flag = builder.or(not_conditional, base_amount_not_zero);
        let base_amount_app_error_flag =
            builder.and_not(base_amount_check_flag, valid_base_size_and_price);
        self.success = builder.and_not(self.success, base_amount_app_error_flag);

        // Spot validations
        {
            let flag = builder.and_not(self.success, self.is_perps_market);

            // Can only be NA or TWAP
            let trigger_status_na =
                builder.is_equal_constant(self.trigger_status, TRIGGER_STATUS_NA as u64);
            let trigger_status_twap =
                builder.is_equal_constant(self.trigger_status, TRIGGER_STATUS_TWAP as u64);
            let is_trigger_status_valid =
                builder.multi_or(&[trigger_status_na, trigger_status_twap]);
            builder.conditional_assert_true(flag, is_trigger_status_valid);
            // Reduce only must be 0
            builder.conditional_assert_zero(flag, self.reduce_only);
            // Base amount can't be 0
            builder.conditional_assert_not_zero(flag, self.base_amount);
            // Trigger price has to be 0
            builder.conditional_assert_zero(flag, self.trigger_price);

            // Public pools can't open spot orders.
            let is_public_pool = builder.is_equal_constant(
                tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
                PUBLIC_POOL_ACCOUNT_TYPE as u64,
            );
            builder.conditional_assert_false(flag, is_public_pool);

            // Insurance funds may spot-trade, but only assets in the margined assets list.
            let is_insurance_fund = builder.is_equal_constant(
                tx_state.accounts[TAKER_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let insurance_fund_spot_flag = builder.and(flag, is_insurance_fund);
            builder.conditional_assert_true(
                insurance_fund_spot_flag,
                BoolTarget::new_unsafe(tx_state.assets[BASE_ASSET_ID].margin_mode),
            );
            builder.conditional_assert_true(
                insurance_fund_spot_flag,
                BoolTarget::new_unsafe(tx_state.assets[QUOTE_ASSET_ID].margin_mode),
            );
        }

        // Perps validations
        {
            let flag = builder.and(self.success, self.is_perps_market);

            builder.conditional_assert_not_zero(flag, tx_state.market_details.index_price);
            builder.conditional_assert_not_zero(flag, tx_state.market_details.mark_price);

            // Reduce only direction validation
            let invalid_reduce_only_direction = is_not_valid_reduce_only_direction(
                builder,
                tx_state.positions[TAKER_ACCOUNT_ID].position.sign,
                self.is_ask,
            );
            let invalid_reduce_only_direction_check = builder.and(
                BoolTarget::new_unsafe(self.reduce_only),
                invalid_reduce_only_direction,
            );

            let should_be_false = builder.and(flag, invalid_reduce_only_direction_check);
            self.success = builder.and_not(self.success, should_be_false);
        }

        // order should not be already expired when created
        let is_order_expiry_in_future =
            builder.is_lt(tx_state.block_timestamp, self.order_expiry, TIMESTAMP_BITS);
        let is_order_expiry_valid = builder.or(is_order_expiry_nil, is_order_expiry_in_future);
        self.success = builder.and(self.success, is_order_expiry_valid);

        // Compute if order needs to be put on register (active)
        self.is_pending_order = builder.or(
            order_type_target.is_twap_order,
            order_type_target.is_conditional_order,
        );

        let should_update_for_pending_order = builder.and(self.success, self.is_pending_order);
        builder.conditional_assert_eq(
            should_update_for_pending_order,
            tx_state.account_order.index_0,
            self.next_order_index,
        );
        builder.conditional_assert_eq(
            should_update_for_pending_order,
            tx_state.account_order.index_1,
            self.client_order_index,
        );
    }
}

impl Apply for L2CreateOrderTxTarget {
    // order_before: If top_order is empty or there is no matching order for limit, order_before is empty and/or taker order
    // oterwise it is always the maker order
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();

        // Set new market
        let ask_nonce_plus_one = builder.add(tx_state.market.ask_nonce, one);
        let bid_nonce_minus_one = builder.sub(tx_state.market.bid_nonce, one);
        let new_ask_nonce =
            builder.select(self.is_ask, ask_nonce_plus_one, tx_state.market.ask_nonce);
        let new_bid_nonce =
            builder.select(self.is_ask, tx_state.market.bid_nonce, bid_nonce_minus_one);
        tx_state.market.ask_nonce =
            builder.select(self.success, new_ask_nonce, tx_state.market.ask_nonce);
        tx_state.market.bid_nonce =
            builder.select(self.success, new_bid_nonce, tx_state.market.bid_nonce);

        // Pending order - put order to account order tree
        {
            let should_update_for_pending_order = builder.and(self.success, self.is_pending_order);
            // Set new account order info
            let new_account_order = self.get_pending_account_order(builder, &tx_state.attributes);
            tx_state.account_order = select_account_order_target(
                builder,
                should_update_for_pending_order,
                &new_account_order,
                &tx_state.account_order,
            );
            increment_order_count_in_place(
                builder,
                tx_state,
                should_update_for_pending_order,
                tx_state.account_order.trigger_status,
                tx_state.account_order.reduce_only,
            );
        }

        // In progress order - call matching engine
        {
            // Set new register
            let new_register = self.get_in_progress_order_register(builder, &tx_state.attributes);
            let in_progress_flag = builder.and_not(self.success, self.is_pending_order);
            tx_state.put_to_instruction_stack_unsafe(builder, in_progress_flag, &new_register, 0);

            // Update matching engine flag if cloid not enabled and order is not pending
            tx_state.matching_engine_flag =
                builder.or(in_progress_flag, tx_state.matching_engine_flag);
        }

        // Set update impact prices flag
        tx_state.update_impact_prices_flag =
            builder.or(self.success, tx_state.update_impact_prices_flag);

        self.success
    }
}

pub trait L2CreateOrderTxTargetWitness<F: PrimeField64> {
    fn set_l2_create_order_tx_target(
        &mut self,
        a: &L2CreateOrderTxTarget,
        b: &L2CreateOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CreateOrderTxTargetWitness<F> for T {
    fn set_l2_create_order_tx_target(
        &mut self,
        a: &L2CreateOrderTxTarget,
        b: &L2CreateOrderTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(
            a.client_order_index,
            F::from_canonical_i64(b.client_order_index),
        )?;
        self.set_target(a.order_type, F::from_canonical_u8(b.order_type))?;
        self.set_target(a.time_in_force, F::from_canonical_u8(b.time_in_force))?;
        self.set_target(a.order_expiry, F::from_canonical_i64(b.order_expiry))?;
        self.set_target(a.base_amount, F::from_canonical_i64(b.base_amount))?;
        self.set_target(a.price, F::from_canonical_i64(b.price))?;
        self.set_bool_target(a.is_ask, b.is_ask == 1)?;
        self.set_target(a.reduce_only, F::from_canonical_u8(b.reduce_only))?;
        self.set_target(a.trigger_price, F::from_canonical_u32(b.trigger_price))?;

        Ok(())
    }
}
