// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
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
use crate::liquidation::get_available_asset_balance_const;
use crate::matching_engine::{
    decrement_locked_balance_for_order, decrement_order_count_in_place,
    get_locked_amount_and_ask_asset_index, get_next_order_nonce, trigger_child_orders,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::account_order::{AccountOrderTarget, select_account_order_target};
use crate::types::account_order_type::AccountOrderTypes;
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::order::{
    OrderTarget, get_market_index_and_order_nonce_from_order_index, select_order_target,
};
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2ModifyOrderTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "m")]
    pub market_index: u16,

    #[serde(rename = "i")]
    pub index: i64, // cloindex or oindex

    #[serde(rename = "b")]
    pub base_amount: i64,

    #[serde(rename = "p")]
    pub price: u32,

    #[serde(rename = "t")]
    pub trigger_price: u32,
}

#[derive(Debug, Clone)]
pub struct L2ModifyOrderTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits
    pub index: Target,         // 56 bits - cloindex or oindex
    pub base_amount: Target,   // 64 bits
    pub price: Target,         // 32 bits
    pub market_index: Target,  // 12 bits
    pub trigger_price: Target, // 32 bits

    // Output
    success: BoolTarget,
    is_perps_market: BoolTarget,
}

impl L2ModifyOrderTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2ModifyOrderTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            index: builder.add_virtual_target(),
            base_amount: builder.add_virtual_target(),
            price: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            trigger_price: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
            is_perps_market: BoolTarget::default(),
        }
    }

    fn get_instruction_from_account_order(
        &self,
        builder: &mut Builder,
        account_order: &AccountOrderTarget,
    ) -> BaseRegisterInfoTarget {
        BaseRegisterInfoTarget {
            instruction_type: builder.constant(F::from_canonical_u8(INSERT_ORDER)),

            market_index: self.market_index,
            account_index: account_order.owner_account_index,
            pending_size: account_order.remaining_base_amount,

            pending_order_index: account_order.order_index,
            pending_client_order_index: account_order.client_order_index,
            pending_initial_size: account_order.initial_base_amount,
            pending_price: account_order.price,
            pending_nonce: account_order.nonce,
            pending_is_ask: account_order.is_ask,

            pending_type: account_order.order_type,
            pending_time_in_force: account_order.time_in_force,
            pending_reduce_only: account_order.reduce_only,
            pending_expiry: account_order.expiry,

            generic_field_0: builder.zero(),

            pending_trigger_price: account_order.trigger_price,
            pending_trigger_status: account_order.trigger_status,
            pending_to_trigger_order_index0: account_order.to_trigger_order_index0,
            pending_to_trigger_order_index1: account_order.to_trigger_order_index1,
            pending_to_cancel_order_index0: account_order.to_cancel_order_index0,
        }
    }

    fn get_modified_order(
        &self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        tx_state: &TxState,
    ) -> (
        AccountOrderTarget,
        BoolTarget, // in progress flag
        BoolTarget, // pending flag
        BoolTarget, // filled flag
    ) {
        let _true = builder._true();
        let _false = builder._false();
        let _zero = builder.zero();
        let mut flag = is_enabled;
        let mut account_order = tx_state.account_order.clone();
        account_order.price = self.price;
        account_order.trigger_price = self.trigger_price;

        let trigger_status_na = builder.constant_from_u8(TRIGGER_STATUS_NA);
        let mut is_pending_order =
            builder.is_not_equal(account_order.trigger_status, trigger_status_na);
        let mut is_in_progress_order = builder.not(is_pending_order);
        let mut is_filled_order = _false;
        let matched_base_amount = builder.sub(
            tx_state.account_order.initial_base_amount,
            tx_state.account_order.remaining_base_amount,
        );

        // Base amount is zero
        {
            let noop_flag = builder.is_zero(self.base_amount);
            flag = builder.and_not(flag, noop_flag);
        }

        // Order is filled
        {
            let filled_flag =
                builder.is_lte(self.base_amount, matched_base_amount, ORDER_SIZE_BITS);
            let filled_flag = builder.and(filled_flag, flag);
            account_order.initial_base_amount = builder.select(
                filled_flag,
                matched_base_amount,
                account_order.initial_base_amount,
            );
            account_order.remaining_base_amount =
                builder.select(filled_flag, _zero, account_order.remaining_base_amount);
            is_pending_order = builder.and_not(is_pending_order, filled_flag);
            is_in_progress_order = builder.and_not(is_in_progress_order, filled_flag);
            is_filled_order = builder.or(filled_flag, is_filled_order);
            flag = builder.and_not(flag, filled_flag);
        }

        // Order is not filled
        {
            account_order.initial_base_amount =
                builder.select(flag, self.base_amount, account_order.initial_base_amount);
            let remaining_base_amount = builder.sub(self.base_amount, matched_base_amount);
            account_order.remaining_base_amount = builder.select(
                flag,
                remaining_base_amount,
                account_order.remaining_base_amount,
            );
        }

        is_pending_order = builder.and(is_pending_order, is_enabled);
        is_in_progress_order = builder.and(is_in_progress_order, is_enabled);
        is_filled_order = builder.and(is_filled_order, is_enabled);

        (
            account_order,
            is_in_progress_order,
            is_pending_order,
            is_filled_order,
        )
    }
}

impl TxHash for L2ModifyOrderTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_MODIFY_ORDER)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.index,
            self.base_amount,
            self.price,
            self.trigger_price,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2ModifyOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_modify_order;

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
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.account_order.owner_account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        self.is_perps_market =
            builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_PERPS);
        let spot_flag = builder.and_not(is_enabled, self.is_perps_market);
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

        // Verify that we load the correct account order
        let is_valid_order_index = builder.is_equal(tx_state.account_order.index_0, self.index);
        let is_valid_client_order_index =
            builder.is_equal(tx_state.account_order.index_1, self.index);
        let is_valid_index = builder.or(is_valid_order_index, is_valid_client_order_index);
        builder.conditional_assert_true(is_enabled, is_valid_index);

        // If the order is empty, sequencer can give random order index(any order index that belongs to an empty order). But because order is empty,
        // "self.success" will still be false because "is_account_order_present" is false.
        let (market_index_from_order, _) = get_market_index_and_order_nonce_from_order_index(
            builder,
            tx_state.account_order.index_0,
        );
        let is_valid_market_index = builder.is_equal(market_index_from_order, self.market_index);

        let is_account_order_empty = tx_state.account_order.is_empty(builder);
        let is_account_order_present = builder.not(is_account_order_empty);
        self.success =
            builder.multi_and(&[is_enabled, is_account_order_present, is_valid_market_index]);

        // We load market only if transaction is successful. Because user may give invalid order index and/or market index. We only cancel orders from active markets
        // to prevent any issues on market closing.
        builder.conditional_assert_eq(
            self.success,
            self.market_index,
            tx_state.market.market_index,
        );
        let is_order_book_enabled =
            builder.is_equal_constant(tx_state.market.status, MARKET_STATUS_ACTIVE as u64);
        builder.conditional_assert_true(self.success, is_order_book_enabled);

        let is_trigger_status_na = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_NA as u64,
        );
        let is_trigger_status_mark_price = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_MARK_PRICE as u64,
        );
        let is_trigger_status_parent_order = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_PARENT_ORDER as u64,
        );
        let is_trigger_status_twap = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_TWAP as u64,
        );

        // Verify that we load the correct order from orderbook
        builder.conditional_assert_eq(
            self.success,
            tx_state.order.nonce_index,
            tx_state.account_order.nonce,
        );
        let success_and_trigger_status_na = builder.and(self.success, is_trigger_status_na);
        builder.conditional_assert_eq(
            success_and_trigger_status_na,
            tx_state.account_order.price,
            tx_state.order.price_index,
        );

        let is_trigger_status_na_or_twap = builder.or(is_trigger_status_na, is_trigger_status_twap);
        let is_trigger_status_mark_price_or_parent_order =
            builder.or(is_trigger_status_mark_price, is_trigger_status_parent_order);
        let is_trigger_status_ok = builder.or(
            is_trigger_status_na_or_twap,
            is_trigger_status_mark_price_or_parent_order,
        );
        self.success = builder.and(self.success, is_trigger_status_ok);

        let is_tx_trigger_price_zero = builder.is_zero(self.trigger_price);
        let is_tx_base_amount_zero = builder.is_zero(self.base_amount);
        let is_order_trigger_price_zero = builder.is_zero(tx_state.account_order.trigger_price);
        let is_order_base_amount_zero = builder.is_zero(tx_state.account_order.initial_base_amount);
        let is_trigger_price_consistent = builder.is_equal(
            is_tx_trigger_price_zero.target,
            is_order_trigger_price_zero.target,
        );
        let is_base_amount_consistent = builder.is_equal(
            is_tx_base_amount_zero.target,
            is_order_base_amount_zero.target,
        );

        let na_or_twap_flag = builder.and(self.success, is_trigger_status_na_or_twap);
        let success_na_or_twap_flag =
            builder.and_not(is_tx_trigger_price_zero, is_tx_base_amount_zero);
        self.success = builder.select_bool(na_or_twap_flag, success_na_or_twap_flag, self.success);

        let mark_price_or_parent_order_flag =
            builder.and(self.success, is_trigger_status_mark_price_or_parent_order);
        let success_mark_price_or_parent_order_flag =
            builder.and(is_base_amount_consistent, is_trigger_price_consistent);
        self.success = builder.select_bool(
            mark_price_or_parent_order_flag,
            success_mark_price_or_parent_order_flag,
            self.success,
        );

        // Price range checks
        builder.conditional_assert_not_zero(is_enabled, self.price);
        builder.register_range_check(self.price, ORDER_PRICE_BITS);

        builder.register_range_check(self.trigger_price, ORDER_PRICE_BITS);

        // BaseAmount - (Must fit in 48 bits) & (can be zero)
        builder.register_range_check(self.base_amount, ORDER_SIZE_BITS);

        let order_type_target = AccountOrderTypes::new(builder, tx_state.account_order.order_type);
        let liquidation_status = tx_state.risk_infos[OWNER_ACCOUNT_ID]
            .current_risk_parameters
            .get_health(builder);
        let healthy_status = builder.constant_from_u8(HEALTHY);
        let pre_liquidation_status = builder.constant_from_u8(PRE_LIQUIDATION);
        let is_healthy = builder.is_equal(liquidation_status, healthy_status);
        let is_pre_liquidation = builder.is_equal(liquidation_status, pre_liquidation_status);
        let is_liquidation_status_ok = builder.or(is_healthy, is_pre_liquidation);
        let health_check_flag = builder.and(is_enabled, self.is_perps_market);
        builder.conditional_assert_true(health_check_flag, is_liquidation_status_ok);

        // only allow order creation if ask nonce < bid nonce, nonces are initially set so that ask nonce is smaller than bid nonce
        // since only the order creation can change one of the ask or bid nonces by exactly one, checking if orderBook.AskNonce != orderBook.BidNonce is enough
        builder.conditional_assert_not_eq(
            self.success,
            tx_state.market.ask_nonce,
            tx_state.market.bid_nonce,
        );

        // Check new order base / quote amounts
        let ioc = builder.constant_from_u8(IOC);
        let is_ioc = builder.is_equal(tx_state.account_order.time_in_force, ioc);
        let is_base_amount_zero = builder.is_zero(self.base_amount);
        let is_valid_base_size_and_price = tx_state.is_valid_base_size_and_price(
            builder,
            self.base_amount,
            self.price,
            order_type_target.is_twap_order,
            is_ioc,
        );
        let is_valid_base_size_and_price_or_zero =
            builder.or(is_valid_base_size_and_price, is_base_amount_zero);
        self.success = builder.and(self.success, is_valid_base_size_and_price_or_zero);

        // Check order expiry
        let is_order_expiry_gt_block_created_at = builder.is_gt(
            tx_state.account_order.expiry,
            tx_state.block_timestamp,
            TIMESTAMP_BITS,
        );
        self.success = builder.and(self.success, is_order_expiry_gt_block_created_at);

        // Spot balance check
        {
            let spot_success_flag = builder.and_not(self.success, self.is_perps_market);

            // Check if the new base amount will exceed the matched base amount (initial - remaining)
            // If so, we calculate old and new locked balances and see if the available asset balance
            // allows that to happen.
            let matched_base_amount = builder.sub(
                tx_state.account_order.initial_base_amount,
                tx_state.account_order.remaining_base_amount,
            );
            let new_base_amount_gt_matched_amount = builder.is_gt(
                self.base_amount,
                matched_base_amount,
                ORDER_BASE_AMOUNT_BITS,
            );
            let flag = builder.and(spot_success_flag, new_base_amount_gt_matched_amount);

            let (old_locked_amount, ask_asset_index) = get_locked_amount_and_ask_asset_index(
                builder,
                flag,
                &tx_state.market,
                tx_state.account_order.remaining_base_amount,
                tx_state.account_order.price,
                tx_state.account_order.is_ask,
            );

            let new_remaining_base_amount = builder.sub(self.base_amount, matched_base_amount);
            let (new_locked_amount, _) = get_locked_amount_and_ask_asset_index(
                builder,
                flag,
                &tx_state.market,
                new_remaining_base_amount,
                self.price,
                tx_state.account_order.is_ask,
            );
            let (locked_amount_delta, old_amount_was_greater) =
                builder.try_sub_biguint(&new_locked_amount, &old_locked_amount);
            let new_locked_gte_old = builder.not(BoolTarget::new_unsafe(old_amount_was_greater.0));

            let is_base_asset = builder.is_equal(
                tx_state.account_assets[OWNER_ACCOUNT_ID][BASE_ASSET_ID].index_0,
                ask_asset_index,
            );

            let base_asset_available_balance = get_available_asset_balance_const(
                builder,
                PRODUCT_TYPE_SPOT,
                &tx_state.accounts[OWNER_ACCOUNT_ID],
                &tx_state.account_assets[OWNER_ACCOUNT_ID][BASE_ASSET_ID],
                tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][BASE_ASSET_ID],
                &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
            );
            let quote_asset_available_balance = get_available_asset_balance_const(
                builder,
                PRODUCT_TYPE_SPOT,
                &tx_state.accounts[OWNER_ACCOUNT_ID],
                &tx_state.account_assets[OWNER_ACCOUNT_ID][QUOTE_ASSET_ID],
                tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][QUOTE_ASSET_ID],
                &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
            );

            let available_balance = builder.select_biguint(
                is_base_asset,
                &base_asset_available_balance,
                &quote_asset_available_balance,
            );
            let not_enough_available_balance =
                builder.is_lt_biguint(&available_balance, &locked_amount_delta);

            let should_be_false =
                builder.multi_and(&[flag, new_locked_gte_old, not_enough_available_balance]);

            builder.conditional_assert_false(self.success, should_be_false);
        }
    }
}

impl Apply for L2ModifyOrderTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();

        let (mut new_account_order, is_in_progress_order, is_pending_order, is_filled_order) =
            self.get_modified_order(builder, self.success, tx_state);

        let nonce_index =
            get_next_order_nonce(builder, &tx_state.market, tx_state.account_order.is_ask);
        let ask_nonce_plus_one = builder.add(tx_state.market.ask_nonce, one);
        let bid_nonce_minus_one = builder.sub(tx_state.market.bid_nonce, one);
        let new_ask_nonce = builder.select(
            tx_state.account_order.is_ask,
            ask_nonce_plus_one,
            tx_state.market.ask_nonce,
        );
        let new_bid_nonce = builder.select(
            tx_state.account_order.is_ask,
            tx_state.market.bid_nonce,
            bid_nonce_minus_one,
        );
        tx_state.market.ask_nonce = builder.select(
            is_in_progress_order,
            new_ask_nonce,
            tx_state.market.ask_nonce,
        );
        tx_state.market.bid_nonce = builder.select(
            is_in_progress_order,
            new_bid_nonce,
            tx_state.market.bid_nonce,
        );
        new_account_order.nonce = builder.select(
            is_in_progress_order,
            nonce_index,
            tx_state.account_order.nonce,
        );

        // Cleanup the order book order
        let empty_order = OrderTarget::empty(
            builder,
            tx_state.order.price_index,
            tx_state.order.nonce_index,
        );
        let is_filled_or_in_progress = builder.or(is_in_progress_order, is_filled_order);
        tx_state.order = select_order_target(
            builder,
            is_filled_or_in_progress,
            &empty_order,
            &tx_state.order,
        );
        let empty_account_order = AccountOrderTarget::empty(
            builder,
            tx_state.account_order.index_0,
            tx_state.account_order.index_1,
            tx_state.account_order.owner_account_index,
        );
        decrement_order_count_in_place(
            builder,
            tx_state,
            TAKER_ACCOUNT_ID,
            is_filled_or_in_progress,
            tx_state.account_order.trigger_status,
            tx_state.account_order.reduce_only,
        );
        let account_order_copy = tx_state.account_order.clone();
        tx_state.account_order = select_account_order_target(
            builder,
            is_filled_or_in_progress,
            &empty_account_order,
            &tx_state.account_order,
        );

        let is_limit_order =
            builder.is_equal_constant(tx_state.account_order.order_type, LIMIT_ORDER as u64);
        let spot_flag = builder.not(self.is_perps_market);
        let locked_balance_flag =
            builder.multi_and(&[is_filled_or_in_progress, spot_flag, is_limit_order]);
        decrement_locked_balance_for_order(
            builder,
            locked_balance_flag,
            &account_order_copy,
            &tx_state.market,
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID],
        );

        // Set new register
        let instruction = self.get_instruction_from_account_order(builder, &new_account_order);
        tx_state.insert_to_instruction_stack(builder, is_in_progress_order, &instruction);

        // Set new account order
        tx_state.account_order = select_account_order_target(
            builder,
            is_pending_order,
            &new_account_order,
            &tx_state.account_order,
        );

        // Trigger child orders
        trigger_child_orders(
            builder,
            is_filled_order,
            tx_state,
            self.market_index,
            new_account_order.owner_account_index,
            new_account_order.to_trigger_order_index0,
            new_account_order.to_trigger_order_index1,
            new_account_order.initial_base_amount,
        );

        tx_state.update_impact_prices_flag =
            builder.or(tx_state.update_impact_prices_flag, self.success);

        self.success
    }
}

pub trait L2ModifyOrderTxTargetWitness<F: PrimeField64> {
    fn set_l2_modify_order_tx_target(
        &mut self,
        a: &L2ModifyOrderTxTarget,
        b: &L2ModifyOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2ModifyOrderTxTargetWitness<F> for T {
    fn set_l2_modify_order_tx_target(
        &mut self,
        a: &L2ModifyOrderTxTarget,
        b: &L2ModifyOrderTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.index, F::from_canonical_i64(b.index))?;
        self.set_target(a.base_amount, F::from_canonical_i64(b.base_amount))?;
        self.set_target(a.price, F::from_canonical_u32(b.price))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(a.trigger_price, F::from_canonical_u32(b.trigger_price))?;

        Ok(())
    }
}
