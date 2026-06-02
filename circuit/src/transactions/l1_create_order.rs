// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::matching_engine::{get_next_order_nonce, is_not_valid_reduce_only_direction};
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::order::get_order_index;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L1CreateOrderTx {
    #[serde(rename = "mai")]
    pub master_account_index: i64,

    #[serde(rename = "ai")]
    pub account_index: i64, // 48 bits

    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "b")]
    pub base_amount: i64, // 48 bits
    #[serde(rename = "p")]
    pub price: u32,

    #[serde(rename = "ia")]
    pub is_ask: u8,

    #[serde(rename = "ot")]
    pub order_type: u8,
}

#[derive(Debug, Clone)]
pub struct L1CreateOrderTxTarget {
    pub master_account_index: Target,
    pub account_index: Target, // 48 bits

    pub market_index: Target, // 8 bits

    pub base_amount: Target, // 48 bits
    pub price: Target,       // 32 bits
    pub is_ask: BoolTarget,
    pub order_type: Target,

    // Helpers
    calculated_base_amount: Target,

    // Output
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1CreateOrderTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            master_account_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            base_amount: builder.add_virtual_target(),
            price: builder.add_virtual_target(),
            is_ask: builder.add_virtual_bool_target_safe(),
            order_type: builder.add_virtual_target(),

            // Helpers
            calculated_base_amount: Target::default(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }

    fn get_pending_order_register(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
    ) -> BaseRegisterInfoTarget {
        let ioc = builder.constant_from_u8(IOC);
        let next_order_nonce = get_next_order_nonce(builder, &tx_state.market, self.is_ask);
        let next_order_index =
            get_order_index(builder, tx_state.market.market_index, next_order_nonce);

        BaseRegisterInfoTarget {
            instruction_type: builder.constant_from_u8(INSERT_ORDER),

            market_index: self.market_index,
            account_index: self.account_index,

            pending_size: self.calculated_base_amount,
            pending_order_index: next_order_index,
            pending_client_order_index: builder.constant_i64(NIL_CLIENT_ORDER_INDEX),
            pending_initial_size: self.calculated_base_amount,
            pending_price: self.price,
            pending_nonce: next_order_nonce,
            pending_is_ask: self.is_ask,
            pending_type: self.order_type,
            pending_time_in_force: ioc,
            pending_reduce_only: builder.one(),
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

impl Verify for L1CreateOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.is_enabled = tx_type.is_l1_create_order;
        self.success = tx_type.is_l1_create_order;

        builder.conditional_assert_eq(
            self.is_enabled,
            self.market_index,
            tx_state.market.market_index,
        );
        builder.conditional_assert_eq(
            self.is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );
        builder.conditional_assert_eq(
            self.is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        // OrderType - Either limit or market
        let is_limit_order = builder.is_equal_constant(self.order_type, LIMIT_ORDER as u64);
        let is_market_order = builder.is_equal_constant(self.order_type, MARKET_ORDER as u64);
        let is_order_type_valid = builder.or(is_limit_order, is_market_order);
        builder.conditional_assert_true(self.success, is_order_type_valid);

        // Check if is_ask is a boolean
        builder.assert_bool(self.is_ask);

        // Range check base amount
        builder.register_range_check(self.base_amount, ORDER_BASE_AMOUNT_BITS);

        // Price - (Must fit in 32 bits) & (can be zero)
        builder.register_range_check(self.price, ORDER_PRICE_BITS);

        // Position tied order base amount
        let tx_base_amount_is_zero = builder.is_zero(self.base_amount);
        let position_tied_order_base_amount = tx_state.positions[TAKER_ACCOUNT_ID]
            .calculate_position_tied_order_base_amount(
                builder,
                tx_state.market_details.quote_multiplier,
                self.price,
                tx_state.market.order_quote_limit,
            );
        self.calculated_base_amount = builder.select(
            tx_base_amount_is_zero,
            position_tied_order_base_amount,
            self.base_amount,
        );
        let tx_base_amount_is_zero = builder.is_zero(self.calculated_base_amount);
        self.success = builder.and_not(self.success, tx_base_amount_is_zero);

        let is_new_account = tx_state.is_new_account[OWNER_ACCOUNT_ID];
        self.success = builder.and_not(self.success, is_new_account);

        let is_master_account_correct = builder.is_equal(
            self.master_account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
        );
        self.success = builder.and(self.success, is_master_account_correct);

        let is_staking_pool = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );
        self.success = builder.and_not(self.success, is_staking_pool);

        // Active market
        let active_market_status = builder.constant_from_u8(MARKET_STATUS_ACTIVE);
        let is_order_book_active = builder.is_equal(tx_state.market.status, active_market_status);
        self.success = builder.and(self.success, is_order_book_active);

        // Spot is disallowed
        builder.conditional_assert_eq_constant(
            self.is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );

        // Oracle prices should be set for the market
        let is_index_price_non_zero = builder.is_not_zero(tx_state.market_details.index_price);
        let is_mark_price_non_zero = builder.is_not_zero(tx_state.market_details.mark_price);
        let is_price_oracle_set = builder.and(is_index_price_non_zero, is_mark_price_non_zero);
        self.success = builder.and(self.success, is_price_oracle_set);

        // Only allow order creation if market is not full, i.e. ask nonce < bid nonce, nonces are initially set so that ask nonce is smaller than bid nonce
        // since only the order creation can change one of the ask or bid nonces by exactly one, checking if orderBook.AskNonce != orderBook.BidNonce is enough
        let is_order_book_full =
            builder.is_equal(tx_state.market.ask_nonce, tx_state.market.bid_nonce);
        self.success = builder.and_not(self.success, is_order_book_full);

        // Is valid reduce only direction
        let is_not_valid_reduce_only_direction = is_not_valid_reduce_only_direction(
            builder,
            tx_state.positions[TAKER_ACCOUNT_ID].position.sign,
            self.is_ask,
        );
        self.success = builder.and_not(self.success, is_not_valid_reduce_only_direction);

        // Verify maximum quote amount
        let base_amount_big = builder.target_to_biguint(self.calculated_base_amount);
        let price_big = builder.target_to_biguint_single_limb_unsafe(self.price); // already range-checked to 32 bit
        let quote = builder.mul_biguint_non_carry(&base_amount_big, &price_big, BIG_U96_LIMBS);
        let order_book_quote_limit = builder.target_to_biguint(tx_state.market.order_quote_limit);
        let quote_lte_limit = builder.is_lte_biguint(&quote, &order_book_quote_limit);
        self.success = builder.and(self.success, quote_lte_limit);
    }
}

impl Apply for L1CreateOrderTxTarget {
    // order_before: If top_order is empty or there is no matching order for limit, order_before is empty and/or taker order
    // oterwise it is always the maker order
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        // Set new register
        let new_register = self.get_pending_order_register(builder, tx_state);
        tx_state.put_to_instruction_stack_unsafe(builder, self.success, &new_register, 0);

        // Set new market
        let is_ask_and_success = builder.and(self.success, self.is_ask);
        let is_bid_and_success = builder.and_not(self.success, self.is_ask);

        tx_state.market.ask_nonce =
            builder.add(tx_state.market.ask_nonce, is_ask_and_success.target);
        tx_state.market.bid_nonce =
            builder.sub(tx_state.market.bid_nonce, is_bid_and_success.target);

        // Set execute matching flag
        tx_state.matching_engine_flag = builder.or(tx_state.matching_engine_flag, self.success);
        // Set update impact prices flag
        tx_state.update_impact_prices_flag =
            builder.or(tx_state.update_impact_prices_flag, self.success);

        self.success
    }
}

impl PriorityOperationsPubData for L1CreateOrderTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let bit_count = [
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_CREATE_ORDER),
            add_target(builder, bytes, self.account_index, 48),
            add_target(builder, bytes, self.master_account_index, 48),
            add_target(builder, bytes, self.market_index, 16),
            add_target(builder, bytes, self.base_amount, 48),
            add_target(builder, bytes, self.price, 32),
            add_byte_target_unsafe(bytes, self.is_ask.target),
            add_byte_target_unsafe(bytes, self.order_type),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, bit_count),
        )
    }
}

pub trait L1CreateOrderTxTargetWitness<F: PrimeField64> {
    fn set_l1_create_order_tx_target(
        &mut self,
        a: &L1CreateOrderTxTarget,
        b: &L1CreateOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1CreateOrderTxTargetWitness<F> for T {
    fn set_l1_create_order_tx_target(
        &mut self,
        a: &L1CreateOrderTxTarget,
        b: &L1CreateOrderTx,
    ) -> Result<()> {
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )?;
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(a.base_amount, F::from_canonical_i64(b.base_amount))?;
        self.set_target(a.price, F::from_canonical_u32(b.price))?;
        self.set_bool_target(a.is_ask, b.is_ask == 1)?;
        self.set_target(a.order_type, F::from_canonical_u8(b.order_type))?;

        Ok(())
    }
}
