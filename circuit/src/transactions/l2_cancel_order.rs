// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::matching_engine::{
    cancel_child_orders, decrement_locked_balance_for_order, decrement_order_count_in_place,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::account_order::{AccountOrderTarget, select_account_order_target};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::order::{
    OrderTarget, get_market_index_and_order_nonce_from_order_index, select_order_target,
};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CancelOrderTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "m")]
    pub market_index: u16,

    #[serde(rename = "i")]
    pub index: i64, // cloindex or oindex
}

#[derive(Debug, Clone)]
pub struct L2CancelOrderTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits
    pub market_index: Target,  // 8 bits
    pub index: Target,         // 56 bits - cloindex or oindex

    // Output
    pub success: BoolTarget,
}

impl L2CancelOrderTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2CancelOrderTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            index: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2CancelOrderTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CANCEL_ORDER)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.index,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CancelOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_cancel_order;
        self.success = tx_type.is_l2_cancel_order;

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
        let is_order_book_active =
            builder.is_equal_constant(tx_state.market.status, MARKET_STATUS_ACTIVE as u64);
        builder.conditional_assert_true(self.success, is_order_book_active);

        // Verify that we load the correct order from orderbook
        builder.conditional_assert_eq(
            self.success,
            tx_state.order.nonce_index,
            tx_state.account_order.nonce,
        );
        let is_trigger_status_na = builder.is_equal_constant(
            tx_state.account_order.trigger_status,
            TRIGGER_STATUS_NA as u64,
        );
        let success_and_trigger_status_na = builder.and(self.success, is_trigger_status_na);
        builder.conditional_assert_eq(
            success_and_trigger_status_na,
            tx_state.account_order.price,
            tx_state.order.price_index,
        );
    }
}

impl Apply for L2CancelOrderTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        decrement_order_count_in_place(
            builder,
            tx_state,
            OWNER_ACCOUNT_ID,
            self.success,
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

        cancel_child_orders(
            builder,
            self.success,
            tx_state,
            self.market_index,
            tx_state.account_order.owner_account_index,
            tx_state.account_order.to_trigger_order_index0,
            tx_state.account_order.to_trigger_order_index1,
            1,
        );

        // If the order's trigger status is not NA, it's nonce is zero and all orders with zero nonce are always empty.
        let empty_order = OrderTarget::empty(
            builder,
            tx_state.order.price_index,
            tx_state.order.nonce_index,
        );
        tx_state.order = select_order_target(builder, self.success, &empty_order, &tx_state.order);

        let empty_account_order = AccountOrderTarget::empty(
            builder,
            tx_state.account_order.index_0,
            tx_state.account_order.index_1,
            tx_state.account_order.owner_account_index,
        );
        tx_state.account_order = select_account_order_target(
            builder,
            self.success,
            &empty_account_order,
            &tx_state.account_order,
        );

        tx_state.update_impact_prices_flag =
            builder.or(self.success, tx_state.update_impact_prices_flag);

        self.success
    }
}

pub trait L2CancelOrderTxTargetWitness<F: PrimeField64> {
    fn set_l2_cancel_order_tx_target(
        &mut self,
        a: &L2CancelOrderTxTarget,
        b: &L2CancelOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CancelOrderTxTargetWitness<F> for T {
    fn set_l2_cancel_order_tx_target(
        &mut self,
        a: &L2CancelOrderTxTarget,
        b: &L2CancelOrderTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.index, F::from_canonical_i64(b.index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;

        Ok(())
    }
}
