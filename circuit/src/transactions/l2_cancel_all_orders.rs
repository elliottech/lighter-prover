// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::tx_utils::apply_immediate_cancel_all;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::transactions::tx_utils::apply_immediate_cancel_all_market;
use crate::tx_attributes::{ATTR_CANCEL_ALL_MARKET_INDEX, ATTR_NIL_VALUES};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CancelAllOrdersTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "tf")]
    pub time_in_force: u8,

    #[serde(rename = "t")]
    pub time: i64, // 48 bits
}

#[derive(Debug)]
pub struct L2CancelAllOrdersTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub time_in_force: Target,
    pub time: Target, // 48 bits

    // output
    pub success: BoolTarget,
}

impl L2CancelAllOrdersTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2CancelAllOrdersTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            time_in_force: builder.add_virtual_target(),
            time: builder.add_virtual_target(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2CancelAllOrdersTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CANCEL_ALL_ORDERS)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.time_in_force,
            self.time,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CancelAllOrdersTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_cancel_all_orders;
        self.success = is_enabled;

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

        let is_immediate_cancel_all =
            builder.is_equal_constant(self.time_in_force, IMMEDIATE_CANCEL_ALL as u64);
        let is_scheduled_cancel_all =
            builder.is_equal_constant(self.time_in_force, SCHEDULED_CANCEL_ALL as u64);
        let is_abort_scheduled_cancel_all =
            builder.is_equal_constant(self.time_in_force, ABORT_SCHEDULED_CANCEL_ALL as u64);
        let is_valid_time_in_force = builder.multi_or(&[
            is_immediate_cancel_all,
            is_scheduled_cancel_all,
            is_abort_scheduled_cancel_all,
        ]);
        builder.conditional_assert_true(is_enabled, is_valid_time_in_force);

        builder.register_range_check(self.time, TIMESTAMP_BITS);

        let is_not_schedule = builder.not(is_scheduled_cancel_all);
        builder.conditional_assert_zero(is_not_schedule, self.time);

        /* Cancel all market index and time in force checks */
        /* (Cancel all market index can't be scheduled)     */

        let market_index = tx_state.attributes.get(ATTR_CANCEL_ALL_MARKET_INDEX);

        let is_market_index_nil =
            builder.is_equal_f(market_index, ATTR_NIL_VALUES[ATTR_CANCEL_ALL_MARKET_INDEX]);

        let is_market_index_enabled = builder.and_not(is_enabled, is_market_index_nil);

        builder.conditional_assert_true(is_market_index_enabled, is_immediate_cancel_all);

        builder.conditional_assert_eq(
            is_market_index_enabled,
            tx_state.market.market_index,
            market_index,
        );
    }
}

impl Apply for L2CancelAllOrdersTxTarget {
    fn apply(&mut self, builder: &mut Builder, state: &mut TxState) -> BoolTarget {
        let is_immediate_cancel_all =
            builder.is_equal_constant(self.time_in_force, IMMEDIATE_CANCEL_ALL as u64);
        let is_scheduled_cancel_all =
            builder.is_equal_constant(self.time_in_force, SCHEDULED_CANCEL_ALL as u64);
        let is_abort_scheduled_cancel_all =
            builder.is_equal_constant(self.time_in_force, ABORT_SCHEDULED_CANCEL_ALL as u64);

        let market_index = state.attributes.get(ATTR_CANCEL_ALL_MARKET_INDEX);

        let is_market_index_nil =
            builder.is_equal_f(market_index, ATTR_NIL_VALUES[ATTR_CANCEL_ALL_MARKET_INDEX]);

        let is_market_index_enabled = builder.and_not(self.success, is_market_index_nil);

        // Immediate Cancel All
        let is_time_in_force_immediate_active = builder.and(is_immediate_cancel_all, self.success);
        let is_cancel_all =
            builder.and_not(is_time_in_force_immediate_active, is_market_index_enabled);
        apply_immediate_cancel_all(builder, is_cancel_all, state, self.account_index);

        // Scheduled Cancel All
        let is_time_in_force_scheduled_active = builder.and(is_scheduled_cancel_all, self.success);
        state.accounts[OWNER_ACCOUNT_ID].cancel_all_time = builder.select(
            is_time_in_force_scheduled_active,
            self.time,
            state.accounts[OWNER_ACCOUNT_ID].cancel_all_time,
        );

        // Abort Scheduled Cancel All
        let is_time_in_force_abort_scheduled_active =
            builder.and(is_abort_scheduled_cancel_all, self.success);
        let zero = builder.zero();
        state.accounts[OWNER_ACCOUNT_ID].cancel_all_time = builder.select(
            is_time_in_force_abort_scheduled_active,
            zero,
            state.accounts[OWNER_ACCOUNT_ID].cancel_all_time,
        );

        // Immediate Cancel All for Market Index
        apply_immediate_cancel_all_market(
            builder,
            is_market_index_enabled,
            state,
            self.account_index,
            market_index,
        );

        self.success
    }
}

pub trait L2CancelAllOrdersTxTargetWitness<F: PrimeField64> {
    fn set_l2_cancel_all_orders_tx_target(
        &mut self,
        a: &L2CancelAllOrdersTxTarget,
        b: &L2CancelAllOrdersTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CancelAllOrdersTxTargetWitness<F> for T {
    fn set_l2_cancel_all_orders_tx_target(
        &mut self,
        a: &L2CancelAllOrdersTxTarget,
        b: &L2CancelAllOrdersTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.time_in_force, F::from_canonical_u8(b.time_in_force))?;
        self.set_target(a.time, F::from_canonical_i64(b.time))?;

        Ok(())
    }
}
