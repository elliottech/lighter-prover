// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::tx_utils::{
    apply_cross_cancel_all, apply_immediate_cancel_all, apply_isolated_cancel_all,
};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::tx_interface::{Apply, Verify};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct InternalCancelAllOrdersTx {
    #[serde(rename = "i")]
    pub account_index: i64,

    #[serde(rename = "mi")]
    pub market_index: u16,
}

#[derive(Debug)]
pub struct InternalCancelAllOrdersTxTarget {
    pub account_index: Target, // 48 bits
    pub market_index: Target,  // 8 bits, perps only

    // helpers
    pub is_liquidation: BoolTarget,
    pub is_dms: BoolTarget,

    // outputs
    pub success: BoolTarget,
}

impl InternalCancelAllOrdersTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        InternalCancelAllOrdersTxTarget {
            account_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),

            // helpers
            is_liquidation: BoolTarget::default(),
            is_dms: BoolTarget::default(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalCancelAllOrdersTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_cancel_all_orders;
        self.success = is_enabled;

        // account index
        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        // market index - only allow perps market to be specified, if isolated position needs liquidation. Otherwise, should load NIL market.
        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );

        // Verify register instruction type
        let execute_transaction = builder.constant_from_u8(EXECUTE_TRANSACTION);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            execute_transaction,
        );

        let is_not_in_liquidation = tx_state.risk_infos[OWNER_ACCOUNT_ID]
            .current_risk_parameters
            .is_not_in_liquidation(builder);
        self.is_liquidation = builder.and_not(is_enabled, is_not_in_liquidation);
        let is_dms = tx_state.accounts[OWNER_ACCOUNT_ID]
            .should_dms_be_triggered(builder, tx_state.block_timestamp);
        self.is_dms = builder.and(is_enabled, is_dms);

        let is_dms_or_liquidation = builder.or(self.is_dms, self.is_liquidation);
        builder.conditional_assert_true(is_enabled, is_dms_or_liquidation);

        // Give priority to DMS over liquidation
        self.is_liquidation = builder.and_not(self.is_liquidation, self.is_dms);

        // If it's not DMS and in liquidation, can't be insurance fund account
        let is_insurance_fund_account = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_liquidation_and_insurance_fund =
            builder.and(self.is_liquidation, is_insurance_fund_account);
        builder.conditional_assert_false(is_enabled, is_liquidation_and_insurance_fund);
    }
}

impl Apply for InternalCancelAllOrdersTxTarget {
    fn apply(&mut self, builder: &mut Builder, state: &mut TxState) -> BoolTarget {
        apply_immediate_cancel_all(builder, self.is_dms, state, self.account_index);

        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
        let is_position_isolated = builder.is_equal(
            state.positions[OWNER_ACCOUNT_ID].margin_mode,
            isolated_margin_mode,
        );
        let is_position_cross = builder.not(is_position_isolated);

        let is_liquidation_and_isolated = builder.and(self.is_liquidation, is_position_isolated);
        let is_liquidation_and_cross = builder.and(self.is_liquidation, is_position_cross);

        apply_isolated_cancel_all(
            builder,
            is_liquidation_and_isolated,
            state,
            self.account_index,
            self.market_index,
        );
        apply_cross_cancel_all(builder, is_liquidation_and_cross, state, self.account_index);

        self.success
    }
}

pub trait InternalCancelAllOrdersTxTargetWitness<F: PrimeField64> {
    fn set_internal_cancel_all_orders_tx_target(
        &mut self,
        a: &InternalCancelAllOrdersTxTarget,
        b: &InternalCancelAllOrdersTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalCancelAllOrdersTxTargetWitness<F> for T {
    fn set_internal_cancel_all_orders_tx_target(
        &mut self,
        a: &InternalCancelAllOrdersTxTarget,
        b: &InternalCancelAllOrdersTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))
    }
}
