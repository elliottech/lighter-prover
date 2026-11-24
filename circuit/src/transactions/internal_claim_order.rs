// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::BoolTarget;
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::tx_interface::{Apply, Verify};
use crate::types::config::Builder;
use crate::types::constants::{
    BASE_ASSET_ID, INSERT_ORDER, MARKET_TYPE_BINARY_OPTIONS, MARKET_TYPE_PERPS, NIL_ASSET_INDEX,
    OWNER_ACCOUNT_ID, QUOTE_ASSET_ID, TX_ASSET_ID, USDC_ASSET_INDEX, USDC_BASE_ASSET_ID,
};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InternalClaimOrderTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "mi")]
    pub market_index: u16,
}

impl Default for InternalClaimOrderTx {
    fn default() -> Self {
        InternalClaimOrderTx::empty()
    }
}

impl InternalClaimOrderTx {
    pub fn empty() -> Self {
        InternalClaimOrderTx {
            account_index: 0,
            market_index: 0,
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct InternalClaimOrderTxTarget {
    // outputs
    pub success: BoolTarget,
}

impl InternalClaimOrderTxTarget {
    pub fn new(_builder: &mut Builder) -> Self {
        InternalClaimOrderTxTarget {
            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalClaimOrderTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_claim_order;

        let insert_order_type = builder.constant_from_u8(INSERT_ORDER);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            insert_order_type,
        );

        builder.conditional_assert_eq(
            is_enabled,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.register_stack[0].account_index,
        );

        builder.conditional_assert_eq(
            is_enabled,
            tx_state.market.market_index,
            tx_state.register_stack[0].market_index,
        );
        let is_perps_market =
            builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_PERPS);
        let is_binary_options_market =
            builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_BINARY_OPTIONS);
        let perps_flag = builder.and(is_enabled, is_perps_market);
        let non_perps_flag = builder.and_not(is_enabled, is_perps_market);
        let spot_flag = builder.and_not(non_perps_flag, is_binary_options_market);
        let bo_flag = builder.and(is_enabled, is_binary_options_market);

        builder.conditional_assert_eq_constant(
            perps_flag,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );
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
        builder.conditional_assert_eq(
            bo_flag,
            tx_state.market.quote_asset_id,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
        );
        builder.conditional_assert_eq_constant(
            bo_flag,
            tx_state.asset_indices[QUOTE_ASSET_ID],
            NIL_ASSET_INDEX,
        );

        self.success = is_enabled;
    }
}

impl Apply for InternalClaimOrderTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.matching_engine_flag = builder.or(self.success, tx_state.matching_engine_flag);
        tx_state.update_impact_prices_flag =
            builder.or(self.success, tx_state.update_impact_prices_flag);

        self.success
    }
}

pub trait InternalClaimOrderTxTargetWitness<F: PrimeField64> {
    fn set_internal_claim_order_tx_target(
        &mut self,
        a: &InternalClaimOrderTxTarget,
        b: &InternalClaimOrderTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalClaimOrderTxTargetWitness<F> for T {
    fn set_internal_claim_order_tx_target(
        &mut self,
        _a: &InternalClaimOrderTxTarget,
        _b: &InternalClaimOrderTx,
    ) -> Result<()> {
        // no-op
        Ok(())
    }
}
