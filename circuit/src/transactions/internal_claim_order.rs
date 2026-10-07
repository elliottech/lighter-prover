// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::BoolTarget;
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::tx_interface::{Apply, Verify};
use crate::types::config::Builder;
use crate::types::constants::{INSERT_ORDER, OWNER_ACCOUNT_ID};
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
