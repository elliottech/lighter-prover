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
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

/// Lets the insurance fund operator assign the market operator of a binary options market slot. The slot
/// may host an active market, a settled market or no market at all; the operator is a property of the
/// slot and survives settlement. It authorizes L2CreateMarket and L2UpdateMarket on the slot.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UpdateMarketSlotTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "mi")]
    pub market_index: i16,
    #[serde(rename = "ma")]
    pub market_operator_account_index: i64,
}

#[derive(Debug)]
pub struct L2UpdateMarketSlotTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub market_index: Target,
    pub market_operator_account_index: Target,

    // Output
    success: BoolTarget,
}

impl L2UpdateMarketSlotTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            market_operator_account_index: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateMarketSlotTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_MARKET_SLOT)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.market_operator_account_index,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateMarketSlotTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_market_slot;
        self.success = is_enabled;

        // Only the insurance fund operator can assign slot operators
        builder.conditional_assert_eq_constant(
            is_enabled,
            self.account_index,
            INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX as u64,
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

        // The new operator must be an existing master or sub account
        builder.conditional_assert_eq(
            is_enabled,
            self.market_operator_account_index,
            tx_state.accounts[MARKET_OPERATOR_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_false(
            is_enabled,
            tx_state.is_new_account[MARKET_OPERATOR_ACCOUNT_ID],
        );
        let nil_account_index = builder.constant_i64(NIL_ACCOUNT_INDEX);
        builder.conditional_assert_not_eq(
            is_enabled,
            self.market_operator_account_index,
            nil_account_index,
        );
        let is_master = builder.is_equal_constant(
            tx_state.accounts[MARKET_OPERATOR_ACCOUNT_ID].account_type,
            MASTER_ACCOUNT_TYPE as u64,
        );
        let is_sub = builder.is_equal_constant(
            tx_state.accounts[MARKET_OPERATOR_ACCOUNT_ID].account_type,
            SUB_ACCOUNT_TYPE as u64,
        );
        let is_master_or_sub = builder.or(is_master, is_sub);
        builder.conditional_assert_true(is_enabled, is_master_or_sub);

        // Whatever the binary options slot holds (an active market, a settled one or nothing) it may
        // be assigned an operator
        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.market_index, nil_market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.binary_options_market_index,
        );
    }
}

impl Apply for L2UpdateMarketSlotTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.market.market_operator_account_index = builder.select(
            self.success,
            self.market_operator_account_index,
            tx_state.market.market_operator_account_index,
        );

        self.success
    }
}

pub trait L2UpdateMarketSlotTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_market_slot_tx_target(
        &mut self,
        a: &L2UpdateMarketSlotTxTarget,
        b: &L2UpdateMarketSlotTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateMarketSlotTxTargetWitness<F> for T {
    fn set_l2_update_market_slot_tx_target(
        &mut self,
        a: &L2UpdateMarketSlotTxTarget,
        b: &L2UpdateMarketSlotTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;
        self.set_target(
            a.market_operator_account_index,
            F::from_canonical_i64(b.market_operator_account_index),
        )?;

        Ok(())
    }
}
