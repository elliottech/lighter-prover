// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::tx_interface::{Apply, Verify};
use crate::types::account::AccountTarget;
use crate::types::config::{BIG_U96_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct InternalPendingUnlockTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "asi")]
    pub asset_index: u16,
}

#[derive(Debug, Clone)]
pub struct InternalPendingUnlockTxTarget {
    pub account_index: Target,
    pub asset_index: Target,

    // Output
    success: BoolTarget,
}

impl InternalPendingUnlockTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            asset_index: builder.add_virtual_target(),

            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalPendingUnlockTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_pending_unlock;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            EXECUTE_TRANSACTION as u64,
        );

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        let next_pending_unlock = tx_state.accounts[OWNER_ACCOUNT_ID].pending_unlocks[0].clone();

        builder.conditional_assert_not_zero_biguint(is_enabled, &next_pending_unlock.amount);
        builder.conditional_assert_eq(
            is_enabled,
            next_pending_unlock.asset_index,
            self.asset_index,
        );
        builder.conditional_assert_lte(
            is_enabled,
            next_pending_unlock.unlock_timestamp,
            tx_state.block_timestamp,
            TIMESTAMP_BITS,
        );

        let new_extended_balance = builder.add_biguint(
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &next_pending_unlock.amount,
        );
        let (success, _) = builder.try_trim_biguint(&new_extended_balance, BIG_U96_LIMBS);
        builder.conditional_assert_true(is_enabled, success);
    }
}

impl Apply for InternalPendingUnlockTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let pending_unlock =
            tx_state.accounts[OWNER_ACCOUNT_ID].pop_pending_unlock(builder, self.success);
        let asset_amount = builder.biguint_to_bigint(&pending_unlock.amount);

        AccountTarget::apply_asset_delta_const(
            builder,
            self.success,
            PRODUCT_TYPE_SPOT,
            &mut tx_state.accounts[OWNER_ACCOUNT_ID],
            Some(&mut tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID]),
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            &asset_amount,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
        );

        self.success
    }
}

pub trait InternalPendingUnlockTxTargetWitness<F: PrimeField64> {
    fn set_internal_pending_unlock_tx_target(
        &mut self,
        a: &InternalPendingUnlockTxTarget,
        b: &InternalPendingUnlockTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalPendingUnlockTxTargetWitness<F> for T {
    fn set_internal_pending_unlock_tx_target(
        &mut self,
        a: &InternalPendingUnlockTxTarget,
        b: &InternalPendingUnlockTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index))?;

        Ok(())
    }
}
