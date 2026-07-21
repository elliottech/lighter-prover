// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CreateSubAccountTx {
    #[serde(rename = "ai")]
    pub account_index: i64, // 48 bits

    #[serde(rename = "ki")]
    pub api_key_index: u8,
}

#[derive(Debug, Clone)]
pub struct L2CreateSubAccountTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits

    // output
    pub success: BoolTarget,
}

impl L2CreateSubAccountTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2CreateSubAccountTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2CreateSubAccountTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_SUB_ACCOUNT)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CreateSubAccountTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_create_sub_account;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[MASTER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );

        // Ensure the sender account is a master account
        let max_master_account_index = builder.constant_i64(MAX_MASTER_ACCOUNT_INDEX);
        builder.conditional_assert_lte(
            is_enabled,
            self.account_index,
            max_master_account_index,
            ACCOUNT_INDEX_BITS,
        );

        let min_sub_account_index = builder.constant_i64(MIN_SUB_ACCOUNT_INDEX);
        builder.conditional_assert_lte(
            is_enabled,
            min_sub_account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
            ACCOUNT_INDEX_BITS,
        );

        // Verify that given sub-account is empty before
        let is_new_account = tx_state.is_new_account[SUB_ACCOUNT_ID];
        builder.conditional_assert_true(is_enabled, is_new_account);

        // nil account index is reserved and always should be empty
        let nil_account_index = builder.constant_i64(NIL_ACCOUNT_INDEX);
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
            nil_account_index,
        );
    }
}

impl Apply for L2CreateSubAccountTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let sub_account_type = builder.constant(F::from_canonical_u8(SUB_ACCOUNT_TYPE));
        tx_state.accounts[SUB_ACCOUNT_ID].account_type = builder.select(
            self.success,
            sub_account_type,
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
        );
        tx_state.accounts[SUB_ACCOUNT_ID].l1_address = builder.select_biguint(
            self.success,
            &tx_state.accounts[MASTER_ACCOUNT_ID].l1_address,
            &tx_state.accounts[SUB_ACCOUNT_ID].l1_address,
        );
        tx_state.accounts[SUB_ACCOUNT_ID].master_account_index = builder.select(
            self.success,
            self.account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );
        let unified_trading_mode =
            builder.constant(F::from_canonical_u8(ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED));
        tx_state.accounts[SUB_ACCOUNT_ID].account_trading_mode = builder.select(
            self.success,
            unified_trading_mode,
            tx_state.accounts[SUB_ACCOUNT_ID].account_trading_mode,
        );

        self.success
    }
}

pub trait L2CreateSubAccountTxTargetWitness<F: PrimeField64> {
    fn set_l2_create_sub_account_tx_target(
        &mut self,
        a: &L2CreateSubAccountTxTarget,
        b: &L2CreateSubAccountTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CreateSubAccountTxTargetWitness<F> for T {
    fn set_l2_create_sub_account_tx_target(
        &mut self,
        a: &L2CreateSubAccountTxTarget,
        b: &L2CreateSubAccountTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;

        Ok(())
    }
}
