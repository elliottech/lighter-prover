// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{get_available_shares_to_burn_for_staking_pool, get_shares_asset_value};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UnstakeAssetsTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "p", default)]
    pub staking_pool_index: i64,

    #[serde(rename = "s", default)]
    pub share_amount: i64,
}

#[derive(Debug)]
pub struct L2UnstakeAssetsTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub staking_pool_index: Target,
    pub share_amount: Target,

    // Helper
    pub is_operator: BoolTarget,
    pub account_shares: Target,

    pub shares_to_burn_lit: Target,
    pub old_principal_amount: Target,

    // Output
    pub success: BoolTarget,
}

impl L2UnstakeAssetsTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UnstakeAssetsTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            staking_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),

            // Helper
            is_operator: builder._false(),
            account_shares: builder.zero(),

            shares_to_burn_lit: builder.zero(),
            old_principal_amount: builder.zero(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UnstakeAssetsTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UNSTAKE_ASSETS)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.staking_pool_index,
            self.share_amount,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UnstakeAssetsTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_unstake_assets;
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
        builder.conditional_assert_eq(
            is_enabled,
            self.staking_pool_index,
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
        );

        // Limit to lit
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            LIT_ASSET_INDEX,
        );

        // Assert share amount is within bounds
        builder.register_range_check(self.share_amount, MAX_STAKING_SHARES_TO_MINT_OR_BURN_BITS);
        builder.conditional_assert_not_zero(is_enabled, self.share_amount);

        // Assert sub account type
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );

        self.is_operator = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );
        self.account_shares = builder.select(
            self.is_operator,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares,
            tx_state.public_pool_share.share_amount,
        );
        builder.conditional_assert_lte(is_enabled, self.share_amount, self.account_shares, 64);

        builder.conditional_assert_lte(
            is_enabled,
            self.share_amount,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            64,
        );

        self.old_principal_amount = tx_state.public_pool_share.principal_amount; // To be used in apply

        let available_shares_to_burn = get_available_shares_to_burn_for_staking_pool(
            builder,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID],
        );
        builder.conditional_assert_lte(is_enabled, self.share_amount, available_shares_to_burn, 64);

        self.shares_to_burn_lit = get_shares_asset_value(
            builder,
            &tx_state.accounts[SUB_ACCOUNT_ID],
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            self.share_amount,
        );
        builder.register_range_check(
            self.shares_to_burn_lit,
            MAX_POOL_SHARES_TO_MINT_OR_BURN_ASSET_BITS,
        );

        {
            let frozen_public_pool = builder.constant_from_u8(FROZEN_PUBLIC_POOL);
            let is_frozen_public_pool = builder.is_equal(
                tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info.status,
                frozen_public_pool,
            );
            let is_not_frozen_and_owner = builder.and_not(self.is_operator, is_frozen_public_pool);

            let new_total_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
                self.share_amount,
            );
            let big_new_total_shares = builder.target_to_biguint(new_total_shares);
            let big_min_operator_share_rate = builder.target_to_biguint(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .min_operator_share_rate,
            );
            let new_operator_shares = builder.sub(self.account_shares, self.share_amount);
            let big_new_operator_shares = builder.target_to_biguint(new_operator_shares);
            let big_share_tick = builder.constant_biguint(&BigUint::from(SHARE_TICK));
            let lhs = builder.mul_biguint(&big_new_total_shares, &big_min_operator_share_rate);
            let rhs = builder.mul_biguint(&big_new_operator_shares, &big_share_tick);

            let check_lhs_lte_rhs = builder.and(is_enabled, is_not_frozen_and_owner);
            builder.conditional_assert_lte_biguint(check_lhs_lte_rhs, &lhs, &rhs);
        }
    }
}

impl Apply for L2UnstakeAssetsTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let big_asset_amount = builder.target_to_biguint(self.shares_to_burn_lit);
        let big_asset_extended = builder.mul_biguint_non_carry(
            &big_asset_amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );

        let new_owner_balance = builder.add_biguint_non_carry(
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &big_asset_extended,
            BIG_U96_LIMBS,
        );
        tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &new_owner_balance,
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        let (new_sub_account_balance, fail) = builder.try_sub_biguint(
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
            &big_asset_extended,
        );
        builder.conditional_assert_zero_u32(self.success, fail);
        tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &new_sub_account_balance,
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        let new_total_shares = builder.sub(
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            self.share_amount,
        );
        tx_state.accounts[SUB_ACCOUNT_ID]
            .public_pool_info
            .total_shares = builder.select(
            self.success,
            new_total_shares,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
        );

        // If is operator
        {
            let op_success = builder.and(self.success, self.is_operator);
            let new_operator_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
                self.share_amount,
            );
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares = builder.select(
                op_success,
                new_operator_shares,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
            );
        }

        // Updates for if enabled and unstaking is not the account operator
        {
            let non_operator_success = builder.and_not(self.success, self.is_operator);

            let big_principal_amount = builder.target_to_biguint(self.old_principal_amount);
            let big_share_amount = builder.target_to_biguint(self.share_amount);
            let big_account_shares = builder.target_to_biguint(self.account_shares);
            let dividend = builder.mul_biguint(&big_principal_amount, &big_share_amount);
            let big_principal_amount_delta = builder.div_biguint(&dividend, &big_account_shares);
            let principal_amount_delta =
                builder.biguint_to_target_unsafe(&big_principal_amount_delta);

            let new_principal_amount = builder.sub(
                tx_state.public_pool_share.principal_amount,
                principal_amount_delta,
            );
            tx_state.public_pool_share.principal_amount = builder.select(
                non_operator_success,
                new_principal_amount,
                tx_state.public_pool_share.principal_amount,
            );
            let new_share_amount =
                builder.sub(tx_state.public_pool_share.share_amount, self.share_amount);
            tx_state.public_pool_share.share_amount = builder.select(
                non_operator_success,
                new_share_amount,
                tx_state.public_pool_share.share_amount,
            );
            tx_state.apply_pool_share_delta_flag =
                builder.or(tx_state.apply_pool_share_delta_flag, non_operator_success);
        }

        self.success
    }
}

pub trait L2UnstakeAssetsTxTargetWitness<F: PrimeField64> {
    fn set_l2_unstake_assets_tx_target(
        &mut self,
        a: &L2UnstakeAssetsTxTarget,
        b: &L2UnstakeAssetsTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UnstakeAssetsTxTargetWitness<F> for T {
    fn set_l2_unstake_assets_tx_target(
        &mut self,
        a: &L2UnstakeAssetsTxTarget,
        b: &L2UnstakeAssetsTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.staking_pool_index,
            F::from_canonical_i64(b.staking_pool_index),
        )?;
        self.set_target(a.share_amount, F::from_canonical_i64(b.share_amount))?;
        Ok(())
    }
}
