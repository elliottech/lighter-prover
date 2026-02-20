// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{
    get_available_asset_balance_const, get_shares_asset_value_for_staking_pool,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2StakeAssetsTx {
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
pub struct L2StakeAssetsTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub staking_pool_index: Target,
    pub share_amount: Target,

    // Helper
    is_operator: BoolTarget,
    lit_amount: BigUintTarget,
    new_total_shares: Target,
    new_principal_amount: Target,
    balance_to_mint_shares: BigUintTarget,

    // Output
    success: BoolTarget,
}

impl L2StakeAssetsTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2StakeAssetsTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            staking_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),

            // Helper
            is_operator: builder._false(),
            lit_amount: builder.zero_biguint(),
            new_total_shares: builder.zero(),
            new_principal_amount: builder.zero(),
            balance_to_mint_shares: builder.zero_biguint(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2StakeAssetsTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_STAKE_ASSETS)),
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

impl Verify for L2StakeAssetsTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_stake_assets;
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

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        // Assert sub account type
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );

        // Assert public pool is active
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info.status,
            ACTIVE_PUBLIC_POOL as u64,
        );

        // If the share amount is greater than the maximum pool shares, fail the transaction
        self.new_total_shares = builder.add(
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            self.share_amount,
        );
        let big_new_total_shares = builder.target_to_biguint(self.new_total_shares);
        builder.range_check_biguint(&big_new_total_shares, MAX_POOL_SHARES_BITS);

        self.lit_amount = get_shares_asset_value_for_staking_pool(
            builder,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            // Because LIT can't be used as margin, we can use asset balance directly without considering unified accounts
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            self.share_amount,
        );
        builder.range_check_biguint(&self.lit_amount, MAX_POOL_SHARES_TO_MINT_OR_BURN_LIT_BITS);
        (_, self.lit_amount) = builder.try_trim_biguint(&self.lit_amount, BIG_U64_LIMBS);

        // Check if the entry asset amount fits in the pool share entry asset limit
        let lit_amount_target = builder.biguint_to_target_safe(&self.lit_amount);
        self.new_principal_amount = builder.add(
            tx_state.public_pool_share.principal_amount,
            lit_amount_target,
        );
        builder.register_range_check(self.new_principal_amount, MAX_POOL_PRINCIPAL_AMOUNT_BITS);

        self.balance_to_mint_shares = builder.mul_biguint_non_carry(
            &self.lit_amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );
        let asset_balance = get_available_asset_balance_const(
            builder,
            PRODUCT_TYPE_SPOT,
            &tx_state.accounts[OWNER_ACCOUNT_ID],
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
        );
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &self.balance_to_mint_shares,
            &asset_balance,
        );

        self.is_operator = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );

        // If staker is not the operator, then check if the minimum share rate is still
        // going to be satisfied for the pool operator
        {
            // If operator shares drop below the minimum operator share rate, fail the transaction
            let big_min_operator_share_rate = builder.target_to_biguint(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .min_operator_share_rate,
            );
            let big_operator_shares = builder.target_to_biguint(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
            );
            let big_share_tick = builder.constant_biguint(&BigUint::from(SHARE_TICK));
            let lhs = builder.mul_biguint(&big_new_total_shares, &big_min_operator_share_rate);
            let rhs = builder.mul_biguint(&big_operator_shares, &big_share_tick);
            let not_operator_and_enabled = builder.and_not(is_enabled, self.is_operator);
            builder.conditional_assert_lte_biguint(not_operator_and_enabled, &lhs, &rhs);
        }
    }
}

impl Apply for L2StakeAssetsTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        // Asset Balance deltas - Because LIT asset can't be used as margin, we don't need to handle unified accounts here
        let (new_owner_balance, fail) = builder.try_sub_biguint(
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &self.balance_to_mint_shares,
        );
        builder.conditional_assert_zero_u32(self.success, fail);
        tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &new_owner_balance,
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        let new_sub_account_balance = builder.add_biguint_non_carry(
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
            &self.balance_to_mint_shares,
            BIG_U96_LIMBS,
        );
        tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &new_sub_account_balance,
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        // Public pool total share
        tx_state.accounts[SUB_ACCOUNT_ID]
            .public_pool_info
            .total_shares = builder.select(
            self.success,
            self.new_total_shares,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
        );

        // Set pool assets - not operator
        {
            let is_success_and_not_operator = builder.and_not(self.success, self.is_operator);

            let new_share_amount =
                builder.add(tx_state.public_pool_share.share_amount, self.share_amount);
            tx_state.public_pool_share.principal_amount = builder.select(
                is_success_and_not_operator,
                self.new_principal_amount,
                tx_state.public_pool_share.principal_amount,
            );
            tx_state.public_pool_share.share_amount = builder.select(
                is_success_and_not_operator,
                new_share_amount,
                tx_state.public_pool_share.share_amount,
            );

            tx_state.apply_pool_share_delta_flag = builder.or(
                tx_state.apply_pool_share_delta_flag,
                is_success_and_not_operator,
            );
        }
        // Set pool assets - is operator
        {
            let is_success_and_operator = builder.and(self.success, self.is_operator);
            let new_operator_assets_for_operator = builder.add(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
                self.share_amount,
            );
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares = builder.select(
                is_success_and_operator,
                new_operator_assets_for_operator,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
            );
        }

        self.success
    }
}

pub trait L2StakeAssetsTxTargetWitness<F: PrimeField64> {
    fn set_l2_stake_assets_tx_target(
        &mut self,
        a: &L2StakeAssetsTxTarget,
        b: &L2StakeAssetsTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2StakeAssetsTxTargetWitness<F> for T {
    fn set_l2_stake_assets_tx_target(
        &mut self,
        a: &L2StakeAssetsTxTarget,
        b: &L2StakeAssetsTx,
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
