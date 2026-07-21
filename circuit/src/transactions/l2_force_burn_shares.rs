// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{
    get_available_shares_to_burn_for_public_pool, get_shares_asset_value_for_staking_pool,
    get_shares_usdc_value_for_public_pool,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2ForceBurnSharesTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "ki", default)]
    pub api_key_index: u8,
    #[serde(rename = "di", default)]
    pub depositor_index: i64,
    #[serde(rename = "s", default)]
    pub share_amount: i64,
}

#[derive(Debug)]
pub struct L2ForceBurnSharesTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub depositor_index: Target,
    pub share_amount: Target,

    // Helper
    account_shares: Target,
    new_principal_amount: Target,
    shares_to_burn: Target,
    shares_to_burn_usdc_value: Target,

    operator_fee_share: Target,

    // Output
    success: BoolTarget,
}

impl L2ForceBurnSharesTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2ForceBurnSharesTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            depositor_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),

            // Helper
            account_shares: builder.zero(),
            new_principal_amount: builder.zero(),
            shares_to_burn: builder.zero(),
            shares_to_burn_usdc_value: builder.zero(),

            operator_fee_share: builder.zero(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2ForceBurnSharesTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_FORCE_BURN_SHARES)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.depositor_index,
            self.share_amount,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2ForceBurnSharesTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_force_burn_shares;
        self.success = is_enabled;

        // Tx sender is in the sub account slot
        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.depositor_index,
            tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID].account_index,
        );

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.system_config.liquidity_pool_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].account_index,
            tx_state.system_config.staking_pool_index,
        );

        // Staking pool should exists. l1_set_system_config ensures that account exists if index is not nil.
        let is_staking_pool_nil_account = builder.is_equal_constant(
            tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].account_index,
            NIL_ACCOUNT_INDEX as u64,
        );
        builder.conditional_assert_false(is_enabled, is_staking_pool_nil_account);

        // Assets: USDC, LIT
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
            USDC_ASSET_INDEX,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[STAKE_ASSET_ID],
            LIT_ASSET_INDEX,
        );

        let is_asset_empty = tx_state.assets[STAKE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        // Operator can't be forced
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID].master_account_index,
            tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID].account_index,
        );

        // Range check share amount
        let big_shares_amount = builder.target_to_biguint(self.share_amount);
        builder.range_check_biguint(
            &big_shares_amount,
            MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS,
        );
        builder.conditional_assert_not_zero(is_enabled, self.share_amount);

        // Has to be insurance fund type
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );

        // Check if pool is in liquidation from pool's strategy risk info
        let is_pool_in_liquidation = tx_state.risk_infos[POOL_CROSS_RISK_ID]
            .cross_risk_parameters
            .is_in_liquidation(builder);
        builder.conditional_assert_false(is_enabled, is_pool_in_liquidation);

        self.account_shares = tx_state.public_pool_share.share_amount;

        // Must have enough to burn
        builder.conditional_assert_lte(is_enabled, self.share_amount, self.account_shares, 64);

        // Pool must have enough shares to burn
        let available_shares_to_burn = get_available_shares_to_burn_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_STRATEGY_RISK_ID].cross_risk_parameters,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SHARE_OWNER_ACCOUNT_ID],
        );
        builder.conditional_assert_lte(is_enabled, self.share_amount, available_shares_to_burn, 64);

        let shares_to_burn_usdc_value = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SHARE_OWNER_ACCOUNT_ID],
            self.share_amount,
        );
        self.shares_to_burn_usdc_value = builder.biguint_to_target_safe(&shares_to_burn_usdc_value);
        builder.register_range_check(
            self.shares_to_burn_usdc_value,
            MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS,
        );

        {
            let big_usdc_to_collateral_multiplier =
                builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));

            let big_entry_usdc =
                builder.target_to_biguint(tx_state.public_pool_share.principal_amount);
            let big_share_amount = builder.target_to_biguint(self.share_amount);
            let big_owned_share_amount = builder.target_to_biguint(self.account_shares);
            let entry_usdc_mul_share_amount =
                builder.mul_biguint(&big_entry_usdc, &big_share_amount);
            let big_usdc_paid_for_shares =
                builder.div_biguint(&entry_usdc_mul_share_amount, &big_owned_share_amount);
            let usd_paid_for_shares = builder.biguint_to_target_safe(&big_usdc_paid_for_shares);
            let has_profit = builder.is_lt(usd_paid_for_shares, self.shares_to_burn_usdc_value, 64);

            {
                let usdc_profit = builder.sub(self.shares_to_burn_usdc_value, usd_paid_for_shares);
                let big_usdc_profit = builder.target_to_biguint(usdc_profit);
                let big_operator_fee = builder.target_to_biguint(
                    tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
                        .public_pool_info
                        .operator_fee,
                );
                let big_usdc_profit_mul_operator_fee =
                    builder.mul_biguint(&big_usdc_profit, &big_operator_fee);

                let big_total_shares = builder.target_to_biguint(
                    tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
                        .public_pool_info
                        .total_shares,
                );
                let big_total_shares_mul_usdc_to_collateral_multiplier =
                    builder.mul_biguint(&big_total_shares, &big_usdc_to_collateral_multiplier);

                let big_fee_tick = builder.constant_biguint(&BigUint::from(FEE_TICK));
                let big_tav = tx_state.risk_infos[POOL_CROSS_RISK_ID]
                    .cross_risk_parameters
                    .total_account_value
                    .abs
                    .clone(); // always positive since account can not be in liquidation
                let big_fee_tick_mul_tav = builder.mul_biguint(&big_fee_tick, &big_tav);

                let a = builder.mul_biguint(
                    &big_usdc_profit_mul_operator_fee,
                    &big_total_shares_mul_usdc_to_collateral_multiplier,
                );
                // e.operatorFeeShareAmount <= publicPool.PublicPoolInfo.TotalShares
                let big_operator_fee_share_amount = builder.div_biguint(&a, &big_fee_tick_mul_tav);
                let operator_fee_share_amount =
                    builder.biguint_to_target_unsafe(&big_operator_fee_share_amount);

                self.operator_fee_share = builder.select(
                    has_profit,
                    operator_fee_share_amount,
                    self.operator_fee_share,
                );
            }
        }

        self.shares_to_burn = builder.sub(self.share_amount, self.operator_fee_share);
        let shares_to_burn_usdc_value = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SHARE_OWNER_ACCOUNT_ID],
            self.shares_to_burn,
        );
        self.shares_to_burn_usdc_value = builder.biguint_to_target_safe(&shares_to_burn_usdc_value);

        // Set new principal amount and new share amount
        {
            let total_force_burned_shares =
                builder.add(self.operator_fee_share, self.shares_to_burn);
            let big_entry_usdc =
                builder.target_to_biguint(tx_state.public_pool_share.principal_amount);
            let big_total_force_burnt_shares = builder.target_to_biguint(total_force_burned_shares);
            let big_owner_shares = builder.target_to_biguint(self.account_shares);
            let big_entry_mul_total_force_burnt =
                builder.mul_biguint(&big_entry_usdc, &big_total_force_burnt_shares);
            let big_principal_amount_delta =
                builder.div_biguint(&big_entry_mul_total_force_burnt, &big_owner_shares);
            let principal_amount_delta =
                builder.biguint_to_target_unsafe(&big_principal_amount_delta);
            self.new_principal_amount = builder.sub(
                tx_state.public_pool_share.principal_amount,
                principal_amount_delta,
            );
        }

        let max_allowed_principal = {
            let staked_share_info = tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID]
                .get_public_pool_share(
                    builder,
                    tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].account_index,
                );
            let staked_lit_amount = get_shares_asset_value_for_staking_pool(
                builder,
                tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
                // Because LIT can't be used as margin, we can use asset balance directly without considering unified accounts
                &tx_state.account_assets[SYSTEM_CONFIG_ACCOUNT_ID][STAKE_ASSET_ID].balance,
                &tx_state.assets[STAKE_ASSET_ID].extension_multiplier,
                staked_share_info.share_amount,
            );
            let llp_to_mint_shares_multiplier =
                builder.constant_biguint(&BigUint::from(LIT_TO_MINT_SHARES_MULTIPLIER));
            builder.mul_biguint(&llp_to_mint_shares_multiplier, &staked_lit_amount)
        };
        let new_principal_amount = {
            let usdc_to_lit_conversion_rate =
                builder.constant_biguint(&BigUint::from(USDC_TO_LIT_CONVERSION_RATE));
            let llp_principal_amount_big = builder.target_to_biguint(self.new_principal_amount);
            builder.mul_biguint(&usdc_to_lit_conversion_rate, &llp_principal_amount_big)
        };

        builder.conditional_assert_lte_biguint(
            is_enabled,
            &max_allowed_principal,
            &new_principal_amount,
        );
    }
}

impl Apply for L2ForceBurnSharesTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let big_shares_usdc_value = builder.target_to_biguint(self.shares_to_burn_usdc_value);
        let big_usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        let collateral_diff = builder.mul_biguint_non_carry(
            &big_shares_usdc_value,
            &big_usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        let positive_collateral_delta = builder.biguint_to_bigint(&collateral_diff);
        tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &positive_collateral_delta,
            &mut tx_state.strategies[LIQUIDITY_POOL_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[LIQUIDITY_POOL_ACCOUNT_ID][USDC_BASE_ASSET_ID]
                .balance,
        );
        let negative_collateral_delta = builder.neg_bigint(&positive_collateral_delta);
        tx_state.accounts[SHARE_OWNER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &negative_collateral_delta,
            &mut tx_state.strategies[SHARE_OWNER_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[SHARE_OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID]
                .balance,
        );

        let new_total_shares = builder.sub(
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            self.shares_to_burn,
        );
        tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
            .public_pool_info
            .total_shares = builder.select(
            self.success,
            new_total_shares,
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
        );

        let new_operator_shares = builder.add(
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
                .public_pool_info
                .operator_shares,
            self.operator_fee_share,
        );
        tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
            .public_pool_info
            .operator_shares = builder.select(
            self.success,
            new_operator_shares,
            tx_state.accounts[SHARE_OWNER_ACCOUNT_ID]
                .public_pool_info
                .operator_shares,
        );

        tx_state.public_pool_share.principal_amount = builder.select(
            self.success,
            self.new_principal_amount,
            tx_state.public_pool_share.principal_amount,
        );
        let total_burned_shares = builder.add(self.operator_fee_share, self.shares_to_burn);
        let new_share_amount =
            builder.sub(tx_state.public_pool_share.share_amount, total_burned_shares);
        tx_state.public_pool_share.share_amount = builder.select(
            self.success,
            new_share_amount,
            tx_state.public_pool_share.share_amount,
        );

        tx_state.apply_pool_share_delta_flag =
            builder.or(tx_state.apply_pool_share_delta_flag, self.success);

        self.success
    }
}

pub trait L2ForceBurnSharesTxTargetWitness<F: PrimeField64> {
    fn set_l2_force_burn_shares_tx_target(
        &mut self,
        a: &L2ForceBurnSharesTxTarget,
        b: &L2ForceBurnSharesTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2ForceBurnSharesTxTargetWitness<F> for T {
    fn set_l2_force_burn_shares_tx_target(
        &mut self,
        a: &L2ForceBurnSharesTxTarget,
        b: &L2ForceBurnSharesTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.depositor_index, F::from_canonical_i64(b.depositor_index))?;
        self.set_target(a.share_amount, F::from_canonical_i64(b.share_amount))?;
        Ok(())
    }
}
