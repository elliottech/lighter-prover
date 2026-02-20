// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{
    get_available_asset_balance_const, get_shares_asset_value_for_staking_pool,
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
pub struct L2MintSharesTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "p", default)]
    pub public_pool_index: i64,

    #[serde(rename = "s", default)]
    pub share_amount: i64,
}

#[derive(Debug)]
pub struct L2MintSharesTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub public_pool_index: Target,
    pub share_amount: Target,

    // Helper
    is_operator: BoolTarget,
    principal_amount: Target,
    new_total_shares: Target,
    new_principal_amount: Target,
    collateral_to_mint_shares: BigIntTarget,

    // Output
    success: BoolTarget,
}

impl L2MintSharesTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2MintSharesTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            public_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),

            // Helper
            is_operator: builder._false(),
            principal_amount: builder.zero(),
            new_total_shares: builder.zero(),
            new_principal_amount: builder.zero(),
            collateral_to_mint_shares: builder.zero_bigint(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2MintSharesTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_MINT_SHARES)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.public_pool_index,
            self.share_amount,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2MintSharesTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_mint_shares;
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
            self.public_pool_index,
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
        );

        // First asset always has to be USDC for minting pool shares
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        let is_llp = builder.is_equal(
            self.public_pool_index,
            tx_state.system_config.liquidity_pool_index,
        );
        let is_llp_flag = builder.and(is_enabled, is_llp);
        builder.conditional_assert_eq(
            is_llp_flag,
            tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].account_index,
            tx_state.system_config.staking_pool_index,
        );
        builder.conditional_assert_eq_constant(
            is_llp_flag,
            tx_state.asset_indices[STAKE_ASSET_ID],
            LIT_ASSET_INDEX,
        );

        let big_shares_amount = builder.target_to_biguint(self.share_amount);
        builder.range_check_biguint(&big_shares_amount, MAX_POOL_SHARES_BITS);
        builder.conditional_assert_not_zero(is_enabled, self.share_amount);

        let public_pool_account_type = builder.constant_from_u8(PUBLIC_POOL_ACCOUNT_TYPE);
        let insurance_fund_account_type = builder.constant_from_u8(INSURANCE_FUND_ACCOUNT_TYPE);
        let is_public_pool_account_type = builder.is_equal(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            public_pool_account_type,
        );
        let is_insurance_fund_account_type = builder.is_equal(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            insurance_fund_account_type,
        );
        let is_valid_account_type =
            builder.or(is_public_pool_account_type, is_insurance_fund_account_type);
        builder.conditional_assert_true(is_enabled, is_valid_account_type);

        let active_public_pool = builder.constant_from_u8(ACTIVE_PUBLIC_POOL);
        let is_active_public_pool = builder.is_equal(
            tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info.status,
            active_public_pool,
        );
        builder.conditional_assert_true(is_enabled, is_active_public_pool);

        self.new_total_shares = builder.add(
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
            self.share_amount,
        );
        let big_new_total_shares = builder.target_to_biguint(self.new_total_shares);
        builder.range_check_biguint(&big_new_total_shares, MAX_POOL_SHARES_BITS);

        let available_collateral_to_mint_shares = get_available_asset_balance_const(
            builder,
            PRODUCT_TYPE_PERPS,
            &tx_state.accounts[OWNER_ACCOUNT_ID],
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID], // usdc
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID], // usdc
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
        );

        self.principal_amount = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[SUB_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
            self.share_amount,
        );
        let big_principal_amount = builder.target_to_biguint(self.principal_amount);
        builder.range_check_biguint(
            &big_principal_amount,
            MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS,
        );

        let usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));

        let collateral_to_mint_shares = builder.mul_biguint_non_carry(
            &big_principal_amount,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &collateral_to_mint_shares,
            &available_collateral_to_mint_shares,
        );
        self.collateral_to_mint_shares = builder.biguint_to_bigint(&collateral_to_mint_shares);

        self.is_operator = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );

        // Not operator checks
        {
            // If minter is not the operator, then check if the minimum share rate is still
            // going to be satisfied for the pool operator. If operator shares drops below the
            // minimum operator share rate, fail the transaction
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

            // Range check new principal amount only for non-operator
            let new_principal_amount = builder.add(
                tx_state.public_pool_share.principal_amount,
                self.principal_amount,
            );
            self.new_principal_amount =
                builder.select_or_zero(not_operator_and_enabled, new_principal_amount);
            builder.register_range_check(self.new_principal_amount, MAX_POOL_PRINCIPAL_AMOUNT_BITS);

            // If staking pool exists, LIT asset registered, current pool is LLP and account is not staking
            // pool operator or LLP operator, verify that the mint amount is within staked LIT limits.
            // A third account must have been sent in this case.
            let is_not_staking_pool_operator = builder.is_not_equal(
                tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
                tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].master_account_index,
            );
            let is_lit_asset_empty = tx_state.assets[STAKE_ASSET_ID].is_empty(builder);
            let is_lit_asset_not_empty = builder.not(is_lit_asset_empty);

            let is_staking_pool_nil_account = builder.is_equal_constant(
                tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].account_index,
                NIL_ACCOUNT_INDEX as u64,
            );
            let is_staking_pool_not_nil_account = builder.not(is_staking_pool_nil_account);
            let stake_limit_check_flag = builder.multi_and(&[
                not_operator_and_enabled,
                is_lit_asset_not_empty,
                is_llp,
                is_not_staking_pool_operator,
                is_staking_pool_not_nil_account,
            ]);

            let depositor_staked_lit_shares = tx_state.accounts[OWNER_ACCOUNT_ID]
                .get_public_pool_share(
                    builder,
                    tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID].account_index,
                )
                .share_amount;

            // stakedLitAmount is in base amount, needs to be divided by 10^lit_decimals to find how many LIT tokens are staked
            let staked_lit_amount = get_shares_asset_value_for_staking_pool(
                builder,
                tx_state.accounts[SYSTEM_CONFIG_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
                // Because LIT can't be used as margin, we can use asset balance directly without considering unified accounts
                &tx_state.account_assets[SYSTEM_CONFIG_ACCOUNT_ID][STAKE_ASSET_ID].balance,
                &tx_state.assets[STAKE_ASSET_ID].extension_multiplier,
                depositor_staked_lit_shares,
            );

            // Allow LIT_TO_MINT_SHARES_MULTIPLIER USDC per 1 LIT staked in the staking pool
            let llp_to_mint_shares_multiplier =
                builder.constant_biguint(&BigUint::from(LIT_TO_MINT_SHARES_MULTIPLIER));
            let max_allowed_principal =
                builder.mul_biguint(&llp_to_mint_shares_multiplier, &staked_lit_amount);
            let usdc_to_lit_conversion_rate =
                builder.constant_biguint(&BigUint::from(USDC_TO_LIT_CONVERSION_RATE));
            let new_principal_amount_big = builder.target_to_biguint(self.new_principal_amount);
            let new_principal_amount =
                builder.mul_biguint(&new_principal_amount_big, &usdc_to_lit_conversion_rate);

            builder.conditional_assert_lte_biguint(
                stake_limit_check_flag,
                &new_principal_amount,
                &max_allowed_principal,
            );
        }
    }
}

impl Apply for L2MintSharesTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.accounts[SUB_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &self.collateral_to_mint_shares,
            &mut tx_state.strategies[SUB_ACCOUNT_ID],
        );

        let neg_collateral_delta = builder.neg_bigint(&self.collateral_to_mint_shares);
        tx_state.accounts[OWNER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &neg_collateral_delta,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
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

        // Set pool shares - not operator
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
            tx_state.public_pool_share.entry_timestamp = builder.select(
                is_success_and_not_operator,
                tx_state.block_timestamp,
                tx_state.public_pool_share.entry_timestamp,
            );
            tx_state.apply_pool_share_delta_flag = builder.or(
                tx_state.apply_pool_share_delta_flag,
                is_success_and_not_operator,
            );
        }
        // Set pool shares - is operator
        {
            let is_success_and_operator = builder.and(self.success, self.is_operator);
            let new_operator_shares_for_operator = builder.add(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
                self.share_amount,
            );
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares = builder.select(
                is_success_and_operator,
                new_operator_shares_for_operator,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
            );
        }

        self.success
    }
}

pub trait L2MintSharesTxTargetWitness<F: PrimeField64> {
    fn set_l2_mint_shares_tx_target(
        &mut self,
        a: &L2MintSharesTxTarget,
        b: &L2MintSharesTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2MintSharesTxTargetWitness<F> for T {
    fn set_l2_mint_shares_tx_target(
        &mut self,
        a: &L2MintSharesTxTarget,
        b: &L2MintSharesTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.public_pool_index,
            F::from_canonical_i64(b.public_pool_index),
        )?;
        self.set_target(a.share_amount, F::from_canonical_i64(b.share_amount))?;
        Ok(())
    }
}
