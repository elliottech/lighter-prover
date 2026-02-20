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
    get_available_shares_to_burn_for_public_pool, get_shares_usdc_value_for_public_pool,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2BurnSharesTx {
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
pub struct L2BurnSharesTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub public_pool_index: Target,
    pub share_amount: Target,

    // Helper
    pub is_operator: BoolTarget,
    pub account_shares: Target,

    pub shares_to_burn: Target,
    pub shares_to_burn_usdc_value: Target,
    pub old_principal_amount: Target,

    pub operator_fee_share: Target,

    // Output
    pub success: BoolTarget,
}

impl L2BurnSharesTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2BurnSharesTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            public_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),

            // Helper
            is_operator: builder._false(),
            account_shares: builder.zero(),

            shares_to_burn: builder.zero(),
            shares_to_burn_usdc_value: builder.zero(),
            old_principal_amount: builder.zero(),

            operator_fee_share: builder.zero(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2BurnSharesTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_BURN_SHARES)),
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

impl Verify for L2BurnSharesTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_burn_shares;

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

        self.success = is_enabled;

        let big_shares_amount = builder.target_to_biguint(self.share_amount);
        builder.range_check_biguint(
            &big_shares_amount,
            MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS,
        );
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

        let is_pool_in_liquidation = tx_state.risk_infos[POOL_CROSS_RISK_ID]
            .cross_risk_parameters
            .is_in_liquidation(builder);
        builder.conditional_assert_false(is_enabled, is_pool_in_liquidation);

        self.is_operator = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );

        // Enforce min burn period for llp
        {
            let is_not_operator = builder.not(self.is_operator);
            let is_llp = builder.is_equal(
                self.public_pool_index,
                tx_state.system_config.liquidity_pool_index,
            );
            let earliest_burn_timestamp = builder.add(
                tx_state.public_pool_share.entry_timestamp,
                tx_state.system_config.liquidity_pool_cooldown_period,
            );
            let burn_period_is_not_elapsed =
                builder.is_lt(tx_state.block_timestamp, earliest_burn_timestamp, 64);

            let should_be_false =
                builder.multi_and(&[is_not_operator, is_llp, burn_period_is_not_elapsed]);
            builder.conditional_assert_false(is_enabled, should_be_false);
        }

        self.old_principal_amount = tx_state.public_pool_share.principal_amount; // To be used in apply

        self.account_shares = builder.select(
            self.is_operator,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares,
            tx_state.public_pool_share.share_amount,
        );

        builder.conditional_assert_lte(is_enabled, self.share_amount, self.account_shares, 64);

        let available_shares_to_burn = get_available_shares_to_burn_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_STRATEGY_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
        );

        builder.conditional_assert_lte(is_enabled, self.share_amount, available_shares_to_burn, 64);

        self.shares_to_burn_usdc_value = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
            self.share_amount,
        );
        builder.register_range_check(
            self.shares_to_burn_usdc_value,
            MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS,
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

        {
            let big_usdc_to_collateral_multiplier =
                builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
            let big_entry_usdc = builder.target_to_biguint(self.old_principal_amount);
            let big_share_amount = builder.target_to_biguint(self.share_amount);
            let big_owned_share_amount = builder.target_to_biguint(self.account_shares);
            let entry_usdc_mul_share_amount =
                builder.mul_biguint(&big_entry_usdc, &big_share_amount);
            let big_usdc_paid_for_shares =
                builder.div_biguint(&entry_usdc_mul_share_amount, &big_owned_share_amount);
            let usd_paid_for_shares = builder.biguint_to_target_safe(&big_usdc_paid_for_shares);
            let has_profit = builder.is_lt(usd_paid_for_shares, self.shares_to_burn_usdc_value, 64);
            let has_profit_and_not_operator = builder.and_not(has_profit, self.is_operator);

            {
                let usdc_profit = builder.sub(self.shares_to_burn_usdc_value, usd_paid_for_shares);
                let big_usdc_profit = builder.target_to_biguint(usdc_profit);
                let big_operator_fee = builder.target_to_biguint(
                    tx_state.accounts[SUB_ACCOUNT_ID]
                        .public_pool_info
                        .operator_fee,
                );
                let big_usdc_profit_mul_operator_fee =
                    builder.mul_biguint(&big_usdc_profit, &big_operator_fee);

                let big_total_shares = builder.target_to_biguint(
                    tx_state.accounts[SUB_ACCOUNT_ID]
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
                    has_profit_and_not_operator,
                    operator_fee_share_amount,
                    self.operator_fee_share,
                );
            }
        }

        self.shares_to_burn = builder.sub(self.share_amount, self.operator_fee_share);
        self.shares_to_burn_usdc_value = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
            self.shares_to_burn,
        );
    }
}

impl Apply for L2BurnSharesTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let big_shares_usdc_value = builder.target_to_biguint(self.shares_to_burn_usdc_value);
        let big_usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        let big_collateral_amount = builder.mul_biguint_non_carry(
            &big_shares_usdc_value,
            &big_usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        let positive_collateral_delta = builder.biguint_to_bigint(&big_collateral_amount);
        let negative_collateral_delta = builder.neg_bigint(&positive_collateral_delta);
        tx_state.accounts[OWNER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &positive_collateral_delta,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
        );
        tx_state.accounts[SUB_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &negative_collateral_delta,
            &mut tx_state.strategies[SUB_ACCOUNT_ID],
        );

        {
            let op_success = builder.and(self.success, self.is_operator);

            let new_operator_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
                self.shares_to_burn,
            );
            let new_total_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
                self.shares_to_burn,
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
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares = builder.select(
                op_success,
                new_total_shares,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
            );
        }

        // Updates for if enabled and burning is not the account operator
        {
            let non_operator_success = builder.and_not(self.success, self.is_operator);

            let new_operator_shares = builder.add(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
                self.operator_fee_share,
            );
            let new_total_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
                self.shares_to_burn,
            );
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares = builder.select(
                non_operator_success,
                new_operator_shares,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
            );
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares = builder.select(
                non_operator_success,
                new_total_shares,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
            );

            let total_burned_shares = builder.add(self.operator_fee_share, self.shares_to_burn);

            // entry usdc
            // let entry_usdc_delta = entry usdc * total_burned / share amount
            let big_entry_usdc = builder.target_to_biguint(self.old_principal_amount);
            let big_total_burnt_shares = builder.target_to_biguint(total_burned_shares);
            let big_owner_shares = builder.target_to_biguint(self.account_shares);
            let big_entry_mul_total_burnt =
                builder.mul_biguint(&big_entry_usdc, &big_total_burnt_shares);
            let big_principal_amount_delta =
                builder.div_biguint(&big_entry_mul_total_burnt, &big_owner_shares);
            let principal_amount_delta =
                builder.biguint_to_target_unsafe(&big_principal_amount_delta);

            let new_share_amount =
                builder.sub(tx_state.public_pool_share.share_amount, total_burned_shares);
            let new_principal_amount = builder.sub(
                tx_state.public_pool_share.principal_amount,
                principal_amount_delta,
            );
            tx_state.public_pool_share.principal_amount = builder.select(
                non_operator_success,
                new_principal_amount,
                tx_state.public_pool_share.principal_amount,
            );
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

pub trait L2BurnSharesTxTargetWitness<F: PrimeField64> {
    fn set_l2_burn_shares_tx_target(
        &mut self,
        a: &L2BurnSharesTxTarget,
        b: &L2BurnSharesTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2BurnSharesTxTargetWitness<F> for T {
    fn set_l2_burn_shares_tx_target(
        &mut self,
        a: &L2BurnSharesTxTarget,
        b: &L2BurnSharesTx,
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
