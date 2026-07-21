// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::liquidation::{
    get_available_shares_to_burn_for_public_pool, get_shares_usdc_value_for_public_pool,
};
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U128_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct L1BurnSharesTx {
    #[serde(rename = "mai")]
    pub master_account_index: i64,
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "p", default)]
    pub public_pool_index: i64,
    #[serde(rename = "s", default)]
    pub share_amount: i64,
}

#[derive(Debug)]
pub struct L1BurnSharesTxTarget {
    pub account_index: Target,
    pub master_account_index: Target,
    pub public_pool_index: Target,
    pub share_amount: Target,

    // Helper
    is_operator: BoolTarget,
    account_shares: Target,

    shares_to_burn: Target,
    old_entry_quote: Target,

    operator_fee_share: Target,
    big_collateral_amount: BigIntTarget,

    // Output
    is_enabled: BoolTarget,
    success: BoolTarget,
}

impl L1BurnSharesTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            master_account_index: builder.add_virtual_target(),
            public_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),

            is_operator: Default::default(),
            account_shares: Default::default(),

            shares_to_burn: Default::default(),
            old_entry_quote: Default::default(),

            operator_fee_share: Default::default(),
            big_collateral_amount: Default::default(),

            is_enabled: Default::default(),
            success: Default::default(),
        }
    }
}

impl Verify for L1BurnSharesTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        let zero = builder.zero();

        self.is_enabled = tx_type.is_l1_burn_shares;
        self.success = self.is_enabled;

        builder.conditional_assert_eq(
            self.is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            self.is_enabled,
            self.public_pool_index,
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
        );

        let is_new_account = tx_state.is_new_account[OWNER_ACCOUNT_ID];
        self.success = builder.and_not(self.success, is_new_account);

        let is_correct_master_account_in_state = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
            self.master_account_index,
        );
        self.success = builder.and(self.success, is_correct_master_account_in_state);

        // Verify public pool account is of correct type. This also ensures that the account is exits because default account type is 0.
        let is_public_pool_account_type = builder.is_equal_constant(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            PUBLIC_POOL_ACCOUNT_TYPE as u64,
        );
        let is_insurance_fund_account_type = builder.is_equal_constant(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_valid_account_type =
            builder.or(is_public_pool_account_type, is_insurance_fund_account_type);
        self.success = builder.and(self.success, is_valid_account_type);

        let is_staking_pool = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );
        self.success = builder.and_not(self.success, is_staking_pool);

        let is_pool_in_liquidation = tx_state.risk_infos[POOL_CROSS_RISK_ID]
            .cross_risk_parameters
            .is_in_liquidation(builder);
        self.success = builder.and_not(self.success, is_pool_in_liquidation);

        self.is_operator = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );

        builder.conditional_assert_eq_constant(
            self.is_enabled,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
            USDC_ASSET_INDEX,
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
            self.success = builder.and_not(self.success, should_be_false);
        }

        self.old_entry_quote = tx_state.public_pool_share.principal_amount;

        self.account_shares = builder.select(
            self.is_operator,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares,
            tx_state.public_pool_share.share_amount,
        );

        let is_valid_burn_share_amount = builder.is_lte(self.share_amount, self.account_shares, 64);
        self.success = builder.and(self.success, is_valid_burn_share_amount);

        let is_pool_tav_positive = builder.is_sign_positive(
            tx_state.risk_infos[POOL_CROSS_RISK_ID]
                .cross_risk_parameters
                .total_account_value
                .sign,
        );
        let is_pool_tav_not_positive = builder.not(is_pool_tav_positive);
        let is_total_pool_shares_zero = builder.is_zero(
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
        );
        let positive_tav_zero_shares =
            builder.or(is_total_pool_shares_zero, is_pool_tav_not_positive);
        self.success = builder.and_not(self.success, positive_tav_zero_shares);

        let available_shares_to_burn = get_available_shares_to_burn_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_STRATEGY_RISK_ID].cross_risk_parameters,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
        );
        let has_enough_shares_to_burn =
            builder.is_lte(self.share_amount, available_shares_to_burn, 64);
        self.success = builder.and(self.success, has_enough_shares_to_burn);

        let shares_to_burn_usdc_value = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
            self.share_amount,
        );

        let max_pool_shares_to_burn =
            &builder.constant_biguint(&BigUint::from(MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC));
        let is_burn_amount_not_too_high =
            builder.is_lte_biguint(&shares_to_burn_usdc_value, max_pool_shares_to_burn);
        self.success = builder.and(self.success, is_burn_amount_not_too_high);
        let shares_to_burn_usdc_value =
            builder.biguint_to_target_unsafe(&shares_to_burn_usdc_value);
        let shares_to_burn_usdc_value =
            builder.select(self.success, shares_to_burn_usdc_value, zero);

        // Operator and public pool not frozen - Share availability check
        {
            let frozen_public_pool = builder.constant(F::from_canonical_u8(FROZEN_PUBLIC_POOL));
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
            let is_sufficient_available_shares = builder.is_lte_biguint(&lhs, &rhs);

            // a -> b <==> a' || b
            let is_not_frozen_and_owner_inverse = builder.not(is_not_frozen_and_owner);
            let check = builder.or(
                is_not_frozen_and_owner_inverse,
                is_sufficient_available_shares,
            );
            self.success = builder.and(self.success, check);
        }

        // Not operator
        {
            let big_entry_usdc = builder.target_to_biguint(self.old_entry_quote);
            let big_share_amount = builder.target_to_biguint(self.share_amount);
            let big_owned_share_amount = builder.target_to_biguint(self.account_shares);
            let entry_usdc_mul_share_amount =
                builder.mul_biguint(&big_entry_usdc, &big_share_amount);
            let big_usdc_paid_for_burnt_shares =
                builder.div_biguint(&entry_usdc_mul_share_amount, &big_owned_share_amount);
            let usd_paid_for_burnt_shares =
                builder.biguint_to_target_safe(&big_usdc_paid_for_burnt_shares);
            let has_profit =
                builder.is_lt(usd_paid_for_burnt_shares, shares_to_burn_usdc_value, 64);
            let has_profit_and_not_operator = builder.and_not(has_profit, self.is_operator);

            let usdc_profit = builder.sub(shares_to_burn_usdc_value, usd_paid_for_burnt_shares);
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
                builder.mul_biguint(&big_total_shares, &usdc_to_collateral_multiplier);

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

            self.operator_fee_share =
                builder.select(has_profit_and_not_operator, operator_fee_share_amount, zero);
        }

        self.shares_to_burn = builder.sub(self.share_amount, self.operator_fee_share);
        let big_shares_usdc_value = get_shares_usdc_value_for_public_pool(
            builder,
            &tx_state.risk_infos[POOL_CROSS_RISK_ID].cross_risk_parameters,
            &tx_state.accounts[SUB_ACCOUNT_ID],
            self.shares_to_burn,
        );
        let (success, big_shares_usdc_value) =
            builder.try_trim_biguint(&big_shares_usdc_value, BIG_U64_LIMBS);
        builder.conditional_assert_true(self.success, success);
        let biguint_collateral_amount = builder.mul_biguint_non_carry(
            &big_shares_usdc_value,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );
        self.big_collateral_amount = builder.biguint_to_bigint(&biguint_collateral_amount);

        // Verify that the usdc margin balance after burning shares can fit 96 bits
        let usdc_margin_balance = builder.add_bigint_non_carry(
            &tx_state.accounts[OWNER_ACCOUNT_ID]
                .get_margined_asset_balance_const(USDC_MARGIN_ASSET_INDEX),
            &self.big_collateral_amount,
            BIG_U128_LIMBS,
        );
        let (is_valid_new_collateral, _) =
            builder.try_trim_biguint(&usdc_margin_balance.abs, BIG_U96_LIMBS);
        self.success = builder.and(self.success, is_valid_new_collateral);
    }
}

impl Apply for L1BurnSharesTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        // Common updates
        {
            tx_state.accounts[OWNER_ACCOUNT_ID].apply_collateral_delta(
                builder,
                self.success,
                &self.big_collateral_amount,
                &mut tx_state.strategies[OWNER_ACCOUNT_ID],
                &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][USDC_MARGIN_ASSET_INDEX]
                    .balance,
            );

            let neg_big_collateral_amount = builder.neg_bigint(&self.big_collateral_amount);
            tx_state.accounts[SUB_ACCOUNT_ID].apply_collateral_delta(
                builder,
                self.success,
                &neg_big_collateral_amount,
                &mut tx_state.strategies[SUB_ACCOUNT_ID],
                &mut tx_state.account_margined_assets[SUB_ACCOUNT_ID][USDC_MARGIN_ASSET_INDEX]
                    .balance,
            );

            let new_total_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
                self.shares_to_burn,
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
        }

        // Operator only updates
        {
            let op_success = builder.and(self.success, self.is_operator);

            let new_operator_shares = builder.sub(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
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
        }

        // No-operator only updates
        {
            let nop_success = builder.and_not(self.success, self.is_operator);

            let new_operator_shares = builder.add(
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
                self.operator_fee_share,
            );
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares = builder.select(
                nop_success,
                new_operator_shares,
                tx_state.accounts[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .operator_shares,
            );

            let total_burned_shares = builder.add(self.operator_fee_share, self.shares_to_burn);
            // let entry_usdc_delta = entry usdc * total_burned / share amount
            let big_entry_usdc = builder.target_to_biguint(self.old_entry_quote);
            let big_total_burnt_shares = builder.target_to_biguint(total_burned_shares);
            let big_owner_shares = builder.target_to_biguint(self.account_shares);
            let big_entry_mul_total_burnt =
                builder.mul_biguint(&big_entry_usdc, &big_total_burnt_shares);
            let big_entry_quote_delta =
                builder.div_biguint(&big_entry_mul_total_burnt, &big_owner_shares);
            let entry_quote_delta = builder.biguint_to_target_unsafe(&big_entry_quote_delta);

            let new_total_shares =
                builder.sub(tx_state.public_pool_share.share_amount, total_burned_shares);
            let new_entry_usdc = builder.sub(
                tx_state.public_pool_share.principal_amount,
                entry_quote_delta,
            );
            tx_state.public_pool_share.principal_amount = builder.select(
                nop_success,
                new_entry_usdc,
                tx_state.public_pool_share.principal_amount,
            );
            tx_state.public_pool_share.share_amount = builder.select(
                nop_success,
                new_total_shares,
                tx_state.public_pool_share.share_amount,
            );
            tx_state.apply_pool_share_delta_flag =
                builder.or(tx_state.apply_pool_share_delta_flag, nop_success);
        }

        self.success
    }
}

impl PriorityOperationsPubData for L1BurnSharesTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let bytes_count = [
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_BURN_SHARES),
            add_account_index_target(builder, bytes, self.account_index),
            add_account_index_target(builder, bytes, self.master_account_index),
            add_account_index_target(builder, bytes, self.public_pool_index),
            add_target(builder, bytes, self.share_amount, 64),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, bytes_count),
        )
    }
}

pub trait L1BurnSharesTxTargetWitness<F: PrimeField64> {
    fn set_l1_burn_shares_tx_target(
        &mut self,
        a: &L1BurnSharesTxTarget,
        b: &L1BurnSharesTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1BurnSharesTxTargetWitness<F> for T {
    fn set_l1_burn_shares_tx_target(
        &mut self,
        a: &L1BurnSharesTxTarget,
        b: &L1BurnSharesTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )?;
        self.set_target(
            a.public_pool_index,
            F::from_canonical_i64(b.public_pool_index),
        )?;
        self.set_target(a.share_amount, F::from_canonical_i64(b.share_amount))?;

        Ok(())
    }
}
