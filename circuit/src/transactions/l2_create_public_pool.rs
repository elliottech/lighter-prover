// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{BoolOrTarget, get_available_asset_balance};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::public_pool::{PublicPoolInfoTarget, select_public_pool_info_target};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2CreatePublicPoolTx {
    #[serde(rename = "ai")]
    pub account_index: i64, // 48 bits

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "o")]
    pub operator_fee: i64,

    #[serde(rename = "i")]
    pub initial_total_shares: i64,

    #[serde(rename = "m")]
    pub min_operator_share_rate: i64,
}

#[derive(Debug, Clone)]
pub struct L2CreatePublicPoolTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits
    pub operator_fee: Target,
    pub initial_total_shares: Target,
    pub min_operator_share_rate: Target,

    // helper
    pub account_type: Target,
    pub account_trading_mode: Target,
    pub collateral_delta: BigIntTarget,

    // output
    pub success: BoolTarget,
}

impl L2CreatePublicPoolTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2CreatePublicPoolTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            operator_fee: builder.add_virtual_target(),
            initial_total_shares: builder.add_virtual_target(),
            min_operator_share_rate: builder.add_virtual_target(),

            // helper
            account_type: builder.zero(),
            account_trading_mode: builder.zero(),
            collateral_delta: builder.zero_bigint(),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2CreatePublicPoolTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_PUBLIC_POOL)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.operator_fee,
            self.initial_total_shares,
            self.min_operator_share_rate,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CreatePublicPoolTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_create_public_pool;
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

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        let fee_tick = builder.constant(F::from_canonical_u64(FEE_TICK));
        builder.register_range_check(self.operator_fee, 24);
        builder.conditional_assert_lte(is_enabled, self.operator_fee, fee_tick, FEE_BITS);

        builder.conditional_assert_not_zero(is_enabled, self.initial_total_shares);
        let max_initial_total_shares = builder.constant_u64(MAX_INITIAL_TOTAL_SHARES);
        builder.register_range_check(self.initial_total_shares, INITIAL_TOTAL_SHARES_BITS);
        builder.conditional_assert_lte(
            is_enabled,
            self.initial_total_shares,
            max_initial_total_shares,
            INITIAL_TOTAL_SHARES_BITS,
        );

        let share_tick = builder.constant(F::from_canonical_u64(SHARE_TICK));
        builder.register_range_check(self.min_operator_share_rate, SHARE_RATE_BITS);
        builder.conditional_assert_lte(
            is_enabled,
            self.min_operator_share_rate,
            share_tick,
            SHARE_RATE_BITS,
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

        let insurance_fund_operator_account_index =
            builder.constant_usize(INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX);
        let is_insurance_fund_operator_account = builder.is_equal(
            tx_state.accounts[MASTER_ACCOUNT_ID].account_index,
            insurance_fund_operator_account_index,
        );

        self.account_type = builder.select_constant(
            is_insurance_fund_operator_account,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
            PUBLIC_POOL_ACCOUNT_TYPE as u64,
        );
        self.account_trading_mode = builder.select_constant(
            is_insurance_fund_operator_account,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED as u64,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE as u64,
        );

        let is_insurance_fund_and_enabled =
            builder.and(is_insurance_fund_operator_account, is_enabled);
        builder.conditional_assert_zero(is_insurance_fund_and_enabled, self.operator_fee);
        builder
            .conditional_assert_zero(is_insurance_fund_and_enabled, self.min_operator_share_rate);

        let initial_pool_share_value = builder.constant_u64(INITIAL_POOL_SHARE_VALUE);
        let pool_usdc_value = builder.mul(self.initial_total_shares, initial_pool_share_value);
        let pool_usdc_value_big = builder.target_to_biguint(pool_usdc_value);
        let usdc_to_collateral_multiplier = builder.constant_u32(USDC_TO_COLLATERAL_MULTIPLIER);
        let collateral_delta = builder.mul_biguint_non_carry(
            &pool_usdc_value_big,
            &BigUintTarget::from(usdc_to_collateral_multiplier),
            BIG_U96_LIMBS,
        );

        let _perps = builder.constant_u64(PRODUCT_TYPE_PERPS);
        let available_collateral_to_transfer = get_available_asset_balance(
            builder,
            _perps,
            tx_state.asset_indices[TX_ASSET_ID],
            &tx_state.accounts[MASTER_ACCOUNT_ID],
            &tx_state.account_assets[MASTER_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[MASTER_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[MASTER_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.margined_asset[TX_ASSET_ID],
            &tx_state.account_margined_assets[MASTER_ACCOUNT_ID][TX_ASSET_ID].balance,
            BoolOrTarget::False,
        );
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &collateral_delta,
            &available_collateral_to_transfer,
        );
        self.collateral_delta = builder.biguint_to_bigint(&collateral_delta);
    }
}

impl Apply for L2CreatePublicPoolTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.accounts[SUB_ACCOUNT_ID].account_type = builder.select(
            self.success,
            self.account_type,
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
        );
        tx_state.accounts[SUB_ACCOUNT_ID].account_trading_mode = builder.select(
            self.success,
            self.account_trading_mode,
            tx_state.accounts[SUB_ACCOUNT_ID].account_trading_mode,
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

        let public_pool_info = &PublicPoolInfoTarget {
            status: builder.constant_from_u8(ACTIVE_PUBLIC_POOL),
            operator_fee: self.operator_fee,
            min_operator_share_rate: self.min_operator_share_rate,
            total_shares: self.initial_total_shares,
            operator_shares: self.initial_total_shares,
            strategies: core::array::from_fn(|_| builder.zero_bigint()),
        };
        tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info = select_public_pool_info_target(
            builder,
            self.success,
            public_pool_info,
            &tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info,
        );

        tx_state.accounts[SUB_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &self.collateral_delta,
            &mut tx_state.strategies[SUB_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        let owner_collateral_delta = builder.neg_bigint(&self.collateral_delta);
        tx_state.accounts[MASTER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &owner_collateral_delta,
            &mut tx_state.strategies[MASTER_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[MASTER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        self.success
    }
}

pub trait L2CreatePublicPoolTxTargetWitness<F: PrimeField64> {
    fn set_l2_create_public_pool_tx_target(
        &mut self,
        a: &L2CreatePublicPoolTxTarget,
        b: &L2CreatePublicPoolTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CreatePublicPoolTxTargetWitness<F> for T {
    fn set_l2_create_public_pool_tx_target(
        &mut self,
        a: &L2CreatePublicPoolTxTarget,
        b: &L2CreatePublicPoolTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.operator_fee, F::from_canonical_i64(b.operator_fee))?;
        self.set_target(
            a.initial_total_shares,
            F::from_canonical_i64(b.initial_total_shares),
        )?;
        self.set_target(
            a.min_operator_share_rate,
            F::from_canonical_i64(b.min_operator_share_rate),
        )?;

        Ok(())
    }
}
