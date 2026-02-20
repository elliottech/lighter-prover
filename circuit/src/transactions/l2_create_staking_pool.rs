// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::get_available_asset_balance_const;
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
pub struct L2CreateStakingPoolTx {
    #[serde(rename = "ai")]
    pub account_index: i64, // 48 bits
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "i")]
    pub initial_total_shares: i64,
    #[serde(rename = "m")]
    pub min_operator_share_rate: i64,
}

#[derive(Debug, Clone)]
pub struct L2CreateStakingPoolTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits
    pub initial_total_shares: Target,
    pub min_operator_share_rate: Target,

    // helper
    amount_for_pool: BigUintTarget,
    // output
    success: BoolTarget,
}

impl L2CreateStakingPoolTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2CreateStakingPoolTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            initial_total_shares: builder.add_virtual_target(),
            min_operator_share_rate: builder.add_virtual_target(),
            amount_for_pool: builder.zero_biguint(),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2CreateStakingPoolTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_STAKING_POOL)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.initial_total_shares,
            self.min_operator_share_rate,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2CreateStakingPoolTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_create_staking_pool;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[MASTER_ACCOUNT_ID].account_index,
        );
        // Only treasury is allowed to create a staking pool
        builder.conditional_assert_eq_constant(
            is_enabled,
            self.account_index,
            TREASURY_ACCOUNT_INDEX as u64,
        );

        builder.conditional_assert_eq(
            is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );

        // Limit to lit
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            LIT_ASSET_INDEX,
        );

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        // Initial total shares
        builder.register_range_check(self.initial_total_shares, INITIAL_TOTAL_STAKING_SHARES_BITS);
        let min_initial_total_staking_shares =
            builder.constant_u64(MIN_INITIAL_TOTAL_STAKING_SHARES);
        builder.conditional_assert_lte(
            is_enabled,
            min_initial_total_staking_shares,
            self.initial_total_shares,
            INITIAL_TOTAL_STAKING_SHARES_BITS,
        );
        let max_initial_total_shares = builder.constant_u64(MAX_INITIAL_TOTAL_STAKING_SHARES);
        builder.conditional_assert_lte(
            is_enabled,
            self.initial_total_shares,
            max_initial_total_shares,
            INITIAL_TOTAL_STAKING_SHARES_BITS,
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

        // Creator must have enough assets to create the pool.
        let initial_pool_share_value = builder.constant_u64(INITIAL_POOL_SHARE_VALUE);
        let initial_total_shares_value =
            builder.mul(self.initial_total_shares, initial_pool_share_value);
        let initial_total_shares_value_big = builder.target_to_biguint(initial_total_shares_value);
        self.amount_for_pool = builder.mul_biguint_non_carry(
            &initial_total_shares_value_big,
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
        builder.conditional_assert_lte_biguint(is_enabled, &self.amount_for_pool, &asset_balance);
    }
}

impl Apply for L2CreateStakingPoolTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let staking_pool_account_type =
            builder.constant(F::from_canonical_u8(LIGHTER_STAKING_POOL_ACCOUNT_TYPE));
        tx_state.accounts[SUB_ACCOUNT_ID].account_type = builder.select(
            self.success,
            staking_pool_account_type,
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
        );
        let simple_trading_mode =
            builder.constant(F::from_canonical_u8(ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE));
        tx_state.accounts[SUB_ACCOUNT_ID].account_trading_mode = builder.select(
            self.success,
            simple_trading_mode,
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

        tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &self.amount_for_pool,
            &tx_state.account_assets[SUB_ACCOUNT_ID][TX_ASSET_ID].balance,
        );
        let (new_owner_balance, fail) = builder.try_sub_biguint(
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &self.amount_for_pool,
        );
        builder.conditional_assert_zero_u32(self.success, fail);
        tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &new_owner_balance,
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        let staking_pool_info = &PublicPoolInfoTarget {
            status: builder.constant_from_u8(ACTIVE_PUBLIC_POOL),
            operator_fee: builder.zero(), // Zero operator fee for staking pools
            min_operator_share_rate: self.min_operator_share_rate,
            total_shares: self.initial_total_shares,
            operator_shares: self.initial_total_shares,
            strategies: core::array::from_fn(|_| builder.zero_bigint()),
        };
        tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info = select_public_pool_info_target(
            builder,
            self.success,
            staking_pool_info,
            &tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info,
        );

        self.success
    }
}

pub trait L2CreateStakingPoolTxTargetWitness<F: PrimeField64> {
    fn set_l2_create_staking_pool_tx_target(
        &mut self,
        a: &L2CreateStakingPoolTxTarget,
        b: &L2CreateStakingPoolTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2CreateStakingPoolTxTargetWitness<F> for T {
    fn set_l2_create_staking_pool_tx_target(
        &mut self,
        a: &L2CreateStakingPoolTxTarget,
        b: &L2CreateStakingPoolTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
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
