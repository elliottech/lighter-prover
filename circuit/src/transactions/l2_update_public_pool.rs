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
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::public_pool::{PublicPoolInfoTarget, select_public_pool_info_target};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UpdatePublicPoolTx {
    #[serde(rename = "ai")]
    pub account_index: i64, // 48 bits

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "p")]
    pub public_pool_index: i64,

    #[serde(rename = "s")]
    pub status: u8,

    #[serde(rename = "o")]
    pub operator_fee: i64,

    #[serde(rename = "m")]
    pub min_operator_share_rate: i64,
}

#[derive(Debug, Clone)]
pub struct L2UpdatePublicPoolTxTarget {
    pub account_index: Target, // 48 bits
    pub api_key_index: Target, // 8 bits
    pub public_pool_index: Target,
    pub status: Target,
    pub operator_fee: Target,
    pub min_operator_share_rate: Target,

    // output
    success: BoolTarget,
}

impl L2UpdatePublicPoolTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdatePublicPoolTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            public_pool_index: builder.add_virtual_target(),
            status: builder.add_virtual_target(),
            operator_fee: builder.add_virtual_target(),
            min_operator_share_rate: builder.add_virtual_target(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdatePublicPoolTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_PUBLIC_POOL)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.public_pool_index,
            self.status,
            self.operator_fee,
            self.min_operator_share_rate,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdatePublicPoolTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_public_pool;
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
        builder.conditional_assert_eq(
            is_enabled,
            self.public_pool_index,
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
        );

        // Ensure given account is a pool account
        let public_pool_account_type =
            builder.constant(F::from_canonical_u8(PUBLIC_POOL_ACCOUNT_TYPE));
        let insurance_fund_account_type =
            builder.constant(F::from_canonical_u8(INSURANCE_FUND_ACCOUNT_TYPE));
        let is_public_pool = builder.is_equal(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            public_pool_account_type,
        );
        let is_insurance_fund_pool = builder.is_equal(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            insurance_fund_account_type,
        );
        let is_pool = builder.or(is_public_pool, is_insurance_fund_pool);
        builder.conditional_assert_true(is_enabled, is_pool);

        // Ensure master account and pool account has the same master account index
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.accounts[MASTER_ACCOUNT_ID].account_index,
            tx_state.accounts[SUB_ACCOUNT_ID].master_account_index,
        );

        // Validate status. Valid values are ACTIVE_PUBLIC_POOL(0) and FROZEN_PUBLIC_POOL(1).
        builder.assert_bool(BoolTarget::new_unsafe(self.status));

        // Validate operator fee
        builder.register_range_check(self.operator_fee, 24);
        builder.conditional_assert_lte(
            is_enabled,
            self.operator_fee,
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_fee,
            32,
        );
        // No need to check new_operator_fee < fee_tick, as it's checked in create_public_pool_tx,
        // and we cap it here by the existing operator fee.

        // Validate min operator share rate
        let share_tick = builder.constant(F::from_canonical_u64(SHARE_TICK));
        builder.register_range_check(self.min_operator_share_rate, SHARE_RATE_BITS);
        builder.conditional_assert_lte(
            is_enabled,
            self.min_operator_share_rate,
            share_tick,
            SHARE_RATE_BITS,
        );

        // Ensure that the public pool is ACTIVE_PUBLIC_POOL(0)
        builder.conditional_assert_zero(
            is_enabled,
            tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info.status,
        );

        let pool_is_healthy = tx_state.risk_infos[SUB_ACCOUNT_ID]
            .cross_risk_parameters
            .is_healthy(builder);
        let are_pool_positions_empty = builder.is_zero_biguint(
            &tx_state.risk_infos[SUB_ACCOUNT_ID]
                .cross_risk_parameters
                .initial_margin_requirement,
        );
        let are_pool_orders_empty =
            builder.is_zero(tx_state.accounts[SUB_ACCOUNT_ID].total_order_count);
        let can_freeze_pool = builder.multi_and(&[
            pool_is_healthy,
            are_pool_positions_empty,
            are_pool_orders_empty,
        ]);
        let is_enabled_for_freeze = builder.and(is_enabled, BoolTarget::new_unsafe(self.status));
        builder.conditional_assert_true(is_enabled_for_freeze, can_freeze_pool);

        // If operator share rate is below the new minimum operator share rate, fail the transaction
        let total_shares = builder.target_to_biguint(
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .total_shares,
        );
        let new_min_operator_share_rate = builder.target_to_biguint(self.min_operator_share_rate);
        let operator_shares = builder.target_to_biguint(
            tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .operator_shares,
        );
        let share_tick = builder.constant_biguint(&BigUint::from(SHARE_TICK));
        let lhs = builder.mul_biguint(&total_shares, &new_min_operator_share_rate);
        let rhs = builder.mul_biguint(&operator_shares, &share_tick);
        let is_min_operator_share_rate_valid = builder.is_lte_biguint(&lhs, &rhs);
        builder.conditional_assert_true(is_enabled, is_min_operator_share_rate_valid);
    }
}

impl Apply for L2UpdatePublicPoolTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let public_pool_info = &PublicPoolInfoTarget {
            status: self.status,
            operator_fee: self.operator_fee,
            min_operator_share_rate: self.min_operator_share_rate,
            strategies: tx_state.accounts[SUB_ACCOUNT_ID]
                .public_pool_info
                .strategies
                .clone(),
            ..tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info
        };
        tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info = select_public_pool_info_target(
            builder,
            self.success,
            public_pool_info,
            &tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info,
        );

        let is_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[SUB_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_llp = builder.is_equal(
            tx_state.accounts[SUB_ACCOUNT_ID].account_index,
            tx_state.system_config.liquidity_pool_index,
        );
        let is_frozen = builder.is_equal_constant(
            tx_state.accounts[SUB_ACCOUNT_ID].public_pool_info.status,
            FROZEN_PUBLIC_POOL as u64,
        );
        let clear_llp_index_flag =
            builder.multi_and(&[self.success, is_insurance_fund, is_llp, is_frozen]);
        let nil_account_index = builder.constant_i64(NIL_ACCOUNT_INDEX);
        tx_state.system_config.liquidity_pool_index = builder.select(
            clear_llp_index_flag,
            nil_account_index,
            tx_state.system_config.liquidity_pool_index,
        );

        self.success
    }
}

pub trait L2UpdatePublicPoolTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_public_pool_tx_target(
        &mut self,
        a: &L2UpdatePublicPoolTxTarget,
        b: &L2UpdatePublicPoolTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdatePublicPoolTxTargetWitness<F> for T {
    fn set_l2_update_public_pool_tx_target(
        &mut self,
        a: &L2UpdatePublicPoolTxTarget,
        b: &L2UpdatePublicPoolTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.public_pool_index,
            F::from_canonical_i64(b.public_pool_index),
        )?;
        self.set_target(a.status, F::from_canonical_u8(b.status))?;
        self.set_target(a.operator_fee, F::from_canonical_i64(b.operator_fee))?;
        self.set_target(
            a.min_operator_share_rate,
            F::from_canonical_i64(b.min_operator_share_rate),
        )?;

        Ok(())
    }
}
