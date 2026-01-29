// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::system_config::{SystemConfigTarget, select_system_config_target};
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct L1SetSystemConfigTx {
    #[serde(rename = "llpai", default)]
    pub liquidity_pool_index: i64,
    #[serde(rename = "lspai", default)]
    pub staking_pool_index: i64,
    #[serde(rename = "mbps", default)]
    pub liquidity_pool_cooldown_period: i64,
    #[serde(rename = "spwlm", default)]
    pub staking_pool_lockup_period: i64,
}

#[derive(Debug)]
pub struct L1SetSystemConfigTxTarget {
    pub liquidity_pool_index: Target,
    pub staking_pool_index: Target,
    pub liquidity_pool_cooldown_period: Target,
    pub staking_pool_lockup_period: Target,

    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1SetSystemConfigTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            liquidity_pool_index: builder.add_virtual_target(),
            staking_pool_index: builder.add_virtual_target(),
            liquidity_pool_cooldown_period: builder.add_virtual_target(),
            staking_pool_lockup_period: builder.add_virtual_target(),

            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }
}

impl PriorityOperationsPubData for L1SetSystemConfigTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let byte_count = [
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_SET_SYSTEM_CONFIG),
            add_account_index_target(builder, bytes, self.liquidity_pool_index),
            add_account_index_target(builder, bytes, self.staking_pool_index),
            add_target(builder, bytes, self.liquidity_pool_cooldown_period, 48),
            add_target(builder, bytes, self.staking_pool_lockup_period, 48),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, byte_count),
        )
    }
}

impl Verify for L1SetSystemConfigTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_types: &TxTypeTargets, tx_state: &TxState) {
        self.success = tx_types.is_l1_set_system_config;
        self.is_enabled = tx_types.is_l1_set_system_config;

        // State leaves
        builder.conditional_assert_eq(
            self.is_enabled,
            self.liquidity_pool_index,
            tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            self.is_enabled,
            self.staking_pool_index,
            tx_state.accounts[STAKING_POOL_ACCOUNT_ID].account_index,
        );

        // Min burn period
        let max_liquidity_pool_cooldown_period =
            builder.constant_i64(MAX_LIQUIDITY_POOL_COOLDOWN_PERIOD);
        builder.conditional_assert_lte(
            self.is_enabled,
            self.liquidity_pool_cooldown_period,
            max_liquidity_pool_cooldown_period,
            64,
        );

        // Staking pool withdrawal latency
        let max_staking_pool_lockup_period = builder.constant_i64(MAX_STAKING_POOL_LOCKUP_PERIOD);
        builder.conditional_assert_lte(
            self.is_enabled,
            self.staking_pool_lockup_period,
            max_staking_pool_lockup_period,
            64,
        );

        // Llp account validations
        {
            // Must be insurance fund account type
            let is_insurance_fund = builder.is_equal_constant(
                tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            self.success = builder.and(self.success, is_insurance_fund);
            // Must be an active pool
            let is_active_pool = builder.is_equal_constant(
                tx_state.accounts[LIQUIDITY_POOL_ACCOUNT_ID]
                    .public_pool_info
                    .status,
                ACTIVE_PUBLIC_POOL as u64,
            );
            self.success = builder.and(self.success, is_active_pool);
        }

        // Lsp account validations
        {
            // Must be staking pool account type
            let is_staking_pool = builder.is_equal_constant(
                tx_state.accounts[STAKING_POOL_ACCOUNT_ID].account_type,
                LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
            );
            self.success = builder.and(self.success, is_staking_pool);
            // Must be an active pool
            let is_active_pool = builder.is_equal_constant(
                tx_state.accounts[STAKING_POOL_ACCOUNT_ID]
                    .public_pool_info
                    .status,
                ACTIVE_PUBLIC_POOL as u64,
            );
            self.success = builder.and(self.success, is_active_pool);
        }
    }
}

impl Apply for L1SetSystemConfigTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.system_config = select_system_config_target(
            builder,
            self.success,
            &SystemConfigTarget {
                liquidity_pool_index: self.liquidity_pool_index,
                staking_pool_index: self.staking_pool_index,
                liquidity_pool_cooldown_period: self.liquidity_pool_cooldown_period,
                staking_pool_lockup_period: self.staking_pool_lockup_period,
            },
            &tx_state.system_config,
        );

        self.success
    }
}

pub trait L1SetSystemConfigTxTargetWitness<F: PrimeField64> {
    fn set_l1_set_system_config_tx_target(
        &mut self,
        a: &L1SetSystemConfigTxTarget,
        b: &L1SetSystemConfigTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1SetSystemConfigTxTargetWitness<F> for T {
    fn set_l1_set_system_config_tx_target(
        &mut self,
        a: &L1SetSystemConfigTxTarget,
        b: &L1SetSystemConfigTx,
    ) -> Result<()> {
        self.set_target(
            a.liquidity_pool_index,
            F::from_canonical_i64(b.liquidity_pool_index),
        )?;
        self.set_target(
            a.staking_pool_index,
            F::from_canonical_i64(b.staking_pool_index),
        )?;
        self.set_target(
            a.liquidity_pool_cooldown_period,
            F::from_canonical_i64(b.liquidity_pool_cooldown_period),
        )?;
        self.set_target(
            a.staking_pool_lockup_period,
            F::from_canonical_i64(b.staking_pool_lockup_period),
        )?;

        Ok(())
    }
}
