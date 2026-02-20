// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::get_available_collateral;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2StrategyTransferTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "fsi", default)]
    pub from_strategy_index: u8,
    #[serde(rename = "tsi", default)]
    pub to_strategy_index: u8,

    #[serde(rename = "a", default)]
    pub asset_index: i16, // 6 bits
    #[serde(rename = "ba", default)]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub amount: BigUint, // 60 bits
}

#[derive(Debug)]
pub struct L2StrategyTransferTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,

    pub from_strategy_index: Target,
    pub to_strategy_index: Target,

    pub asset_index: Target,
    pub amount: BigUintTarget, // 60 bits

    // Helpers
    extended_transfer_amount: BigIntTarget,

    // Output
    success: BoolTarget,
}

impl L2StrategyTransferTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),

            from_strategy_index: builder.add_virtual_target(),
            to_strategy_index: builder.add_virtual_target(),

            amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            asset_index: builder.add_virtual_target(),

            // helpers
            extended_transfer_amount: BigIntTarget::default(),

            // Output
            success: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        builder.register_range_check(self.from_strategy_index, STRATEGY_INDEX_BITS);
        builder.register_range_check(self.to_strategy_index, STRATEGY_INDEX_BITS);

        builder.range_check_biguint(&self.amount, MAX_TRANSFER_BITS);
    }
}

impl TxHash for L2StrategyTransferTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_STRATEGY_TRANSFER)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.from_strategy_index,
            self.to_strategy_index,
            self.asset_index,
        ];

        let mut limbs = self.amount.limbs.clone();
        limbs.resize(BIG_U64_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2StrategyTransferTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_strategy_transfer;
        self.success = is_enabled;

        self.register_range_checks(builder);

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
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );
        ensure_valid_asset_index(builder, is_enabled, self.asset_index);

        let min_sub_account_index = builder.constant_i64(MIN_SUB_ACCOUNT_INDEX);
        builder.conditional_assert_lte(
            is_enabled,
            min_sub_account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            ACCOUNT_INDEX_BITS,
        );

        builder.conditional_assert_not_eq(
            is_enabled,
            self.to_strategy_index,
            self.from_strategy_index,
        );

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        // Only USDC transfers are allowed between strategies for now
        let is_usdc_asset = builder.is_equal_constant(self.asset_index, USDC_ASSET_INDEX);
        builder.conditional_assert_true(is_enabled, is_usdc_asset);

        // Verify against min transfer amount of asset
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &tx_state.assets[TX_ASSET_ID].min_transfer_amount,
            &self.amount,
        );

        // Can only be insurance fund
        let is_insurance_fund_account = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        builder.conditional_assert_true(is_enabled, is_insurance_fund_account);

        // Can not be a frozen pool
        let is_frozen_public_pool = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].public_pool_info.status,
            FROZEN_PUBLIC_POOL as u64,
        );
        builder.conditional_assert_false(is_enabled, is_frozen_public_pool);

        let extended_transfer_amount = builder.mul_biguint_non_carry(
            &self.amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );

        let available_collateral_to_transfer = get_available_collateral(
            builder,
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters, // risk of the "from" strategy
        );
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &extended_transfer_amount,
            &available_collateral_to_transfer,
        );
        self.extended_transfer_amount = builder.biguint_to_bigint(&extended_transfer_amount);
    }
}

impl Apply for L2StrategyTransferTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let from_collateral_diff = builder.neg_bigint(&self.extended_transfer_amount);
        let zero_bigint = builder.zero_bigint();
        for i in 0..NB_STRATEGIES {
            let is_to_strategy = builder.is_equal_constant(self.to_strategy_index, i as u64);
            let to_delta =
                builder.select_bigint(is_to_strategy, &self.extended_transfer_amount, &zero_bigint);
            tx_state.accounts[OWNER_ACCOUNT_ID]
                .public_pool_info
                .strategies[i] = builder.add_bigint_non_carry(
                &tx_state.accounts[OWNER_ACCOUNT_ID]
                    .public_pool_info
                    .strategies[i],
                &to_delta,
                BIG_U96_LIMBS,
            );

            let is_from_strategy = builder.is_equal_constant(self.from_strategy_index, i as u64);
            let from_delta =
                builder.select_bigint(is_from_strategy, &from_collateral_diff, &zero_bigint);
            tx_state.accounts[OWNER_ACCOUNT_ID]
                .public_pool_info
                .strategies[i] = builder.add_bigint_non_carry(
                &tx_state.accounts[OWNER_ACCOUNT_ID]
                    .public_pool_info
                    .strategies[i],
                &from_delta,
                BIG_U96_LIMBS,
            );
        }

        self.success
    }
}

pub trait L2StrategyTransferTxTargetWitness<F: PrimeField64> {
    fn set_l2_strategy_transfer_tx_target(
        &mut self,
        a: &L2StrategyTransferTxTarget,
        b: &L2StrategyTransferTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2StrategyTransferTxTargetWitness<F> for T {
    fn set_l2_strategy_transfer_tx_target(
        &mut self,
        a: &L2StrategyTransferTxTarget,
        b: &L2StrategyTransferTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.from_strategy_index,
            F::from_canonical_u8(b.from_strategy_index),
        )?;
        self.set_target(
            a.to_strategy_index,
            F::from_canonical_u8(b.to_strategy_index),
        )?;
        self.set_biguint_target(&a.amount, &b.amount)?;
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;

        Ok(())
    }
}
