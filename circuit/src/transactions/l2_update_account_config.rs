// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module:
// 1. only valid accounts (master/sub) can switch balance modes.
// 2. switching to unified requires no locked USDC.
// 3. switching to simple requires healthy account, no locked USDC, and non-negative collateral-with-funding.
// 4. switching to unified applies USDC-to-collateral updates.

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
// A transaction that updates the `AccountConfig`
pub struct L2UpdateAccountConfigTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    // "dtm" kept for consistency with existing witness tags.
    #[serde(rename = "dtm")]
    pub account_trading_mode: u8,
}

#[derive(Debug)]
pub struct L2UpdateAccountConfigTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub account_trading_mode: Target,
    // output
    pub success: BoolTarget,
}

impl L2UpdateAccountConfigTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdateAccountConfigTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            account_trading_mode: builder.add_virtual_target(),

            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateAccountConfigTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_ACCOUNT_CONFIG)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.account_trading_mode,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateAccountConfigTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_account_config;
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

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        // Valid values: 0 (simple assets) or 1 (unified asset)
        builder.assert_bool(BoolTarget::new_unsafe(self.account_trading_mode));

        // Master and sub accounts can update their account config
        let is_master = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            MASTER_ACCOUNT_TYPE as u64,
        );
        let is_sub = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            SUB_ACCOUNT_TYPE as u64,
        );
        let is_master_or_sub = builder.or(is_master, is_sub);
        builder.conditional_assert_true(is_enabled, is_master_or_sub);

        // Make sure account index isn't insurance fund operator or treasury
        let treasury_account_index = builder.constant_u64(TREASURY_ACCOUNT_INDEX as u64);
        builder.conditional_assert_not_eq(is_enabled, self.account_index, treasury_account_index);
        let insurance_fund_operator_index =
            builder.constant_u64(INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX as u64);
        builder.conditional_assert_not_eq(
            is_enabled,
            self.account_index,
            insurance_fund_operator_index,
        );

        // Balance mode should change
        let is_mode_same = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode,
            self.account_trading_mode,
        );
        builder.conditional_assert_false(is_enabled, is_mode_same);

        // =========================================
        // statement 3: switching to simple requires no usdc spot orders, a healthy account,
        // a non-negative collateral-with-funding, and no margin enabled assets
        // =========================================
        let is_simple_mode = builder.is_equal_constant(
            self.account_trading_mode,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE as u64,
        );
        let is_enabled_simple = builder.and(is_enabled, is_simple_mode);

        // No locked USDC balance, means no open spot orders
        builder.conditional_assert_zero_biguint(
            is_enabled_simple,
            &tx_state.account_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].locked_balance,
        );

        let is_healthy = tx_state.risk_infos[OWNER_ACCOUNT_ID]
            .cross_risk_parameters
            .is_healthy(builder);
        builder.conditional_assert_true(is_enabled_simple, is_healthy);

        let collateral_with_funding_negative = builder.is_sign_negative(
            tx_state.risk_infos[OWNER_ACCOUNT_ID]
                .cross_risk_parameters
                .usdc_collateral_with_funding
                .sign,
        );
        builder.conditional_assert_false(is_enabled_simple, collateral_with_funding_negative);

        // Make sure that no asset is margin enabled when switching to simple mode. USDC(universal) asset is skipped
        let mut is_margin_enabled_asset_exists = builder._false();
        tx_state.accounts[OWNER_ACCOUNT_ID]
            .margined_assets
            .iter()
            .skip(1)
            .for_each(|asset| {
                let is_used_as_margin = BoolTarget::new_unsafe(asset.margin_mode); // 1 - margin enabled, 0 - not margin enabled
                is_margin_enabled_asset_exists =
                    builder.or(is_margin_enabled_asset_exists, is_used_as_margin);
            });
        builder.conditional_assert_false(is_enabled_simple, is_margin_enabled_asset_exists);
        // === end of statement 3 ===
    }
}

impl Apply for L2UpdateAccountConfigTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let zero_biguint = builder.zero_biguint();

        // Transfer spot to perps balance for switching to unified mode. Note that account is in simple mode here.
        {
            let is_unified_mode = BoolTarget::new_unsafe(self.account_trading_mode);
            let flag = builder.and(self.success, is_unified_mode);

            let balance_bigint = builder.biguint_to_bigint(
                &tx_state.account_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
            );
            let new_margined_balance = builder.add_bigint_non_carry(
                &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
                &balance_bigint,
                BIG_U96_LIMBS,
            );
            tx_state.account_margined_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance =
                builder.select_bigint(
                    flag,
                    &new_margined_balance,
                    &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
                );

            tx_state.account_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance = builder
                .select_biguint(
                    flag,
                    &zero_biguint,
                    &tx_state.account_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
                );
        }

        tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode = builder.select(
            self.success,
            self.account_trading_mode,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode,
        );

        self.success
    }
}

pub trait L2UpdateAccountConfigTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_account_config_tx_target(
        &mut self,
        a: &L2UpdateAccountConfigTxTarget,
        b: &L2UpdateAccountConfigTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateAccountConfigTxTargetWitness<F> for T {
    fn set_l2_update_account_config_tx_target(
        &mut self,
        a: &L2UpdateAccountConfigTxTarget,
        b: &L2UpdateAccountConfigTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.account_trading_mode,
            F::from_canonical_u8(b.account_trading_mode),
        )?;

        Ok(())
    }
}
