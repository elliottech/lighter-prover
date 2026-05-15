// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{BoolOrTarget, get_available_asset_balance};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::account::AccountTarget;
use crate::types::asset::is_universal_asset;
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UpdateAccountAssetConfigTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "asti", default)]
    pub asset_index: i16,

    #[serde(rename = "amm", default)]
    pub asset_margin_mode: u8,
}

#[derive(Debug)]
pub struct L2UpdateAccountAssetConfigTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,

    pub asset_index: Target,
    pub asset_margin_mode: Target,

    // Output
    pub success: BoolTarget,
}

impl L2UpdateAccountAssetConfigTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdateAccountAssetConfigTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),

            asset_index: builder.add_virtual_target(),
            asset_margin_mode: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateAccountAssetConfigTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_ACCOUNT_ASSET_CONFIG)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.asset_index,
            self.asset_margin_mode,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateAccountAssetConfigTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_account_asset_config;
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
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );

        let is_enabling_margin_mode = BoolTarget::new_unsafe(self.asset_margin_mode);
        builder.conditional_assert_bool(is_enabled, is_enabling_margin_mode);

        // Asset may not be empty
        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        // Asset may not be universal
        let is_universal_asset = is_universal_asset(builder, self.asset_index);
        builder.conditional_assert_false(is_enabled, is_universal_asset);

        // Only master and sub accounts
        builder.conditional_assert_bool(
            is_enabled,
            BoolTarget::new_unsafe(tx_state.accounts[OWNER_ACCOUNT_ID].account_type),
        );

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

        // Account must be UTA
        builder.conditional_assert_true(
            is_enabled,
            tx_state.accounts[OWNER_ACCOUNT_ID].is_unified_mode(),
        );

        // Given margin mode must be different from current margin mode
        let is_same_margin_mode = builder.is_equal(
            self.asset_margin_mode,
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID].target,
        );
        builder.conditional_assert_false(is_enabled, is_same_margin_mode);

        // Asset must be margin-enabled
        builder.conditional_assert_true(
            is_enabled,
            BoolTarget::new_unsafe(tx_state.assets[TX_ASSET_ID].margin_mode),
        );

        // Asset must have available balance to cover for margin balance when disabling margin mode
        let _perps = builder.constant_u64(PRODUCT_TYPE_PERPS);
        let base_asset_available_balance = get_available_asset_balance(
            builder,
            _perps,
            tx_state.asset_indices[BASE_ASSET_ID],
            &tx_state.accounts[OWNER_ACCOUNT_ID],
            &tx_state.account_assets[OWNER_ACCOUNT_ID][BASE_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][BASE_ASSET_ID],
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.margined_asset[BASE_ASSET_ID],
            &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][BASE_ASSET_ID].balance,
            BoolOrTarget::True,
        );
        let is_margin_balance_negative = builder.is_sign_negative(
            tx_state.account_margined_assets[OWNER_ACCOUNT_ID][BASE_ASSET_ID]
                .balance
                .sign,
        );
        let is_abs_margin_balance_lte_available_balance = builder.is_lte_biguint(
            &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][BASE_ASSET_ID]
                .balance
                .abs,
            &base_asset_available_balance,
        );
        let is_margin_balance_lte_available_balance = builder.or(
            is_abs_margin_balance_lte_available_balance,
            is_margin_balance_negative,
        );
        let should_be_true = builder.or(
            is_enabling_margin_mode,
            is_margin_balance_lte_available_balance,
        );
        builder.conditional_assert_true(is_enabled, should_be_true);
    }
}

impl Apply for L2UpdateAccountAssetConfigTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let is_enabling_margin_mode = BoolTarget::new_unsafe(self.asset_margin_mode);

        tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].margin_mode = builder
            .select(
                self.success,
                self.asset_margin_mode,
                tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].margin_mode,
            );

        let delta = {
            let asset_balance_bigint = builder
                .biguint_to_bigint(&tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance);
            builder.select_bigint(
                is_enabling_margin_mode,
                &asset_balance_bigint,
                &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            )
        };
        let product_type = builder.not(is_enabling_margin_mode).target;

        let neg_delta = builder.neg_bigint(&delta);
        let neg_product_type = is_enabling_margin_mode.target;

        AccountTarget::apply_asset_delta_raw(
            builder,
            self.success,
            neg_product_type,
            self.asset_index,
            &mut tx_state.margined_asset[TX_ASSET_ID],
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &neg_delta,
            &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            false, // Can't be universal asset
        );
        let _true = builder._true();
        let _false = builder._false();
        AccountTarget::apply_asset_delta(
            builder,
            self.success,
            product_type,
            self.asset_index,
            &mut tx_state.margined_asset[TX_ASSET_ID],
            is_enabling_margin_mode,
            &delta,
            _true,
            _false,
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
            false,
        );

        self.success
    }
}

pub trait L2UpdateAccountAssetConfigTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_account_asset_config_tx_target(
        &mut self,
        a: &L2UpdateAccountAssetConfigTxTarget,
        b: &L2UpdateAccountAssetConfigTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateAccountAssetConfigTxTargetWitness<F> for T {
    fn set_l2_update_account_asset_config_tx_target(
        &mut self,
        a: &L2UpdateAccountAssetConfigTxTarget,
        b: &L2UpdateAccountAssetConfigTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index as i64))?;
        self.set_target(
            a.asset_margin_mode,
            F::from_canonical_u8(b.asset_margin_mode),
        )?;
        Ok(())
    }
}
