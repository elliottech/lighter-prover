// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::asset::is_universal_asset;
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UpdateAssetConfigTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "asti", default)]
    pub asset_index: i16,

    #[serde(rename = "gsc", default)]
    pub global_supply_cap: i64,

    #[serde(rename = "usc", default)]
    pub user_supply_cap: i64,
}

#[derive(Debug)]
pub struct L2UpdateAssetConfigTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,

    pub asset_index: Target,
    pub global_supply_cap: Target,
    pub user_supply_cap: Target,

    // Output
    pub success: BoolTarget,
}

impl L2UpdateAssetConfigTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdateAssetConfigTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),

            asset_index: builder.add_virtual_target(),
            global_supply_cap: builder.add_virtual_target(),
            user_supply_cap: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateAssetConfigTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_ASSET_CONFIG)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.asset_index,
            self.global_supply_cap,
            self.user_supply_cap,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateAssetConfigTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_asset_config;
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

        // Confine to insurance fund operator
        builder.conditional_assert_eq_constant(
            is_enabled,
            self.account_index,
            INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX as u64,
        );

        builder.conditional_assert_eq(
            is_enabled,
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        // Asset must be margin enabled
        builder.conditional_assert_true(
            is_enabled,
            BoolTarget::new_unsafe(tx_state.assets[TX_ASSET_ID].margin_mode),
        );

        // Can't be universal asset
        let is_universal_asset = is_universal_asset(builder, self.asset_index);
        builder.conditional_assert_false(is_enabled, is_universal_asset);

        builder.register_range_check(self.global_supply_cap, GLOBAL_SUPPLY_CAP_BITS);
        builder.register_range_check(self.user_supply_cap, USER_SUPPLY_CAP_BITS);
    }
}

impl Apply for L2UpdateAssetConfigTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let global_supply_cap_big = builder.target_to_biguint(self.global_supply_cap);
        let new_global_supply_cap = builder.mul_biguint_non_carry(
            &global_supply_cap_big,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );
        tx_state.margined_asset[TX_ASSET_ID].global_supply_cap = builder.select_biguint(
            self.success,
            &new_global_supply_cap,
            &tx_state.margined_asset[TX_ASSET_ID].global_supply_cap,
        );

        let user_supply_cap_big = builder.target_to_biguint(self.user_supply_cap);
        let new_user_supply_cap = builder.mul_biguint_non_carry(
            &user_supply_cap_big,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );
        tx_state.margined_asset[TX_ASSET_ID].user_supply_cap = builder.select_biguint(
            self.success,
            &new_user_supply_cap,
            &tx_state.margined_asset[TX_ASSET_ID].user_supply_cap,
        );

        self.success
    }
}

pub trait L2UpdateAssetConfigTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_asset_config_tx_target(
        &mut self,
        a: &L2UpdateAssetConfigTxTarget,
        b: &L2UpdateAssetConfigTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateAssetConfigTxTargetWitness<F> for T {
    fn set_l2_update_asset_config_tx_target(
        &mut self,
        a: &L2UpdateAssetConfigTxTarget,
        b: &L2UpdateAssetConfigTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index as i64))?;
        self.set_target(
            a.global_supply_cap,
            F::from_canonical_i64(b.global_supply_cap),
        )?;
        self.set_target(a.user_supply_cap, F::from_canonical_i64(b.user_supply_cap))?;
        Ok(())
    }
}
