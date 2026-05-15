// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module:
// 1. unified margin-enabled assets use cross-collateral minus locked balance.
// 2. available balances are computed per product context.
// 3. perps-route withdrawals require margin-enabled assets.

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{BoolOrTarget, get_available_asset_balance};
use crate::tx_interface::{Apply, OnChainPubData, TransactionTarget, TxHash, Verify};
use crate::types::account::AccountTarget;
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2WithdrawTx {
    #[serde(rename = "f", default)]
    pub account_index: i64,

    #[serde(rename = "a", default)]
    pub api_key_index: u8,

    #[serde(rename = "ai", default)]
    pub asset_index: i16, // 6 bits

    #[serde(rename = "rt", default)]
    pub route_type: u8,

    #[serde(rename = "u", default)]
    pub amount: u64, // 60 bits
}

#[derive(Debug)]
pub struct L2WithdrawTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub amount: BigUintTarget,
    pub asset_index: Target,
    pub route_type: Target,

    // Output
    success: BoolTarget,
    extended_amount: BigUintTarget,
}

impl L2WithdrawTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            asset_index: builder.add_virtual_target(),
            route_type: builder.add_virtual_target(),

            success: BoolTarget::default(),
            extended_amount: BigUintTarget::default(),
        }
    }
}

impl TxHash for L2WithdrawTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_WITHDRAW)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.asset_index,
            self.route_type,
        ];

        let limbs = self.amount.limbs.clone();
        for limb in limbs {
            elements.push(limb.0);
        }

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl OnChainPubData for L2WithdrawTxTarget {
    fn on_chain_pub_data(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
    ) -> (
        BoolTarget,
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    ) {
        let bytes = &mut Vec::<U8Target>::with_capacity(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE);

        add_pub_data_type_target(builder, bytes, ON_CHAIN_PUB_DATA_TYPE_WITHDRAW);
        add_account_index_target(
            builder,
            bytes,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
        );
        add_target(builder, bytes, self.asset_index, 16);
        add_transfer_usdc_target(builder, bytes, &self.amount);

        (self.success, pad_on_chain_pub_data_target(builder, bytes))
    }
}

impl Verify for L2WithdrawTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_withdraw;
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
        ensure_valid_asset_index(builder, is_enabled, self.asset_index);

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_asset_empty);

        builder.assert_bool(BoolTarget::new_unsafe(self.route_type));
        builder.range_check_biguint(&self.amount, MAX_TRANSFER_BITS);

        builder.conditional_assert_not_zero_biguint(is_enabled, &self.amount);
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &tx_state.assets[TX_ASSET_ID].min_withdrawal_amount,
            &self.amount,
        );

        self.extended_amount = builder.mul_biguint_non_carry(
            &self.amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );

        let is_route_perps = builder.is_equal_constant(self.route_type, ROUTE_TYPE_PERPS);
        let is_route_spot = builder.not(is_route_perps);
        let product_type =
            builder.select_constant(is_route_spot, PRODUCT_TYPE_SPOT, PRODUCT_TYPE_PERPS);

        // =========================================
        // statement 1: unified margin-enabled assets use cross-collateral minus locked balance.
        // =========================================
        let available_balance = get_available_asset_balance(
            builder,
            product_type,
            self.asset_index,
            &tx_state.accounts[OWNER_ACCOUNT_ID],
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.margined_asset[TX_ASSET_ID],
            &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            BoolOrTarget::False,
        );
        // === end of statement 1 ===

        // =========================================
        // statement 2: available balances are computed per product context.
        // =========================================
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &self.extended_amount,
            &available_balance,
        );
        // === end of statement 2 ===

        // =========================================
        // statement 3: perps-route withdrawals require margin-enabled assets.
        // =========================================
        let perps_flag = builder.and(is_enabled, is_route_perps);
        let is_asset_margin_enabled =
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID];
        builder.conditional_assert_true(perps_flag, is_asset_margin_enabled);
        // === end of statement 3 ===
    }
}

impl Apply for L2WithdrawTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let is_route_spot = builder.is_equal_constant(self.route_type, ROUTE_TYPE_SPOT);
        let product_type =
            builder.select_constant(is_route_spot, PRODUCT_TYPE_SPOT, PRODUCT_TYPE_PERPS);
        let withdraw_delta = builder.negative_biguint(&self.extended_amount);

        let _false = builder._false();
        let is_unified = tx_state.accounts[OWNER_ACCOUNT_ID].is_unified_mode();
        AccountTarget::apply_asset_delta(
            builder,
            self.success,
            product_type,
            self.asset_index,
            &mut tx_state.margined_asset[TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            &withdraw_delta,
            is_unified,
            _false,
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
            false,
        );

        self.success
    }
}

pub trait L2WithdrawTxTargetWitness<F: PrimeField64> {
    fn set_l2_withdraw_tx_target(
        &mut self,
        a: &TransactionTarget<L2WithdrawTxTarget>,
        b: &L2WithdrawTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2WithdrawTxTargetWitness<F> for T {
    fn set_l2_withdraw_tx_target(
        &mut self,
        a: &TransactionTarget<L2WithdrawTxTarget>,
        b: &L2WithdrawTx,
    ) -> Result<()> {
        let a = &a.inner;

        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_biguint_target(&a.amount, &BigUint::from(b.amount))?;
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;
        self.set_target(a.route_type, F::from_canonical_u8(b.route_type))?;

        Ok(())
    }
}
