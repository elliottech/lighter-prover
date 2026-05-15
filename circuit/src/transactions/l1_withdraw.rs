// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module:
// 1. unified margin-enabled assets use cross-collateral minus locked balance.
// 2. available balances are computed per product context.
// 3. perps-route withdrawals require margin-enabled assets.

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::liquidation::{BoolOrTarget, get_available_asset_balance};
use crate::tx_interface::{Apply, OnChainPubData, PriorityOperationsPubData, Verify};
use crate::types::account::AccountTarget;
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L1WithdrawTx {
    #[serde(rename = "mai")]
    pub master_account_index: i64,

    #[serde(rename = "i", default)]
    pub account_index: i64,

    #[serde(rename = "ai")]
    pub asset_index: i16, // 6 bits

    #[serde(rename = "rt")]
    pub route_type: u8,

    #[serde(rename = "a", default)]
    pub amount: u64, // 60 bits
}

#[derive(Debug)]
pub struct L1WithdrawTxTarget {
    pub account_index: Target,
    pub master_account_index: Target,
    pub amount: BigUintTarget,
    pub asset_index: Target,
    pub route_type: Target,

    // Output
    extended_amount: BigUintTarget,
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1WithdrawTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            master_account_index: builder.add_virtual_target(),
            amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            asset_index: builder.add_virtual_target(),
            route_type: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
            extended_amount: BigUintTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        builder.assert_bool(BoolTarget::new_unsafe(self.route_type));
        builder.range_check_biguint(&self.amount, MAX_TRANSFER_BITS);
    }
}

impl PriorityOperationsPubData for L1WithdrawTxTarget {
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
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_WITHDRAW),
            add_target(builder, bytes, self.account_index, 48),
            add_target(builder, bytes, self.master_account_index, 48),
            add_target(builder, bytes, self.asset_index, 16),
            add_byte_target_unsafe(bytes, self.route_type),
            add_transfer_usdc_target(builder, bytes, &self.amount),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, byte_count),
        )
    }
}

impl OnChainPubData for L1WithdrawTxTarget {
    fn on_chain_pub_data(
        &self,
        builder: &mut Builder,
        _tx_state: &TxState,
    ) -> (
        BoolTarget,
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    ) {
        let bytes = &mut Vec::<U8Target>::with_capacity(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE);

        add_pub_data_type_target(builder, bytes, ON_CHAIN_PUB_DATA_TYPE_WITHDRAW);
        add_account_index_target(builder, bytes, self.master_account_index);
        add_target(builder, bytes, self.asset_index, 16);
        add_transfer_usdc_target(builder, bytes, &self.amount);

        (self.success, pad_on_chain_pub_data_target(builder, bytes))
    }
}

impl Verify for L1WithdrawTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l1_withdraw;
        self.is_enabled = is_enabled;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        builder.conditional_assert_eq(
            is_enabled,
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );
        ensure_valid_asset_index(builder, is_enabled, self.asset_index);

        self.register_range_checks(builder);

        // Withdraw amount checks - not zero and 60 bits in total
        builder.conditional_assert_not_zero_biguint(is_enabled, &self.amount);

        let is_new_account = tx_state.is_new_account[OWNER_ACCOUNT_ID];
        let is_old_account = builder.not(is_new_account);

        let is_master_account_correct = builder.is_equal(
            self.master_account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
        );

        let master_account_type = builder.constant_from_u8(MASTER_ACCOUNT_TYPE);
        let is_master_account_type = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            master_account_type,
        );
        let sub_account_type = builder.constant_from_u8(SUB_ACCOUNT_TYPE);
        let is_sub_account_type = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            sub_account_type,
        );
        let is_valid_account_type = builder.or(is_master_account_type, is_sub_account_type);

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        let is_asset_not_empty = builder.not(is_asset_empty);

        self.success = builder.multi_and(&[
            self.success,
            is_old_account,
            is_valid_account_type,
            is_master_account_correct,
            is_asset_not_empty,
        ]);

        // Withdrawal balance checks
        self.extended_amount = builder.mul_biguint_non_carry(
            &self.amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );
        let is_spot = builder.is_equal_constant(self.route_type, ROUTE_TYPE_SPOT);
        let is_perps = builder.not(is_spot);
        let product_type = builder.select_constant(is_spot, PRODUCT_TYPE_SPOT, PRODUCT_TYPE_PERPS);

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
        let not_enough_balance = builder.is_gt_biguint(&self.extended_amount, &available_balance);
        let should_be_false = builder.and(self.success, not_enough_balance);
        self.success = builder.and_not(self.success, should_be_false);
        // === end of statement 2 ===

        // =========================================
        // statement 3: perps-route withdrawals require margin-enabled assets.
        // =========================================
        let perps_flag = builder.and(self.success, is_perps);
        let is_asset_margin_enabled =
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID];
        let should_be_false = builder.and_not(perps_flag, is_asset_margin_enabled);
        self.success = builder.and_not(self.success, should_be_false);
        // === end of statement 3 ===
    }
}

impl Apply for L1WithdrawTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let is_route_type_spot = builder.is_equal_constant(self.route_type, ROUTE_TYPE_SPOT);
        let product_type =
            builder.select_constant(is_route_type_spot, PRODUCT_TYPE_SPOT, PRODUCT_TYPE_PERPS);
        let withdraw_delta = builder.negative_biguint(&self.extended_amount);

        let is_unified = tx_state.accounts[OWNER_ACCOUNT_ID].is_unified_mode();
        let _false = builder._false();
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

pub trait L1WithdrawTxTargetWitness<F: PrimeField64> {
    fn set_l1_withdraw_tx_target(&mut self, a: &L1WithdrawTxTarget, b: &L1WithdrawTx)
    -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1WithdrawTxTargetWitness<F> for T {
    fn set_l1_withdraw_tx_target(
        &mut self,
        a: &L1WithdrawTxTarget,
        b: &L1WithdrawTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )?;
        self.set_biguint_target(&a.amount, &BigUint::from(b.amount))?;
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;
        self.set_target(a.route_type, F::from_canonical_u8(b.route_type))?;

        Ok(())
    }
}
