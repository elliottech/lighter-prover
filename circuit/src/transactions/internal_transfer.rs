// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::tx_interface::{Apply, Verify};
use crate::types::account::AccountTarget;
use crate::types::asset::is_universal_asset;
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct InternalTransferTx {
    #[serde(rename = "f", default)]
    pub from_account_index: i64,
    #[serde(rename = "t", default)]
    pub to_account_index: i64,
    #[serde(rename = "ai", default)]
    pub asset_index: i16,
    #[serde(rename = "si", default)]
    pub strategy_index: u8,
    #[serde(rename = "rt", default)]
    pub route_type: u8,
    #[serde(rename = "ba", default)]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub amount: BigUint,
}

#[derive(Debug)]
pub struct InternalTransferTxTarget {
    pub from_account_index: Target,
    pub to_account_index: Target,
    pub asset_index: Target,
    pub route_type: Target,
    pub strategy_index: Target,
    pub amount: BigUintTarget, // 60 bits

    // helpers
    extended_transfer_amount: BigIntTarget,
    is_spot: BoolTarget,

    // outputs
    success: BoolTarget,
}

impl InternalTransferTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            from_account_index: builder.add_virtual_target(),
            to_account_index: builder.add_virtual_target(),
            asset_index: builder.add_virtual_target(),
            route_type: builder.add_virtual_target(),
            strategy_index: builder.add_virtual_target(),
            amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),

            // helpers
            extended_transfer_amount: BigIntTarget::default(),
            is_spot: BoolTarget::default(),

            // outputs
            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalTransferTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_transfer;
        self.success = is_enabled;

        builder.conditional_assert_eq_constant(
            self.success,
            tx_state.register_stack[0].instruction_type,
            TRANSFER_ASSET as u64,
        );

        builder.conditional_assert_eq(
            self.success,
            self.from_account_index,
            tx_state.accounts[SENDER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            self.success,
            tx_state.register_stack[0].account_index, // Fee collector when trade happened
            tx_state.accounts[SENDER_ACCOUNT_ID].account_index,
        );
        self.success = builder.and_not(self.success, tx_state.is_new_account[SENDER_ACCOUNT_ID]);

        builder.conditional_assert_eq(
            self.success,
            self.to_account_index,
            tx_state.accounts[RECEIVER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            self.success,
            tx_state.accounts[RECEIVER_ACCOUNT_ID].account_index,
            tx_state.register_stack[0].generic_field_0,
        );
        self.success = builder.and_not(self.success, tx_state.is_new_account[RECEIVER_ACCOUNT_ID]);

        builder.conditional_assert_eq(
            self.success,
            self.asset_index,
            tx_state.register_stack[0].generic_field_2,
        );
        builder.conditional_assert_eq(
            self.success,
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );

        builder.conditional_assert_eq(
            self.success,
            self.route_type,
            tx_state.register_stack[0].pending_type,
        );

        builder.range_check_biguint(&self.amount, MAX_TRANSFER_BITS);
        let amount_target = builder.biguint_to_target_unsafe(&self.amount);
        builder.conditional_assert_eq(
            self.success,
            amount_target,
            tx_state.register_stack[0].pending_size,
        );

        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        self.success = builder.and_not(self.success, is_asset_empty);

        let is_same_account = builder.is_equal(self.from_account_index, self.to_account_index);
        self.success = builder.and_not(self.success, is_same_account);

        let nil_strategy_index = builder.constant_usize(NIL_STRATEGY_INDEX);
        let is_invalid_strategy_index = builder.is_gte(self.strategy_index, nil_strategy_index, 64);
        self.success = builder.and_not(self.success, is_invalid_strategy_index);

        let extended_transfer_amount_big = builder.mul_biguint_non_carry(
            &self.amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );
        self.extended_transfer_amount = builder.biguint_to_bigint(&extended_transfer_amount_big);

        self.is_spot = BoolTarget::new_unsafe(self.route_type);
    }
}

impl Apply for InternalTransferTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let product_type =
            builder.select_constant(self.is_spot, PRODUCT_TYPE_SPOT, PRODUCT_TYPE_PERPS);
        let is_asset_universal = is_universal_asset(builder, self.asset_index);

        let sender_asset_delta = builder.neg_bigint(&self.extended_transfer_amount);

        let mut margin_asset = tx_state.margined_asset[TX_ASSET_ID].clone();

        let mut sender_account_asset =
            tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].clone();
        let mut sender_account_margined_asset =
            tx_state.account_margined_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].clone();
        let mut sender_strategy = tx_state.strategies[SENDER_ACCOUNT_ID].clone();

        let mut receiver_account_asset =
            tx_state.account_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].clone();
        let mut receiver_account_margined_asset =
            tx_state.account_margined_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].clone();
        let mut receiver_strategy = tx_state.strategies[RECEIVER_ACCOUNT_ID].clone();

        let is_sender_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[SENDER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_sender_unified = tx_state.accounts[SENDER_ACCOUNT_ID].is_unified_mode();
        let is_sender_spot_balance_valid = AccountTarget::apply_asset_delta(
            builder,
            self.success,
            product_type,
            self.asset_index,
            &mut margin_asset,
            tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][TX_ASSET_ID],
            &sender_asset_delta,
            is_sender_unified,
            is_sender_insurance_fund,
            &mut sender_account_asset.balance,
            &mut sender_account_margined_asset.balance,
            &mut sender_strategy,
            true,
        );

        let is_receiver_insurance_fund = builder.is_equal_constant(
            tx_state.accounts[RECEIVER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let is_receiver_unified = tx_state.accounts[RECEIVER_ACCOUNT_ID].is_unified_mode();
        let is_receiver_spot_balance_valid = AccountTarget::apply_asset_delta(
            builder,
            self.success,
            product_type,
            self.asset_index,
            &mut margin_asset,
            tx_state.is_asset_used_as_margin[RECEIVER_ACCOUNT_ID][TX_ASSET_ID],
            &self.extended_transfer_amount,
            is_receiver_unified,
            is_receiver_insurance_fund,
            &mut receiver_account_asset.balance,
            &mut receiver_account_margined_asset.balance,
            &mut receiver_strategy,
            true,
        );

        self.success = builder.and(self.success, is_sender_spot_balance_valid);
        let (success, sender_account_spot_balance) =
            builder.try_trim_biguint(&sender_account_asset.balance, BIG_U96_LIMBS);
        self.success = builder.and(self.success, success);
        let (success, _) =
            builder.try_trim_biguint(&sender_account_margined_asset.balance.abs, BIG_U96_LIMBS);
        self.success = builder.and(self.success, success);
        let sender_is_margin_balance_negative =
            builder.is_sign_negative(sender_account_margined_asset.balance.sign);
        let should_be_false =
            builder.and_not(sender_is_margin_balance_negative, is_asset_universal);
        self.success = builder.and_not(self.success, should_be_false);

        self.success = builder.and(self.success, is_receiver_spot_balance_valid);
        let (success, receiver_account_spot_balance) =
            builder.try_trim_biguint(&receiver_account_asset.balance, BIG_U96_LIMBS);
        self.success = builder.and(self.success, success);
        let (success, _) =
            builder.try_trim_biguint(&receiver_account_margined_asset.balance.abs, BIG_U96_LIMBS);
        self.success = builder.and(self.success, success);
        let receiver_is_margin_balance_negative =
            builder.is_sign_negative(receiver_account_margined_asset.balance.sign);
        let should_be_false =
            builder.and_not(receiver_is_margin_balance_negative, is_asset_universal);
        self.success = builder.and_not(self.success, should_be_false);

        // Apply sender changes
        tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &sender_account_spot_balance,
            &tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );
        tx_state.account_margined_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance = builder
            .select_bigint(
                self.success,
                &sender_account_margined_asset.balance,
                &tx_state.account_margined_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance,
            );
        tx_state.strategies[SENDER_ACCOUNT_ID] = builder.select_bigint(
            self.success,
            &sender_strategy,
            &tx_state.strategies[SENDER_ACCOUNT_ID],
        );

        // Apply receiver changes
        tx_state.account_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            self.success,
            &receiver_account_spot_balance,
            &tx_state.account_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );
        tx_state.account_margined_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].balance = builder
            .select_bigint(
                self.success,
                &receiver_account_margined_asset.balance,
                &tx_state.account_margined_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].balance,
            );
        tx_state.strategies[RECEIVER_ACCOUNT_ID] = builder.select_bigint(
            self.success,
            &receiver_strategy,
            &tx_state.strategies[RECEIVER_ACCOUNT_ID],
        );

        // Update margined asset info's total supplied amount
        tx_state.margined_asset[TX_ASSET_ID].total_supplied_amount = builder.select_biguint(
            self.success,
            &margin_asset.total_supplied_amount,
            &tx_state.margined_asset[TX_ASSET_ID].total_supplied_amount,
        );

        tx_state.register_stack.pop_front(builder, self.success);

        self.success
    }
}

pub trait InternalTransferTxTargetWitness<F: PrimeField64> {
    fn set_internal_transfer_tx_target(
        &mut self,
        a: &InternalTransferTxTarget,
        b: &InternalTransferTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalTransferTxTargetWitness<F> for T {
    fn set_internal_transfer_tx_target(
        &mut self,
        a: &InternalTransferTxTarget,
        b: &InternalTransferTx,
    ) -> Result<()> {
        self.set_target(
            a.from_account_index,
            F::from_canonical_i64(b.from_account_index),
        )?;
        self.set_target(
            a.to_account_index,
            F::from_canonical_i64(b.to_account_index),
        )?;
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;
        self.set_target(a.route_type, F::from_canonical_u8(b.route_type))?;
        self.set_target(a.strategy_index, F::from_canonical_u8(b.strategy_index))?;
        self.set_biguint_target(&a.amount, &b.amount)?;

        Ok(())
    }
}
