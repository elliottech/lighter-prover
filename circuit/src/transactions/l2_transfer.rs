// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module:
// 1. unified USDC uses cross-collateral minus locked USDC.
// 2. available balances are computed per product context.
// 3. unified accounts apply USDC deltas to collateral where applicable.

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::get_available_asset_balance;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::account::AccountTarget;
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2TransferTx {
    #[serde(rename = "f", default)]
    pub from_account_index: i64,

    #[serde(rename = "a", default)]
    pub api_key_index: u8,

    #[serde(rename = "t", default)]
    pub to_account_index: i64,

    #[serde(rename = "ai", default)]
    pub asset_index: i16, // 6 bits

    #[serde(rename = "frt", default)]
    pub from_route_type: u8,

    #[serde(rename = "trt", default)]
    pub to_route_type: u8,

    #[serde(rename = "ba", default)]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub amount: BigUint, // 60 bits

    #[serde(rename = "u", default)]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub usdc_fee: BigUint,

    #[serde(rename = "m")]
    pub memo: [u8; TRANSFER_MEMO_BYTES],
}

#[derive(Debug)]
pub struct L2TransferTxTarget {
    pub from_account_index: Target,
    pub api_key_index: Target,
    pub to_account_index: Target,
    pub amount: BigUintTarget, // 60 bits
    pub asset_index: Target,
    pub from_route_type: Target,
    pub to_route_type: Target,
    pub usdc_fee: BigUintTarget,
    pub memo: [U8Target; TRANSFER_MEMO_BYTES], // Memo hash is not used in the circuit, but included for completeness

    pub success: BoolTarget, // Output

    extended_transfer_amount: BigUintTarget,
    extended_fee_amount: BigUintTarget,
}

impl L2TransferTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            from_account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            to_account_index: builder.add_virtual_target(),
            amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            usdc_fee: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            memo: builder
                .add_virtual_u8_targets_safe(TRANSFER_MEMO_BYTES)
                .try_into()
                .unwrap(),
            asset_index: builder.add_virtual_target(),
            from_route_type: builder.add_virtual_target(),
            to_route_type: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),

            // helpers
            extended_transfer_amount: BigUintTarget::default(),
            extended_fee_amount: BigUintTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        builder.assert_bool(BoolTarget::new_unsafe(self.to_route_type));
        builder.assert_bool(BoolTarget::new_unsafe(self.from_route_type));

        builder.range_check_biguint(&self.amount, MAX_TRANSFER_BITS);
        builder.range_check_biguint(&self.usdc_fee, MAX_TRANSFER_BITS);
    }
}

impl TxHash for L2TransferTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_TRANSFER)),
            tx_nonce,
            tx_expired_at,
            self.from_account_index,
            self.api_key_index,
            self.to_account_index,
            self.asset_index,
            self.from_route_type,
            self.to_route_type,
        ];

        let mut limbs = self.amount.limbs.clone();
        limbs.resize(BIG_U64_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }

        let mut limbs = self.usdc_fee.limbs.clone();
        limbs.resize(BIG_U64_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2TransferTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_transfer;
        self.success = is_enabled;

        self.register_range_checks(builder);

        builder.conditional_assert_eq(
            is_enabled,
            self.from_account_index,
            tx_state.accounts[SENDER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.to_account_index,
            tx_state.accounts[RECEIVER_ACCOUNT_ID].account_index,
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

        let lit_asset_index = builder.constant_u64(LIT_ASSET_INDEX);
        let is_lit_asset = builder.is_equal(self.asset_index, lit_asset_index);
        // Fee asset either be USDC or empty depending on the main asset being USDC or not
        let usdc_asset_index = builder.constant_u64(USDC_ASSET_INDEX);
        let is_usdc_asset = builder.is_equal(self.asset_index, usdc_asset_index);
        // If asset index is usdc, then the second asset slots will be empty assets.
        let usdc_asset_flag = builder.and(is_enabled, is_usdc_asset);
        let second_asset_is_empty = tx_state.assets[FEE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_true(usdc_asset_flag, second_asset_is_empty);
        // If asset index is not usdc, then the second asset slots will be usdc assets because of the fee
        let non_usdc_asset_flag = builder.and_not(is_enabled, is_usdc_asset);
        builder.conditional_assert_eq(
            non_usdc_asset_flag,
            tx_state.asset_indices[FEE_ASSET_ID],
            usdc_asset_index,
        );

        // Transfer amount checks - not zero, 60 bits max, gte min transfer amount
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &tx_state.assets[TX_ASSET_ID].min_transfer_amount,
            &self.amount,
        );
        builder.conditional_assert_not_zero_biguint(is_enabled, &self.amount);

        // Self transfer is only possible with different route types. If account is unified, both route types are point to the same balance, so it is not allowed.
        let is_same_account = builder.is_equal(self.from_account_index, self.to_account_index);
        let mut is_same_route_type = builder.is_equal(self.from_route_type, self.to_route_type);
        let is_account_unified = builder.is_equal_constant(
            tx_state.accounts[SENDER_ACCOUNT_ID].account_trading_mode,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED as u64,
        );
        is_same_route_type = builder.or(is_same_route_type, is_account_unified);
        let is_invalid_self_transfer = builder.and(is_same_account, is_same_route_type);
        builder.conditional_assert_false(is_enabled, is_invalid_self_transfer);

        // Treasury account cannot transfer to non-treasury accounts
        let is_sender_treasury_account = builder.is_equal_constant(
            tx_state.accounts[SENDER_ACCOUNT_ID].master_account_index,
            TREASURY_ACCOUNT_INDEX as u64,
        );
        let is_sender_and_receiver_same_master_account = builder.is_equal(
            tx_state.accounts[SENDER_ACCOUNT_ID].master_account_index,
            tx_state.accounts[RECEIVER_ACCOUNT_ID].master_account_index,
        );
        let is_sender_treasury_account_and_enabled =
            builder.and(is_enabled, is_sender_treasury_account);
        builder.conditional_assert_true(
            is_sender_treasury_account_and_enabled,
            is_sender_and_receiver_same_master_account,
        );

        // For transfers to perps, asset must be used as margin.
        let route_type_perps = builder.constant_u64(ROUTE_TYPE_PERPS);
        let route_type_spot = builder.constant_u64(ROUTE_TYPE_SPOT);
        let is_from_perps = builder.is_equal(self.from_route_type, route_type_perps);
        let is_from_spot = builder.is_equal(self.from_route_type, route_type_spot);
        let is_to_perps = builder.is_equal(self.to_route_type, route_type_perps);
        let is_to_spot = builder.is_equal(self.to_route_type, route_type_spot);

        let is_invalid_from_route_type = builder.and_not(
            is_from_perps,
            tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][TX_ASSET_ID],
        );
        builder.conditional_assert_false(is_enabled, is_invalid_from_route_type);

        let is_invalid_to_route_type = builder.and_not(
            is_to_perps,
            tx_state.is_asset_used_as_margin[RECEIVER_ACCOUNT_ID][TX_ASSET_ID],
        );
        builder.conditional_assert_false(is_enabled, is_invalid_to_route_type);

        // Can only be usdc asset for from/to perps transfers
        let is_perps = builder.or(is_from_perps, is_to_perps);
        let is_to_perps_invalid_route = builder.and_not(is_perps, is_usdc_asset);
        builder.conditional_assert_false(self.success, is_to_perps_invalid_route);

        // Verify that receiver account exists
        let is_receiver_new_account = tx_state.is_new_account[RECEIVER_ACCOUNT_ID];
        builder.conditional_assert_false(is_enabled, is_receiver_new_account);

        // Only allow usdc perps transfers to pools, and lit spot transfers to staking pools
        {
            let is_receiver_public_pool = builder.is_equal_constant(
                tx_state.accounts[RECEIVER_ACCOUNT_ID].account_type,
                PUBLIC_POOL_ACCOUNT_TYPE as u64,
            );
            let is_receiver_insurance_fund = builder.is_equal_constant(
                tx_state.accounts[RECEIVER_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let is_receiver_pool_account =
                builder.or(is_receiver_public_pool, is_receiver_insurance_fund);

            // Pool can receive transfers only when it's active with 0 shares only in Perps USDC
            let is_receiver_active_pool = builder.is_equal_constant(
                tx_state.accounts[RECEIVER_ACCOUNT_ID]
                    .public_pool_info
                    .status,
                ACTIVE_PUBLIC_POOL as u64,
            );
            let is_valid_receiver_pool =
                builder.multi_and(&[is_receiver_active_pool, is_to_perps, is_usdc_asset]);
            let is_invalid_pool_transfer =
                builder.and_not(is_receiver_pool_account, is_valid_receiver_pool);
            builder.conditional_assert_false(is_enabled, is_invalid_pool_transfer);

            let is_receiver_staking_pool = builder.is_equal_constant(
                tx_state.accounts[RECEIVER_ACCOUNT_ID].account_type,
                LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
            );
            let is_valid_receiver_staking_pool =
                builder.multi_and(&[is_receiver_active_pool, is_to_spot, is_lit_asset]);
            let is_invalid_staking_pool_transfer =
                builder.and_not(is_receiver_staking_pool, is_valid_receiver_staking_pool);
            builder.conditional_assert_false(is_enabled, is_invalid_staking_pool_transfer);
        }

        // Verify sender pool accounts
        {
            let is_sender_public_pool = builder.is_equal_constant(
                tx_state.accounts[SENDER_ACCOUNT_ID].account_type,
                PUBLIC_POOL_ACCOUNT_TYPE as u64,
            );
            let is_sender_insurance_fund = builder.is_equal_constant(
                tx_state.accounts[SENDER_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let is_sender_pool_account =
                builder.or(is_sender_public_pool, is_sender_insurance_fund);
            let is_sender_staking_pool = builder.is_equal_constant(
                tx_state.accounts[SENDER_ACCOUNT_ID].account_type,
                LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
            );

            // Pool can transfer outside only when it's frozen with 0 shares only in Perps USDC
            let is_frozen_sender = builder.is_equal_constant(
                tx_state.accounts[SENDER_ACCOUNT_ID].public_pool_info.status,
                FROZEN_PUBLIC_POOL as u64,
            );
            let zero_shares_pool = builder.is_zero(
                tx_state.accounts[SENDER_ACCOUNT_ID]
                    .public_pool_info
                    .total_shares,
            );
            let is_valid_sender_pool = builder.multi_and(&[
                is_frozen_sender,
                zero_shares_pool,
                is_from_perps,
                is_usdc_asset,
            ]);
            let is_invalid_pool_transfer =
                builder.and_not(is_sender_pool_account, is_valid_sender_pool);
            builder.conditional_assert_false(is_enabled, is_invalid_pool_transfer);

            // Staking pool can transfer outside only when it's frozen with 0 shares only in Lit Spot
            let is_valid_sender_staking_pool = builder.multi_and(&[
                is_frozen_sender,
                zero_shares_pool,
                is_from_spot,
                is_lit_asset,
            ]);
            let is_invalid_staking_pool_transfer =
                builder.and_not(is_sender_staking_pool, is_valid_sender_staking_pool);
            builder.conditional_assert_false(is_enabled, is_invalid_staking_pool_transfer);
        }

        // Calculate helper fields
        let usdc_to_collateral_multiplier =
            BigUintTarget::from(builder.constant_u32(USDC_TO_COLLATERAL_MULTIPLIER));
        self.extended_fee_amount = builder.mul_biguint_non_carry(
            &self.usdc_fee,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );
        self.extended_transfer_amount = builder.mul_biguint_non_carry(
            &self.amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );

        let sender_product_type =
            builder.select_constant(is_from_perps, PRODUCT_TYPE_PERPS, PRODUCT_TYPE_SPOT);

        let sender_asset_balance = get_available_asset_balance(
            builder,
            sender_product_type,
            &tx_state.accounts[SENDER_ACCOUNT_ID],
            &tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[SENDER_ACCOUNT_ID].cross_risk_parameters,
        );

        let sender_available_usdc = get_available_asset_balance(
            builder,
            sender_product_type,
            &tx_state.accounts[SENDER_ACCOUNT_ID],
            &tx_state.account_assets[SENDER_ACCOUNT_ID][FEE_ASSET_ID],
            tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][FEE_ASSET_ID],
            &tx_state.risk_infos[SENDER_ACCOUNT_ID].cross_risk_parameters,
        );

        // Sender balance checks
        {
            let flag = self.success;

            // Asset is usdc - amount + fee is paid from asset balance
            let extended_usdc_amount = builder.add_biguint_non_carry(
                &self.extended_fee_amount,
                &self.extended_transfer_amount,
                BIG_U96_LIMBS,
            );
            let flag_if_asset_is_usdc = builder.and(flag, is_usdc_asset);
            builder.conditional_assert_lte_biguint(
                flag_if_asset_is_usdc,
                &extended_usdc_amount,
                &sender_asset_balance,
            );

            // Asset is not usdc - amount is paid from asset balance, fee from usdc balance
            let flag_if_asset_is_not_usdc = builder.and_not(flag, is_usdc_asset);
            builder.conditional_assert_lte_biguint(
                flag_if_asset_is_not_usdc,
                &self.extended_transfer_amount,
                &sender_asset_balance,
            );
            builder.conditional_assert_lte_biguint(
                flag_if_asset_is_not_usdc,
                &self.extended_fee_amount,
                &sender_available_usdc,
            );
        }

        // Verification for receiver is exceeding the maximum account value or not will be done in Apply
    }
}

impl Apply for L2TransferTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let is_usdc_asset = builder.is_equal_constant(self.asset_index, USDC_ASSET_INDEX);
        let is_success_and_not_usdc_asset = builder.and_not(self.success, is_usdc_asset);

        // Sender collateral/balance deltas
        {
            let is_from_perps = builder.is_equal_constant(self.from_route_type, ROUTE_TYPE_PERPS);
            let sender_product_type =
                builder.select_constant(is_from_perps, PRODUCT_TYPE_PERPS, PRODUCT_TYPE_SPOT);

            let sender_is_fee_account = builder.is_equal(
                tx_state.accounts[SENDER_ACCOUNT_ID].account_index,
                tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            );
            let sender_is_fee_account_and_success =
                builder.and(sender_is_fee_account, self.success);

            let mut extended_total_amount = self.extended_transfer_amount.clone();
            let extended_asset_fee =
                builder.mul_biguint_by_bool(&self.extended_fee_amount, is_usdc_asset);
            extended_total_amount = builder.add_biguint_non_carry(
                &extended_total_amount,
                &extended_asset_fee,
                BIG_U96_LIMBS,
            );
            let mut sender_asset_delta = builder.biguint_to_bigint(&extended_total_amount);
            sender_asset_delta = builder.neg_bigint(&sender_asset_delta);

            // Transfer
            AccountTarget::apply_asset_delta(
                builder,
                self.success,
                sender_product_type,
                &mut tx_state.accounts[SENDER_ACCOUNT_ID],
                &mut tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID],
                tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][TX_ASSET_ID],
                &sender_asset_delta,
                &mut tx_state.strategies[SENDER_ACCOUNT_ID],
            );

            // USDC fee - if asset is not USDC
            let mut sender_usdc_fee_amount = builder.biguint_to_bigint(&self.extended_fee_amount);
            sender_usdc_fee_amount = builder.neg_bigint(&sender_usdc_fee_amount);
            AccountTarget::apply_asset_delta(
                builder,
                is_success_and_not_usdc_asset,
                sender_product_type,
                &mut tx_state.accounts[SENDER_ACCOUNT_ID],
                &mut tx_state.account_assets[SENDER_ACCOUNT_ID][FEE_ASSET_ID],
                tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][FEE_ASSET_ID],
                &sender_usdc_fee_amount,
                &mut tx_state.strategies[SENDER_ACCOUNT_ID],
            );

            // Collect fee - if sender and fee accounts are same. Fee is always collected into perps balance
            let sender_fee_collateral_delta = builder.biguint_to_bigint(&self.extended_fee_amount);
            AccountTarget::apply_asset_delta_const(
                builder,
                sender_is_fee_account_and_success,
                PRODUCT_TYPE_PERPS,
                &mut tx_state.accounts[SENDER_ACCOUNT_ID],
                None,
                tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][FEE_ASSET_ID],
                &sender_fee_collateral_delta,
                &mut tx_state.strategies[SENDER_ACCOUNT_ID],
            );
        }

        // Increase balance for receiver
        {
            let is_perps = builder.is_equal_constant(self.to_route_type, ROUTE_TYPE_PERPS);
            let receiver_product_type =
                builder.select_constant(is_perps, PRODUCT_TYPE_PERPS, PRODUCT_TYPE_SPOT);

            let receiver_is_fee_account = builder.is_equal(
                tx_state.accounts[RECEIVER_ACCOUNT_ID].account_index,
                tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            );
            let receiver_is_fee_account_and_success =
                builder.and(receiver_is_fee_account, self.success);

            let receiver_asset_delta = builder.biguint_to_bigint(&self.extended_transfer_amount);

            // Transfer
            AccountTarget::apply_asset_delta(
                builder,
                self.success,
                receiver_product_type,
                &mut tx_state.accounts[RECEIVER_ACCOUNT_ID],
                &mut tx_state.account_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID],
                tx_state.is_asset_used_as_margin[RECEIVER_ACCOUNT_ID][TX_ASSET_ID],
                &receiver_asset_delta,
                &mut tx_state.strategies[RECEIVER_ACCOUNT_ID],
            );

            // Transfer - apply to sender if they are the same account
            let is_sender_receiver_same =
                builder.and_not(self.success, tx_state.is_sender_receiver_different);
            AccountTarget::apply_asset_delta(
                builder,
                is_sender_receiver_same,
                receiver_product_type,
                &mut tx_state.accounts[SENDER_ACCOUNT_ID],
                &mut tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID],
                tx_state.is_asset_used_as_margin[SENDER_ACCOUNT_ID][TX_ASSET_ID],
                &receiver_asset_delta,
                &mut tx_state.strategies[SENDER_ACCOUNT_ID],
            );

            // Collect fee - if receiver and fee accounts are same. Fee is always collected into perps balance
            let receiver_fee_collateral_delta =
                builder.biguint_to_bigint(&self.extended_fee_amount);
            AccountTarget::apply_asset_delta_const(
                builder,
                receiver_is_fee_account_and_success,
                PRODUCT_TYPE_PERPS,
                &mut tx_state.accounts[RECEIVER_ACCOUNT_ID],
                None,
                tx_state.is_asset_used_as_margin[RECEIVER_ACCOUNT_ID][FEE_ASSET_ID],
                &receiver_fee_collateral_delta,
                &mut tx_state.strategies[RECEIVER_ACCOUNT_ID],
            );
        }

        // Increase balance for fee account (if not sender or receiver)
        // If fee account is sender or receiver, fee is already added/deducted in the above sections and this account will be skipped while updating merkle state
        {
            let fee_collateral_delta = builder.biguint_to_bigint(&self.extended_fee_amount);

            // Collect fee
            AccountTarget::apply_asset_delta_const(
                builder,
                self.success,
                PRODUCT_TYPE_PERPS,
                &mut tx_state.accounts[FEE_ACCOUNT_ID],
                None,
                tx_state.is_asset_used_as_margin[FEE_ACCOUNT_ID][FEE_ASSET_ID],
                &fee_collateral_delta,
                &mut tx_state.strategies[FEE_ACCOUNT_ID],
            );
        }

        self.success
    }
}

pub trait L2TransferTxTargetWitness<F: PrimeField64> {
    fn set_l2_transfer_tx_target(&mut self, a: &L2TransferTxTarget, b: &L2TransferTx)
    -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2TransferTxTargetWitness<F> for T {
    fn set_l2_transfer_tx_target(
        &mut self,
        a: &L2TransferTxTarget,
        b: &L2TransferTx,
    ) -> Result<()> {
        self.set_target(
            a.from_account_index,
            F::from_canonical_i64(b.from_account_index),
        )?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.to_account_index,
            F::from_canonical_i64(b.to_account_index),
        )?;
        self.set_biguint_target(&a.amount, &b.amount)?;
        self.set_biguint_target(&a.usdc_fee, &b.usdc_fee)?;
        for (a, b) in a.memo.iter().zip(b.memo.iter()) {
            self.set_target(a.0, F::from_canonical_u8(*b))?;
        }
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;
        self.set_target(a.from_route_type, F::from_canonical_u8(b.from_route_type))?;
        self.set_target(a.to_route_type, F::from_canonical_u8(b.to_route_type))?;

        Ok(())
    }
}
