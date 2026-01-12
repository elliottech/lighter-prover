// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
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
    extended_usdc_amount: BigUintTarget, // fee + transfer if asset index is USDC
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
            extended_usdc_amount: BigUintTarget::default(),
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

        // Self transfer is only possible with different route types
        let is_same_account = builder.is_equal(self.from_account_index, self.to_account_index);
        let is_same_route_type = builder.is_equal(self.from_route_type, self.to_route_type);
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

        // For transfers between spot and perps, asset must be margin-enabled.
        let is_asset_margin_enabled = builder.is_equal_constant(
            tx_state.assets[TX_ASSET_ID].margin_mode,
            ASSET_MARGIN_MODE_ENABLED,
        );
        let route_type_perps = builder.constant_u64(ROUTE_TYPE_PERPS);
        let route_type_spot = builder.constant_u64(ROUTE_TYPE_SPOT);
        let is_from_perps = builder.is_equal(self.from_route_type, route_type_perps);
        let is_from_spot = builder.is_equal(self.from_route_type, route_type_spot);
        let is_to_perps = builder.is_equal(self.to_route_type, route_type_perps);
        let is_to_spot = builder.is_equal(self.to_route_type, route_type_spot);
        let is_perps = builder.or(is_from_perps, is_to_perps);
        let is_invalid_route_type = builder.and_not(is_perps, is_asset_margin_enabled);
        builder.conditional_assert_false(is_enabled, is_invalid_route_type);
        // Can only be usdc asset for from/to perps transfers
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
        let add_to_extended_usdc_amount =
            builder.mul_biguint_by_bool(&self.extended_transfer_amount, is_usdc_asset);
        self.extended_usdc_amount = builder.add_biguint_non_carry(
            &self.extended_fee_amount,
            &add_to_extended_usdc_amount,
            BIG_U96_LIMBS,
        );

        // Sender balance checks: Route Type - From Spot
        {
            let flag = builder.and_not(self.success, is_from_perps);

            let sender_asset_balance = tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID]
                .get_available_balance(builder);

            // Asset is usdc - amount + fee is paid from asset balance
            let flag_if_asset_is_usdc = builder.and(flag, is_usdc_asset);
            builder.conditional_assert_lte_biguint(
                flag_if_asset_is_usdc,
                &self.extended_usdc_amount,
                &sender_asset_balance,
            );

            // Asset is not usdc - amount is paid from asset balance, fee from usdc balance
            let flag_if_asset_is_not_usdc = builder.and_not(flag, is_usdc_asset);
            builder.conditional_assert_lte_biguint(
                flag_if_asset_is_not_usdc,
                &self.extended_transfer_amount,
                &sender_asset_balance,
            );
            let sender_usdc_asset_balance = tx_state.account_assets[OWNER_ACCOUNT_ID][FEE_ASSET_ID]
                .get_available_balance(builder);
            builder.conditional_assert_lte_biguint(
                flag_if_asset_is_not_usdc,
                &self.extended_fee_amount,
                &sender_usdc_asset_balance,
            );
        }

        // Sender balance checks: Route Type - From Perps
        {
            let flag = builder.and(self.success, is_from_perps);

            builder.conditional_assert_true(flag, is_usdc_asset);

            let available_cross_collateral = get_available_collateral(
                builder,
                &tx_state.risk_infos[SENDER_ACCOUNT_ID].cross_risk_parameters,
            );
            builder.conditional_assert_lte_biguint(
                flag,
                &self.extended_usdc_amount,
                &available_cross_collateral,
            );
        }

        // Verification for receiver is exceeding the maximum account value or not will be done in Apply
    }
}

impl Apply for L2TransferTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();
        let zero = builder.zero();
        let neg_one = builder.neg_one();

        let is_usdc_asset = builder.is_equal_constant(self.asset_index, USDC_ASSET_INDEX);

        // Decrease balance from sender
        {
            let sender_is_fee_account = builder.is_equal(
                tx_state.accounts[SENDER_ACCOUNT_ID].account_index,
                tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            );
            let is_from_route_type_spot =
                builder.is_equal_constant(self.from_route_type, ROUTE_TYPE_SPOT);
            // From Perps
            {
                let sender_perps_flag = builder.and_not(self.success, is_from_route_type_spot);
                let deduct_from_sender = builder.select_biguint(
                    sender_is_fee_account,
                    &self.extended_transfer_amount,
                    &self.extended_usdc_amount,
                );
                let add_to_sender_signed = BigIntTarget {
                    abs: builder.mul_biguint_by_bool(&deduct_from_sender, sender_perps_flag),
                    sign: SignTarget::new_unsafe(builder.mul_bool(sender_perps_flag, neg_one)),
                };
                tx_state.accounts[SENDER_ACCOUNT_ID].collateral = builder.add_bigint_non_carry(
                    &tx_state.accounts[SENDER_ACCOUNT_ID].collateral,
                    &add_to_sender_signed,
                    BIG_U96_LIMBS,
                );
            }
            // From Spot
            {
                let flag = builder.and(self.success, is_from_route_type_spot);

                // Deduct transfer amount from the main asset - Will be done in any case
                let deduct_from_first_asset =
                    builder.mul_biguint_by_bool(&self.extended_transfer_amount, flag);
                let (new_sender_asset_balance, fail) = builder.try_sub_biguint(
                    &tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance,
                    &deduct_from_first_asset,
                );
                builder.conditional_assert_zero_u32(flag, fail);
                tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance =
                    new_sender_asset_balance;

                // Asset type is usdc - deduct fee from first asset
                {
                    let flag = builder.and(flag, is_usdc_asset);
                    let deduct_from_first_asset =
                        builder.mul_biguint_by_bool(&self.extended_fee_amount, flag);
                    let (new_sender_asset_balance, fail) = builder.try_sub_biguint(
                        &tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance,
                        &deduct_from_first_asset,
                    );
                    builder.conditional_assert_zero_u32(flag, fail);
                    tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance =
                        new_sender_asset_balance;
                }
                // Asset type is not usdc - Deduct fee from usdc asset
                {
                    let flag = builder.and_not(flag, is_usdc_asset);
                    let deduct_from_second_asset =
                        builder.mul_biguint_by_bool(&self.extended_fee_amount, flag);
                    let (new_sender_usdc_asset_balance, fail) = builder.try_sub_biguint(
                        &tx_state.account_assets[SENDER_ACCOUNT_ID][FEE_ASSET_ID].balance,
                        &deduct_from_second_asset,
                    );
                    builder.conditional_assert_zero_u32(flag, fail);
                    tx_state.account_assets[SENDER_ACCOUNT_ID][FEE_ASSET_ID].balance =
                        new_sender_usdc_asset_balance;
                }

                // Sender is fee account - Add to sender's perps collateral
                {
                    let flag = builder.and(flag, sender_is_fee_account);

                    let add_to_sender =
                        builder.mul_biguint_by_bool(&self.extended_fee_amount, flag);
                    let add_to_sender = builder.biguint_to_bigint(&add_to_sender);
                    tx_state.accounts[SENDER_ACCOUNT_ID].collateral = builder.add_bigint_non_carry(
                        &tx_state.accounts[SENDER_ACCOUNT_ID].collateral,
                        &add_to_sender,
                        BIG_U96_LIMBS,
                    );
                }
            }
        }

        let is_sender_receiver_same = builder.not(tx_state.is_sender_receiver_different);
        // Increase balance for receiver
        {
            let receiver_is_fee_account = builder.is_equal(
                tx_state.accounts[RECEIVER_ACCOUNT_ID].account_index,
                tx_state.accounts[FEE_ACCOUNT_ID].account_index,
            );
            let is_to_route_type_spot =
                builder.is_equal_constant(self.to_route_type, ROUTE_TYPE_SPOT);
            // To Perps
            {
                let flag = builder.and_not(self.success, is_to_route_type_spot);
                let add_to_receiver = builder.select_biguint(
                    receiver_is_fee_account,
                    &self.extended_usdc_amount,
                    &self.extended_transfer_amount,
                );
                let add_to_receiver_signed = BigIntTarget {
                    abs: builder.mul_biguint_by_bool(&add_to_receiver, flag),
                    sign: SignTarget::new_unsafe(builder.mul_bool(flag, one)),
                };
                tx_state.accounts[RECEIVER_ACCOUNT_ID].collateral = builder.add_bigint_non_carry(
                    &tx_state.accounts[RECEIVER_ACCOUNT_ID].collateral,
                    &add_to_receiver_signed,
                    BIG_U96_LIMBS,
                );
                let add_to_sender_signed = BigIntTarget {
                    abs: builder
                        .mul_biguint_by_bool(&add_to_receiver_signed.abs, is_sender_receiver_same),
                    sign: SignTarget::new_unsafe(
                        builder
                            .mul_bool(is_sender_receiver_same, add_to_receiver_signed.sign.target),
                    ),
                };
                tx_state.accounts[SENDER_ACCOUNT_ID].collateral = builder.add_bigint_non_carry(
                    &tx_state.accounts[SENDER_ACCOUNT_ID].collateral,
                    &add_to_sender_signed,
                    BIG_U96_LIMBS,
                );
            }
            // To Spot
            {
                let flag = builder.and(self.success, is_to_route_type_spot);

                // Add transfer amount to the main asset - Will be done in any case
                let add_to_first_asset =
                    builder.mul_biguint_by_bool(&self.extended_transfer_amount, flag);
                tx_state.account_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].balance = builder
                    .add_biguint_non_carry(
                        &tx_state.account_assets[RECEIVER_ACCOUNT_ID][TX_ASSET_ID].balance,
                        &add_to_first_asset,
                        BIG_U96_LIMBS,
                    );
                let add_to_sender =
                    builder.mul_biguint_by_bool(&add_to_first_asset, is_sender_receiver_same);
                tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance = builder
                    .add_biguint_non_carry(
                        &tx_state.account_assets[SENDER_ACCOUNT_ID][TX_ASSET_ID].balance,
                        &add_to_sender,
                        BIG_U96_LIMBS,
                    );

                // Receiver is fee account - Add fee to collateral
                let flag = builder.and(flag, receiver_is_fee_account);
                let add_to_receiver = builder.mul_biguint_by_bool(&self.extended_fee_amount, flag);
                let add_to_receiver = builder.biguint_to_bigint(&add_to_receiver);
                tx_state.accounts[RECEIVER_ACCOUNT_ID].collateral = builder.add_bigint_non_carry(
                    &tx_state.accounts[RECEIVER_ACCOUNT_ID].collateral,
                    &add_to_receiver,
                    BIG_U96_LIMBS,
                );
                let add_to_sender_signed = BigIntTarget {
                    abs: builder.mul_biguint_by_bool(&add_to_receiver.abs, is_sender_receiver_same),
                    sign: SignTarget::new_unsafe(
                        builder.mul_bool(is_sender_receiver_same, add_to_receiver.sign.target),
                    ),
                };
                tx_state.accounts[SENDER_ACCOUNT_ID].collateral = builder.add_bigint_non_carry(
                    &tx_state.accounts[SENDER_ACCOUNT_ID].collateral,
                    &add_to_sender_signed,
                    BIG_U96_LIMBS,
                );
            }
        }

        // Increase balance for fee account (if not sender or receiver)
        // If fee account is sender or receiver, fee is already added/deducted in the above sections and this account will be skipped while updating merkle state
        {
            let is_fee_zero = builder.is_zero_biguint(&self.extended_fee_amount);
            let addition_sign = builder.select(is_fee_zero, zero, one);
            let fee_collateral_after = builder.add_bigint_non_carry(
                &tx_state.accounts[FEE_ACCOUNT_ID].collateral,
                &BigIntTarget {
                    abs: self.extended_fee_amount.clone(),
                    sign: SignTarget::new_unsafe(addition_sign),
                },
                BIG_U96_LIMBS,
            );
            tx_state.accounts[FEE_ACCOUNT_ID].collateral = builder.select_bigint(
                self.success,
                &fee_collateral_after,
                &tx_state.accounts[FEE_ACCOUNT_ID].collateral,
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
