// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module:
// 1. perps-route deposits require margin-enabled assets.
// 2. accepted deposit does not exceed product-context balance cap.
// 3. unified spot deposits for margin-enabled assets are applied to collateral.
// 4. balance updates are applied via account-level asset delta helpers.

use anyhow::Result;
use num::{BigUint, FromPrimitive};
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::deserializers;
use crate::tx_interface::{Apply, OnChainPubData, PriorityOperationsPubData, Verify};
use crate::types::account::AccountTarget;
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U160_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct L1DepositTx {
    #[serde(rename = "i", default)]
    pub account_index: i64,

    #[serde(rename = "l")]
    #[serde(deserialize_with = "deserializers::l1_address_to_biguint")]
    pub l1_address: BigUint,

    #[serde(rename = "ai")]
    pub asset_index: i16, // 6 bits

    #[serde(rename = "rt", default)]
    pub route_type: u8,

    #[serde(rename = "a")]
    pub amount: u64, // 50 bits

    #[serde(rename = "aa", default)]
    pub accepted_amount: u64,
}

#[derive(Debug)]
pub struct L1DepositTxTarget {
    pub account_index: Target,
    pub l1_address: BigUintTarget,

    pub asset_index: Target,
    pub route_type: Target,
    pub amount: BigUintTarget,
    pub accepted_amount: BigUintTarget,

    // Helper
    is_new_account: BoolTarget,

    // Output
    success: BoolTarget,
    is_enabled: BoolTarget,
    on_chain_pub_data_flag: BoolTarget,
}

impl L1DepositTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L1DepositTxTarget {
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_safe(BIG_U160_LIMBS),
            amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            accepted_amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            asset_index: builder.add_virtual_target(),
            route_type: builder.add_virtual_target(),

            // Helper
            is_new_account: BoolTarget::default(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
            on_chain_pub_data_flag: BoolTarget::default(),
        }
    }
}

impl PriorityOperationsPubData for L1DepositTxTarget {
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
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_DEPOSIT),
            add_target_extend(
                builder,
                bytes,
                self.account_index,
                ACCOUNT_INDEX_BITS,
                MASTER_ACCOUNT_INDEX_BITS,
            ),
            add_big_uint_target(builder, bytes, &self.l1_address),
            add_target(builder, bytes, self.asset_index, 16),
            add_byte_target_unsafe(bytes, self.route_type),
            add_deposit_usdc_target(builder, bytes, &self.amount),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, byte_count),
        )
    }
}

impl OnChainPubData for L1DepositTxTarget {
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
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        add_target(builder, bytes, self.asset_index, 16);

        let (withdrawal_amount, borrow) =
            builder.try_sub_biguint(&self.amount, &self.accepted_amount);
        builder.conditional_assert_zero_u32(self.on_chain_pub_data_flag, borrow);

        assert_eq!(withdrawal_amount.limbs.len(), BIG_U64_LIMBS);
        add_big_uint_target(builder, bytes, &withdrawal_amount);

        (
            self.on_chain_pub_data_flag,
            pad_on_chain_pub_data_target(builder, bytes),
        )
    }
}

impl Verify for L1DepositTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_types: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_types.is_l1_deposit;
        self.success = is_enabled;
        self.is_enabled = is_enabled;

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
        ensure_valid_asset_index(builder, self.is_enabled, self.asset_index);

        // SPOT or PERPS
        builder.assert_bool(BoolTarget::new_unsafe(self.route_type));

        builder.range_check_biguint(&self.amount, MAX_EXCHANGE_USDC_BITS);
        builder.range_check_biguint(&self.accepted_amount, MAX_EXCHANGE_USDC_BITS);

        // Sequencer is allowed to accept only a portion of the USDC amount
        builder.conditional_assert_lte_biguint(is_enabled, &self.accepted_amount, &self.amount);

        self.is_new_account = builder.and(is_enabled, tx_state.is_new_account[OWNER_ACCOUNT_ID]);

        // Accepted amount checks
        {
            /*
                Following conditions will lead accepted amount to be zero, as well as skipping account creation
                - Asset is empty
                - Route type is PERPS but asset is not margin-enabled
            */
            let is_accepted_amount_zero = builder.is_zero_biguint(&self.accepted_amount);
            let is_accepted_amount_non_zero = builder.not(is_accepted_amount_zero);

            // If sequencer accepted non-zero amount, asset should be exist in the system
            let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
            let asset_existence_check =
                builder.multi_and(&[is_accepted_amount_non_zero, self.is_enabled]);
            builder.conditional_assert_false(asset_existence_check, is_asset_empty);

            let is_perps = builder.is_equal_constant(self.route_type, ROUTE_TYPE_PERPS);
            let margin_mode_check =
                builder.multi_and(&[is_accepted_amount_non_zero, self.is_enabled, is_perps]);
            let is_asset_margin_enabled =
                tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID];
            builder.conditional_assert_true(margin_mode_check, is_asset_margin_enabled);
        }

        // nil account index is reserved and always should be empty
        let nil_account_index = builder.constant_i64(NIL_ACCOUNT_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.account_index, nil_account_index);

        let is_l1_address_match = builder.is_equal_biguint(
            &self.l1_address,
            &tx_state.accounts[OWNER_ACCOUNT_ID].l1_address,
        );

        let is_valid_address = builder.multi_or(&[is_l1_address_match, self.is_new_account]);
        builder.conditional_assert_true(is_enabled, is_valid_address);
    }
}

impl Apply for L1DepositTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        // Handle account creation
        let master_account_type = builder.constant_from_u8(MASTER_ACCOUNT_TYPE);
        tx_state.accounts[OWNER_ACCOUNT_ID].l1_address = builder.select_biguint(
            self.is_new_account,
            &self.l1_address,
            &tx_state.accounts[OWNER_ACCOUNT_ID].l1_address,
        );
        tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index = builder.select(
            self.is_new_account,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
        );
        tx_state.accounts[OWNER_ACCOUNT_ID].account_type = builder.select(
            self.is_new_account,
            master_account_type,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
        );
        let unified_trading_mode = builder.constant_from_u8(ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED);
        tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode = builder.select(
            self.is_new_account,
            unified_trading_mode,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode,
        );

        let extended_balance_delta = builder.mul_biguint_non_carry(
            &self.accepted_amount,
            &tx_state.assets[TX_ASSET_ID].extension_multiplier,
            BIG_U96_LIMBS,
        );
        let extended_balance_delta_big = builder.biguint_to_bigint(&extended_balance_delta);
        let is_delta_zero = builder.is_zero_bigint(&extended_balance_delta_big);

        // Avoid failures from checks in apply_asset_delta when accepted amount is zero.
        let apply_delta_flag = builder.and_not(self.success, is_delta_zero);
        let _false = builder._false();
        let is_account_unified = tx_state.accounts[OWNER_ACCOUNT_ID].is_unified_mode();
        AccountTarget::apply_asset_delta(
            builder,
            apply_delta_flag,
            self.route_type,
            tx_state.asset_indices[TX_ASSET_ID],
            &mut tx_state.margined_asset[TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            &extended_balance_delta_big,
            is_account_unified,
            _false,
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
            false, // sequencer should accept zero if this operation is overflowed
        );

        // Create withdraw onchain operation when accepted_amount is less than usdc_amount
        let accepted_amount_eq_amount =
            builder.is_equal_biguint(&self.accepted_amount, &self.amount);
        self.on_chain_pub_data_flag = builder.and_not(self.success, accepted_amount_eq_amount);

        self.success
    }
}

pub trait L1DepositTxTargetWitness<F: PrimeField64> {
    fn set_l1_deposit_tx_target(&mut self, a: &L1DepositTxTarget, b: &L1DepositTx) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1DepositTxTargetWitness<F> for T {
    fn set_l1_deposit_tx_target(&mut self, a: &L1DepositTxTarget, b: &L1DepositTx) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_biguint_target(&a.l1_address, &b.l1_address)?;
        self.set_biguint_target(&a.amount, &BigUint::from_u64(b.amount).unwrap())?;
        self.set_biguint_target(
            &a.accepted_amount,
            &BigUint::from_u64(b.accepted_amount).unwrap(),
        )?;
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;
        self.set_target(a.route_type, F::from_canonical_u8(b.route_type))?;

        Ok(())
    }
}
