// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigUint, FromPrimitive};
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::asset::ensure_valid_asset_index;
use crate::types::config::{BIG_U64_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::margined_asset::{MarginedAssetTarget, select_margined_asset_target};
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct L1UpdateAssetTx {
    #[serde(rename = "ai")]
    pub asset_index: i16, // 6 bits
    #[serde(rename = "mta")]
    pub min_transfer_amount: i64, // 60 bits
    #[serde(rename = "mwa")]
    pub min_withdrawal_amount: i64, // 60 bits
    #[serde(rename = "mm", default)]
    pub margin_mode: u8,
    #[serde(rename = "ltv", default)]
    pub loan_to_value: u16,
    #[serde(rename = "lt", default)]
    pub liquidation_threshold: u16,
    #[serde(rename = "lfc", default)]
    pub liquidation_factor: u16,
    #[serde(rename = "lf", default)]
    pub liquidation_fee: u32,

    #[serde(rename = "ip", default)]
    pub index_price: i64, // Given by sequencer
    #[serde(rename = "ipd", default)]
    pub index_price_divider: i64,
}

#[derive(Debug)]
pub struct L1UpdateAssetTxTarget {
    pub asset_index: Target,
    pub min_transfer_amount: BigUintTarget,
    pub min_withdrawal_amount: BigUintTarget,
    pub margin_mode: Target,
    pub loan_to_value: Target,
    pub liquidation_threshold: Target,
    pub liquidation_factor: Target,
    pub liquidation_fee: Target,
    pub index_price: Target,
    pub index_price_divider: Target,

    // helpers
    margin_enabled: BoolTarget,
    enabling_margin: BoolTarget,

    // output
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1UpdateAssetTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L1UpdateAssetTxTarget {
            asset_index: builder.add_virtual_target(),
            min_transfer_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_withdrawal_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            margin_mode: builder.add_virtual_target(),
            loan_to_value: builder.add_virtual_target(),
            liquidation_threshold: builder.add_virtual_target(),
            liquidation_factor: builder.add_virtual_target(),
            liquidation_fee: builder.add_virtual_target(),
            index_price: builder.add_virtual_target(),
            index_price_divider: builder.add_virtual_target(),

            margin_enabled: builder._false(),
            enabling_margin: builder._false(),

            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }
}

impl PriorityOperationsPubData for L1UpdateAssetTxTarget {
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
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_UPDATE_ASSET),
            add_target(builder, bytes, self.asset_index, 16),
            add_big_uint_target(builder, bytes, &self.min_transfer_amount),
            add_big_uint_target(builder, bytes, &self.min_withdrawal_amount),
            add_byte_target_unsafe(bytes, self.margin_mode),
            add_target(builder, bytes, self.loan_to_value, 16),
            add_target(builder, bytes, self.liquidation_threshold, 16),
            add_target(builder, bytes, self.liquidation_factor, 16),
            add_target(builder, bytes, self.liquidation_fee, 32),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, byte_count),
        )
    }
}

impl Verify for L1UpdateAssetTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_types: &TxTypeTargets, tx_state: &TxState) {
        self.success = tx_types.is_l1_update_asset;
        self.is_enabled = tx_types.is_l1_update_asset;

        builder.conditional_assert_eq(
            self.is_enabled,
            self.asset_index,
            tx_state.asset_indices[TX_ASSET_ID],
        );
        ensure_valid_asset_index(builder, self.is_enabled, self.asset_index);

        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.min_transfer_amount);
        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.min_withdrawal_amount);

        builder.register_range_check(self.index_price, MAX_ASSET_PRICE_BITS);
        builder.register_range_check(self.index_price_divider, MAX_ASSET_PRICE_BITS);

        builder.range_check_biguint(&self.min_transfer_amount, MAX_EXCHANGE_ASSET_BALANCE_BITS);
        builder.range_check_biguint(&self.min_withdrawal_amount, MAX_EXCHANGE_ASSET_BALANCE_BITS);
        builder.assert_bool(BoolTarget::new_unsafe(self.margin_mode));

        let asset_margin_tick = builder.constant_u64(ASSET_MARGIN_TICK);
        builder.conditional_assert_lte(
            self.is_enabled,
            self.liquidation_factor,
            asset_margin_tick,
            16,
        );
        builder.conditional_assert_lte(
            self.is_enabled,
            self.liquidation_threshold,
            self.liquidation_factor,
            16,
        );
        builder.conditional_assert_lte(
            self.is_enabled,
            self.loan_to_value,
            self.liquidation_threshold,
            16,
        );

        let fee_tick = builder.constant_u64(FEE_TICK);
        let fee_tick_over_asset_margin_tick = builder.constant_u64(FEE_TICK / ASSET_MARGIN_TICK);
        let should_be_lte_fee_tick = builder.mul_add(
            fee_tick_over_asset_margin_tick,
            self.liquidation_factor,
            self.liquidation_fee,
        );
        builder.conditional_assert_lte(self.is_enabled, should_be_lte_fee_tick, fee_tick, 32);

        self.margin_enabled =
            builder.is_equal_constant(self.margin_mode, ASSET_MARGIN_MODE_ENABLED);
        self.enabling_margin = builder.and_not(
            self.margin_enabled,
            BoolTarget::new_unsafe(tx_state.assets[TX_ASSET_ID].margin_mode),
        );

        // LTV can't be zero if margin mode is enabled
        let is_ltv_zero = builder.is_zero(self.loan_to_value);
        let should_be_false = builder.and(self.margin_enabled, is_ltv_zero);
        builder.conditional_assert_false(self.is_enabled, should_be_false);

        // LT, and liquidation fee should be 0 if margin mode is disabled
        let is_lt_zero = builder.is_zero(self.liquidation_threshold);
        let is_lf_zero = builder.is_zero(self.liquidation_factor);
        let is_liquidation_fee_zero = builder.is_zero(self.liquidation_fee);
        let are_margin_parameters_zero =
            builder.multi_and(&[is_ltv_zero, is_lf_zero, is_lt_zero, is_liquidation_fee_zero]);
        let is_at_least_one_margin_parameter_non_zero = builder.not(are_margin_parameters_zero);
        let should_be_false = builder.and_not(
            is_at_least_one_margin_parameter_non_zero,
            self.margin_enabled,
        );
        builder.conditional_assert_false(self.is_enabled, should_be_false);

        // USDC checks
        {
            let is_usdc_asset = builder.is_equal_constant(self.asset_index, USDC_ASSET_INDEX);

            // USDC must have margin index 0
            let is_usdc_margin_index_zero = builder.is_equal_constant(
                tx_state.next_margin_asset_index,
                USDC_MARGIN_ASSET_INDEX as u64,
            );
            let is_usdc_margin_index_not_zero =
                builder.and_not(is_usdc_asset, is_usdc_margin_index_zero);
            builder.conditional_assert_false(self.is_enabled, is_usdc_margin_index_not_zero);

            // USDC must have margin mode enabled
            let is_usdc_margin_mode_disabled = builder.and_not(is_usdc_asset, self.margin_enabled);
            builder.conditional_assert_false(self.is_enabled, is_usdc_margin_mode_disabled);

            // USDC LTV = liq factor = LT = ASSET_MARGIN_TICK, liquidation fee = 0
            let is_usdc_ltv_correct =
                builder.is_equal_constant(self.loan_to_value, ASSET_MARGIN_TICK);
            builder.conditional_assert_true(is_usdc_asset, is_usdc_ltv_correct);
            let is_usdc_liquidation_fee_zero = builder.is_zero(self.liquidation_fee);
            builder.conditional_assert_true(is_usdc_asset, is_usdc_liquidation_fee_zero);
        }

        // Can't disable margin once it's enabled
        let margin_already_enabled =
            BoolTarget::new_unsafe(tx_state.assets[TX_ASSET_ID].margin_mode);
        let is_disabling_enabled_margin =
            builder.and_not(margin_already_enabled, self.margin_enabled);
        self.success = builder.and_not(self.success, is_disabling_enabled_margin);

        // Ensure that asset is not empty
        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        self.success = builder.and_not(self.success, is_asset_empty);

        // Enabling margin requires margin index to be empty
        let success_and_enabling_margin = builder.and(self.success, self.enabling_margin);
        let is_margin_index_empty = builder.is_zero(tx_state.assets[TX_ASSET_ID].margin_index);
        builder.conditional_assert_true(success_and_enabling_margin, is_margin_index_empty);

        let is_nil_margin_index =
            builder.is_equal_constant(tx_state.next_margin_asset_index, NIL_MARGIN_ASSET_INDEX);
        builder.conditional_assert_false(success_and_enabling_margin, is_nil_margin_index);

        // It can only be ETH if we're enabling margin
        let is_eth_asset = builder.is_equal_constant(self.asset_index, NATIVE_ASSET_INDEX);
        let is_enabling_margin_and_not_eth = builder.and_not(self.enabling_margin, is_eth_asset);
        self.success = builder.and_not(self.success, is_enabling_margin_and_not_eth);
    }
}

impl Apply for L1UpdateAssetTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.assets[TX_ASSET_ID].min_transfer_amount = builder.select_biguint(
            self.success,
            &self.min_transfer_amount,
            &tx_state.assets[TX_ASSET_ID].min_transfer_amount,
        );
        tx_state.assets[TX_ASSET_ID].min_withdrawal_amount = builder.select_biguint(
            self.success,
            &self.min_withdrawal_amount,
            &tx_state.assets[TX_ASSET_ID].min_withdrawal_amount,
        );
        tx_state.assets[TX_ASSET_ID].margin_mode = builder.select(
            self.success,
            self.margin_mode,
            tx_state.assets[TX_ASSET_ID].margin_mode,
        );

        let success_and_enabling_margin = builder.and(self.success, self.enabling_margin);
        tx_state.assets[TX_ASSET_ID].margin_index = builder.select(
            success_and_enabling_margin,
            tx_state.next_margin_asset_index,
            tx_state.assets[TX_ASSET_ID].margin_index,
        );
        tx_state.first_asset_margin_index = builder.select(
            success_and_enabling_margin,
            tx_state.next_margin_asset_index,
            tx_state.first_asset_margin_index,
        );

        let success_and_margin_enabled = builder.and(self.success, self.margin_enabled);
        tx_state.margined_asset[TX_ASSET_ID] = select_margined_asset_target(
            builder,
            success_and_margin_enabled,
            &MarginedAssetTarget {
                asset_index: self.asset_index,
                loan_to_value: self.loan_to_value,
                liquidation_threshold: self.liquidation_threshold,
                liquidation_factor: self.liquidation_factor,
                liquidation_fee: self.liquidation_fee,
                index_price: self.index_price,
                index_price_divider: self.index_price_divider,
                ..Default::default()
            },
            &tx_state.margined_asset[TX_ASSET_ID],
        );

        self.success
    }
}

pub trait L1UpdateAssetTxTargetWitness<F: PrimeField64> {
    fn set_l1_update_asset_tx_target(
        &mut self,
        a: &L1UpdateAssetTxTarget,
        b: &L1UpdateAssetTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1UpdateAssetTxTargetWitness<F> for T {
    fn set_l1_update_asset_tx_target(
        &mut self,
        a: &L1UpdateAssetTxTarget,
        b: &L1UpdateAssetTx,
    ) -> Result<()> {
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index as i64))?;
        self.set_biguint_target(
            &a.min_transfer_amount,
            &BigUint::from_u64(b.min_transfer_amount as u64).unwrap(),
        )?;
        self.set_biguint_target(
            &a.min_withdrawal_amount,
            &BigUint::from_u64(b.min_withdrawal_amount as u64).unwrap(),
        )?;
        self.set_target(a.margin_mode, F::from_canonical_u8(b.margin_mode))?;
        self.set_target(a.loan_to_value, F::from_canonical_u16(b.loan_to_value))?;
        self.set_target(
            a.liquidation_threshold,
            F::from_canonical_u16(b.liquidation_threshold),
        )?;
        self.set_target(
            a.liquidation_factor,
            F::from_canonical_u16(b.liquidation_factor),
        )?;
        self.set_target(a.liquidation_fee, F::from_canonical_u32(b.liquidation_fee))?;
        self.set_target(a.index_price, F::from_canonical_i64(b.index_price))?;
        self.set_target(
            a.index_price_divider,
            F::from_canonical_i64(b.index_price_divider),
        )?;

        Ok(())
    }
}
