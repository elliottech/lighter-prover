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
use crate::types::asset::{AssetTarget, ensure_valid_asset_index, select_asset_target};
use crate::types::config::{BIG_U64_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::margined_asset::{MarginedAssetTarget, select_margined_asset_target};
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct L1RegisterAssetTx {
    #[serde(rename = "ai")]
    pub asset_index: i16, // 6 bits
    #[serde(rename = "em")]
    pub extension_multiplier: i64, // 56 bits
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
    #[serde(rename = "ipd", default)]
    pub index_price_divider: i64,
    #[serde(rename = "mi", default)]
    pub margin_index: u8, // Given by sequencer, circuit verifies that given index was empty
    #[serde(rename = "ip", default)]
    pub index_price: i64, // Given by sequencer
}

#[derive(Debug)]
pub struct L1RegisterAssetTxTarget {
    pub asset_index: Target,
    pub extension_multiplier: BigUintTarget,
    pub min_transfer_amount: BigUintTarget,
    pub min_withdrawal_amount: BigUintTarget,
    pub margin_mode: Target,
    pub loan_to_value: Target,
    pub liquidation_threshold: Target,
    pub liquidation_factor: Target,
    pub liquidation_fee: Target,
    pub index_price_divider: Target,
    pub margin_index: Target,
    pub index_price: Target,

    // helpers
    margin_enabled: BoolTarget,

    // output
    is_enabled: BoolTarget,
    success: BoolTarget,
}

impl L1RegisterAssetTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            asset_index: builder.add_virtual_target(),
            extension_multiplier: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_transfer_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_withdrawal_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            margin_mode: builder.add_virtual_target(),
            loan_to_value: builder.add_virtual_target(),
            liquidation_threshold: builder.add_virtual_target(),
            liquidation_factor: builder.add_virtual_target(),
            liquidation_fee: builder.add_virtual_target(),
            index_price_divider: builder.add_virtual_target(),
            margin_index: builder.add_virtual_target(),
            index_price: builder.add_virtual_target(),

            is_enabled: BoolTarget::default(),
            margin_enabled: builder._false(),

            success: BoolTarget::default(),
        }
    }
}

impl PriorityOperationsPubData for L1RegisterAssetTxTarget {
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
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_REGISTER_ASSET),
            add_target(builder, bytes, self.asset_index, 16),
            add_target(
                builder,
                bytes,
                self.extension_multiplier.limbs[1].0,
                EXTENSION_MULTIPLIER_BITS % 32,
            ),
            add_target(builder, bytes, self.extension_multiplier.limbs[0].0, 32),
            add_big_uint_target(builder, bytes, &self.min_transfer_amount),
            add_big_uint_target(builder, bytes, &self.min_withdrawal_amount),
            add_byte_target_unsafe(bytes, self.margin_mode),
            add_target(builder, bytes, self.loan_to_value, 16),
            add_target(builder, bytes, self.liquidation_threshold, 16),
            add_target(builder, bytes, self.liquidation_factor, 16),
            add_target(builder, bytes, self.liquidation_fee, 32),
            add_target(builder, bytes, self.index_price_divider, 56),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, byte_count),
        )
    }
}

impl Verify for L1RegisterAssetTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_types: &TxTypeTargets, tx_state: &TxState) {
        self.success = tx_types.is_l1_register_asset;
        self.is_enabled = tx_types.is_l1_register_asset;

        builder.conditional_assert_eq(
            self.is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            self.asset_index,
        );
        ensure_valid_asset_index(builder, self.is_enabled, self.asset_index);

        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.extension_multiplier);
        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.min_transfer_amount);
        builder.conditional_assert_not_zero_biguint(self.is_enabled, &self.min_withdrawal_amount);

        builder.range_check_biguint(&self.extension_multiplier, EXTENSION_MULTIPLIER_BITS);
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

        let max_index_price_divider = builder.constant_u64(MAX_INDEX_PRICE_DIVIDER);
        builder.conditional_assert_lte(
            self.is_enabled,
            self.index_price_divider,
            max_index_price_divider,
            64,
        );
        builder.conditional_assert_not_zero(self.is_enabled, self.index_price_divider);

        builder.register_range_check(self.margin_index, MARGINED_ASSET_LIST_SIZE_BITS);
        builder.register_range_check(self.index_price, MAX_ASSET_PRICE_BITS);

        self.margin_enabled = BoolTarget::new_unsafe(self.margin_mode);

        // LTV can't be zero if margin mode is enabled
        let is_ltv_zero = builder.is_zero(self.loan_to_value);
        let should_be_false = builder.and(self.margin_enabled, is_ltv_zero);
        builder.conditional_assert_false(self.is_enabled, should_be_false);

        // LTV, LT, and liquidation fee should be 0 if margin mode is disabled
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
            let is_usdc_margin_index_zero =
                builder.is_equal_constant(self.margin_index, USDC_MARGIN_ASSET_INDEX as u64);
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

        // Ensure that asset is empty
        let is_asset_empty = tx_state.assets[TX_ASSET_ID].is_empty(builder);
        self.success = builder.and(self.success, is_asset_empty);

        // If margin is disabled, margin index should be nil.
        let success_and_margin_disabled = builder.and_not(self.success, self.margin_enabled);
        builder.conditional_assert_eq_constant(
            success_and_margin_disabled,
            self.margin_index,
            NIL_MARGIN_ASSET_INDEX,
        );

        // If margin is enabled but margin index is given as nil by the sequencer, reject the transaction
        let is_margin_index_nil =
            builder.is_equal_constant(self.margin_index, NIL_MARGIN_ASSET_INDEX);
        let is_margin_index_nil_and_margin_enabled =
            builder.and(self.margin_enabled, is_margin_index_nil);
        self.success = builder.and_not(self.success, is_margin_index_nil_and_margin_enabled);

        // Only allow usdc and eth ti be margin enabled for now
        let is_usdc = builder.is_equal_constant(self.asset_index, USDC_ASSET_INDEX);
        let is_eth = builder.is_equal_constant(self.asset_index, NATIVE_ASSET_INDEX);
        let is_allowed_margin_asset = builder.or(is_usdc, is_eth);
        let is_margin_enabled_for_disallowed_asset =
            builder.and_not(self.margin_enabled, is_allowed_margin_asset);
        self.success = builder.and_not(self.success, is_margin_enabled_for_disallowed_asset);

        // If margin enabled and margin index is not nil, check that margin index is empty
        let is_margin_index_empty = tx_state.margined_asset[TX_ASSET_ID].is_empty(builder);
        let success_and_margin_enabled = builder.and(self.success, self.margin_enabled);
        builder.conditional_assert_true(success_and_margin_enabled, is_margin_index_empty);
    }
}

impl Apply for L1RegisterAssetTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let zero_biguint = builder.zero_biguint();
        let zero = builder.zero();

        let is_margin_index_nil =
            builder.is_equal_constant(self.margin_index, NIL_MARGIN_ASSET_INDEX);
        let margin_index = builder.select(is_margin_index_nil, zero, self.margin_index);
        tx_state.assets[TX_ASSET_ID] = select_asset_target(
            builder,
            self.success,
            &AssetTarget {
                extension_multiplier: self.extension_multiplier.clone(),
                min_transfer_amount: self.min_transfer_amount.clone(),
                min_withdrawal_amount: self.min_withdrawal_amount.clone(),
                margin_mode: self.margin_mode,
                margin_index,
            },
            &tx_state.assets[TX_ASSET_ID],
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
                global_supply_cap: zero_biguint.clone(),
                user_supply_cap: zero_biguint.clone(),
                total_supplied_amount: zero_biguint.clone(),
            },
            &tx_state.margined_asset[TX_ASSET_ID],
        );

        self.success
    }
}

pub trait L1RegisterAssetTxTargetWitness<F: PrimeField64> {
    fn set_l1_register_asset_tx_target(
        &mut self,
        a: &L1RegisterAssetTxTarget,
        b: &L1RegisterAssetTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1RegisterAssetTxTargetWitness<F> for T {
    fn set_l1_register_asset_tx_target(
        &mut self,
        a: &L1RegisterAssetTxTarget,
        b: &L1RegisterAssetTx,
    ) -> Result<()> {
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index as i64))?;
        self.set_biguint_target(
            &a.extension_multiplier,
            &BigUint::from_u64(b.extension_multiplier as u64).unwrap(),
        )?;
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
        self.set_target(a.liquidation_fee, F::from_canonical_u32(b.liquidation_fee))?;
        self.set_target(
            a.index_price_divider,
            F::from_canonical_i64(b.index_price_divider),
        )?;
        self.set_target(
            a.liquidation_factor,
            F::from_canonical_u16(b.liquidation_factor),
        )?;
        self.set_target(a.margin_index, F::from_canonical_u8(b.margin_index))?;
        self.set_target(a.index_price, F::from_canonical_i64(b.index_price))?;

        Ok(())
    }
}
