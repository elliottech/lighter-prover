// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::CircuitBuilderBiguint16;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct L2UpdateLeverageTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "imf")]
    pub initial_margin_fraction: u16,

    #[serde(rename = "mmd")]
    pub margin_mode: u8,
}

#[derive(Debug)]
pub struct L2UpdateLeverageTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub market_index: Target,
    pub initial_margin_fraction: Target, // 16 bits
    pub margin_mode: Target,

    // helpers
    is_position_active_on_market: BoolTarget,

    // output
    pub success: BoolTarget,
}

impl L2UpdateLeverageTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdateLeverageTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            initial_margin_fraction: builder.add_virtual_target(),
            margin_mode: builder.add_virtual_target(),

            // helpers
            is_position_active_on_market: BoolTarget::default(),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateLeverageTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_LEVERAGE)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.initial_margin_fraction,
            self.margin_mode,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateLeverageTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_leverage;
        self.success = is_enabled;

        // Check account/apikey and market consistency
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

        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );

        // cancel mode - Possible values: CROSS_MARGIN(0) and ISOLATED_MARGIN(1)
        builder.assert_bool(BoolTarget::new_unsafe(self.margin_mode));

        // We only allow to update margin mode if there is no active position or order in the market
        self.is_position_active_on_market =
            tx_state.positions[OWNER_ACCOUNT_ID].is_order_or_position_open(builder);
        let is_margin_mode_changed = builder.is_not_equal(
            tx_state.positions[OWNER_ACCOUNT_ID].margin_mode,
            self.margin_mode,
        );
        let is_enabled_and_margin_mode_changed = builder.and(is_enabled, is_margin_mode_changed);
        builder.conditional_assert_false(
            is_enabled_and_margin_mode_changed,
            self.is_position_active_on_market,
        );

        // Check if original margin is healthy
        let is_in_liquidation = tx_state.risk_infos[OWNER_ACCOUNT_ID]
            .current_risk_parameters
            .is_in_liquidation(builder);
        builder.conditional_assert_false(is_enabled, is_in_liquidation);

        // Check if the market is active
        let active_market_status = builder.constant_from_u8(MARKET_STATUS_ACTIVE);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.market_details.status,
            active_market_status,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );

        // Check if the initial margin fraction is within the allowed range
        builder.register_range_check(self.initial_margin_fraction, MARGIN_FRACTION_BITS);
        let is_initial_margin_fraction_valid = builder.is_lte(
            tx_state.market_details.min_initial_margin_fraction,
            self.initial_margin_fraction,
            MARGIN_FRACTION_BITS,
        );
        builder.conditional_assert_true(is_enabled, is_initial_margin_fraction_valid);

        // We don't allow pools to change their margin mode to isolated
        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
        let is_new_margin_mode_isolated = builder.is_equal(self.margin_mode, isolated_margin_mode);
        let public_pool_account_type = builder.constant_from_u8(PUBLIC_POOL_ACCOUNT_TYPE);
        let insurance_fund_account_type = builder.constant_from_u8(INSURANCE_FUND_ACCOUNT_TYPE);
        let is_public_pool_account_type = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            public_pool_account_type,
        );
        let is_insurance_fund_account_type = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            insurance_fund_account_type,
        );
        let is_account_pool =
            builder.or(is_public_pool_account_type, is_insurance_fund_account_type);
        let is_account_pool_and_new_margin_mode_isolated =
            builder.and(is_account_pool, is_new_margin_mode_isolated);
        builder.conditional_assert_false(is_enabled, is_account_pool_and_new_margin_mode_isolated);
    }
}

impl Apply for L2UpdateLeverageTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.positions[OWNER_ACCOUNT_ID].initial_margin_fraction = builder.select(
            self.success,
            self.initial_margin_fraction,
            tx_state.positions[OWNER_ACCOUNT_ID].initial_margin_fraction,
        );
        tx_state.positions[OWNER_ACCOUNT_ID].margin_mode = builder.select(
            self.success,
            self.margin_mode,
            tx_state.positions[OWNER_ACCOUNT_ID].margin_mode,
        );

        let old_risk_parameters = tx_state.risk_infos[OWNER_ACCOUNT_ID]
            .current_risk_parameters
            .clone();

        let margin_fraction_multiplier = builder.constant_u64(MARGIN_FRACTION_MULTIPLIER as u64);
        let normalized_position_notional_multiplier = builder.mul_many([
            tx_state.market_details.mark_price,       // 32 bits
            tx_state.market_details.quote_multiplier, // 14 bits
            margin_fraction_multiplier,               // 7 bits
        ]);
        let normalized_position_notional_multiplier =
            builder.target_to_biguint(normalized_position_notional_multiplier); // 59 bits

        let position_initial_margin_fraction = tx_state.positions[OWNER_ACCOUNT_ID]
            .get_initial_margin_fraction(
                builder,
                tx_state.market_details.default_initial_margin_fraction,
                tx_state.market_details.min_initial_margin_fraction,
            ); // 16 bits
        let position_initial_margin_fraction_big =
            builder.target_to_biguint_single_limb_unsafe(position_initial_margin_fraction);

        // mark_price * quote_multiplier * margin_fraction_multiplier * initial_margin_fraction
        let common_multiplier = builder.mul_biguint_non_carry(
            &normalized_position_notional_multiplier,
            &position_initial_margin_fraction_big,
            BIG_U96_LIMBS,
        );

        let position_abs_big =
            builder.biguint_u16_to_biguint(&tx_state.positions[OWNER_ACCOUNT_ID].position.abs);
        let position_requirement =
            builder.mul_biguint_non_carry(&position_abs_big, &common_multiplier, BIG_U96_LIMBS);

        let mut new_risk_parameters = old_risk_parameters.clone();
        new_risk_parameters.initial_margin_requirement = position_requirement;

        let is_valid_risk_change =
            old_risk_parameters.is_valid_risk_change(builder, &new_risk_parameters);
        let is_success_and_is_position_active_on_market =
            builder.and(self.success, self.is_position_active_on_market);
        builder.conditional_assert_true(
            is_success_and_is_position_active_on_market,
            is_valid_risk_change,
        );

        self.success
    }
}

pub trait L2UpdateLeverageTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_leverage_tx_target(
        &mut self,
        a: &L2UpdateLeverageTxTarget,
        b: &L2UpdateLeverageTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateLeverageTxTargetWitness<F> for T {
    fn set_l2_update_leverage_tx_target(
        &mut self,
        a: &L2UpdateLeverageTxTarget,
        b: &L2UpdateLeverageTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(
            a.initial_margin_fraction,
            F::from_canonical_u16(b.initial_margin_fraction),
        )?;
        self.set_target(a.margin_mode, F::from_canonical_u8(b.margin_mode))?;

        Ok(())
    }
}
