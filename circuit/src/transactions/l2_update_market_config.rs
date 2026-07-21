// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::market_details::MarketFlags;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2UpdateMarketConfigTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "mi", default)]
    pub market_index: i16,
    #[serde(rename = "si", default)]
    pub strategy_index: u8,
    #[serde(rename = "mf", default)]
    pub market_flags: i64,
    #[serde(rename = "fpm", default)]
    pub funding_premium_multiplier: u16,
}

#[derive(Debug)]
pub struct L2UpdateMarketConfigTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub market_index: Target,
    pub strategy_index: Target,
    pub market_flags: Target,
    pub funding_premium_multiplier: Target,

    // Output
    success: BoolTarget,
}

impl L2UpdateMarketConfigTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),

            market_index: builder.add_virtual_target(),
            strategy_index: builder.add_virtual_target(),
            market_flags: builder.add_virtual_target(),
            funding_premium_multiplier: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        builder.register_range_check(self.strategy_index, STRATEGY_INDEX_BITS);
        builder.register_range_check(
            self.funding_premium_multiplier,
            FUNDING_PREMIUM_MULTIPLIER_BITS,
        );
    }
}

impl TxHash for L2UpdateMarketConfigTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_MARKET_CONFIG)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.strategy_index,
            self.market_flags,
            self.funding_premium_multiplier,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateMarketConfigTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_market_config;
        self.success = is_enabled;

        self.register_range_checks(builder);

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        // Limit to insurance fund account index
        builder.conditional_assert_eq_constant(
            is_enabled,
            self.account_index,
            INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX as u64,
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

        // Make sure market is active
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market_details.status,
            MARKET_STATUS_ACTIVE as u64,
        );

        // Strategy 1 is reserved for the insurance fund's spot USDC, so a perps
        // market must not be assigned to it.
        let is_spot_strategy = builder.is_equal_constant(
            self.strategy_index,
            INSURANCE_FUND_SPOT_STRATEGY_INDEX as u64,
        );
        builder.conditional_assert_false(is_enabled, is_spot_strategy);

        // Guard transition to isolated-only
        let old_market_flags =
            MarketFlags::from_target(builder, tx_state.market_details.market_flags);
        let new_market_flags = MarketFlags::from_target(builder, self.market_flags);
        let new_is_isolated = new_market_flags.is_isolated_only();
        let becomes_isolated_only =
            builder.and_not(new_is_isolated, old_market_flags.is_isolated_only());

        let is_open_interest_zero = builder.is_zero(tx_state.market_details.open_interest);
        let is_total_order_count_zero = builder.is_zero(tx_state.market.total_order_count);
        let is_no_activity_on_market =
            builder.and(is_open_interest_zero, is_total_order_count_zero);
        let should_be_false = builder.and_not(becomes_isolated_only, is_no_activity_on_market);
        builder.conditional_assert_false(is_enabled, should_be_false);

        // Isolated-only markets must default to isolated margin mode.
        let new_not_isolated_fallback =
            builder.not(BoolTarget::new_unsafe(new_market_flags.default_margin_mode));
        let isolated_only_without_isolated_default =
            builder.and(new_is_isolated, new_not_isolated_fallback);
        builder.conditional_assert_false(is_enabled, isolated_only_without_isolated_default);

        // Funding rate multiplier must be in (0, FUNDING_PREMIUM_MULTIPLIER_TICK]
        builder.conditional_assert_not_zero(is_enabled, self.funding_premium_multiplier);
        let funding_premium_multiplier_tick = builder.constant_u64(FUNDING_PREMIUM_MULTIPLIER_TICK);
        builder.conditional_assert_lte(
            is_enabled,
            self.funding_premium_multiplier,
            funding_premium_multiplier_tick,
            FUNDING_PREMIUM_MULTIPLIER_BITS,
        );
    }
}

impl Apply for L2UpdateMarketConfigTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.market_details.strategy_index = builder.select(
            self.success,
            self.strategy_index,
            tx_state.market_details.strategy_index,
        );
        tx_state.market_details.market_flags = builder.select(
            self.success,
            self.market_flags,
            tx_state.market_details.market_flags,
        );
        tx_state.market_details.funding_premium_multiplier = builder.select(
            self.success,
            self.funding_premium_multiplier,
            tx_state.market_details.funding_premium_multiplier,
        );

        self.success
    }
}

pub trait L2UpdateMarketConfigTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_market_config_tx_target(
        &mut self,
        a: &L2UpdateMarketConfigTxTarget,
        b: &L2UpdateMarketConfigTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateMarketConfigTxTargetWitness<F> for T {
    fn set_l2_update_market_config_tx_target(
        &mut self,
        a: &L2UpdateMarketConfigTxTarget,
        b: &L2UpdateMarketConfigTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;
        self.set_target(a.strategy_index, F::from_canonical_u8(b.strategy_index))?;
        self.set_target(a.market_flags, F::from_canonical_i64(b.market_flags))?;
        self.set_target(
            a.funding_premium_multiplier,
            F::from_canonical_u16(b.funding_premium_multiplier),
        )?;

        Ok(())
    }
}
