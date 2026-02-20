// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use num::BigUint;
use plonky2::iop::target::{BoolTarget, Target};

use super::account::AccountTarget;
use super::account_position::{AccountPositionTarget, get_position_unrealized_pnl};
use super::config::{BIG_U96_LIMBS, Builder};
use super::constants::{
    BANKRUPTCY, MARGIN_FRACTION_MULTIPLIER, POSITION_LIST_SIZE, USDC_TO_COLLATERAL_MULTIPLIER,
};
use super::market_details::{MarketDetailsTarget, select_market_details};
use crate::bigint::big_u16::{CircuitBuilderBigIntU16, CircuitBuilderBiguint16};
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::unsafe_big::{CircuitBuilderUnsafeBig, UnsafeBigTarget};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::signed::signed_target::CircuitBuilderSigned;
use crate::types::config::{BIG_U64_LIMBS, BIGU16_U112_LIMBS};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Default)]
pub struct RiskInfoTarget {
    // Risk parameters for the cross margin, includes all cross positions
    pub cross_risk_parameters: RiskParametersTarget,
    // If current market is isolated, this will be the risk parameters for the isolated market, otherwise it will be the same as cross_risk_parameters
    pub current_risk_parameters: RiskParametersTarget,
}

#[derive(Debug, Clone, Default)]
pub struct RiskParametersTarget {
    pub collateral: BigIntTarget,              // 96 bits
    pub collateral_with_funding: BigIntTarget, // 96 bits
    pub total_account_value: BigIntTarget,     // 96 bits
    pub initial_margin_requirement: BigUintTarget,
    pub maintenance_margin_requirement: BigUintTarget,
    pub close_out_margin_requirement: BigUintTarget,
}

impl RiskInfoTarget {
    pub fn new(
        builder: &mut Builder,
        account: &AccountTarget,
        position: &AccountPositionTarget,
        current_market_details: &MarketDetailsTarget,
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
        strategy_index: Target, // Assumed not to be nil strategy index
    ) -> Self {
        let usdc_to_collateral_multiplier =
            BigUintTarget::from(builder.constant_u32(USDC_TO_COLLATERAL_MULTIPLIER));

        let (position_base_notional_values, cross_position_base_notional_value) =
            get_cross_position_base_notional_values(
                builder,
                &account.positions,
                all_market_details,
                strategy_index,
            );

        let (isolated_position_notional, isolated_position_base_notinal_value) = {
            let zero = builder.zero();
            let one = builder.one();
            let (isolated_position_notional, isolated_positive_tpv_sum, isolated_negative_tpv_sum) =
                position_base_notional(builder, position, current_market_details, strategy_index);
            let is_positive_tpv_sum_zero = builder.is_zero(isolated_positive_tpv_sum);
            let add_sign = builder.select(is_positive_tpv_sum_zero, zero, one);
            let big_positive_tpv_sum = BigIntTarget {
                abs: builder.target_to_biguint(isolated_positive_tpv_sum),
                sign: SignTarget::new_unsafe(add_sign),
            };

            let is_negative_tpv_sum_zero = builder.is_zero(isolated_negative_tpv_sum);
            let add_sign = builder.select(is_negative_tpv_sum_zero, zero, one);
            let big_negative_tpv_sum = BigIntTarget {
                abs: builder.target_to_biguint(isolated_negative_tpv_sum),
                sign: SignTarget::new_unsafe(add_sign),
            };
            (
                builder.target_to_biguint(isolated_position_notional),
                builder.sub_bigint_non_carry(
                    &big_positive_tpv_sum,
                    &big_negative_tpv_sum,
                    BIG_U96_LIMBS,
                ),
            )
        };

        let cross_position_notional_value = builder.mul_bigint_with_biguint_non_carry(
            &cross_position_base_notional_value,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        let isolated_position_notional_value = builder.mul_bigint_with_biguint_non_carry(
            &isolated_position_base_notinal_value,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        let cross_funding = get_cross_unrealized_funding(
            builder,
            &account.positions,
            all_market_details,
            strategy_index,
        );
        let isolated_funding =
            position_unrealized_funding(builder, position, current_market_details);

        let cross_collateral = account.collateral.clone();
        let cross_collateral_with_funding =
            builder.add_bigint_non_carry(&cross_collateral, &cross_funding, BIG_U96_LIMBS);
        let cross_total_account_value = builder.add_bigint_non_carry(
            &cross_collateral_with_funding,
            &cross_position_notional_value,
            BIG_U96_LIMBS,
        );

        let isolated_collateral = position.allocated_margin.clone();
        let isolated_collateral_with_funding =
            builder.add_bigint_non_carry(&isolated_collateral, &isolated_funding, BIG_U96_LIMBS);
        let isolated_total_account_value = builder.add_bigint_non_carry(
            &isolated_collateral_with_funding,
            &isolated_position_notional_value,
            BIG_U96_LIMBS,
        );

        let cross_initial_margin_requirement = get_initial_margin_requirement(
            builder,
            &account.positions,
            &position_base_notional_values,
            all_market_details,
        );

        let cross_maintenance_margin_requirement = get_maintenance_margin_requirement(
            builder,
            &account.positions,
            &position_base_notional_values,
            all_market_details,
        );

        let cross_close_out_margin_requirement = get_close_out_margin_requirement(
            builder,
            &account.positions,
            &position_base_notional_values,
            all_market_details,
        );

        let cross_risk_parameters = RiskParametersTarget {
            collateral: cross_collateral,
            total_account_value: cross_total_account_value,
            collateral_with_funding: cross_collateral_with_funding,
            initial_margin_requirement: cross_initial_margin_requirement,
            maintenance_margin_requirement: cross_maintenance_margin_requirement,
            close_out_margin_requirement: cross_close_out_margin_requirement,
        };

        let (
            isolated_initial_margin_requirement,
            isolated_maintenance_margin_requirement,
            isolated_close_out_margin_requirement,
        ) = position_margin_requirements(
            builder,
            position,
            &isolated_position_notional,
            current_market_details,
        );
        let isolated_risk_parameters = RiskParametersTarget {
            collateral: isolated_collateral,
            total_account_value: isolated_total_account_value,
            collateral_with_funding: isolated_collateral_with_funding,
            initial_margin_requirement: isolated_initial_margin_requirement,
            maintenance_margin_requirement: isolated_maintenance_margin_requirement,
            close_out_margin_requirement: isolated_close_out_margin_requirement,
        };

        let current_risk_parameters = RiskParametersTarget::select(
            builder,
            position.is_isolated(),
            &isolated_risk_parameters,
            &cross_risk_parameters,
        );

        RiskInfoTarget {
            cross_risk_parameters,
            current_risk_parameters,
        }

        // RiskInfoTarget {
        //     cross_risk_parameters: RiskParametersTarget::default(),
        //     current_risk_parameters: RiskParametersTarget::default(),
        // }
    }
}

impl RiskParametersTarget {
    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_bigint(&self.collateral, &format!("{} collateral", tag));
        builder.println_bigint(
            &self.collateral_with_funding,
            &format!("{} collateral with funding", tag),
        );
        builder.println_bigint(
            &self.total_account_value,
            &format!("{} total account value", tag),
        );
        builder.println_biguint(
            &self.initial_margin_requirement,
            &format!("{} initial margin requirement", tag),
        );
        builder.println_biguint(
            &self.maintenance_margin_requirement,
            &format!("{} maintenance margin requirement", tag),
        );
        builder.println_biguint(
            &self.close_out_margin_requirement,
            &format!("{} close out margin requirement", tag),
        );
    }

    pub fn get_health(&self, builder: &mut Builder) -> Target {
        let neg_one = builder.neg_one();

        let is_tav_negative = builder.is_equal(self.total_account_value.sign.target, neg_one);

        let initial_margin_gt = builder.is_lt_biguint(
            &self.total_account_value.abs,
            &self.initial_margin_requirement,
        );
        let maintenance_margin_gt = builder.is_lt_biguint(
            &self.total_account_value.abs,
            &self.maintenance_margin_requirement,
        );
        let close_out_margin_gt = builder.is_lt_biguint(
            &self.total_account_value.abs,
            &self.close_out_margin_requirement,
        );

        let positive_tav_result = builder.add_many([
            initial_margin_gt.target,
            maintenance_margin_gt.target,
            close_out_margin_gt.target,
        ]);

        // If total account value is negative, health status is BANKRUPTCY
        // Otherwise, positive_tav_result could be 0 to 3 i.e. HEALTHY to FULL_LIQUIDATION
        let bancruptcy = builder.constant_from_u8(BANKRUPTCY);
        builder.select(is_tav_negative, bancruptcy, positive_tav_result)
    }

    pub fn is_healthy(&self, builder: &mut Builder) -> BoolTarget {
        let neg_one = builder.neg_one();
        let tav_is_not_negative =
            builder.is_not_equal(self.total_account_value.sign.target, neg_one);
        let abs_tav_gte_initial_margin = builder.is_gte_biguint(
            &self.total_account_value.abs,
            &self.initial_margin_requirement,
        );
        builder.and(tav_is_not_negative, abs_tav_gte_initial_margin)
    }

    /// Returns true if health < PRE_LIQUIDATION
    pub fn is_not_in_liquidation(&self, builder: &mut Builder) -> BoolTarget {
        let neg_one = builder.neg_one();
        let tav_is_not_negative =
            builder.is_not_equal(self.total_account_value.sign.target, neg_one);
        let abs_tav_gte_maintenance_margin = builder.is_gte_biguint(
            &self.total_account_value.abs,
            &self.maintenance_margin_requirement,
        );
        builder.and(tav_is_not_negative, abs_tav_gte_maintenance_margin)
    }

    fn is_health_improved(&self, builder: &mut Builder, new: &Self) -> BoolTarget {
        let left_side = builder.mul_bigint_with_biguint_non_carry(
            &self.total_account_value,
            &new.maintenance_margin_requirement,
            self.total_account_value.abs.limbs.len()
                + new.maintenance_margin_requirement.limbs.len(),
        );
        let right_side = builder.mul_bigint_with_biguint_non_carry(
            &new.total_account_value,
            &self.maintenance_margin_requirement,
            new.total_account_value.abs.limbs.len()
                + self.maintenance_margin_requirement.limbs.len(),
        );

        builder.is_lte_bigint(&left_side, &right_side)
    }

    pub fn is_valid_risk_change(&self, builder: &mut Builder, new: &Self) -> BoolTarget {
        // 1. If new account collateral is not within [-2^96, 2^96], return false
        // 2. If the account is below initial margin requirement, health should improve
        // 3. If the account is above initial margin, it should stay above initial margin requirement

        let is_healthy_before = self.is_healthy(builder);
        let is_health_improved = self.is_health_improved(builder, new);
        let cond_1 = builder.or(is_healthy_before, is_health_improved);

        let is_not_healthy_before = builder.not(is_healthy_before);
        let is_healthy_after = new.is_healthy(builder);
        let cond_2 = builder.or(is_not_healthy_before, is_healthy_after);

        builder.and(cond_1, cond_2)
    }

    pub fn is_in_liquidation(&self, builder: &mut Builder) -> BoolTarget {
        let neg_one = builder.neg_one();
        let is_tav_negative = builder.is_equal(self.total_account_value.sign.target, neg_one);
        let is_tav_abs_less_than_mmr = builder.is_lt_biguint(
            &self.total_account_value.abs,
            &self.maintenance_margin_requirement,
        );
        builder.or(is_tav_negative, is_tav_abs_less_than_mmr)
    }

    pub fn update(
        &self,
        builder: &mut Builder,
        collateral_delta: &BigIntTarget,
        old_position: &AccountPositionTarget,
        new_position: &AccountPositionTarget,
        market_details: &MarketDetailsTarget,
        is_enabled: BoolTarget,
    ) -> Self {
        let zero_bigint = builder.zero_bigint();
        let empty_position = AccountPositionTarget::empty(builder);
        let empty_market = MarketDetailsTarget::empty(builder);

        // Prevent overflow when inactive
        let collateral_delta = builder.select_bigint(is_enabled, collateral_delta, &zero_bigint);
        let old_position = AccountPositionTarget::select_position(
            builder,
            is_enabled,
            old_position,
            &empty_position,
        );
        let new_position = AccountPositionTarget::select_position(
            builder,
            is_enabled,
            new_position,
            &empty_position,
        );
        let market_details =
            select_market_details(builder, is_enabled, market_details, &empty_market);

        // Apply collateral delta
        let collateral =
            builder.add_bigint_non_carry(&self.collateral, &collateral_delta, BIG_U96_LIMBS);
        let collateral_with_funding = builder.add_bigint_non_carry(
            &self.collateral_with_funding,
            &collateral_delta,
            BIG_U96_LIMBS,
        );

        // Apply total account value delta
        let mut total_account_value = builder.add_bigint_non_carry(
            &self.total_account_value,
            &collateral_delta,
            BIG_U96_LIMBS,
        );

        // Update position value changes to the total account value
        let old_position_abs = builder.biguint_u16_to_target(&old_position.position.abs);
        let old_notional = get_position_unrealized_pnl(
            builder,
            &market_details,
            old_position_abs,
            old_position.position.sign,
            old_position.entry_quote,
        );
        let new_position_abs = builder.biguint_u16_to_target(&new_position.position.abs);
        let new_notional = get_position_unrealized_pnl(
            builder,
            &market_details,
            new_position_abs,
            new_position.position.sign,
            new_position.entry_quote,
        );

        let notional_diff = builder.sub_signed(new_notional, old_notional);
        let notional_diff_big = builder.signed_target_to_bigint(notional_diff);

        let usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        let total_account_value_delta = builder.mul_bigint_with_biguint_non_carry(
            &notional_diff_big,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        total_account_value = builder.add_bigint_non_carry(
            &total_account_value,
            &total_account_value_delta,
            BIG_U96_LIMBS,
        );

        // Update margin requirements for the position change
        let margin_fraction_multiplier = builder.constant_u64(MARGIN_FRACTION_MULTIPLIER as u64);
        let normalized_position_notional_multiplier = builder.mul_many([
            market_details.mark_price,       // 32 bits
            market_details.quote_multiplier, // 14 bits
            margin_fraction_multiplier,      // 7 bits
        ]);
        let normalized_position_notional_multiplier =
            builder.target_to_biguint(normalized_position_notional_multiplier);
        let old_position_abs_big = builder.target_to_biguint(old_position_abs);
        let new_position_abs_big = builder.target_to_biguint(new_position_abs);
        let old_normalized_position_notional_value = builder.mul_biguint_non_carry(
            &old_position_abs_big,
            &normalized_position_notional_multiplier,
            BIG_U96_LIMBS,
        );
        let new_normalized_position_notional_value = builder.mul_biguint_non_carry(
            &new_position_abs_big,
            &normalized_position_notional_multiplier,
            BIG_U96_LIMBS,
        );

        // Update initial margin requirement
        let new_position_initial_margin_fraction = new_position.get_initial_margin_fraction(
            builder,
            market_details.default_initial_margin_fraction,
            market_details.min_initial_margin_fraction,
        );
        let new_position_initial_margin_fraction_big =
            builder.target_to_biguint_single_limb_unsafe(new_position_initial_margin_fraction);
        let old_position_initial_margin_fraction = old_position.get_initial_margin_fraction(
            builder,
            market_details.default_initial_margin_fraction,
            market_details.min_initial_margin_fraction,
        );
        let old_position_initial_margin_fraction_big =
            builder.target_to_biguint_single_limb_unsafe(old_position_initial_margin_fraction);
        let initial_margin_requirement_add = builder.mul_biguint_non_carry(
            &new_position_initial_margin_fraction_big,
            &new_normalized_position_notional_value,
            BIG_U96_LIMBS,
        );
        let initial_margin_requirement_sub = builder.mul_biguint_non_carry(
            &old_position_initial_margin_fraction_big,
            &old_normalized_position_notional_value,
            BIG_U96_LIMBS,
        );
        let initial_margin_requirement = builder.add_biguint_non_carry(
            &self.initial_margin_requirement,
            &initial_margin_requirement_add,
            BIG_U96_LIMBS,
        );
        let (initial_margin_requirement, sub_success) =
            builder.try_sub_biguint(&initial_margin_requirement, &initial_margin_requirement_sub);
        builder.conditional_assert_zero(is_enabled, sub_success.0);

        // Update maintenance margin requirement
        let maintenance_margin_fraction_big = builder
            .target_to_biguint_single_limb_unsafe(market_details.maintenance_margin_fraction);
        let maintenance_margin_requirement_add = builder.mul_biguint_non_carry(
            &new_normalized_position_notional_value,
            &maintenance_margin_fraction_big,
            BIG_U96_LIMBS,
        );
        let maintenance_margin_requirement_sub = builder.mul_biguint_non_carry(
            &old_normalized_position_notional_value,
            &maintenance_margin_fraction_big,
            BIG_U96_LIMBS,
        );
        let maintenance_margin_requirement = builder.add_biguint_non_carry(
            &self.maintenance_margin_requirement,
            &maintenance_margin_requirement_add,
            BIG_U96_LIMBS,
        );
        let (maintenance_margin_requirement, sub_success) = builder.try_sub_biguint(
            &maintenance_margin_requirement,
            &maintenance_margin_requirement_sub,
        );
        builder.conditional_assert_zero(is_enabled, sub_success.0);

        // Update close out margin requirement
        let close_out_margin_fraction_big =
            builder.target_to_biguint_single_limb_unsafe(market_details.close_out_margin_fraction);
        let close_out_margin_requirement_add = builder.mul_biguint_non_carry(
            &new_normalized_position_notional_value,
            &close_out_margin_fraction_big,
            BIG_U96_LIMBS,
        );
        let close_out_margin_requirement_sub = builder.mul_biguint_non_carry(
            &old_normalized_position_notional_value,
            &close_out_margin_fraction_big,
            BIG_U96_LIMBS,
        );
        let close_out_margin_requirement = builder.add_biguint_non_carry(
            &self.close_out_margin_requirement,
            &close_out_margin_requirement_add,
            BIG_U96_LIMBS,
        );
        let (close_out_margin_requirement, sub_success) = builder.try_sub_biguint(
            &close_out_margin_requirement,
            &close_out_margin_requirement_sub,
        );
        builder.conditional_assert_zero(is_enabled, sub_success.0);

        Self {
            collateral,
            total_account_value,
            initial_margin_requirement,
            maintenance_margin_requirement,
            close_out_margin_requirement,
            collateral_with_funding,
        }
    }

    pub fn select(builder: &mut Builder, flag: BoolTarget, a: &Self, b: &Self) -> Self {
        let collateral = builder.select_bigint(flag, &a.collateral, &b.collateral);
        let collateral_with_funding =
            builder.select_bigint(flag, &a.collateral_with_funding, &b.collateral_with_funding);
        let total_account_value =
            builder.select_bigint(flag, &a.total_account_value, &b.total_account_value);
        let initial_margin_requirement = builder.select_biguint(
            flag,
            &a.initial_margin_requirement,
            &b.initial_margin_requirement,
        );
        let maintenance_margin_requirement = builder.select_biguint(
            flag,
            &a.maintenance_margin_requirement,
            &b.maintenance_margin_requirement,
        );
        let close_out_margin_requirement = builder.select_biguint(
            flag,
            &a.close_out_margin_requirement,
            &b.close_out_margin_requirement,
        );

        Self {
            collateral,
            collateral_with_funding,
            total_account_value,
            initial_margin_requirement,
            maintenance_margin_requirement,
            close_out_margin_requirement,
        }
    }
}

fn position_base_notional(
    builder: &mut Builder,
    position: &AccountPositionTarget,
    market_details: &MarketDetailsTarget,
    given_strategy_index: Target,
) -> (Target, Target, Target) {
    let is_correct_strategy = builder.is_equal(market_details.strategy_index, given_strategy_index);

    // Compute the position notional value as Target, then convert it to BigInt
    let mark_price_times_quote_multiplier =
        builder.mul(market_details.quote_multiplier, market_details.mark_price);
    let abs_position = builder.biguint_u16_to_target(&position.position.abs);
    let abs_position = builder.select_or_zero(is_correct_strategy, abs_position);
    let abs_position_notional = builder.mul(abs_position, mark_price_times_quote_multiplier);

    let entry_quote = builder.select_or_zero(is_correct_strategy, position.entry_quote);
    let position_is_positive = builder.is_sign_positive(position.position.sign);
    let positive_tpv_component =
        builder.select(position_is_positive, abs_position_notional, entry_quote);
    let negative_tpv_component =
        builder.select(position_is_positive, entry_quote, abs_position_notional);

    (
        builder.mul(market_details.status, abs_position_notional), // Expired market (0 status) -> no margin requirement
        positive_tpv_component,
        negative_tpv_component,
    )
}

fn position_unrealized_funding(
    builder: &mut Builder,
    position: &AccountPositionTarget,
    market_details: &MarketDetailsTarget,
) -> BigIntTarget {
    let last_funding_rate_ps = builder.bigint_u16_to_bigint(&position.last_funding_rate_prefix_sum);
    let market_funding_rate_ps =
        builder.bigint_u16_to_bigint(&market_details.funding_rate_prefix_sum);
    let position = builder.bigint_u16_to_bigint(&position.position);

    let quote_multiplier = BigUintTarget::from(U32Target(market_details.quote_multiplier));

    let abs_position_times_quote_multiplier =
        builder.mul_biguint_non_carry(&position.abs, &quote_multiplier, BIG_U96_LIMBS);

    let funding_rate_ps_diff = builder.sub_bigint(&last_funding_rate_ps, &market_funding_rate_ps);

    BigIntTarget {
        abs: builder.mul_biguint_non_carry(
            &abs_position_times_quote_multiplier,
            &funding_rate_ps_diff.abs,
            BIG_U96_LIMBS,
        ),
        sign: SignTarget::new_unsafe(
            builder.mul(position.sign.target, funding_rate_ps_diff.sign.target),
        ),
    }
}

fn position_margin_requirements(
    builder: &mut Builder,
    position: &AccountPositionTarget,
    position_notional_value: &BigUintTarget,
    market_details: &MarketDetailsTarget,
) -> (BigUintTarget, BigUintTarget, BigUintTarget) {
    let margin_fraction_multiplier =
        builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));

    let initial_margin_fraction = BigUintTarget {
        // Set a single limb from initial margin fraction
        limbs: vec![U32Target(position.get_initial_margin_fraction(
            builder,
            market_details.default_initial_margin_fraction,
            market_details.min_initial_margin_fraction,
        ))],
    };
    let position_times_initial_margin = builder.mul_biguint_non_carry(
        position_notional_value,
        &initial_margin_fraction,
        BIG_U96_LIMBS,
    );
    let initial_margin_requirement = builder.mul_biguint_non_carry(
        &position_times_initial_margin,
        &margin_fraction_multiplier,
        BIG_U96_LIMBS,
    );

    let maintenance_margin_fraction = BigUintTarget {
        // Set a single limb from initial margin fraction
        limbs: vec![U32Target(market_details.maintenance_margin_fraction)],
    };
    let position_times_maintenance_margin = builder.mul_biguint_non_carry(
        position_notional_value,
        &maintenance_margin_fraction,
        BIG_U96_LIMBS,
    );
    let maintenance_margin_requirement = builder.mul_biguint_non_carry(
        &position_times_maintenance_margin,
        &margin_fraction_multiplier,
        BIG_U96_LIMBS,
    );

    let close_out_margin_fraction = BigUintTarget {
        // Set a single limb from initial margin fraction
        limbs: vec![U32Target(market_details.close_out_margin_fraction)],
    };
    let position_times_close_out_margin = builder.mul_biguint_non_carry(
        position_notional_value,
        &close_out_margin_fraction,
        BIG_U96_LIMBS,
    );
    let close_out_margin_requirement = builder.mul_biguint_non_carry(
        &position_times_close_out_margin,
        &margin_fraction_multiplier,
        BIG_U96_LIMBS,
    );

    (
        initial_margin_requirement,
        maintenance_margin_requirement,
        close_out_margin_requirement,
    )
}

fn get_cross_position_base_notional_values(
    builder: &mut Builder,
    account_positions: &[AccountPositionTarget; POSITION_LIST_SIZE],
    all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    strategy_index: Target,
) -> ([BigUintTarget; POSITION_LIST_SIZE], BigIntTarget) {
    let mut base_position_notional_values = core::array::from_fn(|_| builder.zero_biguint());

    let mut cross_positive_tpv_sum = builder.zero();
    let mut cross_negative_tpv_sum = builder.zero();

    for market_index in 0..POSITION_LIST_SIZE {
        let position = &account_positions[market_index];
        let market_details = &all_market_details[market_index];

        let (abs_position_notional, positive_tpv_component, negative_tpv_component) =
            position_base_notional(builder, position, market_details, strategy_index);

        // Accumulate cross margins
        let is_cross_position = position.is_cross(builder);
        cross_positive_tpv_sum = builder.mul_add(
            is_cross_position.target,
            positive_tpv_component,
            cross_positive_tpv_sum,
        );
        cross_negative_tpv_sum = builder.mul_add(
            is_cross_position.target,
            negative_tpv_component,
            cross_negative_tpv_sum,
        );

        base_position_notional_values[market_index] =
            builder.target_to_biguint(abs_position_notional);
    }
    // compute total position notional value from the positive and negative components
    let zero = builder.zero();
    let one = builder.one();

    let cross_position_notional_value = {
        let is_positive_tpv_sum_zero = builder.is_zero(cross_positive_tpv_sum);
        let add_sign = builder.select(is_positive_tpv_sum_zero, zero, one);
        let big_positive_tpv_sum = BigIntTarget {
            abs: builder.target_to_biguint(cross_positive_tpv_sum),
            sign: SignTarget::new_unsafe(add_sign),
        };

        let is_negative_tpv_sum_zero = builder.is_zero(cross_negative_tpv_sum);
        let add_sign = builder.select(is_negative_tpv_sum_zero, zero, one);
        let big_negative_tpv_sum = BigIntTarget {
            abs: builder.target_to_biguint(cross_negative_tpv_sum),
            sign: SignTarget::new_unsafe(add_sign),
        };
        builder.sub_bigint_non_carry(&big_positive_tpv_sum, &big_negative_tpv_sum, BIG_U96_LIMBS)
    };

    (base_position_notional_values, cross_position_notional_value)
}

fn get_cross_unrealized_funding(
    builder: &mut Builder,
    account_positions: &[AccountPositionTarget; POSITION_LIST_SIZE],
    all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    strategy_index: Target,
) -> BigIntTarget {
    let mut unsafe_unrealized_funding = UnsafeBigTarget {
        limbs: vec![builder.zero(); BIGU16_U112_LIMBS],
    };
    for market_index in 0..POSITION_LIST_SIZE {
        let market_details = all_market_details[market_index].clone();
        let position = account_positions[market_index].clone();

        let lhs = builder.sub_bigint_u16_unsafe(
            &position.last_funding_rate_prefix_sum,
            &market_details.funding_rate_prefix_sum,
        ); // (-2^17, 2^17)

        let rhs = builder
            .mul_bigint_u16_and_target_unsafe(&position.position, market_details.quote_multiplier); // (-2^30, 2^30)

        // Multiply the two unsafe bigints, where lhs and rhs each has 4 limbs.
        // Limbwise multiplication is in (-2^47, 2^47) range.
        // Resulting limbs will be at most sum of 4 different limbwise multiplications.
        // Thus resulting limbs are in the range of (-2^49, 2^49).
        let unsafe_position_unrealized_funding =
            builder.mul_unsafe_big(&lhs, &rhs, BIGU16_U112_LIMBS); // (-2^49, 2^49)

        // Accumulate the unrealized funding for at most 255 (2^8 - 1) cross positions
        let is_correct_strategy = builder.is_equal(market_details.strategy_index, strategy_index);
        let is_accumulated = builder.and_not(is_correct_strategy, position.is_isolated());
        unsafe_unrealized_funding = builder.mul_add_unsafe_big(
            &unsafe_position_unrealized_funding,
            is_accumulated.target,
            &unsafe_unrealized_funding,
        ); // (-2^57, 2^57)
    }
    let unrealized_funding =
        builder.unsafe_big16_to_bigint(&unsafe_unrealized_funding, BIGU16_U112_LIMBS);
    BigIntTarget {
        abs: builder.trim_biguint(&unrealized_funding.abs, BIG_U96_LIMBS),
        sign: unrealized_funding.sign,
    }
}

fn get_initial_margin_requirement(
    builder: &mut Builder,
    account_positions: &[AccountPositionTarget; POSITION_LIST_SIZE],
    position_notional_values: &[BigUintTarget; POSITION_LIST_SIZE],
    all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
) -> BigUintTarget {
    let margin_fraction_multiplier =
        builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));

    let mut cross_value = UnsafeBigTarget {
        limbs: vec![builder.zero(); BIG_U64_LIMBS],
    };

    for market_index in 0..POSITION_LIST_SIZE {
        let position = account_positions[market_index].clone();
        let is_cross_position = position.is_cross(builder);
        let margin_fraction = position.get_initial_margin_fraction(
            builder,
            all_market_details[market_index].default_initial_margin_fraction,
            all_market_details[market_index].min_initial_margin_fraction,
        );
        let lhs = builder.unsafe_big_from_biguint(&position_notional_values[market_index]); // each limb 32 bit
        let rhs = builder.mul(margin_fraction, is_cross_position.target); // 14 bits
        cross_value = builder.mul_add_unsafe_big(&lhs, rhs, &cross_value); // each limb 46 bit + accumulating at most 255 markets = each limb 54 bit
    }
    let cross_value = builder.unsafe_big32_to_biguint(&cross_value, BIG_U96_LIMBS);

    builder.mul_biguint_non_carry(&cross_value, &margin_fraction_multiplier, BIG_U96_LIMBS)
}

fn get_maintenance_margin_requirement(
    builder: &mut Builder,
    account_positions: &[AccountPositionTarget; POSITION_LIST_SIZE],
    position_notional_values: &[BigUintTarget; POSITION_LIST_SIZE],
    all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
) -> BigUintTarget {
    let margin_fraction_multiplier =
        builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));

    let mut cross_value = UnsafeBigTarget {
        limbs: vec![builder.zero(); BIG_U64_LIMBS],
    };

    for market_index in 0..POSITION_LIST_SIZE {
        let position = account_positions[market_index].clone();
        let is_cross_position = position.is_cross(builder);
        let lhs = builder.unsafe_big_from_biguint(&position_notional_values[market_index]); // each limb 32 bit
        let rhs = builder.mul(
            all_market_details[market_index].maintenance_margin_fraction,
            is_cross_position.target,
        ); // 14 bits
        cross_value = builder.mul_add_unsafe_big(&lhs, rhs, &cross_value); // each limb 46 bit + accumulating at most 255 markets = each limb 54 bit
    }
    // Sum of cross_values where each cross_value is 42 bits and total 2^8 markets, so each limb is 50 bit
    let cross_value = builder.unsafe_big32_to_biguint(&cross_value, BIG_U96_LIMBS);

    builder.mul_biguint_non_carry(&cross_value, &margin_fraction_multiplier, BIG_U96_LIMBS)
}

fn get_close_out_margin_requirement(
    builder: &mut Builder,
    account_positions: &[AccountPositionTarget; POSITION_LIST_SIZE],
    position_notional_values: &[BigUintTarget; POSITION_LIST_SIZE],
    all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
) -> BigUintTarget {
    let margin_fraction_multiplier =
        builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));

    let mut cross_value = UnsafeBigTarget {
        limbs: vec![builder.zero(); BIG_U64_LIMBS],
    };

    for market_index in 0..POSITION_LIST_SIZE {
        let position = account_positions[market_index].clone();
        let is_cross_position = position.is_cross(builder);
        let lhs = builder.unsafe_big_from_biguint(&position_notional_values[market_index]); // each limb 32 bit
        let rhs = builder.mul(
            all_market_details[market_index].close_out_margin_fraction,
            is_cross_position.target,
        ); // 14 bits
        cross_value = builder.mul_add_unsafe_big(&lhs, rhs, &cross_value); // each limb 46 bit + accumulating at most 255 markets = each limb 54 bit
    }
    // Sum of cross_values where each cross_value is 42 bits and total 2^8 markets, so each limb is 50 bit
    let cross_value = builder.unsafe_big32_to_biguint(&cross_value, BIG_U96_LIMBS);

    builder.mul_biguint_non_carry(&cross_value, &margin_fraction_multiplier, BIG_U96_LIMBS)
}
