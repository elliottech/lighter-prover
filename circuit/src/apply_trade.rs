// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use num::BigUint;
use plonky2::field::types::Field;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::big_u16::{CircuitBuilderBigIntU16, CircuitBuilderBiguint16};
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::hints::CircuitBuilderHints;
use crate::liquidation::get_funding_delta_for_position_and_market;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::types::account_position::{AccountPositionTarget, get_position_unrealized_pnl};
use crate::types::asset::AssetTarget;
use crate::types::config::{
    BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U128_LIMBS, BIGU16_U64_LIMBS, Builder, F,
};
use crate::types::constants::*;
use crate::types::market::MarketTarget;
use crate::types::market_details::{MarketDetailsTarget, MarketFlags};
use crate::types::risk_info::{RiskInfoTarget, RiskParametersTarget};
use crate::utils::CircuitBuilderUtils;

pub struct ApplyTradeParams<'a> {
    pub market_details: &'a MarketDetailsTarget,
    pub market: &'a MarketTarget,
    pub is_taker_ask: BoolTarget,
    pub trade_base: Target,
    pub trade_quote: SignedTarget, // For deleverage tx, quote can be non-positive
    pub taker_position: &'a AccountPositionTarget,
    pub maker_position: &'a AccountPositionTarget,
    pub taker_account_type: Target,
    pub maker_account_type: Target,
    pub taker_risk_info: &'a RiskInfoTarget,
    pub maker_risk_info: &'a RiskInfoTarget,
    pub taker_fee: SignedTarget,
    pub maker_fee: SignedTarget,
}

pub struct ApplySpotTradeParams<'a> {
    pub assets: &'a [AssetTarget; 2], // [base, quote]
    pub fee_account_is_taker: BoolTarget,
    pub fee_account_is_maker: BoolTarget,
}

pub fn apply_spot_trade(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    input: &ApplyTradeParams,
    spot_input: &ApplySpotTradeParams,
) -> (
    BigIntTarget, // taker_base_balance_delta
    BigIntTarget, // taker_quote_balance_delta
    BigIntTarget, // maker_base_balance_delta
    BigIntTarget, // maker_quote_balance_delta
    BigIntTarget, // fee_base_balance_delta
    BigIntTarget, // fee_quote_balance_delta
) {
    let zero = builder.zero();

    let is_spot = builder.is_equal_constant(input.market.market_type, MARKET_TYPE_SPOT);
    let is_enabled = builder.and(is_enabled, is_spot);

    let trade_quote_is_negative = builder.is_negative(input.trade_quote);
    builder.conditional_assert_false(is_enabled, trade_quote_is_negative);
    let is_taker_fee_negative = builder.is_negative(input.taker_fee);
    let is_maker_fee_negative = builder.is_negative(input.maker_fee);
    let is_fee_negative = builder.or(is_taker_fee_negative, is_maker_fee_negative);
    builder.conditional_assert_false(is_enabled, is_fee_negative);

    let fee_tick = builder.constant_u64(FEE_TICK);
    let (base_fee_multiplier, _) =
        builder.div_rem(input.market.size_extension_multiplier, fee_tick, FEE_BITS);
    let base_fee_multiplier = builder.target_to_biguint(base_fee_multiplier);
    let (quote_fee_multiplier, _) =
        builder.div_rem(input.market.quote_extension_multiplier, fee_tick, FEE_BITS);
    let quote_fee_multiplier = builder.target_to_biguint(quote_fee_multiplier);

    let trade_base = builder.select(is_enabled, input.trade_base, zero);
    let base_delta = builder.target_to_biguint(trade_base);
    let trade_quote = builder.select(is_enabled, input.trade_quote.target, zero);
    let quote_delta = builder.target_to_biguint(trade_quote);

    let [ext_fee_base_delta, ext_fee_quote_delta] = {
        let taker_fee_non_zero = builder.is_not_zero(input.taker_fee.target);
        let maker_fee_non_zero = builder.is_not_zero(input.maker_fee.target);
        let fees_enabled = builder.or(taker_fee_non_zero, maker_fee_non_zero);
        [
            (
                &base_delta,
                &base_fee_multiplier,
                input.maker_fee,
                input.taker_fee,
            ),
            (
                &quote_delta,
                &quote_fee_multiplier,
                input.taker_fee,
                input.maker_fee,
            ),
        ]
        .map(|(delta, fee_multiplier, ask_fee, bid_fee)| {
            let delta_multiplied =
                builder.mul_biguint_non_carry(delta, fee_multiplier, BIG_U128_LIMBS);
            let fee = builder.select_signed(input.is_taker_ask, ask_fee, bid_fee);
            let fee = SignedTarget::new_unsafe(builder.mul_bool(fees_enabled, fee.target));
            let fee_big = builder.signed_target_to_bigint(fee);
            builder.mul_bigint_with_biguint_non_carry(&fee_big, &delta_multiplied, BIG_U128_LIMBS)
        })
    };

    let size_multiplier_big = builder.target_to_biguint(input.market.size_extension_multiplier);
    let ext_base_delta =
        builder.mul_biguint_non_carry(&base_delta, &size_multiplier_big, BIG_U128_LIMBS);
    let neg_ext_base_delta = builder.negative_biguint(&ext_base_delta);
    let ext_base_delta = builder.biguint_to_bigint(&ext_base_delta);
    let ext_base_delta_with_fee = builder.sub_bigint(&ext_base_delta, &ext_fee_base_delta);

    let quote_multiplier_big = builder.target_to_biguint(input.market.quote_extension_multiplier);
    let ext_quote_delta =
        builder.mul_biguint_non_carry(&quote_delta, &quote_multiplier_big, BIG_U128_LIMBS);
    let neg_ext_quote_delta = builder.negative_biguint(&ext_quote_delta);
    let ext_quote_delta = builder.biguint_to_bigint(&ext_quote_delta);
    let ext_quote_delta_with_fee = builder.sub_bigint(&ext_quote_delta, &ext_fee_quote_delta);

    // Calculate new asset balances
    let [
        taker_base_balance_delta,
        taker_quote_balance_delta,
        maker_base_balance_delta,
        maker_quote_balance_delta,
    ] = [
        (
            &neg_ext_base_delta,
            &ext_base_delta_with_fee,
            &ext_fee_base_delta,
            spot_input.fee_account_is_taker,
        ),
        (
            &ext_quote_delta_with_fee,
            &neg_ext_quote_delta,
            &ext_fee_quote_delta,
            spot_input.fee_account_is_taker,
        ),
        (
            &ext_base_delta_with_fee,
            &neg_ext_base_delta,
            &ext_fee_base_delta,
            spot_input.fee_account_is_maker,
        ),
        (
            &neg_ext_quote_delta,
            &ext_quote_delta_with_fee,
            &ext_fee_quote_delta,
            spot_input.fee_account_is_maker,
        ),
    ]
    .map(
        |(delta_if_ask, delta_if_bid, fee_delta, fee_account_is_taker)| {
            let delta = builder.select_bigint(input.is_taker_ask, delta_if_ask, delta_if_bid);
            let add_to_balance_if_fee_account = BigIntTarget {
                abs: builder.mul_biguint_by_bool(&fee_delta.abs, fee_account_is_taker),
                sign: SignTarget::new_unsafe(
                    builder.mul_bool(fee_account_is_taker, fee_delta.sign.target),
                ),
            };

            builder.add_bigint_non_carry(&delta, &add_to_balance_if_fee_account, BIG_U128_LIMBS)
        },
    );

    let fee_account_is_taker_or_maker = builder.or(
        spot_input.fee_account_is_taker,
        spot_input.fee_account_is_maker,
    );
    let fee_account_is_different = builder.not(fee_account_is_taker_or_maker);

    let [fee_base_balance_delta, fee_quote_balance_delta] =
        [(&ext_fee_base_delta), (&ext_fee_quote_delta)].map(|ext_fee_delta| BigIntTarget {
            abs: builder.mul_biguint_by_bool(&ext_fee_delta.abs, fee_account_is_different),
            sign: SignTarget::new_unsafe(
                builder.mul_bool(fee_account_is_different, ext_fee_delta.sign.target),
            ),
        });

    (
        taker_base_balance_delta,
        taker_quote_balance_delta,
        maker_base_balance_delta,
        maker_quote_balance_delta,
        fee_base_balance_delta,
        fee_quote_balance_delta,
    )
}

pub fn apply_perps_trade(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    input: &ApplyTradeParams,
) -> (
    AccountPositionTarget, // taker position
    AccountPositionTarget, // maker position
    RiskInfoTarget,        // new taker risk info
    RiskInfoTarget,        // new maker risk info
    BigIntTarget,          // fee collateral delta,
    Target,                // new open interest
    BoolTarget,            // taker position sign changed
    BoolTarget,            // maker position sign changed
    BoolTarget,            // is_taker_position_isolated
    BoolTarget,            // is_maker_position_isolated
    BigIntTarget,          // taker margin delta
    BigIntTarget,          // maker margin delta
) {
    let is_perps = builder.is_equal_constant(input.market.market_type, MARKET_TYPE_PERPS);
    let is_enabled = builder.and(is_enabled, is_perps);

    let mut old_taker_position = input.taker_position.clone();
    let mut old_maker_position = input.maker_position.clone();

    let zero_bigint = builder.zero_bigint();
    let one = builder.one();
    let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);

    let market_flags = MarketFlags::from_target(builder, input.market_details.market_flags);
    old_taker_position.init_if_empty(
        builder,
        is_enabled,
        input.taker_account_type,
        market_flags.default_margin_mode,
    );
    old_maker_position.init_if_empty(
        builder,
        is_enabled,
        input.maker_account_type,
        market_flags.default_margin_mode,
    );

    let is_taker_position_isolated =
        builder.is_equal(old_taker_position.margin_mode, isolated_margin_mode);
    let is_taker_position_isolated_and_enabled =
        builder.and(is_taker_position_isolated, is_enabled);
    let is_taker_position_cross = builder.not(is_taker_position_isolated);
    let is_taker_position_cross_and_enabled = builder.and(is_taker_position_cross, is_enabled);

    // Apply taker funding delta
    let taker_funding_delta = get_funding_delta_for_position_and_market(
        builder,
        &old_taker_position,
        input.market_details,
    );

    let taker_funding_cross_delta =
        builder.mul_bigint_by_bool(&taker_funding_delta, is_taker_position_cross_and_enabled);
    // We are always applying funding delta to current risk parameters, either cross or isolated.
    let taker_funding_current_delta = builder.mul_bigint_by_bool(&taker_funding_delta, is_enabled);

    let new_taker_risk_info = RiskInfoTarget {
        cross_risk_parameters: RiskParametersTarget {
            usdc_collateral: builder.add_bigint_non_carry(
                &taker_funding_cross_delta,
                &input.taker_risk_info.cross_risk_parameters.usdc_collateral,
                BIG_U96_LIMBS,
            ),
            ..input.taker_risk_info.cross_risk_parameters.clone()
        },
        current_risk_parameters: RiskParametersTarget {
            usdc_collateral: builder.add_bigint_non_carry(
                &taker_funding_current_delta,
                &input
                    .taker_risk_info
                    .current_risk_parameters
                    .usdc_collateral,
                BIG_U96_LIMBS,
            ),
            ..input.taker_risk_info.current_risk_parameters.clone()
        },
    };

    let is_maker_position_isolated =
        builder.is_equal(old_maker_position.margin_mode, isolated_margin_mode);
    let is_maker_position_isolated_and_enabled =
        builder.and(is_maker_position_isolated, is_enabled);
    let is_maker_position_cross = builder.not(is_maker_position_isolated);
    let is_maker_position_cross_and_enabled = builder.and(is_maker_position_cross, is_enabled);

    let maker_funding_delta = get_funding_delta_for_position_and_market(
        builder,
        &old_maker_position,
        input.market_details,
    );

    let maker_funding_cross_delta =
        builder.mul_bigint_by_bool(&maker_funding_delta, is_maker_position_cross_and_enabled);
    // We are always applying funding delta to current risk parameters, either cross or isolated.
    let maker_funding_current_delta = builder.mul_bigint_by_bool(&maker_funding_delta, is_enabled);

    let new_maker_risk_info = RiskInfoTarget {
        cross_risk_parameters: RiskParametersTarget {
            usdc_collateral: builder.add_bigint_non_carry(
                &maker_funding_cross_delta,
                &input.maker_risk_info.cross_risk_parameters.usdc_collateral,
                BIG_U96_LIMBS,
            ),
            ..input.maker_risk_info.cross_risk_parameters.clone()
        },
        current_risk_parameters: RiskParametersTarget {
            usdc_collateral: builder.add_bigint_non_carry(
                &maker_funding_current_delta,
                &input
                    .maker_risk_info
                    .current_risk_parameters
                    .usdc_collateral,
                BIG_U96_LIMBS,
            ),
            ..input.maker_risk_info.current_risk_parameters.clone()
        },
    };

    // Apply funding to old taker position
    let old_taker_pos_allocated_margin_with_funding = builder.add_bigint_non_carry(
        &old_taker_position.allocated_margin,
        &taker_funding_current_delta,
        BIG_U96_LIMBS,
    );
    old_taker_position.allocated_margin = builder.select_bigint(
        is_taker_position_isolated_and_enabled,
        &old_taker_pos_allocated_margin_with_funding,
        &old_taker_position.allocated_margin,
    );
    old_taker_position.last_funding_rate_prefix_sum = builder.select_bigint_u16(
        is_enabled,
        &input.market_details.funding_rate_prefix_sum,
        &old_taker_position.last_funding_rate_prefix_sum,
    );

    // (-2) * 1 * is_ask + 1 * 1 = 1 - 2*is_ask
    // If is_ask => -1, else 1
    let taker_position_change_sign = builder.arithmetic(
        F::TWO * F::NEG_ONE,
        F::ONE,
        one,
        input.is_taker_ask.target,
        one,
    );
    let taker_position_delta =
        SignedTarget::new_unsafe(builder.mul(taker_position_change_sign, input.trade_base));
    let (mut taker_new_position, taker_realized_usdc_pnl, taker_open_interest_delta) =
        calculate_position_change(
            builder,
            is_enabled,
            &old_taker_position,
            taker_position_delta,
            input.market_details,
            input.trade_quote,
            input.trade_base,
        );

    let usdc_to_collateral_multiplier =
        builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
    let mut taker_collateral_delta = builder.mul_bigint_with_biguint_non_carry(
        &taker_realized_usdc_pnl,
        &usdc_to_collateral_multiplier,
        BIG_U96_LIMBS,
    );

    // Apply funding to old maker position
    let old_maker_pos_allocated_margin_with_funding = builder.add_bigint_non_carry(
        &old_maker_position.allocated_margin,
        &maker_funding_current_delta,
        BIG_U96_LIMBS,
    );
    old_maker_position.allocated_margin = builder.select_bigint(
        is_maker_position_isolated_and_enabled,
        &old_maker_pos_allocated_margin_with_funding,
        &old_maker_position.allocated_margin,
    );
    old_maker_position.last_funding_rate_prefix_sum = builder.select_bigint_u16(
        is_enabled,
        &input.market_details.funding_rate_prefix_sum,
        &old_maker_position.last_funding_rate_prefix_sum,
    );

    let maker_position_delta = builder.neg_signed(taker_position_delta);
    let (mut maker_new_position, maker_realized_usdc_pnl, maker_open_interest_delta) =
        calculate_position_change(
            builder,
            is_enabled,
            &old_maker_position,
            maker_position_delta,
            input.market_details,
            input.trade_quote,
            input.trade_base,
        );
    let mut maker_collateral_delta = builder.mul_bigint_with_biguint_non_carry(
        &maker_realized_usdc_pnl,
        &usdc_to_collateral_multiplier,
        BIG_U96_LIMBS,
    );

    // Apply Fees
    let taker_fee_non_zero = builder.is_not_zero(input.taker_fee.target);
    let maker_fee_non_zero = builder.is_not_zero(input.maker_fee.target);
    let fees_enabled = builder.or(taker_fee_non_zero, maker_fee_non_zero);

    let trade_quote_for_fees =
        SignedTarget::new_unsafe(builder.mul_bool(fees_enabled, input.trade_quote.target));
    let trade_quote_big = builder.signed_target_to_bigint(trade_quote_for_fees);
    let trade_quote_big = builder.select_bigint(is_enabled, &trade_quote_big, &zero_bigint);

    let (abs_taker_fee, sign_taker_fee) = builder.abs(input.taker_fee);
    let taker_fee_big = BigIntTarget {
        abs: BigUintTarget::from_unsafe(abs_taker_fee),
        sign: sign_taker_fee,
    };
    let new_taker_fee = BigIntTarget {
        abs: builder.mul_biguint_non_carry(&trade_quote_big.abs, &taker_fee_big.abs, BIG_U64_LIMBS),
        sign: SignTarget::new_unsafe(
            builder.mul(trade_quote_big.sign.target, taker_fee_big.sign.target),
        ),
    };

    let (abs_maker_fee, sign_maker_fee) = builder.abs(input.maker_fee);
    let maker_fee_big = BigIntTarget {
        abs: BigUintTarget::from_unsafe(abs_maker_fee),
        sign: sign_maker_fee,
    };
    let new_maker_fee = BigIntTarget {
        abs: builder.mul_biguint_non_carry(&trade_quote_big.abs, &maker_fee_big.abs, BIG_U64_LIMBS),
        sign: SignTarget::new_unsafe(
            builder.mul(trade_quote_big.sign.target, maker_fee_big.sign.target),
        ),
    };

    taker_collateral_delta =
        builder.sub_bigint_non_carry(&taker_collateral_delta, &new_taker_fee, BIG_U96_LIMBS);
    maker_collateral_delta =
        builder.sub_bigint_non_carry(&maker_collateral_delta, &new_maker_fee, BIG_U96_LIMBS);
    let fee_collateral_delta =
        builder.add_bigint_non_carry(&new_taker_fee, &new_maker_fee, BIG_U64_LIMBS);
    // Assert that fee collateral delta is non-negative
    builder.assert_bool(BoolTarget::new_unsafe(fee_collateral_delta.sign.target));

    let taker_position_sign = old_taker_position.position.sign.target;
    let taker_new_position_sign = taker_new_position.position.sign.target;
    let neg_taker_position_sign = builder.neg(taker_position_sign);
    let taker_position_side_flipped =
        builder.is_equal(neg_taker_position_sign, taker_new_position_sign);
    let new_taker_allocated_margin = builder.add_bigint_non_carry(
        &taker_collateral_delta,
        &taker_new_position.allocated_margin,
        BIG_U96_LIMBS,
    );
    taker_new_position.allocated_margin = builder.select_bigint(
        is_taker_position_isolated_and_enabled,
        &new_taker_allocated_margin,
        &taker_new_position.allocated_margin,
    );
    let taker_margin_delta = calculate_isolated_margin_change(
        builder,
        input,
        is_taker_position_isolated,
        taker_open_interest_delta,
        &old_taker_position,
        &taker_new_position,
        taker_position_side_flipped,
        &new_taker_fee,
    );
    let new_taker_allocated_margin = builder.add_bigint_non_carry(
        &taker_margin_delta,
        &taker_new_position.allocated_margin,
        BIG_U96_LIMBS,
    );
    taker_new_position.allocated_margin = builder.select_bigint(
        is_taker_position_isolated_and_enabled,
        &new_taker_allocated_margin,
        &taker_new_position.allocated_margin,
    );

    let maker_position_sign = old_maker_position.position.sign.target;
    let maker_new_position_sign = maker_new_position.position.sign.target;
    let neg_maker_position_sign = builder.neg(maker_position_sign);
    let maker_position_side_flipped =
        builder.is_equal(neg_maker_position_sign, maker_new_position_sign);
    let new_maker_allocated_margin = builder.add_bigint_non_carry(
        &maker_collateral_delta,
        &maker_new_position.allocated_margin,
        BIG_U96_LIMBS,
    );
    maker_new_position.allocated_margin = builder.select_bigint(
        is_maker_position_isolated_and_enabled,
        &new_maker_allocated_margin,
        &maker_new_position.allocated_margin,
    );
    let maker_margin_delta = calculate_isolated_margin_change(
        builder,
        input,
        is_maker_position_isolated,
        maker_open_interest_delta,
        &old_maker_position,
        &maker_new_position,
        maker_position_side_flipped,
        &new_maker_fee,
    );
    let new_maker_allocated_margin = builder.add_bigint_non_carry(
        &maker_margin_delta,
        &maker_new_position.allocated_margin,
        BIG_U96_LIMBS,
    );
    maker_new_position.allocated_margin = builder.select_bigint(
        is_maker_position_isolated_and_enabled,
        &new_maker_allocated_margin,
        &maker_new_position.allocated_margin,
    );

    let taker_collateral_delta =
        builder.add_bigint_non_carry(&taker_collateral_delta, &taker_margin_delta, BIG_U96_LIMBS);
    let maker_collateral_delta =
        builder.add_bigint_non_carry(&maker_collateral_delta, &maker_margin_delta, BIG_U96_LIMBS);

    // If isolated, cross delta is -margin_delta, otherwise taker_collateral_delta
    let taker_cross_collateral_delta_for_isolated = BigIntTarget {
        abs: taker_margin_delta.abs.clone(),
        sign: SignTarget::new_unsafe(builder.neg(taker_margin_delta.sign.target)),
    };
    let taker_cross_collateral_delta = builder.select_bigint(
        is_taker_position_isolated_and_enabled,
        &taker_cross_collateral_delta_for_isolated,
        &taker_collateral_delta,
    );
    let new_taker_risk_info = RiskInfoTarget {
        current_risk_parameters: new_taker_risk_info
            .current_risk_parameters
            .update_for_perps_trade(
                builder,
                &taker_collateral_delta,
                &old_taker_position,
                &taker_new_position,
                input.market_details,
                is_enabled,
            ),
        // If cross_risk_parameters and current_risk_parameters are the same, then margin delta will be zero
        cross_risk_parameters: RiskParametersTarget {
            usdc_collateral: builder.add_bigint_non_carry(
                &new_taker_risk_info.cross_risk_parameters.usdc_collateral,
                &taker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            usdc_collateral_with_funding: builder.add_bigint_non_carry(
                &new_taker_risk_info
                    .cross_risk_parameters
                    .usdc_collateral_with_funding,
                &taker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            total_account_value: builder.add_bigint_non_carry(
                &new_taker_risk_info
                    .cross_risk_parameters
                    .total_account_value,
                &taker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            total_account_liquidation_threshold: builder.add_bigint_non_carry(
                &new_taker_risk_info
                    .cross_risk_parameters
                    .total_account_liquidation_threshold,
                &taker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            ..new_taker_risk_info.cross_risk_parameters.clone()
        },
    };

    // If isolated, cross delta is -margin_delta, otherwise taker_collateral_delta
    let maker_cross_collateral_delta_for_isolated = BigIntTarget {
        abs: maker_margin_delta.abs.clone(),
        sign: SignTarget::new_unsafe(builder.neg(maker_margin_delta.sign.target)),
    };
    let maker_cross_collateral_delta = builder.select_bigint(
        is_maker_position_isolated_and_enabled,
        &maker_cross_collateral_delta_for_isolated,
        &maker_collateral_delta,
    );
    let new_maker_risk_info = RiskInfoTarget {
        current_risk_parameters: new_maker_risk_info
            .current_risk_parameters
            .update_for_perps_trade(
                builder,
                &maker_collateral_delta,
                &old_maker_position,
                &maker_new_position,
                input.market_details,
                is_enabled,
            ),
        // If cross_risk_parameters and current_risk_parameters are the same, then margin delta will be zero
        cross_risk_parameters: RiskParametersTarget {
            usdc_collateral: builder.add_bigint_non_carry(
                &new_maker_risk_info.cross_risk_parameters.usdc_collateral,
                &maker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            usdc_collateral_with_funding: builder.add_bigint_non_carry(
                &new_maker_risk_info
                    .cross_risk_parameters
                    .usdc_collateral_with_funding,
                &maker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            total_account_value: builder.add_bigint_non_carry(
                &new_maker_risk_info
                    .cross_risk_parameters
                    .total_account_value,
                &maker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            total_account_liquidation_threshold: builder.add_bigint_non_carry(
                &new_maker_risk_info
                    .cross_risk_parameters
                    .total_account_liquidation_threshold,
                &maker_cross_collateral_delta,
                BIG_U96_LIMBS,
            ),
            ..new_maker_risk_info.cross_risk_parameters.clone()
        },
    };

    let open_interest_delta =
        builder.add_signed(taker_open_interest_delta, maker_open_interest_delta);
    // Result should be non-negative because open interest is sum of the absolute values of the positions and we are potentially only reducing previously added positions
    let new_open_interest = builder.add_signed(
        SignedTarget::new_unsafe(input.market_details.open_interest),
        open_interest_delta,
    );

    let taker_position_sign_changed = builder.is_not_equal(
        old_taker_position.position.sign.target,
        taker_new_position.position.sign.target,
    );
    let maker_position_sign_changed = builder.is_not_equal(
        old_maker_position.position.sign.target,
        maker_new_position.position.sign.target,
    );

    (
        taker_new_position,
        maker_new_position,
        new_taker_risk_info,
        new_maker_risk_info,
        fee_collateral_delta,
        new_open_interest.target,
        taker_position_sign_changed,
        maker_position_sign_changed,
        is_taker_position_isolated,
        is_maker_position_isolated,
        taker_margin_delta,
        maker_margin_delta,
    )
}

pub fn calculate_position_change(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    position: &AccountPositionTarget,
    position_delta: SignedTarget,
    market_details: &MarketDetailsTarget,
    trade_quote: SignedTarget,
    trade_base: Target,
) -> (AccountPositionTarget, BigIntTarget, SignedTarget) {
    let zero = builder.zero();

    let position_delta_bigint =
        builder.signed_target_to_bigint_u16(position_delta, BIGU16_U64_LIMBS);
    let mut new_position = AccountPositionTarget {
        last_funding_rate_prefix_sum: market_details.funding_rate_prefix_sum.clone(),
        position: builder.add_bigint_u16_non_carry(
            &position.position,
            &position_delta_bigint,
            BIGU16_U64_LIMBS,
        ),
        entry_quote: position.entry_quote,
        initial_margin_fraction: position.get_initial_margin_fraction(
            builder,
            market_details.default_initial_margin_fraction,
            market_details.min_initial_margin_fraction,
        ),
        total_order_count: position.total_order_count,
        total_position_tied_order_count: position.total_position_tied_order_count,
        margin_mode: position.margin_mode,
        margin_set_flag: position.margin_set_flag,
        allocated_margin: position.allocated_margin.clone(),
    };

    let abs_old_position = builder.biguint_u16_to_target(&position.position.abs);
    let abs_new_position = builder.biguint_u16_to_target(&new_position.position.abs);

    let is_position_sign_same = builder.is_equal(
        position.position.sign.target,
        new_position.position.sign.target,
    );
    let is_position_sign_changed = builder.not(is_position_sign_same);

    let is_delta_sign_same = builder.is_equal(
        position_delta_bigint.sign.target,
        position.position.sign.target,
    );

    // If the trade quote is positive, the new entry quote equals to the price of the trade times the new position size
    // The trade quote is non-positive for a new position with a different sign;
    // We only allow non-positive quote in deleverage and with the insurance fund as the deleveger,
    // when the the position changes side, to prevent the entry quote from being negative
    // exchange sets the entry quote for the new position as 0.

    let (abs_trade_quote, trade_quote_sign) = builder.abs(trade_quote);
    let trade_quote_is_positive = builder.is_sign_positive(trade_quote_sign);

    let (trade_quote_div_trade_base, _) =
        builder.div_rem(abs_trade_quote, trade_base, ORDER_BASE_AMOUNT_BITS);
    let new_entry_quote_0 = builder.mul(trade_quote_div_trade_base, abs_new_position);
    let new_entry_quote_0 = builder.select(trade_quote_is_positive, new_entry_quote_0, zero);

    // Entry quote + max(tradeQuote, 0)
    let new_entry_quote_1 = builder.mul_add(
        trade_quote_is_positive.target,
        trade_quote.target,
        position.entry_quote,
    );

    let entry_quote_big = builder.target_to_biguint(position.entry_quote);
    let abs_new_position_big = builder.target_to_biguint(abs_new_position);
    let entry_quote_times_new_position =
        builder.mul_biguint_non_carry(&entry_quote_big, &abs_new_position_big, BIG_U128_LIMBS);
    let abs_old_position_big = builder.target_to_biguint(abs_old_position);
    // The quotient must fit a single target; constrain it to two limbs directly.
    let new_entry_quote_2 = builder.div_biguint_trimmed(
        &entry_quote_times_new_position,
        &abs_old_position_big,
        BIG_U64_LIMBS,
    );
    let new_entry_quote_2 = builder.biguint_to_target_safe(&new_entry_quote_2);

    let new_entry_quote_1_2 =
        builder.select(is_delta_sign_same, new_entry_quote_1, new_entry_quote_2);

    let new_entry_quote = builder.select(
        is_position_sign_changed,
        new_entry_quote_0,
        new_entry_quote_1_2,
    );

    new_position.entry_quote = new_entry_quote;

    let open_interest_delta = builder.sub_signed(
        SignedTarget::new_unsafe(abs_new_position),
        SignedTarget::new_unsafe(abs_old_position),
    );

    let new_position_sign_times_entry_quote = BigIntTarget {
        abs: builder.target_to_biguint(new_entry_quote),
        sign: new_position.position.sign,
    };
    let old_position_sign_times_entry_quote = BigIntTarget {
        abs: entry_quote_big,
        sign: position.position.sign,
    };
    let delta_sign_times_trade_quote = BigIntTarget {
        abs: builder.target_to_biguint(abs_trade_quote),
        sign: SignTarget::new_unsafe(
            builder.mul(position_delta_bigint.sign.target, trade_quote_sign.target),
        ),
    };
    let realized_pnl = builder.add_bigint_non_carry(
        &old_position_sign_times_entry_quote,
        &delta_sign_times_trade_quote,
        BIG_U64_LIMBS,
    );
    let realized_pnl = builder.sub_bigint_non_carry(
        &new_position_sign_times_entry_quote,
        &realized_pnl,
        BIG_U64_LIMBS,
    );

    let zero_bigint = builder.zero_bigint();
    let zero_signed = builder.zero_signed();
    (
        AccountPositionTarget::select_position(builder, is_enabled, &new_position, position),
        builder.select_bigint(is_enabled, &realized_pnl, &zero_bigint),
        builder.select_signed(is_enabled, open_interest_delta, zero_signed),
    )
}

pub fn calculate_isolated_margin_change(
    builder: &mut Builder,
    input: &ApplyTradeParams,
    is_enabled: BoolTarget,
    account_open_interest_delta: SignedTarget, // 56 bits
    old_position: &AccountPositionTarget,
    position: &AccountPositionTarget,
    position_side_flipped: BoolTarget,
    fee: &BigIntTarget,
) -> BigIntTarget {
    let neg_one = builder.neg_one();
    let zero = builder.zero();
    let zero_bigint = builder.zero_bigint();
    let zero_biguint = builder.zero_biguint();
    let usdc_to_collateral_multiplier =
        builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));

    let margin_fraction_multiplier = builder.constant_u64(MARGIN_FRACTION_MULTIPLIER as u64);
    let normalized_position_notional_multiplier = builder.mul_many([
        input.market_details.mark_price,       // 32 bits
        input.market_details.quote_multiplier, // 20 bits
        margin_fraction_multiplier,            // 7 bits
    ]);
    let normalized_position_notional_multiplier =
        builder.target_to_biguint(normalized_position_notional_multiplier); // 59 bits

    let position_initial_margin_fraction = position.get_initial_margin_fraction(
        builder,
        input.market_details.default_initial_margin_fraction,
        input.market_details.min_initial_margin_fraction,
    ); // 16 bits
    let position_initial_margin_fraction_big =
        builder.target_to_biguint_single_limb_unsafe(position_initial_margin_fraction);

    // mark_price * quote_multiplier * margin_fraction_multiplier * initial_margin_fraction
    let common_multiplier = builder.mul_biguint_non_carry(
        &normalized_position_notional_multiplier,
        &position_initial_margin_fraction_big,
        BIG_U96_LIMBS,
    );

    let (open_interest_abs, open_interest_sign) = builder.abs(account_open_interest_delta);
    let is_open_interest_decreased = builder.is_sign_negative(open_interest_sign);
    let open_interest_abs_big = builder.target_to_biguint(open_interest_abs);

    let old_position_abs_target = builder.biguint_u16_to_target(&old_position.position.abs);
    let old_position_abs_big = builder.biguint_u16_to_biguint(&old_position.position.abs);

    let position_abs_big = builder.biguint_u16_to_biguint(&position.position.abs);
    let position_abs_target = builder.biguint_u16_to_target(&position.position.abs);

    let open_interest_requirement =
        builder.mul_biguint_non_carry(&open_interest_abs_big, &common_multiplier, BIG_U96_LIMBS);
    let position_requirement =
        builder.mul_biguint_non_carry(&position_abs_big, &common_multiplier, BIG_U96_LIMBS);

    let is_allocated_margin_negative = builder.is_sign_negative(position.allocated_margin.sign);
    // Return -allocated_margin if new position is closed
    let result_if_position_closed = {
        let mut result_if_position_closed_sign =
            builder.mul(position.allocated_margin.sign.target, neg_one);
        result_if_position_closed_sign = builder.select(
            is_allocated_margin_negative,
            zero,
            result_if_position_closed_sign,
        );
        BigIntTarget {
            abs: builder.select_biguint(
                is_allocated_margin_negative,
                &zero_biguint,
                &position.allocated_margin.abs,
            ),
            sign: SignTarget::new_unsafe(result_if_position_closed_sign),
        }
    };

    let old_position_unrealized_usdc_pnl = get_position_unrealized_pnl(
        builder,
        input.market_details,
        old_position_abs_target,
        old_position.position.sign,
        old_position.entry_quote,
    );
    let old_position_unrealized_usdc_pnl_big =
        builder.signed_target_to_bigint(old_position_unrealized_usdc_pnl);
    let old_position_unrealized_pnl_big = builder.mul_bigint_with_biguint_non_carry(
        &old_position_unrealized_usdc_pnl_big,
        &usdc_to_collateral_multiplier,
        BIG_U96_LIMBS,
    );

    let unrealized_usdc_pnl = get_position_unrealized_pnl(
        builder,
        input.market_details,
        position_abs_target,
        position.position.sign,
        position.entry_quote,
    );

    let unrealized_usdc_pnl_big = builder.signed_target_to_bigint(unrealized_usdc_pnl);
    let unrealized_pnl_big = builder.mul_bigint_with_biguint_non_carry(
        &unrealized_usdc_pnl_big,
        &usdc_to_collateral_multiplier,
        BIG_U96_LIMBS,
    );

    // If position side is changed, return position_requirement - (unrealized pnl + allocated_margin)
    let result_if_position_open_and_side_changed = {
        let position_requirement = BigIntTarget {
            abs: position_requirement.clone(),
            // Discarding the case where position_requirement is zero because if it is zero, it will be discarded
            sign: SignTarget::new_unsafe(builder.one()),
        };

        let allocated_margin_plus_unrealized_pnl = builder.add_bigint_non_carry(
            &position.allocated_margin,
            &unrealized_pnl_big,
            BIG_U96_LIMBS,
        );

        builder.sub_bigint_non_carry(
            &position_requirement,
            &allocated_margin_plus_unrealized_pnl,
            BIG_U96_LIMBS,
        )
    };

    // Return open_interest_requirement if position is open and on same side and open interest is not decreased
    let result_if_position_open_and_open_interest_increased = {
        let trade_usdc_pnl =
            builder.sub_signed(unrealized_usdc_pnl, old_position_unrealized_usdc_pnl);

        let trade_usdc_pnl_big = builder.signed_target_to_bigint(trade_usdc_pnl);
        let mut trade_pnl_big = builder.mul_bigint_with_biguint_non_carry(
            &trade_usdc_pnl_big,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );
        trade_pnl_big = builder.sub_bigint_non_carry(&trade_pnl_big, fee, BIG_U96_LIMBS);

        let open_interest_requirement_bigint =
            builder.biguint_to_bigint(&open_interest_requirement);
        let open_interest_requirement_minus_trade_pnl = builder.sub_bigint_non_carry(
            &open_interest_requirement_bigint,
            &trade_pnl_big,
            BIG_U96_LIMBS,
        );
        let is_open_interest_requirement_minus_trade_pnl_negative =
            builder.is_sign_negative(open_interest_requirement_minus_trade_pnl.sign);
        let result_sign = builder.select(
            is_open_interest_requirement_minus_trade_pnl_negative,
            zero,
            open_interest_requirement_minus_trade_pnl.sign.target,
        );
        BigIntTarget {
            abs: builder.select_biguint(
                is_open_interest_requirement_minus_trade_pnl_negative,
                &zero_biguint,
                &open_interest_requirement_minus_trade_pnl.abs,
            ),
            sign: SignTarget::new_unsafe(result_sign),
        }
    };

    let result_if_position_open_and_open_interest_decreased = {
        let old_total_market_value = builder.add_bigint_non_carry(
            &old_position.allocated_margin,
            &old_position_unrealized_pnl_big,
            BIG_U96_LIMBS,
        );
        let new_total_market_value = builder.add_bigint_non_carry(
            &position.allocated_margin,
            &unrealized_pnl_big,
            BIG_U96_LIMBS,
        );

        let old_total_market_value_times_position = BigIntTarget {
            abs: builder.mul_biguint_non_carry(
                &old_total_market_value.abs,
                &position_abs_big,
                BIG_U128_LIMBS,
            ),
            sign: old_total_market_value.sign, // Ignore the case where position is zero, because if it is zero, it will be discarded
        };

        // Ceil divs won't work as expected for negative target_position_value but because we will select position_requirement in that case, it is fine
        let target_position_value = BigIntTarget {
            abs: builder.ceil_div_biguint(
                &old_total_market_value_times_position.abs,
                &old_position_abs_big,
            ),
            sign: old_total_market_value_times_position.sign,
        };
        let target_position_value = builder.trim_bigint(&target_position_value, BIG_U96_LIMBS);

        // target_position_value = MAX(target_position_value, position_requirement). We know that position_requirement is always positive
        let target_position_value = {
            let max_abs = builder.max_biguint(&target_position_value.abs, &position_requirement);
            let is_target_position_value_positive =
                builder.is_sign_positive(target_position_value.sign);

            BigIntTarget {
                abs: builder.select_biguint(
                    is_target_position_value_positive,
                    &max_abs,
                    &position_requirement,
                ),
                sign: SignTarget::new_unsafe(builder.one()), // Assuming position_requirement is positive
            }
        };
        let target_position_value_delta = builder.sub_bigint_non_carry(
            &new_total_market_value,
            &target_position_value,
            BIG_U96_LIMBS,
        );
        // target_position_value_delta = MAX(target_position_value_delta, 0)
        let target_position_value_delta = {
            let is_delta_positive = builder.is_sign_positive(target_position_value_delta.sign);
            let one = builder.one();
            BigIntTarget {
                abs: builder.select_biguint(
                    is_delta_positive,
                    &target_position_value_delta.abs,
                    &zero_biguint,
                ),
                sign: SignTarget::new_unsafe(builder.select(is_delta_positive, one, zero)),
            }
        };
        // allocated_margin_to_move_out = MAX(allocated_margin, 0)
        let allocated_margin_to_move_out = {
            let is_allocated_margin_positive =
                builder.is_sign_positive(position.allocated_margin.sign);
            let zero_biguint = builder.zero_biguint();
            let one = builder.one();
            BigIntTarget {
                abs: builder.select_biguint(
                    is_allocated_margin_positive,
                    &position.allocated_margin.abs,
                    &zero_biguint,
                ),
                sign: SignTarget::new_unsafe(builder.select(
                    is_allocated_margin_positive,
                    one,
                    zero,
                )),
            }
        };

        // neg_result = MIN(target_position_value_delta, allocated_margin_to_move_out). Because both are >= 0, we can select their abs. Sign is 0 if either of them is zero,
        // 1 if both are positive.
        let neg_result = BigIntTarget {
            abs: builder.min_biguint(
                &target_position_value_delta.abs,
                &allocated_margin_to_move_out.abs,
            ),
            sign: SignTarget::new_unsafe(builder.mul(
                target_position_value_delta.sign.target,
                allocated_margin_to_move_out.sign.target,
            )),
        };

        // Negate the result
        BigIntTarget {
            abs: neg_result.abs,
            sign: SignTarget::new_unsafe(builder.mul(neg_result.sign.target, neg_one)),
        }
    };

    let result_if_position_open_and_side_unchanged = builder.select_bigint(
        is_open_interest_decreased,
        &result_if_position_open_and_open_interest_decreased,
        &result_if_position_open_and_open_interest_increased,
    );

    let result_if_position_open = builder.select_bigint(
        position_side_flipped,
        &result_if_position_open_and_side_changed,
        &result_if_position_open_and_side_unchanged,
    );

    let is_position_closed = builder.is_zero(position.position.sign.target);

    let result = builder.select_bigint(
        is_position_closed,
        &result_if_position_closed,
        &result_if_position_open,
    );

    builder.select_bigint(is_enabled, &result, &zero_bigint)
}
