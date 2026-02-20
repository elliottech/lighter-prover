// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use num::{BigUint, FromPrimitive};
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::big_u16::CircuitBuilderBigIntU16;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::types::account::AccountTarget;
use crate::types::account_asset::AccountAssetTarget;
use crate::types::account_position::AccountPositionTarget;
use crate::types::config::{
    BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U128_LIMBS, BIGU16_U64_LIMBS, Builder,
};
use crate::types::constants::*;
use crate::types::market_details::MarketDetailsTarget;
use crate::types::risk_info::RiskParametersTarget;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

pub fn get_funding_delta_for_position_and_market(
    builder: &mut Builder,
    position: &AccountPositionTarget,
    market_details: &MarketDetailsTarget,
) -> BigIntTarget {
    let quote_multiplier_big =
        builder.target_to_biguint_single_limb_unsafe(market_details.quote_multiplier);

    let position_big_u32 = builder.bigint_u16_to_bigint(&position.position);
    let funding_multiplier = builder.mul_bigint_with_biguint_non_carry(
        &position_big_u32,
        &quote_multiplier_big,
        BIG_U96_LIMBS,
    );
    let funding_rate = builder.sub_bigint_u16_non_carry(
        &position.last_funding_rate_prefix_sum,
        &market_details.funding_rate_prefix_sum,
        BIGU16_U64_LIMBS,
    );
    let funding_rate = builder.bigint_u16_to_bigint(&funding_rate);

    BigIntTarget {
        abs: builder.mul_biguint_non_carry(
            &funding_multiplier.abs,
            &funding_rate.abs,
            BIG_U96_LIMBS,
        ),
        sign: SignTarget::new_unsafe(
            builder.mul(funding_multiplier.sign.target, funding_rate.sign.target),
        ),
    }
}

#[allow(non_snake_case)]
pub fn get_position_zero_price(
    builder: &mut Builder,
    position: &AccountPositionTarget,
    market_details: &MarketDetailsTarget,
    risk_info: &RiskParametersTarget,
) -> BigUintTarget {
    let one = builder.one();
    let one_big = builder.one_biguint();
    let margin_fraction_tick = builder.constant_biguint(&BigUint::from(MARGIN_TICK));

    let margin_requirement_times_tick = builder.mul_biguint_non_carry(
        &risk_info.maintenance_margin_requirement,
        &margin_fraction_tick,
        BIG_U128_LIMBS,
    );

    let mark_price = builder.target_to_biguint_single_limb_unsafe(market_details.mark_price); // safe because data put here is always range-checked in pre-exec circuits
    let A =
        builder.mul_biguint_non_carry(&mark_price, &margin_requirement_times_tick, BIG_U128_LIMBS);
    let A = builder.biguint_to_bigint(&A);

    let mark_price_times_margin_fraction = builder.mul(
        market_details.mark_price,                  // 32 bits
        market_details.maintenance_margin_fraction, // 16 bits
    );
    let mark_price_times_margin_fraction =
        builder.target_to_biguint(mark_price_times_margin_fraction);

    let B = builder.mul_biguint_non_carry(
        &mark_price_times_margin_fraction,
        &risk_info.total_account_value.abs,
        BIG_U128_LIMBS,
    );
    let B = BigIntTarget {
        abs: B,
        sign: SignTarget::new_unsafe(builder.mul_many([
            risk_info.total_account_value.sign.target,
            position.position.sign.target,
        ])),
    };

    let dividend = builder.sub_bigint_non_carry(&A, &B, BIG_U128_LIMBS);
    let divisor = margin_requirement_times_tick;

    let result_if_positive_position = BigIntTarget {
        abs: builder.ceil_div_biguint(&dividend.abs, &divisor),
        sign: dividend.sign,
    };
    let result_if_negative_position = BigIntTarget {
        abs: builder.div_biguint(&dividend.abs, &divisor),
        sign: dividend.sign,
    };

    let position_is_positive = builder.is_sign_positive(position.position.sign);
    let zero_price_big = builder.select_bigint(
        position_is_positive,
        &result_if_positive_position,
        &result_if_negative_position,
    );

    let is_zero_price_sign_positive = builder.is_equal(zero_price_big.sign.target, one);
    let is_zero_price_sign_not_positive = builder.not(is_zero_price_sign_positive);

    let max_order_price_big =
        builder.constant_biguint(&BigUint::from_u64(MAX_ORDER_PRICE).unwrap());
    let should_trim = builder.is_gt_biguint(&zero_price_big.abs, &max_order_price_big);
    let is_sign_positive_and_should_trim = builder.and(is_zero_price_sign_positive, should_trim);

    let zero_price = builder.select_biguint(
        is_zero_price_sign_not_positive,
        &one_big,
        &zero_price_big.abs,
    );

    builder.select_biguint(
        is_sign_positive_and_should_trim,
        &max_order_price_big,
        &zero_price,
    )
}

#[allow(non_snake_case)]
pub fn get_position_zero_quote(
    builder: &mut Builder,
    position: &AccountPositionTarget,
    market_details: &MarketDetailsTarget,
    risk_info: &RiskParametersTarget,
    trade_size: Target,
) -> BigIntTarget {
    let zero = builder.zero();

    let margin_fraction_tick = BigUintTarget::from(builder.constant_u32(MARGIN_TICK));
    let mark_price = builder.target_to_biguint_single_limb_unsafe(market_details.mark_price);
    let quote_multiplier =
        builder.target_to_biguint_single_limb_unsafe(market_details.quote_multiplier);

    let trade_size = builder.target_to_biguint(trade_size);
    let notional_value = builder.mul_many_biguint_non_carry(
        &[&mark_price, &quote_multiplier, &trade_size],
        BIG_U64_LIMBS,
    );

    let position_sign_is_positive = builder.is_sign_positive(position.position.sign);

    let margin_requirement_times_tick = builder.mul_biguint(
        &risk_info.maintenance_margin_requirement,
        &margin_fraction_tick,
    );

    let A = builder.mul_biguint(&notional_value, &margin_requirement_times_tick);
    let A = builder.biguint_to_bigint(&A);

    let maintenance_margin_fraction =
        builder.target_to_biguint_single_limb_unsafe(market_details.maintenance_margin_fraction);

    let notional_times_margin_fraction =
        builder.mul_biguint(&notional_value, &maintenance_margin_fraction);
    let B = builder.mul_biguint(
        &notional_times_margin_fraction,
        &risk_info.total_account_value.abs,
    );
    let B = BigIntTarget {
        abs: B,
        sign: SignTarget::new_unsafe(builder.mul_many([
            risk_info.total_account_value.sign.target,
            position.position.sign.target,
        ])),
    };

    let dividend = builder.sub_bigint(&A, &B);
    let divisor = margin_requirement_times_tick;

    let result_if_ceil = builder.ceil_div_biguint(&dividend.abs, &divisor);
    let result_if_floor = builder.div_biguint(&dividend.abs, &divisor);

    let is_dividend_positive = builder.is_sign_positive(dividend.sign);
    let selector = builder.is_equal(
        is_dividend_positive.target,
        position_sign_is_positive.target,
    );

    let result_abs = builder.select_biguint(selector, &result_if_ceil, &result_if_floor);
    let is_result_abs_zero = builder.is_zero_biguint(&result_abs);
    let result_sign =
        SignTarget::new_unsafe(builder.select(is_result_abs_zero, zero, dividend.sign.target));

    BigIntTarget {
        abs: result_abs,
        sign: result_sign,
    }
}

// Returns the balance of the asset in context of product type, which is constant at circuit generation time.
pub fn get_asset_balance_const(
    builder: &mut Builder,
    product_type: u64,
    account: &AccountTarget,
    account_asset: &AccountAssetTarget,
    is_asset_used_as_margin: BoolTarget,
) -> BigIntTarget {
    if product_type == PRODUCT_TYPE_PERPS {
        account.collateral.clone()
    } else {
        let is_account_isolated = builder.is_equal_constant(
            account.account_trading_mode,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE as u64,
        );
        let is_account_unified = builder.not(is_account_isolated);

        let is_unified_and_margin = builder.and(is_account_unified, is_asset_used_as_margin);

        let asset_balance = builder.biguint_to_bigint(&account_asset.balance);

        builder.select_bigint(is_unified_and_margin, &account.collateral, &asset_balance)
    }
}

// Returns the available balance of the asset in context of product type, which is constant at circuit generation time.
pub fn get_available_asset_balance_const(
    builder: &mut Builder,
    product_type: u64,
    account: &AccountTarget,
    account_asset: &AccountAssetTarget,
    is_asset_used_as_margin: BoolTarget,
    risk_info: &RiskParametersTarget,
) -> BigUintTarget {
    let zero_big = builder.zero_biguint();

    let is_account_isolated = builder.is_equal_constant(
        account.account_trading_mode,
        ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE as u64,
    );
    let is_account_unified = builder.not(is_account_isolated);

    let is_unified_and_not_margin = builder.and_not(is_account_unified, is_asset_used_as_margin);

    let available_cross_collateral = get_available_collateral(builder, risk_info);

    let return_available_asset_balance = builder.or(is_account_isolated, is_unified_and_not_margin);

    let (available_cross_collateral_minus_locked, borrow) =
        builder.try_sub_biguint(&available_cross_collateral, &account_asset.locked_balance);
    let available_cross_collateral_minus_locked = builder.select_biguint(
        BoolTarget::new_unsafe(borrow.0),
        &zero_big,
        &available_cross_collateral_minus_locked,
    );

    let available_cross_collateral = builder.select_biguint(
        is_account_unified,
        &available_cross_collateral_minus_locked,
        &available_cross_collateral,
    );

    if product_type == PRODUCT_TYPE_PERPS {
        available_cross_collateral
    } else {
        let available_asset_balance = account_asset.get_available_balance(builder);

        builder.select_biguint(
            return_available_asset_balance,
            &available_asset_balance,
            &available_cross_collateral,
        )
    }
}

// Returns the available balance of the asset in context of product type
pub fn get_available_asset_balance(
    builder: &mut Builder,
    product_type: Target,
    account: &AccountTarget,
    account_asset: &AccountAssetTarget,
    is_asset_used_as_margin: BoolTarget,
    risk_info: &RiskParametersTarget,
) -> BigUintTarget {
    let zero_big = builder.zero_biguint();

    let is_product_spot = BoolTarget::new_unsafe(product_type);
    let is_account_unified = account.is_unified_mode();
    let is_not_margin = builder.not(is_asset_used_as_margin);

    // spot = (unified && 'margin) || ('unified && spot)
    let return_available_asset_balance =
        builder.select_bool(is_account_unified, is_not_margin, is_product_spot);

    let available_asset_balance = account_asset.get_available_balance(builder);
    let available_cross_collateral = {
        let available_cross_collateral = get_available_collateral(builder, risk_info);
        let locked_balance_delta =
            builder.select_biguint(is_account_unified, &account_asset.locked_balance, &zero_big);
        let (result, borrow) =
            builder.try_sub_biguint(&available_cross_collateral, &locked_balance_delta);
        builder.select_biguint(BoolTarget::new_unsafe(borrow.0), &zero_big, &result)
    };

    builder.select_biguint(
        return_available_asset_balance,
        &available_asset_balance,
        &available_cross_collateral,
    )
}

pub fn get_available_collateral(
    builder: &mut Builder,
    risk_info: &RiskParametersTarget,
) -> BigUintTarget {
    let neg_one = builder.neg_one();

    let is_healthy = risk_info.is_healthy(builder);
    let (mut available_collateral, borrow) = builder.try_sub_biguint(
        &risk_info.total_account_value.abs,
        &risk_info.initial_margin_requirement,
    );
    builder.conditional_assert_zero_u32(is_healthy, borrow);

    available_collateral = builder.mul_biguint_by_bool(&available_collateral, is_healthy);

    let collateral_with_funding = risk_info.collateral_with_funding.clone();
    let is_collateral_with_funding_non_negative =
        builder.is_not_equal(collateral_with_funding.sign.target, neg_one);

    available_collateral = builder.mul_biguint_by_bool(
        &available_collateral,
        is_collateral_with_funding_non_negative,
    );

    // If collateral_with_funding is negative, then available_collateral is zero. So minimum is zero
    builder.min_biguint(&available_collateral, &collateral_with_funding.abs)
}

pub fn get_shares_asset_value_for_staking_pool(
    builder: &mut Builder,
    total_shares: Target,
    asset_balance: &BigUintTarget,
    asset_extension_multiplier: &BigUintTarget,
    share_amount: Target,
) -> BigUintTarget {
    let is_total_shares_zero = builder.is_zero(total_shares);

    let big_share_amount = builder.target_to_biguint(share_amount);
    let big_initial_pool_share_value =
        builder.constant_biguint(&BigUint::from(INITIAL_POOL_SHARE_VALUE));
    let default_usdc_value = builder.mul_biguint(&big_share_amount, &big_initial_pool_share_value);

    let share_amount_mul_total_account_value =
        builder.mul_biguint(&big_share_amount, asset_balance);
    let big_old_total_shares = builder.target_to_biguint(total_shares);
    let old_total_shares_mul_usdc_to_collateral_multiplier =
        builder.mul_biguint(&big_old_total_shares, asset_extension_multiplier);
    let c_big_usdc_to_mint_shares = builder.div_biguint(
        &share_amount_mul_total_account_value,
        &old_total_shares_mul_usdc_to_collateral_multiplier,
    );

    builder.select_biguint(
        is_total_shares_zero,
        &default_usdc_value,
        &c_big_usdc_to_mint_shares,
    )
}

pub fn get_shares_usdc_value_for_public_pool(
    builder: &mut Builder,
    risk_info: &RiskParametersTarget,
    account: &AccountTarget,
    share_amount: Target,
) -> Target {
    let is_total_shares_zero = builder.is_zero(account.public_pool_info.total_shares);

    let big_share_amount = builder.target_to_biguint(share_amount);
    let big_initial_pool_share_value =
        builder.constant_biguint(&BigUint::from(INITIAL_POOL_SHARE_VALUE));
    let default_usdc_value = builder.mul_biguint(&big_share_amount, &big_initial_pool_share_value);

    let share_amount_mul_total_account_value =
        builder.mul_biguint(&big_share_amount, &risk_info.total_account_value.abs);
    let big_old_total_shares = builder.target_to_biguint(account.public_pool_info.total_shares);
    let usdc_to_collateral_multiplier =
        builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
    let old_total_shares_mul_usdc_to_collateral_multiplier =
        builder.mul_biguint(&big_old_total_shares, &usdc_to_collateral_multiplier);
    let c_big_usdc_to_mint_shares = builder.div_biguint(
        &share_amount_mul_total_account_value,
        &old_total_shares_mul_usdc_to_collateral_multiplier,
    );

    let big_usdc_to_mint_shares = builder.select_biguint(
        is_total_shares_zero,
        &default_usdc_value,
        &c_big_usdc_to_mint_shares,
    );

    builder.biguint_to_target_safe(&big_usdc_to_mint_shares)
}

// Ensure total account value is positive before calling this function
pub fn get_available_shares_to_burn_for_public_pool(
    builder: &mut Builder,
    risk_info: &RiskParametersTarget,
    pool_account: &AccountTarget,
) -> Target {
    let available_collateral = get_available_collateral(builder, risk_info);
    let big_total_shares = builder.target_to_biguint(pool_account.public_pool_info.total_shares);
    let available_collateral_mul_total_shares =
        builder.mul_biguint(&available_collateral, &big_total_shares);
    let big_available_shares = builder.div_biguint(
        &available_collateral_mul_total_shares,
        &risk_info.total_account_value.abs,
    ); // since total account value is always bigger than the available collateral, result should be <= total shares
    builder.biguint_to_target_unsafe(&big_available_shares)
}

pub fn get_available_shares_to_burn_for_staking_pool(
    builder: &mut Builder,
    total_shares: Target,
    pool_asset_info: &AccountAssetTarget,
) -> Target {
    let available_asset_balance = pool_asset_info.get_available_balance(builder);
    let big_total_shares = builder.target_to_biguint(total_shares);
    let available_balance_mul_total_shares =
        builder.mul_biguint(&available_asset_balance, &big_total_shares);
    let big_available_shares = builder.div_biguint(
        &available_balance_mul_total_shares,
        &pool_asset_info.balance,
    );
    builder.biguint_to_target_unsafe(&big_available_shares)
}
