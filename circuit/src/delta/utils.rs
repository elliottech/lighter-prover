// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::big_u16::CircuitBuilderBiguint16;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::byte::split::CircuitBuilderByteSplit;
use crate::delta::evaluate_sequence::CircuitBuilderSequenceEvaluator;
use crate::hints::CircuitBuilderHints;
use crate::types::account_delta::{
    BinaryOptionsDeltaTarget, MarketDataDeltaTarget, PublicPoolInfoDeltaTarget,
};
use crate::types::config::Builder;
use crate::types::constants::ASSET_LIST_SIZE_BITS;

pub fn digest(builder: &mut Builder, target: Target, selector: BoolTarget, degree: &mut Target) {
    static mut INDEX: u64 = 0;
    unsafe {
        // when printing index append zeros so that it has length 3
        builder.sequence_digest_target(0, target, selector);
        let one = builder.one();
        *degree = builder.mul_add(one, selector.target, *degree);
        INDEX += 1;
    }
}

// [is_negative + 32 + 16, 16 + 32]
pub fn pack_collateral(builder: &mut Builder, collateral_delta: &BigIntTarget) -> [Target; 2] {
    let _1_bit_shifter = builder.constant_u64(1 << 1);
    let _16_bit_shifter = builder.constant_u64(1 << 16);
    let _32_bit_shifter = builder.constant_u64(1 << 32);

    let second_limb_splitted = builder.split_u64_to_u16s_le(collateral_delta.abs.limbs[1].0, 2);

    let first_limb_without_sign = builder.mul_add(
        _32_bit_shifter,
        second_limb_splitted[0].0,
        collateral_delta.abs.limbs[0].0,
    );

    let is_negative = builder.is_sign_negative(collateral_delta.sign);

    [
        builder.mul_add(_1_bit_shifter, first_limb_without_sign, is_negative.target),
        builder.mul_add(
            _16_bit_shifter,
            collateral_delta.abs.limbs[2].0,
            second_limb_splitted[1].0,
        ),
    ]
}

/// Packs asset_index and balance_delta into 2 Targets.
///
/// Output layout (LSB to MSB):
///   Target[0]: asset_index (6 bits) | is_negative (1 bit) | abs[0..48] (48 bits)
///   Target[1]: abs[48..96] (48 bits)
///
/// Where abs[0..48] = limbs[0] (32 bits) || lower 16 bits of limbs[1]
///   and abs[48..96] = upper 16 bits of limbs[1] || limbs[2] (32 bits)
pub fn pack_asset_balance(
    builder: &mut Builder,
    asset_index: Target,          // 6 bits
    balance_delta: &BigIntTarget, // Non-extended
) -> [Target; 2] {
    let _1_bit_shifter = builder.constant_u64(1 << 1);
    let _asset_index_bit_shifter = builder.constant_u64(1 << ASSET_LIST_SIZE_BITS);
    let _16_bit_shifter = builder.constant_u64(1 << 16);
    let _32_bit_shifter = builder.constant_u64(1 << 32);

    let second_limb_splitted = builder.split_u64_to_u16s_le(balance_delta.abs.limbs[1].0, 2);

    let first_abs_limb = builder.mul_add(
        _32_bit_shifter,
        second_limb_splitted[0].0,
        balance_delta.abs.limbs[0].0,
    );
    let is_negative = builder.is_sign_negative(balance_delta.sign);
    let first_limb_with_is_negative =
        builder.mul_add(_1_bit_shifter, first_abs_limb, is_negative.target);

    [
        builder.mul_add(
            first_limb_with_is_negative,
            _asset_index_bit_shifter,
            asset_index,
        ),
        builder.mul_add(
            _16_bit_shifter,
            balance_delta.abs.limbs[2].0,
            second_limb_splitted[1].0,
        ),
    ]
}

// [market_index, is_negative_size + abs size]
pub fn pack_binary_options_position(
    builder: &mut Builder,
    market_index: Target,
    entry: &BinaryOptionsDeltaTarget,
) -> [Target; 2] {
    let _1_bit_shifter = builder.constant_u64(1 << 1);

    let is_negative = builder.is_sign_negative(entry.size_delta.sign);
    let abs = builder.biguint_u16_to_target(&entry.size_delta.abs);

    [
        market_index,
        builder.mul_add(_1_bit_shifter, abs, is_negative.target),
    ]
}

// [market_index + last 4 bits of frps, first 60 bits of frps, is_negative_frps + is_negative_pos + abs pos]
pub fn pack_position(
    builder: &mut Builder,
    market_index: Target,
    pos_delta: &MarketDataDeltaTarget,
) -> [Target; 3] {
    let _1_bit_shifter = builder.constant_u64(1 << 1);
    let _8_bit_shifter = builder.constant_u64(1 << 8);
    let _16_bit_shifter = builder.constant_u64(1 << 16);

    let is_negative_pos = builder.is_sign_negative(pos_delta.size_delta.sign);
    let pos_abs_target = builder.biguint_u16_to_target(&pos_delta.size_delta.abs);
    let pos_packed = builder.mul_add(_1_bit_shifter, pos_abs_target, is_negative_pos.target);

    let is_negative_frps = builder.is_sign_negative(pos_delta.funding_rate_prefix_sum_delta.sign);

    let divisor = builder.constant_u64(1 << 12);
    let (limb_4_last_4_bits, limb_4_first_12_bits) = builder.div_rem(
        pos_delta.funding_rate_prefix_sum_delta.abs.limbs[3].0,
        divisor,
        13,
    );

    let mut first_60_bits_of_frps = limb_4_first_12_bits;
    for i in (0..3).rev() {
        first_60_bits_of_frps = builder.mul_add(
            _16_bit_shifter,
            first_60_bits_of_frps,
            pos_delta.funding_rate_prefix_sum_delta.abs.limbs[i].0,
        );
    }

    [
        builder.mul_add(_8_bit_shifter, limb_4_last_4_bits, market_index),
        first_60_bits_of_frps,
        builder.mul_add(_1_bit_shifter, pos_packed, is_negative_frps.target),
    ]
}

// hasL1AddressNum | (hasPublicPoolInfoNum << 1) | (delta.AccountType << 2)
pub fn pack_conditionals_with_account_type(
    builder: &mut Builder,
    l1_address_delta: &BigUintTarget,
    account_type_delta: Target,
    public_pool_info_delta: &PublicPoolInfoDeltaTarget,
) -> (BoolTarget, BoolTarget, Target) {
    let _1_bit_shifter = builder.constant_u64(1 << 1);

    let has_l1_address = {
        let no_l1_address = builder.is_zero_biguint(l1_address_delta);
        builder.not(no_l1_address)
    };
    let has_public_pool_info = {
        let is_public_pool_info_empty = public_pool_info_delta.is_empty(builder);
        builder.not(is_public_pool_info_empty)
    };

    let packed = builder.mul_add(
        _1_bit_shifter,
        account_type_delta,
        has_public_pool_info.target,
    );

    (
        has_l1_address,
        has_public_pool_info,
        builder.mul_add(_1_bit_shifter, packed, has_l1_address.target),
    )
}

/// [32 + 24, 8 + 32 + 16, 16 + 32]
pub fn pack_l1_address(builder: &mut Builder, l1_address_delta: &BigUintTarget) -> [Target; 3] {
    let _8_bit_shifter = builder.constant_u64(1 << 8);
    let _16_bit_shifter = builder.constant_u64(1 << 16);
    let _32_bit_shifter = builder.constant_u64(1 << 32);
    let _40_bit_shifter = builder.constant_u64(1 << 40);

    let _2nd_old_limb_splitted = builder.split_bytes(l1_address_delta.limbs[1].0, 4);
    let _2nd_old_limb_first_part = {
        let mut acc = _2nd_old_limb_splitted[2].0;
        acc = builder.mul_add(_8_bit_shifter, acc, _2nd_old_limb_splitted[1].0);
        builder.mul_add(_8_bit_shifter, acc, _2nd_old_limb_splitted[0].0)
    };
    let _2nd_old_limb_second_part = _2nd_old_limb_splitted[3].0;
    let _4th_old_limb_splitted = builder.split_u64_to_u16s_le(l1_address_delta.limbs[3].0, 2);

    let _1st_limb = builder.mul_add(
        _32_bit_shifter,
        _2nd_old_limb_first_part,
        l1_address_delta.limbs[0].0,
    );

    let mut _2nd_limb = builder.mul_add(
        _8_bit_shifter,
        l1_address_delta.limbs[2].0,
        _2nd_old_limb_second_part,
    );
    _2nd_limb = builder.mul_add(_40_bit_shifter, _4th_old_limb_splitted[0].0, _2nd_limb);

    let _3rd_limb = builder.mul_add(
        _16_bit_shifter,
        l1_address_delta.limbs[4].0,
        _4th_old_limb_splitted[1].0,
    );

    [_1st_limb, _2nd_limb, _3rd_limb]
}
