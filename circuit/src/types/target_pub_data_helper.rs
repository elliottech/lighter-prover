// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::Target;

use super::config::Builder;
use super::constants::{
    ACCOUNT_INDEX_BITS, MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX,
    ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE,
};
use crate::bigint::biguint::BigUintTarget;
use crate::byte::split::CircuitBuilderByteSplit;
use crate::types::config::BIG_U64_LIMBS;
use crate::types::constants::{MAX_EXCHANGE_USDC_BITS, MAX_TRANSFER_BITS};
use crate::uint::u8::{CircuitBuilderU8, U8Target};

pub fn add_pub_data_type_target(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    pub_data_type: u8,
) -> usize {
    bytes.push(builder.constant_u8(pub_data_type));
    1
}

pub fn add_u8_target(_builder: &mut Builder, bytes: &mut Vec<U8Target>, target: U8Target) -> usize {
    bytes.push(target);
    1
}

#[track_caller]
pub fn add_account_index_target(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    target: Target,
) -> usize {
    let bit_size = ACCOUNT_INDEX_BITS;
    let byte_size = bit_size / 8;

    let mut limb_bits = builder.split_bytes(target, byte_size);
    limb_bits.reverse();
    bytes.extend_from_slice(&limb_bits);

    byte_size
}

pub fn add_pub_key_target(bytes: &mut Vec<U8Target>, pub_key: &[U8Target]) -> usize {
    bytes.extend_from_slice(pub_key);
    pub_key.len()
}

pub fn add_byte_target_unsafe(bytes: &mut Vec<U8Target>, target: Target) -> usize {
    bytes.push(U8Target(target));
    1
}

#[track_caller]
pub fn add_target(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    target: Target,
    bit_size: usize,
) -> usize {
    let byte_size = bit_size / 8;
    assert_eq!(byte_size * 8, bit_size, "bit size should be multiple of 8");

    let mut limb_bytes = builder.split_bytes(target, byte_size);
    limb_bytes.reverse();
    bytes.extend_from_slice(&limb_bytes);

    byte_size
}

/// Same as [`add_target`] but if "target" is known to be smaller than public data size
#[track_caller]
pub fn add_target_extend(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    target: Target,
    bit_size: usize,
    real_bit_size: usize,
) -> usize {
    assert!(
        real_bit_size < bit_size,
        "real size should be smaller than size"
    );

    let byte_size = bit_size / 8;
    assert_eq!(byte_size * 8, bit_size, "bit size should be multiple of 8");

    let real_byte_size = real_bit_size.div_ceil(8);
    let mut limb_bytes = builder.split_bytes(target, real_byte_size);
    limb_bytes.resize_with(byte_size, || builder.zero_u8());
    limb_bytes.reverse();
    bytes.extend_from_slice(&limb_bytes);

    byte_size
}

#[track_caller]
pub fn add_big_uint_target(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    target: &BigUintTarget,
) -> usize {
    for i in (0..target.num_limbs()).rev() {
        add_target(builder, bytes, target.limbs[i].0, 32);
    }
    target.bit_len() / 8
}

/// Similar to add_big_uint_target but we know a strict range-check. Use for transfers/withdraws
#[track_caller]
pub fn add_transfer_usdc_target(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    target: &BigUintTarget,
) -> usize {
    assert_eq!(
        target.num_limbs(),
        BIG_U64_LIMBS,
        "usdc target should have exactly 2 limb"
    );

    add_target_extend(
        builder,
        bytes,
        target.limbs[1].0,
        32,
        MAX_TRANSFER_BITS % 32,
    );
    add_target(builder, bytes, target.limbs[0].0, 32);

    target.bit_len() / 8
}

/// Similar to add_big_uint_target but we know a strict range-check. Use for deposit
#[track_caller]
pub fn add_deposit_usdc_target(
    builder: &mut Builder,
    bytes: &mut Vec<U8Target>,
    target: &BigUintTarget,
) -> usize {
    assert_eq!(
        target.num_limbs(),
        BIG_U64_LIMBS,
        "usdc target should have exactly 2 limb"
    );

    add_target_extend(
        builder,
        bytes,
        target.limbs[1].0,
        32,
        MAX_EXCHANGE_USDC_BITS % 32,
    );
    add_target(builder, bytes, target.limbs[0].0, 32);

    target.bit_len() / 8
}

pub fn pad_priority_op_pub_data_target(
    builder: &mut Builder,
    bytes: &mut [U8Target],
    expected_len: usize,
) -> [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX] {
    assert_eq!(bytes.len(), expected_len, "Invalid bytes size");

    let mut padded_bytes = bytes.to_vec();
    padded_bytes.resize_with(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX, || {
        builder.zero_u8()
    });
    padded_bytes.try_into().unwrap()
}

pub fn pad_on_chain_pub_data_target(
    builder: &mut Builder,
    bytes: &mut [U8Target],
) -> [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE] {
    let mut padded_bytes = bytes.to_vec();
    padded_bytes.resize_with(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE, || {
        builder.zero_u8()
    });
    padded_bytes.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use num::{BigUint, Num};
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::builder::Builder;
    use crate::types::config::{C, CIRCUIT_CONFIG, D, F};
    use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

    #[allow(clippy::just_underscores_and_digits)]
    #[test]
    fn test_adding_target_bytes() -> Result<()> {
        // let _ = env_logger::try_init_from_env(env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"));

        let mut builder = Builder::<F, D>::new(CIRCUIT_CONFIG);
        let bytes = &mut Vec::<U8Target>::new();
        let l1_ad_val = BigUint::from_str_radix("12345", 10).unwrap();
        let _one = builder.one();
        let _zero = builder.zero();

        let mut limb_values = l1_ad_val.to_u32_digits();
        limb_values.resize(5, 0);
        let limbs = limb_values
            .iter()
            .map(|&l| builder.constant_u32(l))
            .collect();

        let l1_ad = BigUintTarget { limbs };

        add_big_uint_target(&mut builder, bytes, &l1_ad);
        // 12345 corresponds to: [0, .., 0, 48, 57]
        let _48 = builder.constant_u8(48);
        let _57 = builder.constant_u8(57);
        builder.connect_u8(bytes[18], _48);
        builder.connect_u8(bytes[19], _57);

        let _11 = builder.constant_u8(11);
        add_target(&mut builder, bytes, _11.0, 32);
        builder.connect_u8(bytes[23], _11);

        add_pub_data_type_target(&mut builder, bytes, 2);

        let _2 = builder.constant_u8(2);
        builder.connect_u8(bytes[24], _2);

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::new()).unwrap())
    }
}
