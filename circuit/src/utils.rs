// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::hints::CircuitBuilderHints;
use crate::types::config::{BIG_U64_LIMBS, D, F};
use crate::types::constants::{HOUR_IN_MS, MINUTE_IN_MS, SECOND_IN_MS};
use crate::uint::u8::U8Target;

/// Whether intermediate proofs should be eagerly verified right after proving.
/// Always on in debug builds; enabled in release builds with
/// `CIRCUIT_EAGER_VERIFY=true` for better error localization.
pub fn eager_verify_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        cfg!(debug_assertions)
            || std::env::var("CIRCUIT_EAGER_VERIFY")
                .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

pub fn round_unix_timestamp_to_previous_second(
    builder: &mut Builder<F, D>,
    timestamp: Target,
) -> Target {
    // Remove the remainder of the timestamp when divided by 1 000
    let second_in_ms = builder.constant(F::from_canonical_usize(SECOND_IN_MS));
    let (_, remainder) = builder.div_rem(timestamp, second_in_ms, 10);
    builder.sub(timestamp, remainder)
}

pub fn round_unix_timestamp_to_previous_minute(
    builder: &mut Builder<F, D>,
    timestamp: Target,
) -> Target {
    // Remove the remainder of the timestamp when divided by 60 000
    let minute_in_ms = builder.constant(F::from_canonical_usize(MINUTE_IN_MS));
    let (_, remainder) = builder.div_rem(timestamp, minute_in_ms, 16);
    builder.sub(timestamp, remainder)
}

pub fn round_unix_timestamp_to_previous_hour(
    builder: &mut Builder<F, D>,
    timestamp: Target,
) -> Target {
    // Remove the remainder of the timestamp when divided by 3 600 000
    let hour_in_ms = builder.constant(F::from_canonical_usize(HOUR_IN_MS));
    let (_, remainder) = builder.div_rem(timestamp, hour_in_ms, 22);
    builder.sub(timestamp, remainder)
}

pub fn round_unix_timestamp_to_next_hour(builder: &mut Builder<F, D>, timestamp: Target) -> Target {
    // Add the remainder of the timestamp when divided by 3 600 000
    let hour_in_ms = builder.constant(F::from_canonical_usize(HOUR_IN_MS));
    let (_, remainder) = builder.div_rem(timestamp, hour_in_ms, 22);
    let addend = builder.sub(hour_in_ms, remainder);
    builder.add(timestamp, addend)
}

pub trait CircuitBuilderUtils<F: RichField + Extendable<D>, const D: usize> {
    fn is_zero(&mut self, a: Target) -> BoolTarget;
    fn is_not_zero(&mut self, a: Target) -> BoolTarget;

    fn conditional_assert_zero(&mut self, is_enabled: BoolTarget, a: Target);
    fn conditional_assert_not_zero(&mut self, is_enabled: BoolTarget, a: Target);
    fn conditional_assert_one(&mut self, is_enabled: BoolTarget, a: Target);

    fn conditional_assert_not_eq(&mut self, is_enabled: BoolTarget, a: Target, b: Target);
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderUtils<F, D> for Builder<F, D> {
    fn is_zero(&mut self, a: Target) -> BoolTarget {
        let zero = self.zero();
        self.is_equal(a, zero)
    }

    fn is_not_zero(&mut self, a: Target) -> BoolTarget {
        let is_zero = self.is_zero(a);
        self.not(is_zero) // This `not` is free
    }

    fn conditional_assert_zero(&mut self, is_enabled: BoolTarget, a: Target) {
        // If is_enabled is false(0), result is always zero and there will be no condition on `a`
        // Otherwise, `a` has to be 0
        let is_enabled_times_a = self.mul_bool(is_enabled, a);
        self.assert_zero(is_enabled_times_a);
    }

    fn conditional_assert_one(&mut self, is_enabled: BoolTarget, a: Target) {
        let one = self.one();
        let a_minus_one = self.sub(a, one);
        self.conditional_assert_zero(is_enabled, a_minus_one);
    }

    fn conditional_assert_not_eq(&mut self, is_enabled: BoolTarget, a: Target, b: Target) {
        let eq = self.is_equal(a, b);
        let res = self.and(is_enabled, eq);
        self.assert_false(res);
    }

    fn conditional_assert_not_zero(&mut self, is_enabled: BoolTarget, a: Target) {
        let is_zero = self.is_zero(a);
        let res = self.and(is_enabled, is_zero);
        self.assert_false(res);
    }
}

pub const fn ceil_div_usize(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

pub fn to_bits_i64(x: i64) -> [u8; 64] {
    let mut res = [0; 64];

    for i in 0..64 {
        res[i] = ((x >> i) & 1) as u8;
    }

    for byte in 0..8 {
        let curr_byte = &mut res[8 * byte..8 * byte + 8];
        for i in 0..4 {
            curr_byte.swap(i, 7 - i);
        }
    }

    res
}

pub fn split_le_u64(x: u64) -> [u8; 64] {
    let mut res = [0; 64];

    for i in 0..64 {
        res[i] = ((x >> i) & 1) as u8;
    }

    res
}

pub fn le_sum_u64(bits: &[u8]) -> u64 {
    bits.iter()
        .enumerate()
        .fold(0, |acc, (i, &bit)| acc + ((bit as u64) << i))
}

pub fn to_bits_u32(x: u32) -> [u32; 32] {
    let mut res = [0; 32];

    for i in 0..32 {
        res[i] = (x >> i) & 1;
    }

    res
}

pub fn to_bits_u8(x: u8) -> [u8; 8] {
    let mut res = [0; 8];

    for i in 0..8 {
        res[i] = (x >> i) & 1;
    }

    res
}

pub fn le_sum(bits: &[u8]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0, |acc, (i, &bit)| acc + ((bit as u32) << i))
}

/// Convert the limbs of u32 integers in the field `F` to a BigUint in big-endian format
pub fn field_elements_to_be_biguint<F: RichField>(elements: &[F]) -> BigUint {
    BigUint::from_slice(
        &elements
            .iter()
            .map(|x| x.to_canonical_u64() as u32)
            .map(|x| x.to_be())
            .collect::<Vec<_>>(),
    )
}

pub fn u64_into_u32_array_in_field(x: u64) -> [F; BIG_U64_LIMBS] {
    biguint_into_u32_array_in_field::<BIG_U64_LIMBS>(&BigUint::from(x))
}

pub fn biguint_into_u32_array_in_field<const LIMBS: usize>(x: &BigUint) -> [F; LIMBS] {
    let mut limbs = x
        .to_u32_digits()
        .into_iter()
        .map(F::from_canonical_u32)
        .collect::<Vec<F>>();
    limbs.resize(LIMBS, F::ZERO);
    limbs.try_into().unwrap()
}

pub fn u8_to_bits(num: u8) -> Vec<bool> {
    let mut result = Vec::with_capacity(8);
    let mut n = num;
    for _ in 0..8 {
        result.push(n & 1 == 1);
        n >>= 1;
    }
    result
}

pub fn hex_str_to_bits(input: &str) -> Result<Vec<bool>> {
    let input_u8 = hex::decode(input)?;
    let input_bits = input_u8
        .iter()
        .flat_map(|x| u8_to_bits(*x))
        .collect::<Vec<_>>();
    Ok(input_bits)
}

pub fn split_le_base16<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    target: Target,
    num_limbs: usize,
) -> Vec<Target> {
    let four = builder.constant_u64(4);
    builder
        .split_le_base::<4>(target, num_limbs)
        .chunks(2)
        .map(|four_bits| builder.mul_add(four, four_bits[1], four_bits[0]))
        .collect::<Vec<_>>()
}

pub fn bytes_to_hex<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    bytes: &[Target],
) -> Vec<U8Target> {
    // 0-9 (0000-1001) --> 48-57
    // a-f (1010-1111) --> 97-102
    let arr = vec![
        builder.constant(F::from_canonical_u8(48)),
        builder.constant(F::from_canonical_u8(49)),
        builder.constant(F::from_canonical_u8(50)),
        builder.constant(F::from_canonical_u8(51)),
        builder.constant(F::from_canonical_u8(52)),
        builder.constant(F::from_canonical_u8(53)),
        builder.constant(F::from_canonical_u8(54)),
        builder.constant(F::from_canonical_u8(55)),
        builder.constant(F::from_canonical_u8(56)),
        builder.constant(F::from_canonical_u8(57)),
        builder.constant(F::from_canonical_u8(97)),
        builder.constant(F::from_canonical_u8(98)),
        builder.constant(F::from_canonical_u8(99)),
        builder.constant(F::from_canonical_u8(100)),
        builder.constant(F::from_canonical_u8(101)),
        builder.constant(F::from_canonical_u8(102)),
    ];

    let mut result = vec![];
    for byte in bytes.iter() {
        result.push(U8Target(builder.random_access(*byte, arr.clone())));
    }

    result
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};

    use super::*;
    use crate::types::config::{C, CIRCUIT_CONFIG};

    #[test]
    fn round_timestamp_test() -> Result<()> {
        let curr_time = 1725012611317;
        let divisor = MINUTE_IN_MS;

        let dividend = curr_time / divisor;
        let rounded_down = dividend * divisor;

        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let minute_target = builder.add_virtual_public_input();
        let rounded_down_target =
            round_unix_timestamp_to_previous_minute(&mut builder, minute_target);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();

        pw.set_target(minute_target, F::from_canonical_usize(curr_time))?;
        pw.set_target(rounded_down_target, F::from_canonical_usize(rounded_down))?;

        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
