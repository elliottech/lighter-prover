// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::big_u16::BigIntU16Target;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::builder::Builder;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};

/// UnsafeBigTarget is a vector [x] that represents a big integer
/// X = x[0] + x[1] * 2^B + x[2] * 2^(2*B) + ...
/// but x[i] does not have to be smaller than 2^B or positive.
/// We can treat them as signed integers(see [`crate::signed::signed_target::SignedTarget`]).
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct UnsafeBigTarget<const B: usize> {
    pub limbs: Vec<Target>,
}

pub trait CircuitBuilderUnsafeBig<F: RichField + Extendable<D>, const D: usize> {
    fn unsafe_big_from_biguint(&mut self, a: &BigUintTarget) -> UnsafeBigTarget<32>;

    fn mul_unsafe_big_by_bool<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: BoolTarget,
    ) -> UnsafeBigTarget<B>;

    fn target_to_unsafe_big_u32(&mut self, x: Target) -> UnsafeBigTarget<32>;

    fn add_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: &UnsafeBigTarget<B>,
    ) -> UnsafeBigTarget<B>;

    /// Returns [a] * c + [b]
    fn mul_add_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        c: Target,
        b: &UnsafeBigTarget<B>,
    ) -> UnsafeBigTarget<B>;

    fn mul_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: &UnsafeBigTarget<B>,
        num_limbs: usize,
    ) -> UnsafeBigTarget<B>;

    /// Returns [a] * [b] + [acc], accumulating the limb products directly into [acc]'s limbs
    fn mul_acc_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: &UnsafeBigTarget<B>,
        acc: &UnsafeBigTarget<B>,
    ) -> UnsafeBigTarget<B>;

    fn sub_bigint_u16_unsafe(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> UnsafeBigTarget<16>;

    fn mul_bigint_u16_and_target_unsafe(
        &mut self,
        a: &BigIntU16Target,
        b: Target,
    ) -> UnsafeBigTarget<16>;

    fn unsafe_big32_to_biguint(
        &mut self,
        a: &UnsafeBigTarget<32>,
        big_uint_limb_size: usize,
    ) -> BigUintTarget;

    fn unsafe_big16_to_bigint(
        &mut self,
        a: &UnsafeBigTarget<16>,
        big_int_limb_size: usize,
    ) -> BigIntTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderUnsafeBig<F, D> for Builder<F, D> {
    fn unsafe_big_from_biguint(&mut self, a: &BigUintTarget) -> UnsafeBigTarget<32> {
        UnsafeBigTarget {
            limbs: a.limbs.iter().map(|&l| l.0).collect(),
        }
    }

    fn mul_unsafe_big_by_bool<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: BoolTarget,
    ) -> UnsafeBigTarget<B> {
        let t = b.target;

        UnsafeBigTarget {
            limbs: a.limbs.iter().map(|&l| self.mul(l, t)).collect(),
        }
    }

    fn target_to_unsafe_big_u32(&mut self, x: Target) -> UnsafeBigTarget<32> {
        let limbs = self
            .split_u64_to_u32s_le(x)
            .iter()
            .map(|u| u.0)
            .collect::<Vec<_>>();
        UnsafeBigTarget { limbs }
    }

    fn add_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: &UnsafeBigTarget<B>,
    ) -> UnsafeBigTarget<B> {
        assert_eq!(a.limbs.len(), b.limbs.len());
        let num_limbs = a.limbs.len();

        let mut result = UnsafeBigTarget {
            limbs: vec![self.zero(); num_limbs],
        };
        for i in 0..num_limbs {
            result.limbs[i] = self.add(a.limbs[i], b.limbs[i]);
        }

        result
    }

    fn mul_add_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        c: Target,
        b: &UnsafeBigTarget<B>,
    ) -> UnsafeBigTarget<B> {
        assert_eq!(a.limbs.len(), b.limbs.len());
        let num_limbs = a.limbs.len();

        let mut result = UnsafeBigTarget {
            limbs: vec![self.zero(); num_limbs],
        };
        for i in 0..num_limbs {
            result.limbs[i] = self.mul_add(a.limbs[i], c, b.limbs[i]);
        }

        result
    }

    fn mul_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: &UnsafeBigTarget<B>,
        num_limbs: usize,
    ) -> UnsafeBigTarget<B> {
        let mut result = UnsafeBigTarget {
            limbs: vec![self.zero(); num_limbs],
        };
        assert!(a.limbs.len() + b.limbs.len() - 1 <= num_limbs);
        for i in 0..a.limbs.len() {
            for j in 0..b.limbs.len() {
                result.limbs[i + j] = self.mul_add(a.limbs[i], b.limbs[j], result.limbs[i + j]);
            }
        }
        result
    }

    fn mul_acc_unsafe_big<const B: usize>(
        &mut self,
        a: &UnsafeBigTarget<B>,
        b: &UnsafeBigTarget<B>,
        acc: &UnsafeBigTarget<B>,
    ) -> UnsafeBigTarget<B> {
        assert!(a.limbs.len() + b.limbs.len() - 1 <= acc.limbs.len());
        let mut result = acc.clone();
        for i in 0..a.limbs.len() {
            for j in 0..b.limbs.len() {
                result.limbs[i + j] = self.mul_add(a.limbs[i], b.limbs[j], result.limbs[i + j]);
            }
        }
        result
    }

    fn sub_bigint_u16_unsafe(
        &mut self,
        a: &BigIntU16Target,
        b: &BigIntU16Target,
    ) -> UnsafeBigTarget<16> {
        assert_eq!(a.abs.limbs.len(), b.abs.limbs.len());

        let unsafe_a = UnsafeBigTarget::<16> {
            limbs: a
                .abs
                .limbs
                .iter()
                .map(|&l| self.mul(l.0, a.sign.target))
                .collect(),
        };

        UnsafeBigTarget {
            limbs: unsafe_a
                .limbs
                .iter()
                .zip_eq(b.abs.limbs.iter())
                .map(|(&a_limb, &b_limb)| {
                    self.arithmetic(F::NEG_ONE, F::ONE, b_limb.0, b.sign.target, a_limb)
                })
                .collect(),
        }
    }

    fn mul_bigint_u16_and_target_unsafe(
        &mut self,
        a: &BigIntU16Target,
        b: Target,
    ) -> UnsafeBigTarget<16> {
        let multiplier = self.mul(b, a.sign.target);

        UnsafeBigTarget {
            limbs: a
                .abs
                .limbs
                .iter()
                .map(|&l| self.mul(l.0, multiplier))
                .collect(),
        }
    }

    // Only call if limbs are not reduced and 32 % limb_bit_size == 0
    /// Interprets the limbs as unsigned integers and normalizes them to fit into a `BigUintTarget`.
    fn unsafe_big32_to_biguint(
        &mut self,
        a: &UnsafeBigTarget<32>,
        big_uint_limb_size: usize,
    ) -> BigUintTarget {
        assert!(big_uint_limb_size >= a.limbs.len());
        let mut normalized_limbs = vec![self.zero_u32(); big_uint_limb_size];
        let mut carry = self.zero();
        for index in 0..normalized_limbs.len() {
            let mut unsafe_limb_with_carry = carry;
            if index < a.limbs.len() {
                unsafe_limb_with_carry = self.add(unsafe_limb_with_carry, a.limbs[index]);
            }
            let split_limb_with_carry = self.split_u64_to_u32s_le(unsafe_limb_with_carry).to_vec();
            normalized_limbs[index] = split_limb_with_carry[0];
            carry = split_limb_with_carry[1].0;
        }
        self.assert_zero(carry);
        BigUintTarget {
            limbs: normalized_limbs,
        }
    }

    /// Interprets the limbs as signed integers and normalizes them to fit into a `BigIntTarget`.
    fn unsafe_big16_to_bigint(
        &mut self,
        a: &UnsafeBigTarget<16>,
        big_int_limb_size: usize,
    ) -> BigIntTarget {
        assert!(big_int_limb_size * 2 >= a.limbs.len());
        let mut normalized_limbs = vec![self.zero(); big_int_limb_size * 2];
        let mut carry = self.zero();
        let pow = self.constant(F::from_canonical_i64(1 << 16));
        let pow_minus_one = self.constant(F::from_canonical_i64((1 << 16) - 1));
        for index in 0..normalized_limbs.len() {
            let mut unsafe_limb_with_carry = carry;
            if index < a.limbs.len() {
                unsafe_limb_with_carry = self.add(unsafe_limb_with_carry, a.limbs[index]);
            }
            let (abs_limb, sign_limb) = self.abs(SignedTarget::new_unsafe(unsafe_limb_with_carry));
            let split_limb_with_carry = self.split_u64_to_u16s_le(abs_limb, 4).to_vec();
            let is_limb_negative = self.is_sign_negative(sign_limb);

            let mut low = self.mul_add(pow, is_limb_negative.target, split_limb_with_carry[0].0);
            low = self.arithmetic(
                F::NEG_ONE * F::TWO,
                F::ONE,
                is_limb_negative.target,
                split_limb_with_carry[0].0,
                low,
            );
            normalized_limbs[index] = low;

            let mut high =
                self.mul_add(split_limb_with_carry[3].0, pow, split_limb_with_carry[2].0);
            high = self.mul_add(high, pow, split_limb_with_carry[1].0);
            high = self.add(high, is_limb_negative.target);
            carry = self.arithmetic(
                F::NEG_ONE * F::TWO,
                F::ONE,
                is_limb_negative.target,
                high,
                high,
            );
        }
        let is_carry_invalid = self.arithmetic(F::ONE, F::ONE, carry, carry, carry);
        self.assert_zero(is_carry_invalid);

        let zero = self.zero();
        let one = self.one();
        let neg_one = self.neg_one();
        let is_negative = self.is_equal(carry, neg_one);
        let add_to_limbs = self.mul(is_negative.target, pow_minus_one);
        let mul_with_limbs = self.select(is_negative, neg_one, one);
        let mut big_int_limbs = vec![self.zero_u32(); big_int_limb_size];
        carry = is_negative.target;

        // If the final carry is -1, value needs to be converted from x to 2^(16 * num_limbs) - x.
        // 2^(16 * num_limbs) - 1 = [sum (2^(16*i) * (2^16 - 1)) for all i < num_limbs].
        // Substracting x from each side of the equation and adding 1 gives:
        // 2^(16 * num_limbs) - x = [sum (2^(16*i) * (2^16 - 1 - limb[i])) for all i < num_limbs] + 1
        // First, for each i, convert limb[i] to (2^16 - 1 - limb[i]), then add 1 to the result.
        for i in 0..normalized_limbs.len() {
            normalized_limbs[i] = self.mul_add(normalized_limbs[i], mul_with_limbs, add_to_limbs);
            normalized_limbs[i] = self.add(normalized_limbs[i], carry);
            let is_limb_full = self.is_equal(normalized_limbs[i], pow);
            normalized_limbs[i] = self.select(is_limb_full, zero, normalized_limbs[i]);
            if i & 1 == 1 {
                big_int_limbs[i / 2] =
                    U32Target(self.mul_add(normalized_limbs[i], pow, normalized_limbs[i - 1]));
            }
            carry = is_limb_full.target;
        }
        let abs = BigUintTarget {
            limbs: big_int_limbs,
        };
        let is_zero = self.is_zero_biguint(&abs);
        let mut sign = self.select(is_zero, zero, one);
        sign = self.select(is_negative, neg_one, sign);
        BigIntTarget {
            abs,
            sign: SignTarget::new_unsafe(sign),
        }
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use num::{BigInt, BigUint, FromPrimitive};
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};

    use super::*;
    use crate::bigint::big_u16::{CircuitBuilderBigIntU16, WitnessBigInt16};
    use crate::bigint::bigint::WitnessBigInt;
    use crate::bigint::biguint::WitnessBigUint;
    use crate::types::config::{BIGU16_U64_LIMBS, BIGU16_U112_LIMBS, C, CIRCUIT_CONFIG, F};
    use crate::types::constants::{ISOLATED_MARGIN, POSITION_LIST_SIZE};

    #[test]
    fn test_mul_unsafe_big() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let a = builder.add_virtual_biguint_target_unsafe(3);
        let b = builder.add_virtual_biguint_target_unsafe(3);

        let a_unsafe = builder.unsafe_big_from_biguint(&a);
        let b_unsafe = builder.unsafe_big_from_biguint(&b);
        let c_unsafe = builder.mul_unsafe_big(&a_unsafe, &b_unsafe, 6);

        builder.connect_constant(c_unsafe.limbs[0], 10);
        builder.connect_constant(c_unsafe.limbs[1], 29); // 2*7 + 3*5
        builder.connect_constant(c_unsafe.limbs[2], 21); // 3*7
        builder.connect_constant(c_unsafe.limbs[3], 0);
        builder.connect_constant(c_unsafe.limbs[4], 0);
        builder.connect_constant(c_unsafe.limbs[5], 0);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();

        pw.set_biguint_target(&a, &BigUint::from_i64(2 + 3 * (1 << 32)).unwrap())?;
        pw.set_biguint_target(&b, &BigUint::from_i64(5 + 7 * (1 << 32)).unwrap())?;

        data.verify(data.prove(pw).unwrap())?;

        Ok(())
    }

    #[test]
    fn test_sub_bigint_u16_unsafe() -> Result<()> {
        let p: u64 = 0xffffffff00000001;

        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let a = builder.add_virtual_bigint_u16_target_unsafe(3);
        let b = builder.add_virtual_bigint_u16_target_unsafe(3);

        let x = builder.sub_bigint_u16_unsafe(&a, &b);
        builder.connect_constant(x.limbs[0], 3);
        builder.connect_constant(x.limbs[1], p - 4);
        builder.connect_constant(x.limbs[2], 0);

        let c = builder.add_virtual_bigint_u16_target_unsafe(3);
        let d = builder.add_virtual_bigint_u16_target_unsafe(3);
        let x = builder.sub_bigint_u16_unsafe(&c, &d);
        builder.connect_constant(x.limbs[0], 7);
        builder.connect_constant(x.limbs[1], 10);
        builder.connect_constant(x.limbs[2], 0);

        let e = builder.add_virtual_bigint_u16_target_unsafe(3);
        let f = builder.add_virtual_bigint_u16_target_unsafe(3);
        let x = builder.sub_bigint_u16_unsafe(&e, &f);
        builder.connect_constant(x.limbs[0], p - 7);
        builder.connect_constant(x.limbs[1], p - 10);
        builder.connect_constant(x.limbs[2], 0);

        let k = builder.add_virtual_bigint_u16_target_unsafe(3);
        let l = builder.add_virtual_bigint_u16_target_unsafe(3);
        let x = builder.sub_bigint_u16_unsafe(&k, &l);
        builder.connect_constant(x.limbs[0], p - 3);
        builder.connect_constant(x.limbs[1], 4);
        builder.connect_constant(x.limbs[2], 0);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();

        pw.set_bigint_u16_target(&a, &BigInt::from_i64(5 + 3 * (1 << 16)).unwrap())?;
        pw.set_bigint_u16_target(&b, &BigInt::from_i64(2 + 7 * (1 << 16)).unwrap())?;

        pw.set_bigint_u16_target(&c, &BigInt::from_i64(5 + 3 * (1 << 16)).unwrap())?;
        pw.set_bigint_u16_target(&d, &BigInt::from_i64(-(2 + 7 * (1 << 16))).unwrap())?;

        pw.set_bigint_u16_target(&e, &BigInt::from_i64(-(5 + 3 * (1 << 16))).unwrap())?;
        pw.set_bigint_u16_target(&f, &BigInt::from_i64(2 + 7 * (1 << 16)).unwrap())?;

        pw.set_bigint_u16_target(&k, &BigInt::from_i64(-(5 + 3 * (1 << 16))).unwrap())?;
        pw.set_bigint_u16_target(&l, &BigInt::from_i64(-(2 + 7 * (1 << 16))).unwrap())?;

        data.verify(data.prove(pw).unwrap())?;

        Ok(())
    }

    #[test]
    fn test_signed_16_bit_operation() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let mut unsafe_unrealized_funding = UnsafeBigTarget {
            limbs: vec![builder.zero(); BIGU16_U112_LIMBS],
        };

        let market_frps: [BigIntU16Target; POSITION_LIST_SIZE] = core::array::from_fn(|_| {
            builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS)
        });
        let market_quote_muls: [Target; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| builder.add_virtual_target());
        let position_frps: [BigIntU16Target; POSITION_LIST_SIZE] = core::array::from_fn(|_| {
            builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS)
        });
        let positions: [BigIntU16Target; POSITION_LIST_SIZE] = core::array::from_fn(|_| {
            builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS)
        });
        let position_modes: [Target; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| builder.add_virtual_target());

        for market_index in 0..POSITION_LIST_SIZE {
            let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
            let is_isolated_position =
                builder.is_equal(position_modes[market_index], isolated_margin_mode);
            let is_cross_position = builder.not(is_isolated_position);

            let lhs = builder
                .sub_bigint_u16_unsafe(&position_frps[market_index], &market_frps[market_index]);
            let rhs = builder.mul_bigint_u16_and_target_unsafe(
                &positions[market_index],
                market_quote_muls[market_index],
            );

            let unsafe_position_unrealized_funding =
                builder.mul_unsafe_big(&lhs, &rhs, BIGU16_U112_LIMBS);

            unsafe_unrealized_funding = builder.mul_add_unsafe_big(
                &unsafe_position_unrealized_funding,
                is_cross_position.target,
                &unsafe_unrealized_funding,
            );
        }
        let result = builder.unsafe_big16_to_bigint(&unsafe_unrealized_funding, BIGU16_U112_LIMBS);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();

        // Generate random values
        let rnd_market_frps: [BigInt; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| BigInt::from_i64(rand::random::<i64>()).unwrap());
        let rnd_market_quote_muls: [u64; POSITION_LIST_SIZE] = core::array::from_fn(|_| 1u64); // TODO: randomize
        let rnd_position_frps: [BigInt; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| BigInt::from_i64(rand::random::<i64>()).unwrap());
        let rnd_positions: [BigInt; POSITION_LIST_SIZE] = // 56 bit
            core::array::from_fn(|_| {
                let abs = rand::random::<u64>() % (1 << 56);
                if rand::random::<bool>() {
                    BigInt::from_u64(abs).unwrap()
                } else {
                    BigInt::from_u64(abs).unwrap() * BigInt::from_i64(-1).unwrap()
                }
            });
        let rnd_position_modes: [u64; POSITION_LIST_SIZE] =
            core::array::from_fn(|_| rand::random::<u64>() % 2);

        rnd_market_frps
            .iter()
            .zip_eq(market_frps.iter())
            .for_each(|(value, target)| {
                pw.set_bigint_u16_target(target, value).unwrap();
            });
        rnd_market_quote_muls
            .iter()
            .zip_eq(market_quote_muls.iter())
            .for_each(|(value, &target)| {
                pw.set_target(target, F::from_canonical_u64(*value))
                    .unwrap();
            });
        rnd_position_frps
            .iter()
            .zip_eq(position_frps.iter())
            .for_each(|(value, target)| {
                pw.set_bigint_u16_target(target, value).unwrap();
            });
        rnd_positions
            .iter()
            .zip_eq(positions.iter())
            .for_each(|(value, target)| {
                pw.set_bigint_u16_target(target, value).unwrap();
            });
        rnd_position_modes
            .iter()
            .zip_eq(position_modes.iter())
            .for_each(|(value, &target)| {
                pw.set_target(target, F::from_canonical_u64(*value))
                    .unwrap();
            });

        // Calculate real result
        let mut rnd_result = BigInt::ZERO;
        for market_index in 0..POSITION_LIST_SIZE {
            let market_frp = &rnd_market_frps[market_index];
            let market_quote_mul = rnd_market_quote_muls[market_index];
            let position_frp = &rnd_position_frps[market_index];
            let position = &rnd_positions[market_index];
            let position_mode = rnd_position_modes[market_index];
            let isolated_margin_mode = ISOLATED_MARGIN as u64;
            let is_isolated_position = position_mode == isolated_margin_mode;
            let is_cross_position = !is_isolated_position;
            let lhs = position_frp - market_frp;
            let rhs = position * BigInt::from_u64(market_quote_mul).unwrap();
            let unsafe_position_unrealized_funding = lhs * rhs;
            if is_cross_position {
                rnd_result += unsafe_position_unrealized_funding;
            }
        }
        println!("Unsafe unrealized funding: {}", rnd_result);
        pw.set_bigint_target(&result, &rnd_result)?;

        data.verify(data.prove(pw).unwrap())?;

        Ok(())
    }
}
