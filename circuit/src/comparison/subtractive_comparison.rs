// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use log::warn;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::bigint::SignTarget;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::uint::u16::gadgets::arithmetic_u16::{CircuitBuilderU16, U16Target};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
use crate::uint::u48::arithmetic_u48::CircuitBuilderU48;

pub fn cmp_bit_size_bucket(num_bits: usize) -> usize {
    if num_bits <= 16 {
        16
    } else if num_bits <= 32 {
        32
    } else if num_bits <= 48 {
        48
    } else {
        64
    }
}

/// Trait for subtractive comparisons. These will fail if one of the targets exceed given
/// `num_bits`, hence, inputs are expected to be range-checked beforehand.
pub trait CircuitBuilderSubtractiveComparison<F: RichField + Extendable<D>, const D: usize> {
    fn cmp(&mut self, a: Target, b: Target, num_bits: usize) -> SignTarget;

    fn is_lte(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget;
    fn is_lt(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget;
    fn is_gt(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget;
    fn is_gte(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget;

    fn conditional_assert_lte(
        &mut self,
        is_enabled: BoolTarget,
        a: Target,
        b: Target,
        num_bits: usize,
    );
    fn conditional_assert_lt(
        &mut self,
        is_enabled: BoolTarget,
        a: Target,
        b: Target,
        num_bits: usize,
    );
    fn assert_lte(&mut self, a: Target, b: Target, num_bits: usize);

    /// Special case where `a` is SignedTarget, where b is normal `Target` (always non-negative)
    fn conditional_assert_lte_signed_special(
        &mut self,
        is_enabled: BoolTarget,
        a: SignedTarget,
        b: Target,
        num_bits: usize,
    );

    fn min(&mut self, targets: &[Target], max_bits: usize) -> Target;
    fn max(&mut self, targets: &[Target], max_bits: usize) -> Target;
    fn median3(&mut self, a: Target, b: Target, c: Target, num_bits: usize) -> Target;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderSubtractiveComparison<F, D>
    for Builder<F, D>
{
    /// If num_bits > 48, it will use BigUintTarget for comparison. Otherwise, it assumes that
    /// inputs are in the range of 0..2^cmp_bit_size_bucket(num_bits). See [`cmp_bit_size_bucket`]
    fn is_lte(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget {
        assert!(num_bits <= 64, "num_bits must be less than or equal 64");

        if let Some(&cached) = self.is_lte_cache.get(&(a, b)) {
            let cached_num_bits = cached.1;
            if cached_num_bits != num_bits {
                warn!(
                    "is_lte is called with two different num_bits. {} and {}",
                    num_bits, cached_num_bits
                );
            }
            return cached.0;
        }

        let borrow = self.zero();
        let result = if num_bits <= 16 {
            let (_, a_gt_b) = self.sub_u16(U16Target(b), U16Target(a), U16Target(borrow));
            self.not(BoolTarget::new_unsafe(a_gt_b.0))
        } else if num_bits <= 32 {
            let (_, a_gt_b) = self.sub_u32(U32Target(b), U32Target(a), U32Target(borrow));
            self.not(BoolTarget::new_unsafe(a_gt_b.0))
        } else if num_bits <= 48 {
            let (_, a_gt_b) = self.sub_u48(b, a, borrow);
            self.not(BoolTarget::new_unsafe(a_gt_b))
        } else {
            let a_big = self.target_to_biguint(a);
            let b_big = self.target_to_biguint(b);
            self.is_lte_biguint(&a_big, &b_big)
        };

        self.is_lte_cache.insert((a, b), (result, num_bits));

        result
    }

    /// If num_bits > 48, it will use BigUintTarget for comparison. Otherwise, it assumes that
    /// inputs are in the range of 0..2^num_bits.
    fn cmp(&mut self, a: Target, b: Target, num_bits: usize) -> SignTarget {
        assert!(num_bits <= 64, "num_bits must be less than or equal 64");

        if let Some(&cached) = self.cmp_cache.get(&(a, b)) {
            let cached_num_bits = cached.1;
            if cached_num_bits != num_bits {
                warn!(
                    "cmp is called with two different num_bits. {} and {}",
                    num_bits, cached_num_bits
                );
            }
            return cached.0;
        }

        let borrow = self.zero();
        let result = SignTarget::new_unsafe(if num_bits <= 16 {
            let (_, a_gt_b) = self.sub_u16(U16Target(b), U16Target(a), U16Target(borrow));
            let (_, b_gt_a) = self.sub_u16(U16Target(a), U16Target(b), U16Target(borrow));
            self.sub(a_gt_b.0, b_gt_a.0)
        } else if num_bits <= 32 {
            let (_, a_gt_b) = self.sub_u32(U32Target(b), U32Target(a), U32Target(borrow));
            let (_, b_gt_a) = self.sub_u32(U32Target(a), U32Target(b), U32Target(borrow));
            self.sub(a_gt_b.0, b_gt_a.0)
        } else if num_bits <= 48 {
            let (_, a_gt_b) = self.sub_u48(b, a, borrow);
            let (_, b_gt_a) = self.sub_u48(a, b, borrow);
            self.sub(a_gt_b, b_gt_a)
        } else {
            let a_big = self.target_to_biguint(a);
            let b_big = self.target_to_biguint(b);
            let (_, a_gt_b) = self.try_sub_biguint(&b_big, &a_big);
            let (_, b_gt_a) = self.try_sub_biguint(&a_big, &b_big);
            self.sub(a_gt_b.0, b_gt_a.0)
        });

        self.cmp_cache.insert((a, b), (result, num_bits));

        result
    }

    fn is_lt(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget {
        let gte = self.is_gte(a, b, num_bits);
        self.not(gte)
    }

    fn is_gt(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget {
        let lte = self.is_lte(a, b, num_bits);
        self.not(lte)
    }

    fn is_gte(&mut self, a: Target, b: Target, num_bits: usize) -> BoolTarget {
        self.is_lte(b, a, num_bits)
    }

    fn conditional_assert_lte(
        &mut self,
        is_enabled: BoolTarget,
        a: Target,
        b: Target,
        num_bits: usize,
    ) {
        let a_lte_b = self.is_lte(a, b, num_bits);
        let should_be_false = self.and_not(is_enabled, a_lte_b);
        self.assert_false(should_be_false);
    }

    fn assert_lte(&mut self, a: Target, b: Target, num_bits: usize) {
        let a_lte_b = self.is_lte(a, b, num_bits);
        self.assert_true(a_lte_b);
    }

    fn conditional_assert_lte_signed_special(
        &mut self,
        is_enabled: BoolTarget,
        a: SignedTarget,
        b: Target,
        num_bits: usize,
    ) {
        let one = self.one();

        // a > b iff a is positive & abs(a) > b
        let (abs_a, sign_a) = self.abs(a);
        let abs_a_gt_b = self.is_gt(abs_a, b, num_bits);
        let is_a_positive = self.is_equal(sign_a.target, one);
        let a_gt_b = self.and(abs_a_gt_b, is_a_positive);

        let should_be_false = self.and(is_enabled, a_gt_b);
        self.assert_false(should_be_false);
    }

    fn conditional_assert_lt(
        &mut self,
        is_enabled: BoolTarget,
        a: Target,
        b: Target,
        num_bits: usize,
    ) {
        let a_gte_b = self.is_gte(a, b, num_bits);
        let should_be_false = self.and(is_enabled, a_gte_b);
        self.assert_false(should_be_false);
    }

    fn min(&mut self, targets: &[Target], max_bits: usize) -> Target {
        let mut min = targets[0];
        for target in &targets[1..] {
            let is_lte = self.is_lte(*target, min, max_bits);
            min = self.select(is_lte, *target, min);
        }
        min
    }

    fn max(&mut self, targets: &[Target], max_bits: usize) -> Target {
        let mut max = targets[0];
        for target in &targets[1..] {
            let is_gte = self.is_gte(*target, max, max_bits);
            max = self.select(is_gte, *target, max);
        }
        max
    }

    fn median3(&mut self, a: Target, b: Target, c: Target, num_bits: usize) -> Target {
        let max_of_abc = self.max(&[a, b, c], num_bits);
        let min_of_abc = self.min(&[a, b, c], num_bits);

        let sum_abc = self.add_many([a, b, c]);
        let sum_max_min = self.add(max_of_abc, min_of_abc);
        self.sub(sum_abc, sum_max_min)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    #[allow(unused_imports)]
    use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
    use itertools::Itertools;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use rand::Rng;

    use super::*;
    #[allow(unused_imports)]
    use crate::circuit_logger::CircuitBuilderLogging;
    use crate::signed::signed_target::WitnessSigned;
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    #[test]
    fn is_lte_varying_bit_sizes_success() -> Result<()> {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::<F>::new();

        for i in 1..63 {
            for _ in 0..1000 {
                let a = builder.add_virtual_target();
                let b = builder.add_virtual_target();

                builder.assert_lte(a, b, i);

                let mut a_value = rand::thread_rng().gen_range(1u64 << (i - 1)..1u64 << i);
                let mut b_value = rand::thread_rng().gen_range(1u64 << (i - 1)..1u64 << i);
                if a_value > b_value {
                    std::mem::swap(&mut a_value, &mut b_value);
                }
                pw.set_target(a, F::from_canonical_u64(a_value))?;
                pw.set_target(b, F::from_canonical_u64(b_value))?;
            }
        }

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn is_lte_varying_bit_sizes_fail() -> Result<()> {
        for i in 1..63 {
            let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
            let mut pw = PartialWitness::<F>::new();

            let a = builder.add_virtual_target();
            let b = builder.add_virtual_target();

            builder.assert_lte(a, b, i);

            let min = if i == 1 { 0 } else { 1u64 << (i - 1) };
            let max = 1u64 << i;
            let mut a_value = rand::thread_rng().gen_range(min..max);
            let mut b_value = rand::thread_rng().gen_range(min..max);
            loop {
                if b_value != a_value {
                    break;
                }
                b_value = rand::thread_rng().gen_range(min..max);
            }
            if a_value < b_value {
                std::mem::swap(&mut a_value, &mut b_value);
            }
            pw.set_target(a, F::from_canonical_u64(a_value))?;
            pw.set_target(b, F::from_canonical_u64(b_value))?;

            let data = builder.build::<C>();
            if data.prove(pw).is_ok() {
                panic!("Expected proof to fail for i={}", i);
            }
        }
        Ok(())
    }

    #[test]
    fn cmp_varying_bit_sizes_fail() -> Result<()> {
        for i in 1..63 {
            let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
            let mut pw = PartialWitness::<F>::new();

            let a = builder.add_virtual_target();
            let b = builder.add_virtual_target();

            let c = builder.cmp(a, b, i);

            let a_value = rand::thread_rng().gen_range(1u64 << (i - 1)..1u64 << i);
            let b_value = rand::thread_rng().gen_range(1u64 << (i - 1)..1u64 << i);
            let c_value = if a_value < b_value {
                F::ONE
            } else if a_value > b_value {
                F::ZERO
            } else {
                F::NEG_ONE
            };

            pw.set_target(a, F::from_canonical_u64(a_value))?;
            pw.set_target(b, F::from_canonical_u64(b_value))?;
            pw.set_target(c.target, c_value)?;

            let data = builder.build::<C>();
            if data.prove(pw).is_ok() {
                panic!("Expected proof to fail for i={}", i);
            }
        }
        Ok(())
    }

    #[test]
    fn cmp_varying_bit_sizes_success() -> Result<()> {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::<F>::new();

        for i in 1..63 {
            for _ in 0..1000 {
                let a = builder.add_virtual_target();
                let b = builder.add_virtual_target();

                let c = builder.cmp(a, b, i);

                let a_value = rand::thread_rng().gen_range(1u64 << (i - 1)..1u64 << i);
                let b_value = rand::thread_rng().gen_range(1u64 << (i - 1)..1u64 << i);
                let c_value = if a_value < b_value {
                    F::NEG_ONE
                } else if a_value > b_value {
                    F::ONE
                } else {
                    F::ZERO
                };

                pw.set_target(a, F::from_canonical_u64(a_value))?;
                pw.set_target(b, F::from_canonical_u64(b_value))?;
                pw.set_target(c.target, c_value)?;
            }
        }

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn conditional_comparisons_test() -> Result<()> {
        fn conditional_lte(
            builder: &mut Builder<F, 2>,
            pw: &mut PartialWitness<F>,
            a_value: u64,
            b_value: u64,
            c_value: bool,
        ) {
            let a = builder.add_virtual_target();
            let b = builder.add_virtual_target();
            let c = builder.add_virtual_bool_target_unsafe();

            builder.conditional_assert_lte(c, a, b, 63);

            pw.set_target(a, F::from_canonical_u64(a_value)).unwrap();
            pw.set_target(b, F::from_canonical_u64(b_value)).unwrap();
            pw.set_bool_target(c, c_value).unwrap();
        }

        fn conditional_lte_signed(
            builder: &mut Builder<F, 2>,
            pw: &mut PartialWitness<F>,
            a_value: i64,
            b_value: u64,
            c_value: bool,
        ) {
            let a = builder.add_virtual_signed_target();
            let b = builder.add_virtual_target();
            let c = builder.add_virtual_bool_target_unsafe();

            builder.conditional_assert_lte_signed_special(c, a, b, 63);

            pw.set_signed_target(a, a_value).unwrap();
            pw.set_target(b, F::from_canonical_u64(b_value)).unwrap();
            pw.set_bool_target(c, c_value).unwrap();
        }

        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::<F>::new();

        conditional_lte(&mut builder, &mut pw, 3, 5, true);
        conditional_lte(&mut builder, &mut pw, 3, 5, false);
        conditional_lte(&mut builder, &mut pw, 5, 3, false);

        conditional_lte_signed(&mut builder, &mut pw, -1, 1, true);
        conditional_lte_signed(&mut builder, &mut pw, -1, 0, true);
        conditional_lte_signed(&mut builder, &mut pw, 2, 5, true);

        conditional_lte_signed(&mut builder, &mut pw, -1, 1, false);
        conditional_lte_signed(&mut builder, &mut pw, 5, 3, false);

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn min_test() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let targets = [
            builder.add_virtual_target(),
            builder.add_virtual_target(),
            builder.add_virtual_target(),
            builder.add_virtual_target(),
        ];
        let min_target = builder.min(&targets, 63);

        let mut pw = PartialWitness::<F>::new();

        let values: [F; 4] = core::array::from_fn(|_| {
            F::from_canonical_u64(rand::thread_rng().gen_range(1u64 << 0..1u64 << 63))
        });
        let mut min = u64::MAX;
        for value in &values {
            min = std::cmp::min(min, value.to_canonical_u64());
        }
        for (target, value) in targets.iter().zip(values.iter()) {
            pw.set_target(*target, *value).unwrap();
        }
        pw.set_target(min_target, F::from_canonical_u64(min))?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn max_test() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let targets = [
            builder.add_virtual_target(),
            builder.add_virtual_target(),
            builder.add_virtual_target(),
            builder.add_virtual_target(),
        ];
        let max_target = builder.max(&targets, 63);

        let mut pw = PartialWitness::<F>::new();

        let values: [F; 4] = core::array::from_fn(|_| {
            F::from_canonical_u64(rand::thread_rng().gen_range(1u64 << 0..1u64 << 63))
        });
        let mut max = 0;
        for value in &values {
            max = std::cmp::max(max, value.to_canonical_u64());
        }
        for (target, value) in targets.iter().zip_eq(values.iter()) {
            pw.set_target(*target, *value)?;
        }
        pw.set_target(max_target, F::from_canonical_u64(max))?;

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn median3_test() -> Result<()> {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::<F>::new();

        let values = [
            [1, 1, 1, 1],
            [1, 2, 1, 1],
            [1, 1, 2, 1],
            [2, 1, 1, 1],
            [1, 2, 3, 2],
            [3, 2, 1, 2],
            [1, 3, 2, 2],
            [
                (1u64 << 62) + 1,
                (1u64 << 62) + 2,
                (1u64 << 62) + 3,
                (1u64 << 62) + 2,
            ],
            [
                (1u64 << 63) - 1,
                (1u64 << 63) - 2,
                (1u64 << 63) - 3,
                (1u64 << 63) - 2,
            ],
        ];
        let inputs: [Target; 4 * 9] = core::array::from_fn(|_| builder.add_virtual_target());

        inputs.chunks(4).enumerate().for_each(|(i, chunk)| {
            let median = builder.median3(chunk[0], chunk[1], chunk[2], 63);
            builder.connect(median, chunk[3]);

            pw.set_target(chunk[0], F::from_canonical_u64(values[i][0]))
                .unwrap();
            pw.set_target(chunk[1], F::from_canonical_u64(values[i][1]))
                .unwrap();
            pw.set_target(chunk[2], F::from_canonical_u64(values[i][2]))
                .unwrap();
            pw.set_target(chunk[3], F::from_canonical_u64(values[i][3]))
                .unwrap();
        });

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }
}
