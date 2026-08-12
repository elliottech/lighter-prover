// Portions of this file are derived from plonky2-keccak256
// Copyright (c) 2023 qope
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use anyhow::Result;
use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::BoolTarget;
use plonky2::iop::witness::Witness;

use super::constants::*;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::byte::split::CircuitBuilderByteSplit;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
use crate::uint::u32::gadgets::interleaved_u32::CircuitBuilderB32;
use crate::utils::CircuitBuilderUtils;

pub type KeccakInputTarget = Vec<U8Target>;
pub type KeccakOutputTarget = [U8Target; 32];
pub type KeccakOutputBigUintTarget = [U32Target; 8];

#[derive(Clone, Debug)]
pub struct KeccakFTarget {
    words: [[U32Target; 2]; KECCAK_WITDH], // Use 2 U32s to represent each word
}

pub trait CircuitBuilderKeccak<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_keccak_output_target_safe(&mut self) -> KeccakOutputTarget;
    fn add_virtual_keccak_output_target_unsafe(&mut self) -> KeccakOutputTarget;
    fn add_virtual_keccak_output_public_input_safe(&mut self) -> KeccakOutputTarget;
    fn add_virtual_keccak_output_public_input_unsafe(&mut self) -> KeccakOutputTarget;
    fn register_public_keccak_output_input(&mut self, target: KeccakOutputTarget);

    fn zero_keccak_output(&mut self) -> KeccakOutputTarget;

    fn connect_keccakf(&mut self, a: KeccakFTarget, b: KeccakFTarget);
    fn connect_keccak_output(&mut self, a: KeccakOutputTarget, b: KeccakOutputTarget);

    fn conditional_assert_eq_keccak_output(
        &mut self,
        cond: BoolTarget,
        a: KeccakOutputTarget,
        b: KeccakOutputTarget,
    );

    fn is_zero_keccak_output(&mut self, a: KeccakOutputTarget) -> BoolTarget;

    fn select_keccak_output(
        &mut self,
        cond: BoolTarget,
        a: KeccakOutputTarget,
        b: KeccakOutputTarget,
    ) -> KeccakOutputTarget;

    /// Assumes input bits are range-checked
    fn keccak256_circuit(&mut self, input: KeccakInputTarget) -> KeccakOutputTarget;
    fn keccak_round(&mut self, input: &mut KeccakFTarget, rc: [u32; 2]);
    fn keccakf(&mut self, input: &KeccakFTarget) -> KeccakFTarget;

    fn keccak256_circuit_to_biguint(&mut self, input: KeccakInputTarget) -> BigUintTarget;
    fn keccak256_circuit_to_nonnative<FF: Field>(
        &mut self,
        input: KeccakInputTarget,
    ) -> NonNativeTarget<FF>;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderKeccak<F, D> for Builder<F, D> {
    fn add_virtual_keccak_output_target_safe(&mut self) -> KeccakOutputTarget {
        core::array::from_fn(|_| self.add_virtual_u8_target_safe())
    }

    fn add_virtual_keccak_output_target_unsafe(&mut self) -> KeccakOutputTarget {
        core::array::from_fn(|_| self.add_virtual_u8_target_unsafe())
    }

    fn add_virtual_keccak_output_public_input_safe(&mut self) -> KeccakOutputTarget {
        let target = self.add_virtual_keccak_output_target_safe();
        target
            .iter()
            .for_each(|&target| self.register_public_u8_input(target));
        target
    }

    fn add_virtual_keccak_output_public_input_unsafe(&mut self) -> KeccakOutputTarget {
        let target = self.add_virtual_keccak_output_target_unsafe();
        target
            .iter()
            .for_each(|&target| self.register_public_u8_input(target));
        target
    }

    fn register_public_keccak_output_input(&mut self, target: KeccakOutputTarget) {
        target
            .iter()
            .for_each(|&target| self.register_public_u8_input(target));
    }

    fn zero_keccak_output(&mut self) -> KeccakOutputTarget {
        core::array::from_fn(|_| self.zero_u8())
    }

    fn connect_keccakf(&mut self, a: KeccakFTarget, b: KeccakFTarget) {
        a.words
            .iter()
            .flatten()
            .zip_eq(b.words.iter().flatten())
            .for_each(|(&a, &b)| {
                self.connect_u32(a, b);
            })
    }

    fn connect_keccak_output(&mut self, a: KeccakOutputTarget, b: KeccakOutputTarget) {
        a.iter().zip_eq(b.iter()).for_each(|(&a, &b)| {
            self.connect_u8(a, b);
        });
    }

    fn conditional_assert_eq_keccak_output(
        &mut self,
        cond: BoolTarget,
        a: KeccakOutputTarget,
        b: KeccakOutputTarget,
    ) {
        a.iter().zip_eq(b.iter()).for_each(|(&a, &b)| {
            self.conditional_assert_eq(cond, a.0, b.0);
        });
    }

    fn is_zero_keccak_output(&mut self, a: KeccakOutputTarget) -> BoolTarget {
        let results: Vec<BoolTarget> = a.iter().map(|&byte| self.is_zero(byte.0)).collect();

        self.multi_and(&results)
    }

    fn select_keccak_output(
        &mut self,
        cond: BoolTarget,
        a: KeccakOutputTarget,
        b: KeccakOutputTarget,
    ) -> KeccakOutputTarget {
        self.select_arr_u8(cond, &a, &b)
    }

    /// Warn: Assumes inputs are range-checked
    fn keccak256_circuit(&mut self, input: KeccakInputTarget) -> KeccakOutputTarget {
        let mut input = input.clone();
        let block_size_in_bytes = 136; // in bytes
        let input_len_in_bytes = input.len();
        let num_blocks = input_len_in_bytes / block_size_in_bytes + 1;

        input.push(self.one_u8());
        input.resize(num_blocks * block_size_in_bytes, self.zero_u8());
        let last_index = num_blocks * block_size_in_bytes - 1;
        let final_pad_bit = self.constant_u8(128);
        input[last_index] = U8Target(self.add(input[last_index].0, final_pad_bit.0));

        // Convert to biguint
        input.reverse();
        let padded = self.biguint_from_bytes_be(&input);
        assert_eq!(
            padded.byte_len(),
            num_blocks * block_size_in_bytes,
            "Padded input length should match the number of blocks times the block size in bytes"
        );

        let mut m = KeccakFTarget {
            words: core::array::from_fn(|_| [self.zero_u32(); 2]),
        };

        for i in 0..num_blocks {
            for j in 0..block_size_in_bytes / 8 {
                let xor_t = self.xor_u64(
                    &m.words[j],
                    &[
                        padded.limbs[i * block_size_in_bytes / 4 + j * 2],
                        padded.limbs[i * block_size_in_bytes / 4 + j * 2 + 1],
                    ],
                );
                m.words[j] = xor_t;
            }
            m = self.keccakf(&m);
        }

        m.words
            .iter()
            .take(4)
            .flatten()
            .flat_map(|&t| self.split_bytes(t.0, 4))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn keccak_round(&mut self, input: &mut KeccakFTarget, rc: [u32; 2]) {
        // θ step
        let mut c = vec![];
        for x in 0..5 {
            let xor01 = self.xor_u64(&input.words[x], &input.words[x + 5]);
            let xor012 = self.xor_u64(&xor01, &input.words[x + 2 * 5]);
            let xor0123 = self.xor_u64(&xor012, &input.words[x + 3 * 5]);
            let xor01234 = self.xor_u64(&xor0123, &input.words[x + 4 * 5]);
            c.push(xor01234);
        }
        let mut d = vec![];
        for x in 0..5 {
            let rot_c = self.lrot_u64(&c[(x + 1) % 5], 1);
            d.push(self.xor_u64(&c[(x + 4) % 5], &rot_c));
        }
        for x in 0..5 {
            for y in 0..5 {
                input.words[x + y * 5] = self.xor_u64(&input.words[x + y * 5], &d[x]);
            }
        }
        // ρ and π steps
        let mut b_words: [[U32Target; 2]; KECCAK_WITDH] =
            core::array::from_fn(|_| [self.zero_u32(); 2]);
        for x in 0..5 {
            for y in 0..5 {
                let rot_input = self.lrot_u64(&input.words[x + y * 5], ROTR[x + y * 5]);

                b_words[y + ((2 * x + 3 * y) % 5) * 5] = rot_input;
            }
        }
        let b = KeccakFTarget { words: b_words };

        // χ step
        for x in 0..5 {
            for y in 0..5 {
                // b.words[(x + 2) % 5 + y * 5] & !b.words[(x + 1) % 5 + y * 5]
                let not = self.not_u64(&b.words[(x + 1) % 5 + y * 5]);
                let and_not_b = self.and_u64(&b.words[(x + 2) % 5 + y * 5], &not);
                input.words[x + y * 5] = self.xor_u64(&b.words[x + y * 5], &and_not_b);
            }
        }

        let rc: [U32Target; 2] = [self.constant_u32(rc[0]), self.constant_u32(rc[1])];
        input.words[0] = self.xor_u64(&input.words[0], &rc);
    }

    fn keccakf(&mut self, input: &KeccakFTarget) -> KeccakFTarget {
        let mut result = input.clone();
        for round_constant in ROUND_CONSTANTS.into_iter().take(24) {
            self.keccak_round(&mut result, round_constant);
        }

        result
    }

    fn keccak256_circuit_to_biguint(&mut self, input: KeccakInputTarget) -> BigUintTarget {
        let z = self.keccak256_circuit(input);
        self.biguint_from_bytes_be(&z)
    }

    fn keccak256_circuit_to_nonnative<FF: Field>(
        &mut self,
        input: KeccakInputTarget,
    ) -> NonNativeTarget<FF> {
        let z = self.keccak256_circuit(input);
        let big = self.biguint_from_bytes_be(&z);
        self.biguint_to_nonnative(&big)
    }
}

pub trait WitnessKeccak<F: PrimeField64>: Witness<F> {
    fn set_keccak_output_target(
        &mut self,
        target: &KeccakOutputTarget,
        value: &[u8; 32],
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> WitnessKeccak<F> for T {
    fn set_keccak_output_target(
        &mut self,
        target: &KeccakOutputTarget,
        value: &[u8; 32],
    ) -> Result<()> {
        target
            .iter()
            .zip_eq(value.iter())
            .try_for_each(|(target, &byte)| {
                self.set_target(target.0, F::from_canonical_u8(byte))
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use anyhow::Result;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use rand::random;
    use tiny_keccak::{Hasher, Keccak};

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    use super::*;
    use crate::keccak::helpers::keccak;

    fn expected_keccak(input: &[u8]) -> String {
        let mut hasher = Keccak::v256();
        hasher.update(input);
        let mut hash = [0u8; 32];
        hasher.finalize(&mut hash);

        hex::encode(hash)
    }

    #[test]
    fn test_keccak_without_hex() -> Result<()> {
        let input_bytes: [u8; 32] = [
            136, 67, 150, 149, 65, 184, 228, 123, 233, 65, 190, 5, 147, 211, 43, 34, 69, 124, 238,
            8, 75, 26, 5, 70, 45, 221, 34, 207, 144, 28, 81, 37,
        ];
        let output = keccak(&input_bytes);

        let expected_output: [u8; 32] = [
            38, 34, 78, 151, 161, 85, 137, 0, 165, 233, 72, 198, 242, 209, 242, 192, 242, 10, 230,
            3, 126, 254, 109, 28, 62, 35, 138, 66, 215, 182, 185, 201,
        ];
        for i in 0..32 {
            assert_eq!(output[i], expected_output[i]);
        }

        let mut builder = Builder::<F, D>::new(CircuitConfig::standard_recursion_config());

        let mut input_t = vec![];
        for i in 0..input_bytes.len() {
            input_t.push(builder.constant_u8(input_bytes[i]));
        }
        let output_t = builder.keccak256_circuit(input_t);

        let mut pw = PartialWitness::new();
        for i in 0..32 {
            pw.set_target(output_t[i].0, F::from_canonical_u8(expected_output[i]))?;
        }

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn test_keccak256_circuit() -> Result<()> {
        let input = "8f54f1c2d0eb5771cd5bf67a6689fcd6eed9444d91a39e5ef32a9b4ae5ca14ff8f54f1c2d0eb5771cd5bf67a6689fcd6eed9444d91a39e5ef32a9b4ae5ca14ff";
        let expected_output = expected_keccak(&hex::decode(input).unwrap());

        let input_bytes = hex::decode(input)?;
        let exptected_output_bytes = hex::decode(&expected_output)?;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let mut input_t = vec![];
        for i in 0..input_bytes.len() {
            input_t.push(builder.constant_u8(input_bytes[i]));
        }
        let output_t = builder.keccak256_circuit(input_t);

        let mut pw = PartialWitness::new();
        for i in 0..32 {
            pw.set_target(
                output_t[i].0,
                F::from_canonical_u8(exptected_output_bytes[i]),
            )?;
        }

        let data = builder.build::<C>();
        let now = Instant::now();
        let proof = data.prove(pw)?;

        println!("time = {} ms", now.elapsed().as_millis());
        println!(
            "degree = {}, degree_bits= {}",
            data.common.degree(),
            data.common.degree_bits()
        );

        data.verify(proof)?;
        Ok(())
    }

    #[test]
    fn test_random_keccak256_circuit() -> Result<()> {
        let input_len: usize = random();
        let input_len = input_len % 128;
        let input: Vec<u8> = (0..input_len).map(|_| random()).collect();
        let input_bytes = input;

        let expected_output = expected_keccak(&input_bytes);
        let exptected_output_bytes = hex::decode(&expected_output)?;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let mut input_t = vec![];
        for i in 0..input_bytes.len() {
            input_t.push(builder.constant_u8(input_bytes[i]));
        }
        let output_t = builder.keccak256_circuit(input_t);

        let mut pw = PartialWitness::new();
        for i in 0..32 {
            pw.set_target(
                output_t[i].0,
                F::from_canonical_u8(exptected_output_bytes[i]),
            )?;
        }

        let data = builder.build::<C>();
        let now = Instant::now();
        let proof = data.prove(pw)?;

        println!("time = {} ms", now.elapsed().as_millis());
        println!(
            "degree = {}, degree_bits= {}",
            data.common.degree(),
            data.common.degree_bits()
        );

        data.verify(proof)?;
        Ok(())
    }
}
