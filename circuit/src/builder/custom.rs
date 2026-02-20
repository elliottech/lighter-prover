// Portions of this file are derived from plonky2
// Copyright (c) 2022-2025 The Plonky2 Authors
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use anyhow::Result;
use hashbrown::HashMap;
use itertools::Itertools;
use log::warn;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_data::{CircuitData, CommonCircuitData, VerifierOnlyCircuitData};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::recursion::dummy_circuit::dummy_proof;

use super::Builder;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::uint::u8::{CircuitBuilderU8, U8Target};

// Customized delegations
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    #[track_caller]
    pub fn split_le(&mut self, x: Target, bit_count: usize) -> Vec<BoolTarget> {
        if let Some(result) = self.split_le_cache.get(&x) {
            if result.len() == bit_count {
                return result.clone();
            }

            let caller_location = std::panic::Location::caller();
            let caller_file_name = caller_location.file();
            let caller_line_number = caller_location.line();
            warn!(
                "split_le({:?}) is called with different bit_count({}/{}). Please fix this!. Caller {}:{}",
                x,
                result.len(),
                bit_count,
                caller_file_name,
                caller_line_number
            );
        }

        let bits = self.builder.split_le(x, bit_count);
        self.split_le_cache.insert(x, bits.clone());
        bits
    }

    pub fn split_le_base<const B: usize>(&mut self, x: Target, num_limbs: usize) -> Vec<Target> {
        if let Some(result) = self.split_le_base_cache.get(&(B, x)) {
            if result.len() == num_limbs {
                return result.clone();
            }
            warn!("split_le_base is called with different num_limbs. Please fix this!");
        }

        let bits = self.builder.split_le_base::<B>(x, num_limbs);
        self.split_le_base_cache.insert((B, x), bits.clone());
        bits
    }

    #[track_caller]
    pub fn conditional_assert_eq(&mut self, condition: BoolTarget, x: Target, y: Target) {
        self.builder.conditional_assert_eq(condition.target, x, y)
    }

    #[track_caller]
    pub fn conditional_assert_eq_constant(&mut self, condition: BoolTarget, x: Target, y: u64) {
        let y_target = self.constant_u64(y);
        self.builder
            .conditional_assert_eq(condition.target, x, y_target)
    }

    // If b is true, return b1 & b2, else return b1
    pub fn conditional_and(&mut self, b: BoolTarget, b1: BoolTarget, b2: BoolTarget) -> BoolTarget {
        let and = self.and(b1, b2);
        self.select_bool(b, and, b1)
    }

    /// a - b*c
    pub fn neg_mul_sub(&mut self, a: Target, b: Target, c: Target) -> Target {
        self.builder.arithmetic(F::NEG_ONE, F::ONE, b, c, a)
    }

    /// a * b where a is a boolean
    pub fn mul_bool(&mut self, a: BoolTarget, b: Target) -> Target {
        self.builder.mul(a.target, b)
    }

    pub fn constant_from_u8(&mut self, n: u8) -> Target {
        self.constant(F::from_canonical_u8(n))
    }

    pub fn constant_i64(&mut self, n: i64) -> Target {
        self.constant(F::from_canonical_i64(n))
    }

    pub fn constant_u64(&mut self, n: u64) -> Target {
        self.constant(F::from_canonical_u64(n))
    }

    pub fn constant_usize(&mut self, n: usize) -> Target {
        self.constant(F::from_canonical_usize(n))
    }

    #[must_use]
    pub fn is_equal(&mut self, x: Target, y: Target) -> BoolTarget {
        let cached_result = self.is_equal_cache.get(&(x, y));

        if let Some(result) = cached_result {
            return *result;
        }

        let result = self.builder.is_equal(x, y);
        self.is_equal_cache.insert((x, y), result);
        self.is_equal_cache.insert((y, x), result);

        result
    }

    #[must_use]
    pub fn is_equal_constant(&mut self, x: Target, y: u64) -> BoolTarget {
        let y_target = self.constant_u64(y);
        self.is_equal(x, y_target)
    }

    #[must_use]
    pub fn is_not_equal(&mut self, a: Target, b: Target) -> BoolTarget {
        let is_eq = self.is_equal(a, b);

        // This not operation is actually free if not `equality_gate_enable()`
        self.not(is_eq)
    }

    #[must_use]
    pub fn select_arr<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[Target; N],
        b: &[Target; N],
    ) -> [Target; N] {
        a.iter()
            .zip_eq(b.iter())
            .map(|(&a, &b)| self.select(cond, a, b))
            .collect::<Vec<Target>>()
            .try_into()
            .unwrap()
    }

    #[must_use]
    pub fn select_arr_bool<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[BoolTarget; N],
        b: &[BoolTarget; N],
    ) -> [BoolTarget; N] {
        a.iter()
            .zip_eq(b.iter())
            .map(|(&a, &b)| self.select_bool(cond, a, b))
            .collect::<Vec<BoolTarget>>()
            .try_into()
            .unwrap()
    }

    #[must_use]
    pub fn select_arr_u8<const N: usize>(
        &mut self,
        cond: BoolTarget,
        a: &[U8Target; N],
        b: &[U8Target; N],
    ) -> [U8Target; N] {
        a.iter()
            .zip_eq(b.iter())
            .map(|(&a, &b)| self.select_u8(cond, a, b))
            .collect::<Vec<U8Target>>()
            .try_into()
            .unwrap()
    }

    /// This method is almost same with plonky2's version but uses `dummy_proof_and_constant_vk_no_generator` instead of
    /// `dummy_proof_and_vk` and exposes `dummy_proof_with_pis_target` to fill with dummy proof later while setting partial witness
    /// Main difference is moving `set_proof_with_pis_target` out of generator
    pub fn conditionally_verify_cyclic_proof_or_dummy<C: GenericConfig<D, F = F> + 'static>(
        &mut self,
        condition: BoolTarget,
        cyclic_proof_with_pis: &ProofWithPublicInputsTarget<D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Result<ProofWithPublicInputsTarget<D>>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        let (dummy_proof_with_pis_target, dummy_verifier_data_target) = self
            .builder
            .dummy_proof_and_constant_vk_no_generator::<C>(common_data)?;
        self.builder.conditionally_verify_cyclic_proof::<C>(
            condition,
            cyclic_proof_with_pis,
            &dummy_proof_with_pis_target,
            &dummy_verifier_data_target,
            common_data,
        )?;
        Ok(dummy_proof_with_pis_target)
    }

    #[must_use]
    pub fn select(&mut self, b: BoolTarget, x: Target, y: Target) -> Target {
        if let Some(const_b) = self.builder.target_as_constant(b.target) {
            if const_b.is_zero() {
                return y;
            } else {
                return x;
            }
        }

        let const_x = self.builder.target_as_constant(x);
        let const_y = self.builder.target_as_constant(y);

        if const_x.is_some() && const_y.is_some() && const_x.unwrap() == const_y.unwrap() {
            return x; // or y, they are equal
        }

        if let Some(const_x) = const_x
            && const_x.is_zero()
        {
            return self.select_if_zero(b, y);
        }
        if let Some(const_y) = const_y
            && const_y.is_zero()
        {
            return self.select_or_zero(b, x);
        }

        self.builder.select(b, x, y)
    }

    /// Returns x if b, 0 otherwise
    pub fn select_or_zero(&mut self, b: BoolTarget, x: Target) -> Target {
        self.builder.mul(b.target, x)
    }

    /// Returns 0 if b, x otherwise. (1-b)*x = x - x*b
    pub fn select_if_zero(&mut self, b: BoolTarget, x: Target) -> Target {
        self.builder.arithmetic(F::NEG_ONE, F::ONE, b.target, x, x)
    }

    pub fn select_constant(&mut self, b: BoolTarget, x: u64, y: u64) -> Target {
        let x_target = self.constant_u64(x);
        let y_target = self.constant_u64(y);
        self.select(b, x_target, y_target)
    }

    pub fn connect_bool(&mut self, a: BoolTarget, b: BoolTarget) {
        self.builder.connect(a.target, b.target);
    }

    pub fn register_public_hashout(&mut self, h: HashOutTarget) {
        self.builder.register_public_inputs(&h.elements);
    }

    pub fn connect_constant(&mut self, a: Target, c: u64) {
        let constant = self.constant_u64(c);
        self.builder.connect(a, constant);
    }

    #[must_use]
    pub fn le_sum_bytes(&mut self, bytes: &[U8Target]) -> Target {
        if bytes.is_empty() {
            return self.zero();
        }

        let mult = self.constant_u64(256);
        let mut acc = bytes[bytes.len() - 1].0;
        for &byte in bytes.iter().rev().skip(1) {
            acc = self.builder.mul_add(acc, mult, byte.0);
        }

        acc
    }
}

/// Same as [`plonky2::recursion::dummy_circuit::cyclic_base_proof`] but accepts `dummy_circuit` as parameter to not build every time
pub fn cyclic_base_proof<F, C, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    verifier_data: &VerifierOnlyCircuitData<C, D>,
    dummy_circuit: &CircuitData<F, C, D>,
    mut nonzero_public_inputs: HashMap<usize, F>,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: AlgebraicHasher<C::F>,
{
    let pis_len = common_data.num_public_inputs;
    let cap_elements = common_data.config.fri_config.num_cap_elements();
    let start_vk_pis = pis_len - 4 - 4 * cap_elements;

    // Add the cyclic verifier data public inputs.
    nonzero_public_inputs.extend((start_vk_pis..).zip(verifier_data.circuit_digest.elements));
    for i in 0..cap_elements {
        let start = start_vk_pis + 4 + 4 * i;
        nonzero_public_inputs
            .extend((start..).zip(verifier_data.constants_sigmas_cap.0[i].elements));
    }

    dummy_proof::<F, C, D>(dummy_circuit, nonzero_public_inputs)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use rand::Rng;

    use super::*;
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    #[test]
    fn test_select() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let x = builder.add_virtual_target();
        let y = builder.add_virtual_target();
        let b = builder.add_virtual_bool_target_safe();
        let zero = builder.zero();

        let should_zero = builder.select(b, zero, x);
        let should_x = builder.select(b, x, zero);
        let should_y = builder.select(b, y, x);

        builder.connect(should_zero, zero);
        builder.connect(should_x, x);
        builder.connect(should_y, y);

        let mut pw = PartialWitness::<F>::new();
        pw.set_target(x, F::from_canonical_u32(rand::thread_rng().r#gen::<u32>()))
            .unwrap();
        pw.set_target(y, F::from_canonical_u32(rand::thread_rng().r#gen::<u32>()))
            .unwrap();
        pw.set_target(b.target, F::from_canonical_u32(1)).unwrap();

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }
}
