// Portions of this file are derived from plonky2
// Copyright (c) 2022-2025 The Plonky2 Authors
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use core::borrow::Borrow;

use plonky2::field::extension::Extendable;
use plonky2::field::extension::algebra::ExtensionAlgebra;
use plonky2::gates::gate::{Gate, GateRef};
use plonky2::hash::hash_types::{HashOut, HashOutTarget, MerkleCapTarget, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use plonky2::iop::generator::{SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_data::{
    CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, Hasher};
use plonky2::plonk::proof::ProofWithPublicInputsTarget;

use super::types::Builder;

// Core circuit-builder functions
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn add_lookup_rows(
        &mut self,
        last_lu_gate: usize,
        last_lut_gate: usize,
        first_lut_gate: usize,
    ) {
        self.builder
            .add_lookup_rows(last_lu_gate, last_lut_gate, first_lut_gate)
    }

    pub fn update_lookups(&mut self, looking_in: Target, looking_out: Target, lut_index: usize) {
        self.builder
            .update_lookups(looking_in, looking_out, lut_index)
    }

    pub fn num_luts(&self) -> usize {
        self.builder.num_luts()
    }

    pub fn get_lut_lookups(&self, lut_index: usize) -> &[(Target, Target)] {
        self.builder.get_lut_lookups(lut_index)
    }

    pub fn add_gate<G: Gate<F, D>>(&mut self, gate_type: G, constants: Vec<F>) -> usize {
        self.builder.add_gate(gate_type, constants)
    }

    pub fn add_gate_to_gate_set(&mut self, gate: GateRef<F, D>) {
        self.builder.add_gate_to_gate_set(gate)
    }

    pub fn generate_copy(&mut self, src: Target, dst: Target) {
        self.builder.generate_copy(src, dst)
    }

    #[track_caller]
    pub fn assert_zero(&mut self, x: Target) {
        self.builder.assert_zero(x)
    }

    #[track_caller]
    pub fn assert_one(&mut self, x: Target) {
        self.builder.assert_one(x)
    }

    pub fn add_generators(&mut self, generators: Vec<WitnessGeneratorRef<F, D>>) {
        self.builder.add_generators(generators)
    }

    pub fn add_simple_generator<G: SimpleGenerator<F, D>>(&mut self, generator: G) {
        self.builder.add_simple_generator(generator)
    }

    pub fn zero(&mut self) -> Target {
        self.builder.zero()
    }

    pub fn one(&mut self) -> Target {
        self.builder.one()
    }

    pub fn two(&mut self) -> Target {
        self.builder.two()
    }

    pub fn neg_one(&mut self) -> Target {
        self.builder.neg_one()
    }

    pub fn _false(&mut self) -> BoolTarget {
        self.builder._false()
    }

    pub fn _true(&mut self) -> BoolTarget {
        self.builder._true()
    }

    pub fn constant(&mut self, c: F) -> Target {
        self.builder.constant(c)
    }

    pub fn constants(&mut self, constants: &[F]) -> Vec<Target> {
        self.builder.constants(constants)
    }

    pub fn constant_bool(&mut self, b: bool) -> BoolTarget {
        self.builder.constant_bool(b)
    }

    pub fn constant_bools(&mut self, b: &[bool]) -> Vec<BoolTarget> {
        b.iter().map(|&b| self.constant_bool(b)).collect()
    }

    pub fn target_as_constant(&self, target: Target) -> Option<F> {
        self.builder.target_as_constant(target)
    }

    pub fn target_as_constant_ext(&self, target: ExtensionTarget<D>) -> Option<F::Extension> {
        self.builder.target_as_constant_ext(target)
    }

    pub fn find_slot<G: Gate<F, D> + Clone>(
        &mut self,
        gate: G,
        params: &[F],
        constants: &[F],
    ) -> (usize, usize) {
        self.builder.find_slot(gate, params, constants)
    }

    pub fn constant_hash(&mut self, h: HashOut<F>) -> HashOutTarget {
        self.builder.constant_hash(h)
    }

    pub fn constant_merkle_cap<H: Hasher<F, Hash = HashOut<F>>>(
        &mut self,
        cap: &MerkleCap<F, H>,
    ) -> MerkleCapTarget {
        self.builder.constant_merkle_cap(cap)
    }

    pub fn constant_verifier_data<C: GenericConfig<D, F = F>>(
        &mut self,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
    ) -> VerifierCircuitTarget
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        self.builder.constant_verifier_data(verifier_data)
    }

    #[must_use]
    pub fn select_ext(
        &mut self,
        b: BoolTarget,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.select_ext(b, x, y)
    }

    #[must_use]
    pub fn select_ext_generalized(
        &mut self,
        b: ExtensionTarget<D>,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.select_ext_generalized(b, x, y)
    }
}

// Arithmetics
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn neg(&mut self, x: Target) -> Target {
        self.builder.neg(x)
    }

    pub fn square(&mut self, x: Target) -> Target {
        self.builder.square(x)
    }

    pub fn cube(&mut self, x: Target) -> Target {
        self.builder.cube(x)
    }

    pub fn arithmetic(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: Target,
        multiplicand_1: Target,
        addend: Target,
    ) -> Target {
        self.builder
            .arithmetic(const_0, const_1, multiplicand_0, multiplicand_1, addend)
    }

    /// Computes `x * y + z`.
    pub fn mul_add(&mut self, x: Target, y: Target, z: Target) -> Target {
        self.builder.mul_add(x, y, z)
    }

    pub fn add_const(&mut self, x: Target, c: F) -> Target {
        self.builder.add_const(x, c)
    }

    pub fn mul_const(&mut self, c: F, x: Target) -> Target {
        self.builder.mul_const(c, x)
    }

    /// Computes `C * x + y`.
    pub fn mul_const_add(&mut self, c: F, x: Target, y: Target) -> Target {
        self.builder.mul_const_add(c, x, y)
    }

    pub fn mul_sub(&mut self, x: Target, y: Target, z: Target) -> Target {
        self.builder.mul_sub(x, y, z)
    }

    pub fn add_one(&mut self, x: Target) -> Target {
        let one = self.one();
        self.add(x, one)
    }

    pub fn add(&mut self, x: Target, y: Target) -> Target {
        self.builder.add(x, y)
    }

    pub fn add_many<T>(&mut self, terms: impl IntoIterator<Item = T>) -> Target
    where
        T: Borrow<Target>,
    {
        self.builder.add_many(terms)
    }

    pub fn sub(&mut self, x: Target, y: Target) -> Target {
        self.builder.sub(x, y)
    }

    pub fn mul(&mut self, x: Target, y: Target) -> Target {
        self.builder.mul(x, y)
    }

    pub fn mul_many<T>(&mut self, terms: impl IntoIterator<Item = T>) -> Target
    where
        T: Borrow<Target>,
    {
        self.builder.mul_many(terms)
    }

    pub fn exp_power_of_2(&mut self, base: Target, power_log: usize) -> Target {
        self.builder.exp_power_of_2(base, power_log)
    }

    pub fn exp_from_bits(
        &mut self,
        base: Target,
        exponent_bits: impl IntoIterator<Item = impl Borrow<BoolTarget>>,
    ) -> Target {
        self.builder.exp_from_bits(base, exponent_bits)
    }

    pub fn exp(&mut self, base: Target, exponent: Target, num_bits: usize) -> Target {
        self.builder.exp(base, exponent, num_bits)
    }

    pub fn exp_from_bits_const_base(
        &mut self,
        base: F,
        exponent_bits: impl IntoIterator<Item = impl Borrow<BoolTarget>>,
    ) -> Target {
        self.builder.exp_from_bits_const_base(base, exponent_bits)
    }

    pub fn exp_u64(&mut self, base: Target, exponent: u64) -> Target {
        self.builder.exp_u64(base, exponent)
    }

    pub fn div(&mut self, x: Target, y: Target) -> Target {
        self.builder.div(x, y)
    }

    pub fn inverse(&mut self, x: Target) -> Target {
        self.builder.inverse(x)
    }

    pub fn not(&mut self, b: BoolTarget) -> BoolTarget {
        self.builder.not(b)
    }

    pub fn and(&mut self, b1: BoolTarget, b2: BoolTarget) -> BoolTarget {
        self.builder.and(b1, b2)
    }

    pub fn or(&mut self, b1: BoolTarget, b2: BoolTarget) -> BoolTarget {
        self.builder.or(b1, b2)
    }

    pub fn _if(&mut self, b: BoolTarget, x: Target, y: Target) -> Target {
        self.builder._if(b, x, y)
    }
}

// Range-checking functions
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn le_sum(&mut self, bits: impl Iterator<Item = impl Borrow<BoolTarget>>) -> Target {
        self.builder.le_sum(bits)
    }

    pub fn assert_bool(&mut self, b: BoolTarget) {
        self.builder.assert_bool(b)
    }
}

// arithmetic_extension.rs
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn arithmetic_extension(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
        addend: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder
            .arithmetic_extension(const_0, const_1, multiplicand_0, multiplicand_1, addend)
    }

    pub fn wide_arithmetic_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
        d: ExtensionTarget<D>,
        e: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.wide_arithmetic_extension(a, b, c, d, e)
    }

    pub fn inner_product_extension(
        &mut self,
        constant: F,
        starting_acc: ExtensionTarget<D>,
        pairs: Vec<(ExtensionTarget<D>, ExtensionTarget<D>)>,
    ) -> ExtensionTarget<D> {
        self.builder
            .inner_product_extension(constant, starting_acc, pairs)
    }

    pub fn add_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.add_extension(a, b)
    }

    pub fn add_ext_algebra(
        &mut self,
        a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.add_ext_algebra(a, b)
    }

    pub fn add_many_extension<T>(
        &mut self,
        terms: impl IntoIterator<Item = T>,
    ) -> ExtensionTarget<D>
    where
        T: Borrow<ExtensionTarget<D>>,
    {
        self.builder.add_many_extension(terms)
    }

    pub fn sub_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.sub_extension(a, b)
    }

    pub fn sub_ext_algebra(
        &mut self,
        a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.sub_ext_algebra(a, b)
    }

    pub fn mul_extension_with_const(
        &mut self,
        const_0: F,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder
            .mul_extension_with_const(const_0, multiplicand_0, multiplicand_1)
    }

    pub fn mul_extension(
        &mut self,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.mul_extension(multiplicand_0, multiplicand_1)
    }

    pub fn square_extension(&mut self, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.square_extension(x)
    }

    pub fn cube_extension(&mut self, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.cube_extension(x)
    }

    pub fn mul_add_ext_algebra(
        &mut self,
        a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
        c: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.mul_add_ext_algebra(a, b, c)
    }

    pub fn mul_ext_algebra(
        &mut self,
        a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.mul_ext_algebra(a, b)
    }

    pub fn mul_many_extension<T>(
        &mut self,
        terms: impl IntoIterator<Item = T>,
    ) -> ExtensionTarget<D>
    where
        T: Borrow<ExtensionTarget<D>>,
    {
        self.builder.mul_many_extension(terms)
    }

    pub fn mul_add_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.mul_add_extension(a, b, c)
    }

    pub fn add_const_extension(&mut self, x: ExtensionTarget<D>, c: F) -> ExtensionTarget<D> {
        self.builder.add_const_extension(x, c)
    }

    pub fn mul_const_extension(&mut self, c: F, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.mul_const_extension(c, x)
    }

    pub fn mul_const_add_extension(
        &mut self,
        c: F,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.mul_const_add_extension(c, x, y)
    }

    pub fn scalar_mul_add_extension(
        &mut self,
        a: Target,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.scalar_mul_add_extension(a, b, c)
    }

    pub fn mul_sub_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.mul_sub_extension(a, b, c)
    }

    pub fn scalar_mul_sub_extension(
        &mut self,
        a: Target,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.scalar_mul_sub_extension(a, b, c)
    }

    pub fn scalar_mul_ext(&mut self, a: Target, b: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.scalar_mul_ext(a, b)
    }

    pub fn scalar_mul_add_ext_algebra(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionAlgebraTarget<D>,
        c: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.scalar_mul_add_ext_algebra(a, b, c)
    }

    pub fn scalar_mul_ext_algebra(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.scalar_mul_ext_algebra(a, b)
    }

    pub fn exp_power_of_2_extension(
        &mut self,
        base: ExtensionTarget<D>,
        power_log: usize,
    ) -> ExtensionTarget<D> {
        self.builder.exp_power_of_2_extension(base, power_log)
    }

    pub fn exp_u64_extension(
        &mut self,
        base: ExtensionTarget<D>,
        exponent: u64,
    ) -> ExtensionTarget<D> {
        self.builder.exp_u64_extension(base, exponent)
    }

    pub fn div_extension(
        &mut self,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.div_extension(x, y)
    }

    pub fn div_add_extension(
        &mut self,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
        z: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.builder.div_add_extension(x, y, z)
    }

    pub fn inverse_extension(&mut self, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.inverse_extension(x)
    }
}

// Hashing
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn hash_or_noop<H: AlgebraicHasher<F>>(&mut self, inputs: Vec<Target>) -> HashOutTarget {
        self.builder.hash_or_noop::<H>(inputs)
    }

    pub fn hash_n_to_hash_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
    ) -> HashOutTarget {
        self.builder.hash_n_to_hash_no_pad::<H>(inputs)
    }

    pub fn hash_n_to_m_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
        num_outputs: usize,
    ) -> Vec<Target> {
        self.builder.hash_n_to_m_no_pad::<H>(inputs, num_outputs)
    }
}

// Random access
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn random_access(&mut self, access_index: Target, v: Vec<Target>) -> Target {
        if let Some(&cached) = self.random_access_cache.get(&(access_index, v.clone())) {
            return cached;
        }
        let result = self.builder.random_access(access_index, v.clone());
        self.random_access_cache.insert((access_index, v), result);
        result
    }
}

// Recursion
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn verify_proof<C: GenericConfig<D, F = F>>(
        &mut self,
        proof_with_pis: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        self.builder
            .verify_proof::<C>(proof_with_pis, inner_verifier_data, inner_common_data);
    }

    pub fn conditionally_verify_proof<C: GenericConfig<D, F = F>>(
        &mut self,
        condition: BoolTarget,
        proof_with_pis0: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data0: &VerifierCircuitTarget,
        proof_with_pis1: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data1: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        self.builder.conditionally_verify_proof::<C>(
            condition,
            proof_with_pis0,
            inner_verifier_data0,
            proof_with_pis1,
            inner_verifier_data1,
            inner_common_data,
        );
    }
}

// Add Targets
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    #[must_use]
    pub fn add_virtual_targets(&mut self, n: usize) -> Vec<Target> {
        self.builder.add_virtual_targets(n)
    }

    #[must_use]
    pub fn add_virtual_target_arr<const N: usize>(&mut self) -> [Target; N] {
        self.builder.add_virtual_target_arr::<N>()
    }

    #[must_use]
    pub fn add_virtual_hash(&mut self) -> HashOutTarget {
        self.builder.add_virtual_hash()
    }

    #[must_use]
    pub fn add_virtual_cap(&mut self, cap_height: usize) -> MerkleCapTarget {
        self.builder.add_virtual_cap(cap_height)
    }

    #[must_use]
    pub fn add_virtual_hashes(&mut self, n: usize) -> Vec<HashOutTarget> {
        self.builder.add_virtual_hashes(n)
    }

    #[must_use]
    pub fn add_virtual_extension_target(&mut self) -> ExtensionTarget<D> {
        self.builder.add_virtual_extension_target()
    }

    #[must_use]
    pub fn add_virtual_extension_targets(&mut self, n: usize) -> Vec<ExtensionTarget<D>> {
        self.builder.add_virtual_extension_targets(n)
    }

    #[must_use]
    pub fn add_virtual_bool_target_unsafe(&mut self) -> BoolTarget {
        self.builder.add_virtual_bool_target_unsafe()
    }

    #[must_use]
    pub fn add_virtual_bool_target_safe(&mut self) -> BoolTarget {
        self.builder.add_virtual_bool_target_safe()
    }

    #[must_use]
    pub fn add_virtual_verifier_data(&mut self, cap_height: usize) -> VerifierCircuitTarget {
        self.builder.add_virtual_verifier_data(cap_height)
    }

    #[must_use]
    pub fn add_virtual_target(&mut self) -> Target {
        self.builder.add_virtual_target()
    }

    pub fn constant_extension(&mut self, c: F::Extension) -> ExtensionTarget<D> {
        self.builder.constant_extension(c)
    }

    pub fn constant_ext_algebra(
        &mut self,
        c: ExtensionAlgebra<F::Extension, D>,
    ) -> ExtensionAlgebraTarget<D> {
        self.builder.constant_ext_algebra(c)
    }

    pub fn zero_extension(&mut self) -> ExtensionTarget<D> {
        self.builder.zero_extension()
    }

    pub fn one_extension(&mut self) -> ExtensionTarget<D> {
        self.builder.one_extension()
    }

    pub fn two_extension(&mut self) -> ExtensionTarget<D> {
        self.builder.two_extension()
    }

    pub fn neg_one_extension(&mut self) -> ExtensionTarget<D> {
        self.builder.neg_one_extension()
    }

    pub fn zero_ext_algebra(&mut self) -> ExtensionAlgebraTarget<D> {
        self.builder.zero_ext_algebra()
    }

    pub fn convert_to_ext(&mut self, t: Target) -> ExtensionTarget<D> {
        self.builder.convert_to_ext(t)
    }

    pub fn convert_to_ext_algebra(&mut self, et: ExtensionTarget<D>) -> ExtensionAlgebraTarget<D> {
        self.builder.convert_to_ext_algebra(et)
    }
}

// Add Public inputs
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn register_public_input(&mut self, target: Target) {
        self.builder.register_public_input(target)
    }

    pub fn register_public_inputs(&mut self, targets: &[Target]) {
        self.builder.register_public_inputs(targets)
    }

    pub fn num_public_inputs(&self) -> usize {
        self.builder.num_public_inputs()
    }

    #[must_use]
    pub fn add_virtual_hash_public_input(&mut self) -> HashOutTarget {
        self.builder.add_virtual_hash_public_input()
    }

    #[must_use]
    pub fn add_virtual_hashes_public_input(&mut self, n: usize) -> Vec<HashOutTarget> {
        self.builder.add_virtual_hashes_public_input(n)
    }

    #[must_use]
    pub fn add_virtual_public_input(&mut self) -> Target {
        self.builder.add_virtual_public_input()
    }

    #[must_use]
    pub fn add_virtual_public_input_arr<const N: usize>(&mut self) -> [Target; N] {
        self.builder.add_virtual_public_input_arr::<N>()
    }

    #[must_use]
    pub fn add_verifier_data_public_inputs(&mut self) -> VerifierCircuitTarget {
        self.builder.add_verifier_data_public_inputs()
    }

    #[must_use]
    pub fn add_virtual_proof_with_pis(
        &mut self,
        common_data: &CommonCircuitData<F, D>,
    ) -> ProofWithPublicInputsTarget<D> {
        self.builder.add_virtual_proof_with_pis(common_data)
    }
}

// Connect targets
impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    #[track_caller]
    pub fn connect(&mut self, x: Target, y: Target) {
        self.builder.connect(x, y)
    }

    pub fn connect_extension(&mut self, src: ExtensionTarget<D>, dst: ExtensionTarget<D>) {
        self.builder.connect_extension(src, dst)
    }

    pub fn connect_hashes(&mut self, x: HashOutTarget, y: HashOutTarget) {
        self.builder.connect_hashes(x, y)
    }

    pub fn connect_merkle_caps(&mut self, x: &MerkleCapTarget, y: &MerkleCapTarget) {
        self.builder.connect_merkle_caps(x, y)
    }

    pub fn connect_verifier_data(&mut self, x: &VerifierCircuitTarget, y: &VerifierCircuitTarget) {
        self.builder.connect_verifier_data(x, y)
    }
}
