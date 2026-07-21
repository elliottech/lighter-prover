// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashSet;

use anyhow::Result;
use itertools::Itertools;
use log::Level;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOutTarget, MerkleCapTarget};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use serde::Deserialize;
use serde_with::serde_as;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::blob::blob_constraints::BlobEvaluationTarget;
use crate::blob::constants::*;
use crate::blob::evaluate_bitstream::CircuitBuilderBitstreamEvaluator;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::byte::split::CircuitBuilderByteSplit;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::delta::types::AggregatedDeltaTarget;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::keccak::keccak::{CircuitBuilderKeccak, KeccakOutputTarget};
use crate::poseidon_bn128::plonky2_config::PoseidonBN128GoldilocksConfig;
use crate::poseidon2::Poseidon2Hash;
use crate::recursion::batch::{
    BATCH_TARGET_INDEX, BatchTarget, SEGMENT_INFO_INDEX, SegmentInfoTarget,
};
use crate::types::config::{Builder, C, CIRCUIT_CONFIG, D, F};
use crate::types::constants::*;
use crate::types::market_details::{PublicMarketDetailsTarget, connect_public_market_details};
use crate::uint::u8::{CircuitBuilderU8, U8Target, WitnessU8};
use crate::utils::CircuitBuilderUtils;

pub const NUM_CHAINS_PER_BATCH: usize = 8;

#[serde_as]
#[derive(Debug, Deserialize)]
pub struct WrapperInput {
    #[serde(rename = "kvh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub kzg_versioned_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "bc")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub batch_commitment: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "bd")]
    #[serde(deserialize_with = "deserializers::blob_bytes")]
    pub blob_bytes: Box<[u8; BLOB_DATA_BYTES_COUNT]>,

    // Keccak output but bls12-381 scalar field element
    #[serde(rename = "x")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub blob_polynomial_opening_x: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    // Keccak output but bls12-381 scalar field element
    #[serde(rename = "y")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub blob_polynomial_opening_y: [u8; KECCAK_HASH_OUT_BYTE_SIZE],
}

#[derive(Debug, Clone)]
pub struct WrapperInputTarget {
    pub chain_proofs: [ProofWithPublicInputsTarget<D>; NUM_CHAINS_PER_BATCH],
    pub chain_verifier: VerifierCircuitTarget,

    pub delta_chain_proof: ProofWithPublicInputsTarget<D>,
    pub delta_chain_verifier: VerifierCircuitTarget,

    pub blob_evaluation_proof: ProofWithPublicInputsTarget<D>,
    pub blob_evaluation_verifier: VerifierCircuitTarget,

    pub segment_count: Target,
    pub blob_bytes: Box<[U8Target; BLOB_DATA_BYTES_COUNT]>, // 0 byte at the beginning of each 32 byte limb is omitted
    pub kzg_versioned_hash: KeccakOutputTarget,
    pub blob_polynomial_opening_x: KeccakOutputTarget,
    pub blob_polynomial_opening_y: KeccakOutputTarget,
    pub batch_commitment: KeccakOutputTarget, // public
}

pub struct WrapperInnerCircuit {
    pub builder: Builder,
    pub target: WrapperInputTarget,
}

impl WrapperInnerCircuit {
    pub fn new(
        config: CircuitConfig,
        recursion_circuit: &CommonCircuitData<F, D>,
        recursion_verifier: &VerifierOnlyCircuitData<C, D>,
        delta_recursion_circuit: &CommonCircuitData<F, D>,
        delta_recursion_verifier: &VerifierOnlyCircuitData<C, D>,
        blob_evaluation_circuit: &CommonCircuitData<F, D>,
        blob_evaluation_verifier: &VerifierOnlyCircuitData<C, D>,
    ) -> Box<Self> {
        let mut builder = Builder::new(config);
        Box::new(Self {
            target: WrapperInputTarget {
                batch_commitment: builder.add_virtual_keccak_output_public_input_safe(), // Pub in

                chain_proofs: core::array::from_fn(|_| {
                    builder.add_virtual_proof_with_pis(recursion_circuit)
                }),
                chain_verifier: builder.constant_verifier_data(recursion_verifier),

                delta_chain_proof: builder.add_virtual_proof_with_pis(delta_recursion_circuit),
                delta_chain_verifier: builder.constant_verifier_data(delta_recursion_verifier),

                blob_evaluation_proof: builder.add_virtual_proof_with_pis(blob_evaluation_circuit),
                blob_evaluation_verifier: builder.constant_verifier_data(blob_evaluation_verifier),

                segment_count: builder.add_virtual_target(),
                blob_bytes: Box::new(core::array::from_fn(|_| {
                    builder.add_virtual_u8_target_safe()
                })),
                kzg_versioned_hash: builder.add_virtual_keccak_output_target_safe(),
                blob_polynomial_opening_x: builder.add_virtual_keccak_output_target_safe(),
                blob_polynomial_opening_y: builder.add_virtual_keccak_output_target_safe(),
            },
            builder,
        })
    }

    pub fn handle_segment_proofs(
        &mut self,
        recursion_circuit: &CommonCircuitData<F, D>,
    ) -> BatchTarget {
        // There must be at least one segment
        let segment_count_not_zero = self.builder.is_not_zero(self.target.segment_count);
        self.builder.assert_true(segment_count_not_zero);

        // Verify first segment
        self.builder.verify_proof::<C>(
            &self.target.chain_proofs[0],
            &self.target.chain_verifier,
            recursion_circuit,
        );

        let embedded_vk =
            Self::extract_vk_from_public_inputs(&self.target.chain_proofs[0].public_inputs);
        self.builder
            .connect_verifier_data(&embedded_vk, &self.target.chain_verifier);

        // First segment must be empty
        let first_segment = SegmentInfoTarget::from_public_inputs(
            &self.target.chain_proofs[0].public_inputs[BATCH_TARGET_INDEX..],
        );
        let first_segment_empty = first_segment.is_empty(&mut self.builder);
        self.builder.assert_true(first_segment_empty);

        // Get first batch
        let mut batch = BatchTarget::from_public_inputs(
            &self.target.chain_proofs[0].public_inputs[..BATCH_TARGET_INDEX],
        );
        let empty_account_delta_tree_root =
            self.builder.constant_hash(EMPTY_ACCOUNT_DELTA_TREE_ROOT);
        self.builder.connect_hashes(
            batch.old_account_delta_tree_root,
            empty_account_delta_tree_root,
        );

        let mut is_enabled = self.builder._true();

        for i in 1..NUM_CHAINS_PER_BATCH {
            // Flip is_enabled when segment_count == i. After that point, chain_proofs should be empty
            let it = self.builder.constant_usize(i);
            let is_segment_count_not_reached =
                self.builder.is_not_equal(self.target.segment_count, it);
            is_enabled = BoolTarget::new_unsafe(
                self.builder
                    .mul(is_enabled.target, is_segment_count_not_reached.target),
            );

            // Here we are using first segment proof as placeholder here, chain proofs after segment_count is redundant and not selected so they can be anything
            self.builder.conditionally_verify_proof::<C>(
                is_enabled,
                &self.target.chain_proofs[i],
                &self.target.chain_verifier,
                &self.target.chain_proofs[0],
                &self.target.chain_verifier,
                recursion_circuit,
            );

            // Conditionally bind the embedded VK to the expected chain verifier
            let embedded_vk =
                Self::extract_vk_from_public_inputs(&self.target.chain_proofs[i].public_inputs);
            self.builder.conditional_assert_eq_hash(
                is_enabled,
                &embedded_vk.circuit_digest,
                &self.target.chain_verifier.circuit_digest,
            );
            for (h0, h1) in embedded_vk
                .constants_sigmas_cap
                .0
                .iter()
                .zip(self.target.chain_verifier.constants_sigmas_cap.0.iter())
            {
                self.builder.conditional_assert_eq_hash(is_enabled, h0, h1);
            }

            let current_batch = BatchTarget::from_public_inputs(
                &self.target.chain_proofs[i].public_inputs[..BATCH_TARGET_INDEX],
            );
            let current_segment_info = SegmentInfoTarget::from_public_inputs(
                &self.target.chain_proofs[i].public_inputs[BATCH_TARGET_INDEX..],
            );

            self.builder.conditional_assert_eq_keccak_output(
                is_enabled,
                batch.on_chain_operations_pub_data_hash,
                current_segment_info.old_on_chain_operations_pub_data_hash,
            );

            batch = BatchTarget::conditionally_merge_consecutive(
                &mut self.builder,
                is_enabled,
                &batch,
                &current_batch,
            );
        }

        // Assert that either is_enabled is zero (means segment_count in [0,NUM_CHAINS_PER_BATCH)) or segment_count == NUM_CHAINS_PER_BATCH
        let num_of_chains_per_batch = self.builder.constant_usize(NUM_CHAINS_PER_BATCH);
        let segment_count_diff = self
            .builder
            .sub(self.target.segment_count, num_of_chains_per_batch);
        let is_enabled_times_diff = self.builder.mul(is_enabled.target, segment_count_diff);
        self.builder.assert_zero(is_enabled_times_diff);

        batch
    }

    /// Extract the VerifierCircuitTarget embedded at the tail of chain proof public inputs.
    /// Layout: [BatchTarget | SegmentInfoTarget | circuit_digest (4) | constants_sigmas_cap (4*16)]
    fn extract_vk_from_public_inputs(public_inputs: &[Target]) -> VerifierCircuitTarget {
        VerifierCircuitTarget {
            constants_sigmas_cap: MerkleCapTarget(
                (0..CIRCUIT_CONFIG.fri_config.num_cap_elements())
                    .map(|i| HashOutTarget {
                        elements: core::array::from_fn(|j| {
                            public_inputs[SEGMENT_INFO_INDEX + 4 + 4 * i + j]
                        }),
                    })
                    .collect(),
            ),
            circuit_digest: HashOutTarget {
                elements: core::array::from_fn(|i| public_inputs[SEGMENT_INFO_INDEX + i]),
            },
        }
    }

    pub fn verify_batch_commitment(&mut self, batch: &BatchTarget) {
        let mut elems = vec![];

        elems.extend_from_slice(&self.target.blob_polynomial_opening_x);
        elems.extend_from_slice(&self.target.blob_polynomial_opening_y);
        elems.extend_from_slice(&self.target.kzg_versioned_hash);

        let blob_commitment_hash = self.builder.keccak256_circuit(elems.clone());

        let mut elems = vec![];

        let mut block_number_bits = self.builder.split_bytes(batch.end_block_number, 8);
        block_number_bits.reverse();
        elems.extend_from_slice(&block_number_bits);

        let mut batch_size_bits = self.builder.split_bytes(batch.batch_size, 4);
        batch_size_bits.reverse();
        elems.extend_from_slice(&batch_size_bits);

        let mut first_date_bits = self.builder.split_bytes(batch.start_timestamp, 8);
        first_date_bits.reverse();
        elems.extend_from_slice(&first_date_bits);

        let mut last_date_bits = self.builder.split_bytes(batch.end_timestamp, 8);
        last_date_bits.reverse();
        elems.extend_from_slice(&last_date_bits);

        let old_state_root_bits = batch
            .old_state_root
            .elements
            .iter()
            .flat_map(|elem| self.builder.split_bytes(*elem, 8))
            .collect::<Vec<_>>();
        elems.extend_from_slice(&old_state_root_bits);

        let new_state_root_bits = batch
            .new_state_root
            .elements
            .iter()
            .flat_map(|elem| self.builder.split_bytes(*elem, 8))
            .collect::<Vec<_>>();
        elems.extend_from_slice(&new_state_root_bits);

        let new_validium_root_bits = batch
            .new_validium_root
            .elements
            .iter()
            .flat_map(|elem| self.builder.split_bytes(*elem, 8))
            .collect::<Vec<_>>();
        elems.extend_from_slice(&new_validium_root_bits);

        elems.extend_from_slice(&batch.on_chain_operations_pub_data_hash);

        let mut priority_operations_count_bits =
            self.builder.split_bytes(batch.priority_operations_count, 4);
        priority_operations_count_bits.reverse();
        elems.extend_from_slice(&priority_operations_count_bits);

        elems.extend_from_slice(&batch.new_prefix_priority_operation_hash);

        elems.extend_from_slice(&blob_commitment_hash);

        let batch_commitment = self.builder.keccak256_circuit(elems);

        self.builder
            .connect_keccak_output(batch_commitment, self.target.batch_commitment);
    }

    pub fn verify_version_and_reserved_data(&mut self) {
        let zero = self.builder.zero_u8();
        for i in BLOB_VERSION_INDEX..BLOB_MARK_PRICE_INDEX {
            self.builder.connect_u8(self.target.blob_bytes[i], zero);
        }
    }

    /// Verifies blob has correct market details in allocated slots.
    /// Each field is written to blob in big endian byte order
    pub fn verify_latest_market_data(
        &mut self,
        market_details: &[PublicMarketDetailsTarget; POSITION_LIST_SIZE],
    ) {
        let multiplier = self.builder.constant_usize(1 << 8);
        for i in (BLOB_MARK_PRICE_INDEX..BLOB_FUNDING_INDEX).step_by(MARK_PRICE_BYTE_SIZE) {
            let chunk: [U8Target; MARK_PRICE_BYTE_SIZE] = self.target.blob_bytes
                [i..i + MARK_PRICE_BYTE_SIZE]
                .try_into()
                .unwrap();

            let idx = (i - BLOB_MARK_PRICE_INDEX) / MARK_PRICE_BYTE_SIZE;

            let mut res = chunk[0].0;
            for j in 1..MARK_PRICE_BYTE_SIZE {
                res = self.builder.mul_add(res, multiplier, chunk[j].0);
            }
            self.builder.connect(market_details[idx].mark_price, res);
        }

        for i in (BLOB_FUNDING_INDEX..BLOB_QUOTE_MULTIPLIER_INDEX).step_by(FUNDING_BYTE_SIZE) {
            let chunk: [U8Target; FUNDING_BYTE_SIZE] = self.target.blob_bytes
                [i..i + FUNDING_BYTE_SIZE]
                .try_into()
                .unwrap();

            let idx = (i - BLOB_FUNDING_INDEX) / FUNDING_BYTE_SIZE;

            let sign = chunk[0];
            let is_negative = self
                .builder
                .is_sign_negative(market_details[idx].funding_rate_prefix_sum.sign);
            self.builder.connect(is_negative.target, sign.0);

            // First limb (4 bytes)
            let mut res = chunk[1].0;
            for j in 2..5 {
                res = self.builder.mul_add(res, multiplier, chunk[j].0);
            }
            self.builder.connect(
                market_details[idx].funding_rate_prefix_sum.abs.limbs[1].0,
                res,
            );

            // Second limb (4 bytes)
            let mut res2 = chunk[5].0;
            for j in 6..FUNDING_BYTE_SIZE {
                res2 = self.builder.mul_add(res2, multiplier, chunk[j].0);
            }
            self.builder.connect(
                market_details[idx].funding_rate_prefix_sum.abs.limbs[0].0,
                res2,
            );
        }

        for i in
            (BLOB_QUOTE_MULTIPLIER_INDEX..BLOB_ACCOUNT_OFFSET).step_by(QUOTE_MULTIPLIER_BYTE_SIZE)
        {
            let chunk: [U8Target; QUOTE_MULTIPLIER_BYTE_SIZE] = self.target.blob_bytes
                [i..i + QUOTE_MULTIPLIER_BYTE_SIZE]
                .try_into()
                .unwrap();

            let idx = (i - BLOB_QUOTE_MULTIPLIER_INDEX) / QUOTE_MULTIPLIER_BYTE_SIZE;

            let mut res = chunk[0].0;
            for j in 1..QUOTE_MULTIPLIER_BYTE_SIZE {
                res = self.builder.mul_add(res, multiplier, chunk[j].0);
            }
            self.builder
                .connect(market_details[idx].quote_multiplier, res);
        }
    }

    /// Commitment to the part of blob where we write the delta tree leaves.
    /// Each field is written to blob in big endian byte order
    fn _get_blob_pub_data_hash(&mut self) -> HashOutTarget {
        let zero_u8 = self.builder.zero_u8();
        let multiplier = self.builder.constant_usize(1 << 8);
        let mut pub_data_hash_elements = vec![];
        let blob_bytes = &self.target.blob_bytes;
        for chunk in blob_bytes[BLOB_ACCOUNT_OFFSET..].chunks(7) {
            let mut res = chunk.first().unwrap_or(&zero_u8).0;
            for &byte in chunk.iter().skip(1) {
                res = self.builder.mul_add(res, multiplier, byte.0);
            }
            pub_data_hash_elements.push(res);
        }
        self.builder
            .hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_hash_elements)
    }

    pub fn handle_delta_chain_proof(
        &mut self,
        delta_recursion_circuit: &CommonCircuitData<F, D>,
    ) -> AggregatedDeltaTarget {
        self.builder.verify_proof::<C>(
            &self.target.delta_chain_proof,
            &self.target.delta_chain_verifier,
            delta_recursion_circuit,
        );

        AggregatedDeltaTarget::from_public_inputs(
            &self.target.delta_chain_proof.public_inputs[..AggregatedDeltaTarget::END_INDEX],
        )
    }

    pub fn handle_blob_evaluation_proof(
        &mut self,
        batch: &BatchTarget,
        blob_evaluation_circuit: &CommonCircuitData<F, D>,
    ) {
        self.builder.verify_proof::<C>(
            &self.target.blob_evaluation_proof,
            &self.target.blob_evaluation_verifier,
            blob_evaluation_circuit,
        );

        let blob = BlobEvaluationTarget::from_public_inputs(
            &self.target.blob_evaluation_proof.public_inputs.clone(),
        );

        self.builder.connect_hashes(
            batch.new_account_delta_tree_root,
            blob.account_delta_tree_root,
        );

        self.builder
            .connect_keccak_output(self.target.kzg_versioned_hash, blob.kzg_versioned_hash);
        self.builder.connect_keccak_output(
            self.target.blob_polynomial_opening_x,
            blob.blob_polynomial_opening_x,
        );
        self.builder.connect_keccak_output(
            self.target.blob_polynomial_opening_y,
            blob.blob_polynomial_opening_y,
        );

        self.target
            .blob_bytes
            .iter()
            .zip_eq(blob.blob_bytes.iter())
            .for_each(|(a, b)| {
                self.builder.connect_u8(*a, *b);
            });

        connect_public_market_details(
            &mut self.builder,
            &batch.new_public_market_details,
            &blob.public_market_details,
        );
    }

    pub fn verify_aggregated_delta(
        &mut self,
        batch: &BatchTarget,
        aggregated_delta: &AggregatedDeltaTarget,
    ) {
        // Verify that delta layer computed the same root as the tx layer
        let account_delta_tree_root = aggregated_delta.get_root(&mut self.builder);
        self.builder
            .connect_hashes(batch.new_account_delta_tree_root, account_delta_tree_root);

        // Verify the evaluation point
        let pub_data_hash = self._get_blob_pub_data_hash();
        let pub_data_evaluation_point = self
            .builder
            .hash_two_to_one(&pub_data_hash, &account_delta_tree_root);
        let zero = self.builder.zero();
        self.builder.connect_quintic_ext(
            aggregated_delta.evaluation_point,
            QuinticExtensionTarget([
                pub_data_evaluation_point.elements[0],
                pub_data_evaluation_point.elements[1],
                pub_data_evaluation_point.elements[2],
                pub_data_evaluation_point.elements[3],
                zero,
            ]),
        );
    }

    /// Performs the same polynomial evaluation as in delta circuit using the bytes of the blob.
    /// Works with the blob's part where account delta leaves are written in a compressed format,
    /// where each first half-byte declares the size of the following data in half-bytes.
    pub fn verify_delta_polynomial_evaluation(&mut self, aggregated_delta: &AggregatedDeltaTarget) {
        self.builder
            .bitstream_initialize(0, aggregated_delta.evaluation_point);

        let pub_data_half_bytes = self.target.blob_bytes[BLOB_ACCOUNT_OFFSET..]
            .iter()
            .flat_map(|byte| self.builder.split_to_u4s_le(byte.0, 2))
            .collect::<Vec<_>>();

        for half_byte in pub_data_half_bytes.iter() {
            self.builder.bitstream_digest_target(0, *half_byte);
        }
        let state = self.builder.bitstream_export(0);

        self.builder
            .assert_lte(aggregated_delta.degree, state.degree, 18);
        let degree_difference = self.builder.sub(state.degree, aggregated_delta.degree);
        let mul_by =
            self.builder
                .exp_quintic_ext(aggregated_delta.evaluation_point, degree_difference, 18);
        let adjusted_aggregated_evaluation = self
            .builder
            .mul_quintic_ext(aggregated_delta.evaluation, mul_by);

        self.builder
            .connect_quintic_ext(adjusted_aggregated_evaluation, state.sum);
    }
}

pub struct WrapperOutputTarget {
    pub inner_wrapper_proof: ProofWithPublicInputsTarget<D>,
    pub inner_wrapper_verifier: VerifierCircuitTarget,

    pub batch_commitment: KeccakOutputTarget, // public
}

pub struct WrapperOuterCircuit {
    pub builder: Builder,
    pub target: WrapperOutputTarget,
}

impl WrapperOuterCircuit {
    pub fn new(
        config: CircuitConfig,
        inner_circuit: &CommonCircuitData<F, D>,
        inner_verifier: &VerifierOnlyCircuitData<C, D>,
    ) -> Self {
        let mut builder = Builder::new(config);

        let inner_proof = builder.add_virtual_proof_with_pis(inner_circuit);

        Self {
            target: WrapperOutputTarget {
                inner_wrapper_proof: inner_proof.clone(),
                inner_wrapper_verifier: builder.constant_verifier_data(inner_verifier),
                batch_commitment: core::array::from_fn(|i| U8Target(inner_proof.public_inputs[i])),
            },
            builder,
        }
    }
}

pub struct WrapperCircuit;
impl WrapperCircuit {
    pub fn define_inner(
        config: CircuitConfig,
        recursion_circuit: &CommonCircuitData<F, D>,
        recursion_verifier: &VerifierOnlyCircuitData<C, D>,
        delta_recursion_circuit: &CommonCircuitData<F, D>,
        delta_recursion_verifier: &VerifierOnlyCircuitData<C, D>,
        blob_evaluation_circuit: &CommonCircuitData<F, D>,
        blob_evaluation_verifier: &VerifierOnlyCircuitData<C, D>,
    ) -> Box<WrapperInnerCircuit> {
        let mut circuit = WrapperInnerCircuit::new(
            config,
            recursion_circuit,
            recursion_verifier,
            delta_recursion_circuit,
            delta_recursion_verifier,
            blob_evaluation_circuit,
            blob_evaluation_verifier,
        );

        let batch = circuit.handle_segment_proofs(recursion_circuit);

        circuit.verify_batch_commitment(&batch);

        circuit.verify_version_and_reserved_data();

        circuit.verify_latest_market_data(&batch.new_public_market_details);

        let aggregated_delta = circuit.handle_delta_chain_proof(delta_recursion_circuit);

        circuit.verify_aggregated_delta(&batch, &aggregated_delta);

        circuit.handle_blob_evaluation_proof(&batch, blob_evaluation_circuit);

        circuit.verify_delta_polynomial_evaluation(&aggregated_delta);

        circuit
            .builder
            .perform_registered_range_checks_with_custom_range_check_sizes(&HashSet::from([
                4, 16, 32, 48,
            ]));

        circuit
    }

    pub fn define_outer(
        config: CircuitConfig,
        inner_circuit: &CommonCircuitData<F, D>,
        inner_verifier: &VerifierOnlyCircuitData<C, D>,
    ) -> WrapperOuterCircuit {
        let mut circuit = WrapperOuterCircuit::new(config, inner_circuit, inner_verifier);

        circuit
            .builder
            .register_public_u8_inputs(&circuit.target.batch_commitment);

        circuit.builder.verify_proof::<C>(
            &circuit.target.inner_wrapper_proof,
            &circuit.target.inner_wrapper_verifier,
            inner_circuit,
        );

        circuit
    }

    pub fn generate_witness_inner(
        circuit_target: Box<WrapperInputTarget>,
        #[allow(clippy::boxed_local)] info: Box<WrapperInput>,
        chain_proofs: &[ProofWithPublicInputs<F, C, D>],
        segment_count: u64,
        delta_chain_proof: ProofWithPublicInputs<F, C, D>,
        blob_evaluation_proof: ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        assert_eq!(
            chain_proofs.len(),
            NUM_CHAINS_PER_BATCH,
            "wrong number of chain proofs given"
        );
        for i in 0..NUM_CHAINS_PER_BATCH {
            pw.set_proof_with_pis_target(&circuit_target.chain_proofs[i], &chain_proofs[i])?;
        }

        pw.set_target(
            circuit_target.segment_count,
            F::from_canonical_u64(segment_count),
        )?;

        pw.set_proof_with_pis_target(&circuit_target.delta_chain_proof, &delta_chain_proof)?;

        pw.set_proof_with_pis_target(
            &circuit_target.blob_evaluation_proof,
            &blob_evaluation_proof,
        )?;

        for i in 0..KECCAK_HASH_OUT_BYTE_SIZE {
            pw.set_target(
                circuit_target.kzg_versioned_hash[i].0,
                F::from_canonical_u8(info.kzg_versioned_hash[i]),
            )?;
            pw.set_target(
                circuit_target.batch_commitment[i].0,
                F::from_canonical_u8(info.batch_commitment[i]),
            )?;
            pw.set_target(
                circuit_target.blob_polynomial_opening_x[i].0,
                F::from_canonical_u8(info.blob_polynomial_opening_x[i]),
            )?;
            pw.set_target(
                circuit_target.blob_polynomial_opening_y[i].0,
                F::from_canonical_u8(info.blob_polynomial_opening_y[i]),
            )?;
        }

        for i in 0..BLOB_DATA_BYTES_COUNT {
            pw.set_u8_target(circuit_target.blob_bytes[i], info.blob_bytes[i])?;
        }

        Ok(pw)
    }

    pub fn generate_witness_outer(
        target: &WrapperOutputTarget,
        inner_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.inner_wrapper_proof, inner_proof)?;

        Ok(pw)
    }

    pub fn prove_inner(
        circuit: &CircuitData<F, C, D>,
        target: Box<WrapperInputTarget>,
        info: Box<WrapperInput>,
        chain_proofs: &[ProofWithPublicInputs<F, C, D>],
        segment_count: u64,
        delta_chain_proof: ProofWithPublicInputs<F, C, D>,
        blob_evaluation_proof: ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("WrapperCircuit::prove_inner", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness_inner(
                target,
                info,
                chain_proofs,
                segment_count,
                delta_chain_proof,
                blob_evaluation_proof,
            )?
        });

        let proof = prove::<F, C, D>(&circuit.prover_only, &circuit.common, pw, &mut timing)?;
        timed!(timing, "verify", { circuit.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    pub fn prove_outer(
        circuit: &CircuitData<F, PoseidonBN128GoldilocksConfig, D>,
        target: &WrapperOutputTarget,
        inner_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, PoseidonBN128GoldilocksConfig, D>> {
        let mut timing = TimingTree::new("WrapperCircuit::prove_outer", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness_outer(target, inner_proof)?
        });

        let proof = prove::<F, PoseidonBN128GoldilocksConfig, D>(
            &circuit.prover_only,
            &circuit.common,
            pw,
            &mut timing,
        )?;
        timed!(timing, "verify", { circuit.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }
}
