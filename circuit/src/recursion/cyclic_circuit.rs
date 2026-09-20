// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use hashbrown::HashMap;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::equality_base::EqualityGate;
use plonky2::gates::exponentiation::ExponentiationGate;
use plonky2::gates::select_base::SelectionGate;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget,
};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::recursion::dummy_circuit::cyclic_base_proof;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use super::batch::{Batch, BatchTarget, SegmentInfo, SegmentInfoTarget, SegmentInfoTargetWitness};
use crate::byte::split_gate::ByteDecompositionGate;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::keccak::keccak::{CircuitBuilderKeccak, KeccakOutputTarget};
use crate::poseidon2::Poseidon2Gate;
use crate::recursion::batch::{BATCH_TARGET_INDEX, BatchTargetWitness, SEGMENT_INFO_INDEX};
use crate::recursion::block_witness::BlockWitnessTarget;
use crate::types::config::{Builder, C, CIRCUIT_CONFIG, D, F};
use crate::types::constants::TIMESTAMP_BITS;
use crate::uint::u32::gates::arithmetic_u32::U32ArithmeticGate;
use crate::uint::u32::gates::interleave_u32::U32InterleaveGate;
use crate::uint::u32::gates::subtraction_u32::U32SubtractionGate;
use crate::uint::u32::gates::uninterleave_to_u32::UninterleaveToU32Gate;
use crate::uint::u48::subtraction_u48::U48SubtractionGate;

pub trait Circuit<C: GenericConfig<D, F = F>, F: RichField + Extendable<D>, const D: usize> {
    /// Defines the circuit and its each target. Returns `builder` and `target`
    /// `builder` can be used to build circuit via calling [`Builder::build()`]
    /// `target` can be used to assign partial witness in [`CyclicRecursionCircuit::prove()`] function
    fn define(
        config: CircuitConfig,
        block_circuit: &CircuitData<F, C, D>,
        on_chain_operations_limit: usize,
    ) -> Self;

    /// Fills partial witness for batch target with given block data
    fn generate_witness(
        target: &CyclicRecursionTarget,
        circuit_data: &CircuitData<F, C, D>,
        new_batch: &Batch<F>,
        segment_info: &SegmentInfo,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        current_block_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>>;

    fn prove(
        target: &CyclicRecursionTarget,
        circuit_data: &CircuitData<F, C, D>,
        new_batch: &Batch<F>,
        segment_info: &SegmentInfo,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        current_block_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;

    fn cyclic_base_proof(
        circuit_data: &CircuitData<F, C, D>,
        segment_info: &SegmentInfo,
    ) -> ProofWithPublicInputs<F, C, D>;
}

#[derive(Debug)]
pub struct CyclicRecursionCircuit {
    pub builder: Builder,
    pub target: CyclicRecursionTarget,
}

#[derive(Debug)]
pub struct CyclicRecursionTarget {
    pub cyclic_proof: ProofWithPublicInputsTarget<D>, // proof of previous iteration
    pub self_verifier_data: VerifierCircuitTarget,    // Verifier Circuit Data for this circuit
    pub current_block_proof: ProofWithPublicInputsTarget<D>, // proof of next block
    pub not_first_recursion: BoolTarget, // indicator that wheter we are on first iteration or not

    pub dummy_proof_with_pis_target: ProofWithPublicInputsTarget<D>, // Filled with dummy proof

    pub new_batch: BatchTarget,          // Public witness
    pub segment_info: SegmentInfoTarget, // Public witness
}

impl CyclicRecursionCircuit {
    pub fn new(
        config: CircuitConfig,
        block_common_circuit: &CommonCircuitData<F, D>,
    ) -> (Self, CommonCircuitData<F, D>) {
        let mut builder = Builder::new(config);

        // Register public inputs
        let new_batch = BatchTarget::new_public(&mut builder);
        let segment_info = SegmentInfoTarget::new_public(&mut builder);
        let self_verifier_data = builder.add_verifier_data_public_inputs(); // 68

        // IMPORTANT: DO NOT ADD PUBLIC INPUTS AFTER THIS POINT. Building common data for current circuit
        let common_data_for_recursion = CommonCircuitData {
            num_public_inputs: builder.num_public_inputs(),
            ..common_data_for_recursion()
        };

        (
            Self {
                target: CyclicRecursionTarget {
                    cyclic_proof: builder.add_virtual_proof_with_pis(&common_data_for_recursion),
                    not_first_recursion: builder.add_virtual_bool_target_safe(),
                    current_block_proof: builder.add_virtual_proof_with_pis(block_common_circuit),
                    new_batch,
                    segment_info,
                    self_verifier_data,
                    dummy_proof_with_pis_target: builder
                        .add_virtual_proof_with_pis(&common_data_for_recursion), // This value will be overwritten
                },
                builder,
            },
            common_data_for_recursion,
        )
    }

    fn handle_cyclic_and_block_proofs(
        &mut self,
        on_chain_operations_limit: usize,
        self_common_data: CommonCircuitData<F, D>,
        block_circuit: &CircuitData<F, C, D>,
    ) -> (BatchTarget, SegmentInfoTarget, BlockWitnessTarget) {
        // Verify cyclic proof
        self.target.dummy_proof_with_pis_target = self
            .builder
            .conditionally_verify_cyclic_proof_or_dummy::<C>(
                self.target.not_first_recursion,
                &self.target.cyclic_proof,
                &self_common_data,
            )
            .unwrap();

        // Verify block proof
        let block_verifier_data = self
            .builder
            .constant_verifier_data(&block_circuit.verifier_only);
        self.builder.verify_proof::<C>(
            &self.target.current_block_proof,
            &block_verifier_data,
            &block_circuit.common,
        );

        // Extract old batch, initial segment info and current block data from proofs
        let mut batch = BatchTarget::from_public_inputs(
            &self.target.cyclic_proof.public_inputs[..BATCH_TARGET_INDEX],
        );
        let segment_info = SegmentInfoTarget::from_public_inputs(
            &self.target.cyclic_proof.public_inputs[BATCH_TARGET_INDEX..],
        );
        let (current_block, _) = BlockWitnessTarget::from_public_inputs(
            &self.target.current_block_proof.public_inputs,
            on_chain_operations_limit,
            1,
        );

        // Take initial delta root from block for first recursion
        batch.new_delta_root = self.builder.select_hash(
            self.target.not_first_recursion,
            &batch.new_delta_root,
            &current_block.old_delta_root,
        );

        (batch, segment_info, current_block)
    }

    fn perform_sanity_checks(&mut self, batch: &BatchTarget, current_block: &BlockWitnessTarget) {
        // Verify aggregated batch is empty in first recursion
        let is_first_recursion = self.builder.not(self.target.not_first_recursion);
        let batch_before_is_empty = batch.is_empty_for_recursion(&mut self.builder);
        self.builder
            .connect_bool(is_first_recursion, batch_before_is_empty);

        // Verify that block height is continuous
        let next_block_number = self.builder.add_one(batch.end_block_number);
        self.builder.conditional_assert_eq(
            self.target.not_first_recursion,
            next_block_number,
            current_block.block_number,
        );

        // Verify that the last block in the batch was created before current_block
        // No need to range-check the inputs because they are already checked in the block proof
        self.builder.conditional_assert_lte(
            self.target.not_first_recursion,
            batch.end_timestamp,
            current_block.created_at,
            TIMESTAMP_BITS,
        );

        // Verify that current block's old state root is the same as old batch's new state root
        self.builder.conditional_assert_eq_hash(
            self.target.not_first_recursion,
            &batch.new_state_root,
            &current_block.old_state_root,
        );

        self.builder
            .connect_hashes(batch.new_delta_root, current_block.old_delta_root);

        current_block
            .old_prefix_priority_operation_hash
            .iter()
            .zip(batch.new_prefix_priority_operation_hash.iter())
            .for_each(|(&a, &b)| {
                self.builder
                    .conditional_assert_eq(self.target.not_first_recursion, a.0, b.0);
            });
    }

    fn aggregate_on_chain_operations_pub_data(
        &mut self,
        on_chain_operations_pub_data_hash: &KeccakOutputTarget,
        current_block: &BlockWitnessTarget,
        on_chain_operations_limit: usize,
    ) -> KeccakOutputTarget {
        // Calculate new on chain operations hash. For first iteration, `batch.on_chain_operations_pub_data_hash` is
        // zero keccak output(ie. full of zero bits)
        let mut on_chain_operations_hash = *on_chain_operations_pub_data_hash;

        // select_on_chain_pub_data[i] = true iff current block at least have i+1 on chain public data
        let mut select_on_chain_pub_data: Vec<BoolTarget> = (0..on_chain_operations_limit)
            .map(|_| BoolTarget::default())
            .collect();
        for i in 0..on_chain_operations_limit {
            let it = self.builder.constant_usize(i + 1);
            select_on_chain_pub_data[i] = self
                .builder
                .is_equal(current_block.on_chain_operations_count, it);
        }
        for i in (0..(on_chain_operations_limit - 1)).rev() {
            select_on_chain_pub_data[i] = self
                .builder
                .or(select_on_chain_pub_data[i], select_on_chain_pub_data[i + 1]);
        }
        current_block
            .on_chain_operations_pub_data
            .iter()
            .enumerate()
            .for_each(|(i, pub_data)| {
                let mut on_chain_operations_pub_data_input = vec![];

                on_chain_operations_pub_data_input.extend_from_slice(&on_chain_operations_hash);
                on_chain_operations_pub_data_input.extend_from_slice(pub_data);

                let new_on_chain_operations_hash = self
                    .builder
                    .keccak256_circuit(on_chain_operations_pub_data_input);

                on_chain_operations_hash = self.builder.select_keccak_output(
                    select_on_chain_pub_data[i],
                    new_on_chain_operations_hash,
                    on_chain_operations_hash,
                );
            });

        on_chain_operations_hash
    }

    fn aggregate_priority_operations_pub_data(
        &mut self,
        batch: &BatchTarget,
        current_block: &BlockWitnessTarget,
    ) -> (KeccakOutputTarget, Target) {
        current_block
            .old_prefix_priority_operation_hash
            .iter()
            .zip(batch.new_prefix_priority_operation_hash.iter())
            .for_each(|(&a, &b)| {
                self.builder
                    .conditional_assert_eq(self.target.not_first_recursion, a.0, b.0)
            });

        let old_prefix_priority_operation_hash = self.builder.select_arr_u8(
            self.target.not_first_recursion,
            &batch.old_prefix_priority_operation_hash,
            &current_block.old_prefix_priority_operation_hash,
        );

        let priority_operations_count = self.builder.add(
            batch.priority_operations_count,
            current_block.priority_operations_count,
        );

        (
            old_prefix_priority_operation_hash,
            priority_operations_count,
        )
    }
}

impl Circuit<C, F, D> for CyclicRecursionCircuit {
    fn define(
        config: CircuitConfig,
        block_circuit: &CircuitData<F, C, D>,
        on_chain_operations_limit: usize,
    ) -> Self {
        let (mut circuit, common_data) = Self::new(config, &block_circuit.common);

        let (batch, segment_info, current_block) = circuit.handle_cyclic_and_block_proofs(
            on_chain_operations_limit,
            common_data,
            block_circuit,
        );

        circuit.perform_sanity_checks(&batch, &current_block);

        // Take initial on chain data hash from segment for first step
        let on_chain_operations_pub_data_hash = circuit.builder.select_keccak_output(
            circuit.target.not_first_recursion,
            batch.on_chain_operations_pub_data_hash,
            segment_info.old_on_chain_operations_pub_data_hash,
        );
        let on_chain_operations_pub_data_hash = circuit.aggregate_on_chain_operations_pub_data(
            &on_chain_operations_pub_data_hash,
            &current_block,
            on_chain_operations_limit,
        );

        let (old_prefix_priority_operation_hash, priority_operations_count) =
            circuit.aggregate_priority_operations_pub_data(&batch, &current_block);

        let calculated_new_batch = BatchTarget {
            end_block_number: current_block.block_number,
            batch_size: circuit.builder.add_one(batch.batch_size),
            start_timestamp: circuit.builder.select(
                circuit.target.not_first_recursion,
                batch.start_timestamp,
                current_block.created_at,
            ),
            end_timestamp: current_block.created_at,

            old_state_root: circuit.builder.select_hash(
                circuit.target.not_first_recursion,
                &batch.old_state_root,
                &current_block.old_state_root,
            ),
            new_validium_root: current_block.new_validium_root,
            new_state_root: current_block.new_state_root,

            old_delta_root: circuit.builder.select_hash(
                circuit.target.not_first_recursion,
                &batch.old_delta_root,
                &current_block.old_delta_root,
            ),
            new_delta_root: current_block.new_delta_root,

            on_chain_operations_pub_data_hash,

            priority_operations_count,
            old_prefix_priority_operation_hash,
            new_prefix_priority_operation_hash: current_block.new_prefix_priority_operation_hash,

            new_public_market_details: current_block.new_public_market_details,
        };

        circuit
            .target
            .new_batch
            .connect_batches(&mut circuit.builder, &calculated_new_batch);

        circuit
            .target
            .segment_info
            .connect_segments(&mut circuit.builder, &segment_info);

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn generate_witness(
        target: &CyclicRecursionTarget,
        circuit_data: &CircuitData<F, C, D>,
        new_batch: &Batch<F>,
        segment_info: &SegmentInfo,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        current_block_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.cyclic_proof, cyclic_proof)?;
        pw.set_verifier_data_target(&target.self_verifier_data, &circuit_data.verifier_only)?;

        pw.set_proof_with_pis_target(&target.current_block_proof, current_block_proof)?;

        pw.set_bool_target(target.not_first_recursion, not_first_recursion)?;
        pw.set_batch_target(&target.new_batch, new_batch)?;
        pw.set_segment_info_target(&target.segment_info, segment_info)?;

        // This will take place of `DummyProofGenerator`
        pw.set_proof_with_pis_target(&target.dummy_proof_with_pis_target, dummy_proof)?;

        Ok(pw)
    }

    fn prove(
        target: &CyclicRecursionTarget,
        circuit_data: &CircuitData<F, C, D>,
        new_batch: &Batch<F>,
        segment_info: &SegmentInfo,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        current_block_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("recursive prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(
                target,
                circuit_data,
                new_batch,
                segment_info,
                not_first_recursion,
                cyclic_proof,
                dummy_proof,
                current_block_proof,
            )?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    fn cyclic_base_proof(
        circuit_data: &CircuitData<F, C, D>,
        segment_info: &SegmentInfo,
    ) -> ProofWithPublicInputs<F, C, D> {
        let mut nonzero_public_inputs = HashMap::new();
        let public_inputs = segment_info.to_public_inputs();
        (BATCH_TARGET_INDEX..SEGMENT_INFO_INDEX).for_each(|i| {
            nonzero_public_inputs.insert(i, public_inputs[i - BATCH_TARGET_INDEX]);
        });

        cyclic_base_proof(
            &circuit_data.common,
            &circuit_data.verifier_only,
            nonzero_public_inputs,
        )
    }
}

// Generates `CommonCircuitData` usable for recursion.
fn common_data_for_recursion() -> CommonCircuitData<F, D> {
    let builder = Builder::new(CIRCUIT_CONFIG);
    let data = builder.build::<C>();

    let mut builder = Builder::new(CIRCUIT_CONFIG);
    let proof = builder.add_virtual_proof_with_pis(&data.common);
    let verifier_data = builder.add_virtual_verifier_data(data.common.config.fri_config.cap_height);
    builder.verify_proof::<C>(&proof, &verifier_data, &data.common);
    let data = builder.build::<C>();

    let mut builder = Builder::new(CIRCUIT_CONFIG);
    let proof = builder.add_virtual_proof_with_pis(&data.common);
    let verifier_data = builder.add_virtual_verifier_data(data.common.config.fri_config.cap_height);
    builder.verify_proof::<C>(&proof, &verifier_data, &data.common);

    let config = CIRCUIT_CONFIG.clone();
    builder.add_gate(U48SubtractionGate::new_from_config(&config.clone()), vec![]);
    builder.add_gate(Poseidon2Gate::new(), vec![]);
    builder.add_gate(EqualityGate::new_from_config(&config), vec![]);
    builder.add_gate(SelectionGate::new_from_config(&config), vec![]);
    builder.add_gate(U32InterleaveGate::new_from_config(&config), vec![]);
    builder.add_gate(UninterleaveToU32Gate::new_from_config(&config), vec![]);
    builder.add_gate(ByteDecompositionGate::new_from_config(&config, 8), vec![]);
    builder.add_gate(U32ArithmeticGate::new_from_config(&config), vec![]);
    builder.add_gate(U32SubtractionGate::new_from_config(&config), vec![]);
    builder.add_gate(ExponentiationGate::new_from_config(&config), vec![]);
    builder.add_gate(ConstantGate::new(2), vec![]);

    while builder.num_gates() < 1 << 14 {
        builder.add_gate(plonky2::gates::noop::NoopGate, vec![]);
    }

    builder.build::<C>().common
}
