// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use hashbrown::HashMap;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::equality_base::EqualityGate;
use plonky2::gates::select_base::SelectionGate;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget,
};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::bigint::big_u16::CircuitBuilderBigIntU16;
use crate::block_tx::BlockTxWitnessTarget;
use crate::block_tx_chain::BlockTxChainWitnessTarget;
use crate::builder::custom::cyclic_base_proof;
use crate::byte::split_gate::ByteDecompositionGate;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Gate;
use crate::types::approve_integrator::ApproveIntegratorMessageTarget;
use crate::types::asset::all_assets_hash;
use crate::types::change_pub_key::ChangePubKeyMessageTarget;
use crate::types::config::{Builder, C, CIRCUIT_CONFIG, D, F};
use crate::types::constants::ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;
use crate::types::margined_asset::all_margined_assets_hash;
use crate::types::market_details::{
    PublicMarketDetailsTarget, all_market_details_hash, all_public_market_details_hash,
};
use crate::types::state_metadata::{
    STATE_METADATA_SIZE, StateMetadata, StateMetadataTarget, connect_state_metadata_target,
};
use crate::types::transfer::TransferMessageTarget;
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::utils::CircuitBuilderUtils;

pub trait Circuit<C: GenericConfig<D, F = F>, F: RichField + Extendable<D>, const D: usize> {
    /// Defines the circuit and its each target. Returns `builder` and `target`.
    /// `builder` can be used to build circuit via calling [`Builder::build()`].
    /// `target` can be used to assign partial witness in [`BlockTxChainCircuit::prove()`] function
    fn define(
        config: CircuitConfig,
        block_tx_circuit: &CircuitData<F, C, D>,
        tx_per_proof: usize,
        on_chain_operations_limit: usize,
    ) -> Self;

    /// Fills partial witness for batch target with given block data
    fn generate_witness(
        target: &BlockTxChainTarget,
        circuit_data: &CircuitData<F, C, D>,
        tx_index: u64, // index of current tx in the block
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof_cyclic: &ProofWithPublicInputs<F, C, D>,
        current_block_tx_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>>;

    fn prove(
        target: &BlockTxChainTarget,
        circuit_data: &CircuitData<F, C, D>,
        tx_index: u64,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof_cyclic: &ProofWithPublicInputs<F, C, D>,
        current_block_tx_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;

    /// Generates base proof for the cyclic circuit with given public inputs
    fn cyclic_base_proof(
        circuit_data: &CircuitData<F, C, D>,
        dummy_circuit: &CircuitData<F, C, D>,
        block_number: u64,
        created_at: i64,
        old_state_root: HashOut<F>,
        new_state_root: HashOut<F>,
        new_validium_root: HashOut<F>,
        new_account_delta_tree_root: HashOut<F>,
        block_tx_witness_size: usize,
        state_metadata: &StateMetadata,
    ) -> ProofWithPublicInputs<F, C, D>;
}

#[derive(Debug)]
pub struct BlockTxChainCircuit {
    pub builder: Builder,
    pub target: BlockTxChainTarget,
    pub block_tx_witness_size: usize,
}

#[derive(Debug)]
pub struct BlockTxChainTarget {
    pub cyclic_proof: ProofWithPublicInputsTarget<D>, // proof of previous iteration
    pub self_verifier_data: VerifierCircuitTarget,    // Verifier Circuit Data for this circuit

    pub current_block_tx_proof: ProofWithPublicInputsTarget<D>, // proof of next block tx
    pub tx_index: Target,                                       // index of current tx in the block

    pub dummy_proof_with_pis_target_cyclic: ProofWithPublicInputsTarget<D>, // Filled with dummy proof for first iteration

    pub new_block: BlockTxChainWitnessTarget, // Public witness - Block state after iterating current block
    pub state_metadata_target: StateMetadataTarget, // Public witness - Carry data between all iterations to calculate state roots, doesn't change between iterations
}

impl BlockTxChainCircuit {
    pub fn new(
        config: CircuitConfig,
        block_tx_common_circuit: &CommonCircuitData<F, D>,
        tx_per_proof: usize,
        on_chain_operations_limit: usize,
    ) -> (Self, CommonCircuitData<F, D>) {
        let mut builder = Builder::new(config);

        // Register public inputs
        let new_block =
            BlockTxChainWitnessTarget::new_public(&mut builder, on_chain_operations_limit);
        let state_metadata_target = StateMetadataTarget::new_public(&mut builder);

        let self_verifier_data = builder.add_verifier_data_public_inputs();

        let mut log_gates = 13;
        if tx_per_proof > 6 {
            log_gates = 14;
        }

        // IMPORTANT: DO NOT ADD PUBLIC INPUTS AFTER THIS POINT.
        // Building common data for current circuit
        let common_data_for_recursion = CommonCircuitData {
            num_public_inputs: builder.num_public_inputs(),
            ..common_data_for_recursion(log_gates)
        };

        (
            Self {
                target: BlockTxChainTarget {
                    tx_index: builder.add_virtual_target(),
                    new_block,
                    state_metadata_target,

                    self_verifier_data,

                    cyclic_proof: builder.add_virtual_proof_with_pis(&common_data_for_recursion),
                    dummy_proof_with_pis_target_cyclic: builder
                        .add_virtual_proof_with_pis(&common_data_for_recursion), // This value will be overwritten

                    current_block_tx_proof: builder
                        .add_virtual_proof_with_pis(block_tx_common_circuit),
                },
                builder,
                block_tx_witness_size: 0, // will be calculated in `handle_cyclic_and_block_proofs`
            },
            common_data_for_recursion,
        )
    }

    fn handle_cyclic_and_block_proofs(
        &mut self,
        on_chain_operations_limit: usize,
        self_common_data: CommonCircuitData<F, D>,
        block_tx_circuit: &CircuitData<F, C, D>,
    ) -> (
        BlockTxChainWitnessTarget,
        BlockTxWitnessTarget,
        StateMetadataTarget,
    ) {
        let not_first_recursion = self.builder.is_not_zero(self.target.tx_index);

        // Verify cyclic proof
        self.target.dummy_proof_with_pis_target_cyclic = self
            .builder
            .conditionally_verify_cyclic_proof_or_dummy::<C>(
                not_first_recursion,
                &self.target.cyclic_proof,
                &self_common_data,
            )
            .unwrap();

        // Verify current block tx proof
        let block_tx_verifier_data = self
            .builder
            .constant_verifier_data(&block_tx_circuit.verifier_only);
        self.builder.verify_proof::<C>(
            &self.target.current_block_tx_proof,
            &block_tx_verifier_data,
            &block_tx_circuit.common,
        );

        // Extract old block and state metadata from cyclic proof
        let (block, block_pis_size) = BlockTxChainWitnessTarget::from_public_inputs(
            &self.target.cyclic_proof.public_inputs,
            on_chain_operations_limit,
            1,
        );

        self.block_tx_witness_size = block_pis_size;

        let state_metadata_target = StateMetadataTarget {
            last_funding_round_timestamp: self.target.cyclic_proof.public_inputs[block_pis_size],
            last_oracle_price_timestamp: self.target.cyclic_proof.public_inputs[block_pis_size + 1],
            last_premium_timestamp: self.target.cyclic_proof.public_inputs[block_pis_size + 2],
        };

        // Extract current tx from tx proof
        let current_block_tx = BlockTxWitnessTarget::from_public_inputs(
            &self.target.current_block_tx_proof.public_inputs,
        );

        (block, current_block_tx, state_metadata_target)
    }

    fn perform_sanity_checks(
        &mut self,
        block: &BlockTxChainWitnessTarget,
        current_tx: &BlockTxWitnessTarget,
        state_metadata_hash: HashOutTarget,
    ) {
        let is_first_recursion = self.builder.is_zero(self.target.tx_index);

        // Assert that tx, priority and on-chain operations are empty for the first iteration
        self.builder
            .conditional_assert_zero(is_first_recursion, block.on_chain_operations_count);
        self.builder
            .conditional_assert_zero(is_first_recursion, block.priority_operations_count);

        block
            .change_pub_key_message
            .conditional_assert_empty(&mut self.builder, is_first_recursion);
        block
            .transfer_message
            .conditional_assert_empty(&mut self.builder, is_first_recursion);
        block
            .approve_integrator_message
            .conditional_assert_empty(&mut self.builder, is_first_recursion);

        block
            .on_chain_operations_pub_data
            .iter()
            .for_each(|&pub_data| {
                pub_data.iter().for_each(|&pub_data| {
                    self.builder
                        .conditional_assert_zero(is_first_recursion, pub_data.0);
                });
            });
        block
            .priority_operations_pub_data
            .iter()
            .for_each(|&pub_data| {
                self.builder
                    .conditional_assert_zero(is_first_recursion, pub_data.0);
            });

        // Calculate old validium and state roots
        let system_config_hash = current_tx.old_system_config.hash(&mut self.builder);
        let register_stack_hash = current_tx.register_stack_before.hash(&mut self.builder);
        let all_assets_hash = all_assets_hash(&mut self.builder, &current_tx.all_assets_before);
        let all_margined_assets_hash =
            all_margined_assets_hash(&mut self.builder, &current_tx.all_margined_assets_before);
        let all_market_details_hash =
            all_market_details_hash(&mut self.builder, &current_tx.all_market_details_before);
        let all_public_market_details_hash = all_public_market_details_hash(
            &mut self.builder,
            &current_tx.all_market_details_before,
        );
        let validium_root = self.builder.hash_n_to_one(&[
            register_stack_hash,
            current_tx.old_account_tree_root,
            current_tx.old_market_tree_root,
            all_assets_hash,
            all_margined_assets_hash,
            all_market_details_hash,
            state_metadata_hash,
            system_config_hash,
        ]);

        let state_root = self.builder.hash_n_to_one(&[
            current_tx.old_account_pub_data_tree_root,
            all_public_market_details_hash,
            validium_root,
        ]);

        // Verify that old block's new state root is the same as new tx's old state root
        self.builder
            .connect_hashes(block.new_state_root, state_root);
        self.builder
            .connect_hashes(block.new_validium_root, validium_root);

        // Verify that at first recursion, old state root is the same as new state root because there is no tx in the block
        self.builder.conditional_assert_eq_hash(
            is_first_recursion,
            &block.old_state_root,
            &block.new_state_root,
        );

        // Verify continuity of account delta tree root
        self.builder.connect_hashes(
            block.new_account_delta_tree_root,
            current_tx.old_account_delta_tree_root,
        );
    }
}

impl Circuit<C, F, D> for BlockTxChainCircuit {
    fn define(
        config: CircuitConfig,
        block_tx_circuit: &CircuitData<F, C, D>,
        tx_per_proof: usize,
        on_chain_operations_limit: usize,
    ) -> Self {
        let (mut circuit, common_data) = Self::new(
            config,
            &block_tx_circuit.common,
            tx_per_proof,
            on_chain_operations_limit,
        );

        let (block, current_tx, state_metadata) = circuit.handle_cyclic_and_block_proofs(
            on_chain_operations_limit,
            common_data,
            block_tx_circuit,
        );

        let state_metadata_hash = state_metadata.hash(&mut circuit.builder);
        circuit.perform_sanity_checks(&block, &current_tx, state_metadata_hash);

        // Calculate new validium and state root
        let system_config_hash = current_tx.new_system_config.hash(&mut circuit.builder);
        let register_stack_hash = current_tx.register_stack_after.hash(&mut circuit.builder);
        let all_assets_hash = all_assets_hash(&mut circuit.builder, &current_tx.all_assets_after);
        let all_margined_assets_hash =
            all_margined_assets_hash(&mut circuit.builder, &current_tx.all_margined_assets_after);
        let all_market_details_hash =
            all_market_details_hash(&mut circuit.builder, &current_tx.all_market_details_after);
        let all_public_market_details_hash = all_public_market_details_hash(
            &mut circuit.builder,
            &current_tx.all_market_details_after,
        );

        let validium_root = circuit.builder.hash_n_to_one(&[
            register_stack_hash,
            current_tx.new_account_tree_root,
            current_tx.new_market_tree_root,
            all_assets_hash,
            all_margined_assets_hash,
            all_market_details_hash,
            state_metadata_hash,
            system_config_hash,
        ]);

        let state_root = circuit.builder.hash_n_to_one(&[
            current_tx.new_account_pub_data_tree_root,
            all_public_market_details_hash,
            validium_root,
        ]);

        // Treasury can't change pubkey via L2 transaction, so if account index is zero, then change_pub_key_message is empty
        let is_change_pub_key_message_exists = circuit
            .builder
            .is_not_zero(current_tx.change_pub_key_message.account_index);
        circuit.builder.conditional_assert_zero(
            is_change_pub_key_message_exists,
            block.change_pub_key_message.account_index,
        );
        let change_pub_key_message = ChangePubKeyMessageTarget::select(
            &mut circuit.builder,
            is_change_pub_key_message_exists,
            &current_tx.change_pub_key_message,
            &block.change_pub_key_message,
        );

        // Treasury can call L2 transfer, but it can only transfer between its own (sub)accounts. So transfer_message is empty in that case
        let is_transfer_message_exists = circuit
            .builder
            .is_not_zero(current_tx.transfer_message.from_account_index);
        circuit.builder.conditional_assert_zero(
            is_transfer_message_exists,
            block.transfer_message.from_account_index,
        );
        let transfer_message = TransferMessageTarget::select(
            &mut circuit.builder,
            is_transfer_message_exists,
            &current_tx.transfer_message,
            &block.transfer_message,
        );

        // Treasury can't call L2 approve_integrator, so if account index is zero, then approve_integrator_message is empty
        let is_approve_integrator_message_exists = circuit
            .builder
            .is_not_zero(current_tx.approve_integrator_message.account_index);
        circuit.builder.conditional_assert_zero(
            is_approve_integrator_message_exists,
            block.approve_integrator_message.account_index,
        );
        let approve_integrator_message = ApproveIntegratorMessageTarget::select(
            &mut circuit.builder,
            is_approve_integrator_message_exists,
            &current_tx.approve_integrator_message,
            &block.approve_integrator_message,
        );

        let on_chain_operation_exists = circuit
            .builder
            .is_not_zero(current_tx.on_chain_operations_count);
        let priority_operations_pub_data_exists = circuit
            .builder
            .is_not_zero(current_tx.priority_operations_count);

        let new_priority_operations_pub_data = circuit.builder.select_arr_u8(
            priority_operations_pub_data_exists,
            &current_tx.priority_operations_pub_data,
            &block.priority_operations_pub_data,
        );
        // If current tx has a priority operation, then we should not have any before to satisfy the assumption
        circuit.builder.conditional_assert_zero(
            priority_operations_pub_data_exists,
            block.priority_operations_count,
        );
        // Verify that we have at most 1 priority operation per tx segment
        circuit
            .builder
            .assert_bool(BoolTarget::new_unsafe(current_tx.priority_operations_count));

        let mut on_chain_operations_pub_data = block.on_chain_operations_pub_data.clone();
        select_on_chain_pub_data(
            &mut circuit.builder,
            on_chain_operations_limit,
            block.on_chain_operations_count,
            &mut on_chain_operations_pub_data,
            &current_tx.on_chain_operations_pub_data,
            on_chain_operation_exists,
        );

        // If current tx has an on-chain operation, then we should not have on_chain_operations_limit operations before
        let on_chain_operations_limit = circuit.builder.constant_usize(on_chain_operations_limit);
        circuit.builder.conditional_assert_not_eq(
            on_chain_operation_exists,
            block.on_chain_operations_count,
            on_chain_operations_limit,
        );
        // Verify that we have at most 1 onchain operation per tx segment
        circuit
            .builder
            .assert_bool(BoolTarget::new_unsafe(current_tx.on_chain_operations_count));

        let calculated_new_block = BlockTxChainWitnessTarget {
            block_number: block.block_number,
            created_at: block.created_at,
            old_state_root: block.old_state_root,

            new_validium_root: validium_root,
            new_state_root: state_root,
            new_account_delta_tree_root: current_tx.new_account_delta_tree_root,

            change_pub_key_message,
            transfer_message,
            approve_integrator_message,

            on_chain_operations_count: circuit.builder.add(
                block.on_chain_operations_count,
                current_tx.on_chain_operations_count,
            ),
            on_chain_operations_pub_data,

            priority_operations_count: circuit.builder.add(
                block.priority_operations_count,
                current_tx.priority_operations_count,
            ),
            priority_operations_pub_data: new_priority_operations_pub_data,

            new_public_market_details: current_tx
                .all_market_details_after
                .iter()
                .map(|market| PublicMarketDetailsTarget {
                    funding_rate_prefix_sum: circuit
                        .builder
                        .bigint_u16_to_bigint(&market.funding_rate_prefix_sum),
                    mark_price: market.mark_price,
                    quote_multiplier: market.quote_multiplier,
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        };

        circuit
            .target
            .new_block
            .connect_block_witness(&mut circuit.builder, &calculated_new_block);

        connect_state_metadata_target(
            &mut circuit.builder,
            &state_metadata,
            &circuit.target.state_metadata_target,
        );

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn generate_witness(
        target: &BlockTxChainTarget,
        circuit_data: &CircuitData<F, C, D>,
        tx_index: u64,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof_cyclic: &ProofWithPublicInputs<F, C, D>,
        current_block_tx_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.cyclic_proof, cyclic_proof)?;
        pw.set_verifier_data_target(&target.self_verifier_data, &circuit_data.verifier_only)?;

        pw.set_proof_with_pis_target(&target.current_block_tx_proof, current_block_tx_proof)?;

        pw.set_target(target.tx_index, F::from_canonical_u64(tx_index))?;

        // This will take place of `DummyProofGenerator`
        pw.set_proof_with_pis_target(
            &target.dummy_proof_with_pis_target_cyclic,
            dummy_proof_cyclic,
        )?;

        Ok(pw)
    }

    fn prove(
        target: &BlockTxChainTarget,
        circuit_data: &CircuitData<F, C, D>,
        tx_index: u64,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof_cyclic: &ProofWithPublicInputs<F, C, D>,
        current_block_tx_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockTxChainCircuit", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(
                target,
                circuit_data,
                tx_index,
                cyclic_proof,
                dummy_proof_cyclic,
                current_block_tx_proof,
            )?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    /// Public input indexes comes from the public inputs registered in the circuit [`BlockTxChainCircuit::new`]
    /// and [`BlockTxChainWitnessTarget::new_public`]
    fn cyclic_base_proof(
        circuit_data: &CircuitData<F, C, D>,
        dummy_circuit: &CircuitData<F, C, D>,
        block_number: u64,
        created_at: i64,
        old_state_root: HashOut<F>,
        new_state_root: HashOut<F>,
        new_validium_root: HashOut<F>,
        old_account_delta_tree_root: HashOut<F>,
        block_tx_witness_size: usize,
        state_metadata: &StateMetadata,
    ) -> ProofWithPublicInputs<F, C, D> {
        assert_eq!(
            old_state_root, new_state_root,
            "old state root should be equal to new state root at base proof"
        );

        let mut nonzero_public_inputs = HashMap::new();

        nonzero_public_inputs.insert(0, F::from_canonical_u64(block_number));
        nonzero_public_inputs.insert(1, F::from_canonical_u64(created_at as u64));

        for (i, elem) in [
            old_state_root,
            new_validium_root,
            new_state_root,
            old_account_delta_tree_root,
        ]
        .iter()
        .flat_map(|&hash| hash.elements)
        .enumerate()
        {
            nonzero_public_inputs.insert(2 + i, elem);
        }

        let public_inputs = state_metadata.to_public_inputs();
        (block_tx_witness_size..block_tx_witness_size + STATE_METADATA_SIZE).for_each(|i| {
            nonzero_public_inputs.insert(i, public_inputs[i - block_tx_witness_size]);
        });

        cyclic_base_proof(
            &circuit_data.common,
            &circuit_data.verifier_only,
            dummy_circuit,
            nonzero_public_inputs,
        )
        .unwrap()
    }
}

fn select_on_chain_pub_data(
    builder: &mut Builder,
    on_chain_operations_limit: usize,
    on_chain_operations_count: Target,
    on_chain_operations_pub_data: &mut [[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]],
    tx_on_chain_operations_pub_data: &[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    on_chain_operation_exists: BoolTarget,
) {
    assert_eq!(
        on_chain_operations_pub_data.len(),
        on_chain_operations_limit,
        "incorrect on chain pub data length"
    );

    (0..on_chain_operations_limit).for_each(|slot_id| {
        let slot_id_t = builder.constant_usize(slot_id);
        let is_current_slot = builder.is_equal(slot_id_t, on_chain_operations_count);
        let flag = builder.and(on_chain_operation_exists, is_current_slot);

        let slot = on_chain_operations_pub_data.get_mut(slot_id).unwrap();
        for i in 0..ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE {
            slot[i] = builder.select_u8(flag, tx_on_chain_operations_pub_data[i], slot[i]);
        }
    });
}

// Generates `CommonCircuitData` usable for recursion.
fn common_data_for_recursion(log_gates: usize) -> CommonCircuitData<F, D> {
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

    builder.add_gate(Poseidon2Gate::new(), vec![]);
    builder.add_gate(EqualityGate::new_from_config(&CIRCUIT_CONFIG), vec![]);
    builder.add_gate(SelectionGate::new_from_config(&CIRCUIT_CONFIG), vec![]);
    builder.add_gate(
        ByteDecompositionGate::new_from_config(&CIRCUIT_CONFIG, 8),
        vec![],
    );
    builder.add_gate(ConstantGate::new(2), vec![]);

    while builder.num_gates() < 1 << log_gates {
        builder.add_gate(plonky2::gates::noop::NoopGate, vec![]);
    }

    builder.build::<C>().common
}
