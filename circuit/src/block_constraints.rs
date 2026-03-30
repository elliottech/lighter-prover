// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use itertools::Itertools;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::BoolTarget;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData, CommonCircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::{
    CompressedProofWithPublicInputs, ProofWithPublicInputs, ProofWithPublicInputsTarget,
};
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::block::{Block, BlockWitness};
use crate::block_pre_execution::BlockPreExecWitnessTarget;
use crate::block_tx_chain::BlockTxChainWitnessTarget;
use crate::ecdsa::gadgets::ecdsa::{
    CircuitBuilderECDSAPublicKey, CircuitBuilderECDSASignature, conditional_verify_ecdsa_sig,
};
use crate::keccak::keccak::CircuitBuilderKeccak;
use crate::nonnative::CircuitBuilderNonNative;
use crate::recursion::block_witness::BlockWitnessTarget;
use crate::types::config::{Builder, C, D, F};
use crate::types::constants::KECCAK_HASH_OUT_BYTE_SIZE;
use crate::types::market_details::PublicMarketDetailsWitness;
use crate::types::state_metadata::{StateMetadataTarget, connect_state_metadata_target};
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

pub trait Circuit<
    C: GenericConfig<D, F = F>,
    F: RichField + Extendable<D> + Extendable<5>,
    const D: usize,
>
{
    /// Defines the circuit and its each target. Returns `builder` and `target`
    ///
    /// `builder` can be used to build circuit via calling [`Builder::build()`]
    ///
    /// `target` can be used to assign partial witness in [`BlockTxChainCircuit::prove()`] function
    fn define(
        config: CircuitConfig,
        block_pre_exec_circuit: &CircuitData<F, C, D>,
        block_tx_chain_circuit: &CircuitData<F, C, D>,
        on_chain_operations_limit: usize,
    ) -> Self;

    /// Fills partial witness for batch target with given block data
    fn generate_witness(
        target: &BlockTarget,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>>;

    fn prove(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;

    fn prove_and_compress(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<CompressedProofWithPublicInputs<F, C, D>>;
}

#[derive(Debug)]
pub struct BlockCircuit {
    pub builder: Builder,
    pub target: BlockTarget,
}

#[derive(Debug)]
pub struct BlockTarget {
    pub pre_exec_proof: ProofWithPublicInputsTarget<D>, // proof of pre execution beginning of the block
    pub tx_chain_proof: ProofWithPublicInputsTarget<D>, // proof of next txs in the block

    pub block: BlockWitnessTarget, // Public block witness
}

impl BlockCircuit {
    pub fn new(
        config: CircuitConfig,
        block_pre_exec_common_circuit: &CommonCircuitData<F, D>,
        block_tx_chain_common_circuit: &CommonCircuitData<F, D>,
        on_chain_operations_limit: usize,
    ) -> Self {
        let mut builder = Builder::new(config);

        // Register public inputs
        let block = BlockWitnessTarget::new_public(&mut builder, on_chain_operations_limit);

        Self {
            target: BlockTarget {
                block,
                pre_exec_proof: builder.add_virtual_proof_with_pis(block_pre_exec_common_circuit),
                tx_chain_proof: builder.add_virtual_proof_with_pis(block_tx_chain_common_circuit),
            },
            builder,
        }
    }

    fn handle_proofs(
        &mut self,
        on_chain_operations_limit: usize,
        block_pre_exec_circuit: &CircuitData<F, C, D>,
        block_tx_chain_circuit: &CircuitData<F, C, D>,
    ) -> (
        BlockPreExecWitnessTarget,
        BlockTxChainWitnessTarget,
        StateMetadataTarget,
    ) {
        // Verify pre-exec proof
        let block_pre_exec_verifier_data = self
            .builder
            .constant_verifier_data(&block_pre_exec_circuit.verifier_only);
        self.builder.verify_proof::<C>(
            &self.target.pre_exec_proof,
            &block_pre_exec_verifier_data,
            &block_pre_exec_circuit.common,
        );

        // Verify tx chain proof
        let block_tx_chain_verifier_data = self
            .builder
            .constant_verifier_data(&block_tx_chain_circuit.verifier_only);
        self.builder.verify_proof::<C>(
            &self.target.tx_chain_proof,
            &block_tx_chain_verifier_data,
            &block_tx_chain_circuit.common,
        );

        // Extract pre-exec and tx chain witnesses from the proofs
        let pre_exec_witness = BlockPreExecWitnessTarget::from_public_inputs(
            &self.target.pre_exec_proof.public_inputs,
        );
        let (tx_chain_witness, tx_chain_witness_size) =
            BlockTxChainWitnessTarget::from_public_inputs(
                &self.target.tx_chain_proof.public_inputs,
                on_chain_operations_limit,
                1,
            );
        let state_metadata = StateMetadataTarget {
            last_funding_round_timestamp: self.target.tx_chain_proof.public_inputs
                [tx_chain_witness_size],
            last_oracle_price_timestamp: self.target.tx_chain_proof.public_inputs
                [tx_chain_witness_size + 1],
            last_premium_timestamp: self.target.tx_chain_proof.public_inputs
                [tx_chain_witness_size + 2],
        };

        (pre_exec_witness, tx_chain_witness, state_metadata)
    }

    /// Verifies that current tx is the next tx in the block
    fn perform_sanity_checks(
        &mut self,
        pre_exec_witness: &BlockPreExecWitnessTarget,
        tx_chain_witness: &BlockTxChainWitnessTarget,
        state_metadata: &StateMetadataTarget,
    ) {
        connect_state_metadata_target(
            &mut self.builder,
            state_metadata,
            &pre_exec_witness.new_state_metadata,
        );

        self.builder
            .connect(pre_exec_witness.block_number, tx_chain_witness.block_number);
        self.builder
            .connect(pre_exec_witness.created_at, tx_chain_witness.created_at);

        self.builder.connect_hashes(
            pre_exec_witness.new_state_root,
            tx_chain_witness.old_state_root,
        );

        // Skipping pre_exec_witness.new_market_details and pre_exec_witness.new_validium_root
        // because they are only exposed as public witness to use them in the recursion
        // Connecting state roots should be enough to verify that tx chain is successor of pre_exec
    }
}

impl Circuit<C, F, D> for BlockCircuit {
    fn define(
        config: CircuitConfig,
        block_pre_exec_circuit: &CircuitData<F, C, D>,
        block_tx_chain_circuit: &CircuitData<F, C, D>,
        on_chain_operations_limit: usize,
    ) -> Self {
        let mut circuit = Self::new(
            config,
            &block_pre_exec_circuit.common,
            &block_tx_chain_circuit.common,
            on_chain_operations_limit,
        );

        let (pre_exec_witness, tx_chain_witness, state_metadata) = circuit.handle_proofs(
            on_chain_operations_limit,
            block_pre_exec_circuit,
            block_tx_chain_circuit,
        );

        circuit.perform_sanity_checks(&pre_exec_witness, &tx_chain_witness, &state_metadata);

        // Calculate new priority operations hash
        let is_priority_operations_exists = circuit
            .builder
            .is_not_zero(tx_chain_witness.priority_operations_count);
        let keccak_input: Vec<U8Target> = circuit
            .target
            .block
            .old_prefix_priority_operation_hash
            .iter()
            .chain(tx_chain_witness.priority_operations_pub_data.iter())
            .cloned()
            .collect();
        let new_priority_operations_hash = circuit.builder.keccak256_circuit(keccak_input);
        let new_priority_operations_hash = circuit.builder.select_arr_u8(
            is_priority_operations_exists,
            &new_priority_operations_hash,
            &circuit.target.block.old_prefix_priority_operation_hash,
        );

        // Select l1 signature message hash, l1 address, l1 signature and l1 public key
        let is_change_pub_key_exists = circuit
            .builder
            .is_not_zero(tx_chain_witness.change_pub_key_message.account_index);
        let change_pub_key_message = tx_chain_witness
            .change_pub_key_message
            .get_change_pub_key_l1_signature_msg_hash(&mut circuit.builder);
        let (mut l1_message, mut l1_address, mut l1_signature, mut l1_pk) = (
            change_pub_key_message,
            tx_chain_witness.change_pub_key_message.l1_address,
            tx_chain_witness.change_pub_key_message.l1_signature,
            tx_chain_witness.change_pub_key_message.l1_pk,
        );

        let is_transfer_exists = circuit
            .builder
            .is_not_zero(tx_chain_witness.transfer_message.from_account_index);
        let transfer_message = tx_chain_witness
            .transfer_message
            .get_transfer_l1_signature_msg_hash(&mut circuit.builder);
        (l1_message, l1_address, l1_signature, l1_pk) = (
            circuit
                .builder
                .select_nonnative(is_transfer_exists, &transfer_message, &l1_message),
            circuit.builder.select_biguint(
                is_transfer_exists,
                &tx_chain_witness.transfer_message.l1_address,
                &l1_address,
            ),
            circuit.builder.select_ecdsa_signature(
                is_transfer_exists,
                &tx_chain_witness.transfer_message.l1_signature,
                &l1_signature,
            ),
            circuit.builder.select_ecdsa_public_key(
                is_transfer_exists,
                &tx_chain_witness.transfer_message.l1_pk,
                &l1_pk,
            ),
        );

        let is_approve_integrator_exists = circuit
            .builder
            .is_not_zero(tx_chain_witness.approve_integrator_message.account_index);
        let approve_integrator_message = tx_chain_witness
            .approve_integrator_message
            .get_approve_integrator_l1_signature_msg_hash(&mut circuit.builder);
        (l1_message, l1_address, l1_signature, l1_pk) = (
            circuit.builder.select_nonnative(
                is_approve_integrator_exists,
                &approve_integrator_message,
                &l1_message,
            ),
            circuit.builder.select_biguint(
                is_approve_integrator_exists,
                &tx_chain_witness.approve_integrator_message.l1_address,
                &l1_address,
            ),
            circuit.builder.select_ecdsa_signature(
                is_approve_integrator_exists,
                &tx_chain_witness.approve_integrator_message.l1_signature,
                &l1_signature,
            ),
            circuit.builder.select_ecdsa_public_key(
                is_approve_integrator_exists,
                &tx_chain_witness.approve_integrator_message.l1_pk,
                &l1_pk,
            ),
        );

        // A block can have only one l1 signature
        let at_most_one_message = BoolTarget::new_unsafe(circuit.builder.add_many([
            is_change_pub_key_exists.target,
            is_transfer_exists.target,
            is_approve_integrator_exists.target,
        ]));
        circuit.builder.assert_bool(at_most_one_message);

        // Verify l1 signature
        conditional_verify_ecdsa_sig(
            &mut circuit.builder,
            at_most_one_message,
            &l1_message,
            &l1_signature,
            &l1_pk,
        );
        // Connect l1 address
        let l1_address_from_pk = circuit.builder.get_l1_address_from_ecdsa_public_key(&l1_pk);
        circuit.builder.conditional_assert_eq_biguint(
            at_most_one_message,
            &l1_address_from_pk,
            &l1_address,
        );

        let calculated_block = BlockWitnessTarget {
            block_number: pre_exec_witness.block_number,
            created_at: pre_exec_witness.created_at,
            old_state_root: pre_exec_witness.old_state_root,
            new_validium_root: tx_chain_witness.new_validium_root,
            new_state_root: tx_chain_witness.new_state_root,

            old_account_delta_tree_root: circuit.target.block.old_account_delta_tree_root,
            new_account_delta_tree_root: tx_chain_witness.new_account_delta_tree_root,

            on_chain_operations_count: tx_chain_witness.on_chain_operations_count,
            on_chain_operations_pub_data: tx_chain_witness.on_chain_operations_pub_data,
            priority_operations_count: tx_chain_witness.priority_operations_count,
            old_prefix_priority_operation_hash: circuit
                .target
                .block
                .old_prefix_priority_operation_hash,
            new_prefix_priority_operation_hash: new_priority_operations_hash,
            new_public_market_details: tx_chain_witness.new_public_market_details,
        };

        circuit
            .target
            .block
            .connect_block_witness(&mut circuit.builder, &calculated_block);

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn generate_witness(
        target: &BlockTarget,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.pre_exec_proof, pre_exec_proof)?;
        pw.set_proof_with_pis_target(&target.tx_chain_proof, tx_chain_proof)?;

        let block_witness = BlockWitness::from_block(block, 1);

        pw.set_target(
            target.block.block_number,
            F::from_canonical_u64(block.block_number),
        )?;
        pw.set_target(
            target.block.created_at,
            F::from_canonical_u64(block.created_at as u64),
        )?;

        pw.set_hash_target(target.block.old_state_root, block_witness.old_state_root)?;

        pw.set_hash_target(
            target.block.new_validium_root,
            block_witness.new_validium_root,
        )?;
        pw.set_hash_target(target.block.new_state_root, block_witness.new_state_root)?;

        pw.set_hash_target(
            target.block.new_account_delta_tree_root,
            block_witness.new_account_delta_tree_root,
        )?;

        pw.set_hash_target(
            target.block.old_account_delta_tree_root,
            block_witness.old_account_delta_tree_root,
        )?;

        pw.set_target(
            target.block.priority_operations_count,
            F::from_canonical_u64(block_witness.priority_operations_count),
        )?;
        for i in 0..KECCAK_HASH_OUT_BYTE_SIZE {
            pw.set_target(
                target.block.old_prefix_priority_operation_hash[i].0,
                F::from_canonical_u8(block_witness.old_prefix_priority_operation_hash[i]),
            )?;
            pw.set_target(
                target.block.new_prefix_priority_operation_hash[i].0,
                F::from_canonical_u8(block_witness.new_prefix_priority_operation_hash[i]),
            )?;
        }

        pw.set_target(
            target.block.on_chain_operations_count,
            F::from_canonical_u64(block_witness.on_chain_operations_count),
        )?;
        target
            .block
            .on_chain_operations_pub_data
            .iter()
            .zip_eq(block_witness.on_chain_operations_pub_data.iter())
            .try_for_each(|(a, b)| {
                a.iter()
                    .zip_eq(b.iter())
                    .try_for_each(|(&a, &b)| pw.set_target(a.0, F::from_canonical_u8(b)))
            })?;

        // At least one tx per block is must. If block only has pre-exec, then the only tx is the empty tx
        assert!(!block.txs.is_empty());
        target
            .block
            .new_public_market_details
            .iter()
            .zip_eq(block.new_public_market_details.iter())
            .try_for_each(|(t, mi)| pw.set_public_market_details_target(t, mi))?;

        Ok(pw)
    }

    fn prove(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockCircuit", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(target, block, pre_exec_proof, tx_chain_proof)?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    fn prove_and_compress(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<CompressedProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockCircuit", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(target, block, pre_exec_proof, tx_chain_proof)?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });
        let compressed_proof = timed!(timing, "compress", { circuit_data.compress(proof)? });
        timed!(timing, "verify_compressed", {
            circuit_data.verify_compressed(compressed_proof.clone())?
        });

        timing.print();

        Ok(compressed_proof)
    }
}
