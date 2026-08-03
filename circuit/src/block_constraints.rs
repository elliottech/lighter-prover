// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use itertools::Itertools;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData, CommonCircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::{
    CompressedProofWithPublicInputs, ProofWithPublicInputs, ProofWithPublicInputsTarget,
};
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::bigint::big_u16::CircuitBuilderBigIntU16;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::block::{Block, BlockWitness};
use crate::block_pre_execution::BlockPreExecWitnessTarget;
use crate::block_tx::JumpStateTarget;
use crate::block_tx_chain::BlockTxChainWitnessTarget;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::ecdsa::gadgets::ecdsa::{
    CircuitBuilderECDSAPublicKey, CircuitBuilderECDSASignature, conditional_verify_ecdsa_sig,
};
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::keccak::keccak::CircuitBuilderKeccak;
use crate::nonnative::CircuitBuilderNonNative;
use crate::poseidon2::Poseidon2Hash;
use crate::recursion::block_witness::BlockWitnessTarget;
use crate::types::config::{Builder, C, D, F};
use crate::types::constants::{KECCAK_HASH_OUT_BYTE_SIZE, POSITION_LIST_SIZE};
use crate::types::market_details::{
    MarketRiskDetailsTarget, PublicMarketDetailsTarget, PublicMarketDetailsWitness,
    all_public_market_details_hash,
};
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
        block_light_tx_chain_circuit: &CircuitData<F, C, D>,
        block_heavy_tx_chain_circuit: &CircuitData<F, C, D>,
        on_chain_operations_limit: usize,
    ) -> Self;

    /// Fills partial witness for batch target with given block data
    fn generate_witness(
        target: &BlockTarget,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        light_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
        heavy_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>>;

    fn prove(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        light_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
        heavy_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;

    fn prove_and_compress(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        light_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
        heavy_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
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
    pub light_tx_chain_proof: ProofWithPublicInputsTarget<D>, // proof of light tx chain execution
    pub heavy_tx_chain_proof: ProofWithPublicInputsTarget<D>, // proof of heavy tx chain execution

    pub block: BlockWitnessTarget, // Public block witness

    // Private witness: raw public market details after the block.
    pub new_market_risk_details_partial: [MarketRiskDetailsTarget; POSITION_LIST_SIZE],
}

impl BlockCircuit {
    pub fn new(
        config: CircuitConfig,
        block_pre_exec_common_circuit: &CommonCircuitData<F, D>,
        block_light_tx_chain_common_circuit: &CommonCircuitData<F, D>,
        block_heavy_tx_chain_common_circuit: &CommonCircuitData<F, D>,
        on_chain_operations_limit: usize,
    ) -> Self {
        let mut builder = Builder::new(config);

        let new_market_risk_details_partial: [MarketRiskDetailsTarget; POSITION_LIST_SIZE] = (0
            ..POSITION_LIST_SIZE)
            .map(|_| MarketRiskDetailsTarget::new_partial(&mut builder))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let new_public_market_details: [PublicMarketDetailsTarget; POSITION_LIST_SIZE] =
            new_market_risk_details_partial
                .iter()
                .map(|market| PublicMarketDetailsTarget {
                    funding_rate_prefix_sum: builder
                        .bigint_u16_to_bigint(&market.funding_rate_prefix_sum),
                    mark_price: market.mark_price,
                    quote_multiplier: market.quote_multiplier,
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

        // Register public inputs
        let block = BlockWitnessTarget::new_public(
            &mut builder,
            on_chain_operations_limit,
            new_public_market_details,
        );

        Self {
            target: BlockTarget {
                block,
                new_market_risk_details_partial,
                pre_exec_proof: builder.add_virtual_proof_with_pis(block_pre_exec_common_circuit),
                light_tx_chain_proof: builder
                    .add_virtual_proof_with_pis(block_light_tx_chain_common_circuit),
                heavy_tx_chain_proof: builder
                    .add_virtual_proof_with_pis(block_heavy_tx_chain_common_circuit),
            },
            builder,
        }
    }

    fn handle_proofs(
        &mut self,
        on_chain_operations_limit: usize,
        block_pre_exec_circuit: &CircuitData<F, C, D>,
        block_light_tx_chain_circuit: &CircuitData<F, C, D>,
        block_heavy_tx_chain_circuit: &CircuitData<F, C, D>,
    ) -> (
        BlockPreExecWitnessTarget,
        BlockTxChainWitnessTarget,
        BlockTxChainWitnessTarget,
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

        // Verify light tx chain proof
        let block_light_tx_chain_verifier_data = self
            .builder
            .constant_verifier_data(&block_light_tx_chain_circuit.verifier_only);
        self.builder.verify_proof::<C>(
            &self.target.light_tx_chain_proof,
            &block_light_tx_chain_verifier_data,
            &block_light_tx_chain_circuit.common,
        );

        let light_tx_chain_embedded_vk = self.builder.extract_verifier_data_from_public_inputs(
            &self.target.light_tx_chain_proof.public_inputs,
            &block_light_tx_chain_circuit.common,
        );
        self.builder.connect_verifier_data(
            &light_tx_chain_embedded_vk,
            &block_light_tx_chain_verifier_data,
        );

        // Verify heavy tx chain proof
        let block_heavy_tx_chain_verifier_data = self
            .builder
            .constant_verifier_data(&block_heavy_tx_chain_circuit.verifier_only);
        self.builder.verify_proof::<C>(
            &self.target.heavy_tx_chain_proof,
            &block_heavy_tx_chain_verifier_data,
            &block_heavy_tx_chain_circuit.common,
        );

        let heavy_tx_chain_embedded_vk = self.builder.extract_verifier_data_from_public_inputs(
            &self.target.heavy_tx_chain_proof.public_inputs,
            &block_heavy_tx_chain_circuit.common,
        );
        self.builder.connect_verifier_data(
            &heavy_tx_chain_embedded_vk,
            &block_heavy_tx_chain_verifier_data,
        );

        // Extract pre-exec and tx chain witnesses from the proofs
        let pre_exec_witness = BlockPreExecWitnessTarget::from_public_inputs(
            &self.target.pre_exec_proof.public_inputs,
        );
        let (light_tx_chain_witness, _) = BlockTxChainWitnessTarget::from_public_inputs(
            &self.target.light_tx_chain_proof.public_inputs,
            on_chain_operations_limit,
            1,
        );
        let (heavy_tx_chain_witness, _) = BlockTxChainWitnessTarget::from_public_inputs(
            &self.target.heavy_tx_chain_proof.public_inputs,
            on_chain_operations_limit,
            1,
        );

        (
            pre_exec_witness,
            light_tx_chain_witness,
            heavy_tx_chain_witness,
        )
    }

    fn perform_sanity_checks(
        &mut self,
        pre_exec_witness: &BlockPreExecWitnessTarget,
        light_tx_chain_witness: &BlockTxChainWitnessTarget,
        heavy_tx_chain_witness: &BlockTxChainWitnessTarget,
    ) {
        // Connect pre exec output with chains
        self.builder.connect(
            pre_exec_witness.block_number,
            light_tx_chain_witness.block_number,
        );
        self.builder.connect(
            pre_exec_witness.created_at,
            light_tx_chain_witness.created_at,
        );
        self.builder.connect(
            pre_exec_witness.block_number,
            heavy_tx_chain_witness.block_number,
        );
        self.builder.connect(
            pre_exec_witness.created_at,
            heavy_tx_chain_witness.created_at,
        );
        for chain_witness in [light_tx_chain_witness, heavy_tx_chain_witness] {
            self.builder.connect_hashes(
                chain_witness.initial_state_root,
                pre_exec_witness.new_state_root,
            );
            self.builder.connect_hashes(
                chain_witness.initial_account_delta_tree_root,
                self.target.block.old_account_delta_tree_root,
            );
        }

        // Constrain light path to not carry l1 messages, priority operations, or on-chain operations.
        self.builder
            .assert_zero(light_tx_chain_witness.on_chain_operations_count);
        self.builder
            .assert_zero(light_tx_chain_witness.priority_operations_count);
        self.builder
            .assert_zero(light_tx_chain_witness.change_pub_key_message.account_index);
        self.builder
            .assert_zero(light_tx_chain_witness.transfer_message.from_account_index);
        self.builder.assert_zero(
            light_tx_chain_witness
                .approve_integrator_message
                .account_index,
        );
    }

    fn finalize_jump(
        &mut self,
        jump: &JumpStateTarget,
        total_tx_count: Target,
        new_state_root: HashOutTarget,
        new_delta_root: HashOutTarget,
    ) -> (HashOutTarget, HashOutTarget) {
        let one = self.builder.one();
        let prev_plus_one = self.builder.add(jump.last_active_tx_index, one);

        let run_nonempty = self
            .builder
            .is_not_equal(jump.last_active_tx_index, jump.run_start_prev_index);
        let mut coverage_record_in = vec![jump.run_start_prev_index, prev_plus_one];
        coverage_record_in.extend(jump.run_start_old_state_root.elements);
        coverage_record_in.extend(jump.prev_new_state_root.elements);
        coverage_record_in.extend(jump.run_start_old_delta_root.elements);
        coverage_record_in.extend(jump.prev_new_delta_root.elements);
        let coverage_record = self
            .builder
            .hash_n_to_hash_no_pad::<Poseidon2Hash>(coverage_record_in);
        let folded_coverage = self
            .builder
            .hash_two_to_one(&jump.coverage_hash, &coverage_record);
        let coverage_hash =
            self.builder
                .select_hash(run_nonempty, &folded_coverage, &jump.coverage_hash);

        let at_end = self.builder.is_equal(prev_plus_one, total_tx_count);
        let trailing = self.builder.not(at_end);
        let mut claim_record_in = vec![jump.last_active_tx_index, total_tx_count];
        claim_record_in.extend(jump.prev_new_state_root.elements);
        claim_record_in.extend(new_state_root.elements);
        claim_record_in.extend(jump.prev_new_delta_root.elements);
        claim_record_in.extend(new_delta_root.elements);
        let claim_record = self
            .builder
            .hash_n_to_hash_no_pad::<Poseidon2Hash>(claim_record_in);
        let folded_claims = self
            .builder
            .hash_two_to_one(&jump.claims_hash, &claim_record);
        let claims_hash = self
            .builder
            .select_hash(trailing, &folded_claims, &jump.claims_hash);

        (coverage_hash, claims_hash)
    }

    /// Closes both chains' open coverage runs and folds the trailing claim for the chain
    /// that did not observe the block's final segment. Returns `is_end_with_heavy`, true when the
    /// heavy chain contains the block's last tx, so the block outputs are taken from it.
    fn define_jump_checks(
        &mut self,
        light_tx_chain_witness: &BlockTxChainWitnessTarget,
        heavy_tx_chain_witness: &BlockTxChainWitnessTarget,
    ) -> BoolTarget {
        // Total number of active txs in the block. Since active tx indices are unique,
        // sequential and shared between the two chains, the chain that processed the
        // block's last active tx must sit at index `total_tx_count - 1`.
        let total_tx_count = self.builder.add(
            light_tx_chain_witness.jump.tx_count,
            heavy_tx_chain_witness.jump.tx_count,
        );

        // The chain whose last processed tx is the block's last tx is the one that ended
        // the block; the block's final roots are its final roots. The other chain's
        // trailing claim (folded in finalize_jump) binds its final roots to them.
        let heavy_jump = heavy_tx_chain_witness.jump;
        let light_jump = light_tx_chain_witness.jump;

        // At least one chain must sit at the block's end. Padding (empty) txs never get an
        // index or increment `tx_count`, so a block with no active txs has
        // `total_tx_count == 0` and both chains at the `-1` sentinel, satisfying both
        // predicates; hence an OR rather than an exactly-one check.
        let heavy_prev_plus_one = self.builder.add_one(heavy_jump.last_active_tx_index);
        let is_end_with_heavy = self.builder.is_equal(heavy_prev_plus_one, total_tx_count);
        let light_prev_plus_one = self.builder.add_one(light_jump.last_active_tx_index);
        let is_end_with_light = self.builder.is_equal(light_prev_plus_one, total_tx_count);
        let some_chain_at_end = self.builder.or(is_end_with_heavy, is_end_with_light);
        self.builder.assert_true(some_chain_at_end);

        // Capture block's new roots
        let new_state_root = self.builder.select_hash(
            is_end_with_heavy,
            &heavy_tx_chain_witness.new_state_root,
            &light_tx_chain_witness.new_state_root,
        );
        let new_delta_root = self.builder.select_hash(
            is_end_with_heavy,
            &heavy_tx_chain_witness.new_account_delta_tree_root,
            &light_tx_chain_witness.new_account_delta_tree_root,
        );

        // The ending chain's exposed roots must be the roots of the block's last active tx
        // (the anchored initial roots if the block has no active txs), so trailing padding
        // txs cannot substitute an arbitrary valid state as the block's final state.
        let end_jump_state_root = self.builder.select_hash(
            is_end_with_heavy,
            &heavy_jump.prev_new_state_root,
            &light_jump.prev_new_state_root,
        );
        self.builder
            .connect_hashes(new_state_root, end_jump_state_root);
        let end_jump_delta_root = self.builder.select_hash(
            is_end_with_heavy,
            &heavy_jump.prev_new_delta_root,
            &light_jump.prev_new_delta_root,
        );
        self.builder
            .connect_hashes(new_delta_root, end_jump_delta_root);

        // Finalize chains and assert equality
        let (heavy_coverage, heavy_claims) =
            self.finalize_jump(&heavy_jump, total_tx_count, new_state_root, new_delta_root);
        let (light_coverage, light_claims) =
            self.finalize_jump(&light_jump, total_tx_count, new_state_root, new_delta_root);
        self.builder.connect_hashes(heavy_coverage, light_claims);
        self.builder.connect_hashes(light_coverage, heavy_claims);

        is_end_with_heavy
    }
}

impl Circuit<C, F, D> for BlockCircuit {
    fn define(
        config: CircuitConfig,
        block_pre_exec_circuit: &CircuitData<F, C, D>,
        block_light_tx_chain_circuit: &CircuitData<F, C, D>,
        block_heavy_tx_chain_circuit: &CircuitData<F, C, D>,
        on_chain_operations_limit: usize,
    ) -> Self {
        let mut circuit = Self::new(
            config,
            &block_pre_exec_circuit.common,
            &block_light_tx_chain_circuit.common,
            &block_heavy_tx_chain_circuit.common,
            on_chain_operations_limit,
        );

        let (pre_exec_witness, light_tx_chain_witness, heavy_tx_chain_witness) = circuit
            .handle_proofs(
                on_chain_operations_limit,
                block_pre_exec_circuit,
                block_light_tx_chain_circuit,
                block_heavy_tx_chain_circuit,
            );

        circuit.perform_sanity_checks(
            &pre_exec_witness,
            &light_tx_chain_witness,
            &heavy_tx_chain_witness,
        );

        let is_end_with_heavy =
            circuit.define_jump_checks(&light_tx_chain_witness, &heavy_tx_chain_witness);

        // Validate required ECDSA signatures for various txs. Only *heavy* path can have ECDSA signature,
        // so we are only checking *heavy_tx_chain_witness*
        // Calculate new priority operations hash
        let is_priority_operations_exists = circuit
            .builder
            .is_not_zero(heavy_tx_chain_witness.priority_operations_count);
        let keccak_input: Vec<U8Target> = circuit
            .target
            .block
            .old_prefix_priority_operation_hash
            .iter()
            .chain(heavy_tx_chain_witness.priority_operations_pub_data.iter())
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
            .is_not_zero(heavy_tx_chain_witness.change_pub_key_message.account_index);
        let change_pub_key_message = heavy_tx_chain_witness
            .change_pub_key_message
            .get_change_pub_key_l1_signature_msg_hash(&mut circuit.builder);
        let (mut l1_message, mut l1_address, mut l1_signature, mut l1_pk) = (
            change_pub_key_message,
            heavy_tx_chain_witness.change_pub_key_message.l1_address,
            heavy_tx_chain_witness.change_pub_key_message.l1_signature,
            heavy_tx_chain_witness.change_pub_key_message.l1_pk,
        );

        let is_transfer_exists = circuit
            .builder
            .is_not_zero(heavy_tx_chain_witness.transfer_message.from_account_index);
        let transfer_message = heavy_tx_chain_witness
            .transfer_message
            .get_transfer_l1_signature_msg_hash(&mut circuit.builder);
        (l1_message, l1_address, l1_signature, l1_pk) = (
            circuit
                .builder
                .select_nonnative(is_transfer_exists, &transfer_message, &l1_message),
            circuit.builder.select_biguint(
                is_transfer_exists,
                &heavy_tx_chain_witness.transfer_message.l1_address,
                &l1_address,
            ),
            circuit.builder.select_ecdsa_signature(
                is_transfer_exists,
                &heavy_tx_chain_witness.transfer_message.l1_signature,
                &l1_signature,
            ),
            circuit.builder.select_ecdsa_public_key(
                is_transfer_exists,
                &heavy_tx_chain_witness.transfer_message.l1_pk,
                &l1_pk,
            ),
        );

        let is_approve_integrator_exists = circuit.builder.is_not_zero(
            heavy_tx_chain_witness
                .approve_integrator_message
                .account_index,
        );
        let approve_integrator_message = heavy_tx_chain_witness
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
                &heavy_tx_chain_witness.approve_integrator_message.l1_address,
                &l1_address,
            ),
            circuit.builder.select_ecdsa_signature(
                is_approve_integrator_exists,
                &heavy_tx_chain_witness
                    .approve_integrator_message
                    .l1_signature,
                &l1_signature,
            ),
            circuit.builder.select_ecdsa_public_key(
                is_approve_integrator_exists,
                &heavy_tx_chain_witness.approve_integrator_message.l1_pk,
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

        let new_validium_root = circuit.builder.select_hash(
            is_end_with_heavy,
            &heavy_tx_chain_witness.new_validium_root,
            &light_tx_chain_witness.new_validium_root,
        );
        let new_state_root = circuit.builder.select_hash(
            is_end_with_heavy,
            &heavy_tx_chain_witness.new_state_root,
            &light_tx_chain_witness.new_state_root,
        );
        let new_account_delta_tree_root = circuit.builder.select_hash(
            is_end_with_heavy,
            &heavy_tx_chain_witness.new_account_delta_tree_root,
            &light_tx_chain_witness.new_account_delta_tree_root,
        );
        let new_public_market_details_hash = circuit.builder.select_hash(
            is_end_with_heavy,
            &heavy_tx_chain_witness.new_public_market_details_hash,
            &light_tx_chain_witness.new_public_market_details_hash,
        );

        // The raw market details are a private witness; hashing them and
        // connecting to the ending chain's commitment binds every field
        let witnessed_public_market_details_hash = all_public_market_details_hash(
            &mut circuit.builder,
            &circuit.target.new_market_risk_details_partial,
        );
        circuit.builder.connect_hashes(
            witnessed_public_market_details_hash,
            new_public_market_details_hash,
        );

        let calculated_block = BlockWitnessTarget {
            block_number: pre_exec_witness.block_number,
            created_at: pre_exec_witness.created_at,
            old_state_root: pre_exec_witness.old_state_root,
            new_validium_root,
            new_state_root,

            old_account_delta_tree_root: circuit.target.block.old_account_delta_tree_root,
            new_account_delta_tree_root,

            on_chain_operations_count: heavy_tx_chain_witness.on_chain_operations_count,
            on_chain_operations_pub_data: heavy_tx_chain_witness.on_chain_operations_pub_data,
            priority_operations_count: heavy_tx_chain_witness.priority_operations_count,
            old_prefix_priority_operation_hash: circuit
                .target
                .block
                .old_prefix_priority_operation_hash,
            new_prefix_priority_operation_hash: new_priority_operations_hash,
            new_public_market_details: circuit.target.block.new_public_market_details.clone(),
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
        light_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
        heavy_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.pre_exec_proof, pre_exec_proof)?;
        pw.set_proof_with_pis_target(&target.light_tx_chain_proof, light_tx_chain_proof)?;
        pw.set_proof_with_pis_target(&target.heavy_tx_chain_proof, heavy_tx_chain_proof)?;

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
        assert!(!block.tx_chunks.is_empty());
        target
            .new_market_risk_details_partial
            .iter()
            .zip_eq(block.new_public_market_details.iter())
            .try_for_each(|(t, mi)| pw.set_partial_market_risk_details_target(t, mi))?;

        Ok(pw)
    }

    fn prove(
        target: &BlockTarget,
        circuit_data: &CircuitData<F, C, D>,
        block: &Block<F>,
        pre_exec_proof: &ProofWithPublicInputs<F, C, D>,
        light_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
        heavy_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockCircuit", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(
                target,
                block,
                pre_exec_proof,
                light_tx_chain_proof,
                heavy_tx_chain_proof,
            )?
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
        light_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
        heavy_tx_chain_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<CompressedProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockCircuit", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(
                target,
                block,
                pre_exec_proof,
                light_tx_chain_proof,
                heavy_tx_chain_proof,
            )?
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
