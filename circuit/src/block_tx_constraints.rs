// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use itertools::Itertools;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::block_tx::{BlockTx, JumpStateTarget};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::tx_constraints::{TxTarget, TxTargetWitness};
use crate::types::approve_integrator::ApproveIntegratorMessageTarget;
use crate::types::change_pub_key::ChangePubKeyMessageTarget;
use crate::types::config::{Builder, C, D, F};
use crate::types::constants::*;
use crate::types::transfer::TransferMessageTarget;
use crate::uint::u8::{CircuitBuilderU8, U8Target};
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
    /// `target` can be used to assign partial witness in [`BlockTxCircuit::prove()`] function
    fn define(config: CircuitConfig, tx_limit: usize, chain_id: u32, mode: u8) -> Self;
    /// Fills partial witness for block target with given block data
    fn generate_witness(block: &BlockTx<F>, target: &BlockTxTarget) -> Result<PartialWitness<F>>;
    /// Takes `circuit`, block witness and `target` defined in [`BlockTxCircuit::define()`] function
    /// and returns the (not-compressed) proof with public inputs
    fn prove(
        circuit: &CircuitData<F, C, D>,
        block: &BlockTx<F>,
        bt: &BlockTxTarget,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;
}

#[derive(Debug)]
pub struct BlockTxCircuit {
    pub builder: Builder,
    pub target: BlockTxTarget,
}

#[derive(Debug)]
pub struct BlockTxTarget {
    pub created_at: Target, // 48 bits

    /***************************/
    /*  NEW COMMON STATE DATA  */
    /***************************/
    // Hash of the public (pubdata) part of all market details after the batch.
    // Committed to the state root inside each tx; the block circuit opens it.
    pub public_market_details_hash_after: HashOutTarget,

    /**************************/
    /*  NEW STATE TREE ROOTS  */
    /**************************/
    pub state_metadata_hash: HashOutTarget,
    pub new_validium_root: HashOutTarget,
    pub new_state_root: HashOutTarget,

    /*******************************/
    /*  ECDSA SIGNED MESSAGE DATA  */
    /*******************************/
    pub change_pub_key_message: ChangePubKeyMessageTarget,
    pub transfer_message: TransferMessageTarget,
    pub approve_integrator_message: ApproveIntegratorMessageTarget,

    // /*************************/
    // /*       PUB DATA        */
    // /*************************/
    pub new_account_delta_tree_root: HashOutTarget,
    pub priority_operations_count: Target,
    pub priority_operations_pub_data: [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    pub on_chain_operations_count: Target,
    pub on_chain_operations_pub_data: [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],

    /******************/
    /*  TRANSACTIONS  */
    /******************/
    pub txs: Vec<TxTarget>,

    pub old_jump: JumpStateTarget,
    pub new_jump: JumpStateTarget,
}

impl Default for BlockTxTarget {
    fn default() -> Self {
        Self {
            created_at: Target::default(),

            state_metadata_hash: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },

            public_market_details_hash_after: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },

            new_validium_root: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },
            new_state_root: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },

            change_pub_key_message: ChangePubKeyMessageTarget::default(),
            transfer_message: TransferMessageTarget::default(),
            approve_integrator_message: ApproveIntegratorMessageTarget::default(),

            new_account_delta_tree_root: HashOutTarget {
                elements: core::array::from_fn(|_| Target::default()),
            },
            priority_operations_count: Target::default(),
            priority_operations_pub_data: core::array::from_fn(|_| U8Target::default()),
            on_chain_operations_count: Target::default(),
            on_chain_operations_pub_data: core::array::from_fn(|_| U8Target::default()),

            txs: vec![],

            old_jump: JumpStateTarget::default(),
            new_jump: JumpStateTarget::default(),
        }
    }
}

impl Circuit<C, F, D> for BlockTxCircuit {
    fn define(config: CircuitConfig, tx_limit: usize, chain_id: u32, mode: u8) -> Self {
        let mut circuit = Self::new(config, tx_limit);

        circuit.register_public_inputs();

        let (
            on_chain_operations_count,
            on_chain_operations_pub_data,
            priority_operations_count,
            priority_operations_pub_data,
            public_market_details_hash_after,
            account_delta_tree_root_after,
        ) = circuit.define_tx_loop(tx_limit, chain_id, mode);

        circuit.define_post_tx_batch(
            chain_id,
            account_delta_tree_root_after,
            public_market_details_hash_after,
            on_chain_operations_count,
            &on_chain_operations_pub_data,
            priority_operations_count,
            &priority_operations_pub_data,
            mode,
        );

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn prove(
        circuit: &CircuitData<F, C, D>,
        block: &BlockTx<F>,
        target: &BlockTxTarget,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockTxCircuit::prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(block, target)?
        });
        let proof = prove::<F, C, D>(&circuit.prover_only, &circuit.common, pw, &mut timing)?;
        timed!(timing, "verify", { circuit.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    fn generate_witness(block: &BlockTx<F>, target: &BlockTxTarget) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_target(target.created_at, F::from_canonical_i64(block.created_at))?;
        pw.set_hash_target(target.state_metadata_hash, block.state_metadata_hash)?;

        pw.set_target(
            target.old_jump.last_active_tx_index,
            block.old_jump.last_active_tx_index,
        )?;
        pw.set_hash_target(
            target.old_jump.prev_new_state_root,
            block.old_jump.prev_new_state_root,
        )?;
        pw.set_hash_target(
            target.old_jump.prev_new_delta_root,
            block.old_jump.prev_new_delta_root,
        )?;
        pw.set_target(
            target.old_jump.run_start_prev_index,
            block.old_jump.run_start_prev_index,
        )?;
        pw.set_hash_target(
            target.old_jump.run_start_old_state_root,
            block.old_jump.run_start_old_state_root,
        )?;
        pw.set_hash_target(
            target.old_jump.run_start_old_delta_root,
            block.old_jump.run_start_old_delta_root,
        )?;
        pw.set_hash_target(target.old_jump.coverage_hash, block.old_jump.coverage_hash)?;
        pw.set_hash_target(target.old_jump.claims_hash, block.old_jump.claims_hash)?;
        pw.set_target(target.old_jump.tx_count, block.old_jump.tx_count)?;

        target
            .txs
            .iter()
            .zip_eq(block.txs.iter())
            .try_for_each(|(t, tx)| pw.set_tx_target(t, tx))?;

        Ok(pw)
    }
}

impl BlockTxCircuit {
    /// Initializes a new block virtual targets for the given number of transactions.
    pub fn new(config: CircuitConfig, tx_limit: usize) -> Self {
        let mut builder = Builder::new(config);

        Self {
            target: BlockTxTarget {
                created_at: builder.add_virtual_target(),
                state_metadata_hash: builder.add_virtual_hash(),

                txs: (0..tx_limit).map(|_| TxTarget::new(&mut builder)).collect(),

                // ..Default::default() //
                // Comment out the following to avoid "generators weren't run"
                public_market_details_hash_after: builder.add_virtual_hash(),

                new_account_delta_tree_root: builder.add_virtual_hash(),

                new_validium_root: builder.add_virtual_hash(),
                new_state_root: builder.add_virtual_hash(),

                change_pub_key_message: ChangePubKeyMessageTarget::new(&mut builder),
                transfer_message: TransferMessageTarget::new(&mut builder),
                approve_integrator_message: ApproveIntegratorMessageTarget::new(&mut builder),

                priority_operations_count: builder.add_virtual_target(),
                priority_operations_pub_data: builder
                    .add_virtual_u8_targets_unsafe(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX)
                    .try_into()
                    .unwrap(), // safe because it is connected to output of split_bytes which are range-checked

                on_chain_operations_count: builder.add_virtual_target(),
                on_chain_operations_pub_data: builder
                    .add_virtual_u8_targets_unsafe(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE)
                    .try_into()
                    .unwrap(), // safe because it is connected to output of split_bytes which are range-checked

                old_jump: JumpStateTarget::new(&mut builder),
                new_jump: JumpStateTarget::new(&mut builder),
            },

            builder,
        }
    }

    fn register_public_inputs(&mut self) {
        self.builder
            .register_public_hashout(self.target.new_account_delta_tree_root);

        self.builder
            .register_public_hashout(self.target.new_validium_root);
        self.builder
            .register_public_hashout(self.target.new_state_root);

        self.builder
            .register_public_hashout(self.target.public_market_details_hash_after);

        // Change pub key message
        self.target
            .change_pub_key_message
            .register_public_input(&mut self.builder);
        // Transfer message
        self.target
            .transfer_message
            .register_public_input(&mut self.builder);
        // Approve integrator message
        self.target
            .approve_integrator_message
            .register_public_input(&mut self.builder);

        // On chain ops pub data
        self.builder
            .register_public_input(self.target.on_chain_operations_count);
        self.target
            .on_chain_operations_pub_data
            .iter()
            .for_each(|&byte| {
                self.builder.register_public_u8_input(byte);
            });

        // Priority ops pub data
        self.builder
            .register_public_input(self.target.priority_operations_count);
        self.target
            .priority_operations_pub_data
            .iter()
            .for_each(|&byte| {
                self.builder.register_public_u8_input(byte);
            });

        self.target
            .old_jump
            .register_public_inputs(&mut self.builder);
        self.target
            .new_jump
            .register_public_inputs(&mut self.builder);
    }

    fn define_tx_loop(
        &mut self,
        tx_limit: usize,
        chain_id: u32,
        mode: u8,
    ) -> (
        Target,                                                    // on chain operations count
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE], // on chain operations public data
        Target,                                              // priority operations count
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX], // priority operations public data
        HashOutTarget,                                             // new public market details hash
        HashOutTarget,                                             // new account delta tree root
    ) {
        assert_eq!(self.target.txs.len(), tx_limit, "txs count mismatch");
        assert!(
            !self.target.txs.is_empty(),
            "block must contain at least one tx (including empty tx)"
        );

        let mut on_chain_operations_count = self.builder.zero();
        let mut on_chain_operations_pub_data =
            [self.builder.zero_u8(); ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE];

        let mut priority_operations_count = self.builder.zero();
        let mut priority_operations_pub_data =
            [self.builder.zero_u8(); MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX];

        self.builder
            .register_range_check(self.target.created_at, TIMESTAMP_BITS);

        let mut current_account_delta_tree_root = self.builder.zero_hash_out();
        let mut public_market_details_hash_after = self.builder.zero_hash_out();

        let mut jump = self.target.old_jump;

        for (index, tx) in self.target.txs.iter_mut().enumerate() {
            let (
                tx_priority_operations_pub_data,
                priority_operations_pub_data_exists,
                tx_on_chain_operations_pub_data,
                on_chain_pub_data_exists,
                tx_public_market_details_hash_after,
                account_delta_tree_root_after,
            ) = match mode {
                TX_HEAVY => tx.define(
                    index,
                    chain_id,
                    &mut self.builder,
                    self.target.created_at,
                    self.target.state_metadata_hash,
                ),
                TX_LIGHT => tx.define_light(
                    index,
                    chain_id,
                    &mut self.builder,
                    self.target.created_at,
                    self.target.state_metadata_hash,
                ),
                _ => panic!("unknown tx circuit mode: {mode}"),
            };

            current_account_delta_tree_root = account_delta_tree_root_after;
            public_market_details_hash_after = tx_public_market_details_hash_after;

            jump = Self::define_jump_step(
                &mut self.builder,
                jump,
                tx.tx_index,
                tx.tx_type,
                tx.old_state_root,
                tx.new_state_root,
                tx.old_account_delta_tree_root,
                account_delta_tree_root_after,
            );

            if mode == TX_HEAVY {
                on_chain_operations_count = self
                    .builder
                    .add(on_chain_operations_count, on_chain_pub_data_exists.target);
                on_chain_operations_pub_data = self.builder.select_arr_u8(
                    on_chain_pub_data_exists,
                    &tx_on_chain_operations_pub_data,
                    &on_chain_operations_pub_data,
                );
                priority_operations_count = self.builder.add(
                    priority_operations_count,
                    priority_operations_pub_data_exists.target,
                );
                priority_operations_pub_data = self.builder.select_arr_u8(
                    priority_operations_pub_data_exists,
                    &tx_priority_operations_pub_data,
                    &priority_operations_pub_data,
                );
            }
        }

        JumpStateTarget::connect(&mut self.builder, &jump, &self.target.new_jump);

        (
            on_chain_operations_count,
            on_chain_operations_pub_data,
            priority_operations_count,
            priority_operations_pub_data,
            public_market_details_hash_after,
            current_account_delta_tree_root,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn define_jump_step(
        builder: &mut Builder,
        jump: JumpStateTarget,
        tx_index: Target,
        tx_type: Target,
        tx_old_state_root: HashOutTarget,
        tx_new_state_root: HashOutTarget,
        tx_old_delta_root: HashOutTarget,
        tx_new_delta_root: HashOutTarget,
    ) -> JumpStateTarget {
        let one = builder.one();

        let is_padding = builder.is_equal(tx_index, jump.last_active_tx_index);
        let is_active = builder.not(is_padding);
        let is_empty = builder.is_zero(tx_type);
        builder.connect(is_empty.target, is_padding.target);

        let gap = builder.sub(tx_index, jump.last_active_tx_index);
        let effective_gap = builder.select(is_padding, one, gap);
        let gap_minus_one = builder.sub(effective_gap, one);
        builder.register_range_check(gap_minus_one, 32);
        let has_gap = builder.is_not_zero(gap_minus_one);
        let jumped = builder.and(is_active, has_gap);

        let continuous = builder.and_not(is_active, has_gap);
        for i in 0..NUM_HASH_OUT_ELTS {
            builder.conditional_assert_eq(
                continuous,
                tx_old_state_root.elements[i],
                jump.prev_new_state_root.elements[i],
            );
            builder.conditional_assert_eq(
                continuous,
                tx_old_delta_root.elements[i],
                jump.prev_new_delta_root.elements[i],
            );
        }

        let run_empty = builder.is_equal(jump.last_active_tx_index, jump.run_start_prev_index);
        let run_nonempty = builder.not(run_empty);
        let fold_coverage = builder.and(jumped, run_nonempty);
        let prev_plus_one = builder.add(jump.last_active_tx_index, one);
        let mut coverage_record_in = vec![jump.run_start_prev_index, prev_plus_one];
        coverage_record_in.extend(jump.run_start_old_state_root.elements);
        coverage_record_in.extend(jump.prev_new_state_root.elements);
        coverage_record_in.extend(jump.run_start_old_delta_root.elements);
        coverage_record_in.extend(jump.prev_new_delta_root.elements);
        let coverage_record = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(coverage_record_in);
        let folded_coverage = builder.hash_two_to_one(&jump.coverage_hash, &coverage_record);

        let mut claim_record_in = vec![jump.last_active_tx_index, tx_index];
        claim_record_in.extend(jump.prev_new_state_root.elements);
        claim_record_in.extend(tx_old_state_root.elements);
        claim_record_in.extend(jump.prev_new_delta_root.elements);
        claim_record_in.extend(tx_old_delta_root.elements);
        let claim_record = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(claim_record_in);
        let folded_claims = builder.hash_two_to_one(&jump.claims_hash, &claim_record);

        let tx_index_minus_one = builder.sub(tx_index, one);

        JumpStateTarget {
            last_active_tx_index: builder.select(is_active, tx_index, jump.last_active_tx_index),
            prev_new_state_root: builder.select_hash(
                is_active,
                &tx_new_state_root,
                &jump.prev_new_state_root,
            ),
            prev_new_delta_root: builder.select_hash(
                is_active,
                &tx_new_delta_root,
                &jump.prev_new_delta_root,
            ),
            run_start_prev_index: builder.select(
                jumped,
                tx_index_minus_one,
                jump.run_start_prev_index,
            ),
            run_start_old_state_root: builder.select_hash(
                jumped,
                &tx_old_state_root,
                &jump.run_start_old_state_root,
            ),
            run_start_old_delta_root: builder.select_hash(
                jumped,
                &tx_old_delta_root,
                &jump.run_start_old_delta_root,
            ),
            coverage_hash: builder.select_hash(
                fold_coverage,
                &folded_coverage,
                &jump.coverage_hash,
            ),
            claims_hash: builder.select_hash(jumped, &folded_claims, &jump.claims_hash),
            tx_count: builder.add(jump.tx_count, is_active.target),
        }
    }

    fn define_post_tx_batch(
        &mut self,
        chain_id: u32,
        new_account_delta_tree_root: HashOutTarget,
        public_market_details_hash_after: HashOutTarget,
        on_chain_operations_count: Target,
        on_chain_operations_pub_data: &[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
        priority_operations_count: Target,
        priority_operations_pub_data: &[U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
        mode: u8,
    ) {
        match mode {
            TX_HEAVY => {
                self.handle_change_pub_key();
                self.handle_transfer(chain_id);
                self.handle_approve_integrator(chain_id);
            }
            TX_LIGHT => {
                self.handle_zero_change_pub_key();
                self.handle_zero_transfer();
                self.handle_zero_approve_integrator();
            }
            _ => panic!("unknown tx circuit mode: {}", mode),
        }

        // Light txs produce no pub data, so these connect the targets to zeros.
        self.handle_on_chain_pub_data(on_chain_operations_count, on_chain_operations_pub_data);
        self.handle_priority_operation_pub_data(
            priority_operations_count,
            priority_operations_pub_data,
        );

        self.builder.connect_hashes(
            new_account_delta_tree_root,
            self.target.new_account_delta_tree_root,
        );

        let last_tx = self
            .target
            .txs
            .last()
            .expect("block must contain at least one tx");
        self.builder
            .connect_hashes(self.target.new_validium_root, last_tx.new_validium_root);
        self.builder
            .connect_hashes(self.target.new_state_root, last_tx.new_state_root);

        self.builder.connect_hashes(
            self.target.public_market_details_hash_after,
            public_market_details_hash_after,
        );
    }

    fn handle_zero_change_pub_key(&mut self) {
        let change_pub_key_message = ChangePubKeyMessageTarget::empty(&mut self.builder);
        ChangePubKeyMessageTarget::connect(
            &mut self.builder,
            &change_pub_key_message,
            &self.target.change_pub_key_message,
        );
    }

    fn handle_change_pub_key(&mut self) {
        let l2_change_pk = self.builder.constant_from_u8(TX_TYPE_L2_CHANGE_PUB_KEY);
        let mut count = self.builder.zero();

        let mut change_pub_key_message = ChangePubKeyMessageTarget::empty(&mut self.builder);

        for tx in self.target.txs.iter() {
            let is_change_pk = self.builder.is_equal(tx.tx_type, l2_change_pk);

            count = self.builder.add(is_change_pk.target, count);

            change_pub_key_message = ChangePubKeyMessageTarget::select(
                &mut self.builder,
                is_change_pk,
                &ChangePubKeyMessageTarget {
                    l1_address: tx.accounts_before[OWNER_ACCOUNT_ID].l1_address.clone(),
                    account_index: tx.l2_change_pub_key_tx_target.inner.account_index,
                    api_key_index: tx.l2_change_pub_key_tx_target.inner.api_key_index,
                    pub_key: tx.l2_change_pub_key_tx_target.inner.pub_key,
                    nonce: tx.nonce,
                    l1_pk: tx.l1_pub_key.clone(),
                    l1_signature: tx.l1_signature.clone(),
                },
                &change_pub_key_message,
            );
        }

        // Verify that there is at most one change pubkey tx in the block
        let c_sq_minus_c = self.builder.mul_sub(count, count, count);
        self.builder.assert_zero(c_sq_minus_c);

        ChangePubKeyMessageTarget::connect(
            &mut self.builder,
            &change_pub_key_message,
            &self.target.change_pub_key_message,
        );
    }

    fn handle_zero_transfer(&mut self) {
        let transfer_message = TransferMessageTarget::empty(&mut self.builder);
        TransferMessageTarget::connect(
            &mut self.builder,
            &transfer_message,
            &self.target.transfer_message,
        );
    }

    fn handle_transfer(&mut self, chain_id: u32) {
        let l2_transfer = self.builder.constant_from_u8(TX_TYPE_L2_TRANSFER);
        let mut count = self.builder.zero();

        let mut transfer_message = TransferMessageTarget::empty(&mut self.builder);

        let chain_id = self.builder.constant(F::from_canonical_u32(chain_id));
        for tx in self.target.txs.iter() {
            let is_transfer = self.builder.is_equal(tx.tx_type, l2_transfer);
            let is_same_master_account = self.builder.is_equal(
                tx.accounts_before[0].master_account_index,
                tx.accounts_before[1].master_account_index,
            );
            let is_transfer_different_master_account =
                self.builder.and_not(is_transfer, is_same_master_account);

            count = self
                .builder
                .add(is_transfer_different_master_account.target, count);

            transfer_message = TransferMessageTarget::select(
                &mut self.builder,
                is_transfer_different_master_account,
                &TransferMessageTarget {
                    from_account_index: tx.l2_transfer_tx_target.inner.from_account_index,
                    api_key_index: tx.l2_transfer_tx_target.inner.api_key_index,
                    to_account_index: tx.l2_transfer_tx_target.inner.to_account_index,
                    from_route_type: tx.l2_transfer_tx_target.inner.from_route_type,
                    to_route_type: tx.l2_transfer_tx_target.inner.to_route_type,
                    asset_index: tx.l2_transfer_tx_target.inner.asset_index,
                    chain_id,
                    nonce: tx.nonce,
                    amount: tx.l2_transfer_tx_target.inner.amount.clone(),
                    fee: tx.l2_transfer_tx_target.inner.usdc_fee.clone(),
                    memo: tx.l2_transfer_tx_target.inner.memo,
                    l1_address: tx.accounts_before[OWNER_ACCOUNT_ID].l1_address.clone(),
                    l1_pk: tx.l1_pub_key.clone(),
                    l1_signature: tx.l1_signature.clone(),
                },
                &transfer_message,
            );
        }

        // Verify that there is at most one transfer message in the block
        let c_sq_minus_c = self.builder.mul_sub(count, count, count);
        self.builder.assert_zero(c_sq_minus_c);

        TransferMessageTarget::connect(
            &mut self.builder,
            &transfer_message,
            &self.target.transfer_message,
        );
    }

    fn handle_zero_approve_integrator(&mut self) {
        let approve_integrator_message = ApproveIntegratorMessageTarget::empty(&mut self.builder);
        ApproveIntegratorMessageTarget::connect(
            &mut self.builder,
            &approve_integrator_message,
            &self.target.approve_integrator_message,
        );
    }

    fn handle_approve_integrator(&mut self, chain_id: u32) {
        let l2_approve_integrator = self.builder.constant_from_u8(TX_TYPE_L2_APPROVE_INTEGRATOR);
        let mut count = self.builder.zero();

        let mut approve_integrator_message =
            ApproveIntegratorMessageTarget::empty(&mut self.builder);

        let chain_id = self.builder.constant(F::from_canonical_u32(chain_id));
        for tx in self.target.txs.iter() {
            let is_approve_integrator = self.builder.is_equal(tx.tx_type, l2_approve_integrator);
            let is_same_master_account = self.builder.is_equal(
                tx.accounts_before[0].master_account_index,
                tx.accounts_before[1].master_account_index,
            );
            // 0-fee integrators skip L1 signature check (like same master account)
            let fees_added = self.builder.add_many([
                tx.l2_approve_integrator_tx_target.inner.max_perps_taker_fee,
                tx.l2_approve_integrator_tx_target.inner.max_perps_maker_fee,
                tx.l2_approve_integrator_tx_target.inner.max_spot_taker_fee,
                tx.l2_approve_integrator_tx_target.inner.max_spot_maker_fee,
            ]);
            let no_fees = self.builder.is_zero(fees_added);
            let skip_l1_signature = self.builder.or(is_same_master_account, no_fees);
            let needs_l1_signature = self
                .builder
                .and_not(is_approve_integrator, skip_l1_signature);

            count = self.builder.add(needs_l1_signature.target, count);

            approve_integrator_message = ApproveIntegratorMessageTarget::select(
                &mut self.builder,
                needs_l1_signature,
                &ApproveIntegratorMessageTarget {
                    account_index: tx.l2_approve_integrator_tx_target.inner.account_index,
                    api_key_index: tx.l2_approve_integrator_tx_target.inner.api_key_index,

                    integrator_account_index: tx
                        .l2_approve_integrator_tx_target
                        .inner
                        .integrator_account_index,
                    max_perps_taker_fee: tx
                        .l2_approve_integrator_tx_target
                        .inner
                        .max_perps_taker_fee,
                    max_perps_maker_fee: tx
                        .l2_approve_integrator_tx_target
                        .inner
                        .max_perps_maker_fee,
                    max_spot_taker_fee: tx.l2_approve_integrator_tx_target.inner.max_spot_taker_fee,
                    max_spot_maker_fee: tx.l2_approve_integrator_tx_target.inner.max_spot_maker_fee,
                    approval_expiry: tx.l2_approve_integrator_tx_target.inner.approval_expiry,

                    nonce: tx.nonce,
                    chain_id,

                    l1_address: tx.accounts_before[OWNER_ACCOUNT_ID].l1_address.clone(),
                    l1_pk: tx.l1_pub_key.clone(),
                    l1_signature: tx.l1_signature.clone(),
                },
                &approve_integrator_message,
            );
        }

        // Verify that there is at most one approve_integrator message in the block
        let c_sq_minus_c = self.builder.mul_sub(count, count, count);
        self.builder.assert_zero(c_sq_minus_c);

        ApproveIntegratorMessageTarget::connect(
            &mut self.builder,
            &approve_integrator_message,
            &self.target.approve_integrator_message,
        );
    }

    fn handle_on_chain_pub_data(
        &mut self,
        on_chain_operations_count: Target,
        on_chain_operations_pub_data: &[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    ) {
        self.builder.connect(
            on_chain_operations_count,
            self.target.on_chain_operations_count,
        );
        // Connect calculated on chain pub data to block witness
        on_chain_operations_pub_data
            .iter()
            .zip_eq(self.target.on_chain_operations_pub_data.iter())
            .for_each(|(&a, &b)| {
                self.builder.connect_u8(a, b);
            });
    }

    fn handle_priority_operation_pub_data(
        &mut self,
        priority_operations_count: Target,
        priority_operations_pub_data: &[U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        self.builder.connect(
            priority_operations_count,
            self.target.priority_operations_count,
        );
        // Connect calculated priority pub data to block witness
        priority_operations_pub_data
            .iter()
            .zip_eq(self.target.priority_operations_pub_data.iter())
            .for_each(|(&a, &b)| {
                self.builder.connect_u8(a, b);
            });
    }
}
