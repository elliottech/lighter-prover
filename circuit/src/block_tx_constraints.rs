// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use itertools::Itertools;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::block_tx::BlockTx;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::tx_constraints::{TxTarget, TxTargetWitness};
use crate::types::asset::{AssetTarget, AssetTargetWitness, connect_assets};
use crate::types::change_pub_key::ChangePubKeyMessageTarget;
use crate::types::config::{Builder, C, D, F};
use crate::types::constants::{
    ASSET_LIST_SIZE, MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX,
    ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE, OWNER_ACCOUNT_ID, POSITION_LIST_SIZE, TIMESTAMP_BITS,
    TX_TYPE_L2_CHANGE_PUB_KEY, TX_TYPE_L2_TRANSFER,
};
use crate::types::market_details::{
    MarketDetailsTarget, MarketDetailsWitness, connect_market_details,
};
use crate::types::register::{RegisterInfoTargetWitness, RegisterStackTarget};
use crate::types::system_config::{SystemConfigTarget, SystemConfigTargetWitness};
use crate::types::transfer::TransferMessageTarget;
use crate::uint::u8::{CircuitBuilderU8, U8Target};

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
    fn define(config: CircuitConfig, tx_limit: usize, chain_id: u32) -> Self;
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

    /***********************/
    /*  COMMON STATE DATA  */
    /***********************/
    pub old_system_config: SystemConfigTarget,
    pub register_stack_before: RegisterStackTarget,
    pub all_assets_before: [AssetTarget; ASSET_LIST_SIZE],
    pub all_market_details_before: [MarketDetailsTarget; POSITION_LIST_SIZE],

    /**************************/
    /*  OLD STATE TREE ROOTS  */
    /**************************/
    pub old_account_tree_root: HashOutTarget,
    pub old_account_pub_data_tree_root: HashOutTarget,
    pub old_market_tree_root: HashOutTarget,

    /***************************/
    /*  NEW COMMON STATE DATA  */
    /***************************/
    pub new_system_config: SystemConfigTarget,
    pub register_stack_after: RegisterStackTarget,
    pub all_assets_after: [AssetTarget; ASSET_LIST_SIZE],
    pub all_market_details_after: [MarketDetailsTarget; POSITION_LIST_SIZE],

    /**************************/
    /*  NEW STATE TREE ROOTS  */
    /**************************/
    pub new_account_tree_root: HashOutTarget,
    pub new_account_pub_data_tree_root: HashOutTarget,
    pub new_market_tree_root: HashOutTarget,

    /*******************************/
    /*  ECDSA SIGNED MESSAGE DATA  */
    /*******************************/
    pub change_pub_key_message: ChangePubKeyMessageTarget,
    pub transfer_message: TransferMessageTarget,

    // /*************************/
    // /*       PUB DATA        */
    // /*************************/
    pub old_account_delta_tree_root: HashOutTarget,
    pub new_account_delta_tree_root: HashOutTarget,
    pub priority_operations_count: Target,
    pub priority_operations_pub_data: [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    pub on_chain_operations_count: Target,
    pub on_chain_operations_pub_data: [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],

    /******************/
    /*  TRANSACTIONS  */
    /******************/
    pub txs: Vec<TxTarget>,
}

impl Circuit<C, F, D> for BlockTxCircuit {
    fn define(config: CircuitConfig, tx_limit: usize, chain_id: u32) -> Self {
        let mut circuit = Self::new(config, tx_limit);

        circuit.register_public_inputs();

        let (
            on_chain_operations_count,
            on_chain_operations_pub_data,
            priority_operations_count,
            priority_operations_pub_data,
            new_system_config,
            register_stack_after,
            all_assets_after,
            all_market_details_after,
            account_tree_root_after,
            account_pub_data_tree_root_after,
            account_delta_tree_root_after,
            market_tree_root_after,
        ) = circuit.define_tx_loop(tx_limit, chain_id);

        circuit.define_post_block(
            chain_id,
            account_tree_root_after,
            account_pub_data_tree_root_after,
            account_delta_tree_root_after,
            market_tree_root_after,
            &new_system_config,
            &register_stack_after,
            &all_assets_after,
            &all_market_details_after,
            on_chain_operations_count,
            &on_chain_operations_pub_data,
            priority_operations_count,
            &priority_operations_pub_data,
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

        pw.set_system_config_target(&target.old_system_config, &block.old_system_config)?;
        pw.set_register_info_target(&target.register_stack_before, &block.register_stack_before)?;
        target
            .all_assets_before
            .iter()
            .zip_eq(block.all_assets_before.iter())
            .try_for_each(|(t, ai)| pw.set_asset_target(t, ai))?;
        target
            .all_market_details_before
            .iter()
            .zip_eq(block.all_market_details_before.iter())
            .try_for_each(|(t, mi)| pw.set_market_details_target(t, mi))?;

        pw.set_hash_target(target.old_account_tree_root, block.old_account_tree_root)?;
        pw.set_hash_target(
            target.old_account_pub_data_tree_root,
            block.old_account_pub_data_tree_root,
        )?;
        pw.set_hash_target(target.old_market_tree_root, block.old_market_tree_root)?;
        pw.set_hash_target(
            target.old_account_delta_tree_root,
            block.old_account_delta_tree_root,
        )?;

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

                old_system_config: SystemConfigTarget::new(&mut builder),
                register_stack_before: RegisterStackTarget::new(&mut builder),
                all_assets_before: (0..ASSET_LIST_SIZE)
                    .map(|_| AssetTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                all_market_details_before: (0..POSITION_LIST_SIZE)
                    .map(|_| MarketDetailsTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),

                old_account_tree_root: builder.add_virtual_hash(),
                old_account_pub_data_tree_root: builder.add_virtual_hash(),
                old_account_delta_tree_root: builder.add_virtual_hash(),
                old_market_tree_root: builder.add_virtual_hash(),

                new_system_config: SystemConfigTarget::new(&mut builder),
                register_stack_after: RegisterStackTarget::new(&mut builder),
                all_assets_after: (0..ASSET_LIST_SIZE)
                    .map(|_| AssetTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                all_market_details_after: (0..POSITION_LIST_SIZE)
                    .map(|_| MarketDetailsTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),

                new_account_tree_root: builder.add_virtual_hash(),
                new_account_pub_data_tree_root: builder.add_virtual_hash(),
                new_account_delta_tree_root: builder.add_virtual_hash(),
                new_market_tree_root: builder.add_virtual_hash(),

                change_pub_key_message: ChangePubKeyMessageTarget::new(&mut builder),
                transfer_message: TransferMessageTarget::new(&mut builder),

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

                txs: (0..tx_limit).map(|_| TxTarget::new(&mut builder)).collect(),
            },

            builder,
        }
    }

    fn register_public_inputs(&mut self) {
        // Old state tree roots
        self.builder
            .register_public_hashout(self.target.old_account_pub_data_tree_root);
        self.builder
            .register_public_hashout(self.target.old_account_tree_root);
        self.builder
            .register_public_hashout(self.target.old_market_tree_root);
        self.builder
            .register_public_hashout(self.target.old_account_delta_tree_root);

        // Old market details
        self.target.all_assets_before.iter().for_each(|asset| {
            asset.register_public_input(&mut self.builder);
        });
        self.target
            .all_market_details_before
            .iter()
            .for_each(|market| {
                market.register_public_input(&mut self.builder);
            });

        // Old system config
        self.target
            .old_system_config
            .register_public_input(&mut self.builder);

        // Old register stack
        self.target
            .register_stack_before
            .register_public_input(&mut self.builder);

        // New state tree roots
        self.builder
            .register_public_hashout(self.target.new_account_pub_data_tree_root);
        self.builder
            .register_public_hashout(self.target.new_account_tree_root);
        self.builder
            .register_public_hashout(self.target.new_market_tree_root);
        self.builder
            .register_public_hashout(self.target.new_account_delta_tree_root);

        // New market details
        self.target.all_assets_after.iter().for_each(|asset| {
            asset.register_public_input(&mut self.builder);
        });
        self.target
            .all_market_details_after
            .iter()
            .for_each(|market| {
                market.register_public_input(&mut self.builder);
            });

        // Change pub key message
        self.target
            .change_pub_key_message
            .register_public_input(&mut self.builder);
        // Transfer message
        self.target
            .transfer_message
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

        // New system config
        self.target
            .new_system_config
            .register_public_input(&mut self.builder);

        // New register stack
        self.target
            .register_stack_after
            .register_public_input(&mut self.builder);
    }

    fn define_tx_loop(
        &mut self,
        tx_limit: usize,
        chain_id: u32,
    ) -> (
        Target,                                                    // on chain operations count
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE], // on chain operations public data
        Target,                                              // priority operations count
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX], // priority operations public data
        SystemConfigTarget,                                        // new system config
        RegisterStackTarget,                                       // new register stack
        [AssetTarget; ASSET_LIST_SIZE],                            // new assets
        [MarketDetailsTarget; POSITION_LIST_SIZE],                 // new market details
        HashOutTarget,                                             // new account tree root
        HashOutTarget,                                             // new account pub data tree root
        HashOutTarget,                                             // new account delta tree root
        HashOutTarget,                                             // new market tree root
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

        let mut current_system_config = self.target.old_system_config;
        let mut current_register_stack = self.target.register_stack_before;
        let mut current_all_assets = self.target.all_assets_before.clone();
        let mut current_all_market_details = self.target.all_market_details_before.clone();
        let mut current_account_tree_root = self.target.old_account_tree_root;
        let mut current_account_pub_data_tree_root = self.target.old_account_pub_data_tree_root;
        let mut current_account_delta_tree_root = self.target.old_account_delta_tree_root;
        let mut current_market_tree_root = self.target.old_market_tree_root;

        self.builder
            .register_range_check(self.target.created_at, TIMESTAMP_BITS);

        for (index, tx) in self.target.txs.iter_mut().enumerate() {
            let (
                tx_priority_operations_pub_data,
                priority_operations_pub_data_exists,
                tx_on_chain_operations_pub_data,
                on_chain_pub_data_exists,
                new_system_config,
                register_stack_after,
                all_assets_after,
                all_market_details_after,
                account_tree_root_after,
                account_pub_data_tree_root_after,
                account_delta_tree_root_after,
                market_tree_root_after,
            ) = tx.define(
                index,
                chain_id,
                &mut self.builder,
                self.target.created_at,
                &current_system_config,
                &current_register_stack,
                &current_all_assets,
                &current_all_market_details,
                current_account_tree_root,
                current_account_pub_data_tree_root,
                current_account_delta_tree_root,
                current_market_tree_root,
            );

            current_system_config = new_system_config;
            current_register_stack = register_stack_after;
            current_all_assets = all_assets_after;
            current_all_market_details = all_market_details_after;
            current_account_tree_root = account_tree_root_after;
            current_account_pub_data_tree_root = account_pub_data_tree_root_after;
            current_account_delta_tree_root = account_delta_tree_root_after;
            current_market_tree_root = market_tree_root_after;

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

        (
            on_chain_operations_count,
            on_chain_operations_pub_data,
            priority_operations_count,
            priority_operations_pub_data,
            current_system_config,
            current_register_stack,
            current_all_assets,
            current_all_market_details,
            current_account_tree_root,
            current_account_pub_data_tree_root,
            current_account_delta_tree_root,
            current_market_tree_root,
        )
    }

    fn define_post_block(
        &mut self,
        chain_id: u32,
        new_account_tree_root: HashOutTarget,
        new_account_pub_data_tree_root: HashOutTarget,
        new_account_delta_tree_root: HashOutTarget,
        new_market_tree_root: HashOutTarget,
        new_system_config: &SystemConfigTarget,
        register_stack_after: &RegisterStackTarget,
        all_assets_after: &[AssetTarget; ASSET_LIST_SIZE],
        all_market_details_after: &[MarketDetailsTarget; POSITION_LIST_SIZE],
        on_chain_operations_count: Target,
        on_chain_operations_pub_data: &[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
        priority_operations_count: Target,
        priority_operations_pub_data: &[U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        self.handle_change_pub_key();
        self.handle_transfer(chain_id);

        self.handle_on_chain_pub_data(on_chain_operations_count, on_chain_operations_pub_data);
        self.handle_priority_operation_pub_data(
            priority_operations_count,
            priority_operations_pub_data,
        );

        // Connect new state tree roots to block witness
        self.builder
            .connect_hashes(new_account_tree_root, self.target.new_account_tree_root);
        self.builder.connect_hashes(
            new_account_pub_data_tree_root,
            self.target.new_account_pub_data_tree_root,
        );
        self.builder
            .connect_hashes(new_market_tree_root, self.target.new_market_tree_root);
        self.builder.connect_hashes(
            new_account_delta_tree_root,
            self.target.new_account_delta_tree_root,
        );

        // Connect new register stack, new system config and new market details to block witness
        self.target
            .new_system_config
            .connect(&mut self.builder, new_system_config);
        self.target
            .register_stack_after
            .connect(&mut self.builder, register_stack_after);
        self.target
            .all_assets_after
            .iter()
            .zip_eq(all_assets_after)
            .for_each(|(a, b)| {
                connect_assets(&mut self.builder, a, b);
            });
        self.target
            .all_market_details_after
            .iter()
            .zip_eq(all_market_details_after)
            .for_each(|(a, b)| {
                connect_market_details(&mut self.builder, a, b);
            });
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
