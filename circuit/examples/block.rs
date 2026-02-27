#![feature(stmt_expr_attributes)]
#![allow(unused_imports)]

use std::fs;
use std::path::Path;

use circuit::block::{Block, BlockWitness};
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness};
use circuit::block_tx_chain::BlockTxChainWitness;
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::tx;
use circuit::types::config::{C, CIRCUIT_CONFIG, F};
use circuit::types::constants::*;
use circuit::types::state_metadata::StateMetadata;
use circuit::types::{account_delta, state_metadata};
use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
use log::{debug, error, info};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::PrimeField64;
use plonky2::plonk::proof::CompressedProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::{self, dummy_circuit};
use rayon::vec;
use tikv_jemallocator::Jemalloc;
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const CHAIN_ID: u32 = 300;

fn main() {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

    let tx_per_proof = 1;

    let circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID);
    let bt = circuit.target;
    let data = circuit.builder.build::<C>();
    info!("BlockTxCircuit defined!");
    info!(
        "BlockTxCircuit # public inputs = {:?}",
        data.common.num_public_inputs
    );
    info!(
        "BlockTxCircuit # num_gate_constraints = {:?}",
        data.common.num_gate_constraints
    );

    if CIRCUIT_CONFIG.addition_gate_enabled() {
        info!("==> Addition gate enabled");
    }

    if CIRCUIT_CONFIG.multiplication_gate_enabled() {
        info!("==> Multiplication gate enabled");
    }

    if CIRCUIT_CONFIG.quintic_multiplication_gate_enabled() {
        info!("==> Quintic multiplication gate enabled");
    }

    if CIRCUIT_CONFIG.equality_gate_enable() {
        info!("==> Equality gate enabled");
    }

    if CIRCUIT_CONFIG.quintic_squaring_gate_enabled() {
        info!("==> Quintic squaring gate enabled");
    }

    if CIRCUIT_CONFIG.select_gate_enabled() {
        info!("==> Select gate enabled");
    }

    let pre_exec_circuit = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    let pbt = pre_exec_circuit.target;
    let pre_exec_data = pre_exec_circuit.builder.build::<C>();

    info!("BlockPreExecutionCircuit defined!");

    let chain_circuit = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &data, tx_per_proof, 1);
    let chain_circuit_t = chain_circuit.target;
    let chain_circuit_data = chain_circuit.builder.build::<C>();

    info!("BlockTxChainCircuit defined!");
    info!(
        "BlockTxChainCircuit # public inputs = {:?}",
        chain_circuit_data.common.num_public_inputs
    );

    let block_circuit =
        BlockCircuit::define(CIRCUIT_CONFIG, &pre_exec_data, &chain_circuit_data, 1);
    let block_circuit_t = block_circuit.target;
    let block_circuit_data = block_circuit.builder.build::<C>();

    info!("BlockCircuit defined!");
    info!(
        "BlockCircuit # public inputs = {:?}",
        block_circuit_data.common.num_public_inputs
    );

    let dummy_tx_chain_circuit = dummy_circuit(&chain_circuit_data.common);
    info!("Dummy Tx Chain Circuit defined!");

    for block_name in ["mock_witness.json"] {
        info!("Running test case: {:?}", block_name);
        let block = get_test_block_json_file(block_name);

        let block_pre_exec = BlockPreExec::from_block(&block);

        let pre_proof = BlockPreExecutionCircuit::prove(&pre_exec_data, &block_pre_exec, &pbt);
        if let Err(err) = pre_proof {
            error!(
                "Block pre-exec {:?} failed to prove. err = {:?}",
                block_name, err
            );
            panic!("failed!");
        }
        let pre_proof = pre_proof.unwrap();

        let pre_exec_witness =
            BlockPreExecWitness::from_public_inputs(&pre_proof.clone().public_inputs);

        let state_metadata = pre_exec_witness.new_state_metadata.clone();

        let mut all_assets = block.all_assets.clone();
        let mut all_market_details = pre_exec_witness.new_market_details.clone();
        let mut system_config = block.old_system_config;
        let mut register_stack = block.register_stack_before;
        let mut account_tree_root = block.old_account_tree_root;
        let mut account_pub_data_tree_root = block.old_account_pub_data_tree_root;
        let mut account_delta_tree_root = block.old_account_delta_tree_root;
        let mut market_tree_root = block.old_market_tree_root;
        let created_at = block.created_at;

        let mut current_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
            &chain_circuit_data,
            &dummy_tx_chain_circuit,
            block.block_number,
            block.created_at,
            pre_exec_witness.new_state_root,
            pre_exec_witness.new_state_root,
            pre_exec_witness.new_validium_root,
            block.old_account_delta_tree_root,
            chain_circuit.block_tx_witness_size,
            &state_metadata,
        );
        let dummy_proof = cyclic_base_proof(
            &chain_circuit_data.common,
            &chain_circuit_data.verifier_only,
            &dummy_tx_chain_circuit,
            Vec::<F>::new().iter().copied().enumerate().collect(), // create empty hashbrown::HashMap
        )
        .unwrap();

        for (index, tx) in block.txs.chunks(tx_per_proof).enumerate() {
            info!("Running tx: {:?}, type: {}", index, tx[0].tx_type);

            let block_tx = BlockTx {
                created_at,
                old_system_config: system_config,
                register_stack_before: register_stack,
                all_assets_before: all_assets.clone(),
                all_market_details_before: all_market_details.clone(),
                old_account_tree_root: account_tree_root,
                old_account_pub_data_tree_root: account_pub_data_tree_root,
                old_account_delta_tree_root: account_delta_tree_root,
                old_market_tree_root: market_tree_root,
                txs: tx.to_vec(),
            };

            let tx_proof = BlockTxCircuit::prove(&data, &block_tx, &bt);
            if let Err(err) = tx_proof {
                error!(
                    "Failed to prove tx {:?} with type {:?} in block {:?}: {:?}",
                    index, tx[0].tx_type, block.block_number, err
                );
                panic!("failed!");
            }

            let tx_proof = tx_proof.unwrap();

            let tx_witness = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs.clone());
            all_assets = tx_witness.all_assets_after.clone();
            all_market_details = tx_witness.all_market_details_after.clone();
            register_stack = tx_witness.register_stack_after;
            system_config = tx_witness.new_system_config;
            account_tree_root = tx_witness.new_account_tree_root;
            account_pub_data_tree_root = tx_witness.new_account_pub_data_tree_root;
            account_delta_tree_root = tx_witness.new_account_delta_tree_root;
            market_tree_root = tx_witness.new_market_tree_root;

            let chain_proof = BlockTxChainCircuit::prove(
                &chain_circuit_t,
                &chain_circuit_data,
                index as u64,
                &current_chain_proof,
                &dummy_proof,
                &tx_proof,
            );
            if let Err(err) = chain_proof {
                error!(
                    "Block Chain {:?} failed to prove. err = {:?}",
                    block_name, err
                );
                panic!("failed!");
            }
            current_chain_proof = chain_proof.unwrap();

            // let witness = BlockTxChainWitness::from_public_inputs(
            //     &current_chain_proof.public_inputs[..chain_circuit.block_tx_witness_size],
            //     1,
            //     1,
            // );
            // println!("{} witness = {:?}", index, witness);

            // // Uncomment for failing tx debugging
            // //
            // fs::write(
            //     "txpubins",
            //     tx_proof
            //         .public_inputs
            //         .iter()
            //         .flat_map(|e| e.to_canonical_u64().to_le_bytes())
            //         .collect::<Vec<u8>>(),
            // )
            // .unwrap();
            // // Uncomment for failing tx debugging
            // //
            // fs::write("cyclicproof", current_chain_proof.to_bytes()).unwrap();
        }

        // let witness = BlockTxChainWitness::from_public_inputs(
        //     &current_chain_proof.public_inputs[..chain_circuit.block_tx_witness_size],
        //     1,
        //     1,
        // );
        // println!("Final witness = {:?}", witness);
        // let state_metadata = StateMetadata::from_public_inputs(
        //     &current_chain_proof.public_inputs
        //         [chain_circuit.block_tx_witness_size..chain_circuit.block_tx_witness_size + 3],
        // );
        // println!("Final state_metadata = {:?}", state_metadata);
        // assert_eq!(witness.new_validium_root, block.new_validium_root);
        // assert_eq!(witness.new_state_root, block.new_state_root);

        let final_proof = BlockCircuit::prove(
            &block_circuit_t,
            &block_circuit_data,
            &block,
            &pre_proof,
            &current_chain_proof,
        );
        if let Err(err) = final_proof {
            error!(
                "Block final {:?} failed to prove. err = {:?}",
                block_name, err
            );
            continue;
        }

        // let final_proof = final_proof.unwrap();

        // let final_witness = BlockWitness::from_public_inputs(&final_proof.public_inputs, 1, 1);
        // assert_eq!(final_witness.block_number, block.block_number);
        // assert_eq!(final_witness.created_at, block.created_at);
        // assert_eq!(final_witness.old_state_root, block.old_state_root);
        // assert_eq!(final_witness.new_validium_root, block.new_validium_root);
        // assert_eq!(final_witness.new_state_root, block.new_state_root);
    }
}

pub fn get_test_block_json_file(file_name: &str) -> Block<F> {
    let path = Path::new(".")
        .join("circuit")
        .join("examples")
        .join("witnessdata")
        .join(file_name);
    let data = fs::read_to_string(path).expect("Unable to read file");

    serde_json::from_str(&data).expect("JSON does not have correct format.")
}
