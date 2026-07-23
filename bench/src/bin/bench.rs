// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#![feature(stmt_expr_attributes)]
#![allow(unused_imports)]

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use circuit::block::{Block, BlockWitness};
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain::BlockTxChainWitness;
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::tx;
use circuit::types::config::{C, CIRCUIT_CONFIG, F};
use circuit::types::constants::*;
use circuit::types::state_metadata::StateMetadata;
use circuit::types::{account_delta, state_metadata};
use env_logger::{Builder, DEFAULT_FILTER_ENV, Env, try_init_from_env};
use log::{Level, LevelFilter, Log, Metadata, Record, debug, info};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::proof::CompressedProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::{self, dummy_circuit};
use rayon::vec;

const TX_PER_PROOF: usize = 4;
const CHAIN_ID: u32 = 304;

fn main() {
    init_logger_no_warn();

    let block = get_test_block_json_file("bench_test.json");
    let tx_chunks = block.tx_chunks.iter();
    let chunks_count = tx_chunks.len();

    info!(
        concat!(
            "Tx and chain circuits are configured to prove {} txs per proof in each iteration. ",
            "There are {} txs in the test block, so there will be {} iterations of proving.\n\n"
        ),
        TX_PER_PROOF,
        block.tx_chunks.iter().map(|c| c.len()).sum::<usize>(),
        chunks_count
    );

    let circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, TX_PER_PROOF, CHAIN_ID, TX_HEAVY);
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

    let pre_exec_circuit = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    let pbt = pre_exec_circuit.target;
    let pre_exec_data = pre_exec_circuit.builder.build::<C>();
    info!("BlockPreExecutionCircuit defined!");

    let light_circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, TX_PER_PROOF, CHAIN_ID, TX_LIGHT);
    let light_bt = light_circuit.target;
    let light_data = light_circuit.builder.build::<C>();
    info!("Light BlockTxCircuit defined!");

    let heavy_chain_circuit = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &data, 1);
    let heavy_chain_circuit_t = heavy_chain_circuit.target;
    let heavy_chain_circuit_data = heavy_chain_circuit.builder.build::<C>();
    let light_chain_circuit = BlockTxChainCircuit::define(CIRCUIT_CONFIG, &light_data, 1);
    let light_chain_circuit_t = light_chain_circuit.target;
    let light_chain_circuit_data = light_chain_circuit.builder.build::<C>();
    info!("BlockTxChainCircuit defined!");
    info!(
        "BlockTxChainCircuit # public inputs = {:?}",
        heavy_chain_circuit_data.common.num_public_inputs
    );

    let dummy_heavy_tx_chain_circuit = dummy_circuit(&heavy_chain_circuit_data.common);
    let dummy_light_tx_chain_circuit = dummy_circuit(&light_chain_circuit_data.common);
    info!("Dummy Tx Chain Circuit defined!");

    let dummy_heavy_proof = cyclic_base_proof(
        &heavy_chain_circuit_data.common,
        &heavy_chain_circuit_data.verifier_only,
        &dummy_heavy_tx_chain_circuit,
        Vec::<F>::new().iter().copied().enumerate().collect(),
    )
    .unwrap();
    let dummy_light_proof = cyclic_base_proof(
        &light_chain_circuit_data.common,
        &light_chain_circuit_data.verifier_only,
        &dummy_light_tx_chain_circuit,
        Vec::<F>::new().iter().copied().enumerate().collect(),
    )
    .unwrap();

    let block_pre_exec = BlockPreExec::from_block(&block);

    let pre_execution_time = Instant::now();
    let pre_proof = BlockPreExecutionCircuit::prove(&pre_exec_data, &block_pre_exec, &pbt);
    if let Err(err) = pre_proof {
        panic!("Block pre-exec failed to prove. err = {:?}", err);
    }
    let pre_proof = pre_proof.unwrap();
    let pre_execution_total = pre_execution_time.elapsed();

    let pre_exec_witness =
        BlockPreExecWitness::from_public_inputs(&pre_proof.clone().public_inputs);

    let state_metadata = pre_exec_witness.new_state_metadata.clone();
    let created_at = block.created_at;

    let mut current_heavy_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &heavy_chain_circuit_data,
        &dummy_heavy_tx_chain_circuit,
        block.block_number,
        block.created_at,
        pre_exec_witness.new_state_root,
        pre_exec_witness.new_validium_root,
        block.old_account_delta_tree_root,
    );
    let mut current_light_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &light_chain_circuit_data,
        &dummy_light_tx_chain_circuit,
        block.block_number,
        block.created_at,
        pre_exec_witness.new_state_root,
        pre_exec_witness.new_validium_root,
        block.old_account_delta_tree_root,
    );

    let mut heavy_jump = JumpState::initial(
        pre_exec_witness.new_state_root,
        block.old_account_delta_tree_root,
    );
    let mut light_jump = heavy_jump;
    let mut tx_prove_total = Duration::ZERO;
    let mut chain_prove_total = Duration::ZERO;

    let mut heavy_index: u64 = 0;
    let mut light_index: u64 = 0;
    for (index, tx) in tx_chunks.enumerate() {
        let is_light = tx[0].tx_circuit_type == TX_LIGHT;
        let block_tx = BlockTx {
            created_at,
            state_metadata_hash: state_metadata.hash(),
            old_jump: if is_light { light_jump } else { heavy_jump },
            txs: tx.to_vec(),
        };

        let tx_dt = Instant::now();
        let tx_proof = if is_light {
            BlockTxCircuit::prove(&light_data, &block_tx, &light_bt)
        } else {
            BlockTxCircuit::prove(&data, &block_tx, &bt)
        };
        let tx_dt = tx_dt.elapsed();
        if let Err(err) = tx_proof {
            panic!("Failed to prove tx chunk #{}. err = {:?}", index, err);
        }

        info!(
            "tx chunk #{index}/{} BlockTxCircuit::prove time: {:?}",
            chunks_count, tx_dt
        );
        tx_prove_total += tx_dt;

        let tx_proof = tx_proof.unwrap();

        let tx_witness = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs.clone());
        if is_light {
            light_jump = tx_witness.new_jump;
        } else {
            heavy_jump = tx_witness.new_jump;
        }

        let chain_dt = Instant::now();
        let chain_proof = if is_light {
            BlockTxChainCircuit::prove(
                &light_chain_circuit_t,
                &light_chain_circuit_data,
                light_index,
                &current_light_chain_proof,
                &dummy_light_proof,
                &tx_proof,
            )
        } else {
            BlockTxChainCircuit::prove(
                &heavy_chain_circuit_t,
                &heavy_chain_circuit_data,
                heavy_index,
                &current_heavy_chain_proof,
                &dummy_heavy_proof,
                &tx_proof,
            )
        };
        let chain_dt = chain_dt.elapsed();
        if let Err(err) = chain_proof {
            panic!("Block Chain circuit failed to prove. err = {:?}", err);
        }

        chain_prove_total += chain_dt;
        info!(
            "tx chunk #{index}/{} BlockTxChainCircuit::prove time: {:?}\n",
            chunks_count, chain_dt
        );

        if is_light {
            current_light_chain_proof = chain_proof.unwrap();
            light_index += 1;
        } else {
            current_heavy_chain_proof = chain_proof.unwrap();
            heavy_index += 1;
        }
    }

    info!(
        "TOTAL BlockPreExecutionCircuit::prove time: {:?}\n",
        pre_execution_total
    );

    info!("TOTAL BlockTxCircuit::prove time:   {:?}", tx_prove_total);
    info!(
        "AVERAGE BlockTxCircuit::prove time: {:?}\n",
        tx_prove_total / chunks_count as u32
    );

    info!(
        "TOTAL BlockTxChainCircuit::prove time: {:?}",
        chain_prove_total
    );
    info!(
        "AVERAGE BlockTxChainCircuit::prove time: {:?}",
        chain_prove_total / chunks_count as u32
    );
}

pub fn get_test_block_json_file(file_name: &str) -> Block<F> {
    let path = Path::new(".").join(file_name);
    let data = fs::read(path).expect("Unable to read file");

    Block::from_json(&data, TX_PER_PROOF, TX_PER_PROOF).expect("JSON does not have correct format.")
}

struct NoWarnLogger(env_logger::Logger);

impl Log for NoWarnLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() != Level::Warn && self.0.enabled(metadata)
    }

    fn log(&self, record: &Record) {
        if record.level() == Level::Warn {
            return;
        }
        self.0.log(record)
    }

    fn flush(&self) {
        self.0.flush()
    }
}

fn init_logger_no_warn() {
    let env = Env::default().filter_or(DEFAULT_FILTER_ENV, "info");
    let mut b = Builder::from_env(env);
    b.filter_level(LevelFilter::Info);
    let inner = b.build();

    let _ = log::set_boxed_logger(Box::new(NoWarnLogger(inner)));
    log::set_max_level(LevelFilter::Info);
}
