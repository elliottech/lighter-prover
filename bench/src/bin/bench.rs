// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use circuit::block::{Block, BlockWitness};
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution::{BlockPreExec, BlockPreExecWitness};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx::{BlockTx, BlockTxWitness, JumpState};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::eddsa::schnorr_helper::{
    absorb_signature_instance, initial_signature_digest_state, prove_signature_batch_proof,
};
use circuit::types::config::{
    BLOCK_CIRCUIT_CONFIG, BLOCK_TX_CHAIN_CIRCUIT_CONFIG, C, CIRCUIT_CONFIG, F,
};
use circuit::types::constants::*;
use clap::Parser;
use env_logger::{Builder, DEFAULT_FILTER_ENV, Env};
use log::{Level, LevelFilter, Log, Metadata, Record, info};
use plonky2::field::types::Field;
use plonky2::recursion::dummy_circuit::dummy_circuit;

const CHAIN_ID: u32 = 304;

#[derive(Parser, Debug)]
#[command(about = "Lighter circuits block proving benchmark")]
struct Args {
    #[arg(long, default_value_t = 500)]
    tx_count: usize,

    #[arg(long, default_value_t = 5)]
    heavy_tx_per_proof: usize,

    #[arg(long, default_value_t = 15)]
    light_tx_per_proof: usize,

    #[arg(long, default_value = "bench_test.json")]
    witness: String,
}

fn main() {
    init_logger_no_warn();

    let args = Args::parse();
    let light_tx_count = std::cmp::max(1, args.tx_count * 98 / 100);
    let heavy_tx_count = std::cmp::max(1, args.tx_count.saturating_sub(light_tx_count));
    let heavy_tx_per_proof = args.heavy_tx_per_proof;
    let light_tx_per_proof = args.light_tx_per_proof;
    let block = get_test_block_json_file(
        &args.witness,
        heavy_tx_per_proof,
        light_tx_per_proof,
        heavy_tx_count,
        light_tx_count,
    );
    let chunks_count = block.tx_chunks.len();
    let heavy_chunks_count = block
        .tx_chunks
        .iter()
        .filter(|c| c[0].tx_circuit_type != TX_LIGHT)
        .count();
    let light_chunks_count = chunks_count - heavy_chunks_count;

    info!(
        concat!(
            "Tx and chain circuits are configured to prove {} heavy txs and {} light txs ",
            "per proof in each iteration. The block has {} heavy proof groups and {} light ",
            "proof groups.\n\n"
        ),
        heavy_tx_per_proof, light_tx_per_proof, heavy_chunks_count, light_chunks_count
    );

    let circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, heavy_tx_per_proof, CHAIN_ID, TX_HEAVY);
    let bt = circuit.target;
    let data = circuit.builder.build::<C>();

    let pre_exec_circuit = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    let pbt = pre_exec_circuit.target;
    let pre_exec_data = pre_exec_circuit.builder.build::<C>();

    let light_circuit =
        BlockTxCircuit::define(CIRCUIT_CONFIG, light_tx_per_proof, CHAIN_ID, TX_LIGHT);
    let light_bt = light_circuit.target;
    let light_data = light_circuit.builder.build::<C>();

    let heavy_chain_circuit = BlockTxChainCircuit::define(BLOCK_TX_CHAIN_CIRCUIT_CONFIG, &data, 1);
    let heavy_chain_circuit_t = heavy_chain_circuit.target;
    let heavy_chain_circuit_data = heavy_chain_circuit.builder.build::<C>();
    let light_chain_circuit =
        BlockTxChainCircuit::define(BLOCK_TX_CHAIN_CIRCUIT_CONFIG, &light_data, 1);
    let light_chain_circuit_t = light_chain_circuit.target;
    let light_chain_circuit_data = light_chain_circuit.builder.build::<C>();

    let signature_batch_circuit =
        p3_schnorr_recursion::build_transcript_verifier_circuit(MAX_REAL_SIGNATURES)
            .expect("failed to build p3-schnorr-recursion TranscriptVerifierCircuit");

    let block_circuit = BlockCircuit::define(
        BLOCK_CIRCUIT_CONFIG,
        &pre_exec_data,
        &light_chain_circuit_data,
        &heavy_chain_circuit_data,
        &signature_batch_circuit.data,
        1,
    );
    let block_circuit_t = block_circuit.target;
    let block_circuit_data = block_circuit.builder.build::<C>();

    let dummy_heavy_tx_chain_circuit = dummy_circuit(&heavy_chain_circuit_data.common);
    let dummy_light_tx_chain_circuit = dummy_circuit(&light_chain_circuit_data.common);

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

    let (block_heavy_signature_data, block_light_signature_data) = block.signature_batches();
    let total_signed_count =
        (block_heavy_signature_data.len() + block_light_signature_data.len()) as u64;
    let heavy_seed = initial_signature_digest_state(total_signed_count as usize);
    let light_seed = block_heavy_signature_data
        .iter()
        .fold(heavy_seed, |state, (pk, tx_hash, sig)| {
            absorb_signature_instance(state, *pk, *tx_hash, sig)
        });
    let mut current_heavy_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &heavy_chain_circuit_data,
        &dummy_heavy_tx_chain_circuit,
        block.block_number,
        block.created_at,
        pre_exec_witness.new_state_root,
        pre_exec_witness.new_validium_root,
        block.old_account_delta_tree_root,
        total_signed_count,
        heavy_seed,
    );
    let mut current_light_chain_proof = BlockTxChainCircuit::cyclic_base_proof(
        &light_chain_circuit_data,
        &dummy_light_tx_chain_circuit,
        block.block_number,
        block.created_at,
        pre_exec_witness.new_state_root,
        pre_exec_witness.new_validium_root,
        block.old_account_delta_tree_root,
        total_signed_count,
        light_seed,
    );

    let mut heavy_jump = JumpState::initial(
        pre_exec_witness.new_state_root,
        block.old_account_delta_tree_root,
    );
    let mut light_jump = heavy_jump;
    let mut heavy_tx_prove_total = Duration::ZERO;
    let mut light_tx_prove_total = Duration::ZERO;
    let mut heavy_chain_prove_total = Duration::ZERO;
    let mut light_chain_prove_total = Duration::ZERO;

    let mut heavy_index: u64 = 0;
    let mut light_index: u64 = 0;
    let mut all_heavy_tx_signature_data = Vec::new();
    let mut all_light_tx_signature_data = Vec::new();
    let mut heavy_digest_state = heavy_seed;
    let mut light_digest_state = light_seed;
    let mut heavy_signed_so_far = 0u64;
    let mut light_signed_so_far = 0u64;
    for (index, tx) in block.tx_chunks.iter().enumerate() {
        let is_light = tx[0].tx_circuit_type == TX_LIGHT;
        let block_tx = BlockTx {
            created_at,
            state_metadata_hash: state_metadata.hash(),
            old_jump: if is_light { light_jump } else { heavy_jump },
            txs: tx.to_vec(),
            signature_digest_state_before: if is_light {
                light_digest_state
            } else {
                heavy_digest_state
            },
            signed_count_before: F::from_canonical_u64(if is_light {
                light_signed_so_far
            } else {
                heavy_signed_so_far
            }),
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
            "tx chunk #{index} ({}) BlockTxCircuit::prove time: {:?}",
            if is_light { "light" } else { "heavy" },
            tx_dt
        );
        if is_light {
            light_tx_prove_total += tx_dt;
        } else {
            heavy_tx_prove_total += tx_dt;
        }

        let tx_proof = tx_proof.unwrap();

        let tx_witness = BlockTxWitness::from_public_inputs(&tx_proof.public_inputs.clone());
        let signature_slots: Vec<_> = tx.iter().filter_map(|tx| tx.signature_instance()).collect();
        let (digest_state, signed_so_far) = if is_light {
            (&mut light_digest_state, &mut light_signed_so_far)
        } else {
            (&mut heavy_digest_state, &mut heavy_signed_so_far)
        };
        for (pk, tx_hash, sig) in &signature_slots {
            *digest_state = absorb_signature_instance(*digest_state, *pk, *tx_hash, sig);
            *signed_so_far += 1;
        }
        if is_light {
            light_jump = tx_witness.new_jump;
            all_light_tx_signature_data.extend(signature_slots.iter().cloned());
        } else {
            heavy_jump = tx_witness.new_jump;
            all_heavy_tx_signature_data.extend(signature_slots.iter().cloned());
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

        info!(
            "tx chunk #{index} ({}) BlockTxChainCircuit::prove time: {:?}\n",
            if is_light { "light" } else { "heavy" },
            chain_dt
        );

        if is_light {
            light_chain_prove_total += chain_dt;
            current_light_chain_proof = chain_proof.unwrap();
            light_index += 1;
        } else {
            heavy_chain_prove_total += chain_dt;
            current_heavy_chain_proof = chain_proof.unwrap();
            heavy_index += 1;
        }
    }

    let all_tx_signature_data: Vec<_> = all_heavy_tx_signature_data
        .iter()
        .chain(all_light_tx_signature_data.iter())
        .cloned()
        .collect();
    let signature_batch_proof =
        prove_signature_batch_proof(&signature_batch_circuit, &all_tx_signature_data);

    let block_prove_time = Instant::now();
    let final_proof = BlockCircuit::prove(
        &block_circuit_t,
        &block_circuit_data,
        &block,
        &pre_proof,
        &current_light_chain_proof,
        &current_heavy_chain_proof,
        &signature_batch_proof,
    );
    let block_prove_total = block_prove_time.elapsed();
    if let Err(err) = final_proof {
        panic!("Block final proof failed to prove. err = {:?}", err);
    }
    let final_proof = final_proof.unwrap();

    if let Err(err) = block_circuit_data.verify(final_proof.clone()) {
        panic!("Block final proof failed to verify. err = {:?}", err);
    }
    let final_witness = BlockWitness::from_public_inputs(&final_proof.public_inputs, 1, 1);
    assert_eq!(final_witness.block_number, block.block_number);
    assert_eq!(final_witness.created_at, block.created_at);
    assert_eq!(final_witness.old_state_root, block.old_state_root);
    assert_eq!(final_witness.new_validium_root, block.new_validium_root);
    assert_eq!(final_witness.new_state_root, block.new_state_root);
    info!("Final block proof verified!\n");

    let tx_prove_total = heavy_tx_prove_total + light_tx_prove_total;
    let chain_prove_total = heavy_chain_prove_total + light_chain_prove_total;

    let overall = pre_execution_total + tx_prove_total + chain_prove_total + block_prove_total;
    info!(
        "Proved {} heavy txs in {} batches and {} light txs in {} batches. Total block proving time: {:?}",
        heavy_chunks_count * heavy_tx_per_proof,
        heavy_chunks_count,
        light_chunks_count * light_tx_per_proof,
        light_chunks_count,
        overall
    );
}

pub fn get_test_block_json_file(
    file_name: &str,
    heavy_tx_per_proof: usize,
    light_tx_per_proof: usize,
    heavy_tx_count: usize,
    light_tx_count: usize,
) -> Block<F> {
    let path = Path::new(".").join(file_name);
    let data = fs::read(path).expect("Unable to read file");

    Block::from_json_with_empty_txs(
        &data,
        heavy_tx_per_proof,
        light_tx_per_proof,
        heavy_tx_count,
        light_tx_count,
    )
    .expect("JSON does not have correct format.")
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
