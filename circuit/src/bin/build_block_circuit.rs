// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::fs;
use std::path::Path;

use anyhow::Result;
use circuit::block_constraints::{BlockCircuit, Circuit as _};
use circuit::block_pre_execution_constraints::{BlockPreExecutionCircuit, Circuit as _};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, Circuit as _};
use circuit::circuit_serializer::{BlockGateSerializer, BlockGeneratorSerializer};
use circuit::ecdsa::curve::secp256k1::Secp256K1;
use circuit::types::config::{C, CIRCUIT_CONFIG, D};
use clap::Parser;
use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
use log::info;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::config::GenericHashOut;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    tx_per_proof: usize,

    #[arg(long)]
    on_chain_operations_limit: usize,

    #[arg(long)]
    priority_operations_limit: usize,

    #[arg(long)]
    chain_id: u32,

    #[arg(long)]
    path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

    let args = Args::parse();

    info!("tx_per_proof: {}", args.tx_per_proof);
    info!(
        "on_chain_operations_limit: {}",
        args.on_chain_operations_limit
    );
    info!(
        "priority_operations_limit: {}",
        args.priority_operations_limit
    );
    info!("chain_id: {}", args.chain_id);

    assert_eq!(
        args.on_chain_operations_limit, 1,
        "only 1 on-chain operation is supported"
    );
    assert_eq!(
        args.priority_operations_limit, 1,
        "only 1 priority operation is supported"
    );

    // Assume 1 tx per block tx segment. Also we are assuming one priority operation per block and one priority operation and on chain operation per tx segment.
    let tx_circuit = BlockTxCircuit::define(CIRCUIT_CONFIG, args.tx_per_proof, args.chain_id);
    let tx_data = tx_circuit.builder.build::<C>();
    info!("BlockTxCircuit defined!");

    let pre_exec_circuit = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
    let pre_exec_data = pre_exec_circuit.builder.build::<C>();
    info!("BlockPreExecutionCircuit defined!");

    let chain_circuit = BlockTxChainCircuit::define(
        CIRCUIT_CONFIG,
        &tx_data,
        args.tx_per_proof,
        args.on_chain_operations_limit,
    );
    let chain_circuit_data = chain_circuit.builder.build::<C>();
    info!("BlockTxChainCircuit defined!");

    let circuit = BlockCircuit::define(
        CIRCUIT_CONFIG,
        &pre_exec_data,
        &chain_circuit_data,
        args.on_chain_operations_limit,
    );
    let data = circuit.builder.build::<C>();
    info!("BlockCircuit defined!");

    // We can use same serializer for all circuits
    let gate_serializer = BlockGateSerializer;
    let generator_serializer = BlockGeneratorSerializer::<C, D, Secp256K1>::default();

    // Write tx circuit data
    {
        let serialized_circuit = tx_data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert tx circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: block-tx-circuit::t<tx_limit>::<digest>
        let path_name = format!(
            "block-tx-circuit::t{}::{}",
            args.tx_per_proof,
            hex::encode(tx_data.verifier_only.circuit_digest.to_bytes().clone())
        );

        // If parent is given, append the file name
        let parent = args.path.clone();
        let mut path = parent.map_or_else(
            || Path::new(&path_name.clone()).to_path_buf(), // default
            |mut v| {
                // if folder is given
                v.push(path_name.clone());
                v
            },
        );
        path.set_extension("bin");
        info!("{:?}", path);
        fs::write(path.clone(), serialized_circuit)?;
    }

    // Write pre-exec circuit data
    {
        let serialized_circuit = pre_exec_data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert pre-exec circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: block-pre-exec-circuit::<digest>
        let path_name = format!(
            "block-pre-exec-circuit::{}",
            hex::encode(
                pre_exec_data
                    .verifier_only
                    .circuit_digest
                    .to_bytes()
                    .clone()
            )
        );

        // If parent is given, append the file name
        let parent = args.path.clone();
        let mut path = parent.map_or_else(
            || Path::new(&path_name.clone()).to_path_buf(), // default
            |mut v| {
                // if folder is given
                v.push(path_name.clone());
                v
            },
        );
        path.set_extension("bin");
        info!("{:?}", path);
        fs::write(path.clone(), serialized_circuit)?;
    }

    // Write tx chain circuit data
    {
        let serialized_circuit = chain_circuit_data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: block-tx-chain-circuit::o<on-chain-size>::p<priority-size>::<digest>
        let path_name = format!(
            "block-tx-chain-circuit::t{}-o{}-p{}::{}",
            args.tx_per_proof,
            args.on_chain_operations_limit,
            args.priority_operations_limit,
            hex::encode(
                chain_circuit_data
                    .verifier_only
                    .circuit_digest
                    .to_bytes()
                    .clone()
            )
        );

        // If parent is given, append the file name
        let parent = args.path.clone();
        let mut path = parent.map_or_else(
            || Path::new(&path_name.clone()).to_path_buf(), // default
            |mut v| {
                // if folder is given
                v.push(path_name.clone());
                v
            },
        );
        path.set_extension("bin");
        info!("{:?}", path);
        fs::write(path.clone(), serialized_circuit)?;
    }

    // Write block circuit data
    {
        let serialized_circuit = data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: block-circuit::o<on-chain-size>::p<priority-size>::<digest>
        let path_name = format!(
            "block-circuit::o{}-p{}::{}",
            args.on_chain_operations_limit,
            args.priority_operations_limit,
            hex::encode(data.verifier_only.circuit_digest.to_bytes().clone())
        );

        // If parent is given, append the file name
        let parent = args.path.clone();
        let mut path = parent.map_or_else(
            || Path::new(&path_name.clone()).to_path_buf(), // default
            |mut v| {
                // if folder is given
                v.push(path_name.clone());
                v
            },
        );
        path.set_extension("bin");
        info!("{:?}", path);
        fs::write(path.clone(), serialized_circuit)?;

        // Read circuit data to validate the file
        let read_circuit = fs::read(path)?;
        let read_data: CircuitData<GoldilocksField, C, 2> =
            CircuitData::from_bytes(&read_circuit, &gate_serializer, &generator_serializer)
                .map_err(|err| {
                    anyhow::Error::msg(format!("Failed to read circuit data from bytes. {}", err))
                })?;
        assert_eq!(
            read_data, data,
            "read circuit data not match with orginal circuit"
        );
    }

    Ok(())
}
