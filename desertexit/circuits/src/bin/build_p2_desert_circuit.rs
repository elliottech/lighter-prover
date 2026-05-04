// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::fs;
use std::path::Path;

use anyhow::Result;
use circuit::circuit_serializer::DefaultPoseidonBN128GeneratorSerializer;
use circuit::ecdsa::curve::secp256k1::Secp256K1;
use circuit::poseidon_bn128::plonky2_config::PoseidonBN128GoldilocksConfig;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, OUTER_WRAPPER_CONFIG};
use clap::Parser;
use desertexit::circuit_serializer::{DesertGateSerializer, DesertGeneratorSerializer};
use desertexit::inner_circuit::InnerDesertExitCircuit;
use desertexit::outer_circuit::OuterDesertExitCircuit;
use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
use log::info;
use plonky2::plonk::config::GenericHashOut;
use plonky2::util::serialization::DefaultGateSerializer;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "info"));

    let args = Args::parse();

    let inner_circuit = InnerDesertExitCircuit::define(CIRCUIT_CONFIG);
    let inner_circuit_data = inner_circuit.builder.build::<C>();
    let inner_circuit_digest = hex::encode(
        inner_circuit_data
            .verifier_only
            .circuit_digest
            .to_bytes()
            .clone(),
    );
    info!(
        "Inner Desert Exit circuit built! Digest: {}",
        inner_circuit_digest,
    );

    // Write inner desert exit circuit to file
    {
        let inner_gate_serializer = DesertGateSerializer;
        let inner_generator_serializer = DesertGeneratorSerializer::<C, D, Secp256K1>::default();

        let serialized_inner = inner_circuit_data
            .to_bytes(&inner_gate_serializer, &inner_generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: inner-desert-circuit::<digest>
        let path_name = format!("inner-desert-circuit::{}", inner_circuit_digest);

        // If path is given, append the file name
        let mut path = args.path.clone().map_or_else(
            || Path::new(&path_name.clone()).to_path_buf(),
            |mut v| {
                v.push(path_name.clone());
                v
            },
        );
        path.set_extension("bin");
        info!("{:?}", path);
        fs::write(path.clone(), serialized_inner)?;
    }

    let outer_circuit = OuterDesertExitCircuit::define(
        OUTER_WRAPPER_CONFIG,
        &inner_circuit_data.common,
        &inner_circuit_data.verifier_only,
    );
    let outer_circuit_data = outer_circuit
        .builder
        .build::<PoseidonBN128GoldilocksConfig>();
    let outer_circuit_digest = hex::encode(
        outer_circuit_data
            .verifier_only
            .circuit_digest
            .to_bytes()
            .clone(),
    );
    info!(
        "Outer Desert Exit circuit is built! Digest: {}",
        outer_circuit_digest,
    );

    // Write outer desert exit circuit to file
    {
        let outer_gate_serializer = DefaultGateSerializer;
        let outer_generator_serializer =
            DefaultPoseidonBN128GeneratorSerializer::<PoseidonBN128GoldilocksConfig, D>::default();
        let serialized_outer_circuit = outer_circuit_data
            .to_bytes(&outer_gate_serializer, &outer_generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert outer circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: outer-desert-circuit::<inner-circuit-digest>::<digest>
        let path_name = format!(
            "outer-desert-circuit::{}::{}",
            inner_circuit_digest, outer_circuit_digest,
        );

        // If parent is given, append the file name
        let mut path = args.path.clone().map_or_else(
            || Path::new(&path_name.clone()).to_path_buf(),
            |mut v| {
                v.push(path_name.clone());
                v
            },
        );
        path.set_extension("bin");
        fs::write(path.clone(), serialized_outer_circuit)?;
        info!("Outer Desert Exit circuit is written to {:?}", path);

        // Json outputs will be used by gnark wrapper
        let outer_common_data_json = serde_json::to_string(&outer_circuit_data.common)?;
        let outer_verifier_only_json = serde_json::to_string(&outer_circuit_data.verifier_only)?;

        // Format: outer-desert-circuit::<inner-circuit-digest>::<digest>
        let common_path_name = format!(
            "outer-desert-circuit::common_circuit_data::{}",
            outer_circuit_digest,
        );
        let verifier_path_name = format!(
            "outer-desert-circuit::verifier_circuit_data::{}",
            outer_circuit_digest,
        );

        // If parent is given, append the file name
        let mut common_path = args.path.clone().map_or_else(
            || Path::new(&common_path_name.clone()).to_path_buf(),
            |mut v| {
                v.push(common_path_name.clone());
                v
            },
        );
        common_path.set_extension("json");
        let mut verifier_path = args.path.map_or_else(
            || Path::new(&verifier_path_name.clone()).to_path_buf(),
            |mut v| {
                v.push(verifier_path_name.clone());
                v
            },
        );
        verifier_path.set_extension("json");

        fs::write(common_path, outer_common_data_json)?;
        fs::write(verifier_path, outer_verifier_only_json)?;
    }

    Ok(())
}
