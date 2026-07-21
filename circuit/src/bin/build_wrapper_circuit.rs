// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::fs;
use std::path::Path;

use anyhow::Result;
use circuit::blob::blob_constraints::{BlobEvaluationCircuit, Circuit as _, PubDataMode};
use circuit::blob::bls12_381_scalar_field::BLS12381Scalar;
use circuit::circuit_serializer::{
    DefaultPoseidonBN128GeneratorSerializer, InnerWrapperGateSerializer,
    InnerWrapperGeneratorSerializer, RecursionGateSerializer, RecursionGeneratorSerializer,
};
use circuit::poseidon_bn128::plonky2_config::PoseidonBN128GoldilocksConfig;
use circuit::recursion::wrapper_circuit::WrapperCircuit;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, OUTER_WRAPPER_CONFIG};
use clap::Parser;
use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
use log::info;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::config::GenericHashOut;
use plonky2::util::serialization::DefaultGateSerializer;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    recursion_circuit_path: std::path::PathBuf,

    #[arg(long)]
    delta_recursion_circuit_path: std::path::PathBuf,

    #[arg(long)]
    path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

    let args = Args::parse();

    let inner_gate_serializer = InnerWrapperGateSerializer;
    let inner_generator_serializer =
        InnerWrapperGeneratorSerializer::<C, D, BLS12381Scalar>::default();

    // Read cyclic circuit
    let recursion_gate_serializer = RecursionGateSerializer;
    let recursion_generator_serializer = RecursionGeneratorSerializer::<C, D>::default();
    let recursion_circuit_bytes = fs::read(args.recursion_circuit_path.clone())?;
    let recursion_circuit_data: CircuitData<GoldilocksField, C, 2> = CircuitData::from_bytes(
        &recursion_circuit_bytes,
        &recursion_gate_serializer,
        &recursion_generator_serializer,
    )
    .map_err(|err| {
        anyhow::Error::msg(format!(
            "Failed to read recursion circuit data from {:?}. err: {:?}",
            args.recursion_circuit_path.clone(),
            err
        ))
    })?;
    let recursion_circuit_digest = hex::encode(
        recursion_circuit_data
            .verifier_only
            .circuit_digest
            .to_bytes()
            .clone(),
    );
    info!(
        "Recursion circuit {:?} read",
        args.recursion_circuit_path.clone()
    );

    // Read cyclic delta circuit
    let delta_recursion_bytes = fs::read(args.delta_recursion_circuit_path.clone())?;
    let delta_recursion_circuit_data: CircuitData<GoldilocksField, C, 2> = CircuitData::from_bytes(
        &delta_recursion_bytes,
        &recursion_gate_serializer,
        &recursion_generator_serializer,
    )
    .map_err(|err| {
        anyhow::Error::msg(format!(
            "Failed to read delta recursion circuit data from {:?}. err: {:?}",
            args.delta_recursion_circuit_path.clone(),
            err
        ))
    })?;
    let delta_recursion_circuit_digest = hex::encode(
        delta_recursion_circuit_data
            .verifier_only
            .circuit_digest
            .to_bytes()
            .clone(),
    );
    info!(
        "Delta recursion circuit {:?} read",
        args.delta_recursion_circuit_path.clone()
    );

    // Define blob evaluation circuit
    info!("Polynomial circuit in {:?} mode", PubDataMode::from_env());
    let blob_evaluation_circuit = BlobEvaluationCircuit::define(CIRCUIT_CONFIG);
    let blob_evaluation_circuit_data = blob_evaluation_circuit.builder.build::<C>();
    let blob_evaluation_circuit_digest = hex::encode(
        blob_evaluation_circuit_data
            .verifier_only
            .circuit_digest
            .to_bytes()
            .clone(),
    );
    info!(
        "Blob evaluation circuit built!. Digest: {}",
        blob_evaluation_circuit_digest,
    );

    // Write blob evaluation circuit data
    let serialized_blob_evaluation_circuit = blob_evaluation_circuit_data
        .to_bytes(&inner_gate_serializer, &inner_generator_serializer)
        .map_err(|err| {
            anyhow::Error::msg(format!(
                "Failed to convert inner circuit data to bytes. {:?}",
                err
            ))
        })?;
    info!("Blob evaluation circuit serialized");

    // Format: blob-evaluation-circuit::<digest>
    let path_name = format!(
        "blob-evaluation-circuit::{}",
        blob_evaluation_circuit_digest,
    );
    // If parent is given, append the file name
    let mut path = args.path.clone().map_or_else(
        || Path::new(&path_name.clone()).to_path_buf(), // default
        |mut v| {
            // if folder is given
            v.push(path_name.clone());
            v
        },
    );
    path.set_extension("bin");
    fs::write(path.clone(), serialized_blob_evaluation_circuit)?;
    info!("Blob evaluation circuit is written to {:?}", path);

    // Define inner wrapper circuit
    let inner_circuit = WrapperCircuit::define_inner(
        CIRCUIT_CONFIG,
        &recursion_circuit_data.common,
        &recursion_circuit_data.verifier_only,
        &delta_recursion_circuit_data.common,
        &delta_recursion_circuit_data.verifier_only,
        &blob_evaluation_circuit_data.common,
        &blob_evaluation_circuit_data.verifier_only,
    );
    // Build wrapper inner circuit
    let inner_circuit_data = inner_circuit.builder.build::<C>();
    let inner_circuit_digest = hex::encode(
        inner_circuit_data
            .verifier_only
            .circuit_digest
            .to_bytes()
            .clone(),
    );
    info!(
        "Inner wrapper circuit built!. Digest: {}",
        inner_circuit_digest,
    );

    // Write inner circuit data
    let serialized_inner_circuit = inner_circuit_data
        .to_bytes(&inner_gate_serializer, &inner_generator_serializer)
        .map_err(|err| {
            anyhow::Error::msg(format!(
                "Failed to convert inner circuit data to bytes. {:?}",
                err
            ))
        })?;
    info!("Inner wrapper circuit serialized");

    // Format: inner-wrapper-circuit::<recursion-circuit-digest>::<digest>
    let path_name = format!(
        "inner-wrapper-circuit::{}::{}::{}",
        recursion_circuit_digest, delta_recursion_circuit_digest, inner_circuit_digest,
    );
    // If parent is given, append the file name
    let mut path = args.path.clone().map_or_else(
        || Path::new(&path_name.clone()).to_path_buf(), // default
        |mut v| {
            // if folder is given
            v.push(path_name.clone());
            v
        },
    );
    path.set_extension("bin");
    fs::write(path.clone(), serialized_inner_circuit)?;
    info!("Inner wrapper circuit is written to {:?}", path);

    // Define outer wrapper circuit
    let outer_circuit = WrapperCircuit::define_outer(
        OUTER_WRAPPER_CONFIG,
        &inner_circuit_data.common,
        &inner_circuit_data.verifier_only,
    );
    // Build wrapper outer circuit
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
        "Outer wrapper circuit is built! Digest: {}",
        outer_circuit_digest,
    );

    // Write outer circuit data
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
    info!("Outer wrapper circuit serialized");

    // Format: outer-wrapper-circuit::<inner-circuit-digest>::<digest>
    let path_name = format!(
        "outer-wrapper-circuit::{}::{}",
        inner_circuit_digest, outer_circuit_digest,
    );

    // If parent is given, append the file name
    let mut path = args.path.clone().map_or_else(
        || Path::new(&path_name.clone()).to_path_buf(), // default
        |mut v| {
            // if folder is given
            v.push(path_name.clone());
            v
        },
    );
    path.set_extension("bin");
    fs::write(path.clone(), serialized_outer_circuit)?;
    info!("Outer wrapper circuit is written to {:?}", path);

    // Json outputs will be used by gnark wrapper
    let outer_common_data_json = serde_json::to_string(&outer_circuit_data.common)?;
    let outer_verifier_only_json = serde_json::to_string(&outer_circuit_data.verifier_only)?;

    // Format: outer-wrapper-circuit::<inner-circuit-digest>::<digest>
    let common_path_name = format!(
        "outer-wrapper-circuit::common_circuit_data::{}",
        outer_circuit_digest,
    );
    let verifier_path_name = format!(
        "outer-wrapper-circuit::verifier_circuit_data::{}",
        outer_circuit_digest,
    );

    // If parent is given, append the file name
    let mut common_path = args.path.clone().map_or_else(
        || Path::new(&common_path_name.clone()).to_path_buf(), // default
        |mut v| {
            // if folder is given
            v.push(common_path_name.clone());
            v
        },
    );
    common_path.set_extension("json");
    let mut verifier_path = args.path.map_or_else(
        || Path::new(&verifier_path_name.clone()).to_path_buf(), // default
        |mut v| {
            // if folder is given
            v.push(verifier_path_name.clone());
            v
        },
    );
    verifier_path.set_extension("json");

    fs::write(common_path, outer_common_data_json)?;
    fs::write(verifier_path, outer_verifier_only_json)?;

    Ok(())
}
