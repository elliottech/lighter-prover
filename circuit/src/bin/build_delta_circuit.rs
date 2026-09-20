// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::fs;
use std::path::Path;

use anyhow::Result;
use circuit::circuit_serializer::{DeltaGateSerializer, DeltaGeneratorSerializer};
use circuit::delta::delta_constraints::{Circuit as _, DeltaCircuit};
use circuit::types::config::{C, CIRCUIT_CONFIG, D};
use clap::Parser;
use env_logger::{DEFAULT_FILTER_ENV, Env, try_init_from_env};
use log::info;
use plonky2::plonk::config::GenericHashOut;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    account_count: usize,

    #[arg(long)]
    market_count: usize,

    #[arg(long)]
    path: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

    let args = Args::parse();

    assert!(args.account_count > 0, "Account count is zero");

    let delta_circuit = DeltaCircuit::define(CIRCUIT_CONFIG, args.account_count, args.market_count);
    let delta_circuit_data = delta_circuit.builder.build::<C>();
    info!("DeltaCircuit defined!");

    // We can use same serializer for all circuits
    let gate_serializer = DeltaGateSerializer;
    let generator_serializer = DeltaGeneratorSerializer::<C, D>::default();

    // Write circuit data
    {
        let serialized_circuit = delta_circuit_data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|err| {
                anyhow::Error::msg(format!(
                    "Failed to convert tx circuit data to bytes. {:?}",
                    err
                ))
            })?;

        // Format: delta-circuit::<account_count>::<digest>
        let path_name = format!(
            "delta-circuit::t{}::{}",
            args.account_count,
            hex::encode(
                delta_circuit_data
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

    Ok(())
}
