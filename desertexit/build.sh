#!/bin/bash
set -e

# Navigate to script level
cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Define subdirectories
artifacts_dir="artifacts"
output_dir="output"
mkdir -p "$artifacts_dir"
mkdir -p "$output_dir"

# Clean previous artifacts
echo -e "\nCleaning artifacts from previous runs"
rm -rf "$artifacts_dir"/*
rm -rf "$output_dir"/*

# Check SRS file
srs="srs_file"
if [ -n "$PATH_TO_SRS_FILE" ] && [ -f "$PATH_TO_SRS_FILE" ]; then
  srs="$PATH_TO_SRS_FILE"
  echo -e "\nUsing provided SRS file: $srs\n"
else
  echo -e "\nUsing default SRS file path: $srs\n"
fi

# Build plonky2 part of desert circuit binaries
echo -e "\nBuilding binaries"
circuits_dir="circuits"
cd "$circuits_dir"
cargo build --release --bin build_p2_desert_circuit;

cd ../..
pwd

# Build plonky2 part of desert exit
echo -e "\nRunning plonky2 desert circuit builder..."
./target/release/build_p2_desert_circuit --path "desertexit/$artifacts_dir"
export inner_circuit=$(ls -t "desertexit/$artifacts_dir"/inner-desert-circuit*.bin | head -n 1)
export outer_circuit=$(ls -t "desertexit/$artifacts_dir"/outer-desert-circuit*.bin | head -n 1)

# Build gnark wrapper part of desert exit
echo -e "\nRunning desert wrapper circuit builder...\n"
cd desertexit/circuits
go mod tidy
go mod vendor

export outer_vd=$(ls -t "../$artifacts_dir"/outer-desert-circuit::verifier_circuit_data*.json | head -n 1)
export outer_cd=$(ls -t "../$artifacts_dir"/outer-desert-circuit::common_circuit_data*.json | head -n 1)
export wrapper_circuit_digest=$(jq -r '.circuit_digest' "$outer_vd")

pwd
echo "artifacts_dir: ../$artifacts_dir"
echo "outer_cd: $outer_cd"
echo "outer_vd: $outer_vd"
echo "srs: ../$srs"

go run wrapper/build_gnark_wrapper_circuit/main.go \
  -output-path "../$artifacts_dir" \
  -circuit-data $outer_cd \
  -verifier-circuit-data $outer_vd \
  -inner-circuit-digest $wrapper_circuit_digest \
  -srs "../$srs"
