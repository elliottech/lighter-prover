#!/bin/bash
set -e

echo "Clearing all previous circuit files"
ls | grep "block-circuit.*.bin" | xargs rm -f
ls | grep "block-tx-circuit.*.bin" | xargs rm -f
ls | grep "block-pre-exec-circuit.*.bin" | xargs rm -f
ls | grep "block-tx-chain-circuit.*.bin" | xargs rm -f
ls | grep "cyclic-circuit.*.bin" | xargs rm -f
ls | grep "blob-evaluation-circuit.*.bin" | xargs rm -f
ls | grep "delta-circuit.*.bin" | xargs rm -f
ls | grep "cyclic-delta-circuit.*.bin" | xargs rm -f
ls | grep "inner-wrapper-circuit.*.bin" | xargs rm -f
ls | grep "outer-wrapper-circuit.*.bin" | xargs rm -f
ls | grep "outer-wrapper-circuit.*.json" | xargs rm -f
ls | grep "final::.*.pk" | xargs rm -f
ls | grep "final::.*.vk" | xargs rm -f
ls | grep "final::.*.sol" | xargs rm -f
ls | grep "final::.*.r1cs" | xargs rm -f

TX_PER_PROOF=4
ON_CHAIN_OPERATIONS_LIMIT=1
PRIORITY_OPERATIONS_LIMIT=1
CHAIN_ID=${CHAIN_ID:-304} # Default to mainnet if not set
SRS_FILE=${SRS_FILE:-"./srs_file"} # Path to the SRS file. Source: https://aztec-ignition.s3.amazonaws.com/
DELTA_ACCOUNT_COUNT=225

echo "Using:"
echo "TX_PER_PROOF: $TX_PER_PROOF"
echo "ON_CHAIN_OPERATIONS_LIMIT: $ON_CHAIN_OPERATIONS_LIMIT"
echo "PRIORITY_OPERATIONS_LIMIT: $PRIORITY_OPERATIONS_LIMIT"
echo "CHAIN_ID: $CHAIN_ID"
echo "SRS_FILE: $SRS_FILE"
echo "DELTA_ACCOUNT_COUNT: $DELTA_ACCOUNT_COUNT"

# Wait y/Y for user input to continue
read -p "Press y/Y to continue with the build: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Build cancelled."
    exit 1
fi

echo "Building binaries"
export RUSTFLAGS="-C target-cpu=native"
cargo build --release --bin build_block_circuit;
cargo build --release --bin build_recursion_circuit;
cargo build --release --bin build_wrapper_circuit;
cargo build --release --bin build_delta_circuit;
cargo build --release --bin build_delta_recursion_circuit;

echo "Running block circuit builder"
./target/release/build_block_circuit --chain-id $CHAIN_ID --tx-per-proof $TX_PER_PROOF --on-chain-operations-limit $ON_CHAIN_OPERATIONS_LIMIT --priority-operations-limit $PRIORITY_OPERATIONS_LIMIT
export block_circuit=$(ls -t block-circuit*.bin | head -n 1)
export block_tx_circuit=$(ls -t block-tx-circuit*.bin | head -n 1)
export block_pre_exec_circuit=$(ls -t block-pre-exec-circuit*.bin | head -n 1)
export block_tx_chain_circuit=$(ls -t block-tx-chain-circuit*.bin | head -n 1)

echo "Running recursion circuit builder"
./target/release/build_recursion_circuit --on-chain-operations-limit $ON_CHAIN_OPERATIONS_LIMIT --priority-operations-limit $PRIORITY_OPERATIONS_LIMIT --block-circuit-path $block_circuit
export recursion_circuit=$(ls -t cyclic-circuit*.bin | head -n 1)

echo "Running delta circuit builder"
./target/release/build_delta_circuit --account-count $DELTA_ACCOUNT_COUNT
export delta_circuit=$(ls -t delta-circuit*.bin | head -n 1)

echo "Running delta recursion circuit builder"
./target/release/build_delta_recursion_circuit --delta-circuit-path $delta_circuit
export delta_recursion_circuit=$(ls -t cyclic-delta-circuit*.bin | head -n 1)

echo "Running wrapper circuit builder"
./target/release/build_wrapper_circuit --recursion-circuit-path $recursion_circuit --delta-recursion-circuit-path $delta_recursion_circuit
export inner_wrapper_circuit=$(ls -t inner-wrapper-circuit*.bin | head -n 1)
export blob_evaluation_circuit=$(ls -t blob-evaluation-circuit*.bin | head -n 1)
export outer_wrapper_circuit=$(ls -t outer-wrapper-circuit*.bin | head -n 1)
export outer_wrapper_vd=$(ls -t outer-wrapper-circuit::verifier_circuit_data*.json | head -n 1)
export outer_wrapper_cd=$(ls -t outer-wrapper-circuit::common_circuit_data*.json | head -n 1)

echo "Running snark circuit builder"
go run snark/main.go -circuit-data $outer_wrapper_cd -verifier-circuit-data $outer_wrapper_vd -generate-keys -srs $SRS_FILE
export wrapper_circuit_digest=$(jq -r '.circuit_digest' "$outer_wrapper_vd")
echo "Final circuit digest: $wrapper_circuit_digest"
export pk_file="final::${wrapper_circuit_digest}.pk"
export vk_file="final::${wrapper_circuit_digest}.vk"
export sol_file="final::${wrapper_circuit_digest}.sol"
export r1cs_file="final::${wrapper_circuit_digest}.r1cs"

echo "Done."
echo "Solidity verifier exported to: $sol_file"
