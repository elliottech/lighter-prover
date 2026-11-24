//! End-to-end timing: p3-ecdsa batch (ECDSA + message-hash STARKs) -> one Plonky2
//! recursive verifier for both proofs.
//!
//! `cargo run --release -p p3-ecdsa-recursion --features plonky2-nightly --example recursion_timing [capacity] [count]`

use std::time::Instant;

use p3_ecdsa::batch::PreparedBatch;
use p3_ecdsa::testing::deterministic_batch;
use p3_ecdsa_recursion::{
    build_batch_verifier_circuit, prepare_batch_recursive_witness, prove_batch_verifier,
};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let capacity = args.first().copied().unwrap_or(500);
    let count = args.get(1).copied().unwrap_or(capacity);
    println!("p3-ecdsa-recursion end-to-end timing ({count}/{capacity} signatures)");

    let t = Instant::now();
    let circuit = build_batch_verifier_circuit(capacity).expect("build circuit");
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let prepared = PreparedBatch::new(capacity);

    let batch = deterministic_batch(count, 7);
    let t = Instant::now();
    let (proof, public_values) = prepared.prove(&batch).expect("P3 prove");
    let stark_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let witness = prepare_batch_recursive_witness(&prepared, &proof, &public_values)
        .expect("prepare witness");
    let witness_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let proof = prove_batch_verifier(&circuit, &prepared, &witness).expect("Plonky2 prove");
    let plonky2_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    circuit.data.verify(proof).expect("Plonky2 verify");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    let common = &circuit.data.common;
    println!("[offline] circuit build          : {build_ms:.0} ms");
    println!("P3 STARK prove (ECDSA + keccak)  : {stark_ms:.0} ms");
    println!("prepare witness (native vfy)     : {witness_ms:.0} ms");
    println!("Plonky2 recursive prove          : {plonky2_ms:.0} ms");
    println!("Plonky2 verify                   : {verify_ms:.0} ms");
    println!(
        "Plonky2 circuit: gates {} (degree 2^{}; keccak verifier {} gates), wires {}, poseidon2 calls {}",
        circuit.num_gates_before_padding,
        common.degree_bits(),
        circuit.message_hash_gates,
        common.config.num_wires,
        circuit.num_poseidon2_calls
    );
    for (name, m) in [
        ("ecdsa", &circuit.ecdsa_metadata),
        ("keccak", &circuit.message_hash_metadata),
    ] {
        println!(
            "inner {name} proof: degree_bits {}, main {} prep {} perm(flat) {} quotient chunks {}",
            m.degree_bits,
            m.air_width,
            m.preprocessed_width,
            m.permutation_width,
            m.num_quotient_chunks
        );
    }
}
