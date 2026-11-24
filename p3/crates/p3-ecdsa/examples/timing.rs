//! CPU benchmark: `cargo run --release -p p3-ecdsa --example timing [counts...]`.

use std::time::Instant;

use p3_ecdsa::air::layout::*;
use p3_ecdsa::keccak;
use p3_ecdsa::keccak::prove::{prove_message_hashes, verify_message_hashes};
use p3_ecdsa::prove::PreparedEcdsaBatch;
use p3_ecdsa::testing::deterministic_batch;

fn run(count: usize, capacity: usize) {
    let instances = deterministic_batch(count, 42 + count as u64);
    let t = Instant::now();
    let prepared = PreparedEcdsaBatch::new(capacity);
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;
    let out = prepared.prove(&instances).expect("prove");
    let t = Instant::now();
    prepared
        .verify(&out.proof, &out.public_values)
        .expect("verify");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;
    let proof_bytes = bincode::serialize(&out.proof).expect("serialize").len();
    println!(
        "{count:>4}/{capacity:>4} | rows {h:>7} (used {used:>6}) | main {MAIN_WIDTH} prep {PREP_WIDTH} perm {PERM_WIDTH}x2 | \
         setup {setup_ms:>8.1} ms | trace {tg:>8.1} ms | prove {p:>9.1} ms | verify {verify_ms:>6.1} ms | proof {kb:>6.1} KiB",
        h = out.trace_height,
        used = capacity * ROWS_PER_SIGNATURE,
        tg = out.timings.trace_generation.as_secs_f64() * 1e3,
        p = out.timings.prove.as_secs_f64() * 1e3,
        kb = proof_bytes as f64 / 1024.0,
    );

    let kout = prove_message_hashes(&instances, capacity).expect("keccak prove");
    let t = Instant::now();
    verify_message_hashes(&kout.proof, &kout.public_values, capacity).expect("keccak verify");
    let kverify_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        out.public_values[5..],
        kout.public_values[1..],
        "link digests differ"
    );
    let kproof_bytes = bincode::serialize(&kout.proof).expect("serialize").len();
    let kp = kout.prove.as_secs_f64() * 1e3;
    println!(
        "{count:>4}/{capacity:>4} | keccak rows {h:>7} (used {used:>6}) | cols {w} | \
         trace {tg:>8.1} ms | prove {kp:>9.1} ms | verify {kverify_ms:>6.1} ms | proof {kb:>6.1} KiB | \
         ecdsa+keccak prove {total:>9.1} ms",
        h = kout.trace_height,
        used = capacity * keccak::ROWS_PER_SIGNATURE,
        w = keccak::WIDTH,
        tg = kout.trace_generation.as_secs_f64() * 1e3,
        kb = kproof_bytes as f64 / 1024.0,
        total = out.timings.prove.as_secs_f64() * 1e3 + kp,
    );
}

fn main() {
    println!(
        "p3-ecdsa (secp256k1) | threads {} | rows/sig {ROWS_PER_SIGNATURE} | main cols {MAIN_WIDTH} | prep cols {PREP_WIDTH} | perm cols {PERM_WIDTH} (ext deg 2) | max degree {MAX_CONSTRAINT_DEGREE}",
        p3_maybe_rayon::prelude::current_num_threads(),
    );
    println!(
        "message hash (keccak256 personal_sign) | rows/sig {} | cols {} | max degree {}",
        keccak::ROWS_PER_SIGNATURE,
        keccak::WIDTH,
        keccak::MAX_CONSTRAINT_DEGREE,
    );
    let counts: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("count"))
        .collect();
    let counts = if counts.is_empty() {
        vec![1, 32, 128, 500, 501]
    } else {
        counts
    };
    for count in counts {
        run(count, count);
    }
}
