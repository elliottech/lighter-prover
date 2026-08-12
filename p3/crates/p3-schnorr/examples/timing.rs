use std::time::Instant;

use p3_schnorr::air::{
    PREPROCESSED_WIDTH, ROWS_PER_SIGNATURE, SignatureChallengeHint, TRACE_WIDTH,
    generate_trace_with_challenge_hints_and_capacity,
};
use p3_schnorr::proof::{
    prepare_signature_batch_verifier_for_capacity, prove_signature_batch_trace,
    verify_signature_batch_proof_preprocessed,
};
use p3_schnorr::reference::deterministic_batch;
use p3_schnorr::schnorr::verification_witness;

fn challenge_hints(batch: &p3_schnorr::types::Batch) -> Vec<SignatureChallengeHint> {
    batch
        .instances()
        .iter()
        .map(|instance| {
            let witness = verification_witness(instance).expect("reference witness");
            SignatureChallengeHint::from_reference(instance, witness.reconstructed_r)
                .expect("challenge hint")
        })
        .collect()
}

fn run(count: usize, capacity: usize, threads: usize) {
    let rows = capacity * ROWS_PER_SIGNATURE;
    // p3 requires power-of-two height
    let padded_rows = rows.next_power_of_two();

    let batch = deterministic_batch(count).expect("reference batch");
    let hints = challenge_hints(&batch);

    // ── trace generation ────────────────────────────────────────────────────
    let t0 = Instant::now();
    let trace = generate_trace_with_challenge_hints_and_capacity(&batch, &hints, capacity)
        .expect("trace generation");
    let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── prove ───────────────────────────────────────────────────────────────
    let t1 = Instant::now();
    let output = prove_signature_batch_trace(trace).expect("prove");
    let prove_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ── verify ──────────────────────────────────────────────────────────────
    let verifier =
        prepare_signature_batch_verifier_for_capacity(capacity).expect("prepared verifier");
    let t2 = Instant::now();
    verify_signature_batch_proof_preprocessed(&verifier, &output.public_inputs, &output.proof)
        .expect("verify");
    let verify_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{sigs:>4}/{cap:>4} sigs | rows {padded:>6} (raw {raw:>5}) | trace cols {tw:>4} | prep cols {pw:>3} | threads {t:>2} | \
         trace {tg:>8.1} ms | prove {p:>8.1} ms | verify {v:>8.1} ms",
        sigs = count,
        cap = capacity,
        padded = padded_rows,
        raw = rows,
        tw = TRACE_WIDTH,
        pw = PREPROCESSED_WIDTH,
        t = threads,
        tg = trace_ms,
        p = prove_ms,
        v = verify_ms,
    );
}

fn main() {
    let threads = rayon::current_num_threads();

    println!(
        "p3-schnorr benchmark  (parallel={parallel}, threads={threads})\n",
        parallel = cfg!(feature = "parallel"),
        threads = threads,
    );
    println!(
        "{:<9} {:>20} {:>12} {:>11} {:>9} {:>13} {:>13} {:>13}",
        "sigs",
        "rows (padded/raw)",
        "trace cols",
        "prep cols",
        "threads",
        "trace gen",
        "prove",
        "verify"
    );
    println!("{}", "-".repeat(100));

    for &count in &[4usize, 32, 128, 500] {
        run(count, count, threads);
    }
    for &count in &[2usize, 32, 128, 500] {
        run(count, 500, threads);
    }
    for &count in &[2usize, 507] {
        run(count, 507, threads);
    }
}
