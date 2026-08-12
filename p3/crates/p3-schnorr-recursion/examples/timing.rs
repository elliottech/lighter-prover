use std::time::Instant;

use p3_schnorr::prove_signature_batch;
use p3_schnorr::reference::deterministic_batch;
use p3_schnorr_recursion::{
    FIXED_RECURSIVE_SIGNATURE_COUNT, build_transcript_verifier_circuit, prepare_recursive_witness,
    prove_transcript_verifier,
};

fn main() {
    let count = FIXED_RECURSIVE_SIGNATURE_COUNT;
    println!("p3-schnorr-recursion end-to-end timing  ({count} signatures)");
    println!("(circuit build is a one-time offline cost; marked separately)\n");

    // ── [offline] Build Plonky2 circuit ────────────────────────────────────
    // This is a fixed cost independent of the witness. In production it is
    // done once at startup (or pre-compiled) and reused across many proofs.
    print!("[offline] Build Plonky2 circuit ... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t = Instant::now();
    let circuit = build_transcript_verifier_circuit(count).expect("build circuit");
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{build_ms:.0} ms");

    // ── Step 1: P3 STARK prove ──────────────────────────────────────────────
    print!("[  1/3 ] P3 STARK prove ({count} sigs) ... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let batch = deterministic_batch(count).expect("build deterministic batch");
    let t = Instant::now();
    let proving_output = prove_signature_batch(&batch).expect("P3 STARK prove");
    let stark_prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{stark_prove_ms:.0} ms");

    // ── Step 2: prepare recursive witness ──────────────────────────────────
    print!("[  2/3 ] Prepare recursive witness ... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t = Instant::now();
    let witness =
        prepare_recursive_witness(&proving_output, count).expect("prepare recursive witness");
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{witness_ms:.0} ms");

    // ── Step 3: Plonky2 prove ───────────────────────────────────────────────
    print!("[  3/3 ] Plonky2 prove ... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t = Instant::now();
    let proof = prove_transcript_verifier(&circuit, &witness).expect("Plonky2 prove");
    let plonky2_prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{plonky2_prove_ms:.0} ms");

    // ── Verify ──────────────────────────────────────────────────────────────
    print!("         Plonky2 verify ... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t = Instant::now();
    circuit.data.verify(proof).expect("Plonky2 verify");
    let plonky2_verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{plonky2_verify_ms:.0} ms");

    // ── Circuit metrics ─────────────────────────────────────────────────────
    let common = &circuit.data.common;
    println!();
    println!("─── Plonky2 circuit metrics ───────────────────────────────────────");
    println!("  rows (degree)          : {}", common.degree());
    println!(
        "  gates before padding   : {}",
        circuit.num_gates_before_padding
    );
    println!("  degree_bits            : {}", common.degree_bits());
    println!("  wires (cols)           : {}", common.config.num_wires);
    println!(
        "  routed wires           : {}",
        common.config.num_routed_wires
    );
    println!("  gate types             : {}", common.gates.len());
    println!("  public inputs          : {}", common.num_public_inputs);
    println!(
        "  quotient_degree_factor : {}",
        common.quotient_degree_factor
    );
    println!("  poseidon2 calls        : {}", circuit.num_poseidon2_calls);
    println!(
        "  rows/poseidon2         : {:.1}",
        circuit.num_gates_before_padding as f64 / circuit.num_poseidon2_calls as f64
    );
    println!();

    let meta = &circuit.metadata;
    let trace_leaf_perms = (meta.air_width * 2).div_ceil(12); // ceil(elems/rate)
    let quot_leaf_perms = (meta.num_quotient_chunks * 2).div_ceil(12);
    let pre_leaf_perms = (meta.preprocessed_width * 2).max(1).div_ceil(12);
    let input_path_perms = meta.fri_input_opening_proof_len;
    let commit_path_perms: usize = meta.fri_commit_phase_opening_proof_lens.iter().sum();
    let per_query = trace_leaf_perms
        + quot_leaf_perms
        + pre_leaf_perms
        + 3 * input_path_perms
        + commit_path_perms;
    let query_total = meta.fri_query_count * per_query;
    println!("  P3 Poseidon2-16 budget (estimated):");
    println!(
        "    trace leaf hash      : {:3} × {:3} queries = {:5}",
        trace_leaf_perms,
        meta.fri_query_count,
        trace_leaf_perms * meta.fri_query_count
    );
    println!(
        "    quotient leaf hash   : {:3} × {:3} queries = {:5}",
        quot_leaf_perms,
        meta.fri_query_count,
        quot_leaf_perms * meta.fri_query_count
    );
    println!(
        "    preprocessed leaf    : {:3} × {:3} queries = {:5}",
        pre_leaf_perms,
        meta.fri_query_count,
        pre_leaf_perms * meta.fri_query_count
    );
    println!(
        "    input oracle paths   : {:3} × {:3} queries × 3 oracles = {:5}",
        input_path_perms,
        meta.fri_query_count,
        input_path_perms * meta.fri_query_count * 3
    );
    println!(
        "    FRI commit paths     : {:3} × {:3} queries = {:5}",
        commit_path_perms,
        meta.fri_query_count,
        commit_path_perms * meta.fri_query_count
    );
    println!(
        "    subtotal (queries)   :                         {:5}",
        query_total
    );
    println!(
        "    challenger overhead  :                         {:5}",
        circuit.num_poseidon2_calls.saturating_sub(query_total)
    );
    println!(
        "    total observed       :                         {:5}",
        circuit.num_poseidon2_calls
    );
    println!();
    println!("  gate type breakdown:");
    for gate in &common.gates {
        let id = gate.0.id();
        let short = id.split('<').next().unwrap_or(&id).trim();
        println!("    {short}");
    }

    println!();
    println!("─── Wall-clock summary ─────────────────────────────────────────────");
    println!("  [offline] Build circuit  : {build_ms:>8.0} ms  ← one-time cost");
    println!("  P3 STARK prove           : {stark_prove_ms:>8.0} ms");
    println!("  Prepare witness          : {witness_ms:>8.0} ms");
    println!("  Plonky2 prove            : {plonky2_prove_ms:>8.0} ms");
    println!("  Plonky2 verify           : {plonky2_verify_ms:>8.0} ms");
    println!(
        "  Online total             : {:>8.0} ms",
        stark_prove_ms + witness_ms + plonky2_prove_ms + plonky2_verify_ms
    );
}
