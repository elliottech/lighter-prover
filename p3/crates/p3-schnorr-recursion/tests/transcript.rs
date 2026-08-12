#[cfg(feature = "reference")]
#[path = "support/mod.rs"]
mod common;

#[cfg(all(feature = "plonky2-nightly", feature = "reference"))]
mod tests {
    use std::time::Instant;

    use p3_schnorr_recursion::{
        FIXED_RECURSIVE_SIGNATURE_COUNT, PreparedP3RecursiveWitness, RecursionError,
        build_transcript_verifier_circuit, prepare_recursive_witness, prove_transcript_verifier,
    };
    use plonky2::field::types::PrimeField64;

    #[test]
    fn transcript_verifier_circuit_proves() {
        let circuit = build_transcript_verifier_circuit(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = super::common::cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();

        let proof = prove_transcript_verifier(&circuit, &witness).unwrap();
        circuit.data.verify(proof).unwrap();
    }

    #[test]
    fn transcript_verifier_proves_partial_batch() {
        let circuit = build_transcript_verifier_circuit(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output =
            super::common::cached_proving_output_with_capacity(3, FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();

        let proof = prove_transcript_verifier(&circuit, &witness).unwrap();
        assert_eq!(proof.public_inputs[0].to_canonical_u64(), 3);
        circuit.data.verify(proof).unwrap();
    }

    #[test]
    fn prepare_recursive_witness_rejects_wrong_trace_height() {
        let proving_output = super::common::cached_proving_output(2);
        let err = prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT)
            .unwrap_err();
        assert!(matches!(err, RecursionError::InvalidTraceHeight { .. }));
    }

    /// A `PreparedP3RecursiveWitness` built by hand (bypassing
    /// `prepare_recursive_witness`'s guarantee that `proof_bytes` and
    /// `structured_proof` come from the same proof) with mismatched
    /// `proof_bytes` must be rejected before any proving work happens —
    /// `prove_transcript_verifier` only reads `structured_proof`, so without
    /// this check it would happily produce a recursive proof that says
    /// nothing about the unrelated `proof_bytes`.
    #[test]
    fn transcript_verifier_rejects_mismatched_proof_bytes() {
        let circuit = build_transcript_verifier_circuit(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = super::common::cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let honest_witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let (public_inputs, mut tampered_bytes, structured_proof) = honest_witness.into_parts();

        // Flip a byte in the middle of the serialized proof payload (well
        // past the fixed wire header and clear of the trailing length-prefixed
        // fields) so the bytes still decode as a structurally valid proof but
        // with different field values than `structured_proof` was extracted
        // from.
        let tamper_index = tampered_bytes.len() / 2;
        tampered_bytes[tamper_index] ^= 0xFF;

        let tampered =
            PreparedP3RecursiveWitness::new(public_inputs, tampered_bytes, structured_proof);

        let err = prove_transcript_verifier(&circuit, &tampered).unwrap_err();
        assert!(
            matches!(err, RecursionError::StructuredProofMismatch),
            "expected StructuredProofMismatch, got {err:?}"
        );
    }

    #[test]
    #[ignore = "manual performance probe"]
    fn transcript_verifier_metrics() {
        let build_start = Instant::now();
        let circuit = build_transcript_verifier_circuit(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let build_elapsed = build_start.elapsed();

        let proving_output = super::common::cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();

        let prove_start = Instant::now();
        let proof = prove_transcript_verifier(&circuit, &witness).unwrap();
        let prove_elapsed = prove_start.elapsed();

        let verify_start = Instant::now();
        circuit.data.verify(proof).unwrap();
        let verify_elapsed = verify_start.elapsed();

        let common = &circuit.data.common;
        println!("build_time_ms={}", build_elapsed.as_millis());
        println!("prove_time_ms={}", prove_elapsed.as_millis());
        println!("verify_time_ms={}", verify_elapsed.as_millis());
        println!("rows={}", common.degree());
        println!("cols={}", common.config.num_wires);
        println!("routed_cols={}", common.config.num_routed_wires);
        println!("degree_bits={}", common.degree_bits());
        println!("gate_types={}", common.gates.len());
        println!("num_public_inputs={}", common.num_public_inputs);
        println!("quotient_degree_factor={}", common.quotient_degree_factor);
    }
}
