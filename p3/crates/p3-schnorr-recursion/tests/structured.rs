#[cfg(feature = "reference")]
#[path = "support/mod.rs"]
mod common;

#[cfg(feature = "reference")]
mod tests {
    use p3_schnorr::SignatureBatchProvingOutput;
    use p3_schnorr::proof::PLONKY2_FRI_NUM_QUERY_ROUNDS;
    use p3_schnorr_recursion::extract_structured_proof;

    #[test]
    fn structured_proof_survives_wire_round_trip() {
        let output = super::common::cached_proving_output(2);
        let structured = extract_structured_proof(&output.proof);

        let decoded = SignatureBatchProvingOutput::from_bytes(&output.to_bytes().unwrap()).unwrap();
        let structured_decoded = extract_structured_proof(&decoded.proof);

        assert_eq!(structured, structured_decoded);
        assert_eq!(structured.degree_bits, output.proof.log_height());
        assert_eq!(
            structured.opening_proof.query_proofs.len(),
            PLONKY2_FRI_NUM_QUERY_ROUNDS
        );
        assert!(!structured.trace_commitment.roots.is_empty());
        assert!(!structured.trace_local.is_empty());
        assert!(!structured.opening_proof.commit_phase_commits.is_empty());
        assert!(!structured.opening_proof.final_poly.is_empty());
    }
}
