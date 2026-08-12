use p3_field::PrimeCharacteristicRing;
use p3_schnorr::SignatureBatchPublicInputs;
use p3_schnorr_recursion::{
    FIXED_RECURSIVE_SIGNATURE_COUNT, PreparedP3RecursiveWitness, RecursionError, StructuredP3Proof,
    verify_prepared_recursive_witness,
};

fn fake_witness(signature_count: usize) -> PreparedP3RecursiveWitness {
    let digest = [p3_goldilocks::Goldilocks::ZERO; p3_schnorr::PUBLIC_DIGEST_WIDTH];
    let public_inputs = SignatureBatchPublicInputs::new(signature_count, digest).unwrap();
    PreparedP3RecursiveWitness::new(public_inputs, vec![1, 2, 3], fake_structured_proof())
}

fn fake_structured_proof() -> StructuredP3Proof {
    StructuredP3Proof {
        degree_bits: 0,
        trace_commitment: p3_schnorr_recursion::P3MerkleCapWitness { roots: Vec::new() },
        quotient_chunks_commitment: p3_schnorr_recursion::P3MerkleCapWitness { roots: Vec::new() },
        random_commitment: None,
        trace_local: Vec::new(),
        trace_next: None,
        preprocessed_local: None,
        preprocessed_next: None,
        quotient_chunks: Vec::new(),
        random_values: None,
        opening_proof: p3_schnorr_recursion::P3FriProofWitness {
            commit_phase_commits: Vec::new(),
            commit_pow_witnesses: Vec::new(),
            query_proofs: Vec::new(),
            final_poly: Vec::new(),
            query_pow_witness: 0,
        },
    }
}

#[test]
fn prepared_witness_rejects_count_above_capacity() {
    let witness = fake_witness(FIXED_RECURSIVE_SIGNATURE_COUNT + 1);
    let err =
        verify_prepared_recursive_witness(&witness, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap_err();
    assert!(matches!(
        err,
        RecursionError::InvalidSignatureCount {
            expected: FIXED_RECURSIVE_SIGNATURE_COUNT,
            actual
        } if actual == FIXED_RECURSIVE_SIGNATURE_COUNT + 1
    ));
}

#[test]
fn prepared_witness_accepts_count_below_capacity_past_count_check() {
    let witness = fake_witness(FIXED_RECURSIVE_SIGNATURE_COUNT - 1);
    let err =
        verify_prepared_recursive_witness(&witness, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap_err();
    assert!(!matches!(err, RecursionError::InvalidSignatureCount { .. }));
}

#[test]
fn prepared_witness_surfaces_p3_decode_errors() {
    let witness = fake_witness(FIXED_RECURSIVE_SIGNATURE_COUNT);
    let err =
        verify_prepared_recursive_witness(&witness, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap_err();
    assert!(matches!(err, RecursionError::P3ProofVerification(_)));
}
