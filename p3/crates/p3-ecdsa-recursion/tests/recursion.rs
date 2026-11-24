#![cfg(feature = "plonky2-nightly")]

use p3_ecdsa::prove::PreparedEcdsaBatch;
use p3_ecdsa::testing::deterministic_batch;
use p3_ecdsa_recursion::{
    RecursionError, build_transcript_verifier_circuit, prepare_recursive_witness,
    prove_transcript_verifier,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn recursive_verifier_accepts_valid_and_rejects_tampered_proof() {
    let capacity = 4;
    let prepared = PreparedEcdsaBatch::new(capacity);
    let circuit = build_transcript_verifier_circuit(capacity).expect("build circuit");
    let output = prepared.prove(&deterministic_batch(2, 3)).expect("prove");

    let witness = prepare_recursive_witness(&prepared, &output.proof, &output.public_values)
        .expect("witness");
    let proof = prove_transcript_verifier(&circuit, &prepared, &witness).expect("recursive prove");
    assert_eq!(
        proof.public_inputs,
        witness
            .public_inputs()
            .iter()
            .map(|&v| plonky2::field::types::Field::from_canonical_u64(v))
            .collect::<Vec<_>>()
    );
    circuit.data.verify(proof).expect("recursive verify");

    let mut bad_public = output.public_values.clone();
    bad_public[1] += Goldilocks::ONE;
    assert!(matches!(
        prepare_recursive_witness(&prepared, &output.proof, &bad_public),
        Err(RecursionError::InnerProof(_))
    ));
}
