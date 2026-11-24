//! Public API: preparing a `p3-ecdsa` LogUp proof as a recursive witness and
//! converting it to the `u64`-encoded [`crate::types`] structs.

use p3_ecdsa::batch::{BatchProof, PreparedBatch, message_hash_public_values};
use p3_ecdsa::keccak::prove::MessageHashProof;
use p3_ecdsa::prove::{EcdsaBatchProof, PreparedEcdsaBatch};
use p3_field::{BasedVectorSpace, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_schnorr::proof;

use crate::error::RecursionError;
use crate::types::{
    BatchRecursiveWitness, P3BatchOpeningWitness, P3CommitPhaseProofStepWitness, P3FriProofWitness,
    P3FriQueryWitness, P3MerkleCapWitness, PreparedP3RecursiveWitness, StructuredP3Proof,
};

/// Batch capacity the recursive circuit is built for (matches the Schnorr wrapper).
pub const FIXED_RECURSIVE_SIGNATURE_COUNT: usize = 500;

/// Natively verifies `proof` against `public_values` and packages it as a
/// recursive witness.
pub fn prepare_recursive_witness(
    prepared: &PreparedEcdsaBatch,
    proof: &EcdsaBatchProof,
    public_values: &[Goldilocks],
) -> Result<PreparedP3RecursiveWitness, RecursionError> {
    ensure_shape(prepared, proof, public_values)?;
    prepared.verify(proof, public_values)?;
    let proof_bytes =
        bincode::serialize(proof).map_err(|e| RecursionError::Serialization(e.to_string()))?;
    Ok(PreparedP3RecursiveWitness::new(
        public_values.iter().map(|v| v.as_canonical_u64()).collect(),
        proof_bytes,
        extract_structured_proof(proof),
    ))
}

/// Re-decodes and natively re-verifies the proof bytes in `witness`, and
/// checks the structured proof and public inputs agree with them.
pub fn verify_prepared_recursive_witness(
    prepared: &PreparedEcdsaBatch,
    witness: &PreparedP3RecursiveWitness,
) -> Result<(), RecursionError> {
    let proof: EcdsaBatchProof = bincode::deserialize(witness.proof_bytes())
        .map_err(|e| RecursionError::Serialization(e.to_string()))?;
    let public_values: Vec<Goldilocks> = witness
        .public_inputs()
        .iter()
        .map(|&v| Goldilocks::new(v))
        .collect();
    ensure_shape(prepared, &proof, &public_values)?;
    if extract_structured_proof(&proof) != *witness.structured_proof() {
        return Err(RecursionError::StructuredProofMismatch);
    }
    prepared.verify(&proof, &public_values)?;
    Ok(())
}

fn ensure_shape(
    prepared: &PreparedEcdsaBatch,
    proof: &EcdsaBatchProof,
    public_values: &[Goldilocks],
) -> Result<(), RecursionError> {
    let capacity = prepared.capacity();
    let actual = public_values
        .first()
        .map_or(0, |v| v.as_canonical_u64() as usize);
    if public_values.len() != p3_ecdsa::air::layout::NUM_PUBLIC_VALUES
        || actual == 0
        || actual > capacity
    {
        return Err(RecursionError::InvalidSignatureCount {
            expected: capacity,
            actual,
        });
    }
    let expected_rows = 1usize << prepared.air.log_height();
    let actual_rows = 1usize << proof.degree_bits;
    if expected_rows != actual_rows {
        return Err(RecursionError::InvalidTraceHeight {
            expected_rows,
            actual_rows,
        });
    }
    Ok(())
}

/// Natively verifies both halves of `proof` (via [`PreparedBatch::verify`])
/// and packages them as recursive witnesses for the batch circuit.
pub fn prepare_batch_recursive_witness(
    prepared: &PreparedBatch,
    proof: &BatchProof,
    public_values: &[Goldilocks],
) -> Result<BatchRecursiveWitness, RecursionError> {
    ensure_shape(&prepared.ecdsa, &proof.ecdsa, public_values)?;
    ensure_message_hash_height(prepared.capacity(), &proof.message_hash)?;
    prepared.verify(proof, public_values)?;
    let ecdsa_bytes = bincode::serialize(&proof.ecdsa)
        .map_err(|e| RecursionError::Serialization(e.to_string()))?;
    let message_hash_bytes = bincode::serialize(&proof.message_hash)
        .map_err(|e| RecursionError::Serialization(e.to_string()))?;
    let message_hash_pv = message_hash_public_values(public_values)?;
    Ok(BatchRecursiveWitness {
        ecdsa: PreparedP3RecursiveWitness::new(
            words(public_values),
            ecdsa_bytes,
            extract_structured_proof(&proof.ecdsa),
        ),
        message_hash: PreparedP3RecursiveWitness::new(
            words(&message_hash_pv),
            message_hash_bytes,
            extract_message_hash_structured_proof(&proof.message_hash),
        ),
    })
}

/// Re-decodes both proofs in `witness`, checks they match the structured
/// proofs, that the message-hash public inputs are `[n, L]` of the ECDSA
/// ones, and re-verifies the batch natively.
pub fn verify_prepared_batch_witness(
    prepared: &PreparedBatch,
    witness: &BatchRecursiveWitness,
) -> Result<(), RecursionError> {
    let ecdsa: EcdsaBatchProof = bincode::deserialize(witness.ecdsa.proof_bytes())
        .map_err(|e| RecursionError::Serialization(e.to_string()))?;
    let message_hash: MessageHashProof =
        bincode::deserialize(witness.message_hash.proof_bytes())
            .map_err(|e| RecursionError::Serialization(e.to_string()))?;
    let public_values: Vec<Goldilocks> = witness
        .ecdsa
        .public_inputs()
        .iter()
        .map(|&v| Goldilocks::new(v))
        .collect();
    ensure_shape(&prepared.ecdsa, &ecdsa, &public_values)?;
    ensure_message_hash_height(prepared.capacity(), &message_hash)?;
    if words(&message_hash_public_values(&public_values)?) != witness.message_hash.public_inputs() {
        return Err(RecursionError::PublicInputMismatch);
    }
    if extract_structured_proof(&ecdsa) != *witness.ecdsa.structured_proof()
        || extract_message_hash_structured_proof(&message_hash)
            != *witness.message_hash.structured_proof()
    {
        return Err(RecursionError::StructuredProofMismatch);
    }
    prepared.verify(
        &BatchProof {
            ecdsa,
            message_hash,
        },
        &public_values,
    )?;
    Ok(())
}

fn ensure_message_hash_height(
    capacity: usize,
    proof: &MessageHashProof,
) -> Result<(), RecursionError> {
    let expected_rows = p3_ecdsa::keccak::MessageHashAir::new(capacity).height();
    let actual_rows = 1usize << proof.degree_bits;
    if expected_rows != actual_rows {
        return Err(RecursionError::InvalidTraceHeight {
            expected_rows,
            actual_rows,
        });
    }
    Ok(())
}

fn words(values: &[Goldilocks]) -> Vec<u64> {
    values.iter().map(|v| v.as_canonical_u64()).collect()
}

/// Converts a native message-hash (uni-stark) proof into the circuit-friendly
/// `u64` encoding; it has no permutation or preprocessed trace.
pub fn extract_message_hash_structured_proof(proof: &MessageHashProof) -> StructuredP3Proof {
    let ov = &proof.opened_values;
    StructuredP3Proof {
        degree_bits: proof.degree_bits,
        trace_commitment: merkle_cap_witness(&proof.commitments.trace),
        quotient_chunks_commitment: merkle_cap_witness(&proof.commitments.quotient_chunks),
        permutation_commitment: P3MerkleCapWitness { roots: Vec::new() },
        permutation_local: Vec::new(),
        permutation_next: Vec::new(),
        random_commitment: proof.commitments.random.as_ref().map(merkle_cap_witness),
        trace_local: ext_vec_witness(&ov.trace_local),
        trace_next: ov.trace_next.as_deref().map(ext_vec_witness),
        preprocessed_local: ov.preprocessed_local.as_deref().map(ext_vec_witness),
        preprocessed_next: ov.preprocessed_next.as_deref().map(ext_vec_witness),
        quotient_chunks: ov
            .quotient_chunks
            .iter()
            .map(|c| ext_vec_witness(c))
            .collect(),
        random_values: ov.random.as_deref().map(ext_vec_witness),
        opening_proof: fri_proof_witness(&proof.opening_proof),
    }
}

/// Converts a native LogUp proof into the circuit-friendly `u64` encoding.
pub fn extract_structured_proof(proof: &EcdsaBatchProof) -> StructuredP3Proof {
    let ov = &proof.opened_values;
    StructuredP3Proof {
        degree_bits: proof.degree_bits,
        trace_commitment: merkle_cap_witness(&proof.trace_commit),
        quotient_chunks_commitment: merkle_cap_witness(&proof.quotient_commit),
        permutation_commitment: merkle_cap_witness(&proof.permutation_commit),
        permutation_local: ext_vec_witness(&ov.permutation_local),
        permutation_next: ext_vec_witness(&ov.permutation_next),
        random_commitment: None,
        trace_local: ext_vec_witness(&ov.trace_local),
        trace_next: Some(ext_vec_witness(&ov.trace_next)),
        preprocessed_local: Some(ext_vec_witness(&ov.preprocessed_local)),
        preprocessed_next: Some(ext_vec_witness(&ov.preprocessed_next)),
        quotient_chunks: ov
            .quotient_chunks
            .iter()
            .map(|c| ext_vec_witness(c))
            .collect(),
        random_values: None,
        opening_proof: fri_proof_witness(&proof.opening_proof),
    }
}

/// Converts a Merkle cap's digests (8 Goldilocks elements each) to their
/// canonical `u64` form.
fn merkle_cap_witness<F>(cap: &p3_symmetric::MerkleCap<F, [Goldilocks; 8]>) -> P3MerkleCapWitness {
    P3MerkleCapWitness {
        roots: cap.roots().iter().map(goldilocks_digest).collect(),
    }
}

/// Converts a native `p3_fri::FriProof` to [`P3FriProofWitness`].
fn fri_proof_witness(
    proof: &p3_fri::FriProof<
        proof::Challenge,
        proof::ChallengeMmcs,
        Goldilocks,
        Vec<p3_commit::BatchOpening<Goldilocks, proof::ValMmcs>>,
    >,
) -> P3FriProofWitness {
    P3FriProofWitness {
        commit_phase_commits: proof
            .commit_phase_commits
            .iter()
            .map(merkle_cap_witness)
            .collect(),
        commit_pow_witnesses: proof
            .commit_pow_witnesses
            .iter()
            .map(|value| goldilocks_word(*value))
            .collect(),
        query_proofs: proof.query_proofs.iter().map(fri_query_witness).collect(),
        final_poly: ext_vec_witness(&proof.final_poly),
        query_pow_witness: goldilocks_word(proof.query_pow_witness),
    }
}

/// Converts a native `p3_fri::QueryProof` to [`P3FriQueryWitness`].
fn fri_query_witness(
    proof: &p3_fri::QueryProof<
        proof::Challenge,
        proof::ChallengeMmcs,
        Vec<p3_commit::BatchOpening<Goldilocks, proof::ValMmcs>>,
    >,
) -> P3FriQueryWitness {
    P3FriQueryWitness {
        input_proof: proof
            .input_proof
            .iter()
            .map(batch_opening_witness)
            .collect(),
        commit_phase_openings: proof
            .commit_phase_openings
            .iter()
            .map(commit_phase_step_witness)
            .collect(),
    }
}

/// Converts a native `p3_commit::BatchOpening` to [`P3BatchOpeningWitness`].
fn batch_opening_witness(
    opening: &p3_commit::BatchOpening<Goldilocks, proof::ValMmcs>,
) -> P3BatchOpeningWitness {
    P3BatchOpeningWitness {
        opened_values: opening
            .opened_values
            .iter()
            .map(|row: &Vec<Goldilocks>| row.iter().copied().map(goldilocks_word).collect())
            .collect(),
        opening_proof: opening
            .opening_proof
            .iter()
            .map(goldilocks_digest)
            .collect(),
    }
}

/// Converts a native `p3_fri::CommitPhaseProofStep` to [`P3CommitPhaseProofStepWitness`].
fn commit_phase_step_witness(
    step: &p3_fri::CommitPhaseProofStep<proof::Challenge, proof::ChallengeMmcs>,
) -> P3CommitPhaseProofStepWitness {
    P3CommitPhaseProofStepWitness {
        log_arity: step.log_arity,
        sibling_values: ext_vec_witness(&step.sibling_values),
        opening_proof: step.opening_proof.iter().map(goldilocks_digest).collect(),
    }
}

/// Converts a slice of degree-2 extension field elements to their `[u64; 2]`
/// limb encoding, element-wise.
fn ext_vec_witness(values: &[proof::Challenge]) -> Vec<[u64; 2]> {
    values.iter().map(challenge_word).collect()
}

/// One `Challenge` (degree-2 extension) element as its two basis-coefficient
/// limbs, each in canonical `u64` form.
fn challenge_word(value: &proof::Challenge) -> [u64; 2] {
    let coeffs = value.as_basis_coefficients_slice();
    [goldilocks_word(coeffs[0]), goldilocks_word(coeffs[1])]
}

/// An 8-element Poseidon2 digest as canonical `u64` limbs.
fn goldilocks_digest(digest: &[Goldilocks; 8]) -> [u64; 8] {
    core::array::from_fn(|i| goldilocks_word(digest[i]))
}

/// A single Goldilocks element in canonical `u64` form.
fn goldilocks_word(value: Goldilocks) -> u64 {
    value.as_canonical_u64()
}
