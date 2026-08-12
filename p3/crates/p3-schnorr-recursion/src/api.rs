//! The recursion crate's public API: fixing the batch size the recursive
//! circuit supports, preparing a `p3-schnorr` proof as a recursive witness,
//! and the conversion from native Plonky3 proof types to this crate's
//! `u64`-encoded [`crate::types`] structs (see that module's docs for *why*
//! the encoding exists).
//!
//! [`prepare_recursive_witness`]/[`verify_prepared_recursive_witness`] always
//! re-verify the underlying `p3-schnorr` STARK proof natively before/after
//! touching the recursive-witness machinery — the recursive circuit
//! re-deriving the same checks is a separate concern from this crate ever
//! treating a proof as valid without that direct check.

use p3_field::{BasedVectorSpace, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_schnorr::proof::InnerSignatureBatchProof;
use p3_schnorr::{
    SignatureBatchProof, SignatureBatchProvingOutput, SignatureBatchPublicInputs, proof,
    verify_signature_batch_proof,
};

use crate::error::RecursionError;
use crate::types::{
    P3BatchOpeningWitness, P3CommitPhaseProofStepWitness, P3FriProofWitness, P3FriQueryWitness,
    P3MerkleCapWitness, PreparedP3RecursiveWitness, StructuredP3Proof,
};

/// Default batch capacity for callers that only need one fixed-capacity
/// recursive circuit. Each [`TranscriptVerifierCircuit`](crate::TranscriptVerifierCircuit)
/// is independently built for its own capacity (see
/// `build_transcript_verifier_circuit`) — a Plonky2 circuit's shape is static,
/// but nothing stops different call sites from each fixing their own
/// capacity, as long as every proof/witness for that circuit is prepared
/// with the trace height that capacity implies. The proof's public
/// signature count may be any value in `1..=capacity`.
pub const FIXED_RECURSIVE_SIGNATURE_COUNT: usize = 500;

/// Validates `output` (signature count within `capacity`, trace height
/// matching `capacity`, then a full native STARK verification) and
/// repackages it as a [`PreparedP3RecursiveWitness`]: the public inputs, the
/// proof's serialized bytes, and its structured (`u64`-encoded) form, ready
/// to feed into the Plonky2 circuit built for `capacity` signature slots.
pub fn prepare_recursive_witness(
    output: &SignatureBatchProvingOutput,
    capacity: usize,
) -> Result<PreparedP3RecursiveWitness, RecursionError> {
    ensure_signature_count(&output.public_inputs, capacity)?;
    ensure_proof_height(&output.proof, capacity)?;
    verify_signature_batch_proof(&output.public_inputs, &output.proof)?;
    let structured_proof = extract_structured_proof(&output.proof);
    Ok(PreparedP3RecursiveWitness::new(
        output.public_inputs,
        output.to_bytes()?,
        structured_proof,
    ))
}

/// [`prepare_recursive_witness`], starting from the serialized wire bytes of
/// a [`SignatureBatchProvingOutput`] instead of an in-memory one.
pub fn prepare_recursive_witness_from_bytes(
    bytes: &[u8],
    capacity: usize,
) -> Result<PreparedP3RecursiveWitness, RecursionError> {
    let output = SignatureBatchProvingOutput::from_bytes(bytes)?;
    prepare_recursive_witness(&output, capacity)
}

/// Re-derives a [`PreparedP3RecursiveWitness`] from its own `proof_bytes` and
/// checks every field agrees: the signature count fits within `capacity` and
/// the trace height matches it,
/// the decoded public inputs match [`PreparedP3RecursiveWitness::public_inputs`],
/// the freshly re-extracted structured proof matches
/// [`PreparedP3RecursiveWitness::structured_proof`] (catching any divergence
/// between the two encodings of the same proof), and the underlying STARK
/// proof itself verifies.
///
/// This is the integrity check for a witness received from an untrusted
/// source (e.g. read back off disk or over the network) before it's trusted
/// as input to the recursive circuit built for `capacity` signature slots.
pub fn verify_prepared_recursive_witness(
    witness: &PreparedP3RecursiveWitness,
    capacity: usize,
) -> Result<(), RecursionError> {
    ensure_signature_count(witness.public_inputs(), capacity)?;
    let decoded = SignatureBatchProof::from_bytes(witness.proof_bytes())?;
    ensure_proof_height(&decoded.proof, capacity)?;
    if decoded.public_inputs != *witness.public_inputs() {
        return Err(RecursionError::PublicInputMismatch);
    }
    if extract_structured_proof(&decoded.proof) != *witness.structured_proof() {
        return Err(RecursionError::StructuredProofMismatch);
    }
    verify_signature_batch_proof(&decoded.public_inputs, &decoded.proof)?;
    Ok(())
}

/// Converts a [`SignatureBatchProof`] to its `u64`-encoded [`StructuredP3Proof`]
/// form; see `types.rs` for what that encoding represents and why.
pub fn extract_structured_proof(proof: &SignatureBatchProof) -> StructuredP3Proof {
    extract_inner_structured_proof(proof.inner())
}

fn ensure_signature_count(
    public_inputs: &SignatureBatchPublicInputs,
    capacity: usize,
) -> Result<(), RecursionError> {
    let actual = public_inputs.signature_count();
    if actual == 0 || actual > capacity {
        return Err(RecursionError::InvalidSignatureCount {
            expected: capacity,
            actual,
        });
    }
    Ok(())
}

fn ensure_proof_height(proof: &SignatureBatchProof, capacity: usize) -> Result<(), RecursionError> {
    let expected_rows = p3_schnorr::air::trace_height_for_count(capacity);
    if proof.rows() != expected_rows {
        return Err(RecursionError::InvalidTraceHeight {
            expected_rows,
            actual_rows: proof.rows(),
        });
    }
    Ok(())
}

/// Field-for-field conversion from the native `p3_uni_stark::Proof` to
/// [`StructuredP3Proof`], decomposing every field/extension element into its
/// `u64` limb encoding via the `*_witness`/`*_word` helpers below.
fn extract_inner_structured_proof(proof: &InnerSignatureBatchProof) -> StructuredP3Proof {
    StructuredP3Proof {
        degree_bits: proof.degree_bits,
        trace_commitment: merkle_cap_witness(&proof.commitments.trace),
        quotient_chunks_commitment: merkle_cap_witness(&proof.commitments.quotient_chunks),
        random_commitment: proof.commitments.random.as_ref().map(merkle_cap_witness),
        trace_local: ext_vec_witness(&proof.opened_values.trace_local),
        trace_next: proof
            .opened_values
            .trace_next
            .as_ref()
            .map(|v| ext_vec_witness(v)),
        preprocessed_local: proof
            .opened_values
            .preprocessed_local
            .as_ref()
            .map(|v| ext_vec_witness(v)),
        preprocessed_next: proof
            .opened_values
            .preprocessed_next
            .as_ref()
            .map(|v| ext_vec_witness(v)),
        quotient_chunks: proof
            .opened_values
            .quotient_chunks
            .iter()
            .map(|chunk| ext_vec_witness(chunk))
            .collect(),
        random_values: proof
            .opened_values
            .random
            .as_ref()
            .map(|v| ext_vec_witness(v)),
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
