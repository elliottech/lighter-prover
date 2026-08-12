//! Plain, circuit-friendly representations of a Plonky3 STARK proof.
//!
//! The structs in this module mirror the corresponding `p3_commit` / `p3_fri` /
//! `p3_uni_stark` proof types field-for-field, but replace every field element with
//! its canonical `u64` (or `[u64; N]`) encoding:
//!
//! - `[u64; 8]` is a Poseidon2 digest: eight Goldilocks field elements in canonical
//!   form (`POSEIDON_DIGEST_WIDTH = 8`). Used for Merkle caps and MMCS opening proofs.
//! - `[u64; 2]` is a degree-2 binomial extension field element
//!   (`BinomialExtensionField<Goldilocks, 2>`), encoded as its two basis coefficients.
//!   Used for trace/quotient/FRI values sampled out-of-domain.
//! - Bare `u64` is a base-field (Goldilocks) element, used for proof-of-work witnesses.
//!
//! This decomposition exists so the proof can be loaded into a Plonky2 recursive
//! circuit as `Target`/`HashOutTarget` wires without re-deriving field arithmetic from
//! a generic Plonky3 proof type. The conversion from the native Plonky3 proof lives in
//! `api.rs` (`extract_inner_structured_proof` and friends); this module only defines
//! the destination shape.

use p3_schnorr::SignatureBatchPublicInputs;

/// A Merkle cap: the row of digests at the top of a Merkle tree, used as a succinct
/// commitment to a matrix (trace, quotient chunks, or a FRI commit-phase layer).
///
/// Mirrors `p3_symmetric::MerkleCap<Goldilocks, [Goldilocks; 8]>`. `roots` typically
/// has length 1 (a single root) unless the prover configured a wider cap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct P3MerkleCapWitness {
    /// Poseidon2 digests forming the cap, each as 8 canonical Goldilocks limbs.
    pub roots: Vec<[u64; 8]>,
}

/// A single MMCS (mixed matrix commitment scheme) opening: the leaf values opened at
/// a query index in every committed matrix, plus the Merkle path proving they hash to
/// the committed cap.
///
/// Mirrors `p3_commit::BatchOpening<Goldilocks, ValMmcs>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct P3BatchOpeningWitness {
    /// Leaf row opened from each committed matrix, in commitment order.
    pub opened_values: Vec<Vec<u64>>,
    /// Sibling digests along the Merkle path from the leaf to the cap, root-to-leaf.
    pub opening_proof: Vec<[u64; 8]>,
}

/// One round of the FRI commit phase: the sibling needed to fold the current codeword
/// at a query point, plus the Merkle path proving that sibling against the round's
/// commitment.
///
/// Mirrors `p3_fri::CommitPhaseProofStep<Challenge, ChallengeMmcs>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct P3CommitPhaseProofStepWitness {
    /// Base-2 logarithm of the folding arity for this round (normally 1, i.e. arity 2).
    pub log_arity: u8,
    /// Sibling extension-field value(s) needed to recompute the folded evaluation.
    pub sibling_values: Vec<[u64; 2]>,
    /// Merkle path proving the sibling values against this round's commitment.
    pub opening_proof: Vec<[u64; 8]>,
}

/// Everything FRI needs to verify a single query index: the input-layer batch
/// openings plus the per-round commit-phase folding steps.
///
/// Mirrors `p3_fri::QueryProof<Challenge, ChallengeMmcs, Vec<BatchOpening>>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct P3FriQueryWitness {
    /// Batch openings of the original (pre-folding) committed matrices.
    pub input_proof: Vec<P3BatchOpeningWitness>,
    /// Commit-phase folding step for each FRI round, outermost round first.
    pub commit_phase_openings: Vec<P3CommitPhaseProofStepWitness>,
}

/// A full FRI low-degree proof: round commitments, proof-of-work witnesses, the
/// queried openings, and the final constant-size polynomial.
///
/// Mirrors `p3_fri::FriProof<Challenge, ChallengeMmcs, Goldilocks, Vec<BatchOpening>>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct P3FriProofWitness {
    /// Merkle cap committed to at the end of each FRI folding round.
    pub commit_phase_commits: Vec<P3MerkleCapWitness>,
    /// Grinding (proof-of-work) witness for each commit-phase round, if configured.
    pub commit_pow_witnesses: Vec<u64>,
    /// One query proof per FRI query index sampled by the verifier.
    pub query_proofs: Vec<P3FriQueryWitness>,
    /// Coefficients of the final, fully-folded polynomial (sent in the clear).
    pub final_poly: Vec<[u64; 2]>,
    /// Grinding (proof-of-work) witness for the query phase.
    pub query_pow_witness: u64,
}

/// A complete Plonky3 STARK proof for the Schnorr signature-batch AIR, with every
/// field element decomposed into its `u64` limb encoding.
///
/// Mirrors `p3_uni_stark::Proof<SignatureBatchStarkConfig>`. Produced from the native
/// proof by `extract_inner_structured_proof` in `api.rs`; consumed by
/// `prove_transcript_verifier` to populate the Plonky2 recursive circuit's witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredP3Proof {
    /// Base-2 logarithm of the trace height (number of AIR rows).
    pub degree_bits: usize,
    /// Commitment to the (non-preprocessed) execution trace.
    pub trace_commitment: P3MerkleCapWitness,
    /// Commitment to the quotient polynomial, split into chunks.
    pub quotient_chunks_commitment: P3MerkleCapWitness,
    /// Commitment to the prover's random mask polynomial, if zero-knowledge blinding
    /// is enabled for this configuration.
    pub random_commitment: Option<P3MerkleCapWitness>,
    /// Trace columns evaluated at the out-of-domain sample point `zeta`.
    pub trace_local: Vec<[u64; 2]>,
    /// Trace columns evaluated at the next row, `zeta * g` (`g` the trace generator).
    pub trace_next: Option<Vec<[u64; 2]>>,
    /// Preprocessed (fixed) columns evaluated at `zeta`, if the AIR has any.
    pub preprocessed_local: Option<Vec<[u64; 2]>>,
    /// Preprocessed (fixed) columns evaluated at `zeta * g`, if the AIR has any.
    pub preprocessed_next: Option<Vec<[u64; 2]>>,
    /// Quotient polynomial chunks evaluated at `zeta`, one inner vec per chunk.
    pub quotient_chunks: Vec<Vec<[u64; 2]>>,
    /// Random mask polynomial evaluated at `zeta`, if zero-knowledge blinding is
    /// enabled for this configuration.
    pub random_values: Option<Vec<[u64; 2]>>,
    /// FRI proof that the opened values are consistent with low-degree commitments.
    pub opening_proof: P3FriProofWitness,
}

/// A self-contained witness bundle handed to the recursive verifier circuit: the
/// public inputs the proof attests to, the proof's original serialized bytes, and its
/// decomposed (circuit-friendly) form.
///
/// `proof_bytes` and `structured_proof` are *meant* to encode the same underlying
/// Plonky3 proof in two different shapes — `proof_bytes` as the serialized wire
/// format, `structured_proof` for direct loading into Plonky2 circuit wires — but
/// [`PreparedP3RecursiveWitness::new`] does not itself check that they agree, and the
/// fields being private only blocks mutation after construction, not construction
/// with mismatched values. [`crate::prepare_recursive_witness`]/
/// [`crate::prepare_recursive_witness_from_bytes`] derive both fields from the same
/// source proof and so are guaranteed consistent; callers who instead build a
/// `PreparedP3RecursiveWitness` by hand (as `new`'s test-only callers do, to construct
/// deliberately-malformed witnesses) get no such guarantee from the type itself.
/// `structured_proof` is the only field [`crate::prove_transcript_verifier`] actually
/// proves over, but that function independently calls
/// [`crate::verify_prepared_recursive_witness`] before proving, so a mismatched
/// `proof_bytes` is rejected there rather than silently ignored. Call
/// [`crate::verify_prepared_recursive_witness`] directly to check the binding without
/// also running the recursive prover.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedP3RecursiveWitness {
    public_inputs: SignatureBatchPublicInputs,
    proof_bytes: Vec<u8>,
    structured_proof: StructuredP3Proof,
}

impl PreparedP3RecursiveWitness {
    pub fn new(
        public_inputs: SignatureBatchPublicInputs,
        proof_bytes: Vec<u8>,
        structured_proof: StructuredP3Proof,
    ) -> Self {
        Self {
            public_inputs,
            proof_bytes,
            structured_proof,
        }
    }

    pub fn public_inputs(&self) -> &SignatureBatchPublicInputs {
        &self.public_inputs
    }

    pub fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }

    pub fn structured_proof(&self) -> &StructuredP3Proof {
        &self.structured_proof
    }

    pub fn into_parts(self) -> (SignatureBatchPublicInputs, Vec<u8>, StructuredP3Proof) {
        (self.public_inputs, self.proof_bytes, self.structured_proof)
    }
}
