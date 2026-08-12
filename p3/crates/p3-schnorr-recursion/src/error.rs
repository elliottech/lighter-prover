//! Error type for the recursive-witness preparation and Plonky2 circuit
//! layers, wrapping the underlying `p3-schnorr` proof error plus failures
//! specific to fixing the batch size and building/running the recursive
//! verifier circuit.

use p3_schnorr::SignatureBatchProofError;
use thiserror::Error;

/// Errors raised while preparing a P3 proof as a recursive witness, or while
/// building/running the Plonky2 transcript-verifier circuit over it.
#[derive(Debug, Error)]
pub enum RecursionError {
    #[error(
        "recursive wrapper accepts at most {expected} signatures, but public inputs claim {actual}"
    )]
    InvalidSignatureCount { expected: usize, actual: usize },
    #[error(
        "recursive wrapper is fixed to a {expected_rows}-row trace, but the proof has {actual_rows} rows"
    )]
    InvalidTraceHeight {
        expected_rows: usize,
        actual_rows: usize,
    },
    #[error("p3-schnorr proof verification failed: {0}")]
    P3ProofVerification(#[from] SignatureBatchProofError),
    #[error("prepared witness public inputs do not match the decoded proof")]
    PublicInputMismatch,
    #[error("prepared witness structured proof does not match the decoded proof")]
    StructuredProofMismatch,
    #[error("plonky2 witness assignment failed: {0}")]
    Plonky2Witness(String),
    #[error("plonky2 proof generation failed: {0}")]
    Plonky2Proof(String),
    #[error("unsupported P3 verifier feature in recursive circuit: {0}")]
    UnsupportedVerifierFeature(&'static str),
    #[error("recursive verifier invariant violated: {0}")]
    InvariantViolation(&'static str),
}
