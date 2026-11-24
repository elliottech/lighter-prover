use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecursionError {
    #[error("signature count {actual} is not in 1..={expected}")]
    InvalidSignatureCount { expected: usize, actual: usize },
    #[error("proof has {actual_rows} rows, expected {expected_rows}")]
    InvalidTraceHeight {
        expected_rows: usize,
        actual_rows: usize,
    },
    #[error("p3-ecdsa proof verification failed: {0}")]
    InnerProof(#[from] p3_ecdsa::Error),
    #[error("proof (de)serialization failed: {0}")]
    Serialization(String),
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
