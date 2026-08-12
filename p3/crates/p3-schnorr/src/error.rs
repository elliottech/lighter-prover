//! Error type shared by signature-batch decoding, challenge-hint wire parsing, and
//! affine ECgFp5 arithmetic.
//!
//! Variants are split by stage (challenge-hint wire format vs. proof public-input
//! wire format vs. EC arithmetic) so callers can tell decode/parse failures (caused
//! by untrusted input) apart from arithmetic failures (which should never happen for
//! well-formed input and may indicate a bug or a malicious proof).

use thiserror::Error;

/// Errors raised while preparing, decoding, or verifying a Schnorr signature batch.
#[derive(Debug, Error)]
pub enum Error {
    #[error("batch is empty")]
    EmptyBatch,

    #[error("batch has {actual} signatures, exceeding maximum {max}")]
    TooManySignatures { actual: usize, max: usize },

    #[error("batch has {actual} signatures, exceeding capacity {capacity}")]
    BatchExceedsCapacity { actual: usize, capacity: usize },

    #[error("challenge hint count {actual} does not match batch size {expected}")]
    ChallengeHintCountMismatch { actual: usize, expected: usize },

    #[error("challenge hint batch has {actual} hints, exceeding maximum {max}")]
    TooManyChallengeHints { actual: usize, max: usize },

    #[error("challenge hints must be supplied when the `reference` feature is disabled")]
    MissingChallengeHints,

    #[error("invalid challenge-hint wire magic")]
    InvalidChallengeHintWireMagic,

    #[error("invalid challenge-hint batch wire magic")]
    InvalidChallengeHintBatchWireMagic,

    #[error("unsupported challenge-hint wire format version {actual}; expected {expected}")]
    UnsupportedChallengeHintWireFormatVersion { expected: u16, actual: u16 },

    #[error("unsupported challenge-hint statement id {actual}; expected {expected}")]
    UnsupportedChallengeHintStatementId { expected: u16, actual: u16 },

    #[error("wire payload is truncated")]
    TruncatedWirePayload,

    #[error("challenge-hint wire payload has {trailing} trailing bytes")]
    TrailingChallengeHintWireBytes { trailing: usize },

    #[error("reduced challenge scalar does not match the signature challenge scalar")]
    ChallengeScalarMismatch,

    #[error(
        "signature scalar {field} is not canonical (not strictly less than the scalar field order)"
    )]
    NonCanonicalSignatureScalar { field: &'static str },

    #[error("invalid encoded ECgFp5 point in {field}")]
    InvalidEncodedPoint { field: &'static str },

    #[error("affine ECgFp5 arithmetic failed while computing {operation}")]
    EcArithmeticFailed { operation: &'static str },

    #[error("non-canonical Goldilocks limb {limb} in {field}: 0x{value:016x}")]
    NonCanonicalGoldilocksLimb {
        field: &'static str,
        limb: usize,
        value: u64,
    },

    #[cfg(feature = "reference")]
    #[error("reference crypto error: {0}")]
    ReferenceCrypto(#[from] goldilocks_crypto::CryptoError),

    #[cfg(feature = "reference")]
    #[error("reference encoding error: {0}")]
    ReferenceEncoding(String),
}

pub type Result<T> = core::result::Result<T, Error>;
