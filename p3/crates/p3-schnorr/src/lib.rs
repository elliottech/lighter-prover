//! Plonky3 ECgFp5 Schnorr batch verifier.
//!
//! The AIR constrains the public digest, Lighter challenge hash, challenge
//! reduction, decoded EC points, and affine Schnorr equation. The proof API in
//! [`proof`] is the intended verifier/prover boundary; [`air`]
//! remains exposed for trace layout tests and benchmark plumbing while the
//! implementation is hardened.

pub mod affine;
pub mod air;
pub mod error;
pub mod hash;
pub mod lighter_poseidon;
pub mod poseidon;
pub mod proof;
pub mod reference;
pub mod schnorr;
pub mod types;
pub(crate) mod wire_reader;

pub use air::{
    ENCODED_POINT_DECODE_HINT_FIELD_ELEMENTS, EncodedPointDecodeHint,
    SIGNATURE_CHALLENGE_HINT_BATCH_HEADER_BYTES, SIGNATURE_CHALLENGE_HINT_BYTES,
    SIGNATURE_CHALLENGE_HINT_FIELD_ELEMENTS, SIGNATURE_CHALLENGE_HINT_STATEMENT_ID,
    SIGNATURE_CHALLENGE_HINT_WIRE_FORMAT_VERSION, SignatureChallengeHint,
    signature_challenge_hints_from_bytes, signature_challenge_hints_to_bytes,
};
pub use error::{Error, Result};
pub use hash::{PUBLIC_DIGEST_WIDTH, poseidon_public_digest};
#[cfg(feature = "reference")]
pub use proof::prepare_signature_batch_with_public_digest;
pub use proof::{
    PreparedSignatureBatch, PreparedSignatureBatchVerifier, SIGNATURE_BATCH_FRI_PROFILE_ID,
    SIGNATURE_BATCH_PUBLIC_INPUT_BYTES, SIGNATURE_BATCH_STATEMENT_ID,
    SIGNATURE_BATCH_WIRE_FORMAT_VERSION, SignatureBatchProof, SignatureBatchProofError,
    SignatureBatchProvingOutput, SignatureBatchPublicInputs, prepare_signature_batch_verifier,
    prepare_signature_batch_verifier_for_capacity, prove_signature_batch,
    prove_signature_batch_with_capacity, prove_signature_batch_with_challenge_hints,
    prove_signature_batch_with_challenge_hints_and_capacity, verify_signature_batch_proof,
    verify_signature_batch_proof_preprocessed,
};
pub use types::{Batch, Fp5Bytes, SchnorrInstance, SignatureBytes};
