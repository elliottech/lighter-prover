//! Plonky2 recursive verifier for `p3-ecdsa` LogUp proofs, adapted from
//! `p3-schnorr-recursion`.
//!
//! It provides host-side witness preparation and a fixed-capacity Plonky2
//! circuit that replays the `p3-ecdsa` LogUp transcript (trace, permutation,
//! quotient commitments), enforces the OODS / quotient consistency check, and
//! verifies the fixed-shape P3 MMCS / FRI opening proof.

// Vendored from elliottech/p3-lighter-circuits-internal; kept lint-clean via
// targeted allows rather than diverging from the upstream sources.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::needless_question_mark)]
#![allow(unused_variables)]

mod api;
mod error;
#[cfg(feature = "plonky2-nightly")]
mod plonky2;
#[cfg(feature = "plonky2-nightly")]
pub use p3_schnorr_recursion::poseidon2_gate;
mod types;

pub use api::{
    FIXED_RECURSIVE_SIGNATURE_COUNT, extract_message_hash_structured_proof,
    extract_structured_proof, prepare_batch_recursive_witness, prepare_recursive_witness,
    verify_prepared_batch_witness, verify_prepared_recursive_witness,
};
pub use error::RecursionError;
#[cfg(feature = "plonky2-nightly")]
pub use plonky2::{
    BatchVerifierCircuit, FixedVerifierMetadata, InnerProofKind, MerkleCapTargets8,
    TranscriptVerifierCircuit, TranscriptVerifierTargets, build_batch_verifier_circuit,
    build_transcript_verifier_circuit, prove_batch_verifier, prove_transcript_verifier,
};
#[cfg(all(feature = "plonky2-nightly", feature = "insecure-test-only"))]
pub use plonky2::{
    InsecurePublicInputShellCircuit, build_insecure_public_input_shell_circuit,
    prove_insecure_public_input_shell,
};
pub use types::{
    BatchRecursiveWitness, P3BatchOpeningWitness, P3CommitPhaseProofStepWitness, P3FriProofWitness,
    P3FriQueryWitness, P3MerkleCapWitness, PreparedP3RecursiveWitness, StructuredP3Proof,
};
