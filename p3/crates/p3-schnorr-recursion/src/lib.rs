//! Recursion-oriented wrapper APIs for `p3-schnorr`.
//!
//! This crate provides the fixed-500 statement boundary, host-side witness
//! preparation, and a Plonky2 circuit that replays the `p3-schnorr`
//! transcript, enforces the real OODS / quotient consistency check, and
//! verifies the fixed-shape P3 MMCS / FRI opening proof for a
//! `p3_uni_stark::Proof`.

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
pub mod poseidon2_gate;
mod types;

pub use api::{
    FIXED_RECURSIVE_SIGNATURE_COUNT, extract_structured_proof, prepare_recursive_witness,
    prepare_recursive_witness_from_bytes, verify_prepared_recursive_witness,
};
pub use error::RecursionError;
#[cfg(feature = "plonky2-nightly")]
pub use plonky2::{
    FixedVerifierMetadata, MerkleCapTargets8, TranscriptVerifierCircuit, TranscriptVerifierTargets,
    build_transcript_verifier_circuit, prove_transcript_verifier,
};
#[cfg(all(feature = "plonky2-nightly", feature = "insecure-test-only"))]
pub use plonky2::{
    InsecurePublicInputShellCircuit, build_insecure_public_input_shell_circuit,
    prove_insecure_public_input_shell,
};
pub use types::{
    P3BatchOpeningWitness, P3CommitPhaseProofStepWitness, P3FriProofWitness, P3FriQueryWitness,
    P3MerkleCapWitness, PreparedP3RecursiveWitness, StructuredP3Proof,
};
