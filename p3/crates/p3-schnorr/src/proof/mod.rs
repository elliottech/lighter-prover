//! The proof/verifier boundary: STARK configuration, proving/verifying entry
//! points, and the stable wire formats for proofs and public inputs.
//!
//! Like `air/`, these files are `include!`d into one compilation unit, not
//! separate modules:
//! - `config.rs` — the concrete Plonky3 STARK configuration (hash, MMCS,
//!   FRI parameters, challenger) `SignatureBatchAir` is proved/verified under.
//! - `types.rs` — [`SignatureBatchPublicInputs`], [`SignatureBatchProof`], and
//!   the other public-facing wrapper types.
//! - `api.rs` — [`prove_signature_batch`], [`verify_signature_batch_proof`],
//!   and the rest of the proving/verifying entry points.
//! - `wire.rs` — shared little-endian (de)serialization helpers for the
//!   public-input and proof wire formats.
//!
//! This is the *intended* external boundary for callers of this crate — see
//! the crate-level doc comment in `lib.rs`. [`crate::air`] is exposed
//! separately mainly for trace-layout tests and benchmarking.

use bincode::Options;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks, default_goldilocks_poseidon2_16};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    PcsError, PreprocessedVerifierKey, Proof, StarkConfig, VerificationError,
    prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed,
};
use thiserror::Error;

use crate::air::{
    PUBLIC_DIGEST_WIDTH, PUBLIC_VALUES, SignatureBatchAir, SignatureBatchTrace,
    SignatureChallengeHint, generate_trace, generate_trace_with_capacity,
    generate_trace_with_challenge_hints, generate_trace_with_challenge_hints_and_capacity,
    trace_height_for_count,
};
use crate::hash::poseidon_public_digest;
use crate::types::{Batch, MAX_SIGNATURES};

include!("config.rs");
include!("types.rs");
include!("api.rs");
include!("wire.rs");
include!("tests.rs");
