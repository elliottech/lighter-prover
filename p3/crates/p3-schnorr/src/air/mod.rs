//! The AIR (algebraic intermediate representation) that constrains a batch of
//! Schnorr signature verifications, plus the trace/column-layout/witness-hint
//! plumbing it needs.
//!
//! # Module structure
//!
//! These files are stitched together with `include!` (see the bottom of this
//! file), not `mod` — they share one scope and are split purely for editing
//! convenience, not as separate namespaces:
//! - `layout.rs` — column-index constants for the main and preprocessed traces.
//! - `eval.rs` — [`SignatureBatchAir`], the `Air`/`BaseAir` impl that evaluates
//!   every constraint against `(local, next)` trace rows.
//! - `selectors.rs` — phase/round one-hot selector polynomials shared by the
//!   trace generator and the constraint evaluator.
//! - `constraints.rs` — constraint helpers factored out of `eval.rs` (EC row
//!   transitions, point decoding, scalar-chunk binding).
//! - `fp5_expr.rs` — in-circuit `GF(p^5)` arithmetic over `AB::Expr`.
//! - `preprocessed.rs` — builds the fixed (witness-independent) preprocessed
//!   trace: round/window selectors and the generator's precomputed multiples.
//! - `trace_types.rs`, `hints.rs`, `trace_generation.rs` — host-side witness
//!   computation that fills in the main trace matching what `eval.rs` constrains.
//! - `tests.rs` — unit tests exercising the above.
//!
//! # Row layout: one signature per `ROWS_PER_SIGNATURE` rows
//!
//! Each signature instance occupies a fixed-size, two-part block of rows
//! (`ROWS_PER_SIGNATURE = PUBLIC_HASH_ROWS_PER_SIGNATURE + EC_SCALAR_WINDOWS`):
//!
//! 1. **Hashing rows** (`COL_HASH_ACTIVE = 1`): one row per Poseidon2 round,
//!    absorbing `(msg_digest, signature, public_key)` into the public
//!    transcript digest, then re-deriving the Fiat-Shamir challenge with the
//!    Lighter hash over a witnessed `reconstructed_r`, then reducing that
//!    challenge digest modulo the scalar field order. `COL_PHASE` (0..=3)
//!    tracks which sub-step of this is active on a given hashing row:
//!    phase 0 absorbs `reconstructed_r`, phase 1 absorbs the signature's `e`
//!    and carries the message tail, phase 2 absorbs the remaining reduction
//!    chunks, and phase 3 (the final phase) finishes the chunk-wise modular
//!    reduction and decodes the public key.
//! 2. **EC rows** (`COL_HASH_ACTIVE = 0`, `ec_active = COL_ENABLED - COL_HASH_ACTIVE`):
//!    one row per 4-bit window step, performing windowed double-and-add to
//!    compute `R' = s*G + e*pk`. `s*G`'s addends come from a *preprocessed*
//!    table of precomputed multiples of the public generator `G` (since `G`
//!    is a fixed constant); `e*pk`'s addends are recomputed each row from the
//!    witnessed `pk` (since the public key varies per signature). The last EC
//!    row (`COL_EC_FINAL = 1`) asserts the accumulator equals the `R` value
//!    decoded during the hashing rows, closing the loop between the
//!    Fiat-Shamir challenge and the group equation it must satisfy.
//!
//! `SignatureBatchAir::eval` is the single source of truth for all of this;
//! `trace_generation.rs` must reproduce it exactly off-circuit. See
//! [`SignatureBatchAir`] in `eval.rs` for the per-constraint detail.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

use crate::affine::{
    AffineAddWitness, AffineDoubleWitness, AffinePoint, ECGFP5_B_MUL2, Fp5, fp5_zero,
};
use crate::error::Result;
use crate::lighter_poseidon::{
    LIGHTER_EXTERNAL_CONSTANTS, LIGHTER_INTERNAL_CONSTANTS, LIGHTER_POSEIDON_DIGEST_WIDTH,
    LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS, LIGHTER_POSEIDON_RATE,
    LIGHTER_POSEIDON_ROUND_STATE_COUNT, LIGHTER_POSEIDON_WIDTH, lighter_external_full_round,
    lighter_external_initial_linear_layer, lighter_internal_partial_round,
};
use crate::poseidon::{
    POSEIDON_BLOCKS_PER_INSTANCE, POSEIDON_DIGEST_WIDTH, POSEIDON_EXTERNAL_INITIAL_ROUNDS,
    POSEIDON_INTERNAL_ROUNDS, POSEIDON_RATE, POSEIDON_ROUND_STATE_COUNT, POSEIDON_WIDTH,
    initial_public_hash_state, instance_blocks,
};
use crate::types::{Batch, FP5_LIMBS, INSTANCE_FIELD_ELEMENTS, MAX_SIGNATURES, SCALAR_U32_LIMBS};

include!("layout.rs");
include!("eval.rs");
include!("selectors.rs");
include!("constraints.rs");
include!("fp5_expr.rs");
include!("preprocessed.rs");
include!("trace_types.rs");
include!("hints.rs");
include!("trace_generation.rs");
include!("tests.rs");
