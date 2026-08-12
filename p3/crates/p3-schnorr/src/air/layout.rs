// Column-index constants for the main trace, the preprocessed trace, and the
// public-input layout.
//
// `COL_*`/`PREPROCESSED_COL_*` constants are cumulative offsets: each one is
// defined as "the previous column's offset, plus the previous column's
// width", so the whole block must stay in declaration order — moving one
// constant shifts every constant after it. Columns that hold a multi-limb
// value (an `Fp5`, a `[u64; 8]` digest, etc.) name only their first column;
// the remaining limbs follow immediately after. Where a column's offset
// depends on a runtime parameter (e.g. one of `EC_WINDOW_STEPS` repeated
// blocks), a `const fn` (`ec_step_x_col`, `ec_add_x_den_inv_col`, etc.) computes
// it instead of a flat constant. See `air/mod.rs` for what these columns mean
// at the row-block level (hashing rows vs. EC rows).

/// Width of the squeezed public-transcript digest (mirrors [`crate::poseidon::POSEIDON_DIGEST_WIDTH`]).
pub const PUBLIC_DIGEST_WIDTH: usize = POSEIDON_DIGEST_WIDTH;
pub const PUBLIC_COUNT: usize = 0;
pub const PUBLIC_DIGEST_START: usize = 1;
pub const PUBLIC_VALUES: usize = 1 + PUBLIC_DIGEST_WIDTH;

// Wire format for the out-of-circuit "challenge hint": the prover-supplied
// `reconstructed_r`/decoded-point witnesses a verifier can't derive from the
// public inputs alone and must receive alongside the proof. See `hints.rs`
// for the encode/decode logic that uses these constants.
pub const SIGNATURE_CHALLENGE_HINT_WIRE_FORMAT_VERSION: u16 = 1;
pub const SIGNATURE_CHALLENGE_HINT_STATEMENT_ID: u16 = 1;
pub const ENCODED_POINT_DECODE_HINT_FIELD_ELEMENTS: usize = 4 * FP5_LIMBS;
pub const SIGNATURE_CHALLENGE_HINT_FIELD_ELEMENTS: usize =
    LIGHTER_POSEIDON_DIGEST_WIDTH + 2 * ENCODED_POINT_DECODE_HINT_FIELD_ELEMENTS;
pub const SIGNATURE_CHALLENGE_HINT_BYTES: usize =
    8 + 2 + 2 + SIGNATURE_CHALLENGE_HINT_FIELD_ELEMENTS * 8;
pub const SIGNATURE_CHALLENGE_HINT_BATCH_HEADER_BYTES: usize = 8 + 2 + 2 + 8;

const SIGNATURE_CHALLENGE_HINT_WIRE_MAGIC: [u8; 8] = *b"P3SNHNT\0";
const SIGNATURE_CHALLENGE_HINT_BATCH_WIRE_MAGIC: [u8; 8] = *b"P3SNHNS\0";

// Row-state columns, valid on every row regardless of hashing/EC sub-phase.
// `COL_PHASE` (0..=3) only has meaning while `COL_HASH_ACTIVE = 1`; see
// `air/mod.rs` for what each phase value represents.
pub const COL_ENABLED: usize = 0;
pub const COL_PHASE: usize = 1;
pub const COL_REMAINING_BLOCKS: usize = 2;

// Public-transcript Poseidon2 sponge state (hashing rows): absorbs
// `(msg_digest, signature, public_key)` blocks into the batch digest.
pub const COL_STATE_BEFORE: usize = 3;
pub const COL_STATE_AFTER: usize = COL_STATE_BEFORE + POSEIDON_WIDTH;
pub const COL_BLOCK: usize = COL_STATE_AFTER + POSEIDON_WIDTH;

// Phase 0/1 one-hot indicators and the Lighter-Poseidon challenge sponge
// state: re-derives `e = H(reconstructed_r, msg_digest)` from a witnessed
// `reconstructed_r`, run alongside the public-transcript hash.
pub const COL_CHALLENGE_FIRST: usize = COL_BLOCK + POSEIDON_RATE;
pub const COL_CHALLENGE_SECOND: usize = COL_CHALLENGE_FIRST + 1;
pub const COL_CHALLENGE_MSG_TAIL: usize = COL_CHALLENGE_SECOND + 1;
pub const COL_CHALLENGE_STATE_BEFORE: usize = COL_CHALLENGE_MSG_TAIL + 2;
pub const COL_CHALLENGE_STATE_AFTER: usize = COL_CHALLENGE_STATE_BEFORE + LIGHTER_POSEIDON_WIDTH;
pub const COL_CHALLENGE_BLOCK: usize = COL_CHALLENGE_STATE_AFTER + LIGHTER_POSEIDON_WIDTH;
pub const COL_CHALLENGE_DIGEST: usize = COL_CHALLENGE_BLOCK + LIGHTER_POSEIDON_RATE;

// Phase 2/3 modular reduction: reduces the challenge digest (reinterpreted as
// `SCALAR_U32_LIMBS` 32-bit chunks) modulo the scalar field order
// (`SCALAR_MODULUS_U32`) via quotient `COL_REDUCTION_Q` and per-limb carries,
// producing the canonical scalar that must equal the signature's own `e`.
pub const COL_REDUCTION_DIGEST: usize = COL_CHALLENGE_DIGEST + LIGHTER_POSEIDON_DIGEST_WIDTH;
pub const COL_REDUCTION_DIGEST_U32: usize = COL_REDUCTION_DIGEST + LIGHTER_POSEIDON_DIGEST_WIDTH;
pub const COL_REDUCTION_E_CHUNKS: usize = COL_REDUCTION_DIGEST_U32 + SCALAR_U32_LIMBS;
pub const COL_REDUCTION_Q: usize = COL_REDUCTION_E_CHUNKS + SCALAR_U32_LIMBS;
pub const COL_REDUCTION_CARRIES: usize = COL_REDUCTION_Q + 1;
pub const REDUCTION_CARRY_COUNT: usize = SCALAR_U32_LIMBS + 1;

// Windowed scalar-multiplication parameters: each 32-bit scalar chunk is
// split into `EC_WINDOWS_PER_CHUNK` windows of `EC_WINDOW_BITS` bits, and each
// window contributes one EC row with `EC_WINDOW_STEPS` double-and-add steps
// (one set of steps for `s`'s window, one for `e`'s).
pub const EC_WINDOW_BITS: usize = 4;
pub const EC_WINDOW_STEPS: usize = 2 * EC_WINDOW_BITS;
pub const EC_WINDOWS_PER_CHUNK: usize = REDUCTION_BITS_PER_U32 / EC_WINDOW_BITS;
pub const EC_SCALAR_WINDOWS: usize = SCALAR_U32_LIMBS * EC_WINDOWS_PER_CHUNK;

// Bit-decomposition witnesses for the reduction digest, used to extract
// individual bits/windows of the just-computed canonical scalar for the EC
// rows that follow.
pub const COL_REDUCTION_DIGEST_BITS: usize = COL_REDUCTION_CARRIES + REDUCTION_CARRY_COUNT;
pub const COL_REDUCTION_DIGEST_BIT: usize = COL_REDUCTION_DIGEST_BITS;
pub const COL_REDUCTION_DIGEST_WINDOW_ALL_ONES: usize = COL_REDUCTION_DIGEST_BITS + EC_WINDOW_BITS;
pub const COL_REDUCTION_DIGEST_CHUNK_ACC: usize = COL_REDUCTION_DIGEST_WINDOW_ALL_ONES + 1;
pub const COL_REDUCTION_DIGEST_HI_PREFIX_ACC: usize = COL_REDUCTION_DIGEST_CHUNK_ACC + 1;
pub const REDUCTION_BITS_PER_U32: usize = 32;
pub const EC_SCALAR_BITS: usize = SCALAR_U32_LIMBS * REDUCTION_BITS_PER_U32;

// Per-signature row-block sizing: hashing rows (one per Poseidon2 round) plus
// EC rows (one per scalar window).
pub const PUBLIC_HASH_ROWS_PER_SIGNATURE: usize =
    POSEIDON_BLOCKS_PER_INSTANCE * POSEIDON_ROUND_STATE_COUNT;
pub const ROWS_PER_SIGNATURE: usize = PUBLIC_HASH_ROWS_PER_SIGNATURE + EC_SCALAR_WINDOWS;
pub const EC_BIT_SELECTORS: usize = EC_WINDOWS_PER_CHUNK;
pub const EC_CHUNK_SELECTORS: usize = SCALAR_U32_LIMBS;

// Preprocessed (fixed, witness-independent) trace columns: one-hot selectors
// for "which Poseidon round/EC window/EC chunk is this row" plus precomputed
// multiples of the public generator `G`, used as the `s*G` addend table since
// `G` never changes across signatures or proofs.
pub const PREPROCESSED_COL_ROUND_SELECTORS: usize = 0;
pub const PREPROCESSED_COL_EC_BIT_SELECTORS: usize =
    PREPROCESSED_COL_ROUND_SELECTORS + POSEIDON_ROUND_STATE_COUNT;
pub const PREPROCESSED_COL_EC_CHUNK_SELECTORS: usize =
    PREPROCESSED_COL_EC_BIT_SELECTORS + EC_BIT_SELECTORS;
pub const PREPROCESSED_COL_EC_S_ADDEND_X: usize =
    PREPROCESSED_COL_EC_CHUNK_SELECTORS + EC_CHUNK_SELECTORS;
pub const PREPROCESSED_COL_EC_S_ADDEND_U: usize =
    PREPROCESSED_COL_EC_S_ADDEND_X + EC_WINDOW_BITS * FP5_LIMBS;
/// This EC row's window's fixed 4-bit slice of `SCALAR_MODULUS_U32` (the
/// scalar field order), used by [`crate::air::constraints::constrain_scalar_lt_modulus`]
/// to prove `s < scalar_order` and `e < scalar_order` window-by-window
/// against the same `COL_EC_S_BITS`/`COL_EC_E_BITS` already decomposed for
/// the scalar multiplication (one shared column since the modulus value for
/// a given window doesn't depend on which scalar is being checked).
pub const PREPROCESSED_COL_MODULUS_WINDOW: usize =
    PREPROCESSED_COL_EC_S_ADDEND_U + EC_WINDOW_BITS * FP5_LIMBS;
pub const PREPROCESSED_WIDTH: usize = PREPROCESSED_COL_MODULUS_WINDOW + 1;

// Point-decoding witnesses (hashing rows): reconstructing an `AffinePoint`
// from its public encoding requires an inverse and, since the encoding only
// determines a point up to sign/branch, a square-root witness disambiguating
// which curve point was meant. `R` is decoded during phase 0; the public key
// `pk` is decoded on the signature's last hashing row.
pub const COL_R_DECODE_INVERSE: usize = COL_REDUCTION_DIGEST_HI_PREFIX_ACC + 1;
pub const COL_R_DECODE_SQRT_DELTA: usize = COL_R_DECODE_INVERSE + FP5_LIMBS;
pub const COL_PK_DECODE_INVERSE: usize = COL_R_DECODE_SQRT_DELTA + FP5_LIMBS;
pub const COL_PK_DECODE_SQRT_DELTA: usize = COL_PK_DECODE_INVERSE + FP5_LIMBS;
pub const COL_PK_DECODE_OTHER_ROOT_SQRT: usize = COL_PK_DECODE_SQRT_DELTA + FP5_LIMBS;
pub const COL_R_AFFINE_X: usize = COL_PK_DECODE_OTHER_ROOT_SQRT + FP5_LIMBS;
pub const COL_PK_AFFINE_X: usize = COL_R_AFFINE_X + FP5_LIMBS;

// Row-type discriminants: see `air/mod.rs` for the hashing-rows/EC-rows state
// machine these participate in.
pub const COL_HASH_ACTIVE: usize = COL_PK_AFFINE_X + FP5_LIMBS;
pub const COL_EC_FINAL: usize = COL_HASH_ACTIVE + 1;

// EC rows: bit/window decompositions of the current 4-bit window of `s` and
// `e`, the carried-over signature scalar chunks and target `R`, and the
// windowed double-and-add accumulator (`COL_EC_ACC_*`) plus its
// intermediate/final state after this window's `EC_WINDOW_STEPS` steps
// (`COL_EC_STEP_RESULTS`, indexed by `ec_step_*_col`). `COL_EC_E_ADDEND_*`
// holds the on-the-fly-computed multiples of `pk` for this window (the `s*G`
// counterpart comes from the preprocessed table above, since `G` is fixed).
// The `*_DEN_INV` columns are the affine addition/doubling formulas' witnessed
// denominator inverses (see [`crate::affine::AffineAddWitness`]).
pub const COL_EC_S_BITS: usize = COL_EC_FINAL + 1;
pub const COL_EC_S_BIT: usize = COL_EC_S_BITS;
pub const COL_EC_E_BITS: usize = COL_EC_S_BITS + EC_WINDOW_BITS;
pub const COL_EC_E_BIT: usize = COL_EC_E_BITS;
pub const COL_EC_S_CHUNK_ACC: usize = COL_EC_E_BITS + EC_WINDOW_BITS;
pub const COL_EC_E_CHUNK_ACC: usize = COL_EC_S_CHUNK_ACC + 1;
/// Number of bits used to witness one window's shifted borrow-subtraction
/// difference (see [`COL_EC_S_LT_MODULUS_BORROW_IN`]): one more than
/// `EC_WINDOW_BITS`, since `scalar_window - modulus_window - borrow_in +
/// 2^EC_WINDOW_BITS` ranges over `[0, 2^(EC_WINDOW_BITS+1))`, not `[0,
/// 2^EC_WINDOW_BITS)`.
pub const EC_SCALAR_LT_MODULUS_DIFF_BITS: usize = EC_WINDOW_BITS + 1;
/// Per-window borrow-subtraction witness proving `s < SCALAR_MODULUS_U32`
/// and `e < SCALAR_MODULUS_U32` (canonicality of both signature scalars, not
/// merely their congruence mod the order — `e`'s congruence to the
/// recomputed challenge digest is separately proven by the modular
/// reduction in `eval.rs`, but that alone doesn't pin `e`/`s` to their
/// canonical residues; see
/// [`crate::air::constraints::constrain_scalar_lt_modulus`]): computed as
/// `scalar - SCALAR_MODULUS_U32`, so borrowing past the most significant
/// window means `scalar < modulus`. `COL_EC_S_LT_MODULUS_BORROW_IN`/
/// `COL_EC_E_LT_MODULUS_BORROW_IN` is the borrow *into* this window for
/// `s`/`e` respectively (carried window-to-window via
/// `constrain_ec_transition`, LSB window first matching trace order, reset
/// to 0 at each signature's first EC window), and
/// `COL_EC_S_LT_MODULUS_DIFF_BITS`/`COL_EC_E_LT_MODULUS_DIFF_BITS` is the
/// `EC_SCALAR_LT_MODULUS_DIFF_BITS`-bit decomposition of `(scalar_window -
/// modulus_window - borrow_in) + 2^EC_WINDOW_BITS` (shifted into `[0,
/// 2^(EC_WINDOW_BITS+1))` so it can be bit-decomposed at all) — its top bit
/// is 1 iff the shifted difference is `>= 2^EC_WINDOW_BITS`, i.e. iff the
/// *unshifted* difference was non-negative, so `1 - top_bit` gives this
/// window's borrow-out purely as an expression, with no separate borrow-out
/// witness needed. The final window's borrow-out must be 1 — a subtraction
/// that borrows past the most significant window only when the scalar is
/// strictly less than the modulus.
pub const COL_EC_S_LT_MODULUS_BORROW_IN: usize = COL_EC_E_CHUNK_ACC + 1;
pub const COL_EC_S_LT_MODULUS_DIFF_BITS: usize = COL_EC_S_LT_MODULUS_BORROW_IN + 1;
pub const COL_EC_E_LT_MODULUS_BORROW_IN: usize =
    COL_EC_S_LT_MODULUS_DIFF_BITS + EC_SCALAR_LT_MODULUS_DIFF_BITS;
pub const COL_EC_E_LT_MODULUS_DIFF_BITS: usize = COL_EC_E_LT_MODULUS_BORROW_IN + 1;
pub const COL_SIG_S_CHUNKS: usize =
    COL_EC_E_LT_MODULUS_DIFF_BITS + EC_SCALAR_LT_MODULUS_DIFF_BITS;
pub const COL_SIG_E_CHUNKS: usize = COL_SIG_S_CHUNKS + SCALAR_U32_LIMBS;
pub const COL_EC_TARGET_R_X: usize = COL_SIG_E_CHUNKS + SCALAR_U32_LIMBS;
pub const COL_EC_TARGET_R_U: usize = COL_EC_TARGET_R_X + FP5_LIMBS;
pub const COL_EC_E_ADDEND_X: usize = COL_EC_TARGET_R_U + FP5_LIMBS;
pub const COL_EC_E_ADDEND_U: usize = COL_EC_E_ADDEND_X + EC_WINDOW_BITS * FP5_LIMBS;
pub const COL_EC_ACC_X: usize = COL_EC_E_ADDEND_U + EC_WINDOW_BITS * FP5_LIMBS;
pub const COL_EC_ACC_U: usize = COL_EC_ACC_X + FP5_LIMBS;
pub const COL_EC_ACC_IS_IDENTITY: usize = COL_EC_ACC_U + FP5_LIMBS;
pub const EC_POINT_COLUMNS: usize = 2 * FP5_LIMBS + 1;
pub const COL_EC_STEP_RESULTS: usize = COL_EC_ACC_IS_IDENTITY + 1;
pub const COL_EC_MID_X: usize = ec_step_x_col(0);
pub const COL_EC_MID_U: usize = ec_step_u_col(0);
pub const COL_EC_MID_IS_IDENTITY: usize = ec_step_is_identity_col(0);
pub const COL_EC_NEXT_ACC_X: usize = ec_step_x_col(EC_WINDOW_STEPS - 1);
pub const COL_EC_NEXT_ACC_U: usize = ec_step_u_col(EC_WINDOW_STEPS - 1);
pub const COL_EC_NEXT_ACC_IS_IDENTITY: usize = ec_step_is_identity_col(EC_WINDOW_STEPS - 1);
pub const COL_EC_E_DOUBLE_X_DEN_INV: usize =
    COL_EC_STEP_RESULTS + EC_WINDOW_STEPS * EC_POINT_COLUMNS;
pub const COL_EC_E_DOUBLE_U_DEN_INV: usize = COL_EC_E_DOUBLE_X_DEN_INV + EC_WINDOW_BITS * FP5_LIMBS;
pub const COL_EC_ADD_X_DEN_INV: usize = COL_EC_E_DOUBLE_U_DEN_INV + EC_WINDOW_BITS * FP5_LIMBS;
pub const COL_EC_ADD_U_DEN_INV: usize = COL_EC_ADD_X_DEN_INV + EC_WINDOW_STEPS * FP5_LIMBS;
pub const COL_EC_S_ADD_X_DEN_INV: usize = ec_add_x_den_inv_col(0);
pub const COL_EC_S_ADD_U_DEN_INV: usize = ec_add_u_den_inv_col(0);
pub const COL_EC_E_ADD_X_DEN_INV: usize = ec_add_x_den_inv_col(1);
pub const COL_EC_E_ADD_U_DEN_INV: usize = ec_add_u_den_inv_col(1);

/// Total main-trace width (number of columns), the cumulative sum of every
/// column block above.
pub const TRACE_WIDTH: usize = COL_EC_ADD_U_DEN_INV + EC_WINDOW_STEPS * FP5_LIMBS;

/// Column of the `bit`-th preprocessed multiple of `G`'s x-coordinate (one of
/// the `2^EC_WINDOW_BITS` window table entries).
pub const fn preprocessed_s_addend_x_col(bit: usize) -> usize {
    PREPROCESSED_COL_EC_S_ADDEND_X + bit * FP5_LIMBS
}

/// Column of the `bit`-th preprocessed multiple of `G`'s `u`-coordinate.
pub const fn preprocessed_s_addend_u_col(bit: usize) -> usize {
    PREPROCESSED_COL_EC_S_ADDEND_U + bit * FP5_LIMBS
}

/// Column of the top bit of `s`'s (or `e`'s) `EC_SCALAR_LT_MODULUS_DIFF_BITS`-bit
/// shifted borrow-subtraction difference: `1` iff this window did not borrow
/// (see [`COL_EC_S_LT_MODULUS_BORROW_IN`]). Shared by
/// [`crate::air::constraints::constrain_scalar_lt_modulus`] and
/// `constrain_ec_transition`, which must agree on this offset.
pub const fn scalar_lt_modulus_top_bit_col(diff_bits_col: usize) -> usize {
    diff_bits_col + EC_WINDOW_BITS
}

/// Column of the `bit`-th on-the-fly multiple of `pk`'s x-coordinate for this window.
pub const fn ec_e_addend_x_col(bit: usize) -> usize {
    COL_EC_E_ADDEND_X + bit * FP5_LIMBS
}

/// Column of the `bit`-th on-the-fly multiple of `pk`'s `u`-coordinate for this window.
pub const fn ec_e_addend_u_col(bit: usize) -> usize {
    COL_EC_E_ADDEND_U + bit * FP5_LIMBS
}

/// Column of the accumulator's x-coordinate after double-and-add `step`.
pub const fn ec_step_x_col(step: usize) -> usize {
    COL_EC_STEP_RESULTS + step * EC_POINT_COLUMNS
}

/// Column of the accumulator's `u`-coordinate after double-and-add `step`.
pub const fn ec_step_u_col(step: usize) -> usize {
    ec_step_x_col(step) + FP5_LIMBS
}

/// Column of the accumulator's identity flag after double-and-add `step`.
pub const fn ec_step_is_identity_col(step: usize) -> usize {
    ec_step_u_col(step) + FP5_LIMBS
}

/// Column of the witnessed x-denominator inverse for doubling the `bit`-th
/// multiple of `pk` (building this window's `e*pk` addend table).
pub const fn ec_e_double_x_den_inv_col(bit: usize) -> usize {
    COL_EC_E_DOUBLE_X_DEN_INV + bit * FP5_LIMBS
}

/// Column of the witnessed `u`-denominator inverse for doubling the `bit`-th
/// multiple of `pk`.
pub const fn ec_e_double_u_den_inv_col(bit: usize) -> usize {
    COL_EC_E_DOUBLE_U_DEN_INV + bit * FP5_LIMBS
}

/// Column of the witnessed x-denominator inverse for double-and-add `step`
/// (`step` ranges over all `EC_WINDOW_STEPS`, covering both `s` and `e`).
pub const fn ec_add_x_den_inv_col(step: usize) -> usize {
    COL_EC_ADD_X_DEN_INV + step * FP5_LIMBS
}

/// Column of the witnessed `u`-denominator inverse for double-and-add `step`.
pub const fn ec_add_u_den_inv_col(step: usize) -> usize {
    COL_EC_ADD_U_DEN_INV + step * FP5_LIMBS
}

/// Number of [`crate::poseidon::POSEIDON_RATE`] lanes actually used by an
/// instance's final, possibly-partial absorption block.
const LAST_BLOCK_USED_LANES: usize = {
    let rem = INSTANCE_FIELD_ELEMENTS % POSEIDON_RATE;
    if rem == 0 { POSEIDON_RATE } else { rem }
};
// The public-transcript and Lighter-challenge sponges must take the same
// number of rounds per block, since both run in lockstep on the same rows
// (see `selectors.rs`); this compile-time check enforces that invariant.
const _: [(); POSEIDON_ROUND_STATE_COUNT] = [(); LIGHTER_POSEIDON_ROUND_STATE_COUNT];

// Precomputed inverses of `(d)(d-1)(d-2)(d-3)` factor products at the missing
// root, used by `selectors.rs`'s phase/round Lagrange-style indicator
// polynomials to select exactly one of `{0,1,2,3}` without a native equality
// gate.
const PHASE_LAST_INVERSE_DENOMINATOR: u64 = 0xd555555480000001;
const PHASE_ZERO_DENOMINATOR_INVERSE: u64 = 0x2aaaaaaa80000000;
const PHASE_ONE_DENOMINATOR_INVERSE: u64 = 0x7fffffff80000001;
const PHASE_TWO_DENOMINATOR_INVERSE: u64 = 0x7fffffff80000000;

/// `2^32`, the base used to split a Goldilocks element into two 32-bit limbs.
const U32_BASE: u64 = 1 << 32;
const U32_MASK: u64 = U32_BASE - 1;

/// ECgFp5 curve coefficients, duplicated from [`crate::affine`] so this file
/// doesn't need a public re-export; must stay equal to `ECGFP5_A`/`ECGFP5_B`/
/// `ECGFP5_B_MUL4` there.
const ECGFP5_A: [u64; FP5_LIMBS] = [2, 0, 0, 0, 0];
const ECGFP5_B: [u64; FP5_LIMBS] = [0, 263, 0, 0, 0];
const ECGFP5_B_MUL4: [u64; FP5_LIMBS] = [0, 1052, 0, 0, 0];

/// The ECgFp5 scalar field (group order) modulus, as 10 little-endian 32-bit
/// limbs, used by the challenge-reduction constraints in `eval.rs`.
const SCALAR_MODULUS_U32: [u64; SCALAR_U32_LIMBS] = [
    0x948bffe1, 0xe80fd996, 0xd724a09c, 0xe8885c39, 0xcfb80639, 0x7fffffe6, 0x00000016, 0x7ffffff1,
    0x80000007, 0x7ffffffd,
];
