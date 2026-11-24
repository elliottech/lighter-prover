//! Column and row layout of the ECDSA batch AIR.
//!
//! Each signature owns a block of [`ROWS_PER_SIGNATURE`] consecutive rows:
//! 2 setup rows and [`GLV_BITS`] scalar-loop rows (one signed digit of each of
//! the four GLV half-scalars per row); the last loop row writes its result into
//! the `R + 2^GLV_BITS * offset` registers instead of the next row. The public-digest
//! Poseidon2 sponge runs in parallel on the first [`HASH_ROWS_PER_SIGNATURE`]
//! rows of the block (one round per row). A second sponge (the link digest over
//! `tx_hash` and the keccak message hash, see `crate::keccak`) runs alongside it
//! on the first [`LINK_ROWS_PER_SIGNATURE`] rows.

pub use crate::secp::{GLV_BITS, LIMBS};

pub const SETUP_ROWS: usize = 2;
pub const ROWS_PER_SIGNATURE: usize = SETUP_ROWS + GLV_BITS;
pub const EC_FIRST_ROW: usize = SETUP_ROWS;
pub const EC_LAST_ROW: usize = ROWS_PER_SIGNATURE - 1;

pub const POSEIDON_WIDTH: usize = p3_schnorr::poseidon::POSEIDON_WIDTH;
pub const POSEIDON_RATE: usize = p3_schnorr::poseidon::POSEIDON_RATE;
pub const POSEIDON_ROUNDS: usize = p3_schnorr::poseidon::POSEIDON_ROUND_STATE_COUNT;
pub const HASH_BLOCKS: usize = crate::digest::POSEIDON_BLOCKS_PER_INSTANCE;
pub const HASH_ROWS_PER_SIGNATURE: usize = HASH_BLOCKS * POSEIDON_ROUNDS;
const _: () = assert!(HASH_ROWS_PER_SIGNATURE <= ROWS_PER_SIGNATURE);
pub const LINK_ROWS_PER_SIGNATURE: usize = crate::digest::LINK_BLOCKS * POSEIDON_ROUNDS;
const _: () = assert!(LINK_ROWS_PER_SIGNATURE <= HASH_ROWS_PER_SIGNATURE);

/// The range-check table has 2^16 rows, so the trace is never shorter.
pub const RANGE_BITS: usize = 16;
pub const MIN_LOG_HEIGHT: usize = RANGE_BITS;

/// Chunks (16-bit limbs) of a GLV digit integer `B = (K + 2^L - 1) / 2`.
pub const DIGIT_LIMBS: usize = GLV_BITS.div_ceil(16);
pub const NUM_SCALARS: usize = 4;

/// Quotient limbs of a modular relation (`q < 2^261`).
pub const Q_LIMBS: usize = 17;
/// Carries of a relation, one per 32-bit group except the last.
pub const CARRIES: usize = 16;
/// Number of 32-bit coefficient groups (`2 * GROUPS - 1 = 33` coefficients).
pub const GROUPS: usize = CARRIES + 1;
pub const COEFFS: usize = 2 * GROUPS - 1;
pub const CARRY_OFFSET: u64 = 1 << 31;
pub const SLOTS: usize = 11;
pub const SLOT_WIDTH: usize = Q_LIMBS + 2 * CARRIES;
pub const WORK_VALUES: usize = 7;

// ---- main trace: per-signature registers (constant within a block) ----
/// `phi(Q).x = beta * Q.x`; `Q + phi(Q) = (-(Q.x + XPHI), -Q.y)` is used unreduced.
pub const XPHI: usize = 0;
/// `Q - phi(Q)`.
pub const TQ2X: usize = XPHI + LIMBS;
pub const TQ2Y: usize = TQ2X + LIMBS;
pub const QX: usize = TQ2Y + LIMBS;
pub const QY: usize = QX + LIMBS;
/// `R + 2^GLV_BITS * offset`, the expected scalar-loop result. (`R` itself is
/// only used on `Setup1` and lives in that row's `ACCX/ACCY`.)
pub const RPX: usize = QY + LIMBS;
pub const RPY: usize = RPX + LIMBS;
pub const SIG_R: usize = RPY + LIMBS;
pub const SIG_S: usize = SIG_R + LIMBS;
pub const DIGITS: usize = SIG_S + LIMBS;
pub const REG_LIMBS_END: usize = DIGITS + NUM_SCALARS * DIGIT_LIMBS;
/// Bytes of `z`'s limbs, `z[m] = ZLO[m] + 256 * ZHI[m]` (`z` is not stored), range-checked as
/// `b` and `256 * b`.
/// Keccak's output limb `15 - m` is the byte swap `ZHI[m] + 256 * ZLO[m]`.
pub const ZLO: usize = REG_LIMBS_END;
pub const ZHI: usize = ZLO + LIMBS;
pub const TX_HASH: usize = ZHI + LIMBS;
pub const R_WRAP: usize = TX_HASH + crate::types::TX_HASH_ELEMENTS;
pub const Y_PARITY: usize = R_WRAP + 1;
pub const Y_HALF: usize = Y_PARITY + 1;
/// 1 iff the scalar-loop result is the point at infinity; `RPX/RPY` are then unused.
pub const RP_INF: usize = Y_HALF + 1;
pub const REG_END: usize = RP_INF + 1;

// ---- main trace: per-row values ----
pub const ACCX: usize = REG_END;
pub const ACCY: usize = ACCX + LIMBS;
pub const WORK: usize = ACCY + LIMBS;
/// 1 iff this row's accumulator is the point at infinity (`ACCX/ACCY` are then unused).
pub const ACC_INF: usize = WORK + WORK_VALUES * LIMBS;
/// 1 iff `2 * acc + T_Q` is the point at infinity.
pub const ADD_Q_INF: usize = ACC_INF + 1;
/// 1 iff the row's result `2 * acc + T_Q + T_G` is the point at infinity.
pub const ADD_G_INF: usize = ADD_Q_INF + 1;
/// Digit products `b0 * b1` and `b2 * b3`, keeping the table selection at degree 2.
pub const DIGIT_PROD: usize = ADD_G_INF + 1;
pub const BITS: usize = DIGIT_PROD + 2;
pub const CHUNK_ACC: usize = BITS + NUM_SCALARS;
pub const HASH_STATE: usize = CHUNK_ACC + NUM_SCALARS;
pub const LINK_STATE: usize = HASH_STATE + POSEIDON_WIDTH;
/// `ACTIVE * P_LINK`, kept as a column to hold the link sponge's degree at 5.
pub const LINK_GATE: usize = LINK_STATE + POSEIDON_WIDTH;
/// `(HASH_STATE[l] + P_RC[l])^3`: S-box helper keeping the sponge at degree 5.
pub const HASH_SBOX: usize = LINK_GATE + 1;
/// `(LINK_STATE[l] + P_RC[l])^3`.
pub const LINK_SBOX: usize = HASH_SBOX + POSEIDON_WIDTH;
pub const ACTIVE: usize = LINK_SBOX + POSEIDON_WIDTH;
pub const COUNTER: usize = ACTIVE + 1;
pub const RANGE_MULT: usize = COUNTER + 1;
pub const SLOT_BASE: usize = RANGE_MULT + 1;
pub const MAIN_WIDTH: usize = SLOT_BASE + SLOTS * SLOT_WIDTH;

pub const fn work(i: usize) -> usize {
    WORK + i * LIMBS
}
pub const fn slot_q(s: usize) -> usize {
    SLOT_BASE + s * SLOT_WIDTH
}
pub const fn slot_carry_lo(s: usize) -> usize {
    slot_q(s) + Q_LIMBS
}
pub const fn slot_carry_hi(s: usize) -> usize {
    slot_carry_lo(s) + CARRIES
}
pub const fn digit_limbs(j: usize) -> usize {
    DIGITS + j * DIGIT_LIMBS
}

// ---- preprocessed trace ----
/// `BLOCK_SEL[b]`: round 0 (absorb block `b` + initial linear layer).
pub const P_BLOCK_SEL: usize = 0;
/// `ROUND_SEL[r-1]`: Poseidon2 round `r` for `r in 1..POSEIDON_ROUNDS`.
pub const P_ROUND_SEL: usize = P_BLOCK_SEL + HASH_BLOCKS;
pub const P_SIG_START: usize = P_ROUND_SEL + POSEIDON_ROUNDS - 1;
/// 1 iff this row and the next belong to the same signature block (or both to the tail padding).
pub const P_SAME_NEXT: usize = P_SIG_START + 1;
pub const P_SETUP0: usize = P_SAME_NEXT + 1;
pub const P_SETUP1: usize = P_SETUP0 + 1;
pub const P_EC: usize = P_SETUP1 + 1;
pub const P_EC_FIRST: usize = P_EC + 1;
pub const P_EC_LAST: usize = P_EC_FIRST + 1;
pub const P_CHUNK_START: usize = P_EC_LAST + 1;
pub const P_CHUNK_END: usize = P_CHUNK_START + 1;
pub const P_RANGE: usize = P_CHUNK_END + DIGIT_LIMBS;
/// 1 on the link sponge's rows of each signature block.
pub const P_LINK: usize = P_RANGE + 1;
/// This row's Poseidon2 round constants (zero on rows without an S-box round).
pub const P_RC: usize = P_LINK + 1;
pub const PREP_WIDTH: usize = P_RC + POSEIDON_WIDTH;

// ---- LogUp (permutation trace, extension columns) ----
pub const LOOKUPS_PER_HELPER: usize = 4;
/// Range-checked on every row: accumulator, work values, slot quotients and carries.
pub const ROW_LOOKUPS: usize = 2 * LIMBS + WORK_VALUES * LIMBS + SLOTS * SLOT_WIDTH;
/// Range-checked once per signature (registers are constant within a block):
/// all register limbs except `z`'s, `Y_HALF`, and each byte of `z` as `b` and `256 * b`.
pub const REG_LOOKUPS: usize = REG_LIMBS_END + 1 + 4 * LIMBS;
pub const ROW_HELPERS: usize = ROW_LOOKUPS.div_ceil(LOOKUPS_PER_HELPER);
pub const REG_HELPERS: usize = REG_LOOKUPS.div_ceil(LOOKUPS_PER_HELPER);
pub const PERM_TABLE: usize = ROW_HELPERS + REG_HELPERS;
pub const PERM_ACC: usize = PERM_TABLE + 1;
pub const PERM_WIDTH: usize = PERM_ACC + 1;

/// `[n, D0..D3, L0..L3]`: count, public digest, link digest.
pub const NUM_PUBLIC_VALUES: usize = 1 + 2 * crate::digest::PUBLIC_DIGEST_WIDTH;
pub const MAX_CONSTRAINT_DEGREE: usize = LOOKUPS_PER_HELPER + 1;

/// Trace height for a batch capacity.
pub fn log_height_for_capacity(capacity: usize) -> usize {
    let rows = (capacity * ROWS_PER_SIGNATURE).max(1);
    rows.next_power_of_two()
        .trailing_zeros()
        .max(MIN_LOG_HEIGHT as u32) as usize
}
