//! Message-hash AIR: proves `z_i = keccak256(PERSONAL_SIGN_PREFIX || tx_hash_i)`
//! (Ethereum `personal_sign`) for a batch and exposes the link digest over
//! `(tx_hash_i, z_i)` that the ECDSA AIR recomputes, so a verifier ties the two
//! proofs together by checking their link digests are equal.
//!
//! Each signature is one Keccak-f permutation: [`ROWS_PER_SIGNATURE`] rows of the
//! `p3-keccak-air` columns (constraints vendored below so they can run inside a
//! wider trace), plus a Lighter Poseidon2 sponge running [`SLOTS_PER_ROW`]
//! rounds per row on extra columns, so its two blocks fit in the 24 rows.

pub mod prove;
pub mod trace;

use core::array;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use p3_keccak_air::{
    KeccakAir, KeccakCols, NUM_KECCAK_COLS, NUM_ROUNDS, NUM_ROUNDS_MIN_1, U64_LIMBS,
};
use p3_uni_stark::SubAirBuilder;

use crate::air::hash::{is_full_round, round_constants, sbox_round, sponge_step};
use crate::air::layout::{POSEIDON_ROUNDS, POSEIDON_WIDTH};
use crate::digest::{LINK_BLOCKS, LINK_HASH_DOMAIN, PUBLIC_DIGEST_WIDTH};
use crate::types::{MESSAGE_BYTES, PERSONAL_SIGN_PREFIX, TX_HASH_ELEMENTS};

pub const ROWS_PER_SIGNATURE: usize = NUM_ROUNDS;
pub const SLOTS_PER_ROW: usize = 3;
pub const LINK_ROUNDS: usize = LINK_BLOCKS * POSEIDON_ROUNDS;
const _: () = assert!(LINK_ROUNDS <= SLOTS_PER_ROW * ROWS_PER_SIGNATURE);

pub const RATE_BYTES: usize = 136;
pub const STATE_LIMBS: usize = 25 * U64_LIMBS;
/// First u16 limb of the message in the absorbed block (the prefix is 28 bytes).
pub const MSG_LIMB_START: usize = PERSONAL_SIGN_PREFIX.len() / 2;
const _: () = assert!(PERSONAL_SIGN_PREFIX.len().is_multiple_of(2));
const _: () = assert!(PERSONAL_SIGN_PREFIX.len() + MESSAGE_BYTES < RATE_BYTES);
/// `preimage[y][x][limb]` is column `PREIMAGE + 4 * (x + 5y) + limb`.
pub const PREIMAGE: usize = NUM_ROUNDS + 1;
pub const DIGEST_WORDS: usize = 8;

// ---- extra columns after the Keccak ones ----
/// Link sponge state before each of the row's slots (`SLOTS_PER_ROW x 12`).
pub const LINK_STATE: usize = NUM_KECCAK_COLS;
/// Keccak output as little-endian u32 words, held for the whole permutation.
pub const Z_WORDS: usize = LINK_STATE + SLOTS_PER_ROW * POSEIDON_WIDTH;
/// `(hi32(h_e) - (2^32 - 1))^-1` or 0: canonical encoding of each `tx_hash` element.
pub const H_INV: usize = Z_WORDS + DIGEST_WORDS;
pub const ACTIVE: usize = H_INV + TX_HASH_ELEMENTS;
pub const COUNTER: usize = ACTIVE + 1;
/// `(LINK_STATE[slot][l] + rc[l])^3` per slot: S-box helpers keeping the sponge at degree 5.
pub const LINK_SBOX: usize = COUNTER + 1;
pub const WIDTH: usize = LINK_SBOX + SLOTS_PER_ROW * POSEIDON_WIDTH;

/// `[n, L0..L3]`.
pub const NUM_PUBLIC_VALUES: usize = 1 + PUBLIC_DIGEST_WIDTH;
pub const MAX_CONSTRAINT_DEGREE: usize = 5;
/// FRI needs `log_height + log_blowup > log_final_poly_len` (2 and 5 in this config).
pub const MIN_LOG_HEIGHT: usize = 6;

pub const fn link_state(slot: usize) -> usize {
    LINK_STATE + slot * POSEIDON_WIDTH
}

pub const fn link_sbox(slot: usize) -> usize {
    LINK_SBOX + slot * POSEIDON_WIDTH
}

/// Round constants of the link sponge slot `t` (zero on absorb and idle slots).
pub fn slot_round_constants(t: usize) -> [u64; POSEIDON_WIDTH] {
    match slot_kind(t) {
        Some((_, round)) if round != 0 => round_constants(round),
        _ => [0; POSEIDON_WIDTH],
    }
}

/// The padded single rate block `prefix || msg || 0x01 .. 0x80` (Keccak-256 padding).
pub fn preimage_block(msg: &[u8; MESSAGE_BYTES]) -> [u8; RATE_BYTES] {
    let mut block = [0u8; RATE_BYTES];
    let p = PERSONAL_SIGN_PREFIX.len();
    block[..p].copy_from_slice(PERSONAL_SIGN_PREFIX);
    block[p..p + MESSAGE_BYTES].copy_from_slice(msg);
    block[p + MESSAGE_BYTES] = 0x01;
    block[RATE_BYTES - 1] |= 0x80;
    block
}

/// The fixed value of state limb `g` (`None` for message limbs).
pub fn constant_limb(g: usize) -> Option<u64> {
    if (MSG_LIMB_START..MSG_LIMB_START + MESSAGE_BYTES / 2).contains(&g) {
        return None;
    }
    let block = preimage_block(&[0; MESSAGE_BYTES]);
    Some(if 2 * g < RATE_BYTES {
        u64::from(block[2 * g]) | u64::from(block[2 * g + 1]) << 8
    } else {
        0
    })
}

/// Link sponge slot `t = 3 * row + k`: `(block, round)` or `None` when idle.
pub fn slot_kind(t: usize) -> Option<(usize, usize)> {
    (t < LINK_ROUNDS).then_some((t / POSEIDON_ROUNDS, t % POSEIDON_ROUNDS))
}

/// Column of output limb `q` (lanes `y = 0`, `x = 0..4`) on a final-step row.
pub fn output_col(q: usize) -> usize {
    p3_keccak_air::output_limb(q)
}

#[derive(Clone, Debug)]
pub struct MessageHashAir {
    pub capacity: usize,
}

impl MessageHashAir {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
    pub fn height(&self) -> usize {
        (self.capacity.max(1) * ROWS_PER_SIGNATURE)
            .next_power_of_two()
            .max(1 << MIN_LOG_HEIGHT)
    }
}

impl<F> BaseAir<F> for MessageHashAir {
    fn width(&self) -> usize {
        WIDTH
    }
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(MAX_CONSTRAINT_DEGREE)
    }
    fn num_public_values(&self) -> usize {
        NUM_PUBLIC_VALUES
    }
}

impl<AB: AirBuilder> Air<AB> for MessageHashAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.current_slice();
        let next_row = main.next_slice();
        let local: &KeccakCols<AB::Var> = local_row[..NUM_KECCAK_COLS].borrow();
        let next: &KeccakCols<AB::Var> = next_row[..NUM_KECCAK_COLS].borrow();
        // The upstream Keccak-f constraints, on the first `NUM_KECCAK_COLS` columns.
        KeccakAir {}.eval(&mut SubAirBuilder::<AB, KeccakAir, AB::Var>::new(
            builder,
            0..NUM_KECCAK_COLS,
        ));

        let v = |c: usize| -> AB::Expr { local_row[c].into() };
        let nv = |c: usize| -> AB::Expr { next_row[c].into() };
        let public: Vec<AB::Expr> = builder
            .public_values()
            .iter()
            .map(|p| (*p).into())
            .collect();
        let one = AB::Expr::ONE;
        let first_step: AB::Expr = local.step_flags[0].into();
        let final_step: AB::Expr = local.step_flags[NUM_ROUNDS_MIN_1].into();
        let not_final = one.clone() - final_step.clone();

        // ---- activity (a prefix of the permutations) and counter ----
        let active = v(ACTIVE);
        builder.assert_bool(active.clone());
        builder
            .when_transition()
            .assert_zero(not_final.clone() * (nv(ACTIVE) - active.clone()));
        builder
            .when_transition()
            .assert_zero(final_step.clone() * nv(ACTIVE) * (one.clone() - active.clone()));
        builder
            .when_first_row()
            .assert_zero(v(COUNTER) - active.clone());
        builder.when_transition().assert_zero(
            nv(COUNTER) - v(COUNTER) - Into::<AB::Expr>::into(next.step_flags[0]) * nv(ACTIVE),
        );
        builder
            .when_last_row()
            .assert_zero(v(COUNTER) - public[0].clone());

        // ---- the absorbed block is prefix || tx_hash bytes || padding ----
        let on_input = first_step * active.clone();
        for g in 0..STATE_LIMBS {
            if let Some(c) = constant_limb(g) {
                builder.assert_zero(on_input.clone() * (v(PREIMAGE + g) - AB::Expr::from_u64(c)));
            }
        }
        let two16 = AB::Expr::from_u64(1 << 16);
        let two32 = AB::Expr::from_u64(1 << 32);
        let mut fields: Vec<AB::Expr> = Vec::with_capacity(TX_HASH_ELEMENTS + DIGEST_WORDS);
        for e in 0..TX_HASH_ELEMENTS {
            let base = PREIMAGE + MSG_LIMB_START + 4 * e;
            let lo = v(base) + v(base + 1) * two16.clone();
            let hi = v(base + 2) + v(base + 3) * two16.clone();
            let off = hi.clone() - AB::Expr::from_u64(0xffff_ffff);
            builder.assert_zero(on_input.clone() * lo.clone() * (one.clone() - off * v(H_INV + e)));
            fields.push(lo + hi * two32.clone());
        }

        // ---- output words, held across the permutation ----
        for j in 0..DIGEST_WORDS {
            builder
                .when_transition()
                .assert_zero(not_final.clone() * (nv(Z_WORDS + j) - v(Z_WORDS + j)));
            let word = v(output_col(2 * j)) + v(output_col(2 * j + 1)) * two16.clone();
            builder.assert_zero(final_step.clone() * active.clone() * (v(Z_WORDS + j) - word));
            fields.push(v(Z_WORDS + j));
        }

        // ---- link sponge, SLOTS_PER_ROW rounds per row ----
        let xs: [[AB::Expr; POSEIDON_WIDTH]; SLOTS_PER_ROW] =
            array::from_fn(|k| array::from_fn(|l| v(link_state(k) + l)));
        for k in 0..SLOTS_PER_ROW {
            let x = &xs[k];
            let mut absorbed: [AB::Expr; POSEIDON_WIDTH] = array::from_fn(|_| AB::Expr::ZERO);
            let mut rc: [AB::Expr; POSEIDON_WIDTH] = array::from_fn(|_| AB::Expr::ZERO);
            let (mut on, mut full_sel, mut partial_sel) =
                (AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO);
            for i in 0..NUM_ROUNDS {
                let t = SLOTS_PER_ROW * i + k;
                let Some((b, round)) = slot_kind(t) else {
                    continue;
                };
                let flag: AB::Expr = local.step_flags[i].into();
                on += flag.clone();
                if round == 0 {
                    let mut st = x.clone();
                    sponge_step(&mut st, &fields, b, 0);
                    for l in 0..POSEIDON_WIDTH {
                        absorbed[l] += flag.clone() * st[l].clone();
                    }
                    continue;
                }
                if is_full_round(round) {
                    full_sel += flag.clone();
                } else {
                    partial_sel += flag.clone();
                }
                for (l, c) in slot_round_constants(t).into_iter().enumerate() {
                    if c != 0 {
                        rc[l] += flag.clone() * AB::Expr::from_u64(c);
                    }
                }
            }
            let cube: [AB::Expr; POSEIDON_WIDTH] = array::from_fn(|l| v(link_sbox(k) + l));
            for l in 0..POSEIDON_WIDTH {
                builder.assert_zero(cube[l].clone() - (x[l].clone() + rc[l].clone()).cube());
            }
            let rounded = sbox_round(&full_sel, &partial_sel, x, &rc, &cube);
            let stepped: [AB::Expr; POSEIDON_WIDTH] =
                array::from_fn(|l| absorbed[l].clone() + rounded[l].clone());
            for l in 0..POSEIDON_WIDTH {
                let expect = active.clone() * stepped[l].clone()
                    + (one.clone() - active.clone() * on.clone()) * x[l].clone();
                if k + 1 < SLOTS_PER_ROW {
                    builder.assert_zero(xs[k + 1][l].clone() - expect);
                } else {
                    builder
                        .when_transition()
                        .assert_zero(nv(link_state(0) + l) - expect);
                }
            }
        }
        for l in 0..POSEIDON_WIDTH {
            let init = match l {
                0 => public[0].clone(),
                1 => AB::Expr::from_u64(LINK_HASH_DOMAIN),
                _ => AB::Expr::ZERO,
            };
            builder
                .when_first_row()
                .assert_zero(xs[0][l].clone() - init);
        }
        for l in 0..PUBLIC_DIGEST_WIDTH {
            builder
                .when_last_row()
                .assert_zero(xs[0][l].clone() - public[1 + l].clone());
        }
    }
}

#[cfg(test)]
mod tests;
