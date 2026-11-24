//! One-round-per-row view of the Lighter Poseidon2 permutation and the
//! instance-to-sponge-lane mapping, shared by the AIR and trace generation.

use p3_field::PrimeCharacteristicRing;
use p3_schnorr::lighter_poseidon::{
    LIGHTER_EXTERNAL_CONSTANTS, LIGHTER_INTERNAL_CONSTANTS, LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS,
    LIGHTER_POSEIDON_INTERNAL_ROUNDS, external_linear_layer, internal_linear_layer,
    lighter_external_full_round, lighter_external_initial_linear_layer,
    lighter_internal_partial_round,
};

use super::layout::*;
use crate::digest::LINK_FIELD_ELEMENTS;
use crate::types::{INSTANCE_FIELD_ELEMENTS, TX_HASH_ELEMENTS};

/// Whether round `round` (`1..POSEIDON_ROUNDS`) is a full (external) round.
pub const fn is_full_round(round: usize) -> bool {
    let r = round - 1;
    r < LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS
        || r >= LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS + LIGHTER_POSEIDON_INTERNAL_ROUNDS
}

/// Round constants added before the S-box of round `round`; zero on lanes a
/// partial round leaves without an S-box.
pub fn round_constants(round: usize) -> [u64; POSEIDON_WIDTH] {
    let r = round - 1;
    let half = LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS;
    let internal = LIGHTER_POSEIDON_INTERNAL_ROUNDS;
    if r < half {
        LIGHTER_EXTERNAL_CONSTANTS[r]
    } else if r < half + internal {
        let mut rc = [0; POSEIDON_WIDTH];
        rc[0] = LIGHTER_INTERNAL_CONSTANTS[r - half];
        rc
    } else {
        LIGHTER_EXTERNAL_CONSTANTS[r - internal]
    }
}

/// `(Σ full-round selectors, Σ partial-round selectors)` from a preprocessed row.
pub fn round_kind_selectors<R: PrimeCharacteristicRing>(prep: &[R]) -> (R, R) {
    let (mut full, mut partial) = (R::ZERO, R::ZERO);
    for round in 1..POSEIDON_ROUNDS {
        let sel = prep[P_ROUND_SEL + round - 1].dup();
        if is_full_round(round) {
            full += sel;
        } else {
            partial += sel;
        }
    }
    (full, partial)
}

/// `full_sel · full_round(state) + partial_sel · partial_round(state)` using
/// the row's round constants `rc` and the helper cubes `cube = (state + rc)^3`,
/// so the result has degree 4 instead of 8.
pub fn sbox_round<R: PrimeCharacteristicRing>(
    full_sel: &R,
    partial_sel: &R,
    state: &[R; POSEIDON_WIDTH],
    rc: &[R; POSEIDON_WIDTH],
    cube: &[R; POSEIDON_WIDTH],
) -> [R; POSEIDON_WIDTH] {
    let sbox = |l: usize| cube[l].square() * (state[l].dup() + rc[l].dup());
    let mut full: [R; POSEIDON_WIDTH] = core::array::from_fn(sbox);
    external_linear_layer(&mut full);
    let mut partial: [R; POSEIDON_WIDTH] =
        core::array::from_fn(|l| if l == 0 { sbox(0) } else { state[l].dup() });
    internal_linear_layer(&mut partial);
    core::array::from_fn(|l| full_sel.dup() * full[l].dup() + partial_sel.dup() * partial[l].dup())
}

/// Applies Poseidon2 round `round` (`1..POSEIDON_ROUNDS`) in place.
pub fn poseidon_round<R: PrimeCharacteristicRing>(round: usize, state: &mut [R; POSEIDON_WIDTH]) {
    let r = round - 1;
    let half = LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS;
    let internal = LIGHTER_POSEIDON_INTERNAL_ROUNDS;
    if r < half {
        lighter_external_full_round(state, &LIGHTER_EXTERNAL_CONSTANTS[r]);
    } else if r < half + internal {
        lighter_internal_partial_round(state, LIGHTER_INTERNAL_CONSTANTS[r - half]);
    } else {
        lighter_external_full_round(state, &LIGHTER_EXTERNAL_CONSTANTS[r - internal]);
    }
}

/// Round 0: the absorbed state's initial linear layer.
pub fn poseidon_round0<R: PrimeCharacteristicRing>(state: &mut [R; POSEIDON_WIDTH]) {
    lighter_external_initial_linear_layer(state);
}

/// The instance's digest field elements, as expressions in register columns.
pub fn instance_fields<R: PrimeCharacteristicRing>(row: &[R]) -> Vec<R> {
    let two16 = R::from_u64(1 << 16);
    let pair = |c: usize| row[c].dup() + row[c + 1].dup() * two16.dup();
    let mut out = Vec::with_capacity(INSTANCE_FIELD_ELEMENTS);
    out.extend((0..TX_HASH_ELEMENTS).map(|i| row[TX_HASH + i].dup()));
    for start in [SIG_R, SIG_S, QX] {
        for j in 0..8 {
            out.push(pair(start + 2 * j));
        }
    }
    out.push(row[Y_PARITY].dup());
    debug_assert_eq!(out.len(), INSTANCE_FIELD_ELEMENTS);
    out
}

/// The link digest's field elements: `tx_hash` and the keccak output words,
/// rebuilt from the bytes of `z` (`z` is the big-endian reading of the output).
pub fn link_fields<R: PrimeCharacteristicRing>(row: &[R]) -> Vec<R> {
    let two8 = R::from_u64(1 << 8);
    let two16 = R::from_u64(1 << 16);
    let keccak_limb =
        |q: usize| row[ZHI + LIMBS - 1 - q].dup() + row[ZLO + LIMBS - 1 - q].dup() * two8.dup();
    let mut out = Vec::with_capacity(LINK_FIELD_ELEMENTS);
    out.extend((0..TX_HASH_ELEMENTS).map(|i| row[TX_HASH + i].dup()));
    for j in 0..LIMBS / 2 {
        out.push(keccak_limb(2 * j) + keccak_limb(2 * j + 1) * two16.dup());
    }
    debug_assert_eq!(out.len(), LINK_FIELD_ELEMENTS);
    out
}

/// One sponge step: round 0 absorbs `block` of `fields` into the rate lanes and
/// applies the initial linear layer; rounds `1..POSEIDON_ROUNDS` permute.
pub fn sponge_step<R: PrimeCharacteristicRing>(
    state: &mut [R; POSEIDON_WIDTH],
    fields: &[R],
    block: usize,
    round: usize,
) {
    if round == 0 {
        for l in 0..POSEIDON_RATE {
            state[l] = block_lane(fields, block, l);
        }
        poseidon_round0(state);
    } else {
        poseidon_round(round, state);
    }
}

pub fn block_lane<R: PrimeCharacteristicRing>(fields: &[R], block: usize, lane: usize) -> R {
    fields
        .get(block * POSEIDON_RATE + lane)
        .map(|v| v.dup())
        .unwrap_or(R::ZERO)
}
