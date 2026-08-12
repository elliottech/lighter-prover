//! The sponge construction used to commit to a [`Batch`](crate::types::Batch)
//! as a single public digest, built on the width-12 "Lighter" Poseidon2
//! permutation from [`crate::lighter_poseidon`].
//!
//! That permutation is bit-for-bit identical to the Plonky2 fork's native
//! `Poseidon2Hash` permutation used by the `zklighter` circuits, so a Plonky2
//! circuit can recompute (and chain) this digest with its native Poseidon2
//! gate instead of a dedicated wide gate. The per-round helpers it is built
//! from are generic over `R`, so [`crate::air::selectors`] reuses them to
//! build a one-round-per-row trace and to evaluate constraints over
//! `AB::Expr`/`AB::Var` — not just over concrete `Goldilocks` values.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use crate::lighter_poseidon::{
    LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS, LIGHTER_POSEIDON_INTERNAL_ROUNDS, LIGHTER_POSEIDON_RATE,
    LIGHTER_POSEIDON_ROUND_STATE_COUNT, LIGHTER_POSEIDON_WIDTH, lighter_poseidon2_permute,
};
use crate::types::{INSTANCE_FIELD_ELEMENTS, SchnorrInstance};

/// Poseidon2 state width (number of field elements per permutation call).
pub const POSEIDON_WIDTH: usize = LIGHTER_POSEIDON_WIDTH;
/// Sponge rate: field elements absorbed per permutation call (the remaining
/// `POSEIDON_WIDTH - POSEIDON_RATE` elements are capacity).
pub const POSEIDON_RATE: usize = LIGHTER_POSEIDON_RATE;
/// Width of the squeezed digest, in field elements (a Plonky2 `HashOut`).
pub const POSEIDON_DIGEST_WIDTH: usize = 4;
/// Number of rate-sized blocks one [`SchnorrInstance`]'s field-element row is
/// split into (rounded up).
pub const POSEIDON_BLOCKS_PER_INSTANCE: usize = INSTANCE_FIELD_ELEMENTS.div_ceil(POSEIDON_RATE);
/// Number of full (S-box on every lane) rounds before the partial rounds.
pub const POSEIDON_EXTERNAL_INITIAL_ROUNDS: usize = LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS;
/// Number of partial (S-box on lane 0 only) rounds.
pub const POSEIDON_INTERNAL_ROUNDS: usize = LIGHTER_POSEIDON_INTERNAL_ROUNDS;
/// Number of full rounds after the partial rounds.
pub const POSEIDON_EXTERNAL_FINAL_ROUNDS: usize = LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS;
/// Total recorded states per permutation: the initial linear layer plus one
/// state per round.
pub const POSEIDON_ROUND_STATE_COUNT: usize = LIGHTER_POSEIDON_ROUND_STATE_COUNT;

const _: () = assert!(
    POSEIDON_ROUND_STATE_COUNT
        == 1 + POSEIDON_EXTERNAL_INITIAL_ROUNDS
            + POSEIDON_INTERNAL_ROUNDS
            + POSEIDON_EXTERNAL_FINAL_ROUNDS
);

/// Domain-separation constant absorbed into lane 1 of the initial sponge
/// state, so this digest can't collide with a Poseidon2 hash computed for an
/// unrelated purpose over the same input bytes.
pub const PUBLIC_HASH_DOMAIN: u64 = 0x5033_5343_484e_5252;

/// The sponge's initial state before absorbing any instance: lane 0 holds the
/// batch's signature count, lane 1 the domain separator, the rest zero.
///
/// Binding `count` into the state means a digest computed for an N-signature
/// batch cannot be replayed as a valid digest for a different batch size.
pub fn initial_public_hash_state(count: usize) -> [Goldilocks; POSEIDON_WIDTH] {
    let mut state = [Goldilocks::ZERO; POSEIDON_WIDTH];
    state[0] = Goldilocks::from_usize(count);
    state[1] = Goldilocks::from_u64(PUBLIC_HASH_DOMAIN);
    state
}

/// Splits one instance's [`INSTANCE_FIELD_ELEMENTS`]-element row into
/// [`POSEIDON_RATE`]-sized blocks (zero-padded in the last block), ready to
/// absorb one at a time via [`absorb_block`].
pub fn instance_blocks(
    fields: &[Goldilocks; INSTANCE_FIELD_ELEMENTS],
) -> [[Goldilocks; POSEIDON_RATE]; POSEIDON_BLOCKS_PER_INSTANCE] {
    let mut blocks = [[Goldilocks::ZERO; POSEIDON_RATE]; POSEIDON_BLOCKS_PER_INSTANCE];
    let mut cursor = 0;
    for block in &mut blocks {
        let take = (INSTANCE_FIELD_ELEMENTS - cursor).min(POSEIDON_RATE);
        block[..take].copy_from_slice(&fields[cursor..cursor + take]);
        cursor += take;
    }
    blocks
}

/// Overwrites the rate lanes with `block` (additive absorption, since the
/// previous capacity lanes carry the running state) and permutes.
pub fn absorb_block(
    mut state: [Goldilocks; POSEIDON_WIDTH],
    block: [Goldilocks; POSEIDON_RATE],
) -> [Goldilocks; POSEIDON_WIDTH] {
    state[..POSEIDON_RATE].copy_from_slice(&block);
    poseidon2_permute(&mut state);
    state
}

/// Sponge-hashes `instances` (in iteration order) into the public digest bound
/// to a batch of size `count`. This is the host-side computation that
/// [`crate::hash::poseidon_public_digest`] wraps and that the AIR's hashing
/// constraints must reproduce row-by-row.
pub fn digest_instances<'a>(
    count: usize,
    instances: impl IntoIterator<Item = &'a SchnorrInstance>,
) -> crate::Result<[Goldilocks; POSEIDON_DIGEST_WIDTH]> {
    let mut state = initial_public_hash_state(count);
    for instance in instances {
        let fields = instance.to_goldilocks_fields()?;
        for block in instance_blocks(&fields) {
            state = absorb_block(state, block);
        }
    }
    Ok(core::array::from_fn(|i| state[i]))
}

/// The full public-digest Poseidon2 permutation (the Lighter width-12 one).
pub fn poseidon2_permute<R: PrimeCharacteristicRing>(state: &mut [R; POSEIDON_WIDTH]) {
    lighter_poseidon2_permute(state);
}

/// Applies the fixed 4x4 MDS-like matrix used by Poseidon2's external linear
/// layer to one 4-lane chunk, in place.
pub(crate) fn apply_mat4<R: PrimeCharacteristicRing>(x: &mut [R; 4]) {
    let t01 = x[0].dup() + x[1].dup();
    let t23 = x[2].dup() + x[3].dup();
    let t0123 = t01.dup() + t23.dup();
    let t01123 = t0123.dup() + x[1].dup();
    let t01233 = t0123 + x[3].dup();

    x[3] = t01233.dup() + x[0].double();
    x[1] = t01123.dup() + x[2].double();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;

    use super::*;

    /// Pins the public-digest permutation to the external `poseidon-hash`
    /// reference (the same implementation the Plonky2 fork's `Poseidon2Hash`
    /// matches); if they diverge, the AIR would silently verify a different
    /// hash than the one this module computes host-side.
    #[test]
    fn public_digest_permutation_matches_lighter_reference() {
        let input: [Goldilocks; POSEIDON_WIDTH] =
            Goldilocks::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let mut local = input;
        poseidon2_permute(&mut local);

        let mut upstream = input.map(|x| poseidon_hash::Goldilocks(x.as_canonical_u64()));
        poseidon_hash::permute(&mut upstream);
        let upstream = upstream.map(|x| Goldilocks::from_u64(x.to_canonical_u64()));

        assert_eq!(local, upstream);
    }
}
