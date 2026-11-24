//! Public digest over a batch: the same width-12 Lighter Poseidon2 sponge as
//! `p3-schnorr` (count in lane 0, a domain tag in lane 1, rate-8 overwrite
//! absorption), with an ECDSA-specific domain tag and instance encoding.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_schnorr::poseidon::{POSEIDON_RATE, POSEIDON_WIDTH, poseidon2_permute};

use crate::types::{EcdsaInstance, INSTANCE_FIELD_ELEMENTS, TX_HASH_ELEMENTS};

pub const PUBLIC_DIGEST_WIDTH: usize = 4;
/// `"P3ECDSA1"`.
pub const PUBLIC_HASH_DOMAIN: u64 = 0x5033_4543_4453_4131;
pub const POSEIDON_BLOCKS_PER_INSTANCE: usize = INSTANCE_FIELD_ELEMENTS.div_ceil(POSEIDON_RATE);

pub fn initial_public_hash_state(count: usize) -> [Goldilocks; POSEIDON_WIDTH] {
    let mut state = [Goldilocks::ZERO; POSEIDON_WIDTH];
    state[0] = Goldilocks::from_usize(count);
    state[1] = Goldilocks::from_u64(PUBLIC_HASH_DOMAIN);
    state
}

pub fn instance_blocks(
    instance: &EcdsaInstance,
) -> [[Goldilocks; POSEIDON_RATE]; POSEIDON_BLOCKS_PER_INSTANCE] {
    let fields = instance.to_goldilocks_fields();
    core::array::from_fn(|b| {
        core::array::from_fn(|l| {
            fields
                .get(b * POSEIDON_RATE + l)
                .copied()
                .unwrap_or(Goldilocks::ZERO)
        })
    })
}

pub fn digest_instances<'a>(
    count: usize,
    instances: impl IntoIterator<Item = &'a EcdsaInstance>,
) -> [Goldilocks; PUBLIC_DIGEST_WIDTH] {
    absorb_records(
        initial_public_hash_state(count),
        instances
            .into_iter()
            .map(EcdsaInstance::to_goldilocks_fields),
    )
}

/// Absorbs each record in zero-padded rate blocks (overwrite mode) and squeezes
/// the digest.
fn absorb_records<T: AsRef<[Goldilocks]>>(
    mut state: [Goldilocks; POSEIDON_WIDTH],
    records: impl IntoIterator<Item = T>,
) -> [Goldilocks; PUBLIC_DIGEST_WIDTH] {
    for record in records {
        for block in record.as_ref().chunks(POSEIDON_RATE) {
            state[..POSEIDON_RATE].fill(Goldilocks::ZERO);
            state[..block.len()].copy_from_slice(block);
            poseidon2_permute(&mut state);
        }
    }
    core::array::from_fn(|i| state[i])
}

/// `"P3ECLNK1"`: domain tag of the link digest shared with the message-hash proof.
pub const LINK_HASH_DOMAIN: u64 = 0x5033_4543_4c4e_4b31;
/// `tx_hash (5) | keccak256 output as 8 little-endian u32 words`.
pub const LINK_FIELD_ELEMENTS: usize = TX_HASH_ELEMENTS + 8;
pub const LINK_BLOCKS: usize = LINK_FIELD_ELEMENTS.div_ceil(POSEIDON_RATE);

pub fn initial_link_hash_state(count: usize) -> [Goldilocks; POSEIDON_WIDTH] {
    let mut state = [Goldilocks::ZERO; POSEIDON_WIDTH];
    state[0] = Goldilocks::from_usize(count);
    state[1] = Goldilocks::from_u64(LINK_HASH_DOMAIN);
    state
}

pub fn link_fields(instance: &EcdsaInstance) -> [Goldilocks; LINK_FIELD_ELEMENTS] {
    let hash = instance.message_hash();
    let mut out = [Goldilocks::ZERO; LINK_FIELD_ELEMENTS];
    out[..TX_HASH_ELEMENTS].copy_from_slice(&instance.tx_hash);
    for (j, word) in hash.chunks(4).enumerate() {
        out[TX_HASH_ELEMENTS + j] =
            Goldilocks::from_u32(u32::from_le_bytes(word.try_into().unwrap()));
    }
    out
}

/// Digest binding each `tx_hash` to its `personal_sign` keccak output. Both the
/// ECDSA proof and the message-hash proof expose it; they must agree.
pub fn link_digest<'a>(
    count: usize,
    instances: impl IntoIterator<Item = &'a EcdsaInstance>,
) -> [Goldilocks; PUBLIC_DIGEST_WIDTH] {
    absorb_records(
        initial_link_hash_state(count),
        instances.into_iter().map(link_fields),
    )
}
