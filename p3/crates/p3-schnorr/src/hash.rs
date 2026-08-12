use p3_goldilocks::Goldilocks;

use crate::error::Result;
use crate::poseidon::{POSEIDON_DIGEST_WIDTH, digest_instances};
use crate::types::Batch;

pub const PUBLIC_DIGEST_WIDTH: usize = POSEIDON_DIGEST_WIDTH;

/// Host-side Poseidon2 digest for the public input commitment.
pub fn poseidon_public_digest(batch: &Batch) -> Result<[Goldilocks; PUBLIC_DIGEST_WIDTH]> {
    digest_instances(batch.len(), batch.instances())
}
