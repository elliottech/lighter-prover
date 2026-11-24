//! Deterministic valid batches for tests and benchmarks.

use num_bigint::BigUint;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use crate::secp::{self, constants};
use crate::types::EcdsaInstance;

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn scalar(&mut self) -> BigUint {
        let words: Vec<u64> = (0..4).map(|_| self.next()).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        BigUint::from_bytes_le(&bytes) % &constants().n
    }
}

/// `count` valid low-s signatures over random `tx_hash` values.
pub fn deterministic_batch(count: usize, seed: u64) -> Vec<EcdsaInstance> {
    let mut rng = SplitMix(seed);
    let c = constants();
    let one = BigUint::from(1u32);
    (0..count)
        .map(|_| {
            loop {
                let (sk, nonce) = (rng.scalar(), rng.scalar());
                let tx_hash: [Goldilocks; 5] = core::array::from_fn(|_| {
                    Goldilocks::from_u64(rng.next() % 0xffff_ffff_0000_0001)
                });
                let z = EcdsaInstance::from_parts(tx_hash, &one, &one, &c.g).message_scalar();
                let (Some(q), Some((r, s))) =
                    (secp::mul(&c.g, &sk), secp::ecdsa_sign(&sk, &z, &nonce))
                else {
                    continue;
                };
                break EcdsaInstance::from_parts(tx_hash, &r, &s, &q);
            }
        })
        .collect()
}
