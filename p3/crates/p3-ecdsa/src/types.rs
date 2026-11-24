//! Batch inputs: one [`EcdsaInstance`] per `(message, signature, public key)`.

use num_bigint::BigUint;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::secp::{self, Point};

/// Field elements one instance contributes to the public digest:
/// `tx_hash (5) | r (8 x u32) | s (8 x u32) | pk.x (8 x u32) | pk y-parity (1)`.
pub const INSTANCE_FIELD_ELEMENTS: usize = 30;
pub const TX_HASH_ELEMENTS: usize = 5;

/// `personal_sign` prefix for a [`MESSAGE_BYTES`]-byte message.
pub const PERSONAL_SIGN_PREFIX: &[u8; 28] = b"\x19Ethereum Signed Message:\n40";
/// The signed message: the 5 `tx_hash` elements as canonical little-endian `u64`s.
pub const MESSAGE_BYTES: usize = 8 * TX_HASH_ELEMENTS;

/// One ECDSA statement. The wallet signs the Poseidon `tx_hash` with Ethereum
/// `personal_sign` (see [`EcdsaInstance::message_scalar`]), so the block
/// circuit binds exactly the same transaction hash it does for Schnorr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EcdsaInstance {
    pub tx_hash: [Goldilocks; TX_HASH_ELEMENTS],
    /// Big-endian `r`.
    pub r: [u8; 32],
    /// Big-endian `s` (must be low-s).
    pub s: [u8; 32],
    /// SEC1 compressed public key (`0x02|0x03 || x`).
    pub public_key: [u8; 33],
}

impl EcdsaInstance {
    pub fn message_bytes(&self) -> [u8; MESSAGE_BYTES] {
        let mut out = [0u8; MESSAGE_BYTES];
        for (chunk, h) in out.chunks_mut(8).zip(&self.tx_hash) {
            chunk.copy_from_slice(&h.as_canonical_u64().to_le_bytes());
        }
        out
    }

    /// `keccak256(PERSONAL_SIGN_PREFIX || message_bytes)`.
    pub fn message_hash(&self) -> [u8; 32] {
        use tiny_keccak::{Hasher, Keccak};
        let mut k = Keccak::v256();
        k.update(PERSONAL_SIGN_PREFIX);
        k.update(&self.message_bytes());
        let mut out = [0u8; 32];
        k.finalize(&mut out);
        out
    }

    /// `z`: [`Self::message_hash`] as a big-endian integer.
    pub fn message_scalar(&self) -> BigUint {
        BigUint::from_bytes_be(&self.message_hash())
    }

    pub fn r_scalar(&self) -> BigUint {
        BigUint::from_bytes_be(&self.r)
    }

    pub fn s_scalar(&self) -> BigUint {
        BigUint::from_bytes_be(&self.s)
    }

    pub fn public_key_x(&self) -> BigUint {
        BigUint::from_bytes_be(&self.public_key[1..])
    }

    pub fn public_key_y_odd(&self) -> bool {
        self.public_key[0] == 0x03
    }

    pub fn public_key_point(&self) -> Option<Point> {
        if self.public_key[0] != 0x02 && self.public_key[0] != 0x03 {
            return None;
        }
        secp::decompress(&self.public_key_x(), self.public_key_y_odd())
    }

    /// Reference (non-circuit) verification with the same acceptance rules as the AIR.
    pub fn verify_reference(&self) -> bool {
        self.public_key_point().is_some_and(|q| {
            secp::ecdsa_verify(
                &self.message_scalar(),
                &self.r_scalar(),
                &self.s_scalar(),
                &q,
            )
        })
    }

    pub fn to_goldilocks_fields(&self) -> [Goldilocks; INSTANCE_FIELD_ELEMENTS] {
        let mut out = [Goldilocks::ZERO; INSTANCE_FIELD_ELEMENTS];
        out[..TX_HASH_ELEMENTS].copy_from_slice(&self.tx_hash);
        let mut cursor = TX_HASH_ELEMENTS;
        for bytes in [&self.r[..], &self.s[..], &self.public_key[1..]] {
            for j in 0..8 {
                let start = 32 - 4 * (j + 1);
                let word = u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap());
                out[cursor] = Goldilocks::from_u32(word);
                cursor += 1;
            }
        }
        out[cursor] = Goldilocks::from_bool(self.public_key_y_odd());
        out
    }

    /// Builds an instance from integers; `pk` is encoded compressed.
    pub fn from_parts(tx_hash: [Goldilocks; 5], r: &BigUint, s: &BigUint, pk: &Point) -> Self {
        let be32 = |v: &BigUint| -> [u8; 32] {
            let bytes = v.to_bytes_be();
            let mut out = [0u8; 32];
            out[32 - bytes.len()..].copy_from_slice(&bytes);
            out
        };
        let mut public_key = [0u8; 33];
        public_key[0] = if pk.y.bit(0) { 0x03 } else { 0x02 };
        public_key[1..].copy_from_slice(&be32(&pk.x));
        Self {
            tx_hash,
            r: be32(r),
            s: be32(s),
            public_key,
        }
    }
}
