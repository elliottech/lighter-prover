//! Wire-shaped types for one Schnorr instance and a batch of them, plus the
//! byte<->field-element conversions used to build STARK public inputs.
//!
//! `Fp5Bytes`/`SignatureBytes` store raw little-endian bytes (as received over
//! the wire) rather than field elements directly, so decoding can be deferred
//! until the values are checked against the Goldilocks modulus (see
//! [`Fp5Bytes::to_goldilocks_limbs`]) — this keeps "received untrusted bytes"
//! and "validated field element" distinct types.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::error::{Error, Result};

/// Number of Goldilocks limbs in one `GF(p^5)` element (message digest, public
/// key, or challenge).
pub const FP5_LIMBS: usize = 5;
/// Byte length of one `GF(p^5)` element: 5 limbs x 8 bytes.
pub const FP5_BYTES: usize = FP5_LIMBS * 8;
/// Number of 32-bit chunks in one scalar's byte encoding.
pub const SCALAR_U32_LIMBS: usize = FP5_BYTES / 4;
/// Byte length of a signature: the `(s, e)` scalar pair, each `FP5_BYTES` long.
pub const SIGNATURE_BYTES: usize = 2 * FP5_BYTES;
/// Number of Goldilocks field elements after splitting a signature into 32-bit
/// chunks (see [`SignatureBytes::to_u32_field_elements`]).
pub const SIGNATURE_FIELD_ELEMENTS: usize = SIGNATURE_BYTES / 4;
/// Field-element width of one [`SchnorrInstance`]'s public input row:
/// `msg_digest || signature (as u32 chunks) || public_key`.
pub const INSTANCE_FIELD_ELEMENTS: usize = FP5_LIMBS + SIGNATURE_FIELD_ELEMENTS + FP5_LIMBS;
/// Largest batch size this crate's fixed-shape AIR/proof pipeline supports.
pub const MAX_SIGNATURES: usize = 510;

/// Raw little-endian bytes of a `GF(p^5)` element (message digest or public key),
/// not yet validated against the Goldilocks modulus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fp5Bytes(pub [u8; FP5_BYTES]);

impl Fp5Bytes {
    pub const ZERO: Self = Self([0; FP5_BYTES]);

    pub fn from_array(bytes: [u8; FP5_BYTES]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; FP5_BYTES] {
        &self.0
    }

    /// Decodes each 8-byte limb as a little-endian `u64` and validates it is a
    /// canonical (reduced) Goldilocks value.
    ///
    /// `field` is the name reported in [`Error::NonCanonicalGoldilocksLimb`] if a
    /// limb is `>= Goldilocks::ORDER_U64`; this is the boundary where untrusted
    /// wire bytes either become a real field element or are rejected.
    pub fn to_goldilocks_limbs(self, field: &'static str) -> Result<[Goldilocks; FP5_LIMBS]> {
        let mut out = [Goldilocks::ZERO; FP5_LIMBS];
        for (i, chunk) in self.0.chunks_exact(8).enumerate() {
            let value = u64::from_le_bytes(chunk.try_into().expect("chunks_exact yields 8 bytes"));
            if value >= Goldilocks::ORDER_U64 {
                return Err(Error::NonCanonicalGoldilocksLimb {
                    field,
                    limb: i,
                    value,
                });
            }
            out[i] = Goldilocks::from_u64(value);
        }
        Ok(out)
    }
}

/// Raw little-endian bytes of a signature: the response scalar `s` (first
/// [`FP5_BYTES`] bytes) followed by the challenge scalar `e`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignatureBytes(pub [u8; SIGNATURE_BYTES]);

impl SignatureBytes {
    pub const ZERO: Self = Self([0; SIGNATURE_BYTES]);

    pub fn from_array(bytes: [u8; SIGNATURE_BYTES]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; SIGNATURE_BYTES] {
        &self.0
    }

    /// Splits into the `(s, e)` scalar byte pair.
    pub fn split(self) -> (Fp5Bytes, Fp5Bytes) {
        let mut s = [0u8; FP5_BYTES];
        let mut e = [0u8; FP5_BYTES];
        s.copy_from_slice(&self.0[..FP5_BYTES]);
        e.copy_from_slice(&self.0[FP5_BYTES..]);
        (Fp5Bytes(s), Fp5Bytes(e))
    }

    /// Splits the signature into 4-byte chunks and lifts each to a Goldilocks
    /// element via [`Goldilocks::from_u32`].
    ///
    /// Unlike [`Fp5Bytes::to_goldilocks_limbs`], this never fails: every `u32`
    /// chunk is automatically canonical (Goldilocks's modulus exceeds `2^32`).
    /// Scalars are carried through the AIR's public inputs as 32-bit chunks
    /// rather than 64-bit `GF(p^5)` limbs because they are *scalars* (mod the
    /// curve's group order, not mod `p`), and Goldilocks can't validate them —
    /// only an in-circuit range/recomposition check over `u32` chunks can.
    pub fn to_u32_field_elements(self) -> [Goldilocks; SIGNATURE_FIELD_ELEMENTS] {
        core::array::from_fn(|i| {
            let start = i * 4;
            let value = u32::from_le_bytes(
                self.0[start..start + 4]
                    .try_into()
                    .expect("signature chunks are 4 bytes"),
            );
            Goldilocks::from_u32(value)
        })
    }
}

/// One signature to be verified: a message digest, a signature, and the
/// claimed signer's public key, all in their raw wire-byte form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchnorrInstance {
    pub msg_digest: Fp5Bytes,
    pub signature: SignatureBytes,
    pub public_key: Fp5Bytes,
}

impl SchnorrInstance {
    pub fn new(msg_digest: Fp5Bytes, signature: SignatureBytes, public_key: Fp5Bytes) -> Self {
        Self {
            msg_digest,
            signature,
            public_key,
        }
    }

    /// Lays out this instance as one row of [`INSTANCE_FIELD_ELEMENTS`] Goldilocks
    /// elements (`msg_digest || signature_u32_chunks || public_key`), validating
    /// `msg_digest` and `public_key` are canonical along the way.
    pub fn to_goldilocks_fields(self) -> Result<[Goldilocks; INSTANCE_FIELD_ELEMENTS]> {
        let msg = self.msg_digest.to_goldilocks_limbs("msg_digest")?;
        let signature = self.signature.to_u32_field_elements();
        let pk = self.public_key.to_goldilocks_limbs("public_key")?;

        let mut out = [Goldilocks::ZERO; INSTANCE_FIELD_ELEMENTS];
        out[..FP5_LIMBS].copy_from_slice(&msg);
        out[FP5_LIMBS..FP5_LIMBS + SIGNATURE_FIELD_ELEMENTS].copy_from_slice(&signature);
        out[FP5_LIMBS + SIGNATURE_FIELD_ELEMENTS..].copy_from_slice(&pk);
        Ok(out)
    }
}

/// A non-empty batch of at most [`MAX_SIGNATURES`] [`SchnorrInstance`]s to be
/// verified together by one STARK proof.
///
/// The non-empty and max-size invariants are enforced once at construction
/// ([`Batch::new`]) so every downstream consumer (AIR trace generation, proof
/// preparation) can assume a valid batch size without re-checking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Batch {
    instances: Vec<SchnorrInstance>,
}

impl Batch {
    /// Fails with [`Error::EmptyBatch`] or [`Error::TooManySignatures`] if
    /// `instances` doesn't fit in `1..=MAX_SIGNATURES`.
    pub fn new(instances: Vec<SchnorrInstance>) -> Result<Self> {
        if instances.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if instances.len() > MAX_SIGNATURES {
            return Err(Error::TooManySignatures {
                actual: instances.len(),
                max: MAX_SIGNATURES,
            });
        }
        Ok(Self { instances })
    }

    pub fn instances(&self) -> &[SchnorrInstance] {
        &self.instances
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Flattens every instance's [`SchnorrInstance::to_goldilocks_fields`] row
    /// into one contiguous public-input vector, instance-major order.
    pub fn to_goldilocks_fields(&self) -> Result<Vec<Goldilocks>> {
        let mut out = Vec::with_capacity(self.instances.len() * INSTANCE_FIELD_ELEMENTS);
        for instance in &self.instances {
            out.extend(instance.to_goldilocks_fields()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;

    use super::*;

    #[test]
    fn signature_bytes_are_bound_as_u32_chunks() {
        let mut signature = [0u8; SIGNATURE_BYTES];
        signature[..8].copy_from_slice(&Goldilocks::ORDER_U64.to_le_bytes());
        signature[8..16].copy_from_slice(&u64::MAX.to_le_bytes());

        let instance = SchnorrInstance::new(
            Fp5Bytes::ZERO,
            SignatureBytes::from_array(signature),
            Fp5Bytes::ZERO,
        );
        let fields = instance.to_goldilocks_fields().unwrap();

        assert_eq!(fields.len(), INSTANCE_FIELD_ELEMENTS);
        assert_eq!(
            fields[FP5_LIMBS].as_canonical_u64(),
            Goldilocks::ORDER_U64 & 0xffff_ffff
        );
        assert_eq!(
            fields[FP5_LIMBS + 1].as_canonical_u64(),
            Goldilocks::ORDER_U64 >> 32
        );
        assert_eq!(fields[FP5_LIMBS + 2].as_canonical_u64(), 0xffff_ffff);
        assert_eq!(fields[FP5_LIMBS + 3].as_canonical_u64(), 0xffff_ffff);
    }
}
