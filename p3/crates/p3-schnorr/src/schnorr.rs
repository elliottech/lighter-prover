//! The Schnorr verification equation over ECgFp5: `R = s*G + e*P`, and the
//! Fiat-Shamir challenge derivation `e = H(R, msg_digest)` that a valid
//! signature's `e` must reproduce.
//!
//! `(s, e)` is the wire encoding of a Lighter-style Schnorr signature: `s` is
//! the response scalar and `e` is the challenge scalar (rather than the more
//! common `(R, s)` encoding), so verification reconstructs `R` from `s`, `e`,
//! and the public key `P`, then re-derives the challenge from that `R` and
//! checks it equals `e`. See [`crate::air::constraints`] for the in-circuit
//! version of this same check.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use crate::lighter_poseidon::hash_to_quintic_extension;
use crate::types::FP5_LIMBS;

/// Number of field elements in the challenge hash preimage: `reconstructed_r`
/// followed by `msg_digest`, each [`FP5_LIMBS`] elements.
pub const SCHNORR_CHALLENGE_PREIMAGE_ELEMENTS: usize = FP5_LIMBS * 2;

/// Builds the preimage `(reconstructed_r, msg_digest)` hashed to derive the
/// Fiat-Shamir challenge scalar.
pub fn challenge_preimage(
    reconstructed_r: [Goldilocks; FP5_LIMBS],
    msg_digest: [Goldilocks; FP5_LIMBS],
) -> [Goldilocks; SCHNORR_CHALLENGE_PREIMAGE_ELEMENTS] {
    let mut preimage = [Goldilocks::ZERO; SCHNORR_CHALLENGE_PREIMAGE_ELEMENTS];
    preimage[..FP5_LIMBS].copy_from_slice(&reconstructed_r);
    preimage[FP5_LIMBS..].copy_from_slice(&msg_digest);
    preimage
}

/// Derives the Fiat-Shamir challenge `e = H(reconstructed_r, msg_digest)` as an
/// `Fp5` element, using the Lighter hash-to-quintic-extension construction.
///
/// A signature is valid iff this matches the signature's own `e` (after both are
/// reduced to scalars); see [`verification_witness`] for the reference check.
pub fn challenge_fp5(
    reconstructed_r: [Goldilocks; FP5_LIMBS],
    msg_digest: [Goldilocks; FP5_LIMBS],
) -> [Goldilocks; FP5_LIMBS] {
    hash_to_quintic_extension(&challenge_preimage(reconstructed_r, msg_digest))
}

/// Every intermediate value computed while verifying one signature, for use as
/// a reference oracle in tests (compare against the AIR's own trace values).
#[cfg(feature = "reference")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchnorrVerificationWitness {
    pub msg_digest: [Goldilocks; FP5_LIMBS],
    pub signature_s_bytes: [u8; 40],
    pub signature_e_bytes: [u8; 40],
    pub public_key: [Goldilocks; FP5_LIMBS],
    /// `R = s*G + e*P`, reconstructed from the signature and public key.
    pub reconstructed_r: [Goldilocks; FP5_LIMBS],
    pub challenge_preimage: [Goldilocks; SCHNORR_CHALLENGE_PREIMAGE_ELEMENTS],
    /// `H(reconstructed_r, msg_digest)` before reduction to a scalar.
    pub challenge_fp5: [Goldilocks; FP5_LIMBS],
    /// [`Self::challenge_fp5`] reduced to a scalar and serialized, for comparison
    /// against the signature's own `e`.
    pub challenge_scalar_bytes: [u8; 40],
    /// `true` iff [`Self::challenge_scalar_bytes`] equals the signature's `e`.
    pub is_valid: bool,
}

/// Verifies one [`crate::types::SchnorrInstance`] using the `goldilocks-crypto`
/// reference implementation (group operations, scalar field) rather than this
/// crate's AIR-oriented affine arithmetic, and returns every intermediate value.
///
/// This is the independent oracle used to validate the AIR's verification logic
/// in tests; it is gated behind the `reference` feature so it never ships in the
/// production proving/verifying path.
#[cfg(feature = "reference")]
pub fn verification_witness(
    instance: &crate::types::SchnorrInstance,
) -> crate::Result<SchnorrVerificationWitness> {
    use goldilocks_crypto::{Point, ScalarField};

    let msg_digest = instance.msg_digest.to_goldilocks_limbs("msg_digest")?;
    let public_key = instance.public_key.to_goldilocks_limbs("public_key")?;
    let (s_bytes, e_bytes) = instance.signature.split();

    let s = scalar_from_bytes(s_bytes.as_bytes())?;
    let e = scalar_from_bytes(e_bytes.as_bytes())?;

    let public_key_fp5 = crypto_fp5_from_limbs(&public_key);
    let public_point =
        Point::decode(&public_key_fp5).ok_or(goldilocks_crypto::CryptoError::InvalidPublicKey)?;

    let r_point = Point::generator().mul(&s).add(&public_point.mul(&e));
    let reconstructed_r = limbs_from_crypto_fp5(&r_point.encode());
    let challenge_preimage = challenge_preimage(reconstructed_r, msg_digest);
    let challenge_fp5 = hash_to_quintic_extension(&challenge_preimage);
    let challenge_scalar = ScalarField::from_fp5_element(&crypto_fp5_from_limbs(&challenge_fp5));
    let challenge_scalar_bytes = challenge_scalar.to_bytes_le();

    Ok(SchnorrVerificationWitness {
        msg_digest,
        signature_s_bytes: *s_bytes.as_bytes(),
        signature_e_bytes: *e_bytes.as_bytes(),
        public_key,
        reconstructed_r,
        challenge_preimage,
        challenge_fp5,
        challenge_scalar_bytes,
        is_valid: e.equals(&challenge_scalar),
    })
}

#[cfg(feature = "reference")]
fn scalar_from_bytes(bytes: &[u8; 40]) -> crate::Result<goldilocks_crypto::ScalarField> {
    goldilocks_crypto::ScalarField::from_bytes_le(bytes).map_err(crate::Error::ReferenceEncoding)
}

/// Converts this crate's `Fp5` limb representation to `goldilocks-crypto`'s
/// `Fp5Element`, so reference group operations can be applied to values produced
/// by [`crate::affine`].
#[cfg(feature = "reference")]
pub(crate) fn crypto_fp5_from_limbs(
    limbs: &[Goldilocks; FP5_LIMBS],
) -> goldilocks_crypto::Fp5Element {
    use p3_field::PrimeField64;

    goldilocks_crypto::Fp5Element(
        limbs.map(|x| goldilocks_crypto::Goldilocks::from_canonical_u64(x.as_canonical_u64())),
    )
}

/// The inverse of [`crypto_fp5_from_limbs`]: converts a `goldilocks-crypto`
/// `Fp5Element` back to this crate's limb representation.
#[cfg(feature = "reference")]
pub(crate) fn limbs_from_crypto_fp5(
    value: &goldilocks_crypto::Fp5Element,
) -> [Goldilocks; FP5_LIMBS] {
    use p3_field::PrimeCharacteristicRing;

    value.0.map(|x| Goldilocks::from_u64(x.to_canonical_u64()))
}
