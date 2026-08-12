//! Deterministic test fixtures built on the `goldilocks-crypto` reference
//! implementation, gated behind the `reference` feature.
//!
//! These produce real, valid signatures (not arbitrary bytes) so that AIR and
//! proof tests exercise the actual verification equation rather than only its
//! failure paths.

#[cfg(feature = "reference")]
use crate::Result;
use crate::types::SchnorrInstance;
#[cfg(feature = "reference")]
use crate::types::{Fp5Bytes, SignatureBytes};

/// Deterministic test fixture generated with `goldilocks-crypto`, the same
/// Schnorr/ECgFp5 implementation used by the Lighter reference path.
#[cfg(feature = "reference")]
pub fn deterministic_instance(index: u8) -> Result<SchnorrInstance> {
    use goldilocks_crypto::{Point, ScalarField, sign_with_nonce, verify_signature};

    let mut sk_bytes = [0u8; crate::types::FP5_BYTES];
    sk_bytes[0] = index.max(1);

    let mut nonce_bytes = [0u8; crate::types::FP5_BYTES];
    nonce_bytes[0] = index.wrapping_mul(17).max(1);
    nonce_bytes[1] = index.wrapping_mul(29);

    let mut msg_digest = [0u8; crate::types::FP5_BYTES];
    msg_digest[0] = index;
    msg_digest[8] = index.wrapping_mul(3);
    msg_digest[16] = index.wrapping_mul(5);

    let sk = ScalarField::from_bytes_le(&sk_bytes)
        .map_err(|_| goldilocks_crypto::CryptoError::InvalidPrivateKeyLength(sk_bytes.len()))?;
    let public_key = Point::generator().mul(&sk).encode().to_bytes_le();
    let signature = sign_with_nonce(&sk_bytes, &msg_digest, &nonce_bytes)?;
    debug_assert!(verify_signature(&signature, &msg_digest, &public_key)?);

    let signature: [u8; crate::types::SIGNATURE_BYTES] = signature
        .try_into()
        .expect("goldilocks-crypto Schnorr signatures are 80 bytes");

    Ok(SchnorrInstance::new(
        Fp5Bytes::from_array(msg_digest),
        SignatureBytes::from_array(signature),
        Fp5Bytes::from_array(public_key),
    ))
}

/// A batch of `count` valid signatures, each from [`deterministic_instance`] with
/// a distinct 1-based index.
#[cfg(feature = "reference")]
pub fn deterministic_batch(count: usize) -> Result<crate::types::Batch> {
    let instances = (0..count)
        .map(|i| deterministic_instance((i + 1) as u8))
        .collect::<Result<Vec<_>>>()?;
    crate::types::Batch::new(instances)
}

/// Stub present so callers don't need `#[cfg(feature = "reference")]` at every
/// call site; always panics, since fixtures require the `goldilocks-crypto`
/// reference implementation.
#[cfg(not(feature = "reference"))]
pub fn deterministic_instance(_index: u8) -> crate::Result<SchnorrInstance> {
    unimplemented!("enable the `reference` feature to build deterministic fixtures")
}
