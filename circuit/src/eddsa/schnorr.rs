// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use anyhow::Result;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::{Field, PrimeField, PrimeField64, Sample};
use plonky2::hash::hashing::hash_n_to_m_no_pad;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
#[cfg(test)]
use plonky2::plonk::circuit_data::CircuitConfig;
use rand::thread_rng;
use serde::Deserialize;

use super::gadgets::curve::CircuitBuilderEcGFp5;
use crate::bigint::biguint::WitnessBigUint;
use crate::eddsa::curve::curve::ECgFp5Point;
use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::poseidon2::{Poseidon2Hash, Poseidon2Permutation};
use crate::types::config::{Builder, F};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SchnorrSig {
    pub s: ECgFp5Scalar,
    pub e: ECgFp5Scalar,
}

impl SchnorrSig {
    pub const ZERO: Self = Self {
        s: ECgFp5Scalar::ZERO,
        e: ECgFp5Scalar::ZERO,
    };
}

impl Default for SchnorrSig {
    fn default() -> Self {
        Self::ZERO
    }
}

pub const ONE_SK: ECgFp5Scalar = ECgFp5Scalar::ONE;

/// The fixed message used for unsigned tx slots (L1/internal txs, empty padding).
/// Must stay in sync with `dummy_schnorr_instance` — both sides of the circuit boundary
/// (in-circuit constants in `tx_constraints.rs` and off-circuit batch padding in examples)
/// hash this same byte string to produce the dummy tx_hash.
pub const DUMMY_INSTANCE_MESSAGE: &[u8] = b"p3-schnorr dummy signature: unsigned tx slot";

/// Computes the canonical fixed `(pk, tx_hash, sig)` triple for unsigned tx slots.
/// Called at circuit build time by `dummy_schnorr_instance_constants` (in-circuit constants)
/// and at prove time by `prove_signature_batch_proof` (off-circuit batch padding).
/// Both must use this function so they produce bit-identical instances.
///
/// Uses a deterministic nonce (`k`) derived from `(sk, tx_hash)` via Poseidon2 so the
/// output is stable across calls — necessary because the in-circuit constants are baked
/// once at circuit-build time and the off-circuit batch padding is computed at prove time.
pub fn dummy_schnorr_instance() -> (QuinticExtension<F>, QuinticExtension<F>, SchnorrSig) {
    let sk = ONE_SK;
    let pk = schnorr_pk_from_sk(&sk);
    let message: Vec<F> = DUMMY_INSTANCE_MESSAGE
        .iter()
        .map(|b| F::from_canonical_u8(*b))
        .collect();
    let tx_hash = hash_to_quintic_extension(&message);
    let sig = schnorr_sign_hashed_message_deterministic(&tx_hash, &sk);
    (pk, tx_hash, sig)
}

/// A fixed, publicly-known `(account_pk, tx_hash, signature)` triple — a genuinely valid
/// Schnorr signature under the fixed dummy keypair (`ONE_SK`) over a fixed message — baked
/// into the circuit as constants at build time (computed once per circuit build, not per
/// witness). Used by [`crate::types::tx_type::TxTarget::define`] to substitute the exposed
/// signature data for any tx slot that has no real signature to check (L1/internal txs, L2
/// change-pubkey-to-empty, or an empty/padding tx slot within a `BlockTxCircuit` proof).
pub fn dummy_schnorr_instance_constants(
    builder: &mut Builder,
) -> (
    QuinticExtensionTarget,
    QuinticExtensionTarget,
    SchnorrSigTarget,
) {
    let (pk, tx_hash, sig) = dummy_schnorr_instance();
    (
        builder.constant_quintic_ext(pk),
        builder.constant_quintic_ext(tx_hash),
        SchnorrSigTarget {
            s: builder.constant_nonnative(sig.s),
            e: builder.constant_nonnative(sig.e),
        },
    )
}

/// Signs a pre-hashed message with a deterministic nonce derived from `(sk, m_hashed)`
/// via Poseidon2. Unlike [`schnorr_sign_hashed_message`] this is stable across calls,
/// which is required wherever the signature must be reproduced exactly (e.g. dummy
/// instances baked as in-circuit constants).
fn schnorr_sign_hashed_message_deterministic(
    m_hashed: &QuinticExtension<F>,
    sk: &ECgFp5Scalar,
) -> SchnorrSig {
    // Derive a deterministic nonce: k = scalar_from_hash(sk_limbs || m_hashed_limbs).
    // Using Poseidon2 (the same hash already in scope) keeps this self-contained.
    let sk_limbs: Vec<F> = sk
        .to_canonical_biguint()
        .to_u32_digits()
        .iter()
        .map(|&d| F::from_canonical_u32(d))
        .collect();
    let mut nonce_preimage = sk_limbs;
    nonce_preimage.extend_from_slice(&m_hashed.0);
    let nonce_hash = hash_n_to_m_no_pad::<F, Poseidon2Permutation<F>>(&nonce_preimage, 8);
    // Pack the 8 field elements into a BigUint (each fits in u64) and reduce to a scalar.
    let mut k_digits: Vec<u32> = nonce_hash
        .iter()
        .flat_map(|f| {
            let v = f.to_canonical_u64();
            [v as u32, (v >> 32) as u32]
        })
        .collect();
    k_digits.resize(16, 0); // pad to > scalar field size to ensure uniform reduction
    let k = ECgFp5Scalar::from_noncanonical_biguint(num::BigUint::new(k_digits));

    let r = ECgFp5Point::GENERATOR * k;
    let mut preimage = r.encode().0.to_vec();
    preimage.extend_from_slice(&m_hashed.0);
    let e = ECgFp5Scalar::from_gfp5(hash_to_quintic_extension(&preimage));

    SchnorrSig {
        s: k - e * (*sk),
        e,
    }
}

pub fn schnorr_generate_random_sk() -> ECgFp5Scalar {
    ECgFp5Scalar::sample(&mut thread_rng())
}

// Public key is actually an EC point (4 Fp5 elements), but the curve is designed
// in a way that allows points to be encoded as a single Fp5 element.
pub fn schnorr_pk_from_sk(sk: &ECgFp5Scalar) -> QuinticExtension<F> {
    (ECgFp5Point::GENERATOR * sk).encode()
}

/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

// Converts given u8 array into field elements
// Returns (s, e) = (k + e * sk, e)
pub fn schnorr_sign_u8_array(message_bytes: &[u8], sk: &ECgFp5Scalar) -> SchnorrSig {
    schnorr_sign_fe_array(
        &message_bytes
            .iter()
            .map(|b| F::from_canonical_u8(*b))
            .collect::<Vec<_>>(),
        sk,
    )
}

// Hashes given field elements
// Returns (s, e) = (k + e * sk, e)
pub fn schnorr_sign_fe_array(message_bytes_as_fe: &[F], sk: &ECgFp5Scalar) -> SchnorrSig {
    schnorr_sign_hashed_message(
        &hash_to_quintic_extension(message_bytes_as_fe), // Compute H(m)
        sk,
    )
}

// Output is 5 Field elements, which is also wrapped by QuinticExtension / QuinticExtensionTarget
// note: this doesn't apply any padding, so this is vulnerable to length extension attacks
pub fn hash_to_quintic_extension(m: &[F]) -> QuinticExtension<F> {
    QuinticExtension::<F>(
        hash_n_to_m_no_pad::<F, Poseidon2Permutation<F>>(m, 5)
            .try_into()
            .unwrap(),
    )
}

pub fn schnorr_sign_hashed_message(
    m_hashed: &QuinticExtension<F>,
    sk: &ECgFp5Scalar,
) -> SchnorrSig {
    // Sample random scalar `k` and compute `r = k * G`
    let k = ECgFp5Scalar::sample(&mut thread_rng());
    let r = ECgFp5Point::GENERATOR * k;

    // Compute `e = H(r || H(m))`, which is a scalar point
    let mut preimage = r.encode().0.to_vec();
    preimage.extend_from_slice(&m_hashed.0);
    let e = ECgFp5Scalar::from_gfp5(hash_to_quintic_extension(&preimage));

    SchnorrSig {
        s: k - e * (*sk),
        e,
    }
}

/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct SchnorrSigTarget {
    pub s: NonNativeTarget<ECgFp5Scalar>,
    pub e: NonNativeTarget<ECgFp5Scalar>,
}

impl SchnorrSigTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            s: builder.add_virtual_nonnative_target(),
            e: builder.add_virtual_nonnative_target(),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            s: builder.zero_nonnative(),
            e: builder.zero_nonnative(),
        }
    }
}

pub trait SchnorrSigTargetWitness<F: PrimeField64> {
    fn set_schnorr_sig_target(&mut self, t: &SchnorrSigTarget, value: &SchnorrSig) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> SchnorrSigTargetWitness<F> for T {
    fn set_schnorr_sig_target(&mut self, t: &SchnorrSigTarget, value: &SchnorrSig) -> Result<()> {
        self.set_biguint_target(&t.s.value, &value.s.to_canonical_biguint())?;
        self.set_biguint_target(&t.e.value, &value.e.to_canonical_biguint())?;

        Ok(())
    }
}

pub fn verify_schnorr_signature_conditional_circuit(
    builder: &mut Builder,
    flag: BoolTarget,
    pk_target: &QuinticExtensionTarget,
    hashed_message_target: &QuinticExtensionTarget, // H(m)
    sig_target: &SchnorrSigTarget,
) {
    // Decode Fp5 into an EC point (ECgFp5PointTarget consisting of 4 Fp5 elements)
    let pk_target = builder.conditional_ecgfp5_point_decode(flag, *pk_target);

    // r_v = s*G + e*pk
    let r_v = builder.ecgfp5_muladd_2_gen_opt(pk_target, &sig_target.s, &sig_target.e);

    // e_v = H(R || H(m))
    let mut preimage = builder.ecgfp5_point_encode(r_v).0.to_vec();
    preimage.extend(&hashed_message_target.0);
    let e_v_ext = hash_to_quintic_extension_circuit(builder, &preimage);
    let e_v = builder.encode_quintic_ext_as_scalar(e_v_ext);

    // check e_v == e
    builder.conditional_connect_nonnative(flag, &sig_target.e, &e_v);
}

// In-circuit version of sig_hash
pub fn hash_to_quintic_extension_circuit(
    builder: &mut Builder,
    m: &[Target],
) -> QuinticExtensionTarget {
    QuinticExtensionTarget(
        builder
            .hash_n_to_m_no_pad::<Poseidon2Hash>(m.to_vec(), 5)
            .try_into()
            .unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::eddsa::gadgets::base_field::PartialWitnessQuinticExt;
    use crate::types::config::C;

    fn build_optimized_schnorr_gate_count() -> usize {
        let mut builder = Builder::new(CircuitConfig::standard_ecc_config());
        let flag = builder._true();

        let hashed_message_target = builder.add_virtual_quintic_ext_target();
        let sig_target = SchnorrSigTarget::new(&mut builder);
        let pk_target = builder.add_virtual_quintic_ext_target();

        verify_schnorr_signature_conditional_circuit(
            &mut builder,
            flag,
            &pk_target,
            &hashed_message_target,
            &sig_target,
        );

        builder.num_gates()
    }

    #[test]
    fn test_schnorr_signature_gates_after_generator_window_opt() -> Result<()> {
        let optimized_gates = build_optimized_schnorr_gate_count();

        println!(
            "Schnorr gate count (optimized 2-bit joint table): {}",
            optimized_gates
        );

        assert!(optimized_gates > 0, "optimized gates should be non-zero");

        Ok(())
    }

    fn build_and_prove_schnorr_circuit(sk: &ECgFp5Scalar, msg: &[F]) -> Result<()> {
        let message_hash = hash_to_quintic_extension(msg);
        let sig = schnorr_sign_hashed_message(&message_hash, sk);
        let pk = schnorr_pk_from_sk(sk);

        let mut builder = Builder::new(CircuitConfig::standard_ecc_config());
        let flag = builder._true();
        let hashed_message_target = builder.add_virtual_quintic_ext_target();
        let sig_target = SchnorrSigTarget::new(&mut builder);
        let pk_target = builder.add_virtual_quintic_ext_target();

        verify_schnorr_signature_conditional_circuit(
            &mut builder,
            flag,
            &pk_target,
            &hashed_message_target,
            &sig_target,
        );

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(hashed_message_target, message_hash)?;
        pw.set_schnorr_sig_target(&sig_target, &sig)?;
        pw.set_quintic_ext_target(pk_target, pk)?;

        let proof = data.prove(pw)?;
        data.verify(proof)?;

        Ok(())
    }

    #[test]
    fn test_schnorr_signature_proof() -> Result<()> {
        let message_bytes = b"gate-count test payload";
        let msg = message_bytes
            .iter()
            .map(|b| F::from_canonical_u8(*b))
            .collect::<Vec<_>>();
        let sk = schnorr_generate_random_sk();

        build_and_prove_schnorr_circuit(&sk, &msg)?;

        Ok(())
    }

    /// Confirms (without depending on the `p3-schnorr` crate) that this repo's native
    /// `QuinticExtension<F>`/`ECgFp5Scalar` representations of `(pk, tx_hash, signature)`
    /// already carry the exact field-element sequence `p3-schnorr`'s
    /// `SchnorrInstance::to_goldilocks_fields` expects from wire bytes — i.e. exposing these
    /// targets as Plonky2 public inputs and re-packing them off-circuit into
    /// `Fp5Bytes`/`SignatureBytes` is a pure relabeling, not a conversion.
    ///
    /// `p3_*` helpers below reimplement (deliberately, not by importing the crate) the exact
    /// byte-layout logic from `p3-schnorr/src/types.rs`: `Fp5Bytes::to_goldilocks_limbs`,
    /// `SignatureBytes::to_u32_field_elements`, and `SchnorrInstance::to_goldilocks_fields`'s
    /// `msg_digest || signature || public_key` row order.
    #[test]
    fn test_p3_schnorr_encoding_equivalence() -> Result<()> {
        use plonky2::field::goldilocks_field::GoldilocksField;

        // p3-schnorr's Fp5Bytes::to_goldilocks_limbs: 5 little-endian u64 limbs.
        fn p3_fp5_to_goldilocks_limbs(x: QuinticExtension<F>) -> [GoldilocksField; 5] {
            x.0
        }

        // p3-schnorr's SignatureBytes::to_u32_field_elements applied to one scalar's 40
        // wire bytes (little-endian u64 limbs repacked as little-endian u32 chunks).
        fn p3_scalar_to_u32_field_elements(s: ECgFp5Scalar) -> [GoldilocksField; 10] {
            let limbs: [u64; 5] = s
                .to_canonical_biguint()
                .to_u64_digits()
                .iter()
                .copied()
                .chain(std::iter::repeat(0u64))
                .take(5)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let mut bytes = [0u8; 40];
            for (i, limb) in limbs.iter().enumerate() {
                bytes[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
            }
            std::array::from_fn(|i| {
                let chunk = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                GoldilocksField::from_canonical_u32(chunk)
            })
        }

        // SchnorrInstance::to_goldilocks_fields row layout: msg_digest(5) || s(10) || e(10) || pk(5).
        fn p3_instance_row(
            msg_digest: QuinticExtension<F>,
            sig: &SchnorrSig,
            pk: QuinticExtension<F>,
        ) -> Vec<GoldilocksField> {
            let mut row = Vec::with_capacity(30);
            row.extend_from_slice(&p3_fp5_to_goldilocks_limbs(msg_digest));
            row.extend_from_slice(&p3_scalar_to_u32_field_elements(sig.s));
            row.extend_from_slice(&p3_scalar_to_u32_field_elements(sig.e));
            row.extend_from_slice(&p3_fp5_to_goldilocks_limbs(pk));
            row
        }

        // The tx circuit's native targets, as they would be exposed as public inputs:
        // QuinticExtensionTarget (5 raw field elements) for pk/tx_hash, and the
        // NonNativeTarget<ECgFp5Scalar>'s 10 U32Target limbs for each of s/e — read back
        // natively here (no circuit needed) since the claim under test is about the
        // *witness-level* field-element sequence, which is what ends up in public inputs.
        fn tx_circuit_native_row(
            msg_digest: QuinticExtension<F>,
            sig: &SchnorrSig,
            pk: QuinticExtension<F>,
        ) -> Vec<GoldilocksField> {
            fn scalar_u32_limbs(s: ECgFp5Scalar) -> [GoldilocksField; 10] {
                let mut digits = s.to_canonical_biguint().to_u32_digits();
                digits.resize(10, 0);
                std::array::from_fn(|i| GoldilocksField::from_canonical_u32(digits[i]))
            }

            let mut row = Vec::with_capacity(30);
            row.extend_from_slice(&msg_digest.0);
            row.extend_from_slice(&scalar_u32_limbs(sig.s));
            row.extend_from_slice(&scalar_u32_limbs(sig.e));
            row.extend_from_slice(&pk.0);
            row
        }

        let sk = schnorr_generate_random_sk();
        let pk = schnorr_pk_from_sk(&sk);
        let msg_bytes = b"encoding equivalence test payload";
        let msg = msg_bytes
            .iter()
            .map(|b| F::from_canonical_u8(*b))
            .collect::<Vec<_>>();
        let msg_digest = hash_to_quintic_extension(&msg);
        let sig = schnorr_sign_hashed_message(&msg_digest, &sk);

        let expected = p3_instance_row(msg_digest, &sig, pk);
        let actual = tx_circuit_native_row(msg_digest, &sig, pk);

        assert_eq!(
            actual, expected,
            "tx-circuit native field-element sequence must match p3-schnorr's wire encoding"
        );

        Ok(())
    }

    #[test]
    fn test_dummy_schnorr_instance_is_deterministic() {
        // In-circuit constants are baked once at circuit-build time; off-circuit batch padding
        // is computed at prove time. Both call dummy_schnorr_instance(), so it must return the
        // same (pk, tx_hash, sig) on every call — any divergence causes the BlockCircuit
        // digest check to fail nondeterministically.
        let (pk1, tx_hash1, sig1) = dummy_schnorr_instance();
        let (pk2, tx_hash2, sig2) = dummy_schnorr_instance();
        assert_eq!(pk1, pk2);
        assert_eq!(tx_hash1, tx_hash2);
        assert_eq!(
            sig1.s.to_canonical_biguint(),
            sig2.s.to_canonical_biguint(),
            "s component must be stable across calls"
        );
        assert_eq!(
            sig1.e.to_canonical_biguint(),
            sig2.e.to_canonical_biguint(),
            "e component must be stable across calls"
        );
    }
}
