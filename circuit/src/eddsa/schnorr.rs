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

#[derive(Debug, Clone, Deserialize)]
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
}
