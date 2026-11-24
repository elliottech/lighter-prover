// Portions of this file are derived from plonky2-ecdsa
// Copyright (c) 2022 The Plonky2 Authors
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::secp256k1_base::Secp256K1Base;
use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
use plonky2::field::types::{Field, PrimeField, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::BoolTarget;
use plonky2::iop::witness::Witness;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::ecdsa::curve::curve_types::Curve;
use crate::ecdsa::curve::ecdsa::{ECDSAPublicKey, ECDSASignature};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::ecdsa::gadgets::curve::{AffinePointTarget, CircuitBuilderCurve};
use crate::ecdsa::gadgets::curve_complete::{
    CircuitBuilderCurveComplete, fixed_base_curve_mul_complete,
};
use crate::ecdsa::gadgets::glv::CircuitBuilderGlv;
use crate::keccak::keccak::CircuitBuilderKeccak;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::uint::u8::U8Target;
#[derive(Clone, Debug)]
pub struct ECDSASecretKeyTarget<C: Curve>(pub NonNativeTarget<C::ScalarField>);

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct ECDSAPublicKeyTarget<C: Curve>(pub AffinePointTarget<C>);

pub trait CircuitBuilderECDSAPublicKey<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_ecdsa_public_key(&mut self) -> ECDSAPublicKeyTarget<Secp256K1>;

    fn select_ecdsa_public_key(
        &mut self,
        flag: BoolTarget,
        a: &ECDSAPublicKeyTarget<Secp256K1>,
        b: &ECDSAPublicKeyTarget<Secp256K1>,
    ) -> ECDSAPublicKeyTarget<Secp256K1>;

    fn get_l1_address_from_ecdsa_public_key(
        &mut self,
        pk: &ECDSAPublicKeyTarget<Secp256K1>,
    ) -> BigUintTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderECDSAPublicKey<F, D>
    for Builder<F, D>
{
    fn add_virtual_ecdsa_public_key(&mut self) -> ECDSAPublicKeyTarget<Secp256K1> {
        ECDSAPublicKeyTarget(self.add_virtual_affine_point_target())
    }

    fn select_ecdsa_public_key(
        &mut self,
        flag: BoolTarget,
        a: &ECDSAPublicKeyTarget<Secp256K1>,
        b: &ECDSAPublicKeyTarget<Secp256K1>,
    ) -> ECDSAPublicKeyTarget<Secp256K1> {
        ECDSAPublicKeyTarget(AffinePointTarget {
            x: NonNativeTarget::<Secp256K1Base> {
                value: self.select_biguint(flag, &a.0.x.value, &b.0.x.value),
                _phantom: PhantomData,
            },
            y: NonNativeTarget::<Secp256K1Base> {
                value: self.select_biguint(flag, &a.0.y.value, &b.0.y.value),
                _phantom: PhantomData,
            },
        })
    }

    fn get_l1_address_from_ecdsa_public_key(
        &mut self,
        pk: &ECDSAPublicKeyTarget<Secp256K1>,
    ) -> BigUintTarget {
        let mut pk_x_bytes = self.split_nonnative_to_bytes(&pk.0.x);
        pk_x_bytes.reverse();
        let mut pk_y_bytes = self.split_nonnative_to_bytes(&pk.0.y);
        pk_y_bytes.reverse();

        let l1_address_bytes: Vec<U8Target> = self
            .keccak256_circuit(
                pk_x_bytes
                    .iter()
                    .chain(pk_y_bytes.iter())
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .iter()
            .rev()
            .take(20)
            .copied()
            .collect();
        self.le_sum_bytes_biguint(&l1_address_bytes)
    }
}

pub trait ECDSAPublicKeyTargetWitness<F: PrimeField64> {
    fn set_ecdsa_public_key_target(
        &mut self,
        t: &ECDSAPublicKeyTarget<Secp256K1>,
        value: &ECDSAPublicKey<Secp256K1>,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> ECDSAPublicKeyTargetWitness<F> for T {
    fn set_ecdsa_public_key_target(
        &mut self,
        t: &ECDSAPublicKeyTarget<Secp256K1>,
        value: &ECDSAPublicKey<Secp256K1>,
    ) -> Result<()> {
        self.set_biguint_target(&t.0.x.value, &value.0.x.to_canonical_biguint())?;
        self.set_biguint_target(&t.0.y.value, &value.0.y.to_canonical_biguint())?;

        Ok(())
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct ECDSASignatureTarget<C: Curve> {
    pub r: NonNativeTarget<C::ScalarField>,
    pub s: NonNativeTarget<C::ScalarField>,
}

pub trait CircuitBuilderECDSASignature<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_ecdsa_target(&mut self) -> ECDSASignatureTarget<Secp256K1>;

    fn select_ecdsa_signature(
        &mut self,
        flag: BoolTarget,
        a: &ECDSASignatureTarget<Secp256K1>,
        b: &ECDSASignatureTarget<Secp256K1>,
    ) -> ECDSASignatureTarget<Secp256K1>;

    fn is_zero_ecdsa_signature(&mut self, sig: &ECDSASignatureTarget<Secp256K1>) -> BoolTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderECDSASignature<F, D>
    for Builder<F, D>
{
    fn add_virtual_ecdsa_target(&mut self) -> ECDSASignatureTarget<Secp256K1> {
        ECDSASignatureTarget {
            r: self.add_virtual_nonnative_target(),
            s: self.add_virtual_nonnative_target(),
        }
    }

    fn select_ecdsa_signature(
        &mut self,
        flag: BoolTarget,
        a: &ECDSASignatureTarget<Secp256K1>,
        b: &ECDSASignatureTarget<Secp256K1>,
    ) -> ECDSASignatureTarget<Secp256K1> {
        ECDSASignatureTarget {
            r: NonNativeTarget::<Secp256K1Scalar> {
                value: self.select_biguint(flag, &a.r.value, &b.r.value),
                _phantom: core::marker::PhantomData,
            },
            s: NonNativeTarget::<Secp256K1Scalar> {
                value: self.select_biguint(flag, &a.s.value, &b.s.value),
                _phantom: core::marker::PhantomData,
            },
        }
    }

    fn is_zero_ecdsa_signature(&mut self, sig: &ECDSASignatureTarget<Secp256K1>) -> BoolTarget {
        let is_r_zero = self.is_zero_biguint(&sig.r.value);
        let is_s_zero = self.is_zero_biguint(&sig.s.value);
        self.and(is_r_zero, is_s_zero)
    }
}

pub trait ECDSASignatureTargetWitness<F: PrimeField64> {
    fn set_ecdsa_signature_target(
        &mut self,
        t: &ECDSASignatureTarget<Secp256K1>,
        value: &ECDSASignature<Secp256K1>,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> ECDSASignatureTargetWitness<F> for T {
    fn set_ecdsa_signature_target(
        &mut self,
        t: &ECDSASignatureTarget<Secp256K1>,
        value: &ECDSASignature<Secp256K1>,
    ) -> Result<()> {
        self.set_biguint_target(&t.r.value, &value.r.to_canonical_biguint())?;
        self.set_biguint_target(&t.s.value, &value.s.to_canonical_biguint())?;

        Ok(())
    }
}

pub fn verify_ecdsa_sig<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    msg: &NonNativeTarget<Secp256K1Scalar>,
    sig: &ECDSASignatureTarget<Secp256K1>,
    pk: &ECDSAPublicKeyTarget<Secp256K1>,
) {
    let _true = builder._true();
    conditional_verify_ecdsa_sig(builder, _true, msg, sig, pk);
}

/// Verifies the signature when `flag` is set. Uses complete curve arithmetic, so every valid
/// signature is provable and the constraints are satisfiable for any input when `flag` is not set.
pub fn conditional_verify_ecdsa_sig<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    flag: BoolTarget,
    msg: &NonNativeTarget<Secp256K1Scalar>,
    sig: &ECDSASignatureTarget<Secp256K1>,
    pk: &ECDSAPublicKeyTarget<Secp256K1>,
) {
    let ECDSASignatureTarget { r, s } = sig;

    builder.conditional_curve_assert_valid(flag, &pk.0);
    // The curve gadgets need a valid curve point, also when the verification is disabled.
    let generator = builder.constant_affine_point(Secp256K1::GENERATOR_AFFINE);
    let pk_point = AffinePointTarget {
        x: builder.select_nonnative(flag, &pk.0.x, &generator.x),
        y: builder.select_nonnative(flag, &pk.0.y, &generator.y),
    };

    let c = builder.conditional_inv_nonnative(flag, s, None);
    let u1 = builder.mul_nonnative(msg, &c);
    let u2 = builder.mul_nonnative(r, &c);

    let point1 = fixed_base_curve_mul_complete(builder, Secp256K1::GENERATOR_AFFINE, &u1);
    let point2 = builder.glv_mul_complete(&pk_point, &u2);
    let point = builder.curve_add_complete(&point1, &point2);
    let point_x = &point.point.x.value;

    // Verify that point.x === r mod p(scalar) and because point.x is in the base field one subtraction is enough
    // because 2 * Secp256K1Scalar::order() > Secp256K1Base::order()
    let cond_1 = builder.is_equal_biguint(&r.value, point_x);

    let scalar_modulus = builder.constant_biguint(&Secp256K1Scalar::order());
    let (reduced_point_x, borrow) = builder.try_sub_biguint(point_x, &scalar_modulus);
    let mut cond_2 = builder.is_equal_biguint(&r.value, &reduced_point_x);
    cond_2 = builder.and_not(cond_2, BoolTarget::new_unsafe(borrow.0));

    let mut cond = builder.or(cond_1, cond_2);
    // The point at infinity has no x coordinate.
    cond = builder.and_not(cond, point.is_inf);
    // `r` must be non-zero in the scalar field. It fits in 256 bits, so it is zero iff it is `0` or the modulus.
    let is_r_zero = builder.is_zero_biguint(&r.value);
    let is_r_modulus = builder.is_equal_biguint(&r.value, &scalar_modulus);
    let is_r_invalid = builder.or(is_r_zero, is_r_modulus);
    cond = builder.and_not(cond, is_r_invalid);

    builder.conditional_assert_true(flag, cond);
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use num::BigUint;
    use plonky2::field::types::Sample;
    use plonky2::hash::keccak::KeccakHash;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericHashOut, Hasher, PoseidonGoldilocksConfig};

    use super::*;
    use crate::ecdsa::curve::curve_types::{CurveScalar, base_to_scalar};
    use crate::ecdsa::curve::ecdsa::{
        ECDSAPublicKey, ECDSASecretKey, ECDSASignature, sign_message, verify_message,
    };
    use crate::ecdsa::gadgets::curve_fixed_base::fixed_base_curve_mul_circuit;

    fn test_ecdsa_circuit_with_config(config: CircuitConfig) -> Result<()> {
        type C = PoseidonGoldilocksConfig;
        type Curve = Secp256K1;

        let pw = PartialWitness::new();
        let mut builder = Builder::new(config);

        let msg = Secp256K1Scalar::rand();
        let msg_target = builder.constant_nonnative(msg);

        let sk = ECDSASecretKey::<Curve>(Secp256K1Scalar::rand());
        let pk = ECDSAPublicKey((CurveScalar(sk.0) * Curve::GENERATOR_PROJECTIVE).to_affine());

        let pk_target = ECDSAPublicKeyTarget(builder.constant_affine_point(pk.0));
        let sig = sign_message(msg, sk);

        let ECDSASignature { r, s } = sig;
        let r_target = builder.constant_nonnative(r);
        let s_target = builder.constant_nonnative(s);
        let sig_target = ECDSASignatureTarget {
            r: r_target,
            s: s_target,
        };

        verify_ecdsa_sig(&mut builder, &msg_target, &sig_target, &pk_target);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }

    #[test]
    fn test_ecdsa_circuit() -> Result<()> {
        test_ecdsa_circuit_with_config(CircuitConfig::standard_ecc_config())
    }

    /// The verification as it was before complete arithmetic: the scalar multiplications start
    /// from a constant offset point and every addition needs operands with different x.
    /// Kept to check that the exceptional cases below are the ones it could not prove.
    fn verify_ecdsa_sig_incomplete<F: RichField + Extendable<D>, const D: usize>(
        builder: &mut Builder<F, D>,
        msg: &NonNativeTarget<Secp256K1Scalar>,
        sig: &ECDSASignatureTarget<Secp256K1>,
        pk: &ECDSAPublicKeyTarget<Secp256K1>,
    ) {
        let ECDSASignatureTarget { r, s } = sig;

        builder.curve_assert_valid(&pk.0);

        let c = builder.inv_nonnative(s, None);
        let u1 = builder.mul_nonnative(msg, &c);
        let u2 = builder.mul_nonnative(r, &c);

        let point1 = fixed_base_curve_mul_circuit(builder, Secp256K1::GENERATOR_AFFINE, &u1);
        let point2 = builder.glv_mul(&pk.0, &u2);
        let point = builder.curve_add(&point1, &point2);

        let cond_1 = builder.is_equal_biguint(&r.value, &point.x.value);
        let scalar_modulus = builder.constant_biguint(&Secp256K1Scalar::order());
        let (reduced_point_x, borrow) = builder.try_sub_biguint(&point.x.value, &scalar_modulus);
        let mut cond_2 = builder.is_equal_biguint(&r.value, &reduced_point_x);
        cond_2 = builder.and_not(cond_2, BoolTarget::new_unsafe(borrow.0));
        let cond = builder.or(cond_1, cond_2);
        builder.assert_true(cond);
    }

    /// Valid signatures whose verification adds equal or opposite points, or the point at
    /// infinity, must be provable. Invalid ones must not be.
    #[test]
    fn test_conditional_ecdsa_circuit_exceptional_cases() {
        type C = PoseidonGoldilocksConfig;
        type F = <C as plonky2::plonk::config::GenericConfig<2>>::F;
        type Curve = Secp256K1;

        let mut builder = Builder::<F, 2>::new(CircuitConfig::standard_ecc_config());
        let flag = builder.add_virtual_bool_target_safe();
        let msg_target = builder.add_virtual_nonnative_target();
        let pk_target = builder.add_virtual_ecdsa_public_key();
        let sig_target = builder.add_virtual_ecdsa_target();
        conditional_verify_ecdsa_sig(&mut builder, flag, &msg_target, &sig_target, &pk_target);
        let data = builder.build::<C>();

        let proves = |enabled: bool,
                      msg: Secp256K1Scalar,
                      pk: (BigUint, BigUint),
                      (r, s): (BigUint, BigUint)| {
            let mut pw = PartialWitness::new();
            pw.set_bool_target(flag, enabled).unwrap();
            pw.set_biguint_target(&msg_target.value, &msg.to_canonical_biguint())
                .unwrap();
            pw.set_biguint_target(&pk_target.0.x.value, &pk.0).unwrap();
            pw.set_biguint_target(&pk_target.0.y.value, &pk.1).unwrap();
            pw.set_biguint_target(&sig_target.r.value, &r).unwrap();
            pw.set_biguint_target(&sig_target.s.value, &s).unwrap();
            data.prove(pw).and_then(|proof| data.verify(proof)).is_ok()
        };

        let mut builder = Builder::<F, 2>::new(CircuitConfig::standard_ecc_config());
        let incomplete_msg_target = builder.add_virtual_nonnative_target();
        let incomplete_pk_target = builder.add_virtual_ecdsa_public_key();
        let incomplete_sig_target = builder.add_virtual_ecdsa_target();
        verify_ecdsa_sig_incomplete(
            &mut builder,
            &incomplete_msg_target,
            &incomplete_sig_target,
            &incomplete_pk_target,
        );
        let incomplete_data = builder.build::<C>();

        let incomplete_proves =
            |msg: Secp256K1Scalar, pk: (BigUint, BigUint), (r, s): (BigUint, BigUint)| {
                let mut pw = PartialWitness::new();
                pw.set_biguint_target(&incomplete_msg_target.value, &msg.to_canonical_biguint())
                    .unwrap();
                pw.set_biguint_target(&incomplete_pk_target.0.x.value, &pk.0)
                    .unwrap();
                pw.set_biguint_target(&incomplete_pk_target.0.y.value, &pk.1)
                    .unwrap();
                pw.set_biguint_target(&incomplete_sig_target.r.value, &r)
                    .unwrap();
                pw.set_biguint_target(&incomplete_sig_target.s.value, &s)
                    .unwrap();
                // An unsatisfiable witness is an error or, with debug assertions, a panic.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    incomplete_data
                        .prove(pw)
                        .and_then(|proof| incomplete_data.verify(proof))
                        .is_ok()
                }))
                .unwrap_or(false)
            };

        let pk_of = |sk: Secp256K1Scalar| {
            let pk = (CurveScalar(sk) * Curve::GENERATOR_PROJECTIVE).to_affine();
            (pk.x.to_canonical_biguint(), pk.y.to_canonical_biguint())
        };
        let sig_of = |sig: ECDSASignature<Curve>| {
            (sig.r.to_canonical_biguint(), sig.s.to_canonical_biguint())
        };

        let msg = Secp256K1Scalar::rand();
        let two = Secp256K1Scalar::TWO;
        let three = Secp256K1Scalar::from_canonical_u64(3);
        // Discrete log of the offset point that the incomplete scalar multiplications started from.
        let h = Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_le(
            &GenericHashOut::<F>::to_bytes(&KeccakHash::<32>::hash_no_pad(&[F::ZERO])),
        ));
        // The incomplete MSM builds its table as `offset + i * (+-Q)`, with the sign of the first
        // half of the GLV decomposition of `u2`. For `Q = +-offset` it fails for both signs, for
        // the other small multiples only for the sign that walks through the offset.
        let k1_is_negative = |sig: ECDSASignature<Curve>| {
            crate::ecdsa::curve::glv::decompose_secp256k1_scalar(sig.r * sig.s.inverse()).2
        };
        type IncompleteProves = fn(bool) -> bool;
        let cases: [(Secp256K1Scalar, IncompleteProves); 7] = [
            (Secp256K1Scalar::rand(), |_| true),
            (Secp256K1Scalar::ONE, |_| true),
            (Secp256K1Scalar::NEG_ONE, |_| true),
            (h, |_| false),
            (-h, |_| false),
            // `offset - 2 * Q` is infinity.
            (h * two.inverse(), |k1_neg| !k1_neg),
            // `offset + 3 * Q` is infinity.
            (-h * three.inverse(), |k1_neg| k1_neg),
        ];
        for (sk, incomplete_should_prove) in cases {
            let sig = sign_message(msg, ECDSASecretKey::<Curve>(sk));
            assert!(verify_message(msg, sig, ECDSASecretKey(sk).to_public()));
            assert!(proves(true, msg, pk_of(sk), sig_of(sig)), "sk = {sk:?}");
            assert_eq!(
                incomplete_proves(msg, pk_of(sk), sig_of(sig)),
                incomplete_should_prove(k1_is_negative(sig)),
                "sk = {sk:?}"
            );
        }

        // `u1 * G == u2 * Q`: with nonce `k` and `r = (k * G).x`, take `sk = msg / r`.
        let k = Secp256K1Scalar::rand();
        let r =
            base_to_scalar::<Curve>((CurveScalar(k) * Curve::GENERATOR_PROJECTIVE).to_affine().x);
        let sk = msg * r.inverse();
        let sig = ECDSASignature::<Curve> {
            r,
            s: k.inverse() * (msg + r * sk),
        };
        assert!(verify_message(msg, sig, ECDSASecretKey(sk).to_public()));
        assert!(proves(true, msg, pk_of(sk), sig_of(sig)));
        assert!(!incomplete_proves(msg, pk_of(sk), sig_of(sig)));

        // `u1 == 0`.
        let sk = Secp256K1Scalar::rand();
        let zero = Secp256K1Scalar::ZERO;
        let sig = sign_message(zero, ECDSASecretKey::<Curve>(sk));
        assert!(proves(true, zero, pk_of(sk), sig_of(sig)));
        assert!(!incomplete_proves(zero, pk_of(sk), sig_of(sig)));

        // `u1 * G == -u2 * Q`: the result is the point at infinity, which is never a valid signature.
        let sig = sign_message(msg, ECDSASecretKey::<Curve>(sk));
        let (r, s) = sig_of(sig);
        let opposite_sk = -msg * Secp256K1Scalar::from_noncanonical_biguint(r.clone()).inverse();
        assert!(!verify_message(
            msg,
            sig,
            ECDSASecretKey(opposite_sk).to_public()
        ));
        assert!(!proves(
            true,
            msg,
            pk_of(opposite_sk),
            (r.clone(), s.clone())
        ));

        // Invalid signatures.
        let order = Secp256K1Scalar::order();
        let zero_biguint = BigUint::ZERO;
        assert!(proves(true, msg, pk_of(sk), (r.clone(), s.clone())));
        assert!(!proves(
            true,
            msg + Secp256K1Scalar::ONE,
            pk_of(sk),
            (r.clone(), s.clone())
        ));
        assert!(!proves(
            true,
            msg,
            pk_of(sk + Secp256K1Scalar::ONE),
            (r.clone(), s.clone())
        ));
        assert!(!proves(true, msg, pk_of(sk), (r.clone(), &s + 1u32)));
        assert!(!proves(
            true,
            msg,
            pk_of(sk),
            (r.clone(), zero_biguint.clone())
        ));
        assert!(!proves(
            true,
            msg,
            pk_of(sk),
            (zero_biguint.clone(), s.clone())
        ));
        assert!(!proves(true, msg, pk_of(sk), (order.clone(), s.clone())));
        // `r == 0` with `u1 * G` at infinity.
        assert!(!proves(
            true,
            zero,
            pk_of(sk),
            (zero_biguint.clone(), s.clone())
        ));
        assert!(!proves(true, zero, pk_of(sk), (order, s.clone())));
        // Public key not on the curve.
        let (pk_x, pk_y) = pk_of(sk);
        assert!(!proves(
            true,
            msg,
            (pk_x.clone(), &pk_y + 1u32),
            (r.clone(), s.clone())
        ));

        // Nothing is checked when the verification is disabled.
        let zeros = (zero_biguint.clone(), zero_biguint.clone());
        assert!(proves(false, zero, zeros.clone(), zeros.clone()));
        assert!(proves(false, msg, (pk_x.clone(), zero_biguint), zeros));
        assert!(proves(false, msg, (pk_x, &pk_y + 1u32), (r, &s + 1u32)));
    }
}
