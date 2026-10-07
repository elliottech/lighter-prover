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
use crate::ecdsa::gadgets::curve_fixed_base::{
    conditional_fixed_base_curve_mul_circuit, fixed_base_curve_mul_circuit,
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
    let ECDSASignatureTarget { r, s } = sig;

    builder.curve_assert_valid(&pk.0);

    let c = builder.inv_nonnative(s, None);
    let u1 = builder.mul_nonnative(msg, &c);
    let u2 = builder.mul_nonnative(r, &c);

    let point1 = fixed_base_curve_mul_circuit(builder, Secp256K1::GENERATOR_AFFINE, &u1);
    let point2 = builder.glv_mul(&pk.0, &u2);
    let point = builder.curve_add(&point1, &point2);

    builder.connect_biguint(&r.value, &point.x.value);
}

pub fn conditional_verify_ecdsa_sig<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    flag: BoolTarget,
    msg: &NonNativeTarget<Secp256K1Scalar>,
    sig: &ECDSASignatureTarget<Secp256K1>,
    pk: &ECDSAPublicKeyTarget<Secp256K1>,
) {
    let ECDSASignatureTarget { r, s } = sig;

    builder.conditional_curve_assert_valid(flag, &pk.0);

    let c = builder.conditional_inv_nonnative(flag, s, None);
    let u1 = builder.mul_nonnative(msg, &c);
    let u2 = builder.mul_nonnative(r, &c);

    let point1 =
        conditional_fixed_base_curve_mul_circuit(builder, flag, Secp256K1::GENERATOR_AFFINE, &u1);
    let point2 = builder.conditional_glv_mul(flag, &pk.0, &u2);
    let point = builder.conditional_curve_add(flag, &point1, &point2);

    // Verify that point.x === r mod p(scalar) and because point.x is in the base field one subtraction is enough
    // because 2 * Secp256K1Scalar::order() > Secp256K1Base::order()
    let cond_1 = builder.is_equal_biguint(&r.value, &point.x.value);

    let scalar_modulus = builder.constant_biguint(&Secp256K1Scalar::order());
    let (reduced_point_x, borrow) = builder.try_sub_biguint(&point.x.value, &scalar_modulus);
    let mut cond_2 = builder.is_equal_biguint(&r.value, &reduced_point_x);
    cond_2 = builder.and_not(cond_2, BoolTarget::new_unsafe(borrow.0));

    let cond = builder.or(cond_1, cond_2);

    builder.conditional_assert_true(flag, cond);
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Sample;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;

    use super::*;
    use crate::ecdsa::curve::curve_types::CurveScalar;
    use crate::ecdsa::curve::ecdsa::{
        ECDSAPublicKey, ECDSASecretKey, ECDSASignature, sign_message,
    };

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

        let _true = builder._true();
        verify_ecdsa_sig(&mut builder, &msg_target, &sig_target, &pk_target);
        conditional_verify_ecdsa_sig(&mut builder, _true, &msg_target, &sig_target, &pk_target);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }

    #[test]
    fn test_ecdsa_circuit() -> Result<()> {
        test_ecdsa_circuit_with_config(CircuitConfig::standard_ecc_config())
    }
}
