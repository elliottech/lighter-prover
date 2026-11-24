//! Per-transaction signature material exported by the tx circuits and absorbed
//! into the p3-schnorr / p3-ecdsa digest chains.

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::BoolTarget;
use plonky2::iop::witness::Witness;

use super::config::Builder;
use super::ecdsa_pub_key::{ECDSA_PUB_KEY_X_LIMBS, EcdsaPubKeyTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::SchnorrSigTarget;

pub const ECDSA_SIGNATURE_LIMBS: usize = ECDSA_PUB_KEY_X_LIMBS;

/// secp256k1 ECDSA signature `(r, s)`; the witness carries it as `r || s`, each
/// 32 bytes big-endian (the usual 64-byte Ethereum encoding without `v`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EcdsaSignature {
    pub r: BigUint,
    pub s: BigUint,
}

impl EcdsaSignature {
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        EcdsaSignature {
            r: BigUint::from_bytes_be(&bytes[..32]),
            s: BigUint::from_bytes_be(&bytes[32..]),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EcdsaSignatureTarget {
    pub r: BigUintTarget,
    pub s: BigUintTarget,
}

impl EcdsaSignatureTarget {
    pub fn new(builder: &mut Builder) -> Self {
        EcdsaSignatureTarget {
            r: builder.add_virtual_biguint_target_safe(ECDSA_SIGNATURE_LIMBS),
            s: builder.add_virtual_biguint_target_safe(ECDSA_SIGNATURE_LIMBS),
        }
    }
}

pub trait EcdsaSignatureTargetWitness<F: PrimeField64> {
    fn set_l2_ecdsa_signature_target(
        &mut self,
        a: &EcdsaSignatureTarget,
        b: &EcdsaSignature,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> EcdsaSignatureTargetWitness<F> for T {
    fn set_l2_ecdsa_signature_target(
        &mut self,
        a: &EcdsaSignatureTarget,
        b: &EcdsaSignature,
    ) -> Result<()> {
        self.set_biguint_target(&a.r, &b.r)?;
        self.set_biguint_target(&a.s, &b.s)?;
        Ok(())
    }
}

/// What a tx slot contributes to the signature digest chains. Exactly one of
/// `schnorr_check` / `ecdsa_check` can be set, chosen by the signing api key's
/// `key_type` (or by the key installed by a change pubkey transaction).
#[derive(Debug, Clone)]
pub struct SignatureSlotTargets {
    pub tx_hash: QuinticExtensionTarget,

    pub account_pk: QuinticExtensionTarget,
    pub signature: SchnorrSigTarget,
    pub schnorr_check: BoolTarget,

    pub ecdsa_pk: EcdsaPubKeyTarget,
    pub ecdsa_signature: EcdsaSignatureTarget,
    pub ecdsa_check: BoolTarget,
}
