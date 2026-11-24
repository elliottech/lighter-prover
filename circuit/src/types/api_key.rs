// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::{Builder, F};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::{
    CircuitBuilderGFp5, PartialWitnessQuinticExt, QuinticExtensionTarget,
};
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::ecdsa_pub_key::{
    ECDSA_PUB_KEY_FIELD_ELEMENTS, EcdsaPubKey, EcdsaPubKeyTarget, EcdsaPubKeyTargetWitness,
};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
#[serde(default)]
pub struct ApiKey<F>
where
    F: RichField + Extendable<5>,
{
    #[serde(rename = "aki")]
    pub api_key_index: u8,

    #[serde(rename = "pk")]
    #[serde(deserialize_with = "deserializers::pub_key")]
    pub public_key: QuinticExtension<F>, // ECgFp5Point encoded into 1 extension field element

    #[serde(rename = "n")]
    pub nonce: i64,

    /// `API_KEY_TYPE_SCHNORR` (0) or `API_KEY_TYPE_ECDSA` (1); selects which
    /// signature digest chain transactions signed by this key are absorbed into.
    #[serde(rename = "kt")]
    pub key_type: u8,

    /// Compressed secp256k1 key, only set when `key_type == API_KEY_TYPE_ECDSA`.
    #[serde(rename = "epk")]
    #[serde(deserialize_with = "deserializers::deserialize_ecdsa_pk_compressed")]
    pub ecdsa_public_key: EcdsaPubKey,
}

pub const API_KEY_TYPE_SCHNORR: u8 = 0;
pub const API_KEY_TYPE_ECDSA: u8 = 1;

impl<F: RichField + Extendable<5> + Default> Default for ApiKey<F> {
    fn default() -> Self {
        ApiKey {
            api_key_index: 0,
            public_key: QuinticExtension::<F>::ZERO,
            nonce: 0,
            key_type: API_KEY_TYPE_SCHNORR,
            ecdsa_public_key: EcdsaPubKey::zero(),
        }
    }
}

impl ApiKey<F>
where
    F: RichField + Extendable<5>,
{
    pub fn empty(api_key_index: u8) -> Self {
        ApiKey {
            api_key_index,

            public_key: QuinticExtension::<F>::ZERO,
            nonce: 0,
            key_type: API_KEY_TYPE_SCHNORR,
            ecdsa_public_key: EcdsaPubKey::zero(),
        }
    }

    pub fn is_schnorr(&self) -> bool {
        self.key_type == API_KEY_TYPE_SCHNORR
    }

    pub fn is_ecdsa(&self) -> bool {
        self.key_type == API_KEY_TYPE_ECDSA
    }
}

#[derive(Debug, Clone, Deserialize, Default, serde::Serialize)]
pub struct ApiKeyTarget {
    pub api_key_index: Target,

    pub public_key: QuinticExtensionTarget,
    pub nonce: Target,    // 48 bits
    pub key_type: Target, // 0 = schnorr, 1 = ecdsa
    pub ecdsa_public_key: EcdsaPubKeyTarget,
}

impl ApiKeyTarget {
    pub fn new(builder: &mut Builder) -> Self {
        ApiKeyTarget {
            api_key_index: builder.add_virtual_target(),

            public_key: builder.add_virtual_quintic_ext_target(),
            nonce: builder.add_virtual_target(),
            key_type: builder.add_virtual_target(),
            ecdsa_public_key: EcdsaPubKeyTarget::new(builder),
        }
    }

    pub fn is_ecdsa(&self) -> BoolTarget {
        BoolTarget::new_unsafe(self.key_type)
    }

    pub fn is_schnorr(&self, builder: &mut Builder) -> BoolTarget {
        builder.not(self.is_ecdsa())
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let zero_quintic = builder.zero_quintic_ext();

        let assertions = [
            builder.is_equal_quintic_ext(self.public_key, zero_quintic),
            builder.is_zero(self.nonce),
            builder.is_zero(self.key_type),
            self.ecdsa_public_key.is_zero(builder),
        ];

        builder.multi_and(&assertions)
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.api_key_index, &format!("{} api_key_index", tag));
        builder.println_arr(&self.public_key.0, &format!("{} public_key", tag));
        builder.println(self.nonce, &format!("{} nonce", tag));
        builder.println(self.key_type, &format!("{} key_type", tag));
        builder.println_arr(
            &self.ecdsa_public_key.to_targets(),
            &format!("{} ecdsa_public_key", tag),
        );
    }

    pub fn empty(builder: &mut Builder, api_key_index: Target) -> Self {
        ApiKeyTarget {
            api_key_index,

            public_key: builder.zero_quintic_ext(),
            nonce: builder.zero(),
            key_type: builder.zero(),
            ecdsa_public_key: EcdsaPubKeyTarget::zero(builder),
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let mut elements = Vec::with_capacity(7 + ECDSA_PUB_KEY_FIELD_ELEMENTS);
        elements.extend_from_slice(&self.public_key.0);
        elements.push(self.nonce);
        elements.push(self.key_type);
        elements.extend(self.ecdsa_public_key.to_targets());
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let empty_hash = builder.zero_hash_out();
        let is_empty = self.is_empty(builder);

        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }
}

pub trait ApiKeyTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_api_key_target(&mut self, a: &ApiKeyTarget, b: &ApiKey<F>) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    ApiKeyTargetWitness<F> for T
{
    fn set_api_key_target(&mut self, a: &ApiKeyTarget, b: &ApiKey<F>) -> Result<()> {
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_quintic_ext_target(a.public_key, b.public_key)?;
        self.set_target(a.nonce, F::from_canonical_i64(b.nonce))?;
        self.set_target(a.key_type, F::from_canonical_u8(b.key_type))?;
        self.set_ecdsa_pub_key_target(&a.ecdsa_public_key, &b.ecdsa_public_key)?;

        Ok(())
    }
}

pub fn select_api_key_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &ApiKeyTarget,
    b: &ApiKeyTarget,
) -> ApiKeyTarget {
    ApiKeyTarget {
        api_key_index: builder.select(flag, a.api_key_index, b.api_key_index),
        public_key: builder.select_quintic_ext(flag, a.public_key, b.public_key),
        nonce: builder.select(flag, a.nonce, b.nonce),
        key_type: builder.select(flag, a.key_type, b.key_type),
        ecdsa_public_key: EcdsaPubKeyTarget::select(
            builder,
            flag,
            &a.ecdsa_public_key,
            &b.ecdsa_public_key,
        ),
    }
}
