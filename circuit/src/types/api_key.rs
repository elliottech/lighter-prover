// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use plonky2::plonk::config::Hasher;
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
}

impl<F: RichField + Extendable<5> + Default> Default for ApiKey<F> {
    fn default() -> Self {
        ApiKey {
            api_key_index: 0,
            public_key: QuinticExtension::<F>::ZERO,
            nonce: 0,
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
        }
    }

    pub fn is_empty(&self) -> bool {
        self.public_key.is_zero()
    }

    pub fn hash(&self) -> HashOut<F> {
        if self.is_empty() {
            return HashOut::ZERO;
        }

        Poseidon2Hash::hash_no_pad(&[
            self.public_key.0[0],
            self.public_key.0[1],
            self.public_key.0[2],
            self.public_key.0[3],
            self.public_key.0[4],
            F::from_canonical_i64(self.nonce),
        ])
    }
}

#[derive(Debug, Clone, Deserialize, Default, serde::Serialize)]
pub struct ApiKeyTarget {
    pub api_key_index: Target,

    pub public_key: QuinticExtensionTarget,
    pub nonce: Target, // 48 bits
}

impl ApiKeyTarget {
    pub fn new(builder: &mut Builder) -> Self {
        ApiKeyTarget {
            api_key_index: builder.add_virtual_target(),

            public_key: builder.add_virtual_quintic_ext_target(),
            nonce: builder.add_virtual_target(),
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let zero_quintic = builder.zero_quintic_ext();

        let assertions = [
            builder.is_equal_quintic_ext(self.public_key, zero_quintic),
            builder.is_zero(self.nonce),
        ];

        builder.multi_and(&assertions)
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.api_key_index, &format!("{} api_key_index", tag));
        builder.println_arr(&self.public_key.0, &format!("{} public_key", tag));
        builder.println(self.nonce, &format!("{} nonce", tag));
    }

    pub fn empty(builder: &mut Builder, api_key_index: Target) -> Self {
        ApiKeyTarget {
            api_key_index,

            public_key: builder.zero_quintic_ext(),
            nonce: builder.zero(),
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            self.public_key.0[0],
            self.public_key.0[1],
            self.public_key.0[2],
            self.public_key.0[3],
            self.public_key.0[4],
            self.nonce,
        ]);

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
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use rand::Rng;

    use super::*;
    use crate::eddsa::schnorr::{ONE_SK, schnorr_pk_from_sk};
    use crate::types::config::{C, CIRCUIT_CONFIG};

    #[test]
    fn api_key_hash_check() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let api_key_target = ApiKeyTarget::new(&mut builder);
        let api_key_hash_target = api_key_target.hash(&mut builder);

        let data = builder.build::<C>();

        // Set the values
        let mut pw = PartialWitness::new();

        let api_key = ApiKey {
            api_key_index: rand::thread_rng().r#gen(),
            public_key: schnorr_pk_from_sk(&ONE_SK),
            nonce: rand::thread_rng().r#gen::<u32>() as i64,
        };
        let api_key_hash = api_key.hash();

        pw.set_api_key_target(&api_key_target, &api_key)?;
        pw.set_hash_target(api_key_hash_target, api_key_hash)?;

        data.verify(data.prove(pw)?)
    }
}
