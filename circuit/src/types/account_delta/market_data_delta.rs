// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigInt;
use num::bigint::Sign;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::{BigIntU16Target, CircuitBuilderBigIntU16, WitnessBigInt16};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::{BIGU16_U64_LIMBS, Builder};
use crate::types::constants::{MARKET_TYPE_BINARY_OPTIONS, MARKET_TYPE_PERPS};

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct MarketDataDelta {
    #[serde(rename = "lfrps")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub funding_rate_prefix_sum_delta: BigInt, // value is in range [-2^62 + 1, 2^62 - 1], thus the diff is in range [-2^63 + 2, 2^63 - 2]

    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub size_delta: BigInt, // value is in range [-2^56 + 1, 2^56 - 1], thus the diff is in range [-2^57 + 2, 2^57 - 2]
}

#[derive(Debug, Clone, Default)]
pub struct MarketDataDeltaTarget {
    pub funding_rate_prefix_sum_delta: BigIntU16Target,
    pub size_delta: BigIntU16Target,
}

impl MarketDataDeltaTarget {
    pub fn new(builder: &mut Builder) -> Self {
        MarketDataDeltaTarget {
            funding_rate_prefix_sum_delta: builder
                .add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
            size_delta: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        MarketDataDeltaTarget {
            funding_rate_prefix_sum_delta: builder.zero_bigint_u16(),
            size_delta: builder.zero_bigint_u16(),
        }
    }

    pub fn select_market_data_delta(
        builder: &mut Builder,
        flag: BoolTarget,
        a: &Self,
        b: &Self,
    ) -> Self {
        Self {
            funding_rate_prefix_sum_delta: builder.select_bigint_u16(
                flag,
                &a.funding_rate_prefix_sum_delta,
                &b.funding_rate_prefix_sum_delta,
            ),
            size_delta: builder.select_bigint_u16(flag, &a.size_delta, &b.size_delta),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_bigint_u16(
            &self.funding_rate_prefix_sum_delta,
            &format!("{} funding_rate_prefix_sum_delta", tag),
        );
        builder.println_bigint_u16(&self.size_delta, &format!("{} size_delta", tag));
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let is_funding_rate_prefix_sum_delta_zero =
            builder.is_zero_bigint_u16(&self.funding_rate_prefix_sum_delta);
        let is_size_delta_zero = builder.is_zero_bigint_u16(&self.size_delta);
        builder.and(is_funding_rate_prefix_sum_delta_zero, is_size_delta_zero)
    }

    /// Leaf hash `[market_type, frps_delta, size_delta]`, same layout for every market type.
    /// Binary options carry a zero funding rate prefix sum delta.
    pub fn hash(&self, builder: &mut Builder, market_type: Target) -> HashOutTarget {
        let mut elements = vec![market_type, self.funding_rate_prefix_sum_delta.sign.target];
        for limb in self.funding_rate_prefix_sum_delta.abs.limbs.iter() {
            elements.push(limb.0);
        }
        elements.push(self.size_delta.sign.target);
        for limb in self.size_delta.abs.limbs.iter() {
            elements.push(limb.0);
        }
        let nonzero_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let zero_hash = builder.zero_hash_out();

        let is_empty = self.is_empty(builder);
        builder.select_hash(is_empty, &zero_hash, &nonzero_hash)
    }

    pub fn hash_perps(&self, builder: &mut Builder) -> HashOutTarget {
        let market_type = builder.constant_u64(MARKET_TYPE_PERPS);
        self.hash(builder, market_type)
    }
}

/// Size delta of one binary options market slot, `size_delta == 0` marks an untouched slot.
#[derive(Debug, Clone, Default)]
pub struct BinaryOptionsDelta {
    pub size_delta: BigInt,
}

impl BinaryOptionsDelta {
    pub fn is_empty(&self) -> bool {
        self.size_delta.sign() == Sign::NoSign
    }
}

#[derive(Debug, Clone, Default)]
pub struct BinaryOptionsDeltaTarget {
    pub size_delta: BigIntU16Target,
}

impl BinaryOptionsDeltaTarget {
    pub fn new(builder: &mut Builder) -> Self {
        BinaryOptionsDeltaTarget {
            // safe because the leaf hash is pinned to the tree root produced by the tx circuits
            size_delta: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS),
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        builder.is_zero_bigint_u16(&self.size_delta)
    }

    /// Same leaf as the tx circuits write for a binary options market: zero funding rate prefix
    /// sum delta and the binary options market type.
    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let zero = builder.zero();
        let market_data_delta = MarketDataDeltaTarget {
            funding_rate_prefix_sum_delta: BigIntU16Target::from_vec(&[zero; BIGU16_U64_LIMBS + 1]),
            size_delta: self.size_delta.clone(),
        };
        let market_type = builder.constant_u64(MARKET_TYPE_BINARY_OPTIONS);
        market_data_delta.hash(builder, market_type)
    }
}

pub trait BinaryOptionsDeltaTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_binary_options_delta_target(
        &mut self,
        a: &BinaryOptionsDeltaTarget,
        b: &BinaryOptionsDelta,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    BinaryOptionsDeltaTargetWitness<F> for T
{
    fn set_binary_options_delta_target(
        &mut self,
        a: &BinaryOptionsDeltaTarget,
        b: &BinaryOptionsDelta,
    ) -> Result<()> {
        self.set_bigint_u16_target(&a.size_delta, &b.size_delta)?;

        Ok(())
    }
}

pub trait MarketDataDeltaTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_market_data_delta_target(
        &mut self,
        a: &MarketDataDeltaTarget,
        b: &MarketDataDelta,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    MarketDataDeltaTargetWitness<F> for T
{
    fn set_market_data_delta_target(
        &mut self,
        a: &MarketDataDeltaTarget,
        b: &MarketDataDelta,
    ) -> Result<()> {
        self.set_bigint_u16_target(
            &a.funding_rate_prefix_sum_delta,
            &b.funding_rate_prefix_sum_delta,
        )?;
        self.set_bigint_u16_target(&a.size_delta, &b.size_delta)?;

        Ok(())
    }
}
