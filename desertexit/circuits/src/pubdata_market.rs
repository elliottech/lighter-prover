// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::bigint::big_u16::{BigIntU16Target, CircuitBuilderBigIntU16};
use circuit::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use circuit::bigint::biguint::CircuitBuilderBiguint;
use circuit::poseidon2::Poseidon2Hash;
use circuit::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, BIGU16_U64_LIMBS, Builder};
use circuit::types::constants::POSITION_LIST_SIZE;
use circuit::uint::u16::gadgets::arithmetic_u16::CircuitBuilderU16;
use num::BigInt;
use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::Target;
use serde::Deserialize;

use crate::pubdata_account::PubdataAccountPositionTarget;

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct PubdataMarketDetails {
    #[serde(rename = "f", default)]
    #[serde(deserialize_with = "circuit::deserializers::int_to_bigint")]
    pub funding_rate_prefix_sum: BigInt, // 63 bits
    #[serde(rename = "mp", default)]
    pub mark_price: u32, // 32 bits
    #[serde(rename = "qm", default)]
    pub quote_multiplier: u32, // 20 bits
}

#[derive(Debug, Clone, Default)]
pub struct PubdataMarketDetailsTarget {
    pub funding_rate_prefix_sum: BigIntU16Target, // 63 bits
    pub mark_price: Target,                       // 32 bits
    pub quote_multiplier: Target,                 // 20 bits
}

impl PubdataMarketDetailsTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            funding_rate_prefix_sum: builder.add_virtual_bigint_u16_target_safe(BIGU16_U64_LIMBS),
            mark_price: builder.add_virtual_target(),
            quote_multiplier: builder.add_virtual_target(),
        }
    }

    pub fn get_position_base_notional_value(
        &self,
        builder: &mut Builder,
        position: &BigIntU16Target,
    ) -> BigIntTarget {
        let multiplier = builder.mul(self.quote_multiplier, self.mark_price);
        let multiplier_big = builder.target_to_biguint(multiplier);
        let position = builder.bigint_u16_to_bigint(position);
        builder.mul_bigint_with_biguint_non_carry(&position, &multiplier_big, BIG_U64_LIMBS)
    }

    pub fn get_funding_delta_for_position_and_market(
        &self,
        builder: &mut Builder,
        position: &PubdataAccountPositionTarget,
    ) -> BigIntTarget {
        let quote_multiplier_big = builder.target_to_biguint(self.quote_multiplier);

        let position_big_u32 = builder.bigint_u16_to_bigint(&position.position);
        let funding_multiplier = builder.mul_bigint_with_biguint_non_carry(
            &position_big_u32,
            &quote_multiplier_big,
            BIG_U96_LIMBS,
        );
        let funding_rate = builder.sub_bigint_u16_non_carry(
            &position.last_funding_rate_prefix_sum,
            &self.funding_rate_prefix_sum,
            BIGU16_U64_LIMBS,
        );
        let funding_rate = builder.bigint_u16_to_bigint(&funding_rate);

        BigIntTarget {
            abs: builder.mul_biguint_non_carry(
                &funding_multiplier.abs,
                &funding_rate.abs,
                BIG_U96_LIMBS,
            ),
            sign: SignTarget::new_unsafe(
                builder.mul(funding_multiplier.sign.target, funding_rate.sign.target),
            ),
        }
    }
}

pub fn all_public_market_details_hash(
    builder: &mut Builder,
    all_market_details: &[PubdataMarketDetailsTarget; POSITION_LIST_SIZE],
) -> HashOutTarget {
    let mut elements = vec![];
    for market_details in all_market_details.iter() {
        let mut limbs = market_details.funding_rate_prefix_sum.abs.limbs.clone();
        limbs.resize(BIGU16_U64_LIMBS, builder.zero_u16());
        for limb in limbs {
            elements.push(limb.0);
        }
        elements.extend_from_slice(&[
            market_details.funding_rate_prefix_sum.sign.target,
            market_details.mark_price,
            market_details.quote_multiplier,
        ]);
    }
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
}
