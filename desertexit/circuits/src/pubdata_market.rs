// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::bigint::big_u16::{BigIntU16Target, CircuitBuilderBigIntU16};
use circuit::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use circuit::bigint::biguint::CircuitBuilderBiguint;
use circuit::poseidon2::Poseidon2Hash;
use circuit::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, BIGU16_U64_LIMBS, Builder};
use circuit::types::constants::{POSITION_HASH_BUCKET_SIZE, POSITION_LIST_SIZE};
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
    let mut market_details_ext = all_market_details.to_vec();
    market_details_ext.push(PubdataMarketDetailsTarget {
        funding_rate_prefix_sum: builder.zero_bigint_u16(),
        mark_price: builder.zero(),
        quote_multiplier: builder.zero(),
    });

    let mut bucket_hash_elements = vec![];
    for bucket in market_details_ext.chunks(POSITION_HASH_BUCKET_SIZE) {
        let mut elements = vec![];
        for market_details in bucket.iter() {
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
        let bucket_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        bucket_hash_elements.extend_from_slice(&bucket_hash.elements);
    }
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(bucket_hash_elements)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use circuit::bigint::big_u16::{BigIntU16Target, CircuitBuilderBiguint16};
    use circuit::bigint::bigint::{CircuitBuilderBigInt, SignTarget};
    use circuit::builder::Builder;
    use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
    use circuit::types::market_details::MarketRiskDetailsTarget;
    use num::bigint::Sign;
    use num::{BigInt, Signed};
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use rand::{Rng, thread_rng};

    use crate::pubdata_account::PubdataAccountPositionTarget;
    use crate::pubdata_market::PubdataMarketDetailsTarget;

    fn constant_bigint_u16(builder: &mut Builder<F, D>, value: &BigInt) -> BigIntU16Target {
        BigIntU16Target {
            abs: builder.constant_biguint_u16(&value.abs().to_biguint().unwrap()),
            sign: SignTarget::new_unsafe(match value.sign() {
                Sign::Plus => builder.one(),
                Sign::Minus => builder.neg_one(),
                Sign::NoSign => builder.zero(),
            }),
        }
    }

    #[test]
    fn test_hash_equivalence() -> Result<()> {
        let mut rng = thread_rng();

        let mut builder = Builder::<F, D>::new(CIRCUIT_CONFIG);

        let mut markets = vec![];
        let mut pubdata_markets = vec![];
        for _ in 0..255 {
            let funding_rate_prefix_sum = constant_bigint_u16(
                &mut builder,
                &num::BigInt::from(rng.r#gen::<u64>() & ((1u64 << 63) - 1)),
            );
            let mark_price = builder.constant(F::from_canonical_u64(rng.r#gen::<u32>() as u64));
            let qm20: u32 = rng.r#gen::<u32>() & ((1u32 << 20) - 1);
            let quote_multiplier = builder.constant(F::from_canonical_u64(qm20 as u64));
            markets.push(MarketRiskDetailsTarget {
                funding_rate_prefix_sum: funding_rate_prefix_sum.clone(),
                mark_price,
                quote_multiplier,
                ..MarketRiskDetailsTarget::default()
            });
            pubdata_markets.push(PubdataMarketDetailsTarget {
                funding_rate_prefix_sum: funding_rate_prefix_sum.clone(),
                mark_price,
                quote_multiplier,
            });
        }

        let (_, hash1, _) = circuit::types::market_details::all_market_details_hashes(
            &mut builder,
            &markets.try_into().unwrap(),
        );
        let hash2 = crate::pubdata_market::all_public_market_details_hash(
            &mut builder,
            &pubdata_markets.try_into().unwrap(),
        );
        builder.connect_hashes(hash1, hash2);

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::new()).unwrap())
    }

    #[test]
    fn test_get_funding_delta_for_position_and_market_equivalence() -> Result<()> {
        let mut rng = thread_rng();

        let mut builder = Builder::<F, D>::new(CIRCUIT_CONFIG);

        for _ in 0..100 {
            let funding_rate_prefix_sum = constant_bigint_u16(
                &mut builder,
                &num::BigInt::from(rng.r#gen::<u64>() & ((1u64 << 63) - 1)),
            );
            let mark_price = builder.constant(F::from_canonical_u64(rng.r#gen::<u32>() as u64));
            let quote_multiplier = builder.constant(F::from_canonical_u64(
                (rng.r#gen::<u32>() & ((1u32 << 20) - 1)) as u64,
            ));
            let market = MarketRiskDetailsTarget {
                funding_rate_prefix_sum: funding_rate_prefix_sum.clone(),
                mark_price,
                quote_multiplier,
                ..MarketRiskDetailsTarget::default()
            };
            let pubdata_market = PubdataMarketDetailsTarget {
                funding_rate_prefix_sum: funding_rate_prefix_sum.clone(),
                mark_price,
                quote_multiplier,
            };

            let position = constant_bigint_u16(
                &mut builder,
                &if rng.r#gen::<bool>() {
                    num::BigInt::from(rng.gen_range(0u128..=(1u128 << 56)))
                } else {
                    -num::BigInt::from(rng.gen_range(0u128..=(1u128 << 56)))
                },
            );

            let pos = PubdataAccountPositionTarget {
                position: position.clone(),
                last_funding_rate_prefix_sum: funding_rate_prefix_sum.clone(),
            };

            let pnl1 = circuit::liquidation::get_funding_delta_for_position_and_market(
                &mut builder,
                &circuit::types::account_position::AccountPositionTarget {
                    position: position.clone(),
                    last_funding_rate_prefix_sum: funding_rate_prefix_sum.clone(),
                    ..circuit::types::account_position::AccountPositionTarget::default()
                },
                &market,
            );
            let pnl2 = pubdata_market.get_funding_delta_for_position_and_market(&mut builder, &pos);
            builder.connect_bigint(&pnl1, &pnl2);
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::new()).unwrap())
    }
}
