// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::bigint::Sign;
use num::{BigInt, BigUint, Zero};
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::{Deserialize, Serialize};

use super::config::{BIG_U64_LIMBS, BIGU16_U64_LIMBS, Builder};
use super::constants::{MARKET_FLAGS_BITS, POSITION_LIST_SIZE};
use crate::bigint::big_u16::bigint_u16::{
    BigIntU16Target, CircuitBuilderBigIntU16, WitnessBigInt16,
};
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::signed::signed_target::{
    CircuitBuilderSigned, POSITIVE_THRESHOLD_BIT, SignedTarget, WitnessSigned,
};
use crate::types::constants::{
    MARKET_DETAILS_TREE_HEIGHT, POSITION_HASH_BUCKET_COUNT, POSITION_HASH_BUCKET_SIZE,
};
use crate::uint::u16::gadgets::arithmetic_u16::CircuitBuilderU16;
use crate::utils::CircuitBuilderUtils;

pub const MARKET_DETAIL_SIZE: usize = 12;

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct MarketDetails {
    #[serde(rename = "i", default)]
    pub market_index: u16,

    #[serde(rename = "ir", default)]
    pub interest_rate: u32, // 20 bits

    #[serde(rename = "ap", default)]
    pub aggregate_premium_sum: i64,

    // Warning: Witness names impact ask price as bid and vice versa
    #[serde(rename = "ia", default)]
    pub impact_bid_price: u32,

    #[serde(rename = "ib", default)]
    pub impact_ask_price: u32,

    #[serde(rename = "ip", default)]
    pub impact_price: u32,

    #[serde(rename = "o", default)]
    pub open_interest: i64, // 56 bits

    #[serde(rename = "idp", default)]
    pub index_price: u32,

    #[serde(rename = "fcs", default)]
    pub funding_clamp_small: u32,
    #[serde(rename = "fcb", default)]
    pub funding_clamp_big: u32,

    #[serde(rename = "oil", default)]
    pub open_interest_limit: u64,

    #[serde(rename = "mf", default)]
    pub market_flags: i64,

    #[serde(rename = "fpm", default)]
    pub funding_premium_multiplier: u16, // (0, 100] where 100 = 1x
}

impl MarketDetails {
    pub fn from_public_inputs<F>(market_index: u16, pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert_eq!(pis.len(), MARKET_DETAIL_SIZE);

        let aggregate_premium_sum_raw = pis[0].to_canonical_u64();
        let aggregate_premium_sum = if aggregate_premium_sum_raw > 1 << POSITIVE_THRESHOLD_BIT {
            -((F::ORDER - aggregate_premium_sum_raw) as i64)
        } else {
            aggregate_premium_sum_raw as i64
        };

        Self {
            market_index,

            aggregate_premium_sum,
            interest_rate: u32::try_from(pis[1].to_canonical_u64()).unwrap(),
            impact_bid_price: u32::try_from(pis[2].to_canonical_u64()).unwrap(),
            impact_ask_price: u32::try_from(pis[3].to_canonical_u64()).unwrap(),
            impact_price: u32::try_from(pis[4].to_canonical_u64()).unwrap(),
            open_interest: i64::try_from(pis[5].to_canonical_u64()).unwrap(),
            index_price: u32::try_from(pis[6].to_canonical_u64()).unwrap(),
            funding_clamp_small: u32::try_from(pis[7].to_canonical_u64()).unwrap(),
            funding_clamp_big: u32::try_from(pis[8].to_canonical_u64()).unwrap(),
            open_interest_limit: pis[9].to_canonical_u64(),
            market_flags: i64::try_from(pis[10].to_canonical_u64()).unwrap(),
            funding_premium_multiplier: u16::try_from(pis[11].to_canonical_u64()).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MarketDetailsTarget {
    pub aggregate_premium_sum: SignedTarget, // 58 bits + sign
    pub interest_rate: Target,               // 20 bits
    pub impact_bid_price: Target,            // 32 bits
    pub impact_ask_price: Target,            // 32 bits
    pub impact_price: Target,                // 32 bits
    pub open_interest: Target,               // 56 bits
    pub index_price: Target,                 // 32 bits
    pub funding_clamp_small: Target,         // 24 bits
    pub funding_clamp_big: Target,           // 24 bits
    pub open_interest_limit: Target,         // 56 bits
    pub market_flags: Target,
    pub funding_premium_multiplier: Target, // 8 bits, (0, 100] where 100 = 1x
}

impl MarketDetailsTarget {
    fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero(self.aggregate_premium_sum.target),
            builder.is_zero(self.interest_rate),
            builder.is_zero(self.impact_bid_price),
            builder.is_zero(self.impact_ask_price),
            builder.is_zero(self.impact_price),
            builder.is_zero(self.open_interest),
            builder.is_zero(self.index_price),
            builder.is_zero(self.funding_clamp_small),
            builder.is_zero(self.funding_clamp_big),
            builder.is_zero(self.open_interest_limit),
            builder.is_zero(self.market_flags),
            builder.is_zero(self.funding_premium_multiplier),
        ];
        builder.multi_and(&assertions)
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.interest_rate, &format!("{} -- interest_rate", tag));
        builder.println(
            self.aggregate_premium_sum.target,
            &format!("{} -- aggregate_premium_sum", tag),
        );
        builder.println(
            self.impact_bid_price,
            &format!("{} -- impact_bid_price", tag),
        );
        builder.println(
            self.impact_ask_price,
            &format!("{} -- impact_ask_price", tag),
        );
        builder.println(self.impact_price, &format!("{} -- impact_price", tag));
        builder.println(self.open_interest, &format!("{} -- open_interest", tag));
        builder.println(self.index_price, &format!("{} -- index_price", tag));
        builder.println(
            self.funding_clamp_small,
            &format!("{} -- funding_clamp_small", tag),
        );
        builder.println(
            self.funding_clamp_big,
            &format!("{} -- funding_clamp_big", tag),
        );
        builder.println(
            self.open_interest_limit,
            &format!("{} -- open_interest_limit", tag),
        );
        builder.println(self.market_flags, &format!("{} -- market_flags", tag));
        builder.println(
            self.funding_premium_multiplier,
            &format!("{} -- funding_premium_multiplier", tag),
        );
    }

    pub fn new(builder: &mut Builder) -> Self {
        Self {
            interest_rate: builder.add_virtual_target(),
            aggregate_premium_sum: builder.add_virtual_signed_target(),
            impact_bid_price: builder.add_virtual_target(),
            impact_ask_price: builder.add_virtual_target(),
            impact_price: builder.add_virtual_target(),
            open_interest: builder.add_virtual_target(),
            index_price: builder.add_virtual_target(),
            funding_clamp_small: builder.add_virtual_target(),
            funding_clamp_big: builder.add_virtual_target(),
            open_interest_limit: builder.add_virtual_target(),
            market_flags: builder.add_virtual_target(),
            funding_premium_multiplier: builder.add_virtual_target(),
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let empty_hash = builder.zero_hash_out();
        let inputs = vec![
            self.aggregate_premium_sum.target,
            self.interest_rate,
            self.impact_ask_price,
            self.impact_bid_price,
            self.impact_price,
            self.open_interest,
            self.index_price,
            self.funding_clamp_small,
            self.funding_clamp_big,
            self.open_interest_limit,
            self.market_flags,
            self.funding_premium_multiplier,
        ];
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(inputs);

        let is_empty = self.is_empty(builder);
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            interest_rate: builder.zero(),
            aggregate_premium_sum: builder.zero_signed(),
            impact_bid_price: builder.zero(),
            impact_ask_price: builder.zero(),
            impact_price: builder.zero(),
            open_interest: builder.zero(),
            index_price: builder.zero(),
            funding_clamp_small: builder.zero(),
            funding_clamp_big: builder.zero(),
            open_interest_limit: builder.zero(),
            market_flags: builder.zero(),
            funding_premium_multiplier: builder.zero(),
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        let public_inputs_before = builder.num_public_inputs();

        builder.register_public_signed_target(self.aggregate_premium_sum);
        builder.register_public_input(self.interest_rate);
        builder.register_public_input(self.impact_bid_price);
        builder.register_public_input(self.impact_ask_price);
        builder.register_public_input(self.impact_price);
        builder.register_public_input(self.open_interest);
        builder.register_public_input(self.index_price);
        builder.register_public_input(self.funding_clamp_small);
        builder.register_public_input(self.funding_clamp_big);
        builder.register_public_input(self.open_interest_limit);
        builder.register_public_input(self.market_flags);
        builder.register_public_input(self.funding_premium_multiplier);

        let public_inputs_after = builder.num_public_inputs();
        assert_eq!(
            public_inputs_after - public_inputs_before,
            MARKET_DETAIL_SIZE
        );
    }

    pub fn from_public_inputs(pis: Vec<Target>) -> Self {
        assert_eq!(pis.len(), MARKET_DETAIL_SIZE);

        Self {
            aggregate_premium_sum: SignedTarget::new_unsafe(pis[0]),
            interest_rate: pis[1],
            impact_bid_price: pis[2],
            impact_ask_price: pis[3],
            impact_price: pis[4],
            open_interest: pis[5],
            index_price: pis[6],
            funding_clamp_small: pis[7],
            funding_clamp_big: pis[8],
            open_interest_limit: pis[9],
            market_flags: pis[10],
            funding_premium_multiplier: pis[11],
        }
    }
}

pub struct MarketFlags {
    pub margin_mode: Target,
    pub default_margin_mode: Target,
}

impl MarketFlags {
    pub fn from_target(builder: &mut Builder, market_flags: Target) -> Self {
        let le_bits = builder.split_le(market_flags, MARKET_FLAGS_BITS);
        Self {
            margin_mode: le_bits[0].target,
            default_margin_mode: le_bits[1].target,
        }
    }
    pub fn to_target(&self, builder: &mut Builder) -> Target {
        let two = builder.constant_u64(2);
        builder.mul_add(two, self.default_margin_mode, self.margin_mode)
    }
    pub fn is_isolated_only(&self) -> BoolTarget {
        BoolTarget::new_unsafe(self.margin_mode)
    }
}

pub trait MarketDetailsWitness<F: PrimeField64> {
    fn set_market_details_target(
        &mut self,
        t: &MarketDetailsTarget,
        mi: &MarketDetails,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> MarketDetailsWitness<F> for T {
    fn set_market_details_target(
        &mut self,
        t: &MarketDetailsTarget,
        mi: &MarketDetails,
    ) -> Result<()> {
        self.set_signed_target(t.aggregate_premium_sum, mi.aggregate_premium_sum)?;
        self.set_target(t.interest_rate, F::from_canonical_u32(mi.interest_rate))?;

        self.set_target(
            t.impact_bid_price,
            F::from_canonical_u32(mi.impact_bid_price),
        )?;
        self.set_target(
            t.impact_ask_price,
            F::from_canonical_u32(mi.impact_ask_price),
        )?;
        self.set_target(t.impact_price, F::from_canonical_u32(mi.impact_price))?;

        self.set_target(t.open_interest, F::from_canonical_i64(mi.open_interest))?;

        self.set_target(t.index_price, F::from_canonical_u32(mi.index_price))?;
        self.set_target(
            t.funding_clamp_small,
            F::from_canonical_u32(mi.funding_clamp_small),
        )?;
        self.set_target(
            t.funding_clamp_big,
            F::from_canonical_u32(mi.funding_clamp_big),
        )?;
        self.set_target(
            t.open_interest_limit,
            F::from_canonical_u64(mi.open_interest_limit),
        )?;
        self.set_target(t.market_flags, F::from_canonical_i64(mi.market_flags))?;
        self.set_target(
            t.funding_premium_multiplier,
            F::from_canonical_u16(mi.funding_premium_multiplier),
        )?;

        Ok(())
    }
}

pub fn select_market_details(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &MarketDetailsTarget,
    b: &MarketDetailsTarget,
) -> MarketDetailsTarget {
    MarketDetailsTarget {
        interest_rate: builder.select(flag, a.interest_rate, b.interest_rate),
        aggregate_premium_sum: builder.select_signed(
            flag,
            a.aggregate_premium_sum,
            b.aggregate_premium_sum,
        ),
        impact_bid_price: builder.select(flag, a.impact_bid_price, b.impact_bid_price),
        impact_ask_price: builder.select(flag, a.impact_ask_price, b.impact_ask_price),
        impact_price: builder.select(flag, a.impact_price, b.impact_price),
        open_interest: builder.select(flag, a.open_interest, b.open_interest),
        index_price: builder.select(flag, a.index_price, b.index_price),
        funding_clamp_small: builder.select(flag, a.funding_clamp_small, b.funding_clamp_small),
        funding_clamp_big: builder.select(flag, a.funding_clamp_big, b.funding_clamp_big),
        open_interest_limit: builder.select(flag, a.open_interest_limit, b.open_interest_limit),
        market_flags: builder.select(flag, a.market_flags, b.market_flags),
        funding_premium_multiplier: builder.select(
            flag,
            a.funding_premium_multiplier,
            b.funding_premium_multiplier,
        ),
    }
}

/// Calculates difference(or distance) between new and old market details
pub fn diff_market_risk_details(
    builder: &mut Builder,
    new: &MarketRiskDetailsTarget,
    old: &MarketRiskDetailsTarget,
) -> MarketRiskDetailsTarget {
    MarketRiskDetailsTarget {
        default_initial_margin_fraction: builder.sub(
            new.default_initial_margin_fraction,
            old.default_initial_margin_fraction,
        ),
        min_initial_margin_fraction: builder.sub(
            new.min_initial_margin_fraction,
            old.min_initial_margin_fraction,
        ),
        maintenance_margin_fraction: builder.sub(
            new.maintenance_margin_fraction,
            old.maintenance_margin_fraction,
        ),
        close_out_margin_fraction: builder
            .sub(new.close_out_margin_fraction, old.close_out_margin_fraction),
        quote_multiplier: builder.sub(new.quote_multiplier, old.quote_multiplier),
        funding_rate_prefix_sum: builder
            .bigint_u16_vector_diff(&new.funding_rate_prefix_sum, &old.funding_rate_prefix_sum),
        mark_price: builder.sub(new.mark_price, old.mark_price),
        status: builder.sub(new.status, old.status),
        strategy_index: builder.sub(new.strategy_index, old.strategy_index),
    }
}

/// Calculates new market details by applying the difference to an old market details.
/// If difference is calculated from same old market details, result should return
/// new market details from transaction operations
pub fn apply_diff_market_risk_details(
    builder: &mut Builder,
    flag: BoolTarget,
    diff: &MarketRiskDetailsTarget,
    old: &MarketRiskDetailsTarget,
) -> MarketRiskDetailsTarget {
    MarketRiskDetailsTarget {
        default_initial_margin_fraction: builder.mul_add(
            flag.target,
            diff.default_initial_margin_fraction,
            old.default_initial_margin_fraction,
        ),
        min_initial_margin_fraction: builder.mul_add(
            flag.target,
            diff.min_initial_margin_fraction,
            old.min_initial_margin_fraction,
        ),
        maintenance_margin_fraction: builder.mul_add(
            flag.target,
            diff.maintenance_margin_fraction,
            old.maintenance_margin_fraction,
        ),
        close_out_margin_fraction: builder.mul_add(
            flag.target,
            diff.close_out_margin_fraction,
            old.close_out_margin_fraction,
        ),
        quote_multiplier: builder.mul_add(flag.target, diff.quote_multiplier, old.quote_multiplier),
        funding_rate_prefix_sum: builder.bigint_u16_vector_sum(
            flag,
            &diff.funding_rate_prefix_sum,
            &old.funding_rate_prefix_sum,
        ),
        mark_price: builder.mul_add(flag.target, diff.mark_price, old.mark_price),
        status: builder.mul_add(flag.target, diff.status, old.status),
        strategy_index: builder.mul_add(flag.target, diff.strategy_index, old.strategy_index),
    }
}

pub fn get_market_details_tree_root(
    builder: &mut Builder,
    all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
) -> HashOutTarget {
    let mut hashes = vec![];
    for i in (0..POSITION_LIST_SIZE).step_by(2) {
        let left = &all_market_details[i];
        let right = &(if i + 1 < POSITION_LIST_SIZE {
            all_market_details[i + 1].clone()
        } else {
            MarketDetailsTarget::empty(builder)
        });

        let left_hash = left.hash(builder);
        let right_hash = right.hash(builder);

        hashes.push(builder.hash_two_to_one(&left_hash, &right_hash));
    }

    for _ in 1..MARKET_DETAILS_TREE_HEIGHT {
        hashes = hashes
            .chunks(2)
            .map(|pair| builder.hash_two_to_one(&pair[0], &pair[1]))
            .collect();
    }
    assert_eq!(hashes.len(), 1);

    hashes[0]
}

pub fn public_market_details_bucket_hash(
    builder: &mut Builder,
    bucket: &[MarketRiskDetailsTarget],
) -> HashOutTarget {
    let mut public_elements = vec![];
    for market_details in bucket.iter() {
        let mut limbs = market_details.funding_rate_prefix_sum.abs.limbs.clone();
        limbs.resize(BIGU16_U64_LIMBS, builder.zero_u16());
        for limb in limbs {
            public_elements.push(limb.0);
        }
        public_elements.extend_from_slice(&[
            market_details.funding_rate_prefix_sum.sign.target,
            market_details.mark_price,
            market_details.quote_multiplier,
        ]);
    }
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(public_elements)
}

pub fn market_risk_details_bucket_hash(
    builder: &mut Builder,
    bucket: &[MarketRiskDetailsTarget],
) -> [HashOutTarget; 2] {
    let public_market_details_bucket_hash = public_market_details_bucket_hash(builder, bucket);

    let mut elements = public_market_details_bucket_hash.elements.to_vec();
    for market_details in bucket.iter() {
        elements.extend_from_slice(&[
            market_details.status,
            market_details.strategy_index,
            market_details.default_initial_margin_fraction,
            market_details.min_initial_margin_fraction,
            market_details.maintenance_margin_fraction,
            market_details.close_out_margin_fraction,
        ]);
    }

    [
        builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements), // Market risk details bucket hash (contains public)
        public_market_details_bucket_hash, // Public market details bucket hash
    ]
}

pub fn all_market_risk_details_bucket_hashes(
    builder: &mut Builder,
    all_market_details: &[MarketRiskDetailsTarget; POSITION_LIST_SIZE],
) -> [[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2] {
    let mut market_details_ext = all_market_details.to_vec();
    market_details_ext.push(MarketRiskDetailsTarget::empty(builder));
    let bucket_hashes: Vec<[HashOutTarget; 2]> = market_details_ext
        .chunks(POSITION_HASH_BUCKET_SIZE)
        .map(|bucket| market_risk_details_bucket_hash(builder, bucket))
        .collect();
    [
        core::array::from_fn(|i| bucket_hashes[i][0]), // Market risk
        core::array::from_fn(|i| bucket_hashes[i][1]), // Public market
    ]
}

pub fn market_details_hashes_from_bucket_hashes(
    builder: &mut Builder,
    bucket_hashes: &[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2],
) -> (HashOutTarget, HashOutTarget) {
    (
        // market risk
        builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(
            bucket_hashes[0]
                .iter()
                .flat_map(|x| x.elements)
                .collect::<Vec<_>>(),
        ),
        //  public market
        builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(
            bucket_hashes[1]
                .iter()
                .flat_map(|x| x.elements)
                .collect::<Vec<_>>(),
        ),
    )
}

pub fn all_market_details_hashes(
    builder: &mut Builder,
    all_market_details: &[MarketRiskDetailsTarget; POSITION_LIST_SIZE],
) -> (
    HashOutTarget,                                    // market risk
    HashOutTarget,                                    // public market
    [[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2], // market risk, public market
) {
    let bucket_hashes = all_market_risk_details_bucket_hashes(builder, all_market_details);
    let (risk_hash, public_hash) =
        market_details_hashes_from_bucket_hashes(builder, &bucket_hashes);
    (risk_hash, public_hash, bucket_hashes)
}

/// Computes the same public-market-details hash the tx circuit folds into the state root,
/// from the partial (public-only) representation of the market details.
pub fn all_public_market_details_hash(
    builder: &mut Builder,
    all_market_details: &[MarketRiskDetailsTarget; POSITION_LIST_SIZE],
) -> HashOutTarget {
    let mut market_details_ext = all_market_details.to_vec();
    market_details_ext.push(MarketRiskDetailsTarget::empty(builder));
    let bucket_hashes: Vec<HashOutTarget> = market_details_ext
        .chunks(POSITION_HASH_BUCKET_SIZE)
        .map(|bucket| public_market_details_bucket_hash(builder, bucket))
        .collect();
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(
        bucket_hashes
            .iter()
            .flat_map(|x| x.elements)
            .collect::<Vec<_>>(),
    )
}

pub fn all_market_risk_details_hash(
    builder: &mut Builder,
    all_market_details: &[MarketRiskDetailsTarget; POSITION_LIST_SIZE],
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
            market_details.status,
            market_details.strategy_index,
            market_details.default_initial_margin_fraction,
            market_details.min_initial_margin_fraction,
            market_details.maintenance_margin_fraction,
            market_details.close_out_margin_fraction,
        ]);
    }
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct PublicMarketDetails {
    #[serde(rename = "f", default)]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub funding_rate_prefix_sum: BigInt, // 63 bits

    #[serde(rename = "mp", default)]
    pub mark_price: u32,

    #[serde(rename = "qm", default)]
    pub quote_multiplier: u32,
}

impl PublicMarketDetails {
    pub const PUBLIC_INPUTS_SIZE: usize = 5; // u32 funding limbs
    pub const PARTIAL_PUBLIC_INPUTS_SIZE: usize = 7; // u16 funding limbs

    pub fn is_empty(&self) -> bool {
        self.funding_rate_prefix_sum.is_zero() && self.mark_price == 0
    }

    pub fn from_public_inputs<F>(pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert_eq!(pis.len(), PublicMarketDetails::PARTIAL_PUBLIC_INPUTS_SIZE);

        let funding_rate_prefix_sum_sign = if pis[0] == F::ZERO {
            Sign::NoSign
        } else if pis[0] == F::ONE {
            Sign::Plus
        } else if pis[0] == F::NEG_ONE {
            Sign::Minus
        } else {
            panic!(
                "PublicMarketDetails::from_public_inputs() => funding_rate_prefix_sum_sign is not valid"
            );
        };
        let funding_rate_prefix_sum_abs =
            pis[1..5].iter().rev().fold(BigUint::zero(), |acc, limb| {
                (acc << 16) + limb.to_canonical_biguint()
            });
        let funding_rate_prefix_sum =
            BigInt::from_biguint(funding_rate_prefix_sum_sign, funding_rate_prefix_sum_abs);

        Self {
            funding_rate_prefix_sum, // pis[0..5]
            mark_price: u32::try_from(pis[5].to_canonical_u64()).unwrap(),
            quote_multiplier: u32::try_from(pis[6].to_canonical_u64()).unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PublicMarketDetailsTarget {
    pub funding_rate_prefix_sum: BigIntTarget,
    pub mark_price: Target,
    pub quote_multiplier: Target,
}

impl PublicMarketDetailsTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            funding_rate_prefix_sum: builder.add_virtual_bigint_target_unsafe(BIG_U64_LIMBS), // Safe because it is connected to public witness from constrained circuit
            mark_price: builder.add_virtual_target(),
            quote_multiplier: builder.add_virtual_target(),
        }
    }

    pub fn new_public(builder: &mut Builder) -> Self {
        Self {
            funding_rate_prefix_sum: builder.add_virtual_bigint_public_input_unsafe(BIG_U64_LIMBS), // Safe because it is connected to public witness from constrained circuit
            mark_price: builder.add_virtual_public_input(),
            quote_multiplier: builder.add_virtual_public_input(),
        }
    }

    pub fn select(builder: &mut Builder, cond: BoolTarget, a: &Self, b: &Self) -> Self {
        Self {
            funding_rate_prefix_sum: builder.select_bigint(
                cond,
                &a.funding_rate_prefix_sum,
                &b.funding_rate_prefix_sum,
            ),
            mark_price: builder.select(cond, a.mark_price, b.mark_price),
            quote_multiplier: builder.select(cond, a.quote_multiplier, b.quote_multiplier),
        }
    }

    pub fn connect(&self, builder: &mut Builder, b: &Self) {
        builder.connect_bigint(&self.funding_rate_prefix_sum, &b.funding_rate_prefix_sum);
        builder.connect(self.mark_price, b.mark_price);
        builder.connect(self.quote_multiplier, b.quote_multiplier);
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input_bigint(&self.funding_rate_prefix_sum);
        builder.register_public_input(self.mark_price);
        builder.register_public_input(self.quote_multiplier);
    }
}

pub fn connect_public_market_details(
    builder: &mut Builder,
    lhs: &[PublicMarketDetailsTarget; POSITION_LIST_SIZE],
    rhs: &[PublicMarketDetailsTarget; POSITION_LIST_SIZE],
) {
    for i in 0..POSITION_LIST_SIZE {
        lhs[i].connect(builder, &rhs[i]);
    }
}

pub trait PublicMarketDetailsWitness<F: PrimeField64> {
    fn set_public_market_details_target(
        &mut self,
        t: &PublicMarketDetailsTarget,
        mi: &PublicMarketDetails,
    ) -> Result<()>;

    fn set_partial_market_risk_details_target(
        &mut self,
        t: &MarketRiskDetailsTarget,
        mi: &PublicMarketDetails,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> PublicMarketDetailsWitness<F> for T {
    fn set_public_market_details_target(
        &mut self,
        t: &PublicMarketDetailsTarget,
        mi: &PublicMarketDetails,
    ) -> Result<()> {
        self.set_bigint_target(&t.funding_rate_prefix_sum, &mi.funding_rate_prefix_sum)?;
        self.set_target(t.mark_price, F::from_canonical_u32(mi.mark_price))?;
        self.set_target(
            t.quote_multiplier,
            F::from_canonical_u32(mi.quote_multiplier),
        )?;

        Ok(())
    }

    fn set_partial_market_risk_details_target(
        &mut self,
        t: &MarketRiskDetailsTarget,
        mi: &PublicMarketDetails,
    ) -> Result<()> {
        self.set_bigint_u16_target(&t.funding_rate_prefix_sum, &mi.funding_rate_prefix_sum)?;
        self.set_target(t.mark_price, F::from_canonical_u32(mi.mark_price))?;
        self.set_target(
            t.quote_multiplier,
            F::from_canonical_u32(mi.quote_multiplier),
        )?;

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct MarketRiskDetails {
    #[serde(rename = "f", default)]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub funding_rate_prefix_sum: BigInt, // 63 bits
    #[serde(rename = "mp", default)]
    pub mark_price: u32,
    #[serde(rename = "qm", default)]
    pub quote_multiplier: u32,

    #[serde(rename = "s", default)]
    pub status: u8,
    #[serde(rename = "sid", default)]
    pub strategy_index: u8,
    #[serde(rename = "dm", default)]
    pub default_initial_margin_fraction: u16,
    #[serde(rename = "im", default)]
    pub min_initial_margin_fraction: u16,
    #[serde(rename = "mm", default)]
    pub maintenance_margin_fraction: u16,
    #[serde(rename = "cm", default)]
    pub close_out_margin_fraction: u16,
}

#[derive(Debug, Clone, Default)]
pub struct MarketRiskDetailsTarget {
    pub funding_rate_prefix_sum: BigIntU16Target,
    pub mark_price: Target,
    pub quote_multiplier: Target,
    pub status: Target,
    pub strategy_index: Target,
    pub default_initial_margin_fraction: Target,
    pub min_initial_margin_fraction: Target,
    pub maintenance_margin_fraction: Target,
    pub close_out_margin_fraction: Target,
}

impl MarketRiskDetailsTarget {
    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            funding_rate_prefix_sum: builder.zero_bigint_u16(),
            mark_price: builder.zero(),
            quote_multiplier: builder.zero(),
            status: builder.zero(),
            strategy_index: builder.zero(),
            default_initial_margin_fraction: builder.zero(),
            min_initial_margin_fraction: builder.zero(),
            maintenance_margin_fraction: builder.zero(),
            close_out_margin_fraction: builder.zero(),
        }
    }

    pub fn new(builder: &mut Builder) -> Self {
        Self {
            funding_rate_prefix_sum: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS),
            mark_price: builder.add_virtual_target(),
            quote_multiplier: builder.add_virtual_target(),
            status: builder.add_virtual_target(),
            strategy_index: builder.add_virtual_target(),
            default_initial_margin_fraction: builder.add_virtual_target(),
            min_initial_margin_fraction: builder.add_virtual_target(),
            maintenance_margin_fraction: builder.add_virtual_target(),
            close_out_margin_fraction: builder.add_virtual_target(),
        }
    }

    pub fn new_partial(builder: &mut Builder) -> Self {
        Self {
            funding_rate_prefix_sum: builder.add_virtual_bigint_u16_target_unsafe(BIGU16_U64_LIMBS),
            mark_price: builder.add_virtual_target(),
            quote_multiplier: builder.add_virtual_target(),
            ..Default::default()
        }
    }

    pub fn apply_diff_partial(
        builder: &mut Builder,
        flag: BoolTarget,
        diff: &Self,
        old: &Self,
    ) -> Self {
        Self {
            funding_rate_prefix_sum: builder.bigint_u16_vector_sum(
                flag,
                &diff.funding_rate_prefix_sum,
                &old.funding_rate_prefix_sum,
            ),
            mark_price: builder.mul_add(flag.target, diff.mark_price, old.mark_price),
            quote_multiplier: builder.mul_add(
                flag.target,
                diff.quote_multiplier,
                old.quote_multiplier,
            ),
            ..Default::default()
        }
    }

    pub fn connect_partial(&self, builder: &mut Builder, b: &Self) {
        builder.connect_bigint_u16(&self.funding_rate_prefix_sum, &b.funding_rate_prefix_sum);
        builder.connect(self.mark_price, b.mark_price);
        builder.connect(self.quote_multiplier, b.quote_multiplier);
    }

    pub fn partial_register_public_input(&self, builder: &mut Builder) {
        let public_inputs_before = builder.num_public_inputs();

        builder.register_public_input_bigint_u16(&self.funding_rate_prefix_sum);
        builder.register_public_input(self.mark_price);
        builder.register_public_input(self.quote_multiplier);

        let public_inputs_after = builder.num_public_inputs();
        assert_eq!(
            public_inputs_after - public_inputs_before,
            PublicMarketDetails::PARTIAL_PUBLIC_INPUTS_SIZE
        );
    }

    pub fn partial_from_public_inputs(pis: Vec<Target>) -> Self {
        assert_eq!(pis.len(), PublicMarketDetails::PARTIAL_PUBLIC_INPUTS_SIZE);
        Self {
            funding_rate_prefix_sum: BigIntU16Target::from_vec(&pis[0..5]),
            mark_price: pis[5],
            quote_multiplier: pis[6],
            ..Default::default()
        }
    }

    pub fn to_public_market_details(&self) -> Self {
        Self {
            funding_rate_prefix_sum: self.funding_rate_prefix_sum.clone(),
            mark_price: self.mark_price,
            quote_multiplier: self.quote_multiplier,
            ..Default::default()
        }
    }
}

pub fn random_access_market_risk_details(
    builder: &mut Builder,
    access_index: Target,
    v: Vec<MarketRiskDetailsTarget>,
) -> MarketRiskDetailsTarget {
    assert!(v.len().is_power_of_two());
    MarketRiskDetailsTarget {
        default_initial_margin_fraction: builder.random_access(
            access_index,
            v.iter()
                .map(|x| x.default_initial_margin_fraction)
                .collect(),
        ),
        min_initial_margin_fraction: builder.random_access(
            access_index,
            v.iter().map(|x| x.min_initial_margin_fraction).collect(),
        ),
        maintenance_margin_fraction: builder.random_access(
            access_index,
            v.iter().map(|x| x.maintenance_margin_fraction).collect(),
        ),
        close_out_margin_fraction: builder.random_access(
            access_index,
            v.iter().map(|x| x.close_out_margin_fraction).collect(),
        ),
        quote_multiplier: builder
            .random_access(access_index, v.iter().map(|x| x.quote_multiplier).collect()),
        funding_rate_prefix_sum: builder.random_access_bigint_u16(
            access_index,
            v.iter()
                .map(|x| x.funding_rate_prefix_sum.clone())
                .collect(),
            BIGU16_U64_LIMBS,
        ),
        mark_price: builder.random_access(access_index, v.iter().map(|x| x.mark_price).collect()),
        status: builder.random_access(access_index, v.iter().map(|x| x.status).collect()),
        strategy_index: builder
            .random_access(access_index, v.iter().map(|x| x.strategy_index).collect()),
    }
}

pub fn select_market_risk_details(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &MarketRiskDetailsTarget,
    b: &MarketRiskDetailsTarget,
) -> MarketRiskDetailsTarget {
    MarketRiskDetailsTarget {
        default_initial_margin_fraction: builder.select(
            flag,
            a.default_initial_margin_fraction,
            b.default_initial_margin_fraction,
        ),
        min_initial_margin_fraction: builder.select(
            flag,
            a.min_initial_margin_fraction,
            b.min_initial_margin_fraction,
        ),
        maintenance_margin_fraction: builder.select(
            flag,
            a.maintenance_margin_fraction,
            b.maintenance_margin_fraction,
        ),
        close_out_margin_fraction: builder.select(
            flag,
            a.close_out_margin_fraction,
            b.close_out_margin_fraction,
        ),
        quote_multiplier: builder.select(flag, a.quote_multiplier, b.quote_multiplier),
        funding_rate_prefix_sum: builder.select_bigint_u16(
            flag,
            &a.funding_rate_prefix_sum,
            &b.funding_rate_prefix_sum,
        ),
        mark_price: builder.select(flag, a.mark_price, b.mark_price),
        status: builder.select(flag, a.status, b.status),
        strategy_index: builder.select(flag, a.strategy_index, b.strategy_index),
    }
}

pub fn connect_market_risk_details(
    builder: &mut Builder,
    lhs: &MarketRiskDetailsTarget,
    rhs: &MarketRiskDetailsTarget,
) {
    builder.connect(
        lhs.default_initial_margin_fraction,
        rhs.default_initial_margin_fraction,
    );
    builder.connect(
        lhs.min_initial_margin_fraction,
        rhs.min_initial_margin_fraction,
    );
    builder.connect(
        lhs.maintenance_margin_fraction,
        rhs.maintenance_margin_fraction,
    );
    builder.connect(lhs.close_out_margin_fraction, rhs.close_out_margin_fraction);
    builder.connect(lhs.quote_multiplier, rhs.quote_multiplier);
    builder.connect_bigint_u16(&lhs.funding_rate_prefix_sum, &rhs.funding_rate_prefix_sum);
    builder.connect(lhs.mark_price, rhs.mark_price);
    builder.connect(lhs.status, rhs.status);
    builder.connect(lhs.strategy_index, rhs.strategy_index);
}

pub trait MarketRiskDetailsWitness<F: PrimeField64> {
    fn set_market_risk_details_target(
        &mut self,
        t: &MarketRiskDetailsTarget,
        mi: &MarketRiskDetails,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> MarketRiskDetailsWitness<F> for T {
    fn set_market_risk_details_target(
        &mut self,
        t: &MarketRiskDetailsTarget,
        mi: &MarketRiskDetails,
    ) -> Result<()> {
        self.set_bigint_u16_target(&t.funding_rate_prefix_sum, &mi.funding_rate_prefix_sum)?;
        self.set_target(t.mark_price, F::from_canonical_u32(mi.mark_price))?;
        self.set_target(
            t.quote_multiplier,
            F::from_canonical_u32(mi.quote_multiplier),
        )?;
        self.set_target(t.status, F::from_canonical_u8(mi.status))?;
        self.set_target(t.strategy_index, F::from_canonical_u8(mi.strategy_index))?;
        self.set_target(
            t.default_initial_margin_fraction,
            F::from_canonical_u16(mi.default_initial_margin_fraction),
        )?;
        self.set_target(
            t.min_initial_margin_fraction,
            F::from_canonical_u16(mi.min_initial_margin_fraction),
        )?;
        self.set_target(
            t.maintenance_margin_fraction,
            F::from_canonical_u16(mi.maintenance_margin_fraction),
        )?;
        self.set_target(
            t.close_out_margin_fraction,
            F::from_canonical_u16(mi.close_out_margin_fraction),
        )?;

        Ok(())
    }
}
