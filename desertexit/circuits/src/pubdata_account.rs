// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use circuit::bigint::big_u16::{BigIntU16Target, CircuitBuilderBigIntU16, WitnessBigInt16};
use circuit::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use circuit::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::eddsa::gadgets::curve::PartialWitnessCurve;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::poseidon2::Poseidon2Hash;
use circuit::types::config::{BIG_U96_LIMBS, BIG_U160_LIMBS, Builder, *};
use circuit::types::constants::*;
use circuit::uint::u16::gadgets::arithmetic_u16::CircuitBuilderU16;
use circuit::utils::CircuitBuilderUtils;
use num::{BigInt, BigUint};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(bound = "", default)]
pub struct PubdataPublicPoolShare {
    #[serde(rename = "ppi")]
    pub public_pool_index: i64,
    #[serde(rename = "sa")]
    pub share_amount: i64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PubdataPublicPoolShareTarget {
    pub public_pool_index: Target,
    pub share_amount: Target,
}

impl PubdataPublicPoolShareTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            public_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),
        }
    }
    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero(self.public_pool_index),
            builder.is_zero(self.share_amount),
        ];
        builder.multi_and(&assertions)
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
pub struct PubdataPublicPoolInfo {
    #[serde(rename = "ppi_tsa", default)]
    pub total_shares: i64,
    #[serde(rename = "ppi_os", default)]
    pub operator_shares: i64,
}

#[derive(Debug, Clone, Default)]
pub struct PubdataPublicPoolInfoTarget {
    pub total_shares: Target,
    pub operator_shares: Target,
}

impl PubdataPublicPoolInfoTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            total_shares: builder.add_virtual_target(),
            operator_shares: builder.add_virtual_target(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PubdataAccountPosition {
    #[serde(rename = "lfrps")]
    #[serde(deserialize_with = "circuit::deserializers::int_to_bigint")]
    pub last_funding_rate_prefix_sum: BigInt, // 63 bits
    #[serde(rename = "p")]
    #[serde(deserialize_with = "circuit::deserializers::int_to_bigint")]
    pub position: BigInt, // 56 bits
}

#[derive(Debug, Clone, Default)]
pub struct PubdataAccountPositionTarget {
    pub last_funding_rate_prefix_sum: BigIntU16Target, // 63 bits
    pub position: BigIntU16Target,                     // 56 bits
}

impl PubdataAccountPositionTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            last_funding_rate_prefix_sum: builder
                .add_virtual_bigint_u16_target_safe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
            position: builder.add_virtual_bigint_u16_target_safe(BIGU16_U64_LIMBS), // safe because it is read from the state using merkle proofs
        }
    }
    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            last_funding_rate_prefix_sum: builder.zero_bigint_u16(),
            position: builder.zero_bigint_u16(),
        }
    }
    pub fn append_position_pub_data_hash_params(
        &self,
        builder: &mut Builder,
        elements: &mut Vec<Target>,
    ) {
        let mut limbs = self.last_funding_rate_prefix_sum.abs.limbs.clone();
        limbs.resize(BIGU16_U64_LIMBS, builder.zero_u16());
        for limb in limbs {
            elements.push(limb.0);
        }
        elements.push(self.last_funding_rate_prefix_sum.sign.target);

        let mut limbs = self.position.abs.limbs.clone();
        limbs.resize(BIGU16_U64_LIMBS, builder.zero_u16());
        for limb in limbs {
            elements.push(limb.0);
        }
        elements.push(self.position.sign.target);
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "", default)]
pub struct PubdataAccount {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "l1")]
    #[serde(deserialize_with = "circuit::deserializers::l1_address_to_biguint")]
    pub l1_address: BigUint, // 160 bits
    #[serde(rename = "at")]
    pub account_type: u8,
    #[serde(rename = "abal")] // Only included in pub data tree
    #[serde(deserialize_with = "crate::deserializers::aggregated_assets")]
    pub aggregated_assets: [BigInt; ASSET_LIST_SIZE], // 96 bits
    #[serde(rename = "ap")]
    #[serde(deserialize_with = "crate::deserializers::pubdata_positions")]
    pub positions: [PubdataAccountPosition; POSITION_LIST_SIZE],
    #[serde(rename = "pps", default)]
    pub public_pool_shares: [PubdataPublicPoolShare; SHARES_LIST_SIZE],
    #[serde(rename = "ppi")]
    pub public_pool_info: PubdataPublicPoolInfo,
}

impl Default for PubdataAccount {
    fn default() -> Self {
        Self {
            account_index: 0,
            l1_address: BigUint::ZERO,
            account_type: 0,
            aggregated_assets: array::from_fn(|_| BigInt::from(0)),
            positions: array::from_fn(|_| PubdataAccountPosition::default()),
            public_pool_shares: array::from_fn(|_| PubdataPublicPoolShare::default()),
            public_pool_info: PubdataPublicPoolInfo::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PubdataAccountTarget {
    pub account_index: Target,
    pub l1_address: BigUintTarget,
    pub account_type: Target,
    pub aggregated_assets: [BigIntTarget; ASSET_LIST_SIZE],
    pub positions: [PubdataAccountPositionTarget; POSITION_LIST_SIZE],
    pub public_pool_shares: [PubdataPublicPoolShareTarget; SHARES_LIST_SIZE],
    pub public_pool_info: PubdataPublicPoolInfoTarget,
}

impl Default for PubdataAccountTarget {
    fn default() -> Self {
        Self {
            account_index: Target::default(),
            l1_address: BigUintTarget::default(),
            account_type: Target::default(),
            aggregated_assets: array::from_fn(|_| BigIntTarget::default()),
            positions: array::from_fn(|_| PubdataAccountPositionTarget::default()),
            public_pool_shares: array::from_fn(|_| PubdataPublicPoolShareTarget::default()),
            public_pool_info: PubdataPublicPoolInfoTarget::default(),
        }
    }
}

impl PubdataAccountTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_safe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),
            aggregated_assets: array::from_fn(|_| {
                builder.add_virtual_bigint_target_safe(BIG_U96_LIMBS) // safe because it is read from the state using merkle proofs
            }),
            positions: array::from_fn(|_| PubdataAccountPositionTarget::new(builder)),
            public_pool_shares: array::from_fn(|_| PubdataPublicPoolShareTarget::new(builder)),
            public_pool_info: PubdataPublicPoolInfoTarget::new(builder),
        }
    }

    fn get_asset_delta_root(&self, builder: &mut Builder) -> HashOutTarget {
        let mut level_hashes = self
            .aggregated_assets
            .iter()
            .map(|a| {
                let mut elements = vec![a.sign.target];
                elements.extend_from_slice(&a.abs.limbs.iter().map(|x| x.0).collect::<Vec<_>>());
                let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
                let empty_hash = builder.zero_hash_out();
                let is_empty = builder.is_zero_bigint(a);
                builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
            })
            .collect::<Vec<_>>();
        assert!((1 << ASSET_LIST_SIZE_BITS) == level_hashes.len());
        let mut iter_count = level_hashes.len() / 2;
        for _ in 0..ASSET_LIST_SIZE_BITS {
            for j in 0..iter_count {
                level_hashes[j] =
                    builder.hash_two_to_one(&level_hashes[2 * j], &level_hashes[2 * j + 1]);
            }
            iter_count /= 2;
        }
        level_hashes[0]
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        builder.is_equal_constant(self.account_index, NIL_ACCOUNT_INDEX as u64)
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let position_bucket_hashes: [HashOutTarget; POSITION_HASH_BUCKET_COUNT] = {
            let mut positions_ext = self.positions.to_vec();
            positions_ext.push(PubdataAccountPositionTarget::empty(builder));
            positions_ext
                .chunks(POSITION_HASH_BUCKET_SIZE)
                .map(|bucket: &[PubdataAccountPositionTarget]| {
                    let mut pub_data_hash_params = vec![];
                    for pos in bucket {
                        pos.append_position_pub_data_hash_params(
                            builder,
                            &mut pub_data_hash_params,
                        );
                    }
                    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_hash_params)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };

        let partial_pubdata_hash = {
            let mut pub_data_elements = vec![];

            pub_data_elements.extend_from_slice(
                &position_bucket_hashes
                    .iter()
                    .flat_map(|x| x.elements)
                    .collect::<Vec<_>>(),
            );

            pub_data_elements.extend_from_slice(
                &self
                    .public_pool_shares
                    .iter()
                    .flat_map(|pps| [pps.public_pool_index, pps.share_amount])
                    .collect::<Vec<_>>(),
            );

            pub_data_elements.extend_from_slice(&[
                self.public_pool_info.total_shares,
                self.public_pool_info.operator_shares,
            ]);

            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_elements)
        };

        let non_empty_pub_data_hash = {
            let mut pub_data_elements = vec![];

            pub_data_elements.extend_from_slice(&partial_pubdata_hash.elements);
            pub_data_elements.extend_from_slice(
                &self
                    .l1_address
                    .limbs
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>(),
            );
            pub_data_elements.push(self.account_type);

            let asset_delta_root = self.get_asset_delta_root(builder);
            pub_data_elements.extend_from_slice(&asset_delta_root.elements);

            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_elements)
        };

        let empty_hash = builder.zero_hash_out();
        let is_new_account = self.is_empty(builder);

        builder.select_hash(is_new_account, &empty_hash, &non_empty_pub_data_hash)
    }
}

pub trait PubdataAccountTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_pubdata_account_target(
        &mut self,
        a: &PubdataAccountTarget,
        b: &PubdataAccount,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    PubdataAccountTargetWitness<F> for T
{
    fn set_pubdata_account_target(
        &mut self,
        a: &PubdataAccountTarget,
        b: &PubdataAccount,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_biguint_target(&a.l1_address, &b.l1_address)?;
        self.set_target(a.account_type, F::from_canonical_u8(b.account_type))?;

        for i in 0..b.aggregated_assets.len() {
            self.set_bigint_target(&a.aggregated_assets[i], &b.aggregated_assets[i])?;
        }

        for i in 0..b.positions.len() {
            self.set_bigint_u16_target(
                &a.positions[i].last_funding_rate_prefix_sum,
                &b.positions[i].last_funding_rate_prefix_sum,
            )?;
            self.set_bigint_u16_target(&a.positions[i].position, &b.positions[i].position)?;
        }

        for i in 0..b.public_pool_shares.len() {
            self.set_target(
                a.public_pool_shares[i].public_pool_index,
                F::from_canonical_i64(b.public_pool_shares[i].public_pool_index),
            )?;
            self.set_target(
                a.public_pool_shares[i].share_amount,
                F::from_canonical_i64(b.public_pool_shares[i].share_amount),
            )?;
        }

        self.set_target(
            a.public_pool_info.total_shares,
            F::from_canonical_i64(b.public_pool_info.total_shares),
        )?;
        self.set_target(
            a.public_pool_info.operator_shares,
            F::from_canonical_i64(b.public_pool_info.operator_shares),
        )?;

        Ok(())
    }
}
