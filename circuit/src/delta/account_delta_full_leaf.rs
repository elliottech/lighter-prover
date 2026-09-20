// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use num::{BigInt, BigUint};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers::{self, MarketDataDeltas};
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::account_delta::AccountDeltaTarget;
use crate::types::account_delta::market_data_delta::{
    BinaryOptionsDeltaTarget, BinaryOptionsDeltaTargetWitness, MarketDataDeltaTarget,
    MarketDataDeltaTargetWitness,
};
use crate::types::account_delta::public_pool_delta::{
    PublicPoolInfoDelta, PublicPoolInfoDeltaTarget, PublicPoolInfoDeltaWitness,
    PublicPoolShareDelta, PublicPoolShareDeltaTarget, PublicPoolShareDeltaWitness,
};
use crate::types::config::{BIG_U96_LIMBS, BIG_U160_LIMBS, Builder};
use crate::types::constants::{
    ASSET_LIST_SIZE, ASSET_LIST_SIZE_BITS, BINARY_OPTIONS_MARKET_SLOT_COUNT,
    EMPTY_DELTA_TREE_HASHES, MARKET_MERKLE_LEVELS, MAX_BINARY_OPTIONS_MARKET_INDEX,
    MIN_BINARY_OPTIONS_MARKET_INDEX, NIL_ACCOUNT_INDEX, NIL_MASTER_ACCOUNT_INDEX,
    POSITION_LIST_SIZE, SHARES_DELTA_LIST_SIZE,
};

/// Similar to AccountDelta, but comes with all market data deltas (perps positions and binary
/// options positions) instead of one position and a position tree root.
#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
#[serde(default)]
pub struct AccountDeltaFullLeaf {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "l1")]
    #[serde(deserialize_with = "deserializers::l1_address_to_biguint")]
    pub l1_address: BigUint,
    #[serde(rename = "at", default)]
    pub account_type: u8,
    #[serde(rename = "aad")]
    #[serde(deserialize_with = "deserializers::all_aggregated_asset_deltas")]
    pub aggregated_asset_deltas: [BigInt; ASSET_LIST_SIZE],
    #[serde(rename = "pd")]
    #[serde(deserialize_with = "deserializers::market_data_deltas")]
    pub market_data_deltas: MarketDataDeltas,
    #[serde(rename = "ppsd")]
    #[serde(deserialize_with = "deserializers::public_pool_shares_delta")]
    pub public_pool_shares_delta: [PublicPoolShareDelta; SHARES_DELTA_LIST_SIZE],
    #[serde(rename = "ppid", default)]
    pub public_pool_info_delta: PublicPoolInfoDelta,
}

impl AccountDeltaFullLeaf {
    pub fn nil() -> Self {
        Self {
            account_index: NIL_ACCOUNT_INDEX,
            ..Self::default()
        }
    }
}

impl Default for AccountDeltaFullLeaf {
    fn default() -> Self {
        Self {
            account_index: NIL_MASTER_ACCOUNT_INDEX,
            l1_address: BigUint::ZERO,
            account_type: 0,
            aggregated_asset_deltas: array::from_fn(|_| BigInt::ZERO),
            market_data_deltas: MarketDataDeltas::default(),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDelta::default()),
            public_pool_info_delta: PublicPoolInfoDelta::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccountDeltaFullLeafTarget {
    pub account_index: Target,
    pub l1_address: BigUintTarget,
    pub account_type: Target,
    pub aggregated_asset_deltas: [BigIntTarget; ASSET_LIST_SIZE],
    pub perps_deltas: [MarketDataDeltaTarget; POSITION_LIST_SIZE],
    /// Slot `i` is the binary options market `MIN_BINARY_OPTIONS_MARKET_INDEX + i`.
    pub binary_options_deltas: [BinaryOptionsDeltaTarget; BINARY_OPTIONS_MARKET_SLOT_COUNT],
    pub public_pool_shares_delta: [PublicPoolShareDeltaTarget; SHARES_DELTA_LIST_SIZE],
    pub public_pool_info_delta: PublicPoolInfoDeltaTarget,
}

impl Default for AccountDeltaFullLeafTarget {
    fn default() -> Self {
        AccountDeltaFullLeafTarget {
            account_index: Target::default(),
            l1_address: BigUintTarget::default(),
            account_type: Target::default(),
            aggregated_asset_deltas: array::from_fn(|_| BigIntTarget::default()),
            perps_deltas: array::from_fn(|_| MarketDataDeltaTarget::default()),
            binary_options_deltas: array::from_fn(|_| BinaryOptionsDeltaTarget::default()),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::default()),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::default(),
        }
    }
}

impl AccountDeltaFullLeafTarget {
    pub fn new(builder: &mut Builder) -> Self {
        AccountDeltaFullLeafTarget {
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS),
            account_type: builder.add_virtual_target(),
            aggregated_asset_deltas: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),
            perps_deltas: array::from_fn(|_| MarketDataDeltaTarget::new(builder)),
            binary_options_deltas: array::from_fn(|_| BinaryOptionsDeltaTarget::new(builder)),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::new(builder)),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::new(builder),
        }
    }

    pub fn to_account_delta(&self, builder: &mut Builder) -> AccountDeltaTarget {
        AccountDeltaTarget {
            account_index: self.account_index,
            l1_address: self.l1_address.clone(),
            account_type: self.account_type,
            aggregated_asset_deltas: array::from_fn(|i| self.aggregated_asset_deltas[i].clone()),
            public_pool_shares_delta: self.public_pool_shares_delta,
            public_pool_info_delta: self.public_pool_info_delta.clone(),
            asset_delta_root: self.get_asset_delta_root(builder),
            position_delta_root: self.get_position_delta_root(builder),
            market_data_delta: MarketDataDeltaTarget::default(),
            partial_hash: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> (HashOutTarget, BoolTarget) {
        self.to_account_delta(builder).hash_with_is_empty(builder)
    }

    pub fn get_asset_delta_root(&self, builder: &mut Builder) -> HashOutTarget {
        let mut level_hashes = self
            .aggregated_asset_deltas
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

    // Root of the MARKET_MERKLE_LEVELS deep account market data delta tree keyed by market slot.
    // Perps occupy slots `0..=POSITION_LIST_SIZE` (the last one is the nil perps market slot),
    // binary options occupy `MIN_BINARY_OPTIONS_MARKET_INDEX..=MAX_BINARY_OPTIONS_MARKET_INDEX`,
    // every other slot is an empty subtree.
    pub fn get_position_delta_root(&self, builder: &mut Builder) -> HashOutTarget {
        const _: () = assert!(MAX_BINARY_OPTIONS_MARKET_INDEX < (1 << MARKET_MERKLE_LEVELS));
        const _: () = assert!(POSITION_LIST_SIZE < MIN_BINARY_OPTIONS_MARKET_INDEX);

        // (slot, node hash) in strictly increasing slot order, slots not listed are empty
        let mut nodes = self
            .perps_deltas
            .iter()
            .enumerate()
            .map(|(slot, p)| (slot, p.hash_perps(builder)))
            .collect::<Vec<_>>();
        nodes.push((POSITION_LIST_SIZE, builder.zero_hash_out())); // nil perps market slot
        nodes.extend(
            self.binary_options_deltas
                .iter()
                .enumerate()
                .map(|(i, d)| (MIN_BINARY_OPTIONS_MARKET_INDEX + i, d.hash(builder))),
        );

        for level in 0..MARKET_MERKLE_LEVELS {
            let empty = builder.constant_hash(EMPTY_DELTA_TREE_HASHES[level]);
            let mut parents = Vec::with_capacity(nodes.len() / 2 + 1);
            let mut i = 0;
            while i < nodes.len() {
                let (slot, hash) = nodes[i];
                i += 1;
                let (left, right) = if slot % 2 == 1 {
                    (empty, hash)
                } else if i < nodes.len() && nodes[i].0 == slot + 1 {
                    i += 1;
                    (hash, nodes[i - 1].1)
                } else {
                    (hash, empty)
                };
                parents.push((slot >> 1, builder.hash_two_to_one(&left, &right)));
            }
            nodes = parents;
        }
        assert_eq!(nodes.len(), 1);
        nodes[0].1
    }

    pub fn print(&self, builder: &mut Builder, print_assets: bool, tag: &str) {
        builder.println(self.account_index, &format!("{} account_index", tag));
        builder.println_biguint(&self.l1_address, &format!("{}: l1_address", tag));
        builder.println(self.account_type, &format!("{} account_type", tag));
        self.public_pool_info_delta
            .print(builder, &format!("{} public_pool_info_delta", tag));

        if print_assets {
            for i in 0..ASSET_LIST_SIZE {
                builder.println_bigint(
                    &self.aggregated_asset_deltas[i],
                    &format!("{} aggregated_asset_deltas[{}]", tag, i),
                );
            }
        }
    }
}

pub trait AccountDeltaLeafTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_delta_leaf_target(
        &mut self,
        a: &AccountDeltaFullLeafTarget,
        b: &AccountDeltaFullLeaf,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountDeltaLeafTargetWitness<F> for T
{
    fn set_account_delta_leaf_target(
        &mut self,
        a: &AccountDeltaFullLeafTarget,
        b: &AccountDeltaFullLeaf,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_biguint_target(&a.l1_address, &b.l1_address)?;
        self.set_target(a.account_type, F::from_canonical_u8(b.account_type))?;
        for i in 0..b.aggregated_asset_deltas.len() {
            self.set_bigint_target(&a.aggregated_asset_deltas[i], &b.aggregated_asset_deltas[i])?;
        }
        for i in 0..b.market_data_deltas.perps.len() {
            self.set_market_data_delta_target(&a.perps_deltas[i], &b.market_data_deltas.perps[i])?;
        }
        for i in 0..b.market_data_deltas.binary_options.len() {
            self.set_binary_options_delta_target(
                &a.binary_options_deltas[i],
                &b.market_data_deltas.binary_options[i],
            )?;
        }
        self.set_public_pool_info_delta(&a.public_pool_info_delta, &b.public_pool_info_delta)?;
        for i in 0..b.public_pool_shares_delta.len() {
            self.set_public_pool_share_delta(
                &a.public_pool_shares_delta[i],
                &b.public_pool_shares_delta[i],
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::PrimeField64;
    use plonky2::iop::generator::generate_partial_witness;
    use plonky2::iop::witness::{PartialWitness, Witness};

    use super::*;
    use crate::types::config::{C, CIRCUIT_CONFIG};
    use crate::types::constants::EMPTY_POSITION_DELTA_TREE_ROOT;

    fn position_delta_root(
        perps: &[(usize, i64, i64)],
        binary_options: &[(usize, i64)],
    ) -> [u64; 4] {
        let mut leaf = AccountDeltaFullLeaf::default();
        for &(slot, funding_rate_prefix_sum_delta, position_delta) in perps {
            leaf.market_data_deltas.perps[slot].funding_rate_prefix_sum_delta =
                BigInt::from(funding_rate_prefix_sum_delta);
            leaf.market_data_deltas.perps[slot].size_delta = BigInt::from(position_delta);
        }
        for &(market_index, size_delta) in binary_options {
            leaf.market_data_deltas.binary_options
                [market_index - MIN_BINARY_OPTIONS_MARKET_INDEX]
                .size_delta = BigInt::from(size_delta);
        }

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let target = AccountDeltaFullLeafTarget::new(&mut builder);
        let root = target.get_position_delta_root(&mut builder);
        let mut pw = PartialWitness::new();
        pw.set_account_delta_leaf_target(&target, &leaf).unwrap();
        let data = builder.build::<C>();
        let witness = generate_partial_witness(pw, &data.prover_only, &data.common).unwrap();
        array::from_fn(|i| witness.get_target(root.elements[i]).to_canonical_u64())
    }

    /// Expected roots are the 12 level sparse merkle tree roots of the same deltas.
    #[test]
    fn position_delta_root_matches_smt() {
        let empty = EMPTY_POSITION_DELTA_TREE_ROOT
            .elements
            .map(|e| e.to_canonical_u64());
        assert_eq!(position_delta_root(&[], &[]), empty);

        assert_eq!(
            position_delta_root(&[(0, 123456789, -5), (7, -1, 1 << 40), (254, 0, 77)], &[]),
            [
                17268912142426479405,
                1061622204288693832,
                4533079881209133628,
                17351973916199085783
            ]
        );

        assert_eq!(
            position_delta_root(
                &[],
                &[(1000, 1), (1001, -2), (1500, 1 << 50), (2000, -(1 << 33))]
            ),
            [
                17055592521599767197,
                3225394539601077115,
                12991912779457730388,
                575801448487009982
            ]
        );

        assert_eq!(
            position_delta_root(
                &[(3, -(1 << 61), -(1 << 55)), (254, 5, 6)],
                &[(1000, 9), (1023, -9), (1024, 10), (1999, 11), (2000, 12)],
            ),
            [
                1554745711956725730,
                7935720410041012139,
                14764864376122201231,
                4141581196565293354
            ]
        );
    }
}
