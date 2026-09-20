// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use num::{BigInt, BigUint};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::market_data_delta::{
    MarketDataDelta, MarketDataDeltaTarget, MarketDataDeltaTargetWitness,
};
use super::public_pool_delta::{
    PublicPoolInfoDelta, PublicPoolInfoDeltaTarget, PublicPoolInfoDeltaWitness,
    PublicPoolShareDelta, PublicPoolShareDeltaTarget, PublicPoolShareDeltaWitness,
};
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::types::account_delta::select_public_pool_share_delta_target;
use crate::types::config::{BIG_U96_LIMBS, BIG_U160_LIMBS, Builder};
use crate::types::constants::{NB_ASSETS_PER_TX, NIL_MASTER_ACCOUNT_INDEX, SHARES_DELTA_LIST_SIZE};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
#[serde(default)]
pub struct AccountDelta<F>
where
    F: PrimeField64 + Extendable<5> + RichField,
{
    #[serde(rename = "ai", default)]
    pub account_index: i64,

    #[serde(rename = "l1")]
    #[serde(deserialize_with = "deserializers::l1_address_to_biguint")]
    pub l1_address: BigUint, // 160 bits, non-empty for new accounts

    #[serde(rename = "at", default)]
    pub account_type: u8, // 8 bits, non-empty for new accounts

    #[serde(rename = "aad")] // Only included in pub data tree
    #[serde(deserialize_with = "deserializers::aggregated_asset_deltas")]
    pub aggregated_asset_deltas: [BigInt; NB_ASSETS_PER_TX], // 96 bits

    #[serde(rename = "pd", default)]
    pub market_data_delta: MarketDataDelta,

    #[serde(rename = "ppsd")]
    #[serde(deserialize_with = "deserializers::public_pool_shares_delta")]
    pub public_pool_shares_delta: [PublicPoolShareDelta; SHARES_DELTA_LIST_SIZE],

    #[serde(rename = "ppid", default)]
    pub public_pool_info_delta: PublicPoolInfoDelta,

    #[serde(rename = "adr", default)] // Only included in pub data tree
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub asset_delta_root: HashOut<F>,

    #[serde(rename = "ppddr", default)] // Only included in pub data tree
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub position_delta_root: HashOut<F>,

    #[serde(rename = "ph", default)]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub partial_hash: HashOut<F>,
}

impl<F: PrimeField64 + Extendable<5> + RichField> Default for AccountDelta<F> {
    fn default() -> Self {
        Self {
            account_index: NIL_MASTER_ACCOUNT_INDEX,
            l1_address: BigUint::ZERO,
            account_type: 0,
            aggregated_asset_deltas: array::from_fn(|_| BigInt::ZERO),
            market_data_delta: MarketDataDelta::default(),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDelta::default()),
            public_pool_info_delta: PublicPoolInfoDelta::default(),
            asset_delta_root: HashOut::ZERO,
            position_delta_root: HashOut::ZERO,
            partial_hash: HashOut::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccountDeltaTarget {
    pub account_index: Target,
    pub l1_address: BigUintTarget,
    pub account_type: Target,
    pub aggregated_asset_deltas: [BigIntTarget; NB_ASSETS_PER_TX],
    pub market_data_delta: MarketDataDeltaTarget,
    pub public_pool_shares_delta: [PublicPoolShareDeltaTarget; SHARES_DELTA_LIST_SIZE],
    pub public_pool_info_delta: PublicPoolInfoDeltaTarget,

    pub asset_delta_root: HashOutTarget,
    pub position_delta_root: HashOutTarget,

    pub partial_hash: HashOutTarget,
}

impl Default for AccountDeltaTarget {
    fn default() -> Self {
        AccountDeltaTarget {
            account_index: Target::default(),
            l1_address: BigUintTarget::default(),
            account_type: Target::default(),
            aggregated_asset_deltas: core::array::from_fn(|_| BigIntTarget::default()),
            market_data_delta: MarketDataDeltaTarget::default(),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::default()),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::default(),
            asset_delta_root: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
            position_delta_root: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
            partial_hash: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
        }
    }
}

impl AccountDeltaTarget {
    pub fn new(builder: &mut Builder) -> Self {
        AccountDeltaTarget {
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),
            aggregated_asset_deltas: core::array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),
            market_data_delta: MarketDataDeltaTarget::new(builder),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::new(builder)),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::new(builder),
            asset_delta_root: builder.add_virtual_hash(),
            position_delta_root: builder.add_virtual_hash(),
            partial_hash: builder.zero_hash_out(), // Unused for maker and taker accounts
        }
    }

    pub fn new_fee_account(builder: &mut Builder) -> Self {
        AccountDeltaTarget {
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),
            aggregated_asset_deltas: core::array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),
            market_data_delta: MarketDataDeltaTarget::default(),
            public_pool_shares_delta: array::from_fn(|_| PublicPoolShareDeltaTarget::new(builder)),
            public_pool_info_delta: PublicPoolInfoDeltaTarget::new(builder),
            asset_delta_root: builder.add_virtual_hash(),
            position_delta_root: builder.zero_hash_out(),
            partial_hash: builder.add_virtual_hash(), // Hash of fields that wouldn't change for the fee account
        }
    }

    pub fn apply_pool_pub_data_share_delta(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        pool_index: Target,
        share_delta: SignedTarget,
    ) {
        let zero = builder.zero();
        let old_pool_shares = self.public_pool_shares_delta;

        let mut applied = builder._false();
        let mut use_next = builder._false();

        let new_pool_shares_for_empty = PublicPoolShareDeltaTarget {
            public_pool_index: pool_index,
            shares_delta: share_delta,
        };
        let empty_pool_share = PublicPoolShareDeltaTarget::empty(builder, zero);
        let is_share_delta_non_zero = builder.is_not_zero(share_delta.target);
        let is_enabled = builder.and(is_enabled, is_share_delta_non_zero);

        for i in 0..SHARES_DELTA_LIST_SIZE {
            // Put if not applied already and the current slot is empty
            let is_pool_share_slot_empty =
                builder.is_zero(self.public_pool_shares_delta[i].shares_delta.target);
            let empty_and_not_applied = builder.and_not(is_pool_share_slot_empty, applied);
            let apply_empty = builder.and(empty_and_not_applied, is_enabled);
            applied = builder.or(applied, apply_empty);
            self.public_pool_shares_delta[i] = select_public_pool_share_delta_target(
                builder,
                apply_empty,
                &new_pool_shares_for_empty,
                &self.public_pool_shares_delta[i],
            );

            // Update an existing pool share delta. This can leave the current slot empty, and we handle them by
            // toggling use_next to true, which ensures the current and the following iterations to just shift the
            // old pool shares left by one slot.
            let is_pool_index_eq = builder.is_equal(
                self.public_pool_shares_delta[i].public_pool_index,
                pool_index,
            );
            let is_pool_index_eq_and_not_applied = builder.and_not(is_pool_index_eq, applied);
            let apply_delta = builder.and(is_pool_index_eq_and_not_applied, is_enabled);
            applied = builder.or(applied, apply_delta);

            let add_value =
                SignedTarget::new_unsafe(builder.mul_bool(apply_delta, share_delta.target));
            self.public_pool_shares_delta[i].shares_delta =
                builder.add_signed(self.public_pool_shares_delta[i].shares_delta, add_value);

            let is_new_share_amount_empty =
                builder.is_zero(self.public_pool_shares_delta[i].shares_delta.target);
            use_next = builder.select_bool(apply_delta, is_new_share_amount_empty, use_next);
            self.public_pool_shares_delta[i] = select_public_pool_share_delta_target(
                builder,
                use_next,
                &if i < SHARES_DELTA_LIST_SIZE - 1 {
                    old_pool_shares[i + 1]
                } else {
                    empty_pool_share
                },
                &self.public_pool_shares_delta[i],
            );
        }

        builder.conditional_assert_true(is_enabled, applied);
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.account_index, &format!("{} account_index", tag));
        builder.println_biguint(&self.l1_address, &format!("{}: l1_address", tag));
        builder.println(self.account_type, &format!("{} account_type", tag));
        builder.println_bigint(
            &self.aggregated_asset_deltas[0],
            &format!("{} aggregated_asset_deltas[0]", tag),
        );
        builder.println_bigint(
            &self.aggregated_asset_deltas[1],
            &format!("{} aggregated_asset_deltas[1]", tag),
        );
        self.market_data_delta
            .print(builder, &format!("{} market_data_delta", tag));
        self.public_pool_info_delta
            .print(builder, &format!("{} public_pool_info_delta", tag));
        builder.println_hash_out(
            &self.position_delta_root,
            &format!("{} position_delta_root", tag),
        );
        builder.println_hash_out(&self.asset_delta_root, &format!("{} asset_delta_root", tag));
    }
}

pub trait AccountDeltaTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_delta_target(
        &mut self,
        a: &AccountDeltaTarget,
        b: &AccountDelta<F>,
    ) -> Result<()>;

    fn set_fee_account_delta_target(
        &mut self,
        a: &AccountDeltaTarget,
        b: &AccountDelta<F>,
    ) -> Result<()>;

    fn _set_common_targets(&mut self, a: &AccountDeltaTarget, b: &AccountDelta<F>) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountDeltaTargetWitness<F> for T
{
    fn set_account_delta_target(
        &mut self,
        a: &AccountDeltaTarget,
        b: &AccountDelta<F>,
    ) -> Result<()> {
        self._set_common_targets(a, b)?;

        self.set_market_data_delta_target(&a.market_data_delta, &b.market_data_delta)?;
        self.set_public_pool_info_delta(&a.public_pool_info_delta, &b.public_pool_info_delta)?;
        for i in 0..b.public_pool_shares_delta.len() {
            self.set_public_pool_share_delta(
                &a.public_pool_shares_delta[i],
                &b.public_pool_shares_delta[i],
            )?;
        }

        self.set_hash_target(a.position_delta_root, b.position_delta_root)?;

        Ok(())
    }

    fn set_fee_account_delta_target(
        &mut self,
        a: &AccountDeltaTarget,
        b: &AccountDelta<F>,
    ) -> Result<()> {
        self._set_common_targets(a, b)?;
        self.set_hash_target(a.partial_hash, b.partial_hash)?;

        Ok(())
    }

    fn _set_common_targets(&mut self, a: &AccountDeltaTarget, b: &AccountDelta<F>) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_biguint_target(&a.l1_address, &b.l1_address)?;
        self.set_target(a.account_type, F::from_canonical_u8(b.account_type))?;
        for i in 0..NB_ASSETS_PER_TX {
            self.set_bigint_target(&a.aggregated_asset_deltas[i], &b.aggregated_asset_deltas[i])?;
        }
        self.set_hash_target(a.asset_delta_root, b.asset_delta_root)?;

        Ok(())
    }
}
