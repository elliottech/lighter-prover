// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::BIG_U96_LIMBS;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
#[serde(default)]
pub struct AccountAsset {
    #[serde(rename = "i")]
    pub index_0: i64,
    #[serde(rename = "b")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub balance: BigUint,
    #[serde(rename = "lb")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub locked_balance: BigUint,
}

impl Default for AccountAsset {
    fn default() -> Self {
        Self {
            index_0: 0,
            balance: BigUint::ZERO,
            locked_balance: BigUint::ZERO,
        }
    }
}

impl AccountAsset {
    pub fn empty(index_0: i64) -> Self {
        Self {
            index_0,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AccountAssetTarget {
    pub index_0: Target,
    pub balance: BigUintTarget,
    pub locked_balance: BigUintTarget,
}

impl AccountAssetTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            index_0: builder.add_virtual_target(),
            balance: builder.add_virtual_biguint_target_unsafe(BIG_U96_LIMBS),
            locked_balance: builder.add_virtual_biguint_target_unsafe(BIG_U96_LIMBS),
        }
    }

    pub fn get_available_balance(&self, builder: &mut Builder) -> BigUintTarget {
        let (available_balance, borrow) =
            builder.try_sub_biguint(&self.balance, &self.locked_balance);
        let is_borrow_zero = builder.is_zero_u32(borrow);
        builder.mul_biguint_by_bool(&available_balance, is_borrow_zero)
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // Adding 6 u32 limbs and a bool does not overflow Goldilocks, as long as
        // limbs are guaranteed by business logic to fit 32 bits.
        let added = builder.add_many(
            [&self.balance, &self.locked_balance]
                .iter()
                .flat_map(|x| x.limbs.iter().map(|limb| limb.0))
                .collect::<Vec<_>>(),
        );
        builder.is_zero(added)
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.index_0, &format!("{} index_0", tag));
        builder.println_biguint(&self.balance, &format!("{} balance", tag));
        builder.println_biguint(&self.locked_balance, &format!("{} locked_balance", tag));
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let mut elements = vec![];

        let mut limbs = self.balance.limbs.clone();
        limbs.resize(BIG_U96_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }

        let mut limbs = self.locked_balance.limbs.clone();
        limbs.resize(BIG_U96_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }

        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let empty_hash = builder.zero_hash_out();
        let is_empty = self.is_empty(builder);

        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }
}

pub trait AccountAssetTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_asset_target(&mut self, a: &AccountAssetTarget, b: &AccountAsset) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountAssetTargetWitness<F> for T
{
    fn set_account_asset_target(&mut self, a: &AccountAssetTarget, b: &AccountAsset) -> Result<()> {
        self.set_target(a.index_0, F::from_canonical_i64(b.index_0))?;
        self.set_biguint_target(&a.balance, &b.balance)?;
        self.set_biguint_target(&a.locked_balance, &b.locked_balance)?;

        Ok(())
    }
}

pub fn diff_account_asset(
    builder: &mut Builder,
    new: &AccountAssetTarget,
    old: &AccountAssetTarget,
) -> AccountAssetTarget {
    AccountAssetTarget {
        index_0: old.index_0,
        balance: builder.biguint_vector_diff(&new.balance, &old.balance),
        locked_balance: builder.biguint_vector_diff(&new.locked_balance, &old.locked_balance),
    }
}

pub fn apply_diff_account_asset(
    builder: &mut Builder,
    flag: BoolTarget,
    diff: &AccountAssetTarget,
    old: &AccountAssetTarget,
) -> AccountAssetTarget {
    AccountAssetTarget {
        index_0: old.index_0,
        balance: builder.biguint_vector_sum(flag, &diff.balance, &old.balance),
        locked_balance: builder.biguint_vector_sum(flag, &diff.locked_balance, &old.locked_balance),
    }
}

pub fn select_account_asset_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &AccountAssetTarget,
    b: &AccountAssetTarget,
) -> AccountAssetTarget {
    AccountAssetTarget {
        index_0: builder.select(flag, a.index_0, b.index_0),
        balance: builder.select_biguint(flag, &a.balance, &b.balance),
        locked_balance: builder.select_biguint(flag, &a.locked_balance, &b.locked_balance),
    }
}
