// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigInt;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::types::config::BIG_U96_LIMBS;

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(bound = "")]
#[serde(default)]
pub struct AccountMarginedAsset {
    #[serde(rename = "b")]
    pub balance: i128,
    #[serde(rename = "mm", default)]
    pub margin_mode: u8,
}

impl AccountMarginedAsset {
    pub fn empty() -> Self {
        Default::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct AccountMarginedAssetTarget {
    pub balance: BigIntTarget,
    pub margin_mode: Target,
}

impl AccountMarginedAssetTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            balance: builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS),
            margin_mode: builder.add_virtual_target(),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            balance: builder.zero_bigint(),
            margin_mode: builder.zero(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_bigint(&self.balance, &format!("{} balance", tag));
        builder.println(self.margin_mode, &format!("{} margin_mode", tag));
    }
}

pub trait AccountMarginedAssetTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_margined_asset_target(
        &mut self,
        a: &AccountMarginedAssetTarget,
        b: &AccountMarginedAsset,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountMarginedAssetTargetWitness<F> for T
{
    fn set_account_margined_asset_target(
        &mut self,
        a: &AccountMarginedAssetTarget,
        b: &AccountMarginedAsset,
    ) -> Result<()> {
        self.set_bigint_target(&a.balance, &BigInt::from(b.balance))?;
        self.set_target(a.margin_mode, F::from_canonical_u8(b.margin_mode))?;

        Ok(())
    }
}

pub fn diff_account_margined_asset(
    builder: &mut Builder,
    new: &AccountMarginedAssetTarget,
    old: &AccountMarginedAssetTarget,
) -> AccountMarginedAssetTarget {
    AccountMarginedAssetTarget {
        balance: builder.bigint_vector_diff(&new.balance, &old.balance),
        margin_mode: builder.sub(new.margin_mode, old.margin_mode),
    }
}

pub fn apply_diff_account_margined_asset(
    builder: &mut Builder,
    flag: BoolTarget,
    diff: &AccountMarginedAssetTarget,
    old: &AccountMarginedAssetTarget,
) -> AccountMarginedAssetTarget {
    AccountMarginedAssetTarget {
        balance: builder.bigint_vector_sum(flag, &diff.balance, &old.balance),
        margin_mode: builder.mul_add(flag.target, diff.margin_mode, old.margin_mode),
    }
}

pub fn select_account_margined_asset_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &AccountMarginedAssetTarget,
    b: &AccountMarginedAssetTarget,
) -> AccountMarginedAssetTarget {
    AccountMarginedAssetTarget {
        balance: builder.select_bigint(flag, &a.balance, &b.balance),
        margin_mode: builder.select(flag, a.margin_mode, b.margin_mode),
    }
}

pub fn random_access_account_margined_asset_target(
    builder: &mut Builder,
    access_index: Target,
    accounts: &[AccountMarginedAssetTarget],
) -> AccountMarginedAssetTarget {
    AccountMarginedAssetTarget {
        balance: builder.random_access_bigint(
            access_index,
            &accounts
                .iter()
                .map(|a| a.balance.clone())
                .collect::<Vec<_>>(),
            BIG_U96_LIMBS,
        ),
        margin_mode: builder.random_access(
            access_index,
            accounts.iter().map(|a| a.margin_mode).collect::<Vec<_>>(),
        ),
    }
}
