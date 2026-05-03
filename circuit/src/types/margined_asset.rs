// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigUint, Zero};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::BIG_U96_LIMBS;
use crate::types::constants::*;
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::utils::CircuitBuilderUtils;

pub const MARGINED_ASSET_SIZE: usize = 16;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
#[serde(default)]
pub struct MarginedAsset {
    #[serde(rename = "mi")]
    pub margin_index: u8,
    #[serde(rename = "ai")]
    pub asset_index: i16,
    #[serde(rename = "ltv")]
    pub loan_to_value: u16,
    #[serde(rename = "lt")]
    pub liquidation_threshold: u16,
    #[serde(rename = "lfc")]
    pub liquidation_factor: u32,
    #[serde(rename = "lf")]
    pub liquidation_fee: u32,
    #[serde(rename = "ip")]
    pub index_price: i64,
    #[serde(rename = "ipd")]
    pub index_price_divider: i64,
    #[serde(rename = "gsc")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub global_supply_cap: BigUint,
    #[serde(rename = "usc")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub user_supply_cap: BigUint,
    #[serde(rename = "tsa")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub total_supplied_amount: BigUint,
}

impl MarginedAsset {
    pub fn from_public_inputs<F>(margin_index: u8, pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert_eq!(pis.len(), MARGINED_ASSET_SIZE);

        let global_supply_cap = pis[7..10].iter().rev().fold(BigUint::zero(), |acc, limb| {
            (acc << 32) + limb.to_canonical_biguint()
        });

        let user_supply_cap = pis[10..13].iter().rev().fold(BigUint::zero(), |acc, limb| {
            (acc << 32) + limb.to_canonical_biguint()
        });

        let total_supplied_amount = pis[13..16].iter().rev().fold(BigUint::zero(), |acc, limb| {
            (acc << 32) + limb.to_canonical_biguint()
        });

        Self {
            margin_index,

            asset_index: i16::try_from(pis[0].to_canonical_u64()).unwrap(),
            loan_to_value: pis[1].to_canonical_u64() as u16,
            liquidation_threshold: pis[2].to_canonical_u64() as u16,
            liquidation_factor: pis[3].to_canonical_u64() as u32,
            liquidation_fee: pis[4].to_canonical_u64() as u32,
            index_price: pis[5].to_canonical_u64() as i64,
            index_price_divider: pis[6].to_canonical_u64() as i64,
            global_supply_cap,
            user_supply_cap,
            total_supplied_amount,
        }
    }

    pub fn empty(margin_index: u8) -> Self {
        Self {
            margin_index,
            asset_index: 0,
            loan_to_value: 0,
            liquidation_threshold: 0,
            liquidation_factor: 0,
            liquidation_fee: 0,
            index_price: 0,
            index_price_divider: 0,
            global_supply_cap: BigUint::zero(),
            user_supply_cap: BigUint::zero(),
            total_supplied_amount: BigUint::zero(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MarginedAssetTarget {
    pub asset_index: Target,
    pub loan_to_value: Target,
    pub liquidation_threshold: Target,
    pub liquidation_factor: Target,
    pub liquidation_fee: Target,
    pub index_price: Target,
    pub index_price_divider: Target,
    pub global_supply_cap: BigUintTarget,
    pub user_supply_cap: BigUintTarget,
    pub total_supplied_amount: BigUintTarget,
}

impl MarginedAssetTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            asset_index: builder.add_virtual_target(),
            loan_to_value: builder.add_virtual_target(),
            liquidation_threshold: builder.add_virtual_target(),
            liquidation_factor: builder.add_virtual_target(),
            liquidation_fee: builder.add_virtual_target(),
            index_price: builder.add_virtual_target(),
            index_price_divider: builder.add_virtual_target(),
            global_supply_cap: builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS),
            user_supply_cap: builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS),
            total_supplied_amount: builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS),
        }
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert_eq!(pis.len(), MARGINED_ASSET_SIZE);

        Self {
            asset_index: pis[0],
            loan_to_value: pis[1],
            liquidation_threshold: pis[2],
            liquidation_factor: pis[3],
            liquidation_fee: pis[4],
            index_price: pis[5],
            index_price_divider: pis[6],
            global_supply_cap: BigUintTarget::from(&pis[7..10]),
            user_supply_cap: BigUintTarget::from(&pis[10..13]),
            total_supplied_amount: BigUintTarget::from(&pis[13..16]),
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // Adding these limbs and a bool does not overflow Goldilocks, as long as
        // limbs are guaranteed by business logic to fit 32 bits.
        let added = builder.add_many([
            self.asset_index,           // 16 bits
            self.loan_to_value,         // Max 10_000
            self.liquidation_threshold, // Max 10_000
            self.liquidation_factor,    // Max 1_000_000
            self.liquidation_fee,       // Max 1_000_000
            self.index_price,           // 32 bits
            self.index_price_divider,   // 56 bits
            self.global_supply_cap.limbs[0].0,
            self.global_supply_cap.limbs[1].0,
            self.global_supply_cap.limbs[2].0,
            self.user_supply_cap.limbs[0].0,
            self.user_supply_cap.limbs[1].0,
            self.user_supply_cap.limbs[2].0,
            self.total_supplied_amount.limbs[0].0,
            self.total_supplied_amount.limbs[1].0,
            self.total_supplied_amount.limbs[2].0,
        ]);
        builder.is_zero(added)
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.asset_index, &format!("{} asset_index", tag));
        builder.println(self.index_price, &format!("{} index_price", tag));
        builder.println(self.loan_to_value, &format!("{} loan_to_value", tag));
        builder.println(
            self.liquidation_threshold,
            &format!("{} liquidation_threshold", tag),
        );
        builder.println(
            self.liquidation_factor,
            &format!("{} liquidation_factor", tag),
        );
        builder.println(self.liquidation_fee, &format!("{} liquidation_fee", tag));
        builder.println(
            self.index_price_divider,
            &format!("{} index_price_divider", tag),
        );
        builder.println_biguint(
            &self.global_supply_cap,
            &format!("{} global_supply_cap", tag),
        );
        builder.println_biguint(&self.user_supply_cap, &format!("{} user_supply_cap", tag));
        builder.println_biguint(
            &self.total_supplied_amount,
            &format!("{} total_supplied_amount", tag),
        );
    }

    pub fn get_hash_parameters(&self) -> Vec<Target> {
        let mut elements = vec![
            self.asset_index,
            self.index_price,
            self.loan_to_value,
            self.liquidation_threshold,
            self.liquidation_factor,
            self.liquidation_fee,
            self.index_price_divider,
        ];

        [
            &self.global_supply_cap,
            &self.user_supply_cap,
            &self.total_supplied_amount,
        ]
        .iter()
        .for_each(|biguint_target| {
            let mut limbs = biguint_target.limbs.clone();
            limbs.resize(BIG_U96_LIMBS, U32Target::default());
            for limb in limbs {
                elements.push(limb.0);
            }
        });

        elements
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            asset_index: builder.zero(),
            loan_to_value: builder.zero(),
            liquidation_threshold: builder.zero(),
            liquidation_factor: builder.zero(),
            liquidation_fee: builder.zero(),
            index_price: builder.zero(),
            index_price_divider: builder.zero(),
            global_supply_cap: builder.zero_biguint(),
            user_supply_cap: builder.zero_biguint(),
            total_supplied_amount: builder.zero_biguint(),
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        let public_inputs_before = builder.num_public_inputs();

        builder.register_public_inputs(&[
            self.asset_index,
            self.loan_to_value,
            self.liquidation_threshold,
            self.liquidation_factor,
            self.liquidation_fee,
            self.index_price,
            self.index_price_divider,
        ]);
        builder.register_public_input_biguint(&self.global_supply_cap);
        builder.register_public_input_biguint(&self.user_supply_cap);
        builder.register_public_input_biguint(&self.total_supplied_amount);

        let public_inputs_after = builder.num_public_inputs();
        assert_eq!(
            public_inputs_after - public_inputs_before,
            MARGINED_ASSET_SIZE
        );
    }

    pub fn get_remaining_supply_cap(
        &self,
        builder: &mut Builder,
        user_margin_balance: &BigIntTarget,
    ) -> BigUintTarget {
        let zero = builder.zero_biguint();

        let (mut remaining_global_supply_cap, borrow) =
            builder.try_sub_biguint(&self.global_supply_cap, &self.total_supplied_amount);
        remaining_global_supply_cap = builder.select_biguint(
            BoolTarget::new_unsafe(borrow.0),
            &zero,
            &remaining_global_supply_cap,
        );

        let (mut remaining_user_supply_cap, borrow) =
            builder.try_sub_biguint(&self.user_supply_cap, &user_margin_balance.abs);
        let user_balance_negative = builder.is_sign_negative(user_margin_balance.sign);
        let borrow_or_user_balance_negative =
            builder.or(BoolTarget::new_unsafe(borrow.0), user_balance_negative);
        remaining_user_supply_cap = builder.select_biguint(
            borrow_or_user_balance_negative,
            &zero,
            &remaining_user_supply_cap,
        );

        builder.min_biguint(&remaining_global_supply_cap, &remaining_user_supply_cap)
    }

    pub fn partial_select_for_spot_trade(
        builder: &mut Builder,
        selector: BoolTarget,
        a: &Self,
        b: &Self,
    ) -> Self {
        Self {
            total_supplied_amount: builder.select_biguint(
                selector,
                &a.total_supplied_amount,
                &b.total_supplied_amount,
            ),
            global_supply_cap: builder.select_biguint(
                selector,
                &a.global_supply_cap,
                &b.global_supply_cap,
            ),
            user_supply_cap: builder.select_biguint(
                selector,
                &a.user_supply_cap,
                &b.user_supply_cap,
            ),
            ..Default::default()
        }
    }
}

pub fn random_access_margined_assets(
    builder: &mut Builder,
    access_index: Target,
    v: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
) -> MarginedAssetTarget {
    let mut vv = vec![];
    vv.extend_from_slice(v);
    vv.push(MarginedAssetTarget::empty(builder)); // append an empty asset for the case when we accessing nil index
    MarginedAssetTarget {
        asset_index: builder
            .random_access(access_index, vv.iter().map(|x| x.asset_index).collect()),
        loan_to_value: builder
            .random_access(access_index, vv.iter().map(|x| x.loan_to_value).collect()),
        liquidation_threshold: builder.random_access(
            access_index,
            vv.iter().map(|x| x.liquidation_threshold).collect(),
        ),
        liquidation_factor: builder.random_access(
            access_index,
            vv.iter().map(|x| x.liquidation_factor).collect(),
        ),
        liquidation_fee: builder
            .random_access(access_index, vv.iter().map(|x| x.liquidation_fee).collect()),
        index_price: builder
            .random_access(access_index, vv.iter().map(|x| x.index_price).collect()),
        index_price_divider: builder.random_access(
            access_index,
            vv.iter().map(|x| x.index_price_divider).collect(),
        ),
        global_supply_cap: builder.random_access_biguint(
            access_index,
            vv.iter().map(|x| x.global_supply_cap.clone()).collect(),
            BIG_U96_LIMBS,
        ),
        user_supply_cap: builder.random_access_biguint(
            access_index,
            vv.iter().map(|x| x.user_supply_cap.clone()).collect(),
            BIG_U96_LIMBS,
        ),
        total_supplied_amount: builder.random_access_biguint(
            access_index,
            vv.iter().map(|x| x.total_supplied_amount.clone()).collect(),
            BIG_U96_LIMBS,
        ),
    }
}

pub trait MarginedAssetTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_margined_asset_target(
        &mut self,
        a: &MarginedAssetTarget,
        b: &MarginedAsset,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    MarginedAssetTargetWitness<F> for T
{
    fn set_margined_asset_target(
        &mut self,
        a: &MarginedAssetTarget,
        b: &MarginedAsset,
    ) -> Result<()> {
        self.set_target(a.asset_index, F::from_canonical_u16(b.asset_index as u16))?;
        self.set_target(a.loan_to_value, F::from_canonical_u16(b.loan_to_value))?;
        self.set_target(
            a.liquidation_threshold,
            F::from_canonical_u16(b.liquidation_threshold),
        )?;
        self.set_target(
            a.liquidation_factor,
            F::from_canonical_u32(b.liquidation_factor),
        )?;
        self.set_target(a.liquidation_fee, F::from_canonical_u32(b.liquidation_fee))?;
        self.set_target(a.index_price, F::from_canonical_u64(b.index_price as u64))?;
        self.set_target(
            a.index_price_divider,
            F::from_canonical_u64(b.index_price_divider as u64),
        )?;
        self.set_biguint_target(&a.global_supply_cap, &b.global_supply_cap)?;
        self.set_biguint_target(&a.user_supply_cap, &b.user_supply_cap)?;
        self.set_biguint_target(&a.total_supplied_amount, &b.total_supplied_amount)?;

        Ok(())
    }
}

pub fn diff_margined_assets(
    builder: &mut Builder,
    new: &MarginedAssetTarget,
    old: &MarginedAssetTarget,
) -> MarginedAssetTarget {
    MarginedAssetTarget {
        asset_index: builder.sub(new.asset_index, old.asset_index),
        loan_to_value: builder.sub(new.loan_to_value, old.loan_to_value),
        liquidation_threshold: builder.sub(new.liquidation_threshold, old.liquidation_threshold),
        liquidation_factor: builder.sub(new.liquidation_factor, old.liquidation_factor),
        liquidation_fee: builder.sub(new.liquidation_fee, old.liquidation_fee),
        index_price: builder.sub(new.index_price, old.index_price),
        index_price_divider: builder.sub(new.index_price_divider, old.index_price_divider),
        global_supply_cap: builder
            .biguint_vector_diff(&new.global_supply_cap, &old.global_supply_cap),
        user_supply_cap: builder.biguint_vector_diff(&new.user_supply_cap, &old.user_supply_cap),
        total_supplied_amount: builder
            .biguint_vector_diff(&new.total_supplied_amount, &old.total_supplied_amount),
    }
}

pub fn apply_diff_margined_assets(
    builder: &mut Builder,
    flag: BoolTarget,
    diff: &MarginedAssetTarget,
    old: &MarginedAssetTarget,
) -> MarginedAssetTarget {
    MarginedAssetTarget {
        asset_index: builder.mul_add(flag.target, diff.asset_index, old.asset_index),
        loan_to_value: builder.mul_add(flag.target, diff.loan_to_value, old.loan_to_value),
        liquidation_threshold: builder.mul_add(
            flag.target,
            diff.liquidation_threshold,
            old.liquidation_threshold,
        ),
        liquidation_factor: builder.mul_add(
            flag.target,
            diff.liquidation_factor,
            old.liquidation_factor,
        ),
        liquidation_fee: builder.mul_add(flag.target, diff.liquidation_fee, old.liquidation_fee),
        index_price: builder.mul_add(flag.target, diff.index_price, old.index_price),
        index_price_divider: builder.mul_add(
            flag.target,
            diff.index_price_divider,
            old.index_price_divider,
        ),
        global_supply_cap: builder.biguint_vector_sum(
            flag,
            &diff.global_supply_cap,
            &old.global_supply_cap,
        ),
        user_supply_cap: builder.biguint_vector_sum(
            flag,
            &diff.user_supply_cap,
            &old.user_supply_cap,
        ),
        total_supplied_amount: builder.biguint_vector_sum(
            flag,
            &diff.total_supplied_amount,
            &old.total_supplied_amount,
        ),
    }
}

pub fn connect_margined_assets(
    builder: &mut Builder,
    a: &MarginedAssetTarget,
    b: &MarginedAssetTarget,
) {
    builder.connect(a.asset_index, b.asset_index);
    builder.connect(a.loan_to_value, b.loan_to_value);
    builder.connect(a.liquidation_threshold, b.liquidation_threshold);
    builder.connect(a.liquidation_factor, b.liquidation_factor);
    builder.connect(a.liquidation_fee, b.liquidation_fee);
    builder.connect(a.index_price, b.index_price);
    builder.connect(a.index_price_divider, b.index_price_divider);
    builder.connect_biguint(&a.global_supply_cap, &b.global_supply_cap);
    builder.connect_biguint(&a.user_supply_cap, &b.user_supply_cap);
    builder.connect_biguint(&a.total_supplied_amount, &b.total_supplied_amount);
}

pub fn all_margined_assets_hash(
    builder: &mut Builder,
    assets: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
) -> HashOutTarget {
    let mut elements = vec![];
    assets.iter().for_each(|asset| {
        elements.extend_from_slice(&asset.get_hash_parameters());
    });
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
}

pub fn select_margined_asset_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &MarginedAssetTarget,
    b: &MarginedAssetTarget,
) -> MarginedAssetTarget {
    MarginedAssetTarget {
        asset_index: builder.select(flag, a.asset_index, b.asset_index),
        loan_to_value: builder.select(flag, a.loan_to_value, b.loan_to_value),
        liquidation_threshold: builder.select(
            flag,
            a.liquidation_threshold,
            b.liquidation_threshold,
        ),
        liquidation_factor: builder.select(flag, a.liquidation_factor, b.liquidation_factor),
        liquidation_fee: builder.select(flag, a.liquidation_fee, b.liquidation_fee),
        index_price: builder.select(flag, a.index_price, b.index_price),
        index_price_divider: builder.select(flag, a.index_price_divider, b.index_price_divider),
        global_supply_cap: builder.select_biguint(flag, &a.global_supply_cap, &b.global_supply_cap),
        user_supply_cap: builder.select_biguint(flag, &a.user_supply_cap, &b.user_supply_cap),
        total_supplied_amount: builder.select_biguint(
            flag,
            &a.total_supplied_amount,
            &b.total_supplied_amount,
        ),
    }
}
