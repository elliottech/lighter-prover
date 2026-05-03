// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::{BigUint, FromPrimitive};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::BIG_U64_LIMBS;
use crate::types::constants::*;
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::utils::CircuitBuilderUtils;

pub const ASSET_SIZE: usize = 8;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
#[serde(default)]
pub struct Asset {
    #[serde(rename = "ai")]
    pub asset_index: i16,
    #[serde(rename = "em")]
    pub extension_multiplier: i64, // 56 bits
    #[serde(rename = "mta")]
    pub min_transfer_amount: i64, // 60 bits
    #[serde(rename = "mwa")]
    pub min_withdrawal_amount: i64, // 60 bits
    #[serde(rename = "mm", default)]
    pub margin_mode: u8,
    #[serde(rename = "mi", default)]
    pub margin_index: u8,
}

impl Asset {
    pub fn from_public_inputs<F>(asset_index: i16, pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert_eq!(pis.len(), ASSET_SIZE);

        Self {
            asset_index,

            margin_mode: u8::try_from(pis[0].to_canonical_u64()).unwrap(),
            extension_multiplier: (pis[2].to_canonical_u64() << 32 | pis[1].to_canonical_u64())
                as i64,
            min_transfer_amount: (pis[4].to_canonical_u64() << 32 | pis[3].to_canonical_u64())
                as i64,
            min_withdrawal_amount: (pis[6].to_canonical_u64() << 32 | pis[5].to_canonical_u64())
                as i64,
            margin_index: u8::try_from(pis[7].to_canonical_u64()).unwrap(),
        }
    }

    pub fn empty(asset_index: i16) -> Self {
        Self {
            asset_index,
            extension_multiplier: 0,
            min_transfer_amount: 0,
            min_withdrawal_amount: 0,
            margin_mode: 0,
            margin_index: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AssetTarget {
    pub extension_multiplier: BigUintTarget,
    pub min_transfer_amount: BigUintTarget,
    pub min_withdrawal_amount: BigUintTarget,
    pub margin_mode: Target,
    pub margin_index: Target,
}

impl AssetTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            margin_mode: builder.add_virtual_target(),
            extension_multiplier: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_transfer_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_withdrawal_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            margin_index: builder.add_virtual_target(),
        }
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert_eq!(pis.len(), ASSET_SIZE);

        Self {
            margin_mode: pis[0],
            extension_multiplier: BigUintTarget {
                limbs: vec![U32Target(pis[1]), U32Target(pis[2])],
            },
            min_transfer_amount: BigUintTarget {
                limbs: vec![U32Target(pis[3]), U32Target(pis[4])],
            },
            min_withdrawal_amount: BigUintTarget {
                limbs: vec![U32Target(pis[5]), U32Target(pis[6])],
            },
            margin_index: pis[7],
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // Adding 6 u32 limbs and a bool does not overflow Goldilocks, as long as
        // limbs are guaranteed by business logic to fit 32 bits.
        let added = builder.add_many(
            [
                &self.extension_multiplier,
                &self.min_transfer_amount,
                &self.min_withdrawal_amount,
            ]
            .iter()
            .flat_map(|x| x.limbs.iter().map(|limb| limb.0))
            .chain([self.margin_mode, self.margin_index])
            .collect::<Vec<_>>(),
        );
        builder.is_zero(added)
    }

    pub fn margin_index(&self, builder: &mut Builder) -> Target {
        let nil_margined_asset_index = builder.constant_u64(MARGINED_ASSET_LIST_SIZE as u64);
        builder.select(
            BoolTarget::new_unsafe(self.margin_mode), // 0 disabled, 1 enabled
            self.margin_index,
            nil_margined_asset_index,
        )
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_biguint(
            &self.extension_multiplier,
            &format!("{} extension_multiplier", tag),
        );
        builder.println_biguint(
            &self.min_transfer_amount,
            &format!("{} min_transfer_amount", tag),
        );
        builder.println_biguint(
            &self.min_withdrawal_amount,
            &format!("{} min_withdrawal_amount", tag),
        );
        builder.println(self.margin_mode, &format!("{} margin_mode", tag));
        builder.println(self.margin_index, &format!("{} margin_index", tag));
    }

    pub fn get_hash_parameters(&self) -> Vec<Target> {
        let mut elements = vec![self.margin_index, self.margin_mode];

        [
            &self.extension_multiplier,
            &self.min_transfer_amount,
            &self.min_withdrawal_amount,
        ]
        .iter()
        .for_each(|biguint_target| {
            let mut limbs = biguint_target.limbs.clone();
            limbs.resize(BIG_U64_LIMBS, U32Target::default());
            for limb in limbs {
                elements.push(limb.0);
            }
        });

        elements
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            margin_mode: builder.zero(),
            margin_index: builder.zero(),
            extension_multiplier: builder.zero_biguint(),
            min_transfer_amount: builder.zero_biguint(),
            min_withdrawal_amount: builder.zero_biguint(),
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        let public_inputs_before = builder.num_public_inputs();

        builder.register_public_input(self.margin_mode);
        builder.register_public_input_biguint(&self.extension_multiplier);
        builder.register_public_input_biguint(&self.min_transfer_amount);
        builder.register_public_input_biguint(&self.min_withdrawal_amount);
        builder.register_public_input(self.margin_index);

        let public_inputs_after = builder.num_public_inputs();
        assert_eq!(public_inputs_after - public_inputs_before, ASSET_SIZE);
    }
}

pub fn random_access_assets(
    builder: &mut Builder,
    access_index: Target,
    v: Vec<AssetTarget>,
) -> AssetTarget {
    assert!(v.len() % 64 == 0);
    AssetTarget {
        extension_multiplier: builder.random_access_biguint(
            access_index,
            v.iter().map(|x| x.extension_multiplier.clone()).collect(),
            BIG_U64_LIMBS,
        ),
        min_transfer_amount: builder.random_access_biguint(
            access_index,
            v.iter().map(|x| x.min_transfer_amount.clone()).collect(),
            BIG_U64_LIMBS,
        ),
        min_withdrawal_amount: builder.random_access_biguint(
            access_index,
            v.iter().map(|x| x.min_withdrawal_amount.clone()).collect(),
            BIG_U64_LIMBS,
        ),
        margin_mode: builder.random_access(access_index, v.iter().map(|x| x.margin_mode).collect()),
        margin_index: builder
            .random_access(access_index, v.iter().map(|x| x.margin_index).collect()),
    }
}

pub trait AssetTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_asset_target(&mut self, a: &AssetTarget, b: &Asset) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AssetTargetWitness<F> for T
{
    fn set_asset_target(&mut self, a: &AssetTarget, b: &Asset) -> Result<()> {
        self.set_biguint_target(
            &a.extension_multiplier,
            &BigUint::from_u64(b.extension_multiplier as u64).unwrap(),
        )?;
        self.set_biguint_target(
            &a.min_transfer_amount,
            &BigUint::from_u64(b.min_transfer_amount as u64).unwrap(),
        )?;
        self.set_biguint_target(
            &a.min_withdrawal_amount,
            &BigUint::from_u64(b.min_withdrawal_amount as u64).unwrap(),
        )?;
        self.set_target(a.margin_mode, F::from_canonical_u8(b.margin_mode))?;
        self.set_target(a.margin_index, F::from_canonical_u8(b.margin_index))?;

        Ok(())
    }
}

pub fn diff_assets(builder: &mut Builder, new: &AssetTarget, old: &AssetTarget) -> AssetTarget {
    AssetTarget {
        extension_multiplier: builder
            .biguint_vector_diff(&new.extension_multiplier, &old.extension_multiplier),
        min_transfer_amount: builder
            .biguint_vector_diff(&new.min_transfer_amount, &old.min_transfer_amount),
        min_withdrawal_amount: builder
            .biguint_vector_diff(&new.min_withdrawal_amount, &old.min_withdrawal_amount),
        margin_mode: builder.sub(new.margin_mode, old.margin_mode),
        margin_index: builder.sub(new.margin_index, old.margin_index),
    }
}

pub fn apply_diff_assets(
    builder: &mut Builder,
    flag: BoolTarget,
    diff: &AssetTarget,
    old: &AssetTarget,
) -> AssetTarget {
    AssetTarget {
        extension_multiplier: builder.biguint_vector_sum(
            flag,
            &diff.extension_multiplier,
            &old.extension_multiplier,
        ),
        min_transfer_amount: builder.biguint_vector_sum(
            flag,
            &diff.min_transfer_amount,
            &old.min_transfer_amount,
        ),
        min_withdrawal_amount: builder.biguint_vector_sum(
            flag,
            &diff.min_withdrawal_amount,
            &old.min_withdrawal_amount,
        ),
        margin_mode: builder.mul_add(flag.target, diff.margin_mode, old.margin_mode),
        margin_index: builder.mul_add(flag.target, diff.margin_index, old.margin_index),
    }
}

pub fn connect_assets(builder: &mut Builder, a: &AssetTarget, b: &AssetTarget) {
    builder.connect_biguint(&a.extension_multiplier, &b.extension_multiplier);
    builder.connect_biguint(&a.min_transfer_amount, &b.min_transfer_amount);
    builder.connect_biguint(&a.min_withdrawal_amount, &b.min_withdrawal_amount);
    builder.connect(a.margin_mode, b.margin_mode);
    builder.connect(a.margin_index, b.margin_index);
}

pub fn all_assets_hash(
    builder: &mut Builder,
    assets: &[AssetTarget; ASSET_LIST_SIZE],
) -> HashOutTarget {
    let mut elements = vec![];
    for i in MIN_ASSET_INDEX..=MAX_ASSET_INDEX {
        elements.extend_from_slice(&assets[i as usize].get_hash_parameters());
    }
    builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
}

pub fn select_asset_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &AssetTarget,
    b: &AssetTarget,
) -> AssetTarget {
    AssetTarget {
        extension_multiplier: builder.select_biguint(
            flag,
            &a.extension_multiplier,
            &b.extension_multiplier,
        ),
        min_transfer_amount: builder.select_biguint(
            flag,
            &a.min_transfer_amount,
            &b.min_transfer_amount,
        ),
        min_withdrawal_amount: builder.select_biguint(
            flag,
            &a.min_withdrawal_amount,
            &b.min_withdrawal_amount,
        ),
        margin_mode: builder.select(flag, a.margin_mode, b.margin_mode),
        margin_index: builder.select(flag, a.margin_index, b.margin_index),
    }
}

pub fn ensure_valid_asset_index(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    asset_index: Target,
) {
    let nil_asset_index = builder.constant_u64(NIL_ASSET_INDEX);
    builder.conditional_assert_not_eq(is_enabled, asset_index, nil_asset_index);
}

pub fn is_universal_asset(builder: &mut Builder, asset_index: Target) -> BoolTarget {
    let assertions = [builder.is_equal_constant(asset_index, USDC_ASSET_INDEX)];
    builder.multi_or(&assertions)
}
