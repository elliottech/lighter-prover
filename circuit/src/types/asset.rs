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
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::BIG_U64_LIMBS;
use crate::types::constants::{ASSET_LIST_SIZE, MAX_ASSET_INDEX, MIN_ASSET_INDEX};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};
use crate::utils::CircuitBuilderUtils;

pub const ASSET_SIZE: usize = 7;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
#[serde(default)]
pub struct Asset {
    #[serde(rename = "i")]
    pub asset_index: i16,
    #[serde(rename = "em")]
    pub extension_multiplier: i64, // 56 bits
    #[serde(rename = "mta")]
    pub min_transfer_amount: i64, // 60 bits
    #[serde(rename = "mwa")]
    pub min_withdrawal_amount: i64, // 60 bits
    #[serde(rename = "mm", default)]
    pub margin_mode: u8,
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
        }
    }

    pub fn empty(asset_index: i16) -> Self {
        Self {
            asset_index,
            extension_multiplier: 0,
            min_transfer_amount: 0,
            min_withdrawal_amount: 0,
            margin_mode: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AssetTarget {
    pub extension_multiplier: BigUintTarget,
    pub min_transfer_amount: BigUintTarget,
    pub min_withdrawal_amount: BigUintTarget,
    pub margin_mode: Target,
}

impl AssetTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            margin_mode: builder.add_virtual_target(),
            extension_multiplier: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_transfer_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
            min_withdrawal_amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS),
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
            .chain([self.margin_mode])
            .collect::<Vec<_>>(),
        );
        builder.is_zero(added)
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
    }

    pub fn get_hash_parameters(&self) -> Vec<Target> {
        let mut elements = vec![self.margin_mode];

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

        let public_inputs_after = builder.num_public_inputs();
        assert_eq!(public_inputs_after - public_inputs_before, ASSET_SIZE);
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let mut elements = vec![self.margin_mode];

        [
            &self.extension_multiplier,
            &self.min_transfer_amount,
            &self.min_withdrawal_amount,
        ]
        .iter()
        .for_each(|biguint_target| {
            let mut limbs = biguint_target.limbs.clone();
            limbs.resize(BIG_U64_LIMBS, builder.zero_u32());
            for limb in limbs {
                elements.push(limb.0);
            }
        });

        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let empty_hash = builder.zero_hash_out();
        let is_empty = self.is_empty(builder);

        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
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
    }
}

pub fn connect_assets(builder: &mut Builder, a: &AssetTarget, b: &AssetTarget) {
    builder.connect_biguint(&a.extension_multiplier, &b.extension_multiplier);
    builder.connect_biguint(&a.min_transfer_amount, &b.min_transfer_amount);
    builder.connect_biguint(&a.min_withdrawal_amount, &b.min_withdrawal_amount);
    builder.connect(a.margin_mode, b.margin_mode);
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
    }
}

// Caller's responsibility to ensure asset_index is in range [0, 2^6-1]
pub fn ensure_valid_asset_index(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    asset_index: Target,
) {
    let assertions = [
        builder.is_equal_constant(asset_index, MIN_ASSET_INDEX - 1),
        builder.is_equal_constant(asset_index, MAX_ASSET_INDEX + 1),
    ];
    let is_invalid = builder.multi_or(&assertions);
    builder.conditional_assert_false(is_enabled, is_invalid);
}
