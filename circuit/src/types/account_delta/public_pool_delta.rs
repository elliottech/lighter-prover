// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget, WitnessSigned};
use crate::types::config::Builder;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(bound = "", default)]
pub struct PublicPoolShareDelta {
    #[serde(rename = "ppi", default)]
    pub public_pool_index: i64,

    #[serde(rename = "sd", default)]
    pub shares_delta: i64, // value is in range [0, 2^56 - 1], thus the diff is in range [-2^56 + 1, 2^56 - 1]
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct PublicPoolShareDeltaTarget {
    pub public_pool_index: Target,
    pub shares_delta: SignedTarget,
}

impl PublicPoolShareDeltaTarget {
    pub fn new(builder: &mut Builder) -> Self {
        PublicPoolShareDeltaTarget {
            public_pool_index: builder.add_virtual_target(),
            shares_delta: builder.add_virtual_signed_target(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.public_pool_index,
            &format!("{}: public_pool_index", tag),
        );
        builder.println(self.shares_delta.target, &format!("{}: shares_delta", tag));
    }

    pub fn empty(builder: &mut Builder, public_pool_index: Target) -> Self {
        PublicPoolShareDeltaTarget {
            public_pool_index,
            shares_delta: builder.zero_signed(),
        }
    }

    pub fn is_empty_without_metadata(&self, builder: &mut Builder) -> BoolTarget {
        builder.is_zero(self.shares_delta.target)
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
pub struct PublicPoolInfoDelta {
    #[serde(rename = "tsd", default)]
    pub total_shares_delta: i64, // value is in range [0, 2^56 - 1], thus the diff is in range [-2^56 + 1, 2^56 - 1]

    #[serde(rename = "osd", default)]
    pub operator_shares_delta: i64, // value is in range [0, 2^56 - 1], thus the diff is in range [-2^56 + 1, 2^56 - 1]
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PublicPoolInfoDeltaTarget {
    pub total_shares_delta: SignedTarget,
    pub operator_shares_delta: SignedTarget,
}

impl PublicPoolInfoDeltaTarget {
    pub fn new(builder: &mut Builder) -> Self {
        PublicPoolInfoDeltaTarget {
            total_shares_delta: builder.add_virtual_signed_target(),
            operator_shares_delta: builder.add_virtual_signed_target(),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        PublicPoolInfoDeltaTarget {
            total_shares_delta: builder.zero_signed(),
            operator_shares_delta: builder.zero_signed(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_signed_target(
            self.total_shares_delta,
            &format!("{}: total_shares_delta", tag),
        );
        builder.println_signed_target(
            self.operator_shares_delta,
            &format!("{}: operator_shares_delta", tag),
        );
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero(self.total_shares_delta.target),
            builder.is_zero(self.operator_shares_delta.target),
        ];

        builder.multi_and(&assertions)
    }
}

pub fn select_public_pool_info_delta_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &PublicPoolInfoDeltaTarget,
    b: &PublicPoolInfoDeltaTarget,
) -> PublicPoolInfoDeltaTarget {
    PublicPoolInfoDeltaTarget {
        total_shares_delta: builder.select_signed(flag, a.total_shares_delta, b.total_shares_delta),
        operator_shares_delta: builder.select_signed(
            flag,
            a.operator_shares_delta,
            b.operator_shares_delta,
        ),
    }
}

pub fn select_public_pool_share_delta_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &PublicPoolShareDeltaTarget,
    b: &PublicPoolShareDeltaTarget,
) -> PublicPoolShareDeltaTarget {
    PublicPoolShareDeltaTarget {
        public_pool_index: builder.select(flag, a.public_pool_index, b.public_pool_index),
        shares_delta: builder.select_signed(flag, a.shares_delta, b.shares_delta),
    }
}

pub trait PublicPoolShareDeltaWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_public_pool_share_delta(
        &mut self,
        a: &PublicPoolShareDeltaTarget,
        b: &PublicPoolShareDelta,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PublicPoolShareDeltaWitness<F>
    for T
{
    fn set_public_pool_share_delta(
        &mut self,
        a: &PublicPoolShareDeltaTarget,
        b: &PublicPoolShareDelta,
    ) -> Result<()> {
        self.set_target(
            a.public_pool_index,
            F::from_canonical_i64(b.public_pool_index),
        )?;
        self.set_signed_target(a.shares_delta, b.shares_delta)?;

        Ok(())
    }
}

pub trait PublicPoolInfoDeltaWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_public_pool_info_delta(
        &mut self,
        a: &PublicPoolInfoDeltaTarget,
        b: &PublicPoolInfoDelta,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PublicPoolInfoDeltaWitness<F>
    for T
{
    fn set_public_pool_info_delta(
        &mut self,
        a: &PublicPoolInfoDeltaTarget,
        b: &PublicPoolInfoDelta,
    ) -> Result<()> {
        self.set_signed_target(a.total_shares_delta, b.total_shares_delta)?;
        self.set_signed_target(a.operator_shares_delta, b.operator_shares_delta)?;
        Ok(())
    }
}
