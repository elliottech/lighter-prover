// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(bound = "", default)]
pub struct PublicPoolShare {
    #[serde(rename = "ppi")]
    pub public_pool_index: i64,

    #[serde(rename = "sa")]
    pub share_amount: i64,

    #[serde(rename = "eu")]
    pub principal_amount: i64,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
pub struct PublicPoolInfo {
    #[serde(rename = "ppi_s", default)]
    pub status: u8,

    #[serde(rename = "ppi_of", default)]
    pub operator_fee: i64,

    #[serde(rename = "ppi_mosr", default)]
    pub min_operator_share_rate: i64,

    #[serde(rename = "ppi_tsa", default)]
    pub total_shares: i64,

    #[serde(rename = "ppi_os", default)]
    pub operator_shares: i64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PublicPoolShareTarget {
    pub public_pool_index: Target,
    pub share_amount: Target,
    pub principal_amount: Target,
}

const PUBLIC_POOL_SHARE_SIZE: usize = 3;
impl PublicPoolShareTarget {
    pub fn new(builder: &mut Builder) -> Self {
        PublicPoolShareTarget {
            public_pool_index: builder.add_virtual_target(),
            share_amount: builder.add_virtual_target(),
            principal_amount: builder.add_virtual_target(),
        }
    }
    pub fn new_public(builder: &mut Builder) -> Self {
        PublicPoolShareTarget {
            public_pool_index: builder.add_virtual_public_input(),
            share_amount: builder.add_virtual_public_input(),
            principal_amount: builder.add_virtual_public_input(),
        }
    }
    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert!(pis.len() == PUBLIC_POOL_SHARE_SIZE);
        PublicPoolShareTarget {
            public_pool_index: pis[0],
            share_amount: pis[1],
            principal_amount: pis[2],
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.public_pool_index,
            &format!("{}: public_pool_index", tag),
        );
        builder.println(self.share_amount, &format!("{}: share_amount", tag));
        builder.println(self.principal_amount, &format!("{}: entry_usdc", tag));
    }

    pub fn empty(builder: &mut Builder, public_pool_index: Target) -> Self {
        PublicPoolShareTarget {
            public_pool_index,
            share_amount: builder.zero(),
            principal_amount: builder.zero(),
        }
    }

    pub fn is_empty_without_metadata(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero(self.share_amount),
            builder.is_zero(self.principal_amount),
        ];

        builder.multi_and(&assertions)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PublicPoolInfoTarget {
    pub status: Target,
    pub operator_fee: Target,
    pub min_operator_share_rate: Target,
    pub total_shares: Target,
    pub operator_shares: Target,
}

impl PublicPoolInfoTarget {
    pub fn new(builder: &mut Builder) -> Self {
        PublicPoolInfoTarget {
            status: builder.add_virtual_target(),
            operator_fee: builder.add_virtual_target(),
            min_operator_share_rate: builder.add_virtual_target(),
            total_shares: builder.add_virtual_target(),
            operator_shares: builder.add_virtual_target(),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        PublicPoolInfoTarget {
            status: builder.zero(),
            operator_fee: builder.zero(),
            min_operator_share_rate: builder.zero(),
            total_shares: builder.zero(),
            operator_shares: builder.zero(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.status, &format!("{}: status", tag));
        builder.println(self.operator_fee, &format!("{}: operator_fee", tag));
        builder.println(
            self.min_operator_share_rate,
            &format!("{}: min_operator_share_rate", tag),
        );
        builder.println(self.total_shares, &format!("{}: total_shares", tag));
        builder.println(self.operator_shares, &format!("{}: operator_shares", tag));
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero(self.status),
            builder.is_zero(self.operator_fee),
            builder.is_zero(self.min_operator_share_rate),
            builder.is_zero(self.total_shares),
            builder.is_zero(self.operator_shares),
        ];

        builder.multi_and(&assertions)
    }
}

pub fn select_public_pool_info_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &PublicPoolInfoTarget,
    b: &PublicPoolInfoTarget,
) -> PublicPoolInfoTarget {
    PublicPoolInfoTarget {
        status: builder.select(flag, a.status, b.status),
        operator_fee: builder.select(flag, a.operator_fee, b.operator_fee),
        min_operator_share_rate: builder.select(
            flag,
            a.min_operator_share_rate,
            b.min_operator_share_rate,
        ),
        total_shares: builder.select(flag, a.total_shares, b.total_shares),
        operator_shares: builder.select(flag, a.operator_shares, b.operator_shares),
    }
}

pub fn select_public_pool_share_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &PublicPoolShareTarget,
    b: &PublicPoolShareTarget,
) -> PublicPoolShareTarget {
    PublicPoolShareTarget {
        public_pool_index: builder.select(flag, a.public_pool_index, b.public_pool_index),
        share_amount: builder.select(flag, a.share_amount, b.share_amount),
        principal_amount: builder.select(flag, a.principal_amount, b.principal_amount),
    }
}

pub trait PublicPoolShareWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_public_pool_share(
        &mut self,
        a: &PublicPoolShareTarget,
        b: &PublicPoolShare,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PublicPoolShareWitness<F> for T {
    fn set_public_pool_share(
        &mut self,
        a: &PublicPoolShareTarget,
        b: &PublicPoolShare,
    ) -> Result<()> {
        self.set_target(
            a.public_pool_index,
            F::from_canonical_i64(b.public_pool_index),
        )?;
        self.set_target(a.share_amount, F::from_canonical_i64(b.share_amount))?;
        self.set_target(
            a.principal_amount,
            F::from_canonical_i64(b.principal_amount),
        )?;

        Ok(())
    }
}

pub trait PublicPoolInfoWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_public_pool_info(&mut self, a: &PublicPoolInfoTarget, b: &PublicPoolInfo) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PublicPoolInfoWitness<F> for T {
    fn set_public_pool_info(&mut self, a: &PublicPoolInfoTarget, b: &PublicPoolInfo) -> Result<()> {
        self.set_target(a.status, F::from_canonical_u8(b.status))?;
        self.set_target(a.operator_fee, F::from_canonical_i64(b.operator_fee))?;
        self.set_target(
            a.min_operator_share_rate,
            F::from_canonical_i64(b.min_operator_share_rate),
        )?;
        self.set_target(a.total_shares, F::from_canonical_i64(b.total_shares))?;
        self.set_target(a.operator_shares, F::from_canonical_i64(b.operator_shares))?;

        Ok(())
    }
}
