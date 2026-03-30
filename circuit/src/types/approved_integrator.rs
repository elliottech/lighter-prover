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
use crate::circuit_logger::CircuitBuilderLogging;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "", default)]
pub struct ApprovedIntegrator {
    #[serde(rename = "aiw_iai")]
    pub integrator_account_index: i64,
    #[serde(rename = "aiw_mptf")]
    pub max_perps_taker_fee: u32,
    #[serde(rename = "aiw_mpmf")]
    pub max_perps_maker_fee: u32,
    #[serde(rename = "aiw_mstf")]
    pub max_spot_taker_fee: u32,
    #[serde(rename = "aiw_msmf")]
    pub max_spot_maker_fee: u32,
    #[serde(rename = "aiw_exp")]
    pub expiry: i64,
}

#[derive(Debug, Clone, Default, Copy)]
pub struct ApprovedIntegratorTarget {
    pub integrator_account_index: Target,
    pub max_perps_taker_fee: Target,
    pub max_perps_maker_fee: Target,
    pub max_spot_taker_fee: Target,
    pub max_spot_maker_fee: Target,
    pub expiry: Target,
}

impl ApprovedIntegratorTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            integrator_account_index: builder.add_virtual_target(),
            max_perps_taker_fee: builder.add_virtual_target(),
            max_perps_maker_fee: builder.add_virtual_target(),
            max_spot_taker_fee: builder.add_virtual_target(),
            max_spot_maker_fee: builder.add_virtual_target(),
            expiry: builder.add_virtual_target(),
        }
    }
    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.integrator_account_index,
            &format!("{}: integrator_account_index", tag),
        );
        builder.println(
            self.max_perps_taker_fee,
            &format!("{}: max_perps_taker_fee", tag),
        );
        builder.println(
            self.max_perps_maker_fee,
            &format!("{}: max_perps_maker_fee", tag),
        );
        builder.println(
            self.max_spot_taker_fee,
            &format!("{}: max_spot_taker_fee", tag),
        );
        builder.println(
            self.max_spot_maker_fee,
            &format!("{}: max_spot_maker_fee", tag),
        );
        builder.println(self.expiry, &format!("{}: expiry", tag));
    }
    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            integrator_account_index: builder.zero(),
            max_perps_taker_fee: builder.zero(),
            max_perps_maker_fee: builder.zero(),
            max_spot_taker_fee: builder.zero(),
            max_spot_maker_fee: builder.zero(),
            expiry: builder.zero(),
        }
    }
    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        builder.is_zero(self.expiry)
    }
}

pub fn select_approved_integrator_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &ApprovedIntegratorTarget,
    b: &ApprovedIntegratorTarget,
) -> ApprovedIntegratorTarget {
    ApprovedIntegratorTarget {
        integrator_account_index: builder.select(
            flag,
            a.integrator_account_index,
            b.integrator_account_index,
        ),
        max_perps_taker_fee: builder.select(flag, a.max_perps_taker_fee, b.max_perps_taker_fee),
        max_perps_maker_fee: builder.select(flag, a.max_perps_maker_fee, b.max_perps_maker_fee),
        max_spot_taker_fee: builder.select(flag, a.max_spot_taker_fee, b.max_spot_taker_fee),
        max_spot_maker_fee: builder.select(flag, a.max_spot_maker_fee, b.max_spot_maker_fee),
        expiry: builder.select(flag, a.expiry, b.expiry),
    }
}

pub trait ApprovedIntegratorWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_approved_integrator(
        &mut self,
        a: &ApprovedIntegratorTarget,
        b: &ApprovedIntegrator,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> ApprovedIntegratorWitness<F>
    for T
{
    fn set_approved_integrator(
        &mut self,
        a: &ApprovedIntegratorTarget,
        b: &ApprovedIntegrator,
    ) -> Result<()> {
        self.set_target(
            a.integrator_account_index,
            F::from_canonical_i64(b.integrator_account_index),
        )?;
        self.set_target(
            a.max_perps_taker_fee,
            F::from_canonical_u32(b.max_perps_taker_fee),
        )?;
        self.set_target(
            a.max_perps_maker_fee,
            F::from_canonical_u32(b.max_perps_maker_fee),
        )?;
        self.set_target(
            a.max_spot_taker_fee,
            F::from_canonical_u32(b.max_spot_taker_fee),
        )?;
        self.set_target(
            a.max_spot_maker_fee,
            F::from_canonical_u32(b.max_spot_maker_fee),
        )?;
        self.set_target(a.expiry, F::from_canonical_i64(b.expiry))?;

        Ok(())
    }
}
