// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::config::Builder;
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::deserializers;
use crate::types::config::BIG_U96_LIMBS;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "", default)]
pub struct PendingUnlock {
    #[serde(rename = "apw_rt")]
    pub unlock_timestamp: i64,
    #[serde(rename = "apw_asi")]
    pub asset_index: i64,
    #[serde(rename = "apw_amt")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub amount: BigUint,
}

#[derive(Debug, Clone, Default)]
pub struct PendingUnlockTarget {
    pub unlock_timestamp: Target,
    pub asset_index: Target,
    pub amount: BigUintTarget,
}

impl PendingUnlockTarget {
    pub fn new(builder: &mut Builder) -> Self {
        PendingUnlockTarget {
            unlock_timestamp: builder.add_virtual_target(),
            asset_index: builder.add_virtual_target(),
            amount: builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS),
        }
    }
    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.unlock_timestamp, &format!("{}: unlock_timestamp", tag));
        builder.println_biguint(&self.amount, &format!("{}: amount", tag));
        builder.println(self.asset_index, &format!("{}: asset_index", tag));
    }
    pub fn empty(builder: &mut Builder) -> Self {
        PendingUnlockTarget {
            unlock_timestamp: builder.zero(),
            asset_index: builder.zero(),
            amount: builder.zero_biguint(),
        }
    }
}

pub fn select_pending_unlock_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &PendingUnlockTarget,
    b: &PendingUnlockTarget,
) -> PendingUnlockTarget {
    PendingUnlockTarget {
        unlock_timestamp: builder.select(flag, a.unlock_timestamp, b.unlock_timestamp),
        asset_index: builder.select(flag, a.asset_index, b.asset_index),
        amount: builder.select_biguint(flag, &a.amount, &b.amount),
    }
}

pub trait PendingUnlockWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_pending_unlock(&mut self, a: &PendingUnlockTarget, b: &PendingUnlock) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField> PendingUnlockWitness<F> for T {
    fn set_pending_unlock(&mut self, a: &PendingUnlockTarget, b: &PendingUnlock) -> Result<()> {
        self.set_target(
            a.unlock_timestamp,
            F::from_canonical_i64(b.unlock_timestamp),
        )?;
        self.set_target(a.asset_index, F::from_canonical_i64(b.asset_index))?;
        self.set_biguint_target(&a.amount, &b.amount)?;

        Ok(())
    }
}
