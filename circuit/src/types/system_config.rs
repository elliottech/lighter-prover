// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::hash::poseidon2::hash::Poseidon2Hash;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::types::config::Builder;
use crate::utils::CircuitBuilderUtils;

pub const SYSTEM_CONFIG_SIZE: usize = 4;

#[derive(Clone, Debug, Deserialize, Copy)]
#[serde(default)]
pub struct SystemConfig {
    #[serde(rename = "llpai")]
    pub liquidity_pool_index: i64,
    #[serde(rename = "lspai")]
    pub staking_pool_index: i64,
    #[serde(rename = "mbps")]
    pub liquidity_pool_cooldown_period: i64,
    #[serde(rename = "spwlm")]
    pub staking_pool_lockup_period: i64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self::empty()
    }
}

impl SystemConfig {
    pub fn from_public_inputs<F>(pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert!(pis.len() == SYSTEM_CONFIG_SIZE);
        SystemConfig {
            liquidity_pool_index: i64::try_from(pis[0].to_canonical_u64()).unwrap(),
            staking_pool_index: i64::try_from(pis[1].to_canonical_u64()).unwrap(),
            liquidity_pool_cooldown_period: i64::try_from(pis[2].to_canonical_u64()).unwrap(),
            staking_pool_lockup_period: i64::try_from(pis[3].to_canonical_u64()).unwrap(),
        }
    }

    pub fn empty() -> Self {
        SystemConfig {
            liquidity_pool_index: 0,
            staking_pool_index: 0,
            liquidity_pool_cooldown_period: 0,
            staking_pool_lockup_period: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.liquidity_pool_index == 0
            && self.staking_pool_index == 0
            && self.liquidity_pool_cooldown_period == 0
            && self.staking_pool_lockup_period == 0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemConfigTarget {
    pub liquidity_pool_index: Target,
    pub staking_pool_index: Target,
    pub liquidity_pool_cooldown_period: Target,
    pub staking_pool_lockup_period: Target,
}

impl SystemConfigTarget {
    pub fn new(builder: &mut Builder) -> Self {
        SystemConfigTarget {
            liquidity_pool_index: builder.add_virtual_target(),
            staking_pool_index: builder.add_virtual_target(),
            liquidity_pool_cooldown_period: builder.add_virtual_target(),
            staking_pool_lockup_period: builder.add_virtual_target(),
        }
    }

    pub fn connect(&self, builder: &mut Builder, other: &Self) {
        builder.connect(self.liquidity_pool_index, other.liquidity_pool_index);
        builder.connect(self.staking_pool_index, other.staking_pool_index);
        builder.connect(
            self.liquidity_pool_cooldown_period,
            other.liquidity_pool_cooldown_period,
        );
        builder.connect(
            self.staking_pool_lockup_period,
            other.staking_pool_lockup_period,
        );
    }

    pub fn is_equal(builder: &mut Builder, a: &Self, b: &Self) -> BoolTarget {
        let assertions = [
            builder.is_equal(a.liquidity_pool_index, b.liquidity_pool_index),
            builder.is_equal(a.staking_pool_index, b.staking_pool_index),
            builder.is_equal(
                a.liquidity_pool_cooldown_period,
                b.liquidity_pool_cooldown_period,
            ),
            builder.is_equal(a.staking_pool_lockup_period, b.staking_pool_lockup_period),
        ];
        builder.multi_and(&assertions)
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = [
            builder.is_zero(self.liquidity_pool_index),
            builder.is_zero(self.staking_pool_index),
            builder.is_zero(self.liquidity_pool_cooldown_period),
            builder.is_zero(self.staking_pool_lockup_period),
        ];
        builder.multi_and(&assertions)
    }

    pub fn empty(builder: &mut Builder) -> Self {
        SystemConfigTarget {
            liquidity_pool_index: builder.zero(),
            staking_pool_index: builder.zero(),
            liquidity_pool_cooldown_period: builder.zero(),
            staking_pool_lockup_period: builder.zero(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.liquidity_pool_index,
            &format!("{} liquidity_pool_index", tag),
        );
        builder.println(
            self.staking_pool_index,
            &format!("{} staking_pool_index", tag),
        );
        builder.println(
            self.liquidity_pool_cooldown_period,
            &format!("{} liquidity_pool_cooldown_period", tag),
        );
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let elements = vec![
            self.liquidity_pool_index,
            self.staking_pool_index,
            self.liquidity_pool_cooldown_period,
            self.staking_pool_lockup_period,
        ];

        builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.liquidity_pool_index);
        builder.register_public_input(self.staking_pool_index);
        builder.register_public_input(self.liquidity_pool_cooldown_period);
        builder.register_public_input(self.staking_pool_lockup_period);
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert_eq!(pis.len(), SYSTEM_CONFIG_SIZE);
        SystemConfigTarget {
            liquidity_pool_index: pis[0],
            staking_pool_index: pis[1],
            liquidity_pool_cooldown_period: pis[2],
            staking_pool_lockup_period: pis[3],
        }
    }
}

pub trait SystemConfigTargetWitness<F: PrimeField64> {
    fn set_system_config_target(
        &mut self,
        system_config_target: &SystemConfigTarget,
        system_config: &SystemConfig,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> SystemConfigTargetWitness<F> for T {
    fn set_system_config_target(
        &mut self,
        system_config_target: &SystemConfigTarget,
        system_config: &SystemConfig,
    ) -> Result<()> {
        self.set_target(
            system_config_target.liquidity_pool_index,
            F::from_canonical_i64(system_config.liquidity_pool_index),
        )?;
        self.set_target(
            system_config_target.staking_pool_index,
            F::from_canonical_i64(system_config.staking_pool_index),
        )?;
        self.set_target(
            system_config_target.liquidity_pool_cooldown_period,
            F::from_canonical_i64(system_config.liquidity_pool_cooldown_period),
        )?;
        self.set_target(
            system_config_target.staking_pool_lockup_period,
            F::from_canonical_i64(system_config.staking_pool_lockup_period),
        )?;

        Ok(())
    }
}

pub fn select_system_config_target(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    a: &SystemConfigTarget,
    b: &SystemConfigTarget,
) -> SystemConfigTarget {
    SystemConfigTarget {
        liquidity_pool_index: builder.select(
            is_enabled,
            a.liquidity_pool_index,
            b.liquidity_pool_index,
        ),
        staking_pool_index: builder.select(is_enabled, a.staking_pool_index, b.staking_pool_index),
        liquidity_pool_cooldown_period: builder.select(
            is_enabled,
            a.liquidity_pool_cooldown_period,
            b.liquidity_pool_cooldown_period,
        ),
        staking_pool_lockup_period: builder.select(
            is_enabled,
            a.staking_pool_lockup_period,
            b.staking_pool_lockup_period,
        ),
    }
}
