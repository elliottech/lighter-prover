// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field64, PrimeField64};
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::Target;
use plonky2::iop::witness::Witness;
use plonky2::plonk::config::Hasher;
use serde::Deserialize;

use super::config::{Builder, F};
use crate::circuit_logger::CircuitBuilderLogging;
use crate::poseidon2::Poseidon2Hash;

pub const STATE_METADATA_SIZE: usize = 3;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct StateMetadata {
    #[serde(rename = "lfr")]
    pub last_funding_round_timestamp: i64,

    #[serde(rename = "lopt")]
    pub last_oracle_price_timestamp: i64,

    #[serde(rename = "lpt")]
    pub last_premium_timestamp: i64,
}

impl Default for StateMetadata {
    fn default() -> Self {
        StateMetadata::empty()
    }
}

impl StateMetadata {
    pub fn empty() -> Self {
        StateMetadata {
            last_funding_round_timestamp: 0,
            last_oracle_price_timestamp: 0,
            last_premium_timestamp: 0,
        }
    }

    pub fn hash(&self) -> HashOut<F> {
        Poseidon2Hash::hash_no_pad(&[
            F::from_canonical_i64(self.last_funding_round_timestamp),
            F::from_canonical_i64(self.last_oracle_price_timestamp),
            F::from_canonical_i64(self.last_premium_timestamp),
        ])
    }

    pub fn to_public_inputs(&self) -> Vec<F> {
        vec![
            F::from_canonical_i64(self.last_funding_round_timestamp),
            F::from_canonical_i64(self.last_oracle_price_timestamp),
            F::from_canonical_i64(self.last_premium_timestamp),
        ]
    }

    pub fn from_public_inputs<F>(pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert_eq!(pis.len(), STATE_METADATA_SIZE);

        StateMetadata {
            last_funding_round_timestamp: pis[0].to_canonical_u64() as i64,
            last_oracle_price_timestamp: pis[1].to_canonical_u64() as i64,
            last_premium_timestamp: pis[2].to_canonical_u64() as i64,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]

pub struct StateMetadataTarget {
    pub last_funding_round_timestamp: Target,
    pub last_oracle_price_timestamp: Target,
    pub last_premium_timestamp: Target,
}

impl StateMetadataTarget {
    pub fn new(builder: &mut Builder) -> Self {
        StateMetadataTarget {
            last_funding_round_timestamp: builder.add_virtual_target(),
            last_oracle_price_timestamp: builder.add_virtual_target(),
            last_premium_timestamp: builder.add_virtual_target(),
        }
    }

    pub fn new_public(builder: &mut Builder) -> Self {
        StateMetadataTarget {
            last_funding_round_timestamp: builder.add_virtual_public_input(),
            last_oracle_price_timestamp: builder.add_virtual_public_input(),
            last_premium_timestamp: builder.add_virtual_public_input(),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        StateMetadataTarget {
            last_funding_round_timestamp: builder.zero(),
            last_oracle_price_timestamp: builder.zero(),
            last_premium_timestamp: builder.zero(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.last_funding_round_timestamp,
            &format!("{} last_funding_round_timestamp", tag),
        );
        builder.println(
            self.last_oracle_price_timestamp,
            &format!("{} last_oracle_price_timestamp", tag),
        );
        builder.println(
            self.last_premium_timestamp,
            &format!("{} last_premium_timestamp", tag),
        );
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(vec![
            self.last_funding_round_timestamp,
            self.last_oracle_price_timestamp,
            self.last_premium_timestamp,
        ])
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.last_funding_round_timestamp);
        builder.register_public_input(self.last_oracle_price_timestamp);
        builder.register_public_input(self.last_premium_timestamp);
    }
}

pub fn connect_state_metadata_target(
    builder: &mut Builder,
    lhs: &StateMetadataTarget,
    rhs: &StateMetadataTarget,
) {
    builder.connect(
        lhs.last_funding_round_timestamp,
        rhs.last_funding_round_timestamp,
    );
    builder.connect(
        lhs.last_oracle_price_timestamp,
        rhs.last_oracle_price_timestamp,
    );
    builder.connect(lhs.last_premium_timestamp, rhs.last_premium_timestamp);
}

pub trait StateMetadataTargetWitness<F: PrimeField64> {
    fn set_state_metadata_target(
        &mut self,
        state_metadata_target: &StateMetadataTarget,
        state_metadata: &StateMetadata,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> StateMetadataTargetWitness<F> for T {
    fn set_state_metadata_target(
        &mut self,
        state_metadata_target: &StateMetadataTarget,
        state_metadata: &StateMetadata,
    ) -> Result<()> {
        self.set_target(
            state_metadata_target.last_funding_round_timestamp,
            F::from_canonical_i64(state_metadata.last_funding_round_timestamp),
        )?;
        self.set_target(
            state_metadata_target.last_oracle_price_timestamp,
            F::from_canonical_i64(state_metadata.last_oracle_price_timestamp),
        )?;
        self.set_target(
            state_metadata_target.last_premium_timestamp,
            F::from_canonical_i64(state_metadata.last_premium_timestamp),
        )?;

        Ok(())
    }
}
