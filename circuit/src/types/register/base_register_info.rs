// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::tx_attributes::is_integrator_fee_disabled;
use crate::types::config::Builder;
use crate::utils::CircuitBuilderUtils;

pub const BASE_REGISTER_INFO_SIZE: usize = 23;

#[derive(Clone, Debug, Deserialize, Copy)]
#[serde(default)]
pub struct BaseRegisterInfo {
    #[serde(rename = "it")]
    pub instruction_type: u8,

    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ps")]
    pub pending_size: i64, // Remaining execution amount for unrolling

    #[serde(rename = "poi")]
    pub pending_order_index: i64,

    #[serde(rename = "pcoi")]
    pub pending_client_order_index: i64,

    #[serde(rename = "pis")]
    pub pending_initial_size: i64,

    #[serde(rename = "pp")]
    pub pending_price: i64,

    #[serde(rename = "pn")]
    pub pending_nonce: i64,

    #[serde(rename = "pia")]
    pub pending_is_ask: u8,

    #[serde(rename = "pt")]
    pub pending_type: u8,

    #[serde(rename = "ptif")]
    pub pending_time_in_force: u8,

    #[serde(rename = "pro")]
    pub pending_reduce_only: u8,

    #[serde(rename = "pe")]
    pub pending_expiry: i64,

    #[serde(rename = "ptp")]
    pub pending_trigger_price: u32,

    #[serde(rename = "pts")]
    pub pending_trigger_status: u8,

    #[serde(rename = "pttoi0")]
    pub pending_to_trigger_order_index0: i64,

    #[serde(rename = "pttoi1")]
    pub pending_to_trigger_order_index1: i64,

    #[serde(rename = "ptcoi0")]
    pub pending_to_cancel_order_index0: i64,

    #[serde(rename = "gf0")]
    pub generic_field_0: i64,
    #[serde(rename = "gf1")]
    pub generic_field_1: i64,
    #[serde(rename = "gf2")]
    pub generic_field_2: i64,
    #[serde(rename = "gf3")]
    pub generic_field_3: i64,
}

impl Default for BaseRegisterInfo {
    fn default() -> Self {
        Self::empty()
    }
}

impl BaseRegisterInfo {
    pub fn empty() -> Self {
        Self {
            instruction_type: 0,
            market_index: 0,
            account_index: 0,
            pending_size: 0,
            pending_order_index: 0,
            pending_client_order_index: 0,
            pending_initial_size: 0,
            pending_price: 0,
            pending_nonce: 0,
            pending_is_ask: 0,
            pending_time_in_force: 0,
            pending_expiry: 0,
            generic_field_0: 0,
            pending_type: 0,
            pending_reduce_only: 0,
            pending_trigger_price: 0,
            pending_trigger_status: 0,
            pending_to_trigger_order_index0: 0,
            pending_to_trigger_order_index1: 0,
            pending_to_cancel_order_index0: 0,
            generic_field_1: 0,
            generic_field_2: 0,
            generic_field_3: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.instruction_type == 0
            && self.market_index == 0
            && self.account_index == 0
            && self.pending_size == 0
            && self.pending_order_index == 0
            && self.pending_client_order_index == 0
            && self.pending_initial_size == 0
            && self.pending_price == 0
            && self.pending_nonce == 0
            && self.pending_is_ask == 0
            && self.pending_type == 0
            && self.pending_time_in_force == 0
            && self.pending_reduce_only == 0
            && self.pending_expiry == 0
            && self.generic_field_0 == 0
            && self.pending_trigger_price == 0
            && self.pending_trigger_status == 0
            && self.pending_to_trigger_order_index0 == 0
            && self.pending_to_trigger_order_index1 == 0
            && self.pending_to_cancel_order_index0 == 0
            && self.generic_field_1 == 0
            && self.generic_field_2 == 0
            && self.generic_field_3 == 0
    }

    pub fn from_vec<F>(pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert_eq!(pis.len(), BASE_REGISTER_INFO_SIZE);
        BaseRegisterInfo {
            instruction_type: u8::try_from(pis[0].to_canonical_u64()).unwrap(),
            market_index: u16::try_from(pis[1].to_canonical_u64()).unwrap(),
            account_index: i64::try_from(pis[2].to_canonical_u64()).unwrap(),
            pending_size: i64::try_from(pis[3].to_canonical_u64()).unwrap(),
            pending_order_index: i64::try_from(pis[4].to_canonical_u64()).unwrap(),
            pending_client_order_index: i64::try_from(pis[5].to_canonical_u64()).unwrap(),
            pending_initial_size: i64::try_from(pis[6].to_canonical_u64()).unwrap(),
            pending_price: i64::try_from(pis[7].to_canonical_u64()).unwrap(),
            pending_nonce: i64::try_from(pis[8].to_canonical_u64()).unwrap(),
            pending_is_ask: u8::try_from(pis[9].to_canonical_u64()).unwrap(),
            pending_type: u8::try_from(pis[10].to_canonical_u64()).unwrap(),
            pending_time_in_force: u8::try_from(pis[11].to_canonical_u64()).unwrap(),
            pending_reduce_only: u8::try_from(pis[12].to_canonical_u64()).unwrap(),
            pending_expiry: i64::try_from(pis[13].to_canonical_u64()).unwrap(),
            generic_field_0: i64::try_from(pis[14].to_canonical_u64()).unwrap(),
            pending_trigger_price: u32::try_from(pis[15].to_canonical_u64()).unwrap(),
            pending_trigger_status: u8::try_from(pis[16].to_canonical_u64()).unwrap(),
            pending_to_trigger_order_index0: i64::try_from(pis[17].to_canonical_u64()).unwrap(),
            pending_to_trigger_order_index1: i64::try_from(pis[18].to_canonical_u64()).unwrap(),
            pending_to_cancel_order_index0: i64::try_from(pis[19].to_canonical_u64()).unwrap(),
            generic_field_1: i64::try_from(pis[20].to_canonical_u64()).unwrap(),
            generic_field_2: i64::try_from(pis[21].to_canonical_u64()).unwrap(),
            generic_field_3: i64::try_from(pis[22].to_canonical_u64()).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BaseRegisterInfoTarget {
    pub instruction_type: Target,
    pub market_index: Target,
    pub account_index: Target,

    pub pending_size: Target,

    pub pending_order_index: Target,
    pub pending_client_order_index: Target,
    pub pending_initial_size: Target,
    pub pending_price: Target,
    pub pending_nonce: Target,
    pub pending_is_ask: BoolTarget,

    pub pending_type: Target,
    pub pending_time_in_force: Target,
    pub pending_reduce_only: Target,
    pub pending_expiry: Target,

    pub generic_field_0: Target,

    pub pending_trigger_price: Target,
    pub pending_trigger_status: Target,
    pub pending_to_trigger_order_index0: Target,
    pub pending_to_trigger_order_index1: Target,
    pub pending_to_cancel_order_index0: Target,

    pub generic_field_1: Target,
    pub generic_field_2: Target,
    pub generic_field_3: Target,
}

impl BaseRegisterInfoTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            instruction_type: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),

            pending_size: builder.add_virtual_target(),

            pending_order_index: builder.add_virtual_target(),
            pending_client_order_index: builder.add_virtual_target(),
            pending_initial_size: builder.add_virtual_target(),
            pending_price: builder.add_virtual_target(),
            pending_nonce: builder.add_virtual_target(),
            pending_is_ask: builder.add_virtual_bool_target_safe(),

            pending_type: builder.add_virtual_target(),
            pending_time_in_force: builder.add_virtual_target(),
            pending_reduce_only: builder.add_virtual_target(),
            pending_expiry: builder.add_virtual_target(),

            generic_field_0: builder.add_virtual_target(),

            pending_trigger_price: builder.add_virtual_target(),
            pending_trigger_status: builder.add_virtual_target(),
            pending_to_trigger_order_index0: builder.add_virtual_target(),
            pending_to_trigger_order_index1: builder.add_virtual_target(),
            pending_to_cancel_order_index0: builder.add_virtual_target(),

            generic_field_1: builder.add_virtual_target(),
            generic_field_2: builder.add_virtual_target(),
            generic_field_3: builder.add_virtual_target(),
        }
    }

    pub fn connect(&self, builder: &mut Builder, other: &Self) {
        builder.connect(self.instruction_type, other.instruction_type);
        builder.connect(self.market_index, other.market_index);
        builder.connect(self.account_index, other.account_index);
        builder.connect(self.pending_size, other.pending_size);
        builder.connect(self.pending_order_index, other.pending_order_index);
        builder.connect(
            self.pending_client_order_index,
            other.pending_client_order_index,
        );
        builder.connect(self.pending_initial_size, other.pending_initial_size);
        builder.connect(self.pending_price, other.pending_price);
        builder.connect(self.pending_nonce, other.pending_nonce);
        builder.connect(self.pending_is_ask.target, other.pending_is_ask.target);
        builder.connect(self.pending_type, other.pending_type);
        builder.connect(self.pending_time_in_force, other.pending_time_in_force);
        builder.connect(self.pending_reduce_only, other.pending_reduce_only);
        builder.connect(self.pending_expiry, other.pending_expiry);
        builder.connect(self.generic_field_0, other.generic_field_0);
        builder.connect(self.pending_trigger_price, other.pending_trigger_price);
        builder.connect(self.pending_trigger_status, other.pending_trigger_status);
        builder.connect(
            self.pending_to_trigger_order_index0,
            other.pending_to_trigger_order_index0,
        );
        builder.connect(
            self.pending_to_trigger_order_index1,
            other.pending_to_trigger_order_index1,
        );
        builder.connect(
            self.pending_to_cancel_order_index0,
            other.pending_to_cancel_order_index0,
        );
        builder.connect(self.generic_field_1, other.generic_field_1);
        builder.connect(self.generic_field_2, other.generic_field_2);
        builder.connect(self.generic_field_3, other.generic_field_3);
    }

    pub fn is_equal(builder: &mut Builder, a: &Self, b: &Self) -> BoolTarget {
        let assertions = [
            builder.is_equal(a.instruction_type, b.instruction_type),
            builder.is_equal(a.market_index, b.market_index),
            builder.is_equal(a.account_index, b.account_index),
            builder.is_equal(a.pending_size, b.pending_size),
            builder.is_equal(a.pending_order_index, b.pending_order_index),
            builder.is_equal(a.pending_client_order_index, b.pending_client_order_index),
            builder.is_equal(a.pending_initial_size, b.pending_initial_size),
            builder.is_equal(a.pending_price, b.pending_price),
            builder.is_equal(a.pending_nonce, b.pending_nonce),
            builder.is_equal(a.pending_is_ask.target, b.pending_is_ask.target),
            builder.is_equal(a.pending_type, b.pending_type),
            builder.is_equal(a.pending_time_in_force, b.pending_time_in_force),
            builder.is_equal(a.pending_reduce_only, b.pending_reduce_only),
            builder.is_equal(a.pending_expiry, b.pending_expiry),
            builder.is_equal(a.generic_field_0, b.generic_field_0),
            builder.is_equal(a.pending_trigger_price, b.pending_trigger_price),
            builder.is_equal(a.pending_trigger_status, b.pending_trigger_status),
            builder.is_equal(
                a.pending_to_trigger_order_index0,
                b.pending_to_trigger_order_index0,
            ),
            builder.is_equal(
                a.pending_to_trigger_order_index1,
                b.pending_to_trigger_order_index1,
            ),
            builder.is_equal(
                a.pending_to_cancel_order_index0,
                b.pending_to_cancel_order_index0,
            ),
            builder.is_equal(a.generic_field_1, b.generic_field_1),
            builder.is_equal(a.generic_field_2, b.generic_field_2),
            builder.is_equal(a.generic_field_3, b.generic_field_3),
        ];
        builder.multi_and(&assertions)
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // Adding following fields does not overflow Goldilocks, as long as
        // these fields are guaranteed by business logic to fit these sizes.
        let added = builder.add_many([
            self.instruction_type,           // 8 bits
            self.market_index,               // 16 bits
            self.account_index,              // 48 bits
            self.pending_size,               // 48 bits
            self.pending_client_order_index, // 48 bits
            self.pending_initial_size,       // 48 bits
            self.pending_price,              // 32 bits
            self.pending_nonce,              // 48 bits
            self.pending_is_ask.target,      // 1 bit
            self.pending_type,               // 8 bits
            self.pending_time_in_force,      // 8 bits
            self.pending_reduce_only,        // 8 bits
            self.pending_expiry,             // 48 bits
            self.pending_trigger_price,      // 32 bits
            self.pending_trigger_status,     // 2 bits
            self.generic_field_2,            // generic but 32 bits
            self.generic_field_3,            // generic but 32 bits
        ]);
        let assertions = [
            builder.is_zero(added),
            builder.is_zero(self.pending_order_index),
            builder.is_zero(self.pending_to_trigger_order_index0),
            builder.is_zero(self.pending_to_trigger_order_index1),
            builder.is_zero(self.pending_to_cancel_order_index0),
            builder.is_zero(self.generic_field_0), // Generic
            builder.is_zero(self.generic_field_1), // Generic
        ];
        builder.multi_and(&assertions)
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            instruction_type: builder.zero(),
            market_index: builder.zero(),
            account_index: builder.zero(),

            pending_size: builder.zero(),

            pending_order_index: builder.zero(),
            pending_client_order_index: builder.zero(),
            pending_initial_size: builder.zero(),
            pending_price: builder.zero(),
            pending_nonce: builder.zero(),
            pending_is_ask: builder._false(),

            pending_type: builder.zero(),
            pending_time_in_force: builder.zero(),
            pending_reduce_only: builder.zero(),
            pending_expiry: builder.zero(),

            generic_field_0: builder.zero(),

            pending_trigger_price: builder.zero(),
            pending_trigger_status: builder.zero(),
            pending_to_trigger_order_index0: builder.zero(),
            pending_to_trigger_order_index1: builder.zero(),
            pending_to_cancel_order_index0: builder.zero(),

            generic_field_1: builder.zero(),
            generic_field_2: builder.zero(),
            generic_field_3: builder.zero(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.instruction_type, &format!("{} instruction_type", tag));
        builder.println(self.market_index, &format!("{} market_index", tag));
        builder.println(self.account_index, &format!("{} account_index", tag));
        builder.println(self.pending_size, &format!("{} pending_size", tag));
        builder.println(
            self.pending_order_index,
            &format!("{} pending_order_index", tag),
        );
        builder.println(
            self.pending_client_order_index,
            &format!("{} pending_client_order_index", tag),
        );
        builder.println(
            self.pending_initial_size,
            &format!("{} pending_initial_size", tag),
        );
        builder.println(self.pending_price, &format!("{} pending_price", tag));
        builder.println(self.pending_nonce, &format!("{} pending_nonce", tag));
        builder.println(
            self.pending_is_ask.target,
            &format!("{} pending_is_ask", tag),
        );
        builder.println(self.pending_type, &format!("{} pending_type", tag));
        builder.println(
            self.pending_time_in_force,
            &format!("{} pending_time_in_force", tag),
        );
        builder.println(
            self.pending_reduce_only,
            &format!("{} pending_reduce_only", tag),
        );
        builder.println(self.pending_expiry, &format!("{} pending_expiry", tag));
        builder.println(self.generic_field_0, &format!("{} generic_field_0", tag));
        builder.println(
            self.pending_trigger_price,
            &format!("{} pending_trigger_price", tag),
        );
        builder.println(
            self.pending_trigger_status,
            &format!("{} pending_trigger_status", tag),
        );
        builder.println(
            self.pending_to_trigger_order_index0,
            &format!("{} pending_to_trigger_order_index0", tag),
        );
        builder.println(
            self.pending_to_trigger_order_index1,
            &format!("{} pending_to_trigger_order_index1", tag),
        );
        builder.println(
            self.pending_to_cancel_order_index0,
            &format!("{} pending_to_cancel_order_index0", tag),
        );
        builder.println(
            self.generic_field_1,
            &format!("{} pending_integrator_fee_collector_index", tag),
        );
        builder.println(self.generic_field_2, &format!("{} generic_field_2", tag));
        builder.println(self.generic_field_3, &format!("{} generic_field_3", tag));
    }

    pub fn get_hash_parameters(&self) -> Vec<Target> {
        vec![
            self.instruction_type,
            self.market_index,
            self.account_index,
            self.pending_size,
            self.pending_order_index,
            self.pending_client_order_index,
            self.pending_initial_size,
            self.pending_price,
            self.pending_nonce,
            self.pending_is_ask.target,
            self.pending_type,
            self.pending_time_in_force,
            self.pending_reduce_only,
            self.pending_expiry,
            self.generic_field_0,
            self.pending_trigger_price,
            self.pending_trigger_status,
            self.pending_to_trigger_order_index0,
            self.pending_to_trigger_order_index1,
            self.pending_to_cancel_order_index0,
            self.generic_field_1,
            self.generic_field_2,
            self.generic_field_3,
        ]
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.instruction_type);
        builder.register_public_input(self.market_index);
        builder.register_public_input(self.account_index);
        builder.register_public_input(self.pending_size);
        builder.register_public_input(self.pending_order_index);
        builder.register_public_input(self.pending_client_order_index);
        builder.register_public_input(self.pending_initial_size);
        builder.register_public_input(self.pending_price);
        builder.register_public_input(self.pending_nonce);
        builder.register_public_input(self.pending_is_ask.target);
        builder.register_public_input(self.pending_type);
        builder.register_public_input(self.pending_time_in_force);
        builder.register_public_input(self.pending_reduce_only);
        builder.register_public_input(self.pending_expiry);
        builder.register_public_input(self.generic_field_0);
        builder.register_public_input(self.pending_trigger_price);
        builder.register_public_input(self.pending_trigger_status);
        builder.register_public_input(self.pending_to_trigger_order_index0);
        builder.register_public_input(self.pending_to_trigger_order_index1);
        builder.register_public_input(self.pending_to_cancel_order_index0);
        builder.register_public_input(self.generic_field_1);
        builder.register_public_input(self.generic_field_2);
        builder.register_public_input(self.generic_field_3);
    }

    /// Converts a slice of `Target` into a `BaseRegisterInfoTarget`. Follow same order as [`Self::register_public_input`].
    pub fn from_vec(pis: &[Target]) -> Self {
        assert_eq!(pis.len(), BASE_REGISTER_INFO_SIZE);
        BaseRegisterInfoTarget {
            instruction_type: pis[0],
            market_index: pis[1],
            account_index: pis[2],
            pending_size: pis[3],
            pending_order_index: pis[4],
            pending_client_order_index: pis[5],
            pending_initial_size: pis[6],
            pending_price: pis[7],
            pending_nonce: pis[8],
            pending_is_ask: BoolTarget::new_unsafe(pis[9]),
            pending_type: pis[10],
            pending_time_in_force: pis[11],
            pending_reduce_only: pis[12],
            pending_expiry: pis[13],
            generic_field_0: pis[14],
            pending_trigger_price: pis[15],
            pending_trigger_status: pis[16],
            pending_to_trigger_order_index0: pis[17],
            pending_to_trigger_order_index1: pis[18],
            pending_to_cancel_order_index0: pis[19],
            generic_field_1: pis[20],
            generic_field_2: pis[21],
            generic_field_3: pis[22],
        }
    }

    pub fn to_order_fields_from_generic_fields(
        &self,
        builder: &mut Builder,
    ) -> (
        Target, // integrator_fee_collector_index
        Target, // integrator_taker_fee
        Target, // integrator_maker_fee
        Target, // order_flags
    ) {
        let zero = builder.zero();
        let is_integrator_fee_disabled = is_integrator_fee_disabled(builder, self.generic_field_1);
        (
            self.generic_field_1,
            builder.select(is_integrator_fee_disabled, zero, self.generic_field_2),
            builder.select(is_integrator_fee_disabled, zero, self.generic_field_3),
            builder.select(is_integrator_fee_disabled, self.generic_field_2, zero),
        )
    }
}

pub trait BaseRegisterInfoTargetWitness<F: PrimeField64> {
    fn set_base_register_info_target(
        &mut self,
        register_target: &BaseRegisterInfoTarget,
        register: &BaseRegisterInfo,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> BaseRegisterInfoTargetWitness<F> for T {
    fn set_base_register_info_target(
        &mut self,
        register_target: &BaseRegisterInfoTarget,
        register: &BaseRegisterInfo,
    ) -> Result<()> {
        self.set_target(
            register_target.instruction_type,
            F::from_canonical_u8(register.instruction_type),
        )?;
        self.set_target(
            register_target.market_index,
            F::from_canonical_u16(register.market_index),
        )?;
        self.set_target(
            register_target.account_index,
            F::from_canonical_i64(register.account_index),
        )?;
        self.set_target(
            register_target.pending_size,
            F::from_canonical_i64(register.pending_size),
        )?;
        self.set_target(
            register_target.pending_order_index,
            F::from_canonical_i64(register.pending_order_index),
        )?;
        self.set_target(
            register_target.pending_client_order_index,
            F::from_canonical_i64(register.pending_client_order_index),
        )?;
        self.set_target(
            register_target.pending_initial_size,
            F::from_canonical_i64(register.pending_initial_size),
        )?;
        self.set_target(
            register_target.pending_price,
            F::from_canonical_i64(register.pending_price),
        )?;
        self.set_target(
            register_target.pending_nonce,
            F::from_canonical_i64(register.pending_nonce),
        )?;
        self.set_bool_target(register_target.pending_is_ask, register.pending_is_ask == 1)?;
        self.set_target(
            register_target.pending_type,
            F::from_canonical_u8(register.pending_type),
        )?;
        self.set_target(
            register_target.pending_time_in_force,
            F::from_canonical_u8(register.pending_time_in_force),
        )?;
        self.set_target(
            register_target.pending_reduce_only,
            F::from_canonical_u8(register.pending_reduce_only),
        )?;
        self.set_target(
            register_target.pending_expiry,
            F::from_canonical_i64(register.pending_expiry),
        )?;
        self.set_target(
            register_target.generic_field_0,
            F::from_canonical_i64(register.generic_field_0),
        )?;
        self.set_target(
            register_target.pending_trigger_price,
            F::from_canonical_u32(register.pending_trigger_price),
        )?;
        self.set_target(
            register_target.pending_trigger_status,
            F::from_canonical_u8(register.pending_trigger_status),
        )?;
        self.set_target(
            register_target.pending_to_trigger_order_index0,
            F::from_canonical_i64(register.pending_to_trigger_order_index0),
        )?;
        self.set_target(
            register_target.pending_to_trigger_order_index1,
            F::from_canonical_i64(register.pending_to_trigger_order_index1),
        )?;
        self.set_target(
            register_target.pending_to_cancel_order_index0,
            F::from_canonical_i64(register.pending_to_cancel_order_index0),
        )?;
        self.set_target(
            register_target.generic_field_1,
            F::from_canonical_i64(register.generic_field_1),
        )?;
        self.set_target(
            register_target.generic_field_2,
            F::from_canonical_i64(register.generic_field_2),
        )?;
        self.set_target(
            register_target.generic_field_3,
            F::from_canonical_i64(register.generic_field_3),
        )?;

        Ok(())
    }
}

pub fn select_register_target(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    a: &BaseRegisterInfoTarget,
    b: &BaseRegisterInfoTarget,
) -> BaseRegisterInfoTarget {
    BaseRegisterInfoTarget {
        instruction_type: builder.select(is_enabled, a.instruction_type, b.instruction_type),
        market_index: builder.select(is_enabled, a.market_index, b.market_index),
        account_index: builder.select(is_enabled, a.account_index, b.account_index),
        pending_size: builder.select(is_enabled, a.pending_size, b.pending_size),
        pending_client_order_index: builder.select(
            is_enabled,
            a.pending_client_order_index,
            b.pending_client_order_index,
        ),
        pending_order_index: builder.select(
            is_enabled,
            a.pending_order_index,
            b.pending_order_index,
        ),
        pending_initial_size: builder.select(
            is_enabled,
            a.pending_initial_size,
            b.pending_initial_size,
        ),
        pending_price: builder.select(is_enabled, a.pending_price, b.pending_price),
        pending_nonce: builder.select(is_enabled, a.pending_nonce, b.pending_nonce),
        pending_is_ask: builder.select_bool(is_enabled, a.pending_is_ask, b.pending_is_ask),
        pending_type: builder.select(is_enabled, a.pending_type, b.pending_type),
        pending_time_in_force: builder.select(
            is_enabled,
            a.pending_time_in_force,
            b.pending_time_in_force,
        ),
        pending_reduce_only: builder.select(
            is_enabled,
            a.pending_reduce_only,
            b.pending_reduce_only,
        ),
        pending_expiry: builder.select(is_enabled, a.pending_expiry, b.pending_expiry),
        generic_field_0: builder.select(is_enabled, a.generic_field_0, b.generic_field_0),
        pending_trigger_price: builder.select(
            is_enabled,
            a.pending_trigger_price,
            b.pending_trigger_price,
        ),
        pending_trigger_status: builder.select(
            is_enabled,
            a.pending_trigger_status,
            b.pending_trigger_status,
        ),
        pending_to_trigger_order_index0: builder.select(
            is_enabled,
            a.pending_to_trigger_order_index0,
            b.pending_to_trigger_order_index0,
        ),
        pending_to_trigger_order_index1: builder.select(
            is_enabled,
            a.pending_to_trigger_order_index1,
            b.pending_to_trigger_order_index1,
        ),
        pending_to_cancel_order_index0: builder.select(
            is_enabled,
            a.pending_to_cancel_order_index0,
            b.pending_to_cancel_order_index0,
        ),
        generic_field_1: builder.select(is_enabled, a.generic_field_1, b.generic_field_1),
        generic_field_2: builder.select(is_enabled, a.generic_field_2, b.generic_field_2),
        generic_field_3: builder.select(is_enabled, a.generic_field_3, b.generic_field_3),
    }
}

#[cfg(test)]
impl BaseRegisterInfoTarget {
    pub fn random(builder: &mut Builder) -> Self {
        use rand::Rng;

        use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

        Self {
            instruction_type: builder.constant_from_u8(rand::thread_rng().gen_range(0..=255) as u8),
            market_index: builder.constant_from_u8(rand::thread_rng().gen_range(0..=255) as u8),
            account_index: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_size: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_order_index: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_client_order_index: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_initial_size: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_price: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 32) - 1) as i64),
            pending_nonce: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_is_ask: builder.constant_bool(rand::thread_rng().gen_bool(0.5)),
            pending_type: builder.constant_from_u8(rand::thread_rng().gen_range(0..=255) as u8),
            pending_time_in_force: builder
                .constant_from_u8(rand::thread_rng().gen_range(0..=3) as u8),
            pending_reduce_only: builder.constant_from_u8(rand::thread_rng().gen_bool(0.5) as u8),
            pending_expiry: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            generic_field_0: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_trigger_price: builder
                .constant_u32(rand::thread_rng().gen_range(0..=(1usize << 32) - 1) as u32)
                .0,
            pending_trigger_status: builder
                .constant_from_u8(rand::thread_rng().gen_range(0..=3) as u8),
            pending_to_trigger_order_index0: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_to_trigger_order_index1: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            pending_to_cancel_order_index0: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            generic_field_1: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 48) - 1) as i64),
            generic_field_2: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 32) - 1) as i64),
            generic_field_3: builder
                .constant_i64(rand::thread_rng().gen_range(0..=(1usize << 32) - 1) as i64),
        }
    }
}
