// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::tx_attributes::is_integrator_fee_disabled;
use crate::types::config::Builder;
use crate::types::constants::{NIL_ACCOUNT_INDEX, NIL_CLIENT_ORDER_INDEX, NIL_ORDER_INDEX};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AccountOrder {
    #[serde(rename = "i0")]
    pub index_0: i64,

    #[serde(rename = "i1")]
    pub index_1: i64,

    #[serde(rename = "oai", default)]
    pub owner_account_index: i64,

    #[serde(rename = "oi", default)]
    pub order_index: i64,

    #[serde(rename = "coi", default)]
    pub client_order_index: i64,

    #[serde(rename = "iba", default)]
    pub initial_base_amount: i64,

    #[serde(rename = "p", default)]
    pub price: u32,

    #[serde(rename = "n", default)]
    pub nonce: i64,

    #[serde(rename = "rba", default)]
    pub remaining_base_amount: i64,

    #[serde(rename = "a", default)]
    pub is_ask: u8,

    #[serde(rename = "t", default)]
    pub order_type: u8,

    #[serde(rename = "tif", default)]
    pub time_in_force: u8,

    #[serde(rename = "ro", default)]
    pub reduce_only: u8,

    #[serde(rename = "tp", default)]
    pub trigger_price: u32,

    #[serde(rename = "e", default)]
    pub expiry: i64,

    #[serde(rename = "ts", default)]
    pub trigger_status: u8,

    #[serde(rename = "ttoi0", default)]
    pub to_trigger_order_index0: i64,

    #[serde(rename = "ttoi1", default)]
    pub to_trigger_order_index1: i64,

    #[serde(rename = "tcoi0", default)]
    pub to_cancel_order_index0: i64,

    #[serde(rename = "ifci", default)]
    pub integrator_fee_collector_index: i64,
    #[serde(rename = "itf", default)]
    pub integrator_taker_fee: i64,
    #[serde(rename = "imf", default)]
    pub integrator_maker_fee: i64,
    #[serde(rename = "of", default)]
    pub order_flags: u64,
}

impl AccountOrder {
    pub fn empty() -> Self {
        Self {
            index_0: NIL_ORDER_INDEX,
            index_1: NIL_CLIENT_ORDER_INDEX,

            owner_account_index: 0,

            order_index: NIL_ORDER_INDEX,
            client_order_index: NIL_CLIENT_ORDER_INDEX,

            initial_base_amount: 0,
            price: 0,
            nonce: 0,
            remaining_base_amount: 0,
            is_ask: 0,
            order_type: 0,
            time_in_force: 0,
            reduce_only: 0,
            trigger_price: 0,
            expiry: 0,
            trigger_status: 0,
            to_trigger_order_index0: NIL_ORDER_INDEX,
            to_trigger_order_index1: NIL_ORDER_INDEX,
            to_cancel_order_index0: NIL_ORDER_INDEX,

            integrator_fee_collector_index: NIL_ACCOUNT_INDEX,
            integrator_taker_fee: 0,
            integrator_maker_fee: 0,
            order_flags: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AccountOrderTarget {
    pub index_0: Target,
    pub index_1: Target,
    pub order_index: Target,
    pub client_order_index: Target,
    pub owner_account_index: Target,
    pub initial_base_amount: Target,
    pub price: Target,
    pub nonce: Target,
    pub remaining_base_amount: Target,
    pub is_ask: BoolTarget,
    pub order_type: Target,
    pub time_in_force: Target,
    pub reduce_only: Target,
    pub trigger_price: Target,
    pub expiry: Target,
    pub trigger_status: Target,
    pub to_trigger_order_index0: Target,
    pub to_trigger_order_index1: Target,
    pub to_cancel_order_index0: Target,

    pub integrator_fee_collector_index: Target,
    pub integrator_taker_fee: Target,
    pub integrator_maker_fee: Target,
    pub order_flags: Target,
}

impl AccountOrderTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            index_0: builder.add_virtual_target(),
            index_1: builder.add_virtual_target(),
            owner_account_index: builder.add_virtual_target(),
            order_index: builder.add_virtual_target(),
            client_order_index: builder.add_virtual_target(),

            initial_base_amount: builder.add_virtual_target(),
            price: builder.add_virtual_target(),
            nonce: builder.add_virtual_target(),
            remaining_base_amount: builder.add_virtual_target(),
            is_ask: builder.add_virtual_bool_target_safe(),
            order_type: builder.add_virtual_target(),
            time_in_force: builder.add_virtual_target(),
            reduce_only: builder.add_virtual_target(),
            trigger_price: builder.add_virtual_target(),
            expiry: builder.add_virtual_target(),

            trigger_status: builder.add_virtual_target(),
            to_trigger_order_index0: builder.add_virtual_target(),
            to_trigger_order_index1: builder.add_virtual_target(),
            to_cancel_order_index0: builder.add_virtual_target(),

            integrator_fee_collector_index: builder.add_virtual_target(),
            integrator_taker_fee: builder.add_virtual_target(),
            integrator_maker_fee: builder.add_virtual_target(),
            order_flags: builder.add_virtual_target(),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(self.index_0, &format!("{} index_0", tag));
        builder.println(self.index_1, &format!("{} index_1", tag));
        builder.println(self.order_index, &format!("{} order_index", tag));
        builder.println(
            self.client_order_index,
            &format!("{} client_order_index", tag),
        );
        builder.println(
            self.owner_account_index,
            &format!("{} owner_account_index", tag),
        );
        builder.println(
            self.initial_base_amount,
            &format!("{} initial_base_amount", tag),
        );
        builder.println(self.price, &format!("{} price", tag));
        builder.println(self.nonce, &format!("{} nonce", tag));
        builder.println(
            self.remaining_base_amount,
            &format!("{} remaining_base_amount", tag),
        );
        builder.println(self.is_ask.target, &format!("{} is_ask", tag));
        builder.println(self.order_type, &format!("{} order_type", tag));
        builder.println(self.time_in_force, &format!("{} time_in_force", tag));
        builder.println(self.reduce_only, &format!("{} reduce_only", tag));
        builder.println(self.trigger_price, &format!("{} trigger_price", tag));
        builder.println(self.expiry, &format!("{} expiry", tag));
        builder.println(self.trigger_status, &format!("{} trigger_status", tag));
        builder.println(
            self.to_trigger_order_index0,
            &format!("{} to_trigger_order_index0", tag),
        );
        builder.println(
            self.to_trigger_order_index1,
            &format!("{} to_trigger_order_index1", tag),
        );
        builder.println(
            self.to_cancel_order_index0,
            &format!("{} to_cancel_order_index0", tag),
        );
        builder.println(
            self.integrator_fee_collector_index,
            &format!("{} integrator_fee_collector_index", tag),
        );
        builder.println(
            self.integrator_taker_fee,
            &format!("{} integrator_taker_fee", tag),
        );
        builder.println(
            self.integrator_maker_fee,
            &format!("{} integrator_maker_fee", tag),
        );
        builder.println(self.order_flags, &format!("{} order_flags", tag));
    }

    pub fn empty(
        builder: &mut Builder,
        index_0: Target,
        index_1: Target,
        owner_account_index: Target,
    ) -> Self {
        Self {
            index_0,
            index_1,
            owner_account_index,
            order_index: builder.zero(),
            client_order_index: builder.zero(),

            initial_base_amount: builder.zero(),
            price: builder.zero(),
            nonce: builder.zero(),
            remaining_base_amount: builder.zero(),
            is_ask: builder._false(),
            order_type: builder.zero(),
            time_in_force: builder.zero(),
            reduce_only: builder.zero(),
            trigger_price: builder.zero(),
            expiry: builder.zero(),

            trigger_status: builder.zero(),
            to_trigger_order_index0: builder.zero(),
            to_trigger_order_index1: builder.zero(),
            to_cancel_order_index0: builder.zero(),

            integrator_fee_collector_index: builder.zero(),
            integrator_taker_fee: builder.zero(),
            integrator_maker_fee: builder.zero(),
            order_flags: builder.zero(),
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // Adding following fields does not overflow Goldilocks, as long as
        // these fields are guaranteed by business logic to fit these sizes.
        let added = builder.add_many([
            self.client_order_index,             // 48 bits
            self.initial_base_amount,            // 48 bits
            self.price,                          // 32 bits
            self.nonce,                          // 48 bits
            self.remaining_base_amount,          // 48 bits
            self.is_ask.target,                  // 1 bit
            self.order_type,                     // max 8
            self.time_in_force,                  // 2 bits
            self.reduce_only,                    // 1 bit
            self.trigger_price,                  // 32 bits
            self.expiry,                         // 48 bits
            self.trigger_status,                 // 2 bits
            self.integrator_fee_collector_index, // 48 bits
            self.integrator_taker_fee,           // 32 bits
            self.integrator_maker_fee,           // 32 bits
        ]);
        // Order indexes are 63 bits
        let assertions = [
            builder.is_zero(added),
            builder.is_zero(self.order_index),
            builder.is_zero(self.to_trigger_order_index0),
            builder.is_zero(self.to_trigger_order_index1),
            builder.is_zero(self.to_cancel_order_index0),
            builder.is_zero(self.order_flags),
        ];
        builder.multi_and(&assertions)
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let elements = vec![
            self.order_index,
            self.client_order_index,
            self.initial_base_amount,
            self.price,
            self.nonce,
            self.remaining_base_amount,
            self.is_ask.target,
            self.order_type,
            self.time_in_force,
            self.reduce_only,
            self.trigger_price,
            self.expiry,
            self.trigger_status,
            self.to_trigger_order_index0,
            self.to_trigger_order_index1,
            self.to_cancel_order_index0,
            self.integrator_fee_collector_index,
            self.integrator_taker_fee,
            self.integrator_maker_fee,
            self.order_flags,
        ];
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);

        let empty_hash = builder.zero_hash_out();

        let is_empty = self.is_empty(builder);
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    pub fn get_register_generic_fields_from_order(
        &self,
        builder: &mut Builder,
    ) -> (
        Target, // generic_field_1
        Target, // generic_field_2
        Target, // generic_field_3
    ) {
        let is_integrator_fee_disabled =
            is_integrator_fee_disabled(builder, self.integrator_fee_collector_index);
        (
            self.integrator_fee_collector_index,
            builder.select(
                is_integrator_fee_disabled,
                self.order_flags,
                self.integrator_taker_fee,
            ),
            self.integrator_maker_fee,
        )
    }
}

pub trait AccountOrderTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_order_target(&mut self, a: &AccountOrderTarget, b: &AccountOrder) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountOrderTargetWitness<F> for T
{
    fn set_account_order_target(&mut self, a: &AccountOrderTarget, b: &AccountOrder) -> Result<()> {
        self.set_target(a.index_0, F::from_canonical_i64(b.index_0))?;
        self.set_target(a.index_1, F::from_canonical_i64(b.index_1))?;
        self.set_target(
            a.owner_account_index,
            F::from_canonical_i64(b.owner_account_index),
        )?;
        self.set_target(a.order_index, F::from_canonical_i64(b.order_index))?;
        self.set_target(
            a.client_order_index,
            F::from_canonical_i64(b.client_order_index),
        )?;

        self.set_target(
            a.initial_base_amount,
            F::from_canonical_i64(b.initial_base_amount),
        )?;
        self.set_target(a.price, F::from_canonical_u32(b.price))?;
        self.set_target(a.nonce, F::from_canonical_i64(b.nonce))?;
        self.set_target(
            a.remaining_base_amount,
            F::from_canonical_i64(b.remaining_base_amount),
        )?;
        self.set_bool_target(a.is_ask, b.is_ask == 1)?;
        self.set_target(a.order_type, F::from_canonical_u8(b.order_type))?;
        self.set_target(a.time_in_force, F::from_canonical_u8(b.time_in_force))?;
        self.set_target(a.reduce_only, F::from_canonical_u8(b.reduce_only))?;
        self.set_target(a.trigger_price, F::from_canonical_u32(b.trigger_price))?;
        self.set_target(a.expiry, F::from_canonical_i64(b.expiry))?;
        self.set_target(a.trigger_status, F::from_canonical_u8(b.trigger_status))?;
        self.set_target(
            a.to_trigger_order_index0,
            F::from_canonical_i64(b.to_trigger_order_index0),
        )?;
        self.set_target(
            a.to_trigger_order_index1,
            F::from_canonical_i64(b.to_trigger_order_index1),
        )?;
        self.set_target(
            a.to_cancel_order_index0,
            F::from_canonical_i64(b.to_cancel_order_index0),
        )?;
        self.set_target(
            a.integrator_fee_collector_index,
            F::from_canonical_i64(b.integrator_fee_collector_index),
        )?;
        self.set_target(
            a.integrator_taker_fee,
            F::from_canonical_i64(b.integrator_taker_fee),
        )?;
        self.set_target(
            a.integrator_maker_fee,
            F::from_canonical_i64(b.integrator_maker_fee),
        )?;
        self.set_target(a.order_flags, F::from_canonical_u64(b.order_flags))?;

        Ok(())
    }
}

pub fn select_account_order_target(
    builder: &mut Builder,
    flag: BoolTarget,
    a: &AccountOrderTarget,
    b: &AccountOrderTarget,
) -> AccountOrderTarget {
    AccountOrderTarget {
        index_0: builder.select(flag, a.index_0, b.index_0),
        index_1: builder.select(flag, a.index_1, b.index_1),
        owner_account_index: builder.select(flag, a.owner_account_index, b.owner_account_index),
        order_index: builder.select(flag, a.order_index, b.order_index),
        client_order_index: builder.select(flag, a.client_order_index, b.client_order_index),

        initial_base_amount: builder.select(flag, a.initial_base_amount, b.initial_base_amount),
        price: builder.select(flag, a.price, b.price),
        nonce: builder.select(flag, a.nonce, b.nonce),
        remaining_base_amount: builder.select(
            flag,
            a.remaining_base_amount,
            b.remaining_base_amount,
        ),
        is_ask: builder.select_bool(flag, a.is_ask, b.is_ask),
        order_type: builder.select(flag, a.order_type, b.order_type),
        time_in_force: builder.select(flag, a.time_in_force, b.time_in_force),
        reduce_only: builder.select(flag, a.reduce_only, b.reduce_only),
        trigger_price: builder.select(flag, a.trigger_price, b.trigger_price),
        expiry: builder.select(flag, a.expiry, b.expiry),

        trigger_status: builder.select(flag, a.trigger_status, b.trigger_status),
        to_trigger_order_index0: builder.select(
            flag,
            a.to_trigger_order_index0,
            b.to_trigger_order_index0,
        ),
        to_trigger_order_index1: builder.select(
            flag,
            a.to_trigger_order_index1,
            b.to_trigger_order_index1,
        ),
        to_cancel_order_index0: builder.select(
            flag,
            a.to_cancel_order_index0,
            b.to_cancel_order_index0,
        ),

        integrator_fee_collector_index: builder.select(
            flag,
            a.integrator_fee_collector_index,
            b.integrator_fee_collector_index,
        ),
        integrator_taker_fee: builder.select(flag, a.integrator_taker_fee, b.integrator_taker_fee),
        integrator_maker_fee: builder.select(flag, a.integrator_maker_fee, b.integrator_maker_fee),
        order_flags: builder.select(flag, a.order_flags, b.order_flags),
    }
}

pub struct OrderFlags {
    pub self_trade_behavior_mode: Target,
    pub self_trade_equality_mode: Target,
}
impl OrderFlags {
    pub fn from_target(builder: &mut Builder, order_flags: Target) -> Self {
        let le_bits = builder.split_le(order_flags, 3);
        Self {
            self_trade_behavior_mode: builder.le_sum(le_bits[0..2].iter()),
            self_trade_equality_mode: le_bits[2].target,
        }
    }
    pub fn to_target(&self, builder: &mut Builder) -> Target {
        let four = builder.constant_u64(4);
        builder.mul_add(
            four,
            self.self_trade_equality_mode,
            self.self_trade_behavior_mode,
        )
    }
    pub fn is_master_account_index_equality_mode(&self) -> BoolTarget {
        BoolTarget::new_unsafe(self.self_trade_equality_mode)
    }
    pub fn is_account_index_equality_mode(&self, builder: &mut Builder) -> BoolTarget {
        builder.not(self.is_master_account_index_equality_mode())
    }
}
