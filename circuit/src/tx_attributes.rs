// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use hashbrown::HashMap;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;
use serde_with::serde_as;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::types::account::AccountTarget;
use crate::types::account_order::OrderFlags;
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::market::MarketTarget;
use crate::types::system_config::SystemConfigTarget;
use crate::utils::CircuitBuilderUtils;

pub const NB_ATTRIBUTES_PER_TX: usize = 4;

pub const ATTR_NIL: usize = 0;
pub const ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX: usize = 1;
pub const ATTR_INTEGRATOR_TAKER_FEE: usize = 2;
pub const ATTR_INTEGRATOR_MAKER_FEE: usize = 3;
pub const ATTR_SKIP_TX_NONCE: usize = 4;
pub const ATTR_CANCEL_ALL_MARKET_INDEX: usize = 5;
pub const ATTR_SELF_TRADE_BEHAVIOR_MODE: usize = 6;
pub const ATTR_SELF_TRADE_EQUALITY_MODE: usize = 7;
pub const TOTAL_ATTRIBUTE_COUNT: usize = ATTR_SELF_TRADE_EQUALITY_MODE + 1;

pub const ATTRIBUTE_TYPE_BITS: usize = 3;
lazy_static! {
    pub static ref ATTR_BIT_SIZES: HashMap<usize, usize> = {
        let mut m = HashMap::new();
        m.insert(ATTR_NIL, 0);
        m.insert(
            ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX,
            ACCOUNT_INDEX_BITS,
        );
        m.insert(ATTR_INTEGRATOR_TAKER_FEE, 24);
        m.insert(ATTR_INTEGRATOR_MAKER_FEE, 24);
        m.insert(ATTR_SKIP_TX_NONCE, 1);
        m.insert(ATTR_CANCEL_ALL_MARKET_INDEX, MARKET_INDEX_BITS);
        m.insert(ATTR_SELF_TRADE_BEHAVIOR_MODE, 8);
        m.insert(ATTR_SELF_TRADE_EQUALITY_MODE, 8);
        m
    };
    pub static ref ATTR_MAX_VALUES: HashMap<usize, usize> = {
        let mut m = HashMap::new();
        m.insert(ATTR_NIL, 0usize);
        m.insert(
            ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX,
            NIL_ACCOUNT_INDEX as usize,
        );
        m.insert(ATTR_INTEGRATOR_TAKER_FEE, FEE_TICK as usize);
        m.insert(ATTR_INTEGRATOR_MAKER_FEE, FEE_TICK as usize);
        m.insert(ATTR_SKIP_TX_NONCE, 1usize);
        m.insert(ATTR_CANCEL_ALL_MARKET_INDEX, NIL_MARKET_INDEX as usize);
        m.insert(ATTR_SELF_TRADE_BEHAVIOR_MODE, SELF_TRADE_BEHAVIOR_REDUCE as usize);
        m.insert(ATTR_SELF_TRADE_EQUALITY_MODE, SELF_TRADE_EQUALITY_MASTER_ACCOUNT_INDEX as usize);
        m
    };
    pub static ref ATTR_NIL_VALUES: [F; TOTAL_ATTRIBUTE_COUNT] = {
        [
            F::ZERO, // Nil
            F::ZERO, // Integrator Fee Collector Index
            F::ZERO, // Integrator Taker Fee
            F::ZERO, // Integrator Maker Fee
            F::ZERO, // Skip Tx Nonce
            F::from_canonical_u8(NIL_MARKET_INDEX), // Cancel All Market Index
            F::ZERO, // Self-trade behavior mode
            F::ZERO, // Self-trade equality mode
        ]
    };
}

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
pub struct TxAttributes {
    #[serde(rename = "at")]
    pub attribute_types: [u8; NB_ATTRIBUTES_PER_TX],
    #[serde(rename = "av")]
    pub attribute_values: [i64; NB_ATTRIBUTES_PER_TX],
}

impl Default for TxAttributes {
    fn default() -> Self {
        Self {
            attribute_types: [0; NB_ATTRIBUTES_PER_TX],
            attribute_values: [0; NB_ATTRIBUTES_PER_TX],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TxAttributesTarget {
    inner_types: [Target; NB_ATTRIBUTES_PER_TX],
    inner_values: [Target; NB_ATTRIBUTES_PER_TX],
    values: [Target; TOTAL_ATTRIBUTE_COUNT],
}

impl TxAttributesTarget {
    pub fn get(&self, attribute_type: usize) -> Target {
        self.values[attribute_type]
    }

    fn set(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        attribute_type: usize,
        value: Target,
    ) {
        self.values[attribute_type] =
            builder.select(is_enabled, value, self.values[attribute_type]);
    }

    pub fn print_inner(&self, builder: &mut Builder, tag: &str) {
        builder.println_arr(&self.inner_types, &format!("{} attribute types", tag));
        builder.println_arr(&self.inner_values, &format!("{} attribute values", tag));
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_arr(&self.values, tag);
    }

    pub fn new(builder: &mut Builder) -> Self {
        let mut attributes = Self {
            inner_types: core::array::from_fn(|_| builder.add_virtual_target()),
            inner_values: core::array::from_fn(|_| builder.add_virtual_target()),
            values: ATTR_NIL_VALUES.map(|v| builder.constant(v)),
        };
        attributes.prepare(builder);
        attributes
    }

    /// Do not aggregate attributes' hash if they don't exist.
    pub fn aggregate_tx_hash(
        &self,
        builder: &mut Builder,
        tx_hash: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget {
        let combined_hash = self.get_combined_hash(builder, tx_hash);
        let is_empty = self.is_empty(builder);
        builder.select_quintic_ext(is_empty, tx_hash, combined_hash)
    }

    fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        // ATTRIBUTE_MAX_VALUES added together doesn't overflow Goldilocks
        let should_be_zero = builder.add_many(
            self.inner_types
                .iter()
                .chain(self.inner_values.iter())
                .cloned()
                .collect::<Vec<_>>(),
        );
        builder.is_zero(should_be_zero)
    }

    fn get_combined_hash(
        &self,
        builder: &mut Builder,
        tx_hash: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget {
        let attributes_hash = hash_to_quintic_extension_circuit(
            builder,
            &self
                .inner_types
                .iter()
                .zip(self.inner_values.iter())
                .flat_map(|(t, v)| [*t, *v])
                .collect::<Vec<_>>(),
        );
        hash_to_quintic_extension_circuit(
            builder,
            &tx_hash
                .0
                .iter()
                .chain(attributes_hash.0.iter())
                .cloned()
                .collect::<Vec<_>>(),
        )
    }

    /// Check sanity of sent attributes and set the inner attributes array
    fn prepare(&mut self, builder: &mut Builder) {
        self.verify_attribute_types(builder);

        self.set_attributes_array(builder);

        self.range_check_attribute_values(builder);

        self.validate_attribute_types(builder);
    }

    fn verify_attribute_types(&self, builder: &mut Builder) {
        let max_attribute_type = builder.constant_usize(TOTAL_ATTRIBUTE_COUNT - 1);
        let mut last_type = self.inner_types[0];
        for i in 0..NB_ATTRIBUTES_PER_TX {
            let (t, v) = (self.inner_types[i], self.inner_values[i]);

            builder.register_range_check(t, ATTRIBUTE_TYPE_BITS);
            builder.assert_lte(t, max_attribute_type, ATTRIBUTE_TYPE_BITS);

            let is_nil_type = builder.is_zero(t);
            builder.conditional_assert_zero(is_nil_type, v);

            if i == 0 {
                continue;
            }

            let is_type_increasing = builder.is_lt(last_type, t, ATTRIBUTE_TYPE_BITS);
            let should_be_true = builder.or(is_type_increasing, is_nil_type);
            builder.assert_true(should_be_true);

            last_type = t;
        }
    }

    fn set_attributes_array(&mut self, builder: &mut Builder) {
        for (t, v) in self.inner_types.iter().zip(self.inner_values.iter()) {
            for i in 1..TOTAL_ATTRIBUTE_COUNT {
                let is_type = builder.is_equal_constant(*t, i as u64);
                self.values[i] = builder.select(is_type, *v, self.values[i]);
            }
        }
    }

    fn range_check_attribute_values(&self, builder: &mut Builder) {
        for i in 1..TOTAL_ATTRIBUTE_COUNT {
            builder.register_range_check(self.values[i], *ATTR_BIT_SIZES.get(&i).unwrap());
            let max_val = builder.constant_usize(*ATTR_MAX_VALUES.get(&i).unwrap());
            builder.assert_lte(self.values[i], max_val, *ATTR_BIT_SIZES.get(&i).unwrap());
        }
    }

    fn validate_attribute_types(&self, builder: &mut Builder) {
        let is_taker_fee_nil = builder.is_equal_f(
            self.get(ATTR_INTEGRATOR_TAKER_FEE),
            ATTR_NIL_VALUES[ATTR_INTEGRATOR_TAKER_FEE],
        );
        let is_maker_fee_nil = builder.is_equal_f(
            self.get(ATTR_INTEGRATOR_MAKER_FEE),
            ATTR_NIL_VALUES[ATTR_INTEGRATOR_MAKER_FEE],
        );
        let is_integrator_index_nil = builder.is_equal_f(
            self.get(ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX),
            ATTR_NIL_VALUES[ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX],
        );
        let is_self_trade_behavior_mode_nil = builder.is_equal_f(
            self.get(ATTR_SELF_TRADE_BEHAVIOR_MODE),
            ATTR_NIL_VALUES[ATTR_SELF_TRADE_BEHAVIOR_MODE],
        );
        let is_self_trade_equality_mode_nil = builder.is_equal_f(
            self.get(ATTR_SELF_TRADE_EQUALITY_MODE),
            ATTR_NIL_VALUES[ATTR_SELF_TRADE_EQUALITY_MODE],
        );

        // Disallow integrator fees if integrator index is not set
        let is_both_fees_nil = builder.and(is_taker_fee_nil, is_maker_fee_nil);
        let should_be_false = builder.and_not(is_integrator_index_nil, is_both_fees_nil);
        builder.assert_false(should_be_false);

        // Disallow self-trade specifications if integrator index is set
        let is_self_trade_modes_nil = builder.and(
            is_self_trade_behavior_mode_nil,
            is_self_trade_equality_mode_nil,
        );
        let should_be_true = builder.or(is_integrator_index_nil, is_self_trade_modes_nil);
        builder.assert_true(should_be_true);

        // Disallow Reduce mode with master account index equality mode
        let is_master_account_index_equality_mode = builder.is_equal_constant(
            self.get(ATTR_SELF_TRADE_EQUALITY_MODE),
            SELF_TRADE_EQUALITY_MASTER_ACCOUNT_INDEX,
        );
        let is_reduce_behavior_mode = builder.is_equal_constant(
            self.get(ATTR_SELF_TRADE_BEHAVIOR_MODE),
            SELF_TRADE_BEHAVIOR_REDUCE,
        );
        let should_be_false = builder.and(
            is_master_account_index_equality_mode,
            is_reduce_behavior_mode,
        );
        builder.assert_false(should_be_false);
    }

    pub fn sanitize_and_normalize(
        &mut self,
        builder: &mut Builder,
        account: &AccountTarget,
        market: &MarketTarget,
        system_config: &SystemConfigTarget,
        block_created_at: Target,
    ) {
        let is_enabled = {
            let is_nil_integrator_index = builder.is_equal_f(
                self.get(ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX),
                ATTR_NIL_VALUES[ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX],
            );
            let is_not_nil_integrator_index = builder.not(is_nil_integrator_index);

            let is_empty_market = market.is_empty(builder);
            let is_not_empty_market = builder.not(is_empty_market);

            builder.multi_and(&[is_not_nil_integrator_index, is_not_empty_market])
        };

        // Integrator must be approved by the account, and fees sent must be within the integrator's limits
        let is_perps = builder.is_equal_constant(market.market_type, MARKET_TYPE_PERPS);
        let taker_cap = builder.select(
            // Will be zero if system config is empty
            is_perps,
            system_config.max_integrator_perps_taker_fee,
            system_config.max_integrator_spot_taker_fee,
        );
        let maker_cap = builder.select(
            // Will be zero if system config is empty
            is_perps,
            system_config.max_integrator_perps_maker_fee,
            system_config.max_integrator_spot_maker_fee,
        );
        let mut sanitized = builder.not(is_enabled);
        for i in 0..MAX_APPROVED_INTEGRATORS {
            let is_integrator = builder.is_equal(
                account.approved_integrators[i].integrator_account_index,
                self.get(ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX),
            );
            let flag = builder.and_not(is_integrator, sanitized);
            sanitized = builder.or(sanitized, flag);

            // Not expired
            builder.conditional_assert_lt(
                flag,
                block_created_at,
                account.approved_integrators[i].expiry,
                TIMESTAMP_BITS,
            );

            // Taker and maker fees
            let taker_fee = builder.select(
                is_perps,
                account.approved_integrators[i].max_perps_taker_fee,
                account.approved_integrators[i].max_spot_taker_fee,
            );
            let maker_fee = builder.select(
                is_perps,
                account.approved_integrators[i].max_perps_maker_fee,
                account.approved_integrators[i].max_spot_maker_fee,
            );
            for (attr_type, target, cap) in [
                (ATTR_INTEGRATOR_TAKER_FEE, taker_fee, taker_cap),
                (ATTR_INTEGRATOR_MAKER_FEE, maker_fee, maker_cap),
            ] {
                // First assert over given value, then cap
                let fee_bit_size = *ATTR_BIT_SIZES.get(&attr_type).unwrap();
                let fee_value = self.get(attr_type);
                builder.conditional_assert_lte(flag, fee_value, target, fee_bit_size);

                let capped_fee = builder.min(&[fee_value, cap], fee_bit_size);
                self.set(builder, flag, attr_type, capped_fee);
            }
        }
        builder.conditional_assert_true(is_enabled, sanitized);
    }

    pub fn get_register_generic_fields(
        &self,
        builder: &mut Builder,
    ) -> (
        Target, // generic_field_1
        Target, // generic_field_2
        Target, // generic_field_3
    ) {
        let is_integrator_fee_disabled =
            is_integrator_fee_disabled(builder, self.get(ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX));
        let order_flags = OrderFlags {
            self_trade_behavior_mode: self.get(ATTR_SELF_TRADE_BEHAVIOR_MODE),
            self_trade_equality_mode: self.get(ATTR_SELF_TRADE_EQUALITY_MODE),
        }
        .to_target(builder);

        (
            self.get(ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX),
            builder.select(
                is_integrator_fee_disabled,
                order_flags,
                self.get(ATTR_INTEGRATOR_TAKER_FEE),
            ),
            self.get(ATTR_INTEGRATOR_MAKER_FEE),
        )
    }
}

pub trait TxAttributesTargetWitness<F: PrimeField64> {
    fn set_attributes_tx_target(&mut self, a: &TxAttributesTarget, b: &TxAttributes) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> TxAttributesTargetWitness<F> for T {
    fn set_attributes_tx_target(&mut self, a: &TxAttributesTarget, b: &TxAttributes) -> Result<()> {
        for i in 0..NB_ATTRIBUTES_PER_TX {
            self.set_target(a.inner_types[i], F::from_canonical_u8(b.attribute_types[i]))?;
            self.set_target(
                a.inner_values[i],
                F::from_canonical_i64(b.attribute_values[i]),
            )?;
        }

        Ok(())
    }
}

pub fn is_integrator_fee_disabled(
    builder: &mut Builder,
    integrator_fee_collector_index: Target,
) -> BoolTarget {
    builder.is_equal_f(
        integrator_fee_collector_index,
        ATTR_NIL_VALUES[ATTR_INTEGRATOR_FEE_COLLECTOR_INDEX],
    )
}
