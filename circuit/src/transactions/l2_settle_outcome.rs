// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::matching_engine::release_closed_market_slot_if_drained;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2SettleOutcomeTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "mi")]
    pub market_index: i16,
    #[serde(rename = "pmi")]
    pub public_market_index: i64,
    #[serde(rename = "sp")]
    pub settlement_price: u32,
    #[serde(rename = "ir")]
    pub is_refund: u8,
}

#[derive(Debug)]
pub struct L2SettleOutcomeTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub market_index: Target,
    pub public_market_index: Target, // 48 bits
    pub settlement_price: Target,
    pub is_refund: Target,

    // Output
    success: BoolTarget,
}

impl L2SettleOutcomeTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            public_market_index: builder.add_virtual_target(),
            settlement_price: builder.add_virtual_target(),
            is_refund: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2SettleOutcomeTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_SETTLE_OUTCOME)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
            self.public_market_index,
            self.settlement_price,
            self.is_refund,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2SettleOutcomeTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_settle_outcome;
        self.success = is_enabled;

        builder.register_range_check(self.settlement_price, ORDER_PRICE_BITS);
        builder.conditional_assert_bool(is_enabled, BoolTarget::new_unsafe(self.is_refund));

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );

        // The market is addressed by its binary options slot and its public market index;
        // an active binary options market always has one assigned
        let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.market_index, nil_market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.binary_options_market_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.public_market_index,
            tx_state.market.public_market_index,
        );
        let nil_public_market_index = builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64);
        builder.conditional_assert_not_eq(
            is_enabled,
            self.public_market_index,
            nil_public_market_index,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_BINARY_OPTIONS,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.status,
            MARKET_STATUS_ACTIVE as u64,
        );

        // Only the market operator can settle the outcome
        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.market.market_operator_account_index,
        );

        let is_refund = BoolTarget::new_unsafe(self.is_refund);

        // Refunds must not carry an explicit price; the market settles at its default price,
        // which may be interior to [0, cap] even for discrete markets
        let refund_flag = builder.and(is_enabled, is_refund);
        builder.conditional_assert_zero(refund_flag, self.settlement_price);

        let non_refund_flag = builder.and_not(is_enabled, is_refund);
        builder.conditional_assert_lte(
            non_refund_flag,
            self.settlement_price,
            tx_state.market.settlement_cap,
            ORDER_PRICE_BITS,
        );

        // Discrete markets settle at 0 or the settlement cap only
        let is_discrete = builder.is_equal_constant(
            tx_state.market.settlement_type,
            SETTLEMENT_TYPE_DISCRETE as u64,
        );
        let is_price_zero = builder.is_zero(self.settlement_price);
        let is_price_cap = builder.is_equal(self.settlement_price, tx_state.market.settlement_cap);
        let is_price_zero_or_cap = builder.or(is_price_zero, is_price_cap);
        let discrete_non_refund_flag = builder.and(non_refund_flag, is_discrete);
        builder.conditional_assert_true(discrete_non_refund_flag, is_price_zero_or_cap);
    }
}

impl Apply for L2SettleOutcomeTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let is_refund = BoolTarget::new_unsafe(self.is_refund);
        let effective_settlement_price = builder.select(
            is_refund,
            tx_state.market.default_price,
            self.settlement_price,
        );

        // Settling at the cap is a YES outcome, at zero a NO outcome, anything in between NONE
        let is_yes_outcome =
            builder.is_equal(effective_settlement_price, tx_state.market.settlement_cap);
        let is_no_outcome = builder.is_zero(effective_settlement_price);
        let outcome_none = builder.constant_u64(MARKET_OUTCOME_NONE as u64);
        let outcome_no = builder.constant_u64(MARKET_OUTCOME_NO as u64);
        let outcome_yes = builder.constant_u64(MARKET_OUTCOME_YES as u64);
        let outcome = builder.select(is_no_outcome, outcome_no, outcome_none);
        let outcome = builder.select(is_yes_outcome, outcome_yes, outcome);

        let status_in_settlement = builder.constant_u64(MARKET_STATUS_IN_SETTLEMENT as u64);
        tx_state.market.settlement_price = builder.select(
            self.success,
            effective_settlement_price,
            tx_state.market.settlement_price,
        );
        tx_state.market.outcome = builder.select(self.success, outcome, tx_state.market.outcome);
        tx_state.market.status =
            builder.select(self.success, status_in_settlement, tx_state.market.status);

        // A completely empty market has nothing to settle; it is terminal right away, release its slot
        release_closed_market_slot_if_drained(builder, self.success, tx_state);

        self.success
    }
}

pub trait L2SettleOutcomeTxTargetWitness<F: PrimeField64> {
    fn set_l2_settle_outcome_tx_target(
        &mut self,
        a: &L2SettleOutcomeTxTarget,
        b: &L2SettleOutcomeTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2SettleOutcomeTxTargetWitness<F> for T {
    fn set_l2_settle_outcome_tx_target(
        &mut self,
        a: &L2SettleOutcomeTxTarget,
        b: &L2SettleOutcomeTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;
        self.set_target(
            a.public_market_index,
            F::from_canonical_i64(b.public_market_index),
        )?;
        self.set_target(
            a.settlement_price,
            F::from_canonical_u32(b.settlement_price),
        )?;
        self.set_target(a.is_refund, F::from_canonical_u8(b.is_refund))?;

        Ok(())
    }
}
