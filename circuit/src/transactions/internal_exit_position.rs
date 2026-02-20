// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::CircuitBuilderBiguint16;
use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::liquidation::get_funding_delta_for_position_and_market;
use crate::tx_interface::{Apply, Verify};
use crate::types::account_position::{AccountPositionTarget, get_position_unrealized_pnl};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::{
    EMPTY_ORDER_BOOK_TREE_ROOT, EXECUTE_TRANSACTION, MARKET_STATUS_EXPIRED, MARKET_TYPE_PERPS,
    OWNER_ACCOUNT_ID, USDC_TO_COLLATERAL_MULTIPLIER,
};
use crate::types::market::{MarketTarget, select_market};
use crate::types::market_details::{MarketDetailsTarget, select_market_details};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct InternalExitPositionTx {
    #[serde(rename = "a")]
    pub account_index: i64,

    #[serde(rename = "m")]
    pub market_index: u16,
}

#[derive(Debug, Clone)]
pub struct InternalExitPositionTxTarget {
    pub account_index: Target,
    pub market_index: Target,

    // Output
    success: BoolTarget,
}

impl InternalExitPositionTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),

            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalExitPositionTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_exit_position;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );

        let execute_transaction_type = builder.constant(F::from_canonical_u8(EXECUTE_TRANSACTION));
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            execute_transaction_type,
        );

        let market_expired_status = builder.constant(F::from_canonical_u8(MARKET_STATUS_EXPIRED));
        builder.conditional_assert_eq(is_enabled, tx_state.market.status, market_expired_status);

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );
    }
}

impl Apply for InternalExitPositionTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let funding_delta = get_funding_delta_for_position_and_market(
            builder,
            &tx_state.positions[OWNER_ACCOUNT_ID],
            &tx_state.market_details,
        );

        // Calculate unrealized pnl
        let position_abs =
            builder.biguint_u16_to_target(&tx_state.positions[OWNER_ACCOUNT_ID].position.abs);
        let position_value = get_position_unrealized_pnl(
            builder,
            &tx_state.market_details,
            position_abs,
            tx_state.positions[OWNER_ACCOUNT_ID].position.sign,
            tx_state.positions[OWNER_ACCOUNT_ID].entry_quote,
        );
        let position_value_big = builder.signed_target_to_bigint(position_value);
        let usdc_to_collateral_multiplier =
            builder.constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        let unrealized_pnl = builder.mul_bigint_with_biguint_non_carry(
            &position_value_big,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        // Apply collateral deltas to account
        let collateral_delta =
            builder.add_bigint_non_carry(&funding_delta, &unrealized_pnl, BIG_U96_LIMBS);
        // Move allocated margin back to the account
        let allocated_margin = tx_state.positions[OWNER_ACCOUNT_ID]
            .allocated_margin
            .clone();
        let collateral_delta =
            builder.add_bigint_non_carry(&collateral_delta, &allocated_margin, BIG_U96_LIMBS);
        tx_state.accounts[OWNER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &collateral_delta,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
        );

        // Update market details and order book
        let new_open_interest = builder.sub(tx_state.market_details.open_interest, position_abs);
        tx_state.market_details.open_interest = builder.select(
            self.success,
            new_open_interest,
            tx_state.market_details.open_interest,
        );

        let is_market_has_no_order = builder.is_zero(tx_state.market.total_order_count);
        let is_market_has_no_position = builder.is_zero(tx_state.market_details.open_interest);
        let is_expired_market_is_empty_and_enabled = builder.multi_and(&[
            self.success,
            is_market_has_no_order,
            is_market_has_no_position,
        ]);
        let empty_market_details = MarketDetailsTarget::empty(builder);
        tx_state.market_details = select_market_details(
            builder,
            is_expired_market_is_empty_and_enabled,
            &empty_market_details,
            &tx_state.market_details,
        );
        let empty_order_book_tree_root = builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT);
        let empty_order_book = MarketTarget::empty(
            builder,
            tx_state.market.market_index,
            tx_state.market.perps_market_index,
            empty_order_book_tree_root,
        );
        tx_state.market = select_market(
            builder,
            is_expired_market_is_empty_and_enabled,
            &empty_order_book,
            &tx_state.market,
        );

        let empty_position = AccountPositionTarget::empty(builder);

        tx_state.positions[OWNER_ACCOUNT_ID] = AccountPositionTarget::select_position(
            builder,
            self.success,
            &empty_position,
            &tx_state.positions[OWNER_ACCOUNT_ID],
        );

        self.success
    }
}

pub trait InternalExitPositionTxTargetWitness<F: PrimeField64> {
    fn set_internal_exit_position_tx_target(
        &mut self,
        a: &InternalExitPositionTxTarget,
        b: &InternalExitPositionTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalExitPositionTxTargetWitness<F> for T {
    fn set_internal_exit_position_tx_target(
        &mut self,
        a: &InternalExitPositionTxTarget,
        b: &InternalExitPositionTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;

        Ok(())
    }
}
