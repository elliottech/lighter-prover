// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::biguint_u16::CircuitBuilderBiguint16;
use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::matching_engine::release_closed_market_slot_if_drained;
use crate::tx_interface::{Apply, Verify};
use crate::types::account::AccountTarget;
use crate::types::binary_options_position::{
    BinaryOptionsPositionTarget, select_binary_options_position_target,
};
use crate::types::config::{BIG_U128_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct InternalSettleBinaryOptionsPositionTx {
    #[serde(rename = "a")]
    pub account_index: i64,
    #[serde(rename = "m")]
    pub market_index: i16,
}

#[derive(Debug, Clone)]
pub struct InternalSettleBinaryOptionsPositionTxTarget {
    pub account_index: Target,
    pub market_index: Target,

    // Output
    success: BoolTarget,
}

impl InternalSettleBinaryOptionsPositionTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),

            success: BoolTarget::default(),
        }
    }
}

impl Verify for InternalSettleBinaryOptionsPositionTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_internal_settle_binary_options_position;
        self.success = is_enabled;

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_false(is_enabled, tx_state.is_new_account[OWNER_ACCOUNT_ID]);
        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.register_stack[0].instruction_type,
            EXECUTE_TRANSACTION as u64,
        );

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_BINARY_OPTIONS,
        );
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.status,
            MARKET_STATUS_IN_SETTLEMENT as u64,
        );

        // A position is settled only once it has no resting order left
        builder.conditional_assert_zero(
            is_enabled,
            tx_state.binary_options_positions[OWNER_ACCOUNT_ID].total_order_count,
        );

        // Positions are settled against the USDC balance
        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
            USDC_ASSET_INDEX,
        );
        let is_usdc_asset_empty = tx_state.assets[USDC_BASE_ASSET_ID].is_empty(builder);
        builder.conditional_assert_false(is_enabled, is_usdc_asset_empty);
    }
}

impl Apply for InternalSettleBinaryOptionsPositionTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let position = tx_state.binary_options_positions[OWNER_ACCOUNT_ID].clone();

        let position_size_abs = builder.biguint_u16_to_target(&position.size.abs);
        let is_position_open = builder.is_not_zero(position_size_abs);
        let has_position = builder.and(self.success, is_position_open);

        // YES positions receive the settlement price per share, NO positions the cap minus the
        // settlement price. For discrete markets this is the full payout for the winning side.
        let is_no_position = builder.is_sign_negative(position.size.sign);
        let no_payout_price = builder.sub(
            tx_state.market.settlement_cap,
            tx_state.market.settlement_price,
        );
        let payout_price_per_share = builder.select(
            is_no_position,
            no_payout_price,
            tx_state.market.settlement_price,
        );

        let payout_price_big = builder.target_to_biguint(payout_price_per_share);
        let quote_multiplier_big =
            builder.target_to_biguint(tx_state.market.quote_extension_multiplier);
        let extended_payout_price =
            builder.mul_biguint_non_carry(&payout_price_big, &quote_multiplier_big, BIG_U128_LIMBS);
        let position_size_abs_big = builder.target_to_biguint(position_size_abs);
        let payout = builder.mul_biguint_non_carry(
            &position_size_abs_big,
            &extended_payout_price,
            BIG_U128_LIMBS,
        );
        let payout = builder.biguint_to_bigint(&payout);

        let _spot = builder.constant_u64(PRODUCT_TYPE_SPOT);
        let is_account_unified = tx_state.accounts[OWNER_ACCOUNT_ID].is_unified_mode();
        let _false = builder._false();
        _ = AccountTarget::apply_asset_delta(
            builder,
            has_position,
            _spot,
            tx_state.asset_indices[USDC_BASE_ASSET_ID],
            &mut tx_state.margined_asset[USDC_BASE_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID],
            &payout,
            is_account_unified,
            _false,
            &mut tx_state.account_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
            &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][USDC_BASE_ASSET_ID].balance,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
            false,
        );

        // Clear the settled position and release its open interest
        let nil_public_market_index = builder.constant_u64(NIL_PUBLIC_MARKET_INDEX as u64);
        let empty_position = BinaryOptionsPositionTarget::empty(builder, nil_public_market_index);
        tx_state.binary_options_positions[OWNER_ACCOUNT_ID] = select_binary_options_position_target(
            builder,
            has_position,
            &empty_position,
            &tx_state.binary_options_positions[OWNER_ACCOUNT_ID],
        );

        builder.conditional_assert_lte(
            has_position,
            position_size_abs,
            tx_state.market.open_interest,
            MARKET_OPEN_INTEREST_BITS,
        );
        let new_open_interest = builder.sub(tx_state.market.open_interest, position_size_abs);
        tx_state.market.open_interest = builder.select(
            has_position,
            new_open_interest,
            tx_state.market.open_interest,
        );

        release_closed_market_slot_if_drained(builder, self.success, tx_state);

        self.success
    }
}

pub trait InternalSettleBinaryOptionsPositionTxTargetWitness<F: PrimeField64> {
    fn set_internal_settle_binary_options_position_tx_target(
        &mut self,
        a: &InternalSettleBinaryOptionsPositionTxTarget,
        b: &InternalSettleBinaryOptionsPositionTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> InternalSettleBinaryOptionsPositionTxTargetWitness<F> for T {
    fn set_internal_settle_binary_options_position_tx_target(
        &mut self,
        a: &InternalSettleBinaryOptionsPositionTxTarget,
        b: &InternalSettleBinaryOptionsPositionTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.market_index, F::from_canonical_i64(b.market_index as i64))?;

        Ok(())
    }
}
