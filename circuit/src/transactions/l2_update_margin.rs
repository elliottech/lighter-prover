// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use num::BigUint;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::CircuitBuilderBigIntU16;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::liquidation::{
    BoolOrTarget, get_available_asset_balance, get_available_usdc_collateral,
};
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct L2UpdateMarginTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "u")]
    #[serde(deserialize_with = "deserializers::int_to_biguint")]
    pub usdc_amount: BigUint, // 60 bits

    #[serde(rename = "d")]
    pub direction: u8,
}

#[derive(Debug)]
pub struct L2UpdateMarginTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub market_index: Target,
    pub usdc_amount: BigUintTarget, // 60 bits
    pub direction: Target,

    // helpers
    colleteral_to_move: BigUintTarget, // 96 bits

    // output
    success: BoolTarget,
}

impl L2UpdateMarginTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdateMarginTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            market_index: builder.add_virtual_target(),
            usdc_amount: builder.add_virtual_biguint_target_safe(BIG_U64_LIMBS),
            direction: builder.add_virtual_target(),

            // helpers
            colleteral_to_move: BigUintTarget::default(),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateMarginTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_MARGIN)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.market_index,
        ];

        let mut limbs = self.usdc_amount.limbs.clone();
        limbs.resize(BIG_U64_LIMBS, builder.zero_u32());
        for limb in limbs {
            elements.push(limb.0);
        }

        elements.push(self.direction);

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateMarginTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_margin;
        self.success = is_enabled;

        // Check account/apikey and market consistency
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

        builder.conditional_assert_eq(is_enabled, self.market_index, tx_state.market.market_index);
        builder.conditional_assert_eq(
            is_enabled,
            self.market_index,
            tx_state.market.perps_market_index,
        );

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        // Valid values are 0(REMOVE_MARGIN) or 1(ADD_MARGIN)
        builder.assert_bool(BoolTarget::new_unsafe(self.direction));

        let active_market_status = builder.constant_from_u8(MARKET_STATUS_ACTIVE);
        builder.conditional_assert_eq(
            is_enabled,
            tx_state.market_details.status,
            active_market_status,
        );

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.market.market_type,
            MARKET_TYPE_PERPS,
        );

        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
        let is_margin_isolated = builder.is_equal(
            tx_state.positions[OWNER_ACCOUNT_ID].margin_mode,
            isolated_margin_mode,
        );
        builder.conditional_assert_true(is_enabled, is_margin_isolated);

        let is_position_zero =
            builder.is_zero_bigint_u16(&tx_state.positions[OWNER_ACCOUNT_ID].position);
        builder.conditional_assert_false(is_enabled, is_position_zero);

        // Transfer amount checks - not zero and 60 bits in total
        builder.conditional_assert_not_zero_biguint(is_enabled, &self.usdc_amount);
        builder.range_check_biguint(&self.usdc_amount, MAX_TRANSFER_BITS);

        let usdc_to_collateral_multiplier =
            BigUintTarget::from(builder.constant_u32(USDC_TO_COLLATERAL_MULTIPLIER));
        self.colleteral_to_move = builder.mul_biguint_non_carry(
            &self.usdc_amount,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        );

        let _perps = builder.constant_u64(PRODUCT_TYPE_PERPS);
        let available_cross_collateral = get_available_asset_balance(
            builder,
            _perps,
            tx_state.asset_indices[TX_ASSET_ID],
            &tx_state.accounts[OWNER_ACCOUNT_ID],
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            tx_state.is_asset_used_as_margin[OWNER_ACCOUNT_ID][TX_ASSET_ID],
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].cross_risk_parameters,
            &tx_state.margined_asset[TX_ASSET_ID],
            &tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
            BoolOrTarget::False,
        );
        let available_isolated_collateral = get_available_usdc_collateral(
            builder,
            &tx_state.risk_infos[OWNER_ACCOUNT_ID].current_risk_parameters,
        );

        let available_collateral = builder.select_biguint(
            BoolTarget::new_unsafe(self.direction),
            &available_cross_collateral,
            &available_isolated_collateral,
        );

        // Check that we have enough collateral to move
        builder.conditional_assert_lte_biguint(
            is_enabled,
            &self.colleteral_to_move,
            &available_collateral,
        );
    }
}

impl Apply for L2UpdateMarginTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let one = builder.one();
        let neg_one = builder.neg_one();

        let collateral_to_move_isolated = BigIntTarget {
            abs: self.colleteral_to_move.clone(),
            sign: SignTarget::new_unsafe(builder.select(
                BoolTarget::new_unsafe(self.direction),
                one,
                neg_one,
            )), // self.colleteral_to_move can't be zero
        };

        let new_allocated_margin = builder.add_bigint_non_carry(
            &tx_state.positions[OWNER_ACCOUNT_ID].allocated_margin,
            &collateral_to_move_isolated,
            BIG_U96_LIMBS,
        );
        tx_state.positions[OWNER_ACCOUNT_ID].allocated_margin = builder.select_bigint(
            self.success,
            &new_allocated_margin,
            &tx_state.positions[OWNER_ACCOUNT_ID].allocated_margin,
        );

        let collateral_diff = builder.neg_bigint(&collateral_to_move_isolated);
        tx_state.accounts[OWNER_ACCOUNT_ID].apply_collateral_delta(
            builder,
            self.success,
            &collateral_diff,
            &mut tx_state.strategies[OWNER_ACCOUNT_ID],
            &mut tx_state.account_margined_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );

        self.success
    }
}

pub trait L2UpdateMarginTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_margin_tx_target(
        &mut self,
        a: &L2UpdateMarginTxTarget,
        b: &L2UpdateMarginTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateMarginTxTargetWitness<F> for T {
    fn set_l2_update_margin_tx_target(
        &mut self,
        a: &L2UpdateMarginTxTarget,
        b: &L2UpdateMarginTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_biguint_target(&a.usdc_amount, &b.usdc_amount)?;
        self.set_target(a.direction, F::from_canonical_u8(b.direction))?;

        Ok(())
    }
}
