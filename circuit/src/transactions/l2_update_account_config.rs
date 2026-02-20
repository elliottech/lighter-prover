// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1
//
// Proven statements in this circuit module:
// 1. only valid accounts (master/sub) can switch balance modes.
// 2. switching to unified requires no locked USDC.
// 3. switching to simple requires healthy account, no locked USDC, and non-negative collateral-with-funding.
// 4. switching to unified applies USDC-to-collateral updates.

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;

#[derive(Debug, Clone, Deserialize, Default)]
// A transaction that updates the `AccountConfig`
pub struct L2UpdateAccountConfigTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki", default)]
    pub api_key_index: u8,

    // "dtm" kept for consistency with existing witness tags.
    #[serde(rename = "dtm")]
    pub account_trading_mode: u8,
}

#[derive(Debug)]
pub struct L2UpdateAccountConfigTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub account_trading_mode: Target,
    // output
    pub success: BoolTarget,
}

impl L2UpdateAccountConfigTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L2UpdateAccountConfigTxTarget {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            account_trading_mode: builder.add_virtual_target(),

            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2UpdateAccountConfigTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = [
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_ACCOUNT_CONFIG)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.account_trading_mode,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2UpdateAccountConfigTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_update_account_config;
        self.success = is_enabled;

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

        builder.conditional_assert_eq_constant(
            is_enabled,
            tx_state.asset_indices[TX_ASSET_ID],
            USDC_ASSET_INDEX,
        );

        // Valid values: 0 (simple assets) or 1 (unified asset)
        builder.assert_bool(BoolTarget::new_unsafe(self.account_trading_mode));

        // Master and sub accounts can update their account config
        let is_master = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            MASTER_ACCOUNT_TYPE as u64,
        );
        let is_sub = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            SUB_ACCOUNT_TYPE as u64,
        );
        let is_master_or_sub = builder.or(is_master, is_sub);
        builder.conditional_assert_true(is_enabled, is_master_or_sub);

        // Balance mode should change
        let is_mode_same = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode,
            self.account_trading_mode,
        );
        builder.conditional_assert_false(is_enabled, is_mode_same);

        // =========================================
        // statement 3: switching to simple requires healthy account
        // and non-negative collateral-with-funding.
        // =========================================
        let is_simple_mode = builder.is_equal_constant(
            self.account_trading_mode,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE as u64,
        );
        let is_enabled_simple = builder.and(is_enabled, is_simple_mode);

        // No locked USDC balance, means no open spot orders
        builder.conditional_assert_zero_biguint(
            is_enabled_simple,
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].locked_balance,
        );

        let is_healthy = tx_state.risk_infos[OWNER_ACCOUNT_ID]
            .cross_risk_parameters
            .is_healthy(builder);
        builder.conditional_assert_true(is_enabled_simple, is_healthy);

        let collateral_with_funding_negative = builder.is_sign_negative(
            tx_state.risk_infos[OWNER_ACCOUNT_ID]
                .cross_risk_parameters
                .collateral_with_funding
                .sign,
        );
        builder.conditional_assert_false(is_enabled_simple, collateral_with_funding_negative);
        // === end of statement 3 ===
    }
}

impl Apply for L2UpdateAccountConfigTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let zero_biguint = builder.zero_biguint();

        tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode = builder.select(
            self.success,
            self.account_trading_mode,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode,
        );

        let is_unified_mode = builder.is_equal_constant(
            self.account_trading_mode,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED as u64,
        );
        let apply_unify = builder.and(self.success, is_unified_mode);

        // =========================================
        // statement 4: switching to unified applies USDC-to-collateral updates.
        // =========================================
        // Unify: collateral += usdc_balance, usdc_balance = zero
        let mut usdc_balance = tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID]
            .balance
            .clone();
        usdc_balance = builder.mul_biguint_by_bool(&usdc_balance, apply_unify);
        let usdc_balance_delta = builder.biguint_to_bigint(&usdc_balance);
        let new_collateral = builder.add_bigint_non_carry(
            &tx_state.accounts[OWNER_ACCOUNT_ID].collateral,
            &usdc_balance_delta,
            BIG_U96_LIMBS,
        );
        tx_state.accounts[OWNER_ACCOUNT_ID].collateral = builder.select_bigint(
            apply_unify,
            &new_collateral,
            &tx_state.accounts[OWNER_ACCOUNT_ID].collateral,
        );

        tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance = builder.select_biguint(
            apply_unify,
            &zero_biguint,
            &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
        );
        // === end of statement 4 ===

        self.success
    }
}

pub trait L2UpdateAccountConfigTxTargetWitness<F: PrimeField64> {
    fn set_l2_update_account_config_tx_target(
        &mut self,
        a: &L2UpdateAccountConfigTxTarget,
        b: &L2UpdateAccountConfigTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2UpdateAccountConfigTxTargetWitness<F> for T {
    fn set_l2_update_account_config_tx_target(
        &mut self,
        a: &L2UpdateAccountConfigTxTarget,
        b: &L2UpdateAccountConfigTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.account_trading_mode,
            F::from_canonical_u8(b.account_trading_mode),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
    use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
    use crate::bool_utils::CircuitBuilderBoolUtils;
    use crate::types::account_position::AccountPositionTarget;
    use crate::types::config::{C, CIRCUIT_CONFIG};
    use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};

    fn u96_biguint(builder: &mut Builder, value: u64) -> BigUintTarget {
        assert!(
            u32::try_from(value).is_ok(),
            "test value must fit into a u32"
        );
        BigUintTarget {
            limbs: vec![
                U32Target(builder.constant_u64(value)),
                builder.zero_u32(),
                builder.zero_u32(),
            ],
        }
    }

    fn positive_bigint(builder: &mut Builder, value: u64) -> BigIntTarget {
        BigIntTarget {
            abs: u96_biguint(builder, value),
            sign: SignTarget::new_unsafe(builder.one()),
        }
    }

    fn base_tx_state(
        builder: &mut Builder,
        account_type: u8,
        old_account_trading_mode: u8,
        tx_asset_index: u64,
        usdc_balance: u64,
        usdc_locked: u64,
    ) -> TxState {
        let mut tx_state = TxState::default();

        tx_state.accounts[OWNER_ACCOUNT_ID].account_index = builder.constant_i64(42);
        tx_state.accounts[OWNER_ACCOUNT_ID].account_type =
            builder.constant_u64(account_type as u64);
        tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode =
            builder.constant_u64(old_account_trading_mode as u64);
        tx_state.accounts[OWNER_ACCOUNT_ID].collateral = positive_bigint(builder, 5);
        tx_state.accounts[OWNER_ACCOUNT_ID].total_order_count = builder.zero();
        tx_state.accounts[OWNER_ACCOUNT_ID].total_non_cross_order_count = builder.zero();
        for i in 0..POSITION_LIST_SIZE {
            tx_state.accounts[OWNER_ACCOUNT_ID].positions[i] =
                AccountPositionTarget::empty(builder);
        }

        tx_state.api_key.api_key_index = builder.constant_u64(7);
        tx_state.asset_indices[TX_ASSET_ID] = builder.constant_u64(tx_asset_index);
        tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance =
            u96_biguint(builder, usdc_balance);
        tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].locked_balance =
            u96_biguint(builder, usdc_locked);

        tx_state
    }

    fn prove_case(
        account_type: u8,
        old_account_trading_mode: u8,
        new_account_trading_mode: u8,
        tx_asset_index: u64,
        usdc_balance: u64,
        usdc_locked: u64,
        with_apply_assertions: bool,
        expected_collateral: u64,
        expected_usdc_balance: u64,
    ) -> bool {
        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let tx_type_value = builder.constant_u64(TX_TYPE_L2_UPDATE_ACCOUNT_CONFIG as u64);
        let tx_type = TxTypeTargets::new(&mut builder, tx_type_value);
        let mut tx_state = base_tx_state(
            &mut builder,
            account_type,
            old_account_trading_mode,
            tx_asset_index,
            usdc_balance,
            usdc_locked,
        );
        let mut tx_target = L2UpdateAccountConfigTxTarget::new(&mut builder);

        tx_target.verify(&mut builder, &tx_type, &tx_state);
        tx_target.apply(&mut builder, &mut tx_state);
        builder.assert_true(tx_target.success);

        if with_apply_assertions {
            let expected_account_trading_mode =
                builder.constant_u64(new_account_trading_mode as u64);
            builder.conditional_assert_eq(
                tx_target.success,
                tx_state.accounts[OWNER_ACCOUNT_ID].account_trading_mode,
                expected_account_trading_mode,
            );

            let expected_collateral = positive_bigint(&mut builder, expected_collateral);
            let collateral_ok = builder.is_equal_bigint(
                &tx_state.accounts[OWNER_ACCOUNT_ID].collateral,
                &expected_collateral,
            );
            builder.assert_true(collateral_ok);

            let expected_usdc_balance = u96_biguint(&mut builder, expected_usdc_balance);
            let usdc_balance_ok = builder.is_equal_biguint(
                &tx_state.account_assets[OWNER_ACCOUNT_ID][TX_ASSET_ID].balance,
                &expected_usdc_balance,
            );
            builder.assert_true(usdc_balance_ok);
        }

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();
        let tx = L2UpdateAccountConfigTx {
            account_index: 42,
            api_key_index: 7,
            account_trading_mode: new_account_trading_mode,
        };
        pw.set_l2_update_account_config_tx_target(&tx_target, &tx)
            .unwrap();

        data.prove(pw).and_then(|proof| data.verify(proof)).is_ok()
    }

    #[test]
    fn l2_update_account_config_unify_success_and_apply() {
        let ok = prove_case(
            MASTER_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED,
            USDC_ASSET_INDEX,
            10,
            0,
            true,
            15,
            0,
        );
        assert!(ok);
    }

    #[test]
    fn l2_update_account_config_rejects_invalid_account_type() {
        let ok = prove_case(
            PUBLIC_POOL_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED,
            USDC_ASSET_INDEX,
            10,
            0,
            false,
            0,
            0,
        );
        assert!(!ok);
    }

    #[test]
    fn l2_update_account_config_rejects_non_usdc_tx_asset() {
        let ok = prove_case(
            MASTER_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED,
            USDC_ASSET_INDEX + 1,
            10,
            0,
            false,
            0,
            0,
        );
        assert!(!ok);
    }

    #[test]
    fn l2_update_account_config_rejects_unify_with_locked_usdc() {
        let ok = prove_case(
            MASTER_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            USDC_ASSET_INDEX,
            10,
            1,
            false,
            0,
            0,
        );
        assert!(!ok);
    }

    #[test]
    fn l2_update_account_config_unified_to_isolated_success_and_apply() {
        let ok = prove_case(
            MASTER_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            USDC_ASSET_INDEX,
            10,
            0,
            true,
            5,
            10,
        );
        assert!(ok);
    }

    #[test]
    fn l2_update_account_config_rejects_same_mode() {
        let ok = prove_case(
            MASTER_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            USDC_ASSET_INDEX,
            10,
            0,
            false,
            0,
            0,
        );
        assert!(!ok);
    }

    #[test]
    fn l2_update_account_config_sub_account_unify_success_and_apply() {
        let ok = prove_case(
            SUB_ACCOUNT_TYPE,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED,
            USDC_ASSET_INDEX,
            10,
            0,
            true,
            15,
            0,
        );
        assert!(ok);
    }
}
