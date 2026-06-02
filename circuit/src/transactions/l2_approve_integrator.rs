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
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::approved_integrator::{
    ApprovedIntegratorTarget, select_approved_integrator_target,
};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2ApproveIntegratorTx {
    #[serde(rename = "ai", default)]
    pub account_index: i64,
    #[serde(rename = "aki", default)]
    pub api_key_index: u8,
    #[serde(rename = "iai", default)]
    pub integrator_account_index: i64,
    #[serde(rename = "mptf", default)]
    pub max_perps_taker_fee: u32,
    #[serde(rename = "mpmf", default)]
    pub max_perps_maker_fee: u32,
    #[serde(rename = "mstf", default)]
    pub max_spot_taker_fee: u32,
    #[serde(rename = "msmf", default)]
    pub max_spot_maker_fee: u32,
    #[serde(rename = "ae", default)]
    pub approval_expiry: i64,
}

#[derive(Debug)]
pub struct L2ApproveIntegratorTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub integrator_account_index: Target,
    pub max_perps_taker_fee: Target,
    pub max_perps_maker_fee: Target,
    pub max_spot_taker_fee: Target,
    pub max_spot_maker_fee: Target,
    pub approval_expiry: Target,

    // Helpers
    is_revoking_approval: BoolTarget,
    is_granting_approval: BoolTarget,

    // Output
    success: BoolTarget,
}

impl L2ApproveIntegratorTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            integrator_account_index: builder.add_virtual_target(),
            max_perps_taker_fee: builder.add_virtual_target(),
            max_perps_maker_fee: builder.add_virtual_target(),
            max_spot_taker_fee: builder.add_virtual_target(),
            max_spot_maker_fee: builder.add_virtual_target(),
            approval_expiry: builder.add_virtual_target(),

            // Helpers
            is_revoking_approval: BoolTarget::default(),
            is_granting_approval: BoolTarget::default(),

            // Output
            success: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder, is_enabled: BoolTarget) {
        builder.register_range_check(self.integrator_account_index, ACCOUNT_INDEX_BITS);
        let nil_account_index = builder.constant_i64(NIL_ACCOUNT_INDEX);
        builder.conditional_assert_not_eq(
            is_enabled,
            self.integrator_account_index,
            nil_account_index,
        );

        builder.register_range_check(self.approval_expiry, TIMESTAMP_BITS);

        let fee_tick = builder.constant_u64(FEE_TICK);
        [
            self.max_perps_taker_fee,
            self.max_perps_maker_fee,
            self.max_spot_taker_fee,
            self.max_spot_maker_fee,
        ]
        .iter()
        .for_each(|&fee| {
            builder.register_range_check(fee, 24);
            builder.conditional_assert_lte(is_enabled, fee, fee_tick, 24);
        });
    }
}

impl TxHash for L2ApproveIntegratorTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_APPROVE_INTEGRATOR)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
            self.integrator_account_index,
            self.max_perps_taker_fee,
            self.max_perps_maker_fee,
            self.max_spot_taker_fee,
            self.max_spot_maker_fee,
            self.approval_expiry,
        ];

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2ApproveIntegratorTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_approve_integrator;
        self.success = is_enabled;

        self.register_range_checks(builder, is_enabled);

        builder.conditional_assert_eq(
            is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.integrator_account_index,
            tx_state.accounts[INTEGRATOR_ACCOUNT_ID].account_index,
        );
        builder.conditional_assert_eq(
            is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );

        // Revoking approval requires all fees to be zero AND expiry to be zero.
        // 0-fee integrators (fees=0, expiry>0) are allowed and skip L1 signature.
        // Fees are range checked to be <= FEE_TICK so this addition won't overflow.
        let fees_added = builder.add_many([
            self.max_perps_taker_fee,
            self.max_perps_maker_fee,
            self.max_spot_taker_fee,
            self.max_spot_maker_fee,
        ]);
        let no_fees = builder.is_zero(fees_added);
        let is_approval_expiry_zero = builder.is_zero(self.approval_expiry);

        // If expiry is zero, all fees must be zero
        let check_expiry_zero = builder.and(is_enabled, is_approval_expiry_zero);
        builder.conditional_assert_zero(check_expiry_zero, fees_added);

        self.is_revoking_approval =
            builder.multi_and(&[is_enabled, no_fees, is_approval_expiry_zero]);
        self.is_granting_approval = builder.and_not(is_enabled, self.is_revoking_approval);

        // Fees can't exceed the maximums set in the system config
        [
            (
                self.max_perps_taker_fee,
                tx_state.system_config.max_integrator_perps_taker_fee,
            ),
            (
                self.max_perps_maker_fee,
                tx_state.system_config.max_integrator_perps_maker_fee,
            ),
            (
                self.max_spot_taker_fee,
                tx_state.system_config.max_integrator_spot_taker_fee,
            ),
            (
                self.max_spot_maker_fee,
                tx_state.system_config.max_integrator_spot_maker_fee,
            ),
        ]
        .iter()
        .for_each(|&(fee, max_fee)| {
            builder.conditional_assert_lte(is_enabled, fee, max_fee, 24);
        });

        // Treasury sub accounts can only approve other treasury sub accounts
        let is_same_master_account = builder.is_equal(
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
            tx_state.accounts[INTEGRATOR_ACCOUNT_ID].master_account_index,
        );
        let is_owner_treasury = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
            TREASURY_ACCOUNT_INDEX as u64,
        );
        let should_be_false = builder.and_not(is_owner_treasury, is_same_master_account);
        builder.conditional_assert_false(is_enabled, should_be_false);

        // Approval expiry must be > block timestamp when granting approval
        let check_approval_expiry_flag = self.is_granting_approval;
        builder.conditional_assert_lt(
            check_approval_expiry_flag,
            tx_state.block_timestamp,
            self.approval_expiry,
            TIMESTAMP_BITS,
        );

        // If user is granting approval, then owner account must have an available approval slot
        // If integrator already approved, we will reuse the same slot(or revoke it, and append new entry) so we don't need to check for an additional empty slot in that case
        let mut can_approve_accumulator = builder.zero();
        for integrator in tx_state.accounts[OWNER_ACCOUNT_ID].approved_integrators {
            let is_empty = integrator.is_empty(builder);
            let is_same_integrator = builder.is_equal(
                integrator.integrator_account_index,
                self.integrator_account_index,
            );
            let is_expired =
                builder.is_lte(integrator.expiry, tx_state.block_timestamp, TIMESTAMP_BITS);
            can_approve_accumulator = builder.add_many([
                can_approve_accumulator,
                is_empty.target,
                is_same_integrator.target,
                is_expired.target,
            ]);
        }
        builder.conditional_assert_not_zero(self.is_granting_approval, can_approve_accumulator);
    }
}

impl Apply for L2ApproveIntegratorTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let mut new_approved_integrators =
            [ApprovedIntegratorTarget::empty(builder); MAX_APPROVED_INTEGRATORS];

        // Get rid of expired approvals and the current approval if it's being revoked.
        for i in 0..MAX_APPROVED_INTEGRATORS {
            let integrator = &tx_state.accounts[OWNER_ACCOUNT_ID].approved_integrators[i];

            let is_same_integrator = builder.is_equal(
                integrator.integrator_account_index,
                self.integrator_account_index,
            );
            let is_revoke = builder.and(self.is_revoking_approval, is_same_integrator);
            let is_expired_or_empty =
                builder.is_lte(integrator.expiry, tx_state.block_timestamp, TIMESTAMP_BITS);

            let slot_should_go = builder.or(is_revoke, is_expired_or_empty);

            let mut applied = slot_should_go;
            for j in 0..i + 1 {
                let is_slot_in_new_array_empty = new_approved_integrators[j].is_empty(builder);
                let flag = builder.and_not(is_slot_in_new_array_empty, applied);
                applied = builder.or(applied, flag);
                new_approved_integrators[j] = select_approved_integrator_target(
                    builder,
                    flag,
                    integrator,
                    &new_approved_integrators[j],
                );
            }
        }

        // Insert approval, either to first empty slot or on top of existing approval if it's being updated
        // There can't be an empty slot before an existing approval
        let mut applied = builder.not(self.is_granting_approval);
        for i in 0..MAX_APPROVED_INTEGRATORS {
            let is_empty = new_approved_integrators[i].is_empty(builder);

            let is_same_integrator = builder.is_equal(
                new_approved_integrators[i].integrator_account_index,
                self.integrator_account_index,
            );

            let should_apply = builder.or(is_empty, is_same_integrator);

            let flag = builder.and_not(should_apply, applied);
            applied = builder.or(applied, flag);
            new_approved_integrators[i] = select_approved_integrator_target(
                builder,
                flag,
                &ApprovedIntegratorTarget {
                    integrator_account_index: self.integrator_account_index,
                    max_perps_taker_fee: self.max_perps_taker_fee,
                    max_perps_maker_fee: self.max_perps_maker_fee,
                    max_spot_taker_fee: self.max_spot_taker_fee,
                    max_spot_maker_fee: self.max_spot_maker_fee,
                    expiry: self.approval_expiry,
                },
                &new_approved_integrators[i],
            );
        }

        // Apply changes to tx_state
        for i in 0..MAX_APPROVED_INTEGRATORS {
            tx_state.accounts[OWNER_ACCOUNT_ID].approved_integrators[i] =
                select_approved_integrator_target(
                    builder,
                    self.success,
                    &new_approved_integrators[i],
                    &tx_state.accounts[OWNER_ACCOUNT_ID].approved_integrators[i],
                );
        }

        self.success
    }
}

pub trait L2ApproveIntegratorTxTargetWitness<F: PrimeField64> {
    fn set_l2_approve_integrator_tx_target(
        &mut self,
        a: &L2ApproveIntegratorTxTarget,
        b: &L2ApproveIntegratorTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L2ApproveIntegratorTxTargetWitness<F> for T {
    fn set_l2_approve_integrator_tx_target(
        &mut self,
        a: &L2ApproveIntegratorTxTarget,
        b: &L2ApproveIntegratorTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_target(
            a.integrator_account_index,
            F::from_canonical_i64(b.integrator_account_index),
        )?;
        self.set_target(
            a.max_perps_taker_fee,
            F::from_canonical_u32(b.max_perps_taker_fee),
        )?;
        self.set_target(
            a.max_perps_maker_fee,
            F::from_canonical_u32(b.max_perps_maker_fee),
        )?;
        self.set_target(
            a.max_spot_taker_fee,
            F::from_canonical_u32(b.max_spot_taker_fee),
        )?;
        self.set_target(
            a.max_spot_maker_fee,
            F::from_canonical_u32(b.max_spot_maker_fee),
        )?;
        self.set_target(a.approval_expiry, F::from_canonical_i64(b.approval_expiry))?;

        Ok(())
    }
}
