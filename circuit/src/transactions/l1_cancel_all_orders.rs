// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::tx_utils::apply_immediate_cancel_all;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L1CancelAllOrdersTx {
    #[serde(rename = "mai")]
    pub master_account_index: i64,
    #[serde(rename = "a")]
    pub account_index: i64,
}

#[derive(Debug)]
pub struct L1CancelAllOrdersTxTarget {
    pub account_index: Target,
    pub master_account_index: Target,

    // Output
    pub success: BoolTarget,
    pub is_enabled: BoolTarget,
}

impl L1CancelAllOrdersTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L1CancelAllOrdersTxTarget {
            account_index: builder.add_virtual_target(),
            master_account_index: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }
}

impl Verify for L1CancelAllOrdersTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.is_enabled = tx_type.is_l1_cancel_all_orders;
        self.success = tx_type.is_l1_cancel_all_orders;

        builder.conditional_assert_eq(
            self.success,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        let is_new_account = tx_state.is_new_account[OWNER_ACCOUNT_ID];
        self.success = builder.and_not(self.success, is_new_account);

        let is_master_account_correct = builder.is_equal(
            self.master_account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
        );
        self.success = builder.and(self.success, is_master_account_correct);

        let is_staking_pool = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );
        self.success = builder.and_not(self.success, is_staking_pool);
    }
}

impl Apply for L1CancelAllOrdersTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        apply_immediate_cancel_all(builder, self.success, tx_state, self.account_index);

        self.success
    }
}

impl PriorityOperationsPubData for L1CancelAllOrdersTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let bytes_count = [
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_CANCEL_ALL_ORDERS),
            add_target(builder, bytes, self.account_index, ACCOUNT_INDEX_BITS),
            add_target(
                builder,
                bytes,
                self.master_account_index,
                ACCOUNT_INDEX_BITS,
            ),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, bytes_count),
        )
    }
}

pub trait L1CancelAllOrdersTxTargetWitness<F: PrimeField64> {
    fn set_l1_cancel_all_orders_tx_target(
        &mut self,
        a: &L1CancelAllOrdersTxTarget,
        b: &L1CancelAllOrdersTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1CancelAllOrdersTxTargetWitness<F> for T {
    fn set_l1_cancel_all_orders_tx_target(
        &mut self,
        a: &L1CancelAllOrdersTxTarget,
        b: &L1CancelAllOrdersTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )
    }
}
