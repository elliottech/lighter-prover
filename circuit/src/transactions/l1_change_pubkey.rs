// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::{
    CircuitBuilderGFp5, PartialWitnessQuinticExt, QuinticExtensionTarget,
};
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::config::Builder;
use crate::types::constants::{
    LIGHTER_STAKING_POOL_ACCOUNT_TYPE, MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX,
    NIL_API_KEY_INDEX, OWNER_ACCOUNT_ID, PRIORITY_PUB_DATA_TYPE_L1_CHANGE_PUB_KEY,
};
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L1ChangePubKeyTx<F>
where
    F: Field + Extendable<5> + RichField,
{
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "mai")]
    pub master_account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserializers::pub_key")]
    pub pub_key: QuinticExtension<F>,
}

#[derive(Debug)]
pub struct L1ChangePubKeyTxTarget {
    pub account_index: Target,
    pub master_account_index: Target,
    pub api_key_index: Target,
    pub pub_key: QuinticExtensionTarget,

    // Output
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1ChangePubKeyTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            master_account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            pub_key: builder.add_virtual_quintic_ext_target(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }
}

impl Verify for L1ChangePubKeyTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.is_enabled = tx_type.is_l1_change_pub_key;
        self.success = tx_type.is_l1_change_pub_key;

        builder.conditional_assert_eq(
            self.is_enabled,
            self.account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].account_index,
        );

        builder.conditional_assert_eq(
            self.is_enabled,
            self.api_key_index,
            tx_state.api_key.api_key_index,
        );

        // Can not fill NIL_API_KEY_INDEX
        let nil_api_key_index = builder.constant_from_u8(NIL_API_KEY_INDEX);
        builder.conditional_assert_not_eq(self.is_enabled, self.api_key_index, nil_api_key_index);

        let is_new_account = tx_state.is_new_account[OWNER_ACCOUNT_ID];
        self.success = builder.and_not(self.success, is_new_account);

        let is_master_index_equal = builder.is_equal(
            self.master_account_index,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
        );
        self.success = builder.and(self.success, is_master_index_equal);

        let is_staking_pool = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );
        self.success = builder.and_not(self.success, is_staking_pool);
    }
}

impl Apply for L1ChangePubKeyTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.api_key.public_key =
            builder.select_quintic_ext(self.success, self.pub_key, tx_state.api_key.public_key);

        self.success
    }
}

impl PriorityOperationsPubData for L1ChangePubKeyTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let bit_count = [
            add_pub_data_type_target(builder, bytes, PRIORITY_PUB_DATA_TYPE_L1_CHANGE_PUB_KEY),
            add_target(builder, bytes, self.account_index, 48),
            add_target(builder, bytes, self.master_account_index, 48),
            add_byte_target_unsafe(bytes, self.api_key_index),
            add_pub_key_target(builder, bytes, &self.pub_key),
        ]
        .iter()
        .sum();

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, bit_count),
        )
    }
}

pub trait L1ChangePubKeyTxTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_l1_change_pub_key_tx_target(
        &mut self,
        a: &L1ChangePubKeyTxTarget,
        b: &L1ChangePubKeyTx<F>,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + RichField + Extendable<5>> L1ChangePubKeyTxTargetWitness<F>
    for T
{
    fn set_l1_change_pub_key_tx_target(
        &mut self,
        a: &L1ChangePubKeyTxTarget,
        b: &L1ChangePubKeyTx<F>,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_quintic_ext_target(a.pub_key, b.pub_key)?;

        Ok(())
    }
}
