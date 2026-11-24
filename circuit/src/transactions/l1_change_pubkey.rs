// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;
use serde_with::serde_as;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::api_key::API_KEY_TYPE_SCHNORR;
use crate::types::config::{Builder, F};
use crate::types::constants::{
    LIGHTER_STAKING_POOL_ACCOUNT_TYPE, MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX,
    NIL_API_KEY_INDEX, OWNER_ACCOUNT_ID, PRIORITY_PUB_DATA_TYPE_L1_CHANGE_PUB_KEY,
};
use crate::types::ecdsa_pub_key::EcdsaPubKeyTarget;
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::{CircuitBuilderU8, U8Target, WitnessU8};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct L1ChangePubKeyTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "mai")]
    pub master_account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserialize_pub_key")]
    pub pub_key: [u8; 40],
}

impl Default for L1ChangePubKeyTx {
    fn default() -> Self {
        Self {
            account_index: 0,
            master_account_index: 0,
            api_key_index: 0,
            pub_key: [0; 40],
        }
    }
}

fn deserialize_pub_key<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<[u8; 40], D::Error> {
    let values = Vec::<u64>::deserialize(deserializer)?;
    match values.len() {
        5 => Ok(core::array::from_fn(|i| values[i / 8].to_le_bytes()[i % 8])),
        40 => values
            .into_iter()
            .map(|v| u8::try_from(v).map_err(serde::de::Error::custom))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 40 bytes")),
        _ => Err(serde::de::Error::custom(
            "expected 40 bytes or 5 legacy limbs",
        )),
    }
}

#[serde_as]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct L1ChangePubKeyTxTarget {
    pub account_index: Target,
    pub master_account_index: Target,
    pub api_key_index: Target,
    #[serde_as(as = "[_; 40]")]
    pub pub_key: [U8Target; 40],
    schnorr_pub_key: QuinticExtensionTarget,

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
            pub_key: core::array::from_fn(|_| builder.add_virtual_u8_target_safe()),
            schnorr_pub_key: QuinticExtensionTarget::default(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }

    fn validate_pub_key(&mut self, builder: &mut Builder, tx_state: &TxState) -> BoolTarget {
        let (mut schnorr_valid, schnorr_revoke, schnorr_pub_key) = {
            let mut canonical = Vec::new();
            let limbs = self
                .pub_key
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| {
                    let lo = bytes_le(builder, &bytes[..4]);
                    let hi = bytes_le(builder, &bytes[4..]);
                    let hi_max = builder.is_equal_constant(hi, u32::MAX as u64);
                    let lo_nonzero = builder.is_not_zero(lo);
                    let invalid = builder.and(hi_max, lo_nonzero);
                    canonical.push(builder.not(invalid));
                    builder.mul_const_add(F::from_canonical_u64(1 << 32), hi, lo)
                })
                .collect::<Vec<_>>();

            let zero_bytes = self
                .pub_key
                .iter()
                .map(|b| builder.is_zero(b.0))
                .collect::<Vec<_>>();
            (
                builder.multi_and(&canonical),
                builder.multi_and(&zero_bytes),
                QuinticExtensionTarget(limbs.try_into().unwrap()),
            )
        };
        self.schnorr_pub_key = schnorr_pub_key;

        // Only allow revoking old schnorr keys if new schnorr keys cannot be created
        if !tx_state.signature_schemes.can_create_schnorr_keys() {
            let old_schnorr =
                builder.is_equal_constant(tx_state.api_key.key_type, API_KEY_TYPE_SCHNORR as u64);
            schnorr_valid = builder.multi_and(&[schnorr_valid, schnorr_revoke, old_schnorr]);
        }
        schnorr_valid
    }
}

fn bytes_le(builder: &mut Builder, bytes: &[U8Target]) -> Target {
    bytes.iter().rev().fold(builder.zero(), |acc, byte| {
        builder.mul_const_add(F::from_canonical_u64(256), acc, byte.0)
    })
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

        let key_valid = self.validate_pub_key(builder, tx_state);
        self.success = builder.and(self.success, key_valid);
    }
}

impl Apply for L1ChangePubKeyTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let api_key = &mut tx_state.api_key;
        api_key.public_key =
            builder.select_quintic_ext(self.success, self.schnorr_pub_key, api_key.public_key);
        let key_type = builder.constant_from_u8(API_KEY_TYPE_SCHNORR);
        api_key.key_type = builder.select(self.success, key_type, api_key.key_type);
        let zero_ecdsa_pk = EcdsaPubKeyTarget::zero(builder);
        api_key.ecdsa_public_key = EcdsaPubKeyTarget::select(
            builder,
            self.success,
            &zero_ecdsa_pk,
            &api_key.ecdsa_public_key,
        );

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
            add_pub_key_target(bytes, &self.pub_key),
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
        b: &L1ChangePubKeyTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + RichField + Extendable<5>> L1ChangePubKeyTxTargetWitness<F>
    for T
{
    fn set_l1_change_pub_key_tx_target(
        &mut self,
        a: &L1ChangePubKeyTxTarget,
        b: &L1ChangePubKeyTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        for (target, byte) in a.pub_key.iter().zip(b.pub_key) {
            self.set_u8_target(*target, byte)?;
        }

        Ok(())
    }
}
