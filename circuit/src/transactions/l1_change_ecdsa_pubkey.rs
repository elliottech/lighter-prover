// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Installs an ECDSA (secp256k1) api key from L1. The key is the 33 byte compressed point; the Schnorr key of the
//! slot is cleared. The revoke key (even parity, `x = 0`) disables the slot.

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::secp256k1_base::Secp256K1Base;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;
use serde_with::serde_as;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::eddsa::gadgets::base_field::CircuitBuilderGFp5;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::api_key::API_KEY_TYPE_ECDSA;
use crate::types::config::{Builder, F};
use crate::types::constants::{
    LIGHTER_STAKING_POOL_ACCOUNT_TYPE, MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX,
    NIL_API_KEY_INDEX, OWNER_ACCOUNT_ID, PRIORITY_PUB_DATA_TYPE_L1_CHANGE_ECDSA_PUB_KEY,
};
use crate::types::ecdsa_pub_key::{ECDSA_PUB_KEY_COMPRESSED_LEN, EcdsaPubKeyTarget};
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::{CircuitBuilderU8, U8Target, WitnessU8};
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::utils::CircuitBuilderUtils;

#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct L1ChangeECDSAPubKeyTx {
    #[serde(rename = "ai")]
    pub account_index: i64,
    #[serde(rename = "mai")]
    pub master_account_index: i64,
    #[serde(rename = "ki")]
    pub api_key_index: u8,
    #[serde(rename = "p")]
    #[serde_as(as = "[_; ECDSA_PUB_KEY_COMPRESSED_LEN]")]
    pub pub_key: [u8; ECDSA_PUB_KEY_COMPRESSED_LEN],
    #[serde(rename = "cr")]
    pub curve_root: [u8; 32],
    #[serde(rename = "oc")]
    pub on_curve: bool,
}

/// `x = 0` is not on secp256k1 (7 is a non-residue), so the witness of the
/// default (revoke) key is `root^2 = 7 * (0^3 + 7)`, i.e. `root = 7`.
const ZERO_KEY_CURVE_ROOT: [u8; 32] = {
    let mut root = [0; 32];
    root[31] = 7;
    root
};

/// Compressed encoding of the revoke key: even parity and `x = 0`.
const REVOKE_PUB_KEY: [u8; ECDSA_PUB_KEY_COMPRESSED_LEN] = {
    let mut pub_key = [0; ECDSA_PUB_KEY_COMPRESSED_LEN];
    pub_key[0] = 2;
    pub_key
};

impl Default for L1ChangeECDSAPubKeyTx {
    fn default() -> Self {
        Self {
            account_index: 0,
            master_account_index: 0,
            api_key_index: 0,
            pub_key: REVOKE_PUB_KEY,
            curve_root: ZERO_KEY_CURVE_ROOT,
            on_curve: false,
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct L1ChangeECDSAPubKeyTxTarget {
    pub account_index: Target,
    pub master_account_index: Target,
    pub api_key_index: Target,
    #[serde_as(as = "[_; ECDSA_PUB_KEY_COMPRESSED_LEN]")]
    pub pub_key: [U8Target; ECDSA_PUB_KEY_COMPRESSED_LEN],
    curve_root: BigUintTarget,
    on_curve: BoolTarget,
    ecdsa_pub_key: EcdsaPubKeyTarget,

    // Output
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1ChangeECDSAPubKeyTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            master_account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            pub_key: core::array::from_fn(|_| builder.add_virtual_u8_target_safe()),
            curve_root: builder.add_virtual_biguint_target_safe(8),
            on_curve: builder.add_virtual_bool_target_safe(),
            ecdsa_pub_key: EcdsaPubKeyTarget::default(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }

    fn validate_pub_key(&mut self, builder: &mut Builder, tx_state: &TxState) -> BoolTarget {
        let (mut ecdsa_valid, ecdsa_revoke, ecdsa_pub_key) = {
            let zero = builder.zero_biguint();

            let modulus = builder.constant_biguint(&p3_ecdsa::secp::constants().p);

            let y_even = builder.is_equal_constant(self.pub_key[0].0, 2);
            let y_odd = builder.is_equal_constant(self.pub_key[0].0, 3);
            let prefix_valid = builder.or(y_even, y_odd);

            let x = BigUintTarget {
                limbs: self.pub_key[1..33]
                    .rchunks_exact(4)
                    .map(|bytes| {
                        let reversed = bytes.iter().rev().copied().collect::<Vec<_>>();
                        U32Target(bytes_le(builder, &reversed))
                    })
                    .collect(),
            };
            let x_valid = builder.is_lt_biguint(&x, &modulus);
            let x_zero = builder.is_zero_biguint(&x);

            let ecdsa_revoke = builder.and(y_even, x_zero);
            let ecdsa_pub_key = EcdsaPubKeyTarget {
                x: x.clone(),
                y_odd,
            };

            let point_valid = {
                // Validate that the point (x, y) lies on the secp256k1 curve.
                // Revoked key (x = 0, y even) is actually not on the curve, but we allow it as a special case.
                let safe_x = builder.select_biguint(x_valid, &x, &zero);
                let x: NonNativeTarget<Secp256K1Base> = builder.biguint_to_nonnative(&safe_x);
                let x2 = builder.mul_nonnative(&x, &x);
                let x3 = builder.mul_nonnative(&x2, &x);
                let seven = builder.constant_nonnative(Secp256K1Base::from_canonical_u64(7));
                let rhs = builder.add_nonnative(&x3, &seven);
                let nonresidue_rhs = builder.mul_nonnative(&rhs, &seven);
                let expected_square =
                    builder.select_nonnative(self.on_curve, &rhs, &nonresidue_rhs);
                let root: NonNativeTarget<Secp256K1Base> =
                    builder.biguint_to_nonnative(&self.curve_root);
                let square = builder.mul_nonnative(&root, &root);
                builder.connect_nonnative(&square, &expected_square);
                let root_canonical = builder.is_lt_biguint(&self.curve_root, &modulus);
                builder.assert_true(root_canonical);
                let parity = builder.split_le(self.curve_root.limbs[0].0, 32)[0];
                builder.conditional_assert_eq(self.on_curve, parity.target, y_odd.target);

                builder.multi_and(&[prefix_valid, x_valid, self.on_curve])
            };

            (
                builder.or(ecdsa_revoke, point_valid),
                ecdsa_revoke,
                ecdsa_pub_key,
            )
        };
        self.ecdsa_pub_key = ecdsa_pub_key;

        // Only allow revoking old ecdsa keys if new ecdsa keys cannot be created
        if !tx_state.signature_schemes.can_create_ecdsa_keys() {
            let old_ecdsa =
                builder.is_equal_constant(tx_state.api_key.key_type, API_KEY_TYPE_ECDSA as u64);
            ecdsa_valid = builder.multi_and(&[ecdsa_valid, ecdsa_revoke, old_ecdsa]);
        }
        ecdsa_valid
    }
}

fn bytes_le(builder: &mut Builder, bytes: &[U8Target]) -> Target {
    bytes.iter().rev().fold(builder.zero(), |acc, byte| {
        builder.mul_const_add(F::from_canonical_u64(256), acc, byte.0)
    })
}

impl Verify for L1ChangeECDSAPubKeyTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.is_enabled = tx_type.is_l1_change_ecdsa_pub_key;
        self.success = tx_type.is_l1_change_ecdsa_pub_key;

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

impl Apply for L1ChangeECDSAPubKeyTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let api_key = &mut tx_state.api_key;
        let zero_schnorr = builder.zero_quintic_ext();
        api_key.public_key =
            builder.select_quintic_ext(self.success, zero_schnorr, api_key.public_key);
        let key_type = builder.constant_from_u8(API_KEY_TYPE_ECDSA);
        api_key.key_type = builder.select(self.success, key_type, api_key.key_type);
        api_key.ecdsa_public_key = EcdsaPubKeyTarget::select(
            builder,
            self.success,
            &self.ecdsa_pub_key,
            &api_key.ecdsa_public_key,
        );

        self.success
    }
}

impl PriorityOperationsPubData for L1ChangeECDSAPubKeyTxTarget {
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
            add_pub_data_type_target(
                builder,
                bytes,
                PRIORITY_PUB_DATA_TYPE_L1_CHANGE_ECDSA_PUB_KEY,
            ),
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

pub trait L1ChangeECDSAPubKeyTxTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_l1_change_ecdsa_pub_key_tx_target(
        &mut self,
        a: &L1ChangeECDSAPubKeyTxTarget,
        b: &L1ChangeECDSAPubKeyTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + RichField + Extendable<5>>
    L1ChangeECDSAPubKeyTxTargetWitness<F> for T
{
    fn set_l1_change_ecdsa_pub_key_tx_target(
        &mut self,
        a: &L1ChangeECDSAPubKeyTxTarget,
        b: &L1ChangeECDSAPubKeyTx,
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
        self.set_biguint_target(&a.curve_root, &BigUint::from_bytes_be(&b.curve_root))?;
        self.set_bool_target(a.on_curve, b.on_curve)?;

        Ok(())
    }
}
