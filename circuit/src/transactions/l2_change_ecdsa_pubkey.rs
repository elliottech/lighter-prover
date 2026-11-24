// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Switches an api key to ECDSA (secp256k1) signing. The installed key is a
//! compressed secp256k1 point; the Schnorr key of the slot is cleared and the
//! key stays ECDSA afterwards. A zero key disables the slot (no signature is
//! required for that, mirroring the Schnorr change pubkey transaction).

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::deserializers;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_attributes::ATTR_REVOKE_PUB_KEY_WITHOUT_L1_SIGNATURE;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::api_key::API_KEY_TYPE_ECDSA;
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::ecdsa_pub_key::{EcdsaPubKey, EcdsaPubKeyTarget, EcdsaPubKeyTargetWitness};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L2ChangeECDSAPubKeyTx {
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserializers::deserialize_ecdsa_pk_compressed")]
    pub pub_key: EcdsaPubKey,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct L2ChangeECDSAPubKeyTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub pub_key: EcdsaPubKeyTarget,

    // output
    pub success: BoolTarget,
}

impl L2ChangeECDSAPubKeyTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            pub_key: EcdsaPubKeyTarget::new(builder),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2ChangeECDSAPubKeyTxTarget {
    /// tx_hash = poseidon2(chain_id, tx_type, nonce, expired_at, account_index, api_key_index,
    ///                     pk.x as 8 little-endian u32 limbs, pk.y parity)
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CHANGE_ECDSA_PUB_KEY)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
        ];
        elements.extend_from_slice(&self.pub_key.to_targets());

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2ChangeECDSAPubKeyTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_change_ecdsa_pub_key;
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

        // System owned accounts can not change their public keys with L2 transactions
        let treasury_account_index = builder.constant_u64(TREASURY_ACCOUNT_INDEX as u64);
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
            treasury_account_index,
        );
        let insurance_fund_operator_account_index =
            builder.constant_u64(INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX as u64);
        builder.conditional_assert_not_eq(
            is_enabled,
            tx_state.accounts[OWNER_ACCOUNT_ID].master_account_index,
            insurance_fund_operator_account_index,
        );

        // Can not fill NIL_API_KEY_INDEX
        let nil_api_key_index = builder.constant_from_u8(NIL_API_KEY_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.api_key_index, nil_api_key_index);

        let modulus = builder.constant_biguint(&p3_ecdsa::secp::constants().p);
        builder.conditional_assert_lt_biguint(is_enabled, &self.pub_key.x, &modulus);

        let is_zero_x = builder.is_zero_biguint(&self.pub_key.x);
        let revoked = builder.and_not(is_zero_x, self.pub_key.y_odd);

        let old_ecdsa =
            builder.is_equal_constant(tx_state.api_key.key_type, API_KEY_TYPE_ECDSA as u64);

        // A slot revoked with the signature of the key in it must hold an ECDSA key. The attribute is range checked to
        // a single bit.
        let without_l1_signature = BoolTarget::new_unsafe(
            tx_state.get_attribute(ATTR_REVOKE_PUB_KEY_WITHOUT_L1_SIGNATURE),
        );
        let revoke_with_old_key = builder.multi_and(&[is_enabled, revoked, without_l1_signature]);
        builder.conditional_assert_true(revoke_with_old_key, old_ecdsa);

        // Only allow revoking old ecdsa keys if new ecdsa keys cannot be created
        if !tx_state.signature_schemes.can_create_ecdsa_keys() {
            let revoking_old_ecdsa = builder.and(revoked, old_ecdsa);
            builder.conditional_assert_true(self.success, revoking_old_ecdsa);
        }
    }
}

impl Apply for L2ChangeECDSAPubKeyTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let api_key = &mut tx_state.api_key;

        let zero_schnorr_pk = builder.zero_quintic_ext();
        api_key.public_key =
            builder.select_quintic_ext(self.success, zero_schnorr_pk, api_key.public_key);
        api_key.ecdsa_public_key = EcdsaPubKeyTarget::select(
            builder,
            self.success,
            &self.pub_key,
            &api_key.ecdsa_public_key,
        );

        // The api key is ECDSA from now on, even when the installed key is zero (disabled).
        let ecdsa = builder.constant_from_u8(API_KEY_TYPE_ECDSA);
        api_key.key_type = builder.select(self.success, ecdsa, api_key.key_type);

        self.success
    }
}

pub trait L2ChangeECDSAPubKeyTxTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_l2_change_ecdsa_pk_tx_target(
        &mut self,
        a: &L2ChangeECDSAPubKeyTxTarget,
        b: &L2ChangeECDSAPubKeyTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64 + Extendable<5> + RichField>
    L2ChangeECDSAPubKeyTxTargetWitness<F> for T
{
    fn set_l2_change_ecdsa_pk_tx_target(
        &mut self,
        a: &L2ChangeECDSAPubKeyTxTarget,
        b: &L2ChangeECDSAPubKeyTx,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_ecdsa_pub_key_target(&a.pub_key, &b.pub_key)?;

        Ok(())
    }
}
