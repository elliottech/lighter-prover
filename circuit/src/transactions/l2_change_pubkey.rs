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

use crate::deserializers;
use crate::eddsa::gadgets::base_field::{
    CircuitBuilderGFp5, PartialWitnessQuinticExt, QuinticExtensionTarget,
};
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::eddsa::schnorr::hash_to_quintic_extension_circuit;
use crate::tx_interface::{Apply, TxHash, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(bound = "")]
#[serde(default)]
pub struct L2ChangePubKeyTx<F>
where
    F: Field + Extendable<5> + RichField,
{
    #[serde(rename = "ai")]
    pub account_index: i64,

    #[serde(rename = "ki")]
    pub api_key_index: u8,

    #[serde(rename = "p")]
    #[serde(deserialize_with = "deserializers::pub_key")]
    pub pub_key: QuinticExtension<F>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct L2ChangePubKeyTxTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub pub_key: QuinticExtensionTarget,

    // output
    pub success: BoolTarget,
}

impl L2ChangePubKeyTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            pub_key: builder.add_virtual_quintic_ext_target(),

            // output
            success: BoolTarget::default(),
        }
    }
}

impl TxHash for L2ChangePubKeyTxTarget {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut elements = vec![
            builder.constant(F::from_canonical_u32(chain_id)),
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CHANGE_PUB_KEY)),
            tx_nonce,
            tx_expired_at,
            self.account_index,
            self.api_key_index,
        ];
        elements.extend_from_slice(&self.pub_key.0);

        hash_to_quintic_extension_circuit(builder, &elements)
    }
}

impl Verify for L2ChangePubKeyTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        let is_enabled = tx_type.is_l2_change_pub_key;
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

        // Can not fill NIL_API_KEY_INDEX
        let nil_api_key_index = builder.constant_from_u8(NIL_API_KEY_INDEX);
        builder.conditional_assert_not_eq(is_enabled, self.api_key_index, nil_api_key_index);
    }
}

impl Apply for L2ChangePubKeyTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        tx_state.api_key.public_key =
            builder.select_quintic_ext(self.success, self.pub_key, tx_state.api_key.public_key);

        self.success
    }
}

pub trait L2ChangePubKeyTxTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_l2_change_pk_tx_target(
        &mut self,
        a: &L2ChangePubKeyTxTarget,
        b: &L2ChangePubKeyTx<F>,
    ) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    L2ChangePubKeyTxTargetWitness<F> for T
{
    fn set_l2_change_pk_tx_target(
        &mut self,
        a: &L2ChangePubKeyTxTarget,
        b: &L2ChangePubKeyTx<F>,
    ) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(a.api_key_index, F::from_canonical_u8(b.api_key_index))?;
        self.set_quintic_ext_target(a.pub_key, b.pub_key)?;

        Ok(())
    }
}
