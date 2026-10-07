// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::{BoolTarget, Target};

use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::types::config::Builder;
use crate::types::constants::{
    MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX, ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE,
};
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::U8Target;
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TransactionTarget<Inner> {
    pub inner: Inner,
}

// Delegate methods
impl<Inner> TransactionTarget<Inner>
where
    Inner: Apply + Verify,
{
    pub fn new(tx: Inner) -> Self {
        Self { inner: tx }
    }

    pub fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        self.inner.apply(builder, tx_state)
    }

    pub fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.inner.verify(builder, tx_type, tx_state);
    }
}

// Delegate methods
impl<Inner> TransactionTarget<Inner>
where
    Inner: OnChainPubData,
{
    pub fn on_chain_pub_data(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
    ) -> (
        BoolTarget,
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    ) {
        self.inner.on_chain_pub_data(builder, tx_state)
    }
}

// Delegate methods
impl<Inner> TransactionTarget<Inner>
where
    Inner: PriorityOperationsPubData,
{
    pub fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        self.inner.priority_operations_pub_data(builder)
    }
}

// Delegate methods
impl<Inner> TransactionTarget<Inner>
where
    Inner: TxHash,
{
    pub fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        self.inner.hash(builder, tx_nonce, tx_expired_at, chain_id)
    }
}

pub trait Apply {
    /// Returns success
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget;
}

pub trait Verify {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState);
}
pub trait OnChainPubData {
    /// Returns true if the transaction has on-chain public data
    fn on_chain_pub_data(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
    ) -> (
        BoolTarget,
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    );
}

pub trait PriorityOperationsPubData {
    /// Returns true if the transaction has priority operations public data
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    );
}

pub trait TxHash {
    fn hash(
        &self,
        builder: &mut Builder,
        tx_nonce: Target,
        tx_expired_at: Target,
        chain_id: u32,
    ) -> QuinticExtensionTarget;
}
