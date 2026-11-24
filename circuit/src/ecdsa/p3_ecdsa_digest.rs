// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! In-circuit running sponge state for the p3-ecdsa public digest accumulator.
//! Mirrors `p3_ecdsa::digest::digest_instances`: same permutation / width / rate
//! as the Schnorr accumulator, different domain tag and a 30-element instance
//! layout of `tx_hash (5) | r (8 u32 LE limbs) | s (8) | pk.x (8) | pk.y parity`.

use p3_ecdsa::digest::{PUBLIC_DIGEST_WIDTH, PUBLIC_HASH_DOMAIN};
use p3_ecdsa::types::INSTANCE_FIELD_ELEMENTS;
use p3_schnorr::poseidon::POSEIDON_RATE;
use plonky2::field::types::Field;
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::hash::poseidon2::hash::Poseidon2Hash;
use plonky2::iop::target::{BoolTarget, Target};

use crate::eddsa::gadgets::base_field::QuinticExtensionTarget;
use crate::eddsa::p3_schnorr_digest::P3_DIGEST_STATE_WIDTH;
use crate::types::config::Builder;
use crate::types::ecdsa_pub_key::EcdsaPubKeyTarget;
use crate::types::signature_slot::{ECDSA_SIGNATURE_LIMBS, EcdsaSignatureTarget};

#[derive(Clone, Debug)]
pub struct P3EcdsaDigestState {
    pub state: [Target; P3_DIGEST_STATE_WIDTH],
}

impl P3EcdsaDigestState {
    pub fn initial(builder: &mut Builder, count: usize) -> Self {
        let count = builder.constant(Field::from_canonical_usize(count));
        Self::initial_with_count(builder, count)
    }

    pub fn initial_with_count(builder: &mut Builder, count: Target) -> Self {
        let zero = builder.zero();
        let mut state = [zero; P3_DIGEST_STATE_WIDTH];
        state[0] = count;
        state[1] = builder.constant(Field::from_canonical_u64(PUBLIC_HASH_DOMAIN));
        Self { state }
    }

    fn absorb_block(&mut self, builder: &mut Builder, block: [Target; POSEIDON_RATE]) {
        self.state[..POSEIDON_RATE].copy_from_slice(&block);
        let permutation = builder
            .builder
            .permute::<Poseidon2Hash>(PlonkyPermutation::new(self.state));
        self.state.copy_from_slice(permutation.as_ref());
    }

    pub fn instance_fields(
        tx_hash: QuinticExtensionTarget,
        signature: &EcdsaSignatureTarget,
        public_key: &EcdsaPubKeyTarget,
    ) -> [Target; INSTANCE_FIELD_ELEMENTS] {
        assert_eq!(signature.r.limbs.len(), ECDSA_SIGNATURE_LIMBS);
        assert_eq!(signature.s.limbs.len(), ECDSA_SIGNATURE_LIMBS);
        let mut fields = Vec::with_capacity(INSTANCE_FIELD_ELEMENTS);
        fields.extend(tx_hash.0);
        fields.extend(signature.r.limbs.iter().map(|limb| limb.0));
        fields.extend(signature.s.limbs.iter().map(|limb| limb.0));
        fields.extend(public_key.to_targets());
        fields
            .try_into()
            .expect("ecdsa instance has 30 field elements")
    }

    pub fn absorb_instance(
        &mut self,
        builder: &mut Builder,
        tx_hash: QuinticExtensionTarget,
        signature: &EcdsaSignatureTarget,
        public_key: &EcdsaPubKeyTarget,
    ) {
        let zero = builder.zero();
        let fields = Self::instance_fields(tx_hash, signature, public_key);
        let mut cursor = 0;
        while cursor < INSTANCE_FIELD_ELEMENTS {
            let take = (INSTANCE_FIELD_ELEMENTS - cursor).min(POSEIDON_RATE);
            let mut block = [zero; POSEIDON_RATE];
            block[..take].copy_from_slice(&fields[cursor..cursor + take]);
            self.absorb_block(builder, block);
            cursor += take;
        }
    }

    pub fn absorb_instance_conditional(
        &mut self,
        builder: &mut Builder,
        tx_hash: QuinticExtensionTarget,
        signature: &EcdsaSignatureTarget,
        public_key: &EcdsaPubKeyTarget,
        signed: BoolTarget,
    ) {
        let previous_state = self.state;
        self.absorb_instance(builder, tx_hash, signature, public_key);
        self.state = core::array::from_fn(|i| {
            builder
                .builder
                .select(signed, self.state[i], previous_state[i])
        });
    }

    /// The final digest: the first `PUBLIC_DIGEST_WIDTH` lanes of the running state.
    pub fn digest(&self) -> [Target; PUBLIC_DIGEST_WIDTH] {
        core::array::from_fn(|i| self.state[i])
    }
}
