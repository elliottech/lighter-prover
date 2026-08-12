// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Off-circuit helpers for building p3-schnorr signature-batch proofs from a block's
//! tx witnesses.

use p3_field::{PrimeCharacteristicRing, PrimeField64 as P3PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_schnorr::poseidon::{
    POSEIDON_DIGEST_WIDTH, POSEIDON_WIDTH, absorb_block, initial_public_hash_state, instance_blocks,
};
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::{Field, PrimeField, PrimeField64};
use plonky2::plonk::proof::ProofWithPublicInputs;

use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
use crate::eddsa::schnorr::{SchnorrSig, dummy_schnorr_instance};
use crate::types::config::{C, D, F};

/// The sponge's initial state for a batch of `count` signatures, as plonky2 field elements
/// — the `old_signature_digest_state` the first tx pack of a chain starts from.
pub fn initial_signature_digest_state(count: usize) -> [F; POSEIDON_WIDTH] {
    let state = initial_public_hash_state(count);
    core::array::from_fn(|i| F::from_canonical_u64(state[i].as_canonical_u64()))
}

/// Absorbs one real `(account_pk, tx_hash, signature)` instance into a running sponge state,
/// mirroring the in-circuit `P3SchnorrDigestState::absorb_instance` bit-for-bit.
pub fn absorb_signature_instance(
    state: [F; POSEIDON_WIDTH],
    account_pk: QuinticExtension<F>,
    tx_hash: QuinticExtension<F>,
    signature: &SchnorrSig,
) -> [F; POSEIDON_WIDTH] {
    let instance = p3_schnorr::types::SchnorrInstance::new(
        quintic_ext_to_fp5_bytes(tx_hash),
        schnorr_sig_to_signature_bytes(signature),
        quintic_ext_to_fp5_bytes(account_pk),
    );
    let fields = instance
        .to_goldilocks_fields()
        .expect("instance encoding failed");
    let mut p3_state: [Goldilocks; POSEIDON_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64(state[i].to_canonical_u64()));
    for block in instance_blocks(&fields) {
        p3_state = absorb_block(p3_state, block);
    }
    core::array::from_fn(|i| F::from_canonical_u64(p3_state[i].as_canonical_u64()))
}

/// The digest a p3-schnorr batch containing only the fixed dummy instance produces (count 1).
/// Chains with zero signed txs substitute this batch, and `BlockCircuit` hard-codes this
/// digest as the expected public inputs for that case.
pub fn dummy_signature_batch_digest() -> [F; POSEIDON_DIGEST_WIDTH] {
    let (pk, tx_hash, sig) = dummy_schnorr_instance();
    let state = absorb_signature_instance(initial_signature_digest_state(1), pk, tx_hash, &sig);
    core::array::from_fn(|i| state[i])
}

pub fn quintic_ext_to_fp5_bytes(x: QuinticExtension<F>) -> p3_schnorr::types::Fp5Bytes {
    let mut bytes = [0u8; 40];
    for (i, limb) in x.0.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&limb.to_canonical_u64().to_le_bytes());
    }
    p3_schnorr::types::Fp5Bytes::from_array(bytes)
}

pub fn scalar_to_bytes(s: &ECgFp5Scalar) -> [u8; 40] {
    let mut digits = s.to_canonical_biguint().to_u32_digits();
    digits.resize(10, 0);
    let mut bytes = [0u8; 40];
    for (i, d) in digits.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&d.to_le_bytes());
    }
    bytes
}

pub fn schnorr_sig_to_signature_bytes(sig: &SchnorrSig) -> p3_schnorr::types::SignatureBytes {
    let mut bytes = [0u8; 80];
    bytes[..40].copy_from_slice(&scalar_to_bytes(&sig.s));
    bytes[40..].copy_from_slice(&scalar_to_bytes(&sig.e));
    p3_schnorr::types::SignatureBytes::from_array(bytes)
}

/// Builds the block's real (account_pk, tx_hash, signature) tuples into a p3-schnorr
/// `SchnorrInstance` batch, proven at the circuit's fixed capacity: the batch contains only
/// the chain's really-signed slots (unsigned slots contribute nothing to the digest), and
/// the STARK trace's remaining rows up to the capacity are disabled. A chain with zero
/// signed slots gets the fixed single-dummy-instance batch, whose count/digest `BlockCircuit`
/// hard-codes for that case.
pub fn prove_signature_batch_proof(
    signature_batch_circuit: &p3_schnorr_recursion::TranscriptVerifierCircuit,
    tx_signature_data: &[(QuinticExtension<F>, QuinticExtension<F>, SchnorrSig)],
) -> ProofWithPublicInputs<F, C, D> {
    let capacity = signature_batch_circuit.expected_signature_count();
    let dummy_batch = [dummy_schnorr_instance()];
    let tx_signature_data = if tx_signature_data.is_empty() {
        &dummy_batch
    } else {
        tx_signature_data
    };
    assert!(
        tx_signature_data.len() <= capacity,
        "block has more real signatures ({}) than the batch capacity ({})",
        tx_signature_data.len(),
        capacity
    );

    let mut instances = Vec::with_capacity(tx_signature_data.len());
    for (pk, tx_hash, sig) in tx_signature_data {
        instances.push(p3_schnorr::types::SchnorrInstance::new(
            quintic_ext_to_fp5_bytes(*tx_hash),
            schnorr_sig_to_signature_bytes(sig),
            quintic_ext_to_fp5_bytes(*pk),
        ));
    }

    let batch = p3_schnorr::types::Batch::new(instances).expect("batch construction failed");
    let proving_output = p3_schnorr::prove_signature_batch_with_capacity(&batch, capacity)
        .expect("p3-schnorr STARK prove failed");
    let witness = p3_schnorr_recursion::prepare_recursive_witness(&proving_output, capacity)
        .expect("prepare recursive witness failed");
    p3_schnorr_recursion::prove_transcript_verifier(signature_batch_circuit, &witness)
        .expect("p3-schnorr-recursion Plonky2 prove failed")
}
