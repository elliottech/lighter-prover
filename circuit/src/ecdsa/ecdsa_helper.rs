// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Off-circuit helpers for building p3-ecdsa signature-batch proofs from a block's
//! tx witnesses. Counterpart of `eddsa::schnorr_helper` for ECDSA-signed api keys.

use log::Level;
use num::BigUint;
use p3_ecdsa::digest::{PUBLIC_DIGEST_WIDTH, initial_public_hash_state, instance_blocks};
use p3_ecdsa::{EcdsaInstance, PreparedBatch, secp};
use p3_field::{PrimeCharacteristicRing, PrimeField64 as P3PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_schnorr::poseidon::{POSEIDON_WIDTH, absorb_block};
use plonky2::field::types::{Field, PrimeField64};
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::types::config::{C, D, F};

/// The sponge's initial state for a batch of `count` ECDSA signatures, as plonky2 field
/// elements — the `ecdsa_signature_digest_state_before` the first tx pack of a chain starts from.
pub fn initial_ecdsa_signature_digest_state(count: usize) -> [F; POSEIDON_WIDTH] {
    let state = initial_public_hash_state(count);
    core::array::from_fn(|i| F::from_canonical_u64(state[i].as_canonical_u64()))
}

/// Absorbs one real ECDSA instance into a running sponge state, mirroring the in-circuit
/// `P3EcdsaDigestState::absorb_instance` bit-for-bit.
pub fn absorb_ecdsa_signature_instance(
    state: [F; POSEIDON_WIDTH],
    instance: &EcdsaInstance,
) -> [F; POSEIDON_WIDTH] {
    let mut p3_state: [Goldilocks; POSEIDON_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64(state[i].to_canonical_u64()));
    for block in instance_blocks(instance) {
        p3_state = absorb_block(p3_state, block);
    }
    core::array::from_fn(|i| F::from_canonical_u64(p3_state[i].as_canonical_u64()))
}

/// A fixed, valid ECDSA instance (fixed secret key / nonce, all-zero tx hash) used to prove a
/// batch for chains with zero ECDSA-signed txs; the p3-ecdsa STARK rejects empty batches.
pub fn dummy_ecdsa_instance() -> EcdsaInstance {
    let c = secp::constants();
    let sk = BigUint::parse_bytes(
        b"3d1a7f9c2b5e8d04c6a1f3e7b9d2c5a8e4f6b1d3c7a9e2f5b8d1c4a7e3f6b9d2",
        16,
    )
    .expect("fixed dummy secret key");
    let nonce = BigUint::parse_bytes(
        b"5c2e9b4d7a1f3c6e8b0d2f5a7c9e1b4d6f8a0c3e5b7d9f1a3c5e7b9d1f3a5c7e",
        16,
    )
    .expect("fixed dummy nonce");
    let tx_hash = [Goldilocks::ZERO; 5];
    let pk = secp::mul(&c.g, &sk).expect("generator times one");
    let z = EcdsaInstance::from_parts(tx_hash, &sk, &sk, &pk).message_scalar();
    let (r, s) = secp::ecdsa_sign(&sk, &z, &nonce).expect("fixed dummy ecdsa signature");
    let instance = EcdsaInstance::from_parts(tx_hash, &r, &s, &pk);
    debug_assert!(instance.verify_reference());
    instance
}

/// The digest a p3-ecdsa batch containing only the fixed dummy instance produces (count 1).
/// Chains with zero ECDSA-signed txs substitute this batch, and `BlockCircuit` hard-codes
/// this digest as the expected public inputs for that case.
pub fn dummy_ecdsa_signature_batch_digest() -> [F; PUBLIC_DIGEST_WIDTH] {
    let state = absorb_ecdsa_signature_instance(
        initial_ecdsa_signature_digest_state(1),
        &dummy_ecdsa_instance(),
    );
    core::array::from_fn(|i| state[i])
}

/// Proves the block's ECDSA-signed slots as one p3-ecdsa batch at the circuit's fixed
/// capacity and wraps it in the recursive batch verifier, which checks BOTH the ECDSA AIR
/// proof and the keccak message-hash AIR proof (binding each signature's scalar to
/// `personal_sign(tx_hash)`), tied together by the shared count and link digest. A chain with
/// zero signed slots gets the fixed single-dummy-instance batch.
pub fn prove_ecdsa_signature_batch_proof(
    signature_batch_circuit: &p3_ecdsa_recursion::BatchVerifierCircuit,
    instances: &[EcdsaInstance],
) -> ProofWithPublicInputs<F, C, D> {
    let capacity = signature_batch_circuit.signature_count;
    let dummy_batch = [dummy_ecdsa_instance()];
    let instances = if instances.is_empty() {
        &dummy_batch
    } else {
        instances
    };
    assert!(
        instances.len() <= capacity,
        "block has more real ecdsa signatures ({}) than the batch capacity ({})",
        instances.len(),
        capacity
    );

    let mut timing = TimingTree::new("prove_ecdsa_signature_batch_proof", Level::Debug);
    let prepared = PreparedBatch::new(capacity);
    let (proof, public_values) = timed!(timing, "p3-ecdsa STARK prove (ecdsa + keccak)", {
        prepared
            .prove(instances)
            .expect("p3-ecdsa STARK prove failed")
    });
    let witness = timed!(timing, "prepare recursive witness", {
        p3_ecdsa_recursion::prepare_batch_recursive_witness(&prepared, &proof, &public_values)
            .expect("prepare ecdsa batch recursive witness failed")
    });
    let proof = timed!(timing, "plonky2 batch verifier prove", {
        p3_ecdsa_recursion::prove_batch_verifier(signature_batch_circuit, &prepared, &witness)
            .expect("p3-ecdsa-recursion Plonky2 prove failed")
    });
    timing.print();
    proof
}
