//! A complete batch: the ECDSA proof and the message-hash proof, checked together.
//!
//! Each proof alone is not a statement about signed `tx_hash`es: the ECDSA proof
//! uses a witness `z`, and only the shared link digest ties it to
//! `keccak256(personal_sign(tx_hash))`. [`PreparedBatch::verify`] derives the
//! message-hash public values from the ECDSA ones, so the link cannot be skipped.

use p3_goldilocks::Goldilocks;

use crate::air::layout::NUM_PUBLIC_VALUES;
use crate::digest::PUBLIC_DIGEST_WIDTH;
use crate::error::{Error, Result};
use crate::keccak::prove::{MessageHashProof, prove_message_hashes, verify_message_hashes};
use crate::prove::{EcdsaBatchProof, PreparedEcdsaBatch};
use crate::types::EcdsaInstance;

pub struct BatchProof {
    pub ecdsa: EcdsaBatchProof,
    pub message_hash: MessageHashProof,
}

pub struct PreparedBatch {
    pub ecdsa: PreparedEcdsaBatch,
}

/// `[n, L0..L3]` of the message-hash proof, taken from the ECDSA public values.
pub fn message_hash_public_values(public_values: &[Goldilocks]) -> Result<Vec<Goldilocks>> {
    if public_values.len() != NUM_PUBLIC_VALUES {
        return Err(Error::Verification(format!(
            "expected {NUM_PUBLIC_VALUES} public values, got {}",
            public_values.len()
        )));
    }
    let mut out = vec![public_values[0]];
    out.extend_from_slice(&public_values[1 + PUBLIC_DIGEST_WIDTH..]);
    Ok(out)
}

impl PreparedBatch {
    pub fn new(capacity: usize) -> Self {
        Self {
            ecdsa: PreparedEcdsaBatch::new(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.ecdsa.capacity()
    }

    /// Proves both halves; returns the proof and the ECDSA public values `[n, D, L]`.
    pub fn prove(&self, instances: &[EcdsaInstance]) -> Result<(BatchProof, Vec<Goldilocks>)> {
        let ecdsa = self.ecdsa.prove(instances)?;
        let message_hash = prove_message_hashes(instances, self.capacity())?;
        debug_assert_eq!(
            message_hash_public_values(&ecdsa.public_values)?,
            message_hash.public_values
        );
        Ok((
            BatchProof {
                ecdsa: ecdsa.proof,
                message_hash: message_hash.proof,
            },
            ecdsa.public_values,
        ))
    }

    /// Accepts iff both proofs verify against the same `n` and link digest.
    pub fn verify(&self, proof: &BatchProof, public_values: &[Goldilocks]) -> Result<()> {
        let message_hash_pv = message_hash_public_values(public_values)?;
        self.ecdsa.verify(&proof.ecdsa, public_values)?;
        verify_message_hashes(&proof.message_hash, &message_hash_pv, self.capacity())
    }
}
