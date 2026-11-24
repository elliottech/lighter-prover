//! Prove/verify the message-hash AIR (plain uni-stark, same config as the ECDSA proof).

use std::time::{Duration, Instant};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_uni_stark::Proof;

use super::MessageHashAir;
use super::trace::generate_trace;
use crate::digest::link_digest;
use crate::error::{Error, Result};
use crate::prove::{EcdsaStarkConfig, ecdsa_stark_config};
use crate::types::EcdsaInstance;

pub type MessageHashProof = Proof<EcdsaStarkConfig>;

pub struct MessageHashProvingOutput {
    pub proof: MessageHashProof,
    pub public_values: Vec<Goldilocks>,
    pub trace_generation: Duration,
    pub prove: Duration,
    pub trace_height: usize,
}

/// `[n, L0..L3]`; equals `public_values[5..9]` of the ECDSA proof for the same batch.
pub fn public_values(instances: &[EcdsaInstance]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::from_usize(instances.len())];
    out.extend(link_digest(instances.len(), instances));
    out
}

pub fn prove_message_hashes(
    instances: &[EcdsaInstance],
    capacity: usize,
) -> Result<MessageHashProvingOutput> {
    crate::prove::check_capacity(instances.len(), capacity)?;
    let air = MessageHashAir::new(capacity);
    let config = ecdsa_stark_config();
    let start = Instant::now();
    let trace = tracing::info_span!("generate keccak trace")
        .in_scope(|| generate_trace(instances, capacity));
    let trace_generation = start.elapsed();
    let trace_height = trace.values.len() / super::WIDTH;
    let public_values = public_values(instances);
    let start = Instant::now();
    let proof = tracing::info_span!("keccak stark prove")
        .in_scope(|| p3_uni_stark::prove(&config, &air, trace, &public_values));
    Ok(MessageHashProvingOutput {
        proof,
        public_values,
        trace_generation,
        prove: start.elapsed(),
        trace_height,
    })
}

pub fn verify_message_hashes(
    proof: &MessageHashProof,
    public_values: &[Goldilocks],
    capacity: usize,
) -> Result<()> {
    let air = MessageHashAir::new(capacity);
    let expected = air.height().trailing_zeros() as usize;
    if proof.degree_bits != expected {
        return Err(Error::Verification(format!(
            "message-hash proof has degree_bits {}, expected {expected}",
            proof.degree_bits
        )));
    }
    p3_uni_stark::verify(&ecdsa_stark_config(), &air, proof, public_values)
        .map_err(|e| Error::Verification(format!("{e:?}")))
}
