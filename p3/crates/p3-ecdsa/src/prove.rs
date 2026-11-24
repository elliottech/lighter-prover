//! Prove/verify entry points. Uses the exact STARK configuration (FRI
//! profile, Poseidon2-16 transcript) of `p3-schnorr`, so the recursive
//! wrapper's transcript and FRI replay carry over unchanged.

use std::time::{Duration, Instant};

use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
pub use p3_schnorr::proof::SignatureBatchStarkConfig as EcdsaStarkConfig;
use p3_schnorr::proof::stark_config_with_fri;
use p3_uni_stark::{PreprocessedProverData, PreprocessedVerifierKey};

use crate::air::EcdsaBatchAir;
use crate::air::trace::generate_main_trace;
use crate::digest::{digest_instances, link_digest};
use crate::error::{Error, Result};
use crate::stark::{self, LogUpProof};
use crate::types::EcdsaInstance;

pub type EcdsaBatchProof = LogUpProof<EcdsaStarkConfig>;

/// Fixed-capacity proving/verifying key material (preprocessed commitment).
pub struct PreparedEcdsaBatch {
    pub air: EcdsaBatchAir,
    pub config: EcdsaStarkConfig,
    pub prover_data: PreprocessedProverData<EcdsaStarkConfig>,
    pub verifier_key: PreprocessedVerifierKey<EcdsaStarkConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct ProvingTimings {
    pub trace_generation: Duration,
    pub prove: Duration,
}

pub struct EcdsaBatchProvingOutput {
    pub proof: EcdsaBatchProof,
    pub public_values: Vec<Goldilocks>,
    pub timings: ProvingTimings,
    pub trace_height: usize,
}

pub fn public_values(instances: &[EcdsaInstance]) -> Vec<Goldilocks> {
    use p3_field::PrimeCharacteristicRing;
    let mut out = vec![Goldilocks::from_usize(instances.len())];
    out.extend(digest_instances(instances.len(), instances));
    out.extend(link_digest(instances.len(), instances));
    out
}

/// Rejects invalid batches (invalid key or signature).
pub fn check_capacity(len: usize, capacity: usize) -> Result<()> {
    if capacity == 0 {
        return Err(Error::EmptyBatch);
    }
    if len > capacity {
        return Err(Error::CapacityTooSmall { len, capacity });
    }
    Ok(())
}

pub fn validate_batch(instances: &[EcdsaInstance], capacity: usize) -> Result<()> {
    use p3_maybe_rayon::prelude::*;
    check_capacity(instances.len(), capacity)?;
    let _span = tracing::info_span!("validate batch (native ECDSA verify)").entered();
    instances.par_iter().enumerate().try_for_each(|(i, inst)| {
        if inst.public_key_point().is_none() {
            return Err(Error::InvalidPublicKey(i));
        }
        if !inst.verify_reference() {
            return Err(Error::InvalidSignature(i));
        }
        Ok(())
    })
}

/// `log2` of the FRI blowup of the ECDSA and keccak STARKs. Their quotients
/// fit a blowup of 4; the Schnorr AIR's does not and stays at 8.
pub const ECDSA_FRI_RATE_BITS: usize = 2;
/// FRI query rounds at [`ECDSA_FRI_RATE_BITS`]: 42 * 2 + 16 grinding bits keeps
/// the 100 conjectured bits of the Schnorr profile (28 * 3 + 16).
pub const ECDSA_FRI_NUM_QUERY_ROUNDS: usize = 42;

const _: () = {
    let blowup = 1 << ECDSA_FRI_RATE_BITS;
    assert!((crate::air::layout::MAX_CONSTRAINT_DEGREE - 1).next_power_of_two() <= blowup);
    assert!((crate::keccak::MAX_CONSTRAINT_DEGREE - 1).next_power_of_two() <= blowup);
};

/// STARK configuration shared by the ECDSA and keccak AIRs.
pub fn ecdsa_stark_config() -> EcdsaStarkConfig {
    stark_config_with_fri(ECDSA_FRI_RATE_BITS, ECDSA_FRI_NUM_QUERY_ROUNDS)
}

impl PreparedEcdsaBatch {
    pub fn new(capacity: usize) -> Self {
        let air = EcdsaBatchAir::new(capacity);
        let config = ecdsa_stark_config();
        let (prover_data, verifier_key) =
            stark::setup_preprocessed::<EcdsaStarkConfig, _>(&config, &air, air.log_height())
                .expect("ECDSA AIR has preprocessed columns");
        Self {
            air,
            config,
            prover_data,
            verifier_key,
        }
    }

    pub fn capacity(&self) -> usize {
        self.air.capacity
    }

    pub fn prove(&self, instances: &[EcdsaInstance]) -> Result<EcdsaBatchProvingOutput> {
        validate_batch(instances, self.capacity())?;
        let start = Instant::now();
        let trace = tracing::info_span!("generate ecdsa trace")
            .in_scope(|| generate_main_trace(instances, self.capacity()));
        let trace_generation = start.elapsed();
        let trace_height = trace.height();
        let public_values = public_values(instances);
        let start = Instant::now();
        let proof = stark::prove(
            &self.config,
            &self.air,
            trace,
            &public_values,
            Some(&self.prover_data),
        );
        let prove = start.elapsed();
        Ok(EcdsaBatchProvingOutput {
            proof,
            public_values,
            timings: ProvingTimings {
                trace_generation,
                prove,
            },
            trace_height,
        })
    }

    pub fn verify(&self, proof: &EcdsaBatchProof, public_values: &[Goldilocks]) -> Result<()> {
        stark::verify(
            &self.config,
            &self.air,
            proof,
            public_values,
            Some(&self.verifier_key),
        )
        .map_err(|e| Error::Verification(format!("{e:?}")))
    }
}

pub fn prove_ecdsa_batch(
    instances: &[EcdsaInstance],
    capacity: usize,
) -> Result<EcdsaBatchProvingOutput> {
    PreparedEcdsaBatch::new(capacity).prove(instances)
}
