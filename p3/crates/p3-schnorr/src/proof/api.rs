pub fn schnorr_stark_config() -> SignatureBatchStarkConfig {
    let perm = default_goldilocks_poseidon2_16();
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, PLONKY2_FRI_CAP_HEIGHT);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: PLONKY2_FRI_RATE_BITS,
        log_final_poly_len: PLONKY2_FRI_FINAL_POLY_BITS,
        max_log_arity: PLONKY2_FRI_REDUCTION_ARITY_BITS,
        num_queries: PLONKY2_FRI_NUM_QUERY_ROUNDS,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: PLONKY2_FRI_PROOF_OF_WORK_BITS,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(Radix2DitParallel::default(), val_mmcs, fri_params);
    let challenger = DuplexChallenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Generates a trace for `batch` and proves the full Schnorr verification statement.
///
/// This convenience path derives challenge hints through the optional
/// `reference` feature. Without that feature, use
/// [`prove_signature_batch_with_challenge_hints`].
pub fn prove_signature_batch(
    batch: &Batch,
) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
    let public_inputs = SignatureBatchPublicInputs::from_batch(batch)?;
    let output = prove_signature_batch_trace(generate_trace(batch)?)?;
    if output.public_inputs != public_inputs {
        return Err(SignatureBatchProofError::PublicInputMismatch);
    }
    Ok(output)
}

/// [`prove_signature_batch`], but sizing the trace for `capacity` signature
/// slots (see [`generate_trace_with_capacity`]): the proof's shape matches a
/// full `capacity`-sized batch while the public signature count and digest
/// cover only the real instances in `batch`.
pub fn prove_signature_batch_with_capacity(
    batch: &Batch,
    capacity: usize,
) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
    let public_inputs = SignatureBatchPublicInputs::from_batch(batch)?;
    let output = prove_signature_batch_trace(generate_trace_with_capacity(batch, capacity)?)?;
    if output.public_inputs != public_inputs {
        return Err(SignatureBatchProofError::PublicInputMismatch);
    }
    Ok(output)
}

/// Proves `batch` using caller-supplied EC/challenge witnesses.
///
/// This is the production prover entry point when the `reference` feature is
/// disabled or when hints are produced by an external implementation.
pub fn prove_signature_batch_with_challenge_hints(
    batch: &Batch,
    challenge_hints: &[SignatureChallengeHint],
) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
    let public_inputs = SignatureBatchPublicInputs::from_batch(batch)?;
    let output =
        prove_signature_batch_trace(generate_trace_with_challenge_hints(batch, challenge_hints)?)?;
    if output.public_inputs != public_inputs {
        return Err(SignatureBatchProofError::PublicInputMismatch);
    }
    Ok(output)
}

/// [`prove_signature_batch_with_challenge_hints`], but sizing the trace for
/// `capacity` signature slots (see [`prove_signature_batch_with_capacity`]).
pub fn prove_signature_batch_with_challenge_hints_and_capacity(
    batch: &Batch,
    challenge_hints: &[SignatureChallengeHint],
    capacity: usize,
) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
    let public_inputs = SignatureBatchPublicInputs::from_batch(batch)?;
    let output = prove_signature_batch_trace(generate_trace_with_challenge_hints_and_capacity(
        batch,
        challenge_hints,
        capacity,
    )?)?;
    if output.public_inputs != public_inputs {
        return Err(SignatureBatchProofError::PublicInputMismatch);
    }
    Ok(output)
}

/// Validates a public Poseidon digest and derives reference challenge hints.
///
/// This helper is intended for tests, fixture generation, and integrations that
/// deliberately opt into the `reference` feature. Production systems may use
/// the same output shape, but should usually obtain hints from their own
/// witness generator.
#[cfg(feature = "reference")]
pub fn prepare_signature_batch_with_public_digest(
    instances: Vec<crate::types::SchnorrInstance>,
    expected_digest: [Goldilocks; PUBLIC_DIGEST_WIDTH],
) -> Result<PreparedSignatureBatch, SignatureBatchProofError> {
    let batch = Batch::new(instances)?;
    let public_inputs = SignatureBatchPublicInputs::from_batch(&batch)?;
    let actual_digest = *public_inputs.digest();
    if actual_digest != expected_digest {
        return Err(SignatureBatchProofError::PublicDigestMismatch {
            expected: Box::new(expected_digest),
            actual: Box::new(actual_digest),
        });
    }

    let challenge_hints = batch
        .instances()
        .iter()
        .map(|instance| {
            let witness = crate::schnorr::verification_witness(instance)?;
            if !witness.is_valid {
                return Err(crate::Error::ChallengeScalarMismatch);
            }
            SignatureChallengeHint::from_reference(instance, witness.reconstructed_r)
        })
        .collect::<crate::Result<Vec<_>>>()?;

    Ok(PreparedSignatureBatch {
        batch,
        public_inputs,
        challenge_hints,
    })
}

/// Proves an already-generated trace.
///
/// This is primarily for benchmarks and low-level tests. Production callers
/// should prefer [`prove_signature_batch`] so the batch digest and trace are
/// derived through one checked path.
pub fn prove_signature_batch_trace(
    trace: SignatureBatchTrace,
) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
    let public_inputs = SignatureBatchPublicInputs::from_public_values(&trace.public_values)?;
    ensure_trace_height(&public_inputs, trace.rows)?;
    let log_height = trace.rows.ilog2() as usize;
    let config = schnorr_stark_config();
    let (preprocessed, _) =
        setup_preprocessed::<SignatureBatchStarkConfig, _>(&config, &trace.air, log_height)
            .ok_or(SignatureBatchProofError::MissingPreprocessedColumns)?;
    let inner = prove_with_preprocessed(
        &config,
        &trace.air,
        trace.trace,
        &trace.public_values,
        Some(&preprocessed),
    );
    Ok(SignatureBatchProvingOutput {
        proof: SignatureBatchProof { inner, log_height },
        public_inputs,
    })
}

/// Verifies that `proof` proves the statement represented by `public_inputs`.
pub fn verify_signature_batch_proof(
    public_inputs: &SignatureBatchPublicInputs,
    proof: &SignatureBatchProof,
) -> Result<(), SignatureBatchProofError> {
    ensure_trace_height(public_inputs, proof.rows())?;
    let verifier = prepare_signature_batch_verifier_for_rows(proof.rows())?;
    verifier.verify(public_inputs, proof)
}

/// Precomputes verifier-side AIR data for a given signature count / trace height.
pub fn prepare_signature_batch_verifier(
    public_inputs: &SignatureBatchPublicInputs,
) -> Result<PreparedSignatureBatchVerifier, SignatureBatchProofError> {
    prepare_signature_batch_verifier_for_rows(public_inputs.expected_rows())
}

/// Precomputes verifier-side AIR data for a fixed trace height, independent
/// of any particular proof's signature count. Used when the trace height is
/// pinned by a capacity (see [`prove_signature_batch_with_capacity`]) rather
/// than derived from the batch size.
pub fn prepare_signature_batch_verifier_for_capacity(
    capacity: usize,
) -> Result<PreparedSignatureBatchVerifier, SignatureBatchProofError> {
    if capacity == 0 || capacity > MAX_SIGNATURES {
        return Err(SignatureBatchProofError::InvalidSignatureCount {
            count: capacity,
            max: MAX_SIGNATURES,
        });
    }
    prepare_signature_batch_verifier_for_rows(trace_height_for_count(capacity))
}

fn prepare_signature_batch_verifier_for_rows(
    rows: usize,
) -> Result<PreparedSignatureBatchVerifier, SignatureBatchProofError> {
    let log_height = rows.ilog2() as usize;
    let config = schnorr_stark_config();
    let air = SignatureBatchAir::new(rows);
    let (_, preprocessed_vk) =
        setup_preprocessed::<SignatureBatchStarkConfig, _>(&config, &air, log_height)
            .ok_or(SignatureBatchProofError::MissingPreprocessedColumns)?;
    Ok(PreparedSignatureBatchVerifier {
        air,
        log_height,
        preprocessed_vk,
    })
}

/// Verifies `proof` using a caller-supplied preprocessed verifier key wrapper.
pub fn verify_signature_batch_proof_preprocessed(
    verifier: &PreparedSignatureBatchVerifier,
    public_inputs: &SignatureBatchPublicInputs,
    proof: &SignatureBatchProof,
) -> Result<(), SignatureBatchProofError> {
    verifier.verify(public_inputs, proof)
}

fn ensure_trace_height(
    public_inputs: &SignatureBatchPublicInputs,
    actual_rows: usize,
) -> Result<(), SignatureBatchProofError> {
    let expected_rows = public_inputs.expected_rows();
    let max_rows = trace_height_for_count(MAX_SIGNATURES);
    if !actual_rows.is_power_of_two() || actual_rows < expected_rows || actual_rows > max_rows {
        return Err(SignatureBatchProofError::TraceHeightMismatch {
            signature_count: public_inputs.signature_count(),
            expected_rows,
            actual_rows,
        });
    }
    Ok(())
}
