use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_schnorr::affine::AffinePoint;
use p3_schnorr::air::{
    COL_BLOCK, COL_CHALLENGE_BLOCK, COL_CHALLENGE_DIGEST, COL_EC_FINAL, COL_EC_NEXT_ACC_X,
    COL_EC_S_BIT, COL_EC_S_BITS, COL_ENABLED, COL_HASH_ACTIVE, COL_PK_AFFINE_X,
    COL_PK_DECODE_INVERSE, COL_PK_DECODE_OTHER_ROOT_SQRT, COL_R_AFFINE_X, COL_R_DECODE_SQRT_DELTA,
    COL_REDUCTION_DIGEST, COL_REDUCTION_DIGEST_BIT, COL_REDUCTION_DIGEST_U32,
    COL_REDUCTION_E_CHUNKS, COL_REDUCTION_Q, COL_STATE_AFTER, COL_STATE_BEFORE, EC_SCALAR_WINDOWS,
    EC_WINDOW_BITS, PUBLIC_DIGEST_WIDTH, PUBLIC_HASH_ROWS_PER_SIGNATURE, ROWS_PER_SIGNATURE,
    SignatureBatchAir, SignatureBatchTrace, TRACE_WIDTH, check_constraints, generate_trace,
};
use p3_schnorr::hash::poseidon_public_digest;
use p3_schnorr::lighter_poseidon::LIGHTER_POSEIDON_DIGEST_WIDTH;
use p3_schnorr::poseidon::{POSEIDON_ROUND_STATE_COUNT, POSEIDON_WIDTH, initial_public_hash_state};
use p3_schnorr::proof::{
    SignatureBatchPublicInputs, prove_signature_batch_trace, verify_signature_batch_proof,
};
use p3_schnorr::reference::deterministic_batch;
use p3_schnorr::schnorr::verification_witness;
use p3_schnorr::types::{FP5_BYTES, SCALAR_U32_LIMBS};
use p3_schnorr::{Batch, Error, SignatureBytes};

#[test]
fn deterministic_reference_batch_hashes_and_traces() {
    let batch = deterministic_batch(2).unwrap();
    let digest = poseidon_public_digest(&batch).unwrap();
    assert_ne!(digest[0], digest[1]);

    let trace = generate_trace(&batch).unwrap();
    check_constraints(&trace);

    let output = prove_signature_batch_trace(trace).unwrap();
    verify_signature_batch_proof(&output.public_inputs, &output.proof).unwrap();
}

#[test]
fn proof_rejects_wrong_public_digest() {
    let batch = deterministic_batch(2).unwrap();
    let output = prove_signature_batch_trace(generate_trace(&batch).unwrap()).unwrap();
    let mut digest = *output.public_inputs.digest();
    digest[0] += Goldilocks::ONE;
    let wrong_public_inputs =
        SignatureBatchPublicInputs::new(output.public_inputs.signature_count(), digest).unwrap();

    assert!(verify_signature_batch_proof(&wrong_public_inputs, &output.proof).is_err());
}

#[test]
fn proof_rejects_wrong_signature_count() {
    let batch = deterministic_batch(2).unwrap();
    let output = prove_signature_batch_trace(generate_trace(&batch).unwrap()).unwrap();
    let wrong_public_inputs =
        SignatureBatchPublicInputs::new(1, *output.public_inputs.digest()).unwrap();

    assert!(verify_signature_batch_proof(&wrong_public_inputs, &output.proof).is_err());
}

#[test]
fn constraints_accept_disabled_zero_count_trace() {
    let state = initial_public_hash_state(0);
    let mut values = Goldilocks::zero_vec(TRACE_WIDTH);
    values[COL_STATE_BEFORE..COL_STATE_BEFORE + POSEIDON_WIDTH].copy_from_slice(&state);
    values[COL_STATE_AFTER..COL_STATE_AFTER + POSEIDON_WIDTH].copy_from_slice(&state);

    let mut public_values = Vec::with_capacity(1 + PUBLIC_DIGEST_WIDTH);
    public_values.push(Goldilocks::ZERO);
    public_values.extend(state[..PUBLIC_DIGEST_WIDTH].iter().copied());

    let trace = SignatureBatchTrace {
        air: SignatureBatchAir::new(1),
        trace: p3_matrix::dense::RowMajorMatrix::new(values, TRACE_WIDTH),
        public_values,
        rows: 1,
        columns: TRACE_WIDTH,
    };

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_enabled_zero_count_trace() {
    let state = initial_public_hash_state(0);
    let mut values = Goldilocks::zero_vec(TRACE_WIDTH);
    values[COL_ENABLED] = Goldilocks::ONE;
    values[COL_EC_FINAL] = Goldilocks::ONE;
    values[COL_STATE_BEFORE..COL_STATE_BEFORE + POSEIDON_WIDTH].copy_from_slice(&state);
    values[COL_STATE_AFTER..COL_STATE_AFTER + POSEIDON_WIDTH].copy_from_slice(&state);

    let mut public_values = Vec::with_capacity(1 + PUBLIC_DIGEST_WIDTH);
    public_values.push(Goldilocks::ZERO);
    public_values.extend(state[..PUBLIC_DIGEST_WIDTH].iter().copied());

    let trace = SignatureBatchTrace {
        air: SignatureBatchAir::new(1),
        trace: p3_matrix::dense::RowMajorMatrix::new(values, TRACE_WIDTH),
        public_values,
        rows: 1,
        columns: TRACE_WIDTH,
    };

    check_constraints(&trace);
}

#[test]
fn schnorr_reference_witness_matches_goldilocks_crypto_verify() {
    let batch = deterministic_batch(4).unwrap();

    for instance in batch.instances() {
        let witness = verification_witness(instance).unwrap();
        let upstream = goldilocks_crypto::verify_signature(
            instance.signature.as_bytes(),
            instance.msg_digest.as_bytes(),
            instance.public_key.as_bytes(),
        )
        .unwrap();

        assert!(witness.is_valid);
        assert_eq!(witness.signature_e_bytes, witness.challenge_scalar_bytes);
        assert_eq!(witness.is_valid, upstream);
    }
}

#[test]
fn trace_challenge_digest_matches_reference_witness() {
    let batch = deterministic_batch(3).unwrap();
    let trace = generate_trace(&batch).unwrap();

    for (signature_index, instance) in batch.instances().iter().enumerate() {
        let witness = verification_witness(instance).unwrap();
        let challenge_row =
            signature_index * ROWS_PER_SIGNATURE + 2 * POSEIDON_ROUND_STATE_COUNT - 1;
        let offset = challenge_row * trace.columns + COL_CHALLENGE_DIGEST;

        assert_eq!(
            &trace.trace.values[offset..offset + LIGHTER_POSEIDON_DIGEST_WIDTH],
            &witness.challenge_fp5
        );
    }
}

#[test]
fn trace_scalar_reduction_matches_reference_witness() {
    let batch = deterministic_batch(3).unwrap();
    let trace = generate_trace(&batch).unwrap();

    for (signature_index, instance) in batch.instances().iter().enumerate() {
        let witness = verification_witness(instance).unwrap();
        let reduction_row =
            signature_index * ROWS_PER_SIGNATURE + 4 * POSEIDON_ROUND_STATE_COUNT - 1;
        let offset = reduction_row * trace.columns;

        assert_eq!(
            &trace.trace.values
                [offset + COL_REDUCTION_DIGEST..offset + COL_REDUCTION_DIGEST + FP5_BYTES / 8],
            &witness.challenge_fp5
        );

        let actual_digest_u32 = core::array::from_fn::<_, SCALAR_U32_LIMBS, _>(|limb| {
            trace.trace.values[offset + COL_REDUCTION_DIGEST_U32 + limb].as_canonical_u64()
        });
        assert_eq!(actual_digest_u32, fp5_limbs_to_u32(&witness.challenge_fp5));

        let actual_e_chunks = core::array::from_fn::<_, SCALAR_U32_LIMBS, _>(|limb| {
            trace.trace.values[offset + COL_REDUCTION_E_CHUNKS + limb].as_canonical_u64()
        });
        assert_eq!(
            actual_e_chunks,
            bytes_to_u32_chunks(&witness.challenge_scalar_bytes)
        );
        assert!(trace.trace.values[offset + COL_REDUCTION_Q].as_canonical_u64() <= 2);
    }
}

#[test]
fn trace_decoded_affine_points_are_valid() {
    let batch = deterministic_batch(2).unwrap();
    let trace = generate_trace(&batch).unwrap();

    for (signature_index, instance) in batch.instances().iter().enumerate() {
        let witness = verification_witness(instance).unwrap();
        let r_row = signature_index * ROWS_PER_SIGNATURE;
        let r_offset = r_row * trace.columns;
        let r_x = core::array::from_fn(|lane| trace.trace.values[r_offset + COL_R_AFFINE_X + lane]);

        assert!(AffinePoint::encoded_x_is_valid(
            &witness.reconstructed_r,
            &r_x
        ));
        assert!(
            AffinePoint::from_encoded_x(witness.reconstructed_r, r_x)
                .unwrap()
                .is_on_curve()
        );

        let pk_row = signature_index * ROWS_PER_SIGNATURE + PUBLIC_HASH_ROWS_PER_SIGNATURE - 1;
        let pk_offset = pk_row * trace.columns;
        let pk_x =
            core::array::from_fn(|lane| trace.trace.values[pk_offset + COL_PK_AFFINE_X + lane]);
        let pk_encoded = instance
            .public_key
            .to_goldilocks_limbs("public_key")
            .unwrap();

        assert!(AffinePoint::encoded_x_is_valid(&pk_encoded, &pk_x));
        assert!(
            AffinePoint::from_encoded_x(pk_encoded, pk_x)
                .unwrap()
                .is_on_curve()
        );
    }
}

#[test]
fn generate_trace_rejects_wrong_signature_challenge_scalar() {
    let mut instance = deterministic_batch(1).unwrap().instances()[0];
    let mut signature = *instance.signature.as_bytes();
    signature[FP5_BYTES] ^= 1;
    instance.signature = SignatureBytes::from_array(signature);

    let batch = Batch::new(vec![instance]).unwrap();
    let err = generate_trace(&batch).unwrap_err();

    assert!(matches!(err, Error::ChallengeScalarMismatch));
}

#[test]
#[should_panic]
fn constraints_reject_bad_reconstructed_r_affine_x() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();

    trace.trace.values[COL_R_AFFINE_X] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_reconstructed_r_decode_witness() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();

    trace.trace.values[COL_R_DECODE_SQRT_DELTA] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_public_block_changed_within_poseidon_phase() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let second_round_row = 1;
    let offset = second_round_row * trace.columns + COL_BLOCK + 4;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_challenge_block_changed_within_poseidon_phase() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let second_round_row = 1;
    let offset = second_round_row * trace.columns + COL_CHALLENGE_BLOCK + 1;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_public_key_block_changed_before_final_phase_round() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let phase_three_second_round = 3 * POSEIDON_ROUND_STATE_COUNT + 1;
    let offset = phase_three_second_round * trace.columns + COL_BLOCK + 1;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_public_key_affine_x() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let public_key_row = PUBLIC_HASH_ROWS_PER_SIGNATURE - 1;
    let offset = public_key_row * trace.columns + COL_PK_AFFINE_X;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_public_key_decode_witness() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let public_key_row = PUBLIC_HASH_ROWS_PER_SIGNATURE - 1;
    let offset = public_key_row * trace.columns + COL_PK_DECODE_INVERSE;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_public_key_root_choice_witness() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let public_key_row = PUBLIC_HASH_ROWS_PER_SIGNATURE - 1;
    let offset = public_key_row * trace.columns + COL_PK_DECODE_OTHER_ROOT_SQRT;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_ec_scalar_bit() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let first_ec_row = PUBLIC_HASH_ROWS_PER_SIGNATURE;
    let offset = first_ec_row * trace.columns + COL_EC_S_BIT;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_reduction_digest_range_bit() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let first_ec_row = PUBLIC_HASH_ROWS_PER_SIGNATURE;
    let offset = first_ec_row * trace.columns + COL_REDUCTION_DIGEST_BIT;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_hash_active_reasserted_on_first_ec_row() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let first_ec_row = PUBLIC_HASH_ROWS_PER_SIGNATURE;
    let offset = first_ec_row * trace.columns + COL_HASH_ACTIVE;

    trace.trace.values[offset] = Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_reduction_digest_u32_changed_on_ec_row() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let first_ec_row = PUBLIC_HASH_ROWS_PER_SIGNATURE;
    let offset = first_ec_row * trace.columns + COL_REDUCTION_DIGEST_U32;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_bad_schnorr_ec_result() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let final_ec_row = ROWS_PER_SIGNATURE - 1;
    let offset = final_ec_row * trace.columns + COL_EC_NEXT_ACC_X;

    trace.trace.values[offset] += Goldilocks::ONE;

    check_constraints(&trace);
}

#[test]
#[should_panic]
fn constraints_reject_trace_that_stops_before_final_ec_row() {
    let batch = deterministic_batch(1).unwrap();
    let mut trace = generate_trace(&batch).unwrap();
    let truncated_rows = 8;

    trace.trace.values.truncate(truncated_rows * trace.columns);
    trace.trace = p3_matrix::dense::RowMajorMatrix::new(trace.trace.values, trace.columns);
    trace.rows = truncated_rows;

    check_constraints(&trace);
}

#[test]
fn schnorr_reference_witness_rejects_tampered_message() {
    let mut instance = deterministic_batch(1).unwrap().instances()[0];
    instance.msg_digest.0[0] ^= 1;

    let witness = verification_witness(&instance).unwrap();
    let upstream = goldilocks_crypto::verify_signature(
        instance.signature.as_bytes(),
        instance.msg_digest.as_bytes(),
        instance.public_key.as_bytes(),
    )
    .unwrap();

    assert!(!witness.is_valid);
    assert_eq!(witness.is_valid, upstream);
}

fn fp5_limbs_to_u32(limbs: &[p3_goldilocks::Goldilocks; FP5_BYTES / 8]) -> [u64; SCALAR_U32_LIMBS] {
    core::array::from_fn(|limb| {
        let value = limbs[limb / 2].as_canonical_u64();
        if limb % 2 == 0 {
            value & 0xffff_ffff
        } else {
            value >> 32
        }
    })
}

fn bytes_to_u32_chunks(bytes: &[u8; FP5_BYTES]) -> [u64; SCALAR_U32_LIMBS] {
    core::array::from_fn(|limb| {
        u32::from_le_bytes(
            bytes[4 * limb..4 * limb + 4]
                .try_into()
                .expect("scalar bytes split into u32 chunks"),
        ) as u64
    })
}

/// Adversarial probe: instead of asserting `ec_final = 1` on signature 0's
/// true last EC row, delay it and "coast" the accumulator (`s_bits = e_bits
/// = 0`, i.e. every double-and-add step is a no-op) through what would be
/// signature 1's hashing rows, then finally assert `ec_final = 1` at the
/// next row whose *preprocessed* window/chunk selectors say "last window of
/// last chunk" — which, since the preprocessed table repeats with period
/// `ROWS_PER_SIGNATURE`, is exactly `ROWS_PER_SIGNATURE` rows later, deep
/// inside signature 1's row range. The accumulator itself never moves
/// during the coast (every added "point" is the preprocessed-zeroed `s*G`
/// addend with `add_bit = 0`, a true no-op), so it still equals the genuine
/// `s*G + e*pk` for signature 0 when `ec_final` finally fires — the
/// candidate gap is whether anything else in the AIR notices that
/// signature 1 never actually ran its hashing phase.
///
/// It must still be rejected: `COL_REMAINING_BLOCKS` only decrements on a
/// genuine `hash_active * round_last` event, so skipping signature 1's
/// hashing leaves it short of the public signature count by
/// `POSEIDON_BLOCKS_PER_INSTANCE`, and `when_last_row` requires it to be
/// exactly zero.
#[test]
#[should_panic]
fn constraints_reject_ec_phase_coasting_into_next_signature_hash_rows() {
    let batch = deterministic_batch(2).unwrap();
    let mut trace = generate_trace(&batch).unwrap();

    let sig0_final_ec_row = PUBLIC_HASH_ROWS_PER_SIGNATURE + EC_SCALAR_WINDOWS - 1;
    let coast_target_row = sig0_final_ec_row + ROWS_PER_SIGNATURE;
    assert!(
        coast_target_row < trace.rows,
        "test batch must be large enough to coast a full signature's worth of rows"
    );

    // Clear ec_final on the true final EC row, and zero out its s/e window
    // bits so the (now-unused) bit-validity/chunk-accumulator constraints
    // for that row stay satisfied trivially.
    let true_final_offset = sig0_final_ec_row * trace.columns;
    trace.trace.values[true_final_offset + COL_EC_FINAL] = Goldilocks::ZERO;
    for bit in 0..EC_WINDOW_BITS {
        trace.trace.values[true_final_offset + COL_EC_S_BITS + bit] = Goldilocks::ZERO;
    }

    // Coast through every row from just after the true final EC row up to
    // (but not including) the row where we'll re-assert ec_final: keep
    // enabled = 1, hash_active = 0 (stay "EC active"), and zero bits so
    // every double-and-add step is a no-op and the accumulator doesn't move.
    for row in (sig0_final_ec_row + 1)..coast_target_row {
        let offset = row * trace.columns;
        trace.trace.values[offset + COL_ENABLED] = Goldilocks::ONE;
        trace.trace.values[offset + COL_HASH_ACTIVE] = Goldilocks::ZERO;
        trace.trace.values[offset + COL_EC_FINAL] = Goldilocks::ZERO;
        for bit in 0..EC_WINDOW_BITS {
            trace.trace.values[offset + COL_EC_S_BITS + bit] = Goldilocks::ZERO;
        }
    }

    // Finally land on a row whose preprocessed window/chunk selectors are
    // genuinely "last window of last chunk" (true every ROWS_PER_SIGNATURE
    // rows) and re-assert ec_final there, with the accumulator carried
    // forward unchanged from the true final row.
    let coast_offset = coast_target_row * trace.columns;
    trace.trace.values[coast_offset + COL_ENABLED] = Goldilocks::ONE;
    trace.trace.values[coast_offset + COL_HASH_ACTIVE] = Goldilocks::ZERO;
    trace.trace.values[coast_offset + COL_EC_FINAL] = Goldilocks::ONE;
    for bit in 0..EC_WINDOW_BITS {
        trace.trace.values[coast_offset + COL_EC_S_BITS + bit] = Goldilocks::ZERO;
    }

    check_constraints(&trace);
}
