#[cfg(all(test, feature = "reference"))]
mod tests {
    use super::*;
    use p3_air::symbolic::{AirLayout, get_symbolic_constraints};

    use crate::reference::deterministic_batch;
    use crate::schnorr::verification_witness;
    use crate::types::FP5_BYTES;
    use crate::types::SignatureBytes;

    fn challenge_hints(batch: &Batch) -> Vec<SignatureChallengeHint> {
        batch
            .instances()
            .iter()
            .map(|instance| {
                let witness = verification_witness(instance).unwrap();
                SignatureChallengeHint::from_reference(instance, witness.reconstructed_r).unwrap()
            })
            .collect()
    }

    #[test]
    fn signature_batch_constraint_degree_fits_plonky2_fri_rate() {
        let air = SignatureBatchAir::new(1024);
        let layout = AirLayout {
            preprocessed_width: PREPROCESSED_WIDTH,
            main_width: TRACE_WIDTH,
            num_public_values: PUBLIC_VALUES,
            ..Default::default()
        };
        let constraints = get_symbolic_constraints(&air, layout);
        let mut degrees = constraints
            .iter()
            .enumerate()
            .map(|(index, constraint)| (constraint.degree_multiple(), index))
            .collect::<Vec<_>>();
        degrees.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(
            degrees[0].0,
            <SignatureBatchAir as BaseAir<Goldilocks>>::max_constraint_degree(&air).unwrap()
        );
        assert!(degrees[0].0 <= 9);
    }

    #[test]
    fn signature_batch_constraints_accept_generated_trace() {
        let batch = deterministic_batch(3).unwrap();
        let trace = generate_trace(&batch).unwrap();

        assert_eq!(trace.rows, 1024);
        assert_eq!(TRACE_WIDTH, 468);
        assert_eq!(PREPROCESSED_WIDTH, 90);
        assert_eq!(trace.columns, TRACE_WIDTH);
        check_constraints(&trace);
    }

    #[test]
    fn signature_challenge_hint_wire_round_trip() {
        let batch = deterministic_batch(1).unwrap();
        let hints = challenge_hints(&batch);
        let bytes = hints[0].to_bytes();

        assert_eq!(bytes.len(), SIGNATURE_CHALLENGE_HINT_BYTES);
        assert_eq!(
            SignatureChallengeHint::from_bytes(&bytes).unwrap(),
            hints[0]
        );
    }

    #[test]
    fn signature_challenge_hint_batch_wire_round_trip_generates_trace() {
        let batch = deterministic_batch(3).unwrap();
        let hints = challenge_hints(&batch);
        let bytes = signature_challenge_hints_to_bytes(&hints).unwrap();
        let decoded = signature_challenge_hints_from_bytes(&bytes).unwrap();

        assert_eq!(decoded, hints);
        let trace = generate_trace_with_challenge_hints(&batch, &decoded).unwrap();
        check_constraints(&trace);
    }

    #[test]
    fn signature_challenge_hint_wire_rejects_bad_version() {
        let batch = deterministic_batch(1).unwrap();
        let hints = challenge_hints(&batch);
        let mut bytes = hints[0].to_bytes();
        bytes[SIGNATURE_CHALLENGE_HINT_WIRE_MAGIC.len()] ^= 1;

        assert!(matches!(
            SignatureChallengeHint::from_bytes(&bytes),
            Err(crate::Error::UnsupportedChallengeHintWireFormatVersion { .. })
        ));
    }

    #[test]
    fn signature_challenge_hint_wire_rejects_noncanonical_limb() {
        let batch = deterministic_batch(1).unwrap();
        let hints = challenge_hints(&batch);
        let mut bytes = hints[0].to_bytes();
        let limb_offset = SIGNATURE_CHALLENGE_HINT_WIRE_MAGIC.len() + 2 + 2;
        bytes[limb_offset..limb_offset + 8].copy_from_slice(&Goldilocks::ORDER_U64.to_le_bytes());

        assert!(matches!(
            SignatureChallengeHint::from_bytes(&bytes),
            Err(crate::Error::NonCanonicalGoldilocksLimb {
                field: "reconstructed_r",
                limb: 0,
                ..
            })
        ));
    }

    /// Regression coverage for the two gaps closed alongside this test:
    /// `ec_final` was previously only forward-implied by (never implying)
    /// the fixed last window/last chunk, and `ec_active` was never bound to
    /// the preprocessed EC-selector region actually being live. Either gap
    /// let a prover clear `COL_EC_FINAL` on a signature's true last EC row
    /// and coast the accumulator into the next signature's (preprocessed)
    /// hashing rows, where its `e`-bits went unrange-checked against
    /// `sig_e_chunks`. Both attempts must now fail on the row where the
    /// tamper happens, not just downstream in the continuation transition.
    #[test]
    fn constraints_reject_ec_final_cleared_or_spilled_at_last_window() {
        use p3_air::check_all_constraints;

        let batch = deterministic_batch(3).unwrap();
        let trace = generate_trace(&batch).unwrap();
        check_constraints(&trace);

        let idx = |row: usize, col: usize| row * TRACE_WIDTH + col;
        let sig0_last_window = ROWS_PER_SIGNATURE - 1; // 203
        let sig1_first_hash = ROWS_PER_SIGNATURE; // 204

        // Clearing ec_final on the fixed last window is rejected on that
        // same row (the new `ec_active * chunk_end * last_chunk_selector *
        // (ec_final - 1)` constraint in `constrain_ec_row`).
        let mut tampered = trace.clone();
        tampered.trace.values[idx(sig0_last_window, COL_EC_FINAL)] = Goldilocks::ZERO;
        let report =
            check_all_constraints(&tampered.air, &tampered.trace, &tampered.public_values, None);
        assert!(!report.is_ok());
        assert!(report.failures.iter().any(|f| f.row == sig0_last_window));

        // Additionally forcing hash_active=0 on the next signature's first
        // (preprocessed hash-pattern) row, to claim it as an EC row and
        // spill the accumulator into it, is independently rejected (the new
        // `ec_active * (bit_selector_sum - 1)` binding in `eval.rs`).
        let mut spilled = trace.clone();
        spilled.trace.values[idx(sig0_last_window, COL_EC_FINAL)] = Goldilocks::ZERO;
        spilled.trace.values[idx(sig1_first_hash, COL_HASH_ACTIVE)] = Goldilocks::ZERO;
        let report2 =
            check_all_constraints(&spilled.air, &spilled.trace, &spilled.public_values, None);
        assert!(!report2.is_ok());
    }

    /// Regression coverage for the missing `e < SCALAR_MODULUS_U32`
    /// canonicality check: previously the reduction only proved `digest ===
    /// e (mod scalar_order)` via `q in {0,1,2}`, never that `e` itself was
    /// the *smallest* such representative, so a prover could claim `q = 0`
    /// with `e` set to a digest that is itself `>= scalar_order` (i.e. `e`
    /// congruent to, but not equal to, the true canonical scalar). Directly
    /// exercises `scalar_reduction_witness`'s host-side rejection, since
    /// constructing an end-to-end forged signature would additionally
    /// require re-deriving a consistent challenge hash for a chosen digest
    /// value, which isn't necessary to prove the missing check is closed.
    #[test]
    fn scalar_reduction_witness_rejects_non_canonical_e() {
        // digest == SCALAR_MODULUS_U32 exactly, reinterpreted as 5 64-bit
        // limbs (low/high 32-bit chunk pairs) the way `split_digest_u32`
        // expects. e == SCALAR_MODULUS_U32 is congruent to this digest with
        // q=0, but e == modulus is not < modulus, so it must be rejected.
        let digest_at_modulus: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH] = core::array::from_fn(
            |lane| {
                Goldilocks::from_u64(
                    SCALAR_MODULUS_U32[2 * lane] | (SCALAR_MODULUS_U32[2 * lane + 1] << 32),
                )
            },
        );
        let e_chunks = SCALAR_MODULUS_U32.map(Goldilocks::from_u64);

        let err = scalar_reduction_witness(digest_at_modulus, e_chunks).unwrap_err();
        assert!(matches!(
            err,
            crate::Error::NonCanonicalSignatureScalar { field: "e" }
        ));

        // Sanity: e = modulus - 1 (canonical, the largest valid
        // representative, congruent to a digest of the same value with q=0)
        // is accepted.
        let mut e_minus_one = SCALAR_MODULUS_U32;
        e_minus_one[0] -= 1;
        let e_minus_one_chunks = e_minus_one.map(Goldilocks::from_u64);
        let digest_at_modulus_minus_one: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH] =
            core::array::from_fn(|lane| {
                Goldilocks::from_u64(e_minus_one[2 * lane] | (e_minus_one[2 * lane + 1] << 32))
            });
        scalar_reduction_witness(digest_at_modulus_minus_one, e_minus_one_chunks)
            .expect("e = modulus - 1 is canonical and must be accepted");
    }

    /// Regression coverage for the symmetric `s < SCALAR_MODULUS_U32` gap:
    /// unlike `e`, `s` never goes through a reduction against an external
    /// digest, so it had no canonicality check at all prior to
    /// `require_scalar_lt_modulus` being wired into `prepare_signature_trace`.
    /// Without it, `s' = s + SCALAR_MODULUS_U32` (which still fits the
    /// 320-bit wire encoding for roughly half of all valid `s`, since the
    /// modulus is only 319 bits) satisfies `s'*G == s*G`, so a second,
    /// distinct wire-level signature would verify for the same `(msg, pk)` —
    /// a malleability gap, not merely a cosmetic one. This test sets `s`
    /// directly to `SCALAR_MODULUS_U32` (trivially `>= modulus`) and confirms
    /// trace generation now rejects it instead of silently accepting a
    /// non-canonical `s`.
    #[test]
    fn generate_trace_rejects_non_canonical_s() {
        let mut instance = crate::reference::deterministic_instance(1).unwrap();
        let mut signature_bytes = *instance.signature.as_bytes();
        for (limb, chunk) in signature_bytes[..FP5_BYTES].chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&(SCALAR_MODULUS_U32[limb] as u32).to_le_bytes());
        }
        instance.signature = SignatureBytes::from_array(signature_bytes);
        let batch = crate::types::Batch::new(vec![instance]).unwrap();

        let err = generate_trace(&batch).unwrap_err();
        assert!(matches!(
            err,
            crate::Error::NonCanonicalSignatureScalar { field: "s" }
        ));
    }

    /// In-circuit counterpart of `generate_trace_rejects_non_canonical_s`:
    /// confirms the canonicality gadget for `s` is load-bearing in the AIR
    /// itself (not just the host-side witness generator) by flipping the
    /// same top-bit column `constraints_reject_e_lt_modulus_borrow_flipped_at_last_window`
    /// flips for `e`, but for `s`'s columns instead.
    #[test]
    fn constraints_reject_s_lt_modulus_borrow_flipped_at_last_window() {
        use p3_air::check_all_constraints;

        let batch = deterministic_batch(3).unwrap();
        let trace = generate_trace(&batch).unwrap();
        check_constraints(&trace);

        let idx = |row: usize, col: usize| row * TRACE_WIDTH + col;
        let sig0_last_window = ROWS_PER_SIGNATURE - 1; // 203
        let top_bit_col = scalar_lt_modulus_top_bit_col(COL_EC_S_LT_MODULUS_DIFF_BITS);

        assert_eq!(
            trace.trace.values[idx(sig0_last_window, top_bit_col)],
            Goldilocks::ZERO,
            "honest trace's final EC window must show borrow-out = 1 (s < modulus), i.e. top diff bit clear"
        );

        let mut tampered = trace.clone();
        tampered.trace.values[idx(sig0_last_window, top_bit_col)] = Goldilocks::ONE;
        let report =
            check_all_constraints(&tampered.air, &tampered.trace, &tampered.public_values, None);
        assert!(!report.is_ok());
        assert!(report.failures.iter().any(|f| f.row == sig0_last_window));
    }

    /// In-circuit counterpart of `scalar_reduction_witness_rejects_non_canonical_e`:
    /// an honest trace's final EC window has `COL_EC_E_LT_MODULUS_DIFF_BITS`'s
    /// top bit clear (`borrow_out = 1 - top_bit = 1`, i.e. `e <
    /// SCALAR_MODULUS_U32`, since the shifted difference `e_window -
    /// modulus_window - borrow_in + 2^EC_WINDOW_BITS` came out `<
    /// 2^EC_WINDOW_BITS`). Setting that top bit — claiming `borrow_out = 0`,
    /// i.e. `e >= modulus` — must be rejected by
    /// `constrain_e_scalar_canonical`'s final-window assertion, confirming
    /// the canonicality gadget is load-bearing in the AIR itself and not
    /// just in the host-side witness generator.
    #[test]
    fn constraints_reject_e_lt_modulus_borrow_flipped_at_last_window() {
        use p3_air::check_all_constraints;

        let batch = deterministic_batch(3).unwrap();
        let trace = generate_trace(&batch).unwrap();
        check_constraints(&trace);

        let idx = |row: usize, col: usize| row * TRACE_WIDTH + col;
        let sig0_last_window = ROWS_PER_SIGNATURE - 1; // 203
        let top_bit_col = scalar_lt_modulus_top_bit_col(COL_EC_E_LT_MODULUS_DIFF_BITS);

        assert_eq!(
            trace.trace.values[idx(sig0_last_window, top_bit_col)],
            Goldilocks::ZERO,
            "honest trace's final EC window must show borrow-out = 1 (e < modulus), i.e. top diff bit clear"
        );

        let mut tampered = trace.clone();
        tampered.trace.values[idx(sig0_last_window, top_bit_col)] = Goldilocks::ONE;
        let report =
            check_all_constraints(&tampered.air, &tampered.trace, &tampered.public_values, None);
        assert!(!report.is_ok());
        assert!(report.failures.iter().any(|f| f.row == sig0_last_window));
    }
}
