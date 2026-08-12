/// AIR for ECgFp5 Schnorr signature batches.
///
/// This AIR constrains the public digest to the Poseidon2 transcript over
/// `(msg_digest, sig, pk)` blocks, the Lighter Poseidon2 challenge hash for a
/// supplied reconstructed `R` witness, native-field reduction of that challenge
/// to the public signature scalar `e`, public-key decoding, and the affine
/// Schnorr equation `R = s*G + e*pk`.
#[derive(Clone, Debug)]
pub struct SignatureBatchAir {
    rows: usize,
}

impl SignatureBatchAir {
    pub fn new(rows: usize) -> Self {
        Self { rows }
    }
}

impl Default for SignatureBatchAir {
    fn default() -> Self {
        Self { rows: 1 }
    }
}

impl<F> BaseAir<F> for SignatureBatchAir
where
    F: PrimeCharacteristicRing + Send + Sync,
{
    fn width(&self) -> usize {
        TRACE_WIDTH
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(preprocessed_trace_for_rows(self.rows))
    }

    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_WIDTH
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn num_public_values(&self) -> usize {
        PUBLIC_VALUES
    }

    // Declares the maximum total degree (in trace-column variables) of any
    // single constraint polynomial below, which determines how wide the
    // quotient polynomial split must be for the FRI backend. If a future
    // edit to `eval`/`constraints.rs`/`selectors.rs` introduces a
    // higher-degree term than this declares, the prover and verifier would
    // silently disagree on the AIR's true degree — this is a hard upper
    // bound that must be re-checked (not just assumed) whenever a
    // constraint's polynomial structure changes.
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(9)
    }
}

impl<AB> Air<AB> for SignatureBatchAir
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        let public_count = builder.public_values()[PUBLIC_COUNT];
        let public_digest = core::array::from_fn::<_, PUBLIC_DIGEST_WIDTH, _>(|lane| {
            builder.public_values()[PUBLIC_DIGEST_START + lane]
        });
        let (round_selectors, bit_selectors, chunk_selectors, s_addend_x, s_addend_u, modulus_window) = {
            let preprocessed = builder.preprocessed();
            let local = preprocessed.current_slice();
            (
                round_selectors_from_preprocessed::<AB>(local),
                ec_bit_selectors_from_preprocessed::<AB>(local),
                ec_chunk_selectors_from_preprocessed::<AB>(local),
                core::array::from_fn(|bit| {
                    fp5_from_row::<AB>(local, preprocessed_s_addend_x_col(bit))
                }),
                core::array::from_fn(|bit| {
                    fp5_from_row::<AB>(local, preprocessed_s_addend_u_col(bit))
                }),
                Into::<AB::Expr>::into(local[PREPROCESSED_COL_MODULUS_WINDOW]),
            )
        };
        let public_count_expr: AB::Expr = public_count.into();

        // Row-state flags and the derived phase/round indicators reused
        // throughout the rest of `eval`; see `air/mod.rs` for what each
        // means and `selectors.rs` for how the indicators are built.
        let enabled: AB::Expr = local[COL_ENABLED].into();
        let hash_active: AB::Expr = local[COL_HASH_ACTIVE].into();
        let ec_active = enabled.dup() - hash_active.dup();
        let phase: AB::Expr = local[COL_PHASE].into();
        let challenge_first: AB::Expr = local[COL_CHALLENGE_FIRST].into();
        let challenge_second: AB::Expr = local[COL_CHALLENGE_SECOND].into();
        let one = AB::Expr::ONE;
        let disabled = one.dup() - enabled.dup();
        let hash_inactive = one.dup() - hash_active.dup();
        let challenge_active = challenge_first.dup() + challenge_second.dup();
        let challenge_inactive = one.dup() - challenge_active.dup();
        let phase_two = hash_active.dup() * phase_two_indicator::<AB>(phase.dup());
        let phase_three = hash_active.dup() * last_phase_indicator::<AB>(phase.dup());
        let ec_final: AB::Expr = local[COL_EC_FINAL].into();
        let round_zero = round_selectors[0].dup();
        let round_last = round_selectors[POSEIDON_ROUND_STATE_COUNT - 1].dup();
        let signature_last = phase_three.dup() * round_last.dup();
        let challenge_first_round_zero = challenge_first.dup() * round_zero.dup();
        let challenge_second_round_zero = challenge_second.dup() * round_zero.dup();
        let phase_two_round_zero = phase_two.dup() * round_zero.dup();
        let phase_three_round_zero = phase_three.dup() * round_zero.dup();

        // Boolean/range constraints on the flags above, plus the rules tying
        // `phase`/`challenge_first`/`challenge_second` together: `phase` is
        // only meaningful (nonzero) on active hashing rows, `ec_active` rows
        // never have a phase, and `challenge_first`/`challenge_second` are
        // mutually exclusive and forced to track phases 0/1 exactly.
        builder.assert_bool(local[COL_ENABLED]);
        builder.assert_bool(local[COL_HASH_ACTIVE]);
        builder.assert_bool(local[COL_EC_FINAL]);
        builder.assert_bool(local[COL_CHALLENGE_FIRST]);
        builder.assert_bool(local[COL_CHALLENGE_SECOND]);
        builder.assert_zero(hash_active.dup() * (hash_active.dup() - enabled.dup()));
        builder.assert_zero(ec_final.dup() * (one.dup() - ec_active.dup()));
        // Bind the witness-controlled `hash_active`/`ec_active` row-type
        // flags to the fixed preprocessed schedule: a row can only claim to
        // be a hashing row if the preprocessed round-selector table is
        // actually hot there (sum to 1, never 0 on an EC-pattern row), and
        // symmetrically for EC rows against the bit-selector table. Without
        // this, `ec_active` and the preprocessed selectors it's indexed by
        // (`bit_selectors`/`chunk_selectors`/`round_selectors`) are
        // otherwise never tied together, letting the EC accumulator spill
        // into rows the fixed schedule reserves for another signature's
        // Poseidon hashing, where the preprocessed EC selectors are all
        // zero and its `e`-bits go unrange-checked against `sig_e_chunks`.
        let round_selector_sum = round_selectors
            .iter()
            .fold(AB::Expr::ZERO, |acc, s| acc + s.dup());
        let bit_selector_sum = bit_selectors
            .iter()
            .fold(AB::Expr::ZERO, |acc, s| acc + s.dup());
        builder.assert_zero(hash_active.dup() * (round_selector_sum - one.dup()));
        builder.assert_zero(ec_active.dup() * (bit_selector_sum - one.dup()));
        builder.assert_zero(disabled.dup() * phase.dup());
        builder.assert_zero(ec_active.dup() * phase.dup());
        builder.assert_zero(challenge_first.dup() * challenge_second.dup());
        builder.assert_zero(
            challenge_first.dup() - hash_active.dup() * phase_zero_indicator::<AB>(phase.dup()),
        );
        builder.assert_zero(
            challenge_second.dup() - hash_active.dup() * phase_one_indicator::<AB>(phase.dup()),
        );
        builder.assert_zero(
            hash_active.dup()
                * phase.dup()
                * (phase.dup() - one.dup())
                * (phase.dup() - AB::Expr::TWO)
                * (phase.dup() - AB::Expr::from_u64(3)),
        );

        // First-row boundary conditions: either the batch is empty
        // (`public_count == 0`, trivially satisfied with `enabled = 0`
        // throughout) or row 0 starts the first signature's hashing phase
        // with the correct initial Poseidon2 state and block countdown.
        {
            let mut first = builder.when_first_row();
            first.assert_eq(
                local[COL_REMAINING_BLOCKS],
                public_count_expr.dup() * AB::Expr::from_u64(POSEIDON_BLOCKS_PER_INSTANCE as u64),
            );
            first.assert_zero(public_count_expr.dup() * (local[COL_ENABLED] - AB::Expr::ONE));
            first.assert_zero(public_count_expr.dup() * (local[COL_HASH_ACTIVE] - AB::Expr::ONE));
            first.assert_eq(local[COL_ENABLED], local[COL_HASH_ACTIVE]);
            first.assert_zero(local[COL_PHASE]);
            for lane in 0..POSEIDON_WIDTH {
                let expected = initial_state_lane::<AB>(lane, public_count);
                first.assert_eq(local[COL_STATE_BEFORE + lane], expected);
            }
        }

        // Public-transcript Poseidon2: on hash-active rows, the next state
        // must be the selected round's output (`selected_public_poseidon_round`
        // picks the round matching `round_selectors`); on disabled/EC rows,
        // the state must simply not change.
        let public_state_before: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|i| local[COL_STATE_BEFORE + i].into());
        let public_block: [AB::Expr; POSEIDON_RATE] =
            core::array::from_fn(|i| local[COL_BLOCK + i].into());
        let expected_public_after = selected_public_poseidon_round::<AB>(
            &round_selectors,
            &public_state_before,
            &public_block,
        );
        for lane in 0..POSEIDON_WIDTH {
            let after: AB::Expr = local[COL_STATE_AFTER + lane].into();
            builder.assert_zero(
                hash_active.dup() * (after.dup() - expected_public_after[lane].dup())
                    + hash_inactive.dup() * (after - public_state_before[lane].dup()),
            );
        }

        // Block lanes are forced to 0 off the hashing rows, and the unused
        // tail lanes of the instance's final (possibly partial) block must
        // be 0 too — both prevent a malicious prover from sneaking extra
        // nonzero values into the absorption that the witness-filling code
        // wouldn't have put there.
        for lane in 0..POSEIDON_RATE {
            let block: AB::Expr = local[COL_BLOCK + lane].into();
            builder.assert_zero(hash_inactive.dup() * block.dup());
            if lane >= LAST_BLOCK_USED_LANES {
                builder.assert_zero(signature_last.dup() * block);
            }
        }

        // Lighter-challenge Poseidon2, run in lockstep with the
        // public-transcript hash above (same round selectors): advances per
        // `selected_lighter_poseidon_round` while `challenge_active`, frozen
        // (and required zero on round 0) while inactive, and squeezed into
        // `COL_CHALLENGE_DIGEST` on the final round of phase 1
        // (`challenge_digest_active`).
        let challenge_state_before: [AB::Expr; LIGHTER_POSEIDON_WIDTH] =
            core::array::from_fn(|i| local[COL_CHALLENGE_STATE_BEFORE + i].into());
        let challenge_block: [AB::Expr; LIGHTER_POSEIDON_RATE] =
            core::array::from_fn(|i| local[COL_CHALLENGE_BLOCK + i].into());
        let expected_challenge_after = selected_lighter_poseidon_round::<AB>(
            &round_selectors,
            &challenge_state_before,
            &challenge_block,
            challenge_first.dup(),
            challenge_second.dup(),
        );
        let challenge_digest_active = challenge_second.dup() * round_last.dup();
        for lane in 0..LIGHTER_POSEIDON_WIDTH {
            let after: AB::Expr = local[COL_CHALLENGE_STATE_AFTER + lane].into();
            builder.assert_zero(
                challenge_active.dup() * (after.dup() - expected_challenge_after[lane].dup())
                    + challenge_inactive.dup() * after.dup(),
            );
            builder.assert_zero(
                challenge_first.dup() * round_zero.dup() * challenge_state_before[lane].dup(),
            );
            builder.assert_zero(challenge_inactive.dup() * challenge_state_before[lane].dup());
        }
        for lane in 0..LIGHTER_POSEIDON_DIGEST_WIDTH {
            let digest: AB::Expr = local[COL_CHALLENGE_DIGEST + lane].into();
            let after: AB::Expr = local[COL_CHALLENGE_STATE_AFTER + lane].into();
            builder.assert_zero(challenge_digest_active.dup() * (digest.dup() - after));
            builder.assert_zero((one.dup() - challenge_digest_active.dup()) * digest);
        }
        for lane in 0..LIGHTER_POSEIDON_RATE {
            let block: AB::Expr = local[COL_CHALLENGE_BLOCK + lane].into();
            builder.assert_zero(challenge_inactive.dup() * block.dup());
            if (5..8).contains(&lane) {
                // Lanes 5..8 of phase 0's block are `msg_digest`'s tail
                // (lanes 0..5 hold the encoded `reconstructed_r`), copied
                // straight from the public-transcript block being absorbed
                // on the same row.
                let public_msg: AB::Expr = local[COL_BLOCK + lane - 5].into();
                builder.assert_zero(challenge_first_round_zero.dup() * (block.dup() - public_msg));
            }
            if lane < 2 {
                // The last 2 elements of `msg_digest` didn't fit in phase
                // 0's block; they're carried forward in `COL_CHALLENGE_MSG_TAIL`
                // (written at the end of phase 0, see the transition section
                // below) and absorbed here at the start of phase 1.
                let carried_msg: AB::Expr = local[COL_CHALLENGE_MSG_TAIL + lane].into();
                builder
                    .assert_zero(challenge_second_round_zero.dup() * (block.dup() - carried_msg));
            } else {
                builder.assert_zero(challenge_second_round_zero.dup() * block.dup());
            }
        }
        for lane in 0..2 {
            let carried_msg: AB::Expr = local[COL_CHALLENGE_MSG_TAIL + lane].into();
            builder.assert_zero((one.dup() - challenge_second.dup()) * carried_msg);
        }

        // Decode the witnessed `reconstructed_r` (absorbed as phase 0's
        // block) into affine coordinates; see `constraints.rs`'s
        // `constrain_encoded_point_decode`.
        let r_encoded = core::array::from_fn(|lane| local[COL_CHALLENGE_BLOCK + lane].into());
        let r_inverse = core::array::from_fn(|lane| local[COL_R_DECODE_INVERSE + lane].into());
        let r_sqrt_delta =
            core::array::from_fn(|lane| local[COL_R_DECODE_SQRT_DELTA + lane].into());
        let r_affine_x = core::array::from_fn(|lane| local[COL_R_AFFINE_X + lane].into());
        constrain_encoded_point_decode::<AB>(
            builder,
            challenge_first_round_zero.dup(),
            &r_encoded,
            &r_inverse,
            &r_sqrt_delta,
            &r_affine_x,
        );

        // Decode the public key, read off the signature's final
        // public-transcript block (`signature_last`). `SchnorrInstance::to_goldilocks_fields`
        // lays out `msg_digest(5) || signature_u32_chunks(20) || public_key(5)`
        // = 30 elements, split into 4 rate-8 blocks; `public_key` starts at
        // element 25, i.e. lane 1 of the 4th block (`COL_BLOCK + 1`), right
        // after that block's lane 0 (the signature's last `u32` chunk).
        let pk_encoded = core::array::from_fn(|lane| local[COL_BLOCK + 1 + lane].into());
        let pk_inverse = core::array::from_fn(|lane| local[COL_PK_DECODE_INVERSE + lane].into());
        let pk_sqrt_delta =
            core::array::from_fn(|lane| local[COL_PK_DECODE_SQRT_DELTA + lane].into());
        let pk_other_root_sqrt =
            core::array::from_fn(|lane| local[COL_PK_DECODE_OTHER_ROOT_SQRT + lane].into());
        let pk_affine_x = core::array::from_fn(|lane| local[COL_PK_AFFINE_X + lane].into());
        constrain_encoded_point_decode::<AB>(
            builder,
            signature_last.dup(),
            &pk_encoded,
            &pk_inverse,
            &pk_sqrt_delta,
            &pk_affine_x,
        );
        constrain_decoded_public_key_root::<AB>(
            builder,
            signature_last.dup(),
            &pk_encoded,
            &pk_affine_x,
            &pk_other_root_sqrt,
        );

        // Bind the persistent `s`/`e` scalar-chunk columns from the hash
        // absorption rows (see `constraints.rs`'s `constrain_signature_chunks`),
        // and read out the reduction's range-checked digest chunks for the
        // EC rows' canonicality check.
        let sig_s_chunks = core::array::from_fn(|limb| local[COL_SIG_S_CHUNKS + limb].into());
        let sig_e_chunks = core::array::from_fn(|limb| local[COL_SIG_E_CHUNKS + limb].into());
        let reduction_digest_u32 =
            core::array::from_fn(|limb| local[COL_REDUCTION_DIGEST_U32 + limb].into());
        constrain_signature_chunks::<AB>(
            builder,
            challenge_first_round_zero.dup(),
            challenge_second_round_zero.dup(),
            phase_two_round_zero.dup(),
            phase_three_round_zero.dup(),
            local,
            &sig_s_chunks,
            &sig_e_chunks,
        );

        // Seed `target_r` (the value the EC accumulator must reach) from
        // `R`'s decoded affine coordinates at phase 0 — this is the value
        // `constrain_ec_row` checks the windowed `s*G + e*pk` computation
        // against on the signature's final EC row.
        let target_r_x = core::array::from_fn(|lane| local[COL_EC_TARGET_R_X + lane].into());
        let target_r_u = core::array::from_fn(|lane| local[COL_EC_TARGET_R_U + lane].into());
        for lane in 0..FP5_LIMBS {
            builder.assert_zero(
                challenge_first_round_zero.dup()
                    * (target_r_x[lane].dup() - r_affine_x[lane].dup()),
            );
            builder.assert_zero(
                challenge_first_round_zero.dup() * (target_r_u[lane].dup() - r_inverse[lane].dup()),
            );
        }

        constrain_ec_row::<AB>(
            builder,
            ec_active.dup(),
            ec_final.dup(),
            local,
            &s_addend_x,
            &s_addend_u,
            &bit_selectors,
            &chunk_selectors,
            &reduction_digest_u32,
            &sig_s_chunks,
            &sig_e_chunks,
            &target_r_x,
            &target_r_u,
            modulus_window.dup(),
        );

        // Bind the remaining `e` reduction chunks (beyond the first, already
        // bound by `constrain_signature_chunks`) directly from the
        // public-transcript block on the phase-2/3 rows.
        for lane in 0..POSEIDON_RATE {
            let e_chunk: AB::Expr = local[COL_REDUCTION_E_CHUNKS + 1 + lane].into();
            let block: AB::Expr = local[COL_BLOCK + lane].into();
            builder.assert_zero(phase_two_round_zero.dup() * (e_chunk - block));
        }
        let last_e_chunk: AB::Expr = local[COL_REDUCTION_E_CHUNKS + SCALAR_U32_LIMBS - 1].into();
        let last_public_e_chunk: AB::Expr = local[COL_BLOCK].into();
        builder.assert_zero(phase_three_round_zero.dup() * (last_e_chunk - last_public_e_chunk));

        // Modular reduction of the challenge digest, checked once per
        // signature on its last hashing row: the quotient is one of {0,1,2}
        // (see `preprocessed.rs::scalar_reduction_witness` for why 3 values
        // suffice), each digest limb decomposes into its low/high 32-bit
        // chunks, and the carry chain has no carry-in/out at the boundaries.
        let q: AB::Expr = local[COL_REDUCTION_Q].into();
        builder.assert_zero(
            signature_last.dup() * q.dup() * (q.dup() - AB::Expr::ONE) * (q.dup() - AB::Expr::TWO),
        );
        for lane in 0..LIGHTER_POSEIDON_DIGEST_WIDTH {
            let digest: AB::Expr = local[COL_REDUCTION_DIGEST + lane].into();
            let lo: AB::Expr = local[COL_REDUCTION_DIGEST_U32 + 2 * lane].into();
            let hi: AB::Expr = local[COL_REDUCTION_DIGEST_U32 + 2 * lane + 1].into();
            builder.assert_zero(
                signature_last.dup() * (digest - lo - AB::Expr::from_u64(U32_BASE) * hi),
            );
        }
        let first_carry: AB::Expr = local[COL_REDUCTION_CARRIES].into();
        let final_carry: AB::Expr = local[COL_REDUCTION_CARRIES + SCALAR_U32_LIMBS].into();
        builder.assert_zero(signature_last.dup() * first_carry);
        builder.assert_zero(signature_last.dup() * final_carry);
        for carry in 0..REDUCTION_CARRY_COUNT {
            let carry: AB::Expr = local[COL_REDUCTION_CARRIES + carry].into();
            builder.assert_zero(
                signature_last.dup()
                    * carry.dup()
                    * (carry.dup() - AB::Expr::ONE)
                    * (carry.dup() - AB::Expr::TWO)
                    * (carry - AB::Expr::from_u64(3)),
            );
        }
        // Per-limb long-division identity: `digest_limb + 2^32*carry_out ==
        // e_limb + q*modulus_limb + carry_in`, i.e. the claimed scalar `e`
        // plus `q` copies of the modulus (with carries) reconstructs the
        // actual challenge digest. Together with the carry range-checks and
        // boundary conditions above, this fully pins down `q` and proves
        // `digest === e_chunks (mod scalar_order)`.
        for limb in 0..SCALAR_U32_LIMBS {
            let digest_limb: AB::Expr = local[COL_REDUCTION_DIGEST_U32 + limb].into();
            let e_limb: AB::Expr = local[COL_REDUCTION_E_CHUNKS + limb].into();
            let carry_in: AB::Expr = local[COL_REDUCTION_CARRIES + limb].into();
            let carry_out: AB::Expr = local[COL_REDUCTION_CARRIES + limb + 1].into();
            let modulus_limb = AB::Expr::from_u64(SCALAR_MODULUS_U32[limb]);
            builder.assert_zero(
                signature_last.dup()
                    * (digest_limb + AB::Expr::from_u64(U32_BASE) * carry_out
                        - e_limb
                        - q.dup() * modulus_limb
                        - carry_in),
            );
        }

        // Row-to-row transitions: advance the hashing state machine (phase,
        // round, block countdown, sponge carries) and reset the EC
        // accumulator's adjacent bookkeeping at signature boundaries. See
        // `air/mod.rs` for the phase semantics and `constrain_ec_transition`
        // (in `constraints.rs`, called at the bottom of this block) for the
        // EC accumulator's own row-to-row wiring.
        {
            let mut transition = builder.when_transition();
            transition.assert_eq(
                next[COL_REMAINING_BLOCKS],
                local[COL_REMAINING_BLOCKS] - hash_active.dup() * round_last.dup(),
            );
            let next_enabled: AB::Expr = next[COL_ENABLED].into();
            transition.assert_zero(disabled.dup() * next_enabled.dup());

            let next_phase: AB::Expr = next[COL_PHASE].into();
            let next_hash_active: AB::Expr = next[COL_HASH_ACTIVE].into();
            let hash_round_nonlast = hash_active.dup() * (one.dup() - round_last.dup());
            let hash_block_last_nonfinal_phase =
                hash_active.dup() * round_last.dup() * (one.dup() - phase_three.dup());
            transition
                .assert_zero(hash_round_nonlast.dup() * (next_hash_active.dup() - AB::Expr::ONE));
            transition.assert_zero(hash_round_nonlast.dup() * (next_phase.dup() - phase.dup()));
            transition.assert_zero(
                hash_block_last_nonfinal_phase.dup() * (next_hash_active.dup() - AB::Expr::ONE),
            );
            transition.assert_zero(
                hash_block_last_nonfinal_phase.dup()
                    * (next_phase.dup() - phase.dup() - AB::Expr::ONE),
            );
            transition.assert_zero((next_enabled.dup() - next_hash_active.dup()) * next_phase.dup());
            transition.assert_zero(signature_last.dup() * next_enabled.dup() * next_hash_active);
            transition.assert_zero(
                ec_final.dup() * next_enabled.dup() * (next[COL_HASH_ACTIVE] - AB::Expr::ONE),
            );
            transition.assert_zero(ec_final.dup() * next_enabled.dup() * next_phase.dup());
            for lane in 0..POSEIDON_WIDTH {
                transition.assert_eq(next[COL_STATE_BEFORE + lane], local[COL_STATE_AFTER + lane]);
            }
            for lane in 0..POSEIDON_RATE {
                transition.assert_zero(
                    hash_round_nonlast.dup() * (next[COL_BLOCK + lane] - local[COL_BLOCK + lane]),
                );
            }

            let challenge_carries =
                challenge_active.dup() * (one.dup() - challenge_digest_active.dup());
            for lane in 0..LIGHTER_POSEIDON_WIDTH {
                transition.assert_zero(
                    challenge_carries.dup()
                        * (next[COL_CHALLENGE_STATE_BEFORE + lane]
                            - local[COL_CHALLENGE_STATE_AFTER + lane]),
                );
            }
            let challenge_first_final = challenge_first.dup() * round_last.dup();
            transition.assert_zero(
                challenge_first_final.dup() * (next[COL_CHALLENGE_MSG_TAIL] - local[COL_BLOCK + 3]),
            );
            transition.assert_zero(
                challenge_first_final * (next[COL_CHALLENGE_MSG_TAIL + 1] - local[COL_BLOCK + 4]),
            );
            let challenge_second_nonfinal = challenge_second.dup() * (one.dup() - round_last.dup());
            for lane in 0..2 {
                transition.assert_zero(
                    challenge_second_nonfinal.dup()
                        * (next[COL_CHALLENGE_MSG_TAIL + lane]
                            - local[COL_CHALLENGE_MSG_TAIL + lane]),
                );
            }
            let challenge_round_nonfinal = challenge_active.dup() * (one.dup() - round_last.dup());
            for lane in 0..LIGHTER_POSEIDON_RATE {
                transition.assert_zero(
                    challenge_round_nonfinal.dup()
                        * (next[COL_CHALLENGE_BLOCK + lane] - local[COL_CHALLENGE_BLOCK + lane]),
                );
            }

            let phase_three_nonfinal = phase_three.dup() * (one.dup() - round_last.dup());
            let reduction_digest_carries = phase_two.dup() + phase_three_nonfinal.dup();
            for lane in 0..LIGHTER_POSEIDON_DIGEST_WIDTH {
                transition.assert_zero(
                    challenge_digest_active.dup()
                        * (next[COL_REDUCTION_DIGEST + lane] - local[COL_CHALLENGE_DIGEST + lane]),
                );
                transition.assert_zero(
                    reduction_digest_carries.dup()
                        * (next[COL_REDUCTION_DIGEST + lane] - local[COL_REDUCTION_DIGEST + lane]),
                );
            }
            transition.assert_zero(
                challenge_digest_active.dup()
                    * (next[COL_REDUCTION_E_CHUNKS] - local[COL_BLOCK + POSEIDON_RATE - 1]),
            );
            let reduction_low_chunks_carry = phase_two.dup() + phase_three_nonfinal.dup();
            for limb in 0..SCALAR_U32_LIMBS - 1 {
                transition.assert_zero(
                    reduction_low_chunks_carry.dup()
                        * (next[COL_REDUCTION_E_CHUNKS + limb]
                            - local[COL_REDUCTION_E_CHUNKS + limb]),
                );
            }
            transition.assert_zero(
                phase_three_nonfinal
                    * (next[COL_REDUCTION_E_CHUNKS + SCALAR_U32_LIMBS - 1]
                        - local[COL_REDUCTION_E_CHUNKS + SCALAR_U32_LIMBS - 1]),
            );

            let signature_continues = enabled.dup() * (one.dup() - ec_final.dup());
            let digest_u32_carries =
                signature_last.dup() + ec_active.dup() * (one.dup() - ec_final.dup());
            for limb in 0..SCALAR_U32_LIMBS {
                transition.assert_zero(
                    digest_u32_carries.dup()
                        * (next[COL_REDUCTION_DIGEST_U32 + limb]
                            - local[COL_REDUCTION_DIGEST_U32 + limb]),
                );
            }
            for limb in 0..SCALAR_U32_LIMBS {
                transition.assert_zero(
                    signature_continues.dup()
                        * (next[COL_SIG_S_CHUNKS + limb] - local[COL_SIG_S_CHUNKS + limb]),
                );
                transition.assert_zero(
                    signature_continues.dup()
                        * (next[COL_SIG_E_CHUNKS + limb] - local[COL_SIG_E_CHUNKS + limb]),
                );
            }
            for lane in 0..FP5_LIMBS {
                transition.assert_zero(
                    signature_continues.dup()
                        * (next[COL_EC_TARGET_R_X + lane] - local[COL_EC_TARGET_R_X + lane]),
                );
                transition.assert_zero(
                    signature_continues.dup()
                        * (next[COL_EC_TARGET_R_U + lane] - local[COL_EC_TARGET_R_U + lane]),
                );
            }

            constrain_ec_transition(
                &mut transition,
                signature_last,
                ec_active,
                ec_final.dup(),
                &bit_selectors,
                &chunk_selectors,
                local,
                next,
            );
        }

        // Last-row boundary conditions: the hashing countdown must have
        // reached zero (every signature's blocks were fully absorbed), any
        // still-`enabled` row must be a completed EC row (no partial
        // signature at the trace's edge), and the final Poseidon2 state
        // must equal the public digest the verifier was given — this is the
        // one constraint that actually ties the whole trace back to the
        // proof's public inputs.
        {
            let mut last = builder.when_last_row();
            last.assert_zero(local[COL_REMAINING_BLOCKS]);
            last.assert_zero(enabled * (AB::Expr::ONE - ec_final));
            for lane in 0..PUBLIC_DIGEST_WIDTH {
                last.assert_eq(local[COL_STATE_AFTER + lane], public_digest[lane]);
            }
        }
    }
}
