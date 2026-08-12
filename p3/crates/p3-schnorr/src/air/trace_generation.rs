// Host-side witness computation: builds the main trace that `eval.rs`'s
// constraints check. Every value written here must satisfy the
// corresponding constraint exactly — this file has no independent
// correctness criterion of its own; it exists only to make the trace
// `eval.rs` accepts. When auditing, read a column's constraint in
// `eval.rs`/`constraints.rs` first, then find where this file fills it, to
// check the two agree (rather than reading this file in isolation).
//
// Accumulator pattern: `acc`/`s_chunk_acc`/`e_chunk_acc`/`digest_chunk_acc`/
// `digest_hi_prefix_acc` are mutable locals threaded across loop iterations,
// mirroring the *_ACC trace columns' row-to-row carry constraints
// (`constrain_ec_transition`) — each is reset at exactly the row where the
// corresponding constraint expects a reset (chunk/signature boundaries).

/// Witness values for one EC row (one 4-bit window): the window's bit
/// decomposition of `s`/`e`, the chunk accumulators before this window, this
/// window's table of `e*pk` addend multiples (with their doubling witnesses),
/// the running point accumulator before/after each of the
/// [`EC_WINDOW_STEPS`] double-and-add steps, and the addition witnesses for
/// each step. `Option<AffinePoint>` represents "the accumulator might be the
/// identity" — `None` here means identity, matching the `*_IS_IDENTITY`
/// trace columns.
#[derive(Clone, Debug)]
struct EcRowWitness {
    s_bits: [Goldilocks; EC_WINDOW_BITS],
    e_bits: [Goldilocks; EC_WINDOW_BITS],
    s_chunk_acc: Goldilocks,
    e_chunk_acc: Goldilocks,
    e_addends: [AffinePoint; EC_WINDOW_BITS],
    acc: Option<AffinePoint>,
    steps: [Option<AffinePoint>; EC_WINDOW_STEPS],
    e_doubles: [Option<(AffinePoint, AffineDoubleWitness)>; EC_WINDOW_BITS],
    adds: [Option<AffineAddWitness>; EC_WINDOW_STEPS],
    /// The borrow *into* this window (`0` for the signature's first EC
    /// window) and the shifted difference `(scalar_window - modulus_window -
    /// borrow_in) + 2^EC_WINDOW_BITS`'s own bit decomposition, for `s` and
    /// `e` respectively — see [`scalar_lt_modulus_borrow_step`] and
    /// [`crate::air::constraints::constrain_scalar_lt_modulus`].
    s_lt_modulus_borrow_in: Goldilocks,
    s_lt_modulus_diff_bits: [Goldilocks; EC_SCALAR_LT_MODULUS_DIFF_BITS],
    e_lt_modulus_borrow_in: Goldilocks,
    e_lt_modulus_diff_bits: [Goldilocks; EC_SCALAR_LT_MODULUS_DIFF_BITS],
}

/// One step of the per-window borrow-subtraction chain proving `scalar <
/// SCALAR_MODULUS_U32` (see [`crate::air::preprocessed::scalar_lt_modulus_window_borrows`]
/// for the host-side-only mirror used before trace generation, and
/// [`crate::air::constraints::constrain_scalar_lt_modulus`] for the
/// in-circuit constraint this witnesses). Returns this window's
/// `EC_SCALAR_LT_MODULUS_DIFF_BITS`-bit shifted-difference decomposition and
/// the resulting borrow-out (to become the next window's `borrow_in`):
/// subtraction is `scalar - modulus` (not `modulus - scalar`), so borrowing
/// past the most significant window means `scalar < modulus`; the
/// difference is shifted into `[0, 2^(EC_WINDOW_BITS+1))` so it can be
/// bit-decomposed regardless of whether the unshifted difference was
/// negative (see `EC_SCALAR_LT_MODULUS_DIFF_BITS`'s doc comment in
/// `layout.rs`), and the borrow-out is 1 iff the shifted difference's top
/// bit is 0 (i.e. iff the unshifted difference was negative).
fn scalar_lt_modulus_borrow_step(
    scalar_window: u64,
    modulus_window: u64,
    borrow_in: u64,
) -> ([Goldilocks; EC_SCALAR_LT_MODULUS_DIFF_BITS], u64) {
    let diff_shifted = (scalar_window as i64 - modulus_window as i64 - borrow_in as i64
        + (1i64 << EC_WINDOW_BITS)) as u64;
    let diff_bits = core::array::from_fn(|bit| Goldilocks::from_u64((diff_shifted >> bit) & 1));
    let borrow_out = 1 - ((diff_shifted >> EC_WINDOW_BITS) & 1);
    (diff_bits, borrow_out)
}

/// Computes every [`EcRowWitness`] for one signature's `EC_SCALAR_WINDOWS`
/// EC rows: walks `s`/`e` 4 bits at a time, maintaining the running `s*G`
/// window table (doubling the generator), the `e*pk` window table (doubling
/// the decoded public key), and the double-and-add accumulator across all
/// windows. Fails with [`crate::Error::EcArithmeticFailed`] if any
/// add/double hits an exceptional case (denominator zero) — this would mean
/// the public key or an intermediate value is a point of order 2, which
/// shouldn't occur for a valid signature on this curve.
fn ec_row_witnesses(
    s_chunks: &[Goldilocks; SCALAR_U32_LIMBS],
    e_chunks: &[Goldilocks; SCALAR_U32_LIMBS],
    public_key_decode: &EncodedPointDecodeHint,
) -> Result<Vec<EcRowWitness>> {
    let mut rows = Vec::with_capacity(EC_SCALAR_WINDOWS);
    let mut s_addend = AffinePoint::generator();
    let mut e_addend = AffinePoint {
        x: public_key_decode.affine_x,
        u: public_key_decode.inverse,
    };
    let mut acc = None;
    let mut s_chunk_acc = 0u64;
    let mut e_chunk_acc = 0u64;
    let mut s_lt_modulus_borrow_in = 0u64;
    let mut e_lt_modulus_borrow_in = 0u64;

    for window_index in 0..EC_SCALAR_WINDOWS {
        let bit_start = window_index * EC_WINDOW_BITS;
        let chunk_index = bit_start / REDUCTION_BITS_PER_U32;
        let bit_offset = bit_start % REDUCTION_BITS_PER_U32;
        let s_bits = core::array::from_fn(|bit| {
            Goldilocks::from_u64(
                (s_chunks[chunk_index].as_canonical_u64() >> (bit_offset + bit)) & 1,
            )
        });
        let e_bits = core::array::from_fn(|bit| {
            Goldilocks::from_u64(
                (e_chunks[chunk_index].as_canonical_u64() >> (bit_offset + bit)) & 1,
            )
        });

        let window_mask = (1u64 << EC_WINDOW_BITS) - 1;
        let s_window = (s_chunks[chunk_index].as_canonical_u64() >> bit_offset) & window_mask;
        let e_window = (e_chunks[chunk_index].as_canonical_u64() >> bit_offset) & window_mask;
        let modulus_window = (SCALAR_MODULUS_U32[chunk_index] >> bit_offset) & window_mask;
        let s_lt_modulus_borrow_in_field = Goldilocks::from_u64(s_lt_modulus_borrow_in);
        let e_lt_modulus_borrow_in_field = Goldilocks::from_u64(e_lt_modulus_borrow_in);
        let (s_lt_modulus_diff_bits, s_borrow_out) =
            scalar_lt_modulus_borrow_step(s_window, modulus_window, s_lt_modulus_borrow_in);
        let (e_lt_modulus_diff_bits, e_borrow_out) =
            scalar_lt_modulus_borrow_step(e_window, modulus_window, e_lt_modulus_borrow_in);
        s_lt_modulus_borrow_in = s_borrow_out;
        e_lt_modulus_borrow_in = e_borrow_out;

        let mut e_addends = [e_addend; EC_WINDOW_BITS];
        let mut e_doubles = [None; EC_WINDOW_BITS];
        for bit in 0..EC_WINDOW_BITS {
            e_addends[bit] = e_addend;
            if window_index + 1 < EC_SCALAR_WINDOWS || bit + 1 < EC_WINDOW_BITS {
                let double = e_addend.double().ok_or(crate::Error::EcArithmeticFailed {
                    operation: "e addend double",
                })?;
                e_doubles[bit] = Some(double);
                e_addend = double.0;
            }
        }

        let mut running = acc;
        let mut steps = [None; EC_WINDOW_STEPS];
        let mut adds = [None; EC_WINDOW_STEPS];
        for bit in 0..EC_WINDOW_BITS {
            let s_bit = s_bits[bit].as_canonical_u64() == 1;
            let e_bit = e_bits[bit].as_canonical_u64() == 1;
            let (after_s, s_add) = conditional_add_witness(running, s_addend, s_bit, "s add")?;
            steps[2 * bit] = after_s;
            adds[2 * bit] = s_add;

            let (after_e, e_add) =
                conditional_add_witness(after_s, e_addends[bit], e_bit, "e add")?;
            steps[2 * bit + 1] = after_e;
            adds[2 * bit + 1] = e_add;
            running = after_e;

            if window_index + 1 < EC_SCALAR_WINDOWS || bit + 1 < EC_WINDOW_BITS {
                let (next, _) = s_addend.double().ok_or(crate::Error::EcArithmeticFailed {
                    operation: "s addend double",
                })?;
                s_addend = next;
            }
        }

        rows.push(EcRowWitness {
            s_bits,
            e_bits,
            s_chunk_acc: Goldilocks::from_u64(s_chunk_acc),
            e_chunk_acc: Goldilocks::from_u64(e_chunk_acc),
            e_addends,
            acc,
            steps,
            e_doubles,
            adds,
            s_lt_modulus_borrow_in: s_lt_modulus_borrow_in_field,
            s_lt_modulus_diff_bits,
            e_lt_modulus_borrow_in: e_lt_modulus_borrow_in_field,
            e_lt_modulus_diff_bits,
        });

        for bit in 0..EC_WINDOW_BITS {
            let bit_index = bit_offset + bit;
            s_chunk_acc += s_bits[bit].as_canonical_u64() << bit_index;
            e_chunk_acc += e_bits[bit].as_canonical_u64() << bit_index;
        }
        if bit_offset + EC_WINDOW_BITS == REDUCTION_BITS_PER_U32 {
            s_chunk_acc = 0;
            e_chunk_acc = 0;
        }
        acc = running;
    }

    Ok(rows)
}

/// Host-side mirror of [`constrain_conditional_add`]: adds `rhs` to `lhs`
/// only if `take_rhs`, handling `lhs` being the identity (`None`) without
/// calling [`AffinePoint::add`] on it (which doesn't accept an identity
/// input). Returns `None` for the addition witness whenever no real curve
/// addition happened (not added, or added to/from the identity), matching
/// what the AIR only constrains in the `general` case.
fn conditional_add_witness(
    lhs: Option<AffinePoint>,
    rhs: AffinePoint,
    take_rhs: bool,
    operation: &'static str,
) -> Result<(Option<AffinePoint>, Option<AffineAddWitness>)> {
    match (lhs, take_rhs) {
        (acc, false) => Ok((acc, None)),
        (None, true) => Ok((Some(rhs), None)),
        (Some(acc), true) => {
            let (sum, witness) = acc
                .add(&rhs)
                .ok_or(crate::Error::EcArithmeticFailed { operation })?;
            Ok((Some(sum), Some(witness)))
        }
    }
}

/// Everything precomputed for one signature before laying out its rows: the
/// public-transcript absorption blocks, the signature's scalar chunks, and
/// every EC row's witness.
#[derive(Clone, Debug)]
struct PreparedSignatureTrace {
    blocks: [[Goldilocks; POSEIDON_RATE]; POSEIDON_BLOCKS_PER_INSTANCE],
    s_chunks: [Goldilocks; SCALAR_U32_LIMBS],
    e_chunks: [Goldilocks; SCALAR_U32_LIMBS],
    ec_rows: Vec<EcRowWitness>,
}

fn prepare_signature_trace(
    instance: &crate::types::SchnorrInstance,
    hint: &SignatureChallengeHint,
) -> Result<PreparedSignatureTrace> {
    let fields = instance.to_goldilocks_fields()?;
    let blocks = instance_blocks(&fields);
    let signature_chunks = instance.signature.to_u32_field_elements();
    let s_chunks = core::array::from_fn(|limb| signature_chunks[limb]);
    let e_chunks = core::array::from_fn(|limb| signature_chunks[SCALAR_U32_LIMBS + limb]);
    // `e`'s canonicality is checked by `scalar_reduction_witness`, called
    // from `generate_trace_with_challenge_hints` once the challenge digest
    // is available; `s` has no such digest to reduce against, so this is
    // its only canonicality check.
    require_scalar_lt_modulus(&s_chunks.map(|chunk| chunk.as_canonical_u64()), "s")?;
    let ec_rows = ec_row_witnesses(&s_chunks, &e_chunks, &hint.public_key_decode)?;

    Ok(PreparedSignatureTrace {
        blocks,
        s_chunks,
        e_chunks,
        ec_rows,
    })
}

/// Writes one `Fp5`'s 5 limbs into the row at `offset`, starting at `start_col`.
fn fill_fp5(values: &mut [Goldilocks], offset: usize, start_col: usize, value: &Fp5) {
    values[offset + start_col..offset + start_col + FP5_LIMBS].copy_from_slice(value);
}

/// Writes a known (non-identity) [`AffinePoint`]'s `x`/`u` coordinates.
fn fill_affine_point(
    values: &mut [Goldilocks],
    offset: usize,
    x_col: usize,
    u_col: usize,
    point: &AffinePoint,
) {
    fill_fp5(values, offset, x_col, &point.x);
    fill_fp5(values, offset, u_col, &point.u);
}

/// Writes an [`Option<AffinePoint>`], encoding `None` (the identity) as
/// `(x, u) = (0, 0)` with the identity flag set to 1 — the fixed encoding
/// [`constrain_conditional_add`] expects for identity points.
fn fill_optional_affine_point(
    values: &mut [Goldilocks],
    offset: usize,
    x_col: usize,
    u_col: usize,
    identity_col: usize,
    point: Option<AffinePoint>,
) {
    if let Some(point) = point {
        fill_affine_point(values, offset, x_col, u_col, &point);
        values[offset + identity_col] = Goldilocks::ZERO;
    } else {
        fill_fp5(values, offset, x_col, &fp5_zero());
        fill_fp5(values, offset, u_col, &fp5_zero());
        values[offset + identity_col] = Goldilocks::ONE;
    }
}

/// Writes the denominator-inverse witness for one addition step, if a real
/// addition happened (`None` leaves the columns at their default zero,
/// which is fine since [`constrain_conditional_add`] only reads them when
/// `general` selects the real-addition case).
fn fill_add_witness(
    values: &mut [Goldilocks],
    offset: usize,
    x_inv_col: usize,
    u_inv_col: usize,
    witness: Option<AffineAddWitness>,
) {
    if let Some(witness) = witness {
        fill_fp5(values, offset, x_inv_col, &witness.x_denominator_inverse);
        fill_fp5(values, offset, u_inv_col, &witness.u_denominator_inverse);
    }
}

/// Writes the denominator-inverse witness for one doubling step. `None`
/// occurs only for the very last doubling of the very last window (where no
/// further `e*pk` multiple is needed), matching [`ec_row_witnesses`]'s
/// "skip the last double" guard.
fn fill_double_witness(
    values: &mut [Goldilocks],
    offset: usize,
    x_inv_col: usize,
    u_inv_col: usize,
    witness: Option<AffineDoubleWitness>,
) {
    if let Some(witness) = witness {
        fill_fp5(values, offset, x_inv_col, &witness.x_denominator_inverse);
        fill_fp5(values, offset, u_inv_col, &witness.u_denominator_inverse);
    }
}

/// Writes a complete [`EcRowWitness`] into the trace row at `offset`.
fn fill_ec_witness_row(values: &mut [Goldilocks], offset: usize, row: &EcRowWitness) {
    values[offset + COL_EC_S_BITS..offset + COL_EC_S_BITS + EC_WINDOW_BITS]
        .copy_from_slice(&row.s_bits);
    values[offset + COL_EC_E_BITS..offset + COL_EC_E_BITS + EC_WINDOW_BITS]
        .copy_from_slice(&row.e_bits);
    values[offset + COL_EC_S_CHUNK_ACC] = row.s_chunk_acc;
    values[offset + COL_EC_E_CHUNK_ACC] = row.e_chunk_acc;
    values[offset + COL_EC_S_LT_MODULUS_BORROW_IN] = row.s_lt_modulus_borrow_in;
    values[offset + COL_EC_S_LT_MODULUS_DIFF_BITS
        ..offset + COL_EC_S_LT_MODULUS_DIFF_BITS + EC_SCALAR_LT_MODULUS_DIFF_BITS]
        .copy_from_slice(&row.s_lt_modulus_diff_bits);
    values[offset + COL_EC_E_LT_MODULUS_BORROW_IN] = row.e_lt_modulus_borrow_in;
    values[offset + COL_EC_E_LT_MODULUS_DIFF_BITS
        ..offset + COL_EC_E_LT_MODULUS_DIFF_BITS + EC_SCALAR_LT_MODULUS_DIFF_BITS]
        .copy_from_slice(&row.e_lt_modulus_diff_bits);

    for bit in 0..EC_WINDOW_BITS {
        fill_affine_point(
            values,
            offset,
            ec_e_addend_x_col(bit),
            ec_e_addend_u_col(bit),
            &row.e_addends[bit],
        );
    }
    fill_optional_affine_point(
        values,
        offset,
        COL_EC_ACC_X,
        COL_EC_ACC_U,
        COL_EC_ACC_IS_IDENTITY,
        row.acc,
    );
    for step in 0..EC_WINDOW_STEPS {
        fill_optional_affine_point(
            values,
            offset,
            ec_step_x_col(step),
            ec_step_u_col(step),
            ec_step_is_identity_col(step),
            row.steps[step],
        );
    }

    for bit in 0..EC_WINDOW_BITS {
        fill_double_witness(
            values,
            offset,
            ec_e_double_x_den_inv_col(bit),
            ec_e_double_u_den_inv_col(bit),
            row.e_doubles[bit].map(|(_, witness)| witness),
        );
    }
    for step in 0..EC_WINDOW_STEPS {
        fill_add_witness(
            values,
            offset,
            ec_add_x_den_inv_col(step),
            ec_add_u_den_inv_col(step),
            row.adds[step],
        );
    }
}

/// Generates a full trace for `batch` by deriving challenge hints from the
/// `goldilocks-crypto` reference implementation rather than requiring the
/// caller to supply them. Convenience for tests/benchmarks; production
/// proving should supply real hints (see [`generate_trace_with_challenge_hints`])
/// since a real signer/verifier pipeline won't have `reference` enabled.
#[cfg(feature = "reference")]
pub fn generate_trace(batch: &Batch) -> Result<SignatureBatchTrace> {
    generate_trace_with_capacity(batch, batch.len())
}

/// [`generate_trace`], but sizing the trace for `capacity` signature slots:
/// the trace height is the one a `capacity`-sized batch would use, the
/// `batch.len()` real instances occupy the leading slots, and the remaining
/// slots are disabled rows. The public signature count stays `batch.len()`,
/// so the digest covers exactly the real instances while the proof shape
/// matches the fixed `capacity` height.
#[cfg(feature = "reference")]
pub fn generate_trace_with_capacity(batch: &Batch, capacity: usize) -> Result<SignatureBatchTrace> {
    let hints = batch
        .instances()
        .iter()
        .map(|instance| {
            let witness = crate::schnorr::verification_witness(instance)?;
            SignatureChallengeHint::from_reference(instance, witness.reconstructed_r)
        })
        .collect::<Result<Vec<_>>>()?;
    generate_trace_with_challenge_hints_and_capacity(batch, &hints, capacity)
}

/// Stub present so callers don't need `#[cfg(feature = "reference")]` at
/// every call site; always fails, since deriving a challenge hint requires
/// the `goldilocks-crypto` reference implementation.
#[cfg(not(feature = "reference"))]
pub fn generate_trace(_batch: &Batch) -> Result<SignatureBatchTrace> {
    Err(crate::Error::MissingChallengeHints)
}

/// Stub counterpart of [`generate_trace_with_capacity`] when the `reference`
/// feature is disabled; always fails, like [`generate_trace`].
#[cfg(not(feature = "reference"))]
pub fn generate_trace_with_capacity(
    _batch: &Batch,
    _capacity: usize,
) -> Result<SignatureBatchTrace> {
    Err(crate::Error::MissingChallengeHints)
}

/// Builds the complete main trace and public values for `batch`, given a
/// pre-supplied [`SignatureChallengeHint`] per instance (in batch order).
///
/// Walks every row of the (power-of-two-padded) trace height once,
/// maintaining the same accumulator state the AIR's transition constraints
/// expect: the running public-transcript Poseidon2 state, the
/// remaining-blocks countdown, the Lighter-challenge sponge state and
/// message tail, the modular-reduction digest/chunks, and the EC range-check
/// accumulators. Each section below fills exactly the columns the
/// corresponding part of `eval.rs` constrains for that row.
pub fn generate_trace_with_challenge_hints(
    batch: &Batch,
    challenge_hints: &[SignatureChallengeHint],
) -> Result<SignatureBatchTrace> {
    generate_trace_with_challenge_hints_and_capacity(batch, challenge_hints, batch.len())
}

/// [`generate_trace_with_challenge_hints`], but sizing the trace for
/// `capacity` signature slots (see [`generate_trace_with_capacity`]).
pub fn generate_trace_with_challenge_hints_and_capacity(
    batch: &Batch,
    challenge_hints: &[SignatureChallengeHint],
    capacity: usize,
) -> Result<SignatureBatchTrace> {
    assert!(batch.len() <= MAX_SIGNATURES);
    if capacity > MAX_SIGNATURES {
        return Err(crate::Error::TooManySignatures {
            actual: capacity,
            max: MAX_SIGNATURES,
        });
    }
    if batch.len() > capacity {
        return Err(crate::Error::BatchExceedsCapacity {
            actual: batch.len(),
            capacity,
        });
    }
    if challenge_hints.len() != batch.len() {
        return Err(crate::Error::ChallengeHintCountMismatch {
            actual: challenge_hints.len(),
            expected: batch.len(),
        });
    }

    let prepared = batch
        .instances()
        .iter()
        .zip(challenge_hints)
        .map(|(instance, hint)| prepare_signature_trace(instance, hint))
        .collect::<Result<Vec<_>>>()?;
    let height = trace_height_for_count(capacity);
    let mut values = Goldilocks::zero_vec(height * TRACE_WIDTH);
    let mut state = initial_public_hash_state(batch.len());
    let active_rows = batch.len() * ROWS_PER_SIGNATURE;
    let mut remaining = batch.len() * POSEIDON_BLOCKS_PER_INSTANCE;
    let mut challenge_state = [Goldilocks::ZERO; LIGHTER_POSEIDON_WIDTH];
    let mut challenge_msg_tail = [Goldilocks::ZERO; 2];
    let mut reduction_digest = [Goldilocks::ZERO; LIGHTER_POSEIDON_DIGEST_WIDTH];
    let mut reduction_e_chunks = [Goldilocks::ZERO; SCALAR_U32_LIMBS];
    let mut reduction_digest_u32 = [Goldilocks::ZERO; SCALAR_U32_LIMBS];
    let mut digest_chunk_acc = 0u64;
    let mut digest_hi_prefix_acc = 0u64;

    for row in 0..height {
        let offset = row * TRACE_WIDTH;
        let enabled = usize::from(row < active_rows);
        let signature_index = row / ROWS_PER_SIGNATURE;
        let signature_row = row % ROWS_PER_SIGNATURE;
        let hash_active =
            usize::from(enabled == 1 && signature_row < PUBLIC_HASH_ROWS_PER_SIGNATURE);
        let ec_active =
            usize::from(enabled == 1 && signature_row >= PUBLIC_HASH_ROWS_PER_SIGNATURE);
        let phase = if hash_active == 1 {
            signature_row / POSEIDON_ROUND_STATE_COUNT
        } else {
            0
        };
        let round = if hash_active == 1 {
            signature_row % POSEIDON_ROUND_STATE_COUNT
        } else {
            0
        };
        values[offset + COL_ENABLED] = Goldilocks::from_usize(enabled);
        values[offset + COL_HASH_ACTIVE] = Goldilocks::from_usize(hash_active);
        values[offset + COL_PHASE] = Goldilocks::from_usize(phase);
        values[offset + COL_REMAINING_BLOCKS] = Goldilocks::from_usize(remaining);
        values[offset + COL_STATE_BEFORE..offset + COL_STATE_BEFORE + POSEIDON_WIDTH]
            .copy_from_slice(&state);

        // Public-transcript Poseidon2: absorb this signature's current
        // block and advance the state by one round (mirrors
        // `selected_public_poseidon_round` in eval.rs, but deterministically
        // picking the one real round instead of summing all candidates).
        let mut block = [Goldilocks::ZERO; POSEIDON_RATE];
        if hash_active == 1 {
            let signature = &prepared[signature_index];
            block = signature.blocks[phase];
            state = public_poseidon_round_step(round, &state, &block);
            if round == POSEIDON_ROUND_STATE_COUNT - 1 {
                remaining -= 1;
            }
        }

        if enabled == 1 {
            let signature = &prepared[signature_index];
            values[offset + COL_SIG_S_CHUNKS..offset + COL_SIG_S_CHUNKS + SCALAR_U32_LIMBS]
                .copy_from_slice(&signature.s_chunks);
            values[offset + COL_SIG_E_CHUNKS..offset + COL_SIG_E_CHUNKS + SCALAR_U32_LIMBS]
                .copy_from_slice(&signature.e_chunks);
            let hint = challenge_hints[signature_index];
            values[offset + COL_EC_TARGET_R_X..offset + COL_EC_TARGET_R_X + FP5_LIMBS]
                .copy_from_slice(&hint.reconstructed_r_decode.affine_x);
            values[offset + COL_EC_TARGET_R_U..offset + COL_EC_TARGET_R_U + FP5_LIMBS]
                .copy_from_slice(&hint.reconstructed_r_decode.inverse);
        }

        // Lighter-challenge Poseidon2 and modular reduction, one phase at a
        // time (see `air/mod.rs` for what each phase means). Mirrors the
        // phase-gated constraints in eval.rs; the `match` here picks exactly
        // one phase per row, where eval.rs instead sums over indicator
        // polynomials that vanish for the non-matching phases.
        if hash_active == 1 {
            let hint = challenge_hints[signature_index];
            match phase {
                0 => {
                    values[offset + COL_CHALLENGE_FIRST] = Goldilocks::ONE;
                    values[offset + COL_CHALLENGE_STATE_BEFORE
                        ..offset + COL_CHALLENGE_STATE_BEFORE + LIGHTER_POSEIDON_WIDTH]
                        .copy_from_slice(&challenge_state);
                    let mut challenge_block = [Goldilocks::ZERO; LIGHTER_POSEIDON_RATE];
                    challenge_block[..LIGHTER_POSEIDON_DIGEST_WIDTH]
                        .copy_from_slice(&hint.reconstructed_r);
                    challenge_block[LIGHTER_POSEIDON_DIGEST_WIDTH..].copy_from_slice(&block[..3]);

                    values[offset + COL_CHALLENGE_BLOCK
                        ..offset + COL_CHALLENGE_BLOCK + LIGHTER_POSEIDON_RATE]
                        .copy_from_slice(&challenge_block);
                    values
                        [offset + COL_R_DECODE_INVERSE..offset + COL_R_DECODE_INVERSE + FP5_LIMBS]
                        .copy_from_slice(&hint.reconstructed_r_decode.inverse);
                    values[offset + COL_R_DECODE_SQRT_DELTA
                        ..offset + COL_R_DECODE_SQRT_DELTA + FP5_LIMBS]
                        .copy_from_slice(&hint.reconstructed_r_decode.sqrt_delta);
                    values[offset + COL_R_AFFINE_X..offset + COL_R_AFFINE_X + FP5_LIMBS]
                        .copy_from_slice(&hint.reconstructed_r_decode.affine_x);
                    challenge_state = lighter_poseidon_round_step(
                        round,
                        &challenge_state,
                        &challenge_block,
                        true,
                        false,
                    );
                    values[offset + COL_CHALLENGE_STATE_AFTER
                        ..offset + COL_CHALLENGE_STATE_AFTER + LIGHTER_POSEIDON_WIDTH]
                        .copy_from_slice(&challenge_state);
                    if round == LIGHTER_POSEIDON_ROUND_STATE_COUNT - 1 {
                        challenge_msg_tail.copy_from_slice(&block[3..5]);
                    }
                }
                1 => {
                    values[offset + COL_CHALLENGE_SECOND] = Goldilocks::ONE;
                    values[offset + COL_CHALLENGE_MSG_TAIL..offset + COL_CHALLENGE_MSG_TAIL + 2]
                        .copy_from_slice(&challenge_msg_tail);
                    values[offset + COL_CHALLENGE_STATE_BEFORE
                        ..offset + COL_CHALLENGE_STATE_BEFORE + LIGHTER_POSEIDON_WIDTH]
                        .copy_from_slice(&challenge_state);

                    let mut challenge_block = [Goldilocks::ZERO; LIGHTER_POSEIDON_RATE];
                    challenge_block[..2].copy_from_slice(&challenge_msg_tail);

                    values[offset + COL_CHALLENGE_BLOCK
                        ..offset + COL_CHALLENGE_BLOCK + LIGHTER_POSEIDON_RATE]
                        .copy_from_slice(&challenge_block);
                    challenge_state = lighter_poseidon_round_step(
                        round,
                        &challenge_state,
                        &challenge_block,
                        false,
                        true,
                    );
                    values[offset + COL_CHALLENGE_STATE_AFTER
                        ..offset + COL_CHALLENGE_STATE_AFTER + LIGHTER_POSEIDON_WIDTH]
                        .copy_from_slice(&challenge_state);
                    if round == LIGHTER_POSEIDON_ROUND_STATE_COUNT - 1 {
                        let digest = &challenge_state[..LIGHTER_POSEIDON_DIGEST_WIDTH];
                        values[offset + COL_CHALLENGE_DIGEST
                            ..offset + COL_CHALLENGE_DIGEST + LIGHTER_POSEIDON_DIGEST_WIDTH]
                            .copy_from_slice(digest);
                        reduction_digest.copy_from_slice(digest);
                        reduction_e_chunks = [Goldilocks::ZERO; SCALAR_U32_LIMBS];
                        reduction_e_chunks[0] = block[POSEIDON_RATE - 1];
                        challenge_state = [Goldilocks::ZERO; LIGHTER_POSEIDON_WIDTH];
                        challenge_msg_tail = [Goldilocks::ZERO; 2];
                    }
                }
                2 => {
                    values[offset + COL_REDUCTION_DIGEST
                        ..offset + COL_REDUCTION_DIGEST + LIGHTER_POSEIDON_DIGEST_WIDTH]
                        .copy_from_slice(&reduction_digest);
                    reduction_e_chunks[1..1 + POSEIDON_RATE].copy_from_slice(&block);
                    values[offset + COL_REDUCTION_E_CHUNKS
                        ..offset + COL_REDUCTION_E_CHUNKS + SCALAR_U32_LIMBS]
                        .copy_from_slice(&reduction_e_chunks);
                }
                3 => {
                    reduction_e_chunks[SCALAR_U32_LIMBS - 1] = block[0];
                    values[offset + COL_REDUCTION_DIGEST
                        ..offset + COL_REDUCTION_DIGEST + LIGHTER_POSEIDON_DIGEST_WIDTH]
                        .copy_from_slice(&reduction_digest);
                    values[offset + COL_REDUCTION_E_CHUNKS
                        ..offset + COL_REDUCTION_E_CHUNKS + SCALAR_U32_LIMBS]
                        .copy_from_slice(&reduction_e_chunks);
                    if round == POSEIDON_ROUND_STATE_COUNT - 1 {
                        let reduction =
                            scalar_reduction_witness(reduction_digest, reduction_e_chunks)?;
                        values[offset + COL_REDUCTION_DIGEST_U32
                            ..offset + COL_REDUCTION_DIGEST_U32 + SCALAR_U32_LIMBS]
                            .copy_from_slice(&reduction.digest_u32);
                        values[offset + COL_REDUCTION_Q] = reduction.quotient;
                        values[offset + COL_REDUCTION_CARRIES
                            ..offset + COL_REDUCTION_CARRIES + REDUCTION_CARRY_COUNT]
                            .copy_from_slice(&reduction.carries);
                        values[offset + COL_PK_DECODE_INVERSE
                            ..offset + COL_PK_DECODE_INVERSE + FP5_LIMBS]
                            .copy_from_slice(&hint.public_key_decode.inverse);
                        values[offset + COL_PK_DECODE_SQRT_DELTA
                            ..offset + COL_PK_DECODE_SQRT_DELTA + FP5_LIMBS]
                            .copy_from_slice(&hint.public_key_decode.sqrt_delta);
                        values[offset + COL_PK_DECODE_OTHER_ROOT_SQRT
                            ..offset + COL_PK_DECODE_OTHER_ROOT_SQRT + FP5_LIMBS]
                            .copy_from_slice(&hint.public_key_decode.other_root_sqrt);
                        values[offset + COL_PK_AFFINE_X..offset + COL_PK_AFFINE_X + FP5_LIMBS]
                            .copy_from_slice(&hint.public_key_decode.affine_x);
                        for e_chunk in &reduction_e_chunks {
                            debug_assert!(e_chunk.as_canonical_u64() < U32_BASE);
                        }
                        reduction_digest_u32 = reduction.digest_u32;
                        digest_chunk_acc = 0;
                        digest_hi_prefix_acc = 0;
                        reduction_digest = [Goldilocks::ZERO; LIGHTER_POSEIDON_DIGEST_WIDTH];
                        reduction_e_chunks = [Goldilocks::ZERO; SCALAR_U32_LIMBS];
                    }
                }
                _ => {}
            }
        }

        // EC rows: extract this window's digest bits for the canonical-range
        // check (see `constrain_digest_u32_range_row`), mark the final EC
        // row, and fill in this window's full double-and-add witness.
        if ec_active == 1 {
            let ec_window = signature_row - PUBLIC_HASH_ROWS_PER_SIGNATURE;
            let ec_row = &prepared[signature_index].ec_rows[ec_window];
            let digest_chunk = ec_window / EC_WINDOWS_PER_CHUNK;
            let digest_bit_offset = (ec_window % EC_WINDOWS_PER_CHUNK) * EC_WINDOW_BITS;
            let digest_limb = reduction_digest_u32[digest_chunk].as_canonical_u64();
            let digest_bits = core::array::from_fn::<_, EC_WINDOW_BITS, _>(|bit| {
                (digest_limb >> (digest_bit_offset + bit)) & 1
            });

            values[offset + COL_REDUCTION_DIGEST_U32
                ..offset + COL_REDUCTION_DIGEST_U32 + SCALAR_U32_LIMBS]
                .copy_from_slice(&reduction_digest_u32);
            for bit in 0..EC_WINDOW_BITS {
                values[offset + COL_REDUCTION_DIGEST_BITS + bit] =
                    Goldilocks::from_u64(digest_bits[bit]);
            }
            let window_all_ones = digest_bits.iter().copied().product::<u64>();
            values[offset + COL_REDUCTION_DIGEST_WINDOW_ALL_ONES] =
                Goldilocks::from_u64(window_all_ones);
            values[offset + COL_REDUCTION_DIGEST_CHUNK_ACC] =
                Goldilocks::from_u64(digest_chunk_acc);
            values[offset + COL_REDUCTION_DIGEST_HI_PREFIX_ACC] =
                Goldilocks::from_u64(digest_hi_prefix_acc);

            if ec_window == EC_SCALAR_WINDOWS - 1 {
                values[offset + COL_EC_FINAL] = Goldilocks::ONE;
            }
            fill_ec_witness_row(&mut values, offset, ec_row);

            for (bit, digest_bit) in digest_bits.iter().enumerate() {
                digest_chunk_acc += digest_bit << (digest_bit_offset + bit);
            }
            if digest_chunk % 2 == 1 {
                digest_hi_prefix_acc = if digest_bit_offset == 0 {
                    window_all_ones
                } else {
                    digest_hi_prefix_acc & window_all_ones
                };
            } else {
                digest_hi_prefix_acc = 0;
            }
            if digest_bit_offset + EC_WINDOW_BITS == REDUCTION_BITS_PER_U32 {
                digest_chunk_acc = 0;
                digest_hi_prefix_acc = 0;
            }
        }

        values[offset + COL_BLOCK..offset + COL_BLOCK + POSEIDON_RATE].copy_from_slice(&block);
        values[offset + COL_STATE_AFTER..offset + COL_STATE_AFTER + POSEIDON_WIDTH]
            .copy_from_slice(&state);
    }

    let mut public_values = Vec::with_capacity(PUBLIC_VALUES);
    public_values.push(Goldilocks::from_usize(batch.len()));
    public_values.extend(state[..PUBLIC_DIGEST_WIDTH].iter().copied());

    Ok(SignatureBatchTrace {
        air: SignatureBatchAir::new(height),
        trace: RowMajorMatrix::new(values, TRACE_WIDTH),
        public_values,
        rows: height,
        columns: TRACE_WIDTH,
    })
}

/// Re-evaluates every constraint in [`SignatureBatchAir`] against an
/// already-built trace and panics on the first violation. A debug-time
/// sanity check between generating a trace and actually proving it — much
/// cheaper than discovering a malformed trace via a failed STARK proof.
pub fn check_constraints(trace: &SignatureBatchTrace) {
    p3_air::check_constraints(&trace.air, &trace.trace, &trace.public_values);
}

