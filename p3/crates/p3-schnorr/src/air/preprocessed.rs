// Builds the fixed (witness-independent) preprocessed trace: one-hot round
// and EC-window selectors, and the table of `G`'s precomputed multiples used
// as the `s*G` addend in EC rows. Also hosts `scalar_reduction_witness`, the
// host-side search for the modular-reduction quotient/carries that
// `trace_generation.rs` needs when filling in the (non-preprocessed) main
// trace's `COL_REDUCTION_*` columns.
//
// Everything here depends only on the row count, never on a specific
// signature's bytes — the prover and verifier compute the exact same
// preprocessed trace independently, so it never needs to be committed to or
// sent as part of the proof for column values, only have its shape
// (`PREPROCESSED_WIDTH`) agreed on.

fn ec_bit_selector_col(bit: usize) -> usize {
    debug_assert!(bit < EC_BIT_SELECTORS);
    PREPROCESSED_COL_EC_BIT_SELECTORS + bit
}

fn ec_chunk_selector_col(chunk: usize) -> usize {
    debug_assert!(chunk < EC_CHUNK_SELECTORS);
    PREPROCESSED_COL_EC_CHUNK_SELECTORS + chunk
}

/// Writes one `Fp5`'s 5 limbs into `values` at `start_col`, converting from
/// `Goldilocks` to the generic preprocessed-trace field `F`.
fn fill_preprocessed_fp5<F>(values: &mut [F], offset: usize, start_col: usize, value: &Fp5)
where
    F: PrimeCharacteristicRing,
{
    for lane in 0..FP5_LIMBS {
        values[offset + start_col + lane] = F::from_u64(value[lane].as_canonical_u64());
    }
}

/// `EC_SCALAR_BITS` successive powers-of-two multiples of the generator `G`:
/// `ladder[i] = 2^i * G`. Doubling-chain construction, so it costs one
/// [`AffinePoint::double`] per entry rather than `EC_SCALAR_BITS` independent
/// scalar multiplications. Used to fill each EC row's 4-bit window of `s*G`
/// addends (`ladder[window_start_bit + bit]` for `bit in 0..EC_WINDOW_BITS`).
fn ec_generator_ladder() -> Vec<AffinePoint> {
    let mut ladder = Vec::with_capacity(EC_SCALAR_BITS);
    let mut addend = AffinePoint::generator();
    for bit in 0..EC_SCALAR_BITS {
        ladder.push(addend);
        if bit + 1 < EC_SCALAR_BITS {
            let (next, _) = addend
                .double()
                .expect("ECgFp5 generator ladder doubles without exceptional denominator");
            addend = next;
        }
    }
    ladder
}

/// Witness values proving the challenge digest reduces to `e_chunks` modulo
/// the scalar field order: the digest reinterpreted as 32-bit limbs, the
/// quotient, and the per-limb carries from long division.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScalarReductionWitness {
    digest_u32: [Goldilocks; SCALAR_U32_LIMBS],
    quotient: Goldilocks,
    carries: [Goldilocks; REDUCTION_CARRY_COUNT],
}

/// Searches `quotient in {0, 1, 2}` for the one satisfying
/// `digest_u32 == e_chunks + quotient * SCALAR_MODULUS_U32` (as a multi-limb
/// addition with carries), i.e. finds how many copies of the scalar modulus
/// must be added to the claimed scalar `e_chunks` to recover the actual
/// (un-reduced) challenge digest. Three candidates (quotient < 3) suffice
/// because the digest's maximum possible value is less than 3x the scalar
/// modulus. Fails with [`crate::Error::ChallengeScalarMismatch`] if no
/// quotient works (the signature's claimed `e` does not match the
/// recomputed challenge) or if `e_chunks >= SCALAR_MODULUS_U32` — `e`
/// matches the challenge digest's residue class but isn't its *canonical*
/// representative (e.g. the prover supplied `e = digest` unreduced). The
/// in-circuit counterpart of that second check is
/// [`crate::air::constraints::constrain_scalar_lt_modulus`], driven by
/// [`scalar_lt_modulus_window_borrows`]'s per-window witness.
fn scalar_reduction_witness(
    digest: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH],
    e_chunks: [Goldilocks; SCALAR_U32_LIMBS],
) -> Result<ScalarReductionWitness> {
    let digest_u32_u64 = split_digest_u32(digest);
    let e_u32_u64 = e_chunks.map(|chunk| chunk.as_canonical_u64());
    if e_u32_u64.iter().any(|&chunk| chunk >= U32_BASE) {
        return Err(crate::Error::ChallengeScalarMismatch);
    }
    require_scalar_lt_modulus(&e_u32_u64, "e")?;

    for quotient in 0..=2u64 {
        let mut carries = [0u64; REDUCTION_CARRY_COUNT];
        let mut matches_digest = true;
        for limb in 0..SCALAR_U32_LIMBS {
            let total = e_u32_u64[limb] as u128
                + quotient as u128 * SCALAR_MODULUS_U32[limb] as u128
                + carries[limb] as u128;
            let limb_value = (total & U32_MASK as u128) as u64;
            carries[limb + 1] = (total >> 32) as u64;
            if limb_value != digest_u32_u64[limb] {
                matches_digest = false;
                break;
            }
        }

        if matches_digest && carries[SCALAR_U32_LIMBS] == 0 {
            return Ok(ScalarReductionWitness {
                digest_u32: digest_u32_u64.map(Goldilocks::from_u64),
                quotient: Goldilocks::from_u64(quotient),
                carries: carries.map(Goldilocks::from_u64),
            });
        }
    }

    Err(crate::Error::ChallengeScalarMismatch)
}

/// This EC row's window's fixed `EC_WINDOW_BITS`-bit slice of
/// `SCALAR_MODULUS_U32`, given the chunk/window indices any of the three
/// call sites below (the preprocessed-trace filler, and the two host-side
/// borrow-chain computations) derive from a row or window index. Shared so
/// the shift/mask arithmetic is written exactly once.
fn modulus_window_for(chunk: usize, window: usize) -> u64 {
    let window_mask = (1u64 << EC_WINDOW_BITS) - 1;
    (SCALAR_MODULUS_U32[chunk] >> (window * EC_WINDOW_BITS)) & window_mask
}

/// Host-side check mirroring the in-circuit `scalar < SCALAR_MODULUS_U32`
/// gadget (used for both `s` and `e`): fails with
/// [`crate::Error::NonCanonicalSignatureScalar`] (naming `field`, e.g. `"s"`
/// or `"e"`) unless `little_endian_u32_limbs` is strictly less than the
/// modulus, i.e. unless the final window's borrow-out (from
/// [`scalar_lt_modulus_window_borrows`]) is 1.
fn require_scalar_lt_modulus(
    little_endian_u32_limbs: &[u64; SCALAR_U32_LIMBS],
    field: &'static str,
) -> Result<()> {
    if scalar_lt_modulus_window_borrows(little_endian_u32_limbs)[EC_SCALAR_WINDOWS - 1] != 1 {
        return Err(crate::Error::NonCanonicalSignatureScalar { field });
    }
    Ok(())
}

/// Per-EC-window borrow-out bits for the subtraction `scalar -
/// SCALAR_MODULUS_U32`, processed one `EC_WINDOW_BITS`-bit window at a time
/// in the same least-significant-window-first order the EC rows themselves
/// run in (window 0 = bits `0..EC_WINDOW_BITS` of limb 0, ..., window
/// `EC_SCALAR_WINDOWS - 1` = the top window of the top limb). Standard
/// binary-subtraction borrow propagation: `borrow_out = 1` iff
/// `scalar_window - modulus_window - borrow_in` went negative. The final
/// entry is 1 iff `scalar < SCALAR_MODULUS_U32` overall (the subtraction
/// borrowed past the most significant window, which only happens when the
/// minuend is smaller than the subtrahend `SCALAR_MODULUS_U32`) — see
/// [`crate::air::constraints::constrain_scalar_lt_modulus`] for the
/// in-circuit mirror, which also needs each window's *difference* (not
/// computed here, since the borrow-out bits alone are enough to decide
/// canonicality; `trace_generation.rs` recomputes the difference bits
/// directly since it already has the scalar's bits per window).
fn scalar_lt_modulus_window_borrows(
    little_endian_u32_limbs: &[u64; SCALAR_U32_LIMBS],
) -> [u64; EC_SCALAR_WINDOWS] {
    let mut borrows = [0u64; EC_SCALAR_WINDOWS];
    let mut borrow_in = 0u64;
    #[allow(clippy::needless_range_loop)]
    for window in 0..EC_SCALAR_WINDOWS {
        let chunk = window / EC_WINDOWS_PER_CHUNK;
        let bit_offset = (window % EC_WINDOWS_PER_CHUNK) * EC_WINDOW_BITS;
        let window_mask = (1u64 << EC_WINDOW_BITS) - 1;
        let modulus_window = modulus_window_for(chunk, window % EC_WINDOWS_PER_CHUNK);
        let scalar_window = (little_endian_u32_limbs[chunk] >> bit_offset) & window_mask;
        let diff = scalar_window as i64 - modulus_window as i64 - borrow_in as i64;
        borrow_in = if diff < 0 { 1 } else { 0 };
        borrows[window] = borrow_in;
    }
    borrows
}

/// Reinterprets each of the digest's 5 Goldilocks limbs as two 32-bit chunks
/// (low, then high), producing `SCALAR_U32_LIMBS` chunks total.
fn split_digest_u32(
    digest: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH],
) -> [u64; SCALAR_U32_LIMBS] {
    let mut out = [0u64; SCALAR_U32_LIMBS];
    for (lane, value) in digest.iter().enumerate() {
        let value = value.as_canonical_u64();
        out[2 * lane] = value & U32_MASK;
        out[2 * lane + 1] = value >> 32;
    }
    out
}

/// The STARK trace height (a power of two, as required by the FRI backend)
/// for a batch of `count` signatures: enough rows for `count *
/// ROWS_PER_SIGNATURE`, rounded up. The padding rows beyond the real
/// signatures are disabled (`COL_ENABLED = 0`).
pub fn trace_height_for_count(count: usize) -> usize {
    (count * ROWS_PER_SIGNATURE).max(1).next_power_of_two()
}

/// Builds the `rows`-row preprocessed trace by repeating one
/// `ROWS_PER_SIGNATURE`-row pattern: round selectors for the hashing rows,
/// then window/chunk selectors plus this window's slice of the generator
/// ladder for the EC rows. Note this fills selectors for every row up to
/// `rows`, including padding rows past the batch's real signatures —
/// `COL_ENABLED` (a main-trace, not preprocessed, column) is what disables
/// constraints there.
fn preprocessed_trace_for_rows<F>(rows: usize) -> RowMajorMatrix<F>
where
    F: PrimeCharacteristicRing + Send + Sync,
{
    let mut values = vec![F::ZERO; rows * PREPROCESSED_WIDTH];
    let generator_ladder = ec_generator_ladder();
    for row in 0..rows {
        let offset = row * PREPROCESSED_WIDTH;
        let signature_row = row % ROWS_PER_SIGNATURE;
        if signature_row < PUBLIC_HASH_ROWS_PER_SIGNATURE {
            let round = signature_row % POSEIDON_ROUND_STATE_COUNT;
            values[offset + round_selector_col(round)] = F::ONE;
        } else {
            let ec_window = signature_row - PUBLIC_HASH_ROWS_PER_SIGNATURE;
            let window = ec_window % EC_WINDOWS_PER_CHUNK;
            let chunk = ec_window / EC_WINDOWS_PER_CHUNK;
            values[offset + ec_bit_selector_col(window)] = F::ONE;
            values[offset + ec_chunk_selector_col(chunk)] = F::ONE;
            let start_bit = ec_window * EC_WINDOW_BITS;
            for bit in 0..EC_WINDOW_BITS {
                let addend = &generator_ladder[start_bit + bit];
                fill_preprocessed_fp5(
                    values.as_mut_slice(),
                    offset,
                    preprocessed_s_addend_x_col(bit),
                    &addend.x,
                );
                fill_preprocessed_fp5(
                    values.as_mut_slice(),
                    offset,
                    preprocessed_s_addend_u_col(bit),
                    &addend.u,
                );
            }
            values[offset + PREPROCESSED_COL_MODULUS_WINDOW] =
                F::from_u64(modulus_window_for(chunk, window));
        }
    }
    RowMajorMatrix::new(values, PREPROCESSED_WIDTH)
}
