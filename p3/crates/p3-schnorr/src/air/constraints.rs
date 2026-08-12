// Constraint helpers factored out of `eval.rs::SignatureBatchAir::eval`. Pure
// functions of `(builder, selector, ...expressions)` — `selector` gates every
// assertion so a constraint written here has no effect on rows where it
// isn't selected, letting the same helper be reused across phases/row-types
// (e.g. `constrain_affine_add`/`constrain_affine_double` back both the
// public-key/`R` arithmetic and the windowed-scalar-mul EC rows).

/// Checks that `affine_x` is the x-coordinate a public encoding `encoded =
/// 1/u` decodes to, and that `inverse`/`sqrt_delta` are the witnesses that
/// decoding needed: `encoded * inverse == 1`, `sqrt_delta^2 == delta` (where
/// `delta = e^2 - 4B`, `e = encoded^2 - A`), and `affine_x^2 - e*affine_x + B
/// == 0` (the curve equation rearranged to solve for `x` given `e`). See
/// `hints.rs::encoded_point_decode_hint` for how the prover computes these
/// off-circuit, including the root-selection step this function does *not*
/// check — see [`constrain_decoded_public_key_root`] for that.
fn constrain_encoded_point_decode<AB>(
    builder: &mut AB,
    selector: AB::Expr,
    encoded: &[AB::Expr; FP5_LIMBS],
    inverse: &[AB::Expr; FP5_LIMBS],
    sqrt_delta: &[AB::Expr; FP5_LIMBS],
    affine_x: &[AB::Expr; FP5_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let product = fp5_mul_expr::<AB>(encoded, inverse);
    let sqrt_delta_squared = fp5_square_expr::<AB>(sqrt_delta);
    let e = encoded_point_e_expr::<AB>(encoded);
    let delta = encoded_point_delta_expr::<AB>(&e);
    let mut decode_root_relation = fp5_square_expr::<AB>(affine_x);
    let e_mul_x = fp5_mul_expr::<AB>(&e, affine_x);
    for lane in 0..FP5_LIMBS {
        decode_root_relation[lane] -= e_mul_x[lane].dup();
        decode_root_relation[lane] += AB::Expr::from_u64(ECGFP5_B[lane]);
    }

    for lane in 0..FP5_LIMBS {
        let expected_inverse_product = if lane == 0 {
            AB::Expr::ONE
        } else {
            AB::Expr::ZERO
        };
        builder.assert_zero(selector.dup() * (product[lane].dup() - expected_inverse_product));
        builder.assert_zero(selector.dup() * (sqrt_delta_squared[lane].dup() - delta[lane].dup()));
        builder.assert_zero(selector.dup() * decode_root_relation[lane].dup());
    }
}

/// Checks that the public key's decoded `affine_x` is specifically the
/// *non-square* root of `x^2 - e*x + B = 0` (the curve constant `B` is a
/// non-square, so the quadratic's two roots always have opposite square
/// status — see the inline comment below): proves `other_root = e -
/// affine_x` (the root *not* chosen) is itself a square, via the witnessed
/// `other_root_sqrt`. This pins down which of the two valid x-coordinates a
/// public key's encoding refers to without computing a Legendre symbol
/// in-circuit. Only applied to the public key, not `reconstructed_r` — `R`'s
/// root choice doesn't need disambiguating, because [`constrain_ec_row`]
/// independently re-derives `R` from `s`, `e`, and the public key and checks
/// it against the decoded value, which already pins down the correct root.
fn constrain_decoded_public_key_root<AB>(
    builder: &mut AB,
    selector: AB::Expr,
    encoded: &[AB::Expr; FP5_LIMBS],
    affine_x: &[AB::Expr; FP5_LIMBS],
    other_root_sqrt: &[AB::Expr; FP5_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let e = encoded_point_e_expr::<AB>(encoded);
    let other_root = fp5_sub_expr::<AB>(&e, affine_x);
    let other_root_sqrt_squared = fp5_square_expr::<AB>(other_root_sqrt);

    // The curve constant B is a nonsquare, so the two quadratic roots have
    // opposite square status. The reference decoder selects the nonsquare root.
    for lane in 0..FP5_LIMBS {
        builder.assert_zero(
            selector.dup() * (other_root_sqrt_squared[lane].dup() - other_root[lane].dup()),
        );
    }
}

/// Binds the per-row absorbed hash blocks to the persistent `s_chunks`/`e_chunks`
/// columns that the EC rows later consume: on the phase-0 row, copies `s`'s
/// first 3 chunks out of the block lanes not used by `reconstructed_r`; on
/// the phase-1 row, copies the rest of `s` and `e`'s first chunk; `e`'s
/// remaining chunks are bound separately in `eval.rs` on the phase-2/3 rows
/// (against `COL_REDUCTION_E_CHUNKS`, not `s_chunks`/`e_chunks` directly) once
/// they've gone through the reduction-digest absorption. See `air/mod.rs`'s
/// phase descriptions for why the signature's raw bytes are threaded through
/// the hash absorption rows like this rather than loaded directly.
#[allow(clippy::too_many_arguments)]
fn constrain_signature_chunks<AB>(
    builder: &mut AB,
    phase_zero: AB::Expr,
    phase_one: AB::Expr,
    phase_two: AB::Expr,
    phase_three: AB::Expr,
    row: &[AB::Var],
    s_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
    e_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    for chunk in 0..3 {
        let block: AB::Expr = row[COL_BLOCK + 5 + chunk].into();
        builder.assert_zero(phase_zero.dup() * (s_chunks[chunk].dup() - block));
    }
    for chunk in 3..SCALAR_U32_LIMBS {
        let block: AB::Expr = row[COL_BLOCK + chunk - 3].into();
        builder.assert_zero(phase_one.dup() * (s_chunks[chunk].dup() - block));
    }

    let e0_block: AB::Expr = row[COL_BLOCK + 7].into();
    builder.assert_zero(phase_one * (e_chunks[0].dup() - e0_block));
    for chunk in 1..SCALAR_U32_LIMBS - 1 {
        let block: AB::Expr = row[COL_BLOCK + chunk - 1].into();
        builder.assert_zero(phase_two.dup() * (e_chunks[chunk].dup() - block));
    }
    let e9_block: AB::Expr = row[COL_BLOCK].into();
    builder.assert_zero(phase_three * (e_chunks[SCALAR_U32_LIMBS - 1].dup() - e9_block));
}

/// The full constraint set for one EC row: validates this window's bit
/// decomposition of `s` and `e`, accumulates those bits back into the
/// persistent per-chunk accumulators (checked against `s_chunks`/`e_chunks`
/// at each chunk boundary), range-checks the digest chunks via
/// [`constrain_digest_u32_range_row`] (piggybacking on the same bit columns),
/// builds this window's table of `e*pk` addend multiples by repeated
/// doubling, then runs [`EC_WINDOW_STEPS`] conditional-add steps
/// (alternating an `s*G` step and an `e*pk` step) advancing the running
/// `(acc_x, acc_u)` accumulator. On the final EC row (`ec_final`), asserts
/// the accumulator equals the witnessed `target_r` and is not the identity
/// (a valid `R` is never the point at infinity).
#[allow(clippy::too_many_arguments)]
fn constrain_ec_row<AB>(
    builder: &mut AB,
    ec_active: AB::Expr,
    ec_final: AB::Expr,
    row: &[AB::Var],
    s_addend_x: &[[AB::Expr; FP5_LIMBS]; EC_WINDOW_BITS],
    s_addend_u: &[[AB::Expr; FP5_LIMBS]; EC_WINDOW_BITS],
    bit_selectors: &[AB::Expr; EC_BIT_SELECTORS],
    chunk_selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
    digest_u32_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
    s_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
    e_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
    target_r_x: &[AB::Expr; FP5_LIMBS],
    target_r_u: &[AB::Expr; FP5_LIMBS],
    modulus_window: AB::Expr,
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let s_bits: [AB::Expr; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| row[COL_EC_S_BITS + bit].into());
    let e_bits: [AB::Expr; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| row[COL_EC_E_BITS + bit].into());
    for bit in 0..EC_WINDOW_BITS {
        builder
            .assert_zero(ec_active.dup() * s_bits[bit].dup() * (s_bits[bit].dup() - AB::Expr::ONE));
        builder
            .assert_zero(ec_active.dup() * e_bits[bit].dup() * (e_bits[bit].dup() - AB::Expr::ONE));
    }

    let s_lt_modulus_borrow_out = constrain_scalar_lt_modulus::<AB>(
        builder,
        ec_active.dup(),
        row,
        &s_bits,
        modulus_window.dup(),
        COL_EC_S_LT_MODULUS_BORROW_IN,
        COL_EC_S_LT_MODULUS_DIFF_BITS,
    );
    builder.assert_zero(ec_final.dup() * (s_lt_modulus_borrow_out - AB::Expr::ONE));
    let e_lt_modulus_borrow_out = constrain_scalar_lt_modulus::<AB>(
        builder,
        ec_active.dup(),
        row,
        &e_bits,
        modulus_window,
        COL_EC_E_LT_MODULUS_BORROW_IN,
        COL_EC_E_LT_MODULUS_DIFF_BITS,
    );
    builder.assert_zero(ec_final.dup() * (e_lt_modulus_borrow_out - AB::Expr::ONE));

    let chunk_end = bit_selectors[EC_BIT_SELECTORS - 1].dup();
    builder.assert_zero(ec_final.dup() * (chunk_end.dup() - AB::Expr::ONE));
    builder.assert_zero(
        ec_final.dup() * (chunk_selectors[EC_CHUNK_SELECTORS - 1].dup() - AB::Expr::ONE),
    );

    let s_chunk_acc: AB::Expr = row[COL_EC_S_CHUNK_ACC].into();
    let e_chunk_acc: AB::Expr = row[COL_EC_E_CHUNK_ACC].into();
    let s_chunk_after = s_chunk_acc + ec_window_bits_value_expr::<AB>(&s_bits, bit_selectors);
    let e_chunk_after = e_chunk_acc + ec_window_bits_value_expr::<AB>(&e_bits, bit_selectors);
    let selected_s_chunk = selected_chunk_expr::<AB>(chunk_selectors, s_chunks);
    let selected_e_chunk = selected_chunk_expr::<AB>(chunk_selectors, e_chunks);
    builder.assert_zero(ec_active.dup() * chunk_end.dup() * (s_chunk_after - selected_s_chunk));
    builder.assert_zero(ec_active.dup() * chunk_end.dup() * (e_chunk_after - selected_e_chunk));
    constrain_digest_u32_range_row::<AB>(
        builder,
        ec_active.dup(),
        chunk_end.dup(),
        bit_selectors,
        chunk_selectors,
        row,
        digest_u32_chunks,
    );

    let e_addend_x: [[AB::Expr; FP5_LIMBS]; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| fp5_from_row::<AB>(row, ec_e_addend_x_col(bit)));
    let e_addend_u: [[AB::Expr; FP5_LIMBS]; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| fp5_from_row::<AB>(row, ec_e_addend_u_col(bit)));
    for bit in 0..EC_WINDOW_BITS - 1 {
        let e_double_x_inv = fp5_from_row::<AB>(row, ec_e_double_x_den_inv_col(bit));
        let e_double_u_inv = fp5_from_row::<AB>(row, ec_e_double_u_den_inv_col(bit));
        constrain_affine_double::<AB>(
            builder,
            ec_active.dup(),
            &e_addend_x[bit],
            &e_addend_u[bit],
            &e_addend_x[bit + 1],
            &e_addend_u[bit + 1],
            &e_double_x_inv,
            &e_double_u_inv,
        );
    }
    let acc_x = fp5_from_row::<AB>(row, COL_EC_ACC_X);
    let acc_u = fp5_from_row::<AB>(row, COL_EC_ACC_U);
    let acc_is_identity: AB::Expr = row[COL_EC_ACC_IS_IDENTITY].into();
    let step_x: [[AB::Expr; FP5_LIMBS]; EC_WINDOW_STEPS] =
        core::array::from_fn(|step| fp5_from_row::<AB>(row, ec_step_x_col(step)));
    let step_u: [[AB::Expr; FP5_LIMBS]; EC_WINDOW_STEPS] =
        core::array::from_fn(|step| fp5_from_row::<AB>(row, ec_step_u_col(step)));
    let step_is_identity: [AB::Expr; EC_WINDOW_STEPS] =
        core::array::from_fn(|step| row[ec_step_is_identity_col(step)].into());

    builder.assert_zero(
        ec_active.dup() * acc_is_identity.dup() * (acc_is_identity.dup() - AB::Expr::ONE),
    );
    for step in 0..EC_WINDOW_STEPS {
        builder.assert_zero(
            ec_active.dup()
                * step_is_identity[step].dup()
                * (step_is_identity[step].dup() - AB::Expr::ONE),
        );

        let lhs_x = if step == 0 { &acc_x } else { &step_x[step - 1] };
        let lhs_u = if step == 0 { &acc_u } else { &step_u[step - 1] };
        let lhs_is_identity = if step == 0 {
            acc_is_identity.dup()
        } else {
            step_is_identity[step - 1].dup()
        };
        let window_bit = step / 2;
        let add_bit = if step % 2 == 0 {
            s_bits[window_bit].dup()
        } else {
            e_bits[window_bit].dup()
        };
        let rhs_x = if step % 2 == 0 {
            &s_addend_x[window_bit]
        } else {
            &e_addend_x[window_bit]
        };
        let rhs_u = if step % 2 == 0 {
            &s_addend_u[window_bit]
        } else {
            &e_addend_u[window_bit]
        };
        let add_x_inv = fp5_from_row::<AB>(row, ec_add_x_den_inv_col(step));
        let add_u_inv = fp5_from_row::<AB>(row, ec_add_u_den_inv_col(step));
        constrain_conditional_add::<AB>(
            builder,
            ec_active.dup(),
            add_bit,
            lhs_x,
            lhs_u,
            lhs_is_identity,
            rhs_x,
            rhs_u,
            &step_x[step],
            &step_u[step],
            step_is_identity[step].dup(),
            &add_x_inv,
            &add_u_inv,
        );
    }

    for lane in 0..FP5_LIMBS {
        builder.assert_zero(
            ec_final.dup() * (step_x[EC_WINDOW_STEPS - 1][lane].dup() - target_r_x[lane].dup()),
        );
        builder.assert_zero(
            ec_final.dup() * (step_u[EC_WINDOW_STEPS - 1][lane].dup() - target_r_u[lane].dup()),
        );
    }
    builder.assert_zero(ec_final * step_is_identity[EC_WINDOW_STEPS - 1].dup());
}

/// Proves `scalar < SCALAR_MODULUS_U32` window-by-window (canonicality of a
/// signature scalar — `s` or `e` — not merely its congruence to some other
/// value mod the scalar order — see `ScalarReductionWitness`'s doc comment
/// in `preprocessed.rs` for why `e`'s long-division reduction against the
/// challenge digest alone doesn't already guarantee this, and note `s` has
/// no such reduction at all, so this is `s`'s only canonicality check).
/// Reuses this row's `scalar_bits` (already boolean-checked and proven to
/// reconstruct the persistent `s_chunks`/`e_chunks` columns by the caller)
/// rather than decomposing the scalar a second time. Called once for `s`
/// and once for `e` per row, at `(borrow_in_col, diff_bits_col) =
/// (COL_EC_S_LT_MODULUS_BORROW_IN, COL_EC_S_LT_MODULUS_DIFF_BITS)` or the
/// `E` counterparts.
///
/// Standard binary-subtraction borrow propagation for `scalar -
/// SCALAR_MODULUS_U32` (borrowing past the top window means `scalar <
/// modulus`), one `EC_WINDOW_BITS`-bit window at a time in trace order
/// (window 0 = least significant): this window's shifted difference
/// `(scalar_window - modulus_window - borrow_in) + 2^EC_WINDOW_BITS` is
/// witnessed via its `EC_SCALAR_LT_MODULUS_DIFF_BITS`-bit decomposition
/// (`diff_bits_col`), which both range-checks the shifted difference to
/// `[0, 2^(EC_WINDOW_BITS+1))` and — via its top bit, see
/// [`scalar_lt_modulus_top_bit_col`] — determines this window's borrow-out
/// purely as an expression, with no separate borrow-out witness needed
/// (`borrow_out = 1` iff the top bit is 0, i.e. iff the unshifted
/// difference was negative). `borrow_in` itself is also boolean-checked
/// here as defense in depth: it's already pinned to either the constant 0
/// or the previous window's (boolean, by the same argument) borrow-out by
/// `constrain_ec_transition`, but asserting it directly means that
/// invariant doesn't have to hold by construction alone. Returns the
/// borrow-out expression; the caller asserts it's 1 on the final EC window
/// (`ec_final`), and `constrain_ec_transition` wires it into the next
/// window's `borrow_in_col`.
fn constrain_scalar_lt_modulus<AB>(
    builder: &mut AB,
    ec_active: AB::Expr,
    row: &[AB::Var],
    scalar_bits: &[AB::Expr; EC_WINDOW_BITS],
    modulus_window: AB::Expr,
    borrow_in_col: usize,
    diff_bits_col: usize,
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let borrow_in: AB::Expr = row[borrow_in_col].into();
    builder.assert_zero(ec_active.dup() * borrow_in.dup() * (borrow_in.dup() - AB::Expr::ONE));
    let diff_bits: [AB::Expr; EC_SCALAR_LT_MODULUS_DIFF_BITS] =
        core::array::from_fn(|bit| row[diff_bits_col + bit].into());
    for bit in &diff_bits {
        builder.assert_zero(ec_active.dup() * bit.dup() * (bit.dup() - AB::Expr::ONE));
    }

    let scalar_window = bits_to_value_expr::<AB, EC_WINDOW_BITS>(scalar_bits);
    let diff_shifted = bits_to_value_expr::<AB, EC_SCALAR_LT_MODULUS_DIFF_BITS>(&diff_bits);
    builder.assert_zero(
        ec_active.dup()
            * (diff_shifted
                - (scalar_window - modulus_window - borrow_in
                    + AB::Expr::from_u64(1u64 << EC_WINDOW_BITS))),
    );

    AB::Expr::ONE - diff_bits[EC_WINDOW_BITS].dup()
}

/// Little-endian bit-weighted sum: `sum_bit bits[bit] * 2^bit`. Shared by
/// [`constrain_scalar_lt_modulus`]'s `scalar_window`/`diff_shifted`
/// reconstructions; unlike [`ec_window_bits_value_expr`], has no one-hot
/// window selector factor, since callers here always mean "this window"
/// rather than "whichever window is selected".
fn bits_to_value_expr<AB, const N: usize>(bits: &[AB::Expr; N]) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    bits.iter()
        .enumerate()
        .fold(AB::Expr::ZERO, |acc, (bit, value)| {
            acc + value.dup() * AB::Expr::from_u64(1u64 << bit)
        })
}

/// Row-to-row wiring for the EC accumulator: at the start of a new signature
/// (`signature_last`, i.e. the row after the last hashing row), resets the
/// accumulator to the identity and seeds the `e*pk` addend table's first
/// entry from the decoded public key; otherwise (`ec_nonfinal`), carries the
/// previous row's final step result into this row's starting accumulator,
/// doubles the previous row's last `e*pk` addend into this row's first
/// addend (continuing the same doubling chain across window boundaries), and
/// carries the chunk-accumulator/digest-prefix state forward unless this
/// row ended a chunk (in which case it resets to 0, since
/// [`constrain_ec_row`] already validated the completed chunk).
#[allow(clippy::too_many_arguments)]
fn constrain_ec_transition<AB>(
    transition: &mut AB,
    signature_last: AB::Expr,
    ec_active: AB::Expr,
    ec_final: AB::Expr,
    bit_selectors: &[AB::Expr; EC_BIT_SELECTORS],
    chunk_selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
    local: &[AB::Var],
    next: &[AB::Var],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let next_enabled: AB::Expr = next[COL_ENABLED].into();
    let next_hash_active: AB::Expr = next[COL_HASH_ACTIVE].into();
    let next_ec_active = next_enabled - next_hash_active;

    transition.assert_zero(signature_last.dup() * (next_ec_active.dup() - AB::Expr::ONE));
    transition.assert_zero(signature_last.dup() * Into::<AB::Expr>::into(next[COL_EC_ACC_X]));
    transition.assert_zero(signature_last.dup() * Into::<AB::Expr>::into(next[COL_EC_ACC_U]));
    transition.assert_zero(signature_last.dup() * (next[COL_EC_ACC_IS_IDENTITY] - AB::Expr::ONE));
    transition.assert_zero(signature_last.dup() * Into::<AB::Expr>::into(next[COL_EC_S_CHUNK_ACC]));
    transition.assert_zero(signature_last.dup() * Into::<AB::Expr>::into(next[COL_EC_E_CHUNK_ACC]));
    transition.assert_zero(
        signature_last.dup() * Into::<AB::Expr>::into(next[COL_EC_S_LT_MODULUS_BORROW_IN]),
    );
    transition.assert_zero(
        signature_last.dup() * Into::<AB::Expr>::into(next[COL_EC_E_LT_MODULUS_BORROW_IN]),
    );
    transition.assert_zero(
        signature_last.dup() * Into::<AB::Expr>::into(next[COL_REDUCTION_DIGEST_CHUNK_ACC]),
    );
    transition.assert_zero(
        signature_last.dup() * Into::<AB::Expr>::into(next[COL_REDUCTION_DIGEST_HI_PREFIX_ACC]),
    );

    // Wire this window's borrow-out into the next window's borrow-in, for
    // both `s` and `e`. The range check binding `local`'s diff-bits column
    // to `modulus_window - scalar_window - borrow_in` already happened in
    // `constrain_scalar_lt_modulus` (called on every row via
    // `constrain_ec_row`, including this one) — here we only need to read
    // back the same top-bit expression to know the resulting borrow-out.
    let local_s_borrow_out = AB::Expr::ONE
        - Into::<AB::Expr>::into(
            local[scalar_lt_modulus_top_bit_col(COL_EC_S_LT_MODULUS_DIFF_BITS)],
        );
    let local_e_borrow_out = AB::Expr::ONE
        - Into::<AB::Expr>::into(
            local[scalar_lt_modulus_top_bit_col(COL_EC_E_LT_MODULUS_DIFF_BITS)],
        );

    for lane in 0..FP5_LIMBS {
        transition.assert_zero(
            signature_last.dup() * (next[COL_EC_E_ADDEND_X + lane] - local[COL_PK_AFFINE_X + lane]),
        );
        transition.assert_zero(
            signature_last.dup()
                * (next[COL_EC_E_ADDEND_U + lane] - local[COL_PK_DECODE_INVERSE + lane]),
        );
    }

    let ec_nonfinal = ec_active.dup() * (AB::Expr::ONE - ec_final);
    transition.assert_zero(ec_nonfinal.dup() * (next_ec_active - AB::Expr::ONE));
    transition.assert_zero(
        ec_nonfinal.dup() * (next[COL_EC_S_LT_MODULUS_BORROW_IN] - local_s_borrow_out),
    );
    transition.assert_zero(
        ec_nonfinal.dup() * (next[COL_EC_E_LT_MODULUS_BORROW_IN] - local_e_borrow_out),
    );

    let e_addend_x = fp5_from_row::<AB>(local, ec_e_addend_x_col(EC_WINDOW_BITS - 1));
    let e_addend_u = fp5_from_row::<AB>(local, ec_e_addend_u_col(EC_WINDOW_BITS - 1));
    let next_e_addend_x = fp5_from_row::<AB>(next, ec_e_addend_x_col(0));
    let next_e_addend_u = fp5_from_row::<AB>(next, ec_e_addend_u_col(0));
    let e_double_x_inv = fp5_from_row::<AB>(local, ec_e_double_x_den_inv_col(EC_WINDOW_BITS - 1));
    let e_double_u_inv = fp5_from_row::<AB>(local, ec_e_double_u_den_inv_col(EC_WINDOW_BITS - 1));
    constrain_affine_double::<AB>(
        transition,
        ec_nonfinal.dup(),
        &e_addend_x,
        &e_addend_u,
        &next_e_addend_x,
        &next_e_addend_u,
        &e_double_x_inv,
        &e_double_u_inv,
    );

    for lane in 0..FP5_LIMBS {
        transition.assert_zero(
            ec_nonfinal.dup()
                * (next[COL_EC_ACC_X + lane] - local[ec_step_x_col(EC_WINDOW_STEPS - 1) + lane]),
        );
        transition.assert_zero(
            ec_nonfinal.dup()
                * (next[COL_EC_ACC_U + lane] - local[ec_step_u_col(EC_WINDOW_STEPS - 1) + lane]),
        );
    }
    transition.assert_zero(
        ec_nonfinal.dup()
            * (next[COL_EC_ACC_IS_IDENTITY] - local[ec_step_is_identity_col(EC_WINDOW_STEPS - 1)]),
    );

    let chunk_end = bit_selectors[EC_BIT_SELECTORS - 1].dup();
    let s_bits: [AB::Expr; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| local[COL_EC_S_BITS + bit].into());
    let e_bits: [AB::Expr; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| local[COL_EC_E_BITS + bit].into());
    transition.assert_zero(
        ec_nonfinal.dup()
            * (next[COL_EC_S_CHUNK_ACC]
                - (AB::Expr::ONE - chunk_end.dup())
                    * (local[COL_EC_S_CHUNK_ACC].into()
                        + ec_window_bits_value_expr::<AB>(&s_bits, bit_selectors))),
    );
    transition.assert_zero(
        ec_nonfinal.dup()
            * (next[COL_EC_E_CHUNK_ACC]
                - (AB::Expr::ONE - chunk_end.dup())
                    * (local[COL_EC_E_CHUNK_ACC].into()
                        + ec_window_bits_value_expr::<AB>(&e_bits, bit_selectors))),
    );
    transition.assert_zero(
        ec_nonfinal.dup()
            * (next[COL_REDUCTION_DIGEST_CHUNK_ACC]
                - (AB::Expr::ONE - chunk_end.dup())
                    * digest_chunk_after_expr::<AB>(local, bit_selectors)),
    );
    transition.assert_zero(
        ec_nonfinal.dup()
            * (next[COL_REDUCTION_DIGEST_HI_PREFIX_ACC]
                - (AB::Expr::ONE - chunk_end.dup())
                    * digest_hi_prefix_after_expr::<AB>(local, bit_selectors, chunk_selectors)),
    );
}

/// One double-and-add step: if `bit` is 0, the result is `lhs` unchanged
/// (`no_add`); if `bit` is 1 and `lhs` is the identity, the result is `rhs`
/// (`from_identity`, since identity + anything = that thing, and the general
/// affine-add formula can't take the identity as input); otherwise
/// (`general`) the result is the real curve sum via [`constrain_affine_add`].
/// Also propagates the identity flag: identity in implies a fixed (zero)
/// encoding for `lhs`/`result`, used to keep the "is this the identity"
/// bookkeeping itself constrained rather than just trusted from the witness.
#[allow(clippy::too_many_arguments)]
fn constrain_conditional_add<AB>(
    builder: &mut AB,
    selector: AB::Expr,
    bit: AB::Expr,
    lhs_x: &[AB::Expr; FP5_LIMBS],
    lhs_u: &[AB::Expr; FP5_LIMBS],
    lhs_is_identity: AB::Expr,
    rhs_x: &[AB::Expr; FP5_LIMBS],
    rhs_u: &[AB::Expr; FP5_LIMBS],
    result_x: &[AB::Expr; FP5_LIMBS],
    result_u: &[AB::Expr; FP5_LIMBS],
    result_is_identity: AB::Expr,
    x_denominator_inverse: &[AB::Expr; FP5_LIMBS],
    u_denominator_inverse: &[AB::Expr; FP5_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let no_add = selector.dup() * (AB::Expr::ONE - bit.dup());
    let from_identity = selector.dup() * bit.dup() * lhs_is_identity.dup();
    let general = selector * bit * (AB::Expr::ONE - lhs_is_identity.dup());

    for lane in 0..FP5_LIMBS {
        builder.assert_zero(lhs_is_identity.dup() * lhs_x[lane].dup());
        builder.assert_zero(lhs_is_identity.dup() * lhs_u[lane].dup());
        builder.assert_zero(result_is_identity.dup() * result_x[lane].dup());
        builder.assert_zero(result_is_identity.dup() * result_u[lane].dup());
        builder.assert_zero(no_add.dup() * (result_x[lane].dup() - lhs_x[lane].dup()));
        builder.assert_zero(no_add.dup() * (result_u[lane].dup() - lhs_u[lane].dup()));
        builder.assert_zero(from_identity.dup() * (result_x[lane].dup() - rhs_x[lane].dup()));
        builder.assert_zero(from_identity.dup() * (result_u[lane].dup() - rhs_u[lane].dup()));
    }
    builder.assert_zero(no_add * (result_is_identity.dup() - lhs_is_identity));
    builder.assert_zero(from_identity * result_is_identity.dup());
    builder.assert_zero(general.dup() * result_is_identity);

    constrain_affine_add::<AB>(
        builder,
        general,
        lhs_x,
        lhs_u,
        rhs_x,
        rhs_u,
        result_x,
        result_u,
        x_denominator_inverse,
        u_denominator_inverse,
    );
}

/// Checks `(result_x, result_u) == lhs + rhs` using the unreduced addition
/// formula from [`crate::affine::affine_add_terms`] (restated in-circuit by
/// [`affine_add_terms_expr`]) via [`constrain_affine_operation_result`].
/// Callers must not invoke this where `lhs`/`rhs` could be equal, negatives
/// of each other, or the identity — [`constrain_conditional_add`] handles
/// those cases separately before falling through to this general formula.
#[allow(clippy::too_many_arguments)]
fn constrain_affine_add<AB>(
    builder: &mut AB,
    selector: AB::Expr,
    lhs_x: &[AB::Expr; FP5_LIMBS],
    lhs_u: &[AB::Expr; FP5_LIMBS],
    rhs_x: &[AB::Expr; FP5_LIMBS],
    rhs_u: &[AB::Expr; FP5_LIMBS],
    result_x: &[AB::Expr; FP5_LIMBS],
    result_u: &[AB::Expr; FP5_LIMBS],
    x_denominator_inverse: &[AB::Expr; FP5_LIMBS],
    u_denominator_inverse: &[AB::Expr; FP5_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let terms = affine_add_terms_expr::<AB>(lhs_x, lhs_u, rhs_x, rhs_u);
    constrain_affine_operation_result::<AB>(
        builder,
        selector,
        result_x,
        result_u,
        &terms,
        x_denominator_inverse,
        u_denominator_inverse,
    );
}

/// Checks `(result_x, result_u) == 2 * (point_x, point_u)` using the
/// unreduced doubling formula from [`crate::affine::affine_double_terms`]
/// (restated in-circuit by [`affine_double_terms_expr`]) via
/// [`constrain_affine_operation_result`]. Used to build the `e*pk` addend
/// table (doubling `pk` repeatedly) — never for `s*G`, since that table is
/// already precomputed in the preprocessed trace.
#[allow(clippy::too_many_arguments)]
fn constrain_affine_double<AB>(
    builder: &mut AB,
    selector: AB::Expr,
    point_x: &[AB::Expr; FP5_LIMBS],
    point_u: &[AB::Expr; FP5_LIMBS],
    result_x: &[AB::Expr; FP5_LIMBS],
    result_u: &[AB::Expr; FP5_LIMBS],
    x_denominator_inverse: &[AB::Expr; FP5_LIMBS],
    u_denominator_inverse: &[AB::Expr; FP5_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let terms = affine_double_terms_expr::<AB>(point_x, point_u);
    constrain_affine_operation_result::<AB>(
        builder,
        selector,
        result_x,
        result_u,
        &terms,
        x_denominator_inverse,
        u_denominator_inverse,
    );
}

/// In-circuit counterpart of [`crate::affine::AffineOperationTerms`]: the
/// numerator/denominator pairs from an addition or doubling formula, as
/// `AB::Expr` instead of concrete `Fp5` values.
#[derive(Clone, Debug)]
struct AffineOperationTermsExpr<E> {
    x_numerator: [E; FP5_LIMBS],
    x_denominator: [E; FP5_LIMBS],
    u_numerator: [E; FP5_LIMBS],
    u_denominator: [E; FP5_LIMBS],
}

/// Checks `result == numerator / denominator` (for both `x` and `u`)
/// without dividing: asserts `denominator * denominator_inverse == 1` (so
/// the inverse witness is genuine) and `result * denominator == numerator`
/// (so the division, given a genuine inverse, was performed correctly).
fn constrain_affine_operation_result<AB>(
    builder: &mut AB,
    selector: AB::Expr,
    result_x: &[AB::Expr; FP5_LIMBS],
    result_u: &[AB::Expr; FP5_LIMBS],
    terms: &AffineOperationTermsExpr<AB::Expr>,
    x_denominator_inverse: &[AB::Expr; FP5_LIMBS],
    u_denominator_inverse: &[AB::Expr; FP5_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let x_inverse_product = fp5_mul_expr::<AB>(&terms.x_denominator, x_denominator_inverse);
    let u_inverse_product = fp5_mul_expr::<AB>(&terms.u_denominator, u_denominator_inverse);
    let result_x_den = fp5_mul_expr::<AB>(result_x, &terms.x_denominator);
    let result_u_den = fp5_mul_expr::<AB>(result_u, &terms.u_denominator);
    for lane in 0..FP5_LIMBS {
        let expected = if lane == 0 {
            AB::Expr::ONE
        } else {
            AB::Expr::ZERO
        };
        builder.assert_zero(selector.dup() * (x_inverse_product[lane].dup() - expected.dup()));
        builder.assert_zero(selector.dup() * (u_inverse_product[lane].dup() - expected));
        builder.assert_zero(
            selector.dup() * (result_x_den[lane].dup() - terms.x_numerator[lane].dup()),
        );
        builder.assert_zero(
            selector.dup() * (result_u_den[lane].dup() - terms.u_numerator[lane].dup()),
        );
    }
}

/// `e = encoded^2 - A`, the substitution that turns the curve equation into
/// a quadratic in `x` for point decoding (see [`constrain_encoded_point_decode`]).
fn encoded_point_e_expr<AB>(encoded: &[AB::Expr; FP5_LIMBS]) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let mut e = fp5_square_expr::<AB>(encoded);
    for lane in 0..FP5_LIMBS {
        e[lane] -= AB::Expr::from_u64(ECGFP5_A[lane]);
    }
    e
}

/// `delta = e^2 - 4B`, the discriminant of `x^2 - e*x + B = 0`.
fn encoded_point_delta_expr<AB>(e: &[AB::Expr; FP5_LIMBS]) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let mut delta = fp5_square_expr::<AB>(e);
    for lane in 0..FP5_LIMBS {
        delta[lane] -= AB::Expr::from_u64(ECGFP5_B_MUL4[lane]);
    }
    delta
}

/// In-circuit restatement of [`crate::affine::affine_add_terms`]; must stay
/// bit-for-bit equivalent to it.
fn affine_add_terms_expr<AB>(
    lhs_x: &[AB::Expr; FP5_LIMBS],
    lhs_u: &[AB::Expr; FP5_LIMBS],
    rhs_x: &[AB::Expr; FP5_LIMBS],
    rhs_u: &[AB::Expr; FP5_LIMBS],
) -> AffineOperationTermsExpr<AB::Expr>
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let t1 = fp5_mul_expr::<AB>(lhs_x, rhs_x);
    let t3 = fp5_mul_expr::<AB>(lhs_u, rhs_u);
    let t5 = fp5_add_expr::<AB>(lhs_x, rhs_x);
    let t6 = fp5_add_expr::<AB>(lhs_u, rhs_u);
    let t7 = fp5_add_const_expr::<AB>(&t1, ECGFP5_B);
    let t9 = fp5_mul_expr::<AB>(
        &t3,
        &fp5_add_expr::<AB>(
            &fp5_mul_const_expr::<AB>(&t5, ECGFP5_B_MUL2),
            &fp5_double_expr::<AB>(&t7),
        ),
    );
    let t10 = fp5_mul_expr::<AB>(
        &fp5_add_const_expr::<AB>(&fp5_double_expr::<AB>(&t3), [1, 0, 0, 0, 0]),
        &fp5_add_expr::<AB>(&t5, &t7),
    );

    AffineOperationTermsExpr {
        x_numerator: fp5_mul_const_expr::<AB>(&fp5_sub_expr::<AB>(&t10, &t7), ECGFP5_B),
        x_denominator: fp5_sub_expr::<AB>(&t7, &t9),
        u_numerator: fp5_mul_expr::<AB>(&t6, &fp5_sub_from_const_expr::<AB>(ECGFP5_B, &t1)),
        u_denominator: fp5_add_expr::<AB>(&t7, &t9),
    }
}

/// In-circuit restatement of [`crate::affine::affine_double_terms`]; must
/// stay bit-for-bit equivalent to it.
fn affine_double_terms_expr<AB>(
    point_x: &[AB::Expr; FP5_LIMBS],
    point_u: &[AB::Expr; FP5_LIMBS],
) -> AffineOperationTermsExpr<AB::Expr>
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let u2 = fp5_square_expr::<AB>(point_u);
    let w1 = fp5_sub_from_const_expr::<AB>(
        [1, 0, 0, 0, 0],
        &fp5_mul_expr::<AB>(
            &u2,
            &fp5_double_expr::<AB>(&fp5_add_const_expr::<AB>(point_x, [1, 0, 0, 0, 0])),
        ),
    );
    let x_denominator = fp5_square_expr::<AB>(&w1);
    let u_numerator = fp5_sub_expr::<AB>(
        &fp5_square_expr::<AB>(&fp5_add_expr::<AB>(&w1, point_u)),
        &fp5_add_expr::<AB>(&u2, &x_denominator),
    );
    let u_denominator = fp5_sub_from_const_expr::<AB>(
        [2, 0, 0, 0, 0],
        &fp5_add_expr::<AB>(&fp5_mul_const_scalar_expr::<AB>(&u2, 4), &x_denominator),
    );

    AffineOperationTermsExpr {
        x_numerator: fp5_mul_const_expr::<AB>(&u2, ECGFP5_B_MUL4),
        x_denominator,
        u_numerator,
        u_denominator,
    }
}

/// Dot product of one-hot `selectors` with `chunks`: picks out the chunk
/// the currently-selected EC window belongs to.
fn selected_chunk_expr<AB>(
    selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
    chunks: &[AB::Expr; SCALAR_U32_LIMBS],
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    selectors
        .iter()
        .zip(chunks)
        .fold(AB::Expr::ZERO, |acc, (selector, chunk)| {
            acc + selector.dup() * chunk.dup()
        })
}

/// Goldilocks-canonical-range check for the challenge digest's 5 limbs,
/// expressed as 10 32-bit chunks (`digest_u32_chunks[2*lane]` = low half,
/// `[2*lane+1]` = high half of limb `lane`; see `preprocessed.rs::split_digest_u32`).
///
/// A `(lo, hi)` pair represents a value `< Goldilocks::ORDER` (`2^64 - 2^32 +
/// 1`) for every `lo` *except* when `hi == 0xFFFFFFFF` (all ones), in which
/// case canonicality additionally requires `lo == 0`. Rather than adding a
/// dedicated range-check gadget, this constraint piggybacks on the EC rows'
/// existing per-window bit decomposition (`COL_REDUCTION_DIGEST_BITS`,
/// already needed to extract scalar-mul window bits) to track, one 4-bit
/// window at a time as `chunk_selectors` walks through the 10 chunks:
/// whether the current window's bits are all 1 (`digest_window_all_ones`),
/// and — only while the *high* half of a limb is selected
/// (`digest_hi_chunk_selector_from_selectors`) — whether every window seen
/// so far in this chunk has been all-ones (`COL_REDUCTION_DIGEST_HI_PREFIX_ACC`).
/// At the end of a high chunk, if that prefix is all-ones, the corresponding
/// low chunk (`selected_digest_lo_for_hi_expr`) is asserted zero — exactly
/// the non-canonical case above, ruled out.
fn constrain_digest_u32_range_row<AB>(
    builder: &mut AB,
    ec_active: AB::Expr,
    chunk_end: AB::Expr,
    bit_selectors: &[AB::Expr; EC_BIT_SELECTORS],
    chunk_selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
    row: &[AB::Var],
    digest_u32_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
) where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    for bit in 0..EC_WINDOW_BITS {
        let digest_bit: AB::Expr = row[COL_REDUCTION_DIGEST_BITS + bit].into();
        builder.assert_zero(ec_active.dup() * digest_bit.dup() * (digest_bit - AB::Expr::ONE));
    }
    let digest_window_all_ones: AB::Expr = row[COL_REDUCTION_DIGEST_WINDOW_ALL_ONES].into();
    let expected_window_all_ones = (0..EC_WINDOW_BITS).fold(AB::Expr::ONE, |acc, bit| {
        let digest_bit: AB::Expr = row[COL_REDUCTION_DIGEST_BITS + bit].into();
        acc * digest_bit
    });
    builder.assert_zero(ec_active.dup() * (digest_window_all_ones - expected_window_all_ones));

    let digest_chunk_after = digest_chunk_after_expr::<AB>(row, bit_selectors);
    let selected_digest_chunk = selected_chunk_expr::<AB>(chunk_selectors, digest_u32_chunks);
    builder.assert_zero(
        ec_active.dup() * chunk_end.dup() * (digest_chunk_after - selected_digest_chunk),
    );

    let hi_chunk = digest_hi_chunk_selector_from_selectors::<AB>(chunk_selectors);
    let window_zero = bit_selectors[0].dup();
    let prefix_acc: AB::Expr = row[COL_REDUCTION_DIGEST_HI_PREFIX_ACC].into();
    builder.assert_zero(ec_active.dup() * (AB::Expr::ONE - hi_chunk.dup()) * prefix_acc.dup());
    builder.assert_zero(ec_active.dup() * hi_chunk.dup() * window_zero * prefix_acc);

    let hi_prefix_after = digest_hi_prefix_after_expr::<AB>(row, bit_selectors, chunk_selectors);
    let selected_lo = selected_digest_lo_for_hi_expr::<AB>(chunk_selectors, digest_u32_chunks);
    builder.assert_zero(ec_active * chunk_end * hi_prefix_after * selected_lo);
}

/// 1 iff the currently-selected chunk is the high 32-bit half of one of the
/// 5 digest limbs (odd-indexed chunks; see [`constrain_digest_u32_range_row`]).
fn digest_hi_chunk_selector_from_selectors<AB>(
    selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    (0..LIGHTER_POSEIDON_DIGEST_WIDTH).fold(AB::Expr::ZERO, |acc, lane| {
        acc + selectors[2 * lane + 1].dup()
    })
}

/// When the selected chunk is a high half (see
/// [`digest_hi_chunk_selector_from_selectors`]), returns that same limb's low
/// chunk value; 0 otherwise. Used to assert the low chunk is zero whenever
/// the corresponding high chunk turns out to be all-ones.
fn selected_digest_lo_for_hi_expr<AB>(
    selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
    digest_u32_chunks: &[AB::Expr; SCALAR_U32_LIMBS],
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    (0..LIGHTER_POSEIDON_DIGEST_WIDTH).fold(AB::Expr::ZERO, |acc, lane| {
        acc + selectors[2 * lane + 1].dup() * digest_u32_chunks[2 * lane].dup()
    })
}

/// The digest chunk-accumulator value after folding in this window's bits
/// (mirrors `s_chunk_after`/`e_chunk_after` in [`constrain_ec_row`], but for
/// the digest range check rather than the scalar bits themselves).
fn digest_chunk_after_expr<AB>(
    row: &[AB::Var],
    bit_selectors: &[AB::Expr; EC_BIT_SELECTORS],
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let digest_chunk_acc: AB::Expr = row[COL_REDUCTION_DIGEST_CHUNK_ACC].into();
    let digest_bits: [AB::Expr; EC_WINDOW_BITS] =
        core::array::from_fn(|bit| row[COL_REDUCTION_DIGEST_BITS + bit].into());
    digest_chunk_acc + ec_window_bits_value_expr::<AB>(&digest_bits, bit_selectors)
}

/// The "all windows so far in this high chunk were all-ones" accumulator
/// value after this window, only meaningful while a high chunk is selected:
/// at the chunk's first window it's just this window's all-ones flag,
/// otherwise it ANDs the running prefix with this window's flag.
fn digest_hi_prefix_after_expr<AB>(
    row: &[AB::Var],
    bit_selectors: &[AB::Expr; EC_BIT_SELECTORS],
    chunk_selectors: &[AB::Expr; EC_CHUNK_SELECTORS],
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let hi_chunk = digest_hi_chunk_selector_from_selectors::<AB>(chunk_selectors);
    let window_zero = bit_selectors[0].dup();
    let digest_window_all_ones: AB::Expr = row[COL_REDUCTION_DIGEST_WINDOW_ALL_ONES].into();
    let prefix_acc: AB::Expr = row[COL_REDUCTION_DIGEST_HI_PREFIX_ACC].into();
    hi_chunk
        * digest_window_all_ones
        * (window_zero + (AB::Expr::ONE - bit_selectors[0].dup()) * prefix_acc)
}

/// Reconstructs the little-endian integer value of the currently-selected
/// 4-bit window from its individual bit columns: `sum_bit bits[bit] *
/// selector[window] * 2^(window*EC_WINDOW_BITS + bit)`. Since `bit_selectors`
/// is one-hot, this is 0 for every window except the selected one, whose
/// bits are weighted by their position within the full 32-bit chunk.
fn ec_window_bits_value_expr<AB>(
    bits: &[AB::Expr; EC_WINDOW_BITS],
    bit_selectors: &[AB::Expr; EC_BIT_SELECTORS],
) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    bit_selectors
        .iter()
        .enumerate()
        .fold(AB::Expr::ZERO, |acc, (window, selector)| {
            acc + (0..EC_WINDOW_BITS).fold(AB::Expr::ZERO, |window_acc, bit| {
                window_acc
                    + bits[bit].dup()
                        * selector.dup()
                        * AB::Expr::from_u64(1u64 << (window * EC_WINDOW_BITS + bit))
            })
        })
}
