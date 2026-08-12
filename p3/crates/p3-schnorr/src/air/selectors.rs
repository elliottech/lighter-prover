// Selector polynomials and "run one of several possible rounds" helpers
// shared by `eval.rs`'s constraints and `trace_generation.rs`'s witness
// computation, so both sides select the same round/window deterministically
// from `(COL_PHASE, preprocessed selectors)` without an AIR branch primitive.

/// The Poseidon2 sponge's initial-state lane values for the public-transcript
/// hash: lane 0 is the batch's signature count, lane 1 the domain separator,
/// the rest zero (mirrors [`crate::poseidon::initial_public_hash_state`], but
/// expressed as in-circuit values keyed by the public `count` input).
fn initial_state_lane<AB>(lane: usize, count: AB::PublicVar) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    match lane {
        0 => count.into(),
        1 => AB::Expr::from_u64(crate::poseidon::PUBLIC_HASH_DOMAIN),
        _ => AB::Expr::ZERO,
    }
}

// Each `phase_*_indicator` is the Lagrange basis polynomial for `COL_PHASE`
// at value `*` over the domain `{0,1,2,3}`: it evaluates to 1 when
// `phase == *` and 0 at the other three domain points, letting `eval.rs`
// gate a constraint on "we're in phase N" using only field arithmetic (no
// native equality/branch). The `PHASE_*_DENOMINATOR`/`PHASE_LAST_INVERSE_DENOMINATOR`
// constants (see `layout.rs`) are the precomputed inverse of each
// polynomial's normalizing product so this can be one multiplication chain
// rather than a field division.

/// 1 if `phase == 3` (the final reduction/decode phase), else 0.
fn last_phase_indicator<AB>(phase: AB::Expr) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    phase.dup()
        * (phase.dup() - AB::Expr::ONE)
        * (phase - AB::Expr::TWO)
        * AB::Expr::from_u64(PHASE_LAST_INVERSE_DENOMINATOR)
}

/// 1 if `phase == 0` (absorbing `reconstructed_r`), else 0.
fn phase_zero_indicator<AB>(phase: AB::Expr) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    (phase.dup() - AB::Expr::ONE)
        * (phase.dup() - AB::Expr::TWO)
        * (phase - AB::Expr::from_u64(3))
        * AB::Expr::from_u64(PHASE_ZERO_DENOMINATOR_INVERSE)
}

/// 1 if `phase == 1` (absorbing `e` and carrying the message tail), else 0.
fn phase_one_indicator<AB>(phase: AB::Expr) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    phase.dup()
        * (phase.dup() - AB::Expr::TWO)
        * (phase - AB::Expr::from_u64(3))
        * AB::Expr::from_u64(PHASE_ONE_DENOMINATOR_INVERSE)
}

/// 1 if `phase == 2` (absorbing the remaining reduction chunks), else 0.
fn phase_two_indicator<AB>(phase: AB::Expr) -> AB::Expr
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    phase.dup()
        * (phase.dup() - AB::Expr::ONE)
        * (phase - AB::Expr::from_u64(3))
        * AB::Expr::from_u64(PHASE_TWO_DENOMINATOR_INVERSE)
}

fn round_selector_col(round: usize) -> usize {
    debug_assert!(round < POSEIDON_ROUND_STATE_COUNT);
    PREPROCESSED_COL_ROUND_SELECTORS + round
}

/// Reads the preprocessed one-hot round selector for every Poseidon2 round,
/// for the current row: `selectors[r] == 1` iff this row represents round `r`.
fn round_selectors_from_preprocessed<AB>(row: &[AB::Var]) -> [AB::Expr; POSEIDON_ROUND_STATE_COUNT]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|round| row[round_selector_col(round)].into())
}

/// Reads the preprocessed one-hot selector for which window-bit-position
/// within the current 4-bit window this EC row represents.
fn ec_bit_selectors_from_preprocessed<AB>(row: &[AB::Var]) -> [AB::Expr; EC_BIT_SELECTORS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|bit| row[ec_bit_selector_col(bit)].into())
}

/// Reads the preprocessed one-hot selector for which 32-bit scalar chunk
/// (0..SCALAR_U32_LIMBS) this EC row's window belongs to.
fn ec_chunk_selectors_from_preprocessed<AB>(row: &[AB::Var]) -> [AB::Expr; EC_CHUNK_SELECTORS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|chunk| row[ec_chunk_selector_col(chunk)].into())
}

/// Computes what the public-transcript Poseidon2 state would be after
/// applying round number `round` (0 = initial linear layer with the block
/// freshly absorbed, 1..=POSEIDON_ROUND_STATE_COUNT-1 = the corresponding
/// full/partial round) to `state_before`. Used as one candidate in
/// [`selected_public_poseidon_round`]'s round-selection sum; never called
/// with a literal, fixed `round` outside that context, since which round is
/// "real" for a given row depends on the (witness-derived) selectors.
fn public_poseidon_round_step<R>(
    round: usize,
    state_before: &[R; POSEIDON_WIDTH],
    block: &[R; POSEIDON_RATE],
) -> [R; POSEIDON_WIDTH]
where
    R: PrimeCharacteristicRing + Clone,
{
    debug_assert!(round < POSEIDON_ROUND_STATE_COUNT);
    let mut state = core::array::from_fn(|lane| state_before[lane].clone());
    if round == 0 {
        state[..POSEIDON_RATE].clone_from_slice(&block[..POSEIDON_RATE]);
        lighter_external_initial_linear_layer(&mut state);
        return state;
    }

    let round = round - 1;
    if round < POSEIDON_EXTERNAL_INITIAL_ROUNDS {
        lighter_external_full_round(&mut state, &LIGHTER_EXTERNAL_CONSTANTS[round]);
    } else if round < POSEIDON_EXTERNAL_INITIAL_ROUNDS + POSEIDON_INTERNAL_ROUNDS {
        let rc = LIGHTER_INTERNAL_CONSTANTS[round - POSEIDON_EXTERNAL_INITIAL_ROUNDS];
        lighter_internal_partial_round(&mut state, rc);
    } else {
        let round = round - POSEIDON_EXTERNAL_INITIAL_ROUNDS - POSEIDON_INTERNAL_ROUNDS;
        lighter_external_full_round(
            &mut state,
            &LIGHTER_EXTERNAL_CONSTANTS[LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS + round],
        );
    }
    state
}

/// The public-transcript Poseidon2 state this row's selected round actually
/// produces: `sum_r selectors[r] * public_poseidon_round_step(r, ...)`. Since
/// `selectors` is the preprocessed (fixed, not witness-supplied) round table
/// and exactly one entry is 1 by construction (see `preprocessed.rs`), this
/// reduces to the single matching round's output — a one-hot dot product
/// standing in for a `match` the AIR can't express natively.
fn selected_public_poseidon_round<AB>(
    selectors: &[AB::Expr; POSEIDON_ROUND_STATE_COUNT],
    state_before: &[AB::Expr; POSEIDON_WIDTH],
    block: &[AB::Expr; POSEIDON_RATE],
) -> [AB::Expr; POSEIDON_WIDTH]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing + Clone,
{
    let mut selected = core::array::from_fn(|_| AB::Expr::ZERO);
    for (round, selector) in selectors.iter().enumerate() {
        let candidate = public_poseidon_round_step(round, state_before, block);
        for lane in 0..POSEIDON_WIDTH {
            selected[lane] += selector.dup() * candidate[lane].dup();
        }
    }
    selected
}

/// Computes what the Lighter-challenge Poseidon2 state would be after round
/// `round`. At round 0, `overwrite_first_block` selects "absorb the full rate
/// block" (phase 0, absorbing `reconstructed_r`) and `overwrite_second_block`
/// selects "absorb only the first 2 lanes" (phase 1's partial absorption of
/// `e`'s leading chunk before the carried message tail is mixed in by
/// [`selected_lighter_poseidon_round`] instead). Both flags false reuses
/// `state_before` unchanged into the initial linear layer — used for
/// candidate rounds that aren't actually phase 0/1 on this row.
fn lighter_poseidon_round_step<R>(
    round: usize,
    state_before: &[R; LIGHTER_POSEIDON_WIDTH],
    block: &[R; LIGHTER_POSEIDON_RATE],
    overwrite_first_block: bool,
    overwrite_second_block: bool,
) -> [R; LIGHTER_POSEIDON_WIDTH]
where
    R: PrimeCharacteristicRing + Clone,
{
    debug_assert!(round < LIGHTER_POSEIDON_ROUND_STATE_COUNT);
    let mut state = core::array::from_fn(|lane| state_before[lane].clone());
    if round == 0 {
        if overwrite_first_block {
            state[..LIGHTER_POSEIDON_RATE].clone_from_slice(&block[..LIGHTER_POSEIDON_RATE]);
        } else if overwrite_second_block {
            state[..2].clone_from_slice(&block[..2]);
        }
        lighter_external_initial_linear_layer(&mut state);
        return state;
    }

    let round = round - 1;
    if round < LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS {
        lighter_external_full_round(&mut state, &LIGHTER_EXTERNAL_CONSTANTS[round]);
    } else if round < LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS + LIGHTER_INTERNAL_CONSTANTS.len() {
        let rc = LIGHTER_INTERNAL_CONSTANTS[round - LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS];
        lighter_internal_partial_round(&mut state, rc);
    } else {
        let round =
            round - LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS - LIGHTER_INTERNAL_CONSTANTS.len();
        lighter_external_full_round(
            &mut state,
            &LIGHTER_EXTERNAL_CONSTANTS[LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS + round],
        );
    }
    state
}

/// The Lighter-challenge Poseidon2 state this row's selected round actually
/// produces, mirroring [`selected_public_poseidon_round`]'s one-hot dot
/// product but with round 0 special-cased: since `challenge_first` and
/// `challenge_second` are mutually exclusive booleans (`eval.rs` asserts
/// `challenge_first * challenge_second == 0`), the lane overwrite
/// `candidate[lane] + (challenge_first (+ challenge_second if lane < 2)) *
/// (block[lane] - candidate[lane])` evaluates to "overwrite with `block`" when
/// the relevant flag is 1 and "keep `candidate`" when both are 0, letting one
/// expression cover phase 0's full-rate absorption and phase 1's lanes-0-1-only
/// absorption without branching on which phase is active.
fn selected_lighter_poseidon_round<AB>(
    selectors: &[AB::Expr; LIGHTER_POSEIDON_ROUND_STATE_COUNT],
    state_before: &[AB::Expr; LIGHTER_POSEIDON_WIDTH],
    block: &[AB::Expr; LIGHTER_POSEIDON_RATE],
    challenge_first: AB::Expr,
    challenge_second: AB::Expr,
) -> [AB::Expr; LIGHTER_POSEIDON_WIDTH]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing + Clone,
{
    let mut selected = core::array::from_fn(|_| AB::Expr::ZERO);
    for (round, selector) in selectors.iter().enumerate() {
        let mut candidate = core::array::from_fn(|lane| state_before[lane].clone());
        if round == 0 {
            for lane in 0..LIGHTER_POSEIDON_RATE {
                let overwrite = if lane < 2 {
                    challenge_first.dup() + challenge_second.dup()
                } else {
                    challenge_first.dup()
                };
                candidate[lane] =
                    candidate[lane].dup() + overwrite * (block[lane].dup() - candidate[lane].dup());
            }
            lighter_external_initial_linear_layer(&mut candidate);
        } else {
            candidate = lighter_poseidon_round_step(round, state_before, block, false, false);
        }
        for lane in 0..LIGHTER_POSEIDON_WIDTH {
            selected[lane] += selector.dup() * candidate[lane].dup();
        }
    }
    selected
}
