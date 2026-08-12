// In-circuit `GF(p^5)` arithmetic over `AB::Expr`, mirroring `crate::affine`'s
// `fp5_*` functions (which operate on concrete `Goldilocks` values) so
// `eval.rs`/`constraints.rs` can build the same field expressions out of
// trace-column expressions instead of constants. Keep this multiplication
// formula (`fp5_mul_expr`) and `affine::fp5_mul`'s in lockstep — both encode
// `GF(p^5) = GF(p)[w] / (w^5 - 3)` and must reduce identically.

/// Reads 5 consecutive columns starting at `start_col` as an `Fp5` expression.
fn fp5_from_row<AB>(row: &[AB::Var], start_col: usize) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| row[start_col + lane].into())
}

/// `lhs + rhs` in `GF(p^5)`, limb-wise.
fn fp5_add_expr<AB>(
    lhs: &[AB::Expr; FP5_LIMBS],
    rhs: &[AB::Expr; FP5_LIMBS],
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| lhs[lane].dup() + rhs[lane].dup())
}

/// `lhs - rhs` in `GF(p^5)`, limb-wise.
fn fp5_sub_expr<AB>(
    lhs: &[AB::Expr; FP5_LIMBS],
    rhs: &[AB::Expr; FP5_LIMBS],
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| lhs[lane].dup() - rhs[lane].dup())
}

/// `2 * value` in `GF(p^5)`.
fn fp5_double_expr<AB>(value: &[AB::Expr; FP5_LIMBS]) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| value[lane].dup() + value[lane].dup())
}

/// `value + constant` in `GF(p^5)`, where `constant` is a fixed (non-witness) `Fp5`.
fn fp5_add_const_expr<AB>(
    value: &[AB::Expr; FP5_LIMBS],
    constant: [u64; FP5_LIMBS],
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| value[lane].dup() + AB::Expr::from_u64(constant[lane]))
}

/// `constant - value` in `GF(p^5)`, where `constant` is a fixed `Fp5`.
fn fp5_sub_from_const_expr<AB>(
    constant: [u64; FP5_LIMBS],
    value: &[AB::Expr; FP5_LIMBS],
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| AB::Expr::from_u64(constant[lane]) - value[lane].dup())
}

/// `value * constant` in `GF(p^5)`, where `constant` is a fixed `Fp5`.
fn fp5_mul_const_expr<AB>(
    value: &[AB::Expr; FP5_LIMBS],
    constant: [u64; FP5_LIMBS],
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let constant = constant.map(AB::Expr::from_u64);
    fp5_mul_expr::<AB>(value, &constant)
}

/// Multiplies every limb of `value` by a fixed base-field `scalar`.
fn fp5_mul_const_scalar_expr<AB>(
    value: &[AB::Expr; FP5_LIMBS],
    scalar: u64,
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    core::array::from_fn(|lane| value[lane].dup() * AB::Expr::from_u64(scalar))
}

/// `value^2` in `GF(p^5)`.
fn fp5_square_expr<AB>(value: &[AB::Expr; FP5_LIMBS]) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    fp5_mul_expr::<AB>(value, value)
}

/// `GF(p^5)` multiplication, reduced modulo `w^5 - 3` — see [`crate::affine::fp5_mul`]
/// for the equivalent computation over concrete `Goldilocks` values, which
/// this must stay in lockstep with.
fn fp5_mul_expr<AB>(
    lhs: &[AB::Expr; FP5_LIMBS],
    rhs: &[AB::Expr; FP5_LIMBS],
) -> [AB::Expr; FP5_LIMBS]
where
    AB: AirBuilder<F = Goldilocks>,
    AB::Expr: PrimeCharacteristicRing,
{
    let w = AB::Expr::from_u64(3);
    [
        lhs[0].dup() * rhs[0].dup()
            + w.dup()
                * (lhs[1].dup() * rhs[4].dup()
                    + lhs[2].dup() * rhs[3].dup()
                    + lhs[3].dup() * rhs[2].dup()
                    + lhs[4].dup() * rhs[1].dup()),
        lhs[0].dup() * rhs[1].dup()
            + lhs[1].dup() * rhs[0].dup()
            + w.dup()
                * (lhs[2].dup() * rhs[4].dup()
                    + lhs[3].dup() * rhs[3].dup()
                    + lhs[4].dup() * rhs[2].dup()),
        lhs[0].dup() * rhs[2].dup()
            + lhs[1].dup() * rhs[1].dup()
            + lhs[2].dup() * rhs[0].dup()
            + w.dup() * (lhs[3].dup() * rhs[4].dup() + lhs[4].dup() * rhs[3].dup()),
        lhs[0].dup() * rhs[3].dup()
            + lhs[1].dup() * rhs[2].dup()
            + lhs[2].dup() * rhs[1].dup()
            + lhs[3].dup() * rhs[0].dup()
            + w * (lhs[4].dup() * rhs[4].dup()),
        lhs[0].dup() * rhs[4].dup()
            + lhs[1].dup() * rhs[3].dup()
            + lhs[2].dup() * rhs[2].dup()
            + lhs[3].dup() * rhs[1].dup()
            + lhs[4].dup() * rhs[0].dup(),
    ]
}

