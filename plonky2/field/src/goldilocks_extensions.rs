use alloc::vec::Vec;
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
use core::arch::asm;
use core::ops::Mul;

use static_assertions::const_assert;
use unroll::unroll_for_loops;

use crate::extension::quadratic::QuadraticExtension;
use crate::extension::quartic::QuarticExtension;
use crate::extension::quintic::{QuinticExtension, QuinticFirstCoeff};
use crate::extension::{Extendable, Frobenius};
use crate::goldilocks_field::{reduce160, GoldilocksField};
use crate::types::Field;

impl Frobenius<1> for GoldilocksField {}

impl Extendable<2> for GoldilocksField {
    type Extension = QuadraticExtension<Self>;

    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    const W: Self = Self(7);

    // DTH_ROOT = W^((ORDER - 1)/2)
    const DTH_ROOT: Self = Self(18446744069414584320);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 2] = [Self(0), Self(11713931119993638672)];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 2] = [Self(0), Self(7226896044987257365)];

    #[inline]
    fn extension_base_dot_product(
        extension_values: &[QuadraticExtension<Self>],
        base_scalars: &[Self],
    ) -> QuadraticExtension<Self> {
        ext2_base_scalar_dot_product(extension_values, base_scalars)
    }

    #[inline]
    fn extension_base_dot_products_2(
        extension_values: &[QuadraticExtension<Self>],
        base_polynomials: [&[Self]; 2],
    ) -> [QuadraticExtension<Self>; 2] {
        ext2_base_scalar_dot_products_2(extension_values, base_polynomials)
    }

    #[inline]
    fn extension_base_dot_product_with_subgroup_scales(
        extension_values: &[QuadraticExtension<Self>],
        base_coefficients: &[Self],
        subgroup_scales: &[Self],
    ) -> QuadraticExtension<Self> {
        ext2_base_scalar_dot_product_scaled(extension_values, base_coefficients, subgroup_scales)
    }

    #[inline(always)]
    fn mul_fft_quadratic_base_twiddle(twiddle: [Self; 2], value: [Self; 2]) -> [Self; 2] {
        // FFT rows below the quadratic extension's extra two-adic level
        // contain [w, 0], so scalar-multiply the two value limbs. Each limb
        // is one widening product instead of general ext2's four total.
        let [w, _] = twiddle;
        let [a0, a1] = value;
        [w * a0, w * a1]
    }

    #[inline(always)]
    fn fri_fold_arity16(
        terms: &[QuadraticExtension<Self>; 16],
        _beta: QuadraticExtension<Self>,
        beta_powers: &[QuadraticExtension<Self>; 16],
    ) -> QuadraticExtension<Self> {
        ext2_dot_product_arity16(terms, beta_powers)
    }
}

impl Mul for QuadraticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1]) = self;
        let Self([b0, b1]) = rhs;
        let c = ext2_mul([a0.0, a1.0], [b0.0, b1.0]);
        Self(c)
    }
}

impl Extendable<4> for GoldilocksField {
    type Extension = QuarticExtension<Self>;

    const W: Self = Self(7);

    // DTH_ROOT = W^((ORDER - 1)/4)
    const DTH_ROOT: Self = Self(281474976710656);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 4] =
        [Self(0), Self(8295451483910296135), Self(0), Self(0)];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 4] =
        [Self(0), Self(0), Self(0), Self(17216955519093520442)];
}

impl Mul for QuarticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = rhs;
        let c = ext4_mul([a0.0, a1.0, a2.0, a3.0], [b0.0, b1.0, b2.0, b3.0]);
        Self(c)
    }
}

impl Extendable<5> for GoldilocksField {
    type Extension = QuinticExtension<Self>;

    const W: Self = Self(3);

    // DTH_ROOT = W^((ORDER - 1)/5)
    const DTH_ROOT: Self = Self(1041288259238279555);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 5] = [
        Self(4624713872807171977),
        Self(381988216716071028),
        Self(14499722700050429911),
        Self(4870631734967222356),
        Self(4518902370426242880),
    ];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 5] = [
        Self::POWER_OF_TWO_GENERATOR,
        Self(0),
        Self(0),
        Self(0),
        Self(0),
    ];
}

impl Mul for QuinticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = self;
        let Self([b0, b1, b2, b3, b4]) = rhs;
        let c = ext5_mul(
            [a0.0, a1.0, a2.0, a3.0, a4.0],
            [b0.0, b1.0, b2.0, b3.0, b4.0],
        );
        Self(c)
    }
}

impl Frobenius<5> for QuinticExtension<GoldilocksField> {
    fn repeated_frobenius(&self, count: usize) -> Self {
        // The code below assumes DTH_ROOT = W^((p - 1)/5) = 1041288259238279555,
        // which has multiplicative order 5.
        const_assert!(<GoldilocksField as Extendable<5>>::DTH_ROOT.0 == 1041288259238279555u64);

        // FROB_COEFFS[c - 1][i - 1] = DTH_ROOT^(c * i mod 5), the coefficient of
        // limb i under the c-fold Frobenius automorphism.
        const FROB_COEFFS: [[GoldilocksField; 4]; 4] = [
            [
                GoldilocksField(1041288259238279555),
                GoldilocksField(15820824984080659046),
                GoldilocksField(211587555138949697),
                GoldilocksField(1373043270956696022),
            ],
            [
                GoldilocksField(15820824984080659046),
                GoldilocksField(1373043270956696022),
                GoldilocksField(1041288259238279555),
                GoldilocksField(211587555138949697),
            ],
            [
                GoldilocksField(211587555138949697),
                GoldilocksField(1041288259238279555),
                GoldilocksField(1373043270956696022),
                GoldilocksField(15820824984080659046),
            ],
            [
                GoldilocksField(1373043270956696022),
                GoldilocksField(211587555138949697),
                GoldilocksField(15820824984080659046),
                GoldilocksField(1041288259238279555),
            ],
        ];

        let count = count % 5;
        if count == 0 {
            return *self;
        }
        let z = &FROB_COEFFS[count - 1];
        let Self([a0, a1, a2, a3, a4]) = *self;
        Self([a0, a1 * z[0], a2 * z[1], a3 * z[2], a4 * z[3]])
    }
}

impl QuinticFirstCoeff<GoldilocksField> for QuinticExtension<GoldilocksField> {
    #[inline]
    fn mul_first_coeff(&self, rhs: &Self) -> GoldilocksField {
        let Self([a0, a1, a2, a3, a4]) = *self;
        let Self([b0, b1, b2, b3, b4]) = *rhs;
        ext5_add_prods0(
            &[a0.0, a1.0, a2.0, a3.0, a4.0],
            &[b0.0, b1.0, b2.0, b3.0, b4.0],
        )
    }
}

/*
 * The functions extD_add_prods[0-4] are helper functions for
 * computing products for extensions of degree D over the Goldilocks
 * field. They are faster than the generic method because all
 * reductions are delayed until the end which means only one per
 * result coefficient is necessary.
 */

/// Return `a`, `b` such that `a + b*2^128 = 3*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
#[inline(always)]
const fn u160_times_3(x: u128, y: u32) -> (u128, u32) {
    let (s, cy) = x.overflowing_add(x << 1);
    (s, 3 * y + (x >> 127) as u32 + cy as u32)
}

/// Return `a`, `b` such that `a + b*2^128 = 7*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
#[inline(always)]
const fn u160_times_7(x: u128, y: u32) -> (u128, u32) {
    let (d, br) = (x << 3).overflowing_sub(x);
    // NB: subtracting the borrow can't underflow
    (d, 7 * y + (x >> (128 - 3)) as u32 - br as u32)
}

/// Add one 64-by-64-bit product to a little-endian 160-bit accumulator.
/// Callers bound their term counts so the high limb cannot overflow.
#[inline(always)]
fn u160_add_product(lo: &mut u128, hi: &mut u32, a: u64, b: u64) {
    let (sum, carry) = lo.overflowing_add((a as u128) * (b as u128));
    *lo = sum;
    *hi += carry as u32;
}

/// Merge two exact 160-bit partial sums. Callers prove that the combined sum
/// is below `2^160`, so computing the high word through `u64` cannot truncate.
#[inline(always)]
fn u160_add_accumulators(a: (u128, u32), b: (u128, u32)) -> (u128, u32) {
    let (lo, carry) = a.0.overflowing_add(b.0);
    let hi = a.1 as u64 + b.1 as u64 + carry as u64;
    debug_assert!(hi <= u32::MAX as u64);
    (lo, hi as u32)
}

/// Compute `sum_i extension_values[i].scalar_mul(base_scalars[i])` in
/// GF(p^2), delaying reduction across the complete dot product.
///
/// The iterator-compatible result uses the shorter input length. Raw
/// Goldilocks limbs, including non-canonical representatives, are at most
/// `2^64 - 1`; therefore each limb product is at most `(2^64 - 1)^2`.
/// A chunk contains at most `2^32 - 1` terms, whose worst-case sum is
///
/// ```text
/// (2^32 - 1)(2^64 - 1)^2
///   = 2^160 - 2^128 - 2^97 + 2^65 + 2^32 - 1
///   < 2^160 - 2^128 + 2^96.
/// ```
///
/// This is exactly `reduce160`'s precondition. It also leaves the u32 high
/// accumulator below `2^32 - 1`, so `u160_add_product` cannot overflow it.
/// Inputs longer than one safe chunk are reduced chunk-wise; production
/// openings are many orders of magnitude smaller and take the one-reduction
/// path. The returned representative need not match reduce-per-term addition,
/// but it represents the same field element.
#[inline]
fn ext2_base_scalar_dot_product(
    extension_values: &[QuadraticExtension<GoldilocksField>],
    base_scalars: &[GoldilocksField],
) -> QuadraticExtension<GoldilocksField> {
    const MAX_TERMS_PER_REDUCTION: usize = u32::MAX as usize;

    let len = extension_values.len().min(base_scalars.len());
    if len == 0 {
        return QuadraticExtension::ZERO;
    }

    let reduce_chunk = |values: &[QuadraticExtension<GoldilocksField>],
                        scalars: &[GoldilocksField]| {
        debug_assert_eq!(values.len(), scalars.len());
        debug_assert!(values.len() <= MAX_TERMS_PER_REDUCTION);
        // Two accumulator banks break the loop-carried dependency between
        // adjacent products. Each bank receives alternating terms; merging
        // their exact 160-bit sums before the sole reduction preserves the
        // historical delayed-reduction representative.
        let (mut lo00, mut hi00) = (0u128, 0u32);
        let (mut lo01, mut hi01) = (0u128, 0u32);
        let (mut lo10, mut hi10) = (0u128, 0u32);
        let (mut lo11, mut hi11) = (0u128, 0u32);
        for (value_pair, scalar_pair) in values.chunks_exact(2).zip(scalars.chunks_exact(2)) {
            let QuadraticExtension([a00, a10]) = value_pair[0];
            let QuadraticExtension([a01, a11]) = value_pair[1];
            u160_add_product(&mut lo00, &mut hi00, a00.0, scalar_pair[0].0);
            u160_add_product(&mut lo10, &mut hi10, a10.0, scalar_pair[0].0);
            u160_add_product(&mut lo01, &mut hi01, a01.0, scalar_pair[1].0);
            u160_add_product(&mut lo11, &mut hi11, a11.0, scalar_pair[1].0);
        }
        if values.len() % 2 != 0 {
            let QuadraticExtension([a0, a1]) = values[values.len() - 1];
            let scalar = scalars[scalars.len() - 1];
            u160_add_product(&mut lo00, &mut hi00, a0.0, scalar.0);
            u160_add_product(&mut lo10, &mut hi10, a1.0, scalar.0);
        }
        let (lo0, hi0) = u160_add_accumulators((lo00, hi00), (lo01, hi01));
        let (lo1, hi1) = u160_add_accumulators((lo10, hi10), (lo11, hi11));
        // SAFETY: the exact worst-case bound above covers arbitrary u64
        // representatives for every term in this chunk.
        QuadraticExtension([unsafe { reduce160(lo0, hi0) }, unsafe {
            reduce160(lo1, hi1)
        }])
    };

    let first_end = len.min(MAX_TERMS_PER_REDUCTION);
    let mut result = reduce_chunk(&extension_values[..first_end], &base_scalars[..first_end]);
    let mut start = first_end;
    while start < len {
        let end = len.min(start + MAX_TERMS_PER_REDUCTION);
        result += reduce_chunk(&extension_values[start..end], &base_scalars[start..end]);
        start = end;
    }
    result
}

/// Compute two dot products that share one read of `extension_values`:
/// `out[k] = sum_i extension_values[i].scalar_mul(base_polynomials[k][i])`
/// for `k in 0..2`, delaying reduction across each complete dot.
///
/// Both openings in a production wave are committed at the same degree, so
/// the fused loop runs only when the two zipped lengths agree; a ragged pair
/// falls back to two independent [`ext2_base_scalar_dot_product`] calls and
/// is therefore bit-for-bit the historical path.
///
/// Raw-representative exactness of the fused loop: the single-dot form sums
/// its four 160-bit banks with `u160_add_accumulators` before reducing, so
/// its `reduce160` input is the *exact* integer `sum_i v_limb_i * c_i`.
/// This form accumulates the same terms of the same dot into one bank per
/// (output, limb) and therefore forms the same exact integer, so every
/// `reduce160` sees identical inputs and returns identical raw limbs. The
/// four accumulator chains (two outputs by two limbs) also keep the
/// single-dot version's four-way instruction-level parallelism while
/// halving the traffic through the shared powers table.
///
/// The per-accumulator bound is the single-dot bound verbatim: at most
/// `2^32 - 1` terms of `(2^64 - 1)^2` each, i.e. below
/// `2^160 - 2^128 + 2^96`, exactly `reduce160`'s precondition.
#[inline]
fn ext2_base_scalar_dot_products_2(
    extension_values: &[QuadraticExtension<GoldilocksField>],
    base_polynomials: [&[GoldilocksField]; 2],
) -> [QuadraticExtension<GoldilocksField>; 2] {
    const MAX_TERMS_PER_REDUCTION: usize = u32::MAX as usize;

    let len_a = extension_values.len().min(base_polynomials[0].len());
    let len_b = extension_values.len().min(base_polynomials[1].len());
    if len_a != len_b {
        return [
            ext2_base_scalar_dot_product(extension_values, base_polynomials[0]),
            ext2_base_scalar_dot_product(extension_values, base_polynomials[1]),
        ];
    }
    let len = len_a;
    if len == 0 {
        return [QuadraticExtension::ZERO, QuadraticExtension::ZERO];
    }

    let reduce_chunk = |values: &[QuadraticExtension<GoldilocksField>],
                        poly_a: &[GoldilocksField],
                        poly_b: &[GoldilocksField]| {
        debug_assert_eq!(values.len(), poly_a.len());
        debug_assert_eq!(values.len(), poly_b.len());
        debug_assert!(values.len() <= MAX_TERMS_PER_REDUCTION);
        let (mut a0_lo, mut a0_hi) = (0u128, 0u32);
        let (mut a1_lo, mut a1_hi) = (0u128, 0u32);
        let (mut b0_lo, mut b0_hi) = (0u128, 0u32);
        let (mut b1_lo, mut b1_hi) = (0u128, 0u32);
        for ((&QuadraticExtension([v0, v1]), &ca), &cb) in values.iter().zip(poly_a).zip(poly_b) {
            u160_add_product(&mut a0_lo, &mut a0_hi, v0.0, ca.0);
            u160_add_product(&mut a1_lo, &mut a1_hi, v1.0, ca.0);
            u160_add_product(&mut b0_lo, &mut b0_hi, v0.0, cb.0);
            u160_add_product(&mut b1_lo, &mut b1_hi, v1.0, cb.0);
        }
        // SAFETY: the exact worst-case bound documented above covers arbitrary
        // u64 representatives for every term in this chunk.
        [
            QuadraticExtension([unsafe { reduce160(a0_lo, a0_hi) }, unsafe {
                reduce160(a1_lo, a1_hi)
            }]),
            QuadraticExtension([unsafe { reduce160(b0_lo, b0_hi) }, unsafe {
                reduce160(b1_lo, b1_hi)
            }]),
        ]
    };

    let first_end = len.min(MAX_TERMS_PER_REDUCTION);
    let mut result = reduce_chunk(
        &extension_values[..first_end],
        &base_polynomials[0][..first_end],
        &base_polynomials[1][..first_end],
    );
    let mut start = first_end;
    while start < len {
        let end = len.min(start + MAX_TERMS_PER_REDUCTION);
        let [ta, tb] = reduce_chunk(
            &extension_values[start..end],
            &base_polynomials[0][start..end],
            &base_polynomials[1][start..end],
        );
        result[0] += ta;
        result[1] += tb;
        start = end;
    }
    result
}

/// Shifted-opening dot product
/// `sum_i extension_values[i].scalar_mul(base_coefficients[i] *
/// subgroup_scales[i])` in GF(p^2), delaying reduction across the complete
/// dot. Evaluates a base polynomial at `g·ζ` from the existing `ζ` powers
/// and the natural-order subgroup powers without materializing a second
/// `degree`-long quadratic-extension table.
///
/// The base product `coefficient * scale` is an ordinary Goldilocks
/// multiplication, so every widened limb product is at most `(2^64 - 1)^2`
/// and the chunk bound of [`ext2_base_scalar_dot_product`] applies verbatim
/// (up to `2^32 - 1` terms per reduction).
///
/// NOT raw-representative-identical to the materialized-table path: this
/// sums `(c_i·g^i)·ζ^i` where the table path sums `c_i·(ζ^i·g^i)`, and the
/// two exact 160-bit integers differ even though they are congruent mod p.
/// Every consumer of an opening — the sponge, the challenge derivation and
/// the serializer — is congruence-preserving, and the caller keeps the
/// materialized path one environment variable away
/// (`LIGHTER_GZETA_TABLE_ASSERT`).
#[inline]
fn ext2_base_scalar_dot_product_scaled(
    extension_values: &[QuadraticExtension<GoldilocksField>],
    base_coefficients: &[GoldilocksField],
    subgroup_scales: &[GoldilocksField],
) -> QuadraticExtension<GoldilocksField> {
    const MAX_TERMS_PER_REDUCTION: usize = u32::MAX as usize;

    let len = extension_values
        .len()
        .min(base_coefficients.len())
        .min(subgroup_scales.len());
    if len == 0 {
        return QuadraticExtension::ZERO;
    }

    let reduce_chunk = |values: &[QuadraticExtension<GoldilocksField>],
                        coefficients: &[GoldilocksField],
                        scales: &[GoldilocksField]| {
        debug_assert_eq!(values.len(), coefficients.len());
        debug_assert_eq!(values.len(), scales.len());
        debug_assert!(values.len() <= MAX_TERMS_PER_REDUCTION);
        // Two banks per limb, alternating terms, merged exactly before the
        // sole reduction — the same shape as `ext2_base_scalar_dot_product`.
        let (mut lo00, mut hi00) = (0u128, 0u32);
        let (mut lo01, mut hi01) = (0u128, 0u32);
        let (mut lo10, mut hi10) = (0u128, 0u32);
        let (mut lo11, mut hi11) = (0u128, 0u32);
        let mut index = 0;
        while index + 1 < values.len() {
            let QuadraticExtension([a00, a10]) = values[index];
            let QuadraticExtension([a01, a11]) = values[index + 1];
            let scalar_0 = coefficients[index] * scales[index];
            let scalar_1 = coefficients[index + 1] * scales[index + 1];
            u160_add_product(&mut lo00, &mut hi00, a00.0, scalar_0.0);
            u160_add_product(&mut lo10, &mut hi10, a10.0, scalar_0.0);
            u160_add_product(&mut lo01, &mut hi01, a01.0, scalar_1.0);
            u160_add_product(&mut lo11, &mut hi11, a11.0, scalar_1.0);
            index += 2;
        }
        if index < values.len() {
            let QuadraticExtension([a0, a1]) = values[index];
            let scalar = coefficients[index] * scales[index];
            u160_add_product(&mut lo00, &mut hi00, a0.0, scalar.0);
            u160_add_product(&mut lo10, &mut hi10, a1.0, scalar.0);
        }
        let (lo0, hi0) = u160_add_accumulators((lo00, hi00), (lo01, hi01));
        let (lo1, hi1) = u160_add_accumulators((lo10, hi10), (lo11, hi11));
        // SAFETY: the same exact worst-case bound as
        // `ext2_base_scalar_dot_product` covers arbitrary u64 representatives
        // for every term in this chunk.
        QuadraticExtension([unsafe { reduce160(lo0, hi0) }, unsafe {
            reduce160(lo1, hi1)
        }])
    };

    let first_end = len.min(MAX_TERMS_PER_REDUCTION);
    let mut result = reduce_chunk(
        &extension_values[..first_end],
        &base_coefficients[..first_end],
        &subgroup_scales[..first_end],
    );
    let mut start = first_end;
    while start < len {
        let end = len.min(start + MAX_TERMS_PER_REDUCTION);
        result += reduce_chunk(
            &extension_values[start..end],
            &base_coefficients[start..end],
            &subgroup_scales[start..end],
        );
        start = end;
    }
    result
}

/// Fused `a * b + c` for the Goldilocks quadratic extension: the addend is
/// folded into the multiply's 160-bit accumulators so the whole
/// multiply-accumulate costs exactly two `reduce160` calls. Spelled
/// separately, `a * b + c` performs a delayed `ext2` multiply (two
/// `reduce160`) and then a canonicalizing extension addition (two more
/// reductions), i.e. four per step.
///
/// Field-exact, not representative-exact: the separate spelling's extension
/// `Add` returns `x + y` verbatim whenever that sum stays below `2^64`, so it
/// can yield a representative in `[p, 2^64)` where the single `reduce160`
/// here yields the canonical one. The two never differ by more than one `p`,
/// which is what `ext2_mul_add_matches_mul_then_add_as_field_values` asserts
/// over the full boundary cross-product and 200k random triples. Producing
/// noncanonical representatives is normal for this field — `Add` does it for
/// roughly half of all canonical operand pairs — and every consumer of a FRI
/// coefficient is congruence-preserving.
///
/// Bounds: each 64x64 product is below `2^128`, the addend limbs are below
/// `2^64`, and the `7·(a1·b1)` term is below `7·2^128`; the combined `c0`
/// accumulator is below `10·2^128` and `c1` below `3·2^128`, both far under
/// `reduce160`'s `2^160 - 2^128 + 2^96` precondition.
#[inline(always)]
pub fn ext2_mul_add(
    a: QuadraticExtension<GoldilocksField>,
    b: QuadraticExtension<GoldilocksField>,
    c: QuadraticExtension<GoldilocksField>,
) -> QuadraticExtension<GoldilocksField> {
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);
    let QuadraticExtension([a0, a1]) = a;
    let QuadraticExtension([b0, b1]) = b;
    let QuadraticExtension([c0, c1]) = c;

    let (mut c0_plain_lo, mut c0_plain_hi) = (0u128, 0u32);
    let (mut c0_w_lo, mut c0_w_hi) = (0u128, 0u32);
    let (mut c1_lo, mut c1_hi) = (0u128, 0u32);

    u160_add_product(&mut c0_plain_lo, &mut c0_plain_hi, a0.0, b0.0);
    u160_add_product(&mut c0_w_lo, &mut c0_w_hi, a1.0, b1.0);
    u160_add_product(&mut c1_lo, &mut c1_hi, a0.0, b1.0);
    u160_add_product(&mut c1_lo, &mut c1_hi, a1.0, b0.0);

    let (sum, carry) = c0_plain_lo.overflowing_add(c0.0 as u128);
    c0_plain_lo = sum;
    c0_plain_hi += carry as u32;
    let (sum, carry) = c1_lo.overflowing_add(c1.0 as u128);
    c1_lo = sum;
    c1_hi += carry as u32;

    let (c0_w_lo, c0_w_hi) = u160_times_7(c0_w_lo, c0_w_hi);
    let (c0_lo, carry) = c0_plain_lo.overflowing_add(c0_w_lo);
    let c0_hi = c0_plain_hi + c0_w_hi + carry as u32;

    // SAFETY: the bounds documented above are far below reduce160's
    // `2^160 - 2^128 + 2^96` precondition.
    let c0 = unsafe { reduce160(c0_lo, c0_hi) };
    let c1 = unsafe { reduce160(c1_lo, c1_hi) };
    QuadraticExtension([c0, c1])
}

/// Compute `sum_i terms[i] * powers[i]` in GF(p^2), delaying reduction
/// across the complete production FRI arity. For raw limbs below 2^64,
///
/// - c0 < 16 * (1 + 7) * 2^128 = 2^135;
/// - c1 < 16 * 2 * 2^128 = 2^133.
///
/// Both coefficients therefore satisfy `reduce160`'s bound with ample room.
#[inline(always)]
#[unroll_for_loops]
fn ext2_dot_product_arity16(
    terms: &[QuadraticExtension<GoldilocksField>; 16],
    powers: &[QuadraticExtension<GoldilocksField>; 16],
) -> QuadraticExtension<GoldilocksField> {
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);

    let (mut c0_plain_lo, mut c0_plain_hi) = (0u128, 0u32);
    let (mut c0_w_lo, mut c0_w_hi) = (0u128, 0u32);
    let (mut c1_lo, mut c1_hi) = (0u128, 0u32);

    for i in 0..16 {
        let QuadraticExtension([a0, a1]) = terms[i];
        let QuadraticExtension([b0, b1]) = powers[i];
        u160_add_product(&mut c0_plain_lo, &mut c0_plain_hi, a0.0, b0.0);
        u160_add_product(&mut c0_w_lo, &mut c0_w_hi, a1.0, b1.0);
        u160_add_product(&mut c1_lo, &mut c1_hi, a0.0, b1.0);
        u160_add_product(&mut c1_lo, &mut c1_hi, a1.0, b0.0);
    }

    let (c0_w_lo, c0_w_hi) = u160_times_7(c0_w_lo, c0_w_hi);
    let (c0_lo, carry) = c0_plain_lo.overflowing_add(c0_w_lo);
    let c0_hi = c0_plain_hi + c0_w_hi + carry as u32;

    // SAFETY: the bounds documented above are far below reduce160's
    // `2^160 - 2^128 + 2^96` precondition.
    let c0 = unsafe { reduce160(c0_lo, c0_hi) };
    let c1 = unsafe { reduce160(c1_lo, c1_hi) };
    QuadraticExtension([c0, c1])
}

/// Add one product to each of two independent 160-bit accumulators. LLVM
/// cannot see adjacent FRI rows through Rayon's per-row closure; one Apple
/// AArch64 block makes their independent `mul`/`umulh` streams explicit while
/// accumulating the exact integers produced by two `u160_add_product` calls.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(always)]
fn u160_add_product_pair(
    lo0: &mut u128,
    hi0: &mut u32,
    a0: u64,
    b0: u64,
    lo1: &mut u128,
    hi1: &mut u32,
    a1: u64,
    b1: u64,
) {
    let mut acc0_lo = *lo0 as u64;
    let mut acc0_mid = (*lo0 >> 64) as u64;
    let mut acc0_hi = *hi0 as u64;
    let mut acc1_lo = *lo1 as u64;
    let mut acc1_mid = (*lo1 >> 64) as u64;
    let mut acc1_hi = *hi1 as u64;
    let product0 = a0;
    let product1 = a1;

    unsafe {
        asm!(
            "umulh {product_hi0}, {product0}, {rhs0}",
            "umulh {product_hi1}, {product1}, {rhs1}",
            "mul   {product0}, {product0}, {rhs0}",
            "mul   {product1}, {product1}, {rhs1}",
            "adds  {acc0_lo}, {acc0_lo}, {product0}",
            "adc   {product_hi0}, {product_hi0}, xzr",
            "adds  {acc0_mid}, {acc0_mid}, {product_hi0}",
            "adc   {acc0_hi}, {acc0_hi}, xzr",
            "adds  {acc1_lo}, {acc1_lo}, {product1}",
            "adc   {product_hi1}, {product_hi1}, xzr",
            "adds  {acc1_mid}, {acc1_mid}, {product_hi1}",
            "adc   {acc1_hi}, {acc1_hi}, xzr",
            product0 = inout(reg) product0 => _,
            product1 = inout(reg) product1 => _,
            product_hi0 = out(reg) _,
            product_hi1 = out(reg) _,
            rhs0 = in(reg) b0,
            rhs1 = in(reg) b1,
            acc0_lo = inout(reg) acc0_lo,
            acc0_mid = inout(reg) acc0_mid,
            acc0_hi = inout(reg) acc0_hi,
            acc1_lo = inout(reg) acc1_lo,
            acc1_mid = inout(reg) acc1_mid,
            acc1_hi = inout(reg) acc1_hi,
            options(pure, nomem, nostack),
        );
    }

    debug_assert!(acc0_hi <= u32::MAX as u64);
    debug_assert!(acc1_hi <= u32::MAX as u64);
    *lo0 = acc0_lo as u128 | ((acc0_mid as u128) << 64);
    *hi0 = acc0_hi as u32;
    *lo1 = acc1_lo as u128 | ((acc1_mid as u128) << 64);
    *hi1 = acc1_hi as u32;
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(always)]
#[unroll_for_loops]
fn ext2_dot_product_arity16_pair(
    terms0: &[QuadraticExtension<GoldilocksField>; 16],
    terms1: &[QuadraticExtension<GoldilocksField>; 16],
    powers: &[QuadraticExtension<GoldilocksField>; 16],
) -> [QuadraticExtension<GoldilocksField>; 2] {
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);
    let (mut p0_lo, mut p0_hi) = (0u128, 0u32);
    let (mut w0_lo, mut w0_hi) = (0u128, 0u32);
    let (mut c1_0_lo, mut c1_0_hi) = (0u128, 0u32);
    let (mut p1_lo, mut p1_hi) = (0u128, 0u32);
    let (mut w1_lo, mut w1_hi) = (0u128, 0u32);
    let (mut c1_1_lo, mut c1_1_hi) = (0u128, 0u32);

    for i in 0..16 {
        let QuadraticExtension([a00, a01]) = terms0[i];
        let QuadraticExtension([a10, a11]) = terms1[i];
        let QuadraticExtension([b0, b1]) = powers[i];
        u160_add_product_pair(
            &mut p0_lo, &mut p0_hi, a00.0, b0.0, &mut p1_lo, &mut p1_hi, a10.0, b0.0,
        );
        u160_add_product_pair(
            &mut w0_lo, &mut w0_hi, a01.0, b1.0, &mut w1_lo, &mut w1_hi, a11.0, b1.0,
        );
        u160_add_product_pair(
            &mut c1_0_lo,
            &mut c1_0_hi,
            a00.0,
            b1.0,
            &mut c1_1_lo,
            &mut c1_1_hi,
            a10.0,
            b1.0,
        );
        u160_add_product_pair(
            &mut c1_0_lo,
            &mut c1_0_hi,
            a01.0,
            b0.0,
            &mut c1_1_lo,
            &mut c1_1_hi,
            a11.0,
            b0.0,
        );
    }

    let finish = |plain_lo: u128, plain_hi: u32, w_lo: u128, w_hi: u32, c1_lo: u128, c1_hi: u32| {
        let (w_lo, w_hi) = u160_times_7(w_lo, w_hi);
        let (c0_lo, carry) = plain_lo.overflowing_add(w_lo);
        let c0_hi = plain_hi + w_hi + carry as u32;
        // SAFETY: these are exactly the bounded sixteen-term sums accepted by
        // `ext2_dot_product_arity16`; only independent instruction order moved.
        QuadraticExtension([unsafe { reduce160(c0_lo, c0_hi) }, unsafe {
            reduce160(c1_lo, c1_hi)
        }])
    };
    [
        finish(p0_lo, p0_hi, w0_lo, w0_hi, c1_0_lo, c1_0_hi),
        finish(p1_lo, p1_hi, w1_lo, w1_hi, c1_1_lo, c1_1_hi),
    ]
}

/// Apple-AArch64 batch entry point for production arity-16 FRI folds. Row
/// pairs share beta-power loads and the odd tail retains the scalar kernel.
/// Results are raw-representative-identical to independent scalar folds.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline]
pub fn ext2_fri_fold_arity16_batch(
    terms: &[QuadraticExtension<GoldilocksField>],
    powers: &[QuadraticExtension<GoldilocksField>; 16],
    output: &mut [QuadraticExtension<GoldilocksField>],
) {
    assert_eq!(terms.len(), output.len() * 16);
    let paired = output.len() & !1;
    for row in (0..paired).step_by(2) {
        let start = row * 16;
        let terms0 = terms[start..start + 16].try_into().unwrap();
        let terms1 = terms[start + 16..start + 32].try_into().unwrap();
        let pair = ext2_dot_product_arity16_pair(terms0, terms1, powers);
        output[row] = pair[0];
        output[row + 1] = pair[1];
    }
    if paired != output.len() {
        let start = paired * 16;
        let tail = terms[start..start + 16].try_into().unwrap();
        output[paired] = ext2_dot_product_arity16(tail, powers);
    }
}

/// For each output slot `i`, compute
/// `out[i] = sum_j powers[j].scalar_mul(polys[j][start + i])`
/// over every polynomial long enough to reach that slot, delaying modular
/// reduction across the whole polynomial batch: one `reduce160` per extension
/// limb per slot instead of one `reduce128` per limb per *term* plus a
/// canonicalizing extension add per term.
///
/// Scalar multiplication by a base-field coefficient never mixes the two
/// extension limbs, so each limb is a plain dot product of 64-bit raw
/// representatives:
///
/// - `limb0 = sum_j powers[j].0[0] * c_j`
/// - `limb1 = sum_j powers[j].0[1] * c_j`
///
/// Each 64x64 product is below `2^128`, so after `n` terms the 160-bit
/// accumulator holds less than `n * 2^128`: its high limb stays below `n`,
/// and the exact worst-case calculation above shows that `reduce160`'s
/// `2^160 - 2^128 + 2^96` precondition holds through `n = 2^32 - 1`.
/// The asserted bound below is far stricter than that limit and covers every
/// production batch (a few hundred polynomials).
///
/// The result is the same field element as the reduce-per-term form; the raw
/// representative may differ (both forms produce sub-2^64 representatives
/// that later consumers treat value-wise, and proof serialization
/// canonicalizes every limb).
pub fn ext2_base_scalar_dot_slots(
    out: &mut [QuadraticExtension<GoldilocksField>],
    start: usize,
    polys: &[&[GoldilocksField]],
    powers: &[QuadraticExtension<GoldilocksField>],
) {
    assert_eq!(polys.len(), powers.len());
    assert!(polys.len() < 1 << 24);
    let end = start + out.len();
    // Split once so the dense inner loop over fully-covering polynomials
    // runs without per-slot bounds checks; only boundary-length polynomials
    // take the checked loop.
    let mut full: Vec<(&[GoldilocksField], QuadraticExtension<GoldilocksField>)> =
        Vec::with_capacity(polys.len());
    let mut partial: Vec<(&[GoldilocksField], QuadraticExtension<GoldilocksField>)> = Vec::new();
    for (&p, &pw) in polys.iter().zip(powers) {
        if p.len() >= end {
            full.push((&p[start..end], pw));
        } else if p.len() > start {
            partial.push((&p[start..], pw));
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        let (mut lo0, mut hi0) = (0u128, 0u32);
        let (mut lo1, mut hi1) = (0u128, 0u32);
        for &(p, QuadraticExtension([b0, b1])) in &full {
            // SAFETY: every slice in `full` has length exactly `out.len()`.
            let c = unsafe { p.get_unchecked(i).0 };
            u160_add_product(&mut lo0, &mut hi0, b0.0, c);
            u160_add_product(&mut lo1, &mut hi1, b1.0, c);
        }
        for &(p, QuadraticExtension([b0, b1])) in &partial {
            if i < p.len() {
                let c = p[i].0;
                u160_add_product(&mut lo0, &mut hi0, b0.0, c);
                u160_add_product(&mut lo1, &mut hi1, b1.0, c);
            }
        }
        // SAFETY: the accumulator bound documented above — below
        // `polys.len() * 2^128 < 2^152` — is far under reduce160's
        // precondition.
        *o = QuadraticExtension([unsafe { reduce160(lo0, hi0) }, unsafe {
            reduce160(lo1, hi1)
        }]);
    }
}

/*
 * Quadratic multiplication and squaring
 */

#[inline(always)]
fn ext2_add_prods0(a: &[u64; 2], b: &[u64; 2]) -> GoldilocksField {
    // Computes a0 * b0 + W * a1 * b1;
    let [a0, a1] = *a;
    let [b0, b1] = *b;

    let cy;

    // W * a1 * b1
    let (mut cumul_lo, mut cumul_hi) = u160_times_7((a1 as u128) * (b1 as u128), 0u32);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext2_add_prods1(a: &[u64; 2], b: &[u64; 2]) -> GoldilocksField {
    // Computes a0 * b1 + a1 * b0;
    let [a0, a1] = *a;
    let [b0, b1] = *b;

    let cy;

    // a0 * b1
    let mut cumul_lo = (a0 as u128) * (b1 as u128);

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    let cumul_hi = cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

/// Multiply a and b considered as elements of GF(p^2).
#[inline(always)]
pub(crate) fn ext2_mul(a: [u64; 2], b: [u64; 2]) -> [GoldilocksField; 2] {
    // The code in ext2_add_prods[01] assumes the quadratic extension
    // generator is 7.
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);

    let c0 = ext2_add_prods0(&a, &b);
    let c1 = ext2_add_prods1(&a, &b);
    [c0, c1]
}

/*
 * Quartic multiplication and squaring
 */

#[inline(always)]
fn ext4_add_prods0(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c0 = a0 * b0 + W * (a1 * b3 + a2 * b2 + a3 * b1)

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // a1 * b3
    let mut cumul_lo = (a1 as u128) * (b3 as u128);

    // a2 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b2 as u128));
    let mut cumul_hi = cy as u32;

    // a3 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_7(cumul_lo, cumul_hi);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext4_add_prods1(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c1 = a0 * b1 + a1 * b0 + W * (a2 * b3 + a3 * b2);

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // a2 * b3
    let mut cumul_lo = (a2 as u128) * (b3 as u128);

    // a3 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b2 as u128));
    let mut cumul_hi = cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_7(cumul_lo, cumul_hi);

    // a0 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext4_add_prods2(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c2 = a0 * b2 + a1 * b1 + a2 * b0 + W * a3 * b3;

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // W * a3 * b3
    let (mut cumul_lo, mut cumul_hi) = u160_times_7((a3 as u128) * (b3 as u128), 0u32);

    // a0 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a1 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a2 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext4_add_prods3(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // a0 * b3
    let mut cumul_lo = (a0 as u128) * (b3 as u128);

    // a1 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b2 as u128));
    let mut cumul_hi = cy as u32;

    // a2 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a3 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

/// Multiply a and b considered as elements of GF(p^4).
#[inline(always)]
pub(crate) fn ext4_mul(a: [u64; 4], b: [u64; 4]) -> [GoldilocksField; 4] {
    // The code in ext4_add_prods[0-3] assumes the quartic extension
    // generator is 7.
    const_assert!(<GoldilocksField as Extendable<4>>::W.0 == 7u64);

    let c0 = ext4_add_prods0(&a, &b);
    let c1 = ext4_add_prods1(&a, &b);
    let c2 = ext4_add_prods2(&a, &b);
    let c3 = ext4_add_prods3(&a, &b);
    [c0, c1, c2, c3]
}

/*
 * Quintic multiplication and squaring
 */

#[inline(always)]
fn ext5_add_prods0(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c0 = a0 * b0 + W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1)

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a1 * b4
    let mut cumul_lo = (a1 as u128) * (b4 as u128);

    // a2 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // a3 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a4 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods1(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c1 = a0 * b1 + a1 * b0 + W * (a2 * b4 + a3 * b3 + a4 * b2);

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a2 * b4
    let mut cumul_lo = (a2 as u128) * (b4 as u128);

    // a3 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // a4 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

    // a0 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods2(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c2 = a0 * b2 + a1 * b1 + a2 * b0 + W * (a3 * b4 + a4 * b3);

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a3 * b4
    let mut cumul_lo = (a3 as u128) * (b4 as u128);

    // a4 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

    // a0 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a1 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a2 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods3(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + W * a4 * b4;

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // W * a4 * b4
    let (mut cumul_lo, mut cumul_hi) = u160_times_3((a4 as u128) * (b4 as u128), 0u32);

    // a0 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b3 as u128));
    cumul_hi += cy as u32;

    // a1 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a2 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a3 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods4(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a0 * b4
    let mut cumul_lo = (a0 as u128) * (b4 as u128);

    // a1 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // a2 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a3 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a4 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

/// Multiply a and b considered as elements of GF(p^5).
#[inline(always)]
pub(crate) fn ext5_mul(a: [u64; 5], b: [u64; 5]) -> [GoldilocksField; 5] {
    // The code in ext5_add_prods[0-4] assumes the quintic extension
    // generator is 3.
    const_assert!(<GoldilocksField as Extendable<5>>::W.0 == 3u64);

    let c0 = ext5_add_prods0(&a, &b);
    let c1 = ext5_add_prods1(&a, &b);
    let c2 = ext5_add_prods2(&a, &b);
    let c3 = ext5_add_prods3(&a, &b);
    let c4 = ext5_add_prods4(&a, &b);
    [c0, c1, c2, c3, c4]
}

#[cfg(test)]
mod dot_slot_coverage_tests {
    use super::*;

    /// Q4: the composition accumulator is allocated uninitialized because
    /// `ext2_base_scalar_dot_slots` assigns *every* output slot. Poison the
    /// destination first and check nothing survives, over the ragged shape
    /// that is the only way a slot could be skipped: polynomials that stop
    /// before the slot range, inside it, and after it — including a range
    /// past the end of every polynomial, where the answer is exactly zero.
    #[test]
    fn dot_slots_assigns_every_output_slot() {
        const POISON: QuadraticExtension<GoldilocksField> =
            QuadraticExtension([GoldilocksField(u64::MAX), GoldilocksField(u64::MAX)]);
        let lengths = [0usize, 1, 7, 8, 9, 31, 64];
        let polys: Vec<Vec<GoldilocksField>> = lengths
            .iter()
            .map(|&len| {
                (0..len)
                    .map(|i| GoldilocksField::from_canonical_u64(i as u64 * 7 + 1))
                    .collect()
            })
            .collect();
        let slices: Vec<&[GoldilocksField]> = polys.iter().map(|p| p.as_slice()).collect();
        let powers: Vec<QuadraticExtension<GoldilocksField>> = (0..lengths.len())
            .map(|i| {
                QuadraticExtension([
                    GoldilocksField::from_canonical_u64(i as u64 + 3),
                    GoldilocksField::from_canonical_u64(i as u64 * 5 + 11),
                ])
            })
            .collect();
        for (start, len) in [(0usize, 8usize), (4, 8), (60, 8), (64, 8), (96, 8), (0, 96)] {
            let mut out = vec![POISON; len];
            ext2_base_scalar_dot_slots(&mut out, start, &slices, &powers);
            for (i, o) in out.iter().enumerate() {
                assert!(
                    o.0[0].0 != POISON.0[0].0 || o.0[1].0 != POISON.0[1].0,
                    "slot {i} of range {start}..{} was never written",
                    start + len
                );
            }
            if start >= *lengths.iter().max().unwrap() {
                for o in &out {
                    assert_eq!(o.0[0].0, 0);
                    assert_eq!(o.0[1].0, 0);
                }
            }
        }
    }

    /// Sabotage control: a writer that skips the all-zero slots — the exact
    /// failure the uninitialized allocation would expose — must be caught by
    /// the poison sweep above.
    #[test]
    fn poison_sweep_catches_a_skipped_slot() {
        const POISON: QuadraticExtension<GoldilocksField> =
            QuadraticExtension([GoldilocksField(u64::MAX), GoldilocksField(u64::MAX)]);
        let mut out = vec![POISON; 4];
        // Stand-in for a writer that leaves untouched whatever it has no
        // term for.
        for (i, o) in out.iter_mut().enumerate() {
            if i % 2 == 0 {
                *o = QuadraticExtension([GoldilocksField(0), GoldilocksField(0)]);
            }
        }
        assert!(
            out.iter()
                .any(|o| o.0[0].0 == POISON.0[0].0 && o.0[1].0 == POISON.0[1].0),
            "poison sweep failed to notice a skipped slot"
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::extension::quadratic::QuadraticExtension;
    use crate::extension::quartic::QuarticExtension;
    use crate::extension::quintic::{QuinticExtension, QuinticFirstCoeff};
    use crate::extension::{Extendable, FieldExtension, Frobenius, OEF};
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Field64, PrimeField64};

    type GF = GoldilocksField;
    type Q2 = QuadraticExtension<GoldilocksField>;
    type Q4 = QuarticExtension<GoldilocksField>;
    type QE = QuinticExtension<GoldilocksField>;

    fn generic_extension_base_dot_product(values: &[Q2], scalars: &[GF]) -> Q2 {
        values
            .iter()
            .zip(scalars)
            .map(|(&value, &scalar)| <Q2 as FieldExtension<2>>::scalar_mul(&value, scalar))
            .sum()
    }

    fn single_bank_extension_base_dot_product(values: &[Q2], scalars: &[GF]) -> Q2 {
        let mut lo0 = 0u128;
        let mut hi0 = 0u32;
        let mut lo1 = 0u128;
        let mut hi1 = 0u32;
        for (&QuadraticExtension([a0, a1]), &scalar) in values.iter().zip(scalars) {
            super::u160_add_product(&mut lo0, &mut hi0, a0.0, scalar.0);
            super::u160_add_product(&mut lo1, &mut hi1, a1.0, scalar.0);
        }
        QuadraticExtension([
            unsafe { crate::goldilocks_field::reduce160(lo0, hi0) },
            unsafe { crate::goldilocks_field::reduce160(lo1, hi1) },
        ])
    }

    #[test]
    fn extension_base_dot_product_default_matches_scalar_mul_sum() {
        let values: Vec<Q4> = (0..17)
            .map(|i| {
                QuarticExtension(core::array::from_fn(|limb| {
                    GoldilocksField(
                        (i as u64 + 1)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15u64.rotate_left(limb as u32)),
                    )
                }))
            })
            .collect();
        let scalars: Vec<GF> = (0..16)
            .map(|i| GoldilocksField((i as u64).wrapping_mul(u64::MAX - 1)))
            .collect();
        let expected: Q4 = values
            .iter()
            .zip(&scalars)
            .map(|(&value, &scalar)| <Q4 as FieldExtension<4>>::scalar_mul(&value, scalar))
            .sum();
        let actual = <GF as Extendable<4>>::extension_base_dot_product(&values, &scalars);
        assert_eq!(actual, expected);
        assert_eq!(
            <GF as Extendable<4>>::extension_base_dot_product(&[], &scalars),
            Q4::ZERO
        );
    }

    #[test]
    fn ext2_extension_base_dot_product_matches_generic_at_boundaries() {
        let p = GF::ORDER;
        let raw_specials = [0, 1, 2, p - 1, p, p + 1, u64::MAX];
        // Zero/one, unequal zip lengths, SIMD/cache-sized powers of two, and
        // the neighboring lengths most likely to expose loop-tail mistakes.
        let lengths = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 1),
            (15, 15),
            (16, 16),
            (17, 17),
            (63, 64),
            (64, 63),
            (65, 65),
            (255, 255),
            (256, 256),
            (257, 257),
            (2047, 2047),
            (2048, 2048),
            (2049, 2049),
            (4095, 4096),
            (4096, 4095),
            (4097, 4097),
        ];

        let mut state = 0xA076_1D64_78BD_642Fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (values_len, scalars_len) in lengths {
            let values: Vec<Q2> = (0..values_len)
                .map(|i| {
                    let a0 = if i < raw_specials.len() {
                        raw_specials[i]
                    } else {
                        next()
                    };
                    let a1 = if i < raw_specials.len() {
                        raw_specials[raw_specials.len() - 1 - i]
                    } else {
                        next()
                    };
                    QuadraticExtension([GoldilocksField(a0), GoldilocksField(a1)])
                })
                .collect();
            let scalars: Vec<GF> = (0..scalars_len)
                .map(|i| {
                    GoldilocksField(if i < raw_specials.len() {
                        raw_specials[(i * 3) % raw_specials.len()]
                    } else {
                        next()
                    })
                })
                .collect();

            let expected = generic_extension_base_dot_product(&values, &scalars);
            let actual = <GF as Extendable<2>>::extension_base_dot_product(&values, &scalars);
            let raw_reference = single_bank_extension_base_dot_product(&values, &scalars);
            for limb in 0..2 {
                assert_eq!(
                    actual.0[limb].to_canonical_u64(),
                    expected.0[limb].to_canonical_u64(),
                    "canonical limb {limb} mismatch at ({values_len}, {scalars_len})"
                );
                assert_eq!(
                    actual.0[limb].0, raw_reference.0[limb].0,
                    "raw limb {limb} mismatch at ({values_len}, {scalars_len})"
                );
            }
        }
        // Raw representatives are deliberately not part of the assertion:
        // delayed and per-term reduction are required to agree as field
        // values, including when every input can occupy the full u64 range.
    }

    #[test]
    #[ignore = "release-only production-shape microbenchmark"]
    fn ext2_extension_base_dot_product_two_bank_benchmark() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const LEN: usize = 1 << 16;
        const REPEATS: usize = 32;
        const SAMPLES: usize = 41;
        let values: Vec<Q2> = (0..LEN)
            .map(|i| {
                QuadraticExtension([
                    GoldilocksField((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    GoldilocksField(
                        (i as u64)
                            .wrapping_add(1)
                            .wrapping_mul(0xD1B5_4A32_D192_ED03),
                    ),
                ])
            })
            .collect();
        let scalars: Vec<GF> = (0..LEN)
            .map(|i| GoldilocksField((i as u64).wrapping_mul(0xA076_1D64_78BD_642F)))
            .collect();
        let old = || single_bank_extension_base_dot_product(&values, &scalars);
        let new = || <GF as Extendable<2>>::extension_base_dot_product(&values, &scalars);
        assert_eq!(old(), new());
        for _ in 0..3 {
            black_box(old());
            black_box(new());
        }
        let measure = |f: &dyn Fn() -> Q2| {
            let start = Instant::now();
            for _ in 0..REPEATS {
                black_box(f());
            }
            start.elapsed()
        };
        let mut old_samples = Vec::with_capacity(SAMPLES);
        let mut new_samples = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let (old_elapsed, new_elapsed) = if sample % 2 == 0 {
                (measure(&old), measure(&new))
            } else {
                let new_elapsed = measure(&new);
                let old_elapsed = measure(&old);
                (old_elapsed, new_elapsed)
            };
            old_samples.push(old_elapsed);
            new_samples.push(new_elapsed);
        }
        old_samples.sort_unstable();
        new_samples.sort_unstable();
        let old_median: Duration = old_samples[SAMPLES / 2];
        let new_median: Duration = new_samples[SAMPLES / 2];
        eprintln!(
            "single_bank={old_median:?} two_bank={new_median:?} speedup={:.4}x",
            old_median.as_secs_f64() / new_median.as_secs_f64()
        );
    }

    /// Raw-limb differential: the paired dot must return the byte-identical
    /// noncanonical representative that two independent single dots return,
    /// and the same field value the reduce-per-term reference returns, at
    /// every length boundary and for adversarial raw limbs (0, 1, p-1, p,
    /// p+1, u64::MAX) in both the values and both coefficient slices.
    #[test]
    fn ext2_extension_base_dot_products_2_matches_two_singles_raw() {
        let p = GF::ORDER;
        let raw_specials = [0, 1, 2, p - 1, p, p + 1, u64::MAX];
        let lengths = [
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025,
            4095, 4096, 4097,
        ];

        let mut state = 0x0F1E_2D3C_4B5A_6978u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in lengths {
            let values: Vec<Q2> = (0..len)
                .map(|i| {
                    let a0 = if i < raw_specials.len() {
                        raw_specials[i]
                    } else {
                        next()
                    };
                    let a1 = if i < raw_specials.len() {
                        raw_specials[raw_specials.len() - 1 - i]
                    } else {
                        next()
                    };
                    QuadraticExtension([GoldilocksField(a0), GoldilocksField(a1)])
                })
                .collect();
            let poly_a: Vec<GF> = (0..len)
                .map(|i| {
                    GoldilocksField(if i < raw_specials.len() {
                        raw_specials[(i * 7) % raw_specials.len()]
                    } else {
                        next()
                    })
                })
                .collect();
            let poly_b: Vec<GF> = (0..len)
                .map(|i| {
                    GoldilocksField(if i < raw_specials.len() {
                        raw_specials[(i * 5 + 1) % raw_specials.len()]
                    } else {
                        next()
                    })
                })
                .collect();

            let single = [
                <GF as Extendable<2>>::extension_base_dot_product(&values, &poly_a),
                <GF as Extendable<2>>::extension_base_dot_product(&values, &poly_b),
            ];
            let generic = [
                generic_extension_base_dot_product(&values, &poly_a),
                generic_extension_base_dot_product(&values, &poly_b),
            ];
            let paired =
                <GF as Extendable<2>>::extension_base_dot_products_2(&values, [&poly_a, &poly_b]);
            for side in 0..2 {
                for limb in 0..2 {
                    assert_eq!(
                        paired[side].0[limb].0, single[side].0[limb].0,
                        "raw limb {limb} side {side} at len {len}"
                    );
                    assert_eq!(
                        paired[side].0[limb].to_canonical_u64(),
                        generic[side].0[limb].to_canonical_u64(),
                        "canonical limb {limb} side {side} at len {len}"
                    );
                }
            }
        }
    }

    /// Ragged pairs are never produced by the opening waves, but the fallback
    /// they take must still be the two independent single dots exactly.
    #[test]
    fn ext2_extension_base_dot_products_2_ragged_falls_back_exactly() {
        let p = GF::ORDER;
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (values_len, a_len, b_len) in [
            (0usize, 0usize, 1usize),
            (1, 0, 1),
            (4, 3, 4),
            (17, 17, 5),
            (64, 65, 63),
            (256, 255, 257),
        ] {
            let values: Vec<Q2> = (0..values_len)
                .map(|_| QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]))
                .collect();
            let poly_a: Vec<GF> = (0..a_len).map(|_| GoldilocksField(next() % p)).collect();
            let poly_b: Vec<GF> = (0..b_len).map(|_| GoldilocksField(next())).collect();
            let single = [
                <GF as Extendable<2>>::extension_base_dot_product(&values, &poly_a),
                <GF as Extendable<2>>::extension_base_dot_product(&values, &poly_b),
            ];
            let paired =
                <GF as Extendable<2>>::extension_base_dot_products_2(&values, [&poly_a, &poly_b]);
            for side in 0..2 {
                for limb in 0..2 {
                    assert_eq!(
                        paired[side].0[limb].0, single[side].0[limb].0,
                        "ragged raw limb {limb} side {side} at ({values_len},{a_len},{b_len})"
                    );
                }
            }
        }
    }

    /// Differential for the fused shifted opening. The fused form is
    /// congruent, not raw-identical, to the materialized-table form, so the
    /// assertion is canonical against BOTH the delayed-reduction table path
    /// and the reduce-per-term generic reference.
    #[test]
    fn ext2_scaled_dot_matches_materialized_shifted_table() {
        let p = GF::ORDER;
        let raw_specials = [0, 1, 2, p - 1, p, p + 1, u64::MAX];
        let lengths = [
            0usize, 1, 2, 3, 4, 5, 8, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025,
            4095, 4096, 4097,
        ];

        let mut state = 0x5DEE_CE66_D65C_2A63u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in lengths {
            let values: Vec<Q2> = (0..len)
                .map(|i| {
                    let a0 = if i < raw_specials.len() {
                        raw_specials[i]
                    } else {
                        next()
                    };
                    let a1 = if i < raw_specials.len() {
                        raw_specials[raw_specials.len() - 1 - i]
                    } else {
                        next()
                    };
                    QuadraticExtension([GoldilocksField(a0), GoldilocksField(a1)])
                })
                .collect();
            let coefficients: Vec<GF> = (0..len)
                .map(|i| {
                    GoldilocksField(if i < raw_specials.len() {
                        raw_specials[(i * 3) % raw_specials.len()]
                    } else {
                        next()
                    })
                })
                .collect();
            let scales: Vec<GF> = (0..len)
                .map(|i| {
                    GoldilocksField(if i < raw_specials.len() {
                        raw_specials[(i * 5 + 1) % raw_specials.len()]
                    } else {
                        next()
                    })
                })
                .collect();

            let materialized = values
                .iter()
                .zip(&scales)
                .map(|(&value, &scale)| <Q2 as FieldExtension<2>>::scalar_mul(&value, scale))
                .collect::<Vec<_>>();
            let table_path =
                <GF as Extendable<2>>::extension_base_dot_product(&materialized, &coefficients);
            let generic = generic_extension_base_dot_product(&materialized, &coefficients);
            let fused = <GF as Extendable<2>>::extension_base_dot_product_with_subgroup_scales(
                &values,
                &coefficients,
                &scales,
            );
            for limb in 0..2 {
                assert_eq!(
                    fused.0[limb].to_canonical_u64(),
                    table_path.0[limb].to_canonical_u64(),
                    "canonical limb {limb} vs table path at len {len}"
                );
                assert_eq!(
                    fused.0[limb].to_canonical_u64(),
                    generic.0[limb].to_canonical_u64(),
                    "canonical limb {limb} vs generic reference at len {len}"
                );
            }
        }
    }

    /// Differential gate for the fused multiply-accumulate.
    ///
    /// The contract is the field value, plus the structural bound that the
    /// two representatives never differ by more than a single `p`. Raw
    /// equality does NOT hold in general and must not be asserted: the
    /// separate `a * b + c` spelling runs the extension `Add`, whose
    /// Goldilocks addition returns `x + y` verbatim whenever that sum stays
    /// below `2^64`, so it can hand back a representative in `[p, 2^64)`
    /// where the fused form's single `reduce160` returns the canonical one.
    /// The first counterexample this sweep finds is
    /// `a = [p-2, 0], b = [0, p-2], c = [p-2, p-2]`, where limb 1 is `p + 2`
    /// separately and `2` fused. Non-canonical representatives are ordinary
    /// in this field (that `Add` produces them roughly half the time), and
    /// every consumer downstream — the sponge, the challenge derivation, the
    /// serializer — is congruence-preserving.
    #[test]
    fn ext2_mul_add_matches_mul_then_add_as_field_values() {
        use crate::goldilocks_extensions::ext2_mul_add;

        let p = GF::ORDER;
        let check = |actual: Q2, expected: Q2, what: &str| {
            for limb in 0..2 {
                let (a, e) = (actual.0[limb], expected.0[limb]);
                assert_eq!(
                    a.to_canonical_u64(),
                    e.to_canonical_u64(),
                    "canonical limb {limb} mismatch for {what}"
                );
                let spread = a.0.max(e.0) - a.0.min(e.0);
                assert!(
                    spread == 0 || spread == p,
                    "limb {limb} representatives differ by {spread}, not 0 or p, for {what}"
                );
            }
        };

        let raw_specials = [0u64, 1, 2, p - 2, p - 1, p, p + 1, u64::MAX - 1, u64::MAX];
        for &x in &raw_specials {
            for &y in &raw_specials {
                for &z in &raw_specials {
                    let a = QuadraticExtension([GoldilocksField(x), GoldilocksField(y)]);
                    let b = QuadraticExtension([GoldilocksField(y), GoldilocksField(z)]);
                    let c = QuadraticExtension([GoldilocksField(z), GoldilocksField(x)]);
                    check(
                        ext2_mul_add(a, b, c),
                        a * b + c,
                        &format!("boundary ({x}, {y}, {z})"),
                    );
                }
            }
        }

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let a = QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]);
            let b = QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]);
            let c = QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]);
            check(
                ext2_mul_add(a, b, c),
                a * b + c,
                &format!("random {a:?} {b:?} {c:?}"),
            );
        }
    }

    /// Sabotage control for the fused multiply-accumulate: a deliberately
    /// wrong `W` folding (3 instead of 7) must be caught by the differential
    /// above, so a passing run is evidence and not a vacuous assertion.
    #[test]
    fn ext2_mul_add_sabotage_is_detected() {
        let sabotaged = |a: Q2, b: Q2, c: Q2| {
            let QuadraticExtension([a0, a1]) = a;
            let QuadraticExtension([b0, b1]) = b;
            let QuadraticExtension([c0, c1]) = c;
            let (mut p_lo, mut p_hi) = (0u128, 0u32);
            let (mut w_lo, mut w_hi) = (0u128, 0u32);
            let (mut o_lo, mut o_hi) = (0u128, 0u32);
            super::u160_add_product(&mut p_lo, &mut p_hi, a0.0, b0.0);
            super::u160_add_product(&mut w_lo, &mut w_hi, a1.0, b1.0);
            super::u160_add_product(&mut o_lo, &mut o_hi, a0.0, b1.0);
            super::u160_add_product(&mut o_lo, &mut o_hi, a1.0, b0.0);
            let (s, k) = p_lo.overflowing_add(c0.0 as u128);
            p_lo = s;
            p_hi += k as u32;
            let (s, k) = o_lo.overflowing_add(c1.0 as u128);
            o_lo = s;
            o_hi += k as u32;
            // W = 3 instead of 7.
            let (w_lo, w_hi) = super::u160_times_3(w_lo, w_hi);
            let (lo, k) = p_lo.overflowing_add(w_lo);
            let hi = p_hi + w_hi + k as u32;
            QuadraticExtension([unsafe { super::reduce160(lo, hi) }, unsafe {
                super::reduce160(o_lo, o_hi)
            }])
        };
        let a = QuadraticExtension([GoldilocksField(3), GoldilocksField(5)]);
        let b = QuadraticExtension([GoldilocksField(7), GoldilocksField(11)]);
        let c = QuadraticExtension([GoldilocksField(13), GoldilocksField(17)]);
        assert_ne!(
            sabotaged(a, b, c).0[0].to_canonical_u64(),
            (a * b + c).0[0].to_canonical_u64()
        );
    }

    /// Sabotage controls: each differential above must fail when the
    /// mechanism it guards is perturbed, so a green run is evidence.
    #[test]
    fn paired_and_scaled_dot_sabotage_is_detected() {
        let values: Vec<Q2> = (1..=8u64)
            .map(|i| QuadraticExtension([GoldilocksField(i * 7), GoldilocksField(i * 11 + 3)]))
            .collect();
        let poly_a: Vec<GF> = (1..=8u64).map(|i| GoldilocksField(i * 13 + 1)).collect();
        let poly_b: Vec<GF> = (1..=8u64).map(|i| GoldilocksField(i * 17 + 5)).collect();

        // Paired dot: crossing the two coefficient slices must change both
        // outputs, i.e. the differential is not comparing a value to itself.
        let honest =
            <GF as Extendable<2>>::extension_base_dot_products_2(&values, [&poly_a, &poly_b]);
        let crossed =
            <GF as Extendable<2>>::extension_base_dot_products_2(&values, [&poly_b, &poly_a]);
        assert_ne!(
            honest[0].0[0].to_canonical_u64(),
            crossed[0].0[0].to_canonical_u64()
        );
        assert_ne!(
            honest[1].0[0].to_canonical_u64(),
            crossed[1].0[0].to_canonical_u64()
        );
        // And it must equal the two single dots it replaces, bit for bit.
        assert_eq!(
            honest[0].0[0].0,
            <GF as Extendable<2>>::extension_base_dot_product(&values, &poly_a).0[0].0
        );

        // Scaled dot: dropping the subgroup scale must change the result.
        let ones = vec![GF::ONE; values.len()];
        let scaled = <GF as Extendable<2>>::extension_base_dot_product_with_subgroup_scales(
            &values, &poly_a, &poly_b,
        );
        let unscaled = <GF as Extendable<2>>::extension_base_dot_product_with_subgroup_scales(
            &values, &poly_a, &ones,
        );
        assert_ne!(
            scaled.0[0].to_canonical_u64(),
            unscaled.0[0].to_canonical_u64()
        );
        assert_eq!(
            unscaled.0[0].to_canonical_u64(),
            <GF as Extendable<2>>::extension_base_dot_product(&values, &poly_a).0[0]
                .to_canonical_u64()
        );
    }

    #[test]
    fn ext2_extension_base_dot_product_reduce160_bound() {
        use num::BigUint;

        let one = BigUint::from(1u8);
        let max_product = ((&one << 64usize) - &one) * ((&one << 64usize) - &one);
        let reduce160_limit = (&one << 160usize) - (&one << 128usize) + (&one << 96usize);
        let max_safe_sum = BigUint::from(u32::MAX) * &max_product;
        let first_unsafe_worst_case = BigUint::from(u64::from(u32::MAX) + 1) * max_product;
        assert!(max_safe_sum < reduce160_limit);
        assert!(first_unsafe_worst_case >= reduce160_limit);
    }

    #[test]
    fn fri_fold_arity16_matches_horner_raw() {
        let check = |terms: [Q2; 16], beta: Q2| {
            let mut beta_powers = [Q2::ONE; 16];
            for i in 1..16 {
                beta_powers[i] = beta_powers[i - 1] * beta;
            }
            let expected = terms
                .iter()
                .rev()
                .fold(Q2::ZERO, |acc, &term| acc * beta + term);
            let actual = <GF as Extendable<2>>::fri_fold_arity16(&terms, beta, &beta_powers);
            for limb in 0..2 {
                assert_eq!(
                    actual.0[limb].0, expected.0[limb].0,
                    "raw limb {limb} mismatch for beta={beta:?}"
                );
            }
        };

        check([Q2::ZERO; 16], Q2::ZERO);
        check([Q2::ONE; 16], Q2::ONE);

        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let terms = core::array::from_fn(|_| {
                QuadraticExtension(core::array::from_fn(|_| GF::from_noncanonical_u64(next())))
            });
            let beta =
                QuadraticExtension(core::array::from_fn(|_| GF::from_noncanonical_u64(next())));
            check(terms, beta);
        }
    }

    /// The generic `Frobenius::repeated_frobenius` default implementation
    /// (from `extension/mod.rs`), reconstructed as the reference oracle.
    fn generic_repeated_frobenius(x: QE, count: usize) -> QE {
        if count == 0 {
            return x;
        } else if count >= 5 {
            return generic_repeated_frobenius(x, count % 5);
        }
        let arr = x.0;

        let mut z0 = <GF as Extendable<5>>::DTH_ROOT;
        for _ in 1..count {
            z0 *= <GF as Extendable<5>>::DTH_ROOT;
        }

        let mut res = [GF::ZERO; 5];
        for (i, z) in z0.powers().take(5).enumerate() {
            res[i] = arr[i] * z;
        }

        QuinticExtension(res)
    }

    #[test]
    fn quintic_frobenius_specialization_matches_generic() {
        let check = |x: QE| {
            for count in 0..=12 {
                let expected = generic_repeated_frobenius(x, count);
                let actual = x.repeated_frobenius(count);
                for j in 0..5 {
                    assert_eq!(
                        actual.0[j].to_canonical_u64(),
                        expected.0[j].to_canonical_u64(),
                        "limb {j} mismatch for count {count}, x={x:?}"
                    );
                }
                // `frobenius` is defined in terms of `repeated_frobenius`.
                if count == 1 {
                    let frob = x.frobenius();
                    for j in 0..5 {
                        assert_eq!(
                            frob.0[j].to_canonical_u64(),
                            expected.0[j].to_canonical_u64()
                        );
                    }
                }
            }
        };

        // Edge cases: 0, 1, 2, -1, scaled basis vectors, a low-order element
        // and non-canonical representations.
        let p = GF::ORDER;
        check(QE::ZERO);
        check(QE::ONE);
        check(QE::TWO);
        check(QE::NEG_ONE);
        check(QuinticExtension([
            <GF as Extendable<5>>::DTH_ROOT,
            GF::ZERO,
            GF::ZERO,
            GF::ZERO,
            GF::ZERO,
        ]));
        for j in 0..5 {
            for v in [1, p - 1, p, u64::MAX] {
                let mut limbs = [GF::ZERO; 5];
                limbs[j] = GoldilocksField(v);
                check(QuinticExtension(limbs));
            }
        }

        // Randomized differential over the full u64 (non-canonical included) range.
        let mut state = 0xB7E1_5162_8AED_2A6Au64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let limbs = core::array::from_fn(|_| GoldilocksField(next()));
            check(QuinticExtension(limbs));
        }
    }

    /// The generic `QuinticFirstCoeff` default implementation (the `c0` row of the generic
    /// `Mul`), reconstructed as the reference oracle for the widening specialization above.
    fn generic_mul_first_coeff(a: QE, b: QE) -> GF {
        let QuinticExtension([a0, a1, a2, a3, a4]) = a;
        let QuinticExtension([b0, b1, b2, b3, b4]) = b;
        a0 * b0 + <QE as OEF<5>>::W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1)
    }

    /// The specialized (delayed-reduction) first-coefficient helper must agree with the
    /// generic expression as a field value on edge cases and random non-canonical inputs,
    /// and `try_inverse` (its only consumer) must still return exact inverses.
    #[test]
    fn quintic_first_coeff_specialization_matches_generic() {
        let canon = |x: GF| x.to_canonical_u64();
        let check_pair = |a: QE, b: QE| {
            assert_eq!(
                canon(a.mul_first_coeff(&b)),
                canon(generic_mul_first_coeff(a, b)),
                "first coeff mismatch for a={a:?}, b={b:?}"
            );
            // The first coefficient of a full product must also match Mul's c0.
            assert_eq!(canon(a.mul_first_coeff(&b)), canon((a * b).0[0]));
        };

        let p = GF::ORDER;
        let specials = [0u64, 1, 2, p - 1, p, u64::MAX];
        for &u in &specials {
            for &v in &specials {
                let a = QuinticExtension([GoldilocksField(u); 5]);
                let b = QuinticExtension([GoldilocksField(v); 5]);
                check_pair(a, b);
            }
        }

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let a = QuinticExtension(core::array::from_fn(|_| GoldilocksField(next())));
            let b = QuinticExtension(core::array::from_fn(|_| GoldilocksField(next())));
            check_pair(a, b);

            // try_inverse consumes the helper: x * x^-1 == 1 exactly.
            if !a.is_zero() {
                let inv = a
                    .try_inverse()
                    .expect("nonzero element must have an inverse");
                let prod = a * inv;
                let limbs = FieldExtension::<5>::to_basefield_array(&prod);
                assert_eq!(canon(limbs[0]), 1, "a * a^-1 != 1 for a={a:?}");
                for limb in &limbs[1..] {
                    assert_eq!(canon(*limb), 0, "a * a^-1 != 1 for a={a:?}");
                }
            }
        }
        assert!(QE::ZERO.try_inverse().is_none());
    }
}
