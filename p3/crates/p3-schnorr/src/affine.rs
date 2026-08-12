//! Affine arithmetic on ECgFp5, the elliptic curve over `GF(p^5)` (`p` =
//! Goldilocks) that backs the Schnorr signatures verified by this crate.
//!
//! The curve is short Weierstrass `y^2 = x^3 + A*x + B` over the degree-5
//! extension field `GF(p^5)`, but every point here is tracked in the
//! `(x, u)` coordinate system used by Lighter's circuit formulas, where
//! `u = x / y`. This avoids ever computing or storing `y` directly; the
//! public point *encoding* exchanged between parties is `1 / u = y / x`
//! (see [`AffinePoint::encode`]/[`AffinePoint::from_encoded_x`]).
//!
//! All formulas here (`affine_add_terms`, `affine_double_terms`, the curve
//! parameters, and the Frobenius-based field inversion) must match the
//! constraints encoded in the AIR (see [`crate::air`]) bit-for-bit — any
//! divergence would mean the prover and verifier disagree on what a valid
//! signature is. They are intentionally not "simplified" independently of
//! the AIR; treat this file and the AIR's affine constraints as one unit
//! when auditing either.

use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;

use crate::types::FP5_LIMBS;

/// An element of `GF(p^5)` (`p` = Goldilocks), represented as 5 base-field
/// limbs in the polynomial basis `1, w, w^2, w^3, w^4`.
pub type Fp5 = [Goldilocks; FP5_LIMBS];

/// Weierstrass coefficient `A` of `y^2 = x^3 + A*x + B`.
pub const ECGFP5_A: [u64; FP5_LIMBS] = [2, 0, 0, 0, 0];
/// Weierstrass coefficient `B` of `y^2 = x^3 + A*x + B`.
pub const ECGFP5_B: [u64; FP5_LIMBS] = [0, 263, 0, 0, 0];
/// `2 * B`, precomputed for the point-addition formula.
pub const ECGFP5_B_MUL2: [u64; FP5_LIMBS] = [0, 526, 0, 0, 0];
/// `4 * B`, precomputed for the point-doubling formula.
pub const ECGFP5_B_MUL4: [u64; FP5_LIMBS] = [0, 1052, 0, 0, 0];
/// `x`-coordinate of the conventional ECgFp5 base point (generator).
pub const ECGFP5_GENERATOR_X: [u64; FP5_LIMBS] = [
    12883135586176881569,
    4356519642755055268,
    5248930565894896907,
    2165973894480315022,
    2448410071095648785,
];
/// Public encoding (`1 / u`) of the generator, i.e. `AffinePoint::generator().encode()`.
pub const ECGFP5_GENERATOR_ENCODING: [u64; FP5_LIMBS] = [4, 0, 0, 0, 0];

/// Quadratic/quintic non-residue `w` used to build `GF(p^5) = GF(p)[w] / (w^5 - w_nr)`.
const FP5_NONRESIDUE: u64 = 3;
/// `d`-th root of unity used by the Frobenius-based [`fp5_inverse`] (norm computation).
const FP5_FROBENIUS_DTH_ROOT: u64 = 1041288259238279555;

/// Affine ECgFp5 point in the coordinate system used by Lighter's formulas.
///
/// `x` is the curve x-coordinate and `u = x / y`. The public encoding is
/// `y / x = 1 / u`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffinePoint {
    pub x: Fp5,
    pub u: Fp5,
}

/// Denominator inverses computed while adding two distinct [`AffinePoint`]s.
///
/// The AIR re-derives the same addition formula from these witnessed
/// inverses and constrains `denominator * inverse == 1`, rather than
/// dividing directly, since field division has no native AIR gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineAddWitness {
    pub x_denominator_inverse: Fp5,
    pub u_denominator_inverse: Fp5,
}

/// Denominator inverses computed while doubling an [`AffinePoint`].
///
/// Serves the same role as [`AffineAddWitness`] but for the doubling formula.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineDoubleWitness {
    pub x_denominator_inverse: Fp5,
    pub u_denominator_inverse: Fp5,
}

impl AffinePoint {
    /// The conventional ECgFp5 base point used to derive public keys (`P = sk * G`).
    pub fn generator() -> Self {
        Self::from_encoded_x(
            fp5_from_u64_array(ECGFP5_GENERATOR_ENCODING),
            fp5_from_u64_array(ECGFP5_GENERATOR_X),
        )
        .expect("generator encoding is nonzero")
    }

    /// Reconstructs a point from its public encoding (`encoded = 1/u`) and x-coordinate.
    ///
    /// Returns `None` if `encoded` is zero (no valid `u = 1/encoded` exists); this is
    /// the point-at-infinity encoding and is rejected rather than represented.
    pub fn from_encoded_x(encoded: Fp5, x: Fp5) -> Option<Self> {
        fp5_inverse(&encoded).map(|u| Self { x, u })
    }

    /// Computes the public encoding `1 / u` (i.e. `y / x`) of this point.
    ///
    /// Returns `None` if `u` is zero, which cannot occur for a point produced by
    /// [`Self::add`]/[`Self::double`] on a curve with no point of order 2.
    pub fn encode(&self) -> Option<Fp5> {
        fp5_inverse(&self.u)
    }

    /// Negates the point: `-(x, u) = (x, -u)`, since negating `y` negates `u = x/y`.
    pub fn neg(&self) -> Self {
        Self {
            x: self.x,
            u: fp5_neg(&self.u),
        }
    }

    /// Checks `(u*y)^2 == x` is consistent with the curve equation, i.e. that
    /// `u^2 * (x^2 + A*x + B) == x`, which holds iff `(x, x/u)` lies on
    /// `y^2 = x^3 + A*x + B`.
    pub fn is_on_curve(&self) -> bool {
        let x2 = fp5_square(&self.x);
        let ax = fp5_mul(&ecgfp5_a(), &self.x);
        let rhs_without_x = fp5_add(&fp5_add(&x2, &ax), &ecgfp5_b());
        fp5_mul(&fp5_square(&self.u), &rhs_without_x) == self.x
    }

    /// Checks that `encoded` is a valid public encoding for x-coordinate `x`, without
    /// needing to invert `encoded` first.
    ///
    /// Equivalent to (but cheaper than) computing `u = 1/encoded` and calling
    /// [`Self::is_on_curve`]; used when validating an untrusted encoded public key.
    pub fn encoded_x_is_valid(encoded: &Fp5, x: &Fp5) -> bool {
        let e = fp5_sub(&fp5_square(encoded), &ecgfp5_a());
        let mut relation = fp5_square(x);
        relation = fp5_sub(&relation, &fp5_mul(&e, x));
        relation = fp5_add(&relation, &ecgfp5_b());
        fp5_is_zero(&relation)
    }

    /// Adds two distinct, non-identity affine points, returning the sum and
    /// the denominator inverses an AIR trace would witness to prove the
    /// division was correct.
    ///
    /// Returns `None` if either denominator in [`affine_add_terms`] is zero
    /// (e.g. `self == rhs`, which must instead go through [`Self::double`]).
    /// `self == rhs.neg()` (the point-at-infinity case) is **not** caught
    /// this way: both denominators stay nonzero and this returns `Some` with
    /// `(x, u) = (0, 0)` — a value `is_on_curve` wrongly accepts, since
    /// `(0,0)` isn't a real curve point under this encoding but trivially
    /// satisfies the curve-equation check used there. See
    /// `degenerate_cases::add_point_to_its_negation_is_a_nonzero_denominator_degenerate_case`.
    /// Callers needing a complete addition law must special-case the
    /// identity themselves (this crate's AIR does, via its own
    /// `is_identity` witness flags rather than relying on this function's
    /// `Option`).
    pub fn add(&self, rhs: &Self) -> Option<(Self, AffineAddWitness)> {
        let terms = affine_add_terms(self, rhs);
        let x_denominator_inverse = fp5_inverse(&terms.x_denominator)?;
        let u_denominator_inverse = fp5_inverse(&terms.u_denominator)?;
        let result = AffinePoint {
            x: fp5_mul(&terms.x_numerator, &x_denominator_inverse),
            u: fp5_mul(&terms.u_numerator, &u_denominator_inverse),
        };

        Some((
            result,
            AffineAddWitness {
                x_denominator_inverse,
                u_denominator_inverse,
            },
        ))
    }

    /// Doubles this point, returning the result and the denominator inverses an AIR
    /// trace would witness to prove the division was correct.
    ///
    /// Returns `None` if either denominator in [`affine_double_terms`] is zero, i.e.
    /// `self` is a point of order 2 (not present on this curve for valid points).
    pub fn double(&self) -> Option<(Self, AffineDoubleWitness)> {
        let terms = affine_double_terms(self);
        let x_denominator_inverse = fp5_inverse(&terms.x_denominator)?;
        let u_denominator_inverse = fp5_inverse(&terms.u_denominator)?;
        let result = AffinePoint {
            x: fp5_mul(&terms.x_numerator, &x_denominator_inverse),
            u: fp5_mul(&terms.u_numerator, &u_denominator_inverse),
        };

        Some((
            result,
            AffineDoubleWitness {
                x_denominator_inverse,
                u_denominator_inverse,
            },
        ))
    }

    /// Variable-time double-and-add scalar multiplication.
    ///
    /// Branches and the number of point operations both depend on `scalar`, so
    /// this must only be used with public scalars (e.g. STARK witness
    /// generation over public batch data), never with a secret signing key.
    pub fn scalar_mul_u64(&self, mut scalar: u64) -> Option<Option<Self>> {
        let mut accumulator: Option<Self> = None;
        let mut addend = *self;

        while scalar != 0 {
            if scalar & 1 == 1 {
                accumulator = Some(match accumulator {
                    Some(point) => point.add(&addend)?.0,
                    None => addend,
                });
            }
            scalar >>= 1;
            if scalar != 0 {
                addend = addend.double()?.0;
            }
        }

        Some(accumulator)
    }

    /// Variable-time double-and-add scalar multiplication over a little-endian bit
    /// sequence.
    ///
    /// Branches and the number of point operations both depend on `bits`, so
    /// this must only be used with public scalars (e.g. STARK witness
    /// generation over public batch data), never with a secret signing key.
    pub fn scalar_mul_le_bits<I>(&self, bits: I) -> Option<Option<Self>>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bits = bits.into_iter().peekable();
        let mut accumulator: Option<Self> = None;
        let mut addend = *self;

        while let Some(bit) = bits.next() {
            if bit {
                accumulator = Some(match accumulator {
                    Some(point) => point.add(&addend)?.0,
                    None => addend,
                });
            }
            if bits.peek().is_some() {
                addend = addend.double()?.0;
            }
        }

        Some(accumulator)
    }
}

/// The numerator/denominator pairs for `x` and `u` produced by an addition or
/// doubling formula, before dividing. Kept separate from the division so the AIR
/// can constrain `numerator == result * denominator` instead of performing
/// division natively.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineOperationTerms {
    pub x_numerator: Fp5,
    pub x_denominator: Fp5,
    pub u_numerator: Fp5,
    pub u_denominator: Fp5,
}

/// Unreduced addition-formula terms for `lhs + rhs` in `(x, u)` coordinates.
///
/// This is Lighter's complete-ish addition law for ECgFp5, restated without
/// dividing; see the module docs for why the AIR needs it in this form.
pub fn affine_add_terms(lhs: &AffinePoint, rhs: &AffinePoint) -> AffineOperationTerms {
    let t1 = fp5_mul(&lhs.x, &rhs.x);
    let t3 = fp5_mul(&lhs.u, &rhs.u);
    let t5 = fp5_add(&lhs.x, &rhs.x);
    let t6 = fp5_add(&lhs.u, &rhs.u);
    let t7 = fp5_add(&t1, &ecgfp5_b());
    let t9 = fp5_mul(
        &t3,
        &fp5_add(&fp5_mul(&t5, &ecgfp5_b_mul2()), &fp5_double(&t7)),
    );
    let t10 = fp5_mul(&fp5_add(&fp5_one(), &fp5_double(&t3)), &fp5_add(&t5, &t7));

    AffineOperationTerms {
        x_numerator: fp5_mul(&fp5_sub(&t10, &t7), &ecgfp5_b()),
        x_denominator: fp5_sub(&t7, &t9),
        u_numerator: fp5_mul(&t6, &fp5_sub(&ecgfp5_b(), &t1)),
        u_denominator: fp5_add(&t7, &t9),
    }
}

/// Unreduced doubling-formula terms for `2 * point` in `(x, u)` coordinates.
///
/// Lighter's doubling law for ECgFp5, restated without dividing; see the module
/// docs for why the AIR needs it in this form.
pub fn affine_double_terms(point: &AffinePoint) -> AffineOperationTerms {
    let u2 = fp5_square(&point.u);
    let w1 = fp5_sub(
        &fp5_one(),
        &fp5_mul(&u2, &fp5_double(&fp5_add(&point.x, &fp5_one()))),
    );
    let x_denominator = fp5_square(&w1);
    let u_numerator = fp5_sub(
        &fp5_square(&fp5_add(&w1, &point.u)),
        &fp5_add(&u2, &x_denominator),
    );
    let u_denominator = fp5_sub(
        &fp5_from_u64_array([2, 0, 0, 0, 0]),
        &fp5_add(&fp5_scalar_mul_u64(&u2, 4), &x_denominator),
    );

    AffineOperationTerms {
        x_numerator: fp5_mul(&u2, &ecgfp5_b_mul4()),
        x_denominator,
        u_numerator,
        u_denominator,
    }
}

/// Curve coefficient `A` as an `Fp5`. See [`ECGFP5_A`].
pub fn ecgfp5_a() -> Fp5 {
    fp5_from_u64_array(ECGFP5_A)
}

/// Curve coefficient `B` as an `Fp5`. See [`ECGFP5_B`].
pub fn ecgfp5_b() -> Fp5 {
    fp5_from_u64_array(ECGFP5_B)
}

/// `2 * B` as an `Fp5`. See [`ECGFP5_B_MUL2`].
pub fn ecgfp5_b_mul2() -> Fp5 {
    fp5_from_u64_array(ECGFP5_B_MUL2)
}

/// `4 * B` as an `Fp5`. See [`ECGFP5_B_MUL4`].
pub fn ecgfp5_b_mul4() -> Fp5 {
    fp5_from_u64_array(ECGFP5_B_MUL4)
}

/// The additive identity of `GF(p^5)`.
pub fn fp5_zero() -> Fp5 {
    [Goldilocks::ZERO; FP5_LIMBS]
}

/// The multiplicative identity of `GF(p^5)`.
pub fn fp5_one() -> Fp5 {
    let mut out = fp5_zero();
    out[0] = Goldilocks::ONE;
    out
}

/// Lifts 5 raw `u64` limbs (each reduced mod the Goldilocks prime) into an `Fp5`.
pub fn fp5_from_u64_array(value: [u64; FP5_LIMBS]) -> Fp5 {
    value.map(Goldilocks::from_u64)
}

/// Returns `true` iff every limb is zero, i.e. `value == 0` in `GF(p^5)`.
pub fn fp5_is_zero(value: &Fp5) -> bool {
    value.iter().all(|limb| *limb == Goldilocks::ZERO)
}

/// `GF(p^5)` addition, limb-wise (the extension's addition is the base ring's
/// addition applied componentwise in the polynomial basis).
pub fn fp5_add(lhs: &Fp5, rhs: &Fp5) -> Fp5 {
    core::array::from_fn(|lane| lhs[lane] + rhs[lane])
}

/// `GF(p^5)` subtraction, limb-wise.
pub fn fp5_sub(lhs: &Fp5, rhs: &Fp5) -> Fp5 {
    core::array::from_fn(|lane| lhs[lane] - rhs[lane])
}

/// `GF(p^5)` negation, limb-wise.
pub fn fp5_neg(value: &Fp5) -> Fp5 {
    core::array::from_fn(|lane| -value[lane])
}

/// `2 * value` in `GF(p^5)`.
pub fn fp5_double(value: &Fp5) -> Fp5 {
    fp5_add(value, value)
}

/// Multiplies every limb of `value` by a base-field `scalar`.
pub fn fp5_scalar_mul(value: &Fp5, scalar: Goldilocks) -> Fp5 {
    core::array::from_fn(|lane| value[lane] * scalar)
}

/// [`fp5_scalar_mul`] taking the scalar as a raw `u64`.
pub fn fp5_scalar_mul_u64(value: &Fp5, scalar: u64) -> Fp5 {
    fp5_scalar_mul(value, Goldilocks::from_u64(scalar))
}

/// `GF(p^5)` multiplication, reduced modulo `w^5 - FP5_NONRESIDUE` (schoolbook
/// polynomial multiplication of the two degree-4 representatives, folding
/// `w^5` and `w^6` terms back down using the extension's defining relation).
pub fn fp5_mul(lhs: &Fp5, rhs: &Fp5) -> Fp5 {
    let w = Goldilocks::from_u64(FP5_NONRESIDUE);
    [
        lhs[0] * rhs[0]
            + w * (lhs[1] * rhs[4] + lhs[2] * rhs[3] + lhs[3] * rhs[2] + lhs[4] * rhs[1]),
        lhs[0] * rhs[1]
            + lhs[1] * rhs[0]
            + w * (lhs[2] * rhs[4] + lhs[3] * rhs[3] + lhs[4] * rhs[2]),
        lhs[0] * rhs[2]
            + lhs[1] * rhs[1]
            + lhs[2] * rhs[0]
            + w * (lhs[3] * rhs[4] + lhs[4] * rhs[3]),
        lhs[0] * rhs[3] + lhs[1] * rhs[2] + lhs[2] * rhs[1] + lhs[3] * rhs[0] + w * lhs[4] * rhs[4],
        lhs[0] * rhs[4] + lhs[1] * rhs[3] + lhs[2] * rhs[2] + lhs[3] * rhs[1] + lhs[4] * rhs[0],
    ]
}

/// `value^2` in `GF(p^5)`.
pub fn fp5_square(value: &Fp5) -> Fp5 {
    fp5_mul(value, value)
}

/// Multiplicative inverse in `GF(p^5)` via the Frobenius/norm method: computes
/// `f = value^((p^5-1)/(p-1) / p)`-equivalent conjugate product so that
/// `value * f` lands in the base field `GF(p)` (the field norm), then inverts
/// that single base-field element and rescales `f` by it.
///
/// Returns `None` if `value` is zero or, in the extremely unlikely case the norm
/// itself happens to be zero (which cannot occur for nonzero `value`, since the
/// norm map on a field extension is multiplicative and `GF(p^5)` has no zero
/// divisors, but is still checked defensively).
pub fn fp5_inverse(value: &Fp5) -> Option<Fp5> {
    if fp5_is_zero(value) {
        return None;
    }

    let d = fp5_frobenius(value, 1);
    let e = fp5_mul(&d, &fp5_frobenius(&d, 1));
    let f = fp5_mul(&e, &fp5_frobenius(&e, 2));
    let g = fp5_inner_product_first_limb(value, &f);

    if g == Goldilocks::ZERO {
        return None;
    }

    Some(fp5_scalar_mul(&f, g.inverse()))
}

/// Applies the Frobenius endomorphism `x -> x^p` to `value`, `count` times.
///
/// In the polynomial basis this is multiplication of limb `i` by `root^(i*count)`,
/// where `root` is a primitive `FP5_LIMBS`-th root of unity (`FP5_FROBENIUS_DTH_ROOT`)
/// — the Frobenius permutes the basis `w^i` to `(root^i)*w^i` because `w^p =
/// root * w` in this extension.
pub fn fp5_frobenius(value: &Fp5, count: usize) -> Fp5 {
    let count = count % FP5_LIMBS;
    if count == 0 {
        return *value;
    }

    let root = Goldilocks::from_u64(FP5_FROBENIUS_DTH_ROOT);
    let mut z0 = root;
    for _ in 1..count {
        z0 *= root;
    }

    let mut powers = [Goldilocks::ZERO; FP5_LIMBS];
    powers[0] = Goldilocks::ONE;
    for lane in 1..FP5_LIMBS {
        powers[lane] = powers[lane - 1] * z0;
    }

    core::array::from_fn(|lane| value[lane] * powers[lane])
}

/// Computes only limb 0 of `fp5_mul(lhs, rhs)`, used by [`fp5_inverse`] where the
/// final norm step only needs that one limb.
fn fp5_inner_product_first_limb(lhs: &Fp5, rhs: &Fp5) -> Goldilocks {
    let w = Goldilocks::from_u64(FP5_NONRESIDUE);
    lhs[0] * rhs[0] + w * (lhs[1] * rhs[4] + lhs[2] * rhs[3] + lhs[3] * rhs[2] + lhs[4] * rhs[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp5_inverse_multiplies_to_one() {
        let value = fp5_from_u64_array([3, 5, 7, 11, 13]);
        let inverse = fp5_inverse(&value).unwrap();

        assert_eq!(fp5_mul(&value, &inverse), fp5_one());
    }

    #[test]
    fn affine_generator_is_on_curve() {
        let generator = AffinePoint::generator();

        assert!(generator.is_on_curve());
        assert_eq!(
            generator.encode().unwrap(),
            fp5_from_u64_array(ECGFP5_GENERATOR_ENCODING)
        );
    }

    #[test]
    fn affine_double_stays_on_curve() {
        let generator = AffinePoint::generator();
        let (double, witness) = generator.double().unwrap();
        let terms = affine_double_terms(&generator);

        assert!(double.is_on_curve());
        assert_eq!(
            fp5_mul(&terms.x_denominator, &witness.x_denominator_inverse),
            fp5_one()
        );
        assert_eq!(
            fp5_mul(&terms.u_denominator, &witness.u_denominator_inverse),
            fp5_one()
        );
    }

    #[test]
    fn affine_add_stays_on_curve() {
        let generator = AffinePoint::generator();
        let (double, _) = generator.double().unwrap();
        let (triple, witness) = double.add(&generator).unwrap();
        let terms = affine_add_terms(&double, &generator);

        assert!(triple.is_on_curve());
        assert_eq!(
            fp5_mul(&terms.x_denominator, &witness.x_denominator_inverse),
            fp5_one()
        );
        assert_eq!(
            fp5_mul(&terms.u_denominator, &witness.u_denominator_inverse),
            fp5_one()
        );
    }

    #[test]
    fn affine_negation_preserves_curve_and_flips_encoding() {
        let generator = AffinePoint::generator();
        let negated = generator.neg();

        assert!(negated.is_on_curve());
        assert_eq!(
            negated.encode().unwrap(),
            fp5_neg(&generator.encode().unwrap())
        );
    }

    #[test]
    fn affine_scalar_mul_u64_matches_repeated_add() {
        let generator = AffinePoint::generator();
        let (double, _) = generator.double().unwrap();
        let (triple, _) = double.add(&generator).unwrap();

        assert_eq!(generator.scalar_mul_u64(0).unwrap(), None);
        assert_eq!(generator.scalar_mul_u64(1).unwrap(), Some(generator));
        assert_eq!(generator.scalar_mul_u64(3).unwrap(), Some(triple));
        assert_eq!(
            generator
                .scalar_mul_le_bits([true, true, false, false])
                .unwrap(),
            Some(triple)
        );
    }

    #[cfg(feature = "reference")]
    fn ref_fp5_to_local(value: &goldilocks_crypto::Fp5Element) -> Fp5 {
        value
            .0
            .map(|limb| Goldilocks::from_u64(limb.to_canonical_u64()))
    }

    #[cfg(feature = "reference")]
    #[test]
    fn affine_double_matches_goldilocks_crypto_encoding() {
        let (double, _) = AffinePoint::generator().double().unwrap();
        let reference = goldilocks_crypto::Point::generator().double().encode();

        assert_eq!(double.encode().unwrap(), ref_fp5_to_local(&reference));
    }

    #[cfg(feature = "reference")]
    #[test]
    fn affine_add_matches_goldilocks_crypto_encoding() {
        let generator = AffinePoint::generator();
        let (double, _) = generator.double().unwrap();
        let (triple, _) = double.add(&generator).unwrap();

        let reference_generator = goldilocks_crypto::Point::generator();
        let reference = reference_generator
            .double()
            .add(&reference_generator)
            .encode();

        assert_eq!(triple.encode().unwrap(), ref_fp5_to_local(&reference));
    }

    #[cfg(feature = "reference")]
    #[test]
    fn affine_scalar_mul_u64_matches_goldilocks_crypto_encoding() {
        let scalar_mul = AffinePoint::generator().scalar_mul_u64(5).unwrap().unwrap();
        let reference = goldilocks_crypto::Point::generator().mul_simple(5).encode();

        assert_eq!(scalar_mul.encode().unwrap(), ref_fp5_to_local(&reference));
    }
}

#[cfg(test)]
mod degenerate_cases {
    use super::*;

    /// `add(P, -P)` should represent the point at infinity, but this affine
    /// `(x, u)` system has no encoding for infinity — the formula instead
    /// runs to completion with *nonzero* denominators (so [`AffinePoint::add`]
    /// returns `Some`, not `None`) and produces `(x, u) = (0, 0)`, which
    /// `is_on_curve` wrongly accepts (the curve equation `u^2*(x^2+Ax+B)=x`
    /// is trivially satisfied at `x=u=0` regardless of whether `(0,0)` is a
    /// real point). Any caller treating `add`'s `Some` result as "always a
    /// genuine curve point" is wrong for this specific input.
    ///
    /// This is a real quirk of the formula, not an AIR soundness gap: in the
    /// signature-batch AIR, the only point ever compared against the final
    /// EC accumulator is the decoded `R` (`target_r`), and `constrain_encoded_point_decode`
    /// requires `affine_x^2 - e*affine_x + B == 0` for the *decoded* x-coordinate
    /// — at `x=0` that forces `B==0`, which is false for this curve's `B`, so
    /// `target_r`'s x-coordinate (and, symmetrically, its u-coordinate, via
    /// the `encoded*inverse==1` check) can never legitimately be `0`. The
    /// phantom `(0,0)` therefore can never match a genuine `target_r`, even
    /// if it appears as an intermediate accumulator value — see
    /// [`phantom_zero_composes_as_the_additive_identity`] for why an
    /// intermediate occurrence doesn't even corrupt the rest of the chain.
    #[test]
    fn add_point_to_its_negation_is_a_nonzero_denominator_degenerate_case() {
        let g = AffinePoint::generator();
        let neg_g = g.neg();
        let (sum, _witness) = g
            .add(&neg_g)
            .expect("denominators are nonzero for P + (-P)");

        assert_eq!(sum.x, fp5_zero());
        assert_eq!(sum.u, fp5_zero());
        assert!(
            sum.is_on_curve(),
            "is_on_curve has no special case for (0,0) and accepts it"
        );
    }

    /// The phantom `(0, 0)` value from adding a point to its own negation
    /// (see [`add_point_to_its_negation_is_a_nonzero_denominator_degenerate_case`])
    /// composes like a genuine additive identity in further additions, on
    /// either side, again via the *general* nonzero-denominator formula
    /// (not a special case) — so an intermediate cancellation during a
    /// double-and-add chain doesn't corrupt later steps.
    #[test]
    fn phantom_zero_composes_as_the_additive_identity() {
        let g = AffinePoint::generator();
        let (double, _) = g.double().unwrap();
        let phantom = AffinePoint {
            x: fp5_zero(),
            u: fp5_zero(),
        };

        let (resumed, _) = phantom.add(&double).expect("nonzero denominators");
        assert_eq!(resumed, double);

        let (resumed, _) = double.add(&phantom).expect("nonzero denominators");
        assert_eq!(resumed, double);
    }
}

#[cfg(test)]
mod root_selection {
    use p3_field::PrimeField64;

    use super::*;

    /// `constrain_decoded_public_key_root` (in the AIR) picks out the
    /// non-square root of `x^2 - e*x + B = 0` as the canonical decoded
    /// x-coordinate, relying on `B` being a non-square in `GF(p^5)` — since
    /// the two roots multiply to `B`, both being squares or both being
    /// non-squares would make their product a square, so if `B` is a
    /// non-square exactly one root is. This pins down that premise: for an
    /// odd-degree extension `GF(p^5)/GF(p))`, an element is a square in the
    /// extension iff its norm (product of all Frobenius conjugates) is a
    /// square in the base field, checked here via Euler's criterion
    /// (`norm(B)^((p-1)/2) == 1` would mean "square"; it doesn't).
    #[test]
    fn b_is_a_nonsquare_so_root_selection_is_well_defined() {
        let b = ecgfp5_b();
        let d = fp5_frobenius(&b, 1);
        let e = fp5_mul(&d, &fp5_frobenius(&d, 1));
        let f = fp5_mul(&e, &fp5_frobenius(&e, 2));
        let norm_b = fp5_inner_product_first_limb(&b, &f);

        let euler_exponent = (Goldilocks::ORDER_U64 - 1) / 2;
        let legendre = norm_b.exp_u64(euler_exponent);

        assert_ne!(
            legendre,
            Goldilocks::ONE,
            "norm(B) must be a non-square in GF(p) for the AIR's root-selection trick to be sound"
        );
    }
}
