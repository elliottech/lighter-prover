// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::ecdsa::curve::curve_types::{AffinePoint, Curve};
use crate::ecdsa::gadgets::curve::{AffinePointTarget, CircuitBuilderCurve};
use crate::ecdsa::gadgets::curve_windowed_mul::CircuitBuilderWindowedMul;
use crate::nonnative::split_nonnative::CircuitBuilderSplit;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};

/// A point of the curve `C`, possibly the point at infinity.
///
/// Invariant: `point` always satisfies the curve equation. When `is_inf` is set it is an
/// arbitrary curve point whose value must be ignored. Callers that build this struct from
/// witness data must enforce the curve equation themselves.
///
/// The gadgets below are complete: they are satisfiable and correct for every pair of inputs.
/// They require that `C` has no point of order two (true for prime order curves), so that the
/// tangent slope always exists.
#[derive(Clone, Debug)]
pub struct CompletePointTarget<C: Curve> {
    pub point: AffinePointTarget<C>,
    pub is_inf: BoolTarget,
}

pub trait CircuitBuilderCurveComplete<F: RichField + Extendable<D>, const D: usize> {
    /// Wraps a point that is known to be a valid, finite curve point.
    fn complete_point_from_affine<C: Curve>(
        &mut self,
        p: &AffinePointTarget<C>,
    ) -> CompletePointTarget<C>;

    fn curve_double_complete<C: Curve>(
        &mut self,
        p: &CompletePointTarget<C>,
    ) -> CompletePointTarget<C>;

    fn curve_add_complete<C: Curve>(
        &mut self,
        p1: &CompletePointTarget<C>,
        p2: &CompletePointTarget<C>,
    ) -> CompletePointTarget<C>;

    fn random_access_complete_points<C: Curve>(
        &mut self,
        access_index: Target,
        v: &[CompletePointTarget<C>],
    ) -> CompletePointTarget<C>;
}

/// Whether `x` is zero in `FF`. `x` is not assumed to be reduced.
fn is_zero_nonnative<F: RichField + Extendable<D>, const D: usize, FF: PrimeField>(
    builder: &mut Builder<F, D>,
    x: &NonNativeTarget<FF>,
) -> BoolTarget {
    // `x < 2^(32 * num_limbs) < 2 * |FF|`, so the only multiples of `|FF|` it can hold are `0` and `|FF|`.
    assert!(BigUint::from(1u32) << (32 * x.value.num_limbs()) < FF::order() * 2u32);

    let modulus = builder.constant_biguint(&FF::order());
    let is_zero = builder.is_zero_biguint(&x.value);
    let is_modulus = builder.is_equal_biguint(&x.value, &modulus);
    builder.or(is_zero, is_modulus)
}

/// `3 * x^2 + a`, numerator of the tangent slope.
fn tangent_slope_numerator<F: RichField + Extendable<D>, const D: usize, C: Curve>(
    builder: &mut Builder<F, D>,
    x: &NonNativeTarget<C::BaseField>,
) -> NonNativeTarget<C::BaseField> {
    let x_squared = builder.mul_nonnative(x, x);
    let double_x_squared = builder.add_nonnative(&x_squared, &x_squared);
    let triple_x_squared = builder.add_nonnative(&double_x_squared, &x_squared);
    if C::A == C::BaseField::ZERO {
        triple_x_squared
    } else {
        let a = builder.constant_nonnative(C::A);
        builder.add_nonnative(&triple_x_squared, &a)
    }
}

/// Given `(x1, y1)`, `x2` and the slope of the line through both points, returns the third
/// point of the curve on that line, negated.
fn chord_result<F: RichField + Extendable<D>, const D: usize, C: Curve>(
    builder: &mut Builder<F, D>,
    lambda: &NonNativeTarget<C::BaseField>,
    x1: &NonNativeTarget<C::BaseField>,
    y1: &NonNativeTarget<C::BaseField>,
    x2: &NonNativeTarget<C::BaseField>,
) -> AffinePointTarget<C> {
    let lambda_squared = builder.mul_nonnative(lambda, lambda);
    let x_sum = builder.add_nonnative(x1, x2);
    let x3 = builder.sub_nonnative(&lambda_squared, &x_sum);
    let x_diff = builder.sub_nonnative(x1, &x3);
    let prod = builder.mul_nonnative(lambda, &x_diff);
    let y3 = builder.sub_nonnative(&prod, y1);

    AffinePointTarget { x: x3, y: y3 }
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderCurveComplete<F, D>
    for Builder<F, D>
{
    fn complete_point_from_affine<C: Curve>(
        &mut self,
        p: &AffinePointTarget<C>,
    ) -> CompletePointTarget<C> {
        CompletePointTarget {
            point: p.clone(),
            is_inf: self._false(),
        }
    }

    fn curve_double_complete<C: Curve>(
        &mut self,
        p: &CompletePointTarget<C>,
    ) -> CompletePointTarget<C> {
        let AffinePointTarget { x, y } = &p.point;

        // `y != 0` because the curve has no point of order two, so the slope is unique.
        let numerator = tangent_slope_numerator::<F, D, C>(self, x);
        let double_y = self.add_nonnative(y, y);
        let lambda = self.div_nonnative(&numerator, &double_y);

        // When `p` is infinity this doubles the placeholder point, which keeps it on the curve.
        CompletePointTarget {
            point: chord_result(self, &lambda, x, y, x),
            is_inf: p.is_inf,
        }
    }

    fn curve_add_complete<C: Curve>(
        &mut self,
        p1: &CompletePointTarget<C>,
        p2: &CompletePointTarget<C>,
    ) -> CompletePointTarget<C> {
        let AffinePointTarget { x: x1, y: y1 } = &p1.point;
        let AffinePointTarget { x: x2, y: y2 } = &p2.point;

        let x_diff = self.sub_nonnative(x2, x1);
        let y_diff = self.sub_nonnative(y2, y1);
        let x_eq = is_zero_nonnative(self, &x_diff);
        let y_eq = is_zero_nonnative(self, &y_diff);
        // Both points are on the curve, so `x1 == x2` implies `y2 == y1` or `y2 == -y1 != y1`.
        let is_same = self.and(x_eq, y_eq);
        let is_opposite = self.and_not(x_eq, y_eq);

        let either_inf = self.or(p1.is_inf, p2.is_inf);
        let no_slope = self.or(either_inf, is_opposite);
        let has_slope = self.not(no_slope);

        // Tangent slope if the points are equal, chord slope otherwise. Both denominators are
        // non-zero. The slope is unused when `!has_slope`, there it is set to `0 / 1`.
        let tangent_numerator = tangent_slope_numerator::<F, D, C>(self, x1);
        let double_y1 = self.add_nonnative(y1, y1);
        let numerator = self.select_nonnative(is_same, &tangent_numerator, &y_diff);
        let numerator = self.mul_nonnative_by_bool(&numerator, has_slope);
        let denominator = self.select_nonnative(is_same, &double_y1, &x_diff);
        let one = self.constant_nonnative(C::BaseField::ONE);
        let denominator = self.select_nonnative(has_slope, &denominator, &one);
        let lambda = self.div_nonnative(&numerator, &denominator);

        let sum: AffinePointTarget<C> = chord_result(self, &lambda, x1, y1, x2);

        // `inf + p2 = p2`, `p1 + inf = p1`, `p1 + (-p1) = inf` (with `p1` as the placeholder).
        let keep_p1 = self.or(p2.is_inf, is_opposite);
        let x = self.select_nonnative(keep_p1, x1, &sum.x);
        let y = self.select_nonnative(keep_p1, y1, &sum.y);
        let x = self.select_nonnative(p1.is_inf, x2, &x);
        let y = self.select_nonnative(p1.is_inf, y2, &y);

        let both_inf = self.and(p1.is_inf, p2.is_inf);
        let cancel = self.and_not(is_opposite, either_inf);
        let is_inf = self.or(both_inf, cancel);

        CompletePointTarget {
            point: AffinePointTarget { x, y },
            is_inf,
        }
    }

    fn random_access_complete_points<C: Curve>(
        &mut self,
        access_index: Target,
        v: &[CompletePointTarget<C>],
    ) -> CompletePointTarget<C> {
        let points = v.iter().map(|p| p.point.clone()).collect();
        let flags = v.iter().map(|p| p.is_inf.target).collect();

        CompletePointTarget {
            point: self.random_access_curve_points(access_index, points),
            is_inf: BoolTarget::new_unsafe(self.random_access(access_index, flags)),
        }
    }
}

/// Computes `scalar * base` using a 4-bit window.
pub fn fixed_base_curve_mul_complete<C: Curve, F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    base: AffinePoint<C>,
    scalar: &NonNativeTarget<C::ScalarField>,
) -> CompletePointTarget<C> {
    // Holds `(16^i) * base` for `i=0..scalar.value.limbs.len() * 8`.
    let scaled_base = (0..scalar.value.limbs.len() * 8).scan(base, |acc, _| {
        let tmp = *acc;
        for _ in 0..4 {
            *acc = acc.double();
        }
        Some(tmp)
    });

    let limbs = builder.split_nonnative_to_4_bit_limbs(scalar);

    let zero = builder.zero();
    let mut result: Option<CompletePointTarget<C>> = None;
    // `s * P = sum s_i * P_i` with `P_i = (16^i) * P` and `s = sum s_i * (16^i)`.
    for (limb, point) in limbs.into_iter().zip(scaled_base) {
        // `muls_point[t] = t * P_i` for `t=1..16`.
        let mut muls_point = (1..16)
            .scan(point, |acc, _| {
                let tmp = *acc;
                *acc = (point + *acc).to_affine();
                Some(tmp)
            })
            .map(|p| builder.constant_affine_point(p))
            .collect::<Vec<_>>();
        // `0 * P_i` is infinity, the point in position 0 is a placeholder.
        muls_point.insert(0, muls_point[0].clone());
        // `r = s_i * P_i`
        let r = CompletePointTarget {
            point: builder.random_access_curve_points(limb, muls_point),
            is_inf: builder.is_equal(limb, zero),
        };
        result = Some(match result {
            None => r,
            Some(acc) => builder.curve_add_complete(&acc, &r),
        });
    }

    // A constant zero scalar has no limbs.
    result.unwrap_or_else(|| CompletePointTarget {
        point: builder.constant_affine_point(base),
        is_inf: builder._true(),
    })
}

/// Computes `n*p + m*q` using windowed MSM, with a 2-bit window.
/// See Algorithm 9.23 in Handbook of Elliptic and Hyperelliptic Curve Cryptography for a
/// description.
/// `p` and `q` must be valid curve points, there is no other restriction on them.
pub fn curve_msm_complete<C: Curve, F: RichField + Extendable<D>, const D: usize>(
    builder: &mut Builder<F, D>,
    p: &AffinePointTarget<C>,
    q: &AffinePointTarget<C>,
    n: &NonNativeTarget<C::ScalarField>,
    m: &NonNativeTarget<C::ScalarField>,
) -> CompletePointTarget<C> {
    let limbs_n = builder.split_nonnative_to_2_bit_limbs(n);
    let limbs_m = builder.split_nonnative_to_2_bit_limbs(m);
    assert_eq!(limbs_n.len(), limbs_m.len());

    let p = builder.complete_point_from_affine(p);
    let q = builder.complete_point_from_affine(q);

    // Precomputes `precomputation[i + 4*j] = i*p + j*q` for `i,j=0..4`.
    let infinity = CompletePointTarget {
        point: p.point.clone(),
        is_inf: builder._true(),
    };
    let mut precomputation = vec![infinity.clone(); 16];
    precomputation[1] = p.clone();
    precomputation[2] = builder.curve_double_complete(&p);
    precomputation[3] = builder.curve_add_complete(&precomputation[2], &p);
    precomputation[4] = q.clone();
    precomputation[8] = builder.curve_double_complete(&q);
    precomputation[12] = builder.curve_add_complete(&precomputation[8], &q);
    for i in 1..4 {
        for j in 1..4 {
            precomputation[i + 4 * j] =
                builder.curve_add_complete(&precomputation[i], &precomputation[4 * j]);
        }
    }

    let four = builder.constant(F::from_canonical_usize(4));

    let mut result: Option<CompletePointTarget<C>> = None;
    for (limb_n, limb_m) in limbs_n.into_iter().zip(limbs_m).rev() {
        let index = builder.mul_add(four, limb_m, limb_n);
        let r = builder.random_access_complete_points(index, &precomputation);
        result = Some(match result {
            None => r,
            Some(acc) => {
                let acc = builder.curve_double_complete(&acc);
                let acc = builder.curve_double_complete(&acc);
                builder.curve_add_complete(&acc, &r)
            }
        });
    }

    // Constant zero scalars have no limbs.
    result.unwrap_or(infinity)
}

#[cfg(test)]
mod tests {
    use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
    use plonky2::field::types::Sample;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;
    use crate::bigint::biguint::WitnessBigUint;
    use crate::ecdsa::curve::curve_types::CurveScalar;
    use crate::ecdsa::curve::secp256k1::Secp256K1;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type Point = AffinePoint<Secp256K1>;

    fn mul_generator(k: Secp256K1Scalar) -> Point {
        (CurveScalar(k) * Secp256K1::GENERATOR_PROJECTIVE).to_affine()
    }

    fn add_virtual_complete_point(builder: &mut Builder<F, D>) -> CompletePointTarget<Secp256K1> {
        CompletePointTarget {
            point: builder.add_virtual_affine_point_target(),
            is_inf: builder.add_virtual_bool_target_safe(),
        }
    }

    /// Infinity is witnessed as an arbitrary curve point with the flag set.
    fn set_complete_point(
        pw: &mut PartialWitness<F>,
        target: &CompletePointTarget<Secp256K1>,
        value: Point,
    ) {
        let placeholder = if value.zero {
            Secp256K1::GENERATOR_AFFINE
        } else {
            value
        };
        pw.set_biguint_target(&target.point.x.value, &placeholder.x.to_canonical_biguint())
            .unwrap();
        pw.set_biguint_target(&target.point.y.value, &placeholder.y.to_canonical_biguint())
            .unwrap();
        pw.set_bool_target(target.is_inf, value.zero).unwrap();
    }

    /// Asserts that `actual` and `expected` are the same point.
    fn connect_complete_point(
        builder: &mut Builder<F, D>,
        actual: &CompletePointTarget<Secp256K1>,
        expected: &CompletePointTarget<Secp256K1>,
    ) {
        builder.connect(actual.is_inf.target, expected.is_inf.target);
        let is_finite = builder.not(expected.is_inf);
        builder.conditional_connect_nonnative(is_finite, &actual.point.x, &expected.point.x);
        builder.conditional_connect_nonnative(is_finite, &actual.point.y, &expected.point.y);
        // The placeholder of an infinity result must stay on the curve.
        builder.curve_assert_valid(&actual.point);
    }

    #[test]
    fn test_curve_add_and_double_complete() {
        let mut builder = Builder::<F, D>::new(CircuitConfig::standard_ecc_config());
        let p1 = add_virtual_complete_point(&mut builder);
        let p2 = add_virtual_complete_point(&mut builder);
        let expected_sum = add_virtual_complete_point(&mut builder);
        let expected_double = add_virtual_complete_point(&mut builder);

        let sum = builder.curve_add_complete(&p1, &p2);
        connect_complete_point(&mut builder, &sum, &expected_sum);
        let double = builder.curve_double_complete(&p1);
        connect_complete_point(&mut builder, &double, &expected_double);
        let data = builder.build::<C>();

        let p = mul_generator(Secp256K1Scalar::rand());
        let q = mul_generator(Secp256K1Scalar::rand());
        let inf = Point::ZERO;
        for (a, b) in [
            (p, q),
            (p, p),
            (p, -p),
            (inf, q),
            (p, inf),
            (inf, inf),
            (p, p.double()),
            (p.double(), -p),
        ] {
            let mut pw = PartialWitness::new();
            set_complete_point(&mut pw, &p1, a);
            set_complete_point(&mut pw, &p2, b);
            set_complete_point(&mut pw, &expected_sum, (a + b).to_affine());
            set_complete_point(&mut pw, &expected_double, a.double());
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        }

        // A wrong result is rejected, also in the exceptional cases.
        for (a, b, wrong) in [(p, q, p), (p, p, inf), (p, -p, p), (inf, q, inf), (p, p, q)] {
            let mut pw = PartialWitness::new();
            set_complete_point(&mut pw, &p1, a);
            set_complete_point(&mut pw, &p2, b);
            set_complete_point(&mut pw, &expected_sum, wrong);
            set_complete_point(&mut pw, &expected_double, a.double());
            assert!(data.prove(pw).is_err());
        }
    }

    #[test]
    fn test_fixed_base_complete() {
        let mut builder = Builder::<F, D>::new(CircuitConfig::standard_ecc_config());
        let scalar = builder.add_virtual_nonnative_target::<Secp256K1Scalar>();
        let expected = add_virtual_complete_point(&mut builder);

        let res = fixed_base_curve_mul_complete(&mut builder, Secp256K1::GENERATOR_AFFINE, &scalar);
        connect_complete_point(&mut builder, &res, &expected);

        // A constant zero has no limbs.
        let zero = builder.constant_nonnative(Secp256K1Scalar::ZERO);
        let zero_res =
            fixed_base_curve_mul_complete(&mut builder, Secp256K1::GENERATOR_AFFINE, &zero);
        builder.assert_one(zero_res.is_inf.target);
        let generator = builder.constant_affine_point(Secp256K1::GENERATOR_AFFINE);
        let zero_res = curve_msm_complete(&mut builder, &generator, &generator, &zero, &zero);
        builder.assert_one(zero_res.is_inf.target);

        let data = builder.build::<C>();

        for n in [
            Secp256K1Scalar::ZERO,
            Secp256K1Scalar::ONE,
            Secp256K1Scalar::NEG_ONE,
            Secp256K1Scalar::from_canonical_u64(1 << 40),
            Secp256K1Scalar::rand(),
        ] {
            let mut pw = PartialWitness::new();
            pw.set_biguint_target(&scalar.value, &n.to_canonical_biguint())
                .unwrap();
            set_complete_point(&mut pw, &expected, mul_generator(n));
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        }
    }

    #[test]
    fn test_curve_msm_complete() {
        let mut builder = Builder::<F, D>::new(CircuitConfig::standard_ecc_config());
        let p_target = builder.add_virtual_affine_point_target::<Secp256K1>();
        let q_target = builder.add_virtual_affine_point_target::<Secp256K1>();
        let n_target = builder.add_virtual_nonnative_target::<Secp256K1Scalar>();
        let m_target = builder.add_virtual_nonnative_target::<Secp256K1Scalar>();
        let expected = add_virtual_complete_point(&mut builder);

        let res = curve_msm_complete(&mut builder, &p_target, &q_target, &n_target, &m_target);
        connect_complete_point(&mut builder, &res, &expected);
        let data = builder.build::<C>();

        let p = mul_generator(Secp256K1Scalar::rand());
        let q = mul_generator(Secp256K1Scalar::rand());
        let n = Secp256K1Scalar::rand();
        let m = Secp256K1Scalar::rand();
        let zero = Secp256K1Scalar::ZERO;
        let two = Secp256K1Scalar::TWO;
        for (p, q, n, m) in [
            (p, q, n, m),
            (p, p, n, m),
            (p, -p, n, m),
            (p, -p, n, n),
            (p, p.double(), two, Secp256K1Scalar::NEG_ONE),
            (p, q, zero, zero),
            (p, q, zero, m),
            (p, q, n, zero),
        ] {
            let mut pw = PartialWitness::new();
            for (target, value) in [(&p_target, p), (&q_target, q)] {
                pw.set_biguint_target(&target.x.value, &value.x.to_canonical_biguint())
                    .unwrap();
                pw.set_biguint_target(&target.y.value, &value.y.to_canonical_biguint())
                    .unwrap();
            }
            pw.set_biguint_target(&n_target.value, &n.to_canonical_biguint())
                .unwrap();
            pw.set_biguint_target(&m_target.value, &m.to_canonical_biguint())
                .unwrap();
            let res = CurveScalar(n) * p.to_projective() + CurveScalar(m) * q.to_projective();
            set_complete_point(&mut pw, &expected, res.to_affine());
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        }
    }
}
