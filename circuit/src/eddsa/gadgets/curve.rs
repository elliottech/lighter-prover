// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;

use super::base_field::PartialWitnessQuinticExt;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::eddsa::curve::curve::{ECgFp5Point, WeierstrassPoint};
use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::nonnative::NonNativeTarget;
use crate::nonnative::split_nonnative::CircuitBuilderSplit;
use crate::types::config::F;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct ECgFp5PointTarget(pub ([QuinticExtensionTarget; 2], BoolTarget));

pub trait CircuitBuilderEcGFp5 {
    fn add_virtual_ecgfp5_point_target(&mut self) -> ECgFp5PointTarget;
    fn register_ecgfp5_point_public_input(&mut self, point: ECgFp5PointTarget);
    fn ecgfp5_point_constant(&mut self, point: WeierstrassPoint) -> ECgFp5PointTarget;
    fn ecgfp5_zero(&mut self) -> ECgFp5PointTarget;
    fn ecgfp5_generator(&mut self) -> ECgFp5PointTarget;

    fn ecgfp5_point_eq(&mut self, a: ECgFp5PointTarget, b: ECgFp5PointTarget) -> BoolTarget;
    fn select_ecgfp5_point(
        &mut self,
        cond: BoolTarget,
        a: ECgFp5PointTarget,
        b: ECgFp5PointTarget,
    ) -> ECgFp5PointTarget;
    fn ecgfp5_random_access(
        &mut self,
        access_index: Target,
        v: &[ECgFp5PointTarget],
    ) -> ECgFp5PointTarget;

    fn ecgfp5_add(&mut self, a: ECgFp5PointTarget, b: ECgFp5PointTarget) -> ECgFp5PointTarget;
    fn ecgfp5_add_spec(&mut self, a: ECgFp5PointTarget, b: ECgFp5PointTarget) -> ECgFp5PointTarget;
    fn ecgfp5_double(&mut self, a: ECgFp5PointTarget) -> ECgFp5PointTarget;

    fn precompute_window(
        &mut self,
        a: ECgFp5PointTarget,
        window_bits: usize,
    ) -> Vec<ECgFp5PointTarget>;
    fn ecgfp5_scalar_mul(
        &mut self,
        a: ECgFp5PointTarget,
        scalar: &NonNativeTarget<ECgFp5Scalar>,
    ) -> ECgFp5PointTarget;

    fn precompute_window_const(
        &mut self,
        point: ECgFp5Point,
        window_bits: usize,
    ) -> Vec<ECgFp5PointTarget>;
    fn ecgfp5_scalar_mul_const(
        &mut self,
        point: ECgFp5Point,
        scalar: &NonNativeTarget<ECgFp5Scalar>,
    ) -> ECgFp5PointTarget;

    fn ecgfp5_point_encode(&mut self, a: ECgFp5PointTarget) -> QuinticExtensionTarget;
    fn ecgfp5_point_decode(&mut self, w: QuinticExtensionTarget) -> ECgFp5PointTarget;

    fn ecgfp5_muladd_2(
        &mut self,
        a: ECgFp5PointTarget,
        b: ECgFp5PointTarget,
        scalar_a: &NonNativeTarget<ECgFp5Scalar>,
        scalar_b: &NonNativeTarget<ECgFp5Scalar>,
    ) -> ECgFp5PointTarget;
}

macro_rules! impl_circuit_builder_for_extension_degree {
    ($degree:literal) => {
        impl CircuitBuilderEcGFp5 for Builder<F, $degree> {
            fn add_virtual_ecgfp5_point_target(&mut self) -> ECgFp5PointTarget {
                let x = self.add_virtual_quintic_ext_target();
                let y = self.add_virtual_quintic_ext_target();
                let is_inf = self.add_virtual_bool_target_safe();
                ECgFp5PointTarget(([x, y], is_inf))
            }

            fn register_ecgfp5_point_public_input(&mut self, point: ECgFp5PointTarget) {
                let ECgFp5PointTarget(([x, y], is_inf)) = point;
                self.register_quintic_ext_public_input(x);
                self.register_quintic_ext_public_input(y);
                self.register_public_input(is_inf.target);
            }

            fn ecgfp5_point_constant(&mut self, point: WeierstrassPoint) -> ECgFp5PointTarget {
                let WeierstrassPoint { x, y, is_inf } = point;

                let x = self.constant_quintic_ext(x);
                let y = self.constant_quintic_ext(y);
                let is_inf = self.constant_bool(is_inf);
                ECgFp5PointTarget(([x, y], is_inf))
            }

            fn ecgfp5_zero(&mut self) -> ECgFp5PointTarget {
                self.ecgfp5_point_constant(WeierstrassPoint::NEUTRAL)
            }

            fn ecgfp5_generator(&mut self) -> ECgFp5PointTarget {
                self.ecgfp5_point_constant(WeierstrassPoint::GENERATOR)
            }

            fn ecgfp5_point_eq(
                &mut self,
                a: ECgFp5PointTarget,
                b: ECgFp5PointTarget,
            ) -> BoolTarget {
                let ECgFp5PointTarget(([ax, ay], a_is_inf)) = a;
                let ECgFp5PointTarget(([bx, by], b_is_inf)) = b;

                let both_inf = self.and(a_is_inf, b_is_inf);

                let x_eq = self.is_equal_quintic_ext(ax, bx);
                let y_eq = self.is_equal_quintic_ext(ay, by);
                let both_eq = self.and(x_eq, y_eq);

                self.or(both_inf, both_eq)
            }

            fn select_ecgfp5_point(
                &mut self,
                cond: BoolTarget,
                a: ECgFp5PointTarget,
                b: ECgFp5PointTarget,
            ) -> ECgFp5PointTarget {
                let ECgFp5PointTarget(([ax, ay], a_is_inf)) = a;
                let ECgFp5PointTarget(([bx, by], b_is_inf)) = b;
                ECgFp5PointTarget((
                    [
                        self.select_quintic_ext(cond, ax, bx),
                        self.select_quintic_ext(cond, ay, by),
                    ],
                    BoolTarget::new_unsafe(self.select(cond, a_is_inf.target, b_is_inf.target)),
                ))
            }

            fn ecgfp5_random_access(
                &mut self,
                access_index: Target,
                v: &[ECgFp5PointTarget],
            ) -> ECgFp5PointTarget {
                let mut xs = Vec::new();
                let mut ys = Vec::new();
                let mut is_infs = Vec::new();
                for &ECgFp5PointTarget(([x, y], is_inf)) in v {
                    xs.push(x);
                    ys.push(y);
                    is_infs.push(is_inf.target);
                }

                ECgFp5PointTarget((
                    [
                        self.random_access_quintic_ext(access_index, &xs),
                        self.random_access_quintic_ext(access_index, &ys),
                    ],
                    BoolTarget::new_unsafe(self.random_access(access_index, is_infs)),
                ))
            }

            fn ecgfp5_add(
                &mut self,
                a: ECgFp5PointTarget,
                b: ECgFp5PointTarget,
            ) -> ECgFp5PointTarget {
                let ECgFp5PointTarget(([x1, y1], a_is_inf)) = a;
                let ECgFp5PointTarget(([x2, y2], b_is_inf)) = b;

                // note: paper has a typo. sx == 1 when x1 != x2, not when x1 == x2
                let x_same = self.is_equal_quintic_ext(x1, x2);
                let mut y_diff = self.is_equal_quintic_ext(y1, y2);
                y_diff = self.not(y_diff);

                let lambda_0_if_x_not_same = self.sub_quintic_ext(y2, y1);

                let mut lambda_0_if_x_same = self.square_quintic_ext(x1);
                lambda_0_if_x_same = self.triple_quintic_ext(lambda_0_if_x_same);
                lambda_0_if_x_same =
                    self.add_const_quintic_ext(lambda_0_if_x_same, WeierstrassPoint::A);

                let lambda_1_if_x_not_same = self.sub_quintic_ext(x2, x1);
                let lambda_1_if_x_same = self.double_quintic_ext(y1);

                let lambda_0 =
                    self.select_quintic_ext(x_same, lambda_0_if_x_same, lambda_0_if_x_not_same);
                let lambda_1 =
                    self.select_quintic_ext(x_same, lambda_1_if_x_same, lambda_1_if_x_not_same);
                let lambda = self.div_or_zero_quintic_ext(lambda_0, lambda_1);

                let mut x3 = self.square_quintic_ext(lambda);
                x3 = self.sub_quintic_ext(x3, x1);
                x3 = self.sub_quintic_ext(x3, x2);

                let mut y3 = self.sub_quintic_ext(x1, x3);
                y3 = self.mul_quintic_ext(lambda, y3);
                y3 = self.sub_quintic_ext(y3, y1);

                let c_is_inf = self.and(x_same, y_diff);
                let c = ECgFp5PointTarget(([x3, y3], c_is_inf));

                let sel = self.select_ecgfp5_point(a_is_inf, b, c);
                self.select_ecgfp5_point(b_is_inf, a, sel)
            }

            fn ecgfp5_add_spec(
                &mut self,
                a: ECgFp5PointTarget,
                b: ECgFp5PointTarget,
            ) -> ECgFp5PointTarget {
                let ECgFp5PointTarget(([x1, y1], _)) = a;
                let ECgFp5PointTarget(([x2, y2], _)) = b;

                let lambda_0 = self.sub_quintic_ext(y2, y1);
                let lambda_1 = self.sub_quintic_ext(x2, x1);
                let lambda = self.div_or_zero_quintic_ext(lambda_0, lambda_1);

                let mut x3 = self.square_quintic_ext(lambda);
                x3 = self.sub_quintic_ext(x3, x1);
                x3 = self.sub_quintic_ext(x3, x2);

                let mut y3 = self.sub_quintic_ext(x1, x3);
                y3 = self.mul_quintic_ext(lambda, y3);
                y3 = self.sub_quintic_ext(y3, y1);

                ECgFp5PointTarget(([x3, y3], BoolTarget::new_unsafe(self.zero())))
            }

            fn ecgfp5_double(&mut self, a: ECgFp5PointTarget) -> ECgFp5PointTarget {
                let ECgFp5PointTarget(([x, y], is_inf)) = a;

                let mut lambda_0 = self.square_quintic_ext(x);
                lambda_0 = self.triple_quintic_ext(lambda_0);
                lambda_0 = self.add_const_quintic_ext(lambda_0, WeierstrassPoint::A);
                let lambda_1 = self.double_quintic_ext(y);

                let lambda = self.div_or_zero_quintic_ext(lambda_0, lambda_1);

                let mut x2 = self.square_quintic_ext(lambda);
                let two_x = self.double_quintic_ext(x);
                x2 = self.sub_quintic_ext(x2, two_x);

                let mut y2 = self.sub_quintic_ext(x, x2);
                y2 = self.mul_quintic_ext(lambda, y2);
                y2 = self.sub_quintic_ext(y2, y);

                ECgFp5PointTarget(([x2, y2], is_inf))
            }

            fn precompute_window(
                &mut self,
                a: ECgFp5PointTarget,
                window_bits: usize,
            ) -> Vec<ECgFp5PointTarget> {
                debug_assert!(window_bits > 1);
                let mut multiples = vec![self.ecgfp5_zero()];
                multiples.push(a);
                multiples.push(self.ecgfp5_double(a));

                for _ in 3..(1 << window_bits) {
                    multiples.push(self.ecgfp5_add(multiples.last().unwrap().clone(), a));
                }

                multiples
            }

            fn ecgfp5_scalar_mul(
                &mut self,
                a: ECgFp5PointTarget,
                scalar: &NonNativeTarget<ECgFp5Scalar>,
            ) -> ECgFp5PointTarget {
                let window = self.precompute_window(a, 4);
                let four_bit_limbs = self.split_nonnative_to_4_bit_limbs(&scalar);

                let num_limbs = four_bit_limbs.len();
                let mut res = self.ecgfp5_random_access(four_bit_limbs[num_limbs - 1], &window);
                for limb in four_bit_limbs.into_iter().rev().skip(1) {
                    for _ in 0..4 {
                        res = self.ecgfp5_double(res);
                    }

                    let addend = self.ecgfp5_random_access(limb, &window);
                    res = self.ecgfp5_add(res, addend);
                }

                res
            }

            fn precompute_window_const(
                &mut self,
                point: ECgFp5Point,
                window_bits: usize,
            ) -> Vec<ECgFp5PointTarget> {
                let mut curr = point;
                let mut multiples = vec![self.ecgfp5_zero()];

                for _ in 1..(1 << window_bits) {
                    multiples.push(self.ecgfp5_point_constant(curr.to_weierstrass()));
                    curr += point;
                }

                multiples
            }

            fn ecgfp5_scalar_mul_const(
                &mut self,
                point: ECgFp5Point,
                scalar: &NonNativeTarget<ECgFp5Scalar>,
            ) -> ECgFp5PointTarget {
                let window = self.precompute_window_const(point, 4);
                let four_bit_limbs = self.split_nonnative_to_4_bit_limbs(&scalar);

                let num_limbs = four_bit_limbs.len();
                let mut res = self.ecgfp5_random_access(four_bit_limbs[num_limbs - 1], &window);
                for limb in four_bit_limbs.into_iter().rev().skip(1) {
                    for _ in 0..4 {
                        res = self.ecgfp5_double(res);
                    }

                    let addend = self.ecgfp5_random_access(limb, &window);
                    res = self.ecgfp5_add(res, addend);
                }

                res
            }

            fn ecgfp5_point_encode(&mut self, a: ECgFp5PointTarget) -> QuinticExtensionTarget {
                let ECgFp5PointTarget(([x, y], is_inf)) = a;
                let adiv3 = self.constant_quintic_ext(
                    QuinticExtension::<F>::TWO / QuinticExtension::<F>::from_canonical_u16(3),
                );
                let denom = self.sub_quintic_ext(adiv3, x);
                let w = self.div_or_zero_quintic_ext(y, denom);

                let zero = self.zero_quintic_ext();
                self.select_quintic_ext(is_inf, zero, w)
            }

            fn ecgfp5_point_decode(&mut self, w: QuinticExtensionTarget) -> ECgFp5PointTarget {
                let one = self.one();
                let zero_quintic_ext = self.zero_quintic_ext();
                let a = self.constant_quintic_ext(ECgFp5Point::A);
                let bmul4 = self.constant_quintic_ext(ECgFp5Point::B_MUL4);

                let mut e = self.square_quintic_ext(w);
                e = self.sub_quintic_ext(e, a);

                let mut delta = self.square_quintic_ext(e);
                delta = self.sub_quintic_ext(delta, bmul4);

                let (r, delta_is_sqrt) = self.try_any_sqrt_quintic_ext(delta);

                // if delta is not a sqrt, then w must be zero. otherwise, it's not a valid point encoding
                // we check this by asserting that delta_is_sqrt OR w == 0.
                let w_is_zero = self.is_equal_quintic_ext(w, zero_quintic_ext);
                let delta_is_sqrt_or_w_is_zero = self.or(delta_is_sqrt, w_is_zero);
                self.assert_true(delta_is_sqrt_or_w_is_zero);

                let mut x1 = self.add_quintic_ext(e, r);
                x1 = self.div_const_quintic_ext(x1, QuinticExtension::<F>::TWO);

                let mut x2 = self.sub_quintic_ext(e, r);
                x2 = self.div_const_quintic_ext(x2, QuinticExtension::<F>::TWO);

                let legendre_x1 = self.legendre_sym_quintic_ext(x1);
                let legendre_is_one = self.is_equal(legendre_x1, one);
                let x = self.select_quintic_ext(legendre_is_one, x1, x2);

                let negw = self.neg_quintic_ext(w);
                let y = self.mul_quintic_ext(negw, x);

                let x = self.add_const_quintic_ext(
                    x,
                    ECgFp5Point::A / QuinticExtension::<F>::from_canonical_u16(3),
                );
                // since we checked above that w is zero if delta is not a sqrt, we can just set is_inf to delta_is_not_sqrt
                let is_inf = self.not(delta_is_sqrt);
                ECgFp5PointTarget(([x, y], is_inf))
            }

            fn ecgfp5_muladd_2(
                &mut self,
                a: ECgFp5PointTarget,
                b: ECgFp5PointTarget,
                scalar_a: &NonNativeTarget<ECgFp5Scalar>,
                scalar_b: &NonNativeTarget<ECgFp5Scalar>,
            ) -> ECgFp5PointTarget {
                let a_window = self.precompute_window(a, 4);
                let a_four_bit_limbs = self.split_nonnative_to_4_bit_limbs(&scalar_a);

                let b_window = self.precompute_window(b, 4);
                let b_four_bit_limbs = self.split_nonnative_to_4_bit_limbs(&scalar_b);

                debug_assert!(a_four_bit_limbs.len() == b_four_bit_limbs.len());

                let num_limbs = a_four_bit_limbs.len();
                let a_start = self.ecgfp5_random_access(a_four_bit_limbs[num_limbs - 1], &a_window);
                let b_start = self.ecgfp5_random_access(b_four_bit_limbs[num_limbs - 1], &b_window);
                let mut res = self.ecgfp5_add(a_start, b_start);

                for (a_limb, b_limb) in a_four_bit_limbs
                    .into_iter()
                    .zip(b_four_bit_limbs)
                    .rev()
                    .skip(1)
                {
                    for _ in 0..4 {
                        res = self.ecgfp5_double(res);
                    }

                    let a_addend = self.ecgfp5_random_access(a_limb, &a_window);
                    let b_addend = self.ecgfp5_random_access(b_limb, &b_window);
                    let addend = self.ecgfp5_add(a_addend, b_addend);
                    res = self.ecgfp5_add(res, addend);
                }

                res
            }
        }
    };
}

impl_circuit_builder_for_extension_degree!(1);
impl_circuit_builder_for_extension_degree!(2);
impl_circuit_builder_for_extension_degree!(4);
impl_circuit_builder_for_extension_degree!(5);

pub trait PartialWitnessCurve<F: RichField + Extendable<5> + PrimeField64>: Witness<F> {
    fn get_ecgfp5_point_target(&self, target: ECgFp5PointTarget) -> WeierstrassPoint;
    fn get_ecgfp5_point_targets(&self, targets: &[ECgFp5PointTarget]) -> Vec<WeierstrassPoint> {
        targets
            .iter()
            .map(|&t| self.get_ecgfp5_point_target(t))
            .collect()
    }

    fn set_ecgfp5_point_target(
        &mut self,
        target: ECgFp5PointTarget,
        value: WeierstrassPoint,
    ) -> Result<()>;
    fn set_ecgfp5_point_targets(
        &mut self,
        targets: &[ECgFp5PointTarget],
        values: &[WeierstrassPoint],
    ) -> Result<()> {
        for (&t, &v) in targets.iter().zip(values.iter()) {
            self.set_ecgfp5_point_target(t, v)?;
        }

        Ok(())
    }
}

impl<W: PartialWitnessQuinticExt<F>> PartialWitnessCurve<F> for W {
    fn get_ecgfp5_point_target(&self, target: ECgFp5PointTarget) -> WeierstrassPoint {
        let ECgFp5PointTarget(([x, y], is_inf)) = target;
        let x = self.get_quintic_ext_target(x);
        let y = self.get_quintic_ext_target(y);
        let is_inf = self.get_bool_target(is_inf);
        WeierstrassPoint { x, y, is_inf }
    }

    fn set_ecgfp5_point_target(
        &mut self,
        target: ECgFp5PointTarget,
        value: WeierstrassPoint,
    ) -> Result<()> {
        let ECgFp5PointTarget(([x, y], is_inf)) = target;
        self.set_quintic_ext_target(x, value.x)?;
        self.set_quintic_ext_target(y, value.y)?;
        self.set_bool_target(is_inf, value.is_inf)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Sample;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use rand::thread_rng;

    use super::*;
    use crate::nonnative::CircuitBuilderNonNative;

    #[test]
    fn test_ecgfp5_add() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p1 = ECgFp5Point::sample(&mut rng);
        let p2 = ECgFp5Point::sample(&mut rng);
        let p3_expected = p1 + p2;

        let p1 = builder.ecgfp5_point_constant(p1.to_weierstrass());
        let p2 = builder.ecgfp5_point_constant(p2.to_weierstrass());
        let p3 = builder.ecgfp5_add(p1, p2);
        builder.register_ecgfp5_point_public_input(p3);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_ecgfp5_point_target(p3, p3_expected.to_weierstrass())?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_ecgfp5_double() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p1 = ECgFp5Point::sample(&mut rng);
        let p2_expected = p1.double();

        let p1 = builder.ecgfp5_point_constant(p1.to_weierstrass());
        let p2 = builder.ecgfp5_double(p1);
        builder.register_ecgfp5_point_public_input(p2);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_ecgfp5_point_target(p2, p2_expected.to_weierstrass())?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_ecgfp5_scalar_mul() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p = ECgFp5Point::sample(&mut rng);
        let s = ECgFp5Scalar::sample(&mut rng);
        let prod_expected = p * s;

        let p = builder.ecgfp5_point_constant(p.to_weierstrass());
        let s = builder.constant_nonnative(s);

        let prod = builder.ecgfp5_scalar_mul(p, &s);
        builder.register_ecgfp5_point_public_input(prod);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_ecgfp5_point_target(prod, prod_expected.to_weierstrass())?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_ecgfp5_scalar_mul_const() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p = ECgFp5Point::sample(&mut rng);
        let s = ECgFp5Scalar::sample(&mut rng);
        let prod_expected = p * s;

        let s = builder.constant_nonnative(s);

        let prod = builder.ecgfp5_scalar_mul_const(p, &s);
        builder.register_ecgfp5_point_public_input(prod);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_ecgfp5_point_target(prod, prod_expected.to_weierstrass())?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_curve_encode() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p = ECgFp5Point::sample(&mut rng);
        let w_expected = p.encode();

        let p = builder.ecgfp5_point_constant(p.to_weierstrass());
        let w = builder.ecgfp5_point_encode(p);
        builder.register_quintic_ext_public_input(w);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(w, w_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_curve_decode() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p_expected = ECgFp5Point::sample(&mut rng);
        let w = p_expected.encode();

        let w = builder.constant_quintic_ext(w);
        let p = builder.ecgfp5_point_decode(w);
        builder.register_ecgfp5_point_public_input(p);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_ecgfp5_point_target(p, p_expected.to_weierstrass())?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_ecgfp5_muladd_2() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let p1 = ECgFp5Point::sample(&mut rng);
        let p2 = ECgFp5Point::sample(&mut rng);
        let s1 = ECgFp5Scalar::sample(&mut rng);
        let s2 = ECgFp5Scalar::sample(&mut rng);
        let prod_expected = p1 * s1 + p2 * s2;

        let p1 = builder.ecgfp5_point_constant(p1.to_weierstrass());
        let s1 = builder.constant_nonnative(s1);

        let p2 = builder.ecgfp5_point_constant(p2.to_weierstrass());
        let s2 = builder.constant_nonnative(s2);

        let prod = builder.ecgfp5_muladd_2(p1, p2, &s1, &s2);
        builder.register_ecgfp5_point_public_input(prod);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_ecgfp5_point_target(prod, prod_expected.to_weierstrass())?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }
}
