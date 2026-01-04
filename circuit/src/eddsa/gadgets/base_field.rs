// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use std::marker::PhantomData;

use anyhow::Result;
use num::BigUint;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};
use serde::Deserialize;

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::byte::split::CircuitBuilderByteSplit;
use crate::eddsa::curve::base_field::SquareRoot;
use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
use crate::eddsa::gates::mul_quintic_ext_base::QuinticMultiplicationGate;
use crate::eddsa::gates::square_quintic_ext_base::QuinticSquaringGate;
use crate::nonnative::NonNativeTarget;
use crate::types::config::{F, const_f};
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;

const SIX: F = const_f(6);
const THREE: F = const_f(3);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Deserialize)]
#[repr(transparent)]
pub struct QuinticExtensionTarget(pub [Target; 5]);

impl QuinticExtensionTarget {
    pub fn new(limbs: [Target; 5]) -> Self {
        Self(limbs)
    }

    pub fn to_target_array(&self) -> [Target; 5] {
        self.0
    }
}

impl Default for QuinticExtensionTarget {
    fn default() -> Self {
        Self::new([Target::default(); 5])
    }
}

pub trait CircuitBuilderGFp5<F: RichField + Extendable<5>> {
    fn add_virtual_quintic_ext_target(&mut self) -> QuinticExtensionTarget;
    fn add_virtual_public_quintic_ext_target(&mut self) -> QuinticExtensionTarget;
    fn conditional_assert_eq_quintic_ext(
        &mut self,
        cond: BoolTarget,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    );
    fn connect_quintic_ext(&mut self, a: QuinticExtensionTarget, b: QuinticExtensionTarget);
    fn register_quintic_ext_public_input(&mut self, a: QuinticExtensionTarget);

    fn zero_quintic_ext(&mut self) -> QuinticExtensionTarget;
    fn one_quintic_ext(&mut self) -> QuinticExtensionTarget;
    fn constant_quintic_ext(&mut self, c: QuinticExtension<F>) -> QuinticExtensionTarget;
    #[must_use]
    fn select_quintic_ext(
        &mut self,
        cond: BoolTarget,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;
    fn random_access_quintic_ext(
        &mut self,
        access_index: Target,
        v: &[QuinticExtensionTarget],
    ) -> QuinticExtensionTarget;
    #[must_use]
    fn is_equal_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> BoolTarget;
    fn conditional_assert_not_zero_quintic_ext(
        &mut self,
        cond: BoolTarget,
        a: QuinticExtensionTarget,
    );

    fn double_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget;
    fn triple_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget;

    fn neg_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget;
    fn add_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;
    fn add_const_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        c: QuinticExtension<F>,
    ) -> QuinticExtensionTarget;
    fn sub_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;
    fn mul_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;
    fn mul_const_quintic_ext(
        &mut self,
        c: QuinticExtension<F>,
        a: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;

    fn div_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;
    fn div_const_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        c: QuinticExtension<F>,
    ) -> QuinticExtensionTarget;
    fn div_or_zero_quintic_ext(
        &mut self,
        a: QuinticExtensionTarget,
        b: QuinticExtensionTarget,
    ) -> QuinticExtensionTarget;
    fn inverse_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget;

    fn any_sqrt_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget;
    fn try_any_sqrt_quintic_ext(
        &mut self,
        x: QuinticExtensionTarget,
    ) -> (QuinticExtensionTarget, BoolTarget);
    fn try_canonical_sqrt_quintic_ext(
        &mut self,
        x: QuinticExtensionTarget,
    ) -> (QuinticExtensionTarget, BoolTarget);
    fn canonical_sqrt_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget;

    fn sgn0_quintic_ext(&mut self, x: QuinticExtensionTarget) -> BoolTarget;
    fn legendre_sym_quintic_ext(&mut self, x: QuinticExtensionTarget) -> Target;
    fn frob_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget;
    fn frob2_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget;

    fn is_zero_quintic_ext(&mut self, x: QuinticExtensionTarget) -> BoolTarget;

    fn square_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget;
    fn add_many_quintic_ext(
        &mut self,
        terms: Vec<QuinticExtensionTarget>,
    ) -> QuinticExtensionTarget;
    fn dot_product_quintic_ext(
        &mut self,
        a: Vec<QuinticExtensionTarget>,
        b: Vec<QuinticExtensionTarget>,
    ) -> QuinticExtensionTarget;

    fn encode_quintic_ext_as_scalar(
        &mut self,
        x: QuinticExtensionTarget,
    ) -> NonNativeTarget<ECgFp5Scalar>;

    fn exp_quintic_ext(
        &mut self,
        base: QuinticExtensionTarget,
        exp: Target,
        max_exp: usize,
    ) -> QuinticExtensionTarget;
}

pub trait PartialWitnessQuinticExt<F: PrimeField64 + RichField + Extendable<5>>:
    Witness<F>
{
    fn get_quintic_ext_target(&self, target: QuinticExtensionTarget) -> QuinticExtension<F>;

    fn get_quintic_ext_targets(
        &self,
        targets: &[QuinticExtensionTarget],
    ) -> Vec<QuinticExtension<F>> {
        targets
            .iter()
            .map(|&t| self.get_quintic_ext_target(t))
            .collect()
    }

    fn set_quintic_ext_target(
        &mut self,
        target: QuinticExtensionTarget,
        value: QuinticExtension<F>,
    ) -> Result<()>;

    fn set_quintic_ext_targets(
        &mut self,
        targets: &[QuinticExtensionTarget],
        values: &[QuinticExtension<F>],
    ) -> Result<()> {
        for (&t, &v) in targets.iter().zip(values.iter()) {
            self.set_quintic_ext_target(t, v)?;
        }

        Ok(())
    }
}

impl<W: Witness<F>, F: PrimeField64 + RichField + Extendable<5>> PartialWitnessQuinticExt<F> for W {
    fn get_quintic_ext_target(&self, target: QuinticExtensionTarget) -> QuinticExtension<F> {
        let QuinticExtensionTarget([t0, t1, t2, t3, t4]) = target;
        QuinticExtension([
            self.get_target(t0),
            self.get_target(t1),
            self.get_target(t2),
            self.get_target(t3),
            self.get_target(t4),
        ])
    }

    fn set_quintic_ext_target(
        &mut self,
        target: QuinticExtensionTarget,
        value: QuinticExtension<F>,
    ) -> Result<()> {
        let QuinticExtensionTarget([t0, t1, t2, t3, t4]) = target;
        let [v0, v1, v2, v3, v4] = value.0;

        self.set_target(t0, v0)?;
        self.set_target(t1, v1)?;
        self.set_target(t2, v2)?;
        self.set_target(t3, v3)?;
        self.set_target(t4, v4)?;

        Ok(())
    }
}

macro_rules! impl_circuit_builder_for_extension_degree {
    ($degree:literal) => {
        impl CircuitBuilderGFp5<F> for Builder<F, $degree> {
            fn add_virtual_quintic_ext_target(&mut self) -> QuinticExtensionTarget {
                QuinticExtensionTarget::new([
                    self.add_virtual_target(),
                    self.add_virtual_target(),
                    self.add_virtual_target(),
                    self.add_virtual_target(),
                    self.add_virtual_target(),
                ])
            }

            fn add_virtual_public_quintic_ext_target(&mut self) -> QuinticExtensionTarget {
                QuinticExtensionTarget::new([
                    self.add_virtual_public_input(),
                    self.add_virtual_public_input(),
                    self.add_virtual_public_input(),
                    self.add_virtual_public_input(),
                    self.add_virtual_public_input(),
                ])
            }

            fn conditional_assert_eq_quintic_ext(
                &mut self,
                cond: BoolTarget,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) {
                for i in 0..5 {
                    self.conditional_assert_eq(cond, a.0[i], b.0[i])
                }
            }

            fn connect_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) {
                for (lhs, rhs) in a
                    .to_target_array()
                    .into_iter()
                    .zip(b.to_target_array().into_iter())
                {
                    self.connect(lhs, rhs);
                }
            }

            fn register_quintic_ext_public_input(&mut self, a: QuinticExtensionTarget) {
                for t in a.to_target_array().into_iter() {
                    self.register_public_input(t);
                }
            }

            fn zero_quintic_ext(&mut self) -> QuinticExtensionTarget {
                QuinticExtensionTarget::new([self.zero(); 5])
            }

            fn one_quintic_ext(&mut self) -> QuinticExtensionTarget {
                QuinticExtensionTarget::new([
                    self.one(),
                    self.zero(),
                    self.zero(),
                    self.zero(),
                    self.zero(),
                ])
            }

            fn constant_quintic_ext(&mut self, c: QuinticExtension<F>) -> QuinticExtensionTarget {
                let QuinticExtension([c0, c1, c2, c3, c4]) = c;
                QuinticExtensionTarget::new([
                    self.constant(c0),
                    self.constant(c1),
                    self.constant(c2),
                    self.constant(c3),
                    self.constant(c4),
                ])
            }

            fn select_quintic_ext(
                &mut self,
                cond: BoolTarget,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                let QuinticExtensionTarget([b0, b1, b2, b3, b4]) = b;
                QuinticExtensionTarget::new([
                    self.select(cond, a0, b0),
                    self.select(cond, a1, b1),
                    self.select(cond, a2, b2),
                    self.select(cond, a3, b3),
                    self.select(cond, a4, b4),
                ])
            }

            fn random_access_quintic_ext(
                &mut self,
                access_index: Target,
                v: &[QuinticExtensionTarget],
            ) -> QuinticExtensionTarget {
                let mut a0s = Vec::new();
                let mut a1s = Vec::new();
                let mut a2s = Vec::new();
                let mut a3s = Vec::new();
                let mut a4s = Vec::new();
                for &QuinticExtensionTarget([a0, a1, a2, a3, a4]) in v {
                    a0s.push(a0);
                    a1s.push(a1);
                    a2s.push(a2);
                    a3s.push(a3);
                    a4s.push(a4);
                }

                QuinticExtensionTarget([
                    self.random_access(access_index, a0s),
                    self.random_access(access_index, a1s),
                    self.random_access(access_index, a2s),
                    self.random_access(access_index, a3s),
                    self.random_access(access_index, a4s),
                ])
            }

            fn is_equal_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> BoolTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                let QuinticExtensionTarget([b0, b1, b2, b3, b4]) = b;

                let terms = vec![
                    self.is_equal(a0, b0).target,
                    self.is_equal(a1, b1).target,
                    self.is_equal(a2, b2).target,
                    self.is_equal(a3, b3).target,
                    self.is_equal(a4, b4).target,
                ];

                let prod = self.mul_many(terms);
                BoolTarget::new_unsafe(prod)
            }

            fn conditional_assert_not_zero_quintic_ext(
                &mut self,
                cond: BoolTarget,
                a: QuinticExtensionTarget,
            ) {
                let zero_quintic = self.zero_quintic_ext();
                let is_equal = self.is_equal_quintic_ext(a, zero_quintic);
                self.conditional_assert_false(cond, is_equal);
            }

            fn neg_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                QuinticExtensionTarget::new([
                    self.neg(a0),
                    self.neg(a1),
                    self.neg(a2),
                    self.neg(a3),
                    self.neg(a4),
                ])
            }

            fn double_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                QuinticExtensionTarget::new([
                    self.mul_const(F::TWO, a0),
                    self.mul_const(F::TWO, a1),
                    self.mul_const(F::TWO, a2),
                    self.mul_const(F::TWO, a3),
                    self.mul_const(F::TWO, a4),
                ])
            }

            fn triple_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                QuinticExtensionTarget::new([
                    self.mul_const(THREE, a0),
                    self.mul_const(THREE, a1),
                    self.mul_const(THREE, a2),
                    self.mul_const(THREE, a3),
                    self.mul_const(THREE, a4),
                ])
            }

            fn add_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                let QuinticExtensionTarget([b0, b1, b2, b3, b4]) = b;
                QuinticExtensionTarget::new([
                    self.add(a0, b0),
                    self.add(a1, b1),
                    self.add(a2, b2),
                    self.add(a3, b3),
                    self.add(a4, b4),
                ])
            }

            fn add_const_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                c: QuinticExtension<F>,
            ) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                let QuinticExtension([c0, c1, c2, c3, c4]) = c;
                QuinticExtensionTarget::new([
                    self.add_const(a0, c0),
                    self.add_const(a1, c1),
                    self.add_const(a2, c2),
                    self.add_const(a3, c3),
                    self.add_const(a4, c4),
                ])
            }

            fn sub_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                let QuinticExtensionTarget([b0, b1, b2, b3, b4]) = b;
                QuinticExtensionTarget::new([
                    self.sub(a0, b0),
                    self.sub(a1, b1),
                    self.sub(a2, b2),
                    self.sub(a3, b3),
                    self.sub(a4, b4),
                ])
            }

            fn mul_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let use_gate = self.builder.config.quintic_multiplication_gate_enabled();

                if use_gate {
                    let a_array = a.to_target_array();
                    let b_array = b.to_target_array();

                    let gate = QuinticMultiplicationGate::new_from_config(&self.builder.config);
                    let gate_ref = gate.clone();
                    let constants = vec![];

                    let (gate_row, i) = self.builder.find_slot(gate, &constants, &constants);

                    // Assign limbs of a and b to wires
                    for j in 0..5 {
                        let a_wire =
                            Target::wire(gate_row, gate_ref.wire_ith_multiplicand_jth_limb_0(i, j));
                        let b_wire =
                            Target::wire(gate_row, gate_ref.wire_ith_multiplicand_jth_limb_1(i, j));
                        self.builder.connect(a_array[j], a_wire);
                        self.builder.connect(b_array[j], b_wire);
                    }
                    // Construct the output limbs
                    let output_limbs = (0..5)
                        .map(|j| Target::wire(gate_row, gate_ref.wire_ith_output_jth_limb(i, j)))
                        .collect::<Vec<_>>();

                    QuinticExtensionTarget::new([
                        output_limbs[0],
                        output_limbs[1],
                        output_limbs[2],
                        output_limbs[3],
                        output_limbs[4],
                    ])
                } else {
                    let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                    let QuinticExtensionTarget([b0, b1, b2, b3, b4]) = b;

                    // c0 ← a0b0 + 3(a1b4 + a2b3 + a3b2 + a4b1)
                    // c1 ← a0b1 + a1b0 + 3(a2b4 + a3b3 + a4b2)
                    // c2 ← a0b2 + a1b1 + a2b0 + 3(a3b4 + a4b3)
                    // c3 ← a0b3 + a1b2 + a2b1 + a3b0 + 3a4b4
                    // c4 ← a0b4 + a1b3 + a2b2 + a3b1 + a4b0

                    let mut c0 = self.mul(a4, b1);
                    c0 = self.mul_add(a3, b2, c0);
                    c0 = self.mul_add(a2, b3, c0);
                    c0 = self.mul_add(a1, b4, c0);
                    c0 = self.mul_const(F::from_canonical_u64(3), c0);
                    c0 = self.mul_add(a0, b0, c0);

                    let mut c1 = self.mul(a4, b2);
                    c1 = self.mul_add(a3, b3, c1);
                    c1 = self.mul_add(a2, b4, c1);
                    c1 = self.mul_const(F::from_canonical_u64(3), c1);
                    c1 = self.mul_add(a1, b0, c1);
                    c1 = self.mul_add(a0, b1, c1);

                    let mut c2 = self.mul(a4, b3);
                    c2 = self.mul_add(a3, b4, c2);
                    c2 = self.mul_const(F::from_canonical_u64(3), c2);
                    c2 = self.mul_add(a2, b0, c2);
                    c2 = self.mul_add(a1, b1, c2);
                    c2 = self.mul_add(a0, b2, c2);

                    let mut c3 = self.mul(a4, b4);
                    c3 = self.mul_const(F::from_canonical_u64(3), c3);
                    c3 = self.mul_add(a3, b0, c3);
                    c3 = self.mul_add(a2, b1, c3);
                    c3 = self.mul_add(a1, b2, c3);
                    c3 = self.mul_add(a0, b3, c3);

                    let mut c4 = self.mul(a4, b0);
                    c4 = self.mul_add(a3, b1, c4);
                    c4 = self.mul_add(a2, b2, c4);
                    c4 = self.mul_add(a1, b3, c4);
                    c4 = self.mul_add(a0, b4, c4);

                    QuinticExtensionTarget::new([c0, c1, c2, c3, c4])
                }
            }

            fn mul_const_quintic_ext(
                &mut self,
                c: QuinticExtension<F>,
                a: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;
                let QuinticExtension([c0, c1, c2, c3, c4]) = c;
                let one = self.one();

                let lhs = self.arithmetic(c1, c2, one, a4, a3);
                let rhs = self.arithmetic(c3, c4, one, a2, a1);
                let mut r0 = self.add(lhs, rhs);
                r0 = self.arithmetic(c0, THREE, one, a0, r0);

                let mut rhs = self.arithmetic(c2, c3, one, a4, a3);
                rhs = self.arithmetic(c4 * THREE, THREE, one, a2, rhs);
                let lhs = self.arithmetic(c0, c1, one, a1, a0);
                let r1 = self.add(lhs, rhs);

                let mut rhs = self.arithmetic(c3, c4, one, a4, a3);
                rhs = self.arithmetic(c2, THREE, one, a0, rhs);
                let lhs = self.arithmetic(c0, c1, one, a2, a1);
                let r2 = self.add(lhs, rhs);

                let mut rhs = self.arithmetic(c3, THREE * c4, one, a0, a4);
                rhs = self.arithmetic(c2, F::ONE, one, a1, rhs);
                let lhs = self.arithmetic(c0, c1, one, a3, a2);
                let r3 = self.add(lhs, rhs);

                let mut rhs = self.arithmetic(c3, c4, one, a1, a0);
                rhs = self.arithmetic(c2, F::ONE, one, a2, rhs);
                let lhs = self.arithmetic(c0, c1, one, a4, a3);
                let r4 = self.add(lhs, rhs);

                QuinticExtensionTarget::new([r0, r1, r2, r3, r4])
            }

            fn div_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let quotient = self.add_virtual_quintic_ext_target();
                self.add_simple_generator(QuinticQuotientGenerator::new(a, b, quotient));

                let quotient_times_denominator = self.mul_quintic_ext(quotient, b);
                self.connect_quintic_ext(quotient_times_denominator, a);

                quotient
            }

            fn div_or_zero_quintic_ext(
                &mut self,
                a: QuinticExtensionTarget,
                b: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let quotient = self.add_virtual_quintic_ext_target();
                self.add_simple_generator(QuinticQuotientGenerator::new(a, b, quotient));

                let quotient_times_denominator = self.mul_quintic_ext(quotient, b);
                let zero_if_prod_is_a = self.sub_quintic_ext(quotient_times_denominator, a);

                // check zero
                // we can do the multiplication limb-wise here, as their product is zero
                // iff one of them is all zeros
                let QuinticExtensionTarget([b0, b1, b2, b3, b4]) = b;
                let QuinticExtensionTarget([p0, p1, p2, p3, p4]) = zero_if_prod_is_a;
                let z0 = self.mul(b0, p0);
                let z1 = self.mul(b1, p1);
                let z2 = self.mul(b2, p2);
                let z3 = self.mul(b3, p3);
                let z4 = self.mul(b4, p4);
                self.assert_zero(z0);
                self.assert_zero(z1);
                self.assert_zero(z2);
                self.assert_zero(z3);
                self.assert_zero(z4);

                quotient
            }

            fn div_const_quintic_ext(
                &mut self,
                num: QuinticExtensionTarget,
                denom: QuinticExtension<F>,
            ) -> QuinticExtensionTarget {
                let denom = self.constant_quintic_ext(denom);
                self.div_quintic_ext(num, denom)
            }

            fn inverse_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget {
                let one = self.one_quintic_ext();

                let inverse = self.add_virtual_quintic_ext_target();
                self.add_simple_generator(QuinticQuotientGenerator::new(one, x, inverse));

                let should_be_one = self.mul_quintic_ext(inverse, x);
                self.connect_quintic_ext(should_be_one, one);

                inverse
            }

            fn any_sqrt_quintic_ext(
                &mut self,
                x: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let (root_x, _) = self.try_any_sqrt_quintic_ext(x);
                root_x
            }

            fn try_any_sqrt_quintic_ext(
                &mut self,
                x: QuinticExtensionTarget,
            ) -> (QuinticExtensionTarget, BoolTarget) {
                let zero = self.zero_quintic_ext();
                let root_x = self.add_virtual_quintic_ext_target();
                let is_sqrt = self.add_virtual_bool_target_safe();
                self.add_simple_generator(QuinticSqrtGenerator::new(x, root_x, is_sqrt));

                let should_be_x_or_zero = self.square_quintic_ext(root_x);
                let x_or_zero = self.select_quintic_ext(is_sqrt, x, zero);
                self.connect_quintic_ext(should_be_x_or_zero, x_or_zero);

                (root_x, is_sqrt)
            }

            /// returns true or false indicating a notion of "sign" for quintic_ext.
            /// This is used to canonicalize the square root
            /// This is an implementation of the function sgn0 from the IRTF's hash-to-curve document
            /// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-hash-to-curve-07#name-the-sgn0-function
            fn sgn0_quintic_ext(&mut self, x: QuinticExtensionTarget) -> BoolTarget {
                let one = self.one();
                let zero = self.zero();

                let mut sign = self.constant_bool(false);
                let mut is_zero = self.constant_bool(true);
                for limb in x.to_target_array() {
                    let bit_decomp = self.split_le_base::<2>(limb, 64);

                    // sign_i = x_i mod 2
                    // is_zero_i = x_i == 0
                    // SAFETY: targets from bit_decomp guaranteed to contain values of 0 or 1
                    let sign_i = BoolTarget::new_unsafe(self.sub(one, bit_decomp[0]));
                    let is_zero_i = self.is_equal(limb, zero);

                    // sign = sign || (is_zero && sign_i)
                    // is_zero = is_zero && is_zero_i

                    // x or y = x + y - xy
                    let is_zero_and_sign_i = self.and(is_zero_i, sign_i);
                    let sign_and_is_zero_and_sign_i = self.and(sign, is_zero_and_sign_i);
                    let tmp = self.mul_const_add(
                        -F::ONE,
                        sign_and_is_zero_and_sign_i.target,
                        sign.target,
                    );
                    sign = BoolTarget::new_unsafe(self.add(tmp, is_zero_and_sign_i.target));
                    is_zero = self.and(is_zero, is_zero_i);
                }

                sign
            }

            fn legendre_sym_quintic_ext(&mut self, x: QuinticExtensionTarget) -> Target {
                // compute x^r where r = p^4 + p^3 + p^2 + p + 1
                let frob1 = self.frob_quintic_ext(x);
                let frob2 = self.frob2_quintic_ext(x);
                let frob1_times_frob2 = self.mul_quintic_ext(frob1, frob2);
                let frob2_frob1_times_frob2 = self.frob2_quintic_ext(frob1_times_frob2);

                let x_to_r_minus_1 =
                    self.mul_quintic_ext(frob1_times_frob2, frob2_frob1_times_frob2);
                let x_to_r_quintic = self.mul_quintic_ext(x_to_r_minus_1, x);

                // x^r guaranteed to be in base field
                let QuinticExtensionTarget([y, _, _, _, _]) = x_to_r_quintic;

                let y31 = self.exp_power_of_2(y, 31);
                let y63 = self.exp_power_of_2(y31, 32);

                let zero = self.zero();
                let one = self.one();
                let y31_is_zero = self.is_equal(y31, zero);
                let denom = self.select(y31_is_zero, one, y31);
                let res = self.div(y63, denom);

                self.select(y31_is_zero, zero, res)
            }

            fn frob_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget {
                let frob_coeff_1 = F::from_canonical_u64(1041288259238279555);
                let frob_coeff_2 = F::from_canonical_u64(15820824984080659046);
                let frob_coeff_3 = F::from_canonical_u64(211587555138949697);
                let frob_coeff_4 = F::from_canonical_u64(1373043270956696022);

                let QuinticExtensionTarget([c0, mut c1, mut c2, mut c3, mut c4]) = x;

                c1 = self.mul_const(frob_coeff_1, c1);
                c2 = self.mul_const(frob_coeff_2, c2);
                c3 = self.mul_const(frob_coeff_3, c3);
                c4 = self.mul_const(frob_coeff_4, c4);

                QuinticExtensionTarget([c0, c1, c2, c3, c4])
            }

            fn frob2_quintic_ext(&mut self, x: QuinticExtensionTarget) -> QuinticExtensionTarget {
                let frob2_coeff_1 = F::from_canonical_u64(15820824984080659046);
                let frob2_coeff_2 = F::from_canonical_u64(1373043270956696022);
                let frob2_coeff_3 = F::from_canonical_u64(1041288259238279555);
                let frob2_coeff_4 = F::from_canonical_u64(211587555138949697);

                let QuinticExtensionTarget([c0, mut c1, mut c2, mut c3, mut c4]) = x;

                c1 = self.mul_const(frob2_coeff_1, c1);
                c2 = self.mul_const(frob2_coeff_2, c2);
                c3 = self.mul_const(frob2_coeff_3, c3);
                c4 = self.mul_const(frob2_coeff_4, c4);

                QuinticExtensionTarget([c0, c1, c2, c3, c4])
            }

            // returns the sqrt(x) such that `sgn0(sqrt(x)) == false`
            fn canonical_sqrt_quintic_ext(
                &mut self,
                x: QuinticExtensionTarget,
            ) -> QuinticExtensionTarget {
                let root_x = self.any_sqrt_quintic_ext(x);
                let neg_root_x = self.neg_quintic_ext(root_x);

                let sign = self.sgn0_quintic_ext(root_x);
                self.select_quintic_ext(sign, neg_root_x, root_x)
            }

            fn try_canonical_sqrt_quintic_ext(
                &mut self,
                x: QuinticExtensionTarget,
            ) -> (QuinticExtensionTarget, BoolTarget) {
                let (root_x, is_sqrt) = self.try_any_sqrt_quintic_ext(x);
                let neg_root_x = self.neg_quintic_ext(root_x);

                let sign = self.sgn0_quintic_ext(root_x);
                let canonical_root_x = self.select_quintic_ext(sign, neg_root_x, root_x);

                (canonical_root_x, is_sqrt)
            }

            // Square and multiply
            fn exp_quintic_ext(
                &mut self,
                base: QuinticExtensionTarget,
                exp: Target,
                max_exp: usize,
            ) -> QuinticExtensionTarget {
                let mut pow = base;
                let mut result = self.one_quintic_ext();

                for &bit in self.split_le(exp, max_exp).iter() {
                    let new_result = self.mul_quintic_ext(result, pow);
                    result = self.select_quintic_ext(bit, new_result, result);
                    pow = self.square_quintic_ext(pow);
                }

                result
            }

            fn square_quintic_ext(&mut self, a: QuinticExtensionTarget) -> QuinticExtensionTarget {
                //let use_gate = self.builder.config.quintic_multiplication_gate_enabled();
                let use_gate = self.builder.config.quintic_squaring_gate_enabled();
                if use_gate {
                    let a_array = a.to_target_array();

                    let gate = QuinticSquaringGate::new_from_config(&self.builder.config);
                    let gate_ref = gate.clone();
                    let constants = vec![];

                    let (gate_row, i) = self.builder.find_slot(gate, &constants, &constants);

                    // Assign limbs of a to the wires
                    for j in 0..5 {
                        let a_wire =
                            Target::wire(gate_row, gate_ref.wire_ith_multiplicand_jth_limb(i, j));
                        self.builder.connect(a_array[j], a_wire);
                    }
                    // Construct the output limbs
                    let mut output_limbs = Vec::with_capacity(5);
                    for j in 0..5 {
                        let wire = Target::wire(gate_row, gate_ref.wire_ith_output_jth_limb(i, j));
                        output_limbs.push(wire);
                    }

                    QuinticExtensionTarget::new([
                        output_limbs[0],
                        output_limbs[1],
                        output_limbs[2],
                        output_limbs[3],
                        output_limbs[4],
                    ])
                } else {
                    let zero = self.zero();
                    let QuinticExtensionTarget([a0, a1, a2, a3, a4]) = a;

                    // c0 ← a0^2 + 6a1a4 + 6a2a3
                    // c1 ← 3a3^2 + 2a0a1 + 6a2a4
                    // c2 ← a1^2 + 2a0a2 + 6a3a4
                    // c3 ← 3a4^2 + 2a0a3 + 2a1a2
                    // c4 ← a2^2 + 2a0a4 + 2a1a3

                    let mut c0 = self.square(a0);
                    c0 = self.arithmetic(SIX, F::ONE, a1, a4, c0);
                    c0 = self.arithmetic(SIX, F::ONE, a2, a3, c0);

                    let mut c1 = self.arithmetic(THREE, F::ZERO, a3, a3, zero);
                    c1 = self.arithmetic(F::TWO, F::ONE, a0, a1, c1);
                    c1 = self.arithmetic(SIX, F::ONE, a2, a4, c1);

                    let mut c2 = self.square(a1);
                    c2 = self.arithmetic(F::TWO, F::ONE, a0, a2, c2);
                    c2 = self.arithmetic(SIX, F::ONE, a3, a4, c2);

                    let mut c3 = self.arithmetic(THREE, F::ZERO, a4, a4, zero);
                    c3 = self.arithmetic(F::TWO, F::ONE, a0, a3, c3);
                    c3 = self.arithmetic(F::TWO, F::ONE, a1, a2, c3);

                    let mut c4 = self.square(a2);
                    c4 = self.arithmetic(F::TWO, F::ONE, a0, a4, c4);
                    c4 = self.arithmetic(F::TWO, F::ONE, a1, a3, c4);

                    QuinticExtensionTarget([c0, c1, c2, c3, c4])
                }
                // self.mul_quintic_ext(a, a)
            }

            fn add_many_quintic_ext(
                &mut self,
                terms: Vec<QuinticExtensionTarget>,
            ) -> QuinticExtensionTarget {
                let mut sum = self.zero_quintic_ext();
                for term in terms {
                    sum = self.add_quintic_ext(sum, term);
                }
                sum
            }

            fn dot_product_quintic_ext(
                &mut self,
                a: Vec<QuinticExtensionTarget>,
                b: Vec<QuinticExtensionTarget>,
            ) -> QuinticExtensionTarget {
                let mut terms = Vec::new();
                for (a, b) in a.into_iter().zip(b.into_iter()) {
                    terms.push(self.mul_quintic_ext(a, b));
                }
                self.add_many_quintic_ext(terms)
            }

            fn encode_quintic_ext_as_scalar(
                &mut self,
                x: QuinticExtensionTarget,
            ) -> NonNativeTarget<ECgFp5Scalar> {
                let QuinticExtensionTarget([c0, c1, c2, c3, c4]) = x;

                let bytes = [
                    self.split_bytes(c0, 8),
                    self.split_bytes(c1, 8),
                    self.split_bytes(c2, 8),
                    self.split_bytes(c3, 8),
                    self.split_bytes(c4, 8),
                ]
                .concat();

                let limbs_u32 = bytes
                    .chunks(4)
                    .map(|chunk| U32Target(self.le_sum_bytes(chunk)))
                    .collect::<Vec<_>>();

                let number = BigUintTarget { limbs: limbs_u32 };
                let order = ECgFp5Scalar::order();
                let order_target = self.constant_biguint(&order);
                let double_order_target = self.constant_biguint(&(order * BigUint::from(2u8)));

                // Because Math.log2(1067993516717146951041484916571792702745057740581727230159139685185762082554198619328292418486241 * 3) > 320 bits,
                // it is enough to subtract the order and double order from the number
                let (number_minus_order, borrow_1) = self.try_sub_biguint(&number, &order_target);
                let (number_minus_double_order, borrow_2) =
                    self.try_sub_biguint(&number, &double_order_target);

                // borrow is guaranteed to be 0 or 1
                let value = self.select_biguint(
                    BoolTarget::new_unsafe(borrow_1.0),
                    &number,
                    &number_minus_order,
                );
                let value = self.select_biguint(
                    BoolTarget::new_unsafe(borrow_2.0),
                    &value,
                    &number_minus_double_order,
                );

                NonNativeTarget {
                    value,
                    _phantom: PhantomData,
                }
            }

            fn is_zero_quintic_ext(&mut self, x: QuinticExtensionTarget) -> BoolTarget {
                let zero = self.zero_quintic_ext();
                self.is_equal_quintic_ext(x, zero)
            }
        }
    };
}

impl_circuit_builder_for_extension_degree!(1);
impl_circuit_builder_for_extension_degree!(2);
impl_circuit_builder_for_extension_degree!(4);
impl_circuit_builder_for_extension_degree!(5);

#[derive(Debug, Default)]
pub struct QuinticQuotientGenerator {
    numerator: QuinticExtensionTarget,
    denominator: QuinticExtensionTarget,
    quotient: QuinticExtensionTarget,
}

impl QuinticQuotientGenerator {
    pub fn new(
        numerator: QuinticExtensionTarget,
        denominator: QuinticExtensionTarget,
        quotient: QuinticExtensionTarget,
    ) -> Self {
        QuinticQuotientGenerator {
            numerator,
            denominator,
            quotient,
        }
    }
}

impl<F: RichField + Extendable<5> + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for QuinticQuotientGenerator
{
    fn dependencies(&self) -> Vec<Target> {
        let mut deps = self.numerator.to_target_array().to_vec();
        deps.extend(self.denominator.to_target_array());
        deps
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let numerator_limbs = self
            .numerator
            .to_target_array()
            .map(|t| witness.get_target(t));
        let numerator = QuinticExtension::<F>::from_basefield_array(numerator_limbs);

        let denominator_limbs = self
            .denominator
            .to_target_array()
            .map(|t| witness.get_target(t));
        let denominator = QuinticExtension::<F>::from_basefield_array(denominator_limbs);

        let quotient = if denominator == QuinticExtension::<F>::ZERO {
            QuinticExtension::<F>::ZERO
        } else {
            numerator / denominator
        };
        for (lhs, rhs) in self.quotient.to_target_array().into_iter().zip(
            <QuinticExtension<F> as FieldExtension<5>>::to_basefield_array(&quotient).into_iter(),
        ) {
            out_buffer.set_target(lhs, rhs)?;
        }

        Ok(())
    }

    fn id(&self) -> String {
        "QuinticQuotientGenerator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target_array(&self.denominator.0)?;
        dst.write_target_array(&self.numerator.0)?;
        dst.write_target_array(&self.quotient.0)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let denominator = src.read_target_array()?;
        let numerator = src.read_target_array()?;
        let quotient = src.read_target_array()?;
        Ok(Self {
            numerator: QuinticExtensionTarget::new(numerator),
            denominator: QuinticExtensionTarget::new(denominator),
            quotient: QuinticExtensionTarget::new(quotient),
        })
    }
}

#[derive(Debug, Default)]
pub struct QuinticSqrtGenerator {
    x: QuinticExtensionTarget,
    root_x: QuinticExtensionTarget,
    is_sqrt: BoolTarget,
}

impl QuinticSqrtGenerator {
    pub fn new(
        x: QuinticExtensionTarget,
        root_x: QuinticExtensionTarget,
        is_sqrt: BoolTarget,
    ) -> Self {
        QuinticSqrtGenerator { x, root_x, is_sqrt }
    }
}

// try Extendable<5> directly
impl<const D: usize> SimpleGenerator<F, D> for QuinticSqrtGenerator
where
    F: RichField + Extendable<D>,
{
    fn dependencies(&self) -> Vec<Target> {
        self.x.to_target_array().to_vec()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let x_limbs = self.x.to_target_array().map(|t| witness.get_target(t));
        let x = QuinticExtension::<F>::from_basefield_array(x_limbs);

        match x.canonical_sqrt() {
            Some(root_x) => {
                for (lhs, rhs) in self.root_x.to_target_array().into_iter().zip(
                    <QuinticExtension<F> as FieldExtension<5>>::to_basefield_array(&root_x)
                        .into_iter(),
                ) {
                    out_buffer.set_target(lhs, rhs)?;
                }
                out_buffer.set_target(self.is_sqrt.target, F::ONE)
            }
            None => {
                for limb in self.root_x.to_target_array().into_iter() {
                    out_buffer.set_target(limb, F::ZERO)?;
                }
                out_buffer.set_target(self.is_sqrt.target, F::ZERO)
            }
        }
    }

    fn id(&self) -> String {
        "QuinticSqrtGenerator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target_array(&self.x.0)?;
        dst.write_target_array(&self.root_x.0)?;
        dst.write_target_bool(self.is_sqrt)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let x = src.read_target_array()?;
        let root_x = src.read_target_array()?;
        let is_sqrt = src.read_target_bool()?;
        Ok(Self {
            x: QuinticExtensionTarget::new(x),
            root_x: QuinticExtensionTarget::new(root_x),
            is_sqrt,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::{Field, PrimeField64, Sample};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use rand::thread_rng;

    use super::*;
    use crate::builder::Builder;
    use crate::eddsa::curve::scalar_field::biguint_from_array;
    use crate::eddsa::curve::test_utils::{gfp5_random_non_square, gfp5_random_sgn0_eq_0};
    use crate::eddsa::gadgets::scalar_field::{CircuitBuilderScalar, PartialWitnessScalar};
    use crate::nonnative::CircuitBuilderNonNative;
    use crate::types::config::CIRCUIT_CONFIG;

    #[test]
    fn test_add() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x_expected = QuinticExtension::<F>::sample(&mut rng);
        let y_expected = QuinticExtension::<F>::sample(&mut rng);
        let z_expected = x_expected + y_expected;

        let x = builder.constant_quintic_ext(x_expected);
        let y = builder.constant_quintic_ext(y_expected);
        let z = builder.add_quintic_ext(x, y);
        builder.register_quintic_ext_public_input(z);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(z, z_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_mul() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x_expected = QuinticExtension::<F>::sample(&mut rng);
        let y_expected = QuinticExtension::<F>::sample(&mut rng);
        let z_expected = x_expected * y_expected;

        let x = builder.constant_quintic_ext(x_expected);
        let y = builder.constant_quintic_ext(y_expected);
        let z = builder.mul_quintic_ext(x, y);
        builder.register_quintic_ext_public_input(z);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(z, z_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_sub() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x_expected = QuinticExtension::<F>::sample(&mut rng);
        let y_expected = QuinticExtension::<F>::sample(&mut rng);
        let z_expected = x_expected - y_expected;

        let x = builder.constant_quintic_ext(x_expected);
        let y = builder.constant_quintic_ext(y_expected);
        let z = builder.sub_quintic_ext(x, y);
        builder.register_quintic_ext_public_input(z);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(z, z_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_div() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x_expected = QuinticExtension::<F>::sample(&mut rng);
        let y_expected = QuinticExtension::<F>::sample(&mut rng);
        let z_expected = x_expected / y_expected;

        let x = builder.constant_quintic_ext(x_expected);
        let y = builder.constant_quintic_ext(y_expected);
        let z = builder.div_quintic_ext(x, y);
        builder.register_quintic_ext_public_input(z);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(z, z_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_inverse_quintic_ext() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x_expected = QuinticExtension::<F>::sample(&mut rng);
        let x_inv_expected = x_expected.inverse();

        let x = builder.constant_quintic_ext(x_expected);
        let x_inv = builder.inverse_quintic_ext(x);
        builder.register_quintic_ext_public_input(x_inv);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(x_inv, x_inv_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_exp_quintic_ext() -> Result<()> {
        // let _ = env_logger::try_init_from_env(
        //     env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"),
        // );

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut builder = Builder::<F, D>::new(CIRCUIT_CONFIG);

        let mut rng = thread_rng();

        for _ in 0..10 {
            let x = QuinticExtension::<F>::sample(&mut rng);
            let power_bit_size = rand::Rng::gen_range(&mut rng, 0..=32);
            let power = if power_bit_size == 0 {
                0
            } else {
                rand::Rng::gen_range(&mut rng, 0..(1u64 << power_bit_size))
            };
            let result_target = builder.constant_quintic_ext(x.exp_u64(power));

            let x_target = builder.constant_quintic_ext(x);
            let power_target = builder.constant_u64(power);
            let result = builder.exp_quintic_ext(x_target, power_target, power_bit_size);

            builder.connect_quintic_ext(result, result_target);
        }

        let circuit = builder.build::<C>();

        let pw = PartialWitness::new();
        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_any_sqrt_quintic_ext() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x = QuinticExtension::<F>::sample(&mut rng);
        let square_expected = x * x;

        builder.constant_quintic_ext(square_expected);

        let circuit = builder.build::<C>();

        let pw = PartialWitness::new();
        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }
    #[test]

    fn test_canonical_sqrt_quintic_ext() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let x = gfp5_random_sgn0_eq_0();
        let square_expected = x * x;

        let square = builder.constant_quintic_ext(square_expected);
        let sqrt = builder.canonical_sqrt_quintic_ext(square);
        builder.register_quintic_ext_public_input(sqrt);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_quintic_ext_target(sqrt, x)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_try_any_sqrt_quintic_ext() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let zero = builder.zero_quintic_ext();
        let true_target = builder.constant_bool(true);
        let false_target = builder.constant_bool(false);

        let x = QuinticExtension::<F>::sample(&mut rng);
        let square_expected = x * x;

        let square = builder.constant_quintic_ext(square_expected);
        let (_, is_square) = builder.try_any_sqrt_quintic_ext(square);
        builder.connect(true_target.target, is_square.target);

        let non_square = gfp5_random_non_square();
        let non_square = builder.constant_quintic_ext(non_square);
        let (should_be_zero, is_square) = builder.try_any_sqrt_quintic_ext(non_square);
        builder.connect(false_target.target, is_square.target);
        builder.connect_quintic_ext(should_be_zero, zero);

        let circuit = builder.build::<C>();

        let pw = PartialWitness::new();
        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }
    #[test]

    fn test_try_canonical_sqrt_quintic_ext() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::<F, D>::new(config);

        let zero = builder.zero_quintic_ext();
        let true_target = builder.constant_bool(true);
        let false_target = builder.constant_bool(false);

        let x = gfp5_random_sgn0_eq_0();
        let square_expected = x * x;

        let square = builder.constant_quintic_ext(square_expected);
        let (_, is_square) = builder.try_canonical_sqrt_quintic_ext(square);
        builder.connect(true_target.target, is_square.target);

        let non_square = gfp5_random_non_square();
        let non_square = builder.constant_quintic_ext(non_square);
        let (should_be_zero, is_square) = builder.try_canonical_sqrt_quintic_ext(non_square);
        builder.connect(false_target.target, is_square.target);
        builder.connect_quintic_ext(should_be_zero, zero);

        let circuit = builder.build::<C>();

        let pw = PartialWitness::new();
        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_legendre_sym_quintic_ext() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();

        // legendre sym == 1
        let mut builder = Builder::<F, D>::new(config.clone());

        let x = QuinticExtension::<F>::sample(&mut rng);
        let square = builder.constant_quintic_ext(x * x);
        let legendre_sym = builder.legendre_sym_quintic_ext(square);
        builder.register_public_input(legendre_sym);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_target(legendre_sym, F::ONE)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)?;

        // legendre sym == -1
        let mut builder = Builder::<F, D>::new(config.clone());

        let non_square = gfp5_random_non_square();
        let non_square = builder.constant_quintic_ext(non_square);
        let legendre_sym = builder.legendre_sym_quintic_ext(non_square);
        builder.register_public_input(legendre_sym);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_target(legendre_sym, F::NEG_ONE)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)?;

        // legendre sym == 0
        let mut builder = Builder::<F, D>::new(config);

        let zero = builder.zero_quintic_ext();
        let legendre_sym = builder.legendre_sym_quintic_ext(zero);
        builder.register_public_input(legendre_sym);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_target(legendre_sym, F::ZERO)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }

    #[test]
    fn test_encode_as_scalar() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = thread_rng();

        let config = CircuitConfig::standard_recursion_config();

        let mut builder = Builder::<F, D>::new(config.clone());
        let x = QuinticExtension::<F>::sample(&mut rng);

        let QuinticExtension(limbs) = x;
        let encoded_expected = ECgFp5Scalar::from_noncanonical_biguint(biguint_from_array(
            limbs.map(|l| l.to_canonical_u64()),
        ));

        let x = builder.constant_quintic_ext(x);
        let encoded = builder.encode_quintic_ext_as_scalar(x);
        let encoded_as_biguint = builder.nonnative_to_canonical_biguint(&encoded);
        builder.register_scalar_public_input(&encoded_as_biguint);

        let circuit = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_scalar_target(&encoded_as_biguint, encoded_expected)?;

        let proof = circuit.prove(pw)?;
        circuit.verify(proof)
    }
}
