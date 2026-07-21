// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;

use super::constants::BLOB_WIDTH;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::blob::blob_domain::get_brp_roots_of_unity_as_constant;
use crate::blob::bls12_381_scalar_field::{BLS12_381_SCALAR_LIMBS, BLS12381Scalar};
use crate::blob::constants::*;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::poseidon2::Poseidon2;
use crate::types::config::{D, F};
use crate::uint::u8::{CircuitBuilderU8, U8Target};

// Represents evaluations of polynomial P at points w_0, w_1, ..., w_4096
// where w_i is the i'th 4096'th root of unity in bls12-381 scalar field.

#[derive(Debug, Clone)]
pub struct BlobPolynomialTarget(pub [NonNativeTarget<BLS12381Scalar>; BLOB_WIDTH]);

impl BlobPolynomialTarget {
    /// [blob_to_polynomial]: https://github.com/ethereum/consensus-specs/blob/017a8495f7671f5fff2075a9bfc9238c1a0982f8/specs/deneb/polynomial-commitments.md#blob_to_polynomial
    /// BLS12_381 scalar elements are generated from blob data by treating each consecutive 32 bytes as a big-endian integer.
    pub fn from_bytes<F, const D: usize>(
        builder: &mut Builder<F, D>,
        bytes: &[U8Target; BLOB_DATA_BYTES_COUNT],
    ) -> Self
    where
        F: RichField + Extendable<D> + Poseidon2,
    {
        let zero = builder.zero_u8();
        Self(
            bytes
                .chunks(31)
                .map(|chunk| {
                    let elem_big = builder.biguint_from_bytes_be(
                        // Append a zero byte at the most significant position
                        &[zero]
                            .iter()
                            .cloned()
                            .chain(chunk.iter().cloned())
                            .collect::<Vec<_>>(),
                    );
                    builder.biguint_to_nonnative(&elem_big)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }

    pub fn eval_horner(
        &self,
        builder: &mut Builder<F, D>,
        x: &NonNativeTarget<BLS12381Scalar>,
    ) -> NonNativeTarget<BLS12381Scalar> {
        let mut result = builder.zero_nonnative::<BLS12381Scalar>();
        for coeff in self.0.iter().rev() {
            result = builder.mul_nonnative(&result, x);
            result = builder.add_nonnative(&result, coeff);
        }
        result
    }

    /// Evaluate a polynomial (in evaluation form) at an arbitrary point ``z``.
    /// - When ``z`` is in the domain, the evaluation can be found by indexing the polynomial at the
    ///   position that ``z`` is in the domain.
    /// - When ``z`` is not in the domain, the barycentric formula is used:
    ///   f(z) = (z**WIDTH - 1) / WIDTH  *  sum_(i=0)^WIDTH  (f(DOMAIN[i]) * DOMAIN[i]) / (z - DOMAIN[i])
    ///
    /// In our case:
    /// - ``z`` is the challenge point in Fp
    /// - ``WIDTH`` is BLOB_WIDTH
    /// - ``DOMAIN`` is the bit_reversal_permutation roots of unity
    /// - ``f(DOMAIN[i])`` is the blob[i]
    ///
    pub fn eval_barycentric(
        &self,
        builder: &mut Builder<F, D>,
        x: &NonNativeTarget<BLS12381Scalar>,
    ) -> NonNativeTarget<BLS12381Scalar> {
        let one_big = builder.one_biguint();

        let roots_of_unity_brp = get_brp_roots_of_unity_as_constant(builder);

        let mut result = builder.zero_biguint();
        let mut cp_is_not_root_of_unity = builder._true();
        let mut barycentric_evaluation = builder.zero_biguint();
        for i in 0..BLOB_WIDTH {
            // avoid division by zero
            // safe_denominator_i = denominator_i       (denominator_i != 0)
            // safe_denominator_i = 1                   (denominator_i == 0)
            let denominator_i = builder.sub_nonnative(x, &roots_of_unity_brp[i]);
            let is_zero_denominator_i = builder.is_zero_biguint(&denominator_i.value);
            let safe_denominator_i = NonNativeTarget::from(builder.select_biguint(
                is_zero_denominator_i,
                &one_big,
                &denominator_i.value,
            ));
            // update `cp_is_not_root_of_unity`
            // cp_is_not_root_of_unity = 1          (initialize)
            // cp_is_not_root_of_unity = 0          (denominator_i == 0)
            cp_is_not_root_of_unity =
                builder.and_not(cp_is_not_root_of_unity, is_zero_denominator_i);

            // update `result`
            // result = blob[i]     (challenge_point = roots_of_unity_brp[i])
            let select_blob_i =
                builder.mul_biguint_by_bool(&self.0[i].value, is_zero_denominator_i);
            // Assuming fair prover, addition here should not overflow. If it does, circuit will fail but can't prove anything wrong.
            result =
                builder.add_biguint_non_carry(&result, &select_blob_i, BLS12_381_SCALAR_LIMBS + 1);

            let term_i =
                builder.mul_div_nonnative(&roots_of_unity_brp[i], &self.0[i], &safe_denominator_i);
            // Assuming fair prover, addition here should not overflow. If it does, circuit will fail but can't prove anything wrong.
            barycentric_evaluation = builder.add_biguint_non_carry(
                &barycentric_evaluation,
                &term_i.value,
                BLS12_381_SCALAR_LIMBS + 1,
            );
        }
        let result = builder.reduce(&result);
        let mut barycentric_evaluation = builder.reduce(&barycentric_evaluation);

        let cp_to_the_width = BLS12381Scalar::pow_to_const(builder, x, BLOB_WIDTH);
        let cp_to_the_width_minus_one =
            builder.sub_nonnative(&cp_to_the_width, &NonNativeTarget::from(one_big));
        let width = NonNativeTarget::from(builder.constant_biguint(&BigUint::from(BLOB_WIDTH)));
        let factor = builder.div_nonnative(&cp_to_the_width_minus_one, &width);
        barycentric_evaluation = builder.mul_nonnative(&barycentric_evaluation, &factor);

        // if challenge_point is a root of unity, then result = blob[i], else result = barycentric_evaluation
        let select_evaluation = NonNativeTarget::from(
            builder.mul_biguint_by_bool(&barycentric_evaluation.value, cp_is_not_root_of_unity),
        );

        builder.add_nonnative(&result, &select_evaluation)
    }
}
