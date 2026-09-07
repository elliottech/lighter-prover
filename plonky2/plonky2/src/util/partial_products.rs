#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::iter;

use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::plonk::circuit_builder::CircuitBuilder;

#[allow(dead_code)]
pub(crate) fn quotient_chunk_products<F: Field>(
    quotient_values: &[F],
    max_degree: usize,
) -> Vec<F> {
    // One product per chunk: the historical exact capacity of the `collect()` this wrapper
    // replaces.
    let num_chunks = quotient_values.len().div_ceil(max_degree);
    let mut res = Vec::with_capacity(num_chunks);
    res.resize(num_chunks, F::ZERO);
    quotient_chunk_products_into(quotient_values, max_degree, &mut res);
    res
}

/// Allocation-free form of [`quotient_chunk_products`]: writes one product per
/// `max_degree`-sized chunk of `quotient_values` into `out`, which must hold exactly
/// `quotient_values.len().div_ceil(max_degree)` elements. The prover's Z-polynomial pass calls
/// this once per subgroup point with a slice of its flat chunk-products buffer, instead of
/// collecting a fresh Vec per point. Products and their order are identical to the allocating
/// form: each output is the same left-to-right `product()` over the same chunk.
#[allow(dead_code)]
pub(crate) fn quotient_chunk_products_into<F: Field>(
    quotient_values: &[F],
    max_degree: usize,
    out: &mut [F],
) {
    debug_assert!(max_degree > 1);
    assert!(!quotient_values.is_empty());
    let chunk_size = max_degree;
    let chunks = quotient_values.chunks(chunk_size);
    assert_eq!(out.len(), chunks.len());
    for (out_product, chunk) in out.iter_mut().zip(chunks) {
        *out_product = chunk.iter().copied().product();
    }
}

/// Compute partial products of the original vector `v` such that all products consist of `max_degree`
/// or less elements. This is done until we've computed the product `P` of all elements in the vector.
#[allow(dead_code)]
pub(crate) fn partial_products_and_z_gx<F: Field>(z_x: F, quotient_chunk_products: &[F]) -> Vec<F> {
    assert!(!quotient_chunk_products.is_empty());
    let mut res = Vec::with_capacity(quotient_chunk_products.len());
    let mut acc = z_x;
    for &quotient_chunk_product in quotient_chunk_products {
        acc *= quotient_chunk_product;
        res.push(acc);
    }
    res
}

/// Returns the length of the output of `partial_products()` on a vector of length `n`.
pub(crate) fn num_partial_products(n: usize, max_degree: usize) -> usize {
    debug_assert!(max_degree > 1);
    let chunk_size = max_degree;
    // We'll split the product into `n.div_ceil( chunk_size)` chunks, but the last chunk will
    // be associated with Z(gx) itself. Thus we subtract one to get the chunks associated with
    // partial products.
    n.div_ceil(chunk_size) - 1
}

/// Checks the relationship between each pair of partial product accumulators, appending the check
/// terms to `out`. In particular, this sequence of accumulators starts with `Z(x)`, then contains
/// each partial product polynomials `p_i(x)`, and finally `Z(g x)`. See the partial products
/// section of the Plonky2 paper.
///
/// This is the allocation-free form of [`check_partial_products`]: the batched vanishing-poly
/// evaluator calls it once per point and challenge with its worker-local scratch vector, instead
/// of collecting a fresh ten-element `Vec` and immediately `extend`ing it into that same scratch.
/// `out` is deliberately not cleared: for a given point, challenge `i + 1` must append after
/// challenge `i`.
pub(crate) fn check_partial_products_into<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
    out: &mut Vec<F>,
) {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    out.extend(
        numerators
            .chunks(chunk_size)
            .zip_eq(denominators.chunks(chunk_size))
            .zip_eq(product_accs.tuple_windows())
            .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
                let num_chunk_product = nume_chunk.iter().copied().product();
                let den_chunk_product = deno_chunk.iter().copied().product();
                // Assert that next_acc * deno_product = prev_acc * nume_product.
                prev_acc * num_chunk_product - next_acc * den_chunk_product
            }),
    );
}

/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
) -> Vec<F> {
    // One term per numerator/denominator chunk: the historical exact capacity of the
    // `collect()` this wrapper replaces.
    let mut res = Vec::with_capacity(numerators.len().div_ceil(max_degree));
    check_partial_products_into(
        numerators,
        denominators,
        partials,
        z_x,
        z_gx,
        max_degree,
        &mut res,
    );
    res
}

/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    numerators: &[ExtensionTarget<D>],
    denominators: &[ExtensionTarget<D>],
    partials: &[ExtensionTarget<D>],
    z_x: ExtensionTarget<D>,
    z_gx: ExtensionTarget<D>,
    max_degree: usize,
) -> Vec<ExtensionTarget<D>> {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    numerators
        .chunks(chunk_size)
        .zip_eq(denominators.chunks(chunk_size))
        .zip_eq(product_accs.tuple_windows())
        .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
            let nume_product = builder.mul_many_extension(nume_chunk);
            let deno_product = builder.mul_many_extension(deno_chunk);
            let next_acc_deno = builder.mul_extension(next_acc, deno_product);
            // Assert that next_acc * deno_product = prev_acc * nume_product.
            builder.mul_sub_extension(prev_acc, nume_product, next_acc_deno)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::PrimeField64;

    #[test]
    fn test_partial_products() {
        type F = GoldilocksField;
        let denominators = vec![F::ONE; 6];
        let z_x = F::ONE;
        let v = field_vec(&[1, 2, 3, 4, 5, 6]);
        let z_gx = F::from_canonical_u64(720);
        let quotient_chunks_prods = quotient_chunk_products(&v, 2);
        assert_eq!(quotient_chunks_prods, field_vec(&[2, 12, 30]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[2, 24, 720]));

        let nums = num_partial_products(v.len(), 2);
        assert_eq!(pps.len(), nums);
        assert!(check_partial_products(&v, &denominators, pps, z_x, z_gx, 2)
            .iter()
            .all(|x| x.is_zero()));

        let quotient_chunks_prods = quotient_chunk_products(&v, 3);
        assert_eq!(quotient_chunks_prods, field_vec(&[6, 120]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[6, 720]));
        let nums = num_partial_products(v.len(), 3);
        assert_eq!(pps.len(), nums);
        assert!(check_partial_products(&v, &denominators, pps, z_x, z_gx, 3)
            .iter()
            .all(|x| x.is_zero()));
    }

    fn field_vec<F: Field>(xs: &[usize]) -> Vec<F> {
        xs.iter().map(|&x| F::from_canonical_usize(x)).collect()
    }

    /// Independent copy of the pre-`_into` implementation, retained only as a
    /// differential-test oracle. Comparing `check_partial_products` (which now
    /// delegates to `_into`) against `_into` would be self-referential.
    fn check_partial_products_legacy<F: Field>(
        numerators: &[F],
        denominators: &[F],
        partials: &[F],
        z_x: F,
        z_gx: F,
        max_degree: usize,
    ) -> Vec<F> {
        debug_assert!(max_degree > 1);
        let product_accs = iter::once(&z_x)
            .chain(partials.iter())
            .chain(iter::once(&z_gx));
        let chunk_size = max_degree;
        numerators
            .chunks(chunk_size)
            .zip_eq(denominators.chunks(chunk_size))
            .zip_eq(product_accs.tuple_windows())
            .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
                let num_chunk_product = nume_chunk.iter().copied().product();
                let den_chunk_product = deno_chunk.iter().copied().product();
                prev_acc * num_chunk_product - next_acc * den_chunk_product
            })
            .collect()
    }

    /// Deterministic, mostly-noncanonical Goldilocks values so raw-representation
    /// comparisons are meaningful.
    fn noncanonical_vec(len: usize, seed: u64) -> Vec<GoldilocksField> {
        (0..len as u64)
            .map(|i| {
                GoldilocksField::from_noncanonical_u64(
                    u64::MAX - seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(3 * i),
                )
            })
            .collect()
    }

    fn raw(values: &[GoldilocksField]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    /// Independent copy of the pre-`_into` `quotient_chunk_products` implementation, retained
    /// only as a differential-test oracle (the shipping function now delegates to `_into`).
    fn quotient_chunk_products_legacy<F: Field>(
        quotient_values: &[F],
        max_degree: usize,
    ) -> Vec<F> {
        debug_assert!(max_degree > 1);
        assert!(!quotient_values.is_empty());
        let chunk_size = max_degree;
        quotient_values
            .chunks(chunk_size)
            .map(|chunk| chunk.iter().copied().product())
            .collect()
    }

    /// `quotient_chunk_products_into` (and the wrapper that now delegates to it) must produce
    /// exactly the legacy per-point Vec's contents, raw representation equality, for both a
    /// full final chunk and a partial final chunk. Written into an offset slice of a larger
    /// flat buffer, exactly as the prover's per-point loop does.
    #[test]
    fn test_quotient_chunk_products_into_matches_legacy() {
        type F = GoldilocksField;
        // (len, max_degree): 8/2 and 80/8 have all-full chunks, 7/3 has a partial final chunk.
        for &(len, max_degree) in &[(8usize, 2usize), (80, 8), (7, 3)] {
            let quotient_values = noncanonical_vec(len, 5);
            let expected = quotient_chunk_products_legacy(&quotient_values, max_degree);
            let num_chunks = len.div_ceil(max_degree);
            assert_eq!(expected.len(), num_chunks);

            let wrapped = quotient_chunk_products(&quotient_values, max_degree);
            assert_eq!(raw(&wrapped), raw(&expected));
            assert_eq!(wrapped.capacity(), num_chunks);

            // Flat-buffer form: write point products at an interior offset, as the prover does.
            let mut flat = vec![F::ZERO; 3 * num_chunks];
            quotient_chunk_products_into(
                &quotient_values,
                max_degree,
                &mut flat[num_chunks..2 * num_chunks],
            );
            assert_eq!(raw(&flat[num_chunks..2 * num_chunks]), raw(&expected));
            // Neighboring points' slots untouched.
            assert!(flat[..num_chunks]
                .iter()
                .chain(&flat[2 * num_chunks..])
                .all(|x| x.to_noncanonical_u64() == 0));
        }
    }

    /// `_into` must append exactly the legacy terms (raw representation equality),
    /// for both a full final chunk and a partial final chunk, and the allocating
    /// wrapper must preserve the historical exact capacity.
    #[test]
    fn test_check_partial_products_into_matches_legacy() {
        type F = GoldilocksField;
        // (len, max_degree): 8/2 has all-full chunks, 7/3 has a partial final chunk.
        for &(len, max_degree) in &[(8usize, 2usize), (7, 3)] {
            let numerators = noncanonical_vec(len, 1);
            let denominators = noncanonical_vec(len, 2);
            let num_partials = len.div_ceil(max_degree) - 1;
            let partials = noncanonical_vec(num_partials, 3);
            let z_x = F::from_noncanonical_u64(u64::MAX);
            let z_gx = F::from_noncanonical_u64(u64::MAX - 7);

            let expected = check_partial_products_legacy(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );
            assert_eq!(expected.len(), len.div_ceil(max_degree));

            let mut out = Vec::new();
            check_partial_products_into(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
                &mut out,
            );
            assert_eq!(raw(&out), raw(&expected));

            let wrapped = check_partial_products(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );
            assert_eq!(raw(&wrapped), raw(&expected));
            assert_eq!(wrapped.capacity(), len.div_ceil(max_degree));
        }
    }

    /// Two sequential `_into` calls (the per-point challenge loop) must append in
    /// order after an existing prefix, without clearing and without reallocating
    /// when spare capacity suffices.
    #[test]
    fn test_check_partial_products_into_appends_after_prefix() {
        type F = GoldilocksField;
        let len = 8usize;
        let max_degree = 2usize;
        let num_partials = len.div_ceil(max_degree) - 1;

        let mut expected_terms = Vec::new();
        let mut challenge_inputs = Vec::new();
        for challenge in 0..2u64 {
            let numerators = noncanonical_vec(len, 10 + challenge);
            let denominators = noncanonical_vec(len, 20 + challenge);
            let partials = noncanonical_vec(num_partials, 30 + challenge);
            let z_x = F::from_noncanonical_u64(u64::MAX - challenge);
            let z_gx = F::from_noncanonical_u64(u64::MAX - 100 - challenge);
            expected_terms.extend(check_partial_products_legacy(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            ));
            challenge_inputs.push((numerators, denominators, partials, z_x, z_gx));
        }

        let sentinel = F::from_noncanonical_u64(u64::MAX - 42);
        let mut out = Vec::with_capacity(1 + expected_terms.len());
        out.push(sentinel);
        let base_ptr = out.as_ptr();
        for (numerators, denominators, partials, z_x, z_gx) in &challenge_inputs {
            check_partial_products_into(
                numerators,
                denominators,
                partials,
                *z_x,
                *z_gx,
                max_degree,
                &mut out,
            );
        }
        // Sufficient spare capacity: appending must not have reallocated.
        assert_eq!(out.as_ptr(), base_ptr);
        assert_eq!(out[0].to_noncanonical_u64(), sentinel.to_noncanonical_u64());
        assert_eq!(raw(&out[1..]), raw(&expected_terms));

        // Forced growth: capacity exhausted by the sentinel alone. Prefix and
        // terms must survive the reallocation.
        let mut small = Vec::with_capacity(1);
        small.push(sentinel);
        let (numerators, denominators, partials, z_x, z_gx) = &challenge_inputs[0];
        check_partial_products_into(
            numerators,
            denominators,
            partials,
            *z_x,
            *z_gx,
            max_degree,
            &mut small,
        );
        assert_eq!(
            small[0].to_noncanonical_u64(),
            sentinel.to_noncanonical_u64()
        );
        assert_eq!(
            raw(&small[1..]),
            raw(&expected_terms[..len.div_ceil(max_degree)])
        );
    }
}
