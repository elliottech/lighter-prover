use alloc::vec::Vec;
use core::cmp::{max, min};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};
use unroll::unroll_for_loops;

use crate::packable::Packable;
use crate::packed::PackedField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

pub type FftRootTable<F> = Vec<Vec<F>>;

pub fn fft_root_table<F: Field>(n: usize) -> FftRootTable<F> {
    let lg_n = log2_strict(n);
    // bases[i] = g^2^i, for i = 0, ..., lg_n - 1
    let mut bases = Vec::with_capacity(lg_n);
    let mut base = F::primitive_root_of_unity(lg_n);
    bases.push(base);
    for _ in 1..lg_n {
        base = base.square(); // base = g^2^_
        bases.push(base);
    }

    let mut root_table = Vec::with_capacity(lg_n);
    for lg_m in 1..=lg_n {
        let half_m = 1 << (lg_m - 1);
        let base = bases[lg_n - lg_m];
        let root_row = base.powers().take(half_m.max(2)).collect();
        root_table.push(root_row);
    }
    root_table
}

#[inline]
fn fft_dispatch<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    let computed_root_table = root_table.is_none().then(|| fft_root_table(input.len()));
    let used_root_table = root_table.or(computed_root_table.as_ref()).unwrap();

    fft_classic(input, zero_factor.unwrap_or(0), used_root_table);
}

/// Computes an FFT in the caller-provided buffer.
///
/// This is equivalent to [`fft_with_options`], but permits buffers backed by
/// shared CPU/GPU memory to remain in place.
#[inline]
pub fn fft_in_place_with_options<F: Field>(
    buffer: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    fft_dispatch(buffer, zero_factor, root_table);
}

#[inline]
pub fn fft<F: Field>(poly: PolynomialCoeffs<F>) -> PolynomialValues<F> {
    fft_with_options(poly, None, None)
}

#[inline]
pub fn fft_with_options<F: Field>(
    poly: PolynomialCoeffs<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let PolynomialCoeffs { coeffs: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);
    PolynomialValues::new(buffer)
}

#[inline]
pub fn ifft<F: Field>(poly: PolynomialValues<F>) -> PolynomialCoeffs<F> {
    ifft_with_options(poly, None, None)
}

pub fn ifft_with_options<F: Field>(
    poly: PolynomialValues<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialCoeffs<F> {
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);

    let PolynomialValues { values: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);

    // We reverse all values except the first, and divide each by n.
    buffer[0] *= n_inv;
    buffer[n / 2] *= n_inv;
    for i in 1..(n / 2) {
        let j = n - i;
        let coeffs_i = buffer[j] * n_inv;
        let coeffs_j = buffer[i] * n_inv;
        buffer[i] = coeffs_i;
        buffer[j] = coeffs_j;
    }
    PolynomialCoeffs { coeffs: buffer }
}

/// Generic FFT implementation that works with both scalar and packed inputs.
#[unroll_for_loops]
fn fft_classic_simd<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    lg_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_packed_width = log2_strict(P::WIDTH); // 0 when P is a scalar.
    let packed_values = P::pack_slice_mut(values);
    let packed_n = packed_values.len();
    debug_assert!(packed_n == 1 << (lg_n - lg_packed_width));

    // Want the below for loop to unroll, hence the need for a literal.
    // This loop will not run when P is a scalar.
    assert!(lg_packed_width <= 4);
    for lg_half_m in 0..4 {
        if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
            // Intuitively, we split values into m slices: subarr[0], ..., subarr[m - 1]. Each of
            // those slices is split into two halves: subarr[j].left, subarr[j].right. We do
            // (subarr[j].left[k], subarr[j].right[k])
            //   := f(subarr[j].left[k], subarr[j].right[k], omega[k]),
            // where f(u, v, omega) = (u + omega * v, u - omega * v).
            let half_m = 1 << lg_half_m;

            // Set omega to root_table[lg_half_m][0..half_m] but repeated.
            let mut omega = P::default();
            for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                *omega_j = root_table[lg_half_m][j % half_m];
            }

            for k in (0..packed_n).step_by(2) {
                // We have two vectors and want to do math on pairs of adjacent elements (or for
                // lg_half_m > 0, pairs of adjacent blocks of elements). .interleave does the
                // appropriate shuffling and is its own inverse.
                let (u, v) = packed_values[k].interleave(packed_values[k + 1], half_m);
                let t = omega * v;
                (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
            }
        }
    }

    // We've already done the first lg_packed_width (if they were required) iterations.
    let s = max(r, lg_packed_width);
    fft_classic_simd_layers(packed_values, s, lg_n, root_table);
}

#[inline(always)]
fn fft_classic_simd_layers<P: PackedField>(
    packed_values: &mut [P],
    start: usize,
    end: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_packed_width = log2_strict(P::WIDTH);
    for lg_half_m in start..end {
        let lg_m = lg_half_m + 1;
        let m = 1 << lg_m; // Subarray size (in field elements).
        let packed_m = m >> lg_packed_width; // Subarray size (in vectors).
        let half_packed_m = packed_m / 2;
        debug_assert!(half_packed_m != 0);

        // Omega values for this iteration, as a slice of vectors.
        let omega_table = P::pack_slice(&root_table[lg_half_m]);
        for k in (0..packed_values.len()).step_by(packed_m) {
            for j in 0..half_packed_m {
                let omega = omega_table[j];
                let t = omega * packed_values[k + half_packed_m + j];
                let u = packed_values[k + j];
                packed_values[k + j] = u + t;
                packed_values[k + half_packed_m + j] = u - t;
            }
        }
    }
}

#[inline(always)]
fn fft_zero_padded_first_layer_block<P: PackedField>(
    packed_values: &mut [P],
    source_start: usize,
    nonzero_len: usize,
    destination: usize,
    packed_repeat: usize,
    omega_table: &[P],
) {
    debug_assert!(nonzero_len >= 2);

    // Expanding backwards ensures that every source pair is read before an earlier destination
    // can overwrite it. Destinations are vector-aligned because repeat >= the packed width.
    for pair in (0..nonzero_len / 2).rev() {
        let source = source_start + pair * 2;
        let u = packed_values[source / P::WIDTH].as_slice()[source % P::WIDTH];
        let v = packed_values[(source + 1) / P::WIDTH].as_slice()[(source + 1) % P::WIDTH];
        let u = P::from(u);
        let v = P::from(v);
        let pair_destination = destination + pair * 2 * packed_repeat;

        for j in 0..packed_repeat {
            let t = omega_table[j] * v;
            packed_values[pair_destination + j] = u + t;
            packed_values[pair_destination + packed_repeat + j] = u - t;
        }
    }
}

/// Expand a bit-reversed nonzero prefix and perform its first nontrivial FFT layer in one pass.
///
/// This is called only when each repeated run contains at least one packed vector.
fn fft_zero_padded_first_layer<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let repeat = 1 << r;
    let nonzero_len = values.len() >> r;
    debug_assert!(repeat >= P::WIDTH);

    let packed_repeat = repeat / P::WIDTH;
    let packed_values = P::pack_slice_mut(values);
    let omega_table = P::pack_slice(&root_table[r]);
    fft_zero_padded_first_layer_block(packed_values, 0, nonzero_len, 0, packed_repeat, omega_table);
}

fn fft_zero_padded_cache_blocks<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    lg_block_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let repeat = 1 << r;
    let block_len = 1 << lg_block_n;
    let nonzero_per_block = block_len >> r;
    let packed_repeat = repeat / P::WIDTH;
    let packed_block_len = block_len / P::WIDTH;
    let num_blocks = values.len() / block_len;
    let packed_values = P::pack_slice_mut(values);
    let omega_table = P::pack_slice(&root_table[r]);

    // Expand blocks from the end of the buffer so their output cannot clobber unread prefix
    // coefficients. Complete every block-local layer immediately while the block is still hot.
    for block in (0..num_blocks).rev() {
        let source_start = block * nonzero_per_block;
        let destination = block * packed_block_len;
        fft_zero_padded_first_layer_block(
            packed_values,
            source_start,
            nonzero_per_block,
            destination,
            packed_repeat,
            omega_table,
        );
        fft_classic_simd_layers(
            &mut packed_values[destination..destination + packed_block_len],
            r + 1,
            lg_block_n,
            root_table,
        );
    }
}

#[inline(never)]
fn prepare_zero_padded_fft<F: Field>(
    values: &mut [F],
    r: usize,
    lg_n: usize,
    lg_packed_width: usize,
    root_table: &FftRootTable<F>,
) -> usize {
    debug_assert!(r > 0 && r <= lg_n);

    // A zero-padded input only has n/2^r live coefficients. Bit-reversing the full buffer would
    // place those coefficients at multiples of 2^r, after which the skipped FFT layers copy each
    // one across its following 2^r-element run. Produce that exact state directly by reversing
    // just the live prefix.
    let repeat = 1 << r;
    let nonzero_len = values.len() >> r;
    reverse_index_bits_in_place(&mut values[..nonzero_len]);

    if r >= lg_packed_width && r < lg_n {
        // Keep values plus the largest local twiddle row within Apple Silicon's 128 KiB L1D.
        // Both 2^13 base-field and 2^12 quadratic-extension blocks use about 96 KiB.
        let lg_block_n = match core::mem::size_of::<F>() {
            0..=8 => 13,
            9..=16 => 12,
            _ => 11,
        };
        if r + 1 < lg_block_n && lg_block_n <= lg_n {
            fft_zero_padded_cache_blocks::<<F as Packable>::Packing>(
                values, r, lg_block_n, root_table,
            );
            lg_block_n
        } else {
            // Fuse the expansion with the first nontrivial layer, eliminating one full-buffer
            // write/read cycle while retaining the existing skipped-layer semantics.
            fft_zero_padded_first_layer::<<F as Packable>::Packing>(values, r, root_table);
            r + 1
        }
    } else {
        for i in (0..nonzero_len).rev() {
            let value = values[i];
            values[i * repeat..(i + 1) * repeat].fill(value);
        }
        r
    }
}

/// FFT implementation based on Section 32.3 of "Introduction to
/// Algorithms" by Cormen et al.
///
/// The parameter r signifies that the first 1/2^r of the entries of
/// input may be non-zero, but the last 1 - 1/2^r entries are
/// definitely zero.
pub(crate) fn fft_classic<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
    let n = values.len();
    let lg_n = log2_strict(n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    let first_layer = if r == 0 {
        reverse_index_bits_in_place(values);
        0
    } else {
        prepare_zero_padded_fft(values, r, lg_n, lg_packed_width, root_table)
    };

    if lg_n <= lg_packed_width {
        // Need the slice to be at least the width of two packed vectors for the vectorized version
        // to work. Do this tiny problem in scalar.
        fft_classic_simd::<F>(values, first_layer, lg_n, root_table);
    } else {
        fft_classic_simd::<<F as Packable>::Packing>(values, first_layer, lg_n, root_table);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cmp::{max, min};

    use plonky2_util::{log2_ceil, log2_strict, reverse_index_bits_in_place};
    use unroll::unroll_for_loops;

    use crate::extension::quadratic::QuadraticExtension;
    use crate::fft::{fft, fft_classic, fft_root_table, fft_with_options, ifft, FftRootTable};
    use crate::goldilocks_field::GoldilocksField;
    use crate::packable::Packable;
    use crate::packed::PackedField;
    use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
    use crate::types::Field;

    #[test]
    fn fft_and_ifft() {
        type F = GoldilocksField;
        let degree = 200usize;
        let degree_padded = degree.next_power_of_two();

        // Create a vector of coeffs; the first degree of them are
        // "random", the last degree_padded-degree of them are zero.
        let coeffs = (0..degree)
            .map(|i| F::from_canonical_usize(i * 1337 % 100))
            .chain(core::iter::repeat_n(F::ZERO, degree_padded - degree))
            .collect::<Vec<_>>();
        assert_eq!(coeffs.len(), degree_padded);
        let coefficients = PolynomialCoeffs { coeffs };

        let points = fft(coefficients.clone());
        assert_eq!(points, evaluate_naive(&coefficients));

        let interpolated_coefficients = ifft(points);
        for i in 0..degree {
            assert_eq!(interpolated_coefficients.coeffs[i], coefficients.coeffs[i]);
        }
        for i in degree..degree_padded {
            assert_eq!(interpolated_coefficients.coeffs[i], F::ZERO);
        }

        for r in 0..4 {
            // expand coefficients by factor 2^r by filling with zeros
            let zero_tail = coefficients.lde(r);
            assert_eq!(
                fft(zero_tail.clone()),
                fft_with_options(zero_tail, Some(r), None)
            );
        }
    }

    fn evaluate_naive<F: Field>(coefficients: &PolynomialCoeffs<F>) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_padded = 1 << log2_ceil(degree);

        let coefficients_padded = coefficients.padded(degree_padded);
        evaluate_naive_power_of_2(&coefficients_padded)
    }

    fn evaluate_naive_power_of_2<F: Field>(
        coefficients: &PolynomialCoeffs<F>,
    ) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_log = log2_strict(degree);

        let subgroup = F::two_adic_subgroup(degree_log);

        let values = subgroup
            .into_iter()
            .map(|x| evaluate_at_naive(coefficients, x))
            .collect();
        PolynomialValues::new(values)
    }

    fn evaluate_at_naive<F: Field>(coefficients: &PolynomialCoeffs<F>, point: F) -> F {
        let mut sum = F::ZERO;
        let mut point_power = F::ONE;
        for &c in &coefficients.coeffs {
            sum += c * point_power;
            point_power *= point;
        }
        sum
    }

    /// The original stage-major SIMD kernel, retained only as a differential-test oracle.
    #[unroll_for_loops]
    fn fft_classic_simd_reference<P: PackedField>(
        values: &mut [P::Scalar],
        r: usize,
        lg_n: usize,
        root_table: &FftRootTable<P::Scalar>,
    ) {
        let lg_packed_width = log2_strict(P::WIDTH);
        let packed_values = P::pack_slice_mut(values);
        let packed_n = packed_values.len();

        assert!(lg_packed_width <= 4);
        for lg_half_m in 0..4 {
            if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
                let half_m = 1 << lg_half_m;
                let mut omega = P::default();
                for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                    *omega_j = root_table[lg_half_m][j % half_m];
                }

                for k in (0..packed_n).step_by(2) {
                    let (u, v) = packed_values[k].interleave(packed_values[k + 1], half_m);
                    let t = omega * v;
                    (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
                }
            }
        }

        let s = max(r, lg_packed_width);
        for lg_half_m in s..lg_n {
            let packed_m = 1 << (lg_half_m + 1 - lg_packed_width);
            let half_packed_m = packed_m / 2;
            let omega_table = P::pack_slice(&root_table[lg_half_m]);

            for k in (0..packed_n).step_by(packed_m) {
                for j in 0..half_packed_m {
                    let omega = omega_table[j];
                    let t = omega * packed_values[k + half_packed_m + j];
                    let u = packed_values[k + j];
                    packed_values[k + j] = u + t;
                    packed_values[k + half_packed_m + j] = u - t;
                }
            }
        }
    }

    fn fft_classic_reference<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
        reverse_index_bits_in_place(values);
        let n = values.len();
        let lg_n = log2_strict(n);

        if r > 0 {
            let mask = !((1 << r) - 1);
            for i in 0..n {
                values[i] = values[i & mask];
            }
        }

        let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
        if lg_n <= lg_packed_width {
            fft_classic_simd_reference::<F>(values, r, lg_n, root_table);
        } else {
            fft_classic_simd_reference::<<F as Packable>::Packing>(values, r, lg_n, root_table);
        }
    }

    fn deterministic_value(i: usize) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(
            (i as u64).wrapping_mul(0x9e37_79b9).rotate_left(13) % 1_000_000_007,
        )
    }

    fn deterministic_values(n: usize) -> Vec<GoldilocksField> {
        (0..n).map(deterministic_value).collect()
    }

    fn ifft_reference(
        mut values: Vec<GoldilocksField>,
        root_table: &FftRootTable<GoldilocksField>,
    ) -> Vec<GoldilocksField> {
        let n = values.len();
        let lg_n = log2_strict(n);
        let n_inv = GoldilocksField::inverse_2exp(lg_n);
        fft_classic_reference(&mut values, 0, root_table);

        values[0] *= n_inv;
        values[n / 2] *= n_inv;
        for i in 1..n / 2 {
            let j = n - i;
            let coeff_i = values[j] * n_inv;
            let coeff_j = values[i] * n_inv;
            values[i] = coeff_i;
            values[j] = coeff_j;
        }
        values
    }

    #[test]
    fn fft_matches_stage_major_reference_across_sizes() {
        for lg_n in [0, 1, 2, 3, 4, 5, 8, 11, 12, 13, 14, 16, 18] {
            let n = 1 << lg_n;
            let roots = fft_root_table(n);
            let mut expected = deterministic_values(n);
            let mut actual = expected.clone();

            fft_classic_reference(&mut expected, 0, &roots);
            fft_classic(&mut actual, 0, &roots);
            assert_eq!(actual, expected, "FFT mismatch at 2^{lg_n}");
        }
    }

    #[test]
    fn coset_fft_and_ifft_match_stage_major_reference() {
        type F = GoldilocksField;
        let shift = F::coset_shift();

        for lg_n in [3, 8, 12, 15] {
            let n = 1 << lg_n;
            let roots = fft_root_table(n);
            let coeffs = deterministic_values(n);

            let mut expected_fft = coeffs
                .iter()
                .copied()
                .zip(shift.powers())
                .map(|(coefficient, power)| coefficient * power)
                .collect::<Vec<_>>();
            fft_classic_reference(&mut expected_fft, 0, &roots);
            let actual_fft = PolynomialCoeffs::new(coeffs).coset_fft(shift).values;
            assert_eq!(actual_fft, expected_fft, "coset FFT mismatch at 2^{lg_n}");

            let values = deterministic_values(n);
            let expected_ifft = ifft_reference(values.clone(), &roots)
                .into_iter()
                .zip(shift.inverse().powers())
                .map(|(coefficient, power)| coefficient * power)
                .collect::<Vec<_>>();
            let actual_ifft = PolynomialValues::new(values).coset_ifft(shift).coeffs;
            assert_eq!(
                actual_ifft, expected_ifft,
                "coset IFFT mismatch at 2^{lg_n}"
            );
        }
    }

    #[test]
    fn zero_padded_lde_shapes_match_full_and_shortcut_references() {
        type F = GoldilocksField;
        let shift = F::coset_shift();

        for (base_lg_n, r) in [(3, 1), (8, 2), (10, 3), (12, 3), (13, 3), (15, 3)] {
            let lg_n = base_lg_n + r;
            let n = 1 << lg_n;
            let nonzero_len = 1 << base_lg_n;
            let roots = fft_root_table(n);
            let padded = deterministic_values(nonzero_len)
                .into_iter()
                .chain(core::iter::repeat_n(F::ZERO, n - nonzero_len))
                .collect::<Vec<_>>();

            let mut expected_shortcut = padded.clone();
            fft_classic_reference(&mut expected_shortcut, r, &roots);
            let mut expected_full = padded.clone();
            fft_classic_reference(&mut expected_full, 0, &roots);
            assert_eq!(
                expected_shortcut, expected_full,
                "reference shortcut mismatch for 2^{base_lg_n} -> 2^{lg_n}"
            );

            let mut actual = padded.clone();
            fft_classic(&mut actual, r, &roots);
            assert_eq!(
                actual, expected_full,
                "zero-padded FFT mismatch for 2^{base_lg_n} -> 2^{lg_n}"
            );

            let mut expected_coset = padded
                .iter()
                .copied()
                .zip(shift.powers())
                .map(|(coefficient, power)| coefficient * power)
                .collect::<Vec<_>>();
            fft_classic_reference(&mut expected_coset, 0, &roots);
            let actual_coset = PolynomialCoeffs::new(padded)
                .coset_fft_with_options(shift, Some(r), Some(&roots))
                .values;
            assert_eq!(
                actual_coset, expected_coset,
                "zero-padded coset FFT mismatch for 2^{base_lg_n} -> 2^{lg_n}"
            );
        }
    }

    #[test]
    fn zero_padded_extension_fft_matches_reference() {
        type F = GoldilocksField;
        type FE = QuadraticExtension<F>;

        let base_lg_n = 10;
        let r = 3;
        let n = 1 << (base_lg_n + r);
        let nonzero_len = 1 << base_lg_n;
        let roots = fft_root_table(n);
        let mut actual = (0..nonzero_len)
            .map(|i| {
                QuadraticExtension([deterministic_value(i), deterministic_value(i + nonzero_len)])
            })
            .chain(core::iter::repeat_n(FE::ZERO, n - nonzero_len))
            .collect::<Vec<_>>();
        let mut expected = actual.clone();

        fft_classic_reference(&mut expected, r, &roots);
        fft_classic(&mut actual, r, &roots);
        assert_eq!(actual, expected);
    }
}
