#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::batch_util::batch_multiply_inplace;
use crate::field::extension::Extendable;
use crate::field::fft::{fft_in_place_with_options, FftRootTable};
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{ColumnStore, MerkleLeaves, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Route the whole commitment (NTT + hashing) through the GPU backend.
/// Official ranked A/B: submission 644c4257 (this on, over the 8.0011
/// frontier) scored 6.2323 despite a +4.6% controlled local win — the NTT
/// stages extend each tree's exclusive occupancy of the serialized GPU
/// stream, which is the ranked critical path. Keep off; hashing-only GPU
/// trees (`new_columns`) remain on.
const GPU_NTT_COMMITMENTS: bool = false;

/// Output layout for [`PolynomialBatch::fill_lde_batch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchLayout {
    /// `out[point * width + column]`
    PointMajor,
    /// `out[column * num_points + point]`
    PolyMajor,
}

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        if GPU_NTT_COMMITMENTS && !blinding {
            let value_columns: Vec<&[F]> = values.iter().map(|v| v.values.as_slice()).collect();
            if let Some((columns, digests, cap, coeff_columns)) = timed!(
                timing,
                "build Merkle tree",
                C::Hasher::try_build_commitment_from_values(&value_columns, rate_bits, cap_height,)
            ) {
                let degree = values[0].len();
                let merkle_tree = MerkleTree::from_prebuilt_columns(columns, digests, cap);
                return Self {
                    polynomials: coeff_columns
                        .into_iter()
                        .map(PolynomialCoeffs::new)
                        .collect(),
                    merkle_tree,
                    degree_log: log2_strict(degree),
                    rate_bits,
                    blinding,
                };
            }
        }

        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();

        if GPU_NTT_COMMITMENTS && !blinding {
            let coeff_columns: Vec<&[F]> =
                polynomials.iter().map(|p| p.coeffs.as_slice()).collect();
            if let Some((columns, digests, cap)) = timed!(
                timing,
                "build Merkle tree",
                C::Hasher::try_build_commitment_from_coeffs(&coeff_columns, rate_bits, cap_height,)
            ) {
                let merkle_tree = MerkleTree::from_prebuilt_columns(columns, digests, cap);
                return Self {
                    polynomials,
                    merkle_tree,
                    degree_log: log2_strict(degree),
                    rate_bits,
                    blinding,
                };
            }
        }

        // Ranked circuits are non-ZK. Materialize their CPU-computed LDEs for
        // the first time in retained shared Metal storage, then hash that same
        // buffer instead of copying every column through the pooled input.
        let lde_len = degree << rate_bits;
        if !blinding {
            if let Some(mut columns) =
                C::Hasher::try_allocate_merkle_tree_columns(polynomials.len(), lde_len, cap_height)
            {
                let initialized = timed!(
                    timing,
                    "FFT + blinding",
                    Self::fill_lde_column_store(
                        &mut columns,
                        &polynomials,
                        rate_bits,
                        fft_root_table,
                    )
                );
                if initialized {
                    let merkle_tree = timed!(
                        timing,
                        "build Merkle tree",
                        MerkleTree::new_column_store(columns, cap_height)
                    );
                    return Self {
                        polynomials,
                        merkle_tree,
                        degree_log: log2_strict(degree),
                        rate_bits,
                        blinding,
                    };
                }
            }
        }

        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_columns(lde_values, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();
        // Process-global cached coset-shift power table (bit-identical to
        // computing it here): the three commitments per proof share one table
        // per degree instead of each rebuilding the serial power chain.
        let coset_powers = crate::plonk::prover::precomputed::coset_shift_powers::<F>(degree);

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                // Fused zero-pad + shared-coset-powers multiply: one LDE-sized
                // buffer (instead of `lde()` + `coset_fft` each allocating),
                // with the packed batch multiply over the precomputed table.
                let lde_len = degree << rate_bits;
                let mut buffer = Vec::with_capacity(lde_len);
                buffer.extend_from_slice(&p.coeffs);
                if rate_bits == 0 || degree < 2 {
                    buffer.resize(lde_len, F::ZERO);
                } else {
                    // SAFETY: capacity is exactly `lde_len`. With `Some(rate_bits)`
                    // the zero-padded FFT reads only the first `degree` coefficients
                    // and writes every tail element before reading it (all expansion
                    // paths fill back-to-front), so the tail never needs the memset.
                    // `degree < 2` is excluded: the first-layer block writes nothing
                    // for a single live coefficient.
                    unsafe { buffer.set_len(lde_len) };
                }
                batch_multiply_inplace(&mut buffer[..degree], &coset_powers);
                PolynomialCoeffs::new(buffer)
                    .fft_with_options(Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    fn fill_lde_column_store(
        columns: &mut ColumnStore<F>,
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> bool {
        let degree = polynomials[0].len();
        let lde_len = degree << rate_bits;
        let coset_powers = crate::plonk::prover::precomputed::coset_shift_powers::<F>(degree);
        let Some(destinations) = columns.columns_mut() else {
            return false;
        };
        assert_eq!(destinations.len(), polynomials.len());
        assert!(destinations.iter().all(|column| column.len() == lde_len));

        destinations
            .into_par_iter()
            .zip(polynomials.par_iter())
            .for_each(|(destination, polynomial)| {
                assert_eq!(polynomial.len(), degree, "Polynomial degrees inconsistent");
                destination[..degree].copy_from_slice(&polynomial.coeffs);
                if rate_bits == 0 || degree < 2 {
                    destination[degree..].fill(F::ZERO);
                }
                // For a nontrivial zero-padded FFT, the expansion path writes
                // every tail element before reading it. This is the same
                // invariant used by `lde_values` to avoid a dead tail memset.
                batch_multiply_inplace(&mut destination[..degree], &coset_powers);
                fft_in_place_with_options(destination, Some(rate_bits), fft_root_table);
            });
        true
    }

    /// The number of value columns in this oracle, excluding any salt columns.
    pub(crate) fn lde_row_width(&self) -> usize {
        self.merkle_tree.leaf_width() - if self.blinding { SALT_SIZE } else { 0 }
    }

    /// Fetches LDE values at the `index * step`th point. Only available for
    /// row-major leaf storage; column-major oracles use [`Self::fill_lde_batch`].
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = self.merkle_tree.get(index);
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Gathers LDE values for a batch of points into `out`, in either layout,
    /// for both leaf storage modes. Point `k` (of `indices`) and column `c`
    /// (of `col_range`, indexing as in `get_lde_values(i, step)[c]`) land at
    /// `out[k * col_range.len() + (c - start)]` for `PointMajor` or
    /// `out[(c - start) * indices.len() + k]` for `PolyMajor`.
    pub(crate) fn fill_lde_batch(
        &self,
        indices: &[usize],
        step: usize,
        col_range: core::ops::Range<usize>,
        layout: BatchLayout,
        out: &mut Vec<F>,
    ) {
        if layout == BatchLayout::PolyMajor && step == 1 {
            if let Some(&index_start) = indices.first() {
                let contiguous = indices
                    .iter()
                    .enumerate()
                    .all(|(offset, &index)| index_start.checked_add(offset) == Some(index));
                if contiguous {
                    self.fill_lde_batch_contiguous(index_start, indices.len(), col_range, out);
                    return;
                }
            }
        }

        let n = indices.len();
        let start = col_range.start;
        let w = col_range.len();
        // `out` is per-worker scratch reused across the quotient batches, so it
        // already has the right length for all but the last (short) batch. Every
        // arm below writes all `n * w` cells before any is read:
        //   - Columns/PolyMajor: `ci` covers `0..w`, `k` covers `0..n`, writing
        //     each `ci * n + k` exactly once;
        //   - Columns/PointMajor: the same loop nest writes each `k * w + ci`;
        //   - Rows/PointMajor: each `k` copies a full `w`-element row into
        //     `out[k * w..(k + 1) * w]`;
        //   - Rows/PolyMajor: each `k` writes `ci * n + k` for every `ci` in
        //     `0..w` (`row.len() == w`).
        // So the zero-fill of a correctly sized buffer is a dead store: adjust
        // the length only (`resize` is a no-op when it already matches, and
        // still zero-initializes any newly created or grown scratch).
        out.resize(n * w, F::ZERO);
        match &self.merkle_tree.leaves {
            MerkleLeaves::Columns { columns, .. } => {
                for (ci, c) in col_range.enumerate() {
                    let column = columns.col(c);
                    match layout {
                        BatchLayout::PolyMajor => {
                            let destination = &mut out[ci * n..(ci + 1) * n];
                            for (k, &i) in indices.iter().enumerate() {
                                destination[k] = column[i * step];
                            }
                        }
                        BatchLayout::PointMajor => {
                            for (k, &i) in indices.iter().enumerate() {
                                out[k * w + ci] = column[i * step];
                            }
                        }
                    }
                }
            }
            MerkleLeaves::Rows { .. } => {
                for (k, &i) in indices.iter().enumerate() {
                    let row = &self.get_lde_values(i, step)[start..start + w];
                    match layout {
                        BatchLayout::PointMajor => {
                            out[k * w..(k + 1) * w].copy_from_slice(row);
                        }
                        BatchLayout::PolyMajor => {
                            for (ci, &value) in row.iter().enumerate() {
                                out[ci * n + k] = value;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Copies consecutive LDE points into a PolyMajor output buffer.
    ///
    /// Column-backed commitments use one contiguous slice copy per column,
    /// avoiding the indexed element loop in the quotient path. Row-backed
    /// commitments retain the same logical layout through a narrow fallback.
    pub(crate) fn fill_lde_batch_contiguous(
        &self,
        index_start: usize,
        n: usize,
        col_range: core::ops::Range<usize>,
        out: &mut Vec<F>,
    ) {
        let start = col_range.start;
        let w = col_range.len();
        out.resize(n * w, F::ZERO);

        match &self.merkle_tree.leaves {
            MerkleLeaves::Columns { columns, .. } => {
                let index_end = index_start
                    .checked_add(n)
                    .expect("contiguous LDE batch range overflow");
                for (ci, c) in col_range.enumerate() {
                    out[ci * n..(ci + 1) * n]
                        .copy_from_slice(&columns.col(c)[index_start..index_end]);
                }
            }
            MerkleLeaves::Rows { .. } => {
                for k in 0..n {
                    let row = &self.get_lde_values(index_start + k, 1)[start..start + w];
                    for (ci, &value) in row.iter().enumerate() {
                        out[ci * n + k] = value;
                    }
                }
            }
        }
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let leaf_size = self.lde_row_width();
        if let MerkleLeaves::Columns { columns, .. } = &self.merkle_tree.leaves {
            return (0..leaf_size)
                .map(|j| {
                    let column = columns.col(j);
                    let column_len = column.len();
                    debug_assert!(column_len.is_power_of_two());
                    let index_mask = column_len - 1;
                    let mut packed = P::ZEROS;
                    packed
                        .as_slice_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(l, packed_l)| {
                            // Packed STARK batches may straddle the end of the
                            // cyclic evaluation domain. The row-backed path
                            // wraps implicitly when `reverse_bits` discards
                            // bits above the domain width; retained natural-
                            // order columns must make the same wrap explicit.
                            *packed_l = column[((index_start + l) * step) & index_mask];
                        });
                    packed
                })
                .collect_vec();
        }

        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            // Fused (value-exact) form of:
            //   let quotient = composition_poly.divide_by_linear_padded_in_place(*point);
            //   alpha.shift_poly(&mut final_poly);
            //   final_poly += quotient;
            // (where the in-place division runs the classic `divide_by_linear`
            // Horner recurrence and leaves its top slot as the power-of-two
            // pad), writing straight into `final_poly`'s reusable buffer
            // instead of a division pass + shift pass + add pass.
            let shift = alpha.shift_factor();
            accumulate_linear_quotient(&mut final_poly, &composition_poly, *point, shift);
        }

        // `final_poly` is dead after this point, so pad it in place instead of
        // the clone-then-resize that `lde(&self)` performs.
        let mut lde_final_poly = final_poly;
        let live_coeffs = lde_final_poly.len();
        lde_final_poly.coeffs.resize(
            live_coeffs << fri_params.config.rate_bits,
            F::Extension::ZERO,
        );
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            // The top (1 - 1/2^rate_bits) of the padded coefficients are the
            // zeros written by the `resize` just above, so the FFT's zero-run
            // shortcut applies and the coset scaling over that tail is a
            // multiply-by-zero: scale only the `live_coeffs` prefix.
            coset_fft_zero_tail(
                &lde_final_poly,
                F::coset_shift().into(),
                live_coeffs,
                Some(fri_params.config.rate_bits),
                None,
            )
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        );

        fri_proof
    }
}

/// `coeffs.coset_fft_with_options(shift, zero_factor, root_table)` for a
/// coefficient vector whose entries from index `live` on are *known to be
/// zero* — the caller must have written those zeros itself (or otherwise hold a
/// proof of them), since the result is only equal to the classic path under
/// that precondition.
///
/// The classic path materializes `shift^i * c_i` for all `coeffs.len()`
/// coefficients. Where `c_i` is zero the product is zero, so this scales only
/// the live prefix and fills the tail with the very zeros the classic path
/// would have computed there; the FFT input is therefore element-wise
/// identical. With `rate_bits = 3` that deletes 7/8 of the extension-field
/// multiplies *and* 7/8 of the serial `powers()` chain, at one memset.
pub(crate) fn coset_fft_zero_tail<F: Field>(
    coeffs: &PolynomialCoeffs<F>,
    shift: F,
    live: usize,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let len = coeffs.len();
    debug_assert!(live <= len);
    debug_assert!(coeffs.coeffs[live..].iter().all(F::is_zero));
    let mut scaled = Vec::with_capacity(len);
    scaled.extend(
        shift
            .powers()
            .zip(&coeffs.coeffs[..live])
            .map(|(r, &c)| r * c),
    );
    scaled.resize(len, F::ZERO);
    PolynomialCoeffs::new(scaled).fft_with_options(zero_factor, root_table)
}

/// Folds one batch's quotient `(p(X) - p(z))/(X - z)` into the running FRI
/// `final_poly`, fusing the three passes of the previous code into one serial
/// sweep with no per-batch buffer traffic beyond it:
/// `divide_by_linear_padded_in_place` (the in-place Horner division whose top
/// slot is the power-of-two zero pad), `shift_poly`'s `final_poly *= shift`,
/// and `final_poly += quotient`.
///
/// Value-exact: the division's Horner recurrence (`acc = acc * z + c`)
/// produces the quotient's coefficients highest-first, so a descending sweep
/// can emit each coefficient in the same multiply-add order and combine it
/// with the shifted accumulator entry (`old * shift + q_i`) immediately. The
/// only dropped work is the recurrence's final step (the remainder `p(z)`,
/// which the division discards) and the reference's `+ ZERO` on the pad
/// slot / `ZERO * shift` on fresh slots, all of which leave values unchanged.
fn accumulate_linear_quotient<F: Field>(
    final_poly: &mut PolynomialCoeffs<F>,
    composition_poly: &PolynomialCoeffs<F>,
    z: F,
    shift: F,
) {
    let d = composition_poly.len();
    let coeffs = &composition_poly.coeffs;
    let buf = &mut final_poly.coeffs;
    // Entries past the padded quotient's length only see the shift.
    for l in buf.iter_mut().skip(d) {
        *l *= shift;
    }
    if buf.len() < d {
        buf.resize(d, F::ZERO);
    }
    if d == 0 {
        return;
    }
    // Highest slot: the quotient coefficient there is the explicit zero pad.
    buf[d - 1] *= shift;
    // Synthetic division, highest coefficient first: the quotient's
    // coefficient at `x^i` is the accumulator after absorbing `coeffs[i + 1]`.
    let mut acc = F::ZERO;
    for i in (0..d - 1).rev() {
        acc = acc * z + coeffs[i + 1];
        buf[i] = buf[i] * shift + acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::plonk::config::Poseidon2GoldilocksConfig;

    #[test]
    fn shared_coset_powers_match_per_polynomial_shifts() {
        const D: usize = 2;
        const RATE_BITS: usize = 3;
        type F = GoldilocksField;
        type C = Poseidon2GoldilocksConfig;

        let polynomials = (0..7)
            .map(|_| PolynomialCoeffs::new(F::rand_vec(1 << 8)))
            .collect::<Vec<_>>();
        let expected = polynomials
            .iter()
            .map(|polynomial| {
                polynomial
                    .lde(RATE_BITS)
                    .coset_fft_with_options(F::coset_shift(), Some(RATE_BITS), None)
                    .values
            })
            .collect::<Vec<_>>();
        let actual = PolynomialBatch::<F, C, D>::lde_values(&polynomials, RATE_BITS, false, None);

        assert_eq!(actual, expected);
    }

    /// Filling retained column storage must preserve the exact raw field-word
    /// sequence produced by the legacy Vec-backed CPU LDE. Seed the unused
    /// tail with nonzero words so this also checks the zero-padded FFT's
    /// write-before-read invariant rather than relying on fresh zeroed memory.
    #[test]
    fn retained_lde_fill_matches_legacy_vec_raw_words() {
        use crate::field::fft::fft_root_table;
        use crate::field::types::{Field64, PrimeField64};

        const D: usize = 2;
        type F = GoldilocksField;
        type C = Poseidon2GoldilocksConfig;

        for degree in [1usize, 2, 8, 64] {
            let polynomials = (0..5)
                .map(|column| {
                    PolynomialCoeffs::new(
                        (0..degree)
                            .map(|row| {
                                let raw = match (column * degree + row) & 7 {
                                    0 => 0,
                                    1 => 1,
                                    2 => F::ORDER - 1,
                                    3 => F::ORDER,
                                    4 => F::ORDER + 1,
                                    5 => u64::MAX,
                                    _ => ((column + 1) as u64)
                                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                        .wrapping_add(row as u64),
                                };
                                GoldilocksField(raw)
                            })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();

            for rate_bits in 0..=3 {
                let lde_len = degree << rate_bits;
                let roots = fft_root_table::<F>(lde_len);
                for root_table in [None, Some(&roots)] {
                    let expected = PolynomialBatch::<F, C, D>::lde_values(
                        &polynomials,
                        rate_bits,
                        false,
                        root_table,
                    );
                    let mut retained = ColumnStore::Owned(
                        (0..polynomials.len())
                            .map(|_| vec![GoldilocksField(u64::MAX); lde_len])
                            .collect(),
                    );
                    assert!(PolynomialBatch::<F, C, D>::fill_lde_column_store(
                        &mut retained,
                        &polynomials,
                        rate_bits,
                        root_table,
                    ));

                    for (column, expected) in expected.iter().enumerate() {
                        let actual = retained.col(column);
                        let actual_raw = actual
                            .iter()
                            .map(|value| value.to_noncanonical_u64())
                            .collect::<Vec<_>>();
                        let expected_raw = expected
                            .iter()
                            .map(|value| value.to_noncanonical_u64())
                            .collect::<Vec<_>>();
                        assert_eq!(
                            actual_raw, expected_raw,
                            "degree {degree}, rate_bits {rate_bits}, column {column}"
                        );
                    }
                }
            }
        }
    }

    /// Natural-order retained columns must preserve the row-backed oracle's
    /// implicit cyclic wrap when a packed STARK batch crosses the domain end.
    #[test]
    fn retained_columns_packed_gather_matches_rows_across_domain_wrap() {
        use crate::field::packable::Packable;
        use crate::field::types::PrimeField64;

        const D: usize = 2;
        type F = GoldilocksField;
        type C = Poseidon2GoldilocksConfig;
        type P = <F as Packable>::Packing;

        let degree_log = 4;
        let rate_bits = 2;
        let rows_len = 1usize << (degree_log + rate_bits);
        let columns = (0..5)
            .map(|column| {
                (0..rows_len)
                    .map(|row| {
                        GoldilocksField(
                            ((column + 1) as u64)
                                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                .wrapping_add(row as u64),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows = (0..rows_len)
            .map(|leaf| {
                let natural = reverse_bits(leaf, degree_log + rate_bits);
                columns
                    .iter()
                    .map(|column| column[natural])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let row_batch: PolynomialBatch<F, C, D> = PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::new(rows, 0),
            degree_log,
            rate_bits,
            blinding: false,
        };
        let column_batch: PolynomialBatch<F, C, D> = PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::new_columns(columns, 0),
            degree_log,
            rate_bits,
            blinding: false,
        };

        for step in [1usize, 2, 4] {
            let logical_len = rows_len / step;
            for index_start in [0, logical_len - 1, logical_len - P::WIDTH / 2] {
                let expected = row_batch.get_lde_values_packed::<P>(index_start, step);
                let actual = column_batch.get_lde_values_packed::<P>(index_start, step);
                let expected_raw = expected
                    .iter()
                    .flat_map(|packed| packed.as_slice())
                    .map(|value| value.to_noncanonical_u64())
                    .collect::<Vec<_>>();
                let actual_raw = actual
                    .iter()
                    .flat_map(|packed| packed.as_slice())
                    .map(|value| value.to_noncanonical_u64())
                    .collect::<Vec<_>>();
                assert_eq!(actual_raw, expected_raw, "step {step}, start {index_start}");
            }
        }
    }

    /// The zero-tail coset FFT must be value-identical to the classic
    /// full-length coset scaling for every polynomial whose coefficients from
    /// `live` on are zero — the precondition both call sites establish by
    /// writing those zeros themselves. Covers `live` exactly at the zero-run
    /// boundary (`n >> rate_bits`, the production case), `live` below it
    /// (extra zeros), and `live == n` (no tail at all), over both the base
    /// field and the quadratic extension actually used by FRI.
    #[test]
    fn coset_fft_zero_tail_matches_classic() {
        fn check<F: Field + Sample>() {
            for lg_n in [1usize, 2, 4, 6, 9] {
                let n = 1usize << lg_n;
                for rate_bits in 1..=3usize.min(lg_n) {
                    let support = n >> rate_bits;
                    for live in [support, support / 2, support.saturating_sub(1)] {
                        let mut coeffs = F::rand_vec(live);
                        coeffs.resize(n, F::ZERO);
                        let poly = PolynomialCoeffs::new(coeffs);
                        let shift = F::rand();
                        let expected = poly.coset_fft_with_options(shift, Some(rate_bits), None);
                        let actual = coset_fft_zero_tail(&poly, shift, live, Some(rate_bits), None);
                        assert_eq!(actual.values, expected.values);
                    }
                }
                // `rate_bits = 0` is the degenerate no-tail case.
                let poly = PolynomialCoeffs::new(F::rand_vec(n));
                let shift = F::rand();
                assert_eq!(
                    coset_fft_zero_tail(&poly, shift, n, None, None).values,
                    poly.coset_fft_with_options(shift, None, None).values
                );
            }
        }

        check::<GoldilocksField>();
        check::<<GoldilocksField as Extendable<2>>::Extension>();
    }

    /// The fused quotient accumulation must be bit-identical (raw u64
    /// representation) to the pre-fusion op sequence it replaces
    /// (`divide_by_linear` + explicit zero pad + `shift_poly` + add),
    /// including the empty-accumulator first batch and mismatched lengths.
    #[test]
    fn fused_quotient_accumulation_matches_reference() {
        use crate::field::extension::FieldExtension;
        use crate::field::types::PrimeField64;

        type F = <GoldilocksField as Extendable<2>>::Extension;

        fn raw(values: &[F]) -> Vec<u64> {
            values
                .iter()
                .flat_map(FieldExtension::<2>::to_basefield_array)
                .map(|c: GoldilocksField| c.to_noncanonical_u64())
                .collect()
        }

        for &(old_len, d) in &[
            (0usize, 1usize),
            (0, 8),
            (1, 1),
            (8, 8),
            (4, 8),
            (8, 4),
            (256, 256),
        ] {
            let initial = PolynomialCoeffs::new(F::rand_vec(old_len));
            let composition_poly = PolynomialCoeffs::new(F::rand_vec(d));
            let z = F::rand();
            let shift = F::rand();

            // Classic reference: the op sequence in `prove_openings` before
            // either in-place rewrite.
            let mut expected = initial.clone();
            let mut quotient = composition_poly.divide_by_linear(z);
            quotient.coeffs.push(F::ZERO); // pad back to power of two
            expected *= shift; // shift_poly
            expected += quotient;

            let mut actual = initial;
            accumulate_linear_quotient(&mut actual, &composition_poly, z, shift);

            assert_eq!(raw(&actual.coeffs), raw(&expected.coeffs));
        }
    }
}
