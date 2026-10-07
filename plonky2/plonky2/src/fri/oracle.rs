#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
use core::any::TypeId;

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::batch_util::batch_multiply_into;
use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::fft::{
    cached_fft_root_table, fft_in_place_with_options, fft_in_place_with_options_parallel,
    FftRootTable,
};
use crate::field::goldilocks_extensions::ext2_mul_add;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof_with_initial_order;
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

/// Optional compact copy of the even LDE rows of a column-major commitment
/// (row `k` of the companion is row `2k` of the commitment), retained in a
/// shared Metal buffer so the half-domain quotient kernels read contiguous
/// columns instead of a stride-2 gather over the full store. Filled either
/// by the LDE write (`from_coeffs_with_even_companion`) or lazily from an
/// already-retained full store (`get_or_fill_even_rows`) — the latter is
/// how deserialized `constants_sigmas` blobs grow a companion: the values
/// are circuit-fixed, so one fill is reused for every proof.
/// Absent on non-Metal targets.
pub struct EvenColumns<F> {
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    inner: std::sync::OnceLock<Option<crate::hash::poseidon2::metal::MetalColumns<F>>>,
    _phantom: core::marker::PhantomData<F>,
}

impl<F> Default for EvenColumns<F> {
    fn default() -> Self {
        Self {
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            inner: std::sync::OnceLock::new(),
            _phantom: core::marker::PhantomData,
        }
    }
}

// The companion is a derived cache of the commitment's own LDE, so it does
// not participate in equality; two batches with identical polynomials and
// trees are equal whether or not either retained a companion.
impl<F> PartialEq for EvenColumns<F> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<F> Eq for EvenColumns<F> {}
impl<F> core::fmt::Debug for EvenColumns<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("EvenColumns")
    }
}

impl<F> EvenColumns<F> {
    /// Eager companion produced alongside the LDE fill.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn from_ready(
        columns: Option<crate::hash::poseidon2::metal::MetalColumns<F>>,
    ) -> Self {
        let inner = std::sync::OnceLock::new();
        let _ = inner.set(columns);
        Self {
            inner,
            _phantom: core::marker::PhantomData,
        }
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn get(&self) -> Option<&crate::hash::poseidon2::metal::MetalColumns<F>> {
        self.inner.get().and_then(Option::as_ref)
    }

    /// Derive the even-row companion from a full-domain Metal column store.
    ///
    /// `companion[j][k] = full[j][2k]`. The kernel indexes both the wires
    /// companion and this buffer with stride `wires.rows`, so the compact
    /// constants must have the same row count as the compact wires. A
    /// degree-`< 4n` polynomial is determined by its `4n` even-coset
    /// samples; copying those samples does not change any value the
    /// full-domain kernel would have read at those rows.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn get_or_fill_even_rows(
        &self,
        full: &crate::hash::poseidon2::metal::MetalColumns<F>,
    ) -> Option<&crate::hash::poseidon2::metal::MetalColumns<F>>
    where
        F: crate::hash::hash_types::RichField,
    {
        self.inner
            .get_or_init(|| fill_even_companion_from_full(full))
            .as_ref()
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
#[inline]
unsafe fn copy_even_fast<F: crate::hash::hash_types::RichField>(
    dst: *mut F,
    src: &[F],
    half_len: usize,
) {
    if size_of::<F>() == 8 && core::mem::align_of::<F>() >= 8 {
        use core::arch::aarch64::*;
        let mut s = src.as_ptr() as *const u64;
        let mut d = dst as *mut u64;
        let chunks = half_len / 4;
        for _ in 0..chunks {
            let val = vld2q_u64(s);
            vst1q_u64(d, val.0);
            let val2 = vld2q_u64(s.add(4));
            vst1q_u64(d.add(2), val2.0);
            s = s.add(8);
            d = d.add(4);
        }
        let remainder = half_len % 4;
        for i in 0..remainder {
            *d.add(i) = *s.add(2 * i);
        }
    } else {
        let out = core::slice::from_raw_parts_mut(dst, half_len);
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = src[2 * k];
        }
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn fill_even_companion_from_full<F: crate::hash::hash_types::RichField>(
    full: &crate::hash::poseidon2::metal::MetalColumns<F>,
) -> Option<crate::hash::poseidon2::metal::MetalColumns<F>> {
    if full.rows() < 2 || full.rows() % 2 != 0 || full.cols() == 0 {
        return None;
    }
    let half = full.rows() / 2;
    let mut companion =
        crate::hash::poseidon2::metal::allocate_plain_columns::<F>(full.cols(), half)?;
    let dests = companion.columns_mut()?;
    dests.into_par_iter().enumerate().for_each(|(j, dest)| {
        let src = full.col(j);
        debug_assert_eq!(src.len(), half * 2);
        debug_assert_eq!(dest.len(), half);
        unsafe {
            copy_even_fast(dest.as_mut_ptr(), src, half);
        }
    });
    Some(companion)
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
    /// See [`EvenColumns`].
    pub even_columns: EvenColumns<F>,
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
            even_columns: EvenColumns::default(),
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
                    even_columns: EvenColumns::default(),
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
        Self::from_coeffs_with_even_companion(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
            false,
        )
    }

    /// [`Self::from_coeffs`], additionally retaining the compact even-row
    /// companion (see [`EvenColumns`]) when `want_even_companion` is set and
    /// the commitment lands in retained shared column storage. Values and
    /// tree are identical either way; the companion is a pure read-only copy.
    pub fn from_coeffs_with_even_companion(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        want_even_companion: bool,
    ) -> Self {
        let degree = polynomials[0].len();
        #[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
        let _ = want_even_companion;

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
                    even_columns: EvenColumns::default(),
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
                // Streamed exclusive-phase path: the backend absorbs each
                // group of eight LDE columns while the CPU computes the next
                // group, collapsing the serial FFT-then-hash commitment into
                // max(FFT, hash). Falls through to the classic fill + build
                // whenever the backend declines (the group fill below is the
                // same computation `fill_lde_column_store` performs, so a
                // partial fill is simply refilled).
                // Compact even-row companion (Metal only, on request): the
                // fill below writes row `2k` of every column into row `k` of
                // the companion right after that column's FFT, while the
                // column is still cache-resident.
                #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
                let mut even_companion = if want_even_companion && lde_len >= 2 && rate_bits >= 1 {
                    crate::hash::poseidon2::metal::allocate_plain_columns::<F>(
                        polynomials.len(),
                        lde_len / 2,
                    )
                } else {
                    None
                };
                #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
                let even_ptrs: Option<Vec<usize>> = even_companion.as_mut().and_then(|companion| {
                    companion
                        .columns_mut()
                        .map(|cols| cols.into_iter().map(|c| c.as_mut_ptr() as usize).collect())
                });
                #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
                let even_ptrs = &even_ptrs;
                #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
                let half_len = lde_len / 2;
                let copy_even = |_column: usize, _destination: &[F]| {
                    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
                    if let Some(ptrs) = even_ptrs {
                        // SAFETY: each column index is written by exactly one
                        // closure invocation (columns are disjoint), the
                        // companion outlives the fill, and `F` is plain data.
                        unsafe {
                            copy_even_fast(ptrs[_column] as *mut F, _destination, half_len);
                        }
                    }
                };
                let copy_even = &copy_even;
                let streamed = {
                    let coset_powers =
                        crate::plonk::prover::precomputed::coset_shift_powers::<F>(degree);
                    let polys = &polynomials;
                    C::Hasher::try_build_merkle_tree_column_store_streamed(
                        &columns,
                        cap_height,
                        &|group, destinations: &mut [&mut [F]]| {
                            destinations
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(k, destination)| {
                                    let polynomial = &polys[group * 8 + k];
                                    assert_eq!(
                                        polynomial.len(),
                                        degree,
                                        "Polynomial degrees inconsistent"
                                    );
                                    batch_multiply_into(
                                        &mut destination[..degree],
                                        &polynomial.coeffs,
                                        &coset_powers,
                                    );
                                    if rate_bits == 0 || degree < 2 {
                                        destination[degree..].fill(F::ZERO);
                                    }
                                    fft_in_place_with_options(
                                        destination,
                                        Some(rate_bits),
                                        fft_root_table,
                                    );
                                    copy_even(group * 8 + k, destination);
                                });
                        },
                    )
                };
                if let Some((level_digests, cap)) = streamed {
                    let merkle_tree = timed!(
                        timing,
                        "build Merkle tree",
                        MerkleTree::from_prebuilt_columns(columns, level_digests, cap)
                    );
                    return Self {
                        polynomials,
                        merkle_tree,
                        degree_log: log2_strict(degree),
                        rate_bits,
                        blinding,
                        even_columns: {
                            #[cfg(all(
                                feature = "std",
                                target_arch = "aarch64",
                                target_os = "macos"
                            ))]
                            {
                                EvenColumns::from_ready(even_companion)
                            }
                            #[cfg(not(all(
                                feature = "std",
                                target_arch = "aarch64",
                                target_os = "macos"
                            )))]
                            {
                                EvenColumns::default()
                            }
                        },
                    };
                }
                let initialized = timed!(
                    timing,
                    "FFT + blinding",
                    Self::fill_lde_column_store(
                        &mut columns,
                        &polynomials,
                        rate_bits,
                        fft_root_table,
                        copy_even,
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
                        even_columns: {
                            #[cfg(all(
                                feature = "std",
                                target_arch = "aarch64",
                                target_os = "macos"
                            ))]
                            {
                                EvenColumns::from_ready(even_companion)
                            }
                            #[cfg(not(all(
                                feature = "std",
                                target_arch = "aarch64",
                                target_os = "macos"
                            )))]
                            {
                                EvenColumns::default()
                            }
                        },
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
            even_columns: EvenColumns::default(),
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
                // SAFETY: capacity is exactly `lde_len >= degree` and `F` is
                // `Copy`. The `degree` live slots are *assigned* by the
                // `batch_multiply_into` immediately below, which never reads
                // its destination.
                unsafe { buffer.set_len(degree) };
                // Fused copy-and-scale, the twin of the column-store path: the
                // unscaled coefficient image the `extend_from_slice` used to
                // materialize here is never observed (only the coset-scaled
                // values reach the FFT), so writing the product directly deletes
                // one read+write pass over `degree` words per polynomial.
                // `batch_multiply_into` uses the same packed-prefix/scalar-tail
                // schedule as the `batch_multiply_inplace` it replaces, over
                // slices of equal length, so every word is bit-identical.
                batch_multiply_into(&mut buffer[..degree], &p.coeffs, &coset_powers);
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
        copy_even: &(dyn Fn(usize, &[F]) + Sync),
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
            .enumerate()
            .for_each(|(column, (destination, polynomial))| {
                assert_eq!(polynomial.len(), degree, "Polynomial degrees inconsistent");
                // Fused copy-and-scale: the unscaled coefficient image that
                // `copy_from_slice` used to materialize here is never observed —
                // the FFT reads only the coset-scaled values — so writing the
                // product directly deletes one full read+write pass over
                // `degree` words per column, per commitment, per proof. Word
                // values are unchanged: `batch_multiply_into` uses the same
                // packed-prefix/scalar-tail schedule as the
                // `batch_multiply_inplace` it replaces, and it never reads the
                // (possibly uninitialized) destination.
                batch_multiply_into(
                    &mut destination[..degree],
                    &polynomial.coeffs,
                    &coset_powers,
                );
                if rate_bits == 0 || degree < 2 {
                    destination[degree..].fill(F::ZERO);
                }
                // For a nontrivial zero-padded FFT, the expansion path writes
                // every tail element before reading it. This is the same
                // invariant used by `lde_values` to avoid a dead tail memset.
                fft_in_place_with_options(destination, Some(rate_bits), fft_root_table);
                copy_even(column, destination);
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

    /// Extracts the stride-`step` LDE values for a whole quotient domain of
    /// `q_domain` indices, column-major (`PolyMajor`): `out[c * q_domain + i]`
    /// = `columns[c][i * step]`. The constants/sigma columns are
    /// circuit-fixed, so the caller caches the result once per circuit and
    /// every subsequent proof's quotient gathers copy from it instead of
    /// re-walking the strided LDE.
    ///
    /// When `step == 1` (production: `rate_bits == quotient_degree_bits`) each
    /// column's contribution is a contiguous LDE prefix, so the strided map
    /// collapses to a per-column memcpy. Columns are independent and fan
    /// across the pool; the `step > 1` path is unchanged.
    pub fn extract_lde_batch_columns(
        &self,
        step: usize,
        col_range: core::ops::Range<usize>,
        q_domain: usize,
    ) -> Option<Vec<F>> {
        let w = col_range.len();
        match &self.merkle_tree.leaves {
            MerkleLeaves::Columns { columns, .. } => {
                if step == 1 {
                    // Contiguous: column[i * 1] == column[i] for i in 0..q_domain,
                    // so memcpy of the prefix is bit-identical to the strided map.
                    let mut out = Vec::with_capacity(w * q_domain);
                    // SAFETY: every of the `w * q_domain` slots is overwritten by
                    // `copy_from_slice` below before any is read. `F` is a plain
                    // field wrapper (any bit pattern is a valid `F`).
                    unsafe {
                        out.set_len(w * q_domain);
                    }
                    let col_start = col_range.start;
                    out.par_chunks_mut(q_domain)
                        .enumerate()
                        .for_each(|(ci, dest)| {
                            dest.copy_from_slice(&columns.col(col_start + ci)[..q_domain]);
                        });
                    Some(out)
                } else {
                    let mut out = Vec::with_capacity(w * q_domain);
                    for c in col_range {
                        let column = columns.col(c);
                        out.extend((0..q_domain).map(|i| column[i * step]));
                    }
                    Some(out)
                }
            }
            _ => None,
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
        for (batch_index, FriBatchInfo { point, polynomials }) in
            instance.batches.iter().enumerate()
        {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            // The label is formatted unconditionally, but `timing`'s `push` is
            // compiled out unless the `timing` feature is on — which it is not
            // here — so the `String` is allocated, written and dropped without
            // ever being read. A static label costs nothing and reads the same
            // in a timing build.
            // The first (and widest) batch can donate its composition buffer
            // to the quotient without changing the wide reduction's schedule:
            // there is no prior `final_poly` to preserve.
            // Later tiny batches stream fixed-size cache blocks directly
            // into the running quotient, avoiding another full-degree
            // composition allocation and write/read pass.
            if batch_index > 0 && polynomials.len() <= 16 {
                timed!(
                    timing,
                    "reduce and accumulate small opening batch",
                    alpha.accumulate_small_polys_base_linear_quotient(
                        polys_coeff,
                        &mut final_poly,
                        *point,
                    )
                );
                continue;
            }
            let composition_poly = timed!(
                timing,
                "reduce batch of polynomials",
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
            if final_poly.coeffs.is_empty() {
                // Multiplying the empty accumulator by `shift` is a no-op.
                // Reuse the wide composition allocation as the quotient and
                // remove the equally large zero-fill/output pass. Resetting
                // the reducing factor also avoids exponentiating alpha for a
                // factor that would only multiply the empty accumulator.
                alpha.reset();
                final_poly = composition_poly.divide_by_linear_padded_in_place(*point);
            } else {
                let shift = alpha.shift_factor();
                accumulate_linear_quotient(&mut final_poly, &composition_poly, *point, shift);
            }
        }

        // `final_poly` is dead after this point, so pad it in place instead of
        // the clone-then-resize that `lde(&self)` performs.
        let mut lde_final_poly = final_poly;
        let live_coeffs = lde_final_poly.len();
        let lde_len = live_coeffs << fri_params.config.rate_bits;
        // Only a prefix of the padded tail is ever read. `coset_fft_zero_tail`
        // consumes `[..live_coeffs]`; the first commit round then folds
        // `[..live_chunks * arity]`, i.e. `live_coeffs` rounded up to the first
        // round's arity, after which `coeffs` is replaced wholesale by the
        // folded vector and this buffer is dropped. Zero-fill exactly that read
        // window instead of the whole `8x` buffer — for a d18 block proof the
        // deleted memset is 28 MiB per proof (~7 MiB at d16), all of it either
        // immediately overwritten or never touched.
        let first_arity = 1usize
            << fri_params
                .reduction_arity_bits
                .first()
                .copied()
                .unwrap_or(0);
        let read_bound = live_coeffs.next_multiple_of(first_arity).min(lde_len);
        lde_final_poly.coeffs.reserve_exact(lde_len - live_coeffs);
        lde_final_poly.coeffs.resize(read_bound, F::Extension::ZERO);
        // SAFETY: `reserve_exact` guarantees capacity `lde_len`, and every
        // element in `[0, read_bound)` is initialized above. Elements beyond
        // `read_bound` are never read: the zero-tail FFT consumes only the live
        // prefix, and the fold consumes only `[..live_chunks * arity]`, which is
        // `<= read_bound`. Same pattern as the promoted `lde_values` fast path.
        unsafe { lde_final_poly.coeffs.set_len(lde_len) };
        let lde_final_values = timed!(
            timing,
            "perform final FFT",
            // The top (1 - 1/2^rate_bits) of the padded coefficients are the
            // zeros written by the `resize` just above, so the FFT's zero-run
            // shortcut applies and the coset scaling over that tail is a
            // multiply-by-zero: scale only the `live_coeffs` prefix.
            if fri_params.config.rate_bits == 3 {
                coset_fft_zero_tail_base_dif_bitrev::<F, D>(
                    &lde_final_poly,
                    F::coset_shift(),
                    live_coeffs,
                )
            } else {
                coset_fft_zero_tail_base::<F, D>(
                    &lde_final_poly,
                    F::coset_shift(),
                    live_coeffs,
                    Some(fri_params.config.rate_bits),
                    None,
                )
            }
        );

        let fri_proof = fri_proof_with_initial_order::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            fri_params.config.rate_bits == 3,
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
/// zero*. For the exact zero-padded FFT shape selected by `zero_factor`, the
/// FFT overwrites that entire tail before reading it, so those entries may be
/// left uninitialized instead.
///
/// The classic path materializes `shift^i * c_i` for all `coeffs.len()`
/// coefficients. Where `c_i` is zero the product is zero, so this scales only
/// the live prefix and fills the tail with the very zeros the classic path
/// would have computed there; the FFT input is therefore element-wise
/// identical. With `rate_bits = 3` that deletes 7/8 of the extension-field
/// multiplies *and* 7/8 of the serial `powers()` chain, at one memset.
///
/// Both production callers now pass a base-field shift and route through
/// [`coset_fft_zero_tail_base`]; this generic form is retained as that
/// function's differential oracle.
#[allow(dead_code)]
pub(crate) fn coset_fft_zero_tail<F: Field>(
    coeffs: &PolynomialCoeffs<F>,
    shift: F,
    live: usize,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let len = coeffs.len();
    debug_assert!(live <= len);
    let zero_tail_is_unread =
        matches!(zero_factor, Some(r) if r > 0 && live >= 2 && live == len >> r);
    debug_assert!(zero_tail_is_unread || coeffs.coeffs[live..].iter().all(F::is_zero));
    let mut scaled = Vec::with_capacity(len);
    // The FRI folding schedule reuses the same handful of shifts (`g`, then
    // `g^arity` per reduction round) for every proof of a given circuit, so
    // the successive-multiply power chain is built once per process and read
    // back here. `shift_powers` returns the very sequence `shift.powers()`
    // yields, so `powers[i]` is the same word `Powers` would have produced,
    // and the scaling loop below performs the same `r * c` products in the
    // same order. The loop stays serial.
    let powers = crate::plonk::prover::precomputed::shift_powers::<F>(shift, live);
    scaled.extend(
        powers[..live]
            .iter()
            .zip(&coeffs.coeffs[..live])
            .map(|(&r, &c)| r * c),
    );
    if zero_tail_is_unread {
        // SAFETY: capacity is exactly `len`. The zero-padded FFT reads only
        // the live prefix written above, then writes every tail element before
        // reading it (all expansion paths fill back-to-front). This is the same
        // invariant the `lde_values` fast path relies on.
        unsafe { scaled.set_len(len) };
    } else {
        scaled.resize(len, F::ZERO);
    }
    if crate::hash::poseidon2::is_exclusive_gpu_phase() {
        fft_in_place_with_options_parallel(&mut scaled, zero_factor, root_table);
    } else {
        fft_in_place_with_options(&mut scaled, zero_factor, root_table);
    }
    PolynomialValues::new(scaled)
}

/// [`coset_fft_zero_tail`] for the case every production caller actually has:
/// a coset shift that lives in the *base* field, embedded into the extension
/// only to be multiplied back out again.
///
/// Both prover call sites pass `F::coset_shift().into()` or a base
/// `MULTIPLICATIVE_GROUP_GENERATOR` power through `.into()`, so the power
/// chain `shift^i` never leaves the base field. Building it there instead
/// halves the table and replaces every step of the serial dependent multiply
/// chain with one base multiply.
///
/// Raw-representative-exact, not merely field-equal. For
/// `QuadraticExtension<GoldilocksField>` (which does not override
/// `scalar_mul`):
///  * `ext2_mul([r, 0], [c0, c1])` computes `c0` as `u160_times_7(0*c1) +
///    r*c0`, i.e. `reduce160(r*c0, 0)`, and `c1` as `reduce160(r*c1 + 0, 0)`;
///  * `reduce160(x, 0)` and `reduce128(x)` are the identical
///    `t0`/`t1`/`add_no_canonicalize_trashing_input` sequence, and
///    `GoldilocksField::mul` *is* `reduce128`;
///  * so `[r,0] * c` and `c.scalar_mul(r)` produce the same two `u64` words,
///    not merely congruent ones.
/// The same argument makes the base power chain word-identical to the
/// embedded one: every embedded power is `[x, 0]`, and the next step reduces
/// exactly `x * shift` with a zero high accumulator.
/// `base_scalar_mul_matches_embedded_extension_mul_raw_words` is the
/// differential; `coset_fft_zero_tail` stays as the generic test oracle.
pub(crate) fn coset_fft_zero_tail_base<F: Extendable<D>, const D: usize>(
    coeffs: &PolynomialCoeffs<F::Extension>,
    shift_base: F,
    live: usize,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F::Extension>>,
) -> PolynomialValues<F::Extension> {
    let len = coeffs.len();
    debug_assert!(live <= len);
    let zero_tail_is_unread =
        matches!(zero_factor, Some(r) if r > 0 && live >= 2 && live == len >> r);
    debug_assert!(
        zero_tail_is_unread
            || coeffs.coeffs[live..]
                .iter()
                .all(<F::Extension as Field>::is_zero)
    );
    let mut scaled = Vec::with_capacity(len);
    // The opening site's `live` is the circuit degree, so this is the very
    // table the LDE coset scaling already warmed for this process; the FRI
    // folding rounds reuse the arbitrary-shift cache exactly as before, only
    // over the base field.
    let powers = if shift_base == F::coset_shift() {
        let table = crate::plonk::prover::precomputed::coset_shift_powers::<F>(live);
        // A cache-key drift would hand back a shorter prefix and silently
        // truncate the scaling.
        assert_eq!(
            table.len(),
            live,
            "cached coset shift power table must cover the live prefix"
        );
        table
    } else {
        crate::plonk::prover::precomputed::shift_powers::<F>(shift_base, live)
    };
    scaled.extend(
        powers[..live]
            .iter()
            .zip(&coeffs.coeffs[..live])
            .map(|(&r, &c)| <F::Extension as FieldExtension<D>>::scalar_mul(&c, r)),
    );
    if zero_tail_is_unread {
        // SAFETY: identical to `coset_fft_zero_tail`; capacity is exactly
        // `len` and the zero-padded FFT writes every tail element before
        // reading it.
        unsafe { scaled.set_len(len) };
    } else {
        scaled.resize(len, <F::Extension as Field>::ZERO);
    }
    if crate::hash::poseidon2::is_exclusive_gpu_phase() {
        fft_in_place_with_options_parallel(&mut scaled, zero_factor, root_table);
    } else {
        fft_in_place_with_options(&mut scaled, zero_factor, root_table);
    }
    PolynomialValues::new(scaled)
}

/// Forward FFT in decimation-in-frequency order. Natural-order coefficients
/// become bit-reversed evaluations, which is exactly the row order consumed
/// by FRI's Merkle trees.
fn fft_dif_bitrev_base_twiddles<F: Extendable<D>, const D: usize>(values: &mut [F::Extension]) {
    let lg_n = log2_strict(values.len());
    assert!(
        lg_n <= F::TWO_ADICITY,
        "base-field DIF roots must exist at the transform size",
    );
    let roots = cached_fft_root_table::<F>(values.len());
    let core_top = lg_n.saturating_sub(3);
    #[cfg(feature = "diagnostic_profile")]
    let _expand = crate::util::profile::span("fri_dif", "dif_generic_expand3");
    for lg_m in (core_top + 1..=lg_n).rev() {
        let m = 1usize << lg_m;
        let half = m >> 1;
        let omega = &roots[lg_m - 1][..half];
        values.par_chunks_mut(m).for_each(|block| {
            let (low, high) = block.split_at_mut(half);
            for j in 0..half {
                let u = low[j];
                let v = high[j];
                low[j] = u + v;
                high[j] = <F::Extension as FieldExtension<D>>::scalar_mul(&(u - v), omega[j]);
            }
        });
    }
    #[cfg(feature = "diagnostic_profile")]
    drop(_expand);
    #[cfg(feature = "diagnostic_profile")]
    let _core = crate::util::profile::span("fri_dif", "dif_core");
    for lg_m in (1..=core_top).rev() {
        let m = 1usize << lg_m;
        let half = m >> 1;
        let omega = &roots[lg_m - 1][..half];
        values.par_chunks_mut(m).for_each(|block| {
            let (low, high) = block.split_at_mut(half);
            for j in 0..half {
                let u = low[j];
                let v = high[j];
                low[j] = u + v;
                high[j] = <F::Extension as FieldExtension<D>>::scalar_mul(&(u - v), omega[j]);
            }
        });
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dif_mul_base_ext2(
    value: QuadraticExtension<GoldilocksField>,
    twiddle: GoldilocksField,
) -> QuadraticExtension<GoldilocksField> {
    use crate::field::extension::quadratic::NeonGoldilocksField;

    QuadraticExtension((NeonGoldilocksField(value.0) * twiddle).0)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dif_layer_neon_ext2(
    values: &mut [QuadraticExtension<GoldilocksField>],
    lg_half_m: usize,
    omega: &[GoldilocksField],
) {
    use crate::field::extension::quadratic::NeonGoldilocksField;

    let half = 1usize << lg_half_m;
    let m = half << 1;
    for block in values.chunks_exact_mut(m) {
        let (low, high) = block.split_at_mut(half);
        for j in 0..half {
            let u = NeonGoldilocksField(low[j].0);
            let v = NeonGoldilocksField(high[j].0);
            low[j] = QuadraticExtension((u + v).0);
            high[j] = QuadraticExtension(((u - v) * omega[j]).0);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn dif_block_neon_ext2(
    values: &mut [QuadraticExtension<GoldilocksField>],
    roots: &FftRootTable<GoldilocksField>,
) {
    const CACHE_LG: usize = 12;

    let lg_n = log2_strict(values.len());
    let local_lg = lg_n.min(CACHE_LG);
    for lg_m in (local_lg + 1..=lg_n).rev() {
        dif_layer_neon_ext2(values, lg_m - 1, &roots[lg_m - 1]);
    }
    let local_len = 1usize << local_lg;
    for local in values.chunks_exact_mut(local_len) {
        for lg_m in (1..=local_lg).rev() {
            dif_layer_neon_ext2(local, lg_m - 1, &roots[lg_m - 1]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dif_expand_block_neon_ext2(
    source: &[QuadraticExtension<GoldilocksField>],
    destination: &mut [QuadraticExtension<GoldilocksField>],
    twiddles: &[GoldilocksField],
) {
    assert_eq!(source.len(), destination.len());
    assert_eq!(source.len(), twiddles.len());
    for ((destination, &source), &twiddle) in destination.iter_mut().zip(source).zip(twiddles) {
        *destination = dif_mul_base_ext2(source, twiddle);
    }
}

/// Shared destination for the rate-8 expansion. Parallel iteration `j`
/// exclusively owns offsets `j + block * live`, so concurrent writes never
/// alias. Keeping the pointer itself behind this wrapper also preserves its
/// provenance instead of round-tripping it through an integer.
#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
struct DifExt2Ptr(*mut QuadraticExtension<GoldilocksField>);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for DifExt2Ptr {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for DifExt2Ptr {}

#[cfg(target_arch = "aarch64")]
impl DifExt2Ptr {
    #[inline(always)]
    unsafe fn read(&self, index: usize) -> QuadraticExtension<GoldilocksField> {
        unsafe { self.0.add(index).read() }
    }

    #[inline(always)]
    unsafe fn write(&self, index: usize, value: QuadraticExtension<GoldilocksField>) {
        unsafe { self.0.add(index).write(value) }
    }
}

/// Expand the rate-8 live prefix through the first three DIF stages with a
/// zero-tail DAG, then run eight independent cache-blocked DIF cores.
#[cfg(target_arch = "aarch64")]
fn fft_dif_rate8_neon_ext2(values: &mut [QuadraticExtension<GoldilocksField>], live: usize) {
    // Folded-round codewords are far below the worker-pool crossover and run
    // inside an already parallel proof pipeline. Avoid launching a DAG task
    // per live coefficient plus eight nested core tasks for those transforms;
    // the initial large codewords still have enough work to amortize Rayon.
    const PARALLEL_MIN_EXT2: usize = 1 << 18;

    assert!(live > 0);
    assert_eq!(values.len(), live << 3);
    let lg_n = log2_strict(values.len());
    let roots = cached_fft_root_table::<GoldilocksField>(values.len());
    // Slice every row before the parallel writes so malformed root tables fail
    // at the function boundary rather than halfway through the expansion.
    let w1 = &roots[lg_n - 1][..live];
    let w2 = &roots[lg_n - 2][..live];
    let w3 = &roots[lg_n - 3][..live];

    #[cfg(feature = "diagnostic_profile")]
    let _expand = crate::util::profile::span("fri_dif", "dif_dag_expand3");
    let shared = DifExt2Ptr(values.as_mut_ptr());
    let expand = |j| {
        // SAFETY: every iteration owns one `j` column across all eight blocks.
        // Columns are disjoint even when the range executes in parallel, every
        // tail slot was initialized by the caller's first-touch fill, and
        // block 0 is read-only here.
        unsafe {
            let b0 = shared.read(j);
            let b4 = dif_mul_base_ext2(b0, w1[j]);
            let b2 = dif_mul_base_ext2(b0, w2[j]);
            let b6 = dif_mul_base_ext2(b4, w2[j]);
            let b1 = dif_mul_base_ext2(b0, w3[j]);
            let b3 = dif_mul_base_ext2(b2, w3[j]);
            let b5 = dif_mul_base_ext2(b4, w3[j]);
            let b7 = dif_mul_base_ext2(b6, w3[j]);
            shared.write(live + j, b1);
            shared.write(2 * live + j, b2);
            shared.write(3 * live + j, b3);
            shared.write(4 * live + j, b4);
            shared.write(5 * live + j, b5);
            shared.write(6 * live + j, b6);
            shared.write(7 * live + j, b7);
        }
    };
    if values.len() >= PARALLEL_MIN_EXT2 {
        (0..live).into_par_iter().for_each(expand);
    } else {
        // The small transforms run serially, so write one contiguous output
        // block at a time instead of maintaining seven distant store streams.
        // The pass order is the same seven-product dependency DAG used above;
        // only independent `j` iterations are transposed with block writes.
        let mut blocks = values.chunks_exact_mut(live);
        let block0 = blocks.next().unwrap();
        let block1 = blocks.next().unwrap();
        let block2 = blocks.next().unwrap();
        let block3 = blocks.next().unwrap();
        let block4 = blocks.next().unwrap();
        let block5 = blocks.next().unwrap();
        let block6 = blocks.next().unwrap();
        let block7 = blocks.next().unwrap();
        debug_assert!(blocks.next().is_none());

        dif_expand_block_neon_ext2(block0, block4, w1);
        dif_expand_block_neon_ext2(block0, block2, w2);
        dif_expand_block_neon_ext2(block4, block6, w2);
        dif_expand_block_neon_ext2(block0, block1, w3);
        dif_expand_block_neon_ext2(block2, block3, w3);
        dif_expand_block_neon_ext2(block4, block5, w3);
        dif_expand_block_neon_ext2(block6, block7, w3);
    }
    #[cfg(feature = "diagnostic_profile")]
    drop(_expand);

    #[cfg(feature = "diagnostic_profile")]
    let _core = crate::util::profile::span("fri_dif", "dif_neon_core");
    if values.len() >= PARALLEL_MIN_EXT2 {
        values
            .par_chunks_mut(live)
            .for_each(|block| dif_block_neon_ext2(block, &roots));
    } else {
        values
            .chunks_mut(live)
            .for_each(|block| dif_block_neon_ext2(block, &roots));
    }
}

/// Rate-8 FRI producer prototype: retain natural coefficient order and emit
/// bit-reversed evaluation order directly with a DIF transform.
pub(crate) fn coset_fft_zero_tail_base_dif_bitrev<F: Extendable<D>, const D: usize>(
    coeffs: &PolynomialCoeffs<F::Extension>,
    shift_base: F,
    live: usize,
) -> PolynomialValues<F::Extension> {
    let len = coeffs.len();
    assert_eq!(
        live,
        len >> 3,
        "DIF prototype is specialized to rate_bits=3"
    );
    let powers = if shift_base == F::coset_shift() {
        crate::plonk::prover::precomputed::coset_shift_powers::<F>(live)
    } else {
        crate::plonk::prover::precomputed::shift_powers::<F>(shift_base, live)
    };
    let mut scaled: Vec<F::Extension> = Vec::with_capacity(len);
    #[cfg(feature = "diagnostic_profile")]
    let _scale = crate::util::profile::span("fri_dif", "dif_coset_scale");
    scaled.extend(
        powers[..live]
            .iter()
            .zip(&coeffs.coeffs[..live])
            .map(|(&r, &c)| <F::Extension as FieldExtension<D>>::scalar_mul(&c, r)),
    );
    #[cfg(feature = "diagnostic_profile")]
    drop(_scale);
    #[cfg(target_arch = "aarch64")]
    if TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
        && TypeId::of::<F::Extension>() == TypeId::of::<QuadraticExtension<GoldilocksField>>()
    {
        // SAFETY: the TypeId checks prove the exact base and extension types.
        // The explicit fill is intentionally retained: it first-touches the
        // seven output blocks sequentially. Letting the strided DAG fault those
        // pages in itself measured much slower despite deleting the memset.
        #[cfg(feature = "diagnostic_profile")]
        let _zero_fill = crate::util::profile::span("fri_dif", "dif_zero_fill");
        scaled.resize(len, <F::Extension as Field>::ZERO);
        #[cfg(feature = "diagnostic_profile")]
        drop(_zero_fill);
        let ext2 = unsafe {
            core::slice::from_raw_parts_mut(
                scaled
                    .as_mut_ptr()
                    .cast::<QuadraticExtension<GoldilocksField>>(),
                len,
            )
        };
        fft_dif_rate8_neon_ext2(ext2, live);
        return PolynomialValues::new(scaled);
    }
    #[cfg(feature = "diagnostic_profile")]
    let _zero_fill = crate::util::profile::span("fri_dif", "dif_zero_fill");
    scaled.resize(len, <F::Extension as Field>::ZERO);
    #[cfg(feature = "diagnostic_profile")]
    drop(_zero_fill);
    fft_dif_bitrev_base_twiddles::<F, D>(&mut scaled);
    PolynomialValues::new(scaled)
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
    // Production Goldilocks-quadratic fast path. Each synthetic-division step
    // is two multiply-accumulates, and the separate spelling pays four
    // `reduce160` per step (two for the delayed extension multiply, two more
    // for the canonicalizing extension add). `ext2_mul_add` folds the addend
    // into the multiply's accumulators for exactly two per step. The result
    // is the same field element with a representative that may be the
    // canonical one where the separate spelling's `Add` left a `+ p` on it
    // (see `ext2_mul_add_matches_mul_then_add_as_field_values`); noncanonical
    // representatives are ordinary here and every consumer downstream is
    // congruence-preserving.
    if TypeId::of::<F>() == TypeId::of::<QuadraticExtension<GoldilocksField>>() {
        // SAFETY: the `TypeId` comparison proves `F` is exactly
        // `QuadraticExtension<GoldilocksField>`, so the casts below preserve
        // layout, length and alignment, and the reads are of an initialized
        // `Copy` value of that same type.
        let buf_q = unsafe {
            core::slice::from_raw_parts_mut(
                buf.as_mut_ptr()
                    .cast::<QuadraticExtension<GoldilocksField>>(),
                buf.len(),
            )
        };
        let coeffs_q = unsafe {
            core::slice::from_raw_parts(
                coeffs
                    .as_ptr()
                    .cast::<QuadraticExtension<GoldilocksField>>(),
                coeffs.len(),
            )
        };
        let z_q = unsafe { *(&z as *const F).cast::<QuadraticExtension<GoldilocksField>>() };
        let shift_q =
            unsafe { *(&shift as *const F).cast::<QuadraticExtension<GoldilocksField>>() };
        let mut acc = QuadraticExtension::<GoldilocksField>::ZERO;
        for i in (0..d - 1).rev() {
            acc = ext2_mul_add(acc, z_q, coeffs_q[i + 1]);
            buf_q[i] = ext2_mul_add(buf_q[i], shift_q, acc);
        }
        return;
    }
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

    #[test]
    fn zero_tail_dif_emits_classic_bitrev_order() {
        use crate::field::types::{Field64, PrimeField64};

        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        fn raw(x: FE) -> [u64; 2] {
            let limbs = <FE as FieldExtension<D>>::to_basefield_array(&x);
            [
                limbs[0].to_noncanonical_u64(),
                limbs[1].to_noncanonical_u64(),
            ]
        }

        // 13 and 16 cross the 2^12 cache-block boundary used by the optimized
        // core; smaller sizes exercise the direct local-layer path.
        for lg_n in [4usize, 6, 9, 11, 13, 16] {
            let n = 1usize << lg_n;
            let live = n >> 3;
            let edges = [
                F::ZERO,
                F::ONE,
                GoldilocksField(u64::MAX),
                GoldilocksField(F::ORDER),
                GoldilocksField(F::ORDER - 1),
                GoldilocksField(0xffff_ffff),
            ];
            let adversarial = (0..live)
                .map(|i| {
                    FE::from_basefield_array([
                        edges[i % edges.len()],
                        edges[(i * 3 + 1) % edges.len()],
                    ])
                })
                .collect::<Vec<_>>();
            for mut coeffs in [FE::rand_vec(live), adversarial] {
                coeffs.resize(n, FE::ZERO);
                let poly = PolynomialCoeffs::new(coeffs);
                for shift in [
                    F::coset_shift(),
                    F::MULTIPLICATIVE_GROUP_GENERATOR,
                    F::MULTIPLICATIVE_GROUP_GENERATOR.exp_u64(16),
                ] {
                    let natural =
                        coset_fft_zero_tail_base::<F, D>(&poly, shift, live, Some(3), None);
                    let bitrev = coset_fft_zero_tail_base_dif_bitrev::<F, D>(&poly, shift, live);
                    for i in 0..n {
                        assert_eq!(
                            bitrev.values[i],
                            natural.values[reverse_bits(i, lg_n)],
                            "lg_n={lg_n} shift={shift:?} i={i}",
                        );
                        assert_eq!(
                            raw(bitrev.values[i]),
                            raw(natural.values[reverse_bits(i, lg_n)]),
                            "raw lg_n={lg_n} shift={shift:?} i={i}",
                        );
                    }
                }
            }
        }
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
                        &|_, _| {},
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
            even_columns: EvenColumns::default(),
        };
        let column_batch: PolynomialBatch<F, C, D> = PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::new_columns(columns, 0),
            degree_log,
            rate_bits,
            blinding: false,
            even_columns: EvenColumns::default(),
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

    /// A2's whole claim in one place: for the embedded shift both prover call
    /// sites actually pass, `c.scalar_mul(r)` and `[r, 0] * c` produce the
    /// *same raw `u64` words*, not merely the same field element. Compared on
    /// `to_noncanonical_u64` limbs, never on the field type (whose `PartialEq`
    /// canonicalizes and would hide exactly this bug class). The last case is
    /// the sabotage control: one limb of the reference is perturbed and the
    /// differential must fail.
    #[test]
    fn base_scalar_mul_matches_embedded_extension_mul_raw_words() {
        use crate::field::extension::FieldExtension;
        use crate::field::types::{Field64, PrimeField64, Sample};

        type FE = <GoldilocksField as Extendable<2>>::Extension;

        fn raw(x: FE) -> [u64; 2] {
            let limbs: [GoldilocksField; 2] = <FE as FieldExtension<2>>::to_basefield_array(&x);
            [
                limbs[0].to_noncanonical_u64(),
                limbs[1].to_noncanonical_u64(),
            ]
        }

        // Noncanonical and boundary representatives are the interesting
        // inputs: `reduce128`/`reduce160` may leave a `+ p` on the word.
        let edge_base = [
            GoldilocksField::ZERO,
            GoldilocksField::ONE,
            GoldilocksField(u64::MAX),
            GoldilocksField(GoldilocksField::ORDER),
            GoldilocksField(GoldilocksField::ORDER - 1),
            GoldilocksField(0xffff_ffff),
            GoldilocksField::coset_shift(),
            GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR,
        ];
        let mut scalars: Vec<GoldilocksField> = edge_base.to_vec();
        scalars.extend(GoldilocksField::rand_vec(64));

        let mut values: Vec<FE> = Vec::new();
        for &a in &edge_base {
            for &b in &edge_base {
                values.push(FE::from_basefield_array([a, b]));
            }
        }
        values.extend(FE::rand_vec(64));

        let mut compared = 0usize;
        for &r in &scalars {
            let embedded: FE = r.into();
            for &c in &values {
                // Reference: the embedded extension multiply this replaces,
                // in the same operand order the old scaling loop used.
                let expected = raw(embedded * c);
                let actual = raw(<FE as FieldExtension<2>>::scalar_mul(&c, r));
                assert_eq!(actual, expected, "scalar_mul diverges for r={r:?} c={c:?}");
                compared += 1;
            }
        }
        assert!(compared >= 4096, "differential ran on too few pairs");

        // The base power chain must equal the embedded one word for word,
        // which is what lets the cached base table stand in for the
        // extension table.
        for &shift in &[
            GoldilocksField::coset_shift(),
            GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR,
            GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.exp_u64(16),
        ] {
            let base_chain: Vec<GoldilocksField> = shift.powers().take(512).collect();
            let embedded_chain: Vec<FE> = FE::from(shift).powers().take(512).collect();
            for (i, (&b, &e)) in base_chain.iter().zip(&embedded_chain).enumerate() {
                assert_eq!(
                    [b.to_noncanonical_u64(), 0],
                    raw(e),
                    "power chain diverges at {i}"
                );
            }
        }

        // Sabotage control: a differential that has never failed is not
        // evidence. Perturb one limb of the reference and require a mismatch.
        let r = scalars[3];
        let c = values[7];
        let mut sabotaged = raw(<FE as FieldExtension<2>>::scalar_mul(&c, r));
        sabotaged[1] ^= 1;
        assert_ne!(
            sabotaged,
            raw(FE::from(r) * c),
            "sabotage control did not trip: the differential cannot detect a limb flip"
        );
    }

    /// A2 end to end, over the real shapes: the base-shift zero-tail coset FFT
    /// must produce the identical raw words as the generic extension-shift one
    /// it replaces, including at the production `rate_bits = 3` / `live ==
    /// n >> 3` shape and at the FRI folding rounds' arbitrary base shifts.
    #[test]
    fn coset_fft_zero_tail_base_matches_generic_raw_words() {
        use crate::field::extension::FieldExtension;
        use crate::field::types::{PrimeField64, Sample};

        type FE = <GoldilocksField as Extendable<2>>::Extension;

        fn raw(values: &[FE]) -> Vec<u64> {
            let mut out = Vec::with_capacity(values.len() * 2);
            for value in values {
                let limbs: [GoldilocksField; 2] =
                    <FE as FieldExtension<2>>::to_basefield_array(value);
                out.push(limbs[0].to_noncanonical_u64());
                out.push(limbs[1].to_noncanonical_u64());
            }
            out
        }

        let shifts = [
            GoldilocksField::coset_shift(),
            GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR,
            GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.exp_u64(16),
            GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.exp_u64(256),
        ];
        let mut cases = 0usize;
        for lg_n in [4usize, 6, 9, 11] {
            let n = 1usize << lg_n;
            for rate_bits in 1..=3usize.min(lg_n) {
                let support = n >> rate_bits;
                for live in [support, support / 2] {
                    if live < 2 {
                        continue;
                    }
                    let mut coeffs = FE::rand_vec(live);
                    coeffs.resize(n, FE::ZERO);
                    let poly = PolynomialCoeffs::new(coeffs);
                    for &shift in &shifts {
                        let expected =
                            coset_fft_zero_tail(&poly, shift.into(), live, Some(rate_bits), None);
                        let actual = coset_fft_zero_tail_base::<GoldilocksField, 2>(
                            &poly,
                            shift,
                            live,
                            Some(rate_bits),
                            None,
                        );
                        assert_eq!(
                            raw(&actual.values),
                            raw(&expected.values),
                            "lg_n={lg_n} rate_bits={rate_bits} live={live}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases >= 24, "differential ran on too few shapes");

        // Sabotage control: shift the base table by one step and require the
        // raw-word comparison to fail.
        let n = 1usize << 9;
        let live = n >> 3;
        let mut coeffs = FE::rand_vec(live);
        coeffs.resize(n, FE::ZERO);
        let poly = PolynomialCoeffs::new(coeffs);
        let shift = GoldilocksField::coset_shift();
        let good =
            coset_fft_zero_tail_base::<GoldilocksField, 2>(&poly, shift, live, Some(3), None);
        let sabotaged = coset_fft_zero_tail(&poly, (shift * shift).into(), live, Some(3), None);
        assert_ne!(
            raw(&good.values),
            raw(&sabotaged.values),
            "sabotage control did not trip: the differential cannot detect a wrong shift"
        );
    }

    /// The fused quotient accumulation must be bit-identical (raw u64
    /// representation) to the pre-fusion op sequences it replaces: both the
    /// classic reference (`divide_by_linear` + explicit zero pad +
    /// `shift_poly` + add) and this tree's in-place variant
    /// (`divide_by_linear_padded_in_place` + `shift_poly` + add), including
    /// the empty-accumulator first batch and mismatched lengths.
    #[test]
    fn fused_quotient_accumulation_matches_reference() {
        use crate::field::extension::FieldExtension;
        use crate::field::types::PrimeField64;

        type F = <GoldilocksField as Extendable<2>>::Extension;

        fn raw(values: &[F]) -> Vec<u64> {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
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

            // This tree's exact pre-fusion sequence: the consuming in-place
            // division (top slot already the pad) + shift_poly + add.
            let mut expected_in_place = initial.clone();
            let quotient_in_place = composition_poly.clone().divide_by_linear_padded_in_place(z);
            expected_in_place *= shift; // shift_poly
            expected_in_place += quotient_in_place;

            let mut actual = initial;
            accumulate_linear_quotient(&mut actual, &composition_poly, z, shift);

            assert_eq!(raw(&actual.coeffs), raw(&expected.coeffs));
            assert_eq!(raw(&actual.coeffs), raw(&expected_in_place.coeffs));
        }
    }

    /// Streaming a small opening batch must preserve the complete reduction,
    /// alpha-power shift and synthetic-division recurrence. In particular,
    /// block boundaries run in descending order, because Horner's accumulator
    /// carries from the high block into the next lower block.
    #[test]
    fn streamed_small_opening_batch_matches_materialized_path() {
        use crate::field::extension::FieldExtension;
        use crate::field::types::PrimeField64;

        type BF = GoldilocksField;
        type F = <BF as Extendable<2>>::Extension;

        fn raw(values: &[F]) -> Vec<u64> {
            values
                .iter()
                .flat_map(|x| FieldExtension::<2>::to_basefield_array(x))
                .map(|c: BF| c.to_noncanonical_u64())
                .collect()
        }

        for &(num_polys, degree, old_len) in &[
            (1usize, 1usize, 0usize),
            (2, 8, 8),
            (2, 2048, 2048),
            (2, 2049, 2049),
            (2, 4097, 4097),
            (16, 257, 300),
        ] {
            let polys = (0..num_polys)
                .map(|_| PolynomialCoeffs::new(BF::rand_vec(degree)))
                .collect::<Vec<_>>();
            let initial = PolynomialCoeffs::new(F::rand_vec(old_len));
            let base = F::rand();
            let z = F::rand();

            let mut reference_factor = ReducingFactor::new(base);
            let composition = reference_factor.reduce_polys_base::<BF, 2>(polys.iter());
            let shift = reference_factor.shift_factor();
            let mut expected = initial.clone();
            accumulate_linear_quotient(&mut expected, &composition, z, shift);

            let mut streamed_factor = ReducingFactor::new(base);
            let mut actual = initial;
            streamed_factor.accumulate_small_polys_base_linear_quotient::<BF, 2>(
                polys.iter(),
                &mut actual,
                z,
            );

            assert_eq!(raw(&actual.coeffs), raw(&expected.coeffs));
        }
    }

    /// Differential oracle for skipping `constants_sigmas_quotient_cache` at
    /// `step == 1` (`try_build_with_options` / `circuit::embed`).
    ///
    /// The cached quotient path reads `cache[ci * domain + start..][..n]`; the
    /// uncached path calls `fill_lde_batch(indices, 1, range, PolyMajor, out)`
    /// and reads `out[ci * n..][..n]`. This asserts those two byte sequences
    /// are identical for every column of both ranges, over every batch the
    /// quotient loop can produce (including a short final batch), using RANDOM
    /// values — nothing about a real circuit's constants or sigmas is assumed.
    #[test]
    fn quotient_cache_reads_match_uncached_lde_batch_raw_words() {
        use crate::field::types::PrimeField64;

        const D: usize = 2;
        const RATE_BITS: usize = 3;
        const CAP_HEIGHT: usize = 1;
        type F = GoldilocksField;
        type C = Poseidon2GoldilocksConfig;

        // Mirrors the shape the ranked circuits use: a few "constants" columns
        // followed by the routed-wire sigma columns.
        for (degree_bits, num_constants, num_routed) in [(4usize, 2usize, 5usize), (6, 6, 11)] {
            let degree = 1usize << degree_bits;
            let width = num_constants + num_routed;
            let values = (0..width)
                .map(|_| PolynomialValues::new(F::rand_vec(degree)))
                .collect::<Vec<_>>();
            let batch = PolynomialBatch::<F, C, D>::from_values(
                values,
                RATE_BITS,
                false,
                CAP_HEIGHT,
                &mut TimingTree::default(),
                None,
            );
            // `step == 1` is exactly the configuration under test: rate_bits
            // equals the quotient degree bits, so the quotient domain is the
            // full LDE.
            let step = 1usize;
            let domain = degree << RATE_BITS;
            let constants_range = 0..num_constants;
            let sigmas_range = num_constants..width;

            // The cache, assembled exactly as the builder assembled it.
            let mut cache = batch
                .extract_lde_batch_columns(step, constants_range.clone(), domain)
                .expect("column-backed commitment must extract");
            cache.extend(
                batch
                    .extract_lde_batch_columns(step, sigmas_range.clone(), domain)
                    .expect("column-backed commitment must extract"),
            );
            assert_eq!(cache.len(), width * domain);

            let raw = |slice: &[F]| {
                slice
                    .iter()
                    .map(PrimeField64::to_noncanonical_u64)
                    .collect::<Vec<_>>()
            };

            for batch_size in [1usize, 3, 8, 32] {
                let num_batches = domain.div_ceil(batch_size);
                for batch_i in 0..num_batches {
                    let start = batch_size * batch_i;
                    let n = batch_size.min(domain - start);
                    let indices = (start..start + n).collect::<Vec<_>>();

                    for (range, column_offset) in [
                        (constants_range.clone(), 0),
                        (sigmas_range.clone(), num_constants),
                    ] {
                        let w = range.len();
                        let mut uncached = Vec::new();
                        batch.fill_lde_batch(
                            &indices,
                            step,
                            range,
                            BatchLayout::PolyMajor,
                            &mut uncached,
                        );
                        assert_eq!(uncached.len(), w * n);
                        for ci in 0..w {
                            let cached = &cache[(column_offset + ci) * domain + start..][..n];
                            assert_eq!(
                                raw(cached),
                                raw(&uncached[ci * n..][..n]),
                                "degree_bits={degree_bits} batch_size={batch_size} \
                                 batch_i={batch_i} column={ci}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `transpose_poly_values_ref` must be bit-identical to the consuming
    /// `transpose_poly_values`, so the commitment can take the sigma columns by
    /// move instead of by clone. Includes non-canonical raw words, so the
    /// comparison is on raw representatives rather than field congruence.
    #[test]
    fn transpose_poly_values_ref_matches_consuming_transpose_raw_words() {
        use crate::field::types::{Field64, PrimeField64};
        use crate::util::{transpose_poly_values, transpose_poly_values_ref};

        type F = GoldilocksField;

        for (width, degree) in [(1usize, 1usize), (3, 8), (11, 64), (80, 16)] {
            let polys = (0..width)
                .map(|column| {
                    PolynomialValues::new(
                        (0..degree)
                            .map(|row| {
                                let raw = match (column * degree + row) % 7 {
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

            let by_ref = transpose_poly_values_ref(&polys);
            let consumed = transpose_poly_values(polys);

            let raw = |rows: &[Vec<F>]| {
                rows.iter()
                    .map(|row| {
                        row.iter()
                            .map(PrimeField64::to_noncanonical_u64)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                raw(&by_ref),
                raw(&consumed),
                "width={width} degree={degree}"
            );
        }
    }
}
