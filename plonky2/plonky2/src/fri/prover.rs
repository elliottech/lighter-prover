#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::any::TypeId;
use core::mem::{align_of, size_of, ManuallyDrop};

use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::{unflatten, Extendable, FieldExtension};
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::oracle::{coset_fft_zero_tail_base, coset_fft_zero_tail_base_dif_bitrev};
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
const FRI_FOLD_ARITY16_BATCH_WIDTH: usize = 8;

/// Builds a FRI proof.
pub fn fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    // Coefficients of the polynomial on which the LDT is performed. Only the first `1/rate` coefficients are non-zero.
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    // Evaluation of the polynomial on the large domain.
    lde_polynomial_values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    fri_proof_with_initial_order::<F, C, D>(
        initial_merkle_trees,
        lde_polynomial_coeffs,
        lde_polynomial_values,
        false,
        challenger,
        fri_params,
        final_poly_coeff_len,
        max_num_query_steps,
        timing,
    )
}

/// Internal FRI entry point for producers that explicitly know whether their
/// first codeword is already in bit-reversed Merkle-leaf order. The public API
/// above retains its historical natural-order contract.
pub(crate) fn fri_proof_with_initial_order<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    lde_polynomial_values: PolynomialValues<F::Extension>,
    initial_values_bitrev: bool,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    let n = lde_polynomial_values.len();
    assert_eq!(lde_polynomial_coeffs.len(), n);

    // Commit phase
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        fri_committed_trees::<F, C, D>(
            lde_polynomial_coeffs,
            lde_polynomial_values,
            initial_values_bitrev,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
        )
    );

    // PoW phase
    let pow_witness = timed!(
        timing,
        "find proof-of-work witness",
        fri_proof_of_work::<F, C, D>(challenger, &fri_params.config)
    );

    // Query phase
    let query_round_proofs =
        fri_prover_query_rounds::<F, C, D>(initial_merkle_trees, &trees, challenger, n, fri_params);

    FriProof {
        // `fri_prover_query_rounds` above is the last reader of `trees`, and the vector
        // is dropped on return, so the caps can be moved out instead of cloned: the
        // clone allocated and copied a fresh `MerkleCap` per commit round only to free
        // the original moments later.
        commit_phase_merkle_caps: trees.into_iter().map(|t| t.cap).collect(),
        query_round_proofs,
        final_poly: final_coeffs,
        pow_witness,
    }
}

pub(crate) type FriCommitedTrees<F, C, const D: usize> = (
    Vec<MerkleTree<F, <C as GenericConfig<D>>::Hasher>>,
    PolynomialCoeffs<<F as Extendable<D>>::Extension>,
);

pub fn final_poly_coeff_len(mut degree_bits: usize, reduction_arity_bits: &Vec<usize>) -> usize {
    for arity_bits in reduction_arity_bits {
        degree_bits -= *arity_bits;
    }
    1 << degree_bits
}

/// Bit-reversal + flatten in one gather pass: output leaf `i` is the base-field
/// limb array of `values[reverse_bits(i, log2(values.len()))]`, so the returned
/// flat buffer is the bit-reversed codeword laid out row-major, ready for
/// [`MerkleTree::new_flat`].
///
/// The gather is bandwidth- and latency-bound rather than arithmetic-bound:
/// `reverse_bits` scatters consecutive outputs across the whole codeword, so
/// essentially every read is a cache miss and a single thread can only keep a
/// handful of them in flight. Splitting the *output* range into blocks lets one
/// worker per core drive its own independent miss stream. Block `b` owns
/// outputs `b * FLATTEN_BLOCK .. (b + 1) * FLATTEN_BLOCK`, a partition of
/// `0..n`, so every slot is written exactly once and the source is only read —
/// the result is index-for-index identical to the serial fill.
fn bitrev_flatten<F: RichField + Extendable<D>, const D: usize>(values: &[F::Extension]) -> Vec<F> {
    const FLATTEN_BLOCK: usize = 1 << 10;

    let n = values.len();
    let log_n = log2_strict(n);
    let mut flat: Vec<F> = Vec::with_capacity(n * D);
    {
        let spare = &mut flat.spare_capacity_mut()[..n * D];
        spare
            .par_chunks_mut(FLATTEN_BLOCK * D)
            .enumerate()
            .for_each(|(block, out)| {
                let base = block * FLATTEN_BLOCK;
                for (j, slot) in out.chunks_exact_mut(D).enumerate() {
                    let limbs = values[reverse_bits(base + j, log_n)].to_basefield_array();
                    for k in 0..D {
                        slot[k].write(limbs[k]);
                    }
                }
            });
    }
    // SAFETY: the loop above wrote every one of the `n * D` slots of spare
    // capacity exactly once, so the whole prefix is initialized.
    unsafe { flat.set_len(n * D) };
    flat
}

/// Flatten evaluations already emitted in FRI's bit-reversed Merkle-leaf
/// order. Unlike [`bitrev_flatten`], every input read is contiguous.
fn flatten_bitrev_order<F: RichField + Extendable<D>, const D: usize>(
    values: Vec<F::Extension>,
) -> Vec<F> {
    if D == 2
        && TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
        && TypeId::of::<F::Extension>() == TypeId::of::<QuadraticExtension<GoldilocksField>>()
    {
        assert_eq!(size_of::<F::Extension>(), 2 * size_of::<F>());
        assert_eq!(align_of::<F::Extension>(), align_of::<F>());
        let mut values = ManuallyDrop::new(values);
        let len = values.len();
        let capacity = values.capacity();
        // SAFETY: the TypeId checks prove the production pair
        // `QuadraticExtension<GoldilocksField>`, whose transparent
        // representation is exactly `[GoldilocksField; 2]`. The allocation's
        // byte size and alignment are unchanged; only length and capacity are
        // expressed in base-field elements. Ownership moves to the returned
        // Vec and the ManuallyDrop prevents a second free.
        return unsafe {
            Vec::from_raw_parts(values.as_mut_ptr().cast::<F>(), len * 2, capacity * 2)
        };
    }

    const FLATTEN_BLOCK: usize = 1 << 10;

    let n = values.len();
    let mut flat: Vec<F> = Vec::with_capacity(n * D);
    {
        let spare = &mut flat.spare_capacity_mut()[..n * D];
        spare
            .par_chunks_mut(FLATTEN_BLOCK * D)
            .enumerate()
            .for_each(|(block, out)| {
                let base = block * FLATTEN_BLOCK;
                for (j, slot) in out.chunks_exact_mut(D).enumerate() {
                    let limbs = values[base + j].to_basefield_array();
                    for k in 0..D {
                        slot[k].write(limbs[k]);
                    }
                }
            });
    }
    // SAFETY: every spare-capacity slot was written exactly once above.
    unsafe { flat.set_len(n * D) };
    flat
}

/// Production arity-16 folding dispatcher. Apple AArch64 Goldilocks ext2
/// batches independent rows; every other target/field keeps the old map.
fn fri_fold_arity16_chunks<F: RichField + Extendable<D>, const D: usize>(
    terms: &[F::Extension],
    beta: F::Extension,
    beta_powers: &[F::Extension; 16],
) -> Vec<F::Extension> {
    assert_eq!(terms.len() % 16, 0);

    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        use core::any::TypeId;

        use crate::field::extension::quadratic::QuadraticExtension;
        use crate::field::goldilocks_extensions::ext2_fri_fold_arity16_batch;
        use crate::field::goldilocks_field::GoldilocksField;

        type GoldilocksExt2 = QuadraticExtension<GoldilocksField>;
        if TypeId::of::<F::Extension>() == TypeId::of::<GoldilocksExt2>() {
            let folded_len = terms.len() / 16;
            let mut folded: Vec<F::Extension> = Vec::with_capacity(folded_len);
            // SAFETY: ext2_fri_fold_arity16_batch initializes every element below.
            unsafe { folded.set_len(folded_len) };
            // SAFETY: TypeId equality proves all three element types exactly.
            let terms_ext2 = unsafe {
                core::slice::from_raw_parts(terms.as_ptr().cast::<GoldilocksExt2>(), terms.len())
            };
            let powers_ext2 = unsafe {
                &*(beta_powers as *const [F::Extension; 16] as *const [GoldilocksExt2; 16])
            };
            let folded_len = folded.len();
            let folded_ext2 = unsafe {
                core::slice::from_raw_parts_mut(
                    folded.as_mut_ptr().cast::<GoldilocksExt2>(),
                    folded_len,
                )
            };
            folded_ext2
                .par_chunks_mut(FRI_FOLD_ARITY16_BATCH_WIDTH)
                .enumerate()
                .for_each(|(batch, output)| {
                    let start = batch * FRI_FOLD_ARITY16_BATCH_WIDTH * 16;
                    ext2_fri_fold_arity16_batch(
                        &terms_ext2[start..start + output.len() * 16],
                        powers_ext2,
                        output,
                    );
                });
            return folded;
        }
    }

    terms
        .par_chunks_exact(16)
        .map(|chunk| {
            let row = chunk.try_into().unwrap();
            F::fri_fold_arity16(row, beta, beta_powers)
        })
        .collect()
}

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    values: PolynomialValues<F::Extension>,
    mut values_are_bitrev: bool,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());
    let mut values = Some(values);

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    let num_rounds = fri_params.reduction_arity_bits.len();
    for (round, arity_bits) in fri_params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;
        #[cfg(feature = "diagnostic_profile")]
        let round_name = |names: [&'static str; 3]| names.get(round).copied().unwrap_or(names[2]);

        // Fused bit-reversal + flatten: one gather pass writes the flat leaf
        // buffer directly (leaf `i` is the `arity`-chunk of the bit-reversed
        // codeword starting at `i * arity`), instead of a random-access
        // in-place permutation followed by a separate flattening pass with a
        // heap allocation per element.
        let flat_values = {
            #[cfg(feature = "diagnostic_profile")]
            let _span = crate::util::profile::span(
                "fri_commit",
                round_name([
                    "direct_flatten_r0",
                    "direct_flatten_r1",
                    "direct_flatten_r2",
                ]),
            );
            let round_values = values
                .take()
                .expect("every FRI commit round has one codeword")
                .values;
            if values_are_bitrev {
                flatten_bitrev_order::<F, D>(round_values)
            } else {
                bitrev_flatten::<F, D>(&round_values)
            }
        };
        let tree = {
            #[cfg(feature = "diagnostic_profile")]
            let _span = crate::util::profile::span(
                "fri_commit",
                round_name(["merkle_tree_r0", "merkle_tree_r1", "merkle_tree_r2"]),
            );
            MerkleTree::<F, C::Hasher>::new_flat(
                flat_values,
                arity * D,
                fri_params.config.cap_height,
            )
        };

        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        // Only `1/2^rate_bits` of the coefficients are nonzero every round
        // (the zero-tail invariant asserted by the final truncation), and the
        // Horner fold of an all-zero chunk is exactly zero, so fold only the
        // live prefix and extend with the zeros those chunks would produce.
        let n_chunks = coeffs.coeffs.len() / arity;
        let support = coeffs.coeffs.len() >> fri_params.config.rate_bits;
        let live_chunks = support.div_ceil(arity).min(n_chunks);
        let beta_powers_16 = if arity == 16 {
            let mut powers = [F::Extension::ONE; 16];
            for i in 1..16 {
                powers[i] = powers[i - 1] * beta;
            }
            Some(powers)
        } else {
            None
        };
        let mut folded = {
            #[cfg(feature = "diagnostic_profile")]
            let _span = crate::util::profile::span(
                "fri_commit",
                round_name([
                    "coefficient_fold_r0",
                    "coefficient_fold_r1",
                    "coefficient_fold_r2",
                ]),
            );
            match &beta_powers_16 {
                Some(beta_powers) => fri_fold_arity16_chunks::<F, D>(
                    &coeffs.coeffs[..live_chunks * arity],
                    beta,
                    beta_powers,
                ),
                None => coeffs.coeffs[..live_chunks * arity]
                    .par_chunks_exact(arity)
                    .map(|chunk| reduce_with_powers(chunk, beta))
                    .collect::<Vec<_>>(),
            }
        };
        // The historical `resize(n_chunks, ZERO)` zero-filled the whole dead
        // tail. Zeros are actually *read as values* only where the next
        // round's exact-`arity` chunking can reach past the live support —
        // at most `arity_next - 1` slots past `live` — because every other
        // tail consumer (the zero-tail coset FFT and the final truncation +
        // transcript observation) reads only the live prefix. Extend the
        // length without storing the rest.
        let live = folded.len();
        folded.reserve(n_chunks - live);
        // SAFETY: length equals capacity; the slots beyond `pad_end` are
        // never read (see above), and `F::Extension` is plain data.
        unsafe { folded.set_len(n_chunks) };
        let pad_end = if round + 1 < num_rounds {
            n_chunks.min(live + (1 << fri_params.reduction_arity_bits[round + 1]))
        } else {
            live
        };
        for value in folded[live..pad_end].iter_mut() {
            *value = F::Extension::ZERO;
        }
        coeffs = PolynomialCoeffs::new(folded);
        shift = shift.exp_u64(arity as u64);
        // Chunk-wise folding preserves the zero tail: the coefficient vector
        // keeps `1/2^rate_bits` support every round (asserted by the
        // truncation below), so the FFT's zero-run shortcut always applies.
        // The coefficients from `live_chunks` on are the zeros the `resize`
        // above just wrote, and `shift^i * 0 == 0`, so the coset scaling is
        // dead work over that tail: scale only the folded prefix.
        //
        // `values` is read by exactly one thing: the *next* round's leaf
        // gather at the top of this loop. After the final round it is dropped
        // unread — everything below this loop uses only `coeffs` — so the
        // last round's transform is entirely dead work. Skip it.
        if round + 1 < num_rounds {
            values_are_bitrev = fri_params.config.rate_bits == 3;
            values = Some({
                #[cfg(feature = "diagnostic_profile")]
                let _span = crate::util::profile::span(
                    "fri_commit",
                    round_name(["coset_fft_r0", "coset_fft_r1", "coset_fft_r2"]),
                );
                if fri_params.config.rate_bits == 3 {
                    coset_fft_zero_tail_base_dif_bitrev::<F, D>(&coeffs, shift, live_chunks)
                } else {
                    coset_fft_zero_tail_base::<F, D>(
                        &coeffs,
                        shift,
                        live_chunks,
                        Some(fri_params.config.rate_bits),
                        None,
                    )
                }
            });
        }
    }

    // When verifying this proof in a circuit with a different number of query steps,
    // we need the challenger to stay in sync with the verifier. Therefore, the challenger
    // must observe the additional hash caps and generate dummy challenges.
    if let Some(step_count) = max_num_query_steps {
        let cap_len = (1 << fri_params.config.cap_height) * NUM_HASH_OUT_ELTS;
        let zero_cap = vec![F::ZERO; cap_len];
        for _ in fri_params.reduction_arity_bits.len()..step_count {
            challenger.observe_elements(&zero_cap);
            challenger.get_extension_challenge::<D>();
        }
    }

    // The coefficients being removed here should always be zero.
    coeffs
        .coeffs
        .truncate(coeffs.len() >> fri_params.config.rate_bits);

    challenger.observe_extension_elements(&coeffs.coeffs);
    // When verifying this proof in a circuit with a different final polynomial length,
    // the challenger needs to observe the full length of the final polynomial.
    if let Some(len) = final_poly_coeff_len {
        let current_len = coeffs.coeffs.len();
        for _ in current_len..len {
            challenger.observe_extension_element(&F::Extension::ZERO);
        }
    }

    (trees, coeffs)
}

const POW_LANES: usize = 4;
const POW_QUAD_DISABLE_ENV: &str = "LIGHTER_DISABLE_POW_QUAD";

/// The four-lane path is the delivery default. Only the exact diagnostic value
/// `LIGHTER_DISABLE_POW_QUAD=1` selects the scalar control; missing, empty,
/// non-Unicode, and all other values keep the candidate enabled.
#[cfg(feature = "std")]
#[inline]
fn pow_quad_enabled_from_env_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("1"))
}

#[cfg(feature = "std")]
#[inline]
fn pow_quad_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        pow_quad_enabled_from_env_value(std::env::var_os(POW_QUAD_DISABLE_ENV).as_deref())
    })
}

#[cfg(not(feature = "std"))]
#[inline(always)]
const fn pow_quad_enabled() -> bool {
    true
}

/// Returns the candidates in one four-lane PoW group and the number of active
/// lanes. Inactive tail lanes repeat `start` and are never considered valid.
///
/// The caller ranges `quad_index` only through `max_candidate / POW_LANES`, so
/// `start` and each active addition are bounded by `max_candidate`. This avoids
/// forming `max_candidate + 1`, which would overflow when the inclusive upper
/// bound is `u64::MAX`.
#[inline]
fn pow_candidate_quad(quad_index: u64, max_candidate: u64) -> ([u64; POW_LANES], usize) {
    debug_assert!(quad_index <= max_candidate / POW_LANES as u64);
    let start = quad_index * POW_LANES as u64;
    let remaining = max_candidate - start;
    if remaining >= (POW_LANES - 1) as u64 {
        ([start, start + 1, start + 2, start + 3], POW_LANES)
    } else {
        let active_lanes = remaining as usize + 1;
        let mut candidates = [start; POW_LANES];
        for (lane, candidate) in candidates.iter_mut().enumerate().take(active_lanes).skip(1) {
            *candidate = start + lane as u64;
        }
        (candidates, active_lanes)
    }
}

/// Performs the proof-of-work (a.k.a. grinding) step of the FRI protocol. Returns the PoW witness.
pub(crate) fn fri_proof_of_work<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    challenger: &mut Challenger<F, C::Hasher>,
    config: &FriConfig,
) -> F {
    let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;

    // The easiest implementation would be repeatedly clone our Challenger. With each clone, we'd
    // observe an incrementing PoW witness, then get the PoW response. If it contained sufficient
    // leading zeros, we'd end the search, and store this clone as our new challenger.
    //
    // However, performance is critical here. We want to avoid cloning Challenger, particularly
    // since it stores vectors, which means allocations. We'd like a more compact state to clone.
    //
    // We know that a duplex will be performed right after we send the PoW witness, so we can ignore
    // any output_buffer, which will be invalidated. We also know
    // input_buffer.len() < H::Permutation::WIDTH, an invariant of Challenger.
    //
    // We separate the duplex operation into two steps, one which can be performed now, and the
    // other which depends on the PoW witness candidate. The first step is the overwrite our sponge
    // state with any inputs (excluding the PoW witness candidate). The second step is to overwrite
    // one more element of our sponge state with the candidate, then apply the permutation,
    // obtaining our duplex's post-state which contains the PoW response.
    let mut duplex_intermediate_state = challenger.sponge_state;
    let witness_input_pos = challenger.input_buffer.len();
    // The comment above is explicit that this path exists to avoid allocating, and the
    // clone was one: it heap-allocated a copy of a buffer of `Copy` elements shorter than
    // the permutation width purely to hand `set_from_iter` owned values. `set_from_slice`
    // takes the same elements from the same positions with no allocation at all.
    duplex_intermediate_state.set_from_slice(&challenger.input_buffer, 0);

    let max_candidate = F::NEG_ONE.to_canonical_u64();
    let pow_witness = if pow_quad_enabled() {
        (0..=max_candidate / POW_LANES as u64)
            .into_par_iter()
            .map(|quad_index| {
                let (candidates, active_lanes) = pow_candidate_quad(quad_index, max_candidate);
                let mut duplex_states = [duplex_intermediate_state; POW_LANES];
                for (duplex_state, candidate) in duplex_states.iter_mut().zip(candidates) {
                    duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
                }
                <C::Hasher as Hasher<F>>::Permutation::permute_quad(&mut duplex_states);

                (0..active_lanes).find_map(|lane| {
                    let pow_response = duplex_states[lane].squeeze().iter().last().unwrap();
                    let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
                    (leading_zeros >= min_leading_zeros).then_some(candidates[lane])
                })
            })
            .find_any(Option::is_some)
            .flatten()
    } else {
        (0..=max_candidate).into_par_iter().find_any(|&candidate| {
            let mut duplex_state = duplex_intermediate_state;
            duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
            duplex_state.permute();
            let pow_response = duplex_state.squeeze().iter().last().unwrap();
            let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
            leading_zeros >= min_leading_zeros
        })
    }
    .map(F::from_canonical_u64)
    .expect("Proof of work failed. This is highly unlikely!");

    // Recompute pow_response using our normal Challenger code, and make sure it matches.
    challenger.observe_element(pow_witness);
    let pow_response = challenger.get_challenge();
    let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
    assert!(leading_zeros >= min_leading_zeros);
    pow_witness
}

fn fri_prover_query_rounds<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    challenger: &mut Challenger<F, C::Hasher>,
    n: usize,
    fri_params: &FriParams,
) -> Vec<FriQueryRound<F, C::Hasher, D>> {
    challenger
        .get_n_challenges(fri_params.config.num_query_rounds)
        .into_par_iter()
        .map(|rand| {
            let x_index = rand.to_canonical_u64() as usize % n;
            fri_prover_query_round::<F, C, D>(initial_merkle_trees, trees, x_index, fri_params)
        })
        .collect()
}

fn fri_prover_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    mut x_index: usize,
    fri_params: &FriParams,
) -> FriQueryRound<F, C::Hasher, D> {
    let mut query_steps = Vec::new();
    let initial_proof = initial_merkle_trees
        .iter()
        .map(|t| (t.leaf_vec(x_index), t.prove(x_index)))
        .collect::<Vec<_>>();
    for (i, tree) in trees.iter().enumerate() {
        let arity_bits = fri_params.reduction_arity_bits[i];
        let evals = unflatten(tree.get(x_index >> arity_bits));
        let merkle_proof = tree.prove(x_index >> arity_bits);

        query_steps.push(FriQueryStep {
            evals,
            merkle_proof,
        });

        x_index >>= arity_bits;
    }
    FriQueryRound {
        initial_trees_proof: FriInitialTreeProof {
            evals_proofs: initial_proof,
        },
        steps: query_steps,
    }
}

#[cfg(test)]
mod tests {
    use plonky2_field::types::Sample;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::plonk::config::Poseidon2GoldilocksConfig;

    /// `bitrev_flatten` must be raw-`u64`-identical to the serial
    /// gather-and-extend loop it replaced, for every leaf and every limb.
    #[test]
    fn bitrev_flatten_matches_serial_gather() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        // Sizes on both sides of the `FLATTEN_BLOCK = 1 << 10` grain: below it
        // (a single partial chunk), exactly on it, and several blocks past it.
        for log_n in [0usize, 1, 5, 10, 11, 13] {
            let n = 1usize << log_n;
            let values: Vec<FE> = (0..n).map(|_| FE::rand()).collect();

            // Reference: the original serial fill.
            let mut expected: Vec<F> = Vec::with_capacity(n * D);
            for i in 0..n {
                let x: [F; D] = values[reverse_bits(i, log_n)].to_basefield_array();
                expected.extend_from_slice(&x);
            }

            let actual = bitrev_flatten::<F, D>(&values);
            assert_eq!(actual.len(), expected.len(), "length for n = {n}");
            for (k, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(a.0, e.0, "limb {k} of {n}");
            }
        }
    }

    #[test]
    fn flatten_bitrev_order_transfers_ext2_allocation_and_raw_words() {
        use crate::field::extension::quadratic::QuadraticExtension;

        type F = GoldilocksField;
        type FE = QuadraticExtension<F>;

        let values = (0..257usize)
            .map(|i| {
                FE::from_basefield_array([
                    GoldilocksField((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                    GoldilocksField(
                        (i as u64)
                            .wrapping_mul(0xd1b5_4a32_d192_ed03)
                            .wrapping_add(F::ORDER),
                    ),
                ])
            })
            .collect::<Vec<_>>();
        let input_ptr = values.as_ptr().cast::<F>();
        let input_capacity = values.capacity();
        let expected = values
            .iter()
            .flat_map(|value| value.0)
            .map(|limb| limb.0)
            .collect::<Vec<_>>();

        let flat = flatten_bitrev_order::<F, 2>(values);
        assert_eq!(
            flat.as_ptr(),
            input_ptr,
            "the allocation must be transferred"
        );
        assert_eq!(flat.len(), expected.len());
        assert_eq!(flat.capacity(), input_capacity * 2);
        assert_eq!(
            flat.iter().map(|limb| limb.0).collect::<Vec<_>>(),
            expected,
            "flattening must retain every noncanonical raw limb",
        );
    }

    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    fn assert_arity16_batch_matches_scalar_raw(
        terms: &[<GoldilocksField as Extendable<2>>::Extension],
        beta: <GoldilocksField as Extendable<2>>::Extension,
    ) {
        type F = GoldilocksField;
        type FE = <F as Extendable<2>>::Extension;
        let mut powers = [FE::ONE; 16];
        for i in 1..16 {
            powers[i] = powers[i - 1] * beta;
        }
        let expected = terms
            .chunks_exact(16)
            .map(|chunk| {
                <F as Extendable<2>>::fri_fold_arity16(chunk.try_into().unwrap(), beta, &powers)
            })
            .collect::<Vec<_>>();
        let actual = fri_fold_arity16_chunks::<F, 2>(terms, beta, &powers);
        assert_eq!(actual.len(), expected.len());
        for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let actual: [F; 2] = actual.to_basefield_array();
            let expected: [F; 2] = expected.to_basefield_array();
            for limb in 0..2 {
                assert_eq!(
                    actual[limb].0, expected[limb].0,
                    "raw mismatch at row {row}, limb {limb}"
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    #[test]
    fn fri_fold_arity16_apple_batch_matches_scalar_canonical_raw() {
        type F = GoldilocksField;
        type FE = <F as Extendable<2>>::Extension;
        let beta = FE::from_basefield_array([
            F::from_canonical_u64(0x1234_5678_9abc_def0),
            F::from_canonical_u64(0x0fed_cba9_8765_4321),
        ]);
        for rows in [1, 2, 3, 7, 8, 9, 16, 19] {
            let terms = (0..rows * 16)
                .map(|i| {
                    let x = (i as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(0xd1b5_4a32_d192_ed03);
                    FE::from_basefield_array([
                        F::from_canonical_u64(x % F::ORDER),
                        F::from_canonical_u64(x.rotate_left(23) % F::ORDER),
                    ])
                })
                .collect::<Vec<_>>();
            assert_arity16_batch_matches_scalar_raw(&terms, beta);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    #[test]
    fn fri_fold_arity16_apple_batch_matches_scalar_noncanonical_raw_tails() {
        type F = GoldilocksField;
        type FE = <F as Extendable<2>>::Extension;
        let raw = [
            0,
            1,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            1 << 32,
            u64::MAX - 1,
            u64::MAX,
        ];
        let beta = FE::from_basefield_array([GoldilocksField(u64::MAX), GoldilocksField(F::ORDER)]);
        for rows in [1, 3, 7, 8, 9, 17] {
            let terms = (0..rows * 16)
                .map(|i| {
                    FE::from_basefield_array([
                        GoldilocksField(raw[i % raw.len()]),
                        GoldilocksField(raw[(i * 5 + 3) % raw.len()]),
                    ])
                })
                .collect::<Vec<_>>();
            assert_arity16_batch_matches_scalar_raw(&terms, beta);
        }
    }

    #[test]
    fn pow_candidate_quad_covers_tails_and_u64_boundaries() {
        for max_candidate in 0_u64..=10 {
            let actual = (0..=max_candidate / POW_LANES as u64)
                .flat_map(|quad_index| {
                    let (candidates, active_lanes) = pow_candidate_quad(quad_index, max_candidate);
                    candidates.into_iter().take(active_lanes)
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, (0..=max_candidate).collect::<Vec<_>>());
        }

        // These final groups exercise 1, 2, 3, and 4 active lanes immediately
        // below the largest u64, including the inclusive u64::MAX endpoint.
        for max_candidate in (u64::MAX - 3)..=u64::MAX {
            let quad_index = max_candidate / POW_LANES as u64;
            let (candidates, active_lanes) = pow_candidate_quad(quad_index, max_candidate);
            let start = quad_index * POW_LANES as u64;
            assert_eq!(active_lanes, (max_candidate - start) as usize + 1);
            assert_eq!(candidates[0], start);
            assert_eq!(candidates[active_lanes - 1], max_candidate);
            assert!(candidates[..active_lanes]
                .windows(2)
                .all(|pair| pair[1] == pair[0] + 1));
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn pow_quad_is_default_on_and_only_exact_one_disables() {
        use std::ffi::OsStr;

        assert!(pow_quad_enabled_from_env_value(None));
        assert!(!pow_quad_enabled_from_env_value(Some(OsStr::new("1"))));
        for value in ["", "0", "01", "true", " 1", "1 "] {
            assert!(
                pow_quad_enabled_from_env_value(Some(OsStr::new(value))),
                "unexpectedly disabled by {value:?}"
            );
        }
    }

    #[test]
    fn fri_pow_quad_witness_replays_through_challenger() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        let config = FriConfig {
            rate_bits: 3,
            cap_height: 4,
            proof_of_work_bits: 8,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
            num_query_rounds: 28,
        };
        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_elements(&[
            F::from_canonical_u64(7),
            F::from_canonical_u64(11),
            F::from_canonical_u64(13),
        ]);
        let mut replay = challenger.clone();

        let witness = fri_proof_of_work::<F, C, D>(&mut challenger, &config);
        replay.observe_element(witness);
        let response = replay.get_challenge();
        let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;
        assert!(response.to_canonical_u64().leading_zeros() >= min_leading_zeros);
        assert_eq!(challenger.get_n_challenges(16), replay.get_n_challenges(16));
    }
}
