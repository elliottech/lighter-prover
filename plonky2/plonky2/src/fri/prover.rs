#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::{unflatten, Extendable, FieldExtension};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

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
    let n = lde_polynomial_values.len();
    assert_eq!(lde_polynomial_coeffs.len(), n);

    // Commit phase
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        fri_committed_trees::<F, C, D>(
            lde_polynomial_coeffs,
            lde_polynomial_values,
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
        commit_phase_merkle_caps: trees.iter().map(|t| t.cap.clone()).collect(),
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
#[allow(clippy::chunks_exact_to_as_chunks)]
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

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    mut values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    for arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1 << arity_bits;

        // Fused bit-reversal + flatten: one gather pass writes the flat leaf
        // buffer directly (leaf `i` is the `arity`-chunk of the bit-reversed
        // codeword starting at `i * arity`), instead of a random-access
        // in-place permutation followed by a separate flattening pass with a
        // heap allocation per element.
        let flat_values = bitrev_flatten::<F, D>(&values.values);
        let tree = MerkleTree::<F, C::Hasher>::new_flat(
            flat_values,
            arity * D,
            fri_params.config.cap_height,
        );

        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        coeffs = PolynomialCoeffs::new(
            coeffs
                .coeffs
                .par_chunks_exact(arity)
                .map(|chunk| reduce_with_powers(chunk, beta))
                .collect::<Vec<_>>(),
        );
        shift = shift.exp_u64(arity as u64);
        // Chunk-wise folding preserves the zero tail: the coefficient vector
        // keeps `1/2^rate_bits` support every round (asserted by the
        // truncation below), so the FFT's zero-run shortcut always applies.
        values =
            coeffs.coset_fft_with_options(shift.into(), Some(fri_params.config.rate_bits), None)
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
    duplex_intermediate_state.set_from_iter(challenger.input_buffer.clone(), 0);

    let pow_witness = (0..=F::NEG_ONE.to_canonical_u64())
        .into_par_iter()
        .find_any(|&candidate| {
            let mut duplex_state = duplex_intermediate_state;
            duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
            duplex_state.permute();
            let pow_response = duplex_state.squeeze().iter().last().unwrap();
            let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
            leading_zeros >= min_leading_zeros
        })
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
}
