//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::{BatchLayout, PolynomialBatch};
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::LookupSelectors;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::iop::generator::generate_partial_witness;
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::NUM_COINS_LOOKUP;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::{
    eval_vanishing_poly_base_batch, get_lut_poly, PermutationBatch, VanishingScratch,
};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::log2_ceil;
use crate::util::timing::TimingTree;

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
pub fn set_lookup_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    pw: &mut PartitionWitness<F>,
) -> Result<()> {
    for (
        lut_index,
        &LookupWire {
            last_lu_gate: _,
            last_lut_gate,
            first_lut_gate,
        },
    ) in prover_data.lookup_rows.iter().enumerate()
    {
        let lut_len = common_data.luts[lut_index].len();
        let num_entries = LookupGate::num_slots(&common_data.config);
        let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

        // Compute multiplicities.
        let mut multiplicities = vec![0; lut_len];

        let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
            .iter()
            .enumerate()
            .map(|(i, (inp_target, _))| (*inp_target, i))
            .collect();

        for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
            let inp_value = pw.get_target(*inp_target);
            let idx = table_value_to_idx
                .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
                .unwrap();

            multiplicities[*idx] += 1;
        }

        // Pad the last `LookupGate` with the first entry from the LUT.
        let remaining_slots = (num_entries
            - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
            % num_entries;
        let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
        for slot in (num_entries - remaining_slots)..num_entries {
            let inp_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
            let out_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
            pw.set_target(inp_target, F::from_canonical_u16(first_inp_value))?;
            pw.set_target(out_target, F::from_canonical_u16(first_out_value))?;

            multiplicities[0] += 1;
        }

        for lut_entry in 0..lut_len {
            let row = first_lut_gate - lut_entry / num_lut_entries;
            let col = lut_entry % num_lut_entries;

            let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

            pw.set_target(
                mul_target,
                F::from_canonical_usize(multiplicities[lut_entry]),
            )?;
        }
    }

    Ok(())
}

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        generate_partial_witness(inputs, prover_data, common_data)?
    );

    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    mut partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    set_lookup_wires(prover_data, common_data, &mut partition_witness)?;

    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
    let public_inputs_hash = C::InnerHasher::hash_no_pad(&public_inputs);

    let mut witness = timed!(
        timing,
        "compute full witness",
        partition_witness.full_witness()
    );

    // Only the routed columns are read again after this point (the
    // permutation argument covers wires `j < num_routed_wires`; nothing else
    // consumes the matrix), so move the non-routed columns out instead of
    // cloning them.
    let num_routed_wires = common_data.config.num_routed_wires;
    let wires_values: Vec<PolynomialValues<F>> = timed!(
        timing,
        "compute wire polynomials",
        witness
            .wire_values
            .par_iter_mut()
            .enumerate()
            .map(|(j, column)| {
                if j < num_routed_wires {
                    PolynomialValues::new(column.clone())
                } else {
                    PolynomialValues::new(core::mem::take(column))
                }
            })
            .collect()
    );

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_values(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);
    // The quotient numerator uses `beta_i * (k_j * x)` for every routed wire
    // and quotient point. Reassociate this finite-field product once per
    // challenge and wire; the resulting coefficient is reused across all
    // quotient batches.
    let beta_k_is: Vec<F> = betas
        .iter()
        .flat_map(|&beta| common_data.k_is.iter().map(move |&k_i| beta * k_i))
        .collect();

    let deltas = if has_lookup {
        let mut delts = Vec::with_capacity(2 * num_challenges);
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        let additional = challenger.get_n_challenges(num_additional_challenges);
        delts.extend(&betas);
        delts.extend(&gammas);
        delts.extend(additional);
        delts
    } else {
        vec![]
    };

    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    let mut partial_products_and_zs = timed!(
        timing,
        "compute partial products",
        all_wires_permutation_partial_products(
            &witness,
            &betas,
            &beta_k_is,
            &gammas,
            prover_data,
            common_data,
        )
    );

    // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
    let plonk_z_vecs: Vec<_> = partial_products_and_zs
        .iter_mut()
        .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
        .collect();
    let partial_products_len = partial_products_and_zs.iter().map(Vec::len).sum::<usize>();
    let mut zs_partial_products = Vec::with_capacity(plonk_z_vecs.len() + partial_products_len);
    zs_partial_products.extend(plonk_z_vecs);
    zs_partial_products.extend(partial_products_and_zs.into_iter().flatten());

    // All lookup polys: RE and partial SLDCs.
    let lookup_polys =
        compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

    if has_lookup {
        zs_partial_products.extend(lookup_polys);
    }

    let partial_products_zs_and_lookup_commitment = timed!(
        timing,
        "commit to partial products, Z's and, if any, lookup polynomials",
        PolynomialBatch::from_values(
            zs_partial_products,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&partial_products_zs_and_lookup_commitment.merkle_tree.cap);

    let alphas = challenger.get_n_challenges(num_challenges);

    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            &beta_k_is,
            &deltas,
            &alphas,
            // Layout seam: flat column-major permutation data when the
            // circuit has no lookups; the per-point path otherwise.
            !has_lookup,
        )
    );

    // Differential gate for the layout seam: recompute the quotient through
    // the per-point reference path on the same witness, commitments and
    // challenges, and require value-identical polynomials.
    #[cfg(test)]
    if !has_lookup && COMPARE_QUOTIENT_LAYOUTS.load(core::sync::atomic::Ordering::Relaxed) {
        let reference = compute_quotient_polys::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            &beta_k_is,
            &deltas,
            &alphas,
            false,
        );
        assert_eq!(quotient_polys.len(), reference.len());
        for (p, (a, b)) in quotient_polys.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.coeffs.len(), b.coeffs.len());
            for (i, (x, y)) in a.coeffs.iter().zip(b.coeffs.iter()).enumerate() {
                assert_eq!(
                    x.to_canonical_u64(),
                    y.to_canonical_u64(),
                    "quotient layout divergence: poly {p}, coeff {i}"
                );
            }
        }
    }

    let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "split up quotient polys",
        quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim_to_len(quotient_degree).expect(
                    "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                );
                // Split quotient into degree-n chunks.
                quotient_poly.chunks(degree)
            })
            .collect()
    );

    let quotient_polys_commitment = timed!(
        timing,
        "commit to quotient polys",
        PolynomialBatch::<F, C, D>::from_coeffs(
            all_quotient_poly_chunks,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&quotient_polys_commitment.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    let openings = timed!(
        timing,
        "construct the opening set, including lookups",
        OpeningSet::new(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
            common_data
        )
    );
    challenger.observe_openings(&openings.to_fri_openings());
    let instance = common_data.get_fri_instance(zeta);

    let opening_proof = timed!(
        timing,
        "compute opening proofs",
        PolynomialBatch::<F, C, D>::prove_openings(
            &instance,
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &partial_products_zs_and_lookup_commitment,
                &quotient_polys_commitment,
            ],
            &mut challenger,
            &common_data.fri_params,
            None,
            None,
            timing,
        )
    );

    let proof = Proof::<F, C, D> {
        wires_cap: wires_commitment.merkle_tree.cap,
        plonk_zs_partial_products_cap: partial_products_zs_and_lookup_commitment.merkle_tree.cap,
        quotient_polys_cap: quotient_polys_commitment.merkle_tree.cap,
        openings,
        opening_proof,
    };
    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    beta_k_is: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    let num_challenges = common_data.config.num_challenges;
    let num_routed_wires = common_data.config.num_routed_wires;
    debug_assert_eq!(betas.len(), num_challenges);
    debug_assert_eq!(beta_k_is.len(), num_challenges * num_routed_wires);
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                &beta_k_is[i * num_routed_wires..(i + 1) * num_routed_wires],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

#[inline]
fn divide_chunk_products<F: Field>(
    numerator_products: &mut [F],
    denominator_products: &[F],
    inverse_scratch: &mut Vec<F>,
) {
    debug_assert_eq!(numerator_products.len(), denominator_products.len());
    F::batch_multiplicative_inverse_into(denominator_products, inverse_scratch);
    for (product, &inverse) in numerator_products.iter_mut().zip(inverse_scratch.iter()) {
        *product *= inverse;
    }
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    beta: F,
    beta_k_is: &[F],
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let num_prods = common_data.num_partial_products;
    debug_assert_eq!(beta_k_is.len(), common_data.config.num_routed_wires);
    let num_routed_wires = common_data.config.num_routed_wires;
    let num_chunks = num_prods + 1;
    debug_assert_eq!(num_chunks, num_routed_wires.div_ceil(degree));

    // The permutation argument only consumes one numerator/denominator ratio per quotient-degree
    // chunk. Form those products before Montgomery inversion, shrinking each inversion batch by
    // `degree` and reading every witness wire only once.
    const INV_BATCH: usize = 128;
    let mut all_quotient_chunk_products = vec![F::ZERO; subgroup.len() * num_chunks];
    all_quotient_chunk_products
        .par_chunks_mut(INV_BATCH * num_chunks)
        .zip(subgroup.par_chunks(INV_BATCH))
        .enumerate()
        .for_each_init(
            || {
                (
                    Vec::with_capacity(num_chunks * INV_BATCH),
                    Vec::with_capacity(num_chunks * INV_BATCH),
                )
            },
            |scratch, (chunk_idx, (quotient_products, xs))| {
                let base = chunk_idx * INV_BATCH;
                let (denominator_products, denominator_inverses) = scratch;
                denominator_products.clear();
                for (t, &x) in xs.iter().enumerate() {
                    let i = base + t;
                    let s_sigmas = &prover_data.sigmas[i];
                    for chunk in 0..num_chunks {
                        let start = chunk * degree;
                        let end = min(start + degree, num_routed_wires);
                        let mut numerator_product = F::ONE;
                        let mut denominator_product = F::ONE;
                        for j in start..end {
                            let wire_value = witness.get_wire(i, j);
                            numerator_product *= wire_value + beta_k_is[j] * x + gamma;
                            denominator_product *= wire_value + beta * s_sigmas[j] + gamma;
                        }
                        quotient_products[t * num_chunks + chunk] = numerator_product;
                        denominator_products.push(denominator_product);
                    }
                }
                divide_chunk_products(
                    quotient_products,
                    denominator_products,
                    denominator_inverses,
                );
            },
        );

    // Accumulate the sequential Z chain directly into the column-major output
    // polynomials, deleting the per-point row Vec, the row-major intermediate,
    // and the whole-phase transpose. Values and their order are identical: for
    // each point, column k receives the k-th running product, and the last
    // column receives the previous Z(x) exactly as the swap-based version did.
    let n_points = subgroup.len();
    let mut columns: Vec<Vec<F>> = (0..num_prods + 1)
        .map(|_| Vec::with_capacity(n_points))
        .collect();
    let mut z_x = F::ONE;
    for quotient_chunk_products in all_quotient_chunk_products.chunks_exact(num_chunks) {
        let mut acc = z_x;
        for (k, &quotient_chunk_product) in quotient_chunk_products.iter().enumerate() {
            acc *= quotient_chunk_product;
            if k == num_prods {
                // The last term is Z(gx), but we store Z(x) in its place,
                // otherwise Z would end up shifted.
                columns[k].push(z_x);
                z_x = acc;
            } else {
                columns[k].push(acc);
            }
        }
    }

    columns.into_iter().map(PolynomialValues::new).collect()
}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
fn compute_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F; 4],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.degree();
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1;
    let num_partial_lookups = num_lu_slots.div_ceil(max_lookup_degree);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let max_lookup_table_degree = num_lut_slots.div_ceil(num_partial_lookups);

    // First poly is RE, the rest are partial SLDCs.
    let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
    for _ in 0..num_partial_lookups + 1 {
        final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree]));
    }

    for LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in prover_data.lookup_rows.clone()
    {
        // Set values for partial Sums and RE.
        for row in (last_lut_row..(first_lut_row + 1)).rev() {
            // Get combos for Sum.
            let looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
                .collect();
            // Get 1/(alpha - combo).
            let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

            // Get lookup combos, used to check the well formation of the LUT.
            let lookup_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out
                })
                .collect();

            // Compute next row's first value of RE.
            // If `row == first_lut_row`, then `final_poly_vecs[0].values[row + 1] == 0`.
            let mut new_re = final_poly_vecs[0].values[row + 1];
            for elt in &lookup_combos {
                new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
            }
            final_poly_vecs[0].values[row] = new_re;

            for slot in 0..num_partial_lookups {
                let prev = if slot != 0 {
                    final_poly_vecs[slot].values[row]
                } else {
                    // If `row == first_lut_row`, then `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                };
                let sum = (slot * max_lookup_table_degree
                    ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
                    .fold(prev, |acc, s| {
                        acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
                            * looked_combo_inverses[s]
                    });
                final_poly_vecs[slot + 1].values[row] = sum;
            }
        }

        // Set values for partial LDCs.
        for row in (last_lu_row..last_lut_row).rev() {
            // Get looking combos.
            let looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| {
                    let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s));
                    let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s));

                    looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
                .collect();
            // Get 1 / (alpha - combo).
            let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

            for slot in 0..num_partial_lookups {
                let prev = if slot == 0 {
                    // Valid at _any_ row, even `first_lu_row`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                } else {
                    final_poly_vecs[slot].values[row]
                };
                let sum = (slot * max_lookup_degree
                    ..min((slot + 1) * max_lookup_degree, num_lu_slots))
                    .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
                final_poly_vecs[slot + 1].values[row] = prev - sum;
            }
        }
    }

    final_poly_vecs
}

/// Computes lookup polynomials for all challenges.
fn compute_all_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    lookup: bool,
) -> Vec<PolynomialValues<F>> {
    if lookup {
        let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
            .map(|c| {
                compute_lookup_polys(
                    witness,
                    &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
                        .try_into()
                        .unwrap(),
                    prover_data,
                    common_data,
                )
            })
            .collect();
        polys.into_iter().flatten().collect()
    } else {
        vec![]
    }
}

const BATCH_SIZE: usize = 32;

/// Test-only switch: when set, `compute_quotient_polys` evaluates the quotient
/// values twice — once through the default column-major (`PolyMajor`)
/// permutation path and once through the per-point (`PointMajor`) reference
/// path — over the same witness, commitments and challenges, and asserts the
/// two are value-identical. (Cross-run proof-byte comparison is not a usable
/// oracle in this fork: unused wire slots carry nondeterministic padding, so
/// two proofs of the same witness legitimately differ byte-wise.)
#[cfg(test)]
pub(crate) static COMPARE_QUOTIENT_LAYOUTS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

fn compute_quotient_polys<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    beta_k_is: &[F],
    deltas: &[F],
    alphas: &[F],
    col_major_perm: bool,
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges;

    let has_lookup = common_data.num_lookup_polys != 0;

    // The lookup constraint evaluator consumes per-point rows, so the
    // column-major permutation layout is only usable without lookups,
    // whatever the caller asked for.
    let col_major_perm = col_major_perm && !has_lookup;

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `PolynomialBatch`s."
    );

    // We reuse the LDE computed in `PolynomialBatch` and extract every `step` points to get
    // an LDE matching `max_filtered_constraint_degree`.
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point in Plonk, need to look at the point `next_step`
    // steps away since we work on an LDE of degree `max_filtered_constraint_degree`.
    let next_step = 1 << quotient_degree_bits;

    // Process-global cached subgroup (bit-identical to computing it here): the
    // serial 2^19-length dependent multiply chain runs once per process
    // instead of once per proof.
    let points =
        precomputed::two_adic_subgroup::<F>(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);
    // The `L_0` denominator inverses consumed by `eval_l_0` depend only on
    // `(degree_bits, quotient_degree_bits, coset shift)` — not on any challenge — so they are
    // computed once per circuit shape for the process and shared across proofs. Each cached
    // entry is bit-identical to the per-point inversion it replaces.
    #[cfg(feature = "std")]
    let z_h_on_coset =
        z_h_on_coset.with_l_0_denominator_inverses(l_0_table_cache::l_0_denominator_inverses::<F>(
            common_data.degree_bits(),
            quotient_degree_bits,
        ));

    // Precompute the lookup table evals on the challenges in delta
    // These values are used to produce the final RE constraints for each lut,
    // and are the same each time in check_lookup_constraints_batched.
    // lut_poly_evals[i][j] gives the eval for the i'th challenge and the j'th lookup table
    let lut_re_poly_evals: Vec<Vec<F>> = if has_lookup {
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
        (0..num_challenges)
            .map(move |i| {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];
                let cur_challenge_delta = cur_deltas[LookupChallenges::ChallengeDelta as usize];

                (LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors)
                    .map(|r| {
                        let lut_row_number = common_data.luts
                            [r - LookupSelectors::StartEnd as usize]
                            .len()
                            .div_ceil(num_lut_slots);

                        get_lut_poly(
                            common_data,
                            r - LookupSelectors::StartEnd as usize,
                            cur_deltas,
                            num_lut_slots * lut_row_number,
                        )
                        .eval(cur_challenge_delta)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    let lut_re_poly_evals_refs: Vec<&[F]> =
        lut_re_poly_evals.iter().map(|v| v.as_slice()).collect();

    let points_batches = points.par_chunks(BATCH_SIZE);
    let num_batches = points.len().div_ceil(BATCH_SIZE);

    struct QuotientScratch<F: RichField> {
        indices: Vec<usize>,
        indices_next: Vec<usize>,
        shifted_xs: Vec<F>,
        local_constants: Vec<F>,
        local_wires: Vec<F>,
        s_sigmas_flat: Vec<F>,
        zs_local_flat: Vec<F>,
        zs_next_flat: Vec<F>,
        vanishing: VanishingScratch<F>,
    }

    let num_wires = common_data.config.num_wires;
    let zs_row_width = zs_partial_products_and_lookup_commitment.lde_row_width();
    let num_routed_wires = common_data.config.num_routed_wires;

    let mut quotient_values = vec![F::ZERO; points.len() * num_challenges];
    quotient_values
        .par_chunks_mut(BATCH_SIZE * num_challenges)
        .zip(points_batches)
        .enumerate()
        .for_each_init(
            || QuotientScratch::<F> {
                indices: Vec::with_capacity(BATCH_SIZE),
                indices_next: Vec::with_capacity(BATCH_SIZE),
                shifted_xs: Vec::with_capacity(BATCH_SIZE),
                local_constants: Vec::new(),
                local_wires: Vec::new(),
                s_sigmas_flat: Vec::new(),
                zs_local_flat: Vec::new(),
                zs_next_flat: Vec::new(),
                vanishing: VanishingScratch::default(),
            },
            |scratch, (batch_i, (quotient_values_batch, xs_batch))| {
                // Each batch must be the same size, except the last one, which may be smaller.
                debug_assert!(
                    xs_batch.len() == BATCH_SIZE
                        || (batch_i == num_batches - 1 && xs_batch.len() <= BATCH_SIZE)
                );

                let n = xs_batch.len();
                scratch.indices.clear();
                scratch
                    .indices
                    .extend(BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + n);
                scratch.indices_next.clear();
                scratch
                    .indices_next
                    .extend(scratch.indices.iter().map(|&i| (i + next_step) % lde_size));

                scratch.shifted_xs.clear();
                scratch
                    .shifted_xs
                    .extend(xs_batch.iter().map(|&x| F::coset_shift() * x));

                prover_data.constants_sigmas_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    common_data.constants_range(),
                    BatchLayout::PolyMajor,
                    &mut scratch.local_constants,
                );
                // Layout seam: the no-lookup column evaluator consumes the
                // PolyMajor gathers as-is (and the "next" gather narrows to
                // the Z columns, the only ones it reads); the per-point path
                // keeps the full-width PointMajor gathers and row views.
                let (batch_layout, zs_local_range, zs_next_range) = if col_major_perm {
                    (
                        BatchLayout::PolyMajor,
                        0..common_data.partial_products_range().end,
                        common_data.zs_range(),
                    )
                } else {
                    (BatchLayout::PointMajor, 0..zs_row_width, 0..zs_row_width)
                };

                prover_data.constants_sigmas_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    common_data.sigmas_range(),
                    batch_layout,
                    &mut scratch.s_sigmas_flat,
                );
                wires_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    0..num_wires,
                    BatchLayout::PolyMajor,
                    &mut scratch.local_wires,
                );
                zs_partial_products_and_lookup_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    zs_local_range,
                    batch_layout,
                    &mut scratch.zs_local_flat,
                );
                zs_partial_products_and_lookup_commitment.fill_lde_batch(
                    &scratch.indices_next,
                    step,
                    zs_next_range,
                    batch_layout,
                    &mut scratch.zs_next_flat,
                );

                let indices_batch = &scratch.indices;
                // Per-point row views over the PointMajor gathers, built only
                // for the per-point path; the column path passes the flat
                // buffers straight through, so these four allocations vanish
                // from the hot (no-lookup) path entirely.
                type RowViews<'v, F> = (Vec<&'v [F]>, Vec<&'v [F]>, Vec<&'v [F]>, Vec<&'v [F]>);
                let (local_zs_batch, next_zs_batch, partial_products_batch, s_sigmas_batch): RowViews<'_, F> =
                    if col_major_perm {
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                    } else {
                        (
                            (0..n)
                                .map(|k| {
                                    &scratch.zs_local_flat[k * zs_row_width..]
                                        [common_data.zs_range()]
                                })
                                .collect(),
                            (0..n)
                                .map(|k| {
                                    &scratch.zs_next_flat[k * zs_row_width..]
                                        [common_data.zs_range()]
                                })
                                .collect(),
                            (0..n)
                                .map(|k| {
                                    &scratch.zs_local_flat[k * zs_row_width..]
                                        [common_data.partial_products_range()]
                                })
                                .collect(),
                            (0..n)
                                .map(|k| {
                                    &scratch.s_sigmas_flat
                                        [k * num_routed_wires..(k + 1) * num_routed_wires]
                                })
                                .collect(),
                        )
                    };
                let (local_lookup_batch, next_lookup_batch): (Vec<&[F]>, Vec<&[F]>) = if has_lookup
                {
                    (
                        (0..n)
                            .map(|k| {
                                &scratch.zs_local_flat[k * zs_row_width..]
                                    [common_data.lookup_range()]
                            })
                            .collect(),
                        (0..n)
                            .map(|k| {
                                &scratch.zs_next_flat[k * zs_row_width..]
                                    [common_data.lookup_range()]
                            })
                            .collect(),
                    )
                } else {
                    (Vec::new(), Vec::new())
                };

                let perm = if col_major_perm {
                    PermutationBatch::Cols {
                        zs_partial_products_cols: &scratch.zs_local_flat,
                        zs_next_cols: &scratch.zs_next_flat,
                        s_sigmas_cols: &scratch.s_sigmas_flat,
                    }
                } else {
                    PermutationBatch::Rows {
                        local_zs_batch: &local_zs_batch,
                        next_zs_batch: &next_zs_batch,
                        partial_products_batch: &partial_products_batch,
                        s_sigmas_batch: &s_sigmas_batch,
                    }
                };

                let vars_batch = EvaluationVarsBaseBatch::new(
                    n,
                    &scratch.local_constants,
                    &scratch.local_wires,
                    public_inputs_hash,
                );

                let quotient_values_batch = &mut quotient_values_batch[..n * num_challenges];
                eval_vanishing_poly_base_batch::<F, D>(
                    common_data,
                    indices_batch,
                    &scratch.shifted_xs,
                    vars_batch,
                    perm,
                    &local_lookup_batch,
                    &next_lookup_batch,
                    betas,
                    gammas,
                    beta_k_is,
                    deltas,
                    alphas,
                    &z_h_on_coset,
                    &lut_re_poly_evals_refs,
                    &mut scratch.vanishing,
                    quotient_values_batch,
                );

                for (&i, quotient_values) in indices_batch
                    .iter()
                    .zip(quotient_values_batch.chunks_exact_mut(num_challenges))
                {
                    let denominator_inv = z_h_on_coset.eval_inverse(i);
                    quotient_values
                        .iter_mut()
                        .for_each(|v| *v *= denominator_inv);
                }
            },
        );

    debug_assert_eq!(quotient_values.len(), points.len() * num_challenges);
    (0..num_challenges)
        .into_par_iter()
        .map(|challenge| {
            PolynomialValues::new(
                quotient_values
                    .chunks_exact(num_challenges)
                    .map(|point_values| point_values[challenge])
                    .collect(),
            )
        })
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}

/// Process-global caches for deterministic per-degree precomputations that
/// were being redone per proof: the quotient-domain two-adic subgroup (a
/// serial dependent multiply chain over 2^19 points) and the coset-shift power
/// table used by every `PolynomialBatch` LDE. Entries are keyed by field type
/// and size; the stored vectors are exactly what the direct computation
/// returns, computed once, so every lookup is bit-identical to computing in
/// place. (Kept here rather than in `plonky2_field` so the file set stays
/// disjoint from pending `fft.rs` work.)
pub(crate) mod precomputed {
    #[cfg(feature = "std")]
    mod imp {
        use core::any::{Any, TypeId};
        use std::collections::HashMap;
        use std::sync::{Arc, OnceLock, RwLock};

        use crate::field::types::Field;

        type Map = RwLock<HashMap<(TypeId, usize), Arc<dyn Any + Send + Sync>>>;

        static SUBGROUPS: OnceLock<Map> = OnceLock::new();
        static COSET_POWERS: OnceLock<Map> = OnceLock::new();

        fn get_or_compute<F: Field>(
            cache: &'static OnceLock<Map>,
            len_key: usize,
            compute: impl FnOnce() -> Vec<F>,
        ) -> Arc<Vec<F>> {
            let key = (TypeId::of::<F>(), len_key);
            let map = cache.get_or_init(|| RwLock::new(HashMap::new()));
            if let Some(hit) = map.read().unwrap().get(&key) {
                return Arc::clone(hit)
                    .downcast::<Vec<F>>()
                    .unwrap_or_else(|_| panic!("type-keyed cache entry has the keyed type"));
            }
            let computed: Arc<Vec<F>> = Arc::new(compute());
            let mut map = map.write().unwrap();
            // If another thread inserted concurrently, keep its (identical)
            // table so all callers share one allocation.
            let entry = map
                .entry(key)
                .or_insert_with(|| computed as Arc<dyn Any + Send + Sync>);
            Arc::clone(entry)
                .downcast::<Vec<F>>()
                .unwrap_or_else(|_| panic!("type-keyed cache entry has the keyed type"))
        }

        /// Cached `F::two_adic_subgroup(n_log)`.
        pub(crate) fn two_adic_subgroup<F: Field>(n_log: usize) -> Arc<Vec<F>> {
            get_or_compute(&SUBGROUPS, n_log, || F::two_adic_subgroup(n_log))
        }

        /// Cached `F::coset_shift().powers().take(degree)`.
        pub(crate) fn coset_shift_powers<F: Field>(degree: usize) -> Arc<Vec<F>> {
            get_or_compute(&COSET_POWERS, degree, || {
                F::coset_shift().powers().take(degree).collect()
            })
        }
    }

    /// Without `std` there is no process-global synchronization; fall back to
    /// direct (uncached) computation, which is what the callers did before.
    #[cfg(not(feature = "std"))]
    mod imp {
        use alloc::sync::Arc;
        use alloc::vec::Vec;

        use crate::field::types::Field;

        pub(crate) fn two_adic_subgroup<F: Field>(n_log: usize) -> Arc<Vec<F>> {
            Arc::new(F::two_adic_subgroup(n_log))
        }

        pub(crate) fn coset_shift_powers<F: Field>(degree: usize) -> Arc<Vec<F>> {
            Arc::new(F::coset_shift().powers().take(degree).collect::<Vec<F>>())
        }
    }

    pub(crate) use imp::{coset_shift_powers, two_adic_subgroup};
}

#[cfg(test)]
mod quotient_layout_tests {
    use core::sync::atomic::Ordering;

    use anyhow::Result;

    use super::{precomputed, BatchLayout, COMPARE_QUOTIENT_LAYOUTS};
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Field64};
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    fn small_circuit() -> (CircuitData<F, C, D>, PartialWitness<F>) {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        let mut cur = x;
        for i in 0..64 {
            cur = builder.mul_add(cur, cur, x);
            let c = builder.constant(F::from_canonical_usize(i + 1));
            cur = builder.add(cur, c);
        }
        builder.register_public_input(cur);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_target(x, F::from_canonical_u64(3)).unwrap();
        (data, pw)
    }

    /// B1/B2/D1 differential gate: within a single prove call — same witness,
    /// commitments and challenges — the default column-major (`PolyMajor`)
    /// quotient path and the per-point (`PointMajor`) reference path must
    /// produce value-identical quotient polynomials. The element-wise
    /// comparison itself runs inside `prove` (see `COMPARE_QUOTIENT_LAYOUTS`);
    /// the proof must also verify.
    #[test]
    fn quotient_layout_paths_agree() -> Result<()> {
        let (data, pw) = small_circuit();
        assert!(data.common.luts.is_empty());

        COMPARE_QUOTIENT_LAYOUTS.store(true, Ordering::SeqCst);
        let proof = data.prove(pw);
        COMPARE_QUOTIENT_LAYOUTS.store(false, Ordering::SeqCst);

        data.verify(proof?)?;
        Ok(())
    }

    /// Layout seam: `PolyMajor` output is exactly the transpose of
    /// `PointMajor` output, element for element (raw u64 compare).
    #[test]
    fn fill_lde_batch_layouts_agree() {
        let (data, _) = small_circuit();
        let commitment = &data.prover_only.constants_sigmas_commitment;
        let range = data.common.sigmas_range();
        let w = range.len();
        let indices = [0usize, 1, 2, 5, 7, 11, 30];
        let n = indices.len();
        let mut point_major = Vec::new();
        let mut poly_major = Vec::new();
        commitment.fill_lde_batch(
            &indices,
            2,
            range.clone(),
            BatchLayout::PointMajor,
            &mut point_major,
        );
        commitment.fill_lde_batch(&indices, 2, range, BatchLayout::PolyMajor, &mut poly_major);
        assert_eq!(point_major.len(), n * w);
        assert_eq!(poly_major.len(), n * w);
        for k in 0..n {
            for c in 0..w {
                assert_eq!(point_major[k * w + c].0, poly_major[c * n + k].0);
            }
        }
    }

    /// A contiguous PolyMajor gather must produce the same column slices as
    /// the generic indexed gather. This catches off-by-one source ranges and
    /// accidental point-major writes in the quotient fast path.
    #[test]
    fn contiguous_lde_batch_matches_indexed_gather() {
        let (data, _) = small_circuit();
        let commitment = &data.prover_only.constants_sigmas_commitment;
        let range = data.common.sigmas_range();
        let indices = [3usize, 4, 5, 6, 7, 8, 9];
        let mut indexed = Vec::new();
        let mut contiguous = Vec::new();

        commitment.fill_lde_batch(
            &indices,
            1,
            range.clone(),
            BatchLayout::PolyMajor,
            &mut indexed,
        );
        commitment.fill_lde_batch_contiguous(indices[0], indices.len(), range, &mut contiguous);

        assert_eq!(contiguous, indexed);
    }

    /// Scratch reuse: `fill_lde_batch` writes every cell of `out` before any
    /// is read, so dropping the zero-fill of an already correctly sized buffer
    /// must be invisible. A poisoned reused buffer has to gather exactly what
    /// a freshly allocated one does, for both layouts, across the full batch
    /// and the short final batch (which shrinks the buffer).
    #[test]
    fn fill_lde_batch_overwrites_dirty_scratch() {
        let (data, _) = small_circuit();
        let commitment = &data.prover_only.constants_sigmas_commitment;
        let range = data.common.sigmas_range();
        let indices = [0usize, 1, 2, 5, 7, 11, 30];
        let poison = F::from_canonical_u64(0x1234_5678_9abc_def0);
        for layout in [BatchLayout::PointMajor, BatchLayout::PolyMajor] {
            // One buffer reused across a full batch then a short one, exactly
            // as the quotient loop's `for_each_init` scratch is.
            let mut scratch = vec![poison; 3];
            for n in [indices.len(), 3, 3] {
                let mut fresh = Vec::new();
                commitment.fill_lde_batch(&indices[..n], 2, range.clone(), layout, &mut fresh);
                commitment.fill_lde_batch(&indices[..n], 2, range.clone(), layout, &mut scratch);
                assert_eq!(scratch.len(), fresh.len());
                for (actual, expected) in scratch.iter().zip(&fresh) {
                    assert_eq!(actual.0, expected.0);
                }
                // Poison every cell so the next iteration starts from a dirty
                // buffer of the right length (the reuse case being deleted).
                scratch.fill(poison);
            }
        }
    }

    /// C1/C2: cached tables must be bit-identical to direct computation, on
    /// both the miss and hit paths.
    #[test]
    fn precomputed_tables_match_direct() {
        for n_log in [1usize, 4, 9] {
            assert_eq!(
                *precomputed::two_adic_subgroup::<F>(n_log),
                F::two_adic_subgroup(n_log)
            );
            assert_eq!(
                *precomputed::two_adic_subgroup::<F>(n_log),
                F::two_adic_subgroup(n_log)
            );
        }
        for degree in [8usize, 64, 512] {
            let direct: Vec<F> = F::coset_shift().powers().take(degree).collect();
            assert_eq!(*precomputed::coset_shift_powers::<F>(degree), direct);
            assert_eq!(*precomputed::coset_shift_powers::<F>(degree), direct);
        }
    }

    /// D1: `ONE * a == a` bitwise for Goldilocks, canonical or not, so peeling
    /// the first factor of each chunk product into a direct assignment is
    /// value-exact.
    #[test]
    fn mul_by_one_is_bitwise_identity() {
        let order = GoldilocksField::ORDER;
        for raw in [0u64, 1, 1234567, order - 1, order, order + 12345, u64::MAX] {
            let x = GoldilocksField::from_noncanonical_u64(raw);
            assert_eq!((GoldilocksField::ONE * x).0, x.0);
        }
    }
}

/// Process-global cache of the `L_0(x)` denominator inverses `(n * (x - 1))^-1` consumed by
/// `ZeroPolyOnCoset::eval_l_0` in the quotient pass: one entry per LDE point `x = g * w^i`,
/// `2^(degree_bits + quotient_degree_bits)` per circuit shape. The values depend only on
/// `(degree_bits, quotient_degree_bits)` and the field's coset shift, so they are built once
/// per process and shared across proofs, mirroring the precomputed-table style of
/// `field::fft::fft_root_table`.
#[cfg(feature = "std")]
mod l_0_table_cache {
    use core::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use plonky2_maybe_rayon::*;

    use crate::field::types::Field;

    /// Keyed by field type and `(degree_bits, quotient_degree_bits)`; the coset shift is a
    /// constant of the field type.
    #[allow(clippy::type_complexity)]
    static CACHE: OnceLock<Mutex<HashMap<(TypeId, usize, usize), Arc<dyn Any + Send + Sync>>>> =
        OnceLock::new();

    /// Builds the table with, per entry, exactly the operations of the uncached
    /// `eval_l_0(i, g * w^i)` path: `x = g * w^i` from the same `two_adic_subgroup` points the
    /// prover feeds it, then `(n * (x - ONE)).inverse()` — the same inverse of the same
    /// product, so every entry is bit-identical to the value it replaces. Entries are
    /// independent, so the parallel map changes nothing.
    fn build<F: Field>(degree_bits: usize, quotient_degree_bits: usize) -> Vec<F> {
        let n = F::from_canonical_usize(1 << degree_bits);
        F::two_adic_subgroup(degree_bits + quotient_degree_bits)
            .into_par_iter()
            .map(|x| (n * (F::coset_shift() * x - F::ONE)).inverse())
            .collect()
    }

    pub(super) fn l_0_denominator_inverses<F: Field>(
        degree_bits: usize,
        quotient_degree_bits: usize,
    ) -> Arc<Vec<F>> {
        let key = (TypeId::of::<F>(), degree_bits, quotient_degree_bits);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(entry) = cache.lock().unwrap().get(&key) {
            return Arc::clone(entry).downcast::<Vec<F>>().unwrap();
        }
        // Built outside the lock so a slow build never serializes other keys; concurrent
        // builders of the same key produce identical tables and the first insert wins.
        let table: Arc<Vec<F>> = Arc::new(build::<F>(degree_bits, quotient_degree_bits));
        let mut guard = cache.lock().unwrap();
        let entry = guard
            .entry(key)
            .or_insert_with(|| table as Arc<dyn Any + Send + Sync>);
        Arc::clone(entry).downcast::<Vec<F>>().unwrap()
    }
}

#[cfg(test)]
mod flat_chunk_products_tests {
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::{Field, PrimeField64};

    use super::divide_chunk_products;
    use crate::util::partial_products::quotient_chunk_products_into;

    type F = GoldilocksField;

    #[test]
    fn chunk_before_inversion_matches_individual_ratios() {
        for (width, chunk_size) in [(1, 2), (7, 3), (8, 8), (9, 8), (79, 8), (80, 8), (81, 8)] {
            let numerators = (0..width)
                .map(|i| F::from_canonical_usize(17 * i + 3))
                .collect::<Vec<_>>();
            let denominators = (0..width)
                .map(|i| F::from_canonical_usize(29 * i + 5))
                .collect::<Vec<_>>();

            let expected = numerators
                .chunks(chunk_size)
                .zip(denominators.chunks(chunk_size))
                .map(|(ns, ds)| {
                    ns.iter()
                        .zip(ds)
                        .map(|(&n, &d)| n * d.inverse())
                        .product::<F>()
                })
                .collect::<Vec<_>>();
            let mut actual = numerators
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().copied().product())
                .collect::<Vec<_>>();
            let denominator_products = denominators
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().copied().product())
                .collect::<Vec<_>>();
            let mut scratch = vec![F::ONE; width + 3];

            divide_chunk_products(&mut actual, &denominator_products, &mut scratch);
            assert_eq!(actual, expected, "width={width}, chunk_size={chunk_size}");
            assert_eq!(scratch.len(), actual.len());
        }
    }

    /// Deterministic, mostly-noncanonical values so raw-representation comparisons are
    /// meaningful.
    fn noncanonical_vec(len: usize, seed: u64) -> Vec<F> {
        (0..len as u64)
            .map(|i| {
                F::from_noncanonical_u64(
                    u64::MAX - seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(3 * i),
                )
            })
            .collect()
    }

    fn raw(values: &[F]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    /// The Z accumulation exactly as `wires_permutation_partial_products_and_zs` performs it,
    /// over one point's chunk products.
    fn accumulate_point(columns: &mut [Vec<F>], z_x: &mut F, chunk_products: &[F]) {
        let num_prods = columns.len() - 1;
        let mut acc = *z_x;
        for (k, &quotient_chunk_product) in chunk_products.iter().enumerate() {
            acc *= quotient_chunk_product;
            if k == num_prods {
                columns[k].push(*z_x);
                *z_x = acc;
            } else {
                columns[k].push(acc);
            }
        }
    }

    /// Differential test for the flat chunk-products refactor: the legacy pipeline (a fresh
    /// per-point `collect()` of chunk products into a `Vec<Vec<F>>`, then the Z chain over
    /// those rows) against the shipping pipeline (`quotient_chunk_products_into` writing
    /// batch-aligned slices of one flat buffer, then the Z chain over `chunks_exact`). Every
    /// column entry must match in raw representation. Uses the production shape (80 routed
    /// wires, quotient degree 8) and point counts spanning partial and multiple inversion
    /// batches.
    #[test]
    fn flat_chunk_products_and_z_chain_match_legacy() {
        const INV_BATCH: usize = 128;
        let num_routed_wires = 80usize;
        let degree = 8usize;
        let num_chunks = num_routed_wires.div_ceil(degree);
        let _num_prods = num_chunks - 1;

        for &n_points in &[1usize, 5, 128, 300] {
            let points: Vec<Vec<F>> = (0..n_points)
                .map(|i| noncanonical_vec(num_routed_wires, i as u64 + 1))
                .collect();

            // Legacy pipeline.
            let legacy_products: Vec<Vec<F>> = points
                .iter()
                .map(|quotient_values| {
                    quotient_values
                        .chunks(degree)
                        .map(|chunk| chunk.iter().copied().product())
                        .collect()
                })
                .collect();
            let mut legacy_columns: Vec<Vec<F>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(n_points))
                .collect();
            let mut z_x = F::ONE;
            for chunk_products in &legacy_products {
                assert_eq!(chunk_products.len(), num_chunks);
                accumulate_point(&mut legacy_columns, &mut z_x, chunk_products);
            }

            // Shipping pipeline: batch-aligned writes into one flat buffer, exactly as the
            // prover slices it.
            let mut flat = vec![F::ZERO; n_points * num_chunks];
            for (xs, out_chunk) in points
                .chunks(INV_BATCH)
                .zip(flat.chunks_mut(INV_BATCH * num_chunks))
            {
                for (t, quotient_values) in xs.iter().enumerate() {
                    quotient_chunk_products_into(
                        quotient_values,
                        degree,
                        &mut out_chunk[t * num_chunks..(t + 1) * num_chunks],
                    );
                }
            }
            let mut flat_columns: Vec<Vec<F>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(n_points))
                .collect();
            let mut z_x = F::ONE;
            for chunk_products in flat.chunks_exact(num_chunks) {
                accumulate_point(&mut flat_columns, &mut z_x, chunk_products);
            }

            for (k, (flat_column, legacy_column)) in
                flat_columns.iter().zip(&legacy_columns).enumerate()
            {
                assert_eq!(
                    raw(flat_column),
                    raw(legacy_column),
                    "column {k} mismatch for {n_points} points"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod l_0_table_tests {
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::Field;
    use plonky2_field::zero_poly_coset::ZeroPolyOnCoset;

    use super::l_0_table_cache::l_0_denominator_inverses;

    type F = GoldilocksField;

    const COMBOS: [(usize, usize); 5] = [(1, 1), (3, 2), (4, 3), (6, 2), (8, 3)];

    /// Every cached entry must equal the legacy per-point computation
    /// `(n * (g * w^i - 1)).inverse()` bit-for-bit (raw u64 representation, not just field
    /// value), for several (degree_bits, quotient_degree_bits) combos.
    #[test]
    fn table_entries_match_legacy_per_point_inversion() {
        for (degree_bits, quotient_degree_bits) in COMBOS {
            let table = l_0_denominator_inverses::<F>(degree_bits, quotient_degree_bits);
            let points = F::two_adic_subgroup(degree_bits + quotient_degree_bits);
            assert_eq!(table.len(), points.len());
            let n = F::from_canonical_usize(1 << degree_bits);
            for (i, &point) in points.iter().enumerate() {
                // The prover's shifted point, computed exactly as `compute_quotient_polys`
                // computes `shifted_xs`.
                let x = F::coset_shift() * point;
                let legacy = (n * (x - F::ONE)).inverse();
                assert_eq!(
                    table[i].0, legacy.0,
                    "entry {i} of table ({degree_bits}, {quotient_degree_bits})"
                );
            }
        }
    }

    /// `eval_l_0` with the table attached must return raw-identical values to the uncached
    /// path at every LDE point.
    #[test]
    fn eval_l_0_with_table_matches_uncached() {
        for (degree_bits, quotient_degree_bits) in COMBOS {
            let plain = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits);
            let cached = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits)
                .with_l_0_denominator_inverses(l_0_denominator_inverses::<F>(
                    degree_bits,
                    quotient_degree_bits,
                ));
            for (i, point) in F::two_adic_subgroup(degree_bits + quotient_degree_bits)
                .into_iter()
                .enumerate()
            {
                let x = F::coset_shift() * point;
                assert_eq!(
                    cached.eval_l_0(i, x).0,
                    plain.eval_l_0(i, x).0,
                    "eval_l_0({i}) for ({degree_bits}, {quotient_degree_bits})"
                );
            }
        }
    }
}
