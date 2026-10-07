#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;

use plonky2_field::polynomial::PolynomialCoeffs;

use super::circuit_builder::{LookupChallenges, NUM_COINS_LOOKUP};
use super::vars::EvaluationVarsBase;
use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::{Field, PrimeField64};
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::gates::gate::InterleavePairGate;
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::{LookupSelectors, UNUSED_SELECTOR};
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::plonk_common;
use crate::plonk::plonk_common::eval_l_0_circuit;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBaseBatch};
use crate::util::partial_products::{
    check_partial_products, check_partial_products_circuit, check_partial_products_into,
};
use crate::util::reducing::ReducingFactorTarget;
use crate::with_context;

/// Get the polynomial associated to a lookup table with current challenges.
pub(crate) fn get_lut_poly<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    lut_index: usize,
    deltas: &[F],
    degree: usize,
) -> PolynomialCoeffs<F> {
    let b = deltas[LookupChallenges::ChallengeB as usize];
    let mut coeffs = Vec::with_capacity(common_data.luts[lut_index].len());
    let n = common_data.luts[lut_index].len();
    let nb_slots = LookupTableGate::num_slots(&common_data.config);
    let nb_padded_elts = (nb_slots - n % nb_slots) % nb_slots;
    let (padding_inp, padding_out) = common_data.luts[lut_index][0];
    for (input, output) in common_data.luts[lut_index].iter() {
        coeffs.push(F::from_canonical_u16(*input) + b * F::from_canonical_u16(*output));
    }
    // Padding with the first element of the LUT.
    for _ in 0..nb_padded_elts {
        coeffs.push(F::from_canonical_u16(padding_inp) + b * F::from_canonical_u16(padding_out));
    }
    coeffs.append(&mut vec![F::ZERO; degree - (n + nb_padded_elts)]);
    coeffs.reverse();
    PolynomialCoeffs::new(coeffs)
}

/// Evaluate the vanishing polynomial at `x`. In this context, the vanishing polynomial is a random
/// linear combination of gate constraints, plus some other terms relating to the permutation
/// argument. All such terms should vanish on `H`.
pub(crate) fn eval_vanishing_poly<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    x: F::Extension,
    vars: EvaluationVars<F, D>,
    local_zs: &[F::Extension],
    next_zs: &[F::Extension],
    local_lookup_zs: &[F::Extension],
    next_lookup_zs: &[F::Extension],
    partial_products: &[F::Extension],
    s_sigmas: &[F::Extension],
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
    deltas: &[F],
) -> Vec<F::Extension> {
    let has_lookup = common_data.num_lookup_polys != 0;
    let max_degree = common_data.quotient_degree_factor;
    let num_prods = common_data.num_partial_products;

    let constraint_terms = evaluate_gate_constraints::<F, D>(common_data, vars);

    let lookup_selectors = &vars.local_constants[common_data.selectors_info.num_selectors()
        ..common_data.selectors_info.num_selectors() + common_data.num_lookup_selectors];

    // The L_0(x) (Z(x) - 1) vanishing terms.
    let mut vanishing_z_1_terms = Vec::new();

    // The terms checking the lookup constraints, if any.
    let mut vanishing_all_lookup_terms = if has_lookup {
        let num_sldc_polys = common_data.num_lookup_polys - 1;
        Vec::with_capacity(
            common_data.config.num_challenges * (4 + common_data.luts.len() + 2 * num_sldc_polys),
        )
    } else {
        Vec::new()
    };

    // The terms checking the partial products.
    let mut vanishing_partial_products_terms = Vec::new();

    let l_0_x = plonk_common::eval_l_0(common_data.degree(), x);

    for i in 0..common_data.config.num_challenges {
        let z_x = local_zs[i];
        let z_gx = next_zs[i];
        vanishing_z_1_terms.push(l_0_x * (z_x - F::Extension::ONE));

        if has_lookup {
            let cur_local_lookup_zs = &local_lookup_zs
                [common_data.num_lookup_polys * i..common_data.num_lookup_polys * (i + 1)];
            let cur_next_lookup_zs = &next_lookup_zs
                [common_data.num_lookup_polys * i..common_data.num_lookup_polys * (i + 1)];

            let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];

            let lookup_constraints = check_lookup_constraints(
                common_data,
                vars,
                cur_local_lookup_zs,
                cur_next_lookup_zs,
                lookup_selectors,
                cur_deltas.try_into().unwrap(),
            );

            vanishing_all_lookup_terms.extend(lookup_constraints);
        }

        let numerator_values = (0..common_data.config.num_routed_wires)
            .map(|j| {
                let wire_value = vars.local_wires[j];
                let k_i = common_data.k_is[j];
                let s_id = x.scalar_mul(k_i);
                wire_value + s_id.scalar_mul(betas[i]) + gammas[i].into()
            })
            .collect::<Vec<_>>();
        let denominator_values = (0..common_data.config.num_routed_wires)
            .map(|j| {
                let wire_value = vars.local_wires[j];
                let s_sigma = s_sigmas[j];
                wire_value + s_sigma.scalar_mul(betas[i]) + gammas[i].into()
            })
            .collect::<Vec<_>>();

        // The partial products considered for this iteration of `i`.
        let current_partial_products = &partial_products[i * num_prods..(i + 1) * num_prods];
        // Check the quotient partial products.
        let partial_product_checks = check_partial_products(
            &numerator_values,
            &denominator_values,
            current_partial_products,
            z_x,
            z_gx,
            max_degree,
        );
        vanishing_partial_products_terms.extend(partial_product_checks);
    }

    let vanishing_terms = [
        vanishing_z_1_terms,
        vanishing_partial_products_terms,
        vanishing_all_lookup_terms,
        constraint_terms,
    ]
    .concat();

    let alphas = &alphas.iter().map(|&a| a.into()).collect::<Vec<_>>();
    plonk_common::reduce_with_powers_multi(&vanishing_terms, alphas)
}

/// Reusable buffers for [`eval_vanishing_poly_base_batch`], hoisted out of the
/// per-batch hot loop so a rayon worker allocates them once per proof instead of
/// once per 32-point batch.
#[derive(Default)]
pub(crate) struct VanishingScratch<F> {
    pub numerator_values: Vec<F>,
    pub denominator_values: Vec<F>,
    /// Second challenge's product rows in the specialized two-challenge
    /// column evaluator. Kept separate so each challenge preserves its exact
    /// left-to-right multiplication order while sharing column loads.
    pub numerator_values_second: Vec<F>,
    pub denominator_values_second: Vec<F>,
    pub vanishing_z_1_terms: Vec<F>,
    pub vanishing_partial_products_terms: Vec<F>,
    pub vanishing_all_lookup_terms: Vec<F>,
    pub lookup_selectors: Vec<F>,
    pub constraint_terms_batch: Vec<F>,
    /// Reused selector-filter buffer across batches (survivor-list package).
    pub gate_filters: Vec<F>,
    /// Gate-major selector filters produced once per selector group with
    /// shared prefix/suffix products. Only gates selected by
    /// `shared_gate_filter_plan` have initialized rows.
    pub shared_gate_filters: Vec<F>,
    /// One batch-width suffix product used while constructing a selector
    /// group's shared filters.
    pub selector_filter_suffix: Vec<F>,
    /// Reused sum of the two filters consumed by the dense interleave pair.
    pub interleave_summed_filter: Vec<F>,
    /// Proof-local decision: a set bit means that gate's selector filter is
    /// cheaper to obtain from its group's shared prefix/suffix pass than from
    /// the gate-local product. Initialized once per Rayon worker.
    pub shared_gate_filter_plan: Vec<bool>,
}

/// Permutation-argument inputs for [`eval_vanishing_poly_base_batch`], in one
/// of two layouts.
///
/// `Rows` carries per-point slices (entry `k` is the row for point `k`) cut out
/// of point-major gather buffers; it is required whenever the circuit has
/// lookups, whose constraint evaluator consumes per-point rows.
///
/// `Cols` carries flat column-major buffers straight from `fill_lde_batch`'s
/// `PolyMajor` layout — column `c`'s batch values occupy `[c * n..(c + 1) * n]`
/// — which the no-lookup column evaluator reads directly, with no transpose in
/// between.
#[derive(Clone, Copy)]
pub(crate) enum PermutationBatch<'a, F> {
    Rows {
        local_zs_batch: &'a [&'a [F]],
        next_zs_batch: &'a [&'a [F]],
        partial_products_batch: &'a [&'a [F]],
        s_sigmas_batch: &'a [&'a [F]],
    },
    Cols {
        /// Z and partial-product columns `0..(num_partial_products + 1) * num_challenges`
        /// of the zs/partial-products commitment, i.e. `zs_range` followed by
        /// `partial_products_range`.
        zs_partial_products_cols: &'a [F],
        /// Z columns only (`zs_range`), gathered at the "next" indices.
        zs_next_cols: &'a [F],
        /// Sigma columns (`sigmas_range` of the constants-sigmas commitment).
        s_sigmas_cols: &'a [F],
    },
}

const INTERLEAVE_PAIR_WIRES: usize = 136;
const INTERLEAVE_PAIR_CONSTRAINTS: usize = 136;
const INTERLEAVE_OPS: usize = 4;
const UNINTERLEAVE_OPS: usize = 2;
const INTERLEAVE_PAIR_STACK_BATCH: usize = 32;
/// How many base-4 spread bits may share one delayed reduction. A stage seeded
/// by a sub-`2^64` representative and running `k` steps holds at most
/// `(2^64 - 1)(4·4^k - 1)/3`, which stays under `2^96` exactly while
/// `k <= 15`; `11` splits the 32-bit chain into 11/11/10 with wide margin.
const BASE4_SPREAD_STAGE: usize = 11;
const _: () = assert!(BASE4_SPREAD_STAGE <= 15 && BASE4_SPREAD_STAGE > 0);

/// Proof-local recognition of the exact CPU-owned pair used by the ranked
/// final-block circuit. The plan is evaluator-only and changes no circuit data.
pub(crate) struct InterleavePairPlan {
    pub(crate) interleave_index: usize,
    pub(crate) uninterleave_index: usize,
}

/// Returns a plan only when both exact shapes remain in the CPU survivor set.
/// If either gate moves to another backend, is duplicated, or changes shape,
/// both gates retain their ordinary independent evaluators.
pub(crate) fn interleave_pair_plan<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    cpu_gate_indices: &[usize],
) -> Option<InterleavePairPlan> {
    let mut tagged = cpu_gate_indices.iter().filter_map(|&index| {
        common_data.gates[index]
            .0
            .interleave_pair_gate()
            .map(|kind| (index, kind))
    });
    let first = tagged.next()?;
    let second = tagged.next()?;
    if tagged.next().is_some() {
        return None;
    }

    let (interleave_index, uninterleave_index) = match (first, second) {
        (
            (
                interleave_index,
                InterleavePairGate::Interleave {
                    num_ops: INTERLEAVE_OPS,
                },
            ),
            (
                uninterleave_index,
                InterleavePairGate::UninterleaveToU32 {
                    num_ops: UNINTERLEAVE_OPS,
                },
            ),
        ) if interleave_index < uninterleave_index => (interleave_index, uninterleave_index),
        _ => return None,
    };

    [interleave_index, uninterleave_index]
        .into_iter()
        .all(|index| {
            let gate = &common_data.gates[index].0;
            gate.num_wires() == INTERLEAVE_PAIR_WIRES
                && gate.num_constraints() == INTERLEAVE_PAIR_CONSTRAINTS
        })
        .then_some(InterleavePairPlan {
            interleave_index,
            uninterleave_index,
        })
}

/// Computes one selector filter column in the same factor order as
/// `Gate::eval_filtered_base_batch` without consuming the constants prefix.
fn fill_interleave_gate_filter<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    vars_batch: EvaluationVarsBaseBatch<F>,
    gate_index: usize,
    output: &mut [F],
) {
    let batch_size = vars_batch.len();
    debug_assert_eq!(output.len(), batch_size);
    let selector_index = common_data.selectors_info.selector_indices[gate_index];
    let selector_col = &vars_batch.local_constants[selector_index * batch_size..][..batch_size];
    let mut factors = common_data.selectors_info.groups[selector_index]
        .clone()
        .filter(|&index| index != gate_index)
        .chain((common_data.selectors_info.num_selectors() > 1).then_some(UNUSED_SELECTOR));

    if let Some(index) = factors.next() {
        let constant = F::from_canonical_usize(index);
        for (filter, &selector) in output.iter_mut().zip(selector_col) {
            *filter = constant - selector;
        }
    } else {
        output.fill(F::ONE);
    }
    for index in factors {
        let constant = F::from_canonical_usize(index);
        for (filter, &selector) in output.iter_mut().zip(selector_col) {
            *filter *= constant - selector;
        }
    }
}

/// Whether a selector group saves field operations when its active gates share
/// one prefix/suffix pass.
///
/// A gate-local filter with `f` factors performs `f` subtractions and
/// `f - 1` multiplications. For a group of `g` gates, the shared construction
/// performs two `g - 1` factor passes, one multiplication per requested output,
/// and (when there is more than one selector column) one common unused-selector
/// subtraction. The estimate exactly matches the loops in
/// [`fill_shared_selector_group_filters`]; strict inequality keeps small or
/// heavily-offloaded groups on the existing path.
#[inline]
fn shared_selector_group_is_cheaper(
    group_len: usize,
    active_gates: usize,
    include_unused_selector: bool,
) -> bool {
    if group_len == 0 || active_gates < 2 {
        return false;
    }
    let unused = usize::from(include_unused_selector);
    let factors_per_gate = group_len - 1 + unused;
    if factors_per_gate == 0 {
        return false;
    }
    let local_ops = active_gates * (2 * factors_per_gate - 1);
    let shared_ops = 4 * (group_len - 1) + unused + active_gates;
    shared_ops < local_ops
}

/// Selects the CPU-owned gates whose selector filters should share their
/// group's factor products. The plan depends only on the circuit shape and the
/// proof's already-computed CPU gate list, so each worker derives it once and
/// reuses it for every quotient batch it receives.
fn prepare_shared_gate_filter_plan<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    cpu_gate_indices: &[usize],
    plan: &mut Vec<bool>,
) {
    plan.clear();
    plan.resize(common_data.gates.len(), false);
    let include_unused_selector = common_data.selectors_info.num_selectors() > 1;
    for group in &common_data.selectors_info.groups {
        let active_gates = cpu_gate_indices
            .iter()
            .filter(|&&gate| group.contains(&gate))
            .count();
        if shared_selector_group_is_cheaper(group.len(), active_gates, include_unused_selector) {
            for &gate in cpu_gate_indices {
                if group.contains(&gate) {
                    plan[gate] = true;
                }
            }
        }
    }
}

/// Computes selected gate filters for one selector group with shared
/// prefix/suffix products.
///
/// For gate `i` this writes
/// `prod(j - s, j in group, j != i) * (UNUSED_SELECTOR - s)?`, exactly the
/// polynomial used by `Gate::eval_filtered_base_batch`. Prefixes contain the
/// factors below `i`; the descending suffix contains those above it and the
/// optional unused-selector factor. This changes association only, so the
/// result is the same field element while each group factor is formed a
/// constant number of times instead of once per active gate.
fn fill_shared_selector_group_filters<F: Field>(
    selector_col: &[F],
    group: core::ops::Range<usize>,
    include_unused_selector: bool,
    active_gate_plan: &[bool],
    gate_filters: &mut [F],
    suffix: &mut [F],
) {
    let batch_size = selector_col.len();
    debug_assert!(!group.is_empty());
    debug_assert_eq!(suffix.len(), batch_size);
    debug_assert_eq!(gate_filters.len(), active_gate_plan.len() * batch_size);

    let first = group.start;
    gate_filters[first * batch_size..(first + 1) * batch_size].fill(F::ONE);
    for gate in first + 1..group.end {
        let factor_constant = F::from_canonical_usize(gate - 1);
        let split = gate * batch_size;
        let (prefixes, rest) = gate_filters.split_at_mut(split);
        let previous = &prefixes[(gate - 1) * batch_size..gate * batch_size];
        let current = &mut rest[..batch_size];
        for point in 0..batch_size {
            current[point] = previous[point] * (factor_constant - selector_col[point]);
        }
    }

    if include_unused_selector {
        let unused = F::from_canonical_usize(UNUSED_SELECTOR);
        for (value, &selector) in suffix.iter_mut().zip(selector_col) {
            *value = unused - selector;
        }
    } else {
        suffix.fill(F::ONE);
    }

    for gate in group.clone().rev() {
        if active_gate_plan[gate] {
            let output = &mut gate_filters[gate * batch_size..(gate + 1) * batch_size];
            for (filter, &tail) in output.iter_mut().zip(suffix.iter()) {
                *filter *= tail;
            }
        }
        if gate != first {
            let factor_constant = F::from_canonical_usize(gate);
            for (tail, &selector) in suffix.iter_mut().zip(selector_col) {
                *tail *= factor_constant - selector;
            }
        }
    }
}

/// Fills every selector group selected by `plan`. The output is gate-major:
/// gate `i` owns `i * batch_size..(i + 1) * batch_size`.
fn fill_shared_gate_filters<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    vars_batch: EvaluationVarsBaseBatch<F>,
    plan: &[bool],
    gate_filters: &mut Vec<F>,
    suffix: &mut Vec<F>,
) {
    let batch_size = vars_batch.len();
    debug_assert_eq!(plan.len(), common_data.gates.len());
    if gate_filters.len() != common_data.gates.len() * batch_size {
        gate_filters.resize(common_data.gates.len() * batch_size, F::ZERO);
    }
    if suffix.len() != batch_size {
        suffix.resize(batch_size, F::ZERO);
    }

    let include_unused_selector = common_data.selectors_info.num_selectors() > 1;
    for (selector_index, group) in common_data.selectors_info.groups.iter().enumerate() {
        if !plan[group.clone()].iter().any(|&active| active) {
            continue;
        }
        let selector_col = &vars_batch.local_constants[selector_index * batch_size..][..batch_size];
        fill_shared_selector_group_filters(
            selector_col,
            group.clone(),
            include_unused_selector,
            plan,
            gate_filters,
            suffix,
        );
    }
}

/// Evaluates the exact Interleave(4) + Uninterleave(2) pair with one traversal
/// of their shared 128 bit columns while retaining the ordinary 136-row dense
/// matrix and its existing alpha reducer.
///
/// The first 32-bit half of each uninterleave operation lands two rows after
/// the corresponding interleave range row, so those values use two packed
/// MACs. The second half lands on the same row and uses one packed MAC with the
/// pre-summed filters. No range column survives past the current bit.
///
/// The three per-bit recompositions (`x`, the base-4 spread, and the two
/// parity columns) are accumulated in 128 bit integers and reduced once per
/// chain instead of once per bit. Each accumulator is written and read only by
/// its own recurrence inside the bit loop, so deferring its reduction reorders
/// nothing; the range column and the packed MACs are untouched.
///
/// Bounds, derived from `raw < 2^64` only — bit wires are *not* restricted to
/// `{0, 1}` on the quotient domain, so no Boolean argument is used:
///
/// - `x` and each parity column are 32-term base-2 chains, so the exact
///   accumulator is at most `(2^32 - 1)(2^64 - 1) = 2^96 - 2^64 - 2^32 + 1`,
///   strictly below `2^96`. A parity column spans two operations (it is seeded
///   once, before the operation loop), which is why its wide lane is hoisted
///   out of that loop.
/// - the base-4 spread chain would reach `((4^32 - 1)/3)(2^64 - 1) > 2^126`,
///   so it is reduced every `BASE4_SPREAD_STAGE` bits. A stage seeded by a
///   sub-`2^64` representative and running `k` steps holds at most
///   `(2^64 - 1)(4·4^k - 1)/3`; with `k = 11` that is below `2^87`.
///
/// Staying below `2^96` is what makes this bit-identical rather than merely
/// congruent: the high 32 bits of the accumulator are then zero, so
/// `from_noncanonical_u128` skips its borrow correction and lands on exactly
/// the representative the multiply/add chain produces, including for the
/// non-canonical `value + ORDER` representatives that arise when the chain's
/// field value is below `2^32 - 1`.
fn eval_interleave_pair_dense_fused<F: PrimeField64>(
    wires: &[F],
    batch_size: usize,
    interleave_filter: &[F],
    uninterleave_filter: &[F],
    summed_filter: &[F],
    combined: &mut [F],
) {
    debug_assert_eq!(interleave_filter.len(), batch_size);
    debug_assert_eq!(uninterleave_filter.len(), batch_size);
    debug_assert_eq!(summed_filter.len(), batch_size);
    debug_assert!(wires.len() >= INTERLEAVE_PAIR_WIRES * batch_size);
    debug_assert!(combined.len() >= INTERLEAVE_PAIR_CONSTRAINTS * batch_size);

    const STACK_COLS: usize = 10;
    let required = STACK_COLS * batch_size;
    let mut stack = [F::ZERO; STACK_COLS * INTERLEAVE_PAIR_STACK_BATCH];
    let mut heap;
    let scratch: &mut [F] = if batch_size <= INTERLEAVE_PAIR_STACK_BATCH {
        &mut stack[..required]
    } else {
        heap = vec![F::ZERO; required];
        &mut heap
    };
    let (x_accumulators, rest) = scratch.split_at_mut(INTERLEAVE_OPS * batch_size);
    let (parity_accumulators, rest) = rest.split_at_mut(4 * batch_size);
    let (base4_accumulator, range) = rest.split_at_mut(batch_size);

    // Wide lanes for the three delayed chains: one `x` lane and one base-4
    // spread lane (both reset per operation), plus the four parity lanes,
    // which are seeded once because a parity chain spans two operations.
    const WIDE_COLS: usize = 6;
    let wide_required = WIDE_COLS * batch_size;
    let mut wide_stack = [0u128; WIDE_COLS * INTERLEAVE_PAIR_STACK_BATCH];
    let mut wide_heap;
    let wide: &mut [u128] = if batch_size <= INTERLEAVE_PAIR_STACK_BATCH {
        &mut wide_stack[..wide_required]
    } else {
        wide_heap = vec![0u128; wide_required];
        &mut wide_heap
    };
    let (wide_x, rest) = wide.split_at_mut(batch_size);
    let (wide_spread, wide_parity) = rest.split_at_mut(batch_size);
    wide_parity.fill(0);

    for operation in 0..INTERLEAVE_OPS {
        let interleave_row = operation * 34;
        let bit_offset = operation * 32;
        let x = &mut x_accumulators[operation * batch_size..(operation + 1) * batch_size];
        wide_x.fill(0);
        wide_spread.fill(0);

        for bit_index in 0..32 {
            let bit_number = bit_offset + bit_index;
            let bit_col = &wires[(8 + bit_number) * batch_size..][..batch_size];
            let parity_index = 2 * (operation / 2) + bit_index % 2;
            let parity =
                &mut wide_parity[parity_index * batch_size..(parity_index + 1) * batch_size];
            for point in 0..batch_size {
                let bit = bit_col[point];
                let raw = bit.to_noncanonical_u64() as u128;
                wide_x[point] = (wide_x[point] << 1) + raw;
                wide_spread[point] = (wide_spread[point] << 2) + raw;
                parity[point] = (parity[point] << 1) + raw;
                range[point] = bit * (bit - F::ONE);
            }
            if bit_index % BASE4_SPREAD_STAGE == BASE4_SPREAD_STAGE - 1 {
                for point in 0..batch_size {
                    debug_assert!(wide_spread[point] < 1u128 << 96);
                    wide_spread[point] =
                        F::from_noncanonical_u128(wide_spread[point]).to_noncanonical_u64() as u128;
                }
            }

            let row = interleave_row + 2 + bit_index;
            let interleave_output = &mut combined[row * batch_size..(row + 1) * batch_size];
            if operation % 2 == 0 {
                batch_multiply_add_inplace(interleave_output, range, interleave_filter);
                let uninterleave_row = row + 2;
                let uninterleave_output = &mut combined
                    [uninterleave_row * batch_size..(uninterleave_row + 1) * batch_size];
                batch_multiply_add_inplace(uninterleave_output, range, uninterleave_filter);
            } else {
                batch_multiply_add_inplace(interleave_output, range, summed_filter);
            }
        }

        let x_col = &wires[(2 * operation) * batch_size..][..batch_size];
        for point in 0..batch_size {
            debug_assert!(wide_x[point] < 1u128 << 96);
            x[point] = F::from_noncanonical_u128(wide_x[point]);
            range[point] = x[point] - x_col[point];
        }
        let output = &mut combined[interleave_row * batch_size..(interleave_row + 1) * batch_size];
        batch_multiply_add_inplace(output, range, interleave_filter);

        let spread_col = &wires[(2 * operation + 1) * batch_size..][..batch_size];
        for point in 0..batch_size {
            debug_assert!(wide_spread[point] < 1u128 << 96);
            base4_accumulator[point] = F::from_noncanonical_u128(wide_spread[point]);
            range[point] = base4_accumulator[point] - spread_col[point];
        }
        let output =
            &mut combined[(interleave_row + 1) * batch_size..(interleave_row + 2) * batch_size];
        batch_multiply_add_inplace(output, range, interleave_filter);
    }

    for (accumulator, &lane) in parity_accumulators.iter_mut().zip(wide_parity.iter()) {
        debug_assert!(lane < 1u128 << 96);
        *accumulator = F::from_noncanonical_u128(lane);
    }

    let split = F::from_canonical_u64(1 << 32u64);
    let u32_max = F::from_canonical_u32(u32::MAX);
    for operation in 0..UNINTERLEAVE_OPS {
        let row = operation * 68;
        let high = &x_accumulators[(2 * operation) * batch_size..(2 * operation + 1) * batch_size];
        let low =
            &x_accumulators[(2 * operation + 1) * batch_size..(2 * operation + 2) * batch_size];
        let evens =
            &parity_accumulators[(2 * operation) * batch_size..(2 * operation + 1) * batch_size];
        let odds = &parity_accumulators
            [(2 * operation + 1) * batch_size..(2 * operation + 2) * batch_size];
        let interleaved = &wires[(4 * operation) * batch_size..][..batch_size];
        let x_evens = &wires[(4 * operation + 1) * batch_size..][..batch_size];
        let x_odds = &wires[(4 * operation + 2) * batch_size..][..batch_size];
        let inverse = &wires[(4 * operation + 3) * batch_size..][..batch_size];

        for point in 0..batch_size {
            range[point] = (inverse[point] * (u32_max - high[point]) - F::ONE) * low[point];
        }
        let output = &mut combined[row * batch_size..(row + 1) * batch_size];
        batch_multiply_add_inplace(output, range, uninterleave_filter);

        for point in 0..batch_size {
            range[point] = high[point] * split + low[point] - interleaved[point];
        }
        let output = &mut combined[(row + 1) * batch_size..(row + 2) * batch_size];
        batch_multiply_add_inplace(output, range, uninterleave_filter);

        for point in 0..batch_size {
            range[point] = evens[point] - x_evens[point];
        }
        let output = &mut combined[(row + 2) * batch_size..(row + 3) * batch_size];
        batch_multiply_add_inplace(output, range, uninterleave_filter);

        for point in 0..batch_size {
            range[point] = odds[point] - x_odds[point];
        }
        let output = &mut combined[(row + 3) * batch_size..(row + 4) * batch_size];
        batch_multiply_add_inplace(output, range, uninterleave_filter);
    }
}

/// Reduces the per-row constraint terms into `res_out`.
///
/// `clear_as_consumed` zeroes each term as it is read. Every element is read
/// exactly once here, so a caller that owns the buffer as reusable scratch gets
/// it back all-zero and can skip re-zeroing it before the next batch. Doing it
/// on this walk rather than in a separate pass is what makes it nearly free:
/// the line is already resident and dirty from the read. Callers that do not
/// own the buffer, or that want to inspect it afterwards, pass `false`.
fn reduce_gate_constraints_base_batch<F: Field>(
    constraint_terms_batch: &mut [F],
    batch_size: usize,
    alphas: &[F],
    res_out: &mut [F],
    res_out_is_zero_seed: bool,
    clear_as_consumed: bool,
) {
    debug_assert!(batch_size > 0);
    debug_assert_eq!(constraint_terms_batch.len() % batch_size, 0);
    debug_assert_eq!(res_out.len(), batch_size * alphas.len());

    // When `res_out` is known to be an all-zero (or uninitialized) seed, the
    // first reversed row can be *assigned* rather than accumulated: with
    // `*value == F::ZERO`, `term.multiply_accumulate(ZERO, alpha)` is
    // `reduce128(term as u128)`, whose high half is zero, so `reduce128`
    // returns `term` unchanged — a raw-limb-identical copy, not merely a
    // field-value-identical one. Assigning it makes every slot of `res_out`
    // stored before it is read, which lets the quotient caller skip
    // zero-filling the accumulator entirely, and drops one multiply per
    // (point, challenge) on that row.
    //
    // This is NOT valid for a caller that passes a nonzero running
    // accumulator, which the general contract permits, so it is opt-in.
    let mut rows = constraint_terms_batch.chunks_exact_mut(batch_size).rev();
    if alphas.len() == 2 {
        // Production always uses two challenges: load alphas once and walk
        // point-major output in exact pairs instead of rediscovering the
        // runtime slice length for every row and point.
        let alpha_0 = alphas[0];
        let alpha_1 = alphas[1];
        if res_out_is_zero_seed {
            match rows.next() {
                Some(first_row) => {
                    for (term_slot, result) in first_row.iter_mut().zip(res_out.chunks_exact_mut(2))
                    {
                        let term = *term_slot;
                        result[0] = term;
                        result[1] = term;
                        if clear_as_consumed {
                            *term_slot = F::ZERO;
                        }
                    }
                }
                None => res_out.fill(F::ZERO),
            }
        }
        for constraint_row in rows {
            for (term_slot, result) in constraint_row.iter_mut().zip(res_out.chunks_exact_mut(2)) {
                let term = *term_slot;
                result[0] = term.multiply_accumulate(result[0], alpha_0);
                result[1] = term.multiply_accumulate(result[1], alpha_1);
                if clear_as_consumed {
                    *term_slot = F::ZERO;
                }
            }
        }
        return;
    }

    if res_out_is_zero_seed {
        match rows.next() {
            Some(first_row) => {
                for (point, term_slot) in first_row.iter_mut().enumerate() {
                    let term = *term_slot;
                    let result = &mut res_out[point * alphas.len()..(point + 1) * alphas.len()];
                    result.fill(term);
                    if clear_as_consumed {
                        *term_slot = F::ZERO;
                    }
                }
            }
            // No constraint rows: preserve the "every slot written" contract.
            None => res_out.fill(F::ZERO),
        }
    }

    for constraint_row in rows {
        for (point, term_slot) in constraint_row.iter_mut().enumerate() {
            let term = *term_slot;
            let result = &mut res_out[point * alphas.len()..(point + 1) * alphas.len()];
            for (value, &alpha) in result.iter_mut().zip(alphas) {
                *value = term.multiply_accumulate(*value, alpha);
            }
            if clear_as_consumed {
                *term_slot = F::ZERO;
            }
        }
    }
}

#[inline(always)]
fn permutation_factor_fma<F: Field>(wire: F, beta: F, point: F, gamma: F) -> F {
    wire.multiply_accumulate(beta, point) + gamma
}

/// Like `eval_vanishing_poly`, but specialized for base field points. Batched.
///
/// Results are stored point-major: the challenges for point `k` occupy
/// `res_out[k * num_challenges..(k + 1) * num_challenges]`. `res_out` must be
/// zero-initialized by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_vanishing_poly_base_batch<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    indices_batch: &[usize],
    xs_batch: &[F],
    vars_batch: EvaluationVarsBaseBatch<F>,
    perm: PermutationBatch<'_, F>,
    local_lookup_zs_batch: &[&[F]],
    next_lookup_zs_batch: &[&[F]],
    betas: &[F],
    gammas: &[F],
    beta_k_is: &[F],
    deltas: &[F],
    alphas: &[F],
    cpu_gate_indices: &[usize],
    cpu_num_gate_constraints: usize,
    interleave_pair: Option<&InterleavePairPlan>,
    permutation_products_offloaded: bool,
    permutation_gate_scales: &[F],
    z_h_on_coset: &ZeroPolyOnCoset<F>,
    lut_re_poly_evals: &[&[F]],
    scratch: &mut VanishingScratch<F>,
    res_out: &mut [F],
) {
    let has_lookup = common_data.num_lookup_polys != 0;

    let n = indices_batch.len();
    assert_eq!(xs_batch.len(), n);
    assert_eq!(vars_batch.len(), n);
    if has_lookup {
        assert_eq!(local_lookup_zs_batch.len(), n);
        assert_eq!(next_lookup_zs_batch.len(), n);
    } else {
        assert_eq!(local_lookup_zs_batch.len(), 0);
        assert_eq!(next_lookup_zs_batch.len(), 0);
    }

    let max_degree = common_data.quotient_degree_factor;
    let num_prods = common_data.num_partial_products;

    let num_gate_constraints = common_data.num_gate_constraints;

    evaluate_gate_constraints_base_batch_into_cpu_gates::<F, D>(
        common_data,
        vars_batch,
        &mut scratch.constraint_terms_batch,
        cpu_gate_indices,
        &mut scratch.gate_filters,
        &mut scratch.shared_gate_filters,
        &mut scratch.selector_filter_suffix,
        &mut scratch.interleave_summed_filter,
        &mut scratch.shared_gate_filter_plan,
        cpu_num_gate_constraints,
        interleave_pair,
    );
    let constraint_terms_batch = &mut scratch.constraint_terms_batch;
    // `<=`, not `==`: the buffer is sized by the widest gate still on the CPU,
    // which is at most `num_gate_constraints` and strictly less whenever a
    // widest gate has been offloaded.
    debug_assert!(constraint_terms_batch.len() <= n * num_gate_constraints);
    debug_assert_eq!(constraint_terms_batch.len() % n, 0);

    let num_challenges = common_data.config.num_challenges;
    let num_routed_wires = common_data.config.num_routed_wires;
    debug_assert_eq!(betas.len(), num_challenges);
    debug_assert_eq!(gammas.len(), num_challenges);
    debug_assert_eq!(beta_k_is.len(), num_challenges * num_routed_wires);
    reduce_gate_constraints_base_batch(constraint_terms_batch, n, alphas, res_out, true, true);

    if permutation_products_offloaded {
        assert!(!has_lookup, "lookup permutation products stay on the CPU");
        let PermutationBatch::Cols {
            zs_partial_products_cols,
            ..
        } = perm
        else {
            unreachable!("Metal permutation offload requires column-major inputs")
        };
        assert!(zs_partial_products_cols.len() >= num_challenges * n);
        // Global constraint order is
        //   [z1_0, z1_1, partial(0,0..chunks), partial(1,0..chunks), gates...].
        // The Metal job emits only the partial rows at their powers 2..P-1.
        // Shift the CPU gate-only Horner polynomial by P, then add the two
        // inexpensive L_0 rows here. This deletes every routed-wire/sigma/
        // partial-product traversal from the CPU without moving a transcript
        // barrier or changing an alpha exponent.
        assert_eq!(permutation_gate_scales.len(), num_challenges);
        if num_challenges == 2 {
            let alpha0 = alphas[0];
            let alpha1 = alphas[1];
            let gate_scale0 = permutation_gate_scales[0];
            let gate_scale1 = permutation_gate_scales[1];
            for k in 0..n {
                let l_0_x = z_h_on_coset.eval_l_0(indices_batch[k], xs_batch[k]);
                let z1_0 = l_0_x * zs_partial_products_cols[k].sub_one();
                let z1_1 = l_0_x * zs_partial_products_cols[n + k].sub_one();
                let idx = k * 2;
                let val0 = res_out[idx];
                let val1 = res_out[idx + 1];
                res_out[idx] = z1_0 + z1_1 * alpha0 + val0 * gate_scale0;
                res_out[idx + 1] = z1_0 + z1_1 * alpha1 + val1 * gate_scale1;
            }
        } else {
            for k in 0..n {
                let l_0_x = z_h_on_coset.eval_l_0(indices_batch[k], xs_batch[k]);
                let z1_0 = l_0_x * zs_partial_products_cols[k].sub_one();
                let z1_1 = l_0_x * zs_partial_products_cols[n + k].sub_one();
                let point = &mut res_out[k * num_challenges..(k + 1) * num_challenges];
                for ((&alpha, &gate_scale), value) in alphas
                    .iter()
                    .zip(permutation_gate_scales)
                    .zip(point.iter_mut())
                {
                    let gate_terms = *value * gate_scale;
                    *value = z1_0 + z1_1 * alpha + gate_terms;
                }
            }
        }
        return;
    }

    let numerator_values = &mut scratch.numerator_values;
    let denominator_values = &mut scratch.denominator_values;
    numerator_values.clear();
    denominator_values.clear();

    // The L_0(x) (Z(x) - 1) vanishing terms.
    let vanishing_z_1_terms = &mut scratch.vanishing_z_1_terms;
    // The terms checking the partial products.
    let vanishing_partial_products_terms = &mut scratch.vanishing_partial_products_terms;
    // The terms checking the lookup constraints.
    let vanishing_all_lookup_terms = &mut scratch.vanishing_all_lookup_terms;
    vanishing_z_1_terms.clear();
    vanishing_all_lookup_terms.clear();

    debug_assert_eq!(res_out.len(), n * num_challenges);

    // Column-major evaluation of the permutation-argument terms when the
    // circuit has no lookups. Per point this performs exactly the same field
    // operations, in the same order, as the per-point loop below; it only
    // reads each wire, sigma, Z and partial-product column contiguously across
    // the batch — straight out of the `PolyMajor` gather buffers — instead of
    // through per-point strided views. Term rows are laid out in the same
    // order the per-point loop pushes terms, and the final alpha reduction
    // consumes them in the same reversed order, so `res_out` is
    // value-identical.
    if let PermutationBatch::Cols {
        zs_partial_products_cols,
        zs_next_cols,
        s_sigmas_cols,
    } = perm
    {
        assert!(
            !has_lookup,
            "lookup circuits need per-point permutation rows"
        );
        assert_eq!(
            zs_partial_products_cols.len(),
            (num_prods + 1) * num_challenges * n
        );
        assert_eq!(zs_next_cols.len(), num_challenges * n);
        assert_eq!(s_sigmas_cols.len(), num_routed_wires * n);

        let wires = vars_batch.local_wires;
        let chunk_size = max_degree;
        let num_chunks = num_routed_wires.div_ceil(chunk_size);
        debug_assert_eq!(num_chunks, num_prods + 1);
        // The per-point loop chains ALL z_1 terms (i ascending) before ALL
        // partial-product terms (i-major, chunk-minor); the row layout must
        // match exactly so each term meets the same alpha power.
        let num_rows = num_challenges * (1 + num_chunks);

        let term_rows = &mut scratch.vanishing_partial_products_terms;
        if term_rows.len() != num_rows * n {
            term_rows.resize(num_rows * n, F::ZERO);
        }

        let l_0_xs = &mut scratch.vanishing_z_1_terms;
        l_0_xs.clear();
        l_0_xs.extend(
            indices_batch
                .iter()
                .zip(xs_batch)
                .map(|(&index, &x)| z_h_on_coset.eval_l_0(index, x)),
        );

        let num_prod = &mut scratch.numerator_values;
        let den_prod = &mut scratch.denominator_values;
        let num_prod_second = &mut scratch.numerator_values_second;
        let den_prod_second = &mut scratch.denominator_values_second;

        // The accumulator chain for challenge `i` is the column sequence
        // [Z_i(x) | partials i*num_prods..(i+1)*num_prods | Z_i(gx)], read
        // directly out of the PolyMajor buffers: `zs_range` starts at column
        // 0 and `partial_products_range` at column `num_challenges`, and the
        // next-Z buffer was gathered over `zs_range` alone. Column `c` here
        // holds exactly the values the old per-point transpose scattered into
        // `acc_cols[i * acc_stride + c * n..][..n]`.
        let acc_col = |i: usize, c: usize| -> &[F] {
            if c == 0 {
                &zs_partial_products_cols[i * n..][..n]
            } else if c <= num_prods {
                &zs_partial_products_cols[(num_challenges + i * num_prods + (c - 1)) * n..][..n]
            } else {
                &zs_next_cols[i * n..][..n]
            }
        };

        for i in 0..num_challenges {
            let z_col = acc_col(i, 0);
            let z1_row = &mut term_rows[i * n..][..n];
            for k in 0..n {
                z1_row[k] = l_0_xs[k] * z_col[k].sub_one();
            }
        }

        if num_challenges == 2 {
            // Both production challenges traverse the same routed-wire,
            // sigma and point columns. Keep their arithmetic chains separate,
            // but update them side by side so every shared input is loaded
            // once. Operations within either challenge retain the old exact
            // j-ascending order and expression association.
            let beta_0 = betas[0];
            let beta_1 = betas[1];
            let gamma_0 = gammas[0];
            let gamma_1 = gammas[1];
            for c in 0..num_chunks {
                let j_start = c * chunk_size;
                let j_end = ((c + 1) * chunk_size).min(num_routed_wires);

                num_prod.clear();
                den_prod.clear();
                num_prod_second.clear();
                den_prod_second.clear();
                {
                    let wire_col = &wires[j_start * n..][..n];
                    let sigma_col = &s_sigmas_cols[j_start * n..][..n];
                    let beta_k_0 = beta_k_is[j_start];
                    let beta_k_1 = beta_k_is[num_routed_wires + j_start];
                    for k in 0..n {
                        let wire = wire_col[k];
                        let sigma = sigma_col[k];
                        let x = xs_batch[k];
                        num_prod.push(permutation_factor_fma(wire, beta_k_0, x, gamma_0));
                        den_prod.push(permutation_factor_fma(wire, beta_0, sigma, gamma_0));
                        num_prod_second.push(permutation_factor_fma(wire, beta_k_1, x, gamma_1));
                        den_prod_second.push(permutation_factor_fma(wire, beta_1, sigma, gamma_1));
                    }
                }
                for j in j_start + 1..j_end {
                    let wire_col = &wires[j * n..][..n];
                    let sigma_col = &s_sigmas_cols[j * n..][..n];
                    let beta_k_0 = beta_k_is[j];
                    let beta_k_1 = beta_k_is[num_routed_wires + j];
                    for k in 0..n {
                        let wire = wire_col[k];
                        let sigma = sigma_col[k];
                        let x = xs_batch[k];
                        num_prod[k] *= permutation_factor_fma(wire, beta_k_0, x, gamma_0);
                        den_prod[k] *= permutation_factor_fma(wire, beta_0, sigma, gamma_0);
                        num_prod_second[k] *= permutation_factor_fma(wire, beta_k_1, x, gamma_1);
                        den_prod_second[k] *= permutation_factor_fma(wire, beta_1, sigma, gamma_1);
                    }
                }

                let row_0 = (num_challenges + c) * n;
                let row_1 = (num_challenges + num_chunks + c) * n;
                let (rows_before_1, rows_from_1) = term_rows.split_at_mut(row_1);
                let row_0 = &mut rows_before_1[row_0..row_0 + n];
                let row_1 = &mut rows_from_1[..n];
                let prev_0 = acc_col(0, c);
                let next_0 = acc_col(0, c + 1);
                let prev_1 = acc_col(1, c);
                let next_1 = acc_col(1, c + 1);
                for k in 0..n {
                    row_0[k] = prev_0[k] * num_prod[k] - next_0[k] * den_prod[k];
                    row_1[k] = prev_1[k] * num_prod_second[k] - next_1[k] * den_prod_second[k];
                }
            }
        } else {
            for i in 0..num_challenges {
                for c in 0..num_chunks {
                    let j_start = c * chunk_size;
                    let j_end = ((c + 1) * chunk_size).min(num_routed_wires);
                    let beta = betas[i];
                    let gamma = gammas[i];

                    // The first factor of each chunk lands by direct
                    // assignment; multiplying it into `ONE` is a raw-limb
                    // identity for Goldilocks.
                    num_prod.clear();
                    den_prod.clear();
                    {
                        let wire_col = &wires[j_start * n..][..n];
                        let sigma_col = &s_sigmas_cols[j_start * n..][..n];
                        let beta_k_i = beta_k_is[i * num_routed_wires + j_start];
                        for k in 0..n {
                            num_prod.push(wire_col[k] + beta_k_i * xs_batch[k] + gamma);
                            den_prod.push(wire_col[k] + beta * sigma_col[k] + gamma);
                        }
                    }
                    for j in j_start + 1..j_end {
                        let wire_col = &wires[j * n..][..n];
                        let sigma_col = &s_sigmas_cols[j * n..][..n];
                        let beta_k_i = beta_k_is[i * num_routed_wires + j];
                        for k in 0..n {
                            num_prod[k] *= wire_col[k] + beta_k_i * xs_batch[k] + gamma;
                            den_prod[k] *= wire_col[k] + beta * sigma_col[k] + gamma;
                        }
                    }

                    let row = &mut term_rows[(num_challenges + i * num_chunks + c) * n..][..n];
                    let prev_col = acc_col(i, c);
                    let next_col = acc_col(i, c + 1);
                    for k in 0..n {
                        row[k] = prev_col[k] * num_prod[k] - next_col[k] * den_prod[k];
                    }
                }
            }
        }

        // Same reversed-order Horner reduction as the per-point loop.
        for t in (0..num_rows).rev() {
            let row = &term_rows[t * n..][..n];
            for k in 0..n {
                let res = &mut res_out[k * num_challenges..(k + 1) * num_challenges];
                for (c, &alpha) in res.iter_mut().zip(alphas) {
                    *c = row[k].multiply_accumulate(*c, alpha);
                }
            }
        }

        l_0_xs.clear();
        num_prod.clear();
        den_prod.clear();
        num_prod_second.clear();
        den_prod_second.clear();
        return;
    }

    let PermutationBatch::Rows {
        local_zs_batch,
        next_zs_batch,
        partial_products_batch,
        s_sigmas_batch,
    } = perm
    else {
        unreachable!("Cols variant returns above")
    };
    vanishing_partial_products_terms.clear();
    assert_eq!(local_zs_batch.len(), n);
    assert_eq!(next_zs_batch.len(), n);
    assert_eq!(partial_products_batch.len(), n);
    assert_eq!(s_sigmas_batch.len(), n);

    for k in 0..n {
        let index = indices_batch[k];
        let x = xs_batch[k];
        let vars = vars_batch.view(k);

        let lookup_selectors = &mut scratch.lookup_selectors;
        lookup_selectors.clear();
        lookup_selectors.extend(
            (0..common_data.num_lookup_selectors)
                .map(|i| vars.local_constants[common_data.selectors_info.num_selectors() + i]),
        );

        let local_zs = local_zs_batch[k];
        let next_zs = next_zs_batch[k];
        let local_lookup_zs = if has_lookup {
            local_lookup_zs_batch[k]
        } else {
            &[]
        };

        let next_lookup_zs = if has_lookup {
            next_lookup_zs_batch[k]
        } else {
            &[]
        };

        let partial_products = partial_products_batch[k];
        let s_sigmas = s_sigmas_batch[k];

        let l_0_x = z_h_on_coset.eval_l_0(index, x);
        for i in 0..num_challenges {
            let z_x = local_zs[i];
            let z_gx = next_zs[i];
            vanishing_z_1_terms.push(l_0_x * z_x.sub_one());

            // If there are lookups in the circuit, then we add the lookup constraints.
            if has_lookup {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];

                let cur_local_lookup_zs = &local_lookup_zs
                    [common_data.num_lookup_polys * i..common_data.num_lookup_polys * (i + 1)];
                let cur_next_lookup_zs = &next_lookup_zs
                    [common_data.num_lookup_polys * i..common_data.num_lookup_polys * (i + 1)];

                let lookup_constraints = check_lookup_constraints_batch(
                    common_data,
                    vars,
                    cur_local_lookup_zs,
                    cur_next_lookup_zs,
                    lookup_selectors,
                    cur_deltas.try_into().unwrap(),
                    lut_re_poly_evals[i],
                );
                vanishing_all_lookup_terms.extend(lookup_constraints);
            }

            numerator_values.extend((0..num_routed_wires).map(|j| {
                let wire_value = vars.local_wires[j];
                let beta_k_i = beta_k_is[i * num_routed_wires + j];
                wire_value + beta_k_i * x + gammas[i]
            }));
            denominator_values.extend((0..num_routed_wires).map(|j| {
                let wire_value = vars.local_wires[j];
                let s_sigma = s_sigmas[j];
                wire_value + betas[i] * s_sigma + gammas[i]
            }));

            // The partial products considered for this iteration of `i`.
            let current_partial_products = &partial_products[i * num_prods..(i + 1) * num_prods];
            // Check the numerator partial products, appending the terms directly to the
            // worker-local scratch vector instead of collecting a fresh ten-element `Vec`
            // per point and challenge. Not cleared between challenges: challenge `i + 1`
            // must append after challenge `i` for the same point.
            check_partial_products_into(
                numerator_values,
                denominator_values,
                current_partial_products,
                z_x,
                z_gx,
                max_degree,
                vanishing_partial_products_terms,
            );

            numerator_values.clear();
            denominator_values.clear();
        }

        let vanishing_terms = vanishing_z_1_terms
            .iter()
            .chain(vanishing_partial_products_terms.iter())
            .chain(vanishing_all_lookup_terms.iter());
        let res = &mut res_out[k * num_challenges..(k + 1) * num_challenges];
        for &term in vanishing_terms.rev() {
            for (c, &alpha) in res.iter_mut().zip(alphas) {
                *c = term.multiply_accumulate(*c, alpha);
            }
        }

        vanishing_z_1_terms.clear();
        vanishing_partial_products_terms.clear();
        vanishing_all_lookup_terms.clear();
    }
}

/// Evaluates all lookup constraints, based on the logarithmic derivatives paper (<https://eprint.iacr.org/2022/1530.pdf>),
/// following the Tip5 paper's implementation (<https://eprint.iacr.org/2023/107.pdf>).
///
/// There are three polynomials to check:
/// - RE ensures the well formation of lookup tables;
/// - Sum is a running sum of m_i/(X - (input_i + a * output_i)) where (input_i, output_i) are input pairs in the lookup table (LUT);
/// - LDC is a running sum of 1/(X - (input_i + a * output_i)) where (input_i, output_i) are input pairs that look in the LUT.
///
/// Sum and LDC are broken down in partial polynomials to lower the constraint degree, similarly to the permutation argument.
/// They also share the same partial SLDC polynomials, so that the last SLDC value is Sum(end) - LDC(end). The final constraint
/// Sum(end) = LDC(end) becomes simply SLDC(end) = 0, and we can remove the LDC initial constraint.
pub fn check_lookup_constraints<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    vars: EvaluationVars<F, D>,
    local_lookup_zs: &[F::Extension],
    next_lookup_zs: &[F::Extension],
    lookup_selectors: &[F::Extension],
    deltas: &[F; 4],
) -> Vec<F::Extension> {
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let lu_degree = common_data.quotient_degree_factor - 1;
    let num_sldc_polys = local_lookup_zs.len() - 1;
    let lut_degree = num_lut_slots.div_ceil(num_sldc_polys);

    let mut constraints = Vec::with_capacity(4 + common_data.luts.len() + 2 * num_sldc_polys);

    // RE is the first polynomial stored.
    let z_re = local_lookup_zs[0];
    let next_z_re = next_lookup_zs[0];

    // Partial Sums and LDCs are both stored in the remaining SLDC polynomials.
    let z_x_lookup_sldcs = &local_lookup_zs[1..num_sldc_polys + 1];
    let z_gx_lookup_sldcs = &next_lookup_zs[1..num_sldc_polys + 1];

    let delta_challenge_a = F::Extension::from(deltas[LookupChallenges::ChallengeA as usize]);
    let delta_challenge_b = F::Extension::from(deltas[LookupChallenges::ChallengeB as usize]);

    // Compute all current looked and looking combos, i.e. the combos we need for the SLDC polynomials.
    let current_looked_combos: Vec<F::Extension> = (0..num_lut_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupTableGate::wire_ith_looked_inp(s)];
            let output_wire = vars.local_wires[LookupTableGate::wire_ith_looked_out(s)];
            input_wire + delta_challenge_a * output_wire
        })
        .collect();

    let current_looking_combos: Vec<F::Extension> = (0..num_lu_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupGate::wire_ith_looking_inp(s)];
            let output_wire = vars.local_wires[LookupGate::wire_ith_looking_out(s)];
            input_wire + delta_challenge_a * output_wire
        })
        .collect();

    // Compute all current lookup combos, i.e. the combos used to check that the LUT is correct.
    let current_lookup_combos: Vec<F::Extension> = (0..num_lut_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupTableGate::wire_ith_looked_inp(s)];
            let output_wire = vars.local_wires[LookupTableGate::wire_ith_looked_out(s)];
            input_wire + delta_challenge_b * output_wire
        })
        .collect();

    // Check last LDC constraint.
    constraints.push(
        lookup_selectors[LookupSelectors::LastLdc as usize] * z_x_lookup_sldcs[num_sldc_polys - 1],
    );

    // Check initial Sum constraint.
    constraints.push(lookup_selectors[LookupSelectors::InitSre as usize] * z_x_lookup_sldcs[0]);

    // Check initial RE constraint.
    constraints.push(lookup_selectors[LookupSelectors::InitSre as usize] * z_re);

    let current_delta = deltas[LookupChallenges::ChallengeDelta as usize];

    // Check final RE constraints for each different LUT.
    for r in LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors {
        let cur_ends_selector = lookup_selectors[r];
        let lut_row_number = common_data.luts[r - LookupSelectors::StartEnd as usize]
            .len()
            .div_ceil(num_lut_slots);
        let cur_function_eval = get_lut_poly(
            common_data,
            r - LookupSelectors::StartEnd as usize,
            deltas,
            num_lut_slots * lut_row_number,
        )
        .eval(current_delta);

        constraints.push(cur_ends_selector * (z_re - cur_function_eval.into()))
    }

    // Check RE row transition constraint.
    let mut cur_sum = next_z_re;
    for elt in &current_lookup_combos {
        cur_sum =
            cur_sum * F::Extension::from(deltas[LookupChallenges::ChallengeDelta as usize]) + *elt;
    }
    let unfiltered_re_line = z_re - cur_sum;

    constraints.push(lookup_selectors[LookupSelectors::TransSre as usize] * unfiltered_re_line);

    for poly in 0..num_sldc_polys {
        // Compute prod(alpha - combo) for the current slot for Sum.
        let lut_prod: F::Extension = (poly * lut_degree
            ..min((poly + 1) * lut_degree, num_lut_slots))
            .map(|i| {
                F::Extension::from(deltas[LookupChallenges::ChallengeAlpha as usize])
                    - current_looked_combos[i]
            })
            .product();

        // Compute prod(alpha - combo) for the current slot for LDC.
        let lu_prod: F::Extension = (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots))
            .map(|i| {
                F::Extension::from(deltas[LookupChallenges::ChallengeAlpha as usize])
                    - current_looking_combos[i]
            })
            .product();

        // Function which computes, given index i: prod_{j!=i}(alpha - combo_j) for Sum.
        let lut_prod_i = |i| {
            (poly * lut_degree..min((poly + 1) * lut_degree, num_lut_slots))
                .map(|j| {
                    if j != i {
                        F::Extension::from(deltas[LookupChallenges::ChallengeAlpha as usize])
                            - current_looked_combos[j]
                    } else {
                        F::Extension::ONE
                    }
                })
                .product()
        };

        // Function which computes, given index i: prod_{j!=i}(alpha - combo_j) for LDC.
        let lu_prod_i = |i| {
            (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots))
                .map(|j| {
                    if j != i {
                        F::Extension::from(deltas[LookupChallenges::ChallengeAlpha as usize])
                            - current_looking_combos[j]
                    } else {
                        F::Extension::ONE
                    }
                })
                .product()
        };
        // Compute sum_i(prod_{j!=i}(alpha - combo_j)) for LDC.
        let lu_sum_prods = (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots))
            .fold(F::Extension::ZERO, |acc, i| acc + lu_prod_i(i));

        // Compute sum_i(mul_i.prod_{j!=i}(alpha - combo_j)) for Sum.
        let lut_sum_prods_with_mul = (poly * lut_degree
            ..min((poly + 1) * lut_degree, num_lut_slots))
            .fold(F::Extension::ZERO, |acc, i| {
                acc + vars.local_wires[LookupTableGate::wire_ith_multiplicity(i)] * lut_prod_i(i)
            });

        // The previous element is the previous poly of the current row or the last poly of the next row.
        let prev = if poly == 0 {
            z_gx_lookup_sldcs[num_sldc_polys - 1]
        } else {
            z_x_lookup_sldcs[poly - 1]
        };

        // Check Sum row and col transitions. It's the same constraint, with a row transition happening for slot == 0.
        let unfiltered_sum_transition =
            lut_prod * (z_x_lookup_sldcs[poly] - prev) - lut_sum_prods_with_mul;
        constraints
            .push(lookup_selectors[LookupSelectors::TransSre as usize] * unfiltered_sum_transition);

        // Check LDC row and col transitions. It's the same constraint, with a row transition happening for slot == 0.
        let unfiltered_ldc_transition = lu_prod * (z_x_lookup_sldcs[poly] - prev) + lu_sum_prods;
        constraints
            .push(lookup_selectors[LookupSelectors::TransLdc as usize] * unfiltered_ldc_transition);
    }

    constraints
}

/// Same as `check_lookup_constraints`, but for the base field case.
pub fn check_lookup_constraints_batch<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    vars: EvaluationVarsBase<F>,
    local_lookup_zs: &[F],
    next_lookup_zs: &[F],
    lookup_selectors: &[F],
    deltas: &[F; 4],
    lut_re_poly_evals: &[F],
) -> Vec<F> {
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let lu_degree = common_data.quotient_degree_factor - 1;
    let num_sldc_polys = local_lookup_zs.len() - 1;
    let lut_degree = num_lut_slots.div_ceil(num_sldc_polys);

    let mut constraints = Vec::with_capacity(4 + common_data.luts.len() + 2 * num_sldc_polys);

    // RE is the first polynomial stored.
    let z_re = local_lookup_zs[0];
    let next_z_re = next_lookup_zs[0];

    // Partial Sums and LDCs are both stored in the remaining polynomials.
    let z_x_lookup_sldcs = &local_lookup_zs[1..num_sldc_polys + 1];
    let z_gx_lookup_sldcs = &next_lookup_zs[1..num_sldc_polys + 1];

    // Compute all current looked and looking combos, i.e. the combos we need for the SLDC polynomials.
    let current_looked_combos: Vec<F> = (0..num_lut_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupTableGate::wire_ith_looked_inp(s)];
            let output_wire = vars.local_wires[LookupTableGate::wire_ith_looked_out(s)];
            input_wire + deltas[LookupChallenges::ChallengeA as usize] * output_wire
        })
        .collect();

    let current_looking_combos: Vec<F> = (0..num_lu_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupGate::wire_ith_looking_inp(s)];
            let output_wire = vars.local_wires[LookupGate::wire_ith_looking_out(s)];
            input_wire + deltas[LookupChallenges::ChallengeA as usize] * output_wire
        })
        .collect();

    // Compute all current lookup combos, i.e. the combos used to check that the LUT is correct.
    let current_lookup_combos: Vec<F> = (0..num_lut_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupTableGate::wire_ith_looked_inp(s)];
            let output_wire = vars.local_wires[LookupTableGate::wire_ith_looked_out(s)];
            input_wire + deltas[LookupChallenges::ChallengeB as usize] * output_wire
        })
        .collect();

    // Check last LDC constraint.
    constraints.push(
        lookup_selectors[LookupSelectors::LastLdc as usize] * z_x_lookup_sldcs[num_sldc_polys - 1],
    );

    // Check initial Sum constraint.
    constraints.push(lookup_selectors[LookupSelectors::InitSre as usize] * z_x_lookup_sldcs[0]);

    // Check initial RE constraint.
    constraints.push(lookup_selectors[LookupSelectors::InitSre as usize] * z_re);

    // Check final RE constraints for each different LUT.
    for r in LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors {
        let cur_ends_selector = lookup_selectors[r];

        // Use the precomputed value for the lut poly evaluation
        let re_poly_eval = lut_re_poly_evals[r - LookupSelectors::StartEnd as usize];

        constraints.push(cur_ends_selector * (z_re - re_poly_eval))
    }

    // Check RE row transition constraint.
    let mut cur_sum = next_z_re;
    for elt in &current_lookup_combos {
        cur_sum = cur_sum * deltas[LookupChallenges::ChallengeDelta as usize] + *elt;
    }
    let unfiltered_re_line = z_re - cur_sum;

    constraints.push(lookup_selectors[LookupSelectors::TransSre as usize] * unfiltered_re_line);

    for poly in 0..num_sldc_polys {
        // Compute prod(alpha - combo) for the current slot for Sum.
        let lut_prod: F = (poly * lut_degree..min((poly + 1) * lut_degree, num_lut_slots))
            .map(|i| deltas[LookupChallenges::ChallengeAlpha as usize] - current_looked_combos[i])
            .product();

        // Compute prod(alpha - combo) for the current slot for LDC.
        let lu_prod: F = (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots))
            .map(|i| deltas[LookupChallenges::ChallengeAlpha as usize] - current_looking_combos[i])
            .product();

        // Function which computes, given index i: prod_{j!=i}(alpha - combo_j) for Sum.
        let lut_prod_i = |i| {
            (poly * lut_degree..min((poly + 1) * lut_degree, num_lut_slots))
                .map(|j| {
                    if j != i {
                        deltas[LookupChallenges::ChallengeAlpha as usize] - current_looked_combos[j]
                    } else {
                        F::ONE
                    }
                })
                .product()
        };

        // Function which computes, given index i: prod_{j!=i}(alpha - combo_j) for LDC.
        let lu_prod_i = |i| {
            (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots))
                .map(|j| {
                    if j != i {
                        deltas[LookupChallenges::ChallengeAlpha as usize]
                            - current_looking_combos[j]
                    } else {
                        F::ONE
                    }
                })
                .product()
        };

        // Compute sum_i(prod_{j!=i}(alpha - combo_j)) for LDC.
        let lu_sum_prods = (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots))
            .fold(F::ZERO, |acc, i| acc + lu_prod_i(i));

        // Compute sum_i(mul_i.prod_{j!=i}(alpha - combo_j)) for Sum.
        let lut_sum_prods_with_mul = (poly * lut_degree
            ..min((poly + 1) * lut_degree, num_lut_slots))
            .fold(F::ZERO, |acc, i| {
                acc + vars.local_wires[LookupTableGate::wire_ith_multiplicity(i)] * lut_prod_i(i)
            });

        // The previous element is the previous poly of the current row or the last poly of the next row.
        let prev = if poly == 0 {
            z_gx_lookup_sldcs[num_sldc_polys - 1]
        } else {
            z_x_lookup_sldcs[poly - 1]
        };

        // Check Sum row and col transitions. It's the same constraint, with a row transition happening for slot == 0.
        let unfiltered_sum_transition =
            lut_prod * (z_x_lookup_sldcs[poly] - prev) - lut_sum_prods_with_mul;
        constraints
            .push(lookup_selectors[LookupSelectors::TransSre as usize] * unfiltered_sum_transition);

        // Check LDC row and col transitions. It's the same constraint, with a row transition happening for slot == 0.
        let unfiltered_ldc_transition = lu_prod * (z_x_lookup_sldcs[poly] - prev) + lu_sum_prods;
        constraints
            .push(lookup_selectors[LookupSelectors::TransLdc as usize] * unfiltered_ldc_transition);
    }
    constraints
}

/// Evaluates all gate constraints.
///
/// `num_gate_constraints` is the largest number of constraints imposed by any gate. It is not
/// strictly necessary, but it helps performance by ensuring that we allocate a vector with exactly
/// the capacity that we need.
pub fn evaluate_gate_constraints<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    vars: EvaluationVars<F, D>,
) -> Vec<F::Extension> {
    let mut constraints = vec![F::Extension::ZERO; common_data.num_gate_constraints];
    for (i, gate) in common_data.gates.iter().enumerate() {
        let selector_index = common_data.selectors_info.selector_indices[i];
        let gate_constraints = gate.0.eval_filtered(
            vars,
            i,
            selector_index,
            common_data.selectors_info.groups[selector_index].clone(),
            common_data.selectors_info.num_selectors(),
            common_data.num_lookup_selectors,
        );
        for (i, c) in gate_constraints.into_iter().enumerate() {
            debug_assert!(
                i < common_data.num_gate_constraints,
                "num_constraints() gave too low of a number"
            );
            constraints[i] += c;
        }
    }
    constraints
}

pub(crate) fn evaluate_gate_constraints_base_batch_into_cpu_gates<
    F: RichField + Extendable<D>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    vars_batch: EvaluationVarsBaseBatch<F>,
    constraints_batch: &mut Vec<F>,
    cpu_gate_indices: &[usize],
    filters: &mut Vec<F>,
    shared_gate_filters: &mut Vec<F>,
    selector_filter_suffix: &mut Vec<F>,
    interleave_summed_filter: &mut Vec<F>,
    shared_gate_filter_plan: &mut Vec<bool>,
    num_constraint_rows: usize,
    interleave_pair: Option<&InterleavePairPlan>,
) {
    debug_assert!(num_constraint_rows <= common_data.num_gate_constraints);
    // The gates below accumulate, so this buffer must start at zero — but it
    // does not need re-zeroing here. `reduce_gate_constraints_base_batch` is
    // the sole consumer, it runs immediately after this function on every
    // batch, it reads every element exactly once, and it is called with
    // `clear_as_consumed`, so it hands the buffer back all-zero. Only a change
    // of length (a different circuit shape reaching this worker's scratch)
    // needs the full fill. That removes a `num_constraint_rows * batch`
    // element memset from every batch — 136 * 32 elements on the block shape,
    // ~16k batches per proof — and pays for it with stores to lines the
    // reduction has already pulled in and dirtied.
    let required = num_constraint_rows * vars_batch.len();
    if constraints_batch.len() != required {
        constraints_batch.clear();
        constraints_batch.resize(required, F::ZERO);
    }
    debug_assert!(
        constraints_batch.iter().all(|v| *v == F::ZERO),
        "constraint scratch must be zero on entry; the consumer clears it as it reads"
    );
    if filters.len() != vars_batch.len() {
        filters.resize(vars_batch.len(), F::ZERO);
    }
    if shared_gate_filter_plan.len() != common_data.gates.len() {
        prepare_shared_gate_filter_plan(common_data, cpu_gate_indices, shared_gate_filter_plan);
    }
    fill_shared_gate_filters(
        common_data,
        vars_batch,
        shared_gate_filter_plan,
        shared_gate_filters,
        selector_filter_suffix,
    );
    for &i in cpu_gate_indices {
        if let Some(plan) = interleave_pair {
            if i == plan.interleave_index {
                if shared_gate_filter_plan[plan.interleave_index]
                    && shared_gate_filter_plan[plan.uninterleave_index]
                {
                    let batch_size = vars_batch.len();
                    let interleave_filter = &shared_gate_filters[plan.interleave_index * batch_size
                        ..(plan.interleave_index + 1) * batch_size];
                    let uninterleave_filter = &shared_gate_filters[plan.uninterleave_index
                        * batch_size
                        ..(plan.uninterleave_index + 1) * batch_size];
                    if interleave_summed_filter.len() != batch_size {
                        interleave_summed_filter.resize(batch_size, F::ZERO);
                    }
                    for point in 0..batch_size {
                        interleave_summed_filter[point] =
                            interleave_filter[point] + uninterleave_filter[point];
                    }
                    eval_interleave_pair_dense_fused(
                        vars_batch.local_wires,
                        batch_size,
                        interleave_filter,
                        uninterleave_filter,
                        interleave_summed_filter,
                        constraints_batch,
                    );
                    continue;
                }
                // Size the buffer without re-zeroing it. `clear()` then
                // `resize()` memset all three sub-slices on every batch, but
                // each is fully assigned before it is read: the two
                // `fill_interleave_gate_filter` calls below write every point of
                // `interleave_filter` and `uninterleave_filter`, and the loop
                // writes every point of `summed_filter`. `filters` is scratch
                // reused across batches, so after the first batch of a worker
                // thread this resize is a no-op and the memset is gone
                // entirely — 3 * batch * 8 B per batch, ~12 MiB per d16 tx
                // proof. Value-exact: no slot's read can observe the difference.
                if filters.len() != 3 * vars_batch.len() {
                    filters.resize(3 * vars_batch.len(), F::ZERO);
                }
                let (interleave_filter, rest) = filters.split_at_mut(vars_batch.len());
                let (uninterleave_filter, summed_filter) = rest.split_at_mut(vars_batch.len());
                fill_interleave_gate_filter(
                    common_data,
                    vars_batch,
                    plan.interleave_index,
                    interleave_filter,
                );
                fill_interleave_gate_filter(
                    common_data,
                    vars_batch,
                    plan.uninterleave_index,
                    uninterleave_filter,
                );
                for point in 0..vars_batch.len() {
                    summed_filter[point] = interleave_filter[point] + uninterleave_filter[point];
                }
                eval_interleave_pair_dense_fused(
                    vars_batch.local_wires,
                    vars_batch.len(),
                    interleave_filter,
                    uninterleave_filter,
                    summed_filter,
                    constraints_batch,
                );
                continue;
            }
            if i == plan.uninterleave_index {
                continue;
            }
        }
        let gate = &common_data.gates[i];
        if shared_gate_filter_plan[i] {
            let batch_size = vars_batch.len();
            let filter = &shared_gate_filters[i * batch_size..(i + 1) * batch_size];
            let mut unfiltered_vars = vars_batch;
            unfiltered_vars.remove_prefix(
                common_data.selectors_info.num_selectors() + common_data.num_lookup_selectors,
            );
            gate.0.eval_unfiltered_base_batch_accumulate(
                unfiltered_vars,
                filter,
                constraints_batch,
            );
        } else {
            let selector_index = common_data.selectors_info.selector_indices[i];
            gate.0.eval_filtered_base_batch(
                vars_batch,
                i,
                selector_index,
                common_data.selectors_info.groups[selector_index].clone(),
                common_data.selectors_info.num_selectors(),
                common_data.num_lookup_selectors,
                filters,
                constraints_batch,
            );
        }
    }
}

pub fn evaluate_gate_constraints_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    common_data: &CommonCircuitData<F, D>,
    vars: EvaluationTargets<D>,
) -> Vec<ExtensionTarget<D>> {
    let mut all_gate_constraints = vec![builder.zero_extension(); common_data.num_gate_constraints];
    for (i, gate) in common_data.gates.iter().enumerate() {
        let selector_index = common_data.selectors_info.selector_indices[i];
        with_context!(
            builder,
            &format!("evaluate {} constraints", gate.0.id()),
            gate.0.eval_filtered_circuit(
                builder,
                vars,
                i,
                selector_index,
                common_data.selectors_info.groups[selector_index].clone(),
                common_data.selectors_info.num_selectors(),
                common_data.num_lookup_selectors,
                &mut all_gate_constraints,
            )
        );
    }
    all_gate_constraints
}

pub(crate) fn get_lut_poly_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    common_data: &CommonCircuitData<F, D>,
    lut_index: usize,
    deltas: &[Target],
    degree: usize,
) -> Target {
    let b = deltas[LookupChallenges::ChallengeB as usize];
    let delta = deltas[LookupChallenges::ChallengeDelta as usize];
    let n = common_data.luts[lut_index].len();
    let nb_slots = LookupTableGate::num_slots(&common_data.config);
    let nb_padded_elts = (nb_slots - n % nb_slots) % nb_slots;
    let (padding_inp, padding_out) = common_data.luts[lut_index][0];
    let mut coeffs: Vec<Target> = common_data.luts[lut_index]
        .iter()
        .map(|(input, output)| {
            let temp = builder.mul_const(F::from_canonical_u16(*output), b);
            builder.add_const(temp, F::from_canonical_u16(*input))
        })
        .collect();

    // Padding with the first element of the LUT.
    for _ in 0..nb_padded_elts {
        let temp = builder.mul_const(F::from_canonical_u16(padding_out), b);
        let temp = builder.add_const(temp, F::from_canonical_u16(padding_inp));
        coeffs.push(temp);
    }
    for _ in (n + nb_padded_elts)..degree {
        coeffs.push(builder.zero());
    }
    coeffs.reverse();
    coeffs
        .iter()
        .rev()
        .fold(builder.constant(F::ZERO), |acc, &c| {
            let temp = builder.mul(acc, delta);
            builder.add(temp, c)
        })
}

/// Evaluate the vanishing polynomial at `x`. In this context, the vanishing polynomial is a random
/// linear combination of gate constraints, plus some other terms relating to the permutation
/// argument. All such terms should vanish on `H`.
///
/// Assumes `x != 1`; if `x` could be 1 then this is unsound. This is fine if `x` is a random
/// variable drawn from a sufficiently large domain.
pub(crate) fn eval_vanishing_poly_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    common_data: &CommonCircuitData<F, D>,
    x: ExtensionTarget<D>,
    x_pow_deg: ExtensionTarget<D>,
    vars: EvaluationTargets<D>,
    local_zs: &[ExtensionTarget<D>],
    next_zs: &[ExtensionTarget<D>],
    local_lookup_zs: &[ExtensionTarget<D>],
    next_lookup_zs: &[ExtensionTarget<D>],
    partial_products: &[ExtensionTarget<D>],
    s_sigmas: &[ExtensionTarget<D>],
    betas: &[Target],
    gammas: &[Target],
    alphas: &[Target],
    deltas: &[Target],
) -> Vec<ExtensionTarget<D>> {
    let has_lookup = common_data.num_lookup_polys != 0;
    let max_degree = common_data.quotient_degree_factor;
    let num_prods = common_data.num_partial_products;

    let constraint_terms = with_context!(
        builder,
        "evaluate gate constraints",
        evaluate_gate_constraints_circuit::<F, D>(builder, common_data, vars,)
    );

    let lookup_selectors = &vars.local_constants[common_data.selectors_info.num_selectors()
        ..common_data.selectors_info.num_selectors() + common_data.num_lookup_selectors];

    // The L_0(x) (Z(x) - 1) vanishing terms.
    let mut vanishing_z_1_terms = Vec::new();

    // The terms checking lookup constraints.
    let mut vanishing_all_lookup_terms = if has_lookup {
        let num_sldc_polys = common_data.num_lookup_polys - 1;
        Vec::with_capacity(
            common_data.config.num_challenges * (4 + common_data.luts.len() + 2 * num_sldc_polys),
        )
    } else {
        Vec::new()
    };

    // The terms checking the partial products.
    let mut vanishing_partial_products_terms = Vec::new();

    let l_0_x = eval_l_0_circuit(builder, common_data.degree(), x, x_pow_deg);

    // Holds `k[i] * x`.
    let mut s_ids = Vec::with_capacity(common_data.config.num_routed_wires);
    for j in 0..common_data.config.num_routed_wires {
        let k = builder.constant(common_data.k_is[j]);
        s_ids.push(builder.scalar_mul_ext(k, x));
    }

    for i in 0..common_data.config.num_challenges {
        let z_x = local_zs[i];
        let z_gx = next_zs[i];

        // L_0(x) (Z(x) - 1) = 0.
        vanishing_z_1_terms.push(builder.mul_sub_extension(l_0_x, z_x, l_0_x));

        // If there are lookups in the circuit, then we add the lookup constraints
        if has_lookup {
            let cur_local_lookup_zs = &local_lookup_zs
                [common_data.num_lookup_polys * i..common_data.num_lookup_polys * (i + 1)];
            let cur_next_lookup_zs = &next_lookup_zs
                [common_data.num_lookup_polys * i..common_data.num_lookup_polys * (i + 1)];

            let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];

            let lookup_constraints = check_lookup_constraints_circuit(
                builder,
                common_data,
                vars,
                cur_local_lookup_zs,
                cur_next_lookup_zs,
                lookup_selectors,
                cur_deltas,
            );
            vanishing_all_lookup_terms.extend(lookup_constraints);
        }

        let mut numerator_values = Vec::with_capacity(common_data.config.num_routed_wires);
        let mut denominator_values = Vec::with_capacity(common_data.config.num_routed_wires);

        for j in 0..common_data.config.num_routed_wires {
            let wire_value = vars.local_wires[j];
            let beta_ext = builder.convert_to_ext(betas[i]);
            let gamma_ext = builder.convert_to_ext(gammas[i]);

            // The numerator is `beta * s_id + wire_value + gamma`, and the denominator is
            // `beta * s_sigma + wire_value + gamma`.
            let wire_value_plus_gamma = builder.add_extension(wire_value, gamma_ext);
            let numerator = builder.mul_add_extension(beta_ext, s_ids[j], wire_value_plus_gamma);
            let denominator =
                builder.mul_add_extension(beta_ext, s_sigmas[j], wire_value_plus_gamma);
            numerator_values.push(numerator);
            denominator_values.push(denominator);
        }

        // The partial products considered for this iteration of `i`.
        let current_partial_products = &partial_products[i * num_prods..(i + 1) * num_prods];
        // Check the quotient partial products.
        let partial_product_checks = check_partial_products_circuit(
            builder,
            &numerator_values,
            &denominator_values,
            current_partial_products,
            z_x,
            z_gx,
            max_degree,
        );
        vanishing_partial_products_terms.extend(partial_product_checks);
    }

    let vanishing_terms = [
        vanishing_z_1_terms,
        vanishing_partial_products_terms,
        vanishing_all_lookup_terms,
        constraint_terms,
    ]
    .concat();

    alphas
        .iter()
        .map(|&alpha| {
            let alpha = builder.convert_to_ext(alpha);
            let mut alpha = ReducingFactorTarget::new(alpha);
            alpha.reduce(&vanishing_terms, builder)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::PrimeField64;

    use super::*;

    #[test]
    fn shared_selector_filters_match_gate_local_products() {
        type F = GoldilocksField;

        for batch_size in [1usize, 7, 32] {
            let selector_col = (0..batch_size)
                .map(|point| {
                    // Include exact gate labels, the unused sentinel and
                    // ordinary off-domain values so zero and nonzero factors
                    // are both covered.
                    F::from_canonical_usize(match point % 5 {
                        0 => 2,
                        1 => 5,
                        2 => UNUSED_SELECTOR,
                        _ => 17 * point + 11,
                    })
                })
                .collect::<Vec<_>>();
            let group = 2..8;
            let gate_count = 10;

            for include_unused_selector in [false, true] {
                for active_pattern in [0usize, 1] {
                    let active = (0..gate_count)
                        .map(|gate| group.contains(&gate) && (gate + active_pattern) % 2 == 0)
                        .collect::<Vec<_>>();
                    let mut actual = vec![F::ZERO; gate_count * batch_size];
                    let mut suffix = vec![F::ZERO; batch_size];
                    fill_shared_selector_group_filters(
                        &selector_col,
                        group.clone(),
                        include_unused_selector,
                        &active,
                        &mut actual,
                        &mut suffix,
                    );

                    for gate in group.clone().filter(|&gate| active[gate]) {
                        for (point, &selector) in selector_col.iter().enumerate() {
                            let mut factors = group
                                .clone()
                                .filter(|&other| other != gate)
                                .chain(include_unused_selector.then_some(UNUSED_SELECTOR));
                            let expected = match factors.next() {
                                Some(first) => {
                                    let mut value = F::from_canonical_usize(first) - selector;
                                    for factor in factors {
                                        value *= F::from_canonical_usize(factor) - selector;
                                    }
                                    value
                                }
                                None => F::ONE,
                            };
                            let actual = actual[gate * batch_size + point];
                            assert_eq!(
                                actual.to_canonical_u64(),
                                expected.to_canonical_u64(),
                                "gate {gate}, point {point}, unused={include_unused_selector}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shared_selector_filter_plan_keeps_small_groups_local() {
        assert!(!shared_selector_group_is_cheaper(1, 1, false));
        assert!(!shared_selector_group_is_cheaper(2, 2, true));
        assert!(!shared_selector_group_is_cheaper(3, 3, false));
        assert!(shared_selector_group_is_cheaper(3, 3, true));
        assert!(shared_selector_group_is_cheaper(4, 4, false));
        assert!(shared_selector_group_is_cheaper(8, 6, true));
        assert!(!shared_selector_group_is_cheaper(8, 1, true));
    }

    #[test]
    fn constraint_major_reduction_preserves_pointwise_horner_order() {
        type F = GoldilocksField;

        let alphas = [F::from_canonical_u64(3), F::from_canonical_u64(5)];

        // Hand-checked two-point fixture. For the first challenge at point zero,
        // the expected Horner chain is 7 + 3 * (13 + 3 * 5) = 91.
        let mut terms = [
            F::from_canonical_u64(7),
            F::from_canonical_u64(11),
            F::from_canonical_u64(13),
            F::from_canonical_u64(17),
        ];
        let mut actual = [
            F::from_canonical_u64(5),
            F::from_canonical_u64(5),
            F::from_canonical_u64(6),
            F::from_canonical_u64(6),
        ];
        reduce_gate_constraints_base_batch(&mut terms, 2, &alphas, &mut actual, false, false);
        assert_eq!(actual[0], F::from_canonical_u64(91));
        assert_eq!(actual[2], F::from_canonical_u64(116));

        for batch_size in [1, 11, 31, 32] {
            let num_constraints = 7;
            let mut terms = (0..batch_size * num_constraints)
                .map(|i| F::from_canonical_usize(i * 17 + 3))
                .collect::<Vec<_>>();
            let initial = (0..batch_size * alphas.len())
                .map(|i| F::from_canonical_usize(i * 19 + 7))
                .collect::<Vec<_>>();
            let mut expected = initial.clone();
            for point in 0..batch_size {
                let point_result = &mut expected[point * alphas.len()..(point + 1) * alphas.len()];
                for constraint_row in terms.chunks_exact(batch_size).rev() {
                    let term = constraint_row[point];
                    for (result, &alpha) in point_result.iter_mut().zip(&alphas) {
                        *result = term.multiply_accumulate(*result, alpha);
                    }
                }
            }

            let mut actual = initial;
            reduce_gate_constraints_base_batch(
                &mut terms,
                batch_size,
                &alphas,
                &mut actual,
                false,
                false,
            );
            assert_eq!(actual, expected, "batch size {batch_size}");
        }
    }

    /// The pre-transformation fused evaluator, kept compiled as the
    /// differential's reference: one modular reduction per bit in each of the
    /// three Horner chains. Only the scratch allocation differs (heap instead
    /// of the conditional stack buffer), which cannot change any value.
    fn eval_interleave_pair_dense_fused_termwise_reference<F: Field>(
        wires: &[F],
        batch_size: usize,
        interleave_filter: &[F],
        uninterleave_filter: &[F],
        summed_filter: &[F],
        combined: &mut [F],
    ) {
        let mut x_accumulators = vec![F::ZERO; INTERLEAVE_OPS * batch_size];
        let mut parity_accumulators = vec![F::ZERO; 4 * batch_size];
        let mut base4_accumulator = vec![F::ZERO; batch_size];
        let mut range = vec![F::ZERO; batch_size];

        let base2 = F::from_canonical_usize(2);
        let base4 = F::from_canonical_usize(4);

        for operation in 0..INTERLEAVE_OPS {
            let interleave_row = operation * 34;
            let bit_offset = operation * 32;
            let x = &mut x_accumulators[operation * batch_size..(operation + 1) * batch_size];
            x.fill(F::ZERO);
            base4_accumulator.fill(F::ZERO);

            for bit_index in 0..32 {
                let bit_number = bit_offset + bit_index;
                let bit_col = &wires[(8 + bit_number) * batch_size..][..batch_size];
                let parity_index = 2 * (operation / 2) + bit_index % 2;
                let parity = &mut parity_accumulators
                    [parity_index * batch_size..(parity_index + 1) * batch_size];
                for point in 0..batch_size {
                    let bit = bit_col[point];
                    x[point] = x[point] * base2 + bit;
                    base4_accumulator[point] = base4_accumulator[point] * base4 + bit;
                    parity[point] = parity[point] * base2 + bit;
                    range[point] = bit * (bit - F::ONE);
                }

                let row = interleave_row + 2 + bit_index;
                let interleave_output = &mut combined[row * batch_size..(row + 1) * batch_size];
                if operation % 2 == 0 {
                    batch_multiply_add_inplace(interleave_output, &range, interleave_filter);
                    let uninterleave_row = row + 2;
                    let uninterleave_output = &mut combined
                        [uninterleave_row * batch_size..(uninterleave_row + 1) * batch_size];
                    batch_multiply_add_inplace(uninterleave_output, &range, uninterleave_filter);
                } else {
                    batch_multiply_add_inplace(interleave_output, &range, summed_filter);
                }
            }

            let x_col = &wires[(2 * operation) * batch_size..][..batch_size];
            for point in 0..batch_size {
                range[point] = x[point] - x_col[point];
            }
            let output =
                &mut combined[interleave_row * batch_size..(interleave_row + 1) * batch_size];
            batch_multiply_add_inplace(output, &range, interleave_filter);

            let spread_col = &wires[(2 * operation + 1) * batch_size..][..batch_size];
            for point in 0..batch_size {
                range[point] = base4_accumulator[point] - spread_col[point];
            }
            let output =
                &mut combined[(interleave_row + 1) * batch_size..(interleave_row + 2) * batch_size];
            batch_multiply_add_inplace(output, &range, interleave_filter);
        }

        let split = F::from_canonical_u64(1 << 32u64);
        let u32_max = F::from_canonical_u32(u32::MAX);
        for operation in 0..UNINTERLEAVE_OPS {
            let row = operation * 68;
            let high =
                &x_accumulators[(2 * operation) * batch_size..(2 * operation + 1) * batch_size];
            let low =
                &x_accumulators[(2 * operation + 1) * batch_size..(2 * operation + 2) * batch_size];
            let evens = &parity_accumulators
                [(2 * operation) * batch_size..(2 * operation + 1) * batch_size];
            let odds = &parity_accumulators
                [(2 * operation + 1) * batch_size..(2 * operation + 2) * batch_size];
            let interleaved = &wires[(4 * operation) * batch_size..][..batch_size];
            let x_evens = &wires[(4 * operation + 1) * batch_size..][..batch_size];
            let x_odds = &wires[(4 * operation + 2) * batch_size..][..batch_size];
            let inverse = &wires[(4 * operation + 3) * batch_size..][..batch_size];

            for point in 0..batch_size {
                range[point] = (inverse[point] * (u32_max - high[point]) - F::ONE) * low[point];
            }
            let output = &mut combined[row * batch_size..(row + 1) * batch_size];
            batch_multiply_add_inplace(output, &range, uninterleave_filter);

            for point in 0..batch_size {
                range[point] = high[point] * split + low[point] - interleaved[point];
            }
            let output = &mut combined[(row + 1) * batch_size..(row + 2) * batch_size];
            batch_multiply_add_inplace(output, &range, uninterleave_filter);

            for point in 0..batch_size {
                range[point] = evens[point] - x_evens[point];
            }
            let output = &mut combined[(row + 2) * batch_size..(row + 3) * batch_size];
            batch_multiply_add_inplace(output, &range, uninterleave_filter);

            for point in 0..batch_size {
                range[point] = odds[point] - x_odds[point];
            }
            let output = &mut combined[(row + 3) * batch_size..(row + 4) * batch_size];
            batch_multiply_add_inplace(output, &range, uninterleave_filter);
        }
    }

    /// Raw-limb differential between the delayed and the per-bit reduction in
    /// the fused interleave pair, over arbitrary `u64` representatives:
    /// canonical, non-canonical and boundary. Bit wires are deliberately NOT
    /// restricted to `{0, 1}`; quotient-domain bit-wire evaluations are not,
    /// so every bound here comes from `raw < 2^64` alone.
    #[test]
    fn delayed_interleave_pair_accumulators_match_termwise_reduction_in_raw_limbs() {
        use plonky2_field::types::Field64;

        type F = GoldilocksField;

        let boundary: [u64; 15] = [
            0,
            1,
            2,
            3,
            (u32::MAX as u64) - 1,
            u32::MAX as u64,
            1u64 << 32,
            F::ORDER - 2,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            F::ORDER + 2,
            1u64 << 63,
            u64::MAX - 1,
            u64::MAX,
        ];

        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // 32 is the production batch width of the quotient walker; 1 and 7
        // exercise the tail, 33 forces the heap scratch path.
        for batch_size in [32usize, 1, 7, 33] {
            for trial in 0..16usize {
                let mut draw = |k: usize| -> F {
                    if trial == 0 {
                        GoldilocksField(boundary[k % boundary.len()])
                    } else {
                        let r = next();
                        if r % 3 == 0 {
                            GoldilocksField(boundary[(r >> 2) as usize % boundary.len()])
                        } else {
                            GoldilocksField(r)
                        }
                    }
                };

                let wires: Vec<F> = (0..INTERLEAVE_PAIR_WIRES * batch_size)
                    .map(&mut draw)
                    .collect();
                let interleave_filter: Vec<F> = (0..batch_size).map(&mut draw).collect();
                let uninterleave_filter: Vec<F> = (0..batch_size).map(&mut draw).collect();
                let summed_filter: Vec<F> = interleave_filter
                    .iter()
                    .zip(&uninterleave_filter)
                    .map(|(&interleave, &uninterleave)| interleave + uninterleave)
                    .collect();
                // A nonzero seed proves the evaluator accumulates onto prior
                // rows rather than assigning them.
                let seed: Vec<F> = (0..INTERLEAVE_PAIR_CONSTRAINTS * batch_size)
                    .map(&mut draw)
                    .collect();

                let mut expected = seed.clone();
                eval_interleave_pair_dense_fused_termwise_reference(
                    &wires,
                    batch_size,
                    &interleave_filter,
                    &uninterleave_filter,
                    &summed_filter,
                    &mut expected,
                );

                let mut actual = seed;
                eval_interleave_pair_dense_fused(
                    &wires,
                    batch_size,
                    &interleave_filter,
                    &uninterleave_filter,
                    &summed_filter,
                    &mut actual,
                );

                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_noncanonical_u64(),
                        e.to_noncanonical_u64(),
                        "raw-limb mismatch at slot {index} (batch {batch_size}, trial {trial})"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_dense_interleave_pair_matches_independent_alpha_reduction_canonically() {
        type F = GoldilocksField;

        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Deliberately retain arbitrary raw representatives. The fused
            // same-row filter changes legal field association, so equality is
            // required canonically rather than at the raw-limb level.
            GoldilocksField(state)
        };

        for batch_size in [1usize, 7, 32] {
            let wires = (0..INTERLEAVE_PAIR_WIRES * batch_size)
                .map(|_| next())
                .collect::<Vec<_>>();
            let interleave_filter = (0..batch_size).map(|_| next()).collect::<Vec<_>>();
            let uninterleave_filter = (0..batch_size).map(|_| next()).collect::<Vec<_>>();
            let summed_filter = interleave_filter
                .iter()
                .zip(&uninterleave_filter)
                .map(|(&interleave, &uninterleave)| interleave + uninterleave)
                .collect::<Vec<_>>();

            // A nonzero shared seed ensures the specialization really adds to
            // prior gate rows rather than accidentally assigning them.
            let initial_rows = (0..INTERLEAVE_PAIR_CONSTRAINTS * batch_size)
                .map(|_| next())
                .collect::<Vec<_>>();
            let mut expected_rows = initial_rows.clone();
            let base2 = F::from_canonical_usize(2);
            let base4 = F::from_canonical_usize(4);
            let split = F::from_canonical_u64(1 << 32u64);
            let u32_max = F::from_canonical_u32(u32::MAX);

            // Independent dense oracle: evaluate the two gates separately in
            // their original row layouts and apply their distinct filters.
            for point in 0..batch_size {
                let mut halves = [F::ZERO; INTERLEAVE_OPS];
                let mut parities = [F::ZERO; 4];
                for operation in 0..INTERLEAVE_OPS {
                    let row = operation * 34;
                    let mut x = F::ZERO;
                    let mut spread = F::ZERO;
                    for bit_index in 0..32 {
                        let bit_number = operation * 32 + bit_index;
                        let bit = wires[(8 + bit_number) * batch_size + point];
                        x = x * base2 + bit;
                        spread = spread * base4 + bit;
                        let parity_index = 2 * (operation / 2) + bit_index % 2;
                        parities[parity_index] = parities[parity_index] * base2 + bit;
                        let range = bit * (bit - F::ONE);
                        expected_rows[(row + 2 + bit_index) * batch_size + point] +=
                            interleave_filter[point] * range;
                        let uninterleave_row =
                            (operation / 2) * 68 + 4 + (operation % 2) * 32 + bit_index;
                        expected_rows[uninterleave_row * batch_size + point] +=
                            uninterleave_filter[point] * range;
                    }
                    halves[operation] = x;
                    expected_rows[row * batch_size + point] += interleave_filter[point]
                        * (x - wires[(2 * operation) * batch_size + point]);
                    expected_rows[(row + 1) * batch_size + point] += interleave_filter[point]
                        * (spread - wires[(2 * operation + 1) * batch_size + point]);
                }

                for operation in 0..UNINTERLEAVE_OPS {
                    let row = operation * 68;
                    let high = halves[2 * operation];
                    let low = halves[2 * operation + 1];
                    let constraints = [
                        (wires[(4 * operation + 3) * batch_size + point] * (u32_max - high)
                            - F::ONE)
                            * low,
                        high * split + low - wires[(4 * operation) * batch_size + point],
                        parities[2 * operation] - wires[(4 * operation + 1) * batch_size + point],
                        parities[2 * operation + 1]
                            - wires[(4 * operation + 2) * batch_size + point],
                    ];
                    for (offset, constraint) in constraints.into_iter().enumerate() {
                        expected_rows[(row + offset) * batch_size + point] +=
                            uninterleave_filter[point] * constraint;
                    }
                }
            }

            let mut actual_rows = initial_rows;
            eval_interleave_pair_dense_fused(
                &wires,
                batch_size,
                &interleave_filter,
                &uninterleave_filter,
                &summed_filter,
                &mut actual_rows,
            );

            for (index, (&actual, &expected)) in actual_rows.iter().zip(&expected_rows).enumerate()
            {
                assert_eq!(
                    actual.to_canonical_u64(),
                    expected.to_canonical_u64(),
                    "dense canonical mismatch at {index}, batch={batch_size}"
                );
            }

            let alphas = [next(), next()];
            let initial_reduction = (0..batch_size * alphas.len())
                .map(|_| next())
                .collect::<Vec<_>>();
            let mut expected_reduction = initial_reduction.clone();
            let mut actual_reduction = initial_reduction;
            reduce_gate_constraints_base_batch(
                &mut expected_rows,
                batch_size,
                &alphas,
                &mut expected_reduction,
                false,
                false,
            );
            reduce_gate_constraints_base_batch(
                &mut actual_rows,
                batch_size,
                &alphas,
                &mut actual_reduction,
                false,
                false,
            );
            for (index, (&actual, &expected)) in
                actual_reduction.iter().zip(&expected_reduction).enumerate()
            {
                assert_eq!(
                    actual.to_canonical_u64(),
                    expected.to_canonical_u64(),
                    "alpha reduction mismatch at {index}, batch={batch_size}"
                );
            }
        }
    }

    /// The narrowing sizes the shared row buffer by the widest gate still on
    /// the CPU, so rows above it are never materialized. This asserts that
    /// dropping that dead suffix is RAW-LIMB identical to reducing the
    /// full-width buffer with those rows present and zero, which is the
    /// property the whole mechanism rests on. Raw limbs, not `PartialEq`:
    /// Goldilocks does not canonicalize, so field equality would hide a
    /// representation change that alters proof bytes.
    #[test]
    fn narrowed_row_space_matches_full_width_reduction_in_raw_limbs() {
        type F = GoldilocksField;

        let alphas = [F::from_canonical_u64(7), F::from_canonical_u64(11)];

        // Deliberately arbitrary, large witness values: a pass must not be an
        // artifact of terms that happen to be zero or small.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // `>> 1` keeps every value below the Goldilocks order without
            // needing `Field::ORDER` in scope here.
            F::from_canonical_u64(state >> 1)
        };

        for batch_size in [1usize, 7, 32] {
            for k in [1usize, 88, 136] {
                for m in [0usize, 1, k / 2, k] {
                    if m > k {
                        continue;
                    }
                    // Live prefix `[0, m)`; the suffix `[m, k)` stays zero,
                    // exactly as an offloaded gate leaves it.
                    let mut full = vec![F::ZERO; k * batch_size];
                    for row in 0..m {
                        for point in 0..batch_size {
                            full[row * batch_size + point] = next();
                        }
                    }
                    let mut narrowed = full[..m * batch_size].to_vec();

                    let mut expected = vec![F::ZERO; batch_size * alphas.len()];
                    reduce_gate_constraints_base_batch(
                        &mut full,
                        batch_size,
                        &alphas,
                        &mut expected,
                        true,
                        false,
                    );
                    let mut actual = vec![F::ZERO; batch_size * alphas.len()];
                    reduce_gate_constraints_base_batch(
                        &mut narrowed,
                        batch_size,
                        &alphas,
                        &mut actual,
                        true,
                        false,
                    );

                    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.0, e.0,
                            "raw limb mismatch at {i} (k={k}, m={m}, batch={batch_size})"
                        );
                    }
                }
            }
        }
    }
}

/// Same as `check_lookup_constraints`, but for the recursive case.
pub fn check_lookup_constraints_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    common_data: &CommonCircuitData<F, D>,
    vars: EvaluationTargets<D>,
    local_lookup_zs: &[ExtensionTarget<D>],
    next_lookup_zs: &[ExtensionTarget<D>],
    lookup_selectors: &[ExtensionTarget<D>],
    deltas: &[Target],
) -> Vec<ExtensionTarget<D>> {
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let lu_degree = common_data.quotient_degree_factor - 1;
    let num_sldc_polys = local_lookup_zs.len() - 1;
    let lut_degree = num_lut_slots.div_ceil(num_sldc_polys);

    let mut constraints = Vec::with_capacity(4 + common_data.luts.len() + 2 * num_sldc_polys);

    // RE is the first polynomial stored.
    let z_re = local_lookup_zs[0];
    let next_z_re = next_lookup_zs[0];

    // Partial Sums and LDCs (i.e. the SLDC polynomials) are stored in the remaining polynomials.
    let z_x_lookup_sldcs = &local_lookup_zs[1..num_sldc_polys + 1];
    let z_gx_lookup_sldcs = &next_lookup_zs[1..num_sldc_polys + 1];

    // Convert deltas to ExtensionTargets.
    let ext_deltas = deltas
        .iter()
        .map(|d| builder.convert_to_ext(*d))
        .collect::<Vec<_>>();

    // Computing all current looked and looking combos, i.e. the combos we need for the SLDC polynomials.
    let current_looked_combos = (0..num_lut_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupTableGate::wire_ith_looked_inp(s)];
            let output_wire = vars.local_wires[LookupTableGate::wire_ith_looked_out(s)];
            builder.mul_add_extension(
                ext_deltas[LookupChallenges::ChallengeA as usize],
                output_wire,
                input_wire,
            )
        })
        .collect::<Vec<_>>();
    let current_looking_combos = (0..num_lu_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupGate::wire_ith_looking_inp(s)];
            let output_wire = vars.local_wires[LookupGate::wire_ith_looking_out(s)];
            builder.mul_add_extension(
                ext_deltas[LookupChallenges::ChallengeA as usize],
                output_wire,
                input_wire,
            )
        })
        .collect::<Vec<_>>();

    let current_lut_subs = (0..num_lut_slots)
        .map(|s| {
            builder.sub_extension(
                ext_deltas[LookupChallenges::ChallengeAlpha as usize],
                current_looked_combos[s],
            )
        })
        .collect::<Vec<_>>();

    let current_lu_subs = (0..num_lu_slots)
        .map(|s| {
            builder.sub_extension(
                ext_deltas[LookupChallenges::ChallengeAlpha as usize],
                current_looking_combos[s],
            )
        })
        .collect::<Vec<_>>();

    // Computing all current lookup combos, i.e. the combos used to check that the LUT is correct.
    let current_lookup_combos = (0..num_lut_slots)
        .map(|s| {
            let input_wire = vars.local_wires[LookupTableGate::wire_ith_looked_inp(s)];
            let output_wire = vars.local_wires[LookupTableGate::wire_ith_looked_out(s)];
            builder.mul_add_extension(
                ext_deltas[LookupChallenges::ChallengeB as usize],
                output_wire,
                input_wire,
            )
        })
        .collect::<Vec<_>>();

    // Check last LDC constraint.
    constraints.push(builder.mul_extension(
        lookup_selectors[LookupSelectors::LastLdc as usize],
        z_x_lookup_sldcs[num_sldc_polys - 1],
    ));

    // Check initial Sum constraint.
    constraints.push(builder.mul_extension(
        lookup_selectors[LookupSelectors::InitSre as usize],
        z_x_lookup_sldcs[0],
    ));

    // Check initial RE constraint.
    constraints
        .push(builder.mul_extension(lookup_selectors[LookupSelectors::InitSre as usize], z_re));

    // Check final RE constraints for each different LUT.
    for r in LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors {
        let cur_ends_selectors = lookup_selectors[r];
        let lut_row_number = common_data.luts[r - LookupSelectors::StartEnd as usize]
            .len()
            .div_ceil(num_lut_slots);
        let cur_function_eval = get_lut_poly_circuit(
            builder,
            common_data,
            r - LookupSelectors::StartEnd as usize,
            deltas,
            num_lut_slots * lut_row_number,
        );
        let cur_function_eval_ext = builder.convert_to_ext(cur_function_eval);

        let cur_re = builder.sub_extension(z_re, cur_function_eval_ext);
        constraints.push(builder.mul_extension(cur_ends_selectors, cur_re));
    }

    // Check RE row transition constraint.
    let mut cur_sum = next_z_re;
    for elt in &current_lookup_combos {
        cur_sum = builder.mul_add_extension(
            cur_sum,
            ext_deltas[LookupChallenges::ChallengeDelta as usize],
            *elt,
        );
    }
    let unfiltered_re_line = builder.sub_extension(z_re, cur_sum);

    constraints.push(builder.mul_extension(
        lookup_selectors[LookupSelectors::TransSre as usize],
        unfiltered_re_line,
    ));

    for poly in 0..num_sldc_polys {
        // Compute prod(alpha - combo) for the current slot for Sum.
        let mut lut_prod = builder.one_extension();
        for i in poly * lut_degree..min((poly + 1) * lut_degree, num_lut_slots) {
            lut_prod = builder.mul_extension(lut_prod, current_lut_subs[i]);
        }

        // Compute prod(alpha - combo) for the current slot for LDC.
        let mut lu_prod = builder.one_extension();
        for i in poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots) {
            lu_prod = builder.mul_extension(lu_prod, current_lu_subs[i]);
        }

        let one = builder.one_extension();
        let zero = builder.zero_extension();

        // Compute sum_i(prod_{j!=i}(alpha - combo_j)) for LDC.
        let lu_sum_prods =
            (poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots)).fold(zero, |acc, i| {
                let mut prod_i = one;

                for j in poly * lu_degree..min((poly + 1) * lu_degree, num_lu_slots) {
                    if j != i {
                        prod_i = builder.mul_extension(prod_i, current_lu_subs[j]);
                    }
                }
                builder.add_extension(acc, prod_i)
            });

        // Compute sum_i(mul_i.prod_{j!=i}(alpha - combo_j)) for Sum.
        let lut_sum_prods_mul = (poly * lut_degree..min((poly + 1) * lut_degree, num_lut_slots))
            .fold(zero, |acc, i| {
                let mut prod_i = one;

                for j in poly * lut_degree..min((poly + 1) * lut_degree, num_lut_slots) {
                    if j != i {
                        prod_i = builder.mul_extension(prod_i, current_lut_subs[j]);
                    }
                }
                builder.mul_add_extension(
                    prod_i,
                    vars.local_wires[LookupTableGate::wire_ith_multiplicity(i)],
                    acc,
                )
            });

        // The previous element is the previous poly of the current row or the last poly of the next row.
        let prev = if poly == 0 {
            z_gx_lookup_sldcs[num_sldc_polys - 1]
        } else {
            z_x_lookup_sldcs[poly - 1]
        };

        let cur_sub = builder.sub_extension(z_x_lookup_sldcs[poly], prev);

        // Check sum row and col transitions. It's the same constraint, with a row transition happening for slot == 0.
        let unfiltered_sum_transition =
            builder.mul_sub_extension(lut_prod, cur_sub, lut_sum_prods_mul);
        constraints.push(builder.mul_extension(
            lookup_selectors[LookupSelectors::TransSre as usize],
            unfiltered_sum_transition,
        ));

        // Check ldc row and col transitions. It's the same constraint, with a row transition happening for slot == 0.
        let unfiltered_ldc_transition = builder.mul_add_extension(lu_prod, cur_sub, lu_sum_prods);
        constraints.push(builder.mul_extension(
            lookup_selectors[LookupSelectors::TransLdc as usize],
            unfiltered_ldc_transition,
        ));
    }
    constraints
}
