#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::ops::Range;

use hashbrown::HashMap;
use serde::Serialize;

use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialValues;
use crate::gates::gate::{GateInstance, GateRef};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_builder::LookupWire;

/// Placeholder value to indicate that a gate doesn't use a selector polynomial.
pub(crate) const UNUSED_SELECTOR: usize = u32::MAX as usize;

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct SelectorsInfo {
    pub(crate) selector_indices: Vec<usize>,
    pub(crate) groups: Vec<Range<usize>>,
}

impl SelectorsInfo {
    pub fn num_selectors(&self) -> usize {
        self.groups.len()
    }
}

/// Enum listing the different selectors for lookup constraints:
/// - `TransSre` is for Sum and RE transition constraints.
/// - `TransLdc` is for LDC transition constraints.
/// - `InitSre` is for the initial constraint of Sum and Re.
/// - `LastLdc` is for the final LDC (and Sum) constraint.
/// - `StartEnd` indicates where lookup end selectors begin.
pub enum LookupSelectors {
    TransSre = 0,
    TransLdc,
    InitSre,
    LastLdc,
    StartEnd,
}

/// Returns selector polynomials for each LUT. We have two constraint domains (remember that gates are stored upside down):
/// - [last_lut_row, first_lut_row] (Sum and RE transition constraints),
/// - [last_lu_row, last_lut_row - 1] (LDC column transition constraints).
///
/// We also add two more:
/// - {first_lut_row + 1} where we check the initial values of sum and RE (which are 0),
/// - {last_lu_row} where we check that the last value of LDC is 0.
///
/// Conceptually they're part of the selector ends lookups, but since we can have one polynomial for *all* LUTs it's here.
pub(crate) fn selectors_lookup<F: RichField + Extendable<D>, const D: usize>(
    _gates: &[GateRef<F, D>],
    instances: &[GateInstance<F, D>],
    lookup_rows: &[LookupWire],
) -> Vec<PolynomialValues<F>> {
    let n = instances.len();
    let mut lookup_selectors = Vec::with_capacity(LookupSelectors::StartEnd as usize);
    for _ in 0..LookupSelectors::StartEnd as usize {
        lookup_selectors.push(PolynomialValues::<F>::new(vec![F::ZERO; n]));
    }

    for &LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in lookup_rows
    {
        for row in last_lut_row..first_lut_row + 1 {
            lookup_selectors[LookupSelectors::TransSre as usize].values[row] = F::ONE;
        }
        for row in last_lu_row..last_lut_row {
            lookup_selectors[LookupSelectors::TransLdc as usize].values[row] = F::ONE;
        }
        lookup_selectors[LookupSelectors::InitSre as usize].values[first_lut_row + 1] = F::ONE;
        lookup_selectors[LookupSelectors::LastLdc as usize].values[last_lu_row] = F::ONE;
    }
    lookup_selectors
}

/// Returns selectors for checking the validity of the LUTs.
/// Each selector equals one on its respective LUT's `last_lut_row`, and 0 elsewhere.
pub(crate) fn selector_ends_lookups<F: RichField + Extendable<D>, const D: usize>(
    lookup_rows: &[LookupWire],
    instances: &[GateInstance<F, D>],
) -> Vec<PolynomialValues<F>> {
    let n = instances.len();
    let mut lookups_ends = Vec::with_capacity(lookup_rows.len());
    for &LookupWire {
        last_lu_gate: _,
        last_lut_gate: last_lut_row,
        first_lut_gate: _,
    } in lookup_rows
    {
        let mut lookup_ends = PolynomialValues::<F>::new(vec![F::ZERO; n]);
        lookup_ends.values[last_lut_row] = F::ONE;
        lookups_ends.push(lookup_ends);
    }
    lookups_ends
}

/// Returns the selector polynomials and related information.
///
/// Selector polynomials are computed as follows:
/// Partition the gates into (the smallest amount of) groups `{ G_i }`, such that for each group `G`
/// `|G| + max_{g in G} g.degree() <= max_degree`. These groups are constructed greedily from
/// the list of gates sorted by degree.
/// We build a selector polynomial `S_i` for each group `G_i`, with
/// S_i\[j\] =
///     if j-th row gate=g_k in G_i
///         k
///     else
///         UNUSED_SELECTOR
pub(crate) fn selector_polynomials<F: RichField + Extendable<D>, const D: usize>(
    gates: &[GateRef<F, D>],
    instances: &[GateInstance<F, D>],
    max_degree: usize,
) -> (Vec<PolynomialValues<F>>, SelectorsInfo) {
    let n = instances.len();
    let num_gates = gates.len();
    let max_gate_degree = gates.last().expect("No gates?").0.degree();

    // Pre-index the sorted gate list by gate ID so each row instance resolves its
    // sorted position with one `Gate::id()` call plus one expected-constant-time
    // hash lookup, instead of a linear scan that re-generates every candidate
    // gate's ID (`Gate::id()` allocates a fresh `String`, typically via
    // `format!`). `entry().or_insert()` keeps first-position semantics for
    // duplicate IDs, matching the previous `.position()` behavior. The map is
    // used only for lookup, so its iteration order never influences any output.
    let mut gate_index = HashMap::with_capacity(num_gates);
    for (i, g) in gates.iter().enumerate() {
        gate_index.entry(g.0.id()).or_insert(i);
    }
    let index = |id| *gate_index.get(&id).unwrap();

    // Special case if we can use only one selector polynomial.
    if max_gate_degree + num_gates - 1 <= max_degree {
        // We *want* `groups` to be a vector containing one Range (all gates are in one selector group),
        // but Clippy doesn't trust us.
        #[allow(clippy::single_range_in_vec_init)]
        return (
            vec![PolynomialValues::new(
                instances
                    .iter()
                    .map(|g| F::from_canonical_usize(index(g.gate_ref.0.id())))
                    .collect(),
            )],
            SelectorsInfo {
                selector_indices: vec![0; num_gates],
                groups: vec![0..num_gates],
            },
        );
    }

    if max_gate_degree >= max_degree {
        panic!(
            "{} has too high degree. Consider increasing `quotient_degree_factor`.",
            gates.last().unwrap().0.id()
        );
    }

    // Greedily construct the groups.
    let mut groups = Vec::new();
    let mut start = 0;
    while start < num_gates {
        let mut size = 0;
        while (start + size < gates.len()) && (size + gates[start + size].0.degree() < max_degree) {
            size += 1;
        }
        groups.push(start..start + size);
        start += size;
    }

    let group = |i| groups.iter().position(|range| range.contains(&i)).unwrap();

    // `selector_indices[i] = j` iff the `i`-th gate uses the `j`-th selector polynomial.
    let selector_indices: Vec<usize> = (0..num_gates).map(group).collect();

    // Placeholder value to indicate that a gate doesn't use a selector polynomial.
    let unused = F::from_canonical_usize(UNUSED_SELECTOR);

    // Initialize every selector cell to the inactive sentinel, then write only
    // the active cell per row. The final value rule is unchanged:
    //     selector[column][row] = sorted_gate_index  if column is the row
    //                                                gate's selector group;
    //                             UNUSED_SELECTOR    otherwise.
    // The row loop reuses the already-materialized `selector_indices` mapping
    // instead of re-scanning `groups` for every row.
    let mut polynomials = vec![PolynomialValues::constant(unused, n); groups.len()];
    for (j, g) in instances.iter().enumerate() {
        let GateInstance { gate_ref, .. } = g;
        let i = index(gate_ref.0.id());
        polynomials[selector_indices[i]].values[j] = F::from_canonical_usize(i);
    }

    (
        polynomials,
        SelectorsInfo {
            selector_indices,
            groups,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::base_sum::BaseSumGate;
    use crate::gates::constant::ConstantGate;
    use crate::gates::noop::NoopGate;
    use crate::gates::poseidon::PoseidonGate;
    use crate::gates::public_input::PublicInputGate;
    use crate::gates::random_access::RandomAccessGate;
    use crate::plonk::circuit_data::CircuitConfig;

    type F = GoldilocksField;
    const D: usize = 2;

    /// Verbatim copy of the pre-optimization `selector_polynomials`, kept as the
    /// reference implementation for pointwise equivalence.
    fn legacy_selector_polynomials(
        gates: &[GateRef<F, D>],
        instances: &[GateInstance<F, D>],
        max_degree: usize,
    ) -> (Vec<PolynomialValues<F>>, SelectorsInfo) {
        let n = instances.len();
        let num_gates = gates.len();
        let max_gate_degree = gates.last().expect("No gates?").0.degree();

        let index = |id| gates.iter().position(|g| g.0.id() == id).unwrap();

        // Special case if we can use only one selector polynomial.
        if max_gate_degree + num_gates - 1 <= max_degree {
            #[allow(clippy::single_range_in_vec_init)]
            return (
                vec![PolynomialValues::new(
                    instances
                        .iter()
                        .map(|g| F::from_canonical_usize(index(g.gate_ref.0.id())))
                        .collect(),
                )],
                SelectorsInfo {
                    selector_indices: vec![0; num_gates],
                    groups: vec![0..num_gates],
                },
            );
        }

        if max_gate_degree >= max_degree {
            panic!(
                "{} has too high degree. Consider increasing `quotient_degree_factor`.",
                gates.last().unwrap().0.id()
            );
        }

        // Greedily construct the groups.
        let mut groups = Vec::new();
        let mut start = 0;
        while start < num_gates {
            let mut size = 0;
            while (start + size < gates.len())
                && (size + gates[start + size].0.degree() < max_degree)
            {
                size += 1;
            }
            groups.push(start..start + size);
            start += size;
        }

        let group = |i| groups.iter().position(|range| range.contains(&i)).unwrap();

        let selector_indices = (0..num_gates).map(group).collect();

        let unused = F::from_canonical_usize(UNUSED_SELECTOR);

        let mut polynomials = vec![PolynomialValues::zero(n); groups.len()];
        for (j, g) in instances.iter().enumerate() {
            let GateInstance { gate_ref, .. } = g;
            let i = index(gate_ref.0.id());
            let gr = group(i);
            for g in 0..groups.len() {
                polynomials[g].values[j] = if g == gr {
                    F::from_canonical_usize(i)
                } else {
                    unused
                };
            }
        }

        (
            polynomials,
            SelectorsInfo {
                selector_indices,
                groups,
            },
        )
    }

    /// Sorts gates exactly as `CircuitBuilder::build` does before calling
    /// `selector_polynomials`.
    fn sorted_gates(mut gates: Vec<GateRef<F, D>>) -> Vec<GateRef<F, D>> {
        gates.sort_unstable_by_key(|g| (g.0.degree(), g.0.id()));
        gates
    }

    /// Deterministic mixed-order row assignment touching every gate type.
    fn instances_in_mixed_order(gates: &[GateRef<F, D>], n: usize) -> Vec<GateInstance<F, D>> {
        (0..n)
            .map(|row| GateInstance {
                gate_ref: gates[(row * 7 + row / 3) % gates.len()].clone(),
                constants: vec![],
            })
            .collect()
    }

    /// The gate mix used by the production-config tests: distinct IDs across a
    /// spread of degrees (0 through 7), like the challenge circuits' gate sets.
    fn production_style_gates(config: &CircuitConfig) -> Vec<GateRef<F, D>> {
        sorted_gates(vec![
            GateRef::new(NoopGate),
            GateRef::new(PublicInputGate),
            GateRef::new(ConstantGate::new(config.num_constants)),
            GateRef::new(BaseSumGate::<2>::new(8)),
            GateRef::new(ArithmeticGate::new_from_config(config)),
            GateRef::new(RandomAccessGate::new_from_config(config, 4)),
            GateRef::new(PoseidonGate::new()),
        ])
    }

    fn assert_pointwise_equal(
        gates: &[GateRef<F, D>],
        instances: &[GateInstance<F, D>],
        max_degree: usize,
    ) {
        let (legacy_polys, legacy_info) = legacy_selector_polynomials(gates, instances, max_degree);
        let (polys, info) = selector_polynomials(gates, instances, max_degree);
        // `PolynomialValues` equality is exact field-element (u64) equality, so
        // this is a pointwise byte-identity check against the legacy
        // construction.
        assert_eq!(polys, legacy_polys);
        assert_eq!(info, legacy_info);
    }

    #[test]
    fn selector_polynomials_match_legacy_on_production_config() {
        // The builder calls `selector_polynomials(gates, instances,
        // quotient_degree_factor + 1)`; both production configs in the
        // challenge use `max_quotient_degree_factor = 8`.
        let config = CircuitConfig::standard_recursion_config();
        assert_eq!(config.max_quotient_degree_factor, 8);
        let max_degree = config.max_quotient_degree_factor + 1;

        let gates = production_style_gates(&config);
        let instances = instances_in_mixed_order(&gates, 128);
        // Degree-7 Poseidon plus 6 more gates cannot fit one selector group at
        // max_degree 9, so this exercises the multi-selector branch.
        assert!(gates.last().unwrap().0.degree() + gates.len() - 1 > max_degree);
        assert_pointwise_equal(&gates, &instances, max_degree);
    }

    #[test]
    fn selector_polynomials_match_legacy_single_group() {
        let config = CircuitConfig::standard_recursion_config();
        let gates = sorted_gates(vec![
            GateRef::new(NoopGate),
            GateRef::new(PublicInputGate),
            GateRef::new(ArithmeticGate::new_from_config(&config)),
        ]);
        let instances = instances_in_mixed_order(&gates, 64);
        // Max degree 3 with 3 gates fits in one group at max_degree 9.
        assert!(gates.last().unwrap().0.degree() + gates.len() - 1 <= 9);
        assert_pointwise_equal(&gates, &instances, 9);

        let (_, info) = selector_polynomials(&gates, &instances, 9);
        assert_eq!(info.groups, vec![0..3]);
        assert_eq!(info.selector_indices, vec![0, 0, 0]);
    }

    #[test]
    fn selector_polynomials_match_legacy_forced_multi_group() {
        // A tighter degree bound forces more, smaller groups. The gate set
        // tops out at degree 3 (ArithmeticGate) so every swept `max_degree`
        // stays above the largest gate degree, as `selector_polynomials`
        // requires.
        let config = CircuitConfig::standard_recursion_config();
        let gates = sorted_gates(vec![
            GateRef::new(NoopGate),
            GateRef::new(PublicInputGate),
            GateRef::new(ConstantGate::new(config.num_constants)),
            GateRef::new(BaseSumGate::<2>::new(8)),
            GateRef::new(ArithmeticGate::new_from_config(&config)),
        ]);
        let max_gate_degree = gates.last().unwrap().0.degree();
        for max_degree in [4, 5, 6] {
            // Stay in the multi-group branch without tripping the
            // too-high-degree panic.
            assert!(max_gate_degree < max_degree);
            assert!(max_gate_degree + gates.len() - 1 > max_degree);
            let instances = instances_in_mixed_order(&gates, 128);
            assert_pointwise_equal(&gates, &instances, max_degree);
        }
    }

    #[test]
    fn selector_polynomials_inactive_cells_are_unused_sentinel() {
        let config = CircuitConfig::standard_recursion_config();
        let gates = production_style_gates(&config);
        let instances = instances_in_mixed_order(&gates, 128);
        let max_degree = config.max_quotient_degree_factor + 1;
        let (polys, info) = selector_polynomials(&gates, &instances, max_degree);
        assert!(info.groups.len() > 1);

        let index = |id: String| gates.iter().position(|g| g.0.id() == id).unwrap();
        let unused = F::from_canonical_usize(UNUSED_SELECTOR);
        for (j, inst) in instances.iter().enumerate() {
            let i = index(inst.gate_ref.0.id());
            let gr = info.selector_indices[i];
            for (g, poly) in polys.iter().enumerate() {
                if g == gr {
                    assert_eq!(poly.values[j], F::from_canonical_usize(i));
                } else {
                    assert_eq!(poly.values[j], unused);
                }
            }
        }
    }
}
