// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Test-only helper checking a gate's `eval_unfiltered_base_batch` against
//! per-point `eval_unfiltered` across a multi-point batch. The upstream
//! `test_eval_fns` only exercises a batch of one point, which cannot catch
//! column-indexing mistakes in hand-written batched evaluations.

use plonky2::field::batch_util::batch_multiply_add_inplace;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64};
use plonky2::gates::gate::Gate;
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::vars::{EvaluationVars, EvaluationVarsBaseBatch};
use rand::Rng;

#[cfg(test)]
mod vendored_gate_tests {
    use plonky2::gates::poseidon2::Poseidon2Gate;

    use super::*;

    #[test]
    fn poseidon2_base_batch_matches_eval_unfiltered_across_batch() {
        let gate = Poseidon2Gate::<GoldilocksField, 2>::new();
        assert_base_batch_matches_eval_unfiltered(&gate);
    }

    #[test]
    fn random_access_base_batch_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::random_access::RandomAccessGate;
        use plonky2::plonk::circuit_data::CircuitConfig;

        for bits in [1, 2, 4, 6] {
            let gate = RandomAccessGate::<GoldilocksField, 2>::new_from_config(
                &CircuitConfig::standard_recursion_config(),
                bits,
            );
            assert_base_batch_matches_eval_unfiltered(&gate);
        }
    }

    #[test]
    fn reducing_accumulate_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::reducing::ReducingGate;

        // 44 coefficients is the configuration used by the recursion circuits.
        for num_coeffs in [1, 2, 43, 44] {
            let gate = ReducingGate::<2>::new(num_coeffs);
            assert_accumulate_matches_eval_unfiltered(&gate);
        }
    }

    #[test]
    fn reducing_extension_accumulate_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::reducing_extension::ReducingExtensionGate;

        // 33 coefficients is the configuration used by the recursion circuits.
        for num_coeffs in [1, 2, 32, 33] {
            let gate = ReducingExtensionGate::<2>::new(num_coeffs);
            assert_accumulate_matches_eval_unfiltered(&gate);
        }
    }

    #[test]
    fn arithmetic_extension_accumulate_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::arithmetic_extension::ArithmeticExtensionGate;

        // 10 ops is the configuration used by the recursion circuits.
        for num_ops in [1, 2, 10] {
            let gate = ArithmeticExtensionGate::<2> { num_ops };
            assert_accumulate_matches_eval_unfiltered(&gate);
        }
    }

    #[test]
    fn multiplication_extension_accumulate_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::multiplication_extension::MulExtensionGate;

        // 13 ops is the configuration used by the recursion circuits.
        for num_ops in [1, 2, 13] {
            let gate = MulExtensionGate::<2> { num_ops };
            assert_accumulate_matches_eval_unfiltered(&gate);
        }
    }

    #[test]
    fn random_access_accumulate_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::random_access::RandomAccessGate;
        use plonky2::plonk::circuit_data::CircuitConfig;

        // Bits 3, 4 and 6 are the configurations used by the hot circuits;
        // bits 4 and 6 carry two extra constants each.
        for bits in [1, 2, 3, 4, 6] {
            let gate = RandomAccessGate::<GoldilocksField, 2>::new_from_config(
                &CircuitConfig::standard_recursion_config(),
                bits,
            );
            assert_accumulate_matches_eval_unfiltered(&gate);
        }
    }

    #[test]
    fn coset_interpolation_accumulate_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::coset_interpolation::CosetInterpolationGate;

        // `new(4)` yields the full-degree (16) variant with no intermediate
        // wires. The bounded-degree production shape (degree 6) is covered by
        // the vendored tests in `plonky2::gates::coset_interpolation`, which
        // can reach the crate-private bounded-degree constructor.
        let gate = CosetInterpolationGate::<GoldilocksField, 2>::new(4);
        assert_accumulate_matches_eval_unfiltered(&gate);
    }
}

/// Single-threaded microbenchmark comparing the previous accumulate path
/// (materialize `eval_unfiltered_base_batch`, then multiply-add each row) with
/// the fused `eval_unfiltered_base_batch_accumulate` override, on the gate
/// configurations used by the hot circuits. Run with:
/// `cargo test --release -p circuit --lib -- --ignored --nocapture accumulate_microbench`
#[cfg(test)]
mod accumulate_microbench {
    use std::time::Instant;

    use plonky2::field::batch_util::batch_multiply_add_inplace;
    use plonky2::gates::arithmetic_extension::ArithmeticExtensionGate;
    use plonky2::gates::multiplication_extension::MulExtensionGate;
    use plonky2::gates::random_access::RandomAccessGate;
    use plonky2::gates::reducing::ReducingGate;
    use plonky2::gates::reducing_extension::ReducingExtensionGate;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::*;

    fn bench_gate<G>(gate: &G, iters: usize)
    where
        G: Gate<GoldilocksField, 2>,
    {
        type F = GoldilocksField;

        let mut rng = rand::thread_rng();
        let n = 32;
        let wires_batch: Vec<F> = (0..gate.num_wires() * n)
            .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
            .collect();
        let constants_batch: Vec<F> = (0..gate.num_constants() * n)
            .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
            .collect();
        let filters: Vec<F> = (0..n)
            .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
            .collect();
        let public_inputs_hash = HashOut::<F>::ZERO;
        let vars_batch =
            EvaluationVarsBaseBatch::new(n, &constants_batch, &wires_batch, &public_inputs_hash);

        let mut combined = vec![F::ZERO; gate.num_constraints() * n];
        let start = Instant::now();
        for _ in 0..iters {
            let res = gate.eval_unfiltered_base_batch(vars_batch);
            for (acc, row) in combined.chunks_exact_mut(n).zip(res.chunks_exact(n)) {
                batch_multiply_add_inplace(acc, row, &filters);
            }
        }
        let old = start.elapsed();
        let old_sink = combined[0];

        let mut combined = vec![F::ZERO; gate.num_constraints() * n];
        let start = Instant::now();
        for _ in 0..iters {
            gate.eval_unfiltered_base_batch_accumulate(vars_batch, &filters, &mut combined);
        }
        let new = start.elapsed();
        assert_eq!(old_sink, combined[0], "paths diverged for {}", gate.id());

        println!(
            "{:>10.3}us/iter -> {:>10.3}us/iter ({:.2}x)  {}",
            old.as_secs_f64() * 1e6 / iters as f64,
            new.as_secs_f64() * 1e6 / iters as f64,
            old.as_secs_f64() / new.as_secs_f64(),
            gate.id(),
        );
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
    fn accumulate_microbench() {
        const ITERS: usize = 20_000;
        bench_gate(&ReducingGate::<2>::new(44), ITERS);
        bench_gate(&ReducingExtensionGate::<2>::new(33), ITERS);
        bench_gate(&ArithmeticExtensionGate::<2> { num_ops: 10 }, ITERS);
        bench_gate(&MulExtensionGate::<2> { num_ops: 13 }, ITERS);
        for bits in [3, 4, 6] {
            bench_gate(
                &RandomAccessGate::<GoldilocksField, 2>::new_from_config(
                    &CircuitConfig::standard_recursion_config(),
                    bits,
                ),
                ITERS,
            );
        }
    }
}

/// Checks a gate's `eval_unfiltered_base_batch_accumulate` against per-point
/// `eval_unfiltered`: starting from a random accumulator buffer, the fused
/// path must add exactly `filter[p] * constraint[j](p)` to every cell of the
/// row-major constraint matrix.
pub fn assert_accumulate_matches_eval_unfiltered<G>(gate: &G)
where
    G: Gate<GoldilocksField, 2>,
{
    const D: usize = 2;
    type F = GoldilocksField;

    let mut rng = rand::thread_rng();
    let n = 32;
    let num_wires = gate.num_wires();
    let num_constants = gate.num_constants();
    let num_constraints = gate.num_constraints();

    // Column-major layout: wire w for point p at [w * n + p].
    let wires_batch: Vec<F> = (0..num_wires * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let constants_batch: Vec<F> = (0..num_constants * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let filters: Vec<F> = (0..n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let initial: Vec<F> = (0..num_constraints * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let public_inputs_hash = HashOut::<F>::ZERO;

    let vars_batch =
        EvaluationVarsBaseBatch::new(n, &constants_batch, &wires_batch, &public_inputs_hash);

    let mut expected = initial.clone();
    for p in 0..n {
        let to_ext = |value: F| {
            <<F as Extendable<D>>::Extension as FieldExtension<D>>::from_basefield(value)
        };
        let wires_one: Vec<_> = (0..num_wires)
            .map(|w| to_ext(wires_batch[w * n + p]))
            .collect();
        let constants_one: Vec<_> = (0..num_constants)
            .map(|c| to_ext(constants_batch[c * n + p]))
            .collect();
        let vars_one = EvaluationVars::<F, D> {
            local_constants: &constants_one,
            local_wires: &wires_one,
            public_inputs_hash: &public_inputs_hash,
        };
        for (j, constraint) in gate.eval_unfiltered(vars_one).iter().enumerate() {
            assert!(
                <<F as Extendable<D>>::Extension as FieldExtension<D>>::is_in_basefield(constraint)
            );
            let value: [F; D] =
                <<F as Extendable<D>>::Extension as FieldExtension<D>>::to_basefield_array(
                    constraint,
                );
            expected[j * n + p] += value[0] * filters[p];
        }
    }

    let mut actual = initial;
    gate.eval_unfiltered_base_batch_accumulate(vars_batch, &filters, &mut actual);
    for j in 0..num_constraints {
        for p in 0..n {
            assert_eq!(
                actual[j * n + p],
                expected[j * n + p],
                "gate {}, point {p}, constraint {j}",
                gate.id()
            );
        }
    }
}

/// Checks a gate's `eval_unfiltered_base_batch_accumulate` override against
/// the default path (materialize the full unfiltered constraint matrix, then
/// `batch_multiply_add_inplace` row by row) across a multi-point batch with
/// deterministic pseudo-random wires, constants, and filters. Constraint order
/// and values must be bit-identical.
pub fn assert_direct_accumulation_matches_materialized_batch<G>(gate: &G)
where
    G: Gate<GoldilocksField, 2>,
{
    type F = GoldilocksField;
    const N: usize = 11;

    let wires = (0..gate.num_wires() * N)
        .map(|i| F::from_canonical_usize(3 * i + 5))
        .collect::<Vec<_>>();
    let constants = (0..gate.num_constants() * N)
        .map(|i| F::from_canonical_usize(7 * i + 11))
        .collect::<Vec<_>>();
    let hash = HashOut::ZERO;
    let vars = EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
    let filters = (0..N)
        .map(|i| F::from_canonical_usize(2 * i + 1))
        .collect::<Vec<_>>();

    let mut expected = vec![F::ZERO; gate.num_constraints() * N];
    let materialized = gate.eval_unfiltered_base_batch(vars);
    for (acc, constraints) in expected
        .as_chunks_mut::<N>()
        .0
        .iter_mut()
        .zip(materialized.as_chunks::<N>().0)
    {
        batch_multiply_add_inplace(acc, constraints, &filters);
    }
    let mut actual = vec![F::ZERO; expected.len()];
    gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
    assert_eq!(actual, expected, "gate {}", gate.id());
}

/// Batch-size-parameterized form of
/// [`assert_direct_accumulation_matches_materialized_batch`]: the same
/// materialize-then-multiply-add reference semantics, checked at an explicit
/// batch size so callers can cover packing remainders and boundary widths.
/// (The frontier's u32 interleave/uninterleave tests reference this helper but
/// the composition that promoted them lost its definition; restored here.)
pub fn assert_accumulate_matches_materialized_at_batch_size<G>(gate: &G, n: usize)
where
    G: Gate<GoldilocksField, 2>,
{
    type F = GoldilocksField;

    let wires = (0..gate.num_wires() * n)
        .map(|i| F::from_canonical_usize(3 * i + 5))
        .collect::<Vec<_>>();
    let constants = (0..gate.num_constants() * n)
        .map(|i| F::from_canonical_usize(7 * i + 11))
        .collect::<Vec<_>>();
    let hash = HashOut::ZERO;
    let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);
    let filters = (0..n)
        .map(|i| F::from_canonical_usize(2 * i + 1))
        .collect::<Vec<_>>();

    let mut expected = vec![F::ZERO; gate.num_constraints() * n];
    let materialized = gate.eval_unfiltered_base_batch(vars);
    for (acc, constraints) in expected
        .chunks_exact_mut(n)
        .zip(materialized.chunks_exact(n))
    {
        batch_multiply_add_inplace(acc, constraints, &filters);
    }
    let mut actual = vec![F::ZERO; expected.len()];
    gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
    assert_eq!(actual, expected, "gate {} at batch size {n}", gate.id());
}

pub fn assert_base_batch_matches_eval_unfiltered<G>(gate: &G)
where
    G: Gate<GoldilocksField, 2>,
{
    assert_accumulate_matches_default(gate);

    const D: usize = 2;
    type F = GoldilocksField;

    let mut rng = rand::thread_rng();
    let n = 32;
    let num_wires = gate.num_wires();
    let num_constants = gate.num_constants();
    let num_constraints = gate.num_constraints();

    // Column-major layout: wire w for point p at [w * n + p].
    let wires_batch: Vec<F> = (0..num_wires * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let constants_batch: Vec<F> = (0..num_constants * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let public_inputs_hash = HashOut::<F>::ZERO;

    let vars_batch =
        EvaluationVarsBaseBatch::new(n, &constants_batch, &wires_batch, &public_inputs_hash);
    let batch_out = gate.eval_unfiltered_base_batch(vars_batch);
    assert_eq!(batch_out.len(), n * num_constraints);

    for p in 0..n {
        let to_ext = |value: F| {
            <<F as Extendable<D>>::Extension as FieldExtension<D>>::from_basefield(value)
        };
        let wires_one: Vec<_> = (0..num_wires)
            .map(|w| to_ext(wires_batch[w * n + p]))
            .collect();
        let constants_one: Vec<_> = (0..num_constants)
            .map(|c| to_ext(constants_batch[c * n + p]))
            .collect();
        let vars_one = EvaluationVars::<F, D> {
            local_constants: &constants_one,
            local_wires: &wires_one,
            public_inputs_hash: &public_inputs_hash,
        };
        let expected = gate.eval_unfiltered(vars_one);
        for (j, expected_j) in expected.iter().enumerate() {
            assert_eq!(
                to_ext(batch_out[j * n + p]),
                *expected_j,
                "gate {}, point {p}, constraint {j}",
                gate.id()
            );
        }
    }
}

/// Check `eval_unfiltered_base_batch_accumulate` against the reference
/// semantics `combined[j*n+p] += filters[p] * constraint_j(p)` computed from
/// `eval_unfiltered_base_batch`, with random filters and a random pre-seeded
/// accumulator buffer.
pub fn assert_accumulate_matches_default<G>(gate: &G)
where
    G: Gate<GoldilocksField, 2>,
{
    type F = GoldilocksField;
    let mut rng = rand::thread_rng();
    let n = 32;
    let num_wires = gate.num_wires();
    let num_constants = gate.num_constants();
    let num_constraints = gate.num_constraints();

    let rand_f = |rng: &mut rand::rngs::ThreadRng| {
        F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER))
    };
    let wires_batch: Vec<F> = (0..num_wires * n).map(|_| rand_f(&mut rng)).collect();
    let constants_batch: Vec<F> = (0..num_constants * n).map(|_| rand_f(&mut rng)).collect();
    let filters: Vec<F> = (0..n).map(|_| rand_f(&mut rng)).collect();
    let seed: Vec<F> = (0..num_constraints * n).map(|_| rand_f(&mut rng)).collect();
    let public_inputs_hash = HashOut::<F>::ZERO;

    let vars =
        || EvaluationVarsBaseBatch::new(n, &constants_batch, &wires_batch, &public_inputs_hash);

    let mut actual = seed.clone();
    gate.eval_unfiltered_base_batch_accumulate(vars(), &filters, &mut actual);

    let res = gate.eval_unfiltered_base_batch(vars());
    let mut expected = seed;
    for (combined, res_row) in expected.chunks_exact_mut(n).zip(res.chunks_exact(n)) {
        for p in 0..n {
            combined[p] += filters[p] * res_row[p];
        }
    }
    assert_eq!(
        actual,
        expected,
        "accumulate mismatch for gate {}",
        gate.id()
    );
}

#[cfg(test)]
mod added_gate_tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::assert_accumulate_matches_default;

    #[test]
    fn quintic_square_accumulate_matches_default() {
        use crate::eddsa::gates::square_quintic_ext_base::QuinticSquaringGate;

        let gate = QuinticSquaringGate::new_from_config(
            &plonky2::plonk::circuit_data::CircuitConfig::standard_recursion_config(),
        );
        super::assert_accumulate_matches_default(&gate);
    }

    #[test]
    fn quintic_mul_accumulate_matches_default() {
        use crate::eddsa::gates::mul_quintic_ext_base::QuinticMultiplicationGate;

        let gate = QuinticMultiplicationGate::new_from_config(
            &plonky2::plonk::circuit_data::CircuitConfig::standard_recursion_config(),
        );
        super::assert_accumulate_matches_default(&gate);
    }

    #[test]
    fn add_many_u16_accumulate_matches_default() {
        use crate::uint::u16::gates::add_many_u16::U16AddManyGate;

        let gate = U16AddManyGate::<GoldilocksField, 2>::new_from_config(
            &plonky2::plonk::circuit_data::CircuitConfig::standard_recursion_config(),
            4,
        );
        super::assert_accumulate_matches_default(&gate);
    }

    #[test]
    fn comparison_accumulate_matches_default() {
        use crate::uint::u32::gates::comparison::ComparisonGate;

        for (num_bits, num_chunks) in [(32, 16), (32, 8), (30, 10)] {
            let gate = ComparisonGate::<GoldilocksField, 2>::new(num_bits, num_chunks);
            assert_accumulate_matches_default(&gate);
        }
    }

    #[test]
    fn u16_arithmetic_accumulate_matches_default() {
        use crate::uint::u16::gates::arithmetic_u16::U16ArithmeticGate;

        let gate = U16ArithmeticGate::<GoldilocksField, 2>::new_from_config(
            &CircuitConfig::standard_recursion_config(),
        );
        assert_accumulate_matches_default(&gate);
    }
}
