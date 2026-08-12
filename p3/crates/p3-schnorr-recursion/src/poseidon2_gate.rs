//! A single-row Plonky2 gate for the P3 Poseidon2 permutation over GoldilocksField
//! with WIDTH=16, 4 external half-rounds, 22 internal rounds.
//!
//! Wire layout (all non-routed wires except inputs/outputs):
//!   [0..16)           inputs
//!   [16..32)          outputs
//!   [32..80)          full_sbox_0: sbox inputs for external rounds 1..3 (W*(F_HALF-1) = 48)
//!   [80..102)         partial_sbox: sbox inputs for each of the 22 internal rounds
//!   [102..166)        full_sbox_1: sbox inputs for external rounds 0..3 of second half (W*F_HALF = 64)
//!
//! Total: 166 wires. Requires CircuitConfig::num_wires >= 166.
//! Only inputs (0..32) need to be routed; sbox intermediates are unrouted witness cells.
//!
//! # Why a custom gate, and why it computes the permutation 4 times
//!
//! The pinned Plonky2 fork's own `Poseidon2Gate` targets width 12 with
//! Plonky2's own round-constant configuration, not the width-16 permutation
//! with `p3_goldilocks`'s exact round constants that this crate needs to
//! match the P3 STARK's own Poseidon2 hash bit-for-bit. This gate
//! reimplements that specific permutation instead. Packing one full
//! permutation into a single custom gate row is also cheaper than emitting
//! the equivalent arithmetic as ordinary wires/gates, and lets the rest of
//! this crate's circuits treat "permute one P3 Poseidon2 state" as one
//! opaque step.
//!
//! Per Plonky2's `Gate` trait contract, the same round-by-round computation
//! must be implemented four times, once per evaluation domain, and these
//! four must always agree (see `test_p3_poseidon2_gate_matches_native`,
//! which only cross-checks one of them against `p3_goldilocks`'s own
//! permutation — a regression in the others would not be caught by that
//! test alone):
//! - `p3_poseidon2_permute`/`sbox_p`/native `*_linear_layer`: plain
//!   `RichField` arithmetic, used by `P3Poseidon2Generator::run_once` to
//!   fill in the witness.
//! - `eval_unfiltered_base_one`: the same computation over the base field,
//!   re-deriving each sbox's input from the wires and asserting it matches
//!   what filling the witness would have produced — this is the actual
//!   constraint check during fast native verification.
//! - `eval_unfiltered`/`*_ext`: the same over the field extension, for
//!   constraint-polynomial evaluation during proving.
//! - `eval_unfiltered_circuit`/`*_circuit`: the same again, but emitting
//!   `CircuitBuilder` targets/gates instead of evaluating — used when this
//!   gate's constraints themselves need to be proved inside another
//!   (outer) recursive circuit.

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::gates::gate::Gate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::wire::Wire;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

pub const P3_WIDTH: usize = 16;
const ROUNDS_F_HALF: usize = 4;
const ROUNDS_F: usize = 8;
const ROUNDS_P: usize = 22;

// ── Wire index helpers ──────────────────────────────────────────────────────

const fn wire_input(i: usize) -> usize {
    i
}
const fn wire_output(i: usize) -> usize {
    P3_WIDTH + i
}

// Sbox inputs for external rounds 1..ROUNDS_F_HALF-1 (round 0 sbox inputs are
// the gate inputs themselves after the initial MDS, so no extra wire needed).
//
// These `wire_*_sbox_*` wires exist only so each round's sbox input/output
// can be asserted equal via a constraint without recomputing every prior
// round inline at every evaluation site — they're scratch space for the
// permutation's intermediate state, not independently meaningful values.
const START_FULL_0: usize = 2 * P3_WIDTH;
const fn wire_full_sbox_0(round: usize, i: usize) -> usize {
    debug_assert!(round > 0 && round < ROUNDS_F_HALF);
    START_FULL_0 + P3_WIDTH * (round - 1) + i
}

// Sbox inputs for internal rounds.
const START_PARTIAL: usize = START_FULL_0 + P3_WIDTH * (ROUNDS_F_HALF - 1);
const fn wire_partial_sbox(round: usize) -> usize {
    START_PARTIAL + round
}

// Sbox inputs for the second half of external rounds (all ROUNDS_F_HALF of them).
const START_FULL_1: usize = START_PARTIAL + ROUNDS_P;
const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
    START_FULL_1 + P3_WIDTH * round + i
}

pub const NUM_WIRES: usize = START_FULL_1 + P3_WIDTH * ROUNDS_F_HALF; // = 166

// ── P3 round constants (GoldilocksField u64 values) ─────────────────────────

// These are the same arrays used in our hand-rolled poseidon2_permute_targets.
// We import them directly from p3_goldilocks to stay in sync.
use p3_field::PrimeField64 as P3PrimeField64;
use p3_goldilocks::{
    GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    GOLDILOCKS_POSEIDON2_RC_16_INTERNAL, MATRIX_DIAG_16_GOLDILOCKS,
};

fn p3_to_u64(x: p3_goldilocks::Goldilocks) -> u64 {
    x.as_canonical_u64()
}

// ── Native field arithmetic for the permutation ──────────────────────────────
//
// `sbox_p`/`external_linear_layer`/`apply_mat4`/`internal_linear_layer` are
// the Plonky2-`Field`-generic counterparts of `crate::poseidon`'s (and, one
// level further back, `p3_goldilocks`'s) Poseidon2 round functions — same
// math, different field trait bound, since this gate operates over Plonky2
// `RichField`/`Field` types rather than `p3_field::PrimeCharacteristicRing`.

/// `x^7`, the Poseidon2 S-box for this field.
fn sbox_p<F: Field>(x: F) -> F {
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

/// The external (full-round) linear layer: [`apply_mat4`] per 4-lane chunk,
/// then cross-chunk mixing by column sums.
fn external_linear_layer<F: Field>(state: &mut [F; P3_WIDTH]) {
    // Apply mat4 to each group of 4.
    for chunk in state.chunks_exact_mut(4) {
        let x: &mut [F; 4] = chunk.try_into().unwrap();
        apply_mat4(x);
    }
    // Cross-sum: each element gets the sum of the element at its position mod 4
    // across all four groups.
    let sums: [F; 4] =
        core::array::from_fn(|k| (0..P3_WIDTH).step_by(4).map(|j| state[j + k]).sum());
    for i in 0..P3_WIDTH {
        state[i] += sums[i % 4];
    }
}

/// The fixed 4x4 MDS-like matrix applied to one lane chunk.
fn apply_mat4<F: Field>(x: &mut [F; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    let x0d = x[0].double();
    let x2d = x[2].double();
    x[3] = t01233 + x0d;
    x[1] = t01123 + x2d;
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

/// The internal (partial-round) linear layer: `state[i] = state[i]*diag[i] + sum(state)`.
fn internal_linear_layer<F: Field>(state: &mut [F; P3_WIDTH]) {
    let sum: F = state.iter().copied().sum();
    for (i, v) in state.iter_mut().enumerate() {
        let diag = F::from_canonical_u64(p3_to_u64(MATRIX_DIAG_16_GOLDILOCKS[i]));
        *v = *v * diag + sum;
    }
}

/// Dead code: this whole-permutation computation is not called anywhere —
/// the gate's actual constraints/witness-filling (`eval_unfiltered_base_one`,
/// `P3Poseidon2Generator::run_once`, etc.) reimplement these same rounds
/// inline instead (see the module doc's "computes the permutation 4 times"
/// section), and the test that validates the gate calls
/// `p3_goldilocks::default_goldilocks_poseidon2_16` directly as its oracle.
/// Kept only as documentation of the round structure; candidate for removal.
fn p3_poseidon2_permute<F: Field>(state: &mut [F; P3_WIDTH]) {
    // Initial MDS.
    external_linear_layer(state);

    // First half of external rounds.
    for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL
        .iter()
        .enumerate()
    {
        for (v, &c) in state.iter_mut().zip(rc.iter()) {
            *v += F::from_canonical_u64(p3_to_u64(c));
        }
        for v in state.iter_mut() {
            *v = sbox_p(*v);
        }
        external_linear_layer(state);
    }

    // Internal rounds.
    for (r, &rc) in GOLDILOCKS_POSEIDON2_RC_16_INTERNAL.iter().enumerate() {
        state[0] += F::from_canonical_u64(p3_to_u64(rc));
        state[0] = sbox_p(state[0]);
        internal_linear_layer(state);
    }

    // Second half of external rounds.
    for rc in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL.iter() {
        for (v, &c) in state.iter_mut().zip(rc.iter()) {
            *v += F::from_canonical_u64(p3_to_u64(c));
        }
        for v in state.iter_mut() {
            *v = sbox_p(*v);
        }
        external_linear_layer(state);
    }
}

// ── Extension-field helpers for constraint evaluation ────────────────────────

fn external_linear_layer_ext<F: RichField + Extendable<D>, const D: usize>(
    state: &mut [F::Extension; P3_WIDTH],
) {
    for chunk in state.chunks_exact_mut(4) {
        let x: &mut [F::Extension; 4] = chunk.try_into().unwrap();
        apply_mat4(x);
    }
    let sums: [F::Extension; 4] =
        core::array::from_fn(|k| (0..P3_WIDTH).step_by(4).map(|j| state[j + k]).sum());
    for i in 0..P3_WIDTH {
        state[i] += sums[i % 4];
    }
}

fn internal_linear_layer_ext<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; P3_WIDTH],
) {
    let sum = builder.add_many_extension(state.iter().copied());
    for (i, v) in state.iter_mut().enumerate() {
        let diag = F::Extension::from_canonical_u64(p3_to_u64(MATRIX_DIAG_16_GOLDILOCKS[i]));
        let diag_t = builder.constant_extension(diag);
        *v = builder.mul_add_extension(diag_t, *v, sum);
    }
}

fn sbox_p_ext<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    x: ExtensionTarget<D>,
) -> ExtensionTarget<D> {
    builder.exp_u64_extension(x, 7)
}

fn external_linear_layer_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; P3_WIDTH],
) {
    for chunk in state.chunks_exact_mut(4) {
        let x: &mut [ExtensionTarget<D>; 4] = chunk.try_into().unwrap();
        apply_mat4_circuit(builder, x);
    }
    let sums: [ExtensionTarget<D>; 4] = core::array::from_fn(|k| {
        (0..P3_WIDTH)
            .step_by(4)
            .map(|j| state[j + k])
            .reduce(|a, b| builder.add_extension(a, b))
            .unwrap()
    });
    for i in 0..P3_WIDTH {
        state[i] = builder.add_extension(state[i], sums[i % 4]);
    }
}

fn apply_mat4_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    x: &mut [ExtensionTarget<D>; 4],
) {
    let t01 = builder.add_extension(x[0], x[1]);
    let t23 = builder.add_extension(x[2], x[3]);
    let t0123 = builder.add_extension(t01, t23);
    let t01123 = builder.add_extension(t0123, x[1]);
    let t01233 = builder.add_extension(t0123, x[3]);
    let two = builder.constant_extension(F::Extension::TWO);
    let x0d = builder.mul_extension(two, x[0]);
    let x2d = builder.mul_extension(two, x[2]);
    x[3] = builder.add_extension(t01233, x0d);
    x[1] = builder.add_extension(t01123, x2d);
    x[0] = builder.add_extension(t01123, t01);
    x[2] = builder.add_extension(t01233, t23);
}

// ── The Gate ─────────────────────────────────────────────────────────────────

/// The Plonky2 `Gate` implementing one full P3-flavored Poseidon2
/// permutation (width 16) per row. See the module docs for the wire layout
/// and why this exists instead of using the pinned fork's own
/// `Poseidon2Gate`.
#[derive(Debug, Default, Clone)]
pub struct P3Poseidon2Gate<const D: usize> {
    _phantom: PhantomData<()>,
}

impl<const D: usize> P3Poseidon2Gate<D> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for P3Poseidon2Gate<D> {
    fn id(&self) -> String {
        format!("P3Poseidon2Gate<WIDTH={P3_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(150);

        let mut state: [F::Extension; P3_WIDTH] =
            core::array::from_fn(|i| vars.local_wires[wire_input(i)]);

        external_linear_layer_ext::<F, D>(&mut state);

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL
            .iter()
            .enumerate()
        {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                *v += F::Extension::from_canonical_u64(p3_to_u64(c));
            }
            if r != 0 {
                for i in 0..P3_WIDTH {
                    let sbox_in = vars.local_wires[wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            for v in state.iter_mut() {
                *v = v.exp_u64(7);
            }
            external_linear_layer_ext::<F, D>(&mut state);
        }

        for (r, &rc) in GOLDILOCKS_POSEIDON2_RC_16_INTERNAL.iter().enumerate() {
            state[0] += F::Extension::from_canonical_u64(p3_to_u64(rc));
            let sbox_in = vars.local_wires[wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = state[0].exp_u64(7);
            // Internal linear layer (field extension version, no builder needed).
            let sum: F::Extension = state.iter().copied().sum();
            for (i, v) in state.iter_mut().enumerate() {
                let diag =
                    F::Extension::from_canonical_u64(p3_to_u64(MATRIX_DIAG_16_GOLDILOCKS[i]));
                *v = *v * diag + sum;
            }
        }

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL.iter().enumerate() {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                *v += F::Extension::from_canonical_u64(p3_to_u64(c));
            }
            for i in 0..P3_WIDTH {
                let sbox_in = vars.local_wires[wire_full_sbox_1(r, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            for v in state.iter_mut() {
                *v = v.exp_u64(7);
            }
            external_linear_layer_ext::<F, D>(&mut state);
        }

        for i in 0..P3_WIDTH {
            constraints.push(state[i] - vars.local_wires[wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let mut state: [F; P3_WIDTH] = core::array::from_fn(|i| vars.local_wires[wire_input(i)]);

        external_linear_layer(&mut state);

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL
            .iter()
            .enumerate()
        {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                *v += F::from_canonical_u64(p3_to_u64(c));
            }
            if r != 0 {
                for i in 0..P3_WIDTH {
                    let sbox_in = vars.local_wires[wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            for v in state.iter_mut() {
                *v = sbox_p(*v);
            }
            external_linear_layer(&mut state);
        }

        for (r, &rc) in GOLDILOCKS_POSEIDON2_RC_16_INTERNAL.iter().enumerate() {
            state[0] += F::from_canonical_u64(p3_to_u64(rc));
            let sbox_in = vars.local_wires[wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = sbox_p(state[0]);
            internal_linear_layer(&mut state);
        }

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL.iter().enumerate() {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                *v += F::from_canonical_u64(p3_to_u64(c));
            }
            for i in 0..P3_WIDTH {
                let sbox_in = vars.local_wires[wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            for v in state.iter_mut() {
                *v = sbox_p(*v);
            }
            external_linear_layer(&mut state);
        }

        for i in 0..P3_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(150);

        let mut state: [ExtensionTarget<D>; P3_WIDTH] =
            core::array::from_fn(|i| vars.local_wires[wire_input(i)]);

        external_linear_layer_circuit(builder, &mut state);

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL
            .iter()
            .enumerate()
        {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                let rc_t =
                    builder.constant_extension(F::Extension::from_canonical_u64(p3_to_u64(c)));
                *v = builder.add_extension(*v, rc_t);
            }
            if r != 0 {
                for i in 0..P3_WIDTH {
                    let sbox_in = vars.local_wires[wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            for v in state.iter_mut() {
                *v = sbox_p_ext(builder, *v);
            }
            external_linear_layer_circuit(builder, &mut state);
        }

        for (r, &rc) in GOLDILOCKS_POSEIDON2_RC_16_INTERNAL.iter().enumerate() {
            let rc_t = builder.constant_extension(F::Extension::from_canonical_u64(p3_to_u64(rc)));
            state[0] = builder.add_extension(state[0], rc_t);
            let sbox_in = vars.local_wires[wire_partial_sbox(r)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = sbox_in;
            state[0] = sbox_p_ext(builder, state[0]);
            internal_linear_layer_ext(builder, &mut state);
        }

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL.iter().enumerate() {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                let rc_t =
                    builder.constant_extension(F::Extension::from_canonical_u64(p3_to_u64(c)));
                *v = builder.add_extension(*v, rc_t);
            }
            for i in 0..P3_WIDTH {
                let sbox_in = vars.local_wires[wire_full_sbox_1(r, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            for v in state.iter_mut() {
                *v = sbox_p_ext(builder, *v);
            }
            external_linear_layer_circuit(builder, &mut state);
        }

        for i in 0..P3_WIDTH {
            constraints.push(builder.sub_extension(state[i], vars.local_wires[wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            P3Poseidon2Generator::<F, D> {
                row,
                _phantom: PhantomData,
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        NUM_WIRES
    }
    fn num_constants(&self) -> usize {
        0
    }
    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        // full_sbox_0: rounds 1..ROUNDS_F_HALF, W constraints each
        let full_0 = P3_WIDTH * (ROUNDS_F_HALF - 1);
        // partial_sbox: ROUNDS_P constraints
        let partial = ROUNDS_P;
        // full_sbox_1: ROUNDS_F_HALF rounds, W constraints each
        let full_1 = P3_WIDTH * ROUNDS_F_HALF;
        // outputs: W constraints
        let outputs = P3_WIDTH;
        full_0 + partial + full_1 + outputs
    }
}

// ── Witness generator ─────────────────────────────────────────────────────────

/// The witness generator that fills [`P3Poseidon2Gate`]'s sbox-intermediate
/// wires for one gate instance at `row`, given the gate's input wires are
/// already assigned.
#[derive(Debug, Default)]
pub struct P3Poseidon2Generator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for P3Poseidon2Generator<F, D>
{
    fn id(&self) -> String {
        "P3Poseidon2Generator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..P3_WIDTH)
            .map(|i| Target::wire(self.row, wire_input(i)))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state: [F; P3_WIDTH] =
            core::array::from_fn(|i| witness.get_wire(local_wire(wire_input(i))));

        external_linear_layer(&mut state);

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL
            .iter()
            .enumerate()
        {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                *v += F::from_canonical_u64(p3_to_u64(c));
            }
            if r != 0 {
                for i in 0..P3_WIDTH {
                    out_buffer.set_wire(local_wire(wire_full_sbox_0(r, i)), state[i])?;
                }
            }
            for v in state.iter_mut() {
                *v = sbox_p(*v);
            }
            external_linear_layer(&mut state);
        }

        for (r, &rc) in GOLDILOCKS_POSEIDON2_RC_16_INTERNAL.iter().enumerate() {
            state[0] += F::from_canonical_u64(p3_to_u64(rc));
            out_buffer.set_wire(local_wire(wire_partial_sbox(r)), state[0])?;
            state[0] = sbox_p(state[0]);
            internal_linear_layer(&mut state);
        }

        for (r, rc) in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL.iter().enumerate() {
            for (v, &c) in state.iter_mut().zip(rc.iter()) {
                *v += F::from_canonical_u64(p3_to_u64(c));
            }
            for i in 0..P3_WIDTH {
                out_buffer.set_wire(local_wire(wire_full_sbox_1(r, i)), state[i])?;
            }
            for v in state.iter_mut() {
                *v = sbox_p(*v);
            }
            external_linear_layer(&mut state);
        }

        for i in 0..P3_WIDTH {
            out_buffer.set_wire(local_wire(wire_output(i)), state[i])?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            row: src.read_usize()?,
            _phantom: PhantomData,
        })
    }
}

// ── Builder integration ───────────────────────────────────────────────────────

/// Emit one `P3Poseidon2Gate` row and wire `inputs` to its input wires.
/// Returns the 16 output targets.
pub fn p3_poseidon2_permute_gate<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    inputs: &[Target; P3_WIDTH],
) -> [Target; P3_WIDTH] {
    let gate = P3Poseidon2Gate::<D>::new();
    let row = builder.add_gate(gate, vec![]);
    for (i, &inp) in inputs.iter().enumerate() {
        builder.connect(inp, Target::wire(row, wire_input(i)));
    }
    core::array::from_fn(|i| Target::wire(row, wire_output(i)))
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::Poseidon2GoldilocksConfig;

    use super::*;

    type F = GoldilocksField;
    type C = Poseidon2GoldilocksConfig;
    const D: usize = 2;

    fn make_config() -> CircuitConfig {
        let mut cfg = CircuitConfig::standard_recursion_config();
        cfg.num_wires = NUM_WIRES;
        cfg
    }

    #[test]
    fn test_p3_poseidon2_gate_matches_native() {
        use p3_goldilocks::Goldilocks as P3Goldilocks;
        use p3_symmetric::Permutation;

        let input_vals: [u64; P3_WIDTH] =
            core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x1234567890abcdef));
        let mut p3_state: [P3Goldilocks; P3_WIDTH] = input_vals.map(P3Goldilocks::new);

        let p3_perm = p3_goldilocks::default_goldilocks_poseidon2_16();
        p3_perm.permute_mut(&mut p3_state);
        let expected: [u64; P3_WIDTH] = p3_state.map(|v: P3Goldilocks| v.as_canonical_u64());

        let mut builder = CircuitBuilder::<F, D>::new(make_config());
        let input_targets: [Target; P3_WIDTH] =
            input_vals.map(|v| builder.constant(F::from_canonical_u64(v)));
        let output_targets = p3_poseidon2_permute_gate(&mut builder, &input_targets);
        let expected_targets: [Target; P3_WIDTH] =
            expected.map(|v| builder.constant(F::from_canonical_u64(v)));
        for (out, exp) in output_targets.iter().zip(expected_targets.iter()) {
            builder.connect(*out, *exp);
        }

        let data = builder.build::<C>();
        let proof = data.prove(PartialWitness::new()).expect("proof failed");
        data.verify(proof).expect("verify failed");
    }

    #[test]
    fn test_p3_poseidon2_gate_constraint_count() {
        use plonky2::gates::gate::Gate;
        // full_0: W*(F_HALF-1) = 16*3 = 48
        // partial: 22
        // full_1: W*F_HALF = 16*4 = 64
        // outputs: 16
        assert_eq!(
            <P3Poseidon2Gate<D> as Gate<F, D>>::num_constraints(&P3Poseidon2Gate::new()),
            48 + 22 + 64 + 16
        );
        assert_eq!(
            <P3Poseidon2Gate<D> as Gate<F, D>>::num_wires(&P3Poseidon2Gate::new()),
            NUM_WIRES
        );
        assert_eq!(
            <P3Poseidon2Gate<D> as Gate<F, D>>::degree(&P3Poseidon2Gate::new()),
            7
        );
    }
}
