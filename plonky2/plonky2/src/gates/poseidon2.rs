//! Implementation of a Plonky2 gate for an entire Poseidon2 permutation over a
//! state of width 12
use core::marker::PhantomData;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon2::config::*;
use crate::hash::poseidon2::hash::Poseidon2;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Evaluates a full Poseidon2 permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently
/// verifying Merkle proofs. It has a flag which can be used to swap the first
/// four inputs with the next four, for ordering sibling digests.
#[derive(Debug, Default)]
pub struct Poseidon2Gate<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate<F, D> {
    pub fn new() -> Self {
        Poseidon2Gate {
            _phantom: PhantomData,
        }
    }

    /// The wire index for the `i`th input to the permutation.
    pub fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub fn wire_output(i: usize) -> usize {
        WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next
    /// four inputs. This is useful for ordering hashes in Merkle proofs.
    /// Otherwise, this should be set to 0.
    pub const WIRE_SWAP: usize = 2 * WIDTH;

    const START_DELTA: usize = 2 * WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute
    /// the swapped inputs.
    fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_ROUND_F_BEGIN: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th
    /// round of the first set of full rounds.
    fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < ROUNDS_F_HALF);
        debug_assert!(i < WIDTH);
        Self::START_ROUND_F_BEGIN + WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize = Self::START_ROUND_F_BEGIN + WIDTH * (ROUNDS_F_HALF - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of
    /// the partial rounds.
    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < ROUNDS_P);
        Self::START_PARTIAL + round
    }

    const START_ROUND_F_END: usize = Self::START_PARTIAL + ROUNDS_P;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th
    /// round of the second set of full rounds.
    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < ROUNDS_F_HALF);
        debug_assert!(i < WIDTH);
        Self::START_ROUND_F_END + WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    const fn end() -> usize {
        Self::START_ROUND_F_END + WIDTH * ROUNDS_F_HALF
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Gate<F, D> for Poseidon2Gate<F, D> {
    fn id(&self) -> String {
        format!("{:?}<WIDTH={}>", self, WIDTH)
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Poseidon2Gate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::Extension::ZERO; WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        <F as Poseidon2>::external_linear_layer_extension(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc_extension(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon2>::sbox_extension(&mut state);
            <F as Poseidon2>::external_linear_layer_extension(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += F::Extension::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = <F as Poseidon2>::sbox_p_extension(&state[0]);
            <F as Poseidon2>::internal_linear_layer_extension(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc_extension(&mut state, r);
            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon2>::sbox_extension(&mut state);
            <F as Poseidon2>::external_linear_layer_extension(&mut state);
        }

        for i in 0..WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_batch(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBaseBatch<F>,
    ) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_constraints() * n);
        let wires = vars_base.local_wires;
        let col = |w: usize| &wires[w * n..][..n];

        // Batches are 32 points in this prover; keep the scratch row on the
        // stack and fall back to the heap only for oversized batches.
        let mut scratch_stack = [F::ZERO; 64];
        let mut scratch_heap;
        let scratch: &mut [F] = if n <= 64 {
            &mut scratch_stack[..n]
        } else {
            scratch_heap = vec![F::ZERO; n];
            &mut scratch_heap
        };
        let mut constraint_index = 0;
        // Mirrors `eval_unfiltered_base_batch` constraint-for-constraint; each
        // row lands in `scratch` and is folded straight into the shared
        // accumulator instead of a materialized matrix.
        macro_rules! emit {
            () => {{
                let combined = &mut combined_gate_constraints
                    [constraint_index * n..(constraint_index + 1) * n];
                batch_multiply_add_inplace(combined, &scratch, filters);
                constraint_index += 1;
            }};
        }

        let mut states = vec![[F::ZERO; WIDTH]; n];

        // Assert that `swap` is binary.
        let swap = col(Self::WIRE_SWAP);
        for p in 0..n {
            scratch[p] = swap[p] * swap[p].sub_one();
        }
        emit!();

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = col(Self::wire_input(i));
            let input_rhs = col(Self::wire_input(i + 4));
            let delta_i = col(Self::wire_delta(i));
            for p in 0..n {
                scratch[p] = swap[p] * (input_rhs[p] - input_lhs[p]) - delta_i[p];
            }
            emit!();
        }

        // Compute the possibly-swapped input layer.
        for i in 0..4 {
            let delta_i = col(Self::wire_delta(i));
            let input_lhs = col(Self::wire_input(i));
            let input_rhs = col(Self::wire_input(i + 4));
            for p in 0..n {
                states[p][i] = input_lhs[p] + delta_i[p];
                states[p][i + 4] = input_rhs[p] - delta_i[p];
            }
        }
        for i in 8..WIDTH {
            let input = col(Self::wire_input(i));
            for p in 0..n {
                states[p][i] = input[p];
            }
        }

        // The initial linear layer.
        for state in states.iter_mut() {
            <F as Poseidon2>::external_linear_layer(state);
        }

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            for state in states.iter_mut() {
                <F as Poseidon2>::add_rc(state, r);
            }
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = col(Self::wire_full_sbox_0(r, i));
                    for p in 0..n {
                        scratch[p] = states[p][i] - sbox_in[p];
                        states[p][i] = sbox_in[p];
                    }
                    emit!();
                }
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            let rc = F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = col(Self::wire_partial_sbox(r));
            for p in 0..n {
                scratch[p] = states[p][0] + rc - sbox_in[p];
                states[p][0] = <F as Poseidon2>::sbox_p(&sbox_in[p]);
            }
            emit!();
            for state in states.iter_mut() {
                <F as Poseidon2>::internal_linear_layer(state);
            }
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            for state in states.iter_mut() {
                <F as Poseidon2>::add_rc(state, r);
            }
            for i in 0..WIDTH {
                let sbox_in = col(Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i));
                for p in 0..n {
                    scratch[p] = states[p][i] - sbox_in[p];
                    states[p][i] = sbox_in[p];
                }
                emit!();
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        for i in 0..WIDTH {
            let output = col(Self::wire_output(i));
            for p in 0..n {
                scratch[p] = states[p][i] - output[p];
            }
            emit!();
        }

        debug_assert_eq!(constraint_index, self.num_constraints());
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::ZERO; WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        <F as Poseidon2>::external_linear_layer(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = <F as Poseidon2>::sbox_p(&state[0]);
            <F as Poseidon2>::internal_linear_layer(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc(&mut state, r);
            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        for i in 0..WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc_circuit(builder, &mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon2>::sbox_circuit(builder, &mut state);
            <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            let round_constant = F::Extension::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let round_constant = builder.constant_extension(round_constant);
            state[0] = builder.add_extension(state[0], round_constant);

            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = sbox_in;
            state[0] = <F as Poseidon2>::sbox_p_circuit(builder, state[0]);
            <F as Poseidon2>::internal_linear_layer_circuit(builder, &mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc_circuit(builder, &mut state, r);

            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            <F as Poseidon2>::sbox_circuit(builder, &mut state);
            <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
        }

        for i in 0..WIDTH {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let g = Poseidon2Generator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(g.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        WIDTH * (ROUNDS_F - 1) + ROUNDS_P + WIDTH + 1 + 4
    }
}

/// Returns the four `delta` wire values for a Poseidon2 row, and whether the
/// generator must apply the input swap.
///
/// `WIRE_SWAP` is constrained Boolean by the gate, so the generic product
/// `swap * (state[i + 4] - state[i])` can only ever evaluate to `0` or to the
/// plain difference. Branching once on the wire value and emitting the closed
/// form deletes the four field multiplications per permutation (and, in the
/// `ZERO` arm, the four subtractions too).
///
/// The gate *constraints* enforce booleanness, but the generator runs during
/// witness generation, before any constraint is checked, so a caller that wires
/// a non-Boolean value into `WIRE_SWAP` reaches this code with it (outside
/// debug builds, where `debug_assert` catches it). The final arm therefore
/// keeps the original product form verbatim: the generated witness stays
/// value-identical to the pre-specialization code for *every* input, not just
/// Boolean ones, and such a witness still fails the gate's own Boolean
/// constraint downstream exactly as before.
#[inline]
fn swap_deltas<F: Field>(state: &[F; WIDTH], swap_value: F) -> ([F; 4], bool) {
    if swap_value == F::ZERO {
        ([F::ZERO; 4], false)
    } else if swap_value == F::ONE {
        (core::array::from_fn(|i| state[i + 4] - state[i]), true)
    } else {
        (
            core::array::from_fn(|i| swap_value * (state[i + 4] - state[i])),
            false,
        )
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2Generator<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> SimpleGenerator<F, D>
    for Poseidon2Generator<F, D>
{
    fn id(&self) -> String {
        "Poseidon2Generator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..WIDTH)
            .map(|i| Poseidon2Gate::<F, D>::wire_input(i))
            .chain(Some(Poseidon2Gate::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
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

        let mut state: [F; WIDTH] = core::array::from_fn(|i| {
            witness.get_wire(local_wire(Poseidon2Gate::<F, D>::wire_input(i)))
        });

        let swap_value = witness.get_wire(local_wire(Poseidon2Gate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        let (deltas, do_swap) = swap_deltas(&state, swap_value);
        for i in 0..4 {
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_delta(i)), deltas[i])?;
        }

        if do_swap {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        <F as Poseidon2>::external_linear_layer(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    out_buffer.set_wire(
                        local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    )?;
                }
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            out_buffer.set_wire(
                local_wire(Poseidon2Gate::<F, D>::wire_partial_sbox(r)),
                state[0],
            )?;
            state[0] = <F as Poseidon2>::sbox_p(&state[0]);
            <F as Poseidon2>::internal_linear_layer(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc(&mut state, r);
            for i in 0..WIDTH {
                out_buffer.set_wire(
                    local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_1(
                        r - ROUNDS_F_HALF,
                        i,
                    )),
                    state[i],
                )?;
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        for i in 0..WIDTH {
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_output(i)), state[i])?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

// Packed (e.g. 4-lane `WideGoldilocksField`) evaluation of the permutation and
// its consistency constraints. Every helper mirrors the scalar path's add/mul
// sequence field-exactly, so the emitted constraint values are bit-identical
// to `eval_unfiltered_base_batch`; only the lane width differs.
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> PackedEvaluableBase<F, D>
    for Poseidon2Gate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * (swap - P::ONES));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [P::ZEROS; WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            state[i] = vars.local_wires[Self::wire_input(i)] + delta_i;
            state[i + 4] = vars.local_wires[Self::wire_input(i + 4)] - delta_i;
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        packed_external_linear_layer::<F, P>(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            packed_add_rc::<F, P>(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            packed_sbox::<F, P>(&mut state);
            packed_external_linear_layer::<F, P>(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += P::from(F::from_canonical_u64(INTERNAL_CONSTANTS[r]));
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = packed_sbox_p::<F, P>(&sbox_in);
            packed_internal_linear_layer::<F, P>(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            packed_add_rc::<F, P>(&mut state, r);
            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            packed_sbox::<F, P>(&mut state);
            packed_external_linear_layer::<F, P>(&mut state);
        }

        for i in 0..WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }
}

/// Packed S-box `x -> x^7`, same formula as the scalar `sbox_p`.
#[inline]
fn packed_sbox_p<F: Poseidon2, P: PackedField<Scalar = F>>(a: &P) -> P {
    let a2 = a.square();
    let a4 = a2.square();
    let a3 = *a * a2;
    a3 * a4
}

/// Packed full S-box over the whole state.
#[inline]
fn packed_sbox<F: Poseidon2, P: PackedField<Scalar = F>>(state: &mut [P; WIDTH]) {
    state.iter_mut().for_each(|a| *a = packed_sbox_p::<F, P>(a));
}

/// Packed addition of the external round constants.
#[inline]
fn packed_add_rc<F: Poseidon2, P: PackedField<Scalar = F>>(
    state: &mut [P; WIDTH],
    external_round: usize,
) {
    debug_assert!(external_round < EXTERNAL_CONSTANTS.len());
    for i in 0..WIDTH {
        state[i] += P::from(F::from_canonical_u64(EXTERNAL_CONSTANTS[external_round][i]));
    }
}

/// Packed `M_4`-blocked circulant external linear layer. The add sequence is
/// the scalar `external_linear_layer_u128` sequence; packed lanes reduce per
/// op, which yields the same field element as the scalar one-shot reduction.
#[inline]
fn packed_external_linear_layer<F: Poseidon2, P: PackedField<Scalar = F>>(state: &mut [P; WIDTH]) {
    // First, apply M_4 to each consecutive four elements of the state.
    for i in (0..WIDTH).step_by(4) {
        let t01 = state[i] + state[i + 1];
        let t23 = state[i + 2] + state[i + 3];
        let t0123 = t01 + t23;
        let x0 = state[i];
        let x2 = state[i + 2];

        state[i] = t0123 + t01 + state[i + 1]; // 2*x[0] + 3*x[1] + x[2] + x[3]
        state[i + 1] = t0123 + state[i + 1] + x2 + x2; // x[0] + 2*x[1] + 3*x[2] + x[3]
        state[i + 2] = t0123 + t23 + state[i + 3]; // x[0] + x[1] + 2*x[2] + 3*x[3]
        state[i + 3] = t0123 + state[i + 3] + x0 + x0; // 3*x[0] + x[1] + x[2] + 2*x[3]
    }
    // Now, apply the outer circulant matrix (to compute the y_i values).
    let mut sums = [P::ZEROS; 4];
    for i in 0..4 {
        sums[i] = state[i] + state[i + 4] + state[i + 8];
    }
    for i in 0..WIDTH {
        state[i] += sums[i % 4];
    }
}

/// Packed diagonal-12 internal linear layer.
#[inline]
fn packed_internal_linear_layer<F: Poseidon2, P: PackedField<Scalar = F>>(state: &mut [P; WIDTH]) {
    let mut sum = P::ZEROS;
    for s in state.iter() {
        sum += *s;
    }
    for i in 0..WIDTH {
        state[i] = sum + state[i] * P::from(F::from_canonical_u64(MATRIX_DIAG_12_U64[i]));
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{Poseidon2Gate, *};
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon::PoseidonGate;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};
    use crate::plonk::vars::EvaluationVarsBaseBatch;

    #[test]
    fn wire_indices() {
        type F = GoldilocksField;
        type Gate = Poseidon2Gate<F, 4>;

        assert_eq!(Gate::wire_input(0), 0);
        assert_eq!(Gate::wire_input(11), 11);
        assert_eq!(Gate::wire_output(0), 12);
        assert_eq!(Gate::wire_output(11), 23);
        assert_eq!(Gate::WIRE_SWAP, 24);
        assert_eq!(Gate::wire_delta(0), 25);
        assert_eq!(Gate::wire_delta(3), 28);
        assert_eq!(Gate::wire_full_sbox_0(1, 0), 29);
        assert_eq!(Gate::wire_full_sbox_0(3, 0), 53);
        assert_eq!(Gate::wire_full_sbox_0(3, 11), 64);
        assert_eq!(Gate::wire_partial_sbox(0), 65);
        assert_eq!(Gate::wire_partial_sbox(21), 86);
        assert_eq!(Gate::wire_full_sbox_1(0, 0), 87);
        assert_eq!(Gate::wire_full_sbox_1(3, 0), 123);
        assert_eq!(Gate::wire_full_sbox_1(3, 11), 134);
    }

    #[test]
    fn generated_output() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = Poseidon2Gate<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..WIDTH).map(F::from_canonical_usize).collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs
            .set_wire(
                Wire {
                    row,
                    column: Gate::WIRE_SWAP,
                },
                F::ZERO,
            )
            .unwrap();
        for i in 0..WIDTH {
            inputs
                .set_wire(
                    Wire {
                        row,
                        column: Gate::wire_input(i),
                    },
                    permutation_inputs[i],
                )
                .unwrap();
        }

        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        let expected_outputs: [F; WIDTH] = F::poseidon2(permutation_inputs.try_into().unwrap());
        for i in 0..WIDTH {
            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs[i]);
        }
    }

    #[test]
    fn low_degree() {
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, 4>::new();
        test_low_degree(gate);

        let gate = PoseidonGate::<F, 4>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = Poseidon2Gate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)?;

        let gate = PoseidonGate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn direct_filtered_accumulation_matches_materialized_batch() {
        const D: usize = 2;
        const N: usize = 11;
        type F = GoldilocksField;

        let gate = Poseidon2Gate::<F, D>::new();
        let wires = (0..gate.num_wires() * N)
            .map(|i| F::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let constants = Vec::new();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let vars = crate::plonk::vars::EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
        let filters = (0..N)
            .map(|i| F::from_canonical_usize(2 * i + 1))
            .collect::<Vec<_>>();
        let mut expected = vec![F::ZERO; gate.num_constraints() * N];
        let materialized = gate.eval_unfiltered_base_batch(vars);
        for (acc, constraints) in expected
            .chunks_exact_mut(N)
            .zip(materialized.chunks_exact(N))
        {
            crate::field::batch_util::batch_multiply_add_inplace(acc, constraints, &filters);
        }
        let mut actual = vec![F::ZERO; expected.len()];
        gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn packed_batch_matches_scalar_per_point() {
        use plonky2_field::types::Sample;

        use crate::hash::hash_types::HashOut;

        const D: usize = 2;
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, D>::new();
        let n = 32; // several 4-lane packed groups; covers leftovers too
        let wires = F::rand_vec(gate.num_wires() * n);
        let public_inputs_hash = HashOut::rand();
        let vars_batch = EvaluationVarsBaseBatch::new(n, &[], &wires, &public_inputs_hash);

        let packed = gate.eval_unfiltered_base_batch(vars_batch);

        let mut scalar = vec![F::ZERO; n * gate.num_constraints()];
        for (i, vars_one) in vars_batch.iter().enumerate() {
            gate.eval_unfiltered_base_one(
                vars_one,
                StridedConstraintConsumer::new(&mut scalar, n, i),
            );
        }
        assert_eq!(
            packed, scalar,
            "packed batch eval diverges from scalar per-point eval"
        );
    }

    // --- microbenchmark: scalar-fused vs packed-fused vs materialized accumulate ---

    fn scalar_fused_accumulate<F: RichField + Extendable<D> + Poseidon2, const D: usize>(
        gate: &Poseidon2Gate<F, D>,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= gate.num_constraints() * n);
        let wires = vars_base.local_wires;
        let col = |w: usize| &wires[w * n..][..n];

        // Batches are 32 points in this prover; keep the scratch row on the
        // stack and fall back to the heap only for oversized batches.
        let mut scratch_stack = [F::ZERO; 64];
        let mut scratch_heap;
        let scratch: &mut [F] = if n <= 64 {
            &mut scratch_stack[..n]
        } else {
            scratch_heap = vec![F::ZERO; n];
            &mut scratch_heap
        };
        let mut constraint_index = 0;
        // Mirrors `eval_unfiltered_base_batch` constraint-for-constraint; each
        // row lands in `scratch` and is folded straight into the shared
        // accumulator instead of a materialized matrix.
        macro_rules! emit {
            () => {{
                let combined = &mut combined_gate_constraints
                    [constraint_index * n..(constraint_index + 1) * n];
                crate::field::batch_util::batch_multiply_add_inplace(combined, &scratch, filters);
                constraint_index += 1;
            }};
        }

        let mut states = vec![[F::ZERO; WIDTH]; n];

        // Assert that `swap` is binary.
        let swap = col(Poseidon2Gate::<F, D>::WIRE_SWAP);
        for p in 0..n {
            scratch[p] = swap[p] * swap[p].sub_one();
        }
        emit!();

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = col(Poseidon2Gate::<F, D>::wire_input(i));
            let input_rhs = col(Poseidon2Gate::<F, D>::wire_input(i + 4));
            let delta_i = col(Poseidon2Gate::<F, D>::wire_delta(i));
            for p in 0..n {
                scratch[p] = swap[p] * (input_rhs[p] - input_lhs[p]) - delta_i[p];
            }
            emit!();
        }

        // Compute the possibly-swapped input layer.
        for i in 0..4 {
            let delta_i = col(Poseidon2Gate::<F, D>::wire_delta(i));
            let input_lhs = col(Poseidon2Gate::<F, D>::wire_input(i));
            let input_rhs = col(Poseidon2Gate::<F, D>::wire_input(i + 4));
            for p in 0..n {
                states[p][i] = input_lhs[p] + delta_i[p];
                states[p][i + 4] = input_rhs[p] - delta_i[p];
            }
        }
        for i in 8..WIDTH {
            let input = col(Poseidon2Gate::<F, D>::wire_input(i));
            for p in 0..n {
                states[p][i] = input[p];
            }
        }

        // The initial linear layer.
        for state in states.iter_mut() {
            <F as Poseidon2>::external_linear_layer(state);
        }

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            for state in states.iter_mut() {
                <F as Poseidon2>::add_rc(state, r);
            }
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = col(Poseidon2Gate::<F, D>::wire_full_sbox_0(r, i));
                    for p in 0..n {
                        scratch[p] = states[p][i] - sbox_in[p];
                        states[p][i] = sbox_in[p];
                    }
                    emit!();
                }
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            let rc = F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = col(Poseidon2Gate::<F, D>::wire_partial_sbox(r));
            for p in 0..n {
                scratch[p] = states[p][0] + rc - sbox_in[p];
                states[p][0] = <F as Poseidon2>::sbox_p(&sbox_in[p]);
            }
            emit!();
            for state in states.iter_mut() {
                <F as Poseidon2>::internal_linear_layer(state);
            }
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            for state in states.iter_mut() {
                <F as Poseidon2>::add_rc(state, r);
            }
            for i in 0..WIDTH {
                let sbox_in = col(Poseidon2Gate::<F, D>::wire_full_sbox_1(
                    r - ROUNDS_F_HALF,
                    i,
                ));
                for p in 0..n {
                    scratch[p] = states[p][i] - sbox_in[p];
                    states[p][i] = sbox_in[p];
                }
                emit!();
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        for i in 0..WIDTH {
            let output = col(Poseidon2Gate::<F, D>::wire_output(i));
            for p in 0..n {
                scratch[p] = states[p][i] - output[p];
            }
            emit!();
        }

        debug_assert_eq!(constraint_index, gate.num_constraints());
    }

    /// Manual timing harness. Run with:
    /// `cargo test --release -p plonky2 accumulate_micro -- --ignored --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn accumulate_microbenchmark() {
        use core::hint::black_box;
        use std::time::Instant;

        use plonky2_field::types::Sample;

        use crate::gates::gate::Gate;

        const D: usize = 2;
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, D>::new();
        let n = 32;
        let wires = F::rand_vec(gate.num_wires() * n);
        let constants: Vec<F> = Vec::new();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let filters = F::rand_vec(n);
        let nc = gate.num_constraints();
        let mut combined = vec![F::ZERO; nc * n];
        let iters = 20_000u32;

        let vars = crate::plonk::vars::EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

        // Interleave measurement blocks A/B/C/A/B/C to cancel drift.
        let mut t_scalar = 0.0f64;
        let mut t_packed = 0.0f64;
        let mut t_mat = 0.0f64;
        for _ in 0..4 {
            let s = Instant::now();
            for _ in 0..iters {
                scalar_fused_accumulate(&gate, vars, &filters, black_box(&mut combined));
            }
            t_scalar += s.elapsed().as_secs_f64();

            let s = Instant::now();
            for _ in 0..iters {
                gate.eval_unfiltered_base_batch_accumulate(
                    vars,
                    &filters,
                    black_box(&mut combined),
                );
            }
            t_packed += s.elapsed().as_secs_f64();

            let s = Instant::now();
            for _ in 0..iters {
                let res = gate.eval_unfiltered_base_batch(vars);
                for (acc, row) in combined.chunks_exact_mut(n).zip(res.chunks_exact(n)) {
                    crate::field::batch_util::batch_multiply_add_inplace(acc, row, &filters);
                }
            }
            t_mat += s.elapsed().as_secs_f64();
        }
        let per = |t: f64| t / (4.0 * iters as f64) * 1e6;
        println!(
            "poseidon2 accumulate per batch (n=32): scalar-fused {:.2} us, packed-fused {:.2} us, materialized-packed {:.2} us",
            per(t_scalar), per(t_packed), per(t_mat)
        );
    }

    /// The Boolean-specialized `swap_deltas` must be raw-`u64` identical to the
    /// unconditional `swap * (state[i + 4] - state[i])` product it replaced —
    /// on both Boolean wire values *and* on non-Boolean ones, which the
    /// fallback arm still has to reproduce because witness generation runs
    /// before any constraint check.
    #[test]
    fn swap_deltas_matches_product_form() {
        type F = GoldilocksField;

        let mut swaps = vec![F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        swaps.extend((0..8).map(|_| F::rand()));

        for trial in 0..16 {
            let state: [F; WIDTH] =
                core::array::from_fn(|i| F::from_canonical_usize(trial * WIDTH + i) * F::rand());

            for &swap_value in &swaps {
                // Reference: the pre-specialization code, verbatim.
                let expected: [F; 4] =
                    core::array::from_fn(|i| swap_value * (state[i + 4] - state[i]));
                let expected_swap = swap_value == F::ONE;

                let (actual, do_swap) = swap_deltas(&state, swap_value);
                assert_eq!(do_swap, expected_swap, "swap flag for {swap_value}");
                for i in 0..4 {
                    assert_eq!(actual[i].0, expected[i].0, "delta {i} for {swap_value}");
                }
            }
        }
    }

    /// End-to-end through the real generator: for each Boolean `WIRE_SWAP`
    /// value, every `delta` and `output` wire the generator writes must equal
    /// the reference permutation applied to the (conditionally swapped) inputs.
    #[test]
    fn generated_deltas_and_outputs_for_both_swap_values() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Gate = Poseidon2Gate<F, D>;

        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        let row = builder.add_gate(Gate::new(), vec![]);
        let circuit = builder.build_prover::<C>();

        for swap_value in [F::ZERO, F::ONE] {
            let permutation_inputs: [F; WIDTH] =
                core::array::from_fn(|i| F::from_canonical_usize(7 * i + 3));

            let mut inputs = PartialWitness::new();
            inputs
                .set_wire(
                    Wire {
                        row,
                        column: Gate::WIRE_SWAP,
                    },
                    swap_value,
                )
                .unwrap();
            for i in 0..WIDTH {
                inputs
                    .set_wire(
                        Wire {
                            row,
                            column: Gate::wire_input(i),
                        },
                        permutation_inputs[i],
                    )
                    .unwrap();
            }

            let witness =
                generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

            let mut swapped = permutation_inputs;
            if swap_value == F::ONE {
                for i in 0..4 {
                    swapped.swap(i, 4 + i);
                }
            }
            let expected_outputs: [F; WIDTH] = F::poseidon2(swapped);

            for i in 0..4 {
                let expected = swap_value * (permutation_inputs[i + 4] - permutation_inputs[i]);
                let got = witness.get_wire(Wire {
                    row,
                    column: Gate::wire_delta(i),
                });
                assert_eq!(got.0, expected.0, "delta {i} for swap = {swap_value}");
            }
            for i in 0..WIDTH {
                let got = witness.get_wire(Wire {
                    row,
                    column: Gate::wire_output(i),
                });
                assert_eq!(
                    got.0, expected_outputs[i].0,
                    "output {i} for swap = {swap_value}"
                );
            }
        }
    }
}
