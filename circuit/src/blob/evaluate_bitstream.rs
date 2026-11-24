// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::marker::PhantomData;
use core::ops::Range;

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::gates::gate::Gate;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars};

use crate::builder::Builder;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::plonky2::util::serialization::{Buffer, IoResult, Read, Write};
use crate::types::quintuple::*;

#[derive(Debug, Clone, Default)]
pub struct EvaluateBitstreamGate {
    /// Number of Bitstream evaluation operations that can be performed using a single gate
    pub num_states: usize,
}

impl EvaluateBitstreamGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_states: Self::num_states(config),
        }
    }

    const DEGREE_WIRES: usize = 2;
    const WIRES_CONSTANTS: usize = 5;
    const WIRES_PER_STATE: usize = 9;
    //Number of wires representing a transition between states
    const WIRES_PER_TRANSITION: usize = 2;

    //one extra for chunks
    const TOTAL_PER_OP: usize = Self::WIRES_PER_STATE + Self::WIRES_PER_TRANSITION + 1;

    /// Determine the maximum number of states that can fit in one gate for the given config.
    pub(crate) const fn num_states(config: &CircuitConfig) -> usize {
        let unrouted_packed_count =
            (config.num_wires - Self::WIRES_CONSTANTS - Self::DEGREE_WIRES - Self::WIRES_PER_STATE)
                / Self::TOTAL_PER_OP;
        unrouted_packed_count + 1
    }

    //`x` is shared between all operations
    pub(crate) const fn wire_x_and_chunks(&self) -> Range<usize> {
        let start = Self::DEGREE_WIRES;
        start..5 + self.num_states - 1 + Self::DEGREE_WIRES
    }

    pub(crate) const fn wire_x(&self) -> Range<usize> {
        let start = self.wire_x_and_chunks().start;
        start..start + 5
    }

    pub(crate) const fn wire_chunk(&self, i: usize) -> usize {
        assert!(1 <= i && i < self.num_states);
        let start = self.wire_x_and_chunks().start + 5;
        start + i - 1
    }

    //state getters
    pub(crate) const fn wire_state(&self, i: usize) -> Range<usize> {
        assert!(i < self.num_states);
        let start: usize = self.wire_x_and_chunks().end;

        //Places the first state and the last state in the routed wires area so they can be interacted with
        if i == 0 {
            start..start + Self::WIRES_PER_STATE
        } else if i == self.num_states - 1 {
            start + Self::WIRES_PER_STATE..start + 2 * Self::WIRES_PER_STATE
        } else {
            let start = self.wire_x_and_chunks().end + Self::WIRES_PER_STATE * (i + 1);
            start..start + Self::WIRES_PER_STATE
        }
    }

    pub(crate) const fn wire_sum(&self, i: usize) -> Range<usize> {
        let start_of_state: usize = self.wire_state(i).start;
        start_of_state..start_of_state + 5
    }

    pub(crate) const fn wire_number_accumulator(&self, i: usize) -> usize {
        let start_of_state: usize = self.wire_state(i).start;
        start_of_state + 5
    }

    pub(crate) const fn wire_chunks_left(&self, i: usize) -> usize {
        let start_of_state: usize = self.wire_state(i).start;
        start_of_state + 6
    }

    pub(crate) const fn wire_number_ending(&self, i: usize) -> usize {
        let start_of_state: usize = self.wire_state(i).start;
        start_of_state + 7
    }

    pub(crate) const fn wire_not_number_ending(&self, i: usize) -> usize {
        let start_of_transition = self.wire_state(i).start;
        start_of_transition + 8
    }

    //This is in theory necessary only for transitions
    const CHUNK_MAX: usize = 16;

    //transition wire getters
    pub(crate) const fn wire_transition(&self, i: usize) -> Range<usize> {
        assert!(1 <= i && i < self.num_states);
        let start: usize = self.wire_x_and_chunks().end
            + Self::WIRES_PER_STATE * (self.num_states)
            + Self::WIRES_PER_TRANSITION * (i - 1);
        start..start + Self::WIRES_PER_TRANSITION
    }

    pub(crate) const fn wire_chunks_left_inv(&self, i: usize) -> usize {
        self.wire_transition(i).start
    }

    pub(crate) const fn wire_number_accumulator_inv(&self, i: usize) -> usize {
        self.wire_transition(i).start + 1
    }

    pub(crate) const fn wire_degree_prev(&self) -> usize {
        0
    }

    pub(crate) const fn wire_degree(&self) -> usize {
        1
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for EvaluateBitstreamGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_states)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_states = src.read_usize()?;
        Ok(Self { num_states })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let const_1: <F as Extendable<D>>::Extension =
            F::Extension::from_basefield(F::from_canonical_u64(1));
        let field_chunk_max: <F as Extendable<D>>::Extension =
            F::Extension::from_basefield(F::from_canonical_usize(Self::CHUNK_MAX));

        let mut constraints = Vec::with_capacity((self.num_states - 1) * 16 + 1);
        let x = Quintuple::<F, D>::from_slice(&vars.local_wires[self.wire_x()]);

        let mut degree_acc = vars.local_wires[self.wire_degree_prev()];

        for i in 1..self.num_states {
            let sum_old = Quintuple::<F, D>::from_slice(&vars.local_wires[self.wire_sum(i - 1)]);
            let number_accumulator_old = vars.local_wires[self.wire_number_accumulator(i - 1)];
            let chunks_left_old = vars.local_wires[self.wire_chunks_left(i - 1)];
            let number_ending_old = vars.local_wires[self.wire_number_ending(i - 1)];
            let number_not_ending_old = vars.local_wires[self.wire_not_number_ending(i - 1)];

            let sum = Quintuple::<F, D>::from_slice(&vars.local_wires[self.wire_sum(i)]);
            let number_accumulator = vars.local_wires[self.wire_number_accumulator(i)];
            let chunks_left = vars.local_wires[self.wire_chunks_left(i)];
            let number_ending = vars.local_wires[self.wire_number_ending(i)];
            let number_not_ending = vars.local_wires[self.wire_not_number_ending(i)];

            let current_chunk = vars.local_wires[self.wire_chunk(i)];

            // Accumulate degree
            degree_acc += number_ending;

            //Constraints for number ending
            //equivalent to zero check of chunks_left
            let chunks_left_inv = vars.local_wires[self.wire_chunks_left_inv(i)];

            constraints.push((chunks_left * chunks_left_inv) - number_not_ending);
            constraints.push((number_not_ending * chunks_left) - chunks_left);
            constraints.push((const_1 - number_not_ending) - number_ending);
            //Constraints for chunks_left
            let expected_chunks_left = (number_ending_old * current_chunk)
                + (number_not_ending_old) * (chunks_left_old - const_1);
            constraints.push(expected_chunks_left - chunks_left);

            //Constraints for number_accumulator
            let expected_number_accumulator =
                number_not_ending_old * (number_accumulator_old * field_chunk_max + current_chunk);
            constraints.push(expected_number_accumulator - number_accumulator);
            let number_accumulator_inv = vars.local_wires[self.wire_number_accumulator_inv(i)];
            constraints.push(
                number_not_ending_old * (const_1 - number_accumulator * number_accumulator_inv),
            );
            //Constraints for sum
            let expected_sum = (((sum_old * x).add_scalar(number_accumulator))
                .scalar_mul(number_ending))
                + sum_old.scalar_mul(number_not_ending);
            let diff_sum = expected_sum - sum;
            constraints.extend(diff_sum.0);
        }

        let new_degree = vars.local_wires[self.wire_degree()];
        constraints.push(new_degree - degree_acc);

        constraints
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_1: ExtensionTarget<D> =
            builder.constant_extension(F::Extension::from_basefield(F::ONE));
        let field_chunk_max = builder.constant_extension(F::Extension::from_basefield(
            F::from_canonical_usize(Self::CHUNK_MAX),
        ));

        let mut constraints = Vec::with_capacity((self.num_states - 1) * 16 + 1);
        let x = QuintupleTarget::<D>::from_slice(&vars.local_wires[self.wire_x()]);

        let mut degree_acc = vars.local_wires[self.wire_degree_prev()];

        for i in 1..self.num_states {
            let sum_old = QuintupleTarget::<D>::from_slice(&vars.local_wires[self.wire_sum(i - 1)]);
            let number_accumulator_old = vars.local_wires[self.wire_number_accumulator(i - 1)];
            let chunks_left_old = vars.local_wires[self.wire_chunks_left(i - 1)];
            let number_ending_old = vars.local_wires[self.wire_number_ending(i - 1)];
            let number_not_ending_old = vars.local_wires[self.wire_not_number_ending(i - 1)];

            let sum = QuintupleTarget::<D>::from_slice(&vars.local_wires[self.wire_sum(i)]);
            let number_accumulator = vars.local_wires[self.wire_number_accumulator(i)];
            let chunks_left = vars.local_wires[self.wire_chunks_left(i)];
            let number_ending = vars.local_wires[self.wire_number_ending(i)];
            let number_not_ending = vars.local_wires[self.wire_not_number_ending(i)];

            let current_chunk = vars.local_wires[self.wire_chunk(i)];

            // Accumulate degree
            degree_acc = builder.add_extension(degree_acc, number_ending);

            //Constraints for number ending
            //equivalent to zero check of chunks_left
            let chunks_left_inv = vars.local_wires[self.wire_chunks_left_inv(i)];
            constraints.push(builder.mul_sub_extension(
                chunks_left,
                chunks_left_inv,
                number_not_ending,
            ));
            constraints.push(builder.mul_sub_extension(
                number_not_ending,
                chunks_left,
                chunks_left,
            ));
            constraints.push({
                let t = builder.sub_extension(const_1, number_not_ending);
                builder.sub_extension(t, number_ending)
            });

            //Constraints for chunks_left
            let expected_chunks_left = {
                let t1 = builder.sub_extension(chunks_left_old, const_1);
                let t2 = builder.mul_extension(number_not_ending_old, t1);
                builder.mul_add_extension(number_ending_old, current_chunk, t2)
            };
            constraints.push(builder.sub_extension(expected_chunks_left, chunks_left));

            //Constraints for number_accumulator
            let expected_number_accumulator = {
                let t = builder.mul_add_extension(
                    number_accumulator_old,
                    field_chunk_max,
                    current_chunk,
                );
                builder.mul_extension(number_not_ending_old, t)
            };
            constraints
                .push(builder.sub_extension(expected_number_accumulator, number_accumulator));
            let number_accumulator_inv = vars.local_wires[self.wire_number_accumulator_inv(i)];
            constraints.push({
                let t = builder.mul_extension(number_accumulator, number_accumulator_inv);
                let t = builder.sub_extension(const_1, t);
                builder.mul_extension(number_not_ending_old, t)
            });
            //Constraints for sum
            let expected_sum = {
                let t = mul_quintuple(builder, &sum_old, &x);
                let t2 = add_scalar(builder, &t, &number_accumulator);
                let t3 = mul_scalar_quintuple(builder, &t2, number_ending);
                let t4 = mul_scalar_quintuple(builder, &sum_old, number_not_ending);

                add_quintuple(builder, &t3, &t4)
            };
            let diff_sum = sub_quintuple(builder, &expected_sum, &sum);
            constraints.extend(diff_sum.0);
        }

        let new_degree = vars.local_wires[self.wire_degree()];
        constraints.push(builder.sub_extension(new_degree, degree_acc));

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            EvaluateBitstreamBaseGenerator {
                gate: self.clone(),
                row,
                _phantom: PhantomData,
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        Self::WIRES_CONSTANTS
            + Self::WIRES_PER_STATE
            + Self::DEGREE_WIRES
            + (self.num_states - 1) * Self::TOTAL_PER_OP
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        (self.num_states - 1) * 11 + 1
    }
}

#[derive(Clone, Debug, Default)]
pub struct EvaluateBitstreamBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: EvaluateBitstreamGate,
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for EvaluateBitstreamBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "EvaluateBitstreamBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let num_states = self.gate.num_states;

        self.gate
            .wire_state(0) // first state's full range
            .map(|w| Target::wire(self.row, w))
            .chain((self.gate.wire_x()).map(|i| Target::wire(self.row, i)))
            .chain((0..=1).map(|_| Target::wire(self.row, self.gate.wire_degree_prev())))
            .chain((1..num_states).map(|i| {
                let c = self.gate.wire_chunk(i); // <-- single-wire chunk accessor
                Target::wire(self.row, c)
            }))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let const_1 = F::from_canonical_u64(1);
        let field_chunk_max = F::from_canonical_usize(EvaluateBitstreamGate::CHUNK_MAX);

        let row = self.row;

        let get_quintuple = |range: std::ops::Range<usize>| -> QuintupleBase<F, D> {
            let limbs: [F; 5] =
                core::array::from_fn(|j| witness.get_target(Target::wire(row, range.start + j)));
            QuintupleBase::new(limbs)
        };

        let get_element = |wire: usize| -> F { witness.get_target(Target::wire(row, wire)) };

        let set_quintuple_at = |out: &mut GeneratedValues<F>,
                                range: Range<usize>,
                                q: &QuintupleBase<F, D>|
         -> anyhow::Result<()> {
            for (j, limb) in q.as_array().iter().copied().enumerate() {
                out.set_target(Target::wire(row, range.start + j), limb)?;
            }
            Ok(())
        };

        let set_element =
            |out: &mut GeneratedValues<F>, wire: usize, value: F| -> anyhow::Result<()> {
                out.set_target(Target::wire(row, wire), value)
            };

        let x = get_quintuple(self.gate.wire_x());
        let mut sum_old = get_quintuple(self.gate.wire_sum(0));
        let mut number_accumulator_old = get_element(self.gate.wire_number_accumulator(0));
        let mut chunks_left_old = get_element(self.gate.wire_chunks_left(0));
        let mut number_ending_old = get_element(self.gate.wire_number_ending(0));
        let mut number_not_ending_old = get_element(self.gate.wire_not_number_ending(0));

        // Add up "number_endings" together to figure out the degree of the polynomial. Don't count the first
        // slot regardless of its value as it's just transferred from the previous round.
        let mut degree = get_element(self.gate.wire_degree_prev());

        for i in 1..self.gate.num_states {
            let current_chunk = get_element(self.gate.wire_chunk(i));
            //chunks_left
            let chunks_left = (number_ending_old * current_chunk)
                + (number_not_ending_old) * (chunks_left_old - const_1);
            set_element(out_buffer, self.gate.wire_chunks_left(i), chunks_left).unwrap();

            let chunks_left_inv = if chunks_left != F::ZERO {
                (chunks_left).inverse()
            } else {
                F::ZERO
            };
            let number_not_ending = if chunks_left != F::ZERO {
                F::ONE
            } else {
                F::ZERO
            };
            let number_ending = const_1 - number_not_ending;

            set_element(
                out_buffer,
                self.gate.wire_chunks_left_inv(i),
                chunks_left_inv,
            )
            .unwrap();

            set_element(
                out_buffer,
                self.gate.wire_not_number_ending(i),
                number_not_ending,
            )
            .unwrap();
            set_element(out_buffer, self.gate.wire_number_ending(i), number_ending).unwrap();
            degree += number_ending;

            //number_accumulator
            let number_accumulator =
                number_not_ending_old * (number_accumulator_old * field_chunk_max + current_chunk);
            set_element(
                out_buffer,
                self.gate.wire_number_accumulator(i),
                number_accumulator,
            )
            .unwrap();
            let number_accumulator_inv = if number_accumulator != F::ZERO {
                number_accumulator.inverse()
            } else {
                F::ZERO
            };
            set_element(
                out_buffer,
                self.gate.wire_number_accumulator_inv(i),
                number_accumulator_inv,
            )
            .unwrap();

            //Constraints for sum
            let expected_sum = (((sum_old * x).add_scalar(number_accumulator))
                .scalar_mul(number_ending))
                + sum_old.scalar_mul(number_not_ending);
            set_quintuple_at(out_buffer, self.gate.wire_sum(i), &expected_sum).unwrap();

            sum_old = expected_sum;
            number_accumulator_old = number_accumulator;
            chunks_left_old = chunks_left;
            number_ending_old = number_ending;
            number_not_ending_old = number_not_ending;
        }

        // Set the new degree
        set_element(out_buffer, self.gate.wire_degree(), degree)?;

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = EvaluateBitstreamGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            _phantom: PhantomData,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BitstreamStateTarget {
    pub x: QuinticExtensionTarget,
    pub degree: Target,

    pub sum: QuinticExtensionTarget,
    pub number_accumulator: Target,
    pub chunks_left: Target,
    pub number_ending: BoolTarget,
    pub not_number_ending: BoolTarget,

    pub buffer_capacity: usize,
    pub buffer: Vec<Target>,
}

impl BitstreamStateTarget {
    pub fn new(config: &CircuitConfig) -> Self {
        Self {
            x: QuinticExtensionTarget::default(),

            sum: QuinticExtensionTarget::default(),
            number_accumulator: Target::default(),
            chunks_left: Target::default(),
            number_ending: BoolTarget::default(),
            not_number_ending: BoolTarget::default(),
            degree: Target::default(),

            buffer_capacity: BitstreamStateTarget::get_buffer_capacity(config),
            buffer: Vec::new(),
        }
    }
    pub fn get_buffer_capacity(config: &CircuitConfig) -> usize {
        EvaluateBitstreamGate::new_from_config(config).num_states - 1
    }
}
pub trait CircuitBuilderBitstreamEvaluator {
    fn bitstream_initialize(
        &mut self,
        sequence_id: usize,
        evaluation_point: QuinticExtensionTarget,
    );

    fn bitstream_digest_target(&mut self, sequence_id: usize, target: Target);
    fn bitstream_single_digest(&mut self, sequence_id: usize, target: Target);

    fn bitstream_export(&mut self, sequence_id: usize) -> BitstreamStateTarget;
    fn bitstream_import(&mut self, sequence_id: usize, start_state: BitstreamStateTarget);
}

impl<F, const D: usize> CircuitBuilderBitstreamEvaluator for Builder<F, D>
where
    F: RichField + Extendable<D> + Extendable<5>,
    Builder<F, D>: CircuitBuilderGFp5<F>,
{
    fn bitstream_import(&mut self, sequence_id: usize, start_state: BitstreamStateTarget) {
        self.bitstream_state.insert(sequence_id, start_state);
    }

    fn bitstream_initialize(
        &mut self,
        sequence_id: usize,
        evaluation_point: QuinticExtensionTarget,
    ) {
        let z = self.zero();
        let one = self.one();

        let new_state = BitstreamStateTarget {
            x: evaluation_point,
            degree: z,
            sum: QuinticExtensionTarget::new([z, z, z, z, z]),
            number_accumulator: z,
            chunks_left: z,
            number_ending: BoolTarget::new_unsafe(one),
            not_number_ending: BoolTarget::new_unsafe(z),
            buffer_capacity: BitstreamStateTarget::get_buffer_capacity(self.config()),
            buffer: Vec::new(),
        };

        self.bitstream_state.insert(sequence_id, new_state);
    }

    fn bitstream_digest_target(&mut self, sequence_id: usize, target: Target) {
        self.bitstream_state
            .get_mut(&sequence_id)
            .unwrap()
            .buffer
            .push(target);

        let state = self.bitstream_state[&sequence_id].clone();

        if state.buffer.len() == state.buffer_capacity {
            let gate = EvaluateBitstreamGate::new_from_config(self.config());
            let (row, _) = self.find_slot(gate.clone(), &[], &[]);

            let prev_degree = Target::wire(row, gate.wire_degree_prev());
            self.connect(state.degree, prev_degree);

            let degree = Target::wire(row, gate.wire_degree());
            self.bitstream_state.get_mut(&sequence_id).unwrap().degree = degree;

            for i in 0..state.buffer_capacity {
                self.connect(state.buffer[i], Target::wire(row, gate.wire_chunk(i + 1)));
            }
            let x_target = QuinticExtensionTarget::new(
                Target::wires_from_range(row, gate.wire_x())
                    .try_into()
                    .unwrap(),
            );
            self.connect_quintic_ext(state.x, x_target);

            let initial_sum = QuinticExtensionTarget::new(
                Target::wires_from_range(row, gate.wire_sum(0))
                    .try_into()
                    .unwrap(),
            );
            self.connect_quintic_ext(state.sum, initial_sum);

            let initial_number_accumulator = Target::wire(row, gate.wire_number_accumulator(0));
            self.connect(state.number_accumulator, initial_number_accumulator);

            let initial_chunks_left = Target::wire(row, gate.wire_chunks_left(0));
            self.connect(state.chunks_left, initial_chunks_left);

            let initial_number_ending = Target::wire(row, gate.wire_number_ending(0));
            self.connect(state.number_ending.target, initial_number_ending);

            let initial_not_number_ending = Target::wire(row, gate.wire_not_number_ending(0));
            self.connect(state.not_number_ending.target, initial_not_number_ending);

            let last_state_id = state.buffer_capacity;

            let final_sum = QuinticExtensionTarget::new(
                Target::wires_from_range(row, gate.wire_sum(last_state_id))
                    .try_into()
                    .unwrap(),
            );
            self.bitstream_state.get_mut(&sequence_id).unwrap().sum = final_sum;

            let final_number_accumulator =
                Target::wire(row, gate.wire_number_accumulator(last_state_id));
            self.bitstream_state
                .get_mut(&sequence_id)
                .unwrap()
                .number_accumulator = final_number_accumulator;

            let final_chunks_left = Target::wire(row, gate.wire_chunks_left(last_state_id));
            self.bitstream_state
                .get_mut(&sequence_id)
                .unwrap()
                .chunks_left = final_chunks_left;

            let final_number_ending = Target::wire(row, gate.wire_number_ending(last_state_id));
            self.bitstream_state
                .get_mut(&sequence_id)
                .unwrap()
                .number_ending
                .target = final_number_ending;

            let final_not_number_ending =
                Target::wire(row, gate.wire_not_number_ending(last_state_id));
            self.bitstream_state
                .get_mut(&sequence_id)
                .unwrap()
                .not_number_ending
                .target = final_not_number_ending;

            self.bitstream_state
                .get_mut(&sequence_id)
                .unwrap()
                .buffer
                .clear();
        }
    }

    // Intended only for exports
    fn bitstream_single_digest(&mut self, sequence_id: usize, current_chunk: Target) {
        assert!(self.bitstream_state[&sequence_id].buffer.is_empty());
        let one = self.one();
        let zero = self.zero();
        let field_chunk_max = self.constant_usize(EvaluateBitstreamGate::CHUNK_MAX);

        let old_state = self.bitstream_state[&sequence_id].clone();

        self.bitstream_state
            .get_mut(&sequence_id)
            .unwrap()
            .chunks_left = {
            let dec_chunks_left = self.sub(old_state.chunks_left, one);
            self.select(old_state.number_ending, current_chunk, dec_chunks_left)
        };

        self.bitstream_state
            .get_mut(&sequence_id)
            .unwrap()
            .number_ending = self.is_equal(self.bitstream_state[&sequence_id].chunks_left, zero);

        self.bitstream_state
            .get_mut(&sequence_id)
            .unwrap()
            .not_number_ending = self.not(self.bitstream_state[&sequence_id].number_ending);

        self.bitstream_state
            .get_mut(&sequence_id)
            .unwrap()
            .number_accumulator = {
            let t = self.mul_add(old_state.number_accumulator, field_chunk_max, current_chunk);
            self.mul(old_state.not_number_ending.target, t)
        };
        {
            let is_zero =
                self.is_equal(self.bitstream_state[&sequence_id].number_accumulator, zero);
            let t = self.mul(old_state.not_number_ending.target, is_zero.target);
            self.assert_zero(t);
        }

        self.bitstream_state.get_mut(&sequence_id).unwrap().degree = self.add(
            self.bitstream_state[&sequence_id].degree,
            self.bitstream_state[&sequence_id].number_ending.target,
        );

        let number_extended = QuinticExtensionTarget::new([
            self.bitstream_state[&sequence_id].number_accumulator,
            zero,
            zero,
            zero,
            zero,
        ]);
        let new_sum = {
            let new_sum = self.mul_quintic_ext(old_state.sum, old_state.x);
            self.add_quintic_ext(new_sum, number_extended)
        };
        self.bitstream_state.get_mut(&sequence_id).unwrap().sum = self.select_quintic_ext(
            self.bitstream_state[&sequence_id].number_ending,
            new_sum,
            old_state.sum,
        );
    }

    fn bitstream_export(&mut self, sequence_id: usize) -> BitstreamStateTarget {
        let old_state = self.bitstream_state.get_mut(&sequence_id).unwrap().clone();

        if !old_state.buffer.is_empty() {
            self.bitstream_state
                .get_mut(&sequence_id)
                .unwrap()
                .buffer
                .clear();
            for i in 0..old_state.buffer.len() {
                self.bitstream_single_digest(sequence_id, old_state.buffer[i]);
            }
        }
        self.bitstream_state[&sequence_id].clone()
    }
}

pub fn number_to_chunks(mut number: u64) -> Vec<u64> {
    let mut digits = Vec::new();
    let chunk_max = EvaluateBitstreamGate::CHUNK_MAX as u64;
    while number > 0 {
        digits.push(number % chunk_max);
        number /= chunk_max;
    }
    digits.reverse();

    let mut chunks_with_len = Vec::with_capacity(digits.len() + 1);
    chunks_with_len.push(digits.len() as u64);
    chunks_with_len.extend(digits);

    chunks_with_len
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::target::{BoolTarget, Target};
    use plonky2::iop::witness::PartialWitness;
    use rand::Rng;

    use crate::blob::evaluate_bitstream::{
        CircuitBuilderBitstreamEvaluator, EvaluateBitstreamGate, number_to_chunks,
    };
    use crate::builder::Builder;
    use crate::delta::evaluate_sequence::CircuitBuilderSequenceEvaluator;
    use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
    use crate::plonky2::field::goldilocks_field::GoldilocksField;
    use crate::plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonky2::plonk::circuit_data::CircuitConfig;
    use crate::plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::types::config::CIRCUIT_CONFIG;

    #[test]
    fn test_sequence_to_bitstream() -> Result<()> {
        // let _ = env_logger::try_init_from_env(
        //     env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"),
        // );

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut rng = rand::thread_rng();

        for _ in 0..5 {
            let pw = PartialWitness::new();
            let mut builder = Builder::<F, D>::new(CIRCUIT_CONFIG);

            let n = rng.gen_range(500..=2500);

            let mask: u64 = (1u64 << 60) - 1;

            let values: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() & mask).collect();

            let mut chunked_values: Vec<Target> = Vec::new();
            let mut chunked_values_f = vec![];
            for &w in &values {
                chunked_values_f.push(number_to_chunks(w));
                for c in number_to_chunks(w) {
                    chunked_values.push(builder.constant(F::from_canonical_u64(c)));
                }
            }

            let mut values_targets: Vec<(Target, BoolTarget)> = Vec::new();
            for &w in &values {
                let real_t = builder.constant(F::from_canonical_u64(w));
                let real_b = builder.constant_bool(true);

                values_targets.push((real_t, real_b));

                if rng.gen_bool(0.5) {
                    let dummy_val = rng.r#gen::<u64>() & mask;
                    let dummy_t = builder.constant(F::from_canonical_u64(dummy_val));
                    let dummy_b = builder.constant_bool(false);
                    values_targets.push((dummy_t, dummy_b));
                }
            }

            let evaluation_point = QuinticExtensionTarget::new([
                builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
                builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
                builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
                builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
                builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            ]);

            builder.bitstream_initialize(0, evaluation_point);
            builder.sequence_initialize(0, evaluation_point);

            for chunk in chunked_values {
                builder.bitstream_digest_target(0, chunk);
            }

            let mut degree = builder.zero();
            for (value, selector) in values_targets {
                builder.sequence_digest_target(0, value, selector);
                degree = builder.add(degree, selector.target);
            }

            let aggregate_bitstream = builder.bitstream_export(0);
            let aggregate_sequence = builder.sequence_export(0).sum;

            builder.connect_quintic_ext(aggregate_bitstream.sum, aggregate_sequence);
            builder.connect(aggregate_bitstream.degree, degree);

            let data = builder.build::<C>();
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        }

        Ok(())
    }

    #[test]
    fn low_degree() {
        let gate =
            EvaluateBitstreamGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate: EvaluateBitstreamGate =
            EvaluateBitstreamGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn test_gate_equivalence() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_ecc_config();
        let pw = PartialWitness::new();
        let n = 10;
        let mut builder = Builder::<F, D>::new(config.clone());

        let mut rng = rand::thread_rng();
        let mask: u64 = (1u64 << 60) - 1;

        let values: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() & mask).collect();

        let mut chunked_values: Vec<Target> = Vec::new();
        for &w in &values {
            for c in number_to_chunks(w) {
                chunked_values.push(builder.constant(F::from_canonical_u64(c)));
            }
        }

        let evaluation_point = QuinticExtensionTarget::new([
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
        ]);

        builder.bitstream_initialize(0, evaluation_point);
        builder.bitstream_initialize(1, evaluation_point);

        for chunk in chunked_values {
            builder.bitstream_digest_target(0, chunk);
            builder.bitstream_single_digest(1, chunk);
        }

        let aggregate0 = builder.bitstream_export(0).sum;
        let aggregate1 = builder.bitstream_export(1).sum;
        builder.connect_quintic_ext(aggregate0, aggregate1);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
