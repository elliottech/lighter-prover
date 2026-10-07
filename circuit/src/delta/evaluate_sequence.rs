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

/// Performs polynomial evaluation. Initialized with a given point, digests new targets into
/// a buffer, and evaluates a new degree of the polynomial as the buffer is filled.
#[derive(Debug, Clone, Default)]
pub struct EvaluateSequenceGate {
    /// Number of Bitstream evaluation operations that can be performed using a single gate
    pub num_states: usize,
}

impl EvaluateSequenceGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_states: Self::num_states(config),
        }
    }

    const WIRES_CONSTANTS: usize = 5;
    const WIRES_PER_STATE: usize = 5;
    //Number of wires representing a transition between states
    const WIRES_PER_TRANSITION: usize = 0;

    //one extra for field element and selector
    const TOTAL_PER_OP: usize = Self::WIRES_PER_STATE + Self::WIRES_PER_TRANSITION + 2;

    /// Determine the maximum number of states that can fit in one gate for the given config.
    pub(crate) const fn num_states(config: &CircuitConfig) -> usize {
        let unrouted_packed_count =
            (config.num_wires - Self::WIRES_CONSTANTS - Self::WIRES_PER_STATE) / Self::TOTAL_PER_OP;
        unrouted_packed_count + 1
    }

    //`x` is shared between all operations
    pub(crate) const fn wire_x_and_elements(&self) -> Range<usize> {
        let start = 0;
        start..5 + (self.num_states - 1) * 2
    }

    pub(crate) const fn wire_x(&self) -> Range<usize> {
        let start = self.wire_x_and_elements().start;
        start..start + 5
    }

    pub(crate) const fn wire_element(&self, i: usize) -> usize {
        assert!(1 <= i && i < self.num_states);
        let start = self.wire_x_and_elements().start + 5;
        start + i - 1
    }

    pub(crate) const fn wire_selector(&self, i: usize) -> usize {
        assert!(1 <= i && i < self.num_states);
        let start = self.wire_element(self.num_states - 1);
        start + i
    }

    //state getters
    pub(crate) const fn wire_state(&self, i: usize) -> Range<usize> {
        assert!(i < self.num_states);
        let start: usize = self.wire_x_and_elements().end;

        //Places the first state and the last state in the routed wires area so they can be interacted with
        if i == 0 {
            start..start + Self::WIRES_PER_STATE
        } else if i == self.num_states - 1 {
            start + Self::WIRES_PER_STATE..start + 2 * Self::WIRES_PER_STATE
        } else {
            let start = self.wire_x_and_elements().end + Self::WIRES_PER_STATE * (i + 1);
            start..start + Self::WIRES_PER_STATE
        }
    }

    pub(crate) const fn wire_sum(&self, i: usize) -> Range<usize> {
        let start_of_state: usize = self.wire_state(i).start;
        start_of_state..start_of_state + 5
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for EvaluateSequenceGate {
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
        let mut constraints = Vec::with_capacity((self.num_states - 1) * 5);
        let x = Quintuple::<F, D>::from_slice(&vars.local_wires[self.wire_x()]);

        for i in 1..self.num_states {
            let sum_old = Quintuple::<F, D>::from_slice(&vars.local_wires[self.wire_sum(i - 1)]);
            let sum = Quintuple::<F, D>::from_slice(&vars.local_wires[self.wire_sum(i)]);
            let current_element = vars.local_wires[self.wire_element(i)];
            let selector = vars.local_wires[self.wire_selector(i)];
            //Constraints for sum
            let expected_sum = (sum_old * x)
                .add_scalar(current_element)
                .scalar_mul(selector)
                + sum_old.scalar_mul(const_1 - selector);
            let diff_sum = expected_sum - sum;

            constraints.extend(diff_sum.0);
        }
        constraints
    }
    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_1: ExtensionTarget<D> =
            builder.constant_extension(F::Extension::from_basefield(F::ONE));
        let mut constraints = Vec::with_capacity((self.num_states - 1) * 16);
        let x = QuintupleTarget::<D>::from_slice(&vars.local_wires[self.wire_x()]);

        for i in 1..self.num_states {
            let sum_old = QuintupleTarget::<D>::from_slice(&vars.local_wires[self.wire_sum(i - 1)]);
            let sum = QuintupleTarget::<D>::from_slice(&vars.local_wires[self.wire_sum(i)]);
            let current_element = vars.local_wires[self.wire_element(i)];
            let selector = vars.local_wires[self.wire_selector(i)];

            //Constraints for sum
            let expected_sum = {
                let t = mul_quintuple(builder, &sum_old, &x);
                let t2 = add_scalar(builder, &t, &current_element);
                let t3 = mul_scalar_quintuple(builder, &t2, selector);
                let not_selector = builder.sub_extension(const_1, selector);
                let t4 = mul_scalar_quintuple(builder, &sum_old, not_selector);

                add_quintuple(builder, &t3, &t4)
            };
            let diff_sum = sub_quintuple(builder, &expected_sum, &sum);
            constraints.extend(diff_sum.0);
        }
        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            EvaluateSequenceBaseGenerator {
                gate: self.clone(),
                row,
                _phantom: PhantomData,
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        Self::WIRES_CONSTANTS + Self::WIRES_PER_STATE + (self.num_states - 1) * Self::TOTAL_PER_OP
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        (self.num_states - 1) * 5
    }
}

#[derive(Clone, Debug, Default)]
pub struct EvaluateSequenceBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: EvaluateSequenceGate,
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for EvaluateSequenceBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "EvaluateSequenceBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let num_states = self.gate.num_states;

        self.gate
            .wire_state(0) // first state's full range
            .map(|w| Target::wire(self.row, w))
            .chain((self.gate.wire_x()).map(|i| Target::wire(self.row, i)))
            .chain((1..num_states).map(|i| {
                let c = self.gate.wire_element(i); // <-- single-wire element access
                Target::wire(self.row, c)
            }))
            .chain((1..num_states).map(|i| {
                let c = self.gate.wire_selector(i); // <-- single-wire element access
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
        let x = get_quintuple(self.gate.wire_x());

        let mut sum_old = get_quintuple(self.gate.wire_sum(0));
        for i in 1..self.gate.num_states {
            let current_element = get_element(self.gate.wire_element(i));
            let selector = get_element(self.gate.wire_selector(i));
            //Constraints for sum
            let expected_sum = (sum_old * x)
                .add_scalar(current_element)
                .scalar_mul(selector)
                + sum_old.scalar_mul(const_1 - selector);
            set_quintuple_at(out_buffer, self.gate.wire_sum(i), &expected_sum).unwrap();
            sum_old = expected_sum;
        }
        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = EvaluateSequenceGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            _phantom: PhantomData,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SequenceStateTarget {
    pub x: QuinticExtensionTarget,
    pub sum: QuinticExtensionTarget,

    pub buffer_capacity: usize,
    pub buffer: Vec<(Target, BoolTarget)>,
}

impl SequenceStateTarget {
    pub fn new(config: &CircuitConfig) -> Self {
        Self {
            x: QuinticExtensionTarget::default(),
            sum: QuinticExtensionTarget::default(),
            buffer_capacity: SequenceStateTarget::get_buffer_capacity(config),
            buffer: Vec::new(),
        }
    }
    pub fn get_buffer_capacity(config: &CircuitConfig) -> usize {
        EvaluateSequenceGate::new_from_config(config).num_states - 1
    }
}
pub trait CircuitBuilderSequenceEvaluator {
    fn sequence_initialize(&mut self, sequence_id: usize, evaluation_point: QuinticExtensionTarget);

    fn sequence_digest_target(&mut self, sequence_id: usize, target: Target, selector: BoolTarget);
    fn sequence_single_digest(&mut self, sequence_id: usize, target: Target, selector: BoolTarget);

    fn sequence_export(&mut self, sequence_id: usize) -> SequenceStateTarget;
    fn sequence_import(&mut self, sequence_id: usize, start_state: SequenceStateTarget);
}

impl<F, const D: usize> CircuitBuilderSequenceEvaluator for Builder<F, D>
where
    F: RichField + Extendable<D> + Extendable<5>,
    Builder<F, D>: CircuitBuilderGFp5<F>,
{
    fn sequence_import(&mut self, sequence_id: usize, start_state: SequenceStateTarget) {
        self.sequence_state.insert(sequence_id, start_state);
    }

    fn sequence_initialize(
        &mut self,
        sequence_id: usize,
        evaluation_point: QuinticExtensionTarget,
    ) {
        let z = self.zero();
        let new_state = SequenceStateTarget {
            x: evaluation_point,
            sum: QuinticExtensionTarget::new([z, z, z, z, z]),
            buffer_capacity: SequenceStateTarget::get_buffer_capacity(self.config()),
            buffer: Vec::new(),
        };

        self.sequence_state.insert(sequence_id, new_state);
    }

    fn sequence_digest_target(&mut self, sequence_id: usize, target: Target, selector: BoolTarget) {
        self.sequence_state
            .get_mut(&sequence_id)
            .unwrap()
            .buffer
            .push((target, selector));

        let state = self.sequence_state[&sequence_id].clone();

        if state.buffer.len() == state.buffer_capacity {
            let gate = EvaluateSequenceGate::new_from_config(self.config());
            let (row, _) = self.find_slot(gate.clone(), &[], &[]);

            for i in 0..state.buffer_capacity {
                self.connect(
                    state.buffer[i].0,
                    Target::wire(row, gate.wire_element(i + 1)),
                );
                self.connect(
                    state.buffer[i].1.target,
                    Target::wire(row, gate.wire_selector(i + 1)),
                );
            }
            let x_target = QuinticExtensionTarget::new(
                Target::wires_from_range(row, gate.wire_x())
                    .try_into()
                    .unwrap(),
            );
            self.connect_quintic_ext(state.x, x_target);
            let initial_state = QuinticExtensionTarget::new(
                Target::wires_from_range(row, gate.wire_sum(0))
                    .try_into()
                    .unwrap(),
            );
            self.connect_quintic_ext(state.sum, initial_state);
            let final_state = QuinticExtensionTarget::new(
                Target::wires_from_range(row, gate.wire_sum(state.buffer_capacity))
                    .try_into()
                    .unwrap(),
            );

            self.sequence_state.get_mut(&sequence_id).unwrap().sum = final_state;
            self.sequence_state
                .get_mut(&sequence_id)
                .unwrap()
                .buffer
                .clear();
        }
    }

    fn sequence_single_digest(&mut self, sequence_id: usize, target: Target, selector: BoolTarget) {
        assert!(self.sequence_state[&sequence_id].buffer.is_empty());

        let mut new_sum: QuinticExtensionTarget;
        let old_state = self.sequence_state[&sequence_id].clone();
        new_sum = self.mul_quintic_ext(old_state.sum, old_state.x);

        let target_extended = QuinticExtensionTarget::new([
            target,
            self.zero(),
            self.zero(),
            self.zero(),
            self.zero(),
        ]);
        new_sum = self.add_quintic_ext(new_sum, target_extended);

        self.sequence_state.get_mut(&sequence_id).unwrap().sum =
            self.select_quintic_ext(selector, new_sum, old_state.sum);
    }

    fn sequence_export(&mut self, sequence_id: usize) -> SequenceStateTarget {
        let old_state = self.sequence_state[&sequence_id].clone();
        if !old_state.buffer.is_empty() {
            //Could in theory just `sequence_digest_target` a couple of targets with selector set to 0
            //but adding a couple of independent elements is consistent with BitstreamEvaluation
            //and also doesn't add many constraints
            self.sequence_state
                .get_mut(&sequence_id)
                .unwrap()
                .buffer
                .clear();
            for i in 0..old_state.buffer.len() {
                self.sequence_single_digest(
                    sequence_id,
                    old_state.buffer[i].0,
                    old_state.buffer[i].1,
                );
            }
        }
        self.sequence_state.get_mut(&sequence_id).unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::target::{BoolTarget, Target};
    use plonky2::iop::witness::PartialWitness;
    use rand::Rng;

    use crate::builder::Builder;
    use crate::delta::evaluate_sequence::{CircuitBuilderSequenceEvaluator, EvaluateSequenceGate};
    use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
    use crate::plonky2::field::goldilocks_field::GoldilocksField;
    use crate::plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonky2::plonk::circuit_data::CircuitConfig;
    use crate::plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate =
            EvaluateSequenceGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate: EvaluateSequenceGate =
            EvaluateSequenceGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn test_gate_equivalence() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_ecc_config();
        let pw = PartialWitness::new();
        let n = 200;
        let mut builder = Builder::<F, D>::new(config);

        let mut rng = rand::thread_rng();
        let mask: u64 = (1u64 << 60) - 1;

        let values: Vec<Target> = (0..n)
            .map(|_| builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)))
            .collect();
        let selectors: Vec<BoolTarget> = (0..n)
            .map(|_| {
                let bit: bool = rng.r#gen::<bool>();
                builder.constant_bool(bit)
            })
            .collect();

        let evaluation_point = QuinticExtensionTarget::new([
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
            builder.constant(F::from_canonical_u64(rng.r#gen::<u64>() & mask)),
        ]);

        builder.sequence_initialize(0, evaluation_point);
        builder.sequence_initialize(1, evaluation_point);

        for (value, selector) in values.into_iter().zip(selectors) {
            builder.sequence_single_digest(0, value, selector);
            builder.sequence_digest_target(1, value, selector);
        }

        let aggregate0 = builder.sequence_export(0).sum;
        let aggregate1 = builder.sequence_export(1).sum;
        builder.connect_quintic_ext(aggregate0, aggregate1);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
