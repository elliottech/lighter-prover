extern crate alloc;
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate specialized for additions
#[derive(Debug, Clone)]
pub struct AdditionGate {
    /// Number of additions operations performed by an addition gate.
    pub num_ops: usize,
}

impl AdditionGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 3;
        config.num_routed_wires / wires_per_op
    }

    pub(crate) const fn wire_ith_addend_0(i: usize) -> usize {
        3 * i
    }
    pub(crate) const fn wire_ith_addend_1(i: usize) -> usize {
        3 * i + 1
    }
    pub(crate) const fn wire_ith_output(i: usize) -> usize {
        3 * i + 2
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for AdditionGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let addend_0 = vars.local_wires[Self::wire_ith_addend_0(i)];
            let addend_1 = vars.local_wires[Self::wire_ith_addend_1(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = addend_0 * const_0 + addend_1 * const_1;

            constraints.push(output - computed_output);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        self.eval_unfiltered_base_batch_accumulate_packed(
            vars_base,
            filters,
            combined_gate_constraints,
        );
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let addend_0 = vars.local_wires[Self::wire_ith_addend_0(i)];
            let addend_1 = vars.local_wires[Self::wire_ith_addend_1(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];

            let true_addend_0 = builder.mul_extension(const_0, addend_0);
            let true_addend_1 = builder.mul_extension(const_1, addend_1);

            let computed_output = builder.add_extension(true_addend_0, true_addend_1);

            let diff = builder.sub_extension(output, computed_output);
            constraints.push(diff);
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    AdditionBaseGenerator {
                        row,
                        const_0: local_constants[0],
                        const_1: local_constants[1],
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 3
    }

    fn num_constants(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for AdditionGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        for i in 0..self.num_ops {
            let addend_0 = vars.local_wires[Self::wire_ith_addend_0(i)];
            let addend_1 = vars.local_wires[Self::wire_ith_addend_1(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = addend_0 * const_0 + addend_1 * const_1;

            yield_constr.one(output - computed_output);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdditionBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    const_1: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for AdditionBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "AdditionBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            AdditionGate::wire_ith_addend_0(self.i),
            AdditionGate::wire_ith_addend_1(self.i),
        ]
        .iter()
        .map(|&i| Target::wire(self.row, i))
        .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let addend_0 = get_wire(AdditionGate::wire_ith_addend_0(self.i));
        let addend_1 = get_wire(AdditionGate::wire_ith_addend_1(self.i));

        let output_target = Target::wire(self.row, AdditionGate::wire_ith_output(self.i));

        let computed_output = addend_0 * self.const_0 + addend_1 * self.const_1;

        out_buffer.set_target(output_target, computed_output)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_field(self.const_1)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let const_1 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            const_0,
            const_1,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::addition_base::AdditionGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate = AdditionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = AdditionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }
}
