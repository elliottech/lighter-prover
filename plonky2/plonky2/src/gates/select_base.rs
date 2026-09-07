#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};
use core::marker::PhantomData;
extern crate alloc;
use alloc::string::ToString;

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

/// A gate specialized for Selection operations
#[derive(Debug, Clone, Default)]
pub struct SelectionGate {
    /// Number of additions operations performed by a Selection Gate.
    pub num_ops: usize,
}

impl SelectionGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }
    //Number of routed wires necessary for an operation
    const ROUTED_PER_OP: usize = 4;
    const NOT_ROUTED_PER_OP: usize = 1;
    const TOTAL_PER_OP: usize = Self::ROUTED_PER_OP + Self::NOT_ROUTED_PER_OP;
    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let routed_packed_count = config.num_routed_wires / Self::ROUTED_PER_OP;
        let unrouted_packed_count = config.num_wires / Self::TOTAL_PER_OP;
        if routed_packed_count < unrouted_packed_count {
            routed_packed_count
        } else {
            unrouted_packed_count
        }
    }

    pub(crate) const fn wire_ith_selector(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i
    }
    pub(crate) const fn wire_ith_element_0(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 1
    }
    pub(crate) const fn wire_ith_element_1(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 2
    }
    pub(crate) const fn wire_ith_output(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 3
    }

    pub(crate) const fn wire_ith_temporary(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < Self::NOT_ROUTED_PER_OP);
        Self::ROUTED_PER_OP * self.num_ops + i * Self::NOT_ROUTED_PER_OP + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for SelectionGate {
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
        let mut constraints = Vec::with_capacity(self.num_ops * 2);

        for i in 0..self.num_ops {
            let b = vars.local_wires[self.wire_ith_selector(i)];
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let result = vars.local_wires[self.wire_ith_output(i)];
            let temp = vars.local_wires[self.wire_ith_temporary(i, 0)];
            constraints.push((b * y - y) - temp);
            constraints.push((b * x - temp) - result);
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
        let mut constraints = Vec::with_capacity(self.num_ops * 4);

        for i in 0..self.num_ops {
            let b = vars.local_wires[self.wire_ith_selector(i)];
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let result = vars.local_wires[self.wire_ith_output(i)];
            let temp = vars.local_wires[self.wire_ith_temporary(i, 0)];

            constraints.push({
                let inner = builder.mul_sub_extension(b, y, y);
                builder.sub_extension(inner, temp)
            });
            constraints.push({
                let inner = builder.mul_sub_extension(b, x, temp);
                builder.sub_extension(inner, result)
            });
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let result: Vec<WitnessGeneratorRef<F, D>> = (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    SelectionBaseGenerator {
                        gate: self.clone(),
                        row,
                        i,
                        _phantom: PhantomData,
                    }
                    .adapter(),
                )
            })
            .collect();
        result
    }

    fn num_wires(&self) -> usize {
        self.num_ops * Self::TOTAL_PER_OP
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 2
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for SelectionGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for i in 0..self.num_ops {
            let b = vars.local_wires[self.wire_ith_selector(i)];
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let result = vars.local_wires[self.wire_ith_output(i)];
            let temp = vars.local_wires[self.wire_ith_temporary(i, 0)];

            yield_constr.one((b * y - y) - temp);
            yield_constr.one((b * x - temp) - result);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SelectionBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub gate: SelectionGate,
    pub row: usize,
    pub i: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SelectionBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "SelectionBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            self.gate.wire_ith_selector(self.i),
            self.gate.wire_ith_element_0(self.i),
            self.gate.wire_ith_element_1(self.i),
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

        let b = get_wire(self.gate.wire_ith_selector(self.i));
        let x = get_wire(self.gate.wire_ith_element_0(self.i));
        let y = get_wire(self.gate.wire_ith_element_1(self.i));
        let result = Target::wire(self.row, self.gate.wire_ith_output(self.i));
        let temp = Target::wire(self.row, self.gate.wire_ith_temporary(self.i, 0));

        let temp_value = b * y - y;
        let result_value = b * x - temp_value;

        out_buffer.set_target(temp, temp_value)?;
        out_buffer.set_target(result, result_value)?;

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = SelectionGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            i,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    #[allow(unused_imports)]
    use crate::field::types::Field64;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::select_base::SelectionGate;
    use crate::iop::target::Target;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate = SelectionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = SelectionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn test_selection_gate_success() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let b = builder.add_virtual_bool_target_safe();
        let x = builder.add_virtual_target();
        let y = builder.add_virtual_target();

        let gate = SelectionGate::new_from_config(&config);
        let ref_gate = gate.clone();
        let (row, i) = builder.find_slot(gate, &[], &[]);

        builder.connect(b.target, Target::wire(row, ref_gate.wire_ith_selector(i)));
        builder.connect(x, Target::wire(row, ref_gate.wire_ith_element_0(i)));
        builder.connect(y, Target::wire(row, ref_gate.wire_ith_element_1(i)));

        let output = Target::wire(row, ref_gate.wire_ith_output(i));
        let result = builder.add_virtual_target();
        builder.connect(result, output);

        let circuit_data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_bool_target(b, true)?;
        pw.set_target(x, F::from_canonical_u64(123))?;
        pw.set_target(y, F::from_canonical_u64(456))?;

        let proof = circuit_data.prove(pw)?;
        circuit_data.verify(proof)?;
        Ok(())
    }
}
