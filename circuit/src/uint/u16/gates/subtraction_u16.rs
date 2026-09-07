// Portions of this file are derived from plonky2-crypto
// Copyright (c) 2023 Jump Crypto Services LLC.
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Originally from: https://github.com/JumpCrypto/plonky2-crypto/blob/main/src/u32/gates/subtraction_u32.rs
// at 5a743ced38a2b66ecd3e6945b2b7fa468324ea73

// Modifications copyright (c) 2025 Elliot Technologies, Inc.
// This file has been modified from its original version.

use core::marker::PhantomData;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::gates::gate::Gate;
use plonky2::gates::packed_util::PackedEvaluableBase;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::wire::Wire;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate to perform a subtraction on 32-bit limbs: given `x`, `y`, and `borrow`, it returns
/// the result `x - y - borrow` and, if this underflows, a new `borrow`. Inputs are not range-checked.
#[derive(Copy, Clone, Debug, Default)]
pub struct U16SubtractionGate<F: RichField + Extendable<D>, const D: usize> {
    pub num_ops: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> U16SubtractionGate<F, D> {
    pub fn new_from_config(config: &CircuitConfig) -> Self {
        let wires_per_op = 5 + Self::num_limbs();
        let routed_wires_per_op = 5;
        let num_ops =
            (config.num_wires / wires_per_op).min(config.num_routed_wires / routed_wires_per_op);

        Self {
            num_ops,
            _phantom: PhantomData,
        }
    }

    pub fn wire_ith_input_x(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        5 * i
    }
    pub fn wire_ith_input_y(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        5 * i + 1
    }
    pub fn wire_ith_input_borrow(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        5 * i + 2
    }

    pub fn wire_ith_output_result(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        5 * i + 3
    }
    pub fn wire_ith_output_borrow(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        5 * i + 4
    }

    pub fn limb_bits() -> usize {
        2
    }

    // We have limbs for the 16 bits of `output_result`.
    pub fn num_limbs() -> usize {
        16 / Self::limb_bits()
    }

    pub fn wire_ith_output_jth_limb(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.num_ops);
        debug_assert!(j < Self::num_limbs());
        5 * self.num_ops + Self::num_limbs() * i + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for U16SubtractionGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self {
            num_ops,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());
        for i in 0..self.num_ops {
            let input_x = vars.local_wires[self.wire_ith_input_x(i)];
            let input_y = vars.local_wires[self.wire_ith_input_y(i)];
            let input_borrow = vars.local_wires[self.wire_ith_input_borrow(i)];

            let result_initial = input_x - input_y - input_borrow;
            let base = F::Extension::from_canonical_u64(1 << 16u64);

            let output_result = vars.local_wires[self.wire_ith_output_result(i)];
            let output_borrow = vars.local_wires[self.wire_ith_output_borrow(i)];

            constraints.push(output_result - (result_initial + base * output_borrow));

            // Range-check output_result to be at most 16 bits.
            let mut combined_limbs = F::Extension::ZERO;
            let limb_base = F::Extension::from_canonical_u64(1u64 << Self::limb_bits());
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let product = (0..max_limb)
                    .map(|x| this_limb - F::Extension::from_canonical_usize(x))
                    .product();
                constraints.push(product);

                combined_limbs = limb_base * combined_limbs + this_limb;
            }
            constraints.push(combined_limbs - output_result);

            // Range-check output_borrow to be one bit.
            constraints.push(output_borrow * (F::Extension::ONE - output_borrow));
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
        assert_eq!(1 << Self::limb_bits(), 4);
        let n = vars_base.len();
        let wires = vars_base.local_wires;
        let three = F::from_canonical_usize(3);
        let limb_base = F::from_canonical_u64(1u64 << Self::limb_bits());
        let base = F::from_canonical_u64(1 << 16u64);
        let mut res = vec![F::ZERO; n * <Self as Gate<F, D>>::num_constraints(self)];
        let mut chunks = res.chunks_exact_mut(n);
        let mut combined_limbs = vec![F::ZERO; n];

        for i in 0..self.num_ops {
            let input_x = &wires[self.wire_ith_input_x(i) * n..][..n];
            let input_y = &wires[self.wire_ith_input_y(i) * n..][..n];
            let input_borrow = &wires[self.wire_ith_input_borrow(i) * n..][..n];
            let output_result = &wires[self.wire_ith_output_result(i) * n..][..n];
            let output_borrow = &wires[self.wire_ith_output_borrow(i) * n..][..n];

            let out = chunks.next().unwrap();
            for p in 0..n {
                let result_initial = input_x[p] - input_y[p] - input_borrow[p];
                out[p] = output_result[p] - (result_initial + base * output_borrow[p]);
            }

            // Limb range products (base-4: x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3))
            // in the same descending order as `eval_unfiltered`, accumulating
            // the recomposition along the way.
            combined_limbs.fill(F::ZERO);
            for j in (0..Self::num_limbs()).rev() {
                let limb = &wires[self.wire_ith_output_jth_limb(i, j) * n..][..n];
                let out = chunks.next().unwrap();
                for p in 0..n {
                    let x = limb[p];
                    let y = x * (x - three);
                    out[p] = y * (y + F::TWO);
                }
                for p in 0..n {
                    combined_limbs[p] = combined_limbs[p] * limb_base + limb[p];
                }
            }
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = combined_limbs[p] - output_result[p];
            }

            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = output_borrow[p] * (F::ONE - output_borrow[p]);
            }
        }
        assert!(chunks.next().is_none());
        res
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());
        for i in 0..self.num_ops {
            let input_x = vars.local_wires[self.wire_ith_input_x(i)];
            let input_y = vars.local_wires[self.wire_ith_input_y(i)];
            let input_borrow = vars.local_wires[self.wire_ith_input_borrow(i)];

            let diff = builder.sub_extension(input_x, input_y);
            let result_initial = builder.sub_extension(diff, input_borrow);
            let base = builder.constant_extension(F::Extension::from_canonical_u64(1 << 16u64));

            let output_result = vars.local_wires[self.wire_ith_output_result(i)];
            let output_borrow = vars.local_wires[self.wire_ith_output_borrow(i)];

            let computed_output = builder.mul_add_extension(base, output_borrow, result_initial);
            constraints.push(builder.sub_extension(output_result, computed_output));

            // Range-check output_result to be at most 16 bits.
            let mut combined_limbs = builder.zero_extension();
            let limb_base = builder
                .constant_extension(F::Extension::from_canonical_u64(1u64 << Self::limb_bits()));
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let mut product = builder.one_extension();
                for x in 0..max_limb {
                    let x_target =
                        builder.constant_extension(F::Extension::from_canonical_usize(x));
                    let diff = builder.sub_extension(this_limb, x_target);
                    product = builder.mul_extension(product, diff);
                }
                constraints.push(product);

                combined_limbs = builder.mul_add_extension(limb_base, combined_limbs, this_limb);
            }
            constraints.push(builder.sub_extension(combined_limbs, output_result));

            // Range-check output_borrow to be one bit.
            let one = builder.one_extension();
            let not_borrow = builder.sub_extension(one, output_borrow);
            constraints.push(builder.mul_extension(output_borrow, not_borrow));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    U16SubtractionGenerator {
                        gate: *self,
                        row,
                        i,
                        _phantom: PhantomData,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * (5 + Self::num_limbs())
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1 << Self::limb_bits()
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * (3 + Self::num_limbs())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for U16SubtractionGate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for i in 0..self.num_ops {
            let input_x = vars.local_wires[self.wire_ith_input_x(i)];
            let input_y = vars.local_wires[self.wire_ith_input_y(i)];
            let input_borrow = vars.local_wires[self.wire_ith_input_borrow(i)];

            let result_initial = input_x - input_y - input_borrow;
            let base = F::from_canonical_u64(1 << 16u64);

            let output_result = vars.local_wires[self.wire_ith_output_result(i)];
            let output_borrow = vars.local_wires[self.wire_ith_output_borrow(i)];

            yield_constr.one(output_result - (result_initial + output_borrow * base));

            // Range-check output_result to be at most 16 bits.
            let mut combined_limbs = P::ZEROS;
            let limb_base = F::from_canonical_u64(1u64 << Self::limb_bits());
            for j in (0..Self::num_limbs()).rev() {
                let this_limb = vars.local_wires[self.wire_ith_output_jth_limb(i, j)];
                let max_limb = 1 << Self::limb_bits();
                let product = (0..max_limb)
                    .map(|x| this_limb - F::from_canonical_usize(x))
                    .product();
                yield_constr.one(product);

                combined_limbs = combined_limbs * limb_base + this_limb;
            }
            yield_constr.one(combined_limbs - output_result);

            // Range-check output_borrow to be one bit.
            yield_constr.one(output_borrow * (P::ONES - output_borrow));
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct U16SubtractionGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: U16SubtractionGate<F, D>,
    row: usize,
    i: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for U16SubtractionGenerator<F, D>
{
    fn id(&self) -> String {
        "U16SubtractionGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        vec![
            local_target(self.gate.wire_ith_input_x(self.i)),
            local_target(self.gate.wire_ith_input_y(self.i)),
            local_target(self.gate.wire_ith_input_borrow(self.i)),
        ]
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

        let get_local_wire = |column| witness.get_wire(local_wire(column));

        let input_x = get_local_wire(self.gate.wire_ith_input_x(self.i));
        let input_y = get_local_wire(self.gate.wire_ith_input_y(self.i));
        let input_borrow = get_local_wire(self.gate.wire_ith_input_borrow(self.i));

        let result_initial = input_x - input_y - input_borrow;
        let result_initial_u64 = result_initial.to_canonical_u64();
        let output_borrow = if result_initial_u64 > 1 << 16u64 {
            F::ONE
        } else {
            F::ZERO
        };

        let base = F::from_canonical_u64(1 << 16u64);
        let output_result = result_initial + base * output_borrow;

        let output_result_wire = local_wire(self.gate.wire_ith_output_result(self.i));
        let output_borrow_wire = local_wire(self.gate.wire_ith_output_borrow(self.i));

        out_buffer.set_wire(output_result_wire, output_result)?;
        out_buffer.set_wire(output_borrow_wire, output_borrow)?;

        let output_result_u64 = output_result.to_canonical_u64();

        let num_limbs = U16SubtractionGate::<F, D>::num_limbs();
        let limb_base = 1 << U16SubtractionGate::<F, D>::limb_bits();
        let output_limbs: Vec<_> = (0..num_limbs)
            .scan(output_result_u64, |acc, _| {
                let tmp = *acc % limb_base;
                *acc /= limb_base;
                Some(F::from_canonical_u64(tmp))
            })
            .collect();

        for j in 0..num_limbs {
            let wire = local_wire(self.gate.wire_ith_output_jth_limb(self.i, j));
            out_buffer.set_wire(wire, output_limbs[j])?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = U16SubtractionGate::deserialize(src, common_data)?;
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

    use plonky2::field::extension::quartic::QuarticExtension;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{PrimeField64, Sample};
    use plonky2::hash::hash_types::HashOut;
    use rand::Rng;
    use rand::rngs::OsRng;

    use super::*;

    #[test]
    fn test_gate_constraint() {
        type F = GoldilocksField;
        type FF = QuarticExtension<GoldilocksField>;
        const D: usize = 4;
        const NUM_U16_SUBTRACTION_OPS: usize = 3;

        fn get_wires(inputs_x: Vec<u64>, inputs_y: Vec<u64>, borrows: Vec<u64>) -> Vec<FF> {
            let mut v0 = Vec::new();
            let mut v1 = Vec::new();

            let limb_bits = U16SubtractionGate::<F, D>::limb_bits();
            let num_limbs = U16SubtractionGate::<F, D>::num_limbs();
            let limb_base = 1 << limb_bits;
            for c in 0..NUM_U16_SUBTRACTION_OPS {
                let input_x = F::from_canonical_u64(inputs_x[c]);
                let input_y = F::from_canonical_u64(inputs_y[c]);
                let input_borrow = F::from_canonical_u64(borrows[c]);

                let result_initial = input_x - input_y - input_borrow;
                let result_initial_u64 = result_initial.to_canonical_u64();
                let output_borrow = if result_initial_u64 > 1 << 16u64 {
                    F::ONE
                } else {
                    F::ZERO
                };

                let base = F::from_canonical_u64(1 << 16u64);
                let output_result = result_initial + base * output_borrow;

                let output_result_u64 = output_result.to_canonical_u64();

                let mut output_limbs: Vec<_> = (0..num_limbs)
                    .scan(output_result_u64, |acc, _| {
                        let tmp = *acc % limb_base;
                        *acc /= limb_base;
                        Some(F::from_canonical_u64(tmp))
                    })
                    .collect();

                v0.push(input_x);
                v0.push(input_y);
                v0.push(input_borrow);
                v0.push(output_result);
                v0.push(output_borrow);
                v1.append(&mut output_limbs);
            }

            v0.iter().chain(v1.iter()).map(|&x| x.into()).collect()
        }

        let mut rng = OsRng;
        let inputs_x = (0..NUM_U16_SUBTRACTION_OPS)
            .map(|_| rng.r#gen::<u16>() as u64)
            .collect();
        let inputs_y = (0..NUM_U16_SUBTRACTION_OPS)
            .map(|_| rng.r#gen::<u16>() as u64)
            .collect();
        let borrows = (0..NUM_U16_SUBTRACTION_OPS)
            .map(|_| (rng.r#gen::<u16>() % 2) as u64)
            .collect();

        let gate = U16SubtractionGate::<F, D> {
            num_ops: NUM_U16_SUBTRACTION_OPS,
            _phantom: PhantomData,
        };

        let vars = EvaluationVars {
            local_constants: &[],
            local_wires: &get_wires(inputs_x, inputs_y, borrows),
            public_inputs_hash: &HashOut::rand(),
        };

        assert!(
            gate.eval_unfiltered(vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
    }
}

#[cfg(test)]
mod batch_tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::*;
    use crate::gate_batch_testing::assert_base_batch_matches_eval_unfiltered;

    #[test]
    fn base_batch_matches_eval_unfiltered_across_batch() {
        let gate = U16SubtractionGate::<GoldilocksField, 2>::new_from_config(
            &CircuitConfig::standard_recursion_config(),
        );
        assert_base_batch_matches_eval_unfiltered(&gate);
    }
}
