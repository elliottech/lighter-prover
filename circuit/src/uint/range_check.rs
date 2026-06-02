// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

extern crate paste;
use core::marker::PhantomData;
use std::collections::HashSet;

use anyhow::Result;
use log::warn;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::gates::gate::Gate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::builder::Builder;
use crate::byte::split::CircuitBuilderByteSplit;
use crate::utils::ceil_div_usize;

const CUSTOM_GATE_SIZES: &[usize] = &[16, 32, 48];
lazy_static! {
    pub static ref CUSTOM_GATE_SIZES_SET: HashSet<usize> = {
        let mut set = HashSet::new();
        for val in CUSTOM_GATE_SIZES.iter() {
            set.insert(*val);
        }
        set
    };
}

impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    #[track_caller]
    pub fn register_range_check(&mut self, val: Target, bit_size: usize) {
        assert!(
            bit_size <= 64,
            "Bit size for range check must be <= 64, got {}",
            bit_size
        );
        if bit_size == 64 {
            return;
        }

        // Do not add constraints for constant values, do build time assertions
        if let Some(const_val) = self.builder.target_as_constant(val) {
            let const_val_u64 = const_val.to_canonical_u64();
            assert!(
                const_val_u64 < (1 << bit_size),
                "Constant value {} exceeds the range for bit size {}",
                const_val_u64,
                bit_size
            );
            return;
        }

        if let Some(&old_bit_size) = self.range_check_targets_to_bit_sizes.get(&val) {
            if old_bit_size != bit_size {
                let caller = std::panic::Location::caller();
                warn!(
                    "Target {:?} already registered for range check with bit size {}, but now being registered with bit size {}. Called from {}:{}",
                    val,
                    old_bit_size,
                    bit_size,
                    caller.file(),
                    caller.line()
                );
            }

            // Preserve smaller bit size for range check. If old one is the larger one, remove it from the cache
            if old_bit_size <= bit_size {
                return;
            }
            self.range_checks
                .entry(old_bit_size)
                .or_default()
                .remove(&val);
            self.range_check_targets_to_bit_sizes.remove(&val);
        }

        self.range_checks.entry(bit_size).or_default().insert(val);
        self.range_check_targets_to_bit_sizes.insert(val, bit_size);
    }

    pub fn perform_registered_range_checks(&mut self) {
        self.range_check_targets_to_bit_sizes.clear();

        let entries: Vec<_> = self
            .range_checks
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        self.range_checks.clear();
        for (bit_size, targets) in entries {
            let filtered_targets = targets
                .iter()
                .filter(|x| {
                    if let Some(bits) = self.split_le_cache.get(x) {
                        if bits.len() != bit_size {
                            warn!("Target: {:?} passed to split_le with bit size {}, but registered to range check with bit size {}", x, bits.len(), bit_size);
                        }
                        if bits.len() <= bit_size {
                            return false; // Already split to lower number of bits, no need to range check
                        }
                    }

                    if let Some(bytes) = self.split_bytes_cache.get(x) {
                        if bytes.len() * 8 != bit_size {
                            warn!("Target: {:?} passed to split_bytes with byte size {}, but registered to range check with bit size {}", x, bytes.len(), bit_size);
                        }
                        if bytes.len() * 8 <= bit_size {
                            return false; // Already split to lower number of bits, no need to range check
                        }
                    }
                    true
                })
                .cloned()
                .collect::<Vec<_>>();
            for target in filtered_targets {
                if !CUSTOM_GATE_SIZES_SET.contains(&bit_size) {
                    if bit_size % 8 == 0 {
                        self.split_bytes(target, bit_size / 8);
                    } else {
                        self.split_le(target, bit_size);
                    }
                } else {
                    let gate = RangeCheckGate::new_from_config(self.config(), bit_size);
                    let (row, copy) =
                        self.find_slot(gate, &[F::from_canonical_usize(bit_size)], &[]);
                    self.connect(target, Target::wire(row, gate.wire_ith_input(copy)));
                }
            }
        }
    }

    pub fn perform_registered_range_checks_with_custom_range_check_sizes(
        &mut self,
        custom_gate_sizes_set: &HashSet<usize>,
    ) {
        self.range_check_targets_to_bit_sizes.clear();

        let entries: Vec<_> = self
            .range_checks
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        self.range_checks.clear();
        for (bit_size, targets) in entries {
            let filtered_targets = targets
                .iter()
                .filter(|x| {
                    if let Some(bits) = self.split_le_cache.get(x) {
                        if bits.len() != bit_size {
                            warn!("Target: {:?} passed to split_le with bit size {}, but registered to range check with bit size {}", x, bits.len(), bit_size);
                        }
                        if bits.len() <= bit_size {
                            return false; // Already split to lower number of bits, no need to range check
                        }
                    }

                    if let Some(bytes) = self.split_bytes_cache.get(x) {
                        if bytes.len() * 8 != bit_size {
                            warn!("Target: {:?} passed to split_bytes with byte size {}, but registered to range check with bit size {}", x, bytes.len(), bit_size);
                        }
                        if bytes.len() * 8 <= bit_size {
                            return false; // Already split to lower number of bits, no need to range check
                        }
                    }
                    true
                })
                .cloned()
                .collect::<Vec<_>>();
            for target in filtered_targets {
                if !custom_gate_sizes_set.contains(&bit_size) {
                    if bit_size % 8 == 0 {
                        self.split_bytes(target, bit_size / 8);
                    } else {
                        self.split_le(target, bit_size);
                    }
                } else {
                    let gate = RangeCheckGate::new_from_config(self.config(), bit_size);
                    let (row, copy) =
                        self.find_slot(gate, &[F::from_canonical_usize(bit_size)], &[]);
                    self.connect(target, Target::wire(row, gate.wire_ith_input(copy)));
                }
            }
        }
    }
}

//A custom gate to add range check constraints
//It ensures that a certain number fits within a certain number of limbs in a base 2
#[derive(Clone, Debug, Default, Copy)]
pub struct RangeCheckGate<F: RichField + Extendable<D>, const D: usize> {
    pub num_ops: usize,
    pub bit_size: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> RangeCheckGate<F, D> {
    pub fn new_from_config(config: &CircuitConfig, bit_size: usize) -> Self {
        Self {
            num_ops: Self::num_ops(config, bit_size),
            bit_size,
            _phantom: PhantomData,
        }
    }

    pub const AUX_LIMB_BITS: usize = 2;
    pub const BASE: usize = 1 << Self::AUX_LIMB_BITS;

    pub(crate) fn num_ops(config: &CircuitConfig, bit_size: usize) -> usize {
        let routed_wires_per_op = 1;
        let unrouted_wires_per_op = ceil_div_usize(bit_size, Self::AUX_LIMB_BITS);
        let wires_per_op = routed_wires_per_op + unrouted_wires_per_op;
        config.num_wires / wires_per_op.min(config.num_routed_wires / routed_wires_per_op)
    }

    fn aux_limbs_per_input(&self) -> usize {
        ceil_div_usize(self.bit_size, Self::AUX_LIMB_BITS)
    }

    pub fn wire_ith_input(&self, i: usize) -> usize {
        debug_assert!(i < self.num_ops);
        i
    }

    pub fn wire_ith_input_jth_aux_limb(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.num_ops);
        debug_assert!(j < self.aux_limbs_per_input());
        self.num_ops + self.aux_limbs_per_input() * i + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for RangeCheckGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)?;
        dst.write_usize(self.bit_size)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        let bit_size = src.read_usize()?;
        Ok(Self {
            num_ops,
            bit_size,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_ops * (1 + self.aux_limbs_per_input()));

        let base = F::Extension::from_canonical_usize(Self::BASE);
        for i in 0..self.num_ops {
            let input_limb = vars.local_wires[self.wire_ith_input(i)];
            let aux_limbs: Vec<_> = (0..self.aux_limbs_per_input())
                .map(|j| vars.local_wires[self.wire_ith_input_jth_aux_limb(i, j)])
                .collect();
            let computed_sum = reduce_with_powers(&aux_limbs, base);

            constraints.push(computed_sum - input_limb);
            for aux_limb in aux_limbs.iter().take(aux_limbs.len() - 1) {
                constraints.push(
                    (0..Self::BASE)
                        .map(|i| *aux_limb - F::Extension::from_canonical_usize(i))
                        .product(),
                );
            }
            let iter = if self.bit_size % 2 == 1 {
                Self::BASE / 2
            } else {
                Self::BASE
            };
            constraints.push(
                (0..iter)
                    .map(|i| *aux_limbs.last().unwrap() - F::Extension::from_canonical_usize(i))
                    .product(),
            );
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let base = F::from_canonical_usize(Self::BASE);
        for i in 0..self.num_ops {
            let input_limb = vars.local_wires[self.wire_ith_input(i)];
            let aux_limbs: Vec<_> = (0..self.aux_limbs_per_input())
                .map(|j| vars.local_wires[self.wire_ith_input_jth_aux_limb(i, j)])
                .collect();
            let computed_sum = reduce_with_powers(&aux_limbs, base);

            yield_constr.one(computed_sum - input_limb);
            for aux_limb in aux_limbs.iter().take(aux_limbs.len() - 1) {
                yield_constr.one(
                    (0..Self::BASE)
                        .map(|i| *aux_limb - F::from_canonical_usize(i))
                        .product(),
                );
            }
            let iter = if self.bit_size % 2 == 1 {
                Self::BASE / 2
            } else {
                Self::BASE
            };
            yield_constr.one(
                (0..iter)
                    .map(|i| *aux_limbs.last().unwrap() - F::from_canonical_usize(i))
                    .product(),
            );
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_ops * (1 + self.aux_limbs_per_input()));

        let base = builder.constant(F::from_canonical_usize(Self::BASE));
        for i in 0..self.num_ops {
            let input_limb = vars.local_wires[self.wire_ith_input(i)];
            let aux_limbs: Vec<_> = (0..self.aux_limbs_per_input())
                .map(|j| vars.local_wires[self.wire_ith_input_jth_aux_limb(i, j)])
                .collect();
            let computed_sum = reduce_with_powers_ext_circuit(builder, &aux_limbs, base);

            constraints.push(builder.sub_extension(computed_sum, input_limb));
            for aux_limb in aux_limbs.iter().take(aux_limbs.len() - 1) {
                constraints.push({
                    let mut acc = builder.one_extension();
                    (0..Self::BASE).for_each(|i| {
                        // We update our accumulator as:
                        // acc' = acc (x - i)
                        //      = acc x + (-i) acc
                        // Since -i is constant, we can do this in one arithmetic_extension call.
                        let neg_i = -F::from_canonical_usize(i);
                        acc = builder.arithmetic_extension(F::ONE, neg_i, acc, *aux_limb, acc)
                    });
                    acc
                });
            }
            let iter = if self.bit_size % 2 == 1 {
                Self::BASE / 2
            } else {
                Self::BASE
            };
            constraints.push({
                let mut acc = builder.one_extension();
                (0..iter).for_each(|i| {
                    let neg_i = -F::from_canonical_usize(i);
                    let last_limb = *aux_limbs.last().unwrap();
                    acc = builder.arithmetic_extension(F::ONE, neg_i, acc, last_limb, acc)
                });
                acc
            });
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let result: Vec<WitnessGeneratorRef<F, D>> = (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    RangeCheckGenerator {
                        gate: *self,
                        row,
                        i,
                    }
                    .adapter(),
                )
            })
            .collect();
        result
    }

    fn num_wires(&self) -> usize {
        self.num_ops * (1 + self.aux_limbs_per_input())
    }

    fn num_constants(&self) -> usize {
        0
    }

    // Bounded by the range-check (x-0)*(x-1)*...*(x-BASE+1).
    fn degree(&self) -> usize {
        Self::BASE
    }

    // 1 for checking the each sum of aux limbs, plus a range check for each aux limb.
    fn num_constraints(&self) -> usize {
        self.num_ops * (1 + self.aux_limbs_per_input())
    }
}

#[derive(Clone, Debug, Default)]
pub struct RangeCheckGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub gate: RangeCheckGate<F, D>,
    pub row: usize,
    pub i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for RangeCheckGenerator<F, D>
{
    fn id(&self) -> String {
        "RangeCheckGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(self.row, self.gate.wire_ith_input(self.i))]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let sum_value = witness
            .get_target(Target::wire(self.row, self.gate.wire_ith_input(self.i)))
            .to_canonical_u64();

        let base = RangeCheckGate::<F, D>::BASE as u64;
        let limbs = (0..self.gate.aux_limbs_per_input())
            .map(|j| Target::wire(self.row, self.gate.wire_ith_input_jth_aux_limb(self.i, j)));
        let limbs_value = (0..self.gate.aux_limbs_per_input())
            .scan(sum_value, |acc, _| {
                let tmp = *acc % base;
                *acc /= base;
                Some(F::from_canonical_u64(tmp))
            })
            .collect::<Vec<_>>();

        for (b, b_value) in limbs.zip(limbs_value) {
            out_buffer.set_target(b, b_value)?;
        }
        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = RangeCheckGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self { row, gate, i })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use paste::paste;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    #[allow(unused_imports)]
    use plonky2::field::types::Field64;
    use plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use plonky2::iop::target::Target;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use rand::Rng;

    use super::*;

    macro_rules! generate_low_degree_tests {
        ($bit_size:expr) => {
            paste! {
                #[test]
                fn [<low_degree_bits_ $bit_size>]() {

                    let gate = RangeCheckGate::new_from_config(&CircuitConfig::standard_recursion_config(), $bit_size);
                    test_low_degree::<GoldilocksField, _, 4>(gate);
                }
            }
        };
    }

    macro_rules! generate_eval_fns_tests {
        ($bit_size:expr) => {
            paste! {
                #[test]
                fn [<eval_fns_bits_ $bit_size>]() {
                    const D: usize = 2;
                    type C = PoseidonGoldilocksConfig;
                    type F = <C as GenericConfig<D>>::F;
                    let config = CircuitConfig::standard_recursion_config();
                    let gate = RangeCheckGate::new_from_config(&config, $bit_size);
                    test_eval_fns::<F, C, _, D>(gate).unwrap();
                }
            }
        };
    }

    macro_rules! impl_range_check_tests {
        ($bit_size:expr) => {
            paste! {
                #[test]
                fn [<test_rangecheck_success_ $bit_size>]() -> Result<()> {
                    const D: usize = 2;
                    type C = PoseidonGoldilocksConfig;
                    type F = <C as GenericConfig<D>>::F;

                    let config = CircuitConfig::standard_recursion_config();
                    let mut builder = CircuitBuilder::<F, D>::new(config.clone());

                    let input = builder.add_virtual_target();
                    let gate = RangeCheckGate::new_from_config(&config, $bit_size);
                    let gate_ref = gate.clone();
                    let constants = vec![];
                    let (row, _op_index) = builder.find_slot(gate, &constants, &constants);
                    builder.connect(input, Target::wire(row, gate_ref.wire_ith_input(0)));

                    let circuit_data = builder.build::<C>();

                    let mut pw = PartialWitness::new();
                    let value = F::from_canonical_u64(rand::thread_rng().gen_range(0..(1 << $bit_size)));
                    pw.set_target(input, value)?;

                    let proof = circuit_data.prove(pw)?;
                    circuit_data.verify(proof)?;
                    Ok(())
                }

                #[test]
                #[should_panic(expected = "Condition failed")]
                fn [<test_rangecheck_failure_ $bit_size>]() {
                    const D: usize = 2;
                    type C = PoseidonGoldilocksConfig;
                    type F = <C as GenericConfig<D>>::F;

                    let config = CircuitConfig::standard_recursion_config();
                    let mut builder = CircuitBuilder::<F, D>::new(config.clone());

                    let input = builder.add_virtual_target();
                    let gate = RangeCheckGate::new_from_config(&config, $bit_size);
                    let gate_ref = gate.clone();
                    let constants = vec![];
                    let (row, _op_index) = builder.find_slot(gate, &constants, &constants);
                    builder.connect(input, Target::wire(row, gate_ref.wire_ith_input(0)));

                    let circuit_data = builder.build::<C>();

                    let mut pw = PartialWitness::new();
                    let value = F::from_canonical_u64(rand::thread_rng().gen_range((1 << $bit_size)..F::ORDER));
                    pw.set_target(input, value).unwrap();

                    let proof = circuit_data.prove(pw).unwrap();
                    circuit_data.verify(proof).unwrap();
                }
            }
        };
    }

    macro_rules! impl_range_check_all_tests {
        ($($bit_size:expr),*) => {
            $(
                generate_low_degree_tests!($bit_size);
                impl_range_check_tests!($bit_size);
                generate_eval_fns_tests!($bit_size);
            )*
        };
    }

    impl_range_check_all_tests!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48
    );
}
