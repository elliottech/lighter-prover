// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::ops::{Index, IndexMut};

use anyhow::Result;
use itertools::Itertools;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use super::base_register_info::BaseRegisterInfo;
use super::{BaseRegisterInfoTarget, BaseRegisterInfoTargetWitness, select_register_target};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::Builder;
use crate::types::constants::{
    NEW_INSTRUCTIONS_MAX_SIZE, PENDING_BASE_REGISTER_SIZE, REGISTER_STACK_SIZE,
};
use crate::types::register::BASE_REGISTER_INFO_SIZE;
use crate::utils::CircuitBuilderUtils;

pub const REGISTER_INFO_SIZE: usize = 1 + REGISTER_STACK_SIZE * BASE_REGISTER_INFO_SIZE;

#[derive(Clone, Debug, Deserialize, Copy)]
#[serde(default)]
pub struct RegisterStack {
    pub stack: [BaseRegisterInfo; REGISTER_STACK_SIZE],
    pub count: usize,
}

impl RegisterStack {
    pub fn iter(&self) -> std::slice::Iter<'_, BaseRegisterInfo> {
        self.stack.iter()
    }

    pub fn from_public_inputs<F>(pis: &[F]) -> Self
    where
        F: RichField,
    {
        assert!(pis.len() == REGISTER_INFO_SIZE);
        let count = pis[0].to_canonical_u64() as usize;
        let mut stack = vec![];
        for i in 0..REGISTER_STACK_SIZE {
            let start = 1 + i * BASE_REGISTER_INFO_SIZE;
            let end = start + BASE_REGISTER_INFO_SIZE;
            stack.push(BaseRegisterInfo::from_vec(&pis[start..end]));
        }
        Self {
            stack: stack.try_into().unwrap(),
            count,
        }
    }
}

impl Default for RegisterStack {
    fn default() -> Self {
        RegisterStack {
            stack: [BaseRegisterInfo::default(); REGISTER_STACK_SIZE],
            count: 0,
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct RegisterStackTarget {
    pub stack: [BaseRegisterInfoTarget; REGISTER_STACK_SIZE],
    pub count: Target,
}

impl Default for RegisterStackTarget {
    fn default() -> Self {
        RegisterStackTarget {
            stack: [BaseRegisterInfoTarget::default(); REGISTER_STACK_SIZE],
            count: Target::default(),
        }
    }
}

impl Index<usize> for RegisterStackTarget {
    type Output = BaseRegisterInfoTarget;

    fn index(&self, index: usize) -> &Self::Output {
        &self.stack[index]
    }
}

impl IndexMut<usize> for RegisterStackTarget {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.stack[index]
    }
}

impl RegisterStackTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            stack: (0..REGISTER_STACK_SIZE)
                .map(|_| BaseRegisterInfoTarget::new(builder))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            count: builder.add_virtual_target(),
        }
    }

    pub fn print(&self, builder: &mut Builder) {
        builder.println(self.count, "reg_stack_count");
        for (i, reg) in self.iter().enumerate() {
            reg.print(builder, format!("reg_stack[{}]", i).as_str());
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            stack: [BaseRegisterInfoTarget::empty(builder); REGISTER_STACK_SIZE],
            count: builder.constant_usize(0),
        }
    }

    pub fn is_empty(&self, builder: &mut Builder) -> BoolTarget {
        let assertions = self
            .iter()
            .map(|reg| reg.is_empty(builder))
            .collect::<Vec<_>>();
        builder.multi_and(&assertions)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, BaseRegisterInfoTarget> {
        self.stack.iter()
    }

    pub fn connect(&self, builder: &mut Builder, other: &Self) {
        for (reg, reg_target) in self.iter().zip_eq(other.iter()) {
            reg.connect(builder, reg_target);
        }
        builder.connect(self.count, other.count);
    }

    pub fn select(builder: &mut Builder, selector: BoolTarget, a: &Self, b: &Self) -> Self {
        Self {
            stack: (0..REGISTER_STACK_SIZE)
                .map(|i| select_register_target(builder, selector, &a[i], &b[i]))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            count: builder.select(selector, a.count, b.count),
        }
    }

    pub fn pop_front(&mut self, builder: &mut Builder, is_enabled: BoolTarget) {
        for i in 0..REGISTER_STACK_SIZE - 1 {
            self[i] = select_register_target(builder, is_enabled, &self[i + 1], &self[i]);
        }
        let empty = BaseRegisterInfoTarget::empty(builder);
        self[REGISTER_STACK_SIZE - 1] =
            select_register_target(builder, is_enabled, &empty, &self[REGISTER_STACK_SIZE - 1]);

        builder.conditional_assert_not_zero(is_enabled, self.count);
        self.count = builder.sub(self.count, is_enabled.target);
    }

    pub fn push_instructions<const MAX_NEW_INSTR: usize>(
        &mut self,
        builder: &mut Builder,
        new_instructions: &[BaseRegisterInfoTarget; NEW_INSTRUCTIONS_MAX_SIZE],
        new_instructions_count: Target,
    ) {
        // Stack new elements to the left without holes
        let mut new_stack = [BaseRegisterInfoTarget::empty(builder); PENDING_BASE_REGISTER_SIZE];
        let mut placed_count = builder.zero();
        for new_instruction in new_instructions.iter().take(MAX_NEW_INSTR) {
            for j in 0..MAX_NEW_INSTR {
                let j_eq_placed_count = builder.is_equal_constant(placed_count, j as u64);
                new_stack[j] = select_register_target(
                    builder,
                    j_eq_placed_count,
                    new_instruction,
                    &new_stack[j],
                );
            }
            let is_empty = new_instruction.is_empty(builder);
            let placed = builder.not(is_empty);
            placed_count = builder.add(placed_count, placed.target);
        }

        let is_enabled = builder.is_not_zero(new_instructions_count);
        builder.conditional_assert_eq(is_enabled, new_instructions_count, placed_count);

        // Update count
        self.count = builder.add(self.count, new_instructions_count);
        let max_count = builder.constant_usize(REGISTER_STACK_SIZE);
        builder.conditional_assert_lte(is_enabled, self.count, max_count, 8);

        // Shift existing stack to right by new_instructions_count. If base register is empty, ignore one shift.
        // Ignore the last element of register stack as it has to be empty by business logic for new pushes anyway.
        let base_reg_info_is_empty = self[0].is_empty(builder);
        let shift_amount = builder.sub(new_instructions_count, base_reg_info_is_empty.target);
        for i in (0..REGISTER_STACK_SIZE - 1).rev() {
            // Run another for to capture "shift_amount" as an out-circuit variable "j", and shift.
            // Ignore 0 as is_enabled would be false in that case.
            for j in 1..MAX_NEW_INSTR {
                if i + j >= REGISTER_STACK_SIZE {
                    break;
                }
                let is_index = builder.is_equal_constant(shift_amount, j as u64);
                let flag = builder.and(is_enabled, is_index);
                self[i + j] = select_register_target(builder, flag, &self[i], &self[i + j]);
            }
        }

        // Put new elements in the beginning
        let mut reached = builder.not(is_enabled);
        for i in 0..MAX_NEW_INSTR {
            let is_limit_reached = builder.is_equal_constant(new_instructions_count, i as u64);
            reached = builder.or(reached, is_limit_reached);
            self[i] = select_register_target(builder, reached, &self[i], &new_stack[i]);
        }
    }

    pub fn hash(&self, builder: &mut Builder) -> HashOutTarget {
        let mut is_empty_list = vec![];
        let mut elements = vec![];
        for register in self.iter() {
            elements.extend_from_slice(&register.get_hash_parameters());
            is_empty_list.push(register.is_empty(builder));
        }
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        let is_empty = builder.multi_and(&is_empty_list);
        let empty_hash = builder.zero_hash_out();
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.count);
        for register in self.iter() {
            register.register_public_input(builder);
        }
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert!(pis.len() == REGISTER_INFO_SIZE);
        let count = pis[0];
        let mut stack = vec![];
        for i in 0..REGISTER_STACK_SIZE {
            let start = 1 + i * BASE_REGISTER_INFO_SIZE;
            let end = start + BASE_REGISTER_INFO_SIZE;
            stack.push(BaseRegisterInfoTarget::from_vec(&pis[start..end]));
        }
        Self {
            stack: stack.try_into().unwrap(),
            count,
        }
    }
}

pub trait RegisterInfoTargetWitness<F: PrimeField64> {
    fn set_register_info_target(
        &mut self,
        register_target: &RegisterStackTarget,
        register: &RegisterStack,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> RegisterInfoTargetWitness<F> for T {
    fn set_register_info_target(
        &mut self,
        register_target: &RegisterStackTarget,
        register: &RegisterStack,
    ) -> Result<()> {
        for (reg, reg_target) in register.stack.iter().zip(register_target.stack.iter()) {
            self.set_base_register_info_target(reg_target, reg)?;
        }

        self.set_target(
            register_target.count,
            F::from_canonical_usize(register.count),
        )?;

        Ok(())
    }
}

#[cfg(test)]
impl RegisterStackTarget {
    pub fn random(builder: &mut Builder, non_empty_cnt: usize) -> Self {
        assert!(non_empty_cnt <= REGISTER_STACK_SIZE);
        let mut stack = [BaseRegisterInfoTarget::empty(builder); REGISTER_STACK_SIZE];
        for i in 0..non_empty_cnt {
            stack[i] = BaseRegisterInfoTarget::random(builder);
        }
        Self {
            stack,
            count: builder.constant_usize(non_empty_cnt),
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::*;
    use crate::bool_utils::CircuitBuilderBoolUtils;
    use crate::types::config::{Builder, C, F};
    use crate::types::constants::{INSERT_MAX_FIVE_REGISTERS, INSERT_MAX_THREE_REGISTERS};

    #[test]
    fn register_stack_push_pop_push() {
        // let _ = env_logger::try_init_from_env(
        //     env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"),
        // );

        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

        let _true = builder._true();
        let _false = builder._false();

        let mut set_base_register_to_empty = false;
        for i in 0..NEW_INSTRUCTIONS_MAX_SIZE {
            let (new_instructions, new_instructions_count) =
                get_random_new_instructions(&mut builder, i);

            let mut current_register_stack = RegisterStackTarget::empty(&mut builder);
            current_register_stack.push_instructions::<NEW_INSTRUCTIONS_MAX_SIZE>(
                &mut builder,
                &new_instructions,
                new_instructions_count,
            );
            builder.conditional_assert_eq(
                _true,
                current_register_stack.count,
                new_instructions_count,
            );

            // Pop half
            for _ in 0..i / 2 {
                current_register_stack.pop_front(&mut builder, _true);
            }
            let popped_count = builder.constant_usize(i / 2);
            let temp_count = builder.sub(new_instructions_count, popped_count);
            builder.conditional_assert_eq(_true, current_register_stack.count, temp_count);

            // Alternate between an empty base register info and a non-empty one
            if set_base_register_to_empty {
                current_register_stack[0] = BaseRegisterInfoTarget::empty(&mut builder);
            }

            // Push another half
            let (new_instructions_2, new_instructions_count_2) =
                get_random_new_instructions(&mut builder, i / 2);
            current_register_stack.push_instructions::<NEW_INSTRUCTIONS_MAX_SIZE>(
                &mut builder,
                &new_instructions_2,
                new_instructions_count_2,
            );

            // Assert total count
            builder.conditional_assert_eq(
                _true,
                current_register_stack.count,
                new_instructions_count,
            );

            // Assert correct placement - new_instructions_2
            current_register_stack
                .iter()
                .take(i / 2)
                .zip(new_instructions_2.iter().rev().take(i / 2).rev())
                .for_each(|(current_reg, new_reg)| {
                    let is_equal =
                        BaseRegisterInfoTarget::is_equal(&mut builder, current_reg, new_reg);
                    builder.assert_true(is_equal);
                });

            // Assert correct placement - new_instructions_1
            // Skip new_instructions[i / 2] if set_base_register_to_empty == true
            new_instructions
                .iter()
                .rev()
                .take(i)
                .rev()
                .skip(i / 2 + if set_base_register_to_empty { 1 } else { 0 })
                .zip_eq(
                    current_register_stack
                        .iter()
                        .skip(i / 2)
                        .take(i - i / 2 - if set_base_register_to_empty { 1 } else { 0 }),
                )
                .for_each(|(new_reg, current_reg)| {
                    let is_equal =
                        BaseRegisterInfoTarget::is_equal(&mut builder, new_reg, current_reg);
                    builder.assert_true(is_equal);
                });

            // Alternate base register info case
            set_base_register_to_empty = !set_base_register_to_empty;
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }

    #[test]
    fn register_stack_insertions() {
        // let _ = env_logger::try_init_from_env(
        //     env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"),
        // );

        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

        let _true = builder._true();
        let _false = builder._false();

        // Insert two with three capacity, with a hole in the middle
        {
            let mut current_register_stack = RegisterStackTarget::empty(&mut builder);
            let mut new_instructions =
                [BaseRegisterInfoTarget::empty(&mut builder); NEW_INSTRUCTIONS_MAX_SIZE];
            new_instructions[2] = BaseRegisterInfoTarget::random(&mut builder);
            new_instructions[2].account_index = builder.constant_u64(111111111111111);
            new_instructions[0] = BaseRegisterInfoTarget::random(&mut builder);
            new_instructions[0].account_index = builder.constant_u64(222222222222222);
            let new_instructions_count = builder.constant_usize(2);
            current_register_stack.push_instructions::<INSERT_MAX_THREE_REGISTERS>(
                &mut builder,
                &new_instructions,
                new_instructions_count,
            );

            builder.conditional_assert_eq(
                _true,
                current_register_stack[0].account_index,
                new_instructions[0].account_index,
            );
            builder.conditional_assert_eq(
                _true,
                current_register_stack[1].account_index,
                new_instructions[2].account_index,
            );

            for i in 2..REGISTER_STACK_SIZE {
                let is_empty = current_register_stack[i].is_empty(&mut builder);
                builder.assert_true(is_empty);
            }
        }

        // Insert two with five capacity, with three holes
        {
            let mut current_register_stack = RegisterStackTarget::empty(&mut builder);
            let mut new_instructions =
                [BaseRegisterInfoTarget::empty(&mut builder); NEW_INSTRUCTIONS_MAX_SIZE];
            new_instructions[4] = BaseRegisterInfoTarget::random(&mut builder);
            new_instructions[4].account_index = builder.constant_u64(111111111111111);
            new_instructions[1] = BaseRegisterInfoTarget::random(&mut builder);
            new_instructions[1].account_index = builder.constant_u64(222222222222222);
            let new_instructions_count = builder.constant_usize(2);
            current_register_stack.push_instructions::<INSERT_MAX_FIVE_REGISTERS>(
                &mut builder,
                &new_instructions,
                new_instructions_count,
            );

            builder.conditional_assert_eq(
                _true,
                current_register_stack[0].account_index,
                new_instructions[1].account_index,
            );
            builder.conditional_assert_eq(
                _true,
                current_register_stack[1].account_index,
                new_instructions[4].account_index,
            );

            for i in 2..REGISTER_STACK_SIZE {
                let is_empty = current_register_stack[i].is_empty(&mut builder);
                builder.assert_true(is_empty);
            }
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }

    #[test]
    fn register_stack_push_and_pop() {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

        let one = builder.constant_usize(1);
        let _true = builder._true();
        let _false = builder._false();

        for i in 0..NEW_INSTRUCTIONS_MAX_SIZE {
            let (new_instructions, new_instructions_count) =
                get_random_new_instructions(&mut builder, i);

            let mut current_register_stack = RegisterStackTarget::empty(&mut builder);
            current_register_stack.push_instructions::<NEW_INSTRUCTIONS_MAX_SIZE>(
                &mut builder,
                &new_instructions,
                new_instructions_count,
            );
            builder.conditional_assert_eq(
                _true,
                current_register_stack.count,
                new_instructions_count,
            );

            // Make sure elements are pushed in correct order
            new_instructions
                .iter()
                .skip(NEW_INSTRUCTIONS_MAX_SIZE - i)
                .zip(current_register_stack.iter().take(i))
                .for_each(|(a, b)| {
                    let is_equal = BaseRegisterInfoTarget::is_equal(&mut builder, a, b);
                    builder.assert_true(is_equal);
                });
            // The rest should be empty
            new_instructions
                .iter()
                .take(NEW_INSTRUCTIONS_MAX_SIZE - i)
                .chain(current_register_stack.iter().skip(i))
                .for_each(|reg| {
                    let is_empty = reg.is_empty(&mut builder);
                    builder.assert_true(is_empty);
                });

            // Pop one by one
            let mut current_register_stack_count = new_instructions_count;
            for j in 0..i {
                // False flag
                current_register_stack.pop_front(&mut builder, _false);
                builder.conditional_assert_eq(
                    _true,
                    current_register_stack.count,
                    current_register_stack_count,
                );

                // Take a copy before popping
                let copy = current_register_stack;

                // Pop
                current_register_stack.pop_front(&mut builder, _true);

                // Assert Count
                current_register_stack_count = builder.sub(current_register_stack_count, one);
                builder.conditional_assert_eq(
                    _true,
                    current_register_stack.count,
                    current_register_stack_count,
                );

                // Assert shift was done correctly
                for k in 0..REGISTER_STACK_SIZE - 1 - j {
                    let is_equal = BaseRegisterInfoTarget::is_equal(
                        &mut builder,
                        &current_register_stack[k],
                        &copy[k + 1],
                    );
                    builder.assert_true(is_equal);
                }
                let last_is_empty =
                    current_register_stack[REGISTER_STACK_SIZE - 1].is_empty(&mut builder);
                builder.assert_true(last_is_empty);
            }
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }

    fn get_random_new_instructions(
        builder: &mut Builder,
        count: usize,
    ) -> ([BaseRegisterInfoTarget; NEW_INSTRUCTIONS_MAX_SIZE], Target) {
        assert!(count <= NEW_INSTRUCTIONS_MAX_SIZE);
        let mut new_instructions =
            [BaseRegisterInfoTarget::empty(builder); NEW_INSTRUCTIONS_MAX_SIZE];
        for i in 0..count {
            new_instructions[NEW_INSTRUCTIONS_MAX_SIZE - 1 - i] =
                BaseRegisterInfoTarget::random(builder);
        }
        (new_instructions, builder.constant_usize(count))
    }

    #[test]
    fn register_stack_with_holes() {
        fn get_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
            if k == 0 {
                return vec![vec![]];
            }
            if k > n {
                return vec![];
            }
            let mut result = Vec::new();
            let mut indices: Vec<usize> = (0..k).collect();
            loop {
                result.push(indices.clone());
                let mut i = k as isize - 1;
                while i >= 0 && indices[i as usize] == n - k + i as usize {
                    i -= 1;
                }
                if i < 0 {
                    break;
                }
                indices[i as usize] += 1;
                for j in (i as usize + 1)..k {
                    indices[j] = indices[j - 1] + 1;
                }
            }
            result
        }

        fn test_push_with_holes_for_combos<const INSTR_COUNT: usize>(
            non_empty_positions_combinations: &[Vec<usize>],
            non_empty_count: usize,
        ) {
            let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
            let _true = builder._true();

            for non_empty_positions in non_empty_positions_combinations {
                assert_eq!(non_empty_positions.len(), non_empty_count);
                let mut new_instructions =
                    [BaseRegisterInfoTarget::empty(&mut builder); NEW_INSTRUCTIONS_MAX_SIZE];
                for &non_empty_index in non_empty_positions.iter() {
                    new_instructions[non_empty_index] =
                        BaseRegisterInfoTarget::random(&mut builder);
                }
                let new_instructions_count = builder.constant_usize(non_empty_count);

                let mut stack = RegisterStackTarget::empty(&mut builder);
                stack.push_instructions::<INSTR_COUNT>(
                    &mut builder,
                    &new_instructions,
                    new_instructions_count,
                );
                builder.conditional_assert_eq(_true, stack.count, new_instructions_count);

                // Elements of new_instructions arae left-to-right stacked in correct order
                for (i, &pos) in non_empty_positions.iter().enumerate() {
                    let is_equal = BaseRegisterInfoTarget::is_equal(
                        &mut builder,
                        &stack[i],
                        &new_instructions[pos],
                    );
                    builder.assert_true(is_equal);
                }
                // Remaining slots empty
                for i in non_empty_count..REGISTER_STACK_SIZE {
                    let is_empty = stack[i].is_empty(&mut builder);
                    builder.assert_true(is_empty);
                }
            }

            let data = builder.build::<C>();
            data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
                .unwrap();
        }

        for ct in 0..=NEW_INSTRUCTIONS_MAX_SIZE {
            test_push_with_holes_for_combos::<1>(&get_combinations(1, ct), ct);
            test_push_with_holes_for_combos::<2>(&get_combinations(2, ct), ct);
            test_push_with_holes_for_combos::<3>(&get_combinations(3, ct), ct);
            test_push_with_holes_for_combos::<4>(&get_combinations(4, ct), ct);
            test_push_with_holes_for_combos::<5>(&get_combinations(5, ct), ct);
            test_push_with_holes_for_combos::<6>(&get_combinations(6, ct), ct);
            test_push_with_holes_for_combos::<7>(&get_combinations(7, ct), ct);
            test_push_with_holes_for_combos::<8>(&get_combinations(8, ct), ct);
        }
    }
}
