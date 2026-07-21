// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::{BTreeMap, BTreeSet, HashMap};

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::GenericConfig;

use crate::bigint::bigint::SignTarget;
use crate::bigint::biguint::BigUintTarget;
use crate::blob::evaluate_bitstream::BitstreamStateTarget;
use crate::delta::evaluate_sequence::SequenceStateTarget;
use crate::uint::u8::U8Target;
use crate::uint::u16::gadgets::arithmetic_u16::U16Target;
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;
use crate::uint::u32::gadgets::interleaved_u32::B32Target;

#[derive(Debug)]
pub struct Builder<F: RichField + Extendable<D>, const D: usize> {
    pub builder: CircuitBuilder<F, D>,

    pub(crate) u32_arithmetic_results: HashMap<U32ArithmeticOperation, (U32Target, U32Target)>,
    pub(crate) u32_add_many_results: HashMap<U32AddManyOperation, (U32Target, U32Target)>,
    pub(crate) u32_sub_results: HashMap<U32SubtractionOperation, (U32Target, U32Target)>,
    pub(crate) u32_interleave_results: HashMap<U32Target, B32Target>,
    pub(crate) u32_split_cache: HashMap<Target, (U32Target, U32Target)>,

    pub(crate) u16_arithmetic_results: HashMap<U16ArithmeticOperation, (U16Target, U16Target)>,
    pub(crate) u16_add_many_results: HashMap<U16AddManyOperation, (U16Target, U16Target)>,
    pub(crate) u16_sub_results: HashMap<U16SubtractionOperation, (U16Target, U16Target)>,
    pub(crate) u16_split_cache: HashMap<Target, Vec<U16Target>>,

    pub(crate) split_le_cache: HashMap<Target, Vec<BoolTarget>>,
    pub(crate) split_le_base_cache: HashMap<(usize, Target), Vec<Target>>,
    pub(crate) split_bytes_cache: HashMap<Target, Vec<U8Target>>,
    pub(crate) is_equal_cache: HashMap<(Target, Target), BoolTarget>,
    pub(crate) is_lte_cache: HashMap<(Target, Target), (BoolTarget, usize)>,
    pub(crate) is_lte_biguint_cache: HashMap<(BigUintTarget, BigUintTarget), BoolTarget>,
    pub(crate) cmp_cache: HashMap<(Target, Target), (SignTarget, usize)>,
    pub(crate) cmp_biguint_cache: HashMap<(BigUintTarget, BigUintTarget), SignTarget>,
    pub(crate) div_rem_biguint_cache:
        HashMap<(BigUintTarget, BigUintTarget, usize), (BigUintTarget, BigUintTarget)>,
    pub(crate) random_access_cache: HashMap<(Target, Vec<Target>), Target>,

    pub(crate) range_checks: BTreeMap<usize, BTreeSet<Target>>, // Bit size to a set of targets. Using BTreeMap and BTreeSet to get deterministic circuit build
    pub(crate) range_check_targets_to_bit_sizes: HashMap<Target, usize>, // Target to bit size

    pub(crate) sequence_state: HashMap<usize, SequenceStateTarget>,
    pub(crate) bitstream_state: HashMap<usize, BitstreamStateTarget>,
}

impl<F, const D: usize> Builder<F, D>
where
    F: RichField + Extendable<D>,
{
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            builder: CircuitBuilder::<F, D>::new(config),

            u32_arithmetic_results: HashMap::new(),
            u32_add_many_results: HashMap::new(),
            u32_sub_results: HashMap::new(),
            u32_interleave_results: HashMap::new(),
            u32_split_cache: HashMap::new(),

            u16_arithmetic_results: HashMap::new(),
            u16_add_many_results: HashMap::new(),
            u16_sub_results: HashMap::new(),
            u16_split_cache: HashMap::new(),

            split_le_cache: HashMap::new(),
            split_le_base_cache: HashMap::new(),
            split_bytes_cache: HashMap::new(),
            is_equal_cache: HashMap::new(),
            is_lte_cache: HashMap::new(),
            is_lte_biguint_cache: HashMap::new(),
            cmp_cache: HashMap::new(),
            cmp_biguint_cache: HashMap::new(),
            div_rem_biguint_cache: HashMap::new(),
            random_access_cache: HashMap::new(),

            range_checks: BTreeMap::new(),
            range_check_targets_to_bit_sizes: HashMap::new(),

            sequence_state: HashMap::new(),
            bitstream_state: HashMap::new(),
        }
    }

    pub fn build<C: GenericConfig<D, F = F>>(self) -> CircuitData<F, C, D> {
        if !self.range_checks.is_empty() {
            log::warn!(
                "Warning: range_checks still contains {} entries. Make sure you called perform_registered_range_checks at the end of your circuit.",
                self.range_checks.len()
            );
        }

        self.builder.build()
    }

    pub fn config(&self) -> &CircuitConfig {
        &self.builder.config
    }

    pub fn num_gates(&self) -> usize {
        self.builder.num_gates()
    }

    pub fn print_gate_counts(&mut self, min_delta: usize) {
        self.builder.print_gate_counts(min_delta)
    }
}

/// Represents a u32 mul_add arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct U32ArithmeticOperation {
    pub(crate) multiplicand_0: U32Target,
    pub(crate) multiplicand_1: U32Target,
    pub(crate) addend: U32Target,
}

/// Represents a u16 mul_add arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct U16ArithmeticOperation {
    pub(crate) multiplicand_0: U16Target,
    pub(crate) multiplicand_1: U16Target,
    pub(crate) addend: U16Target,
}

/// Represents a u32 add many arithmetic operation in the circuit. Used to memoize results.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct U32AddManyOperation {
    pub(crate) addends: Vec<U32Target>,
    pub(crate) carry: U32Target,
}

/// Represents a u16 add many arithmetic operation in the circuit. Used to memoize results.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct U16AddManyOperation {
    pub(crate) addends: Vec<U16Target>,
    pub(crate) carry: U16Target,
}

/// Represents a u32 sub arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct U32SubtractionOperation {
    pub(crate) x: U32Target,
    pub(crate) y: U32Target,
    pub(crate) borrow: U32Target,
}

/// Represents a u16 sub arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct U16SubtractionOperation {
    pub(crate) x: U16Target,
    pub(crate) y: U16Target,
    pub(crate) borrow: U16Target,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use rand::Rng;

    use super::*;
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    #[test]
    fn test_bits_cache() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        assert_eq!(builder.split_le_cache.len(), 0);

        let a_target = builder.add_virtual_target();
        let result_a = builder.split_le(a_target, 40);
        assert_eq!(builder.split_le_cache.len(), 1);

        let result_b = builder.split_le(a_target, 40);
        assert_eq!(builder.split_le_cache.len(), 1);

        assert_eq!(result_a, result_b);

        let mut pw = PartialWitness::<F>::new();
        pw.set_target(
            a_target,
            F::from_canonical_u32(rand::thread_rng().r#gen::<u32>()),
        )
        .unwrap();

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }

    #[test]
    fn test_is_equal_cache() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        assert_eq!(builder.is_equal_cache.len(), 0);

        let a_target = builder.add_virtual_target();
        let b_target = builder.add_virtual_target();

        let is_equal1 = builder.is_equal(a_target, b_target);
        assert_eq!(builder.is_equal_cache.len(), 2);

        let is_equal2 = builder.is_equal(a_target, b_target);
        assert_eq!(builder.is_equal_cache.len(), 2);

        let is_equal3 = builder.is_equal(b_target, a_target);
        assert_eq!(builder.is_equal_cache.len(), 2);

        assert_eq!(is_equal1, is_equal2);
        assert_eq!(is_equal1, is_equal3);

        let is_not_equal = builder.is_not_equal(a_target, b_target);
        assert_eq!(builder.is_equal_cache.len(), 2);

        let is_not_equal2 = builder.not(is_equal3);
        assert_eq!(is_not_equal, is_not_equal2);

        let mut pw = PartialWitness::<F>::new();
        pw.set_target(
            a_target,
            F::from_canonical_u32(rand::thread_rng().r#gen::<u32>()),
        )
        .unwrap();
        pw.set_target(
            b_target,
            F::from_canonical_u32(rand::thread_rng().r#gen::<u32>()),
        )
        .unwrap();

        let data = builder.build::<C>();
        data.verify(data.prove(pw).unwrap())
    }
}
