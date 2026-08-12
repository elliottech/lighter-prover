// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use log::warn;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::Target;

use super::split_gate::ByteDecompositionGate;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::builder::Builder;
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

const GOLDILOCKS_BYTES: usize = 8;

pub trait CircuitBuilderByteSplit<F: RichField + Extendable<D>, const D: usize> {
    fn split_bytes(&mut self, target: Target, num_bytes: usize) -> Vec<U8Target>;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderByteSplit<F, D> for Builder<F, D> {
    #[track_caller]
    fn split_bytes(&mut self, target: Target, num_bytes: usize) -> Vec<U8Target> {
        debug_assert!(num_bytes <= GOLDILOCKS_BYTES);

        if let Some(result) = self.split_bytes_cache.get(&target) {
            if result.len() == num_bytes {
                return result.clone();
            }

            let caller = std::panic::Location::caller();
            warn!(
                "split_bytes({:?}) is called with different num_bytes({}/{}). Please fix this!. Caller {}:{}",
                target,
                result.len(),
                num_bytes,
                caller.file(),
                caller.line()
            );
        }

        // We always use GOLDILOCKS_BYTES for the gate, even if we only need num_bytes. Because fewer gate types are
        // faster even with extra `assert_zero`s
        let gate_type =
            ByteDecompositionGate::new_from_config(&self.builder.config, GOLDILOCKS_BYTES);
        let (row, copy) =
            self.find_slot(gate_type, &[F::from_canonical_usize(GOLDILOCKS_BYTES)], &[]);

        // Connect the target to the gate
        self.connect(target, Target::wire(row, gate_type.i_th_sum(copy)));

        // Collect results
        let mut result = Target::wires_from_range(row, gate_type.i_th_limbs(copy));
        // Assert that the extra bytes are zero
        result[num_bytes..].iter().for_each(|t| {
            self.assert_zero(*t);
        });
        result.truncate(num_bytes);

        if num_bytes == GOLDILOCKS_BYTES {
            let low_sum = self.add_many([result[0], result[1], result[2], result[3]]);
            let low_is_zero = self.is_zero(low_sum);

            let high_sum = self.add_many([result[4], result[5], result[6], result[7]]);
            let all_high_ones = self.constant(F::from_canonical_u64(4 * 255));
            let high_all_ones = self.is_equal(high_sum, all_high_ones);

            let is_non_canonical = self.and_not(high_all_ones, low_is_zero);
            self.assert_false(is_non_canonical);
        }

        // Convert to U8Target
        let result = result.into_iter().map(U8Target).collect::<Vec<U8Target>>();
        self.split_bytes_cache.insert(target, result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use itertools::Itertools;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::*;
    use crate::types::config::{C, F};

    #[test]
    fn split_bytes_test() -> Result<()> {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::new();

        let target = builder.add_virtual_target();

        let bytes = builder.split_bytes(target, 4);
        bytes
            .iter()
            .zip_eq([76, 198, 250, 255].map(F::from_canonical_u64).iter())
            .for_each(|(b, v)| {
                pw.set_target(b.0, *v).unwrap();
            });
        assert_eq!(bytes.len(), 4);

        let bytes = builder.split_bytes(target, 8);

        bytes
            .iter()
            .zip_eq(
                [76, 198, 250, 255, 0, 0, 0, 0]
                    .map(F::from_canonical_u64)
                    .iter(),
            )
            .for_each(|(b, v)| {
                pw.set_target(b.0, *v).unwrap();
            });
        assert_eq!(bytes.len(), 8);

        pw.set_target(target, F::from_canonical_u64((1_u64 << 32) - 342452))?;
        let circuit = builder.build::<C>();

        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof.clone())
    }

    #[test]
    fn split_bytes_cache_test() -> Result<()> {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::new();

        let target = builder.add_virtual_target();

        let bytes = builder.split_bytes(target, 4);
        bytes
            .iter()
            .zip_eq([76, 198, 250, 255].map(F::from_canonical_u64).iter())
            .for_each(|(b, v)| {
                pw.set_target(b.0, *v).unwrap();
            });
        assert_eq!(bytes.len(), 4);

        let bytes = builder.split_bytes(target, 8);
        bytes
            .iter()
            .zip_eq(
                [76, 198, 250, 255, 0, 0, 0, 0]
                    .map(F::from_canonical_u64)
                    .iter(),
            )
            .for_each(|(b, v)| {
                pw.set_target(b.0, *v).unwrap();
            });
        assert_eq!(bytes.len(), 8);

        let bytes2 = builder.split_bytes(target, 8);
        assert_eq!(bytes, bytes2);

        pw.set_target(target, F::from_canonical_u64((1_u64 << 32) - 342452))?;
        let circuit = builder.build::<C>();

        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof.clone())
    }

    #[test]
    #[should_panic(expected = "was set twice with different values")]
    fn split_bytes_fail_test() {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
        let mut pw = PartialWitness::new();

        let target = builder.add_virtual_target();

        let _ = builder.split_bytes(target, 3);

        pw.set_target(target, F::from_canonical_u64((1_u64 << 32) - 342452))
            .unwrap();
        let circuit = builder.build::<C>();

        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof.clone()).unwrap();
    }
}
