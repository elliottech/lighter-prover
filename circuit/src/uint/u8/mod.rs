// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{Witness, WitnessWrite};

use crate::builder::Builder;

#[derive(
    Clone, Copy, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
)]
pub struct U8Target(pub Target);

pub trait CircuitBuilderU8<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_u8_target_unsafe(&mut self) -> U8Target;
    fn add_virtual_u8_targets_unsafe(&mut self, n: usize) -> Vec<U8Target>;
    fn add_virtual_public_u8_targets_unsafe(&mut self, n: usize) -> Vec<U8Target>;

    fn add_virtual_u8_target_safe(&mut self) -> U8Target;
    fn add_virtual_u8_targets_safe(&mut self, n: usize) -> Vec<U8Target>;
    fn add_virtual_public_u8_targets_safe(&mut self, n: usize) -> Vec<U8Target>;

    fn register_public_u8_input(&mut self, target: U8Target);
    fn register_public_u8_inputs(&mut self, targets: &[U8Target]);

    fn zero_u8(&mut self) -> U8Target;
    fn one_u8(&mut self) -> U8Target;

    fn select_u8(&mut self, cond: BoolTarget, a: U8Target, b: U8Target) -> U8Target;

    fn connect_u8(&mut self, a: U8Target, b: U8Target);

    fn constant_u8(&mut self, value: u8) -> U8Target;
    fn constant_u8s(&mut self, n: &[u8]) -> Vec<U8Target>;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderU8<F, D> for Builder<F, D> {
    fn add_virtual_u8_target_unsafe(&mut self) -> U8Target {
        U8Target(self.add_virtual_target())
    }

    fn add_virtual_u8_targets_unsafe(&mut self, n: usize) -> Vec<U8Target> {
        self.add_virtual_targets(n)
            .into_iter()
            .map(U8Target)
            .collect()
    }

    fn add_virtual_public_u8_targets_unsafe(&mut self, n: usize) -> Vec<U8Target> {
        let targets = self
            .add_virtual_targets(n)
            .into_iter()
            .map(U8Target)
            .collect::<Vec<U8Target>>();
        for target in &targets {
            self.register_public_input(target.0);
        }
        targets
    }

    fn add_virtual_u8_target_safe(&mut self) -> U8Target {
        let target = U8Target(self.add_virtual_target());
        self.register_range_check(target.0, 8);
        target
    }

    fn add_virtual_u8_targets_safe(&mut self, n: usize) -> Vec<U8Target> {
        let targets: Vec<U8Target> = self
            .add_virtual_targets(n)
            .into_iter()
            .map(U8Target)
            .collect();
        for target in &targets {
            self.register_range_check(target.0, 8);
        }
        targets
    }

    fn add_virtual_public_u8_targets_safe(&mut self, n: usize) -> Vec<U8Target> {
        let targets: Vec<U8Target> = self
            .add_virtual_targets(n)
            .into_iter()
            .map(U8Target)
            .collect();
        for target in &targets {
            self.register_public_input(target.0);
            self.register_range_check(target.0, 8);
        }
        targets
    }

    fn register_public_u8_input(&mut self, target: U8Target) {
        self.register_public_input(target.0);
    }

    fn register_public_u8_inputs(&mut self, targets: &[U8Target]) {
        for target in targets {
            self.register_public_input(target.0);
        }
    }

    fn zero_u8(&mut self) -> U8Target {
        U8Target(self.zero())
    }

    fn one_u8(&mut self) -> U8Target {
        U8Target(self.one())
    }

    fn select_u8(&mut self, cond: BoolTarget, a: U8Target, b: U8Target) -> U8Target {
        U8Target(self.select(cond, a.0, b.0))
    }

    fn connect_u8(&mut self, a: U8Target, b: U8Target) {
        self.connect(a.0, b.0);
    }

    fn constant_u8(&mut self, value: u8) -> U8Target {
        U8Target(self.constant(F::from_canonical_u8(value)))
    }

    fn constant_u8s(&mut self, n: &[u8]) -> Vec<U8Target> {
        self.constants(
            &n.iter()
                .map(|&x| F::from_canonical_u8(x))
                .collect::<Vec<F>>(),
        )
        .iter()
        .map(|&x| U8Target(x))
        .collect()
    }
}

pub trait WitnessU8<F: PrimeField64>: Witness<F> {
    fn set_u8_target(&mut self, target: U8Target, value: u8) -> Result<()>;
    fn get_u8_target(&self, target: U8Target) -> u8;
}

impl<T: Witness<F>, F: PrimeField64> WitnessU8<F> for T {
    fn set_u8_target(&mut self, target: U8Target, value: u8) -> Result<()> {
        self.set_target(target.0, F::from_canonical_u8(value))
    }

    fn get_u8_target(&self, target: U8Target) -> u8 {
        let x_u64 = self.get_target(target.0).to_canonical_u64();
        u8::try_from(x_u64).unwrap_or_else(|_| panic!("Value out of range for u8: {}", x_u64))
    }
}

pub trait GeneratedValuesU8<F: Field> {
    fn set_u8_target(&mut self, target: U8Target, value: u8) -> Result<()>;
}

impl<F: Field> GeneratedValuesU8<F> for GeneratedValues<F> {
    fn set_u8_target(&mut self, target: U8Target, value: u8) -> Result<()> {
        self.set_target(target.0, F::from_canonical_u8(value))
    }
}
