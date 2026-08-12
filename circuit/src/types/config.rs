// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::fri::FriConfig;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

use crate::builder::Builder as CircuitBuilder;

pub const D: usize = 2;
pub type C = Poseidon2GoldilocksConfig;
pub type F = GoldilocksField;
pub type Builder = CircuitBuilder<F, D>;
pub type PoseidonHash = <C as GenericConfig<D>>::InnerHasher;

// To be used by const fields
#[inline]
pub const fn const_f(val: u64) -> F {
    GoldilocksField(val)
}

// 32-bit limb sizes
pub const BIG_U32_LIMBS: usize = 1;
pub const BIG_U64_LIMBS: usize = 2;
pub const BIG_U96_LIMBS: usize = 3;
pub const BIG_U128_LIMBS: usize = 4;
pub const BIG_U160_LIMBS: usize = 5;
pub const BIG_U192_LIMBS: usize = 6;
pub const BIG_U256_LIMBS: usize = 8;

// 16-bit limb size
pub const BIGU16_U32_LIMBS: usize = 2;
pub const BIGU16_U64_LIMBS: usize = 4;
pub const BIGU16_U96_LIMBS: usize = 6;
pub const BIGU16_U112_LIMBS: usize = 7;

pub const CIRCUIT_CONFIG: CircuitConfig = CircuitConfig {
    num_wires: 136,
    num_routed_wires: 120,
    num_constants: 2,
    use_base_arithmetic_gate: true,
    security_bits: 100,
    num_challenges: 2,
    zero_knowledge: false,
    max_quotient_degree_factor: 8,
    fri_config: FriConfig {
        rate_bits: 3,
        cap_height: 4,
        proof_of_work_bits: 16,
        reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
        num_query_rounds: 28,
    },
    optimization_flags: (1 << 0) /*+ (1 << 1) */+ (1 << 2) + (1 << 3) + (1 << 4) + (1 << 5),
};

pub const BLOCK_TX_CHAIN_CIRCUIT_CONFIG: CircuitConfig = CircuitConfig {
    num_wires: 136,
    num_routed_wires: 40,
    ..CIRCUIT_CONFIG
};

pub const BLOCK_CIRCUIT_CONFIG: CircuitConfig = CircuitConfig {
    num_wires: 136,
    num_routed_wires: 40,
    ..CIRCUIT_CONFIG
};

pub const OUTER_WRAPPER_CONFIG: CircuitConfig = CircuitConfig {
    num_wires: 136,
    num_routed_wires: 80,
    num_constants: 2,
    use_base_arithmetic_gate: true,
    security_bits: 100,
    num_challenges: 2,
    zero_knowledge: false,
    max_quotient_degree_factor: 8,
    fri_config: FriConfig {
        rate_bits: 3,
        cap_height: 4,
        proof_of_work_bits: 16,
        reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
        num_query_rounds: 28,
    },
    optimization_flags: 0,
};
