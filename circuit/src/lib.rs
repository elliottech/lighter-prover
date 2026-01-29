// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::module_inception)]
#![allow(clippy::identity_op)]
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::get_first)]

#[macro_use(
    read_gate_impl,
    get_gate_tag_impl,
    read_generator_impl,
    get_generator_tag_impl
)]
// Skipping impl_gate_serializer and impl_generator_serializer because we need to re-define here
pub extern crate plonky2;

#[macro_use]
extern crate lazy_static;

pub mod apply_trade;
pub mod bigint;
pub mod blob;
pub mod block;
pub mod block_constraints;
pub mod block_pre_execution;
pub mod block_pre_execution_constraints;
pub mod block_tx;
pub mod block_tx_chain;
pub mod block_tx_chain_constraints;
pub mod block_tx_constraints;
pub mod bool_utils;
pub mod builder;
pub mod byte;
pub mod circuit_logger;
pub mod circuit_serializer;
pub mod comparison;
pub mod delta;
pub mod deserializers;
pub mod ecdsa;
pub mod eddsa;
pub mod hash_utils;
pub mod hints;
pub mod keccak;
pub mod liquidation;
pub mod matching_engine;
pub mod merkle_helpers;
pub mod nonnative;
pub mod order_book_tree_helpers;
pub mod poseidon2;
pub mod poseidon_bn128;
pub mod quintuple;
pub mod recursion;
pub mod signed;
pub mod transactions;
pub mod tx;
pub mod tx_constraints;
pub mod tx_interface;
pub mod types;
pub mod uint;
pub mod utils;
