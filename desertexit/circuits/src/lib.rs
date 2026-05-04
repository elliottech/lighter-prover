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

#[macro_use(
    read_gate_impl,
    get_gate_tag_impl,
    read_generator_impl,
    get_generator_tag_impl
)]
pub extern crate plonky2;

pub mod circuit_serializer;
pub mod deserializers;
pub mod inner_circuit;
pub mod outer_circuit;
pub mod pubdata_account;
pub mod pubdata_market;
