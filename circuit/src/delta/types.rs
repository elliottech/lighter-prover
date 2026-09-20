// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS};
use plonky2::iop::target::Target;

use crate::circuit_logger::CircuitBuilderLogging;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::types::config::Builder;
use crate::types::constants::ACCOUNT_MERKLE_LEVELS;

#[derive(Debug)]
pub struct DeltaPublicInputTarget {
    pub account_index: Target,
    pub path_matrix: [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; 2],
    pub market_delta_hash: HashOutTarget,
    pub remaining_market_delta_count: Target,
    pub evaluation_point: QuinticExtensionTarget,
}

impl DeltaPublicInputTarget {
    pub const ACCOUNT_INDEX_INDEX: usize = 0;
    pub const PATH_MATRIX_START_INDEX: usize = Self::ACCOUNT_INDEX_INDEX + 1;
    const PATH_MATRIX_SINGLE_SIZE: usize = ACCOUNT_MERKLE_LEVELS * NUM_HASH_OUT_ELTS;
    pub const MARKET_DELTA_HASH_START_INDEX: usize =
        Self::PATH_MATRIX_START_INDEX + 2 * Self::PATH_MATRIX_SINGLE_SIZE;
    pub const REMAINING_MARKET_DELTA_COUNT_INDEX: usize =
        Self::MARKET_DELTA_HASH_START_INDEX + NUM_HASH_OUT_ELTS;
    pub const EVALUATION_POINT_START_INDEX: usize = Self::REMAINING_MARKET_DELTA_COUNT_INDEX + 1;

    pub const DELTA_PUB_IN_SIZE: usize = Self::EVALUATION_POINT_START_INDEX + 5;

    pub fn new_public(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_public_input(),
            path_matrix: core::array::from_fn(|_| {
                core::array::from_fn(|_| builder.add_virtual_hash_public_input())
            }),
            market_delta_hash: builder.add_virtual_hash_public_input(),
            remaining_market_delta_count: builder.add_virtual_public_input(),
            evaluation_point: builder.add_virtual_public_quintic_ext_target(),
        }
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert_eq!(
            pis.len(),
            Self::DELTA_PUB_IN_SIZE,
            "Invalid number of public inputs for DeltaPublicInputTarget"
        );

        Self {
            account_index: pis[Self::ACCOUNT_INDEX_INDEX],
            path_matrix: core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    HashOutTarget::from_vec(
                        pis[Self::PATH_MATRIX_START_INDEX
                            + (i * ACCOUNT_MERKLE_LEVELS + j) * NUM_HASH_OUT_ELTS
                            ..Self::PATH_MATRIX_START_INDEX
                                + (i * ACCOUNT_MERKLE_LEVELS + j + 1) * NUM_HASH_OUT_ELTS]
                            .to_vec(),
                    )
                })
            }),
            market_delta_hash: HashOutTarget::from_vec(
                pis[Self::MARKET_DELTA_HASH_START_INDEX
                    ..Self::MARKET_DELTA_HASH_START_INDEX + NUM_HASH_OUT_ELTS]
                    .to_vec(),
            ),
            remaining_market_delta_count: pis[Self::REMAINING_MARKET_DELTA_COUNT_INDEX],
            evaluation_point: QuinticExtensionTarget([
                pis[Self::EVALUATION_POINT_START_INDEX],
                pis[Self::EVALUATION_POINT_START_INDEX + 1],
                pis[Self::EVALUATION_POINT_START_INDEX + 2],
                pis[Self::EVALUATION_POINT_START_INDEX + 3],
                pis[Self::EVALUATION_POINT_START_INDEX + 4],
            ]),
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_arr(
            &self.evaluation_point.0,
            &format!("DELTA PUBLIC INPUT - {tag} - evaluation_point"),
        );
        builder.println_hash_out(
            &self.market_delta_hash,
            &format!("DELTA PUBLIC INPUT - {tag} - market_delta_hash"),
        );
        builder.println(
            self.remaining_market_delta_count,
            &format!("DELTA PUBLIC INPUT - {tag} - remaining_market_delta_count"),
        );
        builder.println(
            self.account_index,
            &format!("DELTA PUBLIC INPUT - {tag} - account_index"),
        );
        builder.println_hash_out(
            &self.path_matrix[0][ACCOUNT_MERKLE_LEVELS - 1],
            &format!("DELTA PUBLIC INPUT - {tag} - path_matrix[0] upper"),
        );
        builder.println_hash_out(
            &self.path_matrix[1][ACCOUNT_MERKLE_LEVELS - 1],
            &format!("DELTA PUBLIC INPUT - {tag} - path_matrix[1] upper"),
        );
    }
}

#[derive(Debug)]
pub struct DeltaPublicOutputTarget {
    pub account_index: Target,
    pub path_matrix: [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; 2],
    pub market_delta_hash: HashOutTarget,
    pub remaining_market_delta_count: Target,
    pub evaluation: QuinticExtensionTarget,
    pub degree: Target,
}

impl DeltaPublicOutputTarget {
    pub const ACCOUNT_INDEX_INDEX: usize = 0;
    pub const PATH_MATRIX_START_INDEX: usize = Self::ACCOUNT_INDEX_INDEX + 1;
    const PATH_MATRIX_SINGLE_SIZE: usize = ACCOUNT_MERKLE_LEVELS * NUM_HASH_OUT_ELTS;
    pub const MARKET_DELTA_HASH_START_INDEX: usize =
        Self::PATH_MATRIX_START_INDEX + 2 * Self::PATH_MATRIX_SINGLE_SIZE;
    pub const REMAINING_MARKET_DELTA_COUNT_INDEX: usize =
        Self::MARKET_DELTA_HASH_START_INDEX + NUM_HASH_OUT_ELTS;
    pub const EVALUATION_START_INDEX: usize = Self::REMAINING_MARKET_DELTA_COUNT_INDEX + 1;
    pub const DEGREE_INDEX: usize = Self::EVALUATION_START_INDEX + 5;

    pub const DELTA_PUB_OUT_SIZE: usize = Self::DEGREE_INDEX + 1;

    pub fn new_public(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_public_input(),
            path_matrix: core::array::from_fn(|_| {
                core::array::from_fn(|_| builder.add_virtual_hash_public_input())
            }),
            market_delta_hash: builder.add_virtual_hash_public_input(),
            remaining_market_delta_count: builder.add_virtual_public_input(),
            evaluation: builder.add_virtual_public_quintic_ext_target(),
            degree: builder.add_virtual_public_input(),
        }
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert_eq!(
            pis.len(),
            Self::DELTA_PUB_OUT_SIZE,
            "Invalid number of public inputs for DeltaPublicOutputTarget"
        );

        Self {
            account_index: pis[Self::ACCOUNT_INDEX_INDEX],
            path_matrix: core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    HashOutTarget::from_vec(
                        pis[Self::PATH_MATRIX_START_INDEX
                            + (i * ACCOUNT_MERKLE_LEVELS + j) * NUM_HASH_OUT_ELTS
                            ..Self::PATH_MATRIX_START_INDEX
                                + (i * ACCOUNT_MERKLE_LEVELS + j + 1) * NUM_HASH_OUT_ELTS]
                            .to_vec(),
                    )
                })
            }),
            market_delta_hash: HashOutTarget::from_vec(
                pis[Self::MARKET_DELTA_HASH_START_INDEX
                    ..Self::MARKET_DELTA_HASH_START_INDEX + NUM_HASH_OUT_ELTS]
                    .to_vec(),
            ),
            remaining_market_delta_count: pis[Self::REMAINING_MARKET_DELTA_COUNT_INDEX],
            evaluation: QuinticExtensionTarget([
                pis[Self::EVALUATION_START_INDEX],
                pis[Self::EVALUATION_START_INDEX + 1],
                pis[Self::EVALUATION_START_INDEX + 2],
                pis[Self::EVALUATION_START_INDEX + 3],
                pis[Self::EVALUATION_START_INDEX + 4],
            ]),
            degree: pis[Self::DEGREE_INDEX],
        }
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_arr(
            &self.evaluation.0,
            &format!("DELTA PUBLIC OUTPUT - {tag} - evaluation"),
        );
        builder.println_hash_out(
            &self.market_delta_hash,
            &format!("DELTA PUBLIC OUTPUT - {tag} - market_delta_hash"),
        );
        builder.println(
            self.remaining_market_delta_count,
            &format!("DELTA PUBLIC OUTPUT - {tag} - remaining_market_delta_count"),
        );
        builder.println(
            self.account_index,
            &format!("DELTA PUBLIC OUTPUT - {tag} - account_index"),
        );
        builder.println(
            self.degree,
            &format!("DELTA PUBLIC OUTPUT - {tag} - degree"),
        );
        builder.println_hash_out(
            &self.path_matrix[0][ACCOUNT_MERKLE_LEVELS - 1],
            &format!("DELTA PUBLIC OUTPUT - {tag} - path_matrix[0] upper"),
        );
        builder.println_hash_out(
            &self.path_matrix[1][ACCOUNT_MERKLE_LEVELS - 1],
            &format!("DELTA PUBLIC OUTPUT - {tag} - path_matrix[1] upper"),
        );
    }
}

#[derive(Debug)]
pub struct AggregatedDeltaTarget {
    pub account_index: Target,
    pub path_matrix: [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; 2],

    pub market_delta_hash: HashOutTarget,
    pub remaining_market_delta_count: Target,

    pub evaluation_point: QuinticExtensionTarget,
    pub evaluation: QuinticExtensionTarget,
    pub degree: Target,
}

impl AggregatedDeltaTarget {
    pub const ACCOUNT_INDEX_INDEX: usize = 0;
    pub const PATH_MATRIX_START_INDEX: usize = Self::ACCOUNT_INDEX_INDEX + 1;
    pub const MARKET_DELTA_HASH_START_INDEX: usize =
        Self::PATH_MATRIX_START_INDEX + 2 * ACCOUNT_MERKLE_LEVELS * NUM_HASH_OUT_ELTS;
    pub const REMAINING_MARKET_DELTA_COUNT_INDEX: usize =
        Self::MARKET_DELTA_HASH_START_INDEX + NUM_HASH_OUT_ELTS;
    pub const EVALUATION_POINT_START_INDEX: usize = Self::REMAINING_MARKET_DELTA_COUNT_INDEX + 1;
    pub const EVALUATION_START_INDEX: usize = Self::EVALUATION_POINT_START_INDEX + 5;
    pub const DEGREE_INDEX: usize = Self::EVALUATION_START_INDEX + 5;

    pub const END_INDEX: usize = Self::DEGREE_INDEX + 1;

    pub fn new_public(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_public_input(),
            path_matrix: core::array::from_fn(|_| {
                core::array::from_fn(|_| builder.add_virtual_hash_public_input())
            }),
            market_delta_hash: builder.add_virtual_hash_public_input(),
            remaining_market_delta_count: builder.add_virtual_public_input(),
            evaluation_point: builder.add_virtual_public_quintic_ext_target(),
            evaluation: builder.add_virtual_public_quintic_ext_target(),
            degree: builder.add_virtual_public_input(),
        }
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        Self {
            account_index: pis[Self::ACCOUNT_INDEX_INDEX],
            path_matrix: core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    HashOutTarget::from_vec(
                        pis[Self::PATH_MATRIX_START_INDEX
                            + (i * ACCOUNT_MERKLE_LEVELS + j) * NUM_HASH_OUT_ELTS
                            ..Self::PATH_MATRIX_START_INDEX
                                + (i * ACCOUNT_MERKLE_LEVELS + j + 1) * NUM_HASH_OUT_ELTS]
                            .to_vec(),
                    )
                })
            }),
            market_delta_hash: HashOutTarget::from_vec(
                pis[Self::MARKET_DELTA_HASH_START_INDEX
                    ..Self::MARKET_DELTA_HASH_START_INDEX + NUM_HASH_OUT_ELTS]
                    .to_vec(),
            ),
            remaining_market_delta_count: pis[Self::REMAINING_MARKET_DELTA_COUNT_INDEX],
            evaluation_point: QuinticExtensionTarget([
                pis[Self::EVALUATION_POINT_START_INDEX],
                pis[Self::EVALUATION_POINT_START_INDEX + 1],
                pis[Self::EVALUATION_POINT_START_INDEX + 2],
                pis[Self::EVALUATION_POINT_START_INDEX + 3],
                pis[Self::EVALUATION_POINT_START_INDEX + 4],
            ]),
            evaluation: QuinticExtensionTarget([
                pis[Self::EVALUATION_START_INDEX],
                pis[Self::EVALUATION_START_INDEX + 1],
                pis[Self::EVALUATION_START_INDEX + 2],
                pis[Self::EVALUATION_START_INDEX + 3],
                pis[Self::EVALUATION_START_INDEX + 4],
            ]),
            degree: pis[Self::DEGREE_INDEX],
        }
    }

    pub fn connect(&self, builder: &mut Builder, other: &Self) {
        builder.connect(self.account_index, other.account_index);
        for i in 0..self.path_matrix.len() {
            for j in 0..self.path_matrix[i].len() {
                builder.connect_hashes(self.path_matrix[i][j], other.path_matrix[i][j]);
            }
        }
        builder.connect_hashes(self.market_delta_hash, other.market_delta_hash);
        builder.connect(
            self.remaining_market_delta_count,
            other.remaining_market_delta_count,
        );
        builder.connect_quintic_ext(self.evaluation_point, other.evaluation_point);
        builder.connect_quintic_ext(self.evaluation, other.evaluation);
        builder.connect(self.degree, other.degree);
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println_arr(
            &self.evaluation_point.0,
            &format!("AGGREGATED DELTA - {tag} - evaluation_point"),
        );
        builder.println(
            self.account_index,
            &format!("AGGREGATED DELTA - {tag} - account_index"),
        );
        builder.println(self.degree, &format!("AGGREGATED DELTA - {tag} - degree"));
        builder.println_arr(
            &self.evaluation.0,
            &format!("AGGREGATED DELTA - {tag} - evaluation"),
        );
        builder.println_hash_out(
            &self.market_delta_hash,
            &format!("AGGREGATED DELTA - {tag} - market_delta_hash"),
        );
        builder.println(
            self.remaining_market_delta_count,
            &format!("AGGREGATED DELTA - {tag} - remaining_market_delta_count"),
        );
        builder.println_hash_out(
            &self.path_matrix[0][ACCOUNT_MERKLE_LEVELS - 1],
            &format!("AGGREGATED DELTA - {tag} - path_matrix[0] upper"),
        );
        builder.println_hash_out(
            &self.path_matrix[1][ACCOUNT_MERKLE_LEVELS - 1],
            &format!("AGGREGATED DELTA - {tag} - path_matrix[1] upper"),
        );
    }
}
