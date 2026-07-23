// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::Target;

use crate::bigint::bigint::{BigIntTarget, SignTarget};
use crate::bigint::biguint::BigUintTarget;
use crate::keccak::keccak::{CircuitBuilderKeccak, KeccakOutputTarget};
use crate::types::config::Builder;
use crate::types::constants::{
    KECCAK_HASH_OUT_BYTE_SIZE, ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE, POSITION_LIST_SIZE,
};
use crate::types::market_details::{PublicMarketDetailsTarget, connect_public_market_details};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;

#[derive(Debug)]
/// In circuit represantion of [`crate::block::BlockWitness`]
pub struct BlockWitnessTarget {
    pub block_number: Target,
    pub created_at: Target, // 48 bits

    pub old_state_root: HashOutTarget,
    pub new_validium_root: HashOutTarget,
    pub new_state_root: HashOutTarget,

    pub old_account_delta_tree_root: HashOutTarget,
    pub new_account_delta_tree_root: HashOutTarget,

    pub on_chain_operations_count: Target,
    pub on_chain_operations_pub_data: Vec<[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    pub priority_operations_count: Target,
    pub old_prefix_priority_operation_hash: KeccakOutputTarget,
    pub new_prefix_priority_operation_hash: KeccakOutputTarget,

    pub new_public_market_details: [PublicMarketDetailsTarget; POSITION_LIST_SIZE],
}

impl BlockWitnessTarget {
    pub fn new_public(
        builder: &mut Builder,
        on_chain_operations_limit: usize,
        new_public_market_details: [PublicMarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> Self {
        Self {
            block_number: builder.add_virtual_public_input(),
            created_at: builder.add_virtual_public_input(),
            old_state_root: builder.add_virtual_hash_public_input(),
            new_validium_root: builder.add_virtual_hash_public_input(),
            new_state_root: builder.add_virtual_hash_public_input(),
            old_account_delta_tree_root: builder.add_virtual_hash_public_input(),
            new_account_delta_tree_root: builder.add_virtual_hash_public_input(),
            new_public_market_details: {
                for market_details in new_public_market_details.iter() {
                    market_details.register_public_input(builder);
                }
                new_public_market_details
            },
            on_chain_operations_count: builder.add_virtual_public_input(),
            on_chain_operations_pub_data: (0..on_chain_operations_limit)
                .map(|_| {
                    builder
                        .add_virtual_public_u8_targets_unsafe(
                            ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE,
                        )
                        .try_into()
                        .unwrap()
                }) // safe because it is connected to public witness from tx chain circuit which range-checked its output
                .collect(),
            priority_operations_count: builder.add_virtual_public_input(),
            old_prefix_priority_operation_hash: builder
                .add_virtual_keccak_output_public_input_safe(),
            new_prefix_priority_operation_hash: builder
                .add_virtual_keccak_output_public_input_unsafe(), // safe because it is output of the keccak circuit which range-checked its output
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.block_number);
        builder.register_public_input(self.created_at);
        builder.register_public_hashout(self.old_state_root);
        builder.register_public_hashout(self.new_validium_root);
        builder.register_public_hashout(self.new_state_root);
        builder.register_public_hashout(self.old_account_delta_tree_root);
        builder.register_public_hashout(self.new_account_delta_tree_root);

        for market_details in self.new_public_market_details.iter() {
            market_details.register_public_input(builder);
        }

        builder.register_public_input(self.on_chain_operations_count);

        for pub_data in &self.on_chain_operations_pub_data {
            for &byte in pub_data {
                builder.register_public_u8_input(byte);
            }
        }

        builder.register_public_input(self.priority_operations_count);

        for i in 0..KECCAK_HASH_OUT_BYTE_SIZE {
            builder.register_public_u8_input(self.old_prefix_priority_operation_hash[i]);
            builder.register_public_u8_input(self.new_prefix_priority_operation_hash[i]);
        }
    }

    /// Similar to [`crate::block::BlockWitness::from_public_inputs`], parses proof target.
    /// Returns the number of public inputs.
    /// Assumes _on_chain_operations_limit and _priority_ops_limit are 1.
    pub fn from_public_inputs(
        pis: &[Target],
        _on_chain_operations_limit: usize,
        _priority_ops_limit: usize,
    ) -> (Self, usize) {
        let new_public_market_details_index = 22;

        let on_chain_operations_count_index =
            new_public_market_details_index + POSITION_LIST_SIZE * 5;
        let on_chain_operations_pub_data_index = on_chain_operations_count_index + 1;

        let priority_operations_count_index =
            on_chain_operations_pub_data_index + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;
        let old_prefix_priority_operation_hash_index = priority_operations_count_index + 1;
        let new_prefix_priority_operation_hash_index =
            old_prefix_priority_operation_hash_index + KECCAK_HASH_OUT_BYTE_SIZE;
        let total_pis_size = new_prefix_priority_operation_hash_index + KECCAK_HASH_OUT_BYTE_SIZE;

        assert!(
            pis.len() >= total_pis_size,
            "Expected {} public inputs, but got {}",
            total_pis_size,
            pis.len()
        );
        let pis: Vec<Target> = pis.iter().copied().take(total_pis_size).collect();

        (
            Self {
                block_number: pis[0],
                created_at: pis[1],
                old_state_root: HashOutTarget {
                    elements: [pis[2], pis[3], pis[4], pis[5]],
                },
                new_validium_root: HashOutTarget {
                    elements: [pis[6], pis[7], pis[8], pis[9]],
                },
                new_state_root: HashOutTarget {
                    elements: [pis[10], pis[11], pis[12], pis[13]],
                },
                old_account_delta_tree_root: HashOutTarget {
                    elements: [pis[14], pis[15], pis[16], pis[17]],
                },
                new_account_delta_tree_root: HashOutTarget {
                    elements: [pis[18], pis[19], pis[20], pis[21]],
                },

                new_public_market_details: pis
                    [new_public_market_details_index..on_chain_operations_count_index]
                    .chunks(5)
                    .map(|chunk| PublicMarketDetailsTarget {
                        funding_rate_prefix_sum: BigIntTarget {
                            sign: SignTarget::new_unsafe(chunk[0]),
                            abs: BigUintTarget {
                                limbs: vec![U32Target(chunk[1]), U32Target(chunk[2])],
                            },
                        },
                        mark_price: chunk[3],
                        quote_multiplier: chunk[4],
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),

                on_chain_operations_count: pis[on_chain_operations_count_index],
                on_chain_operations_pub_data: pis
                    [on_chain_operations_pub_data_index..priority_operations_count_index]
                    .iter()
                    .collect::<Vec<_>>()
                    .chunks(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE)
                    .map(|chunk| core::array::from_fn(|i| U8Target(*chunk[i])))
                    .collect::<Vec<_>>(),

                priority_operations_count: pis[priority_operations_count_index],
                old_prefix_priority_operation_hash: core::array::from_fn(|i| {
                    U8Target(pis[old_prefix_priority_operation_hash_index + i])
                }),
                new_prefix_priority_operation_hash: core::array::from_fn(|i| {
                    U8Target(pis[new_prefix_priority_operation_hash_index + i])
                }),
            },
            total_pis_size,
        )
    }

    pub fn connect_block_witness(&self, builder: &mut Builder, other: &Self) {
        builder.connect(self.block_number, other.block_number);
        builder.connect(self.created_at, other.created_at);

        builder.connect_hashes(self.old_state_root, other.old_state_root);
        builder.connect_hashes(self.new_validium_root, other.new_validium_root);
        builder.connect_hashes(self.new_state_root, other.new_state_root);
        builder.connect_hashes(
            self.old_account_delta_tree_root,
            other.old_account_delta_tree_root,
        );
        builder.connect_hashes(
            self.new_account_delta_tree_root,
            other.new_account_delta_tree_root,
        );

        builder.connect(
            self.on_chain_operations_count,
            other.on_chain_operations_count,
        );
        for (i, pub_data) in self.on_chain_operations_pub_data.iter().enumerate() {
            for (j, byte) in pub_data.iter().enumerate() {
                builder.connect_u8(*byte, other.on_chain_operations_pub_data[i][j]);
            }
        }

        builder.connect(
            self.priority_operations_count,
            other.priority_operations_count,
        );
        for (i, old_prefix_priority_operation_hash) in
            self.old_prefix_priority_operation_hash.iter().enumerate()
        {
            builder.connect_u8(
                *old_prefix_priority_operation_hash,
                other.old_prefix_priority_operation_hash[i],
            );
        }
        for (i, new_prefix_priority_operation_hash) in
            self.new_prefix_priority_operation_hash.iter().enumerate()
        {
            builder.connect_u8(
                *new_prefix_priority_operation_hash,
                other.new_prefix_priority_operation_hash[i],
            );
        }

        connect_public_market_details(
            builder,
            &self.new_public_market_details,
            &other.new_public_market_details,
        );
    }
}
