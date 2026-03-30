// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;
use std::fmt;

use num::BigInt;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::Target;

use crate::bigint::bigint::{BigIntTarget, SignTarget};
use crate::bigint::biguint::BigUintTarget;
use crate::types::approve_integrator::{
    APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN, ApproveIntegratorMessage, ApproveIntegratorMessageTarget,
};
use crate::types::change_pub_key::{
    CHANGE_PK_PUBLIC_INPUTS_LEN, ChangePubKeyMessage, ChangePubKeyMessageTarget,
};
use crate::types::config::Builder;
use crate::types::constants::{
    MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX, ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE,
    POSITION_LIST_SIZE,
};
use crate::types::market_details::{
    PublicMarketDetails, PublicMarketDetailsTarget, connect_public_market_details,
};
use crate::types::transfer::{TRANSFER_PUBLIC_INPUTS_LEN, TransferMessage, TransferMessageTarget};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::U32Target;

#[derive(Clone)]
pub struct BlockTxChainWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub block_number: u64,
    pub created_at: i64,

    pub old_state_root: HashOut<F>,
    pub new_validium_root: HashOut<F>,
    pub new_state_root: HashOut<F>,
    pub new_account_delta_tree_root: HashOut<F>,

    pub change_pub_key_message: ChangePubKeyMessage<F>,
    pub transfer_message: TransferMessage,
    pub approve_integrator_message: ApproveIntegratorMessage,

    pub on_chain_operations_count: u64,
    pub on_chain_operations_pub_data: Vec<[u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    pub priority_operations_count: u64,
    pub priority_operations_pub_data: [u8; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],

    pub new_public_market_details: [PublicMarketDetails; POSITION_LIST_SIZE],
}

impl<F> fmt::Debug for BlockTxChainWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut on_chain_pub_data = vec![];

        self.on_chain_operations_pub_data
            .iter()
            .for_each(|pub_data| {
                on_chain_pub_data.push(hex::encode(pub_data));
            });

        let priority_operations_pub_data = hex::encode(self.priority_operations_pub_data);

        let mut new_market_details = HashMap::<usize, PublicMarketDetails>::new();
        self.new_public_market_details
            .iter()
            .filter(|market_detail| !market_detail.is_empty())
            .enumerate()
            .for_each(|(index, market_details)| {
                new_market_details.insert(index, market_details.clone());
            });

        let new_public_market_details = serde_json::to_string(&new_market_details).unwrap();

        fmt.debug_struct("BlockTxChainWitness<F>")
            .field("block_number", &self.block_number)
            .field("created_at", &self.created_at)
            .field("old_state_root", &self.old_state_root)
            .field("new_validium_root", &self.new_validium_root)
            .field("new_state_root", &self.new_state_root)
            .field(
                "new_account_delta_tree_root",
                &self.new_account_delta_tree_root,
            )
            .field("on_chain_operations_count", &self.on_chain_operations_count)
            .field("on_chain_operations_pub_data", &on_chain_pub_data)
            .field("priority_operations_count", &self.priority_operations_count)
            .field(
                "priority_operations_pub_data",
                &priority_operations_pub_data,
            )
            .field("new_public_market_details", &new_public_market_details)
            .finish()
    }
}

impl<F> BlockTxChainWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    /// Parse public inputs from proof into BlockTxChainWitness
    pub fn from_public_inputs(public_inputs: &[F], _: usize, _: usize) -> Self {
        let new_public_market_details_index = 18;

        let change_pub_key_message_index = new_public_market_details_index + POSITION_LIST_SIZE * 5;
        let transfer_message_index = change_pub_key_message_index + CHANGE_PK_PUBLIC_INPUTS_LEN;
        let approve_integrator_message_index = transfer_message_index + TRANSFER_PUBLIC_INPUTS_LEN;

        let on_chain_operations_count_index =
            approve_integrator_message_index + APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN;
        let on_chain_operations_pub_data_index = on_chain_operations_count_index + 1;

        let priority_operations_count_index =
            on_chain_operations_pub_data_index + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;
        let priority_operations_pub_data = priority_operations_count_index + 1;

        Self {
            block_number: public_inputs[0].to_canonical_u64(),
            created_at: public_inputs[1].to_canonical_u64() as i64,

            old_state_root: HashOut::<F>::from([
                public_inputs[2],
                public_inputs[3],
                public_inputs[4],
                public_inputs[5],
            ]),
            new_validium_root: HashOut::<F>::from([
                public_inputs[6],
                public_inputs[7],
                public_inputs[8],
                public_inputs[9],
            ]),
            new_state_root: HashOut::<F>::from([
                public_inputs[10],
                public_inputs[11],
                public_inputs[12],
                public_inputs[13],
            ]),
            new_account_delta_tree_root: HashOut::<F>::from([
                public_inputs[14],
                public_inputs[15],
                public_inputs[16],
                public_inputs[17],
            ]),

            new_public_market_details: public_inputs
                [new_public_market_details_index..change_pub_key_message_index]
                .chunks(5)
                .map(|chunk| {
                    let mut funding_rate_prefix_sum_abs =
                        (chunk[1].to_canonical_u64() + (chunk[2].to_canonical_u64() << 32)) as i64;
                    if !chunk[0].is_one() && !chunk[0].is_zero() {
                        funding_rate_prefix_sum_abs *= -1;
                    }
                    PublicMarketDetails {
                        funding_rate_prefix_sum: BigInt::from(funding_rate_prefix_sum_abs),
                        mark_price: chunk[3].to_canonical_u64() as u32,
                        quote_multiplier: chunk[4].to_canonical_u64() as u32,
                    }
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),

            change_pub_key_message: ChangePubKeyMessage::from_public_inputs(
                &public_inputs[change_pub_key_message_index..transfer_message_index],
            ),
            transfer_message: TransferMessage::from_public_inputs(
                &public_inputs[transfer_message_index..approve_integrator_message_index],
            ),
            approve_integrator_message: ApproveIntegratorMessage::from_public_inputs(
                &public_inputs[approve_integrator_message_index..on_chain_operations_count_index],
            ),

            // On chain ops pub data
            on_chain_operations_count: public_inputs[on_chain_operations_count_index]
                .to_canonical_u64(),
            on_chain_operations_pub_data: public_inputs
                [on_chain_operations_pub_data_index..priority_operations_count_index]
                .iter()
                .collect::<Vec<_>>()
                .chunks(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE)
                .map(|chunk| core::array::from_fn(|i| chunk[i].to_canonical_u64() as u8))
                .collect::<Vec<_>>(),

            // Priority ops pub data
            priority_operations_count: public_inputs[priority_operations_count_index]
                .to_canonical_u64(),
            priority_operations_pub_data: core::array::from_fn(|index| {
                public_inputs[priority_operations_pub_data + index].to_canonical_u64() as u8
            }),
        }
    }
}

#[derive(Debug)]
/// In circuit represantion of [`crate::block::BlockTxChainWitness`]
pub struct BlockTxChainWitnessTarget {
    pub block_number: Target,
    pub created_at: Target, // 48 bits

    pub old_state_root: HashOutTarget,
    pub new_validium_root: HashOutTarget,
    pub new_state_root: HashOutTarget,

    // Initialized in cyclic_base_proof with a block witness's old_account_delta_tree_root,
    // but represents the new_account_delta_tree_root of the previously executed cyclic group.
    pub new_account_delta_tree_root: HashOutTarget,

    pub change_pub_key_message: ChangePubKeyMessageTarget,
    pub transfer_message: TransferMessageTarget,
    pub approve_integrator_message: ApproveIntegratorMessageTarget,

    pub on_chain_operations_count: Target,
    pub on_chain_operations_pub_data: Vec<[U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    pub priority_operations_count: Target,
    pub priority_operations_pub_data: [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],

    pub new_public_market_details: [PublicMarketDetailsTarget; POSITION_LIST_SIZE],
}

impl BlockTxChainWitnessTarget {
    pub fn new_public(builder: &mut Builder, on_chain_operations_limit: usize) -> Self {
        Self {
            block_number: builder.add_virtual_public_input(),
            created_at: builder.add_virtual_public_input(),
            old_state_root: builder.add_virtual_hash_public_input(),
            new_validium_root: builder.add_virtual_hash_public_input(),
            new_state_root: builder.add_virtual_hash_public_input(),
            new_account_delta_tree_root: builder.add_virtual_hash_public_input(),
            new_public_market_details: core::array::from_fn(|_| {
                PublicMarketDetailsTarget::new_public(builder)
            }),
            change_pub_key_message: ChangePubKeyMessageTarget::new_public(builder),
            transfer_message: TransferMessageTarget::new_public(builder),
            approve_integrator_message: ApproveIntegratorMessageTarget::new_public(builder),
            on_chain_operations_count: builder.add_virtual_public_input(),
            on_chain_operations_pub_data: (0..on_chain_operations_limit)
                .map(|_| {
                    builder
                        .add_virtual_public_u8_targets_unsafe(
                            ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE,
                        )
                        .try_into()
                        .unwrap()
                })
                .collect(), // safe because it is connected to public witness from tx circuit which range-checked its output
            priority_operations_count: builder.add_virtual_public_input(),
            priority_operations_pub_data: builder
                .add_virtual_public_u8_targets_unsafe(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX)
                .try_into()
                .unwrap(), // safe because it is connected to public witness from tx circuit which range-checked its output
        }
    }

    /// Similar to [`BlockTxChainWitness::from_public_inputs`], parses proof target.
    /// Returns the number of public inputs.
    /// Assumes _on_chain_operations_limit and _priority_ops_limit are 1.
    pub fn from_public_inputs(
        pis: &[Target],
        _on_chain_operations_limit: usize,
        _priority_ops_limit: usize,
    ) -> (Self, usize) {
        let new_public_market_details_index = 18;

        let change_pub_key_message_index = new_public_market_details_index + POSITION_LIST_SIZE * 5;
        let transfer_message_index = change_pub_key_message_index + CHANGE_PK_PUBLIC_INPUTS_LEN;
        let approve_integrator_message_index = transfer_message_index + TRANSFER_PUBLIC_INPUTS_LEN;

        let on_chain_operations_count_index =
            approve_integrator_message_index + APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN;
        let on_chain_operations_pub_data_index = on_chain_operations_count_index + 1;

        let priority_operations_count_index =
            on_chain_operations_pub_data_index + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;
        let priority_operations_pub_data = priority_operations_count_index + 1;

        let total_pis_size =
            priority_operations_pub_data + MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX;

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

                new_account_delta_tree_root: HashOutTarget {
                    elements: [pis[14], pis[15], pis[16], pis[17]],
                },

                new_public_market_details: pis
                    [new_public_market_details_index..change_pub_key_message_index]
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

                change_pub_key_message: ChangePubKeyMessageTarget::from_public_inputs(
                    &pis[change_pub_key_message_index..transfer_message_index],
                ),
                transfer_message: TransferMessageTarget::from_public_inputs(
                    &pis[transfer_message_index..approve_integrator_message_index],
                ),
                approve_integrator_message: ApproveIntegratorMessageTarget::from_public_inputs(
                    &pis[approve_integrator_message_index..on_chain_operations_count_index],
                ),

                on_chain_operations_count: pis[on_chain_operations_count_index],
                on_chain_operations_pub_data: pis
                    [on_chain_operations_pub_data_index..priority_operations_count_index]
                    .iter()
                    .collect::<Vec<_>>()
                    .chunks(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE)
                    .map(|chunk| core::array::from_fn(|i| U8Target(*chunk[i])))
                    .collect::<Vec<_>>(),

                priority_operations_count: pis[priority_operations_count_index],
                priority_operations_pub_data: core::array::from_fn(|i| {
                    U8Target(pis[priority_operations_pub_data + i])
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
            self.new_account_delta_tree_root,
            other.new_account_delta_tree_root,
        );

        ChangePubKeyMessageTarget::connect(
            builder,
            &self.change_pub_key_message,
            &other.change_pub_key_message,
        );
        TransferMessageTarget::connect(builder, &self.transfer_message, &other.transfer_message);
        ApproveIntegratorMessageTarget::connect(
            builder,
            &self.approve_integrator_message,
            &other.approve_integrator_message,
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
        for (i, byte) in self.priority_operations_pub_data.iter().enumerate() {
            builder.connect_u8(*byte, other.priority_operations_pub_data[i]);
        }

        connect_public_market_details(
            builder,
            &self.new_public_market_details,
            &other.new_public_market_details,
        );
    }
}
