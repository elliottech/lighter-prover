// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;
use std::fmt;

use num::BigInt;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use serde::Deserialize;
use serde_with::serde_as;

use crate::deserializers;
use crate::tx::Tx;
use crate::types::asset::Asset;
use crate::types::config::F;
use crate::types::constants::{
    ASSET_LIST_SIZE, KECCAK_HASH_OUT_BYTE_SIZE, ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE,
    POSITION_LIST_SIZE,
};
use crate::types::market_details::{MarketDetails, PublicMarketDetails};
use crate::types::price_updates::PriceUpdates;
use crate::types::register::RegisterStack;
use crate::types::state_metadata::StateMetadata;
use crate::types::system_config::SystemConfig;

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
#[serde(bound = "")]
/// Public + Secret Witness for single block. Covers BlockPreExec and BlockTx
pub struct Block<F>
where
    F: Field + Extendable<5> + RichField,
{
    #[serde(rename = "ca")]
    pub created_at: i64,
    #[serde(rename = "bn")]
    pub block_number: u64,

    #[serde(rename = "rb", default)]
    #[serde(deserialize_with = "deserializers::register_stack")]
    pub register_stack_before: RegisterStack,

    #[serde(rename = "osc", default)]
    pub old_system_config: SystemConfig,

    #[serde(rename = "mib")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub all_market_details: [MarketDetails; POSITION_LIST_SIZE],

    #[serde(rename = "aab")]
    #[serde_as(as = "[_; ASSET_LIST_SIZE]")]
    pub all_assets: [Asset; ASSET_LIST_SIZE],

    #[serde(rename = "pmda")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub new_public_market_details: [PublicMarketDetails; POSITION_LIST_SIZE],

    #[serde(rename = "pu", default)]
    pub price_updates: PriceUpdates,

    #[serde(rename = "cp", default)]
    pub calculate_premium: bool,

    #[serde(rename = "cf", default)]
    pub calculate_funding: bool,

    #[serde(rename = "cop", default)]
    pub calculate_oracle_prices: bool,

    #[serde(rename = "oatr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_account_tree_root: HashOut<F>,

    #[serde(rename = "oapt")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_account_pub_data_tree_root: HashOut<F>,

    #[serde(rename = "omtr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_market_tree_root: HashOut<F>,

    #[serde(rename = "osm")]
    #[serde(default)]
    pub state_metadata: StateMetadata,

    #[serde(rename = "osr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_state_root: HashOut<F>,

    #[serde(rename = "oapdtr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "nvr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_validium_root: HashOut<F>,

    #[serde(rename = "nsr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_state_root: HashOut<F>,

    #[serde(rename = "napdtr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "ococ", default)]
    pub on_chain_operations_count: u64,
    #[serde(rename = "ocpd")]
    #[serde(deserialize_with = "deserializers::on_chain_pub_data_vector")]
    pub on_chain_operations_pub_data: Vec<[u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    #[serde(rename = "poc", default)]
    pub priority_operations_count: u64,
    #[serde(rename = "oppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub old_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],
    #[serde(rename = "nppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub new_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "txs")]
    pub txs: Vec<Tx<F>>,
}

#[serde_as]
#[derive(Clone, Deserialize, PartialEq)]
#[serde(bound = "")]
/// Public Block Witness. Used in recursion
pub struct BlockWitness<F>
where
    F: Field + RichField,
{
    #[serde(rename = "bn")]
    pub block_number: u64,
    #[serde(rename = "ca")]
    pub created_at: i64,

    #[serde(rename = "osr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub old_state_root: HashOut<F>,
    #[serde(rename = "nvr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_validium_root: HashOut<F>,
    #[serde(rename = "nsr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub new_state_root: HashOut<F>,

    #[serde(rename = "oapdtr")]
    pub old_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "napdtr")]
    pub new_account_delta_tree_root: HashOut<F>,

    #[serde(rename = "ococ")]
    #[serde(default)]
    pub on_chain_operations_count: u64,

    #[serde(rename = "ocpd")]
    #[serde(deserialize_with = "deserializers::on_chain_pub_data_vector")]
    pub on_chain_operations_pub_data: Vec<[u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>,

    #[serde(rename = "poc")]
    #[serde(default)]
    pub priority_operations_count: u64,

    #[serde(rename = "oppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub old_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "nppoh")]
    #[serde(deserialize_with = "deserializers::hex_to_bytes")]
    pub new_prefix_priority_operation_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    #[serde(rename = "pmda")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub new_public_market_details: [PublicMarketDetails; POSITION_LIST_SIZE],
}

impl<F> fmt::Debug for BlockWitness<F>
where
    F: Field + RichField,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut on_chain_pub_data = vec![];

        self.on_chain_operations_pub_data
            .iter()
            .for_each(|pub_data| {
                on_chain_pub_data.push(hex::encode(pub_data));
            });

        let old_prefix_priority_operation_hash =
            hex::encode(self.old_prefix_priority_operation_hash);
        let new_prefix_priority_operation_hash =
            hex::encode(self.new_prefix_priority_operation_hash);

        let mut new_market_details = HashMap::<usize, PublicMarketDetails>::new();
        self.new_public_market_details
            .iter()
            .filter(|market_detail| !market_detail.is_empty())
            .enumerate()
            .for_each(|(index, market_details)| {
                new_market_details.insert(index, market_details.clone());
            });

        let new_public_market_details = serde_json::to_string(&new_market_details).unwrap();

        fmt.debug_struct("BlockWitness<F>")
            .field("block_number", &self.block_number)
            .field("created_at", &self.created_at)
            .field("old_state_root", &self.old_state_root)
            .field("new_validium_root", &self.new_validium_root)
            .field("new_state_root", &self.new_state_root)
            .field(
                "old_account_delta_tree_root",
                &self.old_account_delta_tree_root,
            )
            .field(
                "new_account_delta_tree_root",
                &self.new_account_delta_tree_root,
            )
            .field("on_chain_operations_count", &self.on_chain_operations_count)
            .field("on_chain_operations_pub_data", &on_chain_pub_data)
            .field("priority_operations_count", &self.priority_operations_count)
            .field(
                "old_prefix_priority_operation_hash",
                &old_prefix_priority_operation_hash,
            )
            .field(
                "new_prefix_priority_operation_hash",
                &new_prefix_priority_operation_hash,
            )
            .field("new_public_market_details", &new_public_market_details)
            .finish()
    }
}

impl BlockWitness<F> {
    pub fn from_block(block: &Block<F>, on_chain_operations_size: usize) -> Self {
        let mut val = Self {
            block_number: block.block_number,
            created_at: block.created_at,
            old_state_root: block.old_state_root,
            new_validium_root: block.new_validium_root,
            new_state_root: block.new_state_root,
            old_account_delta_tree_root: block.old_account_delta_tree_root,
            new_account_delta_tree_root: block.new_account_delta_tree_root,
            on_chain_operations_count: block.on_chain_operations_count,
            on_chain_operations_pub_data: block.on_chain_operations_pub_data.clone(),
            priority_operations_count: block.priority_operations_count,
            old_prefix_priority_operation_hash: block.old_prefix_priority_operation_hash,
            new_prefix_priority_operation_hash: block.new_prefix_priority_operation_hash,
            new_public_market_details: block.new_public_market_details.clone(),
        };

        // Fill public data up to the limits because real block may not have all public data on it
        // i.e. if block is closed early
        assert!(val.on_chain_operations_count <= on_chain_operations_size as u64);
        val.on_chain_operations_pub_data
            .resize_with(on_chain_operations_size, || {
                [0; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]
            });

        val
    }
}

impl<F> BlockWitness<F>
where
    F: Field + RichField,
{
    /// Parse public inputs from proof into BlockWitness
    pub fn from_public_inputs(public_inputs: &[F], _: usize, _: usize) -> Self {
        let new_public_market_details_index = 22;

        let on_chain_operations_count_index =
            new_public_market_details_index + POSITION_LIST_SIZE * 5;
        let on_chain_operations_pub_data_index = on_chain_operations_count_index + 1;

        let priority_operations_count_index =
            on_chain_operations_pub_data_index + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;
        let old_prefix_priority_operation_hash_index = priority_operations_count_index + 1;
        let new_prefix_priority_operation_hash_index =
            old_prefix_priority_operation_hash_index + KECCAK_HASH_OUT_BYTE_SIZE;

        let tx_pub_data_hashes_index =
            new_prefix_priority_operation_hash_index + KECCAK_HASH_OUT_BYTE_SIZE;

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
            old_account_delta_tree_root: HashOut::<F>::from([
                public_inputs[14],
                public_inputs[15],
                public_inputs[16],
                public_inputs[17],
            ]),

            new_account_delta_tree_root: HashOut::<F>::from([
                public_inputs[18],
                public_inputs[19],
                public_inputs[20],
                public_inputs[21],
            ]),

            new_public_market_details: public_inputs
                [new_public_market_details_index..on_chain_operations_count_index]
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

            // On chain ops pub data
            on_chain_operations_count: public_inputs[on_chain_operations_count_index]
                .to_canonical_u64(),
            on_chain_operations_pub_data: public_inputs
                [on_chain_operations_pub_data_index..priority_operations_count_index]
                .iter()
                .collect::<Vec<_>>()
                .chunks(ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE)
                .map(|chunk| {
                    core::array::from_fn(|i| {
                        u8::try_from(chunk[i].to_canonical_u64())
                            .expect("Failed to convert on_chain_operations_pub_data limb to u8")
                    })
                })
                .collect::<Vec<_>>(),

            // Priority ops pub data
            priority_operations_count: public_inputs[priority_operations_count_index]
                .to_canonical_u64(),
            old_prefix_priority_operation_hash: public_inputs
                [old_prefix_priority_operation_hash_index
                    ..new_prefix_priority_operation_hash_index]
                .iter()
                .map(|x| {
                    u8::try_from(x.to_canonical_u64())
                        .expect("Failed to convert old_prefix_priority_operation_hash limb to u8")
                })
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
            new_prefix_priority_operation_hash: public_inputs
                [new_prefix_priority_operation_hash_index..tx_pub_data_hashes_index]
                .iter()
                .map(|x| {
                    u8::try_from(x.to_canonical_u64())
                        .expect("Failed to convert new_prefix_priority_operation_hash limb to u8")
                })
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        }
    }
}
