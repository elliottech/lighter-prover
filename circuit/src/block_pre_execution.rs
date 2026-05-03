// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::Target;

use crate::block::Block;
use crate::types::asset::Asset;
use crate::types::config::F;
use crate::types::constants::{ASSET_LIST_SIZE, MARGINED_ASSET_LIST_SIZE, POSITION_LIST_SIZE};
use crate::types::margined_asset::{MARGINED_ASSET_SIZE, MarginedAsset, MarginedAssetTarget};
use crate::types::market_details::{MARKET_DETAIL_SIZE, MarketDetails, MarketDetailsTarget};
use crate::types::price_updates::PriceUpdates;
use crate::types::register::RegisterStack;
use crate::types::state_metadata::{STATE_METADATA_SIZE, StateMetadata, StateMetadataTarget};
use crate::types::system_config::SystemConfig;

#[derive(Clone, Debug)]
/// Public + Secret Witness for single block pre-exec
pub struct BlockPreExec<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub created_at: i64,
    pub block_number: u64,

    pub old_system_config: SystemConfig,
    pub register_stack_before: RegisterStack,
    pub all_assets: [Asset; ASSET_LIST_SIZE],
    pub all_margined_assets: [MarginedAsset; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details: [MarketDetails; POSITION_LIST_SIZE],
    pub state_metadata: StateMetadata,

    pub price_updates: PriceUpdates,
    pub calculate_premium: bool,
    pub calculate_funding: bool,
    pub calculate_oracle_prices: bool,

    pub old_account_tree_root: HashOut<F>,
    pub old_account_pub_data_tree_root: HashOut<F>,
    pub old_market_tree_root: HashOut<F>,
    pub old_state_root: HashOut<F>,
}

impl BlockPreExec<F> {
    pub fn from_block(block: &Block<F>) -> Self {
        Self {
            created_at: block.created_at,
            block_number: block.block_number,
            old_system_config: block.old_system_config,
            register_stack_before: block.register_stack_before,
            all_assets: block.all_assets.clone(),
            all_margined_assets: block.all_margined_assets.clone(),
            all_market_details: block.all_market_details.clone(),
            price_updates: block.price_updates.clone(),
            calculate_premium: block.calculate_premium,
            calculate_funding: block.calculate_funding,
            calculate_oracle_prices: block.calculate_oracle_prices,
            old_account_tree_root: block.old_account_tree_root,
            old_account_pub_data_tree_root: block.old_account_pub_data_tree_root,
            old_market_tree_root: block.old_market_tree_root,
            old_state_root: block.old_state_root,
            state_metadata: block.state_metadata.clone(),
        }
    }
}

#[derive(Debug, Clone)]
/// Public PreExec Block Witness. Used in recursion
pub struct BlockPreExecWitness<F>
where
    F: Field + RichField,
{
    pub new_state_metadata: StateMetadata,
    pub new_market_details: [MarketDetails; POSITION_LIST_SIZE],
    pub new_margined_assets: [MarginedAsset; MARGINED_ASSET_LIST_SIZE],
    pub old_state_root: HashOut<F>,
    pub new_state_root: HashOut<F>,
    pub new_validium_root: HashOut<F>,
    pub block_number: u64,
    pub created_at: i64,
}

impl<F> BlockPreExecWitness<F>
where
    F: Field + RichField,
{
    /// Parse public inputs from proof into BlockWitness
    /// See [`crate::block_pre_execution_constraints::BlockPreExecutionCircuit::register_public_inputs`]
    pub fn from_public_inputs(public_inputs: &[F]) -> Self {
        Self {
            new_state_metadata: StateMetadata::from_public_inputs(
                &public_inputs[0..STATE_METADATA_SIZE],
            ),
            new_market_details: core::array::from_fn(|market_index| {
                MarketDetails::from_public_inputs(
                    market_index as u16,
                    &public_inputs[STATE_METADATA_SIZE + market_index * MARKET_DETAIL_SIZE
                        ..STATE_METADATA_SIZE + (market_index + 1) * MARKET_DETAIL_SIZE],
                )
            }),
            new_margined_assets: core::array::from_fn(|asset_index| {
                MarginedAsset::from_public_inputs(
                    asset_index as u8,
                    &public_inputs[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + asset_index * MARGINED_ASSET_SIZE
                        ..STATE_METADATA_SIZE
                            + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                            + (asset_index + 1) * MARGINED_ASSET_SIZE],
                )
            }),
            old_state_root: HashOut::<F>::from_vec(vec![
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 1],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 2],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 3],
            ]),
            new_state_root: HashOut::<F>::from_vec(vec![
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 4],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 5],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 6],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 7],
            ]),
            new_validium_root: HashOut::<F>::from_vec(vec![
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 8],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 9],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 10],
                public_inputs[STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                    + 11],
            ]),
            block_number: public_inputs[STATE_METADATA_SIZE
                + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                + 12]
                .to_canonical_u64(),
            created_at: public_inputs[STATE_METADATA_SIZE
                + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                + 13]
                .to_canonical_u64() as i64,
        }
    }
}

#[derive(Debug)]
/// In circuit represantion of [`BlockPreExecWitness`]
pub struct BlockPreExecWitnessTarget {
    pub new_state_metadata: StateMetadataTarget,
    pub new_market_details: [MarketDetailsTarget; POSITION_LIST_SIZE],
    pub new_margined_assets: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    pub old_state_root: HashOutTarget,
    pub new_state_root: HashOutTarget,
    pub new_validium_root: HashOutTarget,
    pub block_number: Target,
    pub created_at: Target,
}

impl BlockPreExecWitnessTarget {
    /// Similar to [`BlockPreExecWitness::from_public_inputs`], parses proof target.
    pub fn from_public_inputs(pis: &[Target]) -> Self {
        Self {
            new_state_metadata: StateMetadataTarget {
                last_funding_round_timestamp: pis[0],
                last_oracle_price_timestamp: pis[1],
                last_premium_timestamp: pis[2],
            },
            new_market_details: pis[STATE_METADATA_SIZE
                ..STATE_METADATA_SIZE + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE]
                .chunks(MARKET_DETAIL_SIZE)
                .map(|chunk| MarketDetailsTarget::from_public_inputs(chunk.to_vec()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            new_margined_assets: pis[STATE_METADATA_SIZE + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                ..STATE_METADATA_SIZE
                    + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                    + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE]
                .chunks(MARGINED_ASSET_SIZE)
                .map(MarginedAssetTarget::from_public_inputs)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            old_state_root: HashOutTarget {
                elements: [
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 1],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 2],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 3],
                ],
            },
            new_state_root: HashOutTarget {
                elements: [
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 4],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 5],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 6],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 7],
                ],
            },
            new_validium_root: HashOutTarget {
                elements: [
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 8],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 9],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 10],
                    pis[STATE_METADATA_SIZE
                        + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                        + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                        + 11],
                ],
            },
            block_number: pis[STATE_METADATA_SIZE
                + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                + 12],
            created_at: pis[STATE_METADATA_SIZE
                + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE
                + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE
                + 13],
        }
    }
}
