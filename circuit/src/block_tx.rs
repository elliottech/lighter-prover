// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::Target;

use crate::tx::Tx;
use crate::types::approve_integrator::{
    APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN, ApproveIntegratorMessage, ApproveIntegratorMessageTarget,
};
use crate::types::asset::{ASSET_SIZE, Asset, AssetTarget};
use crate::types::change_pub_key::{
    CHANGE_PK_PUBLIC_INPUTS_LEN, ChangePubKeyMessage, ChangePubKeyMessageTarget,
};
use crate::types::constants::*;
use crate::types::margined_asset::{MARGINED_ASSET_SIZE, MarginedAsset, MarginedAssetTarget};
use crate::types::market_details::{MARKET_DETAIL_SIZE, MarketDetails, MarketDetailsTarget};
use crate::types::register::{REGISTER_INFO_SIZE, RegisterStack, RegisterStackTarget};
use crate::types::system_config::{SYSTEM_CONFIG_SIZE, SystemConfig, SystemConfigTarget};
use crate::types::transfer::{TRANSFER_PUBLIC_INPUTS_LEN, TransferMessage, TransferMessageTarget};
use crate::uint::u8::U8Target;

pub struct BlockTx<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub created_at: i64,

    pub old_system_config: SystemConfig,
    pub register_stack_before: RegisterStack,
    pub all_assets_before: [Asset; ASSET_LIST_SIZE],
    pub all_margined_assets_before: [MarginedAsset; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details_before: [MarketDetails; POSITION_LIST_SIZE],

    pub old_account_tree_root: HashOut<F>,
    pub old_account_pub_data_tree_root: HashOut<F>,
    pub old_account_delta_tree_root: HashOut<F>,
    pub old_market_tree_root: HashOut<F>,

    pub txs: Vec<Tx<F>>,
}

#[derive(Debug, Clone)]
/// Public Block Transaction Witness
pub struct BlockTxWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub old_system_config: SystemConfig,
    pub register_stack_before: RegisterStack,
    pub all_assets_before: [Asset; ASSET_LIST_SIZE],
    pub all_margined_assets_before: [MarginedAsset; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details_before: [MarketDetails; POSITION_LIST_SIZE],

    pub old_account_tree_root: HashOut<F>,
    pub old_account_pub_data_tree_root: HashOut<F>,
    pub old_market_tree_root: HashOut<F>,
    pub old_account_delta_tree_root: HashOut<F>,

    pub new_system_config: SystemConfig,
    pub register_stack_after: RegisterStack,
    pub all_assets_after: [Asset; ASSET_LIST_SIZE],
    pub all_margined_assets_after: [MarginedAsset; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details_after: [MarketDetails; POSITION_LIST_SIZE],

    pub new_account_tree_root: HashOut<F>,
    pub new_account_pub_data_tree_root: HashOut<F>,
    pub new_account_delta_tree_root: HashOut<F>,
    pub new_market_tree_root: HashOut<F>,

    pub change_pub_key_message: ChangePubKeyMessage<F>,
    pub transfer_message: TransferMessage,
    pub approve_integrator_message: ApproveIntegratorMessage,

    pub on_chain_operations_count: u64,
    pub on_chain_operations_pub_data: [u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],

    pub priority_operations_count: u64,
    pub priority_operations_pub_data: [u8; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
}

impl<F> BlockTxWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    /// Parse public inputs from proof into BlockWitness
    pub fn from_public_inputs(public_inputs: &[F]) -> Self {
        let old_assets_start = 16;
        let old_assets_end = old_assets_start + ASSET_LIST_SIZE * ASSET_SIZE;

        let old_margined_assets_start = old_assets_end;
        let old_margined_assets_end =
            old_margined_assets_start + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE;

        let old_market_details_start = old_margined_assets_end;
        let old_market_details_end =
            old_market_details_start + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE;

        let old_system_config_start = old_market_details_end;
        let old_system_config_end = old_system_config_start + SYSTEM_CONFIG_SIZE;

        let old_register_start = old_system_config_end;
        let old_register_end = old_register_start + REGISTER_INFO_SIZE;

        let assets_start = old_register_end + 16;
        let assets_end = assets_start + ASSET_LIST_SIZE * ASSET_SIZE;

        let margined_assets_start = assets_end;
        let margined_assets_end =
            margined_assets_start + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE;

        let market_details_start = margined_assets_end;
        let market_details_end = market_details_start + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE;

        let change_pub_key_message_start = market_details_end;
        let change_pub_key_message_end = change_pub_key_message_start + CHANGE_PK_PUBLIC_INPUTS_LEN;

        let transfer_message_start = change_pub_key_message_end;
        let transfer_message_end = transfer_message_start + TRANSFER_PUBLIC_INPUTS_LEN;

        let approve_integrator_message_start = transfer_message_end;
        let approve_integrator_message_end =
            approve_integrator_message_start + APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN;

        // on_chain_pub_data_count
        let on_chain_pub_data_start = approve_integrator_message_end + 1;
        let on_chain_pub_data_end =
            on_chain_pub_data_start + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;

        // priority_pub_data_count
        let priority_pub_data_start = on_chain_pub_data_end + 1;
        let priority_pub_data_end =
            priority_pub_data_start + MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX;

        let system_config_start = priority_pub_data_end;
        let system_config_end = system_config_start + SYSTEM_CONFIG_SIZE;

        let register_start = system_config_end;
        let register_end = register_start + REGISTER_INFO_SIZE;

        Self {
            old_account_pub_data_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[0],
                public_inputs[1],
                public_inputs[2],
                public_inputs[3],
            ]),
            old_account_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[4],
                public_inputs[5],
                public_inputs[6],
                public_inputs[7],
            ]),
            old_market_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[8],
                public_inputs[9],
                public_inputs[10],
                public_inputs[11],
            ]),
            old_account_delta_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[12],
                public_inputs[13],
                public_inputs[14],
                public_inputs[15],
            ]),

            all_assets_before: core::array::from_fn(|asset_index| {
                Asset::from_public_inputs(
                    asset_index as i16,
                    &public_inputs[old_assets_start + asset_index * ASSET_SIZE
                        ..old_assets_start + (asset_index + 1) * ASSET_SIZE],
                )
            }),
            all_margined_assets_before: core::array::from_fn(|margined_asset_index| {
                MarginedAsset::from_public_inputs(
                    margined_asset_index as u8,
                    &public_inputs[old_margined_assets_start
                        + margined_asset_index * MARGINED_ASSET_SIZE
                        ..old_margined_assets_start
                            + (margined_asset_index + 1) * MARGINED_ASSET_SIZE],
                )
            }),

            all_market_details_before: core::array::from_fn(|market_index| {
                MarketDetails::from_public_inputs(
                    market_index as u16,
                    &public_inputs[old_market_details_start + market_index * MARKET_DETAIL_SIZE
                        ..old_market_details_start + (market_index + 1) * MARKET_DETAIL_SIZE],
                )
            }),

            old_system_config: SystemConfig::from_public_inputs(
                &public_inputs[old_system_config_start..old_system_config_end],
            ),
            register_stack_before: RegisterStack::from_public_inputs(
                &public_inputs[old_register_start..old_register_end],
            ),

            new_account_pub_data_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[old_register_end],
                public_inputs[old_register_end + 1],
                public_inputs[old_register_end + 2],
                public_inputs[old_register_end + 3],
            ]),
            new_account_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[old_register_end + 4],
                public_inputs[old_register_end + 5],
                public_inputs[old_register_end + 6],
                public_inputs[old_register_end + 7],
            ]),
            new_market_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[old_register_end + 8],
                public_inputs[old_register_end + 9],
                public_inputs[old_register_end + 10],
                public_inputs[old_register_end + 11],
            ]),
            new_account_delta_tree_root: HashOut::<F>::from_vec(vec![
                public_inputs[old_register_end + 12],
                public_inputs[old_register_end + 13],
                public_inputs[old_register_end + 14],
                public_inputs[old_register_end + 15],
            ]),

            all_assets_after: core::array::from_fn(|asset_index| {
                Asset::from_public_inputs(
                    asset_index as i16,
                    &public_inputs[assets_start + asset_index * ASSET_SIZE
                        ..assets_start + (asset_index + 1) * ASSET_SIZE],
                )
            }),
            all_margined_assets_after: core::array::from_fn(|margined_asset_index| {
                MarginedAsset::from_public_inputs(
                    margined_asset_index as u8,
                    &public_inputs[margined_assets_start
                        + margined_asset_index * MARGINED_ASSET_SIZE
                        ..margined_assets_start + (margined_asset_index + 1) * MARGINED_ASSET_SIZE],
                )
            }),
            all_market_details_after: core::array::from_fn(|market_index| {
                MarketDetails::from_public_inputs(
                    market_index as u16,
                    &public_inputs[market_details_start + market_index * MARKET_DETAIL_SIZE
                        ..market_details_start + (market_index + 1) * MARKET_DETAIL_SIZE],
                )
            }),

            change_pub_key_message: ChangePubKeyMessage::from_public_inputs(
                &public_inputs[change_pub_key_message_start..change_pub_key_message_end],
            ),
            transfer_message: TransferMessage::from_public_inputs(
                &public_inputs[transfer_message_start..transfer_message_end],
            ),
            approve_integrator_message: ApproveIntegratorMessage::from_public_inputs(
                &public_inputs[approve_integrator_message_start..approve_integrator_message_end],
            ),

            on_chain_operations_count: public_inputs[approve_integrator_message_end]
                .to_canonical_u64(),
            on_chain_operations_pub_data: core::array::from_fn(|index| {
                public_inputs[on_chain_pub_data_start + index].to_canonical_u64() as u8
            }),

            priority_operations_count: public_inputs[on_chain_pub_data_end].to_canonical_u64(),
            priority_operations_pub_data: core::array::from_fn(|index| {
                public_inputs[priority_pub_data_start + index].to_canonical_u64() as u8
            }),

            new_system_config: SystemConfig::from_public_inputs(
                &public_inputs[system_config_start..system_config_end],
            ),
            register_stack_after: RegisterStack::from_public_inputs(
                &public_inputs[register_start..register_end],
            ),
        }
    }
}

#[derive(Debug)]
/// In circuit represantion of [`crate::block::BlockTxWitness`]
pub struct BlockTxWitnessTarget {
    pub old_system_config: SystemConfigTarget,
    pub register_stack_before: RegisterStackTarget,
    pub all_assets_before: [AssetTarget; ASSET_LIST_SIZE],
    pub all_margined_assets_before: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details_before: [MarketDetailsTarget; POSITION_LIST_SIZE],

    pub old_account_tree_root: HashOutTarget,
    pub old_account_pub_data_tree_root: HashOutTarget,
    pub old_account_delta_tree_root: HashOutTarget,
    pub old_market_tree_root: HashOutTarget,

    pub new_system_config: SystemConfigTarget,
    pub register_stack_after: RegisterStackTarget,
    pub all_assets_after: [AssetTarget; ASSET_LIST_SIZE],
    pub all_margined_assets_after: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details_after: [MarketDetailsTarget; POSITION_LIST_SIZE],

    pub new_account_tree_root: HashOutTarget,
    pub new_account_pub_data_tree_root: HashOutTarget,
    pub new_account_delta_tree_root: HashOutTarget,
    pub new_market_tree_root: HashOutTarget,

    pub change_pub_key_message: ChangePubKeyMessageTarget,
    pub transfer_message: TransferMessageTarget,
    pub approve_integrator_message: ApproveIntegratorMessageTarget,

    pub on_chain_operations_count: Target,
    pub on_chain_operations_pub_data: [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],

    pub priority_operations_count: Target,
    pub priority_operations_pub_data: [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
}

impl BlockTxWitnessTarget {
    /// Similar to [`BlockTxWitness::from_public_inputs`], parses proof target.
    /// Follows the same order as [`crate::block_tx_constraints::BlockCircuit::register_public_inputs`]
    pub fn from_public_inputs(pis: &[Target]) -> Self {
        let old_assets_start = 16;
        let old_assets_end = old_assets_start + ASSET_LIST_SIZE * ASSET_SIZE;

        let old_margined_assets_start = old_assets_end;
        let old_margined_assets_end =
            old_margined_assets_start + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE;

        let old_market_details_start = old_margined_assets_end;
        let old_market_details_end =
            old_market_details_start + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE;

        let old_system_config_start = old_market_details_end;
        let old_system_config_end = old_system_config_start + SYSTEM_CONFIG_SIZE;

        let old_register_start = old_system_config_end;
        let old_register_end = old_register_start + REGISTER_INFO_SIZE;

        let assets_start = old_register_end + 16;
        let assets_end = assets_start + ASSET_LIST_SIZE * ASSET_SIZE;

        let margined_assets_start = assets_end;
        let margined_assets_end =
            margined_assets_start + MARGINED_ASSET_LIST_SIZE * MARGINED_ASSET_SIZE;

        let market_details_start = margined_assets_end;
        let market_details_end = market_details_start + POSITION_LIST_SIZE * MARKET_DETAIL_SIZE;

        let change_pub_key_message_start = market_details_end;
        let change_pub_key_message_end = change_pub_key_message_start + CHANGE_PK_PUBLIC_INPUTS_LEN;

        let transfer_message_start = change_pub_key_message_end;
        let transfer_message_end = transfer_message_start + TRANSFER_PUBLIC_INPUTS_LEN;

        let approve_integrator_message_start = transfer_message_end;
        let approve_integrator_message_end =
            approve_integrator_message_start + APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN;

        let on_chain_pub_data_start = approve_integrator_message_end + 1;
        let on_chain_pub_data_end =
            on_chain_pub_data_start + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;

        let priority_pub_data_start = on_chain_pub_data_end + 1;
        let priority_pub_data_end =
            priority_pub_data_start + MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX;

        let system_config_start = priority_pub_data_end;
        let system_config_end = system_config_start + SYSTEM_CONFIG_SIZE;

        let register_start = system_config_end;
        let register_end = register_start + REGISTER_INFO_SIZE;

        assert_eq!(
            pis.len(),
            register_end,
            "Expected {} public inputs, but got {}",
            register_end,
            pis.len()
        );

        Self {
            old_account_pub_data_tree_root: HashOutTarget::from_vec(pis[0..4].to_vec()),
            old_account_tree_root: HashOutTarget::from_vec(pis[4..8].to_vec()),
            old_market_tree_root: HashOutTarget::from_vec(pis[8..12].to_vec()),
            old_account_delta_tree_root: HashOutTarget::from_vec(pis[12..16].to_vec()),

            all_assets_before: core::array::from_fn(|asset_index| {
                AssetTarget::from_public_inputs(
                    &pis[old_assets_start + asset_index * ASSET_SIZE
                        ..old_assets_start + (asset_index + 1) * ASSET_SIZE],
                )
            }),
            all_margined_assets_before: core::array::from_fn(|margined_asset_index| {
                MarginedAssetTarget::from_public_inputs(
                    &pis[old_margined_assets_start + margined_asset_index * MARGINED_ASSET_SIZE
                        ..old_margined_assets_start
                            + (margined_asset_index + 1) * MARGINED_ASSET_SIZE],
                )
            }),
            all_market_details_before: core::array::from_fn(|market_index| {
                MarketDetailsTarget::from_public_inputs(
                    pis[old_market_details_start + market_index * MARKET_DETAIL_SIZE
                        ..old_market_details_start + (market_index + 1) * MARKET_DETAIL_SIZE]
                        .to_vec(),
                )
            }),

            old_system_config: SystemConfigTarget::from_public_inputs(
                &pis[old_system_config_start..old_system_config_end],
            ),
            register_stack_before: RegisterStackTarget::from_public_inputs(
                &pis[old_register_start..old_register_end],
            ),

            new_account_pub_data_tree_root: HashOutTarget::from_vec(
                pis[old_register_end..old_register_end + 4].to_vec(),
            ),
            new_account_tree_root: HashOutTarget::from_vec(
                pis[old_register_end + 4..old_register_end + 8].to_vec(),
            ),
            new_market_tree_root: HashOutTarget::from_vec(
                pis[old_register_end + 8..old_register_end + 12].to_vec(),
            ),
            new_account_delta_tree_root: HashOutTarget::from_vec(
                pis[old_register_end + 12..old_register_end + 16].to_vec(),
            ),

            all_assets_after: core::array::from_fn(|asset_index| {
                AssetTarget::from_public_inputs(
                    &pis[assets_start + asset_index * ASSET_SIZE
                        ..assets_start + (asset_index + 1) * ASSET_SIZE],
                )
            }),
            all_margined_assets_after: core::array::from_fn(|margined_asset_index| {
                MarginedAssetTarget::from_public_inputs(
                    &pis[margined_assets_start + margined_asset_index * MARGINED_ASSET_SIZE
                        ..margined_assets_start + (margined_asset_index + 1) * MARGINED_ASSET_SIZE],
                )
            }),
            all_market_details_after: core::array::from_fn(|market_index| {
                MarketDetailsTarget::from_public_inputs(
                    pis[market_details_start + market_index * MARKET_DETAIL_SIZE
                        ..market_details_start + (market_index + 1) * MARKET_DETAIL_SIZE]
                        .to_vec(),
                )
            }),

            change_pub_key_message: ChangePubKeyMessageTarget::from_public_inputs(
                &pis[change_pub_key_message_start..change_pub_key_message_end],
            ),
            transfer_message: TransferMessageTarget::from_public_inputs(
                &pis[transfer_message_start..transfer_message_end],
            ),
            approve_integrator_message: ApproveIntegratorMessageTarget::from_public_inputs(
                &pis[approve_integrator_message_start..approve_integrator_message_end],
            ),

            on_chain_operations_count: pis[approve_integrator_message_end],
            on_chain_operations_pub_data: pis[on_chain_pub_data_start..on_chain_pub_data_end]
                .iter()
                .map(|&x| U8Target(x))
                .collect::<Vec<U8Target>>()
                .try_into()
                .unwrap(),

            priority_operations_count: pis[on_chain_pub_data_end],
            priority_operations_pub_data: pis[priority_pub_data_start..priority_pub_data_end]
                .iter()
                .map(|&x| U8Target(x))
                .collect::<Vec<U8Target>>()
                .try_into()
                .unwrap(),

            new_system_config: SystemConfigTarget::from_public_inputs(
                &pis[system_config_start..system_config_end],
            ),
            register_stack_after: RegisterStackTarget::from_public_inputs(
                &pis[register_start..register_end],
            ),
        }
    }
}
