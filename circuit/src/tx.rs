// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use serde::Deserialize;
use serde_with::serde_as;

use crate::deserializers;
use crate::ecdsa::curve::ecdsa::{ECDSAPublicKey, ECDSASignature};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::eddsa::schnorr::SchnorrSig;
use crate::transactions::internal_cancel_all_orders::InternalCancelAllOrdersTx;
use crate::transactions::internal_cancel_order::InternalCancelOrderTx;
use crate::transactions::internal_claim_order::InternalClaimOrderTx;
use crate::transactions::internal_create_order::InternalCreateOrderTx;
use crate::transactions::internal_deleverage::InternalDeleverageTx;
use crate::transactions::internal_exit_position::InternalExitPositionTx;
use crate::transactions::internal_liquidate_position::InternalLiquidatePositionTx;
use crate::transactions::internal_pending_unlock::InternalPendingUnlockTx;
use crate::transactions::l1_burn_shares::L1BurnSharesTx;
use crate::transactions::l1_cancel_all_orders::L1CancelAllOrdersTx;
use crate::transactions::l1_change_pubkey::L1ChangePubKeyTx;
use crate::transactions::l1_create_market::L1CreateMarketTx;
use crate::transactions::l1_create_order::L1CreateOrderTx;
use crate::transactions::l1_deposit::L1DepositTx;
use crate::transactions::l1_register_asset::L1RegisterAssetTx;
use crate::transactions::l1_set_system_config::L1SetSystemConfigTx;
use crate::transactions::l1_update_asset::L1UpdateAssetTx;
use crate::transactions::l1_update_market::L1UpdateMarketTx;
use crate::transactions::l1_withdraw::L1WithdrawTx;
use crate::transactions::l2_burn_shares::L2BurnSharesTx;
use crate::transactions::l2_cancel_all_orders::L2CancelAllOrdersTx;
use crate::transactions::l2_cancel_order::L2CancelOrderTx;
use crate::transactions::l2_change_pubkey::L2ChangePubKeyTx;
use crate::transactions::l2_create_grouped_orders::L2CreateGroupedOrdersTx;
use crate::transactions::l2_create_order::L2CreateOrderTx;
use crate::transactions::l2_create_public_pool::L2CreatePublicPoolTx;
use crate::transactions::l2_create_staking_pool::L2CreateStakingPoolTx;
use crate::transactions::l2_create_sub_account::L2CreateSubAccountTx;
use crate::transactions::l2_force_burn_shares::L2ForceBurnSharesTx;
use crate::transactions::l2_mint_shares::L2MintSharesTx;
use crate::transactions::l2_modify_order::L2ModifyOrderTx;
use crate::transactions::l2_stake_assets::L2StakeAssetsTx;
use crate::transactions::l2_strategy_transfer::L2StrategyTransferTx;
use crate::transactions::l2_transfer::L2TransferTx;
use crate::transactions::l2_unstake_assets::L2UnstakeAssetsTx;
use crate::transactions::l2_update_account_config::L2UpdateAccountConfigTx;
use crate::transactions::l2_update_leverage::L2UpdateLeverageTx;
use crate::transactions::l2_update_margin::L2UpdateMarginTx;
use crate::transactions::l2_update_market_config::L2UpdateMarketConfigTx;
use crate::transactions::l2_update_public_pool::L2UpdatePublicPoolTx;
use crate::transactions::l2_withdraw::L2WithdrawTx;
use crate::types::account::Account;
use crate::types::account_asset::AccountAsset;
use crate::types::account_delta::AccountDelta;
use crate::types::account_order::AccountOrder;
use crate::types::api_key::ApiKey;
use crate::types::config::F;
use crate::types::constants::*;
use crate::types::market::Market;
use crate::types::order::Order;
use crate::types::order_book_node::OrderBookNode;

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
#[serde(bound = "")] 
pub struct Tx<F>
where
    F: Field + Extendable<5> + RichField,
{
    #[serde(rename = "tx_type")]
    pub tx_type: u8,

    #[serde(rename = "1d")]
    #[serde(default)]
    pub l1_deposit_tx: L1DepositTx,

    #[serde(rename = "1c")]
    #[serde(default)]
    pub l1_create_market_tx: L1CreateMarketTx,

    #[serde(rename = "1u")]
    #[serde(default)]
    pub l1_update_market_tx: L1UpdateMarketTx,

    #[serde(rename = "1ca")]
    #[serde(default)]
    pub l1_cancel_all_orders_tx: L1CancelAllOrdersTx,

    #[serde(rename = "1w")]
    #[serde(default)]
    pub l1_withdraw_tx: L1WithdrawTx,

    #[serde(rename = "1cr")]
    #[serde(default)]
    pub l1_create_order_tx: L1CreateOrderTx,

    #[serde(rename = "1cp")]
    #[serde(default)]
    pub l1_change_pub_key_tx: L1ChangePubKeyTx<F>,

    #[serde(rename = "1b")]
    #[serde(default)]
    pub l1_burn_shares_tx: L1BurnSharesTx,

    #[serde(rename = "1ra")]
    #[serde(default)]
    pub l1_register_asset_tx: L1RegisterAssetTx,

    #[serde(rename = "1ssc")]
    #[serde(default)]
    pub l1_set_system_config_tx: L1SetSystemConfigTx,

    #[serde(rename = "1ua")]
    #[serde(default)]
    pub l1_update_asset_tx: L1UpdateAssetTx,

    #[serde(rename = "2cpk")]
    #[serde(default)]
    pub l2_change_pub_key_tx: L2ChangePubKeyTx<F>,

    #[serde(rename = "2cs")]
    #[serde(default)]
    pub l2_create_sub_account_tx: L2CreateSubAccountTx,

    #[serde(rename = "2cp")]
    #[serde(default)]
    pub l2_create_public_pool_tx: L2CreatePublicPoolTx,

    #[serde(rename = "2up")]
    #[serde(default)]
    pub l2_update_public_pool_tx: L2UpdatePublicPoolTx,

    #[serde(rename = "2t")]
    #[serde(default)]
    pub l2_transfer_tx: L2TransferTx,

    #[serde(rename = "2w")]
    #[serde(default)]
    pub l2_withdraw_tx: L2WithdrawTx,

    #[serde(rename = "2cr")]
    #[serde(default)]
    pub l2_create_order_tx: L2CreateOrderTx,

    #[serde(rename = "2co")]
    #[serde(default)]
    pub l2_cancel_order_tx: L2CancelOrderTx,

    #[serde(rename = "2ca")]
    #[serde(default)]
    pub l2_cancel_all_orders_tx: L2CancelAllOrdersTx,

    #[serde(rename = "2mo")]
    #[serde(default)]
    pub l2_modify_order_tx: L2ModifyOrderTx,

    #[serde(rename = "2m")]
    #[serde(default)]
    pub l2_mint_shares_tx: L2MintSharesTx,

    #[serde(rename = "2b")]
    #[serde(default)]
    pub l2_burn_shares_tx: L2BurnSharesTx,

    #[serde(rename = "2ul")]
    #[serde(default)]
    pub l2_update_leverage_tx: L2UpdateLeverageTx,

    #[serde(rename = "2cg")]
    #[serde(default)]
    pub l2_create_grouped_orders_tx: L2CreateGroupedOrdersTx,

    #[serde(rename = "2um")]
    #[serde(default)]
    pub l2_update_margin_tx: L2UpdateMarginTx,

    #[serde(rename = "2csp")]
    #[serde(default)]
    pub l2_create_staking_pool_tx: L2CreateStakingPoolTx,

    #[serde(rename = "2sa")]
    #[serde(default)]
    pub l2_stake_assets_tx: L2StakeAssetsTx,

    #[serde(rename = "2ua")]
    #[serde(default)]
    pub l2_unstake_assets_tx: L2UnstakeAssetsTx,

    #[serde(rename = "2st")]
    #[serde(default)]
    pub l2_strategy_transfer_tx: L2StrategyTransferTx,

    #[serde(rename = "2umc")]
    #[serde(default)]
    pub l2_update_market_config_tx: L2UpdateMarketConfigTx,

    #[serde(rename = "2fbs")]
    #[serde(default)]
    pub l2_force_burn_shares_tx: L2ForceBurnSharesTx,

    #[serde(rename = "2uac")]
    #[serde(default)]
    pub l2_update_account_config_tx: L2UpdateAccountConfigTx,

    #[serde(rename = "Ic")]
    #[serde(default)]
    pub internal_claim_order_tx: InternalClaimOrderTx,

    #[serde(rename = "Ico")]
    #[serde(default)]
    pub internal_cancel_order_tx: InternalCancelOrderTx,

    #[serde(rename = "Id")]
    #[serde(default)]
    pub internal_deleverage_tx: InternalDeleverageTx,

    #[serde(rename = "Iex")]
    #[serde(default)]
    pub internal_exit_position_tx: InternalExitPositionTx,

    #[serde(rename = "Ipw")]
    #[serde(default)]
    pub internal_pending_unlock_tx: InternalPendingUnlockTx,

    #[serde(rename = "Ica")]
    #[serde(default)]
    pub internal_cancel_all_orders_tx: InternalCancelAllOrdersTx,

    #[serde(rename = "Il")]
    #[serde(default)]
    pub internal_liquidate_position_tx: InternalLiquidatePositionTx,

    #[serde(rename = "Icr")]
    #[serde(default)]
    pub internal_create_order_tx: InternalCreateOrderTx,

    #[serde(rename = "nonce", default)]
    pub nonce: i64,

    #[serde(rename = "exat")]
    #[serde(default)]
    pub expired_at: i64,

    #[serde(rename = "sig")]
    #[serde(deserialize_with = "deserializers::signature")]
    #[serde(default)]
    pub signature: SchnorrSig,

    #[serde(rename = "tf", default)]
    pub taker_fee: i64,
    #[serde(rename = "mf", default)]
    pub maker_fee: i64,

    #[serde(
        rename = "l1s",
        deserialize_with = "deserializers::l1_signature",
        default
    )]
    pub l1_signature: Option<ECDSASignature<Secp256K1>>,
    #[serde(
        rename = "l1pk",
        deserialize_with = "deserializers::l1_pub_key",
        default
    )]
    pub l1_pub_key: Option<ECDSAPublicKey<Secp256K1>>,

    /***********************/
    /*  STATE TREE LEAVES  */
    /***********************/
    #[serde(rename = "akb", default)]
    pub api_key_before: ApiKey<F>,

    #[serde(rename = "aob", default)]
    pub account_order_before: AccountOrder,

    #[serde(rename = "ab")]
    pub accounts_before: [Account<F>; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "apdb")] // Account public data delta since the beginning of the batch
    pub accounts_delta_before: [AccountDelta<F>; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "mmb")]
    pub market_before: Market<F>,

    #[serde(rename = "obinfob")]
    #[serde(default)]
    pub order_before: Order,

    #[serde(rename = "aab")]
    #[serde(default)]
    pub account_assets_before: [[AccountAsset; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "ai")]
    #[serde(default)]
    pub asset_indices: [i16; NB_ASSETS_PER_TX],

    /*****************************/
    /*  STATE TREE MERKLE PROOFS */
    /*****************************/
    #[serde(rename = "mpab")]
    #[serde(deserialize_with = "deserializers::account_tree_merkle_proofs")]
    pub account_tree_merkle_proofs: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "mpapd")]
    #[serde(deserialize_with = "deserializers::account_tree_merkle_proofs")]
    pub account_pub_data_tree_merkle_proofs:
        [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "mpapdd")]
    #[serde(deserialize_with = "deserializers::account_tree_merkle_proofs")]
    pub account_delta_tree_merkle_proofs: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "mpaab")]
    #[serde(deserialize_with = "deserializers::asset_tree_merkle_proofs")]
    pub asset_tree_merkle_proofs:
        [[[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],
    #[serde(rename = "mpaa")]
    #[serde(deserialize_with = "deserializers::asset_tree_merkle_proofs")]
    pub public_asset_tree_merkle_proofs:
        [[[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],
    #[serde(rename = "mpad")]
    #[serde(deserialize_with = "deserializers::asset_tree_merkle_proofs")]
    pub asset_delta_tree_merkle_proofs:
        [[[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],

    #[serde(rename = "mpppdd")]
    #[serde(deserialize_with = "deserializers::position_delta_merkle_proofs")]
    pub position_delta_merkle_proofs:
        [[HashOut<F>; POSITION_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX - 1],

    #[serde(rename = "mpakb")]
    #[serde(deserialize_with = "deserializers::api_key_tree_merkle_proof")]
    pub api_key_tree_merkle_proof: [HashOut<F>; API_KEY_MERKLE_LEVELS],

    #[serde(rename = "mpokb")]
    #[serde(deserialize_with = "deserializers::account_orders_tree_merkle_proof")]
    pub account_orders_tree_merkle_proof:
        [[HashOut<F>; ACCOUNT_ORDERS_MERKLE_LEVELS]; NB_ACCOUNT_ORDERS_PATHS_PER_TX],

    #[serde(rename = "mpmmb")]
    #[serde(deserialize_with = "deserializers::market_tree_merkle_proof")]
    pub market_tree_merkle_proof: [HashOut<F>; MARKET_MERKLE_LEVELS],

    #[serde(rename = "obpb")]
    #[serde_as(as = "[_; ORDER_BOOK_MERKLE_LEVELS]")]
    pub order_book_tree_path: [OrderBookNode<F>; ORDER_BOOK_MERKLE_LEVELS],

    /*************************/
    /*  IMPACT PRICE HELPERS */
    /*************************/
    // Warning: Witness names impact ask price as bid and vice versa
    #[serde(rename = "ibo")]
    #[serde(default)]
    pub impact_ask_order: Order,

    // Warning: Witness names impact ask price as bid and vice versa
    #[serde(rename = "iao")]
    #[serde(default)]
    pub impact_bid_order: Order,

    #[serde(rename = "ibop")]
    #[serde_as(as = "[_; ORDER_BOOK_MERKLE_LEVELS]")]
    pub impact_ask_order_book_tree_path: [OrderBookNode<F>; ORDER_BOOK_MERKLE_LEVELS],

    #[serde(rename = "iaop")]
    #[serde_as(as = "[_; ORDER_BOOK_MERKLE_LEVELS]")]
    pub impact_bid_order_book_tree_path: [OrderBookNode<F>; ORDER_BOOK_MERKLE_LEVELS],
}

impl Tx<F> {
    pub fn is_empty(&self) -> bool {
        self.tx_type == TX_TYPE_EMPTY
    }
}
