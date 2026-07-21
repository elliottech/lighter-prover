// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::div_rem::CircuitBuilderBiguintDivRem;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::ecdsa::curve::curve_types::AffinePoint;
use crate::ecdsa::curve::ecdsa::{ECDSAPublicKey, ECDSASignature};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::ecdsa::gadgets::ecdsa::{
    CircuitBuilderECDSAPublicKey, CircuitBuilderECDSASignature, ECDSAPublicKeyTarget,
    ECDSAPublicKeyTargetWitness, ECDSASignatureTarget, ECDSASignatureTargetWitness,
};
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::eddsa::schnorr::{SchnorrSigTarget, SchnorrSigTargetWitness};
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::hints::CircuitBuilderHints;
use crate::matching_engine::{execute_matching, get_impact_prices, get_order_book_path_delta};
use crate::merkle_helpers::{
    account_client_order_index_to_merkle_path, account_index_to_merkle_path,
    account_order_index_to_merkle_path, api_key_index_to_merkle_path, asset_index_to_merkle_path,
    conditional_verify_merkle_proof, market_index_to_merkle_path,
    perps_market_index_to_merkle_path, recalculate_root, try_verify_merkle_proof,
    verify_merkle_proof,
};
use crate::order_book_tree_helpers::{
    order_indexes_to_merkle_path, recalculate_order_book_tree_root,
    verify_order_book_tree_merkle_proof,
};
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget, WitnessSigned};
use crate::transactions::internal_cancel_all_orders::{
    InternalCancelAllOrdersTxTarget, InternalCancelAllOrdersTxTargetWitness,
};
use crate::transactions::internal_cancel_order::{
    InternalCancelOrderTxTarget, InternalCancelOrderTxTargetWitness,
};
use crate::transactions::internal_claim_order::{
    InternalClaimOrderTxTarget, InternalClaimOrderTxTargetWitness,
};
use crate::transactions::internal_create_order::{
    InternalCreateOrderTxTarget, InternalCreateOrderTxTargetWitness,
};
use crate::transactions::internal_deleverage::{
    InternalDeleverageTxTarget, InternalDeleverageTxTargetWitness,
};
use crate::transactions::internal_exit_position::{
    InternalExitPositionTxTarget, InternalExitPositionTxTargetWitness,
};
use crate::transactions::internal_liquidate_position::{
    InternalLiquidatePositionTxTarget, InternalLiquidatePositionTxTargetWitness,
};
use crate::transactions::internal_liquidate_spot::{
    InternalLiquidateSpotTxTarget, InternalLiquidateSpotTxTargetWitness,
};
use crate::transactions::internal_pending_unlock::{
    InternalPendingUnlockTxTarget, InternalPendingUnlockTxTargetWitness,
};
use crate::transactions::internal_transfer::{
    InternalTransferTxTarget, InternalTransferTxTargetWitness,
};
use crate::transactions::l1_burn_shares::{L1BurnSharesTxTarget, L1BurnSharesTxTargetWitness};
use crate::transactions::l1_cancel_all_orders::{
    L1CancelAllOrdersTxTarget, L1CancelAllOrdersTxTargetWitness,
};
use crate::transactions::l1_change_pubkey::{
    L1ChangePubKeyTxTarget, L1ChangePubKeyTxTargetWitness,
};
use crate::transactions::l1_create_market::{
    L1CreateMarketTxTarget, L1CreateMarketTxTargetWitness,
};
use crate::transactions::l1_create_order::{L1CreateOrderTxTarget, L1CreateOrderTxTargetWitness};
use crate::transactions::l1_deposit::{L1DepositTxTarget, L1DepositTxTargetWitness};
use crate::transactions::l1_register_asset::{
    L1RegisterAssetTxTarget, L1RegisterAssetTxTargetWitness,
};
use crate::transactions::l1_set_system_config::{
    L1SetSystemConfigTxTarget, L1SetSystemConfigTxTargetWitness,
};
use crate::transactions::l1_update_asset::{L1UpdateAssetTxTarget, L1UpdateAssetTxTargetWitness};
use crate::transactions::l1_update_market::{
    L1UpdateMarketTxTarget, L1UpdateMarketTxTargetWitness,
};
use crate::transactions::l1_withdraw::{L1WithdrawTxTarget, L1WithdrawTxTargetWitness};
use crate::transactions::l2_approve_integrator::{
    L2ApproveIntegratorTxTarget, L2ApproveIntegratorTxTargetWitness,
};
use crate::transactions::l2_burn_shares::{L2BurnSharesTxTarget, L2BurnSharesTxTargetWitness};
use crate::transactions::l2_cancel_all_orders::{
    L2CancelAllOrdersTxTarget, L2CancelAllOrdersTxTargetWitness,
};
use crate::transactions::l2_cancel_order::{L2CancelOrderTxTarget, L2CancelOrderTxTargetWitness};
use crate::transactions::l2_change_pubkey::{
    L2ChangePubKeyTxTarget, L2ChangePubKeyTxTargetWitness,
};
use crate::transactions::l2_create_grouped_orders::{
    L2CreateGroupedOrdersTxTarget, L2CreateGroupedOrdersTxTargetWitness,
};
use crate::transactions::l2_create_order::{L2CreateOrderTxTarget, L2CreateOrderTxTargetWitness};
use crate::transactions::l2_create_public_pool::{
    L2CreatePublicPoolTxTarget, L2CreatePublicPoolTxTargetWitness,
};
use crate::transactions::l2_create_staking_pool::{
    L2CreateStakingPoolTxTarget, L2CreateStakingPoolTxTargetWitness,
};
use crate::transactions::l2_create_sub_account::{
    L2CreateSubAccountTxTarget, L2CreateSubAccountTxTargetWitness,
};
use crate::transactions::l2_force_burn_shares::{
    L2ForceBurnSharesTxTarget, L2ForceBurnSharesTxTargetWitness,
};
use crate::transactions::l2_mint_shares::{L2MintSharesTxTarget, L2MintSharesTxTargetWitness};
use crate::transactions::l2_modify_order::{L2ModifyOrderTxTarget, L2ModifyOrderTxTargetWitness};
use crate::transactions::l2_stake_assets::{L2StakeAssetsTxTarget, L2StakeAssetsTxTargetWitness};
use crate::transactions::l2_strategy_transfer::{
    L2StrategyTransferTxTarget, L2StrategyTransferTxTargetWitness,
};
use crate::transactions::l2_transfer::{L2TransferTxTarget, L2TransferTxTargetWitness};
use crate::transactions::l2_unstake_assets::{
    L2UnstakeAssetsTxTarget, L2UnstakeAssetsTxTargetWitness,
};
use crate::transactions::l2_update_account_asset_config::{
    L2UpdateAccountAssetConfigTxTarget, L2UpdateAccountAssetConfigTxTargetWitness,
};
use crate::transactions::l2_update_account_config::{
    L2UpdateAccountConfigTxTarget, L2UpdateAccountConfigTxTargetWitness,
};
use crate::transactions::l2_update_asset_config::{
    L2UpdateAssetConfigTxTarget, L2UpdateAssetConfigTxTargetWitness,
};
use crate::transactions::l2_update_leverage::{
    L2UpdateLeverageTxTarget, L2UpdateLeverageTxTargetWitness,
};
use crate::transactions::l2_update_margin::{
    L2UpdateMarginTxTarget, L2UpdateMarginTxTargetWitness,
};
use crate::transactions::l2_update_market_config::{
    L2UpdateMarketConfigTxTarget, L2UpdateMarketConfigTxTargetWitness,
};
use crate::transactions::l2_update_public_pool::{
    L2UpdatePublicPoolTxTarget, L2UpdatePublicPoolTxTargetWitness,
};
use crate::transactions::l2_withdraw::{L2WithdrawTxTarget, L2WithdrawTxTargetWitness};
use crate::tx::Tx;
use crate::tx_attributes::{ATTR_SKIP_TX_NONCE, TxAttributesTarget, TxAttributesTargetWitness};
use crate::tx_interface::TransactionTarget;
use crate::types::account::{AccountTarget, AccountTargetWitness};
use crate::types::account_asset::{AccountAssetTarget, AccountAssetTargetWitness};
use crate::types::account_delta::{AccountDeltaTarget, AccountDeltaTargetWitness};
use crate::types::account_margined_asset::{
    AccountMarginedAssetTarget, select_account_margined_asset_target,
};
use crate::types::account_order::{AccountOrderTarget, AccountOrderTargetWitness};
use crate::types::account_position::{
    AccountPositionTarget, PositionWithDelta, random_access_account_position,
};
use crate::types::api_key::{ApiKeyTarget, ApiKeyTargetWitness};
use crate::types::asset::{AssetTarget, apply_diff_assets, diff_assets, random_access_assets};
use crate::types::config::{BIG_U96_LIMBS, Builder, F};
use crate::types::constants::*;
use crate::types::margined_asset::{
    MarginedAssetTarget, apply_diff_margined_assets, diff_margined_assets,
    random_access_margined_assets,
};
use crate::types::market::{MarketTarget, MarketTargetWitness};
use crate::types::market_details::{
    MarketDetailsTarget, apply_diff_market_details, diff_market_details,
    random_access_market_details, select_market_details,
};
use crate::types::order::{OrderTarget, OrderTargetWitness};
use crate::types::order_book_node::{OrderBookNodeTarget, OrderBookNodeTargetWitness};
use crate::types::public_pool::{PublicPoolInfoTarget, PublicPoolShareTarget};
use crate::types::register::{BaseRegisterInfoTarget, RegisterStackTarget};
use crate::types::risk_info::RiskInfoTarget;
use crate::types::system_config::SystemConfigTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::{TxTypeTargets, TxTypeVerifyTargets};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug)]
pub struct TxTarget {
    pub tx_type: Target,

    /***********************/
    /*   L1 Transactions   */
    /***********************/
    pub l1_deposit_tx_target: TransactionTarget<L1DepositTxTarget>,
    pub l1_create_market_tx_target: TransactionTarget<L1CreateMarketTxTarget>,
    pub l1_update_market_tx_target: TransactionTarget<L1UpdateMarketTxTarget>,
    pub l1_cancel_all_orders_tx_target: TransactionTarget<L1CancelAllOrdersTxTarget>,
    pub l1_withdraw_tx_target: TransactionTarget<L1WithdrawTxTarget>,
    pub l1_create_order_tx_target: TransactionTarget<L1CreateOrderTxTarget>,
    pub l1_change_pub_key_tx_target: TransactionTarget<L1ChangePubKeyTxTarget>,
    pub l1_burn_shares_tx_target: TransactionTarget<L1BurnSharesTxTarget>,
    pub l1_register_asset_tx_target: TransactionTarget<L1RegisterAssetTxTarget>,
    pub l1_update_asset_tx_target: TransactionTarget<L1UpdateAssetTxTarget>,
    pub l1_set_system_config_tx_target: TransactionTarget<L1SetSystemConfigTxTarget>,

    /***********************/
    /*   L2 Transactions   */
    /***********************/
    pub l2_change_pub_key_tx_target: TransactionTarget<L2ChangePubKeyTxTarget>,
    pub l2_create_sub_account_tx_target: TransactionTarget<L2CreateSubAccountTxTarget>,
    pub l2_create_public_pool_tx_target: TransactionTarget<L2CreatePublicPoolTxTarget>,
    pub l2_update_public_pool_tx_target: TransactionTarget<L2UpdatePublicPoolTxTarget>,
    pub l2_transfer_tx_target: TransactionTarget<L2TransferTxTarget>,
    pub l2_withdraw_tx_target: TransactionTarget<L2WithdrawTxTarget>,
    pub l2_create_order_tx_target: TransactionTarget<L2CreateOrderTxTarget>,
    pub l2_cancel_order_tx_target: TransactionTarget<L2CancelOrderTxTarget>,
    pub l2_cancel_all_orders_tx_target: TransactionTarget<L2CancelAllOrdersTxTarget>,
    pub l2_modify_order_tx_target: TransactionTarget<L2ModifyOrderTxTarget>,
    pub l2_mint_shares_tx_target: TransactionTarget<L2MintSharesTxTarget>,
    pub l2_burn_shares_tx_target: TransactionTarget<L2BurnSharesTxTarget>,
    pub l2_update_leverage_tx_target: TransactionTarget<L2UpdateLeverageTxTarget>,
    pub l2_create_grouped_orders_tx_target: TransactionTarget<L2CreateGroupedOrdersTxTarget>,
    pub l2_update_margin_tx_target: TransactionTarget<L2UpdateMarginTxTarget>,
    pub l2_create_staking_pool_tx_target: TransactionTarget<L2CreateStakingPoolTxTarget>,
    pub l2_stake_assets_tx_target: TransactionTarget<L2StakeAssetsTxTarget>,
    pub l2_unstake_assets_tx_target: TransactionTarget<L2UnstakeAssetsTxTarget>,
    pub l2_force_burn_shares_tx_target: TransactionTarget<L2ForceBurnSharesTxTarget>,
    pub l2_update_account_config_tx_target: TransactionTarget<L2UpdateAccountConfigTxTarget>,
    pub l2_strategy_transfer_tx_target: TransactionTarget<L2StrategyTransferTxTarget>,
    pub l2_update_market_config_tx_target: TransactionTarget<L2UpdateMarketConfigTxTarget>,
    pub l2_approve_integrator_tx_target: TransactionTarget<L2ApproveIntegratorTxTarget>,
    pub l2_update_asset_config_tx_target: TransactionTarget<L2UpdateAssetConfigTxTarget>,
    pub l2_update_account_asset_config_tx_target:
        TransactionTarget<L2UpdateAccountAssetConfigTxTarget>,

    /*************************/
    /* Internal Transactions */
    /*************************/
    pub internal_claim_order_tx_target: TransactionTarget<InternalClaimOrderTxTarget>,
    pub internal_cancel_order_tx_target: TransactionTarget<InternalCancelOrderTxTarget>,
    pub internal_deleverage_tx_target: TransactionTarget<InternalDeleverageTxTarget>,
    pub internal_exit_position_tx_target: TransactionTarget<InternalExitPositionTxTarget>,
    pub internal_cancel_all_orders_tx_target: TransactionTarget<InternalCancelAllOrdersTxTarget>,
    pub internal_liquidate_position_tx_target: TransactionTarget<InternalLiquidatePositionTxTarget>,
    pub internal_create_order_tx_target: TransactionTarget<InternalCreateOrderTxTarget>,
    pub internal_pending_unlock_tx_target: TransactionTarget<InternalPendingUnlockTxTarget>,
    pub internal_transfer_tx_target: TransactionTarget<InternalTransferTxTarget>,
    pub internal_liquidate_spot_tx_target: TransactionTarget<InternalLiquidateSpotTxTarget>,

    /***********************/
    /*  Transactions Data  */
    /***********************/
    // Signature related data
    pub nonce: Target,
    pub expired_at: Target, // 48 bits
    pub signature: SchnorrSigTarget,
    // Fee targets for trades
    pub taker_fee: SignedTarget,
    pub maker_fee: SignedTarget,
    // L1 signature data
    pub l1_signature: ECDSASignatureTarget<Secp256K1>,
    pub l1_pub_key: ECDSAPublicKeyTarget<Secp256K1>,
    /***********************/
    /*  State Tree Leaves  */
    /***********************/
    pub accounts_before: [AccountTarget; NB_ACCOUNTS_PER_TX],
    pub account_assets_before: [[AccountAssetTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],
    pub accounts_delta_before: [AccountDeltaTarget; NB_ACCOUNTS_PER_TX],
    pub api_key_before: ApiKeyTarget,
    pub account_order_before: AccountOrderTarget,
    pub market_before: MarketTarget,
    pub order_before: OrderTarget,
    pub asset_indices: [Target; NB_ASSETS_PER_TX],

    /*****************************/
    /*  State Tree Merkle Proofs */
    /*****************************/
    pub account_tree_merkle_proofs: [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX],
    pub account_pub_data_tree_merkle_proofs:
        [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX],
    pub account_delta_tree_merkle_proofs:
        [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX],
    pub asset_tree_merkle_proofs:
        [[[HashOutTarget; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],
    pub public_asset_tree_merkle_proofs:
        [[[HashOutTarget; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],
    pub asset_delta_tree_merkle_proofs:
        [[[HashOutTarget; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],

    pub position_delta_merkle_proofs:
        [[HashOutTarget; POSITION_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX - 1],
    pub api_key_tree_merkle_proof: [HashOutTarget; API_KEY_MERKLE_LEVELS],
    pub account_orders_tree_merkle_proof:
        [[HashOutTarget; ACCOUNT_ORDERS_MERKLE_LEVELS]; NB_ACCOUNT_ORDERS_PATHS_PER_TX],
    pub market_tree_merkle_proof: [HashOutTarget; MARKET_MERKLE_LEVELS],
    pub order_book_tree_path: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],

    /*************************/
    /*  Impact Price Helpers */
    /*************************/
    pub impact_ask_order: OrderTarget,
    pub impact_bid_order: OrderTarget,
    pub impact_ask_order_book_tree_path: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    pub impact_bid_order_book_tree_path: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],

    /******************/
    /*  TX ATTRIBUTES */
    /******************/
    pub attributes: TxAttributesTarget,
}

impl TxTarget {
    /// Initializes the transaction virtual targets
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            tx_type: builder.add_virtual_target(),

            /***********************/
            /*   L1 Transactions   */
            /***********************/
            l1_deposit_tx_target: TransactionTarget::new(L1DepositTxTarget::new(builder)),
            l1_create_market_tx_target: TransactionTarget::new(L1CreateMarketTxTarget::new(
                builder,
            )),
            l1_update_market_tx_target: TransactionTarget::new(L1UpdateMarketTxTarget::new(
                builder,
            )),
            l1_cancel_all_orders_tx_target: TransactionTarget::new(L1CancelAllOrdersTxTarget::new(
                builder,
            )),
            l1_withdraw_tx_target: TransactionTarget::new(L1WithdrawTxTarget::new(builder)),
            l1_create_order_tx_target: TransactionTarget::new(L1CreateOrderTxTarget::new(builder)),
            l1_change_pub_key_tx_target: TransactionTarget::new(L1ChangePubKeyTxTarget::new(
                builder,
            )),
            l1_burn_shares_tx_target: TransactionTarget::new(L1BurnSharesTxTarget::new(builder)),
            l1_register_asset_tx_target: TransactionTarget::new(L1RegisterAssetTxTarget::new(
                builder,
            )),
            l1_update_asset_tx_target: TransactionTarget::new(L1UpdateAssetTxTarget::new(builder)),
            l1_set_system_config_tx_target: TransactionTarget::new(L1SetSystemConfigTxTarget::new(
                builder,
            )),

            /***********************/
            /*   L2 Transactions   */
            /***********************/
            l2_change_pub_key_tx_target: TransactionTarget::new(L2ChangePubKeyTxTarget::new(
                builder,
            )),
            l2_create_sub_account_tx_target: TransactionTarget::new(
                L2CreateSubAccountTxTarget::new(builder),
            ),
            l2_create_public_pool_tx_target: TransactionTarget::new(
                L2CreatePublicPoolTxTarget::new(builder),
            ),
            l2_update_public_pool_tx_target: TransactionTarget::new(
                L2UpdatePublicPoolTxTarget::new(builder),
            ),
            l2_transfer_tx_target: TransactionTarget::new(L2TransferTxTarget::new(builder)),
            l2_withdraw_tx_target: TransactionTarget::new(L2WithdrawTxTarget::new(builder)),
            l2_create_order_tx_target: TransactionTarget::new(L2CreateOrderTxTarget::new(builder)),
            l2_cancel_order_tx_target: TransactionTarget::new(L2CancelOrderTxTarget::new(builder)),
            l2_cancel_all_orders_tx_target: TransactionTarget::new(L2CancelAllOrdersTxTarget::new(
                builder,
            )),
            l2_modify_order_tx_target: TransactionTarget::new(L2ModifyOrderTxTarget::new(builder)),
            l2_mint_shares_tx_target: TransactionTarget::new(L2MintSharesTxTarget::new(builder)),
            l2_burn_shares_tx_target: TransactionTarget::new(L2BurnSharesTxTarget::new(builder)),
            l2_update_leverage_tx_target: TransactionTarget::new(L2UpdateLeverageTxTarget::new(
                builder,
            )),
            l2_create_grouped_orders_tx_target: TransactionTarget::new(
                L2CreateGroupedOrdersTxTarget::new(builder),
            ),
            l2_update_margin_tx_target: TransactionTarget::new(L2UpdateMarginTxTarget::new(
                builder,
            )),
            l2_create_staking_pool_tx_target: TransactionTarget::new(
                L2CreateStakingPoolTxTarget::new(builder),
            ),
            l2_stake_assets_tx_target: TransactionTarget::new(L2StakeAssetsTxTarget::new(builder)),
            l2_unstake_assets_tx_target: TransactionTarget::new(L2UnstakeAssetsTxTarget::new(
                builder,
            )),
            l2_force_burn_shares_tx_target: TransactionTarget::new(L2ForceBurnSharesTxTarget::new(
                builder,
            )),
            l2_update_account_config_tx_target: TransactionTarget::new(
                L2UpdateAccountConfigTxTarget::new(builder),
            ),
            l2_strategy_transfer_tx_target: TransactionTarget::new(
                L2StrategyTransferTxTarget::new(builder),
            ),
            l2_update_market_config_tx_target: TransactionTarget::new(
                L2UpdateMarketConfigTxTarget::new(builder),
            ),
            l2_approve_integrator_tx_target: TransactionTarget::new(
                L2ApproveIntegratorTxTarget::new(builder),
            ),
            l2_update_asset_config_tx_target: TransactionTarget::new(
                L2UpdateAssetConfigTxTarget::new(builder),
            ),
            l2_update_account_asset_config_tx_target: TransactionTarget::new(
                L2UpdateAccountAssetConfigTxTarget::new(builder),
            ),

            /*************************/
            /* Internal Transactions */
            /*************************/
            internal_claim_order_tx_target: TransactionTarget::new(
                InternalClaimOrderTxTarget::new(builder),
            ),
            internal_cancel_order_tx_target: TransactionTarget::new(
                InternalCancelOrderTxTarget::new(builder),
            ),
            internal_deleverage_tx_target: TransactionTarget::new(InternalDeleverageTxTarget::new(
                builder,
            )),
            internal_exit_position_tx_target: TransactionTarget::new(
                InternalExitPositionTxTarget::new(builder),
            ),
            internal_cancel_all_orders_tx_target: TransactionTarget::new(
                InternalCancelAllOrdersTxTarget::new(builder),
            ),
            internal_liquidate_position_tx_target: TransactionTarget::new(
                InternalLiquidatePositionTxTarget::new(builder),
            ),
            internal_create_order_tx_target: TransactionTarget::new(
                InternalCreateOrderTxTarget::new(builder),
            ),
            internal_pending_unlock_tx_target: TransactionTarget::new(
                InternalPendingUnlockTxTarget::new(builder),
            ),
            internal_transfer_tx_target: TransactionTarget::new(InternalTransferTxTarget::new(
                builder,
            )),
            internal_liquidate_spot_tx_target: TransactionTarget::new(
                InternalLiquidateSpotTxTarget::new(builder),
            ),

            /***********************/
            /*  Transactions Data  */
            /***********************/
            nonce: builder.add_virtual_target(),
            expired_at: builder.add_virtual_target(),
            signature: SchnorrSigTarget::new(builder),
            taker_fee: builder.add_virtual_signed_target(),
            maker_fee: builder.add_virtual_signed_target(),

            l1_signature: builder.add_virtual_ecdsa_target(),
            l1_pub_key: builder.add_virtual_ecdsa_public_key(),

            /***********************/
            /*  State Tree Leaves  */
            /***********************/
            accounts_before: [
                AccountTarget::new(builder),
                AccountTarget::new(builder),
                AccountTarget::new_fee_account(builder),
            ],
            account_assets_before: core::array::from_fn(|_| {
                core::array::from_fn(|_| AccountAssetTarget::new(builder))
            }),
            accounts_delta_before: [
                AccountDeltaTarget::new(builder),
                AccountDeltaTarget::new(builder),
                AccountDeltaTarget::new_fee_account(builder),
            ],
            api_key_before: ApiKeyTarget::new(builder),
            account_order_before: AccountOrderTarget::new(builder),
            market_before: MarketTarget::new(builder),
            order_before: OrderTarget::new(builder),
            asset_indices: core::array::from_fn(|_| builder.add_virtual_target()),

            /*****************************/
            /*  State Tree Merkle Proofs */
            /*****************************/
            account_tree_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| builder.add_virtual_hash())
            }),
            account_delta_tree_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| builder.add_virtual_hash())
            }),
            position_delta_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| builder.add_virtual_hash())
            }),
            account_pub_data_tree_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| builder.add_virtual_hash())
            }),
            api_key_tree_merkle_proof: array::from_fn(|_| builder.add_virtual_hash()),
            account_orders_tree_merkle_proof: array::from_fn(|_| {
                array::from_fn(|_| builder.add_virtual_hash())
            }),
            market_tree_merkle_proof: array::from_fn(|_| builder.add_virtual_hash()),
            order_book_tree_path: array::from_fn(|_| OrderBookNodeTarget::new(builder)),
            asset_tree_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| array::from_fn(|_| builder.add_virtual_hash()))
            }),
            public_asset_tree_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| array::from_fn(|_| builder.add_virtual_hash()))
            }),
            asset_delta_tree_merkle_proofs: array::from_fn(|_| {
                array::from_fn(|_| array::from_fn(|_| builder.add_virtual_hash()))
            }),

            /*************************/
            /*  Impact Price Helpers */
            /*************************/
            impact_ask_order: OrderTarget::new(builder),
            impact_bid_order: OrderTarget::new(builder),
            impact_ask_order_book_tree_path: array::from_fn(|_| OrderBookNodeTarget::new(builder)),
            impact_bid_order_book_tree_path: array::from_fn(|_| OrderBookNodeTarget::new(builder)),

            /******************/
            /*  TX ATTRIBUTES */
            /******************/
            attributes: TxAttributesTarget::new(builder),
        }
    }

    pub fn define(
        &mut self,
        _index: usize,
        chain_id: u32,
        builder: &mut Builder,
        block_created_at: Target,
        system_config_before: &SystemConfigTarget,
        register_stack_before: &RegisterStackTarget,
        all_assets_before: &[AssetTarget; ASSET_LIST_SIZE],
        all_margined_assets_before: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
        all_market_details_before: &[MarketDetailsTarget; POSITION_LIST_SIZE],
        account_tree_root_before: HashOutTarget,
        account_pub_data_tree_root_before: HashOutTarget,
        account_delta_tree_root_before: HashOutTarget,
        market_tree_root_before: HashOutTarget,
    ) -> (
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX], // priority operation's public data
        BoolTarget,                                                // is there a priority operation
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE], // on chain operation's public data
        BoolTarget,                                          // is there a on chain operation
        SystemConfigTarget,                                  // system config after
        RegisterStackTarget,                                 // register stack after
        [AssetTarget; ASSET_LIST_SIZE],                      // all assets after
        [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],     // all margined assets after
        [MarketDetailsTarget; POSITION_LIST_SIZE],           // all market details after
        HashOutTarget,                                       // account tree root after
        HashOutTarget, // account public data delta tree root after
        HashOutTarget, // account public data tree root after
        HashOutTarget, // market tree root after
    ) {
        let tx_type = TxTypeTargets::new(builder, self.tx_type);
        let tx_hash = self.select_tx_hash(builder, &tx_type, chain_id);
        let account_pk = self.select_account_pk(builder, &tx_type);
        let partial_main_account = self.select_partial_main_account(builder, &tx_type);

        // Perform common verifications for the transaction.
        tx_type.verify(
            builder,
            &TxTypeVerifyTargets {
                expired_at: self.expired_at,
                block_created_at,
                nonce: self.nonce,
                api_key_before_nonce: self.api_key_before.nonce,
                signature: self.signature.clone(),
                account_pk,
                tx_hash,
                instruction_type: register_stack_before[0].instruction_type,
                tx_sender_account_partial: partial_main_account,
                sub_account_index: self.accounts_before[SUB_ACCOUNT_ID].account_index,
                skip_tx_nonce: self.attributes.get(ATTR_SKIP_TX_NONCE),
            },
        );

        /**********************************/
        /*  Initialize Helper State Data  */
        /**********************************/
        let assets_before: [AssetTarget; NB_ASSETS_PER_TX] = core::array::from_fn(|i| {
            random_access_assets(builder, self.asset_indices[i], all_assets_before.to_vec())
        });
        // Load the margin asset belongs to first asset loaded. For L1 register/update asset, load the target margin index
        let next_margin_asset_index =
            self.get_next_margin_asset_index(builder, all_margined_assets_before);
        let mut first_asset_margin_index = assets_before[0].margin_index(builder);
        first_asset_margin_index = builder.select(
            tx_type.is_l1_register_asset,
            next_margin_asset_index,
            first_asset_margin_index,
        );
        let second_asset_margin_index = assets_before[1].margin_index(builder);
        let margined_asset_before = [
            random_access_margined_assets(
                builder,
                first_asset_margin_index,
                all_margined_assets_before,
            ),
            random_access_margined_assets(
                builder,
                second_asset_margin_index,
                all_margined_assets_before,
            ),
        ];
        let market_details_before =
            self.get_market_details_with_random_access(builder, all_market_details_before);
        let positions_with_pub_data_before: [PositionWithDelta; NB_ACCOUNTS_PER_TX - 1] =
            PositionWithDelta::new_positions_with_pub_data_from_accounts(
                builder,
                self.market_before.perps_market_index,
                &self.accounts_before[..NB_ACCOUNTS_PER_TX - 1],
                &self.accounts_delta_before[..NB_ACCOUNTS_PER_TX - 1],
            );

        let account_margined_assets_before: [[AccountMarginedAssetTarget; NB_ASSETS_PER_TX];
            NB_ACCOUNTS_PER_TX] = AccountTarget::get_margined_asset_balances(
            builder,
            &self.accounts_before,
            &assets_before,
            first_asset_margin_index,
        );
        let is_asset_used_as_margin: [[BoolTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX] =
            core::array::from_fn(|i| {
                account_margined_assets_before[i]
                    .iter()
                    .enumerate()
                    .map(|(j, account_margined_asset)| {
                        let is_usdc =
                            builder.is_equal_constant(self.asset_indices[j], USDC_ASSET_INDEX);
                        builder.or(
                            is_usdc,
                            BoolTarget::new_unsafe(account_margined_asset.margin_mode),
                        )
                    })
                    .collect::<Vec<BoolTarget>>()
                    .try_into()
                    .unwrap()
            });

        let risk_infos_before = self.get_risk_infos_before(
            builder,
            &tx_type,
            &positions_with_pub_data_before,
            &market_details_before,
            all_market_details_before,
            all_margined_assets_before,
        );
        let (strategy_indexes, strategies_before) =
            self.get_strategies_before(builder, &tx_type, &market_details_before);
        let order_path_helper = order_indexes_to_merkle_path(
            builder,
            self.order_before.price_index,
            self.order_before.nonce_index,
        );

        let public_pool_share_before = self.accounts_before[OWNER_ACCOUNT_ID]
            .get_public_pool_share(builder, self.accounts_before[SUB_ACCOUNT_ID].account_index);
        let (mut position_bucket_hashes, old_account_hashes, old_account_empty_infos) =
            self.get_old_account_hashes(builder);
        let old_account_asset_hashes = self.get_old_asset_hashes(builder);
        let old_position_delta_hashes: [HashOutTarget; NB_ACCOUNTS_PER_TX - 1] =
            array::from_fn(|i| positions_with_pub_data_before[i].delta.hash(builder));

        self.attributes.sanitize_and_normalize(
            builder,
            &self.accounts_before[OWNER_ACCOUNT_ID],
            &self.market_before,
            system_config_before,
            block_created_at,
        );

        // /******************************/
        // /*  Initialize Tx State Data  */
        // /******************************/
        let tx_state = &mut TxState {
            first_asset_margin_index,
            next_margin_asset_index,
            new_instructions: [BaseRegisterInfoTarget::empty(builder); NEW_INSTRUCTIONS_MAX_SIZE],
            new_instructions_count: builder.zero(),
            register_stack: *register_stack_before,
            system_config: *system_config_before,
            accounts: self.accounts_before.clone(),
            accounts_delta: self.accounts_delta_before.clone(),
            is_new_account: old_account_empty_infos,
            market: self.market_before.clone(),
            market_details: market_details_before.clone(),
            order: self.order_before.clone(),
            order_book_tree_path: self.order_book_tree_path.clone(),
            positions: core::array::from_fn(|i| positions_with_pub_data_before[i].position.clone()),
            risk_infos: risk_infos_before.clone(),
            strategies: strategies_before.clone(),
            is_asset_used_as_margin,
            order_path_helper,
            matching_engine_flag: builder._false(),
            update_impact_prices_flag: builder._false(),
            taker_fee: self.taker_fee,
            maker_fee: self.maker_fee,
            api_key: self.api_key_before.clone(),
            account_order: self.account_order_before.clone(),
            account_assets: self.account_assets_before.clone(),
            account_margined_assets: account_margined_assets_before.clone(),
            assets: assets_before.clone(),
            margined_asset: margined_asset_before.clone(),
            asset_indices: self.asset_indices,
            block_timestamp: block_created_at,
            is_sender_receiver_different: builder.is_not_equal(
                self.accounts_before[SENDER_ACCOUNT_ID].account_index,
                self.accounts_before[RECEIVER_ACCOUNT_ID].account_index,
            ),
            fee_account_is_taker: builder.is_equal(
                self.accounts_before[FEE_ACCOUNT_ID].account_index,
                self.accounts_before[TAKER_ACCOUNT_ID].account_index,
            ),
            fee_account_is_maker: builder.is_equal(
                self.accounts_before[FEE_ACCOUNT_ID].account_index,
                self.accounts_before[MAKER_ACCOUNT_ID].account_index,
            ),
            is_cloid_unique: self.get_is_cloid_unique_infos(builder, &tx_type),
            public_pool_share: public_pool_share_before,
            apply_pool_share_delta_flag: builder._false(),
            between_strategies_flag: builder
                .is_equal_constant(self.tx_type, TX_TYPE_L2_STRATEGY_TRANSFER as u64),
            attributes: self.attributes.clone(),
        };

        self.validate_asset_indices(builder, tx_state);

        self.verify_transactions(builder, &tx_type, tx_state);

        /*******************************/
        /*      APPLY TRANSACTION      */
        /*******************************/
        self.apply_transaction(builder, tx_state, &tx_type);
        tx_state.push_instruction_stack::<INSERT_MAX_THREE_REGISTERS>(builder);

        execute_matching(builder, tx_state, block_created_at);
        tx_state.push_instruction_stack::<NEW_INSTRUCTIONS_MAX_SIZE>(builder);

        /*******************************/
        /*      GENERATE PUB DATA      */
        /*******************************/
        let (priority_operations_pub_data_exists, priority_operations_pub_data) =
            self.select_priority_operations_pub_data(builder, &tx_type);
        let (on_chain_pub_data_exists, on_chain_operations_pub_data) =
            self.select_on_chain_operations_pub_data(builder, tx_state);

        /*******************************/
        /*      PUSH STATE DELTAS      */
        /*******************************/
        let position_usdc_deltas = self.apply_position_delta(
            builder,
            tx_state,
            &positions_with_pub_data_before,
            &mut position_bucket_hashes,
        );

        self.apply_aggregated_balance_deltas(builder, tx_state, &account_margined_assets_before);

        self.apply_margin_balance_deltas(
            builder,
            tx_state,
            &[first_asset_margin_index, second_asset_margin_index],
        );

        self.apply_public_pool_share_delta(builder, tx_state, &public_pool_share_before);

        self.apply_pool_info_deltas(builder, tx_state);

        self.apply_strategy_deltas(builder, tx_state, strategy_indexes);

        self.apply_account_deltas(builder, tx_state, &position_usdc_deltas);

        self.update_impact_prices(builder, tx_state, &market_details_before);

        let current_all_market_details = self.update_market_details(
            builder,
            all_market_details_before,
            &market_details_before,
            &tx_state.market_details,
        );

        let (current_all_assets, current_all_margined_assets) = self.update_assets(
            builder,
            all_assets_before,
            &assets_before,
            &tx_state.assets,
            all_margined_assets_before,
            &[tx_state.first_asset_margin_index, second_asset_margin_index],
            &margined_asset_before
                .iter()
                .take(2)
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            &tx_state
                .margined_asset
                .iter()
                .take(2)
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );

        /*************************/
        /*  VERIFY STATE LEAVES  */
        /*************************/
        self.verify_position_delta_merkle_proofs(builder, tx_state, &old_position_delta_hashes);

        self.verify_api_key_merkle_proof(builder, tx_state, &tx_type);

        self.verify_account_orders_merkle_proof(builder, tx_state);

        self.verify_assets_merkle_proofs(builder, tx_state, &old_account_asset_hashes);

        let (
            current_account_tree_root,
            current_account_pub_data_tree_root,
            current_account_delta_tree_root,
        ) = self.verify_account_and_pub_data_merkle_proofs(
            builder,
            tx_state,
            account_tree_root_before,
            account_pub_data_tree_root_before,
            account_delta_tree_root_before,
            &old_account_hashes,
            &position_bucket_hashes,
        );

        let current_market_tree_root =
            self.verify_market_and_order_book_proofs(builder, tx_state, market_tree_root_before);

        (
            priority_operations_pub_data,
            priority_operations_pub_data_exists,
            on_chain_operations_pub_data,
            on_chain_pub_data_exists,
            tx_state.system_config,
            tx_state.register_stack,
            current_all_assets,
            current_all_margined_assets,
            current_all_market_details,
            current_account_tree_root,
            current_account_pub_data_tree_root,
            current_account_delta_tree_root,
            current_market_tree_root,
        )

        // (
        //     [builder.zero_u8(); MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
        //     builder._false(),
        //     [builder.zero_u8(); ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
        //     builder._false(),
        //     SystemConfigTarget::empty(builder),
        //     RegisterStackTarget::empty(builder),
        //     all_assets_before.clone(),
        //     all_margined_assets_before.clone(),
        //     all_market_details_before.clone(),
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        // )
    }

    fn get_next_margin_asset_index(
        &self,
        builder: &mut Builder,
        all_margined_assets_before: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    ) -> Target {
        let mut margin_index = builder.constant_u64(NIL_MARGIN_ASSET_INDEX);
        let mut applied = builder._false();
        for i in 0..MARGINED_ASSET_LIST_SIZE {
            let i_target = builder.constant_usize(i);
            let is_empty = builder.is_zero(all_margined_assets_before[i].asset_index);
            let flag = builder.and_not(is_empty, applied);
            margin_index = builder.select(flag, i_target, margin_index);
            applied = builder.or(applied, flag);
        }
        margin_index
    }

    /// Computes the two risk slots we use in tx_state
    ///
    /// 1. Non-insurance fund accounts use regular collateral and markets' strategy ids do not matter (cross risk).
    /// 2. Insurance funds use strategy balance as collateral and risk calculation respects strategy ids.
    /// 3. For transactions that require calculating the share value of a public pool, we need the cross risk of the
    ///    insurance fund. These are of two types:
    ///    a. Minting shares to the insurance fund
    ///    In this case we need cross risks of both owner and the insurance
    ///    fund accounts. We place the owner's cross risk at index 0, and the insurance fund's cross risk at index 1.
    ///    b. Burning shares from the insurance fund
    ///    In this case we don't need owner's risk (as their TAV will increase), but we need the insurance fund's
    ///    cross risk and default strategy risk. Hence we treat the insurance fund account as a regular account and use the
    ///    whole collateral and place this risk info at index 0, and then we calculate the default strategy risk at index 1.
    /// 4. For strategy balance transfers, we set strategy id to `from_strategy_index`.
    ///
    /// - Accounts and positions are partially copied as risk calculations don't use all fields.
    /// - Insurance funds can't have isolated positions. `current_market_details` is only relevant for isolated
    ///   positions so we can just pass it directly.
    /// - For transactions that don't utilize a market (empty market info), strategy index will come as default (0).
    /// - Strategy id being 0 means calculating the cross risk for non-insurance fund accounts, and the default
    ///   strategy for insurance fund accounts.
    fn get_risk_infos_before(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
        positions_with_pub_data_before: &[PositionWithDelta; NB_ACCOUNTS_PER_TX - 1],
        current_market_details_before: &MarketDetailsTarget,
        all_market_details_before: &[MarketDetailsTarget; POSITION_LIST_SIZE],
        all_margined_assets_before: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    ) -> [RiskInfoTarget; NB_ACCOUNTS_PER_TX - 1] {
        let default_strategy_index = builder.constant_usize(DEFAULT_STRATEGY_INDEX);

        // Select among cross risk, insurance fund strategy risk, and pool account cross risk (if share-burning transaction).
        let (partial_account_0, strategy_index_0, all_market_details_0, current_market_details_0) = {
            // Override strategy indices with 0 unless the owner account is an insurance fund. For strategy transfer transactions,
            // use from_strategy_index as the strategy index for the insurance fund account.
            let use_strategy = {
                let assertions = [
                    builder.not(tx_type.is_share_burn_tx), // First slot will be cross risk of sub account, not the owner
                    builder.is_equal_constant(
                        self.accounts_before[OWNER_ACCOUNT_ID].account_type,
                        INSURANCE_FUND_ACCOUNT_TYPE as u64,
                    ),
                ];
                builder.multi_and(&assertions)
            };

            let strategy_index = {
                let mut strategy_index = builder.select(
                    tx_type.is_l2_strategy_transfer,
                    self.l2_strategy_transfer_tx_target
                        .inner
                        .from_strategy_index,
                    current_market_details_before.strategy_index,
                );
                strategy_index = builder.select(
                    tx_type.is_internal_transfer,
                    self.internal_transfer_tx_target.inner.strategy_index,
                    strategy_index,
                );
                builder.select(use_strategy, strategy_index, default_strategy_index)
            };

            let mut margined_assets = self.accounts_before[OWNER_ACCOUNT_ID]
                .margined_assets
                .clone();
            let usdc_collateral = {
                // use insurance fund's whole collateral if is_share_burn_tx
                // if not is_share_burn_tx; use owner account collateral, or strategy balance if insurance fund
                let relevant_collateral = self.accounts_before[OWNER_ACCOUNT_ID]
                    .get_relevant_usdc_collateral(builder, strategy_index);
                builder.select_bigint(
                    tx_type.is_share_burn_tx,
                    &self.accounts_before[SUB_ACCOUNT_ID]
                        .get_margined_asset_balance_const(USDC_MARGIN_ASSET_INDEX),
                    &relevant_collateral,
                )
            };
            margined_assets[USDC_MARGIN_ASSET_INDEX].balance = usdc_collateral;
            let zero_bigint = builder.zero_bigint();
            // For burn shares the cross risk uses the pool's USDC (selected above) and
            // traditionally zeroes non-USDC.  But the insurance fund can now hold
            // non-USDC spot in MarginBalance, and those must count towards TPV for
            // correct share pricing.  So: keep the pool's non-USDC when the pool is
            // the insurance fund, zero them otherwise (original behaviour).
            let is_pool_insurance_fund = builder.is_equal_constant(
                self.accounts_before[SUB_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let use_pool_non_usdc = builder.and(tx_type.is_share_burn_tx, is_pool_insurance_fund);
            let zero_non_usdc = builder.and_not(tx_type.is_share_burn_tx, is_pool_insurance_fund);
            for i in 1..MARGINED_ASSET_LIST_SIZE {
                let zeroed =
                    builder.select_bigint(zero_non_usdc, &zero_bigint, &margined_assets[i].balance);
                margined_assets[i].balance = builder.select_bigint(
                    use_pool_non_usdc,
                    &self.accounts_before[SUB_ACCOUNT_ID].margined_assets[i].balance,
                    &zeroed,
                );
            }

            // Put the pool positions for share-burning txs. If account is not insurance fund, positions' strategy
            // indices will be zero anyways, don't override them. Furthermore, insurance fund can't have isolated
            // positions, so we don't select `margin_mode` (0 means cross) and `allocated_margin`.
            let positions = self.accounts_before[SUB_ACCOUNT_ID]
                .positions
                .iter()
                .zip(self.accounts_before[OWNER_ACCOUNT_ID].positions.iter())
                .map(|(sub_account_position, owner_account_position)| {
                    AccountPositionTarget::partial_select_for_cross_risk(
                        builder,
                        tx_type.is_share_burn_tx,
                        sub_account_position,
                        owner_account_position,
                    )
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let partial_account = AccountTarget {
                positions,
                margined_assets,
                ..AccountTarget::default() // Partial
            };

            // Override strategy indices in market details with 0 for non-insurance fund accounts
            let all_market_details =
                all_market_details_before
                    .clone()
                    .map(|md| MarketDetailsTarget {
                        strategy_index: builder.select(
                            use_strategy,
                            md.strategy_index,
                            default_strategy_index,
                        ),
                        ..md
                    });

            let current_market_details = MarketDetailsTarget {
                strategy_index: builder.select(
                    use_strategy,
                    current_market_details_before.strategy_index,
                    default_strategy_index,
                ),
                ..current_market_details_before.clone()
            };

            (
                partial_account,
                strategy_index,
                all_market_details,
                current_market_details,
            )
        };

        // Select between normal account cross risk, insurance fund strategy risk, or insurance fund cross risk
        let (partial_account_1, strategy_index_1, all_market_details_1, current_market_details_1) = {
            let is_second_account_insurance_fund = builder.is_equal_constant(
                self.accounts_before[SUB_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let use_strategy =
                builder.and_not(is_second_account_insurance_fund, tx_type.is_l2_mint_shares);

            // Override strategy indices with 0 unless the sub account is an insurance fund.
            // Strategy transfer transactions don't use a sub account and the target strategy balance increases,
            // so we don't need to check for l2_strategy_transfer here.
            let strategy_index = builder.select(
                use_strategy,
                current_market_details_before.strategy_index,
                default_strategy_index,
            );

            let mut margined_assets = self.accounts_before[SUB_ACCOUNT_ID].margined_assets.clone();
            let usdc_collateral = {
                // use insurance fund's whole collateral if use_strategy is false
                let relevant_collateral = self.accounts_before[SUB_ACCOUNT_ID]
                    .get_relevant_usdc_collateral(builder, strategy_index);
                builder.select_bigint(
                    use_strategy,
                    &relevant_collateral,
                    &self.accounts_before[SUB_ACCOUNT_ID]
                        .get_margined_asset_balance_const(USDC_MARGIN_ASSET_INDEX),
                )
            };
            margined_assets[USDC_MARGIN_ASSET_INDEX].balance = usdc_collateral;

            let partial_account = AccountTarget {
                positions: self.accounts_before[SUB_ACCOUNT_ID].positions.clone(),
                margined_assets,
                ..AccountTarget::default() // Partial
            };

            let all_market_details =
                all_market_details_before
                    .clone()
                    .map(|md| MarketDetailsTarget {
                        strategy_index: builder.select(
                            use_strategy,
                            md.strategy_index,
                            default_strategy_index,
                        ),
                        ..md
                    });

            let current_market_details = MarketDetailsTarget {
                strategy_index: builder.select(
                    use_strategy,
                    current_market_details_before.strategy_index,
                    default_strategy_index,
                ),
                ..current_market_details_before.clone()
            };

            (
                partial_account,
                strategy_index,
                all_market_details,
                current_market_details,
            )
        };

        [
            RiskInfoTarget::new(
                builder,
                &partial_account_0,
                &positions_with_pub_data_before[OWNER_ACCOUNT_ID].position,
                &current_market_details_0,
                &all_market_details_0,
                all_margined_assets_before,
                strategy_index_0,
            ),
            RiskInfoTarget::new(
                builder,
                &partial_account_1,
                &positions_with_pub_data_before[SUB_ACCOUNT_ID].position,
                &current_market_details_1,
                &all_market_details_1,
                all_margined_assets_before,
                strategy_index_1,
            ),
        ]
    }

    fn get_strategies_before(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
        current_market_details: &MarketDetailsTarget,
    ) -> (
        [Target; NB_ACCOUNTS_PER_TX],
        [BigIntTarget; NB_ACCOUNTS_PER_TX],
    ) {
        let default_strategy_index = builder.constant_usize(DEFAULT_STRATEGY_INDEX);
        let spot_strategy_index = builder.constant_usize(INSURANCE_FUND_SPOT_STRATEGY_INDEX);

        let strategy_index_0 = {
            let is_owner_insurance_fund = builder.is_equal_constant(
                self.accounts_before[OWNER_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let use_strategy = {
                let assertions = [
                    builder.not(tx_type.is_share_burn_tx), // First slot will be cross risk of sub account, not the owner
                    is_owner_insurance_fund,
                ];
                builder.multi_and(&assertions)
            };

            let mut result = builder.select(
                tx_type.is_l2_strategy_transfer,
                self.l2_strategy_transfer_tx_target
                    .inner
                    .from_strategy_index,
                current_market_details.strategy_index,
            );
            result = builder.select(
                tx_type.is_internal_transfer,
                self.internal_transfer_tx_target.inner.strategy_index,
                result,
            );
            // Insurance fund spot orders use strategy-1 for USDC instead of the market's strategy
            let is_market_spot =
                builder.is_equal_constant(self.market_before.market_type, MARKET_TYPE_SPOT);
            let is_insurance_fund_spot = builder.and(is_owner_insurance_fund, is_market_spot);
            result = builder.select(is_insurance_fund_spot, spot_strategy_index, result);
            builder.select(use_strategy, result, default_strategy_index)
        };

        let strategy_index_1 = {
            let is_second_account_insurance_fund = builder.is_equal_constant(
                self.accounts_before[SUB_ACCOUNT_ID].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let use_strategy =
                builder.and_not(is_second_account_insurance_fund, tx_type.is_l2_mint_shares);

            // Override strategy indices with 0 unless the sub account is an insurance fund.
            // Strategy transfer transactions don't use a sub account and the target strategy balance increases,
            // so we don't need to check for l2_strategy_transfer here.
            let mut result = builder.select(
                tx_type.is_l2_strategy_transfer,
                self.l2_strategy_transfer_tx_target.inner.to_strategy_index,
                current_market_details.strategy_index,
            );
            result = builder.select(
                tx_type.is_internal_transfer,
                self.internal_transfer_tx_target.inner.strategy_index,
                result,
            );
            // Insurance fund spot orders (maker side) use strategy-1 for USDC
            let is_market_spot =
                builder.is_equal_constant(self.market_before.market_type, MARKET_TYPE_SPOT);
            let is_insurance_fund_spot =
                builder.and(is_second_account_insurance_fund, is_market_spot);
            result = builder.select(is_insurance_fund_spot, spot_strategy_index, result);
            builder.select(use_strategy, result, default_strategy_index)
        };

        (
            [
                strategy_index_0,
                strategy_index_1,
                current_market_details.strategy_index,
            ],
            [
                self.accounts_before[OWNER_ACCOUNT_ID]
                    .public_pool_info
                    .get_strategy_balance(builder, strategy_index_0),
                self.accounts_before[SUB_ACCOUNT_ID]
                    .public_pool_info
                    .get_strategy_balance(builder, strategy_index_1),
                self.accounts_before[FEE_ACCOUNT_ID]
                    .public_pool_info
                    .get_strategy_balance(builder, current_market_details.strategy_index),
            ],
        )
    }

    fn get_old_asset_hashes(
        &self,
        builder: &mut Builder,
    ) -> (
        [[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // Old asset hashes
        [[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // Old aggregated balance hashes
        [[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // Old asset delta hashes
    ) {
        (
            core::array::from_fn(|i| {
                core::array::from_fn(|j| self.account_assets_before[i][j].hash(builder))
            }),
            core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    self.accounts_before[i].aggregated_balance_hash(builder, j)
                })
            }),
            core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    self.accounts_delta_before[i].aggregated_asset_delta_hash(builder, j)
                })
            }),
        )
    }

    fn get_old_account_hashes(
        &self,
        builder: &mut Builder,
    ) -> (
        [[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; NB_ACCOUNTS_PER_TX - 1];
            NB_ACCOUNTS_PER_TX - 1], // Account position bucket hashes
        [[HashOutTarget; 3]; NB_ACCOUNTS_PER_TX], // Account hash, account pub data hash, account delta hash
        [BoolTarget; NB_ACCOUNTS_PER_TX],         // Account is empty
    ) {
        let position_bucket_hashes_for_account = [
            self.accounts_before[0].get_position_bucket_hashes(builder),
            self.accounts_before[1].get_position_bucket_hashes(builder),
        ];
        let old_hashes_and_is_empty_infos = [
            self.accounts_before[0].hash(builder, &position_bucket_hashes_for_account[0]),
            self.accounts_before[1].hash(builder, &position_bucket_hashes_for_account[1]),
            self.accounts_before[2].fee_account_hash(builder),
        ];

        let old_account_delta_hashes = [
            self.accounts_delta_before[0].hash(builder),
            self.accounts_delta_before[1].hash(builder),
            self.accounts_delta_before[2].fee_account_hash(builder),
        ];

        (
            core::array::from_fn(|i| {
                [
                    position_bucket_hashes_for_account[i][0],
                    position_bucket_hashes_for_account[i][1],
                ]
            }),
            core::array::from_fn(|i| {
                [
                    old_hashes_and_is_empty_infos[i].0, // account hash
                    old_hashes_and_is_empty_infos[i].1, // account pub data hash
                    old_account_delta_hashes[i],        // Delta hash
                ]
            }),
            core::array::from_fn(|i| old_hashes_and_is_empty_infos[i].2),
        )
    }

    /// If account is using unified trading mode, then usdc `balance` should be empty and `collateral` should be used,
    /// which is accounted for in `apply_account_delta`.
    fn apply_aggregated_balance_deltas(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        account_margined_assets_before: &[[AccountMarginedAssetTarget; NB_ASSETS_PER_TX];
             NB_ACCOUNTS_PER_TX],
    ) {
        for i in 0..NB_ACCOUNTS_PER_TX {
            let old_lit_unlock = self.accounts_before[i].get_total_unlock_amount(builder);
            let new_lit_unlock = tx_state.accounts[i].get_total_unlock_amount(builder);

            for j in 0..NB_ASSETS_PER_TX {
                let is_lit = builder.is_equal_constant(tx_state.asset_indices[j], LIT_ASSET_INDEX);
                let add_to_old_balance_for_pending_unlocks =
                    builder.mul_biguint_by_bool(&old_lit_unlock, is_lit);
                let add_to_new_balance_for_pending_unlocs =
                    builder.mul_biguint_by_bool(&new_lit_unlock, is_lit);

                let old_extended_spot_balance = builder.add_biguint_non_carry(
                    &self.account_assets_before[i][j].balance,
                    &add_to_old_balance_for_pending_unlocks,
                    BIG_U96_LIMBS,
                );
                let old_spot_balance = builder.div_biguint(
                    &old_extended_spot_balance,
                    &tx_state.assets[j].extension_multiplier,
                );
                let old_spot_balance_bigint = builder.biguint_to_bigint(&old_spot_balance);
                let old_margin_balance = builder.euclidian_div_by_biguint(
                    &account_margined_assets_before[i][j].balance,
                    &tx_state.assets[j].extension_multiplier,
                    BIG_U96_LIMBS,
                );
                let old_balance_bigint = builder.add_bigint_non_carry(
                    &old_spot_balance_bigint,
                    &old_margin_balance,
                    BIG_U96_LIMBS,
                );

                let new_extended_spot_balance = builder.add_biguint_non_carry(
                    &tx_state.account_assets[i][j].balance,
                    &add_to_new_balance_for_pending_unlocs,
                    BIG_U96_LIMBS,
                );
                let new_spot_balance = builder.div_biguint(
                    &new_extended_spot_balance,
                    &tx_state.assets[j].extension_multiplier,
                );
                let new_spot_balance_bigint = builder.biguint_to_bigint(&new_spot_balance);
                let new_margin_balance = builder.euclidian_div_by_biguint(
                    &tx_state.account_margined_assets[i][j].balance,
                    &tx_state.assets[j].extension_multiplier,
                    BIG_U96_LIMBS,
                );
                let new_balance_bigint = builder.add_bigint_non_carry(
                    &new_spot_balance_bigint,
                    &new_margin_balance,
                    BIG_U96_LIMBS,
                );

                let balance_delta = builder.sub_bigint_non_carry(
                    &new_balance_bigint,
                    &old_balance_bigint,
                    BIG_U96_LIMBS,
                );
                let balance_delta = builder
                    .mul_bigint_by_bool(&balance_delta, tx_state.is_sender_receiver_different);

                tx_state.accounts[i].aggregated_balances[j] = builder.add_bigint_non_carry(
                    &tx_state.accounts[i].aggregated_balances[j],
                    &balance_delta,
                    BIG_U96_LIMBS,
                );
                tx_state.accounts_delta[i].aggregated_asset_deltas[j] = builder
                    .add_bigint_non_carry(
                        &tx_state.accounts_delta[i].aggregated_asset_deltas[j],
                        &balance_delta,
                        BIG_U96_LIMBS,
                    );
            }
        }
    }

    fn apply_margin_balance_deltas(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        asset_margin_indices: &[Target; NB_ASSETS_PER_TX],
    ) {
        for acc in 0..NB_ACCOUNTS_PER_TX {
            for ass in 0..NB_ASSETS_PER_TX {
                for ma in 0..MARGINED_ASSET_LIST_SIZE {
                    let is_margin_index =
                        builder.is_equal_constant(asset_margin_indices[ass], ma as u64);
                    tx_state.accounts[acc].margined_assets[ma] =
                        select_account_margined_asset_target(
                            builder,
                            is_margin_index,
                            &tx_state.account_margined_assets[acc][ass],
                            &tx_state.accounts[acc].margined_assets[ma],
                        );
                }
            }
        }
    }

    fn apply_position_delta(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        positions_with_pub_data_before: &[PositionWithDelta; NB_ACCOUNTS_PER_TX - 1],
        position_bucket_hashes_for_account: &mut [[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; NB_ACCOUNTS_PER_TX - 1];
                 NB_ACCOUNTS_PER_TX - 1],
    ) -> [BigIntTarget; NB_ACCOUNTS_PER_TX - 1] {
        let position_hash_bucket_count = builder.constant_usize(POSITION_HASH_BUCKET_COUNT);
        let (current_bucket_index, index_in_bucket) = builder.div_rem(
            tx_state.market.perps_market_index,
            position_hash_bucket_count,
            5,
        );

        let empty_position = AccountPositionTarget::empty(builder);

        array::from_fn(|i| {
            let (position_with_pub_data, position_usdc_delta) =
                PositionWithDelta::new_position_with_pub_data_from_new_position(
                    builder,
                    &positions_with_pub_data_before[i],
                    &tx_state.positions[i],
                );
            tx_state.accounts_delta[i].positions_delta = position_with_pub_data.delta.clone();
            let position_diff = AccountPositionTarget::diff(
                builder,
                &position_with_pub_data.position,
                &positions_with_pub_data_before[i].position,
            );

            // Load the position bucket that corresponds to current market index
            let mut position_bucket: [AccountPositionTarget; POSITION_HASH_BUCKET_SIZE] =
                array::from_fn(|j| {
                    let candidates = (0..POSITION_HASH_BUCKET_COUNT)
                        .map(|bucket_index| {
                            let position_index = bucket_index * POSITION_HASH_BUCKET_SIZE + j;
                            if position_index < tx_state.accounts[i].positions.len() {
                                tx_state.accounts[i].positions[position_index].clone()
                            } else {
                                empty_position.clone()
                            }
                        })
                        .collect();
                    random_access_account_position(builder, current_bucket_index, candidates)
                });

            // Apply the difference to correct position for the current market
            for market_index_in_bucket in 0..POSITION_HASH_BUCKET_SIZE {
                let t_market_index_in_bucket = builder.constant_usize(market_index_in_bucket);
                let is_current_position =
                    builder.is_equal(t_market_index_in_bucket, index_in_bucket);
                position_bucket[market_index_in_bucket] = AccountPositionTarget::apply_diff(
                    builder,
                    is_current_position,
                    &position_bucket[market_index_in_bucket],
                    &position_diff,
                );
            }

            // Recalculate position bucket hash for account and pub data
            let new_position_bucket_hash =
                AccountTarget::get_position_bucket_hash(builder, &position_bucket);
            for bucket_index in 0..POSITION_HASH_BUCKET_COUNT {
                let t_bucket_index = builder.constant_usize(bucket_index);
                let is_current_position_bucket =
                    builder.is_equal(t_bucket_index, current_bucket_index);

                for j in 0..2 {
                    position_bucket_hashes_for_account[i][j][bucket_index] = builder.select_hash(
                        is_current_position_bucket,
                        &new_position_bucket_hash[j],
                        &position_bucket_hashes_for_account[i][j][bucket_index],
                    );
                }
            }

            position_usdc_delta
        })
    }

    fn apply_public_pool_share_delta(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        public_pool_share_before: &PublicPoolShareTarget,
    ) {
        let share_amount_delta = builder.sub(
            tx_state.public_pool_share.share_amount,
            public_pool_share_before.share_amount,
        );

        tx_state.accounts_delta[OWNER_ACCOUNT_ID].apply_pool_pub_data_share_delta(
            builder,
            tx_state.apply_pool_share_delta_flag,
            tx_state.public_pool_share.public_pool_index,
            SignedTarget::new_unsafe(share_amount_delta),
        );

        let principal_delta = builder.sub(
            tx_state.public_pool_share.principal_amount,
            public_pool_share_before.principal_amount,
        );
        let entry_timestamp_delta = builder.sub(
            tx_state.public_pool_share.entry_timestamp,
            public_pool_share_before.entry_timestamp,
        );
        tx_state.accounts[OWNER_ACCOUNT_ID].apply_pool_share_delta(
            builder,
            tx_state.apply_pool_share_delta_flag,
            tx_state.public_pool_share.public_pool_index,
            share_amount_delta,
            principal_delta,
            entry_timestamp_delta,
        );
    }

    fn apply_pool_info_deltas(&self, builder: &mut Builder, tx_state: &mut TxState) {
        for i in 0..NB_ACCOUNTS_PER_TX - 1 {
            let pool_total_shares_delta = SignedTarget::new_unsafe(builder.sub(
                tx_state.accounts[i].public_pool_info.total_shares,
                self.accounts_before[i].public_pool_info.total_shares,
            ));
            tx_state.accounts_delta[i]
                .public_pool_info_delta
                .total_shares_delta = builder.add_signed(
                tx_state.accounts_delta[i]
                    .public_pool_info_delta
                    .total_shares_delta,
                pool_total_shares_delta,
            );
            let pool_operator_shares_delta = SignedTarget::new_unsafe(builder.sub(
                tx_state.accounts[i].public_pool_info.operator_shares,
                self.accounts_before[i].public_pool_info.operator_shares,
            ));
            tx_state.accounts_delta[i]
                .public_pool_info_delta
                .operator_shares_delta = builder.add_signed(
                tx_state.accounts_delta[i]
                    .public_pool_info_delta
                    .operator_shares_delta,
                pool_operator_shares_delta,
            );
        }
    }

    fn apply_strategy_deltas(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        strategy_indexes: [Target; NB_ACCOUNTS_PER_TX],
    ) {
        for account in 0..NB_ACCOUNTS_PER_TX {
            for strategy in 0..NB_STRATEGIES {
                let correct_strategy =
                    builder.is_equal_constant(strategy_indexes[account], strategy as u64);
                let flag = builder.and_not(correct_strategy, tx_state.between_strategies_flag);
                tx_state.accounts[account].public_pool_info.strategies[strategy] = builder
                    .select_bigint(
                        flag,
                        &tx_state.strategies[account],
                        &tx_state.accounts[account].public_pool_info.strategies[strategy],
                    );
            }
        }
    }

    fn apply_account_deltas(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        position_usdc_deltas: &[BigIntTarget; NB_ACCOUNTS_PER_TX - 1],
    ) {
        for i in 0..NB_ACCOUNTS_PER_TX {
            let mut collateral_delta = builder.zero_bigint();

            // Account for position changes except for fee account
            if i < NB_ACCOUNTS_PER_TX - 1 {
                // If account is a new account, apply l1 address and account type changes to pub data delta
                let is_new_account = tx_state.is_new_account[i];
                tx_state.accounts_delta[i].l1_address = builder.select_biguint(
                    is_new_account,
                    &tx_state.accounts[i].l1_address,
                    &tx_state.accounts_delta[i].l1_address,
                );
                tx_state.accounts_delta[i].account_type = builder.select(
                    is_new_account,
                    tx_state.accounts[i].account_type,
                    tx_state.accounts_delta[i].account_type,
                );

                // Apply aggregated collateral delta for position change
                collateral_delta = builder.add_bigint_non_carry(
                    &collateral_delta,
                    &position_usdc_deltas[i],
                    BIG_U96_LIMBS,
                );

                // Don't apply collateral delta if sender and receiver are the same
                collateral_delta = builder
                    .mul_bigint_by_bool(&collateral_delta, tx_state.is_sender_receiver_different);
            }

            // Apply collateral delta to usdc asset balance
            let zero_bigint = builder.zero_bigint();
            let usdc_asset_index = builder.constant_u64(USDC_ASSET_INDEX);
            for j in TX_ASSET_ID..=FEE_ASSET_ID {
                let is_usdc_asset = builder.is_equal(self.asset_indices[j], usdc_asset_index);
                let usdc_delta =
                    builder.select_bigint(is_usdc_asset, &collateral_delta, &zero_bigint);
                tx_state.accounts[i].aggregated_balances[j] = builder.add_bigint_non_carry(
                    &tx_state.accounts[i].aggregated_balances[j],
                    &usdc_delta,
                    BIG_U96_LIMBS,
                );
                tx_state.accounts_delta[i].aggregated_asset_deltas[j] = builder
                    .add_bigint_non_carry(
                        &tx_state.accounts_delta[i].aggregated_asset_deltas[j],
                        &usdc_delta,
                        BIG_U96_LIMBS,
                    );
            }
        }
    }

    fn verify_position_delta_merkle_proofs(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        old_position_delta_hashes: &[HashOutTarget; NB_ACCOUNTS_PER_TX - 1],
    ) {
        let nil_market_index = builder.constant_usize(NIL_MARKET_INDEX as usize);
        let position_deltas_empty_check = [
            self.accounts_delta_before[0]
                .positions_delta
                .is_empty(builder),
            self.accounts_delta_before[1]
                .positions_delta
                .is_empty(builder),
            tx_state.accounts_delta[0].positions_delta.is_empty(builder),
            tx_state.accounts_delta[1].positions_delta.is_empty(builder),
        ];
        let is_position_deltas_empty = builder.multi_and(&position_deltas_empty_check);
        let delta_market_index = builder.select(
            is_position_deltas_empty,
            nil_market_index,
            tx_state.market.perps_market_index,
        );
        let merkle_path = perps_market_index_to_merkle_path(builder, delta_market_index);

        verify_merkle_proof(
            builder,
            &self.accounts_delta_before[TAKER_ACCOUNT_ID].position_delta_root,
            old_position_delta_hashes[TAKER_ACCOUNT_ID],
            self.position_delta_merkle_proofs[TAKER_ACCOUNT_ID],
            merkle_path,
        );
        let new_position_delta_hash = tx_state.accounts_delta[TAKER_ACCOUNT_ID]
            .positions_delta
            .hash(builder);
        tx_state.accounts_delta[TAKER_ACCOUNT_ID].position_delta_root = recalculate_root(
            builder,
            new_position_delta_hash,
            self.position_delta_merkle_proofs[TAKER_ACCOUNT_ID],
            merkle_path,
        );

        conditional_verify_merkle_proof(
            builder,
            tx_state.is_sender_receiver_different,
            &self.accounts_delta_before[MAKER_ACCOUNT_ID].position_delta_root,
            old_position_delta_hashes[MAKER_ACCOUNT_ID],
            self.position_delta_merkle_proofs[MAKER_ACCOUNT_ID],
            merkle_path,
        );
        let new_position_delta_hash = tx_state.accounts_delta[MAKER_ACCOUNT_ID]
            .positions_delta
            .hash(builder);
        let new_root = recalculate_root(
            builder,
            new_position_delta_hash,
            self.position_delta_merkle_proofs[MAKER_ACCOUNT_ID],
            merkle_path,
        );
        tx_state.accounts_delta[MAKER_ACCOUNT_ID].position_delta_root = builder.select_hash(
            tx_state.is_sender_receiver_different,
            &new_root,
            &self.accounts_delta_before[MAKER_ACCOUNT_ID].position_delta_root,
        );
    }

    fn verify_account_and_pub_data_merkle_proofs(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
        account_tree_root_before: HashOutTarget,
        account_pub_data_tree_root_before: HashOutTarget,
        account_delta_tree_root_before: HashOutTarget,
        old_account_hashes: &[[HashOutTarget; 3]; NB_ACCOUNTS_PER_TX], // 3 account trees per account
        position_bucket_hashes_for_account: &[[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2];
             NB_ACCOUNTS_PER_TX - 1],
    ) -> (HashOutTarget, HashOutTarget, HashOutTarget) {
        let is_sender_receiver_same = builder.not(tx_state.is_sender_receiver_different);
        let fee_account_is_taker_or_maker =
            builder.or(tx_state.fee_account_is_taker, tx_state.fee_account_is_maker);
        let is_fee_account_different = builder.not(fee_account_is_taker_or_maker);

        // Verify receiver account if it is the same as taker account, and fee account if it is the same as taker or maker account
        for i in 0..3 {
            builder.conditional_assert_eq_hash(
                is_sender_receiver_same,
                &old_account_hashes[TAKER_ACCOUNT_ID][i],
                &old_account_hashes[MAKER_ACCOUNT_ID][i],
            );
            builder.conditional_assert_eq_hash(
                tx_state.fee_account_is_taker,
                &old_account_hashes[TAKER_ACCOUNT_ID][i],
                &old_account_hashes[FEE_ACCOUNT_ID][i],
            );
            builder.conditional_assert_eq_hash(
                tx_state.fee_account_is_maker,
                &old_account_hashes[MAKER_ACCOUNT_ID][i],
                &old_account_hashes[FEE_ACCOUNT_ID][i],
            );
        }

        let conditions = [builder._true(), tx_state.is_sender_receiver_different];

        let mut current_account_tree_root = account_tree_root_before;
        let mut current_account_pub_data_tree_root = account_pub_data_tree_root_before;
        let mut current_account_delta_tree_root = account_delta_tree_root_before;

        for i in 0..NB_ACCOUNTS_PER_TX - 1 {
            let (new_account_root, new_account_pub_data_root, new_account_delta_root) = self
                .conditional_verify_single_account_merkle_proofs(
                    builder,
                    conditions[i],
                    tx_state,
                    i,
                    current_account_tree_root,
                    current_account_pub_data_tree_root,
                    current_account_delta_tree_root,
                    old_account_hashes[i][0], // account tree
                    old_account_hashes[i][1], // account pub data tree
                    old_account_hashes[i][2], // account delta tree
                    self.account_tree_merkle_proofs[i],
                    self.account_pub_data_tree_merkle_proofs[i],
                    self.account_delta_tree_merkle_proofs[i],
                    &position_bucket_hashes_for_account[i],
                );
            current_account_tree_root =
                builder.select_hash(conditions[i], &new_account_root, &current_account_tree_root);
            current_account_pub_data_tree_root = builder.select_hash(
                conditions[i],
                &new_account_pub_data_root,
                &current_account_pub_data_tree_root,
            );
            current_account_delta_tree_root = builder.select_hash(
                conditions[i],
                &new_account_delta_root,
                &current_account_delta_tree_root,
            );
        }

        let fee_account_merkle_path =
            account_index_to_merkle_path(builder, self.accounts_before[2].account_index);

        conditional_verify_merkle_proof(
            builder,
            is_fee_account_different,
            &current_account_tree_root,
            old_account_hashes[FEE_ACCOUNT_ID][0],
            self.account_tree_merkle_proofs[FEE_ACCOUNT_ID],
            fee_account_merkle_path,
        );
        conditional_verify_merkle_proof(
            builder,
            is_fee_account_different,
            &current_account_pub_data_tree_root,
            old_account_hashes[FEE_ACCOUNT_ID][1],
            self.account_pub_data_tree_merkle_proofs[FEE_ACCOUNT_ID],
            fee_account_merkle_path,
        );
        conditional_verify_merkle_proof(
            builder,
            is_fee_account_different,
            &current_account_delta_tree_root,
            old_account_hashes[FEE_ACCOUNT_ID][2],
            self.account_delta_tree_merkle_proofs[FEE_ACCOUNT_ID],
            fee_account_merkle_path,
        );

        // Update the account tree root if fee account is different from maker and taker accounts
        let (new_fee_account_hash, new_fee_account_pub_data_hash, _) =
            tx_state.accounts[FEE_ACCOUNT_ID].fee_account_hash(builder);
        let new_fee_account_delta_hash =
            tx_state.accounts_delta[FEE_ACCOUNT_ID].fee_account_hash(builder);

        let new_root = recalculate_root(
            builder,
            new_fee_account_hash,
            self.account_tree_merkle_proofs[FEE_ACCOUNT_ID],
            fee_account_merkle_path,
        );
        current_account_tree_root = builder.select_hash(
            is_fee_account_different,
            &new_root,
            &current_account_tree_root,
        );

        let new_root = recalculate_root(
            builder,
            new_fee_account_pub_data_hash,
            self.account_pub_data_tree_merkle_proofs[FEE_ACCOUNT_ID],
            fee_account_merkle_path,
        );
        current_account_pub_data_tree_root = builder.select_hash(
            is_fee_account_different,
            &new_root,
            &current_account_pub_data_tree_root,
        );

        let new_root = recalculate_root(
            builder,
            new_fee_account_delta_hash,
            self.account_delta_tree_merkle_proofs[FEE_ACCOUNT_ID],
            fee_account_merkle_path,
        );
        current_account_delta_tree_root = builder.select_hash(
            is_fee_account_different,
            &new_root,
            &current_account_delta_tree_root,
        );

        (
            current_account_tree_root,
            current_account_pub_data_tree_root,
            current_account_delta_tree_root,
        )

        // (
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        // )
    }

    #[track_caller]
    fn conditional_verify_single_account_merkle_proofs(
        &self,
        builder: &mut Builder,
        cond: BoolTarget,
        tx_state: &TxState,
        account_id: usize,
        current_account_tree_root: HashOutTarget,
        current_account_pub_data_tree_root: HashOutTarget,
        current_account_delta_tree_root: HashOutTarget,
        old_account_hash: HashOutTarget,
        old_account_pub_data_hash: HashOutTarget,
        old_account_delta_hash: HashOutTarget,
        account_tree_merkle_proof: [HashOutTarget; ACCOUNT_MERKLE_LEVELS],
        account_pub_data_tree_merkle_proof: [HashOutTarget; ACCOUNT_MERKLE_LEVELS],
        account_delta_tree_merkle_proof: [HashOutTarget; ACCOUNT_MERKLE_LEVELS],
        position_bucket_hashes_for_account: &[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2],
    ) -> (HashOutTarget, HashOutTarget, HashOutTarget) {
        let account_merkle_path =
            account_index_to_merkle_path(builder, self.accounts_before[account_id].account_index);

        conditional_verify_merkle_proof(
            builder,
            cond,
            &current_account_tree_root,
            old_account_hash,
            account_tree_merkle_proof,
            account_merkle_path,
        );
        conditional_verify_merkle_proof(
            builder,
            cond,
            &current_account_pub_data_tree_root,
            old_account_pub_data_hash,
            account_pub_data_tree_merkle_proof,
            account_merkle_path,
        );
        conditional_verify_merkle_proof(
            builder,
            cond,
            &current_account_delta_tree_root,
            old_account_delta_hash,
            account_delta_tree_merkle_proof,
            account_merkle_path,
        );

        let (new_account_hash, new_account_pub_data_hash, _) = tx_state.accounts[account_id].hash(
            builder,
            &[
                position_bucket_hashes_for_account[0],
                position_bucket_hashes_for_account[1],
            ],
        );
        let new_account_delta_hash = tx_state.accounts_delta[account_id].hash(builder);

        // builder.println_hash_out(&new_account_hash, "new_account_hash");
        // builder.println_hash_out(&new_account_pub_data_hash, "new_account_pub_data_hash");
        // builder.println_hash_out(&new_account_delta_hash, "new_account_delta_hash");

        (
            recalculate_root(
                builder,
                new_account_hash,
                account_tree_merkle_proof,
                account_merkle_path,
            ),
            recalculate_root(
                builder,
                new_account_pub_data_hash,
                account_pub_data_tree_merkle_proof,
                account_merkle_path,
            ),
            recalculate_root(
                builder,
                new_account_delta_hash,
                account_delta_tree_merkle_proof,
                account_merkle_path,
            ),
        )

        // (
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        //     builder.zero_hash_out(),
        // )
    }

    fn verify_api_key_merkle_proof(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        tx_type: &TxTypeTargets,
    ) {
        let api_key_before_hash = self.api_key_before.hash(builder);
        let api_key_merkle_path =
            api_key_index_to_merkle_path(builder, self.api_key_before.api_key_index);

        // Force burn swaps owner and pool accounts
        let api_key_root = builder.select_hash(
            tx_type.is_l2_force_burn_shares,
            &self.accounts_before[SUB_ACCOUNT_ID].api_key_root,
            &self.accounts_before[OWNER_ACCOUNT_ID].api_key_root,
        );

        verify_merkle_proof(
            builder,
            &api_key_root,
            api_key_before_hash,
            self.api_key_tree_merkle_proof,
            api_key_merkle_path,
        );

        let new_api_key_hash = tx_state.api_key.hash(builder);
        let new_api_key_root = recalculate_root(
            builder,
            new_api_key_hash,
            self.api_key_tree_merkle_proof,
            api_key_merkle_path,
        );

        tx_state.accounts[OWNER_ACCOUNT_ID].api_key_root = builder.select_hash(
            tx_type.is_l2_force_burn_shares,
            &tx_state.accounts[OWNER_ACCOUNT_ID].api_key_root,
            &new_api_key_root,
        );
        tx_state.accounts[SUB_ACCOUNT_ID].api_key_root = builder.select_hash(
            tx_type.is_l2_force_burn_shares,
            &new_api_key_root,
            &tx_state.accounts[SUB_ACCOUNT_ID].api_key_root,
        );
    }

    fn verify_assets_merkle_proofs(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        old_hashes: &(
            [[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // Old asset hashes
            [[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // Old aggregated balance hashes
            [[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // Old asset delta hashes
        ),
    ) {
        let is_sender_receiver_same = builder.not(tx_state.is_sender_receiver_different);
        let fee_account_is_taker_or_maker =
            builder.or(tx_state.fee_account_is_taker, tx_state.fee_account_is_maker);
        let is_fee_account_different = builder.not(fee_account_is_taker_or_maker);

        let old_hashes = [old_hashes.0, old_hashes.1, old_hashes.2];
        let new_hashes: [[[HashOutTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX]; 3] = [
            core::array::from_fn(|i| {
                core::array::from_fn(|j| tx_state.account_assets[i][j].hash(builder))
            }),
            core::array::from_fn(|i| {
                core::array::from_fn(|j| tx_state.accounts[i].aggregated_balance_hash(builder, j))
            }),
            core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    tx_state.accounts_delta[i].aggregated_asset_delta_hash(builder, j)
                })
            }),
        ];

        // builder.println_hash_out(&new_hashes[0][0][0], "0 0");
        // builder.println_hash_out(&new_hashes[1][0][0], "0 1");
        // builder.println_hash_out(&new_hashes[2][0][0], "0 2");

        // builder.println_hash_out(&new_hashes[0][0][1], "1 0");
        // builder.println_hash_out(&new_hashes[1][0][1], "1 1");
        // builder.println_hash_out(&new_hashes[2][0][1], "1 2");

        let mut roots: [[HashOutTarget; NB_ACCOUNTS_PER_TX]; 3] = [
            core::array::from_fn(|j| tx_state.accounts[j].asset_root),
            core::array::from_fn(|j| tx_state.accounts[j].aggregated_balances_root),
            core::array::from_fn(|j| tx_state.accounts_delta[j].asset_delta_root),
        ];
        let merkle_proofs = [
            &self.asset_tree_merkle_proofs,
            &self.public_asset_tree_merkle_proofs,
            &self.asset_delta_tree_merkle_proofs,
        ];
        let merkle_paths = [
            asset_index_to_merkle_path(builder, self.asset_indices[0]),
            asset_index_to_merkle_path(builder, self.asset_indices[1]),
        ];
        let conditions = [
            builder._true(),
            tx_state.is_sender_receiver_different,
            is_fee_account_different,
        ];

        for i in 0..3 {
            for j in 0..NB_ACCOUNTS_PER_TX {
                for k in 0..NB_ASSETS_PER_TX {
                    conditional_verify_merkle_proof(
                        builder,
                        conditions[j],
                        &roots[i][j],
                        old_hashes[i][j][k],
                        merkle_proofs[i][j][k],
                        merkle_paths[k],
                    );
                    let new_root = recalculate_root(
                        builder,
                        new_hashes[i][j][k],
                        merkle_proofs[i][j][k],
                        merkle_paths[k],
                    );
                    roots[i][j] = builder.select_hash(conditions[j], &new_root, &roots[i][j]);

                    if j < 2 {
                        builder.conditional_assert_eq_hash(
                            if j == TAKER_ACCOUNT_ID {
                                tx_state.fee_account_is_taker
                            } else {
                                tx_state.fee_account_is_maker
                            },
                            &old_hashes[i][j][k],
                            &old_hashes[i][FEE_ACCOUNT_ID][k],
                        );
                    }
                    if j == RECEIVER_ACCOUNT_ID {
                        builder.conditional_assert_eq_hash(
                            is_sender_receiver_same,
                            &old_hashes[i][j][k],
                            &old_hashes[i][SENDER_ACCOUNT_ID][k],
                        );
                    }
                }
            }
        }

        for i in 0..NB_ACCOUNTS_PER_TX {
            tx_state.accounts[i].asset_root = roots[0][i];
            tx_state.accounts[i].aggregated_balances_root = roots[1][i];
            tx_state.accounts_delta[i].asset_delta_root = roots[2][i];
        }
    }

    fn verify_account_orders_merkle_proof(&self, builder: &mut Builder, tx_state: &mut TxState) {
        let is_account_order_index_nil =
            builder.is_equal_constant(tx_state.account_order.index_0, NIL_ORDER_INDEX as u64);
        let maker_index_eq_owner_index = builder.is_equal(
            self.accounts_before[MAKER_ACCOUNT_ID].account_index,
            tx_state.account_order.owner_account_index,
        );
        let order_belongs_to_maker_account =
            builder.and_not(maker_index_eq_owner_index, is_account_order_index_nil);
        let use_maker_account = builder.and(
            order_belongs_to_maker_account,
            tx_state.is_sender_receiver_different,
        );

        // Verify that index_0 is either 0 or a valid order index
        let is_index_0_is_zero = builder.is_zero(self.account_order_before.index_0);
        let max_client_order_index = builder.constant_u64(MAX_CLIENT_ORDER_INDEX as u64);
        let is_index_0_valid = builder.is_gt(
            self.account_order_before.index_0,
            max_client_order_index,
            64,
        );
        let is_valid_index_0 = builder.or(is_index_0_is_zero, is_index_0_valid);
        builder.assert_true(is_valid_index_0);

        // Verify that index_1 is either 0 or a valid client order index. Because MAX_CLIENT_ORDER_INDEX = 1 << 48 - 1, single 48 bit range-check is enough.
        builder.register_range_check(self.account_order_before.index_1, CLIENT_ORDER_INDEX_BITS);

        // Set oid leaf and recalculate root
        let old_account_orders_root = builder.select_hash(
            use_maker_account,
            &self.accounts_before[MAKER_ACCOUNT_ID].account_orders_root,
            &self.accounts_before[TAKER_ACCOUNT_ID].account_orders_root,
        );
        let account_order_hash_for_index_0 = self.account_order_before.hash(builder);

        let account_orders_merkle_path_for_index_0 =
            account_order_index_to_merkle_path(builder, self.account_order_before.index_0);
        verify_merkle_proof(
            builder,
            &old_account_orders_root,
            account_order_hash_for_index_0,
            self.account_orders_tree_merkle_proof[TAKER_ACCOUNT_ID],
            account_orders_merkle_path_for_index_0,
        );
        let new_account_order_hash_for_index_0 = tx_state.account_order.hash(builder);

        let account_orders_root_after_oid_leaf_inserted = recalculate_root(
            builder,
            new_account_order_hash_for_index_0,
            self.account_orders_tree_merkle_proof[0],
            account_orders_merkle_path_for_index_0,
        );

        // Set cloid leaf and recalculate root
        let index_1_empty = builder.is_equal_constant(
            self.account_order_before.index_1,
            NIL_CLIENT_ORDER_INDEX as u64,
        );
        let empty_account_order_hash = builder.zero_hash_out();
        let account_order_hash_for_index_1 = builder.select_hash(
            index_1_empty,
            &empty_account_order_hash,
            &account_order_hash_for_index_0,
        );
        let account_orders_merkle_path_for_index_1 =
            account_order_index_to_merkle_path(builder, self.account_order_before.index_1);
        verify_merkle_proof(
            builder,
            &account_orders_root_after_oid_leaf_inserted,
            account_order_hash_for_index_1,
            self.account_orders_tree_merkle_proof[MAKER_ACCOUNT_ID],
            account_orders_merkle_path_for_index_1,
        );

        let new_account_order_hash_for_index_1 = builder.select_hash(
            index_1_empty,
            &empty_account_order_hash,
            &new_account_order_hash_for_index_0,
        );
        let new_account_orders_root = recalculate_root(
            builder,
            new_account_order_hash_for_index_1,
            self.account_orders_tree_merkle_proof[MAKER_ACCOUNT_ID],
            account_orders_merkle_path_for_index_1,
        );

        // Set the new account orders root in the account metadata

        tx_state.accounts[TAKER_ACCOUNT_ID].account_orders_root = builder.select_hash(
            use_maker_account,
            &self.accounts_before[TAKER_ACCOUNT_ID].account_orders_root,
            &new_account_orders_root,
        );
        tx_state.accounts[MAKER_ACCOUNT_ID].account_orders_root = builder.select_hash(
            use_maker_account,
            &new_account_orders_root,
            &self.accounts_before[MAKER_ACCOUNT_ID].account_orders_root,
        );
    }

    fn verify_market_and_order_book_proofs(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        market_tree_root_before: HashOutTarget,
    ) -> HashOutTarget {
        // Verify order leaf against order book tree
        let order_hash = self.order_before.hash(builder);
        verify_order_book_tree_merkle_proof(
            builder,
            &self.market_before.order_book_root,
            order_hash,
            &self.order_book_tree_path,
            tx_state.order_path_helper,
        );

        let market_hash = self.market_before.hash(builder);
        let market_index_merkle_path =
            market_index_to_merkle_path(builder, self.market_before.market_index);
        verify_merkle_proof(
            builder,
            &market_tree_root_before,
            market_hash,
            self.market_tree_merkle_proof,
            market_index_merkle_path,
        );

        let order_book_tree_path_after = get_order_book_path_delta(
            builder,
            &self.order_before,
            &self.order_book_tree_path,
            &tx_state.order,
        );
        let new_order_hash = tx_state.order.hash(builder);
        tx_state.market.order_book_root = recalculate_order_book_tree_root(
            builder,
            new_order_hash,
            &order_book_tree_path_after,
            tx_state.order_path_helper,
        );

        // Only verify impact orders for perps
        let empty_order_book_tree_root = builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT);
        let is_perps = builder.is_equal_constant(tx_state.market.market_type, MARKET_TYPE_PERPS);
        let ob_root_for_impact_orders = builder.select_hash(
            is_perps,
            &tx_state.market.order_book_root,
            &empty_order_book_tree_root,
        );

        // Verify impact ask order leaf against order book tree
        let impact_ask_order_hash = self.impact_ask_order.hash(builder);
        let impact_ask_order_path_helper = order_indexes_to_merkle_path(
            builder,
            self.impact_ask_order.price_index,
            self.impact_ask_order.nonce_index,
        );
        verify_order_book_tree_merkle_proof(
            builder,
            &ob_root_for_impact_orders,
            impact_ask_order_hash,
            &self.impact_ask_order_book_tree_path,
            impact_ask_order_path_helper,
        );

        // Verify impact bid order leaf against order book tree
        let impact_bid_order_hash = self.impact_bid_order.hash(builder);
        let impact_bid_order_path_helper = order_indexes_to_merkle_path(
            builder,
            self.impact_bid_order.price_index,
            self.impact_bid_order.nonce_index,
        );
        verify_order_book_tree_merkle_proof(
            builder,
            &ob_root_for_impact_orders,
            impact_bid_order_hash,
            &self.impact_bid_order_book_tree_path,
            impact_bid_order_path_helper,
        );

        let new_market_hash = tx_state.market.hash(builder);
        recalculate_root(
            builder,
            new_market_hash,
            self.market_tree_merkle_proof,
            market_index_merkle_path,
        )
    }

    /// Selects parts of the main account for tx type related verifications.
    /// Any new field verification added to `verify_l2_tx` requires a change here as well.
    fn select_partial_main_account(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
    ) -> AccountTarget {
        AccountTarget {
            account_index: builder.select(
                tx_type.is_l2_force_burn_shares,
                self.accounts_before[SUB_ACCOUNT_ID].account_index,
                self.accounts_before[OWNER_ACCOUNT_ID].account_index,
            ),
            account_type: builder.select(
                tx_type.is_l2_force_burn_shares,
                self.accounts_before[SUB_ACCOUNT_ID].account_type,
                self.accounts_before[OWNER_ACCOUNT_ID].account_type,
            ),
            cancel_all_time: builder.select(
                tx_type.is_l2_force_burn_shares,
                self.accounts_before[SUB_ACCOUNT_ID].cancel_all_time,
                self.accounts_before[OWNER_ACCOUNT_ID].cancel_all_time,
            ),
            public_pool_info: PublicPoolInfoTarget {
                status: builder.select(
                    tx_type.is_l2_force_burn_shares,
                    self.accounts_before[SUB_ACCOUNT_ID].public_pool_info.status,
                    self.accounts_before[OWNER_ACCOUNT_ID]
                        .public_pool_info
                        .status,
                ),
                ..PublicPoolInfoTarget::default()
            },
            ..AccountTarget::default()
        }
    }

    /// Selects the public key for the account. For L2 Change Pub Key transaction, the new public key in transaction is used.
    fn select_account_pk(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
    ) -> QuinticExtensionTarget {
        builder.select_quintic_ext(
            tx_type.is_l2_change_pub_key,
            self.l2_change_pub_key_tx_target.inner.pub_key,
            self.api_key_before.public_key,
        )
    }

    /// Compute and select tx_hash for L2 transactions
    fn select_tx_hash(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut selected_hash = builder.zero_quintic_ext();

        let l2_change_pub_key_hash =
            self.l2_change_pub_key_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_change_pub_key,
            l2_change_pub_key_hash,
            selected_hash,
        );

        let l2_create_sub_account_tx_hash = self.l2_create_sub_account_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_create_sub_account,
            l2_create_sub_account_tx_hash,
            selected_hash,
        );

        let l2_create_public_pool_tx_hash = self.l2_create_public_pool_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_create_public_pool,
            l2_create_public_pool_tx_hash,
            selected_hash,
        );

        let l2_update_public_pool_tx_hash = self.l2_update_public_pool_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_public_pool,
            l2_update_public_pool_tx_hash,
            selected_hash,
        );

        let l2_transfer_tx_hash =
            self.l2_transfer_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash =
            builder.select_quintic_ext(tx_type.is_l2_transfer, l2_transfer_tx_hash, selected_hash);

        let l2_withdraw_tx_hash =
            self.l2_withdraw_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash =
            builder.select_quintic_ext(tx_type.is_l2_withdraw, l2_withdraw_tx_hash, selected_hash);

        let l2_create_order_tx_hash =
            self.l2_create_order_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_create_order,
            l2_create_order_tx_hash,
            selected_hash,
        );

        let l2_cancel_order_tx_hash =
            self.l2_cancel_order_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_cancel_order,
            l2_cancel_order_tx_hash,
            selected_hash,
        );

        let l2_cancel_all_orders_tx_hash = self.l2_cancel_all_orders_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_cancel_all_orders,
            l2_cancel_all_orders_tx_hash,
            selected_hash,
        );

        let l2_modify_order_tx_hash =
            self.l2_modify_order_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_modify_order,
            l2_modify_order_tx_hash,
            selected_hash,
        );

        let l2_mint_shares_tx_hash =
            self.l2_mint_shares_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_mint_shares,
            l2_mint_shares_tx_hash,
            selected_hash,
        );

        let l2_burn_shares_tx_hash =
            self.l2_burn_shares_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_burn_shares,
            l2_burn_shares_tx_hash,
            selected_hash,
        );

        let l2_update_leverage_tx_hash =
            self.l2_update_leverage_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_leverage,
            l2_update_leverage_tx_hash,
            selected_hash,
        );

        let l2_create_grouped_orders_tx_hash = self.l2_create_grouped_orders_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_create_grouped_orders,
            l2_create_grouped_orders_tx_hash,
            selected_hash,
        );

        let l2_update_margin_tx_hash =
            self.l2_update_margin_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_margin,
            l2_update_margin_tx_hash,
            selected_hash,
        );

        let l2_create_staking_pool_tx_hash = self.l2_create_staking_pool_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_create_staking_pool,
            l2_create_staking_pool_tx_hash,
            selected_hash,
        );

        let l2_stake_assets_tx_hash =
            self.l2_stake_assets_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_stake_assets,
            l2_stake_assets_tx_hash,
            selected_hash,
        );

        let l2_unstake_assets_tx_hash =
            self.l2_unstake_assets_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_unstake_assets,
            l2_unstake_assets_tx_hash,
            selected_hash,
        );

        let l2_force_burn_shares_tx_hash = self.l2_force_burn_shares_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_force_burn_shares,
            l2_force_burn_shares_tx_hash,
            selected_hash,
        );

        let l2_update_account_config_tx_hash = self.l2_update_account_config_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_account_config,
            l2_update_account_config_tx_hash,
            selected_hash,
        );

        let l2_strategy_transfer_tx_hash = self.l2_strategy_transfer_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_strategy_transfer,
            l2_strategy_transfer_tx_hash,
            selected_hash,
        );

        let l2_update_market_config_tx_hash = self.l2_update_market_config_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_market_config,
            l2_update_market_config_tx_hash,
            selected_hash,
        );

        let l2_approve_integrator_tx_hash = self.l2_approve_integrator_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_approve_integrator,
            l2_approve_integrator_tx_hash,
            selected_hash,
        );

        let l2_update_account_asset_config_tx_hash = self
            .l2_update_account_asset_config_tx_target
            .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_account_asset_config,
            l2_update_account_asset_config_tx_hash,
            selected_hash,
        );

        let l2_update_asset_config_tx_hash = self.l2_update_asset_config_tx_target.hash(
            builder,
            self.nonce,
            self.expired_at,
            chain_id,
        );
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_update_asset_config,
            l2_update_asset_config_tx_hash,
            selected_hash,
        );

        self.attributes.aggregate_tx_hash(builder, selected_hash)
    }

    fn get_is_cloid_unique_infos(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
    ) -> [BoolTarget; NB_CLOID_UNIQUENESS_CHECK_PER_TX] {
        let nil_cloid = builder.constant_i64(NIL_CLIENT_ORDER_INDEX);
        let cloids = [
            {
                let select_1 = builder.select(
                    tx_type.is_l2_create_order,
                    self.l2_create_order_tx_target.inner.client_order_index,
                    nil_cloid,
                );
                builder.select(
                    tx_type.is_l2_create_grouped_orders,
                    self.l2_create_grouped_orders_tx_target.inner.orders[0].client_order_index,
                    select_1,
                )
            },
            builder.select(
                tx_type.is_l2_create_grouped_orders,
                self.l2_create_grouped_orders_tx_target.inner.orders[1].client_order_index,
                nil_cloid,
            ),
            builder.select(
                tx_type.is_l2_create_grouped_orders,
                self.l2_create_grouped_orders_tx_target.inner.orders[2].client_order_index,
                nil_cloid,
            ),
        ];
        let empty_hash = builder.zero_hash_out();
        core::array::from_fn(|i| {
            let path = account_client_order_index_to_merkle_path(builder, cloids[i]);
            try_verify_merkle_proof(
                builder,
                &self.accounts_before[0].account_orders_root,
                empty_hash,
                self.account_orders_tree_merkle_proof[2 + i],
                path,
            )
        })
    }

    /// Verifies asset indices
    fn validate_asset_indices(&self, builder: &mut Builder, tx_state: &TxState) {
        let is_nil_asset =
            builder.is_equal_constant(self.asset_indices[TX_ASSET_ID], NIL_ASSET_INDEX);
        let is_different_asset_index = builder.is_not_equal(
            self.asset_indices[TX_ASSET_ID],
            self.asset_indices[FEE_ASSET_ID],
        );
        let asset_index_assertion = builder.or(is_nil_asset, is_different_asset_index);
        builder.assert_true(asset_index_assertion);

        builder.register_range_check(self.asset_indices[TX_ASSET_ID], ASSET_LIST_SIZE_BITS);
        builder.register_range_check(self.asset_indices[FEE_ASSET_ID], ASSET_LIST_SIZE_BITS);

        let fee_account_is_taker_or_maker =
            builder.or(tx_state.fee_account_is_taker, tx_state.fee_account_is_maker);
        let different_fee_account = builder.not(fee_account_is_taker_or_maker);
        for i in [TX_ASSET_ID, FEE_ASSET_ID] {
            for j in [SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID] {
                builder.connect(
                    self.asset_indices[i],
                    self.account_assets_before[j][i].index_0,
                );
            }
            builder.conditional_assert_eq(
                different_fee_account,
                self.asset_indices[i],
                self.account_assets_before[FEE_ACCOUNT_ID][i].index_0,
            )
        }
    }

    /// Verifies each transaction according to the transaction type
    fn verify_transactions(
        &mut self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
        tx_state: &TxState,
    ) {
        /***********************/
        /*   L1 Transactions   */
        /***********************/
        self.l1_deposit_tx_target.verify(builder, tx_type, tx_state);
        self.l1_change_pub_key_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_create_market_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_update_market_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_cancel_all_orders_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_withdraw_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_create_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_burn_shares_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_register_asset_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_update_asset_tx_target
            .verify(builder, tx_type, tx_state);
        self.l1_set_system_config_tx_target
            .verify(builder, tx_type, tx_state);

        /***********************/
        /*   L2 Transactions   */
        /***********************/
        self.l2_change_pub_key_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_create_sub_account_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_create_public_pool_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_public_pool_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_transfer_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_withdraw_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_create_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_cancel_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_cancel_all_orders_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_modify_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_mint_shares_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_burn_shares_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_leverage_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_create_grouped_orders_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_margin_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_create_staking_pool_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_stake_assets_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_unstake_assets_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_force_burn_shares_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_account_config_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_strategy_transfer_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_market_config_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_approve_integrator_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_account_asset_config_tx_target
            .verify(builder, tx_type, tx_state);
        self.l2_update_asset_config_tx_target
            .verify(builder, tx_type, tx_state);

        /*************************/
        /* Internal Transactions */
        /*************************/
        self.internal_claim_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_cancel_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_deleverage_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_exit_position_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_cancel_all_orders_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_liquidate_position_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_create_order_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_pending_unlock_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_transfer_tx_target
            .verify(builder, tx_type, tx_state);
        self.internal_liquidate_spot_tx_target
            .verify(builder, tx_type, tx_state);
    }

    fn select_priority_operations_pub_data(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        let mut result = [builder.zero_u8(); MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX];

        let (exists, l1_withdraw) = self
            .l1_withdraw_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_withdraw, &result);

        let (exists, l1_change_pubkey) = self
            .l1_change_pub_key_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_change_pubkey, &result);

        let (exists, l1_burn_shares) = self
            .l1_burn_shares_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_burn_shares, &result);

        let (exists, l1_create_market) = self
            .l1_create_market_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_create_market, &result);

        let (exists, l1_create_order) = self
            .l1_create_order_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_create_order, &result);

        let (exists, l1_deposit) = self
            .l1_deposit_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_deposit, &result);

        let (exists, l1_update_market) = self
            .l1_update_market_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_update_market, &result);

        let (exists, l1_cancel_all_orders) = self
            .l1_cancel_all_orders_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_cancel_all_orders, &result);

        let (exists, l1_register_asset) = self
            .l1_register_asset_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_register_asset, &result);

        let (exists, l1_update_asset) = self
            .l1_update_asset_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_update_asset, &result);

        let (exists, l1_set_system_config) = self
            .l1_set_system_config_tx_target
            .priority_operations_pub_data(builder);
        result = builder.select_arr_u8(exists, &l1_set_system_config, &result);

        // Instead of taking ORs of `exists` values, we can use `tx_type.is_layer1` because every l1 tx has priority operation pub data
        (tx_type.is_layer1, result)
    }

    fn select_on_chain_operations_pub_data(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
    ) -> (
        BoolTarget,
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
    ) {
        let mut on_chain_pub_data = [builder.zero_u8(); ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE];
        let mut on_chain_pub_data_exists = builder._false();

        /***********************/
        /*   L1 Transactions   */
        /***********************/
        let (exists, l1_withdraw_on_chain_pub_data) = self
            .l1_withdraw_tx_target
            .on_chain_pub_data(builder, tx_state);
        on_chain_pub_data =
            builder.select_arr_u8(exists, &l1_withdraw_on_chain_pub_data, &on_chain_pub_data);
        on_chain_pub_data_exists = builder.or(exists, on_chain_pub_data_exists);

        let (exists, l1_deposit_on_chain_pub_data) = self
            .l1_deposit_tx_target
            .on_chain_pub_data(builder, tx_state);
        on_chain_pub_data =
            builder.select_arr_u8(exists, &l1_deposit_on_chain_pub_data, &on_chain_pub_data);
        on_chain_pub_data_exists = builder.or(exists, on_chain_pub_data_exists);

        /***********************/
        /*   L2 Transactions   */
        /***********************/
        let (exists, l2_withdraw_on_chain_pub_data) = self
            .l2_withdraw_tx_target
            .on_chain_pub_data(builder, tx_state);
        on_chain_pub_data =
            builder.select_arr_u8(exists, &l2_withdraw_on_chain_pub_data, &on_chain_pub_data);
        on_chain_pub_data_exists = builder.or(exists, on_chain_pub_data_exists);

        /*************************/
        /* Internal Transactions */
        /*************************/
        // no internal transactions have on-chain pub data

        (on_chain_pub_data_exists, on_chain_pub_data)
    }

    fn apply_transaction(
        &mut self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        tx_type: &TxTypeTargets,
    ) {
        /***********************/
        /*   L1 Transactions   */
        /***********************/
        self.l1_deposit_tx_target.apply(builder, tx_state);
        self.l1_create_market_tx_target.apply(builder, tx_state);
        self.l1_update_market_tx_target.apply(builder, tx_state);
        self.l1_cancel_all_orders_tx_target.apply(builder, tx_state);
        self.l1_withdraw_tx_target.apply(builder, tx_state);
        self.l1_change_pub_key_tx_target.apply(builder, tx_state);
        self.l1_create_order_tx_target.apply(builder, tx_state);
        self.l1_burn_shares_tx_target.apply(builder, tx_state);
        self.l1_register_asset_tx_target.apply(builder, tx_state);
        self.l1_update_asset_tx_target.apply(builder, tx_state);
        self.l1_set_system_config_tx_target.apply(builder, tx_state);

        /***********************/
        /*   L2 Transactions   */
        /***********************/
        self.l2_change_pub_key_tx_target.apply(builder, tx_state);
        self.l2_create_sub_account_tx_target
            .apply(builder, tx_state);
        self.l2_create_public_pool_tx_target
            .apply(builder, tx_state);
        self.l2_update_public_pool_tx_target
            .apply(builder, tx_state);
        self.l2_transfer_tx_target.apply(builder, tx_state);
        self.l2_create_order_tx_target.apply(builder, tx_state);
        self.l2_withdraw_tx_target.apply(builder, tx_state);
        self.l2_cancel_order_tx_target.apply(builder, tx_state);
        self.l2_cancel_all_orders_tx_target.apply(builder, tx_state);
        self.l2_modify_order_tx_target.apply(builder, tx_state);
        self.l2_mint_shares_tx_target.apply(builder, tx_state);
        self.l2_burn_shares_tx_target.apply(builder, tx_state);
        self.l2_update_leverage_tx_target.apply(builder, tx_state);
        self.l2_create_grouped_orders_tx_target
            .apply(builder, tx_state);
        self.l2_update_margin_tx_target.apply(builder, tx_state);
        self.l2_create_staking_pool_tx_target
            .apply(builder, tx_state);
        self.l2_stake_assets_tx_target.apply(builder, tx_state);
        self.l2_unstake_assets_tx_target.apply(builder, tx_state);
        self.l2_force_burn_shares_tx_target.apply(builder, tx_state);
        self.l2_update_account_config_tx_target
            .apply(builder, tx_state);
        self.l2_strategy_transfer_tx_target.apply(builder, tx_state);
        self.l2_update_market_config_tx_target
            .apply(builder, tx_state);
        self.l2_approve_integrator_tx_target
            .apply(builder, tx_state);
        self.l2_update_account_asset_config_tx_target
            .apply(builder, tx_state);
        self.l2_update_asset_config_tx_target
            .apply(builder, tx_state);

        /*************************/
        /* Internal Transactions */
        /*************************/
        self.internal_exit_position_tx_target
            .apply(builder, tx_state);
        self.internal_claim_order_tx_target.apply(builder, tx_state);
        self.internal_cancel_order_tx_target
            .apply(builder, tx_state);
        self.internal_deleverage_tx_target.apply(builder, tx_state);
        self.internal_cancel_all_orders_tx_target
            .apply(builder, tx_state);
        self.internal_liquidate_position_tx_target
            .apply(builder, tx_state);
        self.internal_create_order_tx_target
            .apply(builder, tx_state);
        self.internal_pending_unlock_tx_target
            .apply(builder, tx_state);
        self.internal_transfer_tx_target.apply(builder, tx_state);
        self.internal_liquidate_spot_tx_target
            .apply(builder, tx_state);

        // Increase ApiKey Nonce for all Layer2 transactions
        let next_nonce = builder.add_one(self.nonce);
        tx_state.api_key.nonce =
            builder.select(tx_type.is_layer2, next_nonce, tx_state.api_key.nonce);
    }

    fn get_market_details_with_random_access(
        &self,
        builder: &mut Builder,
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> MarketDetailsTarget {
        let mut as_vec = all_market_details.to_vec();
        // Pad the vector to a multiple of 64
        as_vec.push(MarketDetailsTarget::empty(builder));
        assert!(as_vec.len() % 64 == 0);

        let access_index = self.market_before.perps_market_index;
        builder.register_range_check(access_index, POSITION_LIST_SIZE_BITS);

        let zero = builder.zero();
        let mut result = MarketDetailsTarget::empty(builder);
        let mut is_result_set = builder._false();
        for i in 0..(as_vec.len() / 64) {
            let start_index = builder.constant_i64((i as i64) * 64);
            let end_index = builder.constant_i64(((i + 1) as i64) * 64 - 1);
            let chunk_access_index = builder.sub(access_index, start_index);
            let contains = builder.is_lte(access_index, end_index, POSITION_LIST_SIZE_BITS);
            let contains = builder.and_not(contains, is_result_set);
            let chunk_access_index = builder.select(contains, chunk_access_index, zero);
            let result_check = random_access_market_details(
                builder,
                chunk_access_index,
                as_vec
                    .iter()
                    .skip(i * 64)
                    .take(64)
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            result = select_market_details(builder, contains, &result_check, &result);
            is_result_set = builder.or(contains, is_result_set);
        }

        result
    }

    fn update_assets(
        &self,
        builder: &mut Builder,
        all_assets: &[AssetTarget; ASSET_LIST_SIZE],
        old_assets: &[AssetTarget; NB_ASSETS_PER_TX],
        new_assets: &[AssetTarget; NB_ASSETS_PER_TX],
        all_margined_assets: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
        margin_indices: &[Target; NB_ASSETS_PER_TX],
        old_margined_assets: &[MarginedAssetTarget; NB_ASSETS_PER_TX],
        new_margined_assets: &[MarginedAssetTarget; NB_ASSETS_PER_TX],
    ) -> (
        [AssetTarget; ASSET_LIST_SIZE],
        [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    ) {
        let (diff0, diff1) = (
            diff_assets(builder, &new_assets[0], &old_assets[0]),
            diff_assets(builder, &new_assets[1], &old_assets[1]),
        );
        let new_all_assets: [AssetTarget; ASSET_LIST_SIZE] = (0..ASSET_LIST_SIZE as u64)
            .map(|asset_index| {
                let mut asset = all_assets[asset_index as usize].clone();
                if !(MIN_ASSET_INDEX..=MAX_ASSET_INDEX).contains(&asset_index) {
                    return asset;
                }

                let asset_index_target = builder.constant_u64(asset_index);
                for i in 0..NB_ASSETS_PER_TX {
                    let is_current_asset =
                        builder.is_equal(asset_index_target, self.asset_indices[i]);
                    let diff = if i == 0 { &diff0 } else { &diff1 };

                    asset = apply_diff_assets(builder, is_current_asset, diff, &asset);
                }

                asset
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let diff_margined_asset_0 =
            diff_margined_assets(builder, &new_margined_assets[0], &old_margined_assets[0]);
        let diff_margined_asset_1 =
            diff_margined_assets(builder, &new_margined_assets[1], &old_margined_assets[1]);

        let new_all_margined_assets: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE] = (0
            ..MARGINED_ASSET_LIST_SIZE as u64)
            .map(|asset_index| {
                let mut margined_asset = all_margined_assets[asset_index as usize].clone();

                let margin_index_target = builder.constant_u64(asset_index);

                for i in 0..NB_ASSETS_PER_TX {
                    let is_current_asset = builder.is_equal(margin_index_target, margin_indices[i]);
                    let diff = if i == 0 {
                        &diff_margined_asset_0
                    } else {
                        &diff_margined_asset_1
                    };

                    margined_asset = apply_diff_margined_assets(
                        builder,
                        is_current_asset,
                        diff,
                        &margined_asset,
                    );
                }

                margined_asset
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        (new_all_assets, new_all_margined_assets)
    }

    fn update_market_details(
        &self,
        builder: &mut Builder,
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
        old_market_detail: &MarketDetailsTarget,
        new_market_detail: &MarketDetailsTarget,
    ) -> [MarketDetailsTarget; POSITION_LIST_SIZE] {
        let current_market_index = self.market_before.perps_market_index;
        let diff = diff_market_details(builder, new_market_detail, old_market_detail);
        (0..POSITION_LIST_SIZE)
            .map(|market_index| {
                let market_index_target = builder.constant(F::from_canonical_usize(market_index));
                let is_current_market = builder.is_equal(current_market_index, market_index_target);
                apply_diff_market_details(
                    builder,
                    is_current_market,
                    &diff,
                    &all_market_details[market_index],
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn update_impact_prices(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        market_details_before: &MarketDetailsTarget,
    ) {
        let active_status = builder.constant_u64(MARKET_STATUS_ACTIVE as u64);
        let is_market_active = builder.is_equal(tx_state.market_details.status, active_status);
        let perps_type = builder.constant_u64(MARKET_TYPE_PERPS);
        let is_perps_market = builder.is_equal(tx_state.market.market_type, perps_type);
        let should_update_impact_prices = builder.multi_and(&[
            tx_state.update_impact_prices_flag,
            is_market_active,
            is_perps_market,
        ]);

        let (impact_ask_price, impact_bid_price) = get_impact_prices(
            builder,
            should_update_impact_prices,
            &self.impact_ask_order_book_tree_path,
            &self.impact_ask_order,
            &self.impact_bid_order_book_tree_path,
            &self.impact_bid_order,
            tx_state.market_details.min_initial_margin_fraction,
            market_details_before.quote_multiplier,
        );

        tx_state.market_details.impact_ask_price = builder.select(
            should_update_impact_prices,
            impact_ask_price,
            tx_state.market_details.impact_ask_price,
        );
        tx_state.market_details.impact_bid_price = builder.select(
            should_update_impact_prices,
            impact_bid_price,
            tx_state.market_details.impact_bid_price,
        );

        let impact_prices_sum = builder.add(impact_ask_price, impact_bid_price);
        let two = builder.two();
        let (avg_impact_price, _) = builder.div_rem(impact_prices_sum, two, 2);

        let is_impact_ask_price_non_zero = builder.is_not_zero(impact_ask_price);
        let is_impact_bid_price_non_zero = builder.is_not_zero(impact_bid_price);
        let are_impact_prices_non_zero =
            builder.and(is_impact_ask_price_non_zero, is_impact_bid_price_non_zero);

        let should_update_impact_prices_and_non_zero =
            builder.and(should_update_impact_prices, are_impact_prices_non_zero);
        tx_state.market_details.impact_price = builder.select(
            should_update_impact_prices_and_non_zero,
            avg_impact_price,
            tx_state.market_details.impact_price,
        );

        let should_update_impact_prices_and_zero =
            builder.and_not(should_update_impact_prices, are_impact_prices_non_zero);

        let zero = builder.zero();
        tx_state.market_details.impact_price = builder.select(
            should_update_impact_prices_and_zero,
            zero,
            tx_state.market_details.impact_price,
        );
    }
}

pub trait TxTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_tx_target(&mut self, a: &TxTarget, b: &Tx<F>) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    TxTargetWitness<F> for T
{
    fn set_tx_target(&mut self, a: &TxTarget, b: &Tx<F>) -> Result<()> {
        self.set_target(a.tx_type, F::from_canonical_u8(b.tx_type))?;

        /***********************/
        /*   L1 Transactions   */
        /***********************/
        self.set_l1_deposit_tx_target(&a.l1_deposit_tx_target.inner, &b.l1_deposit_tx)?;
        self.set_l1_create_market_tx_target(
            &a.l1_create_market_tx_target.inner,
            &b.l1_create_market_tx,
        )?;
        self.set_l1_update_market_tx_target(
            &a.l1_update_market_tx_target.inner,
            &b.l1_update_market_tx,
        )?;
        self.set_l1_cancel_all_orders_tx_target(
            &a.l1_cancel_all_orders_tx_target.inner,
            &b.l1_cancel_all_orders_tx,
        )?;
        self.set_l1_withdraw_tx_target(&a.l1_withdraw_tx_target.inner, &b.l1_withdraw_tx)?;
        self.set_l1_create_order_tx_target(
            &a.l1_create_order_tx_target.inner,
            &b.l1_create_order_tx,
        )?;
        self.set_l1_change_pub_key_tx_target(
            &a.l1_change_pub_key_tx_target.inner,
            &b.l1_change_pub_key_tx,
        )?;
        self.set_l1_burn_shares_tx_target(&a.l1_burn_shares_tx_target.inner, &b.l1_burn_shares_tx)?;
        self.set_l1_register_asset_tx_target(
            &a.l1_register_asset_tx_target.inner,
            &b.l1_register_asset_tx,
        )?;
        self.set_l1_update_asset_tx_target(
            &a.l1_update_asset_tx_target.inner,
            &b.l1_update_asset_tx,
        )?;

        self.set_l1_set_system_config_tx_target(
            &a.l1_set_system_config_tx_target.inner,
            &b.l1_set_system_config_tx,
        )?;

        /***********************/
        /*   L2 Transactions   */
        /***********************/
        self.set_l2_change_pk_tx_target(
            &a.l2_change_pub_key_tx_target.inner,
            &b.l2_change_pub_key_tx,
        )?;
        self.set_l2_create_sub_account_tx_target(
            &a.l2_create_sub_account_tx_target.inner,
            &b.l2_create_sub_account_tx,
        )?;
        self.set_l2_create_public_pool_tx_target(
            &a.l2_create_public_pool_tx_target.inner,
            &b.l2_create_public_pool_tx,
        )?;
        self.set_l2_update_public_pool_tx_target(
            &a.l2_update_public_pool_tx_target.inner,
            &b.l2_update_public_pool_tx,
        )?;
        self.set_l2_transfer_tx_target(&a.l2_transfer_tx_target.inner, &b.l2_transfer_tx)?;
        self.set_l2_withdraw_tx_target(&a.l2_withdraw_tx_target, &b.l2_withdraw_tx)?;
        self.set_l2_create_order_tx_target(
            &a.l2_create_order_tx_target.inner,
            &b.l2_create_order_tx,
        )?;
        self.set_l2_cancel_order_tx_target(
            &a.l2_cancel_order_tx_target.inner,
            &b.l2_cancel_order_tx,
        )?;
        self.set_l2_cancel_all_orders_tx_target(
            &a.l2_cancel_all_orders_tx_target.inner,
            &b.l2_cancel_all_orders_tx,
        )?;
        self.set_l2_modify_order_tx_target(
            &a.l2_modify_order_tx_target.inner,
            &b.l2_modify_order_tx,
        )?;
        self.set_l2_mint_shares_tx_target(&a.l2_mint_shares_tx_target.inner, &b.l2_mint_shares_tx)?;
        self.set_l2_burn_shares_tx_target(&a.l2_burn_shares_tx_target.inner, &b.l2_burn_shares_tx)?;
        self.set_l2_update_leverage_tx_target(
            &a.l2_update_leverage_tx_target.inner,
            &b.l2_update_leverage_tx,
        )?;
        self.set_l2_create_grouped_orders_tx_target(
            &a.l2_create_grouped_orders_tx_target.inner,
            &b.l2_create_grouped_orders_tx,
        )?;
        self.set_l2_update_margin_tx_target(
            &a.l2_update_margin_tx_target.inner,
            &b.l2_update_margin_tx,
        )?;
        self.set_l2_create_staking_pool_tx_target(
            &a.l2_create_staking_pool_tx_target.inner,
            &b.l2_create_staking_pool_tx,
        )?;
        self.set_l2_stake_assets_tx_target(
            &a.l2_stake_assets_tx_target.inner,
            &b.l2_stake_assets_tx,
        )?;
        self.set_l2_unstake_assets_tx_target(
            &a.l2_unstake_assets_tx_target.inner,
            &b.l2_unstake_assets_tx,
        )?;
        self.set_l2_force_burn_shares_tx_target(
            &a.l2_force_burn_shares_tx_target.inner,
            &b.l2_force_burn_shares_tx,
        )?;
        self.set_l2_update_account_config_tx_target(
            &a.l2_update_account_config_tx_target.inner,
            &b.l2_update_account_config_tx,
        )?;
        self.set_l2_strategy_transfer_tx_target(
            &a.l2_strategy_transfer_tx_target.inner,
            &b.l2_strategy_transfer_tx,
        )?;
        self.set_l2_update_market_config_tx_target(
            &a.l2_update_market_config_tx_target.inner,
            &b.l2_update_market_config_tx,
        )?;
        self.set_l2_approve_integrator_tx_target(
            &a.l2_approve_integrator_tx_target.inner,
            &b.l2_approve_integrator_tx,
        )?;
        self.set_l2_update_account_asset_config_tx_target(
            &a.l2_update_account_asset_config_tx_target.inner,
            &b.l2_update_account_asset_config_tx,
        )?;
        self.set_l2_update_asset_config_tx_target(
            &a.l2_update_asset_config_tx_target.inner,
            &b.l2_update_asset_config_tx,
        )?;

        /*************************/
        /* Internal Transactions */
        /*************************/
        self.set_internal_claim_order_tx_target(
            &a.internal_claim_order_tx_target.inner,
            &b.internal_claim_order_tx,
        )?;
        self.set_internal_cancel_order_tx_target(
            &a.internal_cancel_order_tx_target.inner,
            &b.internal_cancel_order_tx,
        )?;
        self.set_internal_deleverage_tx_target(
            &a.internal_deleverage_tx_target.inner,
            &b.internal_deleverage_tx,
        )?;
        self.set_internal_exit_position_tx_target(
            &a.internal_exit_position_tx_target.inner,
            &b.internal_exit_position_tx,
        )?;
        self.set_internal_cancel_all_orders_tx_target(
            &a.internal_cancel_all_orders_tx_target.inner,
            &b.internal_cancel_all_orders_tx,
        )?;
        self.set_internal_liquidate_position_tx_target(
            &a.internal_liquidate_position_tx_target.inner,
            &b.internal_liquidate_position_tx,
        )?;
        self.set_internal_create_order_tx_target(
            &a.internal_create_order_tx_target.inner,
            &b.internal_create_order_tx,
        )?;
        self.set_internal_pending_unlock_tx_target(
            &a.internal_pending_unlock_tx_target.inner,
            &b.internal_pending_unlock_tx,
        )?;
        self.set_internal_transfer_tx_target(
            &a.internal_transfer_tx_target.inner,
            &b.internal_transfer_tx,
        )?;
        self.set_internal_liquidate_spot_tx_target(
            &a.internal_liquidate_spot_tx_target.inner,
            &b.internal_liquidate_spot_tx,
        )?;

        /***********************/
        /*  Transactions Data  */
        /***********************/
        self.set_target(a.nonce, F::from_noncanonical_i64(b.nonce))?;
        self.set_target(a.expired_at, F::from_canonical_i64(b.expired_at))?;
        self.set_schnorr_sig_target(&a.signature, &b.signature)?;
        self.set_signed_target(a.taker_fee, b.taker_fee)?;
        self.set_signed_target(a.maker_fee, b.maker_fee)?;

        if let Some(ref l1_signature) = b.l1_signature {
            self.set_ecdsa_signature_target(&a.l1_signature, l1_signature)?;
        } else {
            self.set_ecdsa_signature_target(
                &a.l1_signature,
                &ECDSASignature {
                    r: Secp256K1Scalar::ZERO,
                    s: Secp256K1Scalar::ZERO,
                },
            )?;
        }
        if let Some(ref l1_pub_key) = b.l1_pub_key {
            if !l1_pub_key.0.is_valid() {
                anyhow::bail!("Invalid L1 public key. {:?}", l1_pub_key);
            }
            self.set_ecdsa_public_key_target(&a.l1_pub_key, l1_pub_key)?;
        } else {
            self.set_ecdsa_public_key_target(&a.l1_pub_key, &ECDSAPublicKey(AffinePoint::ZERO))?;
        }

        /***********************/
        /*  State Tree Leaves  */
        /***********************/
        self.set_account_target(&a.accounts_before[0], &b.accounts_before[0])?;
        self.set_account_target(&a.accounts_before[1], &b.accounts_before[1])?;
        self.set_fee_account_target(&a.accounts_before[2], &b.accounts_before[2])?;
        self.set_account_delta_target(&a.accounts_delta_before[0], &b.accounts_delta_before[0])?;
        self.set_account_delta_target(&a.accounts_delta_before[1], &b.accounts_delta_before[1])?;
        self.set_fee_account_delta_target(
            &a.accounts_delta_before[2],
            &b.accounts_delta_before[2],
        )?;
        for i in 0..NB_ACCOUNTS_PER_TX {
            for j in 0..NB_ASSETS_PER_TX {
                self.set_account_asset_target(
                    &a.account_assets_before[i][j],
                    &b.account_assets_before[i][j],
                )?;
            }
        }
        for i in 0..NB_ASSETS_PER_TX {
            self.set_target(
                a.asset_indices[i],
                F::from_noncanonical_i64(b.asset_indices[i] as i64),
            )?;
        }

        self.set_api_key_target(&a.api_key_before, &b.api_key_before)?;
        self.set_account_order_target(&a.account_order_before, &b.account_order_before)?;
        self.set_market_target(&a.market_before, &b.market_before)?;
        self.set_order_target(&a.order_before, &b.order_before)?;

        /*****************************/
        /*  State Tree Merkle Proofs */
        /*****************************/
        for i in 0..NB_ACCOUNTS_PER_TX {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                self.set_hash_target(
                    a.account_tree_merkle_proofs[i][j],
                    b.account_tree_merkle_proofs[i][j],
                )?;
            }
        }
        for i in 0..NB_ACCOUNTS_PER_TX {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                self.set_hash_target(
                    a.account_pub_data_tree_merkle_proofs[i][j],
                    b.account_pub_data_tree_merkle_proofs[i][j],
                )?;
            }
        }
        for i in 0..NB_ACCOUNTS_PER_TX {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                self.set_hash_target(
                    a.account_delta_tree_merkle_proofs[i][j],
                    b.account_delta_tree_merkle_proofs[i][j],
                )?;
            }
        }
        for i in 0..NB_ACCOUNTS_PER_TX - 1 {
            for j in 0..POSITION_MERKLE_LEVELS {
                self.set_hash_target(
                    a.position_delta_merkle_proofs[i][j],
                    b.position_delta_merkle_proofs[i][j],
                )?;
            }
        }
        for i in 0..API_KEY_MERKLE_LEVELS {
            self.set_hash_target(
                a.api_key_tree_merkle_proof[i],
                b.api_key_tree_merkle_proof[i],
            )?;
        }
        for i in 0..ACCOUNT_ORDERS_MERKLE_LEVELS {
            for j in 0..NB_ACCOUNT_ORDERS_PATHS_PER_TX {
                self.set_hash_target(
                    a.account_orders_tree_merkle_proof[j][i],
                    b.account_orders_tree_merkle_proof[j][i],
                )?;
            }
        }
        for i in 0..NB_ACCOUNTS_PER_TX {
            for j in 0..NB_ASSETS_PER_TX {
                for k in 0..ASSET_MERKLE_LEVELS {
                    self.set_hash_target(
                        a.asset_tree_merkle_proofs[i][j][k],
                        b.asset_tree_merkle_proofs[i][j][k],
                    )?;
                    self.set_hash_target(
                        a.public_asset_tree_merkle_proofs[i][j][k],
                        b.public_asset_tree_merkle_proofs[i][j][k],
                    )?;
                    self.set_hash_target(
                        a.asset_delta_tree_merkle_proofs[i][j][k],
                        b.asset_delta_tree_merkle_proofs[i][j][k],
                    )?;
                }
            }
        }
        for i in 0..MARKET_MERKLE_LEVELS {
            self.set_hash_target(a.market_tree_merkle_proof[i], b.market_tree_merkle_proof[i])?;
        }
        for i in 0..ORDER_BOOK_MERKLE_LEVELS {
            self.set_order_book_node_target(
                &a.order_book_tree_path[i],
                &b.order_book_tree_path[i],
            )?;
        }

        /*************************/
        /*  Impact Price Helpers */
        /*************************/
        self.set_order_target(&a.impact_ask_order, &b.impact_ask_order)?;
        self.set_order_target(&a.impact_bid_order, &b.impact_bid_order)?;
        for i in 0..ORDER_BOOK_MERKLE_LEVELS {
            self.set_order_book_node_target(
                &a.impact_ask_order_book_tree_path[i],
                &b.impact_ask_order_book_tree_path[i],
            )?;
        }
        for i in 0..ORDER_BOOK_MERKLE_LEVELS {
            self.set_order_book_node_target(
                &a.impact_bid_order_book_tree_path[i],
                &b.impact_bid_order_book_tree_path[i],
            )?;
        }

        self.set_attributes_tx_target(&a.attributes, &b.attributes)?;

        Ok(())
    }
}
