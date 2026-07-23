// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::hints::CircuitBuilderHints;
use crate::matching_engine::execute_matching_light;
use crate::merkle_helpers::{
    account_client_order_index_to_merkle_path, account_index_to_merkle_path,
    account_order_index_to_merkle_path, api_key_index_to_merkle_path, asset_index_to_merkle_path,
    recalculate_root, try_verify_merkle_proof, verify_merkle_proof,
};
use crate::order_book_tree_helpers::order_indexes_to_merkle_path;
use crate::tx_attributes::ATTR_SKIP_TX_NONCE;
use crate::tx_constraints::{TxTarget, compute_validium_and_state_root};
use crate::types::account::AccountTarget;
use crate::types::account_margined_asset::AccountMarginedAssetTarget;
use crate::types::account_position::{
    AccountPositionTarget, PositionWithDelta, random_access_account_position,
};
use crate::types::asset::{AssetTarget, random_access_assets};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::margined_asset::{MarginedAssetTarget, random_access_margined_assets};
use crate::types::market_details::MarketRiskDetailsTarget;
use crate::types::public_pool::PublicPoolShareTarget;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::risk_info::RiskInfoTarget;
use crate::types::tx_state::TxState;
use crate::types::tx_type::{TxTypeTargets, TxTypeVerifyTargets};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::utils::CircuitBuilderUtils;

impl TxTarget {
    pub fn define_light(
        &mut self,
        _index: usize,
        chain_id: u32,
        builder: &mut Builder,
        block_created_at: Target,
        state_metadata_hash: HashOutTarget,
    ) -> (
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
        BoolTarget,
        [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
        BoolTarget,
        HashOutTarget, // public market details hash
        HashOutTarget, // account delta tree root
    ) {
        let tx_type = TxTypeTargets::new(builder, self.tx_type);

        let is_allowed_tx_type = builder.multi_or(&[
            tx_type.is_empty,
            tx_type.is_l2_create_order,
            tx_type.is_l2_cancel_order,
            tx_type.is_l2_modify_order,
            tx_type.is_internal_claim_order,
        ]);
        builder.assert_true(is_allowed_tx_type);

        let tx_hash = self.select_light_tx_hash(builder, &tx_type, chain_id);
        let account_pk = self.api_key_before.public_key;
        let partial_main_account = self.select_light_partial_main_account(builder);

        let nil_account_index = builder.constant_i64(NIL_ACCOUNT_INDEX);
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
                instruction_type: self.register_stack_before[0].instruction_type,
                tx_sender_account_partial: partial_main_account,
                sub_account_index: nil_account_index,
                skip_tx_nonce: self.attributes.get(ATTR_SKIP_TX_NONCE),
            },
        );

        let assets_before: [AssetTarget; NB_ASSETS_PER_TX] = core::array::from_fn(|i| {
            random_access_assets(
                builder,
                self.asset_indices[i],
                self.all_assets_before.to_vec(),
            )
        });
        let first_asset_margin_index = assets_before[0].margin_index(builder);
        let second_asset_margin_index = assets_before[1].margin_index(builder);
        let margined_asset_before = [
            random_access_margined_assets(
                builder,
                first_asset_margin_index,
                &self.all_margined_assets_before,
            ),
            random_access_margined_assets(
                builder,
                second_asset_margin_index,
                &self.all_margined_assets_before,
            ),
        ];

        let positions_with_pub_data_before: [PositionWithDelta; 1] =
            PositionWithDelta::new_positions_with_pub_data_from_accounts(
                builder,
                self.market_before.perps_market_index,
                &self.accounts_before[..1],
                &self.accounts_delta_before[..1],
            );
        let owner_position_before = positions_with_pub_data_before[0].position.clone();

        let account_margined_assets_before: [[AccountMarginedAssetTarget; NB_ASSETS_PER_TX]; 1] =
            AccountTarget::get_margined_asset_balances(
                builder,
                array::from_ref(&self.accounts_before[OWNER_ACCOUNT_ID]),
                &assets_before,
                first_asset_margin_index,
            );
        let account_margined_assets_before = account_margined_assets_before[0].clone();
        let is_asset_used_as_margin: [BoolTarget; NB_ASSETS_PER_TX] =
            account_margined_assets_before
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
                .unwrap();

        let market_risk_details_before =
            self.get_market_risk_details_before(builder, &self.all_market_risk_details_before);

        let owner_risk_info_before = self.get_light_owner_risk_info_before(
            builder,
            &owner_position_before,
            &market_risk_details_before,
            &self.all_market_risk_details_before,
            &self.all_margined_assets_before,
        );
        let order_path_helper = order_indexes_to_merkle_path(
            builder,
            self.order_before.price_index,
            self.order_before.nonce_index,
        );

        let mut owner_position_bucket_hashes =
            self.accounts_before[OWNER_ACCOUNT_ID].get_position_bucket_hashes(builder);
        let (old_owner_account_hash, _, owner_account_is_empty) =
            self.accounts_before[OWNER_ACCOUNT_ID].hash(builder, &owner_position_bucket_hashes);
        let old_owner_asset_hashes: [HashOutTarget; NB_ASSETS_PER_TX] =
            core::array::from_fn(|j| self.account_assets_before[OWNER_ACCOUNT_ID][j].hash(builder));

        self.attributes.sanitize_and_normalize(
            builder,
            &self.accounts_before[OWNER_ACCOUNT_ID],
            &self.market_before,
            &self.system_config_before,
            block_created_at,
        );

        let _false = builder._false();
        let _true = builder._true();
        let nil_margin_asset_index = builder.constant_u64(NIL_MARGIN_ASSET_INDEX);
        let zero_bigint = builder.zero_bigint();
        let tx_state = &mut TxState {
            first_asset_margin_index,
            next_margin_asset_index: nil_margin_asset_index,
            new_instructions: [BaseRegisterInfoTarget::empty(builder); NEW_INSTRUCTIONS_MAX_SIZE],
            new_instructions_count: builder.zero(),
            register_stack: self.register_stack_before,
            system_config: self.system_config_before,
            accounts: self.accounts_before.clone(),
            accounts_delta: self.accounts_delta_before.clone(),
            is_new_account: [owner_account_is_empty, _false, _false],
            market: self.market_before.clone(),
            market_details: self.market_details_before.clone(),
            market_risk_details: market_risk_details_before.clone(),
            order: self.order_before.clone(),
            order_book_tree_path: self.order_book_tree_path.clone(),
            positions: [
                owner_position_before.clone(),
                AccountPositionTarget::default(),
            ],
            risk_infos: [owner_risk_info_before.clone(), RiskInfoTarget::default()],
            strategies: core::array::from_fn(|_| zero_bigint.clone()),
            is_asset_used_as_margin: core::array::from_fn(|_| is_asset_used_as_margin),
            order_path_helper,
            matching_engine_flag: builder._false(),
            update_impact_prices_flag: builder._false(),
            taker_fee: self.taker_fee,
            maker_fee: self.maker_fee,
            api_key: self.api_key_before.clone(),
            account_order: self.account_order_before.clone(),
            account_assets: self.account_assets_before.clone(),
            account_margined_assets: core::array::from_fn(|_| {
                account_margined_assets_before.clone()
            }),
            assets: assets_before.clone(),
            margined_asset: margined_asset_before.clone(),
            asset_indices: self.asset_indices,
            block_timestamp: block_created_at,
            is_sender_receiver_different: _false,
            fee_account_is_taker: _true,
            fee_account_is_maker: _false,
            is_cloid_unique: self.get_light_is_cloid_unique_infos(builder, &tx_type),
            public_pool_share: PublicPoolShareTarget::default(),
            apply_pool_share_delta_flag: builder._false(),
            between_strategies_flag: _false,
            attributes: self.attributes.clone(),
        };

        self.validate_light_asset_indices(builder);

        self.l2_create_order_tx_target
            .verify(builder, &tx_type, tx_state);
        self.l2_cancel_order_tx_target
            .verify(builder, &tx_type, tx_state);
        self.l2_modify_order_tx_target
            .verify(builder, &tx_type, tx_state);
        self.internal_claim_order_tx_target
            .verify(builder, &tx_type, tx_state);

        self.l2_create_order_tx_target.apply(builder, tx_state);
        self.l2_cancel_order_tx_target.apply(builder, tx_state);
        self.l2_modify_order_tx_target.apply(builder, tx_state);
        self.internal_claim_order_tx_target.apply(builder, tx_state);
        let next_nonce = builder.add_one(self.nonce);
        tx_state.api_key.nonce =
            builder.select(tx_type.is_layer2, next_nonce, tx_state.api_key.nonce);

        tx_state.push_instruction_stack::<INSERT_MAX_ONE_REGISTER>(builder);

        execute_matching_light(builder, tx_state);
        tx_state.push_instruction_stack::<INSERT_MAX_ONE_REGISTER>(builder);

        self.apply_light_position_diff(
            builder,
            tx_state,
            &owner_position_before,
            &mut owner_position_bucket_hashes,
        );

        self.update_impact_prices(builder, tx_state, &market_risk_details_before);

        let (
            // These fields won't change for light path. Use for old and new state root.
            system_config_hash,
            assets_hash,
            margined_assets_hash,
            market_risk_details_hash,
            public_market_details_hash,
            _,
        ) = self.verify_old_state_root(builder, state_metadata_hash);

        self.verify_light_api_key_merkle_proof(builder, tx_state);

        self.verify_light_account_orders_merkle_proof(builder, tx_state);

        self.verify_light_assets_merkle_proofs(builder, tx_state, &old_owner_asset_hashes);

        let current_account_tree_root = self.verify_light_account_merkle_proof(
            builder,
            tx_state,
            self.old_account_tree_root,
            old_owner_account_hash,
            &owner_position_bucket_hashes,
        );

        let current_market_tree_root =
            self.verify_market_and_order_book_proofs(builder, tx_state, self.old_market_tree_root);

        let current_market_details_tree_root = self.verify_market_details_merkle_proof(
            builder,
            tx_state,
            self.old_market_details_tree_root,
        );

        let current_register_stack_hash = tx_state.register_stack.hash(builder);

        let (new_validium_root, new_state_root) = compute_validium_and_state_root(
            builder,
            system_config_hash,
            assets_hash,
            margined_assets_hash,
            market_risk_details_hash,
            public_market_details_hash,
            current_register_stack_hash,
            current_account_tree_root,
            self.old_account_pub_data_tree_root,
            current_market_details_tree_root,
            current_market_tree_root,
            state_metadata_hash,
        );
        builder.connect_hashes(self.new_validium_root, new_validium_root);
        builder.connect_hashes(self.new_state_root, new_state_root);

        (
            [builder.zero_u8(); MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
            builder._false(),
            [builder.zero_u8(); ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],
            builder._false(),
            public_market_details_hash,
            self.old_account_delta_tree_root,
        )
    }

    fn select_light_tx_hash(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
        chain_id: u32,
    ) -> QuinticExtensionTarget {
        let mut selected_hash = builder.zero_quintic_ext();

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

        let l2_modify_order_tx_hash =
            self.l2_modify_order_tx_target
                .hash(builder, self.nonce, self.expired_at, chain_id);
        selected_hash = builder.select_quintic_ext(
            tx_type.is_l2_modify_order,
            l2_modify_order_tx_hash,
            selected_hash,
        );

        self.attributes.aggregate_tx_hash(builder, selected_hash)
    }

    fn select_light_partial_main_account(&self, _builder: &mut Builder) -> AccountTarget {
        self.accounts_before[OWNER_ACCOUNT_ID].clone()
    }

    fn get_light_owner_risk_info_before(
        &self,
        builder: &mut Builder,
        owner_position_before: &AccountPositionTarget,
        current_market_details_before: &MarketRiskDetailsTarget,
        all_market_risk_details_before: &[MarketRiskDetailsTarget; POSITION_LIST_SIZE],
        all_margined_assets_before: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    ) -> RiskInfoTarget {
        let default_strategy_index = builder.constant_usize(DEFAULT_STRATEGY_INDEX);

        let use_strategy = builder.is_equal_constant(
            self.accounts_before[OWNER_ACCOUNT_ID].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let strategy_index = builder.select(
            use_strategy,
            current_market_details_before.strategy_index,
            default_strategy_index,
        );

        let mut margined_assets = self.accounts_before[OWNER_ACCOUNT_ID]
            .margined_assets
            .clone();
        margined_assets[USDC_MARGIN_ASSET_INDEX].balance = self.accounts_before[OWNER_ACCOUNT_ID]
            .get_relevant_usdc_collateral(builder, strategy_index);

        let partial_account = AccountTarget {
            positions: self.accounts_before[OWNER_ACCOUNT_ID].positions.clone(),
            margined_assets,
            ..AccountTarget::default()
        };

        let all_market_details =
            all_market_risk_details_before
                .clone()
                .map(|md| MarketRiskDetailsTarget {
                    strategy_index: builder.select(
                        use_strategy,
                        md.strategy_index,
                        default_strategy_index,
                    ),
                    ..md
                });

        let current_market_details = MarketRiskDetailsTarget {
            strategy_index: builder.select(
                use_strategy,
                current_market_details_before.strategy_index,
                default_strategy_index,
            ),
            ..current_market_details_before.clone()
        };

        RiskInfoTarget::new_light(
            builder,
            &partial_account,
            owner_position_before,
            &current_market_details,
            &all_market_details,
            all_margined_assets_before,
            strategy_index,
        )
    }

    fn get_light_is_cloid_unique_infos(
        &self,
        builder: &mut Builder,
        tx_type: &TxTypeTargets,
    ) -> [BoolTarget; NB_CLOID_UNIQUENESS_CHECK_PER_TX] {
        let nil_cloid = builder.constant_i64(NIL_CLIENT_ORDER_INDEX);
        let cloid = builder.select(
            tx_type.is_l2_create_order,
            self.l2_create_order_tx_target.inner.client_order_index,
            nil_cloid,
        );
        let empty_hash = builder.zero_hash_out();
        let path = account_client_order_index_to_merkle_path(builder, cloid);
        let is_first_cloid_unique = try_verify_merkle_proof(
            builder,
            &self.accounts_before[OWNER_ACCOUNT_ID].account_orders_root,
            empty_hash,
            self.account_orders_tree_merkle_proof[2],
            path,
        );

        let _true = builder._true();
        let mut result = [_true; NB_CLOID_UNIQUENESS_CHECK_PER_TX];
        result[0] = is_first_cloid_unique;
        result
    }

    fn validate_light_asset_indices(&self, builder: &mut Builder) {
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

        for i in [TX_ASSET_ID, FEE_ASSET_ID] {
            builder.connect(
                self.asset_indices[i],
                self.account_assets_before[OWNER_ACCOUNT_ID][i].index_0,
            );
        }
    }

    fn apply_light_position_diff(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        owner_position_before: &AccountPositionTarget,
        owner_position_bucket_hashes: &mut [[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2],
    ) {
        let position_hash_bucket_count = builder.constant_usize(POSITION_HASH_BUCKET_COUNT);
        let (current_bucket_index, index_in_bucket) = builder.div_rem(
            tx_state.market.perps_market_index,
            position_hash_bucket_count,
            5,
        );

        let empty_position = AccountPositionTarget::empty(builder);

        let position_diff = AccountPositionTarget::diff(
            builder,
            &tx_state.positions[OWNER_ACCOUNT_ID],
            owner_position_before,
        );

        let mut position_bucket: [AccountPositionTarget; POSITION_HASH_BUCKET_SIZE] =
            array::from_fn(|j| {
                let candidates = (0..POSITION_HASH_BUCKET_COUNT)
                    .map(|bucket_index| {
                        let position_index = bucket_index * POSITION_HASH_BUCKET_SIZE + j;
                        if position_index < tx_state.accounts[OWNER_ACCOUNT_ID].positions.len() {
                            tx_state.accounts[OWNER_ACCOUNT_ID].positions[position_index].clone()
                        } else {
                            empty_position.clone()
                        }
                    })
                    .collect();
                random_access_account_position(builder, current_bucket_index, candidates)
            });

        for market_index_in_bucket in 0..POSITION_HASH_BUCKET_SIZE {
            let t_market_index_in_bucket = builder.constant_usize(market_index_in_bucket);
            let is_current_position = builder.is_equal(t_market_index_in_bucket, index_in_bucket);
            position_bucket[market_index_in_bucket] = AccountPositionTarget::apply_diff(
                builder,
                is_current_position,
                &position_bucket[market_index_in_bucket],
                &position_diff,
            );
        }

        let new_position_bucket_hash =
            AccountTarget::get_position_bucket_hash(builder, &position_bucket);
        for bucket_index in 0..POSITION_HASH_BUCKET_COUNT {
            let t_bucket_index = builder.constant_usize(bucket_index);
            let is_current_position_bucket = builder.is_equal(t_bucket_index, current_bucket_index);

            for j in 0..2 {
                owner_position_bucket_hashes[j][bucket_index] = builder.select_hash(
                    is_current_position_bucket,
                    &new_position_bucket_hash[j],
                    &owner_position_bucket_hashes[j][bucket_index],
                );
            }
        }
    }

    fn verify_light_api_key_merkle_proof(&self, builder: &mut Builder, tx_state: &mut TxState) {
        let api_key_before_hash = self.api_key_before.hash(builder);
        let api_key_merkle_path =
            api_key_index_to_merkle_path(builder, self.api_key_before.api_key_index);

        verify_merkle_proof(
            builder,
            &self.accounts_before[OWNER_ACCOUNT_ID].api_key_root,
            api_key_before_hash,
            self.api_key_tree_merkle_proof,
            api_key_merkle_path,
        );

        let new_api_key_hash = tx_state.api_key.hash(builder);
        tx_state.accounts[OWNER_ACCOUNT_ID].api_key_root = recalculate_root(
            builder,
            new_api_key_hash,
            self.api_key_tree_merkle_proof,
            api_key_merkle_path,
        );
    }

    fn verify_light_account_orders_merkle_proof(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
    ) {
        let is_index_0_is_zero = builder.is_zero(self.account_order_before.index_0);
        let max_client_order_index = builder.constant_u64(MAX_CLIENT_ORDER_INDEX as u64);
        let is_index_0_valid = builder.is_gt(
            self.account_order_before.index_0,
            max_client_order_index,
            64,
        );
        let is_valid_index_0 = builder.or(is_index_0_is_zero, is_index_0_valid);
        builder.assert_true(is_valid_index_0);

        builder.register_range_check(self.account_order_before.index_1, CLIENT_ORDER_INDEX_BITS);

        let account_order_hash_for_index_0 = self.account_order_before.hash(builder);
        let account_orders_merkle_path_for_index_0 =
            account_order_index_to_merkle_path(builder, self.account_order_before.index_0);
        verify_merkle_proof(
            builder,
            &self.accounts_before[OWNER_ACCOUNT_ID].account_orders_root,
            account_order_hash_for_index_0,
            self.account_orders_tree_merkle_proof[0],
            account_orders_merkle_path_for_index_0,
        );
        let new_account_order_hash_for_index_0 = tx_state.account_order.hash(builder);

        let account_orders_root_after_oid_leaf_inserted = recalculate_root(
            builder,
            new_account_order_hash_for_index_0,
            self.account_orders_tree_merkle_proof[0],
            account_orders_merkle_path_for_index_0,
        );

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
            self.account_orders_tree_merkle_proof[1],
            account_orders_merkle_path_for_index_1,
        );

        let new_account_order_hash_for_index_1 = builder.select_hash(
            index_1_empty,
            &empty_account_order_hash,
            &new_account_order_hash_for_index_0,
        );
        tx_state.accounts[OWNER_ACCOUNT_ID].account_orders_root = recalculate_root(
            builder,
            new_account_order_hash_for_index_1,
            self.account_orders_tree_merkle_proof[1],
            account_orders_merkle_path_for_index_1,
        );
    }

    fn verify_light_assets_merkle_proofs(
        &self,
        builder: &mut Builder,
        tx_state: &mut TxState,
        old_owner_asset_hashes: &[HashOutTarget; NB_ASSETS_PER_TX],
    ) {
        let merkle_paths = [
            asset_index_to_merkle_path(builder, self.asset_indices[0]),
            asset_index_to_merkle_path(builder, self.asset_indices[1]),
        ];

        let mut asset_root = tx_state.accounts[OWNER_ACCOUNT_ID].asset_root;
        for k in 0..NB_ASSETS_PER_TX {
            verify_merkle_proof(
                builder,
                &asset_root,
                old_owner_asset_hashes[k],
                self.asset_tree_merkle_proofs[OWNER_ACCOUNT_ID][k],
                merkle_paths[k],
            );
            let new_hash = tx_state.account_assets[OWNER_ACCOUNT_ID][k].hash(builder);
            asset_root = recalculate_root(
                builder,
                new_hash,
                self.asset_tree_merkle_proofs[OWNER_ACCOUNT_ID][k],
                merkle_paths[k],
            );
        }
        tx_state.accounts[OWNER_ACCOUNT_ID].asset_root = asset_root;
    }

    fn verify_light_account_merkle_proof(
        &self,
        builder: &mut Builder,
        tx_state: &TxState,
        account_tree_root_before: HashOutTarget,
        old_owner_account_hash: HashOutTarget,
        owner_position_bucket_hashes: &[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2],
    ) -> HashOutTarget {
        let account_merkle_path = account_index_to_merkle_path(
            builder,
            self.accounts_before[OWNER_ACCOUNT_ID].account_index,
        );

        verify_merkle_proof(
            builder,
            &account_tree_root_before,
            old_owner_account_hash,
            self.account_tree_merkle_proofs[OWNER_ACCOUNT_ID],
            account_merkle_path,
        );

        let (new_account_hash, _, _) =
            tx_state.accounts[OWNER_ACCOUNT_ID].hash(builder, owner_position_bucket_hashes);

        recalculate_root(
            builder,
            new_account_hash,
            self.account_tree_merkle_proofs[OWNER_ACCOUNT_ID],
            account_merkle_path,
        )
    }
}
