// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::{BoolTarget, Target};

use super::account_order::AccountOrderTarget;
use super::api_key::ApiKeyTarget;
use super::config::Builder;
use super::register::{RegisterStackTarget, select_register_target};
use crate::bigint::bigint::BigIntTarget;
use crate::bigint::biguint::CircuitBuilderBiguint;
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::signed::signed_target::SignedTarget;
use crate::tx_attributes::TxAttributesTarget;
use crate::types::account::AccountTarget;
use crate::types::account_asset::AccountAssetTarget;
use crate::types::account_delta::AccountDeltaTarget;
use crate::types::account_margined_asset::AccountMarginedAssetTarget;
use crate::types::account_position::AccountPositionTarget;
use crate::types::asset::AssetTarget;
use crate::types::config::BIG_U96_LIMBS;
use crate::types::constants::{
    NB_ACCOUNTS_PER_TX, NB_ASSETS_PER_TX, NB_CLOID_UNIQUENESS_CHECK_PER_TX,
    NEW_INSTRUCTIONS_MAX_SIZE, ORDER_BASE_AMOUNT_BITS, ORDER_BOOK_MERKLE_LEVELS,
};
use crate::types::margined_asset::MarginedAssetTarget;
use crate::types::market::MarketTarget;
use crate::types::market_details::MarketDetailsTarget;
use crate::types::order::OrderTarget;
use crate::types::order_book_node::OrderBookNodeTarget;
use crate::types::public_pool::PublicPoolShareTarget;
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::risk_info::RiskInfoTarget;
use crate::types::system_config::SystemConfigTarget;

pub struct TxState {
    /****************/
    /* State Leaves */
    /****************/
    pub new_instructions: [BaseRegisterInfoTarget; NEW_INSTRUCTIONS_MAX_SIZE],
    pub system_config: SystemConfigTarget,
    pub new_instructions_count: Target,
    pub register_stack: RegisterStackTarget,
    pub api_key: ApiKeyTarget,
    pub account_order: AccountOrderTarget,
    pub accounts: [AccountTarget; NB_ACCOUNTS_PER_TX],
    pub account_assets: [[AccountAssetTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX],
    pub account_margined_assets:
        [[AccountMarginedAssetTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // First two slots are mutable, third slot is immutable and read-only
    pub accounts_delta: [AccountDeltaTarget; NB_ACCOUNTS_PER_TX],
    pub market: MarketTarget,
    pub market_details: MarketDetailsTarget,
    pub order: OrderTarget,
    pub order_book_tree_path: [OrderBookNodeTarget; ORDER_BOOK_MERKLE_LEVELS],
    pub assets: [AssetTarget; NB_ASSETS_PER_TX], // First slot is mutable, second and third slot is immutable and read-only
    pub margined_asset: [MarginedAssetTarget; NB_ASSETS_PER_TX], // First two slots are mutable, third slot is immutable and read-only
    pub asset_indices: [Target; NB_ASSETS_PER_TX],

    /***********/
    /* Helpers */
    /***********/
    pub next_margin_asset_index: Target,
    pub first_asset_margin_index: Target,
    pub is_new_account: [BoolTarget; NB_ACCOUNTS_PER_TX],
    pub positions: [AccountPositionTarget; NB_ACCOUNTS_PER_TX - 1],
    pub risk_infos: [RiskInfoTarget; NB_ACCOUNTS_PER_TX - 1],
    /// Store used strategies in the tx. For L2_STRATEGY_TRANSFER, 0 is from_strategy and 1 is to_strategy.
    ///  For other tx types, 0 is first account's strategy 1 is second account's strategy and 2 is third account's strategy (if exists).
    pub strategies: [BigIntTarget; NB_ACCOUNTS_PER_TX],
    pub is_asset_used_as_margin: [[BoolTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], // If it is true, then the asset is used as margin for the account
    pub order_path_helper: [BoolTarget; ORDER_BOOK_MERKLE_LEVELS],
    pub matching_engine_flag: BoolTarget,
    pub update_impact_prices_flag: BoolTarget,
    pub taker_fee: SignedTarget,
    pub maker_fee: SignedTarget,
    pub block_timestamp: Target,
    pub is_sender_receiver_different: BoolTarget,
    pub fee_account_is_taker: BoolTarget,
    pub fee_account_is_maker: BoolTarget,
    pub is_cloid_unique: [BoolTarget; NB_CLOID_UNIQUENESS_CHECK_PER_TX],
    pub public_pool_share: PublicPoolShareTarget,
    pub apply_pool_share_delta_flag: BoolTarget,
    pub between_strategies_flag: BoolTarget, // Indicates that we are transfering between different strategies of the same account

    /**************/
    /* Attributes */
    /**************/
    pub attributes: TxAttributesTarget,
}

impl Default for TxState {
    fn default() -> Self {
        Self {
            system_config: SystemConfigTarget::default(),
            new_instructions: [BaseRegisterInfoTarget::default(); NEW_INSTRUCTIONS_MAX_SIZE],
            new_instructions_count: Target::default(),
            register_stack: RegisterStackTarget::default(),
            api_key: ApiKeyTarget::default(),
            account_order: AccountOrderTarget::default(),
            accounts: core::array::from_fn(|_| AccountTarget::default()),
            account_assets: core::array::from_fn(|_| {
                core::array::from_fn(|_| AccountAssetTarget::default())
            }),
            account_margined_assets: core::array::from_fn(|_| {
                core::array::from_fn(|_| AccountMarginedAssetTarget::default())
            }),
            accounts_delta: core::array::from_fn(|_| AccountDeltaTarget::default()),
            assets: core::array::from_fn(|_| AssetTarget::default()),
            margined_asset: core::array::from_fn(|_| MarginedAssetTarget::default()),
            asset_indices: core::array::from_fn(|_| Target::default()),
            market: MarketTarget::default(),
            market_details: MarketDetailsTarget::default(), // Only relevant for perps
            order: OrderTarget::default(),
            order_book_tree_path: core::array::from_fn(|_| OrderBookNodeTarget::default()),
            is_new_account: core::array::from_fn(|_| BoolTarget::default()),
            positions: core::array::from_fn(|_| AccountPositionTarget::default()),
            risk_infos: core::array::from_fn(|_| RiskInfoTarget::default()),
            strategies: core::array::from_fn(|_| BigIntTarget::default()),
            is_asset_used_as_margin: core::array::from_fn(|_| {
                core::array::from_fn(|_| BoolTarget::default())
            }),
            order_path_helper: core::array::from_fn(|_| BoolTarget::default()),
            matching_engine_flag: BoolTarget::default(),
            update_impact_prices_flag: BoolTarget::default(),
            taker_fee: SignedTarget::default(),
            maker_fee: SignedTarget::default(),
            block_timestamp: Target::default(),
            is_sender_receiver_different: BoolTarget::default(),
            fee_account_is_taker: BoolTarget::default(),
            fee_account_is_maker: BoolTarget::default(),
            is_cloid_unique: core::array::from_fn(|_| BoolTarget::default()),
            public_pool_share: PublicPoolShareTarget::default(),
            apply_pool_share_delta_flag: BoolTarget::default(),
            between_strategies_flag: BoolTarget::default(),
            next_margin_asset_index: Target::default(),
            first_asset_margin_index: Target::default(),

            attributes: TxAttributesTarget::default(),
        }
    }
}

impl TxState {
    pub fn get_attribute(&self, attribute_type: usize) -> Target {
        self.attributes.get(attribute_type)
    }

    /// Caller's responsibility to ensure there are no overrides, meaning
    /// only one branch of code that writes to same index should be enabled.
    pub fn put_to_instruction_stack_unsafe(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        instruction: &BaseRegisterInfoTarget,
        target_index: usize,
    ) {
        self.new_instructions_count = builder.add(self.new_instructions_count, is_enabled.target);
        self.new_instructions[target_index] = select_register_target(
            builder,
            is_enabled,
            instruction,
            &self.new_instructions[target_index],
        );
    }

    pub fn push_instruction_stack<const MAX_NEW_INSTR: usize>(&mut self, builder: &mut Builder) {
        self.register_stack.push_instructions::<MAX_NEW_INSTR>(
            builder,
            &self.new_instructions,
            self.new_instructions_count,
        );
        self.new_instructions_count = builder.zero();
        self.new_instructions = [BaseRegisterInfoTarget::empty(builder); NEW_INSTRUCTIONS_MAX_SIZE];
    }

    pub fn is_valid_base_size_and_price(
        &self,
        builder: &mut Builder,
        base_amount: Target,
        price: Target,
        is_twap_order: BoolTarget,
        is_ioc: BoolTarget,
    ) -> BoolTarget {
        let price_big = builder.target_to_biguint_single_limb_unsafe(price);
        let base_amount_big = builder.target_to_biguint(base_amount);
        let quote_big = builder.mul_biguint_non_carry(&base_amount_big, &price_big, BIG_U96_LIMBS);

        let min_quote_big = builder.target_to_biguint(self.market.min_quote_amount);
        let max_quote_big = builder.target_to_biguint(self.market.order_quote_limit);

        let quote_lt_min_quote_amount = builder.is_lt_biguint(&quote_big, &min_quote_big);
        let quote_gt_max_quote_amount = builder.is_gt_biguint(&quote_big, &max_quote_big);
        let base_amount_lt_min_base_amount = builder.is_lt(
            base_amount,
            self.market.min_base_amount,
            ORDER_BASE_AMOUNT_BITS,
        );

        let twap_or_ioc = builder.or(is_twap_order, is_ioc);
        let not_twap_nor_ioc = builder.not(twap_or_ioc);
        let assertions = [
            not_twap_nor_ioc,
            builder.or(quote_lt_min_quote_amount, base_amount_lt_min_base_amount),
        ];
        let should_be_false = builder.multi_and(&assertions);

        let should_be_false = builder.or(should_be_false, quote_gt_max_quote_amount);
        builder.not(should_be_false)
    }
}

#[cfg(test)]
mod tests {
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::*;
    use crate::bool_utils::CircuitBuilderBoolUtils;
    use crate::types::config::{Builder, C, F};
    use crate::types::constants::{INSERT_MAX_SIX_REGISTERS, REGISTER_STACK_SIZE};
    use crate::types::tx_state::TxState;

    #[test]
    fn register_stack_insert_and_push() {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

        let _true = builder._true();
        let _false = builder._false();
        let zero = builder.zero();

        for i in 0..NEW_INSTRUCTIONS_MAX_SIZE {
            let mut tx_state = init_tx_state(&mut builder);

            for j in 0..i {
                let new_instruction = BaseRegisterInfoTarget::random(&mut builder);

                // Insert to new instructions stack - False flag
                tx_state.put_to_instruction_stack_unsafe(
                    &mut builder,
                    _false,
                    &new_instruction,
                    NEW_INSTRUCTIONS_MAX_SIZE - 1 - j,
                );
                let current_count = builder.constant_usize(j);
                builder.conditional_assert_eq(
                    _true,
                    tx_state.new_instructions_count,
                    current_count,
                );
                let check = tx_state.new_instructions[NEW_INSTRUCTIONS_MAX_SIZE - 1 - j]
                    .is_empty(&mut builder);
                builder.assert_true(check);

                // Insert to new instructions stack - True flag
                tx_state.put_to_instruction_stack_unsafe(
                    &mut builder,
                    _true,
                    &new_instruction,
                    NEW_INSTRUCTIONS_MAX_SIZE - 1 - j,
                );
                let current_count = builder.constant_usize(j + 1);
                builder.conditional_assert_eq(
                    _true,
                    tx_state.new_instructions_count,
                    current_count,
                );
                let check = BaseRegisterInfoTarget::is_equal(
                    &mut builder,
                    &tx_state.new_instructions[NEW_INSTRUCTIONS_MAX_SIZE - 1 - j],
                    &new_instruction,
                );
                builder.assert_true(check);
            }

            // Push to register stack - True flag
            tx_state.push_instruction_stack::<NEW_INSTRUCTIONS_MAX_SIZE>(&mut builder);
            let current_count_in_new_instructions = zero;
            builder.conditional_assert_eq(
                _true,
                tx_state.new_instructions_count,
                current_count_in_new_instructions,
            );
            let current_count_in_register_stack = builder.constant_usize(i);
            builder.conditional_assert_eq(
                _true,
                tx_state.register_stack.count,
                current_count_in_register_stack,
            );
            tx_state.register_stack.iter().take(i).for_each(|reg| {
                let is_empty = reg.is_empty(&mut builder);
                builder.assert_false(is_empty);
            });
            tx_state.register_stack.iter().skip(i).for_each(|reg| {
                let is_empty = reg.is_empty(&mut builder);
                builder.assert_true(is_empty);
            });
        }

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "Partition containing Wire")]
    fn register_stack_push_too_many() {
        let mut builder = Builder::new(CircuitConfig::standard_recursion_config());

        let _true = builder._true();
        let _false = builder._false();

        let mut tx_state = init_tx_state(&mut builder);

        // insert INSERT_MAX_SIX_REGISTERS instructions
        for i in 0..INSERT_MAX_SIX_REGISTERS {
            let new_instruction = BaseRegisterInfoTarget::random(&mut builder);
            tx_state.put_to_instruction_stack_unsafe(
                &mut builder,
                _true,
                &new_instruction,
                INSERT_MAX_SIX_REGISTERS - 1 - i,
            );
        }
        // Push to register stack
        tx_state.push_instruction_stack::<INSERT_MAX_SIX_REGISTERS>(&mut builder);
        // insert REGISTER_STACK_SIZE - INSERT_MAX_SIX_REGISTERS instructions
        for i in INSERT_MAX_SIX_REGISTERS..REGISTER_STACK_SIZE {
            let new_instruction = BaseRegisterInfoTarget::random(&mut builder);
            tx_state.put_to_instruction_stack_unsafe(
                &mut builder,
                _true,
                &new_instruction,
                REGISTER_STACK_SIZE - 1 - i,
            );
        }
        // Push to register stack
        tx_state.push_instruction_stack::<INSERT_MAX_SIX_REGISTERS>(&mut builder);

        // Now inserting one more instruction should fail
        let new_instruction = BaseRegisterInfoTarget::random(&mut builder);
        tx_state.put_to_instruction_stack_unsafe(&mut builder, _true, &new_instruction, 0);
        tx_state.push_instruction_stack::<INSERT_MAX_SIX_REGISTERS>(&mut builder);

        let data = builder.build::<C>();
        data.verify(data.prove(PartialWitness::<F>::new()).unwrap())
            .unwrap();
    }

    fn init_tx_state(builder: &mut Builder) -> TxState {
        let mut tx_state = TxState::default();
        (tx_state.new_instructions, tx_state.new_instructions_count) =
            get_random_new_instructions(builder, 0);
        tx_state.register_stack = RegisterStackTarget::empty(builder);
        tx_state
    }

    fn get_random_new_instructions(
        builder: &mut Builder,
        count: usize,
    ) -> ([BaseRegisterInfoTarget; NEW_INSTRUCTIONS_MAX_SIZE], Target) {
        assert!(count <= NEW_INSTRUCTIONS_MAX_SIZE);
        let mut new_instructions =
            [BaseRegisterInfoTarget::empty(builder); NEW_INSTRUCTIONS_MAX_SIZE];
        for i in (0..count).rev() {
            new_instructions[NEW_INSTRUCTIONS_MAX_SIZE - 1 - i] =
                BaseRegisterInfoTarget::random(builder);
        }
        (new_instructions, builder.constant_usize(count))
    }
}
