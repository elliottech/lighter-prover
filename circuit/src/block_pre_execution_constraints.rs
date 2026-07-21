// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use itertools::Itertools;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::bigint::big_u16::{BigIntU16Target, CircuitBuilderBigIntU16, CircuitBuilderBiguint16};
use crate::bigint::bigint::SignTarget;
use crate::block_pre_execution::BlockPreExec;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::hints::CircuitBuilderHints;
use crate::signed::signed_target::{CircuitBuilderSigned, SignedTarget};
use crate::types::asset::{AssetTarget, AssetTargetWitness, all_assets_hash};
use crate::types::config::{BIGU16_U64_LIMBS, Builder, C, D, F};
use crate::types::constants::*;
use crate::types::margined_asset::{
    MarginedAssetTarget, MarginedAssetTargetWitness, all_margined_assets_hash,
    connect_margined_assets,
};
use crate::types::market_details::{
    MarketDetailsTarget, MarketDetailsWitness, all_market_details_hash,
    all_public_market_details_hash, connect_market_details,
};
use crate::types::price_updates::{PriceUpdatesTarget, PriceUpdatesWitness};
use crate::types::register::{RegisterInfoTargetWitness, RegisterStackTarget};
use crate::types::state_metadata::{
    StateMetadataTarget, StateMetadataTargetWitness, connect_state_metadata_target,
};
use crate::types::system_config::{SystemConfigTarget, SystemConfigTargetWitness};
use crate::utils::{
    CircuitBuilderUtils, round_unix_timestamp_to_next_hour, round_unix_timestamp_to_previous_minute,
};

pub trait Circuit<
    C: GenericConfig<D, F = F>,
    F: RichField + Extendable<D> + Extendable<5>,
    const D: usize,
>
{
    /// Defines the circuit and its each target. Returns `builder` and `target`
    ///
    /// `builder` can be used to build circuit via calling [`Builder::build()`]
    ///
    /// `target` can be used to assign partial witness in [`BlockPreExecutionCircuit::prove()`] function
    fn define(config: CircuitConfig) -> Self;

    /// Fills partial witness for block target with given block data
    fn generate_witness(
        block: &BlockPreExec<F>,
        target: &BlockPreExecutionTarget,
    ) -> Result<PartialWitness<F>>;
    /// Takes `circuit`, block witness and `target` defined in [`BlockPreExecutionCircuit::define()`] function
    /// and returns the (not-compressed) proof with public inputs
    fn prove(
        circuit: &CircuitData<F, C, D>,
        block: &BlockPreExec<F>,
        bt: &BlockPreExecutionTarget,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;
}

#[derive(Debug)]
pub struct BlockPreExecutionCircuit {
    pub builder: Builder,
    pub target: BlockPreExecutionTarget,
}

#[derive(Debug)]
pub struct BlockPreExecutionTarget {
    pub block_number: Target,
    pub created_at: Target, // 48 bits

    /***********************/
    /*  COMMON STATE DATA  */
    /***********************/
    pub old_system_config: SystemConfigTarget,
    pub register_stack_before: RegisterStackTarget,
    pub all_assets_before: [AssetTarget; ASSET_LIST_SIZE],
    pub all_margined_assets_before: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    pub all_market_details_before: [MarketDetailsTarget; POSITION_LIST_SIZE],
    pub state_metadata_target: StateMetadataTarget,

    /***************************/
    /*  PRE EXECUTION HELPERS  */
    /***************************/
    pub price_updates: PriceUpdatesTarget,
    pub calculate_premium: BoolTarget,
    pub calculate_funding: BoolTarget,
    pub calculate_oracle_prices: BoolTarget,

    /**************************/
    /*  OLD STATE TREE ROOTS  */
    /**************************/
    pub old_account_tree_root: HashOutTarget,
    pub old_account_pub_data_tree_root: HashOutTarget,
    pub old_market_tree_root: HashOutTarget,
    pub old_state_root: HashOutTarget,

    pub all_market_details_after: [MarketDetailsTarget; POSITION_LIST_SIZE], // Public
    pub all_margined_assets_after: [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE], // Public
    pub new_state_metadata_target: StateMetadataTarget,                      // Public
    pub new_state_root: HashOutTarget,                                       // Public
    pub new_validium_root: HashOutTarget,                                    // Public

    // Helpers
    all_assets_hash: HashOutTarget,
}

impl Circuit<C, F, D> for BlockPreExecutionCircuit {
    fn define(config: CircuitConfig) -> Self {
        let mut circuit = Self::new(config);

        circuit.register_public_inputs();

        circuit.define_block_state_data_checks();

        let (all_market_details_after, all_margined_assets_after, new_state_metadata) =
            circuit.define_block_pre_execution();

        circuit.define_post_block_pre_execution(
            &all_market_details_after,
            &all_margined_assets_after,
            &new_state_metadata,
        );

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn prove(
        circuit: &CircuitData<F, C, D>,
        block: &BlockPreExec<F>,
        target: &BlockPreExecutionTarget,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlockPreExecutionCircuit::prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(block, target)?
        });
        let proof = prove::<F, C, D>(&circuit.prover_only, &circuit.common, pw, &mut timing)?;
        timed!(timing, "verify", { circuit.verify(proof.clone())? });

        timing.print();
        Ok(proof)
    }

    fn generate_witness(
        block: &BlockPreExec<F>,
        target: &BlockPreExecutionTarget,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_target(target.created_at, F::from_canonical_i64(block.created_at))?;
        pw.set_target(
            target.block_number,
            F::from_canonical_u64(block.block_number),
        )?;

        pw.set_register_info_target(&target.register_stack_before, &block.register_stack_before)?;
        pw.set_system_config_target(&target.old_system_config, &block.old_system_config)?;

        target
            .all_assets_before
            .iter()
            .zip(block.all_assets.iter())
            .try_for_each(|(t, ai)| pw.set_asset_target(t, ai))?;

        target
            .all_margined_assets_before
            .iter()
            .zip(block.all_margined_assets.iter())
            .try_for_each(|(t, mai)| pw.set_margined_asset_target(t, mai))?;

        target
            .all_market_details_before
            .iter()
            .zip(block.all_market_details.iter())
            .try_for_each(|(t, mi)| pw.set_market_details_target(t, mi))?;

        pw.set_price_updates_target(&target.price_updates, &block.price_updates)?;
        pw.set_bool_target(target.calculate_premium, block.calculate_premium)?;
        pw.set_bool_target(target.calculate_funding, block.calculate_funding)?;
        pw.set_bool_target(
            target.calculate_oracle_prices,
            block.calculate_oracle_prices,
        )?;

        pw.set_hash_target(target.old_account_tree_root, block.old_account_tree_root)?;
        pw.set_hash_target(
            target.old_account_pub_data_tree_root,
            block.old_account_pub_data_tree_root,
        )?;
        pw.set_hash_target(target.old_market_tree_root, block.old_market_tree_root)?;
        pw.set_state_metadata_target(&target.state_metadata_target, &block.state_metadata)?;

        pw.set_hash_target(target.old_state_root, block.old_state_root)?;

        Ok(pw)
    }
}

impl BlockPreExecutionCircuit {
    /// Initializes a new block virtual targets for the given number of transactions.
    pub fn new(config: CircuitConfig) -> Self {
        let mut builder = Builder::new(config);

        Self {
            target: BlockPreExecutionTarget {
                block_number: builder.add_virtual_target(),
                created_at: builder.add_virtual_target(),

                old_system_config: SystemConfigTarget::new(&mut builder),
                register_stack_before: RegisterStackTarget::new(&mut builder),
                all_assets_before: (0..ASSET_LIST_SIZE)
                    .map(|_| AssetTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                all_margined_assets_before: (0..MARGINED_ASSET_LIST_SIZE)
                    .map(|_| MarginedAssetTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                all_market_details_before: (0..POSITION_LIST_SIZE)
                    .map(|_| MarketDetailsTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),

                price_updates: PriceUpdatesTarget::new(&mut builder),
                calculate_premium: builder.add_virtual_bool_target_safe(),
                calculate_funding: builder.add_virtual_bool_target_safe(),
                calculate_oracle_prices: builder.add_virtual_bool_target_safe(),

                old_account_tree_root: builder.add_virtual_hash(),
                old_account_pub_data_tree_root: builder.add_virtual_hash(),
                old_market_tree_root: builder.add_virtual_hash(),
                old_state_root: builder.add_virtual_hash(),

                state_metadata_target: StateMetadataTarget::new(&mut builder),

                all_market_details_after: (0..POSITION_LIST_SIZE)
                    .map(|_| MarketDetailsTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                all_margined_assets_after: (0..MARGINED_ASSET_LIST_SIZE)
                    .map(|_| MarginedAssetTarget::new(&mut builder))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                new_state_metadata_target: StateMetadataTarget::new(&mut builder),
                new_state_root: builder.add_virtual_hash(),
                new_validium_root: builder.add_virtual_hash(),

                all_assets_hash: builder.zero_hash_out(),
            },

            builder,
        }
    }

    fn register_public_inputs(&mut self) {
        // Register new state metadata targets
        self.target
            .new_state_metadata_target
            .register_public_input(&mut self.builder);

        // Register market details
        self.target
            .all_market_details_after
            .iter()
            .for_each(|market| {
                market.register_public_input(&mut self.builder);
            });

        // Register margined assets
        self.target
            .all_margined_assets_after
            .iter()
            .for_each(|asset| {
                asset.register_public_input(&mut self.builder);
            });

        // Register state roots
        self.builder
            .register_public_hashout(self.target.old_state_root);
        self.builder
            .register_public_hashout(self.target.new_state_root);
        self.builder
            .register_public_hashout(self.target.new_validium_root);

        self.builder.register_public_input(self.target.block_number);
        self.builder.register_public_input(self.target.created_at);
    }

    /// Verify consistency of old tree roots and pre-execution register state
    fn define_block_state_data_checks(&mut self) {
        let old_system_config_hash = self.target.old_system_config.hash(&mut self.builder);
        let old_register_stack_hash = self.target.register_stack_before.hash(&mut self.builder);
        let current_all_market_details = self.target.all_market_details_before.clone();
        self.target.all_assets_hash =
            all_assets_hash(&mut self.builder, &self.target.all_assets_before);
        let all_margined_assets_hash =
            all_margined_assets_hash(&mut self.builder, &self.target.all_margined_assets_before);

        let old_all_market_details_hash =
            all_market_details_hash(&mut self.builder, &current_all_market_details);
        let old_state_metadata_hash = self.target.state_metadata_target.hash(&mut self.builder);
        let old_public_market_details_hash =
            all_public_market_details_hash(&mut self.builder, &current_all_market_details);
        let old_validium_root = self.builder.hash_n_to_one(&[
            old_register_stack_hash,
            self.target.old_account_tree_root,
            self.target.old_market_tree_root,
            self.target.all_assets_hash,
            all_margined_assets_hash,
            old_all_market_details_hash,
            old_state_metadata_hash,
            old_system_config_hash,
        ]);

        let old_state_root = self.builder.hash_n_to_one(&[
            self.target.old_account_pub_data_tree_root,
            old_public_market_details_hash,
            old_validium_root,
        ]);

        self.builder
            .connect_hashes(old_state_root, self.target.old_state_root);

        // Register stack should be in Execute Mode when pre-execution is in progress
        let execute_transaction = self.builder.constant_from_u8(EXECUTE_TRANSACTION);
        let is_register_instruction_type_execute = self.builder.is_equal(
            self.target.register_stack_before[0].instruction_type,
            execute_transaction,
        );
        let is_pre_block_execution = self.builder.multi_or(&[
            self.target.calculate_funding,
            self.target.calculate_oracle_prices,
            self.target.calculate_premium,
        ]);
        self.builder
            .conditional_assert_true(is_pre_block_execution, is_register_instruction_type_execute);
    }

    /// Pre-execution, i.e. updating funding rates, oracle prices and premiums for all markets
    fn define_block_pre_execution(
        &mut self,
    ) -> (
        [MarketDetailsTarget; POSITION_LIST_SIZE],
        [MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
        StateMetadataTarget,
    ) {
        let builder = &mut self.builder;
        let zero = builder.zero();
        let one_minute = builder.constant_usize(MINUTE_IN_MS);

        builder.register_range_check(self.target.created_at, TIMESTAMP_BITS);
        builder.register_range_check(
            self.target
                .state_metadata_target
                .last_funding_round_timestamp,
            TIMESTAMP_BITS,
        );
        builder.register_range_check(
            self.target.state_metadata_target.last_premium_timestamp,
            TIMESTAMP_BITS,
        );
        builder.register_range_check(
            self.target
                .state_metadata_target
                .last_oracle_price_timestamp,
            TIMESTAMP_BITS,
        );

        let next_non_applied_funding_timestamp = round_unix_timestamp_to_next_hour(
            builder,
            self.target
                .state_metadata_target
                .last_funding_round_timestamp,
        );
        let need_funding = builder.is_gte(
            self.target.created_at,
            next_non_applied_funding_timestamp,
            TIMESTAMP_BITS,
        );
        builder.connect(need_funding.target, self.target.calculate_funding.target);

        let previous_premium_range_start = round_unix_timestamp_to_previous_minute(
            builder,
            self.target.state_metadata_target.last_premium_timestamp,
        );
        let next_premium_range_start = builder.add(previous_premium_range_start, one_minute);
        let next_premium_range_end_non_inclusive =
            builder.add(next_premium_range_start, one_minute);

        // Enforce premium update if next premium period has been missed
        let need_premium = builder.is_gte(
            self.target.created_at,
            next_premium_range_end_non_inclusive,
            TIMESTAMP_BITS,
        );
        builder.conditional_assert_true(need_premium, self.target.calculate_premium);

        // Check if new premium is not in the same minute interval
        builder.conditional_assert_lte(
            self.target.calculate_premium,
            next_premium_range_start,
            self.target.created_at,
            TIMESTAMP_BITS,
        );

        let eight = builder.constant_usize(8);
        let funding_premium_multiplier_tick = builder.constant_u64(FUNDING_PREMIUM_MULTIPLIER_TICK);
        let market_details_after = core::array::from_fn(|market_index| {
            let mut market_details = self.target.all_market_details_before[market_index].clone();

            let active_market_status = builder.constant(F::from_canonical_u8(MARKET_STATUS_ACTIVE));
            let is_market_active = builder.is_equal(market_details.status, active_market_status);

            /*********************/
            /*   Apply Funding   */
            /*********************/
            let should_apply_funding = builder.and(need_funding, is_market_active);
            let max_premium_sample_count = builder.constant_usize(MAX_PREMIUM_SAMPLE_COUNT);
            let (abs_premium_sum, sign_premium_sum) =
                builder.abs(market_details.aggregate_premium_sum);
            let (abs_avarage_premium, _) =
                builder.div_rem(abs_premium_sum, max_premium_sample_count, 6);
            // Scale the premium by the market funding premium multiplier, in (0, 100] where 100 = 1x
            let scaled_avarage_premium_abs = builder.mul(
                abs_avarage_premium,
                market_details.funding_premium_multiplier,
            );
            let (abs_avarage_premium, _) = builder.div_rem(
                scaled_avarage_premium_abs,
                funding_premium_multiplier_tick,
                FUNDING_PREMIUM_MULTIPLIER_BITS,
            );
            let avarage_premium =
                SignedTarget::new_unsafe(builder.mul(abs_avarage_premium, sign_premium_sum.target));

            // Apply small clamp
            // fundingRateWithSmallClamp = averagePremium + Clamp(int64(marketInfo.InterestRate)-averagePremium, -500, 500)
            let interest_rate_minus_average_premium = builder.sub_signed(
                SignedTarget::new_unsafe(market_details.interest_rate),
                avarage_premium,
            );
            let (interest_rate_minus_average_premium_abs, interest_rate_minus_average_premium_sign) =
                builder.abs(interest_rate_minus_average_premium);
            let interest_rate_minus_average_premium_clamped_abs = builder.min(
                &[
                    interest_rate_minus_average_premium_abs,
                    market_details.funding_clamp_small,
                ],
                FUNDING_RATE_BITS,
            );
            let interest_rate_minus_average_premium_clamped =
                SignedTarget::new_unsafe(builder.mul(
                    interest_rate_minus_average_premium_clamped_abs,
                    interest_rate_minus_average_premium_sign.target,
                ));
            let funding_with_small_clamp =
                builder.add_signed(avarage_premium, interest_rate_minus_average_premium_clamped);

            // Apply big clamp
            // fundingRate = Clamp(fundingRateWithSmallClamp, -%4, %4)
            let (funding_with_small_clamp_abs, funding_with_small_clamp_sign) =
                builder.abs(funding_with_small_clamp);
            let funding_with_big_clamp_abs = builder.min(
                // 40_000 at max
                &[
                    funding_with_small_clamp_abs,
                    market_details.funding_clamp_big,
                ],
                FUNDING_RATE_BITS,
            );

            // Get hourly rate
            let (hourly_funding_rate_abs, _) = // 5_000 at max -> 13 bits
                builder.div_rem(funding_with_big_clamp_abs, eight, 4);
            let hourly_funding_value_abs = // 13 bits * 32 bits = 45 bits = 3 u16 limbs
                builder.mul(hourly_funding_rate_abs, market_details.index_price);

            let is_abs_funding_value_zero = builder.is_zero(hourly_funding_value_abs);
            let funding_value = BigIntU16Target {
                abs: builder.target_to_biguint_u16(hourly_funding_value_abs, 3),
                sign: SignTarget::new_unsafe(builder.select(
                    is_abs_funding_value_zero,
                    zero,
                    funding_with_small_clamp_sign.target,
                )),
            };

            let new_funding_rate_prefix_sum = builder.add_bigint_u16_non_carry(
                &market_details.funding_rate_prefix_sum,
                &funding_value,
                BIGU16_U64_LIMBS,
            );
            builder.range_check_biguint_u16(
                &new_funding_rate_prefix_sum.abs,
                FUNDING_RATE_PREFIX_SUM_BITS,
            );
            market_details.funding_rate_prefix_sum = builder.select_bigint_u16(
                should_apply_funding,
                &new_funding_rate_prefix_sum,
                &market_details.funding_rate_prefix_sum,
            );

            // Reset premium sum at funding rounds
            let new_aggregate_premium_sum = builder.zero_signed();
            market_details.aggregate_premium_sum = builder.select_signed(
                should_apply_funding,
                new_aggregate_premium_sum,
                market_details.aggregate_premium_sum,
            );

            /****************************/
            /*   Update Oracle Prices   */
            /****************************/
            let should_update_oracle_price =
                builder.and(self.target.calculate_oracle_prices, is_market_active);

            builder.register_range_check(
                self.target.price_updates.index_price[market_index],
                ORDER_PRICE_BITS,
            );
            builder.register_range_check(
                self.target.price_updates.mark_price[market_index],
                ORDER_PRICE_BITS,
            );

            let max_mark_price = builder.constant_u64(MAX_ORDER_PRICE);
            let max_market_open_interest_notional =
                builder.constant_u64(MARKET_OPEN_INTEREST_NOTIONAL);
            let (max_open_interest_mark_price, _) = builder.div_rem(
                max_market_open_interest_notional,
                market_details.open_interest,
                MARKET_OPEN_INTEREST_BITS,
            );
            let (max_open_interest_mark_price, _) = builder.div_rem(
                max_open_interest_mark_price,
                market_details.quote_multiplier,
                QUOTE_MULTIPLIER_BITS,
            );

            let is_open_interest_non_zero = builder.is_not_zero(market_details.open_interest);
            let max_open_interest_mark_price = builder.min(
                &[max_open_interest_mark_price, max_mark_price],
                MARKET_OPEN_INTEREST_NOTIONAL_BITS,
            );
            let max_mark_price = builder.select(
                is_open_interest_non_zero,
                max_open_interest_mark_price,
                max_mark_price,
            );

            builder.conditional_assert_lte(
                should_update_oracle_price,
                self.target.price_updates.mark_price[market_index],
                max_mark_price,
                ORDER_PRICE_BITS,
            );

            builder.conditional_assert_not_eq(
                should_update_oracle_price,
                self.target.price_updates.index_price[market_index],
                zero,
            );
            builder.conditional_assert_not_eq(
                should_update_oracle_price,
                self.target.price_updates.mark_price[market_index],
                zero,
            );

            market_details.index_price = builder.select(
                should_update_oracle_price,
                self.target.price_updates.index_price[market_index],
                market_details.index_price,
            );
            market_details.mark_price = builder.select(
                should_update_oracle_price,
                self.target.price_updates.mark_price[market_index],
                market_details.mark_price,
            );

            /**************************/
            /*   Calculate Premiums   */
            /**************************/
            let should_calculate_premium =
                builder.and(self.target.calculate_premium, is_market_active);

            let is_impact_ask_price_zero = builder.is_zero(market_details.impact_ask_price);
            let impact_ask_price = builder.select(
                is_impact_ask_price_zero,
                market_details.index_price,
                market_details.impact_ask_price,
            );

            let is_impact_bid_price_zero = builder.is_zero(market_details.impact_bid_price);
            let impact_bid_price = builder.select(
                is_impact_bid_price_zero,
                market_details.index_price,
                market_details.impact_bid_price,
            );

            // premium = [max(0, impactBidPrice - indexPrice) - max(0, indexPrice - impactAskPrice)] / indexPrice
            let impact_bid_price_gt_index_price = builder.is_gt(
                impact_bid_price,
                market_details.index_price,
                ORDER_PRICE_BITS,
            );
            let impact_bid_price_minus_index_price =
                builder.sub(impact_bid_price, market_details.index_price);
            let bid_component = builder.select(
                impact_bid_price_gt_index_price,
                impact_bid_price_minus_index_price,
                zero,
            );

            let index_price_gt_impact_ask_price = builder.is_gt(
                market_details.index_price,
                impact_ask_price,
                ORDER_PRICE_BITS,
            );
            let index_price_minus_impact_ask_price =
                builder.sub(market_details.index_price, impact_ask_price);
            let ask_component = builder.select(
                index_price_gt_impact_ask_price,
                index_price_minus_impact_ask_price,
                zero,
            );

            let bid_ask_diff = SignedTarget::new_unsafe(builder.sub(bid_component, ask_component));
            let (abs_bid_ask_diff, sign_bid_ask_diff) = builder.abs(bid_ask_diff);
            let funding_tick = builder.constant_u64(FUNDING_RATE_TICK as u64);
            let normalized_abs_bid_ask_diff = builder.mul(abs_bid_ask_diff, funding_tick);
            let (abs_premium, _) = builder.div_rem(
                normalized_abs_bid_ask_diff,
                market_details.index_price,
                ORDER_PRICE_BITS,
            );
            let premium =
                SignedTarget::new_unsafe(builder.mul(abs_premium, sign_bid_ask_diff.target));
            let new_aggregate_premium_sum =
                builder.add_signed(market_details.aggregate_premium_sum, premium);

            market_details.aggregate_premium_sum = builder.select_signed(
                should_calculate_premium,
                new_aggregate_premium_sum,
                market_details.aggregate_premium_sum,
            );

            market_details
        });

        let margined_assets_after = core::array::from_fn(|asset_index| {
            builder.register_range_check(
                self.target.price_updates.asset_index_price[asset_index],
                ASSET_PRICE_BITS,
            );

            let mut margined_asset = self.target.all_margined_assets_before[asset_index].clone();

            let should_update_asset_index_price = self.target.calculate_oracle_prices;
            margined_asset.index_price = builder.select(
                should_update_asset_index_price,
                self.target.price_updates.asset_index_price[asset_index],
                margined_asset.index_price,
            );

            margined_asset
        });

        let mut new_state_metadata = self.target.state_metadata_target.clone();

        new_state_metadata.last_funding_round_timestamp = builder.select(
            self.target.calculate_funding,
            self.target.created_at,
            self.target
                .state_metadata_target
                .last_funding_round_timestamp,
        );

        new_state_metadata.last_oracle_price_timestamp = builder.select(
            self.target.calculate_oracle_prices,
            self.target.created_at,
            self.target
                .state_metadata_target
                .last_oracle_price_timestamp,
        );

        new_state_metadata.last_premium_timestamp = builder.select(
            self.target.calculate_premium,
            self.target.created_at,
            self.target.state_metadata_target.last_premium_timestamp,
        );

        (
            market_details_after,
            margined_assets_after,
            new_state_metadata,
        )
    }

    fn define_post_block_pre_execution(
        &mut self,
        all_market_details_after: &[MarketDetailsTarget; POSITION_LIST_SIZE],
        all_margined_assets_after: &[MarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
        new_state_metadata: &StateMetadataTarget,
    ) {
        let old_system_config_hash = self.target.old_system_config.hash(&mut self.builder);
        let old_register_stack_hash = self.target.register_stack_before.hash(&mut self.builder);
        let new_all_market_details_hash =
            all_market_details_hash(&mut self.builder, all_market_details_after);
        let new_state_metadata_hash = new_state_metadata.hash(&mut self.builder);
        let new_all_public_market_details_hash =
            all_public_market_details_hash(&mut self.builder, all_market_details_after);
        let new_all_margined_assets_hash =
            all_margined_assets_hash(&mut self.builder, all_margined_assets_after);
        let new_validium_root = self.builder.hash_n_to_one(&[
            old_register_stack_hash,
            self.target.old_account_tree_root,
            self.target.old_market_tree_root,
            self.target.all_assets_hash,
            new_all_margined_assets_hash,
            new_all_market_details_hash,
            new_state_metadata_hash,
            old_system_config_hash,
        ]);

        self.builder
            .connect_hashes(new_validium_root, self.target.new_validium_root);

        let new_state_root = self.builder.hash_n_to_one(&[
            self.target.old_account_pub_data_tree_root,
            new_all_public_market_details_hash,
            new_validium_root,
        ]);

        self.builder
            .connect_hashes(new_state_root, self.target.new_state_root);

        // Connect market details and state metadata to public inputs
        connect_state_metadata_target(
            &mut self.builder,
            new_state_metadata,
            &self.target.new_state_metadata_target,
        );

        all_market_details_after
            .iter()
            .zip_eq(self.target.all_market_details_after.iter())
            .for_each(|(x, y)| {
                connect_market_details(&mut self.builder, x, y);
            });

        all_margined_assets_after
            .iter()
            .zip_eq(self.target.all_margined_assets_after.iter())
            .for_each(|(x, y)| {
                connect_margined_assets(&mut self.builder, x, y);
            });
    }
}
