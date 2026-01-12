// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use anyhow::Result;
use num::{BigInt, BigUint};
use plonky2::field::extension::Extendable;
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget, WitnessBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bigint::unsafe_big::{CircuitBuilderUnsafeBig, UnsafeBigTarget};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::types::account_position::{
    AccountPosition, AccountPositionTarget, AccountPositionTargetWitness,
};
use crate::types::config::{BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U160_LIMBS, Builder, *};
use crate::types::constants::{POSITION_LIST_SIZE, TIMESTAMP_BITS, *};
use crate::types::market_details::MarketDetailsTarget;
use crate::types::public_pool::{
    PublicPoolInfo, PublicPoolInfoTarget, PublicPoolInfoWitness, PublicPoolShare,
    PublicPoolShareTarget, PublicPoolShareWitness, select_public_pool_share_target,
};
use crate::types::risk_info::position_base_notional;
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "", default)]
pub struct Account<F>
where
    F: RichField + Extendable<5>,
{
    #[serde(rename = "mai", default)]
    pub master_account_index: i64,

    #[serde(rename = "ai", default)]
    pub account_index: i64,

    #[serde(rename = "l1")]
    #[serde(deserialize_with = "deserializers::l1_address_to_biguint")]
    pub l1_address: BigUint, // 160 bits

    #[serde(rename = "at")]
    pub account_type: u8,

    #[serde(rename = "col")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub collateral: BigInt, // 96 bits

    #[serde(rename = "ab")]
    #[serde(deserialize_with = "deserializers::aggregated_balances")]
    pub aggregated_balances: [BigInt; NB_ASSETS_PER_TX], // 96 bits

    #[serde(rename = "ap")]
    #[serde(deserialize_with = "deserializers::positions")]
    pub positions: [AccountPosition; POSITION_LIST_SIZE],

    #[serde(rename = "pps", default)]
    pub public_pool_shares: [PublicPoolShare; SHARES_LIST_SIZE],

    #[serde(rename = "ppi")]
    pub public_pool_info: PublicPoolInfo,

    #[serde(rename = "toc", default)]
    pub total_order_count: i64,

    #[serde(rename = "tioc", default)]
    pub total_non_cross_order_count: i64,

    #[serde(rename = "cat", default)]
    pub cancel_all_time: i64,

    #[serde(rename = "akr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub api_key_root: HashOut<F>,

    #[serde(rename = "aor")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub account_orders_root: HashOut<F>,

    #[serde(rename = "abr")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub aggregated_balances_root: HashOut<F>,

    #[serde(rename = "asr", default)]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub asset_root: HashOut<F>,

    #[serde(rename = "ph", default)]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub partial_hash: HashOut<F>,

    #[serde(rename = "phpd", default)]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub partial_hash_for_pub_data: HashOut<F>,
}

impl<F> Default for Account<F>
where
    F: RichField + Extendable<5>,
{
    fn default() -> Self {
        Self {
            master_account_index: NIL_MASTER_ACCOUNT_INDEX,
            account_index: 0,
            l1_address: BigUint::ZERO,
            account_type: 0,
            collateral: BigInt::ZERO,
            aggregated_balances: [BigInt::ZERO; NB_ASSETS_PER_TX],
            positions: array::from_fn(|_| AccountPosition::default()),
            public_pool_shares: array::from_fn(|_| PublicPoolShare::default()),
            public_pool_info: PublicPoolInfo::default(),
            total_order_count: 0,
            total_non_cross_order_count: 0,
            cancel_all_time: 0,
            api_key_root: HashOut::ZERO,
            account_orders_root: HashOut::ZERO,
            aggregated_balances_root: HashOut::ZERO,
            asset_root: HashOut::ZERO,

            partial_hash: HashOut::ZERO,
            partial_hash_for_pub_data: HashOut::ZERO,
        }
    }
}
#[derive(Debug, Clone)]
pub struct AccountTarget {
    pub master_account_index: Target,
    pub account_index: Target,
    pub l1_address: BigUintTarget,
    pub account_type: Target,

    pub collateral: BigIntTarget,
    pub aggregated_balances: [BigIntTarget; NB_ASSETS_PER_TX],
    pub positions: [AccountPositionTarget; POSITION_LIST_SIZE],

    pub public_pool_shares: [PublicPoolShareTarget; SHARES_LIST_SIZE],
    pub public_pool_info: PublicPoolInfoTarget,

    pub total_order_count: Target,
    pub total_non_cross_order_count: Target, // includes isolated orders and spot orders
    pub cancel_all_time: Target,

    pub api_key_root: HashOutTarget,
    pub account_orders_root: HashOutTarget,
    pub aggregated_balances_root: HashOutTarget,
    pub asset_root: HashOutTarget,

    pub partial_hash: HashOutTarget,
    pub partial_hash_for_pub_data: HashOutTarget,
}

impl Default for AccountTarget {
    fn default() -> Self {
        AccountTarget {
            master_account_index: Target::default(),
            account_index: Target::default(),
            l1_address: BigUintTarget::default(),
            account_type: Target::default(),

            collateral: BigIntTarget::default(),
            aggregated_balances: array::from_fn(|_| BigIntTarget::default()),

            positions: array::from_fn(|_| AccountPositionTarget::default()),

            public_pool_shares: array::from_fn(|_| PublicPoolShareTarget::default()),
            public_pool_info: PublicPoolInfoTarget::default(),

            total_order_count: Target::default(),
            total_non_cross_order_count: Target::default(),
            cancel_all_time: Target::default(),

            api_key_root: HashOutTarget::from([Target::default(); NUM_HASH_OUT_ELTS]),
            account_orders_root: HashOutTarget::from([Target::default(); NUM_HASH_OUT_ELTS]),
            aggregated_balances_root: HashOutTarget::from([Target::default(); NUM_HASH_OUT_ELTS]),
            asset_root: HashOutTarget::from([Target::default(); NUM_HASH_OUT_ELTS]),

            partial_hash: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
            partial_hash_for_pub_data: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
        }
    }
}

impl AccountTarget {
    pub fn new(builder: &mut Builder) -> Self {
        AccountTarget {
            master_account_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),

            collateral: builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS), // safe because it is read from the state using merkle proofs
            aggregated_balances: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),

            positions: array::from_fn(|_| AccountPositionTarget::new(builder)),

            public_pool_shares: array::from_fn(|_| PublicPoolShareTarget::new(builder)),
            public_pool_info: PublicPoolInfoTarget::new(builder),

            total_order_count: builder.add_virtual_target(),
            total_non_cross_order_count: builder.add_virtual_target(),
            cancel_all_time: builder.add_virtual_target(),

            api_key_root: builder.add_virtual_hash(),
            account_orders_root: builder.add_virtual_hash(),
            aggregated_balances_root: builder.add_virtual_hash(),
            asset_root: builder.add_virtual_hash(),

            partial_hash: builder.zero_hash_out(), // Unused for maker and taker accounts
            partial_hash_for_pub_data: builder.zero_hash_out(), // Unused for maker and taker accounts
        }
    }

    pub fn new_fee_account(builder: &mut Builder) -> Self {
        AccountTarget {
            master_account_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),

            collateral: builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS), // safe because it is read from the state using merkle proofs
            aggregated_balances: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),

            positions: array::from_fn(|_| AccountPositionTarget::default()), // Unused for fee accounts
            public_pool_shares: array::from_fn(|_| PublicPoolShareTarget::default()),
            public_pool_info: PublicPoolInfoTarget::default(),

            total_order_count: builder.add_virtual_target(),
            total_non_cross_order_count: builder.add_virtual_target(),
            cancel_all_time: builder.add_virtual_target(),

            api_key_root: builder.add_virtual_hash(),
            account_orders_root: builder.add_virtual_hash(),
            aggregated_balances_root: builder.add_virtual_hash(),
            asset_root: builder.add_virtual_hash(),

            partial_hash: builder.add_virtual_hash(), // Hash of positions, public pool shares, and public pool info
            partial_hash_for_pub_data: builder.add_virtual_hash(), // Hash of position, public pool shares, and public pool info pub data
        }
    }

    pub fn should_dms_be_triggered(
        &self,
        builder: &mut Builder,
        block_created_at: Target,
    ) -> BoolTarget {
        let is_cancel_all_time_not_zero = builder.is_not_zero(self.cancel_all_time);
        let is_cancel_all_time_lte_block_created_at =
            builder.is_lte(self.cancel_all_time, block_created_at, TIMESTAMP_BITS);

        builder.multi_and(&[
            is_cancel_all_time_not_zero,
            is_cancel_all_time_lte_block_created_at,
        ])
    }

    pub fn get_cross_position_base_notional_values(
        &self,
        builder: &mut Builder,
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> ([BigUintTarget; POSITION_LIST_SIZE], BigIntTarget) {
        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);

        let mut base_position_notional_values = core::array::from_fn(|_| builder.zero_biguint());

        let mut cross_positive_tpv_sum = builder.zero();
        let mut cross_negative_tpv_sum = builder.zero();

        for market_index in 0..POSITION_LIST_SIZE {
            let position = &self.positions[market_index];
            let market_details = &all_market_details[market_index];

            let is_isolated_position = builder.is_equal(position.margin_mode, isolated_margin_mode);
            let is_cross_position = builder.not(is_isolated_position);

            let (abs_position_notional, positive_tpv_component, negative_tpv_component) =
                position_base_notional(builder, position, market_details);

            // Accumulate cross margins
            cross_positive_tpv_sum = builder.mul_add(
                is_cross_position.target,
                positive_tpv_component,
                cross_positive_tpv_sum,
            );
            cross_negative_tpv_sum = builder.mul_add(
                is_cross_position.target,
                negative_tpv_component,
                cross_negative_tpv_sum,
            );

            base_position_notional_values[market_index] =
                builder.target_to_biguint(abs_position_notional);
        }
        // compute total position notional value from the positive and negative components
        let zero = builder.zero();
        let one = builder.one();

        let cross_position_notional_value = {
            let is_positive_tpv_sum_zero = builder.is_zero(cross_positive_tpv_sum);
            let add_sign = builder.select(is_positive_tpv_sum_zero, zero, one);
            let big_positive_tpv_sum = BigIntTarget {
                abs: builder.target_to_biguint(cross_positive_tpv_sum),
                sign: SignTarget::new_unsafe(add_sign),
            };

            let is_negative_tpv_sum_zero = builder.is_zero(cross_negative_tpv_sum);
            let add_sign = builder.select(is_negative_tpv_sum_zero, zero, one);
            let big_negative_tpv_sum = BigIntTarget {
                abs: builder.target_to_biguint(cross_negative_tpv_sum),
                sign: SignTarget::new_unsafe(add_sign),
            };
            builder.sub_bigint_non_carry(
                &big_positive_tpv_sum,
                &big_negative_tpv_sum,
                BIG_U96_LIMBS,
            )
        };

        (base_position_notional_values, cross_position_notional_value)
    }

    pub fn get_cross_unrealized_funding(
        &self,
        builder: &mut Builder,
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> BigIntTarget {
        let mut unsafe_unrealized_funding = UnsafeBigTarget {
            limbs: vec![builder.zero(); BIGU16_U112_LIMBS],
        };
        for market_index in 0..POSITION_LIST_SIZE {
            let market_details = all_market_details[market_index].clone();
            let position = self.positions[market_index].clone();

            let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);
            let is_isolated_position = builder.is_equal(position.margin_mode, isolated_margin_mode);
            let is_cross_position = builder.not(is_isolated_position);

            let lhs = builder.sub_bigint_u16_unsafe(
                &position.last_funding_rate_prefix_sum,
                &market_details.funding_rate_prefix_sum,
            ); // (-2^17, 2^17)

            let rhs = builder.mul_bigint_u16_and_target_unsafe(
                &position.position,
                market_details.quote_multiplier,
            ); // (-2^30, 2^30)

            // Multiply the two unsafe bigints, where lhs and rhs each has 4 limbs.
            // Limbwise multiplication is in (-2^47, 2^47) range.
            // Resulting limbs will be at most sum of 4 different limbwise multiplications.
            // Thus resulting limbs are in the range of (-2^49, 2^49).
            let unsafe_position_unrealized_funding =
                builder.mul_unsafe_big(&lhs, &rhs, BIGU16_U112_LIMBS); // (-2^49, 2^49)

            // Accumulate the unrealized funding for at most 255 (2^8 - 1) cross positions
            unsafe_unrealized_funding = builder.mul_add_unsafe_big(
                &unsafe_position_unrealized_funding,
                is_cross_position.target,
                &unsafe_unrealized_funding,
            ); // (-2^57, 2^57)
        }
        let unrealized_funding =
            builder.unsafe_big16_to_bigint(&unsafe_unrealized_funding, BIGU16_U112_LIMBS);
        BigIntTarget {
            abs: builder.trim_biguint(&unrealized_funding.abs, BIG_U96_LIMBS),
            sign: unrealized_funding.sign,
        }
    }

    pub fn get_initial_margin_requirement(
        &self,
        builder: &mut Builder,
        position_notional_values: &[BigUintTarget; POSITION_LIST_SIZE],
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> BigUintTarget {
        let margin_fraction_multiplier =
            builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));
        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);

        let mut cross_value = UnsafeBigTarget {
            limbs: vec![builder.zero(); BIG_U64_LIMBS],
        };

        for market_index in 0..POSITION_LIST_SIZE {
            let position = self.positions[market_index].clone();
            let is_isolated_position = builder.is_equal(position.margin_mode, isolated_margin_mode);
            let is_cross_position = builder.not(is_isolated_position);
            let margin_fraction = position.get_initial_margin_fraction(
                builder,
                all_market_details[market_index].default_initial_margin_fraction,
                all_market_details[market_index].min_initial_margin_fraction,
            );
            let lhs = builder.unsafe_big_from_biguint(&position_notional_values[market_index]); // each limb 32 bit
            let rhs = builder.mul(margin_fraction, is_cross_position.target); // 14 bits
            cross_value = builder.mul_add_unsafe_big(&lhs, rhs, &cross_value); // each limb 46 bit + accumulating at most 255 markets = each limb 54 bit
        }
        let cross_value = builder.unsafe_big32_to_biguint(&cross_value, BIG_U96_LIMBS);

        builder.mul_biguint_non_carry(&cross_value, &margin_fraction_multiplier, BIG_U96_LIMBS)
    }

    pub fn get_maintenance_margin_requirement(
        &self,
        builder: &mut Builder,
        position_notional_values: &[BigUintTarget; POSITION_LIST_SIZE],
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> BigUintTarget {
        let margin_fraction_multiplier =
            builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));
        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);

        let mut cross_value = UnsafeBigTarget {
            limbs: vec![builder.zero(); BIG_U64_LIMBS],
        };

        for market_index in 0..POSITION_LIST_SIZE {
            let position = self.positions[market_index].clone();
            let is_isolated_position = builder.is_equal(position.margin_mode, isolated_margin_mode);
            let is_cross_position = builder.not(is_isolated_position);
            let lhs = builder.unsafe_big_from_biguint(&position_notional_values[market_index]); // each limb 32 bit
            let rhs = builder.mul(
                all_market_details[market_index].maintenance_margin_fraction,
                is_cross_position.target,
            ); // 14 bits
            cross_value = builder.mul_add_unsafe_big(&lhs, rhs, &cross_value); // each limb 46 bit + accumulating at most 255 markets = each limb 54 bit
        }
        // Sum of cross_values where each cross_value is 42 bits and total 2^8 markets, so each limb is 50 bit
        let cross_value = builder.unsafe_big32_to_biguint(&cross_value, BIG_U96_LIMBS);

        builder.mul_biguint_non_carry(&cross_value, &margin_fraction_multiplier, BIG_U96_LIMBS)
    }

    pub fn get_close_out_margin_requirement(
        &self,
        builder: &mut Builder,
        position_notional_values: &[BigUintTarget; POSITION_LIST_SIZE],
        all_market_details: &[MarketDetailsTarget; POSITION_LIST_SIZE],
    ) -> BigUintTarget {
        let margin_fraction_multiplier =
            builder.constant_biguint(&BigUint::from(MARGIN_FRACTION_MULTIPLIER));
        let isolated_margin_mode = builder.constant_usize(ISOLATED_MARGIN);

        let mut cross_value = UnsafeBigTarget {
            limbs: vec![builder.zero(); BIG_U64_LIMBS],
        };

        for market_index in 0..POSITION_LIST_SIZE {
            let position = self.positions[market_index].clone();
            let is_isolated_position = builder.is_equal(position.margin_mode, isolated_margin_mode);
            let is_cross_position = builder.not(is_isolated_position);
            let lhs = builder.unsafe_big_from_biguint(&position_notional_values[market_index]); // each limb 32 bit
            let rhs = builder.mul(
                all_market_details[market_index].close_out_margin_fraction,
                is_cross_position.target,
            ); // 14 bits
            cross_value = builder.mul_add_unsafe_big(&lhs, rhs, &cross_value); // each limb 46 bit + accumulating at most 255 markets = each limb 54 bit
        }
        // Sum of cross_values where each cross_value is 42 bits and total 2^8 markets, so each limb is 50 bit
        let cross_value = builder.unsafe_big32_to_biguint(&cross_value, BIG_U96_LIMBS);

        builder.mul_biguint_non_carry(&cross_value, &margin_fraction_multiplier, BIG_U96_LIMBS)
    }

    pub fn apply_collateral_delta(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        collateral_delta: BigIntTarget,
    ) {
        let new_collateral =
            builder.add_bigint_non_carry(&self.collateral, &collateral_delta, BIG_U96_LIMBS);
        self.collateral = builder.select_bigint(is_enabled, &new_collateral, &self.collateral);
    }

    pub fn get_public_pool_share(
        &self,
        builder: &mut Builder,
        public_pool_index: Target,
    ) -> PublicPoolShareTarget {
        let mut res = PublicPoolShareTarget::empty(builder, public_pool_index);

        // Try to find the pool share that matches the pool index, replace if found
        for i in 0..SHARES_LIST_SIZE {
            let is_pool_index_equal = builder.is_equal(
                self.public_pool_shares[i].public_pool_index,
                public_pool_index,
            );
            res = select_public_pool_share_target(
                builder,
                is_pool_index_equal,
                &self.public_pool_shares[i],
                &res,
            );
        }
        res
    }

    pub fn mint_pool_shares(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        pool_index: Target,
        share_delta: Target,
        entry_usdc_delta: Target,
    ) {
        let zero = builder.zero();

        // Clone old values to new variables
        let mut new_pool_shares = self.public_pool_shares;

        let mut success = builder._false();
        let mut set_new_entry_usdc = builder.zero();

        let mut new_share_amounts = vec![];
        let mut new_entry_usdcs = vec![];

        // Try to find the pool share that matches the pool index
        // If found, update the share amount and entry usdc (new_pool_shares)
        for i in 0..SHARES_LIST_SIZE {
            let is_pool_index_equal =
                builder.is_equal(new_pool_shares[i].public_pool_index, pool_index);
            let is_enabled_and_matching = builder.and(is_enabled, is_pool_index_equal);
            let update = builder.and_not(is_enabled_and_matching, success);

            let new_share_amount = builder.add(new_pool_shares[i].share_amount, share_delta);
            new_share_amounts.push(new_share_amount);
            let new_entry_usdc = builder.add(new_pool_shares[i].principal_amount, entry_usdc_delta);
            new_entry_usdcs.push(new_entry_usdc);

            new_pool_shares[i].share_amount =
                builder.select(update, new_share_amount, new_pool_shares[i].share_amount);
            new_pool_shares[i].principal_amount =
                builder.select(update, new_entry_usdc, new_pool_shares[i].principal_amount);

            set_new_entry_usdc = builder.select(update, new_entry_usdc, set_new_entry_usdc);

            success = builder.or(success, update);
        }

        // If not found, try to find an empty pool share
        for i in 0..SHARES_LIST_SIZE {
            let is_pool_share_empty = builder.is_zero(new_pool_shares[i].share_amount);
            let is_enabled_and_empty = builder.and(is_enabled, is_pool_share_empty);
            let update = builder.and_not(is_enabled_and_empty, success);

            new_pool_shares[i].share_amount = builder.select(
                update,
                new_share_amounts[i],
                new_pool_shares[i].share_amount,
            );
            new_pool_shares[i].principal_amount = builder.select(
                update,
                new_entry_usdcs[i],
                new_pool_shares[i].principal_amount,
            );
            new_pool_shares[i].public_pool_index =
                builder.select(update, pool_index, new_pool_shares[i].public_pool_index);

            set_new_entry_usdc = builder.select(
                update,
                new_pool_shares[i].principal_amount,
                set_new_entry_usdc,
            );

            success = builder.or(success, update);
        }

        // Fix the empty hole that might have been created in new_pool_shares, when a pool share is empty
        // start overriding with the the next pool share
        let mut use_next = builder._false();
        let empty_pps = PublicPoolShareTarget::empty(builder, zero);
        for i in 0..SHARES_LIST_SIZE {
            let next_pool_share = if i < SHARES_LIST_SIZE - 1 {
                new_pool_shares[i + 1]
            } else {
                empty_pps
            };
            let is_pool_share_empty = builder.is_zero(new_pool_shares[i].share_amount);
            use_next = builder.or(use_next, is_pool_share_empty);
            new_pool_shares[i] = select_public_pool_share_target(
                builder,
                use_next,
                &next_pool_share,
                &new_pool_shares[i],
            );
        }

        // check if set_new_entry_usdc is less than the maximum allowed entry usdc
        let max_entry_usdc = builder.constant_u64(MAX_POOL_PRINCIPAL_AMOUNT);
        let valid_entry_usdc = builder.is_lte(set_new_entry_usdc, max_entry_usdc, 64);
        success = builder.and(success, valid_entry_usdc);
        let update_state = builder.and(success, is_enabled);
        for i in 0..SHARES_LIST_SIZE {
            self.public_pool_shares[i] = select_public_pool_share_target(
                builder,
                update_state,
                &new_pool_shares[i],
                &self.public_pool_shares[i],
            );
        }
        builder.conditional_assert_true(is_enabled, success);
    }

    // Check account has corresponding pool index before calling this.
    pub fn burn_pool_shares(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        pool_index: Target,
        share_delta: Target,
        entry_usdc_delta: Target,
    ) {
        let zero = builder.zero();

        // Find the pool share that matches the pool index, update the share amount and entry usdc
        let mut success = builder._false();
        for i in 0..SHARES_LIST_SIZE {
            let is_pool_index_equal =
                builder.is_equal(self.public_pool_shares[i].public_pool_index, pool_index);
            let is_enabled_and_matching = builder.and(is_enabled, is_pool_index_equal);
            let update = builder.and_not(is_enabled_and_matching, success);

            let new_share_amount =
                builder.sub(self.public_pool_shares[i].share_amount, share_delta);
            let new_entry_usdc = builder.sub(
                self.public_pool_shares[i].principal_amount,
                entry_usdc_delta,
            );

            self.public_pool_shares[i].share_amount = builder.select(
                update,
                new_share_amount,
                self.public_pool_shares[i].share_amount,
            );
            self.public_pool_shares[i].principal_amount = builder.select(
                update,
                new_entry_usdc,
                self.public_pool_shares[i].principal_amount,
            );

            success = builder.or(success, update);
        }

        // Fix the empty hole that might have been created in pool shares, should current share becomes 0 after burning
        let mut use_next = builder._false();
        let empty_pps = PublicPoolShareTarget::empty(builder, zero);
        for i in 0..SHARES_LIST_SIZE {
            let next_pool_share = if i < SHARES_LIST_SIZE - 1 {
                self.public_pool_shares[i + 1]
            } else {
                empty_pps
            };
            let is_pool_share_empty = builder.is_zero(self.public_pool_shares[i].share_amount);
            use_next = builder.or(use_next, is_pool_share_empty);
            self.public_pool_shares[i] = select_public_pool_share_target(
                builder,
                use_next,
                &next_pool_share,
                &self.public_pool_shares[i],
            );
        }
    }

    pub fn apply_pool_share_delta(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        pool_index: Target,
        share_delta: Target,      // Can be negative for burns
        entry_usdc_delta: Target, // Can be negative for burns
    ) {
        let zero = builder.zero();
        let old_pool_shares = self.public_pool_shares;

        let mut applied = builder._false();
        let mut use_next = builder._false();
        let mut use_prev = builder._false();

        let new_pool_shares_for_empty = PublicPoolShareTarget {
            public_pool_index: pool_index,
            share_amount: share_delta,
            principal_amount: entry_usdc_delta,
        };
        let empty_pool_share = PublicPoolShareTarget::empty(builder, zero);
        let is_share_delta_non_zero = builder.is_not_zero(share_delta);
        let is_enabled = builder.and(is_enabled, is_share_delta_non_zero);
        for i in 0..SHARES_LIST_SIZE {
            // Empty case is straightforward, just insert the new pool share.
            // Pool shares list is sorted by pool index, so we may need to insert the delta in between two
            // existing slots. For that case, we stop when we find the first pool index that is greater than
            // the target pool index, and insert the new pool share there. We toggle use_prev to true, which
            // ensures the following iterations to just shift the old pool shares right by one slot.
            // We also toggle it when current slot is empty, but that's a no-op.
            let is_pool_index_gt = builder.is_gt(
                self.public_pool_shares[i].public_pool_index,
                pool_index,
                ACCOUNT_INDEX_BITS,
            );
            let is_pool_share_slot_empty = builder.is_zero(self.public_pool_shares[i].share_amount);
            let empty_or_insert = builder.or(is_pool_share_slot_empty, is_pool_index_gt);
            let empty_or_insert_and_not_applied = builder.and_not(empty_or_insert, applied);
            let apply_delta = builder.and(empty_or_insert_and_not_applied, is_enabled);
            applied = builder.or(applied, apply_delta);

            self.public_pool_shares[i] = select_public_pool_share_target(
                builder,
                apply_delta,
                &new_pool_shares_for_empty,
                &self.public_pool_shares[i],
            );

            self.public_pool_shares[i] = select_public_pool_share_target(
                builder,
                use_prev,
                &if i > 0 {
                    old_pool_shares[i - 1]
                } else {
                    empty_pool_share
                },
                &self.public_pool_shares[i],
            );
            use_prev = builder.or(apply_delta, use_prev);

            // The final case is updating an existing pool share. This can leave the current slot empty for
            // burning cases, and we handle them by toggling use_next to true, which ensures the current and
            // the following iterations to just shift the old pool shares left by one slot.
            let is_pool_index_eq =
                builder.is_equal(self.public_pool_shares[i].public_pool_index, pool_index);
            let is_pool_index_eq_and_not_applied = builder.and_not(is_pool_index_eq, applied);
            let apply_delta = builder.and(is_pool_index_eq_and_not_applied, is_enabled);
            applied = builder.or(applied, apply_delta);

            let add_to_share_amount = builder.mul_bool(apply_delta, share_delta);
            let add_to_entry_usdc = builder.mul_bool(apply_delta, entry_usdc_delta);
            self.public_pool_shares[i].share_amount =
                builder.add(self.public_pool_shares[i].share_amount, add_to_share_amount);
            self.public_pool_shares[i].principal_amount = builder.add(
                self.public_pool_shares[i].principal_amount,
                add_to_entry_usdc,
            );

            let is_new_share_amount_empty =
                builder.is_zero(self.public_pool_shares[i].share_amount);
            use_next = builder.select_bool(apply_delta, is_new_share_amount_empty, use_next);
            self.public_pool_shares[i] = select_public_pool_share_target(
                builder,
                use_next,
                &if i < SHARES_LIST_SIZE - 1 {
                    old_pool_shares[i + 1]
                } else {
                    empty_pool_share
                },
                &self.public_pool_shares[i],
            );
        }

        let last_pool_share_before_non_empty =
            builder.is_not_zero(old_pool_shares[SHARES_LIST_SIZE - 1].share_amount);
        let not_enough_slots = builder.and(last_pool_share_before_non_empty, use_prev);
        builder.conditional_assert_false(is_enabled, not_enough_slots);

        builder.conditional_assert_true(is_enabled, applied);
    }

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.master_account_index,
            &format!("{}: master_account_index", tag),
        );
        builder.println(self.account_index, &format!("{}: account_index", tag));
        builder.println_biguint(&self.l1_address, &format!("{}: l1_address", tag));
        builder.println(self.account_type, &format!("{}: account_type", tag));
        builder.println_bigint(&self.collateral, &format!("{}: collateral", tag));
        builder.println_hash_out(
            &self.aggregated_balances_root,
            &format!("{}: aggregated_balances_root", tag),
        );
        builder.println_hash_out(&self.asset_root, &format!("{}: asset_root", tag));

        builder.println(
            self.total_order_count,
            &format!("{}: total_order_count", tag),
        );
        builder.println(
            self.total_non_cross_order_count,
            &format!("{}: total_non_cross_order_count", tag),
        );
        builder.println(self.cancel_all_time, &format!("{}: cancel_all_time", tag));
        builder.println_hash_out(&self.api_key_root, &format!("{}: api_key_root", tag));
        builder.println_hash_out(
            &self.account_orders_root,
            &format!("{}: account_orders_root", tag),
        );
    }
}

pub trait AccountTargetWitness<F: PrimeField64 + Extendable<5> + RichField> {
    fn set_account_target(&mut self, a: &AccountTarget, b: &Account<F>) -> Result<()>;
    fn set_fee_account_target(&mut self, a: &AccountTarget, b: &Account<F>) -> Result<()>;

    fn _set_common_targets(&mut self, a: &AccountTarget, b: &Account<F>) -> Result<()>;
}

impl<T: Witness<F> + PartialWitnessCurve<F>, F: PrimeField64 + Extendable<5> + RichField>
    AccountTargetWitness<F> for T
{
    fn set_account_target(&mut self, a: &AccountTarget, b: &Account<F>) -> Result<()> {
        self._set_common_targets(a, b)?;

        for i in 0..POSITION_LIST_SIZE {
            self.set_position_target(&a.positions[i], &b.positions[i])?;
        }
        self.set_public_pool_info(&a.public_pool_info, &b.public_pool_info)?;
        for i in 0..b.public_pool_shares.len() {
            self.set_public_pool_share(&a.public_pool_shares[i], &b.public_pool_shares[i])?;
        }

        Ok(())
    }

    fn set_fee_account_target(&mut self, a: &AccountTarget, b: &Account<F>) -> Result<()> {
        self._set_common_targets(a, b)?;
        self.set_hash_target(a.partial_hash, b.partial_hash)?;
        self.set_hash_target(a.partial_hash_for_pub_data, b.partial_hash_for_pub_data)?;

        Ok(())
    }

    fn _set_common_targets(&mut self, a: &AccountTarget, b: &Account<F>) -> Result<()> {
        self.set_target(a.account_index, F::from_canonical_i64(b.account_index))?;
        self.set_target(
            a.master_account_index,
            F::from_canonical_i64(b.master_account_index),
        )?;
        self.set_biguint_target(&a.l1_address, &b.l1_address)?;
        self.set_target(a.account_type, F::from_canonical_u8(b.account_type))?;
        self.set_bigint_target(&a.collateral, &b.collateral)?;
        for i in 0..NB_ASSETS_PER_TX {
            self.set_bigint_target(&a.aggregated_balances[i], &b.aggregated_balances[i])?;
        }
        self.set_target(
            a.total_order_count,
            F::from_canonical_i64(b.total_order_count),
        )?;
        self.set_target(
            a.total_non_cross_order_count,
            F::from_canonical_i64(b.total_non_cross_order_count),
        )?;
        self.set_target(a.cancel_all_time, F::from_canonical_i64(b.cancel_all_time))?;
        self.set_hash_target(a.api_key_root, b.api_key_root)?;
        self.set_hash_target(a.account_orders_root, b.account_orders_root)?;
        self.set_hash_target(a.asset_root, b.asset_root)?;
        self.set_hash_target(a.aggregated_balances_root, b.aggregated_balances_root)?;

        Ok(())
    }
}
