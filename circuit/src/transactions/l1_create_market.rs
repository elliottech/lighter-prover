// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::cmp::max;

use anyhow::{Ok, Result};
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bigint::big_u16::CircuitBuilderBigIntU16;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::signed::signed_target::CircuitBuilderSigned;
use crate::tx_interface::{Apply, PriorityOperationsPubData, Verify};
use crate::types::config::{Builder, F};
use crate::types::constants::*;
use crate::types::market::{MarketTarget, ensure_spot_market_index, select_market};
use crate::types::market_details::{MarketDetailsTarget, select_market_details};
use crate::types::target_pub_data_helper::*;
use crate::types::tx_state::TxState;
use crate::types::tx_type::TxTypeTargets;
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::utils::CircuitBuilderUtils;

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct L1CreateMarketTx {
    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "mt")]
    pub market_type: u8,
    #[serde(rename = "ba")]
    pub base_asset_id: u16,
    #[serde(rename = "qa")]
    pub quote_asset_id: u16,

    #[serde(rename = "qm")]
    pub quote_multiplier: u32, // 14 bits

    #[serde(rename = "sem")]
    pub size_extension_multiplier: i64,
    #[serde(rename = "qem")]
    pub quote_extension_multiplier: i64,

    #[serde(rename = "tk")]
    pub taker_fee: u32,
    #[serde(rename = "mf")]
    pub maker_fee: u32,
    #[serde(rename = "lf")]
    pub liquidation_fee: u32,

    #[serde(rename = "mba")]
    pub min_base_amount: u64,
    #[serde(rename = "mqa")]
    pub min_quote_amount: u64,

    #[serde(rename = "mimf")]
    pub min_initial_margin_fraction: u16,
    #[serde(rename = "dimf")]
    pub default_initial_margin_fraction: u16,
    #[serde(rename = "mmf")]
    pub maintenance_margin_fraction: u16,
    #[serde(rename = "cmf")]
    pub close_out_margin_fraction: u16,

    #[serde(rename = "ir")]
    pub interest_rate: u32, // 20 bits

    #[serde(rename = "fcs")]
    pub funding_clamp_small: u32,
    #[serde(rename = "fcb")]
    pub funding_clamp_big: u32,

    #[serde(rename = "oil")]
    pub open_interest_limit: u64,
    #[serde(rename = "oql")]
    pub order_quote_limit: u64,
}

#[derive(Debug)]
pub struct L1CreateMarketTxTarget {
    pub market_index: Target,

    pub market_type: Target,
    pub base_asset_id: Target,
    pub quote_asset_id: Target,

    pub quote_multiplier: Target,

    pub size_extension_multiplier: Target,
    pub quote_extension_multiplier: Target,

    pub taker_fee: Target,
    pub maker_fee: Target,
    pub liquidation_fee: Target,

    pub min_base_amount: Target,
    pub min_quote_amount: Target,

    pub default_initial_margin_fraction: Target,
    pub min_initial_margin_fraction: Target,
    pub maintenance_margin_fraction: Target,
    pub close_out_margin_fraction: Target,

    pub interest_rate: Target,

    pub funding_clamp_small: Target,
    pub funding_clamp_big: Target,

    pub open_interest_limit: Target,
    pub order_quote_limit: Target,

    // Output
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1CreateMarketTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            market_index: builder.add_virtual_target(),

            market_type: builder.add_virtual_target(),
            base_asset_id: builder.add_virtual_target(),
            quote_asset_id: builder.add_virtual_target(),

            quote_multiplier: builder.add_virtual_target(),
            size_extension_multiplier: builder.add_virtual_target(),
            quote_extension_multiplier: builder.add_virtual_target(),
            taker_fee: builder.add_virtual_target(),
            maker_fee: builder.add_virtual_target(),
            liquidation_fee: builder.add_virtual_target(),
            min_base_amount: builder.add_virtual_target(),
            min_quote_amount: builder.add_virtual_target(),
            default_initial_margin_fraction: builder.add_virtual_target(),
            min_initial_margin_fraction: builder.add_virtual_target(),
            maintenance_margin_fraction: builder.add_virtual_target(),
            close_out_margin_fraction: builder.add_virtual_target(),
            interest_rate: builder.add_virtual_target(),
            funding_clamp_small: builder.add_virtual_target(),
            funding_clamp_big: builder.add_virtual_target(),
            open_interest_limit: builder.add_virtual_target(),
            order_quote_limit: builder.add_virtual_target(),

            // Output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        // No range check on market_index, base_asset_id, quote_asset_id as they're splitted to bits for merkle proofs.

        // Can't deploy something to NIL market index
        let nil_market_index = builder.constant_u64(NIL_MARKET_INDEX as u64);
        builder.conditional_assert_not_eq(self.is_enabled, self.market_index, nil_market_index);

        builder.assert_bool(BoolTarget::new_unsafe(self.market_type));

        // Range check quote multiplier in range (0, MAX_QUOTE_MULTIPLIER]. MAX_QUOTE_MULTIPLIER
        // is around 14 bits, we use 16 bits to hit special range-check case
        builder.register_range_check(self.quote_multiplier, 16);
        builder.register_range_check(
            self.size_extension_multiplier,
            ASSET_EXTENSION_MULTIPLIER_BITS,
        );
        builder.register_range_check(
            self.quote_extension_multiplier,
            ASSET_EXTENSION_MULTIPLIER_BITS,
        );
        builder.register_range_check(self.taker_fee, 24);
        builder.register_range_check(self.maker_fee, 24);
        builder.register_range_check(self.liquidation_fee, 24);
        builder.register_range_check(self.min_base_amount, ORDER_BASE_AMOUNT_BITS);
        builder.register_range_check(self.min_quote_amount, ORDER_QUOTE_SIZE_BITS);
        builder.register_range_check(self.min_initial_margin_fraction, 16);
        builder.register_range_check(self.default_initial_margin_fraction, 16);
        builder.register_range_check(self.maintenance_margin_fraction, 16);
        builder.register_range_check(self.close_out_margin_fraction, 16);
        builder.register_range_check(self.interest_rate, 24);
        builder.register_range_check(self.funding_clamp_small, 24);
        builder.register_range_check(self.funding_clamp_big, 24);
        builder.register_range_check(self.open_interest_limit, MARKET_OPEN_INTEREST_BITS);
        builder.register_range_check(self.order_quote_limit, ORDER_QUOTE_SIZE_BITS);
    }

    fn verify_perps_market_type(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        tx_state: &TxState,
    ) {
        let flag = builder.and(self.success, is_enabled);

        let max_perps_market_index = builder.constant_usize(MAX_PERPS_MARKET_INDEX);
        builder.conditional_assert_lte(is_enabled, self.market_index, max_perps_market_index, 16);

        // Verify that 0 <= close_out <= maintenance <= min_initial <= default_initial < MARGIN_TICK where MARGIN_TICK is around 14 bits.
        // We use 16 bits to hit special range-check case
        let margin_tick = builder.constant(F::from_canonical_u32(MARGIN_TICK));
        builder.conditional_assert_not_zero(flag, self.close_out_margin_fraction);
        builder.conditional_assert_lte(
            flag,
            self.close_out_margin_fraction,
            self.maintenance_margin_fraction,
            16,
        );
        builder.conditional_assert_lte(
            flag,
            self.maintenance_margin_fraction,
            self.min_initial_margin_fraction,
            16,
        );
        builder.conditional_assert_lte(
            flag,
            self.min_initial_margin_fraction,
            self.default_initial_margin_fraction,
            16,
        );
        builder.conditional_assert_lte(flag, self.default_initial_margin_fraction, margin_tick, 16);

        let fee_tick = builder.constant(F::from_canonical_u64(FEE_TICK));
        builder.conditional_assert_lte(flag, self.liquidation_fee, fee_tick, 24);

        let funding_rate_tick = builder.constant(F::from_canonical_u32(FUNDING_RATE_TICK));
        builder.conditional_assert_lte(flag, self.interest_rate, funding_rate_tick, 24);
        builder.conditional_assert_lte(flag, self.funding_clamp_small, funding_rate_tick, 24);
        builder.conditional_assert_lte(flag, self.funding_clamp_big, funding_rate_tick, 24);

        builder.conditional_assert_lte(
            flag,
            self.order_quote_limit,
            self.open_interest_limit,
            MARKET_OPEN_INTEREST_BITS,
        );

        builder.conditional_assert_not_zero(flag, self.quote_multiplier);
        let max_quote_multiplier = builder.constant(F::from_canonical_u32(MAX_QUOTE_MULTIPLIER));
        builder.conditional_assert_lte(flag, self.quote_multiplier, max_quote_multiplier, 16);

        builder.conditional_assert_zero(flag, self.base_asset_id);
        builder.conditional_assert_zero(flag, self.quote_asset_id);
        builder.conditional_assert_zero(flag, self.size_extension_multiplier);
        builder.conditional_assert_zero(flag, self.quote_extension_multiplier);

        let market_status_expired = builder.constant_from_u8(MARKET_STATUS_EXPIRED);
        let market_is_not_expired =
            builder.is_not_equal(tx_state.market_details.status, market_status_expired);
        let is_market_open_interest_not_zero =
            builder.is_not_zero(tx_state.market_details.open_interest);
        let not_expired_or_nonzero_open_interest =
            builder.multi_or(&[market_is_not_expired, is_market_open_interest_not_zero]);
        let should_be_false = builder.and(flag, not_expired_or_nonzero_open_interest);
        self.success = builder.and_not(self.success, should_be_false);
    }

    fn verify_spot_market_type(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        tx_state: &TxState,
    ) {
        let flag = builder.and(self.success, is_enabled);

        ensure_spot_market_index(builder, flag, self.market_index);

        builder.conditional_assert_zero(flag, self.liquidation_fee);
        builder.conditional_assert_zero(flag, self.close_out_margin_fraction);
        builder.conditional_assert_zero(flag, self.maintenance_margin_fraction);
        builder.conditional_assert_zero(flag, self.min_initial_margin_fraction);
        builder.conditional_assert_zero(flag, self.default_initial_margin_fraction);
        builder.conditional_assert_zero(flag, self.interest_rate);
        builder.conditional_assert_zero(flag, self.funding_clamp_small);
        builder.conditional_assert_zero(flag, self.funding_clamp_big);
        builder.conditional_assert_zero(flag, self.open_interest_limit);

        builder.conditional_assert_zero(flag, self.quote_multiplier);
        builder.conditional_assert_not_zero(flag, self.size_extension_multiplier);
        builder.conditional_assert_not_zero(flag, self.quote_extension_multiplier);

        builder.conditional_assert_not_eq(flag, self.base_asset_id, self.quote_asset_id);
        let is_base_asset_empty = tx_state.assets[BASE_ASSET_ID].is_empty(builder);
        let is_quote_asset_empty = tx_state.assets[QUOTE_ASSET_ID].is_empty(builder);
        let is_base_or_quote_asset_empty = builder.or(is_base_asset_empty, is_quote_asset_empty);
        let should_be_false = builder.and(flag, is_base_or_quote_asset_empty);
        self.success = builder.and_not(self.success, should_be_false);
    }
}

impl Verify for L1CreateMarketTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.is_enabled = tx_type.is_l1_create_market;
        self.success = tx_type.is_l1_create_market;

        self.register_range_checks(builder);

        builder.conditional_assert_eq(
            self.success,
            self.market_index,
            tx_state.market.market_index,
        );
        builder.conditional_assert_eq(
            self.success,
            self.base_asset_id,
            tx_state.asset_indices[BASE_ASSET_ID],
        );
        builder.conditional_assert_eq(
            self.success,
            self.quote_asset_id,
            tx_state.asset_indices[QUOTE_ASSET_ID],
        );

        let is_perps_market_type = builder.is_equal_constant(self.market_type, MARKET_TYPE_PERPS);
        let is_spot_market_type = builder.not(is_perps_market_type);

        self.verify_perps_market_type(builder, is_perps_market_type, tx_state);
        self.verify_spot_market_type(builder, is_spot_market_type, tx_state);

        // 0 < min_base_amount < 2^ORDER_SIZE_BITS
        // 0 < min_quote_amount < 2^ORDER_QUOTE_SIZE_BITS and min_quote_amount < order_quote_limit
        builder.conditional_assert_not_zero(self.success, self.min_base_amount);
        builder.conditional_assert_not_zero(self.success, self.min_quote_amount);
        builder.conditional_assert_lte(
            self.success,
            self.min_quote_amount,
            self.order_quote_limit,
            ORDER_QUOTE_SIZE_BITS,
        );

        let fee_tick = builder.constant(F::from_canonical_u64(FEE_TICK));
        builder.conditional_assert_lte(self.success, self.taker_fee, fee_tick, 24);
        builder.conditional_assert_lte(self.success, self.maker_fee, fee_tick, 24);

        // Verify that market is inactive and empty
        let order_book_status_expired = builder.constant_from_u8(MARKET_STATUS_EXPIRED);
        let order_book_is_expired =
            builder.is_equal(tx_state.market.status, order_book_status_expired);
        let order_book_total_order_count_zero = builder.is_zero(tx_state.market.total_order_count);
        let order_book_is_inactive =
            builder.multi_and(&[order_book_is_expired, order_book_total_order_count_zero]);
        self.success = builder.and(self.success, order_book_is_inactive);
    }
}

impl Apply for L1CreateMarketTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        let is_perps_market_type = builder.is_equal_constant(self.market_type, MARKET_TYPE_PERPS);
        let nil_market_index = builder.constant_u64(NIL_MARKET_INDEX as u64);

        let market_after = MarketTarget {
            market_index: self.market_index,
            perps_market_index: builder.select(
                is_perps_market_type,
                self.market_index,
                nil_market_index,
            ),

            status: builder.constant_u8(MARKET_STATUS_ACTIVE).0,
            market_type: self.market_type,
            base_asset_id: self.base_asset_id,
            quote_asset_id: self.quote_asset_id,

            ask_nonce: builder.constant_i64(FIRST_ASK_NONCE),
            bid_nonce: builder.constant_i64(FIRST_BID_NONCE),

            taker_fee: self.taker_fee,
            maker_fee: self.maker_fee,
            liquidation_fee: self.liquidation_fee,
            size_extension_multiplier: self.size_extension_multiplier,
            quote_extension_multiplier: self.quote_extension_multiplier,
            total_order_count: builder.zero(),
            min_base_amount: self.min_base_amount,
            min_quote_amount: self.min_quote_amount,
            order_quote_limit: self.order_quote_limit,

            order_book_root: builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT),
        };
        tx_state.market = select_market(builder, self.success, &market_after, &tx_state.market);

        let update_market_details_flag = builder.and(self.success, is_perps_market_type);
        let market_details_after = MarketDetailsTarget {
            default_initial_margin_fraction: self.default_initial_margin_fraction,
            min_initial_margin_fraction: self.min_initial_margin_fraction,
            maintenance_margin_fraction: self.maintenance_margin_fraction,
            close_out_margin_fraction: self.close_out_margin_fraction,
            quote_multiplier: self.quote_multiplier,
            funding_rate_prefix_sum: builder.zero_bigint_u16(),
            aggregate_premium_sum: builder.zero_signed(),
            interest_rate: self.interest_rate,
            impact_price: builder.zero(),
            impact_bid_price: builder.zero(),
            impact_ask_price: builder.zero(),
            open_interest: builder.zero(),
            index_price: builder.zero(),
            mark_price: builder.zero(),
            status: builder.constant_from_u8(MARKET_STATUS_ACTIVE),
            funding_clamp_small: self.funding_clamp_small,
            funding_clamp_big: self.funding_clamp_big,
            open_interest_limit: self.open_interest_limit,
            strategy_index: builder.constant_usize(DEFAULT_STRATEGY_INDEX),
        };
        tx_state.market_details = select_market_details(
            builder,
            update_market_details_flag,
            &market_details_after,
            &tx_state.market_details,
        );

        self.success
    }
}

impl PriorityOperationsPubData for L1CreateMarketTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        const PACKED_COMMON_PERPS_DATA_BYTES: usize = 59 + 4; // +4 for market type, market index, pub data type
        let perps = &mut Vec::<U8Target>::with_capacity(PACKED_COMMON_PERPS_DATA_BYTES);
        let perps_byte_count = [
            add_pub_data_type_target(builder, perps, PRIORITY_PUB_DATA_TYPE_L1_CREATE_MARKET),
            add_target(builder, perps, self.market_index, 16),
            add_byte_target_unsafe(perps, self.market_type),
            add_target(builder, perps, self.quote_multiplier, 32),
            add_target(builder, perps, self.taker_fee, 32),
            add_target(builder, perps, self.maker_fee, 32),
            add_target(builder, perps, self.liquidation_fee, 32),
            add_target(builder, perps, self.min_base_amount, ORDER_BASE_AMOUNT_BITS),
            add_target(builder, perps, self.min_quote_amount, ORDER_QUOTE_SIZE_BITS),
            add_target(builder, perps, self.default_initial_margin_fraction, 16),
            add_target(builder, perps, self.min_initial_margin_fraction, 16),
            add_target(builder, perps, self.maintenance_margin_fraction, 16),
            add_target(builder, perps, self.close_out_margin_fraction, 16),
            add_target(builder, perps, self.interest_rate, 32),
            add_target(builder, perps, self.funding_clamp_small, 24),
            add_target(builder, perps, self.funding_clamp_big, 24),
            add_target(
                builder,
                perps,
                self.open_interest_limit,
                MARKET_OPEN_INTEREST_BITS,
            ),
            add_target(
                builder,
                perps,
                self.order_quote_limit,
                ORDER_QUOTE_SIZE_BITS,
            ),
        ]
        .iter()
        .sum::<usize>();
        assert_eq!(perps_byte_count, PACKED_COMMON_PERPS_DATA_BYTES);

        const PACKED_COMMON_SPOT_DATA_BYTES: usize = 44 + 4; // +4 for market type, market index, pub data type
        let spot = &mut Vec::<U8Target>::with_capacity(PACKED_COMMON_SPOT_DATA_BYTES);
        let spot_byte_count = [
            add_pub_data_type_target(builder, spot, PRIORITY_PUB_DATA_TYPE_L1_CREATE_MARKET),
            add_target(builder, spot, self.market_index, 16),
            add_byte_target_unsafe(spot, self.market_type),
            add_target(builder, spot, self.base_asset_id, 16),
            add_target(builder, spot, self.quote_asset_id, 16),
            add_target(
                builder,
                spot,
                self.size_extension_multiplier,
                ASSET_EXTENSION_MULTIPLIER_BITS,
            ),
            add_target(
                builder,
                spot,
                self.quote_extension_multiplier,
                ASSET_EXTENSION_MULTIPLIER_BITS,
            ),
            add_target(builder, spot, self.taker_fee, 32),
            add_target(builder, spot, self.maker_fee, 32),
            add_target(builder, spot, self.min_base_amount, ORDER_BASE_AMOUNT_BITS),
            add_target(builder, spot, self.min_quote_amount, ORDER_QUOTE_SIZE_BITS),
            add_target(builder, spot, self.order_quote_limit, ORDER_QUOTE_SIZE_BITS),
        ]
        .iter()
        .sum::<usize>();
        assert_eq!(spot_byte_count, PACKED_COMMON_SPOT_DATA_BYTES);

        let bytes =
            &mut Vec::<U8Target>::with_capacity(MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX);
        let max_priority_pub_data_count = max(
            PACKED_COMMON_PERPS_DATA_BYTES,
            PACKED_COMMON_SPOT_DATA_BYTES,
        );
        let is_spot = builder.is_equal_constant(self.market_type, MARKET_TYPE_SPOT);
        let zero_byte = builder.zero_u8();
        for i in 0..max_priority_pub_data_count {
            let spot_byte = *spot.get(i).unwrap_or(&zero_byte);
            let perps_byte = *perps.get(i).unwrap_or(&zero_byte);
            bytes.push(builder.select_u8(is_spot, spot_byte, perps_byte));
        }

        (
            self.is_enabled,
            pad_priority_op_pub_data_target(builder, bytes, max_priority_pub_data_count),
        )
    }
}

pub trait L1CreateMarketTxTargetWitness<F: PrimeField64> {
    fn set_l1_create_market_tx_target(
        &mut self,
        a: &L1CreateMarketTxTarget,
        b: &L1CreateMarketTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1CreateMarketTxTargetWitness<F> for T {
    fn set_l1_create_market_tx_target(
        &mut self,
        a: &L1CreateMarketTxTarget,
        b: &L1CreateMarketTx,
    ) -> Result<()> {
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;

        self.set_target(a.market_type, F::from_canonical_u8(b.market_type))?;
        self.set_target(a.base_asset_id, F::from_canonical_u16(b.base_asset_id))?;
        self.set_target(a.quote_asset_id, F::from_canonical_u16(b.quote_asset_id))?;

        self.set_target(
            a.quote_multiplier,
            F::from_canonical_u32(b.quote_multiplier),
        )?;
        self.set_target(
            a.size_extension_multiplier,
            F::from_canonical_i64(b.size_extension_multiplier),
        )?;
        self.set_target(
            a.quote_extension_multiplier,
            F::from_canonical_i64(b.quote_extension_multiplier),
        )?;
        self.set_target(a.taker_fee, F::from_canonical_u32(b.taker_fee))?;
        self.set_target(a.maker_fee, F::from_canonical_u32(b.maker_fee))?;
        self.set_target(a.liquidation_fee, F::from_canonical_u32(b.liquidation_fee))?;
        self.set_target(a.min_base_amount, F::from_canonical_u64(b.min_base_amount))?;
        self.set_target(
            a.min_quote_amount,
            F::from_canonical_u64(b.min_quote_amount),
        )?;
        self.set_target(
            a.default_initial_margin_fraction,
            F::from_canonical_u16(b.default_initial_margin_fraction),
        )?;
        self.set_target(
            a.min_initial_margin_fraction,
            F::from_canonical_u16(b.min_initial_margin_fraction),
        )?;
        self.set_target(
            a.maintenance_margin_fraction,
            F::from_canonical_u16(b.maintenance_margin_fraction),
        )?;
        self.set_target(
            a.close_out_margin_fraction,
            F::from_canonical_u16(b.close_out_margin_fraction),
        )?;
        self.set_target(a.interest_rate, F::from_canonical_u32(b.interest_rate))?;
        self.set_target(
            a.funding_clamp_small,
            F::from_canonical_u32(b.funding_clamp_small),
        )?;
        self.set_target(
            a.funding_clamp_big,
            F::from_canonical_u32(b.funding_clamp_big),
        )?;
        self.set_target(
            a.open_interest_limit,
            F::from_canonical_u64(b.open_interest_limit),
        )?;
        self.set_target(
            a.order_quote_limit,
            F::from_canonical_u64(b.order_quote_limit),
        )?;

        Ok(())
    }
}
