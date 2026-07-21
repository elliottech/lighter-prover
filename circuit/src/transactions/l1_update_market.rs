// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::cmp::max;

use anyhow::Result;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::Witness;
use serde::Deserialize;

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
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

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct L1UpdateMarketTx {
    #[serde(rename = "mi")]
    pub market_index: u16,

    #[serde(rename = "mt")]
    pub market_type: u8,

    #[serde(rename = "s")]
    pub status: u8,

    #[serde(rename = "tk")]
    pub taker_fee: u32,

    #[serde(rename = "mf")]
    pub maker_fee: u32,

    #[serde(rename = "lf")]
    pub liquidation_fee: u32,

    #[serde(rename = "mb")]
    pub min_base_amount: u64,

    #[serde(rename = "mq")]
    pub min_quote_amount: u64,

    #[serde(rename = "dm")]
    pub default_initial_margin_fraction: u16,

    #[serde(rename = "im")]
    pub min_initial_margin_fraction: u16,

    #[serde(rename = "mm", default)]
    pub maintenance_margin_fraction: u16,

    #[serde(rename = "cm")]
    pub close_out_margin_fraction: u16,

    #[serde(rename = "ir")]
    pub interest_rate: u32,

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
pub struct L1UpdateMarketTxTarget {
    pub market_index: Target,
    pub market_type: Target,
    pub status: Target,
    pub taker_fee: Target,
    pub maker_fee: Target,
    pub liquidation_fee: Target,
    pub min_base_amount: Target,  // 48 bits
    pub min_quote_amount: Target, // 48 bits
    pub default_initial_margin_fraction: Target,
    pub min_initial_margin_fraction: Target,
    pub maintenance_margin_fraction: Target,
    pub close_out_margin_fraction: Target,
    pub interest_rate: Target, // 20 bits
    pub funding_clamp_small: Target,
    pub funding_clamp_big: Target,
    pub open_interest_limit: Target,
    pub order_quote_limit: Target,

    // output
    success: BoolTarget,
    is_enabled: BoolTarget,
}

impl L1UpdateMarketTxTarget {
    pub fn new(builder: &mut Builder) -> Self {
        L1UpdateMarketTxTarget {
            market_index: builder.add_virtual_target(),
            market_type: builder.add_virtual_target(),
            status: builder.add_virtual_target(),
            taker_fee: builder.add_virtual_target(),
            maker_fee: builder.add_virtual_target(),
            liquidation_fee: builder.add_virtual_target(),
            min_base_amount: builder.add_virtual_target(),
            min_quote_amount: builder.add_virtual_target(),
            default_initial_margin_fraction: builder.add_virtual_target(),
            min_initial_margin_fraction: builder.add_virtual_target(),
            maintenance_margin_fraction: builder.add_virtual_target(),
            interest_rate: builder.add_virtual_target(),
            close_out_margin_fraction: builder.add_virtual_target(),
            funding_clamp_small: builder.add_virtual_target(),
            funding_clamp_big: builder.add_virtual_target(),
            open_interest_limit: builder.add_virtual_target(),
            order_quote_limit: builder.add_virtual_target(),

            // output
            success: BoolTarget::default(),
            is_enabled: BoolTarget::default(),
        }
    }

    fn register_range_checks(&mut self, builder: &mut Builder) {
        // Market type is asserted equal to tx_state

        builder.assert_bool(BoolTarget::new_unsafe(self.status));

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

    fn verify_perps_market_type(&mut self, builder: &mut Builder, is_enabled: BoolTarget) {
        let flag = builder.and(self.is_enabled, is_enabled);

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
    }

    fn verify_spot_market_type(&mut self, builder: &mut Builder, is_enabled: BoolTarget) {
        let flag = builder.and(self.is_enabled, is_enabled);

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
    }
}

impl Verify for L1UpdateMarketTxTarget {
    fn verify(&mut self, builder: &mut Builder, tx_type: &TxTypeTargets, tx_state: &TxState) {
        self.is_enabled = tx_type.is_l1_update_market;
        self.success = tx_type.is_l1_update_market;

        self.register_range_checks(builder);

        builder.conditional_assert_eq(
            self.is_enabled,
            self.market_index,
            tx_state.market.market_index,
        );

        let is_perps_market_type = builder.is_equal_constant(self.market_type, MARKET_TYPE_PERPS);
        let is_spot_market_type = builder.not(is_perps_market_type);

        self.verify_perps_market_type(builder, is_perps_market_type);
        self.verify_spot_market_type(builder, is_spot_market_type);

        // 0 < min_base_amount < 2^ORDER_SIZE_BITS
        // 0 < min_quote_amount < 2^ORDER_QUOTE_SIZE_BITS and min_quote_amount < order_quote_limit
        builder.conditional_assert_not_zero(self.is_enabled, self.min_base_amount);
        builder.conditional_assert_not_zero(self.is_enabled, self.min_quote_amount);
        builder.conditional_assert_lte(
            self.is_enabled,
            self.min_quote_amount,
            self.order_quote_limit,
            ORDER_QUOTE_SIZE_BITS,
        );

        let fee_tick = builder.constant(F::from_canonical_u64(FEE_TICK));
        builder.conditional_assert_lte(self.is_enabled, self.taker_fee, fee_tick, 24);
        builder.conditional_assert_lte(self.is_enabled, self.maker_fee, fee_tick, 24);

        // Do not allow updating an already expired market
        let expired_status = builder.constant_from_u8(MARKET_STATUS_EXPIRED);
        let order_book_expired = builder.is_equal(tx_state.market.status, expired_status);
        self.success = builder.and_not(self.success, order_book_expired);

        // Verify that the market type in the tx matches the market type in the state
        builder.conditional_assert_eq(self.success, self.market_type, tx_state.market.market_type);
    }
}

impl Apply for L1UpdateMarketTxTarget {
    fn apply(&mut self, builder: &mut Builder, tx_state: &mut TxState) -> BoolTarget {
        // Update market and market details
        tx_state.market = MarketTarget {
            status: builder.select(self.success, self.status, tx_state.market.status),
            taker_fee: builder.select(self.success, self.taker_fee, tx_state.market.taker_fee),
            maker_fee: builder.select(self.success, self.maker_fee, tx_state.market.maker_fee),
            liquidation_fee: builder.select(
                self.success,
                self.liquidation_fee,
                tx_state.market.liquidation_fee,
            ),
            min_base_amount: builder.select(
                self.success,
                self.min_base_amount,
                tx_state.market.min_base_amount,
            ),
            min_quote_amount: builder.select(
                self.success,
                self.min_quote_amount,
                tx_state.market.min_quote_amount,
            ),
            order_quote_limit: builder.select(
                self.success,
                self.order_quote_limit,
                tx_state.market.order_quote_limit,
            ),
            ..tx_state.market.clone()
        };

        let is_perps_market_type = builder.is_equal_constant(self.market_type, MARKET_TYPE_PERPS);
        let update_market_details_flag = builder.and(self.success, is_perps_market_type);
        tx_state.market_details = MarketDetailsTarget {
            status: builder.select(
                update_market_details_flag,
                self.status,
                tx_state.market_details.status,
            ),
            min_initial_margin_fraction: builder.select(
                update_market_details_flag,
                self.min_initial_margin_fraction,
                tx_state.market_details.min_initial_margin_fraction,
            ),
            default_initial_margin_fraction: builder.select(
                update_market_details_flag,
                self.default_initial_margin_fraction,
                tx_state.market_details.default_initial_margin_fraction,
            ),
            maintenance_margin_fraction: builder.select(
                update_market_details_flag,
                self.maintenance_margin_fraction,
                tx_state.market_details.maintenance_margin_fraction,
            ),
            close_out_margin_fraction: builder.select(
                update_market_details_flag,
                self.close_out_margin_fraction,
                tx_state.market_details.close_out_margin_fraction,
            ),
            interest_rate: builder.select(
                update_market_details_flag,
                self.interest_rate,
                tx_state.market_details.interest_rate,
            ),
            funding_clamp_small: builder.select(
                update_market_details_flag,
                self.funding_clamp_small,
                tx_state.market_details.funding_clamp_small,
            ),
            funding_clamp_big: builder.select(
                update_market_details_flag,
                self.funding_clamp_big,
                tx_state.market_details.funding_clamp_big,
            ),
            open_interest_limit: builder.select(
                update_market_details_flag,
                self.open_interest_limit,
                tx_state.market_details.open_interest_limit,
            ),
            ..tx_state.market_details.clone()
        };

        // Clear market if expired and empty
        let market_status_expired = builder.constant_from_u8(MARKET_STATUS_EXPIRED);
        let is_market_expired = builder.is_equal(self.status, market_status_expired);
        let no_open_order = builder.is_zero(tx_state.market.total_order_count);
        let no_open_position = builder.is_zero(tx_state.market_details.open_interest);
        let clear_perps_market_flag = builder.multi_and(&[
            self.success,
            is_market_expired,
            no_open_order,
            no_open_position,
            is_perps_market_type,
        ]);
        let is_spot_market_type = builder.not(is_perps_market_type);
        let clear_spot_market_flag = builder.multi_and(&[
            self.success,
            is_market_expired,
            no_open_order,
            is_spot_market_type,
        ]);

        let clear_market_details_flag = clear_perps_market_flag;
        let clear_order_book_flag = builder.or(clear_perps_market_flag, clear_spot_market_flag);

        let empty_order_book_root = builder.constant_hash(EMPTY_ORDER_BOOK_TREE_ROOT);
        let empty_market = MarketTarget::empty(
            builder,
            tx_state.market.market_index,
            tx_state.market.perps_market_index,
            empty_order_book_root,
        );
        tx_state.market = select_market(
            builder,
            clear_order_book_flag,
            &empty_market,
            &tx_state.market,
        );

        // Clear market details if expired and empty
        let empty_market_detail = MarketDetailsTarget::empty(builder);
        tx_state.market_details = select_market_details(
            builder,
            clear_market_details_flag,
            &empty_market_detail,
            &tx_state.market_details,
        );

        self.success
    }
}

impl PriorityOperationsPubData for L1UpdateMarketTxTarget {
    fn priority_operations_pub_data(
        &self,
        builder: &mut Builder,
    ) -> (
        BoolTarget,
        [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],
    ) {
        const PACKED_COMMON_PERPS_DATA_BYTES: usize = 56 + 4; // +4 for market type, market index, pub data type
        let perps = &mut Vec::<U8Target>::with_capacity(PACKED_COMMON_PERPS_DATA_BYTES);
        let perps_byte_count = [
            add_pub_data_type_target(builder, perps, PRIORITY_PUB_DATA_TYPE_L1_UPDATE_MARKET),
            add_target(builder, perps, self.market_index, 16),
            add_byte_target_unsafe(perps, self.market_type),
            add_byte_target_unsafe(perps, self.status),
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

        const PACKED_COMMON_SPOT_DATA_BYTES: usize = 27 + 4; // +4 for market type, market index, pub data type
        let spot = &mut Vec::<U8Target>::with_capacity(PACKED_COMMON_SPOT_DATA_BYTES);
        let spot_byte_count = [
            add_pub_data_type_target(builder, spot, PRIORITY_PUB_DATA_TYPE_L1_UPDATE_MARKET),
            add_target(builder, spot, self.market_index, 16),
            add_byte_target_unsafe(spot, self.market_type),
            add_byte_target_unsafe(spot, self.status),
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

pub trait L1UpdateMarketTxTargetWitness<F: PrimeField64> {
    fn set_l1_update_market_tx_target(
        &mut self,
        a: &L1UpdateMarketTxTarget,
        b: &L1UpdateMarketTx,
    ) -> Result<()>;
}

impl<T: Witness<F>, F: PrimeField64> L1UpdateMarketTxTargetWitness<F> for T {
    fn set_l1_update_market_tx_target(
        &mut self,
        a: &L1UpdateMarketTxTarget,
        b: &L1UpdateMarketTx,
    ) -> Result<()> {
        self.set_target(a.market_index, F::from_canonical_u16(b.market_index))?;
        self.set_target(a.market_type, F::from_canonical_u8(b.market_type))?;
        self.set_target(a.status, F::from_canonical_u8(b.status))?;
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
