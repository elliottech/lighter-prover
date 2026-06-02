// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::{BoolTarget, Target};

use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::types::config::Builder;
use crate::types::constants::{
    CANCEL_ALL_ACCOUNT_ORDERS, CANCEL_ALL_CROSS_MARGIN_ORDERS, CANCEL_ALL_ISOLATED_MARGIN_ORDERS,
    CANCEL_ALL_MARKET_ACCOUNT_ORDERS, MASTER_ACCOUNT_TYPE, NIL_MARKET_INDEX, OWNER_ACCOUNT_ID,
    SUB_ACCOUNT_TYPE,
};
use crate::types::register::BaseRegisterInfoTarget;
use crate::types::tx_state::TxState;
use crate::utils::CircuitBuilderUtils;

/// Places the new register in 0th index, caller must be aware of this.
pub fn apply_immediate_cancel_all(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    account_index: Target,
) {
    let zero = builder.zero();
    let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);

    //clear DMS time
    tx_state.accounts[OWNER_ACCOUNT_ID].cancel_all_time = builder.select(
        is_enabled,
        zero,
        tx_state.accounts[OWNER_ACCOUNT_ID].cancel_all_time,
    );

    let new_register = BaseRegisterInfoTarget {
        instruction_type: builder.constant_from_u8(CANCEL_ALL_ACCOUNT_ORDERS),
        account_index,
        pending_size: tx_state.accounts[OWNER_ACCOUNT_ID].total_order_count,
        market_index: nil_market_index,
        ..BaseRegisterInfoTarget::empty(builder)
    };
    let is_not_open_order_exists =
        builder.is_zero(tx_state.accounts[OWNER_ACCOUNT_ID].total_order_count);
    let is_register_select_active = builder.and_not(is_enabled, is_not_open_order_exists);
    tx_state.put_to_instruction_stack_unsafe(builder, is_register_select_active, &new_register, 0);
}

/// Places the new register in 0th index, caller must be aware of this.
pub fn apply_immediate_cancel_all_market(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    account_index: Target,
    market_index: Target,
) {
    let new_register = BaseRegisterInfoTarget {
        instruction_type: builder.constant_from_u8(CANCEL_ALL_MARKET_ACCOUNT_ORDERS),
        account_index,
        pending_size: tx_state.positions[OWNER_ACCOUNT_ID].total_order_count,
        market_index,
        ..BaseRegisterInfoTarget::empty(builder)
    };
    let is_not_open_order_exists =
        builder.is_zero(tx_state.positions[OWNER_ACCOUNT_ID].total_order_count);
    let is_register_select_active = builder.and_not(is_enabled, is_not_open_order_exists);
    tx_state.put_to_instruction_stack_unsafe(builder, is_register_select_active, &new_register, 0);
}

/// Places the new register in 0th index, caller must be aware of this.
pub fn apply_isolated_cancel_all(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    account_index: Target,
    market_index: Target,
) {
    let new_register = BaseRegisterInfoTarget {
        instruction_type: builder.constant_from_u8(CANCEL_ALL_ISOLATED_MARGIN_ORDERS),
        account_index,
        market_index,
        pending_size: tx_state.positions[OWNER_ACCOUNT_ID].total_order_count,

        ..BaseRegisterInfoTarget::empty(builder)
    };
    let is_not_open_order_exists =
        builder.is_zero(tx_state.positions[OWNER_ACCOUNT_ID].total_order_count);
    let is_register_select_active = builder.and_not(is_enabled, is_not_open_order_exists);
    tx_state.put_to_instruction_stack_unsafe(builder, is_register_select_active, &new_register, 0);
}

/// Places the new register in 0th index, caller must be aware of this.
pub fn apply_cross_cancel_all(
    builder: &mut Builder,
    is_enabled: BoolTarget,
    tx_state: &mut TxState,
    account_index: Target,
) {
    let nil_market_index = builder.constant_from_u8(NIL_MARKET_INDEX);

    let is_using_margined_assets = {
        let is_master_account_type = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            MASTER_ACCOUNT_TYPE as u64,
        );
        let is_sub_account_type = builder.is_equal_constant(
            tx_state.accounts[OWNER_ACCOUNT_ID].account_type,
            SUB_ACCOUNT_TYPE as u64,
        );
        let is_valid_account_type = builder.or(is_master_account_type, is_sub_account_type);
        builder.and(
            is_valid_account_type,
            tx_state.accounts[OWNER_ACCOUNT_ID].is_unified_mode(),
        )
    };
    let is_not_using_margined_assets = builder.not(is_using_margined_assets);
    let subtract_from_total_order_count = builder.mul_bool(
        is_not_using_margined_assets,
        tx_state.accounts[OWNER_ACCOUNT_ID].total_non_cross_order_count,
    );
    let relevant_order_count_for_liquidation = builder.sub(
        tx_state.accounts[OWNER_ACCOUNT_ID].total_order_count,
        subtract_from_total_order_count,
    );
    let new_register = BaseRegisterInfoTarget {
        instruction_type: builder.select_constant(
            is_using_margined_assets,
            CANCEL_ALL_ACCOUNT_ORDERS as u64,
            CANCEL_ALL_CROSS_MARGIN_ORDERS as u64,
        ),
        account_index,
        market_index: nil_market_index,
        pending_size: relevant_order_count_for_liquidation,

        ..BaseRegisterInfoTarget::empty(builder)
    };
    let is_not_open_order_exists = builder.is_zero(relevant_order_count_for_liquidation);
    let is_register_select_active = builder.and_not(is_enabled, is_not_open_order_exists);
    tx_state.put_to_instruction_stack_unsafe(builder, is_register_select_active, &new_register, 0);
}
