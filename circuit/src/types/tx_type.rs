// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::types::Field;
use plonky2::iop::target::{BoolTarget, Target};

use super::account::AccountTarget;
use super::config::{Builder, F};
use super::constants::*;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::eddsa::schnorr::{SchnorrSigTarget, verify_schnorr_signature_conditional_circuit};
use crate::types::constants::{
    TX_TYPE_EMPTY, TX_TYPE_INTERNAL_CLAIM_ORDER, TX_TYPE_INTERNAL_CREATE_ORDER,
    TX_TYPE_INTERNAL_DELEVERAGE, TX_TYPE_INTERNAL_EXIT_POSITION,
    TX_TYPE_INTERNAL_LIQUIDATE_POSITION, TX_TYPE_L1_CREATE_MARKET, TX_TYPE_L1_CREATE_ORDER,
    TX_TYPE_L1_DEPOSIT, TX_TYPE_L1_UPDATE_MARKET, TX_TYPE_L2_BURN_SHARES, TX_TYPE_L2_CANCEL_ORDER,
    TX_TYPE_L2_CREATE_ORDER, TX_TYPE_L2_CREATE_PUBLIC_POOL, TX_TYPE_L2_MINT_SHARES,
    TX_TYPE_L2_UPDATE_LEVERAGE, TX_TYPE_L2_WITHDRAW,
};

#[derive(Debug)]
pub struct TxTypeTargets {
    pub is_empty: BoolTarget,
    pub is_l1_deposit: BoolTarget,
    pub is_l1_change_pub_key: BoolTarget,
    pub is_l1_create_market: BoolTarget,
    pub is_l1_update_market: BoolTarget,
    pub is_l1_cancel_all_orders: BoolTarget,
    pub is_l1_withdraw: BoolTarget,
    pub is_l1_create_order: BoolTarget,
    pub is_l1_burn_shares: BoolTarget,
    pub is_l1_register_asset: BoolTarget,
    pub is_l1_update_asset: BoolTarget,

    pub is_l2_change_pub_key: BoolTarget,
    pub is_l2_create_sub_account: BoolTarget,
    pub is_l2_create_public_pool: BoolTarget,
    pub is_l2_update_public_pool: BoolTarget,
    pub is_l2_transfer: BoolTarget,
    pub is_l2_withdraw: BoolTarget,
    pub is_l2_create_order: BoolTarget,
    pub is_l2_cancel_order: BoolTarget,
    pub is_l2_cancel_all_orders: BoolTarget,
    pub is_l2_modify_order: BoolTarget,
    pub is_l2_mint_shares: BoolTarget,
    pub is_l2_burn_shares: BoolTarget,
    pub is_l2_update_leverage: BoolTarget,
    pub is_l2_create_grouped_orders: BoolTarget,
    pub is_l2_update_margin: BoolTarget,

    pub is_internal_claim_order: BoolTarget,
    pub is_internal_cancel_order: BoolTarget,
    pub is_internal_deleverage: BoolTarget,
    pub is_internal_exit_position: BoolTarget,
    pub is_internal_cancel_all_orders: BoolTarget,
    pub is_internal_liquidate_position: BoolTarget,
    pub is_internal_create_order: BoolTarget,

    pub is_layer1: BoolTarget,
    pub is_layer2: BoolTarget,
    pub is_non_internal: BoolTarget, // Non-internal transactions (L1 and L2)
    pub is_sub_account_tx: BoolTarget, // Operations that second account has to be of type sub-account, public pool, or insurance fund operator
    pub is_dms_blocked_tx: BoolTarget, // Transactions that are blocked if dead man's switch needs to be triggered
}

#[derive(Debug)]
pub struct TxTypeVerifyTargets {
    pub expired_at: Target,
    pub block_created_at: Target,
    pub nonce: Target,
    pub api_key_before_nonce: Target,
    pub signature: SchnorrSigTarget,
    pub account_pk: QuinticExtensionTarget,
    pub tx_hash: QuinticExtensionTarget,
    pub instruction_type: Target,
    pub account: AccountTarget,
    pub sub_account_index: Target,
}

impl TxTypeTargets {
    pub fn new(builder: &mut Builder, tx_type: Target) -> Self {
        let tx_type_empty = builder.constant(F::from_canonical_u8(TX_TYPE_EMPTY));

        let tx_type_l1_deposit = builder.constant(F::from_canonical_u8(TX_TYPE_L1_DEPOSIT));
        let tx_type_l1_change_pub_key =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_CHANGE_PUB_KEY));
        let tx_type_l1_create_market =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_CREATE_MARKET));
        let tx_type_l1_update_market =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_UPDATE_MARKET));
        let tx_type_l1_cancel_all_orders =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_CANCEL_ALL_ORDERS));
        let tx_type_l1_withdraw = builder.constant(F::from_canonical_u8(TX_TYPE_L1_WITHDRAW));
        let tx_type_l1_create_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_CREATE_ORDER));
        let tx_type_l1_burn_shares = builder.constant(F::from_canonical_u8(TX_TYPE_L1_BURN_SHARES));
        let tx_type_l1_register_asset =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_REGISTER_ASSET));
        let tx_type_l1_update_asset =
            builder.constant(F::from_canonical_u8(TX_TYPE_L1_UPDATE_ASSET));

        let tx_type_l2_change_pub_key =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CHANGE_PUB_KEY));
        let tx_type_l2_create_sub_account =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_SUB_ACCOUNT));
        let tx_type_l2_create_public_pool =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_PUBLIC_POOL));
        let tx_type_l2_update_public_pool =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_PUBLIC_POOL));
        let tx_type_l2_transfer = builder.constant(F::from_canonical_u8(TX_TYPE_L2_TRANSFER));
        let tx_type_l2_withdraw = builder.constant(F::from_canonical_u8(TX_TYPE_L2_WITHDRAW));
        let tx_type_l2_create_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_ORDER));
        let tx_type_l2_cancel_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CANCEL_ORDER));
        let tx_type_l2_cancel_all_orders =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CANCEL_ALL_ORDERS));
        let tx_type_l2_modify_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_MODIFY_ORDER));
        let tx_type_l2_mint_shares = builder.constant(F::from_canonical_u8(TX_TYPE_L2_MINT_SHARES));
        let tx_type_l2_burn_shares = builder.constant(F::from_canonical_u8(TX_TYPE_L2_BURN_SHARES));
        let tx_type_l2_update_leverage =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_LEVERAGE));
        let tx_type_l2_create_grouped_orders =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_CREATE_GROUPED_ORDERS));
        let tx_type_l2_update_margin =
            builder.constant(F::from_canonical_u8(TX_TYPE_L2_UPDATE_MARGIN));

        let tx_type_internal_claim_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_CLAIM_ORDER));
        let tx_type_internal_cancel_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_CANCEL_ORDER));
        let tx_type_internal_deleverage =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_DELEVERAGE));
        let tx_type_internal_exit_position =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_EXIT_POSITION));
        let tx_type_internal_cancel_all_orders =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_CANCEL_ALL_ORDERS));
        let tx_type_internal_liquidate_position =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_LIQUIDATE_POSITION));
        let tx_type_internal_create_order =
            builder.constant(F::from_canonical_u8(TX_TYPE_INTERNAL_CREATE_ORDER));

        let is_empty = builder.is_equal(tx_type, tx_type_empty);

        let is_l1_deposit = builder.is_equal(tx_type, tx_type_l1_deposit);
        let is_l1_change_pub_key = builder.is_equal(tx_type, tx_type_l1_change_pub_key);
        let is_l1_create_market = builder.is_equal(tx_type, tx_type_l1_create_market);
        let is_l1_update_market = builder.is_equal(tx_type, tx_type_l1_update_market);
        let is_l1_cancel_all_orders = builder.is_equal(tx_type, tx_type_l1_cancel_all_orders);
        let is_l1_withdraw = builder.is_equal(tx_type, tx_type_l1_withdraw);
        let is_l1_create_order = builder.is_equal(tx_type, tx_type_l1_create_order);
        let is_l1_burn_shares = builder.is_equal(tx_type, tx_type_l1_burn_shares);
        let is_l1_register_asset = builder.is_equal(tx_type, tx_type_l1_register_asset);
        let is_l1_update_asset = builder.is_equal(tx_type, tx_type_l1_update_asset);

        let is_l2_change_pub_key = builder.is_equal(tx_type, tx_type_l2_change_pub_key);
        let is_l2_create_sub_account = builder.is_equal(tx_type, tx_type_l2_create_sub_account);
        let is_l2_create_public_pool = builder.is_equal(tx_type, tx_type_l2_create_public_pool);
        let is_l2_update_public_pool = builder.is_equal(tx_type, tx_type_l2_update_public_pool);
        let is_l2_transfer = builder.is_equal(tx_type, tx_type_l2_transfer);
        let is_l2_withdraw = builder.is_equal(tx_type, tx_type_l2_withdraw);
        let is_l2_create_order = builder.is_equal(tx_type, tx_type_l2_create_order);
        let is_l2_cancel_order = builder.is_equal(tx_type, tx_type_l2_cancel_order);
        let is_l2_cancel_all_orders = builder.is_equal(tx_type, tx_type_l2_cancel_all_orders);
        let is_l2_modify_order = builder.is_equal(tx_type, tx_type_l2_modify_order);
        let is_l2_mint_shares = builder.is_equal(tx_type, tx_type_l2_mint_shares);
        let is_l2_burn_shares = builder.is_equal(tx_type, tx_type_l2_burn_shares);
        let is_l2_update_leverage = builder.is_equal(tx_type, tx_type_l2_update_leverage);
        let is_l2_create_grouped_orders =
            builder.is_equal(tx_type, tx_type_l2_create_grouped_orders);
        let is_l2_update_margin = builder.is_equal(tx_type, tx_type_l2_update_margin);

        let is_internal_claim_order = builder.is_equal(tx_type, tx_type_internal_claim_order);
        let is_internal_cancel_order = builder.is_equal(tx_type, tx_type_internal_cancel_order);
        let is_internal_deleverage = builder.is_equal(tx_type, tx_type_internal_deleverage);
        let is_internal_exit_position = builder.is_equal(tx_type, tx_type_internal_exit_position);
        let is_internal_cancel_all_orders =
            builder.is_equal(tx_type, tx_type_internal_cancel_all_orders);
        let is_internal_liquidate_position =
            builder.is_equal(tx_type, tx_type_internal_liquidate_position);
        let is_internal_create_order = builder.is_equal(tx_type, tx_type_internal_create_order);

        // Using BoolTarget::new_unsafe is safe here because each target is guaranteed to be a boolean and
        // we are validating that their sum (valid_tx_type) is true(one). Because there are less than field order
        // elements, sum can not overflow.
        let is_valid_tx_type = BoolTarget::new_unsafe(builder.add_many(vec![
            is_empty.target,
            is_l1_deposit.target,
            is_l1_create_market.target,
            is_l1_update_market.target,
            is_l1_cancel_all_orders.target,
            is_l1_withdraw.target,
            is_l1_create_order.target,
            is_l1_change_pub_key.target,
            is_l1_burn_shares.target,
            is_l1_register_asset.target,
            is_l1_update_asset.target,
            is_l2_change_pub_key.target,
            is_l2_create_sub_account.target,
            is_l2_create_public_pool.target,
            is_l2_update_public_pool.target,
            is_l2_transfer.target,
            is_l2_withdraw.target,
            is_l2_create_order.target,
            is_l2_cancel_order.target,
            is_l2_cancel_all_orders.target,
            is_l2_modify_order.target,
            is_l2_mint_shares.target,
            is_l2_burn_shares.target,
            is_l2_update_leverage.target,
            is_l2_create_grouped_orders.target,
            is_l2_update_margin.target,
            is_internal_claim_order.target,
            is_internal_cancel_order.target,
            is_internal_deleverage.target,
            is_internal_exit_position.target,
            is_internal_cancel_all_orders.target,
            is_internal_liquidate_position.target,
            is_internal_create_order.target,
        ]));
        builder.assert_true(is_valid_tx_type);

        let is_layer2 = BoolTarget::new_unsafe(builder.add_many(vec![
            is_l2_change_pub_key.target,
            is_l2_create_sub_account.target,
            is_l2_create_public_pool.target,
            is_l2_update_public_pool.target,
            is_l2_transfer.target,
            is_l2_withdraw.target,
            is_l2_create_order.target,
            is_l2_cancel_order.target,
            is_l2_cancel_all_orders.target,
            is_l2_modify_order.target,
            is_l2_mint_shares.target,
            is_l2_burn_shares.target,
            is_l2_update_leverage.target,
            is_l2_create_grouped_orders.target,
            is_l2_update_margin.target,
        ]));

        let is_layer1 = BoolTarget::new_unsafe(builder.add_many(vec![
            is_l1_deposit.target,
            is_l1_create_market.target,
            is_l1_update_market.target,
            is_l1_cancel_all_orders.target,
            is_l1_withdraw.target,
            is_l1_create_order.target,
            is_l1_change_pub_key.target,
            is_l1_burn_shares.target,
            is_l1_register_asset.target,
            is_l1_update_asset.target,
        ]));

        let is_non_internal =
            BoolTarget::new_unsafe(builder.add_many(vec![is_layer1.target, is_layer2.target]));

        let is_sub_account_tx = BoolTarget::new_unsafe(builder.add_many(vec![
            is_l2_create_sub_account.target,
            is_l2_create_public_pool.target,
            is_l2_update_public_pool.target,
            is_l2_mint_shares.target,
            is_l2_burn_shares.target,
            is_l1_burn_shares.target,
        ]));

        let is_dms_blocked_tx = BoolTarget::new_unsafe(builder.add_many(vec![
            is_l2_create_order.target,
            is_l2_modify_order.target,
            is_l2_create_grouped_orders.target,
        ]));

        TxTypeTargets {
            is_empty,
            is_l1_deposit,
            is_l1_create_market,
            is_l1_update_market,
            is_l1_cancel_all_orders,
            is_l1_withdraw,
            is_l1_create_order,
            is_l1_change_pub_key,
            is_l1_burn_shares,
            is_l1_register_asset,
            is_l1_update_asset,

            is_l2_change_pub_key,
            is_l2_create_sub_account,
            is_l2_create_public_pool,
            is_l2_update_public_pool,
            is_l2_transfer,
            is_l2_withdraw,
            is_l2_create_order,
            is_l2_cancel_order,
            is_l2_cancel_all_orders,
            is_l2_modify_order,
            is_l2_mint_shares,
            is_l2_burn_shares,
            is_l2_update_leverage,
            is_l2_create_grouped_orders,
            is_l2_update_margin,

            is_internal_claim_order,
            is_internal_cancel_order,
            is_internal_deleverage,
            is_internal_exit_position,
            is_internal_cancel_all_orders,
            is_internal_liquidate_position,
            is_internal_create_order,

            is_layer1,
            is_layer2,
            is_non_internal,
            is_sub_account_tx,
            is_dms_blocked_tx,
        }
    }

    pub fn verify(&self, builder: &mut Builder, verify_inputs: &TxTypeVerifyTargets) {
        self.verify_l2_tx(builder, verify_inputs);

        // For L1 and L2 transactions, next instruction should be EXECUTE_TRANSACTION.
        // Internal transactions are validated in their corresponding validation functions.
        let execute_transaction_instruction =
            builder.constant(F::from_canonical_u8(EXECUTE_TRANSACTION));
        builder.conditional_assert_eq(
            self.is_non_internal,
            verify_inputs.instruction_type,
            execute_transaction_instruction,
        );

        // For transactions that performs actions on sub-accounts, the second account index must be a valid sub-account index.
        // Since this check is performed here, transaction executors do not need to peform this check again.
        let min_sub_account_index = builder.constant_i64(MIN_SUB_ACCOUNT_INDEX);
        builder.conditional_assert_lte(
            self.is_sub_account_tx,
            min_sub_account_index,
            verify_inputs.sub_account_index,
            ACCOUNT_INDEX_BITS,
        );
    }

    pub fn verify_l2_tx(&self, builder: &mut Builder, verify_inputs: &TxTypeVerifyTargets) {
        // Verify set transaction expiry.
        builder.register_range_check(verify_inputs.expired_at, TIMESTAMP_BITS);
        builder.conditional_assert_lt(
            self.is_layer2,
            verify_inputs.block_created_at,
            verify_inputs.expired_at,
            TIMESTAMP_BITS,
        );

        // Verify api key nonce.
        builder.conditional_assert_eq(
            self.is_layer2,
            verify_inputs.nonce,
            verify_inputs.api_key_before_nonce,
        );

        // Do not allow empty public keys when executing L2 transactions except for L2 change pubkey.
        let pk_check = builder.and_not(self.is_layer2, self.is_l2_change_pub_key);
        builder.conditional_assert_not_zero_quintic_ext(pk_check, verify_inputs.account_pk);

        // Verify transactions signature except for L2 change pubkey with empty new pubkey.
        let is_new_pubkey_zero = builder.is_zero_quintic_ext(verify_inputs.account_pk);
        let no_signature_check = builder.and(self.is_l2_change_pub_key, is_new_pubkey_zero);
        let signature_check = builder.and_not(self.is_layer2, no_signature_check);
        verify_schnorr_signature_conditional_circuit(
            builder,
            signature_check,
            &verify_inputs.account_pk,
            &verify_inputs.tx_hash,
            &verify_inputs.signature,
        );

        // If dead man's switch is supposed to be triggered for transaction initiator,
        // do not allow user to sending dms blocked transactions.
        // Protocol needs to queue internal cancel all transaction to clear the dead man's switch.
        let should_dms_be_triggered = verify_inputs
            .account
            .should_dms_be_triggered(builder, verify_inputs.block_created_at);
        builder.conditional_assert_false(self.is_dms_blocked_tx, should_dms_be_triggered);

        // If transaction initiator is the treasury, only L2 withdrawals are allowed.
        let treasury_index = builder.constant_usize(TREASURY_ACCOUNT_INDEX);
        let is_treasury = builder.is_equal(verify_inputs.account.account_index, treasury_index);
        let is_valid_treasury_tx = builder.multi_or(&[
            self.is_l2_transfer,
            self.is_l2_withdraw,
            self.is_l2_create_order,
            self.is_l2_cancel_order,
            self.is_l2_cancel_all_orders,
            self.is_l2_modify_order,
        ]);
        let check_treasury_tx = builder.and(is_treasury, self.is_layer2);
        builder.conditional_assert_true(check_treasury_tx, is_valid_treasury_tx);

        // If sender is the insurance fund operator, only public pool related transactions,
        // transfers and withdrawals are allowed.
        let insurance_fund_operator_index =
            builder.constant_usize(INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX);
        let is_insurance_fund_operator = builder.is_equal(
            verify_inputs.account.account_index,
            insurance_fund_operator_index,
        );
        let is_valid_insurance_fund_operator_tx = builder.multi_or(&[
            self.is_l2_transfer,
            self.is_l2_withdraw,
            self.is_l2_burn_shares,
            self.is_l2_mint_shares,
            self.is_l2_create_public_pool,
            self.is_l2_update_public_pool,
        ]);
        let check_insurance_fund_operator_tx =
            builder.and(is_insurance_fund_operator, self.is_layer2);
        builder.conditional_assert_true(
            check_insurance_fund_operator_tx,
            is_valid_insurance_fund_operator_tx,
        );

        // For sub-accounts, sub-account creation transactions are not allowed.
        let sub_account_type = builder.constant_from_u8(SUB_ACCOUNT_TYPE);
        let is_sub_account = builder.is_equal(verify_inputs.account.account_type, sub_account_type);
        let is_valid_sub_account_tx = builder.multi_or(&[
            self.is_l2_change_pub_key,
            self.is_l2_transfer,
            self.is_l2_withdraw,
            self.is_l2_create_order,
            self.is_l2_cancel_order,
            self.is_l2_cancel_all_orders,
            self.is_l2_modify_order,
            self.is_l2_burn_shares,
            self.is_l2_mint_shares,
            self.is_l2_update_leverage,
            self.is_l2_create_grouped_orders,
            self.is_l2_update_margin,
        ]);
        let check_sub_account_tx = builder.and(is_sub_account, self.is_layer2);
        builder.conditional_assert_true(check_sub_account_tx, is_valid_sub_account_tx);

        // For public pools and insurance funds, only position and api key management transactions are allowed.
        // Note that public pool and insurance fund accounts currently do not support isolated margin.
        let public_pool_type = builder.constant_from_u8(PUBLIC_POOL_ACCOUNT_TYPE);
        let is_public_pool = builder.is_equal(verify_inputs.account.account_type, public_pool_type);
        let insurance_fund_type = builder.constant_from_u8(INSURANCE_FUND_ACCOUNT_TYPE);
        let is_insurance_fund =
            builder.is_equal(verify_inputs.account.account_type, insurance_fund_type);
        let is_pool_account = builder.or(is_public_pool, is_insurance_fund);
        let is_frozen_status = builder.is_equal_constant(
            verify_inputs.account.public_pool_info.status,
            FROZEN_PUBLIC_POOL as u64,
        );
        let is_frozen_pool = builder.and(is_pool_account, is_frozen_status); // Status == 1, is frozen
        let is_active_pool = builder.and_not(is_pool_account, is_frozen_status); // Status == 0, is active
        let is_valid_active_pool_tx = builder.multi_or(&[
            self.is_l2_change_pub_key,
            self.is_l2_create_order,
            self.is_l2_cancel_order,
            self.is_l2_cancel_all_orders,
            self.is_l2_modify_order,
            self.is_l2_update_leverage,
            self.is_l2_create_grouped_orders,
        ]);
        let is_valid_frozen_pool_tx =
            builder.multi_or(&[self.is_l2_transfer, self.is_l2_change_pub_key]);

        let check_active_pool_tx = builder.and(is_active_pool, self.is_layer2);
        builder.conditional_assert_true(check_active_pool_tx, is_valid_active_pool_tx);
        let check_frozen_pool_tx = builder.and(is_frozen_pool, self.is_layer2);
        builder.conditional_assert_true(check_frozen_pool_tx, is_valid_frozen_pool_tx);
    }
}
