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

use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, WitnessBigInt};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::types::account_asset::AccountAssetTarget;
use crate::types::account_position::{
    AccountPosition, AccountPositionTarget, AccountPositionTargetWitness,
};
use crate::types::config::{BIG_U96_LIMBS, BIG_U160_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::pending_unlock::{
    PendingUnlock, PendingUnlockTarget, PendingUnlockWitness, select_pending_unlock_target,
};
use crate::types::public_pool::{
    PublicPoolInfo, PublicPoolInfoTarget, PublicPoolInfoWitness, PublicPoolShare,
    PublicPoolShareTarget, PublicPoolShareWitness, select_public_pool_share_target,
};
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

    #[serde(rename = "bm", default)]
    pub account_trading_mode: u8,

    #[serde(rename = "col")]
    #[serde(deserialize_with = "deserializers::int_to_bigint")]
    pub collateral: BigInt, // 96 bits

    #[serde(rename = "ab")]
    #[serde(deserialize_with = "deserializers::aggregated_balances")]
    pub aggregated_balances: [BigInt; NB_ASSETS_PER_TX], // 96 bits

    #[serde(rename = "ap")]
    #[serde(deserialize_with = "deserializers::positions")]
    pub positions: [AccountPosition; POSITION_LIST_SIZE],

    #[serde(rename = "pwi", default)]
    pub pending_unlocks: [PendingUnlock; MAX_PENDING_UNLOCKS],

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
            account_trading_mode: ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE,
            collateral: BigInt::ZERO,
            aggregated_balances: [BigInt::ZERO; NB_ASSETS_PER_TX],
            positions: array::from_fn(|_| AccountPosition::default()),
            public_pool_shares: array::from_fn(|_| PublicPoolShare::default()),
            pending_unlocks: array::from_fn(|_| PendingUnlock::default()),
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
    pub account_trading_mode: Target,

    pub collateral: BigIntTarget,
    pub aggregated_balances: [BigIntTarget; NB_ASSETS_PER_TX],
    pub positions: [AccountPositionTarget; POSITION_LIST_SIZE],

    pub pending_unlocks: [PendingUnlockTarget; MAX_PENDING_UNLOCKS],
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
            account_trading_mode: Target::default(),

            collateral: BigIntTarget::default(),
            aggregated_balances: array::from_fn(|_| BigIntTarget::default()),

            positions: array::from_fn(|_| AccountPositionTarget::default()),

            pending_unlocks: array::from_fn(|_| PendingUnlockTarget::default()),
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
            account_trading_mode: builder.add_virtual_target(),

            collateral: builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS), // safe because it is read from the state using merkle proofs
            aggregated_balances: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),

            positions: array::from_fn(|_| AccountPositionTarget::new(builder)),

            pending_unlocks: array::from_fn(|_| PendingUnlockTarget::new(builder)),
            public_pool_shares: array::from_fn(|_| PublicPoolShareTarget::new(builder)),
            public_pool_info: PublicPoolInfoTarget::new(builder),

            total_order_count: builder.add_virtual_target(),
            total_non_cross_order_count: builder.add_virtual_target(),
            cancel_all_time: builder.add_virtual_target(),

            api_key_root: builder.add_virtual_hash(),
            account_orders_root: builder.add_virtual_hash(),
            aggregated_balances_root: builder.add_virtual_hash(),
            asset_root: builder.add_virtual_hash(),

            // Unused for maker and taker accounts.
            partial_hash: builder.zero_hash_out(),
            partial_hash_for_pub_data: builder.zero_hash_out(),
        }
    }

    pub fn new_fee_account(builder: &mut Builder) -> Self {
        AccountTarget {
            master_account_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),
            account_trading_mode: builder.add_virtual_target(),

            collateral: builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS), // safe because it is read from the state using merkle proofs
            aggregated_balances: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS)
            }),

            positions: array::from_fn(|_| AccountPositionTarget::default()), // Unused for fee accounts

            pending_unlocks: array::from_fn(|_| PendingUnlockTarget::new(builder)),
            public_pool_shares: array::from_fn(|_| PublicPoolShareTarget::default()),
            public_pool_info: PublicPoolInfoTarget::new(builder),

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

    pub fn pop_pending_unlock(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
    ) -> PendingUnlockTarget {
        let to_be_popped = self.pending_unlocks[0].clone();
        for i in 0..MAX_PENDING_UNLOCKS - 1 {
            self.pending_unlocks[i] = select_pending_unlock_target(
                builder,
                is_enabled,
                &self.pending_unlocks[i + 1],
                &self.pending_unlocks[i],
            );
        }
        let empty_pending_unlock = PendingUnlockTarget::empty(builder);
        self.pending_unlocks[MAX_PENDING_UNLOCKS - 1] = select_pending_unlock_target(
            builder,
            is_enabled,
            &empty_pending_unlock,
            &self.pending_unlocks[MAX_PENDING_UNLOCKS - 1],
        );
        to_be_popped
    }

    pub fn get_total_unlock_amount(&self, builder: &mut Builder) -> BigUintTarget {
        let mut total_unlock_amount = builder.zero_biguint();
        for pu in self.pending_unlocks.iter() {
            total_unlock_amount =
                builder.add_biguint_non_carry(&total_unlock_amount, &pu.amount, BIG_U96_LIMBS);
        }
        total_unlock_amount
    }

    pub fn add_pending_unlock(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        pending_unlock: &PendingUnlockTarget,
    ) {
        let mut appended = builder.not(is_enabled);
        for i in 0..MAX_PENDING_UNLOCKS {
            let is_slot_empty = builder.is_zero_biguint(&self.pending_unlocks[i].amount);
            let flag = builder.and_not(is_slot_empty, appended);
            appended = builder.or(appended, flag);

            self.pending_unlocks[i] = select_pending_unlock_target(
                builder,
                flag,
                pending_unlock,
                &self.pending_unlocks[i],
            );
        }
        builder.conditional_assert_true(is_enabled, appended);
    }

    pub fn is_unified_mode(&self) -> BoolTarget {
        BoolTarget::new_unsafe(self.account_trading_mode)
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

    pub fn apply_pool_share_delta(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        pool_index: Target,
        share_delta: Target,      // Can be negative for burns
        entry_usdc_delta: Target, // Can be negative for burns
        entry_timestamp_delta: Target,
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
            entry_timestamp: entry_timestamp_delta,
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
            let add_to_entry_timestamp = builder.mul_bool(apply_delta, entry_timestamp_delta);
            self.public_pool_shares[i].share_amount =
                builder.add(self.public_pool_shares[i].share_amount, add_to_share_amount);
            self.public_pool_shares[i].principal_amount = builder.add(
                self.public_pool_shares[i].principal_amount,
                add_to_entry_usdc,
            );
            self.public_pool_shares[i].entry_timestamp = builder.add(
                self.public_pool_shares[i].entry_timestamp,
                add_to_entry_timestamp,
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

    pub fn are_assets_used_as_margin(
        builder: &mut Builder,
        account_assets: &[AccountAssetTarget; NB_ASSETS_PER_TX],
    ) -> (BoolTarget, BoolTarget) {
        let first_asset = &account_assets[0];
        let is_first_asset_usdc = builder.is_equal_constant(first_asset.index_0, USDC_ASSET_INDEX);
        let is_first_asset_used_as_margin =
            builder.is_equal_constant(first_asset.margin_mode, ACCOUNT_ASSET_MARGIN_MODE_ENABLED);
        let is_first_asset_used_as_collateral =
            builder.or(is_first_asset_usdc, is_first_asset_used_as_margin);

        let second_asset = &account_assets[1];
        let is_second_asset_usdc =
            builder.is_equal_constant(second_asset.index_0, USDC_ASSET_INDEX);
        let is_second_asset_used_as_margin =
            builder.is_equal_constant(second_asset.margin_mode, ACCOUNT_ASSET_MARGIN_MODE_ENABLED);
        let is_second_asset_used_as_collateral =
            builder.or(is_second_asset_usdc, is_second_asset_used_as_margin);

        (
            is_first_asset_used_as_collateral,
            is_second_asset_used_as_collateral,
        )
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

        for (i, agg_bal) in self.aggregated_balances.iter().enumerate() {
            builder.println_bigint(agg_bal, &format!("{}: aggregated_balance_{}", tag, i));
        }

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
        builder.println(
            self.account_trading_mode,
            &format!("{}: account_trading_mode", tag),
        );
        builder.println_hash_out(&self.api_key_root, &format!("{}: api_key_root", tag));
        builder.println_hash_out(
            &self.account_orders_root,
            &format!("{}: account_orders_root", tag),
        );

        self.public_pool_info.print(builder, tag);
    }
}

impl AccountTarget {
    /// Return cross or strategy collateral based on account type. Assumes `strategy_index` isn't nil.
    pub fn get_relevant_collateral(
        &self,
        builder: &mut Builder,
        strategy_index: Target,
    ) -> BigIntTarget {
        let strategy_balance = self
            .public_pool_info
            .get_strategy_balance(builder, strategy_index);
        let is_insurance_fund =
            builder.is_equal_constant(self.account_type, INSURANCE_FUND_ACCOUNT_TYPE as u64);
        builder.select_bigint(is_insurance_fund, &strategy_balance, &self.collateral)
    }

    /// Short hand for Perps USDC delta update
    pub fn apply_collateral_delta(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        collateral_delta: &BigIntTarget,
        strategy_balance: &mut BigIntTarget,
    ) {
        let _true = builder._true();
        AccountTarget::apply_asset_delta_const(
            builder,
            is_enabled,
            PRODUCT_TYPE_PERPS,
            self,
            None,
            _true,
            collateral_delta,
            strategy_balance,
        );
    }

    pub fn apply_asset_delta_const(
        builder: &mut Builder,
        is_enabled: BoolTarget,
        product_type: u64,
        account: &mut AccountTarget,
        account_asset: Option<&mut AccountAssetTarget>,
        is_asset_used_as_margin: BoolTarget,
        asset_delta: &BigIntTarget,
        strategy_balance: &mut BigIntTarget,
    ) {
        let asset_delta = builder.mul_bigint_by_bool(asset_delta, is_enabled);

        if product_type == PRODUCT_TYPE_PERPS {
            account.collateral =
                builder.add_bigint_non_carry(&account.collateral, &asset_delta, BIG_U96_LIMBS);
            account.apply_strategy_delta(builder, is_enabled, strategy_balance, &asset_delta);
            return;
        }

        let is_unified_and_asset_used_as_margin =
            builder.and(account.is_unified_mode(), is_asset_used_as_margin);
        let update_collateral = builder.and(is_enabled, is_unified_and_asset_used_as_margin);
        let update_asset_balance = builder.and_not(is_enabled, update_collateral);

        let account_asset =
            account_asset.expect("account asset must be provided for non-perps products");
        let account_asset_balance = builder.biguint_to_bigint(&account_asset.balance);

        let new_asset_balance =
            builder.add_bigint_non_carry(&account_asset_balance, &asset_delta, BIG_U96_LIMBS);
        let is_new_asset_balance_negative = builder.is_sign_negative(new_asset_balance.sign);
        builder.conditional_assert_false(update_asset_balance, is_new_asset_balance_negative); // Asset balance cannot be negative

        account_asset.balance = builder.select_biguint(
            update_asset_balance,
            &new_asset_balance.abs,
            &account_asset.balance,
        );

        let new_collateral =
            builder.add_bigint_non_carry(&account.collateral, &asset_delta, BIG_U96_LIMBS);
        account.collateral =
            builder.select_bigint(update_collateral, &new_collateral, &account.collateral);
        account.apply_strategy_delta(builder, update_collateral, strategy_balance, &asset_delta);
    }

    pub fn apply_asset_delta(
        builder: &mut Builder,
        is_enabled: BoolTarget,
        product_type: Target,
        account: &mut AccountTarget,
        account_asset: &mut AccountAssetTarget,
        is_asset_used_as_margin: BoolTarget,
        asset_delta: &BigIntTarget,
        strategy_balance: &mut BigIntTarget,
    ) {
        let asset_delta = builder.mul_bigint_by_bool(asset_delta, is_enabled);

        let is_spot = BoolTarget::new_unsafe(product_type);

        let is_account_isolated = builder.is_equal_constant(
            account.account_trading_mode,
            ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE as u64,
        );
        let is_asset_not_used_as_margin = builder.not(is_asset_used_as_margin);
        let is_account_isolated_or_asset_not_used_as_margin =
            builder.or(is_account_isolated, is_asset_not_used_as_margin);
        let update_asset_balance = builder.multi_and(&[
            is_enabled,
            is_spot,
            is_account_isolated_or_asset_not_used_as_margin,
        ]);

        let is_account_unified_and_asset_used_as_margin =
            builder.not(is_account_isolated_or_asset_not_used_as_margin);
        let is_perps_and_asset_used_as_margin = builder.and_not(is_asset_used_as_margin, is_spot);
        let mut update_collateral = builder.or(
            is_account_unified_and_asset_used_as_margin,
            is_perps_and_asset_used_as_margin,
        );
        update_collateral = builder.and(update_collateral, is_enabled);

        let account_asset_balance = builder.biguint_to_bigint(&account_asset.balance);

        let new_asset_balance =
            builder.add_bigint_non_carry(&account_asset_balance, &asset_delta, BIG_U96_LIMBS);
        let is_new_asset_balance_negative = builder.is_sign_negative(new_asset_balance.sign);
        builder.conditional_assert_false(update_asset_balance, is_new_asset_balance_negative); // Asset balance cannot be negative
        account_asset.balance = builder.select_biguint(
            update_asset_balance,
            &new_asset_balance.abs,
            &account_asset.balance,
        );

        let new_collateral =
            builder.add_bigint_non_carry(&account.collateral, &asset_delta, BIG_U96_LIMBS);
        account.collateral =
            builder.select_bigint(update_collateral, &new_collateral, &account.collateral);
        account.apply_strategy_delta(builder, update_collateral, strategy_balance, &asset_delta);
    }

    pub fn apply_strategy_delta(
        &mut self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        strategy_balance: &mut BigIntTarget,
        delta: &BigIntTarget,
    ) {
        let is_insurance_fund =
            builder.is_equal_constant(self.account_type, INSURANCE_FUND_ACCOUNT_TYPE as u64);
        let flag = builder.and(is_enabled, is_insurance_fund);

        let delta = builder.mul_bigint_by_bool(delta, flag);

        let new_balance = builder.add_bigint_non_carry(strategy_balance, &delta, BIG_U96_LIMBS);
        *strategy_balance = builder.select_bigint(flag, &new_balance, strategy_balance);
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
        self.set_target(
            a.account_trading_mode,
            F::from_canonical_u8(b.account_trading_mode),
        )?;
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
        self.set_public_pool_info(&a.public_pool_info, &b.public_pool_info)?;
        self.set_hash_target(a.api_key_root, b.api_key_root)?;
        self.set_hash_target(a.account_orders_root, b.account_orders_root)?;
        self.set_hash_target(a.asset_root, b.asset_root)?;
        self.set_hash_target(a.aggregated_balances_root, b.aggregated_balances_root)?;
        for i in 0..b.pending_unlocks.len() {
            self.set_pending_unlock(&a.pending_unlocks[i], &b.pending_unlocks[i])?;
        }

        Ok(())
    }
}
