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
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bigint::unsafe_big::{CircuitBuilderUnsafeBig, UnsafeBigTarget};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::circuit_logger::CircuitBuilderLogging;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::deserializers;
use crate::eddsa::gadgets::curve::PartialWitnessCurve;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::types::account_margined_asset::{
    AccountMarginedAsset, AccountMarginedAssetTarget, AccountMarginedAssetTargetWitness,
    random_access_account_margined_asset_target,
};
use crate::types::account_position::{
    AccountPosition, AccountPositionTarget, AccountPositionTargetWitness,
};
use crate::types::approved_integrator::{
    ApprovedIntegrator, ApprovedIntegratorTarget, ApprovedIntegratorWitness,
};
use crate::types::asset::{AssetTarget, is_universal_asset};
use crate::types::config::{BIG_U96_LIMBS, BIG_U128_LIMBS, BIG_U160_LIMBS, Builder};
use crate::types::constants::*;
use crate::types::margined_asset::MarginedAssetTarget;
use crate::types::pending_unlock::{
    PendingUnlock, PendingUnlockTarget, PendingUnlockWitness, select_pending_unlock_target,
};
use crate::types::public_pool::{
    PublicPoolInfo, PublicPoolInfoTarget, PublicPoolInfoWitness, PublicPoolShare,
    PublicPoolShareTarget, PublicPoolShareWitness, select_public_pool_share_target,
};
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
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

    #[serde(rename = "ma")]
    #[serde(deserialize_with = "deserializers::margined_account_assets")]
    pub margined_assets: [AccountMarginedAsset; MARGINED_ASSET_LIST_SIZE], // 96 bits

    #[serde(rename = "ab")]
    #[serde(deserialize_with = "deserializers::aggregated_balances")]
    pub aggregated_balances: [BigInt; NB_ASSETS_PER_TX], // 96 bits

    #[serde(rename = "ap")]
    #[serde(deserialize_with = "deserializers::positions")]
    pub positions: [AccountPosition; POSITION_LIST_SIZE],

    #[serde(rename = "pwi", default)]
    pub pending_unlocks: [PendingUnlock; MAX_PENDING_UNLOCKS],

    #[serde(rename = "aiw", default)]
    pub approved_integrators: [ApprovedIntegrator; MAX_APPROVED_INTEGRATORS],

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
            margined_assets: [AccountMarginedAsset::empty(); MARGINED_ASSET_LIST_SIZE],
            aggregated_balances: [BigInt::ZERO; NB_ASSETS_PER_TX],
            positions: array::from_fn(|_| AccountPosition::default()),
            public_pool_shares: array::from_fn(|_| PublicPoolShare::default()),
            pending_unlocks: array::from_fn(|_| PendingUnlock::default()),
            approved_integrators: array::from_fn(|_| ApprovedIntegrator::default()),
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

    pub margined_assets: [AccountMarginedAssetTarget; MARGINED_ASSET_LIST_SIZE],
    pub aggregated_balances: [BigIntTarget; NB_ASSETS_PER_TX],
    pub positions: [AccountPositionTarget; POSITION_LIST_SIZE],

    pub approved_integrators: [ApprovedIntegratorTarget; MAX_APPROVED_INTEGRATORS],
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
        Self {
            master_account_index: Target::default(),
            account_index: Target::default(),
            l1_address: BigUintTarget::default(),
            account_type: Target::default(),
            account_trading_mode: Target::default(),

            margined_assets: array::from_fn(|_| AccountMarginedAssetTarget::default()),
            aggregated_balances: array::from_fn(|_| BigIntTarget::default()),

            positions: array::from_fn(|_| AccountPositionTarget::default()),

            approved_integrators: array::from_fn(|_| ApprovedIntegratorTarget::default()),
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
        Self {
            master_account_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),
            account_trading_mode: builder.add_virtual_target(),

            margined_assets: array::from_fn(|_| AccountMarginedAssetTarget::new(builder)), // safe because it is read from the state using merkle proofs
            aggregated_balances: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS) // safe because it is read from the state using merkle proofs
            }),

            positions: array::from_fn(|_| AccountPositionTarget::new(builder)),

            approved_integrators: array::from_fn(|_| ApprovedIntegratorTarget::new(builder)),
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
        Self {
            master_account_index: builder.add_virtual_target(),
            account_index: builder.add_virtual_target(),
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because it is read from the state using merkle proofs
            account_type: builder.add_virtual_target(),
            account_trading_mode: builder.add_virtual_target(),

            margined_assets: array::from_fn(|_| AccountMarginedAssetTarget::new(builder)), // safe because it is read from the state using merkle proofs
            aggregated_balances: array::from_fn(|_| {
                builder.add_virtual_bigint_target_unsafe(BIG_U96_LIMBS) // safe because it is read from the state using merkle proofs
            }),

            positions: array::from_fn(|_| AccountPositionTarget::default()), // Unused for fee accounts

            approved_integrators: array::from_fn(|_| ApprovedIntegratorTarget::new(builder)),
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
        let mut total_unlock_amount = UnsafeBigTarget {
            limbs: vec![builder.zero(); BIG_U96_LIMBS],
        };
        for pu in self.pending_unlocks.iter() {
            let unsafe_amount = builder.unsafe_big_from_biguint(&pu.amount);
            total_unlock_amount = builder.add_unsafe_big(&total_unlock_amount, &unsafe_amount);
        }
        builder.unsafe_big32_to_biguint(&total_unlock_amount, BIG_U96_LIMBS)
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

    pub fn print(&self, builder: &mut Builder, tag: &str) {
        builder.println(
            self.master_account_index,
            &format!("{}: master_account_index", tag),
        );
        builder.println(self.account_index, &format!("{}: account_index", tag));
        builder.println_biguint(&self.l1_address, &format!("{}: l1_address", tag));
        builder.println(self.account_type, &format!("{}: account_type", tag));
        for (i, margined_asset) in self.margined_assets.iter().enumerate() {
            margined_asset.print(builder, &format!("{}: margined_asset_{}", tag, i));
        }
        builder.println_hash_out(
            &self.aggregated_balances_root,
            &format!("{}: aggregated_balances_root", tag),
        );

        for (i, agg_bal) in self.aggregated_balances.iter().enumerate() {
            builder.println_bigint(agg_bal, &format!("{}: aggregated_balance_{}", tag, i));
        }

        for i in 0..MAX_APPROVED_INTEGRATORS {
            self.approved_integrators[i]
                .print(builder, &format!("{}: approved_integrator_{}", tag, i));
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

    pub fn get_margined_asset_balance_const(&self, margin_index: usize) -> BigIntTarget {
        self.margined_assets[margin_index].balance.clone()
    }

    pub fn get_margined_asset_balances(
        builder: &mut Builder,
        accounts: &[AccountTarget; NB_ACCOUNTS_PER_TX],
        assets: &[AssetTarget; NB_ASSETS_PER_TX],
        first_margin_index: Target,
    ) -> [[AccountMarginedAssetTarget; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX] {
        let second_margin_index = assets[1].margin_index(builder);

        array::from_fn(|i| {
            let mut v = accounts[i].margined_assets.to_vec();
            v.push(AccountMarginedAssetTarget::empty(builder));
            [
                random_access_account_margined_asset_target(builder, first_margin_index, &v),
                random_access_account_margined_asset_target(builder, second_margin_index, &v),
            ]
        })
    }

    /// Return cross or strategy collateral based on account type. Assumes `strategy_index` isn't nil.
    pub fn get_relevant_usdc_collateral(
        &self,
        builder: &mut Builder,
        strategy_index: Target,
    ) -> BigIntTarget {
        let strategy_balance = self
            .public_pool_info
            .get_strategy_balance(builder, strategy_index);
        let is_insurance_fund =
            builder.is_equal_constant(self.account_type, INSURANCE_FUND_ACCOUNT_TYPE as u64);
        builder.select_bigint(
            is_insurance_fund,
            &strategy_balance,
            &self.get_margined_asset_balance_const(USDC_MARGIN_ASSET_INDEX),
        )
    }

    /// Short hand for Perps USDC delta update
    /// Because USDC spot balance is always zero for unified accounts, we can directly apply delta to margin balance
    /// ignoring "use spot balance for negative delta" logic in [`AccountTarget::apply_asset_delta`] function
    pub fn apply_collateral_delta(
        &self,
        builder: &mut Builder,
        is_enabled: BoolTarget,
        collateral_delta: &BigIntTarget,
        strategy_balance: &mut BigIntTarget,
        margin_balance: &mut BigIntTarget,
    ) {
        let collateral_delta = builder.mul_bigint_by_bool(collateral_delta, is_enabled);
        *margin_balance =
            builder.add_bigint_non_carry(margin_balance, &collateral_delta, BIG_U96_LIMBS);

        let is_insurance_fund =
            builder.is_equal_constant(self.account_type, INSURANCE_FUND_ACCOUNT_TYPE as u64);
        let strategy_delta = builder.mul_bigint_by_bool(&collateral_delta, is_insurance_fund);
        *strategy_balance =
            builder.add_bigint_non_carry(strategy_balance, &strategy_delta, BIG_U96_LIMBS);
    }

    /// Caller's responsibility to ensure delta is positive, caller account is UTA and asset is margined
    /// For inflows, prioritize adding balance to margin balance for margin enabled assets
    pub fn get_uta_margin_and_balance_inflow_deltas(
        builder: &mut Builder,
        is_enabled: BoolTarget,
        asset_delta: &BigUintTarget,
        margined_asset_balance: &BigIntTarget,
        margined_asset: &MarginedAssetTarget,
    ) -> (BigUintTarget, BigUintTarget) {
        let zero_biguint = builder.zero_biguint();

        let asset_delta = builder.mul_biguint_by_bool(asset_delta, is_enabled);

        // Account for supply caps
        let is_caps_enabled = {
            let is_universal_asset = is_universal_asset(builder, margined_asset.asset_index);
            builder.not(is_universal_asset)
        };
        let remaining_supply =
            margined_asset.get_remaining_supply_cap(builder, margined_asset_balance);
        let is_remaining_lt_delta = builder.is_lt_biguint(&remaining_supply, &asset_delta);
        let cap_flag = builder.multi_and(&[is_caps_enabled, is_remaining_lt_delta]);

        let margined_balance_delta =
            builder.select_biguint(cap_flag, &remaining_supply, &asset_delta);

        let unmargined_balance_delta = builder.sub_biguint(&asset_delta, &remaining_supply);
        let balance_delta =
            builder.select_biguint(cap_flag, &unmargined_balance_delta, &zero_biguint);

        (margined_balance_delta, balance_delta)
    }

    /// Caller's responsibility to ensure delta is negative, caller account is UTA and asset is margined
    /// For outflows, prioritize subtracting balance from spot balance for margin enabled assets to avoid unnecessary margin calls
    pub fn get_uta_margin_and_balance_outflow_deltas(
        builder: &mut Builder,
        is_enabled: BoolTarget,
        asset_delta: &BigIntTarget,
        asset_balance: &BigUintTarget,
    ) -> (BigIntTarget, BigIntTarget) {
        let zero_bigint = builder.zero_bigint();

        let asset_delta = builder.mul_bigint_by_bool(asset_delta, is_enabled);

        // Take from margin balance when balance isn't enough
        let is_insufficient_balance = builder.is_lt_biguint(asset_balance, &asset_delta.abs);

        let excess_delta_abs = builder.sub_biguint(&asset_delta.abs, asset_balance);
        let excess_delta = builder.negative_biguint(&excess_delta_abs);
        let margined_balance_delta =
            builder.select_bigint(is_insufficient_balance, &excess_delta, &zero_bigint);

        let neg_balance = builder.negative_biguint(asset_balance);
        let balance_delta =
            builder.select_bigint(is_insufficient_balance, &neg_balance, &asset_delta);

        (margined_balance_delta, balance_delta)
    }

    pub fn apply_asset_delta_raw(
        builder: &mut Builder,
        is_enabled: BoolTarget,
        product_type: Target,
        asset_index: Target,
        margined_asset: &mut MarginedAssetTarget,
        asset_balance: &mut BigUintTarget,
        asset_delta: &BigIntTarget,
        margin_balance: &mut BigIntTarget,

        can_be_universal_asset: bool,
    ) {
        let asset_delta = builder.mul_bigint_by_bool(asset_delta, is_enabled);

        let is_spot = BoolTarget::new_unsafe(product_type);
        let is_perps = builder.not(is_spot);

        // Spot
        {
            let spot_delta = builder.mul_bigint_by_bool(&asset_delta, is_spot);
            let balance_bigint = builder.biguint_to_bigint(asset_balance);
            let new_spot_balance =
                builder.add_bigint_non_carry(&balance_bigint, &spot_delta, BIG_U96_LIMBS);
            let is_new_spot_balance_negative = builder.is_sign_negative(new_spot_balance.sign);
            builder.assert_false(is_new_spot_balance_negative);
            *asset_balance = new_spot_balance.abs;
        }

        // Perps
        {
            let perps_delta = builder.mul_bigint_by_bool(&asset_delta, is_perps);

            let caps_enabled = if can_be_universal_asset {
                let is_universal_asset = is_universal_asset(builder, asset_index);
                builder.not(is_universal_asset)
            } else {
                builder._true()
            };

            // Remaining supply cap check
            {
                let is_delta_positive = builder.is_sign_positive(perps_delta.sign);
                let remaining_supply_cap =
                    margined_asset.get_remaining_supply_cap(builder, margin_balance);
                let remaining_supply_flag = builder.multi_and(&[caps_enabled, is_delta_positive]);

                let is_remaining_supply_cap_lt_delta =
                    builder.is_lt_biguint(&remaining_supply_cap, &perps_delta.abs);
                builder.conditional_assert_false(
                    remaining_supply_flag,
                    is_remaining_supply_cap_lt_delta,
                );
            }

            // Apply margin delta
            *margin_balance =
                builder.add_bigint_non_carry(margin_balance, &perps_delta, BIG_U96_LIMBS);

            // Apply TSA delta
            let cap_delta = builder.mul_bigint_by_bool(&perps_delta, caps_enabled);
            let tsa_bigint = builder.biguint_to_bigint(&margined_asset.total_supplied_amount);
            let new_tsa = builder.add_bigint_non_carry(&tsa_bigint, &cap_delta, BIG_U96_LIMBS);
            let is_new_tsa_negative = builder.is_sign_negative(new_tsa.sign);
            builder.assert_false(is_new_tsa_negative); // Total supplied amount cannot be negative
            margined_asset.total_supplied_amount = new_tsa.abs;
        }
    }

    /// Call this function for negative part first when transferring, so that total supplied amount
    /// is not overflowed between operations.
    pub fn apply_asset_delta(
        builder: &mut Builder,
        is_enabled: BoolTarget,
        product_type: Target,
        asset_index: Target,
        margined_asset: &mut MarginedAssetTarget, // Has just enough fields for use
        is_asset_used_as_margin: BoolTarget,
        asset_delta: &BigIntTarget,
        is_account_unified: BoolTarget,
        is_insurance_fund: BoolTarget,
        asset_balance: &mut BigUintTarget,
        margin_balance: &mut BigIntTarget,
        strategy_balance: &mut BigIntTarget,
        allow_overflow: bool,
    ) -> BoolTarget {
        let zero_bigint = builder.zero_bigint();
        let mut is_spot_balance_valid = builder._true();
        let limb_count = if allow_overflow {
            BIG_U128_LIMBS
        } else {
            BIG_U96_LIMBS
        };

        let asset_delta = builder.mul_bigint_by_bool(asset_delta, is_enabled);

        let is_perps = builder.is_equal_constant(product_type, PRODUCT_TYPE_PERPS);

        let is_asset_universal = is_universal_asset(builder, asset_index);
        let is_asset_not_universal = builder.not(is_asset_universal);
        let is_account_simple = builder.not(is_account_unified);
        let is_account_unified_and_asset_not_used_as_margin =
            builder.and_not(is_account_unified, is_asset_used_as_margin);

        let is_account_simple_and_asset_non_universal_and_perps =
            builder.multi_and(&[is_account_simple, is_perps, is_asset_not_universal]);
        // Simple accounts can't have non-universal margined assets.
        builder.conditional_assert_false(
            is_enabled,
            is_account_simple_and_asset_non_universal_and_perps,
        );

        let is_account_unified_and_asset_is_not_margin_and_perps =
            builder.multi_and(&[is_account_unified_and_asset_not_used_as_margin, is_perps]);
        // For unified accounts, if asset is not margin enabled, it can't have perps balance
        builder.conditional_assert_false(
            is_enabled,
            is_account_unified_and_asset_is_not_margin_and_perps,
        );

        let is_delta_positive = builder.is_sign_positive(asset_delta.sign);

        let remaining_supply_cap = margined_asset.get_remaining_supply_cap(builder, margin_balance);
        let is_remaining_supply_cap_lt_delta =
            builder.is_lt_biguint(&remaining_supply_cap, &asset_delta.abs);
        let should_cap_margin_delta = builder.multi_and(&[
            is_enabled,
            is_asset_not_universal,
            is_delta_positive,
            is_remaining_supply_cap_lt_delta,
        ]);

        let positive_margin_delta = builder.select_biguint(
            should_cap_margin_delta,
            &remaining_supply_cap,
            &asset_delta.abs,
        );
        let (positive_spot_delta, borrow) =
            builder.try_sub_biguint(&asset_delta.abs, &positive_margin_delta);
        builder.conditional_assert_zero_u32(is_enabled, borrow);

        let spot_balance = asset_balance.clone();
        let negative_spot_delta = builder.min_biguint(&spot_balance, &asset_delta.abs);
        let (negative_margin_delta, borrow) =
            builder.try_sub_biguint(&asset_delta.abs, &negative_spot_delta);
        builder.conditional_assert_zero_u32(is_enabled, borrow);

        let positive_margin_delta = builder.biguint_to_bigint(&positive_margin_delta);
        let negative_margin_delta = builder.negative_biguint(&negative_margin_delta);
        let mut unified_margin_delta = builder.select_bigint(
            is_delta_positive,
            &positive_margin_delta,
            &negative_margin_delta,
        );
        unified_margin_delta = builder.select_bigint(
            is_account_unified_and_asset_not_used_as_margin,
            &zero_bigint,
            &unified_margin_delta,
        );
        let positive_spot_delta = builder.biguint_to_bigint(&positive_spot_delta);
        let negative_spot_delta = builder.negative_biguint(&negative_spot_delta);
        let mut unified_spot_delta = builder.select_bigint(
            is_delta_positive,
            &positive_spot_delta,
            &negative_spot_delta,
        );
        unified_spot_delta = builder.select_bigint(
            is_account_unified_and_asset_not_used_as_margin,
            &asset_delta,
            &unified_spot_delta,
        );

        let simple_margin_delta = builder.select_bigint(is_perps, &asset_delta, &zero_bigint);
        let simple_spot_delta = builder.select_bigint(is_perps, &zero_bigint, &asset_delta);

        let mut margin_delta = builder.select_bigint(
            is_account_unified,
            &unified_margin_delta,
            &simple_margin_delta,
        );
        let mut spot_delta =
            builder.select_bigint(is_account_unified, &unified_spot_delta, &simple_spot_delta);

        // Insurance fund spot: entire delta goes to margin_balance (and strategy for USDC),
        // nothing goes to spot balance, supply caps are skipped.
        let is_insurance_fund_spot = builder.and_not(is_insurance_fund, is_perps);
        margin_delta = builder.select_bigint(is_insurance_fund_spot, &asset_delta, &margin_delta);
        spot_delta = builder.select_bigint(is_insurance_fund_spot, &zero_bigint, &spot_delta);

        // Update margin balance
        *margin_balance = builder.add_bigint_non_carry(margin_balance, &margin_delta, limb_count);
        if !allow_overflow {
            let is_margin_balance_negative = builder.is_sign_negative(margin_balance.sign);
            let is_asset_not_universal_and_balance_negative =
                builder.and(is_asset_not_universal, is_margin_balance_negative);
            // Margin balance cannot be negative for non-universal assets
            builder
                .conditional_assert_false(is_enabled, is_asset_not_universal_and_balance_negative);
        }

        // Update spot balance
        let spot_balance_bigint = builder.biguint_to_bigint(&spot_balance);
        let new_spot_balance =
            builder.add_bigint_non_carry(&spot_balance_bigint, &spot_delta, limb_count);

        let is_new_spot_balance_negative = builder.is_sign_negative(new_spot_balance.sign);
        if !allow_overflow {
            // Spot balance can't be negative
            builder.conditional_assert_false(is_enabled, is_new_spot_balance_negative);
        } else {
            is_spot_balance_valid = builder.not(is_new_spot_balance_negative);
        }

        *asset_balance = builder.select_biguint(is_enabled, &new_spot_balance.abs, asset_balance);

        // Apply strategy delta for the margin delta portion
        let is_asset_usdc = builder.is_equal_constant(asset_index, USDC_ASSET_INDEX);
        let is_strategy_delta = builder.and(is_asset_usdc, is_insurance_fund);
        let strategy_delta = builder.mul_bigint_by_bool(&margin_delta, is_strategy_delta);
        *strategy_balance =
            builder.add_bigint_non_carry(strategy_balance, &strategy_delta, BIG_U96_LIMBS);

        // Apply margin balance total supplied amount (skip for IF spot — not counted towards supply caps)
        let count_supply = builder.and_not(is_asset_not_universal, is_insurance_fund_spot);
        let supply_delta = builder.mul_bigint_by_bool(&margin_delta, count_supply);
        let mut total_supplied_amount_big =
            builder.biguint_to_bigint(&margined_asset.total_supplied_amount);
        total_supplied_amount_big =
            builder.add_bigint_non_carry(&total_supplied_amount_big, &supply_delta, limb_count);
        let new_total_supplied_amount_is_negative =
            builder.is_sign_negative(total_supplied_amount_big.sign);
        builder.conditional_assert_false(is_enabled, new_total_supplied_amount_is_negative);
        margined_asset.total_supplied_amount = total_supplied_amount_big.abs;

        is_spot_balance_valid
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
        for i in 0..b.approved_integrators.len() {
            self.set_approved_integrator(&a.approved_integrators[i], &b.approved_integrators[i])?;
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
        for i in 0..MARGINED_ASSET_LIST_SIZE {
            self.set_account_margined_asset_target(&a.margined_assets[i], &b.margined_assets[i])?;
        }
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
