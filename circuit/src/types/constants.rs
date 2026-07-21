// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::hash::hash_types::{HashOut, NUM_HASH_OUT_ELTS};

use super::config::{F, const_f};

pub const MAX_PREMIUM_SAMPLE_COUNT: usize = 60;

pub const TX_TYPE_BITS: usize = 8;

pub const TIMESTAMP_BITS: usize = 48;
pub const NONCE_BITS: usize = 48;
pub const MAX_NONCE: i64 = (1i64 << NONCE_BITS) - 1;
pub const MAX_SKIP_NONCE_CAP: i64 = MAX_NONCE / 2;

pub const MASTER_ACCOUNT_INDEX_BITS: usize = 47;
pub const ACCOUNT_INDEX_BITS: usize = 48;
pub const API_KEY_INDEX_BITS: usize = 8;
pub const PERPS_MARKET_INDEX_BITS: usize = 8;
pub const MAX_PERPS_MARKET_INDEX: usize = (1 << PERPS_MARKET_INDEX_BITS) - 2;

pub const MIN_SPOT_MARKET_INDEX: usize = 1 << 11;
pub const MAX_SPOT_MARKET_INDEX: usize = (1 << SPOT_MARKET_INDEX_BITS) - 2;
pub const SPOT_MARKET_INDEX_BITS: usize = 12;

pub const MARKET_TYPE_PERPS: u64 = 0;
pub const MARKET_TYPE_SPOT: u64 = 1;

pub const ROUTE_TYPE_PERPS: u64 = 0;
pub const ROUTE_TYPE_SPOT: u64 = 1;

pub const PRODUCT_TYPE_PERPS: u64 = 0;
pub const PRODUCT_TYPE_SPOT: u64 = 1;

pub const MARKET_INDEX_BITS: usize = 12;

pub const GLOBAL_SUPPLY_CAP_BITS: usize = 60;
pub const USER_SUPPLY_CAP_BITS: usize = 60;

pub const ORDER_NONCE_BITS: usize = 48;
pub const ORDER_BASE_AMOUNT_BITS: usize = 48;
pub const ORDER_PRICE_BITS: usize = 32;
pub const TRIGGER_PRICE_BITS: usize = 32;
pub const ORDER_SIZE_BITS: usize = 48;
pub const ORDER_QUOTE_SIZE_BITS: usize = 48;
pub const CLIENT_ORDER_INDEX_BITS: usize = 48;
pub const ORDER_INDEX_BITS: usize = 56;
pub const MAX_ORDER_QUOTE_AMOUNT: u64 = (1 << 48) - 1;
pub const MAX_ORDER_BASE_AMOUNT: u64 = (1 << ORDER_BASE_AMOUNT_BITS) - 1;
pub const MAX_ORDER_PRICE: u64 = (1 << 32) - 1;

pub const ASSET_PRICE_BITS: usize = 56;

pub const MIN_CLIENT_ORDER_INDEX: i64 = 1;
pub const MAX_CLIENT_ORDER_INDEX: i64 = (1 << 48) - 1;

pub const MIN_ORDER_INDEX: i64 = MAX_CLIENT_ORDER_INDEX + 1;
pub const MAX_ORDER_INDEX: i64 = (1 << 56) - 1;

pub const POSITION_SIZE_BITS: usize = 56;
pub const ENTRY_QUOTE_BITS: usize = 56;
pub const NORMALIZED_QUOTE_BITS: usize = 48;

pub const MAX_TRANSFER_BITS: usize = 60;
pub const MIN_PARTIAL_TRANSFER_AMOUNT: usize = 10000000; // 10 USDC
pub const TRANSFER_MEMO_BYTES: usize = 32;
pub const MAX_WITHDRAW_BITS: usize = MAX_TRANSFER_BITS;
pub const MIN_PARTIAL_WITHDRAW_AMOUNT: usize = 10000000; // 10 USDC

pub const FUNDING_RATE_PREFIX_SUM_BITS: usize = 62;
pub const BASE_SUM_BITS: usize = 63;
pub const QUOTE_SUM_BITS: usize = 63;

pub const COLLATERAL_BITS: usize = 96;

pub const EMPTY_ACCOUNT_DELTA_TREE_ROOT: HashOut<F> =
    EMPTY_DELTA_TREE_HASHES[ACCOUNT_MERKLE_LEVELS];

pub const EMPTY_POSITION_DELTA_TREE_ROOT: HashOut<F> = const_hash_out([
    5428970986623951092,
    515484187069299980,
    6723256903412060294,
    2762657640779953643,
]);

pub const EMPTY_API_KEY_TREE_ROOT: HashOut<F> = const_hash_out([
    5428970986623951092,
    515484187069299980,
    6723256903412060294,
    2762657640779953643,
]);

pub const EMPTY_ACCOUNT_ORDERS_TREE_ROOT: HashOut<F> = const_hash_out([
    6653728556450998073,
    18416610459432847463,
    14237760084957206251,
    750257369394168112,
]);

pub const EMPTY_ASSET_TREE_ROOT: HashOut<F> = const_hash_out([
    6844301749605691075,
    15901475079080112138,
    6459841461395793132,
    5448886425767967618,
]);

pub const EMPTY_ORDER_BOOK_TREE_ROOT: HashOut<F> = const_hash_out([
    2269038392415604357,
    1685606050090336416,
    10362812591542117265,
    18095351966852921172,
]);

pub const EMPTY_ACCOUNT_HASH: HashOut<F> = const_hash_out([
    16496574717886153423,
    6271885065761349139,
    16208258067531002960,
    18356876270155499168,
]);

/// Tx Types
pub const TX_TYPE_EMPTY: u8 = 0;

// L1
pub const TX_TYPE_L1_DEPOSIT: u8 = 1;
pub const TX_TYPE_L1_CHANGE_PUB_KEY: u8 = 2;
pub const TX_TYPE_L1_CREATE_MARKET: u8 = 3;
pub const TX_TYPE_L1_UPDATE_MARKET: u8 = 4;
pub const TX_TYPE_L1_CANCEL_ALL_ORDERS: u8 = 5;
pub const TX_TYPE_L1_WITHDRAW: u8 = 6;
pub const TX_TYPE_L1_CREATE_ORDER: u8 = 7;
pub const TX_TYPE_L1_BURN_SHARES: u8 = 30;
pub const TX_TYPE_L1_REGISTER_ASSET: u8 = 31;
pub const TX_TYPE_L1_UPDATE_ASSET: u8 = 32;
pub const TX_TYPE_L1_UNSTAKE_ASSETS: u8 = 37;
pub const TX_TYPE_L1_SET_SYSTEM_CONFIG: u8 = 38;

// L2
pub const TX_TYPE_L2_CHANGE_PUB_KEY: u8 = 8;
pub const TX_TYPE_L2_CREATE_SUB_ACCOUNT: u8 = 9;
pub const TX_TYPE_L2_CREATE_PUBLIC_POOL: u8 = 10;
pub const TX_TYPE_L2_UPDATE_PUBLIC_POOL: u8 = 11;
pub const TX_TYPE_L2_TRANSFER: u8 = 12;
pub const TX_TYPE_L2_WITHDRAW: u8 = 13;
pub const TX_TYPE_L2_CREATE_ORDER: u8 = 14;
pub const TX_TYPE_L2_CANCEL_ORDER: u8 = 15;
pub const TX_TYPE_L2_CANCEL_ALL_ORDERS: u8 = 16;
pub const TX_TYPE_L2_MODIFY_ORDER: u8 = 17;
pub const TX_TYPE_L2_MINT_SHARES: u8 = 18;
pub const TX_TYPE_L2_BURN_SHARES: u8 = 19;
pub const TX_TYPE_L2_UPDATE_LEVERAGE: u8 = 20;
pub const TX_TYPE_L2_CREATE_GROUPED_ORDERS: u8 = 28;
pub const TX_TYPE_L2_UPDATE_MARGIN: u8 = 29;
pub const TX_TYPE_L2_CREATE_STAKING_POOL: u8 = 33;
// pub const TX_TYPE_L2_UPDATE_STAKING_POOL: u8 = 34;
pub const TX_TYPE_L2_STAKE_ASSETS: u8 = 35;
pub const TX_TYPE_L2_UNSTAKE_ASSETS: u8 = 36;
pub const TX_TYPE_L2_FORCE_BURN_SHARES: u8 = 40;
pub const TX_TYPE_L2_UPDATE_ACCOUNT_CONFIG: u8 = 41;
pub const TX_TYPE_L2_UPDATE_ACCOUNT_ASSET_CONFIG: u8 = 42;
pub const TX_TYPE_L2_STRATEGY_TRANSFER: u8 = 43;
pub const TX_TYPE_L2_UPDATE_MARKET_CONFIG: u8 = 44;
pub const TX_TYPE_L2_APPROVE_INTEGRATOR: u8 = 45;
pub const TX_TYPE_L2_UPDATE_ASSET_CONFIG: u8 = 48;

// Internal
pub const TX_TYPE_INTERNAL_CLAIM_ORDER: u8 = 21;
pub const TX_TYPE_INTERNAL_CANCEL_ORDER: u8 = 22;
pub const TX_TYPE_INTERNAL_DELEVERAGE: u8 = 23;
pub const TX_TYPE_INTERNAL_EXIT_POSITION: u8 = 24;
pub const TX_TYPE_INTERNAL_CANCEL_ALL_ORDERS: u8 = 25;
pub const TX_TYPE_INTERNAL_LIQUIDATE_POSITION: u8 = 26;
pub const TX_TYPE_INTERNAL_CREATE_ORDER: u8 = 27;
pub const TX_TYPE_INTERNAL_PENDING_UNLOCK: u8 = 39;
pub const TX_TYPE_INTERNAL_INTEGRATOR_OPERATIONS: u8 = 46;
pub const TX_TYPE_INTERNAL_LIQUIDATE_SPOT: u8 = 47;

// Priority request pub data
pub const PRIORITY_PUB_DATA_TYPE_L1_DEPOSIT: u8 = 41;
pub const PRIORITY_PUB_DATA_TYPE_L1_CHANGE_PUB_KEY: u8 = 42;
pub const PRIORITY_PUB_DATA_TYPE_L1_CREATE_MARKET: u8 = 43;
pub const PRIORITY_PUB_DATA_TYPE_L1_UPDATE_MARKET: u8 = 44;
pub const PRIORITY_PUB_DATA_TYPE_L1_CANCEL_ALL_ORDERS: u8 = 45;
pub const PRIORITY_PUB_DATA_TYPE_L1_WITHDRAW: u8 = 46;
pub const PRIORITY_PUB_DATA_TYPE_L1_CREATE_ORDER: u8 = 47;
pub const PRIORITY_PUB_DATA_TYPE_L1_BURN_SHARES: u8 = 48;
pub const PRIORITY_PUB_DATA_TYPE_L1_REGISTER_ASSET: u8 = 49;
pub const PRIORITY_PUB_DATA_TYPE_L1_UPDATE_ASSET: u8 = 50;
pub const PRIORITY_PUB_DATA_TYPE_L1_UNSTAKE_ASSETS: u8 = 51;
pub const PRIORITY_PUB_DATA_TYPE_L1_SET_SYSTEM_CONFIG: u8 = 52;

// On Chain Log Pubdata
pub const ON_CHAIN_PUB_DATA_TYPE_WITHDRAW: u8 = 2;

pub const USDC_TO_COLLATERAL_MULTIPLIER: u32 = 1_000_000;
pub const LIT_EXTENSION_MULTIPLIER: u32 = 1_000_000;

pub const LIT_TO_MINT_SHARES_MULTIPLIER: u64 = 10;
pub const USDC_TO_LIT_CONVERSION_RATE: u64 = ONE_LIT / ONE_USDC;

pub const MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX: usize = 100;
pub const PRIORITY_OPERATIONS_PUB_DATA_BITS_PER_TX: usize =
    MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX * 8;

pub const ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE: usize = 17;
pub const ON_CHAIN_OPERATIONS_PUB_DATA_BITS_SIZE: usize =
    8 * ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;

pub const KECCAK_HASH_OUT_BIT_SIZE: usize = 256;
pub const KECCAK_HASH_OUT_BYTE_SIZE: usize = KECCAK_HASH_OUT_BIT_SIZE / 8;

pub const ONE_LIT: u64 = 100_000_000;
pub const ONE_USDC: u64 = 1_000_000;
pub const ONE_USDC_COLLATERAL: u64 = ONE_USDC * USDC_TO_COLLATERAL_MULTIPLIER as u64; // 1 USDC in collateral units ~ 40 bits
pub const IMPACT_USDC_AMOUNT: u64 = 500 * ONE_USDC; // 500 USDC

pub const SECOND_IN_MS: usize = 1000; // 1 second = 1,000 ms
pub const MINUTE_IN_MS: usize = SECOND_IN_MS * 60; // 1 minute = 60,000 ms
pub const HOUR_IN_MS: usize = MINUTE_IN_MS * 60; // 1 hour = 3,600,000 ms

pub const INVALID_POSITION_INDEX: usize = POSITION_LIST_SIZE;

pub const FUNDING_SMALL_CLAMP: i64 = 500;
pub const FUNDING_BIG_CLAMP: i64 = 4 * 1_000_000 / 100;
pub const FUNDING_RATE_CLAMP_BITS: usize = 13;
pub const FUNDING_RATE_BITS: usize = 58;
pub const FUNDING_PERIOD: usize = HOUR_IN_MS; // 1 hour = 3,600,000 ms
pub const FUNDING_PREMIUM_MULTIPLIER_TICK: u64 = 100; // scales the market funding premium multiplier, 100 = 1x
pub const FUNDING_PREMIUM_MULTIPLIER_BITS: usize = 8; // multiplier is in (0, 100], keep bit size 8 for split bytes gate

// Ticks
pub const FUNDING_RATE_TICK: u32 = 1_000_000;
pub const FUNDING_RATE_MULTIPLIER: u32 = USDC_TO_COLLATERAL_MULTIPLIER / FUNDING_RATE_TICK;
pub const FEE_TICK: u64 = 1_000_000;
pub const FEE_BITS: usize = 20;
pub const MARGIN_TICK: u32 = 10_000;
pub const MARGIN_FRACTION_MULTIPLIER: u32 = USDC_TO_COLLATERAL_MULTIPLIER / MARGIN_TICK;
pub const MARGIN_FRACTION_BITS: usize = 16;
pub const SHARE_TICK: u64 = 10_000;
pub const SHARE_RATE_BITS: usize = 16;
pub const ASSET_MARGIN_TICK: u64 = 10_000;
pub const ASSET_MARGIN_TO_FEE_TICK_MULTIPLIER: u64 = FEE_TICK / ASSET_MARGIN_TICK;

pub const MARKET_OPEN_INTEREST_BITS: usize = 56; // max open interest is 2^56 - 1
pub const MARKET_OPEN_INTEREST_NOTIONAL_BITS: usize = 56; // max open interest notional is 2^56 - 1
pub const MARKET_OPEN_INTEREST_NOTIONAL: u64 = (1 << MARKET_OPEN_INTEREST_NOTIONAL_BITS) - 1;
pub const MARKET_OPEN_INTEREST: u64 = (1 << MARKET_OPEN_INTEREST_BITS) - 1;

pub const NB_CLOID_UNIQUENESS_CHECK_PER_TX: usize = 3;
pub const NB_ACCOUNTS_PER_TX: usize = 3;
pub const NB_ACCOUNT_ORDERS_PATHS_PER_TX: usize = 5;
pub const NB_ASSETS_PER_TX: usize = 2;
pub const NB_STRATEGIES: usize = 1 << STRATEGY_INDEX_BITS;

pub const TREASURY_ACCOUNT_INDEX: usize = 0;
pub const INSURANCE_FUND_OPERATOR_ACCOUNT_INDEX: usize = 1;

pub const MAX_INDEX_PRICE_DIVIDER: u64 = (1 << 56) - 1;
pub const MAX_ASSET_PRICE_BITS: usize = 56;

// L1 Account address
pub const NIL_L1_ADDRESS: u64 = 0;

// Account index
pub const MIN_ACCOUNT_INDEX: i64 = 0;

pub const STRATEGY_INDEX_BITS: usize = 3;
pub const DEFAULT_STRATEGY_INDEX: usize = 0;
pub const NIL_STRATEGY_INDEX: usize = 8;
pub const MARGIN_STRATEGY_INDEX: usize = 1;
/// The insurance fund holds its spot USDC in strategy bucket 1,
/// separate from perps margin in strategy 0.
pub const INSURANCE_FUND_SPOT_STRATEGY_INDEX: usize = 1;

/// Before changing these, see [`crate::transactions::l2_change_pubkey::L2ChangePubKeyTxTarget::verify()`]
pub const MAX_ACCOUNT_INDEX: i64 = 281474976710654; // 2^48 - 2
pub const NIL_ACCOUNT_INDEX: i64 = 281474976710655; // 2^48 - 1
pub const MAX_MASTER_ACCOUNT_INDEX: i64 = 140737488355327; // 2^47 - 1
pub const MIN_SUB_ACCOUNT_INDEX: i64 = 140737488355328; // 2^47
pub const NIL_MASTER_ACCOUNT_INDEX: i64 = 0;

pub const MAX_LIQUIDITY_POOL_COOLDOWN_PERIOD: i64 = 1000 * 60 * 60 * 24 * 365; // 1 year
pub const MAX_STAKING_POOL_LOCKUP_PERIOD: i64 = 1000 * 60 * 60 * 24 * 30; // 30 days
pub const MAX_MAX_INTEGRATOR_SPOT_TAKER_FEE: i64 = 20_000; // 2% fee
pub const MAX_MAX_INTEGRATOR_SPOT_MAKER_FEE: i64 = 20_000; // 2% fee
pub const MAX_MAX_INTEGRATOR_PERPS_TAKER_FEE: i64 = 5_000; // 0.5% fee
pub const MAX_MAX_INTEGRATOR_PERPS_MAKER_FEE: i64 = 5_000; // 0.5% fee

pub const MAX_INTEGRATOR_FEE_AMOUNT_BITS: usize = 60;
pub const MAX_INTEGRATOR_FEE_AMOUNT: usize = (1 << MAX_INTEGRATOR_FEE_AMOUNT_BITS) - 1;

// Account Type
pub const MASTER_ACCOUNT_TYPE: u8 = 0;
pub const SUB_ACCOUNT_TYPE: u8 = 1;
pub const PUBLIC_POOL_ACCOUNT_TYPE: u8 = 2;
pub const INSURANCE_FUND_ACCOUNT_TYPE: u8 = 3; // Insurance Fund Public Pool
pub const LIGHTER_STAKING_POOL_ACCOUNT_TYPE: u8 = 4; // Lighter Staking Pool

pub const ACCOUNT_ACCOUNT_TRADING_MODE_SIMPLE: u8 = 0;
pub const ACCOUNT_ACCOUNT_TRADING_MODE_UNIFIED: u8 = 1;

// Public pool
pub const INITIAL_POOL_SHARE_VALUE: u64 = 1_000; // 0.001 USDC
pub const MIN_INITIAL_TOTAL_SHARES: u64 = 1_000 * (ONE_USDC / INITIAL_POOL_SHARE_VALUE);
pub const MAX_INITIAL_TOTAL_SHARES: u64 = 1_000_000_000 * (ONE_USDC / INITIAL_POOL_SHARE_VALUE);
pub const MIN_INITIAL_TOTAL_STAKING_SHARES: u64 = 100_000 * (ONE_LIT / INITIAL_POOL_SHARE_VALUE);
pub const MAX_INITIAL_TOTAL_STAKING_SHARES: u64 =
    1_000_000_000 * (ONE_LIT / INITIAL_POOL_SHARE_VALUE);
pub const MIN_STAKING_SHARES_TO_MINT_OR_BURN: u64 = 1;
pub const MAX_STAKING_SHARES_TO_MINT_OR_BURN: u64 =
    (1 << MAX_STAKING_SHARES_TO_MINT_OR_BURN_BITS) - 1;
pub const MAX_STAKING_SHARES_TO_MINT_OR_BURN_BITS: usize = 60;
pub const ACTIVE_PUBLIC_POOL: u8 = 0;
pub const FROZEN_PUBLIC_POOL: u8 = 1;
pub const MAX_POOL_SHARES: u64 = (1 << 60) - 1;
pub const MAX_POOL_SHARES_BITS: usize = 60;
pub const MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC: u64 =
    (1 << MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS) - 1;
pub const MAX_POOL_SHARES_TO_MINT_OR_BURN_USDC_BITS: usize = 60;
pub const MAX_POOL_SHARES_TO_MINT_OR_BURN_LIT: u64 =
    (1 << MAX_POOL_SHARES_TO_MINT_OR_BURN_LIT_BITS) - 1;
pub const MAX_POOL_SHARES_TO_MINT_OR_BURN_LIT_BITS: usize = 60;
pub const MAX_POOL_PRINCIPAL_AMOUNT: u64 = (1 << 56) - 1; // 2^56 - 1 max USDC to invest in a poo
pub const MAX_POOL_PRINCIPAL_AMOUNT_BITS: usize = 56;

pub const INITIAL_TOTAL_SHARES_BITS: usize = 40;
pub const INITIAL_TOTAL_STAKING_SHARES_BITS: usize = 48;

// API key index
pub const MIN_API_KEY_INDEX: u8 = 0;
/// Before changing these, see [`crate::transactions::l2_change_pubkey::L2ChangePubKeyTxTarget::verify()`]
pub const MAX_API_KEY_INDEX: u8 = 254; // 2^8 - 2
pub const NIL_API_KEY_INDEX: u8 = 255; // 2^8 - 1

// Order / Client order index
pub const NIL_ORDER_INDEX: i64 = 0;
pub const NIL_CLIENT_ORDER_INDEX: i64 = 0;

// Market index
pub const NIL_MARKET_INDEX: u8 = 255; // 2^8 - 1
pub const POSITION_LIST_SIZE: usize = 255; // Only markets from 0 to 254 is usable. Last market is always empty and used for empty transactions
pub const POSITION_LIST_SIZE_BITS: usize = 8;
pub const POSITION_HASH_BUCKET_COUNT: usize = 16;
pub const POSITION_HASH_BUCKET_SIZE: usize = 16;
pub const SHARES_LIST_SIZE: usize = 16;
pub const SHARES_DELTA_LIST_SIZE: usize = SHARES_LIST_SIZE * 2;
pub const MAX_APPROVED_INTEGRATORS: usize = 4;

pub const MAX_PENDING_UNLOCKS: usize = 8;

pub const ASSET_LIST_SIZE_BITS: usize = 6;
pub const ASSET_LIST_SIZE: usize = 1 << ASSET_LIST_SIZE_BITS; // first and last slots unused
pub const MARGINED_ASSET_LIST_SIZE_BITS: usize = 3;
pub const MARGINED_ASSET_LIST_SIZE: usize = 7; // Last element is actually empty, reserved for nil
pub const NIL_MARGIN_ASSET_INDEX: u64 = 7;

pub const NATIVE_ASSET_INDEX: u64 = 1;
pub const LIT_ASSET_INDEX: u64 = 2;
pub const USDC_ASSET_INDEX: u64 = 3;
pub const USDC_MARGIN_ASSET_INDEX: usize = 0;
pub const MIN_ASSET_INDEX: u64 = 1;
pub const MAX_ASSET_INDEX: u64 = 62;
pub const NIL_ASSET_INDEX: u64 = 0;

pub const MAX_EXCHANGE_ASSET_BALANCE_BITS: usize = 60;
pub const MAX_EXCHANGE_ASSET_BALANCE: u64 = (1 << MAX_EXCHANGE_ASSET_BALANCE_BITS) - 1;
pub const EXTENSION_MULTIPLIER_BITS: usize = 56;
pub const MAX_EXTENSION_MULTIPLIER: u64 = (1 << EXTENSION_MULTIPLIER_BITS) - 1;
pub const MAX_WITHDRAWAL_AMOUNT: u64 = MAX_EXCHANGE_ASSET_BALANCE;
pub const MAX_TRANSFER_AMOUNT: u64 = MAX_EXCHANGE_ASSET_BALANCE;

pub const MIN_ASSET_BALANCE: i128 = 0i128;
pub const MAX_ASSET_BALANCE: i128 = ((1i128) << COLLATERAL_BITS) - 1;
pub const MIN_COLLATERAL_INT128: i128 = -MAX_ASSET_BALANCE;
pub const MAX_COLLATERAL_INT128: i128 = MAX_ASSET_BALANCE;

pub const ASSET_EXTENSION_MULTIPLIER_BITS: usize = 56;

pub const EMPTY_ASSET_BALANCE: i128 = 0;

// Asset Margin Modes
/// means asset can't be used as margin
pub const ASSET_MARGIN_MODE_DISABLED: u64 = 0;
/// means asset can be used as margin
pub const ASSET_MARGIN_MODE_ENABLED: u64 = 1;

// Account Asset Margin Modes
/// means account is not using this asset as margin
pub const ACCOUNT_ASSET_MARGIN_MODE_DISABLED: u64 = 0;
/// means account is using this asset as margin
pub const ACCOUNT_ASSET_MARGIN_MODE_ENABLED: u64 = 1;

// Collateral
pub const MAX_EXCHANGE_USDC_BITS: usize = 60;

// Order index
pub const NIL_ORDER_PRICE_INDEX: i64 = 0;
pub const NIL_ORDER_NONCE_INDEX: i64 = 0;

// Order expiry
pub const NIL_ORDER_EXPIRY: i64 = 0;
pub const MIN_ORDER_EXPIRY_PERIOD: i64 = 1000 * 60 * 4; // 4 minutes = 240,000 ms
pub const MAX_ORDER_EXPIRY_PERIOD: i64 = 1000 * 60 * 60 * 24 * 30; // 30 days = 2,592,000,000 ms

// Cancel all orders time in force types
pub const IMMEDIATE_CANCEL_ALL: u8 = 0;
pub const SCHEDULED_CANCEL_ALL: u8 = 1;
pub const ABORT_SCHEDULED_CANCEL_ALL: u8 = 2;

pub const MAX_ORDERS_PER_ACCOUNT: i64 = 1000;

pub const MAX_QUOTE_MULTIPLIER: u32 = 10_000; // 10^4
pub const QUOTE_MULTIPLIER_BITS: usize = 14;

pub const MAX_DELEVERAGE_QUOTE: u64 = (1 << 56) - 1;
pub const MAX_DELEVERAGE_QUOTE_BITS: usize = 56;

// Market status
pub const MARKET_STATUS_EXPIRED: u8 = 0;
pub const MARKET_STATUS_ACTIVE: u8 = 1;

// Register instruction types
pub const EXECUTE_TRANSACTION: u8 = 0;
pub const INSERT_ORDER: u8 = 1;
pub const CANCEL_ALL_ACCOUNT_ORDERS: u8 = 2;
pub const CANCEL_SINGLE_ACCOUNT_ORDER: u8 = 3;
pub const CANCEL_POSITION_TIED_ACCOUNT_ORDERS: u8 = 4;
pub const TRIGGER_CHILD_ORDER: u8 = 5;
pub const CANCEL_ALL_CROSS_MARGIN_ORDERS: u8 = 6;
pub const CANCEL_ALL_ISOLATED_MARGIN_ORDERS: u8 = 7;
pub const TRANSFER_ASSET: u8 = 8;
pub const CANCEL_ALL_MARKET_ACCOUNT_ORDERS: u8 = 9;

pub const PENDING_BASE_REGISTER_SIZE: usize = 8;
pub const REGISTER_STACK_SIZE: usize = PENDING_BASE_REGISTER_SIZE + 1;
pub const NEW_INSTRUCTIONS_MAX_SIZE: usize = 8;

pub const INSERT_MAX_THREE_REGISTERS: usize = 3;
pub const INSERT_MAX_FIVE_REGISTERS: usize = 5;
pub const INSERT_MAX_SIX_REGISTERS: usize = 6;

// Tree depths
pub const ACCOUNT_MERKLE_LEVELS: usize = 48;
pub const API_KEY_MERKLE_LEVELS: usize = 8;
pub const ACCOUNT_ORDERS_MERKLE_LEVELS: usize = 60;
pub const POSITION_MERKLE_LEVELS: usize = 8;
pub const MARKET_MERKLE_LEVELS: usize = 12;
pub const ASSET_MERKLE_LEVELS: usize = 6;
pub const ORDER_BOOK_MERKLE_LEVELS: usize = ORDER_PRICE_BITS + ORDER_NONCE_BITS; // 80

pub const FIRST_ASK_NONCE: i64 = 1;
pub const FIRST_BID_NONCE: i64 = (1i64 << ORDER_NONCE_BITS) - 1;

// Order type
pub const LIMIT_ORDER: u8 = 0;
pub const MARKET_ORDER: u8 = 1;
pub const STOP_LOSS_ORDER: u8 = 2;
pub const STOP_LOSS_LIMIT_ORDER: u8 = 3;
pub const TAKE_PROFIT_ORDER: u8 = 4;
pub const TAKE_PROFIT_LIMIT_ORDER: u8 = 5;
pub const TWAP_ORDER: u8 = 6;
pub const TWAP_SUB_ORDER: u8 = 7;
pub const LIQUIDATION_ORDER: u8 = 8;

// Order Time in force
pub const IOC: u8 = 0; // Immediate or Cancel
pub const GTT: u8 = 1; // Good Till Time
pub const POST_ONLY: u8 = 2;

// Order Trigger Status
pub const TRIGGER_STATUS_NA: u8 = 0;
pub const TRIGGER_STATUS_MARK_PRICE: u8 = 1;
pub const TRIGGER_STATUS_TWAP: u8 = 2;
pub const TRIGGER_STATUS_PARENT_ORDER: u8 = 3;

pub const NIL_ORDER_TRIGGER_PRICE: i64 = 0;
pub const MIN_ORDER_TRIGGER_PRICE: i64 = 1;
pub const MAX_ORDER_TRIGGER_PRICE: i64 = (1 << TRIGGER_PRICE_BITS) - 1;

pub const GROUPING_TYPE_NA: u8 = 0;
pub const GROUPING_TYPE_ONE_TRIGGERS_THE_OTHER: u8 = 1;
pub const GROUPING_TYPE_ONE_CANCELS_THE_OTHER: u8 = 2;
pub const GROUPING_TYPE_ONE_TRIGGERS_A_ONE_CANCELS_THE_OTHER: u8 = 3;

pub const MAX_NB_GROUPED_ORDERS: usize = 3;

// Liquidation Status
pub const HEALTHY: u8 = 0;
pub const PRE_LIQUIDATION: u8 = 1;
pub const PARTIAL_LIQUIDATION: u8 = 2;
pub const FULL_LIQUIDATION: u8 = 3;
pub const BANKRUPTCY: u8 = 4;

// Tx Account Ids
pub const POOL_CROSS_RISK_ID: usize = 0;
pub const POOL_STRATEGY_RISK_ID: usize = 1;

pub const FROM_STRATEGY_ID: usize = 0;
pub const TO_STRATEGY_ID: usize = 1;

pub const OWNER_ACCOUNT_ID: usize = 0;

pub const LIQUIDITY_POOL_ACCOUNT_ID: usize = 0;
pub const STAKING_POOL_ACCOUNT_ID: usize = 1;
pub const OLD_LIQUIDITY_POOL_ACCOUNT_ID: usize = STAKING_POOL_ACCOUNT_ID;
pub const SYSTEM_CONFIG_ACCOUNT_ID: usize = 2;

pub const TAKER_ACCOUNT_ID: usize = 0;
pub const MAKER_ACCOUNT_ID: usize = 1;
pub const FEE_ACCOUNT_ID: usize = 2;

pub const MASTER_ACCOUNT_ID: usize = 0;
pub const SUB_ACCOUNT_ID: usize = 1;

pub const SENDER_ACCOUNT_ID: usize = 0;
pub const RECEIVER_ACCOUNT_ID: usize = 1;
pub const INTEGRATOR_ACCOUNT_ID: usize = 1;

pub const BANKRUPT_ACCOUNT_ID: usize = 0;
pub const DELEVERAGER_ACCOUNT_ID: usize = 1;

pub const TX_ASSET_ID: usize = 0;
pub const FEE_ASSET_ID: usize = 1;

pub const BASE_ASSET_ID: usize = 0;
pub const QUOTE_ASSET_ID: usize = 1;

pub const USDC_BASE_ASSET_ID: usize = 0;
pub const SHARE_OWNER_ACCOUNT_ID: usize = 1;
pub const STAKE_ASSET_ID: usize = 1;

pub const OWNER_OR_POOL_ACCOUNT_ID_1: usize = 0;
pub const OWNER_OR_POOL_ACCOUNT_ID_2: usize = 1;

pub const SELF_TRADE_EQUALITY_ACCOUNT_INDEX: u64 = 0;
pub const SELF_TRADE_EQUALITY_MASTER_ACCOUNT_INDEX: u64 = 1;
pub const SELF_TRADE_BEHAVIOR_EXPIRE_MAKER: u64 = 0;
pub const SELF_TRADE_BEHAVIOR_EXPIRE_TAKER: u64 = 1;
pub const SELF_TRADE_BEHAVIOR_EXPIRE_BOTH: u64 = 2;
pub const SELF_TRADE_BEHAVIOR_REDUCE: u64 = 3;

// Margin Modes
pub const CROSS_MARGIN: usize = 0;
pub const ISOLATED_MARGIN: usize = 1;

// Margin Set Flag
pub const MARGIN_UNSET: usize = 0;
pub const MARGIN_SET: usize = 1;

pub const MARKET_FLAGS_BITS: usize = 2;

// Margin Move Types
pub const REMOVE_MARGIN: u8 = 0;
pub const ADD_MARGIN: u8 = 1;

pub const EMPTY_DELTA_TREE_HASHES: [HashOut<F>; ACCOUNT_MERKLE_LEVELS + 1] = [
    const_hash_out([0, 0, 0, 0]),
    const_hash_out([
        7182099517097165596,
        9311216678150108034,
        8831900494918587432,
        10774846510254277933,
    ]),
    const_hash_out([
        7544744595273422282,
        6210592528081981646,
        13108907452876819928,
        12505825290281357043,
    ]),
    const_hash_out([
        11950793831570313021,
        6669060785995335658,
        13660213005409878970,
        10723233046902380284,
    ]),
    const_hash_out([
        7542899550745766553,
        16598719479483031840,
        741166840114983611,
        9333946413243869352,
    ]),
    const_hash_out([
        15365497841755032527,
        14034401094354213190,
        1753969648565243804,
        11653766485014388743,
    ]),
    const_hash_out([
        6844301749605691075,
        15901475079080112138,
        6459841461395793132,
        5448886425767967618,
    ]),
    const_hash_out([
        13304878475572637399,
        4412425491995232712,
        17565527807313298118,
        14714125094697233493,
    ]),
    const_hash_out([
        5428970986623951092,
        515484187069299980,
        6723256903412060294,
        2762657640779953643,
    ]),
    const_hash_out([
        1871519249400096138,
        4425009494732521599,
        11781822474915155228,
        3328803237036608539,
    ]),
    const_hash_out([
        15356338879497970354,
        7533122335064036428,
        12695829648973758204,
        16095408101001589520,
    ]),
    const_hash_out([
        1107332047625712192,
        17347004049776088849,
        7044377870463089772,
        15001543347224867633,
    ]),
    const_hash_out([
        5094422418556072775,
        472024594924382182,
        11064781981674099049,
        14079628902931319245,
    ]),
    const_hash_out([
        4504466422547795533,
        17951636984460411981,
        14793996872745939842,
        7211891724263505903,
    ]),
    const_hash_out([
        14400638176288824918,
        11326852114846721092,
        16641480472023584785,
        12295560441073063535,
    ]),
    const_hash_out([
        8334096000310279342,
        13772585833590824433,
        10292665123044893503,
        6245928278510245389,
    ]),
    const_hash_out([
        2305614600112830660,
        8230806316094319563,
        3120195739492920788,
        11690589135543297980,
    ]),
    const_hash_out([
        439487605525353214,
        4986520621284058322,
        13192652867367997958,
        11090868160209964633,
    ]),
    const_hash_out([
        7116477851833407005,
        18113331609964080622,
        3025134218799726017,
        5031403147735815570,
    ]),
    const_hash_out([
        13061489384171488326,
        6395930051845359908,
        18171534544221653761,
        9922838568806778780,
    ]),
    const_hash_out([
        16522445924152930664,
        3028253438446345964,
        9440814905132810025,
        1426268537539859231,
    ]),
    const_hash_out([
        4110066839651710100,
        9787216899897133265,
        16698639404720457358,
        16847488486516812447,
    ]),
    const_hash_out([
        3692951812690005206,
        17003687800164319453,
        6442084848072795750,
        645145644758924193,
    ]),
    const_hash_out([
        12348457676651097982,
        3492257900422131607,
        2268175771581020335,
        16552837793976857429,
    ]),
    const_hash_out([
        16059488274238494346,
        3395651696316481054,
        5513034827171370626,
        2541640075949687500,
    ]),
    const_hash_out([
        5035822971520232484,
        11058586161915342829,
        12226231754961735928,
        16698845527124164551,
    ]),
    const_hash_out([
        13571293137547706320,
        10820848971911315064,
        7247777944411832317,
        13794530206638424460,
    ]),
    const_hash_out([
        3182668508674226521,
        12722747163897694696,
        9228468924091852542,
        12728053520279960507,
    ]),
    const_hash_out([
        757031281659687748,
        14513441663712888269,
        2718955174403688742,
        1173882940144797212,
    ]),
    const_hash_out([
        14700561091829789199,
        13866761428768191806,
        13212386660980518557,
        802758511095738163,
    ]),
    const_hash_out([
        13116733958178222214,
        2626184581182645202,
        5346925783673197782,
        2772284376495409363,
    ]),
    const_hash_out([
        6081169604289132277,
        1002396114210161913,
        1904469708916793163,
        7012145267475737418,
    ]),
    const_hash_out([
        11554573132205161099,
        13493990443824524670,
        7386336504868289669,
        10109684771443484380,
    ]),
    const_hash_out([
        18170599852649689179,
        8151334788120169114,
        7174062811584270502,
        13040602462206374147,
    ]),
    const_hash_out([
        1442577577661875575,
        650769652016291875,
        9897867549534121891,
        18392418358060587452,
    ]),
    const_hash_out([
        2605684891869028731,
        2110592622127610464,
        12033429688553956303,
        211992476656596747,
    ]),
    const_hash_out([
        15848909220188696207,
        13313383525046189597,
        6582759906725879223,
        117037446427821588,
    ]),
    const_hash_out([
        1197999285424912197,
        6386281190602046989,
        7760311803351404721,
        10577706212533923082,
    ]),
    const_hash_out([
        611145691065802315,
        4862329740359722830,
        12970076627031038720,
        7377131609739867368,
    ]),
    const_hash_out([
        3091756594250885,
        16695774714355135009,
        9634838844620435221,
        14163902814707696773,
    ]),
    const_hash_out([
        3051586169154967032,
        14015963536807818214,
        5124578084404637192,
        8988799171900531939,
    ]),
    const_hash_out([
        2114459152671692211,
        11132290596728840513,
        18064951678598795044,
        6443190549355616632,
    ]),
    const_hash_out([
        13034692407614024623,
        8843726741749389647,
        6036520696571355268,
        15403969679474473447,
    ]),
    const_hash_out([
        924760757293231525,
        17465654233856413618,
        9974326663816939536,
        16442834293085431086,
    ]),
    const_hash_out([
        15405699107088050258,
        142363242006993444,
        9008268439276940654,
        7763750536026901502,
    ]),
    const_hash_out([
        5017343757770459029,
        15454683448835215546,
        16824100978742817913,
        9654774703768705907,
    ]),
    const_hash_out([
        18167535965730234341,
        2143379982901606688,
        16381276431752090171,
        1922927448026564511,
    ]),
    const_hash_out([
        8559797970623705912,
        5554162541476053923,
        8549961741052492826,
        7962154532420794368,
    ]),
    const_hash_out([
        7549499594340718513,
        9334293042152682409,
        16758120779002423496,
        16308680265794004846,
    ]),
];

const fn const_hash_out(e: [u64; NUM_HASH_OUT_ELTS]) -> HashOut<F> {
    HashOut {
        elements: [const_f(e[0]), const_f(e[1]), const_f(e[2]), const_f(e[3])],
    }
}
