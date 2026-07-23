// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::iop::target::Target;

use crate::bigint::big_u16::CircuitBuilderBigIntU16;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::types::account::AccountTarget;
use crate::types::account_delta::{AccountDeltaTarget, PositionDeltaTarget};
use crate::types::account_position::{AccountPositionTarget, random_access_account_position};
use crate::types::config::{BIG_U96_LIMBS, BIGU16_U64_LIMBS, Builder};
use crate::types::constants::POSITION_LIST_SIZE_BITS;

#[derive(Debug, Clone, Default)]
pub struct PositionWithDelta {
    pub position: AccountPositionTarget,
    pub delta: PositionDeltaTarget,
}

impl PositionWithDelta {
    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            position: AccountPositionTarget::empty(builder),
            delta: PositionDeltaTarget::empty(builder),
        }
    }

    pub fn new_positions_with_pub_data_from_accounts<const NB_ACCOUNTS: usize>(
        builder: &mut Builder,
        access_index: Target,
        accounts: &[AccountTarget],
        accounts_delta: &[AccountDeltaTarget],
    ) -> [PositionWithDelta; NB_ACCOUNTS] {
        builder.register_range_check(access_index, POSITION_LIST_SIZE_BITS);

        let mut account_vecs: [Vec<AccountPositionTarget>; NB_ACCOUNTS] =
            core::array::from_fn(|i| accounts[i].positions.to_vec());

        // Pad all vectors to a multiple of 64
        for i in 0..NB_ACCOUNTS {
            account_vecs[i].push(AccountPositionTarget::empty(builder));
        }
        assert!(account_vecs[0].len() % 64 == 0);

        let zero = builder.zero();
        let mut is_position_before_set = builder._false();
        let mut res: [PositionWithDelta; NB_ACCOUNTS] =
            core::array::from_fn(|_| PositionWithDelta::empty(builder));
        for i in 0..(account_vecs[0].len() / 64) {
            let start_index = builder.constant_i64((i as i64) * 64);
            let end_index = builder.constant_i64(((i + 1) as i64) * 64 - 1);
            let chunk_access_index = builder.sub(access_index, start_index);
            let contains = builder.is_lte(access_index, end_index, POSITION_LIST_SIZE_BITS);
            let contains = builder.and_not(contains, is_position_before_set);
            let chunk_access_index = builder.select(contains, chunk_access_index, zero);
            for j in 0..NB_ACCOUNTS {
                let position_check = random_access_account_position(
                    builder,
                    chunk_access_index,
                    account_vecs[j]
                        .iter()
                        .skip(i * 64)
                        .take(64)
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                res[j].position = AccountPositionTarget::select_position(
                    builder,
                    contains,
                    &position_check,
                    &res[j].position,
                );

                res[j].delta = accounts_delta[j].positions_delta.clone();
            }
            is_position_before_set = builder.or(contains, is_position_before_set);
        }

        res
    }

    pub fn new_position_with_pub_data_from_new_position(
        builder: &mut Builder,
        positions_with_pub_data_before: &PositionWithDelta,
        new_position: &AccountPositionTarget,
    ) -> (PositionWithDelta, BigIntTarget) {
        let position_size_delta = builder.sub_bigint_u16_non_carry(
            &new_position.position,
            &positions_with_pub_data_before.position.position,
            BIGU16_U64_LIMBS,
        );
        let funding_delta = builder.sub_bigint_u16_non_carry(
            &new_position.last_funding_rate_prefix_sum,
            &positions_with_pub_data_before
                .position
                .last_funding_rate_prefix_sum,
            BIGU16_U64_LIMBS,
        );
        let old_aggregated_usdc = positions_with_pub_data_before
            .position
            .calculate_aggregated_usdc(builder);
        let new_aggregated_usdc = new_position.calculate_aggregated_usdc(builder);
        (
            PositionWithDelta {
                position: new_position.clone(),
                delta: PositionDeltaTarget {
                    position_delta: builder.add_bigint_u16_non_carry(
                        &positions_with_pub_data_before.delta.position_delta,
                        &position_size_delta,
                        BIGU16_U64_LIMBS,
                    ),
                    funding_rate_prefix_sum_delta: builder.add_bigint_u16_non_carry(
                        &positions_with_pub_data_before
                            .delta
                            .funding_rate_prefix_sum_delta,
                        &funding_delta,
                        BIGU16_U64_LIMBS,
                    ),
                },
            },
            builder.sub_bigint_non_carry(&new_aggregated_usdc, &old_aggregated_usdc, BIG_U96_LIMBS),
        )
    }
}
