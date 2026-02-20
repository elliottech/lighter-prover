// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::array;

use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::BoolTarget;

use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::types::account::AccountTarget;
use crate::types::account_position::AccountPositionTarget;
use crate::types::config::Builder;
use crate::types::constants::{
    EMPTY_ACCOUNT_HASH, POSITION_HASH_BUCKET_COUNT, POSITION_HASH_BUCKET_SIZE,
    TREASURY_ACCOUNT_INDEX,
};

impl AccountTarget {
    pub fn aggregated_balance_hash(&self, builder: &mut Builder, index: usize) -> HashOutTarget {
        let asset_balance = &self.aggregated_balances[index];
        let mut elements = vec![asset_balance.sign.target];
        elements.extend_from_slice(
            &asset_balance
                .abs
                .limbs
                .iter()
                .map(|x| x.0)
                .collect::<Vec<_>>(),
        );
        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        let empty_hash = builder.zero_hash_out();
        let is_empty = builder.is_zero_bigint(asset_balance);
        builder.select_hash(is_empty, &empty_hash, &non_empty_hash)
    }

    pub fn hash(
        &self,
        builder: &mut Builder,
        position_bucket_hashes: &[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2],
    ) -> (
        HashOutTarget, // Account Hash
        HashOutTarget, // Account Pub Data Hash
        BoolTarget,    // Is Empty
    ) {
        let partial_hash = self.partial_hash(builder, position_bucket_hashes);
        self.hash_from_partial_hash(builder, &partial_hash)
    }

    pub fn fee_account_hash(
        &self,
        builder: &mut Builder,
    ) -> (HashOutTarget, HashOutTarget, BoolTarget) {
        self.hash_from_partial_hash(
            builder,
            &[self.partial_hash, self.partial_hash_for_pub_data],
        )
    }

    // Hash of fields that wouldn't change for the fee account
    fn partial_hash(
        &self,
        builder: &mut Builder,
        position_bucket_hashes: &[[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2],
    ) -> [HashOutTarget; 2] {
        // Pub data elements' hash is a subset of the full elements' hash
        let mut pub_data_elements = vec![];
        {
            pub_data_elements.extend_from_slice(
                &position_bucket_hashes[1]
                    .iter()
                    .flat_map(|x| x.elements)
                    .collect::<Vec<_>>(),
            );

            pub_data_elements.extend_from_slice(
                &self
                    .public_pool_shares
                    .iter()
                    .flat_map(|pps| [pps.public_pool_index, pps.share_amount])
                    .collect::<Vec<_>>(),
            );

            pub_data_elements.extend_from_slice(&[
                self.public_pool_info.total_shares,
                self.public_pool_info.operator_shares,
            ]);
        }
        let pub_data_elements_hash =
            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_elements);

        let mut elements = pub_data_elements_hash.elements.to_vec();
        {
            elements.extend_from_slice(
                &position_bucket_hashes[0]
                    .iter()
                    .flat_map(|x| x.elements)
                    .collect::<Vec<_>>(),
            );

            elements.extend_from_slice(
                &self
                    .public_pool_shares
                    .iter()
                    .flat_map(|pps| [pps.principal_amount, pps.entry_timestamp])
                    .collect::<Vec<_>>(),
            );
            elements.extend_from_slice(&[
                self.public_pool_info.status,
                self.public_pool_info.min_operator_share_rate,
                self.public_pool_info.operator_fee,
            ]);

            let pending_unlocks_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(
                self.pending_unlocks
                    .iter()
                    .flat_map(|pw| {
                        [
                            pw.unlock_timestamp,
                            pw.asset_index,
                            pw.amount.limbs[0].0,
                            pw.amount.limbs[1].0,
                            pw.amount.limbs[2].0,
                        ]
                    })
                    .collect::<Vec<_>>(),
            );
            elements.extend_from_slice(&pending_unlocks_hash.elements);
        }

        [
            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements),
            pub_data_elements_hash,
        ]
    }

    fn hash_from_partial_hash(
        &self,
        builder: &mut Builder,
        partial_hash: &[HashOutTarget; 2],
    ) -> (HashOutTarget, HashOutTarget, BoolTarget) {
        let mut pub_data_elements = vec![];
        {
            pub_data_elements.extend_from_slice(&partial_hash[1].elements);
            pub_data_elements.extend_from_slice(
                &self
                    .l1_address
                    .limbs
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>(),
            );
            pub_data_elements.push(self.account_type);

            pub_data_elements.extend_from_slice(&self.aggregated_balances_root.elements);
        }

        let mut elements = vec![];
        {
            elements.extend_from_slice(&partial_hash[0].elements);
            elements.push(self.master_account_index);
            elements.extend_from_slice(
                &self
                    .l1_address
                    .limbs
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>(),
            );
            elements.push(self.account_type);

            elements.extend_from_slice(
                &self
                    .collateral
                    .abs
                    .limbs
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>(),
            );
            elements.push(self.collateral.sign.target);

            let strategy_hash = {
                let mut elements = vec![];
                self.public_pool_info
                    .strategies
                    .iter()
                    .for_each(|strategy| {
                        elements.extend_from_slice(
                            &strategy.abs.limbs.iter().map(|x| x.0).collect::<Vec<_>>(),
                        );
                        elements.push(strategy.sign.target);
                    });
                builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements)
            };
            elements.extend_from_slice(&strategy_hash.elements);

            elements.extend_from_slice(&[
                self.total_order_count,
                self.total_non_cross_order_count,
                self.cancel_all_time,
                self.account_trading_mode,
            ]);

            [
                &self.api_key_root,
                &self.account_orders_root,
                &self.asset_root,
            ]
            .iter()
            .for_each(|hash| {
                elements.extend_from_slice(&hash.elements);
            });
        }

        let non_empty_hash = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(elements);
        let non_empty_pub_data_hash =
            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_elements);

        let is_empty = self.is_empty(builder, &non_empty_hash);
        let empty_hash = builder.zero_hash_out();
        (
            builder.select_hash(is_empty, &empty_hash, &non_empty_hash),
            builder.select_hash(is_empty, &empty_hash, &non_empty_pub_data_hash),
            is_empty,
        )
    }

    pub fn get_position_bucket_hashes(
        &self,
        builder: &mut Builder,
    ) -> [[HashOutTarget; POSITION_HASH_BUCKET_COUNT]; 2] {
        let mut positions_ext = self.positions.to_vec();
        positions_ext.push(AccountPositionTarget::empty(builder));
        let bucket_hashes: Vec<[HashOutTarget; 2]> = positions_ext
            .chunks(POSITION_HASH_BUCKET_SIZE)
            .map(|bucket: &[AccountPositionTarget]| Self::get_position_bucket_hash(builder, bucket))
            .collect::<Vec<_>>();
        [
            array::from_fn(|i| bucket_hashes[i][0]),
            array::from_fn(|i| bucket_hashes[i][1]),
        ]
    }

    pub fn get_position_bucket_hash(
        builder: &mut Builder,
        bucket: &[AccountPositionTarget],
    ) -> [HashOutTarget; 2] {
        // Pub data hash fields is a subset of the full fields' hash, calculate pub data hash first
        let mut pub_data_hash_params = vec![];
        for pos in bucket {
            pos.append_position_pub_data_hash_params(builder, &mut pub_data_hash_params);
        }
        let pub_data_bucket_hash =
            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(pub_data_hash_params);

        let mut hash_params = pub_data_bucket_hash.elements.to_vec();
        for pos in bucket {
            pos.append_position_hash_params(builder, &mut hash_params);
        }

        [
            builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(hash_params),
            pub_data_bucket_hash,
        ]
    }

    /// Treasury account is reserved and should never be considered empty.
    fn is_empty(&self, builder: &mut Builder, hash: &HashOutTarget) -> BoolTarget {
        let empty_account_hash = builder.constant_hash(EMPTY_ACCOUNT_HASH);
        let is_empty_account_hash = builder.is_equal_hash(hash, &empty_account_hash);

        let is_treasury =
            builder.is_equal_constant(self.account_index, TREASURY_ACCOUNT_INDEX as u64);

        builder.and_not(is_empty_account_hash, is_treasury)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::types::account::{Account, AccountTargetWitness};
    use crate::types::config::{C, CIRCUIT_CONFIG, F};
    use crate::types::constants::{
        EMPTY_ACCOUNT_ORDERS_TREE_ROOT, EMPTY_API_KEY_TREE_ROOT, EMPTY_ASSET_TREE_ROOT,
    };

    #[test]
    fn empty_hash_check() -> Result<()> {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let account = AccountTarget::new(&mut builder);

        let pbh = account.get_position_bucket_hashes(&mut builder);
        let (acc_hash, acc_pd_hash, acc_is_empty) = account.hash(&mut builder, &pbh);

        let is_zero_hash = builder.is_zero_hash_out(&acc_hash);
        builder.assert_true(is_zero_hash);
        let is_zero_hash = builder.is_zero_hash_out(&acc_pd_hash);
        builder.assert_true(is_zero_hash);
        builder.assert_true(acc_is_empty);

        let data = builder.build::<C>();
        let mut pw = PartialWitness::<F>::new();

        pw.set_account_target(
            &account,
            &Account::<F> {
                account_index: 234234324,
                api_key_root: EMPTY_API_KEY_TREE_ROOT,
                account_orders_root: EMPTY_ACCOUNT_ORDERS_TREE_ROOT,
                asset_root: EMPTY_ASSET_TREE_ROOT,
                aggregated_balances_root: EMPTY_ASSET_TREE_ROOT,
                ..Account::<F>::default()
            },
        )?;

        data.verify(data.prove(pw).unwrap())
    }
}
