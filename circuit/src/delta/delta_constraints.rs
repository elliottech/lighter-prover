// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use itertools::Itertools;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use serde::Deserialize;
use serde_with::serde_as;

use super::account_delta_full_leaf::{
    AccountDeltaFullLeaf, AccountDeltaFullLeafTarget, AccountDeltaLeafTargetWitness,
};
use super::market_delta_leaf::{
    MarketDeltaLeaf, MarketDeltaLeafTarget, MarketDeltaLeafTargetWitness,
};
use crate::bigint::bigint::CircuitBuilderBigInt;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::comparison::CircuitBuilderSubtractiveComparison;
use crate::delta::evaluate_sequence::CircuitBuilderSequenceEvaluator;
use crate::delta::types::{DeltaPublicInputTarget, DeltaPublicOutputTarget};
use crate::delta::utils::{
    digest, pack_asset_balance, pack_binary_options_position, pack_conditionals_with_account_type,
    pack_l1_address, pack_position,
};
use crate::deserializers;
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, PartialWitnessQuinticExt};
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::poseidon2::Poseidon2Hash;
use crate::signed::signed_target::CircuitBuilderSigned;
use crate::types::config::{Builder, C, D, F};
use crate::types::constants::{
    ACCOUNT_INDEX_BITS, ACCOUNT_MERKLE_LEVELS, ASSET_LIST_SIZE, BINARY_OPTIONS_MARKET_SLOT_COUNT,
    EMPTY_DELTA_TREE_HASHES, MAX_ACCOUNT_INDEX, MAX_ASSET_INDEX, MIN_ASSET_INDEX,
    MIN_BINARY_OPTIONS_MARKET_INDEX, NIL_ACCOUNT_INDEX, POSITION_LIST_SIZE, SHARES_DELTA_LIST_SIZE,
};
use crate::utils::CircuitBuilderUtils;

#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
pub struct DeltaWitness<F>
where
    F: Field + RichField,
{
    #[serde(rename = "ad")]
    pub account_deltas: Vec<AccountDeltaFullLeaf>,
    #[serde(rename = "pi")]
    pub previous_account_index: i64,
    #[serde(rename = "mpad")]
    #[serde(deserialize_with = "deserializers::path_matrix")]
    pub path_matrix: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; 2],

    #[serde(rename = "mdh", default)]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub market_delta_hash: HashOut<F>,
    #[serde(rename = "rmd", default = "default_remaining_market_delta_count")]
    pub remaining_market_delta_count: i64,
    #[serde(rename = "tmd", default)]
    pub total_market_delta_count: u64,
    #[serde(rename = "md", default)]
    pub market_deltas: Vec<MarketDeltaLeaf>,

    #[serde(rename = "pdx")]
    #[serde(deserialize_with = "deserializers::hash_out")]
    pub x: HashOut<F>,
}

fn default_remaining_market_delta_count() -> i64 {
    -1
}

#[derive(Debug)]
pub struct DeltaTarget {
    pub leaves: Vec<AccountDeltaFullLeafTarget>,
    pub market_leaves: Vec<MarketDeltaLeafTarget>,
    pub market_leaf_count: Target,
    pub total_market_delta_count: Target,
    pub public_inputs: DeltaPublicInputTarget,
    pub output: DeltaPublicOutputTarget,
}

#[derive(Debug)]
pub struct DeltaCircuit {
    pub builder: Builder,
    pub target: DeltaTarget,

    // Helpers
    should_evaluate: Vec<BoolTarget>,
    should_evaluate_market: Vec<BoolTarget>,
    is_first_iteration: BoolTarget,
    no_account_yet: BoolTarget,
}

impl DeltaCircuit {
    pub fn new(config: CircuitConfig, account_count: usize, market_count: usize) -> Self {
        let mut builder = Builder::new(config);

        let target = DeltaTarget {
            leaves: (0..account_count)
                .map(|_| AccountDeltaFullLeafTarget::new(&mut builder))
                .collect(),
            market_leaves: (0..market_count)
                .map(|_| MarketDeltaLeafTarget::new(&mut builder))
                .collect(),
            market_leaf_count: builder.add_virtual_target(),
            total_market_delta_count: builder.add_virtual_target(),
            public_inputs: DeltaPublicInputTarget::new_public(&mut builder),
            output: DeltaPublicOutputTarget::new_public(&mut builder),
        };

        let neg_one = builder.neg_one();
        let is_first_iteration =
            builder.is_equal(target.public_inputs.remaining_market_delta_count, neg_one);
        let no_account_yet = builder.is_equal(target.public_inputs.account_index, neg_one);
        builder.conditional_assert_true(is_first_iteration, no_account_yet);

        Self {
            target,
            builder,

            should_evaluate: vec![],
            should_evaluate_market: vec![],
            is_first_iteration,
            no_account_yet,
        }
    }

    fn eval_delta_polynomial(&mut self) {
        let builder = &mut self.builder;

        builder.sequence_initialize(0, self.target.public_inputs.evaluation_point);
        let mut degree = builder.zero();

        digest(
            builder,
            self.target.total_market_delta_count,
            self.is_first_iteration,
            &mut degree,
        );
        for (i, market_leaf) in self.target.market_leaves.iter().enumerate() {
            for limb in market_leaf.pack(builder) {
                digest(builder, limb, self.should_evaluate_market[i], &mut degree);
            }
        }

        let _1_bit_shifter = builder.constant_u64(1 << 1);

        let zero = builder.zero();
        let mut last_account_index = builder.select(
            self.no_account_yet,
            zero,
            self.target.public_inputs.account_index,
        );
        for (i, delta) in self.target.leaves.iter().enumerate() {
            let is_enabled = self.should_evaluate[i]; // Disallow empty leaf insertion
            {
                let target = builder.sub(delta.account_index, last_account_index);
                digest(builder, target, is_enabled, &mut degree);
                last_account_index = delta.account_index;
            }

            // (has_l1_address, has_public_pool_info, account_type)
            let (has_l1_address, has_public_pool_info, conditionals_packed) =
                pack_conditionals_with_account_type(
                    builder,
                    &delta.l1_address,
                    delta.account_type,
                    &delta.public_pool_info_delta,
                );
            {
                digest(builder, conditionals_packed, is_enabled, &mut degree);
            }
            {
                let flag = builder.and(is_enabled, has_l1_address);
                for limb in pack_l1_address(builder, &delta.l1_address) {
                    digest(builder, limb, flag, &mut degree);
                }
            }
            {
                let flag = builder.and(is_enabled, has_public_pool_info);

                let (tsd_abs, tsd_sign) =
                    builder.abs(delta.public_pool_info_delta.total_shares_delta);
                let tsd_negative = builder.is_sign_negative(tsd_sign);
                let (osd_abs, osd_sign) =
                    builder.abs(delta.public_pool_info_delta.operator_shares_delta);
                let osd_negative = builder.is_sign_negative(osd_sign);

                let target =
                    builder.mul_add(_1_bit_shifter, tsd_negative.target, osd_negative.target);
                digest(builder, target, flag, &mut degree);

                digest(builder, tsd_abs, flag, &mut degree);

                digest(builder, osd_abs, flag, &mut degree);
            }
            {
                let mut zero_position_count = builder.zero();
                let mut is_position_empty_list = vec![];
                for i in 0..POSITION_LIST_SIZE {
                    let is_position_empty = delta.perps_deltas[i].is_empty(builder);
                    is_position_empty_list.push(is_position_empty);
                    zero_position_count =
                        builder.add(zero_position_count, is_position_empty.target);
                }
                let total_pos_count = builder.constant_usize(POSITION_LIST_SIZE);
                let nonzero_position_count = builder.sub(total_pos_count, zero_position_count);
                digest(builder, nonzero_position_count, is_enabled, &mut degree);
                for i in 0..POSITION_LIST_SIZE {
                    let pos_delta = &delta.perps_deltas[i];
                    let flag = builder.and_not(is_enabled, is_position_empty_list[i]);
                    let market_index = builder.constant_usize(i);

                    for limb in pack_position(builder, market_index, pos_delta) {
                        digest(builder, limb, flag, &mut degree);
                    }
                }
            }
            {
                let mut nonzero_binary_options_count = builder.zero();
                let mut is_binary_options_empty_list = vec![];
                for i in 0..BINARY_OPTIONS_MARKET_SLOT_COUNT {
                    let is_empty = delta.binary_options_deltas[i].is_empty(builder);
                    is_binary_options_empty_list.push(is_empty);
                    nonzero_binary_options_count =
                        builder.add(nonzero_binary_options_count, is_empty.target);
                }
                let total_count = builder.constant_usize(BINARY_OPTIONS_MARKET_SLOT_COUNT);
                let nonzero_binary_options_count =
                    builder.sub(total_count, nonzero_binary_options_count);
                digest(
                    builder,
                    nonzero_binary_options_count,
                    is_enabled,
                    &mut degree,
                );
                for i in 0..BINARY_OPTIONS_MARKET_SLOT_COUNT {
                    let entry = &delta.binary_options_deltas[i];
                    let flag = builder.and_not(is_enabled, is_binary_options_empty_list[i]);
                    let market_index = builder.constant_usize(MIN_BINARY_OPTIONS_MARKET_INDEX + i);

                    for limb in pack_binary_options_position(builder, market_index, entry) {
                        digest(builder, limb, flag, &mut degree);
                    }
                }
            }
            {
                let mut zero_asset_count = builder.two();
                let mut is_asset_empty_list = vec![];
                for i in MIN_ASSET_INDEX as usize..=MAX_ASSET_INDEX as usize {
                    let is_asset_empty = builder.is_zero_bigint(&delta.aggregated_asset_deltas[i]);
                    is_asset_empty_list.push(is_asset_empty);
                    zero_asset_count = builder.add(zero_asset_count, is_asset_empty.target);
                }
                let total_asset_count = builder.constant_usize(ASSET_LIST_SIZE);
                let nonzero_asset_count = builder.sub(total_asset_count, zero_asset_count);
                digest(builder, nonzero_asset_count, is_enabled, &mut degree);
                for (asset_ctr, i) in
                    (MIN_ASSET_INDEX as usize..=MAX_ASSET_INDEX as usize).enumerate()
                {
                    let flag = builder.and_not(is_enabled, is_asset_empty_list[asset_ctr]);
                    let asset_index = builder.constant_usize(i);

                    for limb in
                        pack_asset_balance(builder, asset_index, &delta.aggregated_asset_deltas[i])
                    {
                        digest(builder, limb, flag, &mut degree);
                    }
                }
            }
            {
                let max_ai = builder.constant_usize(MAX_ACCOUNT_INDEX as usize);
                let mut zero_pool_share_count = builder.zero();
                let mut is_pool_share_empty_list = vec![];
                for i in 0..SHARES_DELTA_LIST_SIZE {
                    let is_pool_share_empty =
                        delta.public_pool_shares_delta[i].is_empty_without_metadata(builder);
                    is_pool_share_empty_list.push(is_pool_share_empty);
                    zero_pool_share_count =
                        builder.add(zero_pool_share_count, is_pool_share_empty.target);
                }
                let total_pool_share_count = builder.constant_usize(SHARES_DELTA_LIST_SIZE);
                let nonzero_pool_share_count =
                    builder.sub(total_pool_share_count, zero_pool_share_count);
                digest(builder, nonzero_pool_share_count, is_enabled, &mut degree);
                for i in 0..SHARES_DELTA_LIST_SIZE {
                    let pool_share = &delta.public_pool_shares_delta[i];
                    let flag = builder.and_not(is_enabled, is_pool_share_empty_list[i]);

                    let (abs, sign) = builder.abs(pool_share.shares_delta);
                    let is_negative = builder.is_sign_negative(sign);

                    let ai_diff = builder.sub(max_ai, pool_share.public_pool_index);
                    let target = builder.mul_add(_1_bit_shifter, ai_diff, is_negative.target);
                    digest(builder, target, flag, &mut degree);

                    digest(builder, abs, flag, &mut degree);
                }
            }
        }

        let current_sum = builder.sequence_export(0).sum;
        builder.connect_quintic_ext(current_sum, self.target.output.evaluation);
        builder.connect(degree, self.target.output.degree);
    }

    fn populate_market_delta_hash(&mut self) {
        let builder = &mut self.builder;

        let remaining_before = builder.select(
            self.is_first_iteration,
            self.target.total_market_delta_count,
            self.target.public_inputs.remaining_market_delta_count,
        );
        let remaining_after = builder.sub(remaining_before, self.target.market_leaf_count);
        builder.register_range_check(self.target.total_market_delta_count, 32);
        builder.register_range_check(remaining_after, 32);
        builder.connect(
            remaining_after,
            self.target.output.remaining_market_delta_count,
        );

        // Account deltas may only start once every market delta of the batch has been consumed
        let all_markets_consumed = builder.is_zero(remaining_after);
        if let Some(first_account_evaluated) = self.should_evaluate.first() {
            let accounts_too_early =
                builder.and_not(*first_account_evaluated, all_markets_consumed);
            builder.assert_false(accounts_too_early);
        }

        let mut reached_end = builder.is_zero(self.target.market_leaf_count);
        let mut market_delta_hash = self.target.public_inputs.market_delta_hash;
        for (i, market_leaf) in self.target.market_leaves.iter().enumerate() {
            let should_evaluate = builder.not(reached_end);
            self.should_evaluate_market.push(should_evaluate);

            let mut inputs = market_delta_hash.elements.to_vec();
            inputs.extend(market_leaf.pack(builder));
            let extended = builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(inputs);
            market_delta_hash = builder.select_hash(should_evaluate, &extended, &market_delta_hash);

            let at_end = builder.is_equal_constant(self.target.market_leaf_count, (i + 1) as u64);
            reached_end = builder.or(reached_end, at_end);
        }
        builder.assert_true(reached_end);

        builder.connect_hashes(market_delta_hash, self.target.output.market_delta_hash);
    }

    /// Iterate through the sorted deltas and construct delta tree root from left to right, by going from the
    /// bottom to the top for each leaf.
    ///
    /// We hold a 2D array `path_matrix` which holds the sibling node hashes of each height for the path from
    /// the previous leaf to the root. We proceed by placing the current height hash to the left or right depending
    /// on if the current leaf is a right or left child. The other slot will be filled with corresponding empty
    /// level hash until the `lowest common ancestor` is reached. The 1D path going from the bottom to the root
    /// intersects with the same path of the previous leaf at some height - the intersection height.
    /// After the intersection height, sibling hashes will be taken from the previous iteration's path_matrix.
    fn populate_delta_tree(&mut self) {
        let mut prev_account_index = self.target.public_inputs.account_index;
        let mut path_matrix = self.target.public_inputs.path_matrix;
        let zero = self.builder.zero();
        let mut previous_leaf_index =
            self.builder
                .select(self.no_account_yet, zero, prev_account_index);

        let nil_account_index = self.builder.constant_i64(NIL_ACCOUNT_INDEX);
        let mut nil_account_hit = self.builder._false();
        for i in 0..self.target.leaves.len() {
            let leaf = self.target.leaves[i].clone();

            let is_current_leaf_nil_account =
                self.builder.is_equal(leaf.account_index, nil_account_index);
            nil_account_hit = self
                .builder
                .or(nil_account_hit, is_current_leaf_nil_account);

            // Once we hit a nil account, all subsequent leaves must also be nil accounts
            self.builder
                .connect(is_current_leaf_nil_account.target, nil_account_hit.target);

            // If nil account is not hit, we must evaluate the leaf and leaf can be empty if and only if it is a nil account
            let (mut current_height_hash, current_is_empty) = leaf.hash(&mut self.builder);
            self.builder
                .connect(current_is_empty.target, nil_account_hit.target);
            let should_evaluate = self.builder.not(nil_account_hit);
            self.should_evaluate.push(should_evaluate);

            let mut is_lt_enabled = should_evaluate;
            if i == 0 {
                is_lt_enabled = self.builder.and_not(is_lt_enabled, self.no_account_yet);
            }

            // do not allow holes in the account leaf data, every non-nil leaf must be evaluated
            self.builder.conditional_assert_lt(
                is_lt_enabled,
                prev_account_index,
                leaf.account_index,
                ACCOUNT_INDEX_BITS,
            ); // Make sure indices are sorted

            let curr_merkle_path = self
                .builder
                .split_le(leaf.account_index, ACCOUNT_MERKLE_LEVELS);

            let lca_height = self._get_lca_height(&curr_merkle_path, previous_leaf_index);

            prev_account_index =
                self.builder
                    .select(should_evaluate, leaf.account_index, prev_account_index);
            previous_leaf_index =
                self.builder
                    .select(should_evaluate, leaf.account_index, previous_leaf_index);

            // Insert current leaf to the tree. `current_height_hash` will be equal to the root after the loop.
            let mut has_common_parent = self.builder._false();

            for j in 0..ACCOUNT_MERKLE_LEVELS {
                let first_common_parent =
                    self.builder.is_equal_constant(lca_height, (j + 1) as u64);
                has_common_parent = self.builder.or(has_common_parent, first_common_parent);

                let is_current_leaf_right_child = curr_merkle_path[j];
                let other_non_empty_hash = self.builder.select_hash(
                    is_current_leaf_right_child,
                    &path_matrix[0][j],
                    &path_matrix[1][j],
                );
                let other_empty_hash = self.builder.constant_hash(EMPTY_DELTA_TREE_HASHES[j]);
                let other_hash_for_height = self.builder.select_hash(
                    has_common_parent,
                    &other_non_empty_hash,
                    &other_empty_hash,
                );

                let new_path_matrix_0_j = self.builder.select_hash(
                    is_current_leaf_right_child,
                    &other_hash_for_height,
                    &current_height_hash,
                );
                path_matrix[0][j] = self.builder.select_hash(
                    should_evaluate,
                    &new_path_matrix_0_j,
                    &path_matrix[0][j],
                );

                let new_path_matrix_1_j = self.builder.select_hash(
                    is_current_leaf_right_child,
                    &current_height_hash,
                    &other_hash_for_height,
                );
                path_matrix[1][j] = self.builder.select_hash(
                    should_evaluate,
                    &new_path_matrix_1_j,
                    &path_matrix[1][j],
                );

                current_height_hash = self
                    .builder
                    .hash_two_to_one(&path_matrix[0][j], &path_matrix[1][j]);
            }
        }

        self.builder
            .connect(prev_account_index, self.target.output.account_index);
        for i in 0..path_matrix.len() {
            for j in 0..path_matrix[i].len() {
                self.builder
                    .connect_hashes(path_matrix[i][j], self.target.output.path_matrix[i][j]);
            }
        }
    }

    /// Get the merkle paths of current and previous account indices to calculate the height of the
    /// lowest common ancestor. Minimum intersection height is 0, that is when both indices are the same.
    fn _get_lca_height(
        &mut self,
        current_merkle_path: &[BoolTarget],
        prev_account_index: Target,
    ) -> Target {
        let neg_one = self.builder.neg_one();

        let prev_merkle_path = self
            .builder
            .split_le(prev_account_index, ACCOUNT_MERKLE_LEVELS);

        // Traverse merkle paths in reverse and decrement the intersection height until we find the first non-common ancestor
        let mut lca_height = self.builder.constant_usize(ACCOUNT_MERKLE_LEVELS);
        let mut update = self.builder._true();
        for i in (0..ACCOUNT_MERKLE_LEVELS).rev() {
            let intersects = self
                .builder
                .is_equal(current_merkle_path[i].target, prev_merkle_path[i].target);
            update = self.builder.and(update, intersects);

            lca_height = self.builder.mul_add(update.target, neg_one, lca_height);
        }
        lca_height
    }
}

pub trait Circuit<
    C: GenericConfig<D, F = F>,
    F: RichField + Extendable<D> + Extendable<5>,
    const D: usize,
>
{
    fn define(config: CircuitConfig, account_count: usize, market_count: usize) -> Self;
    fn generate_witness(
        target: &DeltaTarget,
        witness: &DeltaWitness<F>,
    ) -> Result<PartialWitness<F>>;
    fn prove(
        target: &DeltaTarget,
        circuit_data: &CircuitData<F, C, D>,
        witness: &DeltaWitness<F>,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;
}

impl Circuit<C, F, D> for DeltaCircuit {
    fn define(config: CircuitConfig, account_count: usize, market_count: usize) -> Self {
        let mut circuit = Self::new(config, account_count, market_count);

        circuit.populate_delta_tree();

        circuit.populate_market_delta_hash();

        circuit.eval_delta_polynomial();

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn generate_witness(
        target: &DeltaTarget,
        witness: &DeltaWitness<F>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        target
            .leaves
            .iter()
            .zip_eq(witness.account_deltas.iter())
            .try_for_each(|(t, w)| pw.set_account_delta_leaf_target(t, w))?;

        for i in 0..2 {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                pw.set_hash_target(
                    target.public_inputs.path_matrix[i][j],
                    witness.path_matrix[i][j],
                )?;
            }
        }
        pw.set_target(
            target.public_inputs.account_index,
            F::from_noncanonical_i64(witness.previous_account_index),
        )?;
        pw.set_hash_target(
            target.public_inputs.market_delta_hash,
            witness.market_delta_hash,
        )?;
        pw.set_target(
            target.public_inputs.remaining_market_delta_count,
            F::from_noncanonical_i64(witness.remaining_market_delta_count),
        )?;
        pw.set_target(
            target.total_market_delta_count,
            F::from_canonical_u64(witness.total_market_delta_count),
        )?;
        pw.set_target(
            target.market_leaf_count,
            F::from_canonical_usize(witness.market_deltas.len()),
        )?;
        let padding_market_delta = MarketDeltaLeaf::default();
        for (i, market_leaf) in target.market_leaves.iter().enumerate() {
            let value = witness
                .market_deltas
                .get(i)
                .unwrap_or(&padding_market_delta);
            pw.set_market_delta_leaf_target(market_leaf, value)?;
        }
        pw.set_quintic_ext_target(
            target.public_inputs.evaluation_point,
            QuinticExtension([
                witness.x.elements[0],
                witness.x.elements[1],
                witness.x.elements[2],
                witness.x.elements[3],
                F::ZERO,
            ]),
        )?;

        Ok(pw)
    }

    fn prove(
        target: &DeltaTarget,
        circuit_data: &CircuitData<F, C, D>,
        witness: &DeltaWitness<F>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("delta prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(target, witness)?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }
}
