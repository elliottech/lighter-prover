// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use circuit::bigint::big_u16::WitnessBigInt16;
use circuit::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt};
use circuit::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint, WitnessBigUint};
use circuit::bigint::div_rem::CircuitBuilderBiguintDivRem;
use circuit::bool_utils::CircuitBuilderBoolUtils;
use circuit::byte::split::CircuitBuilderByteSplit;
use circuit::circuit_logger::CircuitBuilderLogging;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::keccak::keccak::{CircuitBuilderKeccak, KeccakOutputTarget};
use circuit::merkle_helpers::{account_index_to_merkle_path, verify_merkle_proof};
use circuit::types::asset::ensure_valid_asset_index;
use circuit::types::config::{
    BIG_U64_LIMBS, BIG_U96_LIMBS, BIG_U128_LIMBS, BIG_U160_LIMBS, Builder, C, D, F,
};
use circuit::types::constants::{
    ACCOUNT_MERKLE_LEVELS, ASSET_LIST_SIZE_BITS, INSURANCE_FUND_ACCOUNT_TYPE,
    LIGHTER_STAKING_POOL_ACCOUNT_TYPE, MAX_ASSET_INDEX, MIN_ASSET_INDEX, NIL_ACCOUNT_INDEX,
    POSITION_LIST_SIZE, PUBLIC_POOL_ACCOUNT_TYPE, SHARES_LIST_SIZE, USDC_ASSET_INDEX,
    USDC_TO_COLLATERAL_MULTIPLIER,
};
use circuit::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use circuit::utils::CircuitBuilderUtils;
use log::Level;
use num::BigUint;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use serde::Deserialize;
use serde_with::serde_as;

use crate::pubdata_account::{PubdataAccount, PubdataAccountTarget, PubdataAccountTargetWitness};
use crate::pubdata_market::{
    PubdataMarketDetails, PubdataMarketDetailsTarget, all_public_market_details_hash,
};

pub const DESERT_NUM_ACCOUNTS: usize = 1 + SHARES_LIST_SIZE;

#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(bound = "")]
pub struct InnerDesertExitWitness<F>
where
    F: Field + RichField,
{
    #[serde(rename = "ai", default)]
    pub asset_index: u16,
    #[serde(rename = "mai", default)]
    pub master_account_index: u64,
    #[serde(rename = "acc")]
    pub accounts: [PubdataAccount; DESERT_NUM_ACCOUNTS], // Main account and pools
    #[serde(rename = "tav")]
    #[serde(deserialize_with = "crate::deserializers::biguint_from_str")]
    pub total_account_value: BigUint,
    #[serde(rename = "apdtr")]
    #[serde(deserialize_with = "circuit::deserializers::hash_out")]
    pub account_pub_data_tree_root: HashOut<F>,
    #[serde(rename = "mpapd")]
    #[serde(deserialize_with = "crate::deserializers::account_tree_merkle_proofs_for_desert")]
    pub account_pub_data_tree_merkle_proofs:
        [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; DESERT_NUM_ACCOUNTS],

    #[serde(rename = "pmda")]
    #[serde_as(as = "[_; POSITION_LIST_SIZE]")]
    pub all_market_details: [PubdataMarketDetails; POSITION_LIST_SIZE],

    #[serde(rename = "vr")]
    #[serde(deserialize_with = "circuit::deserializers::hash_out")]
    pub validium_root: HashOut<F>,
    #[serde(rename = "sr")]
    #[serde(deserialize_with = "circuit::deserializers::hash_out")]
    pub state_root: HashOut<F>,
}

#[derive(Debug)]
pub struct InnerDesertExitTarget {
    pub exit_commitment: KeccakOutputTarget, // Public input

    pub asset_index: Target,
    pub master_account_index: Target,
    pub accounts: [PubdataAccountTarget; DESERT_NUM_ACCOUNTS], // Main account and pools
    pub balance: BigUintTarget, // Will be compared against calculated usdc
    pub account_pub_data_tree_root: HashOutTarget,
    pub account_pub_data_tree_merkle_proofs:
        [[HashOutTarget; ACCOUNT_MERKLE_LEVELS]; DESERT_NUM_ACCOUNTS],

    pub public_market_details: [PubdataMarketDetailsTarget; POSITION_LIST_SIZE],

    pub validium_root: HashOutTarget,
    pub state_root: HashOutTarget,
}

#[derive(Debug)]
pub struct InnerDesertExitCircuit {
    pub builder: Builder,
    pub target: InnerDesertExitTarget,
}

impl InnerDesertExitCircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let mut builder = Builder::new(config);

        Self {
            target: InnerDesertExitTarget {
                exit_commitment: builder.add_virtual_keccak_output_public_input_safe(),

                asset_index: builder.add_virtual_target(),
                master_account_index: builder.add_virtual_target(),
                accounts: core::array::from_fn(|_| PubdataAccountTarget::new(&mut builder)),
                balance: builder.add_virtual_biguint_target_safe(BIG_U96_LIMBS),
                account_pub_data_tree_root: builder.add_virtual_hash(),
                account_pub_data_tree_merkle_proofs: core::array::from_fn(|_| {
                    core::array::from_fn(|_| builder.add_virtual_hash())
                }),
                public_market_details: [(); POSITION_LIST_SIZE]
                    .map(|_| PubdataMarketDetailsTarget::new(&mut builder)),
                validium_root: builder.add_virtual_hash(),
                state_root: builder.add_virtual_hash(),
            },

            builder,
        }
    }

    fn validate_asset_index(&mut self) {
        let _true = self.builder._true();
        ensure_valid_asset_index(&mut self.builder, _true, self.target.asset_index);
        self.builder
            .register_range_check(self.target.asset_index, ASSET_LIST_SIZE_BITS);
    }

    fn verify_accounts(&mut self) {
        let is_account_empty = self.target.accounts[0].is_empty(&mut self.builder);
        self.builder.assert_false(is_account_empty);

        for i in 0..DESERT_NUM_ACCOUNTS {
            let account = &self.target.accounts[i];
            let merkle_proof = self.target.account_pub_data_tree_merkle_proofs[i];

            let merkle_path =
                account_index_to_merkle_path(&mut self.builder, account.account_index);
            let account_pubdata_hash = account.hash(&mut self.builder);

            verify_merkle_proof(
                &mut self.builder,
                &self.target.account_pub_data_tree_root,
                account_pubdata_hash,
                merkle_proof,
                merkle_path,
            );
        }

        // Verify integrity of pool accounts
        let nil_account_index = self
            .builder
            .constant(F::from_canonical_u64(NIL_ACCOUNT_INDEX as u64));
        for i in 1..DESERT_NUM_ACCOUNTS {
            let is_empty_pool =
                self.target.accounts[0].public_pool_shares[i - 1].is_empty(&mut self.builder);
            self.builder.conditional_assert_eq(
                is_empty_pool,
                self.target.accounts[i].account_index,
                nil_account_index,
            );
            let is_not_empty_pool = self.builder.not(is_empty_pool);
            self.builder.conditional_assert_eq(
                is_not_empty_pool,
                self.target.accounts[i].account_index,
                self.target.accounts[0].public_pool_shares[i - 1].public_pool_index,
            );
        }

        // Validate account types for special accounts
        // MAI = 0 <==> AI = 0 or account type is staking pool
        let is_ai_zero = self.builder.is_zero(self.target.accounts[0].account_index);
        let is_type_staking_pool = self.builder.is_equal_constant(
            self.target.accounts[0].account_type,
            LIGHTER_STAKING_POOL_ACCOUNT_TYPE as u64,
        );
        let mai_zero_condition = self.builder.or(is_ai_zero, is_type_staking_pool);
        let is_mai_zero = self.builder.is_zero(self.target.master_account_index);
        self.builder
            .connect(is_mai_zero.target, mai_zero_condition.target);
        // MAI = 1 <==> AI = 1 or account type is insurance fund
        let is_ai_one = self
            .builder
            .is_equal_constant(self.target.accounts[0].account_index, 1);
        let is_type_insurance_fund = self.builder.is_equal_constant(
            self.target.accounts[0].account_type,
            INSURANCE_FUND_ACCOUNT_TYPE as u64,
        );
        let mai_one_condition = self.builder.or(is_ai_one, is_type_insurance_fund);
        let is_mai_one = self
            .builder
            .is_equal_constant(self.target.master_account_index, 1);
        self.builder
            .connect(is_mai_one.target, mai_one_condition.target);
        // If MAI = 0 or 1 <==> l1 address = 0
        let l1_address_zero = self
            .builder
            .is_zero_biguint(&self.target.accounts[0].l1_address);
        let mai_zero_or_one = self.builder.or(is_mai_zero, is_mai_one);
        self.builder
            .connect(mai_zero_or_one.target, l1_address_zero.target);
    }

    fn verify_state_root(&mut self) {
        let public_market_details_hash =
            all_public_market_details_hash(&mut self.builder, &self.target.public_market_details);

        let state_root = self.builder.hash_n_to_one(&[
            self.target.account_pub_data_tree_root,
            public_market_details_hash,
            self.target.validium_root,
        ]);

        self.builder
            .connect_hashes(state_root, self.target.state_root);
    }

    fn get_usdc_balance(&mut self) -> BigIntTarget {
        // Aggregate usdc from positions, pools, and aggregated collateral.
        let mut usdc_balance = {
            let positions_usdc_component =
                self.get_extended_usdc_component_from_positions(self.target.accounts[0].clone());
            let extended_aggregated_collateral = self.usdc_to_extended_usdc(
                &self.target.accounts[0].aggregated_assets[USDC_ASSET_INDEX as usize].clone(),
            );
            let usdc_except_pools = self.builder.add_bigint_non_carry(
                &positions_usdc_component,
                &extended_aggregated_collateral,
                BIG_U128_LIMBS,
            );
            let pools_usdc_component = self.get_usdc_component_from_pools();

            self.builder
                .println_bigint(&positions_usdc_component, "Positions usdc Component");
            self.builder
                .println_bigint(&pools_usdc_component, "Public Pools usdc Component");

            self.builder.add_bigint_non_carry(
                &usdc_except_pools,
                &pools_usdc_component,
                BIG_U128_LIMBS,
            )
        };

        // Negative to zero
        usdc_balance = {
            let zero_bigint = self.builder.zero_bigint();
            let is_usdc_negative = self.builder.is_sign_negative(usdc_balance.sign);
            self.builder
                .select_bigint(is_usdc_negative, &zero_bigint, &usdc_balance)
        };

        self.builder
            .println_bigint(&usdc_balance, "Main Account usdc (extended)");

        // If main account is a pool: usdc = (usdc * operatorShares) / totalShares
        usdc_balance = {
            let operator_shares_big = self
                .builder
                .target_to_biguint(self.target.accounts[0].public_pool_info.operator_shares);
            let usdc_times_operator_shares = self.builder.mul_bigint_with_biguint_non_carry(
                &usdc_balance,
                &operator_shares_big,
                BIG_U160_LIMBS,
            );
            let total_shares_big = self
                .builder
                .target_to_biguint(self.target.accounts[0].public_pool_info.total_shares);
            let proportioned_pool_usdc = self.builder.euclidian_div_by_biguint(
                &usdc_times_operator_shares,
                &total_shares_big,
                BIG_U160_LIMBS,
            );

            self.builder.println_bigint(
                &proportioned_pool_usdc,
                "Main Account usdc if it's is a public pool (extended)",
            );

            let is_public_pool = self.builder.is_equal_constant(
                self.target.accounts[0].account_type,
                PUBLIC_POOL_ACCOUNT_TYPE as u64,
            );
            let is_insurance_fund = self.builder.is_equal_constant(
                self.target.accounts[0].account_type,
                INSURANCE_FUND_ACCOUNT_TYPE as u64,
            );
            let is_pool_operator = self.builder.or(is_public_pool, is_insurance_fund);
            self.builder
                .select_bigint(is_pool_operator, &proportioned_pool_usdc, &usdc_balance)
        };

        // Divide back the multiplier
        let usdc_to_collateral_multiplier = self
            .builder
            .constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        usdc_balance.abs = self
            .builder
            .div_biguint(&usdc_balance.abs, &usdc_to_collateral_multiplier);

        usdc_balance
    }

    fn validate_total_balance(&mut self) {
        let mut balance = self.get_usdc_balance();

        let zero_bigint = self.builder.zero_bigint();
        for i in MIN_ASSET_INDEX..=MAX_ASSET_INDEX {
            if i == USDC_ASSET_INDEX {
                continue;
            }
            let i_target = self.builder.constant_u64(i);
            let is_equal = self.builder.is_equal(self.target.asset_index, i_target);

            balance = self.builder.select_bigint(
                is_equal,
                &self.target.accounts[0].aggregated_assets[i as usize],
                &balance,
            );
        }

        let is_positive = self.builder.is_sign_positive(balance.sign);
        balance = self
            .builder
            .select_bigint(is_positive, &balance, &zero_bigint);

        self.builder
            .connect_biguint(&balance.abs, &self.target.balance);
    }

    fn get_usdc_component_from_pools(&mut self) -> BigIntTarget {
        let usdc_to_collateral_multiplier = self
            .builder
            .constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));

        let mut total_pool_value = self.builder.zero_bigint();
        for i in 1..DESERT_NUM_ACCOUNTS {
            let main_account = self.target.accounts[0].clone();
            let pool_account = self.target.accounts[i].clone();

            let pool_total_account_value = {
                let positions_usdc_component =
                    self.get_extended_usdc_component_from_positions(pool_account.clone());
                let extended_aggregated_collateral = self.usdc_to_extended_usdc(
                    &pool_account.aggregated_assets[USDC_ASSET_INDEX as usize],
                );
                self.builder.add_bigint_non_carry(
                    &positions_usdc_component,
                    &extended_aggregated_collateral,
                    BIG_U128_LIMBS,
                )
            };

            let big_share_amount = self
                .builder
                .target_to_biguint(main_account.public_pool_shares[i - 1].share_amount);
            let share_amount_mul_total_account_value =
                self.builder.mul_bigint_with_biguint_non_carry(
                    &pool_total_account_value,
                    &big_share_amount,
                    BIG_U160_LIMBS,
                );
            let total_shares = self
                .builder
                .target_to_biguint(pool_account.public_pool_info.total_shares);
            let total_shares_mul_usdc_to_collateral_multiplier = self
                .builder
                .mul_biguint(&total_shares, &usdc_to_collateral_multiplier);

            let mut value = self.builder.euclidian_div_by_biguint(
                &share_amount_mul_total_account_value,
                &total_shares_mul_usdc_to_collateral_multiplier,
                BIG_U160_LIMBS,
            );
            value.abs = self.builder.trim_biguint(&value.abs, BIG_U96_LIMBS);

            let extended_value = self.usdc_to_extended_usdc(&value);

            total_pool_value = self.builder.add_bigint_non_carry(
                &total_pool_value,
                &extended_value,
                BIG_U128_LIMBS,
            );
        }

        total_pool_value
    }

    fn get_extended_usdc_component_from_positions(
        &mut self,
        account: PubdataAccountTarget,
    ) -> BigIntTarget {
        let mut base_funding_deltas_sum = self.builder.zero_bigint();
        let mut notional_values_sum = self.builder.zero_bigint();

        for i in 0..POSITION_LIST_SIZE {
            let pos = &account.positions[i];

            // Notional
            let base_notional_value = self.target.public_market_details[i]
                .get_position_base_notional_value(&mut self.builder, &pos.position);
            notional_values_sum = self.builder.add_bigint_non_carry(
                &notional_values_sum,
                &base_notional_value,
                BIG_U64_LIMBS,
            );

            // Funding delta
            let base_funding_delta_for_position = self.target.public_market_details[i]
                .get_funding_delta_for_position_and_market(&mut self.builder, pos);
            base_funding_deltas_sum = self.builder.add_bigint_non_carry(
                &base_funding_deltas_sum,
                &base_funding_delta_for_position,
                BIG_U96_LIMBS,
            );
        }

        let notional_values_sum_extended = self.usdc_to_extended_usdc(&notional_values_sum);

        self.builder.add_bigint_non_carry(
            &notional_values_sum_extended,
            &base_funding_deltas_sum, // Already extended
            BIG_U96_LIMBS,
        )
    }

    fn usdc_to_extended_usdc(&mut self, amount: &BigIntTarget) -> BigIntTarget {
        let usdc_to_collateral_multiplier = self
            .builder
            .constant_biguint(&BigUint::from(USDC_TO_COLLATERAL_MULTIPLIER));
        self.builder.mul_bigint_with_biguint_non_carry(
            amount,
            &usdc_to_collateral_multiplier,
            BIG_U96_LIMBS,
        )
    }

    /// keccak(abi.encodePacked(stateRoot, _accountIndex, _masterAccountIndex, _l1Address, _assetIndex, _totalAccountValue))
    fn set_exit_commitment(&mut self) {
        let exit_commitment = calculate_exit_commitment(
            &mut self.builder,
            &self.target.state_root,
            self.target.accounts[0].account_index,
            self.target.master_account_index,
            &self.target.accounts[0].l1_address,
            self.target.asset_index,
            &self.target.balance,
        );

        self.builder
            .connect_keccak_output(exit_commitment, self.target.exit_commitment);
    }

    pub fn define(config: CircuitConfig) -> Self {
        let mut circuit = Self::new(config);

        circuit.validate_asset_index();

        circuit.verify_accounts();

        circuit.verify_state_root();

        circuit.validate_total_balance();

        circuit.set_exit_commitment();

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    pub fn prove(
        target: &InnerDesertExitTarget,
        circuit_data: &CircuitData<F, C, D>,
        witness: &InnerDesertExitWitness<F>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("desert prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(target, witness)?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    fn generate_witness(
        target: &InnerDesertExitTarget,
        witness: &InnerDesertExitWitness<F>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_target(
            target.asset_index,
            F::from_canonical_u16(witness.asset_index),
        )?;

        pw.set_target(
            target.master_account_index,
            F::from_canonical_u64(witness.master_account_index),
        )?;

        for i in 0..witness.accounts.len() {
            pw.set_pubdata_account_target(&target.accounts[i], &witness.accounts[i])?;
        }

        pw.set_biguint_target(&target.balance, &witness.total_account_value)?;

        pw.set_hash_target(
            target.account_pub_data_tree_root,
            witness.account_pub_data_tree_root,
        )?;

        for i in 0..witness.accounts.len() {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                pw.set_hash_target(
                    target.account_pub_data_tree_merkle_proofs[i][j],
                    witness.account_pub_data_tree_merkle_proofs[i][j],
                )?;
            }
        }

        for i in 0..witness.all_market_details.len() {
            pw.set_bigint_u16_target(
                &target.public_market_details[i].funding_rate_prefix_sum,
                &witness.all_market_details[i].funding_rate_prefix_sum,
            )?;
            pw.set_target(
                target.public_market_details[i].mark_price,
                F::from_canonical_u32(witness.all_market_details[i].mark_price),
            )?;
            pw.set_target(
                target.public_market_details[i].quote_multiplier,
                F::from_canonical_u32(witness.all_market_details[i].quote_multiplier),
            )?;
        }

        pw.set_hash_target(target.validium_root, witness.validium_root)?;
        pw.set_hash_target(target.state_root, witness.state_root)?;

        Ok(pw)
    }
}

fn calculate_exit_commitment(
    builder: &mut Builder,
    state_root: &HashOutTarget,
    account_index: Target,
    master_account_index: Target,
    l1_address: &BigUintTarget,
    asset_index: Target,
    balance: &BigUintTarget,
) -> KeccakOutputTarget {
    let mut elems = vec![];

    let state_root_bytes = state_root
        .elements
        .iter()
        .flat_map(|elem| builder.split_bytes(*elem, 8))
        .collect::<Vec<_>>();
    elems.extend_from_slice(&state_root_bytes);

    let mut account_index_bytes = builder.split_bytes(account_index, 6);
    account_index_bytes.reverse();
    elems.extend_from_slice(&account_index_bytes);

    let mut master_account_index_bytes = builder.split_bytes(master_account_index, 6);
    master_account_index_bytes.reverse();
    elems.extend_from_slice(&master_account_index_bytes);

    let mut l1_address_bytes = l1_address
        .limbs
        .iter()
        .flat_map(|elem| builder.split_bytes(elem.0, 4))
        .collect::<Vec<_>>();
    l1_address_bytes.reverse();
    elems.extend_from_slice(&l1_address_bytes);

    let mut asset_index_bytes = builder.split_bytes(asset_index, 2);
    asset_index_bytes.reverse();
    elems.extend_from_slice(&asset_index_bytes);

    // balance is taken as u128 in the contract
    let mut balance_resized = balance.clone();
    balance_resized
        .limbs
        .resize(BIG_U128_LIMBS, builder.zero_u32());
    let mut balance_bytes = balance_resized
        .limbs
        .iter()
        .flat_map(|limb| builder.split_bytes(limb.0, 4))
        .collect::<Vec<_>>();
    balance_bytes.reverse();
    elems.extend_from_slice(&balance_bytes);

    builder.keccak256_circuit(elems)
}
