// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::target::{BoolTarget, Target};

use crate::tx::Tx;
use crate::types::approve_integrator::{
    APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN, ApproveIntegratorMessage, ApproveIntegratorMessageTarget,
};
use crate::types::change_pub_key::{
    CHANGE_PK_PUBLIC_INPUTS_LEN, ChangePubKeyMessage, ChangePubKeyMessageTarget,
};
use crate::types::config::Builder;
use crate::types::constants::*;
use crate::types::transfer::{TRANSFER_PUBLIC_INPUTS_LEN, TransferMessage, TransferMessageTarget};
use crate::uint::u8::U8Target;
use crate::utils::CircuitBuilderUtils;

pub const JUMP_STATE_SIZE: usize = 27;

#[derive(Debug, Clone, Copy)]
pub struct JumpState<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub last_active_tx_index: F,
    pub prev_new_state_root: HashOut<F>,
    pub prev_new_delta_root: HashOut<F>,
    pub run_start_prev_index: F,
    pub run_start_old_state_root: HashOut<F>,
    pub run_start_old_delta_root: HashOut<F>,
    pub coverage_hash: HashOut<F>,
    pub claims_hash: HashOut<F>,
    pub tx_count: F,
}

impl<F> JumpState<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub fn initial(state_root: HashOut<F>, delta_root: HashOut<F>) -> Self {
        Self {
            last_active_tx_index: F::NEG_ONE,
            prev_new_state_root: state_root,
            prev_new_delta_root: delta_root,
            run_start_prev_index: F::NEG_ONE,
            run_start_old_state_root: state_root,
            run_start_old_delta_root: delta_root,
            coverage_hash: HashOut::ZERO,
            claims_hash: HashOut::ZERO,
            tx_count: F::ZERO,
        }
    }

    pub fn from_public_inputs(pis: &[F]) -> Self {
        assert_eq!(pis.len(), JUMP_STATE_SIZE);
        Self {
            last_active_tx_index: pis[0],
            prev_new_state_root: HashOut::from_vec(pis[1..5].to_vec()),
            prev_new_delta_root: HashOut::from_vec(pis[5..9].to_vec()),
            run_start_prev_index: pis[9],
            run_start_old_state_root: HashOut::from_vec(pis[10..14].to_vec()),
            run_start_old_delta_root: HashOut::from_vec(pis[14..18].to_vec()),
            coverage_hash: HashOut::from_vec(pis[18..22].to_vec()),
            claims_hash: HashOut::from_vec(pis[22..26].to_vec()),
            tx_count: pis[26],
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        let mut v = vec![self.last_active_tx_index];
        v.extend(self.prev_new_state_root.elements);
        v.extend(self.prev_new_delta_root.elements);
        v.push(self.run_start_prev_index);
        v.extend(self.run_start_old_state_root.elements);
        v.extend(self.run_start_old_delta_root.elements);
        v.extend(self.coverage_hash.elements);
        v.extend(self.claims_hash.elements);
        v.push(self.tx_count);
        v
    }
}

#[derive(Debug, Clone, Copy)]
pub struct JumpStateTarget {
    pub last_active_tx_index: Target,
    pub prev_new_state_root: HashOutTarget,
    pub prev_new_delta_root: HashOutTarget,
    pub run_start_prev_index: Target,
    pub run_start_old_state_root: HashOutTarget,
    pub run_start_old_delta_root: HashOutTarget,
    pub coverage_hash: HashOutTarget,
    pub claims_hash: HashOutTarget,
    pub tx_count: Target,
}

impl Default for JumpStateTarget {
    fn default() -> Self {
        let default_hash = || HashOutTarget {
            elements: core::array::from_fn(|_| Target::default()),
        };
        Self {
            last_active_tx_index: Target::default(),
            prev_new_state_root: default_hash(),
            prev_new_delta_root: default_hash(),
            run_start_prev_index: Target::default(),
            run_start_old_state_root: default_hash(),
            run_start_old_delta_root: default_hash(),
            coverage_hash: default_hash(),
            claims_hash: default_hash(),
            tx_count: Target::default(),
        }
    }
}

impl JumpStateTarget {
    pub fn new(builder: &mut Builder) -> Self {
        Self {
            last_active_tx_index: builder.add_virtual_target(),
            prev_new_state_root: builder.add_virtual_hash(),
            prev_new_delta_root: builder.add_virtual_hash(),
            run_start_prev_index: builder.add_virtual_target(),
            run_start_old_state_root: builder.add_virtual_hash(),
            run_start_old_delta_root: builder.add_virtual_hash(),
            coverage_hash: builder.add_virtual_hash(),
            claims_hash: builder.add_virtual_hash(),
            tx_count: builder.add_virtual_target(),
        }
    }

    pub fn new_public(builder: &mut Builder) -> Self {
        let state = Self::new(builder);
        state.register_public_inputs(builder);
        state
    }

    pub fn register_public_inputs(&self, builder: &mut Builder) {
        builder.register_public_input(self.last_active_tx_index);
        builder.register_public_hashout(self.prev_new_state_root);
        builder.register_public_hashout(self.prev_new_delta_root);
        builder.register_public_input(self.run_start_prev_index);
        builder.register_public_hashout(self.run_start_old_state_root);
        builder.register_public_hashout(self.run_start_old_delta_root);
        builder.register_public_hashout(self.coverage_hash);
        builder.register_public_hashout(self.claims_hash);
        builder.register_public_input(self.tx_count);
    }

    pub fn from_public_inputs(pis: &[Target]) -> Self {
        assert_eq!(pis.len(), JUMP_STATE_SIZE);
        Self {
            last_active_tx_index: pis[0],
            prev_new_state_root: HashOutTarget::from_vec(pis[1..5].to_vec()),
            prev_new_delta_root: HashOutTarget::from_vec(pis[5..9].to_vec()),
            run_start_prev_index: pis[9],
            run_start_old_state_root: HashOutTarget::from_vec(pis[10..14].to_vec()),
            run_start_old_delta_root: HashOutTarget::from_vec(pis[14..18].to_vec()),
            coverage_hash: HashOutTarget::from_vec(pis[18..22].to_vec()),
            claims_hash: HashOutTarget::from_vec(pis[22..26].to_vec()),
            tx_count: pis[26],
        }
    }

    pub fn connect(builder: &mut Builder, a: &Self, b: &Self) {
        builder.connect(a.last_active_tx_index, b.last_active_tx_index);
        builder.connect_hashes(a.prev_new_state_root, b.prev_new_state_root);
        builder.connect_hashes(a.prev_new_delta_root, b.prev_new_delta_root);
        builder.connect(a.run_start_prev_index, b.run_start_prev_index);
        builder.connect_hashes(a.run_start_old_state_root, b.run_start_old_state_root);
        builder.connect_hashes(a.run_start_old_delta_root, b.run_start_old_delta_root);
        builder.connect_hashes(a.coverage_hash, b.coverage_hash);
        builder.connect_hashes(a.claims_hash, b.claims_hash);
        builder.connect(a.tx_count, b.tx_count);
    }

    pub fn conditional_assert_initial(
        &self,
        builder: &mut Builder,
        condition: BoolTarget,
        state_root: HashOutTarget,
        delta_root: HashOutTarget,
    ) {
        let one = builder.one();
        let prev_plus_one = builder.add(self.last_active_tx_index, one);
        builder.conditional_assert_zero(condition, prev_plus_one);
        let run_start_plus_one = builder.add(self.run_start_prev_index, one);
        builder.conditional_assert_zero(condition, run_start_plus_one);
        for i in 0..4 {
            builder.conditional_assert_eq(
                condition,
                self.prev_new_state_root.elements[i],
                state_root.elements[i],
            );
            builder.conditional_assert_eq(
                condition,
                self.prev_new_delta_root.elements[i],
                delta_root.elements[i],
            );
            builder.conditional_assert_eq(
                condition,
                self.run_start_old_state_root.elements[i],
                state_root.elements[i],
            );
            builder.conditional_assert_eq(
                condition,
                self.run_start_old_delta_root.elements[i],
                delta_root.elements[i],
            );
            builder.conditional_assert_zero(condition, self.coverage_hash.elements[i]);
            builder.conditional_assert_zero(condition, self.claims_hash.elements[i]);
        }
        builder.conditional_assert_zero(condition, self.tx_count);
    }
}

pub struct BlockTx<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub created_at: i64,

    pub state_metadata_hash: HashOut<F>,
    pub old_jump: JumpState<F>,

    pub txs: Vec<Tx<F>>,
}

#[derive(Debug, Clone)]
/// Public Block Transaction Witness
pub struct BlockTxWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub public_market_details_hash_after: HashOut<F>,

    pub new_account_delta_tree_root: HashOut<F>,
    pub new_validium_root: HashOut<F>,
    pub new_state_root: HashOut<F>,

    pub change_pub_key_message: ChangePubKeyMessage<F>,
    pub transfer_message: TransferMessage,
    pub approve_integrator_message: ApproveIntegratorMessage,

    pub on_chain_operations_count: u64,
    pub on_chain_operations_pub_data: [u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],

    pub priority_operations_count: u64,
    pub priority_operations_pub_data: [u8; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],

    pub old_jump: JumpState<F>,
    pub new_jump: JumpState<F>,
}

impl<F> BlockTxWitness<F>
where
    F: Field + Extendable<5> + RichField,
{
    /// Parse public inputs from proof into BlockWitness
    pub fn from_public_inputs(public_inputs: &[F]) -> Self {
        let public_market_details_hash_start = 12;
        let public_market_details_hash_end = public_market_details_hash_start + 4;

        let change_pub_key_message_start = public_market_details_hash_end;
        let change_pub_key_message_end = change_pub_key_message_start + CHANGE_PK_PUBLIC_INPUTS_LEN;

        let transfer_message_start = change_pub_key_message_end;
        let transfer_message_end = transfer_message_start + TRANSFER_PUBLIC_INPUTS_LEN;

        let approve_integrator_message_start = transfer_message_end;
        let approve_integrator_message_end =
            approve_integrator_message_start + APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN;

        // on_chain_pub_data_count
        let on_chain_pub_data_start = approve_integrator_message_end + 1;
        let on_chain_pub_data_end =
            on_chain_pub_data_start + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;

        // priority_pub_data_count
        let priority_pub_data_start = on_chain_pub_data_end + 1;
        let priority_pub_data_end =
            priority_pub_data_start + MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX;

        let old_jump_start = priority_pub_data_end;
        let new_jump_start = old_jump_start + JUMP_STATE_SIZE;
        let new_jump_end = new_jump_start + JUMP_STATE_SIZE;
        let total_pis_size = new_jump_end;

        assert_eq!(
            public_inputs.len(),
            total_pis_size,
            "Expected {} public inputs, but got {}",
            total_pis_size,
            public_inputs.len()
        );

        Self {
            new_account_delta_tree_root: HashOut::<F>::from_vec(public_inputs[0..4].to_vec()),
            new_validium_root: HashOut::<F>::from_vec(public_inputs[4..8].to_vec()),
            new_state_root: HashOut::<F>::from_vec(public_inputs[8..12].to_vec()),

            public_market_details_hash_after: HashOut::<F>::from_vec(
                public_inputs[public_market_details_hash_start..public_market_details_hash_end]
                    .to_vec(),
            ),

            change_pub_key_message: ChangePubKeyMessage::from_public_inputs(
                &public_inputs[change_pub_key_message_start..change_pub_key_message_end],
            ),
            transfer_message: TransferMessage::from_public_inputs(
                &public_inputs[transfer_message_start..transfer_message_end],
            ),
            approve_integrator_message: ApproveIntegratorMessage::from_public_inputs(
                &public_inputs[approve_integrator_message_start..approve_integrator_message_end],
            ),

            on_chain_operations_count: public_inputs[approve_integrator_message_end]
                .to_canonical_u64(),
            on_chain_operations_pub_data: core::array::from_fn(|index| {
                public_inputs[on_chain_pub_data_start + index].to_canonical_u64() as u8
            }),

            priority_operations_count: public_inputs[on_chain_pub_data_end].to_canonical_u64(),
            priority_operations_pub_data: core::array::from_fn(|index| {
                public_inputs[priority_pub_data_start + index].to_canonical_u64() as u8
            }),

            old_jump: JumpState::from_public_inputs(&public_inputs[old_jump_start..new_jump_start]),
            new_jump: JumpState::from_public_inputs(&public_inputs[new_jump_start..new_jump_end]),
        }
    }
}

#[derive(Debug)]
/// In circuit represantion of [`crate::block::BlockTxWitness`]
pub struct BlockTxWitnessTarget {
    pub public_market_details_hash_after: HashOutTarget,

    pub new_account_delta_tree_root: HashOutTarget,
    pub new_validium_root: HashOutTarget,
    pub new_state_root: HashOutTarget,

    pub change_pub_key_message: ChangePubKeyMessageTarget,
    pub transfer_message: TransferMessageTarget,
    pub approve_integrator_message: ApproveIntegratorMessageTarget,

    pub on_chain_operations_count: Target,
    pub on_chain_operations_pub_data: [U8Target; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE],

    pub priority_operations_count: Target,
    pub priority_operations_pub_data: [U8Target; MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX],

    pub old_jump: JumpStateTarget,
    pub new_jump: JumpStateTarget,
}

impl BlockTxWitnessTarget {
    /// Similar to [`BlockTxWitness::from_public_inputs`], parses proof target.
    /// Follows the same order as [`crate::block_tx_constraints::BlockCircuit::register_public_inputs`]
    pub fn from_public_inputs(pis: &[Target]) -> Self {
        let public_market_details_hash_start = 12;
        let public_market_details_hash_end = public_market_details_hash_start + 4;

        let change_pub_key_message_start = public_market_details_hash_end;
        let change_pub_key_message_end = change_pub_key_message_start + CHANGE_PK_PUBLIC_INPUTS_LEN;

        let transfer_message_start = change_pub_key_message_end;
        let transfer_message_end = transfer_message_start + TRANSFER_PUBLIC_INPUTS_LEN;

        let approve_integrator_message_start = transfer_message_end;
        let approve_integrator_message_end =
            approve_integrator_message_start + APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN;

        let on_chain_pub_data_start = approve_integrator_message_end + 1;
        let on_chain_pub_data_end =
            on_chain_pub_data_start + ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE;

        let priority_pub_data_start = on_chain_pub_data_end + 1;
        let priority_pub_data_end =
            priority_pub_data_start + MAX_PRIORITY_OPERATIONS_PUB_DATA_BYTES_PER_TX;

        let old_jump_start = priority_pub_data_end;
        let new_jump_start = old_jump_start + JUMP_STATE_SIZE;
        let new_jump_end = new_jump_start + JUMP_STATE_SIZE;
        let total_pis_size = new_jump_end;

        assert_eq!(
            pis.len(),
            total_pis_size,
            "Expected {} public inputs, but got {}",
            total_pis_size,
            pis.len()
        );

        Self {
            new_account_delta_tree_root: HashOutTarget::from_vec(pis[0..4].to_vec()),
            new_validium_root: HashOutTarget::from_vec(pis[4..8].to_vec()),
            new_state_root: HashOutTarget::from_vec(pis[8..12].to_vec()),

            public_market_details_hash_after: HashOutTarget::from_vec(
                pis[public_market_details_hash_start..public_market_details_hash_end].to_vec(),
            ),

            change_pub_key_message: ChangePubKeyMessageTarget::from_public_inputs(
                &pis[change_pub_key_message_start..change_pub_key_message_end],
            ),
            transfer_message: TransferMessageTarget::from_public_inputs(
                &pis[transfer_message_start..transfer_message_end],
            ),
            approve_integrator_message: ApproveIntegratorMessageTarget::from_public_inputs(
                &pis[approve_integrator_message_start..approve_integrator_message_end],
            ),

            on_chain_operations_count: pis[approve_integrator_message_end],
            on_chain_operations_pub_data: pis[on_chain_pub_data_start..on_chain_pub_data_end]
                .iter()
                .map(|&x| U8Target(x))
                .collect::<Vec<U8Target>>()
                .try_into()
                .unwrap(),

            priority_operations_count: pis[on_chain_pub_data_end],
            priority_operations_pub_data: pis[priority_pub_data_start..priority_pub_data_end]
                .iter()
                .map(|&x| U8Target(x))
                .collect::<Vec<U8Target>>()
                .try_into()
                .unwrap(),

            old_jump: JumpStateTarget::from_public_inputs(&pis[old_jump_start..new_jump_start]),
            new_jump: JumpStateTarget::from_public_inputs(&pis[new_jump_start..new_jump_end]),
        }
    }
}
