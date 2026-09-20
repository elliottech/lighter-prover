// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::{BoolTarget, Target};

use crate::hash_utils::CircuitBuilderHashUtils;
use crate::types::config::Builder;
use crate::types::constants::*;

pub fn account_index_to_merkle_path(
    builder: &mut Builder,
    account_index: Target,
) -> [BoolTarget; ACCOUNT_MERKLE_LEVELS] {
    let bits = builder.split_le(account_index, ACCOUNT_MERKLE_LEVELS);

    assert!(bits.len() == ACCOUNT_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

pub fn account_order_index_to_merkle_path(
    builder: &mut Builder,
    index: Target, // oid or cloid
) -> [BoolTarget; ACCOUNT_ORDERS_MERKLE_LEVELS] {
    let bits = builder.split_le(index, ACCOUNT_ORDERS_MERKLE_LEVELS);

    assert!(bits.len() == ACCOUNT_ORDERS_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

/// Same as account_order_index_to_merkle_path but uses CLIENT_ORDER_INDEX_BITS if index is known to be a client order index
pub fn account_client_order_index_to_merkle_path(
    builder: &mut Builder,
    index: Target, // cloid
) -> [BoolTarget; ACCOUNT_ORDERS_MERKLE_LEVELS] {
    let mut bits = builder.split_le(index, CLIENT_ORDER_INDEX_BITS);
    bits.resize_with(ACCOUNT_ORDERS_MERKLE_LEVELS, || builder._false());

    assert!(bits.len() == ACCOUNT_ORDERS_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

pub fn api_key_index_to_merkle_path(
    builder: &mut Builder,
    api_key_index: Target,
) -> [BoolTarget; API_KEY_MERKLE_LEVELS] {
    let bits = builder.split_le(api_key_index, API_KEY_MERKLE_LEVELS);

    assert!(bits.len() == API_KEY_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

pub fn asset_index_to_merkle_path(
    builder: &mut Builder,
    asset_index: Target,
) -> [BoolTarget; ASSET_MERKLE_LEVELS] {
    let bits = builder.split_le(asset_index, ASSET_MERKLE_LEVELS);

    assert!(bits.len() == ASSET_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

pub fn perps_market_index_to_merkle_path(
    builder: &mut Builder,
    market_index: Target,
) -> [BoolTarget; POSITION_MERKLE_LEVELS] {
    let bits = builder.split_le(market_index, POSITION_MERKLE_LEVELS);

    assert!(bits.len() == POSITION_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

pub fn market_index_to_merkle_path(
    builder: &mut Builder,
    market_index: Target,
) -> [BoolTarget; MARKET_MERKLE_LEVELS] {
    let bits = builder.split_le(market_index, MARKET_MERKLE_LEVELS);

    assert!(bits.len() == MARKET_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

pub fn public_market_index_to_merkle_path(
    builder: &mut Builder,
    public_market_index: Target,
) -> [BoolTarget; PUBLIC_MARKET_INDEX_MERKLE_LEVELS] {
    let bits = builder.split_le(public_market_index, PUBLIC_MARKET_INDEX_MERKLE_LEVELS);

    assert!(bits.len() == PUBLIC_MARKET_INDEX_MERKLE_LEVELS);
    bits.try_into().unwrap()
}

#[track_caller]
pub fn verify_merkle_proof<const L: usize>(
    builder: &mut Builder,
    root: &HashOutTarget,
    leaf: HashOutTarget,
    proofs: [HashOutTarget; L],
    helpers: [BoolTarget; L],
) {
    let mut state = leaf;

    for (&helper, &sibling) in helpers.iter().zip(&proofs) {
        state = builder.hash_two_to_one_swap(&state, &sibling, helper);
    }

    builder.connect_hashes(state, *root);
}

#[track_caller]
pub fn conditional_verify_merkle_proof<const L: usize>(
    builder: &mut Builder,
    condition: BoolTarget,
    root: &HashOutTarget,
    leaf: HashOutTarget,
    proofs: [HashOutTarget; L],
    helpers: [BoolTarget; L],
) {
    let mut state = leaf;

    for (&helper, &sibling) in helpers.iter().zip(&proofs) {
        state = builder.hash_two_to_one_swap(&state, &sibling, helper);
    }

    builder.conditional_assert_eq_hash(condition, &state, root);
}

pub fn try_verify_merkle_proof<const L: usize>(
    builder: &mut Builder,
    root: &HashOutTarget,
    leaf: HashOutTarget,
    proofs: [HashOutTarget; L],
    helpers: [BoolTarget; L],
) -> BoolTarget {
    let mut state = leaf;

    for (&helper, &sibling) in helpers.iter().zip(&proofs) {
        state = builder.hash_two_to_one_swap(&state, &sibling, helper);
    }

    builder.is_equal_hash(&state, root)
}

pub fn recalculate_root<const L: usize>(
    builder: &mut Builder,
    leaf: HashOutTarget,
    proofs: [HashOutTarget; L],
    helpers: [BoolTarget; L],
) -> HashOutTarget {
    let mut state = leaf;

    for (&helper, &sibling) in helpers.iter().zip(&proofs) {
        state = builder.hash_two_to_one_swap(&state, &sibling, helper);
    }

    state
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;
    use crate::bool_utils::CircuitBuilderBoolUtils;
    use crate::builder::Builder;

    #[test]
    fn test_account_index_to_merkle_path() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_ecc_config();
        let pw = PartialWitness::new();
        let mut builder = Builder::<F, D>::new(config);

        let x = 11;
        let x_target = builder.constant(F::from_canonical_u64(x));
        let bits = account_index_to_merkle_path(&mut builder, x_target);

        builder.assert_true(bits[0]);
        builder.assert_true(bits[1]);
        builder.assert_false(bits[2]);
        builder.assert_true(bits[3]);
        for i in 4..ACCOUNT_MERKLE_LEVELS {
            builder.assert_false(bits[i]);
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
