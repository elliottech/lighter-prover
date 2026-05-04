// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;

use circuit::deserializers::u64_array_to_hash_out;
use circuit::types::constants::{ACCOUNT_MERKLE_LEVELS, ASSET_LIST_SIZE, POSITION_LIST_SIZE};
use num::{BigInt, BigUint, Num};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::HashOut;
use serde::de::{Deserialize, Deserializer};

use crate::inner_circuit::DESERT_NUM_ACCOUNTS;
use crate::pubdata_account::PubdataAccountPosition;

type ProofData = Vec<Vec<[u64; 4]>>;

pub fn biguint_from_str<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    BigUint::from_str_radix(&s, 10).map_err(serde::de::Error::custom)
}

pub fn account_tree_merkle_proofs_for_desert<'de, D, F>(
    deserializer: D,
) -> Result<[[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; DESERT_NUM_ACCOUNTS], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: ProofData = Deserialize::deserialize(deserializer)?;
    let mut proof: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; DESERT_NUM_ACCOUNTS] =
        std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()));

    for account in 0..DESERT_NUM_ACCOUNTS {
        for i in 0..ACCOUNT_MERKLE_LEVELS {
            proof[account][i] = u64_array_to_hash_out(elements[account][i]);
        }
    }
    Ok(proof)
}

pub fn aggregated_assets<'de, D>(deserializer: D) -> Result<[BigInt; ASSET_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: HashMap<String, i128> = Deserialize::deserialize(deserializer)?; // read numbers

    let mut result: [BigInt; ASSET_LIST_SIZE] = core::array::from_fn(|_| BigInt::default());

    for (idx, value) in elements.into_iter() {
        let index = idx.parse::<usize>().map_err(|err| {
            serde::de::Error::custom(format!("Failed to parse asset index: {}, {}", idx, err))
        })?;
        if index >= ASSET_LIST_SIZE {
            return Err(serde::de::Error::custom(format!(
                "Asset index out of bounds: {}",
                index
            )));
        }
        result[index] = BigInt::from(value);
    }

    Ok(result)
}

pub fn pubdata_positions<'de, D>(
    deserializer: D,
) -> Result<[PubdataAccountPosition; POSITION_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: HashMap<String, PubdataAccountPosition> = Deserialize::deserialize(deserializer)?;

    let mut result: [PubdataAccountPosition; POSITION_LIST_SIZE] =
        core::array::from_fn(|_| PubdataAccountPosition::default());

    for (idx, element) in elements.into_iter() {
        match idx.parse::<usize>() {
            Ok(index) => {
                if index >= POSITION_LIST_SIZE {
                    return Err(serde::de::Error::custom(format!(
                        "Position index out of bounds: {}",
                        index
                    )));
                }
                result[index] = element;
            }
            Err(err) => {
                return Err(serde::de::Error::custom(format!(
                    "Failed to parse position index: {}, {}",
                    idx, err
                )));
            }
        }
    }

    Ok(result)
}
