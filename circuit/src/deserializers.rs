// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use std::collections::HashMap;

use hex::FromHex;
use num::{BigInt, BigUint, Num};
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::secp256k1_base::Secp256K1Base;
use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use serde::Deserialize;
use serde::de::{self, Deserializer};

use crate::blob::constants::*;
use crate::ecdsa::curve::curve_types::AffinePoint;
use crate::ecdsa::curve::ecdsa::{ECDSAPublicKey, ECDSASignature};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::eddsa::curve::scalar_field::ECgFp5Scalar;
use crate::eddsa::schnorr::SchnorrSig;
use crate::keccak::helpers::u8_array_to_bits;
use crate::types::account_delta::{PositionDelta, PublicPoolShareDelta};
use crate::types::account_position::AccountPosition;
use crate::types::constants::{
    ACCOUNT_MERKLE_LEVELS, ACCOUNT_ORDERS_MERKLE_LEVELS, API_KEY_MERKLE_LEVELS, ASSET_LIST_SIZE,
    ASSET_MERKLE_LEVELS, KECCAK_HASH_OUT_BIT_SIZE, KECCAK_HASH_OUT_BYTE_SIZE, MARKET_MERKLE_LEVELS,
    NB_ACCOUNT_ORDERS_PATHS_PER_TX, NB_ACCOUNTS_PER_TX, NB_ASSETS_PER_TX,
    ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE, POSITION_LIST_SIZE, POSITION_MERKLE_LEVELS,
    REGISTER_STACK_SIZE, SHARES_DELTA_LIST_SIZE,
};
use crate::types::register::{BaseRegisterInfo, RegisterStack};

type ProofData = Vec<Vec<[u64; 4]>>;

pub fn int_to_biguint<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
where
    D: Deserializer<'de>,
{
    let num: u128 = Deserialize::deserialize(deserializer)?;
    Ok(BigUint::from(num))
}

pub fn int_to_bigint<'de, D>(deserializer: D) -> Result<BigInt, D::Error>
where
    D: Deserializer<'de>,
{
    let num: i128 = Deserialize::deserialize(deserializer)?;
    Ok(BigInt::from(num))
}

pub fn strategies<'de, D, const SIZE: usize>(deserializer: D) -> Result<[BigInt; SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Debug, Clone, Deserialize)]
    struct StrategyValue {
        col: i128,
    }

    let elements: HashMap<String, StrategyValue> = Deserialize::deserialize(deserializer)?;
    let mut result = [BigInt::ZERO; SIZE];

    if elements.len() > SIZE {
        return Err(de::Error::custom(format!(
            "Expected at most {} elements, got {}",
            SIZE,
            elements.len()
        )));
    }

    for (idx, value) in elements.into_iter() {
        let index = idx.parse::<usize>().map_err(|err| {
            de::Error::custom(format!("Failed to parse strategy index: {}, {}", idx, err))
        })?;
        if index >= SIZE {
            return Err(de::Error::custom(format!(
                "Strategy index out of bounds: {}",
                index
            )));
        }
        result[index] = BigInt::from(value.col);
    }

    Ok(result)
}

pub fn aggregated_balances<'de, D>(deserializer: D) -> Result<[BigInt; NB_ASSETS_PER_TX], D::Error>
where
    D: Deserializer<'de>,
{
    let nums: Vec<i128> = Deserialize::deserialize(deserializer)?;
    if nums.len() != NB_ASSETS_PER_TX {
        return Err(serde::de::Error::custom(format!(
            "Expected {} elements, got {}",
            NB_ASSETS_PER_TX,
            nums.len()
        )));
    }
    let mut result = [BigInt::ZERO; NB_ASSETS_PER_TX];
    for (i, num) in nums.into_iter().enumerate() {
        result[i] = BigInt::from(num);
    }
    Ok(result)
}

pub fn int_to_bigint_list<'de, D, const SIZE: usize>(
    deserializer: D,
) -> Result<[BigInt; SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let nums: Vec<i128> = Deserialize::deserialize(deserializer)?;
    if nums.len() != SIZE {
        return Err(serde::de::Error::custom(format!(
            "Expected {} elements, got {}",
            SIZE,
            nums.len()
        )));
    }
    let mut result = [BigInt::ZERO; SIZE];
    for (i, num) in nums.into_iter().enumerate() {
        result[i] = BigInt::from(num);
    }
    Ok(result)
}

pub fn all_aggregated_asset_deltas<'de, D>(
    deserializer: D,
) -> Result<[BigInt; ASSET_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: HashMap<String, i128> = Deserialize::deserialize(deserializer)?;

    let mut result = [BigInt::ZERO; ASSET_LIST_SIZE];

    for (idx, value) in elements.into_iter() {
        match idx.parse::<usize>() {
            Ok(index) => {
                if index >= ASSET_LIST_SIZE {
                    return Err(serde::de::Error::custom(format!(
                        "Asset index out of bounds: {}",
                        index
                    )));
                }
                result[index] = BigInt::from(value);
            }
            Err(err) => {
                return Err(serde::de::Error::custom(format!(
                    "Failed to parse asset index: {}, {}",
                    idx, err
                )));
            }
        }
    }

    Ok(result)
}

pub fn aggregated_asset_deltas<'de, D>(
    deserializer: D,
) -> Result<[BigInt; NB_ASSETS_PER_TX], D::Error>
where
    D: Deserializer<'de>,
{
    int_to_bigint_list::<D, NB_ASSETS_PER_TX>(deserializer)
}

pub fn l1_address_to_biguint<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
where
    D: Deserializer<'de>,
{
    let l1_address: String = Deserialize::deserialize(deserializer)?;
    if let Some(l1_address_hex_part) = l1_address.strip_prefix("0x") {
        return BigUint::from_str_radix(l1_address_hex_part, 16).map_err(|err| {
            serde::de::Error::custom(format!(
                "Error while parsing l1_address to BigUint {}. {}",
                l1_address, err
            ))
        });
    }

    BigUint::from_str_radix(&l1_address, 16).map_err(|err| {
        serde::de::Error::custom(format!(
            "Error while parsing l1_address to BigUint {}. {}",
            l1_address, err
        ))
    })
}

pub fn hex_string_to_biguint<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
where
    D: Deserializer<'de>,
{
    let h: String = Deserialize::deserialize(deserializer)?;
    if let Some(h) = h.strip_prefix("0x") {
        return BigUint::from_str_radix(h, 16).map_err(|err| {
            serde::de::Error::custom(format!(
                "Error while parsing string to BigUint {}. {}",
                h, err
            ))
        });
    }

    BigUint::from_str_radix(&h, 16).map_err(|err| {
        serde::de::Error::custom(format!(
            "Error while parsing string to BigUint {}. {}",
            h, err
        ))
    })
}

pub fn hex_to_bits<'de, D>(deserializer: D) -> Result<[bool; KECCAK_HASH_OUT_BIT_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let h: String = Deserialize::deserialize(deserializer)?;

    let bits = u8_array_to_bits(&Vec::from_hex(&h).map_err(serde::de::Error::custom)?);

    let mut result = [false; KECCAK_HASH_OUT_BIT_SIZE];
    result.copy_from_slice(&bits[..KECCAK_HASH_OUT_BIT_SIZE]);
    Ok(result)
}

pub fn hex_to_bytes<'de, D>(deserializer: D) -> Result<[u8; KECCAK_HASH_OUT_BYTE_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let h: String = Deserialize::deserialize(deserializer)?;

    let bytes = Vec::from_hex(&h).map_err(serde::de::Error::custom)?;

    let mut result = [0; KECCAK_HASH_OUT_BYTE_SIZE];
    result.copy_from_slice(&bytes[..KECCAK_HASH_OUT_BYTE_SIZE]);
    Ok(result)
}

pub fn blob_bytes<'de, D>(deserializer: D) -> Result<Box<[u8; BLOB_DATA_BYTES_COUNT]>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut result = Box::new([0; BLOB_DATA_BYTES_COUNT]);

    let h: String = Deserialize::deserialize(deserializer)?;
    let hex = Vec::from_hex(&h).map_err(serde::de::Error::custom)?;
    if hex.len() != BLOB_DATA_BYTES_COUNT {
        return Err(serde::de::Error::custom(format!(
            "Blob data has incorrect length: expected {}, got {}",
            BLOB_DATA_BYTES_COUNT,
            hex.len()
        )));
    }

    result.copy_from_slice(&hex[..BLOB_DATA_BYTES_COUNT]);

    Ok(result)
}

pub fn signature<'de, D>(deserializer: D) -> Result<SchnorrSig, D::Error>
where
    D: Deserializer<'de>,
{
    let elements: [u64; 10] = Deserialize::deserialize(deserializer)?;
    Ok(SchnorrSig {
        s: ECgFp5Scalar([
            elements[0],
            elements[1],
            elements[2],
            elements[3],
            elements[4],
        ]),
        e: ECgFp5Scalar([
            elements[5],
            elements[6],
            elements[7],
            elements[8],
            elements[9],
        ]),
    })
}

struct ArrayVisitor;
impl<'de> de::Visitor<'de> for ArrayVisitor {
    type Value = [u8; 64];

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a byte array of length 64")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<[u8; 64], A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let mut array = [0u8; 64];
        for i in 0..64 {
            if let Some(value) = seq.next_element()? {
                array[i] = value;
            }
        }
        Ok(array)
    }
}

pub fn l1_signature<'de, D>(deserializer: D) -> Result<Option<ECDSASignature<Secp256K1>>, D::Error>
where
    D: Deserializer<'de>,
{
    let bytes = deserializer.deserialize_tuple(64, ArrayVisitor)?;
    Ok(Some(ECDSASignature {
        r: Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_be(&bytes[..32])),
        s: Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_be(&bytes[32..])),
    }))
}

pub fn l1_pub_key<'de, D>(deserializer: D) -> Result<Option<ECDSAPublicKey<Secp256K1>>, D::Error>
where
    D: Deserializer<'de>,
{
    let bytes = deserializer.deserialize_tuple(64, ArrayVisitor)?;
    let mut point = ECDSAPublicKey::<Secp256K1>(AffinePoint {
        x: Secp256K1Base::from_noncanonical_biguint(BigUint::from_bytes_be(&bytes[..32])),
        y: Secp256K1Base::from_noncanonical_biguint(BigUint::from_bytes_be(&bytes[32..])),
        zero: false,
    });
    if point.0.x.is_zero() && point.0.y.is_zero() {
        point.0.zero = true;
    }
    Ok(Some(point))
}

pub fn price_updates<'de, D>(deserializer: D) -> Result<[u32; POSITION_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: HashMap<String, u32> = Deserialize::deserialize(deserializer)?;
    let mut result = [0u32; POSITION_LIST_SIZE];
    for (i, element) in elements.into_iter() {
        if let Ok(index) = i.parse::<usize>() {
            if index >= POSITION_LIST_SIZE {
                return Err(serde::de::Error::custom(format!(
                    "Price update index out of bounds: {}",
                    index
                )));
            }
            result[index] = element;
        } else {
            return Err(serde::de::Error::custom(format!(
                "Failed to parse price update index: {}",
                i
            )));
        }
    }
    Ok(result)
}

pub fn default_price_updates() -> [u32; POSITION_LIST_SIZE] {
    core::array::from_fn(|_| 0u32)
}

pub fn positions<'de, D>(deserializer: D) -> Result<[AccountPosition; POSITION_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: HashMap<String, AccountPosition> = Deserialize::deserialize(deserializer)?;

    let mut result: [AccountPosition; POSITION_LIST_SIZE] =
        core::array::from_fn(|_| AccountPosition::default());

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

pub fn positions_delta<'de, D>(
    deserializer: D,
) -> Result<[PositionDelta; POSITION_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: HashMap<String, PositionDelta> = Deserialize::deserialize(deserializer)?;

    let mut result: [PositionDelta; POSITION_LIST_SIZE] =
        core::array::from_fn(|_| PositionDelta::default());

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

pub fn pub_key<'de, D, F>(deserializer: D) -> Result<QuinticExtension<F>, D::Error>
where
    D: Deserializer<'de>,
    F: Field + Extendable<5> + RichField,
{
    let elements: [u64; 5] = Deserialize::deserialize(deserializer)?;
    Ok(u64_array_to_quintic_extension(elements))
}

pub fn u64_array_to_quintic_extension<F>(elements: [u64; 5]) -> QuinticExtension<F>
where
    F: Field + Extendable<5> + RichField,
{
    QuinticExtension::<F>([
        F::from_canonical_u64(elements[0]),
        F::from_canonical_u64(elements[1]),
        F::from_canonical_u64(elements[2]),
        F::from_canonical_u64(elements[3]),
        F::from_canonical_u64(elements[4]),
    ])
}

pub fn hash_out<'de, D, F>(deserializer: D) -> Result<HashOut<F>, D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: [u64; 4] = Deserialize::deserialize(deserializer)?;
    Ok(u64_array_to_hash_out(elements))
}

pub fn hash_out_from_hex<'de, D, F>(deserializer: D) -> Result<HashOut<F>, D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let as_string: String = Deserialize::deserialize(deserializer)?;
    Ok(HashOut::<F> {
        elements: Vec::from_hex(as_string.strip_prefix("0x").unwrap_or(&as_string))
            .unwrap()
            .chunks(8)
            .map(|chunk| F::from_canonical_u64(u64::from_le_bytes(chunk.try_into().unwrap())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    })
}

pub fn path_matrix<'de, D, F>(
    deserializer: D,
) -> Result<[[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; 2], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: Vec<Vec<Vec<u64>>> = Deserialize::deserialize(deserializer)?;
    if elements.len() != 2 {
        return Err(serde::de::Error::custom("Outer dimension must be 2"));
    }
    let mut result: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()));
    for i in 0..2 {
        if elements[i].len() != ACCOUNT_MERKLE_LEVELS {
            return Err(serde::de::Error::custom("Inner dimension mismatch"));
        }
        for j in 0..ACCOUNT_MERKLE_LEVELS {
            let arr: [u64; 4] = elements[i][j]
                .as_slice()
                .try_into()
                .map_err(|_| serde::de::Error::custom("Expected array of length 4"))?;
            result[i][j] = u64_array_to_hash_out(arr);
        }
    }
    Ok(result)
}

pub fn market_tree_merkle_proof<'de, D, F>(
    deserializer: D,
) -> Result<[HashOut<F>; MARKET_MERKLE_LEVELS], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: [[u64; 4]; MARKET_MERKLE_LEVELS] = Deserialize::deserialize(deserializer)?;
    let mut proof: [HashOut<F>; MARKET_MERKLE_LEVELS] = Default::default();
    for i in 0..MARKET_MERKLE_LEVELS {
        proof[i] = u64_array_to_hash_out(elements[i]);
    }
    Ok(proof)
}

pub fn asset_tree_merkle_proof<'de, D, F>(
    deserializer: D,
) -> Result<[[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: ProofData = Deserialize::deserialize(deserializer)?;
    let mut proof: [[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX] =
        std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()));

    for account in 0..NB_ASSETS_PER_TX {
        for i in 0..ASSET_MERKLE_LEVELS {
            proof[account][i] = u64_array_to_hash_out(elements[account][i]);
        }
    }

    Ok(proof)
}

pub fn account_orders_tree_merkle_proof<'de, D, F>(
    deserializer: D,
) -> Result<[[HashOut<F>; ACCOUNT_ORDERS_MERKLE_LEVELS]; NB_ACCOUNT_ORDERS_PATHS_PER_TX], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: ProofData = Deserialize::deserialize(deserializer)?;
    let mut proof: [[HashOut<F>; ACCOUNT_ORDERS_MERKLE_LEVELS]; NB_ACCOUNT_ORDERS_PATHS_PER_TX] =
        std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()));

    for account in 0..NB_ACCOUNT_ORDERS_PATHS_PER_TX {
        for i in 0..ACCOUNT_ORDERS_MERKLE_LEVELS {
            proof[account][i] = u64_array_to_hash_out(elements[account][i]);
        }
    }

    Ok(proof)
}

pub fn position_delta_merkle_proofs<'de, D, F>(
    deserializer: D,
) -> Result<[[HashOut<F>; POSITION_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX - 1], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: ProofData = Deserialize::deserialize(deserializer)?;
    let mut proof: [[HashOut<F>; POSITION_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX - 1] =
        std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()));

    for account in 0..NB_ACCOUNTS_PER_TX - 1 {
        for i in 0..POSITION_MERKLE_LEVELS {
            proof[account][i] = u64_array_to_hash_out(elements[account][i]);
        }
    }

    Ok(proof)
}

pub fn public_pool_shares_delta<'de, D>(
    deserializer: D,
) -> Result<[PublicPoolShareDelta; SHARES_DELTA_LIST_SIZE], D::Error>
where
    D: Deserializer<'de>,
{
    let elements: Vec<PublicPoolShareDelta> = Deserialize::deserialize(deserializer)?;
    assert!(elements.len() <= SHARES_DELTA_LIST_SIZE);
    let mut result: [PublicPoolShareDelta; SHARES_DELTA_LIST_SIZE] =
        std::array::from_fn(|_| PublicPoolShareDelta::default());
    result[..elements.len()].copy_from_slice(&elements[..]);
    Ok(result)
}

pub fn api_key_tree_merkle_proof<'de, D, F>(
    deserializer: D,
) -> Result<[HashOut<F>; API_KEY_MERKLE_LEVELS], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: [[u64; 4]; API_KEY_MERKLE_LEVELS] = Deserialize::deserialize(deserializer)?;
    let mut proof: [HashOut<F>; API_KEY_MERKLE_LEVELS] = Default::default();
    for i in 0..API_KEY_MERKLE_LEVELS {
        proof[i] = u64_array_to_hash_out(elements[i]);
    }
    Ok(proof)
}

pub fn account_tree_merkle_proofs<'de, D, F>(
    deserializer: D,
) -> Result<[[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: ProofData = Deserialize::deserialize(deserializer)?;
    let mut proof: [[HashOut<F>; ACCOUNT_MERKLE_LEVELS]; NB_ACCOUNTS_PER_TX] =
        std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()));

    for account in 0..NB_ACCOUNTS_PER_TX {
        for i in 0..ACCOUNT_MERKLE_LEVELS {
            proof[account][i] = u64_array_to_hash_out(elements[account][i]);
        }
    }
    Ok(proof)
}

pub fn asset_tree_merkle_proofs<'de, D, F>(
    deserializer: D,
) -> Result<[[[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX], D::Error>
where
    D: Deserializer<'de>,
    F: Field,
{
    let elements: Vec<Vec<Vec<[u64; 4]>>> = Deserialize::deserialize(deserializer)?;
    let mut proof: [[[HashOut<F>; ASSET_MERKLE_LEVELS]; NB_ASSETS_PER_TX]; NB_ACCOUNTS_PER_TX] =
        std::array::from_fn(|_| {
            std::array::from_fn(|_| std::array::from_fn(|_| HashOut::<F>::default()))
        });

    for i in 0..NB_ACCOUNTS_PER_TX {
        for j in 0..NB_ASSETS_PER_TX {
            for k in 0..ASSET_MERKLE_LEVELS {
                proof[i][j][k] = u64_array_to_hash_out(elements[i][j][k]);
            }
        }
    }

    Ok(proof)
}

pub fn on_chain_pub_data_vector<'de, D>(
    deserializer: D,
) -> Result<Vec<[u8; ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE]>, D::Error>
where
    D: Deserializer<'de>,
{
    let pub_data_hex: Vec<String> = Deserialize::deserialize(deserializer)?;
    Ok(pub_data_hex
        .iter()
        .map(|h| {
            let bytes = &Vec::from_hex(h).unwrap();
            assert_eq!(bytes.len(), ON_CHAIN_OPERATIONS_PUB_DATA_BYTES_SIZE);
            core::array::from_fn(|i| bytes[i])
        })
        .collect::<Vec<_>>())
}

pub fn u64_array_to_hash_out<F>(elements: [u64; 4]) -> HashOut<F>
where
    F: Field,
{
    let field_elements: [F; 4] = [
        F::from_canonical_u64(elements[0]),
        F::from_canonical_u64(elements[1]),
        F::from_canonical_u64(elements[2]),
        F::from_canonical_u64(elements[3]),
    ];
    HashOut::<F>::from_partial(&field_elements)
}

pub fn register_stack<'de, D>(deserializer: D) -> Result<RegisterStack, D::Error>
where
    D: Deserializer<'de>,
{
    let mut register_stack: Vec<Option<BaseRegisterInfo>> = Deserialize::deserialize(deserializer)?;
    register_stack.resize(REGISTER_STACK_SIZE, None);
    let stack: [BaseRegisterInfo; REGISTER_STACK_SIZE] = register_stack
        .iter()
        .map(|h| h.unwrap_or(BaseRegisterInfo::empty()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let count = stack.iter().filter(|&x| !x.is_empty()).count();
    Ok(RegisterStack { stack, count })
}
