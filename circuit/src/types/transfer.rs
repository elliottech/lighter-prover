// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use itertools::Itertools;
use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::secp256k1_base::Secp256K1Base;
use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::ecdsa::curve::curve_types::AffinePoint;
use crate::ecdsa::curve::ecdsa::{ECDSAPublicKey, ECDSASignature};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::ecdsa::gadgets::curve::AffinePointTarget;
use crate::ecdsa::gadgets::ecdsa::{
    CircuitBuilderECDSAPublicKey, CircuitBuilderECDSASignature, ECDSAPublicKeyTarget,
    ECDSASignatureTarget,
};
use crate::keccak::keccak::CircuitBuilderKeccak;
use crate::nonnative::NonNativeTarget;
use crate::types::config::{BIG_U64_LIMBS, BIG_U160_LIMBS, BIG_U256_LIMBS, Builder};
use crate::types::constants::TRANSFER_MEMO_BYTES;
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::{CircuitBuilderUtils, bytes_to_hex, split_le_base16};

#[derive(Debug, Clone)]
pub struct TransferMessage {
    pub from_account_index: i64,
    pub api_key_index: u8,
    pub to_account_index: i64,

    pub from_route_type: u8,
    pub to_route_type: u8,
    pub asset_index: i16,
    pub chain_id: u32,

    pub nonce: i64,
    pub amount: i64,
    pub fee: i64,
    pub memo: [u8; TRANSFER_MEMO_BYTES],
    pub l1_address: BigUint,
    pub l1_signature: ECDSASignature<Secp256K1>,
    pub l1_pk: ECDSAPublicKey<Secp256K1>,
}

impl Default for TransferMessage {
    fn default() -> Self {
        Self {
            from_account_index: 0,
            api_key_index: 0,
            to_account_index: 0,
            from_route_type: 0,
            to_route_type: 0,
            asset_index: 0,
            chain_id: 0,
            nonce: 0,
            amount: 0,
            fee: 0,
            memo: [0; TRANSFER_MEMO_BYTES],
            l1_address: BigUint::default(),
            l1_signature: ECDSASignature::<Secp256K1>::default(),
            l1_pk: ECDSAPublicKey::<Secp256K1>::default(),
        }
    }
}

pub const TRANSFER_PUBLIC_INPUTS_LEN: usize = 81;

impl TransferMessage {
    pub fn from_public_inputs<F: Field + Extendable<5> + RichField>(pis: &[F]) -> Self {
        let from_account_index = pis[0].to_canonical_u64() as i64;
        let api_key_index = pis[1].to_canonical_u64() as u8;
        let to_account_index = pis[2].to_canonical_u64() as i64;

        let from_route_type = pis[3].to_canonical_u64() as u8;
        let to_route_type = pis[4].to_canonical_u64() as u8;
        let asset_index = pis[5].to_canonical_u64() as i16;
        let chain_id = pis[6].to_canonical_u64() as u32;

        let nonce = pis[7].to_canonical_u64() as i64;
        let amount = (pis[8].to_canonical_u64() + (pis[9].to_canonical_u64() << 32)) as i64;
        let fee = (pis[10].to_canonical_u64() + (pis[11].to_canonical_u64() << 32)) as i64;
        let memo = core::array::from_fn(|i| pis[12 + i].to_canonical_u64() as u8);

        // Convert u32 limbs to BigUint
        let mut l1_address = BigUint::ZERO;
        for i in 0..5 {
            l1_address += BigUint::from(pis[44 + i].to_canonical_u64()) << (i * 32);
        }
        let mut r = BigUint::ZERO;
        for i in 0..8 {
            r += BigUint::from(pis[49 + i].to_canonical_u64()) << (i * 32);
        }
        let mut s = BigUint::ZERO;
        for i in 0..8 {
            s += BigUint::from(pis[57 + i].to_canonical_u64()) << (i * 32);
        }
        let l1_signature = ECDSASignature {
            r: Secp256K1Scalar::from_noncanonical_biguint(r),
            s: Secp256K1Scalar::from_noncanonical_biguint(s),
        };
        let mut x = BigUint::ZERO;
        for i in 0..8 {
            x += BigUint::from(pis[65 + i].to_canonical_u64()) << (i * 32);
        }
        let mut y = BigUint::ZERO;
        for i in 0..8 {
            y += BigUint::from(pis[73 + i].to_canonical_u64()) << (i * 32);
        }
        let l1_pk = ECDSAPublicKey(AffinePoint {
            x: Secp256K1Base::from_noncanonical_biguint(x),
            y: Secp256K1Base::from_noncanonical_biguint(y),
            zero: false,
        });

        Self {
            from_account_index,
            api_key_index,
            to_account_index,
            from_route_type,
            to_route_type,
            asset_index,
            chain_id,
            nonce,
            amount,
            fee,
            memo,
            l1_address,
            l1_signature,
            l1_pk,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TransferMessageTarget {
    pub from_account_index: Target,
    pub api_key_index: Target,
    pub to_account_index: Target,
    pub from_route_type: Target,
    pub to_route_type: Target,
    pub asset_index: Target,
    pub chain_id: Target,
    pub nonce: Target,
    pub amount: BigUintTarget,
    pub fee: BigUintTarget,
    pub memo: [U8Target; TRANSFER_MEMO_BYTES],

    pub l1_address: BigUintTarget,
    pub l1_signature: ECDSASignatureTarget<Secp256K1>,
    pub l1_pk: ECDSAPublicKeyTarget<Secp256K1>,
}

impl TransferMessageTarget {
    pub fn from_public_inputs(pis: &[Target]) -> Self {
        let from_account_index = pis[0];
        let api_key_index = pis[1];
        let to_account_index = pis[2];
        let from_route_type = pis[3];
        let to_route_type = pis[4];
        let asset_index = pis[5];
        let chain_id = pis[6];
        let nonce = pis[7];
        let amount = BigUintTarget::from(&pis[8..10]);
        let fee = BigUintTarget::from(&pis[10..12]);
        let memo = core::array::from_fn(|i| U8Target(pis[12 + i]));

        let l1_address = BigUintTarget::from(&pis[44..49]);
        let l1_signature = ECDSASignatureTarget {
            r: NonNativeTarget {
                value: BigUintTarget::from(&pis[49..57]),
                _phantom: std::marker::PhantomData,
            },
            s: NonNativeTarget {
                value: BigUintTarget::from(&pis[57..65]),
                _phantom: std::marker::PhantomData,
            },
        };
        let l1_pk = ECDSAPublicKeyTarget(AffinePointTarget {
            x: NonNativeTarget {
                value: BigUintTarget::from(&pis[65..73]),
                _phantom: std::marker::PhantomData,
            },
            y: NonNativeTarget {
                value: BigUintTarget::from(&pis[73..81]),
                _phantom: std::marker::PhantomData,
            },
        });

        Self {
            from_account_index,
            api_key_index,
            to_account_index,
            from_route_type,
            to_route_type,
            asset_index,
            chain_id,
            nonce,
            amount,
            fee,
            memo,
            l1_address,
            l1_signature,
            l1_pk,
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.from_account_index);
        builder.register_public_input(self.api_key_index);
        builder.register_public_input(self.to_account_index);
        builder.register_public_input(self.from_route_type);
        builder.register_public_input(self.to_route_type);
        builder.register_public_input(self.asset_index);
        builder.register_public_input(self.chain_id);
        builder.register_public_input(self.nonce);
        builder.register_public_input_biguint(&self.amount);
        builder.register_public_input_biguint(&self.fee);
        builder.register_public_u8_inputs(&self.memo);
        builder.register_public_input_biguint(&self.l1_address);
        builder.register_public_input_biguint(&self.l1_signature.r.value);
        builder.register_public_input_biguint(&self.l1_signature.s.value);
        builder.register_public_input_biguint(&self.l1_pk.0.x.value);
        builder.register_public_input_biguint(&self.l1_pk.0.y.value);
    }

    pub fn new(builder: &mut Builder) -> Self {
        Self {
            from_account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            to_account_index: builder.add_virtual_target(),
            from_route_type: builder.add_virtual_target(),
            to_route_type: builder.add_virtual_target(),
            asset_index: builder.add_virtual_target(),
            chain_id: builder.add_virtual_target(),
            nonce: builder.add_virtual_target(),
            amount: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS), // safe because connected to safe inputs
            fee: builder.add_virtual_biguint_target_unsafe(BIG_U64_LIMBS), // safe because connected to safe inputs
            memo: builder
                .add_virtual_u8_targets_safe(TRANSFER_MEMO_BYTES)
                .try_into()
                .unwrap(), // safe because connected to safe inputs
            l1_address: builder.add_virtual_biguint_target_unsafe(BIG_U160_LIMBS), // safe because connected to safe inputs
            l1_signature: ECDSASignatureTarget {
                r: NonNativeTarget {
                    value: builder.add_virtual_biguint_target_unsafe(BIG_U256_LIMBS), // safe because connected to safe inputs
                    _phantom: core::marker::PhantomData,
                },
                s: NonNativeTarget {
                    value: builder.add_virtual_biguint_target_unsafe(BIG_U256_LIMBS), // safe because connected to safe inputs
                    _phantom: core::marker::PhantomData,
                },
            },
            l1_pk: ECDSAPublicKeyTarget(AffinePointTarget {
                x: NonNativeTarget {
                    value: builder.add_virtual_biguint_target_unsafe(BIG_U256_LIMBS), // safe because connected to safe inputs
                    _phantom: core::marker::PhantomData,
                },
                y: NonNativeTarget {
                    value: builder.add_virtual_biguint_target_unsafe(BIG_U256_LIMBS), // safe because connected to safe inputs
                    _phantom: core::marker::PhantomData,
                },
            }),
        }
    }

    pub fn new_public(builder: &mut Builder) -> Self {
        Self {
            from_account_index: builder.add_virtual_public_input(),
            api_key_index: builder.add_virtual_public_input(),
            to_account_index: builder.add_virtual_public_input(),
            from_route_type: builder.add_virtual_public_input(),
            to_route_type: builder.add_virtual_public_input(),
            asset_index: builder.add_virtual_public_input(),
            chain_id: builder.add_virtual_public_input(),
            nonce: builder.add_virtual_public_input(),
            amount: builder.add_virtual_biguint_public_input_unsafe(BIG_U64_LIMBS), // Safe because it is connected to public witness from constrained circuit
            fee: builder.add_virtual_biguint_public_input_unsafe(BIG_U64_LIMBS), // Safe because it is connected to public witness from constrained circuit
            memo: builder
                .add_virtual_public_u8_targets_safe(TRANSFER_MEMO_BYTES)
                .try_into()
                .unwrap(), // Safe because it is connected to public witness from constrained circuit
            l1_address: builder.add_virtual_biguint_public_input_unsafe(BIG_U160_LIMBS), // Safe because it is connected to public witness from constrained circuit
            l1_signature: ECDSASignatureTarget {
                r: NonNativeTarget {
                    value: builder.add_virtual_biguint_public_input_unsafe(BIG_U256_LIMBS), // Safe because it is connected to public witness from constrained circuit
                    _phantom: core::marker::PhantomData,
                },
                s: NonNativeTarget {
                    value: builder.add_virtual_biguint_public_input_unsafe(BIG_U256_LIMBS), // Safe because it is connected to public witness from constrained circuit
                    _phantom: core::marker::PhantomData,
                },
            },
            l1_pk: ECDSAPublicKeyTarget(AffinePointTarget {
                x: NonNativeTarget {
                    value: builder.add_virtual_biguint_public_input_unsafe(BIG_U256_LIMBS), // Safe because it is connected to public witness from constrained circuit
                    _phantom: core::marker::PhantomData,
                },
                y: NonNativeTarget {
                    value: builder.add_virtual_biguint_public_input_unsafe(BIG_U256_LIMBS), // Safe because it is connected to public witness from constrained circuit
                    _phantom: core::marker::PhantomData,
                },
            }),
        }
    }

    pub fn select(builder: &mut Builder, flag: BoolTarget, a: &Self, b: &Self) -> Self {
        Self {
            from_account_index: builder.select(flag, a.from_account_index, b.from_account_index),
            api_key_index: builder.select(flag, a.api_key_index, b.api_key_index),
            to_account_index: builder.select(flag, a.to_account_index, b.to_account_index),
            from_route_type: builder.select(flag, a.from_route_type, b.from_route_type),
            to_route_type: builder.select(flag, a.to_route_type, b.to_route_type),
            asset_index: builder.select(flag, a.asset_index, b.asset_index),
            chain_id: builder.select(flag, a.chain_id, b.chain_id),
            nonce: builder.select(flag, a.nonce, b.nonce),
            amount: builder.select_biguint(flag, &a.amount, &b.amount),
            fee: builder.select_biguint(flag, &a.fee, &b.fee),
            memo: builder.select_arr_u8(flag, &a.memo, &b.memo),

            l1_address: builder.select_biguint(flag, &a.l1_address, &b.l1_address),
            l1_signature: builder.select_ecdsa_signature(flag, &a.l1_signature, &b.l1_signature),
            l1_pk: builder.select_ecdsa_public_key(flag, &a.l1_pk, &b.l1_pk),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            from_account_index: builder.zero(),
            api_key_index: builder.zero(),
            to_account_index: builder.zero(),
            from_route_type: builder.zero(),
            to_route_type: builder.zero(),
            asset_index: builder.zero(),
            chain_id: builder.zero(),
            nonce: builder.zero(),
            amount: builder.zero_biguint(),
            fee: builder.zero_biguint(),
            memo: core::array::from_fn(|_| builder.zero_u8()),

            l1_address: BigUintTarget {
                limbs: vec![builder.zero_u32(); BIG_U160_LIMBS],
            },
            l1_signature: ECDSASignatureTarget {
                r: NonNativeTarget {
                    value: BigUintTarget {
                        limbs: vec![builder.zero_u32(); BIG_U256_LIMBS],
                    },
                    _phantom: core::marker::PhantomData,
                },
                s: NonNativeTarget {
                    value: BigUintTarget {
                        limbs: vec![builder.zero_u32(); BIG_U256_LIMBS],
                    },
                    _phantom: core::marker::PhantomData,
                },
            },
            l1_pk: ECDSAPublicKeyTarget(AffinePointTarget {
                x: NonNativeTarget {
                    value: BigUintTarget {
                        limbs: vec![builder.zero_u32(); BIG_U256_LIMBS],
                    },
                    _phantom: core::marker::PhantomData,
                },
                y: NonNativeTarget {
                    value: BigUintTarget {
                        limbs: vec![builder.zero_u32(); BIG_U256_LIMBS],
                    },
                    _phantom: core::marker::PhantomData,
                },
            }),
        }
    }

    pub fn conditional_assert_empty(&self, builder: &mut Builder, cond: BoolTarget) {
        builder.conditional_assert_zero(cond, self.from_account_index);
        builder.conditional_assert_zero(cond, self.api_key_index);
        builder.conditional_assert_zero(cond, self.to_account_index);
        builder.conditional_assert_zero(cond, self.from_route_type);
        builder.conditional_assert_zero(cond, self.to_route_type);
        builder.conditional_assert_zero(cond, self.asset_index);
        builder.conditional_assert_zero(cond, self.chain_id);
        builder.conditional_assert_zero(cond, self.nonce);
        builder.conditional_assert_zero_biguint(cond, &self.amount);
        builder.conditional_assert_zero_biguint(cond, &self.fee);
        self.memo.iter().for_each(|&t| {
            builder.conditional_assert_zero(cond, t.0);
        });
        builder.conditional_assert_zero_biguint(cond, &self.l1_address);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.r.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.s.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.x.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.y.value);
    }

    pub fn connect(builder: &mut Builder, a: &Self, b: &Self) {
        builder.connect(a.from_account_index, b.from_account_index);
        builder.connect(a.api_key_index, b.api_key_index);
        builder.connect(a.to_account_index, b.to_account_index);
        builder.connect(a.from_route_type, b.from_route_type);
        builder.connect(a.to_route_type, b.to_route_type);
        builder.connect(a.asset_index, b.asset_index);
        builder.connect(a.chain_id, b.chain_id);
        builder.connect(a.nonce, b.nonce);
        builder.connect_biguint(&a.amount, &b.amount);
        builder.connect_biguint(&a.fee, &b.fee);
        a.memo.iter().zip(b.memo.iter()).for_each(|(a, b)| {
            builder.connect(a.0, b.0);
        });
        builder.connect_biguint(&a.l1_address, &b.l1_address);
        builder.connect_biguint(&a.l1_signature.r.value, &b.l1_signature.r.value);
        builder.connect_biguint(&a.l1_signature.s.value, &b.l1_signature.s.value);
        builder.connect_biguint(&a.l1_pk.0.x.value, &b.l1_pk.0.x.value);
        builder.connect_biguint(&a.l1_pk.0.y.value, &b.l1_pk.0.y.value);
    }

    pub fn get_transfer_l1_signature_msg_hash(
        &self,
        builder: &mut Builder,
    ) -> NonNativeTarget<Secp256K1Scalar> {
        let zero_hex_byte = builder.constant_u8(48);
        let x_hex_byte = builder.constant_u8(120);

        let from_account_index_hex = split_le_base16(builder, self.from_account_index, 32);
        let api_key_index_hex = split_le_base16(builder, self.api_key_index, 32);
        let to_account_index_hex = split_le_base16(builder, self.to_account_index, 32);
        let from_route_type_hex = split_le_base16(builder, self.from_route_type, 32);
        let to_route_type_hex = split_le_base16(builder, self.to_route_type, 32);
        let asset_index_hex = split_le_base16(builder, self.asset_index, 32);
        let chain_id_hex = split_le_base16(builder, self.chain_id, 32);
        let nonce_hex = split_le_base16(builder, self.nonce, 32);
        let amount_target = builder.biguint_to_target_unsafe(&self.amount);
        let amount_hex = split_le_base16(builder, amount_target, 32);
        let fee_target = builder.biguint_to_target_unsafe(&self.fee);
        let fee_hex = split_le_base16(builder, fee_target, 32);

        let (
            from_account_index_bytes,
            api_key_index_bytes,
            to_account_index_bytes,
            from_route_type_bytes,
            to_route_type_bytes,
            asset_index_bytes,
            chain_id_bytes,
            nonce_bytes,
            amount_bytes,
            fee_bytes,
        ) = [
            &from_account_index_hex,
            &api_key_index_hex,
            &to_account_index_hex,
            &from_route_type_hex,
            &to_route_type_hex,
            &asset_index_hex,
            &chain_id_hex,
            &nonce_hex,
            &amount_hex,
            &fee_hex,
        ]
        .iter_mut()
        .map(|hex| {
            let mut bytes = bytes_to_hex(builder, hex);
            bytes.reverse(); // Make big-endian
            bytes.insert(0, x_hex_byte);
            bytes.insert(0, zero_hex_byte);
            bytes
        })
        .collect_tuple()
        .unwrap();

        let memo_hex = self
            .memo
            .iter()
            .flat_map(|&t| {
                split_le_base16(builder, t.0, 4)
                    .chunks(2)
                    .flat_map(|chunk| [chunk[1], chunk[0]])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let memo_bytes = bytes_to_hex(builder, &memo_hex);

        // Treat elements of TRANSFER_L1_SIGNATURE_TEMPLATE_BITS as constants
        let l1_signature_body_bytes: [U8Target; TRANSFER_L1_SIGNATURE_TEMPLATE_BYTE_LEN] = [
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[0]), // b"\x19Ethereum Signed Message:\n386Transfer\n"
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[1]), // b"\nnonce: "
            nonce_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[2]), // b"\nfrom: "
            from_account_index_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[3]), // b" (route "
            from_route_type_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[4]), // b")\napi key: "
            api_key_index_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[5]), // b"\nto: "
            to_account_index_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[6]), // b" (route "
            to_route_type_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[7]), // b")\nasset: "
            asset_index_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[8]), // b"\namount: "
            amount_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[9]), // b"\nfee: "
            fee_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[10]), // b"\nchainId: "
            chain_id_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[11]), // b"\nmemo hash: "
            memo_bytes,
            builder.constant_u8s(&TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES[12]), // b"\nOnly sign this message for a trusted client!"
        ]
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<U8Target>>()
        .try_into()
        .unwrap();

        builder.keccak256_circuit_to_nonnative(l1_signature_body_bytes.to_vec())
    }
}

const TRANSFER_L1_SIGNATURE_TEMPLATE_BYTE_LEN: usize = 415;

lazy_static! {
    static ref TRANSFER_L1_SIGNATURE_TEMPLATE_BYTES: Vec<Vec<u8>> = [
        // 26 - "\x19Ethereum Signed Message:\n"
        // 3 - "%d" (body len)
        // 9 - "Transfer\n"
        b"\x19Ethereum Signed Message:\n386Transfer\n".to_vec(), // 38 bytes
        b"\nnonce: ".to_vec(), // 8 bytes
        // nonceHex -> 18 bytes
        b"\nfrom: ".to_vec(), // 7 bytes
        // accountIndexHex -> 18 bytes
        b" (route ".to_vec(), // 8 bytes
        // fromRouteTypeHex -> 18 bytes
        b")\napi key: ".to_vec(), // 11 bytes
        // apiKeyIndexHex -> 18 bytes
        b"\nto: ".to_vec(), // 5 bytes
        // accountIndexHex -> 18 bytes
        b" (route ".to_vec(), // 8 bytes
        // toRouteTypeHex -> 18 bytes
        b")\nasset: ".to_vec(), // 9 bytes
        // assetIndexHex -> 18 bytes
        b"\namount: ".to_vec(), // 9 bytes
        // accountIndexHex -> 18 bytes
        b"\nfee: ".to_vec(), // 6 bytes
        // feeHex -> 18 bytes
        b"\nchainId: ".to_vec(), // 10 bytes
        // chainIdHex -> 18 bytes
        b"\nmemo: ".to_vec(), // 7 bytes
        // memoHex -> 64 bytes
        b"\nOnly sign this message for a trusted client!".to_vec(), // 45 bytes
    ].to_vec();
}

#[cfg(test)]
mod tests {
    use plonky2::field::secp256k1_base::Secp256K1Base;
    use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
    use plonky2::field::types::{Field, Field64};
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use serde::Deserialize;

    use super::*;
    use crate::ecdsa::curve::curve_types::AffinePoint;
    use crate::ecdsa::gadgets::ecdsa::{
        ECDSAPublicKeyTargetWitness, ECDSASignatureTargetWitness, conditional_verify_ecdsa_sig,
    };
    use crate::transactions::l2_transfer::*;
    use crate::types::config::{Builder, C, CIRCUIT_CONFIG, F};

    #[derive(Deserialize)]
    pub struct Sig {
        pub l1_sig: Vec<u8>,
        pub l1_pub_key: Vec<u8>,
        pub from_account_index: i64,
        pub api_key_index: u8,
        pub to_account_index: i64,
        pub from_route_type: u8,
        pub to_route_type: u8,
        pub asset_index: i16,
        pub nonce: i64,
        pub amount: i64,
        pub fee: i64,
        pub memo: [u8; TRANSFER_MEMO_BYTES],
    }

    #[test]
    fn test_transfer_l1_signature_verification() {
        // let _ = env_logger::try_init_from_env(
        //     env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"),
        // );

        let sig = Sig {
            from_account_index: 7,
            api_key_index: 0,
            to_account_index: 8,
            from_route_type: 1,
            to_route_type: 1,
            asset_index: 2,
            nonce: 1,
            amount: 20000,
            fee: 0,
            memo: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            l1_pub_key: vec![
                32, 161, 161, 138, 23, 11, 189, 111, 1, 21, 51, 76, 153, 18, 8, 131, 122, 79, 35,
                223, 116, 214, 133, 178, 11, 48, 129, 31, 213, 212, 190, 30, 70, 221, 184, 238, 8,
                90, 64, 193, 223, 130, 177, 186, 0, 153, 5, 105, 168, 29, 21, 93, 106, 146, 36,
                127, 179, 131, 2, 130, 188, 238, 141, 92,
            ],
            l1_sig: vec![
                249, 24, 227, 193, 101, 114, 239, 111, 30, 135, 250, 213, 59, 231, 134, 1, 97, 2,
                216, 54, 26, 101, 215, 137, 206, 69, 77, 6, 126, 182, 191, 104, 52, 170, 147, 78,
                159, 128, 201, 254, 95, 63, 237, 8, 127, 86, 211, 63, 157, 23, 202, 209, 149, 120,
                18, 114, 126, 210, 218, 235, 161, 145, 117, 239,
            ],
        };

        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let tx_target = L2TransferTxTarget::new(&mut builder);
        let tx_nonce_target = builder.add_virtual_target();

        let msg = TransferMessageTarget {
            from_account_index: tx_target.from_account_index,
            api_key_index: tx_target.api_key_index,
            to_account_index: tx_target.to_account_index,
            from_route_type: tx_target.from_route_type,
            to_route_type: tx_target.to_route_type,
            asset_index: tx_target.asset_index,
            chain_id: builder.constant_u64(300),
            nonce: tx_nonce_target,
            amount: tx_target.amount.clone(),
            fee: tx_target.usdc_fee.clone(),
            memo: tx_target.memo,
            ..TransferMessageTarget::default()
        };
        let hashed_msg = msg.get_transfer_l1_signature_msg_hash(&mut builder);

        let pk_target = builder.add_virtual_ecdsa_public_key();
        let sig_target = builder.add_virtual_ecdsa_target();

        let _true = builder._true();
        conditional_verify_ecdsa_sig(&mut builder, _true, &hashed_msg, &sig_target, &pk_target);

        let data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_l2_transfer_tx_target(
            &tx_target,
            &L2TransferTx {
                from_account_index: sig.from_account_index,
                api_key_index: sig.api_key_index,
                to_account_index: sig.to_account_index,
                from_route_type: sig.from_route_type,
                to_route_type: sig.to_route_type,
                asset_index: sig.asset_index,
                amount: BigUint::from(sig.amount as u64),
                usdc_fee: BigUint::from(sig.fee as u64),
                memo: sig.memo,
            },
        )
        .unwrap();
        pw.set_target(tx_nonce_target, F::from_canonical_i64(sig.nonce))
            .unwrap();
        pw.set_ecdsa_public_key_target(
            &pk_target,
            &ECDSAPublicKey::<Secp256K1>(AffinePoint::<Secp256K1> {
                x: Secp256K1Base::from_noncanonical_biguint(BigUint::from_bytes_be(
                    &sig.l1_pub_key[0..32],
                )),
                y: Secp256K1Base::from_noncanonical_biguint(BigUint::from_bytes_be(
                    &sig.l1_pub_key[32..64],
                )),
                zero: false,
            }),
        )
        .unwrap();
        pw.set_ecdsa_signature_target(
            &sig_target,
            &ECDSASignature::<Secp256K1> {
                r: Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_be(
                    &sig.l1_sig[0..32],
                )),
                s: Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_be(
                    &sig.l1_sig[32..64],
                )),
            },
        )
        .unwrap();

        data.verify(data.prove(pw).unwrap()).unwrap();
    }
}
