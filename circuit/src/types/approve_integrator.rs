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
use crate::types::config::{BIG_U160_LIMBS, BIG_U256_LIMBS, Builder};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::{CircuitBuilderUtils, bytes_to_hex, split_le_base16};

#[derive(Debug, Clone, Default)]
pub struct ApproveIntegratorMessage {
    pub account_index: i64,
    pub api_key_index: u8,
    pub integrator_account_index: i64,
    pub max_perps_taker_fee: i64,
    pub max_perps_maker_fee: i64,
    pub max_spot_taker_fee: i64,
    pub max_spot_maker_fee: i64,
    pub approval_expiry: i64,

    pub nonce: i64,
    pub chain_id: i64,

    pub l1_address: BigUint,
    pub l1_signature: ECDSASignature<Secp256K1>,
    pub l1_pk: ECDSAPublicKey<Secp256K1>,
}

pub const APPROVE_INTEGRATOR_PUBLIC_INPUTS_LEN: usize = 47;

impl ApproveIntegratorMessage {
    pub fn from_public_inputs<F: Field + Extendable<5> + RichField>(pis: &[F]) -> Self {
        let account_index = pis[0].to_canonical_u64() as i64;
        let api_key_index = pis[1].to_canonical_u64() as u8;

        let integrator_account_index = pis[2].to_canonical_u64() as i64;
        let max_perps_taker_fee = pis[3].to_canonical_u64() as i64;
        let max_perps_maker_fee = pis[4].to_canonical_u64() as i64;
        let max_spot_taker_fee = pis[5].to_canonical_u64() as i64;
        let max_spot_maker_fee = pis[6].to_canonical_u64() as i64;
        let approval_expiry = pis[7].to_canonical_u64() as i64;

        let nonce = pis[8].to_canonical_u64() as i64;
        let chain_id = pis[9].to_canonical_u64() as i64;

        // Convert u32 limbs to BigUint
        let mut l1_address = BigUint::ZERO;
        for i in 0..5 {
            l1_address += BigUint::from(pis[10 + i].to_canonical_u64()) << (i * 32);
        }
        let mut r = BigUint::ZERO;
        for i in 0..8 {
            r += BigUint::from(pis[15 + i].to_canonical_u64()) << (i * 32);
        }
        let mut s = BigUint::ZERO;
        for i in 0..8 {
            s += BigUint::from(pis[23 + i].to_canonical_u64()) << (i * 32);
        }
        let l1_signature = ECDSASignature {
            r: Secp256K1Scalar::from_noncanonical_biguint(r),
            s: Secp256K1Scalar::from_noncanonical_biguint(s),
        };
        let mut x = BigUint::ZERO;
        for i in 0..8 {
            x += BigUint::from(pis[31 + i].to_canonical_u64()) << (i * 32);
        }
        let mut y = BigUint::ZERO;
        for i in 0..8 {
            y += BigUint::from(pis[39 + i].to_canonical_u64()) << (i * 32);
        }
        let l1_pk = ECDSAPublicKey(AffinePoint {
            x: Secp256K1Base::from_noncanonical_biguint(x),
            y: Secp256K1Base::from_noncanonical_biguint(y),
            zero: false,
        });

        Self {
            account_index,
            api_key_index,
            integrator_account_index,
            max_perps_taker_fee,
            max_perps_maker_fee,
            max_spot_taker_fee,
            max_spot_maker_fee,
            approval_expiry,

            nonce,
            chain_id,

            l1_address,
            l1_signature,
            l1_pk,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ApproveIntegratorMessageTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub integrator_account_index: Target,
    pub max_perps_taker_fee: Target,
    pub max_perps_maker_fee: Target,
    pub max_spot_taker_fee: Target,
    pub max_spot_maker_fee: Target,
    pub approval_expiry: Target,

    pub nonce: Target,
    pub chain_id: Target,

    pub l1_address: BigUintTarget,
    pub l1_signature: ECDSASignatureTarget<Secp256K1>,
    pub l1_pk: ECDSAPublicKeyTarget<Secp256K1>,
}

impl ApproveIntegratorMessageTarget {
    pub fn from_public_inputs(pis: &[Target]) -> Self {
        let account_index = pis[0];
        let api_key_index = pis[1];

        let integrator_account_index = pis[2];
        let max_perps_taker_fee = pis[3];
        let max_perps_maker_fee = pis[4];
        let max_spot_taker_fee = pis[5];
        let max_spot_maker_fee = pis[6];
        let approval_expiry = pis[7];

        let nonce = pis[8];
        let chain_id = pis[9];

        let l1_address = BigUintTarget::from(&pis[10..15]);
        let l1_signature = ECDSASignatureTarget {
            r: NonNativeTarget {
                value: BigUintTarget::from(&pis[15..23]),
                _phantom: std::marker::PhantomData,
            },
            s: NonNativeTarget {
                value: BigUintTarget::from(&pis[23..31]),
                _phantom: std::marker::PhantomData,
            },
        };
        let l1_pk = ECDSAPublicKeyTarget(AffinePointTarget {
            x: NonNativeTarget {
                value: BigUintTarget::from(&pis[31..39]),
                _phantom: std::marker::PhantomData,
            },
            y: NonNativeTarget {
                value: BigUintTarget::from(&pis[39..47]),
                _phantom: std::marker::PhantomData,
            },
        });

        Self {
            account_index,
            api_key_index,
            integrator_account_index,
            max_perps_taker_fee,
            max_perps_maker_fee,
            max_spot_taker_fee,
            max_spot_maker_fee,
            approval_expiry,
            nonce,
            chain_id,
            l1_address,
            l1_signature,
            l1_pk,
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.account_index);
        builder.register_public_input(self.api_key_index);
        builder.register_public_input(self.integrator_account_index);
        builder.register_public_input(self.max_perps_taker_fee);
        builder.register_public_input(self.max_perps_maker_fee);
        builder.register_public_input(self.max_spot_taker_fee);
        builder.register_public_input(self.max_spot_maker_fee);
        builder.register_public_input(self.approval_expiry);
        builder.register_public_input(self.nonce);
        builder.register_public_input(self.chain_id);
        builder.register_public_input_biguint(&self.l1_address);
        builder.register_public_input_biguint(&self.l1_signature.r.value);
        builder.register_public_input_biguint(&self.l1_signature.s.value);
        builder.register_public_input_biguint(&self.l1_pk.0.x.value);
        builder.register_public_input_biguint(&self.l1_pk.0.y.value);
    }

    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            integrator_account_index: builder.add_virtual_target(),
            max_perps_taker_fee: builder.add_virtual_target(),
            max_perps_maker_fee: builder.add_virtual_target(),
            max_spot_taker_fee: builder.add_virtual_target(),
            max_spot_maker_fee: builder.add_virtual_target(),
            approval_expiry: builder.add_virtual_target(),

            nonce: builder.add_virtual_target(),
            chain_id: builder.add_virtual_target(),

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
            account_index: builder.add_virtual_public_input(),
            api_key_index: builder.add_virtual_public_input(),
            integrator_account_index: builder.add_virtual_public_input(),
            max_perps_taker_fee: builder.add_virtual_public_input(),
            max_perps_maker_fee: builder.add_virtual_public_input(),
            max_spot_taker_fee: builder.add_virtual_public_input(),
            max_spot_maker_fee: builder.add_virtual_public_input(),
            approval_expiry: builder.add_virtual_public_input(),

            nonce: builder.add_virtual_public_input(),
            chain_id: builder.add_virtual_public_input(),

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
            account_index: builder.select(flag, a.account_index, b.account_index),
            api_key_index: builder.select(flag, a.api_key_index, b.api_key_index),
            integrator_account_index: builder.select(
                flag,
                a.integrator_account_index,
                b.integrator_account_index,
            ),
            max_perps_taker_fee: builder.select(flag, a.max_perps_taker_fee, b.max_perps_taker_fee),
            max_perps_maker_fee: builder.select(flag, a.max_perps_maker_fee, b.max_perps_maker_fee),
            max_spot_taker_fee: builder.select(flag, a.max_spot_taker_fee, b.max_spot_taker_fee),
            max_spot_maker_fee: builder.select(flag, a.max_spot_maker_fee, b.max_spot_maker_fee),
            approval_expiry: builder.select(flag, a.approval_expiry, b.approval_expiry),

            nonce: builder.select(flag, a.nonce, b.nonce),
            chain_id: builder.select(flag, a.chain_id, b.chain_id),

            l1_address: builder.select_biguint(flag, &a.l1_address, &b.l1_address),
            l1_signature: builder.select_ecdsa_signature(flag, &a.l1_signature, &b.l1_signature),
            l1_pk: builder.select_ecdsa_public_key(flag, &a.l1_pk, &b.l1_pk),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.zero(),
            api_key_index: builder.zero(),
            integrator_account_index: builder.zero(),
            max_perps_taker_fee: builder.zero(),
            max_perps_maker_fee: builder.zero(),
            max_spot_taker_fee: builder.zero(),
            max_spot_maker_fee: builder.zero(),
            approval_expiry: builder.zero(),

            nonce: builder.zero(),
            chain_id: builder.zero(),

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
        builder.conditional_assert_zero(cond, self.account_index);
        builder.conditional_assert_zero(cond, self.api_key_index);
        builder.conditional_assert_zero(cond, self.integrator_account_index);
        builder.conditional_assert_zero(cond, self.max_perps_taker_fee);
        builder.conditional_assert_zero(cond, self.max_perps_maker_fee);
        builder.conditional_assert_zero(cond, self.max_spot_taker_fee);
        builder.conditional_assert_zero(cond, self.max_spot_maker_fee);
        builder.conditional_assert_zero(cond, self.approval_expiry);
        builder.conditional_assert_zero(cond, self.nonce);
        builder.conditional_assert_zero(cond, self.chain_id);
        builder.conditional_assert_zero_biguint(cond, &self.l1_address);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.r.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.s.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.x.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.y.value);
    }

    pub fn connect(builder: &mut Builder, a: &Self, b: &Self) {
        builder.connect(a.account_index, b.account_index);
        builder.connect(a.api_key_index, b.api_key_index);
        builder.connect(a.integrator_account_index, b.integrator_account_index);
        builder.connect(a.max_perps_taker_fee, b.max_perps_taker_fee);
        builder.connect(a.max_perps_maker_fee, b.max_perps_maker_fee);
        builder.connect(a.max_spot_taker_fee, b.max_spot_taker_fee);
        builder.connect(a.max_spot_maker_fee, b.max_spot_maker_fee);
        builder.connect(a.approval_expiry, b.approval_expiry);
        builder.connect(a.nonce, b.nonce);
        builder.connect(a.chain_id, b.chain_id);
        builder.connect_biguint(&a.l1_address, &b.l1_address);
        builder.connect_biguint(&a.l1_signature.r.value, &b.l1_signature.r.value);
        builder.connect_biguint(&a.l1_signature.s.value, &b.l1_signature.s.value);
        builder.connect_biguint(&a.l1_pk.0.x.value, &b.l1_pk.0.x.value);
        builder.connect_biguint(&a.l1_pk.0.y.value, &b.l1_pk.0.y.value);
    }

    pub fn get_approve_integrator_l1_signature_msg_hash(
        &self,
        builder: &mut Builder,
    ) -> NonNativeTarget<Secp256K1Scalar> {
        let zero_hex_byte = builder.constant_u8(48);
        let x_hex_byte = builder.constant_u8(120);

        let account_index_hex = split_le_base16(builder, self.account_index, 32);
        let api_key_index_hex = split_le_base16(builder, self.api_key_index, 32);
        let integrator_account_index_hex =
            split_le_base16(builder, self.integrator_account_index, 32);
        let max_perps_taker_fee_hex = split_le_base16(builder, self.max_perps_taker_fee, 32);
        let max_perps_maker_fee_hex = split_le_base16(builder, self.max_perps_maker_fee, 32);
        let max_spot_taker_fee_hex = split_le_base16(builder, self.max_spot_taker_fee, 32);
        let max_spot_maker_fee_hex = split_le_base16(builder, self.max_spot_maker_fee, 32);
        let approval_expiry_hex = split_le_base16(builder, self.approval_expiry, 32);
        let nonce_hex = split_le_base16(builder, self.nonce, 32);
        let chain_id_hex = split_le_base16(builder, self.chain_id, 32);

        let (
            account_index_bytes,
            api_key_index_bytes,
            integrator_account_index_bytes,
            max_perps_taker_fee_bytes,
            max_perps_maker_fee_bytes,
            max_spot_taker_fee_bytes,
            max_spot_maker_fee_bytes,
            approval_expiry_bytes,
            nonce_bytes,
            chain_id_bytes,
        ) = [
            &account_index_hex,
            &api_key_index_hex,
            &integrator_account_index_hex,
            &max_perps_taker_fee_hex,
            &max_perps_maker_fee_hex,
            &max_spot_taker_fee_hex,
            &max_spot_maker_fee_hex,
            &approval_expiry_hex,
            &nonce_hex,
            &chain_id_hex,
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

        // Treat elements of APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BITS as constants
        let l1_signature_body_bytes: [U8Target; APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTE_LEN] =
            [
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[0]),
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[1]),
                nonce_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[2]),
                account_index_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[3]),
                api_key_index_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[4]),
                integrator_account_index_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[5]),
                max_perps_taker_fee_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[6]),
                max_perps_maker_fee_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[7]),
                max_spot_taker_fee_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[8]),
                max_spot_maker_fee_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[9]),
                approval_expiry_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[10]),
                chain_id_bytes,
                builder.constant_u8s(&APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES[11]),
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

const APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTE_LEN: usize = 454;

lazy_static! {
    static ref APPROVE_INTEGRATOR_L1_SIGNATURE_TEMPLATE_BYTES: Vec<Vec<u8>> = [
        // 26 - "\x19Ethereum Signed Message:\n"
        // 3 - "%d" (body len)
        // 9 - "Approve Integrator\n"
        b"\x19Ethereum Signed Message:\n425Approve Integrator\n".to_vec(), // 38 bytes
        b"\nnonce: ".to_vec(), // 8 bytes
        // nonceHex -> 10 bytes
        b"\naccount index: ".to_vec(), // 16 bytes
        // accountIndexHex -> 10 bytes
        b"\napi key index: ".to_vec(), // 16 bytes
        // apiKeyIndexHex -> 10 bytes
        b"\nintegrator account index: ".to_vec(),
        // integratorAccountIndexHex -> 10 bytes
        b"\nmax perps taker fee: ".to_vec(),
        // maxPerpsTakerFeeHex -> 10 bytes
        b"\nmax perps maker fee: ".to_vec(),
        // maxPerpsMakerFeeHex -> 10 bytes
        b"\nmax spot taker fee: ".to_vec(),
        // maxSpotTakerFeeHex -> 10 bytes
        b"\nmax spot maker fee: ".to_vec(),
        // maxSpotMakerFeeHex -> 10 bytes
        b"\napproval expiry: ".to_vec(),
        // approvalExpiryHex -> 10 bytes
        b"\nchainId: ".to_vec(), // 10 bytes
        // chainIdHex -> 10 bytes
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
    use crate::transactions::l2_approve_integrator::*;
    use crate::types::config::{Builder, C, CIRCUIT_CONFIG, F};

    #[derive(Deserialize)]
    pub struct Sig {
        pub l1_sig: Vec<u8>,
        pub l1_pub_key: Vec<u8>,
        pub account_index: i64,
        pub api_key_index: u8,
        pub integrator_account_index: i64,
        pub max_perps_taker_fee: i64,
        pub max_perps_maker_fee: i64,
        pub max_spot_taker_fee: i64,
        pub max_spot_maker_fee: i64,
        pub approval_expiry: i64,
        pub nonce: i64,
    }

    #[test]
    fn test_approve_integrator_l1_signature_verification() {
        // let _ = env_logger::try_init_from_env(
        //     env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "debug"),
        // );

        let sig = Sig {
            account_index: 13,
            api_key_index: 0,
            integrator_account_index: 14,
            max_perps_taker_fee: 100,
            max_perps_maker_fee: 200,
            max_spot_taker_fee: 300,
            max_spot_maker_fee: 400,
            approval_expiry: 0,
            nonce: 1,

            l1_pub_key: vec![
                101, 209, 184, 66, 12, 150, 95, 157, 94, 178, 124, 172, 148, 243, 1, 181, 104, 237,
                229, 158, 60, 49, 125, 110, 109, 17, 55, 21, 115, 95, 129, 5, 163, 196, 1, 76, 188,
                178, 175, 19, 239, 255, 68, 115, 47, 224, 133, 243, 37, 195, 203, 148, 139, 11,
                237, 85, 6, 45, 17, 185, 177, 252, 82, 70,
            ],
            l1_sig: vec![
                141, 83, 158, 226, 55, 65, 5, 66, 99, 103, 208, 105, 242, 106, 105, 228, 194, 85,
                218, 223, 11, 45, 51, 250, 112, 203, 124, 128, 213, 162, 138, 150, 112, 216, 37,
                176, 201, 163, 163, 233, 233, 135, 242, 191, 236, 98, 146, 113, 60, 206, 220, 40,
                162, 181, 165, 229, 210, 127, 192, 201, 241, 254, 67, 112,
            ],
        };

        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let tx_target = L2ApproveIntegratorTxTarget::new(&mut builder);
        let tx_nonce_target = builder.add_virtual_target();

        let msg = ApproveIntegratorMessageTarget {
            account_index: tx_target.account_index,
            api_key_index: tx_target.api_key_index,
            integrator_account_index: tx_target.integrator_account_index,
            max_perps_taker_fee: tx_target.max_perps_taker_fee,
            max_perps_maker_fee: tx_target.max_perps_maker_fee,
            max_spot_taker_fee: tx_target.max_spot_taker_fee,
            max_spot_maker_fee: tx_target.max_spot_maker_fee,
            approval_expiry: tx_target.approval_expiry,
            nonce: tx_nonce_target,
            chain_id: builder.constant_u64(300),

            ..ApproveIntegratorMessageTarget::default()
        };
        let hashed_msg = msg.get_approve_integrator_l1_signature_msg_hash(&mut builder);

        let pk_target = builder.add_virtual_ecdsa_public_key();
        let sig_target = builder.add_virtual_ecdsa_target();

        let _true = builder._true();
        conditional_verify_ecdsa_sig(&mut builder, _true, &hashed_msg, &sig_target, &pk_target);

        let data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_l2_approve_integrator_tx_target(
            &tx_target,
            &L2ApproveIntegratorTx {
                account_index: sig.account_index,
                api_key_index: sig.api_key_index,
                integrator_account_index: sig.integrator_account_index,
                max_perps_taker_fee: sig.max_perps_taker_fee as u32,
                max_perps_maker_fee: sig.max_perps_maker_fee as u32,
                max_spot_taker_fee: sig.max_spot_taker_fee as u32,
                max_spot_maker_fee: sig.max_spot_maker_fee as u32,
                approval_expiry: sig.approval_expiry,
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
