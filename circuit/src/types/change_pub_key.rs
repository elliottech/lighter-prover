// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use num::BigUint;
use plonky2::field::extension::Extendable;
use plonky2::field::extension::quintic::QuinticExtension;
use plonky2::field::secp256k1_base::Secp256K1Base;
use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};

use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::bigint::comparison::CircuitBuilderBiguintSubtractiveComparison;
use crate::bool_utils::CircuitBuilderBoolUtils;
use crate::ecdsa::curve::curve_types::AffinePoint;
use crate::ecdsa::curve::ecdsa::{ECDSAPublicKey, ECDSASignature};
use crate::ecdsa::curve::secp256k1::Secp256K1;
use crate::ecdsa::gadgets::curve::AffinePointTarget;
use crate::ecdsa::gadgets::ecdsa::{
    CircuitBuilderECDSAPublicKey, CircuitBuilderECDSASignature, ECDSAPublicKeyTarget,
    ECDSASignatureTarget,
};
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::keccak::keccak::CircuitBuilderKeccak;
use crate::nonnative::NonNativeTarget;
use crate::types::api_key::{API_KEY_TYPE_ECDSA, API_KEY_TYPE_SCHNORR};
use crate::types::config::{BIG_U160_LIMBS, BIG_U256_LIMBS, Builder};
use crate::types::constants::TIMESTAMP_BITS;
use crate::types::ecdsa_pub_key::{ECDSA_PUB_KEY_X_LIMBS, EcdsaPubKey, EcdsaPubKeyTarget};
use crate::uint::u8::{CircuitBuilderU8, U8Target};
use crate::uint::u32::gadgets::arithmetic_u32::CircuitBuilderU32;
use crate::utils::{CircuitBuilderUtils, bytes_to_hex, split_le_base16};

#[derive(Debug, Clone)]
pub struct ChangePubKeyMessage<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub account_index: i64,
    pub api_key_index: u8,
    pub nonce: i64,
    pub pub_key: QuinticExtension<F>,
    /// `API_KEY_TYPE_SCHNORR`: `pub_key` is signed; `API_KEY_TYPE_ECDSA`: `ecdsa_pub_key` is.
    pub key_type: u8,
    pub ecdsa_pub_key: EcdsaPubKey,
    pub l1_address: BigUint,
    pub l1_signature: ECDSASignature<Secp256K1>,
    pub l1_pk: ECDSAPublicKey<Secp256K1>,
    /// Tx expiry and chain id, part of the signed text only for `API_KEY_TYPE_ECDSA`.
    pub expired_at: i64,
    pub chain_id: i64,
    pub is_enabled: bool,
}

impl<F: Field + Extendable<5> + RichField + Default> Default for ChangePubKeyMessage<F> {
    fn default() -> Self {
        Self {
            account_index: 0,
            api_key_index: 0,
            nonce: 0,
            pub_key: QuinticExtension::<F>::ZERO,
            key_type: API_KEY_TYPE_SCHNORR,
            ecdsa_pub_key: EcdsaPubKey::zero(),
            l1_address: BigUint::default(),
            l1_signature: ECDSASignature::<Secp256K1>::default(),
            l1_pk: ECDSAPublicKey::<Secp256K1>::default(),
            expired_at: 0,
            chain_id: 0,
            is_enabled: false,
        }
    }
}

// [0] account index, [1] api key index, [2] nonce, [3..8] Schnorr pk, [8..13] l1 address,
// [13..21] r, [21..29] s, [29..37] l1 pk x, [37..45] l1 pk y, [45] key type,
// [46..54] ECDSA pk x limbs, [54] ECDSA pk y parity, [55] expired at, [56] chain id,
// [57] is enabled
pub const CHANGE_PK_PUBLIC_INPUTS_LEN: usize = 46 + ECDSA_PUB_KEY_X_LIMBS + 4;

impl<F: Field + Extendable<5> + RichField> ChangePubKeyMessage<F> {
    pub fn from_public_inputs(public_inputs: &[F]) -> Self {
        let account_index = public_inputs[0].to_canonical_u64() as i64;
        let api_key_index = public_inputs[1].to_canonical_u64() as u8;
        let nonce = public_inputs[2].to_canonical_u64() as i64;
        let pub_key = QuinticExtension(core::array::from_fn(|index| public_inputs[3 + index])); // [3, 4, 5, 6, 7]
        // Convert u32 limbs to BigUint
        let mut l1_address = BigUint::ZERO;
        for i in 0..5 {
            l1_address += BigUint::from(public_inputs[8 + i].to_canonical_u64()) << (i * 32);
        } // [8, 9, 10, 11, 12]
        let mut r = BigUint::ZERO;
        for i in 0..8 {
            r += BigUint::from(public_inputs[13 + i].to_canonical_u64()) << (i * 32);
        } // [13, 14, 15, 16, 17, 18, 19, 20]
        let mut s = BigUint::ZERO;
        for i in 0..8 {
            s += BigUint::from(public_inputs[21 + i].to_canonical_u64()) << (i * 32);
        } // [21, 22, 23, 24, 25, 26, 27, 28]
        let l1_signature = ECDSASignature {
            r: Secp256K1Scalar::from_noncanonical_biguint(r),
            s: Secp256K1Scalar::from_noncanonical_biguint(s),
        };
        let mut x = BigUint::ZERO;
        for i in 0..8 {
            x += BigUint::from(public_inputs[29 + i].to_canonical_u64()) << (i * 32);
        } // [29, 30, 31, 32, 33, 34, 35, 36]
        let mut y = BigUint::ZERO;
        for i in 0..8 {
            y += BigUint::from(public_inputs[37 + i].to_canonical_u64()) << (i * 32);
        } // [37, 38, 39, 40, 41, 42, 43, 44]
        let l1_pk = ECDSAPublicKey(AffinePoint {
            x: Secp256K1Base::from_noncanonical_biguint(x),
            y: Secp256K1Base::from_noncanonical_biguint(y),
            zero: false,
        });
        let key_type = public_inputs[45].to_canonical_u64() as u8; // [45]
        let mut ecdsa_x = BigUint::ZERO;
        for i in 0..ECDSA_PUB_KEY_X_LIMBS {
            ecdsa_x += BigUint::from(public_inputs[46 + i].to_canonical_u64()) << (i * 32);
        } // [46, 47, 48, 49, 50, 51, 52, 53]
        let ecdsa_pub_key = EcdsaPubKey {
            x: ecdsa_x,
            y_odd: public_inputs[54].is_nonzero(), // [54]
        };
        let expired_at = public_inputs[55].to_canonical_u64() as i64; // [55]
        let chain_id = public_inputs[56].to_canonical_u64() as i64; // [56]
        let is_enabled = public_inputs[57].is_nonzero(); // [57]

        Self {
            account_index,
            api_key_index,
            nonce,
            pub_key,
            key_type,
            ecdsa_pub_key,
            l1_address,
            l1_signature,
            l1_pk,
            expired_at,
            chain_id,
            is_enabled,
        }
    }
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChangePubKeyMessageTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub nonce: Target,
    pub pub_key: QuinticExtensionTarget,
    pub key_type: Target,
    pub ecdsa_pub_key: EcdsaPubKeyTarget,
    pub l1_address: BigUintTarget,
    pub l1_signature: ECDSASignatureTarget<Secp256K1>,
    pub l1_pk: ECDSAPublicKeyTarget<Secp256K1>,
    pub expired_at: Target,
    pub chain_id: Target,
    pub is_enabled: BoolTarget,
}

impl ChangePubKeyMessageTarget {
    pub fn from_public_inputs(public_inputs: &[Target]) -> Self {
        let account_index = public_inputs[0];
        let api_key_index = public_inputs[1];
        let nonce = public_inputs[2];
        let pub_key =
            QuinticExtensionTarget(core::array::from_fn(|index| public_inputs[3 + index])); // [3, 4, 5, 6, 7]
        let l1_address = BigUintTarget::from(&public_inputs[8..13]); // [8, 9, 10, 11, 12]
        let l1_signature = ECDSASignatureTarget {
            r: NonNativeTarget {
                value: BigUintTarget::from(&public_inputs[13..21]),
                _phantom: std::marker::PhantomData,
            },
            s: NonNativeTarget {
                value: BigUintTarget::from(&public_inputs[21..29]),
                _phantom: std::marker::PhantomData,
            },
        }; // [13, 14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27, 28]
        let l1_pk = ECDSAPublicKeyTarget(AffinePointTarget {
            x: NonNativeTarget {
                value: BigUintTarget::from(&public_inputs[29..37]),
                _phantom: std::marker::PhantomData,
            }, // [29, 30, 31, 32, 33, 34, 35, 36]
            y: NonNativeTarget {
                value: BigUintTarget::from(&public_inputs[37..45]),
                _phantom: std::marker::PhantomData,
            }, // [37, 38, 39, 40, 41, 42, 43, 44]
        }); // [29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44]
        let key_type = public_inputs[45]; // [45]
        let ecdsa_pub_key = EcdsaPubKeyTarget {
            x: BigUintTarget::from(&public_inputs[46..54]), // [46, 47, 48, 49, 50, 51, 52, 53]
            y_odd: BoolTarget::new_unsafe(public_inputs[54]), // [54]
        };
        let expired_at = public_inputs[55]; // [55]
        let chain_id = public_inputs[56]; // [56]
        let is_enabled = BoolTarget::new_unsafe(public_inputs[57]); // [57]

        Self {
            account_index,
            api_key_index,
            nonce,
            pub_key,
            key_type,
            ecdsa_pub_key,
            l1_address,
            l1_signature,
            l1_pk,
            expired_at,
            chain_id,
            is_enabled,
        }
    }

    pub fn register_public_input(&self, builder: &mut Builder) {
        builder.register_public_input(self.account_index);
        builder.register_public_input(self.api_key_index);
        builder.register_public_input(self.nonce);
        builder.register_quintic_ext_public_input(self.pub_key);
        builder.register_public_input_biguint(&self.l1_address);
        builder.register_public_input_biguint(&self.l1_signature.r.value);
        builder.register_public_input_biguint(&self.l1_signature.s.value);
        builder.register_public_input_biguint(&self.l1_pk.0.x.value);
        builder.register_public_input_biguint(&self.l1_pk.0.y.value);
        builder.register_public_input(self.key_type);
        builder.register_public_input_biguint(&self.ecdsa_pub_key.x);
        builder.register_public_input(self.ecdsa_pub_key.y_odd.target);
        builder.register_public_input(self.expired_at);
        builder.register_public_input(self.chain_id);
        builder.register_public_input(self.is_enabled.target);
    }

    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            nonce: builder.add_virtual_target(),
            pub_key: builder.add_virtual_quintic_ext_target(),
            key_type: builder.add_virtual_target(),
            ecdsa_pub_key: EcdsaPubKeyTarget {
                x: builder.add_virtual_biguint_target_unsafe(ECDSA_PUB_KEY_X_LIMBS), // safe because connected to safe inputs
                y_odd: BoolTarget::new_unsafe(builder.add_virtual_target()), // safe because connected to safe inputs
            },
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
            expired_at: builder.add_virtual_target(),
            chain_id: builder.add_virtual_target(),
            is_enabled: BoolTarget::new_unsafe(builder.add_virtual_target()), // safe because connected to safe inputs
        }
    }

    pub fn new_public(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_public_input(),
            api_key_index: builder.add_virtual_public_input(),
            nonce: builder.add_virtual_public_input(),
            pub_key: builder.add_virtual_public_quintic_ext_target(),
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
            // Public-input order must match `register_public_input` / `from_public_inputs`:
            // key type and ECDSA key come after the L1 pk ([45..55]).
            key_type: builder.add_virtual_public_input(),
            ecdsa_pub_key: EcdsaPubKeyTarget {
                x: builder.add_virtual_biguint_public_input_unsafe(ECDSA_PUB_KEY_X_LIMBS), // Safe because it is connected to public witness from constrained circuit
                y_odd: BoolTarget::new_unsafe(builder.add_virtual_public_input()), // Safe because it is connected to public witness from constrained circuit
            },
            expired_at: builder.add_virtual_public_input(),
            chain_id: builder.add_virtual_public_input(),
            is_enabled: BoolTarget::new_unsafe(builder.add_virtual_public_input()), // Safe because it is connected to public witness from constrained circuit
        }
    }

    pub fn select(builder: &mut Builder, flag: BoolTarget, a: &Self, b: &Self) -> Self {
        Self {
            account_index: builder.select(flag, a.account_index, b.account_index),
            api_key_index: builder.select(flag, a.api_key_index, b.api_key_index),
            nonce: builder.select(flag, a.nonce, b.nonce),
            pub_key: builder.select_quintic_ext(flag, a.pub_key, b.pub_key),
            key_type: builder.select(flag, a.key_type, b.key_type),
            ecdsa_pub_key: EcdsaPubKeyTarget::select(
                builder,
                flag,
                &a.ecdsa_pub_key,
                &b.ecdsa_pub_key,
            ),
            l1_address: builder.select_biguint(flag, &a.l1_address, &b.l1_address),
            l1_signature: builder.select_ecdsa_signature(flag, &a.l1_signature, &b.l1_signature),
            l1_pk: builder.select_ecdsa_public_key(flag, &a.l1_pk, &b.l1_pk),
            expired_at: builder.select(flag, a.expired_at, b.expired_at),
            chain_id: builder.select(flag, a.chain_id, b.chain_id),
            is_enabled: builder.select_bool(flag, a.is_enabled, b.is_enabled),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.zero(),
            api_key_index: builder.zero(),
            nonce: builder.zero(),
            pub_key: builder.zero_quintic_ext(),
            key_type: builder.zero(),
            ecdsa_pub_key: EcdsaPubKeyTarget::zero(builder),
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
            expired_at: builder.zero(),
            chain_id: builder.zero(),
            is_enabled: builder._false(),
        }
    }

    pub fn conditional_assert_empty(&self, builder: &mut Builder, cond: BoolTarget) {
        builder.conditional_assert_zero(cond, self.account_index);
        builder.conditional_assert_zero(cond, self.api_key_index);
        builder.conditional_assert_zero(cond, self.nonce);
        self.pub_key.0.iter().for_each(|&t| {
            builder.conditional_assert_zero(cond, t);
        });
        builder.conditional_assert_zero(cond, self.key_type);
        builder.conditional_assert_zero_biguint(cond, &self.ecdsa_pub_key.x);
        builder.conditional_assert_zero(cond, self.ecdsa_pub_key.y_odd.target);
        builder.conditional_assert_zero_biguint(cond, &self.l1_address);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.r.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.s.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.x.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.y.value);
        builder.conditional_assert_zero(cond, self.expired_at);
        builder.conditional_assert_zero(cond, self.chain_id);
        builder.conditional_assert_zero(cond, self.is_enabled.target);
    }
    /// An enabled message must come from a non-treasury account with a set L1 address and L1 public key.
    pub fn conditional_assert_signer_not_empty(&self, builder: &mut Builder) {
        builder.conditional_assert_not_zero(self.is_enabled, self.account_index);
        let is_l1_address_zero = builder.is_zero_biguint(&self.l1_address);
        builder.conditional_assert_false(self.is_enabled, is_l1_address_zero);
        let is_l1_pk_x_zero = builder.is_zero_biguint(&self.l1_pk.0.x.value);
        let is_l1_pk_y_zero = builder.is_zero_biguint(&self.l1_pk.0.y.value);
        let is_l1_pk_zero = builder.and(is_l1_pk_x_zero, is_l1_pk_y_zero);
        builder.conditional_assert_false(self.is_enabled, is_l1_pk_zero);
    }

    pub fn connect(builder: &mut Builder, a: &Self, b: &Self) {
        builder.connect(a.account_index, b.account_index);
        builder.connect(a.api_key_index, b.api_key_index);
        builder.connect(a.nonce, b.nonce);
        builder.connect_quintic_ext(a.pub_key, b.pub_key);
        builder.connect(a.key_type, b.key_type);
        EcdsaPubKeyTarget::connect(builder, &a.ecdsa_pub_key, &b.ecdsa_pub_key);
        builder.connect_biguint(&a.l1_address, &b.l1_address);
        builder.connect_biguint(&a.l1_signature.r.value, &b.l1_signature.r.value);
        builder.connect_biguint(&a.l1_signature.s.value, &b.l1_signature.s.value);
        builder.connect_biguint(&a.l1_pk.0.x.value, &b.l1_pk.0.x.value);
        builder.connect_biguint(&a.l1_pk.0.y.value, &b.l1_pk.0.y.value);
        builder.connect(a.expired_at, b.expired_at);
        builder.connect(a.chain_id, b.chain_id);
        builder.connect(a.is_enabled.target, b.is_enabled.target);
    }

    /// keccak of the personal-sign message the L1 wallet signed. Both schemes fill the same 284 bytes:
    /// Schnorr signs an 80 hex char key and the trusted-client line, ECDSA signs a 66 hex char key,
    /// the tx expiry and the chain id.
    pub fn get_change_pub_key_l1_signature_msg_hash(
        &self,
        builder: &mut Builder,
    ) -> NonNativeTarget<Secp256K1Scalar> {
        let is_ecdsa = builder.is_equal_constant(self.key_type, API_KEY_TYPE_ECDSA as u64);
        let [nonce, account_index, api_key_index, expired_at, chain_id] =
            self.l1_signature_rest_hex_bytes(builder);
        let common_bytes = [
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[1]),
            nonce,
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[2]),
            account_index,
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[3]),
            api_key_index,
        ]
        .concat();
        let schnorr_bytes = [
            self.schnorr_pub_key_hex_bytes(builder),
            common_bytes.clone(),
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[4]),
        ]
        .concat();
        let ecdsa_bytes = [
            self.ecdsa_pub_key_hex_bytes(builder),
            common_bytes,
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[5]),
            expired_at,
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[6]),
            chain_id,
        ]
        .concat();
        assert_eq!(schnorr_bytes.len(), ecdsa_bytes.len());

        let mut l1_signature_body_bytes = builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[0]);
        l1_signature_body_bytes.extend(schnorr_bytes.into_iter().zip(ecdsa_bytes).map(
            |(schnorr_byte, ecdsa_byte)| builder.select_u8(is_ecdsa, ecdsa_byte, schnorr_byte),
        ));
        assert_eq!(
            l1_signature_body_bytes.len(),
            L1_SIGNATURE_TEMPLATE_BYTE_LEN
        );

        builder.keccak256_circuit_to_nonnative(l1_signature_body_bytes)
    }

    /// `"0x" + 16 hex chars` (big-endian, zero padded) of nonce, account index, api key index,
    /// expired at and chain id. Expired at and chain id are split at their own bit size, so that
    /// each has a single hex representation.
    fn l1_signature_rest_hex_bytes(&self, builder: &mut Builder) -> [Vec<U8Target>; 5] {
        let zero_hex_byte = builder.constant_u8(48);
        let x_hex_byte = builder.constant_u8(120);
        [
            (self.nonce, 64),
            (self.account_index, 64),
            (self.api_key_index, 64),
            (self.expired_at, TIMESTAMP_BITS),
            (self.chain_id, 32),
        ]
        .map(|(t, bits)| {
            let hex = split_le_base16(builder, t, bits / 2);
            let mut bytes = bytes_to_hex(builder, &hex);
            bytes.resize(16, zero_hex_byte);
            bytes.reverse(); // Make big-endian
            bytes.insert(0, x_hex_byte);
            bytes.insert(0, zero_hex_byte);
            bytes
        })
    }

    fn schnorr_pub_key_hex_bytes(&self, builder: &mut Builder) -> Vec<U8Target> {
        let pub_key_hex = [
            split_le_base16(builder, self.pub_key.0[0], 32),
            split_le_base16(builder, self.pub_key.0[1], 32),
            split_le_base16(builder, self.pub_key.0[2], 32),
            split_le_base16(builder, self.pub_key.0[3], 32),
            split_le_base16(builder, self.pub_key.0[4], 32),
        ];
        let mut pub_key_bytes = vec![];
        for i in 0..pub_key_hex.len() {
            pub_key_bytes.push(bytes_to_hex(builder, &pub_key_hex[i]));
        }
        // Invert consecutive 2 bytes of pubkey: hex_bytes[0][0], hex_bytes[0][1] = hex_bytes[0][1], hex_bytes[0][0]
        // This is to match the hexified bytes from abi.encodePacked from Solidity
        for i in 0..pub_key_bytes.len() {
            for j in (0..16).step_by(2) {
                pub_key_bytes[i].swap(j, j + 1);
            }
        }
        pub_key_bytes.concat()
    }

    /// Hex of the compressed SEC1 key: `02`/`03` (y parity) followed by x big-endian (64 hex chars).
    fn ecdsa_pub_key_hex_bytes(&self, builder: &mut Builder) -> Vec<U8Target> {
        let zero_hex_byte = builder.constant_u8(48);
        let two_hex_byte = builder.constant_u8(50);
        let three_hex_byte = builder.constant_u8(51);
        let parity_byte = builder.select_u8(self.ecdsa_pub_key.y_odd, three_hex_byte, two_hex_byte);

        assert_eq!(self.ecdsa_pub_key.x.limbs.len(), ECDSA_PUB_KEY_X_LIMBS);
        let mut x_hex = Vec::with_capacity(ECDSA_PUB_KEY_X_LIMBS * 8);
        for limb in self.ecdsa_pub_key.x.limbs.iter().rev() {
            let mut nibbles = builder.split_u32_to_4_bit_limbs_le(*limb);
            nibbles.reverse(); // Make big-endian
            x_hex.extend(nibbles);
        }
        let mut bytes = vec![zero_hex_byte, parity_byte];
        bytes.extend(bytes_to_hex(builder, &x_hex));
        bytes
    }
}

const L1_SIGNATURE_TEMPLATE_BYTE_LEN: usize = 284;

lazy_static! {
    static ref L1_SIGNATURE_TEMPLATE_BYTES: Vec<Vec<u8>> = [ // Filled length -> 284 bytes
        // 26 - "\x19Ethereum Signed Message:\n"
        // 3 - %d (body len)
        b"\x19Ethereum Signed Message:\n255Register Lighter Account\n\npubkey: 0x".to_vec(), // 65 bytes
        // L2 pubkey -> 80 bytes (Schnorr), 66 bytes (ECDSA)
        b"\nnonce: ".to_vec(), // 8 bytes
        // nonceHex -> 18 bytes
        b"\naccount index: ".to_vec(), // 16 bytes
        // accountIndexHex -> 18 bytes
        b"\napi key index: ".to_vec(), // 16 bytes
        // apiKeyIndexHex -> 18 bytes
        // Schnorr tail -> 45 bytes
        b"\nOnly sign this message for a trusted client!".to_vec(), // 45 bytes
        // ECDSA tail -> 14 + 45 bytes
        b"\nexpired at: ".to_vec(), // 13 bytes
        // expiredAtHex -> 18 bytes
        b"\nchainId: ".to_vec(), // 10 bytes
        // chainIdHex -> 18 bytes
    ].to_vec();
}

#[cfg(test)]
mod tests {
    use plonky2::field::extension::quintic::QuinticExtension;
    use plonky2::field::secp256k1_base::Secp256K1Base;
    use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
    use plonky2::field::types::{Field, Field64};
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};

    use super::*;
    use crate::ecdsa::curve::curve_types::AffinePoint;
    use crate::ecdsa::gadgets::ecdsa::{
        ECDSAPublicKeyTargetWitness, ECDSASignatureTargetWitness, verify_ecdsa_sig,
    };
    use crate::transactions::l2_change_pubkey::*;
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    #[ignore]
    #[test]
    fn test_l1_signature_verification() {
        let mut builder = Builder::new(CIRCUIT_CONFIG);

        let tx_target = L2ChangePubKeyTxTarget::new(&mut builder);
        let tx_nonce_target = builder.add_virtual_target();

        let message = ChangePubKeyMessageTarget {
            account_index: tx_target.account_index,
            api_key_index: tx_target.api_key_index,
            nonce: tx_nonce_target,
            pub_key: tx_target.pub_key,
            ..ChangePubKeyMessageTarget::default()
        };

        let hashed_msg = message.get_change_pub_key_l1_signature_msg_hash(&mut builder);

        let pk_target = builder.add_virtual_ecdsa_public_key();
        let sig_target = builder.add_virtual_ecdsa_target();

        verify_ecdsa_sig(&mut builder, &hashed_msg, &sig_target, &pk_target);

        let data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_l2_change_pk_tx_target(
            &tx_target,
            &L2ChangePubKeyTx {
                account_index: 40,
                api_key_index: 0,
                pub_key: QuinticExtension::<F>([
                    F::from_canonical_u64(9276946854624046279),
                    F::from_canonical_u64(17598345013240974892),
                    F::from_canonical_u64(2284116666807947770),
                    F::from_canonical_u64(13833805950497912391),
                    F::from_canonical_u64(17986426947097389460),
                ]),
            },
        )
        .unwrap();
        pw.set_target(tx_nonce_target, F::from_canonical_i64(0))
            .unwrap();
        pw.set_ecdsa_public_key_target(
            &pk_target,
            &ECDSAPublicKey::<Secp256K1>(AffinePoint::<Secp256K1> {
                x: Secp256K1Base::from_noncanonical_biguint(BigUint::from_bytes_be(&[
                    62, 136, 91, 54, 140, 43, 101, 151, 238, 211, 217, 108, 93, 121, 154, 36, 64,
                    71, 97, 205, 82, 197, 214, 221, 97, 143, 164, 223, 230, 208, 192, 148,
                ])),
                y: Secp256K1Base::from_noncanonical_biguint(BigUint::from_bytes_be(&[
                    216, 150, 133, 1, 233, 10, 199, 94, 148, 19, 31, 180, 185, 93, 99, 105, 243,
                    220, 85, 235, 82, 77, 10, 28, 46, 176, 208, 44, 41, 38, 247, 181,
                ])),
                zero: false,
            }),
        )
        .unwrap();
        pw.set_ecdsa_signature_target(
            &sig_target,
            &ECDSASignature::<Secp256K1> {
                r: Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_be(&[
                    43, 121, 90, 116, 185, 239, 235, 0, 101, 92, 44, 184, 93, 143, 205, 39, 153,
                    74, 207, 236, 137, 213, 191, 77, 215, 49, 163, 214, 90, 134, 6, 36,
                ])),
                s: Secp256K1Scalar::from_noncanonical_biguint(BigUint::from_bytes_be(&[
                    115, 158, 165, 167, 69, 86, 241, 122, 50, 133, 138, 64, 69, 182, 251, 91, 10,
                    86, 42, 72, 39, 88, 117, 55, 70, 117, 169, 198, 14, 201, 139, 27,
                ])),
            },
        )
        .unwrap();

        data.verify(data.prove(pw).unwrap()).unwrap();
    }
}
