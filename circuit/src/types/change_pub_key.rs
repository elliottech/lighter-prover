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
use crate::types::config::{BIG_U160_LIMBS, BIG_U256_LIMBS, Builder};
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
    pub l1_address: BigUint,
    pub l1_signature: ECDSASignature<Secp256K1>,
    pub l1_pk: ECDSAPublicKey<Secp256K1>,
}

impl<F: Field + Extendable<5> + RichField + Default> Default for ChangePubKeyMessage<F> {
    fn default() -> Self {
        Self {
            account_index: 0,
            api_key_index: 0,
            nonce: 0,
            pub_key: QuinticExtension::<F>::ZERO,
            l1_address: BigUint::default(),
            l1_signature: ECDSASignature::<Secp256K1>::default(),
            l1_pk: ECDSAPublicKey::<Secp256K1>::default(),
        }
    }
}

pub const CHANGE_PK_PUBLIC_INPUTS_LEN: usize = 45;

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

        Self {
            account_index,
            api_key_index,
            nonce,
            pub_key,
            l1_address,
            l1_signature,
            l1_pk,
        }
    }
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChangePubKeyMessageTarget {
    pub account_index: Target,
    pub api_key_index: Target,
    pub nonce: Target,
    pub pub_key: QuinticExtensionTarget,
    pub l1_address: BigUintTarget,
    pub l1_signature: ECDSASignatureTarget<Secp256K1>,
    pub l1_pk: ECDSAPublicKeyTarget<Secp256K1>,
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

        Self {
            account_index,
            api_key_index,
            nonce,
            pub_key,
            l1_address,
            l1_signature,
            l1_pk,
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
    }

    pub fn new(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.add_virtual_target(),
            api_key_index: builder.add_virtual_target(),
            nonce: builder.add_virtual_target(),
            pub_key: builder.add_virtual_quintic_ext_target(),
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
        }
    }

    pub fn select(builder: &mut Builder, flag: BoolTarget, a: &Self, b: &Self) -> Self {
        Self {
            account_index: builder.select(flag, a.account_index, b.account_index),
            api_key_index: builder.select(flag, a.api_key_index, b.api_key_index),
            nonce: builder.select(flag, a.nonce, b.nonce),
            pub_key: builder.select_quintic_ext(flag, a.pub_key, b.pub_key),
            l1_address: builder.select_biguint(flag, &a.l1_address, &b.l1_address),
            l1_signature: builder.select_ecdsa_signature(flag, &a.l1_signature, &b.l1_signature),
            l1_pk: builder.select_ecdsa_public_key(flag, &a.l1_pk, &b.l1_pk),
        }
    }

    pub fn empty(builder: &mut Builder) -> Self {
        Self {
            account_index: builder.zero(),
            api_key_index: builder.zero(),
            nonce: builder.zero(),
            pub_key: builder.zero_quintic_ext(),
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
        builder.conditional_assert_zero(cond, self.nonce);
        self.pub_key.0.iter().for_each(|&t| {
            builder.conditional_assert_zero(cond, t);
        });
        builder.conditional_assert_zero_biguint(cond, &self.l1_address);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.r.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_signature.s.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.x.value);
        builder.conditional_assert_zero_biguint(cond, &self.l1_pk.0.y.value);
    }

    pub fn connect(builder: &mut Builder, a: &Self, b: &Self) {
        builder.connect(a.account_index, b.account_index);
        builder.connect(a.api_key_index, b.api_key_index);
        builder.connect(a.nonce, b.nonce);
        builder.connect_quintic_ext(a.pub_key, b.pub_key);
        builder.connect_biguint(&a.l1_address, &b.l1_address);
        builder.connect_biguint(&a.l1_signature.r.value, &b.l1_signature.r.value);
        builder.connect_biguint(&a.l1_signature.s.value, &b.l1_signature.s.value);
        builder.connect_biguint(&a.l1_pk.0.x.value, &b.l1_pk.0.x.value);
        builder.connect_biguint(&a.l1_pk.0.y.value, &b.l1_pk.0.y.value);
    }

    pub fn get_change_pub_key_l1_signature_msg_hash(
        &self,
        builder: &mut Builder,
    ) -> NonNativeTarget<Secp256K1Scalar> {
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
        let pub_key_bytes = pub_key_bytes.concat();

        let rest_hex = [
            split_le_base16(builder, self.nonce, 32),
            split_le_base16(builder, self.account_index, 32),
            split_le_base16(builder, self.api_key_index, 32),
        ];
        let mut rest_bytes = vec![];
        for i in 0..rest_hex.len() {
            rest_bytes.push(bytes_to_hex(builder, &rest_hex[i]));
        }

        // Append "0x" to nonce, account index and api key index
        let zero_hex_byte = builder.constant_u8(48);
        let x_hex_byte = builder.constant_u8(120);
        for i in 0..rest_bytes.len() {
            rest_bytes[i].reverse(); // Make big-endian
            rest_bytes[i].insert(0, x_hex_byte);
            rest_bytes[i].insert(0, zero_hex_byte);
        }

        // Treat elements of L1_SIGNATURE_TEMPLATE_BYTES as constants
        let l1_signature_body_bytes: [U8Target; L1_SIGNATURE_TEMPLATE_BYTE_LEN] = [
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[0]),
            pub_key_bytes,
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[1]),
            rest_bytes[0].clone(),
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[2]),
            rest_bytes[1].clone(),
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[3]),
            rest_bytes[2].clone(),
            builder.constant_u8s(&L1_SIGNATURE_TEMPLATE_BYTES[4]),
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

const L1_SIGNATURE_TEMPLATE_BYTE_LEN: usize = 284;

lazy_static! {
    static ref L1_SIGNATURE_TEMPLATE_BYTES: Vec<Vec<u8>> = [ // Filled length -> 260 bytes
        // 26 - "\x19Ethereum Signed Message:\n"
        // 3 - %d (body len)
        b"\x19Ethereum Signed Message:\n255Register Lighter Account\n\npubkey: 0x".to_vec(), // 65 bytes
        // L2 pubkey -> 80 bytes
        b"\nnonce: ".to_vec(), // 8 bytes
        // nonceHex -> 10 bytes
        b"\naccount index: ".to_vec(), // 16 bytes
        // accountIndexHex -> 10 bytes
        b"\napi key index: ".to_vec(), // 16 bytes
        // apiKeyIndexHex -> 10 bytes
        b"\nOnly sign this message for a trusted client!".to_vec(), // 45 bytes
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
