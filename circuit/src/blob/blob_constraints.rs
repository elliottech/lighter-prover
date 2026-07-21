// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::Result;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use super::constants::*;
use crate::bigint::bigint::{BigIntTarget, CircuitBuilderBigInt, SignTarget};
use crate::bigint::biguint::{BigUintTarget, CircuitBuilderBiguint};
use crate::blob::blob_polynomial::BlobPolynomialTarget;
use crate::blob::bls12_381_scalar_field::BLS12381Scalar;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::keccak::keccak::{CircuitBuilderKeccak, KeccakOutputTarget};
use crate::nonnative::{CircuitBuilderNonNative, NonNativeTarget};
use crate::poseidon2::Poseidon2Hash;
use crate::types::config::{Builder, C, D, F};
use crate::types::constants::*;
use crate::types::market_details::{
    PublicMarketDetails, PublicMarketDetailsTarget, PublicMarketDetailsWitness,
};
use crate::uint::u8::{CircuitBuilderU8, U8Target, WitnessU8};
use crate::uint::u32::gadgets::arithmetic_u32::{CircuitBuilderU32, U32Target};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubDataMode {
    Blob,
    Calldata,
}

impl PubDataMode {
    pub fn from_env() -> Self {
        match std::env::var("PUBDATA_MODE") {
            Err(_) => PubDataMode::Blob,
            Ok(v) if v.eq_ignore_ascii_case("calldata") => PubDataMode::Calldata,
            Ok(_) => PubDataMode::Blob,
        }
    }
}

#[derive(Debug)]
/// Public + Secret Witness for single blob evaluation
pub struct BlobEvaluation<F>
where
    F: Field + Extendable<5> + RichField,
{
    pub kzg_versioned_hash: [u8; KECCAK_HASH_OUT_BYTE_SIZE],
    pub blob_bytes: Box<[u8; BLOB_DATA_BYTES_COUNT]>,
    pub blob_polynomial_opening_x: [u8; KECCAK_HASH_OUT_BYTE_SIZE],
    pub blob_polynomial_opening_y: [u8; KECCAK_HASH_OUT_BYTE_SIZE],

    pub account_delta_tree_root: HashOut<F>,
    pub public_market_details: [PublicMarketDetails; POSITION_LIST_SIZE],
}

pub trait Circuit<
    C: GenericConfig<D, F = F>,
    F: RichField + Extendable<D> + Extendable<5>,
    const D: usize,
>
{
    /// Defines the circuit and its each target. Returns `builder` and `target`
    ///
    /// `builder` can be used to build circuit via calling [`Builder::build()`]
    ///
    /// `target` can be used to assign partial witness in [`BlobEvaluationCircuit::prove()`] function
    fn define(config: CircuitConfig) -> Self;

    /// Fills partial witness for blob target with given blob data
    fn generate_witness(
        blob: &BlobEvaluation<F>,
        target: &BlobEvaluationTarget,
    ) -> Result<PartialWitness<F>>;
    /// Takes `circuit`, blob witness and `target` defined in [`BlobEvaluationCircuit::define()`] function
    /// and returns the (not-compressed) proof with public inputs
    fn prove(
        circuit: &CircuitData<F, C, D>,
        blob: &BlobEvaluation<F>,
        bt: &BlobEvaluationTarget,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;
}

#[derive(Debug)]
pub struct BlobEvaluationCircuit {
    pub builder: Builder,
    pub target: BlobEvaluationTarget,
}

#[derive(Clone, Debug)]
pub struct BlobEvaluationTarget {
    pub account_delta_tree_root: HashOutTarget,
    pub public_market_details: [PublicMarketDetailsTarget; POSITION_LIST_SIZE],
    pub blob_bytes: Box<[U8Target; BLOB_DATA_BYTES_COUNT]>, // 0 byte at the beginning of each 32 byte limb is omitted
    pub kzg_versioned_hash: KeccakOutputTarget,
    pub blob_polynomial_opening_x: KeccakOutputTarget,
    pub blob_polynomial_opening_y: KeccakOutputTarget,
}

impl BlobEvaluationTarget {
    pub fn from_public_inputs(public_inputs: &[Target]) -> Self {
        let blob_bytes = Box::new(
            public_inputs[0..BLOB_DATA_BYTES_COUNT]
                .iter()
                .map(|&b| U8Target(b))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );

        let kzg_versioned_hash = public_inputs
            [BLOB_DATA_BYTES_COUNT..BLOB_DATA_BYTES_COUNT + KECCAK_HASH_OUT_BYTE_SIZE]
            .iter()
            .map(|&t| U8Target(t))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let blob_polynomial_opening_x = public_inputs[BLOB_DATA_BYTES_COUNT
            + KECCAK_HASH_OUT_BYTE_SIZE
            ..BLOB_DATA_BYTES_COUNT + 2 * KECCAK_HASH_OUT_BYTE_SIZE]
            .iter()
            .map(|&t| U8Target(t))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let blob_polynomial_opening_y = public_inputs[BLOB_DATA_BYTES_COUNT
            + 2 * KECCAK_HASH_OUT_BYTE_SIZE
            ..BLOB_DATA_BYTES_COUNT + 3 * KECCAK_HASH_OUT_BYTE_SIZE]
            .iter()
            .map(|&t| U8Target(t))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut offset = BLOB_DATA_BYTES_COUNT + 3 * KECCAK_HASH_OUT_BYTE_SIZE;
        let public_market_details = public_inputs[offset..offset + POSITION_LIST_SIZE * 5]
            .chunks(5)
            .map(|chunk| PublicMarketDetailsTarget {
                funding_rate_prefix_sum: BigIntTarget {
                    sign: SignTarget::new_unsafe(chunk[0]),
                    abs: BigUintTarget {
                        limbs: vec![U32Target(chunk[1]), U32Target(chunk[2])],
                    },
                },
                mark_price: chunk[3],
                quote_multiplier: chunk[4],
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        offset += POSITION_LIST_SIZE * 5;
        let account_delta_tree_root = HashOutTarget {
            elements: public_inputs[offset..offset + NUM_HASH_OUT_ELTS]
                .try_into()
                .unwrap(),
        };

        Self {
            account_delta_tree_root,
            public_market_details,
            blob_bytes,
            kzg_versioned_hash,
            blob_polynomial_opening_x,
            blob_polynomial_opening_y,
        }
    }
}

impl Circuit<C, F, D> for BlobEvaluationCircuit {
    fn define(config: CircuitConfig) -> Self {
        let mut circuit = Self::new(config);

        circuit.register_public_inputs();

        circuit.verify_pce_evaluation();

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn prove(
        circuit: &CircuitData<F, C, D>,
        block: &BlobEvaluation<F>,
        target: &BlobEvaluationTarget,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("BlobEvaluationCircuit::prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(block, target)?
        });
        let proof = prove::<F, C, D>(&circuit.prover_only, &circuit.common, pw, &mut timing)?;
        timed!(timing, "verify", { circuit.verify(proof.clone())? });

        timing.print();
        Ok(proof)
    }

    fn generate_witness(
        blob: &BlobEvaluation<F>,
        target: &BlobEvaluationTarget,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        for i in 0..KECCAK_HASH_OUT_BYTE_SIZE {
            pw.set_target(
                target.kzg_versioned_hash[i].0,
                F::from_canonical_u8(blob.kzg_versioned_hash[i]),
            )?;

            pw.set_target(
                target.blob_polynomial_opening_x[i].0,
                F::from_canonical_u8(blob.blob_polynomial_opening_x[i]),
            )?;
            pw.set_target(
                target.blob_polynomial_opening_y[i].0,
                F::from_canonical_u8(blob.blob_polynomial_opening_y[i]),
            )?;
        }

        for i in 0..BLOB_DATA_BYTES_COUNT {
            pw.set_u8_target(target.blob_bytes[i], blob.blob_bytes[i])?;
        }

        pw.set_hash_target(target.account_delta_tree_root, blob.account_delta_tree_root)?;
        for i in 0..POSITION_LIST_SIZE {
            pw.set_public_market_details_target(
                &target.public_market_details[i],
                &blob.public_market_details[i],
            )?;
        }

        Ok(pw)
    }
}

impl BlobEvaluationCircuit {
    /// Initializes a new blob virtual targets.
    pub fn new(config: CircuitConfig) -> Self {
        let mut builder = Builder::new(config);

        let public_market_details =
            core::array::from_fn(|_| PublicMarketDetailsTarget::new(&mut builder));

        Self {
            target: BlobEvaluationTarget {
                blob_bytes: Box::new(core::array::from_fn(|_| {
                    builder.add_virtual_u8_target_unsafe()
                })), // Safety of these values are checked in inner witness circuit
                kzg_versioned_hash: builder.add_virtual_keccak_output_target_unsafe(),
                blob_polynomial_opening_x: builder.add_virtual_keccak_output_target_unsafe(),
                blob_polynomial_opening_y: builder.add_virtual_keccak_output_target_unsafe(),
                account_delta_tree_root: builder.add_virtual_hash(),

                public_market_details,
            },
            builder,
        }
    }

    pub fn register_public_inputs(&mut self) {
        self.builder
            .register_public_u8_inputs(&self.target.blob_bytes.to_vec());
        self.builder
            .register_public_keccak_output_input(self.target.kzg_versioned_hash);
        self.builder
            .register_public_keccak_output_input(self.target.blob_polynomial_opening_x);
        self.builder
            .register_public_keccak_output_input(self.target.blob_polynomial_opening_y);
        self.target.public_market_details.iter().for_each(|md| {
            md.register_public_input(&mut self.builder);
        });
        self.builder
            .register_public_hashout(self.target.account_delta_tree_root);
    }

    pub fn verify_pce_evaluation(&mut self) {
        let blob_data_hash = self._get_blob_data_hash();
        let pce_evaluation_point = self._get_pce_evaluation_point(&blob_data_hash);

        let polynomial =
            BlobPolynomialTarget::from_bytes(&mut self.builder, &self.target.blob_bytes);
        let pce_evaluation_result = match PubDataMode::from_env() {
            PubDataMode::Blob => {
                polynomial.eval_barycentric(&mut self.builder, &pce_evaluation_point)
            }
            PubDataMode::Calldata => {
                polynomial.eval_horner(&mut self.builder, &pce_evaluation_point)
            }
        };

        let point_y_nonnative = NonNativeTarget::<BLS12381Scalar>::from(
            self.builder
                .biguint_from_bytes_be(&self.target.blob_polynomial_opening_y),
        );
        self.builder
            .connect_nonnative(&pce_evaluation_result, &point_y_nonnative);
    }

    fn _get_blob_data_hash(&mut self) -> HashOutTarget {
        let reserved_hash = self._get_version_and_reserved_bytes_hash();
        let market_data_hash = self._get_market_data_hash();

        self.builder.hash_n_to_one(&[
            reserved_hash,
            market_data_hash,
            self.target.account_delta_tree_root,
        ])
    }

    /// Commits to version bytes and the reserved section.
    /// Version is written in big endian order, and reserved bytes are splitted into chunks
    /// of 4 bytes, each of which is written in big endian order.
    fn _get_version_and_reserved_bytes_hash(&mut self) -> HashOutTarget {
        let blob_bytes = &self.target.blob_bytes;
        let multiplier = self.builder.constant_u64(1 << 8);
        let version = self.builder.mul_add(
            multiplier,
            blob_bytes[BLOB_VERSION_INDEX].0,
            blob_bytes[BLOB_VERSION_INDEX + 1].0,
        );
        let mut limbs = vec![version];
        for chunk in blob_bytes[BLOB_RESERVED_INDEX..BLOB_MARK_PRICE_INDEX].chunks(4) {
            let mut res = chunk[0].0;
            for i in 1..4 {
                res = self.builder.mul_add(multiplier, res, chunk[i].0);
            }
            limbs.push(res);
        }
        self.builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(limbs)
    }

    /// Commits to the market data
    fn _get_market_data_hash(&mut self) -> HashOutTarget {
        let mut limbs = vec![];
        for i in 0..POSITION_LIST_SIZE {
            limbs.push(self.target.public_market_details[i].mark_price);
        }
        for i in 0..POSITION_LIST_SIZE {
            limbs.push(
                self.builder
                    .is_sign_negative(
                        self.target.public_market_details[i]
                            .funding_rate_prefix_sum
                            .sign,
                    )
                    .target,
            );
            limbs.push(
                self.target.public_market_details[i]
                    .funding_rate_prefix_sum
                    .abs
                    .limbs[1]
                    .0,
            );
            limbs.push(
                self.target.public_market_details[i]
                    .funding_rate_prefix_sum
                    .abs
                    .limbs[0]
                    .0,
            );
        }
        for i in 0..POSITION_LIST_SIZE {
            limbs.push(self.target.public_market_details[i].quote_multiplier);
        }
        self.builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(limbs)
    }

    /// Proof of commitment equivalence - Recompute and verify evaluation point x
    fn _get_pce_evaluation_point(
        &mut self,
        circuit_commitment: &HashOutTarget,
    ) -> NonNativeTarget<BLS12381Scalar> {
        // Fiat shamir with kzg versioned hash and in-circuit blob hash
        let hash_in = circuit_commitment
            .elements
            .iter()
            .chain(self.target.kzg_versioned_hash.iter().map(|x| &x.0))
            .copied()
            .collect::<Vec<_>>();
        let hash_out = self.builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(hash_in);

        let challenge_point_biguint = BigUintTarget {
            limbs: hash_out
                .elements
                .map(|elem| self.builder.split_u64_to_u32s_le(elem))
                .iter()
                .flat_map(|e| *e)
                .collect::<Vec<_>>(),
        };

        let point_x_nonnative: NonNativeTarget<BLS12381Scalar> =
            self.builder.reduce(&challenge_point_biguint);

        let point_x_from_witness: NonNativeTarget<BLS12381Scalar> =
            NonNativeTarget::<BLS12381Scalar>::from(
                self.builder
                    .biguint_from_bytes_be(&self.target.blob_polynomial_opening_x),
            );

        self.builder
            .connect_nonnative(&point_x_from_witness, &point_x_nonnative);

        point_x_from_witness
    }
}
