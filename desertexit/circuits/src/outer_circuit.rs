// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use circuit::keccak::keccak::KeccakOutputTarget;
use circuit::poseidon_bn128::plonky2_config::PoseidonBN128GoldilocksConfig;
use circuit::types::config::{Builder, C, D, F};
use circuit::uint::u8::{CircuitBuilderU8, U8Target};
use log::Level;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::plonk::prover::prove;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

#[derive(Debug)]
pub struct OuterDesertExitTarget {
    pub inner_wrapper_proof: ProofWithPublicInputsTarget<D>,
    pub inner_wrapper_verifier: VerifierCircuitTarget,

    pub exit_commitment: KeccakOutputTarget, // public
}

#[derive(Debug)]
pub struct OuterDesertExitCircuit {
    pub builder: Builder,
    pub target: OuterDesertExitTarget,
}

impl OuterDesertExitCircuit {
    pub fn new(
        config: CircuitConfig,
        inner_circuit: &CommonCircuitData<F, D>,
        inner_verifier: &VerifierOnlyCircuitData<C, D>,
    ) -> Self {
        let mut builder = Builder::new(config);

        let inner_proof = builder.add_virtual_proof_with_pis(inner_circuit);

        Self {
            target: OuterDesertExitTarget {
                inner_wrapper_proof: inner_proof.clone(),
                inner_wrapper_verifier: builder.constant_verifier_data(inner_verifier),
                exit_commitment: core::array::from_fn(|i| U8Target(inner_proof.public_inputs[i])),
            },
            builder,
        }
    }

    pub fn define(
        config: CircuitConfig,
        inner_circuit: &CommonCircuitData<F, D>,
        inner_verifier: &VerifierOnlyCircuitData<C, D>,
    ) -> Self {
        let mut circuit = OuterDesertExitCircuit::new(config, inner_circuit, inner_verifier);

        circuit
            .builder
            .register_public_u8_inputs(&circuit.target.exit_commitment);

        circuit.builder.verify_proof::<C>(
            &circuit.target.inner_wrapper_proof,
            &circuit.target.inner_wrapper_verifier,
            inner_circuit,
        );

        circuit
    }

    pub fn prove(
        circuit: &CircuitData<F, PoseidonBN128GoldilocksConfig, D>,
        target: &OuterDesertExitTarget,
        inner_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, PoseidonBN128GoldilocksConfig, D>> {
        let mut timing = TimingTree::new("WrapperCircuit::prove_outer", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(target, inner_proof)?
        });

        let proof = prove::<F, PoseidonBN128GoldilocksConfig, D>(
            &circuit.prover_only,
            &circuit.common,
            pw,
            &mut timing,
        )?;
        timed!(timing, "verify", { circuit.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    fn generate_witness(
        target: &OuterDesertExitTarget,
        inner_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.inner_wrapper_proof, inner_proof)?;

        Ok(pw)
    }
}
