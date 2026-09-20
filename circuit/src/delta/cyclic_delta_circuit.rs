// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use anyhow::{Ok, Result};
use hashbrown::HashMap;
use log::Level;
use plonky2::field::extension::Extendable;
use plonky2::gates::addition_base::AdditionGate;
use plonky2::gates::arithmetic_extension::ArithmeticExtensionGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::exponentiation::ExponentiationGate;
use plonky2::gates::multiplication_extension::MulExtensionGate;
use plonky2::gates::poseidon::PoseidonGate;
use plonky2::gates::poseidon_mds::PoseidonMdsGate;
use plonky2::gates::random_access::RandomAccessGate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::gates::select_base::SelectionGate;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget,
};
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::recursion::dummy_circuit::cyclic_base_proof;
use plonky2::timed;
use plonky2::util::timing::TimingTree;

use crate::delta::types::{AggregatedDeltaTarget, DeltaPublicInputTarget, DeltaPublicOutputTarget};
use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};
use crate::eddsa::gates::mul_quintic_ext_base::QuinticMultiplicationGate;
use crate::eddsa::gates::square_quintic_ext_base::QuinticSquaringGate;
use crate::hash_utils::CircuitBuilderHashUtils;
use crate::types::config::{Builder, C, CIRCUIT_CONFIG, D, F};
use crate::types::constants::{ACCOUNT_MERKLE_LEVELS, EMPTY_DELTA_TREE_HASHES};

pub trait Circuit<C: GenericConfig<D, F = F>, F: RichField + Extendable<D>, const D: usize> {
    /// Defines the circuit and its each target. Returns `builder` and `target`
    /// `builder` can be used to build circuit via calling [`Builder::build()`]
    /// `target` can be used to assign partial witness in [`CyclicDeltaCircuit::prove()`] function
    fn define(config: CircuitConfig, delta_circuit: &CircuitData<F, C, D>) -> Self;

    fn generate_witness(
        target: &CyclicDeltaTarget,
        circuit_data: &CircuitData<F, C, D>,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        delta_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>>;

    fn prove(
        target: &CyclicDeltaTarget,
        circuit_data: &CircuitData<F, C, D>,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        delta_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>>;

    fn cyclic_base_proof(circuit_data: &CircuitData<F, C, D>) -> ProofWithPublicInputs<F, C, D>;
}

#[derive(Debug)]
pub struct CyclicDeltaCircuit {
    pub builder: Builder,
    pub target: CyclicDeltaTarget,
}

#[derive(Debug)]
pub struct CyclicDeltaTarget {
    // Input
    pub cyclic_proof: ProofWithPublicInputsTarget<D>, // proof of previous iteration
    pub self_verifier_data: VerifierCircuitTarget,    // Verifier Circuit Data for this circuit
    pub delta_proof: ProofWithPublicInputsTarget<D>,  // proof of next delta
    pub not_first_recursion: BoolTarget, // indicator that wheter we are on first iteration or not
    pub dummy_proof_with_pis_target: ProofWithPublicInputsTarget<D>, // Filled with dummy proof

    // Output
    pub aggregated_out: AggregatedDeltaTarget, // Public witness
}

impl CyclicDeltaCircuit {
    pub fn new(
        config: CircuitConfig,
        delta_circuit_common_data: &CommonCircuitData<F, D>,
    ) -> (Self, CommonCircuitData<F, D>) {
        let mut builder = Builder::new(config);

        // Register public inputs
        let aggregated_out = AggregatedDeltaTarget::new_public(&mut builder);
        let self_verifier_data = builder.add_verifier_data_public_inputs(); // 68

        // IMPORTANT: DO NOT ADD PUBLIC INPUTS AFTER THIS POINT. Building common data for current circuit
        let common_data_for_recursion = CommonCircuitData {
            num_public_inputs: builder.num_public_inputs(),
            ..common_data_for_recursion()
        };

        (
            Self {
                target: CyclicDeltaTarget {
                    cyclic_proof: builder.add_virtual_proof_with_pis(&common_data_for_recursion),
                    not_first_recursion: builder.add_virtual_bool_target_safe(),
                    delta_proof: builder.add_virtual_proof_with_pis(delta_circuit_common_data),
                    aggregated_out,
                    self_verifier_data,
                    dummy_proof_with_pis_target: builder
                        .add_virtual_proof_with_pis(&common_data_for_recursion), // This value will be overwritten
                },
                builder,
            },
            common_data_for_recursion,
        )
    }

    /// Verifies proofs and extracts information from them
    fn handle_proofs(
        &mut self,
        self_common_data: CommonCircuitData<F, D>,
        delta_circuit: &CircuitData<F, C, D>,
    ) -> (
        AggregatedDeltaTarget,
        DeltaPublicInputTarget,
        DeltaPublicOutputTarget,
    ) {
        // Verify cyclic proof
        self.target.dummy_proof_with_pis_target = self
            .builder
            .conditionally_verify_cyclic_proof_or_dummy::<C>(
                self.target.not_first_recursion,
                &self.target.cyclic_proof,
                &self_common_data,
            )
            .unwrap();
        let mut aggregated_in =
            AggregatedDeltaTarget::from_public_inputs(&self.target.cyclic_proof.public_inputs);

        // Verify delta proof
        let vd = self
            .builder
            .constant_verifier_data(&delta_circuit.verifier_only);
        self.builder
            .verify_proof::<C>(&self.target.delta_proof, &vd, &delta_circuit.common);
        let delta_in = DeltaPublicInputTarget::from_public_inputs(
            &self.target.delta_proof.public_inputs[..DeltaPublicInputTarget::DELTA_PUB_IN_SIZE],
        );
        let delta_out = DeltaPublicOutputTarget::from_public_inputs(
            &self.target.delta_proof.public_inputs[DeltaPublicInputTarget::DELTA_PUB_IN_SIZE..],
        );

        // Get the evaluation point from the delta input for the first recursion
        aggregated_in.evaluation_point = self.builder.select_quintic_ext(
            self.target.not_first_recursion,
            aggregated_in.evaluation_point,
            delta_in.evaluation_point,
        );
        // Set evaulation to 0 for the first recursion
        let zero_eval = self.builder.zero_quintic_ext();
        aggregated_in.evaluation = self.builder.select_quintic_ext(
            self.target.not_first_recursion,
            aggregated_in.evaluation,
            zero_eval,
        );
        // Set degree to 0 for the first recursion
        let zero = self.builder.zero();
        aggregated_in.degree =
            self.builder
                .select(self.target.not_first_recursion, aggregated_in.degree, zero);
        // Set -1 account index for the first iteration
        let neg_one = self.builder.neg_one();
        aggregated_in.account_index = self.builder.select(
            self.target.not_first_recursion,
            aggregated_in.account_index,
            neg_one,
        );
        // Set empty path for the first iteration
        for i in 0..2 {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                let empty_level_hash = self.builder.constant_hash(EMPTY_DELTA_TREE_HASHES[j]);
                aggregated_in.path_matrix[i][j] = self.builder.select_hash(
                    self.target.not_first_recursion,
                    &aggregated_in.path_matrix[i][j],
                    &empty_level_hash,
                );
            }
        }
        // Make sure first iteration starts with zero market delta hash
        let zero_hash = self.builder.zero_hash_out();
        aggregated_in.market_delta_hash = self.builder.select_hash(
            self.target.not_first_recursion,
            &aggregated_in.market_delta_hash,
            &zero_hash,
        );
        // Set -1 remaining market delta count for the first iteration
        aggregated_in.remaining_market_delta_count = self.builder.select(
            self.target.not_first_recursion,
            aggregated_in.remaining_market_delta_count,
            neg_one,
        );

        (aggregated_in, delta_in, delta_out)
    }

    /// Connects output of the last iteration to the input of the current iteration.
    fn perform_sanity_checks(
        &mut self,
        aggregated_in: &AggregatedDeltaTarget,
        delta_in: &DeltaPublicInputTarget,
    ) {
        self.builder
            .connect_quintic_ext(aggregated_in.evaluation_point, delta_in.evaluation_point);
        self.builder
            .connect(aggregated_in.account_index, delta_in.account_index);
        for i in 0..2 {
            for j in 0..ACCOUNT_MERKLE_LEVELS {
                self.builder
                    .connect_hashes(aggregated_in.path_matrix[i][j], delta_in.path_matrix[i][j]);
            }
        }
        self.builder
            .connect_hashes(aggregated_in.market_delta_hash, delta_in.market_delta_hash);
        self.builder.connect(
            aggregated_in.remaining_market_delta_count,
            delta_in.remaining_market_delta_count,
        );
    }

    /// Shift the evaluation from the current delta proof by the degree aggregated so far,
    /// and update the degree.
    fn merge_evaluations(
        &mut self,
        aggregated_in: &AggregatedDeltaTarget,
        delta_out: &DeltaPublicOutputTarget,
    ) -> (QuinticExtensionTarget, Target) {
        let evaluation_point_to_degree =
            self.builder
                .exp_quintic_ext(aggregated_in.evaluation_point, delta_out.degree, 18);
        let delta_eval_raised = self
            .builder
            .mul_quintic_ext(aggregated_in.evaluation, evaluation_point_to_degree);

        (
            self.builder
                .add_quintic_ext(delta_out.evaluation, delta_eval_raised),
            self.builder.add(aggregated_in.degree, delta_out.degree),
        )
    }

    fn set_aggregated_out(
        &mut self,
        evaluation_point: QuinticExtensionTarget,
        delta_out: &DeltaPublicOutputTarget,
        new_evaluation: &QuinticExtensionTarget,
        new_degree: Target,
    ) {
        let aggregated_out = AggregatedDeltaTarget {
            account_index: delta_out.account_index,
            evaluation_point,
            path_matrix: delta_out.path_matrix,
            evaluation: *new_evaluation,
            degree: new_degree,
            market_delta_hash: delta_out.market_delta_hash,
            remaining_market_delta_count: delta_out.remaining_market_delta_count,
        };

        self.target
            .aggregated_out
            .connect(&mut self.builder, &aggregated_out);
    }
}

impl Circuit<C, F, D> for CyclicDeltaCircuit {
    fn define(config: CircuitConfig, delta_circuit: &CircuitData<F, C, D>) -> Self {
        let (mut circuit, common_data) = Self::new(config, &delta_circuit.common);

        let (aggregated_in, delta_in, delta_out) =
            circuit.handle_proofs(common_data, delta_circuit);

        circuit.perform_sanity_checks(&aggregated_in, &delta_in);

        let (new_evaluation, new_degree) = circuit.merge_evaluations(&aggregated_in, &delta_out);

        circuit.set_aggregated_out(
            aggregated_in.evaluation_point,
            &delta_out,
            &new_evaluation,
            new_degree,
        );

        circuit.builder.perform_registered_range_checks();

        circuit
    }

    fn generate_witness(
        target: &CyclicDeltaTarget,
        circuit_data: &CircuitData<F, C, D>,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        delta_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<PartialWitness<F>> {
        let mut pw = PartialWitness::new();

        pw.set_proof_with_pis_target(&target.cyclic_proof, cyclic_proof)?;
        pw.set_verifier_data_target(&target.self_verifier_data, &circuit_data.verifier_only)?;

        pw.set_proof_with_pis_target(&target.delta_proof, delta_proof)?;

        pw.set_bool_target(target.not_first_recursion, not_first_recursion)?;

        // This will take place of `DummyProofGenerator`
        pw.set_proof_with_pis_target(&target.dummy_proof_with_pis_target, dummy_proof)?;

        Ok(pw)
    }

    fn prove(
        target: &CyclicDeltaTarget,
        circuit_data: &CircuitData<F, C, D>,
        not_first_recursion: bool,
        cyclic_proof: &ProofWithPublicInputs<F, C, D>,
        dummy_proof: &ProofWithPublicInputs<F, C, D>,
        delta_proof: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let mut timing = TimingTree::new("cyclic delta prove", Level::Debug);

        let pw = timed!(timing, "witness", {
            Self::generate_witness(
                target,
                circuit_data,
                not_first_recursion,
                cyclic_proof,
                dummy_proof,
                delta_proof,
            )?
        });
        let proof = circuit_data.prove(pw)?;
        timed!(timing, "verify", { circuit_data.verify(proof.clone())? });

        timing.print();

        Ok(proof)
    }

    fn cyclic_base_proof(circuit_data: &CircuitData<F, C, D>) -> ProofWithPublicInputs<F, C, D> {
        cyclic_base_proof(
            &circuit_data.common,
            &circuit_data.verifier_only,
            HashMap::new(),
        )
    }
}

// Generates `CommonCircuitData` usable for recursion.
fn common_data_for_recursion() -> CommonCircuitData<F, D> {
    let builder = Builder::new(CIRCUIT_CONFIG);
    let data = builder.build::<C>();

    let mut builder = Builder::new(CIRCUIT_CONFIG);
    let proof = builder.add_virtual_proof_with_pis(&data.common);
    let verifier_data = builder.add_virtual_verifier_data(data.common.config.fri_config.cap_height);
    builder.verify_proof::<C>(&proof, &verifier_data, &data.common);
    let data = builder.build::<C>();

    let mut builder = Builder::new(CIRCUIT_CONFIG);
    let proof = builder.add_virtual_proof_with_pis(&data.common);
    let verifier_data = builder.add_virtual_verifier_data(data.common.config.fri_config.cap_height);
    builder.verify_proof::<C>(&proof, &verifier_data, &data.common);

    builder.add_gate(MulExtensionGate::new_from_config(&CIRCUIT_CONFIG), vec![]);
    builder.add_gate(PoseidonGate::new(), vec![]);
    builder.add_gate(PoseidonMdsGate::new(), vec![]);
    builder.add_gate(ReducingGate::new(44), vec![]);
    builder.add_gate(ReducingExtensionGate::new(33), vec![]);
    builder.add_gate(
        ArithmeticExtensionGate::new_from_config(&CIRCUIT_CONFIG),
        vec![],
    );
    builder.add_gate(
        RandomAccessGate::new_from_config(&CIRCUIT_CONFIG, 4),
        vec![],
    );
    builder.add_gate(
        QuinticMultiplicationGate::new_from_config(&CIRCUIT_CONFIG),
        vec![],
    );
    builder.add_gate(
        QuinticSquaringGate::new_from_config(&CIRCUIT_CONFIG),
        vec![],
    );
    builder.add_gate(BaseSumGate::<2>::new(63), vec![]);
    builder.add_gate(ExponentiationGate::new(67), vec![]);
    builder.add_gate(SelectionGate::new_from_config(&CIRCUIT_CONFIG), vec![]);
    builder.add_gate(AdditionGate::new_from_config(&CIRCUIT_CONFIG), vec![]);
    builder.add_gate(ConstantGate::new(2), vec![]);

    while builder.num_gates() < 1 << 13 {
        builder.add_gate(plonky2::gates::noop::NoopGate, vec![]);
    }

    builder.build::<C>().common
}
