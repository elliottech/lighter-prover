use std::time::Instant;

use anyhow::Result;
use log::Level;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::prover::prove;
use plonky2::util::timing::TimingTree;

/// An example of using Plonky2 to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
/// When a == 0 and b == 1, this is proving knowledge of the 100th (standard) Fibonacci number.
fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
    work::<PoseidonGoldilocksConfig>()?;
    work::<Poseidon2GoldilocksConfig>()
}

fn work<C: GenericConfig<2>>() -> Result<()> {
    const D: usize = 2;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<C::F, D>::new(config);

    // The arithmetic circuit.
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..999999 {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    // Public inputs are the two initial values (provided below) and the result (which is generated).
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    // Provide initial values.
    let timer1 = Instant::now();
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, C::F::ZERO)?;
    pw.set_target(initial_b, C::F::ONE)?;

    let data = builder.build::<C>();
    let timer2 = Instant::now();

    // Create a TimingTree to track detailed timing information
    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove::<C::F, C, D>(&data.prover_only, &data.common, pw, &mut timing)?;
    let timer3 = Instant::now();

    // Print the timing tree
    timing.print();

    println!(
        "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
        proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
    );

    println!("Build time: {:?}", timer2.duration_since(timer1));
    println!("Prove time: {:?}", timer3.duration_since(timer2));

    data.verify(proof)
}
