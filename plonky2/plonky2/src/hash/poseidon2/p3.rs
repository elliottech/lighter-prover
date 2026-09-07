use p3_field::AbstractField;
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::Permutation;

use super::config::*;

// Poseidon2 from plonky3
pub fn p3_poseidon2_hash_n_to_m_no_pad(
    inputs: &[Goldilocks],
    num_outputs: usize,
) -> Vec<Goldilocks> {
    let external_linear_layer = Poseidon2ExternalMatrixGeneral;
    let internal_linear_layer = DiffusionMatrixGoldilocks;

    let external_constants = EXTERNAL_CONSTANTS
        .iter()
        .map(|v| {
            v.iter()
                .map(|&x| Goldilocks::from_canonical_u64(x))
                .collect::<Vec<Goldilocks>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<[Goldilocks; WIDTH]>>();

    let internal_constants = INTERNAL_CONSTANTS
        .iter()
        .map(|&x| Goldilocks::from_canonical_u64(x))
        .collect::<Vec<Goldilocks>>();

    let poseidon = Poseidon2::<
        Goldilocks,
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixGoldilocks,
        WIDTH,
        D,
    >::new(
        ROUNDS_F,
        external_constants,
        external_linear_layer,
        ROUNDS_P,
        internal_constants,
        internal_linear_layer,
    );

    let mut perm = [Goldilocks::zero(); WIDTH];

    #[allow(clippy::manual_memcpy)]
    for input_chunk in inputs.chunks(RATE) {
        for i in 0..RATE.min(input_chunk.len()) {
            perm[i] = input_chunk[i];
        }
        poseidon.permute_mut(&mut perm);
    }

    let mut outputs: Vec<Goldilocks> = Vec::new();
    loop {
        for &item in perm[0..RATE].iter() {
            outputs.push(item);
            if outputs.len() == num_outputs {
                return outputs;
            }
        }
        poseidon.permute_mut(&mut perm);
    }
}
