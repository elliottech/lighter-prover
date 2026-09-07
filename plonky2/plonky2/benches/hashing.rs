mod allocator;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::AbstractField;
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_poseidon2::{Poseidon2 as P3Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::Permutation;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Sample;
use plonky2::hash::hash_types::{BytesHash, RichField};
use plonky2::hash::keccak::KeccakHash;
use plonky2::hash::poseidon::{Poseidon, SPONGE_WIDTH};
use plonky2::hash::poseidon2::hash::Poseidon2;
use plonky2::plonk::config::Hasher;
use tynm::type_name;

pub(crate) fn bench_keccak<F: RichField>(c: &mut Criterion) {
    c.bench_function("keccak256", |b| {
        b.iter_batched(
            || (BytesHash::<32>::rand(), BytesHash::<32>::rand()),
            |(left, right)| <KeccakHash<32> as Hasher<F>>::two_to_one(left, right),
            BatchSize::SmallInput,
        )
    });
}

pub(crate) fn bench_poseidon<F: Poseidon>(c: &mut Criterion) {
    c.bench_function(
        &format!("poseidon<{}, {SPONGE_WIDTH}>", type_name::<F>()),
        |b| {
            b.iter_batched(
                || F::rand_array::<SPONGE_WIDTH>(),
                |state| F::poseidon(state),
                BatchSize::SmallInput,
            )
        },
    );
}

pub(crate) fn bench_poseidon2<F: Poseidon2>(c: &mut Criterion) {
    c.bench_function(
        &format!("optimized poseidon2<{}, {SPONGE_WIDTH}>", type_name::<F>()),
        |b| {
            b.iter_batched(
                || F::rand_array::<SPONGE_WIDTH>(),
                |state| F::poseidon2(state),
                BatchSize::SmallInput,
            )
        },
    );
}

pub(crate) fn bench_p3_poseidon2(c: &mut Criterion) {
    const WIDTH: usize = 12;
    const D: u64 = 7;
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 22;

    // Create the Poseidon2 instance
    let external_linear_layer = Poseidon2ExternalMatrixGeneral;
    let internal_linear_layer = DiffusionMatrixGoldilocks;

    let external_constants = plonky2::hash::poseidon2::config::EXTERNAL_CONSTANTS
        .iter()
        .map(|v| {
            v.iter()
                .map(|&x| Goldilocks::from_canonical_u64(x))
                .collect::<Vec<Goldilocks>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<[Goldilocks; WIDTH]>>();

    let internal_constants = plonky2::hash::poseidon2::config::INTERNAL_CONSTANTS
        .iter()
        .map(|&x| Goldilocks::from_canonical_u64(x))
        .collect::<Vec<Goldilocks>>();

    let poseidon = P3Poseidon2::<
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

    c.bench_function("plonky3's poseidon2", |b| {
        b.iter_batched(
            || {
                let mut state = [Goldilocks::zero(); WIDTH];
                state.iter_mut().for_each(|item| {
                    *item = Goldilocks::from_canonical_u64(rand::random::<u64>());
                });
                state
            },
            |mut state| {
                poseidon.permute_mut(&mut state);
                state
            },
            BatchSize::SmallInput,
        )
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_poseidon::<GoldilocksField>(c);
    bench_poseidon2::<GoldilocksField>(c);
    bench_p3_poseidon2(c);
    bench_keccak::<GoldilocksField>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
