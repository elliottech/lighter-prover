use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use p3_schnorr::air::{
    SignatureChallengeHint, check_constraints, generate_trace, generate_trace_with_challenge_hints,
};
use p3_schnorr::proof::{
    prepare_signature_batch_verifier, prove_signature_batch_trace,
    verify_signature_batch_proof_preprocessed,
};
use p3_schnorr::reference::deterministic_batch;
use p3_schnorr::schnorr::verification_witness;
use p3_schnorr::types::Batch;

fn bench_air(c: &mut Criterion) {
    for count in [4, 32, 500] {
        bench_batch(c, count);
    }
}

fn challenge_hints(batch: &Batch) -> Vec<SignatureChallengeHint> {
    batch
        .instances()
        .iter()
        .map(|instance| {
            let witness = verification_witness(instance).expect("reference witness");
            SignatureChallengeHint::from_reference(instance, witness.reconstructed_r)
                .expect("challenge hint")
        })
        .collect()
}

fn bench_batch(c: &mut Criterion, count: usize) {
    let batch = deterministic_batch(count).expect("reference batch");
    let challenge_hints = challenge_hints(&batch);
    let trace = generate_trace(&batch).expect("signature-batch trace");
    let output = prove_signature_batch_trace(trace.clone()).expect("signature-batch proof");
    let verifier =
        prepare_signature_batch_verifier(&output.public_inputs).expect("prepared verifier");

    c.bench_function(&format!("air/trace_generation/{count}"), |b| {
        b.iter(|| generate_trace(black_box(&batch)).expect("signature-batch trace"));
    });

    c.bench_function(&format!("air/trace_generation_with_hints/{count}"), |b| {
        b.iter(|| {
            generate_trace_with_challenge_hints(black_box(&batch), black_box(&challenge_hints))
                .expect("signature-batch trace")
        });
    });

    c.bench_function(&format!("air/check_constraints/{count}"), |b| {
        b.iter(|| check_constraints(black_box(&trace)));
    });

    c.bench_function(&format!("air/prove/{count}"), |b| {
        b.iter_batched(
            || trace.clone(),
            |trace| prove_signature_batch_trace(trace).expect("signature-batch proof"),
            BatchSize::LargeInput,
        );
    });

    c.bench_function(&format!("air/verify/{count}"), |b| {
        b.iter(|| {
            verify_signature_batch_proof_preprocessed(
                black_box(&verifier),
                black_box(&output.public_inputs),
                black_box(&output.proof),
            )
            .expect("verify signature-batch proof");
        });
    });
}

criterion_group!(benches, bench_air);
criterion_main!(benches);
