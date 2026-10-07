use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

fn main() {
    Registry::default()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(ForestLayer::default())
        .init();

    let capacity: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let real: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(capacity);
    let batch = p3_schnorr::reference::deterministic_batch(real).expect("batch");

    for round in 0..2 {
        let t = std::time::Instant::now();
        let output =
            p3_schnorr::prove_signature_batch_with_capacity(&batch, capacity).expect("prove");
        let prove_ms = t.elapsed().as_millis();
        eprintln!(
            "round {round}: capacity={capacity} real={real} trace_gen+stark_prove={prove_ms}ms rows={}",
            output.proof.rows()
        );
        let t = std::time::Instant::now();
        p3_schnorr::verify_signature_batch_proof(&output.public_inputs, &output.proof)
            .expect("verify");
        eprintln!("round {round}: verified in {}ms", t.elapsed().as_millis());
    }
}
