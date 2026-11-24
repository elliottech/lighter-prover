//! Per-phase timings (tracing-forest spans) for one ECDSA + keccak batch proof.
//!
//! `cargo run --release -p p3-ecdsa [--features metal] --example stark_profile [capacity]`

use p3_ecdsa::batch::PreparedBatch;
use p3_ecdsa::testing::deterministic_batch;
use tracing::info_span;
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
    let batch = deterministic_batch(capacity, 7);
    let prepared = info_span!("setup").in_scope(|| PreparedBatch::new(capacity));
    let t = std::time::Instant::now();
    info_span!("ecdsa + keccak prove").in_scope(|| prepared.prove(&batch).expect("prove"));
    eprintln!("capacity={capacity} prove={}ms", t.elapsed().as_millis());
}
