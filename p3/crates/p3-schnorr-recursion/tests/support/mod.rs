#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

use p3_schnorr::reference::deterministic_batch;
use p3_schnorr::{
    SignatureBatchProvingOutput, prove_signature_batch, prove_signature_batch_with_capacity,
};

fn cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-cache/p3-schnorr-recursion")
}

fn cache_path(signature_count: usize) -> PathBuf {
    cache_dir().join(format!("signature-batch-{signature_count}.bin"))
}

fn capacity_cache_path(signature_count: usize, capacity: usize) -> PathBuf {
    cache_dir().join(format!(
        "signature-batch-{signature_count}-of-{capacity}.bin"
    ))
}

pub fn cached_proving_output(signature_count: usize) -> SignatureBatchProvingOutput {
    let path = cache_path(signature_count);
    if let Ok(bytes) = fs::read(&path)
        && let Ok(output) = SignatureBatchProvingOutput::from_bytes(&bytes)
    {
        return output;
    }

    let batch = deterministic_batch(signature_count).expect("build deterministic batch");
    let output = prove_signature_batch(&batch).expect("prove deterministic batch");
    let bytes = output.to_bytes().expect("serialize proving output");

    fs::create_dir_all(path.parent().expect("cache path has parent"))
        .expect("create test cache directory");
    fs::write(&path, bytes).expect("write proving output cache");

    output
}

pub fn cached_proving_output_with_capacity(
    signature_count: usize,
    capacity: usize,
) -> SignatureBatchProvingOutput {
    let path = capacity_cache_path(signature_count, capacity);
    if let Ok(bytes) = fs::read(&path)
        && let Ok(output) = SignatureBatchProvingOutput::from_bytes(&bytes)
    {
        return output;
    }

    let batch = deterministic_batch(signature_count).expect("build deterministic batch");
    let output =
        prove_signature_batch_with_capacity(&batch, capacity).expect("prove deterministic batch");
    let bytes = output.to_bytes().expect("serialize proving output");

    fs::create_dir_all(path.parent().expect("cache path has parent"))
        .expect("create test cache directory");
    fs::write(&path, bytes).expect("write proving output cache");

    output
}
