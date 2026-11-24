#![allow(clippy::needless_range_loop)]

//! Plonky3 batch verifier for standard secp256k1 ECDSA signatures.

pub mod air;
pub mod batch;
pub mod digest;
pub mod error;
pub mod keccak;
pub mod prove;
pub mod secp;
pub mod stark;
pub mod testing;
pub mod types;

pub use air::EcdsaBatchAir;
pub use batch::{BatchProof, PreparedBatch};
pub use error::{Error, Result};
pub use prove::{EcdsaBatchProof, PreparedEcdsaBatch, prove_ecdsa_batch};
pub use types::EcdsaInstance;

#[cfg(test)]
mod tests;
