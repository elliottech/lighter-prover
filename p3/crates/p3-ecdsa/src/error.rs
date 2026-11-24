use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("batch is empty")]
    EmptyBatch,
    #[error("batch of {len} signatures exceeds capacity {capacity}")]
    CapacityTooSmall { len: usize, capacity: usize },
    #[error("instance {0}: public key is not a valid compressed secp256k1 point")]
    InvalidPublicKey(usize),
    #[error("instance {0}: signature does not verify")]
    InvalidSignature(usize),
    #[error("proof verification failed: {0}")]
    Verification(String),
}

pub type Result<T> = core::result::Result<T, Error>;
