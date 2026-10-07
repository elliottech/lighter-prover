#![cfg_attr(
    not(all(feature = "metal", target_arch = "aarch64", target_os = "macos")),
    no_std
)]

extern crate alloc;

#[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
mod metal;

mod hiding_mmcs;
mod merkle_tree;
mod mmcs;
mod pruning;

pub use hiding_mmcs::*;
pub use merkle_tree::MerkleTree;
pub use mmcs::{MerkleTreeError, MerkleTreeMmcs, PrunedBatchOpening, PrunedProofError};
pub use p3_symmetric::MerkleCap;
pub use pruning::{PrunedMerklePaths, PrunedPath};
