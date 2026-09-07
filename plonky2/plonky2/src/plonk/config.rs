//! Hashing configuration to be used when building a circuit.
//!
//! This module defines a [`Hasher`] trait as well as its recursive
//! counterpart [`AlgebraicHasher`] for in-circuit hashing. It also
//! provides concrete configurations, one fully recursive leveraging
//! the Poseidon hash function both internally and natively, and one
//! mixing Poseidon internally and truncated Keccak externally.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::keccak::KeccakHash;
use crate::hash::poseidon::PoseidonHash;
use crate::hash::poseidon2::hash::Poseidon2Hash;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;

pub trait GenericHashOut<F: RichField>:
    Copy + Clone + Debug + Eq + PartialEq + Send + Sync + Serialize + DeserializeOwned
{
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;

    fn to_vec(&self) -> Vec<F>;
}

/// Trait for hash functions.
pub trait Hasher<F: RichField>: Sized + Copy + Debug + Eq + PartialEq {
    /// Size of `Hash` in bytes.
    const HASH_SIZE: usize;

    /// Hash Output
    type Hash: GenericHashOut<F>;

    /// Permutation used in the sponge construction.
    type Permutation: PlonkyPermutation<F>;

    /// Hash a message without any padding step. Note that this can enable length-extension attacks.
    /// However, it is still collision-resistant in cases where the input has a fixed length.
    fn hash_no_pad(input: &[F]) -> Self::Hash;

    /// Pad the message using the `pad10*1` rule, then hash it.
    fn hash_pad(input: &[F]) -> Self::Hash {
        let mut padded_input = input.to_vec();
        padded_input.push(F::ONE);
        while !(padded_input.len() + 1).is_multiple_of(Self::Permutation::RATE) {
            padded_input.push(F::ZERO);
        }
        padded_input.push(F::ONE);
        Self::hash_no_pad(&padded_input)
    }

    /// Hash the slice if necessary to reduce its length to ~256 bits. If it already fits, this is a
    /// no-op.
    fn hash_or_noop(inputs: &[F]) -> Self::Hash {
        if inputs.len() * 8 <= Self::HASH_SIZE {
            let mut inputs_bytes = vec![0u8; Self::HASH_SIZE];
            for i in 0..inputs.len() {
                inputs_bytes[i * 8..(i + 1) * 8]
                    .copy_from_slice(&inputs[i].to_canonical_u64().to_le_bytes());
            }
            Self::Hash::from_bytes(&inputs_bytes)
        } else {
            Self::hash_no_pad(inputs)
        }
    }

    /// Hash two equal-length inputs, allowing implementations to interleave
    /// the two computations. Must return exactly
    /// `(Self::hash_or_noop(input_a), Self::hash_or_noop(input_b))`.
    fn hash_or_noop_pair(input_a: &[F], input_b: &[F]) -> (Self::Hash, Self::Hash) {
        (Self::hash_or_noop(input_a), Self::hash_or_noop(input_b))
    }

    /// Hash four equal-length inputs, allowing implementations to interleave
    /// the four computations. Must return exactly the four individual
    /// `Self::hash_or_noop` results.
    fn hash_or_noop_quad(
        input_a: &[F],
        input_b: &[F],
        input_c: &[F],
        input_d: &[F],
    ) -> (Self::Hash, Self::Hash, Self::Hash, Self::Hash) {
        (
            Self::hash_or_noop(input_a),
            Self::hash_or_noop(input_b),
            Self::hash_or_noop(input_c),
            Self::hash_or_noop(input_d),
        )
    }

    /// Two independent `two_to_one` compressions, allowing implementations to
    /// interleave them. Must return exactly
    /// `(Self::two_to_one(x0, y0), Self::two_to_one(x1, y1))`.
    fn two_to_one_pair(
        x0: Self::Hash,
        y0: Self::Hash,
        x1: Self::Hash,
        y1: Self::Hash,
    ) -> (Self::Hash, Self::Hash) {
        (Self::two_to_one(x0, y0), Self::two_to_one(x1, y1))
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash;

    /// Build the native Merkle digests and cap with a specialized backend, when available.
    ///
    /// `leaves` is one flat row-major buffer holding `num_leaves` leaves of `leaf_width`
    /// field elements each. The first result uses
    /// [`crate::hash::merkle_tree::MerkleTree::digests`] layout.
    #[allow(clippy::type_complexity)]
    fn try_build_merkle_tree(
        _leaves: &[F],
        _leaf_width: usize,
        _num_leaves: usize,
        _cap_height: usize,
    ) -> Option<(Vec<Self::Hash>, Vec<Self::Hash>)> {
        None
    }

    /// Like [`Hasher::try_build_merkle_tree`], but the leaves arrive as
    /// natural-order poly-major columns: tree leaf `i` is
    /// `columns[j][reverse_bits(i, log2(num_leaves))]`.
    #[allow(clippy::type_complexity)]
    fn try_build_merkle_tree_columns(
        _columns: &[Vec<F>],
        _cap_height: usize,
    ) -> Option<(Vec<Self::Hash>, Vec<Self::Hash>)> {
        None
    }

    /// Allocates retained column-major leaf storage suitable for a specialized
    /// Merkle backend. The caller may compute the columns directly in this
    /// storage before passing it to [`Hasher::try_build_merkle_tree_column_store`].
    #[allow(clippy::type_complexity)]
    fn try_allocate_merkle_tree_columns(
        _num_columns: usize,
        _num_rows: usize,
        _cap_height: usize,
    ) -> Option<crate::hash::merkle_tree::ColumnStore<F>> {
        None
    }

    /// Like [`Hasher::try_build_merkle_tree_columns`], but accepts retained
    /// column storage allocated by
    /// [`Hasher::try_allocate_merkle_tree_columns`].
    #[allow(clippy::type_complexity)]
    fn try_build_merkle_tree_column_store(
        columns: &crate::hash::merkle_tree::ColumnStore<F>,
        cap_height: usize,
    ) -> Option<(Vec<Self::Hash>, Vec<Self::Hash>)> {
        match columns {
            crate::hash::merkle_tree::ColumnStore::Owned(columns) => {
                Self::try_build_merkle_tree_columns(columns, cap_height)
            }
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            crate::hash::merkle_tree::ColumnStore::Shared(_) => None,
        }
    }

    /// Computes the coset LDE of the given coefficient columns and the Merkle
    /// tree over the resulting leaves in one fused backend pass, when a
    /// specialized backend is available. Returns the retained LDE column
    /// storage plus digests and cap in
    /// [`crate::hash::merkle_tree::MerkleTree::digests`] layout.
    #[allow(clippy::type_complexity)]
    fn try_build_commitment_from_coeffs(
        _coeff_columns: &[&[F]],
        _rate_bits: usize,
        _cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::ColumnStore<F>,
        Vec<Self::Hash>,
        Vec<Self::Hash>,
    )> {
        None
    }

    /// Like [`Hasher::try_build_commitment_from_coeffs`], but starting from
    /// evaluation values: the backend also performs the IFFT and returns the
    /// coefficient columns.
    #[allow(clippy::type_complexity)]
    fn try_build_commitment_from_values(
        _value_columns: &[&[F]],
        _rate_bits: usize,
        _cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::ColumnStore<F>,
        Vec<Self::Hash>,
        Vec<Self::Hash>,
        Vec<Vec<F>>,
    )> {
        None
    }
}

/// Trait for algebraic hash functions, built from a permutation using the sponge construction.
pub trait AlgebraicHasher<F: RichField>: Hasher<F, Hash = HashOut<F>> {
    type AlgebraicPermutation: PlonkyPermutation<Target>;

    /// Circuit to conditionally swap two chunks of the inputs (useful in verifying Merkle proofs),
    /// then apply the permutation.
    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>;
}

/// Generic configuration trait.
pub trait GenericConfig<const D: usize>:
    Debug + Clone + Sync + Sized + Send + Eq + PartialEq
{
    /// Main field.
    type F: RichField + Extendable<D, Extension = Self::FE>;
    /// Field extension of degree D of the main field.
    type FE: FieldExtension<D, BaseField = Self::F>;
    /// Hash function used for building Merkle trees.
    type Hasher: Hasher<Self::F>;
    /// Algebraic hash function used for the challenger and hashing public inputs.
    type InnerHasher: AlgebraicHasher<Self::F>;
}

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksConfig;
impl GenericConfig<2> for PoseidonGoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonHash;
    type InnerHasher = PoseidonHash;
}

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct Poseidon2GoldilocksConfig;
impl GenericConfig<2> for Poseidon2GoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = Poseidon2Hash;
    type InnerHasher = Poseidon2Hash;
}

/// Configuration using truncated Keccak over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct KeccakGoldilocksConfig;
impl GenericConfig<2> for KeccakGoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = KeccakHash<25>;
    type InnerHasher = PoseidonHash;
}
