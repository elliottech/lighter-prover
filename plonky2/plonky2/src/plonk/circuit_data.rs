//! Circuit data specific to the prover and the verifier.
//!
//! This module also defines a [`CircuitConfig`] to be customized
//! when building circuits for arbitrary statements.
//!
//! After building a circuit, one obtains an instance of [`CircuitData`].
//! This contains both prover and verifier data, allowing to generate
//! proofs for the given circuit and verify them.
//!
//! Most of the [`CircuitData`] is actually prover-specific, and can be
//! extracted by calling [`CircuitData::prover_data`] method.
//! The verifier data can similarly be extracted by calling [`CircuitData::verifier_data`].
//! This is useful to allow even small devices to verify plonky2 proofs.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::ops::{Range, RangeFrom};
#[cfg(feature = "std")]
use std::{collections::BTreeMap, sync::Arc};

use anyhow::Result;
use serde::Serialize;

use super::circuit_builder::LookupWire;
use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::reduction_strategies::FriReductionStrategy;
use crate::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOracleInfo,
    FriPolynomialInfo,
};
use crate::fri::{FriConfig, FriParams};
use crate::gates::gate::GateRef;
use crate::gates::lookup::Lookup;
use crate::gates::lookup_table::LookupTable;
use crate::gates::selectors::SelectorsInfo;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{generate_partial_witness, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use crate::plonk::prover::prove;
use crate::plonk::verifier::verify;
use crate::util::serialization::{
    Buffer, GateSerializer, IoResult, Read, WitnessGeneratorSerializer, Write,
};
use crate::util::timing::TimingTree;

/// Configuration to be used when building a circuit. This defines the shape of the circuit
/// as well as its targeted security level and sub-protocol (e.g. FRI) parameters.
///
/// It supports a [`Default`] implementation tailored for recursion with Poseidon hash (of width 12)
/// as internal hash function and FRI rate of 1/8.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CircuitConfig {
    /// The number of wires available at each row. This corresponds to the "width" of the circuit,
    /// and consists in the sum of routed wires and advice wires.
    pub num_wires: usize,
    /// The number of routed wires, i.e. wires that will be involved in Plonk's permutation argument.
    /// This allows copy constraints, i.e. enforcing that two distant values in a circuit are equal.
    /// Non-routed wires are called advice wires.
    pub num_routed_wires: usize,
    /// The number of constants that can be used per gate. If a gate requires more constants than the config
    /// allows, the [`CircuitBuilder`] will complain when trying to add this gate to its set of gates.
    pub num_constants: usize,
    /// Whether to use a dedicated gate for base field arithmetic, rather than using a single gate
    /// for both base field and extension field arithmetic.
    pub use_base_arithmetic_gate: bool,
    pub security_bits: usize,
    /// The number of challenge points to generate, for IOPs that have soundness errors of (roughly)
    /// `degree / |F|`.
    pub num_challenges: usize,
    /// A boolean to activate the zero-knowledge property. When this is set to `false`, proofs *may*
    /// leak additional information.
    pub zero_knowledge: bool,
    /// A cap on the quotient polynomial's degree factor. The actual degree factor is derived
    /// systematically, but will never exceed this value.
    pub max_quotient_degree_factor: usize,
    pub fri_config: FriConfig,
    pub optimization_flags: usize,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::standard_recursion_config()
    }
}

impl CircuitConfig {
    pub const fn num_advice_wires(&self) -> usize {
        self.num_wires - self.num_routed_wires
    }

    /// A typical recursion config, without zero-knowledge, targeting ~100 bit security.
    pub const fn standard_recursion_config() -> Self {
        Self {
            num_wires: 135,
            num_routed_wires: 80,
            num_constants: 2,
            use_base_arithmetic_gate: true,
            security_bits: 100,
            num_challenges: 2,
            zero_knowledge: false,
            max_quotient_degree_factor: 8,
            fri_config: FriConfig {
                rate_bits: 3,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 28,
            },
            optimization_flags: 0,
        }
    }

    pub fn standard_ecc_config() -> Self {
        Self {
            num_wires: 136,
            ..Self::standard_recursion_config()
        }
    }

    pub fn wide_ecc_config() -> Self {
        Self {
            num_wires: 234,
            ..Self::standard_recursion_config()
        }
    }

    pub fn standard_recursion_zk_config() -> Self {
        CircuitConfig {
            zero_knowledge: true,
            ..Self::standard_recursion_config()
        }
    }

    pub fn addition_gate_enabled(&self) -> bool {
        0 < (self.optimization_flags & (1 << 0))
    }

    pub fn multiplication_gate_enabled(&self) -> bool {
        0 < (self.optimization_flags & (1 << 1))
    }

    pub fn quintic_multiplication_gate_enabled(&self) -> bool {
        0 < (self.optimization_flags & (1 << 2))
    }

    pub fn equality_gate_enable(&self) -> bool {
        0 < (self.optimization_flags & (1 << 3))
    }

    pub fn quintic_squaring_gate_enabled(&self) -> bool {
        0 < (self.optimization_flags & (1 << 4))
    }

    pub fn select_gate_enabled(&self) -> bool {
        0 < (self.optimization_flags & (1 << 5))
    }
}

/// Mock circuit data to only do witness generation without generating a proof.
#[derive(Eq, PartialEq, Debug)]
pub struct MockCircuitData<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    MockCircuitData<F, C, D>
{
    pub fn generate_witness(&self, inputs: PartialWitness<F>) -> PartitionWitness<'_, F> {
        generate_partial_witness::<F, C, D>(inputs, &self.prover_only, &self.common).unwrap()
    }
}

/// Circuit data required by the prover or the verifier.
#[derive(Eq, PartialEq, Debug)]
pub struct CircuitData<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub verifier_only: VerifierOnlyCircuitData<C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_circuit_data(self, gate_serializer, generator_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_circuit_data(gate_serializer, generator_serializer)
    }

    pub fn prove(&self, inputs: PartialWitness<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        prove::<F, C, D>(
            &self.prover_only,
            &self.common,
            inputs,
            &mut TimingTree::default(),
        )
    }

    pub fn verify(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()> {
        verify::<F, C, D>(proof_with_pis, &self.verifier_only, &self.common)
    }

    pub fn verify_compressed(
        &self,
        compressed_proof_with_pis: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<()> {
        compressed_proof_with_pis.verify(&self.verifier_only, &self.common)
    }

    pub fn compress(
        &self,
        proof: ProofWithPublicInputs<F, C, D>,
    ) -> Result<CompressedProofWithPublicInputs<F, C, D>> {
        proof.compress(&self.verifier_only.circuit_digest, &self.common)
    }

    pub fn decompress(
        &self,
        proof: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        proof.decompress(&self.verifier_only.circuit_digest, &self.common)
    }

    pub fn verifier_data(&self) -> VerifierCircuitData<F, C, D> {
        let CircuitData {
            verifier_only,
            common,
            ..
        } = self;
        VerifierCircuitData {
            verifier_only: verifier_only.clone(),
            common: common.clone(),
        }
    }

    pub fn prover_data(self) -> ProverCircuitData<F, C, D> {
        let CircuitData {
            prover_only,
            common,
            ..
        } = self;
        ProverCircuitData {
            prover_only,
            common,
        }
    }
}

/// Circuit data required by the prover. This may be thought of as a proving key, although it
/// includes code for witness generation.
///
/// The goal here is to make proof generation as fast as we can, rather than making this prover
/// structure as succinct as we can. Thus we include various precomputed data which isn't strictly
/// required, like LDEs of preprocessed polynomials. If more succinctness was desired, we could
/// construct a more minimal prover structure and convert back and forth.
#[derive(Debug)]
pub struct ProverCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverCircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_prover_circuit_data(self, gate_serializer, generator_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_prover_circuit_data(gate_serializer, generator_serializer)
    }

    pub fn prove(&self, inputs: PartialWitness<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        prove::<F, C, D>(
            &self.prover_only,
            &self.common,
            inputs,
            &mut TimingTree::default(),
        )
    }
}

/// Circuit data required by the prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub verifier_only: VerifierOnlyCircuitData<C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    VerifierCircuitData<F, C, D>
{
    pub fn to_bytes(&self, gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_verifier_circuit_data(self, gate_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        gate_serializer: &dyn GateSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(&bytes);
        buffer.read_verifier_circuit_data(gate_serializer)
    }

    pub fn verify(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()> {
        verify::<F, C, D>(proof_with_pis, &self.verifier_only, &self.common)
    }

    pub fn verify_compressed(
        &self,
        compressed_proof_with_pis: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<()> {
        compressed_proof_with_pis.verify(&self.verifier_only, &self.common)
    }
}

/// Generator indices grouped by the representative target they watch.
///
/// Representatives are dense target indices, so a CSR offset table turns the prover's frequent
/// watcher lookup into two adjacent loads and one contiguous slice. The previous `BTreeMap`
/// required a pointer-chasing tree walk for every newly populated representative.
#[derive(Eq, PartialEq, Debug)]
pub struct GeneratorWatchIndex {
    offsets: Vec<u32>,
    /// Generator indices, `u32` rather than `usize`: the circuit builder already
    /// refuses a generator count that does not fit a `u32` (see the guards in
    /// [`Self::from_map`] and [`Self::from_sorted_generator_representatives`]),
    /// and the serialized form has always been 4-byte little-endian, so the
    /// narrow payload is the on-disk width.
    watchers: Vec<u32>,
    entries: usize,
    /// One bit per representative, set exactly when that representative's watcher list is
    /// non-empty (`offsets[r] != offsets[r + 1]`).
    ///
    /// `offsets` carries one `u32` per representative -- 36.7 MB on the light transaction
    /// circuit -- and every prover lookup indexes it at a freshly populated representative,
    /// i.e. at a scattered position. Measured on the public fixture, 178.96 M of the light
    /// path's 218.79 M lookups (81.8 %) name a representative nobody watches, so each paid a
    /// scattered probe into that 36.7 MB table only to compare two equal words. This bitmap
    /// answers the same question in 1/32 of the bytes (1.15 MB, cache-resident), so those
    /// lookups never touch `offsets` at all.
    ///
    /// Pure function of `offsets`, derived at construction inside passes that already walk
    /// them; `get` returns exactly what it returned before, so no witness value, no queue
    /// push and no proof byte can move.
    watched: Vec<u64>,
}

/// Sets bit `representative` of a [`GeneratorWatchIndex::watched`] bitmap under construction.
#[inline]
pub fn mark_watched(watched: &mut [u64], representative: usize) {
    watched[representative >> 6] |= 1u64 << (representative & 63);
}

/// Allocates a zeroed [`GeneratorWatchIndex::watched`] bitmap sized for `offsets_len` offsets,
/// i.e. for the `offsets_len - 1` representatives those offsets describe.
pub fn empty_watched(offsets_len: usize) -> Vec<u64> {
    vec![0u64; offsets_len.saturating_sub(1).div_ceil(64)]
}

impl GeneratorWatchIndex {
    pub fn from_map(map: BTreeMap<usize, Vec<usize>>) -> Self {
        let entries = map.values().filter(|watchers| !watchers.is_empty()).count();
        let Some((&max_representative, _)) = map.last_key_value() else {
            return Self {
                offsets: vec![0],
                watchers: Vec::new(),
                entries: 0,
                watched: Vec::new(),
            };
        };

        let offsets_len = max_representative
            .checked_add(2)
            .expect("generator watch representative index overflow");
        let total_watchers = map.values().map(Vec::len).sum::<usize>();
        assert!(
            u32::try_from(total_watchers).is_ok(),
            "generator watch index exceeds u32 offsets"
        );

        let mut offsets = vec![0u32; offsets_len];
        let mut watched = empty_watched(offsets_len);
        let mut watchers = Vec::with_capacity(total_watchers);
        let mut entries_iter = map.into_iter().peekable();
        for representative in 0..=max_representative {
            offsets[representative] = watchers.len() as u32;
            if entries_iter
                .peek()
                .is_some_and(|(key, _)| *key == representative)
            {
                let (_, representative_watchers) = entries_iter.next().unwrap();
                if !representative_watchers.is_empty() {
                    mark_watched(&mut watched, representative);
                }
                watchers.extend(representative_watchers.into_iter().map(|generator| {
                    u32::try_from(generator).expect("generator index exceeds u32 watch index")
                }));
            }
        }
        offsets[max_representative + 1] = watchers.len() as u32;
        debug_assert!(entries_iter.next().is_none());

        Self {
            offsets,
            watchers,
            entries,
            watched,
        }
    }

    /// Builds the CSR directly from consecutive, per-generator groups of sorted, distinct
    /// representative indices. The groups themselves are ordered by generator index.
    ///
    /// This is the circuit builder's trusted construction seam. Keeping the transient edge list
    /// flat avoids a tree node and a separately allocated `Vec` for every watched representative.
    pub(crate) fn from_sorted_generator_representatives(
        representatives: &[u32],
        generator_watch_counts: &[usize],
    ) -> Self {
        debug_assert_eq!(
            generator_watch_counts.iter().sum::<usize>(),
            representatives.len()
        );
        debug_assert!({
            let mut end = 0usize;
            generator_watch_counts.iter().all(|&count| {
                let start = end;
                end += count;
                representatives[start..end]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            })
        });

        let Some(&max_representative) = representatives.iter().max() else {
            return Self {
                offsets: vec![0],
                watchers: Vec::new(),
                entries: 0,
                watched: Vec::new(),
            };
        };
        let max_representative = max_representative as usize;
        let offsets_len = max_representative
            .checked_add(2)
            .expect("generator watch representative index overflow");
        assert!(
            u32::try_from(representatives.len()).is_ok(),
            "generator watch index exceeds u32 offsets"
        );
        assert!(
            u32::try_from(generator_watch_counts.len()).is_ok(),
            "generator count exceeds u32 watch index"
        );

        // First form cumulative end offsets. Counts live in slot `representative + 1`, so the
        // prefix sum is already the normal CSR layout before it is reused as a fill cursor below.
        let mut offsets = vec![0u32; offsets_len];
        let mut entries = 0usize;
        for &representative in representatives {
            let count = &mut offsets[representative as usize + 1];
            entries += usize::from(*count == 0);
            *count += 1;
        }
        let mut total = 0u32;
        for offset in &mut offsets[1..] {
            total += *offset;
            *offset = total;
        }
        debug_assert_eq!(total as usize, representatives.len());

        // Fill each representative's slice backwards while visiting generators backwards. This
        // preserves the old ascending generator order without a second cursor array. Afterwards,
        // each end cursor has become the next representative's start, so one overlapping shift
        // restores the original CSR offsets.
        let mut watchers = vec![0u32; representatives.len()];
        let mut group_end = representatives.len();
        for (generator, &count) in generator_watch_counts.iter().enumerate().rev() {
            let group_start = group_end - count;
            for &representative in &representatives[group_start..group_end] {
                let cursor = &mut offsets[representative as usize + 1];
                *cursor -= 1;
                watchers[*cursor as usize] = generator as u32;
            }
            group_end = group_start;
        }
        debug_assert_eq!(group_end, 0);
        offsets.copy_within(2.., 1);
        *offsets.last_mut().unwrap() = total;

        // Derived from the finished offsets, in the same shape `from_parts` uses.
        let mut watched = empty_watched(offsets.len());
        for (representative, bounds) in offsets.windows(2).enumerate() {
            if bounds[0] != bounds[1] {
                mark_watched(&mut watched, representative);
            }
        }

        Self {
            offsets,
            watchers,
            entries,
            watched,
        }
    }

    /// The raw CSR offset table (`representative -> [start, end)` into
    /// [`Self::watchers`]). Exposed for compact serialization of the index.
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// The flat, concatenated watcher lists indexed by [`Self::offsets`].
    pub fn watchers(&self) -> &[u32] {
        &self.watchers
    }

    /// Rebuilds the index from its raw CSR parts (as exposed by
    /// [`Self::offsets`] and [`Self::watchers`]); the `entries` count is a pure
    /// function of the offsets and is re-derived. The offsets must be
    /// monotonically nondecreasing, start at 0 and end at `watchers.len()`,
    /// exactly as [`Self::from_map`] produces them.
    pub fn from_parts(offsets: Vec<u32>, watchers: Vec<u32>) -> Self {
        assert!(!offsets.is_empty(), "watch index offsets must be non-empty");
        assert_eq!(offsets[0], 0, "watch index offsets must start at zero");
        assert_eq!(
            *offsets.last().unwrap() as usize,
            watchers.len(),
            "watch index offsets must cover the watcher list"
        );
        let mut entries = 0usize;
        // The presence bitmap is filled by this existing validation walk, so deriving it
        // costs no extra traversal of the offsets table on the circuit-loading path.
        let mut watched = empty_watched(offsets.len());
        for (representative, bounds) in offsets.windows(2).enumerate() {
            assert!(bounds[0] <= bounds[1], "watch index offsets must be sorted");
            if bounds[0] != bounds[1] {
                entries += 1;
                mark_watched(&mut watched, representative);
            }
        }
        Self {
            offsets,
            watchers,
            entries,
            watched,
        }
    }

    /// [`Self::from_parts`] for a loader that has already derived the presence bitmap and
    /// entry count while decoding the offsets, using [`empty_watched`] and [`mark_watched`].
    ///
    /// The offsets arrive as a running sum of unsigned deltas, so they are monotonic by
    /// construction and the sortedness assertion of the general constructor has nothing left
    /// to catch; the invariants that are not structural -- first offset zero, last offset
    /// covering the watcher list, bitmap sized for these offsets -- are still checked here.
    /// `representative` is watched exactly when `offsets[r] != offsets[r + 1]`, i.e. exactly
    /// when the delta read for `r + 1` was non-zero, which is the bit the decode loop already
    /// has in hand.
    pub fn from_parts_with_presence(
        offsets: Vec<u32>,
        watchers: Vec<u32>,
        entries: usize,
        watched: Vec<u64>,
    ) -> Self {
        assert!(!offsets.is_empty(), "watch index offsets must be non-empty");
        assert_eq!(offsets[0], 0, "watch index offsets must start at zero");
        assert_eq!(
            *offsets.last().unwrap() as usize,
            watchers.len(),
            "watch index offsets must cover the watcher list"
        );
        assert_eq!(
            watched.len(),
            empty_watched(offsets.len()).len(),
            "watch index presence bitmap is sized for different offsets"
        );
        Self {
            offsets,
            watchers,
            entries,
            watched,
        }
    }

    #[inline]
    pub fn get(&self, representative: &usize) -> Option<&[u32]> {
        let representative = *representative;
        // Answer the common case (nobody watches this representative) out of the 1.15 MB
        // bitmap instead of the 36.7 MB offsets table; see [`Self::watched`]. A bit is set
        // only for representatives strictly below `offsets.len() - 1` whose watcher list is
        // non-empty, so reaching the indexing below implies both `offsets` reads are in
        // bounds and that the slice is non-empty -- exactly the old `Some` condition.
        if (self.watched.get(representative >> 6)? >> (representative & 63)) & 1 == 0 {
            return None;
        }
        let start = self.offsets[representative] as usize;
        let end = self.offsets[representative + 1] as usize;
        debug_assert!(start != end);
        Some(&self.watchers[start..end])
    }

    pub const fn len(&self) -> usize {
        self.entries
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &[u32])> {
        self.offsets
            .windows(2)
            .enumerate()
            .filter_map(|(representative, bounds)| {
                let start = bounds[0] as usize;
                let end = bounds[1] as usize;
                (start != end).then(|| (representative, &self.watchers[start..end]))
            })
    }
}

/// Runtime-only cache of immutable low-degree range-gate selector filters.
///
/// The filters are derived from the constants commitment and never participate
/// in circuit identity, serialization, or equality. On non-Metal targets the
/// wrapper is zero-sized.
pub struct LowRangeSelectorFilterCache<F> {
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    inner: std::sync::OnceLock<Option<LowRangeSelectorFilterCacheEntry<F>>>,
    _phantom: core::marker::PhantomData<F>,
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) struct LowRangeSelectorFilterCacheEntry<F> {
    pub(crate) full_rows: usize,
    pub(crate) gate_signature: Vec<(usize, usize, usize, usize, bool)>,
    pub(crate) filters: Vec<Vec<F>>,
}

impl<F> Default for LowRangeSelectorFilterCache<F> {
    fn default() -> Self {
        Self {
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            inner: std::sync::OnceLock::new(),
            _phantom: core::marker::PhantomData,
        }
    }
}

// This is a derived runtime cache of immutable circuit data, so its fill state
// must not change circuit equality (the same rule as `EvenColumns`).
impl<F> PartialEq for LowRangeSelectorFilterCache<F> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<F> Eq for LowRangeSelectorFilterCache<F> {}
impl<F> core::fmt::Debug for LowRangeSelectorFilterCache<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("LowRangeSelectorFilterCache")
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
impl<F> LowRangeSelectorFilterCache<F> {
    pub(crate) fn get_or_init(
        &self,
        init: impl FnOnce() -> Option<LowRangeSelectorFilterCacheEntry<F>>,
    ) -> Option<&LowRangeSelectorFilterCacheEntry<F>> {
        self.inner.get_or_init(init).as_ref()
    }
}

/// Circuit data required by the prover, but not the verifier.
#[derive(Debug)]
pub struct ProverOnlyCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub generators: Vec<WitnessGeneratorRef<F, D>>,
    /// Generator indices (within the `Vec` above), indexed by the representative of each target
    /// they watch.
    pub generator_indices_by_watches: GeneratorWatchIndex,
    /// For each generator (indexed as in `generators`), the number of *distinct* representatives
    /// it watches — equivalently, the number of entries of `generator_indices_by_watches` whose
    /// watcher list contains that generator.
    ///
    /// Derived once inside the builder's `generator_indices_by_watches` construction pass so that
    /// witness generation can seed its `unresolved_watches` counters by cloning this vector and
    /// decrementing on first population, instead of traversing the entire watcher map at the
    /// start of every proof. Runtime-only: it is a pure function of `generator_indices_by_watches`
    /// and is reconstructed on deserialization, so the serialized format is unchanged.
    pub generator_watch_counts: Vec<usize>,
    /// Whether every generator in [`Self::generators`] reports
    /// [`WitnessGenerator::defers_until_ready`].
    ///
    /// When it holds, the worklist may skip a queued generator whose `unresolved_watches`
    /// counter is still non-zero without dispatching to it, because that dispatch is a proven
    /// no-op. Measured on the public fixture, 66.77 M of the light path's 98.15 M generator
    /// invocations (68 %) are such no-ops, each costing a scattered load out of the 10 MB
    /// `Box<dyn WitnessGenerator>` table and an indirect call.
    ///
    /// Runtime-only: a pure function of `generators`, re-derived wherever they are, so the
    /// serialized format is unchanged.
    pub generators_defer_until_ready: bool,
    /// Commitments to the constants polynomials and sigma polynomials.
    pub constants_sigmas_commitment: PolynomialBatch<F, C, D>,
    /// The transpose of the list of sigma polynomials.
    pub sigmas: Vec<Vec<F>>,
    /// Subgroup of order `degree`.
    pub subgroup: Vec<F>,
    /// Targets to be made public.
    pub public_inputs: Vec<Target>,
    /// A map from each `Target`'s index to the index of its representative in the disjoint-set
    /// forest.
    ///
    /// Stored as `u32` (see [`crate::plonk::permutation_argument::Forest`]); values are
    /// zero-extended at every indexing site. The serialized encoding keeps the legacy 8-byte
    /// per-entry format.
    pub representative_map: Vec<u32>,
    /// One bit per routed `(row, column)` position, in row-major order. A set bit means that the
    /// position is the sole routed member of its copy-constraint component, hence its sigma
    /// permutation target is itself and its permutation factor cancels for every proof.
    ///
    /// Runtime-only: this is derived from [`Self::representative_map`] during circuit construction
    /// and reconstructed during deserialization, so it changes neither the serialized format nor
    /// the circuit digest.
    pub fixed_routed_wires: Vec<u8>,
    /// Pre-computed roots for faster FFT.
    ///
    /// Held by shared handle. The table is a deterministic function of (field,
    /// domain size) and is immutable once built -- `field::fft` keeps one
    /// process-wide copy per size and every reader here takes it as
    /// `Option<&FftRootTable<F>>`. Owning it by value forced each circuit load
    /// to deep-copy that cached `Vec<Vec<F>>`.
    pub fft_root_table: Option<Arc<FftRootTable<F>>>,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash,
    ///The concrete placement of the lookup gates for each lookup table index.
    pub lookup_rows: Vec<LookupWire>,
    /// A vector of (looking_in, looking_out) pairs for each lookup table index.
    pub lut_to_lookups: Vec<Lookup>,
    /// Quotient-domain values of the constants and sigma columns (PolyMajor:
    /// all `constants_range().len() + sigmas_range().len()` columns, each a
    /// `constants_sigmas_quotient_domain`-length slice, constants first), plus
    /// the gather parameters they were extracted with. The constants and sigma
    /// polynomials are circuit-fixed, so these strided LDE values are
    /// identical for every proof of this circuit; the quotient batch loop
    /// copies from here instead of re-walking the LDE. `None` when the
    /// commitment is not column-backed or the cache would be too large.
    /// Runtime-only: not serialized (the quotient path falls back to the
    /// strided gather on a deserialized circuit).
    pub constants_sigmas_quotient_cache: Option<Vec<F>>,
    /// Stride used to extract [`Self::constants_sigmas_quotient_cache`].
    pub constants_sigmas_quotient_step: usize,
    /// Quotient domain size used to extract [`Self::constants_sigmas_quotient_cache`].
    pub constants_sigmas_quotient_domain: usize,
    /// Immutable low-degree range-gate selector filters, filled on first use
    /// only when the process-wide cache budget admits the exact table.
    /// Runtime-only: derived from the constants commitment and not serialized.
    pub low_range_selector_filter_cache: LowRangeSelectorFilterCache<F>,
}

/// Equality is over the serialized content only. Runtime-only fields
/// (`generator_watch_counts`, `generators_defer_until_ready`,
/// `fixed_routed_wires`, the quotient/selector caches) are either pure
/// functions of the compared fields or opportunistic caches that a
/// deserialized circuit legitimately lacks, so they do not participate.
impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> PartialEq
    for ProverOnlyCircuitData<F, C, D>
{
    fn eq(&self, other: &Self) -> bool {
        self.generators == other.generators
            && self.generator_indices_by_watches == other.generator_indices_by_watches
            && self.constants_sigmas_commitment == other.constants_sigmas_commitment
            && self.sigmas == other.sigmas
            && self.subgroup == other.subgroup
            && self.public_inputs == other.public_inputs
            && self.representative_map == other.representative_map
            && self.fft_root_table == other.fft_root_table
            && self.circuit_digest == other.circuit_digest
            && self.lookup_rows == other.lookup_rows
            && self.lut_to_lookups == other.lut_to_lookups
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Eq
    for ProverOnlyCircuitData<F, C, D>
{
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverOnlyCircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_prover_only_circuit_data(self, generator_serializer, common_data)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_prover_only_circuit_data(generator_serializer, common_data)
    }
}

/// Circuit data required by the verifier, but not the prover.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct VerifierOnlyCircuitData<C: GenericConfig<D>, const D: usize> {
    /// A commitment to each constant polynomial and each permutation polynomial.
    pub constants_sigmas_cap: MerkleCap<C::F, C::Hasher>,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
}

impl<C: GenericConfig<D>, const D: usize> VerifierOnlyCircuitData<C, D> {
    pub fn to_bytes(&self) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_verifier_only_circuit_data(self)?;
        Ok(buffer)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> IoResult<Self> {
        let mut buffer = Buffer::new(&bytes);
        buffer.read_verifier_only_circuit_data()
    }
}

/// Circuit data required by both the prover and the verifier.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommonCircuitData<F: RichField + Extendable<D>, const D: usize> {
    pub config: CircuitConfig,

    pub fri_params: FriParams,

    /// The types of gates used in this circuit, along with their prefixes.
    pub gates: Vec<GateRef<F, D>>,

    /// Information on the circuit's selector polynomials.
    pub selectors_info: SelectorsInfo,

    /// The degree of the PLONK quotient polynomial.
    pub quotient_degree_factor: usize,

    /// The largest number of constraints imposed by any gate.
    pub num_gate_constraints: usize,

    /// The number of constant wires.
    pub num_constants: usize,

    pub num_public_inputs: usize,

    /// The `{k_i}` valued used in `S_ID_i` in Plonk's permutation argument.
    pub k_is: Vec<F>,

    /// The number of partial products needed to compute the `Z` polynomials.
    pub num_partial_products: usize,

    /// The number of lookup polynomials.
    pub num_lookup_polys: usize,

    /// The number of lookup selectors.
    pub num_lookup_selectors: usize,

    /// The stored lookup tables.
    pub luts: Vec<LookupTable>,
}

impl<F: RichField + Extendable<D>, const D: usize> CommonCircuitData<F, D> {
    pub fn to_bytes(&self, gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_common_circuit_data(self, gate_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        gate_serializer: &dyn GateSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(&bytes);
        buffer.read_common_circuit_data(gate_serializer)
    }

    pub const fn degree_bits(&self) -> usize {
        self.fri_params.degree_bits
    }

    pub const fn degree(&self) -> usize {
        1 << self.degree_bits()
    }

    pub const fn lde_size(&self) -> usize {
        self.fri_params.lde_size()
    }

    pub fn lde_generator(&self) -> F {
        F::primitive_root_of_unity(self.degree_bits() + self.config.fri_config.rate_bits)
    }

    pub fn constraint_degree(&self) -> usize {
        self.gates
            .iter()
            .map(|g| g.0.degree())
            .max()
            .expect("No gates?")
    }

    pub const fn quotient_degree(&self) -> usize {
        self.quotient_degree_factor * self.degree()
    }

    /// Range of the constants polynomials in the `constants_sigmas_commitment`.
    pub const fn constants_range(&self) -> Range<usize> {
        0..self.num_constants
    }

    /// Range of the sigma polynomials in the `constants_sigmas_commitment`.
    pub const fn sigmas_range(&self) -> Range<usize> {
        self.num_constants..self.num_constants + self.config.num_routed_wires
    }

    /// Range of the `z`s polynomials in the `zs_partial_products_commitment`.
    pub const fn zs_range(&self) -> Range<usize> {
        0..self.config.num_challenges
    }

    /// Range of the partial products polynomials in the `zs_partial_products_lookup_commitment`.
    pub const fn partial_products_range(&self) -> Range<usize> {
        self.config.num_challenges..(self.num_partial_products + 1) * self.config.num_challenges
    }

    /// Range of lookup polynomials in the `zs_partial_products_lookup_commitment`.
    pub const fn lookup_range(&self) -> RangeFrom<usize> {
        self.num_zs_partial_products_polys()..
    }

    /// Range of lookup polynomials needed for evaluation at `g * zeta`.
    pub const fn next_lookup_range(&self, i: usize) -> Range<usize> {
        self.num_zs_partial_products_polys() + i * self.num_lookup_polys
            ..self.num_zs_partial_products_polys() + i * self.num_lookup_polys + 2
    }

    pub(crate) fn get_fri_instance(&self, zeta: F::Extension) -> FriInstanceInfo<F, D> {
        // All polynomials are opened at zeta.
        let zeta_batch = FriBatchInfo {
            point: zeta,
            polynomials: self.fri_all_polys(),
        };

        // The Z polynomials are also opened at g * zeta.
        let g = F::Extension::primitive_root_of_unity(self.degree_bits());
        let zeta_next = g * zeta;
        let zeta_next_batch = FriBatchInfo {
            point: zeta_next,
            polynomials: self.fri_next_batch_polys(),
        };

        let openings = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfo {
            oracles: self.fri_oracles(),
            batches: openings,
        }
    }

    pub(crate) fn get_fri_instance_target(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
    ) -> FriInstanceInfoTarget<D> {
        // All polynomials are opened at zeta.
        let zeta_batch = FriBatchInfoTarget {
            point: zeta,
            polynomials: self.fri_all_polys(),
        };

        // The Z polynomials are also opened at g * zeta.
        let g = F::primitive_root_of_unity(self.degree_bits());
        let zeta_next = builder.mul_const_extension(g, zeta);
        let zeta_next_batch = FriBatchInfoTarget {
            point: zeta_next,
            polynomials: self.fri_next_batch_polys(),
        };

        let openings = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfoTarget {
            oracles: self.fri_oracles(),
            batches: openings,
        }
    }

    fn fri_oracles(&self) -> Vec<FriOracleInfo> {
        vec![
            FriOracleInfo {
                num_polys: self.num_preprocessed_polys(),
                blinding: PlonkOracle::CONSTANTS_SIGMAS.blinding,
            },
            FriOracleInfo {
                num_polys: self.config.num_wires,
                blinding: PlonkOracle::WIRES.blinding,
            },
            FriOracleInfo {
                num_polys: self.num_zs_partial_products_polys() + self.num_all_lookup_polys(),
                blinding: PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            },
            FriOracleInfo {
                num_polys: self.num_quotient_polys(),
                blinding: PlonkOracle::QUOTIENT.blinding,
            },
        ]
    }

    fn fri_preprocessed_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(
            PlonkOracle::CONSTANTS_SIGMAS.index,
            0..self.num_preprocessed_polys(),
        )
    }

    pub(crate) const fn num_preprocessed_polys(&self) -> usize {
        self.sigmas_range().end
    }

    fn fri_wire_polys(&self) -> Vec<FriPolynomialInfo> {
        let num_wire_polys = self.config.num_wires;
        FriPolynomialInfo::from_range(PlonkOracle::WIRES.index, 0..num_wire_polys)
    }

    fn fri_zs_partial_products_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(
            PlonkOracle::ZS_PARTIAL_PRODUCTS.index,
            0..self.num_zs_partial_products_polys(),
        )
    }

    pub(crate) const fn num_zs_partial_products_polys(&self) -> usize {
        self.config.num_challenges * (1 + self.num_partial_products)
    }

    /// Returns the total number of lookup polynomials.
    pub(crate) const fn num_all_lookup_polys(&self) -> usize {
        self.config.num_challenges * self.num_lookup_polys
    }
    fn fri_zs_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(PlonkOracle::ZS_PARTIAL_PRODUCTS.index, self.zs_range())
    }

    /// Returns polynomials that require evaluation at `zeta` and `g * zeta`.
    fn fri_next_batch_polys(&self) -> Vec<FriPolynomialInfo> {
        [self.fri_zs_polys(), self.fri_lookup_polys()].concat()
    }

    fn fri_quotient_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(PlonkOracle::QUOTIENT.index, 0..self.num_quotient_polys())
    }

    /// Returns the information for lookup polynomials, i.e. the index within the oracle and the indices of the polynomials within the commitment.
    fn fri_lookup_polys(&self) -> Vec<FriPolynomialInfo> {
        FriPolynomialInfo::from_range(
            PlonkOracle::ZS_PARTIAL_PRODUCTS.index,
            self.num_zs_partial_products_polys()
                ..self.num_zs_partial_products_polys() + self.num_all_lookup_polys(),
        )
    }
    pub(crate) const fn num_quotient_polys(&self) -> usize {
        self.config.num_challenges * self.quotient_degree_factor
    }

    fn fri_all_polys(&self) -> Vec<FriPolynomialInfo> {
        [
            self.fri_preprocessed_polys(),
            self.fri_wire_polys(),
            self.fri_zs_partial_products_polys(),
            self.fri_quotient_polys(),
            self.fri_lookup_polys(),
        ]
        .concat()
    }
}

/// The `Target` version of `VerifierCircuitData`, for use inside recursive circuits. Note that this
/// is intentionally missing certain fields, such as `CircuitConfig`, because we support only a
/// limited form of dynamic inner circuits. We can't practically make things like the wire count
/// dynamic, at least not without setting a maximum wire count and paying for the worst case.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerifierCircuitTarget {
    /// A commitment to each constant polynomial and each permutation polynomial.
    pub constants_sigmas_cap: MerkleCapTarget,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: HashOutTarget,
}

#[cfg(test)]
mod generator_watch_index_tests {
    use std::collections::BTreeMap;

    use super::GeneratorWatchIndex;

    #[test]
    fn sparse_watch_index_preserves_lists_and_empty_representatives() {
        let map = BTreeMap::from([(1usize, vec![2usize, 5]), (4, vec![3])]);
        let index = GeneratorWatchIndex::from_map(map);

        assert_eq!(index.len(), 2);
        assert_eq!(index.get(&0), None);
        assert_eq!(index.get(&1), Some([2u32, 5].as_slice()));
        assert_eq!(index.get(&2), None);
        assert_eq!(index.get(&3), None);
        assert_eq!(index.get(&4), Some([3u32].as_slice()));
        assert_eq!(index.get(&5), None);

        let entries = index
            .iter()
            .map(|(representative, watchers)| (representative, watchers.to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(entries, vec![(1, vec![2, 5]), (4, vec![3])]);
    }

    /// The narrowed `u32` watcher payload must answer every query with exactly
    /// the sequence the `usize` payload answered, for both construction seams
    /// and both readers. The comparison is on raw `u64` widenings of the stored
    /// words, so a truncation or a sign/width mistake cannot hide behind the
    /// slice type. The last block is the sabotage control.
    #[test]
    fn narrow_watch_index_matches_wide_reference_raw_words() {
        // A deterministic pseudo-random edge list: dense and sparse
        // representatives, singleton and long watcher lists, a representative
        // range with holes, and generator indices past every byte boundary.
        fn edge_lists(seed: u64) -> BTreeMap<usize, Vec<usize>> {
            let mut state = seed | 1;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for _ in 0..512 {
                let representative = (next() % 4096) as usize;
                let generator = (next() % 300_000) as usize;
                let list = map.entry(representative).or_default();
                if !list.contains(&generator) {
                    list.push(generator);
                }
            }
            // Boundary generator indices the narrowing must carry intact.
            map.insert(
                4096,
                vec![0, 1, 255, 256, 65_535, 65_536, 16_777_215, 16_777_216],
            );
            for list in map.values_mut() {
                list.sort_unstable();
            }
            map.retain(|_, list| !list.is_empty());
            map
        }

        // Reference reader over the same edge list, in the pre-narrowing width.
        fn reference_get(
            map: &BTreeMap<usize, Vec<usize>>,
            representative: usize,
        ) -> Option<Vec<u64>> {
            map.get(&representative)
                .filter(|list| !list.is_empty())
                .map(|list| list.iter().map(|&g| g as u64).collect())
        }

        let mut compared = 0usize;
        for seed in [1u64, 0xdead_beef, 0x9e37_79b9_7f4a_7c15, 42] {
            let map = edge_lists(seed);
            let max_representative = *map.keys().max().unwrap();

            // Seam 1: `from_map`.
            let from_map = GeneratorWatchIndex::from_map(map.clone());

            // Seam 2: `from_sorted_generator_representatives`, fed the same
            // edge list transposed into per-generator groups (its contract).
            let mut per_generator: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
            for (&representative, watchers) in &map {
                for &generator in watchers {
                    per_generator
                        .entry(generator)
                        .or_default()
                        .push(representative as u32);
                }
            }
            let max_generator = *per_generator.keys().max().unwrap();
            let mut representatives: Vec<u32> = Vec::new();
            let mut counts: Vec<usize> = vec![0usize; max_generator + 1];
            for generator in 0..=max_generator {
                let mut group = per_generator.get(&generator).cloned().unwrap_or_default();
                group.sort_unstable();
                group.dedup();
                counts[generator] = group.len();
                representatives.extend(group);
            }
            let from_groups = GeneratorWatchIndex::from_sorted_generator_representatives(
                &representatives,
                &counts,
            );

            // Seam 3: `from_parts`, i.e. the embedded-blob loader.
            let from_parts = GeneratorWatchIndex::from_parts(
                from_map.offsets().to_vec(),
                from_map.watchers().to_vec(),
            );

            for representative in 0..=(max_representative + 8) {
                let expected = reference_get(&map, representative);
                for (name, index) in [
                    ("from_map", &from_map),
                    ("from_groups", &from_groups),
                    ("from_parts", &from_parts),
                ] {
                    let actual = index
                        .get(&representative)
                        .map(|watchers| watchers.iter().map(|&g| g as u64).collect::<Vec<u64>>());
                    assert_eq!(actual, expected, "{name} diverges at {representative}");
                    compared += 1;
                }
            }

            // `iter` must enumerate the same (representative, list) pairs.
            let expected_entries: Vec<(usize, Vec<u64>)> = map
                .iter()
                .filter(|(_, list)| !list.is_empty())
                .map(|(&r, list)| (r, list.iter().map(|&g| g as u64).collect()))
                .collect();
            for (name, index) in [
                ("from_map", &from_map),
                ("from_groups", &from_groups),
                ("from_parts", &from_parts),
            ] {
                let actual_entries: Vec<(usize, Vec<u64>)> = index
                    .iter()
                    .map(|(r, list)| (r, list.iter().map(|&g| g as u64).collect()))
                    .collect();
                assert_eq!(actual_entries, expected_entries, "{name} iter diverges");
                assert_eq!(index.len(), expected_entries.len(), "{name} len diverges");
            }
        }
        assert!(compared >= 12_000, "differential ran on too few queries");

        // Sabotage control: truncate one watcher to 16 bits in the reference
        // and require the raw-word comparison to see it. A differential that
        // has never failed is not evidence.
        let map = BTreeMap::from([(7usize, vec![65_536usize, 3])]);
        let index = GeneratorWatchIndex::from_map(map);
        let actual: Vec<u64> = index.get(&7).unwrap().iter().map(|&g| g as u64).collect();
        let truncated: Vec<u64> = vec![(65_536u64 as u16) as u64, 3];
        assert_ne!(
            actual, truncated,
            "sabotage control did not trip: the differential cannot detect a truncated watcher"
        );
        assert_eq!(actual, vec![65_536u64, 3]);
    }
}
