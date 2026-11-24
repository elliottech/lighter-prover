//! A Plonky2 circuit that verifies a `p3-schnorr` STARK proof's transcript
//! and FRI opening proof from scratch, for a fixed batch capacity (by
//! default [`FIXED_RECURSIVE_SIGNATURE_COUNT`]); the proof's public
//! signature count may be any value up to that capacity.
//!
//! # What "verifying the transcript" means here
//!
//! A STARK proof is sound only if the verifier's challenges (`alpha` for
//! constraint folding, `zeta` for the out-of-domain sample point, the FRI
//! `beta`s and query indices) were derived from the *actual* commitments and
//! opened values in the proof, via Fiat-Shamir, rather than supplied
//! independently by the prover. This circuit re-runs that whole derivation
//! in-circuit — replaying the same Poseidon2 duplex sponge
//! ([`P3RecursiveDuplexChallenger`]) the native `p3-schnorr` verifier uses —
//! and then re-checks the two things a STARK proof actually claims:
//! 1. **OODS / quotient consistency**: the AIR's constraint polynomials,
//!    folded by `alpha` and evaluated at `zeta` using the proof's opened
//!    `trace_local`/`trace_next`/`preprocessed_*`/`quotient_chunks` values,
//!    equal `quotient(zeta) * Z_H(zeta)` (see
//!    [`derive_folded_constraints_targets`]/[`recompose_quotient_chunks_targets`]).
//! 2. **FRI low-degree proof**: every queried opening is consistent with the
//!    committed Merkle caps, and the commit-phase folding/final polynomial
//!    check out (see `verify_fixed_fri_input_openings_targets` and the
//!    `verify_*fri*_targets` family).
//!
//! # Why "fixed" everywhere
//!
//! Every `Fixed*` type/function name reflects that this circuit's shape
//! (number of FRI queries, commit-phase rounds, quotient chunk domains, trace
//! widths, etc.) is baked in at circuit-build time from
//! `fixed_verifier_metadata`, which recomputes the same parameters
//! `p3-schnorr`'s [`schnorr_stark_config`]/[`SignatureBatchAir`] would use for
//! a full-capacity batch. A Plonky2 circuit's
//! gate count and wiring are static, so there is no "variable-length FRI
//! proof" representation — changing any STARK parameter that affects this
//! shape (FRI query count, blowup, batch size) requires rebuilding the
//! circuit, not just feeding it different witness data.
//!
//! # `_host` vs `_targets` function pairs
//!
//! Most of the verification logic appears twice, following the pattern
//! already used in `poseidon2_gate.rs`:
//! - `*_host` functions compute the same value natively (plain `u64`/P3 field
//!   arithmetic) — used by [`prove_transcript_verifier`] to derive the
//!   witness values (e.g. `alpha`, `zeta`, the recomposed quotient) that get
//!   assigned to circuit targets before proving.
//! - `*_targets` functions perform the equivalent computation by emitting
//!   `CircuitBuilder` gates over `Target`/`ExtensionTarget` — these *are* the
//!   circuit, and what the recursive proof actually attests was computed
//!   correctly.
//!
//! The two must compute identical results.

use core::cell::Cell;

use p3_air::symbolic::{
    AirLayout, BaseEntry, BaseLeaf, ConstraintLayout, ExtEntry, ExtLeaf, SymbolicAirBuilder,
    SymbolicExpr, SymbolicExpression, SymbolicExpressionExt, SymbolicVariableExt,
};
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::PolynomialSpace;
use p3_ecdsa::EcdsaBatchAir;
use p3_ecdsa::batch::PreparedBatch;
use p3_ecdsa::keccak::MessageHashAir;
use p3_ecdsa::prove::PreparedEcdsaBatch;
use p3_ecdsa::stark::LogUpAir;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field as _, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField,
};
use p3_schnorr::proof::{
    Challenge as P3Challenge, Challenger as P3Challenger,
    TRANSCRIPT_POSEIDON_RATE as POSEIDON_RATE, TRANSCRIPT_POSEIDON_WIDTH as POSEIDON_WIDTH,
};
use plonky2::field::extension::{Extendable, FieldExtension as Plonky2FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::Poseidon2GoldilocksConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;

thread_local! {
    static POSEIDON2_CALL_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn reset_poseidon2_counter() {
    POSEIDON2_CALL_COUNT.with(|c| c.set(0));
}
fn read_poseidon2_counter() -> usize {
    POSEIDON2_CALL_COUNT.with(|c| c.get())
}

#[cfg(any(test, feature = "insecure-test-only"))]
use crate::api::FIXED_RECURSIVE_SIGNATURE_COUNT;
use crate::error::RecursionError;
use crate::types::{
    BatchRecursiveWitness, P3MerkleCapWitness, PreparedP3RecursiveWitness, StructuredP3Proof,
};

pub const RECURSION_D: usize = 2;
pub const RECURSIVE_PUBLIC_INPUTS: usize = p3_ecdsa::air::layout::NUM_PUBLIC_VALUES;
pub const MESSAGE_HASH_PUBLIC_INPUTS: usize = p3_ecdsa::keccak::NUM_PUBLIC_VALUES;

pub type F = GoldilocksField;
pub type C = Poseidon2GoldilocksConfig;

/// In-circuit reimplementation of `p3-schnorr`'s `Challenger` (a Poseidon2
/// duplex sponge): observes circuit values and produces challenge targets in
/// the same order and using the same permutation the native STARK verifier
/// would, so the Fiat-Shamir transcript this circuit replays matches the one
/// the proof was actually generated under.
#[derive(Clone, Debug)]
struct P3RecursiveDuplexChallenger {
    sponge_state: [Target; POSEIDON_WIDTH],
    input_buffer: Vec<Target>,
    output_buffer: Vec<Target>,
}

impl P3RecursiveDuplexChallenger {
    fn new(builder: &mut CircuitBuilder<F, RECURSION_D>) -> Self {
        let zero = builder.zero();
        Self {
            sponge_state: [zero; POSEIDON_WIDTH],
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    fn observe_element(&mut self, builder: &mut CircuitBuilder<F, RECURSION_D>, target: Target) {
        self.output_buffer.clear();
        self.input_buffer.push(target);
        if self.input_buffer.len() == POSEIDON_RATE {
            self.duplexing(builder);
        }
    }

    fn observe_elements(
        &mut self,
        builder: &mut CircuitBuilder<F, RECURSION_D>,
        targets: &[Target],
    ) {
        for &target in targets {
            self.observe_element(builder, target);
        }
    }

    fn get_challenge(&mut self, builder: &mut CircuitBuilder<F, RECURSION_D>) -> Target {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing(builder);
        }
        self.output_buffer
            .pop()
            .expect("recursive P3 challenger output buffer must be non-empty")
    }

    fn get_extension_challenge(
        &mut self,
        builder: &mut CircuitBuilder<F, RECURSION_D>,
    ) -> ExtensionTarget<RECURSION_D> {
        ExtensionTarget(core::array::from_fn(|_| self.get_challenge(builder)))
    }

    fn duplexing(&mut self, builder: &mut CircuitBuilder<F, RECURSION_D>) {
        let num_absorbed = self.input_buffer.len();
        assert!(num_absorbed <= POSEIDON_RATE);

        for (i, &value) in self.input_buffer.iter().enumerate() {
            self.sponge_state[i] = value;
        }

        if num_absorbed > 0 {
            let zero = builder.zero();
            for i in num_absorbed..POSEIDON_RATE {
                self.sponge_state[i] = zero;
            }
            let len_tag = builder.constant(F::from_canonical_usize(num_absorbed));
            self.sponge_state[POSEIDON_RATE] =
                builder.add(self.sponge_state[POSEIDON_RATE], len_tag);
        }

        poseidon2_permute_targets(builder, &mut self.sponge_state);
        self.output_buffer.clear();
        self.output_buffer
            .extend_from_slice(&self.sponge_state[..POSEIDON_RATE]);
        self.input_buffer.clear();
    }
}

/// Permutes `state` in place via the [`crate::poseidon2_gate`] custom gate
/// (one gate row per call) and counts the call for
/// [`TranscriptVerifierCircuit::num_poseidon2_calls`], used as a build-time
/// sanity metric on circuit size.
fn poseidon2_permute_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    state: &mut [Target; POSEIDON_WIDTH],
) {
    POSEIDON2_CALL_COUNT.with(|c| c.set(c.get() + 1));
    let outputs = crate::poseidon2_gate::p3_poseidon2_permute_gate(builder, state);
    state.copy_from_slice(&outputs);
}

/// Returns the canonical little-endian bits of a Goldilocks target.
///
/// Plonky2's `split_le(value, 64)` only reconstructs `value` modulo the
/// circuit field. For Goldilocks, whose modulus is
/// `0xffff_ffff_0000_0001`, a value `c < 2^32 - 1` therefore admits both the
/// 64-bit decompositions of `c` and `c + modulus`. Native P3 challenge bit
/// sampling uses `as_canonical_u64()`, so only the canonical one is accepted.
///
/// Among 64-bit integers, the non-canonical interval `[modulus, 2^64)` is
/// exactly the set whose high 32 bits are all one and whose low 32 bits are
/// nonzero. Excluding that pattern makes the decomposition unique.
fn canonical_goldilocks_low_bits(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    value: Target,
    num_low_bits: usize,
) -> Vec<BoolTarget> {
    assert!(num_low_bits <= 64);
    let mut bits = builder.split_le(value, 64);
    constrain_canonical_goldilocks_bits(builder, &bits);
    bits.truncate(num_low_bits);
    bits
}

/// Constrains a 64-bit little-endian integer to be less than the Goldilocks
/// modulus. Kept separate from the decomposition helper so the boundary and
/// formerly ambiguous representations can be tested directly.
fn constrain_canonical_goldilocks_bits(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    bits: &[BoolTarget],
) {
    assert_eq!(bits.len(), 64);

    let mut high_bits_all_one = bits[32].target;
    for bit in &bits[33..] {
        high_bits_all_one = builder.mul(high_bits_all_one, bit.target);
    }
    for bit in &bits[..32] {
        let noncanonical = builder.mul(high_bits_all_one, bit.target);
        builder.assert_zero(noncanonical);
    }
}

// Dead code: `external_full_round_targets` through `apply_mat4_targets` below
// implement the Poseidon2 permutation as ordinary `CircuitBuilder` gates,
// from before `poseidon2_permute_targets` switched to the dedicated
// `P3Poseidon2Gate` (which packs a full permutation into one custom gate row
// far more cheaply). Nothing in this crate calls them; candidates for removal.
fn external_full_round_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    state: &mut [Target; POSEIDON_WIDTH],
    round_constants: &[F; POSEIDON_WIDTH],
) {
    for (value, rc) in state.iter_mut().zip(round_constants.iter()) {
        *value = add_rc_and_sbox_target(builder, *value, *rc);
    }
    mds_light_permutation_targets(builder, state);
}

fn internal_partial_round_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    state: &mut [Target; POSEIDON_WIDTH],
    rc: F,
) {
    state[0] = add_rc_and_sbox_target(builder, state[0], rc);
    internal_linear_layer_targets(builder, state);
}

fn add_rc_and_sbox_target(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    value: Target,
    rc: F,
) -> Target {
    let value_plus_rc = builder.add_const(value, rc);
    exp_7_target(builder, value_plus_rc)
}

fn exp_7_target(builder: &mut CircuitBuilder<F, RECURSION_D>, value: Target) -> Target {
    let x2 = builder.square(value);
    let x4 = builder.square(x2);
    let x6 = builder.mul(x4, x2);
    builder.mul(x6, value)
}

fn internal_linear_layer_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    state: &mut [Target; POSEIDON_WIDTH],
) {
    let sum = state
        .iter()
        .copied()
        .reduce(|acc, value| builder.add(acc, value))
        .unwrap();
    for (i, value) in state.iter_mut().enumerate() {
        let scaled = builder.mul_const(
            goldilocks_from_p3(p3_goldilocks::MATRIX_DIAG_16_GOLDILOCKS[i]),
            *value,
        );
        *value = builder.add(scaled, sum);
    }
}

fn mds_light_permutation_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    state: &mut [Target; POSEIDON_WIDTH],
) {
    for chunk in state.as_chunks_mut::<4>().0 {
        apply_mat4_targets(builder, chunk);
    }

    let sums: [Target; 4] = core::array::from_fn(|k| {
        (0..POSEIDON_WIDTH)
            .step_by(4)
            .map(|j| state[j + k])
            .reduce(|acc, value| builder.add(acc, value))
            .unwrap()
    });

    for (i, elem) in state.iter_mut().enumerate() {
        *elem = builder.add(*elem, sums[i % 4]);
    }
}

fn apply_mat4_targets(builder: &mut CircuitBuilder<F, RECURSION_D>, x: &mut [Target; 4]) {
    let t01 = builder.add(x[0], x[1]);
    let t23 = builder.add(x[2], x[3]);
    let t0123 = builder.add(t01, t23);
    let t01123 = builder.add(t0123, x[1]);
    let t01233 = builder.add(t0123, x[3]);

    let x0_double = builder.add(x[0], x[0]);
    let x2_double = builder.add(x[2], x[2]);
    x[3] = builder.add(t01233, x0_double);
    x[1] = builder.add(t01123, x2_double);
    x[0] = builder.add(t01123, t01);
    x[2] = builder.add(t01233, t23);
}

/// Every shape/size parameter the recursive circuit needs baked in at
/// build time for its full batch capacity —
/// computed once by `fixed_verifier_metadata` from `p3-schnorr`'s own STARK
/// configuration, not chosen independently. See the module docs' "Why
/// fixed everywhere" section.
#[derive(Clone, Debug)]
pub struct FixedVerifierMetadata {
    pub kind: InnerProofKind,
    pub num_public_values: usize,
    pub degree_bits: usize,
    pub base_degree_bits: usize,
    pub signature_capacity: usize,
    pub trace_subgroup_generator_inverse: u64,
    pub air_width: usize,
    pub preprocessed_width: usize,
    /// Flattened (base-coefficient) width of the LogUp permutation trace.
    pub permutation_width: usize,
    pub num_permutation_challenges: usize,
    pub permutation_cap_roots: usize,
    pub trace_next_width: usize,
    pub preprocessed_next_width: usize,
    pub num_quotient_chunks: usize,
    pub quotient_chunk_domains: Vec<FixedQuotientChunkDomain>,
    pub preprocessed_commitment: P3MerkleCapWitness,
    pub trace_cap_roots: usize,
    pub quotient_cap_roots: usize,
    pub fri_cap_roots: usize,
    pub fri_log_blowup: usize,
    pub fri_query_pow_bits: usize,
    pub fri_commit_pow_bits: usize,
    pub fri_query_count: usize,
    pub fri_final_poly_len: usize,
    pub fri_input_batch_widths: Vec<Vec<usize>>,
    pub fri_input_opening_proof_len: usize,
    pub fri_commit_phase_log_arities: Vec<usize>,
    pub fri_commit_phase_opening_proof_lens: Vec<usize>,
    pub fri_final_log_height: usize,
}

/// Which inner STARK a [`FixedVerifierMetadata`] describes: the ECDSA AIR
/// (`p3-ecdsa`'s LogUp stark, with preprocessed and permutation traces) or the
/// message-hash (Keccak) AIR (plain `p3_uni_stark`, main trace only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerProofKind {
    EcdsaLogUp,
    MessageHash,
}

/// Precomputed constants for evaluating one quotient chunk's vanishing
/// polynomial factor at `zeta`, needed by [`recompose_quotient_chunks_targets`].
#[derive(Clone, Debug)]
pub struct FixedQuotientChunkDomain {
    pub log_size: usize,
    pub shift_inverse: u64,
    pub other_vanishing_inverse: Vec<u64>,
}

/// In-circuit counterpart of [`crate::types::P3MerkleCapWitness`]: a Merkle
/// cap as `Target`s (8-element digests) rather than `u64`s.
#[derive(Clone, Debug)]
pub struct MerkleCapTargets8 {
    pub roots: Vec<[Target; 8]>,
}

/// Every circuit input the recursive verifier needs assigned per proof:
/// the commitments, opened values, and the prover-claimed `alpha`/`zeta`/
/// quotient (`expected_alpha`/`expected_zeta`/`expected_quotient`) that the circuit derives independently
/// and connects back to these via equality constraints (see
/// [`build_transcript_verifier_circuit`]).
#[derive(Clone, Debug)]
pub struct TranscriptVerifierTargets {
    pub degree_bits: Target,
    pub trace_commitment: MerkleCapTargets8,
    pub quotient_chunks_commitment: MerkleCapTargets8,
    pub permutation_commitment: MerkleCapTargets8,
    pub permutation_local: Vec<ExtensionTarget<RECURSION_D>>,
    pub permutation_next: Vec<ExtensionTarget<RECURSION_D>>,
    pub trace_local: Vec<ExtensionTarget<RECURSION_D>>,
    pub trace_next: Vec<ExtensionTarget<RECURSION_D>>,
    pub preprocessed_local: Vec<ExtensionTarget<RECURSION_D>>,
    pub preprocessed_next: Vec<ExtensionTarget<RECURSION_D>>,
    pub quotient_chunks: Vec<Vec<ExtensionTarget<RECURSION_D>>>,
    pub expected_quotient: ExtensionTarget<RECURSION_D>,
    pub expected_alpha: ExtensionTarget<RECURSION_D>,
    pub expected_zeta: ExtensionTarget<RECURSION_D>,
    pub fri: FixedFriProofTargets,
}

/// In-circuit counterpart of [`crate::types::P3FriProofWitness`], shaped by
/// [`FixedVerifierMetadata`] (fixed number of commit-phase rounds/queries).
#[derive(Clone, Debug)]
pub struct FixedFriProofTargets {
    pub commit_phase_commits: Vec<MerkleCapTargets8>,
    pub commit_pow_witnesses: Vec<Target>,
    pub query_proofs: Vec<FixedFriQueryTargets>,
    pub final_poly: Vec<ExtensionTarget<RECURSION_D>>,
    pub query_pow_witness: Target,
}

/// In-circuit counterpart of [`crate::types::P3FriQueryWitness`].
#[derive(Clone, Debug)]
pub struct FixedFriQueryTargets {
    pub input_proof: Vec<FixedBatchOpeningTargets>,
    pub commit_phase_openings: Vec<FixedCommitPhaseOpeningTargets>,
}

/// In-circuit counterpart of [`crate::types::P3BatchOpeningWitness`].
#[derive(Clone, Debug)]
pub struct FixedBatchOpeningTargets {
    pub opened_values: Vec<Vec<Target>>,
    pub opening_proof: Vec<[Target; 8]>,
}

/// In-circuit counterpart of [`crate::types::P3CommitPhaseProofStepWitness`].
#[derive(Clone, Debug)]
pub struct FixedCommitPhaseOpeningTargets {
    pub log_arity: Target,
    pub sibling_values: Vec<ExtensionTarget<RECURSION_D>>,
    pub opening_proof: Vec<[Target; 8]>,
}

/// `SignatureBatchAir`'s constraints, captured symbolically (as expression
/// trees over trace/public-input variables) rather than evaluated — built
/// once via `p3_air::symbolic` and then "played back" as circuit gates by
/// [`resolve_base_expression_targets`]/[`resolve_ext_expression_targets`],
/// so this circuit's constraint-folding logic doesn't need to be
/// hand-written separately from the AIR it's verifying.
#[derive(Debug)]
struct SymbolicAirConstraints {
    base: Vec<SymbolicExpression<p3_goldilocks::Goldilocks>>,
    ext: Vec<SymbolicExpressionExt<p3_goldilocks::Goldilocks, P3Challenge>>,
    layout: ConstraintLayout,
}

/// The three Lagrange selector values (`is_first_row`, `is_last_row`,
/// `is_transition`) evaluated at `zeta`, needed to weight per-row-type
/// constraints when folding the AIR's symbolic constraints at a single
/// out-of-domain point.
#[derive(Clone, Copy, Debug)]
struct LagrangeSelectorTargets {
    is_first_row: ExtensionTarget<RECURSION_D>,
    is_last_row: ExtensionTarget<RECURSION_D>,
    is_transition: ExtensionTarget<RECURSION_D>,
}

/// Bundles everything [`resolve_base_expression_targets`]/
/// [`resolve_ext_expression_targets`] need to evaluate one symbolic AIR
/// expression at `zeta`: the opened trace/preprocessed values and selectors.
#[derive(Clone, Copy)]
struct SymbolicEvaluationContext<'a> {
    trace_local: &'a [ExtensionTarget<RECURSION_D>],
    trace_next: &'a [ExtensionTarget<RECURSION_D>],
    preprocessed_local: &'a [ExtensionTarget<RECURSION_D>],
    preprocessed_next: &'a [ExtensionTarget<RECURSION_D>],
    permutation_local: &'a [ExtensionTarget<RECURSION_D>],
    permutation_next: &'a [ExtensionTarget<RECURSION_D>],
    permutation_challenges: &'a [ExtensionTarget<RECURSION_D>],
    public_inputs: &'a [Target],
    selectors: LagrangeSelectorTargets,
}

/// An insecure circuit that only registers `RECURSIVE_PUBLIC_INPUTS` public
/// inputs and proves nothing about them.
///
/// This exists solely as a test/benchmark baseline for measuring the overhead
/// of the real verifier logic in [`TranscriptVerifierCircuit`]. A proof from
/// this circuit does **not** attest to a Schnorr signature or a P3 proof and
/// must never be accepted by a production verifier.
#[cfg(any(test, feature = "insecure-test-only"))]
pub struct InsecurePublicInputShellCircuit {
    pub data: CircuitData<F, C, RECURSION_D>,
    pub public_inputs: Vec<Target>,
}

/// The real recursive verifier circuit: proves the transcript/FRI checks
/// described in the module docs, built once via
/// [`build_transcript_verifier_circuit`] for a fixed `signature_count`
/// capacity and reused (via [`prove_transcript_verifier`]) for every batch
/// whose proof was generated at that capacity's trace height, with any real
/// signature count in `1..=signature_count`. Different call sites may each
/// build their own instance for a different capacity — a Plonky2 circuit's
/// shape is static, but nothing ties every instance to the same capacity.
pub struct TranscriptVerifierCircuit {
    pub data: CircuitData<F, C, RECURSION_D>,
    pub public_inputs: Vec<Target>,
    pub proof: TranscriptVerifierTargets,
    pub metadata: FixedVerifierMetadata,
    /// The signature capacity this circuit instance was built for; the
    /// proof's public count may be any value in `1..=signature_count`.
    pub signature_count: usize,
    /// Number of gate instances before zero-knowledge blinding and power-of-two padding.
    pub num_gates_before_padding: usize,
    /// Number of Poseidon2-16 permutation calls emitted during circuit build.
    pub num_poseidon2_calls: usize,
}

#[cfg(any(test, feature = "insecure-test-only"))]
impl InsecurePublicInputShellCircuit {
    pub fn expected_signature_count(&self) -> usize {
        FIXED_RECURSIVE_SIGNATURE_COUNT
    }
}

impl TranscriptVerifierCircuit {
    pub fn expected_signature_count(&self) -> usize {
        self.signature_count
    }
}

/// Builds the deliberately unconstrained
/// [`InsecurePublicInputShellCircuit`] test/benchmark baseline.
#[cfg(any(test, feature = "insecure-test-only"))]
pub fn build_insecure_public_input_shell_circuit() -> InsecurePublicInputShellCircuit {
    let mut builder =
        CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
    let public_inputs = builder.add_virtual_targets(RECURSIVE_PUBLIC_INPUTS);
    for &target in &public_inputs {
        builder.register_public_input(target);
    }
    let data = builder.build::<C>();
    InsecurePublicInputShellCircuit {
        data,
        public_inputs,
    }
}

/// Builds the [`TranscriptVerifierCircuit`]: allocates every proof-shaped
/// target ([`TranscriptVerifierTargets`]), re-derives `alpha`/`zeta` and the
/// recomposed quotient/folded-constraint values from those targets, and
/// wires up the equality constraints that make this circuit actually check
/// something rather than just exist:
/// - `proof.degree_bits` must equal the fixed `metadata.degree_bits`.
/// - The derived quotient/`alpha`/`zeta` must equal the values the prover
///   claimed (`proof.expected_*`) — these "expected" targets exist so
///   [`prove_transcript_verifier`] can assign them directly from
///   `*_host`-computed values, and this circuit double-checks the assignment
///   was honest.
/// - `quotient(zeta) * Z_H(zeta)` must equal the folded AIR constraints
///   evaluated at `zeta` — the core OODS/quotient-consistency check.
/// - `verify_fixed_fri_input_openings_targets` additionally checks every
///   FRI query's Merkle openings and commit-phase folding.
///
/// Widens `CircuitConfig::num_wires` to fit [`crate::poseidon2_gate::P3Poseidon2Gate`]
/// and enables the addition/selection gate optimizations Plonky2 offers, both
/// purely for circuit size/performance — neither affects what's proved.
pub fn build_transcript_verifier_circuit(
    signature_count: usize,
) -> Result<TranscriptVerifierCircuit, RecursionError> {
    reset_poseidon2_counter();
    let metadata = fixed_verifier_metadata(signature_count)?;
    let symbolic_constraints = symbolic_constraints_for_metadata(&metadata);
    let mut config = CircuitConfig::standard_recursion_config();
    // Enable specialized gates that pack additions and selects more densely than
    // the generic ArithmeticGate.
    config.optimization_flags |= (1 << 0) // AdditionGate: ~26 adds/row vs 4
        | (1 << 5); // SelectionGate: ~20 selects/row vs 2
    // Widen the trace to accommodate P3Poseidon2Gate (166 wires). Non-routed
    // wires beyond 80 are used only as witness scratch space for sbox inputs;
    // they don't affect routing or the cost of other gate types.
    config.num_wires = crate::poseidon2_gate::NUM_WIRES;
    let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);

    let public_inputs = builder.add_virtual_targets(RECURSIVE_PUBLIC_INPUTS);
    for &target in &public_inputs {
        builder.register_public_input(target);
    }
    let one = builder.one();
    let count_minus_one = builder.sub(public_inputs[0], one);
    builder.range_check(count_minus_one, 16);
    let capacity_target = builder.constant(F::from_canonical_usize(signature_count));
    let capacity_slack = builder.sub(capacity_target, public_inputs[0]);
    builder.range_check(capacity_slack, 16);

    let proof = add_inner_verifier(
        &mut builder,
        &metadata,
        &symbolic_constraints,
        &public_inputs,
    )?;

    let num_gates_before_padding = builder.num_gates();
    let num_poseidon2_calls = read_poseidon2_counter();
    let data = builder.build::<C>();
    Ok(TranscriptVerifierCircuit {
        data,
        public_inputs,
        proof,
        metadata,
        signature_count,
        num_gates_before_padding,
        num_poseidon2_calls,
    })
}

/// Adds one complete inner-STARK verifier (transcript, OODS check, FRI) for
/// `metadata` to `builder`, with `public_inputs` as that proof's public values.
fn add_inner_verifier(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    metadata: &FixedVerifierMetadata,
    symbolic_constraints: &SymbolicAirConstraints,
    public_inputs: &[Target],
) -> Result<TranscriptVerifierTargets, RecursionError> {
    let proof = TranscriptVerifierTargets {
        degree_bits: builder.add_virtual_target(),
        trace_commitment: add_virtual_merkle_cap_8(builder, metadata.trace_cap_roots),
        quotient_chunks_commitment: add_virtual_merkle_cap_8(builder, metadata.quotient_cap_roots),
        permutation_commitment: add_virtual_merkle_cap_8(builder, metadata.permutation_cap_roots),
        permutation_local: builder.add_virtual_extension_targets(metadata.permutation_width),
        permutation_next: builder.add_virtual_extension_targets(metadata.permutation_width),
        trace_local: builder.add_virtual_extension_targets(metadata.air_width),
        trace_next: builder.add_virtual_extension_targets(metadata.trace_next_width),
        preprocessed_local: builder.add_virtual_extension_targets(metadata.preprocessed_width),
        preprocessed_next: builder.add_virtual_extension_targets(metadata.preprocessed_next_width),
        quotient_chunks: (0..metadata.num_quotient_chunks)
            .map(|_| builder.add_virtual_extension_targets(RECURSION_D))
            .collect(),
        expected_quotient: builder.add_virtual_extension_target(),
        expected_alpha: builder.add_virtual_extension_target(),
        expected_zeta: builder.add_virtual_extension_target(),
        fri: add_fixed_fri_targets(builder, metadata),
    };

    let expected_degree_bits = builder.constant(F::from_canonical_usize(metadata.degree_bits));
    builder.connect(proof.degree_bits, expected_degree_bits);

    let (transcript_challenger, permutation_challenges, alpha, zeta) =
        derive_transcript_challenges_targets(builder, public_inputs, &proof, metadata);
    let derived_quotient =
        recompose_quotient_chunks_targets(builder, &proof.quotient_chunks, zeta, metadata);
    let vanishing = z_h_at_zeta_targets(builder, zeta, metadata.base_degree_bits);
    let derived_folded = derive_folded_constraints_targets(
        builder,
        public_inputs,
        &proof,
        &permutation_challenges,
        alpha,
        zeta,
        metadata,
        symbolic_constraints,
    )?;
    let quotient_times_vanishing = builder.mul_extension(derived_quotient, vanishing);
    builder.connect_extension(derived_quotient, proof.expected_quotient);
    builder.connect_extension(quotient_times_vanishing, derived_folded);
    builder.connect_extension(alpha, proof.expected_alpha);
    builder.connect_extension(zeta, proof.expected_zeta);
    verify_fixed_fri_input_openings_targets(
        builder,
        transcript_challenger,
        zeta,
        &proof,
        metadata,
    )?;
    Ok(proof)
}

/// Produces a deliberately insecure proof from the
/// [`InsecurePublicInputShellCircuit`] test/benchmark baseline.
///
/// This only checks the public signature count; it does not verify signatures
/// or an underlying P3 proof.
#[cfg(any(test, feature = "insecure-test-only"))]
pub fn prove_insecure_public_input_shell(
    circuit: &InsecurePublicInputShellCircuit,
    public_inputs: &[u64],
) -> Result<ProofWithPublicInputs<F, C, RECURSION_D>, RecursionError> {
    let values: Vec<p3_goldilocks::Goldilocks> = public_inputs
        .iter()
        .map(|&v| p3_goldilocks::Goldilocks::from_u64(v))
        .collect();
    let mut witness = PartialWitness::new();
    for (target, value) in circuit.public_inputs.iter().zip(values) {
        witness
            .set_target(*target, goldilocks_from_p3(value))
            .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    }

    circuit
        .data
        .prove(witness)
        .map_err(|err| RecursionError::Plonky2Proof(err.to_string()))
}

/// Proves the [`TranscriptVerifierCircuit`] over `witness_data`: derives
/// `alpha`/`zeta`/the recomposed quotient natively (`*_host`), assigns every
/// circuit target from `witness_data.structured_proof()` via
/// `set_transcript_verifier_targets`, then runs the Plonky2 prover. The
/// resulting proof attests that `witness_data`'s underlying `p3-schnorr`
/// proof passes the transcript/FRI checks described in the module docs.
///
/// Only `witness_data.public_inputs()`/`structured_proof()` feed the circuit
/// — `proof_bytes()` is never read here. So before proving anything, this
/// calls `crate::api::verify_prepared_recursive_witness` to confirm
/// `proof_bytes` actually decodes to the same `public_inputs`/`structured_proof`
/// (and that the underlying STARK proof itself verifies); without that check
/// a witness built directly via [`PreparedP3RecursiveWitness::new`] with an
/// unrelated `proof_bytes` would still produce a successful recursive proof,
/// and a caller trusting that proof to authenticate `proof_bytes` would be
/// wrong to do so.
pub fn prove_transcript_verifier(
    circuit: &TranscriptVerifierCircuit,
    prepared: &PreparedEcdsaBatch,
    witness_data: &PreparedP3RecursiveWitness,
) -> Result<ProofWithPublicInputs<F, C, RECURSION_D>, RecursionError> {
    if prepared.capacity() != circuit.expected_signature_count() {
        return Err(RecursionError::InvariantViolation(
            "prepared batch capacity does not match recursive circuit",
        ));
    }
    crate::api::verify_prepared_recursive_witness(prepared, witness_data)?;

    let mut witness = PartialWitness::new();
    set_public_inputs(
        &mut witness,
        &circuit.public_inputs,
        witness_data.public_inputs(),
    )?;
    assign_inner_verifier(
        &mut witness,
        &circuit.proof,
        &circuit.metadata,
        witness_data,
    )?;
    circuit
        .data
        .prove(witness)
        .map_err(|err| RecursionError::Plonky2Proof(err.to_string()))
}

fn set_public_inputs(
    witness: &mut PartialWitness<F>,
    targets: &[Target],
    values: &[u64],
) -> Result<(), RecursionError> {
    for (target, value) in targets.iter().zip(values) {
        witness
            .set_target(*target, F::from_canonical_u64(*value))
            .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    }
    Ok(())
}

/// Assigns one inner verifier's targets: derives `alpha`/`zeta`/the quotient
/// natively and loads the structured proof.
fn assign_inner_verifier(
    witness: &mut PartialWitness<F>,
    targets: &TranscriptVerifierTargets,
    metadata: &FixedVerifierMetadata,
    witness_data: &PreparedP3RecursiveWitness,
) -> Result<(), RecursionError> {
    let (alpha, zeta) = derive_transcript_challenges_host(witness_data, metadata)?;
    let quotient = recompose_quotient_chunks_host(witness_data.structured_proof(), zeta, metadata)?;
    set_transcript_verifier_targets(
        witness,
        targets,
        metadata,
        witness_data.structured_proof(),
        quotient,
        alpha,
        zeta,
    )
}

/// One Plonky2 circuit that verifies both halves of a [`p3_ecdsa::BatchProof`]:
/// the ECDSA LogUp proof against public inputs `[n, D0..D3, L0..L3]` and the
/// message-hash (Keccak) proof against `[n, L0..L3]` taken from the same
/// targets, so the link between the two proofs is enforced in-circuit.
pub struct BatchVerifierCircuit {
    pub data: CircuitData<F, C, RECURSION_D>,
    pub public_inputs: Vec<Target>,
    pub ecdsa: TranscriptVerifierTargets,
    pub message_hash: TranscriptVerifierTargets,
    pub ecdsa_metadata: FixedVerifierMetadata,
    pub message_hash_metadata: FixedVerifierMetadata,
    pub signature_count: usize,
    pub num_gates_before_padding: usize,
    /// Gates emitted for the message-hash verifier alone.
    pub message_hash_gates: usize,
    pub num_poseidon2_calls: usize,
}

pub fn build_batch_verifier_circuit(
    signature_count: usize,
) -> Result<BatchVerifierCircuit, RecursionError> {
    reset_poseidon2_counter();
    let ecdsa_metadata = fixed_metadata(InnerProofKind::EcdsaLogUp, signature_count)?;
    let message_hash_metadata = fixed_metadata(InnerProofKind::MessageHash, signature_count)?;
    let ecdsa_constraints = symbolic_constraints_for_metadata(&ecdsa_metadata);
    let message_hash_constraints = symbolic_constraints_for_metadata(&message_hash_metadata);
    let mut config = CircuitConfig::standard_recursion_config();
    config.optimization_flags |= (1 << 0) | (1 << 5);
    config.num_wires = crate::poseidon2_gate::NUM_WIRES;
    let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);

    let public_inputs = builder.add_virtual_targets(RECURSIVE_PUBLIC_INPUTS);
    for &target in &public_inputs {
        builder.register_public_input(target);
    }
    let one = builder.one();
    let count_minus_one = builder.sub(public_inputs[0], one);
    builder.range_check(count_minus_one, 16);
    let capacity_target = builder.constant(F::from_canonical_usize(signature_count));
    let capacity_slack = builder.sub(capacity_target, public_inputs[0]);
    builder.range_check(capacity_slack, 16);

    let ecdsa = add_inner_verifier(
        &mut builder,
        &ecdsa_metadata,
        &ecdsa_constraints,
        &public_inputs,
    )?;
    let link_inputs: Vec<Target> = core::iter::once(public_inputs[0])
        .chain(
            public_inputs[1 + p3_ecdsa::digest::PUBLIC_DIGEST_WIDTH..]
                .iter()
                .copied(),
        )
        .collect();
    debug_assert_eq!(link_inputs.len(), MESSAGE_HASH_PUBLIC_INPUTS);
    let before = builder.num_gates();
    let message_hash = add_inner_verifier(
        &mut builder,
        &message_hash_metadata,
        &message_hash_constraints,
        &link_inputs,
    )?;
    let message_hash_gates = builder.num_gates() - before;

    let num_gates_before_padding = builder.num_gates();
    let num_poseidon2_calls = read_poseidon2_counter();
    let data = builder.build::<C>();
    Ok(BatchVerifierCircuit {
        data,
        public_inputs,
        ecdsa,
        message_hash,
        ecdsa_metadata,
        message_hash_metadata,
        signature_count,
        num_gates_before_padding,
        message_hash_gates,
        num_poseidon2_calls,
    })
}

/// Proves the [`BatchVerifierCircuit`] after natively re-verifying both
/// proofs in `witness_data` (see [`crate::verify_prepared_batch_witness`]).
pub fn prove_batch_verifier(
    circuit: &BatchVerifierCircuit,
    prepared: &PreparedBatch,
    witness_data: &BatchRecursiveWitness,
) -> Result<ProofWithPublicInputs<F, C, RECURSION_D>, RecursionError> {
    if prepared.capacity() != circuit.signature_count {
        return Err(RecursionError::InvariantViolation(
            "prepared batch capacity does not match recursive circuit",
        ));
    }
    crate::api::verify_prepared_batch_witness(prepared, witness_data)?;

    let mut witness = PartialWitness::new();
    set_public_inputs(
        &mut witness,
        &circuit.public_inputs,
        witness_data.ecdsa.public_inputs(),
    )?;
    assign_inner_verifier(
        &mut witness,
        &circuit.ecdsa,
        &circuit.ecdsa_metadata,
        &witness_data.ecdsa,
    )?;
    assign_inner_verifier(
        &mut witness,
        &circuit.message_hash,
        &circuit.message_hash_metadata,
        &witness_data.message_hash,
    )?;
    circuit
        .data
        .prove(witness)
        .map_err(|err| RecursionError::Plonky2Proof(err.to_string()))
}

/// Derives [`FixedVerifierMetadata`] by instantiating `p3-schnorr`'s own
/// `SignatureBatchAir`/STARK config for a full `signature_count`-capacity
/// batch and reading off every shape parameter (trace/preprocessed
/// widths, quotient chunk count and domains, FRI round/query counts) the
/// circuit needs. This is how the circuit's fixed shape stays correct
/// without duplicating `p3-schnorr`'s sizing logic by hand.
fn fixed_verifier_metadata(
    signature_count: usize,
) -> Result<FixedVerifierMetadata, RecursionError> {
    fixed_metadata(InnerProofKind::EcdsaLogUp, signature_count)
}

fn fixed_metadata(
    kind: InnerProofKind,
    signature_count: usize,
) -> Result<FixedVerifierMetadata, RecursionError> {
    type G = p3_goldilocks::Goldilocks;
    let (
        degree_bits,
        air_width,
        preprocessed_width,
        permutation_width,
        num_permutation_challenges,
        log_num_quotient_chunks,
        preprocessed_commitment,
        num_public_values,
    ) = match kind {
        InnerProofKind::EcdsaLogUp => {
            let prepared = PreparedEcdsaBatch::new(signature_count);
            let air = &prepared.air;
            let constraint_degree = p3_ecdsa::air::layout::MAX_CONSTRAINT_DEGREE.max(2);
            (
                air.log_height(),
                <EcdsaBatchAir as BaseAir<G>>::width(air),
                <EcdsaBatchAir as BaseAir<G>>::preprocessed_width(air),
                <EcdsaBatchAir as LogUpAir<G, P3Challenge>>::permutation_width(air) * RECURSION_D,
                <EcdsaBatchAir as LogUpAir<G, P3Challenge>>::num_permutation_challenges(air),
                (constraint_degree - 1).next_power_of_two().ilog2() as usize,
                merkle_cap_from_p3(&prepared.verifier_key.commitment),
                RECURSIVE_PUBLIC_INPUTS,
            )
        }
        InnerProofKind::MessageHash => {
            let air = MessageHashAir::new(signature_count);
            let width = <MessageHashAir as BaseAir<G>>::width(&air);
            let layout = AirLayout {
                preprocessed_width: 0,
                main_width: width,
                num_public_values: MESSAGE_HASH_PUBLIC_INPUTS,
                permutation_width: 0,
                num_permutation_challenges: 0,
                num_permutation_values: 0,
                num_periodic_columns: 0,
            };
            (
                air.height().trailing_zeros() as usize,
                width,
                0,
                0,
                0,
                p3_uni_stark::get_log_num_quotient_chunks::<G, _>(&air, layout, 0),
                P3MerkleCapWitness { roots: Vec::new() },
                MESSAGE_HASH_PUBLIC_INPUTS,
            )
        }
    };
    let trace_next_width = air_width;
    let preprocessed_next_width = preprocessed_width;
    let num_quotient_chunks = 1usize << log_num_quotient_chunks;
    let trace_domain = TwoAdicMultiplicativeCoset::new(p3_goldilocks::Goldilocks::ONE, degree_bits)
        .expect("fixed recursion trace domain must exist");
    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_num_quotient_chunks));
    let quotient_chunk_domains = quotient_domain.split_domains(num_quotient_chunks);
    let fixed_chunk_domains = quotient_chunk_domains
        .iter()
        .enumerate()
        .map(|(i, domain_i)| FixedQuotientChunkDomain {
            log_size: domain_i.log_size(),
            shift_inverse: domain_i.shift_inverse().as_canonical_u64(),
            other_vanishing_inverse: quotient_chunk_domains
                .iter()
                .enumerate()
                .map(|(j, domain_j)| {
                    if i == j {
                        0
                    } else {
                        domain_j
                            .vanishing_poly_at_point(domain_i.first_point())
                            .inverse()
                            .as_canonical_u64()
                    }
                })
                .collect(),
        })
        .collect();
    let fri_global_log_height = degree_bits + p3_ecdsa::prove::ECDSA_FRI_RATE_BITS;
    let fri_final_log_height =
        p3_ecdsa::prove::ECDSA_FRI_RATE_BITS + p3_schnorr::proof::PLONKY2_FRI_FINAL_POLY_BITS;
    let mut current_log_height = fri_global_log_height;
    let mut fri_commit_phase_log_arities = Vec::new();
    let mut fri_commit_phase_opening_proof_lens = Vec::new();
    while current_log_height > fri_final_log_height {
        let log_arity = (current_log_height - fri_final_log_height)
            .min(p3_schnorr::proof::PLONKY2_FRI_REDUCTION_ARITY_BITS);
        current_log_height -= log_arity;
        fri_commit_phase_log_arities.push(log_arity);
        fri_commit_phase_opening_proof_lens
            .push(current_log_height.saturating_sub(p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT));
    }

    let cap_roots = 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT;
    let mut fri_input_batch_widths = vec![vec![air_width]];
    if permutation_width > 0 {
        fri_input_batch_widths.push(vec![permutation_width]);
    }
    fri_input_batch_widths.push(vec![RECURSION_D; num_quotient_chunks]);
    if preprocessed_width > 0 {
        fri_input_batch_widths.push(vec![preprocessed_width]);
    }

    Ok(FixedVerifierMetadata {
        kind,
        num_public_values,
        degree_bits,
        base_degree_bits: degree_bits,
        trace_subgroup_generator_inverse: trace_domain
            .subgroup_generator()
            .inverse()
            .as_canonical_u64(),
        signature_capacity: signature_count,
        air_width,
        preprocessed_width,
        permutation_width,
        num_permutation_challenges,
        permutation_cap_roots: if permutation_width > 0 { cap_roots } else { 0 },
        trace_next_width,
        preprocessed_next_width,
        num_quotient_chunks,
        quotient_chunk_domains: fixed_chunk_domains,
        preprocessed_commitment,
        trace_cap_roots: 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        quotient_cap_roots: 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        fri_cap_roots: 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        fri_log_blowup: p3_ecdsa::prove::ECDSA_FRI_RATE_BITS,
        fri_query_pow_bits: p3_schnorr::proof::PLONKY2_FRI_PROOF_OF_WORK_BITS,
        fri_commit_pow_bits: 0,
        fri_query_count: p3_ecdsa::prove::ECDSA_FRI_NUM_QUERY_ROUNDS,
        fri_final_poly_len: 1
            << (p3_ecdsa::prove::ECDSA_FRI_RATE_BITS
                + p3_schnorr::proof::PLONKY2_FRI_FINAL_POLY_BITS
                - p3_ecdsa::prove::ECDSA_FRI_RATE_BITS),
        fri_input_batch_widths,
        fri_input_opening_proof_len: degree_bits + p3_ecdsa::prove::ECDSA_FRI_RATE_BITS
            - p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        fri_commit_phase_log_arities,
        fri_commit_phase_opening_proof_lens,
        fri_final_log_height,
    })
}

/// Allocates virtual targets for an entire [`FixedFriProofTargets`], shaped
/// by `metadata` (number of commit-phase rounds, queries, per-query batch
/// widths and proof lengths) — these are assigned real values later by
/// [`set_fixed_fri_targets`].
fn add_fixed_fri_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    metadata: &FixedVerifierMetadata,
) -> FixedFriProofTargets {
    FixedFriProofTargets {
        commit_phase_commits: (0..metadata.fri_commit_phase_log_arities.len())
            .map(|_| add_virtual_merkle_cap_8(builder, metadata.fri_cap_roots))
            .collect(),
        commit_pow_witnesses: (0..metadata.fri_commit_phase_log_arities.len())
            .map(|_| builder.add_virtual_target())
            .collect(),
        query_proofs: (0..metadata.fri_query_count)
            .map(|_| FixedFriQueryTargets {
                input_proof: metadata
                    .fri_input_batch_widths
                    .iter()
                    .map(|widths| FixedBatchOpeningTargets {
                        opened_values: widths
                            .iter()
                            .map(|&width| builder.add_virtual_targets(width))
                            .collect(),
                        opening_proof: (0..metadata.fri_input_opening_proof_len)
                            .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                            .collect(),
                    })
                    .collect(),
                commit_phase_openings: metadata
                    .fri_commit_phase_log_arities
                    .iter()
                    .zip(metadata.fri_commit_phase_opening_proof_lens.iter())
                    .map(|(&log_arity, &proof_len)| FixedCommitPhaseOpeningTargets {
                        log_arity: builder.add_virtual_target(),
                        sibling_values: builder
                            .add_virtual_extension_targets((1usize << log_arity) - 1),
                        opening_proof: (0..proof_len)
                            .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
        final_poly: builder.add_virtual_extension_targets(metadata.fri_final_poly_len),
        query_pow_witness: builder.add_virtual_target(),
    }
}

/// The in-circuit counterpart of [`derive_transcript_challenges_host`] (the
/// pair actually used by [`build_transcript_verifier_circuit`]/
/// [`prove_transcript_verifier`]): replays the same Poseidon2 duplex
/// transcript via [`P3RecursiveDuplexChallenger`] to derive `alpha` then
/// `zeta`. Must observe values in exactly the same order as the host
/// version and as `p3-schnorr`'s own native verifier, or the circuit would
/// derive different challenges than the proof was generated under.
///
/// Returns the challenger itself (in addition to `alpha`/`zeta`) so the
/// caller can keep replaying the *same* transcript (continuing from exactly
/// this sponge state) into `verify_fixed_fri_input_openings_targets` instead
/// of starting a fresh challenger and re-deriving `alpha`/`zeta` a second
/// time at twice the Poseidon2 cost.
fn derive_transcript_challenges_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    public_inputs: &[Target],
    proof: &TranscriptVerifierTargets,
    metadata: &FixedVerifierMetadata,
) -> (
    P3RecursiveDuplexChallenger,
    Vec<ExtensionTarget<RECURSION_D>>,
    ExtensionTarget<RECURSION_D>,
    ExtensionTarget<RECURSION_D>,
) {
    let mut challenger = P3RecursiveDuplexChallenger::new(builder);
    let preprocessed_width = builder.constant(F::from_canonical_usize(metadata.preprocessed_width));
    challenger.observe_element(builder, proof.degree_bits);
    if metadata.kind == InnerProofKind::MessageHash {
        // uni-stark also observes `base_degree_bits` (= `degree_bits` without ZK).
        challenger.observe_element(builder, proof.degree_bits);
    }
    challenger.observe_element(builder, preprocessed_width);
    observe_merkle_cap_8(&mut challenger, builder, &proof.trace_commitment);
    if metadata.preprocessed_width > 0 {
        observe_merkle_cap_8_constant(&mut challenger, builder, &metadata.preprocessed_commitment);
    }
    challenger.observe_elements(builder, public_inputs);
    let permutation_challenges = (0..metadata.num_permutation_challenges)
        .map(|_| challenger.get_extension_challenge(builder))
        .collect();
    if metadata.permutation_width > 0 {
        observe_merkle_cap_8(&mut challenger, builder, &proof.permutation_commitment);
    }
    let alpha = challenger.get_extension_challenge(builder);
    observe_merkle_cap_8(&mut challenger, builder, &proof.quotient_chunks_commitment);
    let zeta = challenger.get_extension_challenge(builder);
    (challenger, permutation_challenges, alpha, zeta)
}

/// Native counterpart of [`derive_transcript_challenges_targets`]; used by
/// [`prove_transcript_verifier`] to compute the `alpha`/`zeta` values
/// assigned to the circuit's `expected_alpha`/`expected_zeta` targets.
fn derive_transcript_challenges_host(
    witness: &PreparedP3RecursiveWitness,
    metadata: &FixedVerifierMetadata,
) -> Result<(P3Challenge, P3Challenge), RecursionError> {
    let public_inputs = witness.public_inputs();
    let structured = witness.structured_proof();
    let mut challenger = P3Challenger::new(p3_goldilocks::default_goldilocks_poseidon2_16());
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        structured.degree_bits,
    ));
    if metadata.kind == InnerProofKind::MessageHash {
        challenger.observe(p3_goldilocks::Goldilocks::from_usize(
            structured.degree_bits,
        ));
    }
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.preprocessed_width,
    ));
    observe_merkle_cap_8_host(&mut challenger, &structured.trace_commitment);
    if metadata.preprocessed_width > 0 {
        observe_merkle_cap_8_host(&mut challenger, &metadata.preprocessed_commitment);
    }
    let public_values: Vec<p3_goldilocks::Goldilocks> = public_inputs
        .iter()
        .map(|&v| p3_goldilocks::Goldilocks::from_u64(v))
        .collect();
    challenger.observe_slice(&public_values);
    for _ in 0..metadata.num_permutation_challenges {
        let _: P3Challenge = challenger.sample_algebra_element();
    }
    if metadata.permutation_width > 0 {
        observe_merkle_cap_8_host(&mut challenger, &structured.permutation_commitment);
    }
    let alpha = challenger.sample_algebra_element();
    observe_merkle_cap_8_host(&mut challenger, &structured.quotient_chunks_commitment);
    let zeta = challenger.sample_algebra_element();
    Ok((alpha, zeta))
}

/// Recomposes the full quotient polynomial's value at `zeta` from its
/// per-chunk evaluations: for each chunk `i`, weights `quotient_chunks[i]`
/// evaluated at `zeta` by the product of every *other* chunk domain's
/// vanishing polynomial at `zeta` (Lagrange-style chunk recombination, the
/// standard technique for evaluating a degree-`d` polynomial split across
/// `d`-many smaller-domain chunks). [`FixedQuotientChunkDomain`]'s
/// `shift_inverse`/`other_vanishing_inverse` are exactly the precomputed
/// constants this needs.
fn recompose_quotient_chunks_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    quotient_chunks: &[Vec<ExtensionTarget<RECURSION_D>>],
    zeta: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
) -> ExtensionTarget<RECURSION_D> {
    let one = builder.one_extension();
    let zps: Vec<_> = metadata
        .quotient_chunk_domains
        .iter()
        .enumerate()
        .map(|(i, _)| {
            metadata
                .quotient_chunk_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold(builder.one_extension(), |acc, (_j, domain_j)| {
                    let unshifted = builder
                        .mul_const_extension(F::from_canonical_u64(domain_j.shift_inverse), zeta);
                    let raised = builder.exp_power_of_2_extension(unshifted, domain_j.log_size);
                    let vanishing = builder.sub_extension(raised, one);
                    let scaled = builder.mul_const_extension(
                        F::from_canonical_u64(domain_j.other_vanishing_inverse[i]),
                        vanishing,
                    );
                    builder.mul_extension(acc, scaled)
                })
        })
        .collect();

    quotient_chunks
        .iter()
        .zip(zps)
        .fold(builder.zero_extension(), |acc, (chunk, zp)| {
            let quotient_i = recompose_ext_coefficients_targets(builder, chunk);
            let weighted = builder.mul_extension(zp, quotient_i);
            builder.add_extension(acc, weighted)
        })
}

/// Native counterpart of [`recompose_quotient_chunks_targets`]; used by
/// [`prove_transcript_verifier`] to compute the value assigned to
/// `expected_quotient`.
fn recompose_quotient_chunks_host(
    proof: &StructuredP3Proof,
    zeta: P3Challenge,
    metadata: &FixedVerifierMetadata,
) -> Result<P3Challenge, RecursionError> {
    validate_structured_openings_shape(metadata, proof)?;

    Ok(metadata
        .quotient_chunk_domains
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let zp = metadata
                .quotient_chunk_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold(P3Challenge::ONE, |acc, (_j, domain_j)| {
                    let shift_inv = p3_goldilocks::Goldilocks::from_u64(domain_j.shift_inverse);
                    let vanishing =
                        (zeta * shift_inv).exp_power_of_2(domain_j.log_size) - P3Challenge::ONE;
                    acc * vanishing
                        * p3_goldilocks::Goldilocks::from_u64(domain_j.other_vanishing_inverse[i])
                });
            p3_challenge_from_ext_pairs(&proof.quotient_chunks[i]).map(|quotient_i| zp * quotient_i)
        })
        .try_fold(P3Challenge::ZERO, |acc, term| term.map(|term| acc + term))?)
}

/// `Z_H(zeta) = zeta^(2^base_degree_bits) - 1`, the trace domain's vanishing
/// polynomial evaluated at `zeta` — the right-hand side's denominator-free
/// form in the OODS check `folded_constraints == quotient(zeta) * Z_H(zeta)`.
fn z_h_at_zeta_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    base_degree_bits: usize,
) -> ExtensionTarget<RECURSION_D> {
    let one = builder.one_extension();
    let zeta_pow_n = builder.exp_power_of_2_extension(zeta, base_degree_bits);
    builder.sub_extension(zeta_pow_n, one)
}

/// Evaluates every one of `SignatureBatchAir`'s constraints (captured in
/// `symbolic_constraints`) at `zeta`, in their original declaration order
/// (interleaving the base-field and extension-field constraint lists per
/// `symbolic_constraints.layout`, since `p3_air::symbolic` records them in
/// two separate lists but the AIR emitted them interleaved), and folds them
/// with `alpha` via Horner's method (`acc = acc*alpha + constraint_i`). The
/// result is one side of the OODS check; [`build_transcript_verifier_circuit`]
/// connects it to `quotient(zeta) * Z_H(zeta)`.
fn derive_folded_constraints_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    public_inputs: &[Target],
    proof: &TranscriptVerifierTargets,
    permutation_challenges: &[ExtensionTarget<RECURSION_D>],
    alpha: ExtensionTarget<RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
    symbolic_constraints: &SymbolicAirConstraints,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    let selectors = lagrange_selectors_targets(builder, zeta, metadata);
    let permutation_local = recompose_permutation_targets(builder, &proof.permutation_local);
    let permutation_next = recompose_permutation_targets(builder, &proof.permutation_next);
    let ctx = SymbolicEvaluationContext {
        trace_local: &proof.trace_local,
        trace_next: &proof.trace_next,
        preprocessed_local: &proof.preprocessed_local,
        preprocessed_next: &proof.preprocessed_next,
        permutation_local: &permutation_local,
        permutation_next: &permutation_next,
        permutation_challenges,
        public_inputs,
        selectors,
    };

    let mut base_iter = symbolic_constraints.base.iter();
    let mut ext_iter = symbolic_constraints.ext.iter();
    let mut base_indices = symbolic_constraints
        .layout
        .base_indices
        .iter()
        .copied()
        .peekable();
    let mut ext_indices = symbolic_constraints
        .layout
        .ext_indices
        .iter()
        .copied()
        .peekable();
    let mut acc = builder.zero_extension();

    for global_idx in 0..symbolic_constraints.layout.total_constraints() {
        acc = builder.mul_extension(acc, alpha);
        let term = if base_indices.peek().copied() == Some(global_idx) {
            base_indices.next();
            resolve_base_expression_targets(
                builder,
                base_iter.next().expect("base constraint missing"),
                &ctx,
            )?
        } else if ext_indices.peek().copied() == Some(global_idx) {
            ext_indices.next();
            resolve_ext_expression_targets(
                builder,
                ext_iter.next().expect("extension constraint missing"),
                &ctx,
            )?
        } else {
            return Err(RecursionError::InvariantViolation(
                "symbolic constraint layout indices do not cover all constraints",
            ));
        };
        acc = builder.add_extension(acc, term);
    }

    Ok(acc)
}

/// Recombines flattened permutation openings (`RECURSION_D` base-coefficient
/// columns per extension column, each opened in the extension field) into
/// extension values: `v = c0 + c1 * X`.
fn recompose_permutation_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    flat: &[ExtensionTarget<RECURSION_D>],
) -> Vec<ExtensionTarget<RECURSION_D>> {
    type Fe = <F as Extendable<RECURSION_D>>::Extension;
    let basis: Vec<ExtensionTarget<RECURSION_D>> = (0..RECURSION_D)
        .map(|d| {
            let mut coeffs = [F::ZERO; RECURSION_D];
            coeffs[d] = F::ONE;
            builder.constant_extension(Fe::from_basefield_array(coeffs))
        })
        .collect();
    flat.chunks(RECURSION_D)
        .map(|coeffs| {
            let mut acc = builder.zero_extension();
            for (c, b) in coeffs.iter().zip(&basis) {
                acc = builder.mul_add_extension(*c, *b, acc);
            }
            acc
        })
        .collect()
}

/// Evaluates the AIR's three Lagrange row selectors at `zeta`, expressed
/// without ever dividing by an unconstrained zero: `is_first_row =
/// Z_H(zeta)/(zeta-1)` and `is_last_row = Z_H(zeta)/(zeta-g^{-1})` (`g` the
/// trace subgroup generator) are the standard closed forms for "is this
/// point the first/last root of the vanishing polynomial", and
/// `is_transition` is left as the bare denominator `zeta - g^{-1}` since
/// callers only ever multiply by it (see [`SymbolicEvaluationContext`]'s
/// `selectors` field).
fn lagrange_selectors_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
) -> LagrangeSelectorTargets {
    let one = builder.one_extension();
    let subgroup_last = builder.constant_extension(
        <<F as Extendable<RECURSION_D>>::Extension as Plonky2FieldExtension<RECURSION_D>>::from_basefield(F::from_canonical_u64(
            metadata.trace_subgroup_generator_inverse,
        )),
    );
    let z_h = z_h_at_zeta_targets(builder, zeta, metadata.base_degree_bits);
    let first_denom = builder.sub_extension(zeta, one);
    let last_denom = builder.sub_extension(zeta, subgroup_last);

    LagrangeSelectorTargets {
        is_first_row: builder.div_extension(z_h, first_denom),
        is_last_row: builder.div_extension(z_h, last_denom),
        is_transition: last_denom,
    }
}

/// Recursively emits circuit gates for one base-field symbolic AIR
/// expression tree node, in the extension field (since every value in this
/// circuit's OODS evaluation is an `ExtensionTarget`, even ones that were
/// base-field in the original AIR).
fn resolve_base_expression_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    expr: &SymbolicExpression<p3_goldilocks::Goldilocks>,
    ctx: &SymbolicEvaluationContext<'_>,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    match expr {
        SymbolicExpr::Leaf(leaf) => resolve_base_leaf_targets(builder, leaf, ctx),
        SymbolicExpr::Add { x, y, .. } => {
            let left = resolve_base_expression_targets(builder, x, ctx)?;
            let right = resolve_base_expression_targets(builder, y, ctx)?;
            Ok(builder.add_extension(left, right))
        }
        SymbolicExpr::Sub { x, y, .. } => {
            let left = resolve_base_expression_targets(builder, x, ctx)?;
            let right = resolve_base_expression_targets(builder, y, ctx)?;
            Ok(builder.sub_extension(left, right))
        }
        SymbolicExpr::Neg { x, .. } => {
            let zero = builder.zero_extension();
            let inner = resolve_base_expression_targets(builder, x, ctx)?;
            Ok(builder.sub_extension(zero, inner))
        }
        SymbolicExpr::Mul { x, y, .. } => {
            let left = resolve_base_expression_targets(builder, x, ctx)?;
            let right = resolve_base_expression_targets(builder, y, ctx)?;
            Ok(builder.mul_extension(left, right))
        }
    }
}

/// Resolves one symbolic-expression leaf to its actual value: a trace
/// column read (local/next, main/preprocessed), a public input, one of the
/// three Lagrange selectors, or a literal constant. `BaseEntry::Periodic`
/// and any other variant are explicitly rejected — `SignatureBatchAir`
/// doesn't use periodic columns, so encountering one here would mean this
/// circuit's assumptions about the AIR's shape are stale.
fn resolve_base_leaf_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    leaf: &BaseLeaf<p3_goldilocks::Goldilocks>,
    ctx: &SymbolicEvaluationContext<'_>,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    match leaf {
        BaseLeaf::Variable(var) => match var.entry {
            BaseEntry::Main { offset: 0 } => {
                Ok(*ctx.trace_local.get(var.index).ok_or_else(|| {
                    RecursionError::Plonky2Witness(
                        "main local symbolic index out of bounds".to_string(),
                    )
                })?)
            }
            BaseEntry::Main { offset: 1 } => {
                Ok(*ctx.trace_next.get(var.index).ok_or_else(|| {
                    RecursionError::Plonky2Witness(
                        "main next symbolic index out of bounds".to_string(),
                    )
                })?)
            }
            BaseEntry::Preprocessed { offset: 0 } => {
                Ok(*ctx.preprocessed_local.get(var.index).ok_or_else(|| {
                    RecursionError::Plonky2Witness(
                        "preprocessed local symbolic index out of bounds".to_string(),
                    )
                })?)
            }
            BaseEntry::Preprocessed { offset: 1 } => {
                Ok(*ctx.preprocessed_next.get(var.index).ok_or_else(|| {
                    RecursionError::Plonky2Witness(
                        "preprocessed next symbolic index out of bounds".to_string(),
                    )
                })?)
            }
            BaseEntry::Public => Ok(builder.convert_to_ext(
                *ctx.public_inputs.get(var.index).ok_or_else(|| {
                    RecursionError::Plonky2Witness(
                        "public symbolic index out of bounds".to_string(),
                    )
                })?,
            )),
            BaseEntry::Periodic => Err(RecursionError::UnsupportedVerifierFeature(
                "periodic columns in recursive symbolic evaluator",
            )),
            _ => Err(RecursionError::UnsupportedVerifierFeature(
                "unexpected symbolic base entry",
            )),
        },
        BaseLeaf::IsFirstRow => Ok(ctx.selectors.is_first_row),
        BaseLeaf::IsLastRow => Ok(ctx.selectors.is_last_row),
        BaseLeaf::IsTransition => Ok(ctx.selectors.is_transition),
        BaseLeaf::Constant(value) => {
            Ok(
                builder.constant_extension(
                    <<F as Extendable<RECURSION_D>>::Extension as Plonky2FieldExtension<
                        RECURSION_D,
                    >>::from_basefield(goldilocks_from_p3(*value)),
                ),
            )
        }
    }
}

/// Recursively emits circuit gates for one extension-field symbolic AIR
/// expression tree node. Mirrors [`resolve_base_expression_targets`]'s
/// recursion exactly, but over `ExtLeaf`.
fn resolve_ext_expression_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    expr: &SymbolicExpressionExt<p3_goldilocks::Goldilocks, P3Challenge>,
    ctx: &SymbolicEvaluationContext<'_>,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    match expr {
        SymbolicExpr::Leaf(leaf) => resolve_ext_leaf_targets(builder, leaf, ctx),
        SymbolicExpr::Add { x, y, .. } => {
            let left = resolve_ext_expression_targets(builder, x, ctx)?;
            let right = resolve_ext_expression_targets(builder, y, ctx)?;
            Ok(builder.add_extension(left, right))
        }
        SymbolicExpr::Sub { x, y, .. } => {
            let left = resolve_ext_expression_targets(builder, x, ctx)?;
            let right = resolve_ext_expression_targets(builder, y, ctx)?;
            Ok(builder.sub_extension(left, right))
        }
        SymbolicExpr::Neg { x, .. } => {
            let zero = builder.zero_extension();
            let inner = resolve_ext_expression_targets(builder, x, ctx)?;
            Ok(builder.sub_extension(zero, inner))
        }
        SymbolicExpr::Mul { x, y, .. } => {
            let left = resolve_ext_expression_targets(builder, x, ctx)?;
            let right = resolve_ext_expression_targets(builder, y, ctx)?;
            Ok(builder.mul_extension(left, right))
        }
    }
}

/// Resolves an extension-field symbolic leaf: either delegates to a nested
/// base-field expression, or a literal extension-field constant.
/// `ExtVariable` is rejected — `SignatureBatchAir` never introduces a free
/// extension-field trace variable, so seeing one here indicates a stale
/// assumption about the AIR's shape, the same as the `BaseEntry::Periodic`
/// case in [`resolve_base_leaf_targets`].
fn resolve_ext_leaf_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    leaf: &ExtLeaf<p3_goldilocks::Goldilocks, P3Challenge>,
    ctx: &SymbolicEvaluationContext<'_>,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    match leaf {
        ExtLeaf::Base(expr) => resolve_base_expression_targets(builder, expr, ctx),
        ExtLeaf::ExtConstant(value) => {
            Ok(builder.constant_extension(p3_challenge_to_plonky2(*value)?))
        }
        ExtLeaf::ExtVariable(SymbolicVariableExt { entry, index, .. }) => {
            let values = match entry {
                ExtEntry::Permutation { offset: 0 } => ctx.permutation_local,
                ExtEntry::Permutation { offset: 1 } => ctx.permutation_next,
                ExtEntry::Challenge => ctx.permutation_challenges,
                _ => {
                    return Err(RecursionError::UnsupportedVerifierFeature(
                        "unexpected symbolic extension entry",
                    ));
                }
            };
            values.get(*index).copied().ok_or_else(|| {
                RecursionError::Plonky2Witness("extension symbolic index out of bounds".to_string())
            })
        }
    }
}

/// Captures `SignatureBatchAir::eval`'s constraints symbolically by running
/// the *actual* `p3-schnorr` AIR against a `SymbolicAirBuilder` instead of a
/// concrete one. This is what guarantees the recursive circuit checks
/// exactly the constraints `p3-schnorr` defines — not a hand-transcribed
/// approximation of them that could silently drift out of sync after an AIR
/// change.
fn symbolic_constraints_for_metadata(metadata: &FixedVerifierMetadata) -> SymbolicAirConstraints {
    let layout = AirLayout {
        preprocessed_width: metadata.preprocessed_width,
        main_width: metadata.air_width,
        num_public_values: metadata.num_public_values,
        permutation_width: metadata.permutation_width / RECURSION_D,
        num_permutation_challenges: metadata.num_permutation_challenges,
        num_permutation_values: 0,
        num_periodic_columns: 0,
    };
    let mut builder = SymbolicAirBuilder::<p3_goldilocks::Goldilocks, P3Challenge>::new(layout);
    match metadata.kind {
        InnerProofKind::EcdsaLogUp => {
            EcdsaBatchAir::new(metadata.signature_capacity).eval(&mut builder)
        }
        InnerProofKind::MessageHash => {
            MessageHashAir::new(metadata.signature_capacity).eval(&mut builder)
        }
    }
    SymbolicAirConstraints {
        base: builder.base_constraints(),
        ext: builder.extension_constraints(),
        layout: builder.constraint_layout(),
    }
}

/// The in-circuit FRI verifier, called from [`build_transcript_verifier_circuit`]
/// after the OODS/quotient check is wired up. Takes the challenger returned
/// by [`derive_transcript_challenges_targets`] (already advanced past
/// `alpha`/`zeta`) and continues replaying the *same* transcript from there:
/// observes the OODS-opened values to derive `fri_alpha`, observes each
/// commit-phase round's cap and optionally checks its grinding
/// proof-of-work, derives every `beta`, checks the query proof-of-work,
/// then derives and verifies each query's FRI proof via
/// [`verify_fri_query_targets`]. This — together with the OODS check — is
/// the circuit's complete soundness argument; see the module docs.
///
/// `challenger` is consumed by value (not re-created here) so this
/// continues the one transcript `derive_transcript_challenges_targets`
/// started, rather than replaying the `degree_bits`/commitment/public-input
/// observations a second time at the cost of another round of Poseidon2
/// permutations.
#[allow(dead_code)]
fn verify_fixed_fri_input_openings_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    mut challenger: P3RecursiveDuplexChallenger,
    zeta: ExtensionTarget<RECURSION_D>,
    proof: &TranscriptVerifierTargets,
    metadata: &FixedVerifierMetadata,
) -> Result<(), RecursionError> {
    observe_extension_targets(builder, &mut challenger, &proof.trace_local);
    observe_extension_targets(builder, &mut challenger, &proof.trace_next);
    observe_extension_targets(builder, &mut challenger, &proof.permutation_local);
    observe_extension_targets(builder, &mut challenger, &proof.permutation_next);
    for chunk in &proof.quotient_chunks {
        observe_extension_targets(builder, &mut challenger, chunk);
    }
    observe_extension_targets(builder, &mut challenger, &proof.preprocessed_local);
    observe_extension_targets(builder, &mut challenger, &proof.preprocessed_next);
    let fri_alpha = challenger.get_extension_challenge(builder);
    let groups = reduced_opening_claims_targets(builder, proof, fri_alpha, metadata);
    let mut fri_betas = Vec::with_capacity(proof.fri.commit_phase_commits.len());

    for (cap, pow_witness) in proof
        .fri
        .commit_phase_commits
        .iter()
        .zip(proof.fri.commit_pow_witnesses.iter())
    {
        observe_merkle_cap_8(&mut challenger, builder, cap);
        if metadata.fri_commit_pow_bits > 0 {
            challenger.observe_element(builder, *pow_witness);
            let sampled = challenger.get_challenge(builder);
            let low_bits =
                canonical_goldilocks_low_bits(builder, sampled, metadata.fri_commit_pow_bits);
            for bit in low_bits {
                builder.assert_zero(bit.target);
            }
        }
        fri_betas.push(challenger.get_extension_challenge(builder));
    }

    observe_extension_targets(builder, &mut challenger, &proof.fri.final_poly);
    for &log_arity_bits in &metadata.fri_commit_phase_log_arities {
        let log_arity = builder.constant(F::from_canonical_usize(log_arity_bits));
        challenger.observe_element(builder, log_arity);
    }
    challenger.observe_element(builder, proof.fri.query_pow_witness);
    let sampled = challenger.get_challenge(builder);
    let low_bits = canonical_goldilocks_low_bits(builder, sampled, metadata.fri_query_pow_bits);
    for bit in low_bits {
        builder.assert_zero(bit.target);
    }

    let query_bits_count = metadata.degree_bits + metadata.fri_log_blowup;
    for query in &proof.fri.query_proofs {
        let challenge = challenger.get_challenge(builder);
        let query_bits = canonical_goldilocks_low_bits(builder, challenge, query_bits_count);
        verify_fri_query_targets(
            builder,
            proof,
            query,
            &query_bits,
            fri_alpha,
            zeta,
            &fri_betas,
            &groups,
            metadata,
        )?;
    }

    Ok(())
}

/// In-circuit counterpart of [`verify_fixed_fri_host`] for one query:
/// checks the three input batch openings (trace, quotient chunks,
/// preprocessed — note the preprocessed cap is a circuit *constant*, via
/// [`verify_fixed_batch_opening_targets_constant_cap`], since it's baked
/// into [`FixedVerifierMetadata`] rather than supplied per-proof), computes
/// the initial DEEP reduced opening, folds it through every commit-phase
/// round (checking each round's Merkle opening as it goes), and finally
/// checks the fully-folded value against the committed final polynomial
/// evaluated at this query's final point.
#[allow(dead_code)]
fn verify_fri_query_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    proof: &TranscriptVerifierTargets,
    query: &FixedFriQueryTargets,
    query_bits: &[BoolTarget],
    fri_alpha: ExtensionTarget<RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    fri_betas: &[ExtensionTarget<RECURSION_D>],
    groups: &[OpeningGroup],
    metadata: &FixedVerifierMetadata,
) -> Result<(), RecursionError> {
    let mut batches = query.input_proof.iter();
    let mut next_batch = || {
        batches.next().ok_or_else(|| {
            RecursionError::Plonky2Witness(
                "fixed FRI query is missing an input batch opening".to_string(),
            )
        })
    };
    verify_fixed_batch_opening_targets(builder, &proof.trace_commitment, next_batch()?, query_bits);
    if metadata.permutation_width > 0 {
        verify_fixed_batch_opening_targets(
            builder,
            &proof.permutation_commitment,
            next_batch()?,
            query_bits,
        );
    }
    verify_fixed_batch_opening_targets(
        builder,
        &proof.quotient_chunks_commitment,
        next_batch()?,
        query_bits,
    );
    if metadata.preprocessed_width > 0 {
        verify_fixed_batch_opening_targets_constant_cap(
            builder,
            &metadata.preprocessed_commitment,
            next_batch()?,
            query_bits,
        );
    }

    let mut folded_eval = compute_initial_fri_reduced_opening_targets(
        builder, query, query_bits, fri_alpha, zeta, groups, metadata,
    )?;
    let mut current_index_bits = query_bits.to_vec();
    let mut current_log_height = metadata.degree_bits + metadata.fri_log_blowup;

    if query.commit_phase_openings.len() != metadata.fri_commit_phase_log_arities.len()
        || fri_betas.len() != metadata.fri_commit_phase_log_arities.len()
    {
        return Err(RecursionError::Plonky2Witness(
            "fixed FRI commit-phase round count does not match metadata".to_string(),
        ));
    }

    for (((opening, commit_cap), &beta), (&log_arity, &proof_len)) in query
        .commit_phase_openings
        .iter()
        .zip(proof.fri.commit_phase_commits.iter())
        .zip(fri_betas.iter())
        .zip(
            metadata
                .fri_commit_phase_log_arities
                .iter()
                .zip(metadata.fri_commit_phase_opening_proof_lens.iter()),
        )
    {
        let expected_log_arity = builder.constant(F::from_canonical_usize(log_arity));
        builder.connect(opening.log_arity, expected_log_arity);
        if opening.sibling_values.len() != (1usize << log_arity) - 1
            || opening.opening_proof.len() != proof_len
        {
            return Err(RecursionError::Plonky2Witness(
                "fixed FRI commit-phase opening shape does not match metadata".to_string(),
            ));
        }

        let index_in_group_bits = &current_index_bits[..log_arity];
        let parent_index_bits = &current_index_bits[log_arity..];
        let index_in_group = bits_to_target(builder, index_in_group_bits);
        let row_evals = reconstruct_commit_phase_row_targets(
            builder,
            index_in_group,
            &opening.sibling_values,
            folded_eval,
        );

        verify_commit_phase_opening_targets(
            builder,
            commit_cap,
            parent_index_bits,
            &row_evals,
            &opening.opening_proof,
        );

        let log_folded_height = current_log_height - log_arity;
        folded_eval = fold_commit_phase_row_targets(
            builder,
            parent_index_bits,
            log_folded_height,
            log_arity,
            beta,
            &row_evals,
        );
        current_index_bits = parent_index_bits.to_vec();
        current_log_height = log_folded_height;
    }

    if current_log_height != metadata.fri_final_log_height {
        return Err(RecursionError::Plonky2Witness(
            "fixed FRI final folded height does not match metadata".to_string(),
        ));
    }

    let final_x = subgroup_point_targets(
        builder,
        &current_index_bits,
        metadata.degree_bits + metadata.fri_log_blowup,
    );
    let final_eval = evaluate_final_poly_targets(builder, &proof.fri.final_poly, final_x);
    builder.connect_extension(final_eval, folded_eval);

    Ok(())
}

/// One committed matrix's openings: its position in each query's input
/// batches, and per opening point `(is_next, alpha^offset, sum_i alpha^i v_i)`.
struct OpeningGroup {
    batch: usize,
    matrix: usize,
    points: Vec<(
        bool,
        ExtensionTarget<RECURSION_D>,
        ExtensionTarget<RECURSION_D>,
    )>,
}

/// `sum_i alpha^i values[i]` by Horner's rule.
fn horner_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    values: &[ExtensionTarget<RECURSION_D>],
    alpha: ExtensionTarget<RECURSION_D>,
) -> ExtensionTarget<RECURSION_D> {
    let zero = builder.zero_extension();
    values
        .iter()
        .rev()
        .fold(zero, |acc, &v| builder.mul_add_extension(acc, alpha, v))
}

/// The query-independent half of the FRI reduced opening, in the order the
/// P3 PCS verifier walks it (batch, matrix, point, column; one `alpha` power
/// sequence since every matrix has the same height): for each (matrix, point),
/// `alpha^offset` and `sum_i alpha^i v_i` over the claimed values `v`.
fn reduced_opening_claims_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    proof: &TranscriptVerifierTargets,
    fri_alpha: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
) -> Vec<OpeningGroup> {
    type Points<'a> = Vec<(bool, &'a [ExtensionTarget<RECURSION_D>])>;
    let mut layout: Vec<(usize, usize, Points<'_>)> = Vec::new();
    let mut batch = 0;
    layout.push((
        batch,
        0,
        vec![(false, &proof.trace_local), (true, &proof.trace_next)],
    ));
    batch += 1;
    if metadata.permutation_width > 0 {
        layout.push((
            batch,
            0,
            vec![
                (false, &proof.permutation_local),
                (true, &proof.permutation_next),
            ],
        ));
        batch += 1;
    }
    for (m, chunk) in proof.quotient_chunks.iter().enumerate() {
        layout.push((batch, m, vec![(false, chunk)]));
    }
    batch += 1;
    if metadata.preprocessed_width > 0 {
        layout.push((
            batch,
            0,
            vec![
                (false, &proof.preprocessed_local),
                (true, &proof.preprocessed_next),
            ],
        ));
    }

    let mut alpha_pow = builder.one_extension();
    layout
        .into_iter()
        .map(|(batch, matrix, points)| OpeningGroup {
            batch,
            matrix,
            points: points
                .into_iter()
                .filter(|(_, values)| !values.is_empty())
                .map(|(is_next, values)| {
                    let claimed = horner_targets(builder, values, fri_alpha);
                    let offset = alpha_pow;
                    let step = builder.exp_u64_extension(fri_alpha, values.len() as u64);
                    alpha_pow = builder.mul_extension(alpha_pow, step);
                    (is_next, offset, claimed)
                })
                .collect(),
        })
        .collect()
}

/// In-circuit counterpart of the P3 PCS verifier's initial reduced opening for
/// one query: `sum alpha^k (v_k - f_k(x)) / (z - x)`, computed per
/// (matrix, point) as `alpha^offset * (sum_i alpha^i v_i - sum_i alpha^i f_i(x)) / (z - x)`.
/// The `v` half comes precomputed from [`reduced_opening_claims_targets`]; the
/// `f(x)` half is one Horner pass per opened row, shared by both points.
fn compute_initial_fri_reduced_opening_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    query: &FixedFriQueryTargets,
    query_bits: &[BoolTarget],
    fri_alpha: ExtensionTarget<RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    groups: &[OpeningGroup],
    metadata: &FixedVerifierMetadata,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    let x = coset_point_targets(
        builder,
        query_bits,
        metadata.degree_bits + metadata.fri_log_blowup,
    );
    let trace_next_point = mul_by_subgroup_generator_targets(builder, zeta, metadata);
    let denom_zeta = builder.sub_extension(zeta, x);
    let denom_next = builder.sub_extension(trace_next_point, x);
    let inv_denom_zeta = builder.inverse_extension(denom_zeta);
    let inv_denom_next = builder.inverse_extension(denom_next);

    let mut reduced_opening = builder.zero_extension();
    for group in groups {
        let row = query
            .input_proof
            .get(group.batch)
            .and_then(|b| b.opened_values.get(group.matrix))
            .ok_or_else(|| {
                RecursionError::Plonky2Witness(
                    "fixed FRI query is missing an opened matrix row".to_string(),
                )
            })?;
        let row: Vec<_> = row.iter().map(|&v| builder.convert_to_ext(v)).collect();
        let at_x = horner_targets(builder, &row, fri_alpha);
        for &(is_next, offset, claimed) in &group.points {
            let inv = if is_next {
                inv_denom_next
            } else {
                inv_denom_zeta
            };
            let diff = builder.sub_extension(claimed, at_x);
            let quotient = builder.mul_extension(diff, inv);
            reduced_opening = builder.mul_add_extension(offset, quotient, reduced_opening);
        }
    }
    Ok(reduced_opening)
}

/// In-circuit counterpart of [`verify_fixed_commit_phase_opening_host`].
#[allow(dead_code)]
fn verify_commit_phase_opening_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    cap: &MerkleCapTargets8,
    parent_index_bits: &[BoolTarget],
    row_evals: &[ExtensionTarget<RECURSION_D>],
    opening_proof: &[[Target; 8]],
) {
    let flat = flatten_extension_targets(row_evals);
    let leaf = padding_free_hash_targets(builder, &flat);
    verify_binary_merkle_path_targets(builder, &cap.roots, leaf, parent_index_bits, opening_proof);
}

/// In-circuit counterpart of [`reconstruct_commit_phase_row_host`].
/// `index_in_group` is a `Target` rather than a known constant (it depends
/// on the query's bit-decomposed index), so the circuit can't simply
/// `Vec::insert` `folded_eval` at a fixed position — instead, for each
/// output slot `j`, it builds a chain of `select_ext`s keyed on
/// `is_equal(index_in_group, k)` for every `k`, picking `folded_eval` when
/// `k == j == index_in_group` and the appropriate sibling otherwise.
#[allow(dead_code)]
fn reconstruct_commit_phase_row_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    index_in_group: Target,
    sibling_values: &[ExtensionTarget<RECURSION_D>],
    folded_eval: ExtensionTarget<RECURSION_D>,
) -> Vec<ExtensionTarget<RECURSION_D>> {
    let arity = sibling_values.len() + 1;
    (0..arity)
        .map(|j| {
            let j_target = builder.constant(F::from_canonical_usize(j));
            let eq = builder.is_equal(index_in_group, j_target);
            let mut value = if j == 0 {
                sibling_values[0]
            } else {
                sibling_values[j - 1]
            };
            for k in 0..arity {
                let candidate = if k == j {
                    folded_eval
                } else if k < j {
                    sibling_values[j - 1]
                } else {
                    sibling_values[j.min(arity - 2)]
                };
                let k_target = builder.constant(F::from_canonical_usize(k));
                let is_self = builder.is_equal(index_in_group, k_target);
                value = builder.select_ext(is_self, candidate, value);
            }
            builder.select_ext(eq, folded_eval, value)
        })
        .collect()
}

/// Folds one commit-phase round's reconstructed row down to a single
/// extension-field value at `beta`, via Lagrange interpolation over the
/// `2^log_arity` group members starting at `subgroup_start` — the in-circuit
/// equivalent of `p3_fri::FriFoldingStrategy::fold_row`, which
/// [`verify_fixed_fri_host`] calls directly on the native side.
#[allow(dead_code)]
fn fold_commit_phase_row_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    parent_index_bits: &[BoolTarget],
    log_folded_height: usize,
    log_arity: usize,
    beta: ExtensionTarget<RECURSION_D>,
    row_evals: &[ExtensionTarget<RECURSION_D>],
) -> ExtensionTarget<RECURSION_D> {
    let subgroup_start =
        subgroup_start_targets(builder, parent_index_bits, log_folded_height, log_arity);
    lagrange_interpolate_targets(builder, subgroup_start, log_arity, row_evals, beta)
}

/// Evaluates the FRI final polynomial at `x` via Horner's method over
/// `coeffs` (`p3_fri`'s coefficient ordering — iterated highest-index first
/// here, then accumulated low-to-high), matching the convention
/// [`verify_fixed_fri_host`] uses on the native side.
#[allow(dead_code)]
fn evaluate_final_poly_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    coeffs: &[ExtensionTarget<RECURSION_D>],
    x: Target,
) -> ExtensionTarget<RECURSION_D> {
    let x_ext = builder.convert_to_ext(x);
    coeffs
        .iter()
        .rev()
        .fold(builder.zero_extension(), |acc, coeff| {
            let term = builder.mul_extension(acc, x_ext);
            builder.add_extension(term, *coeff)
        })
}

/// In-circuit Merkle batch-opening check against a per-proof (virtual
/// target) cap: hashes the opening's flattened values to a leaf and walks
/// [`verify_binary_merkle_path_targets`] up to `cap`.
#[allow(dead_code)]
fn verify_fixed_batch_opening_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    cap: &MerkleCapTargets8,
    opening: &FixedBatchOpeningTargets,
    query_bits: &[BoolTarget],
) {
    let flat: Vec<_> = opening
        .opened_values
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let leaf = padding_free_hash_targets(builder, &flat);
    verify_binary_merkle_path_targets(
        builder,
        &cap.roots,
        leaf,
        query_bits,
        &opening.opening_proof,
    );
}

/// Same check as [`verify_fixed_batch_opening_targets`], but against a cap
/// baked into the circuit as constants rather than supplied per-proof — used
/// for the preprocessed commitment, which is fixed at build time (see
/// [`FixedVerifierMetadata::preprocessed_commitment`]).
#[allow(dead_code)]
fn verify_fixed_batch_opening_targets_constant_cap(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    cap: &P3MerkleCapWitness,
    opening: &FixedBatchOpeningTargets,
    query_bits: &[BoolTarget],
) {
    let flat: Vec<_> = opening
        .opened_values
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let leaf = padding_free_hash_targets(builder, &flat);
    let roots: Vec<[Target; 8]> = cap
        .roots
        .iter()
        .map(|root| root.map(|value| builder.constant(F::from_canonical_u64(value))))
        .collect();
    verify_binary_merkle_path_targets(builder, &roots, leaf, query_bits, &opening.opening_proof);
}

/// Walks a binary Merkle path from `leaf` up to one of `cap_roots`: at each
/// level, `query_bits[level]` (least-significant first) selects whether
/// `sibling` goes left or right of the running `digest` before compressing,
/// and the remaining high bits of `query_bits` (beyond `siblings.len()`)
/// index into the cap via `random_access` (since a Merkle *cap* — rather
/// than a single root — means the tree's top `log2(cap_roots.len())` levels
/// aren't proven, only looked up directly).
#[allow(dead_code)]
fn verify_binary_merkle_path_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    cap_roots: &[[Target; 8]],
    leaf: [Target; 8],
    query_bits: &[BoolTarget],
    siblings: &[[Target; 8]],
) {
    let mut digest = leaf;
    for (bit, sibling) in query_bits.iter().take(siblings.len()).zip(siblings.iter()) {
        let left = core::array::from_fn(|i| builder.select(*bit, sibling[i], digest[i]));
        let right = core::array::from_fn(|i| builder.select(*bit, digest[i], sibling[i]));
        digest = compress_digest_targets(builder, left, right);
    }
    let cap_index = bits_to_target(builder, &query_bits[siblings.len()..]);
    let selected_root: [Target; 8] = core::array::from_fn(|i| {
        builder.random_access(cap_index, cap_roots.iter().map(|root| root[i]).collect())
    });
    for (expected, actual) in selected_root.iter().zip(digest.iter()) {
        builder.connect(*expected, *actual);
    }
}

/// In-circuit counterpart of `p3-schnorr`'s `Hash` (a `PaddingFreeSponge`):
/// absorbs `input` [`POSEIDON_RATE`] elements at a time via
/// [`poseidon2_permute_targets`] and returns the first 8 output lanes as
/// the leaf digest. Must reproduce `PaddingFreeSponge`'s exact absorption
/// behavior (including how a final partial block is handled) bit-for-bit —
/// any divergence here would make every Merkle proof check fail, since the
/// native proof's leaves were hashed with the real sponge.
#[allow(dead_code)]
fn padding_free_hash_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    input: &[Target],
) -> [Target; 8] {
    let mut state = [builder.zero(); POSEIDON_WIDTH];
    let mut offset = 0usize;
    while offset < input.len() {
        let remaining = input.len() - offset;
        let absorbed = remaining.min(POSEIDON_RATE);
        state[..absorbed].copy_from_slice(&input[offset..offset + absorbed]);
        if absorbed == POSEIDON_RATE {
            poseidon2_permute_targets(builder, &mut state);
        } else if absorbed > 0 {
            poseidon2_permute_targets(builder, &mut state);
        }
        offset += absorbed;
    }
    state[..8].try_into().unwrap()
}

/// In-circuit counterpart of `p3-schnorr`'s `Compress` (a
/// `TruncatedPermutation`): permutes `left || right` and returns the first
/// 8 output lanes, the standard 2-to-1 Merkle internal-node compression.
#[allow(dead_code)]
fn compress_digest_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    left: [Target; 8],
    right: [Target; 8],
) -> [Target; 8] {
    let mut state = [builder.zero(); POSEIDON_WIDTH];
    state[..8].copy_from_slice(&left);
    state[8..16].copy_from_slice(&right);
    poseidon2_permute_targets(builder, &mut state);
    state[..8].try_into().unwrap()
}

/// Observes a slice of extension-field targets into the recursive
/// challenger, limb-wise (mirrors [`observe_extension_rows_host`]).
#[allow(dead_code)]
fn observe_extension_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    challenger: &mut P3RecursiveDuplexChallenger,
    values: &[ExtensionTarget<RECURSION_D>],
) {
    for value in values {
        challenger.observe_elements(builder, &value.0);
    }
}

/// Recomposes a little-endian (`bits[0]` = least significant) bit sequence
/// into a single `Target` integer.
#[allow(dead_code)]
fn bits_to_target(builder: &mut CircuitBuilder<F, RECURSION_D>, bits: &[BoolTarget]) -> Target {
    bits.iter()
        .enumerate()
        .fold(builder.zero(), |acc, (i, bit)| {
            let weight = builder.constant(F::from_canonical_u64(1u64 << i));
            builder.mul_add(bit.target, weight, acc)
        })
}

/// Flattens a slice of extension-field targets into their raw base-field
/// limbs, in order.
#[allow(dead_code)]
fn flatten_extension_targets(values: &[ExtensionTarget<RECURSION_D>]) -> Vec<Target> {
    values.iter().flat_map(|value| value.0).collect()
}

/// Reverses the low `bit_len` bits of `value`. FRI's domain indexing uses
/// bit-reversed order (a standard FFT/NTT convention), so converting
/// between "natural" query indices and domain positions needs this.
#[allow(dead_code)]
fn reverse_bits_len_usize(mut value: usize, bit_len: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bit_len {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

/// Computes `generator^(bit-reversed index_bits)` in the `2^total_log_height`-element
/// two-adic subgroup — the domain point a given (bit-reversed) FRI query
/// index corresponds to.
#[allow(dead_code)]
fn subgroup_point_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    index_bits: &[BoolTarget],
    total_log_height: usize,
) -> Target {
    let generator = goldilocks_from_p3(p3_goldilocks::Goldilocks::two_adic_generator(
        total_log_height,
    ));
    let mut exponent_bits = vec![builder._false(); total_log_height - index_bits.len()];
    exponent_bits.extend(index_bits.iter().rev().copied());
    builder.exp_from_bits_const_base(generator, exponent_bits.iter())
}

/// [`subgroup_point_targets`], additionally shifted by the field generator
/// — the actual evaluation point `x` a FRI query corresponds to in the
/// blown-up (coset-shifted) domain.
#[allow(dead_code)]
fn coset_point_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    index_bits: &[BoolTarget],
    total_log_height: usize,
) -> ExtensionTarget<RECURSION_D> {
    let point = subgroup_point_targets(builder, index_bits, total_log_height);
    let shift = builder.constant(goldilocks_from_p3(p3_goldilocks::Goldilocks::GENERATOR));
    let shifted = builder.mul(point, shift);
    builder.convert_to_ext(shifted)
}

/// The first point of the `2^log_arity`-element coset a folded query index
/// belongs to, within the `2^(log_folded_height+log_arity)`-element domain
/// — the base point [`lagrange_interpolate_targets`] interpolates the
/// commit-phase row's siblings around.
#[allow(dead_code)]
fn subgroup_start_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    parent_index_bits: &[BoolTarget],
    log_folded_height: usize,
    log_arity: usize,
) -> Target {
    let generator = goldilocks_from_p3(p3_goldilocks::Goldilocks::two_adic_generator(
        log_folded_height + log_arity,
    ));
    builder.exp_from_bits_const_base(generator, parent_index_bits.iter().rev())
}

/// Evaluates, at `beta`, the unique degree-`<arity` polynomial passing
/// through `(xs[j], ys[j])` for `j in 0..arity` (`xs` derived from
/// `subgroup_start` and the `log_arity`-element subgroup's bit-reversed
/// points) — standard Lagrange interpolation via the basis-polynomial sum
/// `sum_j ys[j] * prod_{m!=j} (beta-x_m)/(x_j-x_m)`. This is what actually
/// folds one commit-phase round's row down to its parent's value.
#[allow(dead_code)]
fn lagrange_interpolate_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    subgroup_start: Target,
    log_arity: usize,
    ys: &[ExtensionTarget<RECURSION_D>],
    beta: ExtensionTarget<RECURSION_D>,
) -> ExtensionTarget<RECURSION_D> {
    let arity = 1usize << log_arity;
    let subgroup_generator = p3_goldilocks::Goldilocks::two_adic_generator(log_arity);
    let xs: Vec<_> = (0..arity)
        .map(|j| {
            let rev = reverse_bits_len_usize(j, log_arity);
            let factor = goldilocks_from_p3(subgroup_generator.exp_u64(rev as u64));
            let factor_target = builder.constant(factor);
            builder.mul(subgroup_start, factor_target)
        })
        .collect();

    ys.iter()
        .enumerate()
        .fold(builder.zero_extension(), |acc, (j, &y_j)| {
            let x_j = builder.convert_to_ext(xs[j]);
            let mut numerator = builder.one_extension();
            let mut denominator = builder.one_extension();
            for (m, &x_m_base) in xs.iter().enumerate() {
                if m == j {
                    continue;
                }
                let x_m = builder.convert_to_ext(x_m_base);
                let beta_minus_xm = builder.sub_extension(beta, x_m);
                numerator = builder.mul_extension(numerator, beta_minus_xm);
                let xj_minus_xm = builder.sub_extension(x_j, x_m);
                denominator = builder.mul_extension(denominator, xj_minus_xm);
            }
            let basis_at_beta = builder.div_extension(numerator, denominator);
            let weighted = builder.mul_extension(y_j, basis_at_beta);
            builder.add_extension(acc, weighted)
        })
}

/// `point * g^{-1}` (`g` the trace subgroup generator) — `zeta`'s
/// counterpart point one trace row earlier, used to evaluate the DEEP
/// quotient's "next row" terms at `zeta_next` instead of `zeta`.
#[allow(dead_code)]
fn mul_by_subgroup_generator_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    point: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
) -> ExtensionTarget<RECURSION_D> {
    let subgroup_generator =
        p3_goldilocks::Goldilocks::from_u64(metadata.trace_subgroup_generator_inverse).inverse();
    builder.mul_const_extension(goldilocks_from_p3(subgroup_generator), point)
}

/// Allocates `roots` virtual 8-element digest targets for a [`MerkleCapTargets8`].
fn add_virtual_merkle_cap_8(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    roots: usize,
) -> MerkleCapTargets8 {
    MerkleCapTargets8 {
        roots: (0..roots)
            .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
            .collect(),
    }
}

/// Observes a per-proof (virtual target) Merkle cap into the recursive challenger.
fn observe_merkle_cap_8(
    challenger: &mut P3RecursiveDuplexChallenger,
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    cap: &MerkleCapTargets8,
) {
    for root in &cap.roots {
        challenger.observe_elements(builder, root);
    }
}

/// Observes a fixed (build-time-constant) Merkle cap into the recursive
/// challenger — used for the preprocessed commitment.
fn observe_merkle_cap_8_constant(
    challenger: &mut P3RecursiveDuplexChallenger,
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    cap: &P3MerkleCapWitness,
) {
    for root in &cap.roots {
        let elements = root.map(|value| builder.constant(F::from_canonical_u64(value)));
        challenger.observe_elements(builder, &elements);
    }
}

/// Native counterpart of [`observe_merkle_cap_8`]/[`observe_merkle_cap_8_constant`].
fn observe_merkle_cap_8_host(challenger: &mut P3Challenger, cap: &P3MerkleCapWitness) {
    for root in &cap.roots {
        let elements = root.map(p3_goldilocks::Goldilocks::from_u64);
        challenger.observe_slice(&elements);
    }
}

// Witness assignment: each `set_*` function below copies a piece of the
// `u64`-encoded [`StructuredP3Proof`] (or a host-computed value, for
// `alpha`/`zeta`/`quotient`) into the corresponding circuit target(s)
// allocated by `build_transcript_verifier_circuit`/`add_fixed_fri_targets`.
// This is plain data-shuffling, not where soundness lives — the circuit's
// own constraints (the `_targets` functions) are what actually check these
// assigned values are consistent with each other and with the claimed
// `alpha`/`zeta`/`quotient`.

/// Assigns every [`TranscriptVerifierTargets`] field from `proof` (the
/// structured P3 proof) and the host-derived `quotient`/`alpha`/`zeta`.
/// Called by [`prove_transcript_verifier`] before invoking the Plonky2 prover.
fn set_transcript_verifier_targets(
    witness: &mut PartialWitness<F>,
    targets: &TranscriptVerifierTargets,
    metadata: &FixedVerifierMetadata,
    proof: &StructuredP3Proof,
    quotient: P3Challenge,
    alpha: P3Challenge,
    zeta: P3Challenge,
) -> Result<(), RecursionError> {
    validate_structured_openings_shape(metadata, proof)?;
    witness
        .set_target(
            targets.degree_bits,
            F::from_canonical_usize(proof.degree_bits),
        )
        .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    set_merkle_cap_targets_8(witness, &targets.trace_commitment, &proof.trace_commitment)?;
    set_merkle_cap_targets_8(
        witness,
        &targets.quotient_chunks_commitment,
        &proof.quotient_chunks_commitment,
    )?;
    set_merkle_cap_targets_8(
        witness,
        &targets.permutation_commitment,
        &proof.permutation_commitment,
    )?;
    set_extension_targets(
        witness,
        &targets.permutation_local,
        &proof.permutation_local,
    )?;
    set_extension_targets(witness, &targets.permutation_next, &proof.permutation_next)?;
    set_extension_targets(witness, &targets.trace_local, &proof.trace_local)?;
    set_extension_targets(
        witness,
        &targets.trace_next,
        proof.trace_next.as_deref().unwrap_or(&[]),
    )?;
    set_extension_targets(
        witness,
        &targets.preprocessed_local,
        proof.preprocessed_local.as_deref().unwrap_or(&[]),
    )?;
    set_extension_targets(
        witness,
        &targets.preprocessed_next,
        proof.preprocessed_next.as_deref().unwrap_or(&[]),
    )?;
    set_nested_extension_targets(witness, &targets.quotient_chunks, &proof.quotient_chunks)?;
    set_extension_target(witness, targets.expected_quotient, quotient)?;
    set_extension_target(witness, targets.expected_alpha, alpha)?;
    set_extension_target(witness, targets.expected_zeta, zeta)?;
    set_fixed_fri_targets(witness, &targets.fri, &proof.opening_proof)?;
    Ok(())
}

/// Checks `proof`'s opened-value vector lengths match what `metadata`
/// expects, before assigning any witness target — an early, clear failure
/// instead of a confusing target-count mismatch deeper in `set_*`/`builder`
/// calls if a caller supplies a proof shaped for a different AIR/batch size.
fn validate_structured_openings_shape(
    metadata: &FixedVerifierMetadata,
    proof: &StructuredP3Proof,
) -> Result<(), RecursionError> {
    let trace_next_len = proof.trace_next.as_ref().map_or(0, Vec::len);
    let preprocessed_local_len = proof.preprocessed_local.as_ref().map_or(0, Vec::len);
    let preprocessed_next_len = proof.preprocessed_next.as_ref().map_or(0, Vec::len);

    if proof.trace_local.len() != metadata.air_width
        || proof.permutation_local.len() != metadata.permutation_width
        || proof.permutation_next.len() != metadata.permutation_width
        || trace_next_len != metadata.trace_next_width
        || preprocessed_local_len != metadata.preprocessed_width
        || preprocessed_next_len != metadata.preprocessed_next_width
        || proof.quotient_chunks.len() != metadata.num_quotient_chunks
        || proof
            .quotient_chunks
            .iter()
            .any(|chunk| chunk.len() != RECURSION_D)
    {
        return Err(RecursionError::Plonky2Witness(
            "structured P3 opening shape does not match fixed verifier metadata".to_string(),
        ));
    }

    Ok(())
}

/// Assigns a [`MerkleCapTargets8`] from a [`P3MerkleCapWitness`].
fn set_merkle_cap_targets_8(
    witness: &mut PartialWitness<F>,
    targets: &MerkleCapTargets8,
    cap: &P3MerkleCapWitness,
) -> Result<(), RecursionError> {
    for (target_root, root) in targets.roots.iter().zip(&cap.roots) {
        for (target, value) in target_root.iter().zip(root.iter()) {
            witness
                .set_target(*target, F::from_canonical_u64(*value))
                .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
        }
    }
    Ok(())
}

/// Assigns a single `ExtensionTarget` from a real `P3Challenge` value.
fn set_extension_target(
    witness: &mut PartialWitness<F>,
    target: ExtensionTarget<RECURSION_D>,
    value: P3Challenge,
) -> Result<(), RecursionError> {
    let coeffs: &[p3_goldilocks::Goldilocks] = value.as_basis_coefficients_slice();
    for (target_limb, value_limb) in target.0.iter().zip(coeffs.iter()) {
        witness
            .set_target(
                *target_limb,
                F::from_canonical_u64(value_limb.as_canonical_u64()),
            )
            .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    }
    Ok(())
}

/// Assigns a slice of `ExtensionTarget`s from their `[u64; 2]`-encoded values.
fn set_extension_targets(
    witness: &mut PartialWitness<F>,
    targets: &[ExtensionTarget<RECURSION_D>],
    values: &[[u64; 2]],
) -> Result<(), RecursionError> {
    if targets.len() != values.len() {
        return Err(RecursionError::Plonky2Witness(
            "extension target count does not match provided values".to_string(),
        ));
    }

    for (target, value) in targets.iter().zip(values.iter()) {
        for (target_limb, value_limb) in target.0.iter().zip(value.iter()) {
            witness
                .set_target(*target_limb, F::from_canonical_u64(*value_limb))
                .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
        }
    }

    Ok(())
}

/// [`set_extension_targets`] for a vector-of-vectors (the per-chunk quotient
/// evaluations).
fn set_nested_extension_targets(
    witness: &mut PartialWitness<F>,
    targets: &[Vec<ExtensionTarget<RECURSION_D>>],
    values: &[Vec<[u64; 2]>],
) -> Result<(), RecursionError> {
    if targets.len() != values.len() {
        return Err(RecursionError::Plonky2Witness(
            "nested extension target outer count does not match provided values".to_string(),
        ));
    }

    for (target_row, value_row) in targets.iter().zip(values.iter()) {
        set_extension_targets(witness, target_row, value_row)?;
    }

    Ok(())
}

/// Assigns every [`FixedFriProofTargets`] field from a [`crate::types::P3FriProofWitness`].
fn set_fixed_fri_targets(
    witness: &mut PartialWitness<F>,
    targets: &FixedFriProofTargets,
    proof: &crate::types::P3FriProofWitness,
) -> Result<(), RecursionError> {
    if targets.commit_phase_commits.len() != proof.commit_phase_commits.len()
        || targets.commit_pow_witnesses.len() != proof.commit_pow_witnesses.len()
        || targets.query_proofs.len() != proof.query_proofs.len()
        || targets.final_poly.len() != proof.final_poly.len()
    {
        return Err(RecursionError::Plonky2Witness(
            "fixed FRI target counts do not match structured proof".to_string(),
        ));
    }

    for (target, cap) in targets
        .commit_phase_commits
        .iter()
        .zip(proof.commit_phase_commits.iter())
    {
        set_merkle_cap_targets_8(witness, target, cap)?;
    }
    for (target, value) in targets
        .commit_pow_witnesses
        .iter()
        .zip(proof.commit_pow_witnesses.iter())
    {
        witness
            .set_target(*target, F::from_canonical_u64(*value))
            .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    }
    for (target, query) in targets.query_proofs.iter().zip(proof.query_proofs.iter()) {
        set_fixed_fri_query_targets(witness, target, query)?;
    }
    set_extension_targets(witness, &targets.final_poly, &proof.final_poly)?;
    witness
        .set_target(
            targets.query_pow_witness,
            F::from_canonical_u64(proof.query_pow_witness),
        )
        .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    Ok(())
}

/// Assigns one [`FixedFriQueryTargets`] from a [`crate::types::P3FriQueryWitness`].
fn set_fixed_fri_query_targets(
    witness: &mut PartialWitness<F>,
    targets: &FixedFriQueryTargets,
    proof: &crate::types::P3FriQueryWitness,
) -> Result<(), RecursionError> {
    if targets.input_proof.len() != proof.input_proof.len()
        || targets.commit_phase_openings.len() != proof.commit_phase_openings.len()
    {
        return Err(RecursionError::Plonky2Witness(
            "fixed FRI query opening count does not match structured proof".to_string(),
        ));
    }
    for (target, opening) in targets.input_proof.iter().zip(proof.input_proof.iter()) {
        set_fixed_batch_opening_targets(witness, target, opening)?;
    }
    for (target, opening) in targets
        .commit_phase_openings
        .iter()
        .zip(proof.commit_phase_openings.iter())
    {
        set_fixed_commit_phase_opening_targets(witness, target, opening)?;
    }
    Ok(())
}

/// Assigns one [`FixedCommitPhaseOpeningTargets`] from a
/// [`crate::types::P3CommitPhaseProofStepWitness`].
fn set_fixed_commit_phase_opening_targets(
    witness: &mut PartialWitness<F>,
    targets: &FixedCommitPhaseOpeningTargets,
    proof: &crate::types::P3CommitPhaseProofStepWitness,
) -> Result<(), RecursionError> {
    if targets.sibling_values.len() != proof.sibling_values.len()
        || targets.opening_proof.len() != proof.opening_proof.len()
    {
        return Err(RecursionError::Plonky2Witness(
            "fixed FRI commit-phase opening shape does not match structured proof".to_string(),
        ));
    }
    witness
        .set_target(
            targets.log_arity,
            F::from_canonical_usize(proof.log_arity as usize),
        )
        .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    set_extension_targets(witness, &targets.sibling_values, &proof.sibling_values)?;
    for (target_digest, digest) in targets.opening_proof.iter().zip(proof.opening_proof.iter()) {
        for (target, value) in target_digest.iter().zip(digest.iter()) {
            witness
                .set_target(*target, F::from_canonical_u64(*value))
                .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
        }
    }
    Ok(())
}

/// Assigns one [`FixedBatchOpeningTargets`] from a [`crate::types::P3BatchOpeningWitness`].
fn set_fixed_batch_opening_targets(
    witness: &mut PartialWitness<F>,
    targets: &FixedBatchOpeningTargets,
    proof: &crate::types::P3BatchOpeningWitness,
) -> Result<(), RecursionError> {
    if targets.opened_values.len() != proof.opened_values.len()
        || targets.opening_proof.len() != proof.opening_proof.len()
    {
        return Err(RecursionError::Plonky2Witness(
            "fixed batch opening target count does not match structured proof".to_string(),
        ));
    }
    for (target_row, value_row) in targets.opened_values.iter().zip(proof.opened_values.iter()) {
        if target_row.len() != value_row.len() {
            return Err(RecursionError::Plonky2Witness(
                "fixed batch opening row width does not match structured proof".to_string(),
            ));
        }
        for (target, value) in target_row.iter().zip(value_row.iter()) {
            witness
                .set_target(*target, F::from_canonical_u64(*value))
                .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
        }
    }
    for (target_digest, digest) in targets.opening_proof.iter().zip(proof.opening_proof.iter()) {
        for (target, value) in target_digest.iter().zip(digest.iter()) {
            witness
                .set_target(*target, F::from_canonical_u64(*value))
                .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
        }
    }
    Ok(())
}

/// Converts a native `p3_symmetric::MerkleCap` (used here for the
/// preprocessed verifier key's commitment) to [`P3MerkleCapWitness`].
fn merkle_cap_from_p3(
    cap: &p3_symmetric::MerkleCap<p3_goldilocks::Goldilocks, [p3_goldilocks::Goldilocks; 8]>,
) -> P3MerkleCapWitness {
    P3MerkleCapWitness {
        roots: cap
            .roots()
            .iter()
            .map(|root| root.map(|value| value.as_canonical_u64()))
            .collect(),
    }
}

/// Converts a P3 `Goldilocks` value to Plonky2's `GoldilocksField` — both
/// represent the same prime field, just as different Rust types from
/// different crates, so this is purely a canonical-`u64` round trip.
fn goldilocks_from_p3(value: p3_goldilocks::Goldilocks) -> F {
    F::from_canonical_u64(value.as_canonical_u64())
}

/// Converts a `P3Challenge` (P3's degree-2 extension) to Plonky2's
/// extension-field type for the same degree, by re-expressing its basis
/// coefficients in the target type. Fails if the coefficient count doesn't
/// match [`RECURSION_D`] — would only happen if the two crates' extension
/// degrees somehow diverged.
fn p3_challenge_to_plonky2(
    value: P3Challenge,
) -> Result<<F as Extendable<RECURSION_D>>::Extension, RecursionError> {
    let coeffs: &[p3_goldilocks::Goldilocks] = value.as_basis_coefficients_slice();
    if coeffs.len() != RECURSION_D {
        return Err(RecursionError::Plonky2Witness(
            "challenge basis coefficient count does not match recursion extension degree"
                .to_string(),
        ));
    }
    let coeffs: [F; RECURSION_D] = core::array::from_fn(|i| goldilocks_from_p3(coeffs[i]));
    Ok(
        <<F as Extendable<RECURSION_D>>::Extension as Plonky2FieldExtension<RECURSION_D>>::from_basefield_array(
            coeffs,
        ),
    )
}

/// Recomposes a length-[`RECURSION_D`] list of "extension-wrapped base
/// values" (each `coeffs[i]` really only carries a base-field value, used
/// as a uniform `ExtensionTarget` representation for the per-chunk quotient
/// values before they're combined) into one true extension-field value:
/// `sum_i coeffs[i] * e_i` where `e_i` is the `i`-th basis vector.
fn recompose_ext_coefficients_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    coeffs: &[ExtensionTarget<RECURSION_D>],
) -> ExtensionTarget<RECURSION_D> {
    coeffs
        .iter()
        .enumerate()
        .fold(builder.zero_extension(), |acc, (i, coeff)| {
            let basis_coeffs: [F; RECURSION_D] =
                core::array::from_fn(|j| if i == j { F::ONE } else { F::ZERO });
            let basis = builder.constant_extension(
                <F as Extendable<RECURSION_D>>::Extension::from_basefield_array(basis_coeffs),
            );
            let weighted = builder.mul_extension(basis, *coeff);
            builder.add_extension(acc, weighted)
        })
}

/// Native counterpart of [`recompose_ext_coefficients_targets`]: reassembles
/// one [`crate::types::P3FriProofWitness`]-style `[u64; 2]` list (one
/// quotient chunk's per-coefficient encoding) into a true `P3Challenge`.
fn p3_challenge_from_ext_pairs(values: &[[u64; 2]]) -> Result<P3Challenge, RecursionError> {
    let coeffs: Result<Vec<_>, _> = values
        .iter()
        .map(|value| {
            P3Challenge::from_basis_coefficients_slice(&[
                p3_goldilocks::Goldilocks::from_u64(value[0]),
                p3_goldilocks::Goldilocks::from_u64(value[1]),
            ])
            .ok_or_else(|| {
                RecursionError::Plonky2Witness(
                    "extension coefficient pair does not match challenge basis dimension"
                        .to_string(),
                )
            })
        })
        .collect();
    let coeffs = coeffs?;

    <P3Challenge as ExtensionField<p3_goldilocks::Goldilocks>>::from_ext_basis_coefficients(&coeffs)
        .ok_or_else(|| {
            RecursionError::Plonky2Witness(
                "quotient chunk coefficient count does not match challenge dimension".to_string(),
            )
        })
}

#[cfg(test)]
mod tests {
    use p3_ecdsa::batch::BatchProof;
    use p3_ecdsa::testing::deterministic_batch;

    use super::*;

    #[test]
    fn batch_circuit_accepts_valid_and_rejects_mismatched_halves() {
        let capacity = 4;
        let prepared = PreparedBatch::new(capacity);
        let circuit = build_batch_verifier_circuit(capacity).expect("build circuit");
        let (a, pv_a) = prepared.prove(&deterministic_batch(2, 3)).expect("prove a");
        let (b, pv_b) = prepared.prove(&deterministic_batch(2, 4)).expect("prove b");

        let wa = crate::api::prepare_batch_recursive_witness(&prepared, &a, &pv_a).unwrap();
        let wb = crate::api::prepare_batch_recursive_witness(&prepared, &b, &pv_b).unwrap();
        let proof = prove_batch_verifier(&circuit, &prepared, &wa).expect("recursive prove");
        circuit.data.verify(proof).expect("recursive verify");

        // Natively, the mixed pair is rejected before proving.
        let mixed = BatchRecursiveWitness {
            ecdsa: wa.ecdsa.clone(),
            message_hash: wb.message_hash.clone(),
        };
        assert!(prove_batch_verifier(&circuit, &prepared, &mixed).is_err());
        assert!(
            crate::api::prepare_batch_recursive_witness(
                &prepared,
                &BatchProof {
                    ecdsa: a.ecdsa,
                    message_hash: b.message_hash,
                },
                &pv_a,
            )
            .is_err()
        );

        // In-circuit: the keccak verifier reads `[n, L]` from the ECDSA public
        // inputs, so batch b's keccak proof cannot be assigned next to batch a's.
        let mut witness = PartialWitness::new();
        set_public_inputs(
            &mut witness,
            &circuit.public_inputs,
            wa.ecdsa.public_inputs(),
        )
        .unwrap();
        assign_inner_verifier(
            &mut witness,
            &circuit.ecdsa,
            &circuit.ecdsa_metadata,
            &wa.ecdsa,
        )
        .unwrap();
        assign_inner_verifier(
            &mut witness,
            &circuit.message_hash,
            &circuit.message_hash_metadata,
            &wb.message_hash,
        )
        .unwrap();
        assert!(circuit.data.prove(witness).is_err());
    }
}
