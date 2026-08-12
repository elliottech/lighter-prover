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
//! If either check were skipped, an adversary could submit a structurally
//! valid-looking proof object with no actual relationship to a real
//! satisfying trace.
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
//! The two must compute identical results; a `_host` function that's even
//! slightly wrong only breaks proving (the witness won't satisfy the
//! circuit's constraints), whereas a wrong `_targets` function can make the
//! circuit accept something it shouldn't — the `_targets` side is where
//! soundness actually lives.

use core::cell::Cell;
use core::marker::PhantomData;

use p3_air::symbolic::{
    AirLayout, BaseEntry, BaseLeaf, ConstraintLayout, ExtLeaf, SymbolicAirBuilder, SymbolicExpr,
    SymbolicExpression, SymbolicExpressionExt, SymbolicVariableExt,
};
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::PolynomialSpace;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field as _, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField,
};
use p3_fri::{FriFoldingStrategy, TwoAdicFriFolding};
#[cfg(any(test, feature = "insecure-test-only"))]
use p3_schnorr::SignatureBatchPublicInputs;
use p3_schnorr::SignatureBatchPublicInputs as PublicInputs;
use p3_schnorr::air::SignatureBatchAir;
use p3_schnorr::proof::{
    Challenge as P3Challenge, Challenger as P3Challenger, Compress as P3Compress, Hash as P3Hash,
    TRANSCRIPT_POSEIDON_RATE as POSEIDON_RATE, TRANSCRIPT_POSEIDON_WIDTH as POSEIDON_WIDTH,
    schnorr_stark_config,
};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_uni_stark::setup_preprocessed;
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
use crate::types::{P3MerkleCapWitness, PreparedP3RecursiveWitness, StructuredP3Proof};

pub const RECURSION_D: usize = 2;
pub const RECURSIVE_PUBLIC_INPUTS: usize = 1 + p3_schnorr::PUBLIC_DIGEST_WIDTH;

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
/// sampling uses `as_canonical_u64()`, so accepting the second decomposition
/// here would let recursive FRI sampling disagree with the native verifier.
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
    for chunk in state.chunks_exact_mut(4) {
        apply_mat4_targets(
            builder,
            chunk.try_into().expect("Poseidon chunk length must be 4"),
        );
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
    pub degree_bits: usize,
    pub base_degree_bits: usize,
    pub trace_subgroup_generator_inverse: u64,
    pub air_width: usize,
    pub preprocessed_width: usize,
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

    let proof = TranscriptVerifierTargets {
        degree_bits: builder.add_virtual_target(),
        trace_commitment: add_virtual_merkle_cap_8(&mut builder, metadata.trace_cap_roots),
        quotient_chunks_commitment: add_virtual_merkle_cap_8(
            &mut builder,
            metadata.quotient_cap_roots,
        ),
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
        fri: add_fixed_fri_targets(&mut builder, &metadata),
    };

    let expected_degree_bits = builder.constant(F::from_canonical_usize(metadata.degree_bits));
    builder.connect(proof.degree_bits, expected_degree_bits);

    let (transcript_challenger, alpha, zeta) =
        derive_transcript_challenges_targets(&mut builder, &public_inputs, &proof, &metadata);
    let derived_quotient =
        recompose_quotient_chunks_targets(&mut builder, &proof.quotient_chunks, zeta, &metadata);
    let vanishing = z_h_at_zeta_targets(&mut builder, zeta, metadata.base_degree_bits);
    let derived_folded = derive_folded_constraints_targets(
        &mut builder,
        &public_inputs,
        &proof,
        alpha,
        zeta,
        &metadata,
        &symbolic_constraints,
    )?;
    let quotient_times_vanishing = builder.mul_extension(derived_quotient, vanishing);
    builder.connect_extension(derived_quotient, proof.expected_quotient);
    builder.connect_extension(quotient_times_vanishing, derived_folded);
    builder.connect_extension(alpha, proof.expected_alpha);
    builder.connect_extension(zeta, proof.expected_zeta);
    verify_fixed_fri_input_openings_targets(
        &mut builder,
        transcript_challenger,
        zeta,
        &proof,
        &metadata,
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

/// Produces a deliberately insecure proof from the
/// [`InsecurePublicInputShellCircuit`] test/benchmark baseline.
///
/// This only checks the public signature count; it does not verify signatures
/// or an underlying P3 proof.
#[cfg(any(test, feature = "insecure-test-only"))]
pub fn prove_insecure_public_input_shell(
    circuit: &InsecurePublicInputShellCircuit,
    public_inputs: &SignatureBatchPublicInputs,
) -> Result<ProofWithPublicInputs<F, C, RECURSION_D>, RecursionError> {
    if public_inputs.signature_count() != FIXED_RECURSIVE_SIGNATURE_COUNT {
        return Err(RecursionError::InvalidSignatureCount {
            expected: FIXED_RECURSIVE_SIGNATURE_COUNT,
            actual: public_inputs.signature_count(),
        });
    }

    let values = public_inputs.public_values();
    let mut witness = PartialWitness::new();
    for (target, value) in circuit.public_inputs.iter().zip(values.into_iter()) {
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
    witness_data: &PreparedP3RecursiveWitness,
) -> Result<ProofWithPublicInputs<F, C, RECURSION_D>, RecursionError> {
    crate::api::verify_prepared_recursive_witness(
        witness_data,
        circuit.expected_signature_count(),
    )?;

    let public_inputs = witness_data.public_inputs();
    let (alpha, zeta) = derive_transcript_challenges_host(witness_data, &circuit.metadata)?;
    let quotient =
        recompose_quotient_chunks_host(witness_data.structured_proof(), zeta, &circuit.metadata)?;
    let mut witness = PartialWitness::new();
    for (target, value) in circuit
        .public_inputs
        .iter()
        .zip(public_inputs.public_values().into_iter())
    {
        witness
            .set_target(*target, goldilocks_from_p3(value))
            .map_err(|err| RecursionError::Plonky2Witness(err.to_string()))?;
    }
    set_transcript_verifier_targets(
        &mut witness,
        &circuit.proof,
        &circuit.metadata,
        witness_data.structured_proof(),
        quotient,
        alpha,
        zeta,
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
    let public_inputs = PublicInputs::new(
        signature_count,
        [p3_goldilocks::Goldilocks::ZERO; p3_schnorr::PUBLIC_DIGEST_WIDTH],
    )?;
    let rows = public_inputs.expected_rows();
    let degree_bits = rows.ilog2() as usize;
    let air = SignatureBatchAir::new(rows);
    let config = schnorr_stark_config();
    let (_, preprocessed_vk) =
        setup_preprocessed::<p3_schnorr::proof::SignatureBatchStarkConfig, _>(
            &config,
            &air,
            degree_bits,
        )
        .ok_or(RecursionError::InvariantViolation(
            "fixed Schnorr preprocessed setup must exist",
        ))?;
    let air_width = <SignatureBatchAir as BaseAir<p3_goldilocks::Goldilocks>>::width(&air);
    let preprocessed_width =
        <SignatureBatchAir as BaseAir<p3_goldilocks::Goldilocks>>::preprocessed_width(&air);
    let trace_next_width =
        if <SignatureBatchAir as BaseAir<p3_goldilocks::Goldilocks>>::main_next_row_columns(&air)
            .is_empty()
        {
            0
        } else {
            air_width
        };
    let preprocessed_next_width =
        if <SignatureBatchAir as BaseAir<p3_goldilocks::Goldilocks>>::preprocessed_next_row_columns(
            &air,
        )
        .is_empty()
        {
            0
        } else {
            preprocessed_width
        };
    let constraint_degree =
        <SignatureBatchAir as BaseAir<p3_goldilocks::Goldilocks>>::max_constraint_degree(&air)
            .unwrap_or(2)
            .max(2);
    let log_num_quotient_chunks = (constraint_degree - 1).next_power_of_two().ilog2() as usize;
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
    let fri_global_log_height = degree_bits + p3_schnorr::proof::PLONKY2_FRI_RATE_BITS;
    let fri_final_log_height =
        p3_schnorr::proof::PLONKY2_FRI_RATE_BITS + p3_schnorr::proof::PLONKY2_FRI_FINAL_POLY_BITS;
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

    Ok(FixedVerifierMetadata {
        degree_bits,
        base_degree_bits: degree_bits,
        trace_subgroup_generator_inverse: trace_domain
            .subgroup_generator()
            .inverse()
            .as_canonical_u64(),
        air_width,
        preprocessed_width,
        trace_next_width,
        preprocessed_next_width,
        num_quotient_chunks,
        quotient_chunk_domains: fixed_chunk_domains,
        preprocessed_commitment: merkle_cap_from_p3(&preprocessed_vk.commitment),
        trace_cap_roots: 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        quotient_cap_roots: 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        fri_cap_roots: 1 << p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        fri_log_blowup: p3_schnorr::proof::PLONKY2_FRI_RATE_BITS,
        fri_query_pow_bits: p3_schnorr::proof::PLONKY2_FRI_PROOF_OF_WORK_BITS,
        fri_commit_pow_bits: 0,
        fri_query_count: p3_schnorr::proof::PLONKY2_FRI_NUM_QUERY_ROUNDS,
        fri_final_poly_len: 1
            << (p3_schnorr::proof::PLONKY2_FRI_RATE_BITS
                + p3_schnorr::proof::PLONKY2_FRI_FINAL_POLY_BITS
                - p3_schnorr::proof::PLONKY2_FRI_RATE_BITS),
        fri_input_batch_widths: vec![
            vec![air_width],
            vec![RECURSION_D; num_quotient_chunks],
            vec![preprocessed_width],
        ],
        fri_input_opening_proof_len: degree_bits + p3_schnorr::proof::PLONKY2_FRI_RATE_BITS
            - p3_schnorr::proof::PLONKY2_FRI_CAP_HEIGHT,
        fri_commit_phase_log_arities,
        fri_commit_phase_opening_proof_lens,
        fri_final_log_height,
    })
}

/// Replays the *entire* Fiat-Shamir transcript natively — `alpha`, `zeta`,
/// the FRI `beta`s, the query proof-of-work check, and finally the query
/// indices — continuing past where [`derive_transcript_challenges_host`]
/// stops (that function only needs `alpha`/`zeta` for production proving).
///
/// Not called by [`prove_transcript_verifier`] or any other production
/// path (`#[allow(dead_code)]`): this exists as an independent native
/// oracle for the `tests` module to cross-check
/// [`derive_transcript_challenges_targets`]/the FRI `_targets` functions
/// against, and to let tests target a specific query index without needing
/// a running circuit.
#[allow(dead_code)]
fn derive_fri_query_indices_host(
    witness: &PreparedP3RecursiveWitness,
    metadata: &FixedVerifierMetadata,
) -> Result<Vec<usize>, RecursionError> {
    let public_inputs = witness.public_inputs();
    let structured = witness.structured_proof();
    let mut challenger = P3Challenger::new(p3_goldilocks::default_goldilocks_poseidon2_16());
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        structured.degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.base_degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.preprocessed_width,
    ));
    observe_merkle_cap_8_host(&mut challenger, &structured.trace_commitment);
    observe_merkle_cap_8_host(&mut challenger, &metadata.preprocessed_commitment);
    challenger.observe_slice(&public_inputs.public_values());
    let _oods_alpha: P3Challenge = challenger.sample_algebra_element();
    observe_merkle_cap_8_host(&mut challenger, &structured.quotient_chunks_commitment);
    let _zeta: P3Challenge = challenger.sample_algebra_element();

    observe_extension_rows_host(&mut challenger, &structured.trace_local);
    observe_extension_rows_host(
        &mut challenger,
        structured.trace_next.as_deref().unwrap_or(&[]),
    );
    for chunk in &structured.quotient_chunks {
        observe_extension_rows_host(&mut challenger, chunk);
    }
    observe_extension_rows_host(
        &mut challenger,
        structured.preprocessed_local.as_deref().unwrap_or(&[]),
    );
    observe_extension_rows_host(
        &mut challenger,
        structured.preprocessed_next.as_deref().unwrap_or(&[]),
    );
    let _fri_alpha: P3Challenge = challenger.sample_algebra_element();

    for cap in &structured.opening_proof.commit_phase_commits {
        observe_merkle_cap_8_host(&mut challenger, cap);
        let _beta: P3Challenge = challenger.sample_algebra_element();
    }

    observe_extension_rows_host(&mut challenger, &structured.opening_proof.final_poly);
    for _ in 0..structured.opening_proof.commit_phase_commits.len() {
        challenger.observe(p3_goldilocks::Goldilocks::from_usize(
            p3_schnorr::proof::PLONKY2_FRI_REDUCTION_ARITY_BITS,
        ));
    }
    if metadata.fri_query_pow_bits > 0
        && !challenger.check_witness(
            metadata.fri_query_pow_bits,
            p3_goldilocks::Goldilocks::from_u64(structured.opening_proof.query_pow_witness),
        )
    {
        return Err(RecursionError::Plonky2Witness(
            "structured FRI query PoW witness is invalid".to_string(),
        ));
    }

    let query_bits = metadata.degree_bits + metadata.fri_log_blowup;
    Ok((0..metadata.fri_query_count)
        .map(|_| challenger.sample_bits(query_bits))
        .collect())
}

/// Native (test-only, see [`derive_fri_query_indices_host`]) cross-check
/// that every FRI query's three input batch openings (trace, quotient
/// chunks, preprocessed) verify against their respective Merkle caps at the
/// transcript-derived query index.
#[allow(dead_code)]
fn verify_fixed_fri_input_openings_host(
    witness: &PreparedP3RecursiveWitness,
    metadata: &FixedVerifierMetadata,
) -> Result<(), RecursionError> {
    let proof = witness.structured_proof();
    let query_indices = derive_fri_query_indices_host(witness, metadata)?;
    if query_indices.len() != proof.opening_proof.query_proofs.len() {
        return Err(RecursionError::Plonky2Witness(
            "structured FRI query count does not match transcript-derived query indices"
                .to_string(),
        ));
    }

    for (query, &index) in proof
        .opening_proof
        .query_proofs
        .iter()
        .zip(query_indices.iter())
    {
        if query.input_proof.len() != metadata.fri_input_batch_widths.len() {
            return Err(RecursionError::Plonky2Witness(
                "structured FRI input proof batch count does not match fixed metadata".to_string(),
            ));
        }

        verify_fixed_batch_opening_host(
            &proof.trace_commitment,
            &query.input_proof[0],
            index,
            metadata.fri_input_opening_proof_len,
            metadata.fri_cap_roots,
        )?;
        verify_fixed_batch_opening_host(
            &proof.quotient_chunks_commitment,
            &query.input_proof[1],
            index,
            metadata.fri_input_opening_proof_len,
            metadata.fri_cap_roots,
        )?;
        verify_fixed_batch_opening_host(
            &metadata.preprocessed_commitment,
            &query.input_proof[2],
            index,
            metadata.fri_input_opening_proof_len,
            metadata.fri_cap_roots,
        )?;
    }

    Ok(())
}

/// Native (test-only) Merkle path verification: hashes `opening.opened_values`
/// to a leaf digest, then folds in each sibling along `opening.opening_proof`
/// (using `index`'s bits to pick left/right at each level) and checks the
/// result lands in `cap`.
#[allow(dead_code)]
fn verify_fixed_batch_opening_host(
    cap: &P3MerkleCapWitness,
    opening: &crate::types::P3BatchOpeningWitness,
    index: usize,
    expected_path_len: usize,
    cap_roots: usize,
) -> Result<(), RecursionError> {
    if opening.opening_proof.len() != expected_path_len {
        return Err(RecursionError::Plonky2Witness(
            "structured batch opening proof length does not match fixed metadata".to_string(),
        ));
    }
    if cap.roots.len() != cap_roots {
        return Err(RecursionError::Plonky2Witness(
            "structured Merkle cap size does not match fixed metadata".to_string(),
        ));
    }

    let hash = P3Hash::new(p3_goldilocks::default_goldilocks_poseidon2_16());
    let compress = P3Compress::new(p3_goldilocks::default_goldilocks_poseidon2_16());
    let flat: Vec<_> = opening
        .opened_values
        .iter()
        .flat_map(|row| row.iter().copied())
        .map(p3_goldilocks::Goldilocks::from_u64)
        .collect();
    let mut digest = hash.hash_iter(flat);

    let mut idx = index;
    for sibling in &opening.opening_proof {
        let sibling = sibling.map(p3_goldilocks::Goldilocks::from_u64);
        digest = if idx & 1 == 0 {
            compress.compress([digest, sibling])
        } else {
            compress.compress([sibling, digest])
        };
        idx >>= 1;
    }

    if idx >= cap.roots.len() {
        return Err(RecursionError::Plonky2Witness(
            "derived Merkle cap index is out of bounds".to_string(),
        ));
    }
    if digest != cap.roots[idx].map(p3_goldilocks::Goldilocks::from_u64) {
        return Err(RecursionError::Plonky2Witness(
            "fixed MMCS batch opening verification failed".to_string(),
        ));
    }
    Ok(())
}

/// Native (test-only, see [`derive_fri_query_indices_host`]) full FRI
/// verifier: for each query, computes the initial reduced-opening
/// evaluation, then folds it round-by-round (verifying each round's commit-
/// phase Merkle opening as it goes), and finally checks the fully-folded
/// value matches the committed final polynomial evaluated at the
/// query's final-round point.
#[allow(dead_code)]
fn verify_fixed_fri_host(
    witness: &PreparedP3RecursiveWitness,
    metadata: &FixedVerifierMetadata,
) -> Result<(), RecursionError> {
    let proof = witness.structured_proof();
    let query_indices = derive_fri_query_indices_host(witness, metadata)?;
    let (_, zeta) = derive_transcript_challenges_host(witness, metadata)?;
    let fri_alpha = derive_fri_alpha_host(witness, metadata)?;
    let fri_betas = derive_fri_betas_host(witness, metadata)?;
    let folding = TwoAdicFriFolding::<Vec<crate::types::P3BatchOpeningWitness>, ()>(PhantomData);

    for ((query, &query_index), round_betas) in proof
        .opening_proof
        .query_proofs
        .iter()
        .zip(query_indices.iter())
        .zip(core::iter::repeat(&fri_betas))
    {
        let mut folded_eval = compute_initial_fri_reduced_opening_host(
            proof,
            query,
            query_index,
            fri_alpha,
            zeta,
            metadata,
        )?;
        let mut current_index = query_index;
        let mut current_log_height = metadata.degree_bits + metadata.fri_log_blowup;

        for (((opening, commit_cap), &beta), (&log_arity, &proof_len)) in query
            .commit_phase_openings
            .iter()
            .zip(proof.opening_proof.commit_phase_commits.iter())
            .zip(round_betas.iter())
            .zip(
                metadata
                    .fri_commit_phase_log_arities
                    .iter()
                    .zip(metadata.fri_commit_phase_opening_proof_lens.iter()),
            )
        {
            if opening.log_arity as usize != log_arity {
                return Err(RecursionError::Plonky2Witness(
                    "structured FRI commit-phase log arity does not match fixed metadata"
                        .to_string(),
                ));
            }
            let arity = 1usize << log_arity;
            let index_in_group = current_index & (arity - 1);
            let parent_index = current_index >> log_arity;
            let row = reconstruct_commit_phase_row_host(
                index_in_group,
                &opening.sibling_values,
                folded_eval,
            )?;
            verify_fixed_commit_phase_opening_host(
                commit_cap,
                parent_index,
                proof_len,
                metadata.fri_cap_roots,
                &row,
                opening,
            )?;
            let log_folded_height = current_log_height - log_arity;
            folded_eval =
                <TwoAdicFriFolding<Vec<crate::types::P3BatchOpeningWitness>, ()> as FriFoldingStrategy<
                    p3_goldilocks::Goldilocks,
                    P3Challenge,
                >>::fold_row(
                    &folding,
                    parent_index,
                    log_folded_height,
                    log_arity,
                    beta,
                    row.into_iter(),
                );
            current_index = parent_index;
            current_log_height = log_folded_height;
        }

        if current_log_height != metadata.fri_final_log_height {
            return Err(RecursionError::Plonky2Witness(
                "fixed FRI host final height does not match metadata".to_string(),
            ));
        }

        let final_x = p3_goldilocks::Goldilocks::two_adic_generator(
            metadata.degree_bits + metadata.fri_log_blowup,
        )
        .exp_u64(reverse_bits_len_usize(
            current_index,
            metadata.degree_bits + metadata.fri_log_blowup,
        ) as u64);
        let mut final_eval = P3Challenge::ZERO;
        for &coeff in proof.opening_proof.final_poly.iter().rev() {
            final_eval = final_eval * final_x + p3_challenge_from_pair(coeff)?;
        }
        if final_eval != folded_eval {
            return Err(RecursionError::Plonky2Witness(
                "fixed FRI final polynomial evaluation mismatch".to_string(),
            ));
        }
    }

    Ok(())
}

/// Native (test-only) re-derivation of the FRI folding challenge `alpha`,
/// continuing the transcript past `zeta` through the opened trace/quotient/
/// preprocessed values — the challenge [`compute_initial_fri_reduced_opening_host`]
/// combines DEEP-quotient terms with.
#[allow(dead_code)]
fn derive_fri_alpha_host(
    witness: &PreparedP3RecursiveWitness,
    metadata: &FixedVerifierMetadata,
) -> Result<P3Challenge, RecursionError> {
    let public_inputs = witness.public_inputs();
    let structured = witness.structured_proof();
    let mut challenger = P3Challenger::new(p3_goldilocks::default_goldilocks_poseidon2_16());
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        structured.degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.base_degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.preprocessed_width,
    ));
    observe_merkle_cap_8_host(&mut challenger, &structured.trace_commitment);
    observe_merkle_cap_8_host(&mut challenger, &metadata.preprocessed_commitment);
    challenger.observe_slice(&public_inputs.public_values());
    let _oods_alpha: P3Challenge = challenger.sample_algebra_element();
    observe_merkle_cap_8_host(&mut challenger, &structured.quotient_chunks_commitment);
    let _zeta: P3Challenge = challenger.sample_algebra_element();
    observe_extension_rows_host(&mut challenger, &structured.trace_local);
    observe_extension_rows_host(
        &mut challenger,
        structured.trace_next.as_deref().unwrap_or(&[]),
    );
    for chunk in &structured.quotient_chunks {
        observe_extension_rows_host(&mut challenger, chunk);
    }
    observe_extension_rows_host(
        &mut challenger,
        structured.preprocessed_local.as_deref().unwrap_or(&[]),
    );
    observe_extension_rows_host(
        &mut challenger,
        structured.preprocessed_next.as_deref().unwrap_or(&[]),
    );
    Ok(challenger.sample_algebra_element())
}

/// Native (test-only) re-derivation of every FRI commit-phase folding
/// challenge `beta`, one per round, sampled in order after observing each
/// round's Merkle cap.
#[allow(dead_code)]
fn derive_fri_betas_host(
    witness: &PreparedP3RecursiveWitness,
    metadata: &FixedVerifierMetadata,
) -> Result<Vec<P3Challenge>, RecursionError> {
    let public_inputs = witness.public_inputs();
    let structured = witness.structured_proof();
    let mut challenger = P3Challenger::new(p3_goldilocks::default_goldilocks_poseidon2_16());
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        structured.degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.base_degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.preprocessed_width,
    ));
    observe_merkle_cap_8_host(&mut challenger, &structured.trace_commitment);
    observe_merkle_cap_8_host(&mut challenger, &metadata.preprocessed_commitment);
    challenger.observe_slice(&public_inputs.public_values());
    let _oods_alpha: P3Challenge = challenger.sample_algebra_element();
    observe_merkle_cap_8_host(&mut challenger, &structured.quotient_chunks_commitment);
    let _zeta: P3Challenge = challenger.sample_algebra_element();
    observe_extension_rows_host(&mut challenger, &structured.trace_local);
    observe_extension_rows_host(
        &mut challenger,
        structured.trace_next.as_deref().unwrap_or(&[]),
    );
    for chunk in &structured.quotient_chunks {
        observe_extension_rows_host(&mut challenger, chunk);
    }
    observe_extension_rows_host(
        &mut challenger,
        structured.preprocessed_local.as_deref().unwrap_or(&[]),
    );
    observe_extension_rows_host(
        &mut challenger,
        structured.preprocessed_next.as_deref().unwrap_or(&[]),
    );
    let _fri_alpha: P3Challenge = challenger.sample_algebra_element();
    Ok(structured
        .opening_proof
        .commit_phase_commits
        .iter()
        .map(|cap| {
            observe_merkle_cap_8_host(&mut challenger, cap);
            Ok(challenger.sample_algebra_element())
        })
        .collect::<Result<Vec<_>, RecursionError>>()?)
}

/// Native (test-only) DEEP/FRI reduced-opening computation for one query:
/// for every opened value (trace local/next, each quotient chunk,
/// preprocessed local/next), accumulates
/// `alpha_pow * (value_at_zeta_or_zeta_next - opened) / (sample_point - x)`
/// where `x` is this query's evaluation point in the (blown-up) domain —
/// this is the value FRI's commit-phase folding then operates on. Must
/// combine terms in the same order [`compute_initial_fri_reduced_opening_targets`]
/// does, since `alpha_pow` is threaded sequentially across all of them.
///
/// Important: `query.input_proof[k].opened_values` is indexed by matrix, not
/// by OODS point. A `p3_commit::BatchOpening` carries one queried row per
/// committed matrix. For the trace batch, `opened_values[0]` is therefore the
/// one committed trace row at the sampled query point `x`, and it is
/// intentionally reused against both `proof.trace_local` (`zeta`) and
/// `proof.trace_next` (`zeta * g`). The preprocessed batch works the same way.
/// This matches native P3 DEEP reduction, which adds one
/// `(f(z) - f(x)) / (z - x)` term for each claimed opening point `z` of the
/// same committed row value `f(x)`.
#[allow(dead_code)]
fn compute_initial_fri_reduced_opening_host(
    proof: &StructuredP3Proof,
    query: &crate::types::P3FriQueryWitness,
    query_index: usize,
    fri_alpha: P3Challenge,
    zeta: P3Challenge,
    metadata: &FixedVerifierMetadata,
) -> Result<P3Challenge, RecursionError> {
    let x = P3Challenge::from(p3_goldilocks::Goldilocks::GENERATOR)
        * P3Challenge::from(
            p3_goldilocks::Goldilocks::two_adic_generator(
                metadata.degree_bits + metadata.fri_log_blowup,
            )
            .exp_u64(reverse_bits_len_usize(
                query_index,
                metadata.degree_bits + metadata.fri_log_blowup,
            ) as u64),
        );
    let subgroup_generator =
        p3_goldilocks::Goldilocks::from_u64(metadata.trace_subgroup_generator_inverse).inverse();
    let zeta_next = zeta * subgroup_generator;
    let mut alpha_pow = P3Challenge::ONE;
    let mut reduced_opening = P3Challenge::ZERO;

    for (&opened, &at_zeta) in query.input_proof[0].opened_values[0]
        .iter()
        .zip(proof.trace_local.iter())
    {
        reduced_opening += alpha_pow
            * (p3_challenge_from_pair(at_zeta)?
                - P3Challenge::from(p3_goldilocks::Goldilocks::from_u64(opened)))
            * (zeta - x).inverse();
        alpha_pow *= fri_alpha;
    }
    for (&opened, &at_next) in query.input_proof[0].opened_values[0]
        .iter()
        .zip(proof.trace_next.as_deref().unwrap_or(&[]).iter())
    {
        reduced_opening += alpha_pow
            * (p3_challenge_from_pair(at_next)?
                - P3Challenge::from(p3_goldilocks::Goldilocks::from_u64(opened)))
            * (zeta_next - x).inverse();
        alpha_pow *= fri_alpha;
    }
    for (opened_row, at_zeta_row) in query.input_proof[1]
        .opened_values
        .iter()
        .zip(proof.quotient_chunks.iter())
    {
        for (&opened, &at_zeta_value) in opened_row.iter().zip(at_zeta_row.iter()) {
            reduced_opening += alpha_pow
                * (p3_challenge_from_pair(at_zeta_value)?
                    - P3Challenge::from(p3_goldilocks::Goldilocks::from_u64(opened)))
                * (zeta - x).inverse();
            alpha_pow *= fri_alpha;
        }
    }
    for (&opened, &at_zeta) in query.input_proof[2].opened_values[0]
        .iter()
        .zip(proof.preprocessed_local.as_deref().unwrap_or(&[]).iter())
    {
        reduced_opening += alpha_pow
            * (p3_challenge_from_pair(at_zeta)?
                - P3Challenge::from(p3_goldilocks::Goldilocks::from_u64(opened)))
            * (zeta - x).inverse();
        alpha_pow *= fri_alpha;
    }
    for (&opened, &at_next) in query.input_proof[2].opened_values[0]
        .iter()
        .zip(proof.preprocessed_next.as_deref().unwrap_or(&[]).iter())
    {
        reduced_opening += alpha_pow
            * (p3_challenge_from_pair(at_next)?
                - P3Challenge::from(p3_goldilocks::Goldilocks::from_u64(opened)))
            * (zeta_next - x).inverse();
        alpha_pow *= fri_alpha;
    }

    Ok(reduced_opening)
}

/// Native (test-only) reassembly of one commit-phase folding round's full
/// row (`folded_eval` plus its siblings, in their original group order)
/// from the proof's flat `sibling_values` list and the position
/// (`index_in_group`) `folded_eval` itself belongs at.
#[allow(dead_code)]
fn reconstruct_commit_phase_row_host(
    index_in_group: usize,
    sibling_values: &[[u64; 2]],
    folded_eval: P3Challenge,
) -> Result<Vec<P3Challenge>, RecursionError> {
    let arity = sibling_values.len() + 1;
    let mut row = Vec::with_capacity(arity);
    let mut sibling_idx = 0usize;
    for slot in 0..arity {
        if slot == index_in_group {
            row.push(folded_eval);
        } else {
            row.push(p3_challenge_from_pair(sibling_values[sibling_idx])?);
            sibling_idx += 1;
        }
    }
    Ok(row)
}

/// Native (test-only) commit-phase Merkle check: flattens one folding
/// round's reconstructed row into a single leaf and delegates to
/// [`verify_fixed_batch_opening_host`].
#[allow(dead_code)]
fn verify_fixed_commit_phase_opening_host(
    cap: &P3MerkleCapWitness,
    parent_index: usize,
    expected_path_len: usize,
    cap_roots: usize,
    row: &[P3Challenge],
    opening: &crate::types::P3CommitPhaseProofStepWitness,
) -> Result<(), RecursionError> {
    let flat_row: Vec<u64> = row
        .iter()
        .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
        .map(|limb: p3_goldilocks::Goldilocks| limb.as_canonical_u64())
        .collect();
    let batch_opening = crate::types::P3BatchOpeningWitness {
        opened_values: vec![flat_row],
        opening_proof: opening.opening_proof.clone(),
    };
    verify_fixed_batch_opening_host(
        cap,
        &batch_opening,
        parent_index,
        expected_path_len,
        cap_roots,
    )
}

/// Reassembles a `[u64; 2]` limb pair (this crate's wire encoding of a
/// degree-2 extension element) back into a real `P3Challenge`.
#[allow(dead_code)]
fn p3_challenge_from_pair(value: [u64; 2]) -> Result<P3Challenge, RecursionError> {
    P3Challenge::from_basis_coefficients_slice(&[
        p3_goldilocks::Goldilocks::from_u64(value[0]),
        p3_goldilocks::Goldilocks::from_u64(value[1]),
    ])
    .ok_or_else(|| {
        RecursionError::Plonky2Witness(
            "challenge pair does not match challenge basis dimension".to_string(),
        )
    })
}

/// Observes a slice of `[u64; 2]`-encoded extension elements into the
/// native challenger, limb-wise.
#[allow(dead_code)]
fn observe_extension_rows_host(challenger: &mut P3Challenger, values: &[[u64; 2]]) {
    for value in values {
        challenger.observe_slice(&[
            p3_goldilocks::Goldilocks::from_u64(value[0]),
            p3_goldilocks::Goldilocks::from_u64(value[1]),
        ]);
    }
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
    ExtensionTarget<RECURSION_D>,
    ExtensionTarget<RECURSION_D>,
) {
    let mut challenger = P3RecursiveDuplexChallenger::new(builder);
    let base_degree_bits = builder.constant(F::from_canonical_usize(metadata.base_degree_bits));
    let preprocessed_width = builder.constant(F::from_canonical_usize(metadata.preprocessed_width));
    challenger.observe_element(builder, proof.degree_bits);
    challenger.observe_element(builder, base_degree_bits);
    challenger.observe_element(builder, preprocessed_width);
    observe_merkle_cap_8(&mut challenger, builder, &proof.trace_commitment);
    observe_merkle_cap_8_constant(&mut challenger, builder, &metadata.preprocessed_commitment);
    challenger.observe_elements(builder, public_inputs);
    let alpha = challenger.get_extension_challenge(builder);
    observe_merkle_cap_8(&mut challenger, builder, &proof.quotient_chunks_commitment);
    let zeta = challenger.get_extension_challenge(builder);
    (challenger, alpha, zeta)
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
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.base_degree_bits,
    ));
    challenger.observe(p3_goldilocks::Goldilocks::from_usize(
        metadata.preprocessed_width,
    ));
    observe_merkle_cap_8_host(&mut challenger, &structured.trace_commitment);
    observe_merkle_cap_8_host(&mut challenger, &metadata.preprocessed_commitment);
    challenger.observe_slice(&public_inputs.public_values());
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
    alpha: ExtensionTarget<RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
    symbolic_constraints: &SymbolicAirConstraints,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    let selectors = lagrange_selectors_targets(builder, zeta, metadata);
    let ctx = SymbolicEvaluationContext {
        trace_local: &proof.trace_local,
        trace_next: &proof.trace_next,
        preprocessed_local: &proof.preprocessed_local,
        preprocessed_next: &proof.preprocessed_next,
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
        ExtLeaf::ExtVariable(SymbolicVariableExt { .. }) => {
            Err(RecursionError::UnsupportedVerifierFeature(
                "extension symbolic variables in AIR constraints",
            ))
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
    let rows = 1usize << metadata.base_degree_bits;
    let air = SignatureBatchAir::new(rows);
    let layout = AirLayout::from_air::<p3_goldilocks::Goldilocks>(&air);
    let mut builder = SymbolicAirBuilder::<p3_goldilocks::Goldilocks, P3Challenge>::new(layout);
    air.eval(&mut builder);
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
    for chunk in &proof.quotient_chunks {
        observe_extension_targets(builder, &mut challenger, chunk);
    }
    observe_extension_targets(builder, &mut challenger, &proof.preprocessed_local);
    observe_extension_targets(builder, &mut challenger, &proof.preprocessed_next);
    let fri_alpha = challenger.get_extension_challenge(builder);
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
    metadata: &FixedVerifierMetadata,
) -> Result<(), RecursionError> {
    let trace_query = query.input_proof.first().ok_or_else(|| {
        RecursionError::Plonky2Witness(
            "fixed FRI query is missing the trace input batch opening".to_string(),
        )
    })?;
    verify_fixed_batch_opening_targets(builder, &proof.trace_commitment, trace_query, query_bits);

    let quotient_query = query.input_proof.get(1).ok_or_else(|| {
        RecursionError::Plonky2Witness(
            "fixed FRI query is missing the quotient input batch opening".to_string(),
        )
    })?;
    verify_fixed_batch_opening_targets(
        builder,
        &proof.quotient_chunks_commitment,
        quotient_query,
        query_bits,
    );

    let preprocessed_query = query.input_proof.get(2).ok_or_else(|| {
        RecursionError::Plonky2Witness(
            "fixed FRI query is missing the preprocessed input batch opening".to_string(),
        )
    })?;
    verify_fixed_batch_opening_targets_constant_cap(
        builder,
        &metadata.preprocessed_commitment,
        preprocessed_query,
        query_bits,
    );

    let mut folded_eval = compute_initial_fri_reduced_opening_targets(
        builder, proof, query, query_bits, fri_alpha, zeta, metadata,
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

/// In-circuit counterpart of [`compute_initial_fri_reduced_opening_host`].
///
/// The repeated use of `query.input_proof[*].opened_values[0]` for both local
/// and next-row OODS terms is intentional. MMCS openings expose one queried row
/// per matrix, not one row per opening point, so the same committed row is
/// compared against both `zeta` and `zeta * g` with different numerators and
/// denominators.
#[allow(dead_code)]
fn compute_initial_fri_reduced_opening_targets(
    builder: &mut CircuitBuilder<F, RECURSION_D>,
    proof: &TranscriptVerifierTargets,
    query: &FixedFriQueryTargets,
    query_bits: &[BoolTarget],
    fri_alpha: ExtensionTarget<RECURSION_D>,
    zeta: ExtensionTarget<RECURSION_D>,
    metadata: &FixedVerifierMetadata,
) -> Result<ExtensionTarget<RECURSION_D>, RecursionError> {
    let x = coset_point_targets(
        builder,
        query_bits,
        metadata.degree_bits + metadata.fri_log_blowup,
    );
    let trace_next_point = mul_by_subgroup_generator_targets(builder, zeta, metadata);

    // Compute the two distinct denominators once and invert each once, then
    // reuse across all columns via multiplication. This replaces O(N)
    // div_extension calls (each requiring an inversion) with 2 inversions
    // + O(N) multiplications.
    let denom_zeta = builder.sub_extension(zeta, x);
    let denom_next = builder.sub_extension(trace_next_point, x);
    let inv_denom_zeta = builder.inverse_extension(denom_zeta);
    let inv_denom_next = builder.inverse_extension(denom_next);

    let mut alpha_pow = builder.one_extension();
    let mut reduced_opening = builder.zero_extension();

    for (&opened, &at_zeta) in query.input_proof[0].opened_values[0]
        .iter()
        .zip(proof.trace_local.iter())
    {
        let opened_ext = builder.convert_to_ext(opened);
        let numerator = builder.sub_extension(at_zeta, opened_ext);
        let quotient = builder.mul_extension(numerator, inv_denom_zeta);
        let weighted = builder.mul_extension(alpha_pow, quotient);
        reduced_opening = builder.add_extension(reduced_opening, weighted);
        alpha_pow = builder.mul_extension(alpha_pow, fri_alpha);
    }
    for (&opened, &at_next) in query.input_proof[0].opened_values[0]
        .iter()
        .zip(proof.trace_next.iter())
    {
        let opened_ext = builder.convert_to_ext(opened);
        let numerator = builder.sub_extension(at_next, opened_ext);
        let quotient = builder.mul_extension(numerator, inv_denom_next);
        let weighted = builder.mul_extension(alpha_pow, quotient);
        reduced_opening = builder.add_extension(reduced_opening, weighted);
        alpha_pow = builder.mul_extension(alpha_pow, fri_alpha);
    }

    for (opened_row, at_zeta_row) in query.input_proof[1]
        .opened_values
        .iter()
        .zip(proof.quotient_chunks.iter())
    {
        for (&opened, &at_zeta_value) in opened_row.iter().zip(at_zeta_row.iter()) {
            let opened_ext = builder.convert_to_ext(opened);
            let numerator = builder.sub_extension(at_zeta_value, opened_ext);
            let quotient = builder.mul_extension(numerator, inv_denom_zeta);
            let weighted = builder.mul_extension(alpha_pow, quotient);
            reduced_opening = builder.add_extension(reduced_opening, weighted);
            alpha_pow = builder.mul_extension(alpha_pow, fri_alpha);
        }
    }

    for (&opened, &at_zeta) in query.input_proof[2].opened_values[0]
        .iter()
        .zip(proof.preprocessed_local.iter())
    {
        let opened_ext = builder.convert_to_ext(opened);
        let numerator = builder.sub_extension(at_zeta, opened_ext);
        let quotient = builder.mul_extension(numerator, inv_denom_zeta);
        let weighted = builder.mul_extension(alpha_pow, quotient);
        reduced_opening = builder.add_extension(reduced_opening, weighted);
        alpha_pow = builder.mul_extension(alpha_pow, fri_alpha);
    }

    for (&opened, &at_next) in query.input_proof[2].opened_values[0]
        .iter()
        .zip(proof.preprocessed_next.iter())
    {
        let opened_ext = builder.convert_to_ext(opened);
        let numerator = builder.sub_extension(at_next, opened_ext);
        let quotient = builder.mul_extension(numerator, inv_denom_next);
        let weighted = builder.mul_extension(alpha_pow, quotient);
        reduced_opening = builder.add_extension(reduced_opening, weighted);
        alpha_pow = builder.mul_extension(alpha_pow, fri_alpha);
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

#[cfg(all(test, feature = "reference"))]
mod tests {
    use std::fs;
    use std::marker::PhantomData;
    use std::path::PathBuf;

    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
    use p3_fri::{FriFoldingStrategy, TwoAdicFriFolding};
    use p3_schnorr::reference::deterministic_batch;
    use p3_schnorr::{SignatureBatchProvingOutput, prove_signature_batch};
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;

    use super::{
        C, F, FixedBatchOpeningTargets, P3Challenge, P3Compress, P3Hash, RECURSION_D,
        add_virtual_merkle_cap_8, canonical_goldilocks_low_bits,
        compute_initial_fri_reduced_opening_host, compute_initial_fri_reduced_opening_targets,
        constrain_canonical_goldilocks_bits, derive_fri_alpha_host, derive_fri_betas_host,
        derive_fri_query_indices_host, derive_transcript_challenges_host,
        evaluate_final_poly_targets, fixed_verifier_metadata, fold_commit_phase_row_targets,
        p3_challenge_from_pair, p3_challenge_to_plonky2, reconstruct_commit_phase_row_host,
        reconstruct_commit_phase_row_targets, set_extension_target, set_extension_targets,
        set_fixed_batch_opening_targets, set_merkle_cap_targets_8,
        verify_commit_phase_opening_targets, verify_fixed_batch_opening_targets,
        verify_fixed_batch_opening_targets_constant_cap, verify_fixed_fri_host,
        verify_fixed_fri_input_openings_host, verify_fri_query_targets,
    };
    use crate::{FIXED_RECURSIVE_SIGNATURE_COUNT, prepare_recursive_witness};

    fn canonical_bits_gadget_accepts(value: u64) -> bool {
        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let bits: Vec<_> = (0..64)
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        constrain_canonical_goldilocks_bits(&mut builder, &bits);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        for (bit, target) in bits.iter().enumerate() {
            pw.set_bool_target(*target, ((value >> bit) & 1) == 1)
                .unwrap();
        }
        data.prove(pw).is_ok()
    }

    #[test]
    fn canonical_goldilocks_bits_enforce_modulus_boundary() {
        const MODULUS: u64 = 0xffff_ffff_0000_0001;
        const AMBIGUOUS_CANONICAL_VALUE: u64 = 0xdead_beef;

        assert!(canonical_bits_gadget_accepts(0));
        assert!(canonical_bits_gadget_accepts(AMBIGUOUS_CANONICAL_VALUE));
        assert!(canonical_bits_gadget_accepts(MODULUS - 1));
        assert!(!canonical_bits_gadget_accepts(MODULUS));
        assert!(!canonical_bits_gadget_accepts(
            AMBIGUOUS_CANONICAL_VALUE + MODULUS
        ));
        assert!(!canonical_bits_gadget_accepts(u64::MAX));
    }

    #[test]
    fn canonical_goldilocks_low_bits_match_native_sampling() {
        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let value = builder.add_virtual_target();
        let low_bits = canonical_goldilocks_low_bits(&mut builder, value, 20);
        let expected = 0x000f_f00du64;
        for (bit, target) in low_bits.iter().enumerate() {
            let expected_bit = builder.constant_bool(((expected >> bit) & 1) == 1);
            builder.connect(target.target, expected_bit.target);
        }
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_target(value, F::from_canonical_u64(expected))
            .unwrap();
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    fn cache_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-cache/p3-schnorr-recursion")
    }

    fn cache_path(signature_count: usize) -> PathBuf {
        cache_dir().join(format!("signature-batch-{signature_count}.bin"))
    }

    fn cached_proving_output(signature_count: usize) -> SignatureBatchProvingOutput {
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

    #[test]
    fn fixed_input_mmcs_openings_match_cached_fixture() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();

        verify_fixed_fri_input_openings_host(&witness, &metadata).unwrap();
    }

    #[test]
    fn fixed_full_fri_matches_cached_fixture() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();

        verify_fixed_fri_host(&witness, &metadata).unwrap();
    }

    #[test]
    #[ignore = "focused MMCS gadget probe"]
    fn first_query_input_mmcs_gadget_proves() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];

        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let trace_cap = add_virtual_merkle_cap_8(&mut builder, metadata.trace_cap_roots);
        let trace_opening = FixedBatchOpeningTargets {
            opened_values: vec![builder.add_virtual_targets(metadata.air_width)],
            opening_proof: (0..metadata.fri_input_opening_proof_len)
                .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                .collect(),
        };
        let quotient_cap = add_virtual_merkle_cap_8(&mut builder, metadata.quotient_cap_roots);
        let quotient_opening = FixedBatchOpeningTargets {
            opened_values: (0..metadata.num_quotient_chunks)
                .map(|_| builder.add_virtual_targets(RECURSION_D))
                .collect(),
            opening_proof: (0..metadata.fri_input_opening_proof_len)
                .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                .collect(),
        };
        let preprocessed_opening = FixedBatchOpeningTargets {
            opened_values: vec![builder.add_virtual_targets(metadata.preprocessed_width)],
            opening_proof: (0..metadata.fri_input_opening_proof_len)
                .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                .collect(),
        };
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();

        verify_fixed_batch_opening_targets(&mut builder, &trace_cap, &trace_opening, &query_bits);
        verify_fixed_batch_opening_targets(
            &mut builder,
            &quotient_cap,
            &quotient_opening,
            &query_bits,
        );
        verify_fixed_batch_opening_targets_constant_cap(
            &mut builder,
            &metadata.preprocessed_commitment,
            &preprocessed_opening,
            &query_bits,
        );

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        set_merkle_cap_targets_8(&mut pw, &trace_cap, &structured.trace_commitment).unwrap();
        set_fixed_batch_opening_targets(&mut pw, &trace_opening, &query.input_proof[0]).unwrap();
        set_merkle_cap_targets_8(
            &mut pw,
            &quotient_cap,
            &structured.quotient_chunks_commitment,
        )
        .unwrap();
        set_fixed_batch_opening_targets(&mut pw, &quotient_opening, &query.input_proof[1]).unwrap();
        set_fixed_batch_opening_targets(&mut pw, &preprocessed_opening, &query.input_proof[2])
            .unwrap();
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }

        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    #[ignore = "focused MMCS gadget probe"]
    fn first_query_trace_opening_gadget_proves() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let trace_cap = add_virtual_merkle_cap_8(&mut builder, metadata.trace_cap_roots);
        let trace_opening = FixedBatchOpeningTargets {
            opened_values: vec![builder.add_virtual_targets(metadata.air_width)],
            opening_proof: (0..metadata.fri_input_opening_proof_len)
                .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                .collect(),
        };
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        verify_fixed_batch_opening_targets(&mut builder, &trace_cap, &trace_opening, &query_bits);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        set_merkle_cap_targets_8(&mut pw, &trace_cap, &structured.trace_commitment).unwrap();
        set_fixed_batch_opening_targets(&mut pw, &trace_opening, &query.input_proof[0]).unwrap();
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    #[ignore = "focused MMCS gadget probe"]
    fn first_query_quotient_opening_gadget_proves() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let quotient_cap = add_virtual_merkle_cap_8(&mut builder, metadata.quotient_cap_roots);
        let quotient_opening = FixedBatchOpeningTargets {
            opened_values: (0..metadata.num_quotient_chunks)
                .map(|_| builder.add_virtual_targets(RECURSION_D))
                .collect(),
            opening_proof: (0..metadata.fri_input_opening_proof_len)
                .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                .collect(),
        };
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        verify_fixed_batch_opening_targets(
            &mut builder,
            &quotient_cap,
            &quotient_opening,
            &query_bits,
        );
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        set_merkle_cap_targets_8(
            &mut pw,
            &quotient_cap,
            &structured.quotient_chunks_commitment,
        )
        .unwrap();
        set_fixed_batch_opening_targets(&mut pw, &quotient_opening, &query.input_proof[1]).unwrap();
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    #[ignore = "focused MMCS gadget probe"]
    fn first_query_preprocessed_opening_gadget_proves() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let query = &witness.structured_proof().opening_proof.query_proofs[0];
        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let preprocessed_opening = FixedBatchOpeningTargets {
            opened_values: vec![builder.add_virtual_targets(metadata.preprocessed_width)],
            opening_proof: (0..metadata.fri_input_opening_proof_len)
                .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
                .collect(),
        };
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        verify_fixed_batch_opening_targets_constant_cap(
            &mut builder,
            &metadata.preprocessed_commitment,
            &preprocessed_opening,
            &query_bits,
        );
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        set_fixed_batch_opening_targets(&mut pw, &preprocessed_opening, &query.input_proof[2])
            .unwrap();
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn padding_free_hash_matches_host_trace_opening() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query = &witness.structured_proof().opening_proof.query_proofs[0];
        let flat: Vec<_> = query.input_proof[0]
            .opened_values
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let expected = P3Hash::new(p3_goldilocks::default_goldilocks_poseidon2_16()).hash_iter(
            flat.iter()
                .copied()
                .map(p3_goldilocks::Goldilocks::from_u64),
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = crate::poseidon2_gate::NUM_WIRES;
        let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);
        let input = builder.add_virtual_targets(metadata.air_width);
        let digest = super::padding_free_hash_targets(&mut builder, &input);
        for (target, value) in digest.iter().zip(expected.iter()) {
            let constant = builder.constant(super::goldilocks_from_p3(*value));
            builder.connect(*target, constant);
        }
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        for (target, value) in input.iter().zip(flat.iter()) {
            pw.set_target(*target, F::from_canonical_u64(*value))
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn compress_digest_matches_host_trace_sibling() {
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query = &witness.structured_proof().opening_proof.query_proofs[0];
        let flat: Vec<_> = query.input_proof[0]
            .opened_values
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let leaf = P3Hash::new(p3_goldilocks::default_goldilocks_poseidon2_16()).hash_iter(
            flat.iter()
                .copied()
                .map(p3_goldilocks::Goldilocks::from_u64),
        );
        let sibling =
            query.input_proof[0].opening_proof[0].map(p3_goldilocks::Goldilocks::from_u64);
        let expected = P3Compress::new(p3_goldilocks::default_goldilocks_poseidon2_16())
            .compress([leaf, sibling]);

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = crate::poseidon2_gate::NUM_WIRES;
        let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);
        let left = builder.add_virtual_targets(8);
        let right = builder.add_virtual_targets(8);
        let left_array: [plonky2::iop::target::Target; 8] = left.clone().try_into().unwrap();
        let right_array: [plonky2::iop::target::Target; 8] = right.clone().try_into().unwrap();
        let digest = super::compress_digest_targets(&mut builder, left_array, right_array);
        for (target, value) in digest.iter().zip(expected.iter()) {
            let constant = builder.constant(super::goldilocks_from_p3(*value));
            builder.connect(*target, constant);
        }
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        for (target, value) in left.iter().zip(leaf.iter()) {
            pw.set_target(*target, super::goldilocks_from_p3(*value))
                .unwrap();
        }
        for (target, value) in right.iter().zip(sibling.iter()) {
            pw.set_target(*target, super::goldilocks_from_p3(*value))
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn merkle_path_matches_host_trace_opening() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let flat: Vec<_> = query.input_proof[0]
            .opened_values
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let leaf = P3Hash::new(p3_goldilocks::default_goldilocks_poseidon2_16()).hash_iter(
            flat.iter()
                .copied()
                .map(p3_goldilocks::Goldilocks::from_u64),
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = crate::poseidon2_gate::NUM_WIRES;
        let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);
        let leaf_targets = builder.add_virtual_targets(8);
        let cap = add_virtual_merkle_cap_8(&mut builder, metadata.trace_cap_roots);
        let siblings: Vec<[plonky2::iop::target::Target; 8]> = (0..metadata
            .fri_input_opening_proof_len)
            .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
            .collect();
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        super::verify_binary_merkle_path_targets(
            &mut builder,
            &cap.roots,
            leaf_targets.clone().try_into().unwrap(),
            &query_bits,
            &siblings,
        );
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        for (target, value) in leaf_targets.iter().zip(leaf.iter()) {
            pw.set_target(*target, super::goldilocks_from_p3(*value))
                .unwrap();
        }
        set_merkle_cap_targets_8(&mut pw, &cap, &structured.trace_commitment).unwrap();
        for (target_digest, digest) in siblings
            .iter()
            .zip(query.input_proof[0].opening_proof.iter())
        {
            for (target, value) in target_digest.iter().zip(digest.iter()) {
                pw.set_target(*target, F::from_canonical_u64(*value))
                    .unwrap();
            }
        }
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn first_query_commit_phase_row_reconstruction_matches_host() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let fri_alpha = derive_fri_alpha_host(&witness, &metadata).unwrap();
        let (_, zeta) = derive_transcript_challenges_host(&witness, &metadata).unwrap();
        let initial_eval = compute_initial_fri_reduced_opening_host(
            structured,
            query,
            query_index,
            fri_alpha,
            zeta,
            &metadata,
        )
        .unwrap();
        let opening = &query.commit_phase_openings[0];
        let log_arity = opening.log_arity as usize;
        let arity = 1usize << log_arity;
        let index_in_group = query_index & (arity - 1);
        let expected_row = reconstruct_commit_phase_row_host(
            index_in_group,
            &opening.sibling_values,
            initial_eval,
        )
        .unwrap();

        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let index_target = builder.add_virtual_target();
        let sibling_targets = builder.add_virtual_extension_targets(arity - 1);
        let folded_eval_target = builder.add_virtual_extension_target();
        let reconstructed = reconstruct_commit_phase_row_targets(
            &mut builder,
            index_target,
            &sibling_targets,
            folded_eval_target,
        );
        for (target, value) in reconstructed.iter().zip(expected_row.iter()) {
            let constant = builder.constant_extension(p3_challenge_to_plonky2(*value).unwrap());
            builder.connect_extension(*target, constant);
        }
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_target(index_target, F::from_canonical_usize(index_in_group))
            .unwrap();
        set_extension_targets(&mut pw, &sibling_targets, &opening.sibling_values).unwrap();
        set_extension_target(&mut pw, folded_eval_target, initial_eval).unwrap();
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn first_query_commit_phase_fold_matches_host() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let fri_alpha = derive_fri_alpha_host(&witness, &metadata).unwrap();
        let fri_betas = derive_fri_betas_host(&witness, &metadata).unwrap();
        let (_, zeta) = derive_transcript_challenges_host(&witness, &metadata).unwrap();
        let initial_eval = compute_initial_fri_reduced_opening_host(
            structured,
            query,
            query_index,
            fri_alpha,
            zeta,
            &metadata,
        )
        .unwrap();
        let opening = &query.commit_phase_openings[0];
        let log_arity = opening.log_arity as usize;
        let arity = 1usize << log_arity;
        let index_in_group = query_index & (arity - 1);
        let parent_index = query_index >> log_arity;
        let row = reconstruct_commit_phase_row_host(
            index_in_group,
            &opening.sibling_values,
            initial_eval,
        )
        .unwrap();
        let host_fold = <TwoAdicFriFolding<Vec<crate::types::P3BatchOpeningWitness>, ()> as FriFoldingStrategy<
            p3_goldilocks::Goldilocks,
            P3Challenge,
        >>::fold_row(
            &TwoAdicFriFolding(PhantomData),
            parent_index,
            metadata.degree_bits + metadata.fri_log_blowup - log_arity,
            log_arity,
            fri_betas[0],
            row.clone().into_iter(),
        );

        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let row_targets = builder.add_virtual_extension_targets(arity);
        let parent_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup - log_arity))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        let beta_target =
            builder.constant_extension(p3_challenge_to_plonky2(fri_betas[0]).unwrap());
        let folded = fold_commit_phase_row_targets(
            &mut builder,
            &parent_bits,
            metadata.degree_bits + metadata.fri_log_blowup - log_arity,
            log_arity,
            beta_target,
            &row_targets,
        );
        let expected = builder.constant_extension(p3_challenge_to_plonky2(host_fold).unwrap());
        builder.connect_extension(folded, expected);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        let row_pairs: Vec<[u64; 2]> = row
            .iter()
            .map(|value| {
                let coeffs: &[p3_goldilocks::Goldilocks] = value.as_basis_coefficients_slice();
                [coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64()]
            })
            .collect();
        set_extension_targets(&mut pw, &row_targets, &row_pairs).unwrap();
        for (i, bit) in parent_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((parent_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn first_query_full_fri_query_gadget_proves() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let fri_alpha = derive_fri_alpha_host(&witness, &metadata).unwrap();
        let fri_betas = derive_fri_betas_host(&witness, &metadata).unwrap();
        let (_, zeta) = derive_transcript_challenges_host(&witness, &metadata).unwrap();
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = crate::poseidon2_gate::NUM_WIRES;
        let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);
        let proof_targets = super::TranscriptVerifierTargets {
            degree_bits: builder.add_virtual_target(),
            trace_commitment: add_virtual_merkle_cap_8(&mut builder, metadata.trace_cap_roots),
            quotient_chunks_commitment: add_virtual_merkle_cap_8(
                &mut builder,
                metadata.quotient_cap_roots,
            ),
            trace_local: builder.add_virtual_extension_targets(metadata.air_width),
            trace_next: builder.add_virtual_extension_targets(metadata.trace_next_width),
            preprocessed_local: builder.add_virtual_extension_targets(metadata.preprocessed_width),
            preprocessed_next: builder
                .add_virtual_extension_targets(metadata.preprocessed_next_width),
            quotient_chunks: (0..metadata.num_quotient_chunks)
                .map(|_| builder.add_virtual_extension_targets(RECURSION_D))
                .collect(),
            expected_quotient: builder.add_virtual_extension_target(),
            expected_alpha: builder.add_virtual_extension_target(),
            expected_zeta: builder.add_virtual_extension_target(),
            fri: super::add_fixed_fri_targets(&mut builder, &metadata),
        };
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        let fri_alpha_target =
            builder.constant_extension(p3_challenge_to_plonky2(fri_alpha).unwrap());
        let zeta_target = builder.constant_extension(p3_challenge_to_plonky2(zeta).unwrap());
        let fri_beta_targets = fri_betas
            .iter()
            .map(|beta| builder.constant_extension(p3_challenge_to_plonky2(*beta).unwrap()))
            .collect::<Vec<_>>();
        verify_fri_query_targets(
            &mut builder,
            &proof_targets,
            &proof_targets.fri.query_proofs[0],
            &query_bits,
            fri_alpha_target,
            zeta_target,
            &fri_beta_targets,
            &metadata,
        )
        .unwrap();

        let data = builder.build::<C>();
        let quotient = super::recompose_quotient_chunks_host(structured, zeta, &metadata).unwrap();
        let mut pw = PartialWitness::new();
        super::set_transcript_verifier_targets(
            &mut pw,
            &proof_targets,
            &metadata,
            structured,
            quotient,
            fri_alpha,
            zeta,
        )
        .unwrap();
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
        let _ = query;
    }

    #[test]
    fn first_query_initial_reduced_opening_matches_host() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let fri_alpha = derive_fri_alpha_host(&witness, &metadata).unwrap();
        let (_, zeta) = derive_transcript_challenges_host(&witness, &metadata).unwrap();
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let expected = compute_initial_fri_reduced_opening_host(
            structured,
            query,
            query_index,
            fri_alpha,
            zeta,
            &metadata,
        )
        .unwrap();

        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let proof_targets = super::TranscriptVerifierTargets {
            degree_bits: builder.add_virtual_target(),
            trace_commitment: add_virtual_merkle_cap_8(&mut builder, metadata.trace_cap_roots),
            quotient_chunks_commitment: add_virtual_merkle_cap_8(
                &mut builder,
                metadata.quotient_cap_roots,
            ),
            trace_local: builder.add_virtual_extension_targets(metadata.air_width),
            trace_next: builder.add_virtual_extension_targets(metadata.trace_next_width),
            preprocessed_local: builder.add_virtual_extension_targets(metadata.preprocessed_width),
            preprocessed_next: builder
                .add_virtual_extension_targets(metadata.preprocessed_next_width),
            quotient_chunks: (0..metadata.num_quotient_chunks)
                .map(|_| builder.add_virtual_extension_targets(RECURSION_D))
                .collect(),
            expected_quotient: builder.add_virtual_extension_target(),
            expected_alpha: builder.add_virtual_extension_target(),
            expected_zeta: builder.add_virtual_extension_target(),
            fri: super::add_fixed_fri_targets(&mut builder, &metadata),
        };
        let query_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        let fri_alpha_target =
            builder.constant_extension(p3_challenge_to_plonky2(fri_alpha).unwrap());
        let zeta_target = builder.constant_extension(p3_challenge_to_plonky2(zeta).unwrap());
        let reduced = compute_initial_fri_reduced_opening_targets(
            &mut builder,
            &proof_targets,
            &proof_targets.fri.query_proofs[0],
            &query_bits,
            fri_alpha_target,
            zeta_target,
            &metadata,
        )
        .unwrap();
        let expected_target =
            builder.constant_extension(p3_challenge_to_plonky2(expected).unwrap());
        builder.connect_extension(reduced, expected_target);
        let data = builder.build::<C>();
        let quotient = super::recompose_quotient_chunks_host(structured, zeta, &metadata).unwrap();
        let mut pw = PartialWitness::new();
        super::set_transcript_verifier_targets(
            &mut pw,
            &proof_targets,
            &metadata,
            structured,
            quotient,
            fri_alpha,
            zeta,
        )
        .unwrap();
        for (i, bit) in query_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((query_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn first_query_round0_commit_phase_opening_gadget_proves() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let fri_alpha = derive_fri_alpha_host(&witness, &metadata).unwrap();
        let (_, zeta) = derive_transcript_challenges_host(&witness, &metadata).unwrap();
        let structured = witness.structured_proof();
        let query = &structured.opening_proof.query_proofs[0];
        let initial_eval = compute_initial_fri_reduced_opening_host(
            structured,
            query,
            query_index,
            fri_alpha,
            zeta,
            &metadata,
        )
        .unwrap();
        let opening = &query.commit_phase_openings[0];
        let log_arity = opening.log_arity as usize;
        let arity = 1usize << log_arity;
        let parent_index = query_index >> log_arity;
        let row = reconstruct_commit_phase_row_host(
            query_index & (arity - 1),
            &opening.sibling_values,
            initial_eval,
        )
        .unwrap();

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = crate::poseidon2_gate::NUM_WIRES;
        let mut builder = CircuitBuilder::<F, RECURSION_D>::new(config);
        let cap = add_virtual_merkle_cap_8(&mut builder, metadata.fri_cap_roots);
        let row_targets = builder.add_virtual_extension_targets(arity);
        let parent_bits: Vec<_> = (0..(metadata.degree_bits + metadata.fri_log_blowup - log_arity))
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        let opening_proof: Vec<_> = (0..metadata.fri_commit_phase_opening_proof_lens[0])
            .map(|_| builder.add_virtual_targets(8).try_into().unwrap())
            .collect();
        verify_commit_phase_opening_targets(
            &mut builder,
            &cap,
            &parent_bits,
            &row_targets,
            &opening_proof,
        );
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        set_merkle_cap_targets_8(
            &mut pw,
            &cap,
            &structured.opening_proof.commit_phase_commits[0],
        )
        .unwrap();
        let row_pairs: Vec<[u64; 2]> = row
            .iter()
            .map(|value| {
                let coeffs: &[p3_goldilocks::Goldilocks] = value.as_basis_coefficients_slice();
                [coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64()]
            })
            .collect();
        set_extension_targets(&mut pw, &row_targets, &row_pairs).unwrap();
        for (target_digest, digest) in opening_proof.iter().zip(opening.opening_proof.iter()) {
            for (target, value) in target_digest.iter().zip(digest.iter()) {
                pw.set_target(*target, F::from_canonical_u64(*value))
                    .unwrap();
            }
        }
        for (i, bit) in parent_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((parent_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn first_query_final_poly_eval_matches_host() {
        let metadata = fixed_verifier_metadata(FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let proving_output = cached_proving_output(FIXED_RECURSIVE_SIGNATURE_COUNT);
        let witness =
            prepare_recursive_witness(&proving_output, FIXED_RECURSIVE_SIGNATURE_COUNT).unwrap();
        let structured = witness.structured_proof();
        let query_index = derive_fri_query_indices_host(&witness, &metadata).unwrap()[0];
        let final_index =
            query_index >> metadata.fri_commit_phase_log_arities.iter().sum::<usize>();
        let final_x = p3_goldilocks::Goldilocks::two_adic_generator(
            metadata.degree_bits + metadata.fri_log_blowup,
        )
        .exp_u64(super::reverse_bits_len_usize(
            final_index,
            metadata.degree_bits + metadata.fri_log_blowup,
        ) as u64);
        let mut expected = P3Challenge::ZERO;
        for &coeff in structured.opening_proof.final_poly.iter().rev() {
            expected = expected * final_x + p3_challenge_from_pair(coeff).unwrap();
        }

        let mut builder =
            CircuitBuilder::<F, RECURSION_D>::new(CircuitConfig::standard_recursion_config());
        let coeff_targets = builder.add_virtual_extension_targets(metadata.fri_final_poly_len);
        let final_bits: Vec<_> = (0..metadata.fri_final_log_height)
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        let final_x_target = super::subgroup_point_targets(
            &mut builder,
            &final_bits,
            metadata.degree_bits + metadata.fri_log_blowup,
        );
        let eval = evaluate_final_poly_targets(&mut builder, &coeff_targets, final_x_target);
        let expected_target =
            builder.constant_extension(p3_challenge_to_plonky2(expected).unwrap());
        builder.connect_extension(eval, expected_target);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        set_extension_targets(
            &mut pw,
            &coeff_targets,
            &structured.opening_proof.final_poly,
        )
        .unwrap();
        for (i, bit) in final_bits.iter().enumerate() {
            pw.set_bool_target(*bit, ((final_index >> i) & 1) == 1)
                .unwrap();
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }
}
