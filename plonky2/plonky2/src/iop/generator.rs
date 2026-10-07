#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Debug;
use core::marker::PhantomData;

use anyhow::{anyhow, Result};
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, GeneratorWatchIndex, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Given a `PartitionWitness` that has only inputs set, populates the rest of the witness using the
/// given set of generators.
pub fn generate_partial_witness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    PendingPartitionWitness::start(inputs, prover_data, common_data)?.finish()
}

/// Ready sets at least this large are executed as one data-parallel round; smaller ones run on
/// the sequential loop, so thin dependency chains keep their single-threaded latency.
const PARALLEL_WORKLIST_THRESHOLD: usize = 64;

/// Generators per parallel-round task. Chunking amortizes per-task scheduling and buffer
/// overhead across cheap generators while leaving enough tasks for load balancing.
const PARALLEL_WORKLIST_CHUNK: usize = 64;

#[cfg(all(feature = "parallel", feature = "std"))]
mod parallel_witness_context {
    use core::cell::Cell;

    std::thread_local! {
        static PARALLEL_WITNESS_ROUNDS: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn enabled() -> bool {
        PARALLEL_WITNESS_ROUNDS.with(Cell::get)
    }

    pub(super) fn replace(enabled: bool) -> bool {
        PARALLEL_WITNESS_ROUNDS.with(|flag| flag.replace(enabled))
    }
}

/// RAII guard opting the current thread's witness generation into data-parallel worklist rounds.
///
/// Parallel rounds run on the rayon pool, so they are opt-in per call site: witness generation
/// that runs concurrently with proving must stay sequential rather than contend with the prover
/// for the pool, while witness generation on an otherwise idle serial section (e.g. the final
/// block proof) can claim it. Dropping the guard restores the previous state, so guards nest.
#[derive(Debug)]
#[must_use = "parallel witness rounds stay enabled only while the guard is alive"]
pub struct ParallelWitnessGuard {
    #[cfg(all(feature = "parallel", feature = "std"))]
    previous: bool,
}

impl ParallelWitnessGuard {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            #[cfg(all(feature = "parallel", feature = "std"))]
            previous: parallel_witness_context::replace(true),
        }
    }
}

#[cfg(all(feature = "parallel", feature = "std"))]
impl Drop for ParallelWitnessGuard {
    fn drop(&mut self) {
        parallel_witness_context::replace(self.previous);
    }
}

/// Parallel rounds require an explicit opt-in on the current thread ([`ParallelWitnessGuard`])
/// and a pool with more than one thread; otherwise the per-round buffer collection is pure
/// overhead over the sequential loop, or worse, contends with concurrent proving.
#[cfg(all(feature = "parallel", feature = "std"))]
fn parallel_rounds_enabled() -> bool {
    parallel_witness_context::enabled() && rayon::current_num_threads() > 1
}

#[cfg(not(all(feature = "parallel", feature = "std")))]
fn parallel_rounds_enabled() -> bool {
    false
}

/// Dense per-generator readiness state driven by first-population events of representative
/// targets.
///
/// The immutable reverse index is already representative-keyed CSR. Keeping its mutable
/// complement here makes the whole scheduling path index-addressable: a `Target` is reduced to
/// its representative by `PartitionWitness`, the CSR yields a contiguous generator slice, and
/// those generator indices update this dense counter vector. `u32` is sufficient by construction
/// because every watch edge is stored in the CSR's `u32` offset space; using the same width here
/// halves the hot readiness working set compared with one `usize` per generator.
struct TargetReadiness<'a> {
    unresolved: Vec<u32>,
    watchers: &'a GeneratorWatchIndex,
}

impl<'a> TargetReadiness<'a> {
    fn new(generator_watch_counts: &[usize], watchers: &'a GeneratorWatchIndex) -> Self {
        let unresolved = generator_watch_counts
            .iter()
            .map(|&count| {
                u32::try_from(count).expect("generator watch count exceeds the CSR u32 edge index")
            })
            .collect();
        Self {
            unresolved,
            watchers,
        }
    }

    #[inline]
    fn is_ready(&self, generator: usize) -> bool {
        self.unresolved[generator] == 0
    }

    /// Builds the first worklist after the input seed has populated its representatives.
    ///
    /// If every generator defers until all of its watches are present, only generators whose
    /// unresolved count is already zero can do anything in the first round. The others are queued
    /// exactly once when their final missing representative is populated. General incremental
    /// generators retain the legacy all-generators first round.
    fn initial_worklist(&self, generators_defer_until_ready: bool) -> Vec<usize> {
        let mut pending = Vec::with_capacity(self.unresolved.len());
        if generators_defer_until_ready {
            pending.extend(
                self.unresolved
                    .iter()
                    .enumerate()
                    .filter_map(|(generator, &unresolved)| (unresolved == 0).then_some(generator)),
            );
        } else {
            pending.extend(0..self.unresolved.len());
        }
        pending
    }

    /// Applies a seed's first-population event. Aliased or duplicated target writes never reach
    /// this method twice because `PartitionWitness` returns a representative only on its first
    /// population.
    #[inline]
    fn seed_representative(&mut self, representative: usize) {
        if let Some(watchers) = self.watchers.get(&representative) {
            for &generator in watchers {
                let unresolved = &mut self.unresolved[generator as usize];
                debug_assert_ne!(*unresolved, 0);
                *unresolved -= 1;
            }
        }
    }

    /// Applies a live first-population event and queues exactly the generators whose scheduling
    /// policy says they may now make progress.
    #[inline]
    fn populate_representative(
        &mut self,
        representative: usize,
        generator_is_expired: &[bool],
        pending: &mut Vec<usize>,
        generators_defer_until_ready: bool,
    ) {
        if let Some(watchers) = self.watchers.get(&representative) {
            self.populate_watchers(
                watchers,
                generator_is_expired,
                pending,
                generators_defer_until_ready,
            );
        }
    }

    #[inline]
    fn populate_watchers(
        &mut self,
        watchers: &[u32],
        generator_is_expired: &[bool],
        pending: &mut Vec<usize>,
        generators_defer_until_ready: bool,
    ) {
        for &generator in watchers {
            let generator = generator as usize;
            if !generator_is_expired[generator]
                && watch_populated(
                    &mut self.unresolved[generator],
                    generators_defer_until_ready,
                )
            {
                pending.push(generator);
            }
        }
    }
}

/// Applies one first-population event to a generator's unresolved-watch count and reports whether
/// it should enter the next worklist.
///
/// Deferred generators are inert before the transition to zero, so suppressing their earlier
/// queue entries removes only proven no-ops. A circuit containing any incremental generator uses
/// the legacy behavior and queues on every transition.
#[inline]
fn watch_populated(unresolved: &mut u32, generators_defer_until_ready: bool) -> bool {
    debug_assert_ne!(*unresolved, 0);
    *unresolved -= 1;
    !generators_defer_until_ready || *unresolved == 0
}

/// Runs the given pending generators, and transitively any generator watching a newly populated
/// representative, until no further progress can be made.
///
/// Rounds whose ready set reaches `parallel_threshold` run all their generators in parallel
/// against the current witness snapshot; the generated values are then merged sequentially in
/// ascending generator-index order. Generators are deterministic functions of their watched
/// values and only merged values mutate the witness, so every schedule (sequential, parallel at
/// any thread count) reaches the same fixpoint, and the deterministic merge order keeps
/// contradiction detection (`set_target_returning_rep`) behavior identical across runs.
fn run_generator_worklist<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &mut PartitionWitness<F>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    readiness: &mut TargetReadiness<'_>,
    generator_is_expired: &mut [bool],
    remaining_generators: &mut usize,
    mut pending_generator_indices: Vec<usize>,
    parallel_threshold: usize,
) -> Result<()> {
    let generators = &prover_data.generators;
    let generator_indices_by_watches = readiness.watchers;
    // When every generator defers until ready, it enters a worklist only after its unresolved
    // count reaches zero. Keep the readiness check as a defensive backstop for duplicate entries
    // and for the general-generator fallback; an unready dispatch is still a proven no-op.
    let skip_unready = prover_data.generators_defer_until_ready;

    let parallel_rounds = parallel_rounds_enabled();
    let mut buffer = GeneratedValues::empty();

    // The two round queues are swapped rather than reallocated. Every round used
    // to start from a fresh `Vec::new()` and end by *moving* it over the old
    // queue, which freed the old buffer and forced the new one to grow from zero
    // capacity again — a fresh allocation plus a geometric doubling chain (each
    // step memcpying everything pushed so far) on every round of every witness
    // generation, and there is one witness generation per transaction chunk and
    // per chain step. Swapping keeps both buffers at their high-water capacity,
    // so after the first few rounds a round costs no allocation and no growth
    // copies at all. `clear()` only resets the length (`usize` has no `Drop`),
    // so each round still observes an empty queue and pushes exactly the same
    // indices in exactly the same order.
    let mut next_pending_generator_indices = Vec::new();

    // Keep running generators until we fail to make progress.
    while !pending_generator_indices.is_empty() {
        next_pending_generator_indices.clear();

        if parallel_rounds && pending_generator_indices.len() >= parallel_threshold {
            // A generator can be enqueued once per newly populated watch, and may have expired
            // in the round that enqueued it; run each remaining generator exactly once.
            pending_generator_indices.sort_unstable();
            pending_generator_indices.dedup();

            // Run phase: every generator reads the same witness snapshot and each chunk writes
            // into its own buffers, so the round is data-parallel while chunking amortizes
            // per-task scheduling and allocation across cheap generators. Each produced value is
            // annotated with the watchers of its (snapshot-unpopulated) representative here,
            // moving those read-only watcher-index lookups off the sequential merge. Chunk
            // boundaries only group the outputs: `collect` preserves chunk order and each chunk
            // records its generators in ready-set order, so the merge below observes ascending
            // generator-index order regardless of thread count.
            let round_witness: &PartitionWitness<F> = witness;
            let round_unresolved_watches: &[u32] = &readiness.unresolved;
            let round_generator_is_expired: &[bool] = generator_is_expired;
            #[allow(clippy::type_complexity)]
            let round_outputs: Vec<(
                Vec<(usize, bool, usize)>,
                Vec<(Target, F, usize, Option<&[u32]>)>,
            )> = pending_generator_indices
                .par_chunks(PARALLEL_WORKLIST_CHUNK)
                .map(|chunk| {
                    let mut entries = Vec::with_capacity(chunk.len());
                    // Same capacity discipline as `entries`: at least one value per
                    // generator, so the geometric doubling chain starts past its
                    // first few reallocations instead of from zero.
                    let mut annotated_values = Vec::with_capacity(chunk.len());
                    let mut round_buffer = GeneratedValues::empty();
                    for &generator_idx in chunk {
                        if round_generator_is_expired[generator_idx] {
                            continue;
                        }
                        let ready = round_unresolved_watches[generator_idx] == 0;
                        if skip_unready && !ready {
                            continue;
                        }
                        let finished = generators[generator_idx].0.run_with_ready_hint(
                            round_witness,
                            &mut round_buffer,
                            ready,
                        );
                        entries.push((generator_idx, finished, round_buffer.target_values.len()));
                        for (t, v) in round_buffer.target_values.drain(..) {
                            let rep_index = round_witness.representative_map
                                [round_witness.target_index(t)]
                                as usize;
                            let watchers = if !round_witness.is_set_by_rep_index(rep_index) {
                                generator_indices_by_watches.get(&rep_index)
                            } else {
                                // The representative is populated in the snapshot, so the merge
                                // cannot newly populate it and never needs watchers.
                                None
                            };
                            annotated_values.push((t, v, rep_index, watchers));
                        }
                    }
                    (entries, annotated_values)
                })
                .collect();

            // Merge phase: sequential and in ascending generator-index order, exactly like the
            // sequential loop's per-generator merge.
            for (entries, annotated_values) in round_outputs {
                let mut annotated_values = annotated_values.into_iter();
                for (generator_idx, finished, value_count) in entries {
                    if finished {
                        generator_is_expired[generator_idx] = true;
                        *remaining_generators -= 1;
                    }

                    for (t, v, rep_index, watchers) in annotated_values.by_ref().take(value_count) {
                        // Reuse the representative the run phase already gathered
                        // instead of gathering it again. `round_witness` above is an
                        // immutable reborrow of this very `witness`, and
                        // `representative_map` is an immutable `&[u32]` for the
                        // witness's whole lifetime, so the index is by construction
                        // the one `set_target_returning_rep` would have looked up.
                        // `set_rep_index_returning_new` is documented as running the
                        // identical sequence on the identical slot from that point on,
                        // so the resulting witness is bit-for-bit unchanged. What goes
                        // away is a second scattered 4-byte read out of a table of
                        // `num_wires * degree` entries — one per generated value, on
                        // the order of 10^6 per proof — and it is a read that misses
                        // twice, because the whole round's outputs are collected
                        // between the two lookups.
                        if witness
                            .set_rep_index_returning_new(rep_index, t, v)?
                            .is_none()
                        {
                            continue;
                        }
                        if let Some(watchers) = watchers {
                            readiness.populate_watchers(
                                watchers,
                                generator_is_expired,
                                &mut next_pending_generator_indices,
                                skip_unready,
                            );
                        }
                    }
                }
            }

            core::mem::swap(
                &mut pending_generator_indices,
                &mut next_pending_generator_indices,
            );
            continue;
        }

        for &generator_idx in &pending_generator_indices {
            if generator_is_expired[generator_idx] {
                continue;
            }
            let ready = readiness.is_ready(generator_idx);
            if skip_unready && !ready {
                continue;
            }

            let finished =
                generators[generator_idx]
                    .0
                    .run_with_ready_hint(witness, &mut buffer, ready);
            if finished {
                generator_is_expired[generator_idx] = true;
                *remaining_generators -= 1;
            }

            // Merge any generated values into our witness and, for each newly populated
            // target's representative, immediately enqueue the unfinished generators watching
            // it. The witness merge (`witness`) and the watcher bookkeeping
            // (`generator_indices_by_watches`, `generator_is_expired`, `unresolved_watches`)
            // touch disjoint state, so fusing the two passes deletes the per-run intermediate
            // rep Vec while preserving both the `set_target_returning_rep` call order and the
            // pending-queue push order exactly.
            for (t, v) in buffer.target_values.drain(..) {
                if let Some(representative) = witness.set_target_returning_rep(t, v)? {
                    readiness.populate_representative(
                        representative,
                        generator_is_expired,
                        &mut next_pending_generator_indices,
                        skip_unready,
                    );
                }
            }
        }

        core::mem::swap(
            &mut pending_generator_indices,
            &mut next_pending_generator_indices,
        );
    }

    Ok(())
}

/// Seeds `inputs` into `witness` and returns, per generator, the number of distinct
/// representatives it watches that are still unpopulated.
///
/// A generator can run once every distinct representative it watches has a value. The count is
/// therefore `(total distinct representatives watched) - (those already populated)`. The first
/// term is `generator_watch_counts`, derived once at circuit-build time; the second is accumulated
/// here by decrementing each watcher of a representative at the moment that representative is
/// *first* populated. `set_target_returning_rep` returns the representative only on first
/// population, so aliased or duplicated inputs decrement at most once and no counter can
/// underflow. This is the exact complement of the previous initialization, which instead walked
/// the entire representative-keyed watcher map on every proof and counted the unpopulated
/// entries — an O(total watch edges) traversal of prover data that is identical for every proof of
/// a given circuit, repeated once per proof.
fn seed_inputs_and_readiness<'a, F: Field>(
    witness: &mut PartitionWitness<F>,
    inputs: PartialWitness<F>,
    generator_watch_counts: &[usize],
    generator_indices_by_watches: &'a GeneratorWatchIndex,
) -> Result<TargetReadiness<'a>> {
    let mut readiness = TargetReadiness::new(generator_watch_counts, generator_indices_by_watches);

    for (t, v) in inputs.target_values.into_iter() {
        if let Some(watch) = witness.set_target_returning_rep(t, v)? {
            readiness.seed_representative(watch);
        }
    }

    Ok(readiness)
}

/// Direct-seeding adapter: writes values straight into the partition's
/// representative slots while maintaining the same per-generator
/// unresolved-watch counters as [`seed_inputs_and_readiness`],
/// without routing the values through a `PartialWitness` map first. The
/// decrement rule is identical: `set_target_returning_rep` returns the
/// representative only on first population, so aliased or duplicated
/// inputs decrement at most once and no counter can underflow.
pub struct PartitionSeeder<'a, 'b, F: Field> {
    witness: &'b mut PartitionWitness<'a, F>,
    readiness: &'b mut TargetReadiness<'a>,
    layout: SeedLayoutMode<'b>,
}

/// A recorded seed layout cannot be applied to this circuit/input shape.
///
/// Only this error may be retried on the generic target-lookup path. Errors
/// raised by the input writer itself keep their original type and propagate,
/// so a stateful writer is never rerun for a reason that would recur.
#[derive(Debug)]
pub struct SeedLayoutMismatch(String);

impl core::fmt::Display for SeedLayoutMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for SeedLayoutMismatch {}

fn seed_layout_mismatch(message: impl Into<String>) -> anyhow::Error {
    SeedLayoutMismatch(message.into()).into()
}

/// Whether an error denotes only seed-layout applicability, for a guarded
/// generic fallback.
pub fn is_seed_layout_mismatch(error: &anyhow::Error) -> bool {
    error.downcast_ref::<SeedLayoutMismatch>().is_some()
}

/// Order-sensitive 64-bit fold of the target sequence a writer produces.
///
/// The recorded layout stores one bare `u32` representative per write rather
/// than a `{Target, u32}` pair, so per-entry target equality is replaced by
/// this single accumulated checksum, compared once at the end of a replay
/// before any watcher decrement is applied or any generator is run. Feeding it
/// `target_index` (pure arithmetic on the `Target` enum, no memory traffic)
/// keeps the scattered `representative_map` gather deleted, which is the whole
/// point of the layout.
#[inline]
fn fold_target_checksum(checksum: u64, target_index: usize) -> u64 {
    checksum.rotate_left(11).wrapping_add(0x517c_c1b7_2722_0a95)
        ^ (target_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Exact target order and aggregate watcher effects of a fixed witness-input
/// writer, replayable across later inputs of the same shape.
///
/// Values are deliberately absent: replay still receives and writes every value
/// the caller supplies. The layout removes only the repeated target-to-
/// representative lookup and the repeated traversal of the immutable watcher
/// CSR for the same circuit/input shape.
pub struct PartitionSeedLayout<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    /// One representative per `set_target` call, in call order.
    representatives: Vec<u32>,
    /// Fold of the recorded target sequence; a replay whose targets differ in
    /// value or order cannot reproduce it.
    target_checksum: u64,
    seeded_watch_decrements: Vec<(u32, u32)>,
    /// Lifetime brand for the exact immutable prover topology used while
    /// recording. Holding the whole owner borrowed prevents drop, mutation and
    /// allocator-address reuse (ABA), while pointer equality binds the
    /// representative map, generator list, watch counts and watcher CSR in O(1)
    /// at replay.
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Debug
    for PartitionSeedLayout<'_, F, C, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PartitionSeedLayout")
            .field("target_write_count", &self.representatives.len())
            .field(
                "changed_generator_count",
                &self.seeded_watch_decrements.len(),
            )
            .finish_non_exhaustive()
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PartitionSeedLayout<'_, F, C, D>
{
    /// Number of `set_target` calls captured by the fixed writer. Exposed for
    /// census output only; replay never trusts a caller-supplied count.
    pub fn target_write_count(&self) -> usize {
        self.representatives.len()
    }

    /// Number of generators whose seeded watch count changes. Replay traverses
    /// exactly this sparse list instead of every watcher list reached by every
    /// input target.
    pub fn changed_generator_count(&self) -> usize {
        self.seeded_watch_decrements.len()
    }
}

enum SeedLayoutMode<'a> {
    Plain,
    Record {
        representatives: &'a mut Vec<u32>,
        checksum: &'a mut u64,
    },
    Replay {
        representatives: &'a [u32],
        cursor: usize,
        checksum: u64,
    },
}

impl<F: Field> Debug for PartitionSeeder<'_, '_, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PartitionSeeder").finish_non_exhaustive()
    }
}

impl<F: Field> WitnessWrite<F> for PartitionSeeder<'_, '_, F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        let newly_set = match &mut self.layout {
            SeedLayoutMode::Plain => self.witness.set_target_returning_rep(target, value)?,
            SeedLayoutMode::Record {
                representatives,
                checksum,
            } => {
                let target_index = self.witness.target_index(target);
                let representative = self.witness.representative_map[target_index];
                representatives.push(representative);
                **checksum = fold_target_checksum(**checksum, target_index);
                self.witness
                    .set_rep_index_returning_new(representative as usize, target, value)?
            }
            SeedLayoutMode::Replay {
                representatives,
                cursor,
                checksum,
            } => {
                let representative = *representatives.get(*cursor).ok_or_else(|| {
                    seed_layout_mismatch(format!(
                        "seed layout ended before target {target:?} at position {cursor}"
                    ))
                })?;
                *checksum = fold_target_checksum(*checksum, self.witness.target_index(target));
                *cursor += 1;
                // A drifted layout can only put the value in the wrong (but
                // in-bounds) slot, which surfaces here as an ordinary
                // contradiction. Report it as a layout mismatch so the caller
                // discards this witness and re-seeds through the generic path,
                // where a genuine contradiction recurs with its own type.
                self.witness
                    .set_rep_index_returning_new(representative as usize, target, value)
                    .map_err(|error| {
                        seed_layout_mismatch(format!("seed layout replay write failed: {error}"))
                    })?
            }
        };

        if matches!(&self.layout, SeedLayoutMode::Replay { .. }) {
            // Watcher decrements are applied once, in aggregate, after the
            // checked replay.
            return Ok(());
        }
        if let Some(representative) = newly_set {
            self.readiness.seed_representative(representative);
        }
        Ok(())
    }
}

impl<F: Field> Witness<F> for PartitionSeeder<'_, '_, F> {
    fn try_get_target(&self, target: Target) -> Option<F> {
        self.witness.try_get_target(target)
    }
}

/// Direct-feeding adapter: the [`PendingPartitionWitness::feed`] analog of
/// [`PartitionSeeder`]. Writes values straight into the partition while
/// applying `feed`'s exact bookkeeping: each first-populated representative
/// decrements its unfinished watchers and queues them for the resume
/// worklist; expired generators are skipped.
pub struct PartitionFeeder<'a, 'b, F: Field> {
    witness: &'b mut PartitionWitness<'a, F>,
    readiness: &'b mut TargetReadiness<'a>,
    generator_is_expired: &'b [bool],
    pending_generator_indices: &'b mut Vec<usize>,
    generators_defer_until_ready: bool,
}

impl<F: Field> Debug for PartitionFeeder<'_, '_, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PartitionFeeder").finish_non_exhaustive()
    }
}

impl<F: Field> WitnessWrite<F> for PartitionFeeder<'_, '_, F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        if let Some(representative) = self.witness.set_target_returning_rep(target, value)? {
            self.readiness.populate_representative(
                representative,
                self.generator_is_expired,
                self.pending_generator_indices,
                self.generators_defer_until_ready,
            );
        }
        Ok(())
    }
}

impl<F: Field> Witness<F> for PartitionFeeder<'_, '_, F> {
    fn try_get_target(&self, target: Target) -> Option<F> {
        self.witness.try_get_target(target)
    }
}

/// Resumable witness generation: [`Self::start`] seeds an initial set of inputs and runs every
/// generator that can already make progress, each [`Self::feed`] sets newly available inputs and
/// resumes only the generators watching them, and [`Self::finish`] performs the same completeness
/// check as [`generate_partial_witness`].
///
/// Generators are deterministic functions of their watched values, so splitting the same inputs
/// across `start`/`feed` calls in any order yields a witness identical to the single-shot path.
pub struct PendingPartitionWitness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    witness: PartitionWitness<'a, F>,
    readiness: TargetReadiness<'a>,
    generator_is_expired: Vec<bool>,
    remaining_generators: usize,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    parallel_threshold: usize,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Debug
    for PendingPartitionWitness<'_, F, C, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PendingPartitionWitness")
            .field("remaining_generators", &self.remaining_generators)
            .finish_non_exhaustive()
    }
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PendingPartitionWitness<'a, F, C, D>
{
    /// Seeds `inputs` and runs generators to quiescence. Unlike [`generate_partial_witness`],
    /// generators whose watched values are still missing are left pending rather than being an
    /// error.
    pub fn start(
        inputs: PartialWitness<F>,
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Result<Self> {
        Self::start_with_threshold(
            inputs,
            prover_data,
            common_data,
            PARALLEL_WORKLIST_THRESHOLD,
        )
    }

    fn start_with_threshold(
        inputs: PartialWitness<F>,
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
        parallel_threshold: usize,
    ) -> Result<Self> {
        let generators = &prover_data.generators;

        let mut witness = PartitionWitness::new(
            common_data.config.num_wires,
            common_data.degree(),
            &prover_data.representative_map,
        );

        let mut readiness = seed_inputs_and_readiness(
            &mut witness,
            inputs,
            &prover_data.generator_watch_counts,
            &prover_data.generator_indices_by_watches,
        )?;

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();

        // Deferred generators enter only when ready; incremental generators keep the legacy
        // all-generators first round.
        let initial_pending = readiness.initial_worklist(prover_data.generators_defer_until_ready);
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut readiness,
            &mut generator_is_expired,
            &mut remaining_generators,
            initial_pending,
            parallel_threshold,
        )?;

        Ok(Self {
            witness,
            readiness,
            generator_is_expired,
            remaining_generators,
            prover_data,
            parallel_threshold,
        })
    }

    /// Like [`Self::start`], but the initial inputs are written by `seed`
    /// directly into the partition through a [`PartitionSeeder`] — no
    /// intermediate `PartialWitness` map is built or replayed. Worklist
    /// initialization uses the same unresolved-watch counters as the map-seeded path.
    pub fn start_seeded(
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
        seed: impl FnOnce(&mut PartitionSeeder<'a, '_, F>) -> Result<()>,
    ) -> Result<Self> {
        let generators = &prover_data.generators;

        let mut witness = PartitionWitness::new(
            common_data.config.num_wires,
            common_data.degree(),
            &prover_data.representative_map,
        );

        let mut readiness = TargetReadiness::new(
            &prover_data.generator_watch_counts,
            &prover_data.generator_indices_by_watches,
        );
        seed(&mut PartitionSeeder {
            witness: &mut witness,
            readiness: &mut readiness,
            layout: SeedLayoutMode::Plain,
        })?;

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();

        let initial_pending = readiness.initial_worklist(prover_data.generators_defer_until_ready);
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut readiness,
            &mut generator_is_expired,
            &mut remaining_generators,
            initial_pending,
            PARALLEL_WORKLIST_THRESHOLD,
        )?;

        Ok(Self {
            witness,
            readiness,
            generator_is_expired,
            remaining_generators,
            prover_data,
            parallel_threshold: PARALLEL_WORKLIST_THRESHOLD,
        })
    }

    /// Records the checked representative layout while performing an ordinary
    /// direct seed. The returned layout is specific to this circuit *instance*
    /// and may be passed to [`Self::start_seeded_with_layout`] for later inputs
    /// written in the same target order.
    pub fn start_seeded_recording(
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &'a CommonCircuitData<F, D>,
        seed: impl FnOnce(&mut PartitionSeeder<'a, '_, F>) -> Result<()>,
    ) -> Result<(Self, PartitionSeedLayout<'a, F, C, D>)> {
        let generators = &prover_data.generators;
        let mut witness = PartitionWitness::new(
            common_data.config.num_wires,
            common_data.degree(),
            &prover_data.representative_map,
        );
        let mut readiness = TargetReadiness::new(
            &prover_data.generator_watch_counts,
            &prover_data.generator_indices_by_watches,
        );
        let mut representatives = Vec::new();
        let mut target_checksum = 0u64;
        seed(&mut PartitionSeeder {
            witness: &mut witness,
            readiness: &mut readiness,
            layout: SeedLayoutMode::Record {
                representatives: &mut representatives,
                checksum: &mut target_checksum,
            },
        })?;
        // The aggregate effect of the seed on every watch counter. A replay
        // applies exactly this sparse list instead of walking a watcher slice
        // per newly populated representative.
        let seeded_watch_decrements = prover_data
            .generator_watch_counts
            .iter()
            .zip(&readiness.unresolved)
            .enumerate()
            .filter_map(|(generator, (&total, &unresolved))| {
                let total = u32::try_from(total)
                    .expect("generator watch count exceeds the CSR u32 edge index");
                let decrement = total - unresolved;
                (decrement != 0).then(|| {
                    Ok((
                        u32::try_from(generator)
                            .map_err(|_| anyhow!("generator index exceeds u32"))?,
                        decrement,
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();
        let initial_pending = readiness.initial_worklist(prover_data.generators_defer_until_ready);
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut readiness,
            &mut generator_is_expired,
            &mut remaining_generators,
            initial_pending,
            PARALLEL_WORKLIST_THRESHOLD,
        )?;
        Ok((
            Self {
                witness,
                readiness,
                generator_is_expired,
                remaining_generators,
                prover_data,
                parallel_threshold: PARALLEL_WORKLIST_THRESHOLD,
            },
            PartitionSeedLayout {
                representatives,
                target_checksum,
                seeded_watch_decrements,
                prover_data,
                common_data,
            },
        ))
    }

    /// Replays a previously recorded layout while preserving all runtime
    /// values. Any target-order, length or circuit-instance mismatch fails
    /// closed with a [`SeedLayoutMismatch`], leaving the partially written
    /// witness to be dropped; [`Self::start_seeded`] remains the fallback.
    pub fn start_seeded_with_layout(
        prover_data: &'a ProverOnlyCircuitData<F, C, D>,
        common_data: &'a CommonCircuitData<F, D>,
        layout: &PartitionSeedLayout<'a, F, C, D>,
        seed: impl FnOnce(&mut PartitionSeeder<'a, '_, F>) -> Result<()>,
    ) -> Result<Self> {
        let generators = &prover_data.generators;
        if !core::ptr::eq(layout.prover_data, prover_data)
            || !core::ptr::eq(layout.common_data, common_data)
        {
            return Err(seed_layout_mismatch(
                "seed layout belongs to a different immutable circuit topology instance",
            ));
        }
        let mut witness = PartitionWitness::new(
            common_data.config.num_wires,
            common_data.degree(),
            &prover_data.representative_map,
        );
        let mut readiness = TargetReadiness::new(
            &prover_data.generator_watch_counts,
            &prover_data.generator_indices_by_watches,
        );
        let (cursor, checksum) = {
            let mut seeder = PartitionSeeder {
                witness: &mut witness,
                readiness: &mut readiness,
                layout: SeedLayoutMode::Replay {
                    representatives: &layout.representatives,
                    cursor: 0,
                    checksum: 0,
                },
            };
            seed(&mut seeder)?;
            match seeder.layout {
                SeedLayoutMode::Replay {
                    cursor, checksum, ..
                } => (cursor, checksum),
                _ => unreachable!(),
            }
        };
        if cursor != layout.representatives.len() {
            return Err(seed_layout_mismatch(format!(
                "seed layout has {} trailing targets after replayed {}",
                layout.representatives.len() - cursor,
                cursor,
            )));
        }
        // Checked before any watcher decrement is applied and before any
        // generator runs, so a drifted target sequence can never reach the
        // worklist.
        if checksum != layout.target_checksum {
            return Err(seed_layout_mismatch(
                "seed layout target sequence checksum mismatch",
            ));
        }
        for &(generator, decrement) in &layout.seeded_watch_decrements {
            let count = &mut readiness.unresolved[generator as usize];
            if *count < decrement {
                return Err(seed_layout_mismatch(
                    "seed layout watcher decrement underflow",
                ));
            }
            *count -= decrement;
        }

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();
        let initial_pending = readiness.initial_worklist(prover_data.generators_defer_until_ready);
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut readiness,
            &mut generator_is_expired,
            &mut remaining_generators,
            initial_pending,
            PARALLEL_WORKLIST_THRESHOLD,
        )?;
        Ok(Self {
            witness,
            readiness,
            generator_is_expired,
            remaining_generators,
            prover_data,
            parallel_threshold: PARALLEL_WORKLIST_THRESHOLD,
        })
    }

    /// Sets newly available inputs and resumes witness generation. Only the unfinished watchers of
    /// newly populated representatives are queued; every other unfinished generator was already
    /// run to quiescence and cannot make progress without new values.
    pub fn feed(&mut self, inputs: PartialWitness<F>) -> Result<()> {
        let mut pending_generator_indices = Vec::new();
        for (t, v) in inputs.target_values.into_iter() {
            if let Some(representative) = self.witness.set_target_returning_rep(t, v)? {
                self.readiness.populate_representative(
                    representative,
                    &self.generator_is_expired,
                    &mut pending_generator_indices,
                    self.prover_data.generators_defer_until_ready,
                );
            }
        }

        run_generator_worklist(
            &mut self.witness,
            self.prover_data,
            &mut self.readiness,
            &mut self.generator_is_expired,
            &mut self.remaining_generators,
            pending_generator_indices,
            self.parallel_threshold,
        )
    }

    /// The [`Self::feed`] analog of [`Self::start_seeded`]: newly available
    /// inputs are written by `seed` directly into the partition through a
    /// [`PartitionFeeder`] — no intermediate `PartialWitness` map is built or
    /// replayed. Resume semantics are identical to [`Self::feed`]: only the
    /// unfinished watchers of newly populated representatives are queued.
    pub fn feed_seeded(
        &mut self,
        seed: impl FnOnce(&mut PartitionFeeder<'a, '_, F>) -> Result<()>,
    ) -> Result<()> {
        let mut pending_generator_indices = Vec::new();
        seed(&mut PartitionFeeder {
            witness: &mut self.witness,
            readiness: &mut self.readiness,
            generator_is_expired: &self.generator_is_expired,
            pending_generator_indices: &mut pending_generator_indices,
            generators_defer_until_ready: self.prover_data.generators_defer_until_ready,
        })?;

        run_generator_worklist(
            &mut self.witness,
            self.prover_data,
            &mut self.readiness,
            &mut self.generator_is_expired,
            &mut self.remaining_generators,
            pending_generator_indices,
            self.parallel_threshold,
        )
    }

    /// Returns the fully populated witness, or an error if some generators still couldn't run.
    pub fn finish(self) -> Result<PartitionWitness<'a, F>> {
        if self.remaining_generators != 0 {
            return Err(anyhow!(
                "{} generators weren't run",
                self.remaining_generators
            ));
        }

        Ok(self.witness)
    }
}

/// A generator participates in the generation of the witness.
pub trait WitnessGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    /// Targets to be "watched" by this generator. Whenever a target in the watch list is populated,
    /// the generator will be queued to run.
    fn watch_list(&self) -> Vec<Target>;

    /// Run this generator, returning a flag indicating whether the generator is finished. If the
    /// flag is true, the generator will never be run again, otherwise it will be queued for another
    /// run next time a target in its watch list is populated.
    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool;

    /// Scheduler entry point carrying a hint that every watched representative is populated.
    ///
    /// General generators may produce values before all watches are populated, so the default
    /// implementation preserves their existing [`Self::run`] behavior. Generators which require
    /// every watch can override this to avoid rediscovering readiness.
    #[doc(hidden)]
    fn run_with_ready_hint(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
        _all_watches_populated: bool,
    ) -> bool {
        self.run(witness, out_buffer)
    }

    /// Whether `run_with_ready_hint(_, _, false)` is guaranteed to be a pure no-op: it writes
    /// nothing to `out_buffer`, reads nothing from the witness, and returns `false`.
    ///
    /// The worklist calls every queued generator once per round even when the generator's
    /// watched representatives are not all populated yet, because the default implementation
    /// above forwards to [`Self::run`], which may legitimately emit partial output on such a
    /// call (see `readiness_hint_preserves_incremental_witness_generator_fallback`). A
    /// generator that opts in here promises that call is inert, which lets the worklist drop
    /// the dispatch entirely instead of paying a scattered load of the boxed generator and an
    /// indirect call to reach a short circuit.
    ///
    /// Conservative default: `false`. Only [`SimpleGeneratorAdapter`] overrides it, and its
    /// `run_with_ready_hint` is literally `all_watches_populated && run_once(..)`, whose `&&`
    /// short-circuits before `run_once` is reached.
    fn defers_until_ready(&self) -> bool {
        false
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

/// A wrapper around an `Box<WitnessGenerator>` which implements `PartialEq`
/// and `Eq` based on generator IDs.
pub struct WitnessGeneratorRef<F: RichField + Extendable<D>, const D: usize>(
    pub Box<dyn WitnessGenerator<F, D>>,
);

impl<F: RichField + Extendable<D>, const D: usize> WitnessGeneratorRef<F, D> {
    pub fn new<G: WitnessGenerator<F, D>>(generator: G) -> WitnessGeneratorRef<F, D> {
        WitnessGeneratorRef(Box::new(generator))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PartialEq for WitnessGeneratorRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Eq for WitnessGeneratorRef<F, D> {}

impl<F: RichField + Extendable<D>, const D: usize> Debug for WitnessGeneratorRef<F, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0.id())
    }
}

/// Values generated by a generator invocation.
#[derive(Debug)]
pub struct GeneratedValues<F: Field> {
    pub target_values: Vec<(Target, F)>,
}

impl<F: Field> From<Vec<(Target, F)>> for GeneratedValues<F> {
    fn from(target_values: Vec<(Target, F)>) -> Self {
        Self { target_values }
    }
}

impl<F: Field> WitnessWrite<F> for GeneratedValues<F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        self.target_values.push((target, value));

        Ok(())
    }
}

impl<F: Field> GeneratedValues<F> {
    pub fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    pub fn empty() -> Self {
        Vec::new().into()
    }

    pub fn singleton_wire(wire: Wire, value: F) -> Self {
        Self::singleton_target(Target::Wire(wire), value)
    }

    pub fn singleton_target(target: Target, value: F) -> Self {
        vec![(target, value)].into()
    }

    pub fn singleton_extension_target<const D: usize>(
        et: ExtensionTarget<D>,
        value: F::Extension,
    ) -> Result<Self>
    where
        F: RichField + Extendable<D>,
    {
        let mut witness = Self::with_capacity(D);
        witness.set_extension_target(et, value)?;

        Ok(witness)
    }
}

/// A generator which runs once after a list of dependencies is present in the witness.
pub trait SimpleGenerator<F: RichField + Extendable<D>, const D: usize>:
    'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    fn dependencies(&self) -> Vec<Target>;

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()>;

    fn adapter(self) -> SimpleGeneratorAdapter<F, Self, D>
    where
        Self: Sized,
    {
        SimpleGeneratorAdapter {
            inner: self,
            _phantom: PhantomData,
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct SimpleGeneratorAdapter<
    F: RichField + Extendable<D>,
    SG: SimpleGenerator<F, D> + ?Sized,
    const D: usize,
> {
    _phantom: PhantomData<F>,
    inner: SG,
}

impl<F: RichField + Extendable<D>, SG: SimpleGenerator<F, D>, const D: usize> WitnessGenerator<F, D>
    for SimpleGeneratorAdapter<F, SG, D>
{
    fn id(&self) -> String {
        self.inner.id()
    }

    fn watch_list(&self) -> Vec<Target> {
        self.inner.dependencies()
    }

    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
        if witness.contains_all(&self.inner.dependencies()) {
            self.inner.run_once(witness, out_buffer).is_ok()
        } else {
            false
        }
    }

    fn run_with_ready_hint(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
        all_watches_populated: bool,
    ) -> bool {
        all_watches_populated && self.inner.run_once(witness, out_buffer).is_ok()
    }

    /// `&&` short-circuits, so a `false` hint returns `false` without reaching `run_once`:
    /// nothing is read and nothing is written. See [`WitnessGenerator::defers_until_ready`].
    fn defers_until_ready(&self) -> bool {
        true
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.inner.serialize(dst, common_data)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            inner: SG::deserialize(src, common_data)?,
            _phantom: PhantomData,
        })
    }
}

/// A generator which copies one wire to another.
#[derive(Debug, Default)]
pub struct CopyGenerator {
    pub(crate) src: Target,
    pub(crate) dst: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for CopyGenerator {
    fn id(&self) -> String {
        "CopyGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.src]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let value = witness.get_target(self.src);
        out_buffer.set_target(self.dst, value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.src)?;
        dst.write_target(self.dst)
    }

    fn deserialize(source: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let src = source.read_target()?;
        let dst = source.read_target()?;
        Ok(Self { src, dst })
    }
}

/// A generator for including a random value
#[derive(Debug, Default)]
pub struct RandomValueGenerator {
    pub(crate) target: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for RandomValueGenerator {
    fn id(&self) -> String {
        "RandomValueGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        Vec::new()
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        // Deterministic instead of `F::rand()`. These targets are the unused
        // public-input-gate wires (`randomize_unused_pi_wires`) and, under
        // `zero_knowledge`, blinding rows — this config never enables ZK. The
        // randomness existed to give a *retry* an independent chance against
        // the astronomically rare permutation-argument division by zero
        // (plonky2 #456); nothing in this prover retries, so a fixed value has
        // the identical single-shot failure probability while making witness
        // generation — and therefore entire proofs — bit-reproducible. That
        // reproducibility is what lets the orchestration differential oracles
        // compare full proof bytes.
        out_buffer.set_target(self.target, F::ZERO)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.target)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let target = src.read_target()?;
        Ok(Self { target })
    }
}

/// A generator for testing if a value equals zero
#[derive(Debug, Default)]
pub struct NonzeroTestGenerator {
    pub(crate) to_test: Target,
    pub(crate) dummy: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for NonzeroTestGenerator {
    fn id(&self) -> String {
        "NonzeroTestGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.to_test]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let to_test_value = witness.get_target(self.to_test);

        let dummy_value = if to_test_value == F::ZERO {
            F::ONE
        } else {
            to_test_value.inverse()
        };

        out_buffer.set_target(self.dummy, dummy_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.to_test)?;
        dst.write_target(self.dummy)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let to_test = src.read_target()?;
        let dummy = src.read_target()?;
        Ok(Self { to_test, dummy })
    }
}

/// Generator used to fill an extra constant.
#[derive(Debug, Clone, Default)]
pub struct ConstantGenerator<F: Field> {
    pub row: usize,
    pub constant_index: usize,
    pub wire_index: usize,
    pub constant: F,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn set_constant(&mut self, c: F) {
        self.constant = c;
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ConstantGenerator<F> {
    fn id(&self) -> String {
        "ConstantGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        out_buffer.set_target(Target::wire(self.row, self.wire_index), self.constant)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.constant_index)?;
        dst.write_usize(self.wire_index)?;
        dst.write_field(self.constant)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let constant_index = src.read_usize()?;
        let wire_index = src.read_usize()?;
        let constant = src.read_field()?;
        Ok(Self {
            row,
            constant_index,
            wire_index,
            constant,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::noop::NoopGate;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    const D: usize = 2;
    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;

    #[derive(Debug)]
    struct CountingSimpleGenerator {
        dependencies: Vec<Target>,
        output: Target,
        dependency_calls: Arc<AtomicUsize>,
        run_calls: Arc<AtomicUsize>,
    }

    impl SimpleGenerator<F, D> for CountingSimpleGenerator {
        fn id(&self) -> String {
            "CountingSimpleGenerator".to_string()
        }

        fn dependencies(&self) -> Vec<Target> {
            self.dependency_calls.fetch_add(1, Ordering::Relaxed);
            self.dependencies.clone()
        }

        fn run_once(
            &self,
            witness: &PartitionWitness<F>,
            out_buffer: &mut GeneratedValues<F>,
        ) -> Result<()> {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            let value = self
                .dependencies
                .iter()
                .map(|&target| witness.get_target(target))
                .sum();
            out_buffer.set_target(self.output, value)
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<()> {
            unreachable!("test generator is never serialized")
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<Self> {
            unreachable!("test generator is never deserialized")
        }
    }

    #[derive(Debug)]
    struct IncrementalGenerator {
        trigger: Target,
        early_output: Target,
        final_output: Target,
        run_calls: Arc<AtomicUsize>,
    }

    impl WitnessGenerator<F, D> for IncrementalGenerator {
        fn id(&self) -> String {
            "IncrementalGenerator".to_string()
        }

        fn watch_list(&self) -> Vec<Target> {
            vec![self.trigger]
        }

        fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
            self.run_calls.fetch_add(1, Ordering::Relaxed);
            if let Some(value) = witness.try_get_target(self.trigger) {
                out_buffer.set_target(self.final_output, value).unwrap();
                true
            } else {
                out_buffer
                    .set_target(self.early_output, F::from_canonical_u64(7))
                    .unwrap();
                false
            }
        }

        fn serialize(
            &self,
            _dst: &mut Vec<u8>,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<()> {
            unreachable!("test generator is never serialized")
        }

        fn deserialize(
            _src: &mut Buffer,
            _common_data: &CommonCircuitData<F, D>,
        ) -> IoResult<Self> {
            unreachable!("test generator is never deserialized")
        }
    }

    #[test]
    fn simple_generator_uses_representative_readiness_without_rescanning_dependencies() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let initial = builder.add_virtual_target();
        let first = builder.constant(F::from_canonical_u64(3));
        let first_alias = builder.add_virtual_target();
        builder.connect(first, first_alias);
        let second = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        let dependency_calls = Arc::new(AtomicUsize::new(0));
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_simple_generator(CountingSimpleGenerator {
            dependencies: vec![initial, initial, first, first_alias, second, second],
            output,
            dependency_calls: Arc::clone(&dependency_calls),
            run_calls: Arc::clone(&run_calls),
        });
        builder.generate_copy(first, second);
        builder.register_public_input(output);

        let circuit = builder.build::<C>();
        let dependency_calls_after_build = dependency_calls.load(Ordering::Relaxed);
        let mut inputs = PartialWitness::new();
        inputs
            .set_target(initial, F::from_canonical_u64(5))
            .unwrap();
        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        assert_eq!(witness.get_target(output), F::from_canonical_u64(22));
        assert_eq!(run_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            dependency_calls.load(Ordering::Relaxed),
            dependency_calls_after_build
        );
    }

    #[test]
    fn deferred_worklist_queues_only_ready_transitions() {
        let watch_index = GeneratorWatchIndex::from_map(Default::default());
        let readiness = TargetReadiness::new(&[0, 3, 1, 0], &watch_index);
        assert_eq!(readiness.initial_worklist(true), vec![0, 3]);
        assert_eq!(readiness.initial_worklist(false), vec![0, 1, 2, 3]);

        let mut unresolved = 2u32;
        assert!(!watch_populated(&mut unresolved, true));
        assert_eq!(unresolved, 1);
        assert!(watch_populated(&mut unresolved, true));
        assert_eq!(unresolved, 0);

        let mut legacy_unresolved = 2u32;
        assert!(watch_populated(&mut legacy_unresolved, false));
        assert_eq!(legacy_unresolved, 1);
    }

    #[test]
    fn dense_target_readiness_preserves_alias_duplicates_and_layout_replay() -> Result<()> {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let seed = builder.add_virtual_target();
        let seed_alias = builder.add_virtual_target();
        builder.connect(seed, seed_alias);
        let late = builder.add_virtual_target();
        let late_alias = builder.add_virtual_target();
        builder.connect(late, late_alias);
        let output = builder.add_virtual_target();
        let dependency_calls = Arc::new(AtomicUsize::new(0));
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_simple_generator(CountingSimpleGenerator {
            dependencies: vec![seed, seed_alias, seed, late, late_alias, late],
            output,
            dependency_calls,
            run_calls: Arc::clone(&run_calls),
        });
        builder.register_public_input(output);
        let circuit = builder.build::<C>();
        let generator = circuit
            .prover_only
            .generators
            .iter()
            .position(|generator| generator.0.id() == "CountingSimpleGenerator")
            .unwrap();
        assert_eq!(circuit.prover_only.generator_watch_counts[generator], 2);

        let seed_values = |seeder: &mut PartitionSeeder<'_, '_, F>| -> Result<()> {
            seeder.set_target(seed, F::from_canonical_u64(5))?;
            seeder.set_target(seed_alias, F::from_canonical_u64(5))?;
            seeder.set_target(seed, F::from_canonical_u64(5))
        };
        let feed_values = |feeder: &mut PartitionFeeder<'_, '_, F>| -> Result<()> {
            feeder.set_target(late, F::from_canonical_u64(7))?;
            feeder.set_target(late_alias, F::from_canonical_u64(7))?;
            feeder.set_target(late, F::from_canonical_u64(7))
        };

        let (mut recorded, layout) = PendingPartitionWitness::start_seeded_recording(
            &circuit.prover_only,
            &circuit.common,
            seed_values,
        )?;
        assert_eq!(layout.changed_generator_count(), 1);
        recorded.feed_seeded(feed_values)?;
        let recorded = recorded.finish()?;

        let mut replayed = PendingPartitionWitness::start_seeded_with_layout(
            &circuit.prover_only,
            &circuit.common,
            &layout,
            seed_values,
        )?;
        replayed.feed_seeded(feed_values)?;
        let replayed = replayed.finish()?;

        assert_eq!(recorded.get_target(output), F::from_canonical_u64(36));
        assert_eq!(recorded.set_bitmap, replayed.set_bitmap);
        for representative in 0..recorded.values.len() {
            if recorded.is_set_by_rep_index(representative) {
                assert_eq!(
                    recorded.values[representative],
                    replayed.values[representative]
                );
            }
        }
        assert_eq!(run_calls.load(Ordering::Relaxed), 2);

        let recorded_proof = crate::plonk::prover::prove_with_partition_witness(
            &circuit.prover_only,
            &circuit.common,
            recorded,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        let replayed_proof = crate::plonk::prover::prove_with_partition_witness(
            &circuit.prover_only,
            &circuit.common,
            replayed,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        circuit.verify(recorded_proof.clone())?;
        circuit.verify(replayed_proof.clone())?;
        // A BITWISE proof comparison is not a valid assertion in this tree, and
        // the original form of this test (`assert_eq!(recorded_proof,
        // replayed_proof)`) fails 3/3 under rayon while passing with
        // RAYON_NUM_THREADS=1. Two independent proving runs of the SAME witness
        // differ for two reasons, neither of them a record/replay defect:
        // `fri_proof_of_work` searches the nonce space with `find_any`, so the
        // PoW witness is whichever nonce a racing thread reaches first; and the
        // two paths leave different residue in the UNSET (unconstrained) wire
        // positions, which the commitment absorbs into the caps. Normalising the
        // nonce alone is therefore not enough.
        //
        // What the record/replay contract actually claims is asserted in full
        // above -- identical set bitmap and identical value at every SET
        // representative -- plus the two `circuit.verify` calls here. The public
        // inputs are the deterministic, contractual part of the proof, and
        // public-input equality is the same oracle this campaign already uses for
        // the chain-step replay harness (`chain_step_two_phase_timing`).
        assert_eq!(recorded_proof.public_inputs, replayed_proof.public_inputs);
        Ok(())
    }

    #[test]
    fn ready_only_worklist_matches_legacy_queue_for_simple_dag() -> Result<()> {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let seed = builder.add_virtual_target();
        let late = builder.add_virtual_target();
        let first = builder.add_virtual_target();
        let second = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        let dependency_calls = Arc::new(AtomicUsize::new(0));
        let run_calls = Arc::new(AtomicUsize::new(0));

        for (dependencies, output) in [
            (vec![seed], first),
            (vec![first], second),
            (vec![second, late], output),
        ] {
            builder.add_simple_generator(CountingSimpleGenerator {
                dependencies,
                output,
                dependency_calls: Arc::clone(&dependency_calls),
                run_calls: Arc::clone(&run_calls),
            });
        }
        builder.register_public_input(output);

        let mut circuit = builder.build::<C>();
        assert!(circuit.prover_only.generators_defer_until_ready);

        let mut early_inputs = PartialWitness::new();
        early_inputs.set_target(seed, F::from_canonical_u64(5))?;
        let mut late_inputs = PartialWitness::new();
        late_inputs.set_target(late, F::from_canonical_u64(7))?;

        // Force the conservative fallback on the same immutable circuit topology. It queues every
        // generator on the first round and on each watch transition, matching the pre-optimization
        // scheduler while retaining the SimpleGenerator adapter's value semantics.
        circuit.prover_only.generators_defer_until_ready = false;
        let mut legacy = PendingPartitionWitness::start(
            early_inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )?;
        legacy.feed(late_inputs.clone())?;
        let legacy = legacy.finish()?;
        assert_eq!(legacy.get_target(output), F::from_canonical_u64(12));
        let legacy_proof = crate::plonk::prover::prove_with_partition_witness(
            &circuit.prover_only,
            &circuit.common,
            legacy,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        circuit.verify(legacy_proof.clone())?;

        circuit.prover_only.generators_defer_until_ready = true;
        let mut ready_only =
            PendingPartitionWitness::start(early_inputs, &circuit.prover_only, &circuit.common)?;
        ready_only.feed(late_inputs)?;
        let ready_only = ready_only.finish()?;
        assert_eq!(ready_only.get_target(output), F::from_canonical_u64(12));
        let ready_only_proof = crate::plonk::prover::prove_with_partition_witness(
            &circuit.prover_only,
            &circuit.common,
            ready_only,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        circuit.verify(ready_only_proof.clone())?;

        assert_eq!(legacy_proof.public_inputs, ready_only_proof.public_inputs);
        assert_eq!(run_calls.load(Ordering::Relaxed), 6);
        Ok(())
    }

    /// Builds an outer circuit verifying two independent inner proofs, mirroring a chain step's
    /// tx-proof/cyclic-proof pair. Returns the outer circuit and the two input halves.
    fn two_inner_proof_fixture() -> Result<(
        crate::plonk::circuit_data::CircuitData<F, C, D>,
        PartialWitness<F>,
        PartialWitness<F>,
    )> {
        let config = CircuitConfig::standard_recursion_config();

        // Inner circuit: expose x^2 as a public input.
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let x = builder.add_virtual_target();
        let x_squared = builder.mul(x, x);
        builder.register_public_input(x_squared);
        for _ in 0..1_000 {
            builder.add_gate(NoopGate, vec![]);
        }
        let inner = builder.build::<C>();

        let mut inputs = PartialWitness::new();
        inputs.set_target(x, F::from_canonical_u64(3))?;
        let inner_proof_a = inner.prove(inputs)?;
        let mut inputs = PartialWitness::new();
        inputs.set_target(x, F::from_canonical_u64(5))?;
        let inner_proof_b = inner.prove(inputs)?;

        let mut builder = CircuitBuilder::<F, D>::new(config);
        let proof_target_a = builder.add_virtual_proof_with_pis(&inner.common);
        let proof_target_b = builder.add_virtual_proof_with_pis(&inner.common);
        let verifier_data = builder.constant_verifier_data(&inner.verifier_only);
        builder.verify_proof::<C>(&proof_target_a, &verifier_data, &inner.common);
        builder.verify_proof::<C>(&proof_target_b, &verifier_data, &inner.common);
        builder.register_public_inputs(&proof_target_a.public_inputs);
        builder.register_public_inputs(&proof_target_b.public_inputs);
        let outer = builder.build::<C>();

        let mut early_inputs = PartialWitness::new();
        early_inputs.set_proof_with_pis_target(&proof_target_a, &inner_proof_a)?;
        let mut late_inputs = PartialWitness::new();
        late_inputs.set_proof_with_pis_target(&proof_target_b, &inner_proof_b)?;

        Ok((outer, early_inputs, late_inputs))
    }

    /// A recorded layout must reproduce the generic seeded path *value for
    /// value on raw `u64` limbs*, and every way of breaking it must fail closed
    /// with a typed mismatch rather than seed a wrong witness. The sabotage
    /// blocks are the controls: each one perturbs exactly one element of the
    /// recorded layout and requires the replay to reject it.
    #[test]
    fn recorded_seed_layout_replays_raw_identically_and_fails_closed() -> Result<()> {
        use crate::field::types::PrimeField64;

        let (outer, early_inputs, late_inputs) = two_inner_proof_fixture()?;
        let prover_data = &outer.prover_only;
        let common_data = &outer.common;
        let all_inputs = merged_inputs(&early_inputs, &late_inputs)?;
        let write = |inputs: &PartialWitness<F>| {
            let pairs: Vec<(Target, F)> = {
                // A deterministic write order, so the layout is well defined.
                let mut pairs: Vec<(Target, F)> =
                    inputs.target_values.iter().map(|(&t, &v)| (t, v)).collect();
                pairs.sort_by_key(|(t, _)| format!("{t:?}"));
                pairs
            };
            move |seeder: &mut PartitionSeeder<'_, '_, F>| -> Result<()> {
                for (t, v) in &pairs {
                    seeder.set_target(*t, *v)?;
                }
                Ok(())
            }
        };

        let raw = |witness: &PartitionWitness<'_, F>| -> Vec<u64> {
            witness
                .values
                .iter()
                .enumerate()
                .filter(|(i, _)| witness.is_set_by_rep_index(*i))
                .map(|(_, v)| v.to_noncanonical_u64())
                .collect()
        };

        // Reference: the generic seeded path this replaces.
        let reference =
            PendingPartitionWitness::start_seeded(prover_data, common_data, write(&all_inputs))
                .and_then(PendingPartitionWitness::finish)?;

        // Record on one input set, replay on the same shape.
        let (recorded_pending, layout) = PendingPartitionWitness::start_seeded_recording(
            prover_data,
            common_data,
            write(&all_inputs),
        )?;
        let recorded = recorded_pending.finish()?;
        assert_eq!(recorded.set_bitmap, reference.set_bitmap);
        assert_eq!(
            raw(&recorded),
            raw(&reference),
            "recorded seed diverges raw"
        );
        assert!(layout.target_write_count() > 0);

        let replayed = PendingPartitionWitness::start_seeded_with_layout(
            prover_data,
            common_data,
            &layout,
            write(&all_inputs),
        )
        .and_then(PendingPartitionWitness::finish)?;
        assert_eq!(replayed.set_bitmap, reference.set_bitmap);
        assert_eq!(
            raw(&replayed),
            raw(&reference),
            "replayed seed diverges raw"
        );

        // Sabotage 1: a truncated layout. Replay must reject, not seed a
        // partial witness.
        let mut short = PendingPartitionWitness::start_seeded_recording(
            prover_data,
            common_data,
            write(&all_inputs),
        )?
        .1;
        short
            .representatives
            .truncate(short.representatives.len() - 1);
        let error = PendingPartitionWitness::start_seeded_with_layout(
            prover_data,
            common_data,
            &short,
            write(&all_inputs),
        )
        .err()
        .expect("a truncated layout must be rejected");
        assert!(is_seed_layout_mismatch(&error), "{error:?}");

        // Sabotage 2: a permuted layout. Not a reachable state -- for a
        // pointer-identical prover data the representatives are a deterministic
        // function of the target sequence the checksum already covers -- but it
        // is the strongest corruption available, so require that it can never
        // silently yield the reference witness. It fails closed either as a
        // replay-write mismatch or as a generator-worklist contradiction.
        let mut swapped = PendingPartitionWitness::start_seeded_recording(
            prover_data,
            common_data,
            write(&all_inputs),
        )?
        .1;
        let n = swapped.representatives.len();
        swapped.representatives.swap(0, n - 1);
        let swapped_result = PendingPartitionWitness::start_seeded_with_layout(
            prover_data,
            common_data,
            &swapped,
            write(&all_inputs),
        )
        .and_then(PendingPartitionWitness::finish);
        match swapped_result {
            Err(_) => {}
            Ok(witness) => assert_ne!(
                raw(&witness),
                raw(&reference),
                "sabotage control did not trip: a permuted layout produced the reference witness"
            ),
        }

        // Sabotage 3: a corrupted target checksum. This is the check that
        // replaces the rival's per-entry `Target` comparison, so it must be the
        // thing that catches a drifted target sequence.
        let mut wrong_checksum = PendingPartitionWitness::start_seeded_recording(
            prover_data,
            common_data,
            write(&all_inputs),
        )?
        .1;
        wrong_checksum.target_checksum ^= 1;
        let error = PendingPartitionWitness::start_seeded_with_layout(
            prover_data,
            common_data,
            &wrong_checksum,
            write(&all_inputs),
        )
        .err()
        .expect("a drifted target sequence must be rejected");
        assert!(is_seed_layout_mismatch(&error), "{error:?}");

        // Sabotage 4: a corrupted aggregate watch decrement must be caught
        // rather than silently changing the generator schedule.
        let mut wrong_decrements = PendingPartitionWitness::start_seeded_recording(
            prover_data,
            common_data,
            write(&all_inputs),
        )?
        .1;
        assert!(!wrong_decrements.seeded_watch_decrements.is_empty());
        wrong_decrements.seeded_watch_decrements[0].1 = u32::MAX;
        let error = PendingPartitionWitness::start_seeded_with_layout(
            prover_data,
            common_data,
            &wrong_decrements,
            write(&all_inputs),
        )
        .err()
        .expect("an impossible watcher decrement must be rejected");
        assert!(is_seed_layout_mismatch(&error), "{error:?}");

        // Sabotage 5: a layout from a different circuit instance.
        let (other, other_early, other_late) = two_inner_proof_fixture()?;
        let other_all = merged_inputs(&other_early, &other_late)?;
        let other_layout = PendingPartitionWitness::start_seeded_recording(
            &other.prover_only,
            &other.common,
            write(&other_all),
        )?
        .1;
        let error = PendingPartitionWitness::start_seeded_with_layout(
            prover_data,
            common_data,
            &other_layout,
            write(&all_inputs),
        )
        .err()
        .expect("a foreign circuit instance must be rejected");
        assert!(is_seed_layout_mismatch(&error), "{error:?}");

        Ok(())
    }

    fn merged_inputs(
        early_inputs: &PartialWitness<F>,
        late_inputs: &PartialWitness<F>,
    ) -> Result<PartialWitness<F>> {
        let mut inputs = early_inputs.clone();
        for (&t, &v) in &late_inputs.target_values {
            inputs.set_target(t, v)?;
        }
        Ok(inputs)
    }

    /// The initialization that `seed_inputs_and_readiness` replaced: seed every input,
    /// then walk the entire representative-keyed watcher map counting, per generator, the
    /// still-unpopulated representatives it watches. Kept as an in-test oracle.
    fn legacy_seed_inputs_and_unresolved_watches<C: GenericConfig<D, F = F>, const D: usize>(
        witness: &mut PartitionWitness<F>,
        inputs: PartialWitness<F>,
        prover_data: &ProverOnlyCircuitData<F, C, D>,
    ) -> Result<Vec<u32>>
    where
        F: RichField + Extendable<D>,
    {
        for (t, v) in inputs.target_values.into_iter() {
            witness.set_target(t, v)?;
        }

        let mut unresolved_watches = vec![0u32; prover_data.generators.len()];
        for (watch, watchers) in prover_data.generator_indices_by_watches.iter() {
            if !witness.is_set_by_rep_index(watch) {
                for &generator_idx in watchers {
                    unresolved_watches[generator_idx as usize] += 1;
                }
            }
        }
        Ok(unresolved_watches)
    }

    /// M2 differential: the precomputed-count initialization must produce exactly the vector the
    /// removed whole-map scan produced, for every seeding of the inputs. `unresolved_watches` is
    /// the only state M2 touches, so equality here implies an identical generator schedule.
    #[test]
    fn precomputed_watch_counts_match_legacy_map_scan() -> Result<()> {
        let (outer, early_inputs, late_inputs) = two_inner_proof_fixture()?;
        let prover_data = &outer.prover_only;

        // The builder-derived counts must equal the number of watcher-list occurrences of each
        // generator across the whole map (the "no representative is populated yet" case).
        let mut occurrences = vec![0usize; prover_data.generators.len()];
        for (_, watchers) in prover_data.generator_indices_by_watches.iter() {
            for &generator_idx in watchers {
                occurrences[generator_idx as usize] += 1;
            }
        }
        assert_eq!(
            prover_data.generator_watch_counts, occurrences,
            "builder-derived watch counts disagree with the watcher index"
        );

        // Watcher lists are deduplicated, so a count is the number of *distinct* representatives.
        for (_, watchers) in prover_data.generator_indices_by_watches.iter() {
            let mut sorted = watchers.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted, watchers, "watcher list is not deduplicated/sorted");
        }

        for inputs in [
            PartialWitness::new(),
            early_inputs.clone(),
            late_inputs.clone(),
            merged_inputs(&early_inputs, &late_inputs)?,
        ] {
            let mut new_witness = PartitionWitness::new(
                outer.common.config.num_wires,
                outer.common.degree(),
                &prover_data.representative_map,
            );
            let new_counts = seed_inputs_and_readiness(
                &mut new_witness,
                inputs.clone(),
                &prover_data.generator_watch_counts,
                &prover_data.generator_indices_by_watches,
            )?;

            let mut legacy_witness = PartitionWitness::new(
                outer.common.config.num_wires,
                outer.common.degree(),
                &prover_data.representative_map,
            );
            let legacy_counts = legacy_seed_inputs_and_unresolved_watches(
                &mut legacy_witness,
                inputs,
                prover_data,
            )?;

            assert_eq!(
                new_counts.unresolved, legacy_counts,
                "unresolved-watch counts diverge from the legacy map scan"
            );
            assert_eq!(new_witness.set_bitmap, legacy_witness.set_bitmap);
            // Unset slots are uninitialized storage; compare set slots only.
            for rep in 0..new_witness.values.len() {
                if new_witness.is_set_by_rep_index(rep) {
                    assert_eq!(new_witness.values[rep], legacy_witness.values[rep]);
                }
            }
        }

        Ok(())
    }

    /// The precomputed count is the number of *distinct* representatives, so duplicated
    /// dependencies and copy-constraint aliases of the same value collapse to one.
    #[test]
    fn watch_counts_collapse_duplicates_and_aliases() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let initial = builder.add_virtual_target();
        let first = builder.constant(F::from_canonical_u64(3));
        let first_alias = builder.add_virtual_target();
        builder.connect(first, first_alias);
        let second = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        let dependency_calls = Arc::new(AtomicUsize::new(0));
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_simple_generator(CountingSimpleGenerator {
            // Six dependencies over three distinct representatives: `initial` twice, `first` and
            // `first_alias` (copy-constrained), and `second` twice (connected to `first`).
            dependencies: vec![initial, initial, first, first_alias, second, second],
            output,
            dependency_calls: Arc::clone(&dependency_calls),
            run_calls: Arc::clone(&run_calls),
        });
        builder.generate_copy(first, second);
        builder.register_public_input(output);

        let circuit = builder.build::<C>();
        let generator_idx = circuit
            .prover_only
            .generators
            .iter()
            .position(|g| g.0.id() == "CountingSimpleGenerator")
            .expect("test generator missing");
        // `first` and `first_alias` are copy-constrained into one representative; `second` is
        // populated by a `CopyGenerator` and keeps its own. Three distinct watches, not six.
        assert_eq!(
            circuit.prover_only.generator_watch_counts[generator_idx], 3,
            "duplicated and aliased dependencies were not collapsed"
        );

        let mut inputs = PartialWitness::new();
        inputs
            .set_target(initial, F::from_canonical_u64(5))
            .unwrap();
        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();
        assert_eq!(witness.get_target(output), F::from_canonical_u64(22));
        assert_eq!(run_calls.load(Ordering::Relaxed), 1);
    }

    /// M1 differential: the fused `full_witness` drain over the narrowed `u32` map must produce
    /// the same matrix, cell for cell as raw canonical `u64`s, as resolving every wire through
    /// the per-target read path.
    #[test]
    fn full_witness_matches_per_target_reads() -> Result<()> {
        use crate::field::types::PrimeField64;

        let (outer, early_inputs, late_inputs) = two_inner_proof_fixture()?;
        let inputs = merged_inputs(&early_inputs, &late_inputs)?;
        let witness = generate_partial_witness(inputs, &outer.prover_only, &outer.common)?;

        let num_wires = outer.common.config.num_wires;
        let degree = outer.common.degree();
        let expected: Vec<Vec<u64>> = (0..num_wires)
            .map(|column| {
                (0..degree)
                    .map(|row| {
                        witness
                            .try_get_target(Target::Wire(Wire { row, column }))
                            .unwrap_or(F::ZERO)
                            .to_canonical_u64()
                    })
                    .collect()
            })
            .collect();

        let matrix = witness.full_witness();
        let actual: Vec<Vec<u64>> = matrix
            .wire_values
            .iter()
            .map(|column| column.iter().map(|v| v.to_canonical_u64()).collect())
            .collect();

        assert_eq!(actual, expected);
        Ok(())
    }

    fn count_random_generators(
        prover_only: &crate::plonk::circuit_data::ProverOnlyCircuitData<F, C, D>,
    ) -> usize {
        prover_only
            .generators
            .iter()
            .filter(|generator| generator.0.id() == "RandomValueGenerator")
            .count()
    }

    #[test]
    fn pending_partition_witness_matches_single_shot_for_recursive_circuit() -> Result<()> {
        let (outer, early_inputs, late_inputs) = two_inner_proof_fixture()?;
        let single_shot_inputs = merged_inputs(&early_inputs, &late_inputs)?;

        let single_shot = generate_partial_witness(
            single_shot_inputs.clone(),
            &outer.prover_only,
            &outer.common,
        )?;
        // Every witness position is deterministic except the outputs of the circuit's
        // `RandomValueGenerator`s (unused public-input-gate wires): a second single-shot run
        // isolates exactly those positions.
        let single_shot_repeat =
            generate_partial_witness(single_shot_inputs, &outer.prover_only, &outer.common)?;
        let num_random_generators = count_random_generators(&outer.prover_only);

        let mut pending =
            PendingPartitionWitness::start(early_inputs, &outer.prover_only, &outer.common)?;
        // A feed with no new targets must be a no-op.
        pending.feed(PartialWitness::new())?;
        pending.feed(late_inputs)?;
        let two_phase = pending.finish()?;

        let mut nondeterministic_positions = 0usize;
        // `values` slots are uninitialized storage unless their bitmap bit is
        // set, so compare only slots every witness actually set; unset slots
        // are F::ZERO at every real observation point (full_witness,
        // try_get_target) and carry no information here.
        for (rep, (single, split)) in single_shot.values.iter().zip(&two_phase.values).enumerate() {
            if !(single_shot.is_set_by_rep_index(rep)
                && single_shot_repeat.is_set_by_rep_index(rep)
                && two_phase.is_set_by_rep_index(rep))
            {
                continue;
            }
            let repeat = &single_shot_repeat.values[rep];
            if single == repeat {
                assert_eq!(single, split);
            } else {
                nondeterministic_positions += 1;
            }
        }
        assert!(
            nondeterministic_positions <= num_random_generators,
            "{nondeterministic_positions} nondeterministic positions exceed the {num_random_generators} random generators"
        );

        let single_shot_proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            single_shot,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        let two_phase_proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            two_phase,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        outer.verify(single_shot_proof)?;
        outer.verify(two_phase_proof)
    }

    #[test]
    fn direct_seeded_pending_witness_matches_map_seeded_recursive_circuit() -> Result<()> {
        let (outer, early_inputs, late_inputs) = two_inner_proof_fixture()?;

        let mut map_seeded = PendingPartitionWitness::start(
            early_inputs.clone(),
            &outer.prover_only,
            &outer.common,
        )?;
        map_seeded.feed(late_inputs.clone())?;
        let map_seeded = map_seeded.finish()?;

        // RandomValueGenerator outputs are expected to differ between the two
        // executions, but all remaining witness slots must match.
        let mut map_seeded_repeat = PendingPartitionWitness::start(
            early_inputs.clone(),
            &outer.prover_only,
            &outer.common,
        )?;
        map_seeded_repeat.feed(late_inputs.clone())?;
        let map_seeded_repeat = map_seeded_repeat.finish()?;
        let num_random_generators = count_random_generators(&outer.prover_only);

        let mut direct_seeded =
            PendingPartitionWitness::start_seeded(&outer.prover_only, &outer.common, |seeder| {
                for (&target, &value) in &early_inputs.target_values {
                    seeder.set_target(target, value)?;
                }
                Ok(())
            })?;
        direct_seeded.feed_seeded(|feeder| {
            for (&target, &value) in &late_inputs.target_values {
                feeder.set_target(target, value)?;
            }
            Ok(())
        })?;
        let direct_seeded = direct_seeded.finish()?;

        let mut nondeterministic_positions = 0usize;
        // Unset slots are uninitialized storage; compare bitmap-set slots only
        // (same masking as the other witness-equality tests in this module).
        for rep in 0..map_seeded.values.len() {
            if !(map_seeded.is_set_by_rep_index(rep)
                && map_seeded_repeat.is_set_by_rep_index(rep)
                && direct_seeded.is_set_by_rep_index(rep))
            {
                continue;
            }
            let map = &map_seeded.values[rep];
            let map_repeat = &map_seeded_repeat.values[rep];
            let direct = &direct_seeded.values[rep];
            if map == map_repeat {
                assert_eq!(map, direct);
            } else {
                nondeterministic_positions += 1;
            }
        }
        assert!(
            nondeterministic_positions <= num_random_generators,
            "{nondeterministic_positions} nondeterministic positions exceed the {num_random_generators} random generators"
        );

        let map_seeded_proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            map_seeded,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        outer.verify(map_seeded_proof)?;

        let proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            direct_seeded,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        outer.verify(proof)
    }

    #[test]
    fn parallel_worklist_matches_sequential_for_recursive_circuit() -> Result<()> {
        let (outer, early_inputs, late_inputs) = two_inner_proof_fixture()?;
        let full_inputs = merged_inputs(&early_inputs, &late_inputs)?;

        // Sequential reference: a threshold no round can reach forces the sequential loop.
        // A second sequential run isolates the `RandomValueGenerator` positions.
        let sequential = PendingPartitionWitness::start_with_threshold(
            full_inputs.clone(),
            &outer.prover_only,
            &outer.common,
            usize::MAX,
        )?
        .finish()?;
        let sequential_repeat = PendingPartitionWitness::start_with_threshold(
            full_inputs.clone(),
            &outer.prover_only,
            &outer.common,
            usize::MAX,
        )?
        .finish()?;
        let num_random_generators = count_random_generators(&outer.prover_only);

        // Without the context guard the stress threshold must still take the sequential path.
        let ungated = PendingPartitionWitness::start_with_threshold(
            full_inputs.clone(),
            &outer.prover_only,
            &outer.common,
            1,
        )?
        .finish()?;

        // Parallel (guard held) at the default threshold, at a stress threshold that
        // parallelizes every round, and split across start/feed with the stress threshold.
        let parallel_guard = ParallelWitnessGuard::new();
        let parallel_default =
            PendingPartitionWitness::start(full_inputs.clone(), &outer.prover_only, &outer.common)?
                .finish()?;
        let parallel_stress = PendingPartitionWitness::start_with_threshold(
            full_inputs,
            &outer.prover_only,
            &outer.common,
            1,
        )?
        .finish()?;
        let mut pending = PendingPartitionWitness::start_with_threshold(
            early_inputs,
            &outer.prover_only,
            &outer.common,
            1,
        )?;
        pending.feed(late_inputs)?;
        let parallel_two_phase = pending.finish()?;
        drop(parallel_guard);

        let mut nondeterministic_positions = 0usize;
        for position in 0..sequential.values.len() {
            // Unset slots are uninitialized storage; only bitmap-set slots
            // carry witness values (all five witnesses set the same slots —
            // their bitmaps are checked for agreement below via full use).
            if !(sequential.is_set_by_rep_index(position)
                && sequential_repeat.is_set_by_rep_index(position)
                && ungated.is_set_by_rep_index(position)
                && parallel_default.is_set_by_rep_index(position)
                && parallel_stress.is_set_by_rep_index(position)
                && parallel_two_phase.is_set_by_rep_index(position))
            {
                continue;
            }
            if sequential.values[position] == sequential_repeat.values[position] {
                assert_eq!(sequential.values[position], ungated.values[position]);
                assert_eq!(
                    sequential.values[position],
                    parallel_default.values[position]
                );
                assert_eq!(
                    sequential.values[position],
                    parallel_stress.values[position]
                );
                assert_eq!(
                    sequential.values[position],
                    parallel_two_phase.values[position]
                );
            } else {
                nondeterministic_positions += 1;
            }
        }
        assert!(
            nondeterministic_positions <= num_random_generators,
            "{nondeterministic_positions} nondeterministic positions exceed the {num_random_generators} random generators"
        );

        let parallel_proof = crate::plonk::prover::prove_with_partition_witness(
            &outer.prover_only,
            &outer.common,
            parallel_stress,
            &mut crate::util::timing::TimingTree::default(),
        )?;
        outer.verify(parallel_proof)
    }

    #[test]
    #[cfg(all(feature = "parallel", feature = "std"))]
    fn parallel_rounds_require_context_flag() {
        let pool_is_parallel = rayon::current_num_threads() > 1;

        // Off by default, on only while a guard is alive, and guards nest and restore.
        assert!(!parallel_rounds_enabled());
        let outer_guard = ParallelWitnessGuard::new();
        assert_eq!(parallel_rounds_enabled(), pool_is_parallel);
        {
            let inner_guard = ParallelWitnessGuard::new();
            assert_eq!(parallel_rounds_enabled(), pool_is_parallel);
            drop(inner_guard);
        }
        assert_eq!(parallel_rounds_enabled(), pool_is_parallel);
        drop(outer_guard);
        assert!(!parallel_rounds_enabled());
    }

    /// Manual timing harness for the adaptive parallel worklist. Run with:
    /// `cargo test --release -p plonky2 --lib -- --ignored parallel_worklist_synthetic --nocapture`
    /// and vary `RAYON_NUM_THREADS` for thread scaling.
    #[test]
    #[ignore = "manual timing harness; run with --release"]
    fn parallel_worklist_synthetic_fanout_timing() -> Result<()> {
        use std::time::Instant;

        use crate::hash::poseidon::PoseidonHash;

        // Many independent Poseidon chains: every round has ~CHAINS ready hash generators, the
        // shape of a witness-heavy fanned-out workload (the ranked block witness).
        const CHAINS: usize = 256;
        const LENGTH: usize = 48;

        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let seeds: Vec<Target> = (0..CHAINS).map(|_| builder.add_virtual_target()).collect();
        for &seed in &seeds {
            let mut state = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![seed]);
            for _ in 1..LENGTH {
                state = builder.hash_n_to_hash_no_pad::<PoseidonHash>(state.elements.to_vec());
            }
            builder.register_public_inputs(&state.elements);
        }
        let circuit = builder.build::<C>();
        println!(
            "synthetic circuit: degree {}, {} generators",
            circuit.common.degree(),
            circuit.prover_only.generators.len()
        );

        let mut inputs = PartialWitness::new();
        for (chain, &seed) in seeds.iter().enumerate() {
            inputs.set_target(seed, F::from_canonical_u64(chain as u64))?;
        }

        let measure = |label: &str, threshold: usize, parallel_context: bool| -> Result<()> {
            let _guard = parallel_context.then(ParallelWitnessGuard::new);
            let mut best = None;
            for _ in 0..5 {
                let round_inputs = inputs.clone();
                let start = Instant::now();
                let witness = PendingPartitionWitness::start_with_threshold(
                    round_inputs,
                    &circuit.prover_only,
                    &circuit.common,
                    threshold,
                )?
                .finish()?;
                let elapsed = start.elapsed();
                drop(witness);
                best = Some(best.map_or(elapsed, |b: core::time::Duration| b.min(elapsed)));
            }
            println!("{label}: {:?}", best.unwrap());
            Ok(())
        };

        measure("sequential (threshold=MAX)", usize::MAX, false)?;
        for &threshold in &[16usize, 64, 256, 1024] {
            measure(&format!("parallel threshold={threshold}"), threshold, true)?;
        }

        Ok(())
    }

    #[test]
    fn pending_partition_witness_finish_and_feed_errors() -> Result<()> {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let product = builder.mul(a, b);
        builder.register_public_input(product);
        let circuit = builder.build::<C>();

        let mut early_inputs = PartialWitness::new();
        early_inputs.set_target(a, F::from_canonical_u64(3))?;

        // Finishing before all inputs are fed reports the unrun generators.
        let pending = PendingPartitionWitness::start(
            early_inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )?;
        let error = pending.finish().unwrap_err();
        assert!(
            error.to_string().contains("generators weren't run"),
            "unexpected finish error: {error:?}"
        );

        // Feeding a value contradicting an already-set target fails.
        let mut pending = PendingPartitionWitness::start(
            early_inputs.clone(),
            &circuit.prover_only,
            &circuit.common,
        )?;
        let mut contradictory_inputs = PartialWitness::new();
        contradictory_inputs.set_target(a, F::from_canonical_u64(4))?;
        assert!(pending.feed(contradictory_inputs).is_err());

        // Feeding the missing input completes witness generation.
        let mut pending =
            PendingPartitionWitness::start(early_inputs, &circuit.prover_only, &circuit.common)?;
        let mut late_inputs = PartialWitness::new();
        late_inputs.set_target(b, F::from_canonical_u64(5))?;
        pending.feed(late_inputs)?;
        let witness = pending.finish()?;
        assert_eq!(witness.get_target(product), F::from_canonical_u64(15));

        Ok(())
    }

    #[test]
    fn readiness_hint_preserves_incremental_witness_generator_fallback() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let trigger = builder.constant(F::from_canonical_u64(11));
        let early_output = builder.add_virtual_target();
        let final_output = builder.add_virtual_target();
        let run_calls = Arc::new(AtomicUsize::new(0));

        builder.add_generators(vec![WitnessGeneratorRef::new(IncrementalGenerator {
            trigger,
            early_output,
            final_output,
            run_calls: Arc::clone(&run_calls),
        })]);
        builder.register_public_inputs(&[early_output, final_output]);

        let circuit = builder.build::<C>();
        assert!(!circuit.prover_only.generators_defer_until_ready);
        let witness =
            generate_partial_witness(PartialWitness::new(), &circuit.prover_only, &circuit.common)
                .unwrap();

        assert_eq!(witness.get_target(early_output), F::from_canonical_u64(7));
        assert_eq!(witness.get_target(final_output), F::from_canonical_u64(11));
        assert_eq!(run_calls.load(Ordering::Relaxed), 2);
    }
}
