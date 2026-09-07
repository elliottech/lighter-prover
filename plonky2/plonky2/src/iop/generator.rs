#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    collections::BTreeMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Debug;
use core::marker::PhantomData;
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
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
    unresolved_watches: &mut [usize],
    generator_is_expired: &mut [bool],
    remaining_generators: &mut usize,
    mut pending_generator_indices: Vec<usize>,
    parallel_threshold: usize,
) -> Result<()> {
    let generators = &prover_data.generators;
    let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

    let parallel_rounds = parallel_rounds_enabled();
    let mut buffer = GeneratedValues::empty();

    // Keep running generators until we fail to make progress.
    while !pending_generator_indices.is_empty() {
        let mut next_pending_generator_indices = Vec::new();

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
            let round_unresolved_watches: &[usize] = unresolved_watches;
            let round_generator_is_expired: &[bool] = generator_is_expired;
            #[allow(clippy::type_complexity)]
            let round_outputs: Vec<(
                Vec<(usize, bool, usize)>,
                Vec<(Target, F, Option<&Vec<usize>>)>,
            )> = pending_generator_indices
                .par_chunks(PARALLEL_WORKLIST_CHUNK)
                .map(|chunk| {
                    let mut entries = Vec::with_capacity(chunk.len());
                    let mut annotated_values = Vec::new();
                    let mut round_buffer = GeneratedValues::empty();
                    for &generator_idx in chunk {
                        if round_generator_is_expired[generator_idx] {
                            continue;
                        }
                        let finished = generators[generator_idx].0.run_with_ready_hint(
                            round_witness,
                            &mut round_buffer,
                            round_unresolved_watches[generator_idx] == 0,
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
                            annotated_values.push((t, v, watchers));
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

                    for (t, v, watchers) in annotated_values.by_ref().take(value_count) {
                        if witness.set_target_returning_rep(t, v)?.is_none() {
                            continue;
                        }
                        if let Some(watchers) = watchers {
                            for &watching_generator_idx in watchers {
                                if !generator_is_expired[watching_generator_idx] {
                                    debug_assert_ne!(unresolved_watches[watching_generator_idx], 0);
                                    unresolved_watches[watching_generator_idx] -= 1;
                                    next_pending_generator_indices.push(watching_generator_idx);
                                }
                            }
                        }
                    }
                }
            }

            pending_generator_indices = next_pending_generator_indices;
            continue;
        }

        for &generator_idx in &pending_generator_indices {
            if generator_is_expired[generator_idx] {
                continue;
            }

            let finished = generators[generator_idx].0.run_with_ready_hint(
                witness,
                &mut buffer,
                unresolved_watches[generator_idx] == 0,
            );
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
                if let Some(watch) = witness.set_target_returning_rep(t, v)? {
                    if let Some(watchers) = generator_indices_by_watches.get(&watch) {
                        for &watching_generator_idx in watchers {
                            if !generator_is_expired[watching_generator_idx] {
                                debug_assert_ne!(unresolved_watches[watching_generator_idx], 0);
                                unresolved_watches[watching_generator_idx] -= 1;
                                next_pending_generator_indices.push(watching_generator_idx);
                            }
                        }
                    }
                }
            }
        }

        pending_generator_indices = next_pending_generator_indices;
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
fn seed_inputs_and_unresolved_watches<F: Field>(
    witness: &mut PartitionWitness<F>,
    inputs: PartialWitness<F>,
    generator_watch_counts: &[usize],
    generator_indices_by_watches: &BTreeMap<usize, Vec<usize>>,
) -> Result<Vec<usize>> {
    let mut unresolved_watches = generator_watch_counts.to_vec();

    for (t, v) in inputs.target_values.into_iter() {
        if let Some(watch) = witness.set_target_returning_rep(t, v)? {
            if let Some(watchers) = generator_indices_by_watches.get(&watch) {
                for &generator_idx in watchers {
                    debug_assert_ne!(unresolved_watches[generator_idx], 0);
                    unresolved_watches[generator_idx] -= 1;
                }
            }
        }
    }

    Ok(unresolved_watches)
}

/// Direct-seeding adapter: writes values straight into the partition's
/// representative slots while maintaining the same per-generator
/// unresolved-watch counters as [`seed_inputs_and_unresolved_watches`],
/// without routing the values through a `PartialWitness` map first. The
/// decrement rule is identical: `set_target_returning_rep` returns the
/// representative only on first population, so aliased or duplicated
/// inputs decrement at most once and no counter can underflow.
pub struct PartitionSeeder<'a, 'b, F: Field> {
    witness: &'b mut PartitionWitness<'a, F>,
    unresolved_watches: &'b mut [usize],
    generator_indices_by_watches: &'b BTreeMap<usize, Vec<usize>>,
}

impl<F: Field> Debug for PartitionSeeder<'_, '_, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PartitionSeeder").finish_non_exhaustive()
    }
}

impl<F: Field> WitnessWrite<F> for PartitionSeeder<'_, '_, F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        if let Some(watch) = self.witness.set_target_returning_rep(target, value)? {
            if let Some(watchers) = self.generator_indices_by_watches.get(&watch) {
                for &generator_idx in watchers {
                    debug_assert_ne!(self.unresolved_watches[generator_idx], 0);
                    self.unresolved_watches[generator_idx] -= 1;
                }
            }
        }
        Ok(())
    }
}

impl<F: Field> Witness<F> for PartitionSeeder<'_, '_, F> {
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
    unresolved_watches: Vec<usize>,
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

        let mut unresolved_watches = seed_inputs_and_unresolved_watches(
            &mut witness,
            inputs,
            &prover_data.generator_watch_counts,
            &prover_data.generator_indices_by_watches,
        )?;

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();

        // Initially, all generators are queued.
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut unresolved_watches,
            &mut generator_is_expired,
            &mut remaining_generators,
            (0..generators.len()).collect(),
            parallel_threshold,
        )?;

        Ok(Self {
            witness,
            unresolved_watches,
            generator_is_expired,
            remaining_generators,
            prover_data,
            parallel_threshold,
        })
    }

    /// Like [`Self::start`], but the initial inputs are written by `seed`
    /// directly into the partition through a [`PartitionSeeder`] — no
    /// intermediate `PartialWitness` map is built or replayed. Worklist
    /// initialization is unchanged: all generators are queued, gated by the
    /// same unresolved-watch counters the map-seeded path would produce.
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

        let mut unresolved_watches = prover_data.generator_watch_counts.to_vec();
        seed(&mut PartitionSeeder {
            witness: &mut witness,
            unresolved_watches: &mut unresolved_watches,
            generator_indices_by_watches: &prover_data.generator_indices_by_watches,
        })?;

        let mut generator_is_expired = vec![false; generators.len()];
        let mut remaining_generators = generators.len();

        // Initially, all generators are queued.
        run_generator_worklist(
            &mut witness,
            prover_data,
            &mut unresolved_watches,
            &mut generator_is_expired,
            &mut remaining_generators,
            (0..generators.len()).collect(),
            PARALLEL_WORKLIST_THRESHOLD,
        )?;

        Ok(Self {
            witness,
            unresolved_watches,
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
        let generator_indices_by_watches = &self.prover_data.generator_indices_by_watches;

        let mut pending_generator_indices = Vec::new();
        for (t, v) in inputs.target_values.into_iter() {
            if let Some(watch) = self.witness.set_target_returning_rep(t, v)? {
                if let Some(watchers) = generator_indices_by_watches.get(&watch) {
                    for &watching_generator_idx in watchers {
                        if !self.generator_is_expired[watching_generator_idx] {
                            debug_assert_ne!(self.unresolved_watches[watching_generator_idx], 0);
                            self.unresolved_watches[watching_generator_idx] -= 1;
                            pending_generator_indices.push(watching_generator_idx);
                        }
                    }
                }
            }
        }

        run_generator_worklist(
            &mut self.witness,
            self.prover_data,
            &mut self.unresolved_watches,
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
        let random_value = F::rand();
        out_buffer.set_target(self.target, random_value)
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

    /// Builds an outer circuit verifying two independent inner proofs, mirroring a chain step's
    /// tx-proof/cyclic-proof pair. Returns the outer circuit and the two input halves.
    #[allow(clippy::type_complexity)]
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

    /// The initialization that `seed_inputs_and_unresolved_watches` replaced: seed every input,
    /// then walk the entire representative-keyed watcher map counting, per generator, the
    /// still-unpopulated representatives it watches. Kept as an in-test oracle.
    fn legacy_seed_inputs_and_unresolved_watches<C: GenericConfig<D, F = F>, const D: usize>(
        witness: &mut PartitionWitness<F>,
        inputs: PartialWitness<F>,
        prover_data: &ProverOnlyCircuitData<F, C, D>,
    ) -> Result<Vec<usize>>
    where
        F: RichField + Extendable<D>,
    {
        for (t, v) in inputs.target_values.into_iter() {
            witness.set_target(t, v)?;
        }

        let mut unresolved_watches = vec![0usize; prover_data.generators.len()];
        for (&watch, watchers) in &prover_data.generator_indices_by_watches {
            if !witness.is_set_by_rep_index(watch) {
                for &generator_idx in watchers {
                    unresolved_watches[generator_idx] += 1;
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
        for watchers in prover_data.generator_indices_by_watches.values() {
            for &generator_idx in watchers {
                occurrences[generator_idx] += 1;
            }
        }
        assert_eq!(
            prover_data.generator_watch_counts, occurrences,
            "builder-derived watch counts disagree with the watcher index"
        );

        // Watcher lists are deduplicated, so a count is the number of *distinct* representatives.
        for watchers in prover_data.generator_indices_by_watches.values() {
            let mut sorted = watchers.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(&sorted, watchers, "watcher list is not deduplicated/sorted");
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
            let new_counts = seed_inputs_and_unresolved_watches(
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
                new_counts, legacy_counts,
                "unresolved-watch counts diverge from the legacy map scan"
            );
            assert_eq!(new_witness.values, legacy_witness.values);
            assert_eq!(new_witness.set_bitmap, legacy_witness.set_bitmap);
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
        for ((single, repeat), split) in single_shot
            .values
            .iter()
            .zip(&single_shot_repeat.values)
            .zip(&two_phase.values)
        {
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
        let witness =
            generate_partial_witness(PartialWitness::new(), &circuit.prover_only, &circuit.common)
                .unwrap();

        assert_eq!(witness.get_target(early_output), F::from_canonical_u64(7));
        assert_eq!(witness.get_target(final_output), F::from_canonical_u64(11));
        assert_eq!(run_calls.load(Ordering::Relaxed), 2);
    }
}
