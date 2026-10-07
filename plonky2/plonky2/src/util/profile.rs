//! Diagnostic whole-proof event recorder.
//!
//! This module is compiled only with the `diagnostic_profile` feature. The
//! production benchmark does not enable that feature, so none of this code or
//! its call sites exist in the scored binary.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::Instant;

static EPOCH: OnceLock<Instant> = OnceLock::new();
static EVENTS: LazyLock<Mutex<Vec<Event>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static PROOF_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

thread_local! {
    static CONTEXTS: RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Context {
    name: String,
    instance: u64,
    args: BTreeMap<String, u64>,
    parent_context: Option<String>,
    parent_instance: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum EventKind {
    Complete { start_ns: u64, duration_ns: u64 },
    Counter { timestamp_ns: u64, value: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Event {
    category: String,
    name: String,
    context: String,
    instance: u64,
    parent_context: Option<String>,
    parent_instance: Option<u64>,
    args: BTreeMap<String, u64>,
    thread_id: u64,
    thread_name: String,
    kind: EventKind,
}

fn timestamp_ns() -> u64 {
    EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

fn thread_identity() -> (u64, String) {
    let thread = std::thread::current();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    thread.id().hash(&mut hasher);
    (
        hasher.finish(),
        thread.name().unwrap_or("unnamed").to_owned(),
    )
}

fn current_context() -> Context {
    CONTEXTS.with(|contexts| {
        contexts
            .borrow()
            .last()
            .cloned()
            .unwrap_or_else(|| Context {
                name: "process".to_owned(),
                instance: 0,
                args: BTreeMap::new(),
                parent_context: None,
                parent_instance: None,
            })
    })
}

fn record(event: Event) {
    EVENTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(event);
}

/// Keeps a stable proof/orchestration identity on the current thread.
/// Nested contexts restore their parent when dropped.
#[derive(Debug)]
pub struct ContextGuard {
    active: bool,
}

pub fn enter_context(name: &str, instance: u64, args: &[(&str, u64)]) -> ContextGuard {
    CONTEXTS.with(|contexts| {
        let mut contexts = contexts.borrow_mut();
        let parent = contexts.last().cloned();
        let mut args = args
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect::<BTreeMap<_, _>>();
        if let Some(parent) = &parent {
            for (parent_name, value) in &parent.args {
                args.entry(format!("parent_{parent_name}"))
                    .or_insert(*value);
            }
        }
        contexts.push(Context {
            name: name.to_owned(),
            instance,
            args,
            parent_context: parent.as_ref().map(|parent| parent.name.clone()),
            parent_instance: parent.as_ref().map(|parent| parent.instance),
        });
    });
    ContextGuard { active: true }
}

/// Enters a uniquely numbered proof context whose stable name and numeric
/// metadata identify the circuit shape without relying on caller labels.
#[allow(clippy::too_many_arguments)]
pub fn enter_proof(
    digest: u64,
    degree_bits: usize,
    num_wires: usize,
    num_routed_wires: usize,
    quotient_degree_factor: usize,
    num_challenges: usize,
    num_gate_constraints: usize,
    num_lookup_polys: usize,
) -> ContextGuard {
    let instance = PROOF_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = format!("proof-d{degree_bits}-w{num_wires}-q{quotient_degree_factor}-{digest:016x}");
    enter_context(
        &name,
        instance,
        &[
            ("digest", digest),
            ("degree_bits", degree_bits as u64),
            ("num_wires", num_wires as u64),
            ("num_routed_wires", num_routed_wires as u64),
            ("quotient_degree_factor", quotient_degree_factor as u64),
            ("num_challenges", num_challenges as u64),
            ("num_gate_constraints", num_gate_constraints as u64),
            ("num_lookup_polys", num_lookup_polys as u64),
        ],
    )
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        if self.active {
            CONTEXTS.with(|contexts| {
                let popped = contexts.borrow_mut().pop();
                debug_assert!(popped.is_some());
            });
        }
    }
}

/// RAII complete event. Dropping it records elapsed wall time and inherited
/// proof context.
#[derive(Debug)]
pub struct Span {
    category: String,
    name: String,
    context: Context,
    thread_id: u64,
    thread_name: String,
    start_ns: u64,
}

pub fn span(category: &str, name: &str) -> Span {
    let (thread_id, thread_name) = thread_identity();
    Span {
        category: category.to_owned(),
        name: name.to_owned(),
        context: current_context(),
        thread_id,
        thread_name,
        start_ns: timestamp_ns(),
    }
}

impl Span {
    /// Records a counter from an asynchronous completion while retaining the
    /// span's originating proof and submit-thread identity.
    pub fn counter(&self, category: &str, name: &str, value: u64) {
        record(Event {
            category: category.to_owned(),
            name: name.to_owned(),
            context: self.context.name.clone(),
            instance: self.context.instance,
            parent_context: self.context.parent_context.clone(),
            parent_instance: self.context.parent_instance,
            args: self.context.args.clone(),
            thread_id: self.thread_id,
            thread_name: self.thread_name.clone(),
            kind: EventKind::Counter {
                timestamp_ns: timestamp_ns(),
                value,
            },
        });
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let end_ns = timestamp_ns();
        record(Event {
            category: self.category.clone(),
            name: self.name.clone(),
            context: self.context.name.clone(),
            instance: self.context.instance,
            parent_context: self.context.parent_context.clone(),
            parent_instance: self.context.parent_instance,
            args: self.context.args.clone(),
            thread_id: self.thread_id,
            thread_name: self.thread_name.clone(),
            kind: EventKind::Complete {
                start_ns: self.start_ns,
                duration_ns: end_ns.saturating_sub(self.start_ns),
            },
        });
    }
}

/// Records an instantaneous numeric counter in the current context.
pub fn counter(category: &str, name: &str, value: u64) {
    let context = current_context();
    let (thread_id, thread_name) = thread_identity();
    record(Event {
        category: category.to_owned(),
        name: name.to_owned(),
        context: context.name,
        instance: context.instance,
        parent_context: context.parent_context,
        parent_instance: context.parent_instance,
        args: context.args,
        thread_id,
        thread_name,
        kind: EventKind::Counter {
            timestamp_ns: timestamp_ns(),
            value,
        },
    });
}

/// Writes all events currently recorded by this process in Chrome trace-event
/// JSON format. Complete events use `ph = "X"`; numeric work counters use
/// `ph = "C"`. Timestamps and durations are emitted in microseconds as the
/// format requires.
pub fn write_chrome_trace_to(writer: &mut impl std::io::Write) -> std::io::Result<()> {
    use serde_json::{json, Map, Value};

    let events = EVENTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    let encoded = events
        .iter()
        .map(|event| {
            let mut args = Map::new();
            args.insert("context".to_owned(), Value::String(event.context.clone()));
            args.insert("instance".to_owned(), Value::from(event.instance));
            if let Some(parent_context) = &event.parent_context {
                args.insert(
                    "parent_context".to_owned(),
                    Value::String(parent_context.clone()),
                );
            }
            if let Some(parent_instance) = event.parent_instance {
                args.insert("parent_instance".to_owned(), Value::from(parent_instance));
            }
            args.insert(
                "thread_name".to_owned(),
                Value::String(event.thread_name.clone()),
            );
            for (name, value) in &event.args {
                args.insert(name.clone(), Value::from(*value));
            }
            match event.kind {
                EventKind::Complete {
                    start_ns,
                    duration_ns,
                } => json!({
                    "name": event.name,
                    "cat": event.category,
                    "ph": "X",
                    "pid": pid,
                    "tid": event.thread_id,
                    "ts": start_ns as f64 / 1_000.0,
                    "dur": duration_ns as f64 / 1_000.0,
                    "args": args,
                }),
                EventKind::Counter {
                    timestamp_ns,
                    value,
                } => {
                    args.insert("value".to_owned(), Value::from(value));
                    json!({
                        "name": event.name,
                        "cat": event.category,
                        "ph": "C",
                        "pid": pid,
                        "tid": event.thread_id,
                        "ts": timestamp_ns as f64 / 1_000.0,
                        "args": args,
                    })
                }
            }
        })
        .collect::<Vec<_>>();
    serde_json::to_writer(writer, &encoded).map_err(std::io::Error::other)
}

/// Creates or truncates `path` and writes the current Chrome trace.
pub fn write_chrome_trace(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_chrome_trace_to(&mut file)
}

#[cfg(test)]
fn reset_for_tests() {
    PROOF_SEQUENCE.store(1, std::sync::atomic::Ordering::Relaxed);
    EVENTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    CONTEXTS.with(|contexts| contexts.borrow_mut().clear());
}

#[cfg(test)]
fn take_events_for_tests() -> Vec<Event> {
    std::mem::take(
        &mut *EVENTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn serial_test() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn nested_proof_context_keeps_logical_parent_identity() {
        let _serial = serial_test();
        reset_for_tests();
        {
            let _logical = enter_context("light_tx", 12, &[("chunk_index", 44)]);
            let _proof = enter_proof(0xabc, 16, 136, 80, 8, 2, 7, 0);
            counter("work", "proof_seen", 1);
        }
        let events = take_events_for_tests();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].parent_context.as_deref(), Some("light_tx"));
        assert_eq!(events[0].parent_instance, Some(12));
        assert_eq!(events[0].args.get("parent_chunk_index"), Some(&44));
    }

    #[test]
    fn prover_pipeline_inherits_automatic_proof_context() {
        let _serial = serial_test();
        use crate::field::types::{Field, PrimeField64};
        use crate::iop::witness::{PartialWitness, WitnessWrite};
        use crate::plonk::circuit_builder::CircuitBuilder;
        use crate::plonk::circuit_data::CircuitConfig;
        use crate::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};
        use crate::plonk::prover::prove;
        use crate::util::timing::TimingTree;

        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let target = builder.add_virtual_target();
        builder.register_public_input(target);
        let data = builder.build::<C>();
        let mut witness = PartialWitness::new();
        witness.set_target(target, F::ONE).expect("set witness");

        reset_for_tests();
        let mut timing = TimingTree::new("integration-root", log::Level::Debug);
        let proof = prove::<F, C, D>(&data.prover_only, &data.common, witness, &mut timing)
            .expect("small proof");
        drop(timing);
        data.verify(proof).expect("small proof verifies");

        let events = take_events_for_tests();
        let pipeline = events
            .iter()
            .find(|event| event.name == "compute full witness")
            .expect("prover TimingTree stage");
        assert!(pipeline.context.starts_with("proof-d"));
        assert_eq!(
            pipeline.args.get("digest"),
            Some(&data.prover_only.circuit_digest.elements[0].to_canonical_u64()),
        );
        assert_eq!(
            pipeline.args.get("degree_bits"),
            Some(&(data.common.degree_bits() as u64)),
        );
    }

    #[test]
    fn proof_contexts_get_unique_ids_and_shape_metadata() {
        let _serial = serial_test();
        reset_for_tests();
        {
            let _proof = enter_proof(0xabc, 16, 136, 80, 8, 2, 7, 0);
            counter("work", "proof_seen", 1);
        }
        {
            let _proof = enter_proof(0xabc, 16, 136, 80, 8, 2, 7, 0);
            counter("work", "proof_seen", 1);
        }

        let events = take_events_for_tests();
        assert_eq!(events.len(), 2);
        assert_ne!(events[0].instance, events[1].instance);
        assert_eq!(events[0].context, events[1].context);
        assert_eq!(events[0].args.get("digest"), Some(&0xabc));
        assert_eq!(events[0].args.get("degree_bits"), Some(&16));
        assert_eq!(events[0].args.get("num_wires"), Some(&136));
        assert_eq!(events[0].args.get("num_routed_wires"), Some(&80));
        assert_eq!(events[0].args.get("quotient_degree_factor"), Some(&8));
        assert_eq!(events[0].args.get("num_challenges"), Some(&2));
        assert_eq!(events[0].args.get("num_gate_constraints"), Some(&7));
        assert_eq!(events[0].args.get("num_lookup_polys"), Some(&0));
    }

    #[test]
    fn timing_tree_scopes_automatically_become_trace_spans() {
        let _serial = serial_test();
        reset_for_tests();
        {
            let _context = enter_context("chain", 4, &[("degree_bits", 14)]);
            let mut timing = crate::util::timing::TimingTree::new("proof-root", log::Level::Debug);
            timing.push("compute quotient", log::Level::Debug);
            timing.pop();
        }

        let events = take_events_for_tests();
        let quotient = events
            .iter()
            .find(|event| event.name == "compute quotient")
            .expect("TimingTree child span");
        assert_eq!(quotient.category, "plonky2");
        assert_eq!(quotient.context, "chain");
        assert_eq!(quotient.instance, 4);
        assert!(events.iter().any(|event| event.name == "proof-root"));
    }

    #[test]
    fn chrome_trace_serializes_complete_and_counter_events() {
        let _serial = serial_test();
        reset_for_tests();
        {
            let _parent = enter_context("light_path", 3, &[("chunk_index", 44)]);
            let _context = enter_context("light\"tx", 9, &[("degree_bits", 16)]);
            counter("work", "bytes_read", 4096);
            let _span = span("proof", "prove");
        }

        let mut encoded = Vec::new();
        write_chrome_trace_to(&mut encoded).expect("trace encoding");
        let trace: serde_json::Value = serde_json::from_slice(&encoded).expect("valid JSON trace");
        let events = trace.as_array().expect("Chrome trace is an array");
        assert_eq!(events.len(), 2);

        let complete = events
            .iter()
            .find(|event| event["ph"] == "X")
            .expect("complete event");
        assert_eq!(complete["cat"], "proof");
        assert_eq!(complete["name"], "prove");
        assert_eq!(complete["args"]["context"], "light\"tx");
        assert_eq!(complete["args"]["instance"], 9);
        assert_eq!(complete["args"]["degree_bits"], 16);
        assert_eq!(complete["args"]["parent_context"], "light_path");
        assert_eq!(complete["args"]["parent_instance"], 3);
        assert_eq!(complete["args"]["parent_chunk_index"], 44);
        assert!(complete["dur"].as_f64().expect("duration") >= 0.0);

        let counter = events
            .iter()
            .find(|event| event["ph"] == "C")
            .expect("counter event");
        assert_eq!(counter["cat"], "work");
        assert_eq!(counter["name"], "bytes_read");
        assert_eq!(counter["args"]["value"], 4096);
    }

    #[test]
    fn async_span_counter_uses_the_captured_context() {
        let _serial = serial_test();
        reset_for_tests();
        let span = {
            let _context = enter_context("light_chain", 7, &[("chunk_index", 12)]);
            span("metal", "tree_command")
        };
        {
            let _other = enter_context("unrelated", 99, &[]);
            span.counter("metal", "gpu_execution_ns", 1234);
        }
        drop(span);
        let events = take_events_for_tests();
        let counter = events
            .iter()
            .find(|event| matches!(event.kind, EventKind::Counter { .. }))
            .expect("captured counter");
        assert_eq!(counter.context, "light_chain");
        assert_eq!(counter.instance, 7);
        assert_eq!(counter.args.get("chunk_index"), Some(&12));
        assert!(matches!(
            counter.kind,
            EventKind::Counter { value: 1234, .. }
        ));
    }

    #[test]
    fn nested_spans_keep_context_and_emit_counter() {
        let _serial = serial_test();
        reset_for_tests();
        {
            let _context = enter_context("light_tx", 7, &[("degree_bits", 16), ("wires", 136)]);
            let _outer = span("proof", "prove");
            counter("work", "field_mul", 1234);
            {
                let _inner = span("commitment", "wires");
            }
        }

        let events = take_events_for_tests();
        let complete = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::Complete { .. }))
            .collect::<Vec<_>>();
        assert_eq!(complete.len(), 2);
        assert!(complete.iter().all(|event| event.context == "light_tx"));
        assert!(complete.iter().all(|event| event.instance == 7));
        assert!(complete.iter().all(|event| {
            event.args.get("degree_bits") == Some(&16) && event.args.get("wires") == Some(&136)
        }));
        assert!(complete.iter().any(|event| event.name == "prove"));
        assert!(complete.iter().any(|event| event.name == "wires"));

        let field_mul = events
            .iter()
            .find(|event| event.name == "field_mul")
            .expect("counter event");
        assert_eq!(field_mul.category, "work");
        assert_eq!(field_mul.context, "light_tx");
        assert_eq!(field_mul.instance, 7);
        assert!(matches!(
            field_mul.kind,
            EventKind::Counter { value: 1234, .. }
        ));
    }
}
