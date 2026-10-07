use log::{log, Level};
#[cfg(feature = "timing")]
use web_time::{Duration, Instant};

/// The hierarchy of scopes, and the time consumed by each one. Useful for profiling.
#[cfg(feature = "timing")]
#[derive(Debug)]
pub struct TimingTree {
    /// The name of this scope.
    name: String,
    /// The level at which to log this scope and its children.
    level: log::Level,
    /// The time when this scope was created.
    enter_time: Instant,
    /// The time when this scope was destroyed, or None if it has not yet been destroyed.
    exit_time: Option<Instant>,
    /// Any child scopes.
    children: Vec<TimingTree>,
    #[cfg(feature = "diagnostic_profile")]
    profile_span: Option<crate::util::profile::Span>,
}

#[cfg(not(feature = "timing"))]
#[derive(Debug)]
pub struct TimingTree {
    level: Level,
    #[cfg(feature = "diagnostic_profile")]
    profile_spans: Vec<crate::util::profile::Span>,
}

#[cfg(feature = "timing")]
impl Default for TimingTree {
    fn default() -> Self {
        TimingTree::new("root", Level::Debug)
    }
}

#[cfg(not(feature = "timing"))]
impl Default for TimingTree {
    fn default() -> Self {
        TimingTree::new("", Level::Debug)
    }
}

impl TimingTree {
    #[cfg(feature = "timing")]
    pub fn new(root_name: &str, level: Level) -> Self {
        Self {
            name: root_name.to_string(),
            level,
            enter_time: Instant::now(),
            exit_time: None,
            children: vec![],
            #[cfg(feature = "diagnostic_profile")]
            profile_span: Some(crate::util::profile::span("plonky2", root_name)),
        }
    }

    #[cfg(not(feature = "timing"))]
    pub fn new(root_name: &str, level: Level) -> Self {
        Self {
            level,
            #[cfg(feature = "diagnostic_profile")]
            profile_spans: vec![crate::util::profile::span("plonky2", root_name)],
        }
    }

    /// Whether this scope is still in scope.
    #[cfg(feature = "timing")]
    const fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// A description of the stack of currently-open scopes.
    #[cfg(feature = "timing")]
    pub fn open_stack(&self) -> String {
        let mut stack = Vec::new();
        self.open_stack_helper(&mut stack);
        stack.join(" > ")
    }

    #[cfg(feature = "timing")]
    fn open_stack_helper(&self, stack: &mut Vec<String>) {
        if self.is_open() {
            stack.push(self.name.clone());
            if let Some(last_child) = self.children.last() {
                last_child.open_stack_helper(stack);
            }
        }
    }

    #[cfg(feature = "timing")]
    pub fn push(&mut self, ctx: &str, mut level: log::Level) {
        assert!(self.is_open());

        // We don't want a scope's log level to be stronger than that of its parent.
        level = level.max(self.level);

        if let Some(last_child) = self.children.last_mut() {
            if last_child.is_open() {
                last_child.push(ctx, level);
                return;
            }
        }

        self.children.push(TimingTree {
            name: ctx.to_string(),
            level,
            enter_time: Instant::now(),
            exit_time: None,
            children: vec![],
            #[cfg(feature = "diagnostic_profile")]
            profile_span: Some(crate::util::profile::span("plonky2", ctx)),
        })
    }

    #[cfg(not(feature = "timing"))]
    pub fn push(&mut self, ctx: &str, _level: log::Level) {
        #[cfg(feature = "diagnostic_profile")]
        self.profile_spans
            .push(crate::util::profile::span("plonky2", ctx));
    }

    /// Close the deepest open scope from this tree.
    #[cfg(feature = "timing")]
    pub fn pop(&mut self) {
        assert!(self.is_open());

        if let Some(last_child) = self.children.last_mut() {
            if last_child.is_open() {
                last_child.pop();
                return;
            }
        }

        self.exit_time = Some(Instant::now());
        #[cfg(feature = "diagnostic_profile")]
        drop(self.profile_span.take());
    }

    #[cfg(not(feature = "timing"))]
    pub fn pop(&mut self) {
        #[cfg(feature = "diagnostic_profile")]
        {
            // Keep the root span alive until the tree itself is dropped.
            if self.profile_spans.len() > 1 {
                self.profile_spans.pop();
            }
        }
    }

    #[cfg(feature = "timing")]
    fn duration(&self) -> Duration {
        self.exit_time
            .unwrap_or_else(Instant::now)
            .duration_since(self.enter_time)
    }

    /// Filter out children with a low duration.
    #[cfg(feature = "timing")]
    pub fn filter(&self, min_delta: Duration) -> Self {
        Self {
            name: self.name.clone(),
            level: self.level,
            enter_time: self.enter_time,
            exit_time: self.exit_time,
            children: self
                .children
                .iter()
                .filter(|c| c.duration() >= min_delta)
                .map(|c| c.filter(min_delta))
                .collect(),
            #[cfg(feature = "diagnostic_profile")]
            profile_span: None,
        }
    }

    #[cfg(feature = "timing")]
    pub fn print(&self) {
        self.print_helper(0);
    }

    #[cfg(not(feature = "timing"))]
    pub fn print(&self) {
        log!(
            self.level,
            "TimingTree is not supported without the 'timing' feature enabled"
        );
    }

    #[cfg(feature = "timing")]
    fn print_helper(&self, depth: usize) {
        let prefix = "| ".repeat(depth);
        log!(
            self.level,
            "{}{:.4}s to {}",
            prefix,
            self.duration().as_secs_f64(),
            self.name
        );
        for child in &self.children {
            child.print_helper(depth + 1);
        }
    }
}

/// Creates a named scope; useful for debugging.
#[macro_export]
macro_rules! timed {
    ($timing_tree:expr, $level:expr, $ctx:expr, $exp:expr) => {{
        $timing_tree.push($ctx, $level);
        let res = $exp;
        $timing_tree.pop();
        res
    }};
    // If no context is specified, default to Debug.
    ($timing_tree:expr, $ctx:expr, $exp:expr) => {{
        $timing_tree.push($ctx, log::Level::Debug);
        let res = $exp;
        $timing_tree.pop();
        res
    }};
}
