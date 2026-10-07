#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::mem::MaybeUninit;

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::plonk::vars::{EvaluationVarsBaseBatch, EvaluationVarsBasePacked};

pub trait PackedEvaluableBase<F: RichField + Extendable<D>, const D: usize>: Gate<F, D> {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars_base: EvaluationVarsBasePacked<P>,
        yield_constr: StridedConstraintConsumer<P>,
    );

    /// Evaluates entire batch of points. Returns a matrix of constraints. Constraint `j` for point
    /// `i` is at `index j * batch_size + i`.
    fn eval_unfiltered_base_batch_packed(&self, vars_batch: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let mut res = vec![F::ZERO; vars_batch.len() * self.num_constraints()];
        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = vars_batch.len() - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new(
                    &mut res[..],
                    vars_batch.len(),
                    <F as Packable>::Packing::WIDTH * i,
                ),
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new(&mut res[..], vars_batch.len(), leftovers_start + i),
            );
        }
        res
    }

    /// Filter-fused variant of [`Self::eval_unfiltered_base_batch_packed`]: evaluates one packed
    /// lane group at a time into an L1-resident `num_constraints x P::WIDTH` scratch block and
    /// immediately multiply-accumulates each row into the shared quotient buffer, instead of
    /// materializing the full `num_constraints x batch_size` matrix and re-walking it in a second
    /// pass. Per point `p` and constraint `j` this performs the same
    /// `combined[j * n + p] += constraint * filters[p]` product-accumulate the materialize-then-add
    /// default performs, with the same packed evaluation producing each constraint value, so the
    /// result is value-identical; only the memory traffic changes.
    fn eval_unfiltered_base_batch_accumulate_packed(
        &self,
        vars_batch: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_batch.len();
        assert_eq!(filters.len(), n);
        let num_constraints = self.num_constraints();
        assert!(combined_gate_constraints.len() >= num_constraints * n);
        let width = <<F as Packable>::Packing as PackedField>::WIDTH;

        // Scratch for one lane group's constraint block. Batches are 32 points, so for
        // typical gates this is at most a few KiB; keep it on the stack when it fits.
        //
        // Only `[..scratch_len]` is ever addressed, so the reservation is sized for
        // the worst case but declared uninitialized and zeroed over exactly that
        // prefix. `[F::ZERO; 1024]` instead zeroed all 8 KiB on every call, of which
        // `1024 - scratch_len` elements are then never read, written or aliased by
        // anything — the slice handed to `StridedConstraintConsumer` stops at
        // `scratch_len`. This is the hottest CPU loop in the prover: the accumulate
        // impl runs once per CPU-resident gate per 32-point quotient batch, i.e.
        // `num_gates * quotient_domain / 32` times per proof, so the deleted stores
        // are hundreds of megabytes of L1 store traffic per proof. The zeroing that
        // remains is bit-for-bit the zeroing the used prefix received before, so
        // every constraint value read below is unchanged, including for a gate that
        // emits fewer values than it advertises.
        let scratch_len = num_constraints * width;
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); 1024];
        let mut scratch_heap;
        let scratch: &mut [F] = if scratch_len <= scratch_stack.len() {
            let prefix = &mut scratch_stack[..scratch_len];
            prefix.fill(MaybeUninit::new(F::ZERO));
            // SAFETY: every element of `prefix` was just written with `F::ZERO`, and
            // `MaybeUninit<F>` has the same layout and alignment as `F`. Same idiom
            // as `RandomAccessGate`'s stack-or-heap scratch.
            unsafe { core::slice::from_raw_parts_mut(prefix.as_mut_ptr().cast::<F>(), scratch_len) }
        } else {
            scratch_heap = vec![F::ZERO; scratch_len];
            &mut scratch_heap
        };

        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = n - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            let offset = width * i;
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new(scratch, width, 0),
            );
            // Load packed lanes directly. The previous form zero-initialized each
            // packed value and then immediately overwrote every lane, so the
            // `ZEROS` store was dead. Alignment-safe by construction: the packed
            // type is `#[repr(transparent)]` over `[GoldilocksField; WIDTH]` and
            // documents that it "deliberately retains the scalar field's
            // eight-byte alignment so that a scalar slice can be reinterpreted as
            // packed values"; `from_slice` also asserts len == WIDTH.
            let filter = *<F as Packable>::Packing::from_slice(&filters[offset..offset + width]);
            for j in 0..num_constraints {
                let combined = &mut combined_gate_constraints[j * n + offset..][..width];
                let acc = *<F as Packable>::Packing::from_slice(combined);
                let row =
                    *<F as Packable>::Packing::from_slice(&scratch[j * width..(j + 1) * width]);
                combined.copy_from_slice(acc.multiply_accumulate(row, filter).as_slice());
            }
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            let point = leftovers_start + i;
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new(&mut scratch[..num_constraints], 1, 0),
            );
            for j in 0..num_constraints {
                combined_gate_constraints[j * n + point] += scratch[j] * filters[point];
            }
        }
    }
}
