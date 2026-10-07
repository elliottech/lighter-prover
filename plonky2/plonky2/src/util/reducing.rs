#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::any::TypeId;
use core::borrow::Borrow;

use plonky2_maybe_rayon::*;

use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::goldilocks_extensions::ext2_base_scalar_dot_slots;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::packed::PackedField;
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::reducing::ReducingGate;
use crate::gates::reducing_extension::ReducingExtensionGate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;

/// When verifying the composition polynomial in FRI we have to compute sums of the form
/// `(sum_0^k a^i * x_i)/d_0 + (sum_k^r a^i * y_i)/d_1`
/// The most efficient way to do this is to compute both quotient separately using Horner's method,
/// scale the second one by `a^(r-1-k)`, and add them up.
/// This struct abstract away these operations by implementing Horner's method and keeping track
/// of the number of multiplications by `a` to compute the scaling factor.
/// See <https://github.com/0xPolygonZero/plonky2/pull/69> for more details and discussions.
#[derive(Debug, Clone)]
pub struct ReducingFactor<F: Field> {
    base: F,
    count: u64,
}

impl<F: Field> ReducingFactor<F> {
    pub const fn new(base: F) -> Self {
        Self { base, count: 0 }
    }

    fn mul(&mut self, x: F) -> F {
        self.count += 1;
        self.base * x
    }

    fn mul_ext<FE, P, const D: usize>(&mut self, x: P) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        self.count += 1;
        x * FE::from_basefield(self.base)
    }

    fn mul_poly(&mut self, p: &mut PolynomialCoeffs<F>) {
        self.count += 1;
        *p *= self.base;
    }

    pub fn reduce(&mut self, iter: impl DoubleEndedIterator<Item = impl Borrow<F>>) -> F {
        iter.rev()
            .fold(F::ZERO, |acc, x| self.mul(acc) + *x.borrow())
    }

    pub fn reduce_ext<FE, P, const D: usize>(
        &mut self,
        iter: impl DoubleEndedIterator<Item = impl Borrow<P>>,
    ) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        iter.rev()
            .fold(P::ZEROS, |acc, x| self.mul_ext(acc) + *x.borrow())
    }

    pub fn reduce_polys(
        &mut self,
        polys: impl DoubleEndedIterator<Item = impl Borrow<PolynomialCoeffs<F>>>,
    ) -> PolynomialCoeffs<F> {
        polys.rev().fold(PolynomialCoeffs::empty(), |mut acc, x| {
            self.mul_poly(&mut acc);
            acc += x.borrow();
            acc
        })
    }

    pub fn reduce_polys_base<BF: Extendable<D, Extension = F>, const D: usize>(
        &mut self,
        polys: impl IntoIterator<Item = impl Borrow<PolynomialCoeffs<BF>> + Sync>,
    ) -> PolynomialCoeffs<F>
    where
        F: FieldExtension<D, BaseField = BF>,
    {
        // Fused multiply-accumulate: each base coefficient is read exactly
        // once and multiplied by its polynomial's power of the reducing base.
        // For the large opening batches this runs on the serial per-proof
        // spine, so split the polynomials into chunks reduced in parallel and
        // then sum the per-chunk partial vectors. Coefficient `i` of the
        // result is `sum_j base^j * c_{j,i}` either way — field addition is
        // exact, commutative and associative, so regrouping the sum by chunk
        // produces the identical field element for every coefficient.
        let polys: Vec<_> = polys.into_iter().collect();
        let num_polys = polys.len();
        let max_len = polys
            .iter()
            .map(|p| p.borrow().coeffs.len())
            .max()
            .unwrap_or(0);
        let base_powers: Vec<F> = self.base.powers().take(num_polys).collect();
        self.count += num_polys as u64;

        // Production fast path: for the quadratic Goldilocks extension the
        // whole batch reduces with one delayed reduction per extension limb
        // per output slot (the `fri_fold_arity16` pattern generalized to the
        // batch width) instead of a `reduce128` pair plus a canonicalizing
        // extension add per term. Field-equal by construction; the raw
        // representative may differ from the reduce-per-term form, under the
        // same license as the parallel path below (every consumer treats
        // these coefficients value-wise and proof serialization canonicalizes
        // every limb).
        if let Some(acc) =
            goldilocks_ext2_reduce_polys_base::<BF, F, D>(&polys, &base_powers, max_len)
        {
            return PolynomialCoeffs::new(acc);
        }

        let accumulate_chunk = |ps: &[_], powers: &[F]| -> Vec<F> {
            // Build the accumulator straight from the chunk's first
            // polynomial's scaled coefficients (the base tree's
            // direct-construction trick: `ZERO + x == x` exactly, so skipping
            // the zero-prefill + read-back over the first polynomial's prefix
            // is value-identical), then zero-fill only the tail beyond it —
            // empty in the common all-equal-degree case.
            let mut ps_iter = powers.iter().zip(ps);
            let mut acc: Vec<F> = match ps_iter.next() {
                Some((base_power, poly)) => {
                    let coeffs: &PolynomialCoeffs<BF> = Borrow::borrow(poly);
                    let mut acc = Vec::with_capacity(max_len);
                    acc.extend(
                        coeffs
                            .coeffs
                            .iter()
                            .map(|&c| <F as FieldExtension<D>>::scalar_mul(base_power, c)),
                    );
                    acc.resize(max_len, F::ZERO);
                    acc
                }
                None => vec![F::ZERO; max_len],
            };
            for (base_power, poly) in ps_iter {
                let coeffs: &PolynomialCoeffs<BF> = Borrow::borrow(poly);
                for (a, &c) in acc.iter_mut().zip(coeffs.coeffs.iter()) {
                    *a += <F as FieldExtension<D>>::scalar_mul(base_power, c);
                }
            }
            acc
        };

        // Small batches (the `g * zeta` batch has two polynomials) are not
        // worth the parallel dispatch or the partial-vector merge.
        const PARALLEL_CHUNK: usize = 16;
        if num_polys <= PARALLEL_CHUNK {
            return PolynomialCoeffs::new(accumulate_chunk(&polys, &base_powers));
        }

        // Coefficient slots are independent. Partition the result so each
        // worker visits all polynomials for one cache-sized output range,
        // deleting the full-degree partial vector per polynomial chunk and
        // the serial merge of those vectors. Each slot receives powers in
        // ascending polynomial order; this is field-equal to the previous
        // regrouped sum, while proof serialization canonicalizes every limb.
        const SLOT_BLOCK: usize = 2048;
        let mut acc = vec![F::ZERO; max_len];
        acc.par_chunks_mut(SLOT_BLOCK)
            .enumerate()
            .for_each(|(block, out)| {
                let start = block * SLOT_BLOCK;
                for (base_power, poly) in base_powers.iter().zip(&polys) {
                    let coeffs: &PolynomialCoeffs<BF> = Borrow::borrow(poly);
                    if coeffs.coeffs.len() <= start {
                        continue;
                    }
                    let live = (coeffs.coeffs.len() - start).min(out.len());
                    for (a, &c) in out[..live]
                        .iter_mut()
                        .zip(&coeffs.coeffs[start..start + live])
                    {
                        *a += <F as FieldExtension<D>>::scalar_mul(base_power, c);
                    }
                }
            });
        PolynomialCoeffs::new(acc)
    }

    /// Reduce a small batch of base-field polynomials and immediately fold
    /// its linear quotient into `final_poly`.
    ///
    /// The regular path materializes the full extension-field composition
    /// polynomial, then reads it backwards while writing the quotient into
    /// `final_poly`. Opening batches after the first one are tiny (normally
    /// the two `g * zeta` polynomials), so preserving that full-degree
    /// intermediate only adds an allocation and a write/read pass. Compute a
    /// cache-sized range of composition coefficients at a time instead and
    /// feed it straight into the same descending Horner recurrence.
    pub fn accumulate_small_polys_base_linear_quotient<
        BF: Extendable<D, Extension = F>,
        const D: usize,
    >(
        &mut self,
        polys: impl IntoIterator<Item = impl Borrow<PolynomialCoeffs<BF>> + Sync>,
        final_poly: &mut PolynomialCoeffs<F>,
        z: F,
    ) where
        F: FieldExtension<D, BaseField = BF>,
    {
        let polys: Vec<_> = polys.into_iter().collect();
        debug_assert!(polys.len() <= 16);
        let max_len = polys
            .iter()
            .map(|p| p.borrow().coeffs.len())
            .max()
            .unwrap_or(0);
        let base_powers: Vec<F> = self.base.powers().take(polys.len()).collect();
        self.count += polys.len() as u64;
        let shift = self.shift_factor();

        let buf = &mut final_poly.coeffs;
        for coefficient in buf.iter_mut().skip(max_len) {
            *coefficient *= shift;
        }
        if buf.len() < max_len {
            buf.resize(max_len, F::ZERO);
        }
        if max_len == 0 {
            return;
        }

        // The padded quotient's highest coefficient is zero.
        buf[max_len - 1] *= shift;

        const SLOT_BLOCK: usize = 2048;
        // This fixed block size is independent of runtime queue state.
        let mut scratch = vec![F::ZERO; SLOT_BLOCK.min(max_len.saturating_sub(1))];
        let mut end = max_len;
        let mut acc = F::ZERO;
        while end > 1 {
            let start = 1.max(end.saturating_sub(SLOT_BLOCK));
            let out = &mut scratch[..end - start];
            if !goldilocks_ext2_reduce_polys_base_into(&polys, &base_powers, start, out) {
                out.fill(F::ZERO);
                for (base_power, poly) in base_powers.iter().zip(&polys) {
                    let coeffs: &PolynomialCoeffs<BF> = Borrow::borrow(poly);
                    if coeffs.coeffs.len() <= start {
                        continue;
                    }
                    let live = (coeffs.coeffs.len() - start).min(out.len());
                    for (reduced, &coefficient) in out[..live]
                        .iter_mut()
                        .zip(&coeffs.coeffs[start..start + live])
                    {
                        *reduced += <F as FieldExtension<D>>::scalar_mul(base_power, coefficient);
                    }
                }
            }
            for (offset, &coefficient) in out.iter().enumerate().rev() {
                acc = acc * z + coefficient;
                let quotient_index = start + offset - 1;
                buf[quotient_index] = buf[quotient_index] * shift + acc;
            }
            end = start;
        }
    }

    pub fn shift(&mut self, x: F) -> F {
        let tmp = self.base.exp_u64(self.count) * x;
        self.count = 0;
        tmp
    }

    pub fn shift_poly(&mut self, p: &mut PolynomialCoeffs<F>) {
        *p *= self.base.exp_u64(self.count);
        self.count = 0;
    }

    /// Returns the factor `shift_poly` would multiply by (`base^count`) and
    /// resets the count, letting callers fuse the multiply into another pass.
    pub fn shift_factor(&mut self) -> F {
        let tmp = self.base.exp_u64(self.count);
        self.count = 0;
        tmp
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }
}

/// Goldilocks-quadratic fast path for [`ReducingFactor::reduce_polys_base`]:
/// delegates every output slot to `ext2_base_scalar_dot_slots`, which delays
/// modular reduction across the whole polynomial batch. Returns `None` for
/// any other field configuration, leaving the generic path untouched.
fn goldilocks_ext2_reduce_polys_base<BF, F, const D: usize>(
    polys: &[impl Borrow<PolynomialCoeffs<BF>> + Sync],
    base_powers: &[F],
    max_len: usize,
) -> Option<Vec<F>>
where
    BF: Extendable<D, Extension = F>,
    F: FieldExtension<D, BaseField = BF>,
{
    // The Goldilocks/quadratic guard is hoisted above the allocation so that
    // only the fast path — the one that assigns every slot — can ever observe
    // the buffer. Any other field configuration returns here, before the
    // allocation exists.
    if TypeId::of::<BF>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<F>() != TypeId::of::<QuadraticExtension<GoldilocksField>>()
    {
        return None;
    }
    let mut acc: Vec<F> = Vec::with_capacity(max_len);
    // SAFETY: capacity is exactly `max_len` and `F: Field` is `Copy`, so the
    // vector has no drop glue. Every slot in `0..max_len` is *assigned*
    // before it is read: `goldilocks_ext2_reduce_polys_base_into` hands the
    // whole slice (serial arm) or a `par_chunks_mut` partition of it
    // (parallel arm) to `ext2_base_scalar_dot_slots`, which writes `*o` for
    // every `o` in `out` unconditionally. The `vec![F::ZERO; max_len]` this
    // replaces was therefore a dead store over the whole composition buffer.
    unsafe { acc.set_len(max_len) };
    if !goldilocks_ext2_reduce_polys_base_into(polys, base_powers, 0, &mut acc) {
        return None;
    }
    Some(acc)
}

/// Fill one contiguous coefficient range of the Goldilocks extension fast
/// path. Keeping the destination caller-owned lets the opening path reuse a
/// small cache buffer instead of materializing the whole composition vector.
fn goldilocks_ext2_reduce_polys_base_into<BF, F, const D: usize>(
    polys: &[impl Borrow<PolynomialCoeffs<BF>> + Sync],
    base_powers: &[F],
    start: usize,
    out: &mut [F],
) -> bool
where
    BF: Extendable<D, Extension = F>,
    F: FieldExtension<D, BaseField = BF>,
{
    if TypeId::of::<BF>() != TypeId::of::<GoldilocksField>()
        || TypeId::of::<F>() != TypeId::of::<QuadraticExtension<GoldilocksField>>()
    {
        return false;
    }
    // SAFETY (all casts below): the `TypeId` compares prove `BF` is exactly
    // `GoldilocksField` and `F` is exactly
    // `QuadraticExtension<GoldilocksField>`; only the generic spelling of the
    // types differs, so the pointer reinterpretations preserve layout,
    // length and alignment exactly.
    let slices: Vec<&[GoldilocksField]> = polys
        .iter()
        .map(|p| {
            let coeffs = p.borrow().coeffs.as_slice();
            unsafe {
                core::slice::from_raw_parts(coeffs.as_ptr().cast::<GoldilocksField>(), coeffs.len())
            }
        })
        .collect();
    let powers = unsafe {
        core::slice::from_raw_parts(
            base_powers
                .as_ptr()
                .cast::<QuadraticExtension<GoldilocksField>>(),
            base_powers.len(),
        )
    };

    // Same shape split as the generic path: small batches stay serial, large
    // batches partition the coefficient slots so each worker visits all
    // polynomials for one cache-sized output range.
    const PARALLEL_CHUNK: usize = 16;
    const SLOT_BLOCK: usize = 2048;
    if slices.len() <= PARALLEL_CHUNK {
        let out = unsafe {
            core::slice::from_raw_parts_mut(
                out.as_mut_ptr()
                    .cast::<QuadraticExtension<GoldilocksField>>(),
                out.len(),
            )
        };
        ext2_base_scalar_dot_slots(out, start, &slices, powers);
    } else {
        out.par_chunks_mut(SLOT_BLOCK)
            .enumerate()
            .for_each(|(block, out)| {
                let block_start = start + block * SLOT_BLOCK;
                let out = unsafe {
                    core::slice::from_raw_parts_mut(
                        out.as_mut_ptr()
                            .cast::<QuadraticExtension<GoldilocksField>>(),
                        out.len(),
                    )
                };
                ext2_base_scalar_dot_slots(out, block_start, &slices, powers);
            });
    }
    true
}

#[derive(Debug, Clone)]
pub struct ReducingFactorTarget<const D: usize> {
    base: ExtensionTarget<D>,
    count: u64,
}

impl<const D: usize> ReducingFactorTarget<D> {
    pub const fn new(base: ExtensionTarget<D>) -> Self {
        Self { base, count: 0 }
    }

    /// Reduces a vector of `Target`s using `ReducingGate`s.
    pub fn reduce_base<F>(
        &mut self,
        terms: &[Target],
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let l = terms.len();

        // For small reductions, use an arithmetic gate.
        if l <= ArithmeticExtensionGate::<D>::new_from_config(&builder.config).num_ops + 1 {
            let terms_ext = terms
                .iter()
                .map(|&t| builder.convert_to_ext(t))
                .collect::<Vec<_>>();
            return self.reduce_arithmetic(&terms_ext, builder);
        }

        let max_coeffs_len = ReducingGate::<D>::max_coeffs_len(
            builder.config.num_wires,
            builder.config.num_routed_wires,
        );
        self.count += l as u64;
        let zero = builder.zero();
        let zero_ext = builder.zero_extension();
        let mut acc = zero_ext;
        let mut reversed_terms = terms.to_vec();
        while !reversed_terms.len().is_multiple_of(max_coeffs_len) {
            reversed_terms.push(zero);
        }
        reversed_terms.reverse();
        for chunk in reversed_terms.chunks_exact(max_coeffs_len) {
            let gate = ReducingGate::new(max_coeffs_len);
            let row = builder.add_gate(gate.clone(), vec![]);

            builder.connect_extension(
                self.base,
                ExtensionTarget::from_range(row, ReducingGate::<D>::wires_alpha()),
            );
            builder.connect_extension(
                acc,
                ExtensionTarget::from_range(row, ReducingGate::<D>::wires_old_acc()),
            );
            for (&t, c) in chunk.iter().zip(gate.wires_coeffs()) {
                builder.connect(t, Target::wire(row, c));
            }

            acc = ExtensionTarget::from_range(row, ReducingGate::<D>::wires_output());
        }

        acc
    }

    /// Reduces a vector of `ExtensionTarget`s using `ReducingExtensionGate`s.
    pub fn reduce<F>(
        &mut self,
        terms: &[ExtensionTarget<D>], // Could probably work with a `DoubleEndedIterator` too.
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let l = terms.len();

        // For small reductions, use an arithmetic gate.
        if l <= ArithmeticExtensionGate::<D>::new_from_config(&builder.config).num_ops + 1 {
            return self.reduce_arithmetic(terms, builder);
        }

        let max_coeffs_len = ReducingExtensionGate::<D>::max_coeffs_len(
            builder.config.num_wires,
            builder.config.num_routed_wires,
        );
        self.count += l as u64;
        let zero_ext = builder.zero_extension();
        let mut acc = zero_ext;
        let mut reversed_terms = terms.to_vec();
        while !reversed_terms.len().is_multiple_of(max_coeffs_len) {
            reversed_terms.push(zero_ext);
        }
        reversed_terms.reverse();
        for chunk in reversed_terms.chunks_exact(max_coeffs_len) {
            let gate = ReducingExtensionGate::new(max_coeffs_len);
            let row = builder.add_gate(gate.clone(), vec![]);

            builder.connect_extension(
                self.base,
                ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_alpha()),
            );
            builder.connect_extension(
                acc,
                ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_old_acc()),
            );
            for (i, &t) in chunk.iter().enumerate() {
                builder.connect_extension(
                    t,
                    ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_coeff(i)),
                );
            }

            acc = ExtensionTarget::from_range(row, ReducingExtensionGate::<D>::wires_output());
        }

        acc
    }

    /// Reduces a vector of `ExtensionTarget`s using `ArithmeticGate`s.
    fn reduce_arithmetic<F>(
        &mut self,
        terms: &[ExtensionTarget<D>],
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        self.count += terms.len() as u64;
        terms
            .iter()
            .rev()
            .fold(builder.zero_extension(), |acc, &et| {
                builder.mul_add_extension(self.base, acc, et)
            })
    }

    pub fn shift<F>(
        &mut self,
        x: ExtensionTarget<D>,
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let zero_ext = builder.zero_extension();
        let exp = if x == zero_ext {
            // The result will get zeroed out, so don't actually compute the exponentiation.
            zero_ext
        } else {
            builder.exp_u64_extension(self.base, self.count)
        };

        self.count = 0;
        builder.mul_extension(exp, x)
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::types::{PrimeField64, Sample};
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    fn test_reduce_gadget_base(n: usize) -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_config();

        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let alpha = FF::rand();
        let vs = F::rand_vec(n);

        let manual_reduce = ReducingFactor::new(alpha).reduce(vs.iter().map(|&v| FF::from(v)));
        let manual_reduce = builder.constant_extension(manual_reduce);

        let mut alpha_t = ReducingFactorTarget::new(builder.constant_extension(alpha));
        let vs_t = builder.add_virtual_targets(vs.len());
        for (&v, &v_t) in vs.iter().zip(&vs_t) {
            pw.set_target(v_t, v)?;
        }
        let circuit_reduce = alpha_t.reduce_base(&vs_t, &mut builder);

        builder.connect_extension(manual_reduce, circuit_reduce);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    fn test_reduce_gadget(n: usize) -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_config();

        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let alpha = FF::rand();
        let vs = (0..n).map(FF::from_canonical_usize).collect::<Vec<_>>();

        let manual_reduce = ReducingFactor::new(alpha).reduce(vs.iter());
        let manual_reduce = builder.constant_extension(manual_reduce);

        let mut alpha_t = ReducingFactorTarget::new(builder.constant_extension(alpha));
        let vs_t = builder.add_virtual_extension_targets(vs.len());
        pw.set_extension_targets(&vs_t, &vs)?;
        let circuit_reduce = alpha_t.reduce(&vs_t, &mut builder);

        builder.connect_extension(manual_reduce, circuit_reduce);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    #[test]
    fn test_reduce_gadget_even() -> Result<()> {
        test_reduce_gadget(10)
    }

    #[test]
    fn test_reduce_gadget_odd() -> Result<()> {
        test_reduce_gadget(11)
    }

    #[test]
    fn test_reduce_gadget_base_100() -> Result<()> {
        test_reduce_gadget_base(100)
    }

    #[test]
    fn test_reduce_gadget_100() -> Result<()> {
        test_reduce_gadget(100)
    }

    /// `reduce_polys_base` must agree with the legacy grow-and-zero
    /// accumulate form on every length shape, including an empty or short
    /// first polynomial (so the accumulator still has to grow mid-fold).
    ///
    /// Agreement is canonical-value equality, not raw-`u64` equality: the
    /// Goldilocks-quadratic fast path delays reduction across the batch, so
    /// its sub-2^64 representatives can differ from the reduce-per-term
    /// form's while denoting the same field element — the license the
    /// parallel slot path already documents (consumers treat these
    /// coefficients value-wise; proof serialization canonicalizes every
    /// limb).
    #[test]
    fn reduce_polys_base_matches_grow_and_zero() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        // Reference: the pre-change body, verbatim.
        fn legacy(alpha: FF, lens: &[usize], polys: &[PolynomialCoeffs<F>]) -> Vec<FF> {
            let mut rf = ReducingFactor::new(alpha);
            let mut acc: Vec<FF> = Vec::new();
            for (base_power, poly) in rf.base.powers().zip(polys.iter()) {
                rf.count += 1;
                let coeffs = &poly.coeffs;
                if coeffs.len() > acc.len() {
                    acc.resize(coeffs.len(), FF::ZERO);
                }
                for (a, &c) in acc.iter_mut().zip(coeffs.iter()) {
                    *a += <FF as FieldExtension<D>>::scalar_mul(&base_power, c);
                }
            }
            let _ = lens;
            acc
        }

        // Length shapes: uniform, growing (first shortest), shrinking, an empty
        // first polynomial, all empty, and the empty batch.
        let shapes: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![0, 0, 0],
            vec![8],
            vec![4, 4, 4, 4],
            vec![1, 3, 7, 16],
            vec![16, 7, 3, 1],
            vec![0, 5, 2, 9],
            vec![0, 0, 6],
        ];

        for lens in &shapes {
            let alpha = FF::rand();
            let polys: Vec<PolynomialCoeffs<F>> = lens
                .iter()
                .map(|&n| PolynomialCoeffs::new(F::rand_vec(n)))
                .collect();

            let expected = legacy(alpha, lens, &polys);
            let actual = ReducingFactor::new(alpha).reduce_polys_base::<F, D>(polys.iter());

            assert_eq!(actual.coeffs.len(), expected.len(), "length for {lens:?}");
            for (i, (a, e)) in actual.coeffs.iter().zip(expected.iter()).enumerate() {
                let a: [F; D] = a.to_basefield_array();
                let e: [F; D] = e.to_basefield_array();
                for d in 0..D {
                    assert_eq!(
                        a[d].to_canonical_u64(),
                        e[d].to_canonical_u64(),
                        "coeff {i} limb {d} for {lens:?}"
                    );
                }
            }
        }
    }

    /// The delayed-reduction fast path must agree with the legacy
    /// reduce-per-term form on batches wide enough to take the parallel
    /// slot-partitioned branch, across block-boundary-straddling lengths and
    /// mixed degrees.
    #[test]
    fn reduce_polys_base_wide_batch_matches_legacy() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        fn legacy(alpha: FF, polys: &[PolynomialCoeffs<F>]) -> Vec<FF> {
            let mut acc: Vec<FF> = Vec::new();
            for (base_power, poly) in alpha.powers().zip(polys.iter()) {
                let coeffs = &poly.coeffs;
                if coeffs.len() > acc.len() {
                    acc.resize(coeffs.len(), FF::ZERO);
                }
                for (a, &c) in acc.iter_mut().zip(coeffs.iter()) {
                    *a += <FF as FieldExtension<D>>::scalar_mul(&base_power, c);
                }
            }
            acc
        }

        // 40 polynomials forces the parallel branch (> PARALLEL_CHUNK); the
        // 5000-coefficient length straddles SLOT_BLOCK boundaries, and the
        // short/empty entries exercise the partial-coverage loop per block.
        let mut lens = vec![5000usize; 34];
        lens.extend([0, 1, 2047, 2048, 2049, 4096]);
        let alpha = FF::rand();
        let polys: Vec<PolynomialCoeffs<F>> = lens
            .iter()
            .map(|&n| PolynomialCoeffs::new(F::rand_vec(n)))
            .collect();

        let expected = legacy(alpha, &polys);
        let actual = ReducingFactor::new(alpha).reduce_polys_base::<F, D>(polys.iter());

        assert_eq!(actual.coeffs.len(), expected.len());
        for (i, (a, e)) in actual.coeffs.iter().zip(expected.iter()).enumerate() {
            let a: [F; D] = a.to_basefield_array();
            let e: [F; D] = e.to_basefield_array();
            for d in 0..D {
                assert_eq!(
                    a[d].to_canonical_u64(),
                    e[d].to_canonical_u64(),
                    "coeff {i} limb {d}"
                );
            }
        }
    }
}
