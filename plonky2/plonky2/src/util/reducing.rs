#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::borrow::Borrow;

use plonky2_maybe_rayon::*;

use crate::field::extension::{Extendable, FieldExtension};
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
        // TODO: Would like to use `FE::scalar_mul`, but it doesn't work with Packed currently.
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

        let partials: Vec<Vec<F>> = polys
            .par_chunks(PARALLEL_CHUNK)
            .zip(base_powers.par_chunks(PARALLEL_CHUNK))
            .map(|(ps, powers)| accumulate_chunk(ps, powers))
            .collect();
        let mut acc = vec![F::ZERO; max_len];
        for partial in partials {
            for (a, p) in acc.iter_mut().zip(partial) {
                *a += p;
            }
        }
        PolynomialCoeffs::new(acc)
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
    use crate::field::types::Sample;
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

    /// The direct-construction first term in `reduce_polys_base` must be
    /// raw-`u64` identical to the grow-and-zero-then-accumulate form it
    /// replaced, including when the first polynomial is empty or shorter than a
    /// later one (so the accumulator still has to grow mid-fold).
    #[test]
    fn reduce_polys_base_matches_grow_and_zero() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        // Reference: the pre-change body, verbatim.
        #[allow(unused_assignments)]
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
                    assert_eq!(a[d].0, e[d].0, "coeff {i} limb {d} for {lens:?}");
                }
            }
        }
    }
}
