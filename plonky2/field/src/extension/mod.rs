use alloc::vec::Vec;

use crate::types::Field;

pub mod algebra;
pub mod quadratic;
pub mod quartic;
pub mod quintic;

/// Optimal extension field trait.
/// A degree `d` field extension is optimal if there exists a base field element `W`,
/// such that the extension is `F[X]/(X^d-W)`.
#[allow(clippy::upper_case_acronyms)]
pub trait OEF<const D: usize>: FieldExtension<D> {
    // Element W of BaseField, such that `X^d - W` is irreducible over BaseField.
    const W: Self::BaseField;

    // Element of BaseField such that DTH_ROOT^D == 1. Implementers
    // should set this to W^((p - 1)/D), where W is as above and p is
    // the order of the BaseField.
    const DTH_ROOT: Self::BaseField;
}

impl<F: Field> OEF<1> for F {
    const W: Self::BaseField = F::ONE;
    const DTH_ROOT: Self::BaseField = F::ONE;
}

pub trait Frobenius<const D: usize>: OEF<D> {
    /// FrobeniusField automorphisms: x -> x^p, where p is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(p^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= D {
            // x |-> x^(p^D) is the identity, so x^(p^count) ==
            // x^(p^(count % D))
            return self.repeated_frobenius(count % D);
        }
        let arr = self.to_basefield_array();

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((p^D-1)/D)
        let mut z0 = Self::DTH_ROOT;
        for _ in 1..count {
            z0 *= Self::DTH_ROOT;
        }

        let mut res = [Self::BaseField::ZERO; D];
        for (i, z) in z0.powers().take(D).enumerate() {
            res[i] = arr[i] * z;
        }

        Self::from_basefield_array(res)
    }
}

pub trait Extendable<const D: usize>: Field + Sized {
    type Extension: Field + OEF<D, BaseField = Self> + Frobenius<D> + From<Self>;

    const W: Self;

    const DTH_ROOT: Self;

    /// Chosen so that when raised to the power `(p^D - 1) >> F::Extension::TWO_ADICITY)`
    /// we obtain F::EXT_POWER_OF_TWO_GENERATOR.
    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; D];

    /// Chosen so that when raised to the power `1<<(Self::TWO_ADICITY-Self::BaseField::TWO_ADICITY)`,
    /// we get `Self::BaseField::POWER_OF_TWO_GENERATOR`. This makes `primitive_root_of_unity` coherent
    /// with the base field which implies that the FFT commutes with field inclusion.
    const EXT_POWER_OF_TWO_GENERATOR: [Self; D];

    /// Compute the dot product of extension-field values and base-field
    /// scalars. The slices are zipped, matching the usual iterator behavior
    /// when their lengths differ.
    ///
    /// The default preserves the reduce-per-product implementation. A base
    /// field may specialize this when it can safely delay modular reduction
    /// across several products.
    #[doc(hidden)]
    #[inline]
    fn extension_base_dot_product(
        extension_values: &[Self::Extension],
        base_scalars: &[Self],
    ) -> Self::Extension {
        extension_values
            .iter()
            .zip(base_scalars)
            .map(|(&value, &scalar)| {
                <Self::Extension as FieldExtension<D>>::scalar_mul(&value, scalar)
            })
            .sum()
    }

    /// Compute two independent dot products of extension-field values with
    /// base-field scalars in one pass over the shared powers slice.
    ///
    /// The default is exactly two calls to [`Self::extension_base_dot_product`]
    /// in input order, so every field and extension degree keeps its prior
    /// semantics unless it explicitly specializes this hook. A base field may
    /// specialize when one traversal loading each extension power once and
    /// folding it into two independent accumulators deletes a full second
    /// scan of the shared powers slice.
    #[doc(hidden)]
    #[inline]
    fn extension_base_dot_products_2(
        extension_values: &[Self::Extension],
        base_polynomials: [&[Self]; 2],
    ) -> [Self::Extension; 2] {
        [
            Self::extension_base_dot_product(extension_values, base_polynomials[0]),
            Self::extension_base_dot_product(extension_values, base_polynomials[1]),
        ]
    }

    /// Dot product `sum_i extension_values[i].scalar_mul(base_coefficients[i]
    /// * subgroup_scales[i])`, zipped across all three slices. This is the
    /// fused form of evaluating a base polynomial at `g·ζ` out of the
    /// already-built `ζ` powers and the natural-order subgroup powers:
    /// `P(g·ζ) = sum_i c_i · g^i · ζ^i`. The default performs ordinary field
    /// arithmetic; a base field may specialize it to delay reduction.
    #[doc(hidden)]
    #[inline]
    fn extension_base_dot_product_with_subgroup_scales(
        extension_values: &[Self::Extension],
        base_coefficients: &[Self],
        subgroup_scales: &[Self],
    ) -> Self::Extension {
        extension_values
            .iter()
            .zip(base_coefficients)
            .zip(subgroup_scales)
            .map(|((&value, &coefficient), &scale)| {
                <Self::Extension as FieldExtension<D>>::scalar_mul(&value, coefficient * scale)
            })
            .sum()
    }

    /// Internal FFT hook. The default preserves general extension
    /// multiplication; a base field may explicitly specialize multiplication
    /// by its own embedded twiddles without overlapping trait impls.
    #[doc(hidden)]
    #[inline(always)]
    fn mul_fft_quadratic_base_twiddle(twiddle: [Self; 2], value: [Self; 2]) -> [Self; 2] {
        let [a0, a1] = twiddle;
        let [b0, b1] = value;
        [a0 * b0 + Self::W * a1 * b1, a0 * b1 + a1 * b0]
    }

    /// Internal fixed-shape FRI hook. The default keeps the historical
    /// reversed Horner recurrence, including its raw field representation.
    /// A base field may specialize the production arity without changing the
    /// generic FRI implementation or any other reduction call site.
    #[doc(hidden)]
    #[inline(always)]
    fn fri_fold_arity16(
        terms: &[Self::Extension; 16],
        beta: Self::Extension,
        _beta_powers: &[Self::Extension; 16],
    ) -> Self::Extension {
        terms
            .iter()
            .rev()
            .fold(Self::Extension::ZERO, |acc, &term| acc * beta + term)
    }
}

impl<F: Field + Frobenius<1> + FieldExtension<1, BaseField = F>> Extendable<1> for F {
    type Extension = F;
    const W: Self = F::ONE;
    const DTH_ROOT: Self = F::ONE;
    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 1] = [F::MULTIPLICATIVE_GROUP_GENERATOR];
    const EXT_POWER_OF_TWO_GENERATOR: [Self; 1] = [F::POWER_OF_TWO_GENERATOR];
}

pub trait FieldExtension<const D: usize>: Field {
    type BaseField: Field;

    fn to_basefield_array(&self) -> [Self::BaseField; D];

    fn from_basefield_array(arr: [Self::BaseField; D]) -> Self;

    fn from_basefield(x: Self::BaseField) -> Self;

    fn is_in_basefield(&self) -> bool {
        self.to_basefield_array()[1..].iter().all(|x| x.is_zero())
    }

    fn scalar_mul(&self, scalar: Self::BaseField) -> Self {
        let mut res = self.to_basefield_array();
        res.iter_mut().for_each(|x| {
            *x *= scalar;
        });
        Self::from_basefield_array(res)
    }
}

impl<F: Field> FieldExtension<1> for F {
    type BaseField = F;

    fn to_basefield_array(&self) -> [Self::BaseField; 1] {
        [*self]
    }

    fn from_basefield_array(arr: [Self::BaseField; 1]) -> Self {
        arr[0]
    }

    fn from_basefield(x: Self::BaseField) -> Self {
        x
    }
}

/// Flatten the slice by sending every extension field element to its D-sized canonical representation.
pub fn flatten<F, const D: usize>(l: &[F::Extension]) -> Vec<F>
where
    F: Field + Extendable<D>,
{
    l.iter()
        .flat_map(|x| x.to_basefield_array().to_vec())
        .collect()
}

/// Batch every D-sized chunks into extension field elements.
pub fn unflatten<F, const D: usize>(l: &[F]) -> Vec<F::Extension>
where
    F: Field + Extendable<D>,
{
    debug_assert_eq!(l.len() % D, 0);
    l.chunks_exact(D)
        .map(|c| F::Extension::from_basefield_array(c.to_vec().try_into().unwrap()))
        .collect()
}
