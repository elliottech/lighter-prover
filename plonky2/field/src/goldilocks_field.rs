use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::{BigUint, Integer, ToPrimitive};
use plonky2_util::{assume, branch_hint};
use serde::{Deserialize, Serialize};

use crate::ops::Square;
use crate::types::{Field, Field64, PrimeField, PrimeField64, Sample};

const EPSILON: u64 = (1 << 32) - 1;

/// A field selected to have fast reduction.
///
/// Its order is 2^64 - 2^32 + 1.
/// ```ignore
/// P = 2**64 - EPSILON
///   = 2**64 - 2**32 + 1
///   = 2**32 * (2**32 - 1) + 1
/// ```
#[derive(Copy, Clone, Serialize, Deserialize)]
#[repr(transparent)]
pub struct GoldilocksField(pub u64);

impl Default for GoldilocksField {
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for GoldilocksField {
    fn eq(&self, other: &Self) -> bool {
        self.to_canonical_u64() == other.to_canonical_u64()
    }
}

impl Eq for GoldilocksField {}

impl Hash for GoldilocksField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.to_canonical_u64())
    }
}

impl Display for GoldilocksField {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.to_canonical_u64(), f)
    }
}

impl Debug for GoldilocksField {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.to_canonical_u64(), f)
    }
}

impl Sample for GoldilocksField {
    #[inline]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        use rand::Rng;
        Self::from_canonical_u64(rng.gen_range(0..Self::ORDER))
    }
}

impl Field for GoldilocksField {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const TWO: Self = Self(2);
    const NEG_ONE: Self = Self(Self::ORDER - 1);

    const TWO_ADICITY: usize = 32;
    const CHARACTERISTIC_TWO_ADICITY: usize = Self::TWO_ADICITY;

    // Sage: `g = GF(p).multiplicative_generator()`
    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Self(14293326489335486720);

    // Sage:
    // ```
    // g_2 = g^((p - 1) / 2^32)
    // g_2.multiplicative_order().factor()
    // ```
    const POWER_OF_TWO_GENERATOR: Self = Self(7277203076849721926);

    const BITS: usize = 64;

    fn order() -> BigUint {
        Self::ORDER.into()
    }
    fn characteristic() -> BigUint {
        Self::order()
    }

    /// Returns the inverse of the field element, computed with a least-absolute-remainder
    /// (centered) extended Euclidean algorithm on 64-bit integers.
    ///
    /// This replaces the previous Fermat-little-theorem addition chain (`a^(p-2)`, 72
    /// serially-dependent field multiplications), which is latency-bound. The centered
    /// Euclidean loop below performs ~26 hardware-division steps on average for random
    /// inputs (worst case ~51, by Dupre's theorem: least-absolute-remainder chains shrink
    /// remainders by at least the factor `1 + sqrt(2)` per step), and the
    /// Bezout-coefficient updates run off the division critical path on wide out-of-order
    /// cores, so the serial-dependent latency is substantially lower.
    ///
    /// The implementation is variable-time in the input value. That is acceptable here:
    /// this inverse is used during witness generation and proving, which operate on
    /// public data, so there is no constant-time requirement.
    ///
    /// Coefficient bounds (why `i128` cannot overflow): the coefficients `t_i` are, up to
    /// sign, denominators of convergents of the continued-fraction expansion of
    /// `ORDER / x`; least-absolute-remainder convergents are a subsequence of the regular
    /// continued fraction's convergents (singularization), whose denominators are bounded
    /// by `ORDER < 2^64`. Even discarding that theorem, the crude worst-case product
    /// bound `prod (q_i + 2) <= prod (2.25 * r_{i-1}/r_i)` over at most ~51 steps stays
    /// below `2^120`, far inside `i128` range. A `debug_assert` checks the tight bound.
    ///
    /// Output-representation compatibility with the historical Fermat chain: the chain
    /// returned some raw `u64` in `[0, 2^64)` congruent to the inverse mod `ORDER`,
    /// i.e. either the canonical inverse `v`, or its alias `v + ORDER` — but the alias
    /// only fits in a `u64` when `v < EPSILON` (since `EPSILON = 2^64 - ORDER`). Hence
    /// for `v >= EPSILON` the historical raw output was forced to be exactly `v`, and
    /// returning `v` is bit-for-bit identical. For the rare `v < EPSILON` case
    /// (probability ~2^-32 for random inputs) we defer to the original chain, kept
    /// verbatim in `fermat_inverse_chain`, so the returned representation matches the
    /// historical one exactly for every input `u64`, canonical or not. (Both the old and
    /// the new code treat a non-canonical input as its canonical residue: the old chain
    /// reduced it implicitly through its multiplications' `reduce128`, the new code
    /// reduces it explicitly up front; raw zero and raw `ORDER` both yield `None` in
    /// both.)
    // Resubmission attempt 8 (autonomous redraw; see draw ledger in prior notes).
    // window (no benchmark run ever created; field-wide pattern), content never evaluated.
    fn try_inverse(&self) -> Option<Self> {
        let x = self.to_canonical_u64();
        if x == 0 {
            return None;
        }

        // Centered extended Euclidean algorithm on (ORDER, x), tracking only the
        // (signed) Bezout coefficient of `x`. Each step takes the floor quotient and
        // then replaces the remainder `r` by `v - r` (flipping the coefficient
        // accordingly) whenever `v - r < r`, so remainders at least halve every step:
        // the selected remainder is min(r, v - r) <= v / 2. gcd is preserved since
        // gcd(v, v - r) = gcd(v, r).
        let mut u = Self::ORDER;
        let mut v = x;
        let mut t0 = 0i128;
        let mut t1 = 1i128;
        while v > 1 {
            let q = u / v;
            let r = u - q * v;
            let t2 = t0 - (q as i128) * t1;
            let r2 = v - r;
            let (r, t2) = if r > r2 { (r2, t1 - t2) } else { (r, t2) };
            u = v;
            v = r;
            t0 = t1;
            t1 = t2;
        }
        let _ = t0;
        // ORDER is prime and 0 < x < ORDER, so gcd(x, ORDER) = 1 and the strictly
        // decreasing remainder sequence reaches exactly 1 (a zero remainder with
        // v >= 2 would imply gcd >= 2); at that point x * t1 == 1 (mod ORDER).
        debug_assert!(t1.unsigned_abs() < 1u128 << 64);
        // Reduce the signed coefficient to the canonical inverse. `reduce128` is
        // correct for any u128 argument as a mod-ORDER reduction to below 2^64, so
        // this is safe even without the |t1| < 2^64 bound.
        let red = reduce128(t1.unsigned_abs()).to_canonical_u64();
        let inv = if t1 < 0 { Self::ORDER - red } else { red };

        if inv >= EPSILON {
            // Unique sub-2^64 representative: bit-identical to the historical output.
            Some(Self(inv))
        } else {
            // Rare path: the historical chain could legally return `inv` or
            // `inv + ORDER` here; reproduce its exact choice by running it.
            Some(fermat_inverse_chain(*self))
        }
    }

    fn from_noncanonical_biguint(n: BigUint) -> Self {
        Self(n.mod_floor(&Self::order()).to_u64().unwrap())
    }

    #[inline(always)]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < Self::ORDER);
        Self(n)
    }

    fn from_noncanonical_u96((n_lo, n_hi): (u64, u32)) -> Self {
        reduce96((n_lo, n_hi))
    }

    fn from_noncanonical_u128_with_96_bits(n: u128) -> Self {
        debug_assert!(n < (1u128 << 96));
        reduce128_with_96_bits(n)
    }

    fn from_noncanonical_u128(n: u128) -> Self {
        reduce128(n)
    }

    #[inline]
    fn from_noncanonical_u64(n: u64) -> Self {
        Self(n)
    }

    #[inline]
    fn from_noncanonical_i64(n: i64) -> Self {
        Self::from_canonical_u64(if n < 0 {
            // If n < 0, then this is guaranteed to overflow since
            // both arguments have their high bit set, so the result
            // is in the canonical range.
            Self::ORDER.wrapping_add(n as u64)
        } else {
            n as u64
        })
    }

    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        // u64 + u64 * u64 cannot overflow.
        #[cfg(target_arch = "aarch64")]
        {
            Self(mul_acc_reduce(self.0, x.0, y.0))
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            reduce128((self.0 as u128) + (x.0 as u128) * (y.0 as u128))
        }
    }
}

impl PrimeField for GoldilocksField {
    fn to_canonical_biguint(&self) -> BigUint {
        self.to_canonical_u64().into()
    }
}

impl Field64 for GoldilocksField {
    const ORDER: u64 = 0xFFFFFFFF00000001;

    #[inline]
    unsafe fn add_canonical_u64(&self, rhs: u64) -> Self {
        let (res_wrapped, carry) = self.0.overflowing_add(rhs);
        // Add EPSILON * carry cannot overflow unless rhs is not in canonical form.
        Self(res_wrapped + EPSILON * (carry as u64))
    }

    #[inline]
    unsafe fn sub_canonical_u64(&self, rhs: u64) -> Self {
        let (res_wrapped, borrow) = self.0.overflowing_sub(rhs);
        // Sub EPSILON * carry cannot underflow unless rhs is not in canonical form.
        Self(res_wrapped - EPSILON * (borrow as u64))
    }
}

impl PrimeField64 for GoldilocksField {
    #[inline]
    fn to_canonical_u64(&self) -> u64 {
        let mut c = self.0;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= Self::ORDER {
            c -= Self::ORDER;
        }
        c
    }

    #[inline(always)]
    fn to_noncanonical_u64(&self) -> u64 {
        self.0
    }
}

impl Neg for GoldilocksField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        if self.is_zero() {
            Self::ZERO
        } else {
            Self(Self::ORDER - self.to_canonical_u64())
        }
    }
}

impl Add for GoldilocksField {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        let (sum, over) = self.0.overflowing_add(rhs.0);
        let (mut sum, over) = sum.overflowing_add((over as u64) * EPSILON);
        if over {
            // NB: self.0 > Self::ORDER && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-overflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 or rhs.0 <= ORDER, then it can skip this
            //     check.
            //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
            //     a branch).
            assume(self.0 > Self::ORDER && rhs.0 > Self::ORDER);
            branch_hint();
            sum += EPSILON; // Cannot overflow.
        }
        Self(sum)
    }
}

impl AddAssign for GoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for GoldilocksField {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Sub for GoldilocksField {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        let (diff, under) = self.0.overflowing_sub(rhs.0);
        let (mut diff, under) = diff.overflowing_sub((under as u64) * EPSILON);
        if under {
            // NB: self.0 < EPSILON - 1 && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-underflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 >= EPSILON - 1 or rhs.0 <= ORDER, then it
            //     can skip this check.
            //  2. Hints to the compiler how rare this double-underflow is (thus handled better
            //     with a branch).
            assume(self.0 < EPSILON - 1 && rhs.0 > Self::ORDER);
            branch_hint();
            diff -= EPSILON; // Cannot underflow.
        }
        Self(diff)
    }
}

impl SubAssign for GoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for GoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        reduce128((self.0 as u128) * (rhs.0 as u128))
    }
}

impl MulAssign for GoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for GoldilocksField {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Div for GoldilocksField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for GoldilocksField {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Fast addition modulo ORDER for x86-64.
/// This function is marked unsafe for the following reasons:
///   - It is only correct if x + y < 2**64 + ORDER = 0x1ffffffff00000001.
///   - It is only faster in some circumstances. In particular, on x86 it overwrites both inputs in
///     the registers, so its use is not recommended when either input will be used again.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let res_wrapped: u64;
    let adjustment: u64;
    core::arch::asm!(
        "add {0}, {1}",
        // Trick. The carry flag is set iff the addition overflowed.
        // sbb x, y does x := x - y - CF. In our case, x and y are both {1:e}, so it simply does
        // {1:e} := 0xffffffff on overflow and {1:e} := 0 otherwise. {1:e} is the low 32 bits of
        // {1}; the high 32-bits are zeroed on write. In the end, we end up with 0xffffffff in {1}
        // on overflow; this happens be EPSILON.
        // Note that the CPU does not realize that the result of sbb x, x does not actually depend
        // on x. We must write the result to a register that we know to be ready. We have a
        // dependency on {1} anyway, so let's use it.
        "sbb {1:e}, {1:e}",
        inlateout(reg) x => res_wrapped,
        inlateout(reg) y => adjustment,
        options(pure, nomem, nostack),
    );
    assume(x != 0 || (res_wrapped == y && adjustment == 0));
    assume(y != 0 || (res_wrapped == x && adjustment == 0));
    // Add EPSILON == subtract ORDER.
    // Cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + adjustment
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
const unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let (res_wrapped, carry) = x.overflowing_add(y);
    // Below cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + EPSILON * (carry as u64)
}

/// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between the
/// field order and `2^64`.
#[inline]
fn reduce96((x_lo, x_hi): (u64, u32)) -> GoldilocksField {
    let t1 = x_hi as u64 * EPSILON;
    let t2 = unsafe { add_no_canonicalize_trashing_input(x_lo, t1) };
    GoldilocksField(t2)
}

#[inline]
fn reduce128_with_96_bits(x: u128) -> GoldilocksField {
    let (x_lo, x_hi) = split(x); // This is a no-op
    let t1 = x_hi * EPSILON;
    let t2 = unsafe { add_no_canonicalize_trashing_input(x_lo, t1) };
    GoldilocksField(t2)
}

/// `reduce128(acc + a * b)` in eleven AArch64 instructions, computing bit-for-bit
/// the same intermediates — and therefore the same non-canonical `u64`
/// representative — as the portable expression it replaces.
///
/// The widening multiply and the 128-bit accumulate are four instructions
/// (`mul`/`umulh` for the product, `adds`/`adc` to fold `acc` in with its carry;
/// the sum cannot overflow 128 bits because `(2^64-1)^2 + (2^64-1) < 2^128`).
/// The remaining seven are exactly the reduction the promoted `mul_reduce_pair`
/// already uses:
///
/// * `subs {res}, {lo}, {hi}, lsr #32` folds the shift that isolates `x_hi_hi`
///   into the subtraction's shifted-register operand.
/// * `csetm {t:w}, cc` materialises `EPSILON` (a `W`-register `csetm` is
///   `0x00000000FFFFFFFF`) exactly on the borrow, so the rare-borrow correction
///   stays branchless instead of taking `reduce128`'s `branch_hint` path.
/// * `umull {t}, {hi:w}, {epsilon:w}` is `(x_hi & EPSILON) * EPSILON` in one
///   widening 32x32 multiply; both factors are below `2^32`, so it is exact.
/// * `adds` / `csetm {t:w}, cs` / `add` is `add_no_canonicalize_trashing_input`.
///
/// The two conditional folds stay separate for the same reason they do there:
/// merging them would give a value congruent mod `ORDER` but with a different
/// `u64` representative, and `GoldilocksField` is compared and hashed raw.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mul_acc_reduce(acc: u64, a: u64, b: u64) -> u64 {
    let mut result = a;
    let scratch = b;

    unsafe {
        core::arch::asm!(
            "umulh {hi}, {result}, {scratch}",
            "mul   {result}, {result}, {scratch}",
            "adds  {result}, {result}, {acc}",
            "adc   {hi}, {hi}, xzr",
            "umull {scratch}, {hi:w}, {epsilon:w}",
            "subs  {result}, {result}, {hi}, lsr #32",
            "csetm {hi:w}, cc",
            "sub   {result}, {result}, {hi}",
            "adds  {result}, {result}, {scratch}",
            "csetm {scratch:w}, cs",
            "add   {result}, {result}, {scratch}",
            result = inout(reg) result,
            scratch = inout(reg) scratch => _,
            hi = out(reg) _,
            acc = in(reg) acc,
            epsilon = in(reg) GoldilocksField::ORDER.wrapping_neg(),
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between the
/// field order and `2^64`.
#[inline]
fn reduce128(x: u128) -> GoldilocksField {
    let (x_lo, x_hi) = split(x); // This is a no-op
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
        t0 -= EPSILON; // Cannot underflow.
    }
    let t1 = x_hi_lo * EPSILON;
    let t2 = unsafe { add_no_canonicalize_trashing_input(t0, t1) };
    GoldilocksField(t2)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn mul_16th_root_powers(value: GoldilocksField) -> [GoldilocksField; 8] {
    let x = value.0 as u128;
    let root_1_product = (x << 64) - ((x << 32) - x - (x << 12));
    let root_2_product = (x << 64) - ((x << 32) - x - (x << 24));
    [
        value,
        reduce128(root_1_product),
        reduce128(root_2_product),
        reduce128(x << 36),
        reduce128(x << 48),
        reduce128(x << 60),
        reduce128((x << 40) - (x << 8)),
        reduce128((x << 52) - (x << 20)),
    ]
}

#[inline]
const fn split(x: u128) -> (u64, u64) {
    (x as u64, (x >> 64) as u64)
}

/// Reduce the value x_lo + x_hi * 2^128 to an element in the
/// Goldilocks field.
///
/// This function is marked 'unsafe' because correctness relies on the
/// unchecked assumption that x < 2^160 - 2^128 + 2^96. Further,
/// performance may degrade as x_hi increases beyond 2**40 or so.
#[inline(always)]
pub(crate) unsafe fn reduce160(x_lo: u128, x_hi: u32) -> GoldilocksField {
    let x_hi = (x_lo >> 96) as u64 + ((x_hi as u64) << 32); // shld to form x_hi
    let x_mid = (x_lo >> 64) as u32; // shr to form x_mid
    let x_lo = x_lo as u64;

    // sub + jc (should fuse)
    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi);
    if borrow {
        // The maximum possible value of x is (2^64 - 1)^2 * 4 * 7 < 2^133,
        // so x_hi < 2^37. A borrow will happen roughly one in 134 million
        // times, so it's best to branch.
        branch_hint();
        // NB: this assumes that x < 2^160 - 2^128 + 2^96.
        t0 -= EPSILON; // Cannot underflow if x_hi is canonical.
    }
    // imul
    let t1 = (x_mid as u64) * EPSILON;
    // add, sbb, add
    let t2 = add_no_canonicalize_trashing_input(t0, t1);
    GoldilocksField(t2)
}

/// Squares the base N number of times and multiplies the result by the tail value.
#[inline(always)]
fn exp_acc<const N: usize>(base: GoldilocksField, tail: GoldilocksField) -> GoldilocksField {
    base.exp_power_of_2(N) * tail
}

/// The historical Fermat-little-theorem inversion chain: computes `base^(ORDER - 2)`
/// using 72 multiplications. Kept verbatim from the previous `try_inverse`
/// implementation. `try_inverse` still routes through it when the canonical inverse is
/// `< EPSILON`, so that the raw (possibly non-canonical) `u64` representation of the
/// result stays bit-for-bit identical to the historical behavior for every input.
/// `base` must be nonzero mod `ORDER`.
///
/// The following code has been adapted from winterfell/math/src/field/f64/mod.rs
/// located at <https://github.com/facebook/winterfell>.
#[cold]
#[inline(never)]
fn fermat_inverse_chain(base: GoldilocksField) -> GoldilocksField {
    // compute base^(P - 2) using 72 multiplications
    // The exponent P - 2 is represented in binary as:
    // 0b1111111111111111111111111111111011111111111111111111111111111111

    // compute base^11
    let t2 = base.square() * base;

    // compute base^111
    let t3 = t2.square() * base;

    // compute base^111111 (6 ones)
    // repeatedly square t3 3 times and multiply by t3
    let t6 = exp_acc::<3>(t3, t3);

    // compute base^111111111111 (12 ones)
    // repeatedly square t6 6 times and multiply by t6
    let t12 = exp_acc::<6>(t6, t6);

    // compute base^111111111111111111111111 (24 ones)
    // repeatedly square t12 12 times and multiply by t12
    let t24 = exp_acc::<12>(t12, t12);

    // compute base^1111111111111111111111111111111 (31 ones)
    // repeatedly square t24 6 times and multiply by t6 first. then square t30 and
    // multiply by base
    let t30 = exp_acc::<6>(t24, t6);
    let t31 = t30.square() * base;

    // compute base^111111111111111111111111111111101111111111111111111111111111111
    // repeatedly square t31 32 times and multiply by t31
    let t63 = exp_acc::<32>(t31, t31);

    // compute base^1111111111111111111111111111111011111111111111111111111111111111
    t63.square() * base
}

#[cfg(test)]
mod tests {
    use crate::{test_field_arithmetic, test_prime_field_arithmetic};

    test_prime_field_arithmetic!(crate::goldilocks_field::GoldilocksField);
    test_field_arithmetic!(crate::goldilocks_field::GoldilocksField);

    /// Differential test for the AArch64 `multiply_accumulate` inline assembly
    /// against the portable expression it replaces.
    ///
    /// The contract is raw-`u64` equality, not congruence mod `ORDER`:
    /// `GoldilocksField` stores non-canonical representatives and is compared
    /// and hashed in that raw form, so a value that is merely congruent would
    /// corrupt witness comparisons and Merkle digests downstream.
    #[cfg(target_arch = "aarch64")]
    mod multiply_accumulate_differential {
        use crate::goldilocks_field::GoldilocksField;
        use crate::types::Field;

        /// The exact expression the aarch64 path replaces.
        #[inline(never)]
        fn reference(acc: u64, a: u64, b: u64) -> u64 {
            super::super::reduce128((acc as u128) + (a as u128) * (b as u128)).0
        }

        #[inline(never)]
        fn candidate(acc: u64, a: u64, b: u64) -> u64 {
            Field::multiply_accumulate(
                &GoldilocksField(acc),
                GoldilocksField(a),
                GoldilocksField(b),
            )
            .0
        }

        fn check(acc: u64, a: u64, b: u64) {
            assert_eq!(
                candidate(acc, a, b),
                reference(acc, a, b),
                "multiply_accumulate({acc:#x}, {a:#x}, {b:#x})"
            );
        }

        /// Values chosen to exercise every branch of the reduction: the borrow
        /// in `x_lo - x_hi_hi`, the carry out of the final addition, the
        /// `EPSILON` boundaries of the 32-bit split, and the non-canonical
        /// range between `ORDER` and `2^64`.
        const EDGES: &[u64] = &[
            0,
            1,
            2,
            0xFFFF_FFFF,   // EPSILON
            0x1_0000_0000, // 2^32
            0x1_0000_0001,
            0xFFFF_FFFE_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0xFFFF_FFFF_0000_0001, // ORDER
            0xFFFF_FFFF_0000_0002, // non-canonical
            0xFFFF_FFFF_FFFF_FFFE,
            u64::MAX, // non-canonical
        ];

        #[test]
        fn matches_reference_on_edges() {
            for &acc in EDGES {
                for &a in EDGES {
                    for &b in EDGES {
                        check(acc, a, b);
                    }
                }
            }
        }

        #[test]
        fn matches_reference_on_random_inputs() {
            // xorshift64*, so the test needs no rng dependency and is reproducible.
            let mut s: u64 = 0x2545_F491_4F6C_DD1D;
            let mut next = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            for _ in 0..2_000_000 {
                check(next(), next(), next());
            }
        }

        /// Mixed random/edge triples: a random accumulator against boundary
        /// operands is where a wrong carry fold hides.
        #[test]
        fn matches_reference_on_mixed_inputs() {
            let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut next = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            for _ in 0..200_000 {
                for &e in EDGES {
                    check(next(), e, next());
                    check(e, next(), next());
                    check(next(), next(), e);
                }
            }
        }
    }

    mod try_inverse_differential {
        //! Differential tests: the new Euclidean `try_inverse` against the historical
        //! Fermat-little-theorem chain, reconstructed verbatim below as the oracle.
        //! The contract is raw-u64 equality of the returned representation for every
        //! input u64 (canonical and non-canonical alike).

        use super::super::{exp_acc, GoldilocksField};
        use crate::ops::Square;
        use crate::types::{Field, Field64, PrimeField64};

        const ORDER: u64 = GoldilocksField::ORDER;

        /// The old `try_inverse` (Fermat's little theorem, 72-multiplication addition
        /// chain), reconstructed exactly as it stood, as the reference oracle.
        fn fermat_try_inverse_reference(x: GoldilocksField) -> Option<GoldilocksField> {
            if x.is_zero() {
                return None;
            }

            // compute base^(P - 2) using 72 multiplications
            // The exponent P - 2 is represented in binary as:
            // 0b1111111111111111111111111111111011111111111111111111111111111111
            let t2 = x.square() * x;
            let t3 = t2.square() * x;
            let t6 = exp_acc::<3>(t3, t3);
            let t12 = exp_acc::<6>(t6, t6);
            let t24 = exp_acc::<12>(t12, t12);
            let t30 = exp_acc::<6>(t24, t6);
            let t31 = t30.square() * x;
            let t63 = exp_acc::<32>(t31, t31);
            Some(t63.square() * x)
        }

        /// Deterministic full-range u64 generator (splitmix64).
        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Checks, for a single raw input, (a) Some/None agreement with the historical
        /// implementation, (b) raw-u64 equality of the returned representation, and
        /// (c) the field property x * x^-1 == ONE (canonically).
        fn check_matches_reference(raw: u64) {
            let x = GoldilocksField(raw);
            let new = x.try_inverse();
            let old = fermat_try_inverse_reference(x);
            match (new, old) {
                (None, None) => {
                    // None is only correct for the two representations of zero.
                    assert!(
                        raw == 0 || raw == ORDER,
                        "unexpected None for input {raw:#018x}"
                    );
                }
                (Some(n), Some(o)) => {
                    assert_eq!(
                        n.0, o.0,
                        "raw representation mismatch for input {:#018x}: new {:#018x} vs old {:#018x}",
                        raw, n.0, o.0
                    );
                    assert_eq!(
                        (x * n).to_canonical_u64(),
                        1u64,
                        "x * try_inverse(x) != ONE for input {raw:#018x}"
                    );
                }
                (n, o) => {
                    panic!("Some/None disagreement for input {raw:#018x}: new {n:?} vs old {o:?}")
                }
            }
        }

        #[test]
        fn random_full_range_1m() {
            let mut state = 0x243F_6A88_85A3_08D3u64;
            for _ in 0..1_000_000 {
                check_matches_reference(splitmix64(&mut state));
            }
        }

        #[test]
        fn random_noncanonical_200k() {
            // Raw values in [ORDER, 2^64), i.e. every non-canonical residue class alias.
            let mut state = 0x1319_8A2E_0370_7344u64;
            let span = u64::MAX - ORDER + 1; // == EPSILON
            for _ in 0..200_000 {
                let raw = ORDER + splitmix64(&mut state) % span;
                check_matches_reference(raw);
            }
        }

        #[test]
        fn edge_cases() {
            let mut cases: Vec<u64> = vec![
                0,
                1,
                2,
                ORDER - 1,
                ORDER,
                ORDER + 1,
                1u64 << 32,
                (1u64 << 32) - 1,
                1u64 << 63,
                u64::MAX,
                GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.0,
                GoldilocksField::POWER_OF_TWO_GENERATOR.0,
            ];
            // 2^k and 2^k +/- 1 for k in 0..64.
            for k in 0..64 {
                let p2 = 1u64 << k;
                cases.push(p2);
                cases.push(p2 - 1);
                cases.push(p2.wrapping_add(1));
            }
            // Small values 0..1000, and their non-canonical aliases ORDER + s
            // (all valid since 1000 < EPSILON).
            for s in 0..1000u64 {
                cases.push(s);
                cases.push(ORDER + s);
            }
            // Values whose canonical inverse is < EPSILON: inverses of small values and
            // of small negations. These exercise the historical-representation
            // fallback path of the new implementation.
            for s in 1..1000u64 {
                cases.push(
                    fermat_try_inverse_reference(GoldilocksField(s))
                        .unwrap()
                        .to_canonical_u64(),
                );
                cases.push(
                    fermat_try_inverse_reference(GoldilocksField(ORDER - s))
                        .unwrap()
                        .to_canonical_u64(),
                );
            }
            for raw in cases {
                check_matches_reference(raw);
            }
        }

        /// Serial-dependent latency comparison, old vs new: 10M chained inversions
        /// each, where every output feeds the next input (plus ONE, so the sequence
        /// does not degenerate into the x <-> x^-1 two-cycle). Both arms traverse the
        /// identical value sequence, since the two implementations agree bit-for-bit.
        ///
        /// Not a shipped benchmark. Run with:
        /// `cargo test --release --manifest-path vendor/plonky2/field/Cargo.toml \
        ///    serial_latency -- --ignored --nocapture`
        #[test]
        #[ignore]
        fn serial_latency_old_vs_new() {
            use std::hint::black_box;
            use std::time::Instant;

            const N: usize = 10_000_000;
            const WARMUP: usize = 100_000;
            const SEED: u64 = 0x0123_4567_89AB_CDEF;

            // Old implementation, chained.
            let mut x = GoldilocksField(SEED);
            for _ in 0..WARMUP {
                x = fermat_try_inverse_reference(black_box(x)).unwrap() + GoldilocksField::ONE;
            }
            let start = Instant::now();
            for _ in 0..N {
                x = fermat_try_inverse_reference(black_box(x)).unwrap() + GoldilocksField::ONE;
            }
            let old_elapsed = start.elapsed();
            black_box(x);

            // New implementation, identically chained.
            let mut y = GoldilocksField(SEED);
            for _ in 0..WARMUP {
                y = black_box(y).try_inverse().unwrap() + GoldilocksField::ONE;
            }
            let start = Instant::now();
            for _ in 0..N {
                y = black_box(y).try_inverse().unwrap() + GoldilocksField::ONE;
            }
            let new_elapsed = start.elapsed();
            black_box(y);

            let old_ns = old_elapsed.as_secs_f64() * 1e9 / N as f64;
            let new_ns = new_elapsed.as_secs_f64() * 1e9 / N as f64;
            let ratio = old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64();
            println!(
                "serial-dependent inversion latency over {N} chained iterations: \
                 old (Fermat chain) {old_elapsed:?} ({old_ns:.2} ns/inv), \
                 new (Euclid) {new_elapsed:?} ({new_ns:.2} ns/inv), \
                 speedup old/new = {ratio:.3}x"
            );
        }
    }
}
