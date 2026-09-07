use core::arch::asm;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::{Field, Field64};

/// Two independent Goldilocks field elements evaluated in parallel on AArch64.
///
/// The representation deliberately retains the scalar field's eight-byte alignment so that a
/// scalar slice can be reinterpreted as packed values. Multiplication interleaves both lanes in one
/// assembly block to hide the latency of AArch64's scalar `mul` and `umulh` instructions.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct NeonGoldilocksField(pub [GoldilocksField; 2]);

impl Add<Self> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Add<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: GoldilocksField) -> Self {
        self + Self::from(rhs)
    }
}

impl Add<NeonGoldilocksField> for GoldilocksField {
    type Output = NeonGoldilocksField;

    #[inline]
    fn add(self, rhs: NeonGoldilocksField) -> NeonGoldilocksField {
        NeonGoldilocksField::from(self) + rhs
    }
}

impl AddAssign<Self> for NeonGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: GoldilocksField) {
        *self = *self + rhs;
    }
}

impl fmt::Debug for NeonGoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NeonGoldilocksField").field(&self.0).finish()
    }
}

impl Default for NeonGoldilocksField {
    #[inline]
    fn default() -> Self {
        Self::ZEROS
    }
}

impl Div<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: GoldilocksField) -> Self {
        self * rhs.inverse()
    }
}

impl DivAssign<GoldilocksField> for NeonGoldilocksField {
    #[allow(clippy::suspicious_op_assign_impl)]
    #[inline]
    fn div_assign(&mut self, rhs: GoldilocksField) {
        *self *= rhs.inverse();
    }
}

impl From<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn from(value: GoldilocksField) -> Self {
        Self([value; 2])
    }
}

impl Mul<Self> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (lane0, lane1) = mul_reduce_pair(self.0[0].0, rhs.0[0].0, self.0[1].0, rhs.0[1].0);
        Self([GoldilocksField(lane0), GoldilocksField(lane1)])
    }
}

impl Mul<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: GoldilocksField) -> Self {
        self * Self::from(rhs)
    }
}

impl Mul<NeonGoldilocksField> for GoldilocksField {
    type Output = NeonGoldilocksField;

    #[inline]
    fn mul(self, rhs: NeonGoldilocksField) -> NeonGoldilocksField {
        NeonGoldilocksField::from(self) * rhs
    }
}

impl MulAssign<Self> for NeonGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: GoldilocksField) {
        *self = *self * rhs;
    }
}

impl Neg for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl Product for NeonGoldilocksField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONES)
    }
}

unsafe impl PackedField for NeonGoldilocksField {
    type Scalar = GoldilocksField;

    const WIDTH: usize = 2;
    const ZEROS: Self = Self([GoldilocksField::ZERO; 2]);
    const ONES: Self = Self([GoldilocksField::ONE; 2]);

    #[inline]
    fn from_slice(slice: &[Self::Scalar]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Scalar]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Scalar] {
        &self.0
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar] {
        &mut self.0
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 => (Self([self.0[0], other.0[0]]), Self([self.0[1], other.0[1]])),
            2 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }
}

impl Square for NeonGoldilocksField {
    #[inline]
    fn square(&self) -> Self {
        let (lane0, lane1) = mul_reduce_pair(self.0[0].0, self.0[0].0, self.0[1].0, self.0[1].0);
        Self([GoldilocksField(lane0), GoldilocksField(lane1)])
    }
}

impl Sub<Self> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Sub<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: GoldilocksField) -> Self {
        self - Self::from(rhs)
    }
}

impl Sub<NeonGoldilocksField> for GoldilocksField {
    type Output = NeonGoldilocksField;

    #[inline]
    fn sub(self, rhs: NeonGoldilocksField) -> NeonGoldilocksField {
        NeonGoldilocksField::from(self) - rhs
    }
}

impl SubAssign<Self> for NeonGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: GoldilocksField) {
        *self = *self - rhs;
    }
}

impl Sum for NeonGoldilocksField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZEROS)
    }
}

/// Reduce two independent 128-bit products modulo
/// `2^64 - 2^32 + 1`, interleaving the instruction streams for ILP.
///
/// Nine instructions per multiply, computing bit-for-bit the same intermediates as the
/// scalar `reduce128`:
///
/// * `subs {result}, {result}, {hi}, lsr #32` folds the `lsr` that isolates `x_hi_hi`
///   into the subtraction's shifted-register operand (AArch64 data-processing
///   instructions take a free `LSR #imm` on the second source).
/// * `umull {scratch}, {hi:w}, {epsilon:w}` is the 32x32 -> 64 widening multiply, which
///   computes `(x_hi & EPSILON) * EPSILON` in one instruction. It replaces the
///   `and` / `lsl #32` / `sub` triple that built the same product out of shifts: the
///   `W`-register operands mask to the low 32 bits for free, and both factors are
///   below `2^32`, so the 64-bit product is exact.
///
/// The two conditional folds stay branchless and stay separate: merging them into a
/// single signed carry counter would produce a value congruent mod `ORDER` but with a
/// different `u64` representative, and `GoldilocksField` values are compared and hashed
/// in their raw non-canonical form.
///
/// Seven registers per pair (down from fifteen), which lets the surrounding butterfly
/// keep its loads, twiddles and accumulators in flight.
#[inline(always)]
fn mul_reduce_pair(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
    let mut result0 = a0;
    let mut result1 = a1;
    let scratch0 = b0;
    let scratch1 = b1;

    unsafe {
        asm!(
            "umulh {hi0}, {result0}, {scratch0}",
            "umulh {hi1}, {result1}, {scratch1}",
            "mul   {result0}, {result0}, {scratch0}",
            "mul   {result1}, {result1}, {scratch1}",
            "umull {scratch0}, {hi0:w}, {epsilon:w}",
            "umull {scratch1}, {hi1:w}, {epsilon:w}",
            "subs  {result0}, {result0}, {hi0}, lsr #32",
            "csetm {hi0:w}, cc",
            "subs  {result1}, {result1}, {hi1}, lsr #32",
            "csetm {hi1:w}, cc",
            "sub   {result0}, {result0}, {hi0}",
            "sub   {result1}, {result1}, {hi1}",
            "adds  {result0}, {result0}, {scratch0}",
            "csetm {scratch0:w}, cs",
            "adds  {result1}, {result1}, {scratch1}",
            "csetm {scratch1:w}, cs",
            "add   {result0}, {result0}, {scratch0}",
            "add   {result1}, {result1}, {scratch1}",
            result0 = inout(reg) result0,
            result1 = inout(reg) result1,
            scratch0 = inout(reg) scratch0 => _,
            scratch1 = inout(reg) scratch1 => _,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            epsilon = in(reg) GoldilocksField::ORDER.wrapping_neg(),
            options(pure, nomem, nostack),
        );
    }

    (result0, result1)
}

#[cfg(test)]
mod tests {
    use super::NeonGoldilocksField;
    use crate::goldilocks_field::GoldilocksField;
    use crate::ops::Square;
    use crate::packed::PackedField;
    use crate::types::{Field, Field64};

    fn values() -> [GoldilocksField; 12] {
        [
            GoldilocksField::ZERO,
            GoldilocksField::ONE,
            GoldilocksField::TWO,
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER - 1),
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER),
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER + 1),
            GoldilocksField::from_noncanonical_u64(u32::MAX as u64),
            GoldilocksField::from_noncanonical_u64(1 << 32),
            GoldilocksField::from_noncanonical_u64(u64::MAX),
            GoldilocksField::from_noncanonical_u64(14_479_013_849_828_404_771),
            GoldilocksField::from_noncanonical_u64(9_087_029_921_428_221_768),
            GoldilocksField::from_noncanonical_u64(2_441_288_194_761_790_662),
        ]
    }

    #[test]
    fn arithmetic_matches_scalar_field() {
        let values = values();
        for i in 0..values.len() {
            for j in 0..values.len() {
                let a = NeonGoldilocksField([values[i], values[(i + 5) % values.len()]]);
                let b = NeonGoldilocksField([values[j], values[(j + 7) % values.len()]]);
                let scalar = values[(i + j) % values.len()];

                assert_eq!((a + b).0, [a.0[0] + b.0[0], a.0[1] + b.0[1]]);
                assert_eq!((a - b).0, [a.0[0] - b.0[0], a.0[1] - b.0[1]]);
                assert_eq!((a * b).0, [a.0[0] * b.0[0], a.0[1] * b.0[1]]);
                assert_eq!((a * scalar).0, [a.0[0] * scalar, a.0[1] * scalar]);
                assert_eq!((-a).0, [-a.0[0], -a.0[1]]);
                assert_eq!(a.square().0, [a.0[0].square(), a.0[1].square()]);
            }
        }
    }

    #[test]
    fn packed_views_and_interleave_preserve_lane_order() {
        let mut values = [
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(1),
            GoldilocksField::from_canonical_u64(10),
            GoldilocksField::from_canonical_u64(11),
        ];
        let packed = NeonGoldilocksField::pack_slice(&values);
        let (left, right) = packed[0].interleave(packed[1], 1);
        assert_eq!(left.0, [values[0], values[2]]);
        assert_eq!(right.0, [values[1], values[3]]);
        assert_eq!(left.interleave(right, 1), (packed[0], packed[1]));

        NeonGoldilocksField::pack_slice_mut(&mut values)[1] += GoldilocksField::ONE;
        assert_eq!(values[2], GoldilocksField::from_canonical_u64(11));
        assert_eq!(values[3], GoldilocksField::from_canonical_u64(12));
    }
}
