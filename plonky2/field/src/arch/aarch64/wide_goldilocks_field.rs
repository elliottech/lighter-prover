use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::{fmt, slice};

use super::neon_goldilocks_field::NeonGoldilocksField;
use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::Field;

/// Four packed Goldilocks elements implemented as two independent AArch64 lane pairs.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct WideGoldilocksField([NeonGoldilocksField; 2]);

impl WideGoldilocksField {
    #[inline]
    fn from_lanes(lanes: [GoldilocksField; 4]) -> Self {
        Self([
            NeonGoldilocksField([lanes[0], lanes[1]]),
            NeonGoldilocksField([lanes[2], lanes[3]]),
        ])
    }

    #[inline]
    fn lanes(self) -> [GoldilocksField; 4] {
        [
            self.0[0].0[0],
            self.0[0].0[1],
            self.0[1].0[0],
            self.0[1].0[1],
        ]
    }
}

impl Add<Self> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Add<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] + rhs, self.0[1] + rhs])
    }
}

impl Add<WideGoldilocksField> for GoldilocksField {
    type Output = WideGoldilocksField;

    #[inline]
    fn add(self, rhs: WideGoldilocksField) -> WideGoldilocksField {
        rhs + self
    }
}

impl AddAssign<Self> for WideGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: GoldilocksField) {
        *self = *self + rhs;
    }
}

impl fmt::Debug for WideGoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WideGoldilocksField")
            .field(&self.lanes())
            .finish()
    }
}

impl Default for WideGoldilocksField {
    #[inline]
    fn default() -> Self {
        Self::ZEROS
    }
}

impl Div<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: GoldilocksField) -> Self {
        self * rhs.inverse()
    }
}

impl From<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn from(value: GoldilocksField) -> Self {
        Self([NeonGoldilocksField::from(value); 2])
    }
}

impl Mul<Self> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Two independent two-lane reduction blocks rather than one four-lane block.
        // The nine-instruction pair kernel needs only seven registers, so the four
        // lanes still issue back-to-back while the register allocator keeps the
        // butterfly's loads, twiddles and accumulators live across the call.
        Self([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1]])
    }
}

impl Mul<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl Mul<WideGoldilocksField> for GoldilocksField {
    type Output = WideGoldilocksField;

    #[inline]
    fn mul(self, rhs: WideGoldilocksField) -> WideGoldilocksField {
        rhs * self
    }
}

impl MulAssign<Self> for WideGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: GoldilocksField) {
        *self = *self * rhs;
    }
}

impl Neg for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl Product for WideGoldilocksField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONES)
    }
}

unsafe impl PackedField for WideGoldilocksField {
    type Scalar = GoldilocksField;

    const WIDTH: usize = 4;
    const ZEROS: Self = Self([NeonGoldilocksField::ZEROS; 2]);
    const ONES: Self = Self([NeonGoldilocksField::ONES; 2]);

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
        unsafe { slice::from_raw_parts(self.0.as_ptr().cast(), Self::WIDTH) }
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar] {
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), Self::WIDTH) }
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let a = self.lanes();
        let b = other.lanes();
        match block_len {
            1 => (
                Self::from_lanes([a[0], b[0], a[2], b[2]]),
                Self::from_lanes([a[1], b[1], a[3], b[3]]),
            ),
            2 => (
                Self::from_lanes([a[0], a[1], b[0], b[1]]),
                Self::from_lanes([a[2], a[3], b[2], b[3]]),
            ),
            4 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }

    /// Delegates to the two `NeonGoldilocksField` halves, each of which fuses
    /// its lane pair into one interleaved multiply-accumulate-reduce block.
    /// Routing lane by lane through the scalar path instead would emit four
    /// separate assembly blocks; the compiler schedules a block as a unit, so
    /// their serial dependency chains could not overlap.
    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        Self([
            self.0[0].multiply_accumulate(x.0[0], y.0[0]),
            self.0[1].multiply_accumulate(x.0[1], y.0[1]),
        ])
    }
}

impl Square for WideGoldilocksField {
    #[inline]
    fn square(&self) -> Self {
        Self([self.0[0].square(), self.0[1].square()])
    }
}

impl Sub<Self> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Sub<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] - rhs, self.0[1] - rhs])
    }
}

impl Sub<WideGoldilocksField> for GoldilocksField {
    type Output = WideGoldilocksField;

    #[inline]
    fn sub(self, rhs: WideGoldilocksField) -> WideGoldilocksField {
        WideGoldilocksField::from(self) - rhs
    }
}

impl SubAssign<Self> for WideGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: GoldilocksField) {
        *self = *self - rhs;
    }
}

impl Sum for WideGoldilocksField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZEROS)
    }
}

#[cfg(test)]
mod tests {
    use super::WideGoldilocksField;
    use crate::goldilocks_field::GoldilocksField;
    use crate::ops::Square;
    use crate::packed::PackedField;
    use crate::types::{Field, Field64};

    fn boundary_values() -> [GoldilocksField; 12] {
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
    fn four_lane_operations_match_scalar_field() {
        let a = [
            GoldilocksField::ZERO,
            GoldilocksField::ONE,
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER),
            GoldilocksField::from_noncanonical_u64(u64::MAX),
        ];
        let b = [
            GoldilocksField::from_noncanonical_u64(14_479_013_849_828_404_771),
            GoldilocksField::from_noncanonical_u64(9_087_029_921_428_221_768),
            GoldilocksField::from_noncanonical_u64(2_441_288_194_761_790_662),
            GoldilocksField::TWO,
        ];
        let packed_a = *WideGoldilocksField::from_slice(&a);
        let packed_b = *WideGoldilocksField::from_slice(&b);
        assert_eq!(
            (packed_a + packed_b).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] + b[i])
        );
        assert_eq!(
            (packed_a - packed_b).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] - b[i])
        );
        assert_eq!(
            (packed_a * packed_b).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] * b[i])
        );
        assert_eq!(
            packed_a.square().as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i].square())
        );
        assert_eq!(
            packed_a.multiply_accumulate(packed_b, packed_a).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] + b[i] * a[i])
        );

        for block_len in [1, 2, 4] {
            let (left, right) = packed_a.interleave(packed_b, block_len);
            assert_eq!(left.interleave(right, block_len), (packed_a, packed_b));
        }
    }

    #[test]
    fn fused_reduction_matches_scalar_boundary_products() {
        let values = boundary_values();
        for i in 0..values.len() {
            for j in 0..values.len() {
                let a: [GoldilocksField; 4] =
                    core::array::from_fn(|lane| values[(i + 2 * lane) % values.len()]);
                let b: [GoldilocksField; 4] =
                    core::array::from_fn(|lane| values[(j + 3 * lane) % values.len()]);
                let packed_a = *WideGoldilocksField::from_slice(&a);
                let packed_b = *WideGoldilocksField::from_slice(&b);

                assert_eq!(
                    (packed_a * packed_b).as_slice(),
                    core::array::from_fn::<_, 4, _>(|lane| a[lane] * b[lane])
                );
                assert_eq!(
                    packed_a.square().as_slice(),
                    core::array::from_fn::<_, 4, _>(|lane| a[lane].square())
                );
            }
        }
    }

    /// The packed `multiply_accumulate` override must agree with the scalar one
    /// on the RAW `u64` representative, not merely modulo `ORDER`.
    ///
    /// `GoldilocksField`'s `PartialEq` canonicalises, so the assertions above
    /// would pass for any congruent value. Non-canonical representatives do
    /// reach Merkle hashing and witness comparison in this prover, so the
    /// packed and scalar paths must be bit-identical.
    #[test]
    fn multiply_accumulate_matches_scalar_raw_representative() {
        fn raw(x: GoldilocksField) -> u64 {
            x.0
        }

        let values = boundary_values();
        // xorshift64*, so the test needs no rng dependency and is reproducible.
        let mut s: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            GoldilocksField(s)
        };

        let mut check =
            |acc: [GoldilocksField; 4], x: [GoldilocksField; 4], y: [GoldilocksField; 4]| {
                let packed = WideGoldilocksField::from_slice(&acc).multiply_accumulate(
                    *WideGoldilocksField::from_slice(&x),
                    *WideGoldilocksField::from_slice(&y),
                );
                let got: [u64; 4] = core::array::from_fn(|lane| raw(packed.as_slice()[lane]));
                let want: [u64; 4] = core::array::from_fn(|lane| {
                    raw(Field::multiply_accumulate(&acc[lane], x[lane], y[lane]))
                });
                assert_eq!(got, want, "acc={acc:?} x={x:?} y={y:?}");
            };

        // Every boundary triple, rotated across lanes so each lane sees a
        // different combination (and therefore a different borrow/carry mix).
        for i in 0..values.len() {
            for j in 0..values.len() {
                for k in 0..values.len() {
                    let acc = core::array::from_fn(|lane| values[(i + lane) % values.len()]);
                    let x = core::array::from_fn(|lane| values[(j + 2 * lane) % values.len()]);
                    let y = core::array::from_fn(|lane| values[(k + 3 * lane) % values.len()]);
                    check(acc, x, y);
                }
            }
        }

        for _ in 0..200_000 {
            let acc = core::array::from_fn(|_| next());
            let x = core::array::from_fn(|_| next());
            let y = core::array::from_fn(|_| next());
            check(acc, x, y);
        }

        // Random accumulator against boundary operands: where a wrong carry
        // fold hides.
        for &v in values.iter() {
            for _ in 0..2_000 {
                let acc = core::array::from_fn(|_| next());
                let x = core::array::from_fn(|_| v);
                let y = core::array::from_fn(|_| next());
                check(acc, x, y);
                let acc = core::array::from_fn(|_| v);
                let x = core::array::from_fn(|_| next());
                let y = core::array::from_fn(|_| next());
                check(acc, x, y);
            }
        }
    }
}
