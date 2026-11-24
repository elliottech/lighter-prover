//! Non-native 256-bit arithmetic over 16-bit limbs.
//!
//! A relation is an integer polynomial identity `E(X) (+ c*M) = q(X)*M(X)` at
//! `X = 2^16`, where `E`'s coefficients are expressions in limb columns. It is
//! checked by grouping coefficients in pairs (radix `2^32`) and propagating
//! signed carries `g_k + c_{k-1} = 2^32 c_k`. All limbs, quotient limbs and
//! carry halves are range-checked to 16 bits, which keeps every grouped
//! identity far below the Goldilocks modulus, so it holds over the integers.
//!
//! The same code runs over [`Int`] (host-side witness generation) and over
//! [`Fx`] (AIR constraint expressions).

use core::ops::{Add, Mul, Neg, Sub};

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive, Zero};
use p3_field::PrimeCharacteristicRing;

use super::layout::{CARRIES, CARRY_OFFSET, COEFFS, GROUPS, Q_LIMBS};
use crate::secp::{constants, to_limbs};

pub trait LimbRing:
    Clone + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self>
{
    fn constant(v: i128) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int(pub i128);

impl Add for Int {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Int(self.0 + o.0)
    }
}
impl Sub for Int {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Int(self.0 - o.0)
    }
}
impl Mul for Int {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Int(self.0 * o.0)
    }
}
impl Neg for Int {
    type Output = Self;
    fn neg(self) -> Self {
        Int(-self.0)
    }
}
impl LimbRing for Int {
    fn constant(v: i128) -> Self {
        Int(v)
    }
}

#[derive(Clone, Debug)]
pub struct Fx<E>(pub E);

impl<E: PrimeCharacteristicRing> Add for Fx<E> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Fx(self.0 + o.0)
    }
}
impl<E: PrimeCharacteristicRing> Sub for Fx<E> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Fx(self.0 - o.0)
    }
}
impl<E: PrimeCharacteristicRing> Mul for Fx<E> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Fx(self.0 * o.0)
    }
}
impl<E: PrimeCharacteristicRing> Neg for Fx<E> {
    type Output = Self;
    fn neg(self) -> Self {
        Fx(-self.0)
    }
}
impl<E: PrimeCharacteristicRing> LimbRing for Fx<E> {
    fn constant(v: i128) -> Self {
        let mag = E::from_u64(v.unsigned_abs() as u64);
        debug_assert!(v.unsigned_abs() < 1 << 64);
        Fx(if v < 0 { -mag } else { mag })
    }
}

pub type Poly<R> = Vec<R>;

pub fn konst<R: LimbRing>(limbs: &[u64]) -> Poly<R> {
    limbs.iter().map(|&l| R::constant(l as i128)).collect()
}

pub fn padd<R: LimbRing>(a: &[R], b: &[R]) -> Poly<R> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| match (a.get(i), b.get(i)) {
            (Some(x), Some(y)) => x.clone() + y.clone(),
            (Some(x), None) => x.clone(),
            (None, Some(y)) => y.clone(),
            (None, None) => unreachable!(),
        })
        .collect()
}

pub fn pneg<R: LimbRing>(a: &[R]) -> Poly<R> {
    a.iter().map(|x| -x.clone()).collect()
}

pub fn psub<R: LimbRing>(a: &[R], b: &[R]) -> Poly<R> {
    padd(a, &pneg(b))
}

pub fn pscale<R: LimbRing>(a: &[R], c: i128) -> Poly<R> {
    a.iter().map(|x| x.clone() * R::constant(c)).collect()
}

pub fn pmul<R: LimbRing>(a: &[R], b: &[R]) -> Poly<R> {
    let mut out: Vec<Option<R>> = vec![None; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let t = x.clone() * y.clone();
            out[i + j] = Some(match out[i + j].take() {
                Some(acc) => acc + t,
                None => t,
            });
        }
    }
    out.into_iter().map(|c| c.unwrap()).collect()
}

/// Sum of polynomials.
pub fn psum<R: LimbRing>(terms: &[Poly<R>]) -> Poly<R> {
    terms
        .iter()
        .skip(1)
        .fold(terms[0].clone(), |acc, t| padd(&acc, t))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modulus {
    /// Exact integer identity, no quotient.
    None,
    /// secp256k1 base field.
    P,
    /// secp256k1 group order.
    N,
}

pub struct Relation<R> {
    pub coeffs: Poly<R>,
    pub modulus: Modulus,
}

impl<R> Relation<R> {
    pub fn modp(coeffs: Poly<R>) -> Self {
        Self {
            coeffs,
            modulus: Modulus::P,
        }
    }
    pub fn modn(coeffs: Poly<R>) -> Self {
        Self {
            coeffs,
            modulus: Modulus::N,
        }
    }
    pub fn exact(coeffs: Poly<R>) -> Self {
        Self {
            coeffs,
            modulus: Modulus::None,
        }
    }
}

/// Exponent of the offset multiple `2^OFFSET_BITS * M` added to modular
/// relations so the quotient is non-negative.
pub const OFFSET_BITS: usize = 260;

pub struct ModulusLimbs {
    pub m: Vec<u64>,
    pub offset: Vec<u64>,
    pub m_int: BigUint,
}

pub fn modulus_limbs(modulus: Modulus) -> Option<&'static ModulusLimbs> {
    use std::sync::OnceLock;
    static P: OnceLock<ModulusLimbs> = OnceLock::new();
    static N: OnceLock<ModulusLimbs> = OnceLock::new();
    let make = |m: &BigUint| ModulusLimbs {
        m: to_limbs(m, 16),
        offset: to_limbs(&(m << OFFSET_BITS), COEFFS),
        m_int: m.clone(),
    };
    match modulus {
        Modulus::None => None,
        Modulus::P => Some(P.get_or_init(|| make(&constants().p))),
        Modulus::N => Some(N.get_or_init(|| make(&constants().n))),
    }
}

/// `E + offset - q*M`, padded to [`COEFFS`] coefficients.
pub fn residual<R: LimbRing>(rel: &Relation<R>, q: &[R]) -> Poly<R> {
    assert!(rel.coeffs.len() <= COEFFS);
    let mut out = rel.coeffs.clone();
    if let Some(ml) = modulus_limbs(rel.modulus) {
        out = padd(&out, &konst::<R>(&ml.offset));
        out = psub(&out, &pmul(q, &konst::<R>(&ml.m)));
    }
    out.resize_with(COEFFS, || R::constant(0));
    out
}

/// The [`GROUPS`] carry-chain constraints of a residual; all are zero iff the
/// residual evaluates to zero at `2^16` (given the range checks).
pub fn group_constraints<R: LimbRing>(res: &[R], carries: &[R]) -> Vec<R> {
    (0..GROUPS)
        .map(|g| {
            let mut acc = res[2 * g].clone();
            if 2 * g + 1 < COEFFS {
                acc = acc + res[2 * g + 1].clone() * R::constant(1 << 16);
            }
            if g > 0 {
                acc = acc + carries[g - 1].clone();
            }
            if g < CARRIES {
                acc = acc - carries[g].clone() * R::constant(1 << 32);
            }
            acc
        })
        .collect()
}

pub fn carry_from_halves<R: LimbRing>(lo: R, hi: R) -> R {
    lo + hi * R::constant(1 << 16) - R::constant(CARRY_OFFSET as i128)
}

fn eval_poly(coeffs: &[Int]) -> BigInt {
    coeffs
        .iter()
        .rev()
        .fold(BigInt::zero(), |acc, c| (acc << 16) + BigInt::from(c.0))
}

/// Host side: the quotient limbs and carry halves `(q, lo, hi)` that satisfy
/// `rel`. Panics if the witness values do not satisfy the relation.
pub fn solve_relation(rel: &Relation<Int>) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let q = match modulus_limbs(rel.modulus) {
        None => vec![0u64; Q_LIMBS],
        Some(ml) => {
            let shifted = eval_poly(&rel.coeffs) + BigInt::from(&ml.m_int << OFFSET_BITS);
            let (q, r) = shifted.div_mod_floor(&BigInt::from(ml.m_int.clone()));
            assert!(
                r.is_zero() && !q.is_negative(),
                "modular relation does not hold"
            );
            to_limbs(&q.to_biguint().unwrap(), Q_LIMBS)
        }
    };
    let q_int: Vec<Int> = q.iter().map(|&x| Int(x as i128)).collect();
    let res = residual(rel, &q_int);
    let mut carries = Vec::with_capacity(CARRIES);
    let mut prev = 0i128;
    for g in 0..GROUPS {
        let mut acc = res[2 * g].0 + prev;
        if 2 * g + 1 < COEFFS {
            acc += res[2 * g + 1].0 << 16;
        }
        if g < CARRIES {
            assert_eq!(acc.rem_euclid(1 << 32), 0, "relation carry is not exact");
            prev = acc >> 32;
            carries.push(prev);
        } else {
            assert_eq!(acc, 0, "relation does not hold over the integers");
        }
    }
    let mut lo = Vec::with_capacity(CARRIES);
    let mut hi = Vec::with_capacity(CARRIES);
    for c in carries {
        let v = (c + CARRY_OFFSET as i128).to_u64().expect("carry in range");
        assert!(v < 1 << 32, "carry out of range");
        lo.push(v & 0xffff);
        hi.push(v >> 16);
    }
    (q, lo, hi)
}

pub fn int_limbs(a: &BigUint, n: usize) -> Vec<Int> {
    to_limbs(a, n).into_iter().map(|x| Int(x as i128)).collect()
}
