//! Host-side secp256k1 arithmetic: curve constants, affine group law, the GLV
//! endomorphism decomposition the AIR's scalar loop consumes, and a reference
//! ECDSA verifier. Everything here is plain big-integer math used for trace
//! generation and tests; none of it is constrained.

use std::sync::OnceLock;

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

/// Number of 16-bit limbs in a 256-bit field element.
pub const LIMBS: usize = 16;
/// Bits per limb.
pub const LIMB_BITS: usize = 16;
/// Number of signed binary digits per GLV half-scalar (one EC loop row per digit).
pub const GLV_BITS: usize = 129;

pub struct Constants {
    pub p: BigUint,
    pub n: BigUint,
    pub beta: BigUint,
    pub lambda: BigUint,
    pub g: Point,
    pub phi_g: Point,
    /// Lattice vectors `(a, b)` with `a + b*lambda == 0 (mod n)`.
    pub v1: (BigInt, BigInt),
    pub v2: (BigInt, BigInt),
    /// Offset point the scalar loop starts from.
    pub offset: Point,
    /// `2^GLV_BITS * offset`: the loop ends at `R + offset_shift`.
    pub offset_shift: Point,
}

fn hex(s: &str) -> BigUint {
    BigUint::parse_bytes(s.as_bytes(), 16).expect("valid hex constant")
}

pub fn constants() -> &'static Constants {
    static C: OnceLock<Constants> = OnceLock::new();
    C.get_or_init(|| {
        let p = hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f");
        let n = hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
        let beta = hex("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
        let lambda = hex("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
        let g = Point {
            x: hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
            y: hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"),
        };
        let a1 = BigInt::from(hex("3086d221a7d46bcde86c90e49284eb15"));
        let b1 = -BigInt::from(hex("e4437ed6010e88286f547fa90abfe4c3"));
        let a2 = BigInt::from(hex("114ca50f7a8e2f3f657c1108d9d44cfd8"));
        let b2 = a1.clone();
        let phi_g = Point {
            x: (&g.x * &beta) % &p,
            y: g.y.clone(),
        };
        let mut c = Constants {
            p,
            n,
            beta,
            lambda,
            g,
            phi_g,
            v1: (a1, b1),
            v2: (a2, b2),
            offset: Point {
                x: BigUint::zero(),
                y: BigUint::zero(),
            },
            offset_shift: Point {
                x: BigUint::zero(),
                y: BigUint::zero(),
            },
        };
        c.offset = hash_to_curve(&c.p, b"p3-ecdsa/secp256k1/loop-offset");
        c.offset_shift = mul_with(&c, &c.offset, &(BigUint::one() << GLV_BITS)).expect("non-zero");
        c
    })
}

/// An affine secp256k1 point (the point at infinity is represented as `None`
/// wherever it can occur).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Point {
    pub x: BigUint,
    pub y: BigUint,
}

pub fn fp_inv(p: &BigUint, a: &BigUint) -> BigUint {
    a.modpow(&(p - 2u32), p)
}

fn sub_mod(a: &BigUint, b: &BigUint, m: &BigUint) -> BigUint {
    ((a % m) + m - (b % m)) % m
}

/// Try-and-increment map from a seed to a curve point with even `y`.
fn hash_to_curve(p: &BigUint, seed: &[u8]) -> Point {
    let mut x = BigUint::from_bytes_be(seed) % p;
    loop {
        let rhs = (x.modpow(&BigUint::from(3u32), p) + 7u32) % p;
        let y = rhs.modpow(&((p + 1u32) >> 2), p);
        if (&y * &y) % p == rhs {
            let y = if y.is_odd() { p - y } else { y };
            return Point { x, y };
        }
        x += 1u32;
    }
}

fn neg_with(c: &Constants, a: &Point) -> Point {
    Point {
        x: a.x.clone(),
        y: (&c.p - &a.y) % &c.p,
    }
}

pub fn neg(a: &Point) -> Point {
    neg_with(constants(), a)
}

fn add_with(c: &Constants, a: Option<&Point>, b: Option<&Point>) -> Option<Point> {
    let p = &c.p;
    let (a, b) = match (a, b) {
        (None, b) => return b.cloned(),
        (a, None) => return a.cloned(),
        (Some(a), Some(b)) => (a, b),
    };
    let lambda = if a.x == b.x {
        if (&a.y + &b.y) % p == BigUint::zero() {
            return None;
        }
        let num = (BigUint::from(3u32) * &a.x * &a.x) % p;
        (num * fp_inv(p, &((BigUint::from(2u32) * &a.y) % p))) % p
    } else {
        (sub_mod(&b.y, &a.y, p) * fp_inv(p, &sub_mod(&b.x, &a.x, p))) % p
    };
    let x = sub_mod(&sub_mod(&(&lambda * &lambda), &a.x, p), &b.x, p);
    let y = sub_mod(&(&lambda * sub_mod(&a.x, &x, p)), &a.y, p);
    Some(Point { x, y })
}

pub fn add(a: Option<&Point>, b: Option<&Point>) -> Option<Point> {
    add_with(constants(), a, b)
}

/// Jacobian `(X, Y, Z)` with `x = X/Z^2`, `y = Y/Z^3`; `None` is infinity.
type Jacobian = Option<(BigUint, BigUint, BigUint)>;

fn jacobian_double(p: &BigUint, a: &Jacobian) -> Jacobian {
    let (x, y, z) = a.as_ref()?;
    if y.is_zero() {
        return None;
    }
    let aa = (x * x) % p;
    let bb = (y * y) % p;
    let cc = (&bb * &bb) % p;
    let t = (x + &bb) % p;
    let d = (sub_mod(&sub_mod(&(&t * &t), &aa, p), &cc, p) * 2u32) % p;
    let e = (&aa * 3u32) % p;
    let x3 = sub_mod(&((&e * &e) % p), &((&d * 2u32) % p), p);
    let y3 = sub_mod(&((&e * sub_mod(&d, &x3, p)) % p), &((&cc * 8u32) % p), p);
    let z3 = (y * z * 2u32) % p;
    Some((x3, y3, z3))
}

fn jacobian_add_affine(p: &BigUint, a: &Jacobian, b: &Point) -> Jacobian {
    let Some((x1, y1, z1)) = a.as_ref() else {
        return Some((b.x.clone(), b.y.clone(), BigUint::from(1u32)));
    };
    let z1z1 = (z1 * z1) % p;
    let u2 = (&b.x * &z1z1) % p;
    let s2 = (&b.y * z1 * &z1z1) % p;
    let h = sub_mod(&u2, x1, p);
    let r = sub_mod(&s2, y1, p);
    if h.is_zero() {
        return if r.is_zero() {
            jacobian_double(p, a)
        } else {
            None
        };
    }
    let hh = (&h * &h) % p;
    let hhh = (&h * &hh) % p;
    let v = (x1 * &hh) % p;
    let x3 = sub_mod(&sub_mod(&((&r * &r) % p), &hhh, p), &((&v * 2u32) % p), p);
    let y3 = sub_mod(&((&r * sub_mod(&v, &x3, p)) % p), &((y1 * &hhh) % p), p);
    let z3 = (z1 * &h) % p;
    Some((x3, y3, z3))
}

fn mul_with(c: &Constants, a: &Point, k: &BigUint) -> Option<Point> {
    let p = &c.p;
    let mut acc: Jacobian = None;
    for i in (0..k.bits()).rev() {
        acc = jacobian_double(p, &acc);
        if k.bit(i) {
            acc = jacobian_add_affine(p, &acc, a);
        }
    }
    let (x, y, z) = acc?;
    let zi = fp_inv(p, &z);
    let zi2 = (&zi * &zi) % p;
    Some(Point {
        x: (x * &zi2) % p,
        y: (y * zi2 * zi) % p,
    })
}

pub fn mul(a: &Point, k: &BigUint) -> Option<Point> {
    mul_with(constants(), a, k)
}

pub fn is_on_curve(a: &Point) -> bool {
    let p = &constants().p;
    a.x < *p && a.y < *p && (&a.y * &a.y) % p == (a.x.modpow(&BigUint::from(3u32), p) + 7u32) % p
}

pub fn endo(a: &Point) -> Point {
    let c = constants();
    Point {
        x: (&a.x * &c.beta) % &c.p,
        y: a.y.clone(),
    }
}

/// Decompresses a SEC1 x-coordinate and y parity.
pub fn decompress(x: &BigUint, y_odd: bool) -> Option<Point> {
    let p = &constants().p;
    if x >= p {
        return None;
    }
    let rhs = (x.modpow(&BigUint::from(3u32), p) + 7u32) % p;
    let y = rhs.modpow(&((p + 1u32) >> 2), p);
    if (&y * &y) % p != rhs {
        return None;
    }
    let y = if y.is_odd() != y_odd { (p - &y) % p } else { y };
    if y.is_odd() != y_odd {
        return None;
    }
    Some(Point { x: x.clone(), y })
}

fn round_div(a: &BigInt, n: &BigInt) -> BigInt {
    let (q, r) = a.div_mod_floor(n);
    if &(r * 2) >= n { q + 1 } else { q }
}

/// Splits `k (mod n)` into odd `(k1, k2)` with `k == k1 + k2*lambda (mod n)` and
/// `|k1|, |k2| < 2^GLV_BITS`, so both have a `{-1, +1}` digit expansion of
/// length [`GLV_BITS`].
///
/// Writes `k_i = 2 m_i + 1` and Babai-rounds `m = (k - 1 - lambda) / 2 (mod n)`
/// against the basis `v1, v2`, so `|m1| <= (|a1| + |a2|) / 2` and
/// `|m2| <= (|b1| + |b2|) / 2`; both give `|k_i| < 2^129` for every `k`.
pub fn glv_decompose(k: &BigUint) -> (BigInt, BigInt) {
    let c = constants();
    let n = BigInt::from(c.n.clone());
    let half: BigInt = (&n + 1) >> 1u32;
    let shifted: BigInt = BigInt::from(k % &c.n) - 1 - BigInt::from(c.lambda.clone());
    let m = (shifted * half).mod_floor(&n);
    let (a1, b1) = &c.v1;
    let (a2, b2) = &c.v2;
    let c1 = round_div(&(b2 * &m), &n);
    let c2 = round_div(&(-b1 * &m), &n);
    let m1 = &m - &c1 * a1 - &c2 * a2;
    let m2 = -&c1 * b1 - &c2 * b2;
    let (k1, k2): (BigInt, BigInt) = (m1 * 2 + 1, m2 * 2 + 1);
    let bound = BigInt::one() << GLV_BITS;
    assert!(
        k1.abs() < bound && k2.abs() < bound,
        "GLV halves out of range"
    );
    (k1, k2)
}

/// The `{0,1}` digit bits `b` (MSB first) with `k = sum (2*b_i - 1) 2^(L-1-i)`.
pub fn signed_digit_bits(k: &BigInt) -> [bool; GLV_BITS] {
    let offset = (BigInt::one() << GLV_BITS) - 1;
    let b: BigInt = (k + offset) >> 1u32;
    let (sign, b) = b.into_parts();
    assert!(sign != Sign::Minus, "scalar out of digit range");
    core::array::from_fn(|i| b.bit((GLV_BITS - 1 - i) as u64))
}

/// `B = (k + 2^L - 1) / 2`, the non-negative integer whose bits are the digits.
pub fn digit_integer(k: &BigInt) -> BigUint {
    let offset = (BigInt::one() << GLV_BITS) - 1;
    let b: BigInt = (k + offset) >> 1u32;
    b.to_biguint().expect("in range")
}

pub fn mod_inv(a: &BigUint, m: &BigUint) -> BigUint {
    a.modpow(&(m - 2u32), m)
}

/// Reference ECDSA verification of `(r, s)` over message scalar `z` and key `q`.
/// Enforces `1 <= r < n`, low-s (`1 <= s <= (n-1)/2`) and an on-curve key.
pub fn ecdsa_verify(z: &BigUint, r: &BigUint, s: &BigUint, q: &Point) -> bool {
    let c = constants();
    let half = (&c.n - 1u32) >> 1;
    if r.is_zero() || *r >= c.n || s.is_zero() || *s > half || !is_on_curve(q) {
        return false;
    }
    let w = mod_inv(s, &c.n);
    let u1 = (z * &w) % &c.n;
    let u2 = (r * &w) % &c.n;
    match add(mul(&c.g, &u1).as_ref(), mul(q, &u2).as_ref()) {
        Some(rp) => (rp.x % &c.n) == *r,
        None => false,
    }
}

/// Reference ECDSA signing with an explicit nonce; normalizes to low-s.
pub fn ecdsa_sign(sk: &BigUint, z: &BigUint, nonce: &BigUint) -> Option<(BigUint, BigUint)> {
    let c = constants();
    let rp = mul(&c.g, nonce)?;
    let r = rp.x % &c.n;
    if r.is_zero() {
        return None;
    }
    let s = (mod_inv(nonce, &c.n) * ((z + &r * sk) % &c.n)) % &c.n;
    if s.is_zero() {
        return None;
    }
    let half = (&c.n - 1u32) >> 1;
    let s = if s > half { &c.n - s } else { s };
    Some((r, s))
}

/// Little-endian 16-bit limbs of `a` (which must fit in `count` limbs).
pub fn to_limbs(a: &BigUint, count: usize) -> Vec<u64> {
    let digits = a.to_u64_digits();
    let mut out = vec![0u64; count];
    for (i, limb) in out.iter_mut().enumerate() {
        let word = digits.get(i / 4).copied().unwrap_or(0);
        *limb = (word >> (16 * (i % 4))) & 0xffff;
    }
    debug_assert!(a.bits() as usize <= count * LIMB_BITS);
    out
}

pub fn from_limbs(limbs: &[u64]) -> BigUint {
    limbs
        .iter()
        .rev()
        .fold(BigUint::zero(), |acc, &l| (acc << LIMB_BITS) + l)
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    fn random_scalar(rng: &mut StdRng) -> BigUint {
        let bytes: [u8; 32] = rng.r#gen();
        BigUint::from_bytes_be(&bytes) % &constants().n
    }

    #[test]
    fn endomorphism_matches_lambda() {
        let c = constants();
        assert_eq!(mul(&c.g, &c.lambda).unwrap(), c.phi_g);
        assert!(is_on_curve(&c.offset));
        for (a, b) in [&c.v1, &c.v2] {
            let n = BigInt::from(c.n.clone());
            assert!(
                (a + b * BigInt::from(c.lambda.clone()))
                    .mod_floor(&n)
                    .is_zero()
            );
        }
    }

    #[test]
    fn glv_rounding_bound_fits_digits() {
        let c = constants();
        let bound = BigInt::one() << GLV_BITS;
        let (a1, b1) = &c.v1;
        let (a2, b2) = &c.v2;
        assert!(a1.abs() + a2.abs() + 1 < bound);
        assert!(b1.abs() + b2.abs() + 1 < bound);
    }

    #[test]
    fn glv_decomposition_is_odd_and_bounded() {
        let c = constants();
        let n = BigInt::from(c.n.clone());
        let lambda = BigInt::from(c.lambda.clone());
        let mut rng = StdRng::seed_from_u64(7);
        for i in 0..2000 {
            let k = match i {
                0 => BigUint::zero(),
                1 => &c.n - 1u32,
                _ => random_scalar(&mut rng),
            };
            let (k1, k2) = glv_decompose(&k);
            assert!(k1.is_odd() && k2.is_odd());
            assert_eq!((&k1 + &k2 * &lambda).mod_floor(&n), BigInt::from(k.clone()));
            let bits = signed_digit_bits(&k1);
            let recomposed = bits
                .iter()
                .fold(BigInt::zero(), |acc, &b| acc * 2 + if b { 1 } else { -1 });
            assert_eq!(recomposed, k1);
        }
    }

    #[test]
    fn sign_verify_roundtrip() {
        let c = constants();
        let mut rng = StdRng::seed_from_u64(1);
        let sk = random_scalar(&mut rng);
        let q = mul(&c.g, &sk).unwrap();
        let z = random_scalar(&mut rng);
        let (r, s) = ecdsa_sign(&sk, &z, &random_scalar(&mut rng)).unwrap();
        assert!(ecdsa_verify(&z, &r, &s, &q));
        assert!(!ecdsa_verify(&(&z + 1u32), &r, &s, &q));
        assert_eq!(decompress(&q.x, q.y.is_odd()).unwrap(), q);
    }
}
