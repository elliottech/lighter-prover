//! The non-native relations each row kind checks, one per relation slot.
//! Shared by the AIR (over [`Fx`](super::limbs::Fx)) and witness generation
//! (over [`Int`](super::limbs::Int)).

use std::sync::OnceLock;

use num_bigint::BigUint;
use num_traits::One;

use super::layout::*;
use super::limbs::{LimbRing, Poly, Relation, konst, padd, pmul, pneg, pscale, psub, psum};
use crate::secp::{self, constants, to_limbs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    Setup0,
    Setup1,
    Ec,
}

pub const ROW_KINDS: [RowKind; 3] = [RowKind::Setup0, RowKind::Setup1, RowKind::Ec];

impl RowKind {
    pub fn selector_col(self) -> usize {
        match self {
            RowKind::Setup0 => P_SETUP0,
            RowKind::Setup1 => P_SETUP1,
            RowKind::Ec => P_EC,
        }
    }

    pub fn at_offset(offset: usize) -> RowKind {
        match offset {
            0 => RowKind::Setup0,
            1 => RowKind::Setup1,
            _ => RowKind::Ec,
        }
    }
}

pub struct ConstLimbs {
    pub p: Vec<u64>,
    pub n: Vec<u64>,
    pub beta: Vec<u64>,
    pub lambda: Vec<u64>,
    pub p_minus_1: Vec<u64>,
    pub n_minus_1: Vec<u64>,
    pub half_n: Vec<u64>,
    pub digit_offset: Vec<u64>,
    pub g1: (Vec<u64>, Vec<u64>),
    pub g2: (Vec<u64>, Vec<u64>),
    pub offset: (Vec<u64>, Vec<u64>),
    pub shift: (Vec<u64>, Vec<u64>),
    /// `-shift`.
    pub neg_shift: (Vec<u64>, Vec<u64>),
}

pub fn const_limbs() -> &'static ConstLimbs {
    static C: OnceLock<ConstLimbs> = OnceLock::new();
    C.get_or_init(|| {
        let c = constants();
        let l = |v: &BigUint| to_limbs(v, LIMBS);
        let pt = |p: &secp::Point| (l(&p.x), l(&p.y));
        let g1 = secp::add(Some(&c.g), Some(&c.phi_g)).unwrap();
        let g2 = secp::add(Some(&c.g), Some(&secp::neg(&c.phi_g))).unwrap();
        ConstLimbs {
            p: l(&c.p),
            n: l(&c.n),
            beta: l(&c.beta),
            lambda: l(&c.lambda),
            p_minus_1: l(&(&c.p - 1u32)),
            n_minus_1: l(&(&c.n - 1u32)),
            half_n: l(&((&c.n - 1u32) >> 1)),
            digit_offset: to_limbs(&((BigUint::one() << GLV_BITS) - 1u32), DIGIT_LIMBS),
            g1: pt(&g1),
            g2: pt(&g2),
            offset: pt(&c.offset),
            shift: pt(&c.offset_shift),
            neg_shift: pt(&secp::neg(&c.offset_shift)),
        }
    })
}

fn col<R: LimbRing>(row: &[R], start: usize) -> Poly<R> {
    row[start..start + LIMBS].to_vec()
}

fn sq<R: LimbRing>(a: &[R]) -> Poly<R> {
    pmul(a, a)
}

/// `K = 2B - (2^L - 1)` for GLV half-scalar `j`.
fn signed_scalar<R: LimbRing>(row: &[R], j: usize) -> Poly<R> {
    let b = row[digit_limbs(j)..digit_limbs(j) + DIGIT_LIMBS].to_vec();
    psub(&pscale(&b, 2), &konst(&const_limbs().digit_offset))
}

/// Point-addition relations `(lambda, x3, y3) = (x1, y1) + (x2, y2)`.
fn add_relations<R: LimbRing>(
    lambda: &[R],
    (x1, y1): (&[R], &[R]),
    (x2, y2): (&[R], &[R]),
    (x3, y3): (&[R], &[R]),
) -> [Relation<R>; 3] {
    [
        Relation::modp(psub(&pmul(lambda, &psub(x2, x1)), &psub(y2, y1))),
        Relation::modp(psum(&[sq(lambda), pneg(x1), pneg(x2), pneg(x3)])),
        Relation::modp(psum(&[pmul(lambda, &psub(x1, x3)), pneg(y1), pneg(y3)])),
    ]
}

/// `a * c` for a single ring element `c`.
fn pmulc<R: LimbRing>(a: &[R], c: &R) -> Poly<R> {
    a.iter().map(|x| x.clone() * c.clone()).collect()
}

/// Addition `P1 + P2 = P3`: `i1` flags `P1` as the point at infinity, `i3` flags `P3`
/// (`None`: `P3` is finite). `x2_sq` is `x2^2` as a limb product.
fn complete_add_relations<R: LimbRing>(
    lambda: &[R],
    (x1, y1, i1): (&[R], &[R], &R),
    (x2, y2, x2_sq): (&[R], &[R], &[R]),
    (x3, y3, i3): (&[R], &[R], Option<&R>),
) -> [Relation<R>; 4] {
    let zero = R::constant(0);
    let i3 = i3.unwrap_or(&zero);
    let c = R::constant(1) - i1.clone() - i3.clone();
    let slope_x = psub(&pmul(lambda, &psub(x2, x1)), &psub(y2, y1));
    let slope_y = psub(
        &pmul(lambda, &padd(y1, y2)),
        &psum(&[sq(x1), pmul(x1, x2), x2_sq.to_vec()]),
    );
    let out_x = psum(&[sq(lambda), pneg(x1), pneg(x2), pneg(x3)]);
    let out_y = psum(&[pmul(lambda, &psub(x1, x3)), pneg(y1), pneg(y3)]);
    [
        Relation::modp(padd(&pmulc(&slope_x, &c), &pmulc(&psub(x2, x1), i3))),
        Relation::modp(padd(&pmulc(&slope_y, &c), &pmulc(&padd(y1, y2), i3))),
        Relation::modp(padd(&pmulc(&out_x, &c), &pmulc(&psub(x2, x3), i1))),
        Relation::modp(padd(&pmulc(&out_y, &c), &pmulc(&psub(y2, y3), i1))),
    ]
}

/// `a - b - d` (so `a >= b` once `d` is range-checked).
fn diff<R: LimbRing>(a: &[R], b: &[R], d: &[R]) -> Poly<R> {
    psub(&psub(a, b), d)
}

/// Two 16-coefficient integer identities in one slot: `E1 + 2^272 * E2 = 0`.
/// `|E1| < 2^272` forces `E1 = 0` and then `E2 = 0`.
fn exact_pair<R: LimbRing>(e1: Poly<R>, e2: Poly<R>) -> Relation<R> {
    debug_assert!(e1.len() <= Q_LIMBS && e2.len() + Q_LIMBS <= COEFFS);
    let mut coeffs = e1;
    coeffs.resize_with(Q_LIMBS, || R::constant(0));
    coeffs.extend(e2);
    Relation::exact(coeffs)
}

/// Table selection for a digit pair `(ba, bb)` with product column `m = ba * bb`: the entry is
/// `P1` when the digits agree and `P2` otherwise, negated when `ba = 0`. Returns
/// `(x, y, x^2)`, each of degree 2 in the digit columns: with `e = [ba == bb] = 1 - ba - bb + 2m`,
/// `x = e*P1.x + (1-e)*P2.x`, `y = (ba+bb-1)*P1.y + (ba-bb)*P2.y`, `x^2 = e*P1.x^2 + (1-e)*P2.x^2`.
fn select<R: LimbRing>(
    (ba, bb, m): (&R, &R, &R),
    (x1, y1, x1_sq): (&[R], &[R], &[R]),
    (x2, y2, x2_sq): (&[R], &[R], &[R]),
) -> (Poly<R>, Poly<R>, Poly<R>) {
    let one = R::constant(1);
    let e = one.clone() - ba.clone() - bb.clone() + m.clone() * R::constant(2);
    let pick = |a: &[R], b: &[R]| padd(&pmulc(a, &e), &pmulc(b, &(one.clone() - e.clone())));
    let y = padd(
        &pmulc(y1, &(ba.clone() + bb.clone() - one.clone())),
        &pmulc(y2, &(ba.clone() - bb.clone())),
    );
    (pick(x1, x2), y, pick(x1_sq, x2_sq))
}

/// The relations checked on a row of `kind`, in slot order. `ec_last` is the
/// preprocessed flag of the last scalar-loop row, whose result is the
/// `RPX/RPY` registers instead of the next row's accumulator.
pub fn relations<R: LimbRing>(
    kind: RowKind,
    local: &[R],
    next: &[R],
    ec_last: &R,
) -> Vec<Relation<R>> {
    let k = const_limbs();
    let w = |i: usize| col(local, work(i));
    let (qx, qy) = (col(local, QX), col(local, QY));
    // `R` occupies the accumulator columns of the `Setup1` row.
    let (rx, ry) = (col(local, ACCX), col(local, ACCY));
    let (sr, ss) = (col(local, SIG_R), col(local, SIG_S));
    let xphi = col(local, XPHI);
    let (t2x, t2y) = (col(local, TQ2X), col(local, TQ2Y));
    let one: Poly<R> = vec![R::constant(1)];
    match kind {
        RowKind::Setup0 => {
            // W0 = lambda of Q - phi(Q), W1 = Qx^2, W2..W5 = range differences.
            let (lam, qx2) = (w(0), w(1));
            let neg_qy = pneg(&qy);
            let [l, x, y] = add_relations(&lam, (&qx, &qy), (&xphi, &neg_qy), (&t2x, &t2y));
            vec![
                Relation::modp(psub(&pmul(&konst(&k.beta), &qx), &xphi)),
                l,
                x,
                y,
                Relation::modp(psub(&sq(&qx), &qx2)),
                Relation::modp(psum(&[
                    sq(&qy),
                    pneg(&pmul(&qx2, &qx)),
                    vec![R::constant(-7)],
                ])),
                // 1 <= r <= n - 1.
                exact_pair(
                    diff(&sr, &one, &w(2)),
                    diff(&konst(&k.n_minus_1), &sr, &w(3)),
                ),
                // Q.x, Q.y < p.
                exact_pair(
                    diff(&konst(&k.p_minus_1), &qx, &w(4)),
                    diff(&konst(&k.p_minus_1), &qy, &w(5)),
                ),
            ]
        }
        RowKind::Setup1 => {
            // W0 = lambda of RP - shift, W1 = s * lambda (mod n), W2..W4 = range differences.
            let (sx, sy) = (konst::<R>(&k.neg_shift.0), konst::<R>(&k.neg_shift.1));
            let (rpx, rpy) = (col(local, RPX), col(local, RPY));
            let [l0, l1, x, y] = complete_add_relations(
                &w(0),
                (&rpx, &rpy, &local[RP_INF]),
                (&sx, &sy, &sq(&sx)),
                (&rx, &ry, None),
            );
            let sl = w(1);
            let z: Poly<R> = (0..LIMBS)
                .map(|m| local[ZLO + m].clone() + local[ZHI + m].clone() * R::constant(256))
                .collect();
            // s * (k_a + lambda * k_b) == target (mod n).
            let glv = |a: usize, b: usize, target: &[R]| {
                Relation::modn(psum(&[
                    pmul(&ss, &signed_scalar(local, a)),
                    pmul(&sl, &signed_scalar(local, b)),
                    pneg(target),
                ]))
            };
            let n = konst::<R>(&k.n);
            let wrap: Poly<R> = n
                .iter()
                .map(|v| v.clone() * local[R_WRAP].clone())
                .collect();
            vec![
                l0,
                x,
                y,
                Relation::modn(psub(&pmul(&ss, &konst(&k.lambda)), &sl)),
                glv(0, 1, &sr),
                glv(2, 3, &z),
                // R.x = r + t*n with t in {0, 1}; R.x < p.
                exact_pair(
                    psum(&[rx.clone(), pneg(&sr), pneg(&wrap)]),
                    diff(&konst(&k.p_minus_1), &rx, &w(2)),
                ),
                // 1 <= s <= (n - 1) / 2.
                exact_pair(diff(&ss, &one, &w(3)), diff(&konst(&k.half_n), &ss, &w(4))),
                l1,
            ]
        }
        RowKind::Ec => {
            // acc' = (2 * acc + T_Q) + T_G with complete additions:
            // W0..W2 = (lambda, x, y) of D = 2 * acc, W3..W5 = (lambda, x, y) of E = D + T_Q,
            // W6 = lambda of E + T_G. `ACC_INF`, `ADD_Q_INF`, `ADD_G_INF` flag acc, E and the
            // result as the point at infinity.
            let b: Vec<R> = (0..NUM_SCALARS).map(|j| local[BITS + j].clone()).collect();
            let (mq, mg) = (&local[DIGIT_PROD], &local[DIGIT_PROD + 1]);
            let t1x = pneg(&padd(&qx, &xphi));
            let t1y = pneg(&qy);
            let (xq, yq, xq_sq) = select(
                (&b[0], &b[1], mq),
                (&t1x, &t1y, &sq(&t1x)),
                (&t2x, &t2y, &sq(&t2x)),
            );
            let (g1x, g1y) = (konst::<R>(&k.g1.0), konst::<R>(&k.g1.1));
            let (g2x, g2y) = (konst::<R>(&k.g2.0), konst::<R>(&k.g2.1));
            let (xg, yg, xg_sq) = select(
                (&b[2], &b[3], mg),
                (&g1x, &g1y, &sq(&g1x)),
                (&g2x, &g2y, &sq(&g2x)),
            );
            let (ld, xd, yd, le, xe, ye, lf) = (w(0), w(1), w(2), w(3), w(4), w(5), w(6));
            let (x1, y1) = (col(local, ACCX), col(local, ACCY));
            let (acc_inf, e_inf, out_inf) = (&local[ACC_INF], &local[ADD_Q_INF], &local[ADD_G_INF]);
            let out = |acc: usize, reg: usize| -> Poly<R> {
                (0..LIMBS)
                    .map(|i| {
                        next[acc + i].clone()
                            + ec_last.clone() * (local[reg + i].clone() - next[acc + i].clone())
                    })
                    .collect()
            };
            let (x5, y5) = (out(ACCX, RPX), out(ACCY, RPY));
            // Doubling: a finite curve point has `y != 0`, so the tangent slope is unique. For
            // acc at infinity D is unused (`ACC_INF` carries over as E's `i1`).
            let mut rels = vec![
                Relation::modp(psub(&pscale(&pmul(&ld, &y1), 2), &pscale(&sq(&x1), 3))),
                Relation::modp(psum(&[sq(&ld), pscale(&x1, -2), pneg(&xd)])),
                Relation::modp(psum(&[pmul(&ld, &psub(&x1, &xd)), pneg(&y1), pneg(&yd)])),
            ];
            rels.extend(complete_add_relations(
                &le,
                (&xd, &yd, acc_inf),
                (&xq, &yq, &xq_sq),
                (&xe, &ye, Some(e_inf)),
            ));
            rels.extend(complete_add_relations(
                &lf,
                (&xe, &ye, e_inf),
                (&xg, &yg, &xg_sq),
                (&x5, &y5, Some(out_inf)),
            ));
            rels
        }
    }
}
