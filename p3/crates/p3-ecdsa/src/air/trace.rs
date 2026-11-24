//! Witness generation for the main and preprocessed traces.

use num_bigint::BigUint;
use num_traits::Zero;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use super::hash::{instance_fields, link_fields, round_constants, sponge_step};
use super::layout::*;
use super::limbs::{Int, solve_relation};
use super::lookups::{reg_lookups, row_lookups};
use super::relations::{RowKind, relations};
use crate::digest::{initial_link_hash_state, initial_public_hash_state};
use crate::secp::{
    self, Point, constants, digit_integer, glv_decompose, signed_digit_bits, to_limbs,
};
use crate::types::EcdsaInstance;

type F = Goldilocks;

fn put(row: &mut [u64], start: usize, v: &BigUint) {
    row[start..start + LIMBS].copy_from_slice(&to_limbs(v, LIMBS));
}

fn fp(v: &BigUint) -> BigUint {
    v % &constants().p
}

fn sub_p(a: &BigUint, b: &BigUint) -> BigUint {
    let p = &constants().p;
    (a % p + p - b % p) % p
}

fn div_p(a: &BigUint, b: &BigUint) -> BigUint {
    let p = &constants().p;
    (a * secp::fp_inv(p, b)) % p
}

/// Slope of the tangent at a finite curve point (`y != 0` on secp256k1).
fn tangent_slope(a: &Point) -> BigUint {
    div_p(&(&a.x * &a.x * 3u32), &(&a.y * 2u32))
}

/// The point `a + b` on the line of slope `lambda` through `a`.
fn line_sum(lambda: &BigUint, a: &Point, b: &Point) -> Point {
    let x = sub_p(&sub_p(&(lambda * lambda), &a.x), &b.x);
    let y = sub_p(&(lambda * sub_p(&a.x, &x)), &a.y);
    Point { x, y }
}

/// The complete addition `a + b` (`b` finite) and its slope witness: the chord or tangent
/// slope, and 0 when `a` is the point at infinity or `a = -b` (the slope is then unused).
fn add_step(a: Option<&Point>, b: &Point) -> (BigUint, Option<Point>) {
    let lambda = match a {
        None => return (BigUint::zero(), Some(b.clone())),
        Some(a) if a.x != b.x => div_p(&sub_p(&b.y, &a.y), &sub_p(&b.x, &a.x)),
        Some(a) if a.y == b.y => tangent_slope(a),
        Some(_) => return (BigUint::zero(), None),
    };
    let sum = line_sum(&lambda, a.unwrap(), b);
    (lambda, Some(sum))
}

/// A fixed valid signature used to fill unused capacity slots.
pub fn dummy_instance() -> EcdsaInstance {
    use std::sync::OnceLock;
    static D: OnceLock<EcdsaInstance> = OnceLock::new();
    D.get_or_init(|| {
        let c = constants();
        let sk = BigUint::from(0x1d2e_3f40_5162_7384u64);
        let q = secp::mul(&c.g, &sk).unwrap();
        let tx_hash = [F::ZERO; 5];
        let z =
            EcdsaInstance::from_parts(tx_hash, &BigUint::from(1u32), &BigUint::from(1u32), &c.g)
                .message_scalar();
        let (r, s) = secp::ecdsa_sign(&sk, &z, &BigUint::from(0x9a8b_7c6d_5e4f_3021u64)).unwrap();
        EcdsaInstance::from_parts(tx_hash, &r, &s, &q)
    })
    .clone()
}

/// Fills one signature block (`ROWS_PER_SIGNATURE` rows of `MAIN_WIDTH`),
/// except hash state, activity, counter and range multiplicities.
fn fill_block(block: &mut [u64], inst: &EcdsaInstance, z: BigUint) {
    let c = constants();
    let q = inst.public_key_point().expect("validated public key");
    let (r, s) = (inst.r_scalar(), inst.s_scalar());
    let w = secp::mod_inv(&s, &c.n);
    let u1 = (&z * &w) % &c.n;
    let u2 = (&r * &w) % &c.n;
    let big_r = secp::add(secp::mul(&c.g, &u1).as_ref(), secp::mul(&q, &u2).as_ref()).unwrap();
    let phi_q = secp::endo(&q);
    let tq1 = secp::add(Some(&q), Some(&phi_q)).unwrap();
    let r_shift = secp::add(Some(&big_r), Some(&c.offset_shift));
    let tq2 = secp::add(Some(&q), Some(&secp::neg(&phi_q))).unwrap();
    let scalars = {
        let (k1, k2) = glv_decompose(&u2);
        let (k3, k4) = glv_decompose(&u1);
        [k1, k2, k3, k4]
    };
    let bits: Vec<[bool; GLV_BITS]> = scalars.iter().map(signed_digit_bits).collect();

    // Registers.
    let mut reg = vec![0u64; REG_END];
    put(&mut reg, XPHI, &phi_q.x);
    put(&mut reg, TQ2X, &tq2.x);
    put(&mut reg, TQ2Y, &tq2.y);
    put(&mut reg, QX, &q.x);
    put(&mut reg, QY, &q.y);
    if let Some(rp) = &r_shift {
        put(&mut reg, RPX, &rp.x);
        put(&mut reg, RPY, &rp.y);
    }
    reg[RP_INF] = u64::from(r_shift.is_none());
    put(&mut reg, SIG_R, &r);
    put(&mut reg, SIG_S, &s);
    for (m, limb) in to_limbs(&z, LIMBS).into_iter().enumerate() {
        reg[ZLO + m] = limb & 0xff;
        reg[ZHI + m] = limb >> 8;
    }
    for (j, k) in scalars.iter().enumerate() {
        let b = to_limbs(&digit_integer(k), DIGIT_LIMBS);
        reg[digit_limbs(j)..digit_limbs(j) + DIGIT_LIMBS].copy_from_slice(&b);
    }
    for (i, h) in inst.tx_hash.iter().enumerate() {
        reg[TX_HASH + i] = h.as_canonical_u64();
    }
    reg[R_WRAP] = u64::from(big_r.x >= c.n);
    reg[Y_PARITY] = u64::from(q.y.bit(0));
    reg[Y_HALF] = reg[QY] >> 1;
    for row in block.chunks_mut(MAIN_WIDTH) {
        row[..REG_END].copy_from_slice(&reg);
    }

    let row = |block: &mut [u64], o: usize| -> std::ops::Range<usize> {
        let _ = block;
        o * MAIN_WIDTH..(o + 1) * MAIN_WIDTH
    };
    let set_work = |block: &mut [u64], o: usize, i: usize, v: &BigUint| {
        let range = row(block, o);
        put(&mut block[range], work(i), v);
    };

    // Setup rows.
    let lam_t2 = div_p(
        &sub_p(&BigUint::zero(), &(&q.y * 2u32)),
        &sub_p(&phi_q.x, &q.x),
    );
    let (lam_shift, _) = add_step(r_shift.as_ref(), &secp::neg(&c.offset_shift));
    set_work(block, 0, 0, &lam_t2);
    set_work(block, 0, 1, &fp(&(&q.x * &q.x)));
    set_work(block, 0, 2, &(&r - 1u32));
    set_work(block, 0, 3, &(&c.n - 1u32 - &r));
    set_work(block, 0, 4, &(&c.p - 1u32 - &q.x));
    set_work(block, 0, 5, &(&c.p - 1u32 - &q.y));
    set_work(block, 1, 0, &lam_shift);
    set_work(block, 1, 1, &((&s * &c.lambda) % &c.n));
    set_work(block, 1, 2, &(&c.p - 1u32 - &big_r.x));
    set_work(block, 1, 3, &(&s - 1u32));
    set_work(block, 1, 4, &(((&c.n - 1u32) >> 1) - &s));
    let range = row(block, 1);
    put(&mut block[range.clone()], ACCX, &big_r.x);
    put(&mut block[range], ACCY, &big_r.y);

    // Scalar loop.
    let g1 = secp::add(Some(&c.g), Some(&c.phi_g)).unwrap();
    let g2 = secp::add(Some(&c.g), Some(&secp::neg(&c.phi_g))).unwrap();
    let mut acc = Some(c.offset.clone());
    let mut chunk = [0u64; NUM_SCALARS];
    let zero = BigUint::zero();
    // Coordinates written for a point at infinity (unused by the relations).
    let xy = |p: &Option<Point>| match p {
        Some(p) => (p.x.clone(), p.y.clone()),
        None => (zero.clone(), zero.clone()),
    };
    for i in 0..GLV_BITS {
        let o = EC_FIRST_ROW + i;
        let pos = GLV_BITS - 1 - i;
        let b: [bool; NUM_SCALARS] = core::array::from_fn(|j| bits[j][i]);
        let pick = |ba: bool, bb: bool, t1: &Point, t2: &Point| {
            let t = if ba == bb { t1 } else { t2 };
            if ba { t.clone() } else { secp::neg(t) }
        };
        let pq = pick(b[0], b[1], &tq1, &tq2);
        let pg = pick(b[2], b[3], &g1, &g2);
        let (ld, d) = match &acc {
            Some(a) => {
                let ld = tangent_slope(a);
                let d = line_sum(&ld, a, a);
                (ld, Some(d))
            }
            None => (BigUint::zero(), None),
        };
        let (le, e) = add_step(d.as_ref(), &pq);
        let (lf, out) = add_step(e.as_ref(), &pg);
        let range = row(block, o);
        let r_ = &mut block[range];
        let ((ax, ay), (dx, dy), (ex, ey)) = (xy(&acc), xy(&d), xy(&e));
        put(r_, ACCX, &ax);
        put(r_, ACCY, &ay);
        for (idx, v) in [&ld, &dx, &dy, &le, &ex, &ey, &lf].into_iter().enumerate() {
            put(r_, work(idx), v);
        }
        r_[ACC_INF] = u64::from(acc.is_none());
        r_[ADD_Q_INF] = u64::from(e.is_none());
        r_[ADD_G_INF] = u64::from(out.is_none());
        r_[DIGIT_PROD] = u64::from(b[0] && b[1]);
        r_[DIGIT_PROD + 1] = u64::from(b[2] && b[3]);
        let chunk_start = pos == GLV_BITS - 1 || pos % 16 == 15;
        for j in 0..NUM_SCALARS {
            chunk[j] = if chunk_start { 0 } else { 2 * chunk[j] } + u64::from(b[j]);
            r_[BITS + j] = u64::from(b[j]);
            r_[CHUNK_ACC + j] = chunk[j];
        }
        acc = out;
    }
    debug_assert_eq!(acc, r_shift);

    // Relation slots.
    let ints: Vec<Int> = block.iter().map(|&v| Int(v as i128)).collect();
    for o in 0..ROWS_PER_SIGNATURE {
        let kind = RowKind::at_offset(o);
        let local = &ints[o * MAIN_WIDTH..(o + 1) * MAIN_WIDTH];
        let next = if o + 1 < ROWS_PER_SIGNATURE {
            &ints[(o + 1) * MAIN_WIDTH..(o + 2) * MAIN_WIDTH]
        } else {
            local
        };
        let ec_last = Int(i128::from(o == EC_LAST_ROW));
        let rels = relations(kind, local, next, &ec_last);
        let r_ = &mut block[o * MAIN_WIDTH..(o + 1) * MAIN_WIDTH];
        for (s, rel) in rels.iter().enumerate() {
            let (qv, lo, hi) = solve_relation(rel);
            r_[slot_q(s)..slot_q(s) + Q_LIMBS].copy_from_slice(&qv);
            r_[slot_carry_lo(s)..slot_carry_lo(s) + CARRIES].copy_from_slice(&lo);
            r_[slot_carry_hi(s)..slot_carry_hi(s) + CARRIES].copy_from_slice(&hi);
        }
    }
}

/// Builds the main trace for `instances` padded to `capacity` signatures.
pub fn generate_main_trace(instances: &[EcdsaInstance], capacity: usize) -> RowMajorMatrix<F> {
    let height = 1usize << log_height_for_capacity(capacity);
    let mut values = vec![0u64; height * MAIN_WIDTH];
    let dummy = dummy_instance();
    let block_len = ROWS_PER_SIGNATURE * MAIN_WIDTH;
    values[..capacity * block_len]
        .par_chunks_mut(block_len)
        .enumerate()
        .for_each(|(i, block)| {
            let inst = instances.get(i).unwrap_or(&dummy);
            fill_block(block, inst, inst.message_scalar())
        });

    // Hash sponge, activity and counter (sequential).
    let count = instances.len();
    let mut state = initial_public_hash_state(count);
    let mut lstate = initial_link_hash_state(count);
    let mut counter = 0u64;
    let mut fields: Vec<F> = Vec::new();
    let mut lfields: Vec<F> = Vec::new();
    for row_idx in 0..height {
        let (blk, o) = (row_idx / ROWS_PER_SIGNATURE, row_idx % ROWS_PER_SIGNATURE);
        let active = blk < count;
        let row = &mut values[row_idx * MAIN_WIDTH..(row_idx + 1) * MAIN_WIDTH];
        if blk < capacity && o == 0 {
            counter += u64::from(active);
            let as_f: Vec<F> = row[..REG_END].iter().map(|&v| F::from_u64(v)).collect();
            fields = instance_fields(&as_f);
            lfields = link_fields(&as_f);
        }
        row[ACTIVE] = u64::from(active);
        row[COUNTER] = counter;
        for (l, v) in state.iter().enumerate() {
            row[HASH_STATE + l] = v.as_canonical_u64();
        }
        for (l, v) in lstate.iter().enumerate() {
            row[LINK_STATE + l] = v.as_canonical_u64();
        }
        let rc = row_round_constants(blk, o, capacity);
        for l in 0..POSEIDON_WIDTH {
            let rc = F::from_u64(rc[l]);
            row[HASH_SBOX + l] = (state[l] + rc).cube().as_canonical_u64();
            row[LINK_SBOX + l] = (lstate[l] + rc).cube().as_canonical_u64();
        }
        let link = active && o < LINK_ROWS_PER_SIGNATURE;
        row[LINK_GATE] = u64::from(link);
        if link {
            let (b, round) = (o / POSEIDON_ROUNDS, o % POSEIDON_ROUNDS);
            sponge_step(&mut lstate, &lfields, b, round);
        }
        if active && o < HASH_ROWS_PER_SIGNATURE {
            let (b, round) = (o / POSEIDON_ROUNDS, o % POSEIDON_ROUNDS);
            sponge_step(&mut state, &fields, b, round);
        }
    }

    // Range multiplicities.
    let row_lk = row_lookups();
    let reg_lk = reg_lookups();
    let counts: Vec<u64> = values
        .par_chunks(MAIN_WIDTH)
        .enumerate()
        .fold(
            || vec![0u64; 1 << RANGE_BITS],
            |mut acc, (row_idx, row)| {
                for &(c, scale) in &row_lk {
                    acc[(row[c] * scale) as usize] += 1;
                }
                let o = row_idx % ROWS_PER_SIGNATURE;
                if row_idx / ROWS_PER_SIGNATURE < capacity && o == 0 {
                    for &(c, scale) in &reg_lk {
                        acc[(row[c] * scale) as usize] += 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0u64; 1 << RANGE_BITS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    for (v, m) in counts.into_iter().enumerate() {
        values[v * MAIN_WIDTH + RANGE_MULT] = m;
    }

    RowMajorMatrix::new(
        values.into_par_iter().map(F::from_u64).collect(),
        MAIN_WIDTH,
    )
}

/// The `P_RC` values of the row at offset `o` of signature block `blk`.
fn row_round_constants(blk: usize, o: usize, capacity: usize) -> [u64; POSEIDON_WIDTH] {
    let round = o % POSEIDON_ROUNDS;
    if blk < capacity && o < HASH_ROWS_PER_SIGNATURE && round != 0 {
        round_constants(round)
    } else {
        [0; POSEIDON_WIDTH]
    }
}

pub fn generate_preprocessed(capacity: usize) -> RowMajorMatrix<F> {
    let height = 1usize << log_height_for_capacity(capacity);
    let mut values = vec![F::ZERO; height * PREP_WIDTH];
    for (row_idx, row) in values.chunks_mut(PREP_WIDTH).enumerate() {
        let (blk, o) = (row_idx / ROWS_PER_SIGNATURE, row_idx % ROWS_PER_SIGNATURE);
        if row_idx < 1 << RANGE_BITS {
            row[P_RANGE] = F::from_usize(row_idx);
        }
        if blk >= capacity {
            row[P_SAME_NEXT] = F::from_bool(row_idx + 1 < height);
            continue;
        }
        if o < HASH_ROWS_PER_SIGNATURE {
            let (b, round) = (o / POSEIDON_ROUNDS, o % POSEIDON_ROUNDS);
            let c = if round == 0 {
                P_BLOCK_SEL + b
            } else {
                P_ROUND_SEL + round - 1
            };
            row[c] = F::ONE;
        }
        for (l, rc) in row_round_constants(blk, o, capacity)
            .into_iter()
            .enumerate()
        {
            row[P_RC + l] = F::from_u64(rc);
        }
        row[P_SIG_START] = F::from_bool(o == 0);
        row[P_LINK] = F::from_bool(o < LINK_ROWS_PER_SIGNATURE);
        row[P_SAME_NEXT] = F::from_bool(o + 1 < ROWS_PER_SIGNATURE && row_idx + 1 < height);
        row[RowKind::at_offset(o).selector_col()] = F::ONE;
        if o >= EC_FIRST_ROW {
            let pos = GLV_BITS - 1 - (o - EC_FIRST_ROW);
            row[P_EC_FIRST] = F::from_bool(o == EC_FIRST_ROW);
            row[P_EC_LAST] = F::from_bool(o == EC_LAST_ROW);
            row[P_CHUNK_START] = F::from_bool(pos == GLV_BITS - 1 || pos % 16 == 15);
            if pos.is_multiple_of(16) {
                row[P_CHUNK_END + pos / 16] = F::ONE;
            }
        }
    }
    RowMajorMatrix::new(values, PREP_WIDTH)
}
