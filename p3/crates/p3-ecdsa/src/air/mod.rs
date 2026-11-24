//! The secp256k1 ECDSA batch-verification AIR. See [`layout`] for the row and
//! column schedule and [`relations`] for the per-row non-native relations.

pub mod hash;
pub mod layout;
pub mod limbs;
pub mod lookups;
pub mod relations;
pub mod trace;

use p3_air::{Air, AirBuilder, BaseAir, ExtensionBuilder, PermutationAirBuilder, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use self::hash::{
    block_lane, instance_fields, link_fields, poseidon_round0, round_kind_selectors, sbox_round,
};
use self::layout::*;
use self::limbs::{
    Fx, Modulus, carry_from_halves, group_constraints, konst, modulus_limbs, pmul, psub,
};
use self::lookups::{reg_lookups, row_lookups};
use self::relations::{ROW_KINDS, const_limbs, relations};
use crate::digest::{LINK_BLOCKS, LINK_HASH_DOMAIN, PUBLIC_DIGEST_WIDTH, PUBLIC_HASH_DOMAIN};
use crate::stark::LogUpAir;

#[derive(Clone, Debug)]
pub struct EcdsaBatchAir {
    pub capacity: usize,
}

impl EcdsaBatchAir {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
    pub fn log_height(&self) -> usize {
        log_height_for_capacity(self.capacity)
    }
}

impl<F: Field> BaseAir<F> for EcdsaBatchAir {
    fn width(&self) -> usize {
        MAIN_WIDTH
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let gl = trace::generate_preprocessed(self.capacity);
        Some(RowMajorMatrix::new(
            gl.values
                .iter()
                .map(|v| F::from_u64(p3_field::PrimeField64::as_canonical_u64(v)))
                .collect(),
            PREP_WIDTH,
        ))
    }
    fn preprocessed_width(&self) -> usize {
        PREP_WIDTH
    }
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(MAX_CONSTRAINT_DEGREE)
    }
    fn num_public_values(&self) -> usize {
        NUM_PUBLIC_VALUES
    }
}

impl<AB: PermutationAirBuilder> Air<AB> for EcdsaBatchAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: Vec<AB::Expr> = main.current_slice().iter().map(|v| (*v).into()).collect();
        let next: Vec<AB::Expr> = main.next_slice().iter().map(|v| (*v).into()).collect();
        let prep = builder.preprocessed().clone();
        let pl: Vec<AB::Expr> = prep.current_slice().iter().map(|v| (*v).into()).collect();
        let pn: Vec<AB::Expr> = prep.next_slice().iter().map(|v| (*v).into()).collect();
        let public: Vec<AB::Expr> = builder
            .public_values()
            .iter()
            .map(|v| (*v).into())
            .collect();
        let one = AB::Expr::ONE;

        // ---- non-native relation slots ----
        let lx: Vec<Fx<AB::Expr>> = local.iter().cloned().map(Fx).collect();
        let nx: Vec<Fx<AB::Expr>> = next.iter().cloned().map(Fx).collect();
        let mut combined: Vec<Option<Vec<Fx<AB::Expr>>>> = (0..SLOTS).map(|_| None).collect();
        let mut used: Vec<Fx<AB::Expr>> = vec![Fx(AB::Expr::ZERO); SLOTS];
        let mut by_mod: Vec<[Fx<AB::Expr>; 2]> = (0..SLOTS)
            .map(|_| [Fx(AB::Expr::ZERO), Fx(AB::Expr::ZERO)])
            .collect();
        let ec_last = Fx(pl[P_EC_LAST].clone());
        for kind in ROW_KINDS {
            let sel = Fx(pl[kind.selector_col()].clone());
            for (s, rel) in relations(kind, &lx, &nx, &ec_last).into_iter().enumerate() {
                let mut coeffs: Vec<Fx<AB::Expr>> =
                    rel.coeffs.into_iter().map(|c| c * sel.clone()).collect();
                coeffs.resize_with(COEFFS, || Fx(AB::Expr::ZERO));
                combined[s] = Some(match combined[s].take() {
                    None => coeffs,
                    Some(prev) => prev.into_iter().zip(coeffs).map(|(a, b)| a + b).collect(),
                });
                used[s] = used[s].clone() + sel.clone();
                match rel.modulus {
                    Modulus::P => by_mod[s][0] = by_mod[s][0].clone() + sel.clone(),
                    Modulus::N => by_mod[s][1] = by_mod[s][1].clone() + sel.clone(),
                    Modulus::None => {}
                }
            }
        }
        for s in 0..SLOTS {
            let Some(mut res) = combined[s].take() else {
                continue;
            };
            let q: Vec<Fx<AB::Expr>> = lx[slot_q(s)..slot_q(s) + Q_LIMBS].to_vec();
            for (m, modulus) in [Modulus::P, Modulus::N].into_iter().enumerate() {
                let ml = modulus_limbs(modulus).unwrap();
                let mut term = psub(&konst::<Fx<AB::Expr>>(&ml.offset), &pmul(&q, &konst(&ml.m)));
                term.resize_with(COEFFS, || Fx(AB::Expr::ZERO));
                res = res
                    .into_iter()
                    .zip(term)
                    .map(|(a, t)| a + t * by_mod[s][m].clone())
                    .collect();
            }
            let carries: Vec<Fx<AB::Expr>> = (0..CARRIES)
                .map(|g| {
                    carry_from_halves(
                        lx[slot_carry_lo(s) + g].clone(),
                        lx[slot_carry_hi(s) + g].clone(),
                    ) * used[s].clone()
                })
                .collect();
            for c in group_constraints(&res, &carries) {
                builder.assert_zero(c.0);
            }
        }

        // ---- scalar loop bookkeeping ----
        let k = const_limbs();
        let ec_first = pl[P_EC_FIRST].clone();
        for i in 0..LIMBS {
            builder.assert_zero(
                ec_first.clone() * (local[ACCX + i].clone() - AB::Expr::from_u64(k.offset.0[i])),
            );
            builder.assert_zero(
                ec_first.clone() * (local[ACCY + i].clone() - AB::Expr::from_u64(k.offset.1[i])),
            );
        }
        // Point-at-infinity flags of the complete additions: the loop starts from the (finite)
        // offset, a row's result flag is the next accumulator's (the `RP_INF` register on the
        // last row), and an addition whose first operand is at infinity returns the (finite)
        // table point.
        let ec = pl[P_EC].clone();
        let (acc_inf, e_inf, out_inf) = (
            local[ACC_INF].clone(),
            local[ADD_Q_INF].clone(),
            local[ADD_G_INF].clone(),
        );
        for f in [&acc_inf, &e_inf, &out_inf] {
            builder.assert_zero(ec.clone() * f.clone() * (f.clone() - one.clone()));
        }
        builder.assert_zero(ec_first.clone() * acc_inf.clone());
        builder.assert_zero(ec.clone() * acc_inf * e_inf.clone());
        builder.assert_zero(ec.clone() * e_inf * out_inf.clone());
        let next_inf = next[ACC_INF].clone()
            + pl[P_EC_LAST].clone() * (local[RP_INF].clone() - next[ACC_INF].clone());
        builder
            .when_transition()
            .assert_zero(ec.clone() * (out_inf - next_inf));
        for (m, j) in [(DIGIT_PROD, 0), (DIGIT_PROD + 1, 2)] {
            builder.assert_zero(
                ec.clone()
                    * (local[m].clone() - local[BITS + j].clone() * local[BITS + j + 1].clone()),
            );
        }
        for j in 0..NUM_SCALARS {
            let b = local[BITS + j].clone();
            builder.assert_zero(pl[P_EC].clone() * b.clone() * (b - one.clone()));
            let step = next[CHUNK_ACC + j].clone()
                - next[BITS + j].clone()
                - (one.clone() - pn[P_CHUNK_START].clone()) * local[CHUNK_ACC + j].clone().double();
            builder
                .when_transition()
                .assert_zero(pn[P_EC].clone() * step);
            let mut end = AB::Expr::ZERO;
            for c in 0..DIGIT_LIMBS {
                end += pl[P_CHUNK_END + c].clone()
                    * (local[CHUNK_ACC + j].clone() - local[digit_limbs(j) + c].clone());
            }
            builder.assert_zero(end);
        }

        // ---- per-signature registers ----
        let same_next = pl[P_SAME_NEXT].clone();
        for c in 0..REG_END {
            builder
                .when_transition()
                .assert_zero(same_next.clone() * (next[c].clone() - local[c].clone()));
        }
        let start = pl[P_SIG_START].clone();
        for c in [R_WRAP, Y_PARITY, RP_INF] {
            builder
                .assert_zero(start.clone() * local[c].clone() * (local[c].clone() - one.clone()));
        }
        builder.assert_zero(
            start.clone()
                * (local[QY].clone() - local[Y_HALF].clone().double() - local[Y_PARITY].clone()),
        );

        // ---- activity and counter ----
        let active = local[ACTIVE].clone();
        builder.assert_zero(start.clone() * active.clone() * (active.clone() - one.clone()));
        builder
            .when_transition()
            .assert_zero(same_next.clone() * (next[ACTIVE].clone() - active.clone()));
        builder.when_transition().assert_zero(
            (one.clone() - same_next) * next[ACTIVE].clone() * (one.clone() - active.clone()),
        );
        builder
            .when_first_row()
            .assert_zero(local[COUNTER].clone() - active.clone());
        builder.when_transition().assert_zero(
            next[COUNTER].clone()
                - local[COUNTER].clone()
                - pn[P_SIG_START].clone() * next[ACTIVE].clone(),
        );
        builder
            .when_last_row()
            .assert_zero(local[COUNTER].clone() - public[0].clone());

        // ---- public digest sponge ----
        let state: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|l| local[HASH_STATE + l].clone());
        for l in 0..POSEIDON_WIDTH {
            let init = match l {
                0 => public[0].clone(),
                1 => AB::Expr::from_u64(PUBLIC_HASH_DOMAIN),
                _ => AB::Expr::ZERO,
            };
            builder
                .when_first_row()
                .assert_zero(state[l].clone() - init);
        }
        let fields = instance_fields(&local);
        let h0: AB::Expr = (0..HASH_BLOCKS).map(|b| pl[P_BLOCK_SEL + b].clone()).sum();
        let mut absorbed: [AB::Expr; POSEIDON_WIDTH] = core::array::from_fn(|l| {
            if l < POSEIDON_RATE {
                (0..HASH_BLOCKS)
                    .map(|b| pl[P_BLOCK_SEL + b].clone() * block_lane(&fields, b, l))
                    .sum()
            } else {
                h0.clone() * state[l].clone()
            }
        });
        poseidon_round0(&mut absorbed);
        let (full_sel, partial_sel) = round_kind_selectors(&pl);
        let rc: [AB::Expr; POSEIDON_WIDTH] = core::array::from_fn(|l| pl[P_RC + l].clone());
        let h = h0 + full_sel.clone() + partial_sel.clone();
        let cube: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|l| local[HASH_SBOX + l].clone());
        for l in 0..POSEIDON_WIDTH {
            builder.assert_zero(cube[l].clone() - (state[l].clone() + rc[l].clone()).cube());
        }
        let rounded = sbox_round(&full_sel, &partial_sel, &state, &rc, &cube);
        let stepped: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|l| absorbed[l].clone() + rounded[l].clone());
        for l in 0..POSEIDON_WIDTH {
            let expect = active.clone() * stepped[l].clone()
                + (one.clone() - active.clone() * h.clone()) * state[l].clone();
            builder
                .when_transition()
                .assert_zero(next[HASH_STATE + l].clone() - expect);
        }
        for l in 0..PUBLIC_DIGEST_WIDTH {
            builder
                .when_last_row()
                .assert_zero(state[l].clone() - public[1 + l].clone());
        }

        // ---- link digest sponge: (tx_hash, keccak output) per signature ----
        let gate = local[LINK_GATE].clone();
        builder.assert_zero(gate.clone() - active.clone() * pl[P_LINK].clone());
        let lstate: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|l| local[LINK_STATE + l].clone());
        for l in 0..POSEIDON_WIDTH {
            let init = match l {
                0 => public[0].clone(),
                1 => AB::Expr::from_u64(LINK_HASH_DOMAIN),
                _ => AB::Expr::ZERO,
            };
            builder
                .when_first_row()
                .assert_zero(lstate[l].clone() - init);
        }
        let lfields = link_fields(&local);
        let lh0: AB::Expr = (0..LINK_BLOCKS).map(|b| pl[P_BLOCK_SEL + b].clone()).sum();
        let mut labsorbed: [AB::Expr; POSEIDON_WIDTH] = core::array::from_fn(|l| {
            if l < POSEIDON_RATE {
                (0..LINK_BLOCKS)
                    .map(|b| pl[P_BLOCK_SEL + b].clone() * block_lane(&lfields, b, l))
                    .sum()
            } else {
                lh0.clone() * lstate[l].clone()
            }
        });
        poseidon_round0(&mut labsorbed);
        let lcube: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|l| local[LINK_SBOX + l].clone());
        for l in 0..POSEIDON_WIDTH {
            builder.assert_zero(lcube[l].clone() - (lstate[l].clone() + rc[l].clone()).cube());
        }
        let lrounded = sbox_round(&full_sel, &partial_sel, &lstate, &rc, &lcube);
        let lstepped: [AB::Expr; POSEIDON_WIDTH] =
            core::array::from_fn(|l| labsorbed[l].clone() + lrounded[l].clone());
        for l in 0..POSEIDON_WIDTH {
            let expect = gate.clone() * lstepped[l].clone()
                + (one.clone() - gate.clone()) * lstate[l].clone();
            builder
                .when_transition()
                .assert_zero(next[LINK_STATE + l].clone() - expect);
        }
        for l in 0..PUBLIC_DIGEST_WIDTH {
            builder
                .when_last_row()
                .assert_zero(lstate[l].clone() - public[1 + PUBLIC_DIGEST_WIDTH + l].clone());
        }

        // ---- LogUp range checks ----
        let perm = builder.permutation();
        let pv: Vec<AB::ExprEF> = perm.current_slice().iter().map(|v| (*v).into()).collect();
        let pnext: Vec<AB::ExprEF> = perm.next_slice().iter().map(|v| (*v).into()).collect();
        let alpha: AB::ExprEF = builder.permutation_randomness()[0].into();
        let value = |&(c, scale): &(usize, u64)| local[c].clone() * AB::Expr::from_u64(scale);
        let mut row_sum = AB::ExprEF::ZERO;
        let mut helper = 0;
        for (lookups, numerator) in [(row_lookups(), None), (reg_lookups(), Some(start.clone()))] {
            for group in lookups.chunks(LOOKUPS_PER_HELPER) {
                let fps: Vec<AB::ExprEF> =
                    group.iter().map(|lk| alpha.clone() - value(lk)).collect();
                let mut prod = AB::ExprEF::ONE;
                for f in &fps {
                    prod *= f.clone();
                }
                let mut sum = AB::ExprEF::ZERO;
                for i in 0..fps.len() {
                    let mut t = AB::ExprEF::ONE;
                    for (j, f) in fps.iter().enumerate() {
                        if i != j {
                            t *= f.clone();
                        }
                    }
                    sum += t;
                }
                if let Some(n) = &numerator {
                    sum *= n.clone();
                }
                builder.assert_zero_ext(pv[helper].clone() * prod - sum);
                row_sum += pv[helper].clone();
                helper += 1;
            }
        }
        debug_assert_eq!(helper, PERM_TABLE);
        let table_fp = alpha - pl[P_RANGE].clone();
        builder.assert_zero_ext(pv[PERM_TABLE].clone() * table_fp - local[RANGE_MULT].clone());
        row_sum -= pv[PERM_TABLE].clone();
        let acc = pv[PERM_ACC].clone();
        builder.when_first_row().assert_zero_ext(acc.clone());
        builder
            .when_transition()
            .assert_zero_ext(pnext[PERM_ACC].clone() - acc.clone() - row_sum.clone());
        builder.when_last_row().assert_zero_ext(acc + row_sum);
    }
}

impl<F: Field, EF: ExtensionField<F>> LogUpAir<F, EF> for EcdsaBatchAir {
    fn permutation_width(&self) -> usize {
        PERM_WIDTH
    }
    fn num_permutation_challenges(&self) -> usize {
        1
    }
    fn generate_permutation_trace(
        &self,
        main: &RowMajorMatrix<F>,
        preprocessed: Option<&RowMajorMatrix<F>>,
        challenges: &[EF],
    ) -> RowMajorMatrix<EF> {
        let prep = preprocessed.expect("ECDSA AIR has a preprocessed trace");
        let alpha = challenges[0];
        let row_lk = row_lookups();
        let reg_lk = reg_lookups();
        let height = main.height();
        let mut values = vec![EF::ZERO; height * PERM_WIDTH];
        // Register lookups only count on block-start rows (their helpers are
        // `start * sum`), so other rows skip those inversions.
        values.par_chunks_mut(PERM_WIDTH).enumerate().for_each_init(
            || Vec::with_capacity(ROW_LOOKUPS + REG_LOOKUPS + 1),
            |fps: &mut Vec<EF>, (r, out)| {
                let row = &main.values[r * MAIN_WIDTH..(r + 1) * MAIN_WIDTH];
                let prow = &prep.values[r * PREP_WIDTH..(r + 1) * PREP_WIDTH];
                let start = prow[P_SIG_START];
                let with_reg = !start.is_zero();
                fps.clear();
                fps.push(alpha - prow[P_RANGE]);
                fps.extend(
                    row_lk
                        .iter()
                        .map(|&(c, scale)| alpha - row[c] * F::from_u64(scale)),
                );
                if with_reg {
                    fps.extend(
                        reg_lk
                            .iter()
                            .map(|&(c, scale)| alpha - row[c] * F::from_u64(scale)),
                    );
                }
                let inv = batch_multiplicative_inverse(fps);
                out[PERM_TABLE] = inv[0] * row[RANGE_MULT];
                let mut h = 0;
                for chunk in inv[1..1 + ROW_LOOKUPS].chunks(LOOKUPS_PER_HELPER) {
                    out[h] = chunk.iter().copied().sum();
                    h += 1;
                }
                if with_reg {
                    for chunk in inv[1 + ROW_LOOKUPS..].chunks(LOOKUPS_PER_HELPER) {
                        out[h] = chunk.iter().copied().sum::<EF>() * start;
                        h += 1;
                    }
                }
            },
        );
        let mut acc = EF::ZERO;
        for r in 0..height {
            let out = &mut values[r * PERM_WIDTH..(r + 1) * PERM_WIDTH];
            let row_sum: EF = out[..PERM_TABLE].iter().copied().sum::<EF>() - out[PERM_TABLE];
            out[PERM_ACC] = acc;
            acc += row_sum;
        }
        assert_eq!(acc, EF::ZERO, "range-check bus is not balanced");
        RowMajorMatrix::new(values, PERM_WIDTH)
    }
}
