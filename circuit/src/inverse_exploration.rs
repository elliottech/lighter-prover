// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Test-only exploration: candidate replacements for the Goldilocks
//! `try_inverse` 72-mul Fermat chain, differential-tested against it and
//! timed head-to-head. Not compiled into the prover. Conclusion (see the
//! standalone note "Goldilocks Fermat inverse already near-optimal"): the
//! Fermat chain wins — kept here as reproducible evidence.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};

const P: u64 = 0xffff_ffff_0000_0001;

/// Variable-time binary extended GCD, tracking coefficients mod P.
fn binary_xgcd_inverse(x: u64) -> u64 {
    debug_assert!(x != 0 && x < P);
    let mut u = x;
    let mut v = P;
    let mut b: u64 = 1;
    let mut c: u64 = 0;
    while u != 1 && v != 1 {
        while u & 1 == 0 {
            u >>= 1;
            b = if b & 1 == 0 {
                b >> 1
            } else {
                ((b as u128 + P as u128) >> 1) as u64
            };
        }
        while v & 1 == 0 {
            v >>= 1;
            c = if c & 1 == 0 {
                c >> 1
            } else {
                ((c as u128 + P as u128) >> 1) as u64
            };
        }
        if u >= v {
            u -= v;
            b = if b >= c {
                b - c
            } else {
                b.wrapping_add(P.wrapping_sub(c))
            };
        } else {
            v -= u;
            c = if c >= b {
                c - b
            } else {
                c.wrapping_add(P.wrapping_sub(b))
            };
        }
    }
    if u == 1 { b } else { c }
}

fn fermat_inverse(x: u64) -> u64 {
    GoldilocksField::from_canonical_u64(x)
        .inverse()
        .to_canonical_u64()
}

fn edge_cases() -> Vec<u64> {
    let mut v = vec![];
    v.extend(1..1000u64);
    v.extend((P - 1000)..P);
    for k in 0..64 {
        v.push(1u64 << k);
        if (1u64 << k) < P - 1 {
            v.push(P - 1 - (1u64 << k));
        }
    }
    v.push(0xffff_ffff);
    v.push(0x1_0000_0000);
    v.push(0xffff_fffe_ffff_ffff % P);
    v.retain(|&x| x != 0 && x < P);
    v
}

#[test]
fn binary_xgcd_matches_fermat() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    for x in edge_cases() {
        assert_eq!(binary_xgcd_inverse(x), fermat_inverse(x), "xgcd edge {x}");
    }
    let mut rng = StdRng::seed_from_u64(0x1257);
    for _ in 0..2_000_000 {
        let x = rng.gen_range(1..P);
        let inv = binary_xgcd_inverse(x);
        assert_eq!(inv, fermat_inverse(x), "xgcd random {x}");
    }
}

#[test]
#[ignore = "manual timing; run with -- --ignored --nocapture on a quiet machine"]
fn time_inverse_candidates() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0xbeef);
    let inputs: Vec<u64> = (0..1_000_000).map(|_| rng.gen_range(1..P)).collect();

    let mut sink = 0u64;
    let t = std::time::Instant::now();
    for &x in &inputs {
        sink ^= fermat_inverse(x);
    }
    println!("fermat:      {:?}  (sink {sink})", t.elapsed());

    let mut sink = 0u64;
    let t = std::time::Instant::now();
    for &x in &inputs {
        sink ^= binary_xgcd_inverse(x);
    }
    println!("binary_xgcd: {:?}  (sink {sink})", t.elapsed());
}
