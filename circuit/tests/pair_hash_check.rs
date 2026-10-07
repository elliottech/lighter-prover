// Temporary validation for the interleaved Poseidon2 pair sponge:
// bit-equality against the individual hash across leaf widths, plus a
// single-threaded timing comparison. Removed before submission if noisy.

use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::Sample;
use plonky2::hash::poseidon2::hash::Poseidon2Hash;
use plonky2::plonk::config::Hasher;

#[test]
fn pair_hash_matches_individual_across_widths() {
    for width in [1usize, 2, 4, 5, 7, 8, 9, 16, 17, 24, 33, 87, 135] {
        let a: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let b: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let (ha, hb) = Poseidon2Hash::hash_or_noop_pair(&a, &b);
        assert_eq!(
            ha,
            <Poseidon2Hash as Hasher<F>>::hash_or_noop(&a),
            "width {width} a"
        );
        assert_eq!(
            hb,
            <Poseidon2Hash as Hasher<F>>::hash_or_noop(&b),
            "width {width} b"
        );
    }
}

#[test]
fn time_sequential_vs_pair_leaf_hash() {
    for width in [8usize, 16, 87, 135] {
        let a: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let b: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let iters = 100_000;

        let t0 = std::time::Instant::now();
        let mut sink_old = F::default();
        for _ in 0..iters {
            let ha = <Poseidon2Hash as Hasher<F>>::hash_or_noop(core::hint::black_box(&a));
            let hb = <Poseidon2Hash as Hasher<F>>::hash_or_noop(core::hint::black_box(&b));
            sink_old += ha.elements[0] + hb.elements[0];
        }
        let old_time = t0.elapsed();

        let t1 = std::time::Instant::now();
        let mut sink_new = F::default();
        for _ in 0..iters {
            let (ha, hb) = Poseidon2Hash::hash_or_noop_pair(
                core::hint::black_box(&a),
                core::hint::black_box(&b),
            );
            sink_new += ha.elements[0] + hb.elements[0];
        }
        let new_time = t1.elapsed();

        assert_eq!(sink_old, sink_new);
        println!(
            "width {width}: sequential {:?}  interleaved {:?}  speedup {:.2}x",
            old_time,
            new_time,
            old_time.as_secs_f64() / new_time.as_secs_f64()
        );
    }
}

#[test]
fn quad_hash_matches_individual_across_widths() {
    for width in [1usize, 4, 5, 8, 9, 17, 87, 135] {
        let ins: Vec<Vec<F>> = (0..4)
            .map(|_| (0..width).map(|_| F::rand()).collect())
            .collect();
        let (h0, h1, h2, h3) = Poseidon2Hash::hash_or_noop_quad(&ins[0], &ins[1], &ins[2], &ins[3]);
        let expect = |i: usize| <Poseidon2Hash as Hasher<F>>::hash_or_noop(&ins[i]);
        assert_eq!(
            (h0, h1, h2, h3),
            (expect(0), expect(1), expect(2), expect(3)),
            "width {width}"
        );
    }
}

#[test]
fn compress_pair_matches_individual() {
    for _ in 0..32 {
        let h: Vec<_> = (0..4)
            .map(|_| {
                let v: Vec<F> = (0..8).map(|_| F::rand()).collect();
                <Poseidon2Hash as Hasher<F>>::hash_or_noop(&v)
            })
            .collect();
        let (n0, n1) = Poseidon2Hash::two_to_one_pair(h[0], h[1], h[2], h[3]);
        assert_eq!(n0, <Poseidon2Hash as Hasher<F>>::two_to_one(h[0], h[1]));
        assert_eq!(n1, <Poseidon2Hash as Hasher<F>>::two_to_one(h[2], h[3]));
    }
}

#[test]
fn time_pairx2_vs_quad_leaf_hash() {
    for width in [8usize, 87, 135] {
        let ins: Vec<Vec<F>> = (0..4)
            .map(|_| (0..width).map(|_| F::rand()).collect())
            .collect();
        let iters = 50_000;

        let t0 = std::time::Instant::now();
        let mut sink_pair = F::default();
        for _ in 0..iters {
            let (h0, h1) = Poseidon2Hash::hash_or_noop_pair(
                core::hint::black_box(&ins[0]),
                core::hint::black_box(&ins[1]),
            );
            let (h2, h3) = Poseidon2Hash::hash_or_noop_pair(
                core::hint::black_box(&ins[2]),
                core::hint::black_box(&ins[3]),
            );
            sink_pair += h0.elements[0] + h1.elements[0] + h2.elements[0] + h3.elements[0];
        }
        let pair_time = t0.elapsed();

        let t1 = std::time::Instant::now();
        let mut sink_quad = F::default();
        for _ in 0..iters {
            let (h0, h1, h2, h3) = Poseidon2Hash::hash_or_noop_quad(
                core::hint::black_box(&ins[0]),
                core::hint::black_box(&ins[1]),
                core::hint::black_box(&ins[2]),
                core::hint::black_box(&ins[3]),
            );
            sink_quad += h0.elements[0] + h1.elements[0] + h2.elements[0] + h3.elements[0];
        }
        let quad_time = t1.elapsed();

        assert_eq!(sink_pair, sink_quad);
        println!(
            "width {width}: 2x pair {:?}  quad {:?}  quad speedup over pairs {:.2}x",
            pair_time,
            quad_time,
            pair_time.as_secs_f64() / quad_time.as_secs_f64()
        );
    }
}

#[test]
fn time_sequential_vs_pair_compress() {
    let h: Vec<_> = (0..4)
        .map(|_| {
            let v: Vec<F> = (0..8).map(|_| F::rand()).collect();
            <Poseidon2Hash as Hasher<F>>::hash_or_noop(&v)
        })
        .collect();
    let iters = 500_000;

    let t0 = std::time::Instant::now();
    let mut sink_old = F::default();
    for _ in 0..iters {
        let n0 = <Poseidon2Hash as Hasher<F>>::two_to_one(
            core::hint::black_box(h[0]),
            core::hint::black_box(h[1]),
        );
        let n1 = <Poseidon2Hash as Hasher<F>>::two_to_one(
            core::hint::black_box(h[2]),
            core::hint::black_box(h[3]),
        );
        sink_old += n0.elements[0] + n1.elements[0];
    }
    let old_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let mut sink_new = F::default();
    for _ in 0..iters {
        let (n0, n1) = Poseidon2Hash::two_to_one_pair(
            core::hint::black_box(h[0]),
            core::hint::black_box(h[1]),
            core::hint::black_box(h[2]),
            core::hint::black_box(h[3]),
        );
        sink_new += n0.elements[0] + n1.elements[0];
    }
    let new_time = t1.elapsed();

    assert_eq!(sink_old, sink_new);
    println!(
        "compress: sequential {:?}  interleaved pair {:?}  speedup {:.2}x",
        old_time,
        new_time,
        old_time.as_secs_f64() / new_time.as_secs_f64()
    );
}

#[test]
fn compress_quad_matches_individual() {
    for _ in 0..32 {
        let h: Vec<_> = (0..8)
            .map(|_| {
                let v: Vec<F> = (0..8).map(|_| F::rand()).collect();
                <Poseidon2Hash as Hasher<F>>::hash_or_noop(&v)
            })
            .collect();
        let outs = Poseidon2Hash::two_to_one_quad([
            (h[0], h[1]),
            (h[2], h[3]),
            (h[4], h[5]),
            (h[6], h[7]),
        ]);
        for k in 0..4 {
            assert_eq!(
                outs[k],
                <Poseidon2Hash as Hasher<F>>::two_to_one(h[2 * k], h[2 * k + 1]),
                "quad compress {k}"
            );
        }
    }
}

#[test]
fn time_pair_vs_quad_compress() {
    let h: Vec<_> = (0..8)
        .map(|_| {
            let v: Vec<F> = (0..8).map(|_| F::rand()).collect();
            <Poseidon2Hash as Hasher<F>>::hash_or_noop(&v)
        })
        .collect();
    let iters = 250_000;

    let t0 = std::time::Instant::now();
    let mut sink_pair = F::default();
    for _ in 0..iters {
        let (n0, n1) = Poseidon2Hash::two_to_one_pair(
            core::hint::black_box(h[0]),
            core::hint::black_box(h[1]),
            core::hint::black_box(h[2]),
            core::hint::black_box(h[3]),
        );
        let (n2, n3) = Poseidon2Hash::two_to_one_pair(
            core::hint::black_box(h[4]),
            core::hint::black_box(h[5]),
            core::hint::black_box(h[6]),
            core::hint::black_box(h[7]),
        );
        sink_pair += n0.elements[0] + n1.elements[0] + n2.elements[0] + n3.elements[0];
    }
    let pair_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let mut sink_quad = F::default();
    for _ in 0..iters {
        let outs = Poseidon2Hash::two_to_one_quad([
            (core::hint::black_box(h[0]), core::hint::black_box(h[1])),
            (core::hint::black_box(h[2]), core::hint::black_box(h[3])),
            (core::hint::black_box(h[4]), core::hint::black_box(h[5])),
            (core::hint::black_box(h[6]), core::hint::black_box(h[7])),
        ]);
        sink_quad +=
            outs[0].elements[0] + outs[1].elements[0] + outs[2].elements[0] + outs[3].elements[0];
    }
    let quad_time = t1.elapsed();

    assert_eq!(sink_pair, sink_quad);
    println!(
        "compress quad: 2x pair {:?}  quad {:?}  speedup {:.2}x",
        pair_time,
        quad_time,
        pair_time.as_secs_f64() / quad_time.as_secs_f64()
    );
}
