use core::fmt::Debug;

use plonky2_field::ops::Square;

use super::config::*;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::goldilocks_field::GoldilocksField as F;
use crate::field::types::{Field, PrimeField64};
use crate::gates::poseidon2::Poseidon2Gate;
use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

pub trait Poseidon2: PrimeField64 {
    #[inline]
    fn poseidon2(input: [Self; WIDTH]) -> [Self; WIDTH] {
        let mut state = input;

        Self::external_linear_layer(&mut state);

        Self::full_rounds(&mut state, 0);
        Self::partial_rounds(&mut state);
        Self::full_rounds(&mut state, ROUNDS_F_HALF);

        state
    }

    /// Permute two independent states together. The layers are applied to the
    /// two states in alternation so the CPU can overlap the serial sbox and
    /// linear-layer dependency chains of one state with the other's. Each
    /// state's output is bit-identical to `Self::poseidon2` on that state.
    #[inline]
    fn poseidon2_x2(
        input_a: [Self; WIDTH],
        input_b: [Self; WIDTH],
    ) -> ([Self; WIDTH], [Self; WIDTH]) {
        let mut a = input_a;
        let mut b = input_b;

        Self::external_linear_layer(&mut a);
        Self::external_linear_layer(&mut b);

        Self::full_rounds_x2(&mut a, &mut b, 0);
        Self::partial_rounds_x2(&mut a, &mut b);
        Self::full_rounds_x2(&mut a, &mut b, ROUNDS_F_HALF);

        (a, b)
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn full_rounds_x2(a: &mut [Self; WIDTH], b: &mut [Self; WIDTH], start: usize) {
        for r in start..(start + ROUNDS_F_HALF) {
            Self::add_rc(a, r);
            Self::add_rc(b, r);
            Self::sbox(a);
            Self::sbox(b);
            Self::external_linear_layer(a);
            Self::external_linear_layer(b);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn partial_rounds_x2(a: &mut [Self; WIDTH], b: &mut [Self; WIDTH]) {
        for r in 0..ROUNDS_P {
            a[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            b[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            a[0] = Self::sbox_p(&a[0]);
            b[0] = Self::sbox_p(&b[0]);
            Self::internal_linear_layer_x2(a, b);
        }
    }

    /// Four-state variant of `poseidon2_x2`; same bit-identity guarantee.
    #[inline]
    fn poseidon2_x4(
        input_a: [Self; WIDTH],
        input_b: [Self; WIDTH],
        input_c: [Self; WIDTH],
        input_d: [Self; WIDTH],
    ) -> ([Self; WIDTH], [Self; WIDTH], [Self; WIDTH], [Self; WIDTH]) {
        let mut a = input_a;
        let mut b = input_b;
        let mut c = input_c;
        let mut d = input_d;

        Self::external_linear_layer(&mut a);
        Self::external_linear_layer(&mut b);
        Self::external_linear_layer(&mut c);
        Self::external_linear_layer(&mut d);

        Self::full_rounds_x4(&mut a, &mut b, &mut c, &mut d, 0);
        Self::partial_rounds_x4(&mut a, &mut b, &mut c, &mut d);
        Self::full_rounds_x4(&mut a, &mut b, &mut c, &mut d, ROUNDS_F_HALF);

        (a, b, c, d)
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn full_rounds_x4(
        a: &mut [Self; WIDTH],
        b: &mut [Self; WIDTH],
        c: &mut [Self; WIDTH],
        d: &mut [Self; WIDTH],
        start: usize,
    ) {
        for r in start..(start + ROUNDS_F_HALF) {
            Self::add_rc(a, r);
            Self::add_rc(b, r);
            Self::add_rc(c, r);
            Self::add_rc(d, r);
            Self::sbox(a);
            Self::sbox(b);
            Self::sbox(c);
            Self::sbox(d);
            Self::external_linear_layer(a);
            Self::external_linear_layer(b);
            Self::external_linear_layer(c);
            Self::external_linear_layer(d);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn partial_rounds_x4(
        a: &mut [Self; WIDTH],
        b: &mut [Self; WIDTH],
        c: &mut [Self; WIDTH],
        d: &mut [Self; WIDTH],
    ) {
        for r in 0..ROUNDS_P {
            a[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            b[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            c[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            d[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            a[0] = Self::sbox_p(&a[0]);
            b[0] = Self::sbox_p(&b[0]);
            c[0] = Self::sbox_p(&c[0]);
            d[0] = Self::sbox_p(&d[0]);
            Self::internal_linear_layer_x4(a, b, c, d);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn full_rounds(state: &mut [Self; WIDTH], start: usize) {
        for r in start..(start + ROUNDS_F_HALF) {
            Self::add_rc(state, r);
            Self::sbox(state);
            Self::external_linear_layer(state);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn partial_rounds(state: &mut [Self; WIDTH]) {
        for r in 0..ROUNDS_P {
            state[0] += Self::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            state[0] = Self::sbox_p(&state[0]);
            Self::internal_linear_layer(state);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn external_linear_layer(state: &mut [Self; WIDTH]) {
        let mut state_u128: [u128; WIDTH] = [0u128; WIDTH];
        for i in 0..WIDTH {
            state_u128[i] = state[i].to_noncanonical_u64() as u128;
        }
        external_linear_layer_u128(&mut state_u128);
        for i in 0..WIDTH {
            state[i] = Self::from_noncanonical_u128_with_96_bits(state_u128[i]);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn external_linear_layer_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; WIDTH],
    ) {
        // First, we apply M_4 to each consecutive four elements of the state.
        // In Appendix B's terminology, this replaces each x_i with x_i'.
        for i in (0..WIDTH).step_by(4) {
            // Would be nice to find a better way to do this.
            let mut state_4 = [state[i], state[i + 1], state[i + 2], state[i + 3]];
            Self::apply_mat4_mut_extension(&mut state_4);
            state[i..i + 4].clone_from_slice(&state_4);
        }
        // Now, we apply the outer circulant matrix (to compute the y_i values).

        // We first precompute the four sums of every four elements.
        let sums: [F; 4] =
            core::array::from_fn(|k| (0..WIDTH).step_by(4).map(|j| state[j + k]).sum::<F>());

        // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
        // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
        for i in 0..WIDTH {
            state[i] += sums[i % 4];
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn internal_linear_layer(state: &mut [Self; WIDTH]) {
        let sum = sum_12(state); // hard coded for WIDTH = 12
        for i in 0..WIDTH {
            state[i] =
                sum.multiply_accumulate(state[i], Self::from_canonical_u64(MATRIX_DIAG_12_U64[i]));
        }
    }

    /// Apply the internal linear layer to two independent states. Default
    /// implementation is two sequential calls; Goldilocks overrides with an
    /// interleaved form that shares the twelve diagonal constants and overlaps
    /// the independent mul/add chains. Bit-identical to two `internal_linear_layer`
    /// calls.
    #[inline]
    fn internal_linear_layer_x2(a: &mut [Self; WIDTH], b: &mut [Self; WIDTH]) {
        Self::internal_linear_layer(a);
        Self::internal_linear_layer(b);
    }

    /// Four-state variant of `internal_linear_layer_x2`.
    #[inline]
    fn internal_linear_layer_x4(
        a: &mut [Self; WIDTH],
        b: &mut [Self; WIDTH],
        c: &mut [Self; WIDTH],
        d: &mut [Self; WIDTH],
    ) {
        Self::internal_linear_layer_x2(a, b);
        Self::internal_linear_layer_x2(c, d);
    }

    #[inline]
    fn internal_linear_layer_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; WIDTH],
    ) {
        let sum: F = state.iter().cloned().sum();
        state
            .iter_mut()
            .zip(MATRIX_DIAG_12_U64.iter())
            .for_each(|(x, &m)| {
                *x = sum.multiply_accumulate(*x, F::from_canonical_u64(m));
            });
    }

    fn add_rc(state: &mut [Self; WIDTH], external_round: usize);

    #[inline]
    #[unroll::unroll_for_loops]
    fn add_rc_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; WIDTH],
        external_round: usize,
    ) {
        debug_assert!(external_round < EXTERNAL_CONSTANTS.len());

        for i in 0..WIDTH {
            state[i] += F::from_canonical_u64(EXTERNAL_CONSTANTS[external_round][i]);
        }
    }

    fn sbox(state: &mut [Self; WIDTH]);

    #[inline]
    fn sbox_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; WIDTH],
    ) {
        state
            .iter_mut()
            .for_each(|a| *a = Self::sbox_p_extension(a));
    }

    fn sbox_p(a: &Self) -> Self;

    fn sbox_p_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(a: &F) -> F;

    #[inline]
    fn apply_mat4_mut_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        x: &mut [F; 4],
    ) {
        let t01 = x[0] + x[1];
        let t23 = x[2] + x[3];
        let t0123 = t01 + t23;
        let t01123 = t0123 + x[1];
        let t01233 = t0123 + x[3];
        // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
        x[3] = t01233 + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
        x[1] = t01123 + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
        x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
        x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
    }

    // In circuit functions
    #[inline]
    #[unroll::unroll_for_loops]
    fn external_linear_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        // First, we apply M_4 to each consecutive four elements of the state.
        // In Appendix B's terminology, this replaces each x_i with x_i'.
        for i in (0..WIDTH).step_by(4) {
            Self::apply_mat4_mut_circuit(builder, (&mut state[i..i + 4]).try_into().unwrap());
        }
        // Now, we apply the outer circulant matrix (to compute the y_i values).

        // We first precompute the four sums of every four elements.
        let sums: [ExtensionTarget<D>; 4] = core::array::from_fn(|k| {
            (0..WIDTH)
                .step_by(4)
                .map(|j| state[j + k])
                .reduce(|acc, t| builder.add_extension(acc, t))
                .unwrap()
        });

        // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
        // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
        for i in 0..WIDTH {
            state[i] = builder.add_extension(state[i], sums[i % 4]);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn apply_mat4_mut_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        x: &mut [ExtensionTarget<D>; 4],
    ) where
        Self: RichField + Extendable<D>,
    {
        let two = builder.constant_extension(Self::Extension::from_canonical_u64(2));

        let t01 = builder.add_extension(x[0], x[1]);
        let t23 = builder.add_extension(x[2], x[3]);
        let t0123 = builder.add_extension(t01, t23);
        let t01123 = builder.add_extension(t0123, x[1]);
        let t01233 = builder.add_extension(t0123, x[3]);
        // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
        let dx0 = builder.mul_extension(x[0], two);
        let dx2 = builder.mul_extension(x[2], two);
        x[3] = builder.add_extension(t01233, dx0); // 3*x[0] + x[1] + x[2] + 2*x[3]
        x[1] = builder.add_extension(t01123, dx2); // x[0] + 2*x[1] + 3*x[2] + x[3]
        x[0] = builder.add_extension(t01123, t01); // 2*x[0] + 3*x[1] + x[2] + x[3]
        x[2] = builder.add_extension(t01233, t23); // x[0] + x[1] + 2*x[2] + 3*x[3]
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn matmul_m4_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        input: &mut [ExtensionTarget<D>; WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..3 {
            let t_0 = builder.mul_const_add_extension(Self::ONE, input[i * 4], input[i * 4 + 1]);
            let t_1 =
                builder.mul_const_add_extension(Self::ONE, input[i * 4 + 2], input[i * 4 + 3]);
            let t_2 = builder.mul_const_add_extension(Self::TWO, input[i * 4 + 1], t_1);
            let t_3 = builder.mul_const_add_extension(Self::TWO, input[i * 4 + 3], t_0);

            let four = Self::TWO + Self::TWO;

            let t_4 = builder.mul_const_add_extension(four, t_1, t_3);
            let t_5 = builder.mul_const_add_extension(four, t_0, t_2);
            let t_6 = builder.mul_const_add_extension(Self::ONE, t_3, t_5);
            let t_7 = builder.mul_const_add_extension(Self::ONE, t_2, t_4);

            input[i * 4] = t_6;
            input[i * 4 + 1] = t_5;
            input[i * 4 + 2] = t_7;
            input[i * 4 + 3] = t_4;
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn add_rc_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        input: &mut [ExtensionTarget<D>; WIDTH],
        rc_index: usize,
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..WIDTH {
            let round_constant =
                Self::Extension::from_canonical_u64(EXTERNAL_CONSTANTS[rc_index][i]);
            let round_constant = builder.constant_extension(round_constant);
            input[i] = builder.add_extension(input[i], round_constant);
        }
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn sbox_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        input: &mut [ExtensionTarget<D>; WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..WIDTH {
            input[i] = Self::sbox_p_circuit(builder, input[i]);
        }
    }

    #[inline]
    fn sbox_p_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        input: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        Self: RichField + Extendable<D>,
    {
        builder.exp_u64_extension(input, super::config::D)
    }

    #[inline]
    #[unroll::unroll_for_loops]
    fn internal_linear_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        input: &mut [ExtensionTarget<D>; WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        let sum = builder.add_many_extension([
            input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
            input[8], input[9], input[10], input[11],
        ]);

        for i in 0..WIDTH {
            let round_constant = Self::Extension::from_canonical_u64(MATRIX_DIAG_12_U64[i]);
            let round_constant = builder.constant_extension(round_constant);

            input[i] = builder.mul_add_extension(round_constant, input[i], sum);
        }
    }
}

#[inline]
#[unroll::unroll_for_loops]
fn external_linear_layer_u128(state: &mut [u128; WIDTH]) {
    // First, we apply M_4 to each consecutive four elements of the state.
    // In Appendix B's terminology, this replaces each x_i with x_i'.
    for i in (0..WIDTH).step_by(4) {
        // Multiply a 4-element vector x by:
        // [ 2 3 1 1 ]
        // [ 1 2 3 1 ]
        // [ 1 1 2 3 ]
        // [ 3 1 1 2 ].
        let t01 = state[i] + state[i + 1];
        let t23 = state[i + 2] + state[i + 3];
        let t0123 = t01 + t23;

        let x0 = state[i];
        let x2 = state[i + 2];

        state[i] = t0123 + t01 + state[i + 1]; // 2*x[0] + 3*x[1] + x[2] + x[3]
        state[i + 1] = t0123 + state[i + 1] + x2 + x2; // x[0] + 2*x[1] + 3*x[2] + x[3]
        state[i + 2] = t0123 + t23 + state[i + 3]; // x[0] + x[1] + 2*x[2] + 3*x[3]
        state[i + 3] = t0123 + state[i + 3] + x0 + x0; // 3*x[0] + x[1] + x[2] + 2*x[3]
    }
    // Now, we apply the outer circulant matrix (to compute the y_i values).

    // We first precompute the four sums of every four elements.
    let mut sums = [0u128; 4];
    for i in 0..4 {
        sums[i] = state[i] + state[i + 4] + state[i + 8];
    }

    // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
    // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
    for i in 0..WIDTH {
        state[i] += sums[i % 4];
    }
}

impl Poseidon2 for F {
    #[inline]
    fn internal_linear_layer(state: &mut [Self; WIDTH]) {
        // AArch64 NEON has no native widening 64x64 multiply. The packed
        // Goldilocks path therefore expands each four-lane product, while the
        // scalar backend lowers each fixed product directly to `mul`/`umulh`.
        // Keep the exact historical multiply-then-add order in every lane.
        let sum = sum_12(state);
        state[0] = sum + state[0] * F(0xc3b6c08e23ba9300);
        state[1] = sum + state[1] * F(0xd84b5de94a324fb6);
        state[2] = sum + state[2] * F(0x0d0c371c5b35b84f);
        state[3] = sum + state[3] * F(0x7964f570e7188037);
        state[4] = sum + state[4] * F(0x5daf18bbd996604b);
        state[5] = sum + state[5] * F(0x6743bc47b9595257);
        state[6] = sum + state[6] * F(0x5528b9362c59bb70);
        state[7] = sum + state[7] * F(0xac45e25b7127b68b);
        state[8] = sum + state[8] * F(0xa2077d7dfbb606b5);
        state[9] = sum + state[9] * F(0xf3faac6faee378ae);
        state[10] = sum + state[10] * F(0x0c6388b51545e883);
        state[11] = sum + state[11] * F(0xd27dbb6944917b60);
    }

    /// Grouped x2 internal layer: sum both states first, then interleave the
    /// 24 independent lane expressions so the twelve diagonal immediates are
    /// shared and the two dependency chains overlap. Bit-identical to two
    /// sequential `internal_linear_layer` calls.
    #[inline]
    fn internal_linear_layer_x2(a: &mut [Self; WIDTH], b: &mut [Self; WIDTH]) {
        let sum_a = sum_12(a);
        let sum_b = sum_12(b);
        a[0] = sum_a + a[0] * F(0xc3b6c08e23ba9300);
        b[0] = sum_b + b[0] * F(0xc3b6c08e23ba9300);
        a[1] = sum_a + a[1] * F(0xd84b5de94a324fb6);
        b[1] = sum_b + b[1] * F(0xd84b5de94a324fb6);
        a[2] = sum_a + a[2] * F(0x0d0c371c5b35b84f);
        b[2] = sum_b + b[2] * F(0x0d0c371c5b35b84f);
        a[3] = sum_a + a[3] * F(0x7964f570e7188037);
        b[3] = sum_b + b[3] * F(0x7964f570e7188037);
        a[4] = sum_a + a[4] * F(0x5daf18bbd996604b);
        b[4] = sum_b + b[4] * F(0x5daf18bbd996604b);
        a[5] = sum_a + a[5] * F(0x6743bc47b9595257);
        b[5] = sum_b + b[5] * F(0x6743bc47b9595257);
        a[6] = sum_a + a[6] * F(0x5528b9362c59bb70);
        b[6] = sum_b + b[6] * F(0x5528b9362c59bb70);
        a[7] = sum_a + a[7] * F(0xac45e25b7127b68b);
        b[7] = sum_b + b[7] * F(0xac45e25b7127b68b);
        a[8] = sum_a + a[8] * F(0xa2077d7dfbb606b5);
        b[8] = sum_b + b[8] * F(0xa2077d7dfbb606b5);
        a[9] = sum_a + a[9] * F(0xf3faac6faee378ae);
        b[9] = sum_b + b[9] * F(0xf3faac6faee378ae);
        a[10] = sum_a + a[10] * F(0x0c6388b51545e883);
        b[10] = sum_b + b[10] * F(0x0c6388b51545e883);
        a[11] = sum_a + a[11] * F(0xd27dbb6944917b60);
        b[11] = sum_b + b[11] * F(0xd27dbb6944917b60);
    }

    /// Grouped x4 internal layer: four independent sums first, then interleaved
    /// lane updates across the four states. Bit-identical to four sequential
    /// `internal_linear_layer` calls.
    #[inline]
    fn internal_linear_layer_x4(
        a: &mut [Self; WIDTH],
        b: &mut [Self; WIDTH],
        c: &mut [Self; WIDTH],
        d: &mut [Self; WIDTH],
    ) {
        let sum_a = sum_12(a);
        let sum_b = sum_12(b);
        let sum_c = sum_12(c);
        let sum_d = sum_12(d);
        a[0] = sum_a + a[0] * F(0xc3b6c08e23ba9300);
        b[0] = sum_b + b[0] * F(0xc3b6c08e23ba9300);
        c[0] = sum_c + c[0] * F(0xc3b6c08e23ba9300);
        d[0] = sum_d + d[0] * F(0xc3b6c08e23ba9300);
        a[1] = sum_a + a[1] * F(0xd84b5de94a324fb6);
        b[1] = sum_b + b[1] * F(0xd84b5de94a324fb6);
        c[1] = sum_c + c[1] * F(0xd84b5de94a324fb6);
        d[1] = sum_d + d[1] * F(0xd84b5de94a324fb6);
        a[2] = sum_a + a[2] * F(0x0d0c371c5b35b84f);
        b[2] = sum_b + b[2] * F(0x0d0c371c5b35b84f);
        c[2] = sum_c + c[2] * F(0x0d0c371c5b35b84f);
        d[2] = sum_d + d[2] * F(0x0d0c371c5b35b84f);
        a[3] = sum_a + a[3] * F(0x7964f570e7188037);
        b[3] = sum_b + b[3] * F(0x7964f570e7188037);
        c[3] = sum_c + c[3] * F(0x7964f570e7188037);
        d[3] = sum_d + d[3] * F(0x7964f570e7188037);
        a[4] = sum_a + a[4] * F(0x5daf18bbd996604b);
        b[4] = sum_b + b[4] * F(0x5daf18bbd996604b);
        c[4] = sum_c + c[4] * F(0x5daf18bbd996604b);
        d[4] = sum_d + d[4] * F(0x5daf18bbd996604b);
        a[5] = sum_a + a[5] * F(0x6743bc47b9595257);
        b[5] = sum_b + b[5] * F(0x6743bc47b9595257);
        c[5] = sum_c + c[5] * F(0x6743bc47b9595257);
        d[5] = sum_d + d[5] * F(0x6743bc47b9595257);
        a[6] = sum_a + a[6] * F(0x5528b9362c59bb70);
        b[6] = sum_b + b[6] * F(0x5528b9362c59bb70);
        c[6] = sum_c + c[6] * F(0x5528b9362c59bb70);
        d[6] = sum_d + d[6] * F(0x5528b9362c59bb70);
        a[7] = sum_a + a[7] * F(0xac45e25b7127b68b);
        b[7] = sum_b + b[7] * F(0xac45e25b7127b68b);
        c[7] = sum_c + c[7] * F(0xac45e25b7127b68b);
        d[7] = sum_d + d[7] * F(0xac45e25b7127b68b);
        a[8] = sum_a + a[8] * F(0xa2077d7dfbb606b5);
        b[8] = sum_b + b[8] * F(0xa2077d7dfbb606b5);
        c[8] = sum_c + c[8] * F(0xa2077d7dfbb606b5);
        d[8] = sum_d + d[8] * F(0xa2077d7dfbb606b5);
        a[9] = sum_a + a[9] * F(0xf3faac6faee378ae);
        b[9] = sum_b + b[9] * F(0xf3faac6faee378ae);
        c[9] = sum_c + c[9] * F(0xf3faac6faee378ae);
        d[9] = sum_d + d[9] * F(0xf3faac6faee378ae);
        a[10] = sum_a + a[10] * F(0x0c6388b51545e883);
        b[10] = sum_b + b[10] * F(0x0c6388b51545e883);
        c[10] = sum_c + c[10] * F(0x0c6388b51545e883);
        d[10] = sum_d + d[10] * F(0x0c6388b51545e883);
        a[11] = sum_a + a[11] * F(0xd27dbb6944917b60);
        b[11] = sum_b + b[11] * F(0xd27dbb6944917b60);
        c[11] = sum_c + c[11] * F(0xd27dbb6944917b60);
        d[11] = sum_d + d[11] * F(0xd27dbb6944917b60);
    }

    #[inline]
    fn sbox_p(a: &Self) -> Self {
        let a2 = a.square();
        let a4 = a2.square();
        let a3 = *a * a2;
        a3 * a4
    }

    #[inline]
    fn sbox_p_extension<F: FieldExtension<D, BaseField = Self>, const D: usize>(a: &F) -> F {
        let a2 = a.square();
        let a4 = a2.square();
        let a3 = *a * a2;
        a3 * a4
    }

    /// Adds one external round's constants, each with a single wraparound
    /// correction.
    ///
    /// `add_canonical_u64` is the whole operation: `x + m` wrapped to 64 bits
    /// plus `EPSILON` on carry. Its precondition is that `m` is canonical,
    /// which holds for every entry of `EXTERNAL_CONSTANTS` — asserted by
    /// `external_constants_are_canonical` — and is what bounds `x + m` below
    /// `2^64 + ORDER` for any `x`, canonical or not, so one correction always
    /// suffices.
    ///
    /// This ran through a vector helper on AArch64 before. The helper is
    /// value-identical, but it reaches the state through `transmute` and
    /// `vld1q_u64`/`vst1q_u64`, so the twelve lanes have to be resident in
    /// memory across it: the copy in, the vector loads, the vector stores and
    /// the copy back are all traffic the scalar form does not have, and the
    /// permutation's other layers keep the same lanes in registers on either
    /// side of it. Twelve independent three-instruction wraparound adds have
    /// no dependency chain to hide behind vectorisation in the first place.
    #[inline]
    fn add_rc(state: &mut [Self; WIDTH], external_round: usize) {
        use plonky2_field::types::Field64;
        debug_assert!(external_round < EXTERNAL_CONSTANTS.len());
        state
            .iter_mut()
            .zip(EXTERNAL_CONSTANTS[external_round].iter())
            .for_each(|(x, &m)| {
                // SAFETY: `m < Self::ORDER` for every external constant.
                *x = unsafe { x.add_canonical_u64(m) };
            });
    }

    #[inline]
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    fn sbox(state: &mut [Self; WIDTH]) {
        state.iter_mut().for_each(|a| *a = Self::sbox_p(a));
    }

    #[inline(always)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn sbox(state: &mut [Self; WIDTH]) {
        unsafe {
            crate::hash::arch::aarch64::poseidon_goldilocks_neon::sbox_layer(state);
        }
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Poseidon2Permutation<T> {
    state: [T; WIDTH],
}

impl<T: Eq> Eq for Poseidon2Permutation<T> {}

impl<T> AsRef<[T]> for Poseidon2Permutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

trait Permuter: Sized {
    fn permute(input: [Self; WIDTH]) -> [Self; WIDTH];

    #[inline]
    fn permute_quad(inputs: [[Self; WIDTH]; 4]) -> [[Self; WIDTH]; 4] {
        inputs.map(Self::permute)
    }
}

impl<F: Poseidon2> Permuter for F {
    fn permute(input: [Self; WIDTH]) -> [Self; WIDTH] {
        <F as Poseidon2>::poseidon2(input)
    }

    #[inline]
    fn permute_quad(inputs: [[Self; WIDTH]; 4]) -> [[Self; WIDTH]; 4] {
        let [a, b, c, d] = inputs;
        let (a, b, c, d) = <F as Poseidon2>::poseidon2_x4(a, b, c, d);
        [a, b, c, d]
    }
}

impl Permuter for Target {
    fn permute(_input: [Self; WIDTH]) -> [Self; WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}

impl<T: Copy + Debug + Default + Eq + Permuter + Send + Sync> PlonkyPermutation<T>
    for Poseidon2Permutation<T>
{
    const RATE: usize = RATE;
    const WIDTH: usize = WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    #[inline]
    fn permute_quad(states: &mut [Self; 4]) {
        let inputs = (*states).map(|permutation| permutation.state);
        let outputs = T::permute_quad(inputs);
        for (state, output) in states.iter_mut().zip(outputs) {
            state.state = output;
        }
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

#[inline]
/// Sum of 12 elements to u128; unrolled for performance.
fn sum_12<F: PrimeField64>(inputs: &[F]) -> F {
    debug_assert!(inputs.len() == 12);
    let tmp = inputs[0].to_noncanonical_u64() as u128
        + inputs[1].to_noncanonical_u64() as u128
        + inputs[2].to_noncanonical_u64() as u128
        + inputs[3].to_noncanonical_u64() as u128
        + inputs[4].to_noncanonical_u64() as u128
        + inputs[5].to_noncanonical_u64() as u128
        + inputs[6].to_noncanonical_u64() as u128
        + inputs[7].to_noncanonical_u64() as u128
        + inputs[8].to_noncanonical_u64() as u128
        + inputs[9].to_noncanonical_u64() as u128
        + inputs[10].to_noncanonical_u64() as u128
        + inputs[11].to_noncanonical_u64() as u128;

    F::from_noncanonical_u128_with_96_bits(tmp)
}

/// Hash two equal-length inputs with two lockstep overwrite-mode sponges whose
/// permutations run interleaved via `poseidon2_x2`. Each output is
/// bit-identical to `hash_n_to_hash_no_pad` on the corresponding input.
pub(crate) fn hash_pair_no_pad<F: RichField + Poseidon2>(
    input_a: &[F],
    input_b: &[F],
) -> (HashOut<F>, HashOut<F>) {
    debug_assert_eq!(input_a.len(), input_b.len());
    let mut state_a = [F::ZERO; WIDTH];
    let mut state_b = [F::ZERO; WIDTH];

    for (chunk_a, chunk_b) in input_a.chunks(RATE).zip(input_b.chunks(RATE)) {
        state_a[..chunk_a.len()].copy_from_slice(chunk_a);
        state_b[..chunk_b.len()].copy_from_slice(chunk_b);
        (state_a, state_b) = F::poseidon2_x2(state_a, state_b);
    }

    (
        HashOut {
            elements: state_a[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        },
        HashOut {
            elements: state_b[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        },
    )
}

/// Four-input variant of `hash_pair_no_pad`: four lockstep overwrite-mode
/// sponges over equal-length inputs, permuted via `poseidon2_x4`. Each output
/// is bit-identical to `hash_n_to_hash_no_pad` on the corresponding input.
pub(crate) fn hash_quad_no_pad<F: RichField + Poseidon2>(
    input_a: &[F],
    input_b: &[F],
    input_c: &[F],
    input_d: &[F],
) -> (HashOut<F>, HashOut<F>, HashOut<F>, HashOut<F>) {
    debug_assert_eq!(input_a.len(), input_b.len());
    debug_assert_eq!(input_a.len(), input_c.len());
    debug_assert_eq!(input_a.len(), input_d.len());
    let mut state_a = [F::ZERO; WIDTH];
    let mut state_b = [F::ZERO; WIDTH];
    let mut state_c = [F::ZERO; WIDTH];
    let mut state_d = [F::ZERO; WIDTH];

    for (((chunk_a, chunk_b), chunk_c), chunk_d) in input_a
        .chunks(RATE)
        .zip(input_b.chunks(RATE))
        .zip(input_c.chunks(RATE))
        .zip(input_d.chunks(RATE))
    {
        state_a[..chunk_a.len()].copy_from_slice(chunk_a);
        state_b[..chunk_b.len()].copy_from_slice(chunk_b);
        state_c[..chunk_c.len()].copy_from_slice(chunk_c);
        state_d[..chunk_d.len()].copy_from_slice(chunk_d);
        (state_a, state_b, state_c, state_d) = F::poseidon2_x4(state_a, state_b, state_c, state_d);
    }

    let out = |state: &[F; WIDTH]| HashOut {
        elements: state[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
    };
    (out(&state_a), out(&state_b), out(&state_c), out(&state_d))
}

/// Two independent `compress` calls with their permutations interleaved via
/// `poseidon2_x2`. Each output is bit-identical to `compress` on that pair.
pub(crate) fn compress_pair<F: RichField + Poseidon2>(
    x0: HashOut<F>,
    y0: HashOut<F>,
    x1: HashOut<F>,
    y1: HashOut<F>,
) -> (HashOut<F>, HashOut<F>) {
    let mut state_a = [F::ZERO; WIDTH];
    let mut state_b = [F::ZERO; WIDTH];
    state_a[..NUM_HASH_OUT_ELTS].copy_from_slice(&x0.elements);
    state_a[NUM_HASH_OUT_ELTS..2 * NUM_HASH_OUT_ELTS].copy_from_slice(&y0.elements);
    state_b[..NUM_HASH_OUT_ELTS].copy_from_slice(&x1.elements);
    state_b[NUM_HASH_OUT_ELTS..2 * NUM_HASH_OUT_ELTS].copy_from_slice(&y1.elements);

    let (state_a, state_b) = F::poseidon2_x2(state_a, state_b);

    (
        HashOut {
            elements: state_a[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        },
        HashOut {
            elements: state_b[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        },
    )
}

/// Four independent `compress` calls with their permutations interleaved via
/// `poseidon2_x4`. Each output is bit-identical to `compress` on that pair.
pub(crate) fn compress_quad<F: RichField + Poseidon2>(
    inputs: [(HashOut<F>, HashOut<F>); 4],
) -> [HashOut<F>; 4] {
    let load = |(x, y): (HashOut<F>, HashOut<F>)| {
        let mut state = [F::ZERO; WIDTH];
        state[..NUM_HASH_OUT_ELTS].copy_from_slice(&x.elements);
        state[NUM_HASH_OUT_ELTS..2 * NUM_HASH_OUT_ELTS].copy_from_slice(&y.elements);
        state
    };
    let (a, b, c, d) = F::poseidon2_x4(
        load(inputs[0]),
        load(inputs[1]),
        load(inputs[2]),
        load(inputs[3]),
    );
    let out = |state: [F; WIDTH]| HashOut {
        elements: state[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
    };
    [out(a), out(b), out(c), out(d)]
}

/// Poseidon2 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2Hash;
impl<F: RichField + Poseidon2> Hasher<F> for Poseidon2Hash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = Poseidon2Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn hash_or_noop_pair(input_a: &[F], input_b: &[F]) -> (Self::Hash, Self::Hash) {
        debug_assert_eq!(input_a.len(), input_b.len());
        if input_a.len() * 8 <= <Self as Hasher<F>>::HASH_SIZE {
            (
                <Self as Hasher<F>>::hash_or_noop(input_a),
                <Self as Hasher<F>>::hash_or_noop(input_b),
            )
        } else {
            hash_pair_no_pad::<F>(input_a, input_b)
        }
    }

    fn hash_or_noop_quad(
        input_a: &[F],
        input_b: &[F],
        input_c: &[F],
        input_d: &[F],
    ) -> (Self::Hash, Self::Hash, Self::Hash, Self::Hash) {
        debug_assert_eq!(input_a.len(), input_b.len());
        debug_assert_eq!(input_a.len(), input_c.len());
        debug_assert_eq!(input_a.len(), input_d.len());
        if input_a.len() * 8 <= <Self as Hasher<F>>::HASH_SIZE {
            (
                <Self as Hasher<F>>::hash_or_noop(input_a),
                <Self as Hasher<F>>::hash_or_noop(input_b),
                <Self as Hasher<F>>::hash_or_noop(input_c),
                <Self as Hasher<F>>::hash_or_noop(input_d),
            )
        } else {
            hash_quad_no_pad::<F>(input_a, input_b, input_c, input_d)
        }
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }

    fn two_to_one_pair(
        x0: Self::Hash,
        y0: Self::Hash,
        x1: Self::Hash,
        y1: Self::Hash,
    ) -> (Self::Hash, Self::Hash) {
        compress_pair::<F>(x0, y0, x1, y1)
    }

    fn two_to_one_quad(inputs: [(Self::Hash, Self::Hash); 4]) -> [Self::Hash; 4] {
        compress_quad::<F>(inputs)
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_build_merkle_tree(
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::LevelOrderDigests<Self::Hash>,
        Vec<Self::Hash>,
    )> {
        super::metal::build_merkle_tree(leaves, leaf_width, num_leaves, cap_height)
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_build_merkle_tree_columns(
        columns: &[Vec<F>],
        cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::LevelOrderDigests<Self::Hash>,
        Vec<Self::Hash>,
    )> {
        super::metal::build_merkle_tree_columns(columns, cap_height)
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_allocate_merkle_tree_columns(
        num_columns: usize,
        num_rows: usize,
        cap_height: usize,
    ) -> Option<crate::hash::merkle_tree::ColumnStore<F>> {
        super::metal::allocate_columns(num_columns, num_rows, cap_height)
            .map(crate::hash::merkle_tree::ColumnStore::Shared)
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_build_merkle_tree_column_store(
        columns: &crate::hash::merkle_tree::ColumnStore<F>,
        cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::LevelOrderDigests<Self::Hash>,
        Vec<Self::Hash>,
    )> {
        match columns {
            crate::hash::merkle_tree::ColumnStore::Owned(columns) => {
                super::metal::build_merkle_tree_columns(columns, cap_height)
            }
            crate::hash::merkle_tree::ColumnStore::Shared(columns) => {
                super::metal::build_merkle_tree_shared(columns, cap_height)
            }
        }
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_build_merkle_tree_column_store_streamed(
        columns: &crate::hash::merkle_tree::ColumnStore<F>,
        cap_height: usize,
        fill_group: &(dyn Fn(usize, &mut [&mut [F]]) + Sync),
    ) -> Option<(
        crate::hash::merkle_tree::LevelOrderDigests<Self::Hash>,
        Vec<Self::Hash>,
    )> {
        match columns {
            crate::hash::merkle_tree::ColumnStore::Owned(_) => None,
            crate::hash::merkle_tree::ColumnStore::Shared(columns) => {
                super::metal::build_merkle_tree_shared_streamed(columns, cap_height, fill_group)
            }
        }
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_build_commitment_from_coeffs(
        coeff_columns: &[&[F]],
        rate_bits: usize,
        cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::ColumnStore<F>,
        crate::hash::merkle_tree::LevelOrderDigests<Self::Hash>,
        Vec<Self::Hash>,
    )> {
        super::metal::build_commitment_from_coeffs(coeff_columns, rate_bits, cap_height).map(
            |(columns, digests, cap)| {
                (
                    crate::hash::merkle_tree::ColumnStore::Shared(columns),
                    digests,
                    cap,
                )
            },
        )
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    fn try_build_commitment_from_values(
        value_columns: &[&[F]],
        rate_bits: usize,
        cap_height: usize,
    ) -> Option<(
        crate::hash::merkle_tree::ColumnStore<F>,
        crate::hash::merkle_tree::LevelOrderDigests<Self::Hash>,
        Vec<Self::Hash>,
        Vec<Vec<F>>,
    )> {
        super::metal::build_commitment_from_values(value_columns, rate_bits, cap_height).map(
            |(columns, digests, cap, coeffs)| {
                (
                    crate::hash::merkle_tree::ColumnStore::Shared(columns),
                    digests,
                    cap,
                    coeffs,
                )
            },
        )
    }
}

impl Poseidon2Hash {
    #[inline]
    #[unroll::unroll_for_loops]
    pub fn hash_n_to_one(
        input: &[<Poseidon2Hash as Hasher<F>>::Hash],
    ) -> <Poseidon2Hash as Hasher<F>>::Hash {
        assert!(!input.is_empty());

        if input.len() == 1 {
            return input[0];
        }

        let mut result = <Poseidon2Hash as Hasher<F>>::two_to_one(input[0], input[1]);

        for i in 2..input.len() {
            result = <Poseidon2Hash as Hasher<F>>::two_to_one(result, input[i]);
        }

        result
    }
}

impl<F: RichField + Poseidon2> AlgebraicHasher<F> for Poseidon2Hash {
    type AlgebraicPermutation = Poseidon2Permutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        let gate_type = Poseidon2Gate::<F, D>::new();
        let gate = builder.add_gate(gate_type, vec![]);

        let swap_wire = Poseidon2Gate::<F, D>::WIRE_SWAP;
        let swap_wire = Target::wire(gate, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        let inputs = inputs.as_ref();
        for i in 0..WIDTH {
            let in_wire = Poseidon2Gate::<F, D>::wire_input(i);
            let in_wire = Target::wire(gate, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Collect output wires.
        Self::AlgebraicPermutation::new(
            (0..WIDTH).map(|i| Target::wire(gate, Poseidon2Gate::<F, D>::wire_output(i))),
        )
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use num::{BigUint, One};
    use p3_field::{AbstractField, PrimeField64 as _};
    use p3_goldilocks::Goldilocks;
    use rand::{thread_rng, RngCore};

    use super::*;
    use crate::field::types::PrimeField64;
    use crate::hash::hashing::hash_n_to_m_no_pad;
    use crate::hash::poseidon2::p3::p3_poseidon2_hash_n_to_m_no_pad;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    #[test]
    fn test_poseidon2_with_plonky3() {
        let mut rng = thread_rng();

        let input: [u32; 12] = core::array::from_fn(|_| rng.next_u32());

        let input_f = input
            .iter()
            .map(|&x| F::from_canonical_u64((x as u64) + 1073741824))
            .collect::<Vec<F>>();
        let expected_output_f = hash_n_to_m_no_pad::<F, Poseidon2Permutation<F>>(&input_f, 12);

        let input_f3 = input
            .iter()
            .map(|&x| Goldilocks::from_canonical_u64((x as u64) + 1073741824))
            .collect::<Vec<Goldilocks>>();
        let expected_output_f3 = p3_poseidon2_hash_n_to_m_no_pad(&input_f3, 12);

        for i in 0..4 {
            assert_eq!(
                expected_output_f[i].to_canonical_u64(),
                expected_output_f3[i].as_canonical_u64()
            );
        }
    }

    #[test]
    fn test_poseidon2_gate() -> Result<()> {
        let mut rng = thread_rng();

        let input: [u32; 12] = core::array::from_fn(|_| rng.next_u32());
        let input_f = input
            .iter()
            .map(|&x| F::from_canonical_u64((x as u64) + 1073741824))
            .collect::<Vec<F>>();

        let expected_output = hash_n_to_m_no_pad::<F, Poseidon2Permutation<F>>(&input_f[0..8], 4);

        let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());

        let input_target: [Target; 12] = input_f
            .iter()
            .map(|&x| builder.constant(x))
            .collect::<Vec<Target>>()
            .try_into()
            .unwrap();
        let output_target =
            builder.hash_n_to_m_no_pad::<Poseidon2Hash>(input_target[0..8].to_vec(), 4);

        let expected_output_target = builder.add_virtual_target_arr::<4>();
        for i in 0..4 {
            builder.connect(expected_output_target[i], output_target[i]);
        }

        let circuit = builder.build::<PoseidonGoldilocksConfig>();
        let mut pw = PartialWitness::new();
        pw.set_target_arr(&expected_output_target, &expected_output)?;

        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof.clone())
    }

    #[test]
    fn test_poseidon2_gate_big() -> Result<()> {
        let input_f: [F; 12] =
            core::array::from_fn(|_| F::from_noncanonical_biguint(F::order() - BigUint::one()));

        let expected_output = hash_n_to_m_no_pad::<F, Poseidon2Permutation<F>>(&input_f[0..8], 4);

        let mut builder = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());

        let input_target: [Target; 12] = input_f
            .iter()
            .map(|&x| builder.constant(x))
            .collect::<Vec<Target>>()
            .try_into()
            .unwrap();
        let output_target =
            builder.hash_n_to_m_no_pad::<Poseidon2Hash>(input_target[0..8].to_vec(), 4);

        let expected_output_target = builder.add_virtual_target_arr::<4>();
        for i in 0..4 {
            builder.connect(expected_output_target[i], output_target[i]);
        }

        let circuit = builder.build::<PoseidonGoldilocksConfig>();
        let mut pw = PartialWitness::new();
        pw.set_target_arr(&expected_output_target, &expected_output)?;

        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof.clone())
    }
}

#[cfg(test)]
mod pair_hash_tests {
    use plonky2_field::types::Sample;

    use super::*;
    use crate::plonk::config::Hasher;

    #[test]
    fn permutation_quad_matches_scalar_in_all_four_lanes() {
        let base =
            Poseidon2Permutation::<F>::new((0..WIDTH).map(|i| F::from_canonical_usize(10_000 + i)));
        let candidates = [17_u64, 29, 41, 53];
        let witness_input_pos = 6;
        let mut quad = [base; 4];
        for (state, candidate) in quad.iter_mut().zip(candidates) {
            state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
        }

        let mut scalar = quad;
        for state in &mut scalar {
            state.permute();
        }
        Poseidon2Permutation::<F>::permute_quad(&mut quad);

        for lane in 0..4 {
            assert_eq!(
                quad[lane].as_ref(),
                scalar[lane].as_ref(),
                "quad permutation changed lane {lane}"
            );
        }
    }

    /// `add_rc` calls `add_canonical_u64`, whose safety and single-correction
    /// argument both require canonical round constants.
    #[test]
    fn external_constants_are_canonical() {
        use plonky2_field::types::Field64;
        for (round, constants) in EXTERNAL_CONSTANTS.iter().enumerate() {
            for (i, &c) in constants.iter().enumerate() {
                assert!(
                    c < <F as Field64>::ORDER,
                    "EXTERNAL_CONSTANTS[{round}][{i}] is not canonical"
                );
            }
        }
    }

    #[test]
    fn pair_hash_matches_individual_across_widths() {
        for width in [1, 2, 4, 5, 7, 8, 9, 16, 17, 24, 33, 87, 135] {
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

    // Not a correctness test: times sequential sibling-leaf hashing against the
    // interleaved pair sponge. Run with --nocapture.
    #[test]
    fn time_sequential_vs_pair_leaf_hash() {
        let width = 87;
        let a: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let b: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let iters = 100_000;

        let t0 = std::time::Instant::now();
        let mut sink_old = F::ZERO;
        for _ in 0..iters {
            let ha = <Poseidon2Hash as Hasher<F>>::hash_or_noop(core::hint::black_box(&a));
            let hb = <Poseidon2Hash as Hasher<F>>::hash_or_noop(core::hint::black_box(&b));
            sink_old += ha.elements[0] + hb.elements[0];
        }
        let old_time = t0.elapsed();

        let t1 = std::time::Instant::now();
        let mut sink_new = F::ZERO;
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
            "sequential: {:?}  interleaved pair: {:?}  speedup: {:.2}x",
            old_time,
            new_time,
            old_time.as_secs_f64() / new_time.as_secs_f64()
        );
    }
}
