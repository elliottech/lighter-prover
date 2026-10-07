#include <metal_stdlib>
using namespace metal;

// Goldilocks arithmetic (adapted from the Lighter prover-challenge Poseidon2
// kernel; same field, width-independent primitives).

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0xffffffffUL;

inline void add_epsilon_u32(thread uint& lo, thread uint& hi, uint active) {
    uint old0 = lo;
    lo -= active;
    uint old1 = hi;
    hi += active & (uint)(old0 != 0);
    uint overflow = active & (uint)(hi < old1);
    old0 = lo;
    lo -= overflow;
    hi += overflow & (uint)(old0 != 0);
}

inline void sub_epsilon_u32(thread uint& lo, thread uint& hi, uint active) {
    uint old0 = lo;
    lo += active;
    uint old1 = hi;
    hi -= active & (uint)(old0 != 0xffffffffU);
    uint underflow = active & (uint)(hi > old1);
    old0 = lo;
    lo += underflow;
    hi -= underflow & (uint)(old0 != 0xffffffffU);
}

inline ulong gl_add(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    uint r0 = a0 + b0;
    uint carry0 = (uint)(r0 < a0);
    uint r1 = a1 + b1;
    uint carry1 = (uint)(r1 < a1);
    uint next = r1 + carry0;
    carry1 += (uint)(next < r1);
    r1 = next;
    add_epsilon_u32(r0, r1, carry1);
    return ((ulong)r1 << 32) | (ulong)r0;
}

inline ulong reduce_top(uint r0, uint r1, int top) {
    ulong r = ((ulong)r1 << 32) | (ulong)r0;
    return r + (ulong)(((long)top << 32) - (long)top);
}

inline void mul_128(
    ulong a,
    ulong b,
    thread uint& l0,
    thread uint& l1,
    thread uint& h0,
    thread uint& h1) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint a0 = av.x;
    uint a1 = av.y;
    uint b0 = bv.x;
    uint b1 = bv.y;
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong t = p01 + (p00 >> 32);
    ulong m = t + p10;
    uint carry = (uint)(m < t);

    l0 = (uint)p00;
    l1 = (uint)m;

    uint q0 = (uint)p11;
    uint q1 = (uint)(p11 >> 32);
    uint mh = (uint)(m >> 32);
    h0 = q0 + mh;
    h1 = q1 + (uint)(h0 < q0) + carry;
}

inline ulong gl_mul(ulong a, ulong b) {
    uint l0;
    uint l1;
    uint h0;
    uint h1;
    mul_128(a, b, l0, l1, h0, h1);

    uint r0 = l0 - h0;
    uint borrow = (uint)(r0 > l0);
    uint next = r0 - h1;
    borrow += (uint)(next > r0);
    r0 = next;

    uint r1 = l1 + h0;
    uint carry = (uint)(r1 < l1);
    next = r1 - borrow;
    uint under = (uint)(next > r1);
    r1 = next;

    return reduce_top(r0, r1, (int)carry - (int)under);
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

// x^7 S-box.
inline ulong gl_exp7(ulong x) {
    ulong x2 = gl_mul(x, x);
    ulong x3 = gl_mul(x2, x);
    ulong x4 = gl_mul(x2, x2);
    return gl_mul(x3, x4);
}

// ---------------------------------------------------------------------------
// Plonky3 Poseidon2 width-16 permutation over Goldilocks.
//
// Round constants and the internal diagonal are passed from the host (read
// straight from p3-goldilocks) in one params buffer laid out as:
//   [0..64)    external initial RCs (4 rounds x 16)
//   [64..128)  external final RCs (4 rounds x 16)
//   [128..150) internal RCs (22)
//   [150..166) internal matrix diagonal (16)
constant uint P2_WIDTH = 16;
constant uint P2_RATE = 8;
constant uint P2_DIGEST = 8;
constant uint P2_EXTERNAL_ROUNDS_HALF = 4;
constant uint P2_INTERNAL_ROUNDS = 22;
constant uint PARAMS_EXTERNAL_INITIAL = 0;
constant uint PARAMS_EXTERNAL_FINAL = 64;
constant uint PARAMS_INTERNAL_RC = 128;
constant uint PARAMS_INTERNAL_DIAG = 150;

// The M4 block matrix [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]] applied to
// each aligned 4-element block, followed by adding the per-position block
// column sums (the standard Poseidon2 "MDS light" external layer).
inline void external_linear_layer(thread ulong* state) {
    for (uint block = 0; block < P2_WIDTH; block += 4) {
        ulong x0 = state[block];
        ulong x1 = state[block + 1];
        ulong x2 = state[block + 2];
        ulong x3 = state[block + 3];
        ulong t01 = gl_add(x0, x1);
        ulong t23 = gl_add(x2, x3);
        ulong t0123 = gl_add(t01, t23);
        ulong t01123 = gl_add(t0123, x1);
        ulong t01233 = gl_add(t0123, x3);
        state[block + 3] = gl_add(t01233, gl_add(x0, x0));
        state[block + 1] = gl_add(t01123, gl_add(x2, x2));
        state[block] = gl_add(t01123, t01);
        state[block + 2] = gl_add(t01233, t23);
    }
    ulong sums[4];
    for (uint k = 0; k < 4; k++) {
        sums[k] = 0;
        for (uint block = 0; block < P2_WIDTH; block += 4) {
            sums[k] = gl_add(sums[k], state[block + k]);
        }
    }
    for (uint i = 0; i < P2_WIDTH; i++) {
        state[i] = gl_add(state[i], sums[i % 4]);
    }
}

inline void internal_linear_layer(thread ulong* state, constant ulong* diag) {
    ulong sum = 0;
    for (uint i = 0; i < P2_WIDTH; i++) {
        sum = gl_add(sum, state[i]);
    }
    for (uint i = 0; i < P2_WIDTH; i++) {
        state[i] = gl_add(gl_mul(state[i], diag[i]), sum);
    }
}

inline void poseidon2_16_permute(thread ulong* state, constant ulong* params) {
    external_linear_layer(state);
    for (uint round = 0; round < P2_EXTERNAL_ROUNDS_HALF; round++) {
        constant ulong* rc = params + PARAMS_EXTERNAL_INITIAL + round * P2_WIDTH;
        for (uint i = 0; i < P2_WIDTH; i++) {
            state[i] = gl_exp7(gl_add(state[i], rc[i]));
        }
        external_linear_layer(state);
    }
    for (uint round = 0; round < P2_INTERNAL_ROUNDS; round++) {
        state[0] = gl_exp7(gl_add(state[0], params[PARAMS_INTERNAL_RC + round]));
        internal_linear_layer(state, params + PARAMS_INTERNAL_DIAG);
    }
    for (uint round = 0; round < P2_EXTERNAL_ROUNDS_HALF; round++) {
        constant ulong* rc = params + PARAMS_EXTERNAL_FINAL + round * P2_WIDTH;
        for (uint i = 0; i < P2_WIDTH; i++) {
            state[i] = gl_exp7(gl_add(state[i], rc[i]));
        }
        external_linear_layer(state);
    }
}

// PaddingFreeSponge<Perm, 16, 8, 8> over one row per thread: overwrite-mode
// absorb in rate-8 chunks — a final partial chunk overwrites only its own
// positions, leaving the rest of the rate untouched — then squeeze the first
// 8 state elements as the digest.
kernel void p3_poseidon2_16_hash_leaves(
    device const ulong* rows [[buffer(0)]],
    device ulong* digests [[buffer(1)]],
    constant ulong* params [[buffer(2)]],
    constant uint& row_width [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= num_rows) {
        return;
    }
    device const ulong* row = rows + (ulong)gid * (ulong)row_width;
    ulong state[16];
    for (uint i = 0; i < P2_WIDTH; i++) {
        state[i] = 0;
    }
    uint offset = 0;
    while (offset < row_width) {
        uint chunk = min(P2_RATE, row_width - offset);
        for (uint i = 0; i < chunk; i++) {
            state[i] = row[offset + i];
        }
        poseidon2_16_permute(state, params);
        offset += chunk;
    }
    device ulong* out = digests + (ulong)gid * (ulong)P2_DIGEST;
    for (uint i = 0; i < P2_DIGEST; i++) {
        out[i] = gl_canonicalize(state[i]);
    }
}
