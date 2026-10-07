#include <metal_stdlib>
using namespace metal;

// Goldilocks arithmetic (same primitives as the Merkle leaf kernel; adapted
// from the Lighter prover-challenge plonky2 NTT kernels).

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;

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

inline ulong gl_sub(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    uint r0 = a0 - b0;
    uint borrow0 = (uint)(r0 > a0);
    uint r1 = a1 - b1;
    uint under = (uint)(r1 > a1);
    uint next = r1 - borrow0;
    under += (uint)(next > r1);
    r1 = next;
    sub_epsilon_u32(r0, r1, under);
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

// Gathers a block of `cols` columns starting at `col0` from a row-major
// matrix of width `total_cols` into a column-major block buffer.
kernel void p3_transpose_in(
    const device ulong* input [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& degree [[buffer(2)]],
    constant uint& total_cols [[buffer(3)]],
    constant uint& col0 [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint r = gid.x;
    if (r >= degree) {
        return;
    }
    uint c = gid.y;
    out[(ulong)c * degree + r] = input[(ulong)r * total_cols + col0 + c];
}

// Scatters a column-major block of LDE columns into the bit-reversed
// row-major output matrix: natural row r lands at output row
// reverse_bits(r, log_lde).
kernel void p3_transpose_out_bitrev(
    const device ulong* col_lde [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& lde_size [[buffer(2)]],
    constant uint& log_lde [[buffer(3)]],
    constant uint& total_cols [[buffer(4)]],
    constant uint& col0 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint r = gid.x;
    if (r >= lde_size) {
        return;
    }
    uint c = gid.y;
    uint out_row = log_lde == 0 ? r : (reverse_bits(r) >> (32 - log_lde));
    out[(ulong)out_row * total_cols + col0 + c] = col_lde[(ulong)c * lde_size + r];
}

// Writes the state of the zero-padded, coset-shifted coefficient array as it
// exists after a bit-reversed gather plus the first `rate_bits` replication
// rounds of a classic DIT FFT: position i of column `col` holds
// shift^k * coeffs[col][k] with k = reverse_bits(i >> rate_bits, log_degree).
kernel void p3_ntt_prepare(
    const device ulong* coeffs [[buffer(0)]],
    const device ulong* shift_pows [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant uint& degree [[buffer(3)]],
    constant uint& lde_size [[buffer(4)]],
    constant uint& log_degree [[buffer(5)]],
    constant uint& rate_bits [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= lde_size) {
        return;
    }
    uint col = gid.y;
    uint k = log_degree == 0
        ? 0
        : (reverse_bits(i >> rate_bits) >> (32 - log_degree));
    ulong value = gl_mul(shift_pows[k], coeffs[(ulong)col * degree + k]);
    out[(ulong)col * lde_size + i] = value;
}

// One radix-2 decimation-in-time butterfly stage over every column:
// (u, v) := (u + w*v, u - w*v) with w = roots[j]. The final stage
// canonicalizes so downstream consumers see canonical representations.
kernel void p3_ntt_stage(
    device ulong* values [[buffer(0)]],
    const device ulong* roots [[buffer(1)]],
    constant uint& lde_size [[buffer(2)]],
    constant uint& log_half_m [[buffer(3)]],
    constant uint& canonicalize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint half_butterflies = lde_size >> 1;
    if (t >= half_butterflies) {
        return;
    }
    ulong colbase = (ulong)gid.y * lde_size;
    uint half_m = 1u << log_half_m;
    uint j = t & (half_m - 1u);
    uint base = ((t >> log_half_m) << (log_half_m + 1u));
    uint u_index = base + j;
    uint v_index = u_index + half_m;

    ulong u = values[colbase + u_index];
    ulong v = values[colbase + v_index];
    ulong w = roots[j];
    ulong wv = gl_mul(w, v);
    ulong out_u = gl_add(u, wv);
    ulong out_v = gl_sub(u, wv);
    if (canonicalize != 0u) {
        out_u = gl_canonicalize(out_u);
        out_v = gl_canonicalize(out_v);
    }
    values[colbase + u_index] = out_u;
    values[colbase + v_index] = out_v;
}

// Converts a forward-FFT output into IFFT coefficients:
// coeffs[i] = fft_out[(n - i) mod n] * n^{-1}, canonicalized.
kernel void p3_ifft_finalize(
    const device ulong* fft_out [[buffer(0)]],
    device ulong* coeffs [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& n_inv [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= n) {
        return;
    }
    ulong colbase = (ulong)gid.y * n;
    uint src = (n - i) & (n - 1u);
    coeffs[colbase + i] = gl_canonicalize(gl_mul(fft_out[colbase + src], n_inv));
}
