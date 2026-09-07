#include <metal_stdlib>
using namespace metal;

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
#if defined(POSEIDON2_NATIVE_ARITHMETIC_REFERENCE)
    ulong sum = a + b;
    ulong carry = sum < a;
    sum += carry * GOLDILOCKS_EPSILON;
    ulong carry2 = (carry != 0UL) && (sum < GOLDILOCKS_EPSILON);
    return sum + carry2 * GOLDILOCKS_EPSILON;
#else
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
#endif
}

inline ulong gl_sub(ulong a, ulong b) {
#if defined(POSEIDON2_NATIVE_ARITHMETIC_REFERENCE)
    ulong diff = a - b;
    ulong under = diff > a;
    diff -= under * GOLDILOCKS_EPSILON;
    ulong under2 = (under != 0UL) && (diff > (~0UL - GOLDILOCKS_EPSILON));
    return diff - under2 * GOLDILOCKS_EPSILON;
#else
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
#endif
}

// Goldilocks multiplication with 32-bit reduction after the native product.
// If low = l0 + l1*B and high = h0 + h1*B for B = 2^32, then
//   low + high*B^2 = (l0 - h0 - h1) + (l1 + h0)*B  (mod p).
// Normalizing those two signed base-B coefficients only needs uint
// add/subtract/carry operations; the only ulong multiplies left are the
// product's low and high halves.
inline ulong gl_mul(ulong a, ulong b) {
#if defined(POSEIDON2_NATIVE_ARITHMETIC_REFERENCE)
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    ulong high_high = high >> 32;
    ulong high_low = high & GOLDILOCKS_EPSILON;
    ulong reduced = low - high_high;
    if (reduced > low) {
        reduced -= GOLDILOCKS_EPSILON;
    }
    ulong addend = high_low * GOLDILOCKS_EPSILON;
    ulong result = reduced + addend;
    return result + (result < reduced) * GOLDILOCKS_EPSILON;
#else
    ulong low = a * b;
    ulong high = metal::mulhi(a, b);
    uint l0 = (uint)low;
    uint l1 = (uint)(low >> 32);
    uint h0 = (uint)high;
    uint h1 = (uint)(high >> 32);

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

    int top = (int)carry - (int)under;
    add_epsilon_u32(r0, r1, (uint)(top > 0));
    sub_epsilon_u32(r0, r1, (uint)(top < 0));
    return ((ulong)r1 << 32) | (ulong)r0;
#endif
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

inline ulong pow7(ulong value) {
    ulong value2 = gl_mul(value, value);
    ulong value4 = gl_mul(value2, value2);
    ulong value3 = gl_mul(value, value2);
    return gl_mul(value3, value4);
}

inline void mat4(thread ulong* values) {
    ulong x0 = values[0];
    ulong x1 = values[1];
    ulong x2 = values[2];
    ulong x3 = values[3];
    ulong t01 = gl_add(x0, x1);
    ulong t23 = gl_add(x2, x3);
    ulong total = gl_add(t01, t23);

    values[0] = gl_add(gl_add(total, t01), x1);
    values[1] = gl_add(gl_add(gl_add(total, x1), x2), x2);
    values[2] = gl_add(gl_add(total, t23), x3);
    values[3] = gl_add(gl_add(gl_add(total, x3), x0), x0);
}

inline void external_linear_layer(thread ulong state[12]) {
    mat4(state);
    mat4(state + 4);
    mat4(state + 8);

    ulong sums[4];
    for (uint i = 0; i < 4; ++i) {
        sums[i] = gl_add(gl_add(state[i], state[i + 4]), state[i + 8]);
    }
    for (uint i = 0; i < 12; ++i) {
        state[i] = gl_add(state[i], sums[i & 3]);
    }
}

inline ulong sum_state(thread const ulong state[12]) {
    ulong sum = 0;
    uint carries = 0;
    for (uint i = 0; i < 12; ++i) {
        ulong next = sum + state[i];
        carries += next < sum;
        sum = next;
    }
    return gl_add(sum, (ulong)carries * GOLDILOCKS_EPSILON);
}

inline void internal_linear_layer(thread ulong state[12], constant ulong* diagonal) {
    ulong sum = sum_state(state);
    for (uint i = 0; i < 12; ++i) {
        state[i] = gl_add(sum, gl_mul(state[i], diagonal[i]));
    }
}

// Parameter layout: 8 x 12 external constants, 22 internal constants,
// then the 12-element internal diagonal.
inline void poseidon2(thread ulong state[12], constant ulong* parameters) {
    constant ulong* external_constants = parameters;
    constant ulong* internal_constants = parameters + 96;
    constant ulong* diagonal = parameters + 118;

    external_linear_layer(state);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add(state[i], external_constants[round * 12 + i]));
        }
        external_linear_layer(state);
    }

    for (uint round = 0; round < 22; ++round) {
        state[0] = pow7(gl_add(state[0], internal_constants[round]));
        internal_linear_layer(state, diagonal);
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add(state[i], external_constants[round * 12 + i]));
        }
        external_linear_layer(state);
    }
}

kernel void poseidon2_hash_leaves(
    const device ulong* leaves [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    const device ulong* input = leaves + (ulong)gid * leaf_width;
    device ulong* output = hashes + (ulong)gid * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(input[i]);
        }
        for (; i < 4; ++i) {
            output[i] = 0;
        }
        return;
    }

    ulong state[12] = { 0 };
    for (uint offset = 0; offset < leaf_width; offset += 8) {
        uint chunk_size = min(8u, leaf_width - offset);
        for (uint i = 0; i < chunk_size; ++i) {
            state[i] = gl_canonicalize(input[offset + i]);
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

// Writes the state of the zero-padded, coset-shifted coefficient array as it
// exists after `reverse_index_bits_in_place` plus the first `rate_bits`
// replication rounds of `fft_classic`: position i of column `col` holds
// shift^k * coeffs[col][k] with k = reverse_bits(i >> rate_bits, log_degree).
kernel void ntt_prepare(
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

// One radix-2 decimation-in-time butterfly stage over every column, matching
// fft_classic: (u, v) := (u + w*v, u - w*v) with w = roots[j]. The final stage
// canonicalizes so downstream consumers see canonical representations.
kernel void ntt_stage(
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

// Converts a forward-FFT output into IFFT coefficients, matching plonky2's
// `ifft`: coeffs[i] = fft_out[(n - i) mod n] * n^{-1}, canonicalized so the
// CPU-side readback and the downstream LDE prepare see canonical values.
kernel void ifft_finalize(
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

kernel void poseidon2_hash_leaves_colmajor(
    const device ulong* leaves [[buffer(0)]],
    device ulong* hashes [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& leaf_width [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }

    // Input is column-major: column j occupies leaves[j * leaf_count..(j + 1) *
    // leaf_count] in natural row order, so adjacent threads read adjacent
    // addresses. The digest of natural row gid belongs to tree leaf
    // reverse_bits(gid), so outputs scatter by bit reversal.
    uint out_row = log_leaf_count == 0
        ? gid
        : (reverse_bits(gid) >> (32 - log_leaf_count));
    device ulong* output = hashes + (ulong)out_row * 4;
    if (leaf_width <= 4) {
        uint i = 0;
        for (; i < leaf_width; ++i) {
            output[i] = gl_canonicalize(leaves[(ulong)i * leaf_count + gid]);
        }
        for (; i < 4; ++i) {
            output[i] = 0;
        }
        return;
    }

    ulong state[12] = { 0 };
    for (uint offset = 0; offset < leaf_width; offset += 8) {
        uint chunk_size = min(8u, leaf_width - offset);
        for (uint i = 0; i < chunk_size; ++i) {
            state[i] = gl_canonicalize(leaves[(ulong)(offset + i) * leaf_count + gid]);
        }
        poseidon2(state, parameters);
    }
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}

kernel void poseidon2_hash_parents(
    const device ulong* children [[buffer(0)]],
    device ulong* parents [[buffer(1)]],
    constant ulong* parameters [[buffer(2)]],
    constant uint& parent_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= parent_count) {
        return;
    }

    const device ulong* input = children + (ulong)gid * 8;
    ulong state[12] = { 0 };
    for (uint i = 0; i < 8; ++i) {
        state[i] = input[i];
    }
    poseidon2(state, parameters);

    device ulong* output = parents + (ulong)gid * 4;
    for (uint i = 0; i < 4; ++i) {
        output[i] = gl_canonicalize(state[i]);
    }
}
