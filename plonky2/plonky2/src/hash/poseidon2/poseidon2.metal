#include <metal_stdlib>
using namespace metal;

constant ulong GOLDILOCKS_PRIME = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0xffffffffUL;

// Compile-time Poseidon2 round constants (same values as config.rs).
// File-scope constant arrays keep the round loops compact so register
// pressure stays near the tip kernels, while still removing device-buffer
// loads of the parameters table for every RC add.
constant ulong POSEIDON2_EXTERNAL_RC[8][12] = {
    { 0xd70193d17ab3b7d6UL, 0xa2c3662a78a9162bUL, 0x7a9fda827556ad44UL, 0xe8d5501818c99643UL, 0x4c7a8fced4d5fd38UL, 0x55ab38985c0c513dUL, 0x28a17bd016210b0bUL, 0x8f8277679ec32fa8UL, 0x768b3c3d68a460e9UL, 0x872a022eb559d941UL, 0xd1316dd4b3b97973UL, 0xa7b608e578321000UL },
    { 0x3fa02c87b0bee026UL, 0x7a38f0022e13c31eUL, 0x00c054f3c5e8d20dUL, 0x439f50f4bca7242fUL, 0x4d0938aa57cd517fUL, 0xb2e03ac5fb6b9a7dUL, 0xe29d1f4237bedca8UL, 0x05b7c844bc99b848UL, 0x91cc0b73f34e17edUL, 0x876e4427694bd755UL, 0x67002ae0725c612dUL, 0x05351f20e0b6315fUL },
    { 0x2e3b9ef5457eb60bUL, 0xd9ac17618c3783ddUL, 0x0807528ad8874bcfUL, 0xc78d546a455d2a0eUL, 0xf8b930c81e2481f0UL, 0x712707d8dff3b041UL, 0xdcb8c0aa0b9d34c3UL, 0x9baddbdf2ee3a468UL, 0x2dd16d50c5176c78UL, 0x89eac5cfbc075cd3UL, 0x2a741dea181587f3UL, 0x1a4d6aa85a113d84UL },
    { 0x4d736286a2387e34UL, 0x8bad5dfc4fcb3ee3UL, 0x84fbd03adb77c56aUL, 0x8d5cdd1a23ec53a2UL, 0x036f08f08fff28ecUL, 0xb717a3f4dbdfb443UL, 0x58a074b5509d645cUL, 0xf92bf834e4b87718UL, 0x1541c3a0baa5ac4bUL, 0x22149e6783e67692UL, 0x9be8b5d9e112476fUL, 0x41e0969f62babb76UL },
    { 0xbc585ad3b9443dbbUL, 0xf28dd3206975cbb1UL, 0xdd8815e53ca045e0UL, 0xde82c416b9e701baUL, 0xc5cb875233afa025UL, 0x7212697cd897ffa9UL, 0x67844790aa63cfd7UL, 0xdc0b9cfa97fe65c3UL, 0xe8fe091869a82070UL, 0x62902bb2e413c6d1UL, 0x29f9f5001fb84f57UL, 0xbe1014796ef5f8beUL },
    { 0x71feb53e9bdba19cUL, 0x251054f592ebb71cUL, 0xe1a57643a4bb284bUL, 0xa4ba6f87a45b739bUL, 0x2c1fcade0b958c49UL, 0xbbb424cda9a3e360UL, 0x2ca647354c5f3f54UL, 0xc9277b64d152e084UL, 0xdbc9ac97445eff17UL, 0x6f6cdf3198969f70UL, 0x1de29d14fa76d8f1UL, 0x73337458a8cc1d19UL },
    { 0xb87e775e2fb3ab23UL, 0xf166a1c7a565c80bUL, 0xb24be06f426c747fUL, 0xc281e8c49482ce00UL, 0x51974c3b3b726c2dUL, 0x87444cf8caf7d619UL, 0x7c362f827a580cedUL, 0x9567af14667647a0UL, 0xcbf0473cbec54e37UL, 0xe3209dedeff4f620UL, 0xd43ad94e45a4c4eeUL, 0x976981ee73f41768UL },
    { 0xef707a224e207258UL, 0x2fc779e10e6362eeUL, 0x29b5ee60ad8c891fUL, 0x96b37b39d8bfd667UL, 0x877df68a8b22e733UL, 0x5c41746f562c8d9fUL, 0x0c9d76751052b71aUL, 0xfb3465341bf1c087UL, 0xa0d14dc614d15eb1UL, 0xdc27d17136906fa6UL, 0x482e163b05ec397fUL, 0x0273a462992366efUL },
};
constant ulong POSEIDON2_INTERNAL_RC[22] = {
    0xa571418d95897b60UL, 0x8f32676574fcf6d3UL, 0x731102d4e3fb1bbeUL, 0x0330f08328a82d2bUL, 0x7f0449b6557f785dUL, 0x62f06210658dcbcbUL, 0xd5a98af9f89c458bUL, 0x77ec69083a346385UL, 0xef7ca48bbc27f890UL, 0x53e9652f61eac532UL, 0xa71c634abff4f0ccUL, 0xb16f5f0d7e28ea29UL, 0xc9dde31d0a003ab2UL, 0x2ddadf9775902533UL, 0xe4fa73fb16408b47UL, 0x90242ebc00d2ee59UL, 0xbb02dffd9f381982UL, 0xdea328364c50907cUL, 0x1395d3b924857cf8UL, 0x7d3ead0d5aec04e6UL, 0xc2f12be3fed74668UL, 0x0ba3c338f8c3d285UL,
};


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

// Multiplying by four is two doublings; a doubling is one field add.
inline ulong gl_add(ulong a, ulong b);
inline ulong gl_quadruple(ulong a) {
    ulong twice = gl_add(a, a);
    return gl_add(twice, twice);
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

// Adds a canonical field element `b < p` to an arbitrary u64 representative.
// If the native sum carries, its wrapped value is at most p - 2, because
// `a <= 2^64 - 1` and `b <= p - 1`. Adding EPSILON therefore cannot carry a
// second time. Round constants satisfy this bound, so their additions can
// omit the general path's unreachable second correction fold.
inline ulong gl_add_canonical_rhs(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint r0 = av.x + bv.x;
    uint carry0 = (uint)(r0 < av.x);
    uint r1 = av.y + bv.y;
    uint carry1 = (uint)(r1 < av.y);
    uint next = r1 + carry0;
    carry1 += (uint)(next < r1);
    r1 = next;

    uint old0 = r0;
    r0 -= carry1;
    r1 += carry1 & (uint)(old0 != 0);
    return ((ulong)r1 << 32) | (ulong)r0;
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

// Subtracts a canonical field element `b < p` from an arbitrary u64
// representative. On native underflow the wrapped difference is at least
// `2^64 - (p - 1) = 2^32`, so subtracting EPSILON cannot underflow again.
// Literal bit/radix constants satisfy this bound and need only one fold.
inline ulong gl_sub_canonical_rhs(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    uint r0 = av.x - bv.x;
    uint borrow0 = (uint)(r0 > av.x);
    uint r1 = av.y - bv.y;
    uint under = (uint)(r1 > av.y);
    uint next = r1 - borrow0;
    under += (uint)(next > r1);
    r1 = next;

    uint old0 = r0;
    r0 += under;
    r1 -= under & (uint)(old0 != 0xffffffffU);
    return ((ulong)r1 << 32) | (ulong)r0;
}

// Final step of the 128-bit Goldilocks reduction shared by gl_mul and
// gl_mul_add. On entry (r0, r1) are the low and high 32-bit limbs of the
// residue and `top` is its 2^64 weight, one of -1, 0, +1, so the value is
// r + top * 2^64, and 2^64 == EPSILON (mod p).
//
// Neither fold can leave the 64-bit range, so unlike gl_add this needs no
// second correction round. Writing the reduced product as
//     V = low + h0 * EPSILON - h1
// with low < 2^64 and h0, h1 < 2^32 bounds it by
//     -(2^32 - 1) <= V <= (2^64 - 1) + (2^32 - 1)^2 = 2^65 - 2^33.
// When top is +1 the residue is r = V - 2^64 <= 2^64 - 2^33, so r + EPSILON
// stays below 2^64 and the add cannot overflow. When top is -1 the residue is
// r = V + 2^64 >= 2^64 - 2^32 + 1 > EPSILON, so the subtract cannot borrow.
// One unconditional 64-bit add of top * EPSILON therefore replaces both
// add_epsilon_u32 and sub_epsilon_u32, including their unreachable second
// correction rounds. gl_add and gl_sub keep theirs: they know nothing about
// their operands and their second rounds are genuinely reachable.
inline ulong reduce_top(uint r0, uint r1, int top) {
    ulong r = ((ulong)r1 << 32) | (ulong)r0;
    return r + (ulong)(((long)top << 32) - (long)top);
}

// Full 128-bit product of two 64-bit operands, delivered as four 32-bit limbs.
//
// `a * b` and `metal::mulhi(a, b)` are lowered as two independent 64-bit
// expansions that each rebuild the 32x32 partial products they need; the
// backend does not share them. Computing the four products once and assembling
// only the limbs the reduction actually consumes removes that duplication and
// the 64-bit pack/unpack around it.
//
// With B = 2^32 and a = a1*B + a0, b = b1*B + b0:
//   t = p01 + (p00 >> 32) <= (B-1)^2 + (B-1) = B^2 - B, so t cannot wrap.
//   m = t + p10 <= 2B^2 - 3B + 1 wraps at most once; `carry` is that 65th bit.
//   low  = (m << 32) | (uint)p00 -> l0 = (uint)p00, l1 = (uint)m.
//   high = p11 + (m >> 32) + carry * B, which is < B^2, so h1 cannot wrap.
//
// The operands are split with a `uint2` bitcast rather than shift-and-truncate.
// Both name the same two halves on a little-endian target, but `(uint)(a >> 32)`
// asks the backend for a 64-bit shift it then has to narrow, while the bitcast
// is a pure reinterpretation of the register pair the value already occupies.
// This is the multiply every hash lane runs four times per S-box, and dropping
// those two shifts is worth ~9% of the leaf-hash kernel.
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
#endif
}

inline ulong gl_canonicalize(ulong value) {
    return value >= GOLDILOCKS_PRIME ? value - GOLDILOCKS_PRIME : value;
}

// A lazy value is (lo, hi) representing hi * 2^32 + lo, with both halves held
// in full 64-bit registers. Splitting the operand at the 32-bit boundary makes
// addition carry-free: every accumulator below sums field elements with
// coefficients totalling at most 28, so each half stays under
// 28 * (2^32 - 1) < 2^37 and neither can overflow. lazy_add is then two
// independent 64-bit adds; the single-word (v, c) form needed a third
// operation per add to recover the carry a 64-bit add cannot report to MSL.
// One materialize per element per layer still replaces the per-add reduction
// chains, and every materialized output is an ordinary u64 representative, so
// downstream arithmetic and gl_canonicalize see the same canonical field
// values as the strict per-op path.
struct lazy_t {
    ulong lo;
    ulong hi;
};

inline lazy_t lazy_of(ulong v) {
    return { v & GOLDILOCKS_EPSILON, v >> 32 };
}

inline lazy_t lazy_add(lazy_t a, lazy_t b) {
    return { a.lo + b.lo, a.hi + b.hi };
}

// Collapses (lo, hi) to an ordinary 64-bit representative. Splitting both
// halves at the 32-bit boundary as
//   lo = lh * 2^32 + ll,  hi = hh * 2^32 + hl   (lh, hh < 2^5)
// rewrites the value as
//   hi * 2^32 + lo == hh * 2^64 + (hl + lh) * 2^32 + ll,
// so with u = (hl + lh) mod 2^32 and e = hh + carry(hl + lh) <= 32 and
// 2^64 == EPSILON == 2^32 - 1 (mod p) the residue is exactly
//   (u + e) * 2^32 + (ll - e).
// Both coefficients are single 32-bit adds whose carry out and borrow out are
// the 2^64 weight of the result, which is the shape reduce_top already folds:
// when that weight is +1 the high limb is at most 32, because u + e >= 2^64
// forces u >= 2^32 - 32, so the EPSILON add cannot wrap; when it is -1 the high
// limb is 0xffffffff, so the subtract cannot borrow. Everything above the four
// limb extractions is 32-bit work, and the wide compares plus the trailing
// correction round of the 64-bit form are gone.
inline ulong lazy_materialize(lazy_t a) {
    uint ll = (uint)a.lo;
    uint lh = (uint)(a.lo >> 32);
    uint hl = (uint)a.hi;
    uint hh = (uint)(a.hi >> 32);
    uint u = hl + lh;
    uint e = hh + (uint)(u < hl);
    uint r0 = ll - e;
    uint borrow = (uint)(r0 > ll);
    uint r1 = u + e;
    uint carry = (uint)(r1 < u);
    uint next = r1 - borrow;
    uint under = (uint)(next > r1);
    r1 = next;
    return reduce_top(r0, r1, (int)carry - (int)under);
}

// r = a * b + addend (mod p): the 128-bit product absorbs the addend before
// one shared reduction, deleting the separate post-multiply gl_add. mulhi is
// at most 2^64 - 2, so the absorbed carry cannot overflow the high word. The
// reduction below is byte-identical to gl_mul's.
inline ulong gl_mul_add(ulong a, ulong b, ulong addend) {
    uint l0;
    uint l1;
    uint h0;
    uint h1;
    mul_128(a, b, l0, l1, h0, h1);

    // Absorb the addend into the low limbs and propagate its single carry into
    // h0. h1 cannot wrap: the 128-bit product of two 64-bit operands is at most
    // 2^128 - 2^65 + 1, so h1 == 0xffffffff forces h0 <= 0xfffffffe.
    uint d0 = (uint)addend;
    uint d1 = (uint)(addend >> 32);
    uint s0 = l0 + d0;
    uint c0 = (uint)(s0 < l0);
    uint s1 = l1 + d1;
    uint c1 = (uint)(s1 < l1);
    uint s1b = s1 + c0;
    c1 += (uint)(s1b < s1);
    l0 = s0;
    l1 = s1b;
    uint hh0 = h0 + c1;
    h1 += (uint)(hh0 < h0);
    h0 = hh0;

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

// x^7 by the addition chain 1 -> 2 -> 3 -> 6 -> 7. Every length-four chain for
// x^7 costs the same four multiplies, but this one keeps only `value` and one
// running power live at a time, and three of the four multiplies take `value`
// or a square as an operand, so the backend reuses one operand's limb split
// instead of re-splitting a fresh pair. The 2/4/3/7 chain is one step shallower
// and measured consistently slower for both reasons.
inline ulong pow7(ulong value) {
    ulong value2 = gl_mul(value, value);
    ulong value3 = gl_mul(value, value2);
    ulong value6 = gl_mul(value3, value3);
    return gl_mul(value, value6);
}

inline void mat4(thread lazy_t* values) {
    lazy_t x0 = values[0];
    lazy_t x1 = values[1];
    lazy_t x2 = values[2];
    lazy_t x3 = values[3];
    lazy_t t01 = lazy_add(x0, x1);
    lazy_t t23 = lazy_add(x2, x3);
    lazy_t total = lazy_add(t01, t23);

    values[0] = lazy_add(lazy_add(total, t01), x1);
    values[1] = lazy_add(lazy_add(lazy_add(total, x1), x2), x2);
    values[2] = lazy_add(lazy_add(total, t23), x3);
    values[3] = lazy_add(lazy_add(lazy_add(total, x3), x0), x0);
}

inline void external_linear_layer(thread ulong state[12]) {
    lazy_t lazy[12];
    for (uint i = 0; i < 12; ++i) {
        lazy[i] = lazy_of(state[i]);
    }
    mat4(lazy);
    mat4(lazy + 4);
    mat4(lazy + 8);

    lazy_t sums[4];
    for (uint i = 0; i < 4; ++i) {
        sums[i] = lazy_add(lazy_add(lazy[i], lazy[i + 4]), lazy[i + 8]);
    }
    for (uint i = 0; i < 12; ++i) {
        state[i] = lazy_materialize(lazy_add(lazy[i], sums[i & 3]));
    }
}

// Same carry-free split as lazy_t: twelve operands keep both halves below
// 12 * (2^32 - 1) < 2^36.
inline ulong sum_state(thread const ulong state[12]) {
    lazy_t sum = lazy_of(state[0]);
    for (uint i = 1; i < 12; ++i) {
        sum = lazy_add(sum, lazy_of(state[i]));
    }
    return lazy_materialize(sum);
}

inline void internal_linear_layer(thread ulong state[12]) {
    ulong sum = sum_state(state);
    // Poseidon2's internal diagonal is fixed by the hash configuration. Spell
    // it as immediates so the Metal compiler can specialize the constant
    // operand of all 264 internal-round multiplications per permutation.
    state[0] = gl_mul_add(state[0], 0xc3b6c08e23ba9300UL, sum);
    state[1] = gl_mul_add(state[1], 0xd84b5de94a324fb6UL, sum);
    state[2] = gl_mul_add(state[2], 0x0d0c371c5b35b84fUL, sum);
    state[3] = gl_mul_add(state[3], 0x7964f570e7188037UL, sum);
    state[4] = gl_mul_add(state[4], 0x5daf18bbd996604bUL, sum);
    state[5] = gl_mul_add(state[5], 0x6743bc47b9595257UL, sum);
    state[6] = gl_mul_add(state[6], 0x5528b9362c59bb70UL, sum);
    state[7] = gl_mul_add(state[7], 0xac45e25b7127b68bUL, sum);
    state[8] = gl_mul_add(state[8], 0xa2077d7dfbb606b5UL, sum);
    state[9] = gl_mul_add(state[9], 0xf3faac6faee378aeUL, sum);
    state[10] = gl_mul_add(state[10], 0x0c6388b51545e883UL, sum);
    state[11] = gl_mul_add(state[11], 0xd27dbb6944917b60UL, sum);
}

// Parameter layout: 8 x 12 external constants, 22 internal constants,
// then the 12-element internal diagonal.
// Parameter buffer remains in the ABI for host binding; sponge arithmetic uses
// compile-time constant tables so RC adds do not depend on device buffer loads.
// Round loops stay compact (unlike full unroll) to preserve occupancy.
inline void poseidon2(thread ulong state[12], constant ulong* /*parameters*/) {
    external_linear_layer(state);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add_canonical_rhs(state[i], POSEIDON2_EXTERNAL_RC[round][i]));
        }
        external_linear_layer(state);
    }

    for (uint round = 0; round < 22; ++round) {
        state[0] = pow7(gl_add_canonical_rhs(state[0], POSEIDON2_INTERNAL_RC[round]));
        internal_linear_layer(state);
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(gl_add_canonical_rhs(state[i], POSEIDON2_EXTERNAL_RC[round][i]));
        }
        external_linear_layer(state);
    }
}

inline ulong poseidon2_gate_wire(
    const device ulong* wires,
    uint column,
    uint rows,
    uint row) {
    return wires[(ulong)column * rows + row];
}

inline void poseidon2_gate_emit(
    ulong constraint,
    constant ulong* alpha_powers,
    thread ulong accumulators[2],
    thread uint& constraint_index) {
    accumulators[0] =
        gl_mul_add(constraint, alpha_powers[constraint_index], accumulators[0]);
    accumulators[1] =
        gl_mul_add(constraint, alpha_powers[123 + constraint_index], accumulators[1]);
    ++constraint_index;
}

// Evaluates the filtered Poseidon2Gate contribution to both quotient
// challenges at one natural-order quotient-domain point. The alpha powers
// already include the prefix occupied by the non-gate vanishing terms, so
// these two values can be added to the CPU evaluator after it has reduced the
// permutation argument.
kernel void poseidon2_gate_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant ulong* alpha_powers [[buffer(4)]],
    constant uint& lde_rows [[buffer(5)]],
    constant uint& quotient_rows [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& selector_column [[buffer(8)]],
    constant uint& gate_index [[buffer(9)]],
    constant uint& group_start [[buffer(10)]],
    constant uint& group_end [[buffer(11)]],
    constant uint& include_unused_selector [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
    ulong filter = 1;
    for (uint i = group_start; i < group_end; ++i) {
        if (i != gate_index) {
            filter = gl_mul(filter, gl_sub((ulong)i, selector));
        }
    }
    if (include_unused_selector != 0u) {
        filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
    }

    // parameters buffer kept for ABI; RCs from compile-time tables.
    ulong accumulators[2] = { 0, 0 };
    uint constraint_index = 0;

    ulong swap = poseidon2_gate_wire(wires, 24, lde_rows, source_row);
    poseidon2_gate_emit(
        gl_mul(swap, gl_sub_canonical_rhs(swap, 1)),
        alpha_powers,
        accumulators,
        constraint_index);

    ulong state[12];
    for (uint i = 0; i < 4; ++i) {
        ulong lhs = poseidon2_gate_wire(wires, i, lde_rows, source_row);
        ulong rhs = poseidon2_gate_wire(wires, i + 4, lde_rows, source_row);
        ulong delta = poseidon2_gate_wire(wires, 25 + i, lde_rows, source_row);
        poseidon2_gate_emit(
            gl_sub(gl_mul(swap, gl_sub(rhs, lhs)), delta),
            alpha_powers,
            accumulators,
            constraint_index);
        state[i] = gl_add(lhs, delta);
        state[i + 4] = gl_sub(rhs, delta);
    }
    for (uint i = 8; i < 12; ++i) {
        state[i] = poseidon2_gate_wire(wires, i, lde_rows, source_row);
    }

    external_linear_layer(state);

    for (uint round = 0; round < 4; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = gl_add_canonical_rhs(state[i], POSEIDON2_EXTERNAL_RC[round][i]);
        }
        if (round != 0) {
            uint saved_start = 29 + (round - 1) * 12;
            for (uint i = 0; i < 12; ++i) {
                ulong saved = poseidon2_gate_wire(
                    wires,
                    saved_start + i,
                    lde_rows,
                    source_row);
                poseidon2_gate_emit(
                    gl_sub(state[i], saved),
                    alpha_powers,
                    accumulators,
                    constraint_index);
                state[i] = saved;
            }
        }
        for (uint i = 0; i < 12; ++i) {
            state[i] = pow7(state[i]);
        }
        external_linear_layer(state);
    }

    for (uint round = 0; round < 22; ++round) {
        ulong saved = poseidon2_gate_wire(wires, 65 + round, lde_rows, source_row);
        poseidon2_gate_emit(
            gl_sub(gl_add_canonical_rhs(state[0], POSEIDON2_INTERNAL_RC[round]), saved),
            alpha_powers,
            accumulators,
            constraint_index);
        state[0] = pow7(saved);
        internal_linear_layer(state);
    }

    for (uint round = 4; round < 8; ++round) {
        for (uint i = 0; i < 12; ++i) {
            state[i] = gl_add_canonical_rhs(state[i], POSEIDON2_EXTERNAL_RC[round][i]);
        }
        uint saved_start = 87 + (round - 4) * 12;
        for (uint i = 0; i < 12; ++i) {
            ulong saved = poseidon2_gate_wire(
                wires,
                saved_start + i,
                lde_rows,
                source_row);
            poseidon2_gate_emit(
                gl_sub(state[i], saved),
                alpha_powers,
                accumulators,
                constraint_index);
            state[i] = pow7(saved);
        }
        external_linear_layer(state);
    }

    for (uint i = 0; i < 12; ++i) {
        ulong expected = poseidon2_gate_wire(wires, 12 + i, lde_rows, source_row);
        poseidon2_gate_emit(
            gl_sub(state[i], expected),
            alpha_powers,
            accumulators,
            constraint_index);
    }

    output[(ulong)gid * 2] = gl_canonicalize(gl_mul(filter, accumulators[0]));
    output[(ulong)gid * 2 + 1] = gl_canonicalize(gl_mul(filter, accumulators[1]));
}

// Evaluates the no-lookup permutation partial-product constraints at one
// quotient-domain point. Alpha rows 0 and 1 are the two L_0 constraints and
// remain on the CPU; challenge-major partial-product rows start at power 2.
kernel void permutation_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants_sigmas [[buffer(1)]],
    const device ulong* zs_partial_products [[buffer(2)]],
    const device ulong* shifted_points [[buffer(3)]],
    device ulong* output [[buffer(4)]],
    constant ulong* alpha_powers [[buffer(5)]],
    constant ulong* challenges [[buffer(6)]],
    constant uint& lde_rows [[buffer(7)]],
    constant uint& quotient_rows [[buffer(8)]],
    constant uint& step [[buffer(9)]],
    constant uint& next_step [[buffer(10)]],
    constant uint& sigma_start [[buffer(11)]],
    constant uint& num_routed_wires [[buffer(12)]],
    constant uint& num_partial_products [[buffer(13)]],
    constant uint& chunk_size [[buffer(14)]],
    constant uint& alpha_stride [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    uint next_gid = (gid + next_step) & (quotient_rows - 1u);
    uint next_source_row = next_gid * step;
    uint num_chunks = num_partial_products + 1u;
    ulong x = shifted_points[gid];
    ulong totals[2] = { 0, 0 };

    ulong beta0 = challenges[0];
    ulong beta1 = challenges[1];
    ulong gamma0 = challenges[2];
    ulong gamma1 = challenges[3];
    for (uint chunk = 0; chunk < num_chunks; ++chunk) {
        uint j_start = chunk * chunk_size;
        uint j_end = min(j_start + chunk_size, num_routed_wires);
        ulong wire = wires[(ulong)j_start * lde_rows + source_row];
        ulong sigma = constants_sigmas[
            (ulong)(sigma_start + j_start) * lde_rows + source_row];
        ulong beta_k0 = challenges[4u + j_start];
        ulong beta_k1 = challenges[4u + num_routed_wires + j_start];
        ulong numerator0 = gl_add(gl_mul_add(beta_k0, x, wire), gamma0);
        ulong denominator0 = gl_add(gl_mul_add(beta0, sigma, wire), gamma0);
        ulong numerator1 = gl_add(gl_mul_add(beta_k1, x, wire), gamma1);
        ulong denominator1 = gl_add(gl_mul_add(beta1, sigma, wire), gamma1);
        for (uint j = j_start + 1u; j < j_end; ++j) {
            wire = wires[(ulong)j * lde_rows + source_row];
            sigma = constants_sigmas[
                (ulong)(sigma_start + j) * lde_rows + source_row];
            beta_k0 = challenges[4u + j];
            beta_k1 = challenges[4u + num_routed_wires + j];
            numerator0 = gl_mul(
                numerator0, gl_add(gl_mul_add(beta_k0, x, wire), gamma0));
            denominator0 = gl_mul(
                denominator0, gl_add(gl_mul_add(beta0, sigma, wire), gamma0));
            numerator1 = gl_mul(
                numerator1, gl_add(gl_mul_add(beta_k1, x, wire), gamma1));
            denominator1 = gl_mul(
                denominator1, gl_add(gl_mul_add(beta1, sigma, wire), gamma1));
        }

        uint previous_column0 = chunk == 0u ? 0u : 1u + chunk;
        ulong previous0 = zs_partial_products[
            (ulong)previous_column0 * lde_rows + source_row];
        ulong next0;
        if (chunk < num_partial_products) {
            next0 = zs_partial_products[(ulong)(2u + chunk) * lde_rows + source_row];
        } else {
            next0 = zs_partial_products[next_source_row];
        }
        ulong term0 = gl_sub(
            gl_mul(previous0, numerator0), gl_mul(next0, denominator0));
        uint alpha_index0 = 2u + chunk;
        totals[0] = gl_mul_add(term0, alpha_powers[alpha_index0], totals[0]);
        totals[1] = gl_mul_add(
            term0, alpha_powers[alpha_stride + alpha_index0], totals[1]);

        uint previous_column1 = chunk == 0u
            ? 1u
            : 1u + num_partial_products + chunk;
        ulong previous1 = zs_partial_products[
            (ulong)previous_column1 * lde_rows + source_row];
        ulong next1;
        if (chunk < num_partial_products) {
            uint next_column1 = 2u + num_partial_products + chunk;
            next1 = zs_partial_products[(ulong)next_column1 * lde_rows + source_row];
        } else {
            next1 = zs_partial_products[(ulong)lde_rows + next_source_row];
        }
        ulong term1 = gl_sub(
            gl_mul(previous1, numerator1), gl_mul(next1, denominator1));
        uint alpha_index1 = 2u + num_chunks + chunk;
        totals[0] = gl_mul_add(term1, alpha_powers[alpha_index1], totals[0]);
        totals[1] = gl_mul_add(
            term1, alpha_powers[alpha_stride + alpha_index1], totals[1]);
    }

    output[(ulong)gid * 2u] = gl_canonicalize(totals[0]);
    output[(ulong)gid * 2u + 1u] = gl_canonicalize(totals[1]);
}

// Deferred-reduction accumulator for one alpha-weighted constraint sum.
//
// `gl_mul_add` folds every product back to a single 64-bit representative, but
// a gate family's running sum never looks at those intermediates: reduction mod
// p is a ring homomorphism, so summing the raw 128-bit products and reducing
// once is congruent to summing the reduced products. `mul_128` already delivers
// each product as four base-2^32 limbs (l0, l1, h0, h1), and adding those limbs
// into four 64-bit registers is carry-free -- exactly the split `lazy_t`
// already uses, here as two pairs: (l0, l1) with weight 2^0 and (h0, h1) with
// weight 2^64.
//
// Headroom. Every limb is at most 2^32 - 1, so after n accumulations each of
// the four sums is at most n * (2^32 - 1) < n * 2^32, and the widest quantity
// `alpha_acc_materialize` forms is low.hi + high.lo < 2 * n * 2^32. That stays
// inside the 2^64 - 2^32 bound the fold needs for every n < 2^31 - 1, so no
// intermediate fold is ever required: n is a family's constraint count, which
// indexes `alpha_powers` and is at most `alpha_stride` (136 in the largest
// production shape, giving sums below 2^40 -- twenty-four spare bits).
struct alpha_acc_t {
    lazy_t low;
    lazy_t high;
};

inline void alpha_acc_mul_add(thread alpha_acc_t& acc, ulong a, ulong b) {
    uint l0;
    uint l1;
    uint h0;
    uint h1;
    mul_128(a, b, l0, l1, h0, h1);
    acc.low.lo += l0;
    acc.low.hi += l1;
    acc.high.lo += h0;
    acc.high.hi += h1;
}

// Collapses the four limb sums to an ordinary 64-bit representative. With
// 2^64 == EPSILON and 2^96 == 2^32 * EPSILON == -1 (mod p) the accumulated
// value is
//   low.lo + low.hi * 2^32 + high.lo * 2^64 + high.hi * 2^96
//     == (low.lo - high.lo - high.hi) + (low.hi + high.lo) * 2^32   (mod p).
// The positive part is precisely `lazy_t`'s (lo, hi) shape, so
// `lazy_materialize` folds it; its proof needs only that the high word leave
// room for `e = hh + carry` not to wrap, i.e. low.hi + high.lo < 2^64 - 2^32,
// which the headroom note above establishes. The subtracted part is below
// 2^33 * n <= 2^41 < p and the minuend is an arbitrary value < 2^64, which is
// the operand range `gl_sub` is written for, so the result is a 64-bit
// representative of the exact residue. Every consumer downstream
// (`gl_mul_add` by the filter, then `gl_canonicalize`) is residue-exact for
// any 64-bit representative, so the kernel's output is bit-identical to the
// per-constraint reduction it replaces.
inline ulong alpha_acc_materialize(alpha_acc_t acc) {
    lazy_t positive = { acc.low.lo, acc.low.hi + acc.high.lo };
    return gl_sub(lazy_materialize(positive), acc.high.lo + acc.high.hi);
}

inline void range_check_gate_emit(
    ulong constraint,
    constant ulong* alpha_powers,
    uint alpha_stride,
    thread alpha_acc_t accumulators[2],
    uint constraint_index) {
    alpha_acc_mul_add(
        accumulators[0], constraint, alpha_powers[constraint_index]);
    alpha_acc_mul_add(
        accumulators[1], constraint, alpha_powers[alpha_stride + constraint_index]);
}

// Reducing emission, for the two quintic families only. Deferring costs four
// extra live registers per challenge for the whole family body, and the
// quintic gates are the only kinds whose own body is register-hungry enough to
// care: the schoolbook product keeps nineteen field elements live and the
// squaring twenty. Measured on the d16-heavy shape, deferring inside them is a
// net loss -- dropping both families from the shape turns the kernel delta from
// +0.4% to -2.3% -- while every other kind gains. They are also the kinds that
// gain least: five and two multiplies per constraint respectively make the
// accumulate a small share of their work. `alpha_acc_of` hands the family
// result back to the shared tail, so the merge below is kind-agnostic.
inline void range_check_gate_emit_strict(
    ulong constraint,
    constant ulong* alpha_powers,
    uint alpha_stride,
    thread ulong accumulators[2],
    uint constraint_index) {
    accumulators[0] =
        gl_mul_add(constraint, alpha_powers[constraint_index], accumulators[0]);
    accumulators[1] = gl_mul_add(
        constraint, alpha_powers[alpha_stride + constraint_index], accumulators[1]);
}

// Lifts an ordinary 64-bit representative into the deferred form: its two
// halves are the (l0, l1) limb sums of a single product with a zero high half.
inline alpha_acc_t alpha_acc_of(ulong value) {
    return { lazy_of(value), { 0, 0 } };
}

// Each RangeCheck metadata record is ten uints:
//   selector column, gate index, group start/end, include UNUSED selector,
//   operation count, base-4 limbs per operation, final-limb range (2 or 4),
//   then two unused words that keep both record kinds the same stride.
// It is followed by promoted-family records with the same five selector words,
// then kind (arithmetic=0, subtraction=1, add-many=2, byte-decomposition=3,
// quintic-multiplication=4, quintic-squaring=5, random-access=6,
// exponentiation=7, equality=8, reducing=9, base-addition=10, base-sum=11,
// selection=12), operation or copy count, and
// three explicit kind words. Random access uses the final words for index
// bits, extra constants, and the raw constant-column base; equality carries
// its constants column and reducing its extension-coefficient flag in the
// addend-count word.
// The result-limb count is what makes the subtraction and add-many branches
// width-generic: a `2 * result_limbs`-bit word recomposes from that many
// base-4 limbs and its overflow weight is `1 << (2 * result_limbs)`, which
// covers the 16-, 32- and 48-bit production gates with one code path.
// Every gate starts again at constraint row zero. This matches the CPU's
// shared row accumulator: reducing each filtered gate locally with the same
// alpha powers and then adding the results is linear in those row values.

// Select within one contiguous eight-item block using the low three index bits.
// Keeping only this block and the at-most-eight block results private avoids a
// 64-word private array for the audited six-bit gate.
inline ulong random_access_select_8(
    const device ulong* wires,
    uint lde_rows,
    uint source_row,
    ulong list_base,
    ulong bit_base,
    uint block) {
    ulong items[8];
    for (uint i = 0; i < 8u; ++i) {
        ulong column = list_base + (ulong)block * 8u + i;
        items[i] = wires[column * lde_rows + source_row];
    }
    uint level_size = 8u;
    for (uint level = 0; level < 3u; ++level) {
        ulong b = wires[(bit_base + level) * lde_rows + source_row];
        for (uint k = 0; k < level_size / 2u; ++k) {
            ulong x = items[2u * k];
            ulong y = items[2u * k + 1u];
            items[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
        }
        level_size /= 2u;
    }
    return items[0];
}

kernel void range_check_gate_quotient(
    const device ulong* wires [[buffer(0)]],
    const device ulong* constants [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant ulong* alpha_powers [[buffer(3)]],
    constant uint* metadata [[buffer(4)]],
    constant uint& lde_rows [[buffer(5)]],
    constant uint& quotient_rows [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& alpha_stride [[buffer(8)]],
    constant uint& range_count [[buffer(9)]],
    constant uint& u32_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= quotient_rows) {
        return;
    }

    uint source_row = gid * step;
    ulong total[2] = { 0, 0 };
    for (uint range_index = 0; range_index < range_count; ++range_index) {
        constant uint* spec = metadata + range_index * 10u;
        uint selector_column = spec[0];
        uint gate_index = spec[1];
        uint group_start = spec[2];
        uint group_end = spec[3];
        uint include_unused_selector = spec[4];
        uint num_ops = spec[5];
        uint num_aux = spec[6];
        uint final_limb_range = spec[7];

        ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
        for (uint op = 0; op < num_ops; ++op) {
            ulong input = wires[(ulong)op * lde_rows + source_row];
            ulong aux_base = (ulong)num_ops + (ulong)num_aux * op;
            ulong computed = wires[(aux_base + num_aux - 1u) * lde_rows + source_row];
            for (uint remaining = num_aux - 1u; remaining > 0u; --remaining) {
                uint j = remaining - 1u;
                ulong limb = wires[(aux_base + j) * lde_rows + source_row];
                computed = gl_add(gl_quadruple(computed), limb);
            }
            range_check_gate_emit(
                gl_sub(computed, input),
                alpha_powers,
                alpha_stride,
                gate_accumulators,
                constraint_index++);

            for (uint j = 0; j < num_aux; ++j) {
                ulong x = wires[(aux_base + j) * lde_rows + source_row];
                ulong constraint;
                if (j + 1u == num_aux && final_limb_range == 2u) {
                    constraint = gl_mul(x, gl_sub_canonical_rhs(x, 1));
                } else {
                    // x(x-1)(x-2)(x-3) = y(y+2), y = x(x-3),
                    // exactly the production CPU specialization.
                    ulong y = gl_mul(x, gl_sub_canonical_rhs(x, 3));
                    constraint = gl_mul(y, gl_add_canonical_rhs(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        }

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }

    constant uint* u32_metadata = metadata + range_count * 10u;
    for (uint u32_index = 0; u32_index < u32_count; ++u32_index) {
        constant uint* spec = u32_metadata + u32_index * 10u;
        uint selector_column = spec[0];
        uint gate_index = spec[1];
        uint group_start = spec[2];
        uint group_end = spec[3];
        uint include_unused_selector = spec[4];
        uint kind = spec[5];
        uint num_ops = spec[6];
        uint num_addends = spec[7];
        uint result_limbs = spec[8];
        uint num_carry_limbs = spec[9];
        // Overflow weight of the recomposed word: 2^16, 2^32 or 2^48.
        ulong word_base = 1UL << (2u * result_limbs);

        ulong selector = constants[(ulong)selector_column * lde_rows + source_row];
        ulong filter = 1;
        for (uint i = group_start; i < group_end; ++i) {
            if (i != gate_index) {
                filter = gl_mul(filter, gl_sub((ulong)i, selector));
            }
        }
        if (include_unused_selector != 0u) {
            filter = gl_mul(filter, gl_sub(0xffffffffUL, selector));
        }

        alpha_acc_t gate_accumulators[2] = {
            { { 0, 0 }, { 0, 0 } },
            { { 0, 0 }, { 0, 0 } },
        };
        uint constraint_index = 0;
        if (kind == 0u) {
            // U32ArithmeticGate: six routed words followed by 32 base-4
            // output limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 6u;
                ulong multiplicand_0 = wires[(routed_base + 0u) * lde_rows + source_row];
                ulong multiplicand_1 = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong addend = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong output_low = wires[(routed_base + 3u) * lde_rows + source_row];
                ulong output_high = wires[(routed_base + 4u) * lde_rows + source_row];
                ulong inverse = wires[(routed_base + 5u) * lde_rows + source_row];

                ulong high_diff = gl_sub(0xffffffffUL, output_high);
                ulong high_not_max = gl_sub_canonical_rhs(gl_mul(inverse, high_diff), 1);
                range_check_gate_emit(
                    gl_mul(high_not_max, output_low),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                ulong computed = gl_add(gl_mul(multiplicand_0, multiplicand_1), addend);
                ulong combined = gl_add(gl_mul(output_high, 4294967296UL), output_low);
                range_check_gate_emit(
                    gl_sub(combined, computed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 6u + (ulong)op * 32u;
                ulong combined_low = 0;
                ulong combined_high = 0;
                for (uint remaining = 32u; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub_canonical_rhs(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add_canonical_rhs(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                    if (j < 16u) {
                        combined_low = gl_add(gl_quadruple(combined_low), x);
                    } else {
                        combined_high = gl_add(gl_quadruple(combined_high), x);
                    }
                }
                range_check_gate_emit(
                    gl_sub(combined_low, output_low),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(combined_high, output_high),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 1u) {
            // U16/U32/U48 SubtractionGate: five routed words followed by
            // `result_limbs` base-4 result limbs per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 5u;
                ulong input_x = wires[(routed_base + 0u) * lde_rows + source_row];
                ulong input_y = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong input_borrow = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong output_result = wires[(routed_base + 3u) * lde_rows + source_row];
                ulong output_borrow = wires[(routed_base + 4u) * lde_rows + source_row];
                ulong result_initial = gl_sub(gl_sub(input_x, input_y), input_borrow);
                ulong borrowed = gl_add(
                    result_initial,
                    gl_mul(word_base, output_borrow));
                range_check_gate_emit(
                    gl_sub(output_result, borrowed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                ulong limb_base = (ulong)num_ops * 5u + (ulong)op * result_limbs;
                ulong recomposed = 0;
                for (uint remaining = result_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub_canonical_rhs(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add_canonical_rhs(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                    recomposed = gl_add(gl_quadruple(recomposed), x);
                }
                range_check_gate_emit(
                    gl_sub(recomposed, output_result),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_mul(output_borrow, gl_sub(1, output_borrow)),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 2u) {
            // U16/U32 AddManyGate: num_addends inputs, carry/result/output-carry,
            // then `result_limbs` result and `num_carry_limbs` carry base-4
            // limbs per operation.
            uint routed_per_op = num_addends + 3u;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong computed = wires[(routed_base + num_addends) * lde_rows + source_row];
                for (uint j = 0; j < num_addends; ++j) {
                    computed = gl_add(
                        computed,
                        wires[(routed_base + j) * lde_rows + source_row]);
                }
                ulong output_result =
                    wires[(routed_base + num_addends + 1u) * lde_rows + source_row];
                ulong output_carry =
                    wires[(routed_base + num_addends + 2u) * lde_rows + source_row];
                ulong combined = gl_add(gl_mul(output_carry, word_base), output_result);
                range_check_gate_emit(
                    gl_sub(combined, computed),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                uint total_limbs = result_limbs + num_carry_limbs;
                ulong limb_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * total_limbs;
                ulong combined_result = 0;
                ulong combined_carry = 0;
                for (uint remaining = total_limbs; remaining > 0u; --remaining) {
                    uint j = remaining - 1u;
                    ulong x = wires[(limb_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub_canonical_rhs(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add_canonical_rhs(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                    if (j < result_limbs) {
                        combined_result = gl_add(gl_quadruple(combined_result), x);
                    } else {
                        combined_carry = gl_add(gl_quadruple(combined_carry), x);
                    }
                }
                range_check_gate_emit(
                    gl_sub(combined_result, output_result),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(combined_carry, output_carry),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 3u) {
            // ByteDecompositionGate: per operation, one routed sum wire and
            // `num_addends` routed byte wires (the metadata word carries the
            // byte count for this kind), then four base-4 aux limbs per
            // byte. Constraint order matches the CPU gate exactly: aux range
            // products in ascending wire order, one base-4 recomposition per
            // byte, then the base-256 byte-to-sum recomposition.
            uint num_limbs = num_addends;
            uint routed_per_op = 1u + num_limbs;
            uint aux_per_op = 4u * num_limbs;
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * routed_per_op;
                ulong aux_base =
                    (ulong)routed_per_op * num_ops + (ulong)op * aux_per_op;
                for (uint j = 0; j < aux_per_op; ++j) {
                    ulong x = wires[(aux_base + j) * lde_rows + source_row];
                    ulong y = gl_mul(x, gl_sub_canonical_rhs(x, 3));
                    range_check_gate_emit(
                        gl_mul(y, gl_add_canonical_rhs(y, 2)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }
                for (uint byte_index = 0; byte_index < num_limbs; ++byte_index) {
                    ulong chunk = aux_base + (ulong)byte_index * 4u;
                    ulong recomposed = wires[(chunk + 3u) * lde_rows + source_row];
                    for (uint remaining = 3u; remaining > 0u; --remaining) {
                        uint k = remaining - 1u;
                        recomposed = gl_add(
                            gl_quadruple(recomposed),
                            wires[(chunk + k) * lde_rows + source_row]);
                    }
                    ulong byte_value =
                        wires[(routed_base + 1u + byte_index) * lde_rows + source_row];
                    range_check_gate_emit(
                        gl_sub(recomposed, byte_value),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }
                ulong recomposed_sum =
                    wires[(routed_base + num_limbs) * lde_rows + source_row];
                for (uint remaining = num_limbs - 1u; remaining > 0u; --remaining) {
                    uint k = remaining - 1u;
                    recomposed_sum = gl_add(
                        gl_mul(recomposed_sum, 256),
                        wires[(routed_base + 1u + k) * lde_rows + source_row]);
                }
                ulong expected_sum = wires[routed_base * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(recomposed_sum, expected_sum),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 4u) {
            ulong strict_accumulators[2] = { 0, 0 };
            // QuinticMultiplicationGate: fifteen routed words per operation
            // (five limbs each for a, b and the claimed product c). The five
            // constraints are the schoolbook product limbs reduced by
            // u^5 = 3, minus the claimed output limbs, in ascending limb
            // order exactly like the CPU accumulator.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 15u;
                ulong a[5];
                ulong b[5];
                for (uint j = 0; j < 5u; ++j) {
                    a[j] = wires[(routed_base + j) * lde_rows + source_row];
                    b[j] = wires[(routed_base + 5u + j) * lde_rows + source_row];
                }
                ulong d[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                for (uint j = 0; j < 5u; ++j) {
                    for (uint k = 0; k < 5u; ++k) {
                        d[j + k] = gl_add(d[j + k], gl_mul(a[j], b[k]));
                    }
                }
                for (uint k = 0; k < 5u; ++k) {
                    ulong term = k < 4u
                        ? gl_add(d[k], gl_mul(3, d[k + 5u]))
                        : d[k];
                    ulong c = wires[(routed_base + 10u + k) * lde_rows + source_row];
                    range_check_gate_emit_strict(
                        gl_sub(term, c),
                        alpha_powers,
                        alpha_stride,
                        strict_accumulators,
                        constraint_index++);
                }
            }
            gate_accumulators[0] = alpha_acc_of(strict_accumulators[0]);
            gate_accumulators[1] = alpha_acc_of(strict_accumulators[1]);
        } else if (kind == 5u) {
            ulong strict_accumulators[2] = { 0, 0 };
            // QuinticSquaringGate: ten routed words per operation (input
            // limbs a then output limbs c) plus ten temporary wires. Each
            // constraint checks one accumulation step of the squaring
            // against its temporary or output, in the exact CPU emission
            // order.
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 10u;
                ulong temp_base = (ulong)num_ops * 10u + (ulong)op * 10u;
                ulong a[5];
                ulong c[5];
                ulong extra[10];
                for (uint j = 0; j < 5u; ++j) {
                    a[j] = wires[(routed_base + j) * lde_rows + source_row];
                    c[j] = wires[(routed_base + 5u + j) * lde_rows + source_row];
                }
                for (uint j = 0; j < 10u; ++j) {
                    extra[j] = wires[(temp_base + j) * lde_rows + source_row];
                }

                // c[0]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(a[0], a[0]), extra[0]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[1]), a[4]), extra[0]), extra[1]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[2]), a[3]), extra[1]), c[0]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);

                // c[1]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(gl_mul(3, a[3]), a[3]), extra[2]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[1]), extra[2]), extra[3]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[2]), a[4]), extra[3]), c[1]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);

                // c[2]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(a[1], a[1]), extra[4]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[2]), extra[4]), extra[5]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(6, a[3]), a[4]), extra[5]), c[2]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);

                // c[3]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(gl_mul(3, a[4]), a[4]), extra[6]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[3]), extra[6]), extra[7]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[1]), a[2]), extra[7]), c[3]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);

                // c[4]
                range_check_gate_emit_strict(
                    gl_sub(gl_mul(a[2], a[2]), extra[8]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[0]), a[4]), extra[8]), extra[9]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
                range_check_gate_emit_strict(
                    gl_sub(gl_add(gl_mul(gl_mul(2, a[1]), a[3]), extra[9]), c[4]),
                    alpha_powers, alpha_stride, strict_accumulators,
                    constraint_index++);
            }
            gate_accumulators[0] = alpha_acc_of(strict_accumulators[0]);
            gate_accumulators[1] = alpha_acc_of(strict_accumulators[1]);
        } else if (kind == 6u) {
            uint bits = num_addends;
            uint num_extra_constants = result_limbs;
            uint constant_base = num_carry_limbs;
            uint num_copies = num_ops;
            ulong vec_size = 1UL << bits;
            ulong routed_per_copy = vec_size + 2u;
            ulong extra_wire_base = routed_per_copy * num_copies;
            ulong bit_base = extra_wire_base + num_extra_constants;

            for (uint copy = 0; copy < num_copies; ++copy) {
                ulong copy_base = routed_per_copy * copy;

                // RandomAccessGate emits boolean constraints for b_0 upward.
                for (uint i = 0; i < bits; ++i) {
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * lde_rows + source_row];
                    range_check_gate_emit(
                        gl_mul(b, gl_sub_canonical_rhs(b, 1)),
                        alpha_powers,
                        alpha_stride,
                        gate_accumulators,
                        constraint_index++);
                }

                // Reconstruct the little-endian index in the CPU's exact
                // reverse-bit `acc.double() + b` order.
                ulong reconstructed_index = 0;
                for (uint remaining = bits; remaining > 0u; --remaining) {
                    uint i = remaining - 1u;
                    ulong b = wires[(bit_base + (ulong)copy * bits + i)
                        * lde_rows + source_row];
                    reconstructed_index = gl_add(
                        gl_add(reconstructed_index, reconstructed_index), b);
                }
                ulong access_index = wires[copy_base * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(reconstructed_index, access_index),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);

                // Fold each eight-item block in ascending pair order, then fold
                // block results with the remaining bits in the same order.
                ulong block_results[8];
                uint block_count = (uint)(vec_size / 8u);
                ulong list_base = copy_base + 2u;
                ulong copy_bit_base = bit_base + (ulong)copy * bits;
                for (uint block = 0; block < block_count; ++block) {
                    block_results[block] = random_access_select_8(
                        wires, lde_rows, source_row, list_base, copy_bit_base, block);
                }
                uint level_size = block_count;
                for (uint i = 3u; i < bits; ++i) {
                    ulong b = wires[(copy_bit_base + i) * lde_rows + source_row];
                    for (uint k = 0; k < level_size / 2u; ++k) {
                        ulong x = block_results[2u * k];
                        ulong y = block_results[2u * k + 1u];
                        block_results[k] = gl_add(x, gl_mul(b, gl_sub(y, x)));
                    }
                    level_size /= 2u;
                }
                ulong claimed_element = wires[(copy_base + 1u) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(block_results[0], claimed_element),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }

            // Raw local constants follow all gate and lookup selectors.
            for (uint i = 0; i < num_extra_constants; ++i) {
                ulong local_constant = constants[
                    ((ulong)constant_base + i) * lde_rows + source_row];
                ulong extra_wire = wires[
                    (extra_wire_base + i) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(local_constant, extra_wire),
                    alpha_powers,
                    alpha_stride,
                    gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 7u) {
            // ExponentiationGate: wire 0 is the base, wires 1..=n the power
            // bits in little-endian order, wire 1+n the output and wires
            // 2+n..2+2n the running intermediate values; `num_ops` carries n.
            // The accumulation walks the bits big-endian and seeds the chain
            // with ONE rather than a square, exactly as the CPU evaluator.
            uint num_power_bits = num_ops;
            ulong exponent_base = wires[(ulong)0 * lde_rows + source_row];
            for (uint i = 0; i < num_power_bits; ++i) {
                ulong previous;
                if (i == 0u) {
                    previous = 1;
                } else {
                    ulong last = wires[((ulong)2u + num_power_bits + i - 1u) * lde_rows
                                       + source_row];
                    previous = gl_mul(last, last);
                }
                ulong current_bit =
                    wires[((ulong)1u + (num_power_bits - i - 1u)) * lde_rows + source_row];
                ulong multiplier =
                    gl_add(gl_mul(current_bit, exponent_base), gl_sub(1, current_bit));
                ulong intermediate =
                    wires[((ulong)2u + num_power_bits + i) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(gl_mul(previous, multiplier), intermediate),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
            ulong output_value = wires[((ulong)1u + num_power_bits) * lde_rows + source_row];
            ulong final_intermediate =
                wires[((ulong)1u + 2u * num_power_bits) * lde_rows + source_row];
            range_check_gate_emit(
                gl_sub(output_value, final_intermediate),
                alpha_powers, alpha_stride, gate_accumulators,
                constraint_index++);
        } else if (kind == 8u) {
            // EqualityGate: three routed words per operation (x, y, equal)
            // followed by three unrouted temporaries (diff, invdiff, prod).
            // The addend slot carries the constants column holding the gate's
            // first constant, its "one" value.
            uint constant_column = num_addends;
            ulong const_0 = constants[(ulong)constant_column * lde_rows + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong routed_base = (ulong)op * 3u;
                ulong x = wires[(routed_base + 0u) * lde_rows + source_row];
                ulong y = wires[(routed_base + 1u) * lde_rows + source_row];
                ulong equal = wires[(routed_base + 2u) * lde_rows + source_row];
                ulong temporary_base = (ulong)num_ops * 3u + (ulong)op * 3u;
                ulong difference = wires[(temporary_base + 0u) * lde_rows + source_row];
                ulong inverse = wires[(temporary_base + 1u) * lde_rows + source_row];
                ulong product = wires[(temporary_base + 2u) * lde_rows + source_row];

                range_check_gate_emit(
                    gl_sub(gl_sub(x, y), difference),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(difference, inverse), product),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_mul(product, difference), difference),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(const_0, product), equal),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 9u) {
            // ReducingGate / ReducingExtensionGate at D == 2. Wires 0..2 are
            // the output, 2..4 alpha, 4..6 the incoming accumulator, then one
            // (base) or two (extension) wires per coefficient, then one
            // two-wire accumulator per step except the last, which aliases the
            // output. Each step emits the two components of
            // `acc * alpha + coeff - next_acc` in that order.
            //
            // The quadratic extension is F[x]/(x^2 - 7) for Goldilocks:
            //   (a0 + a1 x)(b0 + b1 x) = (a0 b0 + 7 a1 b1) + (a0 b1 + a1 b0) x.
            uint extension_coeffs = num_addends;
            uint coeff_wires = extension_coeffs != 0u ? 2u : 1u;
            uint coeff_start = 6u;
            uint acc_start = coeff_start + num_ops * coeff_wires;
            ulong alpha_0 = wires[(ulong)2u * lde_rows + source_row];
            ulong alpha_1 = wires[(ulong)3u * lde_rows + source_row];
            ulong acc_0 = wires[(ulong)4u * lde_rows + source_row];
            ulong acc_1 = wires[(ulong)5u * lde_rows + source_row];
            for (uint i = 0; i < num_ops; ++i) {
                uint next_start = (i + 1u == num_ops) ? 0u : acc_start + 2u * i;
                ulong next_0 = wires[(ulong)next_start * lde_rows + source_row];
                ulong next_1 = wires[((ulong)next_start + 1u) * lde_rows + source_row];

                uint coeff_wire = coeff_start + i * coeff_wires;
                ulong coeff_0 = wires[(ulong)coeff_wire * lde_rows + source_row];
                ulong coeff_1 = extension_coeffs != 0u
                    ? wires[((ulong)coeff_wire + 1u) * lde_rows + source_row]
                    : 0;

                ulong product_0 = gl_add(
                    gl_mul(acc_0, alpha_0),
                    gl_mul(7, gl_mul(acc_1, alpha_1)));
                ulong product_1 = gl_add(
                    gl_mul(acc_0, alpha_1),
                    gl_mul(acc_1, alpha_0));
                range_check_gate_emit(
                    gl_sub(gl_add(product_0, coeff_0), next_0),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_add(product_1, coeff_1), next_1),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);

                acc_0 = next_0;
                acc_1 = next_1;
            }
        } else if (kind == 10u) {
            // AdditionGate: three routed words per operation (x, y, output).
            // The addend-count slot carries the first of its two raw constant
            // columns, immediately after the selector prefix.
            uint constant_base = num_addends;
            ulong const_0 = constants[(ulong)constant_base * lde_rows + source_row];
            ulong const_1 = constants[((ulong)constant_base + 1u) * lde_rows + source_row];
            for (uint op = 0; op < num_ops; ++op) {
                ulong wire_base = (ulong)op * 3u;
                ulong addend_0 = wires[(wire_base + 0u) * lde_rows + source_row];
                ulong addend_1 = wires[(wire_base + 1u) * lde_rows + source_row];
                ulong output_value = wires[(wire_base + 2u) * lde_rows + source_row];
                ulong computed = gl_add(
                    gl_mul(addend_0, const_0),
                    gl_mul(addend_1, const_1));
                range_check_gate_emit(
                    gl_sub(output_value, computed),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 11u) {
            // BaseSumGate: wire 0 is the sum and the next `num_ops` wires are
            // little-endian limbs. The addend-count slot carries base 2 or 4.
            //
            // The Horner step's `gl_mul(computed, base)` looks like free money
            // -- both bases are powers of two, so doubling or `gl_quadruple`
            // would replace a 128-bit product with one or two field adds, 63
            // of them per row on the widest family. Measured, it is not:
            // specializing the base outside the loop costs more in code
            // duplication than the arithmetic saves. Recomposition-only,
            // against the deferred-accumulator kernel: d18 160.9 -> 161.6 ms,
            // d16-heavy 60.40 -> 60.53 ms, d14 4.135 -> 4.112 ms; splitting the
            // range-constraint loop as well costs another 1.5 ms on d18. Both
            // arms bit-exact, so this is a scheduling/footprint effect, not an
            // arithmetic one. Keep the multiply.
            ulong base = num_addends;
            ulong computed = 0;
            for (uint remaining = num_ops; remaining > 0u; --remaining) {
                uint limb = remaining - 1u;
                computed = gl_add(
                    gl_mul(computed, base),
                    wires[((ulong)1u + limb) * lde_rows + source_row]);
            }
            range_check_gate_emit(
                gl_sub(computed, wires[source_row]),
                alpha_powers, alpha_stride, gate_accumulators,
                constraint_index++);
            for (uint limb = 0; limb < num_ops; ++limb) {
                ulong x = wires[((ulong)1u + limb) * lde_rows + source_row];
                ulong constraint;
                if (base == 2u) {
                    constraint = gl_mul(x, gl_sub_canonical_rhs(x, 1));
                } else {
                    ulong y = gl_mul(x, gl_sub_canonical_rhs(x, 3));
                    constraint = gl_mul(y, gl_add_canonical_rhs(y, 2));
                }
                range_check_gate_emit(
                    constraint,
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
        } else if (kind == 12u) {
            // SelectionGate: four routed wires per operation followed by one
            // temporary wire per operation.
            for (uint op = 0; op < num_ops; ++op) {
                ulong b = wires[((ulong)(4u * op)) * lde_rows + source_row];
                ulong x = wires[((ulong)(4u * op + 1u)) * lde_rows + source_row];
                ulong y = wires[((ulong)(4u * op + 2u)) * lde_rows + source_row];
                ulong result = wires[((ulong)(4u * op + 3u)) * lde_rows + source_row];
                ulong temp = wires[
                    ((ulong)(4u * num_ops + op)) * lde_rows + source_row];
                range_check_gate_emit(
                    gl_sub(gl_sub(gl_mul(b, y), y), temp),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
                range_check_gate_emit(
                    gl_sub(gl_sub(gl_mul(b, x), temp), result),
                    alpha_powers, alpha_stride, gate_accumulators,
                    constraint_index++);
            }
        } else {
            // The Rust encoder rejects unknown discriminants; if a malformed
            // record reaches the shader, make its selected row unsatisfiable.
            range_check_gate_emit(
                1, alpha_powers, alpha_stride, gate_accumulators,
                constraint_index++);
        }

        total[0] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[0]), total[0]);
        total[1] = gl_mul_add(
            filter, alpha_acc_materialize(gate_accumulators[1]), total[1]);
    }

    output[(ulong)gid * 2] = gl_canonicalize(total[0]);
    output[(ulong)gid * 2 + 1] = gl_canonicalize(total[1]);
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

// One sponge absorption pass over a group of at most eight natural-order
// columns, with the running 12-lane state parked in `state` between passes
// (column-major: lane i of row gid at state[i * leaf_count + gid]). The
// final pass writes the four-lane digests to `hashes` at the bit-reversed
// row, exactly like poseidon2_hash_leaves_colmajor. Splitting the sponge by
// column group lets the CPU compute group g+1's LDE columns while the GPU
// absorbs group g; the arithmetic per pass is identical to the fused
// kernel's corresponding loop iteration.
kernel void poseidon2_absorb_pass(
    const device ulong* leaves [[buffer(0)]],
    device ulong* state [[buffer(1)]],
    device ulong* hashes [[buffer(2)]],
    constant ulong* parameters [[buffer(3)]],
    constant uint& leaf_count [[buffer(4)]],
    constant uint& log_leaf_count [[buffer(5)]],
    constant uint& col_start [[buffer(6)]],
    constant uint& chunk_size [[buffer(7)]],
    constant uint& first_pass [[buffer(8)]],
    constant uint& final_pass [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= leaf_count) {
        return;
    }
    ulong st[12] = { 0 };
    if (first_pass == 0u) {
        for (uint i = 0; i < 12; ++i) {
            st[i] = state[(ulong)i * leaf_count + gid];
        }
    }
    for (uint i = 0; i < chunk_size; ++i) {
        st[i] = gl_canonicalize(leaves[(ulong)(col_start + i) * leaf_count + gid]);
    }
    poseidon2(st, parameters);
    if (final_pass != 0u) {
        uint out_row = log_leaf_count == 0
            ? gid
            : (reverse_bits(gid) >> (32 - log_leaf_count));
        device ulong* output = hashes + (ulong)out_row * 4;
        for (uint i = 0; i < 4; ++i) {
            output[i] = gl_canonicalize(st[i]);
        }
    } else {
        for (uint i = 0; i < 12; ++i) {
            state[(ulong)i * leaf_count + gid] = st[i];
        }
    }
}
