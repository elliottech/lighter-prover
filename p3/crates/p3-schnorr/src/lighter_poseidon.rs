//! A second, width-12 Poseidon2 permutation matching the `goldilocks-crypto`
//! ("Lighter") reference implementation's `hash_to_quintic_extension`, used
//! specifically to derive the Fiat-Shamir challenge in [`crate::schnorr`].
//!
//! This is deliberately a separate permutation from [`crate::poseidon`]
//! (different width, round constants, and — critically — sponge padding
//! behavior) because it must bit-for-bit match an external, already-fixed
//! protocol (Lighter's signature scheme), not a hash function this crate gets
//! to define. See [`hash_to_quintic_extension`] for the most safety-critical
//! detail: its non-standard, no-padding sponge construction.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::poseidon::apply_mat4;

/// Poseidon2 state width for the Lighter permutation.
pub const LIGHTER_POSEIDON_WIDTH: usize = 12;
/// Sponge rate: field elements absorbed per permutation call.
pub const LIGHTER_POSEIDON_RATE: usize = 8;
/// Width of the squeezed digest: one `GF(p^5)` element (5 limbs).
pub const LIGHTER_POSEIDON_DIGEST_WIDTH: usize = 5;
/// Number of full rounds before *and* after the internal rounds (split evenly,
/// hence "half").
pub const LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS: usize = 4;
/// Number of partial (internal) rounds.
pub const LIGHTER_POSEIDON_INTERNAL_ROUNDS: usize = 22;
/// Total recorded states in [`lighter_poseidon2_round_states`]: the initial
/// linear layer plus one state per round (both external halves and internal).
pub const LIGHTER_POSEIDON_ROUND_STATE_COUNT: usize =
    1 + 2 * LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS + LIGHTER_POSEIDON_INTERNAL_ROUNDS;

/// Round constants for the 8 full (external) rounds: the first
/// [`LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS`] run before the internal rounds,
/// the rest after.
pub(crate) const LIGHTER_EXTERNAL_CONSTANTS: [[u64; LIGHTER_POSEIDON_WIDTH]; 8] = [
    [
        15492826721047263190,
        11728330187201910315,
        8836021247773420868,
        16777404051263952451,
        5510875212538051896,
        6173089941271892285,
        2927757366422211339,
        10340958981325008808,
        8541987352684552425,
        9739599543776434497,
        15073950188101532019,
        12084856431752384512,
    ],
    [
        4584713381960671270,
        8807052963476652830,
        54136601502601741,
        4872702333905478703,
        5551030319979516287,
        12889366755535460989,
        16329242193178844328,
        412018088475211848,
        10505784623379650541,
        9758812378619434837,
        7421979329386275117,
        375240370024755551,
    ],
    [
        3331431125640721931,
        15684937309956309981,
        578521833432107983,
        14379242000670861838,
        17922409828154900976,
        8153494278429192257,
        15904673920630731971,
        11217863998460634216,
        3301540195510742136,
        9937973023749922003,
        3059102938155026419,
        1895288289490976132,
    ],
    [
        5580912693628927540,
        10064804080494788323,
        9582481583369602410,
        10186259561546797986,
        247426333829703916,
        13193193905461376067,
        6386232593701758044,
        17954717245501896472,
        1531720443376282699,
        2455761864255501970,
        11234429217864304495,
        4746959618548874102,
    ],
    [
        13571697342473846203,
        17477857865056504753,
        15963032953523553760,
        16033593225279635898,
        14252634232868282405,
        8219748254835277737,
        7459165569491914711,
        15855939513193752003,
        16788866461340278896,
        7102224659693946577,
        3024718005636976471,
        13695468978618890430,
    ],
    [
        8214202050877825436,
        2670727992739346204,
        16259532062589659211,
        11869922396257088411,
        3179482916972760137,
        13525476046633427808,
        3217337278042947412,
        14494689598654046340,
        15837379330312175383,
        8029037639801151344,
        2153456285263517937,
        8301106462311849241,
    ],
    [
        13294194396455217955,
        17394768489610594315,
        12847609130464867455,
        14015739446356528640,
        5879251655839607853,
        9747000124977436185,
        8950393546890284269,
        10765765936405694368,
        14695323910334139959,
        16366254691123000864,
        15292774414889043182,
        10910394433429313384,
    ],
    [
        17253424460214596184,
        3442854447664030446,
        3005570425335613727,
        10859158614900201063,
        9763230642109343539,
        6647722546511515039,
        909012944955815706,
        18101204076790399111,
        11588128829349125809,
        15863878496612806566,
        5201119062417750399,
        176665553780565743,
    ],
];

/// Round constants for the [`LIGHTER_POSEIDON_INTERNAL_ROUNDS`] partial rounds.
pub(crate) const LIGHTER_INTERNAL_CONSTANTS: [u64; LIGHTER_POSEIDON_INTERNAL_ROUNDS] = [
    11921381764981422944,
    10318423381711320787,
    8291411502347000766,
    229948027109387563,
    9152521390190983261,
    7129306032690285515,
    15395989607365232011,
    8641397269074305925,
    17256848792241043600,
    6046475228902245682,
    12041608676381094092,
    12785542378683951657,
    14546032085337914034,
    3304199118235116851,
    16499627707072547655,
    10386478025625759321,
    13475579315436919170,
    16042710511297532028,
    1411266850385657080,
    9024840976168649958,
    14047056970978379368,
    838728605080212101,
];

/// Diagonal entries for the partial round's internal linear layer (see
/// [`internal_linear_layer`] below).
const LIGHTER_MATRIX_DIAG_12: [u64; LIGHTER_POSEIDON_WIDTH] = [
    0xc3b6c08e23ba9300,
    0xd84b5de94a324fb6,
    0x0d0c371c5b35b84f,
    0x7964f570e7188037,
    0x5daf18bbd996604b,
    0x6743bc47b9595257,
    0x5528b9362c59bb70,
    0xac45e25b7127b68b,
    0xa2077d7dfbb606b5,
    0xf3faac6faee378ae,
    0x0c6388b51545e883,
    0xd27dbb6944917b60,
];

/// Hashes `input` using Lighter's width-12 Poseidon2 sponge.
///
/// This intentionally does not use standard sponge padding: each
/// `LIGHTER_POSEIDON_RATE`-sized chunk only overwrites the rate lanes it
/// covers (`state[..chunk.len()]`), so a short final chunk leaves the
/// remaining rate lanes holding whatever the previous permutation produced
/// there rather than zeroing them. This "no-padding overwrite" behavior is
/// load-bearing for matching Lighter's reference hash (see
/// `hash_to_quintic_extension` tests below and the AIR's two-phase
/// absorption in `air/eval.rs`) and must not be "fixed" into a standard
/// zero-padded sponge without updating the AIR to match.
pub fn hash_to_quintic_extension(
    input: &[Goldilocks],
) -> [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH] {
    let mut state = [Goldilocks::ZERO; LIGHTER_POSEIDON_WIDTH];
    for chunk in input.chunks(LIGHTER_POSEIDON_RATE) {
        for (lane, value) in chunk.iter().copied().enumerate() {
            state[lane] = value;
        }
        lighter_poseidon2_permute(&mut state);
    }
    core::array::from_fn(|i| state[i])
}

/// The full Lighter Poseidon2 permutation: external-initial linear layer,
/// [`LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS`] full rounds, then
/// [`LIGHTER_POSEIDON_INTERNAL_ROUNDS`] partial rounds, then the remaining
/// full rounds.
pub fn lighter_poseidon2_permute<R: PrimeCharacteristicRing>(
    state: &mut [R; LIGHTER_POSEIDON_WIDTH],
) {
    lighter_external_initial_linear_layer(state);
    for round_constants in LIGHTER_EXTERNAL_CONSTANTS
        .iter()
        .take(LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS)
    {
        lighter_external_full_round(state, round_constants);
    }
    for rc in LIGHTER_INTERNAL_CONSTANTS {
        lighter_internal_partial_round(state, rc);
    }
    for round_constants in LIGHTER_EXTERNAL_CONSTANTS
        .iter()
        .skip(LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS)
    {
        lighter_external_full_round(state, round_constants);
    }
}

/// Same permutation as [`lighter_poseidon2_permute`], but records the state
/// after every round; used to cross-check round-by-round trace construction
/// against the whole-permutation result.
#[cfg(test)]
pub(crate) fn lighter_poseidon2_round_states<R: PrimeCharacteristicRing>(
    mut state: [R; LIGHTER_POSEIDON_WIDTH],
) -> [[R; LIGHTER_POSEIDON_WIDTH]; LIGHTER_POSEIDON_ROUND_STATE_COUNT] {
    let mut states = core::array::from_fn(|_| core::array::from_fn(|_| R::ZERO));
    let mut index = 0;

    lighter_external_initial_linear_layer(&mut state);
    states[index] = state.clone();
    index += 1;

    for round_constants in LIGHTER_EXTERNAL_CONSTANTS
        .iter()
        .take(LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS)
    {
        lighter_external_full_round(&mut state, round_constants);
        states[index] = state.clone();
        index += 1;
    }

    for rc in LIGHTER_INTERNAL_CONSTANTS {
        lighter_internal_partial_round(&mut state, rc);
        states[index] = state.clone();
        index += 1;
    }

    for round_constants in LIGHTER_EXTERNAL_CONSTANTS
        .iter()
        .skip(LIGHTER_POSEIDON_EXTERNAL_HALF_ROUNDS)
    {
        lighter_external_full_round(&mut state, round_constants);
        states[index] = state.clone();
        index += 1;
    }

    debug_assert_eq!(index, LIGHTER_POSEIDON_ROUND_STATE_COUNT);
    states
}

/// The linear layer applied once before any round constants, with no S-box.
pub(crate) fn lighter_external_initial_linear_layer<R: PrimeCharacteristicRing>(
    state: &mut [R; LIGHTER_POSEIDON_WIDTH],
) {
    external_linear_layer(state);
}

/// One full round: add round constants and apply the S-box to every lane,
/// then the external linear layer.
pub(crate) fn lighter_external_full_round<R: PrimeCharacteristicRing>(
    state: &mut [R; LIGHTER_POSEIDON_WIDTH],
    round_constants: &[u64; LIGHTER_POSEIDON_WIDTH],
) {
    for (value, rc) in state.iter_mut().zip(round_constants.iter().copied()) {
        *value += R::from_u64(rc);
        *value = exp_7(value);
    }
    external_linear_layer(state);
}

/// One partial round: add the round constant and apply the S-box to lane 0
/// only, then the internal linear layer.
pub(crate) fn lighter_internal_partial_round<R: PrimeCharacteristicRing>(
    state: &mut [R; LIGHTER_POSEIDON_WIDTH],
    rc: u64,
) {
    state[0] += R::from_u64(rc);
    state[0] = exp_7(&state[0]);
    internal_linear_layer(state);
}

/// `value^7`, computed as `(value^2)^2 * value^2 * value`.
fn exp_7<R: PrimeCharacteristicRing>(value: &R) -> R {
    let x2 = value.square();
    let x4 = x2.square();
    x4 * x2 * value.dup()
}

/// The partial round's linear layer: `state[i] = state[i] * diag[i] + sum(state)`.
fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; LIGHTER_POSEIDON_WIDTH]) {
    let sum: R = state.iter().map(|value| value.dup()).sum();
    for (lane, value) in state.iter_mut().enumerate() {
        *value *= R::from_u64(LIGHTER_MATRIX_DIAG_12[lane]);
        *value += sum.dup();
    }
}

/// The external linear layer: [`apply_mat4`] on each 4-lane chunk, then mix
/// across chunks by adding each chunk's column sums.
fn external_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; LIGHTER_POSEIDON_WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        apply_mat4(chunk.try_into().expect("chunk length is 4"));
    }

    let sums: [R; 4] = core::array::from_fn(|k| {
        (0..LIGHTER_POSEIDON_WIDTH)
            .step_by(4)
            .map(|j| state[j + k].dup())
            .sum()
    });

    for (i, elem) in state.iter_mut().enumerate() {
        *elem += sums[i % 4].dup();
    }
}

/// Decodes 40 little-endian bytes as 5 canonical Goldilocks limbs (an `Fp5`
/// element). Convenience wrapper around [`crate::types::Fp5Bytes`] for callers
/// in this module's reference/test code.
pub fn fp5_from_bytes(bytes: &[u8; 40]) -> crate::Result<[Goldilocks; 5]> {
    crate::types::Fp5Bytes::from_array(*bytes).to_goldilocks_limbs("fp5")
}

/// Inverse of [`fp5_from_bytes`]: serializes 5 Goldilocks limbs as 40
/// little-endian bytes.
pub fn fp5_to_bytes(value: &[Goldilocks; 5]) -> [u8; 40] {
    let mut bytes = [0u8; 40];
    for (i, value) in value.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&value.as_canonical_u64().to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lighter_hash_matches_poseidon_hash_crate() {
        let input = Goldilocks::new_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let local = hash_to_quintic_extension(&input);
        let upstream_input = input.map(|x| poseidon_hash::Goldilocks(x.as_canonical_u64()));
        let upstream = poseidon_hash::hash_to_quintic_extension(&upstream_input);
        let upstream = upstream
            .0
            .map(|x| Goldilocks::from_u64(x.to_canonical_u64()));

        assert_eq!(local, upstream);
    }

    #[test]
    fn lighter_permutation_matches_poseidon_hash_crate() {
        for seed in 0..32u64 {
            let mut state = seed;
            let input: [Goldilocks; LIGHTER_POSEIDON_WIDTH] = core::array::from_fn(|_| {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                Goldilocks::from_u64(z ^ (z >> 31))
            });

            let mut local = input;
            lighter_poseidon2_permute(&mut local);

            let mut reference: [poseidon_hash::Goldilocks; LIGHTER_POSEIDON_WIDTH] =
                input.map(|x| poseidon_hash::Goldilocks(x.as_canonical_u64()));
            poseidon_hash::permute(&mut reference);
            let reference = reference.map(|x| Goldilocks::from_u64(x.to_canonical_u64()));

            assert_eq!(local, reference, "seed {seed}");
        }
    }

    #[test]
    fn lighter_round_trace_matches_permutation() {
        let input = Goldilocks::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let states = lighter_poseidon2_round_states(input);
        let mut permuted = input;

        lighter_poseidon2_permute(&mut permuted);

        assert_eq!(states[LIGHTER_POSEIDON_ROUND_STATE_COUNT - 1], permuted);
    }
}
