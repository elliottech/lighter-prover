//! Witness generation for the message-hash AIR.

use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_keccak_air::{NUM_KECCAK_COLS, generate_trace_rows};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use super::*;
use crate::digest::{initial_link_hash_state, link_fields};
use crate::types::EcdsaInstance;

type F = Goldilocks;

/// Keccak-f input lanes (`input[x + 5y]`) for one instance.
pub fn keccak_input(inst: &EcdsaInstance) -> [u64; 25] {
    let block = preimage_block(&inst.message_bytes());
    let mut lanes = [0u64; 25];
    for (lane, chunk) in lanes.iter_mut().zip(block.chunks(8)) {
        *lane = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    lanes
}

pub fn generate_trace(instances: &[EcdsaInstance], capacity: usize) -> RowMajorMatrix<F> {
    assert!(instances.len() <= capacity);
    let air = MessageHashAir::new(capacity);
    // Enough all-zero permutations that `generate_trace_rows` pads to `air.height()`.
    let perms = capacity.max(1).max(air.height() / ROWS_PER_SIGNATURE);
    let inputs: Vec<[u64; 25]> = (0..perms)
        .map(|i| instances.get(i).map(keccak_input).unwrap_or([0; 25]))
        .collect();
    let keccak = generate_trace_rows::<F>(inputs, 0);
    let height = keccak.values.len() / NUM_KECCAK_COLS;
    assert_eq!(height, air.height());
    let mut values = F::zero_vec(height * WIDTH);
    values
        .par_chunks_mut(WIDTH)
        .zip(keccak.values.par_chunks(NUM_KECCAK_COLS))
        .for_each(|(dst, src)| dst[..NUM_KECCAK_COLS].copy_from_slice(src));
    drop(keccak);

    let count = instances.len();
    let mut state = initial_link_hash_state(count);
    let mut counter = 0u64;
    for perm in 0..height.div_ceil(ROWS_PER_SIGNATURE) {
        let active = perm < count;
        counter += u64::from(active);
        let (fields, h_inv) = match instances.get(perm) {
            Some(inst) => {
                let fields = link_fields(inst);
                let h_inv: [F; TX_HASH_ELEMENTS] = core::array::from_fn(|e| {
                    let hi = inst.tx_hash[e].as_canonical_u64() >> 32;
                    (F::from_u64(hi) - F::from_u64(0xffff_ffff))
                        .try_inverse()
                        .unwrap_or(F::ZERO)
                });
                (fields.to_vec(), h_inv)
            }
            None => (
                vec![F::ZERO; TX_HASH_ELEMENTS + DIGEST_WORDS],
                [F::ZERO; TX_HASH_ELEMENTS],
            ),
        };
        for i in 0..ROWS_PER_SIGNATURE {
            let r = perm * ROWS_PER_SIGNATURE + i;
            if r >= height {
                break;
            }
            let row = &mut values[r * WIDTH..(r + 1) * WIDTH];
            row[ACTIVE] = F::from_bool(active);
            row[COUNTER] = F::from_u64(counter);
            row[Z_WORDS..Z_WORDS + DIGEST_WORDS].copy_from_slice(&fields[TX_HASH_ELEMENTS..]);
            row[H_INV..H_INV + TX_HASH_ELEMENTS].copy_from_slice(&h_inv);
            for k in 0..SLOTS_PER_ROW {
                row[link_state(k)..link_state(k) + POSEIDON_WIDTH].copy_from_slice(&state);
                let rc = slot_round_constants(SLOTS_PER_ROW * i + k);
                for l in 0..POSEIDON_WIDTH {
                    row[link_sbox(k) + l] = (state[l] + F::from_u64(rc[l])).cube();
                }
                if !active {
                    continue;
                }
                if let Some((b, round)) = slot_kind(SLOTS_PER_ROW * i + k) {
                    sponge_step(&mut state, &fields, b, round);
                }
            }
        }
    }
    RowMajorMatrix::new(values, WIDTH)
}
