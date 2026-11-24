use p3_air::check_all_constraints;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_keccak_air::input_limb;

use super::prove::{prove_message_hashes, verify_message_hashes};
use super::trace::{generate_trace, keccak_input};
use super::*;
use crate::testing::deterministic_batch;

#[test]
fn column_layout_matches_keccak_air() {
    for g in 0..RATE_BYTES / 2 {
        assert_eq!(input_limb(g), PREIMAGE + g);
    }
}

#[test]
fn keccak_input_matches_personal_sign_hash() {
    // The AIR's block layout reproduces tiny-keccak's personal_sign hash.
    let inst = &deterministic_batch(1, 11)[0];
    let mut state = keccak_input(inst);
    tiny_keccak::keccakf(&mut state);
    let out: Vec<u8> = state[..4].iter().flat_map(|l| l.to_le_bytes()).collect();
    assert_eq!(out, inst.message_hash());
}

#[test]
fn constraints_hold_and_reject_tampering() {
    let instances = deterministic_batch(3, 12);
    let air = MessageHashAir::new(4);
    let pv = prove::public_values(&instances);
    let trace = generate_trace(&instances, 4);
    assert!(check_all_constraints(&air, &trace, &pv, Some(1)).is_ok());

    let mut bad = trace.clone();
    for i in 0..ROWS_PER_SIGNATURE {
        bad.values[(ROWS_PER_SIGNATURE + i) * WIDTH + Z_WORDS] += Goldilocks::ONE;
    }
    assert!(!check_all_constraints(&air, &bad, &pv, Some(1)).is_ok());

    let mut bad_pv = pv.clone();
    bad_pv[1] += Goldilocks::ONE;
    assert!(!check_all_constraints(&air, &trace, &bad_pv, Some(1)).is_ok());
}

#[test]
fn prove_verify_and_link_with_ecdsa() {
    let instances = deterministic_batch(2, 13);
    let out = prove_message_hashes(&instances, 2).unwrap();
    verify_message_hashes(&out.proof, &out.public_values, 2).unwrap();
    let mut bad = out.public_values.clone();
    bad[2] += Goldilocks::ONE;
    assert!(verify_message_hashes(&out.proof, &bad, 2).is_err());
    let ecdsa = crate::prove::public_values(&instances);
    assert_eq!(ecdsa[1 + PUBLIC_DIGEST_WIDTH..], out.public_values[1..]);
}

#[test]
fn batch_verify_rejects_mismatched_halves() {
    let _guard = crate::tests::full_prove_guard();
    use crate::batch::PreparedBatch;
    let prepared = PreparedBatch::new(2);
    let (a, pv_a) = prepared.prove(&deterministic_batch(2, 14)).unwrap();
    let (b, pv_b) = prepared.prove(&deterministic_batch(2, 15)).unwrap();
    prepared.verify(&a, &pv_a).unwrap();
    // Each half is valid on its own, but not against the other half's link digest.
    let mixed = crate::batch::BatchProof {
        ecdsa: a.ecdsa,
        message_hash: b.message_hash,
    };
    assert!(prepared.verify(&mixed, &pv_a).is_err());
    assert!(prepared.verify(&mixed, &pv_b).is_err());
}

#[test]
fn constraints_reject_tampered_padding() {
    let instances = deterministic_batch(1, 16);
    let air = MessageHashAir::new(1);
    let pv = prove::public_values(&instances);
    let mut trace = generate_trace(&instances, 1);
    // The 0x01 padding limb right after the message, on the first row only.
    let g = MSG_LIMB_START + crate::types::MESSAGE_BYTES / 2;
    trace.values[PREIMAGE + g] += Goldilocks::ONE;
    assert!(!check_all_constraints(&air, &trace, &pv, Some(1)).is_ok());
}

#[test]
fn air_degree_matches_declared() {
    use p3_air::symbolic::{AirLayout, SymbolicAirBuilder};
    let layout = AirLayout {
        preprocessed_width: 0,
        main_width: WIDTH,
        num_public_values: NUM_PUBLIC_VALUES,
        permutation_width: 0,
        num_permutation_challenges: 0,
        num_permutation_values: 0,
        num_periodic_columns: 0,
    };
    let mut b = SymbolicAirBuilder::<Goldilocks>::new(layout);
    MessageHashAir::new(1).eval(&mut b);
    let max = b
        .base_constraints()
        .iter()
        .map(|c| c.degree_multiple())
        .max();
    assert_eq!(max, Some(MAX_CONSTRAINT_DEGREE));
}
