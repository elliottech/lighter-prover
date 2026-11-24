use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;

use crate::air::EcdsaBatchAir;
use crate::air::layout::*;
use crate::air::lookups::reg_lookups;
use crate::air::trace::{generate_main_trace, generate_preprocessed};
use crate::prove::{EcdsaStarkConfig, PreparedEcdsaBatch, public_values};
use crate::stark::{LogUpAir, check_constraints};
use crate::types::EcdsaInstance;

type Challenge = p3_schnorr::proof::Challenge;

/// Serializes the full 2^16-row proving tests: run in parallel they peak at ~12 GB, which
/// takes down 16 GB CI runners.
pub(crate) static FULL_PROVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn full_prove_guard() -> std::sync::MutexGuard<'static, ()> {
    FULL_PROVE.lock().unwrap_or_else(|e| e.into_inner())
}

fn random_instances(count: usize, seed: u64) -> Vec<EcdsaInstance> {
    crate::testing::deterministic_batch(count, seed)
}

fn check(instances: &[EcdsaInstance], capacity: usize, tamper: impl FnOnce(&mut Vec<Goldilocks>)) {
    let air = EcdsaBatchAir::new(capacity);
    let mut main = generate_main_trace(instances, capacity);
    tamper(&mut main.values);
    let prep = generate_preprocessed(capacity);
    let challenges = [Challenge::from_u64(0x1234_5678_9abc_def1)];
    let perm = <EcdsaBatchAir as LogUpAir<Goldilocks, Challenge>>::generate_permutation_trace(
        &air,
        &main,
        Some(&prep),
        &challenges,
    );
    check_constraints::<EcdsaStarkConfig, _>(
        &air,
        &main,
        Some(&prep),
        &perm,
        &challenges,
        &public_values(instances),
    );
}

#[test]
fn reference_instances_verify() {
    for inst in random_instances(8, 1) {
        assert!(inst.verify_reference());
    }
}

#[test]
fn constraints_hold_for_valid_batch() {
    let instances = random_instances(3, 2);
    check(&instances, 4, |_| {});
    assert_eq!(generate_main_trace(&instances, 4).width(), MAIN_WIDTH);
}

#[test]
#[should_panic]
fn constraints_reject_tampered_accumulator() {
    let instances = random_instances(1, 3);
    check(&instances, 1, |v| {
        v[(EC_FIRST_ROW + 5) * MAIN_WIDTH + ACCX] += Goldilocks::ONE
    });
}

#[test]
fn prove_and_verify() {
    let _guard = full_prove_guard();
    let instances = random_instances(2, 4);
    let prepared = PreparedEcdsaBatch::new(2);
    let out = prepared.prove(&instances).unwrap();
    prepared.verify(&out.proof, &out.public_values).unwrap();
    let mut bad = out.public_values.clone();
    bad[1] += Goldilocks::ONE;
    assert!(prepared.verify(&out.proof, &bad).is_err());
}

#[test]
fn rejects_invalid_signatures() {
    let mut instances = random_instances(2, 5);
    instances[1].tx_hash[0] += Goldilocks::ONE;
    assert!(crate::prove::validate_batch(&instances, 2).is_err());
    let mut instances = random_instances(1, 6);
    instances[0].public_key[5] ^= 1;
    assert!(crate::prove::validate_batch(&instances, 1).is_err());
}

#[test]
#[should_panic]
fn constraints_reject_tampered_loop_result() {
    let instances = random_instances(1, 7);
    check(&instances, 1, |v| {
        for o in 0..ROWS_PER_SIGNATURE {
            v[o * MAIN_WIDTH + RPX] += Goldilocks::ONE;
        }
    });
}

#[test]
#[should_panic]
fn constraints_reject_tampered_message_byte() {
    let instances = random_instances(1, 8);
    check(&instances, 1, |v| {
        for o in 0..ROWS_PER_SIGNATURE {
            v[o * MAIN_WIDTH + ZHI] += Goldilocks::ONE;
        }
    });
}

/// Values the register lookups send to the u16 table on a signature's first row.
fn reg_lookup_values(row: &[Goldilocks]) -> Vec<u64> {
    use p3_field::PrimeField64;
    reg_lookups()
        .iter()
        .map(|&(c, scale)| (row[c] * Goldilocks::from_u64(scale)).as_canonical_u64())
        .collect()
}

#[test]
fn flipped_parity_needs_out_of_range_lookup() {
    // Claiming the other parity forces `Y_HALF = (QY[0] - parity') / 2` in the
    // field, which is not a u16, so the range bus cannot balance.
    let instances = random_instances(1, 9);
    let main = generate_main_trace(&instances, 1);
    let mut row = main.values[..MAIN_WIDTH].to_vec();
    assert!(reg_lookup_values(&row).iter().all(|&v| v < 1 << 16));
    let parity = Goldilocks::ONE - row[Y_PARITY];
    row[Y_PARITY] = parity;
    row[Y_HALF] = (row[QY] - parity) * Goldilocks::TWO.inverse();
    assert_eq!(row[QY], row[Y_HALF].double() + row[Y_PARITY]);
    assert!(reg_lookup_values(&row).iter().any(|&v| v >= 1 << 16));
}

#[test]
fn non_byte_split_of_z_needs_out_of_range_lookup() {
    let instances = random_instances(1, 10);
    let main = generate_main_trace(&instances, 1);
    let mut row = main.values[..MAIN_WIDTH].to_vec();
    let z0 = row[ZLO] + row[ZHI] * Goldilocks::from_u64(256);
    let lo = row[ZLO] + Goldilocks::ONE;
    row[ZLO] = lo;
    row[ZHI] = (z0 - lo) * Goldilocks::from_u64(256).inverse();
    assert_eq!(z0, row[ZLO] + row[ZHI] * Goldilocks::from_u64(256));
    assert!(reg_lookup_values(&row).iter().any(|&v| v >= 1 << 16));
}

/// Worst-case magnitude of a limb expression (all operations widen).
#[derive(Clone, Copy, Debug)]
struct Abs(u128);

impl core::ops::Add for Abs {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Abs(self.0 + o.0)
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl core::ops::Sub for Abs {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Abs(self.0 + o.0)
    }
}
impl core::ops::Mul for Abs {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Abs(self.0 * o.0)
    }
}
impl core::ops::Neg for Abs {
    type Output = Self;
    fn neg(self) -> Self {
        self
    }
}
impl crate::air::limbs::LimbRing for Abs {
    fn constant(v: i128) -> Self {
        Abs(v.unsigned_abs())
    }
}

/// Soundness of the non-native layer: with every limb, quotient limb and carry
/// half range-checked to 16 bits, each carry-chain group
/// `res[2g] + 2^16 res[2g+1] + c_{g-1} - 2^32 c_g` stays below the Goldilocks
/// modulus in absolute value, so the group constraints hold over the integers.
#[test]
fn relation_groups_cannot_wrap() {
    use crate::air::limbs::residual;
    use crate::air::relations::{ROW_KINDS, relations};
    const P: u128 = 0xffff_ffff_0000_0001;
    let mut local = vec![Abs(0xffff); MAIN_WIDTH];
    for c in (ACC_INF..BITS + NUM_SCALARS).chain([R_WRAP, RP_INF]) {
        local[c] = Abs(1);
    }
    let next = local.clone();
    let q = vec![Abs(0xffff); Q_LIMBS];
    let carry = 1u128 << 31;
    let mut worst = 0u128;
    for kind in ROW_KINDS {
        for rel in relations(kind, &local, &next, &Abs(1)) {
            let res = residual(&rel, &q);
            for g in 0..GROUPS {
                let hi = res.get(2 * g + 1).map_or(0, |v| v.0 << 16);
                let bound = res[2 * g].0 + hi + carry + (carry << 32);
                worst = worst.max(bound);
                assert!(bound < P, "{kind:?} group {g} may wrap: {bound:#x}");
            }
        }
    }
    // |value| < bound < P and value == 0 mod P force value == 0 over the integers.
    // The bound is dominated by the 2^32 * c_g carry term (about 2^63).
    assert!(worst < P, "{worst:#x}");
}

fn with_s(inst: &EcdsaInstance, s: &num_bigint::BigUint) -> EcdsaInstance {
    let q = inst.public_key_point().unwrap();
    EcdsaInstance::from_parts(inst.tx_hash, &inst.r_scalar(), s, &q)
}

#[test]
#[should_panic]
fn constraints_reject_high_s() {
    // (r, n - s) is a valid ECDSA signature mathematically; the AIR enforces low-s.
    let inst = &random_instances(1, 21)[0];
    let high = with_s(inst, &(&crate::secp::constants().n - inst.s_scalar()));
    check(&[high], 1, |_| {});
}

#[test]
#[should_panic]
fn constraints_reject_tampered_r_wrap() {
    // Claiming R.x = r + n (R_WRAP = 1) when R.x = r breaks the R.x relation.
    let instances = random_instances(1, 22);
    check(&instances, 1, |v| {
        for o in 0..ROWS_PER_SIGNATURE {
            v[o * MAIN_WIDTH + R_WRAP] = Goldilocks::ONE - v[o * MAIN_WIDTH + R_WRAP];
        }
    });
}

#[test]
#[should_panic]
fn constraints_reject_off_curve_key() {
    // Shifting Q.y keeps every limb in range but breaks y^2 = x^3 + 7.
    let instances = random_instances(1, 23);
    check(&instances, 1, |v| {
        for o in 0..ROWS_PER_SIGNATURE {
            v[o * MAIN_WIDTH + QY + 1] += Goldilocks::ONE;
        }
    });
}

#[test]
#[should_panic]
fn constraints_reject_r_not_below_n() {
    // `n - 1 - r` is an exact (integer) range difference, so `r = n` has no witness.
    let instances = random_instances(1, 24);
    let n = crate::secp::to_limbs(&crate::secp::constants().n, LIMBS);
    check(&instances, 1, |v| {
        for o in 0..ROWS_PER_SIGNATURE {
            for (m, &limb) in n.iter().enumerate() {
                v[o * MAIN_WIDTH + SIG_R + m] = Goldilocks::from_u64(limb);
            }
        }
    });
}
