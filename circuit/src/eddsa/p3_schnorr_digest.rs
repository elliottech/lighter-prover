// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use p3_schnorr::poseidon::{POSEIDON_DIGEST_WIDTH, POSEIDON_RATE};
pub use p3_schnorr::poseidon::{POSEIDON_WIDTH as P3_DIGEST_STATE_WIDTH, PUBLIC_HASH_DOMAIN};
use p3_schnorr::types::INSTANCE_FIELD_ELEMENTS;
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::hash::poseidon2::hash::Poseidon2Hash;
use plonky2::iop::target::{BoolTarget, Target};

use super::gadgets::base_field::QuinticExtensionTarget;
use super::schnorr::SchnorrSigTarget;
use crate::types::config::Builder;

/// In-circuit running sponge state for the p3-schnorr public digest accumulator.
#[derive(Clone, Debug)]
pub struct P3SchnorrDigestState {
    pub state: [Target; P3_DIGEST_STATE_WIDTH],
}

impl P3SchnorrDigestState {
    pub fn initial(builder: &mut Builder, count: usize) -> Self {
        let count = builder.constant(plonky2::field::types::Field::from_canonical_usize(count));
        Self::initial_with_count(builder, count)
    }

    pub fn initial_with_count(builder: &mut Builder, count: Target) -> Self {
        let zero = builder.zero();
        let mut state = [zero; P3_DIGEST_STATE_WIDTH];
        state[0] = count;
        state[1] = builder.constant(plonky2::field::types::Field::from_canonical_u64(
            PUBLIC_HASH_DOMAIN,
        ));
        Self { state }
    }

    fn absorb_block(&mut self, builder: &mut Builder, block: [Target; POSEIDON_RATE]) {
        self.state[..POSEIDON_RATE].copy_from_slice(&block);
        let permutation = builder
            .builder
            .permute::<Poseidon2Hash>(PlonkyPermutation::new(self.state));
        self.state.copy_from_slice(permutation.as_ref());
    }

    pub fn absorb_instance(
        &mut self,
        builder: &mut Builder,
        account_pk: QuinticExtensionTarget,
        tx_hash: QuinticExtensionTarget,
        signature: &SchnorrSigTarget,
    ) {
        let zero = builder.zero();
        let mut fields = [zero; INSTANCE_FIELD_ELEMENTS];
        fields[0..5].copy_from_slice(&tx_hash.0);
        for (i, limb) in signature.s.value.limbs.iter().enumerate() {
            fields[5 + i] = limb.0;
        }
        for (i, limb) in signature.e.value.limbs.iter().enumerate() {
            fields[15 + i] = limb.0;
        }
        fields[25..30].copy_from_slice(&account_pk.0);

        let mut cursor = 0;
        while cursor < INSTANCE_FIELD_ELEMENTS {
            let take = (INSTANCE_FIELD_ELEMENTS - cursor).min(POSEIDON_RATE);
            let mut block = [zero; POSEIDON_RATE];
            block[..take].copy_from_slice(&fields[cursor..cursor + take]);
            self.absorb_block(builder, block);
            cursor += take;
        }
    }

    pub fn absorb_instance_conditional(
        &mut self,
        builder: &mut Builder,
        account_pk: QuinticExtensionTarget,
        tx_hash: QuinticExtensionTarget,
        signature: &SchnorrSigTarget,
        signed: BoolTarget,
    ) {
        let previous_state = self.state;
        self.absorb_instance(builder, account_pk, tx_hash, signature);
        self.state = core::array::from_fn(|i| {
            builder
                .builder
                .select(signed, self.state[i], previous_state[i])
        });
    }

    /// The final digest: the first `POSEIDON_DIGEST_WIDTH` lanes of the running state.
    pub fn digest(&self) -> [Target; POSEIDON_DIGEST_WIDTH] {
        core::array::from_fn(|i| self.state[i])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_field::PrimeField64 as P3PrimeField64;
    use p3_schnorr::poseidon::digest_instances;
    use p3_schnorr::types::{Fp5Bytes, SchnorrInstance, SignatureBytes};
    use plonky2::field::types::{Field, PrimeField, PrimeField64};
    use plonky2::iop::witness::PartialWitness;

    use super::*;
    use crate::eddsa::gadgets::base_field::{CircuitBuilderGFp5, PartialWitnessQuinticExt};
    use crate::eddsa::schnorr::{
        SchnorrSigTargetWitness, hash_to_quintic_extension, schnorr_generate_random_sk,
        schnorr_pk_from_sk, schnorr_sign_hashed_message,
    };
    use crate::types::config::{C, CIRCUIT_CONFIG, F};

    fn goldilocks_le_bytes(limbs: &[F]) -> Vec<u8> {
        limbs
            .iter()
            .flat_map(|f| f.to_canonical_u64().to_le_bytes())
            .collect()
    }

    fn scalar_le_bytes(scalar: &crate::eddsa::curve::scalar_field::ECgFp5Scalar) -> Vec<u8> {
        let mut digits = scalar.to_canonical_biguint().to_u32_digits();
        digits.resize(10, 0);
        digits.iter().flat_map(|d| d.to_le_bytes()).collect()
    }

    #[test]
    fn test_digest_accumulator_matches_p3_schnorr_natively() -> Result<()> {
        // Build 3 real signatures, compute the expected digest via p3-schnorr's own
        // digest_instances, then check this circuit's accumulator produces the same digest.
        let count = 3;
        let mut instances = Vec::with_capacity(count);
        let mut sigs = Vec::with_capacity(count);
        let mut pks = Vec::with_capacity(count);
        let mut msg_digests = Vec::with_capacity(count);

        for i in 0..count {
            let sk = schnorr_generate_random_sk();
            let pk = schnorr_pk_from_sk(&sk);
            let msg_bytes = format!("digest accumulator test {i}");
            let msg = msg_bytes
                .bytes()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>();
            let msg_digest = hash_to_quintic_extension(&msg);
            let sig = schnorr_sign_hashed_message(&msg_digest, &sk);

            let mut msg_digest_bytes = [0u8; 40];
            msg_digest_bytes.copy_from_slice(&goldilocks_le_bytes(&msg_digest.0));
            let mut pk_bytes = [0u8; 40];
            pk_bytes.copy_from_slice(&goldilocks_le_bytes(&pk.0));
            let mut sig_bytes = [0u8; 80];
            sig_bytes[..40].copy_from_slice(&scalar_le_bytes(&sig.s));
            sig_bytes[40..].copy_from_slice(&scalar_le_bytes(&sig.e));

            instances.push(SchnorrInstance::new(
                Fp5Bytes::from_array(msg_digest_bytes),
                SignatureBytes::from_array(sig_bytes),
                Fp5Bytes::from_array(pk_bytes),
            ));
            sigs.push(sig);
            pks.push(pk);
            msg_digests.push(msg_digest);
        }

        let expected_digest = digest_instances(count, &instances)?;

        let mut builder = Builder::new(CIRCUIT_CONFIG);
        let mut acc = P3SchnorrDigestState::initial(&mut builder, count);

        let pk_targets: Vec<_> = (0..count)
            .map(|_| builder.add_virtual_quintic_ext_target())
            .collect();
        let msg_hash_targets: Vec<_> = (0..count)
            .map(|_| builder.add_virtual_quintic_ext_target())
            .collect();
        let sig_targets: Vec<_> = (0..count)
            .map(|_| SchnorrSigTarget::new(&mut builder))
            .collect();

        for i in 0..count {
            acc.absorb_instance(
                &mut builder,
                pk_targets[i],
                msg_hash_targets[i],
                &sig_targets[i],
            );
        }
        let digest_targets = acc.digest();
        for &t in &digest_targets {
            builder.register_public_input(t);
        }

        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        for i in 0..count {
            pw.set_quintic_ext_target(pk_targets[i], pks[i])?;
            pw.set_quintic_ext_target(msg_hash_targets[i], msg_digests[i])?;
            pw.set_schnorr_sig_target(&sig_targets[i], &sigs[i])?;
        }

        let proof = data.prove(pw)?;
        data.verify(proof.clone())?;

        let actual_digest: Vec<u64> = proof
            .public_inputs
            .iter()
            .map(|f| f.to_canonical_u64())
            .collect();
        let expected_digest_u64: Vec<u64> = expected_digest
            .iter()
            .map(P3PrimeField64::as_canonical_u64)
            .collect();

        assert_eq!(
            actual_digest, expected_digest_u64,
            "in-circuit digest accumulator must match p3-schnorr's native digest_instances"
        );

        Ok(())
    }

    #[test]
    fn test_lighter_permutation_matches_plonky2_fork() {
        use p3_field::PrimeCharacteristicRing;
        use p3_goldilocks::Goldilocks;
        use plonky2::hash::poseidon2::hash::Poseidon2;

        for seed in 0..32u64 {
            let mut state = seed;
            let raw: [u64; P3_DIGEST_STATE_WIDTH] = core::array::from_fn(|_| {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                (z ^ (z >> 31)) % 0xffff_ffff_0000_0001
            });

            let mut p3_state: [Goldilocks; P3_DIGEST_STATE_WIDTH] = raw.map(Goldilocks::from_u64);
            p3_schnorr::lighter_poseidon::lighter_poseidon2_permute(&mut p3_state);

            let plonky2_out = <F as Poseidon2>::poseidon2(raw.map(F::from_canonical_u64));

            let p3_out: [u64; P3_DIGEST_STATE_WIDTH] =
                p3_state.map(|x| P3PrimeField64::as_canonical_u64(&x));
            let plonky2_out: [u64; P3_DIGEST_STATE_WIDTH] =
                plonky2_out.map(|x| x.to_canonical_u64());
            assert_eq!(p3_out, plonky2_out, "seed {seed}");
        }
    }
}
