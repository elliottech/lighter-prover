#[cfg(all(test, feature = "reference"))]
mod tests {
    use super::*;
    use p3_field::PrimeField64;

    use crate::air::SignatureChallengeHint;
    use crate::reference::deterministic_batch;
    use crate::schnorr::verification_witness;
    use crate::types::Batch;

    fn challenge_hints(batch: &Batch) -> Vec<SignatureChallengeHint> {
        batch
            .instances()
            .iter()
            .map(|instance| {
                let witness = verification_witness(instance).unwrap();
                SignatureChallengeHint::from_reference(instance, witness.reconstructed_r).unwrap()
            })
            .collect()
    }

    #[test]
    fn signature_batch_stark_proves_and_verifies() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch(&batch).unwrap();

        verify_signature_batch_proof(&output.public_inputs, &output.proof).unwrap();
    }

    #[test]
    fn signature_batch_stark_proves_with_external_hints() {
        let batch = deterministic_batch(2).unwrap();
        let hints = challenge_hints(&batch);
        let output = prove_signature_batch_with_challenge_hints(&batch, &hints).unwrap();

        verify_signature_batch_proof(&output.public_inputs, &output.proof).unwrap();
    }

    #[test]
    fn signature_batch_stark_verifies_with_preprocessed_vk() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let verifier = prepare_signature_batch_verifier(&output.public_inputs).unwrap();

        verify_signature_batch_proof_preprocessed(&verifier, &output.public_inputs, &output.proof)
            .unwrap();
    }

    #[test]
    fn signature_batch_stark_proves_with_capacity() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch_with_capacity(&batch, 8).unwrap();

        assert_eq!(output.proof.rows(), trace_height_for_count(8));
        assert_eq!(output.public_inputs.signature_count(), 2);
        let full = prove_signature_batch(&batch).unwrap();
        assert_eq!(output.public_inputs, full.public_inputs);

        verify_signature_batch_proof(&output.public_inputs, &output.proof).unwrap();
        let verifier = prepare_signature_batch_verifier_for_capacity(8).unwrap();
        verify_signature_batch_proof_preprocessed(&verifier, &output.public_inputs, &output.proof)
            .unwrap();
    }

    #[test]
    fn capacity_proof_wire_round_trip_verifies() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch_with_capacity(&batch, 8).unwrap();
        let bytes = output.to_bytes().unwrap();
        let decoded = SignatureBatchProvingOutput::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.public_inputs, output.public_inputs);
        assert_eq!(decoded.proof.rows(), trace_height_for_count(8));
        verify_signature_batch_proof(&decoded.public_inputs, &decoded.proof).unwrap();
    }

    #[test]
    fn capacity_proving_rejects_batch_larger_than_capacity() {
        let batch = deterministic_batch(3).unwrap();
        let err = match prove_signature_batch_with_capacity(&batch, 2) {
            Ok(_) => panic!("proving should reject a batch larger than the capacity"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            SignatureBatchProofError::Input(crate::Error::BatchExceedsCapacity {
                actual: 3,
                capacity: 2
            })
        ));
    }

    #[test]
    fn capacity_proving_rejects_capacity_above_max() {
        let batch = deterministic_batch(1).unwrap();
        let err = match prove_signature_batch_with_capacity(&batch, MAX_SIGNATURES + 1) {
            Ok(_) => panic!("proving should reject a capacity above MAX_SIGNATURES"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            SignatureBatchProofError::Input(crate::Error::TooManySignatures { .. })
        ));
    }

    #[test]
    fn capacity_proof_rejects_tampered_count() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch_with_capacity(&batch, 8).unwrap();
        let tampered =
            SignatureBatchPublicInputs::new(3, *output.public_inputs.digest()).unwrap();

        assert!(verify_signature_batch_proof(&tampered, &output.proof).is_err());
    }

    #[test]
    fn capacity_proof_rejects_tampered_digest() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch_with_capacity(&batch, 8).unwrap();
        let mut digest = *output.public_inputs.digest();
        digest[0] += Goldilocks::ONE;
        let tampered =
            SignatureBatchPublicInputs::new(output.public_inputs.signature_count(), digest)
                .unwrap();

        assert!(verify_signature_batch_proof(&tampered, &output.proof).is_err());
    }

    #[test]
    fn signature_batch_stark_rejects_wrong_hint_count() {
        let batch = deterministic_batch(2).unwrap();
        let err = match prove_signature_batch_with_challenge_hints(&batch, &[]) {
            Ok(_) => panic!("proving should reject a mismatched hint count"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            SignatureBatchProofError::Input(crate::Error::ChallengeHintCountMismatch {
                actual: 0,
                expected: 2
            })
        ));
    }

    #[test]
    fn prepares_hints_after_public_digest_validation() {
        let batch = deterministic_batch(2).unwrap();
        let digest = poseidon_public_digest(&batch).unwrap();
        let prepared =
            prepare_signature_batch_with_public_digest(batch.instances().to_vec(), digest).unwrap();

        assert_eq!(prepared.batch.len(), 2);
        assert_eq!(prepared.public_inputs.digest(), &digest);
        assert_eq!(prepared.challenge_hints.len(), 2);

        let output = prepared.prove().unwrap();
        assert_eq!(output.public_inputs, prepared.public_inputs);
        verify_signature_batch_proof(&output.public_inputs, &output.proof).unwrap();
    }

    #[test]
    fn prepare_hints_rejects_wrong_public_digest() {
        let batch = deterministic_batch(2).unwrap();
        let mut digest = poseidon_public_digest(&batch).unwrap();
        digest[0] += Goldilocks::ONE;
        let err = match prepare_signature_batch_with_public_digest(batch.instances().to_vec(), digest)
        {
            Ok(_) => panic!("preparation should reject a mismatched public digest"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            SignatureBatchProofError::PublicDigestMismatch { .. }
        ));
    }

    #[test]
    fn public_inputs_wire_round_trip() {
        let batch = deterministic_batch(2).unwrap();
        let public_inputs = SignatureBatchPublicInputs::from_batch(&batch).unwrap();
        let bytes = public_inputs.to_bytes();

        assert_eq!(bytes.len(), SIGNATURE_BATCH_PUBLIC_INPUT_BYTES);
        assert_eq!(
            SignatureBatchPublicInputs::from_bytes(&bytes).unwrap(),
            public_inputs
        );
    }

    #[test]
    fn proof_wire_round_trip_verifies() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let bytes = output.to_bytes().unwrap();
        let decoded = SignatureBatchProvingOutput::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.public_inputs, output.public_inputs);
        assert_eq!(decoded.proof.log_height(), output.proof.log_height());
        verify_signature_batch_proof(&decoded.public_inputs, &decoded.proof).unwrap();
    }

    #[test]
    fn proof_wire_rejects_wrong_version() {
        let batch = deterministic_batch(1).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let mut bytes = output.to_bytes().unwrap();
        bytes[PROOF_WIRE_MAGIC.len()] ^= 1;

        assert!(matches!(
            SignatureBatchProvingOutput::from_bytes(&bytes),
            Err(SignatureBatchProofError::UnsupportedWireFormatVersion { .. })
        ));
    }

    #[test]
    fn proof_wire_rejects_wrong_fri_profile() {
        let batch = deterministic_batch(1).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let mut bytes = output.to_bytes().unwrap();
        let fri_profile_offset = PROOF_WIRE_MAGIC.len() + 2 + 2;
        bytes[fri_profile_offset] ^= 1;

        assert!(matches!(
            SignatureBatchProvingOutput::from_bytes(&bytes),
            Err(SignatureBatchProofError::UnsupportedFriProfileId { .. })
        ));
    }

    #[test]
    fn proof_wire_rejects_noncanonical_digest() {
        let batch = deterministic_batch(1).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let mut bytes = output.to_bytes().unwrap();
        let digest_offset = PROOF_WIRE_MAGIC.len() + 2 + 2 + 2 + 2 + 8;
        bytes[digest_offset..digest_offset + 8]
            .copy_from_slice(&Goldilocks::ORDER_U64.to_le_bytes());

        assert!(matches!(
            SignatureBatchProvingOutput::from_bytes(&bytes),
            Err(SignatureBatchProofError::NonCanonicalPublicDigest { lane: 0, .. })
        ));
    }

    #[test]
    fn proof_wire_rejects_wrong_signature_count_height() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let mut bytes = output.to_bytes().unwrap();
        let count_offset = PROOF_WIRE_MAGIC.len() + 2 + 2 + 2 + 2;
        bytes[count_offset..count_offset + 8].copy_from_slice(&500u64.to_le_bytes());

        assert!(matches!(
            SignatureBatchProvingOutput::from_bytes(&bytes),
            Err(SignatureBatchProofError::TraceHeightMismatch { .. })
        ));
    }

    #[test]
    fn proof_wire_lowered_count_decodes_but_fails_verification() {
        let batch = deterministic_batch(2).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let mut bytes = output.to_bytes().unwrap();
        let count_offset = PROOF_WIRE_MAGIC.len() + 2 + 2 + 2 + 2;
        bytes[count_offset..count_offset + 8].copy_from_slice(&1u64.to_le_bytes());

        let decoded = SignatureBatchProvingOutput::from_bytes(&bytes).unwrap();
        assert!(verify_signature_batch_proof(&decoded.public_inputs, &decoded.proof).is_err());
    }

    #[test]
    fn proof_wire_rejects_corrupted_payload() {
        let batch = deterministic_batch(1).unwrap();
        let output = prove_signature_batch(&batch).unwrap();
        let mut bytes = output.to_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;

        let decoded = SignatureBatchProvingOutput::from_bytes(&bytes);
        assert!(
            decoded.is_err(),
            "corrupted proof bytes should not deserialize as a valid proof"
        );
    }
}
