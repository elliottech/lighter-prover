/// `2^log_height`, checked against shifting out of range — the inverse of
/// `usize::ilog2`, used when decoding an untrusted `log_height` from the wire.
fn rows_for_log_height(log_height: usize) -> Option<usize> {
    if log_height >= usize::BITS as usize {
        None
    } else {
        Some(1usize << log_height)
    }
}

/// The `bincode` configuration the inner Plonky3 proof is (de)serialized
/// with: little-endian, fixed-width integers (not varint), and trailing
/// bytes rejected. Fixed-width + reject-trailing keeps the encoding
/// unambiguous, which matters because [`SignatureBatchProof::from_bytes`]
/// also independently checks the payload length before calling this codec.
fn proof_codec() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_little_endian()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Writes a [`SignatureBatchPublicInputs`]' body (signature count, then
/// digest limbs) — shared by both the standalone public-input wire format
/// and the public-input section embedded in a proof's wire format.
fn write_public_inputs_body(out: &mut Vec<u8>, public_inputs: &SignatureBatchPublicInputs) {
    push_u64(out, public_inputs.signature_count() as u64);
    for digest in public_inputs.digest() {
        push_u64(out, digest.as_canonical_u64());
    }
}

type WireReader<'a> = crate::wire_reader::ByteReader<'a>;

fn read_wire_version(reader: &mut WireReader<'_>) -> Result<(), SignatureBatchProofError> {
    let actual = reader.read_u16()?;
    if actual != SIGNATURE_BATCH_WIRE_FORMAT_VERSION {
        return Err(SignatureBatchProofError::UnsupportedWireFormatVersion {
            expected: SIGNATURE_BATCH_WIRE_FORMAT_VERSION,
            actual,
        });
    }
    Ok(())
}

fn read_statement_id(reader: &mut WireReader<'_>) -> Result<(), SignatureBatchProofError> {
    let actual = reader.read_u16()?;
    if actual != SIGNATURE_BATCH_STATEMENT_ID {
        return Err(SignatureBatchProofError::UnsupportedStatementId {
            expected: SIGNATURE_BATCH_STATEMENT_ID,
            actual,
        });
    }
    Ok(())
}

fn read_fri_profile_id(reader: &mut WireReader<'_>) -> Result<(), SignatureBatchProofError> {
    let actual = reader.read_u16()?;
    if actual != SIGNATURE_BATCH_FRI_PROFILE_ID {
        return Err(SignatureBatchProofError::UnsupportedFriProfileId {
            expected: SIGNATURE_BATCH_FRI_PROFILE_ID,
            actual,
        });
    }
    Ok(())
}

/// Inverse of [`write_public_inputs_body`], validating each digest limb is a
/// canonical Goldilocks value and the signature count is in range (via
/// [`SignatureBatchPublicInputs::new`]).
fn read_public_inputs_body(
    reader: &mut WireReader<'_>,
) -> Result<SignatureBatchPublicInputs, SignatureBatchProofError> {
    let count_u64 = reader.read_u64()?;
    let count = usize::try_from(count_u64).map_err(|_| {
        SignatureBatchProofError::InvalidSignatureCount {
            count: usize::MAX,
            max: MAX_SIGNATURES,
        }
    })?;
    let mut digest = [Goldilocks::ZERO; PUBLIC_DIGEST_WIDTH];
    for (lane, digest) in digest.iter_mut().enumerate() {
        let value = reader.read_u64()?;
        if value >= Goldilocks::ORDER_U64 {
            return Err(SignatureBatchProofError::NonCanonicalPublicDigest { lane, value });
        }
        *digest = Goldilocks::from_u64(value);
    }
    SignatureBatchPublicInputs::new(count, digest)
}

