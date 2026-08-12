// Out-of-circuit "hints": prover-computed witnesses needed to decode an
// encoded ECgFp5 point (`reconstructed_r` and the public key) that the AIR's
// `constrain_encoded_point_decode` (see `constraints.rs`) checks but cannot
// derive on its own, because computing them requires a square root and a
// Legendre-symbol root choice — operations with no native AIR gate. A
// verifier must receive these alongside the proof; see
// [`SignatureChallengeHint`]'s wire (de)serialization for the format they
// travel in. [`encoded_point_decode_hint`] is the host-side computation that
// produces them, mirroring the curve equation `x^2 - e*x + B = 0` (`e` derived
// from the encoding) that `constrain_encoded_point_decode` checks in-circuit.

/// Witnesses proving one encoded ECgFp5 point decodes to a specific
/// x-coordinate: the encoding's inverse (`u = 1/encoded`), `sqrt(delta)` from
/// the quadratic formula solving for `x`, a square root disambiguating which
/// of the two roots is the real `x` (via Legendre symbol), and the resulting
/// `affine_x` itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodedPointDecodeHint {
    pub inverse: [Goldilocks; FP5_LIMBS],
    pub sqrt_delta: [Goldilocks; FP5_LIMBS],
    pub other_root_sqrt: [Goldilocks; FP5_LIMBS],
    pub affine_x: [Goldilocks; FP5_LIMBS],
}

impl EncodedPointDecodeHint {
    pub fn new(
        inverse: [Goldilocks; FP5_LIMBS],
        sqrt_delta: [Goldilocks; FP5_LIMBS],
        other_root_sqrt: [Goldilocks; FP5_LIMBS],
        affine_x: [Goldilocks; FP5_LIMBS],
    ) -> Self {
        Self {
            inverse,
            sqrt_delta,
            other_root_sqrt,
            affine_x,
        }
    }
}

/// Everything the STARK prover must additionally supply per signature, beyond
/// the public batch digest: the encoded `reconstructed_r = s*G + e*pk` (which
/// the AIR re-derives the challenge from and separately re-verifies against
/// `s`/`e`/`pk` via the EC rows) and the point-decode hints needed to turn
/// both `reconstructed_r` and the public key from their encoded form into
/// affine coordinates in-circuit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignatureChallengeHint {
    pub reconstructed_r: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH],
    pub reconstructed_r_decode: EncodedPointDecodeHint,
    pub public_key_decode: EncodedPointDecodeHint,
}

impl SignatureChallengeHint {
    pub fn new(
        reconstructed_r: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH],
        reconstructed_r_decode: EncodedPointDecodeHint,
        public_key_decode: EncodedPointDecodeHint,
    ) -> Self {
        Self {
            reconstructed_r,
            reconstructed_r_decode,
            public_key_decode,
        }
    }

    /// Serializes this hint in the wire format `from_bytes` reads: an 8-byte
    /// magic, then format-version and statement-id `u16`s, then the hint body.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SIGNATURE_CHALLENGE_HINT_BYTES);
        out.extend_from_slice(&SIGNATURE_CHALLENGE_HINT_WIRE_MAGIC);
        push_u16(&mut out, SIGNATURE_CHALLENGE_HINT_WIRE_FORMAT_VERSION);
        push_u16(&mut out, SIGNATURE_CHALLENGE_HINT_STATEMENT_ID);
        write_signature_challenge_hint_body(&mut out, self);
        out
    }

    /// Decodes one hint from `to_bytes`'s wire format, validating the magic,
    /// version, statement id, every limb's canonicality, and that no trailing
    /// bytes remain.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = crate::wire_reader::ByteReader::new(bytes);
        if reader.read_array::<8>()? != SIGNATURE_CHALLENGE_HINT_WIRE_MAGIC {
            return Err(crate::Error::InvalidChallengeHintWireMagic);
        }
        read_challenge_hint_wire_version(&mut reader)?;
        read_challenge_hint_statement_id(&mut reader)?;
        let hint = read_signature_challenge_hint_body(&mut reader)?;
        if reader.remaining() != 0 {
            return Err(crate::Error::TrailingChallengeHintWireBytes {
                trailing: reader.remaining(),
            });
        }
        Ok(hint)
    }

    /// Builds a hint for `instance` given an already-reconstructed `R`, using
    /// the `goldilocks-crypto` reference implementation to compute the decode
    /// witnesses (square roots, inverses) that this crate's own AIR-oriented
    /// arithmetic doesn't implement directly.
    #[cfg(feature = "reference")]
    pub fn from_reference(
        instance: &crate::types::SchnorrInstance,
        reconstructed_r: [Goldilocks; LIGHTER_POSEIDON_DIGEST_WIDTH],
    ) -> Result<Self> {
        let public_key = instance.public_key.to_goldilocks_limbs("public_key")?;
        Ok(Self::new(
            reconstructed_r,
            encoded_point_decode_hint(reconstructed_r, "reconstructed_r")?,
            encoded_point_decode_hint(public_key, "public_key")?,
        ))
    }
}

/// Serializes a whole batch of hints (one per signature, in batch order) as a
/// single wire payload: an 8-byte magic, version/statement-id `u16`s, a
/// `u64` count, then each hint's body back-to-back.
pub fn signature_challenge_hints_to_bytes(hints: &[SignatureChallengeHint]) -> Result<Vec<u8>> {
    if hints.len() > MAX_SIGNATURES {
        return Err(crate::Error::TooManyChallengeHints {
            actual: hints.len(),
            max: MAX_SIGNATURES,
        });
    }
    let mut out = Vec::with_capacity(
        SIGNATURE_CHALLENGE_HINT_BATCH_HEADER_BYTES
            + hints.len() * SIGNATURE_CHALLENGE_HINT_FIELD_ELEMENTS * 8,
    );
    out.extend_from_slice(&SIGNATURE_CHALLENGE_HINT_BATCH_WIRE_MAGIC);
    push_u16(&mut out, SIGNATURE_CHALLENGE_HINT_WIRE_FORMAT_VERSION);
    push_u16(&mut out, SIGNATURE_CHALLENGE_HINT_STATEMENT_ID);
    push_u64(&mut out, hints.len() as u64);
    for hint in hints {
        write_signature_challenge_hint_body(&mut out, hint);
    }
    Ok(out)
}

/// Decodes a batch payload from [`signature_challenge_hints_to_bytes`],
/// validating the magic, version, statement id, declared count against
/// [`crate::types::MAX_SIGNATURES`], and that the payload length matches
/// exactly (no truncation or trailing bytes).
pub fn signature_challenge_hints_from_bytes(bytes: &[u8]) -> Result<Vec<SignatureChallengeHint>> {
    let mut reader = crate::wire_reader::ByteReader::new(bytes);
    if reader.read_array::<8>()? != SIGNATURE_CHALLENGE_HINT_BATCH_WIRE_MAGIC {
        return Err(crate::Error::InvalidChallengeHintBatchWireMagic);
    }
    read_challenge_hint_wire_version(&mut reader)?;
    read_challenge_hint_statement_id(&mut reader)?;
    let count_u64 = reader.read_u64()?;
    let count = usize::try_from(count_u64).map_err(|_| crate::Error::TooManyChallengeHints {
        actual: usize::MAX,
        max: MAX_SIGNATURES,
    })?;
    if count > MAX_SIGNATURES {
        return Err(crate::Error::TooManyChallengeHints {
            actual: count,
            max: MAX_SIGNATURES,
        });
    }
    let expected_body_bytes = count
        .checked_mul(SIGNATURE_CHALLENGE_HINT_FIELD_ELEMENTS * 8)
        .ok_or(crate::Error::TruncatedWirePayload)?;
    if reader.remaining() < expected_body_bytes {
        return Err(crate::Error::TruncatedWirePayload);
    }
    if reader.remaining() > expected_body_bytes {
        return Err(crate::Error::TrailingChallengeHintWireBytes {
            trailing: reader.remaining() - expected_body_bytes,
        });
    }
    let mut hints = Vec::with_capacity(count);
    for _ in 0..count {
        hints.push(read_signature_challenge_hint_body(&mut reader)?);
    }
    Ok(hints)
}

fn write_signature_challenge_hint_body(out: &mut Vec<u8>, hint: &SignatureChallengeHint) {
    write_hint_fp5(out, &hint.reconstructed_r);
    write_encoded_point_decode_hint(out, &hint.reconstructed_r_decode);
    write_encoded_point_decode_hint(out, &hint.public_key_decode);
}

fn write_encoded_point_decode_hint(out: &mut Vec<u8>, hint: &EncodedPointDecodeHint) {
    write_hint_fp5(out, &hint.inverse);
    write_hint_fp5(out, &hint.sqrt_delta);
    write_hint_fp5(out, &hint.other_root_sqrt);
    write_hint_fp5(out, &hint.affine_x);
}

fn write_hint_fp5(out: &mut Vec<u8>, value: &[Goldilocks; FP5_LIMBS]) {
    for limb in value {
        push_u64(out, limb.as_canonical_u64());
    }
}

fn read_signature_challenge_hint_body(
    reader: &mut crate::wire_reader::ByteReader<'_>,
) -> Result<SignatureChallengeHint> {
    let reconstructed_r = read_hint_fp5(reader, "reconstructed_r")?;
    let reconstructed_r_decode = read_encoded_point_decode_hint(reader, "reconstructed_r_decode")?;
    let public_key_decode = read_encoded_point_decode_hint(reader, "public_key_decode")?;
    Ok(SignatureChallengeHint::new(
        reconstructed_r,
        reconstructed_r_decode,
        public_key_decode,
    ))
}

fn read_encoded_point_decode_hint(
    reader: &mut crate::wire_reader::ByteReader<'_>,
    field: &'static str,
) -> Result<EncodedPointDecodeHint> {
    Ok(EncodedPointDecodeHint::new(
        read_hint_fp5(reader, field)?,
        read_hint_fp5(reader, field)?,
        read_hint_fp5(reader, field)?,
        read_hint_fp5(reader, field)?,
    ))
}

fn read_hint_fp5(
    reader: &mut crate::wire_reader::ByteReader<'_>,
    field: &'static str,
) -> Result<[Goldilocks; FP5_LIMBS]> {
    let mut value = [Goldilocks::ZERO; FP5_LIMBS];
    for (limb, out) in value.iter_mut().enumerate() {
        let raw = reader.read_u64()?;
        if raw >= Goldilocks::ORDER_U64 {
            return Err(crate::Error::NonCanonicalGoldilocksLimb {
                field,
                limb,
                value: raw,
            });
        }
        *out = Goldilocks::from_u64(raw);
    }
    Ok(value)
}

fn read_challenge_hint_wire_version(reader: &mut crate::wire_reader::ByteReader<'_>) -> Result<()> {
    let actual = reader.read_u16()?;
    if actual != SIGNATURE_CHALLENGE_HINT_WIRE_FORMAT_VERSION {
        return Err(crate::Error::UnsupportedChallengeHintWireFormatVersion {
            expected: SIGNATURE_CHALLENGE_HINT_WIRE_FORMAT_VERSION,
            actual,
        });
    }
    Ok(())
}

fn read_challenge_hint_statement_id(reader: &mut crate::wire_reader::ByteReader<'_>) -> Result<()> {
    let actual = reader.read_u16()?;
    if actual != SIGNATURE_CHALLENGE_HINT_STATEMENT_ID {
        return Err(crate::Error::UnsupportedChallengeHintStatementId {
            expected: SIGNATURE_CHALLENGE_HINT_STATEMENT_ID,
            actual,
        });
    }
    Ok(())
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Decodes an encoded point (`encoded = 1/u = y/x`) to its `x`-coordinate and
/// the witnesses `constrain_encoded_point_decode` (in `constraints.rs`) needs
/// to check that decoding in-circuit.
///
/// Derivation: from `e := encoded^2 - A`, the curve equation gives
/// `x^2 - e*x + B = 0`, so `x = (e +/- sqrt(e^2 - 4B)) / 2` by the quadratic
/// formula (`delta = e^2 - 4B`). Both roots multiply to `B` (a fixed
/// non-square curve parameter), so exactly one root is a square — the
/// decoder picks the non-square root as the canonical `x` (matching the
/// reference implementation's convention) via a Legendre-symbol check, then
/// additionally proves `e - x` (the other root) *is* a square, which is what
/// lets the AIR verify the choice without computing a Legendre symbol itself.
/// Fails with [`crate::Error::InvalidEncodedPoint`] if `encoded` is zero or
/// `delta`/`other_root` aren't actually squares, meaning the input doesn't
/// encode a real curve point.
#[cfg(feature = "reference")]
fn encoded_point_decode_hint(
    encoded: [Goldilocks; FP5_LIMBS],
    field: &'static str,
) -> Result<EncodedPointDecodeHint> {
    let encoded = crypto_fp5_from_limbs(&encoded);
    if encoded.is_zero() {
        return Err(crate::Error::InvalidEncodedPoint { field });
    }

    let inverse = encoded.inverse();
    let mut e = encoded.square();
    e.0[0] = e.0[0].sub(&goldilocks_crypto::Goldilocks::from_canonical_u64(
        ECGFP5_A[0],
    ));
    let mut delta = e.square();
    delta.0[1] = delta.0[1].sub(&goldilocks_crypto::Goldilocks::from_canonical_u64(
        ECGFP5_B_MUL4[1],
    ));
    let (sqrt_delta, success) = delta.canonical_sqrt();
    if !success {
        return Err(crate::Error::InvalidEncodedPoint { field });
    }
    let two_inv = goldilocks_crypto::Fp5Element::from_uint64_array([2, 0, 0, 0, 0]).inverse();
    let x1 = e.add(&sqrt_delta).mul(&two_inv);
    let x2 = e.sub(&sqrt_delta).mul(&two_inv);
    let affine_x = if x1.legendre().equals(&goldilocks_crypto::Goldilocks::one()) {
        x2
    } else {
        x1
    };
    let other_root = e.sub(&affine_x);
    let (other_root_sqrt, other_root_is_square) = other_root.canonical_sqrt();
    if !other_root_is_square {
        return Err(crate::Error::InvalidEncodedPoint { field });
    }

    Ok(EncodedPointDecodeHint {
        inverse: limbs_from_crypto_fp5(&inverse),
        sqrt_delta: limbs_from_crypto_fp5(&sqrt_delta),
        other_root_sqrt: limbs_from_crypto_fp5(&other_root_sqrt),
        affine_x: limbs_from_crypto_fp5(&affine_x),
    })
}

#[cfg(feature = "reference")]
use crate::schnorr::{crypto_fp5_from_limbs, limbs_from_crypto_fp5};
