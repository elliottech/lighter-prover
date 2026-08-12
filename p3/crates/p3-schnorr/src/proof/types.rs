#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignatureBatchPublicInputs {
    signature_count: usize,
    digest: [Goldilocks; PUBLIC_DIGEST_WIDTH],
}

impl SignatureBatchPublicInputs {
    /// Builds public inputs while enforcing the production batch-size policy.
    pub fn new(
        signature_count: usize,
        digest: [Goldilocks; PUBLIC_DIGEST_WIDTH],
    ) -> Result<Self, SignatureBatchProofError> {
        if signature_count == 0 || signature_count > MAX_SIGNATURES {
            return Err(SignatureBatchProofError::InvalidSignatureCount {
                count: signature_count,
                max: MAX_SIGNATURES,
            });
        }
        Ok(Self {
            signature_count,
            digest,
        })
    }

    /// Computes the public inputs committed to by `batch`.
    pub fn from_batch(batch: &Batch) -> Result<Self, SignatureBatchProofError> {
        Self::new(batch.len(), poseidon_public_digest(batch)?)
    }

    /// Parses the raw STARK public value vector `[signature_count, digest[0], ..., digest[7]]`.
    pub fn from_public_values(values: &[Goldilocks]) -> Result<Self, SignatureBatchProofError> {
        if values.len() != PUBLIC_VALUES {
            return Err(SignatureBatchProofError::PublicInputLengthMismatch {
                expected: PUBLIC_VALUES,
                actual: values.len(),
            });
        }
        let count = usize::try_from(values[0].as_canonical_u64()).map_err(|_| {
            SignatureBatchProofError::InvalidSignatureCount {
                count: usize::MAX,
                max: MAX_SIGNATURES,
            }
        })?;
        let digest = core::array::from_fn(|lane| values[1 + lane]);
        Self::new(count, digest)
    }

    /// Number of signatures claimed by this proof.
    pub fn signature_count(&self) -> usize {
        self.signature_count
    }

    /// Eight Goldilocks field elements of public transcript digest.
    pub fn digest(&self) -> &[Goldilocks; PUBLIC_DIGEST_WIDTH] {
        &self.digest
    }

    /// Converts typed inputs back to the raw STARK public value vector.
    pub fn public_values(&self) -> [Goldilocks; PUBLIC_VALUES] {
        let mut values = [Goldilocks::ZERO; PUBLIC_VALUES];
        values[0] = Goldilocks::from_usize(self.signature_count);
        values[1..].copy_from_slice(&self.digest);
        values
    }

    /// Serializes public inputs to the stable wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SIGNATURE_BATCH_PUBLIC_INPUT_BYTES);
        out.extend_from_slice(&PUBLIC_INPUT_WIRE_MAGIC);
        push_u16(&mut out, SIGNATURE_BATCH_WIRE_FORMAT_VERSION);
        push_u16(&mut out, SIGNATURE_BATCH_STATEMENT_ID);
        write_public_inputs_body(&mut out, self);
        out
    }

    /// Deserializes public inputs from the stable wire format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SignatureBatchProofError> {
        let mut reader = WireReader::new(bytes);
        if reader.read_array::<8>()? != PUBLIC_INPUT_WIRE_MAGIC {
            return Err(SignatureBatchProofError::InvalidPublicInputWireMagic);
        }
        read_wire_version(&mut reader)?;
        read_statement_id(&mut reader)?;
        let public_inputs = read_public_inputs_body(&mut reader)?;
        if reader.remaining() != 0 {
            return Err(SignatureBatchProofError::TrailingWireBytes {
                trailing: reader.remaining(),
            });
        }
        Ok(public_inputs)
    }

    /// Expected trace height for this signature count.
    pub fn expected_rows(&self) -> usize {
        trace_height_for_count(self.signature_count)
    }
}

/// Opaque proof object for the signature-batch AIR.
pub struct SignatureBatchProof {
    inner: InnerSignatureBatchProof,
    log_height: usize,
}

impl SignatureBatchProof {
    /// Returns the underlying Plonky3 proof for serialization or advanced use.
    pub fn inner(&self) -> &InnerSignatureBatchProof {
        &self.inner
    }

    /// Base-2 logarithm of the trace height used by this proof.
    pub fn log_height(&self) -> usize {
        self.log_height
    }

    /// Trace height used by this proof.
    pub fn rows(&self) -> usize {
        rows_for_log_height(self.log_height).expect("SignatureBatchProof log_height is valid")
    }

    /// Serializes this proof and its public inputs to the stable wire format.
    pub fn to_bytes(
        &self,
        public_inputs: &SignatureBatchPublicInputs,
    ) -> Result<Vec<u8>, SignatureBatchProofError> {
        ensure_trace_height(public_inputs, self.rows())?;
        let payload = proof_codec().serialize(self.inner())?;
        let mut out = Vec::with_capacity(100 + payload.len());
        out.extend_from_slice(&PROOF_WIRE_MAGIC);
        push_u16(&mut out, SIGNATURE_BATCH_WIRE_FORMAT_VERSION);
        push_u16(&mut out, SIGNATURE_BATCH_STATEMENT_ID);
        push_u16(&mut out, SIGNATURE_BATCH_FRI_PROFILE_ID);
        push_u16(&mut out, 0);
        write_public_inputs_body(&mut out, public_inputs);
        push_u32(
            &mut out,
            u32::try_from(self.log_height()).map_err(|_| {
                SignatureBatchProofError::InvalidLogHeight {
                    log_height: self.log_height(),
                }
            })?,
        );
        push_u64(
            &mut out,
            u64::try_from(payload.len())
                .map_err(|_| SignatureBatchProofError::ProofPayloadTooLarge { len: u64::MAX })?,
        );
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Deserializes a proof and its public inputs from the stable wire format.
    pub fn from_bytes(
        bytes: &[u8],
    ) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
        let mut reader = WireReader::new(bytes);
        if reader.read_array::<8>()? != PROOF_WIRE_MAGIC {
            return Err(SignatureBatchProofError::InvalidProofWireMagic);
        }
        read_wire_version(&mut reader)?;
        read_statement_id(&mut reader)?;
        read_fri_profile_id(&mut reader)?;
        let reserved = reader.read_u16()?;
        if reserved != 0 {
            return Err(SignatureBatchProofError::NonZeroWireReserved { actual: reserved });
        }
        let public_inputs = read_public_inputs_body(&mut reader)?;
        let log_height = usize::try_from(reader.read_u32()?).map_err(|_| {
            SignatureBatchProofError::InvalidLogHeight {
                log_height: usize::MAX,
            }
        })?;
        let rows = rows_for_log_height(log_height)
            .ok_or(SignatureBatchProofError::InvalidLogHeight { log_height })?;
        ensure_trace_height(&public_inputs, rows)?;
        let proof_len_u64 = reader.read_u64()?;
        let proof_len = usize::try_from(proof_len_u64)
            .map_err(|_| SignatureBatchProofError::ProofPayloadTooLarge { len: proof_len_u64 })?;
        if reader.remaining() != proof_len {
            return Err(SignatureBatchProofError::ProofPayloadLengthMismatch {
                expected: proof_len,
                actual: reader.remaining(),
            });
        }
        let inner: InnerSignatureBatchProof =
            proof_codec().deserialize(reader.read_bytes(proof_len)?)?;
        if inner.degree_bits != log_height {
            return Err(SignatureBatchProofError::ProofLogHeightMismatch {
                expected: log_height,
                actual: inner.degree_bits,
            });
        }
        Ok(SignatureBatchProvingOutput {
            proof: SignatureBatchProof { inner, log_height },
            public_inputs,
        })
    }
}

/// Reusable verifier data for a fixed signature-batch trace height.
pub struct PreparedSignatureBatchVerifier {
    air: SignatureBatchAir,
    log_height: usize,
    preprocessed_vk: PreprocessedVerifierKey<SignatureBatchStarkConfig>,
}

impl PreparedSignatureBatchVerifier {
    /// Base-2 logarithm of the trace height supported by this verifier.
    pub fn log_height(&self) -> usize {
        self.log_height
    }

    /// Trace height supported by this verifier.
    pub fn rows(&self) -> usize {
        rows_for_log_height(self.log_height)
            .expect("PreparedSignatureBatchVerifier log_height is valid")
    }

    /// Verifies `proof` against `public_inputs` using the cached preprocessed VK.
    pub fn verify(
        &self,
        public_inputs: &SignatureBatchPublicInputs,
        proof: &SignatureBatchProof,
    ) -> Result<(), SignatureBatchProofError> {
        ensure_trace_height(public_inputs, self.rows())?;
        if proof.log_height() != self.log_height {
            return Err(SignatureBatchProofError::TraceHeightMismatch {
                signature_count: public_inputs.signature_count(),
                expected_rows: self.rows(),
                actual_rows: proof.rows(),
            });
        }
        let config = schnorr_stark_config();
        let public_values = public_inputs.public_values();
        verify_with_preprocessed(
            &config,
            &self.air,
            proof.inner(),
            &public_values,
            Some(&self.preprocessed_vk),
        )?;
        Ok(())
    }
}

/// Successful proving result, pairing a proof with the public inputs it proves.
pub struct SignatureBatchProvingOutput {
    pub proof: SignatureBatchProof,
    pub public_inputs: SignatureBatchPublicInputs,
}

impl SignatureBatchProvingOutput {
    /// Serializes the proof and public inputs to the stable wire format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SignatureBatchProofError> {
        self.proof.to_bytes(&self.public_inputs)
    }

    /// Deserializes a proof and its public inputs from the stable wire format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SignatureBatchProofError> {
        SignatureBatchProof::from_bytes(bytes)
    }
}

/// Batch plus validated public inputs and challenge hints ready for proving.
pub struct PreparedSignatureBatch {
    pub batch: Batch,
    pub public_inputs: SignatureBatchPublicInputs,
    pub challenge_hints: Vec<SignatureChallengeHint>,
}

impl PreparedSignatureBatch {
    /// Proves this prepared batch using its validated challenge hints.
    pub fn prove(&self) -> Result<SignatureBatchProvingOutput, SignatureBatchProofError> {
        prove_signature_batch_with_challenge_hints(&self.batch, &self.challenge_hints)
    }
}

/// Errors returned by the production proof API.
#[derive(Debug, Error)]
pub enum SignatureBatchProofError {
    #[error(transparent)]
    Input(#[from] crate::Error),

    #[error("public input length mismatch: expected {expected}, got {actual}")]
    PublicInputLengthMismatch { expected: usize, actual: usize },

    #[error("invalid signature count {count}; expected 1..={max}")]
    InvalidSignatureCount { count: usize, max: usize },

    #[error("generated trace public inputs do not match the input batch")]
    PublicInputMismatch,

    #[error("public digest does not match the supplied batch")]
    PublicDigestMismatch {
        expected: Box<[Goldilocks; PUBLIC_DIGEST_WIDTH]>,
        actual: Box<[Goldilocks; PUBLIC_DIGEST_WIDTH]>,
    },

    #[error("invalid public-input wire magic")]
    InvalidPublicInputWireMagic,

    #[error("invalid proof wire magic")]
    InvalidProofWireMagic,

    #[error("unsupported wire format version {actual}; expected {expected}")]
    UnsupportedWireFormatVersion { expected: u16, actual: u16 },

    #[error("unsupported signature-batch statement id {actual}; expected {expected}")]
    UnsupportedStatementId { expected: u16, actual: u16 },

    #[error("unsupported FRI profile id {actual}; expected {expected}")]
    UnsupportedFriProfileId { expected: u16, actual: u16 },

    #[error("wire reserved field must be zero, got {actual}")]
    NonZeroWireReserved { actual: u16 },

    #[error("wire payload is truncated")]
    TruncatedWirePayload,

    #[error("wire payload has {trailing} trailing bytes")]
    TrailingWireBytes { trailing: usize },

    #[error("non-canonical public digest lane {lane}: 0x{value:016x}")]
    NonCanonicalPublicDigest { lane: usize, value: u64 },

    #[error("invalid proof log height {log_height}")]
    InvalidLogHeight { log_height: usize },

    #[error("proof log height mismatch: expected {expected}, got inner proof degree_bits {actual}")]
    ProofLogHeightMismatch { expected: usize, actual: usize },

    #[error("proof payload is too large: {len}")]
    ProofPayloadTooLarge { len: u64 },

    #[error("proof payload length mismatch: expected {expected}, got {actual}")]
    ProofPayloadLengthMismatch { expected: usize, actual: usize },

    #[error(
        "trace height mismatch for {signature_count} signatures: expected {expected_rows}, got {actual_rows}"
    )]
    TraceHeightMismatch {
        signature_count: usize,
        expected_rows: usize,
        actual_rows: usize,
    },

    #[error("signature-batch AIR has no preprocessed columns")]
    MissingPreprocessedColumns,

    #[error(transparent)]
    Verification(#[from] SignatureBatchVerificationError),

    #[error("proof codec error: {0}")]
    ProofCodec(#[from] Box<bincode::ErrorKind>),
}
