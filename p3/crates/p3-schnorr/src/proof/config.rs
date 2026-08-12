// The concrete Plonky3 STARK configuration `SignatureBatchAir` is
// proved/verified under: extension field, hash/compression/MMCS built on the
// same Poseidon2 permutation as the AIR's own public-transcript hash, and
// FRI parameters. Changing any of these changes the proof's security level
// and/or wire format and must be paired with bumping
// `SIGNATURE_BATCH_FRI_PROFILE_ID` so old and new proofs can't be confused
// for one another.

/// Poseidon2 state width for the FRI transcript/Merkle hasher below. This is
/// the stock width-16 `p3_goldilocks` Poseidon2, deliberately independent of
/// the (width-12, Plonky2-compatible) public-digest sponge in
/// [`crate::poseidon`]: the recursive verifier's transcript replay and its
/// `P3Poseidon2Gate` are built against this permutation.
pub const TRANSCRIPT_POSEIDON_WIDTH: usize = 16;
/// Sponge rate for the FRI transcript/Merkle hasher.
pub const TRANSCRIPT_POSEIDON_RATE: usize = 8;
/// Digest width for the FRI transcript's Merkle caps and compressions.
pub const TRANSCRIPT_DIGEST_WIDTH: usize = 8;

// FRI parameters named for the constraint that produced them: this STARK's
// FRI shape is fixed to match what the `p3-schnorr-recursion` crate's
// Plonky2 circuit expects to verify, not chosen independently. Changing any
// of these without updating the recursive verifier circuit would make
// proofs from this crate unverifiable there.
/// `log2` of the FRI blowup factor (rate).
pub const PLONKY2_FRI_RATE_BITS: usize = 3;
/// Height of the Merkle caps committed to by the value/challenge MMCS.
pub const PLONKY2_FRI_CAP_HEIGHT: usize = 4;
/// Grinding (proof-of-work) bits required for the FRI query phase.
pub const PLONKY2_FRI_PROOF_OF_WORK_BITS: usize = 16;
/// Maximum `log2` folding arity per FRI commit-phase round.
pub const PLONKY2_FRI_REDUCTION_ARITY_BITS: usize = 4;
/// `log2` of the final FRI polynomial's length.
pub const PLONKY2_FRI_FINAL_POLY_BITS: usize = 5;
/// Number of FRI query rounds (controls soundness error along with the blowup).
pub const PLONKY2_FRI_NUM_QUERY_ROUNDS: usize = 28;

// Wire-format identifiers: bumped whenever the corresponding format changes
// in a way that would make an old decoder misinterpret new bytes (or
// vice versa).
/// Version of the public-input/proof wire encoding itself (field order, widths).
pub const SIGNATURE_BATCH_WIRE_FORMAT_VERSION: u16 = 2;
/// Identifies "this is a signature-batch proof" as opposed to some other
/// statement that might reuse the same wire envelope.
pub const SIGNATURE_BATCH_STATEMENT_ID: u16 = 1;
/// Identifies the specific FRI parameter profile (the `PLONKY2_FRI_*`
/// constants above) a proof was generated under.
pub const SIGNATURE_BATCH_FRI_PROFILE_ID: u16 = 1;
/// Byte length of [`SignatureBatchPublicInputs::to_bytes`]'s output.
pub const SIGNATURE_BATCH_PUBLIC_INPUT_BYTES: usize = 52;

const PUBLIC_INPUT_WIRE_MAGIC: [u8; 8] = *b"P3SNPUB\0";
const PROOF_WIRE_MAGIC: [u8; 8] = *b"P3SNPRF\0";

/// Challenge (out-of-domain sampling) field: a degree-2 extension of Goldilocks.
pub type Challenge = BinomialExtensionField<Goldilocks, 2>;
/// The Poseidon2 permutation backing every hash/compression/challenger below.
pub type Perm = Poseidon2Goldilocks<TRANSCRIPT_POSEIDON_WIDTH>;
/// Sponge-based hash used for Merkle leaves.
pub type Hash = PaddingFreeSponge<
    Perm,
    TRANSCRIPT_POSEIDON_WIDTH,
    TRANSCRIPT_POSEIDON_RATE,
    TRANSCRIPT_DIGEST_WIDTH,
>;
/// 2-to-1 compression function for internal Merkle tree nodes.
pub type Compress = TruncatedPermutation<Perm, 2, TRANSCRIPT_DIGEST_WIDTH, TRANSCRIPT_POSEIDON_WIDTH>;
/// Discrete Fourier transform used by the PCS to interpolate/evaluate traces.
pub type Dft = Radix2DitParallel<Goldilocks>;
/// Merkle-tree-based commitment scheme for base-field (trace/quotient) matrices.
pub type ValMmcs = MerkleTreeMmcs<
    <Goldilocks as Field>::Packing,
    <Goldilocks as Field>::Packing,
    Hash,
    Compress,
    2,
    TRANSCRIPT_DIGEST_WIDTH,
>;
/// [`ValMmcs`] lifted to commit to `Challenge`-field matrices (FRI's folded codewords).
pub type ChallengeMmcs = ExtensionMmcs<Goldilocks, Challenge, ValMmcs>;
/// The polynomial commitment scheme: FRI over [`Dft`]/[`ValMmcs`]/[`ChallengeMmcs`].
pub type Pcs = TwoAdicFriPcs<Goldilocks, Dft, ValMmcs, ChallengeMmcs>;
/// Fiat-Shamir transcript, built from the same permutation as the hash/compression.
pub type Challenger =
    DuplexChallenger<Goldilocks, Perm, TRANSCRIPT_POSEIDON_WIDTH, TRANSCRIPT_POSEIDON_RATE>;
/// The complete `p3_uni_stark` configuration `SignatureBatchAir` is proved/verified under.
pub type SignatureBatchStarkConfig = StarkConfig<Pcs, Challenge, Challenger>;
/// The native Plonky3 proof type, before this crate's wire-format wrapping.
pub type InnerSignatureBatchProof = Proof<SignatureBatchStarkConfig>;
/// Verification failure type from `p3_uni_stark`, wrapped by [`SignatureBatchProofError::Verification`].
pub type SignatureBatchVerificationError = VerificationError<PcsError<SignatureBatchStarkConfig>>;
