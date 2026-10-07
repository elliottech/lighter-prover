//! plonky2 proof definition.
//!
//! Proofs can be later compressed to reduce their size, into either
//! [`CompressedProof`] or [`CompressedProofWithPublicInputs`] formats.
//! The latter can be directly passed to a verifier to assert its correctness.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::ensure;
use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::proof::{
    CompressedFriProof, FriChallenges, FriChallengesTarget, FriProof, FriProofTarget,
};
use crate::fri::structure::{
    FriOpeningBatch, FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget,
};
use crate::fri::FriParams;
use crate::hash::hash_types::{MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::verifier::verify_with_challenges;
use crate::util::serialization::{Buffer, Read, Write};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct Proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A batch FRI argument for all openings.
    pub opening_proof: FriProof<F, C::Hasher, D>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProofTarget<const D: usize> {
    pub wires_cap: MerkleCapTarget,
    pub plonk_zs_partial_products_cap: MerkleCapTarget,
    pub quotient_polys_cap: MerkleCapTarget,
    pub openings: OpeningSetTarget<D>,
    pub opening_proof: FriProofTarget<D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Proof<F, C, D> {
    /// Compress the proof.
    pub fn compress(self, indices: &[usize], params: &FriParams) -> CompressedProof<F, C, D> {
        let Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        } = self;

        CompressedProof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof: opening_proof.compress(indices, params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct ProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: Proof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProofWithPublicInputs<F, C, D>
{
    pub fn compress(
        self,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<CompressedProofWithPublicInputs<F, C, D>> {
        let indices = self.fri_query_indices(circuit_digest, common_data)?;
        let compressed_proof = self.proof.compress(&indices, &common_data.fri_params);
        Ok(CompressedProofWithPublicInputs {
            public_inputs: self.public_inputs,
            proof: compressed_proof,
        })
    }

    pub fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedProof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A compressed batch FRI argument for all openings.
    pub opening_proof: CompressedFriProof<F, C::Hasher, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CompressedProof<F, C, D>
{
    /// Decompress the proof.
    pub(crate) fn decompress(
        self,
        challenges: &ProofChallenges<F, D>,
        fri_inferred_elements: FriInferredElements<F, D>,
        params: &FriParams,
    ) -> Proof<F, C, D> {
        let CompressedProof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        } = self;

        Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof: opening_proof.decompress(challenges, fri_inferred_elements, params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: CompressedProof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CompressedProofWithPublicInputs<F, C, D>
{
    pub fn decompress(
        self,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<ProofWithPublicInputs<F, C, D>> {
        let challenges =
            self.get_challenges(self.get_public_inputs_hash(), circuit_digest, common_data)?;
        let fri_inferred_elements = self.get_inferred_elements(&challenges, common_data);
        let decompressed_proof =
            self.proof
                .decompress(&challenges, fri_inferred_elements, &common_data.fri_params);
        Ok(ProofWithPublicInputs {
            public_inputs: self.public_inputs,
            proof: decompressed_proof,
        })
    }

    pub(crate) fn verify(
        self,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<()> {
        ensure!(
            self.public_inputs.len() == common_data.num_public_inputs,
            "Number of public inputs doesn't match circuit data."
        );
        let public_inputs_hash = self.get_public_inputs_hash();
        let challenges = self.get_challenges(
            public_inputs_hash,
            &verifier_data.circuit_digest,
            common_data,
        )?;
        let fri_inferred_elements = self.get_inferred_elements(&challenges, common_data);
        let decompressed_proof =
            self.proof
                .decompress(&challenges, fri_inferred_elements, &common_data.fri_params);
        verify_with_challenges::<F, C, D>(
            decompressed_proof,
            public_inputs_hash,
            challenges,
            verifier_data,
            common_data,
        )
    }

    pub(crate) fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_compressed_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_compressed_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

#[derive(Debug)]
pub struct ProofChallenges<F: RichField + Extendable<D>, const D: usize> {
    /// Random values used in Plonk's permutation argument.
    pub plonk_betas: Vec<F>,

    /// Random values used in Plonk's permutation argument.
    pub plonk_gammas: Vec<F>,

    /// Random values used to combine PLONK constraints.
    pub plonk_alphas: Vec<F>,

    /// Lookup challenges.
    pub plonk_deltas: Vec<F>,

    /// Point at which the PLONK polynomials are opened.
    pub plonk_zeta: F::Extension,

    pub fri_challenges: FriChallenges<F, D>,
}

pub(crate) struct ProofChallengesTarget<const D: usize> {
    pub plonk_betas: Vec<Target>,
    pub plonk_gammas: Vec<Target>,
    pub plonk_alphas: Vec<Target>,
    pub plonk_deltas: Vec<Target>,
    pub plonk_zeta: ExtensionTarget<D>,
    pub fri_challenges: FriChallengesTarget<D>,
}

/// Coset elements that can be inferred in the FRI reduction steps.
pub(crate) struct FriInferredElements<F: RichField + Extendable<D>, const D: usize>(
    pub Vec<F::Extension>,
);

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProofWithPublicInputsTarget<const D: usize> {
    pub proof: ProofTarget<D>,
    pub public_inputs: Vec<Target>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
/// The purported values of each polynomial at a single point.
pub struct OpeningSet<F: RichField + Extendable<D>, const D: usize> {
    pub constants: Vec<F::Extension>,
    pub plonk_sigmas: Vec<F::Extension>,
    pub wires: Vec<F::Extension>,
    pub plonk_zs: Vec<F::Extension>,
    pub plonk_zs_next: Vec<F::Extension>,
    pub partial_products: Vec<F::Extension>,
    pub quotient_polys: Vec<F::Extension>,
    pub lookup_zs: Vec<F::Extension>,
    pub lookup_zs_next: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> OpeningSet<F, D> {
    pub fn new<C: GenericConfig<D, F = F>>(
        zeta: F::Extension,
        g: F::Extension,
        constants_sigmas_commitment: &PolynomialBatch<F, C, D>,
        wires_commitment: &PolynomialBatch<F, C, D>,
        zs_partial_products_lookup_commitment: &PolynomialBatch<F, C, D>,
        quotient_polys_commitment: &PolynomialBatch<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        // Every committed polynomial in these batches has at most `degree`
        // coefficients, so one powers table per opening point covers all of
        // them. Evaluating as `sum_i c_i * z^i` against the table replaces the
        // previous `p.to_extension().eval(z)`, which cloned each polynomial
        // into a degree-sized extension vector (one allocation plus a full
        // conversion pass per polynomial) before running extension-by-
        // extension Horner. The dot product reads the base coefficients in
        // place; `Extendable::extension_base_dot_product` defaults to the old
        // scalar-multiply-and-sum loop and lets concrete fields delay
        // reductions where their representation makes that safe.
        // Value-exactness: the field is exact, `powers()` produces exactly
        // `z^i`, and `sum c_i z^i` under any association equals Horner's
        // `(..(c_{n-1} z + c_{n-2}) z + ..)`, so every opening is the
        // identical field element and the transcript is unchanged.
        let degree = common_data.degree();
        let table = |z: F::Extension| -> Vec<F::Extension> { z.powers().take(degree).collect() };
        let zeta_pows = table(zeta);
        // `g` is the order-`degree` subgroup generator, so `g^i` is exactly
        // the process-cached natural-order two-adic subgroup, and
        // `(g·ζ)^i = g^i · ζ^i` in the exact field. Deriving the shifted
        // table from the `ζ` table by one elementwise base-field scalar
        // product deletes the second serial `powers()` chain (a
        // `degree`-long dependent extension-multiply chain in this serial
        // opening phase) per proof. Representative-exactness is checkable at
        // runtime: `LIGHTER_GZETA_TABLE_ASSERT=1` recomputes the old chain
        // and compares every entry by raw noncanonical limbs.
        let g_subgroup =
            crate::plonk::prover::precomputed::two_adic_subgroup::<F>(common_data.degree_bits());
        // The shifted table is only materialized when something actually
        // consumes a whole table of `(g·ζ)^i`. In the ranked lookup-free
        // configuration the only shifted openings are the `num_challenges`
        // `Z` polynomials, so folding `g^i` into each coefficient and reusing
        // the `ζ` table evaluates `P(gζ) = sum_i c_i · g^i · ζ^i` directly and
        // deletes a `degree`-long extension vector: its allocation, its
        // zero-fault, its fill pass and its read pass. Arithmetic is a wash
        // (one extra base multiply per term against one extension scalar-mul
        // per table entry), the deleted memory traffic is not.
        let use_fused_shifted = common_data.num_lookup_polys == 0
            && common_data.zs_range().len() <= D
            && std::env::var_os("LIGHTER_GZETA_TABLE_ASSERT").is_none();
        let g_zeta_pows: Vec<F::Extension> = if use_fused_shifted {
            Vec::new()
        } else {
            let g_zeta_pows: Vec<F::Extension> = zeta_pows
                .iter()
                .zip(g_subgroup.iter())
                .map(|(&zeta_pow, &g_pow)| zeta_pow.scalar_mul(g_pow))
                .collect();
            if std::env::var_os("LIGHTER_GZETA_TABLE_ASSERT").is_some() {
                let reference = table(g * zeta);
                assert_eq!(reference.len(), g_zeta_pows.len());
                for (i, (a, b)) in reference.iter().zip(&g_zeta_pows).enumerate() {
                    let a_raw: Vec<u64> = a
                        .to_basefield_array()
                        .iter()
                        .map(|c| c.to_noncanonical_u64())
                        .collect();
                    let b_raw: Vec<u64> = b
                        .to_basefield_array()
                        .iter()
                        .map(|c| c.to_noncanonical_u64())
                        .collect();
                    assert_eq!(a_raw, b_raw, "g-zeta table representative mismatch at {i}");
                }
            }
            g_zeta_pows
        };
        // One task per *pair* of polynomials: both dots walk the same
        // `pows` slice, so the shared powers table is read once instead of
        // twice while each output keeps its own 160-bit accumulator pair.
        // `extension_base_dot_products_2` defaults to the two single dots, and
        // the Goldilocks specialization is raw-representative-identical to them
        // (see its differential test).
        //
        // The pairs are drawn from *one* job list spanning every commitment
        // opened in this phase, so the whole opening set is a single Rayon
        // wave instead of one wave per commitment plus a wave for the shifted
        // openings. Each output owns a private 160-bit accumulator that is
        // reduced exactly once, and exact integer accumulation does not depend
        // on which other polynomial shares the task, so every opening is the
        // same raw word under any pairing arrangement -- including the pairs
        // that now straddle a commitment boundary. Peak stream count and total
        // bytes read are unchanged; only the barriers between the waves are
        // gone.
        //
        // Job order is the historical evaluation order, so the slices carved
        // out below are the very vectors the per-commitment waves produced.
        // `true` tags a shifted (`g·ζ`) opening.
        let constants_sigmas_polys = &constants_sigmas_commitment.polynomials;
        let zs_partial_products_lookup_polys = &zs_partial_products_lookup_commitment.polynomials;
        let quotient_polys_src = &quotient_polys_commitment.polynomials;
        let wires_polys = &wires_commitment.polynomials;
        let shifted_polynomials = &zs_partial_products_lookup_commitment.polynomials;
        let zs_range = common_data.zs_range();
        let lookup_range = common_data.lookup_range();
        let shifted_zs_polys = &shifted_polynomials[zs_range.clone()];
        let shifted_lookup_polys = &shifted_polynomials[lookup_range.clone()];

        let mut jobs: Vec<(&[F], bool)> = Vec::with_capacity(
            constants_sigmas_polys.len()
                + zs_partial_products_lookup_polys.len()
                + quotient_polys_src.len()
                + wires_polys.len()
                + shifted_zs_polys.len()
                + shifted_lookup_polys.len(),
        );
        for poly in constants_sigmas_polys {
            jobs.push((poly.coeffs.as_slice(), false));
        }
        for poly in zs_partial_products_lookup_polys {
            jobs.push((poly.coeffs.as_slice(), false));
        }
        for poly in quotient_polys_src {
            jobs.push((poly.coeffs.as_slice(), false));
        }
        for poly in wires_polys {
            jobs.push((poly.coeffs.as_slice(), false));
        }
        // Partial-product polynomials are opened only at `zeta`, never at
        // `g * zeta`; only the shifted Z (and lookup Z) polynomials are
        // consumed by the FRI next batch.
        for poly in shifted_zs_polys {
            jobs.push((poly.coeffs.as_slice(), true));
        }
        for poly in shifted_lookup_polys {
            jobs.push((poly.coeffs.as_slice(), true));
        }

        let mut evals = vec![F::Extension::ZERO; jobs.len()];
        {
            let jobs = &jobs;
            let zeta_pows = &zeta_pows;
            let g_zeta_pows = &g_zeta_pows;
            let g_subgroup = g_subgroup.as_slice();
            evals.par_chunks_mut(2).enumerate().for_each(|(pair, out)| {
                let base = pair * 2;
                let eval_one = |(coeffs, shifted): (&[F], bool)| {
                    if !shifted {
                        F::extension_base_dot_product(zeta_pows, coeffs)
                    } else if use_fused_shifted {
                        // Shifted opening without a materialized `(g·ζ)`
                        // table: fold the natural-order subgroup power
                        // into each coefficient.
                        F::extension_base_dot_product_with_subgroup_scales(
                            zeta_pows, coeffs, g_subgroup,
                        )
                    } else {
                        F::extension_base_dot_product(g_zeta_pows, coeffs)
                    }
                };
                // Only two unshifted openings share a powers slice, so
                // only they can ride the fused two-accumulator dot.
                if out.len() == 2 && !jobs[base].1 && !jobs[base + 1].1 {
                    let [first, second] = F::extension_base_dot_products_2(
                        zeta_pows,
                        [jobs[base].0, jobs[base + 1].0],
                    );
                    out[0] = first;
                    out[1] = second;
                } else {
                    for (slot, &job) in out.iter_mut().zip(&jobs[base..]) {
                        *slot = eval_one(job);
                    }
                }
            });
        }

        let constants_sigmas_len = constants_sigmas_polys.len();
        let zs_partial_products_lookup_len = zs_partial_products_lookup_polys.len();
        let quotient_len = quotient_polys_src.len();
        let wires_len = wires_polys.len();
        let constants_sigmas_at = 0;
        let zs_partial_products_lookup_at = constants_sigmas_at + constants_sigmas_len;
        let quotient_at = zs_partial_products_lookup_at + zs_partial_products_lookup_len;
        let wires_at = quotient_at + quotient_len;
        let zs_next_at = wires_at + wires_len;
        let lookup_next_at = zs_next_at + shifted_zs_polys.len();

        let constants_sigmas_eval =
            &evals[constants_sigmas_at..constants_sigmas_at + constants_sigmas_len];
        // `zs_partial_products_lookup_eval` contains the permutation argument polynomials as well as lookup polynomials.
        let zs_partial_products_lookup_eval = &evals[zs_partial_products_lookup_at
            ..zs_partial_products_lookup_at + zs_partial_products_lookup_len];

        Self {
            constants: constants_sigmas_eval[common_data.constants_range()].to_vec(),
            plonk_sigmas: constants_sigmas_eval[common_data.sigmas_range()].to_vec(),
            wires: evals[wires_at..wires_at + wires_len].to_vec(),
            plonk_zs: zs_partial_products_lookup_eval[common_data.zs_range()].to_vec(),
            plonk_zs_next: evals[zs_next_at..zs_next_at + shifted_zs_polys.len()].to_vec(),
            partial_products: zs_partial_products_lookup_eval[common_data.partial_products_range()]
                .to_vec(),
            quotient_polys: evals[quotient_at..quotient_at + quotient_len].to_vec(),
            lookup_zs: zs_partial_products_lookup_eval[common_data.lookup_range()].to_vec(),
            lookup_zs_next: evals[lookup_next_at..lookup_next_at + shifted_lookup_polys.len()]
                .to_vec(),
        }
    }
    pub(crate) fn to_fri_openings(&self) -> FriOpenings<F, D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                .concat(),
            }
        } else {
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatch {
                values: [self.plonk_zs_next.clone(), self.lookup_zs_next.clone()].concat(),
            }
        } else {
            FriOpeningBatch {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpenings {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

/// The purported values of each polynomial at a single point.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpeningSetTarget<const D: usize> {
    pub constants: Vec<ExtensionTarget<D>>,
    pub plonk_sigmas: Vec<ExtensionTarget<D>>,
    pub wires: Vec<ExtensionTarget<D>>,
    pub plonk_zs: Vec<ExtensionTarget<D>>,
    pub plonk_zs_next: Vec<ExtensionTarget<D>>,
    pub lookup_zs: Vec<ExtensionTarget<D>>,
    pub next_lookup_zs: Vec<ExtensionTarget<D>>,
    pub partial_products: Vec<ExtensionTarget<D>>,
    pub quotient_polys: Vec<ExtensionTarget<D>>,
}

impl<const D: usize> OpeningSetTarget<D> {
    pub(crate) fn to_fri_openings(&self) -> FriOpeningsTarget<D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                .concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [self.plonk_zs_next.clone(), self.next_lookup_zs.clone()].concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpeningsTarget {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::sync::Arc;
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use anyhow::Result;
    use itertools::Itertools;
    use plonky2_field::types::Sample;

    use super::*;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::verifier::verify;

    #[test]
    fn test_proof_compression() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1, 1]);
        config.fri_config.num_query_rounds = 50;

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Build dummy circuit to get a valid proof.
        let x = F::rand();
        let y = F::rand();
        let z = x * y;
        let xt = builder.constant(x);
        let yt = builder.constant(y);
        let zt = builder.constant(z);
        let comp_zt = builder.mul(xt, yt);
        builder.connect(zt, comp_zt);
        for _ in 0..100 {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        verify(proof.clone(), &data.verifier_only, &data.common)?;

        // Verify that `decompress ∘ compress = identity`.
        let compressed_proof = data.compress(proof.clone())?;
        let decompressed_compressed_proof = data.decompress(compressed_proof.clone())?;
        assert_eq!(proof, decompressed_compressed_proof);

        verify(proof, &data.verifier_only, &data.common)?;
        data.verify_compressed(compressed_proof)
    }

    #[test]
    fn test_proof_compression_lookup() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        use plonky2_field::types::Field;
        type F = <C as GenericConfig<D>>::F;

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1, 1]);
        config.fri_config.num_query_rounds = 50;

        let pw = PartialWitness::new();
        let tip5_table = vec![
            0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176,
            32, 8, 110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130,
            208, 6, 44, 71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48,
            35, 119, 49, 88, 242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100,
            129, 112, 55, 221, 102, 218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22,
            141, 65, 57, 123, 12, 244, 54, 219, 231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195,
            205, 134, 245, 30, 9, 188, 59, 142, 186, 197, 181, 144, 92, 31, 224, 163, 111, 74, 58,
            69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90, 122, 2, 250, 101, 75, 178, 159,
            24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209, 108, 238, 91, 187, 18,
            104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89, 94, 193, 135,
            183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217, 158,
            56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20,
            241, 25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70,
            210, 27, 41, 1, 170, 40, 131, 192, 229, 248, 255,
        ];
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let lut_index = builder.add_lookup_table_from_pairs(table);

        // Build dummy circuit with a lookup to get a valid proof.
        let x = F::TWO;
        let out = builder.constant(F::from_canonical_usize(26));

        let xt = builder.constant(x);
        let look_out = builder.add_lookup_from_index(xt, lut_index);
        builder.connect(look_out, out);
        for _ in 0..100 {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<C>();

        let proof = data.prove(pw)?;
        verify(proof.clone(), &data.verifier_only, &data.common)?;

        // Verify that `decompress ∘ compress = identity`.
        let compressed_proof = data.compress(proof.clone())?;
        let decompressed_compressed_proof = data.decompress(compressed_proof.clone())?;
        assert_eq!(proof, decompressed_compressed_proof);

        verify(proof, &data.verifier_only, &data.common)?;
        data.verify_compressed(compressed_proof)
    }
}
