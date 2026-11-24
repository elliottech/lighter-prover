//! A uni-stark prover/verifier extended with one extra committed trace: the
//! LogUp "permutation" trace, generated after the main trace is committed and
//! the bus challenges are sampled. Everything else (quotient, FRI opening)
//! mirrors `p3_uni_stark::prove_with_preprocessed` / `verify_with_preprocessed`.

use alloc::vec;
use alloc::vec::Vec;

use p3_air::symbolic::{AirLayout, SymbolicAirBuilder};
use p3_air::{
    Air, AirBuilder, BaseAir, DebugConstraintBuilder, ExtensionBuilder, PermutationAirBuilder,
    RowWindow,
};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    Domain, InvalidProofShapeError, PackedChallenge, PackedVal, PcsError, PreprocessedProverData,
    PreprocessedVerifierKey, StarkGenericConfig, Val, VerificationError,
    recompose_quotient_from_chunks,
};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use serde::{Deserialize, Serialize};
use tracing::info_span;

extern crate alloc;

/// An AIR that, besides its main constraints, fills a permutation trace from
/// the main/preprocessed traces and the sampled bus challenges.
pub trait LogUpAir<F: Field, EF: ExtensionField<F>>: BaseAir<F> {
    /// Number of extension-field columns of the permutation trace.
    fn permutation_width(&self) -> usize;
    /// Number of extension-field challenges sampled before the permutation trace.
    fn num_permutation_challenges(&self) -> usize;
    fn generate_permutation_trace(
        &self,
        main: &RowMajorMatrix<F>,
        preprocessed: Option<&RowMajorMatrix<F>>,
        challenges: &[EF],
    ) -> RowMajorMatrix<EF>;
}

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogUpProof<SC: StarkGenericConfig> {
    pub trace_commit: Com<SC>,
    pub permutation_commit: Com<SC>,
    pub quotient_commit: Com<SC>,
    pub opened_values: LogUpOpenedValues<SC::Challenge>,
    pub opening_proof: PcsProof<SC>,
    pub degree_bits: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogUpOpenedValues<Challenge> {
    pub trace_local: Vec<Challenge>,
    pub trace_next: Vec<Challenge>,
    /// Flattened base-field columns of the permutation trace.
    pub permutation_local: Vec<Challenge>,
    pub permutation_next: Vec<Challenge>,
    pub preprocessed_local: Vec<Challenge>,
    pub preprocessed_next: Vec<Challenge>,
    pub quotient_chunks: Vec<Vec<Challenge>>,
}

pub trait LogUpStarkAir<SC: StarkGenericConfig>:
    LogUpAir<Val<SC>, SC::Challenge>
    + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
    + for<'a> Air<LogUpProverFolder<'a, SC>>
    + for<'a> Air<LogUpVerifierFolder<'a, SC>>
    + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
{
}
impl<SC: StarkGenericConfig, A> LogUpStarkAir<SC> for A where
    A: LogUpAir<Val<SC>, SC::Challenge>
        + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<LogUpProverFolder<'a, SC>>
        + for<'a> Air<LogUpVerifierFolder<'a, SC>>
        + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
{
}

fn air_layout<SC: StarkGenericConfig, A: LogUpStarkAir<SC>>(
    air: &A,
    preprocessed_width: usize,
) -> AirLayout {
    AirLayout {
        preprocessed_width,
        main_width: air.width(),
        num_public_values: air.num_public_values(),
        permutation_width: air.permutation_width(),
        num_permutation_challenges: air.num_permutation_challenges(),
        num_permutation_values: 0,
        num_periodic_columns: 0,
    }
}

fn log_num_quotient_chunks<SC: StarkGenericConfig, A: LogUpStarkAir<SC>>(air: &A) -> usize {
    let degree = air
        .max_constraint_degree()
        .expect("LogUp AIRs must declare max_constraint_degree");
    log2_ceil_usize(degree.max(2) - 1)
}

/// Checks every constraint of `air` on the given traces natively; panics with
/// the failing rows/constraints. Used by tests and in debug builds.
pub fn check_constraints<SC: StarkGenericConfig, A: LogUpStarkAir<SC>>(
    air: &A,
    main: &RowMajorMatrix<Val<SC>>,
    preprocessed: Option<&RowMajorMatrix<Val<SC>>>,
    permutation: &RowMajorMatrix<SC::Challenge>,
    challenges: &[SC::Challenge],
    public_values: &[Val<SC>],
) {
    let height = main.height();
    let empty: Vec<Val<SC>> = vec![];
    let mut failures = Vec::new();
    for row in 0..height {
        let next = (row + 1) % height;
        let (ml, mn) = (main.row_slice(row).unwrap(), main.row_slice(next).unwrap());
        let main_pair = VerticalPair::new(
            RowMajorMatrixView::new_row(&*ml),
            RowMajorMatrixView::new_row(&*mn),
        );
        let prep_rows =
            preprocessed.map(|prep| (prep.row_slice(row).unwrap(), prep.row_slice(next).unwrap()));
        let prep_pair = match &prep_rows {
            Some((l, n)) => VerticalPair::new(
                RowMajorMatrixView::new_row(&**l),
                RowMajorMatrixView::new_row(&**n),
            ),
            None => VerticalPair::new(
                RowMajorMatrixView::new_row(&empty),
                RowMajorMatrixView::new_row(&empty),
            ),
        };
        let (pl, pn) = (
            permutation.row_slice(row).unwrap(),
            permutation.row_slice(next).unwrap(),
        );
        let perm_pair = VerticalPair::new(
            RowMajorMatrixView::new_row(&*pl),
            RowMajorMatrixView::new_row(&*pn),
        );
        let mut builder = DebugConstraintBuilder::new_with_permutation(
            row,
            main_pair,
            prep_pair,
            public_values,
            Val::<SC>::from_bool(row == 0),
            Val::<SC>::from_bool(row == height - 1),
            Val::<SC>::from_bool(row != height - 1),
            perm_pair,
            challenges,
            &[],
            &[],
        );
        air.eval(&mut builder);
        if builder.has_failures() {
            let mut msg = builder.formatted_failures();
            msg.truncate(200);
            failures.push((row, msg));
            if failures.len() > 8 {
                break;
            }
        }
    }
    assert!(failures.is_empty(), "constraint failures: {failures:?}");
}

#[allow(clippy::too_many_arguments)]
pub fn prove<SC: StarkGenericConfig, A: LogUpStarkAir<SC>>(
    config: &SC,
    air: &A,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &[Val<SC>],
    preprocessed: Option<&PreprocessedProverData<SC>>,
) -> LogUpProof<SC> {
    assert_eq!(config.is_zk(), 0, "zk configs are not supported");
    let degree = trace.height();
    let log_degree = log2_strict_usize(degree);
    let preprocessed_width = preprocessed.map_or(0, |pp| {
        assert_eq!(pp.degree_bits, log_degree);
        pp.width
    });
    assert_eq!(preprocessed_width, air.preprocessed_width());
    let layout = air_layout::<SC, A>(air, preprocessed_width);
    let log_num_quotient_chunks = log_num_quotient_chunks::<SC, A>(air);
    let num_quotient_chunks = 1 << log_num_quotient_chunks;
    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();
    let trace_domain = pcs.natural_domain_for_degree(degree);

    let preprocessed_trace = air.preprocessed_trace();
    let (trace_commit, trace_data) =
        info_span!("commit to trace data").in_scope(|| pcs.commit([(trace_domain, trace.clone())]));
    challenger.observe(Val::<SC>::from_u8(log_degree as u8));
    challenger.observe(Val::<SC>::from_usize(preprocessed_width));
    challenger.observe(trace_commit.clone());
    if let Some(pp) = preprocessed {
        challenger.observe(pp.commitment.clone());
    }
    challenger.observe_slice(public_values);

    let challenges: Vec<SC::Challenge> = (0..air.num_permutation_challenges())
        .map(|_| challenger.sample_algebra_element())
        .collect();
    let permutation = info_span!("generate permutation trace").in_scope(|| {
        air.generate_permutation_trace(&trace, preprocessed_trace.as_ref(), &challenges)
    });
    assert_eq!(permutation.width(), air.permutation_width());
    assert_eq!(permutation.height(), degree);
    #[cfg(debug_assertions)]
    check_constraints::<SC, A>(
        air,
        &trace,
        preprocessed_trace.as_ref(),
        &permutation,
        &challenges,
        public_values,
    );
    let permutation_flat = permutation.flatten_to_base::<Val<SC>>();
    let (permutation_commit, permutation_data) = info_span!("commit to permutation trace")
        .in_scope(|| pcs.commit([(trace_domain, permutation_flat)]));
    challenger.observe(permutation_commit.clone());
    let alpha: SC::Challenge = challenger.sample_algebra_element();

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (log_degree + log_num_quotient_chunks));
    let _lde_span = info_span!("LDE to quotient domain").entered();
    let trace_on_qd = pcs.get_evaluations_on_domain(&trace_data, 0, quotient_domain);
    let perm_on_qd = pcs.get_evaluations_on_domain(&permutation_data, 0, quotient_domain);
    let prep_on_qd = preprocessed
        .map(|pp| pcs.get_evaluations_on_domain_no_random(&pp.prover_data, 0, quotient_domain));
    drop(_lde_span);
    let _qv_span = info_span!("evaluate constraints on quotient domain").entered();
    let quotient_values = quotient_values::<SC, A, _>(
        air,
        public_values,
        layout,
        trace_domain,
        quotient_domain,
        &trace_on_qd,
        &perm_on_qd,
        prep_on_qd.as_ref(),
        &challenges,
        alpha,
    );
    drop(_qv_span);
    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
    let (quotient_commit, quotient_data) = info_span!("commit to quotient poly chunks")
        .in_scope(|| pcs.commit_quotient(quotient_domain, quotient_flat, num_quotient_chunks));
    challenger.observe(quotient_commit.clone());

    let zeta: SC::Challenge = challenger.sample_algebra_element();
    let zeta_next = trace_domain.next_point(zeta).expect("next point");
    let (opened_values, opening_proof) = info_span!("open").in_scope(|| {
        let mut rounds = vec![
            (&trace_data, vec![vec![zeta, zeta_next]]),
            (&permutation_data, vec![vec![zeta, zeta_next]]),
            (&quotient_data, vec![vec![zeta]; num_quotient_chunks]),
        ];
        if let Some(pp) = preprocessed {
            rounds.push((&pp.prover_data, vec![vec![zeta, zeta_next]]));
        }
        pcs.open(rounds, &mut challenger)
    });
    let (preprocessed_local, preprocessed_next) = if preprocessed.is_some() {
        (
            opened_values[3][0][0].clone(),
            opened_values[3][0][1].clone(),
        )
    } else {
        (vec![], vec![])
    };
    LogUpProof {
        trace_commit,
        permutation_commit,
        quotient_commit,
        opened_values: LogUpOpenedValues {
            trace_local: opened_values[0][0][0].clone(),
            trace_next: opened_values[0][0][1].clone(),
            permutation_local: opened_values[1][0][0].clone(),
            permutation_next: opened_values[1][0][1].clone(),
            preprocessed_local,
            preprocessed_next,
            quotient_chunks: opened_values[2].iter().map(|v| v[0].clone()).collect(),
        },
        opening_proof,
        degree_bits: log_degree,
    }
}

#[allow(clippy::too_many_arguments)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_qd: &Mat,
    perm_on_qd: &Mat,
    prep_on_qd: Option<&Mat>,
    challenges: &[SC::Challenge],
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: LogUpStarkAir<SC>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = trace_on_qd.width();
    let perm_flat_width = perm_on_qd.width();
    let ext_degree = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
    let perm_width = perm_flat_width / ext_degree;
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);
    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }
    let constraint_layout = {
        let mut builder = SymbolicAirBuilder::<Val<SC>, SC::Challenge>::new(layout);
        air.eval(&mut builder);
        builder.constraint_layout()
    };
    let (base_alpha_powers, ext_alpha_powers) = constraint_layout.decompose_alpha(alpha);
    let packed_challenges: Vec<PackedChallenge<SC>> = challenges
        .iter()
        .map(|c| PackedChallenge::<SC>::from(*c))
        .collect();
    let pack_width = PackedVal::<SC>::WIDTH;

    let mut quotient_values = SC::Challenge::zero_vec(quotient_size);
    quotient_values
        .par_chunks_mut(pack_width)
        .enumerate()
        .for_each(|(group, quotient_chunk)| {
            let i_start = group * pack_width;
            let i_range = i_start..i_start + pack_width;
            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

            let mut main_buf: Vec<PackedVal<SC>> = Vec::with_capacity(2 * width);
            main_buf.extend(trace_on_qd.vertically_packed_row::<PackedVal<SC>>(i_start));
            main_buf
                .extend(trace_on_qd.vertically_packed_row::<PackedVal<SC>>(i_start + next_step));
            let main = RowMajorMatrixView::new(&main_buf, width);

            let perm_rows: Vec<Vec<PackedVal<SC>>> = [i_start, i_start + next_step]
                .into_iter()
                .map(|i| {
                    perm_on_qd
                        .vertically_packed_row::<PackedVal<SC>>(i)
                        .collect()
                })
                .collect();
            let perm_buf: Vec<PackedChallenge<SC>> = perm_rows
                .iter()
                .flat_map(|row| {
                    (0..perm_width).map(move |c| {
                        PackedChallenge::<SC>::from_basis_coefficients_fn(|d| {
                            row[c * ext_degree + d]
                        })
                    })
                })
                .collect();
            let permutation = RowMajorMatrixView::new(&perm_buf, perm_width);

            let prep_buf: Vec<PackedVal<SC>> = match prep_on_qd {
                Some(prep) => {
                    let mut buf = Vec::with_capacity(2 * prep.width());
                    buf.extend(prep.vertically_packed_row::<PackedVal<SC>>(i_start));
                    buf.extend(prep.vertically_packed_row::<PackedVal<SC>>(i_start + next_step));
                    buf
                }
                None => vec![],
            };
            let preprocessed =
                RowMajorMatrixView::new(&prep_buf, prep_on_qd.map_or(0, |p| p.width()));

            let mut folder = LogUpProverFolder {
                main,
                preprocessed,
                preprocessed_window: RowWindow::from_view(&preprocessed),
                permutation,
                permutation_challenges: &packed_challenges,
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                base_alpha_powers: &base_alpha_powers,
                ext_alpha_powers: &ext_alpha_powers,
                base_constraints: Vec::with_capacity(constraint_layout.base_indices.len()),
                ext_constraints: Vec::with_capacity(constraint_layout.ext_indices.len()),
                constraint_index: 0,
                constraint_count: constraint_layout.total_constraints(),
            };
            air.eval(&mut folder);
            let quotient = folder.finalize_constraints() * inv_vanishing;
            for (idx, q) in quotient_chunk.iter_mut().enumerate() {
                *q = quotient.extract(idx);
            }
        });
    quotient_values
}

pub fn verify<SC: StarkGenericConfig, A: LogUpStarkAir<SC>>(
    config: &SC,
    air: &A,
    proof: &LogUpProof<SC>,
    public_values: &[Val<SC>],
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<(), VerificationError<PcsError<SC>>> {
    let LogUpProof {
        trace_commit,
        permutation_commit,
        quotient_commit,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;
    let degree_bits = *degree_bits;
    let pcs = config.pcs();
    let degree = 1usize << degree_bits;
    let trace_domain = pcs.natural_domain_for_degree(degree);
    let preprocessed_width = air.preprocessed_width();
    if preprocessed_width > 0 {
        let vk =
            preprocessed_vk.ok_or(InvalidProofShapeError::PreprocessedVerifierKeyInconsistency)?;
        if vk.width != preprocessed_width || vk.degree_bits != degree_bits {
            return Err(InvalidProofShapeError::PreprocessedVerifierKeyInconsistency.into());
        }
    }
    let layout = air_layout::<SC, A>(air, preprocessed_width);
    let log_num_quotient_chunks = log_num_quotient_chunks::<SC, A>(air);
    let num_quotient_chunks = 1usize << log_num_quotient_chunks;
    let ext_degree = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
    let perm_flat_width = air.permutation_width() * ext_degree;

    if public_values.len() != air.num_public_values() {
        return Err(InvalidProofShapeError::PublicValuesLengthMismatch {
            expected: air.num_public_values(),
            got: public_values.len(),
        }
        .into());
    }
    let valid_shape = opened_values.trace_local.len() == air.width()
        && opened_values.trace_next.len() == air.width()
        && opened_values.permutation_local.len() == perm_flat_width
        && opened_values.permutation_next.len() == perm_flat_width
        && opened_values.preprocessed_local.len() == preprocessed_width
        && opened_values.preprocessed_next.len() == preprocessed_width
        && opened_values.quotient_chunks.len() == num_quotient_chunks
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == ext_degree);
    if !valid_shape {
        return Err(InvalidProofShapeError::OpenedValuesDimensionMismatch.into());
    }

    let mut challenger = config.initialise_challenger();
    challenger.observe(Val::<SC>::from_u8(degree_bits as u8));
    challenger.observe(Val::<SC>::from_usize(preprocessed_width));
    challenger.observe(trace_commit.clone());
    if let Some(vk) = preprocessed_vk.filter(|_| preprocessed_width > 0) {
        challenger.observe(vk.commitment.clone());
    }
    challenger.observe_slice(public_values);
    let challenges: Vec<SC::Challenge> = (0..air.num_permutation_challenges())
        .map(|_| challenger.sample_algebra_element())
        .collect();
    challenger.observe(permutation_commit.clone());
    let alpha: SC::Challenge = challenger.sample_algebra_element();
    challenger.observe(quotient_commit.clone());
    let zeta: SC::Challenge = challenger.sample_algebra_element();
    if trace_domain.vanishing_poly_at_point(zeta).is_zero() {
        return Err(VerificationError::OodPointInDomain);
    }
    let zeta_next = trace_domain
        .next_point(zeta)
        .ok_or(VerificationError::NextPointUnavailable)?;

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_num_quotient_chunks));
    let quotient_chunks_domains = quotient_domain.split_domains(num_quotient_chunks);

    let mut coms_to_verify = vec![
        (
            trace_commit.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.trace_local.clone()),
                    (zeta_next, opened_values.trace_next.clone()),
                ],
            )],
        ),
        (
            permutation_commit.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.permutation_local.clone()),
                    (zeta_next, opened_values.permutation_next.clone()),
                ],
            )],
        ),
        (
            quotient_commit.clone(),
            quotient_chunks_domains
                .iter()
                .zip(&opened_values.quotient_chunks)
                .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
                .collect(),
        ),
    ];
    if let Some(vk) = preprocessed_vk.filter(|_| preprocessed_width > 0) {
        coms_to_verify.push((
            vk.commitment.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.preprocessed_local.clone()),
                    (zeta_next, opened_values.preprocessed_next.clone()),
                ],
            )],
        ));
    }
    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    let quotient = recompose_quotient_from_chunks::<SC>(
        &quotient_chunks_domains,
        &opened_values.quotient_chunks,
        zeta,
    );
    let unflatten = |flat: &[SC::Challenge]| -> Vec<SC::Challenge> {
        flat.chunks(ext_degree)
            .map(|coeffs| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(d, c)| {
                        *c * <SC::Challenge as BasedVectorSpace<Val<SC>>>::ith_basis_element(d)
                            .unwrap()
                    })
                    .sum()
            })
            .collect()
    };
    let perm_local = unflatten(&opened_values.permutation_local);
    let perm_next = unflatten(&opened_values.permutation_next);
    let sels = trace_domain.selectors_at_point(zeta);
    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(&opened_values.trace_local),
        RowMajorMatrixView::new_row(&opened_values.trace_next),
    );
    let preprocessed_window = RowWindow::from_two_rows(
        &opened_values.preprocessed_local,
        &opened_values.preprocessed_next,
    );
    let permutation = VerticalPair::new(
        RowMajorMatrixView::new_row(&perm_local),
        RowMajorMatrixView::new_row(&perm_next),
    );
    let _ = layout;
    let mut folder = LogUpVerifierFolder {
        main,
        preprocessed_window,
        permutation,
        permutation_challenges: &challenges,
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SC::Challenge::ZERO,
    };
    air.eval(&mut folder);
    if folder.accumulator * sels.inv_vanishing != quotient {
        return Err(VerificationError::OodEvaluationMismatch { index: None });
    }
    Ok(())
}

pub struct LogUpProverFolder<'a, SC: StarkGenericConfig> {
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    pub preprocessed: RowMajorMatrixView<'a, PackedVal<SC>>,
    pub preprocessed_window: RowWindow<'a, PackedVal<SC>>,
    pub permutation: RowMajorMatrixView<'a, PackedChallenge<SC>>,
    pub permutation_challenges: &'a [PackedChallenge<SC>],
    pub public_values: &'a [Val<SC>],
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub base_alpha_powers: &'a [Vec<Val<SC>>],
    pub ext_alpha_powers: &'a [SC::Challenge],
    pub base_constraints: Vec<PackedVal<SC>>,
    pub ext_constraints: Vec<PackedChallenge<SC>>,
    pub constraint_index: usize,
    pub constraint_count: usize,
}

impl<SC: StarkGenericConfig> LogUpProverFolder<'_, SC> {
    fn finalize_constraints(&self) -> PackedChallenge<SC> {
        debug_assert_eq!(self.constraint_index, self.constraint_count);
        let acc = PackedChallenge::<SC>::from_basis_coefficients_fn(|d| {
            PackedVal::<SC>::batched_linear_combination(
                &self.base_constraints,
                &self.base_alpha_powers[d],
            )
        });
        acc + PackedChallenge::<SC>::batched_linear_combination(
            &self.ext_constraints,
            self.ext_alpha_powers,
        )
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for LogUpProverFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type PreprocessedWindow = RowWindow<'a, PackedVal<SC>>;
    type MainWindow = RowWindow<'a, PackedVal<SC>>;
    type PublicVar = Val<SC>;
    type PeriodicVar = PackedVal<SC>;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_view(&self.main)
    }
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }
    fn is_transition(&self) -> Self::Expr {
        self.is_transition
    }
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_index += 1;
    }
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &[]
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for LogUpProverFolder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        self.ext_constraints.push(x.into());
        self.constraint_index += 1;
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder for LogUpProverFolder<'a, SC> {
    type MP = RowWindow<'a, PackedChallenge<SC>>;
    type RandomVar = PackedChallenge<SC>;
    type PermutationVar = PackedChallenge<SC>;

    fn permutation(&self) -> Self::MP {
        RowWindow::from_view(&self.permutation)
    }
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &[]
    }
}

pub struct LogUpVerifierFolder<'a, SC: StarkGenericConfig> {
    pub main:
        VerticalPair<RowMajorMatrixView<'a, SC::Challenge>, RowMajorMatrixView<'a, SC::Challenge>>,
    pub preprocessed_window: RowWindow<'a, SC::Challenge>,
    pub permutation:
        VerticalPair<RowMajorMatrixView<'a, SC::Challenge>, RowMajorMatrixView<'a, SC::Challenge>>,
    pub permutation_challenges: &'a [SC::Challenge],
    pub public_values: &'a [Val<SC>],
    pub is_first_row: SC::Challenge,
    pub is_last_row: SC::Challenge,
    pub is_transition: SC::Challenge,
    pub alpha: SC::Challenge,
    pub accumulator: SC::Challenge,
}

impl<'a, SC: StarkGenericConfig> AirBuilder for LogUpVerifierFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type PreprocessedWindow = RowWindow<'a, SC::Challenge>;
    type MainWindow = RowWindow<'a, SC::Challenge>;
    type PublicVar = Val<SC>;
    type PeriodicVar = SC::Challenge;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.main.top.values, self.main.bottom.values)
    }
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }
    fn is_transition(&self) -> Self::Expr {
        self.is_transition
    }
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &[]
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for LogUpVerifierFolder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder for LogUpVerifierFolder<'a, SC> {
    type MP = RowWindow<'a, SC::Challenge>;
    type RandomVar = SC::Challenge;
    type PermutationVar = SC::Challenge;

    fn permutation(&self) -> Self::MP {
        RowWindow::from_two_rows(self.permutation.top.values, self.permutation.bottom.values)
    }
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &[]
    }
}

/// Commits the AIR's preprocessed trace for traces of height `1 << degree_bits`.
pub fn setup_preprocessed<SC: StarkGenericConfig, A: LogUpStarkAir<SC>>(
    config: &SC,
    air: &A,
    degree_bits: usize,
) -> Option<(PreprocessedProverData<SC>, PreprocessedVerifierKey<SC>)> {
    let preprocessed = air.preprocessed_trace()?;
    let width = preprocessed.width();
    if width == 0 {
        return None;
    }
    assert_eq!(preprocessed.height(), 1 << degree_bits);
    let pcs = config.pcs();
    let trace_domain = pcs.natural_domain_for_degree(1 << degree_bits);
    let (commitment, prover_data) = pcs.commit_preprocessing([(trace_domain, preprocessed)]);
    Some((
        PreprocessedProverData {
            width,
            degree_bits,
            commitment: commitment.clone(),
            prover_data,
        },
        PreprocessedVerifierKey {
            width,
            degree_bits,
            commitment,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::EcdsaBatchAir;
    use crate::air::layout::{MAX_CONSTRAINT_DEGREE, PREP_WIDTH};
    use crate::prove::EcdsaStarkConfig;

    /// The declared degree sets the number of quotient chunks; it must match
    /// the AIR's actual highest-degree constraint.
    #[test]
    fn ecdsa_air_degree_matches_declared() {
        type SC = EcdsaStarkConfig;
        let air = EcdsaBatchAir::new(1);
        let layout = air_layout::<SC, _>(&air, PREP_WIDTH);
        let mut b =
            SymbolicAirBuilder::<Val<SC>, <SC as StarkGenericConfig>::Challenge>::new(layout);
        air.eval(&mut b);
        let base = b
            .base_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max();
        let ext = b
            .extension_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max();
        assert_eq!(base.max(ext), Some(MAX_CONSTRAINT_DEGREE));
    }
}
