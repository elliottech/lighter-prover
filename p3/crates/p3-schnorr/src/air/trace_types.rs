/// A fully materialized main trace for [`SignatureBatchAir`], plus the
/// metadata needed to hand it to the STARK prover: the `Air` instance sized
/// for `rows`, the public values it must match, and the trace's own
/// dimensions (`rows` x `columns`, matching [`TRACE_WIDTH`]).
#[derive(Clone, Debug)]
pub struct SignatureBatchTrace {
    pub air: SignatureBatchAir,
    pub trace: RowMajorMatrix<Goldilocks>,
    pub public_values: Vec<Goldilocks>,
    pub rows: usize,
    pub columns: usize,
}
