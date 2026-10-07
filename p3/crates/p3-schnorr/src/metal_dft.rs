//! Metal-accelerated coset LDE for the signature-batch STARK (Apple Silicon
//! only). Prover-side optimization: the LDE values produced are bit-for-bit
//! the field elements the CPU DFT would produce, so commitments, proofs, and
//! verification are unchanged. Falls back to [`Radix2DitParallel`] on any
//! Metal failure, on small inputs, and on non-macOS targets.

use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks;
use p3_matrix::bitrev::BitReversedMatrixView;
use p3_matrix::dense::RowMajorMatrix;

/// Drop-in replacement for `Radix2DitParallel<Goldilocks>` in the STARK
/// config: same `Evaluations` type, same results; large coset LDEs are
/// offloaded to the GPU on Apple Silicon.
#[derive(Debug, Default, Clone)]
pub struct MetalDft {
    inner: Radix2DitParallel<Goldilocks>,
}

impl TwoAdicSubgroupDft<Goldilocks> for MetalDft {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<Goldilocks>>;

    fn dft_batch(&self, mat: RowMajorMatrix<Goldilocks>) -> Self::Evaluations {
        self.inner.dft_batch(mat)
    }

    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) -> Self::Evaluations {
        #[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
        {
            match metal_impl::coset_lde_batch_gpu(&mat, added_bits, shift) {
                Some(out) => return out,
                None => {}
            }
        }
        self.inner.coset_lde_batch(mat, added_bits, shift)
    }
}

#[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
mod metal_impl {
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::sync::{LazyLock, Mutex};

    use metal::{
        Buffer, CommandBufferRef, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
        Device, MTLResourceOptions, MTLSize, NSUInteger,
    };
    use p3_field::{PrimeCharacteristicRing, PrimeField64, TwoAdicField};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;
    use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView};
    use p3_matrix::dense::RowMajorMatrix;
    use rayon::prelude::*;

    const SHADER_SOURCE: &str = include_str!("ntt.metal");
    /// Columns processed per GPU pass; bounds the working buffers
    /// (64 cols at lde 2^20 = 512 MiB for the LDE buffer).
    const CHUNK_COLS: usize = 64;
    /// Below this total butterfly count the dispatch overhead beats the win.
    const MIN_GPU_ELEMS: usize = 1 << 24;

    struct Ctx {
        device: Device,
        queue: metal::CommandQueue,
        prepare: ComputePipelineState,
        stage: ComputePipelineState,
        finalize: ComputePipelineState,
        transpose_in: ComputePipelineState,
        transpose_out: ComputePipelineState,
        /// Per-log-size twiddle tables: concatenated stage tables plus their
        /// per-stage offsets (stage s table = 2^s twiddles).
        roots: Mutex<HashMap<u32, (Buffer, Vec<usize>)>>,
        /// Per-log-degree coset-shift power tables (shift^k, k < degree).
        shifts: Mutex<HashMap<(u32, u64), Buffer>>,
        /// Per-log-degree all-ones tables (identity "shift" for plain FFTs).
        ones: Mutex<HashMap<u32, Buffer>>,
    }

    unsafe impl Send for Ctx {}
    unsafe impl Sync for Ctx {}

    static CTX: LazyLock<Option<Ctx>> = LazyLock::new(|| {
        let device = Device::system_default()?;
        let library = match device.new_library_with_source(SHADER_SOURCE, &CompileOptions::new()) {
            Ok(lib) => lib,
            Err(err) => {
                eprintln!("p3 metal ntt: shader compile failed: {err}");
                return None;
            }
        };
        let pipeline = |name: &str| -> Option<ComputePipelineState> {
            let function = library.get_function(name, None).ok()?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .ok()
        };
        Some(Ctx {
            prepare: pipeline("p3_ntt_prepare")?,
            stage: pipeline("p3_ntt_stage")?,
            finalize: pipeline("p3_ifft_finalize")?,
            transpose_in: pipeline("p3_transpose_in")?,
            transpose_out: pipeline("p3_transpose_out_bitrev")?,
            queue: device.new_command_queue(),
            roots: Mutex::new(HashMap::new()),
            shifts: Mutex::new(HashMap::new()),
            ones: Mutex::new(HashMap::new()),
            device,
        })
    });

    fn roots_for(ctx: &Ctx, log_n: u32) -> (Buffer, Vec<usize>) {
        let mut cache = ctx.roots.lock().unwrap();
        if let Some(entry) = cache.get(&log_n) {
            return entry.clone();
        }
        let lg_n = log_n as usize;
        let g = Goldilocks::two_adic_generator(lg_n);
        let mut bases = Vec::with_capacity(lg_n);
        let mut base = g;
        bases.push(base);
        for _ in 1..lg_n {
            base = base * base;
            bases.push(base);
        }
        let mut values: Vec<u64> = Vec::with_capacity(1 << lg_n);
        let mut offsets = Vec::with_capacity(lg_n);
        for s in 0..lg_n {
            offsets.push(values.len());
            // Stage s twiddles: powers of g^(2^(lg_n - 1 - s)).
            let row_base = bases[lg_n - 1 - s];
            let mut power = Goldilocks::ONE;
            for _ in 0..(1usize << s) {
                values.push(power.as_canonical_u64());
                power = power * row_base;
            }
        }
        let buffer = ctx.device.new_buffer_with_data(
            values.as_ptr().cast(),
            (values.len() * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        cache.insert(log_n, (buffer.clone(), offsets.clone()));
        (buffer, offsets)
    }

    fn pow_table_for(ctx: &Ctx, degree: usize, shift: Goldilocks) -> Buffer {
        let log_degree = degree.ilog2();
        let key = (log_degree, shift.as_canonical_u64());
        let mut cache = ctx.shifts.lock().unwrap();
        if let Some(buffer) = cache.get(&key) {
            return buffer.clone();
        }
        let mut values: Vec<u64> = Vec::with_capacity(degree);
        let mut power = Goldilocks::ONE;
        for _ in 0..degree {
            values.push(power.as_canonical_u64());
            power = power * shift;
        }
        let buffer = ctx.device.new_buffer_with_data(
            values.as_ptr().cast(),
            (values.len() * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        cache.insert(key, buffer.clone());
        buffer
    }

    fn ones_for(ctx: &Ctx, degree: usize) -> Buffer {
        let log_degree = degree.ilog2();
        let mut cache = ctx.ones.lock().unwrap();
        if let Some(buffer) = cache.get(&log_degree) {
            return buffer.clone();
        }
        let values: Vec<u64> = vec![1u64; degree];
        let buffer = ctx.device.new_buffer_with_data(
            values.as_ptr().cast(),
            (values.len() * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        cache.insert(log_degree, buffer.clone());
        buffer
    }

    fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(index, 4, (&value as *const u32).cast());
    }

    fn dispatch2d(
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        x: usize,
        y: usize,
    ) {
        let threads = pipeline.max_total_threads_per_threadgroup().min(256);
        let groups = (x as NSUInteger).div_ceil(threads);
        encoder.dispatch_thread_groups(
            MTLSize::new(groups, y as NSUInteger, 1),
            MTLSize::new(threads, 1, 1),
        );
    }

    /// Encodes iDFT + zero-padded coset DFT of `cols` columns held
    /// column-major in `col_deg` (degree-sized), writing the natural-order
    /// column-major LDE into `col_lde` (lde-sized) via `coeffs` scratch.
    #[allow(clippy::too_many_arguments)]
    fn encode_lde(
        ctx: &Ctx,
        cmd: &CommandBufferRef,
        col_deg: &Buffer,
        coeffs: &Buffer,
        col_lde: &Buffer,
        ones: &Buffer,
        shift_pows: &Buffer,
        roots_deg: &(Buffer, Vec<usize>),
        roots_lde: &(Buffer, Vec<usize>),
        degree: usize,
        lde_size: usize,
        rate_bits: usize,
        cols: usize,
        n_inv: u64,
    ) {
        let degree_u32 = degree as u32;
        let lde_u32 = lde_size as u32;
        let log_degree = degree.ilog2();
        let log_lde = lde_size.ilog2();

        // Forward FFT of the values (bit-reversed gather then DIT stages).
        let gather = cmd.new_compute_command_encoder();
        gather.set_compute_pipeline_state(&ctx.prepare);
        gather.set_buffer(0, Some(col_deg), 0);
        gather.set_buffer(1, Some(ones), 0);
        gather.set_buffer(2, Some(coeffs), 0);
        set_u32(gather, 3, degree_u32);
        set_u32(gather, 4, degree_u32);
        set_u32(gather, 5, log_degree);
        set_u32(gather, 6, 0);
        dispatch2d(gather, &ctx.prepare, degree, cols);
        gather.end_encoding();

        for stage in 0..log_degree {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.stage);
            enc.set_buffer(0, Some(coeffs), 0);
            enc.set_buffer(
                1,
                Some(&roots_deg.0),
                (roots_deg.1[stage as usize] * size_of::<u64>()) as NSUInteger,
            );
            set_u32(enc, 2, degree_u32);
            set_u32(enc, 3, stage);
            set_u32(enc, 4, 0);
            dispatch2d(enc, &ctx.stage, degree / 2, cols);
            enc.end_encoding();
        }

        // Forward FFT -> iDFT coefficients (reverse + scale by n^{-1}),
        // written into `col_deg` (its values are dead now).
        let fin = cmd.new_compute_command_encoder();
        fin.set_compute_pipeline_state(&ctx.finalize);
        fin.set_buffer(0, Some(coeffs), 0);
        fin.set_buffer(1, Some(col_deg), 0);
        set_u32(fin, 2, degree_u32);
        fin.set_bytes(
            3,
            size_of::<u64>() as NSUInteger,
            (&n_inv as *const u64).cast(),
        );
        dispatch2d(fin, &ctx.finalize, degree, cols);
        fin.end_encoding();

        // Coset LDE of the coefficients: shift-scaled bit-reversed gather with
        // zero-run replication, then the remaining DIT stages.
        let prepare = cmd.new_compute_command_encoder();
        prepare.set_compute_pipeline_state(&ctx.prepare);
        prepare.set_buffer(0, Some(col_deg), 0);
        prepare.set_buffer(1, Some(shift_pows), 0);
        prepare.set_buffer(2, Some(col_lde), 0);
        set_u32(prepare, 3, degree_u32);
        set_u32(prepare, 4, lde_u32);
        set_u32(prepare, 5, log_degree);
        set_u32(prepare, 6, rate_bits as u32);
        dispatch2d(prepare, &ctx.prepare, lde_size, cols);
        prepare.end_encoding();

        for stage in rate_bits as u32..log_lde {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.stage);
            enc.set_buffer(0, Some(col_lde), 0);
            enc.set_buffer(
                1,
                Some(&roots_lde.0),
                (roots_lde.1[stage as usize] * size_of::<u64>()) as NSUInteger,
            );
            set_u32(enc, 2, lde_u32);
            set_u32(enc, 3, stage);
            set_u32(enc, 4, u32::from(stage == log_lde - 1));
            dispatch2d(enc, &ctx.stage, lde_size / 2, cols);
            enc.end_encoding();
        }
    }

    /// GPU coset LDE. Returns `None` (fall back to the CPU path) when Metal
    /// is unavailable, the input is too small, or any check fails.
    pub(super) fn coset_lde_batch_gpu(
        mat: &RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) -> Option<BitReversedMatrixView<RowMajorMatrix<Goldilocks>>> {
        let degree = mat.height();
        let cols = mat.width();
        if degree < 2
            || !degree.is_power_of_two()
            || cols == 0
            || added_bits == 0
            || degree
                .checked_shl(added_bits as u32)?
                .checked_mul(cols)
                .is_none_or(|n| n < MIN_GPU_ELEMS)
        {
            return None;
        }
        // Measured on M4: the GPU NTT is bandwidth-bound and slightly slower
        // than the parallel CPU FFT when the CPU is otherwise idle, but frees
        // ~1s of CPU when the block pipeline has the cores saturated. Opt-in.
        static ENABLED: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("P3_METAL_NTT").is_ok_and(|v| v == "1"));
        if !*ENABLED {
            return None;
        }
        let ctx = CTX.as_ref()?;
        let lde_size = degree << added_bits;
        let log_lde = lde_size.ilog2();

        let roots_deg = roots_for(ctx, degree.ilog2());
        let roots_lde = roots_for(ctx, log_lde);
        let ones = ones_for(ctx, degree);
        let shift_pows = pow_table_for(ctx, degree, shift);
        let n_inv = Goldilocks::ONE
            .halve()
            .exp_u64(degree.ilog2() as u64)
            .as_canonical_u64();

        let chunk = CHUNK_COLS.min(cols);
        let col_deg = ctx.device.new_buffer(
            (chunk * degree * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        let coeffs = ctx.device.new_buffer(
            (chunk * degree * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        let col_lde = ctx.device.new_buffer(
            (chunk * lde_size * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );

        // Upload the input matrix once; the GPU gathers column blocks from it.
        let input = ctx.device.new_buffer_with_data(
            mat.values.as_ptr().cast(),
            (mat.values.len() * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );

        // Output is in bit-reversed row order, matching Radix2DitParallel.
        // The GPU scatters straight into the output Vec's pages when they are
        // page-aligned (macOS large allocations always are); otherwise it
        // scatters into a staging buffer that is memcpy'd out afterwards.
        let mut out = Goldilocks::zero_vec(lde_size * cols);
        let out_bytes = out.len() * size_of::<u64>();
        let page: usize = 16384;
        let aligned = (out.as_ptr() as usize) % page == 0 && out_bytes % page == 0;
        let out_buffer = if aligned {
            ctx.device.new_buffer_with_bytes_no_copy(
                out.as_mut_ptr().cast(),
                out_bytes as NSUInteger,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            ctx.device.new_buffer(
                out_bytes as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // All chunks go into a single command buffer: the per-chunk working
        // buffers are reused, which is safe because encoders execute in order.
        let ok = objc::rc::autoreleasepool(|| {
            let cmd = ctx.queue.new_command_buffer();
            let mut c0 = 0usize;
            while c0 < cols {
                let c_n = chunk.min(cols - c0);

                let t_in = cmd.new_compute_command_encoder();
                t_in.set_compute_pipeline_state(&ctx.transpose_in);
                t_in.set_buffer(0, Some(&input), 0);
                t_in.set_buffer(1, Some(&col_deg), 0);
                set_u32(t_in, 2, degree as u32);
                set_u32(t_in, 3, cols as u32);
                set_u32(t_in, 4, c0 as u32);
                dispatch2d(t_in, &ctx.transpose_in, degree, c_n);
                t_in.end_encoding();

                encode_lde(
                    ctx,
                    cmd,
                    &col_deg,
                    &coeffs,
                    &col_lde,
                    &ones,
                    &shift_pows,
                    &roots_deg,
                    &roots_lde,
                    degree,
                    lde_size,
                    added_bits,
                    c_n,
                    n_inv,
                );

                let t_out = cmd.new_compute_command_encoder();
                t_out.set_compute_pipeline_state(&ctx.transpose_out);
                t_out.set_buffer(0, Some(&col_lde), 0);
                t_out.set_buffer(1, Some(&out_buffer), 0);
                set_u32(t_out, 2, lde_size as u32);
                set_u32(t_out, 3, log_lde);
                set_u32(t_out, 4, cols as u32);
                set_u32(t_out, 5, c0 as u32);
                dispatch2d(t_out, &ctx.transpose_out, lde_size, c_n);
                t_out.end_encoding();

                c0 += c_n;
            }
            cmd.commit();
            cmd.wait_until_completed();
            cmd.status() == metal::MTLCommandBufferStatus::Completed
        });
        if !ok {
            eprintln!("p3 metal ntt: command buffer failed; falling back to CPU");
            return None;
        }

        if !aligned {
            let staged: &[Goldilocks] = unsafe {
                std::slice::from_raw_parts(out_buffer.contents().cast::<Goldilocks>(), out.len())
            };
            out.par_chunks_mut(1 << 20)
                .zip(staged.par_chunks(1 << 20))
                .for_each(|(dst, src)| dst.copy_from_slice(src));
        }

        Some(BitReversalPerm::new_view(RowMajorMatrix::new(out, cols)))
    }
}

#[cfg(test)]
mod tests {
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;

    use super::MetalDft;

    #[test]
    fn metal_dft_matches_cpu_small() {
        // Small input: exercises the fallback path everywhere.
        let mat = RowMajorMatrix::new(
            (0..(1usize << 10) * 3)
                .map(|i| Goldilocks::from_u64(i as u64 * 7919 + 13))
                .collect(),
            3,
        );
        let shift = Goldilocks::GENERATOR;
        let cpu = Radix2DitParallel::<Goldilocks>::default()
            .coset_lde_batch(mat.clone(), 3, shift)
            .to_row_major_matrix();
        let gpu = MetalDft::default()
            .coset_lde_batch(mat, 3, shift)
            .to_row_major_matrix();
        assert_eq!(cpu, gpu);
    }

    #[test]
    #[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
    fn metal_dft_matches_cpu_large() {
        // Large enough to take the GPU path (2^16 x 512 x blowup 8 = 2^28).
        unsafe { std::env::set_var("P3_METAL_NTT", "1") };
        let cols = 512;
        let degree = 1usize << 16;
        let mat = RowMajorMatrix::new(
            (0..degree * cols)
                .map(|i| Goldilocks::from_u64((i as u64).wrapping_mul(0x9E3779B97F4A7C15)))
                .collect(),
            cols,
        );
        let shift = Goldilocks::GENERATOR;
        let cpu = Radix2DitParallel::<Goldilocks>::default()
            .coset_lde_batch(mat.clone(), 3, shift)
            .to_row_major_matrix();
        let gpu = MetalDft::default()
            .coset_lde_batch(mat, 3, shift)
            .to_row_major_matrix();
        assert_eq!(cpu, gpu);
    }
}
