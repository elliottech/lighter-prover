//! Metal-accelerated first digest layer for large Goldilocks/Poseidon2-16
//! Merkle trees (Apple Silicon only).
//!
//! This is a prover-side optimization only: the digests produced are the same
//! field elements the CPU sponge would produce, so roots, proofs, and
//! verification are unchanged. The GPU path self-checks by re-hashing sample
//! rows on the CPU and falls back to the CPU path on any mismatch or Metal
//! failure.

use alloc::vec;
use alloc::vec::Vec;
use core::any::TypeId;

use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger,
};
use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{
    GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    GOLDILOCKS_POSEIDON2_RC_16_INTERNAL, Goldilocks, MATRIX_DIAG_16_GOLDILOCKS,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;
use std::sync::LazyLock;
use tracing::{debug, warn};

const SHADER_SOURCE: &str = include_str!("poseidon2_16.metal");
const KERNEL_NAME: &str = "p3_poseidon2_16_hash_leaves";
const DIGEST_WIDTH: usize = 8;
const SPONGE_RATE: usize = 8;
/// Below this many permutations the dispatch overhead beats the GPU win.
const MIN_GPU_PERMUTATIONS: usize = 1 << 21;
/// Leaves hashed per dispatch; bounds the staging buffer (rows this size at
/// width 468 stage ~1 GiB).
const CHUNK_LEAVES: usize = 1 << 18;

struct MetalContext {
    device: Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
    params: Buffer,
}

// Metal objects are internally thread-safe reference-counted ObjC objects;
// commands are only encoded while holding the global context.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

fn build_params(device: &Device) -> Buffer {
    let mut params: Vec<u64> = Vec::with_capacity(166);
    for round in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_INITIAL.iter() {
        params.extend(round.iter().map(|c| c.as_canonical_u64()));
    }
    for round in GOLDILOCKS_POSEIDON2_RC_16_EXTERNAL_FINAL.iter() {
        params.extend(round.iter().map(|c| c.as_canonical_u64()));
    }
    params.extend(
        GOLDILOCKS_POSEIDON2_RC_16_INTERNAL
            .iter()
            .map(|c| c.as_canonical_u64()),
    );
    params.extend(
        MATRIX_DIAG_16_GOLDILOCKS
            .iter()
            .map(|c| c.as_canonical_u64()),
    );
    device.new_buffer_with_data(
        params.as_ptr().cast(),
        (params.len() * size_of::<u64>()) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    )
}

static CONTEXT: LazyLock<Option<MetalContext>> = LazyLock::new(|| {
    let device = Device::system_default()?;
    let library = match device.new_library_with_source(SHADER_SOURCE, &CompileOptions::new()) {
        Ok(lib) => lib,
        Err(err) => {
            warn!("p3 metal: shader compile failed: {err}");
            return None;
        }
    };
    let function = library.get_function(KERNEL_NAME, None).ok()?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .ok()?;
    let params = build_params(&device);
    let queue = device.new_command_queue();
    Some(MetalContext {
        device,
        queue,
        pipeline,
        params,
    })
});

/// Attempts the first digest layer on the GPU. Returns `None` (caller falls
/// back to the CPU path) unless the concrete field/digest shape matches the
/// Goldilocks Poseidon2-16 sponge, the tree is large enough to be worth the
/// dispatch, Metal is available, and the sample self-check passes.
pub(crate) fn try_first_digest_layer<P, PW, H, M, const DIGEST_ELEMS: usize>(
    h: &H,
    tallest_matrices: &[&M],
    padded_height: usize,
) -> Option<Vec<[PW::Value; DIGEST_ELEMS]>>
where
    P: PackedValue,
    P::Value: 'static,
    PW: PackedValue,
    PW::Value: 'static,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]> + Sync,
    M: Matrix<P::Value>,
{
    if DIGEST_ELEMS != DIGEST_WIDTH
        || TypeId::of::<P::Value>() != TypeId::of::<Goldilocks>()
        || TypeId::of::<PW::Value>() != TypeId::of::<Goldilocks>()
    {
        return None;
    }
    let height = tallest_matrices[0].height();
    let row_width: usize = tallest_matrices.iter().map(|m| m.width()).sum();
    if row_width == 0 || height * row_width.div_ceil(SPONGE_RATE) < MIN_GPU_PERMUTATIONS {
        return None;
    }
    let ctx = CONTEXT.as_ref()?;

    let default_digest = [PW::Value::default(); DIGEST_ELEMS];
    let mut digests = vec![default_digest; padded_height];

    // Two in-flight chunks: while the GPU hashes chunk k, the CPU stages
    // chunk k+1 into the other buffer pair and drains chunk k-1's digests.
    let chunk_rows = CHUNK_LEAVES.min(height);
    let staging: [Buffer; 2] = core::array::from_fn(|_| {
        ctx.device.new_buffer(
            (chunk_rows * row_width * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    });
    let out: [Buffer; 2] = core::array::from_fn(|_| {
        ctx.device.new_buffer(
            (chunk_rows * DIGEST_WIDTH * size_of::<u64>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let stage_chunk = |buf: &Buffer, start: usize, rows: usize| {
        let staging_slice: &mut [u64] = unsafe {
            core::slice::from_raw_parts_mut(buf.contents().cast::<u64>(), rows * row_width)
        };
        staging_slice
            .par_chunks_exact_mut(row_width)
            .enumerate()
            .for_each(|(i, dst)| {
                let r = start + i;
                let mut offset = 0;
                for m in tallest_matrices {
                    let w = m.width();
                    // Safety: r < height and Goldilocks is a transparent u64.
                    unsafe {
                        for (dst_v, v) in dst[offset..offset + w]
                            .iter_mut()
                            .zip(m.row_unchecked(r).into_iter())
                        {
                            *dst_v = core::mem::transmute_copy::<P::Value, u64>(&v);
                        }
                    }
                    offset += w;
                }
            });
    };

    let dispatch_chunk = |slot: usize, rows: usize| -> metal::CommandBuffer {
        objc::rc::autoreleasepool(|| {
            let cmd = ctx.queue.new_command_buffer().to_owned();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&ctx.pipeline);
            encoder.set_buffer(0, Some(&staging[slot]), 0);
            encoder.set_buffer(1, Some(&out[slot]), 0);
            encoder.set_buffer(2, Some(&ctx.params), 0);
            let row_width_u32 = row_width as u32;
            let rows_u32 = rows as u32;
            encoder.set_bytes(3, 4, (&row_width_u32 as *const u32).cast());
            encoder.set_bytes(4, 4, (&rows_u32 as *const u32).cast());
            let tpg_env = std::env::var("P3_METAL_TPG").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(256);
            let threads_per_group = ctx.pipeline.max_total_threads_per_threadgroup().min(tpg_env);
            let groups = (rows as NSUInteger).div_ceil(threads_per_group);
            encoder.dispatch_thread_groups(
                MTLSize::new(groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
            cmd.commit();
            cmd
        })
    };

    let drain_chunk = |digests: &mut [[PW::Value; DIGEST_ELEMS]], slot: usize, start: usize, rows: usize| {
        let out_slice: &[u64] = unsafe {
            core::slice::from_raw_parts(out[slot].contents().cast::<u64>(), rows * DIGEST_WIDTH)
        };
        for (i, digest) in digests[start..start + rows].iter_mut().enumerate() {
            for (j, d) in digest.iter_mut().enumerate() {
                let v = Goldilocks::from_u64(out_slice[i * DIGEST_WIDTH + j]);
                // Safety: PW::Value == Goldilocks (checked above).
                *d = unsafe { core::mem::transmute_copy::<Goldilocks, PW::Value>(&v) };
            }
        }
    };

    let mut in_flight: Option<(metal::CommandBuffer, usize, usize, usize)> = None;
    let mut start = 0usize;
    let mut slot = 0usize;
    while start < height {
        let rows = CHUNK_LEAVES.min(height - start);
        stage_chunk(&staging[slot], start, rows);
        let cmd = dispatch_chunk(slot, rows);
        if let Some((prev_cmd, prev_slot, prev_start, prev_rows)) = in_flight.take() {
            prev_cmd.wait_until_completed();
            drain_chunk(&mut digests, prev_slot, prev_start, prev_rows);
        }
        in_flight = Some((cmd, slot, start, rows));
        slot ^= 1;
        start += rows;
    }
    if let Some((prev_cmd, prev_slot, prev_start, prev_rows)) = in_flight.take() {
        prev_cmd.wait_until_completed();
        drain_chunk(&mut digests, prev_slot, prev_start, prev_rows);
    }

    // Self-check: re-hash sample rows on the CPU. This guards against a
    // permutation whose runtime constants differ from the p3 defaults as
    // well as any kernel/driver fault.
    for r in [0, height / 2, height - 1] {
        let expected: [PW::Value; DIGEST_ELEMS] =
            unsafe { h.hash_iter(tallest_matrices.iter().flat_map(|m| m.row_unchecked(r))) };
        let expected_u64: Vec<u64> = expected
            .iter()
            .map(|v| unsafe { core::mem::transmute_copy::<PW::Value, Goldilocks>(v) }.as_canonical_u64())
            .collect();
        let got_u64: Vec<u64> = digests[r]
            .iter()
            .map(|v| unsafe { core::mem::transmute_copy::<PW::Value, Goldilocks>(v) }.as_canonical_u64())
            .collect();
        if expected_u64 != got_u64 {
            warn!("p3 metal: GPU digest mismatch at row {r}; falling back to CPU");
            return None;
        }
    }

    debug!("p3 metal: hashed {height} leaves of width {row_width} on GPU");
    Some(digests)
}
