#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }
}

/// Natural-order poly-major column storage, either CPU-owned or retained in a
/// CPU-visible GPU buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ColumnStore<F> {
    Owned(Vec<Vec<F>>),
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    Shared(crate::hash::poseidon2::metal::MetalColumns<F>),
}

impl<F: RichField> ColumnStore<F> {
    pub fn num_cols(&self) -> usize {
        match self {
            ColumnStore::Owned(columns) => columns.len(),
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.cols(),
        }
    }

    pub fn num_rows(&self) -> usize {
        match self {
            ColumnStore::Owned(columns) => columns.first().map_or(0, Vec::len),
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.rows(),
        }
    }

    pub fn col(&self, j: usize) -> &[F] {
        match self {
            ColumnStore::Owned(columns) => &columns[j],
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.col(j),
        }
    }

    pub(crate) fn columns_mut(&mut self) -> Option<Vec<&mut [F]>> {
        match self {
            ColumnStore::Owned(columns) => {
                Some(columns.iter_mut().map(Vec::as_mut_slice).collect())
            }
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(columns) => columns.columns_mut(),
        }
    }
}

/// Backing storage for the Merkle tree leaves.
///
/// Equality is over the logical leaf matrix, not the storage layout: a
/// row-major tree and a column-major tree holding the same leaves compare
/// equal.
#[derive(Clone, Debug)]
pub enum MerkleLeaves<F> {
    /// One flat row-major buffer: leaf `i` occupies `data[i * width..(i + 1) * width]`.
    Rows { data: Vec<F>, width: usize },
    /// Natural-order poly-major columns: leaf `i` holds
    /// `columns.col(j)[reverse_bits(i, log_rows)]` for each column `j`. This is
    /// the layout LDEs are produced in, so committing to them requires no
    /// transpose.
    Columns {
        columns: ColumnStore<F>,
        log_rows: usize,
    },
}

impl<F: RichField> PartialEq for MerkleLeaves<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                MerkleLeaves::Rows { data: a, width: wa },
                MerkleLeaves::Rows { data: b, width: wb },
            ) => wa == wb && a == b,
            (
                MerkleLeaves::Columns {
                    columns: a,
                    log_rows: la,
                },
                MerkleLeaves::Columns {
                    columns: b,
                    log_rows: lb,
                },
            ) => {
                la == lb
                    && a.num_cols() == b.num_cols()
                    && (0..a.num_cols()).all(|j| a.col(j) == b.col(j))
            }
            (MerkleLeaves::Rows { data, width }, MerkleLeaves::Columns { columns, log_rows })
            | (MerkleLeaves::Columns { columns, log_rows }, MerkleLeaves::Rows { data, width }) => {
                let rows = 1usize << log_rows;
                *width == columns.num_cols()
                    && data.len() == rows * width
                    && (0..columns.num_cols()).all(|j| {
                        let col = columns.col(j);
                        (0..rows).all(|i| {
                            data[i * width + j] == col[crate::util::reverse_bits(i, *log_rows)]
                        })
                    })
            }
        }
    }
}

impl<F: RichField> Eq for MerkleLeaves<F> {}

/// Storage for level-order digests. Metal-produced nodes can remain in their
/// immutable shared buffer, avoiding a full-tree copy before sparse queries.
#[derive(Clone, Debug)]
pub enum DigestStore<T> {
    Owned(Vec<T>),
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    Shared(crate::hash::poseidon2::metal::MetalDigests<T>),
}

impl<T> DigestStore<T> {
    pub fn is_shared(&self) -> bool {
        match self {
            Self::Owned(_) => false,
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            Self::Shared(_) => true,
        }
    }
}

impl<T: PartialEq> PartialEq for DigestStore<T> {
    fn eq(&self, other: &Self) -> bool {
        &**self == &**other
    }
}

impl<T: Eq> Eq for DigestStore<T> {}

impl<T> From<Vec<T>> for DigestStore<T> {
    fn from(nodes: Vec<T>) -> Self {
        Self::Owned(nodes)
    }
}

impl<T> core::ops::Deref for DigestStore<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(nodes) => nodes,
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            Self::Shared(nodes) => nodes.as_slice(),
        }
    }
}

/// Merkle tree digests stored in level order, exactly as the fused GPU
/// pipeline writes them: `nodes[level_offsets[l] + i]` is node `i` of level
/// `l`, where level 0 holds the leaf digests and node `i` of level `l` is the
/// digest of the subtree covering leaves `i << l..(i + 1) << l`. The final
/// level (the last `level_offsets` entry) is the cap, so
/// `level_offsets.len() - 1` is the number of hashing layers below the cap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LevelOrderDigests<T> {
    /// All tree nodes as one contiguous level-order array, cap level included.
    pub nodes: DigestStore<T>,
    /// Element offset of each level's first node within `nodes`.
    pub level_offsets: Vec<usize>,
}

impl<T: Copy> LevelOrderDigests<T> {
    /// Node `index` of level `level`.
    pub fn node(&self, level: usize, index: usize) -> T {
        self.nodes[self.level_offsets[level] + index]
    }

    /// Merkle path siblings for `leaf_index`: the sibling hashed in at layer
    /// `l` (layer 0 first) is node `(leaf_index >> l) ^ 1` of level `l`.
    pub(crate) fn prove_siblings(&self, leaf_index: usize) -> Vec<T> {
        (0..self.level_offsets.len() - 1)
            .map(|level| self.node(level, (leaf_index >> level) ^ 1))
            .collect()
    }

    /// Materializes the interleaved recursive-subtree layout documented on
    /// [`MerkleTree::digests`]. Cold path; only serialization needs it.
    pub fn to_interleaved(&self) -> Vec<T> {
        let num_layers = self.level_offsets.len() - 1;
        if num_layers == 0 {
            return Vec::new();
        }
        let num_leaves = self.level_offsets[1] - self.level_offsets[0];
        let cap_count = num_leaves >> num_layers;
        let subtree_leaf_count = num_leaves / cap_count;
        let num_digests = 2 * (num_leaves - cap_count);
        let mut digests = Vec::with_capacity(num_digests);
        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        digests_buf
            .chunks_exact_mut(2 * (subtree_leaf_count - 1))
            .enumerate()
            .for_each(|(cap_index, subtree)| {
                self.fill_interleaved_subtree(
                    subtree,
                    cap_index * subtree_leaf_count,
                    subtree_leaf_count,
                );
            });
        unsafe {
            // SAFETY: `fill_interleaved_subtree` wrote every slot of each
            // subtree chunk, and the chunks cover the whole buffer.
            digests.set_len(num_digests);
        }
        digests
    }

    /// Writes the subtree over `leaf_count` leaves starting at `start_leaf`
    /// in the interleaved layout and returns the subtree root, mirroring
    /// [`fill_subtree`]'s recursion but reading precomputed level-order nodes.
    fn fill_interleaved_subtree(
        &self,
        digests: &mut [MaybeUninit<T>],
        start_leaf: usize,
        leaf_count: usize,
    ) -> T {
        if leaf_count == 1 {
            return self.node(0, start_leaf);
        }
        let (left_half, right_half) = digests.split_at_mut(digests.len() / 2);
        let (left_root, left_digests) = left_half.split_last_mut().unwrap();
        let (right_root, right_digests) = right_half.split_first_mut().unwrap();
        let half = leaf_count / 2;
        left_root.write(self.fill_interleaved_subtree(left_digests, start_leaf, half));
        right_root.write(self.fill_interleaved_subtree(right_digests, start_leaf + half, half));
        let level = leaf_count.ilog2() as usize;
        self.node(level, start_leaf / leaf_count)
    }
}

#[derive(Clone, Debug)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: MerkleLeaves<F>,

    /// The number of leaves.
    pub num_leaves: usize,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// Alternative level-order digest storage, used when a builder backend
    /// (the GPU pipeline) already produced the nodes in that layout. When
    /// this is `Some`, `digests` is empty: `prove` reads the levels directly
    /// and serialization materializes the interleaved layout on demand.
    pub level_digests: Option<LevelOrderDigests<H::Hash>>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

/// Equality is over the logical tree, independent of storage layout: leaves
/// compare as matrices and digests compare in the interleaved layout, so a
/// tree round-tripped through serialization equals the original even when the
/// original held column-major leaves or level-order digests.
impl<F: RichField, H: Hasher<F>> PartialEq for MerkleTree<F, H> {
    fn eq(&self, other: &Self) -> bool {
        self.num_leaves == other.num_leaves
            && self.cap == other.cap
            && self.leaves == other.leaves
            && match (&self.level_digests, &other.level_digests) {
                (None, None) => self.digests == other.digests,
                (Some(a), Some(b)) => a == b,
                (Some(levels), None) => levels.to_interleaved() == other.digests,
                (None, Some(levels)) => self.digests == levels.to_interleaved(),
            }
    }
}

impl<F: RichField, H: Hasher<F>> Eq for MerkleTree<F, H> {}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: MerkleLeaves::Rows {
                data: Vec::new(),
                width: 0,
            },
            num_leaves: 0,
            digests: Vec::new(),
            level_digests: None,
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

pub(crate) fn fill_subtree_flat<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_width: usize,
    num_leaves: usize,
) -> H::Hash {
    debug_assert_eq!(num_leaves, digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(leaves)
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        let half = num_leaves / 2;
        let (left_leaves, right_leaves) = leaves.split_at(half * leaf_width);

        // Sibling leaves are independent; hash them as one interleaved pair so
        // the two permutation dependency chains overlap in the pipeline.
        if num_leaves == 2 {
            let (left_digest, right_digest) = H::hash_or_noop_pair(left_leaves, right_leaves);
            left_digest_mem.write(left_digest);
            right_digest_mem.write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // Same idea one level up: hash the four leaves as one interleaved
        // quad, then compress the two sibling parent nodes as a pair.
        if num_leaves == 4 {
            let (leaf_0, leaf_1) = left_leaves.split_at(leaf_width);
            let (leaf_2, leaf_3) = right_leaves.split_at(leaf_width);
            let (h0, h1, h2, h3) = H::hash_or_noop_quad(leaf_0, leaf_1, leaf_2, leaf_3);
            left_digests_buf[0].write(h0);
            left_digests_buf[1].write(h1);
            right_digests_buf[0].write(h2);
            right_digests_buf[1].write(h3);
            let (left_digest, right_digest) = H::two_to_one_pair(h0, h1, h2, h3);
            left_digest_mem.write(left_digest);
            right_digest_mem.write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // And one more level: two leaf quads, then a quad of level-1
        // compressions, then a pair of level-2 compressions.
        if num_leaves == 8 {
            let (leaf_0, rest) = left_leaves.split_at(leaf_width);
            let (leaf_1, rest2) = rest.split_at(leaf_width);
            let (leaf_2, leaf_3) = rest2.split_at(leaf_width);
            let (leaf_4, rest) = right_leaves.split_at(leaf_width);
            let (leaf_5, rest2) = rest.split_at(leaf_width);
            let (leaf_6, leaf_7) = rest2.split_at(leaf_width);

            let (h0, h1, h2, h3) = H::hash_or_noop_quad(leaf_0, leaf_1, leaf_2, leaf_3);
            let (h4, h5, h6, h7) = H::hash_or_noop_quad(leaf_4, leaf_5, leaf_6, leaf_7);
            let [n01, n23, n45, n67] = H::two_to_one_quad([(h0, h1), (h2, h3), (h4, h5), (h6, h7)]);

            left_digests_buf[0].write(h0);
            left_digests_buf[1].write(h1);
            left_digests_buf[2].write(n01);
            left_digests_buf[3].write(n23);
            left_digests_buf[4].write(h2);
            left_digests_buf[5].write(h3);
            right_digests_buf[0].write(h4);
            right_digests_buf[1].write(h5);
            right_digests_buf[2].write(n45);
            right_digests_buf[3].write(n67);
            right_digests_buf[4].write(h6);
            right_digests_buf[5].write(h7);

            let (left_digest, right_digest) = H::two_to_one_pair(n01, n23, n45, n67);
            left_digest_mem.write(left_digest);
            right_digest_mem.write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // Whole synchronous 16-leaf subtree, level order: four leaf quads,
        // two quads then one quad of level-1/2 compressions, one level-3 pair.
        if num_leaves == 16 {
            let mut leaf_digests = [core::mem::MaybeUninit::<H::Hash>::uninit(); 16];
            for q in 0..4 {
                let base = q * 4 * leaf_width;
                let quad_leaves = if q < 2 { left_leaves } else { right_leaves };
                let base = base - if q < 2 { 0 } else { 8 * leaf_width };
                let (leaf_a, rest) = quad_leaves[base..base + 4 * leaf_width].split_at(leaf_width);
                let (leaf_b, rest2) = rest.split_at(leaf_width);
                let (leaf_c, leaf_d) = rest2.split_at(leaf_width);
                let (ha, hb, hc, hd) = H::hash_or_noop_quad(leaf_a, leaf_b, leaf_c, leaf_d);
                leaf_digests[4 * q].write(ha);
                leaf_digests[4 * q + 1].write(hb);
                leaf_digests[4 * q + 2].write(hc);
                leaf_digests[4 * q + 3].write(hd);
            }
            // SAFETY: all 16 entries were initialized above.
            let h: [H::Hash; 16] = unsafe { core::mem::transmute_copy(&leaf_digests) };

            let n_a = H::two_to_one_quad([(h[0], h[1]), (h[2], h[3]), (h[4], h[5]), (h[6], h[7])]);
            let n_b =
                H::two_to_one_quad([(h[8], h[9]), (h[10], h[11]), (h[12], h[13]), (h[14], h[15])]);
            let m = H::two_to_one_quad([
                (n_a[0], n_a[1]),
                (n_a[2], n_a[3]),
                (n_b[0], n_b[1]),
                (n_b[2], n_b[3]),
            ]);
            let (left_digest, right_digest) = H::two_to_one_pair(m[0], m[1], m[2], m[3]);

            // Each 8-leaf child's buffer half follows the recursive layout:
            // [4-leaf region | node | node | 4-leaf region], with leaf digests
            // at region positions 0, 1, 4, 5 and level-1 nodes at 2, 3.
            for half in 0..2 {
                let buf: &mut [MaybeUninit<H::Hash>] = if half == 0 {
                    &mut *left_digests_buf
                } else {
                    &mut *right_digests_buf
                };
                let h = &h[8 * half..8 * (half + 1)];
                let n = if half == 0 { &n_a } else { &n_b };
                let m = &m[2 * half..2 * (half + 1)];
                buf[0].write(h[0]);
                buf[1].write(h[1]);
                buf[2].write(n[0]);
                buf[3].write(n[1]);
                buf[4].write(h[2]);
                buf[5].write(h[3]);
                buf[6].write(m[0]);
                buf[7].write(m[1]);
                buf[8].write(h[4]);
                buf[9].write(h[5]);
                buf[10].write(n[2]);
                buf[11].write(n[3]);
                buf[12].write(h[6]);
                buf[13].write(h[7]);
            }
            left_digest_mem.write(left_digest);
            right_digest_mem.write(right_digest);
            return H::two_to_one(left_digest, right_digest);
        }

        // Rayon task creation dominates the tiny subtrees near the leaves. Keep
        // enough parallelism at the upper levels, then recurse synchronously.
        let (left_digest, right_digest) = if num_leaves > 64 {
            plonky2_maybe_rayon::join(
                || fill_subtree_flat::<F, H>(left_digests_buf, left_leaves, leaf_width, half),
                || fill_subtree_flat::<F, H>(right_digests_buf, right_leaves, leaf_width, half),
            )
        } else {
            (
                fill_subtree_flat::<F, H>(left_digests_buf, left_leaves, leaf_width, half),
                fill_subtree_flat::<F, H>(right_digests_buf, right_leaves, leaf_width, half),
            )
        };

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

/// Leaves materialized per gathered tile. This is exactly the size of
/// [`fill_subtree_flat`]'s fully unrolled synchronous case, so one tile is
/// consumed by one call and never outlives it.
const GATHER_TILE_LEAVES: usize = 16;

/// Widest leaf the gathering path keeps on the stack. The tile is
/// `GATHER_TILE_LEAVES * GATHER_MAX_WIDTH` field elements (17 KiB at
/// `size_of::<F>() == 8`). The array is created only in the base case, so one
/// tile is live per active leaf task rather than at every recursion level.
/// Wider leaves take the materializing path in
/// [`MerkleTree::new_column_store`].
const GATHER_MAX_WIDTH: usize = 136;

/// [`fill_subtree_flat`] over natural-order columns instead of a materialized
/// bit-reversed row-major matrix: leaf `i` of the tree is
/// `columns[j][reverse_bits(i, log_rows)]` for each column `j`.
///
/// The recursion, the `rayon::join` threshold and the digest layout are
/// character for character those of [`fill_subtree_flat`]; the only difference
/// is where the leaf bytes come from. At the bottom the tile is gathered and
/// handed to [`fill_subtree_flat`] itself, so every hash — leaf sponge,
/// interleaved pair/quad compressions, the whole 16-leaf unrolled block — is
/// the same code operating on the same leaf bytes in the same order. The
/// gathered tile holds `columns[j][reverse_bits(start_leaf + leaf, log_rows)]`
/// at offset `leaf * width + j`, which is by definition the row that
/// `transpose_to_bitrev_flat` writes at output row `start_leaf + leaf`, so
/// each leaf's field elements are bit-for-bit the ones the materializing path
/// hashes.
pub(crate) fn fill_subtree_gather<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    columns: &[&[F]],
    log_rows: usize,
    start_leaf: usize,
    num_leaves: usize,
) -> H::Hash {
    debug_assert_eq!(num_leaves, digests_buf.len() / 2 + 1);
    let width = columns.len();
    debug_assert!(width <= GATHER_MAX_WIDTH);

    if num_leaves <= GATHER_TILE_LEAVES {
        // SAFETY: an array of `MaybeUninit` needs no initialization; this is
        // the stable spelling of the unstable `MaybeUninit::uninit_array`.
        let mut tile: [MaybeUninit<F>; GATHER_TILE_LEAVES * GATHER_MAX_WIDTH] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for leaf in 0..num_leaves {
            let natural = crate::util::reverse_bits(start_leaf + leaf, log_rows);
            let row = &mut tile[leaf * width..(leaf + 1) * width];
            for (slot, column) in row.iter_mut().zip(columns) {
                // SAFETY: `natural < 1 << log_rows == column.len()` because
                // `reverse_bits(_, log_rows)` maps `0..1 << log_rows` onto
                // itself, and `start_leaf + leaf` is within the tree.
                slot.write(unsafe { *column.get_unchecked(natural) });
            }
        }
        // SAFETY: the loop above wrote every one of the first
        // `num_leaves * width` slots exactly once, and `num_leaves <=
        // GATHER_TILE_LEAVES` and `width <= GATHER_MAX_WIDTH` keep that prefix
        // inside the array. `MaybeUninit<F>` has the same layout as `F`.
        let leaves: &[F] =
            unsafe { slice::from_raw_parts(tile.as_ptr().cast::<F>(), num_leaves * width) };
        return fill_subtree_flat::<F, H>(digests_buf, leaves, width, num_leaves);
    }

    // Layout is: left recursive output || left child digest
    //             || right child digest || right recursive output.
    let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
    let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
    let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
    let half = num_leaves / 2;

    // Same threshold as `fill_subtree_flat`: rayon task creation dominates the
    // tiny subtrees near the leaves.
    let (left_digest, right_digest) = if num_leaves > 64 {
        plonky2_maybe_rayon::join(
            || fill_subtree_gather::<F, H>(left_digests_buf, columns, log_rows, start_leaf, half),
            || {
                fill_subtree_gather::<F, H>(
                    right_digests_buf,
                    columns,
                    log_rows,
                    start_leaf + half,
                    half,
                )
            },
        )
    } else {
        (
            fill_subtree_gather::<F, H>(left_digests_buf, columns, log_rows, start_leaf, half),
            fill_subtree_gather::<F, H>(
                right_digests_buf,
                columns,
                log_rows,
                start_leaf + half,
                half,
            ),
        )
    };

    left_digest_mem.write(left_digest);
    right_digest_mem.write(right_digest);
    H::two_to_one(left_digest, right_digest)
}

/// [`fill_digests_buf_flat`] driven from natural-order columns. The subtree
/// partition is identical; only the leaf source differs.
pub(crate) fn fill_digests_buf_gather<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    columns: &[&[F]],
    log_rows: usize,
    num_leaves: usize,
    cap_height: usize,
) {
    let width = columns.len();
    // Special case of a tree that's all cap.
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), num_leaves);
        cap_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(leaf, cap_buf)| {
                // SAFETY: as in `fill_subtree_gather`; one leaf's worth.
                let mut tile: [MaybeUninit<F>; GATHER_MAX_WIDTH] =
                    unsafe { MaybeUninit::uninit().assume_init() };
                let natural = crate::util::reverse_bits(leaf, log_rows);
                for (slot, column) in tile[..width].iter_mut().zip(columns) {
                    slot.write(unsafe { *column.get_unchecked(natural) });
                }
                // SAFETY: the first `width` slots were just written.
                let leaf: &[F] = unsafe { slice::from_raw_parts(tile.as_ptr().cast::<F>(), width) };
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = num_leaves >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    digests_chunks.zip(cap_buf).enumerate().for_each(
        |(subtree_index, (subtree_digests, subtree_cap))| {
            subtree_cap.write(fill_subtree_gather::<F, H>(
                subtree_digests,
                columns,
                log_rows,
                subtree_index * subtree_leaves_len,
                subtree_leaves_len,
            ));
        },
    );
}

pub(crate) fn fill_digests_buf_flat<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_width: usize,
    num_leaves: usize,
    cap_height: usize,
) {
    // Special case of a tree that's all cap.
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), num_leaves);
        cap_buf.par_iter_mut().enumerate().for_each(|(i, cap_buf)| {
            cap_buf.write(H::hash_or_noop(
                &leaves[i * leaf_width..(i + 1) * leaf_width],
            ));
        });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = num_leaves >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    digests_chunks.zip(cap_buf).enumerate().for_each(
        |(subtree_index, (subtree_digests, subtree_cap))| {
            let leaf_start = subtree_index * subtree_leaves_len * leaf_width;
            let leaf_end = (subtree_index + 1) * subtree_leaves_len * leaf_width;
            subtree_cap.write(fill_subtree_flat::<F, H>(
                subtree_digests,
                &leaves[leaf_start..leaf_end],
                leaf_width,
                subtree_leaves_len,
            ));
        },
    );
}

pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    /// Returns the natural-order, column-major Metal leaf storage when this
    /// commitment retained its LDE in a shared GPU-visible buffer.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn shared_columns(&self) -> Option<&crate::hash::poseidon2::metal::MetalColumns<F>> {
        match &self.leaves {
            MerkleLeaves::Columns {
                columns: ColumnStore::Shared(columns),
                ..
            } => Some(columns),
            _ => None,
        }
    }

    /// Build a tree from per-leaf vectors. All leaves must have the same width.
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let num_leaves = leaves.len();
        let leaf_width = leaves.first().map_or(0, Vec::len);
        debug_assert!(
            leaves.iter().all(|leaf| leaf.len() == leaf_width),
            "all leaves must have the same width"
        );
        let mut flat = Vec::with_capacity(num_leaves * leaf_width);
        for leaf in &leaves {
            flat.extend_from_slice(leaf);
        }
        Self::from_flat_parts(flat, leaf_width, num_leaves, cap_height)
    }

    /// Build a tree from one flat row-major buffer of `leaves.len() / leaf_width` leaves.
    pub fn new_flat(leaves: Vec<F>, leaf_width: usize, cap_height: usize) -> Self {
        assert!(leaf_width > 0, "flat construction requires nonzero width");
        let num_leaves = leaves.len() / leaf_width;
        assert_eq!(leaves.len(), num_leaves * leaf_width);
        Self::from_flat_parts(leaves, leaf_width, num_leaves, cap_height)
    }

    /// Build a tree directly from the natural-order poly-major LDE columns,
    /// without materializing the transposed leaf matrix. Leaf `i` is
    /// `columns[j][reverse_bits(i, log_rows)]`.
    pub fn new_columns(columns: Vec<Vec<F>>, cap_height: usize) -> Self {
        Self::new_column_store(ColumnStore::Owned(columns), cap_height)
    }

    /// Build a tree from retained natural-order poly-major column storage.
    pub(crate) fn new_column_store(columns: ColumnStore<F>, cap_height: usize) -> Self {
        let num_leaves = columns.num_rows();
        debug_assert!((0..columns.num_cols()).all(|j| columns.col(j).len() == num_leaves));
        let log_rows = log2_strict(num_leaves);
        assert!(
            cap_height <= log_rows,
            "cap_height={cap_height} should be at most log2(leaves.len())={log_rows}"
        );

        if let Some((level_digests, cap)) =
            H::try_build_merkle_tree_column_store(&columns, cap_height)
        {
            debug_assert_eq!(
                level_digests.nodes.len(),
                2 * num_leaves - (1 << cap_height)
            );
            debug_assert_eq!(cap.len(), 1 << cap_height);
            return Self {
                leaves: MerkleLeaves::Columns { columns, log_rows },
                num_leaves,
                digests: Vec::new(),
                level_digests: Some(level_digests),
                cap: MerkleCap(cap),
            };
        }

        // CPU fallback. The leaves are already in memory as natural-order
        // columns, so the row-major bit-reversed matrix the hashing wants is
        // pure data movement: the materializing path below writes every leaf
        // to a `num_leaves * num_columns` buffer and the hashing then reads it
        // straight back. The gathering path skips that buffer and assembles
        // one 16-leaf tile at a time directly into the hasher's input, so the
        // scattered column reads issue alongside the permutations that consume
        // them instead of forming their own pass, and the tile stays in L1
        // instead of streaming a buffer that is tens of megabytes at the
        // shapes this path sees. Both paths present each leaf to
        // `fill_subtree_flat` as the same `num_columns` field elements in the
        // same order, so every digest is bit-identical (asserted below by
        // `gathered_matches_materialized`).
        let num_columns = columns.num_cols();
        if num_columns <= GATHER_MAX_WIDTH {
            let (digests, cap) = {
                let borrowed: Vec<&[F]> = (0..num_columns).map(|j| columns.col(j)).collect();
                Self::cpu_digests_gather(&borrowed, log_rows, num_leaves, cap_height)
            };
            return Self {
                leaves: MerkleLeaves::Columns { columns, log_rows },
                num_leaves,
                digests,
                level_digests: None,
                cap: MerkleCap(cap),
            };
        }

        let flat = match &columns {
            ColumnStore::Owned(owned) => crate::util::transpose_to_bitrev_flat(owned),
            #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
            ColumnStore::Shared(_) => {
                // The fill below writes every slot of this buffer before anything
                // reads it, so the zeroing `vec![F::ZERO; _]` performed first was a
                // dead store pass over the whole `num_leaves * num_columns`
                // allocation — tens of megabytes at the shapes this path sees, and
                // not `calloc`-elidable because the pass then dirties every page
                // anyway. Reserving the capacity uninitialized and writing straight
                // into it is the pattern the sibling `Owned` arm of this same match
                // already uses (`crate::util::transpose_to_bitrev_flat`). The values
                // written, their order, and the buffer handed to `cpu_digests` are
                // unchanged.
                let mut flat: Vec<F> = Vec::with_capacity(num_leaves * num_columns);
                {
                    let spare = &mut flat.spare_capacity_mut()[..num_leaves * num_columns];
                    spare
                        .par_chunks_mut(num_columns)
                        .enumerate()
                        .for_each(|(leaf, row)| {
                            let natural = crate::util::reverse_bits(leaf, log_rows);
                            for (column, slot) in row.iter_mut().enumerate() {
                                slot.write(columns.col(column)[natural]);
                            }
                        });
                }

                // SAFETY: `par_chunks_mut(num_columns)` partitions the whole
                // `num_leaves * num_columns` prefix into `num_leaves` disjoint chunks
                // of exactly `num_columns` slots (the length is an exact multiple, so
                // there is no short remainder), and the loop above writes every slot
                // of every chunk exactly once, so the prefix is fully initialized.
                unsafe { flat.set_len(num_leaves * num_columns) };
                flat
            }
        };
        let (digests, cap) = Self::cpu_digests(&flat, num_columns, num_leaves, cap_height);
        Self {
            leaves: MerkleLeaves::Columns { columns, log_rows },
            num_leaves,
            digests,
            level_digests: None,
            cap: MerkleCap(cap),
        }
    }

    /// Wraps an already-hashed column store (e.g. from the fused GPU NTT +
    /// Merkle pipeline) into a tree.
    pub fn from_prebuilt_columns(
        columns: ColumnStore<F>,
        level_digests: LevelOrderDigests<H::Hash>,
        cap: Vec<H::Hash>,
    ) -> Self {
        let num_leaves = columns.num_rows();
        let log_rows = log2_strict(num_leaves);
        debug_assert_eq!(level_digests.nodes.len(), 2 * num_leaves - cap.len());
        Self {
            leaves: MerkleLeaves::Columns { columns, log_rows },
            num_leaves,
            digests: Vec::new(),
            level_digests: Some(level_digests),
            cap: MerkleCap(cap),
        }
    }

    /// [`Self::cpu_digests`] reading natural-order columns instead of a
    /// materialized bit-reversed matrix.
    fn cpu_digests_gather(
        columns: &[&[F]],
        log_rows: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> (Vec<H::Hash>, Vec<H::Hash>) {
        let num_digests = 2 * (num_leaves - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf_gather::<F, H>(
            digests_buf,
            cap_buf,
            columns,
            log_rows,
            num_leaves,
            cap_height,
        );

        unsafe {
            // SAFETY: as in `cpu_digests` — `fill_digests_buf_gather` writes
            // every one of the `num_digests` digest slots and every one of the
            // `len_cap` cap slots before returning.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        (digests, cap)
    }

    fn cpu_digests(
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> (Vec<H::Hash>, Vec<H::Hash>) {
        let num_digests = 2 * (num_leaves - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf_flat::<F, H>(
            digests_buf,
            cap_buf,
            leaves,
            leaf_width,
            num_leaves,
            cap_height,
        );

        unsafe {
            // SAFETY: `fill_digests_buf_flat` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        (digests, cap)
    }

    fn from_flat_parts(
        leaves: Vec<F>,
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> Self {
        let log2_leaves_len = log2_strict(num_leaves);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        if let Some((level_digests, cap)) =
            H::try_build_merkle_tree(&leaves, leaf_width, num_leaves, cap_height)
        {
            debug_assert_eq!(
                level_digests.nodes.len(),
                2 * num_leaves - (1 << cap_height)
            );
            debug_assert_eq!(cap.len(), 1 << cap_height);
            return Self {
                leaves: MerkleLeaves::Rows {
                    data: leaves,
                    width: leaf_width,
                },
                num_leaves,
                digests: Vec::new(),
                level_digests: Some(level_digests),
                cap: MerkleCap(cap),
            };
        }

        let (digests, cap) = Self::cpu_digests(&leaves, leaf_width, num_leaves, cap_height);
        Self {
            leaves: MerkleLeaves::Rows {
                data: leaves,
                width: leaf_width,
            },
            num_leaves,
            digests,
            level_digests: None,
            cap: MerkleCap(cap),
        }
    }

    /// The number of field elements per leaf.
    pub fn leaf_width(&self) -> usize {
        match &self.leaves {
            MerkleLeaves::Rows { width, .. } => *width,
            MerkleLeaves::Columns { columns, .. } => columns.num_cols(),
        }
    }

    /// Borrow leaf `i`. Only available for row-major storage.
    pub fn get(&self, i: usize) -> &[F] {
        match &self.leaves {
            MerkleLeaves::Rows { data, width } => &data[i * width..(i + 1) * width],
            MerkleLeaves::Columns { .. } => {
                panic!("MerkleTree::get is unavailable for column-major leaves")
            }
        }
    }

    /// Copy leaf `i` out of either storage layout.
    pub fn leaf_vec(&self, i: usize) -> Vec<F> {
        match &self.leaves {
            MerkleLeaves::Rows { data, width } => data[i * width..(i + 1) * width].to_vec(),
            MerkleLeaves::Columns { columns, log_rows } => {
                let natural = crate::util::reverse_bits(i, *log_rows);
                (0..columns.num_cols())
                    .map(|j| columns.col(j)[natural])
                    .collect()
            }
        }
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings = match &self.level_digests {
            Some(levels) => {
                debug_assert_eq!(
                    levels.level_offsets.len() - 1,
                    log2_strict(self.num_leaves) - cap_height
                );
                levels.prove_siblings(leaf_index)
            }
            None => {
                merkle_tree_prove::<F, H>(leaf_index, self.num_leaves, cap_height, &self.digests)
            }
        };

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;
    use plonky2_field::types::{PrimeField64, Sample};

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    fn equality_is_layout_agnostic() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        let log_n = 6;
        let n = 1 << log_n;
        let width = 7;
        let leaves = random_data::<F>(n, width);
        let row_tree = MerkleTree::<F, H>::new(leaves.clone(), 1);

        // Natural-order columns: column j holds leaf reverse_bits(i)'s j-th element.
        let columns: Vec<Vec<F>> = (0..width)
            .map(|j| {
                (0..n)
                    .map(|natural| leaves[crate::util::reverse_bits(natural, log_n)][j])
                    .collect()
            })
            .collect();
        let col_tree = MerkleTree::<F, H>::new_columns(columns, 1);

        assert_eq!(row_tree, col_tree);
        assert_eq!(col_tree, row_tree);

        let other = MerkleTree::<F, H>::new(random_data::<F>(n, width), 1);
        assert_ne!(row_tree, other);
        assert_ne!(col_tree, other);
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    /// Differential: the gathering CPU build must reproduce the materializing
    /// one bit for bit. The comparison is on raw `to_noncanonical_u64` limbs,
    /// not on `HashOut`'s `PartialEq`, because the field canonicalises when it
    /// compares and would hide a changed representative.
    #[test]
    fn gathered_matches_materialized() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        let raw = |digests: &[<H as Hasher<F>>::Hash]| -> Vec<u64> {
            digests
                .iter()
                .flat_map(|hash| hash.elements.iter().map(|e| e.to_noncanonical_u64()))
                .collect()
        };

        // Widths spanning the `hash_or_noop` no-op boundary (<= 4 elements),
        // one and two sponge chunks, a non-power width and the widest the
        // gathering path accepts; leaf counts spanning the tile size, the
        // rayon-join threshold and several levels above it; cap heights
        // including the all-cap degenerate tree.
        for &(width, log_n, cap_height) in &[
            (1usize, 5usize, 0usize),
            (4, 5, 5),
            (4, 6, 2),
            (7, 4, 0),
            (7, 8, 3),
            (12, 4, 4),
            (16, 10, 4),
            (20, 9, 2),
            (31, 7, 1),
            (32, 11, 4),
            (33, 7, 2),
            (40, 9, 3),
            (63, 8, 4),
            (64, 10, 4),
            (65, 7, 2),
            (82, 9, 3),
            (135, 8, 4),
            (136, 10, 4),
        ] {
            let n = 1usize << log_n;
            let columns: Vec<Vec<F>> = (0..width).map(|_| F::rand_vec(n)).collect();

            // Materializing reference, built through the same hashing code the
            // gathering path calls, from the transposed matrix.
            let flat = crate::util::transpose_to_bitrev_flat(&columns);
            let (want_digests, want_cap) =
                MerkleTree::<F, H>::cpu_digests(&flat, width, n, cap_height);

            let borrowed: Vec<&[F]> = columns.iter().map(Vec::as_slice).collect();
            let (got_digests, got_cap) =
                MerkleTree::<F, H>::cpu_digests_gather(&borrowed, log_n, n, cap_height);

            assert_eq!(
                raw(&want_digests),
                raw(&got_digests),
                "digests: width={width} n={n} cap_height={cap_height}"
            );
            assert_eq!(
                raw(&want_cap),
                raw(&got_cap),
                "cap: width={width} n={n} cap_height={cap_height}"
            );

            // And end to end through the public constructor, including the
            // Merkle proofs the tree serves out of the interleaved layout.
            let tree = MerkleTree::<F, H>::new_columns(columns.clone(), cap_height);
            assert!(tree.level_digests.is_none());
            assert_eq!(raw(&tree.cap.0), raw(&want_cap));
            assert_eq!(raw(&tree.digests), raw(&want_digests));
            for i in (0..n).step_by(((n / 8) + 1).max(1)) {
                let proof = tree.prove(i);
                verify_merkle_proof_to_cap(tree.leaf_vec(i), i, &tree.cap, &proof).unwrap();
            }
        }
    }

    /// Builds the level-order digest representation directly from the leaves,
    /// bottom-up, matching the GPU buffer convention: level 0 holds the leaf
    /// digests, each level's parents follow, the last level is the cap.
    fn cpu_level_order<F: RichField, H: Hasher<F>>(
        leaves: &[Vec<F>],
        cap_height: usize,
    ) -> (LevelOrderDigests<H::Hash>, Vec<H::Hash>) {
        let num_layers = log2_strict(leaves.len()) - cap_height;
        let mut nodes: Vec<H::Hash> = leaves.iter().map(|leaf| H::hash_or_noop(leaf)).collect();
        let mut level_offsets = vec![0];
        let mut level_start = 0;
        for _ in 0..num_layers {
            let next_start = nodes.len();
            for pair in 0..(next_start - level_start) / 2 {
                let left = nodes[level_start + 2 * pair];
                let right = nodes[level_start + 2 * pair + 1];
                nodes.push(H::two_to_one(left, right));
            }
            level_offsets.push(next_start);
            level_start = next_start;
        }
        let cap = nodes[level_start..].to_vec();
        (
            LevelOrderDigests {
                nodes: nodes.into(),
                level_offsets,
            },
            cap,
        )
    }

    /// Differential test: a tree backed by level-order digests must produce
    /// exactly the same cap, the same proof for every leaf, and the same
    /// materialized interleaved digest array as the CPU-built interleaved
    /// tree over the same leaves.
    #[test]
    fn test_level_order_digests_match_interleaved() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        for &(n, cap_height) in &[(32usize, 0usize), (256, 3), (1024, 10)] {
            // 7 is a non-power width; 4 exercises the hash_or_noop leaves.
            for &width in &[4usize, 7, 12] {
                let leaves = random_data::<F>(n, width);
                let interleaved = MerkleTree::<F, H>::new(leaves.clone(), cap_height);
                assert!(
                    interleaved.level_digests.is_none(),
                    "reference tree must use the interleaved CPU layout"
                );
                let (levels, cap) = cpu_level_order::<F, H>(&leaves, cap_height);

                assert_eq!(
                    cap, interleaved.cap.0,
                    "cap: n={n} cap_height={cap_height} width={width}"
                );
                assert_eq!(
                    levels.to_interleaved(),
                    interleaved.digests,
                    "materialized digests: n={n} cap_height={cap_height} width={width}"
                );

                let level_tree = MerkleTree::<F, H> {
                    leaves: interleaved.leaves.clone(),
                    num_leaves: n,
                    digests: Vec::new(),
                    level_digests: Some(levels),
                    cap: MerkleCap(cap),
                };

                for (i, leaf) in leaves.iter().enumerate() {
                    let expected = interleaved.prove(i);
                    let actual = level_tree.prove(i);
                    assert_eq!(
                        expected.siblings, actual.siblings,
                        "proof: leaf {i}, n={n} cap_height={cap_height} width={width}"
                    );
                    verify_merkle_proof_to_cap(leaf.clone(), i, &interleaved.cap, &expected)?;
                    verify_merkle_proof_to_cap(leaf.clone(), i, &level_tree.cap, &actual)?;
                }
            }
        }

        Ok(())
    }

    /// The `ColumnStore::Shared` arm of [`MerkleTree::new_column_store`]'s
    /// materializing fallback — the arm taken when a shared allocation is
    /// admitted but the GPU then declines the build that follows it — fills a
    /// reservation of uninitialized capacity rather than overwriting a zeroed
    /// buffer. Nothing else in this suite reaches that arm: the gathering path
    /// above it absorbs every width up to `GATHER_MAX_WIDTH`, and every tree
    /// this prover builds is narrower than that.
    ///
    /// So the reservation is checked here against the sibling `Owned` arm of
    /// the same `match`, which materializes the same matrix through
    /// `transpose_to_bitrev_flat`. Both arms hand their buffer to the same
    /// `cpu_digests`, and every element of a leaf feeds its digest, so
    /// agreement on raw digest limbs is agreement on every slot of the
    /// buffer: a slot left uninitialized, or written with the wrong value,
    /// moves the digest of the leaf that contains it.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn shared_column_fallback_matches_owned() {
        use crate::hash::poseidon2::metal;

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        // Wider than the gathering path accepts, so the materializing `match`
        // is reached at all. The shape is then chosen so `gpu_worthwhile` is
        // true under the exclusive-phase cutoff (2^16 permutations) and false
        // under the ordinary one (2^19): 25 * 4096 leaf permutations + 4095
        // parent permutations = 106495. The allocation is therefore admitted
        // and the build that follows it is declined, which is precisely the
        // pairing that reaches the arm under test.
        const WIDTH: usize = 200;
        const LOG_N: usize = 12;
        const CAP_HEIGHT: usize = 0;
        assert!(WIDTH > GATHER_MAX_WIDTH);
        let n = 1usize << LOG_N;

        let columns: Vec<Vec<F>> = (0..WIDTH).map(|_| F::rand_vec(n)).collect();

        metal::set_exclusive_gpu_phase(true);
        // The allocation entry point spends a one-shot latch while the Metal
        // context is still coming up; the request after it routes normally.
        let mut shared = None;
        for _ in 0..3 {
            shared = metal::allocate_columns::<F>(WIDTH, n, CAP_HEIGHT);
            if shared.is_some() {
                break;
            }
        }
        metal::set_exclusive_gpu_phase(false);

        let Some(mut shared) = shared else {
            // No Metal device on this host: no shared store can exist, so the
            // arm is unreachable here rather than unchecked.
            return;
        };

        {
            let mut slots = shared.columns_mut().expect("a fresh handle is unique");
            assert_eq!(slots.len(), WIDTH);
            for (slot, column) in slots.iter_mut().zip(&columns) {
                slot.copy_from_slice(column);
            }
        }

        let got = MerkleTree::<F, H>::new_column_store(ColumnStore::Shared(shared), CAP_HEIGHT);
        let want = MerkleTree::<F, H>::new_column_store(ColumnStore::Owned(columns), CAP_HEIGHT);

        // Both must have taken the CPU fallback: a GPU-built tree carries its
        // digests in `level_digests` instead, which would leave the comparison
        // below comparing two empty vectors.
        assert!(
            got.level_digests.is_none(),
            "the shared-column CPU fallback was not reached"
        );
        assert!(
            want.level_digests.is_none(),
            "the owned CPU fallback was not reached"
        );
        assert!(!got.digests.is_empty());

        let raw = |digests: &[<H as Hasher<F>>::Hash]| -> Vec<u64> {
            digests
                .iter()
                .flat_map(|hash| hash.elements.iter().map(|e| e.to_noncanonical_u64()))
                .collect()
        };
        assert_eq!(raw(&got.digests), raw(&want.digests), "digests");
        assert_eq!(raw(&got.cap.0), raw(&want.cap.0), "cap");
    }
}
