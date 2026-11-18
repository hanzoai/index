// SPDX-License-Identifier: Apache-2.0

//! HIP-1346 / [Bounty #21] (`hanzoai/index#25`): **Stable online ANN insertion** (OdinANN).
//!
//! Clean-room, deterministic, stdlib-only simulation harness that benchmarks
//! *direct online vector insertion into a file-backed on-disk ANN index under
//! continuous query traffic*, comparing three index-maintenance policies:
//!
//! - **(A) `MaintenancePolicy::BatchRebuild`** — stop-the-world baseline: inserts
//!   are buffered into an unindexed tail that queries must linear-scan; the whole
//!   graph is rebuilt from scratch every `rebuild_interval` inserts (stall time
//!   measured, never amortized away).
//! - **(B) `MaintenancePolicy::InPlaceOnline`** — naive online insertion: every
//!   insert enters the graph immediately, and no structural maintenance ever runs.
//! - **(C) `MaintenancePolicy::DeltaMerge`** — OdinANN-style bounded maintenance:
//!   inserts land in a bounded delta layer (linear-scanned by queries), absorbed
//!   into the main graph at a fixed per-query merge budget, with an emergency
//!   full-drain merge (stall) only when the delta overflows.
//!
//! # Immutable metadata enforcement (HIP-1346 §1)
//!
//! Every insert and every query passes through the existing
//! [`crate::metadata::IndexEmbeddingMetadata`] contract
//! (`validate_insert` / `validate_query`) **before any mutation**. A request whose
//! `(model_id, model_revision, dimensions, normalization, task_prefix)` disagrees
//! with the index's bound metadata is rejected with
//! [`AnnInsertError::Metadata`] and the index is left bit-for-bit unchanged — the
//! bounty's `task_prefix` maps onto the finer-grained existing
//! `query_prefix` / `doc_prefix` pair.
//!
//! # Honest boundaries (read before quoting any number from this module)
//!
//! - The ANN structure is a **single-layer greedy beam-search graph**, not a full
//!   HNSW/DiskANN reimplementation; figures measure *maintenance-policy behavior*,
//!   not upstream OdinANN numbers.
//! - Vectors are synthetic clustered unit-sphere embeddings generated from a
//!   fixed seed; ground truth (`recall@10`) is brute-force L2 over the exact
//!   present set at each query, precomputed once per workload and shared across
//!   all policy runs of that workload.
//! - "On-disk" is a real append-only vector page file with a **simulated,
//!   configurable page-read penalty** (unit tests run at zero penalty; the
//!   benchmark runner models a NVMe-like penalty). No physical NVMe hardware is
//!   exercised.
//! - The scheduler is single-threaded and order-invariant with respect to the
//!   seeded workload only; stall time is reported separately and is **not**
//!   amortized into query latency.
//! - Every figure produced here is a property of this simulation on the authoring
//!   host. **None of them is a Hanzo performance expectation** until reproduced
//!   on Hanzo hardware against the real on-disk ANN stack.
//!
//! In this simplified graph model, policy (B)'s long-run structural degradation
//! is deliberately under-modeled (each online insert uses the same greedy beam
//! construction as bulk build), so (B) is reported as a *latency floor
//! reference*; recall differences between policies are measured and reported as
//! observed rather than asserted.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::metadata::{IndexEmbeddingMetadata, MetadataError};

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64) — no external crates.
// ---------------------------------------------------------------------------

/// Seeded SplitMix64 generator; identical seeds produce identical workloads,
/// graph structures, and ground truth on every host.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator seeded with `seed` (0 is a valid seed).
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Produces the next 64-bit state value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform `usize` in `[0, bound)`.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Distance + vector helpers.
// ---------------------------------------------------------------------------

/// Squared L2 distance accumulated in `f64` and narrowed to `f32`; deterministic
/// regardless of compilation flags.
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2sq requires equal-length vectors");
    let mut sum: f64 = 0.0;
    for i in 0..a.len().min(b.len()) {
        let d = f64::from(a[i]) - f64::from(b[i]);
        sum += d * d;
    }
    sum as f32
}

/// L2-normalizes `v` in place (f64 accumulate); the all-but-zero vector is left
/// untouched rather than producing NaN.
fn normalize(v: &mut [f32]) {
    let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let norm = norm_sq.sqrt();
    if norm > f64::EPSILON {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// Candidate ordering (deterministic tie-break by distance, then id).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct Cand {
    dist: f32,
    id: u32,
}

impl Eq for Cand {}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// File-backed page store (the "on-disk" layer of the simulation).
// ---------------------------------------------------------------------------

/// Simulated IO cost model: physical page size and a per-page read penalty
/// applied to measured query latency. Set `page_penalty_ns = 0` for pure-CPU
/// timing (unit tests); the benchmark runner configures an NVMe-like penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoCostModel {
    /// Physical page size in bytes (4096 by default).
    pub page_size: usize,
    /// Simulated penalty per distinct page read, in nanoseconds.
    pub page_penalty_ns: u64,
}

impl Default for IoCostModel {
    fn default() -> Self {
        Self { page_size: 4096, page_penalty_ns: 0 }
    }
}

/// Append-only little-endian `f32` vector file with an in-memory mirror.
///
/// Writes go to the real file on every insert; `reload_from_disk` performs a
/// real read-back (used by policies (A)/(C) at rebuild/merge boundaries, exactly
/// the moments an on-disk ANN re-materializes its structure). Search-time reads
/// are served from the mirror while the *page accounting* still counts every
/// touched page, so the configured penalty reflects true access patterns.
pub struct VecStore {
    file: File,
    path: PathBuf,
    dim: usize,
    mirror: Vec<Vec<f32>>,
    io: IoCostModel,
    bytes_per_vec: usize,
    pages_written: u64,
    pages_read: u64,
    window_reads: u64,
}

impl VecStore {
    /// Creates (truncating) a vector file named `name` inside `dir`.
    pub fn create(dir: &Path, name: &str, dim: usize, io: IoCostModel) -> std::io::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{name}.vec"));
        let file = OpenOptions::new().create(true).read(true).write(true).open(&path)?;
        Ok(Self {
            file,
            path,
            dim,
            mirror: Vec::new(),
            io,
            bytes_per_vec: dim * 4,
            pages_written: 0,
            pages_read: 0,
            window_reads: 0,
        })
    }

    /// Path of the backing file (exposed for tests / diagnostics).
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn pages_spanning(&self, offset: u64, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }
        let ps = self.io.page_size as u64;
        ((offset + len - 1) / ps) - (offset / ps) + 1
    }

    /// Appends one vector at the next slot (insertion order == slot order).
    pub fn append(&mut self, v: &[f32]) -> std::io::Result<()> {
        debug_assert_eq!(v.len(), self.dim, "append requires the configured dimension");
        let offset = self.mirror.len() as u64 * self.bytes_per_vec as u64;
        let mut bytes = Vec::with_capacity(self.bytes_per_vec);
        for x in v {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&bytes)?;
        self.pages_written += self.pages_spanning(offset, bytes.len() as u64);
        self.mirror.push(v.to_vec());
        Ok(())
    }

    /// Real read-back of the whole file (used at rebuild / merge boundaries).
    pub fn reload_from_disk(&mut self) -> std::io::Result<()> {
        let mut buf = Vec::new();
        self.file.seek(SeekFrom::Start(0))?;
        self.file.read_to_end(&mut buf)?;
        self.pages_read += self.pages_spanning(0, buf.len() as u64);
        self.window_reads += self.pages_spanning(0, buf.len() as u64);
        self.mirror = buf
            .chunks_exact(self.bytes_per_vec)
            .map(|c| {
                c.chunks_exact(4)
                    .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                    .collect::<Vec<f32>>()
            })
            .collect();
        Ok(())
    }

    /// Number of vectors currently persisted.
    pub fn len(&self) -> usize {
        self.mirror.len()
    }

    /// Whether nothing has been persisted yet.
    pub fn is_empty(&self) -> bool {
        self.mirror.is_empty()
    }

    /// On-disk file length in bytes (equals `len() * bytes_per_vec`).
    pub fn file_bytes(&self) -> u64 {
        self.mirror.len() as u64 * self.bytes_per_vec as u64
    }

    /// Touches the page accounting for `id` and returns its vector.
    pub fn vec(&mut self, id: usize) -> &[f32] {
        let offset = id as u64 * self.bytes_per_vec as u64;
        let pages = self.pages_spanning(offset, self.bytes_per_vec as u64);
        self.pages_read += pages;
        self.window_reads += pages;
        &self.mirror[id]
    }

    /// Distance between `q` and stored vector `id`, with page accounting.
    pub fn dist(&mut self, q: &[f32], id: usize) -> f32 {
        l2sq(q, self.vec(id))
    }

    /// Total pages written since creation.
    pub fn pages_written(&self) -> u64 {
        self.pages_written
    }

    /// Total pages read since creation.
    pub fn pages_read(&self) -> u64 {
        self.pages_read
    }

    /// Per-page penalty from the cost model.
    pub fn penalty_ns(&self) -> u64 {
        self.io.page_penalty_ns
    }

    fn begin_window(&mut self) {
        self.window_reads = 0;
    }

    /// Pages touched in the current window and the penalty they imply.
    fn take_window(&mut self) -> u64 {
        let reads = self.window_reads;
        self.window_reads = 0;
        reads
    }
}

// ---------------------------------------------------------------------------
// Graph ANN (single-layer greedy beam search; deterministic).
// ---------------------------------------------------------------------------

/// Single-layer greedy beam-search ANN graph over the shared [`VecStore`].
///
/// Deliberately minimal but **navigable**: one layer, fixed entry point,
/// M-bounded neighbor lists, and **Vamana/DiskANN-style α-diversity robust
/// pruning** (clean-room from the published design) — without diversity
/// pruning, a greedy insertion-order graph loses its cross-cluster bridge
/// edges to keep-closest pruning and shatters into disconnected clusters,
/// which would corrupt any maintenance-policy comparison. It exists to expose
/// *maintenance-policy* costs, not to reproduce HNSW/Vamana quality curves.
pub struct GraphAnn {
    adj: Vec<Vec<u32>>,
    entry: Option<u32>,
    m_max: usize,
    ef_search: usize,
    ef_construction: usize,
    /// α-diversity factor for robust pruning (squared-L2 domain).
    alpha: f32,
}

impl GraphAnn {
    /// Creates an empty graph with the given beam/M parameters and the default
    /// α-diversity factor (1.2).
    pub fn new(m_max: usize, ef_search: usize, ef_construction: usize) -> Self {
        Self { adj: Vec::new(), entry: None, m_max, ef_search, ef_construction, alpha: 1.2 }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.adj.len()
    }

    fn beam(
        &self,
        store: &mut VecStore,
        q: &[f32],
        ef: usize,
    ) -> Result<Vec<Cand>, AnnInsertError> {
        let entry = self.entry.ok_or(AnnInsertError::EmptyIndex)?;
        let n = self.adj.len();
        let mut visited = vec![false; n];
        let mut cand: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        let mut best: BinaryHeap<Cand> = BinaryHeap::new(); // max-heap by (dist, id) → worst on top

        let d0 = store.dist(q, entry as usize);
        visited[entry as usize] = true;
        cand.push(Reverse(Cand { dist: d0, id: entry }));
        best.push(Cand { dist: d0, id: entry });

        while let Some(Reverse(c)) = cand.pop() {
            if best.len() >= ef && c.dist > best.peek().map(|w| w.dist).unwrap_or(c.dist) {
                break;
            }
            // Clone the neighbor list: `store` is borrowed mutably inside the loop.
            for nb in self.adj[c.id as usize].clone() {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let d = store.dist(q, nb as usize);
                let worse = best.peek().map(|w| w.dist).unwrap_or(f32::INFINITY);
                if best.len() < ef || d < worse {
                    cand.push(Reverse(Cand { dist: d, id: nb }));
                    best.push(Cand { dist: d, id: nb });
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }

        let mut out: Vec<Cand> = best.into_iter().collect();
        out.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    /// Top-`k` approximate neighbors of `q` (ids ascending tie-break).
    pub fn search(
        &self,
        store: &mut VecStore,
        q: &[f32],
        k: usize,
    ) -> Result<Vec<(u32, f32)>, AnnInsertError> {
        let cands = self.beam(store, q, self.ef_search.max(k))?;
        Ok(cands.into_iter().take(k).map(|c| (c.id, c.dist)).collect())
    }

    /// Vamana/DiskANN-style α-diversity selection over `candidates` already
    /// sorted ascending by distance to the query vector `qv`. A candidate is
    /// kept only when it is not redundant with any already-selected neighbor:
    /// `d(c, s) > α · d(c, qv)` for every selected `s`, with `d` the true
    /// (square-rooted) L2 distance — matching the published Vamana semantics.
    /// Cross-cluster bridge edges survive this prune; same-cluster duplicates
    /// do not.
    fn robust_select(&self, store: &mut VecStore, candidates: &[Cand], m: usize) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m.min(candidates.len()));
        for c in candidates {
            if selected.len() >= m {
                break;
            }
            let cv = store.vec(c.id as usize).to_vec();
            let keep = selected.iter().all(|&s| {
                let sv = store.vec(s as usize).to_vec();
                l2sq(&cv, &sv).sqrt() > self.alpha * c.dist.sqrt()
            });
            if keep {
                selected.push(c.id);
            }
        }
        selected
    }

    /// Links a freshly appended vector (slot `id`) into the graph.
    pub fn insert_node(&mut self, store: &mut VecStore, id: usize) {
        if self.entry.is_none() {
            self.adj.push(Vec::new());
            self.entry = Some(id as u32);
            return;
        }
        let v = store.vec(id).to_vec();
        let cands = match self.beam(store, &v, self.ef_construction) {
            Ok(c) => c,
            Err(_) => return, // graph cannot be empty here; defensive no-op
        };
        self.adj.push(Vec::new());
        let links = self.robust_select(store, &cands, self.m_max);
        for nb in links.clone() {
            self.adj[nb as usize].push(id as u32);
            if self.adj[nb as usize].len() > self.m_max {
                self.robust_prune_node(store, nb as usize);
            }
        }
        self.adj[id] = links;
    }

    /// α-diversity prune of `node`'s adjacency to `self.m_max` links.
    fn robust_prune_node(&mut self, store: &mut VecStore, node: usize) {
        let nv = store.vec(node).to_vec();
        let mut cands: Vec<Cand> = self.adj[node]
            .iter()
            .map(|&c| Cand { dist: store.dist(&nv, c as usize), id: c })
            .collect();
        cands.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
        self.adj[node] = self.robust_select(store, &cands, self.m_max);
    }

    /// Stop-the-world rebuild: clears the graph and bulk-inserts `ids` in order.
    pub fn rebuild(&mut self, store: &mut VecStore, ids: &[usize]) {
        self.adj.clear();
        self.entry = None;
        for &id in ids {
            self.insert_node(store, id);
        }
    }
}

// ---------------------------------------------------------------------------
// Maintenance policies + runner.
// ---------------------------------------------------------------------------

/// Index-maintenance policy under continuous insert traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenancePolicy {
    /// (A) Stop-the-world batch rebuild baseline.
    BatchRebuild {
        /// Rebuild the whole graph after this many buffered inserts.
        rebuild_interval: usize,
    },
    /// (B) Naive in-place online insertion, no maintenance afterwards.
    InPlaceOnline,
    /// (C) OdinANN-style bounded delta layer + amortized merge.
    DeltaMerge {
        /// Delta layer capacity; overflow triggers an emergency full merge.
        delta_capacity: usize,
        /// Delta nodes absorbed into the main graph per served query.
        merge_budget_per_query: usize,
    },
}

impl fmt::Display for MaintenancePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchRebuild { rebuild_interval } => {
                write!(f, "batch-rebuild(interval={rebuild_interval})")
            }
            Self::InPlaceOnline => write!(f, "in-place-online"),
            Self::DeltaMerge { delta_capacity, merge_budget_per_query } => {
                write!(f, "delta-merge(capacity={delta_capacity},budget={merge_budget_per_query})")
            }
        }
    }
}

/// Per-run counters and raw latency samples.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RunStats {
    /// Inserts accepted (metadata-validated) during the run.
    pub inserts_served: u64,
    /// Queries served during the run.
    pub queries_served: u64,
    /// Per-query wall latency in nanoseconds (IO penalty included).
    pub query_latencies_ns: Vec<u64>,
    /// Total wall time spent inside stop-the-world rebuild/merge stalls (ns).
    pub stall_ns: u64,
    /// Background (between-query) maintenance time for policy (C)'s amortized
    /// delta absorption — timed separately, never counted as query latency.
    pub maintenance_ns: u64,
    /// Number of full rebuilds performed.
    pub rebuilds: u64,
    /// Number of emergency (overflow) delta merges performed.
    pub emergency_merges: u64,
    /// Inserts rejected by the immutable metadata contract.
    pub rejected_inserts: u64,
    /// Queries rejected by the immutable metadata contract.
    pub rejected_queries: u64,
    /// Total insert-path wall time (ns), including inline graph insertion.
    pub insert_wall_ns: u64,
}

/// Errors surfaced by the harness.
#[derive(Debug, Error)]
pub enum AnnInsertError {
    /// Immutable metadata contract violation (HIP-1346 §1); the index is never
    /// mutated when this is returned.
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    /// Search on an index that contains no entry point.
    #[error("index is empty: no entry point for search")]
    EmptyIndex,
    /// Malformed configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Backing-file IO failure.
    #[error("vector store io failure: {0}")]
    StoreIo(#[from] std::io::Error),
}

/// Online-insertion runner: owns the immutable metadata gate, the main graph,
/// the page store, and the policy state (pending tail / delta queue).
pub struct AnnMaintenanceRunner {
    meta: IndexEmbeddingMetadata,
    policy: MaintenancePolicy,
    ann: GraphAnn,
    store: VecStore,
    pending: Vec<usize>,
    merge_queue: Vec<usize>,
    next_id: usize,
    stats: RunStats,
}

impl AnnMaintenanceRunner {
    /// Creates a runner bound to `meta`; every later insert/query must present
    /// byte-identical metadata or be rejected without mutation.
    pub fn new(
        meta: IndexEmbeddingMetadata,
        policy: MaintenancePolicy,
        storage_dir: &Path,
        io: IoCostModel,
    ) -> Result<Self, AnnInsertError> {
        meta.validate_definition()?;
        let store = VecStore::create(storage_dir, "vectors", meta.dimensions, io)?;
        Ok(Self {
            meta,
            policy,
            ann: GraphAnn::new(8, 48, 48),
            store,
            pending: Vec::new(),
            merge_queue: Vec::new(),
            next_id: 0,
            stats: RunStats::default(),
        })
    }

    /// Overrides graph tuning knobs (M / ef_search / ef_construction / α).
    pub fn set_graph_params(
        &mut self,
        m_max: usize,
        ef_search: usize,
        ef_construction: usize,
        alpha: f32,
    ) {
        self.ann = GraphAnn { adj: Vec::new(), entry: None, m_max, ef_search, ef_construction, alpha };
    }

    /// Bulk-loads the initial corpus (graph built immediately for all
    /// policies — the benchmark starts from a healthy pre-existing index).
    /// Returns the initial build wall time in nanoseconds.
    pub fn build_base(
        &mut self,
        runtime: &IndexEmbeddingMetadata,
        vectors: &[Vec<f32>],
    ) -> Result<u64, AnnInsertError> {
        let t = Instant::now();
        for v in vectors {
            self.meta.validate_insert(runtime, v, None).map_err(|e| {
                self.stats.rejected_inserts += 1;
                e
            })?;
            let id = self.next_id;
            self.store.append(v)?;
            self.next_id += 1;
            self.ann.insert_node(&mut self.store, id);
        }
        self.stats.inserts_served += vectors.len() as u64;
        // Every policy starts from a healthy pre-built index: no pending tail,
        // no delta debt.
        self.pending.clear();
        self.merge_queue.clear();
        Ok(t.elapsed().as_nanos() as u64)
    }

    /// Metadata-gated insert; **rejects before any mutation** on contract
    /// violation (HIP-1346 §1 isolation guarantee).
    pub fn insert(
        &mut self,
        runtime: &IndexEmbeddingMetadata,
        vector: &[f32],
    ) -> Result<(), AnnInsertError> {
        self.meta.validate_insert(runtime, vector, None).map_err(|e| {
            self.stats.rejected_inserts += 1;
            e
        })?;
        let t = Instant::now();
        let id = self.next_id;
        self.store.append(vector)?;
        self.next_id += 1;
        self.stats.inserts_served += 1;
        match self.policy {
            MaintenancePolicy::BatchRebuild { rebuild_interval } => {
                self.pending.push(id);
                if self.pending.len() >= rebuild_interval {
                    let stall = Instant::now();
                    self.store.reload_from_disk()?;
                    let all: Vec<usize> = (0..self.next_id).collect();
                    self.ann.rebuild(&mut self.store, &all);
                    self.stats.stall_ns += stall.elapsed().as_nanos() as u64;
                    self.stats.rebuilds += 1;
                    self.pending.clear();
                }
            }
            MaintenancePolicy::InPlaceOnline => {
                self.ann.insert_node(&mut self.store, id);
            }
            MaintenancePolicy::DeltaMerge { delta_capacity, .. } => {
                self.merge_queue.push(id);
                if self.merge_queue.len() >= delta_capacity {
                    let stall = Instant::now();
                    self.store.reload_from_disk()?;
                    let drained = std::mem::take(&mut self.merge_queue);
                    for nid in drained {
                        self.ann.insert_node(&mut self.store, nid);
                    }
                    self.stats.stall_ns += stall.elapsed().as_nanos() as u64;
                    self.stats.emergency_merges += 1;
                }
            }
        }
        self.stats.insert_wall_ns += t.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Metadata-gated query; **rejects before any mutation or latency
    /// accounting** on contract violation.
    pub fn search(
        &mut self,
        runtime: &IndexEmbeddingMetadata,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(u32, f32)>, AnnInsertError> {
        self.meta.validate_query(runtime, query, None).map_err(|e| {
            self.stats.rejected_queries += 1;
            e
        })?;
        self.stats.queries_served += 1;
        self.store.begin_window();
        let t = Instant::now();

        let mut graph_hits = self.ann.search(&mut self.store, query, k)?;
        match self.policy {
            MaintenancePolicy::BatchRebuild { .. } if !self.pending.is_empty() => {
                // Unindexed tail must stay searchable: linear scan of pending.
                let mut scan: Vec<Cand> = self
                    .pending
                    .iter()
                    .map(|&pid| Cand { dist: self.store.dist(query, pid), id: pid as u32 })
                    .collect();
                scan.extend(graph_hits.iter().map(|&(id, d)| Cand { dist: d, id }));
                scan.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
                scan.truncate(k);
                graph_hits = scan.into_iter().map(|c| (c.id, c.dist)).collect();
            }
            MaintenancePolicy::DeltaMerge { .. } => {
                if !self.merge_queue.is_empty() {
                    let mut scan: Vec<Cand> = self
                        .merge_queue
                        .iter()
                        .map(|&did| Cand { dist: self.store.dist(query, did), id: did as u32 })
                        .collect();
                    scan.extend(graph_hits.iter().map(|&(id, d)| Cand { dist: d, id }));
                    scan.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
                    scan.truncate(k);
                    graph_hits = scan.into_iter().map(|c| (c.id, c.dist)).collect();
                }
            }
            _ => {}
        }

        let window = self.store.take_window();
        let wall =
            t.elapsed().as_nanos() as u64 + window * self.store.penalty_ns();
        self.stats.query_latencies_ns.push(wall);

        // OdinANN's promise: merges never block queries. The amortized delta
        // absorption therefore runs *after* the query has been served (in the
        // gap between query arrivals in this single-threaded model) and is
        // timed separately as background maintenance, not as query latency.
        if let MaintenancePolicy::DeltaMerge { merge_budget_per_query, .. } = self.policy {
            let drain = merge_budget_per_query.min(self.merge_queue.len());
            if drain > 0 {
                let maintenance = Instant::now();
                for nid in self.merge_queue.drain(..drain) {
                    self.ann.insert_node(&mut self.store, nid);
                }
                self.stats.maintenance_ns += maintenance.elapsed().as_nanos() as u64;
            }
        }

        Ok(graph_hits)
    }

    /// Nodes currently in the main graph.
    pub fn node_count(&self) -> usize {
        self.ann.node_count()
    }

    /// Size of policy (A)'s unindexed tail.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Size of policy (C)'s delta queue (unabsorbed inserts).
    pub fn merge_queue_len(&self) -> usize {
        self.merge_queue.len()
    }

    /// Run statistics snapshot.
    pub fn stats(&self) -> &RunStats {
        &self.stats
    }

    /// Backing-file size in bytes (mutation-freeness assertions in tests).
    pub fn store_bytes(&self) -> u64 {
        self.store.file_bytes()
    }

    /// Total page writes/reads recorded by the store.
    pub fn io_pages(&self) -> (u64, u64) {
        (self.store.pages_written(), self.store.pages_read())
    }
}

// ---------------------------------------------------------------------------
// Workload generation + shared brute-force ground truth.
// ---------------------------------------------------------------------------

/// One scheduled operation in the interleaved workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    /// Insert the `stream_index`-th stream vector.
    Insert { stream_index: usize },
    /// Run the `query_index`-th traffic query.
    Query { query_index: usize },
}

/// Generated workload: initial corpus, online insert stream, query traffic,
/// the interleaved schedule, and the **shared brute-force ground truth**
/// (top-10 over the exact present set at each query, precomputed once and
/// reused by every policy run of the same workload).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    /// Initial corpus (bulk-loaded before the stream starts).
    pub base: Vec<Vec<f32>>,
    /// Online insert stream.
    pub inserts: Vec<Vec<f32>>,
    /// Query traffic vectors.
    pub queries: Vec<Vec<f32>>,
    /// Deterministic interleaved schedule.
    pub ops: Vec<Op>,
    /// `ground_truth[q]` = top-10 ids for the `q`-th query in schedule order,
    /// computed against the present set at that point in the schedule.
    pub ground_truth: Vec<Vec<u32>>,
}

fn gen_clustered(
    rng: &mut SplitMix64,
    n: usize,
    dim: usize,
    n_clusters: usize,
    centers: Option<&[Vec<f32>]>,
    noise: f32,
) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = match centers {
        Some(c) => c.to_vec(),
        None => (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
            .collect(),
    };
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let c = centers[rng.below(centers.len())].clone();
        let mut v: Vec<f32> = c
            .iter()
            .map(|&x| x + (rng.next_f32() * 2.0 - 1.0) * noise)
            .collect();
        normalize(&mut v);
        out.push(v);
    }
    out
}

fn bf_top10(q: &[f32], present: &[usize], base: &[Vec<f32>], ins: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for &gi in present {
        let v: &[f32] = if gi < base.len() { &base[gi] } else { &ins[gi - base.len()] };
        let d = l2sq(q, v);
        best.push((d, gi as u32));
        if best.len() > k * 4 {
            best.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            best.truncate(k);
        }
    }
    best.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    best.truncate(k);
    best.into_iter().map(|(_, id)| id).collect()
}

/// Generates the workload for `rate` (inserts per query) from `cfg`.
pub fn generate_workload(cfg: &BenchConfig, rate: f64) -> Result<Workload, AnnInsertError> {
    if cfg.dim == 0 || cfg.n_base == 0 || cfg.n_queries == 0 {
        return Err(AnnInsertError::InvalidConfig(
            "dim, n_base and n_queries must be positive".into(),
        ));
    }
    let mut rng = SplitMix64::new(cfg.seed);
    let centers: Vec<Vec<f32>> = (0..cfg.n_clusters)
        .map(|_| (0..cfg.dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect();
    let base = gen_clustered(&mut rng, cfg.n_base, cfg.dim, cfg.n_clusters, Some(&centers), 0.35);
    let n_inserts = (rate * cfg.n_queries as f64).round() as usize;
    let inserts = gen_clustered(&mut rng, n_inserts, cfg.dim, cfg.n_clusters, Some(&centers), 0.35);
    let queries = gen_clustered(&mut rng, cfg.n_queries, cfg.dim, cfg.n_clusters, Some(&centers), 0.15);

    let mut ops: Vec<Op> = Vec::with_capacity(n_inserts + cfg.n_queries);
    for i in 0..n_inserts {
        ops.push(Op::Insert { stream_index: i });
    }
    for i in 0..cfg.n_queries {
        ops.push(Op::Query { query_index: i });
    }
    // Deterministic Fisher–Yates shuffle of the schedule.
    for i in (1..ops.len()).rev() {
        let j = rng.below(i + 1);
        ops.swap(i, j);
    }

    // Shared ground truth: brute-force top-10 over the exact present set at
    // each query, in schedule order. Computed once, reused by all policies.
    let mut present: Vec<usize> = (0..cfg.n_base).collect();
    let mut ground_truth: Vec<Vec<u32>> = Vec::with_capacity(cfg.n_queries);
    for op in &ops {
        match *op {
            Op::Insert { stream_index } => present.push(cfg.n_base + stream_index),
            Op::Query { query_index } => {
                ground_truth.push(bf_top10(
                    &queries[query_index],
                    &present,
                    &base,
                    &inserts,
                    10,
                ));
            }
        }
    }

    Ok(Workload { base, inserts, queries, ops, ground_truth })
}

// ---------------------------------------------------------------------------
// Benchmark driver + results schema.
// ---------------------------------------------------------------------------

/// Benchmark configuration (deterministic; every field is part of the results).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    /// Seed for corpus + schedule generation.
    pub seed: u64,
    /// Vector width (positive; the runner default is a Matryoshka width).
    pub dim: usize,
    /// Initial bulk-loaded corpus size.
    pub n_base: usize,
    /// Traffic queries per (policy, rate) run.
    pub n_queries: usize,
    /// Insert rates to sweep, in inserts-per-query.
    pub rates: Vec<f64>,
    /// Graph neighbor cap.
    pub m_max: usize,
    /// Search beam width.
    pub ef_search: usize,
    /// Construction beam width.
    pub ef_construction: usize,
    /// α-diversity factor for Vamana-style robust pruning.
    pub alpha: f32,
    /// Policy (A) rebuild interval.
    pub rebuild_interval: usize,
    /// Policy (C) delta capacity.
    pub delta_capacity: usize,
    /// Policy (C) per-query merge budget.
    pub merge_budget_per_query: usize,
    /// Simulated page size in bytes.
    pub page_size: usize,
    /// Simulated per-page read penalty in nanoseconds.
    pub io_penalty_ns: u64,
    /// Top-k for recall (bounty fixes k=10).
    pub top_k: usize,
    /// Cluster count for synthetic data.
    pub n_clusters: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            seed: 0x21,
            dim: 128,
            n_base: 3_000,
            n_queries: 400,
            rates: vec![0.25, 0.5, 1.0, 2.0, 4.0],
            m_max: 16,
            ef_search: 64,
            ef_construction: 96,
            alpha: 1.2,
            rebuild_interval: 128,
            delta_capacity: 256,
            merge_budget_per_query: 8,
            page_size: 4096,
            io_penalty_ns: 50_000,
            top_k: 10,
            n_clusters: 32,
        }
    }
}

impl BenchConfig {
    /// Small deterministic configuration for unit/integration tests.
    pub fn test_config() -> Self {
        Self {
            seed: 7,
            dim: 16,
            n_base: 128,
            n_queries: 24,
            rates: vec![1.0],
            m_max: 6,
            ef_search: 24,
            ef_construction: 24,
            alpha: 1.2,
            rebuild_interval: 16,
            delta_capacity: 8,
            merge_budget_per_query: 4,
            page_size: 512,
            io_penalty_ns: 0,
            top_k: 10,
            n_clusters: 6,
        }
    }
}

/// One measured (policy, rate) run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Policy label.
    pub policy: String,
    /// Insert rate (inserts per query).
    pub rate: f64,
    /// Mean `recall@10` over all traffic queries (vs shared brute-force truth).
    pub recall_mean: f32,
    /// Worst single-query `recall@10`.
    pub recall_min: f32,
    /// Median query latency (ns, IO penalty included).
    pub p50_query_ns: u64,
    /// p99 query latency (ns, IO penalty included).
    pub p99_query_ns: u64,
    /// Mean query latency (ns).
    pub mean_query_ns: u64,
    /// Total stop-the-world stall time (ns), reported separately.
    pub stall_ns: u64,
    /// Background (between-query) delta-absorption maintenance time (ns).
    pub maintenance_ns: u64,
    /// Full rebuilds performed.
    pub rebuilds: u64,
    /// Emergency (overflow) merges performed.
    pub emergency_merges: u64,
    /// Mean insert wall time (ns).
    pub mean_insert_ns: u64,
    /// Page writes recorded by the backing store.
    pub page_writes: u64,
    /// Page reads recorded by the backing store.
    pub page_reads: u64,
    /// Initial bulk-build wall time (ns).
    pub initial_build_ns: u64,
    /// Inserts accepted.
    pub inserts: u64,
    /// Queries served.
    pub queries: u64,
}

fn percentile(samples: &[u64], p: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    let n = s.len();
    let idx = ((p * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    s[idx]
}

/// Falsifiable-hypothesis verdicts, computed from measured runs (never asserted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateVerdict {
    /// Insert rate of this row.
    pub rate: f64,
    /// (C) p99 within `c_p99_factor_of_b` × (B) p99?
    pub c_p99_ok_vs_b: bool,
    /// (C) performed zero emergency merges (no stall) at this rate?
    pub c_no_emergency_stall: bool,
    /// (A) stall time grows with rate relative to the previous lower rate?
    pub a_stall_increases: bool,
    /// (C) mean recall within `recall_epsilon` of (A)?
    pub c_recall_ok_vs_a: bool,
    /// Measured (A) stall time at this rate (ns).
    pub a_stall_ns: u64,
    /// Measured (C) p99 (ns).
    pub c_p99_ns: u64,
    /// Measured (B) p99 (ns) — the latency-floor reference.
    pub b_p99_ns: u64,
}

/// Full report emitted by [`run_full_matrix`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    /// Bounty this report answers.
    pub bounty: String,
    /// Umbrella HIP.
    pub hip: String,
    /// Host boundary note.
    pub host_note: String,
    /// Honest-boundary labels (quote these, not just the numbers).
    pub honesty_labels: Vec<String>,
    /// Configuration used.
    pub config: BenchConfig,
    /// All measured runs.
    pub runs: Vec<RunResult>,
    /// Measured hypothesis verdicts per rate.
    pub hypothesis_verdicts: Vec<RateVerdict>,
}

/// Runs the full (policy × rate) matrix: for each rate the workload (and its
/// shared brute-force ground truth) is generated **once** and every policy
/// replays the identical schedule over a fresh index.
pub fn run_full_matrix(cfg: &BenchConfig) -> Result<BenchReport, AnnInsertError> {
    let runtime_meta = IndexEmbeddingMetadata::new(
        "hanzo/zen-embed-v1",
        "sha256:b21odinann0000000000000000000000000000000000000000000000000000",
        cfg.dim,
        crate::metadata::NormalizationPolicy::L2,
        "query: ",
        "passage: ",
    )?;
    let io = IoCostModel { page_size: cfg.page_size, page_penalty_ns: cfg.io_penalty_ns };

    let mut runs: Vec<RunResult> = Vec::new();
    let mut prev_a_stall: Option<u64> = None;
    let mut verdicts: Vec<RateVerdict> = Vec::new();

    for &rate in &cfg.rates {
        let wl = generate_workload(cfg, rate)?;
        let mut per_policy: Vec<(String, RunResult)> = Vec::new();

        for policy in [
            MaintenancePolicy::BatchRebuild { rebuild_interval: cfg.rebuild_interval },
            MaintenancePolicy::InPlaceOnline,
            MaintenancePolicy::DeltaMerge {
                delta_capacity: cfg.delta_capacity,
                merge_budget_per_query: cfg.merge_budget_per_query,
            },
        ] {
            let tag = format!("{}-r{}", policy.to_string().replace(['(', ')', '=', ',', ' '], "-"), rate);
            let dir = std::env::temp_dir().join(format!("anninsert-{}-{}", cfg.seed, tag));
            let mut runner = AnnMaintenanceRunner::new(runtime_meta.clone(), policy, &dir, io)?;
            runner.set_graph_params(cfg.m_max, cfg.ef_search, cfg.ef_construction, cfg.alpha);
            let build_ns = runner.build_base(&runtime_meta, &wl.base)?;

            let mut recall_samples: Vec<f32> = Vec::with_capacity(cfg.n_queries);
            for op in &wl.ops {
                match *op {
                    Op::Insert { stream_index } => {
                        runner.insert(&runtime_meta, &wl.inserts[stream_index])?;
                    }
                    Op::Query { query_index } => {
                        let q = recall_samples.len();
                        let hits = runner.search(&runtime_meta, &wl.queries[query_index], cfg.top_k)?;
                        let truth = &wl.ground_truth[q];
                        let hits_count = hits
                            .iter()
                            .filter(|(id, _)| truth.contains(id))
                            .count();
                        recall_samples.push(hits_count as f32 / truth.len().max(1) as f32);
                    }
                }
            }

            let stats = runner.stats().clone();
            let lats = &stats.query_latencies_ns;
            let mean = if lats.is_empty() {
                0
            } else {
                lats.iter().sum::<u64>() / lats.len() as u64
            };
            let mean_recall =
                recall_samples.iter().sum::<f32>() / recall_samples.len().max(1) as f32;
            let min_recall = recall_samples.iter().copied().fold(f32::INFINITY, f32::min);
            let (pw, pr) = runner.io_pages();
            let result = RunResult {
                policy: policy.to_string(),
                rate,
                recall_mean: mean_recall,
                recall_min: if recall_samples.is_empty() { 0.0 } else { min_recall },
                p50_query_ns: percentile(lats, 0.50),
                p99_query_ns: percentile(lats, 0.99),
                mean_query_ns: mean,
                stall_ns: stats.stall_ns,
                maintenance_ns: stats.maintenance_ns,
                rebuilds: stats.rebuilds,
                emergency_merges: stats.emergency_merges,
                mean_insert_ns: if stats.inserts_served > 0 {
                    stats.insert_wall_ns / stats.inserts_served
                } else {
                    0
                },
                page_writes: pw,
                page_reads: pr,
                initial_build_ns: build_ns,
                inserts: stats.inserts_served,
                queries: stats.queries_served,
            };
            per_policy.push((result.policy.clone(), result.clone()));
            runs.push(result);
            let _ = fs::remove_dir_all(&dir);
        }

        // Verdict computation from measured runs.
        let a = &per_policy[0].1;
        let b = &per_policy[1].1;
        let c = &per_policy[2].1;
        let c_factor_limit = 3.0;
        verdicts.push(RateVerdict {
            rate,
            c_p99_ok_vs_b: c.p99_query_ns <= (b.p99_query_ns as f64 * c_factor_limit) as u64,
            c_no_emergency_stall: c.emergency_merges == 0,
            a_stall_increases: prev_a_stall.map(|p| a.stall_ns > p).unwrap_or(true),
            c_recall_ok_vs_a: (c.recall_mean - a.recall_mean).abs() <= 0.05,
            a_stall_ns: a.stall_ns,
            c_p99_ns: c.p99_query_ns,
            b_p99_ns: b.p99_query_ns,
        });
        prev_a_stall = Some(a.stall_ns);
    }

    Ok(BenchReport {
        bounty: "hanzoai/index#25 [Bounty #21] Stable online ANN insertion (OdinANN)".into(),
        hip: "HIP-1346: Evidence-Grounded Retrieval and Decision Evaluation".into(),
        host_note: "single-host CPU timing + simulated page-read penalty; not Hanzo hardware"
            .into(),
        honesty_labels: vec![
            "single-layer greedy beam graph with Vamana-style α-diversity pruning, not full HNSW/DiskANN".into(),
            "synthetic clustered unit-sphere vectors, fixed seed".into(),
            "recall@10 via brute-force L2 over exact present set at each query".into(),
            "on-disk = real append-only page file + simulated configurable page penalty".into(),
            "single-threaded order-invariant scheduler; stall reported separately".into(),
            "policy (C) delta absorption runs between query arrivals (background); maintenance time tracked separately, never as query latency".into(),
            "all figures are simulation outputs, NOT Hanzo expectations".into(),
        ],
        config: cfg.clone(),
        runs,
        hypothesis_verdicts: verdicts,
    })
}

/// Serializes `report` to pretty JSON at `path` (parent dirs created).
pub fn write_results_json(report: &BenchReport, path: &Path) -> Result<(), AnnInsertError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| AnnInsertError::InvalidConfig(format!("serialize results: {e}")))?;
    fs::write(path, json)?;
    Ok(())
}
