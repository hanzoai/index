// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # `vector-ann-bench` (`HIP-1347` / `hanzoai/index#15`)
//!
//! Zero-dependency, pure-Rust `std` clean-room benchmark harness evaluating
//! **ANN and object-storage-native vector index architectures** (DiskANN,
//! USearch, and LanceDB as references) under Hanzo Index's strict architectural
//! invariants:
//!
//! 1. **Canonical/Derived Separation**: [`CanonicalStore`] owns authoritative
//!    `(source_id, version, tenant_id, acl_group, filter_tags, vector)` records.
//!    Every derived ANN index can be dropped and rebuilt bit-exact from canonical
//!    source truth without mutating or losing canonical identity.
//! 2. **Fail-Closed Pre-Return Tenant/ACL Isolation**: Every index enforces
//!    tenant and ACL checks before returning any hit (`0.0%` cross-tenant/ACL leaks).
//! 3. **Four Matched Reference Architectures**:
//!    - [`ExactFlatBaseline`] (`exact_flat_baseline`): 100% Recall@10 ground-truth
//!      oracle (+ [`PrefilterInvertedFallback`] for ultra-selective `<= 1%` filters).
//!    - [`LocalMmapVamanaIndex`] (`local_mmap_vamana`): DiskANN/USearch-style
//!      compact disk-backed Vamana graph + SQ8 slab with 4KB paged cold/warm I/O,
//!      filtered search, tombstones, fresh delta overlay, and compaction.
//!    - [`ObjectNativeIvfPqIndex`] (`object_native_ivf_pq`): LanceDB/Hanzo S3/VFS-style
//!      immutable coarse-partitioned IVF-PQ + SQ8 segments with hot compute-local
//!      routing cache executing **1 bounded parallel batch of contiguous `RangeGET`s**.
//!    - [`ObjectStoreRandomGraphIndex`] (`object_store_random_graph_hops`):
//!      Explicit negative control proving why object storage MUST NOT serve
//!      random pointer-chasing graph hops.

pub mod bench_runner;
pub mod canonical;
pub mod corpus;
pub mod exact_flat;
pub mod json;
pub mod local_mmap_vamana;
pub mod metrics;
pub mod object_graph_negative_control;
pub mod object_native_ivf_pq;
pub mod object_store;
pub mod scaling;

pub use bench_runner::{
    run_full_benchmark, DownstreamTaskRow, FullBenchmarkReport, MutationLifecycleReport,
    RebuildParityReport, SliceMetricRow, UltraSelectiveRegimeRow,
};
pub use canonical::{
    cmp_search_hits, AccessContext, AnnError, CanonicalStore, CanonicalVectorRecord,
    DerivedRecordMeta, HitSecurityAudit, IndexFootprint, SearchHit,
};
pub use corpus::{build_benchmark_corpus, BenchmarkCorpus, BenchmarkQuery, CorpusConfig};
pub use exact_flat::{ExactFlatBaseline, PrefilterInvertedFallback};
pub use json::JsonVal;
pub use local_mmap_vamana::{
    CompactionReport, LocalMmapVamanaIndex, LocalPageIoStats, VamanaConfig,
};
pub use metrics::{
    crc32_ieee, l2_squared, normalize_l2, recall_at_k, summarize_latencies_ms,
    validate_query_vector, LatencySummaryMs, ProductQuantizer, SplitMix64, Sq8Quantizer,
};
pub use object_graph_negative_control::ObjectStoreRandomGraphIndex;
pub use object_native_ivf_pq::{IvfPqConfig, ObjectNativeIvfPqIndex};
pub use object_store::{
    ByteRangeRequest, InstrumentedObjectStore, ObjectStoreCostModel, ObjectStoreStats,
};
pub use scaling::{project_scale_matrix, ScaleProjectionRow};
