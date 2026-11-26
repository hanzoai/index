#![forbid(unsafe_code)]

//! Clean-room ColPali late-interaction visual document retrieval benchmark and derived index
//! (`hanzoai/index#23`, `HIP-1347`).
//!
//! # Architecture Invariants
//! - **Zero external crate dependencies**: pure Rust `std`, `Apache-2.0 OR MIT`, `#![forbid(unsafe_code)]`.
//! - **Strict canonical/derived separation**: [`CanonicalPageStore`] holds authoritative page records
//!   with `(doc_id, page_no, version, tenant_id, acl_group, page_modality)`. All 4 retrieval engines
//!   are strictly derived views with bit-exact Drop & Rebuild parity and CRC32 integrity checks.
//! - **Fail-closed pre-return tenant/ACL isolation**: empty tenant IDs and empty ACL group lists fail
//!   closed before touching index data; unauthorized pages are never scored into top-k or returned.
//! - **Real architectural tradeoffs**: compares [`OcrBm25HybridIndex`], [`ColPaliFp32Index`],
//!   [`ColPaliBinaryTwoStageIndex`] (1-bit POPCNT stage-1 + SQ8 stage-2 rerank), and
//!   [`HybridVisualRouterIndex`] (selective visual indexing & routing) across all 4 page modalities
//!   plus explicit Negative Regimes 1 (exact keywords) and 2 (100k–1M page storage explosion).

pub mod benchmark;
pub mod canonical;
pub mod colpali_index;
pub mod corpus;
pub mod math;
pub mod metrics;
pub mod ocr_hybrid;
pub mod router;
pub mod scaling;
pub mod types;

pub use benchmark::{report_to_json, run_benchmark, BenchmarkConfig, BenchmarkReport};
pub use canonical::CanonicalPageStore;
pub use colpali_index::{ColPaliBinaryTwoStageIndex, ColPaliFp32Index};
pub use ocr_hybrid::OcrBm25HybridIndex;
pub use router::HybridVisualRouterIndex;
pub use types::{
    AccessScope, CanonicalPageRecord, ColPaliError, IndexFootprint, PageKey, PageLayoutFeatures,
    PageModality, RouteKind, SearchHit, StrategyId, VisualQuery, DEFAULT_EMBEDDING_DIM,
};
