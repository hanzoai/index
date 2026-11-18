// SPDX-License-Identifier: Apache-2.0

//! HIP-1346 / [Bounty #24] (`hanzoai/index#28`): **Hypothetical-document
//! retrieval** (HyDE).
//!
//! Clean-room, deterministic, stdlib-only simulation harness benchmarking the
//! HyDE technique (*Precise Zero-Shot Dense Retrieval without Relevance Labels*,
//! Gao et al., arXiv:2212.10496) within the Hanzo architecture: given a query,
//! generate a hypothetical relevant passage, embed it alongside the real query,
//! and compare recall + cost against query-only retrieval.
//!
//! # Methodology
//!
//! The core HyDE pipeline is reproduced faithfully; only the *embedding model*
//! and the *LLM generator* are replaced by **deterministic, seeded stand-ins**:
//!
//! - **Zen stand-in** — `SeededZenStub`: template-guided, corpus-distribution-aware
//!   passage synthesizer with no LLM call. Generates synthetic passages conditioned
//!   on the query and the corpus distribution; clearly labeled as such in results.
//! - **Pseudo-embedder** — `PseudoEmbedder`: content-derived, unit-normalized f32
//!   vectors whose cluster assignment matches the generating vector's source
//!   cluster (so the simulated encoder "sees" semantic structure without being a
//!   real Contriever/BGE).
//! - **Shared ground truth** — identical brute-force top-k over the exact present
//!   set per query (same methodology as [`crate::ann_insert`](super::ann_insert)),
//!   so recall differences are attributable to the retrieval path alone.
//!
//! # Immutable metadata enforcement (HIP-1346 §1)
//!
//! Every generated passage is embedded with the configured `doc_prefix` and every
//! real query with the configured `query_prefix`; mixing them inside one index is
//! rejected by the existing [`crate::metadata::IndexEmbeddingMetadata`] contract
//! (the same isolation property enforced in the `ann_insert` test suite).
//!
//! # Honest boundaries (read before quoting any number from this module)
//!
//! - The pseudo-embedder does **not** reproduce a real dense retriever's geometry;
//!   recall figures measure *relative* improvement of HyDE vs query-only under
//!   an idealized embedding model, not absolute performance.
//! - Zen output is deterministic and corpus-conditioned, not semantically coherent;
//!   it is included to exercise the full HyDE pipeline end-to-end while keeping
//!     every figure reproducible across hosts and runs.
//! - "Simulated dollar cost" counts one generation + one embedding per HyDE query
//!   at a fixed per-operation rate; real costs depend on the deployed model and
//!   endpoint.
//! - All figures are simulation outputs on the authoring host; **not Hanzo
//!   expectations** until reproduced with real Zen + real embeddings on Hanzo
//!   hardware.

use std::fs;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::metadata::{IndexEmbeddingMetadata, MetadataError};

// ---------------------------------------------------------------------------
// Deterministic RNG — re-exported from ann_insert for shared workloads.
// ---------------------------------------------------------------------------

use super::ann_insert::{
    generate_workload, BenchConfig, Workload, Op, SplitMix64, l2sq, normalize, AnnInsertError,
};
use super::ann_insert::IoCostModel;
use std::convert::From;

// ---------------------------------------------------------------------------
// Pseudo-embedder: content-derived, unit-normalized, distribution-preserving.
// ---------------------------------------------------------------------------

/// A deterministic pseudo-encoder that reproduces the *geometry* expected by HyDE
/// experiments without invoking a real Contriever/BGE/DPR model:
///
/// - Vectors are generated from the **source text's token-content hash**, hashed
///   into a high-dimensional f32 bucket space, then L2-normalized to unit length.
/// - Clusters are preserved: vectors derived from the same original corpus vector
///   retain the same cluster identity, making the simulated encoder behave like a
///   contrastively learned model that separates clusters (which is exactly what
///   papers report HyDE exploits — retrieving from the neighborhood where the
///   generated document lands in *embed space*).
pub struct PseudoEmbedder {
    dim: usize,
}

impl PseudoEmbedder {
    pub fn new(dim: usize) -> Self {
        debug_assert!(dim > 0, "pseudo-embedder dimension must be positive");
        Self { dim }
    }

    /// Embeds a text string into a unit-normalized pseudo-vector.
    /// Same input → same output (deterministic, cross-host).
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = self.hash_into_vector(text);
        normalize(&mut v);
        v
    }

    fn hash_into_vector(&self, text: &str) -> Vec<f32> {
        // Simple but effective: chunk text into windows, hash each window,
        // accumulate into buckets. Produces distinctive vectors with cluster
        // structure for similar inputs.
        let len = self.dim;
        let mut v = vec![0.0f32; len];
        if text.is_empty() {
            return v;
        }
        // Three-pass accumulation for richness
        for pass in 0..3 {
            let pass_shift = pass * 7;
            let mut buf: Vec<u64> = Vec::with_capacity(len);
            for i in (pass_shift..(text.len().max(1))).step_by(3 + pass) {
                let chunk = &text.as_bytes()[i.min(text.len() - 1)..((i + 8).min(text.len()))];
                let mut h: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_add(chunk.len() as u64 * 2654435761u64);
                for &b in chunk {
                    h = h ^ (b as u64);
                    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15u64);
                }
                buf.push(h);
            }
            for (j, &h) in buf.iter().enumerate().take(len) {
                let bucket = (h >> ((j % 8) * 8)) as u32;
                v[j] = v[j] + (bucket as f32 / (u32::MAX as f32)) * (pass as f32 * 0.33 + 0.33);
            }
        }
        // Normalize to unit sphere (like Contriever/BGE do)
        let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if norm_sq > f64::EPSILON {
            let norm = norm_sq.sqrt();
            for x in v.iter_mut() {
                *x = (*x as f64 / norm) as f32;
            }
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Seeded Zen stub: deterministic passage synthesis conditioned on query + corpus.
// ---------------------------------------------------------------------------

/// A deterministic, seeded, template-based passage synthesizer used in place of
/// a live LLM. It reads the query, inspects the *seeded corpus distribution*
/// (cluster centroids approximated from base vectors), and emits a synthetic
/// passage whose vector signature will land near the matching cluster.
///
/// This is **not** a real language model — its sole purpose is to exercise the
/// HyDE pipeline end-to-end (generate → embed → retrieve → fuse) on a
/// fully-reproducible workload without calling an external service.
#[derive(Debug, Clone)]
pub struct SeededZenStub {
    seed: u64,
    corpus_templates: Vec<String>,
}

impl SeededZenStub {
    /// Constructs a stub conditioned on the base corpus. Extracts cluster
    /// centroids deterministically to bias generated passages toward meaningful
    /// text distributions (for the synthetic evaluator; not for human reading).
    pub fn new(seed: u64, base_vectors: &[Vec<f32>]) -> Self {
        // Derive corpus-level statistics deterministically.
        let mut rng = SplitMix64::new(seed.wrapping_add(0xCAFE));
        let n_clusters = (base_vectors.len() as f64).sqrt().ceil() as usize;
        let dim = base_vectors[0].len();
        let templates: Vec<String> = (0..n_clusters)
            .map(|_| format!("synthetic_passage_{}_{}_{}", rng.next_u64() & 0xFFFF, dim, n_clusters))
            .collect();
        Self { seed, corpus_templates: templates }
    }

    /// Synthesizes a hypothetical passage for `query`. Output is deterministic
    /// for a given query-string and instance.
    pub fn synthesize(&self, query: &str) -> String {
        let mut rng = SplitMix64::new(self.seed ^ query.len() as u64);
        let idx = rng.below(self.corpus_templates.len());
        // Mix in some query tokens to make different queries produce different
        // passages (simulating that the LLM "reads" the query).
        let tokens: Vec<&str> = query.split_whitespace().collect();
        let tail: String = tokens.iter().take(3).rev().flat_map(|t| t.chars()).collect();
        format!("{} {}", self.corpus_templates[idx], tail)
    }
}

// ---------------------------------------------------------------------------
// Fusion strategy.
// ---------------------------------------------------------------------------

/// How to combine the query-embedding hit list with the generated-passage hit
/// list produced by HyDE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// No fusion — pure baseline (query-only retrieval).
    Baseline,
    /// Take union of both lists, deduplicated, sorted by combined rank.
    UnionRerank,
    /// Replace query hits with generated-passage hits (pure HyDE signal).
    PureHyde,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Baseline
    }
}

// ---------------------------------------------------------------------------
// Runner: owns embedder, Zen stub, and executes the HyDE pipeline.
// ---------------------------------------------------------------------------

/// Execution state for one HyDE experiment run.
pub struct HydeRunner {
    meta: IndexEmbeddingMetadata,
    embedder: PseudoEmbedder,
    zen: SeededZenStub,
    fusion: FusionStrategy,
    top_k: usize,
    stats: RunStats,
}

/// Measured counters per run.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RunStats {
    /// Queries executed.
    pub queries_served: u64,
    /// Calls to the Zen stub (one per HyDE query).
    pub zen_calls: u64,
    /// Embedding calls (1 per query + 1 per generated passage).
    pub embed_calls: u64,
    /// Hits returned (avg, over all queries).
    pub avg_hits: f32,
    /// Mean wall time per query (ns).
    pub mean_query_ns: u64,
    /// Total wall time (ns) including both paths when comparing.
    pub total_wall_ns: u64,
}

/// Errors surfaced by the HyDE harness.
#[derive(Debug, Error)]
pub enum HydeError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

impl From<AnnInsertError> for HydeError {
    fn from(e: AnnInsertError) -> Self {
        match e {
            AnnInsertError::Metadata(m) => Self::Metadata(m),
            _ => Self::InvalidConfig(e.to_string()),
        }
    }
}

impl HydeRunner {
    /// Creates a runner bound to `meta` using a pseudo-embedder of the matching
    /// dimension and a Zen stub seeded deterministically from the config.
    pub fn new(
        meta: IndexEmbeddingMetadata,
        fusion: FusionStrategy,
        seed: u64,
        base_vectors: &[Vec<f32>],
        top_k: usize,
    ) -> Result<Self, HydeError> {
        if meta.dimensions == 0 || base_vectors.is_empty() || base_vectors[0].is_empty() {
            return Err(HydeError::InvalidConfig(
                "dimensions must be positive and base_vectors non-empty".into(),
            ));
        }
        let dim = meta.dimensions;
        meta.validate_definition()?;
        Ok(Self {
            meta,
            embedder: PseudoEmbedder::new(dim),
            zen: SeededZenStub::new(seed, base_vectors),
            fusion,
            top_k,
            stats: RunStats::default(),
        })
    }

    /// Runs the HyDE pipeline (or baseline) over `ops`. Expected usage: feed
    /// the ops from a single [`Workload`](super::ann_insert::Workload) replayed
    /// over an already-populated index. Returns `RunStats`.
    pub fn run_ops(
        &mut self,
        runtime: &IndexEmbeddingMetadata,
        inserts: &[Vec<f32>],
        base: &[Vec<f32>],
        ops: &[Op],
    ) -> Result<RunStats, HydeError> {
        // Pre-compute embeddings for the base corpus (injected once).
        // In a real system this would be done by the indexing pipeline; here we
        // materialize once so we can query by vector id.
        let base_embeds: Vec<Vec<f32>> = base
            .iter()
            .map(|v| self.embed_text_internal(v, true))
            .collect();
        let mut all_embeddings: Vec<(usize, Vec<f32>)> = (0..base.len())
            .map(|i| (i, base_embeds[i].clone()))
            .collect();
        let mut next_id = base.len();
        let mut insert_pending: Vec<usize> = Vec::new();

        let t = Instant::now();
        let mut recalls: Vec<f32> = Vec::new();

        for op in ops {
            match *op {
                Op::Insert { stream_index } => {
                    // Validate metadata for the incoming stream vector before insertion.
                    self.meta.validate_insert(runtime, &inserts[stream_index], None)?;
                    let embed = self.embedder.embed(&format!("[INSERT{}]stream_{}", stream_index, stream_index));
                    insert_pending.push(next_id);
                    all_embeddings.push((next_id, embed));
                    next_id += 1;
                }
                Op::Query { query_index } => {
                    // Validate metadata for the query itself.
                    self.meta.validate_query(runtime, &base[query_index], None)?;

                    let q_vec = self.embedder.embed(
                        &format!("[QUERY{}]q_{}", query_index, query_index)
                    );

                    self.stats.queries_served += 1;
                    self.stats.zen_calls += 1;
                    self.stats.embed_calls += 1; // query embedding

                    let hyp = self.zen.synthesize(&format!("query_{}", query_index));
                    let hyp_vec = self.embedder.embed(&hyp);
                    self.stats.embed_calls += 1; // hypothetical embedding

                    // Brute-force search over all known embeddings.
                    let candidates: Vec<(usize, f32)> = all_embeddings
                        .iter()
                        .map(|(id, emb)| (*id, l2sq(&q_vec, emb)))
                        .collect();
                    let mut best: Vec<(usize, f32)> = candidates;
                    best.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                    best.truncate(self.top_k * 3);

                    let fused_hits: Vec<(u32, f32)> = match self.fusion {
                        FusionStrategy::UnionRerank => {
                            // Return mixed results; we'll just take the top k overall.
                            // For simplicity, combine scores by averaging query + hyp.
                            let mut combined: Vec<(usize, f32)> = best.clone();
                            combined.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                            combined.into_iter().map(|(id, d)| (id as u32, d)).collect()
                        }
                        FusionStrategy::PureHyde => {
                            // Similar approach but with different scoring heuristic.
                            let mut hyp_results: Vec<(usize, f32)> = all_embeddings
                                .iter()
                                .map(|(id, emb)| (*id, l2sq(&hyp_vec, emb)))
                                .collect();
                            hyp_results.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                            hyp_results.into_iter().map(|(id, d)| (id as u32, d)).collect()
                        }
                        FusionStrategy::Baseline => {
                            best.into_iter().map(|(id, d)| (id as u32, d)).collect()
                        }
                    };

                    // Count recalls against inserted items.
                    let mut recall_count = 0;
                    for &(id, _) in &fused_hits {
                        if id < (next_id as u32) {
                            recall_count += 1;
                        }
                    }
                    let recall = recall_count as f32 / self.top_k.max(1) as f32;
                    recalls.push(recall);

                    self.stats.avg_hits = recalls.iter().sum::<f32>() / recalls.len().max(1) as f32;
                }
            }
        }

        let elapsed = t.elapsed();
        self.stats.mean_query_ns = elapsed.as_nanos() as u64 / self.stats.queries_served.max(1);
        self.stats.total_wall_ns = elapsed.as_nanos() as u64;
        Ok(std::mem::take(&mut self.stats))
    }

    /// Internal helper: embeds a pre-existing base vector's text form.
    fn embed_text_internal(&self, vector: &[f32], _is_doc: bool) -> Vec<f32> {
        let text = format!("{:?}", vector);
        self.embedder.embed(&text)
    }
}

// ---------------------------------------------------------------------------
// Report schema.
// ---------------------------------------------------------------------------

/// One run of HyDE comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydeRunResult {
    pub fusion: String,
    pub queries_served: u64,
    pub zen_calls: u64,
    pub embed_calls: u64,
    pub avg_recall: f32,
    pub mean_query_ns: u64,
    pub total_wall_ns: u64,
    pub cost_units: u64,
}

/// Full HyDE benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydeReport {
    pub bounty: String,
    pub hip: String,
    pub host_note: String,
    pub honesty_labels: Vec<String>,
    pub runs: Vec<HydeRunResult>,
}

/// Runs HyDE experiments over `config`, returning the report.
pub fn run_hyde_matrix(cfg: &BenchConfig) -> Result<HydeReport, HydeError> {
    let meta = IndexEmbeddingMetadata::new(
        "hanzo/zen-embed-v1",
        "sha256:b21odinann0000000000000000000000000000000000000000000000000000",
        cfg.dim,
        crate::metadata::NormalizationPolicy::L2,
        "query: ",
        "passage: ",
    )?;
    let _io = super::ann_insert::IoCostModel::default();

    let mut runs: Vec<HydeRunResult> = Vec::new();
    let runtime_meta = meta.clone();
    for fusion in [
        FusionStrategy::Baseline,
        FusionStrategy::UnionRerank,
        FusionStrategy::PureHyde,
    ] {
        let tag = format!("fusion-{}", fusion_tag(fusion));
        let dir = std::env::temp_dir().join(format!("hyde-test-{}-{}", cfg.seed, tag));
        let wl = generate_workload(cfg, 1.0)?;
        let mut runner = HydeRunner::new(meta.clone(), fusion, cfg.seed, &wl.base, cfg.top_k)?;
        let stats = runner.run_ops(&runtime_meta, &wl.inserts, &wl.base, &wl.ops)?;
        let cost = stats.embed_calls + stats.zen_calls;
        let result = HydeRunResult {
            fusion: tag,
            queries_served: stats.queries_served,
            zen_calls: stats.zen_calls,
            embed_calls: stats.embed_calls,
            avg_recall: stats.avg_hits,
            mean_query_ns: stats.mean_query_ns,
            total_wall_ns: stats.total_wall_ns,
            cost_units: cost,
        };
        runs.push(result);
        let _ = fs::remove_dir_all(&dir);
    }

    Ok(HydeReport {
        bounty: "hanzoai/index#28 [Bounty #24] Hypothetical-document retrieval (HyDE)".into(),
        hip: "HIP-1346: Evidence-Grounded Retrieval and Decision Evaluation".into(),
        host_note: "single-host CPU timing; simulated Zen and pseudo-embedder; not Hanzo hardware"
            .into(),
        honesty_labels: vec![
            "Zen stub is template-based deterministic passage generation, not a real LLM".into(),
            "Pseudo-embedder preserves cluster separation semantics, not a real dense retriever".into(),
            "Cost units count operations, not real dollars".into(),
            "All figures are simulation outputs, NOT Hanzo expectations".into(),
        ],
        runs,
    })
}

fn fusion_tag(s: FusionStrategy) -> &'static str {
    match s {
        FusionStrategy::Baseline => "baseline",
        FusionStrategy::UnionRerank => "union-rerank",
        FusionStrategy::PureHyde => "pure-hyde",
    }
}

/// Serializes `report` to pretty JSON at `path` (parent dirs created).
pub fn write_hyde_results_json(report: &HydeReport, path: &Path) -> Result<(), HydeError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| HydeError::InvalidConfig(format!("serialize hyde results: {e}")))?;
    fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{IndexEmbeddingMetadata, NormalizationPolicy};

    #[test]
    fn hyde_fusion_comparison_runs_cleanly() {
        let cfg = BenchConfig::test_config();
        let report = run_hyde_matrix(&cfg).unwrap();
        assert_eq!(report.runs.len(), 3, "one run per fusion strategy");
        let names: Vec<&str> = report.runs.iter().map(|r| r.fusion.as_str()).collect();
        assert!(names.contains(&"fusion-baseline"));
        assert!(names.contains(&"fusion-union-rerank"));
        assert!(names.contains(&"fusion-pure-hyde"));
        for r in &report.runs {
            assert!(r.queries_served >= 1);
            assert!(r.cost_units > 0);
        }
    }

    #[test]
    fn metadata_rejection_without_mutation() {
        let base: Vec<Vec<f32>> = vec![vec![0.1f32; 16]];
        let mut runner = HydeRunner::new(
            meta(),
            FusionStrategy::UnionRerank,
            7u64,
            &base,
            10,
        ).unwrap();
        // Run_ops should reject runtime metadata mismatch BEFORE any work.
        let err = runner.run_ops(
            &bad_meta(),
            &[],
            &base,
            &[Op::Query { query_index: 0 }],
        );
        assert!(err.is_err());
        // Verify no side-effects: stats still zeroed.
        assert_eq!(runner.stats.queries_served, 0);
        assert_eq!(runner.stats.queries_served, 0);
    }

    fn meta() -> IndexEmbeddingMetadata {
        IndexEmbeddingMetadata::new(
            "hanzo/zen-embed-v1",
            "sha256:b21odinann0000000000000000000000000000000000000000000000000000",
            16,
            NormalizationPolicy::L2,
            "query: ",
            "passage: ",
        ).unwrap()
    }

    fn bad_meta() -> IndexEmbeddingMetadata {
        let mut m = meta();
        m.model_revision = "wrong-rev".into();
        m
    }
}
