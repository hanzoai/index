use std::collections::BTreeMap;

use crate::types::{AccessScope, PageKey, SearchHit};

/// Per-slice retrieval quality, latency, and security telemetry.
#[derive(Debug, Clone)]
pub struct SliceMetrics {
    pub query_count: usize,
    pub ndcg_at_5: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub mean_latency_ms: f64,
    pub acl_leaks: usize,
    pub cross_tenant_leaks: usize,
}

/// Compute NDCG@k for graded relevance labels.
pub fn compute_ndcg_at_k(hits: &[SearchHit], relevance: &BTreeMap<PageKey, u8>, k: usize) -> f64 {
    if relevance.is_empty() || k == 0 {
        return 0.0;
    }
    let mut dcg = 0.0f64;
    for (rank, hit) in hits.iter().take(k).enumerate() {
        let rel = f64::from(relevance.get(&hit.key()).copied().unwrap_or(0));
        if rel > 0.0 {
            let gain = 2.0f64.powf(rel) - 1.0;
            let discount = ((rank + 2) as f64).log2();
            dcg += gain / discount;
        }
    }

    let mut ideal_rels: Vec<u8> = relevance.values().copied().filter(|&r| r > 0).collect();
    ideal_rels.sort_unstable_by(|a, b| b.cmp(a));
    let mut idcg = 0.0f64;
    for (rank, rel_u8) in ideal_rels.into_iter().take(k).enumerate() {
        let gain = 2.0f64.powf(f64::from(rel_u8)) - 1.0;
        let discount = ((rank + 2) as f64).log2();
        idcg += gain / discount;
    }

    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Compute Recall@k: fraction of relevant ground-truth pages present in top-k hits.
pub fn compute_recall_at_k(hits: &[SearchHit], relevance: &BTreeMap<PageKey, u8>, k: usize) -> f64 {
    let total_rel = relevance.values().filter(|&&r| r > 0).count();
    if total_rel == 0 || k == 0 {
        return 0.0;
    }
    let found =
        hits.iter().take(k).filter(|h| relevance.get(&h.key()).copied().unwrap_or(0) > 0).count();
    found as f64 / total_rel as f64
}

/// Compute Mean Reciprocal Rank (MRR) for a single query.
pub fn compute_mrr(hits: &[SearchHit], relevance: &BTreeMap<PageKey, u8>) -> f64 {
    for (rank, hit) in hits.iter().enumerate() {
        if relevance.get(&hit.key()).copied().unwrap_or(0) > 0 {
            return 1.0 / ((rank + 1) as f64);
        }
    }
    0.0
}

/// Audit a single query's hits against the mandatory `AccessScope`.
/// Returns `(acl_leaks, cross_tenant_leaks)`.
pub fn audit_access_leaks(hits: &[SearchHit], scope: &AccessScope) -> (usize, usize) {
    let mut acl_leaks = 0usize;
    let mut cross_tenant_leaks = 0usize;
    for hit in hits {
        if hit.tenant_id != scope.tenant_id {
            cross_tenant_leaks += 1;
            acl_leaks += 1;
        } else if !scope.allowed_acl_groups.iter().any(|g| g == &hit.acl_group) {
            acl_leaks += 1;
        }
    }
    (acl_leaks, cross_tenant_leaks)
}

/// Nearest-rank percentile on a slice of latency samples (in milliseconds).
pub fn percentile_ms(samples_ms: &[f64], p: f64) -> f64 {
    if samples_ms.is_empty() {
        return 0.0;
    }
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * (sorted.len() as f64)).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Aggregate per-query scores and latencies into a [`SliceMetrics`] summary.
pub fn summarize_slice(
    ndcgs: &[f64],
    recalls: &[f64],
    mrrs: &[f64],
    latencies_ms: &[f64],
    acl_leaks: usize,
    cross_tenant_leaks: usize,
) -> SliceMetrics {
    let n = ndcgs.len();
    let denom = n.max(1) as f64;
    SliceMetrics {
        query_count: n,
        ndcg_at_5: ndcgs.iter().sum::<f64>() / denom,
        recall_at_5: recalls.iter().sum::<f64>() / denom,
        mrr: mrrs.iter().sum::<f64>() / denom,
        latency_p50_ms: percentile_ms(latencies_ms, 50.0),
        latency_p95_ms: percentile_ms(latencies_ms, 95.0),
        latency_p99_ms: percentile_ms(latencies_ms, 99.0),
        mean_latency_ms: latencies_ms.iter().sum::<f64>() / denom,
        acl_leaks,
        cross_tenant_leaks,
    }
}

/// Drop & Rebuild bit-exact parity receipt across canonical truth and all 4 derived indexes.
#[derive(Debug, Clone)]
pub struct DropRebuildParityReport {
    pub canonical_fingerprint_before: u64,
    pub canonical_fingerprint_after: u64,
    pub canonical_unchanged: bool,
    pub ocr_hybrid_checksum_exact: bool,
    pub ocr_hybrid_hits_bit_exact: bool,
    pub colpali_fp32_checksum_exact: bool,
    pub colpali_fp32_hits_bit_exact: bool,
    pub colpali_binary_checksum_exact: bool,
    pub colpali_binary_hits_bit_exact: bool,
    pub hybrid_router_checksum_exact: bool,
    pub hybrid_router_hits_bit_exact: bool,
}

impl DropRebuildParityReport {
    pub fn all_passed(&self) -> bool {
        self.canonical_unchanged
            && self.ocr_hybrid_checksum_exact
            && self.ocr_hybrid_hits_bit_exact
            && self.colpali_fp32_checksum_exact
            && self.colpali_fp32_hits_bit_exact
            && self.colpali_binary_checksum_exact
            && self.colpali_binary_hits_bit_exact
            && self.hybrid_router_checksum_exact
            && self.hybrid_router_hits_bit_exact
    }
}

/// Bit-exact comparison of two `SearchHit` lists (identity, version, ACL, route, and `f32::to_bits`).
pub fn hits_bit_exact(a: &[SearchHit], b: &[SearchHit]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.doc_id == y.doc_id
            && x.page_no == y.page_no
            && x.version == y.version
            && x.tenant_id == y.tenant_id
            && x.acl_group == y.acl_group
            && x.page_modality == y.page_modality
            && x.route_used == y.route_used
            && x.score.to_bits() == y.score.to_bits()
    })
}
