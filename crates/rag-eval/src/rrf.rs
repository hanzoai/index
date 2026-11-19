// SPDX-License-Identifier: Apache-2.0

//! Reciprocal Rank Fusion (RRF) and pre-filtered hybrid search evaluation
//! for `hanzoai/index`.

use std::collections::{HashMap, HashSet};

/// A ranked candidate document returned by a single retrieval modality
/// (e.g., dense vector search or sparse BM25).
#[derive(Debug, Clone, PartialEq)]
pub struct RankedDoc {
    pub doc_id: String,
    pub rank: usize,
}

/// A fused document with its combined RRF score.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedDoc {
    pub doc_id: String,
    pub score: f64,
}

/// Configuration for Reciprocal Rank Fusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrfConfig {
    /// Smoothing constant `k` (standard default: 60.0).
    pub k: f64,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

/// Computes Reciprocal Rank Fusion across multiple ranked lists, optionally
/// restricting candidates to an allowed pre-filter set of document IDs.
pub fn fuse_rrf(
    ranked_lists: &[Vec<RankedDoc>],
    config: RrfConfig,
    allowed_ids: Option<&HashSet<String>>,
) -> Vec<FusedDoc> {
    let k = if config.k > 0.0 { config.k } else { 60.0 };
    let mut scores: HashMap<String, f64> = HashMap::new();

    for list in ranked_lists {
        for item in list {
            if let Some(allowed) = allowed_ids {
                if !allowed.contains(&item.doc_id) {
                    continue;
                }
            }
            let contribution = 1.0 / (k + (item.rank as f64));
            *scores.entry(item.doc_id.clone()).or_insert(0.0) += contribution;
        }
    }

    let mut fused: Vec<FusedDoc> = scores
        .into_iter()
        .map(|(doc_id, score)| FusedDoc { doc_id, score })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    fused
}

/// Runs a `k`-value ablation sweep and returns the top-1 score for each `k`.
pub fn rrf_k_ablation(
    ranked_lists: &[Vec<RankedDoc>],
    k_values: &[f64],
) -> Vec<(f64, Option<FusedDoc>)> {
    k_values
        .iter()
        .map(|&k| {
            let fused = fuse_rrf(ranked_lists, RrfConfig { k }, None);
            (k, fused.into_iter().next())
        })
        .collect()
}