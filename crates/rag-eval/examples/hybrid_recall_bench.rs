// SPDX-License-Identifier: Apache-2.0

//! Reproducible full-text, dense-vector, and hybrid RRF recall + wall-clock query
//! latency benchmark for `hanzoai/index#55`.
//!
//! ```text
//! cargo run --release -p rag-eval --example hybrid_recall_bench --locked -- [out.json]
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use rag_eval::{
    build_canonical_hybrid_index, canonical_hybrid_queries, evaluate_hybrid_benchmark,
    DeterministicCodeEditZenSynthesizer, DeterministicKaiQueryPlanner,
    DeterministicSemanticCodeEmbedder, DownstreamCodeEditJudge, HybridCostModel, HybridRrfConfig,
    HybridStrategy, LegWeights, DEFAULT_RRF_K,
};

#[derive(Serialize)]
struct StrategyWallClockRow {
    strategy: String,
    recall_at_1: f64,
    recall_at_3: f64,
    recall_at_5: f64,
    recall_at_10: f64,
    mrr_at_10: f64,
    ndcg_at_10: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    acl_leaks: usize,
}

fn weight_override_for(strategy: HybridStrategy) -> Option<LegWeights> {
    match strategy {
        HybridStrategy::LexicalOnly => {
            Some(LegWeights { lexical: 1.0, vector: 0.0, structural: 0.0 })
        }
        HybridStrategy::VectorOnly => {
            Some(LegWeights { lexical: 0.0, vector: 1.0, structural: 0.0 })
        }
        HybridStrategy::StructuralOnly => {
            Some(LegWeights { lexical: 0.0, vector: 0.0, structural: 1.0 })
        }
        HybridStrategy::UniformRrf { .. } => Some(LegWeights::uniform()),
        HybridStrategy::KaiPlannedHybridRrf { .. } | HybridStrategy::UncalibratedRawScoreSum => {
            None
        }
    }
}

fn main() {
    let out_path = std::env::args().nth(1).map(PathBuf::from);
    let (index, meta) =
        build_canonical_hybrid_index().expect("failed to build canonical hybrid index");
    let cases = canonical_hybrid_queries();
    let planner = DeterministicKaiQueryPlanner;
    let embedder = DeterministicSemanticCodeEmbedder;
    let zen = DeterministicCodeEditZenSynthesizer;
    let judge = DownstreamCodeEditJudge;
    let cost = HybridCostModel::default();

    let strategies = [
        HybridStrategy::LexicalOnly,
        HybridStrategy::VectorOnly,
        HybridStrategy::StructuralOnly,
        HybridStrategy::UncalibratedRawScoreSum,
        HybridStrategy::UniformRrf { k: DEFAULT_RRF_K },
        HybridStrategy::KaiPlannedHybridRrf { k: DEFAULT_RRF_K },
    ];

    let report = evaluate_hybrid_benchmark(
        &index,
        "hip1347_hybrid",
        &meta,
        &cases,
        &strategies,
        &planner,
        &embedder,
        &zen,
        &judge,
        &cost,
        10,
    )
    .expect("hybrid benchmark evaluation failed");

    let warmup_rounds = 3usize;
    let measured_rounds = 25usize;
    let mut rows = Vec::with_capacity(strategies.len());

    for (strategy, summary) in strategies.iter().zip(report.strategy_reports.iter()) {
        let cfg = HybridRrfConfig {
            rrf_k: DEFAULT_RRF_K,
            per_leg_candidate_depth: 32,
            top_k: 10,
            weight_override: weight_override_for(*strategy),
        };

        for _ in 0..warmup_rounds {
            for case in &cases {
                if matches!(strategy, HybridStrategy::UncalibratedRawScoreSum) {
                    let _ = index
                        .search_uncalibrated_raw_sum(
                            "hip1347_hybrid",
                            &meta,
                            &case.filter,
                            &case.raw_query,
                            10,
                            &embedder,
                            &cost,
                        )
                        .unwrap();
                } else {
                    let _ = index
                        .search_hybrid(
                            "hip1347_hybrid",
                            &meta,
                            &case.filter,
                            &case.raw_query,
                            &cfg,
                            &planner,
                            &embedder,
                            &cost,
                        )
                        .unwrap();
                }
            }
        }

        let mut samples_us = Vec::with_capacity(cases.len() * measured_rounds);
        let mut r3_hits = 0usize;

        for round in 0..measured_rounds {
            for case in &cases {
                let start = Instant::now();
                let outcome = if matches!(strategy, HybridStrategy::UncalibratedRawScoreSum) {
                    index
                        .search_uncalibrated_raw_sum(
                            "hip1347_hybrid",
                            &meta,
                            &case.filter,
                            &case.raw_query,
                            10,
                            &embedder,
                            &cost,
                        )
                        .unwrap()
                } else {
                    index
                        .search_hybrid(
                            "hip1347_hybrid",
                            &meta,
                            &case.filter,
                            &case.raw_query,
                            &cfg,
                            &planner,
                            &embedder,
                            &cost,
                        )
                        .unwrap()
                };
                samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);

                if round == 0
                    && outcome
                        .hits
                        .iter()
                        .take(3)
                        .any(|h| case.relevant_doc_ids.contains(&h.doc_id))
                {
                    r3_hits += 1;
                }
            }
        }

        samples_us.sort_by(|a, b| a.total_cmp(b));
        let len = samples_us.len();
        let mean_us = samples_us.iter().sum::<f64>() / (len as f64);
        let p50_idx = ((len as f64) * 0.50).ceil() as usize;
        let p95_idx = ((len as f64) * 0.95).ceil() as usize;

        rows.push(StrategyWallClockRow {
            strategy: summary.strategy_label.clone(),
            recall_at_1: summary.recall_at_1,
            recall_at_3: (r3_hits as f64) / (cases.len() as f64),
            recall_at_5: summary.recall_at_5,
            recall_at_10: summary.recall_at_10,
            mrr_at_10: summary.mrr_at_10,
            ndcg_at_10: summary.ndcg_at_10,
            mean_us,
            p50_us: samples_us[p50_idx.saturating_sub(1).min(len - 1)],
            p95_us: samples_us[p95_idx.saturating_sub(1).min(len - 1)],
            acl_leaks: summary.unauthorized_acl_leaks,
        });
    }

    println!(
        "# Hybrid Full-Text, Vector & RRF Recall / Latency Benchmark (corpus={} records, queries={}, measured_rounds={})\n",
        report.corpus_record_count, report.query_count, measured_rounds
    );
    println!("| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean (us) | p50 (us) | p95 (us) | ACL Leaks |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for row in &rows {
        println!(
            "| `{}` | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} | {} |",
            row.strategy,
            row.recall_at_1,
            row.recall_at_3,
            row.recall_at_5,
            row.recall_at_10,
            row.mrr_at_10,
            row.ndcg_at_10,
            row.mean_us,
            row.p50_us,
            row.p95_us,
            row.acl_leaks,
        );
    }

    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).expect("failed to create output directory");
            }
        }
        let json = serde_json::to_string_pretty(&rows).expect("failed to serialize JSON rows");
        fs::write(&path, json).expect("failed to write benchmark JSON");
    }
}
