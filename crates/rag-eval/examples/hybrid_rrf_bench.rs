// SPDX-License-Identifier: Apache-2.0

//! Reproducible benchmark CLI runner for `hanzoai/index#11` (`HIP-1347 p0`):
//! Hybrid Lexical, Vector & Structural Search Planner with Reciprocal Rank Fusion (RRF).
//!
//! Executes the 6-strategy matched baseline across the canonical 22-record
//! multi-tenant corpus and 20-query 5-slice workload, prints the Markdown
//! summary table to stdout, and optionally writes machine-readable JSON to
//! the path supplied as the first CLI argument.
//!
//! ```text
//! cargo run --release -p rag-eval --example hybrid_rrf_bench -- [out.json]
//! ```

use std::fs;
use std::path::PathBuf;

use rag_eval::{
    build_canonical_hybrid_index, canonical_hybrid_queries, evaluate_hybrid_benchmark,
    DeterministicCodeEditZenSynthesizer, DeterministicKaiQueryPlanner,
    DeterministicSemanticCodeEmbedder, DownstreamCodeEditJudge, HybridCostModel, HybridStrategy,
    DEFAULT_RRF_K,
};

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
    .expect("hybrid RRF benchmark failed");

    println!(
        "# HIP-1347 Hybrid Search & RRF Benchmark (corpus={} records, queries={}, top_k={})\n",
        report.corpus_record_count, report.query_count, report.top_k
    );
    println!(
        "| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Exact R@1 | Semantic R@1 | AST R@1 | Path R@1 | MultiLeg R@1 | Task Success | p50 (ms) | p95 (ms) | Total Cost (USD) | ACL Leaks |"
    );
    println!(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    for r in &report.strategy_reports {
        println!(
            "| `{}` | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | ${:.6} | {} |",
            r.strategy_label,
            r.recall_at_1,
            r.recall_at_5,
            r.recall_at_10,
            r.mrr_at_10,
            r.ndcg_at_10,
            r.exact_symbol_recall_at_1,
            r.semantic_paraphrase_recall_at_1,
            r.structural_ast_recall_at_1,
            r.path_scoped_recall_at_1,
            r.hybrid_multileg_recall_at_1,
            r.downstream_task_success_rate,
            r.p50_latency_ms,
            r.p95_latency_ms,
            r.total_cost_usd,
            r.unauthorized_acl_leaks,
        );
    }

    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).expect("failed to create output directory");
            }
        }
        let json =
            serde_json::to_string_pretty(&report).expect("failed to serialize report to JSON");
        fs::write(&path, json).expect("failed to write benchmark JSON");
        eprintln!("wrote {}", path.display());
    }
}
