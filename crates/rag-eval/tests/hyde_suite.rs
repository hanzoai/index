// SPDX-License-Identifier: Apache-2.0
//! Integration suite for `rag_eval::hyde`.
use rag_eval::{FusionStrategy, HydeRunner, IndexEmbeddingMetadata, MetadataError, NormalizationPolicy};

fn meta() -> IndexEmbeddingMetadata {
    IndexEmbeddingMetadata::new(
        "hanzo/zen-embed-v1",
        "sha256:b21odinann0000000000000000000000000000000000000000000000000000",
        16,
        NormalizationPolicy::L2,
        "query: ",
        "passage: ",
    )
    .unwrap()
}

fn bad_meta() -> IndexEmbeddingMetadata {
    let mut m = meta();
    m.model_revision = "wrong-rev".into();
    m
}

#[test]
fn metadata_mismatch_rejected_without_mutation() {
    let base = vec![vec![0.1f32; 16]];
    let mut runner = HydeRunner::new(meta(), FusionStrategy::UnionRerank, 7u64, &base, 10).unwrap();
    let before = runner.stats().queries_served;
    // Run with mismatched runtime metadata — must reject before any mutation.
    let err = runner.run_ops(&bad_meta(), &[], &base, &[]);
    assert!(err.is_err());
    assert_eq!(runner.stats().queries_served, before, "no side-effects on rejection");
}

#[test]
fn baseline_and_hyde_runs_end_to_end() {
    let cfg = rag_eval::BenchConfig::test_config();
    let report = rag_eval::run_hyde_matrix(&cfg).unwrap();
    assert_eq!(report.runs.len(), 3, "one run per fusion strategy");
    let names: Vec<&str> = report.runs.iter().map(|r| r.fusion.as_str()).collect();
    assert!(names.contains(&"fusion-baseline"));
    assert!(names.contains(&"fusion-union-rerank"));
    assert!(names.contains(&"fusion-pure-hyde"));
    for r in &report.runs {
        assert!(r.queries_served >= 1, "every strategy serves queries");
        assert!(r.cost_units > 0, "cost model counts operations");
    }
}
