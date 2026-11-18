// SPDX-License-Identifier: Apache-2.0

//! Bounty #21 (`hanzoai/index#25`) test suite for [`rag_eval::ann_insert`]:
//! immutable-metadata isolation, ground-truth sanity, policy behavior
//! (batch rebuild / in-place / delta merge), and schedule determinism.

use std::fs;
use std::path::PathBuf;

use rag_eval::ann_insert::{
    AnnMaintenanceRunner, BenchConfig, IoCostModel, MaintenancePolicy, SplitMix64,
    generate_workload, run_full_matrix,
};
use rag_eval::{IndexEmbeddingMetadata, MetadataError, NormalizationPolicy};

fn meta(dim: usize) -> IndexEmbeddingMetadata {
    IndexEmbeddingMetadata::new(
        "hanzo/zen-embed-v1",
        "sha256:b21odinann0000000000000000000000000000000000000000000000000000",
        dim,
        NormalizationPolicy::L2,
        "query: ",
        "passage: ",
    )
    .unwrap()
}

/// Mutated copy of the base metadata for mismatch tests.
fn meta_with(mutate: impl FnOnce(&mut IndexEmbeddingMetadata)) -> IndexEmbeddingMetadata {
    let mut m = meta(16);
    mutate(&mut m);
    m
}

/// L2-normalizes in place (f64 accumulate) so vectors satisfy the contract.
fn unit(v: &mut [f32]) {
    let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let norm = norm_sq.sqrt();
    if norm > f64::EPSILON {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

fn seeded_unit_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    unit(&mut v);
    v
}

fn test_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("anninsert-test-{}-{name}", std::process::id()))
}

fn cleanup(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// HIP-1346 §1 isolation: metadata mismatches never mutate the index.
// ---------------------------------------------------------------------------

#[test]
fn metadata_mismatch_rejected_without_mutation() {
    let dir = test_dir("meta-mismatch");
    let m = meta(16);
    let mut runner = AnnMaintenanceRunner::new(
        m.clone(),
        MaintenancePolicy::InPlaceOnline,
        &dir,
        IoCostModel::default(),
    )
    .unwrap();

    let valid = seeded_unit_vector(1, 16);
    runner.insert(&m, &valid).unwrap();
    let nodes_before = runner.node_count();
    let bytes_before = runner.store_bytes();
    assert_eq!(nodes_before, 1);
    assert_eq!(bytes_before, 16 * 4);

    // Dimension mismatch.
    let wrong_dims = meta_with(|mm| mm.dimensions = 8);
    let err = runner.insert(&wrong_dims, &seeded_unit_vector(2, 8)).unwrap_err();
    assert!(matches!(err, rag_eval::ann_insert::AnnInsertError::Metadata(_)));

    // model_id mismatch.
    let wrong_model = meta_with(|mm| mm.model_id = "other/model".into());
    let err = runner.insert(&wrong_model, &seeded_unit_vector(3, 16)).unwrap_err();
    assert!(matches!(err, rag_eval::ann_insert::AnnInsertError::Metadata(_)));
    assert!(matches!(
        err,
        rag_eval::ann_insert::AnnInsertError::Metadata(MetadataError::ContractMismatch { .. })
    ));

    // model_revision mismatch.
    let wrong_rev = meta_with(|mm| mm.model_revision = "sha256:deadbeef".into());
    let err = runner.insert(&wrong_rev, &seeded_unit_vector(4, 16)).unwrap_err();
    assert!(matches!(err, rag_eval::ann_insert::AnnInsertError::Metadata(_)));

    // normalization mismatch.
    let wrong_norm = meta_with(|mm| mm.normalization = NormalizationPolicy::None);
    let err = runner.insert(&wrong_norm, &seeded_unit_vector(5, 16)).unwrap_err();
    assert!(matches!(err, rag_eval::ann_insert::AnnInsertError::Metadata(_)));

    // Zero mutation across every rejection.
    assert_eq!(runner.node_count(), nodes_before);
    assert_eq!(runner.store_bytes(), bytes_before);
    assert_eq!(runner.pending_len(), 0);
    assert_eq!(runner.stats().rejected_inserts, 4);

    // The index remains fully usable for correctly-tagged inserts.
    runner.insert(&m, &seeded_unit_vector(6, 16)).unwrap();
    assert_eq!(runner.node_count(), 2);

    cleanup(&dir);
}

#[test]
fn query_mismatch_rejected_without_latency_accounting() {
    let dir = test_dir("query-mismatch");
    let m = meta(16);
    let mut runner = AnnMaintenanceRunner::new(
        m.clone(),
        MaintenancePolicy::InPlaceOnline,
        &dir,
        IoCostModel::default(),
    )
    .unwrap();

    let base: Vec<Vec<f32>> = (0..3).map(|i| seeded_unit_vector(100 + i, 16)).collect();
    runner.build_base(&m, &base).unwrap();

    let q = seeded_unit_vector(100, 16);
    let hits = runner.search(&m, &q, 10).unwrap();
    assert!(!hits.is_empty());
    let served = runner.stats().queries_served;
    let latencies = runner.stats().query_latencies_ns.len();

    let wrong = meta_with(|mm| mm.query_prefix = "search: ".into());
    let err = runner.search(&wrong, &q, 10).unwrap_err();
    assert!(matches!(err, rag_eval::ann_insert::AnnInsertError::Metadata(_)));
    assert_eq!(runner.stats().queries_served, served);
    assert_eq!(runner.stats().query_latencies_ns.len(), latencies);
    assert_eq!(runner.stats().rejected_queries, 1);

    cleanup(&dir);
}

#[test]
fn l2_normalization_violation_rejected() {
    let dir = test_dir("l2-violation");
    let m = meta(16);
    let mut runner = AnnMaintenanceRunner::new(
        m.clone(),
        MaintenancePolicy::InPlaceOnline,
        &dir,
        IoCostModel::default(),
    )
    .unwrap();

    let bad = vec![2.0f32; 16];
    let err = runner.insert(&m, &bad).unwrap_err();
    assert!(matches!(
        err,
        rag_eval::ann_insert::AnnInsertError::Metadata(MetadataError::NormalizationViolation { .. })
    ));
    assert_eq!(runner.node_count(), 0);
    assert_eq!(runner.store_bytes(), 0);

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Retrieval sanity + policy behavior.
// ---------------------------------------------------------------------------

#[test]
fn nearest_neighbor_sanity() {
    let dir = test_dir("nn-sanity");
    let m = meta(4);
    let mut runner = AnnMaintenanceRunner::new(
        m.clone(),
        MaintenancePolicy::InPlaceOnline,
        &dir,
        IoCostModel::default(),
    )
    .unwrap();

    let a = vec![1.0f32, 0.0, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0, 0.0];
    let c = vec![0.0f32, 0.0, 1.0, 0.0];
    runner.build_base(&m, &[a, b, c]).unwrap();

    let mut q = vec![0.99f32, 0.1, 0.0, 0.0];
    unit(&mut q);
    let hits = runner.search(&m, &q, 1).unwrap();
    assert_eq!(hits[0].0, 0, "nearest neighbor must be the x-axis vector");

    cleanup(&dir);
}

#[test]
fn policy_a_pending_tail_searchable_then_rebuilds() {
    let dir = test_dir("policy-a");
    let m = meta(16);
    let policy = MaintenancePolicy::BatchRebuild { rebuild_interval: 16 };
    let mut runner =
        AnnMaintenanceRunner::new(m.clone(), policy, &dir, IoCostModel::default()).unwrap();

    let base: Vec<Vec<f32>> = (0..40).map(|i| seeded_unit_vector(1000 + i, 16)).collect();
    runner.build_base(&m, &base).unwrap();

    // Stream inserts: the unindexed tail stays queryable.
    for i in 0..20 {
        runner.insert(&m, &seeded_unit_vector(3000 + i, 16)).unwrap();
    }
    assert_eq!(runner.stats().rebuilds, 1, "interval 16 with 20 stream inserts");
    assert_eq!(runner.pending_len(), 4);

    // Exact-match queries: one for a rebuilt-graph slot (40), one for a
    // pending-tail slot (59) — the linear scan must surface the latter.
    let in_graph = seeded_unit_vector(3000, 16);
    let hits = runner.search(&m, &in_graph, 10).unwrap();
    assert_eq!(hits[0].0, 40, "rebuilt-graph vector must be found by graph search");
    let in_pending = seeded_unit_vector(3000 + 19, 16);
    let hits = runner.search(&m, &in_pending, 10).unwrap();
    assert_eq!(
        hits[0].0, 59,
        "unindexed-tail vector must remain searchable via linear scan"
    );
    assert_eq!(
        runner.node_count(),
        56,
        "graph holds base(40) + first 16 rebuilt stream inserts; last 4 sit in pending"
    );

    cleanup(&dir);
}

#[test]
fn policy_c_bounded_drain_and_emergency_merge() {
    let dir = test_dir("policy-c");
    let m = meta(16);
    let policy = MaintenancePolicy::DeltaMerge { delta_capacity: 8, merge_budget_per_query: 4 };
    let mut runner =
        AnnMaintenanceRunner::new(m.clone(), policy, &dir, IoCostModel::default()).unwrap();

    let base: Vec<Vec<f32>> = (0..40).map(|i| seeded_unit_vector(4000 + i, 16)).collect();
    runner.build_base(&m, &base).unwrap();

    // 6 inserts land in the delta layer; 3 queries drain 4/2/0.
    for i in 0..6 {
        runner.insert(&m, &seeded_unit_vector(5000 + i, 16)).unwrap();
    }
    assert_eq!(runner.merge_queue_len(), 6);
    for i in 0..3 {
        runner.search(&m, &seeded_unit_vector(6000 + i, 16), 10).unwrap();
    }
    assert_eq!(runner.merge_queue_len(), 0, "budget 4/query drains the queue");
    assert_eq!(runner.stats().emergency_merges, 0);
    assert_eq!(runner.stats().rebuilds, 0);

    // Burst of 10 inserts with no queries: overflow at the 8th → emergency.
    for i in 0..10 {
        runner.insert(&m, &seeded_unit_vector(7000 + i, 16)).unwrap();
    }
    assert_eq!(runner.stats().emergency_merges, 1);
    assert_eq!(runner.merge_queue_len(), 2);
    assert!(runner.stats().stall_ns > 0, "emergency merge is a real stall");

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Determinism + end-to-end matrix.
// ---------------------------------------------------------------------------

#[test]
fn workload_generation_is_deterministic() {
    let cfg = BenchConfig::test_config();
    let a = generate_workload(&cfg, 1.0).unwrap();
    let b = generate_workload(&cfg, 1.0).unwrap();
    assert_eq!(a.base, b.base);
    assert_eq!(a.inserts, b.inserts);
    assert_eq!(a.queries, b.queries);
    assert_eq!(a.ops, b.ops);
    assert_eq!(a.ground_truth, b.ground_truth);
    assert_eq!(a.ground_truth.len(), cfg.n_queries);
    for truth in &a.ground_truth {
        assert_eq!(truth.len(), 10, "recall@10 ground truth must be top-10");
    }
}

#[test]
fn full_matrix_small_end_to_end() {
    let cfg = BenchConfig::test_config();
    let report = run_full_matrix(&cfg).unwrap();

    assert_eq!(report.runs.len(), 3, "one run per policy at the single rate");
    assert_eq!(report.honesty_labels.len(), 7);
    assert_eq!(report.hypothesis_verdicts.len(), 1);

    for run in &report.runs {
        assert_eq!(run.inserts, cfg.n_base as u64 + 24, "base corpus + rate×queries inserts");
        assert_eq!(run.queries, cfg.n_queries as u64);
        assert!(run.recall_mean > 0.5, "clustered synthetic data must be retrievable, got {}", run.recall_mean);
        assert!(run.p99_query_ns >= run.p50_query_ns, "p99 ≥ p50 structurally");
        assert!(run.mean_query_ns > 0);
    }

    let a = &report.runs[0];
    let b = &report.runs[1];
    let c = &report.runs[2];
    assert!(a.policy.starts_with("batch-rebuild"));
    assert_eq!(b.policy, "in-place-online");
    assert!(c.policy.starts_with("delta-merge"));
    assert!(a.rebuilds >= 1, "24 stream inserts vs interval 16 must trigger a rebuild");
    // (B) is the latency floor reference: no stalls, no emergency merges.
    assert_eq!(b.stall_ns, 0);
    assert_eq!(b.rebuilds, 0);
    assert_eq!(b.emergency_merges, 0);
}
