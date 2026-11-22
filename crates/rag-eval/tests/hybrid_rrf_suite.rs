// SPDX-License-Identifier: Apache-2.0

//! Integration and reproducible benchmark suite for `hanzoai/index#11` (`HIP-1347 p0`):
//! Hybrid Lexical, Vector & Structural Search Planner with Reciprocal Rank Fusion (RRF).

use std::sync::Arc;
use std::thread;

use rag_eval::{
    build_canonical_hybrid_index, canonical_hybrid_queries, evaluate_hybrid_benchmark,
    CanonicalEvidenceRecord, DecisionHead, DeterministicCodeEditZenSynthesizer,
    DeterministicKaiQueryPlanner, DeterministicSemanticCodeEmbedder, DownstreamCodeEditJudge,
    HybridAccessFilter, HybridBenchmarkSuiteReport, HybridCostModel, HybridQueryClass,
    HybridRrfConfig, HybridSearchOutcome, HybridStrategy, SourceArtifactKind,
    SourceIdentity, StructuralNodeKind, StructuralSymbolNode, DEFAULT_RRF_K,
};

#[test]
fn issue_11_canonical_queries_record_finish_and_agent_freeze_deadlock_rank_first() {
    let (index, meta) = build_canonical_hybrid_index().unwrap();
    let planner = DeterministicKaiQueryPlanner;
    let embedder = DeterministicSemanticCodeEmbedder;
    let cost = HybridCostModel::default();
    let cfg = HybridRrfConfig::default();
    let filter = HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);

    // 1. Canonical exact symbol query from Issue #11: "record_finish"
    let exact_out = index
        .search_hybrid(
            "hip1347_hybrid",
            &meta,
            &filter,
            "record_finish",
            &cfg,
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();

    assert_eq!(exact_out.plan.decision_head, DecisionHead::Kai);
    assert_eq!(
        exact_out.plan.query_class,
        HybridQueryClass::ExactSymbolOrIdentifier
    );
    assert!(!exact_out.hits.is_empty());
    let top_exact = &exact_out.hits[0];
    assert_eq!(
        top_exact.doc_id,
        "blob:crates/agent/src/runner.rs#record_finish"
    );
    assert_eq!(top_exact.final_rank, 1);
    assert!(top_exact.explanation.lexical.is_some());
    assert!(top_exact.explanation.structural.is_some());
    assert_eq!(
        top_exact.explanation.lexical.as_ref().unwrap().rank_1_based,
        1
    );
    assert_eq!(
        top_exact
            .explanation
            .structural
            .as_ref()
            .unwrap()
            .rank_1_based,
        1
    );
    assert!(top_exact
        .explanation
        .matched_symbols
        .contains(&"hanzo_agent::runner::AgentRunner::record_finish".to_string()));

    // 2. Canonical paraphrased natural-language query from Issue #11:
    // "why does the agent freeze after completing a request?"
    let sem_out = index
        .search_hybrid(
            "hip1347_hybrid",
            &meta,
            &filter,
            "why does the agent freeze after completing a request?",
            &cfg,
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();

    assert_eq!(
        sem_out.plan.query_class,
        HybridQueryClass::NaturalLanguageSemantic
    );
    assert!(!sem_out.hits.is_empty());
    let top_sem = &sem_out.hits[0];
    assert_eq!(
        top_sem.doc_id,
        "doc:incidents/INC-409-agent-completion-deadlock.md"
    );
    assert_eq!(top_sem.final_rank, 1);
    assert!(top_sem.explanation.vector.is_some());
    assert_eq!(
        top_sem.explanation.vector.as_ref().unwrap().rank_1_based,
        1
    );
}

#[test]
fn matched_six_strategy_benchmark_proves_rrf_beats_single_leg_and_uncalibrated_raw_sums() {
    let (index, meta) = build_canonical_hybrid_index().unwrap();
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

    let suite = evaluate_hybrid_benchmark(
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
    .unwrap();

    assert_eq!(suite.query_count, 20);
    assert_eq!(suite.strategy_reports.len(), 6);

    for r in &suite.strategy_reports {
        println!(
            "HIP1347_BENCH strategy=\"{}\" R@1={:.3} R@5={:.3} R@10={:.3} MRR@10={:.3} nDCG@10={:.3} \
             exact_R@1={:.3} sem_R@1={:.3} ast_R@1={:.3} path_R@1={:.3} multi_R@1={:.3} \
             downstream_success={:.3} p50_ms={:.2} p95_ms={:.2} total_cost_usd=${:.6} acl_leaks={}",
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
        assert_eq!(
            r.unauthorized_acl_leaks, 0,
            "zero cross-tenant or role ACL leakage permitted in strategy {}",
            r.strategy_label
        );
    }

    let lexical = &suite.strategy_reports[0];
    let vector = &suite.strategy_reports[1];
    let structural = &suite.strategy_reports[2];
    let raw_sum = &suite.strategy_reports[3];
    let uniform_rrf = &suite.strategy_reports[4];
    let kai_rrf = &suite.strategy_reports[5];

    // 1. KaiPlannedHybridRrf achieves 100% Recall@1, Recall@10, MRR@10, nDCG@10, and downstream task success.
    assert!((kai_rrf.recall_at_1 - 1.0).abs() < 1e-9);
    assert!((kai_rrf.recall_at_5 - 1.0).abs() < 1e-9);
    assert!((kai_rrf.recall_at_10 - 1.0).abs() < 1e-9);
    assert!((kai_rrf.mrr_at_10 - 1.0).abs() < 1e-9);
    assert!((kai_rrf.ndcg_at_10 - 1.0).abs() < 1e-9);
    assert!((kai_rrf.downstream_task_success_rate - 1.0).abs() < 1e-9);

    // 2. Single-leg blind spots:
    // - LexicalOnly fails on pure vocabulary-mismatch semantic paraphrases.
    assert!(
        lexical.semantic_paraphrase_recall_at_1 < kai_rrf.semantic_paraphrase_recall_at_1,
        "LexicalOnly must underperform KaiPlannedHybridRrf on vocabulary-mismatch semantic queries"
    );
    // - StructuralOnly fails on natural-language semantic queries that carry no AST symbols.
    assert!(
        structural.semantic_paraphrase_recall_at_1 < 0.25,
        "StructuralOnly cannot answer pure natural-language semantic queries without AST symbols"
    );
    // - KaiPlannedHybridRrf strictly outperforms every single-leg baseline on overall Recall@1, MRR@10, and downstream task success.
    for single in [lexical, vector, structural] {
        assert!(
            kai_rrf.recall_at_1 > single.recall_at_1,
            "KaiPlannedHybridRrf R@1 ({:.3}) must beat {} ({:.3})",
            kai_rrf.recall_at_1,
            single.strategy_label,
            single.recall_at_1
        );
        assert!(
            kai_rrf.mrr_at_10 > single.mrr_at_10,
            "KaiPlannedHybridRrf MRR@10 ({:.3}) must beat {} ({:.3})",
            kai_rrf.mrr_at_10,
            single.strategy_label,
            single.mrr_at_10
        );
        assert!(
            kai_rrf.downstream_task_success_rate > single.downstream_task_success_rate,
            "KaiPlannedHybridRrf downstream success ({:.3}) must beat {} ({:.3})",
            kai_rrf.downstream_task_success_rate,
            single.strategy_label,
            single.downstream_task_success_rate
        );
    }

    // 3. Negative control: UncalibratedRawScoreSum is degraded by high-magnitude BM25 distractors
    // (`doc:distractor/release_notes_archive.md`, `doc:distractor/ui_tooltip_animation.md`).
    assert!(
        kai_rrf.recall_at_1 > raw_sum.recall_at_1,
        "KaiPlannedHybridRrf R@1 ({:.3}) must beat UncalibratedRawScoreSum ({:.3})",
        kai_rrf.recall_at_1,
        raw_sum.recall_at_1
    );
    assert!(
        uniform_rrf.recall_at_1 >= raw_sum.recall_at_1,
        "RRF rank normalization must be at least as accurate as uncalibrated raw score summation"
    );
    assert!(
        kai_rrf.recall_at_1 >= uniform_rrf.recall_at_1,
        "KaiPlannedHybridRrf must be >= UniformRrf on Recall@1"
    );
}

#[test]
fn prefiltering_prevents_topk_starvation_under_high_density_other_tenant_distractors() {
    let (mut index, meta) = build_canonical_hybrid_index().unwrap();
    let planner = DeterministicKaiQueryPlanner;
    let embedder = DeterministicSemanticCodeEmbedder;
    let cost = HybridCostModel::default();

    // Insert 50 high-density adversarial records in `tenant_adversary` and restricted `root_only`
    // role in `tenant_hanzo` that repeat `record_finish` and `release_turn_lock`.
    // If ACL filtering were applied POST-scoring after a small per-leg candidate cutoff,
    // these 50 records would starve the top-1 heap and cause 0 results for `tenant_hanzo`.
    for i in 0..50usize {
        let tenant = if i % 2 == 0 {
            "tenant_adversary"
        } else {
            "tenant_hanzo"
        };
        let principals = if i % 2 == 0 {
            vec![]
        } else {
            vec!["root_only".to_string()]
        };

        let rec = CanonicalEvidenceRecord {
            identity: SourceIdentity {
                doc_id: format!("adv:distractor_{i}"),
                source_id: "repo:adversary".to_string(),
                version_id: "git:adv1".to_string(),
                artifact_kind: SourceArtifactKind::RepositoryBlob,
                tenant_id: tenant.to_string(),
                project_id: "proj_index".to_string(),
                repo_id: "hanzoai/index".to_string(),
                allowed_principals: principals,
                path: "crates/agent/src/runner.rs".to_string(),
                language: "rust".to_string(),
            },
            title: "record_finish record_finish release_turn_lock".to_string(),
            raw_text: "record_finish record_finish record_finish release_turn_lock deadlock parking_lot mutex"
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "record_finish".to_string(),
                qualified_name: "adv::AgentRunner::record_finish".to_string(),
                node_kind: StructuralNodeKind::Function,
                signature: "pub async fn record_finish(&mut self)".to_string(),
                parent_scope: Some("AgentRunner".to_string()),
                imports: vec![],
                calls: vec!["release_turn_lock".to_string()],
                called_by: vec![],
                start_line: 1,
                end_line: 10,
            }],
        };
        index
            .upsert_record("hip1347_hybrid", &meta, rec, &embedder)
            .unwrap();
    }

    // Query with tightest possible candidate budget: `top_k = 1`, `per_leg_candidate_depth = 1`.
    let tight_cfg = HybridRrfConfig {
        rrf_k: DEFAULT_RRF_K,
        per_leg_candidate_depth: 1,
        top_k: 1,
        weight_override: None,
    };
    let dev_filter =
        HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);
    let out = index
        .search_hybrid(
            "hip1347_hybrid",
            &meta,
            &dev_filter,
            "record_finish",
            &tight_cfg,
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();

    assert_eq!(out.hits.len(), 1);
    assert_eq!(
        out.hits[0].doc_id,
        "blob:crates/agent/src/runner.rs#record_finish"
    );
    assert_eq!(out.hits[0].identity.tenant_id, "tenant_hanzo");
}

#[test]
fn serde_json_roundtrip_and_cross_arch_float_stability() {
    let (index, meta) = build_canonical_hybrid_index().unwrap();
    let cases = canonical_hybrid_queries();
    let planner = DeterministicKaiQueryPlanner;
    let embedder = DeterministicSemanticCodeEmbedder;
    let zen = DeterministicCodeEditZenSynthesizer;
    let judge = DownstreamCodeEditJudge;
    let cost = HybridCostModel::default();

    let strategies = [
        HybridStrategy::UniformRrf { k: DEFAULT_RRF_K },
        HybridStrategy::KaiPlannedHybridRrf { k: DEFAULT_RRF_K },
    ];
    let suite = evaluate_hybrid_benchmark(
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
    .unwrap();

    let json = serde_json::to_string_pretty(&suite).unwrap();
    let restored: HybridBenchmarkSuiteReport = serde_json::from_str(&json).unwrap();
    assert_eq!(suite.collection, restored.collection);
    assert_eq!(suite.corpus_record_count, restored.corpus_record_count);
    assert_eq!(suite.query_count, restored.query_count);
    assert_eq!(suite.strategy_reports.len(), restored.strategy_reports.len());

    for (orig, rest) in suite
        .strategy_reports
        .iter()
        .zip(restored.strategy_reports.iter())
    {
        assert_eq!(orig.strategy_label, rest.strategy_label);
        assert_eq!(orig.unauthorized_acl_leaks, rest.unauthorized_acl_leaks);
        assert!((orig.recall_at_1 - rest.recall_at_1).abs() < 1e-9);
        assert!((orig.recall_at_10 - rest.recall_at_10).abs() < 1e-9);
        assert!((orig.mrr_at_10 - rest.mrr_at_10).abs() < 1e-9);
        assert!((orig.ndcg_at_10 - rest.ndcg_at_10).abs() < 1e-9);
        assert!(
            (orig.downstream_task_success_rate - rest.downstream_task_success_rate).abs() < 1e-9
        );
        assert!((orig.total_cost_usd - rest.total_cost_usd).abs() < 1e-9);
    }

    // Also verify SearchHit & HybridSearchOutcome serde roundtrip with epsilon tolerance.
    let filter = HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);
    let outcome = index
        .search_hybrid(
            "hip1347_hybrid",
            &meta,
            &filter,
            "record_finish",
            &HybridRrfConfig::default(),
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();
    let out_json = serde_json::to_string(&outcome).unwrap();
    let out_restored: HybridSearchOutcome = serde_json::from_str(&out_json).unwrap();
    assert_eq!(outcome.hits.len(), out_restored.hits.len());
    for (h1, h2) in outcome.hits.iter().zip(out_restored.hits.iter()) {
        assert_eq!(h1.doc_id, h2.doc_id);
        assert_eq!(h1.final_rank, h2.final_rank);
        assert!((h1.rrf_score - h2.rrf_score).abs() < 1e-9);
    }
}

#[test]
fn concurrent_16_thread_hybrid_search_preserves_determinism_and_tenant_isolation() {
    let (index, meta) = build_canonical_hybrid_index().unwrap();
    let shared_index = Arc::new(index);
    let mut handles = Vec::with_capacity(16);

    for tid in 0..16usize {
        let idx = Arc::clone(&shared_index);
        let m = meta.clone();
        handles.push(thread::spawn(move || {
            let planner = DeterministicKaiQueryPlanner;
            let embedder = DeterministicSemanticCodeEmbedder;
            let cost = HybridCostModel::default();
            let cfg = HybridRrfConfig::default();

            let (tenant, principals, query, expected_top1) = match tid % 4 {
                0 => (
                    "tenant_hanzo",
                    vec!["developer".to_string()],
                    "record_finish",
                    "blob:crates/agent/src/runner.rs#record_finish",
                ),
                1 => (
                    "tenant_hanzo",
                    vec!["developer".to_string()],
                    "why does the agent freeze after completing a request?",
                    "doc:incidents/INC-409-agent-completion-deadlock.md",
                ),
                2 => (
                    "tenant_hanzo",
                    vec!["security_admin".to_string()],
                    "rotating SEV-SNP enclave attestation keys and master KMS seals",
                    "doc:security/enclave_kms_rotation.md",
                ),
                _ => (
                    "tenant_external",
                    vec![],
                    "record_finish",
                    "blob:tenant_external/runner.rs#record_finish",
                ),
            };

            let filter = HybridAccessFilter::for_tenant(tenant, principals.clone());
            let out = idx
                .search_hybrid(
                    "hip1347_hybrid",
                    &m,
                    &filter,
                    query,
                    &cfg,
                    &planner,
                    &embedder,
                    &cost,
                )
                .unwrap();

            assert!(!out.hits.is_empty());
            assert_eq!(out.hits[0].doc_id, expected_top1);
            for hit in &out.hits {
                assert_eq!(hit.identity.tenant_id, tenant);
                if !principals.contains(&"security_admin".to_string()) {
                    assert_ne!(hit.doc_id, "doc:security/enclave_kms_rotation.md");
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}
