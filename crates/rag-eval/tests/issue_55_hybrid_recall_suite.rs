// SPDX-License-Identifier: Apache-2.0

//! Deterministic full-text, vector, and hybrid RRF ranking test suite with
//! pre-scoring filter predicate checks and reproducible top-k recall/latency metrics
//! for `hanzoai/index#55`.

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use rag_eval::{
    build_canonical_hybrid_index, canonical_hybrid_corpus, canonical_hybrid_embedding_metadata,
    canonical_hybrid_queries, Bm25LexicalIndex, CanonicalEvidenceRecord, DenseVectorIndex,
    DeterministicKaiQueryPlanner, DeterministicSemanticCodeEmbedder, HybridAccessFilter,
    HybridCostModel, HybridQueryClass, HybridRrfConfig, HybridSearchError, HybridSearchIndex,
    HybridStrategy, IndexOperation, LegWeights, PrefixedTextEmbedder, SourceArtifactKind,
    SourceIdentity, StructuralNodeKind, StructuralSymbolNode, DEFAULT_RRF_K,
};

#[derive(Debug, Clone)]
struct RecallLatencyMetrics {
    label: String,
    recall_at_1: f64,
    recall_at_3: f64,
    recall_at_5: f64,
    recall_at_10: f64,
    mrr_at_10: f64,
    mean_latency_us: f64,
    p50_latency_us: f64,
    p95_latency_us: f64,
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
        HybridStrategy::KaiPlannedHybridRrf { .. } => None,
        HybridStrategy::UncalibratedRawScoreSum => None,
    }
}

fn measure_strategy_recall_and_latency(
    index: &HybridSearchIndex,
    meta: &rag_eval::IndexEmbeddingMetadata,
    strategy: HybridStrategy,
    warmup_rounds: usize,
    measured_rounds: usize,
) -> RecallLatencyMetrics {
    let cases = canonical_hybrid_queries();
    let planner = DeterministicKaiQueryPlanner;
    let embedder = DeterministicSemanticCodeEmbedder;
    let cost = HybridCostModel::default();
    let rrf_k = match strategy {
        HybridStrategy::UniformRrf { k } | HybridStrategy::KaiPlannedHybridRrf { k } => k,
        _ => DEFAULT_RRF_K,
    };
    let cfg = HybridRrfConfig {
        rrf_k,
        per_leg_candidate_depth: 32,
        top_k: 10,
        weight_override: weight_override_for(strategy),
    };

    for _ in 0..warmup_rounds {
        for case in &cases {
            if matches!(strategy, HybridStrategy::UncalibratedRawScoreSum) {
                let _ = index
                    .search_uncalibrated_raw_sum(
                        "hip1347_hybrid",
                        meta,
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
                        meta,
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

    let mut latencies_us = Vec::with_capacity(cases.len() * measured_rounds);
    let mut r1 = 0.0_f64;
    let mut r3 = 0.0_f64;
    let mut r5 = 0.0_f64;
    let mut r10 = 0.0_f64;
    let mut mrr = 0.0_f64;

    for round in 0..measured_rounds {
        for case in &cases {
            let start = Instant::now();
            let outcome = if matches!(strategy, HybridStrategy::UncalibratedRawScoreSum) {
                index
                    .search_uncalibrated_raw_sum(
                        "hip1347_hybrid",
                        meta,
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
                        meta,
                        &case.filter,
                        &case.raw_query,
                        &cfg,
                        &planner,
                        &embedder,
                        &cost,
                    )
                    .unwrap()
            };
            latencies_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);

            if round == 0 {
                let hit_ids: Vec<&str> = outcome.hits.iter().map(|h| h.doc_id.as_str()).collect();
                let is_rel = |id: &str| case.relevant_doc_ids.iter().any(|r| r == id);
                if hit_ids.first().is_some_and(|id| is_rel(id)) {
                    r1 += 1.0;
                }
                if hit_ids.iter().take(3).any(|id| is_rel(id)) {
                    r3 += 1.0;
                }
                if hit_ids.iter().take(5).any(|id| is_rel(id)) {
                    r5 += 1.0;
                }
                if hit_ids.iter().take(10).any(|id| is_rel(id)) {
                    r10 += 1.0;
                }
                if let Some(pos) = hit_ids.iter().take(10).position(|id| is_rel(id)) {
                    mrr += 1.0 / ((pos + 1) as f64);
                }
            }
        }
    }

    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let len = latencies_us.len();
    let n_queries = cases.len() as f64;
    let p50_idx = ((len as f64) * 0.50).ceil() as usize;
    let p95_idx = ((len as f64) * 0.95).ceil() as usize;

    RecallLatencyMetrics {
        label: strategy.label(),
        recall_at_1: r1 / n_queries,
        recall_at_3: r3 / n_queries,
        recall_at_5: r5 / n_queries,
        recall_at_10: r10 / n_queries,
        mrr_at_10: mrr / n_queries,
        mean_latency_us: latencies_us.iter().sum::<f64>() / (len as f64),
        p50_latency_us: latencies_us[p50_idx.saturating_sub(1).min(len - 1)],
        p95_latency_us: latencies_us[p95_idx.saturating_sub(1).min(len - 1)],
    }
}

#[test]
fn full_text_vector_and_hybrid_rrf_rankings_are_deterministic_across_insertion_order() {
    let meta = canonical_hybrid_embedding_metadata();
    let embedder = DeterministicSemanticCodeEmbedder;
    let forward = canonical_hybrid_corpus();

    let mut reversed = forward.clone();
    reversed.reverse();

    let mut idx_forward = HybridSearchIndex::new();
    let mut idx_reversed = HybridSearchIndex::new();
    idx_forward.register_collection("hip1347_hybrid", meta.clone()).unwrap();
    idx_reversed.register_collection("hip1347_hybrid", meta.clone()).unwrap();

    for rec in &forward {
        idx_forward.upsert_record("hip1347_hybrid", &meta, rec.clone(), &embedder).unwrap();
    }
    for rec in &reversed {
        idx_reversed.upsert_record("hip1347_hybrid", &meta, rec.clone(), &embedder).unwrap();
    }

    let planner = DeterministicKaiQueryPlanner;
    let cost = HybridCostModel::default();
    let queries = canonical_hybrid_queries();

    for strategy in [
        HybridStrategy::LexicalOnly,
        HybridStrategy::VectorOnly,
        HybridStrategy::UniformRrf { k: DEFAULT_RRF_K },
        HybridStrategy::KaiPlannedHybridRrf { k: DEFAULT_RRF_K },
    ] {
        let cfg = HybridRrfConfig {
            rrf_k: DEFAULT_RRF_K,
            per_leg_candidate_depth: 32,
            top_k: 10,
            weight_override: weight_override_for(strategy),
        };

        for case in &queries {
            let out_a = idx_forward
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
            let out_b = idx_reversed
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

            let ids_a: Vec<&str> = out_a.hits.iter().map(|h| h.doc_id.as_str()).collect();
            let ids_b: Vec<&str> = out_b.hits.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                ids_a,
                ids_b,
                "insertion order altered ranking for {} / {}",
                strategy.label(),
                case.case_id
            );

            // Scores must strictly follow sorted rank order and match across insertion permutations.
            for win in out_a.hits.windows(2) {
                assert!(
                    win[0].rrf_score + 1e-12 >= win[1].rrf_score,
                    "hits out of descending order in {} / {}",
                    strategy.label(),
                    case.case_id
                );
            }
            for (ha, hb) in out_a.hits.iter().zip(out_b.hits.iter()) {
                assert_eq!(ha.final_rank, hb.final_rank);
                assert!((ha.rrf_score - hb.rrf_score).abs() < 1e-12);
            }
        }
    }

    // Standalone Bm25LexicalIndex and DenseVectorIndex deterministic tie-breaking checks.
    let mut bm25 = Bm25LexicalIndex::new();
    let mut dense = DenseVectorIndex::new();
    for doc_id in ["z_doc", "a_doc", "m_doc"] {
        let rec = CanonicalEvidenceRecord {
            identity: SourceIdentity {
                doc_id: doc_id.to_string(),
                source_id: "repo:tie".to_string(),
                version_id: "v1".to_string(),
                artifact_kind: SourceArtifactKind::Document,
                tenant_id: "t1".to_string(),
                project_id: "p1".to_string(),
                repo_id: "r1".to_string(),
                allowed_principals: vec![],
                path: "docs/tie.md".to_string(),
                language: "markdown".to_string(),
            },
            title: "identical retrieval token".to_string(),
            raw_text: "identical retrieval token body".to_string(),
            symbols: vec![],
        };
        bm25.upsert_record(&rec);
        let q_doc = meta.format_document("identical retrieval token body");
        let v = embedder.embed_prefixed(&q_doc, IndexOperation::Insert, &meta).unwrap();
        dense.upsert_validated_vector(doc_id.to_string(), v);
    }

    let auth_ids = vec!["z_doc".to_string(), "a_doc".to_string(), "m_doc".to_string()];
    let lex_ties: Vec<String> = bm25
        .search_authorized(&auth_ids, "identical retrieval", &[], 10)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(lex_ties, vec!["a_doc", "m_doc", "z_doc"]);

    let q_prefixed = meta.format_query("identical retrieval token body");
    let q_vec = embedder.embed_prefixed(&q_prefixed, IndexOperation::Query, &meta).unwrap();
    let vec_ties: Vec<String> =
        dense.search_authorized(&auth_ids, &q_vec, 10).into_iter().map(|(id, _)| id).collect();
    assert_eq!(vec_ties, vec!["a_doc", "m_doc", "z_doc"]);
}

#[allow(clippy::too_many_arguments)]
fn make_scoped_record(
    doc_id: &str,
    tenant_id: &str,
    project_id: &str,
    repo_id: &str,
    principals: Vec<&str>,
    path: &str,
    language: &str,
    artifact_kind: SourceArtifactKind,
    node_kind: StructuralNodeKind,
    boost_repeats: usize,
) -> CanonicalEvidenceRecord {
    let repeated_keywords = vec!["resolve_hybrid_route"; boost_repeats].join(" ");
    CanonicalEvidenceRecord {
        identity: SourceIdentity {
            doc_id: doc_id.to_string(),
            source_id: format!("repo:{repo_id}"),
            version_id: "git:issue55".to_string(),
            artifact_kind,
            tenant_id: tenant_id.to_string(),
            project_id: project_id.to_string(),
            repo_id: repo_id.to_string(),
            allowed_principals: principals.into_iter().map(ToString::to_string).collect(),
            path: path.to_string(),
            language: language.to_string(),
        },
        title: format!("Hybrid route handler for {doc_id}"),
        raw_text: format!("{repeated_keywords} hybrid route handler pipeline search dispatch"),
        symbols: vec![StructuralSymbolNode {
            symbol_name: "resolve_hybrid_route".to_string(),
            qualified_name: format!("mod::{doc_id}::resolve_hybrid_route"),
            node_kind,
            signature: "pub fn resolve_hybrid_route(query: &str)".to_string(),
            parent_scope: Some("HybridRouter".to_string()),
            imports: vec![],
            calls: vec![],
            called_by: vec![],
            start_line: 10,
            end_line: 42,
        }],
    }
}

#[test]
fn filter_predicates_enforced_pre_scoring_across_all_scope_dimensions() {
    let meta = canonical_hybrid_embedding_metadata();
    let embedder = DeterministicSemanticCodeEmbedder;
    let planner = DeterministicKaiQueryPlanner;
    let cost = HybridCostModel::default();
    let mut index = HybridSearchIndex::new();
    index.register_collection("scoped", meta.clone()).unwrap();

    // Target record in `tenant_alpha`, `proj_core`, `hanzoai/index`, role `search_dev`,
    // path `crates/search/src/planner.rs`, `rust`, `RepositoryBlob`, `Function`.
    let target = make_scoped_record(
        "target_fn",
        "tenant_alpha",
        "proj_core",
        "hanzoai/index",
        vec!["search_dev"],
        "crates/search/src/planner.rs",
        "rust",
        SourceArtifactKind::RepositoryBlob,
        StructuralNodeKind::Function,
        1,
    );

    // Higher term-frequency distractors (5x keywords) differing in ONE scope dimension each.
    // Under strict `top_k = 1, per_leg_candidate_depth = 1`, any post-scoring filter bug
    // would return 0 hits because these distractors outscore `target_fn` in BM25 & AST.
    let distractors = vec![
        make_scoped_record(
            "wrong_tenant",
            "tenant_beta",
            "proj_core",
            "hanzoai/index",
            vec!["search_dev"],
            "crates/search/src/planner.rs",
            "rust",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_role",
            "tenant_alpha",
            "proj_core",
            "hanzoai/index",
            vec!["admin_only"],
            "crates/search/src/planner.rs",
            "rust",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_project",
            "tenant_alpha",
            "proj_billing",
            "hanzoai/index",
            vec!["search_dev"],
            "crates/search/src/planner.rs",
            "rust",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_repo",
            "tenant_alpha",
            "proj_core",
            "hanzoai/other-repo",
            vec!["search_dev"],
            "crates/search/src/planner.rs",
            "rust",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_path_prefix",
            "tenant_alpha",
            "proj_core",
            "hanzoai/index",
            vec!["search_dev"],
            "crates/other/src/planner.rs",
            "rust",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_language",
            "tenant_alpha",
            "proj_core",
            "hanzoai/index",
            vec!["search_dev"],
            "crates/search/src/planner.rs",
            "python",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_artifact",
            "tenant_alpha",
            "proj_core",
            "hanzoai/index",
            vec!["search_dev"],
            "crates/search/src/planner.rs",
            "rust",
            SourceArtifactKind::GitCommit,
            StructuralNodeKind::Function,
            6,
        ),
        make_scoped_record(
            "wrong_node_kind",
            "tenant_alpha",
            "proj_core",
            "hanzoai/index",
            vec!["search_dev"],
            "crates/search/src/planner.rs",
            "rust",
            SourceArtifactKind::RepositoryBlob,
            StructuralNodeKind::Struct,
            6,
        ),
    ];

    index.upsert_record("scoped", &meta, target, &embedder).unwrap();
    for rec in distractors {
        index.upsert_record("scoped", &meta, rec, &embedder).unwrap();
    }

    let tight_cfg = HybridRrfConfig {
        rrf_k: DEFAULT_RRF_K,
        per_leg_candidate_depth: 1,
        top_k: 1,
        weight_override: None,
    };

    let full_filter =
        HybridAccessFilter::for_tenant("tenant_alpha", vec!["search_dev".to_string()])
            .with_project("proj_core")
            .with_repo("hanzoai/index")
            .with_path_prefix("crates/search/")
            .with_path_glob("crates/**/src/*.rs")
            .with_language("RUST")
            .with_artifact_kind(SourceArtifactKind::RepositoryBlob)
            .with_node_kind(StructuralNodeKind::Function);

    let out = index
        .search_hybrid(
            "scoped",
            &meta,
            &full_filter,
            "resolve_hybrid_route",
            &tight_cfg,
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();

    assert_eq!(out.authorized_candidate_count, 1);
    assert_eq!(out.hits.len(), 1);
    assert_eq!(out.hits[0].doc_id, "target_fn");

    // Verify zero results when caller lacks role or specifies a non-matching glob.
    let no_role_filter = full_filter.clone();
    let denied_filter =
        HybridAccessFilter { caller_principals: vec!["viewer".to_string()], ..no_role_filter };
    let denied_out = index
        .search_hybrid(
            "scoped",
            &meta,
            &denied_filter,
            "resolve_hybrid_route",
            &tight_cfg,
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();
    assert_eq!(denied_out.authorized_candidate_count, 0);
    assert!(denied_out.hits.is_empty());

    let unmatched_glob_filter = full_filter.with_path_glob("sdks/**/*.py");
    let glob_out = index
        .search_hybrid(
            "scoped",
            &meta,
            &unmatched_glob_filter,
            "resolve_hybrid_route",
            &tight_cfg,
            &planner,
            &embedder,
            &cost,
        )
        .unwrap();
    assert!(glob_out.hits.is_empty());
}

#[test]
fn reproducible_topk_recall_mrr_and_wall_clock_latency_benchmark() {
    let (index, meta) = build_canonical_hybrid_index().unwrap();
    let strategies = [
        HybridStrategy::LexicalOnly,
        HybridStrategy::VectorOnly,
        HybridStrategy::StructuralOnly,
        HybridStrategy::UniformRrf { k: DEFAULT_RRF_K },
        HybridStrategy::KaiPlannedHybridRrf { k: DEFAULT_RRF_K },
    ];

    let metrics: Vec<RecallLatencyMetrics> = strategies
        .iter()
        .map(|&s| measure_strategy_recall_and_latency(&index, &meta, s, 2, 10))
        .collect();

    for m in &metrics {
        println!(
            "ISSUE55_EVAL strategy=\"{}\" R@1={:.3} R@3={:.3} R@5={:.3} R@10={:.3} MRR@10={:.3} mean_us={:.1} p50_us={:.1} p95_us={:.1}",
            m.label,
            m.recall_at_1,
            m.recall_at_3,
            m.recall_at_5,
            m.recall_at_10,
            m.mrr_at_10,
            m.mean_latency_us,
            m.p50_latency_us,
            m.p95_latency_us,
        );
        assert!(m.recall_at_1 <= m.recall_at_3 + 1e-12);
        assert!(m.recall_at_3 <= m.recall_at_5 + 1e-12);
        assert!(m.recall_at_5 <= m.recall_at_10 + 1e-12);
        assert!(m.mean_latency_us.is_finite() && m.mean_latency_us > 0.0);
        assert!(m.p95_latency_us >= m.p50_latency_us);
    }

    let lexical = &metrics[0];
    let vector = &metrics[1];
    let uniform_rrf = &metrics[3];
    let kai_rrf = &metrics[4];

    assert!((kai_rrf.recall_at_1 - 1.0).abs() < 1e-9);
    assert!((kai_rrf.mrr_at_10 - 1.0).abs() < 1e-9);
    assert!(kai_rrf.recall_at_1 > lexical.recall_at_1);
    assert!(kai_rrf.recall_at_1 > vector.recall_at_1);
    assert!(kai_rrf.mrr_at_10 > lexical.mrr_at_10);
    assert!(kai_rrf.mrr_at_10 > vector.mrr_at_10);
    assert!(uniform_rrf.mrr_at_10 >= lexical.mrr_at_10);
    assert!(uniform_rrf.mrr_at_10 >= vector.mrr_at_10);
}

#[test]
fn hybrid_rrf_lifts_semantic_paraphrases_while_keeping_exact_symbol_precision() {
    let (index, meta) = build_canonical_hybrid_index().unwrap();
    let planner = DeterministicKaiQueryPlanner;
    let embedder = DeterministicSemanticCodeEmbedder;
    let cost = HybridCostModel::default();
    let filter = HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);
    let hybrid_cfg = HybridRrfConfig::default();

    let mut semantic_lex_hits_at_1 = 0usize;
    let mut semantic_hybrid_hits_at_1 = 0usize;

    for case in canonical_hybrid_queries()
        .into_iter()
        .filter(|c| c.expected_class == HybridQueryClass::NaturalLanguageSemantic)
    {
        let lex_cfg = HybridRrfConfig {
            weight_override: Some(LegWeights { lexical: 1.0, vector: 0.0, structural: 0.0 }),
            ..Default::default()
        };
        let lex_out = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &case.filter,
                &case.raw_query,
                &lex_cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap();
        let hyb_out = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &case.filter,
                &case.raw_query,
                &hybrid_cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap();

        if lex_out.hits.first().is_some_and(|h| case.relevant_doc_ids.contains(&h.doc_id)) {
            semantic_lex_hits_at_1 += 1;
        }
        if hyb_out.hits.first().is_some_and(|h| case.relevant_doc_ids.contains(&h.doc_id)) {
            semantic_hybrid_hits_at_1 += 1;
        }
    }

    assert!(
        semantic_hybrid_hits_at_1 > semantic_lex_hits_at_1,
        "Hybrid RRF ({semantic_hybrid_hits_at_1}/4) must outperform LexicalOnly ({semantic_lex_hits_at_1}/4) on semantic paraphrase queries"
    );

    // Concurrent reads + empty-query guard + strict top_k cap verification.
    let shared_index = Arc::new(index);
    let mut workers = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&shared_index);
        let m = meta.clone();
        let f = filter.clone();
        workers.push(thread::spawn(move || {
            let cfg = HybridRrfConfig { top_k: 3, ..Default::default() };
            let out = idx
                .search_hybrid(
                    "hip1347_hybrid",
                    &m,
                    &f,
                    "MetadataError::SwappedTaskPrefix",
                    &cfg,
                    &DeterministicKaiQueryPlanner,
                    &DeterministicSemanticCodeEmbedder,
                    &HybridCostModel::default(),
                )
                .unwrap();
            assert!(out.hits.len() <= 3);
            let unique_docs: HashSet<&str> = out.hits.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(unique_docs.len(), out.hits.len(), "duplicate doc_ids in RRF hits");
            assert_eq!(
                out.hits[0].doc_id,
                "blob:crates/rag-eval/src/metadata.rs#ImmutableIndexContract"
            );

            let empty_err = idx
                .search_hybrid(
                    "hip1347_hybrid",
                    &m,
                    &f,
                    "   ",
                    &cfg,
                    &DeterministicKaiQueryPlanner,
                    &DeterministicSemanticCodeEmbedder,
                    &HybridCostModel::default(),
                )
                .unwrap_err();
            assert!(matches!(empty_err, HybridSearchError::EmptyQuery));
        }));
    }
    for w in workers {
        w.join().unwrap();
    }
}
