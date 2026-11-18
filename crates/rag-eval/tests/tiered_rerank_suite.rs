// SPDX-License-Identifier: Apache-2.0

//! Integration and Pareto verification suite for `hanzoai/index#14` (`HIP-1346` / `HIP-1347`):
//! Tiered Cross-Encoder Reranking and Retrieval Distillation.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use rag_eval::{
    default_distillation_training_set, default_multi_tenant_benchmark, enforce_tenant_acl,
    evaluate_policy, evaluate_tiered_reranking, pre_stage_access_filter,
    rebuild_stage1_candidates, AccessContext, AccessDenialReason, BenchmarkQueryCase,
    BroadStageCandidate, CalibratedFastCrossEncoder, CanonicalPassageRecord, DecisionHead,
    DeterministicZenGrounded, DistillationTrainer, DistilledDomainReranker, EscalationAction,
    EscalationGate, FastCrossEncoder, PassageIdentity, QueryDifficulty, RerankPolicy, SourceKind,
    SpanAlignedHeavyCrossEncoder, TeacherJudgment, TieredCostModel, TieredRerankConfig,
    TieredRerankError, TieredRerankPipeline,
};

fn sample_candidate(
    passage_id: &str,
    tenant_id: &str,
    roles: &[&str],
    text: &str,
    score: f64,
) -> BroadStageCandidate {
    BroadStageCandidate {
        identity: PassageIdentity {
            source_id: format!("src_{passage_id}"),
            passage_id: passage_id.to_string(),
            version_id: format!("sha256:{passage_id}_v1"),
            source_kind: SourceKind::Document,
            tenant_id: tenant_id.to_string(),
            allowed_roles: roles.iter().map(|r| (*r).to_string()).collect(),
        },
        text: text.to_string(),
        stage1_raw_score: score,
    }
}

#[test]
fn tenant_and_role_acl_fail_closed_before_stage1_scoring() {
    let access = AccessContext {
        tenant_id: "tenant_acme".to_string(),
        principal_roles: vec!["search_reader".to_string()],
    };
    let raw = vec![
        sample_candidate(
            "cross_tenant_hit",
            "tenant_globex",
            &[],
            "Exact answer leaked from another tenant.",
            0.99,
        ),
        sample_candidate(
            "unauth_role_hit",
            "tenant_acme",
            &["security_admin"],
            "Restricted executive runbook.",
            0.98,
        ),
        sample_candidate(
            "corrupt_mojibake",
            "tenant_acme",
            &[],
            "Corrupted \u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD} payload",
            0.97,
        ),
        sample_candidate(
            "nan_score",
            "tenant_acme",
            &[],
            "Valid text with non-finite Stage-1 score.",
            f64::NAN,
        ),
        sample_candidate(
            "authorized_hit",
            "tenant_acme",
            &["search_reader"],
            "Authorized snapshot crc32c manifest specification.",
            0.70,
        ),
    ];

    let (kept, audit) = pre_stage_access_filter(&access, &raw, 0.05).unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].identity.passage_id, "authorized_hit");
    assert_eq!(audit.len(), 4);
    assert_eq!(audit[0].reason, AccessDenialReason::CrossTenantMismatch);
    assert_eq!(audit[1].reason, AccessDenialReason::InsufficientRole);
    assert_eq!(audit[2].reason, AccessDenialReason::CorruptedOrInvalidScore);
    assert_eq!(audit[3].reason, AccessDenialReason::CorruptedOrInvalidScore);
}

#[test]
fn empty_or_control_char_access_context_fails_closed() {
    let bad_empty = AccessContext { tenant_id: "   ".to_string(), principal_roles: Vec::new() };
    let bad_ctrl =
        AccessContext { tenant_id: "tenant\0acme".to_string(), principal_roles: Vec::new() };
    let cands = vec![sample_candidate("p1", "tenant_acme", &[], "clean text", 0.8)];

    assert_eq!(
        pre_stage_access_filter(&bad_empty, &cands, 0.05),
        Err(TieredRerankError::InvalidAccessContext)
    );
    assert_eq!(
        pre_stage_access_filter(&bad_ctrl, &cands, 0.05),
        Err(TieredRerankError::InvalidAccessContext)
    );

    let mut malformed_cand = cands[0].clone();
    malformed_cand.identity.version_id = String::new();
    let valid_ctx =
        AccessContext { tenant_id: "tenant_acme".to_string(), principal_roles: Vec::new() };
    assert_eq!(
        enforce_tenant_acl(&valid_ctx, &malformed_cand),
        Err(AccessDenialReason::MalformedIdentity)
    );
}

#[test]
fn derived_stage1_rebuild_preserves_immutable_passage_identity_and_version() {
    let records = vec![
        CanonicalPassageRecord {
            identity: PassageIdentity {
                source_id: "repo:hanzo/index:milli/lib.rs".to_string(),
                passage_id: "chunk_01".to_string(),
                version_id: "git:331c4f168e4b".to_string(),
                source_kind: SourceKind::RepositoryBlob,
                tenant_id: "tenant_acme".to_string(),
                allowed_roles: vec!["index_ops".to_string()],
            },
            text: "Roaring bitmap posting lists compress document deltas with zstd.".to_string(),
        },
        CanonicalPassageRecord {
            identity: PassageIdentity {
                source_id: "decision:kai:gate_v2".to_string(),
                passage_id: "chunk_02".to_string(),
                version_id: "rev:00042".to_string(),
                source_kind: SourceKind::StoredDecision,
                tenant_id: "tenant_acme".to_string(),
                allowed_roles: Vec::new(),
            },
            text: "Kai emits non-generative discrete escalation probabilities.".to_string(),
        },
    ];

    let rebuilt_a = rebuild_stage1_candidates(&records, "roaring bitmap zstd posting lists");
    let rebuilt_b = rebuild_stage1_candidates(&records, "Kai escalation probabilities");

    assert_eq!(rebuilt_a[0].identity, records[0].identity);
    assert_eq!(rebuilt_b[0].identity, records[0].identity);
    assert_eq!(rebuilt_a[1].identity, records[1].identity);
    assert!(rebuilt_a[0].stage1_raw_score > rebuilt_a[1].stage1_raw_score);
    assert!(rebuilt_b[1].stage1_raw_score > rebuilt_b[0].stage1_raw_score);
}

#[test]
fn kai_escalation_gate_calibrates_margin_entropy_and_score_floor() {
    let gate = EscalationGate::default();

    // Easy confident Stage-2 profile: high top-1, wide margin, low entropy.
    let easy = gate.evaluate(&[0.86, 0.31, 0.22, 0.15]);
    assert_eq!(easy.decision_head, DecisionHead::Kai);
    assert_eq!(easy.action, EscalationAction::ResolveAtStage2);
    assert!((easy.p_escalate + easy.p_resolve_stage2 - 1.0).abs() < 1e-12);
    assert!(easy.top1_top2_margin > gate.min_margin);
    assert!(easy.normalized_entropy < gate.max_normalized_entropy);
    assert!(easy.p_resolve_stage2 > 0.50);

    // Ambiguous near-tie profile: narrow margin + high entropy triggers Stage-3 escalation.
    let hard = gate.evaluate(&[0.74, 0.71, 0.69, 0.66]);
    assert_eq!(hard.decision_head, DecisionHead::Kai);
    assert_eq!(hard.action, EscalationAction::EscalateToStage3);
    assert!((hard.p_escalate + hard.p_resolve_stage2 - 1.0).abs() < 1e-12);
    assert!(hard.top1_top2_margin < gate.min_margin);
    assert!(hard.normalized_entropy > gate.max_normalized_entropy);
    assert!(hard.p_escalate > 0.50);

    // Low score floor profile: wide relative gap on weak scores still escalates.
    let weak = gate.evaluate(&[0.42, 0.10, 0.05]);
    assert_eq!(weak.action, EscalationAction::EscalateToStage3);
}

#[test]
fn kai_escalation_gate_validates_configuration_bounds() {
    assert!(EscalationGate::new(-0.1, 0.15, 0.75, 4, 0.15).is_err());
    assert!(EscalationGate::new(0.58, 1.2, 0.75, 4, 0.15).is_err());
    assert!(EscalationGate::new(0.58, 0.15, 0.75, 1, 0.15).is_err());
    assert!(EscalationGate::new(0.58, 0.15, 0.75, 4, 0.0).is_err());
    assert!(EscalationGate::new(0.58, 0.15, 0.75, 4, f64::NAN).is_err());
}

#[test]
fn stage3_candidate_cap_and_token_budget_are_strictly_enforced() {
    let cfg = TieredRerankConfig {
        top_k_broad: 6,
        final_top_k: 4,
        max_stage3_candidates: 2,
        max_stage3_rerank_tokens: 45,
        max_corruption_ratio: 0.05,
    };
    let pipeline =
        TieredRerankPipeline::new(cfg, EscalationGate::default(), TieredCostModel::default())
            .unwrap();
    let access =
        AccessContext { tenant_id: "tenant_acme".to_string(), principal_roles: Vec::new() };
    let query = "Why must replica nodes not execute log truncation?";
    let cands = vec![
        sample_candidate(
            "c1",
            "tenant_acme",
            &[],
            "Replica nodes execute log truncation during local flushes.",
            0.92,
        ),
        sample_candidate(
            "c2",
            "tenant_acme",
            &[],
            "Replica nodes must never execute log truncation without leader commit.",
            0.88,
        ),
        sample_candidate(
            "c3",
            "tenant_acme",
            &[],
            "Replica nodes replicate log entries over TLS streams.",
            0.85,
        ),
        sample_candidate(
            "c4",
            "tenant_acme",
            &[],
            "Log truncation checkpoints segment headers.",
            0.80,
        ),
    ];

    let run = pipeline
        .run_query(
            RerankPolicy::TieredCascade,
            &access,
            query,
            &cands,
            &CalibratedFastCrossEncoder::default(),
            &SpanAlignedHeavyCrossEncoder::default(),
        )
        .unwrap();

    assert!(run.escalated_to_stage3);
    assert!(run.stage3_rerank_tokens <= cfg.max_stage3_rerank_tokens);
    let stage3_scored_count =
        run.full_ranked_pool.iter().filter(|p| p.stage3_score.is_some()).count();
    assert!(stage3_scored_count <= cfg.max_stage3_candidates);
    assert!(stage3_scored_count >= 1);
}

#[test]
fn degenerate_empty_single_and_tie_candidate_pools_behave_deterministically() {
    let pipeline = TieredRerankPipeline::default();
    let access =
        AccessContext { tenant_id: "tenant_acme".to_string(), principal_roles: Vec::new() };
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();

    // Empty candidate pool.
    let empty_run = pipeline
        .run_query(
            RerankPolicy::TieredCascade,
            &access,
            "crc32c manifest",
            &[],
            &fast_ce,
            &heavy_ce,
        )
        .unwrap();
    assert!(empty_run.ranked_passages.is_empty());
    assert_eq!(empty_run.total_rerank_tokens, 0);
    assert!(!empty_run.escalated_to_stage3);

    // Single high-confidence candidate resolves at Stage 2.
    let single = vec![sample_candidate(
        "only_doc",
        "tenant_acme",
        &[],
        "Snapshot manifest segments validate integrity with crc32c checksums.",
        0.85,
    )];
    let single_run = pipeline
        .run_query(
            RerankPolicy::TieredCascade,
            &access,
            "What checksum validates snapshot manifest segments?",
            &single,
            &fast_ce,
            &heavy_ce,
        )
        .unwrap();
    assert_eq!(single_run.ranked_passages.len(), 1);
    assert!(!single_run.escalated_to_stage3);

    // Exact Stage-1 & Stage-2 score tie breaks deterministically by passage_id.
    let ties = vec![
        sample_candidate("doc_b", "tenant_acme", &[], "identical passage text", 0.80),
        sample_candidate("doc_a", "tenant_acme", &[], "identical passage text", 0.80),
    ];
    let tie_run = pipeline
        .run_query(
            RerankPolicy::TieredCascade,
            &access,
            "identical passage",
            &ties,
            &fast_ce,
            &heavy_ce,
        )
        .unwrap();
    assert_eq!(tie_run.ranked_passages[0].identity.passage_id, "doc_a");
    assert_eq!(tie_run.ranked_passages[1].identity.passage_id, "doc_b");
}

#[test]
fn distillation_rejects_cross_tenant_contaminated_teacher_judgments() {
    let trainer = DistillationTrainer::default();
    let contaminated = vec![TeacherJudgment {
        query_id: "bad_jdg".to_string(),
        tenant_id: "tenant_acme".to_string(),
        query: "crc32c checksum".to_string(),
        candidates: vec![
            sample_candidate("ok", "tenant_acme", &[], "crc32c checksum", 0.8),
            sample_candidate("leak", "tenant_globex", &[], "foreign tenant doc", 0.9),
        ],
        stage1_norm_scores: vec![0.0, 1.0],
        teacher_scores: vec![0.9, 0.1],
    }];

    let err = trainer.distill(&contaminated).unwrap_err();
    assert_eq!(
        err,
        TieredRerankError::AccessDenied {
            passage_id: "leak".to_string(),
            reason: AccessDenialReason::CrossTenantMismatch,
        }
    );
}

#[test]
fn distillation_monotonically_reduces_loss_and_lifts_unseen_student_ndcg_mrr_recall() {
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();
    let zen = DeterministicZenGrounded;
    let pipeline = TieredRerankPipeline::default();

    let train_judgments = default_distillation_training_set(&fast_ce, &heavy_ce);
    let eval_cases = default_multi_tenant_benchmark();

    // Ensure zero query overlap between teacher training set and 24-query evaluation benchmark.
    for tr in &train_judgments {
        assert!(eval_cases.iter().all(|ev| ev.query != tr.query && ev.query_id != tr.query_id));
    }

    let untrained = DistilledDomainReranker::untrained();
    let pre = evaluate_policy(
        &pipeline,
        RerankPolicy::AlwaysFastStage2,
        &eval_cases,
        &untrained,
        &heavy_ce,
        &zen,
    )
    .unwrap();

    let (distilled, report) = DistillationTrainer::default().distill(&train_judgments).unwrap();
    assert!(report.monotonic_non_increasing);
    assert!(report.final_loss < report.initial_loss * 0.25);

    let post = evaluate_policy(
        &pipeline,
        RerankPolicy::AlwaysFastStage2,
        &eval_cases,
        &distilled,
        &heavy_ce,
        &zen,
    )
    .unwrap();

    assert!(post.ndcg_at_5 > pre.ndcg_at_5 + 0.30);
    assert!(post.mrr > pre.mrr + 0.35);
    assert!(post.recall_at_5 > pre.recall_at_5 + 0.10);
    assert!(post.answer_accuracy > pre.answer_accuracy + 0.50);
    assert!((post.ndcg_at_5 - 1.0).abs() < 1e-9);
}

#[test]
fn tiered_cascade_cuts_compute_over_55_pct_while_matching_heavy_accuracy() {
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();
    let train = default_distillation_training_set(&fast_ce, &heavy_ce);
    let cases = default_multi_tenant_benchmark();
    let pipeline = TieredRerankPipeline::default();

    let summary = evaluate_tiered_reranking(&pipeline, &train, &cases).unwrap();

    // Mandatory zero cross-tenant leakage across all 5 policies (48 adversarial probes each).
    for rep in [
        &summary.stage1_only,
        &summary.always_fast_stage2,
        &summary.always_heavy_stage3,
        &summary.tiered_cascade,
        &summary.distilled_student_cascade,
    ] {
        assert_eq!(rep.cross_tenant_leaks, 0);
        assert_eq!(rep.acl_filtered_candidates, 48);
    }

    // TieredCascade saves >= 55% compute while matching >= 98% (100%) heavy accuracy.
    assert!(summary.tiered_compute_savings_pct >= 55.0);
    assert!(
        summary.tiered_cascade.answer_accuracy
            >= 0.98 * summary.always_heavy_stage3.answer_accuracy
    );
    assert!((summary.tiered_cascade.ndcg_at_5 - 1.0).abs() < 1e-9);
    assert!(
        summary.tiered_cascade.p50_latency_ms < summary.always_heavy_stage3.p50_latency_ms * 0.25
    );
    assert!(
        summary.tiered_cascade.cost_per_correct_passage_usd
            < summary.always_heavy_stage3.cost_per_correct_passage_usd
    );

    // DistilledStudentCascade saves >= 85% compute while strictly beating AlwaysFastStage2 accuracy
    // and achieving the lowest cost per correctly retrieved passage across all 5 systems.
    assert!(summary.distilled_compute_savings_pct >= 85.0);
    assert!(
        summary.distilled_student_cascade.answer_accuracy
            > summary.always_fast_stage2.answer_accuracy
    );
    assert!(
        summary.distilled_student_cascade.answer_accuracy
            >= 0.98 * summary.always_heavy_stage3.answer_accuracy
    );
    assert!(
        summary.distilled_student_cascade.cost_per_correct_passage_usd
            < summary.stage1_only.cost_per_correct_passage_usd
    );
    assert!(
        summary.distilled_student_cascade.cost_per_correct_passage_usd
            < summary.always_fast_stage2.cost_per_correct_passage_usd
    );

    println!(
        "TIERED_RERANK_BENCHMARK: \
         stage1(ndcg5={:.3}, r5={:.3}, mrr={:.3}, acc={:.3}, p50={:.2}ms, p95={:.2}ms, comp={:.1}, $/corr=${:.6}) | \
         fast_s2(ndcg5={:.3}, r5={:.3}, mrr={:.3}, acc={:.3}, p50={:.2}ms, p95={:.2}ms, comp={:.1}, $/corr=${:.6}) | \
         heavy_s3(ndcg5={:.3}, r5={:.3}, mrr={:.3}, acc={:.3}, p50={:.2}ms, p95={:.2}ms, comp={:.1}, $/corr=${:.6}) | \
         tiered(ndcg5={:.3}, r5={:.3}, mrr={:.3}, acc={:.3}, p50={:.2}ms, p95={:.2}ms, s3_toks={}, comp={:.1} [-{:.2}%], esc={:.1}%, $/corr=${:.6}) | \
         distilled_cascade(ndcg5={:.3}, r5={:.3}, mrr={:.3}, acc={:.3}, p50={:.2}ms, p95={:.2}ms, s3_toks={}, comp={:.1} [-{:.2}%], esc={:.1}%, $/corr=${:.6}) | \
         distill_loss({:.4}->{:.4})",
        summary.stage1_only.ndcg_at_5,
        summary.stage1_only.recall_at_5,
        summary.stage1_only.mrr,
        summary.stage1_only.answer_accuracy,
        summary.stage1_only.p50_latency_ms,
        summary.stage1_only.p95_latency_ms,
        summary.stage1_only.compute_units,
        summary.stage1_only.cost_per_correct_passage_usd,
        summary.always_fast_stage2.ndcg_at_5,
        summary.always_fast_stage2.recall_at_5,
        summary.always_fast_stage2.mrr,
        summary.always_fast_stage2.answer_accuracy,
        summary.always_fast_stage2.p50_latency_ms,
        summary.always_fast_stage2.p95_latency_ms,
        summary.always_fast_stage2.compute_units,
        summary.always_fast_stage2.cost_per_correct_passage_usd,
        summary.always_heavy_stage3.ndcg_at_5,
        summary.always_heavy_stage3.recall_at_5,
        summary.always_heavy_stage3.mrr,
        summary.always_heavy_stage3.answer_accuracy,
        summary.always_heavy_stage3.p50_latency_ms,
        summary.always_heavy_stage3.p95_latency_ms,
        summary.always_heavy_stage3.compute_units,
        summary.always_heavy_stage3.cost_per_correct_passage_usd,
        summary.tiered_cascade.ndcg_at_5,
        summary.tiered_cascade.recall_at_5,
        summary.tiered_cascade.mrr,
        summary.tiered_cascade.answer_accuracy,
        summary.tiered_cascade.p50_latency_ms,
        summary.tiered_cascade.p95_latency_ms,
        summary.tiered_cascade.stage3_rerank_tokens,
        summary.tiered_cascade.compute_units,
        summary.tiered_compute_savings_pct,
        summary.tiered_cascade.stage3_escalation_rate * 100.0,
        summary.tiered_cascade.cost_per_correct_passage_usd,
        summary.distilled_student_cascade.ndcg_at_5,
        summary.distilled_student_cascade.recall_at_5,
        summary.distilled_student_cascade.mrr,
        summary.distilled_student_cascade.answer_accuracy,
        summary.distilled_student_cascade.p50_latency_ms,
        summary.distilled_student_cascade.p95_latency_ms,
        summary.distilled_student_cascade.stage3_rerank_tokens,
        summary.distilled_student_cascade.compute_units,
        summary.distilled_compute_savings_pct,
        summary.distilled_student_cascade.stage3_escalation_rate * 100.0,
        summary.distilled_student_cascade.cost_per_correct_passage_usd,
        summary.distillation.initial_loss,
        summary.distillation.final_loss,
    );
}

#[test]
fn easy_queries_resolve_at_stage2_while_ambiguous_and_hard_escalate() {
    let pipeline = TieredRerankPipeline::default();
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();
    let cases = default_multi_tenant_benchmark();

    let mut easy_escalations = 0usize;
    let mut hard_or_ambig_escalations = 0usize;
    let mut hard_or_ambig_total = 0usize;

    for case in &cases {
        let run = pipeline
            .run_query(
                RerankPolicy::TieredCascade,
                &case.access,
                &case.query,
                &case.raw_candidates,
                &fast_ce,
                &heavy_ce,
            )
            .unwrap();
        match case.difficulty {
            QueryDifficulty::Easy => {
                if run.escalated_to_stage3 {
                    easy_escalations += 1;
                }
            }
            QueryDifficulty::Ambiguous | QueryDifficulty::HardMultiHop => {
                hard_or_ambig_total += 1;
                if run.escalated_to_stage3 {
                    hard_or_ambig_escalations += 1;
                }
            }
        }
    }

    assert_eq!(easy_escalations, 0);
    assert_eq!(hard_or_ambig_escalations, hard_or_ambig_total);
}

#[test]
fn right_file_wrong_answer_polarity_is_rejected_by_downstream_task_judge() {
    // Demonstrates the shared program rule: ranking a document that fails to help
    // the downstream agent complete the factual task is scored as task failure.
    let pipeline = TieredRerankPipeline::default();
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();
    let zen = DeterministicZenGrounded;

    let mut grades = HashMap::new();
    grades.insert("misleading_doc".to_string(), 3u8);
    let case = BenchmarkQueryCase {
        query_id: "judge_control".to_string(),
        difficulty: QueryDifficulty::Ambiguous,
        access: AccessContext {
            tenant_id: "tenant_acme".to_string(),
            principal_roles: Vec::new(),
        },
        query: "Why must replica nodes not truncate write-ahead logs?".to_string(),
        raw_candidates: vec![sample_candidate(
            "misleading_doc",
            "tenant_acme",
            &[],
            "Replica nodes truncate write-ahead logs immediately.",
            0.95,
        )],
        relevance_grades: grades,
        required_answer_facts: vec!["truncate".to_string()],
    };

    let rep =
        evaluate_policy(&pipeline, RerankPolicy::Stage1Only, &[case], &fast_ce, &heavy_ce, &zen)
            .unwrap();
    assert!((rep.recall_at_5 - 1.0).abs() < 1e-9);
    assert!((rep.answer_accuracy - 0.0).abs() < 1e-9);
}

#[test]
fn serde_json_roundtrip_preserves_distilled_student_and_summary_metrics() {
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();
    let train = default_distillation_training_set(&fast_ce, &heavy_ce);
    let (student, _) = DistillationTrainer::default().distill(&train).unwrap();

    let encoded = serde_json::to_string(&student).unwrap();
    let restored: DistilledDomainReranker = serde_json::from_str(&encoded).unwrap();
    assert_eq!(student, restored);
    let dyn_student: &dyn FastCrossEncoder = &restored;
    assert_eq!(dyn_student.model_id(), "hanzo/distilled-domain-student-mlp-v1");
    assert!((dyn_student.compute_multiplier() - 0.35).abs() < 1e-12);

    let cand = sample_candidate(
        "c1",
        "tenant_acme",
        &[],
        "Protocol 2.4 triggers step-down at 350 milliseconds.",
        0.8,
    );
    let s_orig = student.score_pair("What timeout triggers step-down in 2.4?", &cand, 0.7);
    let s_rest = restored.score_pair("What timeout triggers step-down in 2.4?", &cand, 0.7);
    assert!((s_orig - s_rest).abs() < 1e-15);
}

#[test]
fn concurrent_multithreaded_queries_preserve_strict_tenant_isolation() {
    let pipeline = Arc::new(TieredRerankPipeline::default());
    let fast_ce = Arc::new(CalibratedFastCrossEncoder::default());
    let heavy_ce = Arc::new(SpanAlignedHeavyCrossEncoder::default());
    let cases = Arc::new(default_multi_tenant_benchmark());

    let mut handles = Vec::with_capacity(8);
    for worker_id in 0..8usize {
        let p = Arc::clone(&pipeline);
        let f = Arc::clone(&fast_ce);
        let h = Arc::clone(&heavy_ce);
        let c = Arc::clone(&cases);
        handles.push(thread::spawn(move || {
            for idx in 0..3usize {
                let case = &c[(worker_id * 3 + idx) % c.len()];
                let run = p
                    .run_query(
                        RerankPolicy::TieredCascade,
                        &case.access,
                        &case.query,
                        &case.raw_candidates,
                        f.as_ref(),
                        h.as_ref(),
                    )
                    .unwrap();
                assert!(!run.ranked_passages.is_empty());
                assert_eq!(run.ranked_passages[0].identity.passage_id, "gold_primary");
                for hit in &run.ranked_passages {
                    assert_eq!(hit.identity.tenant_id, case.access.tenant_id);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
}
