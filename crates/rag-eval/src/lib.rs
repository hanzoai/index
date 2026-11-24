// SPDX-License-Identifier: Apache-2.0

//! # `rag-eval` (`HIP-1346`)
//!
//! Lightweight, pure-Rust, Apache-2.0 evaluation and gating library implementing
//! **HIP-1346** (*Evidence-Grounded Retrieval and Decision Evaluation*) for
//! `hanzoai/index` while strictly preserving the **Kai** (structured non-generative
//! decision head) vs. **Zen** (generative synthesis) architectural boundary.
//!
//! ## Modules
//! - [`ann_insert`] — Bounty #21 (`hanzoai/index#19`) stable online ANN
//!   insertion (OdinANN): maintenance-policy benchmark over a file-backed
//!   on-disk ANN under continuous query traffic with shared brute-force
//!   `recall@10` ground truth and immutable-metadata gating.
//! - [`metadata`] — HIP-1346 §1 immutable vector index metadata contract
//!   (`model_id`, `model_revision`, `dimensions`, `normalization`, `query_prefix`,
//!   `doc_prefix`, Matryoshka `128`/`256`/`512`/`768` validation).
//! - [`self_rag`] — Wave 1 #8 / Bounty #22 (`hanzoai/index#26`) adaptive
//!   retrieve-or-answer gating via Kai probabilities and calibrated `theta`.
//! - [`corrective`] — Bounty #23 (`hanzoai/index#27`) Corrective RAG detector,
//!   query reformulation, and secondary retrieval recovery.
//! - [`hyde`] - Bounty #24 (`hanzoai/index#28`) Hypothetical-Document Retrieval
//!   (`HyDE`) with Zen synthesis, Kai selective routing, and immutable
//!   `doc_prefix` / `query_prefix` enforcement.
//! - [`diagnostics`] — Bounty #25 (`hanzoai/index#29`) RAGChecker diagnostics
//!   separating `retrieval_miss`, `irrelevant_evidence`, `unsupported_claim`,
//!   and `answer_omission`.
//! - [`factscore`] — Bounty #30 (`hanzoai/index#31`) atomic claim decomposer,
//!   span evidence binder, supported-claim precision (`FActScore`), and Expected
//!   Calibration Error (`compute_ece`).

//! - [`tiered_rerank`] - Bounty #14 (`hanzoai/index#14`, `HIP-1347`) 3-stage
//!   cascaded cross-encoder reranking with pre-stage tenant/ACL isolation, Kai
//!   margin/entropy escalation gating, and teacher-to-student distillation.

//! - [`hybrid_rrf`] — HIP-1347 p0 (`hanzoai/index#11` / Roadmap `#16` Item #1)
//!   Hybrid Lexical, Vector & Structural Search Planner with Reciprocal Rank
//!   Fusion (`RRF`), mandatory pre-scoring ACL/scope isolation, and derived
//!   index rebuild verification.

//! - [`domain_embeddings`] - Bounty #13 (`hanzoai/index#13`, `HIP-1347`)
//!   comparative harness for ModernBERT/Laya, Qwen3 Embedding (`0.6B`/`4B`/`8B`),
//!   BGE-M3 (dense + sparse + multi-vector), and distilled code/memory retrievers.

pub mod ann_insert;
pub mod corrective;
pub mod diagnostics;
pub mod factscore;
pub mod hyde;
pub mod metadata;
pub mod hybrid_rrf;
pub mod domain_embeddings;
pub mod self_rag;
mod text;
pub mod tiered_rerank;

pub use ann_insert::{
    generate_workload, run_full_matrix, write_results_json, AnnInsertError, AnnMaintenanceRunner,
    BenchConfig, GraphAnn, IoCostModel, MaintenancePolicy, Op, RunResult, RunStats, SplitMix64,
    Workload,
};
pub use corrective::{
    CorrectiveAction, CorrectiveConfig, CorrectiveDiagnostics, CorrectiveError, CorrectivePlan,
    CorrectiveRetrievalPolicy, CorrectiveRunResult, EvidenceGrade, KeywordQueryReformulator,
    QueryReformulator, RejectedCandidate, RejectionReason, RetrievedCandidate, SecondaryRetriever,
};
pub use diagnostics::{
    DiagnosticFinding, DiagnosticsError, RagCaseReport, RagChecker, RagDiagnosticCategory,
    RagDiagnosticSummary, RagEvalCase, RetrievedChunk,
};
pub use factscore::{
    compute_ece, AtomicClaim, AtomicClaimDecomposer, CalibrationBin, CalibrationReport,
    CalibrationSample, ClaimVerdict, ClaimVerdictStatus, EvidenceDocument, EvidenceSpan,
    FactScoreError, FactScoreEvaluator, FactScoreReport, KaiClaimVerification, SpanEvidenceBinder,
    VerifierSource,
};
pub use metadata::{
    truncate_matryoshka, ImmutableIndexContract, IndexEmbeddingMetadata, IndexOperation,
    MetadataError, MetadataField, MetadataMismatch, NormalizationPolicy, L2_NORM_TOLERANCE,
    MATRYOSHKA_DIMS,
};
pub use self_rag::{
    route_with_kai_and_zen, select_optimal_theta, simulate_gate, sweep_thresholds, DecisionHead,
    GateError, GateSimulationReport, GateSimulationSample, KaiDecisionHead, KaiRetrieveProbs,
    RetrievalAction, RetrievalCostModel, RetrieveGateDecision, RetrieveOrAnswerGate,
    RoutedAnswer, SynthesisTarget, ZenSynthesizer,
};

pub use hyde::{
    evaluate_hyde_benchmark, CallerAccessContext, DeterministicHashedEmbedder,
    DocumentOrigin, HydeBenchmarkCase, HydeBenchmarkStrategy, HydeConfig,
    HydeCostModel, HydeDocument, HydeError, HydeIndex, HydeQuerySlice,
    HydeRouteAction, HydeScoredHit, HydeSearchOutcome, HydeStrategyReport,
    KaiHydeClassifier, KaiHydeDecision, KaiHydeGate, KaiHydeProbs,
    PrefixedTextEmbedder, StructuralKaiHydeClassifier, ZenHypotheticalPassage,
    ZenHypothesisGenerator,
};

pub use tiered_rerank::{
    default_distillation_training_set, default_multi_tenant_benchmark, enforce_tenant_acl,
    evaluate_policy, evaluate_tiered_reranking, extract_pair_features, pre_stage_access_filter,
    rebuild_stage1_candidates, AccessContext, AccessDenialReason, AccessFilteredAudit,
    BenchmarkQueryCase, BroadStageCandidate, CalibratedFastCrossEncoder, CanonicalPassageRecord,
    DeterministicZenGrounded, DistillationConfig, DistillationReport, DistillationTrainer,
    DistilledDomainReranker, EscalationAction, EscalationDecision, EscalationGate,
    FastCrossEncoder, HeavyCrossEncoder, PassageIdentity, PolicyEvaluationReport, QueryDifficulty,
    RankedPassage, RerankPolicy, SourceKind, SpanAlignedHeavyCrossEncoder,
    StudentDistillationLift, TeacherJudgment, TieredBenchmarkSummary, TieredCostModel,
    TieredQueryRun, TieredRerankConfig, TieredRerankError, TieredRerankPipeline,
    DISTILL_FEATURE_DIM, DISTILL_HIDDEN_DIM,
};

pub use hybrid_rrf::{
    build_canonical_hybrid_index, canonical_hybrid_corpus, canonical_hybrid_embedding_metadata,
    canonical_hybrid_queries, compare_search_hits_deterministic, evaluate_hybrid_benchmark,
    extract_code_identifiers, glob_matches, is_record_authorized_and_in_scope,
    split_identifier_subtokens, tokenize_code_and_prose, Bm25LexicalIndex,
    CanonicalEvidenceRecord, DenseVectorIndex, DeterministicCodeEditZenSynthesizer,
    DeterministicKaiQueryPlanner, DeterministicSemanticCodeEmbedder, DownstreamCodeEditJudge,
    HybridAccessFilter, HybridBenchmarkCase, HybridBenchmarkSuiteReport, HybridCostModel,
    HybridQueryClass, HybridQueryPlan, HybridRrfConfig, HybridSearchError, HybridSearchIndex,
    HybridSearchOutcome, HybridStrategy, HybridStrategyReport, KaiHybridQueryPlanner,
    LegContribution, LegWeights, RankExplanation, ReciprocalRankFusion, SearchHit,
    SourceArtifactKind, SourceIdentity, StructuralAstIndex, StructuralNodeKind,
    StructuralSymbolNode, TokenizedCodeSpan, DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_RRF_K,
};

pub use domain_embeddings::{
    check_record_access, default_domain_benchmark, encode_text, evaluate_encoder,
    standard_encoder_specs, ArtifactKind, CanonicalRecord, DerivedDomainIndex,
    DerivedEmbeddingEntry, DistillationJustificationDecision, DistillationJustificationGate,
    DomainAccessContext, DomainEmbeddingError, DomainQueryCase, DomainSlice, EncoderEvaluationReport,
    EncoderFamily, EncoderSpec, ModelLicenseAudit, SourceRecordIdentity,
};
