// SPDX-License-Identifier: Apache-2.0

//! # `rag-eval` (`HIP-1346`)
//!
//! Lightweight, pure-Rust, Apache-2.0 evaluation and gating library implementing
//! **HIP-1346** (*Evidence-Grounded Retrieval and Decision Evaluation*) for
//! `hanzoai/index` while strictly preserving the **Kai** (structured non-generative
//! decision head) vs. **Zen** (generative synthesis) architectural boundary.
//!
//! ## Modules
//! - [`metadata`] — HIP-1346 §1 immutable vector index metadata contract
//!   (`model_id`, `model_revision`, `dimensions`, `normalization`, `query_prefix`,
//!   `doc_prefix`, Matryoshka `128`/`256`/`512`/`768` validation).
//! - [`self_rag`] — Wave 1 #8 / Bounty #22 (`hanzoai/index#26`) adaptive
//!   retrieve-or-answer gating via Kai probabilities and calibrated `theta`.
//! - [`corrective`] — Bounty #23 (`hanzoai/index#27`) Corrective RAG detector,
//!   query reformulation, and secondary retrieval recovery.
//! - [`diagnostics`] — Bounty #25 (`hanzoai/index#29`) RAGChecker diagnostics
//!   separating `retrieval_miss`, `irrelevant_evidence`, `unsupported_claim`,
//!   and `answer_omission`.
//! - [`factscore`] — Bounty #30 (`hanzoai/index#31`) atomic claim decomposer,
//!   span evidence binder, supported-claim precision (`FActScore`), and Expected
//!   Calibration Error (`compute_ece`).

pub mod corrective;
pub mod diagnostics;
pub mod factscore;
pub mod metadata;
pub mod self_rag;
mod text;

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
