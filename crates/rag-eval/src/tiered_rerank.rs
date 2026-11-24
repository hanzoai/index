// SPDX-License-Identifier: Apache-2.0

//! Bounty #14 (`hanzoai/index#14`, `HIP-1346` / `HIP-1347`): Tiered Cross-Encoder
//! Reranking and Retrieval Distillation.
//!
//! Enforces three core architectural invariants:
//! 1. **Immutable Source Identity & Pre-Stage Tenant/ACL Isolation**: Every source
//!    record carries a stable `(source_id, passage_id, version_id, source_kind)` and
//!    mandatory `(tenant_id, allowed_roles)` access boundary evaluated **before**
//!    Stage 1, Stage 2, Stage 3, and teacher-to-student distillation.
//! 2. **Kai vs. Zen Split**: `EscalationGate` operates strictly under
//!    [`DecisionHead::Kai`] (non-generative margin, normalized Shannon entropy, and
//!    score-floor calibration) to resolve easy queries at Stage 2 and escalate only
//!    hard or ambiguous queries to Stage 3; [`ZenSynthesizer`] owns grounded answer
//!    synthesis over the final ranked evidence.
//! 3. **Teacher-to-Student Distillation**: [`DistillationTrainer`] distills Stage-3
//!    teacher judgments (temperature-scaled KL divergence, pairwise margin ranking,
//!    and calibrated pointwise cross-entropy) into a compact domain student
//!    ([`DistilledDomainReranker`]).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::self_rag::{DecisionHead, SynthesisTarget, ZenSynthesizer};
use crate::text::{corruption_ratio, extract_numbers, is_negated, tokenize_content, tokenize_raw};

/// Dimensionality of the generalizable cross-interaction feature vector.
pub const DISTILL_FEATURE_DIM: usize = 10;

/// Hidden width of the compact residual domain student MLP.
pub const DISTILL_HIDDEN_DIM: usize = 6;

/// Errors returned by tiered reranking configuration, access control, or distillation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TieredRerankError {
    #[error("access context `tenant_id` must be non-empty and free of control characters")]
    InvalidAccessContext,
    #[error("invalid tiered rerank configuration `{field}`={value}: {reason}")]
    InvalidConfig { field: &'static str, value: f64, reason: &'static str },
    #[error("access denied for passage `{passage_id}`: {reason:?}")]
    AccessDenied { passage_id: String, reason: AccessDenialReason },
    #[error("distillation requires at least one valid teacher judgment")]
    EmptyDistillationDataset,
    #[error("teacher judgment `{query_id}` has mismatched candidate/score lengths")]
    MalformedTeacherJudgment { query_id: String },
    #[error("benchmark evaluation requires at least one query case")]
    EmptyBenchmarkSuite,
}

/// Canonical source record category over which search indexes are derived views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Document,
    RepositoryBlob,
    GitCommit,
    AgentEvent,
    StoredDecision,
}

/// Immutable source identity, version, and access-control metadata for a passage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PassageIdentity {
    pub source_id: String,
    pub passage_id: String,
    pub version_id: String,
    pub source_kind: SourceKind,
    pub tenant_id: String,
    /// Principal roles permitted to read this passage within `tenant_id`.
    /// Empty slice permits any authenticated principal belonging to `tenant_id`.
    pub allowed_roles: Vec<String>,
}

impl PassageIdentity {
    /// Validates that identity and tenant fields are non-empty and free of control bytes.
    pub fn validate(&self) -> Result<(), AccessDenialReason> {
        let valid_token = |s: &str| !s.trim().is_empty() && !s.chars().any(char::is_control);
        if !valid_token(&self.source_id)
            || !valid_token(&self.passage_id)
            || !valid_token(&self.version_id)
            || !valid_token(&self.tenant_id)
        {
            return Err(AccessDenialReason::MalformedIdentity);
        }
        if self.allowed_roles.iter().any(|r| !valid_token(r)) {
            return Err(AccessDenialReason::MalformedIdentity);
        }
        Ok(())
    }
}

/// Authoritative source record from which Stage-1 index candidates are derived
/// without mutating the underlying identity or version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalPassageRecord {
    pub identity: PassageIdentity,
    pub text: String,
}

/// Authenticated caller security context enforced prior to every pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessContext {
    pub tenant_id: String,
    pub principal_roles: Vec<String>,
}

impl AccessContext {
    /// Validates that the caller context specifies a well-formed `tenant_id`.
    pub fn validate(&self) -> Result<(), TieredRerankError> {
        if self.tenant_id.trim().is_empty() || self.tenant_id.chars().any(char::is_control) {
            return Err(TieredRerankError::InvalidAccessContext);
        }
        Ok(())
    }
}

/// Audit reason recorded when a candidate is rejected prior to scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessDenialReason {
    MalformedIdentity,
    CrossTenantMismatch,
    InsufficientRole,
    CorruptedOrInvalidScore,
}

/// Pre-stage audit log entry for an excluded candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessFilteredAudit {
    pub passage_id: String,
    pub reason: AccessDenialReason,
}

/// Derived Stage-1 broad retrieval candidate referencing an immutable [`PassageIdentity`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BroadStageCandidate {
    pub identity: PassageIdentity,
    pub text: String,
    pub stage1_raw_score: f64,
}

/// Rebuilds derived Stage-1 broad candidates over immutable [`CanonicalPassageRecord`]s
/// without altering `identity` or `version_id`.
pub fn rebuild_stage1_candidates(
    records: &[CanonicalPassageRecord],
    query: &str,
) -> Vec<BroadStageCandidate> {
    let q_terms: HashSet<String> = tokenize_content(query).into_iter().collect();
    records
        .iter()
        .map(|rec| {
            let d_terms: HashSet<String> = tokenize_content(&rec.text).into_iter().collect();
            let overlap = if q_terms.is_empty() {
                0.0
            } else {
                q_terms.iter().filter(|t| d_terms.contains(*t)).count() as f64
                    / q_terms.len() as f64
            };
            BroadStageCandidate {
                identity: rec.identity.clone(),
                text: rec.text.clone(),
                stage1_raw_score: (0.15 + 0.85 * overlap).clamp(0.0, 1.0),
            }
        })
        .collect()
}

/// Enforces fail-closed tenant and role ACL isolation on a single candidate.
pub fn enforce_tenant_acl(
    access: &AccessContext,
    candidate: &BroadStageCandidate,
) -> Result<(), AccessDenialReason> {
    candidate.identity.validate()?;
    if candidate.identity.tenant_id != access.tenant_id {
        return Err(AccessDenialReason::CrossTenantMismatch);
    }
    if !candidate.identity.allowed_roles.is_empty()
        && !candidate
            .identity
            .allowed_roles
            .iter()
            .any(|req| access.principal_roles.iter().any(|held| held == req))
    {
        return Err(AccessDenialReason::InsufficientRole);
    }
    Ok(())
}

/// Applies mandatory pre-stage Tenant, Role ACL, and integrity filtering prior to Stage 1.
pub fn pre_stage_access_filter(
    access: &AccessContext,
    raw_candidates: &[BroadStageCandidate],
    max_corruption_ratio: f64,
) -> Result<(Vec<BroadStageCandidate>, Vec<AccessFilteredAudit>), TieredRerankError> {
    access.validate()?;
    let mut authorized = Vec::with_capacity(raw_candidates.len());
    let mut rejected = Vec::new();
    for cand in raw_candidates {
        match enforce_tenant_acl(access, cand) {
            Ok(()) => {
                if !cand.stage1_raw_score.is_finite()
                    || corruption_ratio(&cand.text) > max_corruption_ratio
                {
                    rejected.push(AccessFilteredAudit {
                        passage_id: cand.identity.passage_id.clone(),
                        reason: AccessDenialReason::CorruptedOrInvalidScore,
                    });
                } else {
                    authorized.push(cand.clone());
                }
            }
            Err(reason) => rejected.push(AccessFilteredAudit {
                passage_id: cand.identity.passage_id.clone(),
                reason,
            }),
        }
    }
    Ok((authorized, rejected))
}

fn split_query_clauses(query: &str) -> Vec<String> {
    let lower = query.to_ascii_lowercase();
    for sep in [" while ", " whereas ", " before ", " without ", " and ", "; "] {
        if lower.contains(sep) {
            let parts: Vec<String> = lower
                .split(sep)
                .map(str::trim)
                .filter(|p| tokenize_content(p).len() >= 2)
                .map(ToString::to_string)
                .collect();
            if parts.len() >= 2 {
                return parts;
            }
        }
    }
    vec![query.to_string()]
}

/// Extracts the 10-dimensional structural cross-interaction feature vector for `(query, text)`.
pub fn extract_pair_features(
    query: &str,
    doc_text: &str,
    stage1_norm: f64,
) -> [f64; DISTILL_FEATURE_DIM] {
    let q_toks = tokenize_content(query);
    let d_toks = tokenize_content(doc_text);
    if q_toks.is_empty() || d_toks.is_empty() {
        return [stage1_norm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }

    let q_set: HashSet<&str> = q_toks.iter().map(String::as_str).collect();
    let d_set: HashSet<&str> = d_toks.iter().map(String::as_str).collect();
    let matched = q_set.iter().filter(|t| d_set.contains(*t)).count();
    let token_recall = matched as f64 / q_set.len() as f64;
    let token_precision = ((matched as f64 / d_set.len().max(1) as f64) * 2.0).clamp(0.0, 1.0);

    let bigram_overlap = if q_toks.len() >= 2 && d_toks.len() >= 2 {
        let d_bg: HashSet<(&str, &str)> =
            d_toks.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect();
        let q_bg: Vec<(&str, &str)> =
            q_toks.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect();
        q_bg.iter().filter(|bg| d_bg.contains(*bg)).count() as f64 / q_bg.len() as f64
    } else {
        token_recall
    };

    let q_raw_ids: HashSet<String> = tokenize_raw(query)
        .into_iter()
        .filter(|t| {
            (t.bytes().any(|b| b.is_ascii_digit()) && t.bytes().any(|b| b.is_ascii_alphabetic()))
                || t.contains('-')
        })
        .collect();
    let d_raw: HashSet<String> = tokenize_raw(doc_text).into_iter().collect();
    let id_match = if q_raw_ids.is_empty() {
        token_recall
    } else {
        q_raw_ids.iter().filter(|t| d_raw.contains(*t)).count() as f64 / q_raw_ids.len() as f64
    };

    let mut pos_map: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, tok) in d_toks.iter().enumerate() {
        pos_map.entry(tok.as_str()).or_default().push(idx);
    }
    let q_present: Vec<&str> =
        q_toks.iter().map(String::as_str).filter(|t| pos_map.contains_key(*t)).collect();
    let ordered_prox = if q_present.len() >= 2 {
        let mut ordered_hits = 0usize;
        for pair in q_present.windows(2) {
            let pos_a = &pos_map[pair[0]];
            let pos_b = &pos_map[pair[1]];
            if pos_a.iter().any(|&pa| pos_b.iter().any(|&pb| pb > pa && pb - pa <= 8)) {
                ordered_hits += 1;
            }
        }
        ordered_hits as f64 / (q_present.len() - 1) as f64
    } else {
        token_recall
    };

    let clauses = split_query_clauses(query);
    let clause_cov = if clauses.len() >= 2 {
        let mut min_rec = 1.0f64;
        let mut any_clause = false;
        for cl in &clauses {
            let ct: HashSet<String> = tokenize_content(cl).into_iter().collect();
            if !ct.is_empty() {
                any_clause = true;
                let hits = ct.iter().filter(|t| d_set.contains(t.as_str())).count();
                min_rec = min_rec.min(hits as f64 / ct.len() as f64);
            }
        }
        if any_clause { min_rec } else { token_recall }
    } else {
        token_recall
    };

    let q_nums = extract_numbers(query);
    let d_nums: HashSet<String> = extract_numbers(doc_text).into_iter().collect();
    let num_cons = if q_nums.is_empty() {
        0.5
    } else if q_nums.iter().all(|n| d_nums.contains(n)) {
        1.0
    } else {
        -1.0
    };

    let pol_cons = if is_negated(query) == is_negated(doc_text) { 1.0 } else { -1.0 };
    let multi_clause_gap = clause_cov - token_recall;

    [
        stage1_norm,
        token_recall,
        token_precision,
        bigram_overlap,
        id_match,
        ordered_prox,
        clause_cov,
        num_cons,
        pol_cons,
        multi_clause_gap,
    ]
}

/// Stage-2 fast lightweight cross-encoder trait.
pub trait FastCrossEncoder: Send + Sync {
    fn model_id(&self) -> &str;
    fn score_pair(&self, query: &str, candidate: &BroadStageCandidate, stage1_norm: f64) -> f64;
    fn compute_multiplier(&self) -> f64 {
        1.0
    }
}

/// Deterministic calibrated Stage-2 lexical-interaction and bi-gram cross-encoder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibratedFastCrossEncoder {
    pub stage1_prior_weight: f64,
}

impl Default for CalibratedFastCrossEncoder {
    fn default() -> Self {
        Self { stage1_prior_weight: 0.12 }
    }
}

impl FastCrossEncoder for CalibratedFastCrossEncoder {
    fn model_id(&self) -> &str {
        "hanzo/fast-lexical-ce-v1"
    }

    fn score_pair(&self, query: &str, candidate: &BroadStageCandidate, stage1_norm: f64) -> f64 {
        let f = extract_pair_features(query, &candidate.text, stage1_norm);
        let lexical = 0.54 * f[1] + 0.16 * f[2] + 0.16 * f[3] + 0.14 * f[4];
        let raw = self.stage1_prior_weight * f[0] + (1.0 - self.stage1_prior_weight) * lexical;
        raw.clamp(0.0, 1.0)
    }
}

/// Stage-3 heavy cross-encoder teacher trait invoked strictly on escalated hard queries.
pub trait HeavyCrossEncoder: Send + Sync {
    fn model_id(&self) -> &str;
    fn score_pair(&self, query: &str, candidate: &BroadStageCandidate, stage2_score: f64) -> f64;
}

/// Deterministic Stage-3 span-aligned, polarity/numeric-verified, multi-clause teacher.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpanAlignedHeavyCrossEncoder {
    pub stage2_prior_weight: f64,
}

impl Default for SpanAlignedHeavyCrossEncoder {
    fn default() -> Self {
        Self { stage2_prior_weight: 0.10 }
    }
}

impl HeavyCrossEncoder for SpanAlignedHeavyCrossEncoder {
    fn model_id(&self) -> &str {
        "hanzo/heavy-span-teacher-ce-v1"
    }

    fn score_pair(&self, query: &str, candidate: &BroadStageCandidate, stage2_score: f64) -> f64 {
        let f = extract_pair_features(query, &candidate.text, 0.5);
        let base =
            0.28 * f[1] + 0.10 * f[2] + 0.14 * f[3] + 0.12 * f[4] + 0.14 * f[5] + 0.22 * f[6];
        let heavy = if f[7] < 0.0 || f[8] < 0.0 {
            base * 0.20
        } else if f[9] < -0.18 {
            base * 0.42
        } else {
            let num_bonus = if f[7] > 0.8 { 0.08 } else { 0.0 };
            let pol_bonus = if is_negated(query) && f[8] > 0.0 { 0.05 } else { 0.0 };
            (base + num_bonus + pol_bonus).clamp(0.0, 1.0)
        };
        ((1.0 - self.stage2_prior_weight) * heavy + self.stage2_prior_weight * stage2_score)
            .clamp(0.0, 1.0)
    }
}

/// Discrete Stage-2 -> Stage-3 routing action emitted by Kai [`EscalationGate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationAction {
    ResolveAtStage2,
    EscalateToStage3,
}

/// Structured non-generative decision emitted by [`EscalationGate`] under [`DecisionHead::Kai`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EscalationDecision {
    pub action: EscalationAction,
    pub decision_head: DecisionHead,
    pub top1_score: f64,
    pub top1_top2_margin: f64,
    pub normalized_entropy: f64,
    pub p_escalate: f64,
    pub p_resolve_stage2: f64,
}

/// Kai non-generative escalation gate evaluating Stage-2 score floor, margin, and entropy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EscalationGate {
    pub min_top1_score: f64,
    pub min_margin: f64,
    pub max_normalized_entropy: f64,
    pub top_m: usize,
    pub softmax_temperature: f64,
}

impl Default for EscalationGate {
    fn default() -> Self {
        Self {
            min_top1_score: 0.50,
            min_margin: 0.28,
            max_normalized_entropy: 0.76,
            top_m: 4,
            softmax_temperature: 0.15,
        }
    }
}

impl EscalationGate {
    /// Validates gate hyperparameters and constructs an [`EscalationGate`].
    pub fn new(
        min_top1_score: f64,
        min_margin: f64,
        max_normalized_entropy: f64,
        top_m: usize,
        softmax_temperature: f64,
    ) -> Result<Self, TieredRerankError> {
        for (field, value) in [
            ("min_top1_score", min_top1_score),
            ("min_margin", min_margin),
            ("max_normalized_entropy", max_normalized_entropy),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TieredRerankError::InvalidConfig {
                    field,
                    value,
                    reason: "must be finite and in [0.0, 1.0]",
                });
            }
        }
        if top_m < 2 {
            return Err(TieredRerankError::InvalidConfig {
                field: "top_m",
                value: top_m as f64,
                reason: "must be >= 2",
            });
        }
        if !softmax_temperature.is_finite() || softmax_temperature <= 0.0 {
            return Err(TieredRerankError::InvalidConfig {
                field: "softmax_temperature",
                value: softmax_temperature,
                reason: "must be finite and > 0.0",
            });
        }
        Ok(Self { min_top1_score, min_margin, max_normalized_entropy, top_m, softmax_temperature })
    }

    /// Evaluates Stage-2 candidate scores and emits a calibrated Kai escalation decision.
    pub fn evaluate(&self, stage2_scores: &[f64]) -> EscalationDecision {
        if stage2_scores.is_empty() {
            return EscalationDecision {
                action: EscalationAction::ResolveAtStage2,
                decision_head: DecisionHead::Kai,
                top1_score: 0.0,
                top1_top2_margin: 0.0,
                normalized_entropy: 0.0,
                p_escalate: 0.0,
                p_resolve_stage2: 1.0,
            };
        }
        let mut sorted: Vec<f64> =
            stage2_scores.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted.is_empty() {
            return EscalationDecision {
                action: EscalationAction::ResolveAtStage2,
                decision_head: DecisionHead::Kai,
                top1_score: 0.0,
                top1_top2_margin: 0.0,
                normalized_entropy: 0.0,
                p_escalate: 0.0,
                p_resolve_stage2: 1.0,
            };
        }
        let top1 = sorted[0];
        if sorted.len() == 1 {
            let escalate = top1 < self.min_top1_score;
            let p_esc = if escalate { 0.80 } else { 0.05 };
            return EscalationDecision {
                action: if escalate {
                    EscalationAction::EscalateToStage3
                } else {
                    EscalationAction::ResolveAtStage2
                },
                decision_head: DecisionHead::Kai,
                top1_score: top1,
                top1_top2_margin: top1,
                normalized_entropy: 0.0,
                p_escalate: p_esc,
                p_resolve_stage2: 1.0 - p_esc,
            };
        }

        let margin = sorted[0] - sorted[1];
        let m = sorted.len().min(self.top_m.max(2));
        let max_s = sorted[0];
        let exps: Vec<f64> =
            sorted[..m].iter().map(|&s| ((s - max_s) / self.softmax_temperature).exp()).collect();
        let denom: f64 = exps.iter().sum();
        let entropy: f64 = exps
            .iter()
            .map(|&e| {
                let p = (e / denom).max(1e-12);
                -p * p.ln()
            })
            .sum();
        let norm_entropy = (entropy / (m as f64).ln()).clamp(0.0, 1.0);

        let escalate = top1 < self.min_top1_score
            || margin < self.min_margin
            || norm_entropy > self.max_normalized_entropy;

        let d_top1 = (self.min_top1_score - top1) * 6.0;
        let d_margin = (self.min_margin - margin) * 10.0;
        let d_ent = (norm_entropy - self.max_normalized_entropy) * 6.0;
        let logit = d_top1.max(d_margin).max(d_ent).clamp(-15.0, 15.0);
        let p_escalate = 1.0 / (1.0 + (-logit).exp());

        EscalationDecision {
            action: if escalate {
                EscalationAction::EscalateToStage3
            } else {
                EscalationAction::ResolveAtStage2
            },
            decision_head: DecisionHead::Kai,
            top1_score: top1,
            top1_top2_margin: margin,
            normalized_entropy: norm_entropy,
            p_escalate,
            p_resolve_stage2: 1.0 - p_escalate,
        }
    }
}

/// Reranking policy evaluated across matched retrieval budgets and corpus snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankPolicy {
    Stage1Only,
    AlwaysFastStage2,
    AlwaysHeavyStage3,
    TieredCascade,
    DistilledStudentCascade,
}

/// Budget and candidate limits for [`TieredRerankPipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TieredRerankConfig {
    pub top_k_broad: usize,
    pub final_top_k: usize,
    pub max_stage3_candidates: usize,
    pub max_stage3_rerank_tokens: usize,
    pub max_corruption_ratio: f64,
}

impl Default for TieredRerankConfig {
    fn default() -> Self {
        Self {
            top_k_broad: 8,
            final_top_k: 5,
            max_stage3_candidates: 5,
            max_stage3_rerank_tokens: 1200,
            max_corruption_ratio: 0.05,
        }
    }
}

/// Latency, compute-unit, and dollar cost telemetry model for matched stage comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TieredCostModel {
    pub stage1_base_latency_ms: f64,
    pub stage1_cost_usd: f64,
    pub stage2_ms_per_100_tokens: f64,
    pub stage2_usd_per_1k_tokens: f64,
    pub stage2_compute_per_token: f64,
    pub student_ms_per_100_tokens: f64,
    pub student_usd_per_1k_tokens: f64,
    pub student_compute_per_token: f64,
    pub stage3_ms_per_100_tokens: f64,
    pub stage3_usd_per_1k_tokens: f64,
    pub stage3_compute_per_token: f64,
}

impl Default for TieredCostModel {
    fn default() -> Self {
        Self {
            stage1_base_latency_ms: 1.80,
            stage1_cost_usd: 0.00004,
            stage2_ms_per_100_tokens: 1.20,
            stage2_usd_per_1k_tokens: 0.00015,
            stage2_compute_per_token: 1.0,
            student_ms_per_100_tokens: 0.45,
            student_usd_per_1k_tokens: 0.00005,
            student_compute_per_token: 0.35,
            stage3_ms_per_100_tokens: 11.50,
            stage3_usd_per_1k_tokens: 0.00120,
            stage3_compute_per_token: 8.0,
        }
    }
}

/// Final ranked evidence passage carrying stage provenance and preserved identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedPassage {
    pub identity: PassageIdentity,
    pub text: String,
    pub stage1_raw_score: f64,
    pub stage1_norm_score: f64,
    pub stage2_score: Option<f64>,
    pub stage3_score: Option<f64>,
    pub final_score: f64,
}

/// Per-query execution trace returned by [`TieredRerankPipeline::run_query`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TieredQueryRun {
    pub policy: RerankPolicy,
    pub ranked_passages: Vec<RankedPassage>,
    pub full_ranked_pool: Vec<RankedPassage>,
    pub access_filtered: Vec<AccessFilteredAudit>,
    pub escalation: Option<EscalationDecision>,
    pub escalated_to_stage3: bool,
    pub stage2_rerank_tokens: usize,
    pub stage3_rerank_tokens: usize,
    pub total_rerank_tokens: usize,
    pub compute_units: f64,
    pub latency_ms: f64,
    pub cost_usd: f64,
}

/// Three-stage cascaded reranking pipeline with mandatory pre-stage ACL enforcement.
#[derive(Debug, Clone, Default)]
pub struct TieredRerankPipeline {
    pub config: TieredRerankConfig,
    pub gate: EscalationGate,
    pub costs: TieredCostModel,
}

impl TieredRerankPipeline {
    /// Validates pipeline parameters and builds a [`TieredRerankPipeline`].
    pub fn new(
        config: TieredRerankConfig,
        gate: EscalationGate,
        costs: TieredCostModel,
    ) -> Result<Self, TieredRerankError> {
        if config.top_k_broad == 0 || config.final_top_k == 0 {
            return Err(TieredRerankError::InvalidConfig {
                field: "top_k_broad/final_top_k",
                value: 0.0,
                reason: "must be >= 1",
            });
        }
        if !config.max_corruption_ratio.is_finite()
            || !(0.0..=1.0).contains(&config.max_corruption_ratio)
        {
            return Err(TieredRerankError::InvalidConfig {
                field: "max_corruption_ratio",
                value: config.max_corruption_ratio,
                reason: "must be finite and in [0.0, 1.0]",
            });
        }
        Ok(Self { config, gate, costs })
    }

    /// Executes the selected reranking policy with pre-Stage-1, pre-Stage-2, and
    /// pre-Stage-3 Tenant & Role ACL verification.
    pub fn run_query(
        &self,
        policy: RerankPolicy,
        access: &AccessContext,
        query: &str,
        raw_candidates: &[BroadStageCandidate],
        fast_ce: &dyn FastCrossEncoder,
        heavy_ce: &dyn HeavyCrossEncoder,
    ) -> Result<TieredQueryRun, TieredRerankError> {
        // Stage 0 / Pre-Stage 1: Mandatory fail-closed Tenant & ACL isolation.
        let (mut stage1_pool, access_filtered) =
            pre_stage_access_filter(access, raw_candidates, self.config.max_corruption_ratio)?;

        stage1_pool.sort_by(|a, b| {
            b.stage1_raw_score
                .total_cmp(&a.stage1_raw_score)
                .then_with(|| a.identity.passage_id.cmp(&b.identity.passage_id))
        });
        stage1_pool.truncate(self.config.top_k_broad);

        if stage1_pool.is_empty() {
            return Ok(TieredQueryRun {
                policy,
                ranked_passages: Vec::new(),
                full_ranked_pool: Vec::new(),
                access_filtered,
                escalation: None,
                escalated_to_stage3: false,
                stage2_rerank_tokens: 0,
                stage3_rerank_tokens: 0,
                total_rerank_tokens: 0,
                compute_units: 0.0,
                latency_ms: self.costs.stage1_base_latency_ms,
                cost_usd: self.costs.stage1_cost_usd,
            });
        }

        let s_min = stage1_pool.iter().map(|c| c.stage1_raw_score).fold(f64::INFINITY, f64::min);
        let s_max =
            stage1_pool.iter().map(|c| c.stage1_raw_score).fold(f64::NEG_INFINITY, f64::max);
        let span = (s_max - s_min).max(1e-9);
        let s1_norms: Vec<f64> = if (s_max - s_min).abs() < 1e-12 {
            vec![1.0; stage1_pool.len()]
        } else {
            stage1_pool.iter().map(|c| (c.stage1_raw_score - s_min) / span).collect()
        };

        let q_tok_len = tokenize_raw(query).len();
        let pair_tokens: Vec<usize> =
            stage1_pool.iter().map(|c| q_tok_len + tokenize_raw(&c.text).len()).collect();

        let mut latency_ms = self.costs.stage1_base_latency_ms;
        let mut cost_usd = self.costs.stage1_cost_usd;
        let mut compute_units = 0.0f64;
        let mut stage2_rerank_tokens = 0usize;
        let mut stage3_rerank_tokens = 0usize;
        let mut escalation = None;
        let mut escalated_to_stage3 = false;

        let mut working: Vec<RankedPassage> = match policy {
            RerankPolicy::Stage1Only => stage1_pool
                .iter()
                .zip(s1_norms.iter())
                .map(|(c, &s1n)| RankedPassage {
                    identity: c.identity.clone(),
                    text: c.text.clone(),
                    stage1_raw_score: c.stage1_raw_score,
                    stage1_norm_score: s1n,
                    stage2_score: None,
                    stage3_score: None,
                    final_score: c.stage1_raw_score,
                })
                .collect(),

            RerankPolicy::AlwaysFastStage2 => {
                let mut out = Vec::with_capacity(stage1_pool.len());
                for ((c, &s1n), &pt) in
                    stage1_pool.iter().zip(s1_norms.iter()).zip(pair_tokens.iter())
                {
                    enforce_tenant_acl(access, c).map_err(|reason| {
                        TieredRerankError::AccessDenied {
                            passage_id: c.identity.passage_id.clone(),
                            reason,
                        }
                    })?;
                    let sc2 = fast_ce.score_pair(query, c, s1n);
                    stage2_rerank_tokens += pt;
                    out.push(RankedPassage {
                        identity: c.identity.clone(),
                        text: c.text.clone(),
                        stage1_raw_score: c.stage1_raw_score,
                        stage1_norm_score: s1n,
                        stage2_score: Some(sc2),
                        stage3_score: None,
                        final_score: sc2,
                    });
                }
                let t = stage2_rerank_tokens as f64;
                compute_units += t * self.costs.stage2_compute_per_token;
                latency_ms += (t / 100.0) * self.costs.stage2_ms_per_100_tokens;
                cost_usd += (t / 1000.0) * self.costs.stage2_usd_per_1k_tokens;
                out
            }

            RerankPolicy::AlwaysHeavyStage3 => {
                escalated_to_stage3 = true;
                let mut out = Vec::with_capacity(stage1_pool.len());
                for ((c, &s1n), &pt) in
                    stage1_pool.iter().zip(s1_norms.iter()).zip(pair_tokens.iter())
                {
                    enforce_tenant_acl(access, c).map_err(|reason| {
                        TieredRerankError::AccessDenied {
                            passage_id: c.identity.passage_id.clone(),
                            reason,
                        }
                    })?;
                    let sc2 = fast_ce.score_pair(query, c, s1n);
                    let sc3 = heavy_ce.score_pair(query, c, sc2);
                    stage3_rerank_tokens += pt;
                    out.push(RankedPassage {
                        identity: c.identity.clone(),
                        text: c.text.clone(),
                        stage1_raw_score: c.stage1_raw_score,
                        stage1_norm_score: s1n,
                        stage2_score: Some(sc2),
                        stage3_score: Some(sc3),
                        final_score: sc3,
                    });
                }
                let t = stage3_rerank_tokens as f64;
                compute_units += t * self.costs.stage3_compute_per_token;
                latency_ms += (t / 100.0) * self.costs.stage3_ms_per_100_tokens;
                cost_usd += (t / 1000.0) * self.costs.stage3_usd_per_1k_tokens;
                out
            }

            RerankPolicy::TieredCascade | RerankPolicy::DistilledStudentCascade => {
                let is_student = policy == RerankPolicy::DistilledStudentCascade;
                let mut stage2_items = Vec::with_capacity(stage1_pool.len());
                let mut s2_scores = Vec::with_capacity(stage1_pool.len());

                for ((c, &s1n), &pt) in
                    stage1_pool.iter().zip(s1_norms.iter()).zip(pair_tokens.iter())
                {
                    enforce_tenant_acl(access, c).map_err(|reason| {
                        TieredRerankError::AccessDenied {
                            passage_id: c.identity.passage_id.clone(),
                            reason,
                        }
                    })?;
                    let sc2 = fast_ce.score_pair(query, c, s1n);
                    stage2_rerank_tokens += pt;
                    s2_scores.push(sc2);
                    stage2_items.push((c, s1n, sc2, pt));
                }

                let t2 = stage2_rerank_tokens as f64;
                if is_student {
                    compute_units += t2 * self.costs.student_compute_per_token;
                    latency_ms += (t2 / 100.0) * self.costs.student_ms_per_100_tokens;
                    cost_usd += (t2 / 1000.0) * self.costs.student_usd_per_1k_tokens;
                } else {
                    compute_units += t2 * self.costs.stage2_compute_per_token;
                    latency_ms += (t2 / 100.0) * self.costs.stage2_ms_per_100_tokens;
                    cost_usd += (t2 / 1000.0) * self.costs.stage2_usd_per_1k_tokens;
                }

                let active_gate = if is_student {
                    EscalationGate {
                        min_margin: self.gate.min_margin.min(0.12),
                        ..self.gate
                    }
                } else {
                    self.gate
                };
                let decision = active_gate.evaluate(&s2_scores);
                escalation = Some(decision);

                stage2_items.sort_by(|a, b| {
                    b.2.total_cmp(&a.2)
                        .then_with(|| a.0.identity.passage_id.cmp(&b.0.identity.passage_id))
                });

                if decision.action == EscalationAction::ResolveAtStage2
                    || self.config.max_stage3_candidates == 0
                    || self.config.max_stage3_rerank_tokens == 0
                {
                    stage2_items
                        .into_iter()
                        .map(|(c, s1n, sc2, _)| RankedPassage {
                            identity: c.identity.clone(),
                            text: c.text.clone(),
                            stage1_raw_score: c.stage1_raw_score,
                            stage1_norm_score: s1n,
                            stage2_score: Some(sc2),
                            stage3_score: None,
                            final_score: sc2,
                        })
                        .collect()
                } else {
                    escalated_to_stage3 = true;
                    let mut out = Vec::with_capacity(stage2_items.len());
                    let mut s3_budget_used = 0usize;
                    for (idx, (c, s1n, sc2, pt)) in stage2_items.into_iter().enumerate() {
                        let within_count = idx < self.config.max_stage3_candidates;
                        let within_tokens =
                            s3_budget_used + pt <= self.config.max_stage3_rerank_tokens;
                        if within_count && within_tokens {
                            enforce_tenant_acl(access, c).map_err(|reason| {
                                TieredRerankError::AccessDenied {
                                    passage_id: c.identity.passage_id.clone(),
                                    reason,
                                }
                            })?;
                            let sc3 = heavy_ce.score_pair(query, c, sc2);
                            s3_budget_used += pt;
                            stage3_rerank_tokens += pt;
                            out.push(RankedPassage {
                                identity: c.identity.clone(),
                                text: c.text.clone(),
                                stage1_raw_score: c.stage1_raw_score,
                                stage1_norm_score: s1n,
                                stage2_score: Some(sc2),
                                stage3_score: Some(sc3),
                                final_score: sc3,
                            });
                        } else {
                            out.push(RankedPassage {
                                identity: c.identity.clone(),
                                text: c.text.clone(),
                                stage1_raw_score: c.stage1_raw_score,
                                stage1_norm_score: s1n,
                                stage2_score: Some(sc2),
                                stage3_score: None,
                                final_score: sc2 * 0.45,
                            });
                        }
                    }
                    let t3 = stage3_rerank_tokens as f64;
                    compute_units += t3 * self.costs.stage3_compute_per_token;
                    latency_ms += (t3 / 100.0) * self.costs.stage3_ms_per_100_tokens;
                    cost_usd += (t3 / 1000.0) * self.costs.stage3_usd_per_1k_tokens;
                    out
                }
            }
        };

        working.sort_by(|a, b| {
            b.final_score
                .total_cmp(&a.final_score)
                .then_with(|| a.identity.passage_id.cmp(&b.identity.passage_id))
        });

        let full_ranked_pool = working.clone();
        working.truncate(self.config.final_top_k);

        Ok(TieredQueryRun {
            policy,
            ranked_passages: working,
            full_ranked_pool,
            access_filtered,
            escalation,
            escalated_to_stage3,
            stage2_rerank_tokens,
            stage3_rerank_tokens,
            total_rerank_tokens: stage2_rerank_tokens + stage3_rerank_tokens,
            compute_units,
            latency_ms,
            cost_usd,
        })
    }
}

/// Tenant-scoped Stage-3 teacher judgment recorded for student distillation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherJudgment {
    pub query_id: String,
    pub tenant_id: String,
    pub query: String,
    pub candidates: Vec<BroadStageCandidate>,
    pub stage1_norm_scores: Vec<f64>,
    pub teacher_scores: Vec<f64>,
}

/// Hyperparameters for [`DistillationTrainer`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistillationConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub temperature: f64,
    pub kl_weight: f64,
    pub margin_weight: f64,
    pub bce_weight: f64,
    pub margin_scale: f64,
    pub min_teacher_pair_gap: f64,
    pub l2_reg: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            epochs: 140,
            learning_rate: 0.10,
            temperature: 0.35,
            kl_weight: 0.35,
            margin_weight: 0.35,
            bce_weight: 0.30,
            margin_scale: 4.0,
            min_teacher_pair_gap: 0.06,
            l2_reg: 1e-4,
        }
    }
}

/// Optimization trajectory emitted by [`DistillationTrainer::distill`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistillationReport {
    pub epochs: usize,
    pub judgment_count: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub loss_history: Vec<f64>,
    pub monotonic_non_increasing: bool,
}

/// Compact 83-parameter residual cross-interaction MLP student distilled from Stage 3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistilledDomainReranker {
    pub linear_weights: [f64; DISTILL_FEATURE_DIM],
    pub output_bias: f64,
    pub hidden_weights: [[f64; DISTILL_FEATURE_DIM]; DISTILL_HIDDEN_DIM],
    pub hidden_bias: [f64; DISTILL_HIDDEN_DIM],
    pub hidden_proj: [f64; DISTILL_HIDDEN_DIM],
}

impl Default for DistilledDomainReranker {
    fn default() -> Self {
        Self::untrained()
    }
}

impl DistilledDomainReranker {
    /// Initializes an untrained student relying only on generic surface-overlap features.
    pub fn untrained() -> Self {
        let mut hidden_weights = [[0.0; DISTILL_FEATURE_DIM]; DISTILL_HIDDEN_DIM];
        for (h, row) in hidden_weights.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (((h * 7 + j * 13 + 3) % 19) as f64 - 9.0) * 0.015;
            }
        }
        let mut hidden_proj = [0.0; DISTILL_HIDDEN_DIM];
        for (h, slot) in hidden_proj.iter_mut().enumerate() {
            *slot = (((h * 5 + 1) % 7) as f64 - 3.0) * 0.03;
        }
        Self {
            linear_weights: [1.05, 0.80, 0.20, 0.20, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            output_bias: -1.20,
            hidden_weights,
            hidden_bias: [0.0; DISTILL_HIDDEN_DIM],
            hidden_proj,
        }
    }

    /// Evaluates raw logit and hidden `tanh` activations for a feature vector.
    pub fn forward_logit(
        &self,
        features: &[f64; DISTILL_FEATURE_DIM],
    ) -> (f64, [f64; DISTILL_HIDDEN_DIM]) {
        let mut h_act = [0.0; DISTILL_HIDDEN_DIM];
        for (h, out_h) in h_act.iter_mut().enumerate() {
            let dot: f64 = self.hidden_weights[h]
                .iter()
                .zip(features.iter())
                .map(|(w, x)| w * x)
                .sum();
            *out_h = (dot + self.hidden_bias[h]).tanh();
        }
        let lin: f64 =
            self.linear_weights.iter().zip(features.iter()).map(|(w, x)| w * x).sum();
        let hid: f64 = self.hidden_proj.iter().zip(h_act.iter()).map(|(w, h)| w * h).sum();
        (lin + hid + self.output_bias, h_act)
    }

    /// Scores a pre-extracted feature vector into a calibrated `[0.0, 1.0]` relevance probability.
    pub fn score_features(&self, features: &[f64; DISTILL_FEATURE_DIM]) -> f64 {
        let (z, _) = self.forward_logit(features);
        let zc = z.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-zc).exp())
    }
}

impl FastCrossEncoder for DistilledDomainReranker {
    fn model_id(&self) -> &str {
        "hanzo/distilled-domain-student-mlp-v1"
    }

    fn score_pair(&self, query: &str, candidate: &BroadStageCandidate, stage1_norm: f64) -> f64 {
        let f = extract_pair_features(query, &candidate.text, stage1_norm);
        self.score_features(&f)
    }

    fn compute_multiplier(&self) -> f64 {
        0.35
    }
}

/// Teacher-to-student distillation trainer combining temperature-scaled KL divergence,
/// pairwise margin ranking loss, and calibrated pointwise cross-entropy.
#[derive(Debug, Clone, Default)]
pub struct DistillationTrainer {
    pub config: DistillationConfig,
}

impl DistillationTrainer {
    /// Distills Stage-3 `judgments` into a [`DistilledDomainReranker`], enforcing strict
    /// per-judgment tenant isolation before ingesting any candidate.
    pub fn distill(
        &self,
        judgments: &[TeacherJudgment],
    ) -> Result<(DistilledDomainReranker, DistillationReport), TieredRerankError> {
        if judgments.is_empty() {
            return Err(TieredRerankError::EmptyDistillationDataset);
        }

        struct PreparedItem {
            feats: Vec<[f64; DISTILL_FEATURE_DIM]>,
            t_scores: Vec<f64>,
            t_logits: Vec<f64>,
        }

        let mut prepared = Vec::with_capacity(judgments.len());
        for jdg in judgments {
            if jdg.tenant_id.trim().is_empty() {
                return Err(TieredRerankError::InvalidAccessContext);
            }
            if jdg.candidates.is_empty()
                || jdg.candidates.len() != jdg.stage1_norm_scores.len()
                || jdg.candidates.len() != jdg.teacher_scores.len()
            {
                return Err(TieredRerankError::MalformedTeacherJudgment {
                    query_id: jdg.query_id.clone(),
                });
            }
            let mut feats = Vec::with_capacity(jdg.candidates.len());
            for (cand, &s1n) in jdg.candidates.iter().zip(jdg.stage1_norm_scores.iter()) {
                cand.identity.validate().map_err(|reason| TieredRerankError::AccessDenied {
                    passage_id: cand.identity.passage_id.clone(),
                    reason,
                })?;
                if cand.identity.tenant_id != jdg.tenant_id {
                    return Err(TieredRerankError::AccessDenied {
                        passage_id: cand.identity.passage_id.clone(),
                        reason: AccessDenialReason::CrossTenantMismatch,
                    });
                }
                feats.push(extract_pair_features(&jdg.query, &cand.text, s1n));
            }
            let t_logits: Vec<f64> = jdg
                .teacher_scores
                .iter()
                .map(|&ts| {
                    let p = ts.clamp(1e-3, 0.999);
                    (p / (1.0 - p)).ln()
                })
                .collect();
            prepared.push(PreparedItem { feats, t_scores: jdg.teacher_scores.clone(), t_logits });
        }

        let mut student = DistilledDomainReranker::untrained();
        let cfg = &self.config;
        let mut loss_history = Vec::with_capacity(cfg.epochs + 1);
        let q_count = prepared.len() as f64;

        for ep in 0..=cfg.epochs {
            let mut total_loss = 0.0f64;
            let mut gw_lin = [0.0; DISTILL_FEATURE_DIM];
            let mut gb_out = 0.0f64;
            let mut gw_hid_in = [[0.0; DISTILL_FEATURE_DIM]; DISTILL_HIDDEN_DIM];
            let mut gb_hid = [0.0; DISTILL_HIDDEN_DIM];
            let mut gw_hid_proj = [0.0; DISTILL_HIDDEN_DIM];

            for item in &prepared {
                let n_c = item.feats.len();
                let mut z_list = Vec::with_capacity(n_c);
                let mut h_list = Vec::with_capacity(n_c);
                let mut sig_list = Vec::with_capacity(n_c);
                for f in &item.feats {
                    let (z, h) = student.forward_logit(f);
                    z_list.push(z);
                    h_list.push(h);
                    let zc = z.clamp(-20.0, 20.0);
                    sig_list.push(1.0 / (1.0 + (-zc).exp()));
                }

                let mx_t = item.t_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let et: Vec<f64> =
                    item.t_logits.iter().map(|&x| ((x - mx_t) / cfg.temperature).exp()).collect();
                let sum_et: f64 = et.iter().sum();
                let pt: Vec<f64> = et.iter().map(|&e| e / sum_et).collect();

                let mx_s = z_list.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let es: Vec<f64> =
                    z_list.iter().map(|&x| ((x - mx_s) / cfg.temperature).exp()).collect();
                let sum_es: f64 = es.iter().sum();
                let ps: Vec<f64> = es.iter().map(|&e| e / sum_es).collect();

                let kl: f64 = cfg.temperature.powi(2)
                    * pt.iter()
                        .zip(ps.iter())
                        .map(|(&pti, &psi)| pti * (pti.max(1e-12) / psi.max(1e-12)).ln())
                        .sum::<f64>();
                let mut dz: Vec<f64> = pt
                    .iter()
                    .zip(ps.iter())
                    .map(|(&pti, &psi)| cfg.kl_weight * cfg.temperature * (psi - pti))
                    .collect();

                let mut bce = 0.0f64;
                for ((&ts, &sig), dz_i) in
                    item.t_scores.iter().zip(sig_list.iter()).zip(dz.iter_mut())
                {
                    let yt = ts.clamp(1e-4, 1.0 - 1e-4);
                    let ys = sig.clamp(1e-6, 1.0 - 1e-6);
                    bce += -(yt * ys.ln() + (1.0 - yt) * (1.0 - ys).ln()) / n_c as f64;
                    *dz_i += cfg.bce_weight * (sig - yt) / n_c as f64;
                }

                let mut pair_cnt = 0usize;
                for &ti in &item.t_scores {
                    for &tj in &item.t_scores {
                        if ti - tj > cfg.min_teacher_pair_gap {
                            pair_cnt += 1;
                        }
                    }
                }
                let mut pair_loss = 0.0f64;
                if pair_cnt > 0 {
                    let inv_pairs = 1.0 / pair_cnt as f64;
                    for (i, (&ti, &zi)) in item.t_scores.iter().zip(z_list.iter()).enumerate() {
                        for (j, (&tj, &zj)) in item.t_scores.iter().zip(z_list.iter()).enumerate() {
                            let dt = ti - tj;
                            if dt > cfg.min_teacher_pair_gap {
                                let viol = cfg.margin_scale * dt - (zi - zj);
                                if viol > 0.0 {
                                    pair_loss += viol * inv_pairs;
                                    let w = cfg.margin_weight * inv_pairs;
                                    dz[i] -= w;
                                    dz[j] += w;
                                }
                            }
                        }
                    }
                }

                total_loss +=
                    cfg.kl_weight * kl + cfg.margin_weight * pair_loss + cfg.bce_weight * bce;

                for ((&g, f), h) in dz.iter().zip(item.feats.iter()).zip(h_list.iter()) {
                    gb_out += g;
                    for (gw, &fx) in gw_lin.iter_mut().zip(f.iter()) {
                        *gw += g * fx;
                    }
                    for k in 0..DISTILL_HIDDEN_DIM {
                        gw_hid_proj[k] += g * h[k];
                        let dh = g * student.hidden_proj[k] * (1.0 - h[k] * h[k]);
                        gb_hid[k] += dh;
                        for (gw_in, &fx) in gw_hid_in[k].iter_mut().zip(f.iter()) {
                            *gw_in += dh * fx;
                        }
                    }
                }
            }

            let reg = 0.5
                * cfg.l2_reg
                * (student.linear_weights.iter().map(|x| x * x).sum::<f64>()
                    + student.hidden_proj.iter().map(|x| x * x).sum::<f64>()
                    + student
                        .hidden_weights
                        .iter()
                        .flat_map(|r| r.iter())
                        .map(|x| x * x)
                        .sum::<f64>());
            loss_history.push(total_loss / q_count + reg);
            if ep == cfg.epochs {
                break;
            }

            for (w, &gw) in student.linear_weights.iter_mut().zip(gw_lin.iter()) {
                *w -= cfg.learning_rate * (gw / q_count + cfg.l2_reg * *w);
            }
            student.output_bias -= cfg.learning_rate * (gb_out / q_count);
            for k in 0..DISTILL_HIDDEN_DIM {
                student.hidden_proj[k] -= cfg.learning_rate
                    * (gw_hid_proj[k] / q_count + cfg.l2_reg * student.hidden_proj[k]);
                student.hidden_bias[k] -= cfg.learning_rate * (gb_hid[k] / q_count);
                for (w_in, &gw_in) in
                    student.hidden_weights[k].iter_mut().zip(gw_hid_in[k].iter())
                {
                    *w_in -= cfg.learning_rate * (gw_in / q_count + cfg.l2_reg * *w_in);
                }
            }
        }

        let monotonic = loss_history.windows(2).all(|w| w[0] >= w[1] - 1e-9);
        let report = DistillationReport {
            epochs: cfg.epochs,
            judgment_count: judgments.len(),
            initial_loss: loss_history.first().copied().unwrap_or(0.0),
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            loss_history,
            monotonic_non_increasing: monotonic,
        };
        Ok((student, report))
    }
}

/// Difficulty stratum in the multi-tenant retrieval evaluation suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryDifficulty {
    Easy,
    Ambiguous,
    HardMultiHop,
}

/// Single benchmark query with caller access context, raw candidate pool, graded
/// relevance labels, and downstream answer fact requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkQueryCase {
    pub query_id: String,
    pub difficulty: QueryDifficulty,
    pub access: AccessContext,
    pub query: String,
    pub raw_candidates: Vec<BroadStageCandidate>,
    /// Graded relevance map (`passage_id -> grade` in `1..=3`).
    pub relevance_grades: HashMap<String, u8>,
    pub required_answer_facts: Vec<String>,
}

/// Fixed deterministic `Zen` grounded synthesizer used identically across all 5 policies.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicZenGrounded;

impl ZenSynthesizer for DeterministicZenGrounded {
    fn synthesize(
        &self,
        _query: &str,
        retrieved_evidence: &[String],
        target: SynthesisTarget,
    ) -> String {
        match target {
            SynthesisTarget::ZenParametric => String::new(),
            SynthesisTarget::ZenGrounded => {
                retrieved_evidence.first().cloned().unwrap_or_default()
            }
        }
    }
}

/// Matched retrieval, accuracy, latency, token/compute, and cost metrics for one policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvaluationReport {
    pub policy: RerankPolicy,
    pub query_count: usize,
    pub ndcg_at_5: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    pub answer_accuracy: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub mean_latency_ms: f64,
    pub total_rerank_tokens: usize,
    pub stage2_rerank_tokens: usize,
    pub stage3_rerank_tokens: usize,
    pub compute_units: f64,
    pub stage3_escalation_rate: f64,
    pub total_cost_usd: f64,
    pub cost_per_correct_passage_usd: f64,
    pub cross_tenant_leaks: usize,
    pub acl_filtered_candidates: usize,
}

/// Standalone Stage-2 student ranking lift on unseen queries before vs. after distillation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudentDistillationLift {
    pub pre_ndcg_at_5: f64,
    pub post_ndcg_at_5: f64,
    pub pre_mrr: f64,
    pub post_mrr: f64,
    pub pre_recall_at_5: f64,
    pub post_recall_at_5: f64,
    pub pre_answer_accuracy: f64,
    pub post_answer_accuracy: f64,
}

/// Complete Pareto evaluation summary across all five policies plus distillation telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TieredBenchmarkSummary {
    pub stage1_only: PolicyEvaluationReport,
    pub always_fast_stage2: PolicyEvaluationReport,
    pub always_heavy_stage3: PolicyEvaluationReport,
    pub tiered_cascade: PolicyEvaluationReport,
    pub distilled_student_cascade: PolicyEvaluationReport,
    pub distillation: DistillationReport,
    pub student_lift: StudentDistillationLift,
    pub tiered_compute_savings_pct: f64,
    pub distilled_compute_savings_pct: f64,
}

const BACKGROUND_POOL: &[(&str, SourceKind, &str)] = &[
    (
        "bg_wal",
        SourceKind::Document,
        "Write-ahead log segments rotate at 64 MiB boundaries before archive upload.",
    ),
    (
        "bg_hnsw",
        SourceKind::RepositoryBlob,
        "Vector index builders allocate arena memory buffers per worker thread during graph build.",
    ),
    (
        "bg_rbac",
        SourceKind::StoredDecision,
        "Tenant access policies bind principal role tokens at request ingress prior to search.",
    ),
    (
        "bg_snap",
        SourceKind::GitCommit,
        "Incremental snapshot delta manifests record segment byte offsets and dictionary IDs.",
    ),
    (
        "bg_raft",
        SourceKind::AgentEvent,
        "Consensus state machines persist term ballots to durable media prior to election.",
    ),
    (
        "bg_bm25",
        SourceKind::Document,
        "Inverted posting lists compress document frequency deltas with SIMD bitpacking blocks.",
    ),
];

fn extra_hard_distractors(qid: &str) -> &'static [(&'static str, &'static str, f64)] {
    match qid {
        "q15" => &[
            (
                "dist_ex1",
                "Write-ahead log truncation executes on replica nodes \
                 during hourly replica maintenance.",
                0.93,
            ),
            (
                "dist_ex2",
                "Read-only replica nodes schedule write-ahead log truncation \
                 after local replica sync.",
                0.92,
            ),
            (
                "dist_ex3",
                "Replica nodes run write-ahead log truncation threads \
                 across read-only storage volumes.",
                0.91,
            ),
            (
                "dist_ex4",
                "Write-ahead log truncation compacts replica node journals \
                 on read-only standby hosts.",
                0.90,
            ),
        ],
        "q18" => &[
            (
                "dist_ex1",
                "HIP-1346 contract version 1 enforces a maximum Matryoshka \
                 embedding dimension of 1024 floats.",
                0.93,
            ),
            (
                "dist_ex2",
                "HIP-1346 contract version 2 enforces a maximum Matryoshka \
                 embedding dimension of 1536 floats.",
                0.92,
            ),
            (
                "dist_ex3",
                "HIP-1346 contract version 4 enforces a maximum Matryoshka \
                 embedding dimension of 2048 floats.",
                0.91,
            ),
            (
                "dist_ex4",
                "HIP-1346 contract version 5 enforces a maximum Matryoshka \
                 embedding dimension of 4096 floats.",
                0.90,
            ),
        ],
        "q20" => &[
            (
                "dist_ex1",
                "HNSW graph compaction preserves tombstoned vector IDs \
                 in memory-mapped HNSW segment headers.",
                0.94,
            ),
            (
                "dist_ex2",
                "HNSW graph compaction scans tombstoned vector IDs \
                 during HNSW layer rebalancing.",
                0.93,
            ),
            (
                "dist_ex3",
                "HNSW graph compaction indexes tombstoned vector IDs \
                 inside dense HNSW neighbor tables.",
                0.92,
            ),
            (
                "dist_ex4",
                "HNSW graph compaction prunes tombstoned vector IDs \
                 across HNSW shard partitions.",
                0.91,
            ),
        ],
        _ => &[],
    }
}

fn make_candidate_pool(
    qid: &str,
    tenant: &str,
    role: &str,
    gold_text: &str,
    s1_gold: f64,
    dist_text: &str,
    s1_dist: f64,
) -> Vec<BroadStageCandidate> {
    let other_tenant = if tenant == "tenant_acme" { "tenant_globex" } else { "tenant_acme" };
    let mk = |pid: &str, ten: &str, roles: Vec<String>, kind: SourceKind, txt: &str, sc: f64| {
        BroadStageCandidate {
            identity: PassageIdentity {
                source_id: format!("src_{qid}_{pid}"),
                passage_id: pid.to_string(),
                version_id: format!("rev_{qid}_{pid}_v1"),
                source_kind: kind,
                tenant_id: ten.to_string(),
                allowed_roles: roles,
            },
            text: txt.to_string(),
            stage1_raw_score: sc,
        }
    };

    let mut raw = vec![
        mk("adv_cross_tenant", other_tenant, Vec::new(), SourceKind::Document, gold_text, 0.99),
        mk(
            "adv_unauth_role",
            tenant,
            vec!["root_auditor".to_string()],
            SourceKind::StoredDecision,
            gold_text,
            0.98,
        ),
        mk(
            "gold_primary",
            tenant,
            vec![role.to_string()],
            SourceKind::Document,
            gold_text,
            s1_gold,
        ),
        mk("dist_primary", tenant, Vec::new(), SourceKind::RepositoryBlob, dist_text, s1_dist),
    ];

    let extras = extra_hard_distractors(qid);
    for &(ex_id, ex_txt, ex_sc) in extras {
        raw.push(mk(ex_id, tenant, Vec::new(), SourceKind::Document, ex_txt, ex_sc));
    }
    let bg_needed = 6usize.saturating_sub(extras.len());
    for (idx, &(bg_id, kind, bg_txt)) in BACKGROUND_POOL.iter().take(bg_needed).enumerate() {
        raw.push(mk(bg_id, tenant, Vec::new(), kind, bg_txt, 0.86 - idx as f64 * 0.025));
    }
    raw
}

/// Builds a 12-query teacher-labeled distillation dataset completely disjoint from
/// [`default_multi_tenant_benchmark`].
pub fn default_distillation_training_set(
    fast_ce: &dyn FastCrossEncoder,
    heavy_ce: &dyn HeavyCrossEncoder,
) -> Vec<TeacherJudgment> {
    struct TrainSpec<'a> {
        qid: &'a str,
        tenant: &'a str,
        query: &'a str,
        gold: &'a str,
        s1_gold: f64,
        dist: &'a str,
        s1_dist: f64,
    }
    let specs: &[TrainSpec<'_>] = &[
        TrainSpec {
            qid: "tr01",
            tenant: "tenant_acme",
            query: "Why must compaction workers not delete active snapshot hardlinks?",
            gold: "Compaction workers must never delete active snapshot hardlinks \
                   while readers hold open snapshot descriptors.",
            s1_gold: 0.74,
            dist: "Compaction workers delete active snapshot hardlinks \
                   immediately after segment merge completion.",
            s1_dist: 0.93,
        },
        TrainSpec {
            qid: "tr02",
            tenant: "tenant_globex",
            query: "What gRPC keepalive interval in seconds applies to schema version 5?",
            gold: "In schema version 5, inter-node gRPC keepalive probes fire every 15 seconds.",
            s1_gold: 0.74,
            dist: "In schema version 2, inter-node gRPC keepalive probes fire every 90 seconds.",
            s1_dist: 0.93,
        },
        TrainSpec {
            qid: "tr03",
            tenant: "tenant_acme",
            query: "How does prefix cache invalidation purge stale trie nodes \
                    while bounding arena epoch memory?",
            gold: "Prefix cache invalidation purges stale trie nodes via generation counters \
                   while bounding arena epoch memory under 128 MiB.",
            s1_gold: 0.73,
            dist: "Prefix cache invalidation purges stale trie nodes \
                   during prefix dictionary rebuilds.",
            s1_dist: 0.94,
        },
        TrainSpec {
            qid: "tr04",
            tenant: "tenant_globex",
            query: "Why cannot quantized int4 vectors bypass L2 norm verification?",
            gold: "Quantized int4 vectors cannot bypass L2 norm verification \
                   without corrupting inner-product rank order.",
            s1_gold: 0.73,
            dist: "Quantized int4 vectors bypass L2 norm verification \
                   during uncalibrated fast ingestion.",
            s1_dist: 0.92,
        },
        TrainSpec {
            qid: "tr05",
            tenant: "tenant_acme",
            query: "What maximum batch size in documents is enforced in ingestion pipeline 4.1?",
            gold: "Ingestion pipeline 4.1 enforces a maximum batch size \
                   of 5000 documents per commit.",
            s1_gold: 0.74,
            dist: "Ingestion pipeline 3.0 enforces a maximum batch size \
                   of 1000 documents per commit.",
            s1_dist: 0.93,
        },
        TrainSpec {
            qid: "tr06",
            tenant: "tenant_globex",
            query: "How does quantized reranking fuse lexical exact-match boosts \
                    while respecting tenant ACL boundaries?",
            gold: "Quantized reranking fuses lexical exact-match boosts with cross-encoder logits \
                   while respecting tenant ACL boundaries prior to scoring.",
            s1_gold: 0.72,
            dist: "Quantized reranking fuses lexical exact-match boosts \
                   across title and body fields.",
            s1_dist: 0.94,
        },
        TrainSpec {
            qid: "tr07",
            tenant: "tenant_acme",
            query: "What compression level does cold backup policy 7 apply to roaring bitmaps?",
            gold: "Cold backup policy 7 compresses serialized roaring bitmaps at zstd level 9.",
            s1_gold: 0.74,
            dist: "Cold backup policy 3 compresses serialized roaring bitmaps at zstd level 1.",
            s1_dist: 0.92,
        },
        TrainSpec {
            qid: "tr08",
            tenant: "tenant_globex",
            query: "Why must follower nodes not accept direct client write mutations?",
            gold: "Follower nodes must never accept direct client write mutations \
                   and always forward writes to the raft leader.",
            s1_gold: 0.74,
            dist: "Follower nodes accept direct client write mutations \
                   in uncoordinated local test modes.",
            s1_dist: 0.93,
        },
        TrainSpec {
            qid: "tr09",
            tenant: "tenant_acme",
            query: "How does FActScore atomic verification bind character spans \
                    while computing expected calibration error?",
            gold: "FActScore atomic verification binds exact sentence character spans \
                   while computing expected calibration error across 10 confidence bins.",
            s1_gold: 0.73,
            dist: "FActScore atomic verification binds character spans \
                   during FActScore evaluation.",
            s1_dist: 0.94,
        },
        TrainSpec {
            qid: "tr10",
            tenant: "tenant_globex",
            query: "What bloom filter false-positive percentage targets segment format 8?",
            gold: "Segment format 8 sizes per-block bloom filters \
                   for a 0.5 percent false-positive rate.",
            s1_gold: 0.74,
            dist: "Segment format 4 sizes per-block bloom filters \
                   for a 2.5 percent false-positive rate.",
            s1_dist: 0.92,
        },
        TrainSpec {
            qid: "tr11",
            tenant: "tenant_acme",
            query: "Which syscall flushes durable LMDB meta pages on Linux kernels?",
            gold: "Durable LMDB commits invoke fdatasync to flush dirty meta pages to NVMe media.",
            s1_gold: 0.88,
            dist: "Page cache readahead windows prefetch sequential blocks \
                   during full table scans.",
            s1_dist: 0.80,
        },
        TrainSpec {
            qid: "tr12",
            tenant: "tenant_globex",
            query: "How does synonym expansion prevent combinatorial query blowup \
                    while preserving phrase order?",
            gold: "Synonym expansion caps lattice branching at 16 paths to prevent \
                   combinatorial query blowup while preserving phrase order via token positions.",
            s1_gold: 0.73,
            dist: "Synonym expansion prevents combinatorial query blowup \
                   using JSON synonym tables.",
            s1_dist: 0.93,
        },
    ];

    let mut out = Vec::with_capacity(specs.len());
    for sp in specs {
        let role = if sp.tenant == "tenant_acme" { "index_ops" } else { "ml_engineer" };
        let access = AccessContext {
            tenant_id: sp.tenant.to_string(),
            principal_roles: vec!["search_reader".to_string(), role.to_string()],
        };
        let raw = make_candidate_pool(
            sp.qid,
            sp.tenant,
            role,
            sp.gold,
            sp.s1_gold,
            sp.dist,
            sp.s1_dist,
        );
        let (mut kept, _) = pre_stage_access_filter(&access, &raw, 0.05).expect("valid access");
        kept.sort_by(|a, b| {
            b.stage1_raw_score
                .total_cmp(&a.stage1_raw_score)
                .then_with(|| a.identity.passage_id.cmp(&b.identity.passage_id))
        });
        kept.truncate(8);
        let s_min = kept.iter().map(|c| c.stage1_raw_score).fold(f64::INFINITY, f64::min);
        let s_max = kept.iter().map(|c| c.stage1_raw_score).fold(f64::NEG_INFINITY, f64::max);
        let span = (s_max - s_min).max(1e-9);
        let s1_norms: Vec<f64> = kept.iter().map(|c| (c.stage1_raw_score - s_min) / span).collect();
        let t_scores: Vec<f64> = kept
            .iter()
            .zip(s1_norms.iter())
            .map(|(c, &s1n)| {
                let s2 = fast_ce.score_pair(sp.query, c, s1n);
                heavy_ce.score_pair(sp.query, c, s2)
            })
            .collect();
        out.push(TeacherJudgment {
            query_id: sp.qid.to_string(),
            tenant_id: sp.tenant.to_string(),
            query: sp.query.to_string(),
            candidates: kept,
            stage1_norm_scores: s1_norms,
            teacher_scores: t_scores,
        });
    }
    out
}

/// Builds the 24-query multi-tenant benchmark (`14` easy, `5` ambiguous, `5` hard
/// multi-hop, plus `48` cross-tenant and insufficient-role negative controls).
pub fn default_multi_tenant_benchmark() -> Vec<BenchmarkQueryCase> {
    use QueryDifficulty::{Ambiguous, Easy, HardMultiHop};
    struct BenchSpec<'a> {
        qid: &'a str,
        tenant: &'a str,
        diff: QueryDifficulty,
        query: &'a str,
        facts: &'a [&'a str],
        gold: &'a str,
        s1_gold: f64,
        dist: &'a str,
        s1_dist: f64,
    }
    let specs: &[BenchSpec<'_>] = &[
        BenchSpec {
            qid: "q01",
            tenant: "tenant_acme",
            diff: Easy,
            query: "What checksum algorithm validates snapshot manifest segments \
                    during cold restore?",
            facts: &["crc32c", "manifest"],
            gold: "Snapshot manifest segments validate cold restore integrity \
                   using hardware-accelerated crc32c checksums.",
            s1_gold: 0.91,
            dist: "Archive background workers schedule cold storage segment merges \
                   across object buckets.",
            s1_dist: 0.84,
        },
        BenchSpec {
            qid: "q02",
            tenant: "tenant_acme",
            diff: Easy,
            query: "Which compression codec encodes roaring bitmap posting lists \
                    in shard snapshots?",
            facts: &["zstd", "roaring"],
            gold: "Shard snapshots encode roaring bitmap posting lists \
                   with dictionary-trained zstd frames.",
            s1_gold: 0.90,
            dist: "Posting list cursors iterate over document IDs in ascending order \
                   during conjunctive scans.",
            s1_dist: 0.85,
        },
        BenchSpec {
            qid: "q03",
            tenant: "tenant_globex",
            diff: Easy,
            query: "How are Matryoshka truncated query vectors normalized \
                    before cosine similarity search?",
            facts: &["l2", "matryoshka"],
            gold: "Matryoshka truncated query vectors enforce unit l2 normalization \
                   before cosine similarity search.",
            s1_gold: 0.92,
            dist: "Cosine distance kernels dispatch AVX2 fused multiply-add instructions \
                   over contiguous float slices.",
            s1_dist: 0.83,
        },
        BenchSpec {
            qid: "q04",
            tenant: "tenant_globex",
            diff: Easy,
            query: "Where does index-scheduler persist pending batch task queue entries?",
            facts: &["lmdb", "index-scheduler"],
            gold: "The index-scheduler crate persists pending batch task queue entries \
                   inside an atomic lmdb environment.",
            s1_gold: 0.89,
            dist: "Batch worker pools poll pending jobs from priority queues \
                   during peak ingestion.",
            s1_dist: 0.84,
        },
        BenchSpec {
            qid: "q05",
            tenant: "tenant_acme",
            diff: Easy,
            query: "What hash digest pins immutable embedding model revisions \
                    in HIP-1346 metadata?",
            facts: &["sha256", "model_revision"],
            gold: "HIP-1346 metadata pins immutable model_revision fields \
                   using canonical sha256 content digests.",
            s1_gold: 0.91,
            dist: "Embedding pipelines batch tokenized passages \
                   according to maximum sequence length.",
            s1_dist: 0.82,
        },
        BenchSpec {
            qid: "q06",
            tenant: "tenant_globex",
            diff: Easy,
            query: "Which tokenizer library segments CJK and multi-script text in milli?",
            facts: &["charabia", "cjk"],
            gold: "The milli engine delegates CJK and multi-script text segmentation \
                   to the charabia library.",
            s1_gold: 0.72,
            dist: "Multi-script lexical analyzers normalize Unicode diacritics \
                   during query parsing.",
            s1_dist: 0.88,
        },
        BenchSpec {
            qid: "q07",
            tenant: "tenant_acme",
            diff: Easy,
            query: "What file format stores immutable memory-mapped grenad sort runs?",
            facts: &["grenad", "mmap"],
            gold: "External merge sorters write immutable memory-mapped grenad sort runs \
                   for zero-copy mmap merging.",
            s1_gold: 0.73,
            dist: "Temporary merge directories purge spilled sort buffers \
                   once indexing transactions commit.",
            s1_dist: 0.89,
        },
        BenchSpec {
            qid: "q08",
            tenant: "tenant_globex",
            diff: Easy,
            query: "How does search-auth sign scoped API keys for authorized index patterns?",
            facts: &["hmac", "search-auth"],
            gold: "The search-auth module signs scoped API keys with hmac digests \
                   bound to authorized index patterns.",
            s1_gold: 0.71,
            dist: "Gateway middleware parses bearer authorization headers \
                   on incoming HTTP search requests.",
            s1_dist: 0.88,
        },
        BenchSpec {
            qid: "q09",
            tenant: "tenant_acme",
            diff: Easy,
            query: "Which global allocator replaces the default system heap in release binaries?",
            facts: &["mimalloc", "heap"],
            gold: "Release binaries configure mimalloc as the global Rust allocator \
                   to reduce multi-threaded heap contention.",
            s1_gold: 0.72,
            dist: "Thread-local scratch buffers amortize small vector allocations \
                   across search queries.",
            s1_dist: 0.87,
        },
        BenchSpec {
            qid: "q10",
            tenant: "tenant_globex",
            diff: Easy,
            query: "What spatial tree indexes geo-point bounding box filters in milli?",
            facts: &["rtree", "bounding"],
            gold: "Geo-point spatial filters index latitude and longitude coordinates \
                   inside a packed rtree bounding hierarchy.",
            s1_gold: 0.73,
            dist: "Facet filter evaluators intersect roaring bitmaps \
                   across conjunctive predicate clauses.",
            s1_dist: 0.89,
        },
        BenchSpec {
            qid: "q11",
            tenant: "tenant_acme",
            diff: Easy,
            query: "How does dumpless upgrade preserve live search availability during migration?",
            facts: &["shadow", "dumpless"],
            gold: "Experimental dumpless upgrade builds shadow index structures concurrently \
                   so live search stays available during migration.",
            s1_gold: 0.72,
            dist: "Rolling cluster restarts drain active HTTP connections \
                   over a configurable grace interval.",
            s1_dist: 0.88,
        },
        BenchSpec {
            qid: "q12",
            tenant: "tenant_globex",
            diff: Easy,
            query: "Which hardware instruction accelerates binary quantized Hamming distance \
                    in stage one?",
            facts: &["popcnt", "hamming"],
            gold: "Binary quantized vectors accelerate stage-one hamming distance \
                   using hardware xor popcnt instructions.",
            s1_gold: 0.71,
            dist: "Scalar int8 quantization calibrates symmetric per-channel clipping bounds \
                   on sample vectors.",
            s1_dist: 0.89,
        },
        BenchSpec {
            qid: "q13",
            tenant: "tenant_acme",
            diff: Easy,
            query: "What recovery probe state guards remote object-storage \
                    chunk circuit breakers?",
            facts: &["half-open", "backoff"],
            gold: "Remote object-storage chunk circuit breakers transition through \
                   a half-open recovery probe state with exponential backoff.",
            s1_gold: 0.73,
            dist: "Object store clients pool TLS keepalive sockets \
                   across regional storage endpoints.",
            s1_dist: 0.88,
        },
        BenchSpec {
            qid: "q14",
            tenant: "tenant_globex",
            diff: Easy,
            query: "How does filter-parser represent disjunctive facet expressions \
                    before roaring evaluation?",
            facts: &["ast", "filter-parser"],
            gold: "The filter-parser crate compiles disjunctive facet expressions \
                   into a normalized ast before roaring bitmap evaluation.",
            s1_gold: 0.72,
            dist: "Numeric range predicates scan B-tree dictionary pages during query planning.",
            s1_dist: 0.88,
        },
        BenchSpec {
            qid: "q15",
            tenant: "tenant_acme",
            diff: Ambiguous,
            query: "Why must read-only replica nodes not execute write-ahead log truncation?",
            facts: &["never", "leader"],
            gold: "Read-only replica nodes must never execute write-ahead log truncation \
                   because only the leader advances commit watermarks.",
            s1_gold: 0.74,
            dist: "Read-only replica nodes execute write-ahead log truncation \
                   immediately during local checkpoint compaction.",
            s1_dist: 0.94,
        },
        BenchSpec {
            qid: "q16",
            tenant: "tenant_globex",
            diff: Ambiguous,
            query: "What heartbeat timeout in milliseconds triggers raft lease step-down \
                    in protocol 2.4?",
            facts: &["2.4", "350"],
            gold: "In protocol 2.4, raft lease step-down triggers when follower heartbeat \
                   timeout exceeds 350 milliseconds.",
            s1_gold: 0.74,
            dist: "In protocol 1.8, raft lease step-down triggers when follower heartbeat \
                   timeout exceeds 1200 milliseconds.",
            s1_dist: 0.94,
        },
        BenchSpec {
            qid: "q17",
            tenant: "tenant_acme",
            diff: Ambiguous,
            query: "Why must synthetic HyDE passages never enter persistent canonical \
                    document collections?",
            facts: &["never", "ephemeral"],
            gold: "Synthetic HyDE passages must never enter persistent canonical document \
                   collections and remain strictly ephemeral query probes.",
            s1_gold: 0.73,
            dist: "Synthetic HyDE passages enter persistent canonical document collections \
                   directly to warm dense vector caches.",
            s1_dist: 0.94,
        },
        BenchSpec {
            qid: "q18",
            tenant: "tenant_globex",
            diff: Ambiguous,
            query: "What maximum Matryoshka embedding dimension is enforced \
                    by HIP-1346 contract version 3?",
            facts: &["3", "768"],
            gold: "HIP-1346 contract version 3 enforces a maximum Matryoshka \
                   embedding dimension of 768 floats.",
            s1_gold: 0.74,
            dist: "HIP-1346 contract version 1 enforces a maximum Matryoshka \
                   embedding dimension of 1536 floats.",
            s1_dist: 0.94,
        },
        BenchSpec {
            qid: "q19",
            tenant: "tenant_acme",
            diff: Ambiguous,
            query: "Why cannot tenant access filters run after stage-two cross-encoder scoring?",
            facts: &["cannot", "leakage"],
            gold: "Tenant access filters cannot run after stage-two cross-encoder scoring \
                   because post-filtering risks timing and top-k leakage.",
            s1_gold: 0.73,
            dist: "Tenant access filters run after stage-two cross-encoder scoring \
                   to maximize approximate nearest neighbor recall.",
            s1_dist: 0.94,
        },
        BenchSpec {
            qid: "q20",
            tenant: "tenant_acme",
            diff: HardMultiHop,
            query: "How does HNSW graph compaction preserve tombstoned vector IDs \
                    while streaming quantized PQ8 codes to object storage?",
            facts: &["bitmap", "multipart"],
            gold: "HNSW graph compaction preserves tombstoned vector IDs in a roaring bitmap \
                   while streaming quantized PQ8 codes via multipart object storage uploads.",
            s1_gold: 0.73,
            dist: "HNSW graph compaction preserves tombstoned vector IDs \
                   in local HNSW graph vector tables.",
            s1_dist: 0.95,
        },
        BenchSpec {
            qid: "q21",
            tenant: "tenant_globex",
            diff: HardMultiHop,
            query: "How does Kai calibrate retrieve-or-answer probabilities \
                    while preserving Zen grounded synthesis isolation?",
            facts: &["ece", "non-generative"],
            gold: "Kai calibrates retrieve-or-answer probabilities via temperature scaling \
                   and ece tracking while remaining strictly non-generative \
                   for Zen grounded synthesis isolation.",
            s1_gold: 0.72,
            dist: "Kai calibrates retrieve-or-answer probabilities \
                   via retrieve-or-answer logit sweeps.",
            s1_dist: 0.95,
        },
        BenchSpec {
            qid: "q22",
            tenant: "tenant_acme",
            diff: HardMultiHop,
            query: "How does snapshot replication verify chunk Merkle roots \
                    before atomically swapping live LMDB pointers?",
            facts: &["merkle", "rename"],
            gold: "Snapshot replication verifies chunk merkle roots against signed manifests \
                   before atomically swapping live LMDB pointers via POSIX rename.",
            s1_gold: 0.73,
            dist: "Snapshot replication verifies chunk merkle roots \
                   during parallel chunk archive downloads.",
            s1_dist: 0.95,
        },
        BenchSpec {
            qid: "q23",
            tenant: "tenant_globex",
            diff: HardMultiHop,
            query: "How does hybrid RRF fusion combine BM25 lexical ranks \
                    while enforcing immutable vector prefix contracts?",
            facts: &["60", "validate_query"],
            gold: "Hybrid RRF fusion combines BM25 lexical ranks using reciprocal constant 60 \
                   while enforcing immutable vector prefix contracts through validate_query.",
            s1_gold: 0.72,
            dist: "Hybrid RRF fusion combines BM25 lexical ranks \
                   across weighted BM25 lexical fields.",
            s1_dist: 0.95,
        },
        BenchSpec {
            qid: "q24",
            tenant: "tenant_acme",
            diff: HardMultiHop,
            query: "How does corrective retrieval strip high-risk poisoned chunks \
                    before reformulating secondary disambiguation queries?",
            facts: &["kai_action_risk", "keywords"],
            gold: "Corrective retrieval strips high-risk poisoned chunks exceeding \
                   kai_action_risk thresholds before reformulating secondary \
                   disambiguation keywords.",
            s1_gold: 0.73,
            dist: "Corrective retrieval strips high-risk poisoned chunks \
                   during primary retrieval grading.",
            s1_dist: 0.95,
        },
    ];

    specs
        .iter()
        .map(|sp| {
            let role = if sp.tenant == "tenant_acme" { "index_ops" } else { "ml_engineer" };
            let mut grades = HashMap::new();
            grades.insert("gold_primary".to_string(), 3u8);
            BenchmarkQueryCase {
                query_id: sp.qid.to_string(),
                difficulty: sp.diff,
                access: AccessContext {
                    tenant_id: sp.tenant.to_string(),
                    principal_roles: vec!["search_reader".to_string(), role.to_string()],
                },
                query: sp.query.to_string(),
                raw_candidates: make_candidate_pool(
                    sp.qid,
                    sp.tenant,
                    role,
                    sp.gold,
                    sp.s1_gold,
                    sp.dist,
                    sp.s1_dist,
                ),
                relevance_grades: grades,
                required_answer_facts: sp.facts.iter().map(|s| (*s).to_string()).collect(),
            }
        })
        .collect()
}

/// Evaluates a single [`RerankPolicy`] across `cases` using a fixed [`ZenSynthesizer`]
/// and fixed downstream task-completion judge.
pub fn evaluate_policy(
    pipeline: &TieredRerankPipeline,
    policy: RerankPolicy,
    cases: &[BenchmarkQueryCase],
    fast_ce: &dyn FastCrossEncoder,
    heavy_ce: &dyn HeavyCrossEncoder,
    zen: &dyn ZenSynthesizer,
) -> Result<PolicyEvaluationReport, TieredRerankError> {
    if cases.is_empty() {
        return Err(TieredRerankError::EmptyBenchmarkSuite);
    }

    let n = cases.len();
    let mut ndcg_sum = 0.0f64;
    let mut recall_sum = 0.0f64;
    let mut mrr_sum = 0.0f64;
    let mut acc_sum = 0.0f64;
    let mut latencies = Vec::with_capacity(n);
    let mut total_rerank_tokens = 0usize;
    let mut stage2_rerank_tokens = 0usize;
    let mut stage3_rerank_tokens = 0usize;
    let mut compute_units = 0.0f64;
    let mut total_cost_usd = 0.0f64;
    let mut escalated_count = 0usize;
    let mut correct_passages = 0usize;
    let mut cross_tenant_leaks = 0usize;
    let mut acl_filtered_candidates = 0usize;

    for case in cases {
        let run = pipeline.run_query(
            policy,
            &case.access,
            &case.query,
            &case.raw_candidates,
            fast_ce,
            heavy_ce,
        )?;
        acl_filtered_candidates += run.access_filtered.len();
        if run.escalated_to_stage3 {
            escalated_count += 1;
        }
        for hit in &run.ranked_passages {
            if hit.identity.tenant_id != case.access.tenant_id {
                cross_tenant_leaks += 1;
            }
        }

        let top5_len = run.ranked_passages.len().min(5);
        let mut dcg = 0.0f64;
        for (rank_idx, hit) in run.ranked_passages[..top5_len].iter().enumerate() {
            let rel = case.relevance_grades.get(&hit.identity.passage_id).copied().unwrap_or(0);
            if rel > 0 {
                dcg += ((1u32 << rel) - 1) as f64 / ((rank_idx + 2) as f64).log2();
            }
        }
        let mut ideal: Vec<u8> = case.relevance_grades.values().copied().collect();
        ideal.sort_by(|a, b| b.cmp(a));
        let idcg: f64 = ideal
            .iter()
            .take(5)
            .enumerate()
            .map(|(idx, &rel)| ((1u32 << rel) - 1) as f64 / ((idx + 2) as f64).log2())
            .sum();
        if idcg > 0.0 {
            ndcg_sum += dcg / idcg;
        }

        let total_rel = case.relevance_grades.values().filter(|&&r| r > 0).count();
        if total_rel > 0 {
            let retrieved_rel = run.ranked_passages[..top5_len]
                .iter()
                .filter(|h| {
                    case.relevance_grades.get(&h.identity.passage_id).copied().unwrap_or(0) > 0
                })
                .count();
            recall_sum += retrieved_rel as f64 / total_rel as f64;
            correct_passages += retrieved_rel;
        }

        let rr = run
            .full_ranked_pool
            .iter()
            .position(|h| {
                case.relevance_grades.get(&h.identity.passage_id).copied().unwrap_or(0) > 0
            })
            .map_or(0.0, |idx| 1.0 / (idx + 1) as f64);
        mrr_sum += rr;

        let evidence_texts: Vec<String> =
            run.ranked_passages.iter().map(|p| p.text.clone()).collect();
        let zen_answer = zen.synthesize(&case.query, &evidence_texts, SynthesisTarget::ZenGrounded);
        let lower_ans = zen_answer.to_ascii_lowercase();
        let facts_ok = case
            .required_answer_facts
            .iter()
            .all(|f| lower_ans.contains(&f.to_ascii_lowercase()));
        let pol_ok = if is_negated(&case.query) { is_negated(&zen_answer) } else { true };
        if !zen_answer.is_empty() && facts_ok && pol_ok {
            acc_sum += 1.0;
        }

        latencies.push(run.latency_ms);
        total_rerank_tokens += run.total_rerank_tokens;
        stage2_rerank_tokens += run.stage2_rerank_tokens;
        stage3_rerank_tokens += run.stage3_rerank_tokens;
        compute_units += run.compute_units;
        total_cost_usd += run.cost_usd;
    }

    latencies.sort_by(|a, b| a.total_cmp(b));
    let p50_latency_ms = latencies[(n - 1) / 2];
    let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    let p95_latency_ms = latencies[p95_idx];
    let mean_latency_ms = latencies.iter().sum::<f64>() / n as f64;
    let cost_per_correct_passage_usd = if correct_passages > 0 {
        total_cost_usd / correct_passages as f64
    } else {
        f64::INFINITY
    };

    Ok(PolicyEvaluationReport {
        policy,
        query_count: n,
        ndcg_at_5: ndcg_sum / n as f64,
        recall_at_5: recall_sum / n as f64,
        mrr: mrr_sum / n as f64,
        answer_accuracy: acc_sum / n as f64,
        p50_latency_ms,
        p95_latency_ms,
        mean_latency_ms,
        total_rerank_tokens,
        stage2_rerank_tokens,
        stage3_rerank_tokens,
        compute_units,
        stage3_escalation_rate: escalated_count as f64 / n as f64,
        total_cost_usd,
        cost_per_correct_passage_usd,
        cross_tenant_leaks,
        acl_filtered_candidates,
    })
}

/// Runs the full matched 5-policy Pareto evaluation and teacher-to-student distillation
/// suite over `cases` and disjoint `training_judgments`.
pub fn evaluate_tiered_reranking(
    pipeline: &TieredRerankPipeline,
    training_judgments: &[TeacherJudgment],
    cases: &[BenchmarkQueryCase],
) -> Result<TieredBenchmarkSummary, TieredRerankError> {
    let fast_ce = CalibratedFastCrossEncoder::default();
    let heavy_ce = SpanAlignedHeavyCrossEncoder::default();
    let zen = DeterministicZenGrounded;

    let untrained_student = DistilledDomainReranker::untrained();
    let pre_student_eval = evaluate_policy(
        pipeline,
        RerankPolicy::AlwaysFastStage2,
        cases,
        &untrained_student,
        &heavy_ce,
        &zen,
    )?;

    let trainer = DistillationTrainer::default();
    let (distilled_student, distillation) = trainer.distill(training_judgments)?;
    let post_student_eval = evaluate_policy(
        pipeline,
        RerankPolicy::AlwaysFastStage2,
        cases,
        &distilled_student,
        &heavy_ce,
        &zen,
    )?;

    let stage1_only =
        evaluate_policy(pipeline, RerankPolicy::Stage1Only, cases, &fast_ce, &heavy_ce, &zen)?;
    let always_fast_stage2 = evaluate_policy(
        pipeline,
        RerankPolicy::AlwaysFastStage2,
        cases,
        &fast_ce,
        &heavy_ce,
        &zen,
    )?;
    let always_heavy_stage3 = evaluate_policy(
        pipeline,
        RerankPolicy::AlwaysHeavyStage3,
        cases,
        &fast_ce,
        &heavy_ce,
        &zen,
    )?;
    let tiered_cascade =
        evaluate_policy(pipeline, RerankPolicy::TieredCascade, cases, &fast_ce, &heavy_ce, &zen)?;
    let distilled_student_cascade = evaluate_policy(
        pipeline,
        RerankPolicy::DistilledStudentCascade,
        cases,
        &distilled_student,
        &heavy_ce,
        &zen,
    )?;

    let heavy_comp = always_heavy_stage3.compute_units.max(1e-9);
    let tiered_compute_savings_pct =
        ((always_heavy_stage3.compute_units - tiered_cascade.compute_units) / heavy_comp) * 100.0;
    let distilled_compute_savings_pct = ((always_heavy_stage3.compute_units
        - distilled_student_cascade.compute_units)
        / heavy_comp)
        * 100.0;

    Ok(TieredBenchmarkSummary {
        stage1_only,
        always_fast_stage2,
        always_heavy_stage3,
        tiered_cascade,
        distilled_student_cascade,
        distillation,
        student_lift: StudentDistillationLift {
            pre_ndcg_at_5: pre_student_eval.ndcg_at_5,
            post_ndcg_at_5: post_student_eval.ndcg_at_5,
            pre_mrr: pre_student_eval.mrr,
            post_mrr: post_student_eval.mrr,
            pre_recall_at_5: pre_student_eval.recall_at_5,
            post_recall_at_5: post_student_eval.recall_at_5,
            pre_answer_accuracy: pre_student_eval.answer_accuracy,
            post_answer_accuracy: post_student_eval.answer_accuracy,
        },
        tiered_compute_savings_pct,
        distilled_compute_savings_pct,
    })
}
