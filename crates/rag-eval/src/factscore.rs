// SPDX-License-Identifier: Apache-2.0

//! Bounty #30 (`hanzoai/index#31`): Atomic Factuality Verification (FActScore)
//! and Verifier Calibration (ECE).
//!
//! Decomposes Zen-generated responses into atomic testable claims, binds
//! supporting evidence spans from retrieved documents, measures supported-claim
//! precision (`FActScore`, Min et al., EMNLP '23), accepts structured Kai
//! verification decisions, and computes Expected Calibration Error (ECE).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::text::{best_sentence_support, split_sentences, tokenize_raw};

/// Common finite/auxiliary verbs used to detect clause predicates and carry
/// subject phrases across coordinated atomic clauses.
const VERB_INDICATORS: &[&str] = &[
    "allows", "are", "avoids", "binds", "builds", "can", "computes", "contains", "could",
    "decomposes", "detects", "did", "does", "emits", "enforces", "evaluates", "executes",
    "expands", "fails", "filters", "forces", "gates", "generates", "grades", "grounds", "had",
    "handles", "has", "have", "implements", "includes", "invokes", "is", "leaves", "maintains",
    "may", "measures", "might", "owns", "performs", "persists", "prevents", "produces",
    "provides", "records", "reduces", "rejects", "remains", "requires", "returns", "runs",
    "scores", "selects", "serves", "should", "skips", "splits", "stores", "supports",
    "synthesizes", "triggers", "truncates", "uses", "validates", "verifies", "was", "were",
    "will", "would",
];

/// A single atomic, self-contained factual assertion extracted from a Zen answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicClaim {
    pub claim_id: String,
    pub text: String,
    pub sentence_index: usize,
    pub char_span: (usize, usize),
}

/// Source document available to the span evidence binder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDocument {
    pub doc_id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub text: String,
}

/// Localized supporting (or contradicting) evidence span bound to a claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSpan {
    pub doc_id: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub excerpt: String,
    pub support_score: f64,
    pub polarity_match: bool,
    pub numeric_match: bool,
}

/// Verification outcome for a single atomic claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimVerdictStatus {
    Supported,
    Unsupported,
    Contradicted,
}

/// Provenance of a verification judgment (strictly non-generative).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierSource {
    SpanBinder,
    KaiGate,
}

/// Per-claim verification verdict produced by `FactScoreEvaluator`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimVerdict {
    pub claim: AtomicClaim,
    pub status: ClaimVerdictStatus,
    pub confidence: f64,
    pub bound_span: Option<EvidenceSpan>,
    pub verifier_source: VerifierSource,
}

/// Structured non-generative claim verification judgment emitted by Kai.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KaiClaimVerification {
    pub claim_id: String,
    pub p_supported: f64,
    #[serde(default)]
    pub contradicted: bool,
    #[serde(default)]
    pub bound_span: Option<EvidenceSpan>,
}

/// Summary report for atomic factuality evaluation (FActScore).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactScoreReport {
    pub total_claims: usize,
    pub supported_claims: usize,
    pub unsupported_claims: usize,
    pub contradicted_claims: usize,
    /// Supported-claim precision (`supported_claims / total_claims`).
    pub fact_score: f64,
    /// Mean verifier confidence across all atomic claims.
    pub mean_confidence: f64,
    pub verdicts: Vec<ClaimVerdict>,
}

impl FactScoreReport {
    /// Computes FActScore with the Min et al. (EMNLP '23) brevity penalty
    /// `exp(1 - gamma / total_claims)` when `total_claims < gamma`.
    pub fn length_penalized_fact_score(&self, gamma: usize) -> f64 {
        if self.total_claims == 0 {
            return 0.0;
        }
        if gamma == 0 || self.total_claims >= gamma {
            return self.fact_score;
        }
        let penalty = (1.0 - (gamma as f64 / self.total_claims as f64)).exp();
        self.fact_score * penalty
    }
}

/// Single calibrated prediction observation for ECE evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub confidence: f64,
    pub is_correct: bool,
}

/// One equal-width probability bin in `[lower, upper]` of a reliability diagram.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub bin_index: usize,
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub mean_confidence: f64,
    pub empirical_accuracy: f64,
    pub abs_gap: f64,
}

/// Calibration summary including Expected Calibration Error (ECE), Maximum
/// Calibration Error (MCE), and Brier score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub sample_count: usize,
    pub num_bins: usize,
    pub ece: f64,
    pub mce: f64,
    pub brier_score: f64,
    pub mean_confidence: f64,
    pub empirical_accuracy: f64,
    pub bins: Vec<CalibrationBin>,
}

/// Errors returned by FActScore evaluation or ECE calibration computation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FactScoreError {
    #[error("threshold `{0}` must be finite and in [0.0, 1.0]")]
    InvalidThreshold(f64),
    #[error("`num_bins` must be at least 1")]
    ZeroBins,
    #[error("calibration requires at least one sample")]
    EmptyCalibrationSamples,
    #[error("calibration confidence `{0}` at index {1} must be finite and in [0.0, 1.0]")]
    InvalidConfidence(f64, usize),
    #[error("missing Kai verification decision for claim `{0}`")]
    MissingKaiDecision(String),
}

/// Deterministic pure-Rust decomposer splitting Zen answers into atomic claims.
#[derive(Debug, Clone)]
pub struct AtomicClaimDecomposer {
    /// Minimum word count for a decomposed fragment to be kept as a claim.
    pub min_words: usize,
}

impl Default for AtomicClaimDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicClaimDecomposer {
    /// Creates a decomposer requiring at least 3 words per atomic claim.
    pub fn new() -> Self {
        Self { min_words: 3 }
    }

    /// Decomposes `response` into self-contained atomic claims with source spans.
    pub fn decompose(&self, response: &str) -> Vec<AtomicClaim> {
        let min_words = self.min_words.max(1);
        let mut claims = Vec::new();
        let sentences = split_sentences(response);

        for (sent_idx, (sent_start, sent_end, sentence)) in sentences.into_iter().enumerate() {
            let cleaned = strip_list_prefix(sentence).trim();
            if cleaned.is_empty() {
                continue;
            }
            let prefix_offset = sentence.find(cleaned).unwrap_or(0);
            let base_offset = sent_start + prefix_offset;

            let clauses = split_coordinated_clauses(cleaned);
            for (clause_offset, clause_end, clause_text) in clauses {
                let trimmed = clause_text.trim().trim_end_matches(['.', ';', ',']);
                if trimmed.split_whitespace().count() < min_words {
                    continue;
                }
                let claim_id = format!("c{}_{}", sent_idx, claims.len());
                let span_start = (base_offset + clause_offset).min(sent_end);
                let span_end = (base_offset + clause_end).min(sent_end);
                claims.push(AtomicClaim {
                    claim_id,
                    text: trimmed.to_string(),
                    sentence_index: sent_idx,
                    char_span: (span_start, span_end),
                });
            }
        }

        claims
    }
}

fn strip_list_prefix(s: &str) -> &str {
    let trimmed = s.trim_start();
    for marker in ["- ", "* ", "• "] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return rest;
        }
    }
    if let Some(dot_pos) = trimmed.find(". ") {
        if dot_pos > 0 && dot_pos <= 3 && trimmed[..dot_pos].bytes().all(|b| b.is_ascii_digit()) {
            return &trimmed[dot_pos + 2..];
        }
    }
    trimmed
}

fn looks_like_verb(word: &str) -> bool {
    let lower = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    VERB_INDICATORS.contains(&lower.as_str())
        || (lower.len() >= 5 && (lower.ends_with("izes") || lower.ends_with("ates")))
}

fn contains_finite_verb(clause: &str) -> bool {
    clause.split_whitespace().any(looks_like_verb)
}

fn extract_subject_prefix(clause: &str) -> Option<String> {
    let words: Vec<&str> = clause.split_whitespace().collect();
    let verb_idx = words.iter().position(|w| looks_like_verb(w))?;
    if verb_idx == 0 {
        return None;
    }
    Some(words[..verb_idx].join(" "))
}

fn starts_with_verb(clause: &str) -> bool {
    clause.split_whitespace().next().is_some_and(looks_like_verb)
}

/// Splits a sentence on semicolons and clause-level conjunctions (`; `, `, and `,
/// ` and `, ` while `, ` whereas `, `, which `) while carrying forward the
/// grammatical subject when a coordinated clause begins directly with a verb.
fn split_coordinated_clauses(sentence: &str) -> Vec<(usize, usize, String)> {
    let body = sentence.trim_end_matches(['.', '!', '?']);
    let mut semi_parts = Vec::new();
    let mut cursor = 0usize;
    for (idx, ch) in body.char_indices() {
        if ch == ';' {
            semi_parts.push((cursor, idx, &body[cursor..idx]));
            cursor = idx + 1;
        }
    }
    if cursor < body.len() {
        semi_parts.push((cursor, body.len(), &body[cursor..]));
    }

    let mut out = Vec::new();
    const CONJUNCTIONS: &[&str] = &[", and ", " and ", ", while ", " while ", ", whereas ", " whereas ", ", which "];

    for (part_start, part_end, part) in semi_parts {
        let trimmed_part = part.trim();
        if trimmed_part.is_empty() {
            continue;
        }
        let mut active_start = part_start;
        let mut current = trimmed_part;
        let mut inherited_subject = extract_subject_prefix(trimmed_part);

        loop {
            let mut earliest: Option<(usize, &str)> = None;
            for &conj in CONJUNCTIONS {
                if let Some(pos) = current.find(conj) {
                    if earliest.is_none_or(|(best_pos, _)| pos < best_pos) {
                        earliest = Some((pos, conj));
                    }
                }
            }

            let Some((split_pos, conj)) = earliest else {
                let text = reconstruct_clause(current, inherited_subject.as_deref(), false);
                out.push((active_start, part_end, text));
                break;
            };

            let left = current[..split_pos].trim().trim_end_matches(',');
            let right_raw = current[split_pos + conj.len()..].trim();

            // Only split if the right segment contains a predicate verb or is `which ...`
            // so noun enumerations like `"BM25 and vector search"` stay unified.
            let is_which = conj.contains("which");
            if left.split_whitespace().count() >= 2
                && right_raw.split_whitespace().count() >= 2
                && (is_which || contains_finite_verb(right_raw))
            {
                if inherited_subject.is_none() {
                    inherited_subject = extract_subject_prefix(left);
                }
                let left_end = active_start + split_pos;
                out.push((active_start, left_end, left.to_string()));
                active_start += split_pos + conj.len();
                current = right_raw;
                if is_which && starts_with_verb(current) {
                    let expanded = reconstruct_clause(current, inherited_subject.as_deref(), true);
                    out.push((active_start, part_end, expanded));
                    break;
                }
            } else {
                out.push((active_start, part_end, current.to_string()));
                break;
            }
        }
    }

    out
}

fn reconstruct_clause(clause: &str, subject: Option<&str>, force_subject: bool) -> String {
    let trimmed = clause.trim();
    if let Some(subj) = subject {
        if (force_subject || starts_with_verb(trimmed)) && !subj.is_empty() {
            return format!("{subj} {trimmed}");
        }
    }
    trimmed.to_string()
}

/// Deterministic span binder locating the strongest supporting (or conflicting)
/// evidence excerpt in retrieved documents for a given atomic claim.
#[derive(Debug, Clone)]
pub struct SpanEvidenceBinder {
    pub support_threshold: f64,
}

impl Default for SpanEvidenceBinder {
    fn default() -> Self {
        Self { support_threshold: 0.50 }
    }
}

impl SpanEvidenceBinder {
    /// Creates a `SpanEvidenceBinder` with custom lexical support threshold.
    pub fn new(support_threshold: f64) -> Result<Self, FactScoreError> {
        if !support_threshold.is_finite() || !(0.0..=1.0).contains(&support_threshold) {
            return Err(FactScoreError::InvalidThreshold(support_threshold));
        }
        Ok(Self { support_threshold })
    }

    /// Binds the best evidence span across `docs` for `claim`.
    pub fn bind(&self, claim: &AtomicClaim, docs: &[EvidenceDocument]) -> ClaimVerdict {
        let mut best: Option<(EvidenceSpan, f64)> = None;

        for doc in docs {
            if let Some((byte_start, byte_end, assessment)) =
                best_sentence_support(&claim.text, &doc.text)
            {
                let excerpt = doc.text[byte_start..byte_end].to_string();
                let span = EvidenceSpan {
                    doc_id: doc.doc_id.clone(),
                    byte_start,
                    byte_end,
                    excerpt,
                    support_score: assessment.score,
                    polarity_match: assessment.polarity_match,
                    numeric_match: assessment.numeric_match,
                };
                // Prefer highest support score; if contradicted with strong topic overlap,
                // keep track of the highest-overlap evidence span for auditability.
                let rank_key = if assessment.polarity_match && assessment.numeric_match {
                    assessment.score
                } else {
                    assessment.token_recall * 0.45
                };
                if best.as_ref().is_none_or(|(_, prev_key)| rank_key > *prev_key) {
                    best = Some((span, assessment.token_recall));
                }
            }
        }

        let Some((span, token_recall)) = best else {
            return ClaimVerdict {
                claim: claim.clone(),
                status: ClaimVerdictStatus::Unsupported,
                confidence: 0.05,
                bound_span: None,
                verifier_source: VerifierSource::SpanBinder,
            };
        };

        let is_contradicted =
            (!span.polarity_match || !span.numeric_match) && token_recall >= 0.45;
        let (status, confidence) = if is_contradicted {
            (ClaimVerdictStatus::Contradicted, (span.support_score * 0.4).clamp(0.02, 0.25))
        } else if span.support_score >= self.support_threshold {
            (ClaimVerdictStatus::Supported, span.support_score.clamp(0.50, 0.99))
        } else {
            (ClaimVerdictStatus::Unsupported, span.support_score.clamp(0.05, 0.49))
        };

        // Discard irrelevant bound spans below minimal lexical overlap so diagnostics stay clean.
        let bound_span = if span.support_score > 0.15 || is_contradicted {
            Some(span)
        } else {
            None
        };

        ClaimVerdict {
            claim: claim.clone(),
            status,
            confidence,
            bound_span,
            verifier_source: VerifierSource::SpanBinder,
        }
    }
}

/// Evaluates atomic factuality (FActScore) over Zen outputs using either the
/// pure-Rust span binder or structured non-generative Kai judgments.
#[derive(Debug, Clone, Default)]
pub struct FactScoreEvaluator {
    pub decomposer: AtomicClaimDecomposer,
    pub binder: SpanEvidenceBinder,
}

impl FactScoreEvaluator {
    /// Creates an evaluator with the specified claim support threshold in `[0.0, 1.0]`.
    pub fn new(support_threshold: f64) -> Result<Self, FactScoreError> {
        Ok(Self {
            decomposer: AtomicClaimDecomposer::new(),
            binder: SpanEvidenceBinder::new(support_threshold)?,
        })
    }

    /// Decomposes `zen_response`, binds spans from `docs`, and scores supported-claim precision.
    pub fn evaluate_with_evidence(
        &self,
        zen_response: &str,
        docs: &[EvidenceDocument],
    ) -> FactScoreReport {
        let claims = self.decomposer.decompose(zen_response);
        let verdicts: Vec<ClaimVerdict> =
            claims.iter().map(|c| self.binder.bind(c, docs)).collect();
        summarize_verdicts(verdicts)
    }

    /// Evaluates pre-decomposed or auto-decomposed claims from `zen_response` using
    /// structured Kai verification probabilities (`HIP-1346 §2` boundary).
    pub fn evaluate_with_kai(
        &self,
        zen_response: &str,
        kai_decisions: &[KaiClaimVerification],
    ) -> Result<FactScoreReport, FactScoreError> {
        let claims = self.decomposer.decompose(zen_response);
        self.evaluate_claims_with_kai(&claims, kai_decisions)
    }

    /// Evaluates explicit atomic claims against Kai verification judgments.
    pub fn evaluate_claims_with_kai(
        &self,
        claims: &[AtomicClaim],
        kai_decisions: &[KaiClaimVerification],
    ) -> Result<FactScoreReport, FactScoreError> {
        let by_id: HashMap<&str, &KaiClaimVerification> =
            kai_decisions.iter().map(|d| (d.claim_id.as_str(), d)).collect();
        let mut verdicts = Vec::with_capacity(claims.len());

        for (idx, claim) in claims.iter().enumerate() {
            let decision = by_id
                .get(claim.claim_id.as_str())
                .copied()
                .ok_or_else(|| FactScoreError::MissingKaiDecision(claim.claim_id.clone()))?;
            if !decision.p_supported.is_finite() || !(0.0..=1.0).contains(&decision.p_supported) {
                return Err(FactScoreError::InvalidConfidence(decision.p_supported, idx));
            }
            let status = if decision.contradicted {
                ClaimVerdictStatus::Contradicted
            } else if decision.p_supported >= self.binder.support_threshold {
                ClaimVerdictStatus::Supported
            } else {
                ClaimVerdictStatus::Unsupported
            };
            verdicts.push(ClaimVerdict {
                claim: claim.clone(),
                status,
                confidence: decision.p_supported,
                bound_span: decision.bound_span.clone(),
                verifier_source: VerifierSource::KaiGate,
            });
        }

        Ok(summarize_verdicts(verdicts))
    }
}

fn summarize_verdicts(verdicts: Vec<ClaimVerdict>) -> FactScoreReport {
    let total_claims = verdicts.len();
    let mut supported_claims = 0usize;
    let mut unsupported_claims = 0usize;
    let mut contradicted_claims = 0usize;
    let mut conf_sum = 0.0f64;

    for v in &verdicts {
        conf_sum += v.confidence;
        match v.status {
            ClaimVerdictStatus::Supported => supported_claims += 1,
            ClaimVerdictStatus::Unsupported => unsupported_claims += 1,
            ClaimVerdictStatus::Contradicted => contradicted_claims += 1,
        }
    }

    let fact_score = if total_claims == 0 {
        0.0
    } else {
        supported_claims as f64 / total_claims as f64
    };
    let mean_confidence = if total_claims == 0 {
        0.0
    } else {
        conf_sum / total_claims as f64
    };

    FactScoreReport {
        total_claims,
        supported_claims,
        unsupported_claims,
        contradicted_claims,
        fact_score,
        mean_confidence,
        verdicts,
    }
}

/// Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
/// and Brier score over `num_bins` equal-width confidence bins in `[0.0, 1.0]`.
pub fn compute_ece(
    samples: &[CalibrationSample],
    num_bins: usize,
) -> Result<CalibrationReport, FactScoreError> {
    if num_bins == 0 {
        return Err(FactScoreError::ZeroBins);
    }
    if samples.is_empty() {
        return Err(FactScoreError::EmptyCalibrationSamples);
    }

    let mut counts = vec![0usize; num_bins];
    let mut conf_sums = vec![0.0f64; num_bins];
    let mut acc_sums = vec![0.0f64; num_bins];

    let mut total_conf = 0.0f64;
    let mut total_acc = 0.0f64;
    let mut brier_sum = 0.0f64;

    for (idx, sample) in samples.iter().enumerate() {
        if !sample.confidence.is_finite() || !(0.0..=1.0).contains(&sample.confidence) {
            return Err(FactScoreError::InvalidConfidence(sample.confidence, idx));
        }
        let y = if sample.is_correct { 1.0 } else { 0.0 };
        let bin_idx = ((sample.confidence * num_bins as f64).floor() as usize).min(num_bins - 1);
        counts[bin_idx] += 1;
        conf_sums[bin_idx] += sample.confidence;
        acc_sums[bin_idx] += y;

        total_conf += sample.confidence;
        total_acc += y;
        let diff = sample.confidence - y;
        brier_sum += diff * diff;
    }

    let n = samples.len() as f64;
    let mut ece = 0.0f64;
    let mut mce = 0.0f64;
    let mut bins = Vec::with_capacity(num_bins);

    for bin_index in 0..num_bins {
        let lower = bin_index as f64 / num_bins as f64;
        let upper = (bin_index + 1) as f64 / num_bins as f64;
        let count = counts[bin_index];
        if count == 0 {
            bins.push(CalibrationBin {
                bin_index,
                lower,
                upper,
                count: 0,
                mean_confidence: 0.0,
                empirical_accuracy: 0.0,
                abs_gap: 0.0,
            });
            continue;
        }
        let mean_confidence = conf_sums[bin_index] / count as f64;
        let empirical_accuracy = acc_sums[bin_index] / count as f64;
        let abs_gap = (empirical_accuracy - mean_confidence).abs();
        ece += (count as f64 / n) * abs_gap;
        if abs_gap > mce {
            mce = abs_gap;
        }
        bins.push(CalibrationBin {
            bin_index,
            lower,
            upper,
            count,
            mean_confidence,
            empirical_accuracy,
            abs_gap,
        });
    }

    Ok(CalibrationReport {
        sample_count: samples.len(),
        num_bins,
        ece,
        mce,
        brier_score: brier_sum / n,
        mean_confidence: total_conf / n,
        empirical_accuracy: total_acc / n,
        bins,
    })
}

/// Checks whether a claim has enough content words to be meaningfully verified.
#[allow(dead_code)]
pub(crate) fn is_substantive_claim(text: &str) -> bool {
    tokenize_raw(text).len() >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposes_compound_zen_sentences_with_subject_carryover() {
        let decomposer = AtomicClaimDecomposer::new();
        let response = "Hanzo Index uses Rust and supports hybrid BM25 search; \
                        Kai emits structured decisions while Zen synthesizes grounded text.";
        let claims = decomposer.decompose(response);
        assert_eq!(claims.len(), 4, "expected 4 atomic claims, got {claims:?}");
        assert_eq!(claims[0].text, "Hanzo Index uses Rust");
        assert_eq!(claims[1].text, "Hanzo Index supports hybrid BM25 search");
        assert_eq!(claims[2].text, "Kai emits structured decisions");
        assert_eq!(claims[3].text, "Zen synthesizes grounded text");
    }

    #[test]
    fn binds_evidence_spans_and_detects_numeric_and_negation_contradictions() {
        let docs = vec![
            EvidenceDocument {
                doc_id: "doc_hip".to_string(),
                title: Some("HIP-1346".to_string()),
                text: "Hanzo Index persists immutable collection metadata with 768 Matryoshka dimensions. \
                       Kai emits structured discrete gating probabilities without generating text."
                    .to_string(),
            },
            EvidenceDocument {
                doc_id: "doc_zen".to_string(),
                title: None,
                text: "Zen handles response synthesis after Kai finishes retrieval gating.".to_string(),
            },
        ];

        let evaluator = FactScoreEvaluator::new(0.55).unwrap();
        let zen_answer = "Hanzo Index persists immutable collection metadata with 768 Matryoshka dimensions. \
                          Hanzo Index persists immutable collection metadata with 256 Matryoshka dimensions. \
                          Kai generates full prose responses directly. \
                          Hanzo Index supports quantum teleportation natively.";
        let report = evaluator.evaluate_with_evidence(zen_answer, &docs);

        assert_eq!(report.total_claims, 4);
        assert_eq!(report.supported_claims, 1);
        assert_eq!(report.contradicted_claims, 1); // 256 vs 768 dimensions
        assert_eq!(report.unsupported_claims, 2);
        assert!((report.fact_score - 0.25).abs() < 1e-9);

        let supported_span = report.verdicts[0].bound_span.as_ref().unwrap();
        assert_eq!(supported_span.doc_id, "doc_hip");
        assert!(supported_span.excerpt.contains("768 Matryoshka dimensions"));
        assert_eq!(report.verdicts[1].status, ClaimVerdictStatus::Contradicted);
    }

    #[test]
    fn kai_structured_verification_and_ece_calibration() {
        let evaluator = FactScoreEvaluator::new(0.6).unwrap();
        let zen_answer = "Hanzo Index supports hybrid search. Matryoshka uses 512 dims.";
        let claims = evaluator.decomposer.decompose(zen_answer);
        assert_eq!(claims.len(), 2);

        let kai_decisions = vec![
            KaiClaimVerification {
                claim_id: claims[0].claim_id.clone(),
                p_supported: 0.92,
                contradicted: false,
                bound_span: None,
            },
            KaiClaimVerification {
                claim_id: claims[1].claim_id.clone(),
                p_supported: 0.20,
                contradicted: false,
                bound_span: None,
            },
        ];
        let report = evaluator.evaluate_with_kai(zen_answer, &kai_decisions).unwrap();
        assert_eq!(report.supported_claims, 1);
        assert_eq!(report.unsupported_claims, 1);
        assert!((report.fact_score - 0.5).abs() < 1e-9);
        assert!(report.length_penalized_fact_score(4) < report.fact_score);

        // Well-calibrated samples yield small ECE; overconfident wrong samples yield high ECE.
        let calibrated = vec![
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: false },
            CalibrationSample { confidence: 0.20, is_correct: false },
            CalibrationSample { confidence: 0.20, is_correct: false },
            CalibrationSample { confidence: 0.20, is_correct: false },
            CalibrationSample { confidence: 0.20, is_correct: false },
            CalibrationSample { confidence: 0.20, is_correct: true },
        ];
        let cal_ok = compute_ece(&calibrated, 10).unwrap();
        assert!(cal_ok.ece < 1e-9, "expected exact 0.0 ECE, got {}", cal_ok.ece);

        let overconfident = vec![
            CalibrationSample { confidence: 0.95, is_correct: false },
            CalibrationSample { confidence: 0.92, is_correct: false },
            CalibrationSample { confidence: 0.88, is_correct: true },
            CalibrationSample { confidence: 0.90, is_correct: false },
        ];
        let cal_bad = compute_ece(&overconfident, 10).unwrap();
        assert!(cal_bad.ece > 0.60, "expected large overconfidence ECE, got {}", cal_bad.ece);
        assert!(cal_bad.brier_score > cal_ok.brier_score);
    }
}
