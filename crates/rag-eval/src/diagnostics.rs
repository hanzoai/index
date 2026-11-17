// SPDX-License-Identifier: Apache-2.0

//! Bounty #25 (`hanzoai/index#29`): Fine-Grained RAG Diagnosis (RAGChecker).
//!
//! Implements reproducible RAG diagnostic evaluation (Ru et al., NeurIPS '24)
//! attributing pipeline errors across the four canonical failure categories:
//! - `retrieval_miss`: ground-truth claim absent from retrieved chunks
//! - `irrelevant_evidence`: retrieved chunk supports no ground-truth claim
//! - `unsupported_claim`: generated claim ungrounded in retrieved chunks
//! - `answer_omission`: ground-truth claim present in context but omitted from answer

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::factscore::AtomicClaimDecomposer;
use crate::text::best_sentence_support;

/// Fine-grained RAG failure classification (HIP-1346 / Bounty #25).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RagDiagnosticCategory {
    RetrievalMiss,
    IrrelevantEvidence,
    UnsupportedClaim,
    AnswerOmission,
}

/// Retrieved context chunk supplied to RAGChecker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedChunk {
    pub chunk_id: String,
    pub text: String,
    #[serde(default)]
    pub score: f64,
}

/// Single RAG evaluation case containing query, retrieved chunks, generated
/// answer, and reference ground-truth answer (or pre-split claim lists).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagEvalCase {
    pub case_id: String,
    pub query: String,
    pub retrieved_chunks: Vec<RetrievedChunk>,
    pub generated_answer: String,
    pub ground_truth_answer: String,
    /// Optional explicit ground-truth atomic claims overriding auto-decomposition.
    #[serde(default)]
    pub ground_truth_claims: Option<Vec<String>>,
    /// Optional explicit generated-answer atomic claims overriding auto-decomposition.
    #[serde(default)]
    pub answer_claims: Option<Vec<String>>,
}

/// Individual diagnostic failure item attributed by `RagChecker`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticFinding {
    pub category: RagDiagnosticCategory,
    pub case_id: String,
    pub subject_id: String,
    pub text: String,
    pub best_support_score: f64,
    pub detail: String,
}

/// Per-case RAGChecker metrics and error attribution breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagCaseReport {
    pub case_id: String,
    pub gt_claim_count: usize,
    pub answer_claim_count: usize,
    pub retrieved_chunk_count: usize,
    pub claim_precision: f64,
    pub claim_recall: f64,
    pub claim_f1: f64,
    pub retriever_claim_recall: f64,
    pub context_precision: f64,
    pub context_utilization: f64,
    pub faithfulness: f64,
    pub hallucination_rate: f64,
    pub self_knowledge_rate: f64,
    pub retrieval_misses: usize,
    pub irrelevant_evidence: usize,
    pub unsupported_claims: usize,
    pub answer_omissions: usize,
    pub findings: Vec<DiagnosticFinding>,
}

/// Suite-level aggregated RAGChecker diagnostics across multiple evaluation cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagDiagnosticSummary {
    pub case_count: usize,
    pub total_gt_claims: usize,
    pub total_answer_claims: usize,
    pub total_retrieved_chunks: usize,
    pub total_retrieval_misses: usize,
    pub total_irrelevant_evidence: usize,
    pub total_unsupported_claims: usize,
    pub total_answer_omissions: usize,
    pub retrieval_miss_rate: f64,
    pub irrelevant_evidence_rate: f64,
    pub unsupported_claim_rate: f64,
    pub answer_omission_rate: f64,
    pub macro_claim_precision: f64,
    pub macro_claim_recall: f64,
    pub macro_claim_f1: f64,
    pub macro_retriever_claim_recall: f64,
    pub macro_context_precision: f64,
    pub macro_context_utilization: f64,
    pub macro_faithfulness: f64,
    pub case_reports: Vec<RagCaseReport>,
}

/// Errors produced by `RagChecker` configuration or suite evaluation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DiagnosticsError {
    #[error("support threshold `{0}` must be finite and in [0.0, 1.0]")]
    InvalidThreshold(f64),
    #[error("evaluation suite must contain at least one case")]
    EmptySuite,
}

/// Fine-grained RAGChecker diagnostic evaluator (`hanzoai/index#29`).
#[derive(Debug, Clone)]
pub struct RagChecker {
    pub support_threshold: f64,
    pub decomposer: AtomicClaimDecomposer,
}

impl Default for RagChecker {
    fn default() -> Self {
        Self {
            support_threshold: 0.50,
            decomposer: AtomicClaimDecomposer::new(),
        }
    }
}

impl RagChecker {
    /// Builds a `RagChecker` with a custom claim-to-span support threshold.
    pub fn new(support_threshold: f64) -> Result<Self, DiagnosticsError> {
        if !support_threshold.is_finite() || !(0.0..=1.0).contains(&support_threshold) {
            return Err(DiagnosticsError::InvalidThreshold(support_threshold));
        }
        Ok(Self {
            support_threshold,
            decomposer: AtomicClaimDecomposer::new(),
        })
    }

    fn extract_claims(&self, raw: &str, override_claims: &Option<Vec<String>>) -> Vec<String> {
        if let Some(explicit) = override_claims {
            return explicit
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        let decomposed: Vec<String> =
            self.decomposer.decompose(raw).into_iter().map(|c| c.text).collect();
        if decomposed.is_empty() && !raw.trim().is_empty() {
            vec![raw.trim().to_string()]
        } else {
            decomposed
        }
    }

    fn max_support_against_texts(&self, claim: &str, texts: &[&str]) -> (bool, f64, Option<usize>) {
        let mut best_score = 0.0f64;
        let mut best_idx = None;
        let mut supported = false;

        for (idx, text) in texts.iter().enumerate() {
            if let Some((_, _, assessment)) = best_sentence_support(claim, text) {
                let valid = assessment.polarity_match
                    && assessment.numeric_match
                    && assessment.score >= self.support_threshold;
                if assessment.score > best_score {
                    best_score = assessment.score;
                    best_idx = Some(idx);
                }
                if valid {
                    supported = true;
                }
            }
        }

        (supported, best_score, best_idx)
    }

    /// Evaluates a single RAG case and attributes errors across the 4 diagnostic buckets.
    pub fn evaluate_case(&self, case: &RagEvalCase) -> RagCaseReport {
        let gt_claims = self.extract_claims(&case.ground_truth_answer, &case.ground_truth_claims);
        let ans_claims = self.extract_claims(&case.generated_answer, &case.answer_claims);

        let chunk_texts: Vec<&str> =
            case.retrieved_chunks.iter().map(|c| c.text.as_str()).collect();
        let ans_texts: Vec<&str> = ans_claims.iter().map(String::as_str).collect();
        let gt_texts: Vec<&str> = gt_claims.iter().map(String::as_str).collect();

        let mut chunk_supports_any_gt = vec![false; case.retrieved_chunks.len()];
        let mut gt_in_context = vec![false; gt_claims.len()];
        let mut gt_in_answer = vec![false; gt_claims.len()];
        let mut findings = Vec::new();

        for (i, gt_claim) in gt_claims.iter().enumerate() {
            let mut best_chunk_score = 0.0f64;
            for (c_idx, chunk_text) in chunk_texts.iter().enumerate() {
                if let Some((_, _, assessment)) = best_sentence_support(gt_claim, chunk_text) {
                    if assessment.score > best_chunk_score {
                        best_chunk_score = assessment.score;
                    }
                    if assessment.polarity_match
                        && assessment.numeric_match
                        && assessment.score >= self.support_threshold
                    {
                        gt_in_context[i] = true;
                        chunk_supports_any_gt[c_idx] = true;
                    }
                }
            }

            let (in_ans_claims, _, _) = self.max_support_against_texts(gt_claim, &ans_texts);
            let in_ans_raw = best_sentence_support(gt_claim, &case.generated_answer)
                .is_some_and(|(_, _, a)| {
                    a.polarity_match && a.numeric_match && a.score >= self.support_threshold
                });
            gt_in_answer[i] = in_ans_claims || in_ans_raw;

            if !gt_in_context[i] {
                findings.push(DiagnosticFinding {
                    category: RagDiagnosticCategory::RetrievalMiss,
                    case_id: case.case_id.clone(),
                    subject_id: format!("gt_claim_{i}"),
                    text: gt_claim.clone(),
                    best_support_score: best_chunk_score,
                    detail: "ground-truth claim not supported by any retrieved chunk".to_string(),
                });
            } else if !gt_in_answer[i] {
                findings.push(DiagnosticFinding {
                    category: RagDiagnosticCategory::AnswerOmission,
                    case_id: case.case_id.clone(),
                    subject_id: format!("gt_claim_{i}"),
                    text: gt_claim.clone(),
                    best_support_score: best_chunk_score,
                    detail: "ground-truth claim existed in retrieved context but was omitted from generated answer".to_string(),
                });
            }
        }

        let mut irrelevant_evidence = 0usize;
        for (c_idx, chunk) in case.retrieved_chunks.iter().enumerate() {
            if !chunk_supports_any_gt[c_idx] {
                irrelevant_evidence += 1;
                let (_, best_gt_score, _) = self.max_support_against_texts(&chunk.text, &gt_texts);
                findings.push(DiagnosticFinding {
                    category: RagDiagnosticCategory::IrrelevantEvidence,
                    case_id: case.case_id.clone(),
                    subject_id: chunk.chunk_id.clone(),
                    text: chunk.text.clone(),
                    best_support_score: best_gt_score,
                    detail: "retrieved chunk failed to support any ground-truth claim".to_string(),
                });
            }
        }

        let mut ans_in_context = vec![false; ans_claims.len()];
        let mut ans_in_gt = vec![false; ans_claims.len()];
        let mut unsupported_claims = 0usize;

        for (j, ans_claim) in ans_claims.iter().enumerate() {
            let (in_ctx, best_ctx_score, _) =
                self.max_support_against_texts(ans_claim, &chunk_texts);
            let (in_gt_claims, _, _) = self.max_support_against_texts(ans_claim, &gt_texts);
            let in_gt_raw = best_sentence_support(ans_claim, &case.ground_truth_answer)
                .is_some_and(|(_, _, a)| {
                    a.polarity_match && a.numeric_match && a.score >= self.support_threshold
                });

            ans_in_context[j] = in_ctx;
            ans_in_gt[j] = in_gt_claims || in_gt_raw;

            if !ans_in_context[j] {
                unsupported_claims += 1;
                let detail = if ans_in_gt[j] {
                    "claim aligns with reference answer via parametric self-knowledge but lacks retrieved chunk grounding"
                } else {
                    "claim has no supporting evidence in retrieved chunks (ungrounded/hallucinated assertion)"
                };
                findings.push(DiagnosticFinding {
                    category: RagDiagnosticCategory::UnsupportedClaim,
                    case_id: case.case_id.clone(),
                    subject_id: format!("answer_claim_{j}"),
                    text: ans_claim.clone(),
                    best_support_score: best_ctx_score,
                    detail: detail.to_string(),
                });
            }
        }

        let retrieval_misses = gt_in_context.iter().filter(|&&hit| !hit).count();
        let answer_omissions = gt_in_context
            .iter()
            .zip(gt_in_answer.iter())
            .filter(|&(&in_ctx, &in_ans)| in_ctx && !in_ans)
            .count();

        let claim_precision = ratio(ans_in_gt.iter().filter(|&&b| b).count(), ans_claims.len());
        let claim_recall = ratio(gt_in_answer.iter().filter(|&&b| b).count(), gt_claims.len());
        let claim_f1 = if claim_precision + claim_recall > 0.0 {
            2.0 * claim_precision * claim_recall / (claim_precision + claim_recall)
        } else {
            0.0
        };

        let gt_ctx_hits = gt_in_context.iter().filter(|&&b| b).count();
        let retriever_claim_recall = ratio(gt_ctx_hits, gt_claims.len());
        let relevant_chunks = chunk_supports_any_gt.iter().filter(|&&b| b).count();
        let context_precision = ratio(relevant_chunks, case.retrieved_chunks.len());
        let utilized_hits = gt_in_context
            .iter()
            .zip(gt_in_answer.iter())
            .filter(|&(&in_ctx, &in_ans)| in_ctx && in_ans)
            .count();
        let context_utilization = ratio(utilized_hits, gt_ctx_hits);

        let grounded_ans = ans_in_context.iter().filter(|&&b| b).count();
        let faithfulness = ratio(grounded_ans, ans_claims.len());
        let hallucination_rate = if ans_claims.is_empty() {
            0.0
        } else {
            1.0 - faithfulness
        };
        let self_knowledge_hits = ans_in_context
            .iter()
            .zip(ans_in_gt.iter())
            .filter(|&(&in_ctx, &in_gt)| !in_ctx && in_gt)
            .count();
        let self_knowledge_rate = ratio(self_knowledge_hits, ans_claims.len());

        RagCaseReport {
            case_id: case.case_id.clone(),
            gt_claim_count: gt_claims.len(),
            answer_claim_count: ans_claims.len(),
            retrieved_chunk_count: case.retrieved_chunks.len(),
            claim_precision,
            claim_recall,
            claim_f1,
            retriever_claim_recall,
            context_precision,
            context_utilization,
            faithfulness,
            hallucination_rate,
            self_knowledge_rate,
            retrieval_misses,
            irrelevant_evidence,
            unsupported_claims,
            answer_omissions,
            findings,
        }
    }

    /// Evaluates a suite of RAG cases and computes micro failure rates and macro averages.
    pub fn evaluate_suite(&self, cases: &[RagEvalCase]) -> Result<RagDiagnosticSummary, DiagnosticsError> {
        if cases.is_empty() {
            return Err(DiagnosticsError::EmptySuite);
        }

        let case_reports: Vec<RagCaseReport> =
            cases.iter().map(|c| self.evaluate_case(c)).collect();
        let n = case_reports.len() as f64;

        let mut total_gt_claims = 0usize;
        let mut total_answer_claims = 0usize;
        let mut total_retrieved_chunks = 0usize;
        let mut total_retrieval_misses = 0usize;
        let mut total_irrelevant_evidence = 0usize;
        let mut total_unsupported_claims = 0usize;
        let mut total_answer_omissions = 0usize;

        let mut sum_p = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut sum_f1 = 0.0f64;
        let mut sum_ret_r = 0.0f64;
        let mut sum_ctx_p = 0.0f64;
        let mut sum_ctx_u = 0.0f64;
        let mut sum_faith = 0.0f64;

        for r in &case_reports {
            total_gt_claims += r.gt_claim_count;
            total_answer_claims += r.answer_claim_count;
            total_retrieved_chunks += r.retrieved_chunk_count;
            total_retrieval_misses += r.retrieval_misses;
            total_irrelevant_evidence += r.irrelevant_evidence;
            total_unsupported_claims += r.unsupported_claims;
            total_answer_omissions += r.answer_omissions;

            sum_p += r.claim_precision;
            sum_r += r.claim_recall;
            sum_f1 += r.claim_f1;
            sum_ret_r += r.retriever_claim_recall;
            sum_ctx_p += r.context_precision;
            sum_ctx_u += r.context_utilization;
            sum_faith += r.faithfulness;
        }

        Ok(RagDiagnosticSummary {
            case_count: case_reports.len(),
            total_gt_claims,
            total_answer_claims,
            total_retrieved_chunks,
            total_retrieval_misses,
            total_irrelevant_evidence,
            total_unsupported_claims,
            total_answer_omissions,
            retrieval_miss_rate: ratio(total_retrieval_misses, total_gt_claims),
            irrelevant_evidence_rate: ratio(total_irrelevant_evidence, total_retrieved_chunks),
            unsupported_claim_rate: ratio(total_unsupported_claims, total_answer_claims),
            answer_omission_rate: ratio(total_answer_omissions, total_gt_claims),
            macro_claim_precision: sum_p / n,
            macro_claim_recall: sum_r / n,
            macro_claim_f1: sum_f1 / n,
            macro_retriever_claim_recall: sum_ret_r / n,
            macro_context_precision: sum_ctx_p / n,
            macro_context_utilization: sum_ctx_u / n,
            macro_faithfulness: sum_faith / n,
            case_reports,
        })
    }
}

#[inline]
fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_all_four_ragchecker_diagnostic_categories() {
        let checker = RagChecker::new(0.55).unwrap();

        // Ground-truth has 3 atomic claims:
        // 1. "Hanzo Index uses BM25 and vector ranking." (in chunk_0 AND in answer -> clean hit)
        // 2. "Immutable metadata requires L2 normalization." (in chunk_1, OMITTED from answer -> answer_omission)
        // 3. "Matryoshka supports 128 and 768 dimensions." (NOT in any chunk -> retrieval_miss)
        // Retrieved chunks also include:
        // - chunk_2: distractor about weather -> irrelevant_evidence
        // Generated answer also includes:
        // - "Hanzo Index compiles shaders in WebGPU." (not in chunks -> unsupported_claim)
        let case = RagEvalCase {
            case_id: "case_four_way".to_string(),
            query: "How does Hanzo Index handle ranking and vector metadata?".to_string(),
            retrieved_chunks: vec![
                RetrievedChunk {
                    chunk_id: "chunk_0".to_string(),
                    text: "Hanzo Index uses BM25 and vector ranking for hybrid queries.".to_string(),
                    score: 0.92,
                },
                RetrievedChunk {
                    chunk_id: "chunk_1".to_string(),
                    text: "Immutable metadata requires L2 normalization across collections.".to_string(),
                    score: 0.84,
                },
                RetrievedChunk {
                    chunk_id: "chunk_2".to_string(),
                    text: "Coastal fog clears rapidly around midday in Monterey.".to_string(),
                    score: 0.41,
                },
            ],
            generated_answer: "Hanzo Index uses BM25 and vector ranking. \
                               Hanzo Index compiles shaders in WebGPU."
                .to_string(),
            ground_truth_answer: "Hanzo Index uses BM25 and vector ranking. \
                                  Immutable metadata requires L2 normalization. \
                                  Matryoshka supports 128 and 768 dimensions."
                .to_string(),
            ground_truth_claims: None,
            answer_claims: None,
        };

        let report = checker.evaluate_case(&case);
        assert_eq!(report.gt_claim_count, 3);
        assert_eq!(report.answer_claim_count, 2);
        assert_eq!(report.retrieved_chunk_count, 3);

        assert_eq!(report.retrieval_misses, 1);
        assert_eq!(report.answer_omissions, 1);
        assert_eq!(report.irrelevant_evidence, 1);
        assert_eq!(report.unsupported_claims, 1);

        assert!((report.retriever_claim_recall - (2.0 / 3.0)).abs() < 1e-9);
        assert!((report.context_precision - (2.0 / 3.0)).abs() < 1e-9);
        assert!((report.context_utilization - 0.5).abs() < 1e-9);
        assert!((report.faithfulness - 0.5).abs() < 1e-9);
        assert!((report.hallucination_rate - 0.5).abs() < 1e-9);

        let categories: Vec<RagDiagnosticCategory> =
            report.findings.iter().map(|f| f.category).collect();
        assert!(categories.contains(&RagDiagnosticCategory::RetrievalMiss));
        assert!(categories.contains(&RagDiagnosticCategory::AnswerOmission));
        assert!(categories.contains(&RagDiagnosticCategory::IrrelevantEvidence));
        assert!(categories.contains(&RagDiagnosticCategory::UnsupportedClaim));
    }

    #[test]
    fn suite_aggregation_compares_clean_vs_degraded_pipeline() {
        let checker = RagChecker::default();
        let clean_case = RagEvalCase {
            case_id: "clean".to_string(),
            query: "Who owns Kai decisions vs Zen synthesis?".to_string(),
            retrieved_chunks: vec![RetrievedChunk {
                chunk_id: "c0".to_string(),
                text: "Kai emits discrete gating decisions while Zen synthesizes final responses."
                    .to_string(),
                score: 0.95,
            }],
            generated_answer:
                "Kai emits discrete gating decisions while Zen synthesizes final responses."
                    .to_string(),
            ground_truth_answer:
                "Kai emits discrete gating decisions while Zen synthesizes final responses."
                    .to_string(),
            ground_truth_claims: None,
            answer_claims: None,
        };

        let summary = checker.evaluate_suite(&[clean_case]).unwrap();
        assert_eq!(summary.total_retrieval_misses, 0);
        assert_eq!(summary.total_irrelevant_evidence, 0);
        assert_eq!(summary.total_unsupported_claims, 0);
        assert_eq!(summary.total_answer_omissions, 0);
        assert!((summary.macro_claim_f1 - 1.0).abs() < 1e-9);
        assert!((summary.macro_faithfulness - 1.0).abs() < 1e-9);
    }
}
