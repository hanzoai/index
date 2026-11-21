// SPDX-License-Identifier: Apache-2.0

//! Wave 1 #8 — Bounty #22 (`hanzoai/index#26`): Adaptive Retrieve-or-Answer
//! Policy (Self-RAG via Kai).
//!
//! Enforces the HIP-1346 §2 architectural split:
//! - **Kai** (`HIP-1332`) is strictly a non-generative decision head emitting
//!   calibrated discrete probabilities (`retrieve` vs `answer_directly`).
//! - **Zen** (`hanzoai/ml` / `hanzoai/engine`) strictly owns generative response
//!   synthesis, either directly (`ZenParametric`) or conditioned on retrieved
//!   index evidence (`ZenGrounded`).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::factscore::{compute_ece, CalibrationSample};

/// Non-generative decision head identifier (HIP-1346 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionHead {
    Kai,
}

/// Generative synthesis engine identifier (HIP-1346 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisTarget {
    /// Zen synthesizes directly from parametric knowledge without index retrieval.
    ZenParametric,
    /// Zen synthesizes grounded in retrieved index evidence.
    ZenGrounded,
}

/// Structured discrete action emitted by `RetrieveOrAnswerGate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalAction {
    Retrieve,
    AnswerDirectly,
}

/// Structured discrete probabilities emitted by Kai's retrieval-gate head.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KaiRetrieveProbs {
    pub p_retrieve: f64,
    pub p_answer_directly: f64,
}

impl KaiRetrieveProbs {
    /// Validates and constructs discrete gate probabilities summing to `1.0`.
    pub fn new(p_retrieve: f64, p_answer_directly: f64) -> Result<Self, GateError> {
        if !p_retrieve.is_finite()
            || !p_answer_directly.is_finite()
            || !(0.0..=1.0).contains(&p_retrieve)
            || !(0.0..=1.0).contains(&p_answer_directly)
        {
            return Err(GateError::InvalidProbability { p_retrieve, p_answer_directly });
        }
        let sum = p_retrieve + p_answer_directly;
        if (sum - 1.0).abs() > 1e-4 {
            return Err(GateError::ProbabilitiesDoNotSumToOne { sum });
        }
        Ok(Self {
            p_retrieve: p_retrieve / sum,
            p_answer_directly: p_answer_directly / sum,
        })
    }

    /// Computes normalized discrete probabilities from Kai's raw binary logits
    /// `[logit_retrieve, logit_answer_directly]` via numerically stable softmax.
    pub fn from_logits(logit_retrieve: f64, logit_answer_directly: f64) -> Result<Self, GateError> {
        if !logit_retrieve.is_finite() || !logit_answer_directly.is_finite() {
            return Err(GateError::InvalidProbability {
                p_retrieve: logit_retrieve,
                p_answer_directly: logit_answer_directly,
            });
        }
        let max_l = logit_retrieve.max(logit_answer_directly);
        let e_ret = (logit_retrieve - max_l).exp();
        let e_ans = (logit_answer_directly - max_l).exp();
        let denom = e_ret + e_ans;
        Ok(Self {
            p_retrieve: e_ret / denom,
            p_answer_directly: e_ans / denom,
        })
    }
}

/// Errors produced by `RetrieveOrAnswerGate` configuration or evaluation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GateError {
    #[error("threshold `theta`={0} must be finite and in [0.0, 1.0]")]
    InvalidThreshold(f64),
    #[error("invalid Kai probabilities (p_retrieve={p_retrieve}, p_answer_directly={p_answer_directly})")]
    InvalidProbability { p_retrieve: f64, p_answer_directly: f64 },
    #[error("Kai probabilities must sum to 1.0 (got {sum:.6})")]
    ProbabilitiesDoNotSumToOne { sum: f64 },
    #[error("simulation requires at least one query sample")]
    EmptySimulation,
}

/// Structured gate output preserving the Kai-decides / Zen-synthesizes split.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetrieveGateDecision {
    pub action: RetrievalAction,
    pub decision_head: DecisionHead,
    pub synthesis_target: SynthesisTarget,
    pub p_retrieve: f64,
    pub p_answer_directly: f64,
    pub theta: f64,
    pub margin: f64,
    pub skipped_retrieval: bool,
}

/// Calibrated retrieve-or-answer gate consuming Kai discrete probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetrieveOrAnswerGate {
    /// Minimum `p_retrieve` required to invoke index retrieval.
    pub theta: f64,
}

impl Default for RetrieveOrAnswerGate {
    fn default() -> Self {
        Self { theta: 0.5 }
    }
}

impl RetrieveOrAnswerGate {
    /// Creates a `RetrieveOrAnswerGate` calibrated at threshold `theta` in `[0.0, 1.0]`.
    pub fn new(theta: f64) -> Result<Self, GateError> {
        if !theta.is_finite() || !(0.0..=1.0).contains(&theta) {
            return Err(GateError::InvalidThreshold(theta));
        }
        Ok(Self { theta })
    }

    /// Emits a structured non-generative decision from Kai probabilities.
    ///
    /// Retrieval triggers iff `probs.p_retrieve >= self.theta`; otherwise index
    /// retrieval is skipped and Zen synthesizes directly (`ZenParametric`).
    pub fn evaluate(&self, probs: KaiRetrieveProbs) -> RetrieveGateDecision {
        let should_retrieve = probs.p_retrieve >= self.theta;
        let (action, synthesis_target, skipped_retrieval) = if should_retrieve {
            (RetrievalAction::Retrieve, SynthesisTarget::ZenGrounded, false)
        } else {
            (RetrievalAction::AnswerDirectly, SynthesisTarget::ZenParametric, true)
        };
        RetrieveGateDecision {
            action,
            decision_head: DecisionHead::Kai,
            synthesis_target,
            p_retrieve: probs.p_retrieve,
            p_answer_directly: probs.p_answer_directly,
            theta: self.theta,
            margin: probs.p_retrieve - self.theta,
            skipped_retrieval,
        }
    }
}

/// Non-generative Kai scoring interface (HIP-1346 §2).
pub trait KaiDecisionHead {
    fn score_retrieve_or_answer(&self, query: &str) -> Result<KaiRetrieveProbs, GateError>;
}

/// Generative Zen synthesis interface (HIP-1346 §2).
pub trait ZenSynthesizer {
    fn synthesize(&self, query: &str, retrieved_evidence: &[String], target: SynthesisTarget) -> String;
}

/// Outcome of routing a single query through Kai gating -> optional retrieval -> Zen synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedAnswer {
    pub decision: RetrieveGateDecision,
    pub retrieved_chunks: Vec<String>,
    pub zen_answer: String,
}

/// Routes a query strictly along the HIP-1346 boundary: Kai decides whether to
/// retrieve; retrieval executes only when `Retrieve`; Zen synthesizes in all cases.
pub fn route_with_kai_and_zen<F>(
    gate: &RetrieveOrAnswerGate,
    kai: &dyn KaiDecisionHead,
    zen: &dyn ZenSynthesizer,
    query: &str,
    mut retrieve_fn: F,
) -> Result<RoutedAnswer, GateError>
where
    F: FnMut(&str) -> Vec<String>,
{
    let probs = kai.score_retrieve_or_answer(query)?;
    let decision = gate.evaluate(probs);
    let retrieved_chunks = match decision.action {
        RetrievalAction::Retrieve => retrieve_fn(query),
        RetrievalAction::AnswerDirectly => Vec::new(),
    };
    let zen_answer = zen.synthesize(query, &retrieved_chunks, decision.synthesis_target);
    Ok(RoutedAnswer { decision, retrieved_chunks, zen_answer })
}

/// Unit cost and latency parameters for Self-RAG retrieval vs direct Zen synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetrievalCostModel {
    pub retrieval_cost_usd: f64,
    pub retrieval_latency_ms: f64,
    pub zen_grounded_cost_usd: f64,
    pub zen_direct_cost_usd: f64,
    pub zen_grounded_latency_ms: f64,
    pub zen_direct_latency_ms: f64,
}

impl Default for RetrievalCostModel {
    fn default() -> Self {
        // Representative per-query telemetry: vector+BM25 index lookup + extra context tokens
        // vs direct Zen generation without retrieved chunk overhead.
        Self {
            retrieval_cost_usd: 0.00045,
            retrieval_latency_ms: 28.0,
            zen_grounded_cost_usd: 0.00110,
            zen_direct_cost_usd: 0.00045,
            zen_grounded_latency_ms: 95.0,
            zen_direct_latency_ms: 48.0,
        }
    }
}

/// Single query evaluation point for Self-RAG gate simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateSimulationSample {
    pub query_id: String,
    pub kai_probs: KaiRetrieveProbs,
    pub accurate_if_direct: bool,
    pub accurate_if_retrieved: bool,
}

/// End-to-end simulation metrics measuring avoided retrievals, factual
/// accuracy, latency, and dollar cost savings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateSimulationReport {
    pub theta: f64,
    pub total_queries: usize,
    pub retrievals_executed: usize,
    pub retrievals_avoided: usize,
    pub avoided_retrieval_rate: f64,
    pub factual_accuracy: f64,
    pub always_retrieve_accuracy: f64,
    pub never_retrieve_accuracy: f64,
    pub accuracy_delta_vs_always_retrieve: f64,
    pub total_cost_usd: f64,
    pub always_retrieve_cost_usd: f64,
    pub cost_savings_usd: f64,
    pub cost_savings_pct: f64,
    pub mean_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub always_retrieve_mean_latency_ms: f64,
    /// Expected Calibration Error of Kai's `p_retrieve` against true retrieval need (`!accurate_if_direct`).
    pub kai_ece: f64,
}

/// Runs deterministic Self-RAG simulation over `samples` at gate threshold `gate.theta`.
pub fn simulate_gate(
    gate: &RetrieveOrAnswerGate,
    samples: &[GateSimulationSample],
    costs: &RetrievalCostModel,
) -> Result<GateSimulationReport, GateError> {
    if samples.is_empty() {
        return Err(GateError::EmptySimulation);
    }

    let n = samples.len();
    let mut retrievals_executed = 0usize;
    let mut gated_correct = 0usize;
    let mut always_correct = 0usize;
    let mut never_correct = 0usize;
    let mut total_cost_usd = 0.0f64;
    let mut latencies = Vec::with_capacity(n);
    let mut cal_samples = Vec::with_capacity(n);

    let always_query_cost = costs.retrieval_cost_usd + costs.zen_grounded_cost_usd;
    let direct_query_cost = costs.zen_direct_cost_usd;
    let always_query_latency = costs.retrieval_latency_ms + costs.zen_grounded_latency_ms;
    let direct_query_latency = costs.zen_direct_latency_ms;

    for sample in samples {
        let decision = gate.evaluate(sample.kai_probs);
        if sample.accurate_if_retrieved {
            always_correct += 1;
        }
        if sample.accurate_if_direct {
            never_correct += 1;
        }

        if decision.action == RetrievalAction::Retrieve {
            retrievals_executed += 1;
            if sample.accurate_if_retrieved {
                gated_correct += 1;
            }
            total_cost_usd += always_query_cost;
            latencies.push(always_query_latency);
        } else {
            if sample.accurate_if_direct {
                gated_correct += 1;
            }
            total_cost_usd += direct_query_cost;
            latencies.push(direct_query_latency);
        }

        // Kai's `p_retrieve` models probability that direct parametric answer fails.
        cal_samples.push(CalibrationSample {
            confidence: sample.kai_probs.p_retrieve,
            is_correct: !sample.accurate_if_direct,
        });
    }

    latencies.sort_by(|a, b| a.total_cmp(b));
    let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    let p95_latency_ms = latencies[p95_idx];
    let mean_latency_ms = latencies.iter().sum::<f64>() / n as f64;

    let retrievals_avoided = n - retrievals_executed;
    let avoided_retrieval_rate = retrievals_avoided as f64 / n as f64;
    let factual_accuracy = gated_correct as f64 / n as f64;
    let always_retrieve_accuracy = always_correct as f64 / n as f64;
    let never_retrieve_accuracy = never_correct as f64 / n as f64;
    let accuracy_delta_vs_always_retrieve = factual_accuracy - always_retrieve_accuracy;

    let always_retrieve_cost_usd = always_query_cost * n as f64;
    let cost_savings_usd = always_retrieve_cost_usd - total_cost_usd;
    let cost_savings_pct = if always_retrieve_cost_usd > 0.0 {
        (cost_savings_usd / always_retrieve_cost_usd) * 100.0
    } else {
        0.0
    };

    let kai_ece = compute_ece(&cal_samples, 10).map(|r| r.ece).unwrap_or(0.0);

    Ok(GateSimulationReport {
        theta: gate.theta,
        total_queries: n,
        retrievals_executed,
        retrievals_avoided,
        avoided_retrieval_rate,
        factual_accuracy,
        always_retrieve_accuracy,
        never_retrieve_accuracy,
        accuracy_delta_vs_always_retrieve,
        total_cost_usd,
        always_retrieve_cost_usd,
        cost_savings_usd,
        cost_savings_pct,
        mean_latency_ms,
        p95_latency_ms,
        always_retrieve_mean_latency_ms: always_query_latency,
        kai_ece,
    })
}

/// Evaluates `samples` across a slice of candidate `theta` values.
pub fn sweep_thresholds(
    samples: &[GateSimulationSample],
    thetas: &[f64],
    costs: &RetrievalCostModel,
) -> Result<Vec<GateSimulationReport>, GateError> {
    let mut reports = Vec::with_capacity(thetas.len());
    for &theta in thetas {
        let gate = RetrieveOrAnswerGate::new(theta)?;
        reports.push(simulate_gate(&gate, samples, costs)?);
    }
    Ok(reports)
}

/// Selects the highest-savings `theta` whose factual accuracy remains within
/// `max_accuracy_drop` of the always-retrieve baseline.
pub fn select_optimal_theta(
    samples: &[GateSimulationSample],
    thetas: &[f64],
    costs: &RetrievalCostModel,
    max_accuracy_drop: f64,
) -> Result<Option<GateSimulationReport>, GateError> {
    let reports = sweep_thresholds(samples, thetas, costs)?;
    let best = reports
        .into_iter()
        .filter(|r| r.accuracy_delta_vs_always_retrieve >= -max_accuracy_drop)
        .max_by(|a, b| {
            a.cost_savings_usd
                .total_cmp(&b.cost_savings_usd)
                .then_with(|| a.factual_accuracy.total_cmp(&b.factual_accuracy))
        });
    Ok(best)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct MockKaiHead;
    impl KaiDecisionHead for MockKaiHead {
        fn score_retrieve_or_answer(&self, query: &str) -> Result<KaiRetrieveProbs, GateError> {
            if query.contains("HIP-1346") || query.contains("internal") {
                KaiRetrieveProbs::from_logits(2.4, -1.2)
            } else {
                KaiRetrieveProbs::from_logits(-1.8, 2.1)
            }
        }
    }

    struct MockZenEngine;
    impl ZenSynthesizer for MockZenEngine {
        fn synthesize(
            &self,
            query: &str,
            retrieved_evidence: &[String],
            target: SynthesisTarget,
        ) -> String {
            match target {
                SynthesisTarget::ZenParametric => {
                    assert!(retrieved_evidence.is_empty());
                    format!("ZenParametric({query})")
                }
                SynthesisTarget::ZenGrounded => {
                    format!("ZenGrounded({query}; docs={})", retrieved_evidence.len())
                }
            }
        }
    }

    #[test]
    fn preserves_kai_decision_vs_zen_synthesis_boundary() {
        let gate = RetrieveOrAnswerGate::new(0.55).unwrap();
        let kai = MockKaiHead;
        let zen = MockZenEngine;
        let retrieval_calls = AtomicUsize::new(0);

        // Parametric general-knowledge query skips retrieval while Zen synthesizes directly.
        let direct = route_with_kai_and_zen(&gate, &kai, &zen, "What is 2 + 2?", |_| {
            retrieval_calls.fetch_add(1, Ordering::SeqCst);
            vec!["unused".to_string()]
        })
        .unwrap();
        assert_eq!(direct.decision.action, RetrievalAction::AnswerDirectly);
        assert_eq!(direct.decision.decision_head, DecisionHead::Kai);
        assert_eq!(direct.decision.synthesis_target, SynthesisTarget::ZenParametric);
        assert!(direct.decision.skipped_retrieval);
        assert_eq!(retrieval_calls.load(Ordering::SeqCst), 0);
        assert!(direct.zen_answer.starts_with("ZenParametric("));

        // Domain-specific query triggers index retrieval and Zen grounded synthesis.
        let grounded = route_with_kai_and_zen(
            &gate,
            &kai,
            &zen,
            "Explain HIP-1346 §1 dimensions",
            |_| {
                retrieval_calls.fetch_add(1, Ordering::SeqCst);
                vec!["HIP-1346 specifies 128/256/512/768 Matryoshka dims".to_string()]
            },
        )
        .unwrap();
        assert_eq!(grounded.decision.action, RetrievalAction::Retrieve);
        assert_eq!(grounded.decision.synthesis_target, SynthesisTarget::ZenGrounded);
        assert!(!grounded.decision.skipped_retrieval);
        assert_eq!(retrieval_calls.load(Ordering::SeqCst), 1);
        assert!(grounded.zen_answer.starts_with("ZenGrounded("));
    }

    #[test]
    fn simulation_measures_avoided_retrievals_factual_accuracy_and_cost_savings() {
        // 20-query calibrated benchmark:
        // - 11 general/parametric queries (accurate directly; retrieval adds noise on 2 cases)
        // - 9 private/time-sensitive queries (require retrieval for factual accuracy)
        let mut samples = Vec::new();
        for i in 0..11 {
            let p_ret = 0.08 + (i as f64) * 0.025; // [0.08 .. 0.33]
            samples.push(GateSimulationSample {
                query_id: format!("parametric_{i}"),
                kai_probs: KaiRetrieveProbs::new(p_ret, 1.0 - p_ret).unwrap(),
                accurate_if_direct: true,
                accurate_if_retrieved: i >= 2, // 2 cases hurt by retrieval distractors
            });
        }
        for i in 0..9 {
            let p_ret = 0.68 + (i as f64) * 0.03; // [0.68 .. 0.92]
            samples.push(GateSimulationSample {
                query_id: format!("grounded_{i}"),
                kai_probs: KaiRetrieveProbs::new(p_ret, 1.0 - p_ret).unwrap(),
                accurate_if_direct: false,
                accurate_if_retrieved: true,
            });
        }

        let costs = RetrievalCostModel::default();
        let always_retrieve =
            simulate_gate(&RetrieveOrAnswerGate::new(0.0).unwrap(), &samples, &costs).unwrap();
        let never_retrieve =
            simulate_gate(&RetrieveOrAnswerGate::new(1.0).unwrap(), &samples, &costs).unwrap();

        let optimal = select_optimal_theta(
            &samples,
            &[0.10, 0.25, 0.40, 0.50, 0.65, 0.80],
            &costs,
            0.0,
        )
        .unwrap()
        .expect("optimal theta found");

        assert_eq!(always_retrieve.retrievals_avoided, 0);
        assert_eq!(never_retrieve.retrievals_executed, 0);
        assert!(never_retrieve.factual_accuracy < 0.60);

        // Calibrated gate avoids all 11 unnecessary retrievals (55% savings),
        // strictly beats always-retrieve accuracy (100% vs 90%), and cuts cost & latency.
        assert_eq!(optimal.retrievals_avoided, 11);
        assert!((optimal.avoided_retrieval_rate - 0.55).abs() < 1e-9);
        assert!((optimal.factual_accuracy - 1.0).abs() < 1e-9);
        assert!(optimal.factual_accuracy > always_retrieve.factual_accuracy);
        assert!(optimal.cost_savings_usd > 0.0);
        assert!(optimal.cost_savings_pct > 35.0);
        assert!(optimal.mean_latency_ms < always_retrieve.mean_latency_ms);
        assert!(optimal.kai_ece < 0.25);
    }
}
