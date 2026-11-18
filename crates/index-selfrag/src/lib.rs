// SPDX-License-Identifier: Apache-2.0
//! [Bounty #22] (`hanzoai/index#26`): **Adaptive retrieve-or-answer policy** (Self-RAG).
//!
//! Clean-room simulation harness for deciding, per query, whether to retrieve
//! external evidence or answer directly from parametric memory.
//!
//! # Design
//!
//! Queries differ in how much they depend on fresh or long-tail evidence. The
//! gate scores each query from cheap features (entity novelty, question type,
//! staleness hint) and retrieves only when the score crosses a threshold.
//!
//! - `AlwaysRetrieve`: retrieve for every query (accuracy ceiling, cost floor).
//! - `NeverRetrieve`: answer parametrically (cheap, degrades on long-tail).
//! - `SelfRagGate`: retrieve only when the gate says the query needs evidence.
//!
//! Metrics: answer accuracy, retrieval rate, cost per query, cost per correct
//! answer, and the accuracy lost against always-retrieve.
//!
//! # Honest boundaries
//! - No LLM and no retriever are executed. Answer correctness is a modelled
//!   probability driven by query features.
//! - Gate features are synthetic; the gate's separation quality is hand-set.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced with a real retriever and model.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SelfRagError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Query features that a cheap gate could actually observe.
#[derive(Debug, Clone)]
pub struct Query {
    pub id: u32,
    /// 0 = well-known entity, 1 = rare/long-tail entity.
    pub entity_novelty: f64,
    /// 0 = stable fact, 1 = time-sensitive fact.
    pub staleness: f64,
    /// 0 = simple lookup, 1 = multi-hop reasoning.
    pub multi_hop: f64,
    /// Ground truth: does this query actually need retrieved evidence?
    pub needs_evidence: bool,
}

pub fn generate_queries(seed: u64, n: usize, need_ratio: f64) -> Vec<Query> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|i| {
            let needs = rng.next_f64() < need_ratio;
            // Queries that need evidence skew towards novel / stale / multi-hop.
            let bias = if needs { 0.65 } else { 0.3 };
            Query {
                id: i as u32,
                entity_novelty: (rng.next_f64() * 0.6 + bias * 0.4).clamp(0.0, 1.0),
                staleness: (rng.next_f64() * 0.6 + bias * 0.4).clamp(0.0, 1.0),
                multi_hop: (rng.next_f64() * 0.6 + bias * 0.3).clamp(0.0, 1.0),
                needs_evidence: needs,
            }
        })
        .collect()
}

/// Cheap gate score in [0,1]. Higher = more likely to need retrieval.
pub fn gate_score(q: &Query) -> f64 {
    (0.45 * q.entity_novelty + 0.35 * q.staleness + 0.20 * q.multi_hop).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    AlwaysRetrieve,
    NeverRetrieve,
    SelfRagGate { threshold: f64 },
}

impl Policy {
    pub fn label(&self) -> String {
        match self {
            Policy::AlwaysRetrieve => "always-retrieve".into(),
            Policy::NeverRetrieve => "never-retrieve".into(),
            Policy::SelfRagGate { threshold } => format!("self-rag-gate(threshold={:.2})", threshold),
        }
    }

    pub fn retrieves(&self, q: &Query) -> bool {
        match self {
            Policy::AlwaysRetrieve => true,
            Policy::NeverRetrieve => false,
            Policy::SelfRagGate { threshold } => gate_score(q) > *threshold,
        }
    }

    /// Cost per query: retrieval is expensive, parametric answer is cheap.
    pub fn cost(&self, retrieved: bool) -> f64 {
        if retrieved {
            1.0
        } else {
            0.15
        }
    }
}

/// Probability the answer is correct.
pub fn correctness(q: &Query, retrieved: bool) -> f64 {
    if retrieved {
        // Retrieval helps when evidence is needed, and is roughly neutral otherwise
        // (occasional distractor noise costs a little).
        if q.needs_evidence {
            0.92
        } else {
            0.93
        }
    } else if q.needs_evidence {
        // Parametric memory fails on long-tail / time-sensitive facts.
        0.42
    } else {
        0.90
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub queries: usize,
    pub retrieval_rate: f64,
    pub accuracy: f64,
    pub cost_per_query: f64,
    pub cost_per_correct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub n_queries: usize,
    /// Fraction of queries that genuinely require retrieved evidence.
    pub need_ratio: f64,
    /// Gate threshold swept for the adaptive policy.
    pub gate_threshold: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0x5E1F_2A6, n_queries: 5_000, need_ratio: 0.4, gate_threshold: 0.45 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, n_queries: 600, need_ratio: 0.4, gate_threshold: 0.45 }
    }
}

pub fn evaluate(policy: Policy, queries: &[Query]) -> PolicyResult {
    let mut retrieved = 0usize;
    let mut correct = 0usize;
    let mut cost = 0.0f64;

    for q in queries {
        let r = policy.retrieves(q);
        if r {
            retrieved += 1;
        }
        cost += policy.cost(r);
        // Deterministic verdict from query shape.
        let draw = ((q.id as u64 * 2_654_435_761 + (q.entity_novelty * 1000.0) as u64) % 1000) as f64 / 1000.0;
        if draw < correctness(q, r) {
            correct += 1;
        }
    }

    let n = queries.len().max(1) as f64;
    PolicyResult {
        policy: policy.label(),
        queries: queries.len(),
        retrieval_rate: retrieved as f64 / n,
        accuracy: correct as f64 / n,
        cost_per_query: cost / n,
        cost_per_correct: if correct > 0 { cost / correct as f64 } else { f64::INFINITY },
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, SelfRagError> {
    if cfg.n_queries == 0 {
        return Err(SelfRagError::InvalidConfig("n_queries must be positive".into()));
    }
    if !(0.0..=1.0).contains(&cfg.need_ratio) {
        return Err(SelfRagError::InvalidConfig("need_ratio must be in [0,1]".into()));
    }
    let queries = generate_queries(cfg.seed, cfg.n_queries, cfg.need_ratio);
    let results = vec![
        evaluate(Policy::AlwaysRetrieve, &queries),
        evaluate(Policy::NeverRetrieve, &queries),
        evaluate(Policy::SelfRagGate { threshold: cfg.gate_threshold }, &queries),
    ];

    let mut notes = Vec::new();
    if let (Some(always), Some(gate)) = (
        results.iter().find(|r| r.policy == "always-retrieve"),
        results.iter().find(|r| r.policy.starts_with("self-rag-gate")),
    ) {
        notes.push(format!(
            "gate keeps {:.1}% of always-retrieve accuracy ({:.3} vs {:.3}) at {:.0}% of the retrieval rate",
            gate.accuracy / always.accuracy.max(1e-9) * 100.0,
            gate.accuracy,
            always.accuracy,
            gate.retrieval_rate / always.retrieval_rate.max(1e-9) * 100.0
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/index#26 [Bounty #22] Adaptive retrieve-or-answer policy (Self-RAG)".into(),
        hip: "HIP-1346: Evidence-Grounded Retrieval and Decision Evaluation".into(),
        honesty_labels: vec![
            "no LLM and no retriever executed — correctness is a modelled probability".into(),
            "gate features are synthetic; separation quality is hand-set".into(),
            "cost model (1.0 retrieve / 0.15 answer) is relative, not measured".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced with a real retriever and model.".into(),
        ],
        config: cfg.clone(),
        results,
        notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_uses_fewer_retrievals_than_always() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let always = report.results.iter().find(|r| r.policy == "always-retrieve").unwrap();
        let gate = report.results.iter().find(|r| r.policy.starts_with("self-rag-gate")).unwrap();
        assert_eq!(always.retrieval_rate, 1.0);
        assert!(gate.retrieval_rate < 1.0, "gate retrieved everything");
    }

    #[test]
    fn gate_beats_never_retrieve_on_accuracy() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let never = report.results.iter().find(|r| r.policy == "never-retrieve").unwrap();
        let gate = report.results.iter().find(|r| r.policy.starts_with("self-rag-gate")).unwrap();
        assert!(
            gate.accuracy > never.accuracy,
            "gate {:.3} should beat never-retrieve {:.3}",
            gate.accuracy,
            never.accuracy
        );
    }

    #[test]
    fn gate_is_cheaper_than_always() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let always = report.results.iter().find(|r| r.policy == "always-retrieve").unwrap();
        let gate = report.results.iter().find(|r| r.policy.starts_with("self-rag-gate")).unwrap();
        assert!(gate.cost_per_query < always.cost_per_query);
    }

    #[test]
    fn lower_threshold_retrieves_more() {
        let qs = generate_queries(11, 500, 0.4);
        let low = evaluate(Policy::SelfRagGate { threshold: 0.2 }, &qs);
        let high = evaluate(Policy::SelfRagGate { threshold: 0.8 }, &qs);
        assert!(low.retrieval_rate > high.retrieval_rate);
        assert!(low.accuracy >= high.accuracy);
    }

    #[test]
    fn bad_config_rejected() {
        assert!(run_bench(&BenchConfig { seed: 1, n_queries: 0, need_ratio: 0.4, gate_threshold: 0.5 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, n_queries: 10, need_ratio: 2.0, gate_threshold: 0.5 }).is_err());
    }
}
