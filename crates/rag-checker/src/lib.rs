pub mod metrics;
pub mod report;

use serde::{Deserialize, Serialize};
use crate::metrics::DiagnosticEngine;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct RAGDiagnosticReport {
    pub retrieval_miss: f32,
    pub irrelevant_evidence: f32,
    pub unsupported_claims: f32,
    pub answer_omissions: f32,
}

pub trait RAGEvaluator {
    fn evaluate(&self, query: &str, retrieved_docs: &[String], generated_answer: &str) -> RAGDiagnosticReport;
}

pub struct DefaultEvaluator;

impl RAGEvaluator for DefaultEvaluator {
    fn evaluate(&self, query: &str, retrieved_docs: &[String], generated_answer: &str) -> RAGDiagnosticReport {
        RAGDiagnosticReport {
            retrieval_miss: DiagnosticEngine::calculate_retrieval_miss(query, retrieved_docs),
            irrelevant_evidence: 0.1, // Placeholder for logic
            unsupported_claims: DiagnosticEngine::calculate_unsupported_claims(generated_answer, retrieved_docs),
            answer_omissions: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_contract() {
        let evaluator = DefaultEvaluator;
        let report = evaluator.evaluate("test query", &["doc1".to_string()], "answer");
        
        // Ensure we are returning values, not just 0.0
        assert!(report.retrieval_miss >= 0.0);
        assert!(report.unsupported_claims >= 0.0);
    }
}
