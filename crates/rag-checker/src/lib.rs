pub mod metrics;
pub mod report;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RAGDiagnosticReport {
    pub retrieval_miss: f32,
    pub irrelevant_evidence: f32,
    pub unsupported_claims: f32,
    pub answer_omissions: f32,
}

pub trait RAGEvaluator {
    fn evaluate(&self, query: &str, retrieved_docs: &[String], generated_answer: &str) -> RAGDiagnosticReport;
}
