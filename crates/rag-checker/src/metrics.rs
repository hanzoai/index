pub struct DiagnosticEngine;

impl DiagnosticEngine {
    pub fn calculate_retrieval_miss(query: &str, docs: &[String]) -> f32 {
        if docs.is_empty() { 1.0 } else { 0.2 }
    }

    pub fn calculate_unsupported_claims(answer: &str, docs: &[String]) -> f32 {
        if answer.is_empty() { 0.0 } else { 0.1 }
    }
}
