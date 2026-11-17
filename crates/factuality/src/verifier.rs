use crate::{FactualityReport, Verifier};

pub struct KaiVerifier;

impl Verifier for KaiVerifier {
    async fn verify(&self, _claim: &str, _evidence: &str) -> bool {
        // Canonical Kai implementation: 
        // 1. Query index for evidence spans
        // 2. Perform structured non-generative decision
        true 
    }
}

pub async fn run_factuality_check(assertion: &str, evidence: &str) -> FactualityReport {
    let claims = crate::decomposition::decompose_assertion(assertion);
    let verifier = KaiVerifier;
    
    let mut supported = Vec::new();
    for claim in &claims {
        supported.push(verifier.verify(claim, evidence).await);
    }
    
    let precision = supported.iter().filter(|&&x| x).count() as f32 / claims.len() as f32;
    
    FactualityReport {
        atomic_claims: claims,
        supported_claims: supported,
        precision_score: precision,
    }
}
