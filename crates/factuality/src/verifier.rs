use crate::{FactualityReport, Verifier};
use async_trait::async_trait;

pub struct KaiVerifier;

#[async_trait]
impl Verifier for KaiVerifier {
    async fn verify(&self, claim: &str, evidence: &str) -> anyhow::Result<bool> {
        // Logic: Check if the claim exists within the evidence string
        // This replaces the placeholder 'true' with a functional check
        if claim.trim().is_empty() {
            return Ok(true);
        }
        Ok(evidence.contains(claim))
    }
}

pub async fn run_factuality_check(assertion: &str, evidence: &str) -> anyhow::Result<FactualityReport> {
    let claims = crate::decomposition::decompose_assertion(assertion);
    let verifier = KaiVerifier;
    
    let mut supported = Vec::new();
    for claim in &claims {
        let is_supported = verifier.verify(claim, evidence).await?;
        supported.push(is_supported);
    }
    
    let precision = if claims.is_empty() {
        0.0
    } else {
        supported.iter().filter(|&&x| x).count() as f32 / claims.len() as f32
    };
    
    Ok(FactualityReport {
        atomic_claims: claims,
        supported_claims: supported,
        precision_score: precision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verifier_logic() {
        let verifier = KaiVerifier;
        let evidence = "The sky is blue.";
        let result = verifier.verify("The sky is blue", evidence).await.unwrap();
        assert!(result);
        
        let result_false = verifier.verify("The sky is green", evidence).await.unwrap();
        assert!(!result_false);
    }
}
