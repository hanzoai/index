pub mod verifier;
pub mod decomposition;

use async_trait::async_trait;

pub struct FactualityReport {
    pub atomic_claims: Vec<String>,
    pub supported_claims: Vec<bool>,
    pub precision_score: f32,
}

#[async_trait]
pub trait Verifier {
    async fn verify(&self, claim: &str, evidence: &str) -> anyhow::Result<bool>;
}
