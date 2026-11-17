pub mod verifier;
pub mod decomposition;

pub struct FactualityReport {
    pub atomic_claims: Vec<String>,
    pub supported_claims: Vec<bool>,
    pub precision_score: f32,
}

pub trait Verifier {
    async fn verify(&self, claim: &str, evidence: &str) -> bool;
}
