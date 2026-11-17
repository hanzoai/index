use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RetrievalConfidence {
    Correct,
    Ambiguous,
    Incorrect,
}

pub struct CorrectiveEvaluator {
    pub confidence_threshold: f32,
}

impl CorrectiveEvaluator {
    pub fn new(threshold: f32) -> Self {
        Self { confidence_threshold: threshold }
    }

    /// Evaluates retrieved documents against the query.
    /// In a production scenario, this integrates with Kai's risk scoring.
    pub fn evaluate(&self, score: f32) -> RetrievalConfidence {
        if score >= self.confidence_threshold {
            RetrievalConfidence::Correct
        } else if score >= (self.confidence_threshold * 0.5) {
            RetrievalConfidence::Ambiguous
        } else {
            RetrievalConfidence::Incorrect
        }
    }

    /// Triggers secondary retrieval if the confidence is not 'Correct'
    pub fn should_trigger_expansion(&self, confidence: &RetrievalConfidence) -> bool {
        match confidence {
            RetrievalConfidence::Correct => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crag_logic() {
        let evaluator = CorrectiveEvaluator::new(0.8);
        
        // Test high confidence
        assert!(matches!(evaluator.evaluate(0.9), RetrievalConfidence::Correct));
        
        // Test low confidence (corrupted set)
        let conf = evaluator.evaluate(0.2);
        assert!(evaluator.should_trigger_expansion(&conf));
    }
}
