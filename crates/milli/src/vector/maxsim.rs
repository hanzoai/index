use std::iter::Sum;

/// Implements the MaxSim (Maximum Similarity) operator used by ColPali.
/// For a query Q with vectors {q_i} and a document D with vectors {d_j},
/// MaxSim(Q, D) = sum_i(max_j(q_i · d_j))
pub struct MaxSim;

impl MaxSim {
    /// Calculates the MaxSim score between query multi-vectors and document multi-vectors.
    /// Assumes vectors are normalized for cosine similarity via dot product.
    pub fn score(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f32 {
        if query_embeddings.is_empty() || doc_embeddings.is_empty() {
            return 0.0;
        }

        query_embeddings
            .iter()
            .map(|q| {
                doc_embeddings
                    .iter()
                    .map(|d| Self::dot_product(q, d))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    #[inline]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_logic() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        
        // MaxSim = max(q1.d1, q1.d2) + max(q2.d1, q2.d2)
        // q1.d1 = 1.0, q1.d2 = 0.5 -> max = 1.0
        // q2.d1 = 0.0, q2.d2 = 0.5 -> max = 0.5
        // Result = 1.5
        let result = MaxSim::score(&query, &doc);
        assert_eq!(result, 1.5);
    }
}
