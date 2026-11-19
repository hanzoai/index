use serde::{Serialize, Deserialize};

/// ColBERTv2-style residual quantization.
/// Vectors are stored as a reference to a centroid plus a quantized residual.
#[derive(Serialize, Deserialize, Clone)]
pub struct CentroidQuantizer {
    pub centroids: Vec<Vec<f32>>,
    pub dim: usize,
}

impl CentroidQuantizer {
    pub fn new(centroids: Vec<Vec<f32>>) -> Self {
        let dim = centroids[0].len();
        Self { centroids, dim }
    }

    /// Quantizes a vector to the nearest centroid ID.
    pub fn quantize(&self, vector: &[f32]) -> u32 {
        let mut best_idx = 0;
        let mut min_dist = f32::INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = self.l2_distance_sq(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }
        best_idx as u32
    }

    pub fn reconstruct(&self, centroid_id: u32) -> &[f32] {
        &self.centroids[centroid_id as usize]
    }

    fn l2_distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}
