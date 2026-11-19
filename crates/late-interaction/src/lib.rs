pub mod maxsim;
pub mod quantization;
pub mod index;

pub use maxsim::maxsim_score;
pub use quantization::CentroidQuantizer;
pub use index::CompressedMultiVectorIndex;

/// Represents a multi-vector representation of a document (one vector per token).
pub struct MultiVectorDoc {
    pub id: u64,
    pub embeddings: Vec<Vec<f32>>,
}
