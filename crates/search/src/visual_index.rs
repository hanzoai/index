use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// HIP-1346 §1: Separate metadata lock for page-image index.
/// This ensures visual retrieval lifecycle is decoupled from text chunking.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VisualPageMetadata {
    pub page_number: u32,
    pub document_id: String,
    pub s3_image_path: String,
    pub multi_vector_count: usize,
    pub dimensions: usize,
}

pub struct VisualRetrievalPath {
    pub metadata_store: Arc<VisualPageMetadata>,
}

impl VisualRetrievalPath {
    pub fn new(metadata: VisualPageMetadata) -> Self {
        Self {
            metadata_store: Arc::new(metadata),
        }
    }

    /// Validates the stack boundary: vfs -> hanzoai/s3 -> visor
    pub fn verify_storage_boundary(&self) -> bool {
        self.metadata_store.s3_image_path.starts_with("s3://")
    }
}
