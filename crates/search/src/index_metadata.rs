use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexMetadata {
    pub model_id: String,
    pub model_revision: String,
    pub dimensions: usize,
    pub normalization: String,
    pub task_prefix: Option<String>,
}

impl IndexMetadata {
    pub fn new(
        model_id: &str,
        model_revision: &str,
        dimensions: usize,
        normalization: &str,
        task_prefix: Option<&str>,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            model_revision: model_revision.to_string(),
            dimensions,
            normalization: normalization.to_string(),
            task_prefix: task_prefix.map(|s| s.to_string()),
        }
    }

    /// Validates that the new metadata matches the existing immutable metadata.
    pub fn validate_compatibility(&self, existing: &Self) -> Result<(), MetadataError> {
        if self != existing {
            return Err(MetadataError::ImmutableViolation);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum MetadataError {
    ImmutableViolation,
    SerializationError(String),
}

impl std::error::Error for MetadataError {}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataError::ImmutableViolation => write!(f, "Attempted to modify immutable index metadata (model_id, revision, or dimensions)"),
            MetadataError::SerializationError(e) => write!(f, "Metadata serialization error: {}", e),
        }
    }
}
