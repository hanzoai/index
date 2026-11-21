// SPDX-License-Identifier: Apache-2.0

//! HIP-1346 §1: Immutable Vector Index Metadata Contract.
//!
//! Enforces strict metadata parity (`model_id`, `model_revision`, `dimensions`,
//! `normalization`, `query_prefix`, `doc_prefix`) upon collection creation,
//! document insertion, and query execution, rejecting silent vector-space,
//! Matryoshka-width (`128`/`256`/`512`/`768`), normalization, or task-prefix
//! mismatches.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Canonical Matryoshka Representation Learning (MRL) embedding widths
/// supported by HIP-1346 §1.
pub const MATRYOSHKA_DIMS: [usize; 4] = [128, 256, 512, 768];

/// Default tolerance for verifying L2-normalized vectors.
pub const L2_NORM_TOLERANCE: f32 = 1e-3;

/// Mutable branch/alias tags disallowed as pinned `model_revision` values.
const UNPINNED_REVISIONS: &[&str] = &["head", "latest", "main", "master"];

/// Embedding vector normalization policy stored in index metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NormalizationPolicy {
    /// Unit L2 Euclidean norm (`||v||_2 = 1.0`).
    L2,
    /// Unnormalized raw embeddings.
    None,
    /// Named custom normalization policy (must match identically at runtime).
    Custom(String),
}

impl NormalizationPolicy {
    /// Returns the canonical lowercase wire representation.
    pub fn as_str(&self) -> &str {
        match self {
            Self::L2 => "l2",
            Self::None => "none",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Parses a normalization policy string (`"l2"`, `"none"`, or custom).
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("l2") {
            Self::L2
        } else if trimmed.eq_ignore_ascii_case("none") {
            Self::None
        } else {
            Self::Custom(trimmed.to_string())
        }
    }
}

impl fmt::Display for NormalizationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for NormalizationPolicy {
    fn from(value: &str) -> Self {
        Self::parse(value)
    }
}

impl Serialize for NormalizationPolicy {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for NormalizationPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}

/// Operation kind audited against immutable index metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexOperation {
    Create,
    Insert,
    Query,
}

impl fmt::Display for IndexOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Create => f.write_str("create"),
            Self::Insert => f.write_str("insert"),
            Self::Query => f.write_str("query"),
        }
    }
}

/// Fields protected by the HIP-1346 §1 immutable metadata contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataField {
    ModelId,
    ModelRevision,
    Dimensions,
    Normalization,
    QueryPrefix,
    DocPrefix,
}

impl fmt::Display for MetadataField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelId => f.write_str("model_id"),
            Self::ModelRevision => f.write_str("model_revision"),
            Self::Dimensions => f.write_str("dimensions"),
            Self::Normalization => f.write_str("normalization"),
            Self::QueryPrefix => f.write_str("query_prefix"),
            Self::DocPrefix => f.write_str("doc_prefix"),
        }
    }
}

/// Single field discrepancy between persisted index metadata and a runtime request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataMismatch {
    pub field: MetadataField,
    pub expected: String,
    pub actual: String,
}

impl fmt::Display for MetadataMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} mismatch (expected `{}` != actual `{}`)",
            self.field, self.expected, self.actual
        )
    }
}

/// Errors raised when index metadata is invalid, mutated, or mismatched on insert/query.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MetadataError {
    #[error("`model_id` must not be empty")]
    EmptyModelId,
    #[error("`model_revision` must be pinned to an immutable digest/version, got `{0}`")]
    UnpinnedModelRevision(String),
    #[error("`dimensions` must be a positive integer")]
    ZeroDimensions,
    #[error("unsupported Matryoshka dimension `{0}`; expected one of {MATRYOSHKA_DIMS:?}")]
    UnsupportedMatryoshkaDimension(usize),
    #[error("cannot truncate embedding of length {actual} to target dimension {target}")]
    TruncationSourceTooShort { actual: usize, target: usize },
    #[error("{operation} rejected due to immutable metadata mismatch: {detail}")]
    ContractMismatch {
        operation: IndexOperation,
        mismatches: Vec<MetadataMismatch>,
        detail: String,
    },
    #[error("{operation} vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { operation: IndexOperation, expected: usize, actual: usize },
    #[error("{operation} vector contains non-finite component at index {index}")]
    NonFiniteVectorValue { operation: IndexOperation, index: usize },
    #[error("{operation} vector violates `{normalization}` normalization (norm={norm:.6})")]
    NormalizationViolation {
        operation: IndexOperation,
        normalization: NormalizationPolicy,
        norm: f32,
    },
    #[error("{operation} text missing required `{field}` prefix `{expected}`")]
    MissingRequiredPrefix { operation: IndexOperation, field: MetadataField, expected: String },
    #[error(
        "{operation} text used wrong task prefix (expected `{expected_field}`=`{expected_prefix}`, detected `{actual_field}`=`{actual_prefix}`)"
    )]
    SwappedTaskPrefix {
        operation: IndexOperation,
        expected_field: MetadataField,
        expected_prefix: String,
        actual_field: MetadataField,
        actual_prefix: String,
    },
    #[error("collection `{collection}` already exists with immutable metadata and cannot be mutated")]
    ImmutableCollectionExists { collection: String, mismatches: Vec<MetadataMismatch> },
    #[error("collection `{0}` is not registered in the metadata contract")]
    UnknownCollection(String),
}

/// Immutable embedding contract persisted per vector index or collection (HIP-1346 §1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexEmbeddingMetadata {
    /// Canonical embedding model identifier (e.g. `"hanzo/zen-embed-v1"`).
    pub model_id: String,
    /// Pinned immutable artifact revision / commit digest (e.g. `"sha256:9f84..."`).
    pub model_revision: String,
    /// Positive vector width (e.g. `128`, `256`, `512`, or `768` under Matryoshka truncation).
    pub dimensions: usize,
    /// Vector normalization policy (e.g. `NormalizationPolicy::L2`).
    pub normalization: NormalizationPolicy,
    /// Required asymmetric task prefix prepended to query inputs.
    pub query_prefix: String,
    /// Required asymmetric task prefix prepended to document inputs.
    pub doc_prefix: String,
}

impl IndexEmbeddingMetadata {
    /// Constructs and validates a new `IndexEmbeddingMetadata` contract.
    pub fn new(
        model_id: impl Into<String>,
        model_revision: impl Into<String>,
        dimensions: usize,
        normalization: NormalizationPolicy,
        query_prefix: impl Into<String>,
        doc_prefix: impl Into<String>,
    ) -> Result<Self, MetadataError> {
        let meta = Self {
            model_id: model_id.into(),
            model_revision: model_revision.into(),
            dimensions,
            normalization,
            query_prefix: query_prefix.into(),
            doc_prefix: doc_prefix.into(),
        };
        meta.validate_definition()?;
        Ok(meta)
    }

    /// Returns `true` if `dimensions` is one of the canonical Matryoshka widths
    /// (`128`, `256`, `512`, `768`).
    pub fn is_standard_matryoshka_dim(dimensions: usize) -> bool {
        MATRYOSHKA_DIMS.contains(&dimensions)
    }

    /// Validates that the metadata definition is well-formed and pinned.
    pub fn validate_definition(&self) -> Result<(), MetadataError> {
        if self.model_id.trim().is_empty() {
            return Err(MetadataError::EmptyModelId);
        }
        let rev = self.model_revision.trim();
        if rev.is_empty() || UNPINNED_REVISIONS.iter().any(|bad| rev.eq_ignore_ascii_case(bad)) {
            return Err(MetadataError::UnpinnedModelRevision(self.model_revision.clone()));
        }
        if self.dimensions == 0 {
            return Err(MetadataError::ZeroDimensions);
        }
        Ok(())
    }

    /// Validates definition plus standard Matryoshka width (`128`/`256`/`512`/`768`).
    pub fn validate_matryoshka_definition(&self) -> Result<(), MetadataError> {
        self.validate_definition()?;
        if !Self::is_standard_matryoshka_dim(self.dimensions) {
            return Err(MetadataError::UnsupportedMatryoshkaDimension(self.dimensions));
        }
        Ok(())
    }

    /// Collects all field mismatches between `self` (persisted) and `runtime`.
    pub fn diff(&self, runtime: &Self) -> Vec<MetadataMismatch> {
        let mut mismatches = Vec::new();
        if self.model_id != runtime.model_id {
            mismatches.push(MetadataMismatch {
                field: MetadataField::ModelId,
                expected: self.model_id.clone(),
                actual: runtime.model_id.clone(),
            });
        }
        if self.model_revision != runtime.model_revision {
            mismatches.push(MetadataMismatch {
                field: MetadataField::ModelRevision,
                expected: self.model_revision.clone(),
                actual: runtime.model_revision.clone(),
            });
        }
        if self.dimensions != runtime.dimensions {
            mismatches.push(MetadataMismatch {
                field: MetadataField::Dimensions,
                expected: self.dimensions.to_string(),
                actual: runtime.dimensions.to_string(),
            });
        }
        if self.normalization != runtime.normalization {
            mismatches.push(MetadataMismatch {
                field: MetadataField::Normalization,
                expected: self.normalization.to_string(),
                actual: runtime.normalization.to_string(),
            });
        }
        if self.query_prefix != runtime.query_prefix {
            mismatches.push(MetadataMismatch {
                field: MetadataField::QueryPrefix,
                expected: self.query_prefix.clone(),
                actual: runtime.query_prefix.clone(),
            });
        }
        if self.doc_prefix != runtime.doc_prefix {
            mismatches.push(MetadataMismatch {
                field: MetadataField::DocPrefix,
                expected: self.doc_prefix.clone(),
                actual: runtime.doc_prefix.clone(),
            });
        }
        mismatches
    }

    /// Rejects any disagreement between persisted metadata and runtime metadata on
    /// `Create`, `Insert`, or `Query`.
    pub fn validate_runtime(
        &self,
        runtime: &Self,
        operation: IndexOperation,
    ) -> Result<(), MetadataError> {
        self.validate_definition()?;
        runtime.validate_definition()?;
        let mismatches = self.diff(runtime);
        if mismatches.is_empty() {
            return Ok(());
        }
        let detail = mismatches.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ");
        Err(MetadataError::ContractMismatch { operation, mismatches, detail })
    }

    /// Validates an embedding vector's width, finiteness, and normalization policy.
    pub fn validate_vector(
        &self,
        vector: &[f32],
        operation: IndexOperation,
    ) -> Result<(), MetadataError> {
        self.validate_definition()?;
        if vector.len() != self.dimensions {
            return Err(MetadataError::VectorDimensionMismatch {
                operation,
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        for (index, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(MetadataError::NonFiniteVectorValue { operation, index });
            }
        }
        if self.normalization == NormalizationPolicy::L2 {
            let norm_sq: f64 = vector.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let norm = norm_sq.sqrt() as f32;
            if !norm.is_finite() || (norm - 1.0).abs() > L2_NORM_TOLERANCE {
                return Err(MetadataError::NormalizationViolation {
                    operation,
                    normalization: self.normalization.clone(),
                    norm,
                });
            }
        }
        Ok(())
    }

    /// Formats a raw query string using `self.query_prefix`.
    pub fn format_query(&self, raw_query: &str) -> String {
        format!("{}{}", self.query_prefix, raw_query)
    }

    /// Formats a raw document string using `self.doc_prefix`.
    pub fn format_document(&self, raw_document: &str) -> String {
        format!("{}{}", self.doc_prefix, raw_document)
    }

    /// Validates that an input text carries the exact asymmetric prefix required
    /// for `operation` (`doc_prefix` on `Insert`, `query_prefix` on `Query`).
    pub fn validate_prefixed_text(
        &self,
        prefixed_text: &str,
        operation: IndexOperation,
    ) -> Result<(), MetadataError> {
        let (expected_field, expected_prefix, other_field, other_prefix) = match operation {
            IndexOperation::Insert => (
                MetadataField::DocPrefix,
                self.doc_prefix.as_str(),
                MetadataField::QueryPrefix,
                self.query_prefix.as_str(),
            ),
            IndexOperation::Query => (
                MetadataField::QueryPrefix,
                self.query_prefix.as_str(),
                MetadataField::DocPrefix,
                self.doc_prefix.as_str(),
            ),
            IndexOperation::Create => return Ok(()),
        };

        if !expected_prefix.is_empty() && !prefixed_text.starts_with(expected_prefix) {
            if !other_prefix.is_empty()
                && other_prefix != expected_prefix
                && prefixed_text.starts_with(other_prefix)
            {
                return Err(MetadataError::SwappedTaskPrefix {
                    operation,
                    expected_field,
                    expected_prefix: expected_prefix.to_string(),
                    actual_field: other_field,
                    actual_prefix: other_prefix.to_string(),
                });
            }
            return Err(MetadataError::MissingRequiredPrefix {
                operation,
                field: expected_field,
                expected: expected_prefix.to_string(),
            });
        }
        Ok(())
    }

    /// Performs complete insert validation (runtime metadata, vector shape/norm, optional prefixed text).
    pub fn validate_insert(
        &self,
        runtime: &Self,
        vector: &[f32],
        prefixed_doc_text: Option<&str>,
    ) -> Result<(), MetadataError> {
        self.validate_runtime(runtime, IndexOperation::Insert)?;
        self.validate_vector(vector, IndexOperation::Insert)?;
        if let Some(text) = prefixed_doc_text {
            self.validate_prefixed_text(text, IndexOperation::Insert)?;
        }
        Ok(())
    }

    /// Performs complete query validation (runtime metadata, vector shape/norm, optional prefixed text).
    pub fn validate_query(
        &self,
        runtime: &Self,
        vector: &[f32],
        prefixed_query_text: Option<&str>,
    ) -> Result<(), MetadataError> {
        self.validate_runtime(runtime, IndexOperation::Query)?;
        self.validate_vector(vector, IndexOperation::Query)?;
        if let Some(text) = prefixed_query_text {
            self.validate_prefixed_text(text, IndexOperation::Query)?;
        }
        Ok(())
    }
}

/// Truncates a full-width embedding vector to a canonical Matryoshka prefix
/// (`128`, `256`, `512`, or `768`) and optionally L2-renormalizes.
pub fn truncate_matryoshka(
    vector: &[f32],
    target_dim: usize,
    normalization: &NormalizationPolicy,
) -> Result<Vec<f32>, MetadataError> {
    if !IndexEmbeddingMetadata::is_standard_matryoshka_dim(target_dim) {
        return Err(MetadataError::UnsupportedMatryoshkaDimension(target_dim));
    }
    if vector.len() < target_dim {
        return Err(MetadataError::TruncationSourceTooShort {
            actual: vector.len(),
            target: target_dim,
        });
    }
    let mut out = vector[..target_dim].to_vec();
    for (index, &val) in out.iter().enumerate() {
        if !val.is_finite() {
            return Err(MetadataError::NonFiniteVectorValue {
                operation: IndexOperation::Create,
                index,
            });
        }
    }
    if *normalization == NormalizationPolicy::L2 {
        let norm_sq: f64 = out.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let norm = norm_sq.sqrt() as f32;
        if !norm.is_finite() || norm == 0.0 {
            return Err(MetadataError::NormalizationViolation {
                operation: IndexOperation::Create,
                normalization: normalization.clone(),
                norm,
            });
        }
        for v in &mut out {
            *v /= norm;
        }
    }
    Ok(out)
}

/// Persisted collection-level metadata contract enforcing immutability across
/// creation, document insertion, and query time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableIndexContract {
    collections: BTreeMap<String, IndexEmbeddingMetadata>,
}

impl ImmutableIndexContract {
    /// Creates an empty immutable collection metadata registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a collection with immutable embedding metadata. Re-registering
    /// the same collection with identical metadata is idempotent; any change fails.
    pub fn register(
        &mut self,
        collection: impl Into<String>,
        metadata: IndexEmbeddingMetadata,
    ) -> Result<(), MetadataError> {
        metadata.validate_definition()?;
        let name = collection.into();
        if let Some(existing) = self.collections.get(&name) {
            let mismatches = existing.diff(&metadata);
            if !mismatches.is_empty() {
                return Err(MetadataError::ImmutableCollectionExists {
                    collection: name,
                    mismatches,
                });
            }
            return Ok(());
        }
        self.collections.insert(name, metadata);
        Ok(())
    }

    /// Returns the persisted metadata for `collection`, if registered.
    pub fn get(&self, collection: &str) -> Option<&IndexEmbeddingMetadata> {
        self.collections.get(collection)
    }

    /// Validates an `Insert` against a registered collection's immutable metadata.
    pub fn validate_insert(
        &self,
        collection: &str,
        runtime: &IndexEmbeddingMetadata,
        vector: &[f32],
        prefixed_doc_text: Option<&str>,
    ) -> Result<(), MetadataError> {
        let persisted = self
            .collections
            .get(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;
        persisted.validate_insert(runtime, vector, prefixed_doc_text)
    }

    /// Validates a `Query` against a registered collection's immutable metadata.
    pub fn validate_query(
        &self,
        collection: &str,
        runtime: &IndexEmbeddingMetadata,
        vector: &[f32],
        prefixed_query_text: Option<&str>,
    ) -> Result<(), MetadataError> {
        let persisted = self
            .collections
            .get(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;
        persisted.validate_query(runtime, vector, prefixed_query_text)
    }

    /// Serializes the persisted metadata contract to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Restores a persisted metadata contract from JSON and validates every entry.
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contract: Self = serde_json::from_str(json)?;
        for meta in contract.collections.values() {
            meta.validate_definition()?;
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize) -> Vec<f32> {
        let val = 1.0 / (dim as f32).sqrt();
        vec![val; dim]
    }

    fn base_meta(dims: usize) -> IndexEmbeddingMetadata {
        IndexEmbeddingMetadata {
            model_id: "hanzo/zen-embed-v1".to_string(),
            model_revision: "sha256:4a7f2c90b1e3".to_string(),
            dimensions: dims,
            normalization: NormalizationPolicy::L2,
            query_prefix: "search_query: ".to_string(),
            doc_prefix: "search_document: ".to_string(),
        }
    }

    #[test]
    fn matryoshka_dim_mismatches_rejected_on_insert_and_query() {
        for &index_dim in &MATRYOSHKA_DIMS {
            let mut contract = ImmutableIndexContract::new();
            let index_meta = base_meta(index_dim);
            index_meta.validate_matryoshka_definition().expect("valid MRL dim");
            let collection = format!("docs_{index_dim}");
            contract.register(collection.clone(), index_meta.clone()).unwrap();

            // Valid matching insert & query succeed.
            let valid_vec = unit_vec(index_dim);
            let q_text = index_meta.format_query("what is hip-1346");
            let d_text = index_meta.format_document("hip-1346 defines metadata contracts");
            assert!(contract.validate_insert(&collection, &index_meta, &valid_vec, Some(&d_text)).is_ok());
            assert!(contract.validate_query(&collection, &index_meta, &valid_vec, Some(&q_text)).is_ok());

            // Every other Matryoshka dimension MUST fail explicitly on both insert and query.
            for &other_dim in &MATRYOSHKA_DIMS {
                if other_dim == index_dim {
                    continue;
                }
                let bad_runtime = base_meta(other_dim);
                let bad_vec = unit_vec(other_dim);

                let insert_err = contract
                    .validate_insert(&collection, &bad_runtime, &bad_vec, Some(&d_text))
                    .unwrap_err();
                assert!(matches!(
                    insert_err,
                    MetadataError::ContractMismatch { operation: IndexOperation::Insert, .. }
                ));

                let query_err = contract
                    .validate_query(&collection, &bad_runtime, &bad_vec, Some(&q_text))
                    .unwrap_err();
                assert!(matches!(
                    query_err,
                    MetadataError::ContractMismatch { operation: IndexOperation::Query, .. }
                ));

                // Even if caller passes matching metadata header, mismatched raw vector length fails.
                let vec_err = contract
                    .validate_query(&collection, &index_meta, &bad_vec, Some(&q_text))
                    .unwrap_err();
                assert!(matches!(
                    vec_err,
                    MetadataError::VectorDimensionMismatch {
                        operation: IndexOperation::Query,
                        expected,
                        actual,
                    } if expected == index_dim && actual == other_dim
                ));
            }
        }
    }

    #[test]
    fn prefix_normalization_model_and_revision_mismatches_rejected() {
        let mut contract = ImmutableIndexContract::new();
        let persisted = base_meta(768);
        contract.register("kb", persisted.clone()).unwrap();
        let v = unit_vec(768);

        // Model ID mismatch.
        let mut wrong_model = persisted.clone();
        wrong_model.model_id = "hanzo/zen-embed-v2".to_string();
        assert!(contract.validate_query("kb", &wrong_model, &v, None).is_err());

        // Revision mismatch.
        let mut wrong_rev = persisted.clone();
        wrong_rev.model_revision = "sha256:deadbeef".to_string();
        assert!(contract.validate_insert("kb", &wrong_rev, &v, None).is_err());

        // Normalization policy mismatch.
        let mut wrong_norm = persisted.clone();
        wrong_norm.normalization = NormalizationPolicy::None;
        assert!(contract.validate_query("kb", &wrong_norm, &v, None).is_err());

        // Query / doc prefix metadata mismatch.
        let mut wrong_qprefix = persisted.clone();
        wrong_qprefix.query_prefix = "query: ".to_string();
        assert!(contract.validate_query("kb", &wrong_qprefix, &v, None).is_err());

        let mut wrong_dprefix = persisted.clone();
        wrong_dprefix.doc_prefix = "".to_string();
        assert!(contract.validate_insert("kb", &wrong_dprefix, &v, None).is_err());

        // Swapped runtime text prefix (using query_prefix on insert or doc_prefix on query).
        let q_text = persisted.format_query("hello");
        let d_text = persisted.format_document("world");
        let swap_insert = contract.validate_insert("kb", &persisted, &v, Some(&q_text)).unwrap_err();
        assert!(matches!(swap_insert, MetadataError::SwappedTaskPrefix { .. }));

        let swap_query = contract.validate_query("kb", &persisted, &v, Some(&d_text)).unwrap_err();
        assert!(matches!(swap_query, MetadataError::SwappedTaskPrefix { .. }));

        // Unnormalized vector rejected under L2 contract.
        let unnormalized = vec![1.0_f32; 768];
        let norm_err = contract.validate_insert("kb", &persisted, &unnormalized, Some(&d_text)).unwrap_err();
        assert!(matches!(norm_err, MetadataError::NormalizationViolation { .. }));
    }

    #[test]
    fn immutable_contract_blocks_mutation_and_unpinned_revisions() {
        let unpinned = IndexEmbeddingMetadata::new(
            "hanzo/zen-embed-v1",
            "latest",
            512,
            NormalizationPolicy::L2,
            "query: ",
            "passage: ",
        );
        assert!(matches!(unpinned, Err(MetadataError::UnpinnedModelRevision(_))));

        let mut contract = ImmutableIndexContract::new();
        let meta_512 = base_meta(512);
        contract.register("corpus", meta_512.clone()).unwrap();
        // Idempotent re-registration succeeds.
        assert!(contract.register("corpus", meta_512).is_ok());
        // Mutating dimensions fails.
        let err = contract.register("corpus", base_meta(256)).unwrap_err();
        assert!(matches!(err, MetadataError::ImmutableCollectionExists { .. }));

        // Matryoshka truncation + L2 renormalization produces valid unit vector for target dim.
        let full = vec![0.5_f32; 768];
        let trunc_256 = truncate_matryoshka(&full, 256, &NormalizationPolicy::L2).unwrap();
        assert_eq!(trunc_256.len(), 256);
        let meta_256 = base_meta(256);
        assert!(meta_256.validate_vector(&trunc_256, IndexOperation::Insert).is_ok());

        // Serialization round-trip preserves exact immutable contract.
        let json = contract.to_json().unwrap();
        let restored = ImmutableIndexContract::from_json(&json).unwrap();
        assert_eq!(contract, restored);
    }
}
