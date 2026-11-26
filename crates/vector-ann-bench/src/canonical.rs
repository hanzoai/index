// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical source-of-truth vector records, pre-return ACL isolation, and
//! derived-index rebuild contracts for Hanzo Index (`hanzoai/index#15` / HIP-1347).
//!
//! # Invariants
//! 1. **Canonical/Derived Separation**: [`CanonicalStore`] owns the authoritative
//!    `(source_id, version, tenant_id, acl_group, filter_tags, vector)` records.
//!    Every ANN index (`exact_flat_baseline`, `local_mmap_vamana`,
//!    `object_native_ivf_pq`, `object_store_random_graph_hops`) is strictly
//!    **derived** and can be dropped and rebuilt bit-exact from canonical truth.
//! 2. **Fail-Closed Pre-Return Tenant/ACL Isolation**: Empty `tenant_id` or
//!    empty `allowed_acl_groups` fail closed with an explicit [`AnnError`].
//!    Tenant and ACL checks execute before any hit is returned (`0.0%` leaks).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Error type for canonical validation, ACL isolation, quantization, and storage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnError {
    /// Caller supplied an empty `tenant_id` (fail-closed isolation boundary).
    EmptyTenantId,
    /// Caller supplied an empty `allowed_acl_groups` list (fail-closed isolation boundary).
    EmptyAclGroups,
    /// Record contained an empty `source_id`.
    EmptySourceId,
    /// Record contained an empty `acl_group`.
    EmptyAclGroup,
    /// Vector dimension mismatch against the store or index schema.
    DimensionMismatch { expected: usize, actual: usize },
    /// Vector contained NaN or infinite values.
    NonFiniteVector,
    /// Vector had zero norm and could not be normalized.
    ZeroNormVector,
    /// Index build or query received invalid configuration parameters.
    InvalidConfig(String),
    /// Binary serialization/deserialization or checksum validation failed.
    CorruptFormat(String),
    /// Local filesystem or object store I/O error.
    Io(String),
}

impl fmt::Display for AnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTenantId => write!(f, "tenant_id must be non-empty (fail-closed ACL)"),
            Self::EmptyAclGroups => {
                write!(f, "allowed_acl_groups must be non-empty (fail-closed ACL)")
            }
            Self::EmptySourceId => write!(f, "source_id must be non-empty"),
            Self::EmptyAclGroup => write!(f, "acl_group must be non-empty"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "vector dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NonFiniteVector => write!(f, "vector contains NaN or infinite components"),
            Self::ZeroNormVector => write!(f, "vector has zero L2 norm"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::CorruptFormat(msg) => write!(f, "corrupt index format: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for AnnError {}

/// Canonical source record stored in Hanzo's authoritative store.
///
/// Every derived vector index references records strictly by `(source_id, version)`
/// and never becomes the source of truth.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalVectorRecord {
    /// Stable canonical identifier (e.g., `blob:crates/milli/src/lib.rs#chunk0`).
    pub source_id: String,
    /// Monotonically increasing version for this `source_id`.
    pub version: u64,
    /// Owning tenant identifier. Must be non-empty.
    pub tenant_id: String,
    /// Required ACL group for authorization (e.g., `eng-core`, `sec-ops`, `public`).
    pub acl_group: String,
    /// Structured metadata filter tags (sorted & deduplicated on ingest).
    pub filter_tags: Vec<String>,
    /// L2-normalized embedding vector.
    pub vector: Vec<f32>,
}

impl CanonicalVectorRecord {
    /// Validate record invariants and return a normalized copy (sorted/deduped tags).
    pub fn validated(mut self, expected_dim: usize) -> Result<Self, AnnError> {
        if self.source_id.trim().is_empty() {
            return Err(AnnError::EmptySourceId);
        }
        if self.tenant_id.trim().is_empty() {
            return Err(AnnError::EmptyTenantId);
        }
        if self.acl_group.trim().is_empty() {
            return Err(AnnError::EmptyAclGroup);
        }
        if self.vector.len() != expected_dim {
            return Err(AnnError::DimensionMismatch {
                expected: expected_dim,
                actual: self.vector.len(),
            });
        }
        let mut norm_sq = 0.0_f64;
        for &v in &self.vector {
            if !v.is_finite() {
                return Err(AnnError::NonFiniteVector);
            }
            norm_sq += f64::from(v) * f64::from(v);
        }
        if norm_sq <= 1e-24 {
            return Err(AnnError::ZeroNormVector);
        }
        let inv_norm = (1.0 / norm_sq.sqrt()) as f32;
        for v in &mut self.vector {
            *v *= inv_norm;
        }
        self.filter_tags.sort();
        self.filter_tags.dedup();
        Ok(self)
    }

    /// Returns true if this record carries `tag`.
    #[inline]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.filter_tags.binary_search_by(|t| t.as_str().cmp(tag)).is_ok()
    }
}

/// Caller security and filter context enforced **before** any hit is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessContext {
    /// Authenticated tenant identifier. Empty fails closed.
    pub tenant_id: String,
    /// Authorized ACL groups held by the caller. Empty fails closed.
    pub allowed_acl_groups: Vec<String>,
    /// Optional structured metadata filter predicate (`filter_tags` must contain this tag).
    pub required_tag: Option<String>,
}

impl AccessContext {
    /// Construct a new caller access context.
    pub fn new(
        tenant_id: impl Into<String>,
        allowed_acl_groups: Vec<String>,
        required_tag: Option<String>,
    ) -> Self {
        Self { tenant_id: tenant_id.into(), allowed_acl_groups, required_tag }
    }

    /// Validate fail-closed security preconditions before executing any query.
    pub fn validate(&self) -> Result<(), AnnError> {
        if self.tenant_id.trim().is_empty() {
            return Err(AnnError::EmptyTenantId);
        }
        if self.allowed_acl_groups.is_empty()
            || self.allowed_acl_groups.iter().all(|g| g.trim().is_empty())
        {
            return Err(AnnError::EmptyAclGroups);
        }
        Ok(())
    }

    /// Check tenant + ACL isolation ONLY (ignoring optional metadata filter tag).
    #[inline]
    pub fn authorizes_acl(&self, tenant_id: &str, acl_group: &str) -> bool {
        tenant_id == self.tenant_id
            && self.allowed_acl_groups.iter().any(|g| g.as_str() == acl_group)
    }

    /// Check tenant + ACL isolation AND optional structured metadata filter tag.
    #[inline]
    pub fn matches_record(&self, record: &CanonicalVectorRecord) -> bool {
        if !self.authorizes_acl(&record.tenant_id, &record.acl_group) {
            return false;
        }
        match &self.required_tag {
            Some(tag) => record.has_tag(tag),
            None => true,
        }
    }
}

/// Compact metadata summary carried inside derived index entries so ACL checks
/// execute before any hit is returned (and during traversal/segment scan).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedRecordMeta {
    pub source_id: String,
    pub version: u64,
    pub tenant_id: String,
    pub acl_group: String,
    pub filter_tags: Vec<String>,
}

impl DerivedRecordMeta {
    pub fn from_record(record: &CanonicalVectorRecord) -> Self {
        Self {
            source_id: record.source_id.clone(),
            version: record.version,
            tenant_id: record.tenant_id.clone(),
            acl_group: record.acl_group.clone(),
            filter_tags: record.filter_tags.clone(),
        }
    }

    /// Exact heap + struct memory footprint in bytes for accurate RAM accounting.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.source_id.capacity()
            + self.tenant_id.capacity()
            + self.acl_group.capacity()
            + self.filter_tags.capacity() * std::mem::size_of::<String>()
            + self.filter_tags.iter().map(|t| t.capacity()).sum::<usize>()
    }

    #[inline]
    pub fn matches_access(&self, ctx: &AccessContext) -> bool {
        if !ctx.authorizes_acl(&self.tenant_id, &self.acl_group) {
            return false;
        }
        match &ctx.required_tag {
            Some(tag) => {
                self.filter_tags.binary_search_by(|t| t.as_str().cmp(tag.as_str())).is_ok()
            }
            None => true,
        }
    }
}

/// Ranked vector search hit returned to callers after pre-return ACL filtering.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub source_id: String,
    pub version: u64,
    pub tenant_id: String,
    pub acl_group: String,
    /// Squared L2 distance (smaller is closer; for unit-norm vectors `l2_sq = 2 * (1 - cosine)`).
    pub distance: f32,
}

/// Exact RAM and storage footprint measured directly from live data structures.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IndexFootprint {
    pub live_vectors: usize,
    pub tombstoned_vectors: usize,
    pub ram_bytes: usize,
    pub disk_or_object_bytes: usize,
    pub bytes_per_vector_ram: f64,
    pub bytes_per_vector_disk: f64,
}

impl IndexFootprint {
    pub fn new(
        live_vectors: usize,
        tombstoned_vectors: usize,
        ram_bytes: usize,
        disk_or_object_bytes: usize,
    ) -> Self {
        let denom = live_vectors.max(1) as f64;
        Self {
            live_vectors,
            tombstoned_vectors,
            ram_bytes,
            disk_or_object_bytes,
            bytes_per_vector_ram: ram_bytes as f64 / denom,
            bytes_per_vector_disk: disk_or_object_bytes as f64 / denom,
        }
    }
}

/// Authoritative canonical store holding live records keyed deterministically by `source_id`.
///
/// Using `BTreeMap` guarantees deterministic iteration order across platforms and runs,
/// enabling bit-exact derived index rebuilds.
#[derive(Debug, Clone)]
pub struct CanonicalStore {
    dim: usize,
    records: BTreeMap<String, CanonicalVectorRecord>,
    deleted_ids: BTreeSet<String>,
}

impl CanonicalStore {
    /// Create a new canonical store enforcing fixed vector dimensionality `dim`.
    pub fn new(dim: usize) -> Result<Self, AnnError> {
        if dim == 0 {
            return Err(AnnError::InvalidConfig("vector dimension must be > 0".into()));
        }
        Ok(Self { dim, records: BTreeMap::new(), deleted_ids: BTreeSet::new() })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Insert or update a canonical vector record.
    pub fn upsert(&mut self, record: CanonicalVectorRecord) -> Result<(), AnnError> {
        let valid = record.validated(self.dim)?;
        self.deleted_ids.remove(&valid.source_id);
        self.records.insert(valid.source_id.clone(), valid);
        Ok(())
    }

    /// Delete a canonical vector record by `source_id`. Returns `true` if present.
    pub fn delete(&mut self, source_id: &str) -> bool {
        if self.records.remove(source_id).is_some() {
            self.deleted_ids.insert(source_id.to_string());
            true
        } else {
            false
        }
    }

    pub fn get(&self, source_id: &str) -> Option<&CanonicalVectorRecord> {
        self.records.get(source_id)
    }

    /// Deterministic ordered snapshot of all live canonical records.
    pub fn snapshot_live(&self) -> Vec<CanonicalVectorRecord> {
        self.records.values().cloned().collect()
    }

    /// Count how many live records authorize and match `ctx`.
    pub fn count_matching(&self, ctx: &AccessContext) -> Result<usize, AnnError> {
        ctx.validate()?;
        Ok(self.records.values().filter(|r| ctx.matches_record(r)).count())
    }

    /// Verify zero ACL/tenant leaks AND zero stale/tombstoned versions across returned hits.
    pub fn audit_hits(
        &self,
        ctx: &AccessContext,
        hits: &[SearchHit],
    ) -> Result<HitSecurityAudit, AnnError> {
        ctx.validate()?;
        let mut acl_leaks = 0usize;
        let mut filter_violations = 0usize;
        let mut stale_or_deleted_hits = 0usize;

        for hit in hits {
            if !ctx.authorizes_acl(&hit.tenant_id, &hit.acl_group) {
                acl_leaks += 1;
            }
            match self.records.get(&hit.source_id) {
                None => stale_or_deleted_hits += 1,
                Some(live) => {
                    if live.version != hit.version {
                        stale_or_deleted_hits += 1;
                    }
                    if !ctx.authorizes_acl(&live.tenant_id, &live.acl_group) {
                        acl_leaks += 1;
                    }
                    if !ctx.matches_record(live) {
                        filter_violations += 1;
                    }
                }
            }
        }

        Ok(HitSecurityAudit {
            checked_hits: hits.len(),
            acl_leaks,
            filter_violations,
            stale_or_deleted_hits,
        })
    }

    /// Deterministic FNV-1a 64-bit fingerprint of all canonical identities & versions.
    pub fn canonical_fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for rec in self.records.values() {
            for b in rec.source_id.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in rec.version.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in rec.tenant_id.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in rec.acl_group.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
}

/// Security and consistency verification report for a batch of search hits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HitSecurityAudit {
    pub checked_hits: usize,
    pub acl_leaks: usize,
    pub filter_violations: usize,
    pub stale_or_deleted_hits: usize,
}

impl HitSecurityAudit {
    pub fn is_clean(&self) -> bool {
        self.acl_leaks == 0 && self.filter_violations == 0 && self.stale_or_deleted_hits == 0
    }
}

/// Common deterministic tie-breaking comparison for [`SearchHit`]:
/// `(distance ASC, source_id ASC, version DESC)`.
#[inline]
pub fn cmp_search_hits(a: &SearchHit, b: &SearchHit) -> std::cmp::Ordering {
    a.distance
        .total_cmp(&b.distance)
        .then_with(|| a.source_id.cmp(&b.source_id))
        .then_with(|| b.version.cmp(&a.version))
}
