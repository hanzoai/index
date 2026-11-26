use std::collections::BTreeMap;

use crate::math::Fnv1a64;
use crate::types::{CanonicalPageRecord, ColPaliError, PageKey};

/// Authoritative canonical document page store.
///
/// Mirrors Hanzo's fixed canonical architecture (`vfs -> Hanzo S3 -> Visor`):
/// every page lives here with stable identity `(doc_id, page_no, version, tenant_id, acl_group)`
/// and canonical content. All retrieval indexes are strictly derived views that can be dropped
/// and rebuilt bit-for-bit from `CanonicalPageStore` without mutating canonical state.
#[derive(Debug, Clone)]
pub struct CanonicalPageStore {
    dim: usize,
    pages: BTreeMap<PageKey, CanonicalPageRecord>,
}

impl CanonicalPageStore {
    pub fn new(dim: usize) -> Self {
        Self { dim, pages: BTreeMap::new() }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Upsert a canonical page record after validating tenant/ACL non-emptiness and vector dimensions.
    pub fn upsert(&mut self, record: CanonicalPageRecord) -> Result<(), ColPaliError> {
        if record.tenant_id.trim().is_empty() {
            return Err(ColPaliError::EmptyTenantId);
        }
        if record.acl_group.trim().is_empty() {
            return Err(ColPaliError::EmptyAclGroups);
        }
        if record.ocr_dense_embedding.len() != self.dim {
            return Err(ColPaliError::InvalidDimension {
                expected: self.dim,
                got: record.ocr_dense_embedding.len(),
            });
        }
        if record.patch_embeddings.is_empty() {
            return Err(ColPaliError::EmptyPatchSet {
                doc_id: record.doc_id,
                page_no: record.page_no,
            });
        }
        for patch in &record.patch_embeddings {
            if patch.len() != self.dim {
                return Err(ColPaliError::InvalidDimension {
                    expected: self.dim,
                    got: patch.len(),
                });
            }
        }
        let key = record.key();
        self.pages.insert(key, record);
        Ok(())
    }

    /// Remove a canonical page record by `(doc_id, page_no)`.
    pub fn remove(&mut self, key: &PageKey) -> Option<CanonicalPageRecord> {
        self.pages.remove(key)
    }

    pub fn get(&self, key: &PageKey) -> Option<&CanonicalPageRecord> {
        self.pages.get(key)
    }

    /// Deterministic ordered iteration over all canonical pages (`BTreeMap` order by `(doc_id, page_no)`).
    pub fn iter(&self) -> impl Iterator<Item = &CanonicalPageRecord> {
        self.pages.values()
    }

    /// Deterministic 64-bit FNV-1a fingerprint of the entire canonical store.
    /// Used to verify that derived index Drop & Rebuild never mutates canonical truth.
    pub fn fingerprint(&self) -> u64 {
        let mut h = Fnv1a64::new();
        h.update_u64(self.dim as u64);
        h.update_u64(self.pages.len() as u64);
        for rec in self.pages.values() {
            h.update_str(&rec.doc_id);
            h.update_u32(rec.page_no);
            h.update_u64(rec.version);
            h.update_str(&rec.tenant_id);
            h.update_str(&rec.acl_group);
            h.update_u8(rec.page_modality.tag());
            h.update_str(&rec.source_origin);
            h.update_f32(rec.layout_features.text_coverage);
            h.update_f32(rec.layout_features.table_region_ratio);
            h.update_f32(rec.layout_features.figure_region_ratio);
            h.update_f32(rec.layout_features.formula_region_ratio);
            h.update_f32(rec.layout_features.ocr_mean_confidence);
            h.update_u8(rec.layout_features.column_count);
            h.update_str(&rec.ocr_text);
            h.update_f32(rec.ocr_corruption_rate);
            for &x in &rec.ocr_dense_embedding {
                h.update_f32(x);
            }
            h.update_u16(rec.patch_grid_rows);
            h.update_u16(rec.patch_grid_cols);
            h.update_u64(rec.patch_embeddings.len() as u64);
            for patch in &rec.patch_embeddings {
                for &x in patch {
                    h.update_f32(x);
                }
            }
        }
        h.finish()
    }
}
