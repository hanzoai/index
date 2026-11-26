// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Architecture 1 (`exact_flat_baseline`) & Ultra-Selective Pre-Filter Fallback.
//!
//! - [`ExactFlatBaseline`]: Exact brute-force L2 scan enforcing pre-scoring ACL and
//!   structured tag isolation. Serves as the 100% `Recall@10` ground-truth oracle.
//! - [`PrefilterInvertedFallback`]: Maintains a compact `(tenant_id, acl_group, tag)`
//!   inverted posting list over canonical vectors so ultra-selective filters (`<= 1%`)
//!   scan ONLY matching candidates directly instead of starving post-filtered ANN.

use std::collections::HashMap;

use crate::canonical::{
    cmp_search_hits, AccessContext, AnnError, CanonicalStore, CanonicalVectorRecord,
    DerivedRecordMeta, IndexFootprint, SearchHit,
};
use crate::metrics::{l2_squared, validate_query_vector};

/// Exact brute-force L2 flat scan index (100% Recall@k oracle).
#[derive(Debug, Clone)]
pub struct ExactFlatBaseline {
    dim: usize,
    /// Flat contiguous `f32` vector matrix (`count * dim`).
    vectors: Vec<f32>,
    meta: Vec<DerivedRecordMeta>,
    /// Lookup from `source_id -> slot` for O(1) delete/update maintenance.
    id_to_slot: HashMap<String, usize>,
    tombstones: Vec<bool>,
    tombstone_count: usize,
}

impl ExactFlatBaseline {
    /// Build a derived exact flat index strictly from [`CanonicalStore`].
    pub fn build_from_canonical(store: &CanonicalStore) -> Result<Self, AnnError> {
        let dim = store.dim();
        let records = store.snapshot_live();
        let count = records.len();
        let mut vectors = Vec::with_capacity(count.saturating_mul(dim));
        let mut meta = Vec::with_capacity(count);
        let mut id_to_slot = HashMap::with_capacity(count);

        for (slot, rec) in records.iter().enumerate() {
            vectors.extend_from_slice(&rec.vector);
            meta.push(DerivedRecordMeta::from_record(rec));
            id_to_slot.insert(rec.source_id.clone(), slot);
        }

        Ok(Self {
            dim,
            vectors,
            meta,
            id_to_slot,
            tombstones: vec![false; count],
            tombstone_count: 0,
        })
    }

    /// Drop all derived state and rebuild cleanly from [`CanonicalStore`].
    pub fn rebuild_from_canonical(&mut self, store: &CanonicalStore) -> Result<(), AnnError> {
        *self = Self::build_from_canonical(store)?;
        Ok(())
    }

    /// Online upsert (tombstones prior version if present, appends new record).
    pub fn upsert(&mut self, record: &CanonicalVectorRecord) -> Result<(), AnnError> {
        let valid = record.clone().validated(self.dim)?;
        if let Some(&old_slot) = self.id_to_slot.get(&valid.source_id) {
            if !self.tombstones[old_slot] {
                self.tombstones[old_slot] = true;
                self.tombstone_count += 1;
            }
        }
        let slot = self.meta.len();
        self.vectors.extend_from_slice(&valid.vector);
        self.meta.push(DerivedRecordMeta::from_record(&valid));
        self.tombstones.push(false);
        self.id_to_slot.insert(valid.source_id, slot);
        Ok(())
    }

    /// Online tombstone delete by `source_id`.
    pub fn delete(&mut self, source_id: &str) -> bool {
        if let Some(&slot) = self.id_to_slot.get(source_id) {
            if !self.tombstones[slot] {
                self.tombstones[slot] = true;
                self.tombstone_count += 1;
                return true;
            }
        }
        false
    }

    /// Exact pre-filtered L2 search returning top-`k` authorized hits.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ctx: &AccessContext,
    ) -> Result<Vec<SearchHit>, AnnError> {
        ctx.validate()?;
        validate_query_vector(query, self.dim)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut candidates = Vec::new();
        for slot in 0..self.meta.len() {
            if self.tombstones[slot] {
                continue;
            }
            let m = &self.meta[slot];
            // Pre-scoring ACL & filter check: unauthorized vectors are never scored or returned.
            if !m.matches_access(ctx) {
                continue;
            }
            let base = slot * self.dim;
            let dist = l2_squared(query, &self.vectors[base..base + self.dim]);
            candidates.push(SearchHit {
                source_id: m.source_id.clone(),
                version: m.version,
                tenant_id: m.tenant_id.clone(),
                acl_group: m.acl_group.clone(),
                distance: dist,
            });
        }

        candidates.sort_by(cmp_search_hits);
        candidates.truncate(k);
        Ok(candidates)
    }

    pub fn footprint(&self) -> IndexFootprint {
        let live = self.meta.len().saturating_sub(self.tombstone_count);
        let vec_bytes = self.vectors.len() * std::mem::size_of::<f32>();
        let meta_bytes: usize = self.meta.iter().map(|m| m.memory_bytes()).sum();
        let aux_bytes = self.tombstones.len()
            + self.id_to_slot.len()
                * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());
        let ram = vec_bytes + meta_bytes + aux_bytes;
        // Exact flat has no separate disk index beyond raw vector + metadata store.
        IndexFootprint::new(live, self.tombstone_count, ram, vec_bytes)
    }
}

/// Inverted-index pre-filter fallback for ultra-selective filters (`<= 1%` selectivity).
///
/// Demonstrates the exact negative regime where blind post-filtered ANN starves or
/// over-scans, while inverted-posting pre-filtering + exact scoring achieves 100%
/// Recall@10 in microseconds.
#[derive(Debug, Clone)]
pub struct PrefilterInvertedFallback {
    dim: usize,
    vectors: Vec<f32>,
    meta: Vec<DerivedRecordMeta>,
    /// Posting lists keyed by `(tenant_id, tag)` -> matching slots.
    tag_postings: HashMap<(String, String), Vec<usize>>,
    /// Fallback posting lists keyed by `tenant_id` when no tag predicate is supplied.
    tenant_postings: HashMap<String, Vec<usize>>,
}

impl PrefilterInvertedFallback {
    pub fn build_from_canonical(store: &CanonicalStore) -> Result<Self, AnnError> {
        let dim = store.dim();
        let records = store.snapshot_live();
        let mut vectors = Vec::with_capacity(records.len().saturating_mul(dim));
        let mut meta = Vec::with_capacity(records.len());
        let mut tag_postings: HashMap<(String, String), Vec<usize>> = HashMap::new();
        let mut tenant_postings: HashMap<String, Vec<usize>> = HashMap::new();

        for (slot, rec) in records.iter().enumerate() {
            vectors.extend_from_slice(&rec.vector);
            tenant_postings.entry(rec.tenant_id.clone()).or_default().push(slot);
            for tag in &rec.filter_tags {
                tag_postings.entry((rec.tenant_id.clone(), tag.clone())).or_default().push(slot);
            }
            meta.push(DerivedRecordMeta::from_record(rec));
        }

        Ok(Self { dim, vectors, meta, tag_postings, tenant_postings })
    }

    /// Execute inverted-posting pre-filter search and return `(hits, candidates_scanned)`.
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        ctx: &AccessContext,
    ) -> Result<(Vec<SearchHit>, usize), AnnError> {
        ctx.validate()?;
        validate_query_vector(query, self.dim)?;
        if k == 0 {
            return Ok((Vec::new(), 0));
        }

        let posting = match &ctx.required_tag {
            Some(tag) => self.tag_postings.get(&(ctx.tenant_id.clone(), tag.clone())),
            None => self.tenant_postings.get(&ctx.tenant_id),
        };

        let Some(slots) = posting else {
            return Ok((Vec::new(), 0));
        };

        let mut scanned = 0usize;
        let mut hits = Vec::new();
        for &slot in slots {
            let m = &self.meta[slot];
            if !m.matches_access(ctx) {
                continue;
            }
            scanned += 1;
            let base = slot * self.dim;
            let dist = l2_squared(query, &self.vectors[base..base + self.dim]);
            hits.push(SearchHit {
                source_id: m.source_id.clone(),
                version: m.version,
                tenant_id: m.tenant_id.clone(),
                acl_group: m.acl_group.clone(),
                distance: dist,
            });
        }

        hits.sort_by(cmp_search_hits);
        hits.truncate(k);
        Ok((hits, scanned))
    }
}
