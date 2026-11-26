// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Architecture 3 (`object_native_ivf_pq`): LanceDB / Hanzo S3/VFS-style
//! immutable coarse-partitioned IVF-PQ + SQ8 segment index.
//!
//! # Architectural Principle
//! Object storage MUST NOT serve random pointer-chasing graph hops. Instead:
//! 1. **Hot compute-local routing state** caches ONLY the compact coarse centroid
//!    table, PQ codebook, partition byte-range directory, and per-partition
//!    tenant/tag bloom summaries.
//! 2. **Immutable segment packs** on [`InstrumentedObjectStore`] pack each coarse
//!    partition's `[record_id, pq_codes, sq8_vec]` contiguously.
//! 3. A cold-cache query executes **1 bounded parallel batch of `nprobe` contiguous
//!    `RangeGET`s** (`sequential_rtts == 1`), then filters ACLs pre-return, scores
//!    via O(M) PQ ADC lookup tables, and refines top candidates with the contiguous
//!    SQ8 codes fetched in the same single batch!
//! 4. Online deletes and upserts NEVER mutate existing S3 objects; they write small
//!    immutable tombstone/delta segment versions and compact cleanly into version `v+1`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::canonical::{
    cmp_search_hits, AccessContext, AnnError, CanonicalStore, CanonicalVectorRecord,
    DerivedRecordMeta, IndexFootprint, SearchHit,
};
use crate::local_mmap_vamana::CompactionReport;
use crate::metrics::{
    l2_squared, validate_query_vector, ProductQuantizer, SplitMix64, Sq8Quantizer,
};
use crate::object_store::{ByteRangeRequest, InstrumentedObjectStore, ObjectStoreStats};

/// Configuration for [`ObjectNativeIvfPqIndex`].
#[derive(Debug, Clone, Copy)]
pub struct IvfPqConfig {
    /// Number of coarse IVF partitions (`C`).
    pub num_partitions: usize,
    /// Number of closest partitions probed per query (`nprobe`).
    pub nprobe: usize,
    /// Number of PQ subspaces (`M`, must divide `dim`).
    pub pq_subspaces: usize,
    /// Centroids per PQ subspace (`K_sub`, `<= 256`).
    pub pq_k_sub: usize,
    /// Number of PQ candidates refined with contiguous SQ8 vectors.
    pub refine_factor: usize,
    /// K-means Lloyd iterations for coarse centroids & PQ training.
    pub kmeans_iters: usize,
    /// Deterministic training seed.
    pub seed: u64,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            num_partitions: 64,
            nprobe: 8,
            pq_subspaces: 8,
            pq_k_sub: 64,
            refine_factor: 6,
            kmeans_iters: 10,
            seed: 1347,
        }
    }
}

/// Partition directory entry cached hot next to compute.
#[derive(Debug, Clone)]
struct PartitionEntry {
    object_key: String,
    byte_offset: usize,
    byte_len: usize,
    count: usize,
    /// Compact per-partition tenant summary for partition pruning.
    tenants_present: HashSet<String>,
    /// Compact per-partition tag summary for filtered partition pruning.
    tags_present: HashSet<String>,
}

#[derive(Debug, Clone)]
struct DeltaSegmentEntry {
    meta: DerivedRecordMeta,
    pq_codes: Vec<u8>,
    sq8_code: Vec<u8>,
    tombstoned: bool,
}

/// Architecture 3: Immutable segment IVF-PQ + SQ8 index backed by [`InstrumentedObjectStore`].
pub struct ObjectNativeIvfPqIndex {
    dim: usize,
    config: IvfPqConfig,
    segment_version: u64,
    store: Arc<InstrumentedObjectStore>,
    /// Hot compute-local coarse centroids: `actual_partitions * dim` floats.
    centroids: Vec<f32>,
    actual_partitions: usize,
    /// Hot compute-local PQ codebook.
    pq: ProductQuantizer,
    /// Hot compute-local SQ8 parameters for single-round-trip reranking.
    sq8: Sq8Quantizer,
    /// Hot compute-local partition byte-range directory.
    partitions: Vec<PartitionEntry>,
    /// Compact record metadata indexed by dense `record_idx` (for pre-return ACL checks).
    meta: Vec<DerivedRecordMeta>,
    id_to_base_idx: HashMap<String, usize>,
    base_tombstones: Vec<bool>,
    base_tombstone_count: usize,
    /// Immutable delta segment records written after base segment creation.
    delta_entries: Vec<DeltaSegmentEntry>,
    delta_id_to_idx: HashMap<String, usize>,
    delta_tombstone_count: usize,
    delta_version: u64,
    tombstone_manifest_version: u64,
}

impl ObjectNativeIvfPqIndex {
    /// Build immutable IVF-PQ + SQ8 segments strictly from [`CanonicalStore`] and
    /// persist them to `store`.
    pub fn build_from_canonical(
        canonical: &CanonicalStore,
        store: Arc<InstrumentedObjectStore>,
        config: IvfPqConfig,
    ) -> Result<Self, AnnError> {
        Self::build_version(canonical, store, config, 1)
    }

    fn build_version(
        canonical: &CanonicalStore,
        store: Arc<InstrumentedObjectStore>,
        config: IvfPqConfig,
        segment_version: u64,
    ) -> Result<Self, AnnError> {
        if config.num_partitions == 0 || config.nprobe == 0 {
            return Err(AnnError::InvalidConfig("num_partitions and nprobe must be > 0".into()));
        }

        let dim = canonical.dim();
        let records = canonical.snapshot_live();
        let n = records.len();
        let vec_refs: Vec<&[f32]> = records.iter().map(|r| r.vector.as_slice()).collect();

        let sq8 = Sq8Quantizer::fit(dim, &vec_refs)?;
        let pq = ProductQuantizer::train(
            dim,
            config.pq_subspaces,
            config.pq_k_sub,
            &vec_refs,
            config.kmeans_iters,
            config.seed,
        )?;

        let actual_partitions = config.num_partitions.min(n.max(1));
        let (centroids, assignments) =
            train_coarse_ivf(dim, &vec_refs, actual_partitions, config.kmeans_iters, config.seed);

        let mut partition_members: Vec<Vec<usize>> = vec![Vec::new(); actual_partitions];
        for (idx, &part_id) in assignments.iter().enumerate() {
            partition_members[part_id].push(idx);
        }

        let pack_key = format!("ivf_pq/v{segment_version}/partitions.pack");
        let manifest_key = format!("ivf_pq/v{segment_version}/manifest.bin");
        let entry_stride = 4 + pq.num_subspaces + dim;
        let mut pack_bytes = Vec::with_capacity(n.saturating_mul(entry_stride));
        let mut partitions = Vec::with_capacity(actual_partitions);

        for members in &partition_members {
            let byte_offset = pack_bytes.len();
            let mut tenants_present = HashSet::new();
            let mut tags_present = HashSet::new();

            for &rec_idx in members {
                let rec = &records[rec_idx];
                tenants_present.insert(rec.tenant_id.clone());
                for tag in &rec.filter_tags {
                    tags_present.insert(tag.clone());
                }
                let pq_code = pq.encode(&rec.vector);
                let sq8_code = sq8.encode(&rec.vector);
                pack_bytes.extend_from_slice(&(rec_idx as u32).to_le_bytes());
                pack_bytes.extend_from_slice(&pq_code);
                pack_bytes.extend_from_slice(&sq8_code);
            }

            let byte_len = pack_bytes.len() - byte_offset;
            partitions.push(PartitionEntry {
                object_key: pack_key.clone(),
                byte_offset,
                byte_len,
                count: members.len(),
                tenants_present,
                tags_present,
            });
        }

        // Persist immutable partition pack & manifest objects.
        store.put_immutable(&pack_key, pack_bytes)?;

        let mut manifest_bytes = Vec::with_capacity(
            32 + centroids.len() * 4
                + pq.codebooks.len() * 4
                + sq8.serialized_bytes()
                + actual_partitions * 16,
        );
        manifest_bytes.extend_from_slice(b"HSEG");
        manifest_bytes.extend_from_slice(&1u32.to_le_bytes());
        manifest_bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        manifest_bytes.extend_from_slice(&(n as u32).to_le_bytes());
        manifest_bytes.extend_from_slice(&(actual_partitions as u32).to_le_bytes());
        manifest_bytes.extend_from_slice(&(pq.num_subspaces as u32).to_le_bytes());
        manifest_bytes.extend_from_slice(&(pq.k_sub as u32).to_le_bytes());
        for &c in &centroids {
            manifest_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for &c in &pq.codebooks {
            manifest_bytes.extend_from_slice(&c.to_le_bytes());
        }
        sq8.write_to_buf(&mut manifest_bytes);
        for p in &partitions {
            manifest_bytes.extend_from_slice(&(p.byte_offset as u64).to_le_bytes());
            manifest_bytes.extend_from_slice(&(p.byte_len as u64).to_le_bytes());
        }
        store.put_immutable(&manifest_key, manifest_bytes)?;

        let mut meta = Vec::with_capacity(n);
        let mut id_to_base_idx = HashMap::with_capacity(n);
        for (i, rec) in records.iter().enumerate() {
            meta.push(DerivedRecordMeta::from_record(rec));
            id_to_base_idx.insert(rec.source_id.clone(), i);
        }

        Ok(Self {
            dim,
            config,
            segment_version,
            store,
            centroids,
            actual_partitions,
            pq,
            sq8,
            partitions,
            meta,
            id_to_base_idx,
            base_tombstones: vec![false; n],
            base_tombstone_count: 0,
            delta_entries: Vec::new(),
            delta_id_to_idx: HashMap::new(),
            delta_tombstone_count: 0,
            delta_version: 0,
            tombstone_manifest_version: 0,
        })
    }

    /// Evict compute-local partition cache to measure cold-cache S3 performance.
    pub fn clear_local_cache(&self) -> Result<(), AnnError> {
        self.store.clear_cache()
    }

    pub fn io_stats(&self) -> ObjectStoreStats {
        self.store.stats()
    }

    pub fn store(&self) -> &Arc<InstrumentedObjectStore> {
        &self.store
    }

    /// Drop all immutable objects & rebuild cleanly from [`CanonicalStore`].
    pub fn rebuild_from_canonical(&mut self, canonical: &CanonicalStore) -> Result<(), AnnError> {
        self.store.clear_all()?;
        let rebuilt = Self::build_version(canonical, Arc::clone(&self.store), self.config, 1)?;
        *self = rebuilt;
        Ok(())
    }

    /// Immutable tombstone delete: appends a new tombstone manifest object without
    /// mutating existing immutable partition objects.
    pub fn delete(&mut self, source_id: &str) -> Result<bool, AnnError> {
        let mut deleted = false;
        if let Some(&idx) = self.id_to_base_idx.get(source_id) {
            if !self.base_tombstones[idx] {
                self.base_tombstones[idx] = true;
                self.base_tombstone_count += 1;
                deleted = true;
            }
        }
        if let Some(&idx) = self.delta_id_to_idx.get(source_id) {
            if !self.delta_entries[idx].tombstoned {
                self.delta_entries[idx].tombstoned = true;
                self.delta_tombstone_count += 1;
                deleted = true;
            }
        }
        if deleted {
            self.tombstone_manifest_version += 1;
            let key = format!(
                "ivf_pq/v{}/tombstones_{}.bin",
                self.segment_version, self.tombstone_manifest_version
            );
            self.store.put_immutable(key, source_id.as_bytes().to_vec())?;
        }
        Ok(deleted)
    }

    /// Immutable delta segment upsert: tombstones prior version if present and persists
    /// an immutable delta object on [`InstrumentedObjectStore`].
    pub fn upsert(&mut self, record: &CanonicalVectorRecord) -> Result<(), AnnError> {
        let valid = record.clone().validated(self.dim)?;
        let _ = self.delete(&valid.source_id)?;
        let pq_codes = self.pq.encode(&valid.vector);
        let sq8_code = self.sq8.encode(&valid.vector);

        self.delta_version += 1;
        let key = format!("ivf_pq/v{}/delta_{}.bin", self.segment_version, self.delta_version);
        let mut payload = Vec::with_capacity(pq_codes.len() + sq8_code.len() + 8);
        payload.extend_from_slice(&(valid.version).to_le_bytes());
        payload.extend_from_slice(&pq_codes);
        payload.extend_from_slice(&sq8_code);
        self.store.put_immutable(key, payload)?;

        let idx = self.delta_entries.len();
        self.delta_id_to_idx.insert(valid.source_id.clone(), idx);
        self.delta_entries.push(DeltaSegmentEntry {
            meta: DerivedRecordMeta::from_record(&valid),
            pq_codes,
            sq8_code,
            tombstoned: false,
        });
        Ok(())
    }

    /// Compact immutable segments into a clean next-version pack dropping all tombstones.
    pub fn compact(&mut self, canonical: &CanonicalStore) -> Result<CompactionReport, AnnError> {
        let start = std::time::Instant::now();
        let bytes_before = self.store.total_stored_bytes();
        let tombstones_reclaimed = self.base_tombstone_count + self.delta_tombstone_count;
        let delta_entries_merged = self.delta_entries.iter().filter(|e| !e.tombstoned).count();
        let next_ver = self.segment_version + 1;

        self.store.clear_all()?;
        let rebuilt =
            Self::build_version(canonical, Arc::clone(&self.store), self.config, next_ver)?;
        *self = rebuilt;

        let bytes_after = self.store.total_stored_bytes();
        let duration_ms = start.elapsed().as_secs_f64() * 1_000.0;
        Ok(CompactionReport {
            tombstones_reclaimed,
            delta_entries_merged,
            bytes_before,
            bytes_after,
            duration_ms,
        })
    }

    /// Object-native IVF-PQ search:
    /// 1. Rank coarse centroids in hot compute-local RAM (0 S3 GETs).
    /// 2. Issue **1 parallel batch of `nprobe` contiguous `RangeGET`s** (`1` sequential RTT).
    /// 3. Pre-return ACL + filter + tombstone enforcement -> PQ ADC -> SQ8 refine!
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

        // Step 1: Score coarse centroids locally (hot compute RAM).
        let mut part_scores: Vec<(f32, usize)> = Vec::with_capacity(self.actual_partitions);
        for p in 0..self.actual_partitions {
            let entry = &self.partitions[p];
            if entry.count == 0 {
                continue;
            }
            // Partition-level tenant & tag pruning (skip partitions guaranteed to have 0 matches).
            if !entry.tenants_present.contains(&ctx.tenant_id) {
                continue;
            }
            if let Some(tag) = &ctx.required_tag {
                if !entry.tags_present.contains(tag) {
                    continue;
                }
            }
            let c_off = p * self.dim;
            let dist = l2_squared(query, &self.centroids[c_off..c_off + self.dim]);
            part_scores.push((dist, p));
        }
        part_scores.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // Widen nprobe slightly under structured tag filters while keeping it 1 parallel batch RTT!
        let effective_nprobe = if ctx.required_tag.is_some() {
            (self.config.nprobe * 2).min(part_scores.len())
        } else {
            self.config.nprobe.min(part_scores.len())
        };

        let requests: Vec<ByteRangeRequest> = part_scores
            .iter()
            .take(effective_nprobe)
            .map(|&(_, p)| {
                let pe = &self.partitions[p];
                ByteRangeRequest {
                    key: pe.object_key.clone(),
                    offset: pe.byte_offset,
                    len: pe.byte_len,
                }
            })
            .collect();

        let adc_table = self.pq.build_adc_table(query);
        let entry_stride = 4 + self.pq.num_subspaces + self.dim;
        let mut pq_shortlist: Vec<(f32, usize, Vec<u8>)> = Vec::new();

        // Step 2: Single bounded parallel RangeGET batch (1 sequential RTT on cold cache!).
        if !requests.is_empty() {
            let chunks = self.store.get_ranges_batch(&requests)?;
            for chunk in chunks {
                let mut pos = 0usize;
                while pos + entry_stride <= chunk.len() {
                    let rec_idx =
                        u32::from_le_bytes(chunk[pos..pos + 4].try_into().unwrap()) as usize;
                    let pq_start = pos + 4;
                    let pq_end = pq_start + self.pq.num_subspaces;
                    let sq8_end = pq_end + self.dim;
                    pos += entry_stride;

                    if rec_idx >= self.meta.len() || self.base_tombstones[rec_idx] {
                        continue;
                    }
                    // Pre-return ACL + tenant + filter check BEFORE scoring/ranking.
                    if !self.meta[rec_idx].matches_access(ctx) {
                        continue;
                    }
                    let pq_dist = self.pq.adc_distance(&adc_table, &chunk[pq_start..pq_end]);
                    pq_shortlist.push((pq_dist, rec_idx, chunk[pq_end..sq8_end].to_vec()));
                }
            }
        }

        // Step 3: Refine top PQ candidates using contiguous SQ8 vectors already fetched in the same batch.
        pq_shortlist.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let refine_cap = k.saturating_mul(self.config.refine_factor).max(k);
        let mut hits: Vec<SearchHit> = Vec::with_capacity(refine_cap + self.delta_entries.len());

        for (_, rec_idx, sq8_code) in pq_shortlist.into_iter().take(refine_cap) {
            let dist = self.sq8.asymmetric_l2_sq(query, &sq8_code);
            let m = &self.meta[rec_idx];
            hits.push(SearchHit {
                source_id: m.source_id.clone(),
                version: m.version,
                tenant_id: m.tenant_id.clone(),
                acl_group: m.acl_group.clone(),
                distance: dist,
            });
        }

        // Step 4: Merge fresh immutable delta segments.
        for entry in &self.delta_entries {
            if entry.tombstoned || !entry.meta.matches_access(ctx) {
                continue;
            }
            let dist = self.sq8.asymmetric_l2_sq(query, &entry.sq8_code);
            hits.push(SearchHit {
                source_id: entry.meta.source_id.clone(),
                version: entry.meta.version,
                tenant_id: entry.meta.tenant_id.clone(),
                acl_group: entry.meta.acl_group.clone(),
                distance: dist,
            });
        }

        hits.sort_by(cmp_search_hits);
        hits.truncate(k);
        Ok(hits)
    }

    /// Blind post-filtered IVF-PQ search (negative-regime control for `1%` selectivity):
    /// probes fixed `nprobe` partitions WITHOUT partition-level tag pruning or widening
    /// and truncates to `postfilter_cap` PQ candidates BEFORE applying the tag filter.
    pub fn search_blind_postfilter(
        &self,
        query: &[f32],
        k: usize,
        postfilter_cap: usize,
        ctx: &AccessContext,
    ) -> Result<Vec<SearchHit>, AnnError> {
        ctx.validate()?;
        validate_query_vector(query, self.dim)?;
        if k == 0 || self.actual_partitions == 0 {
            return Ok(Vec::new());
        }

        let mut part_scores: Vec<(f32, usize)> = (0..self.actual_partitions)
            .filter(|&p| self.partitions[p].count > 0)
            .map(|p| {
                let c_off = p * self.dim;
                (l2_squared(query, &self.centroids[c_off..c_off + self.dim]), p)
            })
            .collect();
        part_scores.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let requests: Vec<ByteRangeRequest> = part_scores
            .iter()
            .take(self.config.nprobe.min(part_scores.len()))
            .map(|&(_, p)| {
                let pe = &self.partitions[p];
                ByteRangeRequest {
                    key: pe.object_key.clone(),
                    offset: pe.byte_offset,
                    len: pe.byte_len,
                }
            })
            .collect();

        let adc_table = self.pq.build_adc_table(query);
        let entry_stride = 4 + self.pq.num_subspaces + self.dim;
        let mut unfiltered: Vec<(f32, usize, Vec<u8>)> = Vec::new();

        if !requests.is_empty() {
            let chunks = self.store.get_ranges_batch(&requests)?;
            for chunk in chunks {
                let mut pos = 0usize;
                while pos + entry_stride <= chunk.len() {
                    let rec_idx =
                        u32::from_le_bytes(chunk[pos..pos + 4].try_into().unwrap()) as usize;
                    let pq_start = pos + 4;
                    let pq_end = pq_start + self.pq.num_subspaces;
                    let sq8_end = pq_end + self.dim;
                    pos += entry_stride;
                    if rec_idx >= self.meta.len() || self.base_tombstones[rec_idx] {
                        continue;
                    }
                    // Tenant/ACL isolation is STILL enforced (never relaxed!), only tag filter is post-applied.
                    if !ctx.authorizes_acl(
                        &self.meta[rec_idx].tenant_id,
                        &self.meta[rec_idx].acl_group,
                    ) {
                        continue;
                    }
                    let pq_dist = self.pq.adc_distance(&adc_table, &chunk[pq_start..pq_end]);
                    unfiltered.push((pq_dist, rec_idx, chunk[pq_end..sq8_end].to_vec()));
                }
            }
        }

        unfiltered.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut hits = Vec::new();
        for (_, rec_idx, sq8_code) in unfiltered.into_iter().take(postfilter_cap.max(k)) {
            let m = &self.meta[rec_idx];
            if m.matches_access(ctx) {
                let dist = self.sq8.asymmetric_l2_sq(query, &sq8_code);
                hits.push(SearchHit {
                    source_id: m.source_id.clone(),
                    version: m.version,
                    tenant_id: m.tenant_id.clone(),
                    acl_group: m.acl_group.clone(),
                    distance: dist,
                });
            }
        }
        hits.sort_by(cmp_search_hits);
        hits.truncate(k);
        Ok(hits)
    }

    /// Exact resident compute RAM vs immutable S3 object storage footprint.
    ///
    /// Counts ALL resident compute-local allocations consistently with the flat
    /// and local Vamana baselines: routing state (centroids, PQ codebook, SQ8
    /// params, partition directory & tenant/tag summaries), resident metadata
    /// table (`meta`), ID lookup maps (`id_to_base_idx`, `delta_id_to_idx`),
    /// tombstone bitvec, delta entries, and local object-cache bytes.
    pub fn footprint(&self) -> IndexFootprint {
        let live_base = self.meta.len().saturating_sub(self.base_tombstone_count);
        let live_delta = self.delta_entries.len().saturating_sub(self.delta_tombstone_count);
        let live = live_base + live_delta;
        let tombstones = self.base_tombstone_count + self.delta_tombstone_count;

        let centroids_ram = self.centroids.capacity() * std::mem::size_of::<f32>();
        let pq_ram = self.pq.memory_bytes();
        let sq8_ram = self.sq8.serialized_bytes();
        let dir_ram: usize = self
            .partitions
            .iter()
            .map(|p| {
                std::mem::size_of::<PartitionEntry>()
                    + p.object_key.capacity()
                    + p.tenants_present
                        .iter()
                        .map(|s| std::mem::size_of::<String>() + s.capacity())
                        .sum::<usize>()
                    + p.tags_present
                        .iter()
                        .map(|s| std::mem::size_of::<String>() + s.capacity())
                        .sum::<usize>()
            })
            .sum();
        let meta_ram: usize = self.meta.iter().map(|m| m.memory_bytes()).sum();
        let id_map_ram = crate::canonical::string_index_map_bytes(&self.id_to_base_idx)
            + crate::canonical::string_index_map_bytes(&self.delta_id_to_idx);
        let tombstone_ram = self.base_tombstones.capacity() * std::mem::size_of::<bool>();
        let delta_ram: usize = self
            .delta_entries
            .iter()
            .map(|e| {
                std::mem::size_of::<DeltaSegmentEntry>() - std::mem::size_of::<DerivedRecordMeta>()
                    + e.meta.memory_bytes()
                    + e.pq_codes.capacity()
                    + e.sq8_code.capacity()
            })
            .sum();
        let cache_ram = self.store.cached_bytes();
        let resident_ram = centroids_ram
            + pq_ram
            + sq8_ram
            + dir_ram
            + meta_ram
            + id_map_ram
            + tombstone_ram
            + delta_ram
            + cache_ram;
        let object_bytes = self.store.total_stored_bytes();

        IndexFootprint::new(live, tombstones, resident_ram, object_bytes)
    }

    /// Exact bytes stored per vector inside the immutable S3 partition pack (`4 + M + D`).
    pub fn packed_record_bytes(&self) -> usize {
        4 + self.pq.num_subspaces + self.dim
    }
}

/// Deterministic k-means++ initialization + Lloyd iterations for coarse IVF partitioning.
fn train_coarse_ivf(
    dim: usize,
    vectors: &[&[f32]],
    k: usize,
    iters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<usize>) {
    let n = vectors.len();
    if n == 0 {
        return (vec![0.0; k * dim], Vec::new());
    }
    let mut rng = SplitMix64::new(seed ^ 0x4956_465f_4b4d);
    let mut centroids = vec![0.0_f32; k * dim];

    // Deterministic k-means++ seeding.
    let first = rng.next_usize(n);
    centroids[..dim].copy_from_slice(vectors[first]);
    let mut min_dist_sq = vec![f32::INFINITY; n];

    for c in 1..k {
        let prev_off = (c - 1) * dim;
        let mut total_w = 0.0_f64;
        for (i, v) in vectors.iter().enumerate() {
            let d = l2_squared(v, &centroids[prev_off..prev_off + dim]);
            if d < min_dist_sq[i] {
                min_dist_sq[i] = d;
            }
            total_w += f64::from(min_dist_sq[i]);
        }
        let chosen = if total_w <= 1e-12 {
            rng.next_usize(n)
        } else {
            let target = rng.next_f64() * total_w;
            let mut acc = 0.0_f64;
            let mut pick = n - 1;
            for (i, &w) in min_dist_sq.iter().enumerate() {
                acc += f64::from(w);
                if acc >= target {
                    pick = i;
                    break;
                }
            }
            pick
        };
        let off = c * dim;
        centroids[off..off + dim].copy_from_slice(vectors[chosen]);
    }

    let mut assignments = vec![0usize; n];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0_f32; k * dim];

    for _ in 0..iters.max(1) {
        counts.fill(0);
        sums.fill(0.0);

        for (i, v) in vectors.iter().enumerate() {
            let mut best_c = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let off = c * dim;
                let d = l2_squared(v, &centroids[off..off + dim]);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
            counts[best_c] += 1;
            let s_off = best_c * dim;
            for d in 0..dim {
                sums[s_off + d] += v[d];
            }
        }

        for (c, &cnt) in counts.iter().enumerate() {
            if cnt > 0 {
                let inv = 1.0 / (cnt as f32);
                let off = c * dim;
                for d in 0..dim {
                    centroids[off + d] = sums[off + d] * inv;
                }
            }
        }
    }

    (centroids, assignments)
}
