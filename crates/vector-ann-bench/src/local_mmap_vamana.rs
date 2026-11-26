// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Architecture 2 (`local_mmap_vamana`): DiskANN / USearch-style local compact
//! memory-mappable Vamana graph + SQ8 vector slab index.
//!
//! # Key Properties
//! - Clean-room two-pass Vamana navigable graph (`RobustPrune` with `alpha = 1.2`)
//!   stored in a fixed-stride `mmap`-compatible binary file (`HVAN` v1 + CRC32).
//! - Real file-backed 4KB paged reader (`FileExt::read_at` on Unix) with explicit
//!   cold-cache (`evict_page_cache`) vs warm-cache accounting.
//! - Pre-return tenant/ACL & tag filtering during filtered beam traversal.
//! - FreshDiskANN-style online updates/deletes via O(1) tombstones + fresh delta
//!   overlay + full compaction.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(unix)]
use std::os::unix::fs::FileExt;

use crate::canonical::{
    cmp_search_hits, AccessContext, AnnError, CanonicalStore, CanonicalVectorRecord,
    DerivedRecordMeta, IndexFootprint, SearchHit,
};
use crate::metrics::{crc32_ieee, l2_squared, validate_query_vector, SplitMix64, Sq8Quantizer};

const MAGIC: [u8; 4] = *b"HVAN";
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = 40;
const PAGE_SIZE: usize = 4096;

/// Configuration for [`LocalMmapVamanaIndex`].
#[derive(Debug, Clone, Copy)]
pub struct VamanaConfig {
    /// Max out-degree per node (`R`).
    pub max_degree: usize,
    /// Build-time candidate beam width (`L_build`).
    pub build_beam: usize,
    /// Query-time base beam width (`L_search`).
    pub search_beam: usize,
    /// RobustPrune diversity parameter (`alpha`, typically 1.2).
    pub alpha: f32,
    /// Deterministic RNG seed for initial random graph edges.
    pub seed: u64,
    /// Local NVMe cold 4KB page read latency (microseconds) for I/O modeling.
    pub nvme_page_read_us: f64,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            max_degree: 24,
            build_beam: 48,
            search_beam: 64,
            alpha: 1.2,
            seed: 1347,
            nvme_page_read_us: 85.0,
        }
    }
}

/// Per-query I/O telemetry for local paged/mmap reads.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalPageIoStats {
    pub page_hits: u64,
    pub page_faults: u64,
    pub bytes_read_from_disk: u64,
    pub nodes_visited: u64,
    pub simulated_nvme_us: u64,
}

impl LocalPageIoStats {
    pub fn delta_since(&self, earlier: Self) -> Self {
        Self {
            page_hits: self.page_hits.saturating_sub(earlier.page_hits),
            page_faults: self.page_faults.saturating_sub(earlier.page_faults),
            bytes_read_from_disk: self
                .bytes_read_from_disk
                .saturating_sub(earlier.bytes_read_from_disk),
            nodes_visited: self.nodes_visited.saturating_sub(earlier.nodes_visited),
            simulated_nvme_us: self.simulated_nvme_us.saturating_sub(earlier.simulated_nvme_us),
        }
    }

    pub fn simulated_nvme_ms(&self) -> f64 {
        self.simulated_nvme_us as f64 / 1_000.0
    }
}

/// Compaction telemetry returned by [`LocalMmapVamanaIndex::compact`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompactionReport {
    pub tombstones_reclaimed: usize,
    pub delta_entries_merged: usize,
    pub bytes_before: usize,
    pub bytes_after: usize,
    pub duration_ms: f64,
}

/// Ordered candidate for min-heap / max-heap traversal.
#[derive(Debug, Clone, Copy)]
struct HeapCandidate {
    dist: f32,
    node_id: usize,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist.to_bits() == other.dist.to_bits() && self.node_id == other.node_id
    }
}
impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering so BinaryHeap behaves as a min-heap on distance.
        other.dist.total_cmp(&self.dist).then_with(|| other.node_id.cmp(&self.node_id))
    }
}

#[derive(Debug, Clone)]
struct DeltaEntry {
    meta: DerivedRecordMeta,
    sq8_code: Vec<u8>,
    tombstoned: bool,
}

/// Disk-backed 4KB page-cached reader over the compact `HVAN` binary file.
#[derive(Debug)]
struct PagedBinaryFile {
    path: PathBuf,
    file: File,
    file_len: usize,
    delete_on_drop: bool,
    pages: Mutex<HashMap<usize, Vec<u8>>>,
    page_hits: AtomicU64,
    page_faults: AtomicU64,
    bytes_read_from_disk: AtomicU64,
    nodes_visited: AtomicU64,
    simulated_nvme_us: AtomicU64,
    nvme_page_read_us: f64,
}

impl PagedBinaryFile {
    fn create(
        path: PathBuf,
        bytes: &[u8],
        nvme_page_read_us: f64,
        delete_on_drop: bool,
    ) -> Result<Self, AnnError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| AnnError::Io(format!("failed to create {}: {e}", path.display())))?;
        file.write_all(bytes)
            .map_err(|e| AnnError::Io(format!("failed to write {}: {e}", path.display())))?;
        file.flush()
            .map_err(|e| AnnError::Io(format!("failed to flush {}: {e}", path.display())))?;
        Ok(Self {
            path,
            file,
            file_len: bytes.len(),
            delete_on_drop,
            pages: Mutex::new(HashMap::new()),
            page_hits: AtomicU64::new(0),
            page_faults: AtomicU64::new(0),
            bytes_read_from_disk: AtomicU64::new(0),
            nodes_visited: AtomicU64::new(0),
            simulated_nvme_us: AtomicU64::new(0),
            nvme_page_read_us,
        })
    }

    fn evict_cache(&self) {
        if let Ok(mut pages) = self.pages.lock() {
            pages.clear();
        }
    }

    fn warm_all_pages(&self) -> Result<(), AnnError> {
        let num_pages = self.file_len.div_ceil(PAGE_SIZE);
        for p in 0..num_pages {
            self.load_page(p)?;
        }
        Ok(())
    }

    fn stats(&self) -> LocalPageIoStats {
        LocalPageIoStats {
            page_hits: self.page_hits.load(Ordering::Relaxed),
            page_faults: self.page_faults.load(Ordering::Relaxed),
            bytes_read_from_disk: self.bytes_read_from_disk.load(Ordering::Relaxed),
            nodes_visited: self.nodes_visited.load(Ordering::Relaxed),
            simulated_nvme_us: self.simulated_nvme_us.load(Ordering::Relaxed),
        }
    }

    fn cached_bytes(&self) -> usize {
        self.pages.lock().map(|p| p.values().map(|v| v.len()).sum()).unwrap_or(0)
    }

    fn load_page(&self, page_idx: usize) -> Result<Vec<u8>, AnnError> {
        if let Ok(pages) = self.pages.lock() {
            if let Some(hit) = pages.get(&page_idx) {
                self.page_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(hit.clone());
            }
        }
        let offset = page_idx
            .checked_mul(PAGE_SIZE)
            .ok_or_else(|| AnnError::Io("page offset overflow".into()))?;
        if offset >= self.file_len {
            return Err(AnnError::Io(format!("page {page_idx} out of bounds")));
        }
        let to_read = (self.file_len - offset).min(PAGE_SIZE);
        let mut buf = vec![0u8; to_read];
        self.read_exact_at(&mut buf, offset as u64)?;
        self.page_faults.fetch_add(1, Ordering::Relaxed);
        self.bytes_read_from_disk.fetch_add(to_read as u64, Ordering::Relaxed);
        self.simulated_nvme_us.fetch_add(self.nvme_page_read_us.round() as u64, Ordering::Relaxed);
        if let Ok(mut pages) = self.pages.lock() {
            pages.insert(page_idx, buf.clone());
        }
        Ok(buf)
    }

    #[cfg(unix)]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<(), AnnError> {
        let mut filled = 0usize;
        while filled < buf.len() {
            let n = self
                .file
                .read_at(&mut buf[filled..], offset + filled as u64)
                .map_err(|e| AnnError::Io(format!("read_at failed: {e}")))?;
            if n == 0 {
                return Err(AnnError::Io("unexpected EOF during read_at".into()));
            }
            filled += n;
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<(), AnnError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f =
            self.file.try_clone().map_err(|e| AnnError::Io(format!("file clone failed: {e}")))?;
        f.seek(SeekFrom::Start(offset)).map_err(|e| AnnError::Io(format!("seek failed: {e}")))?;
        f.read_exact(buf).map_err(|e| AnnError::Io(format!("read_exact failed: {e}")))?;
        Ok(())
    }

    /// Read an arbitrary contiguous byte range through the 4KB page cache.
    fn read_range(&self, offset: usize, len: usize) -> Result<Vec<u8>, AnnError> {
        let end =
            offset.checked_add(len).ok_or_else(|| AnnError::Io("read_range overflow".into()))?;
        if end > self.file_len {
            return Err(AnnError::Io(format!(
                "range {offset}..{end} exceeds file len {}",
                self.file_len
            )));
        }
        let mut out = Vec::with_capacity(len);
        let mut pos = offset;
        while pos < end {
            let page_idx = pos / PAGE_SIZE;
            let page_off = pos % PAGE_SIZE;
            let page = self.load_page(page_idx)?;
            let take = (end - pos).min(page.len().saturating_sub(page_off));
            out.extend_from_slice(&page[page_off..page_off + take]);
            pos += take;
        }
        Ok(out)
    }

    fn read_entire_file(&self) -> Result<Vec<u8>, AnnError> {
        let mut buf = vec![0u8; self.file_len];
        self.read_exact_at(&mut buf, 0)?;
        Ok(buf)
    }
}

impl Drop for PagedBinaryFile {
    fn drop(&mut self) {
        if self.delete_on_drop {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Architecture 2: DiskANN / USearch-style compact disk-backed Vamana graph + SQ8 slab.
pub struct LocalMmapVamanaIndex {
    dim: usize,
    config: VamanaConfig,
    quantizer: Sq8Quantizer,
    medoid: usize,
    node_count: usize,
    record_stride: usize,
    nodes_offset: usize,
    file: PagedBinaryFile,
    /// Metadata table for base graph nodes (kept compact in RAM for pre-return ACL checks).
    meta: Vec<DerivedRecordMeta>,
    /// O(1) lookup from `source_id -> base node_id`.
    base_id_to_node: HashMap<String, usize>,
    base_tombstones: Vec<bool>,
    base_tombstone_count: usize,
    /// FreshDiskANN-style in-memory delta overlay for online inserts/updates.
    fresh_delta: Vec<DeltaEntry>,
    delta_id_to_idx: HashMap<String, usize>,
    delta_tombstone_count: usize,
}

impl LocalMmapVamanaIndex {
    /// Build a compact binary Vamana graph strictly from [`CanonicalStore`].
    pub fn build_from_canonical(
        store: &CanonicalStore,
        config: VamanaConfig,
        backing_path: Option<PathBuf>,
    ) -> Result<Self, AnnError> {
        if config.max_degree == 0 || config.build_beam == 0 || config.search_beam == 0 {
            return Err(AnnError::InvalidConfig(
                "Vamana max_degree, build_beam, and search_beam must be > 0".into(),
            ));
        }

        let dim = store.dim();
        let records = store.snapshot_live();
        let n = records.len();
        let vec_refs: Vec<&[f32]> = records.iter().map(|r| r.vector.as_slice()).collect();
        let quantizer = Sq8Quantizer::fit(dim, &vec_refs)?;

        let sq8_codes: Vec<Vec<u8>> = records.iter().map(|r| quantizer.encode(&r.vector)).collect();
        let (medoid, adjacency) = build_vamana_graph(dim, &records, &config);

        let record_stride = dim
            .checked_add(4)
            .and_then(|v| v.checked_add(config.max_degree.checked_mul(4)?))
            .ok_or_else(|| AnnError::InvalidConfig("record_stride overflow".into()))?;
        let nodes_offset = HEADER_SIZE + quantizer.serialized_bytes();

        let bytes = serialize_hvan_binary(
            dim,
            n,
            config.max_degree,
            medoid,
            record_stride,
            nodes_offset,
            &quantizer,
            &sq8_codes,
            &adjacency,
        )?;

        let (path, delete_on_drop) = match backing_path {
            Some(p) => (p, false),
            None => (unique_temp_hvan_path(config.seed), true),
        };
        let file = PagedBinaryFile::create(path, &bytes, config.nvme_page_read_us, delete_on_drop)?;

        let mut meta = Vec::with_capacity(n);
        let mut base_id_to_node = HashMap::with_capacity(n);
        for (i, rec) in records.iter().enumerate() {
            meta.push(DerivedRecordMeta::from_record(rec));
            base_id_to_node.insert(rec.source_id.clone(), i);
        }

        Ok(Self {
            dim,
            config,
            quantizer,
            medoid,
            node_count: n,
            record_stride,
            nodes_offset,
            file,
            meta,
            base_id_to_node,
            base_tombstones: vec![false; n],
            base_tombstone_count: 0,
            fresh_delta: Vec::new(),
            delta_id_to_idx: HashMap::new(),
            delta_tombstone_count: 0,
        })
    }

    /// Validate on-disk binary header magic, version, and CRC32 checksum.
    pub fn verify_on_disk_integrity(&self) -> Result<(), AnnError> {
        let bytes = self.file.read_entire_file()?;
        verify_hvan_binary(&bytes)
    }

    /// Return the exact raw bytes of the backing binary file (for bit-exact rebuild tests).
    pub fn on_disk_bytes(&self) -> Result<Vec<u8>, AnnError> {
        self.file.read_entire_file()
    }

    pub fn backing_path(&self) -> &Path {
        &self.file.path
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn medoid(&self) -> usize {
        self.medoid
    }

    pub fn config(&self) -> VamanaConfig {
        self.config
    }

    pub fn quantizer(&self) -> &Sq8Quantizer {
        &self.quantizer
    }

    pub fn meta(&self) -> &[DerivedRecordMeta] {
        &self.meta
    }

    /// Evict the 4KB local OS page cache to measure cold-cache performance.
    pub fn evict_page_cache(&self) {
        self.file.evict_cache();
    }

    /// Pre-warm all pages into the 4KB page cache (warm-cache regime).
    pub fn warm_page_cache(&self) -> Result<(), AnnError> {
        self.file.warm_all_pages()
    }

    /// Snapshot cumulative local page I/O counters.
    pub fn io_stats(&self) -> LocalPageIoStats {
        self.file.stats()
    }

    /// Drop and rebuild the derived Vamana index bit-exact from [`CanonicalStore`].
    pub fn rebuild_from_canonical(&mut self, store: &CanonicalStore) -> Result<(), AnnError> {
        let rebuilt = Self::build_from_canonical(store, self.config, None)?;
        *self = rebuilt;
        Ok(())
    }

    /// O(1) tombstone deletion of a record by `source_id`.
    pub fn delete(&mut self, source_id: &str) -> bool {
        let mut deleted = false;
        if let Some(&node_id) = self.base_id_to_node.get(source_id) {
            if !self.base_tombstones[node_id] {
                self.base_tombstones[node_id] = true;
                self.base_tombstone_count += 1;
                deleted = true;
            }
        }
        if let Some(&idx) = self.delta_id_to_idx.get(source_id) {
            if !self.fresh_delta[idx].tombstoned {
                self.fresh_delta[idx].tombstoned = true;
                self.delta_tombstone_count += 1;
                deleted = true;
            }
        }
        deleted
    }

    /// FreshDiskANN-style online upsert: tombstones any prior version and appends
    /// the SQ8-encoded record to the fresh delta overlay for immediate visibility.
    pub fn upsert(&mut self, record: &CanonicalVectorRecord) -> Result<(), AnnError> {
        let valid = record.clone().validated(self.dim)?;
        let _ = self.delete(&valid.source_id);
        let code = self.quantizer.encode(&valid.vector);
        let idx = self.fresh_delta.len();
        self.delta_id_to_idx.insert(valid.source_id.clone(), idx);
        self.fresh_delta.push(DeltaEntry {
            meta: DerivedRecordMeta::from_record(&valid),
            sq8_code: code,
            tombstoned: false,
        });
        Ok(())
    }

    /// Compact base graph + fresh delta + tombstones from canonical truth.
    pub fn compact(&mut self, store: &CanonicalStore) -> Result<CompactionReport, AnnError> {
        let start = std::time::Instant::now();
        let bytes_before = self.file.file_len;
        let tombstones_reclaimed = self.base_tombstone_count + self.delta_tombstone_count;
        let delta_entries_merged = self.fresh_delta.iter().filter(|e| !e.tombstoned).count();

        self.rebuild_from_canonical(store)?;

        let bytes_after = self.file.file_len;
        let duration_ms = start.elapsed().as_secs_f64() * 1_000.0;
        Ok(CompactionReport {
            tombstones_reclaimed,
            delta_entries_merged,
            bytes_before,
            bytes_after,
            duration_ms,
        })
    }

    /// Filtered Vamana beam search with mandatory pre-return ACL & tag isolation.
    ///
    /// Keeps graph connectivity by traversing through non-matching hops while ONLY
    /// admitting live, authorized, filter-matching records into the returned heap.
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

        // Adaptively widen search beam when a structured metadata filter is active.
        let beam = if ctx.required_tag.is_some() {
            (self.config.search_beam * 3).max(k * 8)
        } else {
            self.config.search_beam.max(k * 2)
        };

        let mut results: Vec<SearchHit> = Vec::with_capacity(k * 2);

        if self.node_count > 0 {
            let mut visited: HashSet<usize> = HashSet::with_capacity(beam * 2);
            let mut frontier: BinaryHeap<HeapCandidate> = BinaryHeap::new();
            let mut best_traversal: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);

            // Seed beam with medoid plus deterministic anchor nodes matching the filter
            // (Filtered-DiskANN style multi-entry seeding improves low-selectivity recall).
            let mut seed_nodes = vec![self.medoid];
            if ctx.required_tag.is_some() {
                let stride = (self.node_count / 32).max(1);
                let mut added = 0usize;
                let mut i = 0usize;
                while i < self.node_count && added < 8 {
                    if !self.base_tombstones[i]
                        && self.meta[i].matches_access(ctx)
                        && i != self.medoid
                    {
                        seed_nodes.push(i);
                        added += 1;
                    }
                    i += stride;
                }
            }

            for start_node in seed_nodes {
                if !visited.insert(start_node) {
                    continue;
                }
                let (sq8, _) = self.read_node(start_node)?;
                let dist = self.quantizer.asymmetric_l2_sq(query, &sq8);
                frontier.push(HeapCandidate { dist, node_id: start_node });
                insert_sorted_capped(&mut best_traversal, (dist, start_node), beam);
                if !self.base_tombstones[start_node] && self.meta[start_node].matches_access(ctx) {
                    let m = &self.meta[start_node];
                    results.push(SearchHit {
                        source_id: m.source_id.clone(),
                        version: m.version,
                        tenant_id: m.tenant_id.clone(),
                        acl_group: m.acl_group.clone(),
                        distance: dist,
                    });
                }
            }

            let max_expansions = if ctx.required_tag.is_some() {
                (beam * 3).min(self.node_count)
            } else {
                beam.min(self.node_count)
            };
            let mut expansions = 0usize;

            while let Some(curr) = frontier.pop() {
                if expansions >= max_expansions {
                    break;
                }
                // Early termination once frontier exceeds worst element in beam (when enough filtered hits exist).
                if best_traversal.len() >= beam
                    && curr.dist > best_traversal[beam - 1].0
                    && results.len() >= k
                {
                    break;
                }
                expansions += 1;

                let (_, neighbors) = self.read_node(curr.node_id)?;
                for nb in neighbors {
                    if !visited.insert(nb) {
                        continue;
                    }
                    let (sq8, _) = self.read_node(nb)?;
                    let dist = self.quantizer.asymmetric_l2_sq(query, &sq8);
                    let threshold = if best_traversal.len() < beam {
                        f32::INFINITY
                    } else {
                        best_traversal[beam - 1].0
                    };
                    if dist < threshold || results.len() < k {
                        frontier.push(HeapCandidate { dist, node_id: nb });
                        insert_sorted_capped(&mut best_traversal, (dist, nb), beam);
                    }
                    // Pre-return ACL + tombstone + filter check.
                    if !self.base_tombstones[nb] && self.meta[nb].matches_access(ctx) {
                        let m = &self.meta[nb];
                        results.push(SearchHit {
                            source_id: m.source_id.clone(),
                            version: m.version,
                            tenant_id: m.tenant_id.clone(),
                            acl_group: m.acl_group.clone(),
                            distance: dist,
                        });
                    }
                }
            }
        }

        // Scan fresh delta overlay (immediate read-your-writes freshness).
        for entry in &self.fresh_delta {
            if entry.tombstoned {
                continue;
            }
            if !entry.meta.matches_access(ctx) {
                continue;
            }
            let dist = self.quantizer.asymmetric_l2_sq(query, &entry.sq8_code);
            results.push(SearchHit {
                source_id: entry.meta.source_id.clone(),
                version: entry.meta.version,
                tenant_id: entry.meta.tenant_id.clone(),
                acl_group: entry.meta.acl_group.clone(),
                distance: dist,
            });
        }

        results.sort_by(cmp_search_hits);
        results.truncate(k);
        Ok(results)
    }

    /// Blind post-filtered ANN search (negative-regime control for ultra-selective `1%` filters):
    /// fetches only top-`postfilter_cap` unfiltered ANN neighbors then filters by ACL + tag.
    pub fn search_blind_postfilter(
        &self,
        query: &[f32],
        k: usize,
        postfilter_cap: usize,
        ctx: &AccessContext,
    ) -> Result<Vec<SearchHit>, AnnError> {
        ctx.validate()?;
        validate_query_vector(query, self.dim)?;
        if k == 0 || self.node_count == 0 {
            return Ok(Vec::new());
        }

        let beam = postfilter_cap.max(k);
        let mut visited = HashSet::with_capacity(beam * 2);
        let mut frontier = BinaryHeap::new();
        let mut top_unfiltered: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);

        visited.insert(self.medoid);
        let (sq8, _) = self.read_node(self.medoid)?;
        let d0 = self.quantizer.asymmetric_l2_sq(query, &sq8);
        frontier.push(HeapCandidate { dist: d0, node_id: self.medoid });
        top_unfiltered.push((d0, self.medoid));

        let mut expansions = 0usize;
        while let Some(curr) = frontier.pop() {
            if expansions >= beam {
                break;
            }
            expansions += 1;
            let (_, neighbors) = self.read_node(curr.node_id)?;
            for nb in neighbors {
                if !visited.insert(nb) {
                    continue;
                }
                let (nsq8, _) = self.read_node(nb)?;
                let dist = self.quantizer.asymmetric_l2_sq(query, &nsq8);
                let threshold = if top_unfiltered.len() < beam {
                    f32::INFINITY
                } else {
                    top_unfiltered[beam - 1].0
                };
                if dist < threshold {
                    frontier.push(HeapCandidate { dist, node_id: nb });
                    insert_sorted_capped(&mut top_unfiltered, (dist, nb), beam);
                }
            }
        }

        let mut filtered = Vec::new();
        for (dist, node_id) in top_unfiltered {
            if self.base_tombstones[node_id] {
                continue;
            }
            let m = &self.meta[node_id];
            if m.matches_access(ctx) {
                filtered.push(SearchHit {
                    source_id: m.source_id.clone(),
                    version: m.version,
                    tenant_id: m.tenant_id.clone(),
                    acl_group: m.acl_group.clone(),
                    distance: dist,
                });
            }
        }
        filtered.sort_by(cmp_search_hits);
        filtered.truncate(k);
        Ok(filtered)
    }

    /// Read a node's SQ8 vector and neighbor IDs from the paged binary file.
    pub(crate) fn read_node(&self, node_id: usize) -> Result<(Vec<u8>, Vec<usize>), AnnError> {
        if node_id >= self.node_count {
            return Err(AnnError::CorruptFormat(format!(
                "node_id {node_id} out of range (count {})",
                self.node_count
            )));
        }
        let offset = self
            .nodes_offset
            .checked_add(
                node_id
                    .checked_mul(self.record_stride)
                    .ok_or_else(|| AnnError::CorruptFormat("node offset overflow".into()))?,
            )
            .ok_or_else(|| AnnError::CorruptFormat("node offset overflow".into()))?;
        self.file.nodes_visited.fetch_add(1, Ordering::Relaxed);
        let raw = self.file.read_range(offset, self.record_stride)?;
        let sq8 = raw[..self.dim].to_vec();
        let deg = u32::from_le_bytes(raw[self.dim..self.dim + 4].try_into().unwrap()) as usize;
        let deg = deg.min(self.config.max_degree);
        let mut neighbors = Vec::with_capacity(deg);
        let nb_base = self.dim + 4;
        for i in 0..deg {
            let off = nb_base + i * 4;
            let nb = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as usize;
            if nb < self.node_count {
                neighbors.push(nb);
            }
        }
        Ok((sq8, neighbors))
    }

    /// Exact RAM and on-disk footprint measurement.
    pub fn footprint(&self) -> IndexFootprint {
        let live_base = self.node_count.saturating_sub(self.base_tombstone_count);
        let live_delta = self.fresh_delta.len().saturating_sub(self.delta_tombstone_count);
        let live = live_base + live_delta;
        let tombstones = self.base_tombstone_count + self.delta_tombstone_count;

        let meta_ram: usize = self.meta.iter().map(|m| m.memory_bytes()).sum();
        let delta_ram: usize =
            self.fresh_delta.iter().map(|e| e.meta.memory_bytes() + e.sq8_code.len()).sum();
        let cache_ram = self.file.cached_bytes();
        let quantizer_ram = self.quantizer.serialized_bytes();
        let ram_bytes =
            meta_ram + delta_ram + cache_ram + quantizer_ram + self.base_tombstones.len();

        IndexFootprint::new(live, tombstones, ram_bytes, self.file.file_len)
    }

    /// Compact disk-only node stride in bytes (pure vector + adjacency without string metadata).
    pub fn node_disk_stride_bytes(&self) -> usize {
        self.record_stride
    }
}

/// Clean-room two-pass Vamana graph construction (DiskANN RobustPrune).
fn build_vamana_graph(
    dim: usize,
    records: &[CanonicalVectorRecord],
    config: &VamanaConfig,
) -> (usize, Vec<Vec<usize>>) {
    let n = records.len();
    if n == 0 {
        return (0, Vec::new());
    }
    if n == 1 {
        return (0, vec![Vec::new()]);
    }

    // Step 1: Compute medoid (point closest to dataset centroid).
    let mut centroid = vec![0.0_f32; dim];
    for r in records {
        for (c, &v) in centroid.iter_mut().zip(r.vector.iter()) {
            *c += v;
        }
    }
    let inv_n = 1.0 / (n as f32);
    for c in &mut centroid {
        *c *= inv_n;
    }
    let mut medoid = 0usize;
    let mut best_medoid_d = f32::INFINITY;
    for (i, r) in records.iter().enumerate() {
        let d = l2_squared(&r.vector, &centroid);
        if d < best_medoid_d {
            best_medoid_d = d;
            medoid = i;
        }
    }

    // Step 2: Initialize deterministic random R-regular directed graph.
    let mut rng = SplitMix64::new(config.seed ^ 0x5641_4d41_4e41);
    let r_cap = config.max_degree.min(n - 1);
    let mut adj: Vec<Vec<usize>> = vec![Vec::with_capacity(r_cap); n];
    for (i, neighbors) in adj.iter_mut().enumerate() {
        let mut seen = HashSet::with_capacity(r_cap * 2);
        seen.insert(i);
        while neighbors.len() < r_cap {
            let cand = rng.next_usize(n);
            if seen.insert(cand) {
                neighbors.push(cand);
            }
        }
    }

    // Step 3: Two Vamana passes (alpha = 1.0 then alpha = config.alpha).
    let alphas = [1.0_f32, config.alpha];
    let mut order: Vec<usize> = (0..n).collect();
    for (pass_idx, &alpha) in alphas.iter().enumerate() {
        // Deterministic Fisher-Yates permutation per pass.
        let mut perm_rng =
            SplitMix64::new(config.seed.wrapping_add((pass_idx as u64 + 1) * 0x9e37));
        for i in (1..n).rev() {
            let j = perm_rng.next_usize(i + 1);
            order.swap(i, j);
        }

        for &u in &order {
            let visited =
                greedy_search_build(records, &adj, medoid, &records[u].vector, config.build_beam);
            let pruned = robust_prune(records, u, &visited, alpha, r_cap);
            adj[u] = pruned.clone();

            // Add reverse edges and prune if out-degree exceeds R.
            for &v in &pruned {
                if v == u {
                    continue;
                }
                if !adj[v].contains(&u) {
                    if adj[v].len() < r_cap {
                        adj[v].push(u);
                    } else {
                        let mut cands = adj[v].clone();
                        cands.push(u);
                        adj[v] = robust_prune(records, v, &cands, alpha, r_cap);
                    }
                }
            }
        }
    }

    (medoid, adj)
}

fn greedy_search_build(
    records: &[CanonicalVectorRecord],
    adj: &[Vec<usize>],
    medoid: usize,
    query: &[f32],
    beam: usize,
) -> Vec<usize> {
    let mut visited_set = HashSet::with_capacity(beam * 2);
    let mut visited_list = Vec::with_capacity(beam * 2);
    let mut frontier = BinaryHeap::new();
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);

    let d0 = l2_squared(query, &records[medoid].vector);
    visited_set.insert(medoid);
    visited_list.push(medoid);
    frontier.push(HeapCandidate { dist: d0, node_id: medoid });
    best.push((d0, medoid));

    let mut steps = 0usize;
    while let Some(curr) = frontier.pop() {
        if steps >= beam {
            break;
        }
        if best.len() >= beam && curr.dist > best[beam - 1].0 {
            break;
        }
        steps += 1;
        for &nb in &adj[curr.node_id] {
            if !visited_set.insert(nb) {
                continue;
            }
            visited_list.push(nb);
            let d = l2_squared(query, &records[nb].vector);
            let thresh = if best.len() < beam { f32::INFINITY } else { best[beam - 1].0 };
            if d < thresh {
                frontier.push(HeapCandidate { dist: d, node_id: nb });
                insert_sorted_capped(&mut best, (d, nb), beam);
            }
        }
    }
    visited_list
}

fn robust_prune(
    records: &[CanonicalVectorRecord],
    p: usize,
    candidates: &[usize],
    alpha: f32,
    max_degree: usize,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = candidates
        .iter()
        .copied()
        .filter(|&c| c != p)
        .map(|c| (l2_squared(&records[p].vector, &records[c].vector), c))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.dedup_by_key(|x| x.1);

    let alpha_sq = alpha * alpha;
    let mut out: Vec<usize> = Vec::with_capacity(max_degree);
    for (dist_p_cand, cand) in scored {
        if out.len() >= max_degree {
            break;
        }
        let mut dominated = false;
        for &chosen in &out {
            let dist_chosen_cand = l2_squared(&records[chosen].vector, &records[cand].vector);
            if alpha_sq * dist_chosen_cand <= dist_p_cand {
                dominated = true;
                break;
            }
        }
        if !dominated {
            out.push(cand);
        }
    }
    out
}

#[inline]
fn insert_sorted_capped(vec: &mut Vec<(f32, usize)>, item: (f32, usize), cap: usize) {
    let pos = vec
        .binary_search_by(|x| x.0.total_cmp(&item.0).then_with(|| x.1.cmp(&item.1)))
        .unwrap_or_else(|e| e);
    if pos < cap {
        vec.insert(pos, item);
        if vec.len() > cap {
            vec.pop();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn serialize_hvan_binary(
    dim: usize,
    node_count: usize,
    max_degree: usize,
    medoid: usize,
    record_stride: usize,
    nodes_offset: usize,
    quantizer: &Sq8Quantizer,
    sq8_codes: &[Vec<u8>],
    adjacency: &[Vec<usize>],
) -> Result<Vec<u8>, AnnError> {
    let total_len = nodes_offset
        .checked_add(
            node_count
                .checked_mul(record_stride)
                .ok_or_else(|| AnnError::InvalidConfig("binary index size overflow".into()))?,
        )
        .ok_or_else(|| AnnError::InvalidConfig("binary index size overflow".into()))?;

    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(node_count as u32).to_le_bytes());
    buf.extend_from_slice(&(max_degree as u32).to_le_bytes());
    buf.extend_from_slice(&(medoid as u32).to_le_bytes());
    buf.extend_from_slice(&(record_stride as u32).to_le_bytes());
    buf.extend_from_slice(&(nodes_offset as u64).to_le_bytes());
    // Placeholder for CRC32 at bytes 36..40.
    buf.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(buf.len(), HEADER_SIZE);

    quantizer.write_to_buf(&mut buf);
    debug_assert_eq!(buf.len(), nodes_offset);

    for i in 0..node_count {
        buf.extend_from_slice(&sq8_codes[i]);
        let deg = adjacency[i].len().min(max_degree);
        buf.extend_from_slice(&(deg as u32).to_le_bytes());
        for &nb in adjacency[i].iter().take(deg) {
            buf.extend_from_slice(&(nb as u32).to_le_bytes());
        }
        for _ in deg..max_degree {
            buf.extend_from_slice(&u32::MAX.to_le_bytes());
        }
    }
    debug_assert_eq!(buf.len(), total_len);

    // Compute CRC32 over header (bytes 0..36) + body (bytes 40..).
    let mut crc_input = Vec::with_capacity(buf.len() - 4);
    crc_input.extend_from_slice(&buf[0..36]);
    crc_input.extend_from_slice(&buf[40..]);
    let crc = crc32_ieee(&crc_input);
    buf[36..40].copy_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

fn verify_hvan_binary(bytes: &[u8]) -> Result<(), AnnError> {
    if bytes.len() < HEADER_SIZE {
        return Err(AnnError::CorruptFormat("HVAN file shorter than header".into()));
    }
    if bytes[0..4] != MAGIC {
        return Err(AnnError::CorruptFormat("invalid HVAN magic".into()));
    }
    let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if ver != FORMAT_VERSION {
        return Err(AnnError::CorruptFormat(format!("unsupported HVAN version {ver}")));
    }
    let stored_crc = u32::from_le_bytes(bytes[36..40].try_into().unwrap());
    let mut crc_input = Vec::with_capacity(bytes.len() - 4);
    crc_input.extend_from_slice(&bytes[0..36]);
    crc_input.extend_from_slice(&bytes[40..]);
    let computed = crc32_ieee(&crc_input);
    if stored_crc != computed {
        return Err(AnnError::CorruptFormat(format!(
            "HVAN CRC32 mismatch: stored {stored_crc:#x}, computed {computed:#x}"
        )));
    }
    Ok(())
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

fn unique_temp_hvan_path(seed: u64) -> PathBuf {
    let seq = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("hanzo_hvan_{pid}_{seed}_{seq}.bin"))
}
