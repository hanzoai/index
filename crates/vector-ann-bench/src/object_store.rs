// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Instrumented immutable ObjectStore abstraction modeling Hanzo S3/VFS storage.
//!
//! Tracks exact `GET` and `RangeGET` request counts, sequential vs batched
//! round-trips (RTTs), total bytes transferred, local compute-side LRU cache
//! hits/misses, and deterministic I/O latency + dollar cost accounting.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::canonical::AnnError;

/// Cloud object-storage + local compute latency and pricing parameters.
///
/// Default values reflect public AWS S3 Standard & cloud compute reference rates
/// (explicitly documented in the HIP-1347 research dossier).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectStoreCostModel {
    /// First-byte round-trip latency per sequential object-storage batch (ms).
    /// S3 Standard typical median ~18 ms; S3 Express / local VFS ~3-5 ms.
    pub s3_rtt_ms: f64,
    /// Per-request tail jitter factor applied to p99 modeling (deterministic multiplier).
    pub s3_bandwidth_mb_per_sec: f64,
    /// Cost per 1,000 GET / RangeGET requests in USD (AWS S3 Standard: $0.0004).
    pub s3_get_cost_per_1000_usd: f64,
    /// Object storage capacity cost per GB-month in USD (AWS S3 Standard: $0.023).
    pub s3_storage_cost_per_gb_month_usd: f64,
    /// Local NVMe SSD storage cost per GB-month in USD (reference cloud NVMe: $0.12).
    pub nvme_storage_cost_per_gb_month_usd: f64,
    /// Compute DRAM cost per GB-month in USD (reference cloud compute RAM: $4.50).
    pub dram_cost_per_gb_month_usd: f64,
    /// Local NVMe 4KB cold page read latency in microseconds (reference NVMe: 85 us).
    pub local_nvme_page_read_us: f64,
}

impl Default for ObjectStoreCostModel {
    fn default() -> Self {
        Self {
            s3_rtt_ms: 15.0,
            s3_bandwidth_mb_per_sec: 400.0,
            s3_get_cost_per_1000_usd: 0.0004,
            s3_storage_cost_per_gb_month_usd: 0.023,
            nvme_storage_cost_per_gb_month_usd: 0.12,
            dram_cost_per_gb_month_usd: 4.50,
            local_nvme_page_read_us: 85.0,
        }
    }
}

/// Point-in-time I/O telemetry counters for [`InstrumentedObjectStore`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObjectStoreStats {
    pub get_requests: u64,
    pub range_get_requests: u64,
    /// Sequential network round trips (a parallel batch of `k` RangeGETs counts as `1` RTT;
    /// `k` sequential pointer-chasing graph hops count as `k` RTTs).
    pub sequential_rtts: u64,
    pub bytes_fetched: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub simulated_io_us: u64,
}

impl ObjectStoreStats {
    pub fn total_requests(&self) -> u64 {
        self.get_requests.saturating_add(self.range_get_requests)
    }

    pub fn delta_since(&self, earlier: ObjectStoreStats) -> ObjectStoreStats {
        ObjectStoreStats {
            get_requests: self.get_requests.saturating_sub(earlier.get_requests),
            range_get_requests: self.range_get_requests.saturating_sub(earlier.range_get_requests),
            sequential_rtts: self.sequential_rtts.saturating_sub(earlier.sequential_rtts),
            bytes_fetched: self.bytes_fetched.saturating_sub(earlier.bytes_fetched),
            cache_hits: self.cache_hits.saturating_sub(earlier.cache_hits),
            cache_misses: self.cache_misses.saturating_sub(earlier.cache_misses),
            simulated_io_us: self.simulated_io_us.saturating_sub(earlier.simulated_io_us),
        }
    }

    pub fn simulated_io_ms(&self) -> f64 {
        self.simulated_io_us as f64 / 1_000.0
    }
}

/// Byte-range descriptor for batched object-storage reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteRangeRequest {
    pub key: String,
    pub offset: usize,
    pub len: usize,
}

/// Simple bounded byte-capacity LRU cache modeling compute-local NVMe/RAM block cache.
#[derive(Debug, Default)]
struct LocalBlockCache {
    max_bytes: usize,
    used_bytes: usize,
    entries: HashMap<(String, usize, usize), Vec<u8>>,
    order: VecDeque<(String, usize, usize)>,
}

impl LocalBlockCache {
    fn new(max_bytes: usize) -> Self {
        Self { max_bytes, used_bytes: 0, entries: HashMap::new(), order: VecDeque::new() }
    }

    fn get(&mut self, key: &(String, usize, usize)) -> Option<Vec<u8>> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: (String, usize, usize), data: Vec<u8>) {
        if self.max_bytes == 0 || data.len() > self.max_bytes {
            return;
        }
        if self.entries.contains_key(&key) {
            return;
        }
        while self.used_bytes + data.len() > self.max_bytes {
            if let Some(evicted_key) = self.order.pop_front() {
                if let Some(evicted_val) = self.entries.remove(&evicted_key) {
                    self.used_bytes = self.used_bytes.saturating_sub(evicted_val.len());
                }
            } else {
                break;
            }
        }
        self.used_bytes += data.len();
        self.order.push_back(key.clone());
        self.entries.insert(key, data);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.used_bytes = 0;
    }
}

/// Immutable S3/VFS-like object store with exact request, byte, RTT, and cache instrumentation.
#[derive(Debug)]
pub struct InstrumentedObjectStore {
    objects: Mutex<BTreeMap<String, Vec<u8>>>,
    cache: Mutex<LocalBlockCache>,
    cost_model: ObjectStoreCostModel,
    get_requests: AtomicU64,
    range_get_requests: AtomicU64,
    sequential_rtts: AtomicU64,
    bytes_fetched: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    simulated_io_us: AtomicU64,
}

impl InstrumentedObjectStore {
    pub fn new(cost_model: ObjectStoreCostModel, local_cache_bytes: usize) -> Self {
        Self {
            objects: Mutex::new(BTreeMap::new()),
            cache: Mutex::new(LocalBlockCache::new(local_cache_bytes)),
            cost_model,
            get_requests: AtomicU64::new(0),
            range_get_requests: AtomicU64::new(0),
            sequential_rtts: AtomicU64::new(0),
            bytes_fetched: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            simulated_io_us: AtomicU64::new(0),
        }
    }

    pub fn cost_model(&self) -> ObjectStoreCostModel {
        self.cost_model
    }

    /// Write an immutable object (errors if key already exists to enforce immutability).
    pub fn put_immutable(&self, key: impl Into<String>, payload: Vec<u8>) -> Result<(), AnnError> {
        let key = key.into();
        let mut guard =
            self.objects.lock().map_err(|_| AnnError::Io("object store lock poisoned".into()))?;
        if guard.contains_key(&key) {
            return Err(AnnError::Io(format!("immutable object key already exists: {key}")));
        }
        guard.insert(key, payload);
        Ok(())
    }

    /// Delete all stored objects and clear local cache (used during drop/rebuild verification).
    pub fn clear_all(&self) -> Result<(), AnnError> {
        let mut guard =
            self.objects.lock().map_err(|_| AnnError::Io("object store lock poisoned".into()))?;
        guard.clear();
        drop(guard);
        self.clear_cache()?;
        Ok(())
    }

    /// Evict the compute-local block cache to force a cold-cache regime.
    pub fn clear_cache(&self) -> Result<(), AnnError> {
        let mut cache =
            self.cache.lock().map_err(|_| AnnError::Io("cache lock poisoned".into()))?;
        cache.clear();
        Ok(())
    }

    /// Total immutable bytes currently stored across all objects.
    pub fn total_stored_bytes(&self) -> usize {
        self.objects.lock().map(|g| g.values().map(|v| v.len()).sum()).unwrap_or(0)
    }

    /// Total bytes currently resident in the compute-local cache.
    pub fn cached_bytes(&self) -> usize {
        self.cache.lock().map(|c| c.used_bytes).unwrap_or(0)
    }

    /// Snapshot current I/O statistics.
    pub fn stats(&self) -> ObjectStoreStats {
        ObjectStoreStats {
            get_requests: self.get_requests.load(Ordering::Relaxed),
            range_get_requests: self.range_get_requests.load(Ordering::Relaxed),
            sequential_rtts: self.sequential_rtts.load(Ordering::Relaxed),
            bytes_fetched: self.bytes_fetched.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            simulated_io_us: self.simulated_io_us.load(Ordering::Relaxed),
        }
    }

    /// Execute a full object `GET` (counts as 1 sequential RTT on cache miss).
    pub fn get(&self, key: &str) -> Result<Vec<u8>, AnnError> {
        let cache_key = (key.to_string(), 0usize, usize::MAX);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(hit) = cache.get(&cache_key) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(hit);
            }
        }
        let data = {
            let guard = self
                .objects
                .lock()
                .map_err(|_| AnnError::Io("object store lock poisoned".into()))?;
            guard
                .get(key)
                .cloned()
                .ok_or_else(|| AnnError::Io(format!("object not found: {key}")))?
        };
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.get_requests.fetch_add(1, Ordering::Relaxed);
        self.sequential_rtts.fetch_add(1, Ordering::Relaxed);
        self.bytes_fetched.fetch_add(data.len() as u64, Ordering::Relaxed);
        let transfer_us = self.transfer_time_us(data.len());
        let rtt_us = (self.cost_model.s3_rtt_ms * 1_000.0) as u64;
        self.simulated_io_us.fetch_add(rtt_us.saturating_add(transfer_us), Ordering::Relaxed);
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(cache_key, data.clone());
        }
        Ok(data)
    }

    /// Execute a single `RangeGET` (1 sequential RTT on cache miss).
    pub fn get_range(&self, key: &str, offset: usize, len: usize) -> Result<Vec<u8>, AnnError> {
        let mut results =
            self.get_ranges_batch(&[ByteRangeRequest { key: key.to_string(), offset, len }])?;
        Ok(results.pop().unwrap_or_default())
    }

    /// Execute a bounded parallel batch of contiguous `RangeGET`s.
    ///
    /// Cache misses in the same batch issue concurrently over HTTP/2 or parallel S3
    /// connections, paying **1 sequential RTT** (plus aggregate transfer time) instead
    /// of `N` sequential round trips!
    pub fn get_ranges_batch(
        &self,
        requests: &[ByteRangeRequest],
    ) -> Result<Vec<Vec<u8>>, AnnError> {
        let mut outputs: Vec<Option<Vec<u8>>> = vec![None; requests.len()];
        let mut miss_indices = Vec::new();

        {
            let mut cache =
                self.cache.lock().map_err(|_| AnnError::Io("cache lock poisoned".into()))?;
            for (idx, req) in requests.iter().enumerate() {
                let ckey = (req.key.clone(), req.offset, req.len);
                if let Some(hit) = cache.get(&ckey) {
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    outputs[idx] = Some(hit);
                } else {
                    miss_indices.push(idx);
                }
            }
        }

        if !miss_indices.is_empty() {
            let mut batch_bytes = 0usize;
            let mut fetched: Vec<(usize, Vec<u8>)> = Vec::with_capacity(miss_indices.len());
            {
                let guard = self
                    .objects
                    .lock()
                    .map_err(|_| AnnError::Io("object store lock poisoned".into()))?;
                for &idx in &miss_indices {
                    let req = &requests[idx];
                    let obj = guard
                        .get(&req.key)
                        .ok_or_else(|| AnnError::Io(format!("object not found: {}", req.key)))?;
                    let end = req
                        .offset
                        .checked_add(req.len)
                        .ok_or_else(|| AnnError::Io("range offset+len overflow".into()))?;
                    if end > obj.len() {
                        return Err(AnnError::Io(format!(
                            "range {}..{} out of bounds for {} (len {})",
                            req.offset,
                            end,
                            req.key,
                            obj.len()
                        )));
                    }
                    let slice = obj[req.offset..end].to_vec();
                    batch_bytes = batch_bytes.saturating_add(slice.len());
                    fetched.push((idx, slice));
                }
            }

            let miss_count = miss_indices.len() as u64;
            self.cache_misses.fetch_add(miss_count, Ordering::Relaxed);
            self.range_get_requests.fetch_add(miss_count, Ordering::Relaxed);
            // Bounded concurrent batch incurs 1 sequential RTT, plus small concurrency tail overhead.
            self.sequential_rtts.fetch_add(1, Ordering::Relaxed);
            self.bytes_fetched.fetch_add(batch_bytes as u64, Ordering::Relaxed);
            let rtt_us = (self.cost_model.s3_rtt_ms * 1_000.0) as u64;
            // Small per-concurrent-stream fanout tail (120 us per extra concurrent RangeGET).
            let fanout_us = miss_count.saturating_sub(1).saturating_mul(120);
            let transfer_us = self.transfer_time_us(batch_bytes);
            self.simulated_io_us.fetch_add(
                rtt_us.saturating_add(fanout_us).saturating_add(transfer_us),
                Ordering::Relaxed,
            );

            let mut cache =
                self.cache.lock().map_err(|_| AnnError::Io("cache lock poisoned".into()))?;
            for (idx, data) in fetched {
                let req = &requests[idx];
                cache.insert((req.key.clone(), req.offset, req.len), data.clone());
                outputs[idx] = Some(data);
            }
        }

        Ok(outputs.into_iter().map(|o| o.unwrap_or_default()).collect())
    }

    #[inline]
    fn transfer_time_us(&self, bytes: usize) -> u64 {
        let bw_bytes_per_us = (self.cost_model.s3_bandwidth_mb_per_sec * 1_000_000.0) / 1_000_000.0;
        if bw_bytes_per_us <= 0.0 {
            return 0;
        }
        (bytes as f64 / bw_bytes_per_us).ceil() as u64
    }
}
