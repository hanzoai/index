// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Architecture 4 (Explicit Negative Control: `object_store_random_graph_hops`).
//!
//! Stores the exact same compact Vamana graph binary (`HVAN`) inside
//! [`InstrumentedObjectStore`] and executes greedy graph traversal directly over
//! object storage **without** a hot local graph cache.
//!
//! Because hop `t + 1` cannot know which neighbor node byte-ranges to fetch until
//! hop `t` returns from object storage, every hop incurs a **sequential S3 RTT**
//! plus `R` random `RangeGET`s — quantitatively proving WHY object storage MUST
//! NOT serve random graph hops (`10x-40x` higher p99 latency and `20x+` S3 GET
//! amplification versus [`crate::object_native_ivf_pq::ObjectNativeIvfPqIndex`]).

use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use crate::canonical::{
    cmp_search_hits, AccessContext, AnnError, CanonicalStore, DerivedRecordMeta, IndexFootprint,
    SearchHit,
};
use crate::local_mmap_vamana::{LocalMmapVamanaIndex, VamanaConfig};
use crate::metrics::{validate_query_vector, Sq8Quantizer};
use crate::object_store::{ByteRangeRequest, InstrumentedObjectStore, ObjectStoreStats};

type FetchedNode = (Vec<u8>, Vec<usize>);

#[derive(Debug, Clone, Copy)]
struct HopCand {
    dist: f32,
    node_id: usize,
}

impl PartialEq for HopCand {
    fn eq(&self, other: &Self) -> bool {
        self.dist.to_bits() == other.dist.to_bits() && self.node_id == other.node_id
    }
}
impl Eq for HopCand {}

impl PartialOrd for HopCand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HopCand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.total_cmp(&self.dist).then_with(|| other.node_id.cmp(&self.node_id))
    }
}

/// Negative control: Vamana graph traversal directly over [`InstrumentedObjectStore`].
pub struct ObjectStoreRandomGraphIndex {
    dim: usize,
    node_count: usize,
    max_degree: usize,
    medoid: usize,
    search_beam: usize,
    record_stride: usize,
    nodes_offset: usize,
    object_key: String,
    quantizer: Sq8Quantizer,
    meta: Vec<DerivedRecordMeta>,
    store: Arc<InstrumentedObjectStore>,
}

impl ObjectStoreRandomGraphIndex {
    /// Build by serializing the exact same Vamana graph into `store`.
    pub fn build_from_canonical(
        canonical: &CanonicalStore,
        store: Arc<InstrumentedObjectStore>,
        config: VamanaConfig,
    ) -> Result<Self, AnnError> {
        let local = LocalMmapVamanaIndex::build_from_canonical(canonical, config, None)?;
        let raw_bytes = local.on_disk_bytes()?;
        let object_key = "neg_control/vamana_graph_v1.hvan".to_string();
        store.put_immutable(&object_key, raw_bytes)?;

        let dim = local.dim();
        let nodes_offset = 40 + local.quantizer().serialized_bytes();
        Ok(Self {
            dim,
            node_count: local.node_count(),
            max_degree: config.max_degree,
            medoid: local.medoid(),
            search_beam: config.search_beam,
            record_stride: local.node_disk_stride_bytes(),
            nodes_offset,
            object_key,
            quantizer: local.quantizer().clone(),
            meta: local.meta().to_vec(),
            store,
        })
    }

    pub fn clear_local_cache(&self) -> Result<(), AnnError> {
        self.store.clear_cache()
    }

    pub fn io_stats(&self) -> ObjectStoreStats {
        self.store.stats()
    }

    /// Execute pointer-chasing Vamana graph hops directly against object storage.
    ///
    /// At each hop, expanding `curr` discovers up to `max_degree` unvisited neighbors
    /// whose byte-ranges are fetched in a hop-dependent round trip — inherently
    /// serializing `H` sequential S3 RTTs!
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ctx: &AccessContext,
    ) -> Result<Vec<SearchHit>, AnnError> {
        ctx.validate()?;
        validate_query_vector(query, self.dim)?;
        if k == 0 || self.node_count == 0 {
            return Ok(Vec::new());
        }

        let beam = if ctx.required_tag.is_some() {
            (self.search_beam * 2).max(k * 6)
        } else {
            self.search_beam.max(k * 2)
        };
        // Bound hops so benchmark completes deterministically while faithfully exposing
        // the 15-40+ sequential RTT amplification of graph traversal over S3.
        let max_hops = if ctx.required_tag.is_some() { 32 } else { 20 };

        let mut visited = HashSet::with_capacity(beam * 2);
        let mut frontier = BinaryHeap::new();
        let mut best_beam: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);
        let mut results: Vec<SearchHit> = Vec::new();
        let mut cached_neighbors: Vec<Option<Vec<usize>>> = vec![None; self.node_count];

        // Hop 0: fetch medoid via RangeGET (1 sequential RTT).
        visited.insert(self.medoid);
        let (sq8_0, nbs_0) = self.fetch_single_node(self.medoid)?;
        let d0 = self.quantizer.asymmetric_l2_sq(query, &sq8_0);
        cached_neighbors[self.medoid] = Some(nbs_0);
        frontier.push(HopCand { dist: d0, node_id: self.medoid });
        best_beam.push((d0, self.medoid));
        if self.meta[self.medoid].matches_access(ctx) {
            let m = &self.meta[self.medoid];
            results.push(SearchHit {
                source_id: m.source_id.clone(),
                version: m.version,
                tenant_id: m.tenant_id.clone(),
                acl_group: m.acl_group.clone(),
                distance: d0,
            });
        }

        let mut hops = 0usize;
        while let Some(curr) = frontier.pop() {
            if hops >= max_hops {
                break;
            }
            if best_beam.len() >= beam && curr.dist > best_beam[beam - 1].0 && results.len() >= k {
                break;
            }

            let neighbors = match &cached_neighbors[curr.node_id] {
                Some(nbs) => nbs.clone(),
                None => {
                    let (_, nbs) = self.fetch_single_node(curr.node_id)?;
                    cached_neighbors[curr.node_id] = Some(nbs.clone());
                    nbs
                }
            };

            let unvisited: Vec<usize> =
                neighbors.into_iter().filter(|&nb| visited.insert(nb)).collect();
            if unvisited.is_empty() {
                continue;
            }

            hops += 1;
            // Even when batching the current node's neighbor reads, the NEXT hop cannot
            // be dispatched until THIS hop returns -> 1 mandatory sequential RTT per hop!
            let fetched = self.fetch_nodes_hop(&unvisited)?;
            for (nb, (sq8, nbs)) in unvisited.into_iter().zip(fetched.into_iter()) {
                cached_neighbors[nb] = Some(nbs);
                let dist = self.quantizer.asymmetric_l2_sq(query, &sq8);
                let thresh =
                    if best_beam.len() < beam { f32::INFINITY } else { best_beam[beam - 1].0 };
                if dist < thresh || results.len() < k {
                    frontier.push(HopCand { dist, node_id: nb });
                    let pos = best_beam
                        .binary_search_by(|x| x.0.total_cmp(&dist).then_with(|| x.1.cmp(&nb)))
                        .unwrap_or_else(|e| e);
                    if pos < beam {
                        best_beam.insert(pos, (dist, nb));
                        if best_beam.len() > beam {
                            best_beam.pop();
                        }
                    }
                }
                if self.meta[nb].matches_access(ctx) {
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

        results.sort_by(cmp_search_hits);
        results.truncate(k);
        Ok(results)
    }

    fn fetch_single_node(&self, node_id: usize) -> Result<FetchedNode, AnnError> {
        let mut res = self.fetch_nodes_hop(&[node_id])?;
        Ok(res.pop().unwrap())
    }

    fn fetch_nodes_hop(&self, node_ids: &[usize]) -> Result<Vec<FetchedNode>, AnnError> {
        let reqs: Vec<ByteRangeRequest> = node_ids
            .iter()
            .map(|&id| ByteRangeRequest {
                key: self.object_key.clone(),
                offset: self.nodes_offset + id * self.record_stride,
                len: self.record_stride,
            })
            .collect();
        let payloads = self.store.get_ranges_batch(&reqs)?;
        let mut parsed = Vec::with_capacity(payloads.len());
        for raw in payloads {
            let sq8 = raw[..self.dim].to_vec();
            let deg = (u32::from_le_bytes(raw[self.dim..self.dim + 4].try_into().unwrap())
                as usize)
                .min(self.max_degree);
            let mut nbs = Vec::with_capacity(deg);
            let base = self.dim + 4;
            for i in 0..deg {
                let off = base + i * 4;
                let nb = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as usize;
                if nb < self.node_count {
                    nbs.push(nb);
                }
            }
            parsed.push((sq8, nbs));
        }
        Ok(parsed)
    }

    pub fn footprint(&self) -> IndexFootprint {
        let meta_ram: usize = self.meta.iter().map(|m| m.memory_bytes()).sum();
        let ram = self.quantizer.serialized_bytes()
            + meta_ram
            + self.object_key.capacity()
            + self.store.cached_bytes();
        let obj = self.store.total_stored_bytes();
        IndexFootprint::new(self.node_count, 0, ram, obj)
    }
}
