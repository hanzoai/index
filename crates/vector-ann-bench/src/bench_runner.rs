// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Unified matched-baseline benchmark runner comparing all four reference
//! architectures (`exact_flat_baseline`, `local_mmap_vamana`,
//! `object_native_ivf_pq`, `object_store_random_graph_hops`) plus the
//! ultra-selective `1%` pre-filter fallback under identical corpus, queries,
//! ground truth, and pre-return ACL isolation.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::canonical::{AnnError, CanonicalStore, IndexFootprint, SearchHit};
use crate::corpus::{build_benchmark_corpus, BenchmarkCorpus, BenchmarkQuery, CorpusConfig};
use crate::exact_flat::{ExactFlatBaseline, PrefilterInvertedFallback};
use crate::json::JsonVal;
use crate::local_mmap_vamana::{CompactionReport, LocalMmapVamanaIndex, VamanaConfig};
use crate::metrics::{recall_at_k, summarize_latencies_ms, LatencySummaryMs};
use crate::object_graph_negative_control::ObjectStoreRandomGraphIndex;
use crate::object_native_ivf_pq::{IvfPqConfig, ObjectNativeIvfPqIndex};
use crate::object_store::{InstrumentedObjectStore, ObjectStoreCostModel};
use crate::scaling::{project_scale_matrix, ScaleProjectionRow};

/// Measured benchmark metrics for one `(architecture, selectivity, cache_regime)` slice.
#[derive(Debug, Clone)]
pub struct SliceMetricRow {
    pub architecture: String,
    pub selectivity: String,
    pub cache_regime: String,
    pub is_projected: bool,
    pub query_count: usize,
    pub recall_at_10: f64,
    pub cpu_latency: LatencySummaryMs,
    pub total_latency_with_io: LatencySummaryMs,
    pub avg_s3_gets: f64,
    pub avg_sequential_rtts: f64,
    pub avg_bytes_fetched: f64,
    pub avg_local_page_faults: f64,
    pub acl_leaks: usize,
    pub filter_violations: usize,
    pub stale_hits: usize,
}

/// Measured negative-regime comparison at ultra-selective `1%` filter.
#[derive(Debug, Clone)]
pub struct UltraSelectiveRegimeRow {
    pub strategy: String,
    pub selectivity: String,
    pub recall_at_10: f64,
    pub avg_hits_returned: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub acl_leaks: usize,
}

/// Measured update, delete, and compaction lifecycle metrics.
#[derive(Debug, Clone)]
pub struct MutationLifecycleReport {
    pub architecture: String,
    pub inserts_applied: usize,
    pub updates_applied: usize,
    pub deletes_applied: usize,
    pub avg_upsert_us: f64,
    pub avg_delete_us: f64,
    pub post_mutation_recall_at_10: f64,
    pub post_mutation_acl_leaks: usize,
    pub post_mutation_stale_or_deleted_hits: usize,
    pub compaction: CompactionReport,
    pub post_compaction_recall_at_10: f64,
}

/// Measured downstream agent file-resolution task accuracy grounded in real repo files.
#[derive(Debug, Clone)]
pub struct DownstreamTaskRow {
    pub architecture: String,
    pub task_count: usize,
    pub recall_at_10: f64,
    pub file_resolution_success_at_1: f64,
    pub file_resolution_success_at_5: f64,
    pub p99_ms_warm: f64,
    pub acl_leaks: usize,
}

/// Drop & rebuild parity audit verifying strict Canonical/Derived separation.
#[derive(Debug, Clone)]
pub struct RebuildParityReport {
    pub canonical_fingerprint_before: u64,
    pub canonical_fingerprint_after: u64,
    pub canonical_unchanged: bool,
    pub vamana_binary_bit_exact: bool,
    pub vamana_hits_bit_exact: bool,
    pub ivf_pq_hits_bit_exact: bool,
    pub exact_flat_hits_bit_exact: bool,
}

/// Full benchmark receipt written to `results.json`.
#[derive(Debug, Clone)]
pub struct FullBenchmarkReport {
    pub seed: u64,
    pub dim: usize,
    pub total_vectors: usize,
    pub repo_files_scanned: usize,
    pub repo_chunks_loaded: usize,
    pub cost_model: ObjectStoreCostModel,
    pub build_times_ms: Vec<(String, f64)>,
    pub footprints: Vec<(String, IndexFootprint)>,
    pub measured_slices: Vec<SliceMetricRow>,
    pub ultra_selective_1pct: Vec<UltraSelectiveRegimeRow>,
    pub mutations: Vec<MutationLifecycleReport>,
    pub downstream_tasks: Vec<DownstreamTaskRow>,
    pub rebuild_parity: RebuildParityReport,
    pub projected_scaling_768d: Vec<ScaleProjectionRow>,
    pub projected_scaling_1536d: Vec<ScaleProjectionRow>,
}

/// Execute the full matched-baseline benchmark suite.
pub fn run_full_benchmark(
    repo_root: Option<&Path>,
    corpus_cfg: &CorpusConfig,
    cost_model: ObjectStoreCostModel,
) -> Result<FullBenchmarkReport, AnnError> {
    let corpus = build_benchmark_corpus(repo_root, corpus_cfg)?;
    let vamana_cfg = VamanaConfig {
        max_degree: 24,
        build_beam: 48,
        search_beam: 64,
        alpha: 1.2,
        seed: corpus.seed,
        nvme_page_read_us: cost_model.local_nvme_page_read_us,
    };
    let ivf_cfg = IvfPqConfig {
        num_partitions: 64,
        nprobe: 12,
        pq_subspaces: 8,
        pq_k_sub: 64,
        refine_factor: 8,
        kmeans_iters: 10,
        seed: corpus.seed,
    };

    // Build all 4 derived architectures from the EXACT same CanonicalStore.
    let mut build_times_ms = Vec::new();

    let t0 = Instant::now();
    let exact_flat = ExactFlatBaseline::build_from_canonical(&corpus.store)?;
    build_times_ms.push(("exact_flat_baseline".to_string(), elapsed_ms(t0)));

    let t0 = Instant::now();
    let local_vamana = LocalMmapVamanaIndex::build_from_canonical(&corpus.store, vamana_cfg, None)?;
    local_vamana.verify_on_disk_integrity()?;
    build_times_ms.push(("local_mmap_vamana".to_string(), elapsed_ms(t0)));

    let ivf_store = Arc::new(InstrumentedObjectStore::new(cost_model, 128 * 1024 * 1024));
    let t0 = Instant::now();
    let object_ivf = ObjectNativeIvfPqIndex::build_from_canonical(
        &corpus.store,
        Arc::clone(&ivf_store),
        ivf_cfg,
    )?;
    build_times_ms.push(("object_native_ivf_pq".to_string(), elapsed_ms(t0)));

    let neg_store = Arc::new(InstrumentedObjectStore::new(cost_model, 0));
    let t0 = Instant::now();
    let neg_graph = ObjectStoreRandomGraphIndex::build_from_canonical(
        &corpus.store,
        Arc::clone(&neg_store),
        vamana_cfg,
    )?;
    build_times_ms.push(("object_store_random_graph_hops".to_string(), elapsed_ms(t0)));

    let prefilter_fallback = PrefilterInvertedFallback::build_from_canonical(&corpus.store)?;

    // Measure exact footprints (with cold vs warm cache noted cleanly: measure cold base footprint first).
    local_vamana.evict_page_cache();
    object_ivf.clear_local_cache()?;
    let footprints = vec![
        ("exact_flat_baseline".to_string(), exact_flat.footprint()),
        ("local_mmap_vamana".to_string(), local_vamana.footprint()),
        ("object_native_ivf_pq".to_string(), object_ivf.footprint()),
        ("object_store_random_graph_hops".to_string(), neg_graph.footprint()),
    ];

    // Precompute exact 100% Recall@10 ground truth for all selectivity slices.
    let gt_100 = compute_ground_truth(&exact_flat, &corpus.queries_100, 10)?;
    let gt_25 = compute_ground_truth(&exact_flat, &corpus.queries_25, 10)?;
    let gt_05 = compute_ground_truth(&exact_flat, &corpus.queries_05, 10)?;
    let gt_01 = compute_ground_truth(&exact_flat, &corpus.queries_01, 10)?;

    type SelectivitySlice<'a> = (&'a str, &'a [BenchmarkQuery], &'a [Vec<SearchHit>]);
    let mut measured_slices = Vec::new();
    let slices: &[SelectivitySlice<'_>] = &[
        ("100%", &corpus.queries_100, &gt_100),
        ("25%", &corpus.queries_25, &gt_25),
        ("5%", &corpus.queries_05, &gt_05),
    ];

    for &(sel, queries, gt) in slices {
        // 1. exact_flat_baseline (in-RAM)
        measured_slices.push(eval_exact_flat(&corpus.store, &exact_flat, queries, gt, sel)?);

        // 2. local_mmap_vamana (cold cache + warm cache)
        measured_slices.push(eval_local_vamana(
            &corpus.store,
            &local_vamana,
            queries,
            gt,
            sel,
            true,
        )?);
        measured_slices.push(eval_local_vamana(
            &corpus.store,
            &local_vamana,
            queries,
            gt,
            sel,
            false,
        )?);

        // 3. object_native_ivf_pq (cold cache + warm cache)
        measured_slices.push(eval_object_ivf(&corpus.store, &object_ivf, queries, gt, sel, true)?);
        measured_slices.push(eval_object_ivf(&corpus.store, &object_ivf, queries, gt, sel, false)?);

        // 4. object_store_random_graph_hops (cold cache negative control)
        measured_slices.push(eval_neg_graph(&corpus.store, &neg_graph, queries, gt, sel)?);
    }

    // Explicit Negative Regime: Ultra-selective 1% filter comparing blind post-filtered ANN
    // against filtered ANN and inverted-posting pre-filter fallback.
    local_vamana.warm_page_cache()?;
    let ultra_selective_1pct = eval_ultra_selective_regime(
        &corpus.store,
        &local_vamana,
        &object_ivf,
        &prefilter_fallback,
        &corpus.queries_01,
        &gt_01,
    )?;

    // Downstream agent file-resolution task accuracy grounded in real repo source chunks.
    let downstream_tasks = eval_downstream_tasks(
        &corpus.store,
        &exact_flat,
        &local_vamana,
        &object_ivf,
        &corpus.downstream_tasks,
    )?;

    // Drop & Rebuild Bit-Exact Verification BEFORE mutation test.
    let rebuild_parity = verify_drop_and_rebuild_parity(
        &corpus,
        &exact_flat,
        &local_vamana,
        &object_ivf,
        vamana_cfg,
        ivf_cfg,
        cost_model,
    )?;

    // Online Update, Tombstone Delete & Compaction Lifecycle Benchmark.
    let mutations = run_mutation_lifecycle_benchmark(&corpus, vamana_cfg, ivf_cfg, cost_model)?;

    // Analytical 1M / 10M / 100M scaling projections grounded in measured negative-control hops/GETs.
    let neg_100 = measured_slices
        .iter()
        .find(|r| r.architecture == "object_store_random_graph_hops" && r.selectivity == "100%");
    let neg_hops = neg_100.map(|r| r.avg_sequential_rtts).unwrap_or(21.0);
    let neg_gets = neg_100.map(|r| r.avg_s3_gets).unwrap_or(240.0);

    let projected_scaling_768d = project_scale_matrix(768, cost_model, neg_hops, neg_gets);
    let projected_scaling_1536d = project_scale_matrix(1536, cost_model, neg_hops, neg_gets);

    Ok(FullBenchmarkReport {
        seed: corpus.seed,
        dim: corpus.dim,
        total_vectors: corpus.total_vectors,
        repo_files_scanned: corpus.repo_files_scanned,
        repo_chunks_loaded: corpus.repo_chunks_loaded,
        cost_model,
        build_times_ms,
        footprints,
        measured_slices,
        ultra_selective_1pct,
        mutations,
        downstream_tasks,
        rebuild_parity,
        projected_scaling_768d,
        projected_scaling_1536d,
    })
}

fn compute_ground_truth(
    exact: &ExactFlatBaseline,
    queries: &[BenchmarkQuery],
    k: usize,
) -> Result<Vec<Vec<SearchHit>>, AnnError> {
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        out.push(exact.search(&q.vector, k, &q.access)?);
    }
    Ok(out)
}

fn eval_exact_flat(
    store: &CanonicalStore,
    index: &ExactFlatBaseline,
    queries: &[BenchmarkQuery],
    gt: &[Vec<SearchHit>],
    sel: &str,
) -> Result<SliceMetricRow, AnnError> {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut cpu_ms = Vec::with_capacity(queries.len());
    let mut acl_leaks = 0;
    let mut filter_violations = 0;
    let mut stale_hits = 0;

    for (q, truth) in queries.iter().zip(gt.iter()) {
        let t0 = Instant::now();
        let hits = index.search(&q.vector, 10, &q.access)?;
        cpu_ms.push(elapsed_ms(t0));
        recalls.push(recall_at_k(truth, &hits, 10));
        let audit = store.audit_hits(&q.access, &hits)?;
        acl_leaks += audit.acl_leaks;
        filter_violations += audit.filter_violations;
        stale_hits += audit.stale_or_deleted_hits;
    }

    let cpu_summary = summarize_latencies_ms(&cpu_ms);
    Ok(SliceMetricRow {
        architecture: "exact_flat_baseline".to_string(),
        selectivity: sel.to_string(),
        cache_regime: "warm-ram".to_string(),
        is_projected: false,
        query_count: queries.len(),
        recall_at_10: mean(&recalls),
        cpu_latency: cpu_summary,
        total_latency_with_io: cpu_summary,
        avg_s3_gets: 0.0,
        avg_sequential_rtts: 0.0,
        avg_bytes_fetched: 0.0,
        avg_local_page_faults: 0.0,
        acl_leaks,
        filter_violations,
        stale_hits,
    })
}

fn eval_local_vamana(
    store: &CanonicalStore,
    index: &LocalMmapVamanaIndex,
    queries: &[BenchmarkQuery],
    gt: &[Vec<SearchHit>],
    sel: &str,
    cold: bool,
) -> Result<SliceMetricRow, AnnError> {
    if !cold {
        index.warm_page_cache()?;
    }

    let mut recalls = Vec::with_capacity(queries.len());
    let mut cpu_ms = Vec::with_capacity(queries.len());
    let mut total_ms = Vec::with_capacity(queries.len());
    let mut page_faults_sum = 0u64;
    let mut bytes_sum = 0u64;
    let mut acl_leaks = 0;
    let mut filter_violations = 0;
    let mut stale_hits = 0;

    for (q, truth) in queries.iter().zip(gt.iter()) {
        if cold {
            index.evict_page_cache();
        }
        let before = index.io_stats();
        let t0 = Instant::now();
        let hits = index.search(&q.vector, 10, &q.access)?;
        let dt_ms = elapsed_ms(t0);
        let delta = index.io_stats().delta_since(before);

        cpu_ms.push(dt_ms);
        total_ms.push(dt_ms + delta.simulated_nvme_ms());
        page_faults_sum += delta.page_faults;
        bytes_sum += delta.bytes_read_from_disk;
        recalls.push(recall_at_k(truth, &hits, 10));

        let audit = store.audit_hits(&q.access, &hits)?;
        acl_leaks += audit.acl_leaks;
        filter_violations += audit.filter_violations;
        stale_hits += audit.stale_or_deleted_hits;
    }

    let n = queries.len().max(1) as f64;
    Ok(SliceMetricRow {
        architecture: "local_mmap_vamana".to_string(),
        selectivity: sel.to_string(),
        cache_regime: if cold { "cold-cache" } else { "warm-cache" }.to_string(),
        is_projected: false,
        query_count: queries.len(),
        recall_at_10: mean(&recalls),
        cpu_latency: summarize_latencies_ms(&cpu_ms),
        total_latency_with_io: summarize_latencies_ms(&total_ms),
        avg_s3_gets: 0.0,
        avg_sequential_rtts: 0.0,
        avg_bytes_fetched: bytes_sum as f64 / n,
        avg_local_page_faults: page_faults_sum as f64 / n,
        acl_leaks,
        filter_violations,
        stale_hits,
    })
}

fn eval_object_ivf(
    store: &CanonicalStore,
    index: &ObjectNativeIvfPqIndex,
    queries: &[BenchmarkQuery],
    gt: &[Vec<SearchHit>],
    sel: &str,
    cold: bool,
) -> Result<SliceMetricRow, AnnError> {
    if !cold {
        // Warm up the compute-local partition cache with a dry pass.
        for q in queries {
            let _ = index.search(&q.vector, 10, &q.access)?;
        }
    }

    let mut recalls = Vec::with_capacity(queries.len());
    let mut cpu_ms = Vec::with_capacity(queries.len());
    let mut total_ms = Vec::with_capacity(queries.len());
    let mut gets_sum = 0u64;
    let mut rtts_sum = 0u64;
    let mut bytes_sum = 0u64;
    let mut acl_leaks = 0;
    let mut filter_violations = 0;
    let mut stale_hits = 0;

    for (q, truth) in queries.iter().zip(gt.iter()) {
        if cold {
            index.clear_local_cache()?;
        }
        let before = index.io_stats();
        let t0 = Instant::now();
        let hits = index.search(&q.vector, 10, &q.access)?;
        let dt_ms = elapsed_ms(t0);
        let delta = index.io_stats().delta_since(before);

        cpu_ms.push(dt_ms);
        total_ms.push(dt_ms + delta.simulated_io_ms());
        gets_sum += delta.total_requests();
        rtts_sum += delta.sequential_rtts;
        bytes_sum += delta.bytes_fetched;
        recalls.push(recall_at_k(truth, &hits, 10));

        let audit = store.audit_hits(&q.access, &hits)?;
        acl_leaks += audit.acl_leaks;
        filter_violations += audit.filter_violations;
        stale_hits += audit.stale_or_deleted_hits;
    }

    let n = queries.len().max(1) as f64;
    Ok(SliceMetricRow {
        architecture: "object_native_ivf_pq".to_string(),
        selectivity: sel.to_string(),
        cache_regime: if cold { "cold-cache" } else { "warm-cache" }.to_string(),
        is_projected: false,
        query_count: queries.len(),
        recall_at_10: mean(&recalls),
        cpu_latency: summarize_latencies_ms(&cpu_ms),
        total_latency_with_io: summarize_latencies_ms(&total_ms),
        avg_s3_gets: gets_sum as f64 / n,
        avg_sequential_rtts: rtts_sum as f64 / n,
        avg_bytes_fetched: bytes_sum as f64 / n,
        avg_local_page_faults: 0.0,
        acl_leaks,
        filter_violations,
        stale_hits,
    })
}

fn eval_neg_graph(
    store: &CanonicalStore,
    index: &ObjectStoreRandomGraphIndex,
    queries: &[BenchmarkQuery],
    gt: &[Vec<SearchHit>],
    sel: &str,
) -> Result<SliceMetricRow, AnnError> {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut cpu_ms = Vec::with_capacity(queries.len());
    let mut total_ms = Vec::with_capacity(queries.len());
    let mut gets_sum = 0u64;
    let mut rtts_sum = 0u64;
    let mut bytes_sum = 0u64;
    let mut acl_leaks = 0;
    let mut filter_violations = 0;
    let mut stale_hits = 0;

    for (q, truth) in queries.iter().zip(gt.iter()) {
        index.clear_local_cache()?;
        let before = index.io_stats();
        let t0 = Instant::now();
        let hits = index.search(&q.vector, 10, &q.access)?;
        let dt_ms = elapsed_ms(t0);
        let delta = index.io_stats().delta_since(before);

        cpu_ms.push(dt_ms);
        total_ms.push(dt_ms + delta.simulated_io_ms());
        gets_sum += delta.total_requests();
        rtts_sum += delta.sequential_rtts;
        bytes_sum += delta.bytes_fetched;
        recalls.push(recall_at_k(truth, &hits, 10));

        let audit = store.audit_hits(&q.access, &hits)?;
        acl_leaks += audit.acl_leaks;
        filter_violations += audit.filter_violations;
        stale_hits += audit.stale_or_deleted_hits;
    }

    let n = queries.len().max(1) as f64;
    Ok(SliceMetricRow {
        architecture: "object_store_random_graph_hops".to_string(),
        selectivity: sel.to_string(),
        cache_regime: "cold-cache".to_string(),
        is_projected: false,
        query_count: queries.len(),
        recall_at_10: mean(&recalls),
        cpu_latency: summarize_latencies_ms(&cpu_ms),
        total_latency_with_io: summarize_latencies_ms(&total_ms),
        avg_s3_gets: gets_sum as f64 / n,
        avg_sequential_rtts: rtts_sum as f64 / n,
        avg_bytes_fetched: bytes_sum as f64 / n,
        avg_local_page_faults: 0.0,
        acl_leaks,
        filter_violations,
        stale_hits,
    })
}

fn eval_ultra_selective_regime(
    store: &CanonicalStore,
    vamana: &LocalMmapVamanaIndex,
    ivf: &ObjectNativeIvfPqIndex,
    prefilter: &PrefilterInvertedFallback,
    queries: &[BenchmarkQuery],
    gt: &[Vec<SearchHit>],
) -> Result<Vec<UltraSelectiveRegimeRow>, AnnError> {
    let mut rows = Vec::new();

    // Strategy A: Blind post-filtered Vamana (top-32 unfiltered ANN -> filter 1% tag)
    {
        let mut recalls = Vec::new();
        let mut hit_counts = Vec::new();
        let mut lats = Vec::new();
        let mut leaks = 0;
        for (q, truth) in queries.iter().zip(gt.iter()) {
            let t0 = Instant::now();
            let hits = vamana.search_blind_postfilter(&q.vector, 10, 32, &q.access)?;
            lats.push(elapsed_ms(t0));
            hit_counts.push(hits.len() as f64);
            recalls.push(recall_at_k(truth, &hits, 10));
            leaks += store.audit_hits(&q.access, &hits)?.acl_leaks;
        }
        let s = summarize_latencies_ms(&lats);
        rows.push(UltraSelectiveRegimeRow {
            strategy: "vamana_blind_postfilter_top32".to_string(),
            selectivity: "1%".to_string(),
            recall_at_10: mean(&recalls),
            avg_hits_returned: mean(&hit_counts),
            p50_ms: s.p50_ms,
            p99_ms: s.p99_ms,
            acl_leaks: leaks,
        });
    }

    // Strategy B: Blind post-filtered IVF-PQ (top-32 unfiltered PQ -> filter 1% tag)
    {
        let mut recalls = Vec::new();
        let mut hit_counts = Vec::new();
        let mut lats = Vec::new();
        let mut leaks = 0;
        for (q, truth) in queries.iter().zip(gt.iter()) {
            let t0 = Instant::now();
            let hits = ivf.search_blind_postfilter(&q.vector, 10, 32, &q.access)?;
            lats.push(elapsed_ms(t0));
            hit_counts.push(hits.len() as f64);
            recalls.push(recall_at_k(truth, &hits, 10));
            leaks += store.audit_hits(&q.access, &hits)?.acl_leaks;
        }
        let s = summarize_latencies_ms(&lats);
        rows.push(UltraSelectiveRegimeRow {
            strategy: "ivf_pq_blind_postfilter_top32".to_string(),
            selectivity: "1%".to_string(),
            recall_at_10: mean(&recalls),
            avg_hits_returned: mean(&hit_counts),
            p50_ms: s.p50_ms,
            p99_ms: s.p99_ms,
            acl_leaks: leaks,
        });
    }

    // Strategy C: Filtered Vamana (widened beam + filter-aware traversal)
    {
        let mut recalls = Vec::new();
        let mut hit_counts = Vec::new();
        let mut lats = Vec::new();
        let mut leaks = 0;
        for (q, truth) in queries.iter().zip(gt.iter()) {
            let t0 = Instant::now();
            let hits = vamana.search(&q.vector, 10, &q.access)?;
            lats.push(elapsed_ms(t0));
            hit_counts.push(hits.len() as f64);
            recalls.push(recall_at_k(truth, &hits, 10));
            leaks += store.audit_hits(&q.access, &hits)?.acl_leaks;
        }
        let s = summarize_latencies_ms(&lats);
        rows.push(UltraSelectiveRegimeRow {
            strategy: "local_mmap_vamana_filtered".to_string(),
            selectivity: "1%".to_string(),
            recall_at_10: mean(&recalls),
            avg_hits_returned: mean(&hit_counts),
            p50_ms: s.p50_ms,
            p99_ms: s.p99_ms,
            acl_leaks: leaks,
        });
    }

    // Strategy D: Pre-filter inverted-index fallback (RECOMMENDED at <= 1-2% selectivity!)
    {
        let mut recalls = Vec::new();
        let mut hit_counts = Vec::new();
        let mut lats = Vec::new();
        let mut leaks = 0;
        for (q, truth) in queries.iter().zip(gt.iter()) {
            let t0 = Instant::now();
            let (hits, _scanned) = prefilter.search_with_stats(&q.vector, 10, &q.access)?;
            lats.push(elapsed_ms(t0));
            hit_counts.push(hits.len() as f64);
            recalls.push(recall_at_k(truth, &hits, 10));
            leaks += store.audit_hits(&q.access, &hits)?.acl_leaks;
        }
        let s = summarize_latencies_ms(&lats);
        rows.push(UltraSelectiveRegimeRow {
            strategy: "prefilter_inverted_fallback".to_string(),
            selectivity: "1%".to_string(),
            recall_at_10: mean(&recalls),
            avg_hits_returned: mean(&hit_counts),
            p50_ms: s.p50_ms,
            p99_ms: s.p99_ms,
            acl_leaks: leaks,
        });
    }

    Ok(rows)
}

fn eval_downstream_tasks(
    store: &CanonicalStore,
    exact: &ExactFlatBaseline,
    vamana: &LocalMmapVamanaIndex,
    ivf: &ObjectNativeIvfPqIndex,
    tasks: &[BenchmarkQuery],
) -> Result<Vec<DownstreamTaskRow>, AnnError> {
    let gt = compute_ground_truth(exact, tasks, 10)?;
    vamana.warm_page_cache()?;
    // Warm IVF cache for repo tasks.
    for t in tasks {
        let _ = ivf.search(&t.vector, 10, &t.access)?;
    }

    let mut out = Vec::new();
    for arch in ["exact_flat_baseline", "local_mmap_vamana", "object_native_ivf_pq"] {
        let mut recalls = Vec::new();
        let mut succ1 = 0usize;
        let mut succ5 = 0usize;
        let mut lats = Vec::new();
        let mut leaks = 0usize;

        for (task, truth) in tasks.iter().zip(gt.iter()) {
            let t0 = Instant::now();
            let hits = match arch {
                "exact_flat_baseline" => exact.search(&task.vector, 10, &task.access)?,
                "local_mmap_vamana" => vamana.search(&task.vector, 10, &task.access)?,
                _ => ivf.search(&task.vector, 10, &task.access)?,
            };
            lats.push(elapsed_ms(t0));
            recalls.push(recall_at_k(truth, &hits, 10));
            leaks += store.audit_hits(&task.access, &hits)?.acl_leaks;

            if let Some(prefix) = &task.expected_file_prefix {
                if hits.first().map(|h| h.source_id.starts_with(prefix)).unwrap_or(false) {
                    succ1 += 1;
                }
                if hits.iter().take(5).any(|h| h.source_id.starts_with(prefix)) {
                    succ5 += 1;
                }
            }
        }
        let n = tasks.len().max(1) as f64;
        out.push(DownstreamTaskRow {
            architecture: arch.to_string(),
            task_count: tasks.len(),
            recall_at_10: mean(&recalls),
            file_resolution_success_at_1: succ1 as f64 / n,
            file_resolution_success_at_5: succ5 as f64 / n,
            p99_ms_warm: summarize_latencies_ms(&lats).p99_ms,
            acl_leaks: leaks,
        });
    }
    Ok(out)
}

fn verify_drop_and_rebuild_parity(
    corpus: &BenchmarkCorpus,
    exact_orig: &ExactFlatBaseline,
    vamana_orig: &LocalMmapVamanaIndex,
    ivf_orig: &ObjectNativeIvfPqIndex,
    vamana_cfg: VamanaConfig,
    ivf_cfg: IvfPqConfig,
    cost_model: ObjectStoreCostModel,
) -> Result<RebuildParityReport, AnnError> {
    let fp_before = corpus.store.canonical_fingerprint();
    let orig_vamana_bytes = vamana_orig.on_disk_bytes()?;

    let mut exact_rebuilt = ExactFlatBaseline::build_from_canonical(&corpus.store)?;
    exact_rebuilt.rebuild_from_canonical(&corpus.store)?;

    let mut vamana_rebuilt =
        LocalMmapVamanaIndex::build_from_canonical(&corpus.store, vamana_cfg, None)?;
    vamana_rebuilt.rebuild_from_canonical(&corpus.store)?;
    let rebuilt_vamana_bytes = vamana_rebuilt.on_disk_bytes()?;

    let ivf_store_2 = Arc::new(InstrumentedObjectStore::new(cost_model, 64 * 1024 * 1024));
    let mut ivf_rebuilt =
        ObjectNativeIvfPqIndex::build_from_canonical(&corpus.store, ivf_store_2, ivf_cfg)?;
    ivf_rebuilt.rebuild_from_canonical(&corpus.store)?;

    let probe_queries = &corpus.queries_100[..corpus.queries_100.len().min(15)];
    let mut exact_eq = true;
    let mut vamana_eq = true;
    let mut ivf_eq = true;

    for q in probe_queries {
        if exact_orig.search(&q.vector, 10, &q.access)?
            != exact_rebuilt.search(&q.vector, 10, &q.access)?
        {
            exact_eq = false;
        }
        if vamana_orig.search(&q.vector, 10, &q.access)?
            != vamana_rebuilt.search(&q.vector, 10, &q.access)?
        {
            vamana_eq = false;
        }
        if ivf_orig.search(&q.vector, 10, &q.access)?
            != ivf_rebuilt.search(&q.vector, 10, &q.access)?
        {
            ivf_eq = false;
        }
    }

    let fp_after = corpus.store.canonical_fingerprint();
    Ok(RebuildParityReport {
        canonical_fingerprint_before: fp_before,
        canonical_fingerprint_after: fp_after,
        canonical_unchanged: fp_before == fp_after,
        vamana_binary_bit_exact: orig_vamana_bytes == rebuilt_vamana_bytes,
        vamana_hits_bit_exact: vamana_eq,
        ivf_pq_hits_bit_exact: ivf_eq,
        exact_flat_hits_bit_exact: exact_eq,
    })
}

fn run_mutation_lifecycle_benchmark(
    corpus: &BenchmarkCorpus,
    vamana_cfg: VamanaConfig,
    ivf_cfg: IvfPqConfig,
    cost_model: ObjectStoreCostModel,
) -> Result<Vec<MutationLifecycleReport>, AnnError> {
    // Apply mutations to a cloned CanonicalStore to compute exact post-mutation ground truth.
    let mut mutated_store = corpus.store.clone();
    for rec in &corpus.mutation_inserts {
        mutated_store.upsert(rec.clone())?;
    }
    for rec in &corpus.mutation_updates {
        mutated_store.upsert(rec.clone())?;
    }
    for id in &corpus.mutation_delete_ids {
        mutated_store.delete(id);
    }
    let mutated_oracle = ExactFlatBaseline::build_from_canonical(&mutated_store)?;
    let post_gt = compute_ground_truth(&mutated_oracle, &corpus.queries_100, 10)?;

    let mut reports = Vec::new();

    // 1. local_mmap_vamana mutation & compaction
    {
        let mut vamana =
            LocalMmapVamanaIndex::build_from_canonical(&corpus.store, vamana_cfg, None)?;
        vamana.warm_page_cache()?;

        let t_up = Instant::now();
        for rec in &corpus.mutation_inserts {
            vamana.upsert(rec)?;
        }
        for rec in &corpus.mutation_updates {
            vamana.upsert(rec)?;
        }
        let upsert_total_us = t_up.elapsed().as_secs_f64() * 1_000_000.0;
        let total_upserts = corpus.mutation_inserts.len() + corpus.mutation_updates.len();

        let t_del = Instant::now();
        let mut del_ok = 0usize;
        for id in &corpus.mutation_delete_ids {
            if vamana.delete(id) {
                del_ok += 1;
            }
        }
        let del_total_us = t_del.elapsed().as_secs_f64() * 1_000_000.0;

        let (post_recall, leaks, stale) =
            eval_post_mutation_queries(&mutated_store, &corpus.queries_100, &post_gt, |q| {
                vamana.search(&q.vector, 10, &q.access)
            })?;

        let compaction = vamana.compact(&mutated_store)?;
        vamana.warm_page_cache()?;
        let (compact_recall, c_leaks, c_stale) =
            eval_post_mutation_queries(&mutated_store, &corpus.queries_100, &post_gt, |q| {
                vamana.search(&q.vector, 10, &q.access)
            })?;

        reports.push(MutationLifecycleReport {
            architecture: "local_mmap_vamana".to_string(),
            inserts_applied: corpus.mutation_inserts.len(),
            updates_applied: corpus.mutation_updates.len(),
            deletes_applied: del_ok,
            avg_upsert_us: upsert_total_us / total_upserts.max(1) as f64,
            avg_delete_us: del_total_us / del_ok.max(1) as f64,
            post_mutation_recall_at_10: post_recall,
            post_mutation_acl_leaks: leaks + c_leaks,
            post_mutation_stale_or_deleted_hits: stale + c_stale,
            compaction,
            post_compaction_recall_at_10: compact_recall,
        });
    }

    // 2. object_native_ivf_pq immutable delta + tombstone + compaction
    {
        let store = Arc::new(InstrumentedObjectStore::new(cost_model, 128 * 1024 * 1024));
        let mut ivf = ObjectNativeIvfPqIndex::build_from_canonical(&corpus.store, store, ivf_cfg)?;

        let t_up = Instant::now();
        for rec in &corpus.mutation_inserts {
            ivf.upsert(rec)?;
        }
        for rec in &corpus.mutation_updates {
            ivf.upsert(rec)?;
        }
        let upsert_total_us = t_up.elapsed().as_secs_f64() * 1_000_000.0;
        let total_upserts = corpus.mutation_inserts.len() + corpus.mutation_updates.len();

        let t_del = Instant::now();
        let mut del_ok = 0usize;
        for id in &corpus.mutation_delete_ids {
            if ivf.delete(id)? {
                del_ok += 1;
            }
        }
        let del_total_us = t_del.elapsed().as_secs_f64() * 1_000_000.0;

        let (post_recall, leaks, stale) =
            eval_post_mutation_queries(&mutated_store, &corpus.queries_100, &post_gt, |q| {
                ivf.search(&q.vector, 10, &q.access)
            })?;

        let compaction = ivf.compact(&mutated_store)?;
        let (compact_recall, c_leaks, c_stale) =
            eval_post_mutation_queries(&mutated_store, &corpus.queries_100, &post_gt, |q| {
                ivf.search(&q.vector, 10, &q.access)
            })?;

        reports.push(MutationLifecycleReport {
            architecture: "object_native_ivf_pq".to_string(),
            inserts_applied: corpus.mutation_inserts.len(),
            updates_applied: corpus.mutation_updates.len(),
            deletes_applied: del_ok,
            avg_upsert_us: upsert_total_us / total_upserts.max(1) as f64,
            avg_delete_us: del_total_us / del_ok.max(1) as f64,
            post_mutation_recall_at_10: post_recall,
            post_mutation_acl_leaks: leaks + c_leaks,
            post_mutation_stale_or_deleted_hits: stale + c_stale,
            compaction,
            post_compaction_recall_at_10: compact_recall,
        });
    }

    Ok(reports)
}

fn eval_post_mutation_queries<F>(
    store: &CanonicalStore,
    queries: &[BenchmarkQuery],
    gt: &[Vec<SearchHit>],
    search_fn: F,
) -> Result<(f64, usize, usize), AnnError>
where
    F: Fn(&BenchmarkQuery) -> Result<Vec<SearchHit>, AnnError>,
{
    let mut recalls = Vec::with_capacity(queries.len());
    let mut leaks = 0usize;
    let mut stale = 0usize;
    for (q, truth) in queries.iter().zip(gt.iter()) {
        let hits = search_fn(q)?;
        recalls.push(recall_at_k(truth, &hits, 10));
        let audit = store.audit_hits(&q.access, &hits)?;
        leaks += audit.acl_leaks;
        stale += audit.stale_or_deleted_hits;
    }
    Ok((mean(&recalls), leaks, stale))
}

impl FullBenchmarkReport {
    /// Convert the entire benchmark report into a deterministic JSON receipt.
    pub fn to_json(&self) -> JsonVal {
        JsonVal::obj(vec![
            ("schema_version", JsonVal::Str("hip-1347-ann-bench/v1".into())),
            ("issue", JsonVal::Str("hanzoai/index#15".into())),
            ("seed", JsonVal::UInt(self.seed)),
            ("dim", JsonVal::UInt(self.dim as u64)),
            ("total_vectors", JsonVal::UInt(self.total_vectors as u64)),
            ("repo_files_scanned", JsonVal::UInt(self.repo_files_scanned as u64)),
            ("repo_chunks_loaded", JsonVal::UInt(self.repo_chunks_loaded as u64)),
            (
                "cost_model",
                JsonVal::obj(vec![
                    ("s3_rtt_ms", JsonVal::Float(self.cost_model.s3_rtt_ms)),
                    (
                        "s3_bandwidth_mb_per_sec",
                        JsonVal::Float(self.cost_model.s3_bandwidth_mb_per_sec),
                    ),
                    (
                        "s3_get_cost_per_1000_usd",
                        JsonVal::Float(self.cost_model.s3_get_cost_per_1000_usd),
                    ),
                    (
                        "s3_storage_cost_per_gb_month_usd",
                        JsonVal::Float(self.cost_model.s3_storage_cost_per_gb_month_usd),
                    ),
                    (
                        "nvme_storage_cost_per_gb_month_usd",
                        JsonVal::Float(self.cost_model.nvme_storage_cost_per_gb_month_usd),
                    ),
                    (
                        "dram_cost_per_gb_month_usd",
                        JsonVal::Float(self.cost_model.dram_cost_per_gb_month_usd),
                    ),
                    (
                        "local_nvme_page_read_us",
                        JsonVal::Float(self.cost_model.local_nvme_page_read_us),
                    ),
                ]),
            ),
            (
                "build_times_ms",
                JsonVal::Array(
                    self.build_times_ms
                        .iter()
                        .map(|(k, v)| {
                            JsonVal::obj(vec![
                                ("architecture", JsonVal::Str(k.clone())),
                                ("build_ms", JsonVal::Float(*v)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "measured_footprints",
                JsonVal::Array(
                    self.footprints
                        .iter()
                        .map(|(arch, fp)| {
                            JsonVal::obj(vec![
                                ("architecture", JsonVal::Str(arch.clone())),
                                ("live_vectors", JsonVal::UInt(fp.live_vectors as u64)),
                                ("ram_bytes", JsonVal::UInt(fp.ram_bytes as u64)),
                                (
                                    "disk_or_object_bytes",
                                    JsonVal::UInt(fp.disk_or_object_bytes as u64),
                                ),
                                ("bytes_per_vector_ram", JsonVal::Float(fp.bytes_per_vector_ram)),
                                ("bytes_per_vector_disk", JsonVal::Float(fp.bytes_per_vector_disk)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "measured_slices",
                JsonVal::Array(
                    self.measured_slices
                        .iter()
                        .map(|r| {
                            JsonVal::obj(vec![
                                ("architecture", JsonVal::Str(r.architecture.clone())),
                                ("selectivity", JsonVal::Str(r.selectivity.clone())),
                                ("cache_regime", JsonVal::Str(r.cache_regime.clone())),
                                ("is_projected", JsonVal::Bool(r.is_projected)),
                                ("query_count", JsonVal::UInt(r.query_count as u64)),
                                ("recall_at_10", JsonVal::Float(r.recall_at_10)),
                                ("cpu_p50_ms", JsonVal::Float(r.cpu_latency.p50_ms)),
                                ("cpu_p95_ms", JsonVal::Float(r.cpu_latency.p95_ms)),
                                ("cpu_p99_ms", JsonVal::Float(r.cpu_latency.p99_ms)),
                                ("total_p50_ms", JsonVal::Float(r.total_latency_with_io.p50_ms)),
                                ("total_p95_ms", JsonVal::Float(r.total_latency_with_io.p95_ms)),
                                ("total_p99_ms", JsonVal::Float(r.total_latency_with_io.p99_ms)),
                                ("avg_s3_gets", JsonVal::Float(r.avg_s3_gets)),
                                ("avg_sequential_rtts", JsonVal::Float(r.avg_sequential_rtts)),
                                ("avg_bytes_fetched", JsonVal::Float(r.avg_bytes_fetched)),
                                ("avg_local_page_faults", JsonVal::Float(r.avg_local_page_faults)),
                                ("acl_leaks", JsonVal::UInt(r.acl_leaks as u64)),
                                ("filter_violations", JsonVal::UInt(r.filter_violations as u64)),
                                ("stale_hits", JsonVal::UInt(r.stale_hits as u64)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "ultra_selective_1pct_negative_regime",
                JsonVal::Array(
                    self.ultra_selective_1pct
                        .iter()
                        .map(|r| {
                            JsonVal::obj(vec![
                                ("strategy", JsonVal::Str(r.strategy.clone())),
                                ("selectivity", JsonVal::Str(r.selectivity.clone())),
                                ("recall_at_10", JsonVal::Float(r.recall_at_10)),
                                ("avg_hits_returned", JsonVal::Float(r.avg_hits_returned)),
                                ("p50_ms", JsonVal::Float(r.p50_ms)),
                                ("p99_ms", JsonVal::Float(r.p99_ms)),
                                ("acl_leaks", JsonVal::UInt(r.acl_leaks as u64)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "mutation_lifecycle",
                JsonVal::Array(
                    self.mutations
                        .iter()
                        .map(|m| {
                            JsonVal::obj(vec![
                                ("architecture", JsonVal::Str(m.architecture.clone())),
                                ("inserts_applied", JsonVal::UInt(m.inserts_applied as u64)),
                                ("updates_applied", JsonVal::UInt(m.updates_applied as u64)),
                                ("deletes_applied", JsonVal::UInt(m.deletes_applied as u64)),
                                ("avg_upsert_us", JsonVal::Float(m.avg_upsert_us)),
                                ("avg_delete_us", JsonVal::Float(m.avg_delete_us)),
                                (
                                    "post_mutation_recall_at_10",
                                    JsonVal::Float(m.post_mutation_recall_at_10),
                                ),
                                (
                                    "post_mutation_acl_leaks",
                                    JsonVal::UInt(m.post_mutation_acl_leaks as u64),
                                ),
                                (
                                    "post_mutation_stale_or_deleted_hits",
                                    JsonVal::UInt(m.post_mutation_stale_or_deleted_hits as u64),
                                ),
                                (
                                    "compaction_tombstones_reclaimed",
                                    JsonVal::UInt(m.compaction.tombstones_reclaimed as u64),
                                ),
                                (
                                    "compaction_delta_merged",
                                    JsonVal::UInt(m.compaction.delta_entries_merged as u64),
                                ),
                                (
                                    "compaction_bytes_before",
                                    JsonVal::UInt(m.compaction.bytes_before as u64),
                                ),
                                (
                                    "compaction_bytes_after",
                                    JsonVal::UInt(m.compaction.bytes_after as u64),
                                ),
                                (
                                    "compaction_duration_ms",
                                    JsonVal::Float(m.compaction.duration_ms),
                                ),
                                (
                                    "post_compaction_recall_at_10",
                                    JsonVal::Float(m.post_compaction_recall_at_10),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "downstream_repo_tasks",
                JsonVal::Array(
                    self.downstream_tasks
                        .iter()
                        .map(|t| {
                            JsonVal::obj(vec![
                                ("architecture", JsonVal::Str(t.architecture.clone())),
                                ("task_count", JsonVal::UInt(t.task_count as u64)),
                                ("recall_at_10", JsonVal::Float(t.recall_at_10)),
                                (
                                    "file_resolution_success_at_1",
                                    JsonVal::Float(t.file_resolution_success_at_1),
                                ),
                                (
                                    "file_resolution_success_at_5",
                                    JsonVal::Float(t.file_resolution_success_at_5),
                                ),
                                ("p99_ms_warm", JsonVal::Float(t.p99_ms_warm)),
                                ("acl_leaks", JsonVal::UInt(t.acl_leaks as u64)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "drop_rebuild_parity",
                JsonVal::obj(vec![
                    (
                        "canonical_fingerprint_before",
                        JsonVal::Str(format!(
                            "0x{:016x}",
                            self.rebuild_parity.canonical_fingerprint_before
                        )),
                    ),
                    (
                        "canonical_fingerprint_after",
                        JsonVal::Str(format!(
                            "0x{:016x}",
                            self.rebuild_parity.canonical_fingerprint_after
                        )),
                    ),
                    ("canonical_unchanged", JsonVal::Bool(self.rebuild_parity.canonical_unchanged)),
                    (
                        "vamana_binary_bit_exact",
                        JsonVal::Bool(self.rebuild_parity.vamana_binary_bit_exact),
                    ),
                    (
                        "vamana_hits_bit_exact",
                        JsonVal::Bool(self.rebuild_parity.vamana_hits_bit_exact),
                    ),
                    (
                        "ivf_pq_hits_bit_exact",
                        JsonVal::Bool(self.rebuild_parity.ivf_pq_hits_bit_exact),
                    ),
                    (
                        "exact_flat_hits_bit_exact",
                        JsonVal::Bool(self.rebuild_parity.exact_flat_hits_bit_exact),
                    ),
                ]),
            ),
            (
                "projected_scaling_768d",
                JsonVal::Array(self.projected_scaling_768d.iter().map(scale_row_to_json).collect()),
            ),
            (
                "projected_scaling_1536d",
                JsonVal::Array(
                    self.projected_scaling_1536d.iter().map(scale_row_to_json).collect(),
                ),
            ),
        ])
    }
}

fn scale_row_to_json(r: &ScaleProjectionRow) -> JsonVal {
    JsonVal::obj(vec![
        ("architecture", JsonVal::Str(r.architecture.clone())),
        ("num_vectors", JsonVal::UInt(r.num_vectors)),
        ("dim", JsonVal::UInt(r.dim as u64)),
        ("is_projected", JsonVal::Bool(r.is_projected)),
        ("ram_bytes_per_vector", JsonVal::Float(r.ram_bytes_per_vector)),
        ("disk_bytes_per_vector", JsonVal::Float(r.disk_bytes_per_vector)),
        ("total_ram_gb", JsonVal::Float(r.total_ram_gb)),
        ("total_storage_gb", JsonVal::Float(r.total_storage_gb)),
        ("s3_gets_per_query_cold", JsonVal::Float(r.s3_gets_per_query_cold)),
        ("sequential_rtts_cold", JsonVal::Float(r.sequential_rtts_cold)),
        ("bytes_fetched_per_query_kb", JsonVal::Float(r.bytes_fetched_per_query_kb)),
        ("est_cold_p99_ms", JsonVal::Float(r.est_cold_p99_ms)),
        ("monthly_storage_cost_usd", JsonVal::Float(r.monthly_storage_cost_usd)),
        ("monthly_dram_cost_usd", JsonVal::Float(r.monthly_dram_cost_usd)),
        ("s3_get_cost_per_1m_queries_usd", JsonVal::Float(r.s3_get_cost_per_1m_queries_usd)),
        ("notes", JsonVal::Str(r.notes.clone())),
    ])
}

#[inline]
fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1_000.0
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / (vals.len() as f64)
    }
}
