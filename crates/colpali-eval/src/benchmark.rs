use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use crate::colpali_index::{ColPaliBinaryTwoStageIndex, ColPaliFp32Index};
use crate::corpus::{build_benchmark_corpus, CorpusProvenance, EvalQuery};
use crate::metrics::{
    audit_access_leaks, compute_mrr, compute_ndcg_at_k, compute_recall_at_k, hits_bit_exact,
    summarize_slice, DropRebuildParityReport, SliceMetrics,
};
use crate::ocr_hybrid::OcrBm25HybridIndex;
use crate::router::HybridVisualRouterIndex;
use crate::scaling::{
    compute_storage_explosion_projections, StorageCostAssumptions, StorageProjectionRow,
};
use crate::types::{
    CanonicalPageRecord, ColPaliError, IndexFootprint, PageModality, SearchHit, StrategyId,
    VisualQuery, DEFAULT_EMBEDDING_DIM,
};

/// Configuration for a reproducible ColPali visual retrieval benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub seed: u64,
    pub total_pages: usize,
    pub patches_per_page: usize,
    pub dim: usize,
    pub queries_per_slice: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            seed: 1347,
            total_pages: 512,
            patches_per_page: 64,
            dim: DEFAULT_EMBEDDING_DIM,
            queries_per_slice: 24,
        }
    }
}

/// Complete measured results for one strategy (`is_projected: false`).
#[derive(Debug, Clone)]
pub struct StrategyReport {
    pub strategy: StrategyId,
    pub build_time_ms: f64,
    pub footprint: IndexFootprint,
    pub overall_multimodal: SliceMetrics,
    pub by_modality: BTreeMap<PageModality, SliceMetrics>,
    pub negative_regime_1_exact_keyword: SliceMetrics,
    pub binary_patch_compression_ratio: Option<f64>,
    pub router_visual_indexing_ratio: Option<f64>,
}

/// Adversarial ACL clone isolation check result.
#[derive(Debug, Clone)]
pub struct AdversarialAclReport {
    pub planted_clone_pages: usize,
    pub queries_tested_per_strategy: usize,
    pub total_acl_leaks: usize,
    pub total_cross_tenant_leaks: usize,
    pub fail_closed_checks_passed: bool,
}

/// Top-level reproducible benchmark receipt serialized to `results.json`.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub schema_version: String,
    pub tracking_issue: String,
    pub hip: String,
    pub provenance: CorpusProvenance,
    pub strategies: Vec<StrategyReport>,
    pub binary_speedup_vs_fp32_p50: f64,
    pub binary_speedup_vs_fp32_p99: f64,
    pub binary_ram_reduction_vs_fp32: f64,
    pub router_ram_reduction_vs_fp32: f64,
    pub drop_rebuild_parity: DropRebuildParityReport,
    pub adversarial_acl_audit: AdversarialAclReport,
    pub cost_assumptions: StorageCostAssumptions,
    pub negative_regime_2_storage_projections: Vec<StorageProjectionRow>,
}

/// Execute the full deterministic HIP-1347 ColPali benchmark suite.
pub fn run_benchmark(
    repo_root: &Path,
    config: &BenchmarkConfig,
) -> Result<BenchmarkReport, ColPaliError> {
    let corpus = build_benchmark_corpus(
        repo_root,
        config.seed,
        config.total_pages,
        config.patches_per_page,
        config.dim,
        config.queries_per_slice,
    )?;

    let canonical_fp_before = corpus.store.fingerprint();

    // Build all 4 derived indexes from canonical truth and time each build.
    let t0 = Instant::now();
    let ocr_idx = OcrBm25HybridIndex::build(&corpus.store)?;
    let ocr_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let fp32_idx = ColPaliFp32Index::build(&corpus.store)?;
    let fp32_build_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let bin_idx = ColPaliBinaryTwoStageIndex::build(&corpus.store)?;
    let bin_build_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    let router_idx = HybridVisualRouterIndex::build(&corpus.store)?;
    let router_build_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // Evaluate all 4 strategies across all 4 modalities + Negative Regime 1.
    let ocr_report = evaluate_strategy(
        StrategyId::OcrBm25HybridBaseline,
        ocr_build_ms,
        ocr_idx.footprint(),
        None,
        None,
        &corpus.modality_queries,
        &corpus.exact_keyword_queries,
        |q| ocr_idx.search(q),
    )?;

    let fp32_report = evaluate_strategy(
        StrategyId::ColPaliFp32MaxSim,
        fp32_build_ms,
        fp32_idx.footprint(),
        None,
        None,
        &corpus.modality_queries,
        &corpus.exact_keyword_queries,
        |q| fp32_idx.search(q),
    )?;

    let bin_report = evaluate_strategy(
        StrategyId::ColPaliBinaryQuantizedTwoStage,
        bin_build_ms,
        bin_idx.footprint(),
        Some(bin_idx.binary_patch_compression_ratio()),
        None,
        &corpus.modality_queries,
        &corpus.exact_keyword_queries,
        |q| bin_idx.search(q),
    )?;

    let router_report = evaluate_strategy(
        StrategyId::HybridTextPlusColPaliRouter,
        router_build_ms,
        router_idx.footprint(),
        None,
        Some(router_idx.visual_indexing_ratio()),
        &corpus.modality_queries,
        &corpus.exact_keyword_queries,
        |q| router_idx.search(q),
    )?;

    // Drop & Rebuild bit-exact parity audit across all 4 derived indexes.
    let parity_queries: Vec<&VisualQuery> =
        corpus.modality_queries.iter().take(12).map(|eq| &eq.query).collect();

    let ocr_ck_before = ocr_idx.checksum_crc32();
    let fp32_ck_before = fp32_idx.checksum_crc32();
    let bin_ck_before = bin_idx.checksum_crc32();
    let router_ck_before = router_idx.checksum_crc32();

    let ocr_hits_before = run_query_batch(&parity_queries, |q| ocr_idx.search(q))?;
    let fp32_hits_before = run_query_batch(&parity_queries, |q| fp32_idx.search(q))?;
    let bin_hits_before = run_query_batch(&parity_queries, |q| bin_idx.search(q))?;
    let router_hits_before = run_query_batch(&parity_queries, |q| router_idx.search(q))?;

    // Explicitly drop all derived views and rebuild strictly from canonical truth.
    drop(ocr_idx);
    drop(fp32_idx);
    drop(bin_idx);
    drop(router_idx);

    let ocr_rebuilt = OcrBm25HybridIndex::build(&corpus.store)?;
    let fp32_rebuilt = ColPaliFp32Index::build(&corpus.store)?;
    let bin_rebuilt = ColPaliBinaryTwoStageIndex::build(&corpus.store)?;
    let router_rebuilt = HybridVisualRouterIndex::build(&corpus.store)?;

    let ocr_hits_after = run_query_batch(&parity_queries, |q| ocr_rebuilt.search(q))?;
    let fp32_hits_after = run_query_batch(&parity_queries, |q| fp32_rebuilt.search(q))?;
    let bin_hits_after = run_query_batch(&parity_queries, |q| bin_rebuilt.search(q))?;
    let router_hits_after = run_query_batch(&parity_queries, |q| router_rebuilt.search(q))?;

    let canonical_fp_after = corpus.store.fingerprint();
    let drop_rebuild_parity = DropRebuildParityReport {
        canonical_fingerprint_before: canonical_fp_before,
        canonical_fingerprint_after: canonical_fp_after,
        canonical_unchanged: canonical_fp_before == canonical_fp_after,
        ocr_hybrid_checksum_exact: ocr_ck_before == ocr_rebuilt.checksum_crc32(),
        ocr_hybrid_hits_bit_exact: batch_hits_bit_exact(&ocr_hits_before, &ocr_hits_after),
        colpali_fp32_checksum_exact: fp32_ck_before == fp32_rebuilt.checksum_crc32(),
        colpali_fp32_hits_bit_exact: batch_hits_bit_exact(&fp32_hits_before, &fp32_hits_after),
        colpali_binary_checksum_exact: bin_ck_before == bin_rebuilt.checksum_crc32(),
        colpali_binary_hits_bit_exact: batch_hits_bit_exact(&bin_hits_before, &bin_hits_after),
        hybrid_router_checksum_exact: router_ck_before == router_rebuilt.checksum_crc32(),
        hybrid_router_hits_bit_exact: batch_hits_bit_exact(&router_hits_before, &router_hits_after),
    };

    // Adversarial ACL exact-clone audit: plant distance-0.0 clones of real target pages
    // belonging to `tenant-external-lab` and `sec-confidential` into a cloned store and
    // verify 0 leaks across all 4 engines + fail-closed empty tenant/ACL checks.
    let adversarial_acl_audit = run_adversarial_acl_audit(&corpus.store, &corpus.modality_queries)?;

    let binary_speedup_vs_fp32_p50 = if bin_report.overall_multimodal.latency_p50_ms > 0.0 {
        fp32_report.overall_multimodal.latency_p50_ms / bin_report.overall_multimodal.latency_p50_ms
    } else {
        0.0
    };
    let binary_speedup_vs_fp32_p99 = if bin_report.overall_multimodal.latency_p99_ms > 0.0 {
        fp32_report.overall_multimodal.latency_p99_ms / bin_report.overall_multimodal.latency_p99_ms
    } else {
        0.0
    };
    let binary_ram_reduction_vs_fp32 = if bin_report.footprint.bytes_per_page_ram > 0.0 {
        fp32_report.footprint.bytes_per_page_ram / bin_report.footprint.bytes_per_page_ram
    } else {
        0.0
    };
    let router_ram_reduction_vs_fp32 = if router_report.footprint.bytes_per_page_ram > 0.0 {
        fp32_report.footprint.bytes_per_page_ram / router_report.footprint.bytes_per_page_ram
    } else {
        0.0
    };

    let cost_assumptions = StorageCostAssumptions::default();
    let negative_regime_2_storage_projections =
        compute_storage_explosion_projections(cost_assumptions);

    Ok(BenchmarkReport {
        schema_version: "hip-1347-colpali-bench/v1".to_string(),
        tracking_issue: "hanzoai/index#23".to_string(),
        hip: "HIP-1347".to_string(),
        provenance: corpus.provenance,
        strategies: vec![ocr_report, fp32_report, bin_report, router_report],
        binary_speedup_vs_fp32_p50,
        binary_speedup_vs_fp32_p99,
        binary_ram_reduction_vs_fp32,
        router_ram_reduction_vs_fp32,
        drop_rebuild_parity,
        adversarial_acl_audit,
        cost_assumptions,
        negative_regime_2_storage_projections,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_strategy<F>(
    strategy: StrategyId,
    build_time_ms: f64,
    footprint: IndexFootprint,
    binary_patch_compression_ratio: Option<f64>,
    router_visual_indexing_ratio: Option<f64>,
    modality_queries: &[EvalQuery],
    exact_keyword_queries: &[EvalQuery],
    search_fn: F,
) -> Result<StrategyReport, ColPaliError>
where
    F: Fn(&VisualQuery) -> Result<Vec<SearchHit>, ColPaliError>,
{
    // Warm up 2 queries so initial allocator/branch predictor state does not skew p99.
    for eq in modality_queries.iter().take(2) {
        let _ = search_fn(&eq.query)?;
    }

    let mut all_ndcg = Vec::with_capacity(modality_queries.len());
    let mut all_recall = Vec::with_capacity(modality_queries.len());
    let mut all_mrr = Vec::with_capacity(modality_queries.len());
    let mut all_lat = Vec::with_capacity(modality_queries.len());
    let mut all_acl_leaks = 0usize;
    let mut all_tenant_leaks = 0usize;

    let mut mod_ndcg: BTreeMap<PageModality, Vec<f64>> = BTreeMap::new();
    let mut mod_recall: BTreeMap<PageModality, Vec<f64>> = BTreeMap::new();
    let mut mod_mrr: BTreeMap<PageModality, Vec<f64>> = BTreeMap::new();
    let mut mod_lat: BTreeMap<PageModality, Vec<f64>> = BTreeMap::new();
    let mut mod_acl: BTreeMap<PageModality, usize> = BTreeMap::new();
    let mut mod_tenant: BTreeMap<PageModality, usize> = BTreeMap::new();

    for eq in modality_queries {
        let t0 = Instant::now();
        let hits = search_fn(&eq.query)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let ndcg = compute_ndcg_at_k(&hits, &eq.relevance, 5);
        let recall = compute_recall_at_k(&hits, &eq.relevance, 5);
        let mrr = compute_mrr(&hits, &eq.relevance);
        let (acl_l, ten_l) = audit_access_leaks(&hits, &eq.query.scope);

        all_ndcg.push(ndcg);
        all_recall.push(recall);
        all_mrr.push(mrr);
        all_lat.push(elapsed_ms);
        all_acl_leaks += acl_l;
        all_tenant_leaks += ten_l;

        mod_ndcg.entry(eq.target_modality).or_default().push(ndcg);
        mod_recall.entry(eq.target_modality).or_default().push(recall);
        mod_mrr.entry(eq.target_modality).or_default().push(mrr);
        mod_lat.entry(eq.target_modality).or_default().push(elapsed_ms);
        *mod_acl.entry(eq.target_modality).or_insert(0) += acl_l;
        *mod_tenant.entry(eq.target_modality).or_insert(0) += ten_l;
    }

    let overall_multimodal = summarize_slice(
        &all_ndcg,
        &all_recall,
        &all_mrr,
        &all_lat,
        all_acl_leaks,
        all_tenant_leaks,
    );

    let mut by_modality = BTreeMap::new();
    for m in PageModality::all() {
        let slice = summarize_slice(
            mod_ndcg.get(&m).map(Vec::as_slice).unwrap_or(&[]),
            mod_recall.get(&m).map(Vec::as_slice).unwrap_or(&[]),
            mod_mrr.get(&m).map(Vec::as_slice).unwrap_or(&[]),
            mod_lat.get(&m).map(Vec::as_slice).unwrap_or(&[]),
            mod_acl.get(&m).copied().unwrap_or(0),
            mod_tenant.get(&m).copied().unwrap_or(0),
        );
        by_modality.insert(m, slice);
    }

    let mut neg1_ndcg = Vec::with_capacity(exact_keyword_queries.len());
    let mut neg1_recall = Vec::with_capacity(exact_keyword_queries.len());
    let mut neg1_mrr = Vec::with_capacity(exact_keyword_queries.len());
    let mut neg1_lat = Vec::with_capacity(exact_keyword_queries.len());
    let mut neg1_acl = 0usize;
    let mut neg1_ten = 0usize;

    for eq in exact_keyword_queries {
        let t0 = Instant::now();
        let hits = search_fn(&eq.query)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (acl_l, ten_l) = audit_access_leaks(&hits, &eq.query.scope);
        neg1_ndcg.push(compute_ndcg_at_k(&hits, &eq.relevance, 5));
        neg1_recall.push(compute_recall_at_k(&hits, &eq.relevance, 5));
        neg1_mrr.push(compute_mrr(&hits, &eq.relevance));
        neg1_lat.push(elapsed_ms);
        neg1_acl += acl_l;
        neg1_ten += ten_l;
    }

    let negative_regime_1_exact_keyword =
        summarize_slice(&neg1_ndcg, &neg1_recall, &neg1_mrr, &neg1_lat, neg1_acl, neg1_ten);

    Ok(StrategyReport {
        strategy,
        build_time_ms,
        footprint,
        overall_multimodal,
        by_modality,
        negative_regime_1_exact_keyword,
        binary_patch_compression_ratio,
        router_visual_indexing_ratio,
    })
}

fn run_query_batch<F>(
    queries: &[&VisualQuery],
    search_fn: F,
) -> Result<Vec<Vec<SearchHit>>, ColPaliError>
where
    F: Fn(&VisualQuery) -> Result<Vec<SearchHit>, ColPaliError>,
{
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        out.push(search_fn(q)?);
    }
    Ok(out)
}

fn batch_hits_bit_exact(a: &[Vec<SearchHit>], b: &[Vec<SearchHit>]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| hits_bit_exact(x, y))
}

fn run_adversarial_acl_audit(
    base_store: &crate::canonical::CanonicalPageStore,
    modality_queries: &[EvalQuery],
) -> Result<AdversarialAclReport, ColPaliError> {
    let mut adv_store = base_store.clone();
    let sample_queries: Vec<&EvalQuery> = modality_queries.iter().take(12).collect();
    let mut planted = 0usize;

    for (i, eq) in sample_queries.iter().enumerate() {
        if let Some(target_key) = eq.relevance.keys().next() {
            if let Some(orig) = base_store.get(target_key) {
                // Plant cross-tenant exact clone
                let mut cross_tenant_clone: CanonicalPageRecord = orig.clone();
                cross_tenant_clone.doc_id = format!("adv-cross-tenant-{i:03}");
                cross_tenant_clone.tenant_id = "tenant-adversarial-other".to_string();
                adv_store.upsert(cross_tenant_clone)?;
                planted += 1;

                // Plant same-tenant restricted ACL exact clone
                let mut restricted_acl_clone: CanonicalPageRecord = orig.clone();
                restricted_acl_clone.doc_id = format!("adv-restricted-acl-{i:03}");
                restricted_acl_clone.acl_group = "sec-top-secret".to_string();
                adv_store.upsert(restricted_acl_clone)?;
                planted += 1;
            }
        }
    }

    let ocr = OcrBm25HybridIndex::build(&adv_store)?;
    let fp32 = ColPaliFp32Index::build(&adv_store)?;
    let bin = ColPaliBinaryTwoStageIndex::build(&adv_store)?;
    let router = HybridVisualRouterIndex::build(&adv_store)?;

    let mut total_acl_leaks = 0usize;
    let mut total_cross_tenant_leaks = 0usize;
    for eq in &sample_queries {
        for hits in [
            ocr.search(&eq.query)?,
            fp32.search(&eq.query)?,
            bin.search(&eq.query)?,
            router.search(&eq.query)?,
        ] {
            let (a, t) = audit_access_leaks(&hits, &eq.query.scope);
            total_acl_leaks += a;
            total_cross_tenant_leaks += t;
        }
    }

    // Fail-closed checks: empty tenant & empty ACL groups must return Err across all 4 indexes.
    let mut fail_closed_ok = true;
    if let Some(first) = sample_queries.first() {
        let mut bad_tenant = first.query.clone();
        bad_tenant.scope.tenant_id = "   ".to_string();
        let mut bad_acl = first.query.clone();
        bad_acl.scope.allowed_acl_groups.clear();

        for res in [
            ocr.search(&bad_tenant),
            fp32.search(&bad_tenant),
            bin.search(&bad_tenant),
            router.search(&bad_tenant),
        ] {
            if !matches!(res, Err(ColPaliError::EmptyTenantId)) {
                fail_closed_ok = false;
            }
        }
        for res in [
            ocr.search(&bad_acl),
            fp32.search(&bad_acl),
            bin.search(&bad_acl),
            router.search(&bad_acl),
        ] {
            if !matches!(res, Err(ColPaliError::EmptyAclGroups)) {
                fail_closed_ok = false;
            }
        }
    }

    Ok(AdversarialAclReport {
        planted_clone_pages: planted,
        queries_tested_per_strategy: sample_queries.len(),
        total_acl_leaks,
        total_cross_tenant_leaks,
        fail_closed_checks_passed: fail_closed_ok,
    })
}

/// Pure-Rust zero-dependency pretty JSON serializer for [`BenchmarkReport`].
pub fn report_to_json(report: &BenchmarkReport) -> String {
    let mut out = String::with_capacity(16_384);
    out.push_str("{\n");
    push_kv_str(&mut out, 2, "schema_version", &report.schema_version, true);
    push_kv_str(&mut out, 2, "tracking_issue", &report.tracking_issue, true);
    push_kv_str(&mut out, 2, "hip", &report.hip, true);
    push_kv_bool(&mut out, 2, "measured_section_is_projected", false, true);

    // Provenance
    out.push_str("  \"corpus_provenance\": {\n");
    push_kv_u64(&mut out, 4, "seed", report.provenance.seed, true);
    push_kv_usize(&mut out, 4, "total_pages", report.provenance.total_pages, true);
    push_kv_usize(&mut out, 4, "patches_per_page", report.provenance.patches_per_page, true);
    push_kv_usize(&mut out, 4, "dim", report.provenance.dim, true);
    out.push_str("    \"repo_files_loaded\": [");
    for (i, f) in report.provenance.repo_files_loaded.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('"');
        out.push_str(&escape_json(f));
        out.push('"');
    }
    out.push_str("],\n");
    out.push_str("    \"modality_page_counts\": {\n");
    let mods = PageModality::all();
    for (i, m) in mods.iter().enumerate() {
        let cnt = report.provenance.modality_page_counts.get(m).copied().unwrap_or(0);
        push_kv_usize(&mut out, 6, m.as_str(), cnt, i + 1 < mods.len());
    }
    out.push_str("    },\n");
    out.push_str("    \"mean_ocr_corruption_rate_by_modality\": {\n");
    for (i, m) in mods.iter().enumerate() {
        let rate = report.provenance.mean_ocr_corruption_by_modality.get(m).copied().unwrap_or(0.0);
        push_kv_f64(&mut out, 6, m.as_str(), rate, i + 1 < mods.len());
    }
    out.push_str("    }\n");
    out.push_str("  },\n");

    // Headline measured comparisons
    out.push_str("  \"headline_measured_ratios\": {\n");
    push_kv_bool(&mut out, 4, "is_projected", false, true);
    push_kv_f64(&mut out, 4, "binary_speedup_vs_fp32_p50", report.binary_speedup_vs_fp32_p50, true);
    push_kv_f64(&mut out, 4, "binary_speedup_vs_fp32_p99", report.binary_speedup_vs_fp32_p99, true);
    push_kv_f64(
        &mut out,
        4,
        "binary_ram_reduction_vs_fp32",
        report.binary_ram_reduction_vs_fp32,
        true,
    );
    push_kv_f64(
        &mut out,
        4,
        "router_ram_reduction_vs_fp32",
        report.router_ram_reduction_vs_fp32,
        false,
    );
    out.push_str("  },\n");

    // Measured strategies
    out.push_str("  \"measured_strategies\": [\n");
    for (si, s) in report.strategies.iter().enumerate() {
        out.push_str("    {\n");
        push_kv_bool(&mut out, 6, "is_projected", false, true);
        push_kv_str(&mut out, 6, "strategy", s.strategy.as_str(), true);
        push_kv_f64(&mut out, 6, "build_time_ms", s.build_time_ms, true);
        out.push_str("      \"footprint\": {\n");
        push_kv_usize(&mut out, 8, "ram_bytes", s.footprint.ram_bytes, true);
        push_kv_usize(&mut out, 8, "disk_bytes", s.footprint.disk_bytes, true);
        push_kv_usize(&mut out, 8, "indexed_pages", s.footprint.indexed_pages, true);
        push_kv_usize(&mut out, 8, "visual_pages_indexed", s.footprint.visual_pages_indexed, true);
        push_kv_f64(&mut out, 8, "bytes_per_page_ram", s.footprint.bytes_per_page_ram, true);
        push_kv_f64(&mut out, 8, "bytes_per_page_disk", s.footprint.bytes_per_page_disk, false);
        out.push_str("      },\n");
        if let Some(ratio) = s.binary_patch_compression_ratio {
            push_kv_f64(&mut out, 6, "binary_patch_compression_ratio", ratio, true);
        }
        if let Some(ratio) = s.router_visual_indexing_ratio {
            push_kv_f64(&mut out, 6, "router_visual_indexing_ratio", ratio, true);
        }
        out.push_str("      \"overall_multimodal\": ");
        push_slice_metrics(&mut out, 6, &s.overall_multimodal);
        out.push_str(",\n");
        out.push_str("      \"by_modality\": {\n");
        for (mi, m) in mods.iter().enumerate() {
            out.push_str("        \"");
            out.push_str(m.as_str());
            out.push_str("\": ");
            if let Some(slice) = s.by_modality.get(m) {
                push_slice_metrics(&mut out, 8, slice);
            } else {
                out.push_str("{}");
            }
            if mi + 1 < mods.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("      },\n");
        out.push_str("      \"negative_regime_1_exact_keyword\": ");
        push_slice_metrics(&mut out, 6, &s.negative_regime_1_exact_keyword);
        out.push('\n');
        out.push_str("    }");
        if si + 1 < report.strategies.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ],\n");

    // Drop/Rebuild parity
    let dr = &report.drop_rebuild_parity;
    out.push_str("  \"drop_rebuild_parity\": {\n");
    push_kv_str(
        &mut out,
        4,
        "canonical_fingerprint_before",
        &format!("0x{:016x}", dr.canonical_fingerprint_before),
        true,
    );
    push_kv_str(
        &mut out,
        4,
        "canonical_fingerprint_after",
        &format!("0x{:016x}", dr.canonical_fingerprint_after),
        true,
    );
    push_kv_bool(&mut out, 4, "canonical_unchanged", dr.canonical_unchanged, true);
    push_kv_bool(&mut out, 4, "ocr_hybrid_checksum_exact", dr.ocr_hybrid_checksum_exact, true);
    push_kv_bool(&mut out, 4, "ocr_hybrid_hits_bit_exact", dr.ocr_hybrid_hits_bit_exact, true);
    push_kv_bool(&mut out, 4, "colpali_fp32_checksum_exact", dr.colpali_fp32_checksum_exact, true);
    push_kv_bool(&mut out, 4, "colpali_fp32_hits_bit_exact", dr.colpali_fp32_hits_bit_exact, true);
    push_kv_bool(
        &mut out,
        4,
        "colpali_binary_checksum_exact",
        dr.colpali_binary_checksum_exact,
        true,
    );
    push_kv_bool(
        &mut out,
        4,
        "colpali_binary_hits_bit_exact",
        dr.colpali_binary_hits_bit_exact,
        true,
    );
    push_kv_bool(
        &mut out,
        4,
        "hybrid_router_checksum_exact",
        dr.hybrid_router_checksum_exact,
        true,
    );
    push_kv_bool(
        &mut out,
        4,
        "hybrid_router_hits_bit_exact",
        dr.hybrid_router_hits_bit_exact,
        true,
    );
    push_kv_bool(&mut out, 4, "all_passed", dr.all_passed(), false);
    out.push_str("  },\n");

    // Adversarial ACL audit
    let adv = &report.adversarial_acl_audit;
    out.push_str("  \"adversarial_acl_audit\": {\n");
    push_kv_usize(&mut out, 4, "planted_clone_pages", adv.planted_clone_pages, true);
    push_kv_usize(
        &mut out,
        4,
        "queries_tested_per_strategy",
        adv.queries_tested_per_strategy,
        true,
    );
    push_kv_usize(&mut out, 4, "total_acl_leaks", adv.total_acl_leaks, true);
    push_kv_usize(&mut out, 4, "total_cross_tenant_leaks", adv.total_cross_tenant_leaks, true);
    push_kv_bool(&mut out, 4, "fail_closed_checks_passed", adv.fail_closed_checks_passed, false);
    out.push_str("  },\n");

    // Negative Regime 2: Storage scaling projections (`is_projected: true`)
    out.push_str("  \"negative_regime_2_storage_projections\": {\n");
    push_kv_bool(&mut out, 4, "is_projected", true, true);
    out.push_str("    \"cost_assumptions\": {\n");
    push_kv_f64(
        &mut out,
        6,
        "dram_usd_per_gb_month",
        report.cost_assumptions.dram_usd_per_gb_month,
        true,
    );
    push_kv_f64(
        &mut out,
        6,
        "nvme_usd_per_gb_month",
        report.cost_assumptions.nvme_usd_per_gb_month,
        true,
    );
    push_kv_f64(
        &mut out,
        6,
        "s3_usd_per_gb_month",
        report.cost_assumptions.s3_usd_per_gb_month,
        false,
    );
    out.push_str("    },\n");
    out.push_str("    \"rows\": [\n");
    for (ri, row) in report.negative_regime_2_storage_projections.iter().enumerate() {
        push_projection_row(&mut out, 6, row);
        if ri + 1 < report.negative_regime_2_storage_projections.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("    ]\n");
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

fn push_slice_metrics(out: &mut String, indent: usize, m: &SliceMetrics) {
    let pad = " ".repeat(indent);
    out.push_str("{\n");
    push_kv_usize(out, indent + 2, "query_count", m.query_count, true);
    push_kv_f64(out, indent + 2, "ndcg_at_5", m.ndcg_at_5, true);
    push_kv_f64(out, indent + 2, "recall_at_5", m.recall_at_5, true);
    push_kv_f64(out, indent + 2, "mrr", m.mrr, true);
    push_kv_f64(out, indent + 2, "latency_p50_ms", m.latency_p50_ms, true);
    push_kv_f64(out, indent + 2, "latency_p95_ms", m.latency_p95_ms, true);
    push_kv_f64(out, indent + 2, "latency_p99_ms", m.latency_p99_ms, true);
    push_kv_f64(out, indent + 2, "mean_latency_ms", m.mean_latency_ms, true);
    push_kv_usize(out, indent + 2, "acl_leaks", m.acl_leaks, true);
    push_kv_usize(out, indent + 2, "cross_tenant_leaks", m.cross_tenant_leaks, false);
    out.push_str(&pad);
    out.push('}');
}

fn push_projection_row(out: &mut String, indent: usize, r: &StorageProjectionRow) {
    let pad = " ".repeat(indent);
    out.push_str(&pad);
    out.push_str("{\n");
    push_kv_bool(out, indent + 2, "is_projected", r.is_projected, true);
    push_kv_u64(out, indent + 2, "scale_pages", r.scale_pages, true);
    push_kv_str(out, indent + 2, "architecture", &r.architecture, true);
    push_kv_usize(out, indent + 2, "patches_per_page", r.patches_per_page, true);
    push_kv_usize(out, indent + 2, "dim", r.dim, true);
    push_kv_f64(out, indent + 2, "visual_page_fraction", r.visual_page_fraction, true);
    push_kv_f64(out, indent + 2, "bytes_per_page_ram", r.bytes_per_page_ram, true);
    push_kv_f64(out, indent + 2, "bytes_per_page_disk", r.bytes_per_page_disk, true);
    push_kv_f64(out, indent + 2, "total_ram_gb", r.total_ram_gb, true);
    push_kv_f64(out, indent + 2, "total_disk_gb", r.total_disk_gb, true);
    push_kv_f64(out, indent + 2, "compression_vs_fp32_ram", r.compression_vs_fp32_ram, true);
    push_kv_f64(out, indent + 2, "monthly_dram_usd", r.monthly_dram_usd, true);
    push_kv_f64(out, indent + 2, "monthly_s3_usd", r.monthly_s3_usd, false);
    out.push_str(&pad);
    out.push('}');
}

fn push_kv_str(out: &mut String, indent: usize, key: &str, val: &str, comma: bool) {
    out.push_str(&" ".repeat(indent));
    out.push('"');
    out.push_str(key);
    out.push_str("\": \"");
    out.push_str(&escape_json(val));
    out.push('"');
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_kv_u64(out: &mut String, indent: usize, key: &str, val: u64, comma: bool) {
    out.push_str(&" ".repeat(indent));
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(&val.to_string());
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_kv_usize(out: &mut String, indent: usize, key: &str, val: usize, comma: bool) {
    push_kv_u64(out, indent, key, val as u64, comma);
}

fn push_kv_bool(out: &mut String, indent: usize, key: &str, val: bool, comma: bool) {
    out.push_str(&" ".repeat(indent));
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(if val { "true" } else { "false" });
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_kv_f64(out: &mut String, indent: usize, key: &str, val: f64, comma: bool) {
    out.push_str(&" ".repeat(indent));
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
    let clean = if val.is_finite() { val } else { 0.0 };
    out.push_str(&format!("{clean:.6}"));
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}
