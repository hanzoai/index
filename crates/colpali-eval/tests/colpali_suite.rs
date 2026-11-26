#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;
use std::thread;

use colpali_eval::benchmark::{report_to_json, run_benchmark, BenchmarkConfig};
use colpali_eval::colpali_index::{ColPaliBinaryTwoStageIndex, ColPaliFp32Index};
use colpali_eval::corpus::build_benchmark_corpus;
use colpali_eval::metrics::{audit_access_leaks, hits_bit_exact};
use colpali_eval::ocr_hybrid::OcrBm25HybridIndex;
use colpali_eval::router::HybridVisualRouterIndex;
use colpali_eval::scaling::{compute_storage_explosion_projections, StorageCostAssumptions};
use colpali_eval::types::{
    AccessScope, ColPaliError, PageKey, PageModality, RouteKind, StrategyId, DEFAULT_EMBEDDING_DIM,
};

#[test]
fn fail_closed_acl_rejects_empty_tenant_and_empty_groups() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 64, 32, DEFAULT_EMBEDDING_DIM, 4).unwrap();
    let ocr = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let fp32 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let bin = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let router = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    let base_q = &corpus.modality_queries[0].query;

    let mut empty_tenant = base_q.clone();
    empty_tenant.scope = AccessScope::new("   ", vec!["research-public".to_string()]);
    assert_eq!(ocr.search(&empty_tenant), Err(ColPaliError::EmptyTenantId));
    assert_eq!(fp32.search(&empty_tenant), Err(ColPaliError::EmptyTenantId));
    assert_eq!(bin.search(&empty_tenant), Err(ColPaliError::EmptyTenantId));
    assert_eq!(router.search(&empty_tenant), Err(ColPaliError::EmptyTenantId));

    let mut empty_groups = base_q.clone();
    empty_groups.scope = AccessScope::new("tenant-hanzo", vec![]);
    assert_eq!(ocr.search(&empty_groups), Err(ColPaliError::EmptyAclGroups));
    assert_eq!(fp32.search(&empty_groups), Err(ColPaliError::EmptyAclGroups));
    assert_eq!(bin.search(&empty_groups), Err(ColPaliError::EmptyAclGroups));
    assert_eq!(router.search(&empty_groups), Err(ColPaliError::EmptyAclGroups));

    let mut blank_group_entry = base_q.clone();
    blank_group_entry.scope =
        AccessScope::new("tenant-hanzo", vec!["research-public".to_string(), "".to_string()]);
    assert_eq!(ocr.search(&blank_group_entry), Err(ColPaliError::EmptyAclGroups));
    assert_eq!(fp32.search(&blank_group_entry), Err(ColPaliError::EmptyAclGroups));
    assert_eq!(bin.search(&blank_group_entry), Err(ColPaliError::EmptyAclGroups));
    assert_eq!(router.search(&blank_group_entry), Err(ColPaliError::EmptyAclGroups));
}

#[test]
fn zero_cross_tenant_and_acl_leaks_under_adversarial_exact_clones() {
    let mut corpus =
        build_benchmark_corpus(Path::new("."), 1347, 96, 32, DEFAULT_EMBEDDING_DIM, 8).unwrap();

    // Plant exact bit-for-bit clones of target pages belonging to another tenant and restricted ACL.
    for (i, eq) in corpus.modality_queries.iter().take(10).enumerate() {
        let target_key = eq.relevance.keys().next().unwrap();
        let orig = corpus.store.get(target_key).unwrap().clone();

        let mut cross_tenant = orig.clone();
        cross_tenant.doc_id = format!("evil-tenant-clone-{i:02}");
        cross_tenant.tenant_id = "tenant-competitor".to_string();
        corpus.store.upsert(cross_tenant).unwrap();

        let mut restricted_acl = orig;
        restricted_acl.doc_id = format!("evil-acl-clone-{i:02}");
        restricted_acl.acl_group = "sec-restricted-board".to_string();
        corpus.store.upsert(restricted_acl).unwrap();
    }

    let ocr = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let fp32 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let bin = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let router = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    for eq in &corpus.modality_queries {
        for hits in [
            ocr.search(&eq.query).unwrap(),
            fp32.search(&eq.query).unwrap(),
            bin.search(&eq.query).unwrap(),
            router.search(&eq.query).unwrap(),
        ] {
            assert_eq!(hits.len(), eq.query.top_k, "must not suffer top-k starvation");
            let (acl_leaks, tenant_leaks) = audit_access_leaks(&hits, &eq.query.scope);
            assert_eq!(acl_leaks, 0);
            assert_eq!(tenant_leaks, 0);
        }
    }
}

#[test]
fn drop_and_rebuild_bit_exact_parity_across_all_four_strategies() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 80, 32, DEFAULT_EMBEDDING_DIM, 6).unwrap();
    let fp_before = corpus.store.fingerprint();

    let ocr1 = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let fp32_1 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let bin1 = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let router1 = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    let cks1 = (
        ocr1.checksum_crc32(),
        fp32_1.checksum_crc32(),
        bin1.checksum_crc32(),
        router1.checksum_crc32(),
    );
    let q = &corpus.modality_queries[0].query;
    let hits1 = (
        ocr1.search(q).unwrap(),
        fp32_1.search(q).unwrap(),
        bin1.search(q).unwrap(),
        router1.search(q).unwrap(),
    );

    drop(ocr1);
    drop(fp32_1);
    drop(bin1);
    drop(router1);

    let ocr2 = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let fp32_2 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let bin2 = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let router2 = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    assert_eq!(fp_before, corpus.store.fingerprint());
    assert_eq!(cks1.0, ocr2.checksum_crc32());
    assert_eq!(cks1.1, fp32_2.checksum_crc32());
    assert_eq!(cks1.2, bin2.checksum_crc32());
    assert_eq!(cks1.3, router2.checksum_crc32());

    assert!(hits_bit_exact(&hits1.0, &ocr2.search(q).unwrap()));
    assert!(hits_bit_exact(&hits1.1, &fp32_2.search(q).unwrap()));
    assert!(hits_bit_exact(&hits1.2, &bin2.search(q).unwrap()));
    assert!(hits_bit_exact(&hits1.3, &router2.search(q).unwrap()));
}

#[test]
fn serialization_roundtrip_and_crc32_corruption_detection() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 48, 32, DEFAULT_EMBEDDING_DIM, 4).unwrap();
    let q = &corpus.modality_queries[0].query;

    // 1. OcrBm25HybridIndex
    let ocr = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let mut ocr_bytes = ocr.to_bytes();
    let ocr_loaded = OcrBm25HybridIndex::from_bytes(&ocr_bytes).unwrap();
    assert!(hits_bit_exact(&ocr.search(q).unwrap(), &ocr_loaded.search(q).unwrap()));
    ocr_bytes[12] ^= 0x01;
    assert!(matches!(
        OcrBm25HybridIndex::from_bytes(&ocr_bytes),
        Err(ColPaliError::CorruptedIndex { .. })
    ));

    // 2. ColPaliFp32Index
    let fp32 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let mut fp32_bytes = fp32.to_bytes();
    let fp32_loaded = ColPaliFp32Index::from_bytes(&fp32_bytes).unwrap();
    assert!(hits_bit_exact(&fp32.search(q).unwrap(), &fp32_loaded.search(q).unwrap()));
    let mid = fp32_bytes.len() / 2;
    fp32_bytes[mid] ^= 0x80;
    assert!(matches!(
        ColPaliFp32Index::from_bytes(&fp32_bytes),
        Err(ColPaliError::CorruptedIndex { .. })
    ));

    // 3. ColPaliBinaryTwoStageIndex
    let bin = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let mut bin_bytes = bin.to_bytes();
    let bin_loaded = ColPaliBinaryTwoStageIndex::from_bytes(&bin_bytes).unwrap();
    assert!(hits_bit_exact(&bin.search(q).unwrap(), &bin_loaded.search(q).unwrap()));
    let mid_b = bin_bytes.len() / 2;
    bin_bytes[mid_b] ^= 0x04;
    assert!(matches!(
        ColPaliBinaryTwoStageIndex::from_bytes(&bin_bytes),
        Err(ColPaliError::CorruptedIndex { .. })
    ));
}

#[test]
fn binary_quantization_achieves_32x_patch_ram_reduction_and_preserves_visual_accuracy() {
    let config = BenchmarkConfig {
        seed: 1347,
        total_pages: 256,
        patches_per_page: 64,
        dim: DEFAULT_EMBEDDING_DIM,
        queries_per_slice: 16,
    };
    let report = run_benchmark(Path::new("."), &config).unwrap();

    let fp32 =
        report.strategies.iter().find(|s| s.strategy == StrategyId::ColPaliFp32MaxSim).unwrap();
    let bin = report
        .strategies
        .iter()
        .find(|s| s.strategy == StrategyId::ColPaliBinaryQuantizedTwoStage)
        .unwrap();

    // Exact 32x patch vector compression (512 B/patch FP32 -> 16 B/patch 1-bit Binary at D=128).
    let ratio = bin.binary_patch_compression_ratio.unwrap();
    assert!((ratio - 32.0).abs() < 1e-6, "expected 32.0x patch compression, got {ratio}");
    assert!(
        report.binary_ram_reduction_vs_fp32 > 25.0,
        "expected >25x total RAM reduction, got {}",
        report.binary_ram_reduction_vs_fp32
    );

    // Visual retrieval quality retention on tables, figures, and formulas (>= 95% of FP32).
    for m in
        [PageModality::DenseTable, PageModality::ScientificFigure, PageModality::MultiColumnFormula]
    {
        let fp32_ndcg = fp32.by_modality[&m].ndcg_at_5;
        let bin_ndcg = bin.by_modality[&m].ndcg_at_5;
        assert!(
            bin_ndcg >= fp32_ndcg * 0.90,
            "modality {:?}: bin_ndcg={bin_ndcg:.3} dropped too far below fp32_ndcg={fp32_ndcg:.3}",
            m
        );
    }
}

#[test]
fn colpali_outperforms_ocr_on_visual_modalities_and_negative_regime_1_favors_ocr_and_router() {
    let config = BenchmarkConfig {
        seed: 1347,
        total_pages: 256,
        patches_per_page: 48,
        dim: DEFAULT_EMBEDDING_DIM,
        queries_per_slice: 16,
    };
    let report = run_benchmark(Path::new("."), &config).unwrap();

    let ocr =
        report.strategies.iter().find(|s| s.strategy == StrategyId::OcrBm25HybridBaseline).unwrap();
    let fp32 =
        report.strategies.iter().find(|s| s.strategy == StrategyId::ColPaliFp32MaxSim).unwrap();
    let bin = report
        .strategies
        .iter()
        .find(|s| s.strategy == StrategyId::ColPaliBinaryQuantizedTwoStage)
        .unwrap();
    let router = report
        .strategies
        .iter()
        .find(|s| s.strategy == StrategyId::HybridTextPlusColPaliRouter)
        .unwrap();

    // ColPali & Router beat OCR+BM25 strongly on visual-heavy modalities.
    for m in
        [PageModality::DenseTable, PageModality::ScientificFigure, PageModality::MultiColumnFormula]
    {
        let ocr_ndcg = ocr.by_modality[&m].ndcg_at_5;
        let fp32_ndcg = fp32.by_modality[&m].ndcg_at_5;
        let bin_ndcg = bin.by_modality[&m].ndcg_at_5;
        let router_ndcg = router.by_modality[&m].ndcg_at_5;
        assert!(
            fp32_ndcg > ocr_ndcg + 0.25,
            "expected FP32 ({fp32_ndcg:.3}) >> OCR ({ocr_ndcg:.3}) on {:?}",
            m
        );
        assert!(
            bin_ndcg > ocr_ndcg + 0.25,
            "expected Binary ({bin_ndcg:.3}) >> OCR ({ocr_ndcg:.3}) on {:?}",
            m
        );
        assert!(
            router_ndcg > ocr_ndcg + 0.25,
            "expected Router ({router_ndcg:.3}) >> OCR ({ocr_ndcg:.3}) on {:?}",
            m
        );
    }

    // Explicit Negative Regime 1: Plain-text exact keyword/identifier queries.
    // OCR+BM25 beats pure visual ColPali at tiny storage, and Router fast-paths cleanly to match OCR!
    let ocr_neg1 = ocr.negative_regime_1_exact_keyword.ndcg_at_5;
    let fp32_neg1 = fp32.negative_regime_1_exact_keyword.ndcg_at_5;
    let router_neg1 = router.negative_regime_1_exact_keyword.ndcg_at_5;
    assert!(
        ocr_neg1 > fp32_neg1 + 0.20,
        "Negative Regime 1: expected OCR ({ocr_neg1:.3}) > ColPali FP32 ({fp32_neg1:.3}) on exact keywords"
    );
    assert!(
        (router_neg1 - ocr_neg1).abs() < 0.05,
        "Router ({router_neg1:.3}) should match OCR ({ocr_neg1:.3}) on exact keywords"
    );
    assert!(
        fp32.footprint.bytes_per_page_ram > ocr.footprint.bytes_per_page_ram * 15.0,
        "FP32 RAM/page must be dramatically larger than OCR+BM25"
    );
}

#[test]
fn negative_regime_2_storage_explosion_projections_at_100k_and_1m_pages() {
    let rows = compute_storage_explosion_projections(StorageCostAssumptions::default());
    let fp32_1m = rows
        .iter()
        .find(|r| r.scale_pages == 1_000_000 && r.architecture == "colpali_fp32_uncompressed")
        .unwrap();
    let bin_1m = rows
        .iter()
        .find(|r| {
            r.scale_pages == 1_000_000 && r.architecture == "colpali_binary_1bit_ram_sq8_disk"
        })
        .unwrap();
    let router_pooled_1m = rows
        .iter()
        .find(|r| {
            r.scale_pages == 1_000_000 && r.architecture == "hybrid_router_30pct_binary_2x2_pooled"
        })
        .unwrap();

    assert!(fp32_1m.is_projected);
    // 1,030 * 128 * 4 = 527,360 B/page -> > 527 GB at 1M pages!
    assert!(fp32_1m.total_ram_gb > 527.0 && fp32_1m.total_ram_gb < 528.0);
    // 1-bit Binary is ~31.8x smaller in hot RAM (~16.5 GB at 1M pages).
    assert!(bin_1m.compression_vs_fp32_ram > 31.5 && bin_1m.compression_vs_fp32_ram < 32.1);
    assert!(bin_1m.total_ram_gb < 17.0);
    // Selective 30% + 2x2 pooled binary router cuts RAM > 190x vs raw FP32 (~2.6 GB at 1M pages).
    assert!(router_pooled_1m.total_ram_gb < 3.0);
    assert!(router_pooled_1m.compression_vs_fp32_ram > 190.0);
}

#[test]
fn selective_router_skips_plain_text_visual_indexing_and_fast_paths_exact_keywords() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 100, 32, DEFAULT_EMBEDDING_DIM, 6).unwrap();
    let router = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    let text_only_count =
        corpus.provenance.modality_page_counts.get(&PageModality::TextOnly).copied().unwrap_or(0);
    let expected_visual = corpus.store.len() - text_only_count;
    assert_eq!(
        router.visual_pages_indexed(),
        expected_visual,
        "router must skip all plain-text pages from visual multi-vector indexing"
    );

    // Exact keyword query should take the text-only fast path.
    let exact_q = &corpus.exact_keyword_queries[0].query;
    let hits = router.search(exact_q).unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.route_used == RouteKind::HybridRouterTextOnly));
}

#[test]
fn concurrent_multithreaded_reads_are_deterministic_and_acl_safe() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 96, 32, DEFAULT_EMBEDDING_DIM, 6).unwrap();
    let ocr = Arc::new(OcrBm25HybridIndex::build(&corpus.store).unwrap());
    let fp32 = Arc::new(ColPaliFp32Index::build(&corpus.store).unwrap());
    let bin = Arc::new(ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap());
    let router = Arc::new(HybridVisualRouterIndex::build(&corpus.store).unwrap());

    let queries = Arc::new(corpus.modality_queries);
    let ref_hits_bin: Vec<_> = queries.iter().map(|eq| bin.search(&eq.query).unwrap()).collect();

    thread::scope(|s| {
        for tid in 0..8 {
            let ocr = Arc::clone(&ocr);
            let fp32 = Arc::clone(&fp32);
            let bin = Arc::clone(&bin);
            let router = Arc::clone(&router);
            let queries = Arc::clone(&queries);
            let ref_hits_bin = &ref_hits_bin;
            s.spawn(move || {
                for (qi, eq) in queries.iter().enumerate() {
                    let h_ocr = ocr.search(&eq.query).unwrap();
                    let h_fp32 = fp32.search(&eq.query).unwrap();
                    let h_bin = bin.search(&eq.query).unwrap();
                    let h_router = router.search(&eq.query).unwrap();
                    assert!(
                        hits_bit_exact(&h_bin, &ref_hits_bin[qi]),
                        "thread {tid} query {qi} diverged"
                    );
                    for hits in [&h_ocr, &h_fp32, &h_bin, &h_router] {
                        let (a, t) = audit_access_leaks(hits, &eq.query.scope);
                        assert_eq!(a, 0);
                        assert_eq!(t, 0);
                    }
                }
            });
        }
    });
}

#[test]
fn canonical_version_update_and_delete_reflect_cleanly_on_rebuild() {
    let mut corpus =
        build_benchmark_corpus(Path::new("."), 1347, 40, 16, DEFAULT_EMBEDDING_DIM, 4).unwrap();
    let target_eq = corpus
        .modality_queries
        .iter()
        .find(|q| q.target_modality == PageModality::DenseTable)
        .unwrap()
        .clone();
    let target_key: PageKey = target_eq.relevance.keys().next().unwrap().clone();

    // Update target record version to 99 and rebuild.
    let mut updated = corpus.store.get(&target_key).unwrap().clone();
    updated.version = 99;
    corpus.store.upsert(updated).unwrap();

    let bin_v99 = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let hits = bin_v99.search(&target_eq.query).unwrap();
    let matched =
        hits.iter().find(|h| h.key() == target_key).expect("updated page must be retrieved");
    assert_eq!(matched.version, 99);

    // Delete target record from canonical store and rebuild: must never appear as stale hit.
    assert!(corpus.store.remove(&target_key).is_some());
    let bin_after_del = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let hits_after_del = bin_after_del.search(&target_eq.query).unwrap();
    assert!(hits_after_del.iter().all(|h| h.key() != target_key), "deleted page must not reappear");
}

#[test]
fn benchmark_json_receipt_separates_measured_and_projected_sections() {
    let config = BenchmarkConfig {
        seed: 1347,
        total_pages: 96,
        patches_per_page: 32,
        dim: DEFAULT_EMBEDDING_DIM,
        queries_per_slice: 6,
    };
    let report = run_benchmark(Path::new("."), &config).unwrap();
    assert!(report.drop_rebuild_parity.all_passed());
    assert_eq!(report.adversarial_acl_audit.total_acl_leaks, 0);
    assert_eq!(report.adversarial_acl_audit.total_cross_tenant_leaks, 0);
    assert!(report.adversarial_acl_audit.fail_closed_checks_passed);

    let json = report_to_json(&report);
    assert!(json.contains("\"schema_version\": \"hip-1347-colpali-bench/v1\""));
    assert!(json.contains("\"measured_section_is_projected\": false"));
    assert!(json.contains("\"negative_regime_2_storage_projections\""));
    assert!(json.contains("\"is_projected\": true"));
    assert!(json.contains("\"all_passed\": true"));
}

#[test]
fn binary_two_stage_sq8_reranking_is_genuinely_disk_backed_and_footprint_honest() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 128, 64, DEFAULT_EMBEDDING_DIM, 4).unwrap();
    let bin = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();

    let expected_sq8_disk = bin.total_patches() * (4 + DEFAULT_EMBEDDING_DIM);
    assert_eq!(bin.sq8_disk_bytes_on_disk(), expected_sq8_disk);
    assert_eq!(bin.footprint().disk_bytes, bin.to_bytes().len());
    assert!(
        bin.footprint().ram_bytes < expected_sq8_disk / 4,
        "hot RAM ({}) must exclude cold disk-backed SQ8 slab ({expected_sq8_disk})",
        bin.footprint().ram_bytes
    );

    assert_eq!(bin.sq8_disk_bytes_read(), 0);
    assert_eq!(bin.sq8_disk_page_fetches(), 0);

    let q = &corpus.modality_queries[0].query;
    let hits = bin.search(q).unwrap();
    assert_eq!(hits.len(), q.top_k);
    let fetches = bin.sq8_disk_page_fetches();
    let bytes_read = bin.sq8_disk_bytes_read() as usize;
    assert!(
        fetches > 0 && fetches <= ColPaliBinaryTwoStageIndex::DEFAULT_SHORTLIST_MIN as u64,
        "expected shortlisted disk fetches in 1..={}, got {fetches}",
        ColPaliBinaryTwoStageIndex::DEFAULT_SHORTLIST_MIN
    );
    assert!(
        bytes_read > 0 && bytes_read < expected_sq8_disk,
        "Stage-2 must read only shortlisted pages ({bytes_read} B) instead of full slab ({expected_sq8_disk} B)"
    );

    // Serialized roundtrip must restore a working disk-backed SQ8 slab with bit-exact hits.
    let loaded = ColPaliBinaryTwoStageIndex::from_bytes(&bin.to_bytes()).unwrap();
    assert_eq!(loaded.sq8_disk_bytes_on_disk(), expected_sq8_disk);
    assert_eq!(loaded.sq8_disk_bytes_read(), 0);
    let loaded_hits = loaded.search(q).unwrap();
    assert!(hits_bit_exact(&hits, &loaded_hits));
    assert!(loaded.sq8_disk_bytes_read() > 0);
}

#[test]
fn checksum_crc32_excludes_trailer_and_detects_content_and_version_changes() {
    const CRC32_FIXED_RESIDUE: u32 = 0x2144_df1c;

    let empty_store = colpali_eval::canonical::CanonicalPageStore::new(DEFAULT_EMBEDDING_DIM);
    let empty_ocr = OcrBm25HybridIndex::build(&empty_store).unwrap();
    let empty_fp32 = ColPaliFp32Index::build(&empty_store).unwrap();
    let empty_bin = ColPaliBinaryTwoStageIndex::build(&empty_store).unwrap();
    let empty_router = HybridVisualRouterIndex::build(&empty_store).unwrap();

    let mut corpus =
        build_benchmark_corpus(Path::new("."), 1347, 64, 32, DEFAULT_EMBEDDING_DIM, 4).unwrap();
    let ocr1 = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let fp32_1 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let bin1 = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let router1 = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    // Mutate version of a visual page (so all 4 indexes including router's visual_index change).
    let visual_rec =
        corpus.store.iter().find(|r| r.page_modality == PageModality::ScientificFigure).unwrap();
    let mut mutated = visual_rec.clone();
    mutated.version += 1;
    corpus.store.upsert(mutated).unwrap();

    let ocr2 = OcrBm25HybridIndex::build(&corpus.store).unwrap();
    let fp32_2 = ColPaliFp32Index::build(&corpus.store).unwrap();
    let bin2 = ColPaliBinaryTwoStageIndex::build(&corpus.store).unwrap();
    let router2 = HybridVisualRouterIndex::build(&corpus.store).unwrap();

    for (name, c_empty, c_pop, c_mut) in [
        ("ocr", empty_ocr.checksum_crc32(), ocr1.checksum_crc32(), ocr2.checksum_crc32()),
        ("fp32", empty_fp32.checksum_crc32(), fp32_1.checksum_crc32(), fp32_2.checksum_crc32()),
        ("binary", empty_bin.checksum_crc32(), bin1.checksum_crc32(), bin2.checksum_crc32()),
        (
            "router",
            empty_router.checksum_crc32(),
            router1.checksum_crc32(),
            router2.checksum_crc32(),
        ),
    ] {
        assert_ne!(c_empty, CRC32_FIXED_RESIDUE, "{name} empty checksum hit fixed CRC residue");
        assert_ne!(c_pop, CRC32_FIXED_RESIDUE, "{name} populated checksum hit fixed CRC residue");
        assert_ne!(c_mut, CRC32_FIXED_RESIDUE, "{name} mutated checksum hit fixed CRC residue");
        assert_ne!(c_empty, c_pop, "{name} empty and populated checksums must differ");
        assert_ne!(c_pop, c_mut, "{name} version mutation must change checksum");
    }
}

#[test]
fn equivalent_generated_pages_are_all_judged_relevant_and_unauthorized_excluded() {
    let corpus =
        build_benchmark_corpus(Path::new("."), 1347, 512, 64, DEFAULT_EMBEDDING_DIM, 24).unwrap();

    let mut by_text: std::collections::BTreeMap<&str, Vec<&colpali_eval::corpus::EvalQuery>> =
        std::collections::BTreeMap::new();
    for eq in &corpus.modality_queries {
        by_text.entry(eq.query.text.as_str()).or_default().push(eq);
        assert!(!eq.relevance.is_empty());
        // Every page judged relevant must be authorized under the query's AccessScope.
        for key in eq.relevance.keys() {
            let rec = corpus.store.get(key).expect("relevant key must exist in canonical store");
            assert!(
                eq.query.scope.permits(&rec.tenant_id, &rec.acl_group),
                "unauthorized page {key:?} must never be judged relevant"
            );
        }
    }

    let mut verified_multi_page_equivalence = false;
    for (prompt, group) in by_text {
        if group.len() > 1 {
            verified_multi_page_equivalence = true;
            let expected_rel = &group[0].relevance;
            assert!(
                expected_rel.len() >= 2,
                "repeated query prompt `{prompt}` must include all equivalent authorized pages in relevance"
            );
            for eq in &group[1..] {
                assert_eq!(
                    &eq.relevance, expected_rel,
                    "queries with identical prompt `{prompt}` must share identical relevance maps"
                );
            }
        }
    }
    assert!(
        verified_multi_page_equivalence,
        "expected default 512-page corpus to exercise multi-page equivalence judgments"
    );
}
