// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive integration & security test suite for `vector-ann-bench`
//! (`hanzoai/index#15` / HIP-1347).

use std::sync::Arc;

use vector_ann_bench::{
    build_benchmark_corpus, normalize_l2, project_scale_matrix, recall_at_k, run_full_benchmark,
    AccessContext, AnnError, CanonicalStore, CanonicalVectorRecord, CorpusConfig,
    ExactFlatBaseline, InstrumentedObjectStore, IvfPqConfig, LocalMmapVamanaIndex,
    ObjectNativeIvfPqIndex, ObjectStoreCostModel, ObjectStoreRandomGraphIndex,
    PrefilterInvertedFallback, ProductQuantizer, SplitMix64, Sq8Quantizer, VamanaConfig,
};

fn make_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut v = vec![0.0_f32; dim];
    for x in &mut v {
        *x = rng.next_normal_f32();
    }
    normalize_l2(&mut v).unwrap();
    v
}

#[test]
fn fail_closed_acl_rejects_empty_tenant_and_empty_groups() {
    let dim = 32;
    let mut store = CanonicalStore::new(dim).unwrap();
    store
        .upsert(CanonicalVectorRecord {
            source_id: "doc:1".into(),
            version: 1,
            tenant_id: "tenant-a".into(),
            acl_group: "eng".into(),
            filter_tags: vec!["lang:rust".into()],
            vector: make_unit_vec(dim, 1),
        })
        .unwrap();

    let flat = ExactFlatBaseline::build_from_canonical(&store).unwrap();
    let vamana = LocalMmapVamanaIndex::build_from_canonical(
        &store,
        VamanaConfig { max_degree: 8, build_beam: 16, search_beam: 16, ..VamanaConfig::default() },
        None,
    )
    .unwrap();
    let obj_store =
        Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 1024 * 1024));
    let ivf = ObjectNativeIvfPqIndex::build_from_canonical(
        &store,
        Arc::clone(&obj_store),
        IvfPqConfig {
            num_partitions: 4,
            nprobe: 2,
            pq_subspaces: 4,
            pq_k_sub: 8,
            ..IvfPqConfig::default()
        },
    )
    .unwrap();
    let neg_store = Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 0));
    let neg = ObjectStoreRandomGraphIndex::build_from_canonical(
        &store,
        neg_store,
        VamanaConfig { max_degree: 8, build_beam: 16, search_beam: 16, ..VamanaConfig::default() },
    )
    .unwrap();
    let pre = PrefilterInvertedFallback::build_from_canonical(&store).unwrap();

    let q = make_unit_vec(dim, 99);
    let bad_tenant = AccessContext::new("   ", vec!["eng".into()], None);
    let bad_groups = AccessContext::new("tenant-a", vec![], None);

    assert_eq!(flat.search(&q, 5, &bad_tenant), Err(AnnError::EmptyTenantId));
    assert_eq!(flat.search(&q, 5, &bad_groups), Err(AnnError::EmptyAclGroups));
    assert_eq!(vamana.search(&q, 5, &bad_tenant), Err(AnnError::EmptyTenantId));
    assert_eq!(vamana.search(&q, 5, &bad_groups), Err(AnnError::EmptyAclGroups));
    assert_eq!(ivf.search(&q, 5, &bad_tenant), Err(AnnError::EmptyTenantId));
    assert_eq!(ivf.search(&q, 5, &bad_groups), Err(AnnError::EmptyAclGroups));
    assert_eq!(neg.search(&q, 5, &bad_tenant), Err(AnnError::EmptyTenantId));
    assert_eq!(neg.search(&q, 5, &bad_groups), Err(AnnError::EmptyAclGroups));
    assert_eq!(pre.search_with_stats(&q, 5, &bad_tenant), Err(AnnError::EmptyTenantId));
    assert_eq!(pre.search_with_stats(&q, 5, &bad_groups), Err(AnnError::EmptyAclGroups));
}

#[test]
fn zero_cross_tenant_and_acl_leaks_under_adversarial_exact_clone_vectors() {
    let dim = 32;
    let target_vec = make_unit_vec(dim, 777);
    let mut store = CanonicalStore::new(dim).unwrap();

    // Plant 15 adversarial exact-distance-0.0 clones in ANOTHER tenant and a restricted ACL group.
    for i in 0..10 {
        store
            .upsert(CanonicalVectorRecord {
                source_id: format!("secret:other-tenant-{i}"),
                version: 1,
                tenant_id: "tenant-enterprise".into(),
                acl_group: "eng-core".into(),
                filter_tags: vec!["sel:05".into()],
                vector: target_vec.clone(),
            })
            .unwrap();
    }
    for i in 0..5 {
        store
            .upsert(CanonicalVectorRecord {
                source_id: format!("secret:restricted-acl-{i}"),
                version: 1,
                tenant_id: "tenant-hanzo".into(),
                acl_group: "sec-ops".into(),
                filter_tags: vec!["sel:05".into()],
                vector: target_vec.clone(),
            })
            .unwrap();
    }

    // Plant 40 authorized records slightly farther from target_vec.
    let mut rng = SplitMix64::new(42);
    for i in 0..40 {
        let mut v = target_vec.clone();
        for x in &mut v {
            *x += rng.next_normal_f32() * 0.25;
        }
        normalize_l2(&mut v).unwrap();
        store
            .upsert(CanonicalVectorRecord {
                source_id: format!("allowed:hanzo-{i}"),
                version: 1,
                tenant_id: "tenant-hanzo".into(),
                acl_group: "eng-core".into(),
                filter_tags: vec!["sel:05".into()],
                vector: v,
            })
            .unwrap();
    }

    let ctx = AccessContext::new("tenant-hanzo", vec!["eng-core".into()], Some("sel:05".into()));
    let vcfg =
        VamanaConfig { max_degree: 12, build_beam: 24, search_beam: 32, ..VamanaConfig::default() };
    let icfg = IvfPqConfig {
        num_partitions: 8,
        nprobe: 4,
        pq_subspaces: 4,
        pq_k_sub: 16,
        ..IvfPqConfig::default()
    };

    let flat = ExactFlatBaseline::build_from_canonical(&store).unwrap();
    let vamana = LocalMmapVamanaIndex::build_from_canonical(&store, vcfg, None).unwrap();
    let ivf_store =
        Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 1024 * 1024));
    let ivf = ObjectNativeIvfPqIndex::build_from_canonical(&store, ivf_store, icfg).unwrap();
    let neg_store = Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 0));
    let neg = ObjectStoreRandomGraphIndex::build_from_canonical(&store, neg_store, vcfg).unwrap();
    let pre = PrefilterInvertedFallback::build_from_canonical(&store).unwrap();

    for (name, hits) in [
        ("exact_flat", flat.search(&target_vec, 10, &ctx).unwrap()),
        ("local_mmap_vamana", vamana.search(&target_vec, 10, &ctx).unwrap()),
        ("object_native_ivf_pq", ivf.search(&target_vec, 10, &ctx).unwrap()),
        ("object_store_random_graph_hops", neg.search(&target_vec, 10, &ctx).unwrap()),
        ("prefilter", pre.search_with_stats(&target_vec, 10, &ctx).unwrap().0),
    ] {
        assert!(!hits.is_empty(), "{name} should still return authorized hits despite distractors");
        let audit = store.audit_hits(&ctx, &hits).unwrap();
        assert!(audit.is_clean(), "{name} leaked unauthorized or invalid records: {audit:?}");
    }
}

#[test]
fn canonical_derived_drop_and_bit_exact_rebuild() {
    let corpus = build_benchmark_corpus(
        None,
        &CorpusConfig {
            seed: 1347,
            dim: 32,
            target_vectors: 320,
            num_clusters: 8,
            queries_per_selectivity: 12,
            downstream_task_count: 0,
        },
    )
    .unwrap();

    let fp_before = corpus.store.canonical_fingerprint();
    let vcfg = VamanaConfig {
        max_degree: 12,
        build_beam: 24,
        search_beam: 32,
        seed: 1347,
        ..VamanaConfig::default()
    };
    let icfg = IvfPqConfig {
        num_partitions: 16,
        nprobe: 4,
        pq_subspaces: 4,
        pq_k_sub: 16,
        seed: 1347,
        ..IvfPqConfig::default()
    };

    let mut vamana = LocalMmapVamanaIndex::build_from_canonical(&corpus.store, vcfg, None).unwrap();
    vamana.verify_on_disk_integrity().unwrap();
    let bytes_before = vamana.on_disk_bytes().unwrap();
    let q = &corpus.queries_100[0];
    let v_hits_before = vamana.search(&q.vector, 10, &q.access).unwrap();

    vamana.rebuild_from_canonical(&corpus.store).unwrap();
    let bytes_after = vamana.on_disk_bytes().unwrap();
    let v_hits_after = vamana.search(&q.vector, 10, &q.access).unwrap();
    assert_eq!(bytes_before, bytes_after, "HVAN binary must rebuild bit-exact");
    assert_eq!(v_hits_before, v_hits_after, "Vamana hits must match bit-exact");

    let obj_store =
        Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 16 * 1024 * 1024));
    let mut ivf =
        ObjectNativeIvfPqIndex::build_from_canonical(&corpus.store, obj_store, icfg).unwrap();
    let i_hits_before = ivf.search(&q.vector, 10, &q.access).unwrap();
    ivf.rebuild_from_canonical(&corpus.store).unwrap();
    let i_hits_after = ivf.search(&q.vector, 10, &q.access).unwrap();
    assert_eq!(i_hits_before, i_hits_after, "IVF-PQ hits must match bit-exact");

    assert_eq!(
        fp_before,
        corpus.store.canonical_fingerprint(),
        "Canonical store must never be mutated by derived rebuilds"
    );
}

#[test]
fn fresh_updates_tombstone_deletes_and_compaction_correctness() {
    let corpus = build_benchmark_corpus(
        None,
        &CorpusConfig {
            seed: 2026,
            dim: 32,
            target_vectors: 300,
            num_clusters: 8,
            queries_per_selectivity: 15,
            downstream_task_count: 0,
        },
    )
    .unwrap();

    let vcfg = VamanaConfig {
        max_degree: 12,
        build_beam: 24,
        search_beam: 32,
        seed: 2026,
        ..VamanaConfig::default()
    };
    let icfg = IvfPqConfig {
        num_partitions: 16,
        nprobe: 6,
        pq_subspaces: 4,
        pq_k_sub: 16,
        seed: 2026,
        ..IvfPqConfig::default()
    };

    let mut mutated_canonical = corpus.store.clone();
    let mut vamana = LocalMmapVamanaIndex::build_from_canonical(&corpus.store, vcfg, None).unwrap();
    let obj_store =
        Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 16 * 1024 * 1024));
    let mut ivf =
        ObjectNativeIvfPqIndex::build_from_canonical(&corpus.store, obj_store, icfg).unwrap();

    for rec in &corpus.mutation_inserts {
        mutated_canonical.upsert(rec.clone()).unwrap();
        vamana.upsert(rec).unwrap();
        ivf.upsert(rec).unwrap();
    }
    for rec in &corpus.mutation_updates {
        mutated_canonical.upsert(rec.clone()).unwrap();
        vamana.upsert(rec).unwrap();
        ivf.upsert(rec).unwrap();
    }
    for id in &corpus.mutation_delete_ids {
        mutated_canonical.delete(id);
        vamana.delete(id);
        ivf.delete(id).unwrap();
    }

    // Query directly using a freshly inserted vector -> MUST rank #1 immediately (read-your-writes)!
    let fresh_probe = &corpus.mutation_inserts[0];
    let fresh_ctx = AccessContext::new(
        fresh_probe.tenant_id.clone(),
        vec![fresh_probe.acl_group.clone()],
        None,
    );
    let v_fresh = vamana.search(&fresh_probe.vector, 5, &fresh_ctx).unwrap();
    let i_fresh = ivf.search(&fresh_probe.vector, 5, &fresh_ctx).unwrap();
    assert_eq!(v_fresh[0].source_id, fresh_probe.source_id);
    assert_eq!(i_fresh[0].source_id, fresh_probe.source_id);

    // Verify 0 tombstones, 0 stale version-1 hits, and 0 ACL leaks across all queries.
    for q in &corpus.queries_100 {
        let vh = vamana.search(&q.vector, 10, &q.access).unwrap();
        let ih = ivf.search(&q.vector, 10, &q.access).unwrap();
        assert!(mutated_canonical.audit_hits(&q.access, &vh).unwrap().is_clean());
        assert!(mutated_canonical.audit_hits(&q.access, &ih).unwrap().is_clean());
    }

    // Compact both indexes and verify tombstones reclaimed == expected and integrity holds.
    let vc = vamana.compact(&mutated_canonical).unwrap();
    let ic = ivf.compact(&mutated_canonical).unwrap();
    assert!(vc.tombstones_reclaimed > 0);
    assert!(ic.tombstones_reclaimed > 0);
    assert_eq!(vamana.footprint().tombstoned_vectors, 0);
    assert_eq!(ivf.footprint().tombstoned_vectors, 0);
    vamana.verify_on_disk_integrity().unwrap();
}

#[test]
fn object_native_ivf_pq_crushes_random_graph_hops_negative_control() {
    let corpus = build_benchmark_corpus(
        None,
        &CorpusConfig {
            seed: 1347,
            dim: 64,
            target_vectors: 1024,
            num_clusters: 16,
            queries_per_selectivity: 20,
            downstream_task_count: 0,
        },
    )
    .unwrap();
    let cost = ObjectStoreCostModel::default();
    let vcfg = VamanaConfig {
        max_degree: 20,
        build_beam: 40,
        search_beam: 48,
        seed: 1347,
        ..VamanaConfig::default()
    };
    let icfg = IvfPqConfig {
        num_partitions: 32,
        nprobe: 6,
        pq_subspaces: 8,
        pq_k_sub: 32,
        seed: 1347,
        ..IvfPqConfig::default()
    };

    let ivf_store = Arc::new(InstrumentedObjectStore::new(cost, 64 * 1024 * 1024));
    let ivf = ObjectNativeIvfPqIndex::build_from_canonical(&corpus.store, ivf_store, icfg).unwrap();
    let neg_store = Arc::new(InstrumentedObjectStore::new(cost, 0));
    let neg =
        ObjectStoreRandomGraphIndex::build_from_canonical(&corpus.store, neg_store, vcfg).unwrap();

    let mut ivf_rtts = 0u64;
    let mut ivf_gets = 0u64;
    let mut ivf_io_us = 0u64;
    let mut neg_rtts = 0u64;
    let mut neg_gets = 0u64;
    let mut neg_io_us = 0u64;

    for q in &corpus.queries_100 {
        ivf.clear_local_cache().unwrap();
        let b = ivf.io_stats();
        let _ = ivf.search(&q.vector, 10, &q.access).unwrap();
        let d = ivf.io_stats().delta_since(b);
        ivf_rtts += d.sequential_rtts;
        ivf_gets += d.total_requests();
        ivf_io_us += d.simulated_io_us;

        neg.clear_local_cache().unwrap();
        let b2 = neg.io_stats();
        let _ = neg.search(&q.vector, 10, &q.access).unwrap();
        let d2 = neg.io_stats().delta_since(b2);
        neg_rtts += d2.sequential_rtts;
        neg_gets += d2.total_requests();
        neg_io_us += d2.simulated_io_us;
    }

    let n = corpus.queries_100.len() as f64;
    let avg_ivf_rtt = ivf_rtts as f64 / n;
    let avg_neg_rtt = neg_rtts as f64 / n;
    let avg_ivf_gets = ivf_gets as f64 / n;
    let avg_neg_gets = neg_gets as f64 / n;

    // Object-native IVF-PQ MUST execute 1 bounded batch RTT on cold cache!
    assert!((avg_ivf_rtt - 1.0).abs() < 1e-6, "expected 1 RTT, got {avg_ivf_rtt}");
    // Random graph hops over S3 MUST incur >= 10x sequential RTTs and >= 10x GET amplification!
    assert!(
        avg_neg_rtt >= avg_ivf_rtt * 10.0,
        "expected >=10x RTT amplification: neg={avg_neg_rtt}, ivf={avg_ivf_rtt}"
    );
    assert!(
        avg_neg_gets >= avg_ivf_gets * 10.0,
        "expected >=10x S3 GET amplification: neg={avg_neg_gets}, ivf={avg_ivf_gets}"
    );
    assert!(
        neg_io_us >= ivf_io_us * 10,
        "expected >=10x simulated S3 I/O latency: neg={neg_io_us}us, ivf={ivf_io_us}us"
    );
}

#[test]
fn cold_vs_warm_cache_regime_separation() {
    let corpus = build_benchmark_corpus(
        None,
        &CorpusConfig {
            seed: 1347,
            dim: 32,
            target_vectors: 400,
            num_clusters: 8,
            queries_per_selectivity: 10,
            downstream_task_count: 0,
        },
    )
    .unwrap();
    let vamana = LocalMmapVamanaIndex::build_from_canonical(
        &corpus.store,
        VamanaConfig { max_degree: 12, build_beam: 24, search_beam: 32, ..VamanaConfig::default() },
        None,
    )
    .unwrap();

    let q = &corpus.queries_100[0];
    vamana.evict_page_cache();
    let b_cold = vamana.io_stats();
    let _ = vamana.search(&q.vector, 10, &q.access).unwrap();
    let cold_delta = vamana.io_stats().delta_since(b_cold);
    assert!(cold_delta.page_faults > 0, "cold cache must incur disk page faults");

    vamana.warm_page_cache().unwrap();
    let b_warm = vamana.io_stats();
    let _ = vamana.search(&q.vector, 10, &q.access).unwrap();
    let warm_delta = vamana.io_stats().delta_since(b_warm);
    assert_eq!(warm_delta.page_faults, 0, "warm cache must incur 0 page faults");

    let ivf_store =
        Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 64 * 1024 * 1024));
    let ivf = ObjectNativeIvfPqIndex::build_from_canonical(
        &corpus.store,
        ivf_store,
        IvfPqConfig {
            num_partitions: 16,
            nprobe: 4,
            pq_subspaces: 4,
            pq_k_sub: 16,
            ..IvfPqConfig::default()
        },
    )
    .unwrap();
    ivf.clear_local_cache().unwrap();
    let b1 = ivf.io_stats();
    let _ = ivf.search(&q.vector, 10, &q.access).unwrap();
    let d1 = ivf.io_stats().delta_since(b1);
    assert_eq!(d1.sequential_rtts, 1);
    assert!(d1.range_get_requests >= 1);

    let b2 = ivf.io_stats();
    let _ = ivf.search(&q.vector, 10, &q.access).unwrap();
    let d2 = ivf.io_stats().delta_since(b2);
    assert_eq!(d2.total_requests(), 0, "warm IVF cache must issue 0 S3 GETs");
}

#[test]
fn ultra_selective_1pct_prefilter_fallback_beats_blind_postfilter() {
    let corpus = build_benchmark_corpus(
        None,
        &CorpusConfig {
            seed: 1347,
            dim: 64,
            target_vectors: 2000,
            num_clusters: 16,
            queries_per_selectivity: 20,
            downstream_task_count: 0,
        },
    )
    .unwrap();

    let exact = ExactFlatBaseline::build_from_canonical(&corpus.store).unwrap();
    let vamana =
        LocalMmapVamanaIndex::build_from_canonical(&corpus.store, VamanaConfig::default(), None)
            .unwrap();
    let prefilter = PrefilterInvertedFallback::build_from_canonical(&corpus.store).unwrap();

    let mut blind_recalls = Vec::new();
    let mut pre_recalls = Vec::new();
    for q in &corpus.queries_01 {
        let gt = exact.search(&q.vector, 10, &q.access).unwrap();
        let blind = vamana.search_blind_postfilter(&q.vector, 10, 32, &q.access).unwrap();
        let (pre_hits, _scanned) = prefilter.search_with_stats(&q.vector, 10, &q.access).unwrap();
        blind_recalls.push(recall_at_k(&gt, &blind, 10));
        pre_recalls.push(recall_at_k(&gt, &pre_hits, 10));
    }

    let blind_mean = blind_recalls.iter().sum::<f64>() / blind_recalls.len() as f64;
    let pre_mean = pre_recalls.iter().sum::<f64>() / pre_recalls.len() as f64;
    assert!((pre_mean - 1.0).abs() < 1e-9, "pre-filter fallback must hit 1.0 Recall@10");
    assert!(
        pre_mean > blind_mean + 0.30,
        "pre-filter ({pre_mean:.3}) must decisively beat blind post-filter ({blind_mean:.3}) at 1%"
    );
}

#[test]
fn quantizers_and_object_store_immutability_invariants() {
    let dim = 32;
    let vecs: Vec<Vec<f32>> = (0..128).map(|i| make_unit_vec(dim, i + 10)).collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let sq8 = Sq8Quantizer::fit(dim, &refs).unwrap();
    let mut raw = Vec::new();
    sq8.write_to_buf(&mut raw);
    let sq8_roundtrip = Sq8Quantizer::read_from_slice(dim, &raw).unwrap();
    assert_eq!(sq8, sq8_roundtrip);

    let code = sq8.encode(&vecs[0]);
    assert_eq!(code.len(), dim);
    let self_dist = sq8.asymmetric_l2_sq(&vecs[0], &code);
    assert!(self_dist < 0.01, "SQ8 self reconstruction error too large: {self_dist}");

    let pq = ProductQuantizer::train(dim, 8, 16, &refs, 8, 1347).unwrap();
    let adc = pq.build_adc_table(&vecs[0]);
    let pq_code = pq.encode(&vecs[0]);
    assert_eq!(pq_code.len(), 8);
    assert!(pq.adc_distance(&adc, &pq_code).is_finite());

    // Immutable ObjectStore must reject duplicate key overwrites.
    let store = InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 1024);
    store.put_immutable("seg/v1.bin", vec![1, 2, 3, 4]).unwrap();
    assert!(store.put_immutable("seg/v1.bin", vec![5, 6]).is_err());
    assert!(store.get_range("seg/v1.bin", 2, 10).is_err());
}

#[test]
fn scale_projections_1m_10m_100m_monotonic_and_labeled_projected() {
    let rows = project_scale_matrix(768, ObjectStoreCostModel::default(), 21.0, 260.0);
    assert_eq!(rows.len(), 12);
    assert!(rows.iter().all(|r| r.is_projected));

    let ivf_100m = rows
        .iter()
        .find(|r| r.architecture == "object_native_ivf_pq" && r.num_vectors == 100_000_000)
        .unwrap();
    let neg_100m = rows
        .iter()
        .find(|r| {
            r.architecture == "object_store_random_graph_hops" && r.num_vectors == 100_000_000
        })
        .unwrap();
    let flat_100m = rows
        .iter()
        .find(|r| r.architecture == "exact_flat_baseline" && r.num_vectors == 100_000_000)
        .unwrap();

    assert!(ivf_100m.total_ram_gb < flat_100m.total_ram_gb * 0.02);
    assert!(neg_100m.est_cold_p99_ms > ivf_100m.est_cold_p99_ms * 10.0);
    assert!(
        neg_100m.s3_get_cost_per_1m_queries_usd > ivf_100m.s3_get_cost_per_1m_queries_usd * 10.0
    );
}

#[test]
fn end_to_end_benchmark_runner_produces_clean_receipt() {
    let report = run_full_benchmark(
        None,
        &CorpusConfig {
            seed: 1347,
            dim: 64,
            target_vectors: 512,
            num_clusters: 16,
            queries_per_selectivity: 12,
            downstream_task_count: 0,
        },
        ObjectStoreCostModel::default(),
    )
    .unwrap();

    assert!(report.rebuild_parity.canonical_unchanged);
    assert!(report.rebuild_parity.vamana_binary_bit_exact);
    assert!(report.rebuild_parity.vamana_hits_bit_exact);
    assert!(report.rebuild_parity.ivf_pq_hits_bit_exact);
    assert!(report.rebuild_parity.exact_flat_hits_bit_exact);

    for slice in &report.measured_slices {
        assert_eq!(slice.acl_leaks, 0);
        assert_eq!(slice.filter_violations, 0);
        assert_eq!(slice.stale_hits, 0);
        assert!(!slice.is_projected);
    }
    for m in &report.mutations {
        assert_eq!(m.post_mutation_acl_leaks, 0);
        assert_eq!(m.post_mutation_stale_or_deleted_hits, 0);
    }

    assert!(!report.measured_slices.is_empty());
    assert!(report.measured_slices.iter().all(|s| s.query_count > 0));
    assert_eq!(report.ultra_selective_1pct.len(), 4);

    let json = report.to_json().to_pretty_string();
    assert!(json.contains("\"hip-1347-ann-bench/v1\""));
    assert!(json.contains("\"object_native_ivf_pq\""));
    assert!(json.contains("\"object_store_random_graph_hops\""));
}

#[test]
fn vamana_on_disk_crc32_detects_bit_corruption() {
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};

    let mut store = CanonicalStore::new(16).unwrap();
    for i in 0..32 {
        store
            .upsert(CanonicalVectorRecord {
                source_id: format!("doc:{i}"),
                version: 1,
                tenant_id: "tenant-hanzo".into(),
                acl_group: "eng-core".into(),
                filter_tags: vec!["sel:100".into()],
                vector: make_unit_vec(16, 100 + i as u64),
            })
            .unwrap();
    }

    let vamana = LocalMmapVamanaIndex::build_from_canonical(
        &store,
        VamanaConfig { max_degree: 8, build_beam: 16, search_beam: 16, ..VamanaConfig::default() },
        None,
    )
    .unwrap();

    vamana.verify_on_disk_integrity().unwrap();

    // Flip 1 byte in the on-disk HVAN file and verify CRC32 catches corruption.
    let mut f = OpenOptions::new().read(true).write(true).open(vamana.backing_path()).unwrap();
    f.seek(SeekFrom::Start(44)).unwrap();
    f.write_all(&[0xff]).unwrap();
    f.flush().unwrap();

    assert!(
        matches!(vamana.verify_on_disk_integrity(), Err(AnnError::CorruptFormat(_))),
        "CRC32 integrity check must catch single-byte corruption"
    );
}

#[test]
fn object_store_footprints_account_for_resident_metadata_and_id_maps() {
    use vector_ann_bench::DerivedRecordMeta;

    let corpus = build_benchmark_corpus(
        None,
        &CorpusConfig {
            seed: 1347,
            dim: 64,
            target_vectors: 256,
            num_clusters: 8,
            queries_per_selectivity: 4,
            downstream_task_count: 0,
        },
    )
    .unwrap();

    let expected_meta_bytes: usize = corpus
        .store
        .snapshot_live()
        .iter()
        .map(|r| DerivedRecordMeta::from_record(r).memory_bytes())
        .sum();
    let expected_fp32_vec_bytes = corpus.total_vectors * corpus.dim * std::mem::size_of::<f32>();

    let flat = ExactFlatBaseline::build_from_canonical(&corpus.store).unwrap();
    let vamana = LocalMmapVamanaIndex::build_from_canonical(
        &corpus.store,
        VamanaConfig { max_degree: 12, build_beam: 24, search_beam: 24, ..VamanaConfig::default() },
        None,
    )
    .unwrap();
    vamana.evict_page_cache();

    let ivf_store =
        Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 16 * 1024 * 1024));
    let mut ivf = ObjectNativeIvfPqIndex::build_from_canonical(
        &corpus.store,
        ivf_store,
        IvfPqConfig {
            num_partitions: 16,
            nprobe: 4,
            pq_subspaces: 8,
            pq_k_sub: 16,
            ..IvfPqConfig::default()
        },
    )
    .unwrap();
    ivf.clear_local_cache().unwrap();

    let neg_store = Arc::new(InstrumentedObjectStore::new(ObjectStoreCostModel::default(), 0));
    let neg = ObjectStoreRandomGraphIndex::build_from_canonical(
        &corpus.store,
        neg_store,
        VamanaConfig { max_degree: 12, build_beam: 24, search_beam: 24, ..VamanaConfig::default() },
    )
    .unwrap();

    let flat_fp = flat.footprint();
    let vamana_fp = vamana.footprint();
    let ivf_fp = ivf.footprint();
    let neg_fp = neg.footprint();

    assert!(
        flat_fp.ram_bytes >= expected_meta_bytes + expected_fp32_vec_bytes,
        "flat RAM ({}) must include resident metadata ({expected_meta_bytes}) + FP32 vectors ({expected_fp32_vec_bytes})",
        flat_fp.ram_bytes
    );
    assert!(
        vamana_fp.ram_bytes >= expected_meta_bytes + corpus.total_vectors,
        "vamana RAM ({}) must include resident metadata ({expected_meta_bytes}) + tombstones/maps",
        vamana_fp.ram_bytes
    );
    assert!(
        ivf_fp.ram_bytes >= expected_meta_bytes + corpus.total_vectors,
        "object_native_ivf_pq RAM ({}) must include resident metadata ({expected_meta_bytes}) + ID maps + tombstones",
        ivf_fp.ram_bytes
    );
    assert!(
        neg_fp.ram_bytes >= expected_meta_bytes,
        "object_store_random_graph_hops RAM ({}) must include resident metadata ({expected_meta_bytes})",
        neg_fp.ram_bytes
    );

    // Upserting a fresh delta record must increase measured resident RAM to account for delta metadata & map.
    let ram_before = ivf_fp.ram_bytes;
    ivf.upsert(&CanonicalVectorRecord {
        source_id: "fresh:footprint-delta-check".into(),
        version: 1,
        tenant_id: "tenant-hanzo".into(),
        acl_group: "eng-core".into(),
        filter_tags: vec!["sel:100".into(), "tag:delta-check".into()],
        vector: make_unit_vec(64, 9999),
    })
    .unwrap();
    assert!(
        ivf.footprint().ram_bytes > ram_before,
        "IVF-PQ resident RAM must increase after delta upsert"
    );
}

#[test]
fn repo_corpus_caps_chunks_at_requested_target_vectors() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let cfg = CorpusConfig {
        seed: 1347,
        dim: 32,
        target_vectors: 128,
        num_clusters: 8,
        queries_per_selectivity: 4,
        downstream_task_count: 8,
    };

    let with_repo = build_benchmark_corpus(Some(repo_root), &cfg).unwrap();
    let without_repo = build_benchmark_corpus(None, &cfg).unwrap();

    assert_eq!(with_repo.total_vectors, 128);
    assert_eq!(with_repo.store.len(), 128);
    assert_eq!(without_repo.total_vectors, 128);
    assert!(
        with_repo.repo_chunks_loaded > 0 && with_repo.repo_chunks_loaded <= 64,
        "repo chunks ({}) must be bounded by target_vectors / 2 (64)",
        with_repo.repo_chunks_loaded
    );
}

#[test]
fn repo_corpus_skips_downstream_sampling_when_count_is_zero() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let cfg = CorpusConfig {
        seed: 1347,
        dim: 32,
        target_vectors: 128,
        num_clusters: 8,
        queries_per_selectivity: 4,
        downstream_task_count: 0,
    };

    let corpus = build_benchmark_corpus(Some(repo_root), &cfg).unwrap();
    assert_eq!(corpus.total_vectors, 128);
    assert!(corpus.repo_chunks_loaded > 0, "expected real repo chunks to be loaded");
    assert!(
        corpus.downstream_tasks.is_empty(),
        "downstream_tasks must be empty when downstream_task_count == 0"
    );
}
