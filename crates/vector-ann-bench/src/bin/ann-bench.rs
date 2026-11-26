// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reproducible benchmark CLI (`ann-bench`) for `hanzoai/index#15` (HIP-1347).
//!
//! Usage:
//! ```bash
//! cargo run -p vector-ann-bench --release --bin ann-bench -- \
//!   --seed 1347 \
//!   --vectors 4096 \
//!   --dim 64 \
//!   --output docs/research/hip-1347-object-native-ann/results.json
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;

use vector_ann_bench::{run_full_benchmark, CorpusConfig, ObjectStoreCostModel};

fn main() {
    let mut seed: u64 = 1347;
    let mut target_vectors: usize = 4096;
    let mut dim: usize = 64;
    let mut queries_per_selectivity: usize = 60;
    let mut repo_root: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("docs/research/hip-1347-object-native-ann/results.json");

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).expect("--seed requires a u64");
            }
            "--vectors" => {
                i += 1;
                target_vectors =
                    args.get(i).and_then(|s| s.parse().ok()).expect("--vectors requires a usize");
            }
            "--dim" => {
                i += 1;
                dim = args.get(i).and_then(|s| s.parse().ok()).expect("--dim requires a usize");
            }
            "--queries" => {
                i += 1;
                queries_per_selectivity =
                    args.get(i).and_then(|s| s.parse().ok()).expect("--queries requires a usize");
            }
            "--repo-root" => {
                i += 1;
                repo_root = Some(PathBuf::from(args.get(i).expect("--repo-root requires a path")));
            }
            "--output" => {
                i += 1;
                output_path = PathBuf::from(args.get(i).expect("--output requires a path"));
            }
            "--help" | "-h" => {
                println!(
                    "ann-bench (HIP-1347 / hanzoai/index#15)\n\
                     Options:\n\
                       --seed <u64>        Deterministic seed (default: 1347)\n\
                       --vectors <usize>   Corpus vector count (default: 4096)\n\
                       --dim <usize>       Embedding dimension (default: 64)\n\
                       --queries <usize>   Queries per selectivity slice (default: 60)\n\
                       --repo-root <path>  Repository root for real source chunks (default: cwd)\n\
                       --output <path>     JSON receipt output path"
                );
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let resolved_root = repo_root.unwrap_or_else(|| env::current_dir().unwrap_or_default());
    let cfg = CorpusConfig {
        seed,
        dim,
        target_vectors,
        num_clusters: 32,
        queries_per_selectivity,
        downstream_task_count: 40,
    };
    let cost_model = ObjectStoreCostModel::default();

    println!(
        "Running HIP-1347 ANN & Object-Storage-Native Benchmark (seed={}, N={}, D={}, repo_root={})...",
        cfg.seed,
        cfg.target_vectors,
        cfg.dim,
        resolved_root.display()
    );

    let report = match run_full_benchmark(Some(&resolved_root), &cfg, cost_model) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("benchmark failed: {e}");
            std::process::exit(1);
        }
    };

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("failed to create output directory");
        }
    }
    let json_text = report.to_json().to_pretty_string();
    fs::write(&output_path, &json_text).expect("failed to write results.json");

    println!("\n=== Corpus & Build Summary ===");
    println!(
        "Repo files scanned: {}, Real repo chunks loaded: {}, Total vectors: {}, Dim: {}",
        report.repo_files_scanned, report.repo_chunks_loaded, report.total_vectors, report.dim
    );
    for (arch, ms) in &report.build_times_ms {
        println!("  build {:<32} : {:>8.2} ms", arch, ms);
    }

    println!("\n=== Measured Footprints (Cold Base) ===");
    for (arch, fp) in &report.footprints {
        println!(
            "  {:<32} | RAM: {:>8.1} B/vec | Disk/Obj: {:>8.1} B/vec",
            arch, fp.bytes_per_vector_ram, fp.bytes_per_vector_disk
        );
    }

    println!("\n=== Measured Query Performance (Recall@10, Total Latency incl. I/O, S3 GETs/RTTs, ACL Leaks) ===");
    println!(
        "{:<32} | {:>4} | {:<10} | {:>6} | {:>8} | {:>8} | {:>7} | {:>5} | {:>5}",
        "Architecture", "Sel", "Cache", "R@10", "p50(ms)", "p99(ms)", "S3GETs", "RTTs", "Leaks"
    );
    for r in &report.measured_slices {
        println!(
            "{:<32} | {:>4} | {:<10} | {:>6.3} | {:>8.3} | {:>8.3} | {:>7.1} | {:>5.1} | {:>5}",
            r.architecture,
            r.selectivity,
            r.cache_regime,
            r.recall_at_10,
            r.total_latency_with_io.p50_ms,
            r.total_latency_with_io.p99_ms,
            r.avg_s3_gets,
            r.avg_sequential_rtts,
            r.acl_leaks
        );
    }

    println!("\n=== Explicit Negative Regime: Ultra-Selective 1% Filter ===");
    for r in &report.ultra_selective_1pct {
        println!(
            "  {:<32} | R@10={:.3} | hits={:.1} | p50={:.3}ms | p99={:.3}ms | leaks={}",
            r.strategy, r.recall_at_10, r.avg_hits_returned, r.p50_ms, r.p99_ms, r.acl_leaks
        );
    }

    println!("\n=== Mutation, Tombstone & Compaction Lifecycle ===");
    for m in &report.mutations {
        println!(
            "  {:<24} | upserts={} ({:.1}us/op) | deletes={} ({:.1}us/op) | post-R@10={:.3} | compact={:.1}ms -> R@10={:.3} | leaks={} | stale={}",
            m.architecture,
            m.inserts_applied + m.updates_applied,
            m.avg_upsert_us,
            m.deletes_applied,
            m.avg_delete_us,
            m.post_mutation_recall_at_10,
            m.compaction.duration_ms,
            m.post_compaction_recall_at_10,
            m.post_mutation_acl_leaks,
            m.post_mutation_stale_or_deleted_hits
        );
    }

    println!("\n=== Downstream Real-Repo File Resolution Tasks ===");
    for t in &report.downstream_tasks {
        println!(
            "  {:<24} | tasks={} | R@10={:.3} | File@1={:.3} | File@5={:.3} | p99={:.3}ms | leaks={}",
            t.architecture,
            t.task_count,
            t.recall_at_10,
            t.file_resolution_success_at_1,
            t.file_resolution_success_at_5,
            t.p99_ms_warm,
            t.acl_leaks
        );
    }

    println!("\n=== Drop & Rebuild Bit-Exact Parity ===");
    println!(
        "  canonical_unchanged={} | vamana_binary_bit_exact={} | vamana_hits_bit_exact={} | ivf_pq_hits_bit_exact={} | exact_flat_bit_exact={}",
        report.rebuild_parity.canonical_unchanged,
        report.rebuild_parity.vamana_binary_bit_exact,
        report.rebuild_parity.vamana_hits_bit_exact,
        report.rebuild_parity.ivf_pq_hits_bit_exact,
        report.rebuild_parity.exact_flat_hits_bit_exact
    );

    println!("\nWrote raw JSON receipt to {}", output_path.display());
}
