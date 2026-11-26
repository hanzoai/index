#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use colpali_eval::benchmark::{report_to_json, run_benchmark, BenchmarkConfig};
use colpali_eval::types::PageModality;

fn main() {
    if let Err(err) = run() {
        eprintln!("colpali-bench error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut config = BenchmarkConfig::default();
    let mut repo_root = PathBuf::from(".");
    let mut output_path = PathBuf::from("docs/research/hip-1347-colpali/results.json");

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                config.seed = parse_arg(&args, i, "--seed")?;
            }
            "--pages" => {
                i += 1;
                config.total_pages = parse_arg(&args, i, "--pages")?;
            }
            "--patches" => {
                i += 1;
                config.patches_per_page = parse_arg(&args, i, "--patches")?;
            }
            "--dim" => {
                i += 1;
                config.dim = parse_arg(&args, i, "--dim")?;
            }
            "--queries" => {
                i += 1;
                config.queries_per_slice = parse_arg(&args, i, "--queries")?;
            }
            "--repo-root" => {
                i += 1;
                let s = args.get(i).ok_or_else(|| "missing value for --repo-root".to_string())?;
                repo_root = PathBuf::from(s);
            }
            "--output" => {
                i += 1;
                let s = args.get(i).ok_or_else(|| "missing value for --output".to_string())?;
                output_path = PathBuf::from(s);
            }
            "--help" | "-h" => {
                println!(
                    "colpali-bench (HIP-1347 / hanzoai/index#23)\n\
                     Usage: colpali-bench [--seed 1347] [--pages 512] [--patches 64] [--dim 128] \
                     [--queries 24] [--repo-root .] [--output docs/research/hip-1347-colpali/results.json]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let report = run_benchmark(&repo_root, &config).map_err(|e| e.to_string())?;
    let json = report_to_json(&report);

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("failed to create output dir: {e}"))?;
        }
    }
    fs::write(&output_path, &json)
        .map_err(|e| format!("failed to write {}: {e}", output_path.display()))?;

    println!("=== HIP-1347 ColPali Visual Document Retrieval Benchmark (hanzoai/index#23) ===");
    println!(
        "Seed: {} | Pages: {} | Patches/Page: {} | Dim: {} | Repo Docs Loaded: {}",
        report.provenance.seed,
        report.provenance.total_pages,
        report.provenance.patches_per_page,
        report.provenance.dim,
        report.provenance.repo_files_loaded.len()
    );
    println!(
        "\n{:<34} {:>8} {:>8} {:>8} {:>9} {:>9} {:>10} {:>10} {:>6}",
        "Strategy (Overall Multimodal)",
        "NDCG@5",
        "Rec@5",
        "MRR",
        "p50(ms)",
        "p99(ms)",
        "RAM B/pg",
        "Disk B/pg",
        "Leaks"
    );
    for s in &report.strategies {
        let m = &s.overall_multimodal;
        println!(
            "{:<34} {:>8.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3} {:>10.1} {:>10.1} {:>6}",
            s.strategy.as_str(),
            m.ndcg_at_5,
            m.recall_at_5,
            m.mrr,
            m.latency_p50_ms,
            m.latency_p99_ms,
            s.footprint.bytes_per_page_ram,
            s.footprint.bytes_per_page_disk,
            m.acl_leaks
        );
    }

    println!("\nPer-Modality NDCG@5 Breakdown:");
    println!(
        "{:<34} {:>11} {:>12} {:>13} {:>13} {:>13}",
        "Strategy", "TextOnly", "DenseTable", "SciFigure", "MultiColEq", "Neg1:ExactKW"
    );
    for s in &report.strategies {
        let get_ndcg = |m: PageModality| s.by_modality.get(&m).map(|x| x.ndcg_at_5).unwrap_or(0.0);
        println!(
            "{:<34} {:>11.3} {:>12.3} {:>13.3} {:>13.3} {:>13.3}",
            s.strategy.as_str(),
            get_ndcg(PageModality::TextOnly),
            get_ndcg(PageModality::DenseTable),
            get_ndcg(PageModality::ScientificFigure),
            get_ndcg(PageModality::MultiColumnFormula),
            s.negative_regime_1_exact_keyword.ndcg_at_5
        );
    }

    println!(
        "\nHeadline Measured Ratios: Binary vs FP32 p50={:.2}x, p99={:.2}x, RAM={:.2}x | Router vs FP32 RAM={:.2}x",
        report.binary_speedup_vs_fp32_p50,
        report.binary_speedup_vs_fp32_p99,
        report.binary_ram_reduction_vs_fp32,
        report.router_ram_reduction_vs_fp32
    );
    println!(
        "Drop/Rebuild Parity: {} (canonical={:#018x}) | Adversarial ACL Leaks: {} (fail-closed={})",
        report.drop_rebuild_parity.all_passed(),
        report.drop_rebuild_parity.canonical_fingerprint_before,
        report.adversarial_acl_audit.total_acl_leaks,
        report.adversarial_acl_audit.fail_closed_checks_passed
    );
    println!("Wrote raw JSON receipt to {}", output_path.display());

    Ok(())
}

fn parse_arg<T: std::str::FromStr>(args: &[String], idx: usize, flag: &str) -> Result<T, String> {
    let raw = args.get(idx).ok_or_else(|| format!("missing value for {flag}"))?;
    raw.parse::<T>().map_err(|_| format!("invalid value '{raw}' for {flag}"))
}
