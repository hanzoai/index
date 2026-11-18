// SPDX-License-Identifier: Apache-2.0
//! Reproducible HyDE benchmark runner.
use rag_eval::{BenchConfig, run_hyde_matrix, write_hyde_results_json};
use std::path::PathBuf;

fn main() {
    let out = std::env::args().nth(1).map(PathBuf::from).unwrap_or(PathBuf::from("results/hyde-results.json"));
    let cfg = BenchConfig::default();
    eprintln!(
        "hyde bench: seed={} dim={} n_base={} queries={}",
        cfg.seed, cfg.dim, cfg.n_base, cfg.n_queries
    );
    let report = run_hyde_matrix(&cfg).expect("hyde matrix failed");
    write_hyde_results_json(&report, &out).expect("failed to write results json");
    eprintln!("wrote {}", out.display());
    for run in &report.runs {
        println!("{}", serde_json::to_string(run).expect("serialize run"));
    }
}
