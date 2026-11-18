// SPDX-License-Identifier: Apache-2.0

//! Bounty #21 (`hanzoai/index#25`) reproducible runner.
//!
//! Runs the full (maintenance-policy × insert-rate) matrix and writes the
//! results JSON. Default output: `results/ann-insert-results.json` relative to
//! the crate directory.
//!
//! ```text
//! cargo run --release -p rag-eval --example ann_insert_bench -- [out.json]
//! ```
//!
//! Every figure in the output is a simulation result on the authoring host and
//! is **not** a Hanzo performance expectation; see `honesty_labels` in the JSON.

use rag_eval::ann_insert::{BenchConfig, run_full_matrix, write_results_json};
use std::path::PathBuf;

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ann-insert-results.json"));

    let cfg = BenchConfig::default();
    eprintln!(
        "ann_insert bench: seed={} dim={} n_base={} queries={} rates={:?} penalty={}ns/page",
        cfg.seed, cfg.dim, cfg.n_base, cfg.n_queries, cfg.rates, cfg.io_penalty_ns
    );

    let report = run_full_matrix(&cfg).expect("benchmark matrix failed");
    write_results_json(&report, &out).expect("failed to write results json");
    eprintln!("wrote {}", out.display());

    for run in &report.runs {
        println!(
            "{}",
            serde_json::to_string(run).expect("serialize run result")
        );
    }
    for verdict in &report.hypothesis_verdicts {
        println!(
            "{}",
            serde_json::to_string(verdict).expect("serialize verdict")
        );
    }
}
