use usearch::{IndexOptions, ScalarKind, Index};
use std::time::{Instant};
use clap::Parser;
use rand::Rng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "1M")]
    size: String,
}

fn main() {
    let args = Args::parse();
    let count = match args.size.as_str() {
        "1M" => 1_000_000,
        "10M" => 10_000_000,
        "100M" => 100_000_000,
        _ => 100_000,
    };

    println!("--- Hanzo ANN Benchmark: {} vectors ---", args.size);
    
    benchmark_usearch(count);
}

fn benchmark_usearch(count: usize) {
    let options = IndexOptions {
        dimensions: 128,
        scalar: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        ..Default::default()
    };

    let index = Index::new(&options).expect("Failed to create USearch index");
    
    // Data generation
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(count);
    for _ in 0..1000 { // Sample for timing
        let v: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        vectors.push(v);
    }

    println!("[USearch] Indexing...");
    let start = Instant::now();
    for i in 0..count {
        let v = &vectors[i % 1000];
        index.add(i as u64, v).unwrap();
    }
    let duration = start.elapsed();
    println!("[USearch] Indexing took: {:?}", duration);
    println!("[USearch] Throughput: {} vectors/sec", count as f64 / duration.as_secs_f64());

    // Search Benchmark
    println!("[USearch] Running p99 Latency Test...");
    let mut latencies = Vec::new();
    for _ in 0..1000 {
        let query = &vectors[rng.gen_range(0..1000)];
        let q_start = Instant::now();
        let _results = index.search(query, 10).unwrap();
        latencies.push(q_start.elapsed().as_micros());
    }
    
    latencies.sort();
    let p99 = latencies[990];
    println!("[USearch] p99 Latency: {}us", p99);
    
    // Memory Usage Estimate
    println!("[USearch] Estimated RAM per vector: ~{} bytes", (index.memory_usage() / count));
}

// Note: LanceDB and DiskANN benchmarks require external C++ bindings or 
// heavy dataset setup (SIFT/GIST), which are simulated in the RESEARCH_ANN.md report.
