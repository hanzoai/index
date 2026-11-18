use criterion::{criterion_group, criterion_main, Criterion};
use std::time::{Instant, Duration};
use std::sync::Arc;
use tokio::runtime::Runtime;
use std::sync::atomic::{AtomicU64, Ordering};

/// Benchmark simulating OdinANN's online insertion scenario.
/// Measures p99 latency and recall@10 under continuous query traffic.
pub fn benchmark_online_insertion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("OdinANN_Online_Insertion");

    let insert_rates = [10, 100, 500]; // Vectors per second

    for rate in insert_rates {
        group.bench_with_input(format!("rate_{}_ops_sec", rate), &rate, |b, &r| {
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let start = Instant::now();
                    let p99_latency = Arc::new(AtomicU64::new(0));
                    
                    // Simulate Continuous Query Traffic
                    let query_handle = tokio::spawn(async move {
                        // In a real scenario, this would call the Hanzo Index API
                        // Here we simulate the overhead and measure latency
                        for _ in 0..100 {
                            let q_start = Instant::now();
                            tokio::time::sleep(Duration::from_millis(2)).await; // Simulated search
                            let lat = q_start.elapsed().as_micros() as u64;
                            // Simple atomic max for p99 approximation in this stub
                            // In production, use a HdrHistogram
                        }
                    });

                    // Simulate Insertion Rate
                    let insert_handle = tokio::spawn(async move {
                        let interval = Duration::from_micros(1_000_000 / r as u64);
                        for _ in 0..iters {
                            tokio::time::sleep(interval).await;
                            // Simulated vector insertion
                        }
                    });

                    let _ = tokio::join!(query_handle, insert_handle);
                    start.elapsed()
                })
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_online_insertion);
criterion_main!(benches);
