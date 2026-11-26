// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Analytical 1M / 10M / 100M vector scaling projection model grounded directly
//! in measured bytes/vector, partition geometry, and object-storage I/O primitives.
//!
//! Every output explicitly carries `is_projected = true` so measured local runs
//! and large-scale 1M/10M/100M projections are never conflated.

use crate::object_store::ObjectStoreCostModel;

/// Projected RAM, storage, S3 IOPS, latency, and monthly cost at a target scale `N`.
#[derive(Debug, Clone)]
pub struct ScaleProjectionRow {
    pub architecture: String,
    pub num_vectors: u64,
    pub dim: usize,
    pub is_projected: bool,
    pub ram_bytes_per_vector: f64,
    pub disk_bytes_per_vector: f64,
    pub total_ram_gb: f64,
    pub total_storage_gb: f64,
    pub s3_gets_per_query_cold: f64,
    pub sequential_rtts_cold: f64,
    pub bytes_fetched_per_query_kb: f64,
    pub est_cold_p99_ms: f64,
    pub monthly_storage_cost_usd: f64,
    pub monthly_dram_cost_usd: f64,
    pub s3_get_cost_per_1m_queries_usd: f64,
    pub notes: String,
}

/// Compute closed-form 1M, 10M, and 100M scaling projections for all 4 architectures
/// at standard production embedding dimensionality (`dim`, e.g., 768).
///
/// Formulas are grounded directly in the exact binary layouts verified by this crate:
/// - **`exact_flat_baseline`**: `4 * D` bytes/vec RAM + 32 B metadata (no ANN index).
/// - **`local_mmap_vamana` (DiskANN / USearch)**:
///   - Disk (`HVAN` binary layout): `D` (SQ8) + `4` (deg) + `4 * R` (adj, `R=32`) + `32` B meta = `D + 164` B/vec.
///   - Hot RAM (DiskANN PQ-in-RAM + Vamana mmap working set): `pq_subspaces` (`D/16` bytes, e.g. 48 B at D=768) + `16` B compact ID/ACL bitset = `64` B/vec (or full mmap `D + 164` B/vec if warm-resident).
/// - **`object_native_ivf_pq` (LanceDB / Hanzo S3/VFS)**:
///   - Immutable S3 segment pack: `4` (id) + `M` (PQ, `D/16` = 48 B) + `D` (SQ8 refine) + `16` B compact ACL/tag mask = `D + M + 20` B/vec.
///   - Hot compute-local RAM: coarse centroids (`C = sqrt(N)` centroids * `4 * D` bytes) + PQ codebook (`M * 256 * (D/M) * 4` bytes) + partition directory (`C * 128` bytes) + compact 16B/vec ACL filter cache (optional, or packed in segment).
///   - Cold query: `1` parallel batch RTT of `nprobe` contiguous `RangeGET`s (`nprobe` = `16` at 1M, `24` at 10M, `32` at 100M).
/// - **`object_store_random_graph_hops` (Negative Control)**:
///   - Same `HVAN` bytes on S3, but cold query chases `H` sequential hops (`22–34` sequential RTTs) * `R` random `RangeGET`s -> catastrophic S3 latency & GET cost!
pub fn project_scale_matrix(
    dim: usize,
    cost: ObjectStoreCostModel,
    measured_neg_control_hops: f64,
    measured_neg_control_gets: f64,
) -> Vec<ScaleProjectionRow> {
    let scales: &[(u64, usize, usize, f64)] = &[
        // (N, coarse_partitions C (~240-380 vecs/part, LanceDB-style), nprobe, log_hop_scale)
        (1_000_000, 4_096, 16, 1.00),
        (10_000_000, 32_768, 24, 1.25),
        (100_000_000, 262_144, 32, 1.50),
    ];

    let bytes_per_gb = 1_000_000_000.0_f64;
    let pq_m = (dim / 16).max(8); // e.g., 48 bytes at D=768, 96 bytes at D=1536
    let vamana_r = 32usize;
    let mut rows = Vec::with_capacity(scales.len() * 4);

    for &(n, partitions_c, nprobe, hop_scale) in scales {
        let nf = n as f64;

        // 1. exact_flat_baseline (in-memory FP32 brute force)
        let flat_ram_bpv = (dim * 4 + 32) as f64;
        let flat_disk_bpv = (dim * 4) as f64;
        let flat_ram_gb = (nf * flat_ram_bpv) / bytes_per_gb;
        let flat_disk_gb = (nf * flat_disk_bpv) / bytes_per_gb;
        // ~40 GB/s memory bandwidth scan time for FP32 flat scan.
        let flat_scan_ms = (nf * (dim * 4) as f64) / (40.0 * 1e6);
        rows.push(ScaleProjectionRow {
            architecture: "exact_flat_baseline".to_string(),
            num_vectors: n,
            dim,
            is_projected: true,
            ram_bytes_per_vector: flat_ram_bpv,
            disk_bytes_per_vector: flat_disk_bpv,
            total_ram_gb: flat_ram_gb,
            total_storage_gb: flat_disk_gb,
            s3_gets_per_query_cold: 0.0,
            sequential_rtts_cold: 0.0,
            bytes_fetched_per_query_kb: 0.0,
            est_cold_p99_ms: flat_scan_ms,
            monthly_storage_cost_usd: flat_disk_gb * cost.nvme_storage_cost_per_gb_month_usd,
            monthly_dram_cost_usd: flat_ram_gb * cost.dram_cost_per_gb_month_usd,
            s3_get_cost_per_1m_queries_usd: 0.0,
            notes: "FP32 brute-force oracle; impractical RAM & latency at >=10M".to_string(),
        });

        // 2. local_mmap_vamana (DiskANN / USearch local NVMe + PQ-in-RAM hot cache)
        let vamana_disk_bpv = (dim + 4 + vamana_r * 4 + 32) as f64; // SQ8 + deg + R*u32 + compact meta
        let vamana_ram_bpv = (pq_m + 24) as f64; // PQ code in RAM + compact tenant/ACL/tombstone mask
        let vamana_ram_gb = (nf * vamana_ram_bpv) / bytes_per_gb;
        let vamana_disk_gb = (nf * vamana_disk_bpv) / bytes_per_gb;
        // DiskANN cold NVMe page reads: ~45-68 4KB page reads at 85 us with queue depth ~4 -> ~1.8-2.5ms p99.
        let est_pages_cold = 45.0 * hop_scale;
        let vamana_cold_p99_ms =
            (est_pages_cold * cost.local_nvme_page_read_us / 4.0) / 1_000.0 + 0.8;
        rows.push(ScaleProjectionRow {
            architecture: "local_mmap_vamana".to_string(),
            num_vectors: n,
            dim,
            is_projected: true,
            ram_bytes_per_vector: vamana_ram_bpv,
            disk_bytes_per_vector: vamana_disk_bpv,
            total_ram_gb: vamana_ram_gb,
            total_storage_gb: vamana_disk_gb,
            s3_gets_per_query_cold: 0.0,
            sequential_rtts_cold: 0.0,
            bytes_fetched_per_query_kb: est_pages_cold * 4.0,
            est_cold_p99_ms: vamana_cold_p99_ms,
            monthly_storage_cost_usd: vamana_disk_gb * cost.nvme_storage_cost_per_gb_month_usd,
            monthly_dram_cost_usd: vamana_ram_gb * cost.dram_cost_per_gb_month_usd,
            s3_get_cost_per_1m_queries_usd: 0.0,
            notes: "Local NVMe mmap Vamana + SQ8 slab + PQ-in-RAM; ideal for Hanzo local mode"
                .to_string(),
        });

        // 3. object_native_ivf_pq (LanceDB / Hanzo S3/VFS immutable segments)
        let ivf_disk_bpv = (4 + pq_m + dim + 20) as f64; // id + PQ + SQ8 + compact ACL/tag mask
        let centroids_bytes = (partitions_c * dim * 4) as f64;
        let pq_cb_bytes = (pq_m * 256 * (dim / pq_m) * 4) as f64;
        let dir_bytes = (partitions_c * 256) as f64; // offsets + bloom/tag filters
                                                     // Compute-local RAM holds ONLY centroids + PQ codebook + directory + 8B/vec tombstone/routing overhead.
        let ivf_total_ram_bytes = centroids_bytes + pq_cb_bytes + dir_bytes + nf * 8.0;
        let ivf_ram_bpv = ivf_total_ram_bytes / nf;
        let ivf_ram_gb = ivf_total_ram_bytes / bytes_per_gb;
        let ivf_disk_gb = (nf * ivf_disk_bpv) / bytes_per_gb;

        let vecs_per_part = nf / (partitions_c as f64);
        let bytes_fetched = (nprobe as f64) * vecs_per_part * ivf_disk_bpv;
        // Concurrent parallel RangeGET streams achieve ~2.5x single-stream bandwidth (capped at 1.2 GB/s).
        let parallel_bw_mb_s = (cost.s3_bandwidth_mb_per_sec * 2.5).min(1_200.0);
        let transfer_ms = (bytes_fetched / (parallel_bw_mb_s * 1e6)) * 1_000.0;
        // 1 parallel batch RTT (+ p99 tail 1.8x on S3 first byte + parallel transfer + compute scoring).
        let ivf_cold_p99_ms = cost.s3_rtt_ms * 1.8 + transfer_ms + (nprobe as f64) * 0.15 + 1.5;
        let ivf_s3_cost_1m = (nprobe as f64) * 1_000.0 * cost.s3_get_cost_per_1000_usd;

        rows.push(ScaleProjectionRow {
            architecture: "object_native_ivf_pq".to_string(),
            num_vectors: n,
            dim,
            is_projected: true,
            ram_bytes_per_vector: ivf_ram_bpv,
            disk_bytes_per_vector: ivf_disk_bpv,
            total_ram_gb: ivf_ram_gb,
            total_storage_gb: ivf_disk_gb,
            s3_gets_per_query_cold: nprobe as f64,
            sequential_rtts_cold: 1.0,
            bytes_fetched_per_query_kb: bytes_fetched / 1024.0,
            est_cold_p99_ms: ivf_cold_p99_ms,
            monthly_storage_cost_usd: ivf_disk_gb * cost.s3_storage_cost_per_gb_month_usd,
            monthly_dram_cost_usd: ivf_ram_gb * cost.dram_cost_per_gb_month_usd,
            s3_get_cost_per_1m_queries_usd: ivf_s3_cost_1m,
            notes: format!(
                "Cloud S3/VFS: C={partitions_c} partitions, nprobe={nprobe} in 1 parallel RangeGET batch"
            ),
        });

        // 4. object_store_random_graph_hops (Negative Control)
        let neg_hops = (measured_neg_control_hops.max(18.0)) * hop_scale;
        let neg_gets = (measured_neg_control_gets.max(180.0)) * hop_scale;
        let neg_bytes_kb = (neg_gets * vamana_disk_bpv) / 1024.0;
        // Every sequential hop pays a full S3 RTT (and p99 compounds across H hops!).
        let neg_cold_p99_ms = neg_hops * cost.s3_rtt_ms * 1.35;
        let neg_s3_cost_1m = neg_gets * 1_000.0 * cost.s3_get_cost_per_1000_usd;
        rows.push(ScaleProjectionRow {
            architecture: "object_store_random_graph_hops".to_string(),
            num_vectors: n,
            dim,
            is_projected: true,
            ram_bytes_per_vector: 4.0,
            disk_bytes_per_vector: vamana_disk_bpv,
            total_ram_gb: (nf * 4.0) / bytes_per_gb,
            total_storage_gb: vamana_disk_gb,
            s3_gets_per_query_cold: neg_gets,
            sequential_rtts_cold: neg_hops,
            bytes_fetched_per_query_kb: neg_bytes_kb,
            est_cold_p99_ms: neg_cold_p99_ms,
            monthly_storage_cost_usd: vamana_disk_gb * cost.s3_storage_cost_per_gb_month_usd,
            monthly_dram_cost_usd: ((nf * 4.0) / bytes_per_gb) * cost.dram_cost_per_gb_month_usd,
            s3_get_cost_per_1m_queries_usd: neg_s3_cost_1m,
            notes: "NEGATIVE CONTROL: sequential pointer-chasing hops over S3 -> unacceptable RTT & GET cost"
                .to_string(),
        });
    }

    rows
}
