//! Negative Regime 2: Analytical closed-form storage, DRAM, and cloud cost projections
//! for ColPali multi-vector visual page indexing at `100,000` and `1,000,000` pages.
//!
//! Strictly flagged as `is_projected: true` in JSON receipts and separated from local
//! wall-clock benchmark measurements (`is_projected: false`).
//!
//! Reference ColPali v1.2 / v1.3 geometry (Faysse et al., ICLR 2025):
//! - `1,030` visual patch + instruction tokens per page (`32x32 = 1,024` image patches + 6 prompt tokens)
//! - `D = 128` embedding dimension per patch
//! - `2x2` spatial mean-pooling option: `256` patches per page × `128`-d

/// Cloud infrastructure unit pricing assumptions used transparently in projections.
#[derive(Debug, Clone, Copy)]
pub struct StorageCostAssumptions {
    /// Cloud DRAM cost in USD / GB-month (standard hyperscaler memory-optimized node).
    pub dram_usd_per_gb_month: f64,
    /// Local NVMe SSD cost in USD / GB-month.
    pub nvme_usd_per_gb_month: f64,
    /// Object storage (Hanzo S3) cost in USD / GB-month.
    pub s3_usd_per_gb_month: f64,
}

impl Default for StorageCostAssumptions {
    fn default() -> Self {
        Self {
            dram_usd_per_gb_month: 4.50,
            nvme_usd_per_gb_month: 0.12,
            s3_usd_per_gb_month: 0.023,
        }
    }
}

/// One row of the analytical storage/RAM explosion projection table.
#[derive(Debug, Clone)]
pub struct StorageProjectionRow {
    pub is_projected: bool,
    pub scale_pages: u64,
    pub architecture: String,
    pub patches_per_page: usize,
    pub dim: usize,
    pub visual_page_fraction: f64,
    pub bytes_per_page_ram: f64,
    pub bytes_per_page_disk: f64,
    pub total_ram_gb: f64,
    pub total_disk_gb: f64,
    pub compression_vs_fp32_ram: f64,
    pub monthly_dram_usd: f64,
    pub monthly_s3_usd: f64,
}

/// Compute exact closed-form storage projections for `100,000` and `1,000,000` pages
/// using standard ColPali (`1,030` patches × `128`-d) and selective routing / pooling variants.
pub fn compute_storage_explosion_projections(
    costs: StorageCostAssumptions,
) -> Vec<StorageProjectionRow> {
    const COLPALI_PATCHES_FULL: usize = 1030;
    const COLPALI_PATCHES_POOLED_2X2: usize = 256;
    const COLPALI_DIM: usize = 128;
    const PAGE_META_BYTES: f64 = 96.0;
    // OCR+BM25+single-vector (128-d FP32 = 512 B + ~750 B postings + 96 B meta ≈ 1,358 B/page).
    const OCR_TEXT_BYTES_PER_PAGE: f64 = 1358.0;

    let fp32_full_bytes = (COLPALI_PATCHES_FULL * COLPALI_DIM * 4) as f64 + PAGE_META_BYTES;
    let fp16_full_bytes = (COLPALI_PATCHES_FULL * COLPALI_DIM * 2) as f64 + PAGE_META_BYTES;
    // SQ8: 1 byte/dim + 4 bytes scale per patch.
    let sq8_full_bytes = (COLPALI_PATCHES_FULL * (COLPALI_DIM + 4)) as f64 + PAGE_META_BYTES;
    // 1-bit Binary: 128/8 = 16 bytes per patch (exact 32x patch vector compression).
    let bin1_full_bytes = (COLPALI_PATCHES_FULL * (COLPALI_DIM / 8)) as f64 + PAGE_META_BYTES;
    // Two-stage Binary (hot RAM) + SQ8 (cold S3/disk rerank slab).
    let bin1_plus_sq8_disk = bin1_full_bytes + (COLPALI_PATCHES_FULL * (COLPALI_DIM + 4)) as f64;
    // 2x2 spatially pooled 1-bit Binary (256 patches × 16 B + meta).
    let bin1_pooled_bytes =
        (COLPALI_PATCHES_POOLED_2X2 * (COLPALI_DIM / 8)) as f64 + PAGE_META_BYTES;
    let bin1_pooled_sq8_disk =
        bin1_pooled_bytes + (COLPALI_PATCHES_POOLED_2X2 * (COLPALI_DIM + 4)) as f64;

    // Realistic enterprise corpus: 30% visual-heavy pages (tables/figures/formulas), 70% plain text.
    let selective_fraction = 0.30f64;
    let router_bin_ram = OCR_TEXT_BYTES_PER_PAGE + selective_fraction * bin1_full_bytes;
    let router_bin_disk = OCR_TEXT_BYTES_PER_PAGE + selective_fraction * bin1_plus_sq8_disk;
    let router_pooled_ram = OCR_TEXT_BYTES_PER_PAGE + selective_fraction * bin1_pooled_bytes;
    let router_pooled_disk = OCR_TEXT_BYTES_PER_PAGE + selective_fraction * bin1_pooled_sq8_disk;

    let specs: &[(&str, usize, f64, f64, f64)] = &[
        ("ocr_bm25_hybrid_baseline", 0, 0.0, OCR_TEXT_BYTES_PER_PAGE, OCR_TEXT_BYTES_PER_PAGE),
        ("colpali_fp32_uncompressed", COLPALI_PATCHES_FULL, 1.0, fp32_full_bytes, fp32_full_bytes),
        ("colpali_bf16_fp16", COLPALI_PATCHES_FULL, 1.0, fp16_full_bytes, fp16_full_bytes),
        ("colpali_sq8_int8", COLPALI_PATCHES_FULL, 1.0, sq8_full_bytes, sq8_full_bytes),
        (
            "colpali_binary_1bit_ram_sq8_disk",
            COLPALI_PATCHES_FULL,
            1.0,
            bin1_full_bytes,
            bin1_plus_sq8_disk,
        ),
        (
            "hybrid_router_30pct_binary_full",
            COLPALI_PATCHES_FULL,
            selective_fraction,
            router_bin_ram,
            router_bin_disk,
        ),
        (
            "hybrid_router_30pct_binary_2x2_pooled",
            COLPALI_PATCHES_POOLED_2X2,
            selective_fraction,
            router_pooled_ram,
            router_pooled_disk,
        ),
    ];

    let mut rows = Vec::new();
    for &scale_pages in &[100_000u64, 1_000_000u64] {
        for &(arch, patches, frac, ram_b, disk_b) in specs {
            let total_ram_gb = (ram_b * (scale_pages as f64)) / 1_000_000_000.0;
            let total_disk_gb = (disk_b * (scale_pages as f64)) / 1_000_000_000.0;
            rows.push(StorageProjectionRow {
                is_projected: true,
                scale_pages,
                architecture: arch.to_string(),
                patches_per_page: patches,
                dim: COLPALI_DIM,
                visual_page_fraction: frac,
                bytes_per_page_ram: ram_b,
                bytes_per_page_disk: disk_b,
                total_ram_gb,
                total_disk_gb,
                compression_vs_fp32_ram: fp32_full_bytes / ram_b,
                monthly_dram_usd: total_ram_gb * costs.dram_usd_per_gb_month,
                monthly_s3_usd: total_disk_gb * costs.s3_usd_per_gb_month,
            });
        }
    }
    rows
}
