// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic PRNG, distance kernels, SQ8 & Product Quantization (PQ),
//! CRC32 integrity checks, Recall@k, and latency percentile metrics.

use std::collections::HashSet;

use crate::canonical::{AnnError, SearchHit};

/// Deterministic SplitMix64 PRNG (zero external dependencies, bit-identical across targets).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal `N(0, 1)` via Box-Muller transform.
    pub fn next_normal_f32(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos()) as f32
    }

    /// Uniform integer in `0..upper` (panics if `upper == 0`).
    #[inline]
    pub fn next_usize(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be > 0");
        (self.next_u64() as usize) % upper
    }
}

/// Normalize a vector to unit L2 norm in place.
pub fn normalize_l2(v: &mut [f32]) -> Result<(), AnnError> {
    let mut norm_sq = 0.0_f64;
    for &x in v.iter() {
        if !x.is_finite() {
            return Err(AnnError::NonFiniteVector);
        }
        norm_sq += f64::from(x) * f64::from(x);
    }
    if norm_sq <= 1e-24 {
        return Err(AnnError::ZeroNormVector);
    }
    let inv = (1.0 / norm_sq.sqrt()) as f32;
    for x in v.iter_mut() {
        *x *= inv;
    }
    Ok(())
}

/// Squared L2 distance between two equal-length slices (4-way unrolled for auto-vectorization).
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut i = 0;
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        i += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Validate that a query vector has the expected dimension and finite components.
pub fn validate_query_vector(query: &[f32], dim: usize) -> Result<(), AnnError> {
    if query.len() != dim {
        return Err(AnnError::DimensionMismatch { expected: dim, actual: query.len() });
    }
    if query.iter().any(|x| !x.is_finite()) {
        return Err(AnnError::NonFiniteVector);
    }
    Ok(())
}

/// Per-dimension affine 8-bit scalar quantizer (SQ8).
///
/// Compresses each `f32` vector (`4 * D` bytes) to `D` bytes (`4.0x` vector compression)
/// while retaining high recall on unit-norm embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8Quantizer {
    pub dim: usize,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
}

impl Sq8Quantizer {
    /// Fit per-dimension min/max bounds across a training slice of vectors.
    pub fn fit(dim: usize, vectors: &[&[f32]]) -> Result<Self, AnnError> {
        if dim == 0 {
            return Err(AnnError::InvalidConfig("SQ8 dim must be > 0".into()));
        }
        if vectors.is_empty() {
            return Ok(Self { dim, mins: vec![-1.0; dim], scales: vec![2.0 / 255.0; dim] });
        }
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            if v.len() != dim {
                return Err(AnnError::DimensionMismatch { expected: dim, actual: v.len() });
            }
            for d in 0..dim {
                if v[d] < mins[d] {
                    mins[d] = v[d];
                }
                if v[d] > maxs[d] {
                    maxs[d] = v[d];
                }
            }
        }
        let mut scales = Vec::with_capacity(dim);
        for d in 0..dim {
            let span = (maxs[d] - mins[d]).max(1e-6);
            // Add a small 2% headroom margin so post-build online inserts stay in range.
            let pad = span * 0.02;
            mins[d] -= pad;
            let padded_span = span + 2.0 * pad;
            scales.push(padded_span / 255.0);
        }
        Ok(Self { dim, mins, scales })
    }

    /// Encode an `f32` vector into `dim` bytes.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        debug_assert_eq!(v.len(), self.dim);
        let mut out = Vec::with_capacity(self.dim);
        for ((&val, &min), &scale) in v.iter().zip(&self.mins).zip(&self.scales) {
            let q = ((val - min) / scale).round();
            out.push(q.clamp(0.0, 255.0) as u8);
        }
        out
    }

    /// Reconstruct approximate `f32` vector from SQ8 codes.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        debug_assert_eq!(code.len(), self.dim);
        let mut out = Vec::with_capacity(self.dim);
        for ((&c, &min), &scale) in code.iter().zip(&self.mins).zip(&self.scales) {
            out.push(min + f32::from(c) * scale);
        }
        out
    }

    /// Asymmetric squared L2 distance between an exact `f32` query and an SQ8-encoded vector.
    #[inline]
    pub fn asymmetric_l2_sq(&self, query: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(query.len(), self.dim);
        debug_assert_eq!(code.len(), self.dim);
        let mut sum = 0.0_f32;
        for d in 0..self.dim {
            let approx = self.mins[d] + f32::from(code[d]) * self.scales[d];
            let diff = query[d] - approx;
            sum += diff * diff;
        }
        sum
    }

    /// Serialized binary byte size of the quantizer parameters (`8 * dim` bytes).
    pub fn serialized_bytes(&self) -> usize {
        self.dim.saturating_mul(8)
    }

    pub fn write_to_buf(&self, buf: &mut Vec<u8>) {
        for &m in &self.mins {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        for &s in &self.scales {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    pub fn read_from_slice(dim: usize, data: &[u8]) -> Result<Self, AnnError> {
        let needed = dim
            .checked_mul(8)
            .ok_or_else(|| AnnError::CorruptFormat("SQ8 parameter size overflow".into()))?;
        if data.len() < needed {
            return Err(AnnError::CorruptFormat(format!(
                "truncated SQ8 params: need {needed} bytes, got {}",
                data.len()
            )));
        }
        let mut mins = Vec::with_capacity(dim);
        let mut scales = Vec::with_capacity(dim);
        let half = dim * 4;
        for d in 0..dim {
            let m = f32::from_le_bytes(data[d * 4..d * 4 + 4].try_into().unwrap());
            let s = f32::from_le_bytes(data[half + d * 4..half + d * 4 + 4].try_into().unwrap());
            if !m.is_finite() || !s.is_finite() || s <= 0.0 {
                return Err(AnnError::CorruptFormat("invalid SQ8 min/scale value".into()));
            }
            mins.push(m);
            scales.push(s);
        }
        Ok(Self { dim, mins, scales })
    }
}

/// Product Quantizer (PQ) splitting `D` dimensions into `num_subspaces` subspaces,
/// each quantized to `k_sub <= 256` centroids (1 byte per subspace).
///
/// Used by `object_native_ivf_pq` (LanceDB / DiskANN PQ-code style) for compact
/// ADC (Asymmetric Distance Computation) lookup tables.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    pub dim: usize,
    pub num_subspaces: usize,
    pub sub_dim: usize,
    pub k_sub: usize,
    /// Flat codebook: `[m][k][sub_dim]` -> `num_subspaces * k_sub * sub_dim` floats.
    pub codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Train a deterministic Product Quantizer using k-means per subspace.
    pub fn train(
        dim: usize,
        num_subspaces: usize,
        k_sub: usize,
        vectors: &[&[f32]],
        iters: usize,
        seed: u64,
    ) -> Result<Self, AnnError> {
        if dim == 0 || num_subspaces == 0 || !dim.is_multiple_of(num_subspaces) {
            return Err(AnnError::InvalidConfig(format!(
                "dim ({dim}) must be a positive multiple of num_subspaces ({num_subspaces})"
            )));
        }
        if k_sub == 0 || k_sub > 256 {
            return Err(AnnError::InvalidConfig(format!("k_sub ({k_sub}) must be in 1..=256")));
        }
        let sub_dim = dim / num_subspaces;
        let total_floats = num_subspaces
            .checked_mul(k_sub)
            .and_then(|v| v.checked_mul(sub_dim))
            .ok_or_else(|| AnnError::InvalidConfig("PQ codebook size overflow".into()))?;
        let mut codebooks = vec![0.0_f32; total_floats];

        if vectors.is_empty() {
            return Ok(Self { dim, num_subspaces, sub_dim, k_sub, codebooks });
        }

        let mut rng = SplitMix64::new(seed ^ 0x5051_5f54_5241_494e);
        // Cap training sample size for deterministic speed when N is large.
        let train_cap = vectors.len().min(2048);
        let step = (vectors.len() / train_cap).max(1);
        let sample: Vec<&[f32]> = vectors.iter().step_by(step).take(train_cap).copied().collect();

        for m in 0..num_subspaces {
            let dim_start = m * sub_dim;
            let dim_end = dim_start + sub_dim;
            let cb_base = m * k_sub * sub_dim;

            // Deterministic spread initialization over training sample.
            for k in 0..k_sub {
                let src = sample[(k * 131 + rng.next_usize(sample.len())) % sample.len()];
                let dst = cb_base + k * sub_dim;
                codebooks[dst..dst + sub_dim].copy_from_slice(&src[dim_start..dim_end]);
            }

            let mut counts = vec![0usize; k_sub];
            let mut sums = vec![0.0_f32; k_sub * sub_dim];

            for _ in 0..iters.max(1) {
                counts.fill(0);
                sums.fill(0.0);

                for v in &sample {
                    let sub_v = &v[dim_start..dim_end];
                    let mut best_k = 0usize;
                    let mut best_d = f32::INFINITY;
                    for k in 0..k_sub {
                        let c_off = cb_base + k * sub_dim;
                        let d = l2_squared(sub_v, &codebooks[c_off..c_off + sub_dim]);
                        if d < best_d {
                            best_d = d;
                            best_k = k;
                        }
                    }
                    counts[best_k] += 1;
                    let s_off = best_k * sub_dim;
                    for j in 0..sub_dim {
                        sums[s_off + j] += sub_v[j];
                    }
                }

                for (k, &cnt) in counts.iter().enumerate() {
                    if cnt > 0 {
                        let inv = 1.0 / (cnt as f32);
                        let c_off = cb_base + k * sub_dim;
                        let s_off = k * sub_dim;
                        for j in 0..sub_dim {
                            codebooks[c_off + j] = sums[s_off + j] * inv;
                        }
                    }
                }
            }
        }

        Ok(Self { dim, num_subspaces, sub_dim, k_sub, codebooks })
    }

    /// Encode a single vector into `num_subspaces` 1-byte PQ codes.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        debug_assert_eq!(v.len(), self.dim);
        let mut codes = Vec::with_capacity(self.num_subspaces);
        for m in 0..self.num_subspaces {
            let sub_v = &v[m * self.sub_dim..(m + 1) * self.sub_dim];
            let cb_base = m * self.k_sub * self.sub_dim;
            let mut best_k = 0usize;
            let mut best_d = f32::INFINITY;
            for k in 0..self.k_sub {
                let c_off = cb_base + k * self.sub_dim;
                let d = l2_squared(sub_v, &self.codebooks[c_off..c_off + self.sub_dim]);
                if d < best_d {
                    best_d = d;
                    best_k = k;
                }
            }
            codes.push(best_k as u8);
        }
        codes
    }

    /// Precompute the `num_subspaces * k_sub` Asymmetric Distance Computation (ADC) table for a query.
    pub fn build_adc_table(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.dim);
        let mut table = Vec::with_capacity(self.num_subspaces * self.k_sub);
        for m in 0..self.num_subspaces {
            let sub_q = &query[m * self.sub_dim..(m + 1) * self.sub_dim];
            let cb_base = m * self.k_sub * self.sub_dim;
            for k in 0..self.k_sub {
                let c_off = cb_base + k * self.sub_dim;
                table.push(l2_squared(sub_q, &self.codebooks[c_off..c_off + self.sub_dim]));
            }
        }
        table
    }

    /// Evaluate approximate squared L2 distance via O(M) ADC table lookups.
    #[inline]
    pub fn adc_distance(&self, adc_table: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.num_subspaces);
        let mut dist = 0.0_f32;
        for (m, &code) in codes.iter().enumerate() {
            let idx = m * self.k_sub + usize::from(code);
            dist += adc_table[idx];
        }
        dist
    }

    /// Exact memory footprint of the PQ codebook in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.codebooks.len() * std::mem::size_of::<f32>()
    }
}

/// Standard IEEE 802.3 CRC32 checksum for on-disk binary index integrity verification.
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Compute `Recall@k` of `returned` against exact brute-force `ground_truth`.
///
/// If `ground_truth` has fewer than `k` items (e.g. ultra-selective filter),
/// denominator is `ground_truth.len().min(k)`. If `ground_truth` is empty, returns `1.0`
/// iff `returned` is also empty.
pub fn recall_at_k(ground_truth: &[SearchHit], returned: &[SearchHit], k: usize) -> f64 {
    let target_k = ground_truth.len().min(k);
    if target_k == 0 {
        return if returned.is_empty() { 1.0 } else { 0.0 };
    }
    let truth_ids: HashSet<(&str, u64)> =
        ground_truth.iter().take(target_k).map(|h| (h.source_id.as_str(), h.version)).collect();
    let matched = returned
        .iter()
        .take(k)
        .filter(|h| truth_ids.contains(&(h.source_id.as_str(), h.version)))
        .count();
    matched as f64 / target_k as f64
}

/// Latency summary statistics in milliseconds (`p50`, `p95`, `p99`, `mean`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LatencySummaryMs {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Compute exact nearest-rank percentiles over a slice of latency samples (ms).
pub fn summarize_latencies_ms(samples: &[f64]) -> LatencySummaryMs {
    if samples.is_empty() {
        return LatencySummaryMs::default();
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean_ms = sorted.iter().sum::<f64>() / (sorted.len() as f64);
    LatencySummaryMs {
        mean_ms,
        p50_ms: percentile_sorted(&sorted, 0.50),
        p95_ms: percentile_sorted(&sorted, 0.95),
        p99_ms: percentile_sorted(&sorted, 0.99),
    }
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() as f64) * q).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_ieee_standard_vector() {
        assert_eq!(crc32_ieee(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn vector_validation_rejects_nan_inf_zero_norm_and_dim_mismatch() {
        assert_eq!(normalize_l2(&mut [0.0, 0.0]), Err(AnnError::ZeroNormVector));
        assert_eq!(normalize_l2(&mut [f32::NAN, 1.0]), Err(AnnError::NonFiniteVector));
        assert_eq!(validate_query_vector(&[1.0, f32::INFINITY], 2), Err(AnnError::NonFiniteVector));
        assert_eq!(
            validate_query_vector(&[1.0], 2),
            Err(AnnError::DimensionMismatch { expected: 2, actual: 1 })
        );
    }

    #[test]
    fn sq8_deserialization_guards_overflow_and_truncation() {
        assert!(matches!(
            Sq8Quantizer::read_from_slice(usize::MAX, &[]),
            Err(AnnError::CorruptFormat(_))
        ));
        assert!(matches!(
            Sq8Quantizer::read_from_slice(4, &[0u8; 8]),
            Err(AnnError::CorruptFormat(_))
        ));
    }

    #[test]
    fn latency_percentiles_exact_nearest_rank() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let s = summarize_latencies_ms(&samples);
        assert!((s.p50_ms - 50.0).abs() < 1e-9);
        assert!((s.p95_ms - 95.0).abs() < 1e-9);
        assert!((s.p99_ms - 99.0).abs() < 1e-9);
    }
}
