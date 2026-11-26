//! Deterministic math, PRNG, Fast Walsh-Hadamard Transform, quantization, and checksum primitives.
//!
//! Zero external crate dependencies; `#![forbid(unsafe_code)]` compliant.

/// Deterministic SplitMix64 PRNG for reproducible corpora, queries, and tests.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / ((1u32 << 24) as f32)
    }

    /// Standard normal sample via Box-Muller transform.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
}

/// Deterministic 64-bit FNV-1a hash helper for canonical fingerprinting and feature hashing.
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a64 {
    hash: u64,
}

impl Fnv1a64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub const fn new() -> Self {
        Self { hash: Self::OFFSET_BASIS }
    }

    pub fn update_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash ^= u64::from(b);
            self.hash = self.hash.wrapping_mul(Self::PRIME);
        }
    }

    pub fn update_str(&mut self, s: &str) {
        self.update_u64(s.len() as u64);
        self.update_bytes(s.as_bytes());
    }

    pub fn update_u8(&mut self, v: u8) {
        self.update_bytes(&[v]);
    }

    pub fn update_u16(&mut self, v: u16) {
        self.update_bytes(&v.to_le_bytes());
    }

    pub fn update_u32(&mut self, v: u32) {
        self.update_bytes(&v.to_le_bytes());
    }

    pub fn update_u64(&mut self, v: u64) {
        self.update_bytes(&v.to_le_bytes());
    }

    pub fn update_f32(&mut self, v: f32) {
        let canonical = if v == 0.0 { 0.0f32 } else { v };
        self.update_bytes(&canonical.to_le_bytes());
    }

    pub const fn finish(self) -> u64 {
        self.hash
    }
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard IEEE 802.3 CRC32 checksum (pure Rust, table-free bit-by-bit or 256-entry LUT).
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// L2-normalize a vector in place. Leaves all-zero vectors unchanged.
pub fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 1e-12 {
        let inv = norm_sq.sqrt().recip();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Exact FP32 inner product between equal-length slices.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * b[idx];
        sum1 += a[idx + 1] * b[idx + 1];
        sum2 += a[idx + 2] * b[idx + 2];
        sum3 += a[idx + 3] * b[idx + 3];
    }
    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for i in (chunks * 4)..a.len() {
        total += a[i] * b[i];
    }
    total
}

/// Orthogonal Fast Walsh-Hadamard Transform (normalized by `1 / sqrt(n)`) for power-of-two lengths.
///
/// Preserves inner products (`<Hx, Hy> == <x, y>`) while spreading coordinate energy uniformly
/// across all dimensions prior to 1-bit binary sign quantization (analogous to Hadamard/ITQ
/// rotation used in production binary-quantized ColPali indexes).
pub fn fwht_orthogonal(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    if n == 0 || !n.is_power_of_two() {
        return v.to_vec();
    }
    let mut out = v.to_vec();
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        let mut i = 0usize;
        while i < n {
            for j in i..(i + h) {
                let x = out[j];
                let y = out[j + h];
                out[j] = x + y;
                out[j + h] = x - y;
            }
            i += step;
        }
        h = step;
    }
    let scale = (n as f32).sqrt().recip();
    for x in &mut out {
        *x *= scale;
    }
    out
}

/// Pack a float vector into 1-bit binary `u64` words (bit = 1 iff value >= 0.0).
///
/// For `D = 128`, this produces exactly 2 `u64` words (16 bytes vs 512 bytes FP32 -> 32.0x smaller).
pub fn pack_binary_words(rotated_vec: &[f32]) -> Vec<u64> {
    let words = rotated_vec.len().div_ceil(64);
    let mut out = vec![0u64; words];
    for (i, &val) in rotated_vec.iter().enumerate() {
        if val >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

/// Fast Hamming similarity for 1-bit packed vectors: returns `dim - 2 * hamming_distance`,
/// which lies in `[-dim, +dim]` and correlates monotonically with cosine similarity.
#[inline]
pub fn binary_similarity_words(a: &[u64], b: &[u64], dim: usize) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut mismatches = 0u32;
    for i in 0..a.len() {
        mismatches += (a[i] ^ b[i]).count_ones();
    }
    (dim as i32) - 2 * (mismatches as i32)
}

/// Fast specialized 128-bit (`[u64; 2]`) Hamming similarity using two XOR+POPCNT pairs.
#[inline]
pub fn binary_similarity_128(a: [u64; 2], b: [u64; 2]) -> i32 {
    let mismatches = (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones();
    128 - 2 * (mismatches as i32)
}

/// Symmetric per-vector int8 scalar quantization (SQ8): maps `[-max_abs, +max_abs]` to `[-127, +127]`.
///
/// Stores `scale = max_abs / 127.0` alongside `D` `i8` codes (`D + 4` bytes vs `4 * D` bytes FP32).
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8Vector {
    pub scale: f32,
    pub codes: Vec<i8>,
}

impl Sq8Vector {
    pub fn quantize(v: &[f32]) -> Self {
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_abs <= 1e-12 {
            return Self { scale: 0.0, codes: vec![0i8; v.len()] };
        }
        let scale = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;
        let codes = v.iter().map(|&x| (x * inv_scale).round().clamp(-127.0, 127.0) as i8).collect();
        Self { scale, codes }
    }

    /// Inner product against a pre-quantized SQ8 query vector using pure integer accumulation.
    #[inline]
    pub fn dot_sq8(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.codes.len(), other.codes.len());
        let mut acc: i32 = 0;
        let chunks = self.codes.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            acc += i32::from(self.codes[idx]) * i32::from(other.codes[idx])
                + i32::from(self.codes[idx + 1]) * i32::from(other.codes[idx + 1])
                + i32::from(self.codes[idx + 2]) * i32::from(other.codes[idx + 2])
                + i32::from(self.codes[idx + 3]) * i32::from(other.codes[idx + 3]);
        }
        for i in (chunks * 4)..self.codes.len() {
            acc += i32::from(self.codes[i]) * i32::from(other.codes[i]);
        }
        (acc as f32) * self.scale * other.scale
    }

    /// Inner product against an FP32 query vector.
    #[inline]
    pub fn dot_f32(&self, query: &[f32]) -> f32 {
        debug_assert_eq!(self.codes.len(), query.len());
        let mut acc = 0.0f32;
        for (&c, &q) in self.codes.iter().zip(query.iter()) {
            acc += f32::from(c) * q;
        }
        acc * self.scale
    }

    pub fn byte_size(&self) -> usize {
        self.codes.len() + std::mem::size_of::<f32>()
    }
}

/// Deterministic ASCII/alphanumeric tokenizer preserving underscores, hyphens, and dotted identifiers
/// (critical for exact keyword/config/code identifier retrieval in Negative Regime 1).
pub fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' || ch == ':' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            let trimmed = current.trim_matches(|c| c == '.' || c == '-' || c == ':');
            if !trimmed.is_empty() {
                tokens.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        let trimmed = current.trim_matches(|c| c == '.' || c == '-' || c == ':');
        if !trimmed.is_empty() {
            tokens.push(trimmed.to_string());
        }
    }
    tokens
}

/// Deterministic unit-sphere basis vector derived from a seed and feature string.
pub fn seeded_unit_vector(seed: u64, label: &str, dim: usize) -> Vec<f32> {
    let mut h = Fnv1a64::new();
    h.update_u64(seed);
    h.update_str(label);
    let mut rng = SplitMix64::new(h.finish());
    let mut v = vec![0.0f32; dim];
    for x in &mut v {
        *x = rng.next_normal();
    }
    l2_normalize(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_preserves_inner_product_and_norm() {
        let a = seeded_unit_vector(42, "patch_a", 128);
        let b = seeded_unit_vector(42, "patch_b", 128);
        let ha = fwht_orthogonal(&a);
        let hb = fwht_orthogonal(&b);
        let orig = dot_f32(&a, &b);
        let rot = dot_f32(&ha, &hb);
        assert!((orig - rot).abs() < 1e-4, "orig={orig}, rot={rot}");
        let norm_ha = dot_f32(&ha, &ha);
        assert!((norm_ha - 1.0).abs() < 1e-4, "norm_ha={norm_ha}");
    }

    #[test]
    fn sq8_and_binary_preserve_similarity_ordering() {
        let q = seeded_unit_vector(7, "query", 128);
        let mut close = q.clone();
        let noise = seeded_unit_vector(7, "noise", 128);
        for i in 0..128 {
            close[i] = 0.85 * close[i] + 0.15 * noise[i];
        }
        l2_normalize(&mut close);
        let far = seeded_unit_vector(7, "far_unrelated", 128);

        let sq_q = Sq8Vector::quantize(&q);
        let sq_close = Sq8Vector::quantize(&close);
        let sq_far = Sq8Vector::quantize(&far);
        assert!(sq_q.dot_sq8(&sq_close) > sq_q.dot_sq8(&sq_far) + 0.5);

        let bq = pack_binary_words(&fwht_orthogonal(&q));
        let bc = pack_binary_words(&fwht_orthogonal(&close));
        let bf = pack_binary_words(&fwht_orthogonal(&far));
        assert!(binary_similarity_words(&bq, &bc, 128) > binary_similarity_words(&bq, &bf, 128));
    }

    #[test]
    fn crc32_matches_known_standard_vector() {
        // Standard IEEE 802.3 CRC32 check value for ASCII "123456789" is 0xCBF43926.
        assert_eq!(crc32_ieee(b"123456789"), 0xcbf4_3926);
    }
}
