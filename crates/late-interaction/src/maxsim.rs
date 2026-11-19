#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes the MaxSim score between a query (multiple vectors) and a document (multiple vectors).
/// s(q,d) = sum_{q_i} max_{d_j} (dot_product(q_i, d_j))
pub fn maxsim_score(query: &[Vec<f32>], document: &[Vec<f32>]) -> f32 {
    query.iter()
        .map(|q_vec| {
            document.iter()
                .map(|d_vec| dot_product(q_vec, d_vec))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Optimized dot product with SIMD support.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }
    dot_product_fallback(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    let mut res = [0.0f32; 8];
    _mm256_storeu_ps(res.as_mut_ptr(), sum);
    let mut final_sum = res.iter().sum::<f32>();

    while i < n {
        final_sum += a[i] * b[i];
        i += 1;
    }
    final_sum
}

fn dot_product_fallback(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_basic() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = vec![vec![1.0, 0.1], vec![0.1, 1.0]];
        let score = maxsim_score(&q, &d);
        // max(1*1 + 0*0.1, 1*0.1 + 0*1) = 1.0
        // max(0*1 + 1*0.1, 0*0.1 + 1*1) = 1.0
        // sum = 2.0
        assert!((score - 2.0).abs() < 1e-5);
    }
}
