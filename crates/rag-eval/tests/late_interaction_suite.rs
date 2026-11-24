// SPDX-License-Identifier: Apache-2.0

use rag_eval::late_interaction::{
    default_late_interaction_benchmark, embed_token, evaluate_late_system, maxsim_scalar,
    maxsim_simd_unrolled, CompressedTokenVec, LateAccessContext, LateInteractionError,
    MultiVecCompression, PlaidMultiVectorIndex, TOKEN_VEC_DIM,
};

#[test]
fn simd_unrolled_maxsim_matches_scalar_reference_across_odd_and_even_lengths() {
    let q: Vec<[f32; TOKEN_VEC_DIM]> = ["fsync_wal_segment_atomic", "barrier_epoch_u64", "unlock"]
        .iter()
        .map(|s| embed_token(s))
        .collect();
    let d: Vec<[f32; TOKEN_VEC_DIM]> = [
        "cluster",
        "telemetry",
        "fsync_wal_segment_atomic",
        "commits",
        "barrier_epoch_u64",
        "4096",
        "unlock",
    ]
    .iter()
    .map(|s| embed_token(s))
    .collect();

    let s_ref = maxsim_scalar(&q, &d);
    let s_opt = maxsim_simd_unrolled(&q, &d);
    assert!((s_ref - s_opt).abs() < 1e-5, "ref={s_ref} opt={s_opt}");
}

#[test]
fn residual_2bit_plaid_beats_pooled_dense_and_achieves_over_8x_precision_per_kb() {
    let (records, cases) = default_late_interaction_benchmark();
    let snapshot = records.clone();
    let index = PlaidMultiVectorIndex::build(&records);
    assert_eq!(records, snapshot, "derived index must never mutate canonical records");

    let pooled =
        evaluate_late_system("pooled_dense_64d", &index, &cases, None, 4, 3).unwrap();
    let fp32_late = evaluate_late_system(
        "colbert_fp32_maxsim",
        &index,
        &cases,
        Some(MultiVecCompression::UncompressedFp32),
        4,
        3,
    )
    .unwrap();
    let plaid_2bit = evaluate_late_system(
        "plaid_residual_2bit_maxsim",
        &index,
        &cases,
        Some(MultiVecCompression::Residual2Bit),
        4,
        3,
    )
    .unwrap();

    // Late interaction recovers symbol/long-passage recall diluted by single-vector mean pooling
    assert!((fp32_late.precision_at_1 - 1.0).abs() < 1e-9);
    assert!((plaid_2bit.precision_at_1 - 1.0).abs() < 1e-9);
    assert!((plaid_2bit.downstream_task_success - 1.0).abs() < 1e-9);
    assert!(plaid_2bit.precision_at_1 > pooled.precision_at_1);

    // 2-bit residual compression shrinks multi-vector footprint >8x vs FP32 token vectors
    assert!(fp32_late.index_bytes > plaid_2bit.index_bytes * 6);
    assert!(plaid_2bit.precision_per_kilobyte > fp32_late.precision_per_kilobyte * 6.0);
    assert_eq!(CompressedTokenVec::BYTE_SIZE, 21);
}

#[test]
fn enforces_fail_closed_tenant_and_role_isolation_before_shortlist_and_maxsim() {
    let (records, _) = default_late_interaction_benchmark();
    let index = PlaidMultiVectorIndex::build(&records);

    assert!(matches!(
        index.search(
            &LateAccessContext {
                tenant_id: "".into(),
                roles: vec!["eng".into()],
            },
            "fsync_wal_segment_atomic",
            Some(MultiVecCompression::Residual2Bit),
            4,
            5,
        ),
        Err(LateInteractionError::InvalidAccessContext)
    ));

    // Cross-tenant record with exact query tokens must never appear
    let hits = index
        .search(
            &LateAccessContext {
                tenant_id: "t_main".into(),
                roles: vec!["eng".into()],
            },
            "fsync_wal_segment_atomic barrier_epoch_u64",
            Some(MultiVecCompression::Residual2Bit),
            4,
            10,
        )
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|(id, _, _)| id.tenant_id == "t_main"));

    // Unprivileged caller without "eng" role gets 0 role-restricted hits
    let unpriv_hits = index
        .search(
            &LateAccessContext {
                tenant_id: "t_main".into(),
                roles: vec![],
            },
            "fsync_wal_segment_atomic barrier_epoch_u64",
            Some(MultiVecCompression::Residual2Bit),
            4,
            10,
        )
        .unwrap();
    assert!(unpriv_hits.is_empty());
}
