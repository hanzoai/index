// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use rag_eval::{fuse_rrf, rrf_k_ablation, RankedDoc, RrfConfig};

#[test]
fn test_rrf_fusion_and_prefilter() {
    let dense = vec![
        RankedDoc { doc_id: "doc_a".into(), rank: 1 },
        RankedDoc { doc_id: "doc_b".into(), rank: 2 },
    ];
    let sparse = vec![
        RankedDoc { doc_id: "doc_b".into(), rank: 1 },
        RankedDoc { doc_id: "doc_c".into(), rank: 2 },
    ];

    let fused = fuse_rrf(&[dense.clone(), sparse.clone()], RrfConfig::default(), None);
    assert_eq!(fused[0].doc_id, "doc_b");

    let mut allowed = HashSet::new();
    allowed.insert("doc_a".to_string());
    let filtered = fuse_rrf(&[dense, sparse], RrfConfig::default(), Some(&allowed));
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].doc_id, "doc_a");
}

#[test]
fn test_rrf_k_ablation_sweep() {
    let dense = vec![RankedDoc { doc_id: "doc_x".into(), rank: 1 }];
    let sparse = vec![RankedDoc { doc_id: "doc_x".into(), rank: 2 }];
    let sweep = rrf_k_ablation(&[dense, sparse], &[10.0, 60.0, 100.0]);
    assert_eq!(sweep.len(), 3);
    assert!(sweep[0].1.as_ref().unwrap().score > sweep[1].1.as_ref().unwrap().score);
}