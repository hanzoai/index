// SPDX-License-Identifier: Apache-2.0

use rag_eval::domain_embeddings::{
    default_domain_benchmark, evaluate_encoder, standard_encoder_specs, ArtifactKind,
    CanonicalRecord, DerivedDomainIndex, DistillationJustificationGate, DomainAccessContext,
    DomainEmbeddingError, DomainQueryCase, DomainSlice, EncoderFamily, ModelLicenseAudit,
    SourceRecordIdentity,
};
use std::collections::BTreeMap;

#[test]
fn compares_modernbert_laya_qwen3_bgem3_and_justifies_code_and_memory_distillation() {
    let (records, cases) = default_domain_benchmark();
    let specs = standard_encoder_specs();

    let mut reports = BTreeMap::new();
    let mut spec_map = BTreeMap::new();
    for spec in specs {
        let fam = spec.family;
        spec_map.insert(format!("{fam:?}"), spec.clone());
        let index = DerivedDomainIndex::build(spec, &records).expect("permissive license");
        let rep = evaluate_encoder(&index, &cases, 5).expect("non-empty benchmark");
        reports.insert(format!("{fam:?}"), rep);
    }

    let mb_base = &reports["ModernBertBase"];
    let laya = &reports["ModernBertLaya"];
    let qwen_06 = &reports["Qwen3Embedding0_6B"];
    let qwen_8b = &reports["Qwen3Embedding8B"];
    let bge_m3 = &reports["BgeM3DenseSparseMultiVec"];
    let code_student = &reports["DistilledCodeStudent"];
    let mem_student = &reports["DistilledMemoryStudent"];

    assert!(qwen_8b.ndcg_at_5 > mb_base.ndcg_at_5);
    assert!(bge_m3.ndcg_at_5 > mb_base.ndcg_at_5);
    assert!(laya.throughput_docs_per_sec > qwen_06.throughput_docs_per_sec);
    assert!(qwen_8b.memory_gib_per_million_docs > laya.memory_gib_per_million_docs * 4.0);

    let gate = DistillationJustificationGate::default();
    let code_dec = gate.evaluate(
        mb_base,
        qwen_8b,
        code_student,
        &spec_map["DistilledCodeStudent"],
        &spec_map["Qwen3Embedding8B"],
        DomainSlice::CodeRetrieval,
    );
    assert!(code_dec.justified, "code student distillation must justify maintenance: {code_dec:?}");
    assert!(code_dec.net_annual_savings_usd > 10_000.0);

    let mem_dec = gate.evaluate(
        mb_base,
        qwen_8b,
        mem_student,
        &spec_map["DistilledMemoryStudent"],
        &spec_map["Qwen3Embedding8B"],
        DomainSlice::AgentMemory,
    );
    assert!(mem_dec.justified, "memory student distillation must justify maintenance: {mem_dec:?}");
}

#[test]
fn enforces_fail_closed_tenant_acl_derived_rebuild_and_license_gate() {
    let (records, _) = default_domain_benchmark();
    let laya_spec = standard_encoder_specs()
        .into_iter()
        .find(|s| s.family == EncoderFamily::ModernBertLaya)
        .unwrap();

    let mut bad_spec = laya_spec.clone();
    bad_spec.license = ModelLicenseAudit {
        model_id: "vendor/non-commercial-embed".into(),
        spdx_license: "CC-BY-NC-4.0".into(),
        attribution_notice: "Study only".into(),
        commercial_allowed: false,
    };
    assert!(matches!(
        DerivedDomainIndex::build(bad_spec, &records),
        Err(DomainEmbeddingError::DisallowedLicense { .. })
    ));

    let snapshot_before = records.clone();
    let index = DerivedDomainIndex::build(laya_spec, &records).unwrap();
    assert_eq!(records, snapshot_before);

    let unpriv_alpha = DomainAccessContext {
        tenant_id: "tenant_alpha".into(),
        roles: vec![],
    };
    let hits = index
        .search(
            &unpriv_alpha,
            "Speicherleck memory_leak secret key HnswLayer::rewire_bidirectional",
            &["HnswLayer::rewire_bidirectional".into()],
            None,
            10,
        )
        .unwrap();
    for (id, _, _) in &hits {
        assert_eq!(id.tenant_id, "tenant_alpha");
        assert!(id.allowed_roles.is_empty(), "role-gated blob leaked to unprivileged caller");
    }
}

#[test]
fn downstream_judge_rejects_right_file_that_contradicts_task_completion() {
    let laya_spec = standard_encoder_specs()
        .into_iter()
        .find(|s| s.family == EncoderFamily::ModernBertLaya)
        .unwrap();
    let rec = CanonicalRecord {
        identity: SourceRecordIdentity {
            record_id: "stale_doc".into(),
            version: 2,
            kind: ArtifactKind::SourceDocument,
            tenant_id: "t1".into(),
            allowed_roles: vec![],
        },
        domain: DomainSlice::CodeRetrieval,
        language: "en".into(),
        title: "TLS config".into(),
        body: "TLS 1.0 fallback is deprecated and disabled; never enable insecure_renegotiation.".into(),
        symbols: vec!["TlsConfig".into()],
    };
    let idx = DerivedDomainIndex::build(laya_spec, &[rec]).unwrap();
    let case = DomainQueryCase {
        query_id: "control_task".into(),
        domain: DomainSlice::CodeRetrieval,
        access: DomainAccessContext {
            tenant_id: "t1".into(),
            roles: vec![],
        },
        query: "TLS config".into(),
        query_symbols: vec!["TlsConfig".into()],
        relevant_grades: BTreeMap::from([("stale_doc".into(), 3)]),
        required_answer_facts: vec!["tls".into()],
        forbidden_contradictions: vec!["insecure_renegotiation".into()],
    };
    let rep = evaluate_encoder(&idx, &[case], 5).unwrap();
    assert!((rep.ndcg_at_5 - 1.0).abs() < 1e-9);
    assert!((rep.downstream_task_success - 0.0).abs() < 1e-9);
}
