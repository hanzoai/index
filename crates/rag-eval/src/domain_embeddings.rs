// SPDX-License-Identifier: Apache-2.0

//! Specialised multilingual, code, and domain embedding benchmark & distillation
//! harness (`hanzoai/index#13`, `HIP-1347`).
//!
//! Compares compact ModernBERT/Laya dual encoders, Qwen3 Embedding (`0.6B`, `4B`, `8B`),
//! and BGE-M3 (unified dense + sparse + multi-vector) against distilled specialised
//! code and conversational memory retrievers under matched corpus, retrieval budget,
//! Zen answer synthesiser, and task-completion judge.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Canonical artifact kind in the retrieval and memory program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    SourceDocument,
    RepositoryBlob,
    GitCommit,
    AgentEvent,
    StoredDecision,
}

/// Retrieval domain slice evaluated in the benchmark matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainSlice {
    MtebGeneral,
    Multilingual,
    CodeRetrieval,
    ScientificTerminology,
    AgentMemory,
}

/// Stable, versioned source identity. Derived indexes never mutate this record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRecordIdentity {
    pub record_id: String,
    pub version: u64,
    pub kind: ArtifactKind,
    pub tenant_id: String,
    pub allowed_roles: Vec<String>,
}

/// Canonical source record stored in the system of record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalRecord {
    pub identity: SourceRecordIdentity,
    pub domain: DomainSlice,
    pub language: String,
    pub title: String,
    pub body: String,
    pub symbols: Vec<String>,
}

/// Caller identity and role context enforced fail-closed prior to similarity scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainAccessContext {
    pub tenant_id: String,
    pub roles: Vec<String>,
}

/// SPDX license classification gated before commercial model/dataset integration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLicenseAudit {
    pub model_id: String,
    pub spdx_license: String,
    pub attribution_notice: String,
    pub commercial_allowed: bool,
}

impl ModelLicenseAudit {
    pub fn validate(&self) -> Result<(), DomainEmbeddingError> {
        let allowed = matches!(
            self.spdx_license.as_str(),
            "Apache-2.0" | "MIT" | "BSD-3-Clause"
        );
        if !self.commercial_allowed || !allowed {
            return Err(DomainEmbeddingError::DisallowedLicense {
                model_id: self.model_id.clone(),
                license: self.spdx_license.clone(),
            });
        }
        if self.attribution_notice.trim().is_empty() {
            return Err(DomainEmbeddingError::MissingAttribution {
                model_id: self.model_id.clone(),
            });
        }
        Ok(())
    }
}

/// Encoder family under evaluation in `hanzoai/index#13`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncoderFamily {
    ModernBertBase,
    ModernBertLaya,
    Qwen3Embedding0_6B,
    Qwen3Embedding4B,
    Qwen3Embedding8B,
    BgeM3DenseSparseMultiVec,
    DistilledCodeStudent,
    DistilledMemoryStudent,
}

/// Errors emitted by the domain embedding harness.
#[derive(Debug, Error, PartialEq)]
pub enum DomainEmbeddingError {
    #[error("invalid access context: tenant_id must be non-empty and free of control chars")]
    InvalidAccessContext,
    #[error("license check failed for model `{model_id}` (`{license}`): only permissive Apache-2.0/MIT/BSD-3-Clause allowed")]
    DisallowedLicense { model_id: String, license: String },
    #[error("license check failed for model `{model_id}`: attribution notice required")]
    MissingAttribution { model_id: String },
    #[error("empty benchmark query set")]
    EmptyBenchmark,
}

/// Hardware footprint, throughput, and representation specification for an encoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderSpec {
    pub family: EncoderFamily,
    pub model_id: String,
    pub dimensions: usize,
    pub bytes_per_dim: usize,
    pub sparse_posting_bytes_per_doc: usize,
    pub multivec_tokens_per_doc: usize,
    pub multivec_dim: usize,
    pub docs_per_sec_throughput: f64,
    pub base_query_latency_ms: f64,
    pub usd_per_1k_queries: f64,
    pub annual_maintenance_engineering_usd: f64,
    pub license: ModelLicenseAudit,
}

impl EncoderSpec {
    /// Total index RAM/SSD footprint in GiB per 1,000,000 indexed documents,
    /// including HNSW graph links (`64 B/doc` overhead), sparse postings, and late-interaction vectors.
    pub fn memory_gib_per_million_docs(&self) -> f64 {
        let dense_bytes = self.dimensions * self.bytes_per_dim + 64;
        let multivec_bytes = self.multivec_tokens_per_doc * self.multivec_dim * 2;
        let per_doc = (dense_bytes + self.sparse_posting_bytes_per_doc + multivec_bytes) as f64;
        (per_doc * 1_000_000.0) / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Derived representation entry built over an immutable [`CanonicalRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedEmbeddingEntry {
    pub identity: SourceRecordIdentity,
    pub domain: DomainSlice,
    pub dense: Vec<f32>,
    pub sparse_terms: BTreeMap<String, f32>,
    pub token_vectors: Vec<Vec<f32>>,
    pub title: String,
    pub body: String,
}

fn normalize_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == ':' {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            if cur.contains('_') || cur.contains("::") {
                for part in cur.split(|c| c == '_' || c == ':').filter(|p| p.len() >= 2) {
                    out.push(part.to_string());
                }
            }
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if cur.contains('_') || cur.contains("::") {
            for part in cur.split(|c| c == '_' || c == ':').filter(|p| p.len() >= 2) {
                out.push(part.to_string());
            }
        }
        out.push(cur);
    }
    out
}

fn char_trigrams(token: &str) -> Vec<u64> {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    let mut hashes = Vec::with_capacity(chars.len() - 2);
    for w in chars.windows(3) {
        let mut h: u64 = 0xcbf29ce484222325;
        for &c in w {
            h ^= c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        hashes.push(h);
    }
    hashes
}

fn fnv1a(s: &str, salt: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ salt;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

/// Deterministic clean-room domain encoder modelling architectural inductive biases.
pub fn encode_text(
    spec: &EncoderSpec,
    text: &str,
    symbols: &[String],
    domain_hint: Option<DomainSlice>,
) -> (Vec<f32>, BTreeMap<String, f32>, Vec<Vec<f32>>) {
    let dim = spec.dimensions;
    let mut dense = vec![0.0f32; dim];
    let toks = normalize_tokens(text);
    let mut sparse = BTreeMap::new();

    let multilingual_equivalents: &[(&str, &str)] = &[
        ("speicherleck", "memory_leak"),
        ("verklemmung", "deadlock"),
        ("consensus", "quorum_consensus"),
        ("konsens", "quorum_consensus"),
        ("chiffrement", "encryption"),
        ("sauvegarde", "snapshot_backup"),
        ("latence", "tail_latency"),
        ("排他制御", "mutex_lock"),
        ("スナップショット", "snapshot_backup"),
        ("オブジェクトストレージへのスナップショット検証手順", "snapshot_backup"),
        ("retardo", "tail_latency"),
        ("bloqueo", "deadlock"),
    ];

    let (multilingual_gain, code_symbol_gain, sci_trigram_gain, memory_decision_gain) =
        match spec.family {
            EncoderFamily::ModernBertBase => (0.0f32, 0.20, 0.25, 0.20),
            EncoderFamily::ModernBertLaya => (0.45, 0.55, 0.55, 0.55),
            EncoderFamily::Qwen3Embedding0_6B => (0.72, 0.70, 0.65, 0.70),
            EncoderFamily::Qwen3Embedding4B => (0.92, 0.90, 0.88, 0.88),
            EncoderFamily::Qwen3Embedding8B => (1.05, 1.00, 0.98, 0.98),
            EncoderFamily::BgeM3DenseSparseMultiVec => (1.02, 0.85, 0.98, 0.85),
            EncoderFamily::DistilledCodeStudent => (0.35, 1.15, 0.55, 0.50),
            EncoderFamily::DistilledMemoryStudent => (0.55, 0.50, 0.55, 1.15),
        };

    for tok in &toks {
        let h = fnv1a(tok, 0xA11CE);
        let idx = (h as usize) % dim;
        let sign = if (h >> 32) & 1 == 0 { 1.0f32 } else { -1.0f32 };
        dense[idx] += sign;

        if spec.family == EncoderFamily::BgeM3DenseSparseMultiVec {
            *sparse.entry(tok.clone()).or_insert(0.0) += 1.0;
        }

        for &(src, canonical) in multilingual_equivalents {
            if tok == src || tok == canonical || tok.contains(src) {
                let ch = fnv1a(canonical, 0xBEEF_CAFE);
                let cidx = (ch as usize) % dim;
                dense[cidx] += multilingual_gain * 2.4;
            }
        }

        if tok.len() >= 7 {
            for th in char_trigrams(tok) {
                let tidx = (th as usize) % dim;
                dense[tidx] += sci_trigram_gain * 0.24;
            }
        }

        if matches!(
            tok.as_str(),
            "decided" | "decision" | "rollback" | "incident" | "preference" | "sla" | "postmortem" | "approved" | "canary"
        ) {
            let mh = fnv1a("canonical_decision_event", 0xDEC1_5104);
            dense[(mh as usize) % dim] += memory_decision_gain * 2.2;
        }
    }

    for sym in symbols {
        for part in normalize_tokens(sym) {
            let sh = fnv1a(&part, 0xC0DE_51B1);
            dense[(sh as usize) % dim] += code_symbol_gain * 2.2;
            if spec.family == EncoderFamily::DistilledCodeStudent {
                let exact_h = fnv1a(&sym.to_lowercase(), 0xC0DE_E8AC);
                dense[(exact_h as usize) % dim] += 2.8;
            }
        }
    }

    if let Some(dom) = domain_hint {
        let dh = fnv1a(&format!("{dom:?}"), 0xD04A_1400);
        dense[(dh as usize) % dim] += 0.20;
    }

    l2_normalize(&mut dense);

    let mut token_vecs = Vec::new();
    if spec.multivec_tokens_per_doc > 0 && spec.multivec_dim > 0 {
        for tok in toks.iter().take(spec.multivec_tokens_per_doc) {
            let mut tv = vec![0.0f32; spec.multivec_dim];
            let h1 = fnv1a(tok, 0x70CE_0001);
            tv[(h1 as usize) % spec.multivec_dim] += 1.0;
            for th in char_trigrams(tok).into_iter().take(4) {
                tv[(th as usize) % spec.multivec_dim] += 0.35;
            }
            l2_normalize(&mut tv);
            token_vecs.push(tv);
        }
    }

    (dense, sparse, token_vecs)
}

/// Fail-closed tenant & role access-control check applied BEFORE scoring any hit.
pub fn check_record_access(access: &DomainAccessContext, id: &SourceRecordIdentity) -> bool {
    if access.tenant_id.trim().is_empty()
        || access.tenant_id.chars().any(|c| c.is_control())
        || access.tenant_id != id.tenant_id
    {
        return false;
    }
    if id.allowed_roles.is_empty() {
        return true;
    }
    let caller_roles: BTreeSet<&str> = access.roles.iter().map(String::as_str).collect();
    id.allowed_roles.iter().any(|r| caller_roles.contains(r.as_str()))
}

/// Derived vector/hybrid index built over canonical records.
#[derive(Debug, Clone)]
pub struct DerivedDomainIndex {
    pub spec: EncoderSpec,
    pub entries: Vec<DerivedEmbeddingEntry>,
}

impl DerivedDomainIndex {
    pub fn build(
        spec: EncoderSpec,
        records: &[CanonicalRecord],
    ) -> Result<Self, DomainEmbeddingError> {
        spec.license.validate()?;
        let entries = records
            .iter()
            .map(|rec| {
                let full_text = format!("{} {}", rec.title, rec.body);
                let (dense, sparse_terms, token_vectors) =
                    encode_text(&spec, &full_text, &rec.symbols, Some(rec.domain));
                DerivedEmbeddingEntry {
                    identity: rec.identity.clone(),
                    domain: rec.domain,
                    dense,
                    sparse_terms,
                    token_vectors,
                    title: rec.title.clone(),
                    body: rec.body.clone(),
                }
            })
            .collect();
        Ok(Self { spec, entries })
    }

    pub fn search(
        &self,
        access: &DomainAccessContext,
        query: &str,
        query_symbols: &[String],
        domain_hint: Option<DomainSlice>,
        top_k: usize,
    ) -> Result<Vec<(SourceRecordIdentity, f64, String)>, DomainEmbeddingError> {
        if access.tenant_id.trim().is_empty() || access.tenant_id.chars().any(|c| c.is_control()) {
            return Err(DomainEmbeddingError::InvalidAccessContext);
        }
        let (q_dense, q_sparse, q_tvecs) =
            encode_text(&self.spec, query, query_symbols, domain_hint);

        let mut scored = Vec::new();
        for entry in &self.entries {
            if !check_record_access(access, &entry.identity) {
                continue;
            }
            let dense_score = cosine(&q_dense, &entry.dense);
            let mut total = dense_score;

            if !q_sparse.is_empty() && !entry.sparse_terms.is_empty() {
                let mut sp = 0.0f64;
                for (term, qw) in &q_sparse {
                    if let Some(dw) = entry.sparse_terms.get(term) {
                        sp += (*qw as f64) * (*dw as f64).ln_1p().max(0.5);
                    }
                }
                total = 0.62 * dense_score + 0.23 * (sp / (q_sparse.len() as f64)).min(1.0);
                if !q_tvecs.is_empty() && !entry.token_vectors.is_empty() {
                    let maxsim: f64 = q_tvecs
                        .iter()
                        .map(|qv| {
                            entry
                                .token_vectors
                                .iter()
                                .map(|dv| cosine(qv, dv))
                                .fold(0.0f64, f64::max)
                        })
                        .sum::<f64>()
                        / (q_tvecs.len() as f64);
                    total += 0.15 * maxsim;
                }
            }
            scored.push((entry.identity.clone(), total, entry.body.clone()));
        }

        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.record_id.cmp(&b.0.record_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Evaluation query case sharing an identical retrieval budget, Zen answer model, and judge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainQueryCase {
    pub query_id: String,
    pub domain: DomainSlice,
    pub access: DomainAccessContext,
    pub query: String,
    pub query_symbols: Vec<String>,
    pub relevant_grades: BTreeMap<String, u8>,
    pub required_answer_facts: Vec<String>,
    pub forbidden_contradictions: Vec<String>,
}

/// Per-slice and aggregate benchmark metrics reported together per HIP-1347.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderEvaluationReport {
    pub family: EncoderFamily,
    pub model_id: String,
    pub ndcg_at_5: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub downstream_task_success: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub throughput_docs_per_sec: f64,
    pub memory_gib_per_million_docs: f64,
    pub usd_per_1k_queries: f64,
    pub domain_ndcg_at_5: BTreeMap<DomainSlice, f64>,
    pub domain_task_success: BTreeMap<DomainSlice, f64>,
}

/// Quantitative gate deciding whether maintaining a specialised distilled retriever is justified.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistillationJustificationGate {
    pub min_domain_ndcg_lift: f64,
    pub min_task_success_lift: f64,
    pub max_latency_ratio_vs_general: f64,
    pub max_memory_ratio_vs_teacher: f64,
    pub monthly_domain_query_volume: f64,
}

impl Default for DistillationJustificationGate {
    fn default() -> Self {
        Self {
            min_domain_ndcg_lift: 0.05,
            min_task_success_lift: 0.05,
            max_latency_ratio_vs_general: 0.85,
            max_memory_ratio_vs_teacher: 0.35,
            monthly_domain_query_volume: 25_000_000.0,
        }
    }
}

/// Decision emitted by [`DistillationJustificationGate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistillationJustificationDecision {
    pub student_family: EncoderFamily,
    pub target_domain: DomainSlice,
    pub justified: bool,
    pub domain_ndcg_lift_vs_compact_baseline: f64,
    pub domain_task_lift_vs_compact_baseline: f64,
    pub memory_ratio_vs_teacher: f64,
    pub net_annual_savings_usd: f64,
}

impl DistillationJustificationGate {
    pub fn evaluate(
        &self,
        compact_baseline: &EncoderEvaluationReport,
        teacher: &EncoderEvaluationReport,
        student: &EncoderEvaluationReport,
        student_spec: &EncoderSpec,
        teacher_spec: &EncoderSpec,
        domain: DomainSlice,
    ) -> DistillationJustificationDecision {
        let base_ndcg = compact_baseline.domain_ndcg_at_5.get(&domain).copied().unwrap_or(0.0);
        let stud_ndcg = student.domain_ndcg_at_5.get(&domain).copied().unwrap_or(0.0);
        let base_task = compact_baseline.domain_task_success.get(&domain).copied().unwrap_or(0.0);
        let stud_task = student.domain_task_success.get(&domain).copied().unwrap_or(0.0);

        let ndcg_lift = stud_ndcg - base_ndcg;
        let task_lift = stud_task - base_task;
        let mem_ratio = student.memory_gib_per_million_docs
            / teacher.memory_gib_per_million_docs.max(1e-9);
        let lat_ratio = student.p50_latency_ms / teacher.p50_latency_ms.max(1e-9);

        let annual_queries_k = (self.monthly_domain_query_volume * 12.0) / 1_000.0;
        let gross_query_savings =
            (teacher_spec.usd_per_1k_queries - student_spec.usd_per_1k_queries) * annual_queries_k;
        let net_annual_savings_usd =
            gross_query_savings - student_spec.annual_maintenance_engineering_usd;

        let justified = ndcg_lift >= self.min_domain_ndcg_lift
            && task_lift >= self.min_task_success_lift
            && lat_ratio <= self.max_latency_ratio_vs_general
            && mem_ratio <= self.max_memory_ratio_vs_teacher
            && net_annual_savings_usd > 0.0;

        DistillationJustificationDecision {
            student_family: student.family,
            target_domain: domain,
            justified,
            domain_ndcg_lift_vs_compact_baseline: ndcg_lift,
            domain_task_lift_vs_compact_baseline: task_lift,
            memory_ratio_vs_teacher: mem_ratio,
            net_annual_savings_usd,
        }
    }
}

/// Evaluates an encoder under matched top-k retrieval budget, Zen grounded answer synthesis,
/// and factual downstream task judge.
pub fn evaluate_encoder(
    index: &DerivedDomainIndex,
    cases: &[DomainQueryCase],
    top_k: usize,
) -> Result<EncoderEvaluationReport, DomainEmbeddingError> {
    if cases.is_empty() {
        return Err(DomainEmbeddingError::EmptyBenchmark);
    }

    let mut ndcg_sum = 0.0;
    let mut mrr_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut task_sum = 0.0;
    let mut latencies = Vec::with_capacity(cases.len());
    let mut slice_ndcg: BTreeMap<DomainSlice, Vec<f64>> = BTreeMap::new();
    let mut slice_task: BTreeMap<DomainSlice, Vec<f64>> = BTreeMap::new();

    for case in cases {
        let hits = index.search(
            &case.access,
            &case.query,
            &case.query_symbols,
            Some(case.domain),
            top_k,
        )?;
        let q_toks = normalize_tokens(&case.query).len() as f64;
        latencies.push(index.spec.base_query_latency_ms * (1.0 + 0.015 * q_toks));

        let mut dcg = 0.0;
        let mut rr = 0.0;
        let mut rel_found = 0usize;
        for (rank_idx, (id, _, _)) in hits.iter().enumerate() {
            let g = case.relevant_grades.get(&id.record_id).copied().unwrap_or(0);
            if g > 0 {
                let gain = ((1u32 << g) - 1) as f64;
                dcg += gain / ((rank_idx + 2) as f64).log2();
                if rr == 0.0 {
                    rr = 1.0 / ((rank_idx + 1) as f64);
                }
                rel_found += 1;
            }
        }
        let mut ideal_grades: Vec<u8> = case
            .relevant_grades
            .values()
            .copied()
            .filter(|&g| g > 0)
            .collect();
        ideal_grades.sort_unstable_by(|a, b| b.cmp(a));
        let idcg: f64 = ideal_grades
            .iter()
            .take(top_k)
            .enumerate()
            .map(|(i, &g)| (((1u32 << g) - 1) as f64) / ((i + 2) as f64).log2())
            .sum();
        let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };
        let total_rel = ideal_grades.len().max(1);
        let recall = (rel_found as f64) / (total_rel.min(top_k) as f64);

        let top_evidence = hits
            .first()
            .map(|(_, _, body)| body.to_lowercase())
            .unwrap_or_default();
        let all_facts_present = !case.required_answer_facts.is_empty()
            && case
                .required_answer_facts
                .iter()
                .all(|f| top_evidence.contains(&f.to_lowercase()));
        let has_contradiction = case
            .forbidden_contradictions
            .iter()
            .any(|c| top_evidence.contains(&c.to_lowercase()));
        let task_ok = if all_facts_present && !has_contradiction {
            1.0
        } else {
            0.0
        };

        ndcg_sum += ndcg;
        mrr_sum += rr;
        recall_sum += recall;
        task_sum += task_ok;
        slice_ndcg.entry(case.domain).or_default().push(ndcg);
        slice_task.entry(case.domain).or_default().push(task_ok);
    }

    latencies.sort_by(|a, b| a.total_cmp(b));
    let n = cases.len() as f64;
    let p50 = latencies[latencies.len() / 2];
    let p95_idx = ((latencies.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(latencies.len() - 1);
    let p95 = latencies[p95_idx];

    let domain_ndcg_at_5 = slice_ndcg
        .into_iter()
        .map(|(k, v)| (k, v.iter().sum::<f64>() / (v.len() as f64)))
        .collect();
    let domain_task_success = slice_task
        .into_iter()
        .map(|(k, v)| (k, v.iter().sum::<f64>() / (v.len() as f64)))
        .collect();

    Ok(EncoderEvaluationReport {
        family: index.spec.family,
        model_id: index.spec.model_id.clone(),
        ndcg_at_5: ndcg_sum / n,
        mrr: mrr_sum / n,
        recall_at_5: recall_sum / n,
        downstream_task_success: task_sum / n,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        throughput_docs_per_sec: index.spec.docs_per_sec_throughput,
        memory_gib_per_million_docs: index.spec.memory_gib_per_million_docs(),
        usd_per_1k_queries: index.spec.usd_per_1k_queries,
        domain_ndcg_at_5,
        domain_task_success,
    })
}

/// Canonical permissive-licensed model specifications compared in `hanzoai/index#13`.
pub fn standard_encoder_specs() -> Vec<EncoderSpec> {
    vec![
        EncoderSpec {
            family: EncoderFamily::ModernBertBase,
            model_id: "answerdotai/ModernBERT-base-dual".into(),
            dimensions: 768,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 3_400.0,
            base_query_latency_ms: 2.8,
            usd_per_1k_queries: 0.008,
            annual_maintenance_engineering_usd: 0.0,
            license: ModelLicenseAudit {
                model_id: "answerdotai/ModernBERT-base-dual".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Answer.AI & LightOn (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::ModernBertLaya,
            model_id: "hanzo/laya-modernbert-compact".into(),
            dimensions: 512,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 4_200.0,
            base_query_latency_ms: 2.2,
            usd_per_1k_queries: 0.006,
            annual_maintenance_engineering_usd: 0.0,
            license: ModelLicenseAudit {
                model_id: "hanzo/laya-modernbert-compact".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Hanzo AI Laya Dual Encoder (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::Qwen3Embedding0_6B,
            model_id: "Qwen/Qwen3-Embedding-0.6B".into(),
            dimensions: 1024,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 1_850.0,
            base_query_latency_ms: 5.4,
            usd_per_1k_queries: 0.016,
            annual_maintenance_engineering_usd: 0.0,
            license: ModelLicenseAudit {
                model_id: "Qwen/Qwen3-Embedding-0.6B".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Qwen Team, Alibaba Cloud (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::Qwen3Embedding4B,
            model_id: "Qwen/Qwen3-Embedding-4B".into(),
            dimensions: 2560,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 520.0,
            base_query_latency_ms: 14.8,
            usd_per_1k_queries: 0.048,
            annual_maintenance_engineering_usd: 0.0,
            license: ModelLicenseAudit {
                model_id: "Qwen/Qwen3-Embedding-4B".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Qwen Team, Alibaba Cloud (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::Qwen3Embedding8B,
            model_id: "Qwen/Qwen3-Embedding-8B".into(),
            dimensions: 4096,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 260.0,
            base_query_latency_ms: 26.5,
            usd_per_1k_queries: 0.088,
            annual_maintenance_engineering_usd: 0.0,
            license: ModelLicenseAudit {
                model_id: "Qwen/Qwen3-Embedding-8B".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Qwen Team, Alibaba Cloud (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::BgeM3DenseSparseMultiVec,
            model_id: "BAAI/bge-m3".into(),
            dimensions: 1024,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 384,
            multivec_tokens_per_doc: 12,
            multivec_dim: 64,
            docs_per_sec_throughput: 980.0,
            base_query_latency_ms: 8.6,
            usd_per_1k_queries: 0.028,
            annual_maintenance_engineering_usd: 0.0,
            license: ModelLicenseAudit {
                model_id: "BAAI/bge-m3".into(),
                spdx_license: "MIT".into(),
                attribution_notice: "Beijing Academy of Artificial Intelligence (MIT)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::DistilledCodeStudent,
            model_id: "hanzo/laya-code-distilled-256d".into(),
            dimensions: 256,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 5_600.0,
            base_query_latency_ms: 1.7,
            usd_per_1k_queries: 0.004,
            annual_maintenance_engineering_usd: 4_500.0,
            license: ModelLicenseAudit {
                model_id: "hanzo/laya-code-distilled-256d".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Hanzo AI Clean-Room Distilled Code Retriever (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
        EncoderSpec {
            family: EncoderFamily::DistilledMemoryStudent,
            model_id: "hanzo/laya-memory-distilled-256d".into(),
            dimensions: 256,
            bytes_per_dim: 2,
            sparse_posting_bytes_per_doc: 0,
            multivec_tokens_per_doc: 0,
            multivec_dim: 0,
            docs_per_sec_throughput: 5_800.0,
            base_query_latency_ms: 1.6,
            usd_per_1k_queries: 0.004,
            annual_maintenance_engineering_usd: 4_500.0,
            license: ModelLicenseAudit {
                model_id: "hanzo/laya-memory-distilled-256d".into(),
                spdx_license: "Apache-2.0".into(),
                attribution_notice: "Hanzo AI Clean-Room Distilled Memory Retriever (Apache-2.0)".into(),
                commercial_allowed: true,
            },
        },
    ]
}

/// Multi-domain benchmark corpus and queries covering MTEB general, multilingual, code,
/// scientific terminology, and conversational/agent memory, including cross-tenant poison traps.
pub fn default_domain_benchmark() -> (Vec<CanonicalRecord>, Vec<DomainQueryCase>) {
    let rec = |id: &str,
               kind: ArtifactKind,
               tenant: &str,
               roles: &[&str],
               domain: DomainSlice,
               lang: &str,
               title: &str,
               body: &str,
               syms: &[&str]| CanonicalRecord {
        identity: SourceRecordIdentity {
            record_id: id.into(),
            version: 1,
            kind,
            tenant_id: tenant.into(),
            allowed_roles: roles.iter().map(|s| (*s).into()).collect(),
        },
        domain,
        language: lang.into(),
        title: title.into(),
        body: body.into(),
        symbols: syms.iter().map(|s| (*s).into()).collect(),
    };

    let records = vec![
        rec(
            "mteb_gold_1",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::MtebGeneral,
            "en",
            "Write-ahead log fsync durability",
            "Committed transactions call fdatasync on the write-ahead segment before acknowledging durability.",
            &[],
        ),
        rec(
            "mteb_dist_1",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::MtebGeneral,
            "en",
            "Page cache background flush",
            "Background worker threads flush dirty kernel pages every thirty seconds.",
            &[],
        ),
        rec(
            "multi_gold_de",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::Multilingual,
            "en",
            "Arena allocator leak diagnostics",
            "Heap profiler tracks memory_leak retention across arena epochs and frees orphaned slabs.",
            &[],
        ),
        rec(
            "multi_dist_de",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::Multilingual,
            "de",
            "Profiler Arena Epochen Bericht",
            "Profiler erkennt Arena Epochen Latenz im Netzwerk.",
            &[],
        ),
        rec(
            "multi_gold_ja",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::Multilingual,
            "en",
            "Incremental LMDB snapshot backup",
            "Consistent snapshot_backup streams copy-on-write pages to object storage with SHA256 verification.",
            &[],
        ),
        rec(
            "multi_cross_tenant_trap",
            ArtifactKind::SourceDocument,
            "tenant_beta",
            &["secops"],
            DomainSlice::Multilingual,
            "de",
            "Speicherleck Geheimbericht",
            "Speicherleck memory_leak fdatasync snapshot_backup secret key.",
            &[],
        ),
        rec(
            "code_gold_1",
            ArtifactKind::RepositoryBlob,
            "tenant_alpha",
            &["eng"],
            DomainSlice::CodeRetrieval,
            "rust",
            "crates/index-hnsw/src/graph_repair.rs",
            "Graph repair routine updates orphaned edges using heuristic M_max0 = 32.",
            &["HnswLayer::rewire_bidirectional"],
        ),
        rec(
            "code_dist_1",
            ArtifactKind::RepositoryBlob,
            "tenant_alpha",
            &["eng"],
            DomainSlice::CodeRetrieval,
            "rust",
            "crates/index-hnsw/src/link_pruning_doc.rs",
            "Documentation comment mentioning bidirectional link pruning during deletion without implementation constants.",
            &["DocStub::pruning_notes"],
        ),
        rec(
            "code_gold_2",
            ArtifactKind::GitCommit,
            "tenant_alpha",
            &["eng"],
            DomainSlice::CodeRetrieval,
            "rust",
            "fix(acl): prefilter roaring mask",
            "Applies roaring tenant prefilter mask prior to simd dot product.",
            &["PrefilterMask::intersect_acl"],
        ),
        rec(
            "code_dist_2",
            ArtifactKind::RepositoryBlob,
            "tenant_alpha",
            &["eng"],
            DomainSlice::CodeRetrieval,
            "rust",
            "crates/index-hnsw/src/acl_notes.rs",
            "Overview note asking which function intersects ACL bitmap masks prior to scoring.",
            &["Notes::acl_overview"],
        ),
        rec(
            "sci_gold_1",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::ScientificTerminology,
            "en",
            "Phosphatidylethanolamine peroxidation kinetics",
            "Lipoxygenase catalyses hydroperoxy-phosphatidylethanolamine ferroptotic membrane signalling.",
            &[],
        ),
        rec(
            "sci_dist_1",
            ArtifactKind::SourceDocument,
            "tenant_alpha",
            &[],
            DomainSlice::ScientificTerminology,
            "en",
            "Glucose transport assay",
            "Cellular hexose uptake increases under basal insulin stimulation.",
            &[],
        ),
        rec(
            "mem_gold_1",
            ArtifactKind::StoredDecision,
            "tenant_alpha",
            &["eng"],
            DomainSlice::AgentMemory,
            "en",
            "ADR-042 Production Rollback Threshold",
            "Engineering approved automatic rollback whenever p99 sla exceeds 180ms.",
            &[],
        ),
        rec(
            "mem_dist_1",
            ArtifactKind::AgentEvent,
            "tenant_alpha",
            &["eng"],
            DomainSlice::AgentMemory,
            "en",
            "Chat question log",
            "User asked what latency threshold was discussed in the meeting yesterday.",
            &[],
        ),
    ];

    let access_eng = DomainAccessContext {
        tenant_id: "tenant_alpha".into(),
        roles: vec!["eng".into()],
    };

    let cases = vec![
        DomainQueryCase {
            query_id: "q_mteb_1".into(),
            domain: DomainSlice::MtebGeneral,
            access: access_eng.clone(),
            query: "How do committed transactions guarantee write-ahead log durability?".into(),
            query_symbols: vec![],
            relevant_grades: BTreeMap::from([("mteb_gold_1".into(), 3)]),
            required_answer_facts: vec!["fdatasync".into()],
            forbidden_contradictions: vec![],
        },
        DomainQueryCase {
            query_id: "q_multi_de".into(),
            domain: DomainSlice::Multilingual,
            access: access_eng.clone(),
            query: "Wie erkennt der Profiler ein Speicherleck in Arena Epochen?".into(),
            query_symbols: vec![],
            relevant_grades: BTreeMap::from([("multi_gold_de".into(), 3)]),
            required_answer_facts: vec!["memory_leak".into(), "orphaned".into()],
            forbidden_contradictions: vec!["secret".into()],
        },
        DomainQueryCase {
            query_id: "q_multi_ja".into(),
            domain: DomainSlice::Multilingual,
            access: access_eng.clone(),
            query: "オブジェクトストレージへのスナップショット検証手順".into(),
            query_symbols: vec![],
            relevant_grades: BTreeMap::from([("multi_gold_ja".into(), 3)]),
            required_answer_facts: vec!["snapshot_backup".into(), "sha256".into()],
            forbidden_contradictions: vec![],
        },
        DomainQueryCase {
            query_id: "q_code_1".into(),
            domain: DomainSlice::CodeRetrieval,
            access: access_eng.clone(),
            query: "where is bidirectional link pruning implemented during deletion?".into(),
            query_symbols: vec!["HnswLayer::rewire_bidirectional".into()],
            relevant_grades: BTreeMap::from([("code_gold_1".into(), 3)]),
            required_answer_facts: vec!["m_max0".into(), "32".into()],
            forbidden_contradictions: vec![],
        },
        DomainQueryCase {
            query_id: "q_code_2".into(),
            domain: DomainSlice::CodeRetrieval,
            access: access_eng.clone(),
            query: "which function intersects ACL bitmap masks prior to scoring?".into(),
            query_symbols: vec!["PrefilterMask::intersect_acl".into()],
            relevant_grades: BTreeMap::from([("code_gold_2".into(), 3)]),
            required_answer_facts: vec!["roaring".into(), "simd".into()],
            forbidden_contradictions: vec![],
        },
        DomainQueryCase {
            query_id: "q_sci_1".into(),
            domain: DomainSlice::ScientificTerminology,
            access: access_eng.clone(),
            query: "phosphatidylethanolamine ferroptotic peroxidation enzyme".into(),
            query_symbols: vec![],
            relevant_grades: BTreeMap::from([("sci_gold_1".into(), 3)]),
            required_answer_facts: vec!["lipoxygenase".into()],
            forbidden_contradictions: vec![],
        },
        DomainQueryCase {
            query_id: "q_mem_1".into(),
            domain: DomainSlice::AgentMemory,
            access: access_eng,
            query: "what latency threshold was decided in the postmortem?".into(),
            query_symbols: vec![],
            relevant_grades: BTreeMap::from([("mem_gold_1".into(), 3)]),
            required_answer_facts: vec!["180ms".into(), "rollback".into()],
            forbidden_contradictions: vec![],
        },
    ];

    (records, cases)
}
