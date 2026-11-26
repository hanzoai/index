// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reproducible repository-grounded + multi-cluster benchmark corpus builder.
//!
//! Per `hanzoai/.github/PR_CONVERGENCE.md` (External Research & Bounty PR Policy):
//! - Extracts real text/code chunks from the public Apache-2.0/MIT `hanzoai/index`
//!   repository (`crates/**/*.rs`, `docs/**/*`, `README.md`) with deterministic
//!   canonical `source_id`s (`blob:<rel_path>#chunk<N>`).
//! - Generates deterministic L2-normalized embeddings (`--seed 1347`) combining
//!   FNV-1a token/subtoken hashing with semantic cluster geometry so both
//!   ANN neighborhood structure and downstream file-resolution accuracy are grounded
//!   in real repository content.
//! - Tags records deterministically across multiple tenants (`tenant-hanzo`,
//!   `tenant-enterprise`), ACL groups (`eng-core`, `sec-ops`, `public`), and exact
//!   selectivity buckets (`sel:25`, `sel:05`, `sel:01`) for matched filtered-ANN
//!   evaluation.

use std::fs;
use std::path::{Path, PathBuf};

use crate::canonical::{AccessContext, AnnError, CanonicalStore, CanonicalVectorRecord};
use crate::metrics::{normalize_l2, SplitMix64};

/// A benchmark query with caller ACL context and optional expected repo file target
/// for downstream task-success evaluation.
#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    pub query_id: String,
    pub vector: Vec<f32>,
    pub access: AccessContext,
    /// Selectivity label (`"100%"`, `"25%"`, `"5%"`, `"1%"`).
    pub selectivity_label: String,
    /// Optional target `source_id` prefix (real file path) for downstream file-resolution task.
    pub expected_file_prefix: Option<String>,
}

/// Complete deterministic benchmark workload built from repo files + multi-cluster expansion.
#[derive(Debug, Clone)]
pub struct BenchmarkCorpus {
    pub seed: u64,
    pub dim: usize,
    pub repo_chunks_loaded: usize,
    pub repo_files_scanned: usize,
    pub total_vectors: usize,
    pub store: CanonicalStore,
    pub queries_100: Vec<BenchmarkQuery>,
    pub queries_25: Vec<BenchmarkQuery>,
    pub queries_05: Vec<BenchmarkQuery>,
    pub queries_01: Vec<BenchmarkQuery>,
    /// Downstream agent code-lookup tasks grounded in real repository chunks.
    pub downstream_tasks: Vec<BenchmarkQuery>,
    /// Mutation batch: new records to insert, existing IDs to update, existing IDs to delete.
    pub mutation_inserts: Vec<CanonicalVectorRecord>,
    pub mutation_updates: Vec<CanonicalVectorRecord>,
    pub mutation_delete_ids: Vec<String>,
}

/// Configuration for building [`BenchmarkCorpus`].
#[derive(Debug, Clone)]
pub struct CorpusConfig {
    pub seed: u64,
    pub dim: usize,
    pub target_vectors: usize,
    pub num_clusters: usize,
    pub queries_per_selectivity: usize,
    pub downstream_task_count: usize,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            seed: 1347,
            dim: 64,
            target_vectors: 4096,
            num_clusters: 32,
            queries_per_selectivity: 60,
            downstream_task_count: 40,
        }
    }
}

/// Build the deterministic benchmark corpus, loading real source files from `repo_root`
/// when present and expanding to `config.target_vectors` with structured cluster geometry.
pub fn build_benchmark_corpus(
    repo_root: Option<&Path>,
    config: &CorpusConfig,
) -> Result<BenchmarkCorpus, AnnError> {
    let mut rng = SplitMix64::new(config.seed);
    let mut store = CanonicalStore::new(config.dim)?;

    // Generate deterministic cluster centers on the unit sphere.
    let mut cluster_centers: Vec<Vec<f32>> = Vec::with_capacity(config.num_clusters);
    for _ in 0..config.num_clusters {
        let mut c = vec![0.0_f32; config.dim];
        for v in &mut c {
            *v = rng.next_normal_f32();
        }
        normalize_l2(&mut c)?;
        cluster_centers.push(c);
    }

    // Step 1: Load real repository chunks from `hanzoai/index` (if repo_root exists).
    let mut repo_files_scanned = 0usize;
    let mut repo_chunks: Vec<(String, String, String)> = Vec::new();
    if let Some(root) = repo_root {
        let mut files = Vec::new();
        collect_repo_source_files(root, root, &mut files);
        files.sort();
        for abs_path in files {
            if repo_chunks.len() >= (config.target_vectors / 2).max(512) {
                break;
            }
            if let Ok(content) = fs::read_to_string(&abs_path) {
                let rel = abs_path
                    .strip_prefix(root)
                    .unwrap_or(&abs_path)
                    .to_string_lossy()
                    .replace('\\', "/");
                repo_files_scanned += 1;
                let chunks = chunk_text_lines(&content, 28);
                for (ci, text) in chunks.into_iter().enumerate() {
                    if text.trim().len() < 40 {
                        continue;
                    }
                    let source_id = format!("blob:{rel}#chunk{ci}");
                    repo_chunks.push((source_id, rel.clone(), text));
                    if repo_chunks.len() >= (config.target_vectors / 2).max(512) {
                        break;
                    }
                }
            }
        }
    }

    let repo_chunks_loaded = repo_chunks.len();
    let mut primary_records: Vec<CanonicalVectorRecord> = Vec::with_capacity(config.target_vectors);
    let mut repo_task_sources: Vec<(String, String, Vec<f32>, String, String)> = Vec::new();

    // Ingest real repository chunks first.
    for (idx, (source_id, rel_path, text)) in repo_chunks.into_iter().enumerate() {
        let cluster_id = (fnv1a_64(rel_path.as_bytes()) as usize) % config.num_clusters;
        let vec = embed_text_with_cluster(&text, &cluster_centers[cluster_id], config.dim, 0.55)?;
        let (tenant_id, acl_group) = assign_tenant_and_acl(idx);
        let mut tags = assign_selectivity_tags(idx);
        tags.push("origin:repo".to_string());
        if rel_path.ends_with(".rs") {
            tags.push("lang:rust".to_string());
        } else if rel_path.ends_with(".md") || rel_path.ends_with(".mdx") {
            tags.push("lang:docs".to_string());
        }

        let rec = CanonicalVectorRecord {
            source_id: source_id.clone(),
            version: 1,
            tenant_id: tenant_id.clone(),
            acl_group: acl_group.clone(),
            filter_tags: tags,
            vector: vec.clone(),
        };
        if tenant_id == "tenant-hanzo" && (acl_group == "eng-core" || acl_group == "public") {
            repo_task_sources.push((
                source_id,
                format!("blob:{rel_path}#"),
                vec,
                tenant_id,
                acl_group,
            ));
        }
        store.upsert(rec.clone())?;
        primary_records.push(rec);
    }

    // Step 2: Expand to `config.target_vectors` with structured multi-cluster benchmark records.
    for idx in repo_chunks_loaded..config.target_vectors {
        let cluster_id = idx % config.num_clusters;
        let mut vec = cluster_centers[cluster_id].clone();
        for v in &mut vec {
            *v += rng.next_normal_f32() * 0.24;
        }
        normalize_l2(&mut vec)?;

        let (tenant_id, acl_group) = assign_tenant_and_acl(idx);
        let mut tags = assign_selectivity_tags(idx);
        tags.push("origin:cluster-bench".to_string());
        tags.push(format!("cluster:{cluster_id:02}"));

        let rec = CanonicalVectorRecord {
            source_id: format!("bench:cluster-{cluster_id:02}/doc-{idx:06}"),
            version: 1,
            tenant_id,
            acl_group,
            filter_tags: tags,
            vector: vec,
        };
        store.upsert(rec.clone())?;
        primary_records.push(rec);
    }

    // Build deterministic matched queries for 100%, 25%, 5%, and 1% selectivities
    // from the validated CanonicalStore snapshot (guaranteed sorted filter_tags).
    // All queries authenticate as `tenant-hanzo` with `["eng-core", "public"]`, while
    // `tenant-enterprise` and `sec-ops` records remain present in the same derived indexes
    // as active adversarial cross-tenant / elevated-ACL distractors!
    let base_tenant = "tenant-hanzo".to_string();
    let base_acls = vec!["eng-core".to_string(), "public".to_string()];

    let live_snapshot = store.snapshot_live();
    let _ = primary_records;
    let authorized_pool: Vec<&CanonicalVectorRecord> = live_snapshot
        .iter()
        .filter(|r| r.tenant_id == base_tenant && base_acls.contains(&r.acl_group))
        .collect();

    let queries_100 = build_queries_for_selectivity(
        &authorized_pool,
        &base_tenant,
        &base_acls,
        None,
        "100%",
        config.queries_per_selectivity,
        config.seed ^ 0x100,
    )?;
    let queries_25 = build_queries_for_selectivity(
        &authorized_pool,
        &base_tenant,
        &base_acls,
        Some("sel:25"),
        "25%",
        config.queries_per_selectivity,
        config.seed ^ 0x25,
    )?;
    let queries_05 = build_queries_for_selectivity(
        &authorized_pool,
        &base_tenant,
        &base_acls,
        Some("sel:05"),
        "5%",
        config.queries_per_selectivity,
        config.seed ^ 0x05,
    )?;
    let queries_01 = build_queries_for_selectivity(
        &authorized_pool,
        &base_tenant,
        &base_acls,
        Some("sel:01"),
        "1%",
        config.queries_per_selectivity,
        config.seed ^ 0x01,
    )?;

    // Build downstream agent file-resolution tasks grounded in real repo chunks.
    let mut downstream_tasks = Vec::new();
    let mut task_rng = SplitMix64::new(config.seed ^ 0x7a5c_cafe);
    if !repo_task_sources.is_empty() {
        let count = config.downstream_task_count.min(repo_task_sources.len());
        let step = (repo_task_sources.len() / count).max(1);
        for (ti, idx) in (0..repo_task_sources.len()).step_by(step).take(count).enumerate() {
            let (_sid, file_prefix, base_vec, tenant, acl) = &repo_task_sources[idx];
            let mut qv = base_vec.clone();
            // Simulate paraphrased agent query drift around the true code chunk embedding.
            for x in &mut qv {
                *x += task_rng.next_normal_f32() * 0.14;
            }
            normalize_l2(&mut qv)?;
            downstream_tasks.push(BenchmarkQuery {
                query_id: format!("repo-task-{ti:03}"),
                vector: qv,
                access: AccessContext::new(
                    tenant.clone(),
                    vec![acl.clone(), "public".to_string()],
                    Some("origin:repo".to_string()),
                ),
                selectivity_label: "repo-task".to_string(),
                expected_file_prefix: Some(file_prefix.clone()),
            });
        }
    }

    // Build deterministic online mutation workload:
    // - 5% fresh inserts
    // - 5% version-2 updates to existing authorized records
    // - 10% deletions of existing records
    let insert_count = (config.target_vectors / 20).max(20);
    let update_count = (config.target_vectors / 20).max(20);
    let delete_count = (config.target_vectors / 10).max(40);

    let mut mut_rng = SplitMix64::new(config.seed ^ 0xdead_beef);
    let mut mutation_inserts = Vec::with_capacity(insert_count);
    for i in 0..insert_count {
        let cid = i % config.num_clusters;
        let mut vec = cluster_centers[cid].clone();
        for v in &mut vec {
            *v += mut_rng.next_normal_f32() * 0.20;
        }
        normalize_l2(&mut vec)?;
        mutation_inserts.push(CanonicalVectorRecord {
            source_id: format!("fresh:insert-{i:05}"),
            version: 1,
            tenant_id: base_tenant.clone(),
            acl_group: "eng-core".to_string(),
            filter_tags: assign_selectivity_tags(i),
            vector: vec,
        });
    }

    let mut mutation_updates = Vec::with_capacity(update_count);
    let mut mutation_delete_ids = Vec::with_capacity(delete_count);
    for (i, rec) in authorized_pool.iter().enumerate() {
        if i < update_count {
            let mut vec = rec.vector.clone();
            for v in &mut vec {
                *v += mut_rng.next_normal_f32() * 0.12;
            }
            normalize_l2(&mut vec)?;
            mutation_updates.push(CanonicalVectorRecord {
                source_id: rec.source_id.clone(),
                version: rec.version + 1,
                tenant_id: rec.tenant_id.clone(),
                acl_group: rec.acl_group.clone(),
                filter_tags: rec.filter_tags.clone(),
                vector: vec,
            });
        } else if i < update_count + delete_count {
            mutation_delete_ids.push(rec.source_id.clone());
        } else {
            break;
        }
    }

    Ok(BenchmarkCorpus {
        seed: config.seed,
        dim: config.dim,
        repo_chunks_loaded,
        repo_files_scanned,
        total_vectors: store.len(),
        store,
        queries_100,
        queries_25,
        queries_05,
        queries_01,
        downstream_tasks,
        mutation_inserts,
        mutation_updates,
        mutation_delete_ids,
    })
}

fn assign_tenant_and_acl(idx: usize) -> (String, String) {
    // 70% tenant-hanzo (50% eng-core, 15% public, 5% sec-ops restricted),
    // 30% tenant-enterprise (active cross-tenant distractor records).
    match idx % 20 {
        0..=9 => ("tenant-hanzo".to_string(), "eng-core".to_string()),
        10..=12 => ("tenant-hanzo".to_string(), "public".to_string()),
        13 => ("tenant-hanzo".to_string(), "sec-ops".to_string()),
        14..=17 => ("tenant-enterprise".to_string(), "eng-core".to_string()),
        _ => ("tenant-enterprise".to_string(), "sec-ops".to_string()),
    }
}

fn assign_selectivity_tags(idx: usize) -> Vec<String> {
    let mut tags = vec!["sel:100".to_string()];
    if idx.is_multiple_of(4) {
        tags.push("sel:25".to_string());
    }
    if idx.is_multiple_of(20) {
        tags.push("sel:05".to_string());
    }
    if idx.is_multiple_of(100) {
        tags.push("sel:01".to_string());
    }
    tags
}

fn build_queries_for_selectivity(
    authorized_pool: &[&CanonicalVectorRecord],
    tenant_id: &str,
    acls: &[String],
    required_tag: Option<&str>,
    label: &str,
    count: usize,
    seed: u64,
) -> Result<Vec<BenchmarkQuery>, AnnError> {
    let mut rng = SplitMix64::new(seed);
    let eligible: Vec<&CanonicalVectorRecord> = match required_tag {
        Some(tag) => authorized_pool.iter().copied().filter(|r| r.has_tag(tag)).collect(),
        None => authorized_pool.to_vec(),
    };
    if eligible.is_empty() {
        return Ok(Vec::new());
    }

    let mut queries = Vec::with_capacity(count);
    for qi in 0..count {
        let anchor = eligible[(qi * 17 + rng.next_usize(eligible.len())) % eligible.len()];
        let mut qv = anchor.vector.clone();
        for x in &mut qv {
            *x += rng.next_normal_f32() * 0.16;
        }
        normalize_l2(&mut qv)?;
        queries.push(BenchmarkQuery {
            query_id: format!("q-{label}-{qi:03}"),
            vector: qv,
            access: AccessContext::new(
                tenant_id.to_string(),
                acls.to_vec(),
                required_tag.map(str::to_string),
            ),
            selectivity_label: label.to_string(),
            expected_file_prefix: None,
        });
    }
    Ok(queries)
}

fn embed_text_with_cluster(
    text: &str,
    cluster_center: &[f32],
    dim: usize,
    cluster_weight: f32,
) -> Result<Vec<f32>, AnnError> {
    let mut lexical = vec![0.0_f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|t| t.len() >= 3) {
        let h = fnv1a_64(token.to_ascii_lowercase().as_bytes());
        let bucket = (h as usize) % dim;
        let sign = if (h >> 63) == 0 { 1.0_f32 } else { -1.0_f32 };
        lexical[bucket] += sign;
    }
    if normalize_l2(&mut lexical).is_err() {
        lexical = cluster_center.to_vec();
    }
    let lex_w = 1.0 - cluster_weight;
    let mut out = vec![0.0_f32; dim];
    for d in 0..dim {
        out[d] = cluster_weight * cluster_center[d] + lex_w * lexical[d];
    }
    normalize_l2(&mut out)?;
    Ok(out)
}

fn collect_repo_source_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|x| x.path())).collect();
    paths.sort();

    for path in paths {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            // Focus on crates/ and docs/ so corpus stays fast and cleanly licensed.
            if dir == root && name != "crates" && name != "docs" {
                continue;
            }
            collect_repo_source_files(root, &path, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "rs" | "md" | "mdx") {
                out.push(path);
            }
        }
    }
}

fn chunk_text_lines(content: &str, lines_per_chunk: usize) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let end = (i + lines_per_chunk).min(lines.len());
        out.push(lines[i..end].join("\n"));
        i = end;
    }
    out
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
