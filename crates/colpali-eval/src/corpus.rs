use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::canonical::CanonicalPageStore;
use crate::math::{l2_normalize, seeded_unit_vector, tokenize_text, SplitMix64};
use crate::types::{
    AccessScope, CanonicalPageRecord, ColPaliError, PageKey, PageLayoutFeatures, PageModality,
    VisualQuery, DEFAULT_EMBEDDING_DIM,
};

/// Benchmark query with ground-truth relevance judgments (used ONLY for evaluation, never
/// exposed to retrieval indexes).
#[derive(Debug, Clone)]
pub struct EvalQuery {
    pub query: VisualQuery,
    /// Relevant canonical page keys with graded relevance (`2` = primary exact target, `1` = related).
    pub relevance: BTreeMap<PageKey, u8>,
    pub target_modality: PageModality,
    pub is_exact_keyword_negative_regime: bool,
}

/// Transparent corpus provenance and measured OCR corruption rates per modality.
#[derive(Debug, Clone)]
pub struct CorpusProvenance {
    pub seed: u64,
    pub total_pages: usize,
    pub patches_per_page: usize,
    pub dim: usize,
    pub repo_files_loaded: Vec<String>,
    pub modality_page_counts: BTreeMap<PageModality, usize>,
    pub mean_ocr_corruption_by_modality: BTreeMap<PageModality, f64>,
    pub tenant_page_counts: BTreeMap<String, usize>,
    pub acl_page_counts: BTreeMap<String, usize>,
}

/// Complete deterministic multi-modal benchmark dataset.
#[derive(Debug, Clone)]
pub struct BenchmarkCorpus {
    pub store: CanonicalPageStore,
    pub modality_queries: Vec<EvalQuery>,
    pub exact_keyword_queries: Vec<EvalQuery>,
    pub provenance: CorpusProvenance,
}

const REPO_DOC_CANDIDATES: &[&str] = &[
    "docs/index.mdx",
    "BENCHMARKS.md",
    "README.md",
    "documentation/experimental-features.md",
    "documentation/prototypes.md",
    "documentation/release.md",
    "documentation/versioning-policy.md",
    "SECURITY.md",
    "TESTING.md",
    "Cargo.toml",
];

const DOMAINS: &[&str] = &[
    "vector_ann_index",
    "hybrid_lexical_rrf",
    "s3_vfs_snapshots",
    "binary_sq8_quantization",
    "tenant_acl_isolation",
    "chat_rag_grounding",
    "distributed_shard_scheduler",
    "latency_cost_profiling",
];

const TABLE_ROWS: &[&str] = &[
    "p99_cold_latency_ms",
    "recall_at_5_score",
    "ram_bytes_per_page",
    "s3_range_get_iops",
    "throughput_qps_peak",
    "compaction_wall_ms",
    "ndcg_at_5_vidore",
    "monthly_dram_usd",
];

const TABLE_COLS: &[&str] = &[
    "ocr_bm25_baseline",
    "colpali_fp32_full",
    "colpali_bin1_sq8",
    "hybrid_visual_router",
    "ivf_pq_object_native",
    "vamana_mmap_local",
];

const TABLE_VALS: &[&str] = &[
    "0.942", "18.4ms", "16.4KB", "527.3KB", "32.0x", "0.618", "1.82ms", "99.4pct", "412.5qps",
    "2.31usd",
];

const CHART_TYPES: &[&str] = &[
    "pareto_frontier_curve",
    "grouped_bar_comparison",
    "log_scale_latency_plot",
    "architecture_flow_diagram",
    "scatter_recall_footprint",
    "ablation_waterfall_chart",
];

const CHART_AXES: &[&str] = &[
    "x_compression_ratio_vs_y_ndcg5",
    "x_page_count_1m_vs_y_ram_gb",
    "x_shortlist_k_vs_y_p99_ms",
    "x_ocr_noise_rate_vs_y_recall5",
    "x_selectivity_pct_vs_y_qps",
    "x_patch_grid_size_vs_y_disk_gb",
];

const CHART_TRENDS: &[&str] = &[
    "flat_plateau_up_to_32x_compression",
    "sharp_knee_below_0.28_complexity",
    "linear_storage_explosion_fp32",
    "crossover_point_at_100k_pages",
    "monotonic_ocr_collapse_above_60pct",
    "sublinear_popcount_scaling",
];

const FORMULA_FAMILIES: &[&str] = &[
    "maxsim_late_interaction",
    "hamming_arccos_simhash_bound",
    "bm25_idf_length_norm",
    "sq8_symmetric_quant_error",
    "fwht_orthogonal_energy_spread",
    "pareto_cost_ram_s3_objective",
];

const FORMULA_OPERATORS: &[&str] = &[
    "sum_i_max_j_inner_q_p",
    "one_minus_two_over_pi_arccos",
    "k1_plus_one_tf_over_dl_avgdl",
    "max_abs_over_127_round_clamp",
    "hadamard_inv_sqrt_d_norm_preserve",
    "alpha_dram_plus_beta_s3_plus_gamma_get",
];

const FORMULA_BOUNDS: &[&str] = &[
    "O_q_n_p_d_over_64_popcnt",
    "eps_le_0.008_per_dim_sq8",
    "32x_ram_compression_exact",
    "zero_cross_tenant_leakage_invariant",
    "bit_exact_drop_rebuild_parity",
    "1_rtt_bounded_range_fetch",
];

/// Load real public vocabulary chunks from `hanzoai/index` documentation files.
fn load_repo_doc_vocabulary(repo_root: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let mut loaded_files = Vec::new();
    let mut doc_token_pools: Vec<Vec<String>> = Vec::new();

    for rel in REPO_DOC_CANDIDATES {
        let full = repo_root.join(rel);
        if let Ok(content) = fs::read_to_string(&full) {
            let tokens: Vec<String> = tokenize_text(&content)
                .into_iter()
                .filter(|t| t.len() >= 3 && t.len() <= 32)
                .take(180)
                .collect();
            if !tokens.is_empty() {
                loaded_files.push((*rel).to_string());
                doc_token_pools.push(tokens);
            }
        }
    }

    if doc_token_pools.is_empty() {
        // Fallback excerpts directly from hanzoai/index docs if executed outside the repo root.
        loaded_files.push("embedded:hanzoai/index:docs/index.mdx".to_string());
        doc_token_pools.push(tokenize_text(
            "hanzo index rust search hybrid semantic full-text faceted filtering vector hnsw \
             embedders prometheus metrics s3 snapshots multi-tenancy api key permissions milli \
             index-scheduler search-auth file-store",
        ));
    }

    (loaded_files, doc_token_pools)
}

/// Build a deterministic multi-modal scientific & engineering PDF benchmark corpus grounded
/// in real `hanzoai/index` documentation and ViDoRe-style visual layout regimes.
pub fn build_benchmark_corpus(
    repo_root: &Path,
    seed: u64,
    total_pages: usize,
    patches_per_page: usize,
    dim: usize,
    queries_per_slice: usize,
) -> Result<BenchmarkCorpus, ColPaliError> {
    let total_pages = total_pages.max(32);
    let patches_per_page = patches_per_page.max(16);
    let dim = if dim == 0 { DEFAULT_EMBEDDING_DIM } else { dim };

    let (repo_files_loaded, doc_token_pools) = load_repo_doc_vocabulary(repo_root);
    let mut rng = SplitMix64::new(seed ^ 0x51a7_c01b_a11f_1347);
    let mut store = CanonicalPageStore::new(dim);

    // Realistic distribution: 40% TextOnly, 20% DenseTable, 20% ScientificFigure, 20% MultiColumnFormula.
    let n_text = (total_pages * 40) / 100;
    let n_table = (total_pages * 20) / 100;
    let n_figure = (total_pages * 20) / 100;
    let n_formula = total_pages - n_text - n_table - n_figure;

    let mut page_specs: Vec<PageModality> = Vec::with_capacity(total_pages);
    page_specs.extend(std::iter::repeat_n(PageModality::TextOnly, n_text));
    page_specs.extend(std::iter::repeat_n(PageModality::DenseTable, n_table));
    page_specs.extend(std::iter::repeat_n(PageModality::ScientificFigure, n_figure));
    page_specs.extend(std::iter::repeat_n(PageModality::MultiColumnFormula, n_formula));

    let grid_cols = (patches_per_page as f64).sqrt().ceil() as u16;
    let grid_rows = (patches_per_page as u16).div_ceil(grid_cols.max(1));

    struct PageTruth {
        key: PageKey,
        modality: PageModality,
        tenant_id: String,
        acl_group: String,
        domain: &'static str,
        exact_ident: String,
        concept_labels: Vec<String>,
        query_prompt_text: String,
        exact_keyword_prompt: String,
    }

    let mut truths: Vec<PageTruth> = Vec::with_capacity(total_pages);
    let mut modality_counts: BTreeMap<PageModality, usize> = BTreeMap::new();
    let mut modality_ocr_corr_sum: BTreeMap<PageModality, f64> = BTreeMap::new();
    let mut tenant_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut acl_counts: BTreeMap<String, usize> = BTreeMap::new();

    for (idx, modality) in page_specs.into_iter().enumerate() {
        let domain = DOMAINS[idx % DOMAINS.len()];
        let pool_idx = idx % doc_token_pools.len();
        let pool = &doc_token_pools[pool_idx];
        let source_file = &repo_files_loaded[pool_idx];

        // Multi-tenant + multi-ACL assignment:
        // ~75% tenant-hanzo, ~25% tenant-external-lab; within tenant-hanzo: ~45% research-public,
        // ~35% eng-core, ~20% sec-confidential (unauthorized for default eval scope!).
        let tenant_id = if idx % 4 == 3 {
            "tenant-external-lab".to_string()
        } else {
            "tenant-hanzo".to_string()
        };
        let acl_group = match idx % 10 {
            0..=4 => "research-public".to_string(),
            5..=7 => "eng-core".to_string(),
            _ => "sec-confidential".to_string(),
        };

        let doc_id = format!("hanzo-sci-{}-{:03}", modality.as_str(), idx / 4);
        let page_no = (idx % 4) as u32 + 1;
        let version = 1 + (idx as u64 % 3);

        // Sample 6 real repo vocabulary tokens for realistic document prose context.
        let mut repo_snippet = Vec::with_capacity(6);
        for k in 0..6 {
            let tok = &pool[(idx * 7 + k * 13) % pool.len()];
            repo_snippet.push(tok.clone());
        }

        let exact_ident = format!("hanzo_{}_cfg_{:04x}", &domain[..4], 0x1000 + idx);
        let alt_code_sym = format!("crates::milli::hip1347_mod_{:03}", idx);

        let (
            layout_features,
            ocr_text,
            ocr_corruption_rate,
            concept_labels,
            query_prompt_text,
            exact_keyword_prompt,
        ) = match modality {
            PageModality::TextOnly => {
                let sub_topic = format!("text_spec_{}_{}", domain, idx % 37);
                let flag_token = format!("--hanzo-flag-{:03}", idx);
                let layout = PageLayoutFeatures {
                    text_coverage: 0.86 + 0.08 * rng.next_f32(),
                    table_region_ratio: 0.01 * rng.next_f32(),
                    figure_region_ratio: 0.01 * rng.next_f32(),
                    formula_region_ratio: 0.01 * rng.next_f32(),
                    ocr_mean_confidence: 0.96 + 0.03 * rng.next_f32(),
                    column_count: 1,
                };
                let corruption = 0.015 + 0.02 * rng.next_f32();
                let ocr = format!(
                        "{domain} specification {sub_topic} {exact_ident} {alt_code_sym} {flag_token} {} {} {} {} \
                         canonical record identity pre-return acl verification deterministic configuration",
                        repo_snippet[0], repo_snippet[1], repo_snippet[2], repo_snippet[3]
                    );
                let concepts = vec![
                    format!("domain:{domain}"),
                    format!("prose:{sub_topic}"),
                    format!("repo_term:{}", repo_snippet[0]),
                    format!("repo_term:{}", repo_snippet[1]),
                ];
                let q_prompt = format!(
                    "{domain} {sub_topic} {} {} canonical configuration",
                    repo_snippet[0], repo_snippet[1]
                );
                let exact_prompt =
                    format!("exact config {exact_ident} {alt_code_sym} {flag_token}");
                (layout, ocr, corruption, concepts, q_prompt, exact_prompt)
            }
            PageModality::DenseTable => {
                let r1 = TABLE_ROWS[idx % TABLE_ROWS.len()];
                let c1 = TABLE_COLS[(idx / 3 + 1) % TABLE_COLS.len()];
                let v1 = TABLE_VALS[(idx * 5 + 2) % TABLE_VALS.len()];
                let r2 = TABLE_ROWS[(idx + 3) % TABLE_ROWS.len()];
                let c2 = TABLE_COLS[(idx / 5 + 2) % TABLE_COLS.len()];
                let v2 = TABLE_VALS[(idx * 3 + 7) % TABLE_VALS.len()];
                let table_id = format!("tbl_{:03}", idx);

                let layout = PageLayoutFeatures {
                    text_coverage: 0.25 + 0.10 * rng.next_f32(),
                    table_region_ratio: 0.52 + 0.18 * rng.next_f32(),
                    figure_region_ratio: 0.04 * rng.next_f32(),
                    formula_region_ratio: 0.03 * rng.next_f32(),
                    ocr_mean_confidence: 0.56 + 0.12 * rng.next_f32(),
                    column_count: 3,
                };
                // Realistic multi-column table OCR corruption (~60% binding loss):
                // OCR loses row-column alignment, mangles digits/decimal points, and drops header bindings.
                let corruption = 0.58 + 0.10 * rng.next_f32();
                let garbled_v1 = mangle_ocr_token(v1);
                let garbled_v2 = mangle_ocr_token(v2);
                let ocr = format!(
                        "summary {domain} {} {} benchmark numbers {garbled_v1} 0.812 {garbled_v2} 14.9 \
                         column merged total avg std dev page {}",
                        repo_snippet[0], repo_snippet[2], page_no
                    );
                let concepts = vec![
                    format!("domain:{domain}"),
                    format!("table_cell:{r1}::{c1}::{v1}"),
                    format!("table_cell:{r2}::{c2}::{v2}"),
                    format!("table_layout:{c1}_{c2}"),
                ];
                let q_prompt = format!(
                    "table comparison {domain} row {r1} column {c1} value {v1} vs {r2} {c2} {v2}"
                );
                let exact_prompt = format!("table {table_id} {exact_ident}");
                (layout, ocr, corruption, concepts, q_prompt, exact_prompt)
            }
            PageModality::ScientificFigure => {
                let chart = CHART_TYPES[idx % CHART_TYPES.len()];
                let axes = CHART_AXES[(idx / 2 + 1) % CHART_AXES.len()];
                let trend = CHART_TRENDS[(idx / 3 + 2) % CHART_TRENDS.len()];
                let series = TABLE_COLS[(idx + 4) % TABLE_COLS.len()];
                let fig_id = format!("fig_{:03}", idx);

                let layout = PageLayoutFeatures {
                    text_coverage: 0.18 + 0.08 * rng.next_f32(),
                    table_region_ratio: 0.04 * rng.next_f32(),
                    figure_region_ratio: 0.58 + 0.20 * rng.next_f32(),
                    formula_region_ratio: 0.02 * rng.next_f32(),
                    ocr_mean_confidence: 0.44 + 0.12 * rng.next_f32(),
                    column_count: 2,
                };
                // Realistic scientific chart OCR degradation (~78% loss):
                // OCR sees only sparse caption text; plotted curves, rotated axis ticks,
                // legend colors, and Pareto knees are purely visual.
                let corruption = 0.74 + 0.10 * rng.next_f32();
                let ocr = format!(
                    "figure caption {domain} {} results evaluation overview legend n/a axis",
                    repo_snippet[1]
                );
                let concepts = vec![
                    format!("domain:{domain}"),
                    format!("chart_kind:{chart}"),
                    format!("chart_axes:{axes}"),
                    format!("chart_trend:{trend}"),
                    format!("chart_series:{series}"),
                ];
                let q_prompt = format!(
                        "scientific figure {chart} on {domain} plotting {axes} showing {trend} for {series}"
                    );
                let exact_prompt = format!("figure {fig_id} {exact_ident}");
                (layout, ocr, corruption, concepts, q_prompt, exact_prompt)
            }
            PageModality::MultiColumnFormula => {
                let fam = FORMULA_FAMILIES[idx % FORMULA_FAMILIES.len()];
                let op = FORMULA_OPERATORS[(idx / 2 + 3) % FORMULA_OPERATORS.len()];
                let bound = FORMULA_BOUNDS[(idx / 3 + 1) % FORMULA_BOUNDS.len()];
                let eq_id = format!("eq_{:03}", idx);

                let layout = PageLayoutFeatures {
                    text_coverage: 0.36 + 0.10 * rng.next_f32(),
                    table_region_ratio: 0.03 * rng.next_f32(),
                    figure_region_ratio: 0.05 * rng.next_f32(),
                    formula_region_ratio: 0.44 + 0.18 * rng.next_f32(),
                    ocr_mean_confidence: 0.52 + 0.12 * rng.next_f32(),
                    column_count: 2,
                };
                // Realistic two-column math OCR degradation (~65% loss):
                // Left/right column interleaving + dropped Greek/integral/subscript tokens.
                let corruption = 0.62 + 0.10 * rng.next_f32();
                let ocr = format!(
                    "section {domain} {} proof follows from left col right col x i j sum ? ? \
                         bound holds asymptotically {}",
                    repo_snippet[3], repo_snippet[4]
                );
                let concepts = vec![
                    format!("domain:{domain}"),
                    format!("formula_family:{fam}"),
                    format!("formula_op:{op}"),
                    format!("formula_bound:{bound}"),
                ];
                let q_prompt = format!(
                    "two-column formula proof {fam} equation {op} with bound {bound} in {domain}"
                );
                let exact_prompt = format!("equation {eq_id} {exact_ident}");
                (layout, ocr, corruption, concepts, q_prompt, exact_prompt)
            }
        };

        // 1. Single-vector OCR dense embedding derived strictly from `ocr_text`!
        let ocr_dense_embedding = embed_ocr_text(seed, &ocr_text, dim);

        // 2. Multi-vector visual page patches (`patches_per_page` localized patches on an 8x8 grid).
        // Each salient concept occupies dedicated spatial patches (preserving exact 2D table/figure/
        // equation bindings), while background/margin patches carry layout & template noise.
        // Importantly, for TextOnly pages, fine-grained exact alphanumeric identifiers (`cfg_1042`)
        // undergo realistic 448x448 VLM patch resolution blur (mapped to a shared template bucket
        // rather than exact character-level discrimination).
        let patch_embeddings = build_visual_page_patches(
            seed,
            &mut rng,
            modality,
            domain,
            &concept_labels,
            patches_per_page,
            dim,
        );

        *modality_counts.entry(modality).or_insert(0) += 1;
        *modality_ocr_corr_sum.entry(modality).or_insert(0.0) += f64::from(ocr_corruption_rate);
        *tenant_counts.entry(tenant_id.clone()).or_insert(0) += 1;
        *acl_counts.entry(acl_group.clone()).or_insert(0) += 1;

        let record = CanonicalPageRecord {
            doc_id: doc_id.clone(),
            page_no,
            version,
            tenant_id: tenant_id.clone(),
            acl_group: acl_group.clone(),
            page_modality: modality,
            source_origin: format!("hanzoai/index:{source_file}"),
            layout_features,
            ocr_text,
            ocr_corruption_rate,
            ocr_dense_embedding,
            patch_grid_rows: grid_rows,
            patch_grid_cols: grid_cols,
            patch_embeddings,
        };
        store.upsert(record)?;

        truths.push(PageTruth {
            key: PageKey::new(doc_id, page_no),
            modality,
            tenant_id,
            acl_group,
            domain,
            exact_ident,
            concept_labels,
            query_prompt_text,
            exact_keyword_prompt,
        });
    }

    // Default evaluation access scope: `tenant-hanzo` with `research-public` + `eng-core`.
    // Pages in `sec-confidential` or `tenant-external-lab` MUST NEVER be returned.
    let eval_scope = AccessScope::new(
        "tenant-hanzo",
        vec!["research-public".to_string(), "eng-core".to_string()],
    );

    // Build balanced evaluation queries across all 4 modalities + Negative Regime 1 exact keywords.
    let mut modality_queries = Vec::new();
    let mut q_counter = 0usize;
    for modality in PageModality::all() {
        let eligible: Vec<&PageTruth> = truths
            .iter()
            .filter(|t| t.modality == modality && eval_scope.permits(&t.tenant_id, &t.acl_group))
            .collect();

        let count = queries_per_slice.min(eligible.len());
        let stride = (eligible.len() / count.max(1)).max(1);
        for i in 0..count {
            let target = eligible[(i * stride) % eligible.len()];
            q_counter += 1;

            let dense_embedding = embed_ocr_text(seed, &target.query_prompt_text, dim);
            let token_embeddings = build_query_token_embeddings(
                seed,
                &mut rng,
                &target.concept_labels,
                &target.query_prompt_text,
                false,
                dim,
            );

            let mut relevance = BTreeMap::new();
            for candidate in &eligible {
                if candidate.query_prompt_text == target.query_prompt_text
                    && candidate.concept_labels == target.concept_labels
                {
                    relevance.insert(candidate.key.clone(), 2);
                }
            }

            modality_queries.push(EvalQuery {
                query: VisualQuery {
                    query_id: format!("q_{}_{:03}", modality.as_str(), q_counter),
                    text: target.query_prompt_text.clone(),
                    dense_embedding,
                    token_embeddings,
                    scope: eval_scope.clone(),
                    top_k: 5,
                },
                relevance,
                target_modality: modality,
                is_exact_keyword_negative_regime: false,
            });
        }
    }

    // Explicit Negative Regime 1: exact keyword / alphanumeric identifier queries on TextOnly pages.
    let text_eligible: Vec<&PageTruth> = truths
        .iter()
        .filter(|t| {
            t.modality == PageModality::TextOnly && eval_scope.permits(&t.tenant_id, &t.acl_group)
        })
        .collect();
    let mut exact_keyword_queries = Vec::new();
    let exact_count = queries_per_slice.min(text_eligible.len());
    let stride = (text_eligible.len() / exact_count.max(1)).max(1);
    for i in 0..exact_count {
        let target = text_eligible[(i * stride) % text_eligible.len()];
        let dense_embedding = embed_ocr_text(seed, &target.exact_keyword_prompt, dim);
        // Realistic VLM query tokens on an exact alphanumeric identifier: carries domain +
        // coarse visual identifier template, NOT exact character-by-character hash discrimination.
        let coarse_visual_concepts = vec![
            format!("domain:{}", target.domain),
            format!("ident_template:{}", &target.exact_ident[..10]),
            "visual_prose_code_block".to_string(),
        ];
        let token_embeddings = build_query_token_embeddings(
            seed,
            &mut rng,
            &coarse_visual_concepts,
            &target.exact_keyword_prompt,
            true,
            dim,
        );

        let mut relevance = BTreeMap::new();
        relevance.insert(target.key.clone(), 2);

        exact_keyword_queries.push(EvalQuery {
            query: VisualQuery {
                query_id: format!("q_neg1_exact_{:03}", i + 1),
                text: target.exact_keyword_prompt.clone(),
                dense_embedding,
                token_embeddings,
                scope: eval_scope.clone(),
                top_k: 5,
            },
            relevance,
            target_modality: PageModality::TextOnly,
            is_exact_keyword_negative_regime: true,
        });
    }

    let mut mean_ocr_corruption_by_modality = BTreeMap::new();
    for modality in PageModality::all() {
        let cnt = modality_counts.get(&modality).copied().unwrap_or(1) as f64;
        let sum = modality_ocr_corr_sum.get(&modality).copied().unwrap_or(0.0);
        mean_ocr_corruption_by_modality.insert(modality, sum / cnt);
    }

    let provenance = CorpusProvenance {
        seed,
        total_pages,
        patches_per_page,
        dim,
        repo_files_loaded,
        modality_page_counts: modality_counts,
        mean_ocr_corruption_by_modality,
        tenant_page_counts: tenant_counts,
        acl_page_counts: acl_counts,
    };

    Ok(BenchmarkCorpus { store, modality_queries, exact_keyword_queries, provenance })
}

/// Deterministic bag-of-tokens + bigram dense text embedding strictly from OCR text.
pub fn embed_ocr_text(seed: u64, text: &str, dim: usize) -> Vec<f32> {
    let tokens = tokenize_text(text);
    let mut acc = vec![0.0f32; dim];
    if tokens.is_empty() {
        return acc;
    }
    for tok in &tokens {
        let weight =
            if tok.contains('_') || tok.contains("::") || tok.contains('-') { 1.6 } else { 1.0 };
        let v = seeded_unit_vector(seed ^ 0x1111_2222, tok, dim);
        for i in 0..dim {
            acc[i] += weight * v[i];
        }
    }
    for win in tokens.windows(2) {
        let bigram = format!("{}|{}", win[0], win[1]);
        let v = seeded_unit_vector(seed ^ 0x3333_4444, &bigram, dim);
        for i in 0..dim {
            acc[i] += 0.65 * v[i];
        }
    }
    l2_normalize(&mut acc);
    acc
}

fn build_visual_page_patches(
    seed: u64,
    rng: &mut SplitMix64,
    modality: PageModality,
    domain: &str,
    concept_labels: &[String],
    patches_per_page: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    let mut patches = Vec::with_capacity(patches_per_page);
    let modality_basis = seeded_unit_vector(seed ^ 0x7777_8888, modality.as_str(), dim);
    let domain_basis = seeded_unit_vector(seed ^ 0x9999_aaaa, domain, dim);
    let inv_sqrt_d = (dim as f32).sqrt().recip();

    // Place salient visual/semantic patches in dedicated spatial slots (2 patches per concept)
    // and fill remaining grid patches with realistic background layout, margin, and text-line noise.
    for p_idx in 0..patches_per_page {
        let mut v = vec![0.0f32; dim];
        if p_idx < concept_labels.len() * 2 {
            let c_idx = p_idx / 2;
            let concept_vec = seeded_unit_vector(seed ^ 0x5555_6666, &concept_labels[c_idx], dim);
            let jitter_amp = if modality == PageModality::TextOnly { 0.48 } else { 0.28 };
            for d in 0..dim {
                let jitter = jitter_amp * inv_sqrt_d * rng.next_normal();
                v[d] = 0.82 * concept_vec[d]
                    + 0.14 * domain_basis[d]
                    + 0.08 * modality_basis[d]
                    + jitter;
            }
        } else if modality == PageModality::TextOnly && p_idx == concept_labels.len() * 2 {
            // Downsampled 448x448 visual patch of a code/config block: captures template only,
            // blurring fine-grained trailing numeric identifier digits.
            let tmpl = seeded_unit_vector(seed ^ 0x5555_6666, "visual_prose_code_block", dim);
            for d in 0..dim {
                v[d] =
                    0.65 * tmpl[d] + 0.25 * domain_basis[d] + 0.30 * inv_sqrt_d * rng.next_normal();
            }
        } else {
            // Margin, header chrome, whitespace, and unrelated distractor text-line patches.
            let bg_label = format!("bg_layout_{}_{}", modality.as_str(), p_idx % 11);
            let bg_vec = seeded_unit_vector(seed ^ 0xbbbb_cccc, &bg_label, dim);
            for d in 0..dim {
                v[d] = 0.50 * bg_vec[d]
                    + 0.15 * modality_basis[d]
                    + 0.10 * domain_basis[d]
                    + 0.45 * inv_sqrt_d * rng.next_normal();
            }
        }
        l2_normalize(&mut v);
        patches.push(v);
    }

    patches
}

fn build_query_token_embeddings(
    seed: u64,
    rng: &mut SplitMix64,
    concept_labels: &[String],
    query_text: &str,
    is_exact_identifier_query: bool,
    dim: usize,
) -> Vec<Vec<f32>> {
    let inv_sqrt_d = (dim as f32).sqrt().recip();
    let mut token_vecs = Vec::new();
    for concept in concept_labels {
        let base = seeded_unit_vector(seed ^ 0x5555_6666, concept, dim);
        let noise_scale = if is_exact_identifier_query { 0.40 } else { 0.24 };
        let mut v = vec![0.0f32; dim];
        for d in 0..dim {
            v[d] = base[d] + noise_scale * inv_sqrt_d * rng.next_normal();
        }
        l2_normalize(&mut v);
        token_vecs.push(v);
    }
    // Pad with realistic lexical context tokens to reach ~8–10 query tokens (typical short query).
    let raw_tokens = tokenize_text(query_text);
    for tok in raw_tokens.iter().take(4) {
        let mut v = seeded_unit_vector(seed ^ 0xdddd_eeee, tok, dim);
        for x in &mut v {
            *x += 0.20 * inv_sqrt_d * rng.next_normal();
        }
        l2_normalize(&mut v);
        token_vecs.push(v);
    }
    token_vecs
}

fn mangle_ocr_token(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '0' => 'O',
            '1' => 'l',
            '5' => 'S',
            '.' => ',',
            'm' => 'r',
            _ => c,
        })
        .collect()
}
