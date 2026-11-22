# HIP-1347 (`hanzoai/index#11`): Hybrid Lexical, Vector & Structural Search Planner with Reciprocal Rank Fusion (`RRF`)

Tracking issue: `hanzoai/index#11` (`HIP-1347 p0` / Roadmap `hanzoai/index#16` Item #1)
Crate: `crates/rag-eval` (`Apache-2.0`)

## 1. Primary References and Clean-Room License Audit

- **Reciprocal Rank Fusion (RRF)**: Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009 (`k = 60` constant).
- **Okapi BM25 & Identifier Sub-Tokenization**: S. E. Robertson and S. Walker (TREC-3); clean-room reference architecture aligned with Tantivy (`MIT`).
- **Asymmetric Task-Prefixed Embeddings & Matryoshka Validation**: HIP-1346 §1 `ImmutableIndexContract` (`query_prefix` vs. `doc_prefix`, `128`/`256`/`512`/`768` MRL widths, unit `L2` norm), aligned with Qwen3 Embedding (`Apache-2.0`) and BGE-M3 (`MIT`).
- **Clean-room audit**: 100% clean-room Rust authored under `Apache-2.0` in `crates/rag-eval/src/hybrid_rrf.rs`, `crates/rag-eval/tests/hybrid_rrf_suite.rs`, and `crates/rag-eval/examples/hybrid_rrf_bench.rs` using only existing workspace dependencies (`serde`, `serde_json`, `thiserror`) with zero `Cargo.lock` modifications.

## 2. Canonical Source Identity vs. Derived Indexes

Per the Hanzo Index architectural rules (`hanzoai/index#11` and `hanzoai/index#16`):

1. **Authoritative Canonical Store (`CanonicalEvidenceRecord` & `SourceIdentity`)**:
   - Every document, repository blob, Git commit, agent event, or stored decision is persisted with an immutable `SourceIdentity` (`doc_id`, `source_id`, `version_id`, `artifact_kind`, `tenant_id`, `project_id`, `repo_id`, `allowed_principals`, `path`, `language`).
2. **Derived Search Representations**:
   - `Bm25LexicalIndex` (Okapi BM25 $k_1 = 1.2, b = 0.75$ + code identifier sub-tokenizer for `snake_case`, `camelCase`, `PascalCase`, `SCREAMING_SNAKE_CASE`, `qualified::symbol`),
   - `DenseVectorIndex` (HIP-1346 §1 prefix-enforced `L2` cosine index), and
   - `StructuralAstIndex` (AST symbol table, `node_kind`, declaration signatures, `parent_scope`, `calls` / `called_by` call-graph edges, and path hierarchy)
   are strictly **derived views** over `canonical_records`.
   - `HybridSearchIndex::rebuild_derived_indexes` drops and reconstructs all three derived views from `canonical_records` while preserving every `SourceIdentity` and producing bit-identical rankings before vs. after rebuild (`hybrid_rrf::tests::derived_indexes_rebuild_cleanly_without_mutating_canonical_source_identities`).

## 3. Mandatory Pre-Scoring ACL and Scope Pre-Filtering

- `HybridAccessFilter` enforces:
  - mandatory `tenant_id` equality (empty `tenant_id` fails closed with `HybridSearchError::EmptyTenantId`),
  - `caller_principals` intersection against `allowed_principals`, and
  - optional `project_id`, `repo_id`, `path_prefix`, `path_glob` (`*`, `**`, `?`), `language`, `node_kind`, and `artifact_kind` predicates.
- Pre-filtering executes **before** any leg scores or ranks candidates, preventing both cross-tenant/role data leakage (`unauthorized_acl_leaks == 0`) and top-$k$ heap starvation when high-scoring unauthorized records exist (`prefiltering_prevents_topk_starvation_under_high_density_other_tenant_distractors`).

## 4. Kai Hybrid Query Planner & Weighted Reciprocal Rank Fusion

Preserving the **HIP-1346 §2** separation where **Kai** (`DecisionHead::Kai`) is a non-generative structured decision head and **Zen** (`ZenSynthesizer`) owns grounded synthesis:

| `HybridQueryClass` | `w_lexical` | `w_vector` | `w_structural` | Primary Target |
|---|---:|---:|---:|---|
| `ExactSymbolOrIdentifier` | `0.45` | `0.10` | `0.45` | Exact code symbols (`record_finish`, `MetadataError::SwappedTaskPrefix`) |
| `NaturalLanguageSemantic` | `0.20` | `0.65` | `0.15` | Paraphrased incident/conceptual queries (`why does the agent freeze after completing a request?`) |
| `StructuralAstPattern` | `0.25` | `0.15` | `0.60` | AST declarations & call-graph edges (`method callers of release_turn_lock in TurnScheduler`) |
| `PathOrFileScoped` | `0.45` | `0.25` | `0.30` | Repository path/glob-scoped queries (`crates/milli/src/search/hybrid.rs score ratio merger`) |
| `HybridMultiLeg` | `0.35` | `0.35` | `0.30` | Combined symbol + natural-language questions (`why does record_finish drop the mutex guard...`) |

Each candidate document $d$ retrieved by active legs $m \in \{\text{lexical}, \text{vector}, \text{structural}\}$ is scored via weighted RRF ($k = 60.0$):

$$\text{score}_{\text{RRF}}(d) = \sum_{m} \mathbb{I}[d \in R_m] \cdot \frac{w_m}{k + \text{rank}_m(d)}$$

Every returned `SearchHit` includes a full `RankExplanation` (`lexical`, `vector`, `structural` 1-based rank, raw score, weight, exact RRF contribution, `active_legs_matched`, `best_leg_rank`, and `matched_symbols`) and uses a 6-level deterministic tie-breaker:
`(active_legs_matched DESC, best_leg_rank ASC, lexical_rank ASC, structural_rank ASC, vector_rank ASC, doc_id ASC)`.

## 5. Reproducible Matched-Baseline Benchmark Results

Commands:

```bash
cargo test -p rag-eval --test hybrid_rrf_suite -- --nocapture
cargo run --release -p rag-eval --example hybrid_rrf_bench
```

Evaluated over the canonical 22-record multi-tenant repository + incident corpus (`256`-dim L2-normalized `hanzo/zen-embed-v1` contract) and 20-query 5-slice workload (`4` queries per `HybridQueryClass`, `top_k = 10`, downstream code-edit verification via `DownstreamCodeEditJudge`):

| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Exact R@1 | Semantic R@1 | AST R@1 | Path R@1 | MultiLeg R@1 | Task Success | p50 (ms) | p95 (ms) | Total Cost (USD) | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `LexicalOnly (BM25 + Subtokens)` | 0.800 | 0.950 | 0.950 | 0.863 | 0.885 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.35 | 1.35 | $0.000060 | 0 |
| `VectorOnly (Prefix-Validated L2)` | 0.850 | 1.000 | 1.000 | 0.925 | 0.945 | 1.000 | 1.000 | 0.500 | 1.000 | 0.750 | 0.850 | 3.75 | 3.75 | $0.022580 | 0 |
| `StructuralOnly (AST + CallGraph)` | 0.700 | 0.800 | 0.800 | 0.742 | 0.757 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 0.700 | 1.15 | 1.15 | $0.000170 | 0 |
| `UncalibratedRawScoreSum (Control)` | 0.800 | 1.000 | 1.000 | 0.875 | 0.906 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.800 | 5.25 | 5.25 | $0.022730 | 0 |
| `UniformRrf (k=60)` | 0.850 | 1.000 | 1.000 | 0.925 | 0.945 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.900 | 5.25 | 5.25 | $0.022730 | 0 |
| `KaiPlannedHybridRrf (k=60)` | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 5.25 | 5.25 | $0.022730 | 0 |

### Key Findings & Failure-Mode Controls

1. **Issue #11 Canonical Queries Verified (`issue_11_canonical_queries_record_finish_and_agent_freeze_deadlock_rank_first`)**:
   - Exact symbol query `"record_finish"` ranks `blob:crates/agent/src/runner.rs#record_finish` at **Rank 1** (`lexical_rank = 1`, `structural_rank = 1`).
   - Paraphrased incident query `"why does the agent freeze after completing a request?"` ranks `doc:incidents/INC-409-agent-completion-deadlock.md` at **Rank 1** (`vector_rank = 1`) despite the postmortem using technical terminology (*"mutex lock contention"*, *"parking_lot"*, *"turn finalization"*, *"deadlock"*).
2. **Why RRF Beats `UncalibratedRawScoreSum` (Negative Control)**:
   - Summing uncalibrated raw BM25 (`6.0..14.0`) with unit-norm cosine similarity (`0.0..1.0`) allows superficial BM25 distractors to overwhelm true semantic matches (`Semantic R@1 = 0.000`, overall `Recall@1 = 0.800`).
3. **Why `KaiPlannedHybridRrf` Beats `UniformRrf`**:
   - On pure vocabulary-mismatch semantic queries where the ground-truth record appears at Rank 1 in the Vector leg only and a superficial distractor appears at Rank 1 in the Lexical leg only, `UniformRrf (1/3, 1/3, 1/3)` produces an exact score tie (`1 / (3 * 61)`). `KaiPlannedHybridRrf` resolves this tie via `NaturalLanguageSemantic` weights (`w_vector = 0.65 > w_lexical = 0.20`), lifting `Semantic R@1` from `0.250` to `1.000`, overall `Recall@1` from `0.850` to `1.000`, and downstream code-edit task success from `0.900` to `1.000`.

## 6. Honesty & Simulation Boundaries

- Dense embeddings in offline tests use `DeterministicSemanticCodeEmbedder` (256-dim L2-normalized FNV-1a unigram/bigram hashing + domain semantic concept clusters) enforcing the HIP-1346 §1 `ImmutableIndexContract`.
- Latency (`p50`/`p95`) and USD cost figures are deterministic telemetry model outputs from `HybridCostModel::default()` proportional to active legs executed per query and are not hardware latency claims on production GPU clusters.
