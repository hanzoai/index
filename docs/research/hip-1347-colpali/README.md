# HIP-1347 (`hanzoai/index#23`): Visual Document Retrieval for Scientific PDFs, Figures, and Tables (ColPali)

- **Tracking Issue**: [`hanzoai/index#23`](https://github.com/hanzoai/index/issues/23) (`p2`, `bounty`, `research`, `hip-1347`)
- **Canonical Crate**: [`crates/colpali-eval`](../../../crates/colpali-eval) (`Apache-2.0 OR MIT`, `#![forbid(unsafe_code)]`, zero external crate dependencies)
- **Raw Machine-Readable Receipt**: [`results.json`](./results.json) (`schema_version: "hip-1347-colpali-bench/v1"`, `--seed 1347`)

---

## 1. Primary Paper, Code Links, Summary & License Audit

### Primary Reference
- **Paper**: Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo — *ColPali: Efficient Document Retrieval with Vision Language Models*, ICLR 2025 ([OpenReview / ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99e9e141aafc314f76b0ca3dd66898b3-Abstract-Conference.html), [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)).
- **Summary**: Instead of brittle PDF OCR + layout-chunking pipelines that lose 2D table row/column bindings, chart legends/axes, and two-column equations, ColPali feeds rendered PDF page images directly into a Vision-Language Model (VLM) projected to a `128`-d multi-vector patch representation (`~1,030` patch vectors/page) and scores queries via ColBERT-style late interaction:
  $$\text{MaxSim}(Q, P) = \sum_{i \in Q} \max_{j \in P} \langle q_i, p_j \rangle$$
  This dramatically improves retrieval over visually rich documents (ViDoRe), at the cost of a massive multi-vector storage footprint (`~527 KB/page` in FP32) and weaker exact-string discrimination compared to inverted BM25 indexes.

### License & Dependency Audit (Commercial Integration Posture)

| Upstream Artifact | Repository / Checkpoint | License | Hanzo Production Posture | Notes |
|---|---|---|---|---|
| **ColPali Engine** | `illuin-tech/colpali` | `MIT` | Permissive reference | Python PyTorch training/inference library; not imported into Rust core |
| **ViDoRe Benchmark Harness** | `vidore/vidore-benchmark` | `MIT` | Permissive harness | Underlying benchmark sub-datasets carry mixed licenses (DocVQA, InfoVQA, TAT-DQA, ArXivQA) — audit per dataset before redistribution |
| **ColPali v1.2 / v1.3 Weights** | `vidore/colpali-v1.3` (PaliGemma-3B) | **Gemma Terms of Use** | **Restricted (Not Apache-2.0/MIT)** | Upstream PaliGemma base carries Gemma ToU use restrictions; do **not** vendor weights into `hanzoai/index` |
| **ColQwen2 / ColSmolVLM Weights** | `vidore/colqwen2-v1.0`, `vidore/colSmol-500M` | **`Apache-2.0`** | **Preferred Enso model backend** | Permissive commercial weights using the identical 128-d MaxSim patch contract; route via **Enso**, not inside `index` |
| **Hanzo Clean-Room Implementation** | [`crates/colpali-eval`](../../../crates/colpali-eval) | **`Apache-2.0 OR MIT`** | **Approved clean-room Rust** | 100% clean-room pure Rust `std`, `#![forbid(unsafe_code)]`, **zero external crate dependencies**, offline build & test |

> [!IMPORTANT]
> **Published Paper Numbers Disclaimer**: ViDoRe numbers reported by Faysse et al. (ICLR 2025) are the authors' own published results and are **not** Hanzo performance claims until reproduced with live Apache-2.0 VLM weights (e.g., ColQwen2 / ColSmolVLM via Enso) on Hanzo GPU hardware. All measured tables below (`is_projected: false`) are exact deterministic wall-clock & byte-accounting measurements from our clean-room Rust harness on the authoring host; all 100k–1M scaling figures (`is_projected: true`) are transparent closed-form analytical projections.

---

## 2. Reproducible Baseline, Corpus Provenance & Hardware Execution

### Exact Offline Reproduction Commands

```bash
# 1. Verify formatting, strict Clippy lints (-D warnings), and 17/17 offline release tests
cargo fmt -p colpali-eval -- --check
cargo clippy -p colpali-eval --all-targets --offline -- -D warnings
cargo test -p colpali-eval --release --offline

# 2. Run reproducible benchmark & regenerate docs/research/hip-1347-colpali/results.json
cargo run -p colpali-eval --release --offline --bin colpali-bench -- \
  --seed 1347 \
  --pages 512 \
  --patches 64 \
  --dim 128 \
  --queries 24 \
  --output docs/research/hip-1347-colpali/results.json
```

### Corpus Provenance & Transparent OCR Degradation Profile

- **Deterministic Seed**: `1347` | **Pages**: `512` | **Patches/Page**: `64` (`8×8` spatial grid) | **Patch Dim**: `128` (exact ColPali head dimension)
- **Public Real-Repo Grounding**: Scans `10` real public documentation & spec files directly from `hanzoai/index` (`docs/index.mdx`, `BENCHMARKS.md`, `README.md`, `documentation/experimental-features.md`, `documentation/prototypes.md`, `documentation/release.md`, `documentation/versioning-policy.md`, `SECURITY.md`, `TESTING.md`, `Cargo.toml`) to ground technical vocabulary, architecture terms, and configuration specs alongside ViDoRe-style scientific PDF page layouts.
- **Multi-Tenant / Multi-ACL Partitioning**: `2` tenants (`tenant-hanzo` ~75%, `tenant-external-lab` ~25%) and `3` ACL groups (`research-public`, `eng-core`, `sec-confidential`). Default benchmark queries run scoped to `tenant-hanzo` with `["research-public", "eng-core"]`; pages in `tenant-external-lab` and `sec-confidential` must never leak. Equivalent authorized pages sharing identical generated prompt and concept descriptors are all judged relevant so metrics measure retrieval quality rather than arbitrary target tie-breaking.
- **Documented Mean OCR Corruption Rate by Modality** (`results.json -> corpus_provenance.mean_ocr_corruption_rate_by_modality`):

| Page Modality | Pages | Mean OCR Corruption Rate | Physical Failure Mechanism Modeled |
|---|---:|---:|---|
| `text_only` | 204 (39.8%) | **2.5%** (`0.0253`) | Clean linear PDF text; exact identifiers & flags preserved verbatim |
| `dense_table` | 102 (19.9%) | **62.7%** (`0.6269`) | Multi-column row/col headers detached, numeric cells mangled (`0`->`O`, `1`->`l`, `.`->`,`) |
| `scientific_figure` | 102 (19.9%) | **78.9%** (`0.7886`) | Only sparse caption survives OCR; curves, rotated axes, legends, and Pareto knees lost |
| `multi_column_formula` | 104 (20.3%) | **66.4%** (`0.6639`) | Left/right columns interleave; Greek letters, subscripts, and operators dropped |

---

## 3. Falsifiable Hypothesis, Metrics & Explicit Negative Regimes

### Hypothesis
1. **Visual Advantage**: On OCR-hostile scientific pages (`DenseTable`, `ScientificFigure`, `MultiColumnFormula`), ColPali late-interaction patch MaxSim outperforms standard OCR+BM25+single-vector hybrid retrieval by `>= +0.25` NDCG@5.
2. **Quantization Efficiency**: Applying orthogonal Fast Walsh-Hadamard rotation (`FWHT-128`) + 1-bit binary patch quantization (`XOR + POPCNT` Stage-1 shortlist in hot RAM) + genuine disk-backed SQ8 Stage-2 rerank cuts hot patch-vector RAM by **`32.0×`** (`16 B/patch` vs `512 B/patch`) and query latency by **`>= 5×`** while retaining **`>= 90%`** of FP32 NDCG@5.
3. **Negative Regime 1 (Plain-Text Exact Keywords/Identifiers)**: On exact alphanumeric identifiers, Rust paths, and CLI flags (`hanzo_*_cfg_*`, `crates::milli::*`, `--hanzo-flag-*`), `ocr_bm25_hybrid_baseline` beats pure visual ColPali by `>= +0.20` NDCG@5 at `> 30×` smaller RAM and `> 100×` lower latency because `448×448` VLM patches blur fine-grained character-level literals.
4. **Negative Regime 2 (Uncompressed Multi-Vector Storage Explosion)**: At `100k–1M` pages with standard ColPali (`1,030 × 128`), raw FP32 requires `> 527 GB` RAM (`$2,373.55/mo` DRAM) at 1M pages, proving raw FP32 is economically unviable without 1-bit binary quantization, `2×2` spatial pooling, and selective visual routing.

---

## 4. Bounded Clean-Room Implementation (`crates/colpali-eval`)

All four retrieval strategies are strictly **derived views** built solely from [`CanonicalPageStore`](../../../crates/colpali-eval/src/canonical.rs) (authoritative identity `(doc_id, page_no, version, tenant_id, acl_group, page_modality)`):

1. **`ocr_bm25_hybrid_baseline`** ([`OcrBm25HybridIndex`](../../../crates/colpali-eval/src/ocr_hybrid.rs)): Deterministic BM25 inverted index (`k1 = 1.5`, `b = 0.75`) fused with single-vector OCR dense cosine similarity, `HOCR` binary layout + IEEE CRC32.
2. **`colpali_fp32_maxsim`** ([`ColPaliFp32Index`](../../../crates/colpali-eval/src/colpali_index.rs)): Exact FP32 multi-vector late-interaction MaxSim over contiguous patch slab, `HCFP` binary layout + IEEE CRC32.
3. **`colpali_binary_quantized_two_stage`** ([`ColPaliBinaryTwoStageIndex`](../../../crates/colpali-eval/src/colpali_index.rs)): Orthogonal `FWHT-128` + 1-bit packed `[u64; 2]` patch signatures in hot RAM (`16 B/patch` -> exact `32.0×` patch RAM compression) scored via `XOR + count_ones()` Stage-1 shortlist, followed by **genuine disk-backed SQ8 (`i8 + f32` scale) Stage-2 MaxSim rerank** (`Sq8DiskSlab` streamed to disk during build and fetched on demand for shortlisted authorized pages via lock-free `FileExt::read_at` with zero resident SQ8 heap slab); `HCB2` binary layout + IEEE CRC32.
4. **`hybrid_text_plus_colpali_router`** ([`HybridVisualRouterIndex`](../../../crates/colpali-eval/src/router.rs)): Ingest-time selective indexer using ONLY observable `PageLayoutFeatures::visual_complexity_score() >= 0.28` (zero ground-truth label leakage — indexes `0%` of `TextOnly` pages visually) paired with query-time lexical exact-identifier fast-pathing (`HybridRouterTextOnly`) and calibrated text+visual score fusion (`HybridRouterVisualFused`).

---

## 5. Correctness, Concurrency, Security & Regression Evidence

- **Test Suite**: **17/17 offline release tests passing** (`3` unit tests in `math.rs` + `14` integration tests in [`tests/colpali_suite.rs`](../../../crates/colpali-eval/tests/colpali_suite.rs)) with zero external crates and zero Clippy warnings (`-D warnings`).
- **Fail-Closed Pre-Return Tenant & ACL Isolation**:
  - Empty/whitespace `tenant_id` (`ColPaliError::EmptyTenantId`) and empty or blank `allowed_acl_groups` (`ColPaliError::EmptyAclGroups`) fail closed across all 4 strategies before touching index data (`fail_closed_acl_rejects_empty_tenant_and_empty_groups`).
  - Adversarial exact-clone audit (`adversarial_acl_audit` in `results.json`): `24` bit-for-bit clones of target pages planted under `tenant-adversarial-other` and `sec-top-secret`; all 4 engines return **`0` ACL leaks, `0` cross-tenant leaks**, and full `top_k` authorized hits (no top-$k$ starvation).
  - Across all `480` measured benchmark queries (`4 strategies × (96 multimodal + 24 exact-keyword)`) plus `48` adversarial queries: **`acl_leaks = 0` (`0.0%`), `cross_tenant_leaks = 0` (`0.0%`)**.
- **Drop & Rebuild Bit-Exact Parity (`results.json -> drop_rebuild_parity`)**:
  - `canonical_fingerprint_before == canonical_fingerprint_after == 0x3d88632cf26b10cb` (`canonical_unchanged: true`).
  - All 4 derived indexes achieve `checksum_exact: true` (payload IEEE CRC32 excluding the 4-byte trailer, verified non-vacuous against empty, populated, and version-mutated indexes) and `hits_bit_exact: true` (`f32::to_bits` identical) after full drop & rebuild (`all_passed: true`).
- **Integrity, Lifecycle & Concurrency**:
  - Single-bit corruption in serialized `HOCR`, `HCFP`, and `HCB2` binary payloads is rejected via IEEE CRC32 (`serialization_roundtrip_and_crc32_corruption_detection`).
  - Version updates (`v99`) and canonical deletes propagate cleanly on rebuild with zero stale hits (`canonical_version_update_and_delete_reflect_cleanly_on_rebuild`).
  - 8-thread concurrent read stress test over `Arc`-shared derived indexes confirms deterministic bit-exact results and `0` ACL leaks under concurrent execution (`concurrent_multithreaded_reads_are_deterministic_and_acl_safe`).
  - Crate-wide `#![forbid(unsafe_code)]` and checked integer arithmetic on all binary header offsets/strides.

---

## 6. Before/After Measured Results, Negative Regimes & HIP-1347 Integration Recommendation

### A. Overall Multimodal Measured Results (`N = 512` pages, `64` patches/page, `D = 128`, `96` queries, `is_projected: false`)

| Strategy | Build (ms) | NDCG@5 | Recall@5 | MRR | p50 (ms) | p95 (ms) | p99 (ms) | RAM (B/page) | Disk (B/page) | Visual Pages | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ocr_bm25_hybrid_baseline` | 4.78 | 0.375 | 0.500 | 0.337 | **0.089** | **0.156** | **0.363** | **963.3** | **794.4** | 0 / 512 (0%) | 0 |
| `colpali_fp32_maxsim` | 13.95 | **1.000** | **1.000** | **1.000** | 24.126 | 26.921 | 28.099 | 32,926.1 | 32,859.1 | 512 / 512 (100%) | 0 |
| `colpali_binary_quantized_two_stage` | 96.29 | **1.000** | **1.000** | **1.000** | **2.565** | **3.097** | **3.276** | **1,182.3** | 9,563.2 | 512 / 512 (100%) | 0 |
| **`hybrid_text_plus_colpali_router`** | 61.12 | **1.000** | **1.000** | **1.000** | 2.976 *(mean 2.389)* | 3.531 | 5.235 | 1,676.2 | **6,549.0** | **308 / 512 (60.2%)** | **0** |

**Headline Measured Tradeoff Ratios (`is_projected: false`)**:
- `colpali_binary_quantized_two_stage` vs `colpali_fp32_maxsim`: **`32.00×` exact patch-vector RAM compression** (`27.85×` total measured resident RAM/page incl. metadata and `Sq8DiskSlab` handle; SQ8 Stage-2 slab lives strictly on disk and is fetched on demand for shortlisted pages), **`9.41×` p50 latency speedup** (`24.126 ms -> 2.565 ms`), **`8.58×` p99 latency speedup** (`28.099 ms -> 3.276 ms`), **`100.0%` NDCG@5 / Recall@5 retention** (`1.000` / `1.000`).
- `hybrid_text_plus_colpali_router` vs `colpali_fp32_maxsim`: **`19.64×` RAM reduction**, **`5.02×` disk reduction** (`31.5%` smaller disk than standalone Binary+SQ8 because `204` plain-text pages require zero visual patch storage), **`10.28×` mean latency speedup** (`24.554 ms -> 2.389 ms`).

### B. Per-Modality Breakdown & Explicit Negative Regime 1 (`24` Queries per Slice, `is_projected: false`)

| Strategy | `TextOnly` NDCG@5 (p50 ms) | `DenseTable` NDCG@5 (p50 ms) | `ScientificFigure` NDCG@5 (p50 ms) | `MultiColumnFormula` NDCG@5 (p50 ms) | **Neg. Regime 1: Exact Keyword NDCG@5 / Rec@5 (p50 ms)** |
|---|---|---|---|---|---|
| `ocr_bm25_hybrid_baseline` | **1.000** (0.087) | 0.183 (0.093) | 0.149 (0.093) | 0.169 (0.087) | **1.000 / 1.000 (0.067 ms)** |
| `colpali_fp32_maxsim` | 1.000 (23.787) | **1.000** (24.272) | **1.000** (26.090) | **1.000** (23.356) | 0.200 / 0.375 (20.480 ms) |
| `colpali_binary_quantized_two_stage` | 1.000 (2.497) | **1.000** (2.492) | **1.000** (2.880) | **1.000** (2.452) | 0.225 / 0.417 (2.292 ms) |
| **`hybrid_text_plus_colpali_router`** | **1.000 (0.134)** | **1.000** (2.997) | **1.000** (3.408) | **1.000** (2.903) | **1.000 / 1.000 (0.086 ms)** |

**Negative Regime 1 Proof**: On plain-text exact identifier/config/code queries, pure visual ColPali (`FP32` and `Binary Two-Stage`) collapses to `0.200–0.225` NDCG@5 at `2.29–20.48 ms` p50 and `34.2×` larger RAM (FP32), whereas `ocr_bm25_hybrid_baseline` achieves `1.000` NDCG@5 at `0.067 ms` p50. `hybrid_text_plus_colpali_router` detects lexical exact-identifier cues and fast-paths directly to BM25 (`0.086 ms` p50, `1.000` NDCG@5), preserving full accuracy across **every** regime.

### C. Explicit Negative Regime 2: Uncompressed Multi-Vector Storage Explosion at `100k–1M` Pages (`is_projected: true`)

> [!IMPORTANT]
> The table below is a **closed-form analytical projection** (`is_projected: true` in [`results.json`](./results.json)) using standard ColPali geometry (`1,030` patches/page × `128`-d, or `2×2` spatially pooled `256` patches/page × `128`-d), `30%` visual-heavy enterprise page mix for the selective router, and public cloud pricing (`$4.50 / GB-mo` DRAM, `$0.023 / GB-mo` S3).

| Scale (Pages) | Architecture | Patches/Pg | Visual % | RAM (B/pg) | Disk (B/pg) | Total RAM (GB) | Total Disk/S3 (GB) | RAM Ratio vs FP32 | DRAM (\$/mo) | S3 (\$/mo) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **100,000** | `ocr_bm25_hybrid_baseline` | 0 | 0% | 1,358.0 | 1,358.0 | 0.14 | 0.14 | 388.4× | \$0.61 | \$0.003 |
| **100,000** | `colpali_fp32_uncompressed` *(Neg.)* | 1,030 | 100% | 527,456.0 | 527,456.0 | **52.75** | **52.75** | 1.0× | **\$237.36** | \$1.21 |
| **100,000** | `colpali_bf16_fp16` | 1,030 | 100% | 263,776.0 | 263,776.0 | 26.38 | 26.38 | 2.0× | \$118.70 | \$0.61 |
| **100,000** | `colpali_sq8_int8` | 1,030 | 100% | 136,056.0 | 136,056.0 | 13.61 | 13.61 | 3.9× | \$61.23 | \$0.31 |
| **100,000** | `colpali_binary_1bit_ram_sq8_disk` | 1,030 | 100% | 16,576.0 | 152,536.0 | **1.66** | 15.25 | **31.8×** | **\$7.46** | \$0.35 |
| **100,000** | `hybrid_router_30pct_binary_full` | 1,030 | 30% | 6,330.8 | 47,118.8 | **0.63** | 4.71 | **83.3×** | **\$2.85** | \$0.11 |
| **100,000** | **`hybrid_router_30pct_binary_2x2_pooled`** | 256 | 30% | 2,615.6 | 12,753.2 | **0.26** | **1.28** | **201.7×** | **\$1.18** | **\$0.03** |
| **1,000,000** | `ocr_bm25_hybrid_baseline` | 0 | 0% | 1,358.0 | 1,358.0 | 1.36 | 1.36 | 388.4× | \$6.11 | \$0.03 |
| **1,000,000** | `colpali_fp32_uncompressed` *(Neg.)* | 1,030 | 100% | 527,456.0 | 527,456.0 | **527.46** | **527.46** | 1.0× | **\$2,373.55** | \$12.13 |
| **1,000,000** | `colpali_bf16_fp16` | 1,030 | 100% | 263,776.0 | 263,776.0 | 263.78 | 263.78 | 2.0× | \$1,186.99 | \$6.07 |
| **1,000,000** | `colpali_sq8_int8` | 1,030 | 100% | 136,056.0 | 136,056.0 | 136.06 | 136.06 | 3.9× | \$612.25 | \$3.13 |
| **1,000,000** | `colpali_binary_1bit_ram_sq8_disk` | 1,030 | 100% | 16,576.0 | 152,536.0 | **16.58** | 152.54 | **31.8×** | **\$74.59** | \$3.51 |
| **1,000,000** | `hybrid_router_30pct_binary_full` | 1,030 | 30% | 6,330.8 | 47,118.8 | **6.33** | 47.12 | **83.3×** | **\$28.49** | \$1.08 |
| **1,000,000** | **`hybrid_router_30pct_binary_2x2_pooled`** | 256 | 30% | 2,615.6 | 12,753.2 | **2.62** | **12.75** | **201.7×** | **\$11.77** | **\$0.29** |

**Negative Regime 2 Proof**: Indexing 1M pages with uncompressed FP32 ColPali demands **`527.46 GB` of DRAM (`$2,373.55/mo`)** — **`388.4×`** more memory than OCR+BM25 (`1.36 GB`). Combining **1-bit binary quantization (`31.8×`)**, **selective visual routing (`30%` visual pages -> `83.3×`)**, and **`2×2` spatial patch pooling (`256` patches -> `201.7×`)** brings 1M-page hot DRAM down to **`2.62 GB` (`$11.77/mo`)** and cold Hanzo S3 SQ8 rerank storage to **`12.75 GB` (`$0.29/mo`)**.

### D. Concrete HIP-1347 Production Integration Recommendation (Canonical Hanzo Architecture)

Do **not** replace text/BM25 retrieval with blanket all-page FP32 ColPali, and do **not** introduce a parallel visual search service. Integrate selective visual indexing directly into Hanzo's fixed canonical architecture:

1. **Canonical Storage (`vfs -> Hanzo S3 -> Visor`)**: Store canonical PDF page blobs, layout metadata, OCR text, and page identity `(doc_id, page_no, version, tenant_id, acl_group)` in Hanzo VFS/S3. All visual patch indexes remain strictly derived, droppable, and bit-exact rebuildable.
2. **Ingest Gate & Model Routing (**Enso**)**: At PDF ingest, run a fast CPU layout inspector (`PageLayoutFeatures::visual_complexity_score() >= 0.28`). Plain-text pages skip visual encoding completely (`0` VLM GPU calls, `0` patch bytes). Visual-heavy pages (tables, figures, formulas, scanned slides) are routed through **Enso** to an **`Apache-2.0` VLM encoder** (`ColQwen2` / `ColSmolVLM`, avoiding PaliGemma's Gemma ToU restriction).
3. **Derived Index Tiering (`crates/milli` + `crates/file-store`)**: Persist `HCB2` derived slabs with orthogonal FWHT-128 1-bit binary patch signatures in hot RAM/NVMe (`16 B/patch`, optional `2×2` spatial pooling) and SQ8 Stage-2 rerank slabs as immutable cold segments on Hanzo S3 (`1` bounded parallel `RangeGET` batch for the top-`M` Stage-1 authorized candidates).
4. **Query Routing & Security (`crates/search` + `crates/search-auth` + **Kai** / **Zen**)**: Enforce fail-closed pre-scoring tenant/ACL predicates before Stage-1 POPCNT scoring; fast-path exact identifier/code queries on BM25+hybrid; fuse visual Stage-2 SQ8 scores for analytical/visual queries before grounding **Zen** generation.
5. **Lifecycle & Billing (**Cloud**)**: Meter VLM page-encoding calls via Enso/Cloud only on pages crossing the visual complexity gate (~70% ingest GPU cost savings on text-heavy enterprise corpora).

---

## 7. Required PR Convergence Metadata (`PR_CONVERGENCE.md`)

1. **Canonical Owning Repository**: `hanzoai/index` (`crates/colpali-eval` + `docs/research/hip-1347-colpali/`).
2. **Qualification Criteria**:
   - Zero external crate dependencies (`cargo test -p colpali-eval --release --offline`), `#![forbid(unsafe_code)]`, `Apache-2.0 OR MIT`.
   - `cargo fmt -p colpali-eval -- --check` and `cargo clippy -p colpali-eval --all-targets --offline -- -D warnings` pass cleanly.
   - `0.0%` ACL/cross-tenant leaks, fail-closed empty tenant/ACL rejection, bit-exact Drop/Rebuild parity, CRC32 corruption detection, multi-threaded read determinism, and quantitative proof of Negative Regimes 1 & 2.
3. **Verification Evidence**:
   - `17/17` offline release tests passing (`3` unit + `14` integration in `colpali_suite.rs`).
   - Reproducible raw JSON receipt committed at [`docs/research/hip-1347-colpali/results.json`](./results.json) (`--seed 1347`).
4. **Security Review**:
   - Fail-closed `AccessScope::validate()` on all 4 search entrypoints; pre-scoring tenant/ACL filtering verified against `24` distance-`0.0` adversarial exact clones (`0` leaks); `#![forbid(unsafe_code)]`; checked offset/slab integer arithmetic; IEEE CRC32 integrity verification on all serialized binary formats (`HOCR`, `HCFP`, `HCB2`).
5. **Rollout & Rollback Plan**:
   - **Rollout**: Additive research & derived-index crate `crates/colpali-eval` with zero edits to shipping `crates/milli` or `crates/search` request paths. Phase 2 implementation bounty will wire `HCB2` binary+SQ8 derived segments and selective layout routing behind an opt-in index feature flag + live Apache-2.0 ColQwen2 validation on Hanzo GPU hardware via Enso.
   - **Rollback**: Single-commit clean revert removing `"crates/colpali-eval"` from root `Cargo.toml`, `crates/colpali-eval/`, and `docs/research/hip-1347-colpali/` with zero schema or data-migration impact.
