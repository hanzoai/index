# HIP-1346 / HIP-1347: Tiered Cross-Encoder Reranking and Retrieval Distillation (`hanzoai/index#14`)

**License**: `Apache-2.0`  
**Crate**: `crates/rag-eval` (`rag_eval::tiered_rerank`)  
**Issue**: `hanzoai/index#14` (*Tiered cross-encoder reranking and retrieval distillation*)

---

## 1. Primary References and Commercial License Audit

| Reference | Citation / Artifact | Upstream License | Integration Disposition |
|---|---|---|---|
| **Qwen3-Reranker** | Yanzhao Zhang et al., *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models*, 2025 (`https://arxiv.org/abs/2506.05176`, `Qwen/Qwen3-Reranker-0.6B` / `4B` / `8B`) | `Apache-2.0` | Architectural reference for lightweight-vs-heavy cross-encoder cascading; zero model weights or external code copied |
| **Rerankers Evaluation Library** | Benjamin Clavié, *rerankers: A Lightweight Python Library to Unify Ranking Methods*, 2024 (`https://github.com/AnswerDotAI/rerankers`) | `MIT` | Study reference for unified multi-stage reranker evaluation contracts; independent pure-Rust implementation |
| **Cross-Encoder Distillation** | Sebastian Hofstätter et al., *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation*, 2020 (`https://arxiv.org/abs/2010.02666`) | Paper (mathematical formulation) | Clean-room Rust implementation of listwise KL + pairwise margin + pointwise calibrated distillation |
| **Crate Dependencies** | `serde` (`1.0.228`), `serde_json` (`1.0.145`), `thiserror` (`2.0.17`) | `MIT OR Apache-2.0` | Existing workspace dependencies in `crates/rag-eval/Cargo.toml`; zero new crates added |

All code in `crates/rag-eval/src/tiered_rerank.rs` and `crates/rag-eval/tests/tiered_rerank_suite.rs` is authored 100% from scratch in pure Rust under `Apache-2.0`.

---

## 2. Architecture and Program Invariants

### 2.1 Immutable Source Identity and Derived Index Views

Every candidate passage carries an immutable `PassageIdentity`:
- `source_id`, `passage_id`, `version_id` (content digest, Git commit SHA, or revision counter), and `source_kind` (`Document`, `RepositoryBlob`, `GitCommit`, `AgentEvent`, `StoredDecision`).
- `rebuild_stage1_candidates` demonstrates that Stage-1 broad retrieval scores are strictly **derived representations** over canonical `CanonicalPassageRecord` entries and can be rebuilt across arbitrary queries without mutating `PassageIdentity` or `version_id`.

### 2.2 Mandatory Pre-Stage Tenant and Role ACL Isolation

- `pre_stage_access_filter` and `enforce_tenant_acl` validate `AccessContext` (`tenant_id` + `principal_roles`) **before Stage 1** candidate truncation/normalization, **before Stage 2** fast cross-encoder scoring, **before Stage 3** heavy cross-encoder teacher scoring, and **before `DistillationTrainer`** ingests any `TeacherJudgment`.
- Tenant isolation is fail-closed: empty or control-character `tenant_id` values, malformed `PassageIdentity` fields, cross-tenant mismatches, and missing principal roles are rejected immediately and logged in `AccessFilteredAudit` before any hit can enter top-$k$ pools or student training weights.

### 2.3 Three-Stage Cascaded Pipeline and Kai vs. Zen Split

1. **Stage 1 (`top_k_broad`)**: Admits authorized, non-corrupted tenant candidates, sorts deterministically by Stage-1 hybrid score (`(stage1_raw_score, passage_id)`), and computes min-max normalized `stage1_norm_score`.
2. **Stage 2 (`FastCrossEncoder` + Kai `EscalationGate`)**:
   - `CalibratedFastCrossEncoder` scores `(query, passage)` lexical stem recall, length-normalized token precision, stem bi-gram overlap, and exact technical identifier matches blended with Stage-1 prior.
   - `EscalationGate` operates strictly as a non-generative structured decision head under `DecisionHead::Kai`, computing:
     - `top1_score` against `min_top1_score` (`0.58`),
     - `top1_top2_margin` ($s_{(1)} - s_{(2)}$) against `min_margin` (`0.15`),
     - normalized Shannon entropy $H_{\text{norm}} = -\frac{1}{\ln m}\sum_{i=1}^{m} p_i \ln p_i \in [0, 1]$ over temperature-scaled ($\tau = 0.15$) top-$m$ ($m = 4$) softmax probabilities against `max_normalized_entropy` (`0.76`),
     - calibrated discrete probabilities `(p_escalate, p_resolve_stage2)` summing to `1.0`.
   - Easy high-margin queries terminate immediately at Stage 2 (`EscalationAction::ResolveAtStage2`) with zero Stage-3 compute.
3. **Stage 3 (`HeavyCrossEncoder`)**:
   - Invoked strictly on escalated ambiguous/hard queries, bounded by both `max_stage3_candidates` (`5`) and `max_stage3_rerank_tokens` (`1200`).
   - `SpanAlignedHeavyCrossEncoder` evaluates ordered 8-token proximity, multi-clause conjunction coverage, numeric/version literal verification, and negation polarity consistency.
4. **Grounded Synthesis (`ZenSynthesizer`)**:
   - `DeterministicZenGrounded` synthesizes answers from top-ranked grounded evidence under `SynthesisTarget::ZenGrounded`. Every compared policy uses the exact same `ZenSynthesizer`, downstream factual + polarity task judge, candidate budget (`top_k_broad = 8`, `final_top_k = 5`), cost model (`TieredCostModel`), and corpus.

### 2.4 Teacher-to-Student Knowledge Distillation (`DistillationTrainer`)

`DistillationTrainer` records Stage-3 `TeacherJudgment` slates on a **12-query training set completely disjoint from the 24-query evaluation suite** and trains a compact **83-parameter** residual cross-interaction MLP (`DistilledDomainReranker`, `10` generalizable structural features $\to$ `6` `tanh` hidden units + linear skip connection, `0.35x` per-token compute multiplier) via exact analytical backpropagation over:

$$\mathcal{L}(\theta) = w_{\text{KL}} \tau^2 D_{\text{KL}}\!\left(P_T^{(\tau)} \,\|\, P_S^{(\tau)}\right) + w_{\text{margin}} \mathcal{L}_{\text{pair}}(z_S, s_T) + w_{\text{BCE}} \mathcal{L}_{\text{BCE}}(\sigma(z_S), s_T) + \frac{\lambda}{2}\|\theta\|_2^2$$

---

## 3. Matched 5-Policy Pareto Benchmark and Raw Measurements

### 3.1 Benchmark Setup & Honest Telemetry Scope

- **Evaluation Corpus**: `24` multi-tenant queries (`14` easy identifier/config queries, `5` ambiguous numeric-version & negation-polarity queries, `5` hard two-clause multi-hop queries) across `tenant_acme` (`index_ops`) and `tenant_globex` (`ml_engineer`).
- **Adversarial Negative Controls**: Every query injects `2` high-score (`0.99` cross-tenant, `0.98` unauthorized `root_auditor` role) adversarial probes containing exact gold answer text (`48` total probes per policy run) to verify pre-stage fail-closed ACL enforcement.
- **Disjoint Distillation Split**: `DistilledDomainReranker` is distilled on `12` separate teacher-labeled queries (`tr01..tr12`) with zero query overlap against `q01..q24`.
- **Telemetry Disclosure**: Per-stage latency, compute units, and dollar costs are computed deterministically from exact tokenized `(query + passage)` pair lengths via `TieredCostModel` (`Stage 1`: `1.80 ms` / `$0.00004`; `Fast Stage 2`: `1.20 ms/100 tok`, `1.0x` compute, `$0.00015/1k tok`; `Distilled Student`: `0.45 ms/100 tok`, `0.35x` compute, `$0.00005/1k tok`; `Heavy Stage 3`: `11.50 ms/100 tok`, `8.0x` compute, `$0.00120/1k tok`) so all comparisons are 100% deterministic and reproducible across hosts.

### 3.2 Measured 5-Policy Pareto Table (`24` Queries, `top_k_broad = 8`, `final_top_k = 5`)

| Policy | NDCG@5 | Recall@5 | MRR | Answer Accuracy | p50 Latency | p95 Latency | Mean Latency | Rerank Tokens (Stage 3) | Compute Units (vs. Heavy) | Stage-3 Escalation | Cost / Correct Passage | Cross-Tenant Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Stage1Only` | `0.208` | `0.208` | `0.308` | `0.208` | `1.80 ms` | `1.80 ms` | `1.80 ms` | `0` (`0`) | `0.0` | `0.0%` | `$0.000192` | `0` (`48` blocked) |
| `AlwaysFastStage2` | `0.891` | `0.958` | `0.875` | `0.792` | `4.00 ms` | `4.48 ms` | `4.06 ms` | `4,517` (`0`) | `4,517.0` (`-87.50%`) | `0.0%` | `$0.000071` | `0` (`48` blocked) |
| `AlwaysHeavyStage3` | `1.000` | `1.000` | `1.000` | `1.000` | `22.85 ms` | `27.45 ms` | `23.44 ms` | `4,517` (`4,517`) | `36,136.0` (`baseline`) | `100.0%` | `$0.000266` | `0` (`48` blocked) |
| `TieredCascade` | `1.000` | `1.000` | `1.000` | `1.000` | `4.08 ms` | `19.87 ms` | `9.96 ms` | `5,748` (`1,231`) | `14,365.0` (**`-60.25%`**) | `41.7%` (`10/24`) | `$0.000130` | `0` (`48` blocked) |
| `DistilledStudentCascade` | `1.000` | `1.000` | `1.000` | `1.000` | `2.62 ms` | `15.36 ms` | `3.73 ms` | `4,744` (`227`) | `3,397.0` (**`-90.60%`**) | `8.3%` (`2/24`) | **`$0.000061`** | `0` (`48` blocked) |

### 3.3 Teacher-to-Student Distillation Lift (Evaluated Standalone on `24` Unseen Queries)

| Student State (`AlwaysFastStage2` Slot) | Distillation Loss (`140` Epochs) | Unseen NDCG@5 | Unseen MRR | Unseen Recall@5 | Unseen Answer Accuracy |
|---|---:|---:|---:|---:|---:|
| `DistilledDomainReranker::untrained()` | `0.6710` (epoch `0`) | `0.650` | `0.594` | `0.875` | `0.333` |
| `DistilledDomainReranker` (post-distillation) | `0.1008` (monotonic) | **`1.000`** (`+0.350`) | **`1.000`** (`+0.406`) | **`1.000`** (`+0.125`) | **`1.000`** (`+0.667`) |

### 3.4 Key Findings

1. **Cost per Correct Passage vs. Raw Stage Cost**: Although `Stage1Only` spends zero reranking tokens, its low `Recall@5` (`0.208`) inflates cost per correctly retrieved passage to `$0.000192`—more expensive per useful evidence passage than both `TieredCascade` (`$0.000130`) and `DistilledStudentCascade` (`$0.000061`).
2. **Selective Kai Escalation**: `TieredCascade` resolves `14/14` easy queries at Stage 2 (`p50 = 4.08 ms` vs. `22.85 ms`, an `82.1%` median latency reduction) and escalates `10/10` ambiguous/hard queries to Stage 3, cutting total reranking compute by **`60.25%`** and Stage-3 tokens by **`72.7%`** (`1,231` vs. `4,517`) with **`0.0%` accuracy loss** (`1.000` vs. `1.000`).
3. **Domain Student Distillation**: Distilling Stage-3 judgments on `12` disjoint queries into the 83-parameter `DistilledDomainReranker` lowers Stage-3 escalation on unseen queries from `41.7%` down to **`8.3%` (`2/24`)**, achieving **`90.60%` compute savings** vs. `AlwaysHeavyStage3` and the global Pareto-optimal **`$0.000061` per correctly retrieved passage**.

---

## 4. Reproducing

```bash
cargo test -p rag-eval --test tiered_rerank_suite -- --nocapture
```
