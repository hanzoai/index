# `rag-eval` (HIP-1346)

Lightweight, pure-Rust, Apache-2.0 library implementing **HIP-1346** (*Evidence-Grounded Retrieval and Decision Evaluation*) for `hanzoai/index` while preserving the canonical **Kai** (structured decision head) vs. **Zen** (generative synthesis) architectural boundary.

## Scope & Bounty Mapping

| Module | HIP-1346 / Bounty | Primary Reference | Core Components |
|---|---|---|---|
| `metadata` | HIP-1346 §1 | Immutable Vector Index Metadata Contract | `IndexEmbeddingMetadata`, `ImmutableIndexContract`, Matryoshka (`128`/`256`/`512`/`768`) & prefix validators |
| `self_rag` | Bounty #22 (`hanzoai/index#26`, Wave 1 #8) | Self-RAG (Asai et al., ICLR '24) | `RetrieveOrAnswerGate`, `KaiRetrieveProbs`, `ZenSynthesizer`, threshold sweep & cost/accuracy simulator |
| `corrective` | Bounty #23 (`hanzoai/index#27`) | Corrective RAG (Yan et al., arXiv:2401.15884) | `CorrectiveRetrievalPolicy`, distractor/corruption/contradiction detector, secondary retrieval & query reformulation |
| `hyde` | Bounty #24 (`hanzoai/index#28`) | HyDE (Gao et al., ACL '23; `texttron/hyde`) | `HydeIndex`, `ZenHypothesisGenerator`, `KaiHydeGate`, immutable `doc_prefix`/`query_prefix` centroid & cost/recall harness |
| `diagnostics` | Bounty #25 (`hanzoai/index#29`) | RAGChecker (Ru et al., NeurIPS '24) | `RagChecker`, fine-grained `retrieval_miss`, `irrelevant_evidence`, `unsupported_claim`, `answer_omission` diagnostics |
| `factscore` | Bounty #30 (`hanzoai/index#31`) | FActScore (Min et al., EMNLP '23) | `AtomicClaimDecomposer`, `SpanEvidenceBinder`, `FactScoreEvaluator`, `compute_ece` calibration |

## Architectural Boundary

- **Kai** (`HIP-1332`) remains strictly a non-generative structured decision head returning calibrated discrete probabilities (`retrieve` vs `answer_directly`, relevance/risk grades, claim verification probabilities).
- **Zen** (`hanzoai/ml` / `hanzoai/engine`) owns all text synthesis, both parametric (`ZenParametric`) and evidence-grounded (`ZenGrounded`).
- Clean-room Apache-2.0 implementation built exclusively from pure Rust (`serde`, `serde_json`, `thiserror`) with no GPL/AGPL/BUSL dependencies.

## Testing

```bash
cargo test -p rag-eval
```
