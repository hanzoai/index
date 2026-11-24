# HIP-1347: Specialised Multilingual, Code, and Domain Embedding Benchmark & Distillation (`hanzoai/index#13`)

## Overview

`crates/rag-eval/src/domain_embeddings.rs` implements a reproducible, clean-room Apache-2.0 evaluation and distillation-justification harness comparing universal vs specialised retrievers across **MTEB general retrieval**, **Multilingual queries**, **Code retrieval**, **Scientific terminology**, and **Conversational/Agent memory**:

* `answerdotai/ModernBERT-base-dual` (Apache-2.0, `768d`)
* `hanzo/laya-modernbert-compact` (Apache-2.0, `512d` compact dual encoder)
* `Qwen/Qwen3-Embedding-0.6B` / `4B` / `8B` (Apache-2.0, `1024d` to `4096d`)
* `BAAI/bge-m3` (MIT, unified dense + lexical sparse + multi-vector late interaction)
* `hanzo/laya-code-distilled-256d` & `hanzo/laya-memory-distilled-256d` (Apache-2.0 `256d` specialised students gated by `DistillationJustificationGate`)
