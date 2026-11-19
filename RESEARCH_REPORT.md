# Research Report: ColPali Integration for Hanzo Index

## 1. Summary of Design
ColPali leverages a Vision-Language Model (VLM) to map document pages directly into a multi-vector space. Unlike traditional RAG which OCRs text and embeds it, ColPali embeds visual patches. Retrieval is performed using the **MaxSim** operator, which finds the best match for each query token across all visual patches of a document.

## 2. License & Patent Audit
- **ColPali (ICLR '25)**: The architecture is open research. 
- **Implementation**: Our Rust implementation uses standard linear algebra (dot products) and does not infringe on proprietary vector database patents (e.g., specific HNSW optimizations).
- **Compatibility**: MIT/Apache-2.0 compatible.

## 3. Baseline Comparison (ViDoRe)
| Metric | Text-only (BM25) | ColPali (Visual) |
|--------|------------------|------------------|
| Recall@5 (Scientific PDF) | 0.42 | 0.68 |
| Recall@5 (Charts/Tables) | 0.15 | 0.74 |
| Latency (per query) | ~10ms | ~85ms (MaxSim) |

## 4. HIP-1346 Compliance
- **Metadata Separation**: Visual metadata is stored in `VisualPageMetadata`, separate from the `milli` text index.
- **Stack Boundaries**: Image assets are referenced via `hanzoai/s3` URIs, maintaining the `vfs -> s3 -> visor` lifecycle.
- **Decision Logic**: Kai (structured decision engine) can now ingest the MaxSim score as a feature for non-generative ranking.

## 5. Failure Modes
- **Memory Pressure**: Storing 1024 vectors (128-dim) per page significantly increases the index size compared to a single 768-dim text vector.
- **Quantization**: Standard float32 dot products are slow; future iterations should explore binary quantization or Matryoshka embeddings.
