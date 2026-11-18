# HIP-1346 status update — Bounty #21 (hanzoai/index#25, OdinANN)

**Status: research delivered (draft PR), pending maintainer review.**

`hanzoai/index` `crates/rag-eval` now carries a bounded, deterministic,
stdlib-only simulation harness (`ann_insert`) benchmarking direct online
vector insertion into a file-backed on-disk ANN index under continuous query
traffic, per [Bounty #21](https://github.com/hanzoai/index/issues/25):

- Three maintenance policies over an identical seeded workload:
  (A) batch-rebuild baseline (stop-the-world, pending linear-scan tail),
  (B) naive in-place online insertion, and
  (C) OdinANN-style bounded delta layer + amortized per-query merge with
  emergency overflow merges.
- Metrics: `recall@10` (against **brute-force L2 ground truth over the exact
  present set at each query**, precomputed once per workload and shared across
  all policy runs), p50/p99/mean query latency, stall time (reported
  separately, never amortized), rebuild/emergency-merge counts, page I/O
  accounting, and insert throughput.
- The HIP-1346 §1 immutable metadata contract (`IndexEmbeddingMetadata`) gates
  every insert and query **before any mutation**; the mismatch-rejection
  isolation property is enforced by tests
  (`crates/rag-eval/tests/ann_insert_suite.rs`).

## Proposed HIP-1346 status text

> `hanzoai/index` (`crates/rag-eval::ann_insert`, draft PR for [Bounty #21]) adds
> a clean-room simulation of online vector insertion into a file-backed on-disk
> ANN index under continuous query traffic, comparing stop-the-world batch
> rebuild, naive in-place insertion, and bounded delta-merge maintenance at
> insert rates 0.25×–4× per query. Measured outputs are simulation figures on
> the authoring host (synthetic clustered unit-sphere vectors, single-layer
> greedy beam graph, simulated page-read penalty); **no Hanzo performance
> expectation is asserted until reproduced on Hanzo hardware** against the real
> on-disk ANN stack. Immutable `(model_id, model_revision, dimensions,
> normalization, task-prefix)` enforcement is test-enforced with zero-mutation
> rejection semantics.

## Honest boundaries

1. The ANN structure is a single-layer greedy beam-search graph — figures
   measure maintenance-policy behavior, not upstream OdinANN numbers.
2. Vectors are synthetic; ground truth is exact brute force, so `recall@10` is
   honestly computed, but the corpus is not Hanzo production data.
3. "On-disk" is a real append-only page file plus a **simulated, configurable**
   page-read penalty (tests run at zero penalty; the runner models an
   NVMe-like penalty). No physical NVMe hardware is exercised.
4. Single-threaded, order-invariant scheduler over the seeded workload;
   stall time is reported separately. Policy (C)'s amortized delta absorption
   runs **between query arrivals** (background) and is timed as
   `maintenance_ns`, never as query latency.
5. Policy (B)'s long-run structural degradation is deliberately under-modeled
   in this graph formulation; (B) is reported as a latency-floor reference and
   its recall is reported as observed.
