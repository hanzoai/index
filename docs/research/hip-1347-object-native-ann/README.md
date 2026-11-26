# HIP-1347 (`hanzoai/index#15`): ANN & Object-Storage-Native Vector Indexes (DiskANN, USearch, LanceDB as References)

- **Tracking Issue**: [`hanzoai/index#15`](https://github.com/hanzoai/index/issues/15) (`p1`, `bounty`, `retrieval`, `research`, `hip-1347`)
- **Canonical Crate**: [`crates/vector-ann-bench`](../../../crates/vector-ann-bench) (`Apache-2.0 OR MIT`, zero external crate dependencies)
- **Raw Machine-Readable Receipt**: [`results.json`](./results.json) (`schema_version: "hip-1347-ann-bench/v1"`, `--seed 1347`)

---

## 1. Executive Summary & Concrete Hanzo Architecture Recommendation

Hanzo Index MUST NOT bolt on an external, stateful, incompatible vector database (e.g., Qdrant, Milvus, Pinecone, Weaviate) that duplicates canonical records, splits tenant/ACL enforcement, or breaks S3/VFS object-storage economics. Instead, based on matched clean-room benchmarks of all three reference architectures plus two explicit negative controls, we recommend a **unified dual-tier derived vector index architecture** sharing one canonical record contract and one pre-return ACL filter:

| Deployment Tier | Recommended Derived Architecture | Reference Lineage | Storage Layout | Hot Compute-Local State | Query I/O Profile |
|---|---|---|---|---|---|
| **Local / Single-Node** | **`local_mmap_vamana`** | DiskANN (`MIT`) + USearch (`Apache-2.0`) | Compact `mmap`-compatible binary `HVAN` file (`SQ8` slab + fixed-stride `R`-neighbor Vamana graph + CRC32) on local NVMe | PQ/SQ8 routing state + OS 4KB page cache + bounded FreshDiskANN delta overlay + O(1) tombstone bitset | `0` S3 GETs; `0.58–1.38 ms` warm p50 (`8.27–9.60 ms` warm p99 measured), `~1.8–2.2 ms` projected NVMe cold p99 at 1M–100M |
| **Cloud / Multi-Node** | **`object_native_ivf_pq`** | LanceDB (`Apache-2.0`) | Immutable coarse-partitioned IVF-PQ + SQ8 segment packs (`partitions.pack`, `manifest.bin`, immutable `delta_vN.bin` & `tombstones_vN.bin`) on Hanzo S3/VFS | Coarse centroids (`C` partitions), PQ codebook, partition byte-range directory, per-partition tenant/tag summaries (`~16.7–22.4 B/vec` projected at 1M–100M when metadata is segment-packed; `393.0 B/vec` total measured at `D=64` with full resident Rust `String` metadata + ID lookup map) | **1 bounded parallel batch of `nprobe` contiguous `RangeGET`s** (`sequential_rtts = 1.0`); `0` S3 GETs on warm local cache |
| **Ultra-Selective Filter (`<= 2%`)** | **`prefilter_inverted_fallback`** | Hanzo lexical/filter posting lists | Compact `(tenant_id, tag) -> slots` inverted posting list | Resident bitmap/posting index | Direct exact scan over matching postings (`1.000` Recall@10, `0.066 ms` p99 vs `0.093–0.132` Recall@10 for blind post-filtered ANN) |

**Key Quantitative Proof Against Graph Traversal on Object Storage**:
Running the exact same Vamana graph directly over `InstrumentedObjectStore` without a hot local graph cache (`object_store_random_graph_hops`, explicit negative control) causes **`21.0x–33.0x` more sequential S3 round trips**, **`13.2x–20.4x` more S3 GET requests**, **`20.5x–29.0x` higher cold-cache p50 latency** (`341.0–528.0 ms` vs `16.6–18.3 ms`), and **`17.0x–21.3x` higher cold-cache p99 latency** (`362.2–562.5 ms` vs `21.3–28.1 ms`) than `object_native_ivf_pq` on the exact same corpus, queries, and cost model.

---

## 2. Primary References & Clean-Room License Audit

| Reference System | Upstream Repository | License | Commercial Integration Posture | Core Strengths | Why Not Adopted Verbatim Alone |
|---|---|---|---|---|---|
| **DiskANN** (Vamana + FreshDiskANN + Filtered-DiskANN) | `microsoft/DiskANN` | `MIT` | Permissive; clean-room Rust port preserves full compatibility without C++/MKL toolchain overhead | High filtered Recall@10 (`0.925–0.985` measured), low RAM via PQ-in-RAM + NVMe graph, FreshDiskANN delta + tombstones | Graph pointer-chasing is ideal for local NVMe `mmap`/`io_uring`, **catastrophic** over S3 object storage (`21–33` sequential RTTs) |
| **USearch** | `unum-cloud/usearch` | `Apache-2.0` | Permissive; compact memory-mappable single-file binary layout & SQ8/int8/f16 quantization patterns | Compact single-file `mmap` binary layout, minimal per-vector RAM, fast local cold-start | Local-disk HNSW/graph-centric; still requires object-native coarse partitioning for S3/VFS cloud tier |
| **LanceDB** (Lance IVF-PQ immutable segments) | `lancedb/lancedb` / `lancedb/lance` | `Apache-2.0` | Permissive; immutable columnar segment & coarse IVF-PQ + range-read architecture matches Hanzo S3/VFS | Write-once immutable segments on S3, `1`-RTT parallel `RangeGET` partition reads, minimal routing RAM (`~17–22 B/vec` projected) | Higher single-node cold latency than local NVMe `mmap` Vamana; best paired with local `mmap` tier for developer/edge nodes |

**Clean-Room Implementation Audit (`crates/vector-ann-bench`)**:
- Authored 100% clean-room in Rust under `Apache-2.0 OR MIT`.
- **Zero external crate dependencies** (`[dependencies]` in `crates/vector-ann-bench/Cargo.toml` is intentionally empty — pure Rust `std`, zero `Cargo.lock` additions of third-party crates).
- Compiles and executes all 18 tests completely offline via `cargo test -p vector-ann-bench --release --offline`.

---

## 3. Canonical/Derived Separation & Pre-Return Tenant/ACL Isolation Audit

Per the shared Hanzo Index retrieval rules (`hanzoai/index#15`):

1. **Authoritative Canonical Truth (`CanonicalStore`)**:
   - Every record carries immutable canonical identity `(source_id, version, tenant_id, acl_group, filter_tags, vector)`.
   - All four vector indexes (`exact_flat_baseline`, `local_mmap_vamana`, `object_native_ivf_pq`, `object_store_random_graph_hops`) and `prefilter_inverted_fallback` are strictly **derived views**.
   - **Drop & Rebuild Parity Receipt** (`results.json -> drop_rebuild_parity`):
     - `canonical_fingerprint_before == canonical_fingerprint_after == 0x4a5de6b533063490` (`canonical_unchanged = true`)
     - `vamana_binary_bit_exact = true` (byte-for-byte identical on-disk `HVAN` binary + CRC32 after full drop & rebuild)
     - `vamana_hits_bit_exact = true`, `ivf_pq_hits_bit_exact = true`, `exact_flat_hits_bit_exact = true`
2. **Fail-Closed Pre-Return Tenant/ACL Isolation**:
   - Empty `tenant_id` (`AnnError::EmptyTenantId`) and empty `allowed_acl_groups` (`AnnError::EmptyAclGroups`) fail closed before touching any index structure (verified across all 5 search entrypoints in `fail_closed_acl_rejects_empty_tenant_and_empty_groups`).
   - Adversarial exact-clone test (`zero_cross_tenant_and_acl_leaks_under_adversarial_exact_clone_vectors`) plants 15 distance-`0.0` clones belonging to `tenant-enterprise` and restricted `sec-ops` ACL groups inside the same derived indexes; all 5 engines return **0 unauthorized records** while still returning valid authorized hits (no top-$k$ starvation).
   - Across the entire benchmark run (**1,080+ measured queries** across selectivities, negative controls, downstream repo tasks, and post-mutation runs): **`acl_leaks = 0` (`0.0%`), `filter_violations = 0`, `stale_hits = 0`**.

---

## 4. Reproducible Matched-Baseline Measured Results (`is_projected: false`)

### Reproduction Commands

```bash
# 1. Verify formatting, strict Clippy lints, and 18/18 offline release tests
cargo fmt -p vector-ann-bench -- --check
cargo clippy -p vector-ann-bench --all-targets --offline -- -D warnings
cargo test -p vector-ann-bench --release --offline

# 2. Run reproducible benchmark & regenerate docs/research/hip-1347-object-native-ann/results.json
cargo run -p vector-ann-bench --release --offline --bin ann-bench -- \
  --seed 1347 \
  --vectors 4096 \
  --dim 64 \
  --queries 60 \
  --output docs/research/hip-1347-object-native-ann/results.json
```

### Corpus Provenance & Measured Build / Footprint Summary

- **Seed**: `1347` | **Dimensions**: `64` (L2 unit-normalized) | **Total Vectors**: `4,096`
- **Public Real-Repo Content**: `200` real source/doc files scanned from `hanzoai/index` (`crates/**/*.rs`, `docs/**/*`), yielding `2,048` real canonical repository chunks (`blob:<rel_path>#chunk<N>`) + `2,048` multi-cluster benchmark vectors across 32 unit-sphere clusters, 2 tenants (`tenant-hanzo`, `tenant-enterprise`), and 3 ACL groups (`eng-core`, `public`, `sec-ops`).
- **Telemetry Cost Model** (`ObjectStoreCostModel::default()`): S3 RTT = `15.0 ms`, single-stream bandwidth = `400 MB/s`, S3 GET = `\$0.0004 / 1,000`, S3 storage = `\$0.023 / GB-mo`, NVMe = `\$0.12 / GB-mo`, DRAM = `\$4.50 / GB-mo`, NVMe cold 4KB page read = `85 us`.

| Architecture | Build Time (ms) | Measured Cold RAM (B/vec) | Measured Disk / Object (B/vec) | Notes |
|---|---:|---:|---:|---|
| `exact_flat_baseline` | 8.66 | 631.8 | 256.0 | FP32 vectors (`256.0 B/vec`) + resident `DerivedRecordMeta` (`~277.7 B/vec`) + `id_to_slot` map & tombstones (`2,587,907 B` total RAM) |
| `local_mmap_vamana` | 3,118.86 | 375.9 | 164.1 | `HVAN` binary on disk (`64 B` SQ8 + `4 B` deg + `24*4 B` adj = `164 B/vec`); cold RAM includes resident `DerivedRecordMeta` + `base_id_to_node` map + tombstones (`1,539,843 B` total RAM; `1.68x` less RAM than exact flat at `D=64`) |
| `object_native_ivf_pq` | 469.23 | 393.0 | **84.4** | Cold RAM counts ALL resident allocations: `DerivedRecordMeta` + `id_to_base_idx` map + tombstones + `~17.1 B/vec` hot IVF centroids/PQ codebook/partition directory (`1,609,629 B` total RAM; `1.61x` less RAM than exact flat at `D=64`); S3 pack = `4 + 8 (PQ) + 64 (SQ8)` + manifest |
| `object_store_random_graph_hops` | 3,068.19 | 277.8 | 164.1 | Negative control: resident `DerivedRecordMeta` + SQ8 quantizer (`1,138,027 B` total RAM; read-only so no mutation ID map; `0` local graph cache; full `HVAN` graph on S3) |

> **Honest RAM Accounting Note**: All four architectures count their resident `DerivedRecordMeta` (`Vec<String>` tags + `String` IDs, `~277.7 B/vec`) and mutable ID lookup maps (`~98 B/vec`) consistently. At `D = 64` (only `256 B/vec` raw FP32), this shared Rust string-metadata table dominates resident RAM across every architecture; removing FP32 vectors from RAM saves `~239–256 B/vec` (`1.61x–1.68x` reduction at `D = 64`), while the hot IVF-PQ routing state itself (`centroids + PQ codebook + SQ8 params + partition directory`) adds only `~17.1 B/vec`. At production `D = 768` (`3,072 B/vec` FP32) with compact integer IDs / segment-co-located ACL metadata (Section 5 analytical projections), vector RAM dominates and yields `~139x–186x` RAM reduction (`16.7–22.4 B/vec` vs `3,104 B/vec`).

### Filtered Recall@10, Warm vs. Cold Latency, S3 IOPS & ACL Isolation (`N = 4,096`, `60` Queries/Slice)

| Architecture | Selectivity | Cache Regime | Recall@10 | CPU p50 (ms) | Total p50 incl. I/O (ms) | Total p99 incl. I/O (ms) | Avg S3 GETs | Seq. RTTs | Avg Fetched (KB) | Local 4KB Faults | ACL Leaks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact_flat_baseline` | 100% | warm-ram | 1.000 | 1.097 | 1.097 | 6.987 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `local_mmap_vamana` | 100% | cold-cache | **0.955** | 0.789 | 10.185 | 21.233 | 0.0 | 0.0 | 401.3 | 100.5 | 0 |
| `local_mmap_vamana` | 100% | warm-cache | **0.955** | 0.579 | 0.579 | 8.273 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `object_native_ivf_pq` | 100% | cold-cache | 0.777 | 0.175 | 16.597 | **21.278** | **12.0** | **1.0** | 39.2 | 0.0 | 0 |
| `object_native_ivf_pq` | 100% | warm-cache | 0.777 | 0.134 | 0.134 | **4.353** | **0.0** | **0.0** | 0.0 | 0.0 | 0 |
| `object_store_random_graph_hops` *(Neg. Control)* | 100% | cold-cache | 0.592 | 0.300 | 341.000 | **362.172** | **244.3** | **21.0** | 39.1 | 0.0 | 0 |
| `exact_flat_baseline` | 25% | warm-ram | 1.000 | 0.484 | 0.484 | 1.414 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `local_mmap_vamana` | 25% | cold-cache | **0.985** | 1.881 | 15.271 | 23.150 | 0.0 | 0.0 | 601.4 | 150.8 | 0 |
| `local_mmap_vamana` | 25% | warm-cache | **0.985** | 1.349 | 1.349 | 8.993 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `object_native_ivf_pq` | 25% | cold-cache | **0.938** | 0.256 | 18.256 | **28.140** | **24.0** | **1.0** | 89.2 | 0.0 | 0 |
| `object_native_ivf_pq` | 25% | warm-cache | **0.938** | 0.210 | 0.210 | **3.452** | **0.0** | **0.0** | 0.0 | 0.0 | 0 |
| `object_store_random_graph_hops` *(Neg. Control)* | 25% | cold-cache | 0.760 | 0.400 | 527.959 | **555.761** | **321.7** | **33.0** | 51.5 | 0.0 | 0 |
| `exact_flat_baseline` | 5% | warm-ram | 1.000 | 0.300 | 0.300 | 0.684 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `local_mmap_vamana` | 5% | cold-cache | **0.925** | 1.645 | 14.923 | 20.093 | 0.0 | 0.0 | 601.5 | 150.9 | 0 |
| `local_mmap_vamana` | 5% | warm-cache | **0.925** | 1.378 | 1.378 | 9.604 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `object_native_ivf_pq` | 5% | cold-cache | **0.922** | 0.214 | 18.231 | **26.394** | **24.0** | **1.0** | 100.3 | 0.0 | 0 |
| `object_native_ivf_pq` | 5% | warm-cache | **0.922** | 0.174 | 0.174 | **4.303** | **0.0** | **0.0** | 0.0 | 0.0 | 0 |
| `object_store_random_graph_hops` *(Neg. Control)* | 5% | cold-cache | 0.535 | 0.404 | 525.863 | **562.469** | **317.4** | **33.0** | 50.8 | 0.0 | 0 |

### Explicit Negative Regime: Ultra-Selective `1%` Filter (`60` Queries)

When filter selectivity drops to `1%`, blind post-filtering (fetching top-32 ANN hits then filtering) collapses catastrophically due to top-$k$ starvation, whereas pre-filter inverted-posting lookup beats both blind post-filtered ANN and filtered graph traversal on **both accuracy and latency**:

| Strategy (`1%` Filter) | Recall@10 | Avg Hits Returned | p50 (ms) | p99 (ms) | ACL Leaks | Takeaway |
|---|---:|---:|---:|---:|---:|---|
| `vamana_blind_postfilter_top32` | 0.093 | 0.9 | 0.220 | 4.395 | 0 | Severe top-$k$ starvation (fails) |
| `ivf_pq_blind_postfilter_top32` | 0.132 | 1.4 | 0.112 | 4.191 | 0 | Severe top-$k$ starvation (fails) |
| `local_mmap_vamana_filtered` | 0.730 | 10.0 | 1.353 | 10.047 | 0 | Survives via widened beam, but `152.5x` slower p99 |
| **`prefilter_inverted_fallback`** | **1.000** | **10.0** | **0.014** | **0.066** | **0** | **Recommended planner fallback below `~2%` selectivity** |

### Online Update, Tombstone Delete & Compaction Lifecycle (`+204` Inserts, `204` v2 Updates, `409` Deletes)

| Architecture | Avg Upsert (us) | Avg Delete (us) | Post-Mutation Recall@10 | Tombstones Reclaimed | Compaction Time (ms) | Bytes Before -> After | Post-Compaction Recall@10 | ACL Leaks / Stale Hits |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `local_mmap_vamana` | 1.8 *(sub-ms batch)* | 0.2 *(O(1) bitset)* | 0.962 | 613 | 2,695.4 | `672,296 -> 638,676 B` | 0.902 | `0 / 0` |
| `object_native_ivf_pq` | 10.3 | 0.7 | 0.830 | 613 | **438.0** | `394,819 -> 330,048 B` | 0.770 | `0 / 0` |

> Note: `object_native_ivf_pq` compacts `6.2x` faster than Vamana (`438.0 ms` vs `2,695.4 ms`) because k-means IVF repacking is $O(N \cdot C)$ sequential math vs $O(N \cdot L \cdot R)$ random graph pruning, while never mutating any immutable S3 object in place.

### Downstream Real-Repo Agent File-Resolution Tasks (`40` Tasks over `200` `hanzoai/index` Files)

Per issue #15 (*"Benchmarks report accuracy, latency, cost and downstream task success together"*):

| Architecture | Tasks | Recall@10 | File Resolution @1 | File Resolution @5 | Warm p99 (ms) | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|
| `exact_flat_baseline` | 40 | 1.000 | 0.950 | 1.000 | 1.208 | 0 |
| `local_mmap_vamana` | 40 | 0.990 | **0.950** | **1.000** | 19.519 | 0 |
| `object_native_ivf_pq` | 40 | 0.988 | **0.950** | **1.000** | **8.252** | 0 |

Both recommended architectures match the exact brute-force oracle on downstream file resolution (`95.0%` @1, `100.0%` @5) with `0` ACL leaks!

---

## 5. Analytical 1M, 10M, and 100M Vector Scaling Projections (`is_projected: true`)

> [!IMPORTANT]
> The tables below are **analytical scaling projections** (`is_projected: true` in `results.json`), strictly separated from the `N = 4,096` local benchmark measurements above. Every cell is derived by closed-form equations (`crates/vector-ann-bench/src/scaling.rs`) parameterized by our exact binary strides (`HVAN` disk stride = `D + 4 + 4*R + 32` B at `R=32`; IVF-PQ segment stride = `4 + D/16 + D + 20` B; LanceDB-style `~240–380` vectors/partition: `C = 4,096` at 1M, `32,768` at 10M, `262,144` at 100M), our measured negative-control hop/GET counts (`21.0` hops / `244.3` GETs scaled by `1.0x / 1.25x / 1.5x`), and public cloud pricing (`\$0.023/GB-mo` S3, `\$0.0004/1K` S3 GETs, `\$0.12/GB-mo` NVMe, `\$4.50/GB-mo` DRAM).

### Projected Scaling at `D = 768` (ModernBERT / Qwen3-MRL-768 / BGE-M3)

| Scale (`N`) | Architecture | RAM (B/vec) | Total RAM (GB) | Disk/S3 (B/vec) | Total Storage (GB) | Cold S3 GETs/q | Cold Seq. RTTs | Est. Cold p99 (ms) | Storage (\$/mo) | DRAM (\$/mo) | S3 GETs (\$/1M q) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1M** | `exact_flat_baseline` | 3,104.0 | 3.10 | 3,072.0 | 3.07 | 0.0 | 0.0 | 76.80 | \$0.37 | \$13.97 | \$0.00 |
| **1M** | **`local_mmap_vamana`** | 72.0 | **0.07** | 932.0 | 0.93 | 0.0 | 0.0 | **1.76** | \$0.11 | \$0.32 | \$0.00 |
| **1M** | **`object_native_ivf_pq`** | **22.4** | **0.02** | **840.0** | **0.84** | **16.0** | **1.0** | **34.18** | **\$0.02** | **\$0.10** | **\$6.40** |
| **1M** | `object_store_random_graph_hops` *(Neg.)* | 4.0 | 0.004 | 932.0 | 0.93 | 244.3 | 21.0 | 425.25 | \$0.02 | \$0.02 | \$97.72 |
| **10M** | `exact_flat_baseline` | 3,104.0 | 31.04 | 3,072.0 | 30.72 | 0.0 | 0.0 | 768.00 | \$3.69 | \$139.68 | \$0.00 |
| **10M** | **`local_mmap_vamana`** | 72.0 | **0.72** | 932.0 | 9.32 | 0.0 | 0.0 | **2.00** | \$1.12 | \$3.24 | \$0.00 |
| **10M** | **`object_native_ivf_pq`** | **19.0** | **0.19** | **840.0** | **8.40** | **24.0** | **1.0** | **38.25** | **\$0.19** | **\$0.85** | **\$9.60** |
| **10M** | `object_store_random_graph_hops` *(Neg.)* | 4.0 | 0.04 | 932.0 | 9.32 | 305.4 | 26.3 | 531.56 | \$0.21 | \$0.18 | \$122.15 |
| **100M** | `exact_flat_baseline` | 3,104.0 | 310.40 | 3,072.0 | 307.20 | 0.0 | 0.0 | 7,680.00 | \$36.86 | \$1,396.80 | \$0.00 |
| **100M** | **`local_mmap_vamana`** | 72.0 | **7.20** | 932.0 | 93.20 | 0.0 | 0.0 | **2.23** | \$11.18 | \$32.40 | \$0.00 |
| **100M** | **`object_native_ivf_pq`** | **16.7** | **1.67** | **840.0** | **84.00** | **32.0** | **1.0** | **43.55** | **\$1.93** | **\$7.53** | **\$12.80** |
| **100M** | `object_store_random_graph_hops` *(Neg.)* | 4.0 | 0.40 | 932.0 | 93.20 | 366.5 | 31.5 | 637.88 | \$2.14 | \$1.80 | \$146.58 |

*(`D = 1536` projections are also published in full inside [`results.json`](./results.json) -> `projected_scaling_1536d`: at 100M `D=1536`, `object_native_ivf_pq` uses `2.48 GB` hot RAM & `165.6 GB` S3 (`\$3.81/mo`) at `53.5 ms` cold p99 vs `617.6 GB` RAM (`\$2,779/mo`) for exact flat and `637.9 ms` / `\$146.58` per 1M queries for random S3 graph hops.)*

---

## 6. PR Convergence Metadata (`PR_CONVERGENCE.md`), Honesty Boundaries & Rollout Plan

### Required PR Metadata
1. **Canonical Owning Repository**: `hanzoai/index` (`crates/vector-ann-bench` + `docs/research/hip-1347-object-native-ann/`).
2. **Qualification Criteria**:
   - Zero new external crate dependencies (`cargo test -p vector-ann-bench --release --offline`).
   - `cargo fmt -p vector-ann-bench -- --check` and `cargo clippy -p vector-ann-bench --all-targets --offline -- -D warnings` pass cleanly.
   - `0.0%` cross-tenant/ACL leaks, fail-closed empty-tenant/ACL validation, bit-exact drop & rebuild parity, CRC32 corruption detection, tombstone/freshness correctness, and quantitative negative-control verification (`>= 10x` S3 RTT/GET penalty for random graph hops; pre-filter superiority at `1%`).
3. **Verification Evidence**:
   - 18/18 unit & integration tests passing offline (`4` unit + `14` integration in `vector_ann_suite.rs`).
   - Raw reproducible benchmark receipt committed at [`docs/research/hip-1347-object-native-ann/results.json`](./results.json) (`--seed 1347`, 200 real repo files, 4,096 vectors).
4. **Security Review**:
   - Pre-return tenant & ACL enforcement verified under adversarial distance-`0.0` cross-tenant/restricted-ACL clones; fail-closed validation on empty tenant or ACL group sets; checked integer arithmetic on binary offsets/strides; CRC32 integrity verification on binary `HVAN` files; immutable write-once object store semantics rejecting in-place overwrites.
5. **Rollout & Rollback Plan**:
   - **Rollout**: Research crate `crates/vector-ann-bench` is isolated and additive with zero changes to shipping `crates/milli` or `crates/search` request paths. Phase 2 production wiring will plug `HVAN` local mmap Vamana into Hanzo local storage (`crates/file-store` / `crates/milli`) and `HSEG` immutable IVF-PQ segments into Hanzo Cloud S3/VFS behind a feature flag, sharing `CanonicalStore` identity and pre-scoring ACL predicates.
   - **Rollback**: Single-commit clean revert removing `"crates/vector-ann-bench"` from root `Cargo.toml`, `crates/vector-ann-bench/`, and `docs/research/hip-1347-object-native-ann/` with zero schema or data-migration impact.

### Honesty & Simulation Boundaries
- **Measured vs. Projected**: All `N = 4,096` metrics (`is_projected: false`) are real wall-clock CPU timings + real file-backed 4KB `read_at` I/O + deterministic I/O latency modeling (`s3_rtt_ms = 15.0 ms`, `local_nvme_page_read_us = 85.0 us`). All 1M / 10M / 100M figures (`is_projected: true`) are transparent analytical projections; **no production hardware SLA is claimed until reproduced on Hanzo production NVMe and live S3/VFS infrastructure**.
- **Embeddings**: Offline benchmark embeddings are deterministic hash + cluster projections derived from real public `hanzoai/index` source files (no external model weights or proprietary Hanzo evaluation corpora are shipped or required).
