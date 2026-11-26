# `vector-ann-bench` (`HIP-1347` / `hanzoai/index#15`)

Zero-dependency, pure-Rust `std` clean-room benchmark harness evaluating ANN and object-storage-native vector index architectures (DiskANN, USearch, and LanceDB as architectural references) for Hanzo Index.

## Architectures Evaluated

1. **`exact_flat_baseline`** — Exact brute-force L2 scan (`100%` Recall@10 ground-truth oracle) + `PrefilterInvertedFallback` for ultra-selective (`<= 1%`) metadata filters.
2. **`local_mmap_vamana`** — DiskANN / USearch-style compact `mmap`-compatible binary Vamana graph + SQ8 vector slab (`HVAN` v1 + CRC32), 4KB paged cold/warm I/O, filtered beam traversal, FreshDiskANN-style tombstones + fresh delta overlay, and compaction.
3. **`object_native_ivf_pq`** — LanceDB / Hanzo S3/VFS-style immutable coarse-partitioned IVF-PQ + SQ8 segments backed by an instrumented `ObjectStore` with hot compute-local routing state issuing **1 bounded parallel batch of contiguous `RangeGET`s** per cold query.
4. **`object_store_random_graph_hops`** — Explicit negative control executing Vamana graph hops directly over object storage without a hot local graph cache, quantifying the `10x-40x` RTT and S3 GET amplification penalty.

## Commands

```bash
# Compile & run full test suite completely offline (zero external dependencies)
cargo test -p vector-ann-bench --release --offline

# Run reproducible benchmark and write raw JSON receipt
cargo run -p vector-ann-bench --release --bin ann-bench -- \
  --seed 1347 \
  --vectors 4096 \
  --dim 64 \
  --output docs/research/hip-1347-object-native-ann/results.json
```

See [HIP-1347 Research Dossier](../../docs/research/hip-1347-object-native-ann/README.md) for the complete architecture recommendation, license audit, measured tables, negative regimes, and 1M / 10M / 100M scaling projections.
