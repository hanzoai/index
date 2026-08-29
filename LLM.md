# Hanzo Index

## Overview
A search index in Rust: documents in, ranked results out. Full-text (BM25, typo
tolerant), vector, and hybrid retrieval, with faceted filtering, geo search, and chat
completions grounded in your own indexes.

One instance holds one tenant's indexes behind one master key, so a deployment is per
tenant rather than one shared keyspace.

**Upstream**: forked from [Meilisearch](https://github.com/meilisearch/meilisearch)
(Meili SAS, MIT + a commercial EE). LICENSE, LICENSE-MIT and LICENSE-EE keep their
copyright and NOTICE records the fork — that is what a fork owes, and it stays.

## Tech Stack
- **Language**: Rust (toolchain pinned by `rust-toolchain.toml`)

## Build & Run
```bash
cargo build --release -p search -p searchtool
./target/release/search --master-key="…"
```
Binaries are named after their crates: the server is `search`, the CLI companion is
`searchtool`. The container's WORKDIR is `/data`, which is where a volume belongs.

## Naming
Four different things, deliberately not merged:

| thing | what it is |
|---|---|
| **this repo** (`hanzoai/index`) | the index engine — where documents go |
| `search.hanzo.ai` | Hanzo's own instance of it, serving the docs corpus |
| cloud `clients/websearch` | queries the outside world |
| `hanzoai/crawl` | fetches pages |

## Environment
Every option is a `const` in `crates/search/src/option.rs`, all prefixed `INDEX_`
(`INDEX_MASTER_KEY`, `INDEX_DB_PATH`, `INDEX_ENV`, …). With `INDEX_ENV=production` and
no master key the process refuses to start, so a misconfigured name is a crash-loop and
never an unauthenticated index.

`mini-dashboard` is NOT a default feature — it vendors a prebuilt UI from an upstream
release. Without it `/` answers `{"status":"Hanzo Index is running"}` in every
environment.

## Upgrading a live database
The database records the version that wrote it, and a newer binary will not open an
older one. `INDEX_EXPERIMENTAL_DUMPLESS_UPGRADE=true` migrates in place on boot
(supported from 1.12.0 up). Take a dump first — `POST /dumps` — and check its document
counts against `/stats` before rolling: that dump restores into the newer binary with
`--import-dump`, and that is the rollback.

## CI
No GitHub Actions. Every push reaches the platform through the Hanzo GitHub App and the
platform builds on our own runner fabric; pipelines live in `.hanzo/workflows`.

## Key Files
- `crates/search/` -- HTTP server, routes, auth, search queue
- `crates/milli/` -- the engine: indexing, ranking, filtering, vectors
- `crates/index-scheduler/` -- async task scheduling, batching, upgrade steps
- `Dockerfile` -- container build
- `docs/index.mdx` -- the reference
## Version pin (kept here, not in customer docs)

Current workspace version **1.45.2**, 23 workspace members. Naming as shipped:
server binary `search`, operator CLI `searchtool`, auth crate `search-auth`,
core index engine `milli` (unchanged), env var prefix `INDEX_*`.

docs.hanzo.ai/docs/skills/hanzo-search no longer pins the upstream release in
its introduction — it had drifted (it said v1.37.0) and it is an engineering
fact, not a customer-facing one. Update the version above when the base moves.
