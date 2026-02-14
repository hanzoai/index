<p align="center">
  <a href="https://hanzo.ai" target="_blank">
    <img src="https://github.com/hanzoai/.github/raw/main/profile/logo-light.png" alt="Hanzo AI" width="300">
  </a>
</p>

<h1 align="center">Hanzo Search</h1>

<p align="center">
  A lightning-fast search engine built on Meilisearch, optimized for AI workloads and seamless integration with the Hanzo ecosystem.
</p>

<h4 align="center">
  <a href="https://hanzo.ai">Website</a> |
  <a href="https://docs.hanzo.ai/search">Documentation</a> |
  <a href="https://github.com/hanzoai/search">GitHub</a>
</h4>

<p align="center">
  <a href="https://github.com/hanzoai/search/actions"><img src="https://img.shields.io/github/actions/workflow/status/hanzoai/search/ci.yml?branch=main&label=CI" alt="CI Status"></a>
  <a href="https://github.com/hanzoai/search/releases"><img src="https://img.shields.io/github/v/release/hanzoai/search" alt="Latest Release"></a>
  <a href="https://github.com/hanzoai/search/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-informational" alt="License"></a>
  <a href="https://deps.rs/repo/github/hanzoai/search"><img src="https://deps.rs/repo/github/hanzoai/search/status.svg" alt="Dependency Status"></a>
</p>

---

## Features

- **Hybrid Search** -- Combine semantic and full-text search for maximum relevance across structured and unstructured data
- **Search-as-you-type** -- Sub-50ms query responses for instant, interactive search experiences
- **Typo Tolerance** -- Return relevant results even when queries contain typos and misspellings
- **Filtering and Faceted Search** -- Build rich, filterable search interfaces with custom facets in a few lines of code
- **Sorting** -- Sort results by price, date, relevance score, or any custom attribute
- **Synonym Support** -- Configure synonym mappings to broaden search coverage without duplicating data
- **Geosearch** -- Filter and sort documents by geographic coordinates and distance
- **Multi-Language** -- Search datasets in any language, with optimized tokenization for Chinese, Japanese, Hebrew, and Latin-alphabet languages
- **Security and Multi-Tenancy** -- Fine-grained API key permissions and tenant token isolation for SaaS applications
- **AI-Ready** -- First-class integration with the Model Context Protocol (MCP) and LangChain for retrieval-augmented generation (RAG) and agent workflows
- **RESTful API** -- Clean, well-documented HTTP API for indexing, searching, and managing configuration

## Quick Start

### Docker

```bash
docker run -d \
  --name hanzo-search \
  -p 7700:7700 \
  -v hanzo_search_data:/meili_data \
  hanzoai/search:latest \
  hanzo-search --master-key="YOUR_MASTER_KEY"
```

### From Source

```bash
git clone https://github.com/hanzoai/search.git
cd search
cargo build --release
./target/release/hanzo-search --master-key="YOUR_MASTER_KEY"
```

### Add Documents

```bash
curl -X POST 'http://localhost:7700/indexes/products/documents' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_MASTER_KEY' \
  --data-binary '[
    { "id": 1, "title": "Hanzo Agent SDK", "category": "AI" },
    { "id": 2, "title": "Hanzo MCP Server", "category": "Infrastructure" },
    { "id": 3, "title": "Hanzo LLM Gateway", "category": "AI" }
  ]'
```

### Search

```bash
curl 'http://localhost:7700/indexes/products/search' \
  -H 'Authorization: Bearer YOUR_MASTER_KEY' \
  --data-binary '{ "q": "agent" }'
```

## SDKs

Hanzo Search provides official SDKs for seamless integration:

| Language | Package | Repository |
|----------|---------|------------|
| Go | `github.com/hanzoai/search-go` | [hanzoai/search-go](https://github.com/hanzoai/search-go) |
| JavaScript / TypeScript | `@hanzo/search` | [hanzoai/search-js](https://github.com/hanzoai/search-js) |
| Python | `hanzo-search` | [hanzoai/search-python](https://github.com/hanzoai/search-python) |
| Rust | `hanzo-search` | [hanzoai/search-rust](https://github.com/hanzoai/search-rust) |

## Documentation

Full documentation is available at [docs.hanzo.ai](https://docs.hanzo.ai).

- [Getting Started](https://docs.hanzo.ai/search/getting-started)
- [API Reference](https://docs.hanzo.ai/search/api-reference)
- [Configuration](https://docs.hanzo.ai/search/configuration)
- [Hybrid Search Guide](https://docs.hanzo.ai/search/hybrid-search)
- [Security and API Keys](https://docs.hanzo.ai/search/security)
- [MCP Integration](https://docs.hanzo.ai/search/mcp)

## Editions and Licensing

### Community Edition

- Open source under the [MIT License](./LICENSE)
- Full search engine with hybrid, semantic, and full-text search
- Free for any use, including commercial

### Enterprise Edition

- Advanced features: sharding, S3-streaming snapshots, dedicated support
- Governed by the [Business Source License 1.1](./LICENSE-EE)
- Contact [sales@hanzo.ai](mailto:sales@hanzo.ai) for production use

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

- [Open an issue](https://github.com/hanzoai/search/issues) to report bugs
- [Start a discussion](https://github.com/hanzoai/search/discussions) for feature requests

---

<p align="center">
  Based on <a href="https://github.com/meilisearch/meilisearch">Meilisearch</a>. See upstream <a href="https://github.com/meilisearch/meilisearch/blob/main/LICENSE">LICENSE</a> for attribution.
</p>

<p align="center">
  Copyright 2025 Hanzo AI Inc. All rights reserved.
</p>
