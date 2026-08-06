<p align="center"><img src=".github/hero.svg" alt="search" width="880"></p>

<h1 align="center">Hanzo Index</h1>

<h4 align="center">
  <a href="https://hanzo.ai">Website</a> |
  <a href="https://docs.hanzo.ai">Documentation</a>
</h4>

<p align="center">
  <a href="https://deps.rs/repo/github/hanzoai/index"><img src="https://deps.rs/repo/github/hanzoai/index/status.svg" alt="Dependency status"></a>
  <a href="./LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-informational" alt="License"></a>
</p>

<p align="center">⚡ A lightning-fast search engine that fits effortlessly into your apps, websites, and workflow 🔍</p>

Hanzo Index helps you shape a delightful search experience in a snap, offering features that work out of the box to speed up your workflow.


> **License:** Hanzo Index is distributed under the MIT License (see [`LICENSE-MIT`](./LICENSE-MIT)). Upstream's Business Source License 1.1 Enterprise Edition code is not included in this repository. See [License](#-license) below.

## ✨ Features
- **Hybrid search:** Combine the best of both semantic & full-text search to get the most relevant results
- **Search-as-you-type:** Find & display results in less than 50 milliseconds to provide an intuitive experience
- **Typo tolerance:** get relevant matches even when queries contain typos and misspellings
- **Filtering and faceted search:** enhance your users' search experience with custom filters and build a faceted search interface in a few lines of code
- **Sorting:** sort results based on price, date, or pretty much anything else your users need
- **Synonym support:** configure synonyms to include more relevant content in your search results
- **Geosearch:** filter and sort documents based on geographic data
- **Extensive language support:** search datasets in any language, with optimized support for Chinese, Japanese, Hebrew, and languages using the Latin alphabet
- **Security management:** control which users can access what data with API keys that allow fine-grained permissions handling
- **Multi-Tenancy:** personalize search results for any number of application tenants
- **Highly Customizable:** customize Hanzo Index to your specific needs or use our out-of-the-box and hassle-free presets
- **RESTful API:** integrate Hanzo Index in your technical stack with our plugins and SDKs
- **Conversational search:** let users ask questions in natural language and get AI-generated answers grounded in your search results
- **Personalization:** tailor search results to individual users based on their preferences and behavior
- **Search rules:** define custom rules to dynamically adjust search behavior based on context
- **Document relations:** link documents across indexes to enrich search results with related data
- **Replication & sharding:** scale horizontally by distributing your data across multiple nodes
- **AI-ready:** works out of the box with LangChain and the Model Context Protocol (MCP)
- **Easy to install, deploy, and maintain**

## 📖 Documentation

You can consult the Hanzo Index documentation at [docs.hanzo.ai](https://docs.hanzo.ai).

## 🚀 Getting started

For basic instructions on how to set up Hanzo Index, add documents to an index, and search for documents, take a look at our [documentation](https://docs.hanzo.ai) guide.

## 🧰 SDKs & integration tools

Install one of our SDKs in your project for seamless integration between Hanzo Index and your favorite language or framework.

Take a look at the complete integration list in the [documentation](https://docs.hanzo.ai).

## ⚙️ Advanced usage

Experienced users will want to keep the [API Reference](https://docs.hanzo.ai) close at hand.

We also offer a wide range of dedicated guides to all Hanzo Index features, such as filtering, sorting, geosearch, API keys, and tenant tokens.

Finally, for more in-depth information, refer to our articles explaining fundamental concepts such as documents and indexes.

## 🧾 License

Hanzo Index is distributed under the [MIT License](./LICENSE-MIT).

This distribution uses **only** the MIT-licensed core search engine: fast and relevant full-text, semantic, or hybrid search, free to use for anyone, including commercial usage.

Upstream's Business Source License 1.1 (BUSL-1.1) Enterprise Edition code — the sharding, networked-search and S3-streaming snapshot features — is not included in this repository, is not built, and is not distributed. Nothing here is governed by BUSL-1.1.

### 📦 Upstream attribution

Hanzo Index is a fork of [Meilisearch](https://github.com/meilisearch/meilisearch), a search engine created by Meili SAS, available under the MIT License. The upstream copyright and license notices are retained in [`LICENSE`](./LICENSE) and [`LICENSE-MIT`](./LICENSE-MIT).

### 📦 External crates

This project vendors the following MIT-licensed external crates with code modifications to use the HTTP client located in `crates/http-clients`:

- `external-crates/async-openai` and `external-crates/async-openai-macros` from <https://github.com/64bit/async-openai>
- `external-crates/reqwest-eventsource` from <https://github.com/jpopesculian/reqwest-eventsource>

## 📊 Telemetry

Hanzo Index collects **anonymized** usage data to help improve the product. You can deactivate this whenever you want; refer to the telemetry configuration in the [documentation](https://docs.hanzo.ai).

## 👩‍💻 Contributing

Hanzo Index is, and will always be, open-source. If you want to contribute to the project, please look at [our contribution guidelines](CONTRIBUTING.md).

## 📦 Versioning

Hanzo Index releases and their associated binaries are available on the project's [releases page](https://github.com/hanzoai/index/releases).

The binaries are versioned following [SemVer conventions](https://semver.org/). To know more, read our [versioning policy](./documentation/versioning-policy.md).

Differently from the binaries, crates in this repository are not currently available on [crates.io](https://crates.io/) and do not follow [SemVer conventions](https://semver.org).
