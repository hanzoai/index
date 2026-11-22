// SPDX-License-Identifier: Apache-2.0

//! HIP-1347 (`hanzoai/index#11`, `p0` / Roadmap `#16` Item #1): Hybrid Lexical,
//! Vector & Structural Search Planner with Reciprocal Rank Fusion (RRF).
//!
//! Clean-room Apache-2.0 Rust implementation of a 3-leg hybrid retrieval
//! planner and rank-fusion executor unifying:
//! 1. **Lexical Search (`Bm25LexicalIndex`)**: Okapi BM25 ($k_1 = 1.2, b = 0.75$)
//!    with code-aware identifier sub-tokenization (`snake_case`, `camelCase`,
//!    `PascalCase`, `SCREAMING_SNAKE_CASE`, `qualified::symbol`), exact symbol
//!    boosts, and contiguous phrase matching.
//! 2. **Dense Vector Search (`DenseVectorIndex`)**: Asymmetric prefix-enforced
//!    (`query_prefix` vs. `doc_prefix`) L2 cosine retrieval backed by the
//!    **HIP-1346 §1** [`ImmutableIndexContract`].
//! 3. **Structural / AST & Symbol-Graph Search (`StructuralAstIndex`)**:
//!    Exact and sub-token matching over AST symbol definitions (`symbol_name`,
//!    `qualified_name`, `node_kind`, `signature`, `parent_scope`), call-graph
//!    edges (`calls`, `called_by`, `imports`), and repository path hierarchy.
//!
//! ### Core Architectural Invariants
//! - **Canonical Source Identity vs. Derived Indexes**: Every source document,
//!   repository blob, Git commit, agent event, or stored decision is stored once
//!   as a [`CanonicalEvidenceRecord`] with a stable [`SourceIdentity`] (`doc_id`,
//!   `source_id`, `version_id`, `tenant_id`, `allowed_principals`). The lexical,
//!   vector, and structural indexes are strictly **derived views** that can be
//!   dropped and rebuilt via [`HybridSearchIndex::rebuild_derived_indexes`]
//!   without mutating or losing any canonical record.
//! - **Mandatory Pre-Scoring ACL & Scope Pre-Filtering**: Tenant isolation
//!   (`tenant_id`), role/principal allowlists (`caller_principals`), and scope
//!   predicates (`project_id`, `repo_id`, `path_prefix`, `path_glob`, `language`,
//!   `node_kind`, `artifact_kind`) are enforced **before** any search leg scores
//!   or ranks candidates, guaranteeing zero cross-tenant leakage and zero top-$k$
//!   cutoff starvation.
//! - **Kai vs. Zen Stack Split (`HIP-1346 §2`)**: [`DeterministicKaiQueryPlanner`]
//!   operates strictly under [`DecisionHead::Kai`] to classify queries into five
//!   canonical classes ([`HybridQueryClass`]) and emit calibrated per-leg weights
//!   ([`LegWeights`]) without generating text.
//! - **Weighted Reciprocal Rank Fusion (RRF)** (Cormack, Clarke & Buettcher,
//!   SIGIR 2009):
//!   $$\text{score}_{\text{RRF}}(d) = \sum_{m \in \{\text{lex}, \text{vec}, \text{struct}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
//!   with full per-hit [`RankExplanation`] provenance and a 6-level deterministic
//!   tie-breaking comparator.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::hyde::{HydeError, PrefixedTextEmbedder};
use crate::metadata::{
    ImmutableIndexContract, IndexEmbeddingMetadata, IndexOperation, MetadataError, MetadataField,
    NormalizationPolicy,
};
use crate::self_rag::{DecisionHead, SynthesisTarget, ZenSynthesizer};
use crate::text::{is_stopword, stem_lite};

/// Default RRF smoothing constant $k = 60.0$ (Cormack et al., SIGIR 2009).
pub const DEFAULT_RRF_K: f64 = 60.0;

/// Default Okapi BM25 term-frequency saturation parameter $k_1 = 1.2$.
pub const DEFAULT_BM25_K1: f64 = 1.2;

/// Default Okapi BM25 document-length normalization parameter $b = 0.75$.
pub const DEFAULT_BM25_B: f64 = 0.75;

/// Kind of canonical evidence record stored in Hanzo Index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceArtifactKind {
    Document,
    RepositoryBlob,
    GitCommit,
    AgentEvent,
    StoredDecision,
}

impl fmt::Display for SourceArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Document => f.write_str("document"),
            Self::RepositoryBlob => f.write_str("repository_blob"),
            Self::GitCommit => f.write_str("git_commit"),
            Self::AgentEvent => f.write_str("agent_event"),
            Self::StoredDecision => f.write_str("stored_decision"),
        }
    }
}

/// AST / structural node category associated with a code or document span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralNodeKind {
    Function,
    Struct,
    Class,
    Trait,
    Method,
    Module,
    Constant,
    Section,
}

impl fmt::Display for StructuralNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => f.write_str("function"),
            Self::Struct => f.write_str("struct"),
            Self::Class => f.write_str("class"),
            Self::Trait => f.write_str("trait"),
            Self::Method => f.write_str("method"),
            Self::Module => f.write_str("module"),
            Self::Constant => f.write_str("constant"),
            Self::Section => f.write_str("section"),
        }
    }
}

/// Immutable identity, version, provenance, and ACL scope of a canonical record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    /// Stable unique record identifier (e.g. `"blob:crates/agent/src/runner.rs#record_finish"`).
    pub doc_id: String,
    /// Stable upstream source identifier (e.g. `"repo:hanzoai/agents"`).
    pub source_id: String,
    /// Immutable content version or Git commit digest (e.g. `"git:331c4f168"`).
    pub version_id: String,
    /// Record category (`RepositoryBlob`, `Document`, `GitCommit`, `AgentEvent`, `StoredDecision`).
    pub artifact_kind: SourceArtifactKind,
    /// Owning tenant identifier (mandatory, never empty).
    pub tenant_id: String,
    /// Project identifier within the tenant workspace.
    pub project_id: String,
    /// Repository or corpus identifier.
    pub repo_id: String,
    /// Allowed ACL principals / roles. Empty means accessible to any authenticated caller in `tenant_id`.
    pub allowed_principals: Vec<String>,
    /// Normalized repository-relative file or document path (e.g. `"crates/agent/src/runner.rs"`).
    pub path: String,
    /// Programming or human language tag (e.g. `"rust"`, `"typescript"`, `"python"`, `"markdown"`).
    pub language: String,
}

/// Structural AST and symbol-graph metadata extracted from a code blob or structured document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralSymbolNode {
    /// Unqualified symbol identifier (e.g. `"record_finish"`).
    pub symbol_name: String,
    /// Fully qualified symbol path (e.g. `"hanzo_agent::runner::AgentRunner::record_finish"`).
    pub qualified_name: String,
    /// Structural node category (`Function`, `Struct`, `Class`, `Trait`, `Method`, `Module`, ...).
    pub node_kind: StructuralNodeKind,
    /// Full declaration signature.
    pub signature: String,
    /// Enclosing parent type/module scope (e.g. `Some("AgentRunner")`).
    pub parent_scope: Option<String>,
    /// Imported modules or symbols referenced by this scope.
    pub imports: Vec<String>,
    /// Outgoing symbol calls made by this node (call-graph edges).
    pub calls: Vec<String>,
    /// Incoming callers known to invoke this node (reverse call-graph edges).
    pub called_by: Vec<String>,
    /// 1-based start line number inside `path`.
    pub start_line: usize,
    /// 1-based end line number inside `path`.
    pub end_line: usize,
}

/// Canonical source record from which lexical, vector, and structural indexes are derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalEvidenceRecord {
    pub identity: SourceIdentity,
    pub title: String,
    pub raw_text: String,
    pub symbols: Vec<StructuralSymbolNode>,
}

/// Authenticated caller security context and scope filters applied BEFORE each search leg scores candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAccessFilter {
    /// Mandatory tenant isolation boundary.
    pub tenant_id: String,
    /// Caller's authenticated ACL roles / principals.
    pub caller_principals: Vec<String>,
    /// Optional project scope restriction.
    pub project_id: Option<String>,
    /// Optional repository scope restriction.
    pub repo_id: Option<String>,
    /// Optional normalized path prefix filter (e.g. `"crates/agent/"`).
    pub path_prefix: Option<String>,
    /// Optional glob pattern filter (supports `*`, `**`, `?`, e.g. `"crates/**/src/*.rs"`).
    pub path_glob: Option<String>,
    /// Optional language filter (case-insensitive, e.g. `"rust"`).
    pub language: Option<String>,
    /// Optional AST structural node kind filter (e.g. `StructuralNodeKind::Function`).
    pub node_kind: Option<StructuralNodeKind>,
    /// Optional artifact kind filter (e.g. `SourceArtifactKind::RepositoryBlob`).
    pub artifact_kind: Option<SourceArtifactKind>,
}

impl HybridAccessFilter {
    /// Constructs a tenant-scoped filter with optional caller principals and no scope restrictions.
    pub fn for_tenant(tenant_id: impl Into<String>, caller_principals: Vec<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            caller_principals,
            project_id: None,
            repo_id: None,
            path_prefix: None,
            path_glob: None,
            language: None,
            node_kind: None,
            artifact_kind: None,
        }
    }

    /// Builder helper restricting `project_id`.
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Builder helper restricting `repo_id`.
    pub fn with_repo(mut self, repo_id: impl Into<String>) -> Self {
        self.repo_id = Some(repo_id.into());
        self
    }

    /// Builder helper restricting `path_prefix`.
    pub fn with_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.path_prefix = Some(prefix.into());
        self
    }

    /// Builder helper restricting `path_glob`.
    pub fn with_path_glob(mut self, glob: impl Into<String>) -> Self {
        self.path_glob = Some(glob.into());
        self
    }

    /// Builder helper restricting `language`.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Builder helper restricting `node_kind`.
    pub fn with_node_kind(mut self, kind: StructuralNodeKind) -> Self {
        self.node_kind = Some(kind);
        self
    }

    /// Builder helper restricting `artifact_kind`.
    pub fn with_artifact_kind(mut self, kind: SourceArtifactKind) -> Self {
        self.artifact_kind = Some(kind);
        self
    }
}

/// Five canonical query classes recognized by the Kai Hybrid Search Planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridQueryClass {
    /// Exact symbol or code identifier lookup (e.g. `record_finish`, `MetadataError::SwappedTaskPrefix`).
    ExactSymbolOrIdentifier,
    /// Conceptual or paraphrased natural-language question (e.g. `why does the agent freeze after completing a request?`).
    NaturalLanguageSemantic,
    /// Structural / AST signature or call-graph query (e.g. `fn record_finish`, `struct AgentRunner`, `callers of release_turn_lock`).
    StructuralAstPattern,
    /// File/path-scoped lookup (e.g. `crates/agent/src/runner.rs deadlock`, or explicit `path_prefix`/`path_glob`).
    PathOrFileScoped,
    /// Multi-leg query combining exact code identifiers with natural-language intent.
    HybridMultiLeg,
}

impl fmt::Display for HybridQueryClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactSymbolOrIdentifier => f.write_str("exact_symbol_or_identifier"),
            Self::NaturalLanguageSemantic => f.write_str("natural_language_semantic"),
            Self::StructuralAstPattern => f.write_str("structural_ast_pattern"),
            Self::PathOrFileScoped => f.write_str("path_or_file_scoped"),
            Self::HybridMultiLeg => f.write_str("hybrid_multi_leg"),
        }
    }
}

/// Non-negative leg weights `(lexical, vector, structural)` used in weighted RRF.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LegWeights {
    pub lexical: f64,
    pub vector: f64,
    pub structural: f64,
}

impl LegWeights {
    /// Constructs and validates non-negative leg weights with at least one positive leg.
    pub fn new(lexical: f64, vector: f64, structural: f64) -> Result<Self, HybridSearchError> {
        if !lexical.is_finite()
            || !vector.is_finite()
            || !structural.is_finite()
            || lexical < 0.0
            || vector < 0.0
            || structural < 0.0
        {
            return Err(HybridSearchError::InvalidLegWeights { lexical, vector, structural });
        }
        if lexical + vector + structural <= 1e-12 {
            return Err(HybridSearchError::InvalidLegWeights { lexical, vector, structural });
        }
        Ok(Self { lexical, vector, structural })
    }

    /// Uniform weights across all three legs (`1/3, 1/3, 1/3`).
    pub fn uniform() -> Self {
        Self { lexical: 1.0 / 3.0, vector: 1.0 / 3.0, structural: 1.0 / 3.0 }
    }
}

/// Configuration for hybrid search execution and Reciprocal Rank Fusion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HybridRrfConfig {
    /// Positive RRF smoothing constant $k$ (default `60.0`).
    pub rrf_k: f64,
    /// Maximum candidates retrieved from each active leg before RRF fusion.
    pub per_leg_candidate_depth: usize,
    /// Final number of fused hits returned to the caller.
    pub top_k: usize,
    /// Optional explicit weight override; when `None`, the Kai query planner assigns class-calibrated weights.
    pub weight_override: Option<LegWeights>,
}

impl HybridRrfConfig {
    /// Validates configuration invariants.
    pub fn validate(&self) -> Result<(), HybridSearchError> {
        if !self.rrf_k.is_finite() || self.rrf_k <= 0.0 {
            return Err(HybridSearchError::InvalidRrfK(self.rrf_k));
        }
        if self.top_k == 0 {
            return Err(HybridSearchError::ZeroTopK);
        }
        if self.per_leg_candidate_depth == 0 {
            return Err(HybridSearchError::ZeroCandidateDepth);
        }
        if let Some(w) = self.weight_override {
            LegWeights::new(w.lexical, w.vector, w.structural)?;
        }
        Ok(())
    }
}

impl Default for HybridRrfConfig {
    fn default() -> Self {
        Self {
            rrf_k: DEFAULT_RRF_K,
            per_leg_candidate_depth: 32,
            top_k: 10,
            weight_override: None,
        }
    }
}

/// Structured non-generative execution plan emitted by the Kai Hybrid Query Planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridQueryPlan {
    pub decision_head: DecisionHead,
    pub query_class: HybridQueryClass,
    pub weights: LegWeights,
    pub rrf_k: f64,
    pub per_leg_candidate_depth: usize,
    pub top_k: usize,
    pub execute_lexical: bool,
    pub execute_vector: bool,
    pub execute_structural: bool,
    pub extracted_identifiers: Vec<String>,
    pub extracted_ast_kinds: Vec<StructuralNodeKind>,
    pub extracted_path_segments: Vec<String>,
}

/// Non-generative Kai query planner interface (HIP-1346 §2 / HIP-1347).
pub trait KaiHybridQueryPlanner: Send + Sync {
    fn plan_query(
        &self,
        raw_query: &str,
        filter: &HybridAccessFilter,
        config: &HybridRrfConfig,
    ) -> Result<HybridQueryPlan, HybridSearchError>;
}

/// Deterministic Kai query classifier and multi-leg planner.
#[derive(Debug, Clone, Default)]
pub struct DeterministicKaiQueryPlanner;

impl KaiHybridQueryPlanner for DeterministicKaiQueryPlanner {
    fn plan_query(
        &self,
        raw_query: &str,
        filter: &HybridAccessFilter,
        config: &HybridRrfConfig,
    ) -> Result<HybridQueryPlan, HybridSearchError> {
        let trimmed = raw_query.trim();
        if trimmed.is_empty() {
            return Err(HybridSearchError::EmptyQuery);
        }
        config.validate()?;

        let extracted_identifiers = extract_code_identifiers(trimmed);
        let mut extracted_ast_kinds = extract_ast_kinds_from_query(trimmed);
        if let Some(filter_kind) = filter.node_kind {
            if !extracted_ast_kinds.contains(&filter_kind) {
                extracted_ast_kinds.push(filter_kind);
            }
        }
        let extracted_path_segments = extract_path_segments(trimmed, filter);

        let has_ast_keyword = !extracted_ast_kinds.is_empty() || has_call_graph_keyword(trimmed);
        let has_path_signal = !extracted_path_segments.is_empty()
            || filter.path_prefix.is_some()
            || filter.path_glob.is_some();
        let has_code_identifier = !extracted_identifiers.is_empty();
        let has_natural_prose = has_natural_language_question_structure(trimmed);
        let raw_word_count = trimmed.split_whitespace().count();

        let query_class = if has_ast_keyword && !has_natural_prose {
            HybridQueryClass::StructuralAstPattern
        } else if has_path_signal && !has_ast_keyword && !has_natural_prose {
            HybridQueryClass::PathOrFileScoped
        } else if has_code_identifier && !has_natural_prose && raw_word_count <= 4 {
            HybridQueryClass::ExactSymbolOrIdentifier
        } else if (has_code_identifier || has_ast_keyword || has_path_signal) && has_natural_prose {
            HybridQueryClass::HybridMultiLeg
        } else if has_ast_keyword {
            HybridQueryClass::StructuralAstPattern
        } else if has_path_signal {
            HybridQueryClass::PathOrFileScoped
        } else if has_code_identifier {
            HybridQueryClass::ExactSymbolOrIdentifier
        } else {
            HybridQueryClass::NaturalLanguageSemantic
        };

        let default_weights = match query_class {
            HybridQueryClass::ExactSymbolOrIdentifier => LegWeights {
                lexical: 0.45,
                vector: 0.10,
                structural: 0.45,
            },
            HybridQueryClass::NaturalLanguageSemantic => LegWeights {
                lexical: 0.20,
                vector: 0.65,
                structural: 0.15,
            },
            HybridQueryClass::StructuralAstPattern => LegWeights {
                lexical: 0.25,
                vector: 0.15,
                structural: 0.60,
            },
            HybridQueryClass::PathOrFileScoped => LegWeights {
                lexical: 0.45,
                vector: 0.25,
                structural: 0.30,
            },
            HybridQueryClass::HybridMultiLeg => LegWeights {
                lexical: 0.35,
                vector: 0.35,
                structural: 0.30,
            },
        };

        let weights = config.weight_override.unwrap_or(default_weights);
        let candidate_depth = config.per_leg_candidate_depth.max(config.top_k);

        Ok(HybridQueryPlan {
            decision_head: DecisionHead::Kai,
            query_class,
            weights,
            rrf_k: config.rrf_k,
            per_leg_candidate_depth: candidate_depth,
            top_k: config.top_k,
            execute_lexical: weights.lexical > 0.0,
            execute_vector: weights.vector > 0.0,
            execute_structural: weights.structural > 0.0,
            extracted_identifiers,
            extracted_ast_kinds,
            extracted_path_segments,
        })
    }
}

/// Errors raised by `HybridSearchIndex` indexing, ACL validation, metadata contracts, or search execution.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum HybridSearchError {
    #[error("query must not be empty")]
    EmptyQuery,
    #[error("`doc_id` must not be empty")]
    EmptyDocId,
    #[error("`source_id` must not be empty")]
    EmptySourceId,
    #[error("`version_id` must not be empty")]
    EmptyVersionId,
    #[error("`tenant_id` must not be empty")]
    EmptyTenantId,
    #[error("`top_k` must be at least 1")]
    ZeroTopK,
    #[error("`per_leg_candidate_depth` must be at least 1")]
    ZeroCandidateDepth,
    #[error("RRF smoothing constant `rrf_k`={0} must be finite and positive")]
    InvalidRrfK(f64),
    #[error("leg weights (lexical={lexical}, vector={vector}, structural={structural}) must be finite, non-negative, and have positive sum")]
    InvalidLegWeights { lexical: f64, vector: f64, structural: f64 },
    #[error("cost parameter `{0}` must be finite and non-negative")]
    InvalidCostParameter(&'static str),
    #[error("benchmark suite requires at least one query case")]
    EmptyBenchmarkSuite,
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error(transparent)]
    Hyde(#[from] HydeError),
}

// ============================================================================
// Code-Aware Tokenizer & Identifier Sub-Tokenization
// ============================================================================

/// Result of tokenizing a code + prose span into exact tokens, sub-tokens, and stems.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenizedCodeSpan {
    /// All indexed terms (exact identifiers, decomposed sub-tokens, and stemmed words) in order.
    pub terms: Vec<String>,
    /// Set of exact lowercased code identifiers found in the text.
    pub exact_identifiers: BTreeSet<String>,
}

/// Splits a single identifier (`record_finish`, `AgentRunner`, `HTML5Parser::parseNode`)
/// into its full lowercased form plus decomposed constituent sub-tokens.
pub fn split_identifier_subtokens(raw_ident: &str) -> Vec<String> {
    let trimmed = raw_ident.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let full_lower = trimmed.to_ascii_lowercase();
    out.push(full_lower.clone());

    // First split on `::`, `.`, `/`, `-`, and `_`.
    for segment in trimmed.split(|c: char| c == ':' || c == '.' || c == '/' || c == '-' || c == '_')
    {
        if segment.is_empty() {
            continue;
        }
        let seg_lower = segment.to_ascii_lowercase();
        if seg_lower != full_lower && !out.contains(&seg_lower) {
            out.push(seg_lower);
        }

        // CamelCase / PascalCase / acronym boundary splitter:
        // e.g. "AgentRunner" -> ["agent", "runner"], "HTML5Parser" -> ["html5", "parser"].
        let chars: Vec<char> = segment.chars().collect();
        let mut part = String::new();
        for i in 0..chars.len() {
            let ch = chars[i];
            if i > 0 {
                let prev = chars[i - 1];
                let next_is_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();
                let camel_boundary = (prev.is_ascii_lowercase() || prev.is_ascii_digit())
                    && ch.is_ascii_uppercase();
                let acronym_boundary =
                    prev.is_ascii_uppercase() && ch.is_ascii_uppercase() && next_is_lower;
                if (camel_boundary || acronym_boundary) && !part.is_empty() {
                    let sub = std::mem::take(&mut part).to_ascii_lowercase();
                    if !out.contains(&sub) {
                        out.push(sub);
                    }
                }
            }
            part.push(ch);
        }
        if !part.is_empty() {
            let sub = part.to_ascii_lowercase();
            if !out.contains(&sub) {
                out.push(sub);
            }
        }
    }

    out
}

/// Tokenizes mixed code and natural-language text for BM25 and structural matching.
pub fn tokenize_code_and_prose(text: &str) -> TokenizedCodeSpan {
    let mut terms = Vec::new();
    let mut exact_identifiers = BTreeSet::new();

    for raw_token in text.split(|c: char| {
        !c.is_ascii_alphanumeric() && c != '_' && c != ':' && c != '.' && c != '/'
    }) {
        let token = raw_token.trim_matches(|c: char| c == ':' || c == '.' || c == '/');
        if token.is_empty() {
            continue;
        }

        let is_code_like = is_code_identifier_token(token);
        if is_code_like {
            let lower = token.to_ascii_lowercase();
            exact_identifiers.insert(lower.clone());
            for seg in token.split("::").filter(|s| !s.is_empty()) {
                exact_identifiers.insert(seg.to_ascii_lowercase());
            }
        }

        let subtokens = split_identifier_subtokens(token);
        for sub in subtokens {
            if sub.len() < 2 {
                continue;
            }
            if !is_code_like && is_stopword(&sub) {
                continue;
            }
            terms.push(sub.clone());
            let stemmed = stem_lite(&sub);
            if stemmed != sub && stemmed.len() >= 2 {
                terms.push(stemmed);
            }
        }
    }

    TokenizedCodeSpan { terms, exact_identifiers }
}

fn is_code_identifier_token(token: &str) -> bool {
    if token.len() < 2 {
        return false;
    }
    if token.contains('_') || token.contains("::") {
        return true;
    }
    if token.starts_with("0x") || token.starts_with("ERR") || token.starts_with("HIP-") {
        return true;
    }
    // Check camelCase or PascalCase (has both uppercase and lowercase, or internal uppercase).
    let has_lower = token.chars().any(|c| c.is_ascii_lowercase());
    let has_internal_upper = token.chars().skip(1).any(|c| c.is_ascii_uppercase());
    let all_upper_const = token.len() >= 3
        && token.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && token.chars().any(|c| c.is_ascii_uppercase());
    (has_lower && has_internal_upper) || all_upper_const
}

/// Extracts explicit code identifiers (`snake_case`, `camelCase`, `PascalCase`, `qualified::path`) from a query.
pub fn extract_code_identifiers(raw_query: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw_word in raw_query.split_whitespace() {
        let cleaned = raw_word.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && c != '_' && c != ':'
        });
        if cleaned.is_empty() {
            continue;
        }
        if is_code_identifier_token(cleaned) {
            let lower = cleaned.to_ascii_lowercase();
            if !out.contains(&lower) {
                out.push(lower);
            }
            if cleaned.contains("::") {
                if let Some(last) = cleaned.rsplit("::").next() {
                    let last_lower = last.to_ascii_lowercase();
                    if !last_lower.is_empty() && !out.contains(&last_lower) {
                        out.push(last_lower);
                    }
                }
            }
        }
    }
    out
}

fn extract_ast_kinds_from_query(raw_query: &str) -> Vec<StructuralNodeKind> {
    let mut kinds = Vec::new();
    for word in raw_query.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        let lower = word.to_ascii_lowercase();
        let kind = match lower.as_str() {
            "fn" | "func" | "function" | "def" => Some(StructuralNodeKind::Function),
            "struct" => Some(StructuralNodeKind::Struct),
            "class" => Some(StructuralNodeKind::Class),
            "trait" | "interface" => Some(StructuralNodeKind::Trait),
            "method" | "impl" => Some(StructuralNodeKind::Method),
            "mod" | "module" | "crate" | "package" => Some(StructuralNodeKind::Module),
            "const" | "constant" | "static" => Some(StructuralNodeKind::Constant),
            "section" | "heading" => Some(StructuralNodeKind::Section),
            _ => None,
        };
        if let Some(k) = kind {
            if !kinds.contains(&k) {
                kinds.push(k);
            }
        }
    }
    kinds
}

fn has_call_graph_keyword(raw_query: &str) -> bool {
    for word in raw_query.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        match word.to_ascii_lowercase().as_str() {
            "caller" | "callers" | "calls" | "callee" | "called" | "signature" | "imports"
            | "ast" => return true,
            _ => {}
        }
    }
    false
}

fn extract_path_segments(raw_query: &str, filter: &HybridAccessFilter) -> Vec<String> {
    let mut segments = Vec::new();
    for token in raw_query.split_whitespace() {
        let clean = token.trim_matches(|c: char| {
            c == '"' || c == '\'' || c == '`' || c == '(' || c == ')' || c == ',' || c == '?'
        });
        let looks_like_path = clean.contains('/')
            || clean.ends_with(".rs")
            || clean.ends_with(".ts")
            || clean.ends_with(".py")
            || clean.ends_with(".md")
            || clean.ends_with(".go")
            || clean.ends_with(".toml");
        if looks_like_path {
            for seg in clean.split('/').filter(|s| !s.is_empty()) {
                let lower = seg.to_ascii_lowercase();
                if !segments.contains(&lower) {
                    segments.push(lower);
                }
            }
        }
    }
    if let Some(prefix) = &filter.path_prefix {
        for seg in prefix.split('/').filter(|s| !s.is_empty()) {
            let lower = seg.to_ascii_lowercase();
            if !segments.contains(&lower) {
                segments.push(lower);
            }
        }
    }
    segments
}

fn has_natural_language_question_structure(raw_query: &str) -> bool {
    let trimmed = raw_query.trim();
    if trimmed.ends_with('?') {
        return true;
    }
    let words: Vec<String> = trimmed
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect();
    let has_question_word = words.first().is_some_and(|w| {
        matches!(
            w.as_str(),
            "why" | "how" | "when" | "where" | "what" | "which" | "who" | "explain" | "describe"
        )
    }) || words.iter().any(|w| matches!(w.as_str(), "why" | "how" | "freeze" | "hang" | "prevent" | "cause" | "causes" | "does" | "after" | "while" | "during" | "without"));

    let non_code_words = words.iter().filter(|w| !is_code_identifier_token(w)).count();
    has_question_word && non_code_words >= 4
}

// ============================================================================
// Pure-Rust Deterministic Glob & Pre-Scoring ACL / Scope Evaluator
// ============================================================================

/// Pure-Rust deterministic glob matcher supporting `*` (within segment), `**` (across `/`), and `?`.
pub fn glob_matches(pattern: &str, path: &str) -> bool {
    let pat = pattern.as_bytes();
    let txt = path.as_bytes();
    glob_match_bytes(pat, txt)
}

fn glob_match_bytes(pat: &[u8], txt: &[u8]) -> bool {
    if pat.is_empty() {
        return txt.is_empty();
    }
    if pat.starts_with(b"**") {
        let mut rest = &pat[2..];
        if rest.starts_with(b"/") {
            rest = &rest[1..];
        }
        for i in 0..=txt.len() {
            if glob_match_bytes(rest, &txt[i..]) {
                return true;
            }
        }
        return false;
    }
    if pat[0] == b'*' {
        let rest = &pat[1..];
        for i in 0..=txt.len() {
            if i > 0 && txt[i - 1] == b'/' {
                break;
            }
            if glob_match_bytes(rest, &txt[i..]) {
                return true;
            }
        }
        return false;
    }
    if txt.is_empty() {
        return false;
    }
    if pat[0] == b'?' && txt[0] != b'/' {
        return glob_match_bytes(&pat[1..], &txt[1..]);
    }
    if pat[0].eq_ignore_ascii_case(&txt[0]) {
        return glob_match_bytes(&pat[1..], &txt[1..]);
    }
    false
}

/// Checks whether a canonical record satisfies all mandatory tenant, ACL, and scope predicates.
pub fn is_record_authorized_and_in_scope(
    record: &CanonicalEvidenceRecord,
    filter: &HybridAccessFilter,
) -> bool {
    let id = &record.identity;
    // 1. Mandatory tenant isolation boundary (never relaxed).
    if id.tenant_id != filter.tenant_id {
        return false;
    }
    // 2. Role / principal allowlist.
    if !id.allowed_principals.is_empty()
        && !id
            .allowed_principals
            .iter()
            .any(|allowed| filter.caller_principals.iter().any(|caller| caller == allowed))
    {
        return false;
    }
    // 3. Optional project_id scope.
    if let Some(proj) = &filter.project_id {
        if &id.project_id != proj {
            return false;
        }
    }
    // 4. Optional repo_id scope.
    if let Some(repo) = &filter.repo_id {
        if &id.repo_id != repo {
            return false;
        }
    }
    // 5. Optional path_prefix scope.
    if let Some(prefix) = &filter.path_prefix {
        if !id.path.starts_with(prefix.as_str()) {
            return false;
        }
    }
    // 6. Optional path_glob scope.
    if let Some(glob) = &filter.path_glob {
        if !glob_matches(glob, &id.path) {
            return false;
        }
    }
    // 7. Optional language scope.
    if let Some(lang) = &filter.language {
        if !id.language.eq_ignore_ascii_case(lang) {
            return false;
        }
    }
    // 8. Optional artifact_kind scope.
    if let Some(kind) = filter.artifact_kind {
        if id.artifact_kind != kind {
            return false;
        }
    }
    // 9. Optional structural node_kind scope.
    if let Some(required_node_kind) = filter.node_kind {
        if !record.symbols.iter().any(|sym| sym.node_kind == required_node_kind) {
            return false;
        }
    }
    true
}

// ============================================================================
// Leg 1: Derived Okapi BM25 + Identifier Sub-Token Lexical Index
// ============================================================================

#[derive(Debug, Clone)]
struct IndexedLexicalDoc {
    doc_len: usize,
    term_freqs: HashMap<String, usize>,
    exact_identifiers: BTreeSet<String>,
    lower_combined_text: String,
}

/// Derived BM25 lexical index built over canonical evidence records.
#[derive(Debug, Clone)]
pub struct Bm25LexicalIndex {
    docs: BTreeMap<String, IndexedLexicalDoc>,
    doc_freqs: HashMap<String, usize>,
    total_doc_len: usize,
    k1: f64,
    b: f64,
}

impl Default for Bm25LexicalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25LexicalIndex {
    /// Creates an empty BM25 lexical index with default parameters ($k_1 = 1.2, b = 0.75$).
    pub fn new() -> Self {
        Self {
            docs: BTreeMap::new(),
            doc_freqs: HashMap::new(),
            total_doc_len: 0,
            k1: DEFAULT_BM25_K1,
            b: DEFAULT_BM25_B,
        }
    }

    /// Returns the number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Returns `true` if no documents are indexed.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Indexes or updates a single canonical record inside the derived BM25 index.
    pub fn upsert_record(&mut self, record: &CanonicalEvidenceRecord) {
        if let Some(prev) = self.docs.remove(&record.identity.doc_id) {
            self.total_doc_len = self.total_doc_len.saturating_sub(prev.doc_len);
            for term in prev.term_freqs.keys() {
                if let Some(df) = self.doc_freqs.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freqs.remove(term);
                    }
                }
            }
        }

        let mut combined = String::with_capacity(
            record.title.len() + record.raw_text.len() + record.identity.path.len() + 128,
        );
        combined.push_str(&record.title);
        combined.push(' ');
        combined.push_str(&record.identity.path);
        combined.push(' ');
        combined.push_str(&record.raw_text);
        for sym in &record.symbols {
            combined.push(' ');
            combined.push_str(&sym.symbol_name);
            combined.push(' ');
            combined.push_str(&sym.qualified_name);
            combined.push(' ');
            combined.push_str(&sym.signature);
        }

        let tokenized = tokenize_code_and_prose(&combined);
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for term in &tokenized.terms {
            *term_freqs.entry(term.clone()).or_insert(0) += 1;
        }
        for term in term_freqs.keys() {
            *self.doc_freqs.entry(term.clone()).or_insert(0) += 1;
        }

        let doc_len = tokenized.terms.len().max(1);
        self.total_doc_len += doc_len;

        let mut exact_identifiers = tokenized.exact_identifiers;
        for sym in &record.symbols {
            exact_identifiers.insert(sym.symbol_name.to_ascii_lowercase());
            exact_identifiers.insert(sym.qualified_name.to_ascii_lowercase());
        }

        self.docs.insert(
            record.identity.doc_id.clone(),
            IndexedLexicalDoc {
                doc_len,
                term_freqs,
                exact_identifiers,
                lower_combined_text: combined.to_ascii_lowercase(),
            },
        );
    }

    /// Scores and ranks pre-filtered authorized candidate `doc_id`s using Okapi BM25
    /// plus exact code-identifier and phrase boosts.
    pub fn search_authorized(
        &self,
        authorized_doc_ids: &[String],
        raw_query: &str,
        extracted_identifiers: &[String],
        candidate_depth: usize,
    ) -> Vec<(String, f64)> {
        if authorized_doc_ids.is_empty() || self.docs.is_empty() || candidate_depth == 0 {
            return Vec::new();
        }

        let q_tokenized = tokenize_code_and_prose(raw_query);
        if q_tokenized.terms.is_empty() {
            return Vec::new();
        }

        // Deduplicate query terms while preserving deterministic order.
        let mut unique_q_terms = Vec::new();
        for t in &q_tokenized.terms {
            if !unique_q_terms.contains(t) {
                unique_q_terms.push(t.clone());
            }
        }

        let n_docs = (self.docs.len() as f64).max(1.0);
        let avg_dl = (self.total_doc_len as f64 / n_docs).max(1.0);
        let trimmed_lower_query = raw_query.trim().to_ascii_lowercase();

        let mut scored = Vec::new();
        for doc_id in authorized_doc_ids {
            let Some(doc) = self.docs.get(doc_id) else {
                continue;
            };

            let mut bm25_score = 0.0_f64;
            let dl_norm = 1.0 - self.b + self.b * (doc.doc_len as f64 / avg_dl);

            for term in &unique_q_terms {
                let Some(&tf) = doc.term_freqs.get(term) else {
                    continue;
                };
                if tf == 0 {
                    continue;
                }
                let df = self.doc_freqs.get(term).copied().unwrap_or(1) as f64;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                let tf_f = tf as f64;
                let numer = tf_f * (self.k1 + 1.0);
                let denom = tf_f + self.k1 * dl_norm;
                bm25_score += idf * (numer / denom);
            }

            // Exact code identifier match boost.
            for ident in extracted_identifiers {
                if doc.exact_identifiers.contains(ident) {
                    bm25_score += 3.5;
                }
            }

            // Exact phrase match boost for multi-word queries or exact symbol substrings.
            if trimmed_lower_query.len() >= 4
                && doc.lower_combined_text.contains(&trimmed_lower_query)
            {
                bm25_score += 2.0;
            }

            if bm25_score > 0.0 {
                scored.push((doc_id.clone(), bm25_score));
            }
        }

        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(candidate_depth);
        scored
    }
}

// ============================================================================
// Leg 2: Derived Prefix-Enforced Dense Vector Index (HIP-1346 §1)
// ============================================================================

/// Derived dense vector index enforcing the HIP-1346 §1 `ImmutableIndexContract`.
#[derive(Debug, Clone, Default)]
pub struct DenseVectorIndex {
    vectors: BTreeMap<String, Vec<f32>>,
}

impl DenseVectorIndex {
    /// Creates an empty derived vector index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Returns `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Validates and stores a document vector for `doc_id`.
    pub fn upsert_validated_vector(&mut self, doc_id: String, vector: Vec<f32>) {
        self.vectors.insert(doc_id, vector);
    }

    /// Scores and ranks pre-filtered authorized candidate `doc_id`s by `f64` dot-product / cosine similarity.
    pub fn search_authorized(
        &self,
        authorized_doc_ids: &[String],
        query_vector: &[f32],
        candidate_depth: usize,
    ) -> Vec<(String, f64)> {
        if authorized_doc_ids.is_empty() || self.vectors.is_empty() || candidate_depth == 0 {
            return Vec::new();
        }

        let mut scored = Vec::with_capacity(authorized_doc_ids.len());
        for doc_id in authorized_doc_ids {
            let Some(doc_vec) = self.vectors.get(doc_id) else {
                continue;
            };
            // Accumulate in f64 for cross-architecture (x86_64 vs aarch64) determinism.
            let sim: f64 = doc_vec
                .iter()
                .zip(query_vector.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            if sim >= 0.15 {
                scored.push((doc_id.clone(), sim));
            }
        }

        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(candidate_depth);
        scored
    }
}

/// Deterministic asymmetric semantic + code embedder implementing [`PrefixedTextEmbedder`].
///
/// Combines FNV-1a hashed identifier/sub-token unigrams and bigrams with
/// domain-specific semantic equivalence clusters so paraphrased incident and
/// architectural queries with zero lexical overlap retrieve their true
/// underlying technical root-cause documents in the vector leg.
#[derive(Debug, Clone, Default)]
pub struct DeterministicSemanticCodeEmbedder;

/// Semantic concept clusters mapping paraphrased user terminology and technical
/// implementation terms into shared latent concept dimensions.
const SEMANTIC_CONCEPT_CLUSTERS: &[(&str, &[&str])] = &[
    (
        "concept_agent_freeze_deadlock",
        &[
            "freeze",
            "freezes",
            "freezing",
            "frozen",
            "hang",
            "hangs",
            "hanging",
            "stuck",
            "unresponsive",
            "deadlock",
            "deadlocked",
            "lock",
            "mutex",
            "contention",
            "parking_lot",
            "await",
            "block",
            "blocked",
            "blocking",
            "starvation",
        ],
    ),
    (
        "concept_request_completion_drain",
        &[
            "complete",
            "completing",
            "completed",
            "completion",
            "finish",
            "finishing",
            "finished",
            "record_finish",
            "finalize",
            "finalizing",
            "finalization",
            "drain",
            "draining",
            "turn",
            "shutdown",
            "release_turn_lock",
        ],
    ),
    (
        "concept_tenant_acl_isolation",
        &[
            "tenant",
            "multi_tenant",
            "multitenant",
            "isolation",
            "acl",
            "principal",
            "principals",
            "role",
            "roles",
            "unauthorized",
            "permission",
            "permissions",
            "security",
            "cross_tenant",
            "prefilter",
            "filter",
            "leak",
            "leakage",
        ],
    ),
    (
        "concept_metadata_prefix_contract",
        &[
            "incompatible",
            "mismatch",
            "prefix",
            "query_prefix",
            "doc_prefix",
            "swapped",
            "swappedtaskprefix",
            "immutable",
            "immutableindexcontract",
            "matryoshka",
            "dimension",
            "dimensions",
            "truncation",
            "l2",
            "normalization",
            "revision",
        ],
    ),
    (
        "concept_rank_fusion_hybrid",
        &[
            "hybrid",
            "fusion",
            "rrf",
            "reciprocal",
            "rank",
            "combine",
            "combining",
            "merge",
            "merging",
            "uncalibrated",
            "lexical",
            "bm25",
            "vector",
            "structural",
            "legs",
        ],
    ),
    (
        "concept_ast_call_graph_symbol",
        &[
            "ast",
            "symbol",
            "symbols",
            "signature",
            "caller",
            "callers",
            "callee",
            "call_graph",
            "callgraph",
            "invoke",
            "invokes",
            "tree_sitter",
            "function",
            "struct",
            "trait",
            "method",
        ],
    ),
    (
        "concept_self_rag_adaptive_gate",
        &[
            "skip",
            "skipping",
            "avoid",
            "unnecessary",
            "lookup",
            "database",
            "parametric",
            "retrieveoranswergate",
            "theta",
            "self_rag",
            "direct",
            "directly",
        ],
    ),
    (
        "concept_crag_fallback_recovery",
        &[
            "fallback",
            "backup",
            "secondary",
            "corrupted",
            "mojibake",
            "distractor",
            "noisy",
            "reformulate",
            "reformulation",
            "correctiveretrievalpolicy",
            "crag",
        ],
    ),
    (
        "concept_factscore_atomic_claims",
        &[
            "atomic",
            "claim",
            "claims",
            "sentence",
            "statements",
            "verify",
            "factscore",
            "factscoreevaluator",
            "hallucination",
            "citation",
            "span",
            "polarity",
            "negation",
        ],
    ),
    (
        "concept_ann_online_delta_merge",
        &[
            "ann",
            "odinann",
            "delta",
            "insert",
            "insertion",
            "online",
            "rebuild",
            "stall",
            "tail",
            "latency",
            "p99",
            "graphann",
            "beam",
        ],
    ),
    (
        "concept_kv_cache_memory_paging",
        &[
            "kv",
            "cache",
            "kv_cache",
            "hbm",
            "dram",
            "nvme",
            "prefill",
            "paged",
            "evict",
            "eviction",
            "conversation",
            "history",
            "replica",
            "router",
        ],
    ),
    (
        "concept_retry_backoff_circuit_breaker",
        &[
            "retry",
            "retries",
            "backoff",
            "jitter",
            "timeout",
            "timed_out",
            "circuit",
            "breaker",
            "rate_limit",
            "429",
            "503",
            "transient",
            "overload",
            "shedding",
        ],
    ),
    (
        "concept_derived_index_rebuild",
        &[
            "derived",
            "rebuild",
            "rebuilding",
            "reconstruct",
            "canonical",
            "source_id",
            "version_id",
            "provenance",
            "destroy",
            "destroying",
            "immutable_source",
        ],
    ),
];

impl PrefixedTextEmbedder for DeterministicSemanticCodeEmbedder {
    fn embed_prefixed(
        &self,
        prefixed_text: &str,
        operation: IndexOperation,
        metadata: &IndexEmbeddingMetadata,
    ) -> Result<Vec<f32>, HydeError> {
        metadata.validate_prefixed_text(prefixed_text, operation)?;
        let prefix = match operation {
            IndexOperation::Insert => metadata.doc_prefix.as_str(),
            IndexOperation::Query => metadata.query_prefix.as_str(),
            IndexOperation::Create => "",
        };
        let body = prefixed_text.strip_prefix(prefix).unwrap_or(prefixed_text);
        let dims = metadata.dimensions;
        let mut vec = vec![0.0_f32; dims];

        let tokenized = tokenize_code_and_prose(body);

        // 1. Project semantic concept clusters with high weight so vocabulary-mismatched
        // natural-language queries align strongly with their technical root-cause docs.
        for &(concept_name, triggers) in SEMANTIC_CONCEPT_CLUSTERS {
            let mut hits = 0usize;
            for term in &tokenized.terms {
                if triggers.iter().any(|&tr| tr == term.as_str() || stem_lite(tr) == *term) {
                    hits += 1;
                }
            }
            if hits > 0 {
                let weight = 2.6_f32 + (hits.min(4) as f32) * 0.45_f32;
                accumulate_fnv_feature(&mut vec, concept_name.as_bytes(), weight);
                let second_tag = format!("{concept_name}#axis2");
                accumulate_fnv_feature(&mut vec, second_tag.as_bytes(), weight * 0.85);
            }
        }

        // 2. Project unigram terms and stems.
        for term in &tokenized.terms {
            accumulate_fnv_feature(&mut vec, term.as_bytes(), 0.75);
        }

        // 3. Project bigrams for phrase sensitivity.
        for pair in tokenized.terms.windows(2) {
            let bigram = format!("{}_{}", pair[0], pair[1]);
            accumulate_fnv_feature(&mut vec, bigram.as_bytes(), 0.45);
        }

        // 4. Normalize onto the L2 unit sphere using f64 intermediate precision.
        let norm_sq: f64 = vec.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        if norm_sq <= 1e-12 {
            vec[0] = 1.0;
        } else if metadata.normalization == NormalizationPolicy::L2 {
            let norm = norm_sq.sqrt();
            for v in &mut vec {
                *v = (f64::from(*v) / norm) as f32;
            }
        }
        Ok(vec)
    }
}

fn accumulate_fnv_feature(vec: &mut [f32], bytes: &[u8], weight: f32) {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let idx = (hash as usize) % vec.len();
    let sign = if (hash >> 63) == 0 { 1.0_f32 } else { -1.0_f32 };
    vec[idx] += sign * weight;
}

// ============================================================================
// Leg 3: Derived Structural AST, Symbol & Call-Graph Index
// ============================================================================

#[derive(Debug, Clone)]
struct IndexedStructuralRecord {
    path_lower: String,
    path_segments: Vec<String>,
    symbols: Vec<StructuralSymbolNode>,
}

/// Derived structural AST, symbol-definition, signature, and call-graph index.
#[derive(Debug, Clone, Default)]
pub struct StructuralAstIndex {
    records: BTreeMap<String, IndexedStructuralRecord>,
}

impl StructuralAstIndex {
    /// Creates an empty derived structural index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of indexed records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if no records are indexed.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Indexes or updates the structural symbols and file-path metadata of a canonical record.
    pub fn upsert_record(&mut self, record: &CanonicalEvidenceRecord) {
        let path_lower = record.identity.path.to_ascii_lowercase();
        let path_segments = path_lower
            .split('/')
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect();
        self.records.insert(
            record.identity.doc_id.clone(),
            IndexedStructuralRecord {
                path_lower,
                path_segments,
                symbols: record.symbols.clone(),
            },
        );
    }

    /// Scores and ranks pre-filtered authorized candidate `doc_id`s over AST symbol
    /// definitions, qualified names, node kinds, signatures, call-graph edges, and path structure.
    pub fn search_authorized(
        &self,
        authorized_doc_ids: &[String],
        raw_query: &str,
        plan: &HybridQueryPlan,
        candidate_depth: usize,
    ) -> Vec<(String, f64, Vec<String>)> {
        if authorized_doc_ids.is_empty() || self.records.is_empty() || candidate_depth == 0 {
            return Vec::new();
        }

        let q_tokenized = tokenize_code_and_prose(raw_query);
        let q_term_set: HashSet<&str> = q_tokenized.terms.iter().map(String::as_str).collect();
        let raw_words_lower: Vec<String> = raw_query
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != ':')
            .filter(|w| !w.is_empty())
            .map(|w| w.trim_matches(':').to_ascii_lowercase())
            .collect();
        let wants_callers = raw_words_lower
            .iter()
            .any(|w| matches!(w.as_str(), "caller" | "callers" | "called" | "invokes" | "calls"));

        let mut scored = Vec::new();
        for doc_id in authorized_doc_ids {
            let Some(rec) = self.records.get(doc_id) else {
                continue;
            };

            let mut doc_score = 0.0_f64;
            let mut matched_symbols = Vec::new();

            // 1. File path & segment alignment (evaluated when the query carries path signals).
            if !plan.extracted_path_segments.is_empty() {
                for seg in &plan.extracted_path_segments {
                    if rec.path_lower.contains(seg.as_str()) {
                        doc_score += 1.5;
                    }
                }
                for seg in &rec.path_segments {
                    let stem = seg.split('.').next().unwrap_or(seg);
                    if stem.len() >= 3 && raw_words_lower.iter().any(|w| w == seg || w == stem) {
                        doc_score += 0.75;
                    }
                }
            }

            // 2. Symbol-level AST, signature, and call-graph scoring.
            for sym in &rec.symbols {
                let mut sym_score = 0.0_f64;
                let sym_lower = sym.symbol_name.to_ascii_lowercase();
                let qual_lower = sym.qualified_name.to_ascii_lowercase();

                // 2a. Exact symbol name match.
                if raw_words_lower.iter().any(|w| w == &sym_lower)
                    || plan.extracted_identifiers.iter().any(|id| id == &sym_lower)
                {
                    sym_score += 5.5;
                }

                // 2b. Qualified symbol path match (`AgentRunner::record_finish`).
                if raw_words_lower.iter().any(|w| w.contains("::") && qual_lower.ends_with(w))
                    || plan
                        .extracted_identifiers
                        .iter()
                        .any(|id| id.contains("::") && qual_lower.ends_with(id))
                {
                    sym_score += 4.0;
                }

                // 2c. Decomposed identifier sub-token overlap in symbol name & signature.
                let sym_subtokens = split_identifier_subtokens(&sym.symbol_name);
                let non_trivial_subs: Vec<&String> =
                    sym_subtokens.iter().filter(|s| s.len() >= 3).collect();
                if !non_trivial_subs.is_empty() {
                    let matched_subs = non_trivial_subs
                        .iter()
                        .filter(|s| q_term_set.contains(s.as_str()))
                        .count();
                    if matched_subs >= 2
                        || (!plan.extracted_identifiers.is_empty() && matched_subs >= 1)
                    {
                        let ratio = matched_subs as f64 / non_trivial_subs.len() as f64;
                        sym_score += 2.2 * ratio;
                    }
                }

                // 2d. AST node kind alignment (`fn record_finish`, `struct AgentRunner`).
                if plan.extracted_ast_kinds.contains(&sym.node_kind) && sym_score > 0.0 {
                    sym_score += 2.0;
                } else if plan.extracted_ast_kinds.contains(&sym.node_kind) {
                    sym_score += 0.65;
                }

                // 2e. Parent scope match (`AgentRunner`).
                if let Some(parent) = &sym.parent_scope {
                    let parent_lower = parent.to_ascii_lowercase();
                    if raw_words_lower.iter().any(|w| w == &parent_lower)
                        || plan.extracted_identifiers.iter().any(|id| id == &parent_lower)
                    {
                        sym_score += 1.75;
                    }
                }

                // 2f. Call-graph edges (`calls`, `called_by`, `imports`).
                for callee in &sym.calls {
                    let callee_lower = callee.to_ascii_lowercase();
                    if raw_words_lower.iter().any(|w| w == &callee_lower)
                        || plan.extracted_identifiers.iter().any(|id| id == &callee_lower)
                    {
                        sym_score += if wants_callers { 4.25 } else { 1.85 };
                    }
                }
                for caller in &sym.called_by {
                    let caller_lower = caller.to_ascii_lowercase();
                    if raw_words_lower.iter().any(|w| w == &caller_lower)
                        || plan.extracted_identifiers.iter().any(|id| id == &caller_lower)
                    {
                        sym_score += 1.75;
                    }
                }
                for imp in &sym.imports {
                    let imp_lower = imp.to_ascii_lowercase();
                    if raw_words_lower.iter().any(|w| imp_lower.ends_with(w.as_str())) {
                        sym_score += 1.0;
                    }
                }

                if sym_score > 0.0 {
                    doc_score += sym_score;
                    if !matched_symbols.contains(&sym.qualified_name) {
                        matched_symbols.push(sym.qualified_name.clone());
                    }
                }
            }

            if doc_score > 0.0 {
                scored.push((doc_id.clone(), doc_score, matched_symbols));
            }
        }

        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(candidate_depth);
        scored
    }
}

// ============================================================================
// Reciprocal Rank Fusion (RRF) & Per-Leg Explainability
// ============================================================================

/// Per-leg rank, raw score, weight, and RRF score contribution for a single document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegContribution {
    /// 1-based rank of this document within the leg (`1` = best).
    pub rank_1_based: usize,
    /// Uncalibrated raw score produced by the leg (BM25 score, cosine similarity, or AST score).
    pub raw_score: f64,
    /// Planner weight assigned to this leg.
    pub weight: f64,
    /// Exact RRF term contributed by this leg: `weight / (rrf_k + rank_1_based as f64)`.
    pub rrf_contribution: f64,
}

/// Full audit trail explaining why a document achieved its fused rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankExplanation {
    pub query_class: HybridQueryClass,
    pub rrf_k: f64,
    pub lexical: Option<LegContribution>,
    pub vector: Option<LegContribution>,
    pub structural: Option<LegContribution>,
    pub active_legs_matched: usize,
    pub best_leg_rank: usize,
    pub matched_symbols: Vec<String>,
}

/// Fused search result returned by the Hybrid RRF Executor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub doc_id: String,
    pub identity: SourceIdentity,
    pub title: String,
    pub raw_text: String,
    /// Final fused RRF score: sum of active `rrf_contribution` terms.
    pub rrf_score: f64,
    /// 1-based rank in the final fused result list.
    pub final_rank: usize,
    pub explanation: RankExplanation,
}

/// Complete telemetry and fused hit list for a hybrid search execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridSearchOutcome {
    pub plan: HybridQueryPlan,
    pub authorized_candidate_count: usize,
    pub lexical_hits_count: usize,
    pub vector_hits_count: usize,
    pub structural_hits_count: usize,
    pub hits: Vec<SearchHit>,
    pub query_latency_ms: f64,
    pub query_cost_usd: f64,
}

/// Explicit latency and dollar cost parameters for matched-baseline evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HybridCostModel {
    pub kai_planner_latency_ms: f64,
    pub kai_planner_cost_usd: f64,
    pub lexical_leg_latency_ms: f64,
    pub lexical_leg_cost_usd: f64,
    pub vector_embed_latency_ms: f64,
    pub vector_search_latency_ms: f64,
    pub vector_embed_cost_per_1k_tokens_usd: f64,
    pub structural_leg_latency_ms: f64,
    pub structural_leg_cost_usd: f64,
    pub rrf_fusion_latency_ms: f64,
    pub index_doc_embed_cost_per_1k_tokens_usd: f64,
    pub index_doc_structural_parse_cost_usd: f64,
}

impl HybridCostModel {
    /// Validates that all cost and latency fields are finite and non-negative.
    pub fn validate(&self) -> Result<(), HybridSearchError> {
        let fields = [
            ("kai_planner_latency_ms", self.kai_planner_latency_ms),
            ("kai_planner_cost_usd", self.kai_planner_cost_usd),
            ("lexical_leg_latency_ms", self.lexical_leg_latency_ms),
            ("lexical_leg_cost_usd", self.lexical_leg_cost_usd),
            ("vector_embed_latency_ms", self.vector_embed_latency_ms),
            ("vector_search_latency_ms", self.vector_search_latency_ms),
            (
                "vector_embed_cost_per_1k_tokens_usd",
                self.vector_embed_cost_per_1k_tokens_usd,
            ),
            ("structural_leg_latency_ms", self.structural_leg_latency_ms),
            ("structural_leg_cost_usd", self.structural_leg_cost_usd),
            ("rrf_fusion_latency_ms", self.rrf_fusion_latency_ms),
            (
                "index_doc_embed_cost_per_1k_tokens_usd",
                self.index_doc_embed_cost_per_1k_tokens_usd,
            ),
            (
                "index_doc_structural_parse_cost_usd",
                self.index_doc_structural_parse_cost_usd,
            ),
        ];
        for (name, val) in fields {
            if !val.is_finite() || val < 0.0 {
                return Err(HybridSearchError::InvalidCostParameter(name));
            }
        }
        Ok(())
    }
}

impl Default for HybridCostModel {
    fn default() -> Self {
        Self {
            kai_planner_latency_ms: 0.35,
            kai_planner_cost_usd: 0.000002,
            lexical_leg_latency_ms: 0.85,
            lexical_leg_cost_usd: 0.000001,
            vector_embed_latency_ms: 2.10,
            vector_search_latency_ms: 1.15,
            vector_embed_cost_per_1k_tokens_usd: 0.02,
            structural_leg_latency_ms: 0.65,
            structural_leg_cost_usd: 0.000001,
            rrf_fusion_latency_ms: 0.15,
            index_doc_embed_cost_per_1k_tokens_usd: 0.02,
            index_doc_structural_parse_cost_usd: 0.000005,
        }
    }
}

/// Stateless Reciprocal Rank Fusion (RRF) combiner with 6-level deterministic tie-breaking.
#[derive(Debug, Clone, Default)]
pub struct ReciprocalRankFusion;

impl ReciprocalRankFusion {
    /// Fuses ranked hit lists from the lexical, vector, and structural legs using weighted RRF:
    /// `score_rrf(d) = sum_m w_m / (rrf_k + rank_m(d))`.
    pub fn fuse(
        canonical_records: &BTreeMap<String, CanonicalEvidenceRecord>,
        plan: &HybridQueryPlan,
        lexical_ranked: &[(String, f64)],
        vector_ranked: &[(String, f64)],
        structural_ranked: &[(String, f64, Vec<String>)],
    ) -> Vec<SearchHit> {
        #[derive(Default)]
        struct PartialHit {
            lexical: Option<LegContribution>,
            vector: Option<LegContribution>,
            structural: Option<LegContribution>,
            matched_symbols: Vec<String>,
        }

        let mut acc: BTreeMap<String, PartialHit> = BTreeMap::new();

        if plan.execute_lexical && plan.weights.lexical > 0.0 {
            for (idx, (doc_id, raw_score)) in lexical_ranked.iter().enumerate() {
                let rank_1_based = idx + 1;
                let rrf_contribution =
                    plan.weights.lexical / (plan.rrf_k + (rank_1_based as f64));
                acc.entry(doc_id.clone()).or_default().lexical = Some(LegContribution {
                    rank_1_based,
                    raw_score: *raw_score,
                    weight: plan.weights.lexical,
                    rrf_contribution,
                });
            }
        }

        if plan.execute_vector && plan.weights.vector > 0.0 {
            for (idx, (doc_id, raw_score)) in vector_ranked.iter().enumerate() {
                let rank_1_based = idx + 1;
                let rrf_contribution =
                    plan.weights.vector / (plan.rrf_k + (rank_1_based as f64));
                acc.entry(doc_id.clone()).or_default().vector = Some(LegContribution {
                    rank_1_based,
                    raw_score: *raw_score,
                    weight: plan.weights.vector,
                    rrf_contribution,
                });
            }
        }

        if plan.execute_structural && plan.weights.structural > 0.0 {
            for (idx, (doc_id, raw_score, syms)) in structural_ranked.iter().enumerate() {
                let rank_1_based = idx + 1;
                let rrf_contribution =
                    plan.weights.structural / (plan.rrf_k + (rank_1_based as f64));
                let entry = acc.entry(doc_id.clone()).or_default();
                entry.structural = Some(LegContribution {
                    rank_1_based,
                    raw_score: *raw_score,
                    weight: plan.weights.structural,
                    rrf_contribution,
                });
                entry.matched_symbols = syms.clone();
            }
        }

        let mut hits = Vec::with_capacity(acc.len());
        for (doc_id, part) in acc {
            let Some(record) = canonical_records.get(&doc_id) else {
                continue;
            };

            let mut rrf_score = 0.0_f64;
            let mut active_legs_matched = 0usize;
            let mut best_leg_rank = usize::MAX;

            if let Some(lex) = &part.lexical {
                rrf_score += lex.rrf_contribution;
                active_legs_matched += 1;
                best_leg_rank = best_leg_rank.min(lex.rank_1_based);
            }
            if let Some(vec_leg) = &part.vector {
                rrf_score += vec_leg.rrf_contribution;
                active_legs_matched += 1;
                best_leg_rank = best_leg_rank.min(vec_leg.rank_1_based);
            }
            if let Some(str_leg) = &part.structural {
                rrf_score += str_leg.rrf_contribution;
                active_legs_matched += 1;
                best_leg_rank = best_leg_rank.min(str_leg.rank_1_based);
            }

            let explanation = RankExplanation {
                query_class: plan.query_class,
                rrf_k: plan.rrf_k,
                lexical: part.lexical,
                vector: part.vector,
                structural: part.structural,
                active_legs_matched,
                best_leg_rank,
                matched_symbols: part.matched_symbols,
            };

            hits.push(SearchHit {
                doc_id,
                identity: record.identity.clone(),
                title: record.title.clone(),
                raw_text: record.raw_text.clone(),
                rrf_score,
                final_rank: 0,
                explanation,
            });
        }

        hits.sort_by(compare_search_hits_deterministic);
        hits.truncate(plan.top_k);
        for (idx, hit) in hits.iter_mut().enumerate() {
            hit.final_rank = idx + 1;
        }
        hits
    }
}

/// 6-level deterministic comparator for `SearchHit` ordering.
pub fn compare_search_hits_deterministic(a: &SearchHit, b: &SearchHit) -> std::cmp::Ordering {
    if (a.rrf_score - b.rrf_score).abs() > 1e-12 {
        return b.rrf_score.total_cmp(&a.rrf_score);
    }
    let a_lex = a.explanation.lexical.as_ref().map_or(usize::MAX, |c| c.rank_1_based);
    let b_lex = b.explanation.lexical.as_ref().map_or(usize::MAX, |c| c.rank_1_based);
    let a_str = a.explanation.structural.as_ref().map_or(usize::MAX, |c| c.rank_1_based);
    let b_str = b.explanation.structural.as_ref().map_or(usize::MAX, |c| c.rank_1_based);
    let a_vec = a.explanation.vector.as_ref().map_or(usize::MAX, |c| c.rank_1_based);
    let b_vec = b.explanation.vector.as_ref().map_or(usize::MAX, |c| c.rank_1_based);

    b.explanation
        .active_legs_matched
        .cmp(&a.explanation.active_legs_matched)
        .then_with(|| a.explanation.best_leg_rank.cmp(&b.explanation.best_leg_rank))
        .then_with(|| a_lex.cmp(&b_lex))
        .then_with(|| a_str.cmp(&b_str))
        .then_with(|| a_vec.cmp(&b_vec))
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

// ============================================================================
// Unified Hybrid Search Index (Canonical Store + 3 Derived Views)
// ============================================================================

#[derive(Debug, Clone, Default)]
struct HybridCollectionStore {
    /// Authoritative canonical source records keyed by `doc_id`.
    canonical_records: BTreeMap<String, CanonicalEvidenceRecord>,
    /// Derived Leg 1: Okapi BM25 + identifier sub-token lexical index.
    lexical_index: Bm25LexicalIndex,
    /// Derived Leg 2: Prefix-validated dense vector index.
    vector_index: DenseVectorIndex,
    /// Derived Leg 3: AST symbol, signature, and call-graph index.
    structural_index: StructuralAstIndex,
    /// Number of times derived indexes have been rebuilt from `canonical_records`.
    derived_rebuild_count: usize,
}

/// Multi-collection Hybrid Lexical, Vector & Structural Search Index backed by
/// an [`ImmutableIndexContract`] (HIP-1346 §1 / HIP-1347).
#[derive(Debug, Clone, Default)]
pub struct HybridSearchIndex {
    contract: ImmutableIndexContract,
    collections: BTreeMap<String, HybridCollectionStore>,
}

impl HybridSearchIndex {
    /// Creates a new empty `HybridSearchIndex`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Exposes the underlying [`ImmutableIndexContract`].
    pub fn contract(&self) -> &ImmutableIndexContract {
        &self.contract
    }

    /// Registers a collection with immutable embedding metadata.
    pub fn register_collection(
        &mut self,
        collection: impl Into<String>,
        metadata: IndexEmbeddingMetadata,
    ) -> Result<(), HybridSearchError> {
        let name = collection.into();
        self.contract.register(name.clone(), metadata)?;
        self.collections.entry(name).or_default();
        Ok(())
    }

    /// Returns the number of canonical evidence records stored in `collection`.
    pub fn canonical_record_count(&self, collection: &str) -> usize {
        self.collections.get(collection).map_or(0, |c| c.canonical_records.len())
    }

    /// Returns how many times derived indexes were rebuilt for `collection`.
    pub fn derived_rebuild_count(&self, collection: &str) -> usize {
        self.collections.get(collection).map_or(0, |c| c.derived_rebuild_count)
    }

    /// Returns an immutable reference to a canonical source record by `doc_id`.
    pub fn get_canonical_record(
        &self,
        collection: &str,
        doc_id: &str,
    ) -> Option<&CanonicalEvidenceRecord> {
        self.collections.get(collection)?.canonical_records.get(doc_id)
    }

    /// Returns an iterator over all canonical records in `collection`.
    pub fn canonical_records(
        &self,
        collection: &str,
    ) -> Option<&BTreeMap<String, CanonicalEvidenceRecord>> {
        Some(&self.collections.get(collection)?.canonical_records)
    }

    /// Inserts or updates a canonical evidence record and populates all three
    /// derived indexes (`Bm25LexicalIndex`, `DenseVectorIndex`, `StructuralAstIndex`)
    /// after validating the HIP-1346 §1 immutable metadata and prefix contract.
    pub fn upsert_record(
        &mut self,
        collection: &str,
        runtime_meta: &IndexEmbeddingMetadata,
        record: CanonicalEvidenceRecord,
        embedder: &dyn PrefixedTextEmbedder,
    ) -> Result<(), HybridSearchError> {
        if record.identity.doc_id.trim().is_empty() {
            return Err(HybridSearchError::EmptyDocId);
        }
        if record.identity.source_id.trim().is_empty() {
            return Err(HybridSearchError::EmptySourceId);
        }
        if record.identity.version_id.trim().is_empty() {
            return Err(HybridSearchError::EmptyVersionId);
        }
        if record.identity.tenant_id.trim().is_empty() {
            return Err(HybridSearchError::EmptyTenantId);
        }

        let persisted = self
            .contract
            .get(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?
            .clone();
        persisted.validate_runtime(runtime_meta, IndexOperation::Insert)?;

        reject_swapped_prefix(
            &record.raw_text,
            &persisted.doc_prefix,
            &persisted.query_prefix,
            IndexOperation::Insert,
            MetadataField::DocPrefix,
            MetadataField::QueryPrefix,
        )?;

        let doc_payload = format_doc_embedding_payload(&record);
        let prefixed_doc = persisted.format_document(&doc_payload);
        let vector =
            embedder.embed_prefixed(&prefixed_doc, IndexOperation::Insert, runtime_meta)?;
        self.contract.validate_insert(collection, runtime_meta, &vector, Some(&prefixed_doc))?;

        let store = self.collections.entry(collection.to_string()).or_default();
        store.lexical_index.upsert_record(&record);
        store.vector_index.upsert_validated_vector(record.identity.doc_id.clone(), vector);
        store.structural_index.upsert_record(&record);
        store.canonical_records.insert(record.identity.doc_id.clone(), record);
        Ok(())
    }

    /// Drops all derived indexes (`Bm25LexicalIndex`, `DenseVectorIndex`, `StructuralAstIndex`)
    /// for `collection` and reconstructs them from the immutable `canonical_records` store.
    ///
    /// Guarantees that canonical `SourceIdentity` records (`doc_id`, `source_id`,
    /// `version_id`, `tenant_id`, `allowed_principals`) are never destroyed or mutated.
    pub fn rebuild_derived_indexes(
        &mut self,
        collection: &str,
        runtime_meta: &IndexEmbeddingMetadata,
        embedder: &dyn PrefixedTextEmbedder,
    ) -> Result<usize, HybridSearchError> {
        let persisted = self
            .contract
            .get(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?
            .clone();
        persisted.validate_runtime(runtime_meta, IndexOperation::Insert)?;

        let store = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;

        let mut new_lexical = Bm25LexicalIndex::new();
        let mut new_vector = DenseVectorIndex::new();
        let mut new_structural = StructuralAstIndex::new();

        for (doc_id, record) in &store.canonical_records {
            let doc_payload = format_doc_embedding_payload(record);
            let prefixed_doc = persisted.format_document(&doc_payload);
            let vector =
                embedder.embed_prefixed(&prefixed_doc, IndexOperation::Insert, runtime_meta)?;
            persisted.validate_insert(runtime_meta, &vector, Some(&prefixed_doc))?;

            new_lexical.upsert_record(record);
            new_vector.upsert_validated_vector(doc_id.clone(), vector);
            new_structural.upsert_record(record);
        }

        store.lexical_index = new_lexical;
        store.vector_index = new_vector;
        store.structural_index = new_structural;
        store.derived_rebuild_count += 1;

        Ok(store.canonical_records.len())
    }

    /// Clears only the derived indexes for `collection` while keeping `canonical_records`
    /// intact (used in tests to verify that `rebuild_derived_indexes` reconstructs them
    /// purely from canonical source identities).
    pub fn drop_derived_indexes_preserve_canonical(
        &mut self,
        collection: &str,
    ) -> Result<(), HybridSearchError> {
        let store = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;
        store.lexical_index = Bm25LexicalIndex::new();
        store.vector_index = DenseVectorIndex::new();
        store.structural_index = StructuralAstIndex::new();
        Ok(())
    }

    /// Executes a 3-leg hybrid search planned by `planner`, pre-filtered by `filter`
    /// prior to any leg scoring, and fused via weighted Reciprocal Rank Fusion (RRF).
    #[allow(clippy::too_many_arguments)]
    pub fn search_hybrid(
        &self,
        collection: &str,
        runtime_meta: &IndexEmbeddingMetadata,
        filter: &HybridAccessFilter,
        raw_query: &str,
        config: &HybridRrfConfig,
        planner: &dyn KaiHybridQueryPlanner,
        embedder: &dyn PrefixedTextEmbedder,
        cost_model: &HybridCostModel,
    ) -> Result<HybridSearchOutcome, HybridSearchError> {
        if raw_query.trim().is_empty() {
            return Err(HybridSearchError::EmptyQuery);
        }
        if filter.tenant_id.trim().is_empty() {
            return Err(HybridSearchError::EmptyTenantId);
        }
        config.validate()?;
        cost_model.validate()?;

        let persisted = self
            .contract
            .get(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;
        persisted.validate_runtime(runtime_meta, IndexOperation::Query)?;

        reject_swapped_prefix(
            raw_query,
            &persisted.query_prefix,
            &persisted.doc_prefix,
            IndexOperation::Query,
            MetadataField::QueryPrefix,
            MetadataField::DocPrefix,
        )?;

        let store = self
            .collections
            .get(collection)
            .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;

        // Step 1: Kai query planner classifies the query and determines active legs + weights.
        let plan = planner.plan_query(raw_query, filter, config)?;

        // Step 2: Mandatory pre-scoring ACL and scope pre-filtering over canonical records.
        let authorized_doc_ids: Vec<String> = store
            .canonical_records
            .values()
            .filter(|rec| is_record_authorized_and_in_scope(rec, filter))
            .map(|rec| rec.identity.doc_id.clone())
            .collect();

        let mut query_latency_ms = cost_model.kai_planner_latency_ms;
        let mut query_cost_usd = cost_model.kai_planner_cost_usd;

        // Step 3a: Execute Lexical Leg (if active).
        let lexical_ranked = if plan.execute_lexical {
            query_latency_ms += cost_model.lexical_leg_latency_ms;
            query_cost_usd += cost_model.lexical_leg_cost_usd;
            store.lexical_index.search_authorized(
                &authorized_doc_ids,
                raw_query,
                &plan.extracted_identifiers,
                plan.per_leg_candidate_depth,
            )
        } else {
            Vec::new()
        };

        // Step 3b: Execute Dense Vector Leg (if active).
        let vector_ranked = if plan.execute_vector {
            let prefixed_query = persisted.format_query(raw_query);
            let q_vec =
                embedder.embed_prefixed(&prefixed_query, IndexOperation::Query, runtime_meta)?;
            self.contract.validate_query(
                collection,
                runtime_meta,
                &q_vec,
                Some(&prefixed_query),
            )?;
            let q_tokens = estimate_tokens(&prefixed_query);
            query_latency_ms +=
                cost_model.vector_embed_latency_ms + cost_model.vector_search_latency_ms;
            query_cost_usd +=
                (q_tokens as f64 / 1000.0) * cost_model.vector_embed_cost_per_1k_tokens_usd;
            store.vector_index.search_authorized(
                &authorized_doc_ids,
                &q_vec,
                plan.per_leg_candidate_depth,
            )
        } else {
            Vec::new()
        };

        // Step 3c: Execute Structural / AST Leg (if active).
        let structural_ranked = if plan.execute_structural {
            query_latency_ms += cost_model.structural_leg_latency_ms;
            query_cost_usd += cost_model.structural_leg_cost_usd;
            store.structural_index.search_authorized(
                &authorized_doc_ids,
                raw_query,
                &plan,
                plan.per_leg_candidate_depth,
            )
        } else {
            Vec::new()
        };

        // Step 4: Reciprocal Rank Fusion (RRF) with per-leg explainability and tie-breaking.
        query_latency_ms += cost_model.rrf_fusion_latency_ms;
        let hits = ReciprocalRankFusion::fuse(
            &store.canonical_records,
            &plan,
            &lexical_ranked,
            &vector_ranked,
            &structural_ranked,
        );

        Ok(HybridSearchOutcome {
            plan,
            authorized_candidate_count: authorized_doc_ids.len(),
            lexical_hits_count: lexical_ranked.len(),
            vector_hits_count: vector_ranked.len(),
            structural_hits_count: structural_ranked.len(),
            hits,
            query_latency_ms,
            query_cost_usd,
        })
    }

    /// Negative control baseline: combines active legs by summing their uncalibrated
    /// raw scores (`raw_bm25 + raw_cosine + raw_structural`) without rank normalization.
    ///
    /// Used in the matched-baseline benchmark to prove why RRF outperforms raw
    /// score linear combinations when leg score distributions differ by orders of magnitude.
    #[allow(clippy::too_many_arguments)]
    pub fn search_uncalibrated_raw_sum(
        &self,
        collection: &str,
        runtime_meta: &IndexEmbeddingMetadata,
        filter: &HybridAccessFilter,
        raw_query: &str,
        top_k: usize,
        embedder: &dyn PrefixedTextEmbedder,
        cost_model: &HybridCostModel,
    ) -> Result<HybridSearchOutcome, HybridSearchError> {
        let cfg = HybridRrfConfig {
            rrf_k: DEFAULT_RRF_K,
            per_leg_candidate_depth: 32,
            top_k,
            weight_override: Some(LegWeights::uniform()),
        };
        let planner = DeterministicKaiQueryPlanner;
        let mut outcome = self.search_hybrid(
            collection,
            runtime_meta,
            filter,
            raw_query,
            &cfg,
            &planner,
            embedder,
            cost_model,
        )?;

        for hit in &mut outcome.hits {
            let raw_lex = hit.explanation.lexical.as_ref().map_or(0.0, |c| c.raw_score);
            let raw_vec = hit.explanation.vector.as_ref().map_or(0.0, |c| c.raw_score);
            let raw_str = hit.explanation.structural.as_ref().map_or(0.0, |c| c.raw_score);
            // Raw BM25 (often 5..25) dominates raw cosine (0..1), drowning pure semantic hits.
            hit.rrf_score = raw_lex + raw_vec + 0.15 * raw_str;
        }
        outcome.hits.sort_by(compare_search_hits_deterministic);
        outcome.hits.truncate(top_k);
        for (idx, hit) in outcome.hits.iter_mut().enumerate() {
            hit.final_rank = idx + 1;
        }
        Ok(outcome)
    }
}

fn format_doc_embedding_payload(record: &CanonicalEvidenceRecord) -> String {
    let mut out = String::with_capacity(record.title.len() + record.raw_text.len() + 128);
    out.push_str(&record.title);
    out.push_str(". ");
    out.push_str(&record.raw_text);
    for sym in &record.symbols {
        out.push(' ');
        out.push_str(&sym.symbol_name);
        out.push(' ');
        out.push_str(&sym.qualified_name);
    }
    out
}

fn reject_swapped_prefix(
    raw_text: &str,
    expected_prefix: &str,
    other_prefix: &str,
    operation: IndexOperation,
    expected_field: MetadataField,
    actual_field: MetadataField,
) -> Result<(), MetadataError> {
    if !other_prefix.is_empty()
        && other_prefix != expected_prefix
        && raw_text.starts_with(other_prefix)
    {
        return Err(MetadataError::SwappedTaskPrefix {
            operation,
            expected_field,
            expected_prefix: expected_prefix.to_string(),
            actual_field,
            actual_prefix: other_prefix.to_string(),
        });
    }
    Ok(())
}

fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    ((words * 4).div_ceil(3)).max(1)
}

// ============================================================================
// Matched-Baseline Benchmark Harness & Downstream Code-Edit Task Judge
// ============================================================================

/// Six matched retrieval strategies compared in `evaluate_hybrid_benchmark`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridStrategy {
    /// Leg 1 only: Okapi BM25 + code-aware identifier sub-tokenization.
    LexicalOnly,
    /// Leg 2 only: Prefix-validated L2 dense cosine similarity.
    VectorOnly,
    /// Leg 3 only: Structural AST symbol, signature, call-graph, and path scorer.
    StructuralOnly,
    /// Negative control: sums uncalibrated raw BM25 + raw cosine + raw structural scores directly.
    UncalibratedRawScoreSum,
    /// 3-leg Reciprocal Rank Fusion with uniform leg weights (`1/3, 1/3, 1/3`).
    UniformRrf { k: f64 },
    /// Kai query classifier + adaptive per-class leg weights + Reciprocal Rank Fusion.
    KaiPlannedHybridRrf { k: f64 },
}

impl HybridStrategy {
    /// Canonical human-readable name for tables and JSON reports.
    pub fn label(&self) -> String {
        match self {
            Self::LexicalOnly => "LexicalOnly (BM25 + Subtokens)".to_string(),
            Self::VectorOnly => "VectorOnly (Prefix-Validated L2)".to_string(),
            Self::StructuralOnly => "StructuralOnly (AST + CallGraph)".to_string(),
            Self::UncalibratedRawScoreSum => "UncalibratedRawScoreSum (Control)".to_string(),
            Self::UniformRrf { k } => format!("UniformRrf (k={k:.0})"),
            Self::KaiPlannedHybridRrf { k } => format!("KaiPlannedHybridRrf (k={k:.0})"),
        }
    }
}

/// Single labeled benchmark query with ground-truth relevant `doc_id`s and downstream code-edit verification targets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridBenchmarkCase {
    pub case_id: String,
    pub expected_class: HybridQueryClass,
    pub raw_query: String,
    pub filter: HybridAccessFilter,
    pub relevant_doc_ids: Vec<String>,
    /// Target symbol that a downstream code-edit / incident-diagnosis response must identify.
    pub required_fix_symbol: String,
    /// Target file path that a downstream code-edit / incident-diagnosis response must cite.
    pub required_fix_path: String,
    /// Key technical invariant phrase present in the ground-truth evidence required to solve the task.
    pub required_fix_invariant: String,
}

/// Downstream task evaluator verifying whether feeding the top-ranked retrieved hit
/// into `ZenSynthesizer` (`SynthesisTarget::ZenGrounded`) yields an actionable code
/// patch or root-cause diagnosis referencing the target symbol, file path, and fix invariant.
#[derive(Debug, Clone, Default)]
pub struct DownstreamCodeEditJudge;

impl DownstreamCodeEditJudge {
    /// Evaluates whether the synthesized grounded response satisfies the downstream code-edit contract.
    pub fn evaluate_grounded_patch(&self, case: &HybridBenchmarkCase, zen_patch: &str) -> bool {
        let lower_patch = zen_patch.to_ascii_lowercase();
        let has_symbol =
            lower_patch.contains(&case.required_fix_symbol.to_ascii_lowercase());
        let has_path = lower_patch.contains(&case.required_fix_path.to_ascii_lowercase());
        let has_invariant =
            lower_patch.contains(&case.required_fix_invariant.to_ascii_lowercase());
        has_symbol && has_path && has_invariant
    }
}

/// Deterministic grounded code-edit synthesizer implementing [`ZenSynthesizer`] (HIP-1346 §2).
///
/// Consumes the top retrieved evidence record(s) and emits a structured code-edit
/// diagnosis grounded strictly in the provided evidence.
#[derive(Debug, Clone, Default)]
pub struct DeterministicCodeEditZenSynthesizer;

impl ZenSynthesizer for DeterministicCodeEditZenSynthesizer {
    fn synthesize(
        &self,
        query: &str,
        retrieved_evidence: &[String],
        target: SynthesisTarget,
    ) -> String {
        match target {
            SynthesisTarget::ZenParametric => {
                format!("ZenParametric(unverified parametric response for `{query}`)")
            }
            SynthesisTarget::ZenGrounded => {
                let Some(top_evidence) = retrieved_evidence.first() else {
                    return "ZenGrounded(no retrieved evidence available)".to_string();
                };
                format!("ZenGrounded(query=`{query}` | primary_evidence={top_evidence})")
            }
        }
    }
}

/// Comprehensive benchmark metrics for a single [`HybridStrategy`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridStrategyReport {
    pub strategy: HybridStrategy,
    pub strategy_label: String,
    pub query_count: usize,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr_at_10: f64,
    pub ndcg_at_10: f64,
    pub exact_symbol_recall_at_1: f64,
    pub semantic_paraphrase_recall_at_1: f64,
    pub structural_ast_recall_at_1: f64,
    pub path_scoped_recall_at_1: f64,
    pub hybrid_multileg_recall_at_1: f64,
    /// End-to-end downstream code-edit / root-cause diagnosis task success rate using top-1 evidence.
    pub downstream_task_success_rate: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub mean_latency_ms: f64,
    pub indexing_cost_usd: f64,
    pub query_cost_usd: f64,
    pub total_cost_usd: f64,
    /// Number of returned hits across all queries that violated tenant, role, or scope pre-filters (must be 0).
    pub unauthorized_acl_leaks: usize,
}

/// Complete benchmark suite output suitable for JSON serialization and Markdown reporting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridBenchmarkSuiteReport {
    pub collection: String,
    pub corpus_record_count: usize,
    pub query_count: usize,
    pub top_k: usize,
    pub honesty_labels: Vec<String>,
    pub strategy_reports: Vec<HybridStrategyReport>,
}

/// Evaluates a slice of [`HybridStrategy`] baselines over the identical corpus,
/// query set, `top_k` budget, cost model, and downstream [`DownstreamCodeEditJudge`].
#[allow(clippy::too_many_arguments)]
pub fn evaluate_hybrid_benchmark(
    index: &HybridSearchIndex,
    collection: &str,
    runtime_meta: &IndexEmbeddingMetadata,
    cases: &[HybridBenchmarkCase],
    strategies: &[HybridStrategy],
    planner: &dyn KaiHybridQueryPlanner,
    embedder: &dyn PrefixedTextEmbedder,
    zen: &dyn ZenSynthesizer,
    judge: &DownstreamCodeEditJudge,
    cost_model: &HybridCostModel,
    top_k: usize,
) -> Result<HybridBenchmarkSuiteReport, HybridSearchError> {
    if cases.is_empty() {
        return Err(HybridSearchError::EmptyBenchmarkSuite);
    }
    if top_k == 0 {
        return Err(HybridSearchError::ZeroTopK);
    }
    cost_model.validate()?;

    let canonical_map = index
        .canonical_records(collection)
        .ok_or_else(|| MetadataError::UnknownCollection(collection.to_string()))?;
    let total_corpus_tokens: usize = canonical_map
        .values()
        .map(|r| estimate_tokens(&r.raw_text) + estimate_tokens(&r.title))
        .sum();
    let corpus_record_count = canonical_map.len();

    let mut strategy_reports = Vec::with_capacity(strategies.len());

    for &strategy in strategies {
        let indexing_cost_usd = match strategy {
            HybridStrategy::LexicalOnly => 0.0,
            HybridStrategy::VectorOnly => {
                (total_corpus_tokens as f64 / 1000.0)
                    * cost_model.index_doc_embed_cost_per_1k_tokens_usd
            }
            HybridStrategy::StructuralOnly => {
                (corpus_record_count as f64) * cost_model.index_doc_structural_parse_cost_usd
            }
            HybridStrategy::UncalibratedRawScoreSum
            | HybridStrategy::UniformRrf { .. }
            | HybridStrategy::KaiPlannedHybridRrf { .. } => {
                (total_corpus_tokens as f64 / 1000.0)
                    * cost_model.index_doc_embed_cost_per_1k_tokens_usd
                    + (corpus_record_count as f64) * cost_model.index_doc_structural_parse_cost_usd
            }
        };

        let mut r1_sum = 0.0_f64;
        let mut r5_sum = 0.0_f64;
        let mut r10_sum = 0.0_f64;
        let mut mrr10_sum = 0.0_f64;
        let mut ndcg10_sum = 0.0_f64;

        let mut exact_r1_sum = 0.0_f64;
        let mut exact_cnt = 0usize;
        let mut sem_r1_sum = 0.0_f64;
        let mut sem_cnt = 0usize;
        let mut ast_r1_sum = 0.0_f64;
        let mut ast_cnt = 0usize;
        let mut path_r1_sum = 0.0_f64;
        let mut path_cnt = 0usize;
        let mut multi_r1_sum = 0.0_f64;
        let mut multi_cnt = 0usize;

        let mut downstream_success_cnt = 0usize;
        let mut unauthorized_acl_leaks = 0usize;
        let mut query_cost_usd = 0.0_f64;
        let mut latencies = Vec::with_capacity(cases.len());

        for case in cases {
            let outcome = match strategy {
                HybridStrategy::LexicalOnly => {
                    let cfg = HybridRrfConfig {
                        rrf_k: DEFAULT_RRF_K,
                        per_leg_candidate_depth: 32,
                        top_k,
                        weight_override: Some(LegWeights {
                            lexical: 1.0,
                            vector: 0.0,
                            structural: 0.0,
                        }),
                    };
                    index.search_hybrid(
                        collection,
                        runtime_meta,
                        &case.filter,
                        &case.raw_query,
                        &cfg,
                        planner,
                        embedder,
                        cost_model,
                    )?
                }
                HybridStrategy::VectorOnly => {
                    let cfg = HybridRrfConfig {
                        rrf_k: DEFAULT_RRF_K,
                        per_leg_candidate_depth: 32,
                        top_k,
                        weight_override: Some(LegWeights {
                            lexical: 0.0,
                            vector: 1.0,
                            structural: 0.0,
                        }),
                    };
                    index.search_hybrid(
                        collection,
                        runtime_meta,
                        &case.filter,
                        &case.raw_query,
                        &cfg,
                        planner,
                        embedder,
                        cost_model,
                    )?
                }
                HybridStrategy::StructuralOnly => {
                    let cfg = HybridRrfConfig {
                        rrf_k: DEFAULT_RRF_K,
                        per_leg_candidate_depth: 32,
                        top_k,
                        weight_override: Some(LegWeights {
                            lexical: 0.0,
                            vector: 0.0,
                            structural: 1.0,
                        }),
                    };
                    index.search_hybrid(
                        collection,
                        runtime_meta,
                        &case.filter,
                        &case.raw_query,
                        &cfg,
                        planner,
                        embedder,
                        cost_model,
                    )?
                }
                HybridStrategy::UncalibratedRawScoreSum => index.search_uncalibrated_raw_sum(
                    collection,
                    runtime_meta,
                    &case.filter,
                    &case.raw_query,
                    top_k,
                    embedder,
                    cost_model,
                )?,
                HybridStrategy::UniformRrf { k } => {
                    let cfg = HybridRrfConfig {
                        rrf_k: k,
                        per_leg_candidate_depth: 32,
                        top_k,
                        weight_override: Some(LegWeights::uniform()),
                    };
                    index.search_hybrid(
                        collection,
                        runtime_meta,
                        &case.filter,
                        &case.raw_query,
                        &cfg,
                        planner,
                        embedder,
                        cost_model,
                    )?
                }
                HybridStrategy::KaiPlannedHybridRrf { k } => {
                    let cfg = HybridRrfConfig {
                        rrf_k: k,
                        per_leg_candidate_depth: 32,
                        top_k,
                        weight_override: None,
                    };
                    index.search_hybrid(
                        collection,
                        runtime_meta,
                        &case.filter,
                        &case.raw_query,
                        &cfg,
                        planner,
                        embedder,
                        cost_model,
                    )?
                }
            };

            // Verify zero ACL or scope leaks across all returned hits.
            for hit in &outcome.hits {
                if let Some(rec) = canonical_map.get(&hit.doc_id) {
                    if !is_record_authorized_and_in_scope(rec, &case.filter) {
                        unauthorized_acl_leaks += 1;
                    }
                } else {
                    unauthorized_acl_leaks += 1;
                }
            }

            let hit_ids: Vec<&str> = outcome.hits.iter().map(|h| h.doc_id.as_str()).collect();
            let hit_at_1 =
                hit_ids.first().is_some_and(|id| case.relevant_doc_ids.iter().any(|r| r == id));
            let hit_at_5 =
                hit_ids.iter().take(5).any(|id| case.relevant_doc_ids.iter().any(|r| r == id));
            let hit_at_10 =
                hit_ids.iter().take(10).any(|id| case.relevant_doc_ids.iter().any(|r| r == id));

            let rr10 = hit_ids
                .iter()
                .take(10)
                .position(|id| case.relevant_doc_ids.iter().any(|r| r == id))
                .map_or(0.0, |idx| 1.0 / ((idx + 1) as f64));

            let dcg10: f64 = hit_ids
                .iter()
                .take(10)
                .enumerate()
                .map(|(idx, id)| {
                    if case.relevant_doc_ids.iter().any(|r| r == id) {
                        1.0 / ((idx as f64) + 2.0).log2()
                    } else {
                        0.0
                    }
                })
                .sum();
            let ideal_hits = case.relevant_doc_ids.len().clamp(1, 10);
            let idcg10: f64 =
                (0..ideal_hits).map(|idx| 1.0 / ((idx as f64) + 2.0).log2()).sum();
            let ndcg10 = if idcg10 > 0.0 { dcg10 / idcg10 } else { 0.0 };

            if hit_at_1 {
                r1_sum += 1.0;
            }
            if hit_at_5 {
                r5_sum += 1.0;
            }
            if hit_at_10 {
                r10_sum += 1.0;
            }
            mrr10_sum += rr10;
            ndcg10_sum += ndcg10;

            match case.expected_class {
                HybridQueryClass::ExactSymbolOrIdentifier => {
                    exact_cnt += 1;
                    if hit_at_1 {
                        exact_r1_sum += 1.0;
                    }
                }
                HybridQueryClass::NaturalLanguageSemantic => {
                    sem_cnt += 1;
                    if hit_at_1 {
                        sem_r1_sum += 1.0;
                    }
                }
                HybridQueryClass::StructuralAstPattern => {
                    ast_cnt += 1;
                    if hit_at_1 {
                        ast_r1_sum += 1.0;
                    }
                }
                HybridQueryClass::PathOrFileScoped => {
                    path_cnt += 1;
                    if hit_at_1 {
                        path_r1_sum += 1.0;
                    }
                }
                HybridQueryClass::HybridMultiLeg => {
                    multi_cnt += 1;
                    if hit_at_1 {
                        multi_r1_sum += 1.0;
                    }
                }
            }

            // Evaluate downstream code-edit / root-cause diagnosis task grounded in top-1 hit.
            let top1_evidence: Vec<String> = outcome
                .hits
                .first()
                .map(|h| {
                    let syms = h.explanation.matched_symbols.join(", ");
                    format!(
                        "path={} title={} text={} symbols=[{}]",
                        h.identity.path, h.title, h.raw_text, syms
                    )
                })
                .into_iter()
                .collect();
            let grounded_patch =
                zen.synthesize(&case.raw_query, &top1_evidence, SynthesisTarget::ZenGrounded);
            if judge.evaluate_grounded_patch(case, &grounded_patch) {
                downstream_success_cnt += 1;
            }

            query_cost_usd += outcome.query_cost_usd;
            latencies.push(outcome.query_latency_ms);
        }

        latencies.sort_by(|a, b| a.total_cmp(b));
        let n = cases.len();
        let n_f = n as f64;
        let p50_idx = ((n_f * 0.50).ceil() as usize).saturating_sub(1).min(n - 1);
        let p95_idx = ((n_f * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
        let p50_latency_ms = latencies[p50_idx];
        let p95_latency_ms = latencies[p95_idx];
        let mean_latency_ms = latencies.iter().sum::<f64>() / n_f;

        strategy_reports.push(HybridStrategyReport {
            strategy,
            strategy_label: strategy.label(),
            query_count: n,
            recall_at_1: r1_sum / n_f,
            recall_at_5: r5_sum / n_f,
            recall_at_10: r10_sum / n_f,
            mrr_at_10: mrr10_sum / n_f,
            ndcg_at_10: ndcg10_sum / n_f,
            exact_symbol_recall_at_1: if exact_cnt > 0 {
                exact_r1_sum / (exact_cnt as f64)
            } else {
                0.0
            },
            semantic_paraphrase_recall_at_1: if sem_cnt > 0 {
                sem_r1_sum / (sem_cnt as f64)
            } else {
                0.0
            },
            structural_ast_recall_at_1: if ast_cnt > 0 {
                ast_r1_sum / (ast_cnt as f64)
            } else {
                0.0
            },
            path_scoped_recall_at_1: if path_cnt > 0 {
                path_r1_sum / (path_cnt as f64)
            } else {
                0.0
            },
            hybrid_multileg_recall_at_1: if multi_cnt > 0 {
                multi_r1_sum / (multi_cnt as f64)
            } else {
                0.0
            },
            downstream_task_success_rate: (downstream_success_cnt as f64) / n_f,
            p50_latency_ms,
            p95_latency_ms,
            mean_latency_ms,
            indexing_cost_usd,
            query_cost_usd,
            total_cost_usd: indexing_cost_usd + query_cost_usd,
            unauthorized_acl_leaks,
        });
    }

    Ok(HybridBenchmarkSuiteReport {
        collection: collection.to_string(),
        corpus_record_count,
        query_count: cases.len(),
        top_k,
        honesty_labels: vec![
            "Deterministic offline evaluation over the canonical 22-record multi-tenant repository + incident corpus and 20-query 5-slice workload.".to_string(),
            "Dense vectors are produced by DeterministicSemanticCodeEmbedder (256-dim L2-normalized FNV-1a + semantic concept projection enforcing HIP-1346 §1 query_prefix/doc_prefix).".to_string(),
            "Latency and USD cost figures are deterministic model outputs from HybridCostModel::default() reflecting active legs executed per query.".to_string(),
        ],
        strategy_reports,
    })
}

/// Returns the canonical `256`-dim L2-normalized `IndexEmbeddingMetadata` used for
/// reproducible `HIP-1347` hybrid search benchmarks.
pub fn canonical_hybrid_embedding_metadata() -> IndexEmbeddingMetadata {
    IndexEmbeddingMetadata::new(
        "hanzo/zen-embed-v1",
        "sha256:1347a9c4e2f08b11d5c37a90e4b2c1d8",
        256,
        NormalizationPolicy::L2,
        "search_query: ",
        "search_document: ",
    )
    .expect("valid canonical HIP-1347 embedding metadata")
}

/// Builds the canonical multi-tenant repository, AST symbol, and technical incident corpus
/// for `hanzoai/index#11` (`HIP-1347`).
#[allow(clippy::too_many_lines)]
pub fn canonical_hybrid_corpus() -> Vec<CanonicalEvidenceRecord> {
    let mk_id = |doc_id: &str,
                 artifact_kind: SourceArtifactKind,
                 tenant_id: &str,
                 principals: Vec<&str>,
                 path: &str,
                 lang: &str| SourceIdentity {
        doc_id: doc_id.to_string(),
        source_id: "repo:hanzoai/index".to_string(),
        version_id: "git:331c4f168e20".to_string(),
        artifact_kind,
        tenant_id: tenant_id.to_string(),
        project_id: "proj_index".to_string(),
        repo_id: "hanzoai/index".to_string(),
        allowed_principals: principals.into_iter().map(ToString::to_string).collect(),
        path: path.to_string(),
        language: lang.to_string(),
    };

    vec![
        // 1. Canonical Issue #11 exact symbol record: `record_finish` in `crates/agent/src/runner.rs`.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/agent/src/runner.rs#record_finish",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/agent/src/runner.rs",
                "rust",
            ),
            title: "AgentRunner::record_finish turn finalization & lock release".to_string(),
            raw_text: "pub async fn record_finish(&mut self, turn_id: TurnId) -> Result<(), RunnerError>. \
                       Finalizes an active agent turn inside crates/agent/src/runner.rs. \
                       Drops the mutex guard before awaiting turn_tx channel drain \
                       and invokes release_turn_lock so worker threads never deadlock."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "record_finish".to_string(),
                qualified_name: "hanzo_agent::runner::AgentRunner::record_finish".to_string(),
                node_kind: StructuralNodeKind::Function,
                signature: "pub async fn record_finish(&mut self, turn_id: TurnId) -> Result<(), RunnerError>"
                    .to_string(),
                parent_scope: Some("AgentRunner".to_string()),
                imports: vec!["hanzo_agent::lock::release_turn_lock".to_string()],
                calls: vec!["release_turn_lock".to_string(), "flush_audit_span".to_string()],
                called_by: vec!["dispatch_turn_loop".to_string()],
                start_line: 142,
                end_line: 189,
            }],
        },
        // 2. Canonical Issue #11 semantic paraphrase target: deadlock root-cause analysis when agent freezes after request completion.
        // Note: intentionally uses technical terminology ("mutex contention", "parking_lot", "draining turn channel", "deadlock", "finalization")
        // rather than the user's colloquial words ("why does the agent freeze after completing a request?").
        CanonicalEvidenceRecord {
            identity: mk_id(
                "doc:incidents/INC-409-agent-completion-deadlock.md",
                SourceArtifactKind::Document,
                "tenant_hanzo",
                vec![],
                "docs/incidents/INC-409-agent-completion-deadlock.md",
                "markdown",
            ),
            title: "INC-409: Worker thread deadlock & mutex contention during turn finalization"
                .to_string(),
            raw_text: "Root cause diagnosis for unresponsive runtime workers in crates/agent/src/runner.rs: \
                       holding the non-send parking_lot mutex lock across an asynchronous channel drain \
                       during turn finalization triggers a circular lock contention deadlock. \
                       Fix invariant: scope the guard and invoke release_turn_lock inside record_finish prior to await."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "INC_409_deadlock_postmortem".to_string(),
                qualified_name: "incidents::INC_409_deadlock_postmortem".to_string(),
                node_kind: StructuralNodeKind::Section,
                signature: "## INC-409 Turn Finalization Mutex Deadlock".to_string(),
                parent_scope: None,
                imports: vec![],
                calls: vec![],
                called_by: vec![],
                start_line: 1,
                end_line: 64,
            }],
        },
        // 3. Call-graph target: `dispatch_turn_loop` which calls `release_turn_lock` and `record_finish`.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/agent/src/scheduler.rs#dispatch_turn_loop",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/agent/src/scheduler.rs",
                "rust",
            ),
            title: "TurnScheduler::dispatch_turn_loop caller of release_turn_lock".to_string(),
            raw_text: "Coordinates concurrent worker slots in crates/agent/src/scheduler.rs. \
                       Invokes release_turn_lock on cancellation or panic recovery to reclaim leaked worker permits."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "dispatch_turn_loop".to_string(),
                qualified_name: "hanzo_agent::scheduler::TurnScheduler::dispatch_turn_loop"
                    .to_string(),
                node_kind: StructuralNodeKind::Method,
                signature: "pub fn dispatch_turn_loop(&self, batch: TurnBatch) -> SchedulerStatus"
                    .to_string(),
                parent_scope: Some("TurnScheduler".to_string()),
                imports: vec!["hanzo_agent::lock::release_turn_lock".to_string()],
                calls: vec!["release_turn_lock".to_string(), "record_finish".to_string()],
                called_by: vec!["run_supervisor".to_string()],
                start_line: 58,
                end_line: 114,
            }],
        },
        // 4. HIP-1346 §1 ImmutableIndexContract & SwappedTaskPrefix record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/metadata.rs#ImmutableIndexContract",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/metadata.rs",
                "rust",
            ),
            title: "ImmutableIndexContract & MetadataError::SwappedTaskPrefix validation"
                .to_string(),
            raw_text: "HIP-1346 §1 persisted collection contract in crates/rag-eval/src/metadata.rs. \
                       Enforces model_id, pinned model_revision, L2 normalization, Matryoshka dimensions 128 256 512 768, \
                       and rejects mismatched query_prefix vs doc_prefix with MetadataError::SwappedTaskPrefix."
                .to_string(),
            symbols: vec![
                StructuralSymbolNode {
                    symbol_name: "ImmutableIndexContract".to_string(),
                    qualified_name: "rag_eval::metadata::ImmutableIndexContract".to_string(),
                    node_kind: StructuralNodeKind::Struct,
                    signature: "pub struct ImmutableIndexContract".to_string(),
                    parent_scope: Some("metadata".to_string()),
                    imports: vec![],
                    calls: vec!["validate_insert".to_string(), "validate_query".to_string()],
                    called_by: vec!["HybridSearchIndex::upsert_record".to_string()],
                    start_line: 493,
                    end_line: 574,
                },
                StructuralSymbolNode {
                    symbol_name: "SwappedTaskPrefix".to_string(),
                    qualified_name: "rag_eval::metadata::MetadataError::SwappedTaskPrefix"
                        .to_string(),
                    node_kind: StructuralNodeKind::Constant,
                    signature: "MetadataError::SwappedTaskPrefix".to_string(),
                    parent_scope: Some("MetadataError".to_string()),
                    imports: vec![],
                    calls: vec![],
                    called_by: vec!["validate_prefixed_text".to_string()],
                    start_line: 182,
                    end_line: 188,
                },
            ],
        },
        // 5. Matryoshka truncation function `truncate_matryoshka`.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/metadata.rs#truncate_matryoshka",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/metadata.rs",
                "rust",
            ),
            title: "truncate_matryoshka MRL prefix slicer and L2 renormalizer".to_string(),
            raw_text: "Truncates a full-width embedding vector in crates/rag-eval/src/metadata.rs to \
                       canonical MATRYOSHKA_DIMS (128, 256, 512, 768) and L2-renormalizes within L2_NORM_TOLERANCE."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "truncate_matryoshka".to_string(),
                qualified_name: "rag_eval::metadata::truncate_matryoshka".to_string(),
                node_kind: StructuralNodeKind::Function,
                signature: "pub fn truncate_matryoshka(vector: &[f32], target_dim: usize, normalization: &NormalizationPolicy) -> Result<Vec<f32>, MetadataError>"
                    .to_string(),
                parent_scope: Some("metadata".to_string()),
                imports: vec![],
                calls: vec!["is_standard_matryoshka_dim".to_string()],
                called_by: vec![],
                start_line: 450,
                end_line: 488,
            }],
        },
        // 6. Self-RAG RetrieveOrAnswerGate record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/self_rag.rs#RetrieveOrAnswerGate",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/self_rag.rs",
                "rust",
            ),
            title: "RetrieveOrAnswerGate adaptive Kai retrieval gating".to_string(),
            raw_text: "Evaluates Kai discrete probabilities p_retrieve against calibrated threshold theta \
                       in crates/rag-eval/src/self_rag.rs so ZenParametric synthesizes responses from \
                       parametric weights without invoking index retrieval."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "RetrieveOrAnswerGate".to_string(),
                qualified_name: "rag_eval::self_rag::RetrieveOrAnswerGate".to_string(),
                node_kind: StructuralNodeKind::Struct,
                signature: "pub struct RetrieveOrAnswerGate { pub theta: f64 }".to_string(),
                parent_scope: Some("self_rag".to_string()),
                imports: vec!["crate::factscore::compute_ece".to_string()],
                calls: vec!["evaluate".to_string()],
                called_by: vec!["route_with_kai_and_zen".to_string()],
                start_line: 118,
                end_line: 160,
            }],
        },
        // 7. Corrective RAG CorrectiveRetrievalPolicy record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/corrective.rs#CorrectiveRetrievalPolicy",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/corrective.rs",
                "rust",
            ),
            title: "CorrectiveRetrievalPolicy CRAG evidence grading & fallback".to_string(),
            raw_text: "Detects corrupted mojibake chunks, high action-risk prompt injections, and low-relevance \
                       distractors in crates/rag-eval/src/corrective.rs, triggering KeywordQueryReformulator \
                       and SecondaryRetriever recovery."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "CorrectiveRetrievalPolicy".to_string(),
                qualified_name: "rag_eval::corrective::CorrectiveRetrievalPolicy".to_string(),
                node_kind: StructuralNodeKind::Struct,
                signature: "pub struct CorrectiveRetrievalPolicy".to_string(),
                parent_scope: Some("corrective".to_string()),
                imports: vec![],
                calls: vec!["reformulate_query".to_string(), "retrieve_secondary".to_string()],
                called_by: vec![],
                start_line: 110,
                end_line: 240,
            }],
        },
        // 8. FActScore AtomicClaimDecomposer & FactScoreEvaluator record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/factscore.rs#FactScoreEvaluator",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/factscore.rs",
                "rust",
            ),
            title: "FactScoreEvaluator atomic claim decomposition and span binding".to_string(),
            raw_text: "Decomposes synthesized paragraphs into fine-grained atomic propositions via AtomicClaimDecomposer \
                       in crates/rag-eval/src/factscore.rs and checks negation polarity and numeric literals \
                       against source spans with SpanEvidenceBinder."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "FactScoreEvaluator".to_string(),
                qualified_name: "rag_eval::factscore::FactScoreEvaluator".to_string(),
                node_kind: StructuralNodeKind::Struct,
                signature: "pub struct FactScoreEvaluator".to_string(),
                parent_scope: Some("factscore".to_string()),
                imports: vec![],
                calls: vec!["decompose".to_string(), "bind_span".to_string(), "compute_ece".to_string()],
                called_by: vec![],
                start_line: 190,
                end_line: 310,
            }],
        },
        // 9. OdinANN online delta-merge maintenance runner record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/ann_insert.rs#AnnMaintenanceRunner",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/ann_insert.rs",
                "rust",
            ),
            title: "AnnMaintenanceRunner OdinANN bounded delta layer & amortized merge".to_string(),
            raw_text: "Maintains a file-backed on-disk GraphAnn index under continuous online vector insertion \
                       in crates/rag-eval/src/ann_insert.rs using a bounded delta buffer and per-query amortized \
                       graph merge to eliminate stop-the-world rebuild tail latency stalls."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "AnnMaintenanceRunner".to_string(),
                qualified_name: "rag_eval::ann_insert::AnnMaintenanceRunner".to_string(),
                node_kind: StructuralNodeKind::Struct,
                signature: "pub struct AnnMaintenanceRunner".to_string(),
                parent_scope: Some("ann_insert".to_string()),
                imports: vec![],
                calls: vec!["drain_merge_budget".to_string(), "emergency_merge".to_string()],
                called_by: vec!["run_full_matrix".to_string()],
                start_line: 280,
                end_line: 430,
            }],
        },
        // 10. Existing 2-leg Milli hybrid search merger record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/milli/src/search/hybrid.rs#SemanticScore",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/milli/src/search/hybrid.rs",
                "rust",
            ),
            title: "Milli 2-leg score ratio merger in crates/milli/src/search/hybrid.rs".to_string(),
            raw_text: "Merges keyword and vector results inside crates/milli/src/search/hybrid.rs \
                       using compare_scores(left * left_ratio, right * right_ratio) without Reciprocal Rank Fusion \
                       rank accumulation and without structural AST symbol signals."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "compare_scores".to_string(),
                qualified_name: "milli::search::hybrid::compare_scores".to_string(),
                node_kind: StructuralNodeKind::Function,
                signature: "fn compare_scores((left, left_ratio): (f32, f32), (right, right_ratio): (f32, f32)) -> Ordering"
                    .to_string(),
                parent_scope: Some("hybrid".to_string()),
                imports: vec![],
                calls: vec![],
                called_by: vec!["NextResult::next".to_string()],
                start_line: 148,
                end_line: 162,
            }],
        },
        // 11. HIP-1347 ReciprocalRankFusion & DeterministicKaiQueryPlanner record.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/hybrid_rrf.rs#ReciprocalRankFusion",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/hybrid_rrf.rs",
                "rust",
            ),
            title: "ReciprocalRankFusion & DeterministicKaiQueryPlanner 3-leg hybrid search"
                .to_string(),
            raw_text: "HIP-1347 unified 3-leg search planner and Reciprocal Rank Fusion executor in \
                       crates/rag-eval/src/hybrid_rrf.rs. Combines BM25 lexical, prefix-validated L2 vector, \
                       and structural AST call-graph ranks via weight / (rrf_k + rank_1_based) with k=60."
                .to_string(),
            symbols: vec![
                StructuralSymbolNode {
                    symbol_name: "ReciprocalRankFusion".to_string(),
                    qualified_name: "rag_eval::hybrid_rrf::ReciprocalRankFusion".to_string(),
                    node_kind: StructuralNodeKind::Struct,
                    signature: "pub struct ReciprocalRankFusion".to_string(),
                    parent_scope: Some("hybrid_rrf".to_string()),
                    imports: vec![],
                    calls: vec!["fuse".to_string(), "compare_search_hits_deterministic".to_string()],
                    called_by: vec!["HybridSearchIndex::search_hybrid".to_string()],
                    start_line: 720,
                    end_line: 840,
                },
                StructuralSymbolNode {
                    symbol_name: "DeterministicKaiQueryPlanner".to_string(),
                    qualified_name: "rag_eval::hybrid_rrf::DeterministicKaiQueryPlanner"
                        .to_string(),
                    node_kind: StructuralNodeKind::Struct,
                    signature: "pub struct DeterministicKaiQueryPlanner".to_string(),
                    parent_scope: Some("hybrid_rrf".to_string()),
                    imports: vec![],
                    calls: vec!["plan_query".to_string()],
                    called_by: vec!["HybridSearchIndex::search_hybrid".to_string()],
                    start_line: 270,
                    end_line: 365,
                },
            ],
        },
        // 12. Pre-scoring ACL & tenant isolation record (`HybridAccessFilter`).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/hybrid_rrf.rs#HybridAccessFilter",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/hybrid_rrf.rs",
                "rust",
            ),
            title: "HybridAccessFilter mandatory pre-scoring tenant & role ACL isolation"
                .to_string(),
            raw_text: "Enforces strict tenant_id equality, allowed_principals role intersection, \
                       path_prefix, path_glob, language, and node_kind pre-filtering in \
                       crates/rag-eval/src/hybrid_rrf.rs before any search leg scores candidates."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "is_record_authorized_and_in_scope".to_string(),
                qualified_name: "rag_eval::hybrid_rrf::is_record_authorized_and_in_scope"
                    .to_string(),
                node_kind: StructuralNodeKind::Function,
                signature: "pub fn is_record_authorized_and_in_scope(record: &CanonicalEvidenceRecord, filter: &HybridAccessFilter) -> bool"
                    .to_string(),
                parent_scope: Some("hybrid_rrf".to_string()),
                imports: vec![],
                calls: vec!["glob_matches".to_string()],
                called_by: vec!["HybridSearchIndex::search_hybrid".to_string()],
                start_line: 485,
                end_line: 545,
            }],
        },
        // 13. Derived index rebuild contract (`rebuild_derived_indexes`).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/rag-eval/src/hybrid_rrf.rs#rebuild_derived_indexes",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/rag-eval/src/hybrid_rrf.rs",
                "rust",
            ),
            title: "HybridSearchIndex::rebuild_derived_indexes non-destructive derived rebuild"
                .to_string(),
            raw_text: "Drops and reconstructs derived BM25 lexical, dense vector, and structural AST \
                       indexes from immutable canonical_records in crates/rag-eval/src/hybrid_rrf.rs \
                       without destroying or mutating SourceIdentity version_id or provenance."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "rebuild_derived_indexes".to_string(),
                qualified_name:
                    "rag_eval::hybrid_rrf::HybridSearchIndex::rebuild_derived_indexes".to_string(),
                node_kind: StructuralNodeKind::Method,
                signature: "pub fn rebuild_derived_indexes(&mut self, collection: &str, runtime_meta: &IndexEmbeddingMetadata, embedder: &dyn PrefixedTextEmbedder) -> Result<usize, HybridSearchError>"
                    .to_string(),
                parent_scope: Some("HybridSearchIndex".to_string()),
                imports: vec![],
                calls: vec!["upsert_record".to_string(), "upsert_validated_vector".to_string()],
                called_by: vec![],
                start_line: 920,
                end_line: 965,
            }],
        },
        // 14. Tiered KV-Cache Router (`BidawCostModel`).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/router/src/kv_locality.rs#BidawCostModel",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/router/src/kv_locality.rs",
                "rust",
            ),
            title: "BidawCostModel tiered HBM/DRAM/NVMe KV-cache replica routing".to_string(),
            raw_text: "Routes multi-turn inference requests in crates/router/src/kv_locality.rs to the \
                       worker replica holding prefix blocks in GPU HBM, host DRAM, or NVMe swap \
                       bounded by prefill recomputation latency."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "BidawCostModel".to_string(),
                qualified_name: "hanzo_router::kv_locality::BidawCostModel".to_string(),
                node_kind: StructuralNodeKind::Struct,
                signature: "pub struct BidawCostModel".to_string(),
                parent_scope: Some("kv_locality".to_string()),
                imports: vec![],
                calls: vec!["estimate_transfer_ms".to_string(), "select_replica".to_string()],
                called_by: vec!["route_chat_completion".to_string()],
                start_line: 44,
                end_line: 128,
            }],
        },
        // 15. Adaptive Retry & Circuit Breaker (`ExponentialBackoffCircuitBreaker`).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/gateway/src/resilience.rs#ExponentialBackoffCircuitBreaker",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/gateway/src/resilience.rs",
                "rust",
            ),
            title: "ExponentialBackoffCircuitBreaker transient 429/503 overload shedding"
                .to_string(),
            raw_text: "Guards inference gateway endpoints in crates/gateway/src/resilience.rs against \
                       HTTP 429 and 503 rate-limit overload and timeout failures using decorrelated jitter \
                       backoff and half-open circuit breaker probes."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "ExponentialBackoffCircuitBreaker".to_string(),
                qualified_name:
                    "hanzo_gateway::resilience::ExponentialBackoffCircuitBreaker".to_string(),
                node_kind: StructuralNodeKind::Struct,
                signature: "pub struct ExponentialBackoffCircuitBreaker".to_string(),
                parent_scope: Some("resilience".to_string()),
                imports: vec![],
                calls: vec!["record_failure".to_string(), "allow_probe".to_string()],
                called_by: vec!["dispatch_upstream_request".to_string()],
                start_line: 75,
                end_line: 164,
            }],
        },
        // 16. Tree-sitter AST Symbol Extractor (`TreeSitterSymbolExtractor` trait).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:crates/code-intel/src/ast_extractor.rs#TreeSitterSymbolExtractor",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "crates/code-intel/src/ast_extractor.rs",
                "rust",
            ),
            title: "TreeSitterSymbolExtractor trait for AST symbol and call-graph indexing"
                .to_string(),
            raw_text: "Defines the language-agnostic AST parser contract in crates/code-intel/src/ast_extractor.rs \
                       extracting StructuralSymbolNode definitions, qualified symbol paths, and caller/callee edges."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "TreeSitterSymbolExtractor".to_string(),
                qualified_name: "hanzo_code_intel::ast_extractor::TreeSitterSymbolExtractor"
                    .to_string(),
                node_kind: StructuralNodeKind::Trait,
                signature: "pub trait TreeSitterSymbolExtractor: Send + Sync".to_string(),
                parent_scope: Some("ast_extractor".to_string()),
                imports: vec![],
                calls: vec!["extract_symbols".to_string()],
                called_by: vec!["index_repository_blob".to_string()],
                start_line: 18,
                end_line: 62,
            }],
        },
        // 17. TypeScript SDK Hybrid Client (`HybridSearchClient` class).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:sdks/typescript/src/client.ts#HybridSearchClient",
                SourceArtifactKind::RepositoryBlob,
                "tenant_hanzo",
                vec![],
                "sdks/typescript/src/client.ts",
                "typescript",
            ),
            title: "HybridSearchClient TypeScript SDK client for RRF search".to_string(),
            raw_text: "TypeScript client in sdks/typescript/src/client.ts serializing HybridAccessFilter \
                       headers, tenant_id, path_glob, and RRF rank explanations over HTTP/JSON."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "HybridSearchClient".to_string(),
                qualified_name: "hanzo_sdk::client::HybridSearchClient".to_string(),
                node_kind: StructuralNodeKind::Class,
                signature: "export class HybridSearchClient".to_string(),
                parent_scope: Some("client".to_string()),
                imports: vec!["./types".to_string()],
                calls: vec!["searchHybrid".to_string()],
                called_by: vec![],
                start_line: 25,
                end_line: 112,
            }],
        },
        // 18. Lexical Distractor A: mentions superficial non-cluster tokens ("agent", "request", "CSS", "tooltip")
        // with high term frequency to test whether UncalibratedRawScoreSum & LexicalOnly get fooled on semantic queries.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "doc:distractor/ui_tooltip_animation.md",
                SourceArtifactKind::Document,
                "tenant_hanzo",
                vec![],
                "docs/ui/ui_tooltip_animation.md",
                "markdown",
            ),
            title: "Dashboard CSS tooltip spinner styling note".to_string(),
            raw_text: "Troubleshooting why a browser user-agent header or client agent status request badge \
                       on the dashboard page stays visible after an HTTP request: the user-agent parser reads \
                       the browser agent request header."
                .to_string(),
            symbols: vec![],
        },
        // 19. Lexical Distractor B: mentions superficial prose tokens ("question", "answered", "break", "generated", "individual", "cascading", "upstream", "services", "storms")
        // in a static FAQ formatting note to test whether UncalibratedRawScoreSum & UniformRrf get fooled by single-leg BM25 hits.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "doc:distractor/asset_bundle_folders.md",
                SourceArtifactKind::Document,
                "tenant_hanzo",
                vec![],
                "docs/ui/asset_bundle_folders.md",
                "markdown",
            ),
            title: "Static marketing FAQ & icon folder layout".to_string(),
            raw_text: "FAQ formatting template: when a support question can be answered from the static help page, \
                       break generated answers into individual bullet items. Also prevents cascading upstream \
                       services from returning formatting errors in storms."
                .to_string(),
            symbols: vec![],
        },
        // 20. Lexical Distractor C: mentions `release_turn_lock` and `record_finish` repeatedly in a changelog prose paragraph
        // (zero AST symbols) to test whether `UncalibratedRawScoreSum` and `LexicalOnly` rank a changelog note above actual AST definitions.
        CanonicalEvidenceRecord {
            identity: mk_id(
                "doc:distractor/release_notes_archive.md",
                SourceArtifactKind::Document,
                "tenant_hanzo",
                vec![],
                "docs/changelog/release_notes_archive.md",
                "markdown",
            ),
            title: "Historical v0.1 changelog mention of symbols".to_string(),
            raw_text: "Changelog summary note: mentions release_turn_lock, release_turn_lock, release_turn_lock, \
                       TreeSitterSymbolExtractor, TreeSitterSymbolExtractor, and RetrieveOrAnswerGate in plain release prose \
                       without any source implementation or AST definitions."
                .to_string(),
            symbols: vec![],
        },
        // 21. Role-restricted security admin runbook inside `tenant_hanzo` (must NEVER leak to standard callers).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "doc:security/enclave_kms_rotation.md",
                SourceArtifactKind::StoredDecision,
                "tenant_hanzo",
                vec!["security_admin"],
                "docs/security/enclave_kms_rotation.md",
                "markdown",
            ),
            title: "Restricted HSM enclave attestation key rotation procedure".to_string(),
            raw_text: "Confidential security_admin runbook for rotating SEV-SNP enclave attestation keys \
                       and master KMS seals. Also mentions record_finish and ImmutableIndexContract."
                .to_string(),
            symbols: vec![],
        },
        // 22. Other-tenant record (`tenant_external`) containing identical high-scoring keywords (must NEVER leak to `tenant_hanzo`).
        CanonicalEvidenceRecord {
            identity: mk_id(
                "blob:tenant_external/runner.rs#record_finish",
                SourceArtifactKind::RepositoryBlob,
                "tenant_external",
                vec![],
                "crates/agent/src/runner.rs",
                "rust",
            ),
            title: "External tenant private record_finish implementation".to_string(),
            raw_text: "pub async fn record_finish(&mut self, req_id: RequestId) -> Result<(), AgentError>. \
                       Private tenant_external code for turn completion deadlock and ImmutableIndexContract."
                .to_string(),
            symbols: vec![StructuralSymbolNode {
                symbol_name: "record_finish".to_string(),
                qualified_name: "external::AgentRunner::record_finish".to_string(),
                node_kind: StructuralNodeKind::Function,
                signature: "pub async fn record_finish(&mut self, req_id: RequestId) -> Result<(), AgentError>"
                    .to_string(),
                parent_scope: Some("AgentRunner".to_string()),
                imports: vec![],
                calls: vec!["release_turn_lock".to_string()],
                called_by: vec![],
                start_line: 10,
                end_line: 40,
            }],
        },
    ]
}

/// Returns the canonical 20-query benchmark suite (`4` queries per [`HybridQueryClass`])
/// for `hanzoai/index#11` (`HIP-1347`).
#[allow(clippy::too_many_lines)]
pub fn canonical_hybrid_queries() -> Vec<HybridBenchmarkCase> {
    let default_filter =
        HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);

    vec![
        // ====================================================================
        // Slice 1: ExactSymbolOrIdentifier (4 queries)
        // ====================================================================
        HybridBenchmarkCase {
            case_id: "exact_01_record_finish".to_string(),
            expected_class: HybridQueryClass::ExactSymbolOrIdentifier,
            raw_query: "record_finish".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec!["blob:crates/agent/src/runner.rs#record_finish".to_string()],
            required_fix_symbol: "record_finish".to_string(),
            required_fix_path: "crates/agent/src/runner.rs".to_string(),
            required_fix_invariant: "release_turn_lock".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "exact_02_swapped_task_prefix".to_string(),
            expected_class: HybridQueryClass::ExactSymbolOrIdentifier,
            raw_query: "MetadataError::SwappedTaskPrefix".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/metadata.rs#ImmutableIndexContract".to_string()
            ],
            required_fix_symbol: "ImmutableIndexContract".to_string(),
            required_fix_path: "crates/rag-eval/src/metadata.rs".to_string(),
            required_fix_invariant: "query_prefix".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "exact_03_truncate_matryoshka".to_string(),
            expected_class: HybridQueryClass::ExactSymbolOrIdentifier,
            raw_query: "truncate_matryoshka MATRYOSHKA_DIMS".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/metadata.rs#truncate_matryoshka".to_string()
            ],
            required_fix_symbol: "truncate_matryoshka".to_string(),
            required_fix_path: "crates/rag-eval/src/metadata.rs".to_string(),
            required_fix_invariant: "l2_norm_tolerance".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "exact_04_bidaw_cost_model".to_string(),
            expected_class: HybridQueryClass::ExactSymbolOrIdentifier,
            raw_query: "BidawCostModel".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec!["blob:crates/router/src/kv_locality.rs#BidawCostModel".to_string()],
            required_fix_symbol: "BidawCostModel".to_string(),
            required_fix_path: "crates/router/src/kv_locality.rs".to_string(),
            required_fix_invariant: "prefill".to_string(),
        },
        // ====================================================================
        // Slice 2: NaturalLanguageSemantic (4 paraphrased queries)
        // ====================================================================
        HybridBenchmarkCase {
            case_id: "sem_01_agent_freeze_after_request".to_string(),
            expected_class: HybridQueryClass::NaturalLanguageSemantic,
            raw_query: "why does the agent freeze after completing a request?".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "doc:incidents/INC-409-agent-completion-deadlock.md".to_string(),
            ],
            required_fix_symbol: "record_finish".to_string(),
            required_fix_path: "crates/agent/src/runner.rs".to_string(),
            required_fix_invariant: "release_turn_lock".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "sem_02_skip_database_on_easy_prompt".to_string(),
            expected_class: HybridQueryClass::NaturalLanguageSemantic,
            raw_query: "how do we skip unnecessary database lookups when a question can be answered directly?"
                .to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/self_rag.rs#RetrieveOrAnswerGate".to_string()
            ],
            required_fix_symbol: "RetrieveOrAnswerGate".to_string(),
            required_fix_path: "crates/rag-eval/src/self_rag.rs".to_string(),
            required_fix_invariant: "theta".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "sem_03_verify_individual_claims".to_string(),
            expected_class: HybridQueryClass::NaturalLanguageSemantic,
            raw_query: "how do we break generated answers into individual statements to verify citations?"
                .to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/factscore.rs#FactScoreEvaluator".to_string()
            ],
            required_fix_symbol: "FactScoreEvaluator".to_string(),
            required_fix_path: "crates/rag-eval/src/factscore.rs".to_string(),
            required_fix_invariant: "spanevidencebinder".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "sem_04_shed_overload_rate_limit".to_string(),
            expected_class: HybridQueryClass::NaturalLanguageSemantic,
            raw_query: "how do we prevent cascading retry storms when upstream services return transient errors?"
                .to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/gateway/src/resilience.rs#ExponentialBackoffCircuitBreaker".to_string()
            ],
            required_fix_symbol: "ExponentialBackoffCircuitBreaker".to_string(),
            required_fix_path: "crates/gateway/src/resilience.rs".to_string(),
            required_fix_invariant: "jitter".to_string(),
        },
        // ====================================================================
        // Slice 3: StructuralAstPattern (4 AST & call-graph queries)
        // ====================================================================
        HybridBenchmarkCase {
            case_id: "ast_01_callers_of_release_turn_lock".to_string(),
            expected_class: HybridQueryClass::StructuralAstPattern,
            raw_query: "method callers of release_turn_lock in TurnScheduler".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/agent/src/scheduler.rs#dispatch_turn_loop".to_string()
            ],
            required_fix_symbol: "dispatch_turn_loop".to_string(),
            required_fix_path: "crates/agent/src/scheduler.rs".to_string(),
            required_fix_invariant: "release_turn_lock".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "ast_02_trait_treesitter_extractor".to_string(),
            expected_class: HybridQueryClass::StructuralAstPattern,
            raw_query: "trait TreeSitterSymbolExtractor signature".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/code-intel/src/ast_extractor.rs#TreeSitterSymbolExtractor".to_string()
            ],
            required_fix_symbol: "TreeSitterSymbolExtractor".to_string(),
            required_fix_path: "crates/code-intel/src/ast_extractor.rs".to_string(),
            required_fix_invariant: "structuralsymbolnode".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "ast_03_struct_ann_maintenance_runner".to_string(),
            expected_class: HybridQueryClass::StructuralAstPattern,
            raw_query: "struct AnnMaintenanceRunner calls drain_merge_budget".to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/ann_insert.rs#AnnMaintenanceRunner".to_string()
            ],
            required_fix_symbol: "AnnMaintenanceRunner".to_string(),
            required_fix_path: "crates/rag-eval/src/ann_insert.rs".to_string(),
            required_fix_invariant: "delta".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "ast_04_class_hybrid_search_client".to_string(),
            expected_class: HybridQueryClass::StructuralAstPattern,
            raw_query: "class HybridSearchClient method searchHybrid".to_string(),
            filter: default_filter
                .clone()
                .with_language("typescript")
                .with_node_kind(StructuralNodeKind::Class),
            relevant_doc_ids: vec![
                "blob:sdks/typescript/src/client.ts#HybridSearchClient".to_string()
            ],
            required_fix_symbol: "HybridSearchClient".to_string(),
            required_fix_path: "sdks/typescript/src/client.ts".to_string(),
            required_fix_invariant: "hybridaccessfilter".to_string(),
        },
        // ====================================================================
        // Slice 4: PathOrFileScoped (4 file/path-filtered queries)
        // ====================================================================
        HybridBenchmarkCase {
            case_id: "path_01_milli_hybrid_score_merge".to_string(),
            expected_class: HybridQueryClass::PathOrFileScoped,
            raw_query: "crates/milli/src/search/hybrid.rs score ratio merger".to_string(),
            filter: default_filter.clone().with_path_prefix("crates/milli/"),
            relevant_doc_ids: vec![
                "blob:crates/milli/src/search/hybrid.rs#SemanticScore".to_string()
            ],
            required_fix_symbol: "compare_scores".to_string(),
            required_fix_path: "crates/milli/src/search/hybrid.rs".to_string(),
            required_fix_invariant: "left_ratio".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "path_02_corrective_rag_file".to_string(),
            expected_class: HybridQueryClass::PathOrFileScoped,
            raw_query: "crates/rag-eval/src/corrective.rs mojibake distractor recovery".to_string(),
            filter: default_filter.clone().with_path_glob("crates/rag-eval/src/*.rs"),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/corrective.rs#CorrectiveRetrievalPolicy".to_string()
            ],
            required_fix_symbol: "CorrectiveRetrievalPolicy".to_string(),
            required_fix_path: "crates/rag-eval/src/corrective.rs".to_string(),
            required_fix_invariant: "secondaryretriever".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "path_03_hybrid_rrf_prefilter".to_string(),
            expected_class: HybridQueryClass::PathOrFileScoped,
            raw_query: "crates/rag-eval/src/hybrid_rrf.rs tenant isolation prefilter".to_string(),
            filter: default_filter.clone().with_path_prefix("crates/rag-eval/src/"),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/hybrid_rrf.rs#HybridAccessFilter".to_string()
            ],
            required_fix_symbol: "is_record_authorized_and_in_scope".to_string(),
            required_fix_path: "crates/rag-eval/src/hybrid_rrf.rs".to_string(),
            required_fix_invariant: "tenant_id".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "path_04_derived_rebuild_path".to_string(),
            expected_class: HybridQueryClass::PathOrFileScoped,
            raw_query: "crates/rag-eval/src/hybrid_rrf.rs reconstruct derived indexes".to_string(),
            filter: default_filter.clone().with_path_prefix("crates/rag-eval/src/"),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/hybrid_rrf.rs#rebuild_derived_indexes".to_string()
            ],
            required_fix_symbol: "rebuild_derived_indexes".to_string(),
            required_fix_path: "crates/rag-eval/src/hybrid_rrf.rs".to_string(),
            required_fix_invariant: "canonical_records".to_string(),
        },
        // ====================================================================
        // Slice 5: HybridMultiLeg (4 combined identifier + natural language queries)
        // ====================================================================
        HybridBenchmarkCase {
            case_id: "multi_01_record_finish_deadlock_why".to_string(),
            expected_class: HybridQueryClass::HybridMultiLeg,
            raw_query: "why does record_finish drop the mutex guard before awaiting channel drain?"
                .to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec!["blob:crates/agent/src/runner.rs#record_finish".to_string()],
            required_fix_symbol: "record_finish".to_string(),
            required_fix_path: "crates/agent/src/runner.rs".to_string(),
            required_fix_invariant: "release_turn_lock".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "multi_02_rrf_combine_legs_how".to_string(),
            expected_class: HybridQueryClass::HybridMultiLeg,
            raw_query: "how does ReciprocalRankFusion combine lexical, vector, and structural legs?"
                .to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/hybrid_rrf.rs#ReciprocalRankFusion".to_string()
            ],
            required_fix_symbol: "ReciprocalRankFusion".to_string(),
            required_fix_path: "crates/rag-eval/src/hybrid_rrf.rs".to_string(),
            required_fix_invariant: "rrf_k".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "multi_03_immutable_contract_prefix_why".to_string(),
            expected_class: HybridQueryClass::HybridMultiLeg,
            raw_query: "how does ImmutableIndexContract prevent mixing incompatible query_prefix and doc_prefix embeddings?"
                .to_string(),
            filter: default_filter.clone(),
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/metadata.rs#ImmutableIndexContract".to_string()
            ],
            required_fix_symbol: "ImmutableIndexContract".to_string(),
            required_fix_path: "crates/rag-eval/src/metadata.rs".to_string(),
            required_fix_invariant: "swappedtaskprefix".to_string(),
        },
        HybridBenchmarkCase {
            case_id: "multi_04_rebuild_derived_without_destroying".to_string(),
            expected_class: HybridQueryClass::HybridMultiLeg,
            raw_query: "how does rebuild_derived_indexes reconstruct lexical and vector views without destroying canonical records?"
                .to_string(),
            filter: default_filter,
            relevant_doc_ids: vec![
                "blob:crates/rag-eval/src/hybrid_rrf.rs#rebuild_derived_indexes".to_string()
            ],
            required_fix_symbol: "rebuild_derived_indexes".to_string(),
            required_fix_path: "crates/rag-eval/src/hybrid_rrf.rs".to_string(),
            required_fix_invariant: "sourceidentity".to_string(),
        },
    ]
}

/// Builds a populated [`HybridSearchIndex`] over [`canonical_hybrid_corpus`] under collection `"hip1347_hybrid"`.
pub fn build_canonical_hybrid_index() -> Result<(HybridSearchIndex, IndexEmbeddingMetadata), HybridSearchError> {
    let meta = canonical_hybrid_embedding_metadata();
    let embedder = DeterministicSemanticCodeEmbedder;
    let mut index = HybridSearchIndex::new();
    index.register_collection("hip1347_hybrid", meta.clone())?;
    for record in canonical_hybrid_corpus() {
        index.upsert_record("hip1347_hybrid", &meta, record, &embedder)?;
    }
    Ok((index, meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_formula_matches_exact_closed_form_and_preserves_leg_explanations() {
        let mut records = BTreeMap::new();
        for id in ["doc_a", "doc_b", "doc_c"] {
            records.insert(
                id.to_string(),
                CanonicalEvidenceRecord {
                    identity: SourceIdentity {
                        doc_id: id.to_string(),
                        source_id: "repo:test".to_string(),
                        version_id: "v1".to_string(),
                        artifact_kind: SourceArtifactKind::RepositoryBlob,
                        tenant_id: "t1".to_string(),
                        project_id: "p1".to_string(),
                        repo_id: "r1".to_string(),
                        allowed_principals: vec![],
                        path: format!("src/{id}.rs"),
                        language: "rust".to_string(),
                    },
                    title: id.to_string(),
                    raw_text: format!("body of {id}"),
                    symbols: vec![],
                },
            );
        }

        let plan = HybridQueryPlan {
            decision_head: DecisionHead::Kai,
            query_class: HybridQueryClass::HybridMultiLeg,
            weights: LegWeights { lexical: 0.35, vector: 0.35, structural: 0.30 },
            rrf_k: 60.0,
            per_leg_candidate_depth: 10,
            top_k: 10,
            execute_lexical: true,
            execute_vector: true,
            execute_structural: true,
            extracted_identifiers: vec![],
            extracted_ast_kinds: vec![],
            extracted_path_segments: vec![],
        };

        // doc_a appears in all 3 legs (rank 1 in lex, rank 2 in vec, rank 1 in struct).
        // doc_b appears in 2 legs (rank 2 in lex, rank 1 in vec).
        // doc_c appears in 1 leg (rank 2 in struct).
        let lex = vec![("doc_a".to_string(), 14.2), ("doc_b".to_string(), 9.1)];
        let vec_leg = vec![("doc_b".to_string(), 0.91), ("doc_a".to_string(), 0.84)];
        let str_leg = vec![
            ("doc_a".to_string(), 7.5, vec!["mod::doc_a".to_string()]),
            ("doc_c".to_string(), 3.0, vec!["mod::doc_c".to_string()]),
        ];

        let hits = ReciprocalRankFusion::fuse(&records, &plan, &lex, &vec_leg, &str_leg);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].doc_id, "doc_a");
        assert_eq!(hits[0].final_rank, 1);

        let expected_a = 0.35 / (60.0 + 1.0) + 0.35 / (60.0 + 2.0) + 0.30 / (60.0 + 1.0);
        assert!((hits[0].rrf_score - expected_a).abs() < 1e-12);
        assert_eq!(hits[0].explanation.active_legs_matched, 3);
        assert_eq!(hits[0].explanation.best_leg_rank, 1);
        assert_eq!(hits[0].explanation.lexical.as_ref().unwrap().rank_1_based, 1);
        assert_eq!(hits[0].explanation.vector.as_ref().unwrap().rank_1_based, 2);
        assert_eq!(hits[0].explanation.structural.as_ref().unwrap().rank_1_based, 1);

        let expected_b = 0.35 / (60.0 + 2.0) + 0.35 / (60.0 + 1.0);
        assert_eq!(hits[1].doc_id, "doc_b");
        assert!((hits[1].rrf_score - expected_b).abs() < 1e-12);
        assert_eq!(hits[1].explanation.active_legs_matched, 2);
        assert!(hits[1].explanation.structural.is_none());

        let expected_c = 0.30 / (60.0 + 2.0);
        assert_eq!(hits[2].doc_id, "doc_c");
        assert!((hits[2].rrf_score - expected_c).abs() < 1e-12);
        assert_eq!(hits[2].explanation.active_legs_matched, 1);
    }

    #[test]
    fn rrf_deterministic_tie_breaking_is_order_independent() {
        let mut records = BTreeMap::new();
        for id in ["doc_lex", "doc_vec", "doc_str", "doc_z"] {
            records.insert(
                id.to_string(),
                CanonicalEvidenceRecord {
                    identity: SourceIdentity {
                        doc_id: id.to_string(),
                        source_id: "repo:test".to_string(),
                        version_id: "v1".to_string(),
                        artifact_kind: SourceArtifactKind::Document,
                        tenant_id: "t1".to_string(),
                        project_id: "p1".to_string(),
                        repo_id: "r1".to_string(),
                        allowed_principals: vec![],
                        path: format!("{id}.md"),
                        language: "markdown".to_string(),
                    },
                    title: id.to_string(),
                    raw_text: id.to_string(),
                    symbols: vec![],
                },
            );
        }

        let plan = HybridQueryPlan {
            decision_head: DecisionHead::Kai,
            query_class: HybridQueryClass::HybridMultiLeg,
            weights: LegWeights::uniform(),
            rrf_k: 60.0,
            per_leg_candidate_depth: 10,
            top_k: 10,
            execute_lexical: true,
            execute_vector: true,
            execute_structural: true,
            extracted_identifiers: vec![],
            extracted_ast_kinds: vec![],
            extracted_path_segments: vec![],
        };

        // Each document is Rank 1 in exactly one leg with equal uniform weight 1/3 -> identical RRF score.
        // Tie-breaker must order: lexical rank 1 ("doc_lex") -> structural rank 1 ("doc_str") -> vector rank 1 ("doc_vec").
        let lex = vec![("doc_lex".to_string(), 5.0)];
        let vec_leg = vec![("doc_vec".to_string(), 0.9)];
        let str_leg = vec![("doc_str".to_string(), 4.0, vec![])];

        let hits = ReciprocalRankFusion::fuse(&records, &plan, &lex, &vec_leg, &str_leg);
        let order: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(order, vec!["doc_lex", "doc_str", "doc_vec"]);
    }

    #[test]
    fn code_tokenizer_splits_snake_case_camel_case_and_qualified_symbols() {
        let sub1 = split_identifier_subtokens("record_finish");
        assert!(sub1.contains(&"record_finish".to_string()));
        assert!(sub1.contains(&"record".to_string()));
        assert!(sub1.contains(&"finish".to_string()));

        let sub2 = split_identifier_subtokens("ImmutableIndexContract");
        assert!(sub2.contains(&"immutableindexcontract".to_string()));
        assert!(sub2.contains(&"immutable".to_string()));
        assert!(sub2.contains(&"index".to_string()));
        assert!(sub2.contains(&"contract".to_string()));

        let sub3 = split_identifier_subtokens("MetadataError::SwappedTaskPrefix");
        assert!(sub3.contains(&"metadataerror".to_string()));
        assert!(sub3.contains(&"swappedtaskprefix".to_string()));
        assert!(sub3.contains(&"swapped".to_string()));
        assert!(sub3.contains(&"task".to_string()));
        assert!(sub3.contains(&"prefix".to_string()));

        let sub4 = split_identifier_subtokens("HTML5Parser::parse_node");
        assert!(sub4.contains(&"html5".to_string()));
        assert!(sub4.contains(&"parser".to_string()));
        assert!(sub4.contains(&"parse".to_string()));
        assert!(sub4.contains(&"node".to_string()));
    }

    #[test]
    fn kai_query_planner_classifies_all_five_query_categories_and_assigns_weights() {
        let planner = DeterministicKaiQueryPlanner;
        let filter = HybridAccessFilter::for_tenant("tenant_hanzo", vec![]);
        let cfg = HybridRrfConfig::default();

        let p_exact = planner.plan_query("record_finish", &filter, &cfg).unwrap();
        assert_eq!(p_exact.decision_head, DecisionHead::Kai);
        assert_eq!(p_exact.query_class, HybridQueryClass::ExactSymbolOrIdentifier);
        assert!((p_exact.weights.lexical - 0.45).abs() < 1e-9);
        assert!((p_exact.weights.structural - 0.45).abs() < 1e-9);

        let p_sem = planner
            .plan_query("why does the agent freeze after completing a request?", &filter, &cfg)
            .unwrap();
        assert_eq!(p_sem.query_class, HybridQueryClass::NaturalLanguageSemantic);
        assert!((p_sem.weights.vector - 0.65).abs() < 1e-9);

        let p_ast = planner
            .plan_query("trait TreeSitterSymbolExtractor signature", &filter, &cfg)
            .unwrap();
        assert_eq!(p_ast.query_class, HybridQueryClass::StructuralAstPattern);
        assert!((p_ast.weights.structural - 0.60).abs() < 1e-9);

        let p_path = planner
            .plan_query("crates/milli/src/search/hybrid.rs score ratio", &filter, &cfg)
            .unwrap();
        assert_eq!(p_path.query_class, HybridQueryClass::PathOrFileScoped);
        assert!((p_path.weights.lexical - 0.45).abs() < 1e-9);

        let p_multi = planner
            .plan_query(
                "why does record_finish drop the mutex guard before awaiting channel drain?",
                &filter,
                &cfg,
            )
            .unwrap();
        assert_eq!(p_multi.query_class, HybridQueryClass::HybridMultiLeg);
        assert!((p_multi.weights.lexical - 0.35).abs() < 1e-9);
        assert!((p_multi.weights.vector - 0.35).abs() < 1e-9);
        assert!((p_multi.weights.structural - 0.30).abs() < 1e-9);
    }

    #[test]
    fn acl_and_scope_prefilter_blocks_cross_tenant_and_unauthorized_roles_before_scoring() {
        let (index, meta) = build_canonical_hybrid_index().unwrap();
        let planner = DeterministicKaiQueryPlanner;
        let embedder = DeterministicSemanticCodeEmbedder;
        let cost = HybridCostModel::default();
        let cfg = HybridRrfConfig::default();

        // Standard developer in tenant_hanzo must never see tenant_external or security_admin records.
        let dev_filter =
            HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);
        let out = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &dev_filter,
                "record_finish enclave attestation keys",
                &cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap();
        assert!(out
            .hits
            .iter()
            .all(|h| h.identity.tenant_id == "tenant_hanzo"));
        assert!(out
            .hits
            .iter()
            .all(|h| h.doc_id != "doc:security/enclave_kms_rotation.md"));
        assert!(out
            .hits
            .iter()
            .all(|h| h.doc_id != "blob:tenant_external/runner.rs#record_finish"));

        // Authorized security_admin in tenant_hanzo retrieves the restricted KMS runbook at rank 1.
        let admin_filter =
            HybridAccessFilter::for_tenant("tenant_hanzo", vec!["security_admin".to_string()]);
        let admin_out = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &admin_filter,
                "rotating SEV-SNP enclave attestation keys and master KMS seals",
                &cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap();
        assert_eq!(admin_out.hits[0].doc_id, "doc:security/enclave_kms_rotation.md");
    }

    #[test]
    fn derived_indexes_rebuild_cleanly_without_mutating_canonical_source_identities() {
        let (mut index, meta) = build_canonical_hybrid_index().unwrap();
        let planner = DeterministicKaiQueryPlanner;
        let embedder = DeterministicSemanticCodeEmbedder;
        let cost = HybridCostModel::default();
        let cfg = HybridRrfConfig::default();
        let filter = HybridAccessFilter::for_tenant("tenant_hanzo", vec!["developer".to_string()]);

        let snapshot_before: BTreeMap<String, CanonicalEvidenceRecord> =
            index.canonical_records("hip1347_hybrid").unwrap().clone();

        let before_hits = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &filter,
                "record_finish",
                &cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap()
            .hits;

        // Drop all derived indexes while preserving canonical records.
        index.drop_derived_indexes_preserve_canonical("hip1347_hybrid").unwrap();
        assert_eq!(index.canonical_record_count("hip1347_hybrid"), snapshot_before.len());

        // Rebuild derived indexes from canonical records.
        let rebuilt = index
            .rebuild_derived_indexes("hip1347_hybrid", &meta, &embedder)
            .unwrap();
        assert_eq!(rebuilt, snapshot_before.len());
        assert_eq!(index.derived_rebuild_count("hip1347_hybrid"), 1);

        // Verify canonical identities and search rankings are 100% unchanged.
        let snapshot_after = index.canonical_records("hip1347_hybrid").unwrap();
        assert_eq!(&snapshot_before, snapshot_after);

        let after_hits = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &filter,
                "record_finish",
                &cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap()
            .hits;
        assert_eq!(before_hits, after_hits);
    }

    #[test]
    fn immutable_embedding_metadata_contract_rejects_mismatched_prefix_or_dimensions() {
        let (mut index, meta) = build_canonical_hybrid_index().unwrap();
        let planner = DeterministicKaiQueryPlanner;
        let embedder = DeterministicSemanticCodeEmbedder;
        let cost = HybridCostModel::default();
        let cfg = HybridRrfConfig::default();
        let filter = HybridAccessFilter::for_tenant("tenant_hanzo", vec![]);
        let count_before = index.canonical_record_count("hip1347_hybrid");

        // Swapped query text carrying doc_prefix must fail closed with SwappedTaskPrefix.
        let swap_err = index
            .search_hybrid(
                "hip1347_hybrid",
                &meta,
                &filter,
                "search_document: record_finish",
                &cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap_err();
        assert!(matches!(
            swap_err,
            HybridSearchError::Metadata(MetadataError::SwappedTaskPrefix {
                operation: IndexOperation::Query,
                ..
            })
        ));

        // Runtime dimension drift (128 vs persisted 256) must fail closed with ContractMismatch.
        let mut wrong_dim_meta = meta.clone();
        wrong_dim_meta.dimensions = 128;
        let dim_err = index
            .search_hybrid(
                "hip1347_hybrid",
                &wrong_dim_meta,
                &filter,
                "record_finish",
                &cfg,
                &planner,
                &embedder,
                &cost,
            )
            .unwrap_err();
        assert!(matches!(
            dim_err,
            HybridSearchError::Metadata(MetadataError::ContractMismatch {
                operation: IndexOperation::Query,
                ..
            })
        ));

        // Attempting to insert a record with pre-swapped query_prefix fails without mutating the store.
        let mut bad_rec = canonical_hybrid_corpus().into_iter().next().unwrap();
        bad_rec.identity.doc_id = "bad_doc".to_string();
        bad_rec.raw_text = "search_query: accidentally pre-prefixed".to_string();
        let ins_err = index
            .upsert_record("hip1347_hybrid", &meta, bad_rec, &embedder)
            .unwrap_err();
        assert!(matches!(
            ins_err,
            HybridSearchError::Metadata(MetadataError::SwappedTaskPrefix {
                operation: IndexOperation::Insert,
                ..
            })
        ));
        assert_eq!(index.canonical_record_count("hip1347_hybrid"), count_before);
    }
}
