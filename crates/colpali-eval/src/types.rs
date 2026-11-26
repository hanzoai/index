use std::fmt;

/// Fixed ColPali patch embedding dimension (matches ColPali / ColQwen2 128-d projection head).
pub const DEFAULT_EMBEDDING_DIM: usize = 128;

/// Error type for clean-room ColPali visual document retrieval operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColPaliError {
    /// Fail-closed rejection when `tenant_id` is empty or whitespace-only.
    EmptyTenantId,
    /// Fail-closed rejection when `allowed_acl_groups` is empty or contains an empty group.
    EmptyAclGroups,
    /// Vector dimension mismatch against the index's configured embedding dimension.
    InvalidDimension { expected: usize, got: usize },
    /// Attempted to ingest a canonical page with zero visual patch embeddings.
    EmptyPatchSet { doc_id: String, page_no: u32 },
    /// Query contained zero token embeddings or invalid `top_k`.
    InvalidQuery { reason: String },
    /// Binary or checksum corruption detected when loading a derived index.
    CorruptedIndex { reason: String },
    /// I/O error while reading repository sources or writing receipts.
    Io(String),
}

impl fmt::Display for ColPaliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTenantId => {
                write!(f, "fail-closed ACL rejection: tenant_id must be non-empty")
            }
            Self::EmptyAclGroups => {
                write!(f, "fail-closed ACL rejection: allowed_acl_groups must be non-empty")
            }
            Self::InvalidDimension { expected, got } => {
                write!(f, "invalid embedding dimension: expected {expected}, got {got}")
            }
            Self::EmptyPatchSet { doc_id, page_no } => {
                write!(f, "canonical page {doc_id}#p{page_no} has empty visual patch set")
            }
            Self::InvalidQuery { reason } => write!(f, "invalid query: {reason}"),
            Self::CorruptedIndex { reason } => write!(f, "corrupted derived index: {reason}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for ColPaliError {}

/// Document page modality classification used for evaluation slicing and layout characterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PageModality {
    /// Clean prose, specifications, code/config references, and exact identifiers.
    TextOnly,
    /// Multi-column 2D numeric benchmark tables, matrices, and structured schedules.
    DenseTable,
    /// Scientific plots, bar/line charts, Pareto curves, and architecture diagrams.
    ScientificFigure,
    /// Two-column academic pages with dense LaTeX equations, subscripts, and Greek notation.
    MultiColumnFormula,
}

impl PageModality {
    /// Canonical lowercase identifier for JSON receipts and tables.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextOnly => "text_only",
            Self::DenseTable => "dense_table",
            Self::ScientificFigure => "scientific_figure",
            Self::MultiColumnFormula => "multi_column_formula",
        }
    }

    /// Fixed-order list of all four page modalities.
    pub const fn all() -> [Self; 4] {
        [Self::TextOnly, Self::DenseTable, Self::ScientificFigure, Self::MultiColumnFormula]
    }

    /// Stable numeric discriminant for canonical hashing and binary serialization.
    pub const fn tag(self) -> u8 {
        match self {
            Self::TextOnly => 0,
            Self::DenseTable => 1,
            Self::ScientificFigure => 2,
            Self::MultiColumnFormula => 3,
        }
    }

    /// Reconstruct modality from serialized discriminant.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::TextOnly),
            1 => Some(Self::DenseTable),
            2 => Some(Self::ScientificFigure),
            3 => Some(Self::MultiColumnFormula),
            _ => None,
        }
    }

    /// Returns true if the page contains non-trivial 2D visual layout elements.
    pub const fn is_visual_heavy(self) -> bool {
        !matches!(self, Self::TextOnly)
    }
}

/// Stable canonical primary key for a document page.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageKey {
    pub doc_id: String,
    pub page_no: u32,
}

impl PageKey {
    pub fn new(doc_id: impl Into<String>, page_no: u32) -> Self {
        Self { doc_id: doc_id.into(), page_no }
    }
}

/// Observable PDF/raster layout features extracted cheaply during document ingestion
/// (analogous to lightweight PDF object inspection / MuPDF layout heuristics).
///
/// The selective visual router relies ONLY on these observable ingest-time features—never
/// on ground-truth modality labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageLayoutFeatures {
    /// Fraction of page area covered by clean linear text runs (`0.0..=1.0`).
    pub text_coverage: f32,
    /// Fraction of page area covered by grid/table bounding boxes (`0.0..=1.0`).
    pub table_region_ratio: f32,
    /// Fraction of page area covered by embedded raster/vector figures (`0.0..=1.0`).
    pub figure_region_ratio: f32,
    /// Fraction of page area covered by display/inline math glyph clusters (`0.0..=1.0`).
    pub formula_region_ratio: f32,
    /// Mean character/word confidence emitted by the OCR/PDF text extractor (`0.0..=1.0`).
    pub ocr_mean_confidence: f32,
    /// Detected column count (`1`, `2`, `3`, ...).
    pub column_count: u8,
}

impl PageLayoutFeatures {
    /// Computes an ingest-time visual complexity score in `[0.0, 1.0]`.
    ///
    /// Higher scores indicate pages where standard OCR/text extraction drops 2D layout,
    /// chart semantics, or equation structure, making ColPali patch indexing worthwhile.
    pub fn visual_complexity_score(&self) -> f32 {
        let visual_area = (self.table_region_ratio * 1.15
            + self.figure_region_ratio * 1.30
            + self.formula_region_ratio * 1.10)
            .clamp(0.0, 1.0);
        let ocr_uncertainty = (1.0 - self.ocr_mean_confidence).clamp(0.0, 1.0);
        let multi_col_bonus = if self.column_count > 1 { 0.12 } else { 0.0 };
        (0.55 * visual_area + 0.35 * ocr_uncertainty + multi_col_bonus).clamp(0.0, 1.0)
    }
}

/// Authoritative canonical document page record stored in Hanzo VFS/S3 canonical truth.
///
/// All retrieval indexes (BM25+OCR hybrid, ColPali FP32, ColPali 1-bit Binary+SQ8,
/// and the Selective Hybrid Visual Router) are strictly derived views rebuilt from
/// `CanonicalPageRecord` entries.
#[derive(Debug, Clone)]
pub struct CanonicalPageRecord {
    pub doc_id: String,
    pub page_no: u32,
    pub version: u64,
    pub tenant_id: String,
    pub acl_group: String,
    pub page_modality: PageModality,
    /// Provenance tag linking back to real repository docs or scientific page templates.
    pub source_origin: String,
    pub layout_features: PageLayoutFeatures,
    /// Extracted OCR / PDF-text output (reflects modality-specific OCR fidelity/corruption).
    pub ocr_text: String,
    /// Measured fraction of ground-truth tokens/relations corrupted or lost by OCR (`0.0..=1.0`).
    pub ocr_corruption_rate: f32,
    /// Single-vector dense embedding computed from `ocr_text` (dimension `D`, L2-normalized).
    pub ocr_dense_embedding: Vec<f32>,
    /// Visual patch grid rows (e.g. `8` for 8x8 pooled or `32` for full ColPali 1024).
    pub patch_grid_rows: u16,
    /// Visual patch grid columns.
    pub patch_grid_cols: u16,
    /// Multi-vector visual page patch embeddings (`N_patches` vectors of dimension `D`).
    pub patch_embeddings: Vec<Vec<f32>>,
}

impl CanonicalPageRecord {
    pub fn key(&self) -> PageKey {
        PageKey { doc_id: self.doc_id.clone(), page_no: self.page_no }
    }
}

/// Mandatory tenant and ACL scope enforced pre-scoring / pre-return on every query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessScope {
    pub tenant_id: String,
    pub allowed_acl_groups: Vec<String>,
}

impl AccessScope {
    pub fn new(tenant_id: impl Into<String>, allowed_acl_groups: Vec<String>) -> Self {
        Self { tenant_id: tenant_id.into(), allowed_acl_groups }
    }

    /// Fail-closed validation: rejects empty tenant IDs or empty ACL group sets.
    pub fn validate(&self) -> Result<(), ColPaliError> {
        if self.tenant_id.trim().is_empty() {
            return Err(ColPaliError::EmptyTenantId);
        }
        if self.allowed_acl_groups.is_empty()
            || self.allowed_acl_groups.iter().any(|g| g.trim().is_empty())
        {
            return Err(ColPaliError::EmptyAclGroups);
        }
        Ok(())
    }

    /// Constant-semantics pre-return predicate checking tenant and ACL membership.
    pub fn permits(&self, tenant_id: &str, acl_group: &str) -> bool {
        self.tenant_id == tenant_id
            && self.allowed_acl_groups.iter().any(|allowed| allowed == acl_group)
    }
}

/// Retrieval query containing both raw query text (for BM25/lexical routing) and
/// query embeddings (single-vector for OCR dense baseline; multi-vector token embeddings
/// for ColPali MaxSim late interaction).
#[derive(Debug, Clone)]
pub struct VisualQuery {
    pub query_id: String,
    pub text: String,
    pub dense_embedding: Vec<f32>,
    pub token_embeddings: Vec<Vec<f32>>,
    pub scope: AccessScope,
    pub top_k: usize,
}

impl VisualQuery {
    pub fn validate(&self, expected_dim: usize) -> Result<(), ColPaliError> {
        self.scope.validate()?;
        if self.top_k == 0 {
            return Err(ColPaliError::InvalidQuery { reason: "top_k must be >= 1".to_string() });
        }
        if self.dense_embedding.len() != expected_dim {
            return Err(ColPaliError::InvalidDimension {
                expected: expected_dim,
                got: self.dense_embedding.len(),
            });
        }
        if self.token_embeddings.is_empty() {
            return Err(ColPaliError::InvalidQuery {
                reason: "token_embeddings must be non-empty".to_string(),
            });
        }
        for tok in &self.token_embeddings {
            if tok.len() != expected_dim {
                return Err(ColPaliError::InvalidDimension {
                    expected: expected_dim,
                    got: tok.len(),
                });
            }
        }
        Ok(())
    }
}

/// Which retrieval path produced a given search hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteKind {
    OcrBm25Hybrid,
    ColPaliFp32MaxSim,
    ColPaliBinaryTwoStage,
    HybridRouterTextOnly,
    HybridRouterVisualFused,
}

impl RouteKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OcrBm25Hybrid => "ocr_bm25_hybrid",
            Self::ColPaliFp32MaxSim => "colpali_fp32_maxsim",
            Self::ColPaliBinaryTwoStage => "colpali_binary_two_stage",
            Self::HybridRouterTextOnly => "hybrid_router_text_only",
            Self::HybridRouterVisualFused => "hybrid_router_visual_fused",
        }
    }
}

/// Identifier for each of the four evaluated retrieval strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrategyId {
    OcrBm25HybridBaseline,
    ColPaliFp32MaxSim,
    ColPaliBinaryQuantizedTwoStage,
    HybridTextPlusColPaliRouter,
}

impl StrategyId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OcrBm25HybridBaseline => "ocr_bm25_hybrid_baseline",
            Self::ColPaliFp32MaxSim => "colpali_fp32_maxsim",
            Self::ColPaliBinaryQuantizedTwoStage => "colpali_binary_quantized_two_stage",
            Self::HybridTextPlusColPaliRouter => "hybrid_text_plus_colpali_router",
        }
    }

    pub const fn all() -> [Self; 4] {
        [
            Self::OcrBm25HybridBaseline,
            Self::ColPaliFp32MaxSim,
            Self::ColPaliBinaryQuantizedTwoStage,
            Self::HybridTextPlusColPaliRouter,
        ]
    }
}

/// Ranked page hit returned by any derived retrieval index.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub doc_id: String,
    pub page_no: u32,
    pub version: u64,
    pub tenant_id: String,
    pub acl_group: String,
    pub page_modality: PageModality,
    pub score: f32,
    pub route_used: RouteKind,
}

impl SearchHit {
    pub fn key(&self) -> PageKey {
        PageKey { doc_id: self.doc_id.clone(), page_no: self.page_no }
    }
}

/// Exact memory and disk footprint accounting for a derived index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexFootprint {
    pub ram_bytes: usize,
    pub disk_bytes: usize,
    pub indexed_pages: usize,
    pub visual_pages_indexed: usize,
    pub bytes_per_page_ram: f64,
    pub bytes_per_page_disk: f64,
}

impl IndexFootprint {
    pub fn new(
        ram_bytes: usize,
        disk_bytes: usize,
        indexed_pages: usize,
        visual_pages_indexed: usize,
    ) -> Self {
        let denom = indexed_pages.max(1) as f64;
        Self {
            ram_bytes,
            disk_bytes,
            indexed_pages,
            visual_pages_indexed,
            bytes_per_page_ram: ram_bytes as f64 / denom,
            bytes_per_page_disk: disk_bytes as f64 / denom,
        }
    }
}
