use std::collections::HashMap;

use crate::canonical::CanonicalPageStore;
use crate::colpali_index::ColPaliBinaryTwoStageIndex;
use crate::math::{crc32_ieee, tokenize_text};
use crate::ocr_hybrid::OcrBm25HybridIndex;
use crate::types::{
    CanonicalPageRecord, ColPaliError, IndexFootprint, PageKey, RouteKind, SearchHit, VisualQuery,
};

/// Strategy 4 (`hybrid_text_plus_colpali_router`): Selective visual document retrieval router.
///
/// Architectural principle for Hanzo Index:
/// - **Ingest-time selectivity**: Every page is indexed in the compact `OcrBm25HybridIndex`.
///   ONLY pages whose observable PDF/layout complexity (`PageLayoutFeatures::visual_complexity_score`)
///   meets or exceeds `visual_complexity_threshold` are indexed in `ColPaliBinaryTwoStageIndex`.
///   Plain-text pages incur **zero** multi-vector visual patch storage!
/// - **Query-time selectivity**: Pure exact-identifier/code-literal queries with strong lexical
///   BM25 confidence and no visual-intent cues fast-path through `OcrBm25HybridIndex` without
///   invoking visual MaxSim; visual/analytical/mixed queries fuse authorized text and selective
///   visual hits on a calibrated score scale.
#[derive(Debug, Clone)]
pub struct HybridVisualRouterIndex {
    dim: usize,
    visual_complexity_threshold: f32,
    text_index: OcrBm25HybridIndex,
    visual_index: ColPaliBinaryTwoStageIndex,
    total_pages: usize,
    visual_pages_indexed: usize,
}

impl HybridVisualRouterIndex {
    /// Default ingest-time visual complexity threshold separating plain-text pages from
    /// table/figure/formula-heavy pages using observable layout heuristics only.
    pub const DEFAULT_VISUAL_THRESHOLD: f32 = 0.28;

    /// Build a selective hybrid router index from canonical page truth.
    pub fn build(store: &CanonicalPageStore) -> Result<Self, ColPaliError> {
        Self::build_with_threshold(store, Self::DEFAULT_VISUAL_THRESHOLD)
    }

    pub fn build_with_threshold(
        store: &CanonicalPageStore,
        visual_complexity_threshold: f32,
    ) -> Result<Self, ColPaliError> {
        let text_index = OcrBm25HybridIndex::build(store)?;
        let visual_records: Vec<&CanonicalPageRecord> = store
            .iter()
            .filter(|rec| {
                rec.layout_features.visual_complexity_score() >= visual_complexity_threshold
            })
            .collect();
        let visual_pages_indexed = visual_records.len();
        let visual_index = ColPaliBinaryTwoStageIndex::build_from_iter(
            store.dim(),
            visual_records.into_iter(),
            ColPaliBinaryTwoStageIndex::DEFAULT_SHORTLIST_MIN,
            ColPaliBinaryTwoStageIndex::DEFAULT_SHORTLIST_MULT,
        )?;
        Ok(Self {
            dim: store.dim(),
            visual_complexity_threshold,
            text_index,
            visual_index,
            total_pages: store.len(),
            visual_pages_indexed,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    pub fn visual_pages_indexed(&self) -> usize {
        self.visual_pages_indexed
    }

    pub fn visual_indexing_ratio(&self) -> f64 {
        if self.total_pages == 0 {
            0.0
        } else {
            self.visual_pages_indexed as f64 / self.total_pages as f64
        }
    }

    /// Execute selective routing with strict fail-closed pre-return tenant/ACL enforcement.
    pub fn search(&self, query: &VisualQuery) -> Result<Vec<SearchHit>, ColPaliError> {
        query.validate(self.dim)?;
        if self.total_pages == 0 {
            return Ok(Vec::new());
        }

        let lexical_exact_cue = query_has_exact_identifier_cue(&query.text);
        let visual_intent_cue = query_has_visual_intent_cue(&query.text);

        // Fetch broadened candidate set from cheap OCR+BM25 hybrid index.
        let mut text_query = query.clone();
        text_query.top_k = (query.top_k * 3).max(16);
        let text_hits = self.text_index.search(&text_query)?;

        // Fast-path: exact identifier / literal keyword query without visual cues and with
        // strong BM25 lexical signal skips visual MaxSim completely.
        if lexical_exact_cue && !visual_intent_cue {
            let top_bm25 = self.text_index.top_bm25_lexical_signal(query)?;
            if top_bm25 >= 6.0 || self.visual_index.is_empty() {
                return Ok(text_hits
                    .into_iter()
                    .take(query.top_k)
                    .map(|mut hit| {
                        hit.route_used = RouteKind::HybridRouterTextOnly;
                        hit
                    })
                    .collect());
            }
        }

        if self.visual_index.is_empty() {
            return Ok(text_hits
                .into_iter()
                .take(query.top_k)
                .map(|mut hit| {
                    hit.route_used = RouteKind::HybridRouterTextOnly;
                    hit
                })
                .collect());
        }

        // Otherwise also query the selectively-indexed visual pages using fast 1-bit Binary + SQ8.
        let mut visual_query = query.clone();
        visual_query.top_k = query.top_k.saturating_mul(2).max(8);
        let visual_hits = self.visual_index.search(&visual_query)?;

        // Calibrated merge keyed by (doc_id, page_no).
        // Both text_hits and visual_hits are already pre-filtered by `query.scope`.
        let mut merged: HashMap<PageKey, SearchHit> = HashMap::new();
        for mut hit in text_hits {
            hit.route_used = RouteKind::HybridRouterTextOnly;
            merged.insert(hit.key(), hit);
        }

        // Visual weight boosts visual MaxSim slightly when visual cues are present,
        // while keeping clean text hits competitive on non-visual queries.
        let visual_bias = if visual_intent_cue { 1.08 } else { 0.96 };
        for vhit in visual_hits {
            // Map ColPali MaxSim (typically [0.0, 1.0] for matching unit vectors) to calibrated [0.0, 1.0].
            let calibrated_visual = (vhit.score.clamp(-1.0, 1.0) * visual_bias).clamp(0.0, 1.25);
            let key = vhit.key();
            match merged.get_mut(&key) {
                Some(existing) => {
                    // Combine text evidence and visual MaxSim for visual-indexed pages.
                    let combined = existing.score.max(calibrated_visual)
                        + 0.12 * existing.score.min(calibrated_visual);
                    existing.score = combined;
                    existing.route_used = RouteKind::HybridRouterVisualFused;
                }
                None => {
                    merged.insert(
                        key,
                        SearchHit {
                            doc_id: vhit.doc_id,
                            page_no: vhit.page_no,
                            version: vhit.version,
                            tenant_id: vhit.tenant_id,
                            acl_group: vhit.acl_group,
                            page_modality: vhit.page_modality,
                            score: calibrated_visual,
                            route_used: RouteKind::HybridRouterVisualFused,
                        },
                    );
                }
            }
        }

        let mut results: Vec<SearchHit> = merged.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
                .then_with(|| a.page_no.cmp(&b.page_no))
        });
        results.truncate(query.top_k);
        Ok(results)
    }

    /// Deterministic combined CRC32 checksum across text + selective visual derived indexes.
    pub fn checksum_crc32(&self) -> u32 {
        let mut combined = Vec::new();
        combined.extend_from_slice(&self.visual_complexity_threshold.to_le_bytes());
        combined.extend_from_slice(&self.text_index.checksum_crc32().to_le_bytes());
        combined.extend_from_slice(&self.visual_index.checksum_crc32().to_le_bytes());
        crc32_ieee(&combined)
    }

    /// Exact combined RAM and serialized disk footprint (text index + selective visual index only).
    pub fn footprint(&self) -> IndexFootprint {
        let t = self.text_index.footprint();
        let v = self.visual_index.footprint();
        IndexFootprint::new(
            t.ram_bytes + v.ram_bytes,
            t.disk_bytes + v.disk_bytes,
            self.total_pages,
            self.visual_pages_indexed,
        )
    }
}

/// Pure query-text heuristic detecting exact code/config/identifier tokens (snake_case,
/// `::` Rust paths, `ERR_`/`HIP-`/`CFG-` codes, `--flag` CLI flags, dotted crate paths).
pub(crate) fn query_has_exact_identifier_cue(text: &str) -> bool {
    if text.contains("::") || text.contains("--") || text.contains("0x") {
        return true;
    }
    for tok in tokenize_text(text) {
        let has_underscore = tok.contains('_');
        let has_digit_hyphen = tok.contains('-') && tok.chars().any(|c| c.is_ascii_digit());
        let has_dot_ident = tok.contains('.') && tok.chars().any(|c| c.is_ascii_alphabetic());
        if has_underscore || has_digit_hyphen || has_dot_ident {
            return true;
        }
    }
    false
}

/// Pure query-text heuristic detecting table/chart/figure/formula visual-layout intent.
pub(crate) fn query_has_visual_intent_cue(text: &str) -> bool {
    const VISUAL_KEYWORDS: &[&str] = &[
        "table",
        "column",
        "row",
        "cell",
        "matrix",
        "figure",
        "chart",
        "plot",
        "graph",
        "axis",
        "legend",
        "curve",
        "pareto",
        "diagram",
        "bar",
        "scatter",
        "slope",
        "crossover",
        "formula",
        "equation",
        "bound",
        "theorem",
        "integral",
        "summation",
        "sigma",
        "lambda",
        "proof",
        "ratio",
    ];
    for tok in tokenize_text(text) {
        if VISUAL_KEYWORDS.contains(&tok.as_str()) {
            return true;
        }
    }
    false
}
