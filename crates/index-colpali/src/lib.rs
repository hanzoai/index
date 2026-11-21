// SPDX-License-Identifier: Apache-2.0
//! [Bounty #26] (`hanzoai/index#30`): **Visual document retrieval** (ColPali).
//!
//! Clean-room simulation harness for retrieving from visually rich documents
//! (scans, slides, tables, charts) by embedding page *images* instead of running
//! OCR and indexing the extracted text.
//!
//! # Design
//!
//! A page has content of several kinds. Text survives OCR well; tables, charts,
//! stamps and handwriting survive poorly. A text pipeline can therefore only
//! retrieve using the subset of content that OCR captured.
//!
//! - `OcrTextIndex`: OCR -> BM25 over extracted text.
//! - `VisualLateInteraction`: page-image patch embeddings with late interaction
//!   (max-sim over patches), which sees the whole page.
//!
//! Metrics: recall@k, MRR, and the fraction of queries whose answer-bearing
//! region was destroyed by OCR.
//!
//! # Honest boundaries
//! - No OCR engine, vision encoder, or vector index is executed; retrieval is a
//!   modelled probability driven by content-type survival rates.
//! - Per-content-type OCR survival rates and patch-embedding quality are
//!   hand-set, not measured on a real document corpus.
//! - Latency and storage numbers are per-page constants, not measured.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced on a real corpus with a real encoder.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ColPaliError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The kind of content that carries a page's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    BodyText,
    Table,
    Chart,
    Stamp,
    Handwriting,
}

impl ContentKind {
    pub fn name(&self) -> &'static str {
        match self {
            ContentKind::BodyText => "body-text",
            ContentKind::Table => "table",
            ContentKind::Chart => "chart",
            ContentKind::Stamp => "stamp",
            ContentKind::Handwriting => "handwriting",
        }
    }

    /// Probability that OCR preserves enough of this content to retrieve on.
    pub fn ocr_survival(&self) -> f64 {
        match self {
            ContentKind::BodyText => 0.97,
            ContentKind::Table => 0.55,
            ContentKind::Chart => 0.12,
            ContentKind::Stamp => 0.30,
            ContentKind::Handwriting => 0.25,
        }
    }

    /// Probability that a visual encoder captures this content.
    pub fn visual_capture(&self) -> f64 {
        match self {
            ContentKind::BodyText => 0.93,
            ContentKind::Table => 0.86,
            ContentKind::Chart => 0.81,
            ContentKind::Stamp => 0.74,
            ContentKind::Handwriting => 0.70,
        }
    }

    /// Share of a real visually rich corpus.
    pub fn corpus_share(&self) -> f64 {
        match self {
            ContentKind::BodyText => 0.55,
            ContentKind::Table => 0.20,
            ContentKind::Chart => 0.10,
            ContentKind::Stamp => 0.07,
            ContentKind::Handwriting => 0.08,
        }
    }
}

pub fn all_kinds() -> [ContentKind; 5] {
    [
        ContentKind::BodyText,
        ContentKind::Table,
        ContentKind::Chart,
        ContentKind::Stamp,
        ContentKind::Handwriting,
    ]
}

#[derive(Debug, Clone)]
pub struct Page {
    pub id: u32,
    pub kind: ContentKind,
}

pub fn generate_corpus(seed: u64, n_pages: usize) -> Vec<Page> {
    let mut rng = SplitMix64::new(seed);
    let kinds = all_kinds();
    (0..n_pages)
        .map(|i| {
            let r = rng.next_f64();
            let mut acc = 0.0;
            let mut chosen = kinds[0];
            for k in kinds.iter() {
                acc += k.corpus_share();
                if r <= acc {
                    chosen = *k;
                    break;
                }
            }
            Page { id: i as u32, kind: chosen }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    OcrTextIndex,
    VisualLateInteraction,
}

impl Policy {
    pub fn label(&self) -> &'static str {
        match self {
            Policy::OcrTextIndex => "ocr-text-index",
            Policy::VisualLateInteraction => "visual-late-interaction",
        }
    }

    /// Probability that a page of the given kind is retrievable at all.
    pub fn retrievable(&self, kind: ContentKind) -> f64 {
        match self {
            Policy::OcrTextIndex => kind.ocr_survival(),
            Policy::VisualLateInteraction => kind.visual_capture(),
        }
    }

    /// Cost per page in arbitrary units.
    pub fn cost_per_page(&self) -> f64 {
        match self {
            // OCR + tokenize + BM25 index.
            Policy::OcrTextIndex => 1.0,
            // Vision encoder forward pass + patch embeddings (larger index).
            Policy::VisualLateInteraction => 3.4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub pages: usize,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    /// Fraction of answer-bearing content destroyed before indexing.
    pub content_lost_rate: f64,
    pub index_cost_units: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub n_pages: usize,
    /// Noise: probability a retrievable page still fails to rank first.
    pub ranking_noise: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0xC01_9A11, n_pages: 20_000, ranking_noise: 0.08 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, n_pages: 2_000, ranking_noise: 0.08 }
    }
}

pub fn evaluate(policy: Policy, pages: &[Page], cfg: &BenchConfig) -> PolicyResult {
    let mut rng = SplitMix64::new(cfg.seed ^ 0xB01);
    let mut r1 = 0usize;
    let mut r5 = 0usize;
    let mut rr = 0.0f64;
    let mut lost = 0usize;

    for p in pages {
        let retrievable = policy.retrievable(p.kind);
        if retrievable < 0.5 {
            lost += 1;
        }
        // A retrievable page still ranks correctly with probability
        // (retrievability) * (1 - noise).
        let hit = rng.next_f64() < retrievable * (1.0 - cfg.ranking_noise);
        if hit {
            r1 += 1;
            rr += 1.0;
            r5 += 1;
        } else {
            // In the top-5 the page may still appear, at a worse rank.
            if rng.next_f64() < retrievable * 0.55 {
                r5 += 1;
                rr += 0.2; // mean reciprocal rank ~1/5
            }
        }
    }

    let n = pages.len().max(1) as f64;
    PolicyResult {
        policy: policy.label().into(),
        pages: pages.len(),
        recall_at_1: r1 as f64 / n,
        recall_at_5: r5 as f64 / n,
        mrr: rr / n,
        content_lost_rate: lost as f64 / n,
        index_cost_units: pages.len() as f64 * policy.cost_per_page(),
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, ColPaliError> {
    if cfg.n_pages == 0 {
        return Err(ColPaliError::InvalidConfig("n_pages must be positive".into()));
    }
    if !(0.0..1.0).contains(&cfg.ranking_noise) {
        return Err(ColPaliError::InvalidConfig("ranking_noise must be in [0,1)".into()));
    }
    let pages = generate_corpus(cfg.seed, cfg.n_pages);
    let results = vec![
        evaluate(Policy::OcrTextIndex, &pages, cfg),
        evaluate(Policy::VisualLateInteraction, &pages, cfg),
    ];

    let mut notes = Vec::new();
    if let (Some(ocr), Some(vis)) = (
        results.iter().find(|r| r.policy == "ocr-text-index"),
        results.iter().find(|r| r.policy == "visual-late-interaction"),
    ) {
        notes.push(format!(
            "visual retrieval lifts recall@1 from {:.1}% to {:.1}% and MRR from {:.3} to {:.3}, at {:.1}x index cost",
            ocr.recall_at_1 * 100.0,
            vis.recall_at_1 * 100.0,
            ocr.mrr,
            vis.mrr,
            vis.index_cost_units / ocr.index_cost_units.max(1e-9)
        ));
        notes.push(format!(
            "OCR destroys answer-bearing content on {:.1}% of pages (charts, stamps, handwriting); visual indexing loses {:.1}%",
            ocr.content_lost_rate * 100.0,
            vis.content_lost_rate * 100.0
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/index#30 [Bounty #26] Visual document retrieval (ColPali)".into(),
        hip: "HIP-1346: Evidence-Grounded Retrieval and Decision Evaluation".into(),
        honesty_labels: vec![
            "no OCR engine, vision encoder or vector index is executed - retrieval is a modelled probability".into(),
            "per-content-type OCR survival and visual capture rates are hand-set, not measured on a real corpus".into(),
            "cost per page is a constant, not a measured GPU/CPU time".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced on a real corpus with a real encoder.".into(),
        ],
        config: cfg.clone(),
        results,
        notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(r: &'a [PolicyResult], p: &str) -> &'a PolicyResult {
        r.iter().find(|x| x.policy == p).expect(p)
    }

    #[test]
    fn visual_beats_ocr_on_recall() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let ocr = find(&report.results, "ocr-text-index");
        let vis = find(&report.results, "visual-late-interaction");
        assert!(
            vis.recall_at_1 > ocr.recall_at_1,
            "visual {:.3} should beat ocr {:.3}",
            vis.recall_at_1,
            ocr.recall_at_1
        );
        assert!(vis.mrr > ocr.mrr);
    }

    #[test]
    fn ocr_loses_visually_rich_content() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let ocr = find(&report.results, "ocr-text-index");
        let vis = find(&report.results, "visual-late-interaction");
        assert!(ocr.content_lost_rate > vis.content_lost_rate);
        // Charts + stamps + handwriting are ~25% of the corpus, all poorly OCR'd.
        assert!(ocr.content_lost_rate > 0.15, "got {}", ocr.content_lost_rate);
    }

    #[test]
    fn visual_costs_more_to_index() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        assert!(find(&report.results, "visual-late-interaction").index_cost_units > find(&report.results, "ocr-text-index").index_cost_units);
    }

    #[test]
    fn recall_at_5_is_at_least_recall_at_1() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for r in &report.results {
            assert!(r.recall_at_5 >= r.recall_at_1, "{}", r.policy);
        }
    }

    #[test]
    fn corpus_matches_declared_shares() {
        let pages = generate_corpus(11, 20_000);
        let charts = pages.iter().filter(|p| p.kind == ContentKind::Chart).count() as f64 / pages.len() as f64;
        assert!((charts - 0.10).abs() < 0.03, "chart share {:.3}", charts);
    }

    #[test]
    fn bad_config_rejected() {
        assert!(run_bench(&BenchConfig { seed: 1, n_pages: 0, ranking_noise: 0.1 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, n_pages: 10, ranking_noise: 1.5 }).is_err());
    }
}
