use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use crate::canonical::CanonicalPageStore;
use crate::math::{crc32_ieee, dot_f32, tokenize_text};
use crate::types::{ColPaliError, IndexFootprint, PageModality, RouteKind, SearchHit, VisualQuery};

const MAGIC_OCR_HYBRID: &[u8; 4] = b"HOCR";
const FORMAT_VERSION: u16 = 1;
const BM25_K1: f32 = 1.5;
const BM25_B: f32 = 0.75;

#[derive(Debug, Clone)]
struct OcrPageMeta {
    doc_id: String,
    page_no: u32,
    version: u64,
    tenant_id: String,
    acl_group: String,
    page_modality: PageModality,
    token_len: u32,
}

/// Strategy 1 (`ocr_bm25_hybrid_baseline`): Standard OCR text extraction + BM25 inverted index
/// fused with a single dense OCR embedding per page.
///
/// Strictly derived from [`CanonicalPageStore`]. Enforces fail-closed pre-return tenant/ACL
/// isolation and deterministic binary serialization with CRC32 integrity verification.
#[derive(Debug, Clone)]
pub struct OcrBm25HybridIndex {
    dim: usize,
    avg_doc_len: f32,
    pages: Vec<OcrPageMeta>,
    /// Contiguous flat single-vector OCR embeddings (`pages.len() * dim`).
    dense_vectors: Vec<f32>,
    /// Deterministic inverted posting lists: `term -> Vec<(page_slot, term_freq)>`.
    postings: BTreeMap<String, Vec<(u32, u32)>>,
}

impl OcrBm25HybridIndex {
    /// Build a derived OCR+BM25+Dense hybrid index from canonical page truth.
    pub fn build(store: &CanonicalPageStore) -> Result<Self, ColPaliError> {
        let dim = store.dim();
        let mut pages = Vec::with_capacity(store.len());
        let mut dense_vectors = Vec::with_capacity(store.len().saturating_mul(dim));
        let mut postings: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
        let mut total_tokens: u64 = 0;

        for (slot_idx, rec) in store.iter().enumerate() {
            if rec.ocr_dense_embedding.len() != dim {
                return Err(ColPaliError::InvalidDimension {
                    expected: dim,
                    got: rec.ocr_dense_embedding.len(),
                });
            }
            let tokens = tokenize_text(&rec.ocr_text);
            let token_len = tokens.len() as u32;
            total_tokens += u64::from(token_len);

            let mut tf_map: HashMap<String, u32> = HashMap::new();
            for tok in tokens {
                *tf_map.entry(tok).or_insert(0) += 1;
            }
            let mut tf_sorted: Vec<(String, u32)> = tf_map.into_iter().collect();
            tf_sorted.sort_by(|a, b| a.0.cmp(&b.0));

            let slot_u32 = slot_idx as u32;
            for (term, tf) in tf_sorted {
                postings.entry(term).or_default().push((slot_u32, tf));
            }

            dense_vectors.extend_from_slice(&rec.ocr_dense_embedding);
            pages.push(OcrPageMeta {
                doc_id: rec.doc_id.clone(),
                page_no: rec.page_no,
                version: rec.version,
                tenant_id: rec.tenant_id.clone(),
                acl_group: rec.acl_group.clone(),
                page_modality: rec.page_modality,
                token_len,
            });
        }

        let avg_doc_len = if pages.is_empty() {
            1.0
        } else {
            (total_tokens as f32 / pages.len() as f32).max(1.0)
        };

        Ok(Self { dim, avg_doc_len, pages, dense_vectors, postings })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Execute a hybrid BM25 + single-vector dense OCR query with strict pre-return ACL filtering.
    pub fn search(&self, query: &VisualQuery) -> Result<Vec<SearchHit>, ColPaliError> {
        query.validate(self.dim)?;
        if self.pages.is_empty() {
            return Ok(Vec::new());
        }

        let n_docs = self.pages.len() as f32;
        let query_tokens = tokenize_text(&query.text);
        let mut bm25_scores: HashMap<usize, f32> = HashMap::new();

        for term in &query_tokens {
            if let Some(plist) = self.postings.get(term) {
                let df = plist.len() as f32;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                for &(slot, tf) in plist {
                    let idx = slot as usize;
                    let meta = &self.pages[idx];
                    // Pre-scoring tenant/ACL isolation: unauthorized docs are never accumulated.
                    if !query.scope.permits(&meta.tenant_id, &meta.acl_group) {
                        continue;
                    }
                    let tf_f = tf as f32;
                    let dl = (meta.token_len as f32).max(1.0);
                    let denom = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / self.avg_doc_len));
                    let term_score = idf * ((tf_f * (BM25_K1 + 1.0)) / denom);
                    *bm25_scores.entry(idx).or_insert(0.0) += term_score;
                }
            }
        }

        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (idx, meta) in self.pages.iter().enumerate() {
            // Strict pre-return tenant & ACL filter.
            if !query.scope.permits(&meta.tenant_id, &meta.acl_group) {
                continue;
            }
            let vec_start = idx * self.dim;
            let dense_sim = dot_f32(
                &query.dense_embedding,
                &self.dense_vectors[vec_start..vec_start + self.dim],
            );
            let dense_norm = ((dense_sim + 1.0) * 0.5).clamp(0.0, 1.0);
            let bm25_raw = bm25_scores.get(&idx).copied().unwrap_or(0.0);
            let bm25_norm = bm25_raw / (bm25_raw + 5.0);
            // Weighted calibrated fusion in [0.0, 1.0].
            let fused = 0.62 * bm25_norm + 0.38 * dense_norm;
            scored.push((idx, fused));
        }

        scored.sort_by(|&(ia, sa), &(ib, sb)| {
            sb.total_cmp(&sa)
                .then_with(|| self.pages[ia].doc_id.cmp(&self.pages[ib].doc_id))
                .then_with(|| self.pages[ia].page_no.cmp(&self.pages[ib].page_no))
        });
        scored.truncate(query.top_k);

        Ok(scored
            .into_iter()
            .map(|(idx, score)| {
                let meta = &self.pages[idx];
                SearchHit {
                    doc_id: meta.doc_id.clone(),
                    page_no: meta.page_no,
                    version: meta.version,
                    tenant_id: meta.tenant_id.clone(),
                    acl_group: meta.acl_group.clone(),
                    page_modality: meta.page_modality,
                    score,
                    route_used: RouteKind::OcrBm25Hybrid,
                }
            })
            .collect())
    }

    /// Raw BM25 top score diagnostic used by the selective visual router to detect
    /// high-confidence exact lexical hits on plain-text pages.
    pub(crate) fn top_bm25_lexical_signal(&self, query: &VisualQuery) -> Result<f32, ColPaliError> {
        query.validate(self.dim)?;
        if self.pages.is_empty() {
            return Ok(0.0);
        }
        let n_docs = self.pages.len() as f32;
        let query_tokens = tokenize_text(&query.text);
        let mut bm25_scores: HashMap<usize, f32> = HashMap::new();
        for term in &query_tokens {
            if let Some(plist) = self.postings.get(term) {
                let df = plist.len() as f32;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                for &(slot, tf) in plist {
                    let idx = slot as usize;
                    let meta = &self.pages[idx];
                    if !query.scope.permits(&meta.tenant_id, &meta.acl_group) {
                        continue;
                    }
                    let tf_f = tf as f32;
                    let dl = (meta.token_len as f32).max(1.0);
                    let denom = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / self.avg_doc_len));
                    *bm25_scores.entry(idx).or_insert(0.0) +=
                        idf * ((tf_f * (BM25_K1 + 1.0)) / denom);
                }
            }
        }
        let best = bm25_scores
            .into_values()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);
        Ok(best)
    }

    /// Deterministic binary serialization with trailing IEEE CRC32 checksum.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_OCR_HYBRID);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&self.avg_doc_len.to_le_bytes());
        buf.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        for p in &self.pages {
            write_str(&mut buf, &p.doc_id);
            buf.extend_from_slice(&p.page_no.to_le_bytes());
            buf.extend_from_slice(&p.version.to_le_bytes());
            write_str(&mut buf, &p.tenant_id);
            write_str(&mut buf, &p.acl_group);
            buf.push(p.page_modality.tag());
            buf.extend_from_slice(&p.token_len.to_le_bytes());
        }
        for &v in &self.dense_vectors {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&(self.postings.len() as u32).to_le_bytes());
        for (term, plist) in &self.postings {
            write_str(&mut buf, term);
            buf.extend_from_slice(&(plist.len() as u32).to_le_bytes());
            for &(slot, tf) in plist {
                buf.extend_from_slice(&slot.to_le_bytes());
                buf.extend_from_slice(&tf.to_le_bytes());
            }
        }
        let crc = crc32_ieee(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Deserialize with strict magic, version, bounds, and CRC32 integrity verification.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ColPaliError> {
        if bytes.len() < 18 {
            return Err(ColPaliError::CorruptedIndex {
                reason: "HOCR buffer truncated".to_string(),
            });
        }
        let payload_len = bytes.len() - 4;
        let expected_crc = u32::from_le_bytes(bytes[payload_len..].try_into().map_err(|_| {
            ColPaliError::CorruptedIndex { reason: "missing CRC32 trailer".to_string() }
        })?);
        let actual_crc = crc32_ieee(&bytes[..payload_len]);
        if expected_crc != actual_crc {
            return Err(ColPaliError::CorruptedIndex {
                reason: format!(
                    "HOCR CRC32 mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
                ),
            });
        }

        let mut cursor = ByteCursor::new(&bytes[..payload_len]);
        let magic = cursor.read_exact(4)?;
        if magic != MAGIC_OCR_HYBRID {
            return Err(ColPaliError::CorruptedIndex {
                reason: "invalid HOCR magic header".to_string(),
            });
        }
        let version = cursor.read_u16()?;
        if version != FORMAT_VERSION {
            return Err(ColPaliError::CorruptedIndex {
                reason: format!("unsupported HOCR version {version}"),
            });
        }
        let dim = cursor.read_u32()? as usize;
        let avg_doc_len = cursor.read_f32()?;
        let page_count = cursor.read_u32()? as usize;

        let mut pages = Vec::with_capacity(page_count);
        for _ in 0..page_count {
            let doc_id = cursor.read_str()?;
            let page_no = cursor.read_u32()?;
            let rec_version = cursor.read_u64()?;
            let tenant_id = cursor.read_str()?;
            let acl_group = cursor.read_str()?;
            let tag = cursor.read_u8()?;
            let page_modality = PageModality::from_tag(tag).ok_or_else(|| {
                ColPaliError::CorruptedIndex { reason: format!("invalid modality tag {tag}") }
            })?;
            let token_len = cursor.read_u32()?;
            pages.push(OcrPageMeta {
                doc_id,
                page_no,
                version: rec_version,
                tenant_id,
                acl_group,
                page_modality,
                token_len,
            });
        }

        let vec_floats = page_count.checked_mul(dim).ok_or_else(|| {
            ColPaliError::CorruptedIndex { reason: "dense vector size overflow".to_string() }
        })?;
        let mut dense_vectors = Vec::with_capacity(vec_floats);
        for _ in 0..vec_floats {
            dense_vectors.push(cursor.read_f32()?);
        }

        let term_count = cursor.read_u32()? as usize;
        let mut postings = BTreeMap::new();
        for _ in 0..term_count {
            let term = cursor.read_str()?;
            let plist_len = cursor.read_u32()? as usize;
            let mut plist = Vec::with_capacity(plist_len);
            for _ in 0..plist_len {
                let slot = cursor.read_u32()?;
                let tf = cursor.read_u32()?;
                plist.push((slot, tf));
            }
            postings.insert(term, plist);
        }

        Ok(Self { dim, avg_doc_len, pages, dense_vectors, postings })
    }

    /// Deterministic CRC32 checksum of the serialized index bytes.
    pub fn checksum_crc32(&self) -> u32 {
        crc32_ieee(&self.to_bytes())
    }

    /// Exact RAM and serialized disk footprint accounting.
    pub fn footprint(&self) -> IndexFootprint {
        let meta_bytes: usize = self
            .pages
            .iter()
            .map(|p| {
                p.doc_id.len()
                    + p.tenant_id.len()
                    + p.acl_group.len()
                    + std::mem::size_of::<OcrPageMeta>()
            })
            .sum();
        let dense_bytes = self.dense_vectors.len() * std::mem::size_of::<f32>();
        let postings_bytes: usize = self
            .postings
            .iter()
            .map(|(k, v)| k.len() + 24 + v.len() * std::mem::size_of::<(u32, u32)>())
            .sum();
        let ram_bytes = meta_bytes + dense_bytes + postings_bytes;
        let disk_bytes = self.to_bytes().len();
        IndexFootprint::new(ram_bytes, disk_bytes, self.pages.len(), 0)
    }
}

pub(crate) fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

pub(crate) struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn read_exact(&mut self, len: usize) -> Result<&'a [u8], ColPaliError> {
        let end = self.pos.checked_add(len).ok_or_else(|| ColPaliError::CorruptedIndex {
            reason: "cursor offset overflow".to_string(),
        })?;
        if end > self.data.len() {
            return Err(ColPaliError::CorruptedIndex {
                reason: "unexpected EOF while decoding derived index".to_string(),
            });
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8, ColPaliError> {
        Ok(self.read_exact(1)?[0])
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16, ColPaliError> {
        let b = self.read_exact(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, ColPaliError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64, ColPaliError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32, ColPaliError> {
        let b = self.read_exact(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn read_str(&mut self) -> Result<String, ColPaliError> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ColPaliError::CorruptedIndex {
            reason: "invalid UTF-8 string in derived index".to_string(),
        })
    }
}
