use crate::canonical::CanonicalPageStore;
use crate::math::{
    binary_similarity_128, binary_similarity_words, crc32_ieee, dot_f32, fwht_orthogonal,
    pack_binary_words, Sq8Vector,
};
use crate::ocr_hybrid::{write_str, ByteCursor};
use crate::types::{
    CanonicalPageRecord, ColPaliError, IndexFootprint, PageModality, RouteKind, SearchHit,
    VisualQuery,
};

const MAGIC_COLPALI_FP32: &[u8; 4] = b"HCFP";
const MAGIC_COLPALI_BIN2: &[u8; 4] = b"HCB2";
const FORMAT_VERSION: u16 = 1;

#[derive(Debug, Clone)]
struct VisualPageMeta {
    doc_id: String,
    page_no: u32,
    version: u64,
    tenant_id: String,
    acl_group: String,
    page_modality: PageModality,
    patch_offset: usize,
    patch_count: usize,
}

/// Strategy 2 (`colpali_fp32_maxsim`): Exact FP32 ColPali late-interaction multi-vector index.
///
/// Computes exact `MaxSim(Q, P) = (1 / |Q|) * sum_{i in Q} max_{j in P} <q_i, p_j>` across all
/// visual page patches of authorized pages. Serves as the high-accuracy, high-storage/latency
/// reference point against 1-bit binary quantization and selective visual routing.
#[derive(Debug, Clone)]
pub struct ColPaliFp32Index {
    dim: usize,
    pages: Vec<VisualPageMeta>,
    /// Contiguous FP32 patch slab: `total_patches * dim` floats.
    patch_slab: Vec<f32>,
    total_patches: usize,
}

impl ColPaliFp32Index {
    /// Build an exact FP32 ColPali multi-vector index from canonical page truth.
    pub fn build(store: &CanonicalPageStore) -> Result<Self, ColPaliError> {
        Self::build_from_iter(store.dim(), store.iter())
    }

    pub(crate) fn build_from_iter<'a>(
        dim: usize,
        records: impl Iterator<Item = &'a CanonicalPageRecord>,
    ) -> Result<Self, ColPaliError> {
        let mut pages = Vec::new();
        let mut patch_slab = Vec::new();
        let mut total_patches = 0usize;

        for rec in records {
            if rec.patch_embeddings.is_empty() {
                return Err(ColPaliError::EmptyPatchSet {
                    doc_id: rec.doc_id.clone(),
                    page_no: rec.page_no,
                });
            }
            let patch_offset = total_patches;
            let patch_count = rec.patch_embeddings.len();
            for patch in &rec.patch_embeddings {
                if patch.len() != dim {
                    return Err(ColPaliError::InvalidDimension { expected: dim, got: patch.len() });
                }
                patch_slab.extend_from_slice(patch);
            }
            total_patches = total_patches.checked_add(patch_count).ok_or_else(|| {
                ColPaliError::CorruptedIndex { reason: "patch count overflow".to_string() }
            })?;
            pages.push(VisualPageMeta {
                doc_id: rec.doc_id.clone(),
                page_no: rec.page_no,
                version: rec.version,
                tenant_id: rec.tenant_id.clone(),
                acl_group: rec.acl_group.clone(),
                page_modality: rec.page_modality,
                patch_offset,
                patch_count,
            });
        }

        Ok(Self { dim, pages, patch_slab, total_patches })
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

    pub fn total_patches(&self) -> usize {
        self.total_patches
    }

    /// Execute exact FP32 late-interaction MaxSim search with pre-scoring tenant/ACL isolation.
    pub fn search(&self, query: &VisualQuery) -> Result<Vec<SearchHit>, ColPaliError> {
        query.validate(self.dim)?;
        if self.pages.is_empty() {
            return Ok(Vec::new());
        }

        let inv_q = 1.0 / (query.token_embeddings.len() as f32);
        let mut scored: Vec<(usize, f32)> = Vec::new();

        for (idx, meta) in self.pages.iter().enumerate() {
            // Strict pre-scoring tenant/ACL isolation.
            if !query.scope.permits(&meta.tenant_id, &meta.acl_group) {
                continue;
            }
            let mut maxsim_sum = 0.0f32;
            let base_float = meta.patch_offset * self.dim;
            for q_tok in &query.token_embeddings {
                let mut best = f32::NEG_INFINITY;
                for p in 0..meta.patch_count {
                    let start = base_float + p * self.dim;
                    let sim = dot_f32(q_tok, &self.patch_slab[start..start + self.dim]);
                    if sim > best {
                        best = sim;
                    }
                }
                maxsim_sum += best;
            }
            scored.push((idx, maxsim_sum * inv_q));
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
                    route_used: RouteKind::ColPaliFp32MaxSim,
                }
            })
            .collect())
    }

    /// Deterministic binary serialization (`HCFP` + trailing CRC32).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_COLPALI_FP32);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.total_patches as u64).to_le_bytes());
        for p in &self.pages {
            write_str(&mut buf, &p.doc_id);
            buf.extend_from_slice(&p.page_no.to_le_bytes());
            buf.extend_from_slice(&p.version.to_le_bytes());
            write_str(&mut buf, &p.tenant_id);
            write_str(&mut buf, &p.acl_group);
            buf.push(p.page_modality.tag());
            buf.extend_from_slice(&(p.patch_offset as u64).to_le_bytes());
            buf.extend_from_slice(&(p.patch_count as u32).to_le_bytes());
        }
        for &x in &self.patch_slab {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        let crc = crc32_ieee(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Deserialize with strict magic, bounds, and CRC32 verification.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ColPaliError> {
        if bytes.len() < 26 {
            return Err(ColPaliError::CorruptedIndex {
                reason: "HCFP buffer truncated".to_string(),
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
                    "HCFP CRC32 mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
                ),
            });
        }

        let mut cursor = ByteCursor::new(&bytes[..payload_len]);
        if cursor.read_exact(4)? != MAGIC_COLPALI_FP32 {
            return Err(ColPaliError::CorruptedIndex { reason: "invalid HCFP magic".to_string() });
        }
        let version = cursor.read_u16()?;
        if version != FORMAT_VERSION {
            return Err(ColPaliError::CorruptedIndex {
                reason: format!("unsupported HCFP version {version}"),
            });
        }
        let dim = cursor.read_u32()? as usize;
        let page_count = cursor.read_u32()? as usize;
        let total_patches = cursor.read_u64()? as usize;

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
            let patch_offset = cursor.read_u64()? as usize;
            let patch_count = cursor.read_u32()? as usize;
            let end_patch = patch_offset.checked_add(patch_count).ok_or_else(|| {
                ColPaliError::CorruptedIndex { reason: "patch range overflow".to_string() }
            })?;
            if end_patch > total_patches {
                return Err(ColPaliError::CorruptedIndex {
                    reason: "patch range exceeds slab".to_string(),
                });
            }
            pages.push(VisualPageMeta {
                doc_id,
                page_no,
                version: rec_version,
                tenant_id,
                acl_group,
                page_modality,
                patch_offset,
                patch_count,
            });
        }

        let total_floats = total_patches.checked_mul(dim).ok_or_else(|| {
            ColPaliError::CorruptedIndex { reason: "FP32 slab size overflow".to_string() }
        })?;
        let mut patch_slab = Vec::with_capacity(total_floats);
        for _ in 0..total_floats {
            patch_slab.push(cursor.read_f32()?);
        }

        Ok(Self { dim, pages, patch_slab, total_patches })
    }

    pub fn checksum_crc32(&self) -> u32 {
        crc32_ieee(&self.to_bytes())
    }

    pub fn footprint(&self) -> IndexFootprint {
        let meta_bytes: usize = self
            .pages
            .iter()
            .map(|p| {
                p.doc_id.len()
                    + p.tenant_id.len()
                    + p.acl_group.len()
                    + std::mem::size_of::<VisualPageMeta>()
            })
            .sum();
        let slab_bytes = self.patch_slab.len() * std::mem::size_of::<f32>();
        let ram_bytes = meta_bytes + slab_bytes;
        let disk_bytes = self.to_bytes().len();
        IndexFootprint::new(ram_bytes, disk_bytes, self.pages.len(), self.pages.len())
    }
}

/// Strategy 3 (`colpali_binary_quantized_two_stage`):
///
/// - **Stage 1 (Hot RAM)**: 1-bit binary patch signatures packed into `u64` words after
///   orthogonal Fast Walsh-Hadamard rotation (`32.0x` patch vector memory reduction vs FP32:
///   `16 B/patch` vs `512 B/patch` at `D = 128`), scored with hardware `XOR + POPCNT` MaxSim
///   to shortlist top-`M` authorized pages.
/// - **Stage 2 (Cold Disk/Object Slab)**: SQ8 (`i8` + per-patch `f32` scale) MaxSim rerank
///   executed ONLY on the `M` shortlisted authorized pages (`~3.88x` smaller than FP32 on disk).
#[derive(Debug, Clone)]
pub struct ColPaliBinaryTwoStageIndex {
    dim: usize,
    words_per_patch: usize,
    stage1_shortlist_min: usize,
    stage1_shortlist_mult: usize,
    pages: Vec<VisualPageMeta>,
    /// Hot RAM: 1-bit binary patch signatures (`total_patches * words_per_patch` `u64` words).
    binary_slab: Vec<u64>,
    /// Cold Stage-2 rerank slab: SQ8 patches (`total_patches` entries).
    sq8_slab: Vec<Sq8Vector>,
    total_patches: usize,
}

impl ColPaliBinaryTwoStageIndex {
    pub const DEFAULT_SHORTLIST_MIN: usize = 24;
    pub const DEFAULT_SHORTLIST_MULT: usize = 3;

    /// Build a 2-stage binary + SQ8 derived ColPali index from canonical page truth.
    pub fn build(store: &CanonicalPageStore) -> Result<Self, ColPaliError> {
        Self::build_from_iter(
            store.dim(),
            store.iter(),
            Self::DEFAULT_SHORTLIST_MIN,
            Self::DEFAULT_SHORTLIST_MULT,
        )
    }

    pub(crate) fn build_from_iter<'a>(
        dim: usize,
        records: impl Iterator<Item = &'a CanonicalPageRecord>,
        stage1_shortlist_min: usize,
        stage1_shortlist_mult: usize,
    ) -> Result<Self, ColPaliError> {
        let words_per_patch = dim.div_ceil(64);
        let mut pages = Vec::new();
        let mut binary_slab = Vec::new();
        let mut sq8_slab = Vec::new();
        let mut total_patches = 0usize;

        for rec in records {
            if rec.patch_embeddings.is_empty() {
                return Err(ColPaliError::EmptyPatchSet {
                    doc_id: rec.doc_id.clone(),
                    page_no: rec.page_no,
                });
            }
            let patch_offset = total_patches;
            let patch_count = rec.patch_embeddings.len();
            for patch in &rec.patch_embeddings {
                if patch.len() != dim {
                    return Err(ColPaliError::InvalidDimension { expected: dim, got: patch.len() });
                }
                let rotated = fwht_orthogonal(patch);
                let packed = pack_binary_words(&rotated);
                binary_slab.extend_from_slice(&packed);
                // Store SQ8 in the same orthogonal basis (or raw basis; raw basis avoids rotating query twice,
                // while keeping exact inner product preservation).
                sq8_slab.push(Sq8Vector::quantize(patch));
            }
            total_patches = total_patches.checked_add(patch_count).ok_or_else(|| {
                ColPaliError::CorruptedIndex { reason: "patch count overflow".to_string() }
            })?;
            pages.push(VisualPageMeta {
                doc_id: rec.doc_id.clone(),
                page_no: rec.page_no,
                version: rec.version,
                tenant_id: rec.tenant_id.clone(),
                acl_group: rec.acl_group.clone(),
                page_modality: rec.page_modality,
                patch_offset,
                patch_count,
            });
        }

        Ok(Self {
            dim,
            words_per_patch,
            stage1_shortlist_min,
            stage1_shortlist_mult,
            pages,
            binary_slab,
            sq8_slab,
            total_patches,
        })
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

    pub fn total_patches(&self) -> usize {
        self.total_patches
    }

    /// Execute two-stage retrieval:
    /// 1. Stage-1 1-bit `XOR + POPCNT` MaxSim shortlist over authorized pages.
    /// 2. Stage-2 SQ8 MaxSim rerank over shortlisted authorized pages.
    pub fn search(&self, query: &VisualQuery) -> Result<Vec<SearchHit>, ColPaliError> {
        query.validate(self.dim)?;
        if self.pages.is_empty() {
            return Ok(Vec::new());
        }

        let q_bin: Vec<Vec<u64>> = query
            .token_embeddings
            .iter()
            .map(|tok| pack_binary_words(&fwht_orthogonal(tok)))
            .collect();

        // Stage 1: fast 1-bit Hamming/popcount MaxSim over authorized pages only.
        let mut stage1: Vec<(usize, i32)> = Vec::new();
        if self.dim == 128 && self.words_per_patch == 2 {
            let q_pairs: Vec<[u64; 2]> = q_bin.iter().map(|w| [w[0], w[1]]).collect();
            for (idx, meta) in self.pages.iter().enumerate() {
                if !query.scope.permits(&meta.tenant_id, &meta.acl_group) {
                    continue;
                }
                let mut total_sim: i32 = 0;
                let base_word = meta.patch_offset * 2;
                for &q_pair in &q_pairs {
                    let mut best = i32::MIN;
                    for p in 0..meta.patch_count {
                        let off = base_word + p * 2;
                        let p_pair = [self.binary_slab[off], self.binary_slab[off + 1]];
                        let sim = binary_similarity_128(q_pair, p_pair);
                        if sim > best {
                            best = sim;
                        }
                    }
                    total_sim += best;
                }
                stage1.push((idx, total_sim));
            }
        } else {
            for (idx, meta) in self.pages.iter().enumerate() {
                if !query.scope.permits(&meta.tenant_id, &meta.acl_group) {
                    continue;
                }
                let mut total_sim: i32 = 0;
                let base_word = meta.patch_offset * self.words_per_patch;
                for q_words in &q_bin {
                    let mut best = i32::MIN;
                    for p in 0..meta.patch_count {
                        let start = base_word + p * self.words_per_patch;
                        let p_words = &self.binary_slab[start..start + self.words_per_patch];
                        let sim = binary_similarity_words(q_words, p_words, self.dim);
                        if sim > best {
                            best = sim;
                        }
                    }
                    total_sim += best;
                }
                stage1.push((idx, total_sim));
            }
        }

        let shortlist_k =
            (query.top_k.saturating_mul(self.stage1_shortlist_mult)).max(self.stage1_shortlist_min);
        stage1.sort_by(|&(ia, sa), &(ib, sb)| {
            sb.cmp(&sa)
                .then_with(|| self.pages[ia].doc_id.cmp(&self.pages[ib].doc_id))
                .then_with(|| self.pages[ia].page_no.cmp(&self.pages[ib].page_no))
        });
        stage1.truncate(shortlist_k);

        // Stage 2: SQ8 int8 MaxSim rerank over shortlisted authorized candidates.
        let q_sq8: Vec<Sq8Vector> =
            query.token_embeddings.iter().map(|tok| Sq8Vector::quantize(tok)).collect();
        let inv_q = 1.0 / (q_sq8.len() as f32);

        let mut reranked: Vec<(usize, f32)> = Vec::with_capacity(stage1.len());
        for (idx, _stage1_score) in stage1 {
            let meta = &self.pages[idx];
            let mut maxsim_sum = 0.0f32;
            let p_start = meta.patch_offset;
            let p_end = p_start + meta.patch_count;
            let page_patches = &self.sq8_slab[p_start..p_end];
            for q_tok in &q_sq8 {
                let mut best = f32::NEG_INFINITY;
                for p_sq8 in page_patches {
                    let sim = q_tok.dot_sq8(p_sq8);
                    if sim > best {
                        best = sim;
                    }
                }
                maxsim_sum += best;
            }
            reranked.push((idx, maxsim_sum * inv_q));
        }

        reranked.sort_by(|&(ia, sa), &(ib, sb)| {
            sb.total_cmp(&sa)
                .then_with(|| self.pages[ia].doc_id.cmp(&self.pages[ib].doc_id))
                .then_with(|| self.pages[ia].page_no.cmp(&self.pages[ib].page_no))
        });
        reranked.truncate(query.top_k);

        Ok(reranked
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
                    route_used: RouteKind::ColPaliBinaryTwoStage,
                }
            })
            .collect())
    }

    /// Deterministic binary serialization (`HCB2` + trailing CRC32).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_COLPALI_BIN2);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.words_per_patch as u32).to_le_bytes());
        buf.extend_from_slice(&(self.stage1_shortlist_min as u32).to_le_bytes());
        buf.extend_from_slice(&(self.stage1_shortlist_mult as u32).to_le_bytes());
        buf.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.total_patches as u64).to_le_bytes());
        for p in &self.pages {
            write_str(&mut buf, &p.doc_id);
            buf.extend_from_slice(&p.page_no.to_le_bytes());
            buf.extend_from_slice(&p.version.to_le_bytes());
            write_str(&mut buf, &p.tenant_id);
            write_str(&mut buf, &p.acl_group);
            buf.push(p.page_modality.tag());
            buf.extend_from_slice(&(p.patch_offset as u64).to_le_bytes());
            buf.extend_from_slice(&(p.patch_count as u32).to_le_bytes());
        }
        for &w in &self.binary_slab {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for sq in &self.sq8_slab {
            buf.extend_from_slice(&sq.scale.to_le_bytes());
            for &c in &sq.codes {
                buf.push(c as u8);
            }
        }
        let crc = crc32_ieee(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Deserialize with strict magic, bounds, and CRC32 verification.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ColPaliError> {
        if bytes.len() < 38 {
            return Err(ColPaliError::CorruptedIndex {
                reason: "HCB2 buffer truncated".to_string(),
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
                    "HCB2 CRC32 mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
                ),
            });
        }

        let mut cursor = ByteCursor::new(&bytes[..payload_len]);
        if cursor.read_exact(4)? != MAGIC_COLPALI_BIN2 {
            return Err(ColPaliError::CorruptedIndex { reason: "invalid HCB2 magic".to_string() });
        }
        let version = cursor.read_u16()?;
        if version != FORMAT_VERSION {
            return Err(ColPaliError::CorruptedIndex {
                reason: format!("unsupported HCB2 version {version}"),
            });
        }
        let dim = cursor.read_u32()? as usize;
        let words_per_patch = cursor.read_u32()? as usize;
        let stage1_shortlist_min = cursor.read_u32()? as usize;
        let stage1_shortlist_mult = cursor.read_u32()? as usize;
        let page_count = cursor.read_u32()? as usize;
        let total_patches = cursor.read_u64()? as usize;

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
            let patch_offset = cursor.read_u64()? as usize;
            let patch_count = cursor.read_u32()? as usize;
            let end_patch = patch_offset.checked_add(patch_count).ok_or_else(|| {
                ColPaliError::CorruptedIndex { reason: "patch range overflow".to_string() }
            })?;
            if end_patch > total_patches {
                return Err(ColPaliError::CorruptedIndex {
                    reason: "patch range exceeds slab".to_string(),
                });
            }
            pages.push(VisualPageMeta {
                doc_id,
                page_no,
                version: rec_version,
                tenant_id,
                acl_group,
                page_modality,
                patch_offset,
                patch_count,
            });
        }

        let total_words = total_patches.checked_mul(words_per_patch).ok_or_else(|| {
            ColPaliError::CorruptedIndex { reason: "binary slab size overflow".to_string() }
        })?;
        let mut binary_slab = Vec::with_capacity(total_words);
        for _ in 0..total_words {
            binary_slab.push(cursor.read_u64()?);
        }

        let mut sq8_slab = Vec::with_capacity(total_patches);
        for _ in 0..total_patches {
            let scale = cursor.read_f32()?;
            let raw = cursor.read_exact(dim)?;
            let codes = raw.iter().map(|&b| b as i8).collect();
            sq8_slab.push(Sq8Vector { scale, codes });
        }

        Ok(Self {
            dim,
            words_per_patch,
            stage1_shortlist_min,
            stage1_shortlist_mult,
            pages,
            binary_slab,
            sq8_slab,
            total_patches,
        })
    }

    pub fn checksum_crc32(&self) -> u32 {
        crc32_ieee(&self.to_bytes())
    }

    /// Exact patch-vector compression ratio of Stage-1 1-bit binary RAM vs FP32.
    pub fn binary_patch_compression_ratio(&self) -> f64 {
        let fp32_bytes = self.total_patches * self.dim * std::mem::size_of::<f32>();
        let bin_bytes = self.binary_slab.len() * std::mem::size_of::<u64>();
        if bin_bytes == 0 {
            0.0
        } else {
            fp32_bytes as f64 / bin_bytes as f64
        }
    }

    /// Hot RAM holds ONLY metadata + 1-bit binary patch signatures (`32x` smaller patch payload);
    /// Cold disk/S3 holds both binary signatures and SQ8 Stage-2 rerank slabs.
    pub fn footprint(&self) -> IndexFootprint {
        let meta_bytes: usize = self
            .pages
            .iter()
            .map(|p| {
                p.doc_id.len()
                    + p.tenant_id.len()
                    + p.acl_group.len()
                    + std::mem::size_of::<VisualPageMeta>()
            })
            .sum();
        let binary_bytes = self.binary_slab.len() * std::mem::size_of::<u64>();
        let hot_ram_bytes = meta_bytes + binary_bytes;
        let disk_bytes = self.to_bytes().len();
        IndexFootprint::new(hot_ram_bytes, disk_bytes, self.pages.len(), self.pages.len())
    }
}
