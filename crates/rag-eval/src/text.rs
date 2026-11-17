// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust lightweight text utilities shared across corrective retrieval,
//! RAGChecker diagnostics, atomic claim decomposition, and span verification.

use std::collections::HashSet;

/// Sorted English stopwords used for lexical overlap and query reformulation.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "done", "during", "each", "either", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "however",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "may", "me", "might", "more",
    "most", "must", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "shall", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "whereas", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
];

/// Sorted negation markers (checked on lowercased raw tokens).
const NEGATION_WORDS: &[&str] = &[
    "cannot",
    "disabled",
    "false",
    "neither",
    "never",
    "no",
    "none",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "unsupported",
    "without",
];

/// Sorted common abbreviations that end with `.` without ending a sentence.
const ABBREVIATIONS: &[&str] = &[
    "al.", "approx.", "ca.", "cf.", "dr.", "e.g.", "eg.", "etc.", "fig.", "i.e.", "ie.", "inc.",
    "mr.", "mrs.", "ms.", "no.", "prof.", "rev.", "st.", "u.k.", "u.s.", "vs.",
];

/// Result of comparing an atomic claim against a candidate evidence span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SupportAssessment {
    pub token_recall: f64,
    pub bigram_overlap: f64,
    pub polarity_match: bool,
    pub numeric_match: bool,
    pub score: f64,
}

/// Checks whether a lowercased word is a common stopword.
#[inline]
pub(crate) fn is_stopword(word: &str) -> bool {
    STOPWORDS.binary_search(&word).is_ok()
}

/// Computes the fraction of replacement (`\u{FFFD}`) or non-whitespace control
/// characters in `text`. Empty or whitespace-only text returns `1.0`.
pub(crate) fn corruption_ratio(text: &str) -> f64 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 1.0;
    }
    let mut total = 0usize;
    let mut corrupted = 0usize;
    for ch in trimmed.chars() {
        total += 1;
        if ch == '\u{FFFD}' || (ch.is_control() && !ch.is_whitespace()) {
            corrupted += 1;
        }
    }
    corrupted as f64 / total as f64
}

/// Splits `text` into trimmed `(byte_start, byte_end, slice)` sentence spans
/// while preserving decimal/version numbers, numbered list prefixes, and common
/// abbreviations.
pub(crate) fn split_sentences(text: &str) -> Vec<(usize, usize, &str)> {
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;

    let mut i = 0usize;
    while i < bytes.len() {
        let ch = text[i..].chars().next().expect("valid utf-8 boundary");
        let ch_len = ch.len_utf8();

        if ch == '\n' {
            push_trimmed_span(text, start, i, &mut spans);
            start = i + ch_len;
            i += ch_len;
            continue;
        }

        if ch == '!' || ch == '?' {
            push_trimmed_span(text, start, i + ch_len, &mut spans);
            start = i + ch_len;
            i += ch_len;
            continue;
        }

        if ch == '.' {
            let prev_digit = i > 0 && bytes[i - 1].is_ascii_digit();
            let next_byte = bytes.get(i + 1).copied();
            let next_digit = next_byte.is_some_and(|b| b.is_ascii_digit());
            let prev_dot = i > 0 && bytes[i - 1] == b'.';
            let next_dot = next_byte == Some(b'.');

            // Keep decimals / semvers / ellipses intact.
            if (prev_digit && next_digit) || prev_dot || next_dot {
                i += ch_len;
                continue;
            }

            // Keep numbered list markers like `1. First item` attached to their item.
            let prefix_trimmed = text[start..i].trim();
            if !prefix_trimmed.is_empty()
                && prefix_trimmed.len() <= 3
                && prefix_trimmed.bytes().all(|b| b.is_ascii_digit())
                && next_byte.is_some_and(|b| b.is_ascii_whitespace())
            {
                i += ch_len;
                continue;
            }

            // Keep common abbreviations (`e.g.`, `Dr.`, `U.S.`, `vs.`) attached.
            let candidate_word =
                text[start..=i].rsplit(|c: char| c.is_whitespace()).next().unwrap_or("");
            let lower = candidate_word.to_ascii_lowercase();
            if ABBREVIATIONS.binary_search(&lower.as_str()).is_ok() {
                i += ch_len;
                continue;
            }

            // Split only at end-of-text or when followed by whitespace / quote.
            let followed_by_boundary = next_byte.is_none_or(|b| {
                b.is_ascii_whitespace() || b == b'"' || b == b'\'' || b == b')' || b == b']'
            });
            if followed_by_boundary {
                push_trimmed_span(text, start, i + ch_len, &mut spans);
                start = i + ch_len;
            }
        }

        i += ch_len;
    }

    if start < text.len() {
        push_trimmed_span(text, start, text.len(), &mut spans);
    }

    spans
}

fn push_trimmed_span<'a>(
    text: &'a str,
    raw_start: usize,
    raw_end: usize,
    out: &mut Vec<(usize, usize, &'a str)>,
) {
    if raw_start >= raw_end {
        return;
    }
    let slice = &text[raw_start..raw_end];
    let leading = slice.len() - slice.trim_start().len();
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        return;
    }
    let start = raw_start + leading;
    let end = start + trimmed.len();
    out.push((start, end, &text[start..end]));
}

/// Tokenizes raw words while preserving internal numeric dots (`3.14`, `1.45.2`)
/// and apostrophes (`doesn't`) for negation and numeric matching.
pub(crate) fn tokenize_raw(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (idx, &ch) in chars.iter().enumerate() {
        let prev_digit = idx > 0 && chars[idx - 1].is_ascii_digit();
        let next_digit = idx + 1 < chars.len() && chars[idx + 1].is_ascii_digit();
        let prev_alpha = idx > 0 && chars[idx - 1].is_ascii_alphabetic();
        let next_alpha = idx + 1 < chars.len() && chars[idx + 1].is_ascii_alphabetic();

        let keep = ch.is_alphanumeric()
            || (ch == '.' && prev_digit && next_digit)
            || ((ch == '\'' || ch == '’' || ch == '-') && prev_alpha && next_alpha);

        if keep {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Lightweight deterministic stemmer for English content words.
pub(crate) fn stem_lite(word: &str) -> String {
    if word.len() <= 4 || !word.bytes().all(|b| b.is_ascii_alphabetic()) {
        return word.to_string();
    }
    if let Some(stem) = word.strip_suffix("ies") {
        if stem.len() >= 2 {
            return format!("{stem}y");
        }
    }
    if let Some(stem) = word.strip_suffix("ied") {
        if stem.len() >= 2 {
            return format!("{stem}y");
        }
    }
    if let Some(stem) = word.strip_suffix("ing") {
        if stem.len() >= 3 {
            return stem.to_string();
        }
    }
    if let Some(stem) = word.strip_suffix("ed") {
        if stem.len() >= 3 {
            return stem.to_string();
        }
    }
    if let Some(stem) = word.strip_suffix("es") {
        if stem.len() >= 3
            && (stem.ends_with('s')
                || stem.ends_with('x')
                || stem.ends_with('z')
                || stem.ends_with("ch")
                || stem.ends_with("sh"))
        {
            return stem.to_string();
        }
    }
    if let Some(stem) = word.strip_suffix('s') {
        if stem.len() >= 3
            && !word.ends_with("ss")
            && !word.ends_with("us")
            && !word.ends_with("is")
        {
            return stem.to_string();
        }
    }
    word.to_string()
}

/// Returns stemmed, stopword-filtered content tokens from `text`.
pub(crate) fn tokenize_content(text: &str) -> Vec<String> {
    tokenize_raw(text)
        .into_iter()
        .filter(|tok| !is_stopword(tok))
        .map(|tok| stem_lite(&tok))
        .collect()
}

/// Extracts all numeric literals (integers, decimals, versions) from `text`.
pub(crate) fn extract_numbers(text: &str) -> Vec<String> {
    tokenize_raw(text)
        .into_iter()
        .filter(|tok| tok.bytes().any(|b| b.is_ascii_digit()) && !tok.bytes().any(|b| b.is_ascii_alphabetic()))
        .collect()
}

/// Counts explicit negation cues within a span.
pub(crate) fn negation_count(text: &str) -> usize {
    tokenize_raw(text)
        .into_iter()
        .filter(|tok| {
            let normalized = tok.replace('’', "'");
            NEGATION_WORDS.binary_search(&normalized.as_str()).is_ok()
                || normalized.ends_with("n't")
        })
        .count()
}

/// Checks whether a span has odd (active) negation polarity.
#[inline]
pub(crate) fn is_negated(text: &str) -> bool {
    negation_count(text) % 2 == 1
}

/// Evaluates lexical, phrase, polarity, and numeric support of `claim` against `passage`.
pub(crate) fn assess_support(claim: &str, passage: &str) -> SupportAssessment {
    let claim_tokens = tokenize_content(claim);
    let passage_tokens = tokenize_content(passage);

    if claim_tokens.is_empty() || passage_tokens.is_empty() {
        return SupportAssessment {
            token_recall: 0.0,
            bigram_overlap: 0.0,
            polarity_match: true,
            numeric_match: true,
            score: 0.0,
        };
    }

    let passage_set: HashSet<&str> = passage_tokens.iter().map(String::as_str).collect();
    let unique_claim: HashSet<&str> = claim_tokens.iter().map(String::as_str).collect();
    let matched_tokens = unique_claim.iter().filter(|tok| passage_set.contains(*tok)).count();
    let token_recall = matched_tokens as f64 / unique_claim.len() as f64;

    let bigram_overlap = if claim_tokens.len() >= 2 && passage_tokens.len() >= 2 {
        let passage_bigrams: HashSet<(&str, &str)> = passage_tokens
            .windows(2)
            .map(|w| (w[0].as_str(), w[1].as_str()))
            .collect();
        let claim_bigrams: Vec<(&str, &str)> =
            claim_tokens.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect();
        let matched_bigrams =
            claim_bigrams.iter().filter(|bg| passage_bigrams.contains(bg)).count();
        matched_bigrams as f64 / claim_bigrams.len() as f64
    } else {
        token_recall
    };

    let claim_nums = extract_numbers(claim);
    let passage_nums: HashSet<String> = extract_numbers(passage).into_iter().collect();
    let numeric_match = if claim_nums.is_empty() {
        true
    } else {
        claim_nums.iter().all(|n| passage_nums.contains(n))
    };

    let polarity_match = is_negated(claim) == is_negated(passage);

    let raw = 0.75 * token_recall + 0.25 * bigram_overlap;
    let score = if polarity_match && numeric_match {
        raw.clamp(0.0, 1.0)
    } else {
        // Explicit polarity or numeric conflicts must never pass as strong support.
        (raw * 0.25).clamp(0.0, 1.0)
    };

    SupportAssessment { token_recall, bigram_overlap, polarity_match, numeric_match, score }
}

/// Finds the highest-scoring sentence or 2-sentence window within `document`
/// supporting `claim`, returning `(byte_start, byte_end, assessment)`.
pub(crate) fn best_sentence_support(claim: &str, document: &str) -> Option<(usize, usize, SupportAssessment)> {
    let sentences = split_sentences(document);
    if sentences.is_empty() {
        return None;
    }

    let mut best: Option<(usize, usize, SupportAssessment)> = None;

    for &(start, end, slice) in &sentences {
        let assessment = assess_support(claim, slice);
        let replace = match &best {
            None => true,
            Some((_, _, prev)) => assessment.score > prev.score,
        };
        if replace {
            best = Some((start, end, assessment));
        }
    }

    // Check 2-sentence windows when neither single sentence achieves full support.
    if sentences.len() >= 2 {
        for pair in sentences.windows(2) {
            let start = pair[0].0;
            let end = pair[1].1;
            let window_slice = &document[start..end];
            let assessment = assess_support(claim, window_slice);
            let replace = match &best {
                None => true,
                // Prefer tighter single-sentence span unless 2-sentence window improves score meaningfully.
                Some((_, _, prev)) => assessment.score > prev.score + 0.05,
            };
            if replace {
                best = Some((start, end, assessment));
            }
        }
    }

    best
}
