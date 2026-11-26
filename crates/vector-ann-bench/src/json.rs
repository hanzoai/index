// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal, deterministic, zero-dependency JSON value & pretty-printer for benchmark receipts.

use std::fmt::Write as _;

/// Minimal JSON AST for serializing structured benchmark receipts without external crates.
#[derive(Debug, Clone)]
pub enum JsonVal {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<JsonVal>),
    Object(Vec<(String, JsonVal)>),
}

impl JsonVal {
    pub fn obj(fields: Vec<(&str, JsonVal)>) -> Self {
        Self::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn to_pretty_string(&self) -> String {
        let mut out = String::with_capacity(8192);
        self.write_pretty(&mut out, 0);
        out.push('\n');
        out
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Self::UInt(n) => {
                let _ = write!(out, "{n}");
            }
            Self::Float(f) => {
                if !f.is_finite() {
                    out.push_str("null");
                } else {
                    let _ = write!(out, "{f:.6}");
                }
            }
            Self::Str(s) => write_escaped_str(out, s),
            Self::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                let child_indent = indent + 2;
                for (i, item) in items.iter().enumerate() {
                    push_spaces(out, child_indent);
                    item.write_pretty(out, child_indent);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_spaces(out, indent);
                out.push(']');
            }
            Self::Object(entries) => {
                if entries.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                let child_indent = indent + 2;
                for (i, (k, v)) in entries.iter().enumerate() {
                    push_spaces(out, child_indent);
                    write_escaped_str(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, child_indent);
                    if i + 1 < entries.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_spaces(out, indent);
                out.push('}');
            }
        }
    }
}

fn push_spaces(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push(' ');
    }
}

fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
