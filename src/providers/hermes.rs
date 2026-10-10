//! Hermes Agent session evidence, read from Hermes's own state database.
//!
//! Hermes publishes no subscription quota, so this is not a quota collector:
//! it supplies the model, the backend a session is billed to, the generated
//! session title, the prompt-cache hit rate, and the current context for one
//! exact session. Everything comes from `$HERMES_HOME/state.db` (default
//! `~/.hermes/state.db`), opened read-only and queried by the exact Herdr
//! `agent_session` id, plus two files Hermes writes beside it for model
//! windows. Nothing here opens `auth.json`, `.env`, or `config.yaml`, and
//! nothing is ever written.
//!
//! Context follows Hermes's own usage anchor (`model_config._usage_anchor`):
//! the provider's `prompt_tokens + completion_tokens` for the last priced
//! response, plus a bytes/4 estimate of only the messages appended since. The
//! anchor is honoured only while the active transcript still matches the
//! role and SHA-256 fingerprints it recorded, which is the same check Hermes
//! runs when it resumes a session. A missing, stale, or unverifiable anchor
//! yields no context rather than a lifetime total dressed up as the current
//! window; the next priced response re-anchors it.

use crate::model::{CacheTotals, CacheUsage, ContextUsage};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Hermes stores list/dict message content as JSON behind this marker.
const CONTENT_JSON_PREFIX: &str = "\u{0}json:";
/// The usage anchor key in a session's `model_config` JSON.
const USAGE_ANCHOR_KEY: &str = "_usage_anchor";
/// Hermes's rough chars-per-token ratio for non-CJK text.
const CHARS_PER_TOKEN: usize = 4;

/// What one Hermes session says about the pane drawing it.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionEvidence {
    pub model: Option<String>,
    /// Hermes's `billing_provider` for the session (`xai-oauth`,
    /// `openai-codex`, `anthropic`, ...). Identity only, never attribution.
    pub billing_provider: Option<String>,
    /// Hermes's generated session title, the same evidence Cursor's title is.
    pub title: Option<String>,
    pub context: Option<ContextUsage>,
}

/// `$HERMES_HOME`, else `~/.hermes`.
pub fn home() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HERMES_HOME") {
        return Some(PathBuf::from(home));
    }
    Some(PathBuf::from(std::env::var_os("HOME")?).join(".hermes"))
}

pub fn session_evidence(session_id: &str) -> Option<SessionEvidence> {
    session_evidence_in(&home()?, session_id)
}

struct SessionRow {
    model: Option<String>,
    model_config: Option<String>,
    billing_provider: Option<String>,
    billing_base_url: Option<String>,
    input_tokens: Option<i64>,
    cache_read_tokens: Option<i64>,
    cache_write_tokens: Option<i64>,
    title: Option<String>,
}

/// One active transcript row as Hermes loads it for the model.
struct Message {
    role: String,
    content: Option<Value>,
    api_content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<Value>,
    reasoning_content: Option<String>,
}

pub(crate) fn session_evidence_in(home: &Path, session_id: &str) -> Option<SessionEvidence> {
    if session_id.is_empty() {
        return None;
    }
    let connection =
        Connection::open_with_flags(home.join("state.db"), OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    let row = connection
        .query_row(
            "SELECT model, model_config, billing_provider, billing_base_url, input_tokens, \
             cache_read_tokens, cache_write_tokens, title FROM sessions WHERE id = ?1",
            rusqlite::params![session_id],
            |row| {
                Ok(SessionRow {
                    model: row.get(0)?,
                    model_config: row.get(1)?,
                    billing_provider: row.get(2)?,
                    billing_base_url: row.get(3)?,
                    input_tokens: row.get(4)?,
                    cache_read_tokens: row.get(5)?,
                    cache_write_tokens: row.get(6)?,
                    title: row.get(7)?,
                })
            },
        )
        .ok()?;
    let model_config: Value = row
        .model_config
        .as_deref()
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or(Value::Null);
    let cache = cache_usage(&row, session_id);
    let model = non_empty(row.model);
    // The window and the anchor are independent reads; both must agree
    // before any percentage is shown, and the cache rides on the context.
    let context = model
        .as_deref()
        .and_then(|model| {
            context_window(
                home,
                &model_config,
                model,
                row.billing_base_url.as_deref().unwrap_or(""),
            )
        })
        .and_then(|window| {
            let messages = active_messages(&connection, session_id).ok()?;
            let tokens = anchored_context_tokens(&messages, model_config.get(USAGE_ANCHOR_KEY)?)?;
            let used_percent = (tokens as f64 / window as f64 * 100.0).clamp(0.0, 100.0);
            ContextUsage::new(used_percent).ok()
        })
        .map(|context| context.with_cache(cache));
    Some(SessionEvidence {
        model,
        billing_provider: non_empty(row.billing_provider),
        title: non_empty(row.title),
        context,
    })
}

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Cumulative hit rate over the session: `cache_read / (input + cache_read +
/// cache_write)`. Hermes's `input_tokens` already excludes cached input.
fn cache_usage(row: &SessionRow, session_id: &str) -> Option<CacheUsage> {
    let count = |value: Option<i64>| value.unwrap_or(0).max(0) as u64;
    let fresh = count(row.input_tokens);
    let read = count(row.cache_read_tokens);
    let creation = count(row.cache_write_tokens);
    let totals = CacheTotals::from_token_counts(fresh, read, creation)?;
    Some(CacheUsage {
        fresh_input_tokens: fresh,
        read_tokens: read,
        creation_tokens: creation,
        hit_percent: totals.hit_percent,
        ttl_seconds: None,
        last_activity_unix: None,
        expires_at_unix: None,
        session_totals: Some(totals),
        session_id: Some(session_id.to_string()),
        transcript_offset: 0,
    })
}

fn active_messages(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<Message>, rusqlite::Error> {
    let mut statement = connection.prepare(
        "SELECT role, content, api_content, tool_call_id, tool_calls, reasoning_content \
         FROM messages WHERE session_id = ?1 AND active = 1 ORDER BY id",
    )?;
    let rows = statement.query_map(rusqlite::params![session_id], |row| {
        let role: String = row.get(0)?;
        let content: Option<String> = row.get(1)?;
        let tool_calls: Option<String> = row.get(4)?;
        Ok(Message {
            content: content.map(|content| loaded_content(&role, content)),
            api_content: row
                .get::<_, Option<String>>(2)?
                .filter(|value| !value.is_empty()),
            tool_call_id: row
                .get::<_, Option<String>>(3)?
                .filter(|value| !value.is_empty()),
            // Hermes parses a stored tool_calls string and falls back to an
            // empty list, which still counts as a value in the fingerprint.
            tool_calls: tool_calls
                .filter(|value| !value.is_empty())
                .map(|value| serde_json::from_str(&value).unwrap_or(Value::Array(Vec::new()))),
            reasoning_content: row.get(5)?,
            role,
        })
    })?;
    rows.collect()
}

/// Content as Hermes hands it to the model: structured content decoded from
/// its marker, and user/assistant text stripped on load.
fn loaded_content(role: &str, content: String) -> Value {
    if let Some(encoded) = content.strip_prefix(CONTENT_JSON_PREFIX) {
        if let Ok(value) = serde_json::from_str::<Value>(encoded) {
            return value;
        }
    }
    if role == "user" || role == "assistant" {
        // Hermes also removes injected memory-context blocks here. A row that
        // carries one fails the fingerprint instead, and the pane shows no
        // context until the next priced response.
        return Value::String(content.trim().to_string());
    }
    Value::String(content)
}

/// Hermes's `message_fingerprint`: SHA-256 over the provider-visible fields
/// that round-trip the database, serialised the way Python's `json.dumps`
/// does with sorted keys, compact separators, and ASCII escapes.
fn message_fingerprint(message: &Message) -> String {
    let mut fields: Vec<(&str, Value)> = Vec::new();
    if let Some(api_content) = &message.api_content {
        fields.push(("api_content", Value::String(api_content.clone())));
    }
    if let Some(content) = &message.content {
        fields.push(("content", content.clone()));
    }
    fields.push(("role", Value::String(message.role.clone())));
    if let Some(tool_call_id) = &message.tool_call_id {
        fields.push(("tool_call_id", Value::String(tool_call_id.clone())));
    }
    if let Some(tool_calls) = &message.tool_calls {
        fields.push(("tool_calls", tool_calls.clone()));
    }
    let mut raw = String::from("{");
    for (index, (key, value)) in fields.iter().enumerate() {
        if index > 0 {
            raw.push(',');
        }
        python_json_string(key, &mut raw);
        raw.push(':');
        python_json(value, &mut raw);
    }
    raw.push('}');
    format!("{:x}", Sha256::digest(raw.as_bytes()))
}

/// Hermes's `_priced_prefix_fingerprint`: the digest of every `(role,
/// fingerprint)` pair in the priced prefix, so a compaction that keeps the
/// last row but rewrites earlier ones still fails closed.
fn prefix_fingerprint(messages: &[Message]) -> String {
    let mut raw = String::from("[");
    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            raw.push(',');
        }
        raw.push('[');
        python_json_string(&message.role, &mut raw);
        raw.push(',');
        python_json_string(&message_fingerprint(message), &mut raw);
        raw.push(']');
    }
    raw.push(']');
    format!("{:x}", Sha256::digest(raw.as_bytes()))
}

/// Hermes's `anchored_context_tokens`: the anchored prompt and completion
/// counts plus a rough estimate of only the messages appended since, or
/// `None` when the anchor no longer describes the active transcript.
fn anchored_context_tokens(messages: &[Message], anchor: &Value) -> Option<u64> {
    let count = |key: &str| anchor.get(key).and_then(json_count);
    let prompt_tokens = count("prompt_tokens").filter(|tokens| *tokens > 0)?;
    let completion_tokens = count("completion_tokens").unwrap_or(0);
    let base_count = count("base_count").filter(|count| *count > 0)? as usize;
    if messages.len() < base_count {
        return None;
    }
    let base = &messages[base_count - 1];
    if anchor.get("base_last_role").and_then(Value::as_str) != Some(base.role.as_str()) {
        return None;
    }
    let digest = |key: &str| {
        anchor
            .get(key)
            .and_then(Value::as_str)
            .filter(|fp| !fp.is_empty())
    };
    if digest("base_last_fp")? != message_fingerprint(base) {
        return None;
    }
    if digest("base_prefix_fp")? != prefix_fingerprint(&messages[..base_count]) {
        return None;
    }
    let mut delta = &messages[base_count..];
    // The anchored response's own reply is already in completion_tokens.
    if delta
        .first()
        .is_some_and(|message| message.role == "assistant")
    {
        delta = &delta[1..];
    }
    let estimated: u64 = delta
        .iter()
        .map(|message| estimate_tokens_rough(&estimate_text(message)) as u64)
        .sum();
    Some(prompt_tokens + completion_tokens + estimated)
}

fn json_count(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
}

/// Text standing in for Hermes's wire shadow of a message. The provider
/// sees `api_content` instead of `content` on user/assistant rows when the
/// sidecar is set. This is a rough estimate by construction (bytes/4), and
/// it only ever prices the few messages after the anchor.
fn estimate_text(message: &Message) -> String {
    let mut text = message.role.clone();
    let sidecar = message
        .api_content
        .as_deref()
        .filter(|_| message.role == "user" || message.role == "assistant");
    match (sidecar, &message.content) {
        (Some(sidecar), _) => text.push_str(sidecar),
        (None, Some(Value::String(content))) => text.push_str(content),
        (None, Some(content)) => text.push_str(&content.to_string()),
        (None, None) => {}
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        text.push_str(tool_call_id);
    }
    if let Some(tool_calls) = &message.tool_calls {
        text.push_str(&tool_calls.to_string());
    }
    if let Some(reasoning) = &message.reasoning_content {
        text.push_str(reasoning);
    }
    text
}

/// Hermes's `estimate_tokens_rough`: CJK/Hangul/Kana code points count one
/// token each; everything else is `ceil(utf8 bytes / 4)`.
fn estimate_tokens_rough(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    if text.is_ascii() {
        return (text.len() + 3) / CHARS_PER_TOKEN;
    }
    let (dense, other_bytes) = text.chars().fold((0usize, 0usize), |(dense, bytes), c| {
        if is_token_dense(c) {
            (dense + 1, bytes)
        } else {
            (dense, bytes + c.len_utf8())
        }
    });
    dense + (other_bytes + 3) / CHARS_PER_TOKEN
}

fn is_token_dense(c: char) -> bool {
    matches!(
        c as u32,
        0x1100..=0x11ff
            | 0x2e80..=0x9fff
            | 0xa960..=0xa97f
            | 0xac00..=0xd7af
            | 0xf900..=0xfaff
            | 0xff00..=0xffef
    )
}

/// `json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))`.
fn python_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                let _ = write!(out, "{int}");
            } else if let Some(int) = number.as_u64() {
                let _ = write!(out, "{int}");
            } else if let Some(float) = number.as_f64() {
                out.push_str(&python_float(float));
            }
        }
        Value::String(text) => python_json_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                python_json_string(key, out);
                out.push(':');
                python_json(&map[key], out);
            }
            out.push('}');
        }
    }
}

/// Python's `float.__repr__` for the values a transcript can hold: whole
/// numbers keep their `.0`, everything else is the shortest round trip.
/// Exponent forms are not reproduced; such a row fails the fingerprint.
fn python_float(float: f64) -> String {
    if float.is_finite() && float.fract() == 0.0 && float.abs() < 1e16 {
        format!("{float:.1}")
    } else {
        format!("{float}")
    }
}

/// Python's `json.dumps` string escaping with `ensure_ascii=True`: the five
/// short escapes, and `\uXXXX` (surrogate pairs above the BMP) for anything
/// outside printable ASCII.
fn python_json_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// The model's window, from Hermes's own records in the order Hermes
/// resolves them: an explicit `context_length` in the session's model
/// config, then `context_length_cache.yaml` keyed by exact `model@base_url`,
/// then the endpoint `/models` probe memo under `cache/`. No network, and no
/// family guess: an unknown window means no context row.
fn context_window(home: &Path, model_config: &Value, model: &str, base_url: &str) -> Option<u64> {
    if let Some(explicit) = model_config
        .get("context_length")
        .and_then(json_count)
        .filter(|length| *length > 0)
    {
        return Some(explicit);
    }
    let base_url = base_url.trim();
    let candidates = model_candidates(model);
    let yaml = std::fs::read_to_string(home.join("context_length_cache.yaml")).ok();
    if let Some(window) = yaml
        .as_deref()
        .and_then(|yaml| cached_context_length(yaml, &candidates, base_url))
    {
        return Some(window);
    }
    let probe = std::fs::read_to_string(home.join("cache").join("endpoint_model_metadata.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())?;
    endpoint_context_length(&probe, &candidates, base_url)
}

/// The spellings a window may be recorded under: the model as stored, then
/// without a `provider:` prefix, then the bare name behind an `org/` prefix.
/// The stored spelling wins; a bare alias is only the fallback.
fn model_candidates(model: &str) -> Vec<String> {
    let mut candidates = vec![model.to_string()];
    if !model.starts_with("http") {
        if let Some((_, suffix)) = model.split_once(':') {
            candidates.push(suffix.to_string());
        }
    }
    if let Some((_, bare)) = model.rsplit_once('/') {
        candidates.push(bare.to_string());
    }
    candidates.dedup();
    candidates
}

/// Exact `model@base_url` entries from the `context_lengths:` map, probing
/// the canonical (trailing slash stripped), literal, and slashed forms the
/// way Hermes does. Values must be whole numbers.
fn cached_context_length(yaml: &str, models: &[String], base_url: &str) -> Option<u64> {
    let canonical = base_url.trim_end_matches('/');
    let mut entries = Vec::new();
    let mut in_map = false;
    for line in yaml.lines() {
        if line.starts_with("context_lengths:") {
            in_map = true;
            continue;
        }
        if !line.starts_with(' ') {
            in_map = false;
            continue;
        }
        if !in_map {
            continue;
        }
        let Some((key, value)) = line.trim().rsplit_once(':') else {
            continue;
        };
        let key = key.trim().trim_matches(|c| c == '\'' || c == '"');
        if let Ok(value) = value.trim().parse::<u64>() {
            entries.push((key.to_string(), value));
        }
    }
    models.iter().find_map(|model| {
        [
            format!("{model}@{canonical}"),
            format!("{model}@{base_url}"),
            format!("{model}@{canonical}/"),
        ]
        .iter()
        .find_map(|wanted| {
            entries
                .iter()
                .find(|(key, _)| key == wanted)
                .map(|(_, value)| *value)
        })
    })
}

/// `cache/endpoint_model_metadata.json` is Hermes's memo of a provider's own
/// `/models` answer, keyed by the base URL it asked.
fn endpoint_context_length(probe: &Value, models: &[String], base_url: &str) -> Option<u64> {
    let canonical = base_url.trim_end_matches('/');
    let entry = probe.get(canonical).or_else(|| probe.get(base_url))?;
    let catalog = entry.get("models")?;
    models.iter().find_map(|model| {
        catalog
            .get(model)
            .and_then(|model| model.get("context_length"))
            .and_then(json_count)
            .filter(|length| *length > 0)
    })
}
