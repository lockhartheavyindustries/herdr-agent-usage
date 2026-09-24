use crate::model::{CacheTotals, CacheUsage, ContextUsage, ProviderSnapshot};
use crate::providers::ProviderError;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Read the active model label from a statusLine payload. The display name
/// is preferred so the sidebar stays readable at a glance; a payload that
/// reports only an id (a model released after this build, or a new alias)
/// shows that id rather than an empty identity row.
pub fn parse_model(value: &Value) -> Option<String> {
    let model = value.get("model")?.as_object()?;
    ["display_name", "displayName", "id"]
        .into_iter()
        .filter_map(|key| model.get(key).and_then(Value::as_str))
        .map(str::trim)
        .find(|model| !model.is_empty())
        .map(str::to_string)
}

pub fn parse_context(
    value: Option<&Value>,
) -> std::result::Result<Option<ContextUsage>, ProviderError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let Some(object) = value.as_object() else {
        return Err(ProviderError::UnsupportedResponse(
            "context_window is not an object".to_string(),
        ));
    };
    let used = object
        .get("used_percentage")
        .or_else(|| object.get("usedPercentage"))
        .and_then(Value::as_f64);
    let remaining = object
        .get("remaining_percentage")
        .or_else(|| object.get("remainingPercentage"))
        .and_then(Value::as_f64);
    let Some(percent) = used.or_else(|| remaining.map(|value| 100.0 - value)) else {
        return Ok(None);
    };
    let cache = parse_cache_usage(object.get("current_usage"));
    ContextUsage::new(percent)
        .map(|context| context.with_cache(cache))
        .map(Some)
        .map_err(|error| ProviderError::UnsupportedResponse(error.to_string()))
}

fn parse_cache_usage(value: Option<&Value>) -> Option<CacheUsage> {
    let object = value?.as_object()?;
    let has_cache_counters = [
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
    ]
    .into_iter()
    .any(|name| object.contains_key(name));
    if !has_cache_counters {
        return None;
    }
    let fresh = token_count(object, "input_tokens", "inputTokens");
    let read = token_count(object, "cache_read_input_tokens", "cacheReadInputTokens");
    let creation = token_count(
        object,
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
    );
    CacheUsage::from_token_counts(fresh, read, creation)
}

fn token_count(object: &serde_json::Map<String, Value>, snake: &str, camel: &str) -> u64 {
    object
        .get(snake)
        .or_else(|| object.get(camel))
        .and_then(Value::as_u64)
        .unwrap_or_default()
}

/// Digest statusLine fields whose values are derived from provider responses.
///
/// Claude Code can run the statusLine command again on `refreshInterval`
/// without another API response. This fingerprint lets the cache distinguish
/// that redraw from a newly completed response without treating prompt ids,
/// transcript mtimes, or hook arrival time as freshness evidence.
///
/// The digest is intentionally *not* an account identity. Missing evidence is
/// `None`, so callers fail closed and require an actual quota change before
/// advancing freshness.
pub fn api_generation(value: &Value) -> Option<String> {
    let cost = value.get("cost").and_then(Value::as_object);
    let context = value
        .get("context_window")
        .or_else(|| value.get("contextWindow"))
        .and_then(Value::as_object);
    let current = context
        .and_then(|context| {
            context
                .get("current_usage")
                .or_else(|| context.get("currentUsage"))
        })
        .and_then(Value::as_object);

    // Keep the fingerprint canonical: hash a fixed-order tuple of documented
    // scalar fields rather than serializing the current_usage object itself.
    // Object key order is not part of Claude's statusLine contract.
    let field = |object: Option<&serde_json::Map<String, Value>>, snake: &str, camel: &str| {
        object
            .and_then(|object| object.get(snake).or_else(|| object.get(camel)))
            .cloned()
    };
    let evidence = serde_json::json!([
        field(cost, "total_api_duration_ms", "totalApiDurationMs"),
        field(context, "total_input_tokens", "totalInputTokens"),
        field(context, "total_output_tokens", "totalOutputTokens"),
        field(current, "input_tokens", "inputTokens"),
        field(current, "output_tokens", "outputTokens"),
        field(
            current,
            "cache_creation_input_tokens",
            "cacheCreationInputTokens"
        ),
        field(current, "cache_read_input_tokens", "cacheReadInputTokens"),
    ]);
    let has_evidence = evidence
        .as_array()
        .is_some_and(|values| values.iter().any(|value| !value.is_null()));
    has_evidence.then(|| format!("{:x}", Sha256::digest(evidence.to_string().as_bytes())))
}

/// Accumulate cache counters from the provider session transcript.
///
/// StatusLine's `current_usage` is deliberately a latest-request view. The
/// transcript is the only local source that lets us present a session total,
/// so this function stores a byte offset and reads only appended complete
/// lines after the first call. It never starts a model turn or contacts a
/// provider.
pub fn enrich_cache_session(
    snapshot: &mut ProviderSnapshot,
    statusline: &Value,
    previous_cache: Option<&CacheUsage>,
) {
    let Some(context) = snapshot.context.as_mut() else {
        return;
    };
    let Some(cache) = context.cache.as_mut() else {
        return;
    };
    let Some(session_id) = statusline
        .get("session_id")
        .or_else(|| statusline.get("sessionId"))
        .or_else(|| statusline.get("conversation_id"))
        .or_else(|| statusline.get("conversationId"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return;
    };
    // Keep the session boundary even when this payload has no transcript
    // path. That prevents a later statusLine update from inheriting another
    // session's cache totals.
    cache.session_id = Some(session_id.to_string());
    let Some(path) = statusline.get("transcript_path").and_then(Value::as_str) else {
        return;
    };

    let matching_previous =
        previous_cache.filter(|previous| previous.session_id.as_deref() == Some(session_id));
    let previous_offset = matching_previous
        .map(|previous| previous.transcript_offset)
        .unwrap_or_default();
    let mut increment_totals: Option<CacheTotals> = None;
    let Some((next_offset, transcript_reset)) = read_transcript_increment(
        Path::new(path),
        if matching_previous.is_some() {
            previous_offset
        } else {
            0
        },
        |line| {
            let Ok(entry) = serde_json::from_str::<Value>(line) else {
                return;
            };
            let Some(usage) = assistant_usage(&entry) else {
                return;
            };
            let fresh = token_count(usage, "input_tokens", "inputTokens");
            let read = token_count(usage, "cache_read_input_tokens", "cacheReadInputTokens");
            let creation = token_count(
                usage,
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
            );
            if fresh == 0 && read == 0 && creation == 0 {
                return;
            }
            if let Some(existing) = increment_totals.as_mut() {
                existing.add_token_counts(fresh, read, creation);
            } else {
                increment_totals = CacheTotals::from_token_counts(fresh, read, creation);
            }
        },
    ) else {
        return;
    };

    let mut totals = if transcript_reset {
        None
    } else {
        matching_previous.and_then(|previous| previous.session_totals.clone())
    };
    if let Some(increment) = increment_totals {
        if let Some(existing) = totals.as_mut() {
            existing.add_token_counts(
                increment.fresh_input_tokens,
                increment.read_tokens,
                increment.creation_tokens,
            );
        } else {
            totals = Some(increment);
        }
    }

    cache.session_totals = totals;
    cache.transcript_offset = next_offset;
}

fn read_transcript_increment<F>(path: &Path, offset: u64, mut on_line: F) -> Option<(u64, bool)>
where
    F: FnMut(&str),
{
    let mut file = File::open(path).ok()?;
    let length = file.metadata().ok()?.len();
    let transcript_reset = offset > length;
    let start = if transcript_reset { 0 } else { offset };
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut reader = BufReader::new(file.take(length - start));
    let mut line = Vec::new();
    let mut next_offset = start;
    loop {
        line.clear();
        let bytes_read = reader.read_until(b'\n', &mut line).ok()?;
        if bytes_read == 0 || line.last() != Some(&b'\n') {
            break;
        }
        next_offset += bytes_read as u64;
        let content = &line[..line.len() - 1];
        on_line(&String::from_utf8_lossy(content));
    }
    Some((next_offset, transcript_reset))
}

fn assistant_usage(entry: &Value) -> Option<&serde_json::Map<String, Value>> {
    if entry.get("type").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    entry
        .get("message")
        .and_then(Value::as_object)
        .and_then(|message| message.get("usage"))
        .or_else(|| entry.get("usage"))
        .and_then(Value::as_object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ContextUsage, Provider, UsageWindow, WindowKind};
    use serde_json::json;
    use std::io::Write;

    fn snapshot_with_cache() -> ProviderSnapshot {
        ProviderSnapshot::new(
            Provider::Claude,
            vec![UsageWindow::new(WindowKind::Weekly, 10.0, None).unwrap()],
            0,
        )
        .with_context(Some(
            ContextUsage::new(24.0)
                .unwrap()
                .with_cache(CacheUsage::from_token_counts(1, 1, 1)),
        ))
    }

    #[test]
    fn model_parser_prefers_the_display_name_and_never_drops_a_reported_id() {
        assert_eq!(
            parse_model(&json!({
                "model": {"id": "claude-opus-5-5", "display_name": "Opus 5.5"}
            })),
            Some("Opus 5.5".to_string())
        );
        assert_eq!(
            parse_model(&json!({"model": {"displayName": "Sonnet"}})),
            Some("Sonnet".to_string())
        );
        for id in ["claude-opus-5-5", "zz-unreleased-model-9"] {
            assert_eq!(
                parse_model(&json!({"model": {"id": id, "display_name": " "}})),
                Some(id.to_string())
            );
        }
        assert_eq!(parse_model(&json!({"model": {"id": ""}})), None);
    }

    #[test]
    fn api_generation_changes_only_with_api_derived_evidence() {
        let base = json!({
            "session_id": "session-1",
            "prompt_id": "prompt-a",
            "cost": {"total_api_duration_ms": 1200},
            "context_window": {
                "total_input_tokens": 100,
                "total_output_tokens": 20,
                "current_usage": {"input_tokens": 80}
            }
        });
        let first = api_generation(&base).unwrap();

        let mut prompt_only = base.clone();
        prompt_only["prompt_id"] = json!("prompt-b");
        assert_eq!(
            api_generation(&prompt_only).as_deref(),
            Some(first.as_str())
        );

        let mut unrelated_current_usage_field = base.clone();
        unrelated_current_usage_field["context_window"]["current_usage"]["future_field"] =
            json!("ignored");
        assert_eq!(
            api_generation(&unrelated_current_usage_field).as_deref(),
            Some(first.as_str())
        );

        let mut next_response = base.clone();
        next_response["cost"]["total_api_duration_ms"] = json!(1800);
        assert_ne!(
            api_generation(&next_response).as_deref(),
            Some(first.as_str())
        );

        assert_eq!(api_generation(&json!({"prompt_id":"only-a-prompt"})), None);
    }

    #[test]
    fn zero_cache_reads_remain_a_real_zero_percent_hit() {
        let value = json!({
            "used_percentage": 3.4,
            "current_usage": {
                "input_tokens": 25943,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0
            }
        });
        let context = parse_context(Some(&value)).unwrap().unwrap();
        let cache = context.cache.unwrap();
        assert_eq!(cache.fresh_input_tokens, 25943);
        assert_eq!(cache.read_tokens, 0);
        assert_eq!(cache.creation_tokens, 0);
        assert_eq!(cache.hit_percent, 0.0);
    }

    #[test]
    fn accumulates_session_cache_counters_once_per_transcript_offset() {
        let transcript = tempfile::NamedTempFile::new().unwrap();
        let first = br#"{"type":"assistant","timestamp":"2026-08-22T10:00:00Z","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":800,"cache_creation_input_tokens":100,"cache_creation":{"ephemeral_1h_input_tokens":100}}}}
{"type":"assistant","timestamp":"2026-08-22T10:01:00Z","message":{"usage":{"input_tokens":50,"cache_read_input_tokens":450,"cache_creation_input_tokens":0}}}
"#;
        std::fs::write(transcript.path(), first).unwrap();
        let statusline = json!({
            "session_id": "session-1",
            "transcript_path": transcript.path(),
        });

        let mut first_snapshot = snapshot_with_cache();
        enrich_cache_session(&mut first_snapshot, &statusline, None);
        let first_cache = first_snapshot.context.unwrap().cache.unwrap();
        let first_totals = first_cache.session_totals.clone().unwrap();
        assert_eq!(first_totals.fresh_input_tokens, 150);
        assert_eq!(first_totals.read_tokens, 1_250);
        assert_eq!(first_totals.creation_tokens, 100);
        assert!(first_cache.ttl_seconds.is_none());
        assert!(first_cache.last_activity_unix.is_none());
        assert!(first_cache.transcript_offset > 0);

        let third = br#"{"type":"assistant","timestamp":"2026-08-22T10:02:00Z","message":{"usage":{"input_tokens":20,"cache_read_input_tokens":180,"cache_creation_input_tokens":0}}}
"#;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(transcript.path())
            .unwrap();
        file.write_all(third).unwrap();

        let mut second_snapshot = snapshot_with_cache();
        enrich_cache_session(&mut second_snapshot, &statusline, Some(&first_cache));
        let second_cache = second_snapshot.context.unwrap().cache.unwrap();
        let second_totals = second_cache.session_totals.as_ref().unwrap();
        assert_eq!(second_totals.fresh_input_tokens, 170);
        assert_eq!(second_totals.read_tokens, 1_430);
        assert_eq!(second_totals.creation_tokens, 100);

        let mut unchanged_snapshot = snapshot_with_cache();
        enrich_cache_session(&mut unchanged_snapshot, &statusline, Some(&second_cache));
        assert_eq!(
            unchanged_snapshot
                .context
                .unwrap()
                .cache
                .unwrap()
                .session_totals,
            second_cache.session_totals
        );
    }
}
