//! Loading a recorded trace and scheduling its requests at their original offsets.

use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Duration;

/// One recorded request. `messages` are sent as recorded, so shared prefixes (system prompts,
/// retrieved passages) reach the server exactly as they did in production.
#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    pub offset_ms: u64,
    pub messages: Vec<serde_json::Value>,
    pub max_tokens: u32,
    #[serde(default)]
    pub prompt_tokens: u32,
}

pub fn load(path: &Path) -> Result<Vec<Request>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "zst") {
        Box::new(
            ruzstd::streaming_decoder::StreamingDecoder::new(file)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        )
    } else {
        Box::new(file)
    };
    let mut out = Vec::new();
    for (n, line) in BufReader::new(reader).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut r: Request = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        if r.prompt_tokens == 0 {
            r.prompt_tokens = estimate_tokens(&r.messages);
        }
        out.push(r);
    }
    out.sort_by_key(|r| r.offset_ms);
    Ok(out)
}

/// About four characters per token, when the trace did not record the count.
pub fn estimate_tokens(messages: &[serde_json::Value]) -> u32 {
    let chars: usize = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .map(str::len)
        .sum();
    (chars / 4).max(1) as u32
}

/// When to send each request, relative to the start of the run. The recorded inter-arrival
/// times are kept and scaled by `speed` (2.0 replays twice as fast); requests after `duration`
/// are dropped. A trace shorter than the duration is not looped.
pub fn plan(
    requests: &[Request],
    speed: f64,
    duration: Option<Duration>,
) -> Vec<(Duration, usize)> {
    assert!(speed > 0.0, "speed must be positive");
    let start = requests.first().map_or(0, |r| r.offset_ms);
    requests
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (
                Duration::from_secs_f64((r.offset_ms - start) as f64 / 1000.0 / speed),
                i,
            )
        })
        .filter(|(at, _)| duration.is_none_or(|d| *at <= d))
        .collect()
}

/// Parse "15m", "90s", "2h" or plain seconds.
pub fn parse_duration(s: &str) -> Result<Duration, String> {
    let (num, unit) = s.split_at(
        s.find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(s.len()),
    );
    let v: f64 = num.parse().map_err(|_| format!("bad duration {s}"))?;
    let secs = match unit {
        "" | "s" => v,
        "m" => v * 60.0,
        "h" => v * 3600.0,
        _ => return Err(format!("bad duration unit in {s}")),
    };
    Ok(Duration::from_secs_f64(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(offset_ms: u64) -> Request {
        Request {
            offset_ms,
            messages: vec![],
            max_tokens: 8,
            prompt_tokens: 1,
        }
    }

    #[test]
    fn keeps_inter_arrival_times_scaled_by_speed() {
        let rs = vec![req(1000), req(1500), req(3000)];
        let p = plan(&rs, 2.0, None);
        assert_eq!(
            p.iter().map(|(d, _)| d.as_millis()).collect::<Vec<_>>(),
            vec![0, 250, 1000]
        );
        assert_eq!(plan(&rs, 1.0, Some(Duration::from_millis(600))).len(), 2);
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("15m").unwrap(), Duration::from_secs(900));
        assert_eq!(parse_duration("2.5").unwrap(), Duration::from_millis(2500));
        assert!(parse_duration("5d").is_err());
    }
}
