//! Streaming chat completions against an OpenAI-compatible server, timestamped per token.

use crate::schedule::Request;
use serde_json::{json, Value};
use serve_bench_metrics::Sample;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

/// Send one request and time it. Time to first token is measured from the moment the request
/// is sent, not from the first byte of the response headers: a server that answers headers
/// at once and then queues the request would otherwise look fast.
pub fn send(agent: &ureq::Agent, endpoint: &str, model: &str, r: &Request) -> Sample {
    let body = json!({
        "model": model,
        "messages": r.messages,
        "max_tokens": r.max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let sent = Instant::now();
    let mut sample = Sample {
        prompt_tokens: r.prompt_tokens,
        ..Default::default()
    };
    let resp = match agent
        .post(&format!(
            "{}/chat/completions",
            endpoint.trim_end_matches('/')
        ))
        .send_json(body)
    {
        Ok(resp) => resp,
        Err(_) => {
            sample.e2e_us = sent.elapsed().as_micros() as u64;
            return sample;
        }
    };
    let mut last: Option<Instant> = None;
    let mut chunks = 0u32;
    let mut usage_tokens: Option<u32> = None;
    for line in BufReader::new(resp.into_reader()).lines() {
        let Ok(line) = line else { break };
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(n) = v
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
        {
            usage_tokens = Some(n as u32);
        }
        let has_text = v
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty());
        if !has_text {
            continue;
        }
        let now = Instant::now();
        match last {
            None => sample.ttft_us = Some(now.duration_since(sent).as_micros() as u64),
            Some(prev) => sample
                .itl_us
                .push(now.duration_since(prev).as_micros() as u64),
        }
        last = Some(now);
        chunks += 1;
    }
    sample.e2e_us = sent.elapsed().as_micros() as u64;
    sample.output_tokens = usage_tokens.unwrap_or(chunks);
    sample.ok = sample.ttft_us.is_some();
    sample
}

/// One HTTP connection pool per worker thread, so threads never contend for connections.
pub fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_read(timeout)
        .timeout_connect(Duration::from_secs(10))
        .max_idle_connections_per_host(64)
        .build()
}
