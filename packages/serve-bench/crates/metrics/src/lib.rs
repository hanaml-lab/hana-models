//! Latency metrics per prompt-length bucket: time to first token, inter-token latency and
//! end-to-end latency, in microseconds.

pub mod histogram;

use histogram::Histogram;
use serde::Serialize;

/// Prompt-length buckets in tokens; long RAG prompts behave differently from chat turns.
pub const BUCKETS: [(u32, u32); 4] = [(0, 512), (512, 2048), (2048, 8192), (8192, u32::MAX)];

pub fn bucket_name(prompt_tokens: u32) -> String {
    let (lo, hi) = BUCKETS
        .iter()
        .copied()
        .find(|(lo, hi)| prompt_tokens >= *lo && prompt_tokens < *hi)
        .unwrap();
    if hi == u32::MAX {
        format!("{lo}+")
    } else {
        format!("{lo}-{hi}")
    }
}

/// One completed (or failed) request.
#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub ttft_us: Option<u64>,
    pub itl_us: Vec<u64>,
    pub e2e_us: u64,
    pub ok: bool,
}

#[derive(Clone, Debug)]
pub struct Series {
    pub ttft: Histogram,
    pub itl: Histogram,
    pub e2e: Histogram,
    pub requests: u64,
    pub errors: u64,
    pub output_tokens: u64,
}

impl Default for Series {
    fn default() -> Self {
        Series {
            ttft: Histogram::new(7),
            itl: Histogram::new(7),
            e2e: Histogram::new(7),
            requests: 0,
            errors: 0,
            output_tokens: 0,
        }
    }
}

impl Series {
    pub fn add(&mut self, s: &Sample) {
        self.requests += 1;
        if !s.ok {
            self.errors += 1;
            return;
        }
        if let Some(t) = s.ttft_us {
            self.ttft.record(t);
        }
        for &d in &s.itl_us {
            self.itl.record(d);
        }
        self.e2e.record(s.e2e_us);
        self.output_tokens += s.output_tokens as u64;
    }
}

#[derive(Serialize, Debug)]
pub struct Percentiles {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
}

fn pct(h: &Histogram) -> Percentiles {
    let ms = |us: u64| us as f64 / 1000.0;
    Percentiles {
        p50_ms: ms(h.percentile(50.0)),
        p95_ms: ms(h.percentile(95.0)),
        p99_ms: ms(h.percentile(99.0)),
        mean_ms: h.mean() / 1000.0,
    }
}

#[derive(Serialize, Debug)]
pub struct SeriesSummary {
    pub requests: u64,
    pub errors: u64,
    pub output_tokens: u64,
    pub ttft: Percentiles,
    pub itl: Percentiles,
    pub e2e: Percentiles,
}

impl From<&Series> for SeriesSummary {
    fn from(s: &Series) -> Self {
        SeriesSummary {
            requests: s.requests,
            errors: s.errors,
            output_tokens: s.output_tokens,
            ttft: pct(&s.ttft),
            itl: pct(&s.itl),
            e2e: pct(&s.e2e),
        }
    }
}

/// All samples of a run, overall and by prompt-length bucket.
#[derive(Default)]
pub struct Recorder {
    pub all: Series,
    pub by_bucket: std::collections::BTreeMap<String, Series>,
}

impl Recorder {
    pub fn add(&mut self, s: &Sample) {
        self.all.add(s);
        self.by_bucket
            .entry(bucket_name(s.prompt_tokens))
            .or_default()
            .add(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_errors() {
        assert_eq!(bucket_name(100), "0-512");
        assert_eq!(bucket_name(4000), "2048-8192");
        assert_eq!(bucket_name(9000), "8192+");
        let mut r = Recorder::default();
        r.add(&Sample {
            prompt_tokens: 100,
            output_tokens: 10,
            ttft_us: Some(200_000),
            itl_us: vec![20_000; 9],
            e2e_us: 380_000,
            ok: true,
        });
        r.add(&Sample {
            prompt_tokens: 3000,
            ok: false,
            ..Default::default()
        });
        let s = SeriesSummary::from(&r.all);
        assert_eq!((s.requests, s.errors, s.output_tokens), (2, 1, 10));
        assert!((s.ttft.p50_ms - 200.0).abs() / 200.0 < 0.01);
        assert_eq!(r.by_bucket["2048-8192"].errors, 1);
    }
}
