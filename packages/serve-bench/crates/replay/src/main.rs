//! serve-bench: replay recorded traffic against an LLM server.
//!
//!   serve-bench replay --trace traces/rag-sample.jsonl.zst --endpoint http://localhost:8000/v1 \
//!       --model meta-llama/Llama-3.1-70B-Instruct --speed 1.0 --duration 15m --out runs/fp8-eagle
//!   serve-bench accuracy ...   and   serve-bench report ...   run the Python steps (python/serve_bench)

mod client;
mod schedule;

use serde::Serialize;
use serve_bench_metrics::{Recorder, SeriesSummary};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

struct Args {
    trace: PathBuf,
    endpoint: String,
    model: String,
    speed: f64,
    duration: Option<Duration>,
    out: PathBuf,
    max_inflight: usize,
}

fn usage() -> ExitCode {
    eprintln!("usage: serve-bench replay --trace FILE --endpoint URL [--model NAME] [--speed X] [--duration 15m] --out DIR [--max-inflight N]");
    eprintln!("       serve-bench accuracy ... | serve-bench report ...   (python -m serve_bench.<command>)");
    ExitCode::from(2)
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        trace: PathBuf::new(),
        endpoint: String::new(),
        model: "default".into(),
        speed: 1.0,
        duration: None,
        out: PathBuf::new(),
        max_inflight: 512,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut val = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--trace" => a.trace = val()?.into(),
            "--endpoint" => a.endpoint = val()?,
            "--model" => a.model = val()?,
            "--speed" => a.speed = val()?.parse().map_err(|_| "bad --speed")?,
            "--duration" => a.duration = Some(schedule::parse_duration(&val()?)?),
            "--out" => a.out = val()?.into(),
            "--max-inflight" => {
                a.max_inflight = val()?.parse().map_err(|_| "bad --max-inflight")?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.trace.as_os_str().is_empty() || a.endpoint.is_empty() || a.out.as_os_str().is_empty() {
        return Err("--trace, --endpoint and --out are required".into());
    }
    Ok(a)
}

#[derive(Serialize)]
struct Summary {
    trace: String,
    endpoint: String,
    model: String,
    speed: f64,
    wall_s: f64,
    output_tokens_per_s: f64,
    overall: SeriesSummary,
    by_prompt_tokens: BTreeMap<String, SeriesSummary>,
}

fn replay(a: Args) -> Result<(), String> {
    let requests = schedule::load(&a.trace)?;
    let plan = schedule::plan(&requests, a.speed, a.duration);
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let (tx, rx) = mpsc::channel();
    let start = Instant::now();
    let inflight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(plan.len());
    for (at, i) in plan {
        if let Some(wait) = at.checked_sub(start.elapsed()) {
            thread::sleep(wait);
        }
        while inflight.load(std::sync::atomic::Ordering::Acquire) >= a.max_inflight {
            thread::sleep(Duration::from_millis(1));
        }
        inflight.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let (tx, r, endpoint, model, inflight) = (
            tx.clone(),
            requests[i].clone(),
            a.endpoint.clone(),
            a.model.clone(),
            inflight.clone(),
        );
        handles.push(thread::spawn(move || {
            let agent = client::agent(Duration::from_secs(600));
            let sample = client::send(&agent, &endpoint, &model, &r);
            inflight.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            let _ = tx.send((i, sample));
        }));
    }
    drop(tx);
    for h in handles {
        let _ = h.join();
    }
    let wall = start.elapsed().as_secs_f64();
    let mut rec = Recorder::default();
    let mut lines = String::new();
    for (i, s) in rx {
        rec.add(&s);
        lines.push_str(&serde_json::json!({
            "index": i, "ok": s.ok, "prompt_tokens": s.prompt_tokens, "output_tokens": s.output_tokens,
            "ttft_ms": s.ttft_us.map(|t| t as f64 / 1000.0), "e2e_ms": s.e2e_us as f64 / 1000.0,
        }).to_string());
        lines.push('\n');
    }
    let summary = Summary {
        trace: a.trace.display().to_string(),
        endpoint: a.endpoint,
        model: a.model,
        speed: a.speed,
        wall_s: wall,
        output_tokens_per_s: rec.all.output_tokens as f64 / wall,
        overall: (&rec.all).into(),
        by_prompt_tokens: rec
            .by_bucket
            .iter()
            .map(|(k, v)| (k.clone(), v.into()))
            .collect(),
    };
    std::fs::write(a.out.join("requests.jsonl"), lines).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    std::fs::write(a.out.join("summary.json"), &json).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("replay") => match parse(&args[1..]).and_then(replay) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("serve-bench: {e}");
                ExitCode::FAILURE
            }
        },
        Some(cmd @ ("accuracy" | "report")) => {
            let python = std::env::var("SERVE_BENCH_PYTHON").unwrap_or_else(|_| "python3".into());
            match Command::new(python)
                .arg("-m")
                .arg(format!("serve_bench.{cmd}"))
                .args(&args[1..])
                .status()
            {
                Ok(s) if s.success() => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(e) => {
                    eprintln!(
                        "serve-bench: cannot run the Python step ({e}); pip install ./python"
                    );
                    ExitCode::FAILURE
                }
            }
        }
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A minimal OpenAI-compatible server that streams five tokens 20 ms apart after 50 ms.
    fn fake_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let mut s = stream.unwrap();
                thread::spawn(move || {
                    let mut buf = [0u8; 65536];
                    let _ = s.read(&mut buf);
                    let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                    thread::sleep(Duration::from_millis(50));
                    for t in 0..5 {
                        let _ = write!(
                            s,
                            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{t} \"}}}}]}}\n\n"
                        );
                        let _ = s.flush();
                        thread::sleep(Duration::from_millis(20));
                    }
                    let _ = s.write_all(b"data: {\"choices\":[],\"usage\":{\"completion_tokens\":5}}\n\ndata: [DONE]\n\n");
                });
            }
        });
        format!("http://{addr}/v1")
    }

    #[test]
    fn replay_against_a_streaming_server() {
        let endpoint = fake_server();
        let dir = std::env::temp_dir().join(format!("serve-bench-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let trace = dir.join("t.jsonl");
        std::fs::write(&trace, (0..6).map(|i| format!("{{\"offset_ms\":{},\"messages\":[{{\"role\":\"user\",\"content\":\"q{i}\"}}],\"max_tokens\":5}}\n", i * 30)).collect::<String>()).unwrap();
        let args = Args {
            trace,
            endpoint,
            model: "m".into(),
            speed: 1.0,
            duration: None,
            out: dir.join("run"),
            max_inflight: 8,
        };
        replay(args).unwrap();
        let s: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("run/summary.json")).unwrap())
                .unwrap();
        assert_eq!(s["overall"]["requests"], 6);
        assert_eq!(s["overall"]["errors"], 0);
        assert_eq!(s["overall"]["output_tokens"], 30);
        let ttft = s["overall"]["ttft"]["p50_ms"].as_f64().unwrap();
        assert!((45.0..200.0).contains(&ttft), "ttft {ttft}");
        let itl = s["overall"]["itl"]["p50_ms"].as_f64().unwrap();
        assert!((15.0..80.0).contains(&itl), "itl {itl}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
