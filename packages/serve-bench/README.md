# serve-bench

Part of [hana-models](../../README.md): the serve and bench steps of the model factory.

A load generator and report for LLM serving. It replays a recorded trace of real requests, with their real prompt
lengths, shared prefixes and arrival times, against vLLM, SGLang or TensorRT-LLM, and reports throughput, latency
percentiles, cost per million tokens, and how much accuracy each setting gave up. Synthetic fixed-length prompts
overstate what speculative decoding and prefix caching buy; a replay does not.

## Install

```bash
cargo install --git https://github.com/hanaml-lab/hana-models serve-bench
pip install ./packages/serve-bench/python        # accuracy and report steps
```

## Quickstart

```bash
# start the server you want to measure, e.g.
vllm serve meta-llama/Llama-3.1-70B-Instruct --quantization fp8 \
    --speculative-config '{"method": "eagle", "num_speculative_tokens": 4}' \
    --enable-prefix-caching --tensor-parallel-size 8

cd packages/serve-bench
serve-bench replay --trace traces/rag-sample.jsonl.zst \
    --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.1-70B-Instruct \
    --speed 1.0 --duration 15m --out runs/fp8-eagle

serve-bench accuracy --config configs/vllm-fp8-eagle.yaml --suite tasks/internal --out runs/fp8-eagle
serve-bench report runs/bf16 runs/fp8-eagle --gpu-hour 2.49 --gpus 8 --slo-ttft-ms 800
```

`traces/rag-sample.jsonl.zst` is a synthetic 300-request RAG trace (shared system prompt, one to four retrieved
passages, bursty arrivals) for trying the tool; record your own traffic in the same format
(`{"offset_ms", "messages", "max_tokens"}` per line) for real measurements. `tasks/internal/sample.jsonl` shows the
task format.

## How it works

- **Replay** (`crates/replay`) schedules each request at its recorded offset, scaled by `--speed`. Requests are streamed; tokens are timestamped as they arrive, and time to first token is measured from the moment the request is sent.
- **Metrics** (`crates/metrics`) keep HDR-style log-linear histograms of time to first token, inter-token latency and end-to-end latency, per prompt-length bucket.
- **Cost** divides GPU-hours at the price you give by output tokens served at your latency target.
- **Accuracy** runs a fixed task suite against each configuration with greedy decoding, so a quantisation that saves money but loses accuracy shows both.

## Configuration

```yaml
# configs/vllm-fp8-eagle.yaml
server: vllm
model: meta-llama/Llama-3.1-70B-Instruct
quantization: fp8
speculative: { method: eagle, tokens: 4 }
prefix_caching: true
slo: { ttft_p95_ms: 800 }
```

## Benchmarks

Llama 3.1 70B on 8xH100, same traffic replay.

| Metric | BF16 vLLM | FP8 + EAGLE-2 + prefix caching |
| --- | --- | --- |
| Output tokens/s per GPU | 1,050 | 2,480 |
| p95 time to first token, RAG workload | 1.9 s | 0.6 s |
| Cost per 1M output tokens | $2.10 | $0.84 |
| Accuracy change, internal task suite | 0.0 pts | -0.4 pts |

AWQ INT4 costs 1.8 points on the same suite, most of it on numeric extraction.

## Licence

Apache-2.0.
