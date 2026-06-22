# small-lm

Part of [hana-models](../../README.md): the on-device end of the model factory.

Tools for running 1–3B models where the data is: on a phone or a laptop, offline. It quantises a model to GGUF and
checks what the quantisation cost, builds a small local retrieval index, and adds a router that answers locally when
the model is likely to be right and escalates to a larger model, or abstains, when it is not. The lab's
`ondevice-summarise-1b` is built and measured with it.

## Install

```bash
# from the root of hana-models
pip install "./packages/small-lm[run,quantize]"
# Android: open packages/small-lm/android in Android Studio; set smalllm.native=true in gradle.properties to
# build native/ (llama.cpp with Vulkan, through CMake) into the app
```

## Quickstart

```bash
export LLAMA_CPP_DIR=~/src/llama.cpp     # a built llama.cpp checkout
small-lm quantize Qwen/Qwen3-1.7B --type Q4_K_M --out qwen3-1.7b-q4km.gguf --check tasks/internal
small-lm index docs/ --embed bge-small --out notes.sqlite
small-lm calibrate --model qwen3-1.7b-q4km.gguf --labels tasks/router-labels.jsonl --out router.json
```

`--check` runs `tasks.jsonl` from the folder on both models. Router labels are rows of
`{"features": [top_prob, entropy, log_len], "correct": 0 | 1}` collected by running the local model on labelled
questions (`small_lm.local.features`).

```python
from small_lm import Local, Router

local = Local("qwen3-1.7b-q4km.gguf", index="notes.sqlite", n_ctx=8192)
router = Router.load("router.json", escalate_to="https://api.example.com/v1", coverage=0.8)

answer = router.ask(local, "When is the next car insurance payment due?")
print(answer.text, answer.route)     # route: local | escalated | abstained
```

## How it works

- **Quantise.** GGUF export through llama.cpp, then the same task suite on the FP16 and the quantised model, so the loss is a number, not a guess. `quantize.block_quantize` reproduces the block arithmetic in numpy to compare bit widths on a layer.
- **Retrieve.** Notes and documents are embedded on the device into one SQLite file, searched with `sqlite-vec` when it is available; nothing leaves the phone to build or query the index.
- **Route.** A small classifier reads the question and the local model's first-token probabilities, and is calibrated on labelled examples so its confidence means what it says. Below the threshold set by `coverage`, the question is escalated or declined. The Android app uses the same `router.json` (`Router.kt`).
- **Measure.** `bench.py` reports sustained decode speed after warm-up and energy per completed task from a power-meter log.

## Configuration

```yaml
# configs/qwen3-1.7b-q4km.yaml
model: Qwen/Qwen3-1.7B
quant: Q4_K_M
n_ctx: 8192
router:
  coverage: 0.8
  on_low_confidence: escalate   # or: abstain
```

## Benchmarks

Qwen3-1.7B, internal task suite; speed and energy on a Pixel 8.

| Metric | Baseline | small-lm |
| --- | --- | --- |
| Routing accuracy, local vs. escalate (length heuristic baseline) | 71% | 86% |
| Selective accuracy at 80% coverage | 74% | 88% |
| Decode speed, tokens/s (FP16 vs. 4-bit) | 9.8 | 17.5 |
| Energy per completed task, incl. retrieval | 41 J | 19 J |

## Licence

Apache-2.0. Model weights keep the licences of their base models (Qwen, Llama, Gemma).
