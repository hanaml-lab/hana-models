# Changelog

## 0.6.0 (2026-10-14)

- distill-kit `clean`: personal data redacted (email, phone, card, IBAN, NI number) and Gopher-style quality rules
  before any document reaches the teacher; the factory's distil step reads only cleaned documents
- distill-kit `plan` (GATE): teacher-conditional coverage picks the student size, and each prompt gets GATE's weight
  and a signal (teacher, verifier, replay or drop); shown in the end-to-end example
- distill-kit: LoRA recipes, with Llama 3.2 3B on one 24 GB GPU; a recipe is ZeRO-3 or LoRA, not both
- serve-bench 0.8.0: configurations are checked; vLLM BF16, vLLM AWQ INT4 and TensorRT-LLM FP8 configurations
- README with results, papers and citation; CITATION.cff; CONTRIBUTING.md
- CI: read-only token, concurrency, caches, a locked sync and a citation check

## 0.5.0 (2026-10-08)

- hana-models pipeline and end-to-end example
- integration test across the three packages

## 0.4.0 (2026-06-29)

- small-lm: Android app and llama.cpp bridge
- offline stand-ins for the factory

## 0.3.0 (2025-12-08)

- distill-kit: DPO pairs and CLI
- small-lm: quantisation check, notes index, calibrated router

## 0.2.0 (2025-03-05)

- serve-bench: trace replay, latency histograms, cost and accuracy reports

## 0.1.0 (2024-07-26)

- distill-kit data pipeline: teacher sampling, schema filter, MinHash dedup
- SFT recipe
