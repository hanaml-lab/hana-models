# distill-kit

Part of [hana-models](../../README.md): the train and distil steps of the model factory.

Recipes for turning a frontier model's behaviour on one task into a small open model you can run yourself. You give
it documents and a schema; it cleans the documents, collects teacher outputs, filters them, deduplicates the data,
plans the run with GATE, fine-tunes a student, then runs a round of preference tuning on the student's own mistakes.
It is the pipeline behind the lab's `clause-extract-8b`.

## Install

```bash
# from the root of hana-models
pip install "./packages/distill-kit[train]"    # pulls TRL, Transformers and DeepSpeed
```

The full fine-tune recipes assume 8 GPUs with 80 GB each; the LoRA recipe fits one 24 GB GPU. The data and planning
steps run on a laptop.

## Quickstart

Run from `packages/distill-kit`, with your documents in `data/claims/`:

```bash
# 0. Clean: redact personal data, drop documents that fail the quality rules
distill-kit clean --docs data/claims/ --out data/clean/

# 1. Teacher traces: the teacher fills the schema for each document (OPENAI_BASE_URL, OPENAI_API_KEY)
distill-kit teach --docs data/clean/ --schema schemas/claim.json \
    --teacher api:frontier --out traces.jsonl      # TEACHER_MODEL names the model behind "frontier"

# 2. Filter and deduplicate
distill-kit filter traces.jsonl --schema schemas/claim.json --min-agree 2 --out clean.jsonl
distill-kit dedup clean.jsonl --minhash --threshold 0.8 --out train.jsonl

# 3. GATE plan: which student size, and which signal for each prompt
distill-kit plan outcomes.jsonl --out routes.jsonl

# 4. Supervised fine-tuning (full, or recipes/sft-llama3.2-3b-lora.yaml), then DPO on the student's errors
bash scripts/train.sh recipes/sft-qwen3-8b.yaml
distill-kit predict --model runs/sft --data train.jsonl --schema schemas/claim.json --out runs/sft/predictions.jsonl
distill-kit pairs --student runs/sft --data train.jsonl --schema schemas/claim.json --out pairs.jsonl
bash scripts/train.sh recipes/dpo-qwen3-8b.yaml
```

```python
from distill_kit.eval import evaluate
print(evaluate("runs/dpo", task="data/claims-heldout/", general="mt-bench"))
```

`outcomes.jsonl` holds one line per prompt and model, with the verifier's verdict on each sample:
`{"id": "claim-001", "model": "teacher", "correct": [true, true, false, true]}`, and for each candidate student also
`"params_b": 1.7`. The plan prints each student's coverage rho_k and the one it chose, and writes one route per
prompt: its solve rates, its regime, GATE's weight lambda(x) and the signal (`teacher`, `verifier`, `replay` or
`drop`).

`evaluate` reads `predictions.jsonl` from the run and `gold.jsonl` from the task folder; the MT-Bench score is
read from `mt_bench.json` in the run folder, written by FastChat's MT-Bench scripts.

## How it works

1. **Clean.** Email addresses, phone numbers, card numbers (Luhn checked), IBANs (mod-97 checked) and UK National Insurance numbers are replaced by tags, and documents are held to Gopher-style quality rules (length, repeated lines, symbols, share of words with letters, word length). Nothing else is sent to the teacher.
2. **Teacher.** Each document is sent to the teacher with the extraction template (`templates/extract.jinja`). Outputs that do not parse are repaired once, then dropped.
3. **Filter.** With `--min-agree 2`, a field is kept only when two teacher samples agree on it.
4. **Dedup.** MinHash over word 5-gram shingles, with LSH banding, removes near-duplicate documents, so the held-out set stays held out.
5. **Plan.** GATE's teacher-conditional coverage rho_k picks the smallest candidate student that clears 0.952, and each prompt gets the paper's weight lambda(x), so teachable prompts are distilled from the teacher, explorable ones trained on the verifier, solved ones kept for replay and unlearnable ones dropped ([paper](https://hanaml.com/research/papers/domain-distillation-rl), [reference trainer](https://github.com/hanaml-lab/domain-distillation-rl)).
6. **SFT.** Full fine-tune of the student on the filtered traces (TRL, DeepSpeed ZeRO-3), or a LoRA adapter where memory is short.
7. **DPO.** The student runs on the training set; where it is wrong, its output becomes the rejected side of a pair and the teacher's the chosen side.
8. **Eval.** Every run reports the task score and a general-ability score together, so a regression cannot hide.

## Benchmarks

Insurance claims extraction, held-out carriers. Student is Qwen3-8B.

| Metric | Prompted Qwen3-8B | SFT + DPO |
| --- | --- | --- |
| Extraction F1 | 0.72 | 0.86 |
| MT-Bench (general ability kept) | 8.1 | 7.9 |
| Cost per 1k documents, one L40S | $14.20 (teacher API) | $0.90 |

The frontier teacher scores 0.89 on the same task. DPO on 12k pairs wins 64% of pairwise comparisons against
SFT alone.

## Licence

Apache-2.0. The student inherits the licence of its base model.
