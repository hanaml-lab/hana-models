"""The task suite run on the FP16 and the quantised model, so the cost of quantising is a number.

A task folder holds ``tasks.jsonl`` rows ``{"prompt", "answer"}``; a row is right when the
normalised answer appears in the model's greedy output.
"""

from __future__ import annotations

import json
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def score(generate, tasks: list[dict]) -> float:
    right = sum(_norm(t["answer"]) in _norm(generate(t["prompt"])) for t in tasks)
    return right / max(len(tasks), 1)


def compare(fp16_model: str, gguf: str | Path, task_dir: str | Path) -> dict:
    tasks = [json.loads(x) for x in (Path(task_dir) / "tasks.jsonl").read_text().splitlines() if x.strip()]
    from llama_cpp import Llama
    from transformers import pipeline

    ref = pipeline("text-generation", model=fp16_model, torch_dtype="float16", device_map="auto")
    quant = Llama(model_path=str(gguf), n_ctx=4096, verbose=False)
    fp16 = score(lambda p: ref(p, max_new_tokens=64, do_sample=False, return_full_text=False)[0]["generated_text"],
                 tasks)
    q = score(lambda p: quant(p, max_tokens=64, temperature=0)["choices"][0]["text"], tasks)
    return {"tasks": len(tasks), "fp16": round(fp16, 4), "quantized": round(q, 4),
            "delta_pts": round(100 * (q - fp16), 2)}
