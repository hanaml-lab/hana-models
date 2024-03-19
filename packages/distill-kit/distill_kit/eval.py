"""Score a run on the task and report its general ability beside it, so a regression cannot hide.

Task score is field-level F1 between predicted and gold records. The general score is read from
the run folder (``mt_bench.json``, written by FastChat's ``gen_judgment`` / ``show_result``),
because running MT-Bench needs a GPU and a judge model.
"""

from __future__ import annotations

import json
from pathlib import Path


def _norm(v) -> str:
    return " ".join(str(v).split()).casefold() if v is not None else ""


def field_f1(pred: dict, gold: dict) -> tuple[int, int, int]:
    """(true positives, predicted fields, gold fields), ignoring nulls."""
    p = {k: _norm(v) for k, v in pred.items() if v is not None}
    g = {k: _norm(v) for k, v in gold.items() if v is not None}
    tp = sum(1 for k, v in p.items() if g.get(k) == v)
    return tp, len(p), len(g)


def f1(preds: list[dict], golds: list[dict]) -> float:
    tp = npred = ngold = 0
    for p, g in zip(preds, golds, strict=True):
        a, b, c = field_f1(p, g)
        tp, npred, ngold = tp + a, npred + b, ngold + c
    if tp == 0:
        return 0.0
    precision, recall = tp / npred, tp / ngold
    return 2 * precision * recall / (precision + recall)


def _read(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def evaluate(run: str | Path, task: str | Path, general: str | None = "mt-bench") -> dict:
    """``run``: folder with ``predictions.jsonl`` ({"id", "prediction"}); ``task``: folder with
    ``gold.jsonl`` ({"id", "target"})."""
    run, task = Path(run), Path(task)
    preds = _read(run / "predictions.jsonl")
    gold = _read(task / "gold.jsonl")
    ids = sorted(gold)
    missing = [i for i in ids if i not in preds]
    predicted = [preds.get(i, {}).get("prediction") or {} for i in ids]
    out = {
        "task_f1": round(f1(predicted, [gold[i]["target"] for i in ids]), 4),
        "documents": len(ids),
        "missing_predictions": len(missing),
    }
    if general == "mt-bench":
        mt = run / "mt_bench.json"
        out["mt_bench"] = json.loads(mt.read_text())["score"] if mt.exists() else None
    return out
