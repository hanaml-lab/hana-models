"""Run a fixed task suite against a serving configuration with greedy decoding.

    python -m serve_bench.accuracy --config configs/vllm-fp8-eagle.yaml --suite tasks/internal \
        --endpoint http://localhost:8000/v1 --out runs/fp8-eagle

Task files are JSON Lines: {"id", "prompt", "answer", "kind": "exact" | "number" | "contains"}.
The result goes to <out>/accuracy.json, next to the replay's summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

import yaml

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def grade(output: str, answer: str, kind: str = "exact", tolerance: float = 1e-6) -> bool:
    if kind == "number":
        want = float(answer.replace(",", ""))
        for n in _NUM.findall(output):
            try:
                got = float(n.replace(",", "").rstrip("."))
            except ValueError:
                continue
            if abs(got - want) <= tolerance * max(1.0, abs(want)):
                return True
        return False
    if kind == "contains":
        return answer.casefold() in output.casefold()
    return output.strip().casefold() == answer.strip().casefold()


def load_suite(folder: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        rows += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"no tasks in {folder}")
    return rows


def chat(endpoint: str, model: str) -> Callable[[str], str]:
    def call(prompt: str) -> str:
        body = {"model": model, "temperature": 0, "max_tokens": 256, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]

    return call


def run(suite: list[dict], generate: Callable[[str], str]) -> dict:
    by_kind: dict[str, list[bool]] = {}
    failures = []
    for t in suite:
        ok = grade(generate(t["prompt"]), t["answer"], t.get("kind", "exact"))
        by_kind.setdefault(t.get("kind", "exact"), []).append(ok)
        if not ok:
            failures.append(t["id"])
    every = [ok for v in by_kind.values() for ok in v]
    return {"tasks": len(every), "accuracy": round(sum(every) / len(every), 4),
            "by_kind": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())}, "failed": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="serve-bench accuracy")
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or f"runs/{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    result = {"config": cfg, **run(load_suite(args.suite), chat(args.endpoint, cfg["model"]))}
    (out / "accuracy.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("tasks", "accuracy", "by_kind")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
