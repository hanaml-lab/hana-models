"""Train -> distil -> serve -> bench, with each step done by the package that owns it."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from distill_kit.data.clean import CleanStats, clean_folder
from distill_kit.data.dedup import dedup
from distill_kit.data.filter import filter_traces
from distill_kit.eval import f1
from distill_kit.pairs import build_pairs
from distill_kit.teacher import render, replay_teacher, teach
from hana_models.simserver import PROFILES, serve
from hana_models.student import extract
from small_lm import calibrate as cal
from small_lm.local import Generation, features


@dataclass
class DistillResult:
    cleaned: CleanStats
    documents: int
    kept_after_filter: int
    kept_after_dedup: int
    invalid_teacher_samples: int
    train: list[dict]
    pairs: list[dict]
    student_f1_heldout: float


def distil(claims: Path, heldout: Path, schema: dict, teacher_samples: Path) -> DistillResult:
    with tempfile.TemporaryDirectory() as tmp:
        cleaned = clean_folder(claims, tmp)  # nothing reaches the teacher before it is cleaned
        traces = teach(tmp, schema, replay_teacher(teacher_samples), samples=3)
    kept, stats = filter_traces(traces, schema, min_agree=2)
    keep = dedup([r["document"] for r in kept], threshold=0.8)
    train = [kept[i] for i in keep]
    student = {r["id"]: extract(r["document"]).record for r in train}
    pairs = build_pairs(train, student, {r["id"]: render(r["document"], schema) for r in train})
    gold = [json.loads(x) for x in (heldout / "gold.jsonl").read_text().splitlines() if x.strip()]
    preds = [extract((heldout / f"{g['id']}.txt").read_text()).record for g in gold]
    return DistillResult(cleaned, len(traces), stats.kept, len(train), stats.invalid_samples, train, pairs,
                         round(f1(preds, [g["target"] for g in gold]), 4))


@dataclass
class RouteResult:
    coverage: float
    threshold: float
    accuracy_all: float
    selective_accuracy: float


def route(documents: list[tuple[str, dict]], coverage: float = 0.8) -> RouteResult:
    """Calibrate the small-lm router on whether the student got each document fully right."""
    X, y = [], []
    for text, gold in documents:
        ex = extract(text)
        X.append(features(text, Generation("", ex.first_token_logprobs())))
        y.append(float(all(ex.record.get(k) == v for k, v in gold.items())))
    X, y = np.array(X), np.array(y)
    model = cal.fit(X, y, seed=0)
    conf = model.confidence(X)
    return RouteResult(coverage, model.threshold(coverage), float(y.mean()),
                       cal.selective_accuracy(conf, y, coverage))


def find_serve_bench(root: Path) -> Path:
    for p in (root / "packages/serve-bench/target/release/serve-bench",
              root / "packages/serve-bench/target/debug/serve-bench"):
        if p.exists():
            return p
    found = shutil.which("serve-bench")
    if not found:
        raise FileNotFoundError("build it first: cargo build --release --manifest-path packages/serve-bench/Cargo.toml")
    return Path(found)


def bench(root: Path, out: Path, trace: Path, speed: float = 20.0, duration: str = "60s") -> dict[str, dict]:
    """Replay the sample trace against each simulated serving profile with the serve-bench binary."""
    binary = find_serve_bench(root)
    summaries = {}
    for name, profile in PROFILES.items():
        httpd, endpoint = serve(profile)
        try:
            subprocess.run([str(binary), "replay", "--trace", str(trace), "--endpoint", endpoint, "--model", name,
                            "--speed", str(speed), "--duration", duration, "--out", str(out / name)],
                           check=True, stdout=subprocess.DEVNULL)
        finally:
            httpd.shutdown()
        summaries[name] = json.loads((out / name / "summary.json").read_text())
    return summaries
