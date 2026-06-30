"""Train -> distil -> serve -> bench, offline: distill-kit data steps, the small-lm router, and the
serve-bench binary against simulated servers."""

import json
from pathlib import Path

import pytest

from hana_models.pipeline import bench, distil, route
from serve_bench.report import row

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples/data"


@pytest.fixture(scope="module")
def distilled():
    schema = json.loads((ROOT / "packages/distill-kit/schemas/claim.json").read_text())
    return distil(DATA / "claims", DATA / "heldout", schema, DATA / "teacher-samples.jsonl")


def test_distil_filters_dedups_and_pairs(distilled):
    assert distilled.documents == 41
    assert distilled.kept_after_filter < distilled.documents  # self-contradicting teacher output is dropped
    assert distilled.kept_after_dedup == distilled.kept_after_filter - 1  # the resubmitted claim
    assert distilled.pairs and all(json.loads(p["chosen"]) != json.loads(p["rejected"]) for p in distilled.pairs)
    assert 0.5 < distilled.student_f1_heldout < 1.0


def test_router_keeps_what_the_student_gets_right(distilled):
    r = route([(t["document"], t["target"]) for t in distilled.train], coverage=0.8)
    assert r.selective_accuracy > r.accuracy_all


def test_bench_replays_trace_against_both_profiles(tmp_path):
    summaries = bench(ROOT, tmp_path, ROOT / "packages/serve-bench/traces/rag-sample.jsonl.zst", speed=40,
                      duration="20s")
    assert set(summaries) == {"bf16", "fp8-eagle-prefix"}
    for s in summaries.values():
        assert s["overall"]["requests"] > 5 and s["overall"]["errors"] == 0
    fast, slow = (row(tmp_path / n, 8, 2.49, None) for n in ("fp8-eagle-prefix", "bf16"))
    assert fast["ttft_p95_ms"] < slow["ttft_p95_ms"]
    assert fast["itl_p50_ms"] < slow["itl_p50_ms"]
