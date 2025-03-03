import json

import pytest

from serve_bench.accuracy import grade, run
from serve_bench.cost import cost_per_million, tokens_within_slo
from serve_bench.report import row, table


def test_grading():
    assert grade("The total is 1,250.00 GBP.", "1250", "number")
    assert not grade("The total is 1,205.00 GBP.", "1250", "number")
    assert grade("  Leeds ", "leeds")
    assert grade("It renews on 1 November.", "1 November", "contains")


def test_run_counts_by_kind():
    suite = [{"id": "a", "prompt": "p1", "answer": "4", "kind": "number"},
             {"id": "b", "prompt": "p2", "answer": "yes"}]
    out = run(suite, lambda p: "4" if p == "p1" else "no")
    assert out["accuracy"] == 0.5 and out["failed"] == ["b"]


def test_cost_counts_only_tokens_within_slo():
    rows = [{"ok": True, "ttft_ms": 300, "output_tokens": 500_000},
            {"ok": True, "ttft_ms": 1200, "output_tokens": 500_000},
            {"ok": False, "ttft_ms": None, "output_tokens": 0}]
    assert tokens_within_slo(rows, 800) == 500_000
    assert cost_per_million(3600, 8, 2.49, 500_000) == pytest.approx(39.84)
    assert cost_per_million(3600, 8, 2.49, 0) is None


def test_report(tmp_path):
    for name, tps, acc in (("bf16", 8400.0, 0.80), ("fp8", 19840.0, 0.796)):
        d = tmp_path / name
        d.mkdir()
        lat = {"p50_ms": 100.0, "p95_ms": 200.0, "p99_ms": 300.0, "mean_ms": 120.0}
        (d / "summary.json").write_text(json.dumps({"output_tokens_per_s": tps, "wall_s": 900.0, "overall": {
            "errors": 0, "ttft": lat, "itl": lat, "e2e": lat}}))
        (d / "requests.jsonl").write_text(json.dumps({"ok": True, "ttft_ms": 150.0, "output_tokens": int(tps * 900)}))
        (d / "accuracy.json").write_text(json.dumps({"accuracy": acc}))
    rows = [row(tmp_path / n, 8, 2.49, 800) for n in ("bf16", "fp8")]
    assert rows[1]["cost_per_1m_output_tokens"] < rows[0]["cost_per_1m_output_tokens"]
    out = table(rows)
    assert "-0.4" in out and out.count("\n") == 3
