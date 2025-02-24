"""Compare replay runs side by side.

    python -m serve_bench.report runs/bf16 runs/fp8-eagle --gpu-hour 2.49 --gpus 8 [--slo-ttft-ms 800]

The first run is the baseline for the accuracy delta. Each run folder holds summary.json from
the replay and, if the accuracy step ran, accuracy.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from serve_bench.cost import cost_per_million, requests, tokens_within_slo


def row(run_dir: Path, gpus: int, gpu_hour: float, slo: float | None) -> dict:
    s = json.loads((run_dir / "summary.json").read_text())
    acc_path = run_dir / "accuracy.json"
    acc = json.loads(acc_path.read_text())["accuracy"] if acc_path.exists() else None
    useful = tokens_within_slo(requests(run_dir), slo)
    return {
        "run": run_dir.name,
        "output_tokens_per_s_per_gpu": round(s["output_tokens_per_s"] / gpus, 1),
        "ttft_p95_ms": s["overall"]["ttft"]["p95_ms"],
        "itl_p50_ms": s["overall"]["itl"]["p50_ms"],
        "e2e_p95_ms": s["overall"]["e2e"]["p95_ms"],
        "errors": s["overall"]["errors"],
        "cost_per_1m_output_tokens": (round(c, 4) if (c := cost_per_million(s["wall_s"], gpus, gpu_hour, useful))
                                      is not None else None),
        "accuracy": acc,
    }


def table(rows: list[dict]) -> str:
    base = rows[0]["accuracy"]
    cols = ["run", "output_tokens_per_s_per_gpu", "ttft_p95_ms", "itl_p50_ms", "e2e_p95_ms",
            "cost_per_1m_output_tokens", "accuracy_delta_pts", "errors"]
    for r in rows:
        r["accuracy_delta_pts"] = (round(100 * (r["accuracy"] - base), 2)
                                   if r["accuracy"] is not None and base is not None else None)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="serve-bench report")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gpu-hour", type=float, required=True, help="price of one GPU-hour")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--slo-ttft-ms", type=float, default=None, help="count only tokens of requests within this TTFT")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = [row(Path(r), args.gpus, args.gpu_hour, args.slo_ttft_ms) for r in args.runs]
    print(json.dumps(rows, indent=2) if args.json else table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
