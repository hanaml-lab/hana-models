"""Cost per million output tokens, counting only tokens served within the latency target."""

from __future__ import annotations

import json
from pathlib import Path


def requests(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "requests.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def tokens_within_slo(rows: list[dict], ttft_ms: float | None) -> int:
    return sum(r["output_tokens"] for r in rows
               if r["ok"] and (ttft_ms is None or (r["ttft_ms"] is not None and r["ttft_ms"] <= ttft_ms)))


def cost_per_million(wall_s: float, gpus: int, gpu_hour: float, tokens: int) -> float | None:
    """Price of the GPU time the run took, divided by the useful tokens it produced."""
    if tokens == 0:
        return None
    return gpu_hour * gpus * (wall_s / 3600.0) / (tokens / 1e6)
