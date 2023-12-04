"""Keep teacher traces that parse, validate and agree with each other.

Each trace line holds one document's teacher samples. A sample is parsed (and repaired once);
one that still fails the schema is dropped. With ``min_agree=2`` a field is kept only when at
least two valid samples give it the same value; a document left with no agreed field is dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from distill_kit.schema import parse


@dataclass
class FilterStats:
    documents: int = 0
    kept: int = 0
    samples: int = 0
    invalid_samples: int = 0
    fields_dropped: int = 0


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True)


def agree(samples: list[dict], min_agree: int) -> tuple[dict, int]:
    """Field-wise vote. Returns the agreed record and how many fields were dropped."""
    keys = {k for s in samples for k in s}
    out, dropped = {}, 0
    for key in sorted(keys):
        votes = Counter(_canon(s[key]) for s in samples if key in s)
        value, count = votes.most_common(1)[0]
        if count >= min_agree:
            out[key] = json.loads(value)
        else:
            dropped += 1
    return out, dropped


def filter_traces(rows: list[dict], schema: dict, min_agree: int = 2) -> tuple[list[dict], FilterStats]:
    stats = FilterStats()
    kept = []
    for row in rows:
        stats.documents += 1
        valid = []
        for text in row["samples"]:
            stats.samples += 1
            value, errs = parse(text, schema)
            if errs:
                stats.invalid_samples += 1
            else:
                valid.append(value)
        if len(valid) < min_agree:
            continue
        record, dropped = agree(valid, min_agree)
        stats.fields_dropped += dropped
        if not record:
            continue
        for key in schema.get("required", []):
            if key not in record:
                break
        else:
            kept.append({"id": row["id"], "document": row["document"], "target": record})
            stats.kept += 1
    return kept, stats


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
