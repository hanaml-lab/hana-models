"""Preference pairs from the student's own mistakes: where the SFT student's output differs from
the teacher's, the student's output is the rejected side and the teacher's the chosen side."""

from __future__ import annotations

import json

from distill_kit.eval import field_f1


def build_pairs(train: list[dict], student: dict[str, dict], prompts: dict[str, str],
                max_f1: float = 0.999) -> list[dict]:
    """``train`` rows have ``id`` and ``target`` (the filtered teacher record); ``student`` maps
    id to the student's parsed output (None when it did not parse)."""
    pairs = []
    for row in train:
        out = student.get(row["id"])
        chosen = json.dumps(row["target"], ensure_ascii=False, sort_keys=True)
        if out is None:
            rejected, score = "{}", 0.0
        else:
            tp, npred, ngold = field_f1(out, row["target"])
            score = 0.0 if tp == 0 else 2 * tp / (npred + ngold)
            rejected = json.dumps(out, ensure_ascii=False, sort_keys=True)
        if score <= max_f1 and rejected != chosen:
            pairs.append({"id": row["id"], "prompt": prompts[row["id"]], "chosen": chosen, "rejected": rejected,
                          "student_f1": round(score, 4)})
    return pairs
