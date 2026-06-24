"""A rule-based stand-in for a trained student, so the factory runs end to end on a laptop.

It reads claim forms with regular expressions and reports a confidence per field. Its mistakes
are the kind a small model makes (it reads "No police report was made" as a police report), which
gives distill-kit real errors to turn into preference pairs and small-lm real uncertainty to
route on. Swap in a GGUF model through small_lm.Local for the real thing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], start=1)}
KEYWORDS = {"collision": ["struck", "collision", "crash"], "theft": ["taken", "stolen", "theft"],
            "fire": ["fire", "smoke"], "water": ["flood", "pipe", "leak"]}


@dataclass
class Extraction:
    record: dict
    confidence: dict[str, float]

    def first_token_logprobs(self) -> list[float]:
        """The least confident field, as the top of a five-way distribution, in the shape the
        small-lm router reads."""
        p = min(self.confidence.values())
        rest = (1 - p) / 4
        return [math.log(p)] + [math.log(rest)] * 4


def extract(text: str) -> Extraction:
    rec: dict = {}
    conf: dict[str, float] = {}
    m = re.search(r"Policy number:\s*(\S+)", text)
    rec["policy_number"], conf["policy_number"] = (m.group(1), 0.99) if m else (None, 0.2)
    m = re.search(r"Policyholder:\s*(.+)", text)
    rec["claimant"], conf["claimant"] = (m.group(1).strip(), 0.97) if m else (None, 0.2)
    m = re.search(r"On (\d{1,2}) (\w+) (\d{4})", text)
    if m and m.group(2) in MONTHS:
        rec["incident_date"] = f"{m.group(3)}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
        conf["incident_date"] = 0.95
    else:
        rec["incident_date"], conf["incident_date"] = None, 0.2
    low = text.lower()
    hits = [k for k, words in KEYWORDS.items() if any(w in low for w in words)]
    rec["incident_type"] = hits[0] if hits else "other"
    conf["incident_type"] = 0.9 if len(hits) == 1 else 0.7 if not hits else 0.45
    m = re.search(r"([\d,]+\.\d{2})\s*GBP", text)
    rec["amount_claimed"] = float(m.group(1).replace(",", "")) if m else None
    rec["currency"] = "GBP" if m else None
    conf["amount_claimed"] = conf["currency"] = 0.95 if m else 0.3
    rec["police_report"] = "police report" in low  # misses the negation: a typical small-model slip
    conf["police_report"] = 0.55 if re.search(r"\bno police report\b", low) else 0.9
    return Extraction(rec, conf)
