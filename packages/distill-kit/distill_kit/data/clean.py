"""Clean documents before any of them reaches the teacher.

Personal data the task does not need is redacted: email addresses, phone numbers, payment card numbers (Luhn
checked), IBANs (mod-97 checked) and UK National Insurance numbers, each replaced by a tag such as ``[EMAIL]``.
Documents are then held to quality rules in the style of Gopher (Rae et al., 2021,
https://arxiv.org/abs/2112.11446), with a lower word minimum because claim forms are short. A rejected document
is not sent anywhere.

    out = clean(text)                      # Cleaned(text, redacted={"email": 1}, rejected=None)
    stats = clean_folder("raw/", "clean/")  # what distill-kit teach should read
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Rules:
    min_words: int = 20
    max_words: int = 100_000
    max_repeated_lines: float = 0.3  # share of non-empty lines that repeat an earlier line
    max_symbols_per_word: float = 0.1  # "#" and "..." per word
    min_alpha_words: float = 0.8  # share of words with at least one letter
    mean_word_length: tuple[float, float] = (3.0, 10.0)


DEFAULT_RULES = Rules()


@dataclass(frozen=True)
class Cleaned:
    text: str
    redacted: dict[str, int]  # kind -> spans replaced
    rejected: str | None  # the rule the document failed, if any


@dataclass
class CleanStats:
    documents: int = 0
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    redacted: dict[str, int] = field(default_factory=dict)


# One alternation, scanned left to right, so a span claimed by one kind is never re-read as another.
_PII = re.compile(
    r"(?P<email>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?P<iban>\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b)"
    r"|(?P<card>(?<!\w)(?:\d[ -]?){12,18}\d(?!\w))"
    r"|(?P<ni_number>\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b)"
    r"|(?P<phone>(?<![\w+])\+?\d[\d ().-]{8,}\d(?!\w))"
)


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban(text: str) -> bool:
    s = text.replace(" ", "")
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def _kind(m: re.Match) -> str | None:
    kind, span = m.lastgroup, m.group()
    digits = re.sub(r"\D", "", span)
    if kind == "iban":
        return "iban" if _iban(span) else None
    if kind == "card":
        if _luhn(digits):
            return "card"
        kind = "phone"  # a long number that is not a card may still be a phone number
    if kind == "phone":
        return "phone" if 10 <= len(digits) <= 15 else None
    return kind


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()

    def replace(m: re.Match) -> str:
        kind = _kind(m)
        if kind is None:
            return m.group()
        counts[kind] += 1
        return f"[{kind.upper()}]"

    return _PII.sub(replace, text), dict(counts)


def quality(text: str, rules: Rules = DEFAULT_RULES) -> str | None:
    """The first rule the text fails, or None."""
    words = text.split()
    if len(words) < rules.min_words:
        return "too_short"
    if len(words) > rules.max_words:
        return "too_long"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and 1 - len(set(lines)) / len(lines) > rules.max_repeated_lines:
        return "repeated_lines"
    if (text.count("#") + text.count("...") + text.count("…")) / len(words) > rules.max_symbols_per_word:
        return "symbols"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < rules.min_alpha_words:
        return "non_alpha"
    low, high = rules.mean_word_length
    if not low <= sum(len(w) for w in words) / len(words) <= high:
        return "word_length"
    return None


def clean(text: str, rules: Rules = DEFAULT_RULES) -> Cleaned:
    redacted_text, counts = redact(text)
    return Cleaned(redacted_text, counts, quality(redacted_text, rules))


def clean_folder(src: str | Path, dst: str | Path, rules: Rules = DEFAULT_RULES) -> CleanStats:
    """Clean every .txt and .md document in ``src`` into ``dst`` under the same name; rejected ones are skipped."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    stats = CleanStats()
    for path in sorted(Path(src).glob("*")):
        if path.suffix not in {".txt", ".md"}:
            continue
        stats.documents += 1
        out = clean(path.read_text(encoding="utf-8"), rules)
        if out.rejected:
            stats.rejected[out.rejected] = stats.rejected.get(out.rejected, 0) + 1
            continue
        for kind, n in out.redacted.items():
            stats.redacted[kind] = stats.redacted.get(kind, 0) + n
        (dst / path.name).write_text(out.text, encoding="utf-8")
        stats.kept += 1
    return stats
