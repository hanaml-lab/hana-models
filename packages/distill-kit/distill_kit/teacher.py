"""Collect teacher outputs: each document is rendered into the extraction template and sent to
the teacher ``samples`` times. ``--teacher api:<model>`` calls an OpenAI-compatible endpoint
(``OPENAI_BASE_URL``); ``--teacher replay:<file>`` reads outputs recorded earlier, which is how
a pipeline is rerun without paying for the teacher twice."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_HERE = Path(__file__).resolve().parent
TEMPLATES = _HERE / "templates" if (_HERE / "templates").is_dir() else _HERE.parent / "templates"


def render(document: str, schema: dict, template: str = "extract.jinja", templates: Path = TEMPLATES) -> str:
    env = Environment(loader=FileSystemLoader(templates), undefined=StrictUndefined, keep_trailing_newline=True)
    return env.get_template(template).render(document=document, schema=json.dumps(schema, indent=2),
                                             fields=list(schema.get("properties", {})))


Teacher = Callable[[str, int], list[str]]  # (prompt, n) -> n completions


def api_teacher(model: str, temperature: float = 0.7, endpoint: str | None = None) -> Teacher:
    base = (endpoint or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")

    def call(prompt: str, n: int) -> list[str]:
        body = {"model": model, "n": n, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return [c["message"]["content"] for c in json.load(resp)["choices"]]

    return call


def replay_teacher(path: str | Path) -> Teacher:
    recorded: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            recorded[row["prompt_sha"]] = row["samples"]

    def call(prompt: str, n: int) -> list[str]:
        try:
            return recorded[_sha(prompt)][:n]
        except KeyError:
            raise KeyError("no recorded teacher output for this prompt; was the template changed?") from None

    return call


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def load_teacher(spec: str) -> Teacher:
    kind, _, arg = spec.partition(":")
    if kind == "api":
        return api_teacher(os.environ.get("TEACHER_MODEL", arg) if arg == "frontier" else arg)
    if kind == "replay":
        return replay_teacher(arg)
    raise ValueError("teacher is api:<model> or replay:<file>")


def documents(folder: str | Path) -> Iterable[tuple[str, str]]:
    for p in sorted(Path(folder).glob("*")):
        if p.suffix in {".txt", ".md"}:
            yield p.stem, p.read_text(encoding="utf-8")


def teach(folder: str | Path, schema: dict, teacher: Teacher, samples: int = 3) -> list[dict]:
    rows = []
    for doc_id, text in documents(folder):
        prompt = render(text, schema)
        rows.append({"id": doc_id, "document": text, "prompt_sha": _sha(prompt), "samples": teacher(prompt, samples)})
    return rows
