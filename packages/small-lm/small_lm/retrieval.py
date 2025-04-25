"""An on-device notes index in one SQLite file.

Vectors are stored as float32 blobs. When the ``sqlite-vec`` extension is available the query
runs inside SQLite (``vec_distance_cosine``); otherwise the same file is searched with numpy.
Nothing leaves the device to build or query it.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class HashEmbedder:
    """Hashed word unigrams and bigrams; no weights, a few kilobytes of code."""

    name = "hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"\w+", t.lower())
            for f in words + [a + "_" + b for a, b in zip(words, words[1:], strict=False)]:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=4).digest(), "little")
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class BgeSmall:
    name = "bge-small"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def embedder(name: str):
    return BgeSmall() if name == "bge-small" else HashEmbedder()


@dataclass
class Hit:
    id: int
    source: str
    text: str
    score: float


def chunks(text: str, words: int = 120, overlap: int = 20) -> list[str]:
    toks = text.split()
    step = words - overlap
    return [" ".join(toks[i:i + words]) for i in range(0, max(len(toks) - overlap, 1), step)]


class Index:
    def __init__(self, db: sqlite3.Connection, emb):
        self.db, self.emb = db, emb
        self.vec = _load_vec(db)

    @classmethod
    def build(cls, folder: str | Path, out: str | Path, embed: str = "hash") -> Index:
        emb = embedder(embed)
        db = sqlite3.connect(out)
        db.executescript("""
            DROP TABLE IF EXISTS notes; DROP TABLE IF EXISTS meta;
            CREATE TABLE notes (id INTEGER PRIMARY KEY, source TEXT, text TEXT, vector BLOB);
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        db.execute("INSERT INTO meta VALUES ('embedder', ?)", (embed,))
        rows = [(str(p.name), c) for p in sorted(Path(folder).rglob("*")) if p.suffix in {".txt", ".md"}
                for c in chunks(p.read_text(encoding="utf-8"))]
        vecs = emb.embed([c for _, c in rows]) if rows else np.zeros((0, 1), np.float32)
        db.executemany("INSERT INTO notes (source, text, vector) VALUES (?, ?, ?)",
                       [(s, c, v.tobytes()) for (s, c), v in zip(rows, vecs, strict=True)])
        db.commit()
        return cls(db, emb)

    @classmethod
    def open(cls, path: str | Path) -> Index:
        db = sqlite3.connect(path)
        name = db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()[0]
        return cls(db, embedder(name))

    def __len__(self) -> int:
        return self.db.execute("SELECT count(*) FROM notes").fetchone()[0]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self.emb.embed([query])[0]
        if self.vec:
            rows = self.db.execute(
                "SELECT id, source, text, 1 - vec_distance_cosine(vector, ?) AS s FROM notes ORDER BY s DESC LIMIT ?",
                (q.tobytes(), k)).fetchall()
            return [Hit(*r) for r in rows]
        rows = self.db.execute("SELECT id, source, text, vector FROM notes").fetchall()
        if not rows:
            return []
        m = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        scores = m @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(rows[i][0], rows[i][1], rows[i][2], float(scores[i])) for i in order]


def _load_vec(db: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return True
    except (ImportError, AttributeError, sqlite3.OperationalError):
        return False
