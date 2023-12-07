"""Near-duplicate removal with MinHash over word 5-gram shingles and LSH banding.

Documents whose estimated Jaccard similarity reaches the threshold are clustered and only the
first of each cluster is kept, so a held-out set built afterwards stays held out.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_PRIME = (1 << 31) - 1  # a, x < 2^31 keeps a*x + b inside uint64


def shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        words = words + [""] * (n - len(words))
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest(), "little")
            for i in range(len(words) - n + 1)}


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _PRIME, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _PRIME, num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, items: set[int]) -> np.ndarray:
        x = np.fromiter((i % _PRIME for i in items), dtype=np.uint64, count=len(items))
        return ((self.a[:, None] * x[None, :] + self.b[:, None]) % _PRIME).min(axis=1)


def jaccard_estimate(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def bands_for(threshold: float, num_perm: int) -> tuple[int, int]:
    """The (bands, rows) split whose S-curve midpoint (1/b)^(1/r) sits a little below the
    threshold, so pairs just above it are almost always candidates; the exact signature
    comparison afterwards removes the false positives."""
    threshold = 0.9 * threshold
    best = (1, num_perm)
    best_err = float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b:
            continue
        r = num_perm // b
        err = abs((1 / b) ** (1 / r) - threshold)
        if err < best_err:
            best, best_err = (b, r), err
    return best


def dedup(texts: list[str], threshold: float = 0.8, num_perm: int = 128, n: int = 5) -> list[int]:
    """Indices of the documents to keep."""
    hasher = MinHasher(num_perm)
    sigs = [hasher.signature(shingles(t, n)) for t in texts]
    b, r = bands_for(threshold, num_perm)
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for band in range(b):
        buckets: dict[bytes, list[int]] = {}
        for i, sig in enumerate(sigs):
            buckets.setdefault(sig[band * r:(band + 1) * r].tobytes(), []).append(i)
        for members in buckets.values():
            for j in members[1:]:
                if jaccard_estimate(sigs[members[0]], sigs[j]) >= threshold:
                    ri, rj = find(members[0]), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    return [i for i in range(len(texts)) if find(i) == i]
