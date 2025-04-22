"""Fit the router: a logistic model on the router features, calibrated so that its confidence
is the probability the local answer is right, and a threshold that gives the wanted coverage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 200) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb / len(y) + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One temperature on the logit, chosen by grid search on log loss."""
    best_t, best_loss = 1.0, float("inf")
    for t in np.exp(np.linspace(np.log(0.25), np.log(4.0), 81)):
        p = np.clip(1 / (1 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def selective_accuracy(conf: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """Accuracy on the most confident ``coverage`` share of questions."""
    n = max(1, int(round(coverage * len(conf))))
    top = np.argsort(-conf, kind="stable")[:n]
    return float(y[top].mean())


@dataclass
class RouterModel:
    weights: list[float]
    temperature: float
    mean: list[float]
    std: list[float]
    thresholds: dict[str, float]  # coverage -> confidence threshold, as strings for JSON

    def confidence(self, feats) -> np.ndarray:
        X = (np.atleast_2d(np.asarray(feats, float)) - self.mean) / self.std
        z = np.hstack([X, np.ones((len(X), 1))]) @ np.asarray(self.weights)
        return 1 / (1 + np.exp(-z / self.temperature))

    def threshold(self, coverage: float) -> float:
        key = f"{coverage:.2f}"
        if key in self.thresholds:
            return self.thresholds[key]
        known = sorted((float(k), v) for k, v in self.thresholds.items())
        return float(np.interp(coverage, [k for k, _ in known], [v for _, v in known]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RouterModel:
        return cls(**json.loads(Path(path).read_text()))


def fit(features: np.ndarray, correct: np.ndarray, seed: int = 0) -> RouterModel:
    """Fit on half the labels and calibrate the temperature and thresholds on the other half."""
    X, y = np.asarray(features, float), np.asarray(correct, float)
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, cal = idx[: len(y) // 2], idx[len(y) // 2:]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-9
    w = fit_logistic((X[tr] - mean) / std, y[tr])
    z = np.hstack([(X[cal] - mean) / std, np.ones((len(cal), 1))]) @ w
    t = fit_temperature(z, y[cal])
    conf = 1 / (1 + np.exp(-z / t))
    thresholds = {}
    for cov in np.round(np.arange(0.1, 1.0001, 0.05), 2):
        thresholds[f"{cov:.2f}"] = float(np.quantile(conf, 1 - cov)) if cov < 1 else 0.0
    return RouterModel(w.tolist(), t, mean.tolist(), std.tolist(), thresholds)
