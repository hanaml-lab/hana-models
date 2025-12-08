import math

import numpy as np
import pytest

from small_lm import Router
from small_lm.bench import PowerLog, energy_per_task, sustained_tokens_per_s
from small_lm.calibrate import fit, selective_accuracy
from small_lm.local import Generation, features
from small_lm.quantize import block_quantize, relative_error
from small_lm.retrieval import Index


def test_block_quantize_error_falls_with_bits():
    w = np.random.default_rng(0).standard_normal((256, 64)).astype(np.float32)
    errs = [relative_error(w, b) for b in (4, 5, 8)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.01
    assert block_quantize(np.zeros(7), 4).tolist() == [0.0] * 7


def synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    top_p = rng.uniform(0.2, 1.0, n)
    entropy = (1 - top_p) * 2 + rng.normal(0, 0.1, n)
    length = rng.uniform(1, 4, n)
    p_right = 1 / (1 + np.exp(-(6 * (top_p - 0.6) - 0.3 * (length - 2.5))))
    correct = (rng.random(n) < p_right).astype(float)
    return np.c_[top_p, entropy, length], correct


def test_calibrated_router_beats_length_heuristic():
    X, y = synthetic()
    model = fit(X, y)
    Xt, yt = synthetic(seed=1)
    conf = model.confidence(Xt)
    ours = selective_accuracy(conf, yt, 0.8)
    by_length = selective_accuracy(-Xt[:, 2], yt, 0.8)
    assert ours > by_length + 0.03
    # calibrated: in the top-confidence decile, accuracy is close to the mean confidence
    top = np.argsort(-conf)[: len(conf) // 10]
    assert abs(conf[top].mean() - yt[top].mean()) < 0.08
    assert 0.0 < model.threshold(0.8) < model.threshold(0.5)


class FakeLocal:
    def __init__(self, p):
        self.p = p

    def generate(self, prompt, max_tokens=256):
        rest = [math.log((1 - self.p) / 4)] * 4  # the other four of the top five share what is left
        return Generation("Your insurance payment is due on 1 November.", [math.log(self.p), *rest])


def test_router_routes_by_confidence(tmp_path):
    X, y = synthetic()
    path = tmp_path / "router.json"
    fit(X, y).save(path)
    router = Router.load(str(path), coverage=0.8, on_low_confidence="abstain")
    assert router.ask(FakeLocal(0.97), "When is the payment due?").route == "local"
    assert router.ask(FakeLocal(0.30), "When is the payment due?").route == "abstained"
    with pytest.raises(ValueError):
        Router.load(str(path), on_low_confidence="escalate")


def test_features():
    f = features("a b c", Generation("x", [math.log(0.9), math.log(0.1)]))
    assert f[0] == pytest.approx(0.9) and f[1] > 0


def test_index_finds_the_note(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/insurance.md").write_text(
        "Car insurance renews on 1 November; the payment is taken by direct debit.")
    (tmp_path / "notes/garden.md").write_text("Prune the roses in late February before new growth starts.")
    idx = Index.build(tmp_path / "notes", tmp_path / "notes.sqlite")
    hits = Index.open(tmp_path / "notes.sqlite").search("when is the car insurance payment", k=1)
    assert len(idx) == 2 and hits[0].source == "insurance.md"


def test_bench(tmp_path):
    times = [i * 0.1 for i in range(600)]  # 10 tokens/s for 60 s
    assert sustained_tokens_per_s(times, warmup_s=30) == pytest.approx(10.0)
    log = tmp_path / "power.csv"
    log.write_text("time_s,volts,amps\n" + "\n".join(f"{t},5.0,0.4" for t in range(0, 21)))
    p = PowerLog.read(log)
    assert p.joules(0, 10) == pytest.approx(20.0)
    assert energy_per_task(p, [(0, 10, True), (10, 20, False)]) == pytest.approx(40.0)
