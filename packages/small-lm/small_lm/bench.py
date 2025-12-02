"""Measure what matters on a phone: sustained decode speed after warm-up, and energy per
completed task, read from a USB power meter's CSV log (time_s, volts, amps)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sustained_tokens_per_s(token_times: list[float], warmup_s: float = 30.0) -> float:
    """Decode speed over tokens emitted after the first ``warmup_s`` seconds of the run, when
    the phone has heated up and its clocks have settled."""
    t = np.asarray(token_times, float)
    if len(t) < 2:
        raise ValueError("need at least two token timestamps")
    late = t[t >= t[0] + warmup_s]
    if len(late) < 2:
        raise ValueError("run is shorter than the warm-up; record for longer")
    return float((len(late) - 1) / (late[-1] - late[0]))


@dataclass
class PowerLog:
    t: np.ndarray
    watts: np.ndarray

    @classmethod
    def read(cls, path: str | Path) -> PowerLog:
        with Path(path).open() as f:
            rows = [(float(r["time_s"]), float(r["volts"]) * float(r["amps"])) for r in csv.DictReader(f)]
        t, w = zip(*sorted(rows), strict=True)
        return cls(np.array(t), np.array(w))

    def joules(self, start: float, end: float) -> float:
        """Energy between two times, by the trapezoid rule over the samples in between."""
        mask = (self.t >= start) & (self.t <= end)
        t, w = self.t[mask], self.watts[mask]
        if len(t) < 2:
            raise ValueError("fewer than two power samples in the window")
        return float(np.sum((w[1:] + w[:-1]) / 2 * np.diff(t)))


def energy_per_task(log: PowerLog, tasks: list[tuple[float, float, bool]], idle_watts: float = 0.0) -> float:
    """Joules per *completed* task, above idle. ``tasks`` are (start, end, completed); energy
    spent on failed tasks still counts, divided over the completed ones."""
    total = sum(log.joules(s, e) - idle_watts * (e - s) for s, e, _ in tasks)
    done = sum(1 for *_, ok in tasks if ok)
    if done == 0:
        raise ValueError("no completed tasks")
    return total / done
