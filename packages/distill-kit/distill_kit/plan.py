"""Plan a run with GATE before committing compute.

GATE (Agarwal, 2026; https://hanaml.com/research/papers/domain-distillation-rl) mixes on-policy distillation and
verifiable-reward RL per prompt. Two of its parts need only verifier outcomes, so the factory runs them before any
training:

- **Student size.** The teacher-conditional coverage rho_k is the share of teacher-solved prompts that the untrained
  student solves at least once in k samples. The paper's fit puts the gap to the teacher within one point when
  rho_64 >= 0.952; the plan picks the smallest candidate student that clears the threshold.
- **Routing.** Each prompt gets the paper's weight
  lambda(x) = 1{p_T >= tau} * sigmoid(beta * ([p_T - p_S]_+ - gamma * sqrt(p_S (1 - p_S))))
  and a signal: ``teacher`` (lambda >= 0.5, distil from the teacher), ``verifier`` (train on the verifier reward),
  ``replay`` (already solved; keep in the small replay pool) or ``drop`` (neither model solves it).

The reference trainer is github.com/hanaml-lab/domain-distillation-rl; this module only plans.

    p = plan(teacher={"q1": [True, False, ...]}, students={"qwen3-4b": Student(4.0, {"q1": [False, ...]})})
    p.student, p.coverage, p.shares, p.routes
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

Outcomes = Mapping[str, Sequence[bool]]  # prompt id -> verifier outcome of each sample


@dataclass(frozen=True)
class Gate:
    """The paper's settings (Table 6): sigmoid temperature, weight of the reward's standard deviation, teacher
    competence threshold, and the coverage a student needs."""

    beta: float = 8.0
    gamma: float = 1.0
    tau: float = 0.15
    threshold: float = 0.952


PAPER = Gate()


@dataclass(frozen=True)
class Student:
    params_b: float  # size in billions of parameters; the plan prefers the smallest that clears the threshold
    outcomes: Outcomes


@dataclass(frozen=True)
class Route:
    id: str
    p_teacher: float
    p_student: float
    regime: str  # teachable, explorable, saturated or unlearnable
    weight: float  # lambda(x): the share of the update that comes from the teacher
    signal: str  # teacher, verifier, replay or drop


@dataclass(frozen=True)
class Plan:
    coverage: dict[str, float]  # rho_k per candidate student
    k: dict[str, int]  # the k each coverage was measured at
    student: str | None  # the smallest student that clears the threshold, if any
    routed: str  # the student the routes are for: the chosen one, else the largest
    routes: list[Route] = field(default_factory=list)
    shares: dict[str, float] = field(default_factory=dict)


def plan(teacher: Outcomes, students: Mapping[str, Student], gate: Gate = PAPER) -> Plan:
    if not students:
        raise ValueError("no student to plan for")
    for name, s in students.items():
        missing = set(teacher) - set(s.outcomes)
        if missing:
            raise ValueError(f"{name}: outcomes missing for {len(missing)} prompts, e.g. {sorted(missing)[0]}")
    k = {name: min(len(v) for v in s.outcomes.values()) for name, s in students.items()}
    coverage = {name: _coverage(teacher, s.outcomes, k[name]) for name, s in students.items()}
    by_size = sorted(students, key=lambda n: students[n].params_b)
    chosen = next((n for n in by_size if coverage[n] >= gate.threshold), None)
    routed = chosen or by_size[-1]
    routes = [_route(i, _rate(teacher[i]), _rate(students[routed].outcomes[i]), gate) for i in teacher]
    counts = Counter(r.signal for r in routes)
    shares = {s: counts[s] / len(routes) for s in ("teacher", "verifier", "replay", "drop")}
    return Plan(coverage, k, chosen, routed, routes, shares)


def _rate(outcomes: Sequence[bool]) -> float:
    return sum(bool(o) for o in outcomes) / len(outcomes)


def _coverage(teacher: Outcomes, student: Outcomes, k: int) -> float:
    solved = [i for i, v in teacher.items() if any(v)]
    if not solved:
        return math.nan
    return sum(any(student[i][:k]) for i in solved) / len(solved)


def _route(prompt: str, p_t: float, p_s: float, gate: Gate) -> Route:
    if p_t < gate.tau:
        weight = 0.0
    else:
        z = gate.beta * (max(p_t - p_s, 0.0) - gate.gamma * math.sqrt(p_s * (1.0 - p_s)))
        weight = 1.0 / (1.0 + math.exp(-z))
    if p_s >= 1.0:
        regime, signal = "saturated", "replay"
    elif p_s <= 0.0 and p_t < gate.tau:
        regime, signal = "unlearnable", "drop"
    else:
        regime = "teachable" if p_s <= 0.0 else "explorable"
        signal = "teacher" if weight >= 0.5 else "verifier"
    return Route(prompt, p_t, p_s, regime, weight, signal)
