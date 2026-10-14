"""GATE planning: teacher-conditional coverage picks the student size, and each prompt is routed to the teacher
signal, the verifier signal, the replay pool or dropped. Expected values are worked by hand from the paper's
equations (https://hanaml.com/research/papers/domain-distillation-rl, Section 2 and Section 5)."""

import math

import pytest

from distill_kit.plan import Gate, Student, plan

T, F = True, False

# Four prompts, one per regime, worked by hand with beta=8, gamma=1, tau=0.15:
#   a  p_T=0.75, p_S=0     teachable    lambda = sigmoid(8 * 0.75)                          = 0.997527
#   b  p_T=1,    p_S=0.75  explorable   lambda = sigmoid(8 * (0.25 - sqrt(0.75 * 0.25)))   = 0.187841
#   c  p_T=0,    p_S=0     unlearnable  lambda = 0 (teacher below tau)
#   d  p_T=1,    p_S=1     saturated    lambda = sigmoid(0)                                 = 0.5
TEACHER = {"a": [T, T, T, F], "b": [T, T, T, T], "c": [F, F, F, F], "d": [T, T, T, T]}
STUDENT = {"a": [F, F, F, F], "b": [T, T, T, F], "c": [F, F, F, F], "d": [T, T, T, T]}


def test_each_prompt_gets_the_regime_weight_and_signal_of_the_paper():
    p = plan(TEACHER, {"8b": Student(8.0, STUDENT)})
    routes = {r.id: r for r in p.routes}
    assert [routes[i].regime for i in "abcd"] == ["teachable", "explorable", "unlearnable", "saturated"]
    assert routes["a"].weight == pytest.approx(0.997527, abs=1e-6)
    assert routes["b"].weight == pytest.approx(0.187841, abs=1e-6)
    assert routes["c"].weight == 0.0
    assert routes["d"].weight == pytest.approx(0.5)
    assert [routes[i].signal for i in "abcd"] == ["teacher", "verifier", "drop", "replay"]
    assert p.shares == {"teacher": 0.25, "verifier": 0.25, "replay": 0.25, "drop": 0.25}


def test_coverage_is_the_share_of_teacher_solved_prompts_the_student_solves_at_least_once():
    # Teacher solves a, b, d; the student solves b and d at least once in its four samples: rho_4 = 2/3.
    p = plan(TEACHER, {"8b": Student(8.0, STUDENT)})
    assert p.k == {"8b": 4}
    assert p.coverage["8b"] == pytest.approx(2 / 3)


def test_the_smallest_student_that_clears_the_coverage_threshold_is_chosen():
    teacher = {str(i): [T] for i in range(20)}
    # 18/20 = 0.90, 19/20 = 0.95, 20/20 = 1.0 against the paper's threshold of 0.952
    students = {
        "qwen3-1.7b": Student(1.7, {str(i): [i >= 2] for i in range(20)}),
        "qwen3-4b": Student(4.0, {str(i): [i >= 1] for i in range(20)}),
        "qwen3-8b": Student(8.0, {str(i): [T] for i in range(20)}),
    }
    p = plan(teacher, students)
    assert p.coverage == {"qwen3-1.7b": pytest.approx(0.90), "qwen3-4b": pytest.approx(0.95), "qwen3-8b": 1.0}
    assert p.student == "qwen3-8b"
    assert plan(teacher, students, Gate(threshold=0.9)).student == "qwen3-1.7b"


def test_when_no_student_clears_the_threshold_none_is_chosen_and_the_largest_is_routed():
    teacher = {"x": [T], "y": [T]}
    students = {"small": Student(1.0, {"x": [F], "y": [F]}), "large": Student(3.0, {"x": [T], "y": [F]})}
    p = plan(teacher, students)
    assert p.student is None
    assert p.routed == "large"
    assert {r.id: r.signal for r in p.routes} == {"x": "replay", "y": "teacher"}


def test_coverage_uses_the_same_k_for_every_prompt():
    # One prompt has 4 samples, the other 2: k is 2, so the late success on prompt "x" does not count.
    teacher = {"x": [T], "y": [T]}
    p = plan(teacher, {"s": Student(1.0, {"x": [F, F, F, T], "y": [T, F]})})
    assert p.k == {"s": 2}
    assert p.coverage["s"] == 0.5


def test_inputs_that_cannot_be_planned_are_rejected():
    with pytest.raises(ValueError, match="no student"):
        plan(TEACHER, {})
    with pytest.raises(ValueError, match="missing"):
        plan(TEACHER, {"s": Student(1.0, {"a": [T]})})
    assert math.isnan(plan({"a": [F]}, {"s": Student(1.0, {"a": [F]})}).coverage["s"])


def test_the_plan_command_reads_outcomes_and_writes_one_route_per_prompt(tmp_path, capsys):
    import json

    from distill_kit.cli import main

    rows = [{"id": i, "model": "teacher", "correct": TEACHER[i]} for i in "abcd"]
    rows += [{"id": i, "model": "qwen3-8b", "params_b": 8.0, "correct": STUDENT[i]} for i in "abcd"]
    (tmp_path / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert main(["plan", str(tmp_path / "outcomes.jsonl"), "--out", str(tmp_path / "routes.jsonl")]) == 0
    routes = [json.loads(x) for x in (tmp_path / "routes.jsonl").read_text().splitlines()]
    assert [(r["id"], r["signal"]) for r in routes] == [("a", "teacher"), ("b", "verifier"), ("c", "drop"),
                                                        ("d", "replay")]
    summary = json.loads(capsys.readouterr().out)
    assert summary["student"] is None and summary["coverage"] == {"qwen3-8b": 0.6667}
