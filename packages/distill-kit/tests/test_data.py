import json
from pathlib import Path

import pytest

from distill_kit.data.dedup import bands_for, dedup
from distill_kit.data.filter import agree, filter_traces
from distill_kit.eval import f1
from distill_kit.pairs import build_pairs
from distill_kit.schema import errors, parse
from distill_kit.teacher import render
from distill_kit.train import load_recipe

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schemas/claim.json").read_text())
GOOD = {"policy_number": "MTR-48812", "claimant": "Aisha Raman", "incident_date": "2026-03-14",
        "incident_type": "collision", "amount_claimed": 2350.0, "currency": "GBP", "police_report": True}


def test_schema_and_repair():
    assert errors(GOOD, SCHEMA) == []
    assert errors({**GOOD, "incident_type": "flood"}, SCHEMA)
    assert errors({**GOOD, "amount_claimed": True}, SCHEMA)  # a bool is not a number
    broken = "Here you go:\n```json\n" + json.dumps(GOOD)[:-1] + ",}\n```"  # fenced, trailing comma
    value, errs = parse(broken, SCHEMA)
    assert errs == [] and value["claimant"] == "Aisha Raman"
    value, errs = parse('{"policy_number": "A"}', SCHEMA)
    assert value is None and any("missing claimant" in e for e in errs)


def test_agreement_keeps_only_fields_two_samples_share():
    a, b, c = dict(GOOD), {**GOOD, "amount_claimed": 2530.0}, {**GOOD, "currency": "EUR"}
    record, dropped = agree([a, b, c], 2)
    assert record["amount_claimed"] == 2350.0 and record["currency"] == "GBP"
    record, dropped = agree([a, b], 2)
    assert "amount_claimed" not in record and dropped == 1


def test_filter_drops_unrepairable_and_disagreeing():
    rows = [
        {"id": "d1", "document": "x", "samples": [json.dumps(GOOD), json.dumps(GOOD), "not json at all"]},
        {"id": "d2", "document": "y", "samples": [json.dumps(GOOD), "{}", "nope"]},
    ]
    kept, stats = filter_traces(rows, SCHEMA, 2)
    assert [r["id"] for r in kept] == ["d1"]
    assert stats.invalid_samples == 3


def test_minhash_dedup():
    base = ("The claimant reports that on 14 March 2026 her car was struck from behind at the junction of "
            "Mill Lane and Station Road while waiting at a red light, and asks for repair costs of 2350 pounds.")
    near = base.replace("2350", "2,350")
    other = "A kitchen fire on 2 February damaged the cooker and the ceiling; the fire service attended."
    assert dedup([base, near, other], threshold=0.8) == [0, 2]
    b, r = bands_for(0.8, 128)
    assert b * r == 128 and abs((1 / b) ** (1 / r) - 0.8) < 0.1


def test_pairs_and_f1():
    train = [{"id": "d1", "target": GOOD}, {"id": "d2", "target": GOOD}]
    student = {"d1": GOOD, "d2": {**GOOD, "amount_claimed": 235.0}}
    pairs = build_pairs(train, student, {"d1": "p1", "d2": "p2"})
    assert [p["id"] for p in pairs] == ["d2"]
    assert json.loads(pairs[0]["chosen"])["amount_claimed"] == 2350.0
    assert f1([GOOD], [GOOD]) == 1.0
    assert f1([{**GOOD, "claimant": "A. Raman"}], [GOOD]) == pytest.approx(6 / 7)


def test_template_and_recipes():
    prompt = render("Policy MTR-48812 ...", SCHEMA)
    assert "policy_number, claimant" in prompt and "Policy MTR-48812" in prompt
    assert load_recipe(ROOT / "recipes/sft-qwen3-8b.yaml")["stage"] == "sft"
    assert load_recipe(ROOT / "recipes/dpo-qwen3-8b.yaml")["trainer"]["beta"] == 0.1


def test_lora_recipe_for_a_small_base_and_its_checks(tmp_path):
    recipe = load_recipe(ROOT / "recipes/sft-llama3.2-3b-lora.yaml")
    assert recipe["model"] == "meta-llama/Llama-3.2-3B-Instruct" and "deepspeed" not in recipe
    assert recipe["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage: sft\nmodel: m\ndata: d\noutput_dir: o\nlora: {r: 8, rank_dropout: 0.1}\n")
    with pytest.raises(ValueError, match="unknown lora settings: rank_dropout"):
        load_recipe(bad)
    both = tmp_path / "both.yaml"
    both.write_text("stage: sft\nmodel: m\ndata: d\noutput_dir: o\ndeepspeed: ds.json\nlora: {r: 8}\n")
    with pytest.raises(ValueError, match="LoRA or ZeRO-3"):
        load_recipe(both)
