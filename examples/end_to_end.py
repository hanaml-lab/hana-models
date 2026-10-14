"""The model factory end to end, offline.

    make build-rust && uv run python examples/end_to_end.py

1. distil   distill-kit cleans the documents (personal data redacted, quality rules), turns
            recorded teacher samples into a filtered, deduplicated training set, and the student's
            own mistakes into DPO pairs;
2. plan     GATE's coverage rule and per-prompt routing (distill_kit.plan) on the same documents;
3. route    small-lm calibrates a router on where the student is right, and picks the share of
            questions it can keep on the device;
4. bench    serve-bench replays the sample RAG trace against two simulated serving profiles and
            reports latency, throughput and cost side by side.

The teacher samples in examples/data are recorded fixtures, the student is a rule-based stand-in
(hana_models.student) and the servers are simulations (hana_models.simserver), so every number
printed here describes this example, not a model.
"""

import json
import tempfile
from pathlib import Path

from hana_models.pipeline import bench, distil, route
from serve_bench.report import row, table

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/data"


def main() -> None:
    schema = json.loads((ROOT / "packages/distill-kit/schemas/claim.json").read_text())
    d = distil(DATA / "claims", DATA / "heldout", schema, DATA / "teacher-samples.jsonl")
    c = d.cleaned
    print(f"clean:  {c.kept} of {c.documents} documents pass the quality rules; "
          f"{sum(c.redacted.values())} personal-data spans redacted")
    print(f"distil: {d.documents} documents -> {d.kept_after_filter} after filtering "
          f"({d.invalid_teacher_samples} teacher samples failed the schema) -> {d.kept_after_dedup} after dedup")
    print(f"        {len(d.pairs)} DPO pairs from the student's mistakes; student held-out F1 {d.student_f1_heldout}")

    p = d.plan
    shares = ", ".join(f"{v:.0%} {k}" for k, v in p.shares.items())
    verdict = f"chosen: {p.student}" if p.student else "no candidate clears it"
    print(f"plan:   coverage of the stand-in student rho_{p.k['stand-in']} = {p.coverage['stand-in']:.2f} "
          f"(GATE needs >= 0.952; {verdict})")
    print(f"        routes: {shares}")

    docs = [(r["document"], r["target"]) for r in d.train]
    r = route(docs * 4)
    print(f"route:  student fully right on {r.accuracy_all:.0%} of documents; at {r.coverage:.0%} coverage "
          f"the router keeps the ones it is right on {r.selective_accuracy:.0%} of the time")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        bench(ROOT, out, ROOT / "packages/serve-bench/traces/rag-sample.jsonl.zst", speed=20, duration="40s")
        rows = [row(out / name, gpus=8, gpu_hour=2.49, slo=800) for name in ("bf16", "fp8-eagle-prefix")]
        print("bench (simulated servers):")
        print(table(rows))


if __name__ == "__main__":
    main()
