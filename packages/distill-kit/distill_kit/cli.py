"""distill-kit clean | teach | filter | dedup | predict | pairs | eval"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from distill_kit.data.clean import clean_folder
from distill_kit.data.dedup import dedup
from distill_kit.data.filter import filter_traces, read_jsonl, write_jsonl
from distill_kit.eval import evaluate
from distill_kit.pairs import build_pairs
from distill_kit.schema import parse
from distill_kit.teacher import load_teacher, render, teach


def predict(model_dir: str, rows: list[dict], schema: dict, max_tokens: int = 1024) -> list[dict]:
    """Greedy outputs of a trained student, with vLLM when installed, else transformers."""
    prompts = [render(r["document"], schema) for r in rows]
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model_dir)
        outs = [o.outputs[0].text for o in llm.chat([[{"role": "user", "content": p}] for p in prompts],
                                                   SamplingParams(temperature=0, max_tokens=max_tokens))]
    except ImportError:
        from transformers import pipeline

        gen = pipeline("text-generation", model=model_dir, torch_dtype="auto", device_map="auto")
        outs = [gen([{"role": "user", "content": p}], max_new_tokens=max_tokens, do_sample=False)[0]
                ["generated_text"][-1]["content"] for p in prompts]
    return [{"id": r["id"], "prediction": parse(o, schema)[0], "raw": o} for r, o in zip(rows, outs, strict=True)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="distill-kit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clean", help="redact personal data and apply the quality rules before teaching")
    c.add_argument("--docs", required=True)
    c.add_argument("--out", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--docs", required=True)
    t.add_argument("--schema", required=True)
    t.add_argument("--teacher", required=True, help="api:<model> or replay:<file>")
    t.add_argument("--samples", type=int, default=3)
    t.add_argument("--out", required=True)
    f = sub.add_parser("filter")
    f.add_argument("traces")
    f.add_argument("--schema", required=True)
    f.add_argument("--min-agree", type=int, default=2)
    f.add_argument("--out", required=True)
    d = sub.add_parser("dedup")
    d.add_argument("data")
    d.add_argument("--minhash", action="store_true", help="MinHash over word 5-grams (the only method)")
    d.add_argument("--threshold", type=float, default=0.8)
    d.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    pr = sub.add_parser("pairs")
    pr.add_argument("--student", required=True, help="run folder holding predictions.jsonl")
    pr.add_argument("--data", required=True)
    pr.add_argument("--schema", required=True)
    pr.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("run")
    e.add_argument("--task", required=True)
    e.add_argument("--general", default="mt-bench")
    args = ap.parse_args(argv)

    if args.cmd == "clean":
        print(json.dumps(clean_folder(args.docs, args.out).__dict__), file=sys.stderr)
    elif args.cmd == "teach":
        schema = json.loads(Path(args.schema).read_text())
        write_jsonl(args.out, teach(args.docs, schema, load_teacher(args.teacher), args.samples))
    elif args.cmd == "filter":
        schema = json.loads(Path(args.schema).read_text())
        kept, stats = filter_traces(read_jsonl(args.traces), schema, args.min_agree)
        write_jsonl(args.out, kept)
        print(json.dumps(stats.__dict__), file=sys.stderr)
    elif args.cmd == "dedup":
        rows = read_jsonl(args.data)
        keep = dedup([r["document"] for r in rows], threshold=args.threshold)
        write_jsonl(args.out, [rows[i] for i in keep])
        print(json.dumps({"in": len(rows), "kept": len(keep)}), file=sys.stderr)
    elif args.cmd == "predict":
        schema = json.loads(Path(args.schema).read_text())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out, predict(args.model, read_jsonl(args.data), schema))
    elif args.cmd == "pairs":
        schema = json.loads(Path(args.schema).read_text())
        rows = read_jsonl(args.data)
        student = {r["id"]: r["prediction"] for r in read_jsonl(Path(args.student) / "predictions.jsonl")}
        prompts = {r["id"]: render(r["document"], schema) for r in rows}
        pairs = build_pairs(rows, student, prompts)
        write_jsonl(args.out, pairs)
        print(json.dumps({"pairs": len(pairs), "of": len(rows)}), file=sys.stderr)
    else:
        print(json.dumps(evaluate(args.run, args.task, args.general), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
