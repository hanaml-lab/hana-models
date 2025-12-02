"""small-lm quantize | index | calibrate"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from small_lm import calibrate as cal
from small_lm.quantize import export
from small_lm.retrieval import Index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="small-lm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("model")
    q.add_argument("--type", default="Q4_K_M")
    q.add_argument("--out", required=True)
    q.add_argument("--check", help="task folder: report the task score of the FP16 and quantised models")
    i = sub.add_parser("index")
    i.add_argument("folder")
    i.add_argument("--embed", default="hash", choices=["hash", "bge-small"])
    i.add_argument("--out", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--labels", required=True, help='JSONL rows {"features": [...], "correct": 0|1}')
    c.add_argument("--model", help="the GGUF the labels were collected with, recorded beside the router")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "quantize":
        path = export(args.model, args.type, args.out)
        print(path)
        if args.check:
            from small_lm.accuracy import compare

            print(json.dumps(compare(args.model, path, args.check), indent=2))
    elif args.cmd == "index":
        idx = Index.build(args.folder, args.out, args.embed)
        print(f"{len(idx)} chunks -> {args.out}")
    else:
        rows = [json.loads(x) for x in Path(args.labels).read_text().splitlines() if x.strip()]
        X = np.array([r["features"] for r in rows], float)
        y = np.array([r["correct"] for r in rows], float)
        model = cal.fit(X, y)
        model.save(args.out)
        conf = model.confidence(X)
        print(json.dumps({"labels": len(rows), "model": args.model, "temperature": round(model.temperature, 3),
                          "selective_accuracy@0.8": round(cal.selective_accuracy(conf, y, 0.8), 3),
                          "accuracy@1.0": round(float(y.mean()), 3)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
