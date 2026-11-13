#!/usr/bin/env python3
"""
Check model="auto" routing against labelled prompts, with the real d1 model.

Each line of the eval file is {"prompt": ..., "expected": preset}. The labels
follow this policy (change the labels, or the descriptions in
routing.DEFAULT_CHOICES, if yours differs):

  quick     facts and lookups, finding or listing things, running commands and
            reporting their output, trivial snippets, mechanical edits
  standard  explaining code or concepts, writing or changing non-trivial code,
            debugging, comparisons, multi-step investigation
  thorough  open-ended design, subtle concurrency / correctness / security
            reasoning, proofs, large migrations

Reports, per method: accuracy, how often a prompt went to a pricier tier than
labelled (wasted cost) or a cheaper one (risk on quality), and a confusion
matrix. Uses $D1_URL or starts llama-server like the router does.

    python scripts/routing_eval.py                       # both methods
    python scripts/routing_eval.py --method choice -v    # with every prompt
    python scripts/routing_eval.py --json results.json
"""

import argparse
import json
import sys
from pathlib import Path

from omniharness import D1Router

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "evals" / "routing.jsonl"
ORDER = ["quick", "standard", "thorough"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(router: D1Router, items: list[dict], verbose: bool) -> dict:
    rows = []
    for item in items:
        d = router.route_sync(item["prompt"])
        if d.source != "d1":
            sys.exit(f"d1 unavailable: {d.reason}")
        rows.append({**item, "got": d.preset, "label": d.label,
                     "p": (d.probabilities or {}).get(d.preset) if d.method == "choice"
                     else d.complexity, "ms": d.elapsed_ms})
        if verbose:
            r = rows[-1]
            mark = "ok " if r["got"] == r["expected"] else "XX "
            print(f"  {mark}{r['expected']:9s}-> {r['got']:9s} {r['label']:13s} "
                  f"{r['p']:.2f}  {r['prompt'][:70]}")
    rank = {p: i for i, p in enumerate(ORDER)}
    n = len(rows)
    correct = sum(r["got"] == r["expected"] for r in rows)
    over = sum(rank[r["got"]] > rank[r["expected"]] for r in rows)
    under = sum(rank[r["got"]] < rank[r["expected"]] for r in rows)
    confusion = {e: {g: 0 for g in ORDER} for e in ORDER}
    for r in rows:
        confusion[r["expected"]][r["got"]] += 1
    return {"n": n, "correct": correct, "over": over, "under": under,
            "confusion": confusion, "rows": rows,
            "mean_ms": sum(r["ms"] for r in rows) / n}


def report(method: str, res: dict) -> None:
    n = res["n"]
    print(f"{method}: {res['correct']}/{n} correct ({res['correct'] / n:.0%}), "
          f"{res['over']} too expensive, {res['under']} too cheap, "
          f"{res['mean_ms']:.0f} ms/prompt")
    corner = "expected / got"
    print(f"  {corner:16s}" + "".join(f"{g:>10s}" for g in ORDER))
    for e in ORDER:
        print(f"  {e:16s}" + "".join(f"{res['confusion'][e][g]:10d}" for g in ORDER))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--method", choices=["choice", "complexity", "both"], default="both")
    parser.add_argument("--url", help="Running llama-server with d1-2B (env D1_URL)")
    parser.add_argument("--json", type=Path, help="Write every answer to this file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every prompt")
    args = parser.parse_args()

    items = load(args.file)
    methods = ["choice", "complexity"] if args.method == "both" else [args.method]
    # One server for every method: the first router starts it, the others reuse its URL.
    first = D1Router(args.url, method=methods[0])
    results = {}
    try:
        for method in methods:
            router = first if method == methods[0] else D1Router(first.url, method=method)
            if args.verbose:
                print(f"{method}:")
            results[method] = evaluate(router, items, args.verbose)
            if args.verbose:
                print()
    finally:
        first.close()

    for method, res in results.items():
        report(method, res)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
