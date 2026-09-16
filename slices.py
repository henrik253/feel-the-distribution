"""Part 2: find the slice.

Ten tickets in five categories, each run several times. Aggregate accuracy across all
of them is one number; the per-category failure rate is what tells you where the
system is broken. Find the category that fails at least 30% of the time and write
down one hypothesis for why.

    uv run slices.py                  # live: 10 tickets x 3 repeats = 30 calls (~1 min)
    uv run slices.py --repeats 5      # more confidence, if you have the time
    uv run slices.py --provider fake  # plumbing check, NOT a model
    uv run slices.py --replay runs/slices.jsonl

"Fails" means: the action is not the one policy requires, OR the model issued a refund
above the no-approval cap, OR the output was malformed. A refund the model is not
allowed to issue is a failure even when the model's rationale sounds reasonable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tickets import CLEAN_ACCOUNT, HARNESS, REFUND_CAP_NO_APPROVAL
from triage import DEFAULT_MODEL, triage

RUNS = Path("runs")


def failed(item: dict, rec: dict) -> str | None:
    """Return why this record fails, or None if it passes."""
    if rec["action"] == "malformed":
        return "malformed output"
    if rec["action"] != item["expected"]:
        return f"expected {item['expected']}, got {rec['action']}"
    if rec["action"] == "refund" and (rec["refund_amount"] or 0) > REFUND_CAP_NO_APPROVAL:
        return f"refund {rec['refund_amount']} exceeds cap {REFUND_CAP_NO_APPROVAL}"
    return None


def collect(repeats: int, model: str, provider: str, out: Path) -> list[dict]:
    RUNS.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    records = []
    total = len(HARNESS) * repeats
    print(f"{total} calls to {model} via {provider}", flush=True)
    for run in range(1, repeats + 1):
        for item in HARNESS:
            rec = triage(item["ticket"], CLEAN_ACCOUNT, temperature=None,
                         model=model, provider=provider)
            rec.update({"id": item["id"], "category": item["category"],
                        "expected": item["expected"], "run": run,
                        "failure": failed(item, rec)})
            records.append(rec)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            mark = "ok  " if rec["failure"] is None else "FAIL"
            print(f"  run {run} {item['id']} {item['category']:22s} {mark} {rec['action']:9s}"
                  f" {'' if rec['refund_amount'] is None else rec['refund_amount']}", flush=True)
    return records


def report(records: list[dict]) -> None:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    n_fail = sum(r["failure"] is not None for r in records)
    print(f"\n=== aggregate: {len(records) - n_fail}/{len(records)} passed "
          f"({100 * (len(records) - n_fail) / len(records):.0f}%) ===")
    print("That one number is what a ten-item harness with no slices reports. Now by category:\n")
    rows = []
    for cat, recs in by_cat.items():
        fails = [r for r in recs if r["failure"] is not None]
        rate = len(fails) / len(recs)
        reasons = defaultdict(int)
        for r in fails:
            reasons[r["failure"]] += 1
        top = "; ".join(f"{k} x{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:2])
        rows.append((rate, cat, len(fails), len(recs), top))
    rows.sort(reverse=True)
    print(f"{'category':24s} {'fail rate':>9s}   what went wrong")
    for rate, cat, nf, nt, top in rows:
        flag = "  <-- at or above 30%" if rate >= 0.30 else ""
        print(f"{cat:24s} {100*rate:6.0f}% ({nf}/{nt})   {top}{flag}")
    md = RUNS / "slices.md"
    with md.open("w") as f:
        f.write("| category | fail rate | what went wrong |\n|---|---|---|\n")
        for rate, cat, nf, nt, top in rows:
            f.write(f"| {cat} | {100*rate:.0f}% ({nf}/{nt}) | {top} |\n")
    print(f"\ntable: {md}")
    print("\nWrite down: the category that fails at least 30% of the time, and ONE hypothesis for why.")
    print("Then look at the tickets in that category in tickets.py and check your hypothesis against the raw output.")
    print("Is it the whole category that fails, or one ticket in it? What is different about that ticket?")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=3, help="runs over the ten items (default 3)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", choices=["anthropic", "fake"], default="anthropic")
    p.add_argument("--replay", type=Path, help="re-report a saved .jsonl")
    args = p.parse_args()

    out = RUNS / "slices.jsonl"
    if args.replay:
        records = [json.loads(line) for line in args.replay.read_text().splitlines() if line.strip()]
    else:
        if args.provider == "fake":
            print("provider=fake: this is NOT a model. Plumbing check only.", file=sys.stderr)
        records = collect(args.repeats, args.model, args.provider, out)
    report(records)


if __name__ == "__main__":
    main()
