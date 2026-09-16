"""Part 1: feel the distribution.

One fixed ticket. Fifty calls at temperature 0, fifty at the default.
Plot the action the model chose and the refund amount it proposed.

    uv run distribution.py                 # live, 50 calls per temperature (~2 min)
    uv run distribution.py --n 100         # the real thing, if you have the time
    uv run distribution.py --n 20          # a quick look
    uv run distribution.py --provider fake # plumbing check, no key, NOT a model
    uv run distribution.py --replay runs/distribution.jsonl   # re-plot a saved run

Every sample scrolls past as it lands: the decision, the amount, and the sentence the
model gave for it. Watch the sentence. Every call is also appended to
runs/distribution.jsonl, so a run you stop at call 33 has still given you 33 samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tickets import FIXED_ACCOUNT, FIXED_TICKET
from triage import DEFAULT_MODEL, triage

RUNS = Path("runs")
INK, BLUE, RED, GREY, PAPER = "#22303A", "#2F6DB5", "#C9463D", "#8C99A3", "#FBFBF9"
ACTION_ORDER = ["answer", "refund", "hold", "escalate", "malformed"]


def collect(n: int, temperatures: list, model: str, provider: str, out: Path) -> list[dict]:
    """Make the calls. Appends one JSON line per call so partial runs are kept."""
    RUNS.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    records = []
    for temperature in temperatures:
        label = "default" if temperature is None else f"T={temperature}"
        print(f"\n{'=' * 78}\n{label}: {n} calls to {model} via {provider}. Same ticket every time.\n{'=' * 78}", flush=True)
        actions = Counter()
        for i in range(1, n + 1):
            rec = triage(FIXED_TICKET, FIXED_ACCOUNT, temperature=temperature,
                         model=model, provider=provider)
            records.append(rec)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            actions[rec["action"]] += 1
            print(f"  #{i:<3d} {rec['action']:9s} {fmt_amount(rec['refund_amount']):>5s}   "
                  f"{(rec['rationale'] or rec['raw'].replace(chr(10), ' '))[:70]}", flush=True)
            if i % 10 == 0 or i == n:
                print("       " + tally(actions), flush=True)
    return records


def fmt_amount(a) -> str:
    return "" if a is None else f"${a:g}"


def tally(actions: Counter) -> str:
    """A one-line bar chart of the actions so far."""
    return "  ".join(f"{a} {'#' * actions[a]} {actions[a]}" for a in ACTION_ORDER if actions[a])


def summarize(records: list[dict]) -> None:
    """The questions the lab asks, answered from the data."""
    by_temp: dict[str, list[dict]] = {}
    for r in records:
        by_temp.setdefault(str(r["temperature"]), []).append(r)
    for temp, recs in by_temp.items():
        outputs = Counter((r["action"], r["refund_amount"]) for r in recs)
        actions = Counter(r["action"] for r in recs)
        modal, modal_n = outputs.most_common(1)[0]
        print(f"\n=== temperature {temp}: {len(recs)} samples ===")
        print(f"distinct (action, amount) outputs: {len(outputs)}")
        print(f"modal output: {modal[0]} {modal[1]}  ({modal_n}/{len(recs)} = {100*modal_n/len(recs):.0f}%)")
        print("actions: " + ", ".join(f"{a}={actions[a]}" for a in ACTION_ORDER if actions[a]))
        amounts = sorted({r["refund_amount"] for r in recs if r["refund_amount"] is not None})
        if amounts:
            print(f"refund amounts proposed: {amounts}")
        rationales = Counter(r["rationale"] for r in recs if r["rationale"])
        print(f"distinct rationale sentences: {len(rationales)} of {len(recs)}")
        lat = sorted(r["latency_ms"] for r in recs)
        print(f"latency ms: p50={lat[len(lat)//2]} max={lat[-1]}")
    print("\nAnswer, one sentence each:")
    print("  1. How many distinct decisions at temperature 0? How many distinct sentences?")
    print("  2. What is the modal answer, and is it right? (Read the ticket. Decide for yourself.)")
    print("  3. Where did temperature change the decision, and where did it only change the words?")


def plot(records: list[dict], out: Path) -> None:
    """One row per temperature: action counts on the left, refund amounts on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temps = list(dict.fromkeys(str(r["temperature"]) for r in records))
    fig, axes = plt.subplots(len(temps), 2, figsize=(10, 3.4 * len(temps)), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    for row, temp in enumerate(temps):
        recs = [r for r in records if str(r["temperature"]) == temp]
        counts = Counter(r["action"] for r in recs)
        ax = axes[row][0]
        ax.bar(ACTION_ORDER, [counts[a] for a in ACTION_ORDER],
               color=[BLUE, BLUE, BLUE, BLUE, RED])
        ax.set_title(f"temperature {temp}: action chosen (n={len(recs)})", color=INK)
        ax.set_ylabel("calls")
        amounts = [r["refund_amount"] for r in recs if r["refund_amount"] is not None]
        ax = axes[row][1]
        if amounts:
            ax.hist(amounts, bins=20, color=BLUE, edgecolor=PAPER)
        ax.set_title(f"temperature {temp}: refund amount proposed (n={len(amounts)})", color=INK)
        ax.set_xlabel("dollars")
        for a in axes[row]:
            a.set_facecolor(PAPER)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(colors=INK)
    fig.suptitle("One ticket, many calls: the output is a distribution", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nplot: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50, help="calls per temperature (default 50)")
    p.add_argument("--temperatures", nargs="+", default=["default", "0"],
                   help="temperatures to run, in order; 'default' means the provider's default "
                        "(1 for Claude). The range is 0 to 1.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", choices=["anthropic", "fake"], default="anthropic")
    p.add_argument("--replay", type=Path, help="re-summarize and re-plot a saved .jsonl")
    args = p.parse_args()

    out = RUNS / "distribution.jsonl"
    if args.replay:
        records = [json.loads(line) for line in args.replay.read_text().splitlines() if line.strip()]
    else:
        temps = [None if t == "default" else float(t) for t in args.temperatures]
        if args.provider == "fake":
            print("provider=fake: this is NOT a model. Plumbing check only.", file=sys.stderr)
        records = collect(args.n, temps, args.model, args.provider, out)
    summarize(records)
    plot(records, RUNS / "distribution.png")


if __name__ == "__main__":
    main()
