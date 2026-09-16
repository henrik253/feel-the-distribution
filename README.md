# Feel the distribution

Send one fixed input through a model fifty times and look at what comes back.
Then run a ten-item harness across ticket categories and find the one that fails.

Everything tonight's lecture said follows from one fact: the component returns a
*sample from a distribution*, not an answer. This lab is where you see it.

## What you need

- Python 3.12 or newer, and [`uv`](https://docs.astral.sh/uv/) (or plain `pip`).
- An Anthropic API key, one per student: https://console.anthropic.com/settings/keys
  The API is pay-as-you-go; a few dollars of credit covers this lab many times over
  (the default model costs about ten cents for both parts).
- Nothing on AWS. The Learner Lab setup and the usage alarms are a checklist to complete
  before Week 2; see *Configuring AWS* in the course guide.

## Setup

Install `uv` if you don't have it (one line, then reopen your terminal):

```bash
# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```bash
git clone https://github.com/aise-stthomas/feel-the-distribution
cd feel-the-distribution
cp .env.example .env        # then paste your key into .env
uv sync                     # or: python3 -m venv .venv && . .venv/bin/activate && pip install -e .
uv run distribution.py --n 3
```

If the last command prints three decisions, you are set. The key lives in `.env`, which
is ignored by git. It never goes in code, a fixture, or a commit.

**No key yet?** `--provider fake` runs the same scripts against a stand-in that samples
from a made-up distribution. It checks the plumbing. It tells you nothing about a model,
and the homework questions are about the model.

## Part 1: the distribution

```bash
uv run distribution.py
```

One ticket, fifty calls at temperature 0, fifty at the provider's default. About two
minutes; the entry tier allows fifty calls a minute, and the script pauses on rate
limits and continues. Every call is appended to `runs/distribution.jsonl`
as it lands, and the summary and plot (`runs/distribution.png`) come at the end.

Fifty is a small sample for a distribution. It is enough to see the shape tonight;
`--n 100` is the real thing if you have the time, and Week 5 is where sample size
becomes the subject.

**What to watch.** Each sample scrolls past as it lands: the decision, the amount, and
the one-sentence reason. The ticket asks for a $38 lamp plus $14.99 shipping, which is
$52.99 against a $50 cap. Watch what the model does at the edge of that rule. Watch
whether the sentence changes when the decision does not. Every ten calls a one-line
tally shows the shape so far.

Answer, one sentence each:

1. **How many distinct decisions at temperature 0? How many distinct sentences?** Then
   read homework question 8 before you decide what those numbers mean.
2. **What is the modal answer, and is it right?** Read the ticket in `tickets.py` and the
   policy in `triage.py`, and decide for yourself before you look at the rationale. If
   you see a refund of exactly $50, ask where that number came from.
3. **Where did temperature change the decision, and where did it only change the words?**

While it runs, read `triage.py` top to bottom. It is about eighty lines and it is the
entire mechanism: render, sample, parse. Notice where the policy lives, what the model
is allowed to see, and what the code does when the model returns something that is
not a decision.

## Part 2: find the slice

```bash
uv run slices.py
```

Ten tickets, five categories, three runs each, about a minute. The script prints
the aggregate pass rate first, then the failure rate per category. One category fails
at least 30% of the time.

Write down **the category** and **one hypothesis for why**, then open `tickets.py`,
read the tickets in that category, and check your hypothesis against the raw model
output in `runs/slices.jsonl`. Is it the whole category, or one ticket in it? Read the
rationale the model gave on the failures. It will often state the rule it is breaking.

If setup ate the time, do this part before the homework. It is thirty calls.

## Keep

- `runs/distribution.png` (the plot)
- your answer to "how many distinct outputs at temperature 0"
- the failing category and your hypothesis

The homework asks about all three.

## If something breaks

| Symptom | What it is |
|---|---|
| `Authentication failed` / 401 | `.env` is not in this directory or the key was pasted with a trailing space |
| `rate limited; sleeping 5s` repeatedly | Normal on the entry tier. The run continues. Check your limits at https://console.anthropic.com/settings/limits |
| `malformed` appears in the action counts | Not a bug. The model returned something that was not a decision. It is counted, because it is a sample too. |
| `… refused: … credit balance …` | Your account is out of credit. Add a few dollars at https://console.anthropic.com/settings/billing and rerun. |
| `… refused: … temperature …` | You set `ANTHROPIC_MODEL` to a model that rejects the temperature parameter (Opus 4.7 and later, Sonnet 5, Fable). Use the default model, or run with `--temperatures default`. |
| `… keeps returning 5xx` / `overloaded` | The model is overloaded on the provider's side. Set `ANTHROPIC_MODEL=claude-sonnet-4-6` in `.env`, or wait a few minutes. |
| Model not found | The default model is pinned in `triage.py`. Set `ANTHROPIC_MODEL` in `.env` to a current model. |

## What this is one instance of

The SDK, the model name, and the price per token are September 2026 details and will change.
What does not change: a learned component is a function from input to a *distribution*
over outputs; temperature is a systems parameter that reshapes that distribution; the
aggregate hides which slice is failing; and the code around the model, not the model,
decides what happens to an output that does not fit the contract. 
