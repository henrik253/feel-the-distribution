"""The whole mechanism.

    render  -> the ticket and account become tokens in a prompt
    sample  -> the model returns *one sample* from a distribution over outputs
    parse   -> your code turns the text back into a decision, or refuses to
"""
from __future__ import annotations

import json
import os
import random
import re
import time

from dotenv import load_dotenv

from tickets import REFUND_CAP_NO_APPROVAL, REFUND_CAP_WITH_APPROVAL

load_dotenv()  # reads ANTHROPIC_API_KEY (and optional ANTHROPIC_MODEL) from .env

# Haiku 4.5 is the cheapest current Claude model and one of the few that still accepts a
# temperature (see call_model). Opus 5, Sonnet 5 and the Fable models reject the parameter.
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
ACTIONS = ("answer", "refund", "hold", "escalate")

# --- 1. render ----------------------------------------------------------------
# Everything the model knows about the job, the policy, and the customer is in this
# string. There is no other channel. Read it as the interface, because it is.

POLICY = f"""You are the triage step for a customer-support system. Read the ticket and the
account summary, then decide ONE action:

  answer   - resolve with information only; no money moves
  refund   - issue a refund yourself; allowed only up to ${REFUND_CAP_NO_APPROVAL}
  hold     - a refund above ${REFUND_CAP_NO_APPROVAL} and up to ${REFUND_CAP_WITH_APPROVAL}; park it for human approval
  escalate - anything above ${REFUND_CAP_WITH_APPROVAL}, anything unclear, or anything you are not sure about

Respond with a JSON object and nothing else:
  {{"action": "answer|refund|hold|escalate", "refund_amount": <number or null>, "rationale": "<one sentence>"}}
"""


def render(ticket: str, account: dict) -> str:
    """Turn the ticket and the account into the text the model will read."""
    return (
        f"{POLICY}\n"
        f"ACCOUNT SUMMARY:\n{json.dumps(account, indent=2)}\n\n"
        f"TICKET:\n{ticket}\n"
    )


# --- 2. sample ----------------------------------------------------------------

def call_model(prompt: str, temperature: float | None, model: str) -> str:
    """Send the prompt once and return the raw text the model produced.

    temperature=None means "use the provider's default" (1.0 for Claude; the range is 0
    to 1). Temperature is a knob on the softmax over the next token: 0 sharpens the
    distribution toward the most likely token, it does not turn the model into a
    function. You will see that in the data.

    The Anthropic SDK removed temperature from its typed signature because the newest
    models (Opus 4.7 and later, Sonnet 5, Fable) refuse it with a 400. The Claude 4.5/4.6
    line (Haiku 4.5, Sonnet 4.6, Opus 4.6) still honours it, so we pass it through
    extra_body, which merges it into the request JSON as-is.

    Retries on rate limits (HTTP 429) with a short sleep, because the entry tier *is* a
    per-minute rate limit: a 429 means "the minute is not over yet", not "back off for
    a long time". The run should finish rather than crash at call 37.
    """
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("\nANTHROPIC_API_KEY is not set. Copy .env.example to .env in this "
                         "directory and paste your key, or run with --provider fake.")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    extra = {} if temperature is None else {"temperature": temperature}
    switch = ("Set ANTHROPIC_MODEL in .env to another model (for example claude-sonnet-4-6) "
              "and rerun; a saved run replays with --replay.")
    delay = 5
    server_errors = 0
    for attempt in range(30):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra,
            )
            # A request for JSON, not a guarantee: the prompt asks for it, nothing enforces
            # it. (The API can enforce a schema via output_config; the lab leaves it off on
            # purpose so you see what an unconstrained sample looks like.)
            return "".join(b.text for b in resp.content if b.type == "text")
        except anthropic.AuthenticationError as e:
            raise SystemExit(f"\nAuthentication failed: {getattr(e, 'message', e)}\n"
                             "Is ANTHROPIC_API_KEY set in .env in this directory?")
        except anthropic.BadRequestError as e:
            msg = getattr(e, "message", str(e))
            if any(w in msg.lower() for w in ("credit", "billing", "balance")):
                # Out of credit, not a per-minute limit: waiting will not help.
                raise SystemExit(f"\n{model} refused: {msg[:200]}\n"
                                 "Add credit at https://console.anthropic.com/settings/billing")
            if "temperature" in msg.lower():
                raise SystemExit(f"\n{model} refused: {msg[:200]}\n"
                                 "This model does not take a temperature. Use the default model, "
                                 "or run with --temperatures default.")
            raise
        except anthropic.RateLimitError:
            print(f"    rate limited; sleeping {delay}s", flush=True)
            time.sleep(delay)
        except anthropic.InternalServerError as e:
            server_errors += 1
            if server_errors == 1:
                print(f"    server error {e.status_code}: {getattr(e, 'message', str(e))[:120]}", flush=True)
            if server_errors >= 5:
                raise SystemExit(f"\n{model} keeps returning {e.status_code}; that is the provider, not you. {switch}")
            print(f"    retrying in {delay}s", flush=True)
            time.sleep(delay)
        except anthropic.APIConnectionError as e:
            server_errors += 1
            if server_errors >= 5:
                raise SystemExit(f"\nCannot reach the API: {e}. Check your network connection.")
            print(f"    connection error; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise SystemExit(f"\nGave up after repeated rate limits on {model}. {switch}")


def fake_model(prompt: str, temperature: float | None, model: str) -> str:
    """NOT a model. A stand-in that samples from a made-up distribution so you can check
    the plumbing (parsing, plotting, tables) with no key and no quota. Any conclusion
    you draw from fake output is a conclusion about this function, not about a model.
    """
    t = 1.0 if temperature is None else temperature
    amounts = [38, 38, 50, 50, 50, 50, 52.99]  # the fixed ticket's spread
    if "SYSTEM OVERRIDE" in prompt or "pre-approved" in prompt:
        weights = {"answer": 0, "refund": 1, "hold": 0, "escalate": 9}; amounts = [480]
    elif "$88" in prompt:
        weights = {"answer": 0, "refund": 5, "hold": 5, "escalate": 0}; amounts = [88]
    elif "$34" in prompt:
        weights = {"answer": 0, "refund": 9, "hold": 1, "escalate": 0}; amounts = [34]
    elif any(w in prompt.lower() for w in ("password", "log in", "locked out", "where is", "tracking")):
        weights = {"answer": 19, "refund": 0, "hold": 0, "escalate": 1}
    else:  # the ambiguous fixed ticket
        weights = {"answer": 0, "refund": 8, "hold": 1, "escalate": 1}
    if t == 0:  # sharpen: mostly the mode, a little leakage
        mode = max(weights, key=weights.get)
        weights = {k: (20 if k == mode else 1) for k in weights}
        amounts = [amounts[0]] * 20 + amounts[-1:]
    action = random.choices(list(weights), weights=list(weights.values()))[0]
    amount = None
    if action in ("refund", "hold"):
        amount = random.choice(amounts)
    if random.random() < 0.03:
        return "Sure! Here is my decision: refund the lamp."  # malformed on purpose
    return json.dumps({"action": action, "refund_amount": amount, "rationale": "fake"})


PROVIDERS = {"anthropic": call_model, "fake": fake_model}


# --- 3. parse -----------------------------------------------------------------

def parse(raw: str) -> dict:
    """Turn the model's text into a decision, or label it malformed.

    The model can emit anything: prose, a fenced block, JSON with a typo, an action
    not in the list. None of that is an exception. It is a sample from the same
    distribution as the good answers, and it gets counted, not crashed on.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # tolerate a code fence
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"action": "malformed", "refund_amount": None, "rationale": None}
    if not isinstance(obj, dict) or obj.get("action") not in ACTIONS:
        return {"action": "malformed", "refund_amount": None, "rationale": None}
    amount = obj.get("refund_amount")
    try:
        amount = None if amount is None else round(float(amount), 2)
    except (TypeError, ValueError):
        amount = None
    return {"action": obj["action"], "refund_amount": amount, "rationale": obj.get("rationale")}


def triage(ticket: str, account: dict, *, temperature: float | None,
           model: str = DEFAULT_MODEL, provider: str = "anthropic") -> dict:
    """render -> sample -> parse. Returns one record you can write to a file."""
    prompt = render(ticket, account)
    t0 = time.perf_counter()
    raw = PROVIDERS[provider](prompt, temperature, model)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    rec = parse(raw)
    rec.update({
        "model": model if provider != "fake" else f"FAKE({model})",
        "temperature": "default" if temperature is None else temperature,
        "latency_ms": latency_ms,
        "raw": raw,
    })
    return rec
