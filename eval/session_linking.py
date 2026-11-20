"""Can the same client be recognized across sessions from the pattern alone?

The model itself keeps no memory between API calls, but the provider's logs (or an agent
with a memory feature) see every prompt. Tokens change every session, so the question is
whether the rest of the context links two sessions anyway.

For each target client we build a "last quarter" context and a "this quarter" context:
new session tokens, holdings weights moved by up to 3 points, assets moved by up to 8%,
age up a year for half the clients, and the goal three months closer. The attacker sees
the old context and a lineup of current contexts (the same client plus random others
from the book) and picks the match. Notes are dropped in every context, since real notes
differ between meetings.

Two attackers:
    model   an LLM asked to pick the match (or NONE)
    script  a deterministic linker: the candidate with the most identical lines,
            then the closest numbers (what anyone holding the logs could run)

    GOOGLE_CLOUD_PROJECT=<project> python3 eval/session_linking.py --n 50 --lineup 10

It also runs the script linker against whole books (every client in the book is a
candidate, which is what someone holding a firm's logs faces), and repeats that for the
meeting brief with one field removed at a time.

Writes eval/results_real/session_linking.csv, session_linking_summary.csv,
session_linking_full_book.csv and session_linking_fields.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context import real_data  # noqa: E402
from secure_context.llm import get_backend  # noqa: E402
from secure_context.policy import ALL_EXACT, TASKS, render_context  # noqa: E402
from secure_context.tokens import Session  # noqa: E402

CONTEXTS = {
    "masked, all exact": dict(ALL_EXACT, notes="drop"),
    "concentration, exact tickers": dict(TASKS["concentration"]["default"], holdings="exact"),
    "concentration policy (anonymized)": dict(TASKS["concentration"]["default"]),
    "concentration, sectors": dict(TASKS["concentration"]["default"], holdings="sectors"),
    "meeting brief policy": dict(TASKS["meeting_prep"]["default"], notes="drop"),
}

PROMPT = """You review logs of an AI assistant used by financial advisors. Client names are replaced
by tokens that change every session, so tokens cannot be compared.

Context from a session last quarter:
{old}

Contexts from sessions this quarter:
{lineup}

Which one, if any, is the same client as last quarter? Holdings, assets and goals can drift
a little between quarters. Reply with JSON only, no other text:
{{"match": the number of the matching context or "NONE", "confidence": 0 to 100}}"""


def drift(c, rng):
    """The same client one quarter later."""
    hs = [h for h in c.holdings if h.ticker != "CASH"]
    hs = [replace(h, weight=max(1, h.weight + rng.randint(-3, 3))) if h.weight > 0 else h for h in hs]
    cash = max(0, 100 - sum(h.weight for h in hs))
    hs.append(next(replace(h, weight=cash) for h in c.holdings if h.ticker == "CASH"))
    m = re.search(r"within (\d+) months", c.liquidity or "")
    liq = c.liquidity.replace(m.group(0), f"within {max(1, int(m.group(1)) - 3)} months") if m else c.liquidity
    return replace(c, holdings=hs, aum=c.aum * rng.uniform(0.92, 1.08), liquidity=liq,
                   age=c.age + (1 if rng.random() < 0.5 else 0))


def card(c, levels):
    s = Session()
    lines = [f"Client ref: {s.token('PERSON', c.name)}"]
    lines += [f"{k}: {v}" for k, v in render_context(c, levels).items()]
    return "\n".join(lines)


def numbers(text):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", re.sub(r"<[^>]+>", "", text))]


def script_link(old, lineup):
    """Most identical lines, then the smallest total difference between the numbers."""
    o_lines = set(old.splitlines()[1:])
    o_nums = numbers(old)

    def score(cand):
        same = len(o_lines & set(cand.splitlines()[1:]))
        n = numbers(cand)
        dist = sum(abs(a - b) for a, b in zip(o_nums, n)) + 100 * abs(len(o_nums) - len(n))
        return (-same, dist)
    return min(range(len(lineup)), key=lambda i: score(lineup[i]))


def full_book(book_sizes, n_targets, seed):
    """Script linker with every client in the book as a candidate."""
    rows, fields = [], []
    for size in book_sizes:
        _, book = real_data.load_real_environment(n_clients=size)
        rng = random.Random(seed)
        later = {c.client_id: drift(c, rng) for c in book}

        def rate(levels):
            cands = [card(later[c.client_id], levels) for c in book]
            hits = sum(script_link(card(t, levels), cands) == i for i, t in enumerate(book[:n_targets]))
            return hits / n_targets
        for name, levels in CONTEXTS.items():
            rows.append(dict(book_size=size, context=name, script_correct=rate(levels), chance=1 / size))
        if size == max(book_sizes):
            base = CONTEXTS["meeting brief policy"]
            fields.append(dict(removed="(none)", script_correct=rate(base)))
            for f, lv in base.items():
                if lv != "drop":
                    fields.append(dict(removed=f, script_correct=rate(dict(base, **{f: "drop"}))))
    return pd.DataFrame(rows), pd.DataFrame(fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=50, help="target clients")
    ap.add_argument("--lineup", type=int, default=10, help="candidates per question, including the match")
    ap.add_argument("--book", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-books", default="200,1000", help="book sizes for the script-only whole-book test")
    args = ap.parse_args()

    llm = get_backend(args.backend, args.model)
    rng = random.Random(args.seed)
    _, book = real_data.load_real_environment(n_clients=args.book)
    later = {c.client_id: drift(c, rng) for c in book}

    jobs = []
    for t in book[:args.n]:
        others = rng.sample([c for c in book if c.client_id != t.client_id], args.lineup - 1)
        pos = rng.randrange(args.lineup)
        cands = others[:pos] + [t] + others[pos:]
        for name, levels in CONTEXTS.items():
            old = card(t, levels)
            lineup = [card(later[c.client_id], levels) for c in cands]
            jobs.append((t.client_id, name, old, lineup, pos))
    print(f"attacker: {llm.name}   targets: {args.n}   lineup: {args.lineup}   calls: {len(jobs)}")

    def run(job):
        cid, name, old, lineup, pos = job
        text = "\n\n".join(f"[{i + 1}]\n{x}" for i, x in enumerate(lineup))
        reply = llm.complete("Answer with JSON only.", PROMPT.format(old=old, lineup=text))
        try:
            g = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
            m = str(g.get("match", "")).strip().upper()
            pick = int(m) - 1 if m.isdigit() else None
            conf = float(g.get("confidence", 0) or 0)
            parsed = True
        except (ValueError, TypeError):
            pick, conf, parsed = None, 0.0, False
        return dict(client=cid, context=name, answer=pos, model_pick=pick, model_conf=conf, parsed=parsed,
                    model_correct=pick == pos, model_confident_correct=pick == pos and conf >= 50,
                    model_said_none=parsed and pick is None, script_correct=script_link(old, lineup) == pos,
                    identical_lines=len(set(old.splitlines()[1:]) & set(lineup[pos].splitlines()[1:])),
                    lines=len(old.splitlines()) - 1)

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, jobs))
    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results_real"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "session_linking.csv", index=False)
    summ = df.groupby("context", sort=False).agg(
        n=("client", "size"), model_correct=("model_correct", "mean"),
        model_confident_correct=("model_confident_correct", "mean"), model_said_none=("model_said_none", "mean"),
        script_correct=("script_correct", "mean"), unparsed=("parsed", lambda s: 1 - s.mean())).reset_index()
    summ["chance"] = 1 / args.lineup
    summ.to_csv(out / "session_linking_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))

    sizes = [int(x) for x in args.full_books.split(",") if x]
    if sizes:
        fb, fields = full_book(sizes, min(200, min(sizes)), args.seed)
        fb.to_csv(out / "session_linking_full_book.csv", index=False)
        fields.to_csv(out / "session_linking_fields.csv", index=False)
        print("\nScript linker, every client in the book is a candidate:")
        print(fb.round(3).to_string(index=False))
        print("\nMeeting brief, one field removed:")
        print(fields.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
