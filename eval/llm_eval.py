"""Optional evaluation with a real LLM: answer quality (LLM judge) and an LLM attacker.

Needs an API key, so it is not part of run_all.py.

    export ANTHROPIC_API_KEY=...  LLM_MODEL=<model id>
    python eval/llm_eval.py --backend anthropic --n 30

    export OPENAI_API_KEY=...     LLM_MODEL=<model id>
    python eval/llm_eval.py --backend openai --n 30

Writes eval/results/llm_quality.csv and eval/results/llm_attacker.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context.data import REGION, load_environment  # noqa: E402
from secure_context.detect import default_detector  # noqa: E402
from secure_context.llm import get_backend  # noqa: E402
from secure_context.pipeline import Pipeline  # noqa: E402
from secure_context.policy import TASKS  # noqa: E402
from secure_context.privacy import KEngine  # noqa: E402
from secure_context.tokens import Session  # noqa: E402
from secure_context.vault import SplitStorage  # noqa: E402

JUDGE = """You grade an assistant's answer for a financial advisor.
Reference answer (written with the full client file):
{ref}

Candidate answer (written with a reduced, privacy-protected file):
{cand}

Score the candidate from 1 to 5 for how well it serves the advisor compared with the reference:
5 = equally useful, 4 = minor gaps, 3 = noticeably less specific, 2 = misses key points, 1 = not useful.
Reply with only the number."""

ATTACK = """Here is a privacy-protected client context sent to an AI assistant:
{ctx}

Guess the client's hidden attributes. Reply with JSON only:
{{"occupation_category": one of {cats}, "region": one of ["Northeast","Midwest","South","West"],
  "age_band": like "50-59"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    llm = get_backend(args.backend, args.model)

    pop, clients = load_environment()
    clients = clients[:args.n]
    storage = SplitStorage(default_detector())
    ctx_ids = [storage.ingest(c) for c in clients]
    engine = KEngine(pop)

    quality = []
    for mode in ("full", "masked", "default", "adaptive"):
        pipe = Pipeline(storage, engine, llm)
        for c, ctx in zip(clients, ctx_ids):
            for task in TASKS:
                tr = pipe.run(c, ctx, task, mode=mode, session=Session())
                quality.append(dict(client=c.client_id, task=task, mode=mode, answer=tr.rehydrated,
                                    unresolved=len(tr.unresolved), k=tr.k_single, refused=tr.refused))
    q = pd.DataFrame(quality)
    ref = q[q["mode"] == "full"].set_index(["client", "task"])["answer"]
    scores = []
    for r in q.itertuples():
        if r.mode == "full":
            scores.append(5)
            continue
        out = llm.complete("You are a strict grader.", JUDGE.format(ref=ref[(r.client, r.task)], cand=r.answer))
        m = re.search(r"[1-5]", out)
        scores.append(int(m.group(0)) if m else None)
    q["judge_score"] = scores
    (ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    q.to_csv(ROOT / "eval" / "results" / "llm_quality.csv", index=False)
    print(q.groupby(["task", "mode"])[["judge_score", "unresolved"]].mean().round(2))

    cats = sorted(pop.occ_cat.unique())
    rows = []
    for c in clients:
        for task in TASKS:
            s = Session()
            from secure_context.tasks import build_prompt
            ctx = build_prompt(task, c, TASKS[task]["default"], s)
            out = llm.complete("Answer with JSON only.", ATTACK.format(ctx=ctx, cats=json.dumps(cats)))
            try:
                g = json.loads(out[out.index("{"):out.rindex("}") + 1])
            except ValueError:
                g = {}
            lo = (c.age // 10) * 10
            rows.append(dict(client=c.client_id, task=task,
                             occupation_hit=g.get("occupation_category") == c.occ_cat,
                             region_hit=g.get("region") == REGION[c.state],
                             age_band_hit=g.get("age_band") == f"{lo}-{lo + 9}"))
    a = pd.DataFrame(rows)
    a.to_csv(ROOT / "eval" / "results" / "llm_attacker.csv", index=False)
    print(a.groupby("task")[["occupation_hit", "region_hit", "age_band_hit"]].mean().round(2))


if __name__ == "__main__":
    main()
