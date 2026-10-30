"""Real-model evaluation: answer quality (LLM judge) and an LLM attacker, on the public-data client book.

    export GROQ_API_KEY=...
    python3 eval/llm_eval.py --backend groq --n 8

Other backends: anthropic (ANTHROPIC_API_KEY + LLM_MODEL), openai (OPENAI_API_KEY + LLM_MODEL),
hf-api (HF_TOKEN), hf-local (downloads a model).

For each client and core task it asks the model four times, with full context, masked context,
the task policy and the adaptive policy, then has the model grade each answer against the
full-context one (1 to 5). Separately, the model sees the task-policy context and tries to
guess the client's hidden occupation group, region and age band.

Writes eval/results_real/llm_quality.csv, llm_quality_summary.csv, llm_attacker.csv.
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

from secure_context import real_data  # noqa: E402
from secure_context.data import REGION  # noqa: E402
from secure_context.detect import CRMDictionaryDetector, EnsembleDetector, RegexDetector  # noqa: E402
from secure_context.llm import get_backend  # noqa: E402
from secure_context.pipeline import Pipeline  # noqa: E402
from secure_context.policy import TASKS  # noqa: E402
from secure_context.privacy import KEngine  # noqa: E402
from secure_context.tasks import CORE_TASKS, build_prompt  # noqa: E402
from secure_context.tokens import Session  # noqa: E402
from secure_context.vault import SplitStorage  # noqa: E402

JUDGE = """You grade an AI assistant's answer for a financial advisor.

Reference answer, written with the full client file:
{ref}

Candidate answer, written with a reduced, privacy-protected file:
{cand}

Score the candidate from 1 to 5 for how well it serves the advisor compared with the reference:
5 equally useful, 4 minor gaps, 3 noticeably less specific, 2 misses key points, 1 not useful.
Reply with only the number."""

ATTACK = """This privacy-protected client context was sent to an AI assistant:
{ctx}

Guess the client's hidden attributes. Reply with JSON only, no other text:
{{"occupation_group": one of {cats}, "region": one of ["Northeast", "Midwest", "South", "West"],
  "age_band": a decade such as "50-59"}}"""


def evaluation_session(identity):
    """Stable sessions only for this public-data benchmark, to resume exact prompts."""
    import hashlib
    import os
    directory = ROOT / "data/cache/eval_sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / hashlib.sha256(identity.encode()).hexdigest()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError(f"Invalid evaluation session checkpoint: {path}")
    return Session(key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=15, help="clients to evaluate")
    args = ap.parse_args()
    llm = get_backend(args.backend, args.model)
    out = ROOT / "eval" / "results_real"
    out.mkdir(parents=True, exist_ok=True)
    print(f"model: {llm.name}")

    directory, clients = real_data.load_real_environment(n_clients=max(args.n, 50))
    clients = clients[:args.n]
    storage = SplitStorage(EnsembleDetector([RegexDetector(), CRMDictionaryDetector()]))
    ctx_ids = [storage.ingest(c) for c in clients]
    engine = KEngine(directory)

    rows = []
    for mode in ("full", "masked", "default", "adaptive"):
        pipe = Pipeline(storage, engine, llm)
        for i, (c, ctx) in enumerate(zip(clients, ctx_ids)):
            for task in CORE_TASKS:
                tr = pipe.run(c, ctx, task, mode=mode, session=evaluation_session(f"{mode}:{c.client_id}:{task}"))
                rows.append(dict(client=c.client_id, task=task, mode=mode, k=tr.k_single, refused=tr.refused,
                                 unresolved_tokens=len(tr.unresolved), name_leaked_to_model=c.name in tr.prompt,
                                 answer=tr.rehydrated))
            print(f"  {mode}: {i + 1}/{len(clients)} clients", end="\r")
        print()
    q = pd.DataFrame(rows)
    ref = q[q["mode"] == "full"].set_index(["client", "task"])["answer"]
    scores = []
    for r in q.itertuples():
        if r.mode == "full":
            scores.append(5)
            continue
        if r.refused:
            scores.append(1)
            continue
        g = llm.complete("You are a strict, consistent grader.", JUDGE.format(ref=ref[(r.client, r.task)], cand=r.answer))
        m = re.search(r"[1-5]", g)
        scores.append(int(m.group(0)) if m else None)
    q["judge_score"] = scores
    q.to_csv(out / "llm_quality.csv", index=False)
    summ = q.groupby(["task", "mode"]).agg(judge_score=("judge_score", "mean"), k_median=("k", "median"),
                                           unresolved=("unresolved_tokens", "mean"),
                                           name_leaked=("name_leaked_to_model", "mean")).reset_index()
    summ.to_csv(out / "llm_quality_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))

    cats = sorted(directory.occ_cat.unique())
    att = []
    for c in clients:
        for task in CORE_TASKS:
            ctx = build_prompt(task, c, TASKS[task]["default"], evaluation_session(f"attack:{c.client_id}:{task}"))
            g = llm.complete("Answer with JSON only.", ATTACK.format(ctx=ctx, cats=json.dumps(cats)))
            try:
                guess = json.loads(g[g.index("{"):g.rindex("}") + 1])
            except ValueError:
                guess = {}
            lo = (c.age // 10) * 10
            att.append(dict(client=c.client_id, task=task,
                            occupation_hit=guess.get("occupation_group") == c.occ_cat,
                            region_hit=guess.get("region") == REGION[c.state],
                            age_band_hit=str(guess.get("age_band", "")).replace(" ", "") == f"{lo}-{lo + 9}"))
    a = pd.DataFrame(att)
    a.to_csv(out / "llm_attacker.csv", index=False)
    print(a.groupby("task")[["occupation_hit", "region_hit", "age_band_hit"]].mean().round(2).to_string())
    base = dict(occupation=directory.occ_cat.value_counts(normalize=True).iloc[0],
                region=directory.region.value_counts(normalize=True).iloc[0],
                age_band=(directory.age // 10).value_counts(normalize=True).iloc[0])
    pd.DataFrame([base]).to_csv(out / "llm_attacker_baseline.csv", index=False)
    print("majority-guess baseline:", {k: round(v, 2) for k, v in base.items()})


if __name__ == "__main__":
    main()
