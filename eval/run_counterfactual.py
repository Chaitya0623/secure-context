"""Counterfactual necessity test: measure which client fields each task's answer depends on.

Offline, with the template simulator (checks the plumbing; no key needed):
    python3 eval/run_counterfactual.py

With a real model on Groq (answers and judge default to openai/gpt-oss-120b):
    export GROQ_API_KEY=...
    python3 eval/run_counterfactual.py --backend groq --n 10

For each client and task the model answers with the full tokenized context, then once per
counterfactual: each field at every coarser level, each field's value swapped with another
client's, and a null control that only changes session tokens. Real-model answers are compared
by an LLM judge (would an advisor act differently?); simulator answers by a deterministic
comparison that ignores restated inputs. See secure_context/counterfactual.py.

Cost: about 32 answers and up to 31 judge calls per client and task (19 coarser levels, 11 swaps,
baseline, null), so --n 10 on the three core tasks is roughly 1,900 calls. Groq responses are
cached in data/cache/groq/, so an interrupted run resumes where it stopped.

Writes to eval/results_counterfactual/<backend>/:
    raw.csv       one row per client, task and counterfactual
    summary.csv   change rate per task, field and level
    needs.csv     measured need vs. the hand-coded need in policy.py, with swap sensitivity
    privacy.csv   crowd size k of the measured policy vs. the hand-coded task policy
    change_rates.png
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context import counterfactual as cf  # noqa: E402
from secure_context import real_data  # noqa: E402
from secure_context.data import load_environment  # noqa: E402
from secure_context.llm import get_backend  # noqa: E402
from secure_context.policy import FIELDS, TASKS  # noqa: E402
from secure_context.privacy import KEngine  # noqa: E402
from secure_context.tasks import CORE_TASKS  # noqa: E402
from secure_context.tokens import Session  # noqa: E402


def stable_session(identity: str) -> Session:
    """Benchmark-only: a fixed key per prompt so cached real-model calls resume exactly."""
    d = ROOT / "data/cache/eval_sessions"
    d.mkdir(parents=True, exist_ok=True)
    path = d / hashlib.sha256(identity.encode()).hexdigest()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    return Session(key=path.read_bytes())


def privacy_of_policies(needs: pd.DataFrame, clients, engine: KEngine, k_min: int) -> pd.DataFrame:
    rows = []
    for task in needs.task.unique():
        for name, levels in (("hand-coded task policy", TASKS[task]["default"]),
                             ("measured policy", cf.measured_policy(needs, task))):
            k = np.array([engine.k(c, levels) for c in clients])
            rows.append(dict(task=task, policy=name, k_median=float(np.median(k)), pct_k_lt_min=float((k < k_min).mean()),
                             pct_unique=float((k == 1).mean()),
                             fields_sent=", ".join(f"{f}={lv}" for f, lv in levels.items() if lv != "drop")))
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    s = summary[summary.kind.isin(["level", "swap"])].copy()
    s["row"] = np.where(s.kind == "swap", s.field + ": value swapped", s.field + ": " + s.level)
    order = []
    for f in FIELDS:
        order += [r for r in s[s.field == f].row.drop_duplicates() if not r.endswith("swapped")]
        order += [r for r in s[s.field == f].row.drop_duplicates() if r.endswith("swapped")]
    m = s.pivot_table(index="row", columns="task", values="change_rate").reindex(order)
    m = m[[t for t in TASKS if t in m.columns]]
    # one-hue sequential ramp, light (no change) to dark (always changes)
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(2.2 + 1.5 * m.shape[1], 0.27 * len(m) + 1.4))
    ax.imshow(m.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 0.55 else "#1f1f1f")
    ax.set_xticks(range(m.shape[1]), [TASKS[t]["name"] for t in m.columns], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=7.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=9, pad=28, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="simulated", help="simulated, groq, nvidia, anthropic, openai, hf-api, hf-local")
    ap.add_argument("--model", default=None, help="answer model")
    ap.add_argument("--judge-model", default=None, help="judge model (defaults to the answer model)")
    ap.add_argument("--n", type=int, default=None, help="clients (default 50 simulated, 10 for a real model)")
    ap.add_argument("--tasks", default=",".join(CORE_TASKS), help="comma-separated, or 'all'")
    ap.add_argument("--fields", default=None, help="comma-separated subset of fields (default all)")
    ap.add_argument("--tolerance", type=float, default=0.10, help="allowed change rate above the null control")
    ap.add_argument("--k-min", type=int, default=10)
    ap.add_argument("--synthetic", action="store_true", help="use the fully synthetic environment")
    ap.add_argument("--keep-answers", action="store_true", help="store answer text in raw.csv")
    args = ap.parse_args()

    simulated = args.backend == "simulated"
    n = args.n or (50 if simulated else 10)
    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    fields = args.fields.split(",") if args.fields else None
    llm = get_backend(args.backend, args.model)
    if simulated:
        comparator, sessions = cf.echo_masked, None
    else:
        comparator = cf.Judge(get_backend(args.backend, args.judge_model or args.model))
        sessions = stable_session

    if args.synthetic or not real_data.available():
        directory, book = load_environment(n_pop=50_000, n_clients=max(n, 200))
        source = "synthetic"
    else:
        directory, book = real_data.load_real_environment(n_clients=max(n, 200))
        source = "public data"
    clients = book[:n]
    print(f"answers: {llm.name}   comparator: {getattr(comparator, 'name', 'echo_masked')}   "
          f"data: {source}   clients: {n}   tasks: {', '.join(tasks)}")

    raw = cf.run(llm, clients, tasks, comparator=comparator, pool=book, fields=fields, session_factory=sessions,
                 keep_answers=args.keep_answers, progress=lambda i, t: print(f"  {i}/{t} clients", end="\r"))
    print()
    summary = cf.summarize(raw)
    needs = cf.measured_needs(raw, args.tolerance)
    privacy = privacy_of_policies(needs, book, KEngine(directory), args.k_min)

    out = ROOT / "eval" / "results_counterfactual" / args.backend.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    needs.to_csv(out / "needs.csv", index=False)
    privacy.to_csv(out / "privacy.csv", index=False)
    plot(summary, out / "change_rates.png",
         f"Share of answers that changed when one field changed ({llm.name}, {n} clients)")

    pd.set_option("display.width", 200)
    print("\nNull control (only session tokens changed):")
    print(summary[summary.kind == "null"][["task", "change_rate", "unparsed"]].to_string(index=False))
    print("\nMeasured vs. hand-coded needs:")
    print(needs[["task", "field", "measured_need", "hand_coded_need", "agree", "drop_change_rate",
                "swap_change_rate"]].round(2).to_string(index=False))
    off = needs[~needs.agree]
    print(f"\n{len(off)} of {len(needs)} task-field pairs disagree with policy.py")
    print("\nCrowd size of each policy:")
    print(privacy[["task", "policy", "k_median", "pct_k_lt_min", "pct_unique"]].round(3).to_string(index=False))
    print(f"\nWritten to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
