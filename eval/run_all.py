"""Run the full evaluation and write CSVs, figures and summary.json.

    python eval/run_all.py                 # everything, about 2 to 4 minutes on a laptop
    python eval/run_all.py --no-presidio   # skip the spaCy/Presidio detector
    python eval/run_all.py --clients 300   # smaller run
    python eval/run_all.py --gliner nvidia/gliner-PII   # add GLiNER (needs `pip install gliner` and HF access)

Results land in eval/results/ and eval/figures/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context.attacks import attribute_inference, key_attacks  # noqa: E402
from secure_context.data import load_environment, make_stress_notes  # noqa: E402
from secure_context.detect import (CRMDictionaryDetector, EnsembleDetector, RegexDetector,  # noqa: E402
                                   resolve_coreference, score_detection)
from secure_context.dp import differencing_attack, utility_error  # noqa: E402
from secure_context.llm import SimulatedLLM  # noqa: E402
from secure_context.pipeline import Pipeline  # noqa: E402
from secure_context.policy import ALL_EXACT, TASKS  # noqa: E402
from secure_context.privacy import ExposureLedger, KEngine, candidate_policies, choose_policy  # noqa: E402
from secure_context.strategy import extraction_curve, honest_user_error  # noqa: E402
from secure_context.tasks import CORE_TASKS, simulated_answer, utility  # noqa: E402
from secure_context.tokens import Session, rehydrate  # noqa: E402
from secure_context.vault import SplitStorage  # noqa: E402

RES = ROOT / "eval" / "results"
FIG = ROOT / "eval" / "figures"
K_MIN = 10
_API_DETECTORS = []

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C = {"masked": "#B23A30", "default": "#A6620F", "adaptive": "#22704C", "full": "#5A6A74"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eval_detection(clients, use_presidio, gliner_models=()):
    dets = {"regex": [RegexDetector()], "crm_dictionary": [CRMDictionaryDetector()],
            "regex + crm_dictionary": [RegexDetector(), CRMDictionaryDetector()]}
    if use_presidio:
        from secure_context.detect import PresidioDetector
        p = PresidioDetector()
        dets["presidio"] = [p]
        dets["regex + presidio"] = [RegexDetector(), p]
        dets["regex + crm_dictionary + presidio"] = [RegexDetector(), CRMDictionaryDetector(), p]
    for mid in gliner_models:
        from secure_context.detect import model_detector
        g = model_detector(mid)
        _API_DETECTORS.append(g)
        dets[mid] = [g]
        dets[f"regex + crm_dictionary + {mid}"] = [RegexDetector(), CRMDictionaryDetector(), g]
    stress = make_stress_notes(clients)
    sets = {"template": [(c.notes, c.gold_spans, c) for c in clients],
            "stress": [(t, g, c) for (t, g), c in zip(stress, clients)]}
    rows, per_label = [], []
    for set_name, items in sets.items():
        for name, ds in dets.items():
            det = EnsembleDetector(ds)
            caught_all, labels, tp, npred, leaky, t = [], [], 0, 0, 0, 0.0
            for text, gold, c in items:
                t0 = time.perf_counter()
                pred = det.detect(text, c)
                t += time.perf_counter() - t0
                caught, tpp, npp = score_detection(gold, pred, len(text))
                caught_all += caught
                labels += [g.label for g in gold]
                tp += tpp
                npred += npp
                leaky += not all(caught)
            caught_all = np.array(caught_all)
            rows.append(dict(note_set=set_name, detector=name, span_recall=caught_all.mean(),
                             precision=tp / max(npred, 1), notes_with_leak=leaky / len(items),
                             ms_per_note=t / len(items) * 1e3))
            for lab in sorted(set(labels)):
                m = np.array(labels) == lab
                per_label.append(dict(note_set=set_name, detector=name, label=lab, recall=caught_all[m].mean(),
                                      n=int(m.sum())))
    df, pl = pd.DataFrame(rows), pd.DataFrame(per_label)
    df.to_csv(RES / "detection.csv", index=False)
    pl.to_csv(RES / "detection_by_label.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, set_name in zip(axes, sets):
        piv = pl[pl.note_set == set_name].pivot(index="label", columns="detector", values="recall")
        piv.plot.bar(ax=ax, width=.8, legend=set_name == "stress")
        ax.set_title(f"{set_name} notes")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
    axes[0].set_ylabel("Span recall")
    axes[1].legend(fontsize=6, ncol=2, loc="lower left")
    fig.suptitle("PII detection recall by type (one missed span is one leak)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "detection_recall.png")
    plt.close(fig)
    return df


def eval_single_request(clients, engine):
    rows = []
    for task in CORE_TASKS:
        modes = {"masked": lambda c: dict(ALL_EXACT), "default": lambda c: dict(TASKS[task]["default"])}
        for mode, fn in modes.items():
            ks = np.array([engine.k(c, fn(c)) for c in clients])
            us = np.array([utility(task, c, fn(c)) for c in clients])
            rows.append(dict(task=task, mode=mode, pct_unique=(ks == 1).mean(), pct_k_lt_10=(ks < K_MIN).mean(),
                             median_k=float(np.median(ks)), mean_utility=us.mean(), refused=0.0))
        ks, us, ref = [], [], 0
        for c in clients:
            best = choose_policy(task, c, engine, K_MIN)
            if best is None:
                ref += 1
                ks.append(engine.n)
                us.append(0.0)
            else:
                ks.append(best[2])
                us.append(best[1])
        ks, us = np.array(ks), np.array(us)
        rows.append(dict(task=task, mode="adaptive", pct_unique=(ks == 1).mean(), pct_k_lt_10=(ks < K_MIN).mean(),
                         median_k=float(np.median(ks)), mean_utility=us.mean(), refused=ref / len(clients)))
        rows.append(dict(task=task, mode="full", pct_unique=1.0, pct_k_lt_10=1.0, median_k=1.0, mean_utility=1.0,
                         refused=0.0))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "single_request.csv", index=False)
    return df


def eval_frontier(clients, engine):
    rows = []
    for task in CORE_TASKS:
        for lv in candidate_policies(task):
            ks = np.array([engine.k(c, lv) for c in clients])
            us = np.array([utility(task, c, lv) for c in clients])
            rows.append(dict(task=task, policy=json.dumps(lv), mean_utility=us.mean(), pct_k_lt_10=(ks < K_MIN).mean(),
                             median_k=float(np.median(ks)), is_default=lv == TASKS[task]["default"]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "frontier.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, task in zip(axes, CORE_TASKS):
        d = df[df.task == task]
        ax.scatter(d.pct_k_lt_10 * 100, d.mean_utility, s=10, color="#9DB3C2", label="candidate policies")
        dd = d[d.is_default]
        ax.scatter(dd.pct_k_lt_10 * 100, dd.mean_utility, s=50, color=C["default"], label="task default", zorder=3)
        ks = np.array([engine.k(c, ALL_EXACT) for c in clients])
        ax.scatter([(ks < K_MIN).mean() * 100], [1.0], s=50, color=C["masked"], marker="s", label="masking only",
                   zorder=3)
        ax.set_title(TASKS[task]["name"])
        ax.set_xlabel("Clients with k < 10 (%)")
    axes[0].set_ylabel("Task utility")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Privacy vs. usefulness for every policy the minimizer can choose", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "frontier.png")
    plt.close(fig)
    return df


def eval_cumulative(clients, engine, n_requests=12, seed=3):
    rng = random.Random(seed)
    seqs = {c.client_id: [rng.choice(list(CORE_TASKS)) for _ in range(n_requests)] for c in clients}
    rows = []
    for mode in ("default", "adaptive"):
        ledger = ExposureLedger(engine)
        for step in range(n_requests):
            ks, us, ref = [], [], 0
            for c in clients:
                task = seqs[c.client_id][step]
                if mode == "default":
                    lv = TASKS[task]["default"]
                    u = utility(task, c, lv)
                    ledger.record(c, lv, task)
                else:
                    best = choose_policy(task, c, engine, K_MIN, ledger)
                    if best is None:
                        ref += 1
                        u = 0.0
                    else:
                        ledger.record(c, best[0], task)
                        u = best[1]
                ks.append(ledger.k_now(c))
                us.append(u)
            ks = np.array(ks)
            rows.append(dict(mode=mode, request=step + 1, pct_k_lt_10=(ks < K_MIN).mean(), pct_unique=(ks == 1).mean(),
                             median_k=float(np.median(ks)), mean_utility=float(np.mean(us)), refusal_rate=ref / len(clients)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "cumulative.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for mode in ("default", "adaptive"):
        d = df[df["mode"] == mode]
        lab = "fixed task policy" if mode == "default" else "adaptive + exposure ledger"
        axes[0].plot(d.request, d.pct_k_lt_10 * 100, marker="o", ms=3, color=C[mode], label=lab)
        axes[1].plot(d.request, d.mean_utility, marker="o", ms=3, color=C[mode], label=lab)
    axes[0].set_ylabel("Clients with cumulative k < 10 (%)")
    axes[1].set_ylabel("Mean task utility of that request")
    for ax in axes:
        ax.set_xlabel("Request number (random tasks per client)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Exposure accumulates across requests unless it is budgeted", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "cumulative.png")
    plt.close(fig)
    return df


def eval_inference(pop, clients):
    out = []
    for task in CORE_TASKS:
        df = attribute_inference(pop, clients, TASKS[task]["default"])
        df.insert(0, "task", task)
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "attribute_inference.csv", index=False)
    return df


def eval_strategy(pop):
    curve = extraction_curve(pop)
    curve.to_csv(RES / "strategy_extraction.csv", index=False)
    honest = honest_user_error()
    pd.DataFrame([honest]).to_csv(RES / "strategy_honest_error.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    col = {"none": "#B23A30", "coarse": "#1E5B86", "noise": "#A6620F", "noise_cached": "#7A4BA0", "book_bound": "#22704C",
           "book_bound+coarse": "#13222C"}
    for d, g in curve.groupby("defense"):
        axes[0].plot(g.queries, g.equity_mae, marker="o", ms=3, color=col[d], label=d)
        axes[1].plot(g.queries, g.alt_rule_acc, marker="o", ms=3, color=col[d], label=d)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Surrogate error on equity % (higher = safer)")
    axes[1].set_ylabel("Recovered 'alternatives' rule accuracy")
    for ax in axes:
        ax.set_xlabel("Attacker queries")
    axes[0].legend(fontsize=7)
    fig.suptitle("Extracting the planted model-portfolio strategy", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "strategy_extraction.png")
    plt.close(fig)
    return curve, honest


def eval_dp(clients):
    eq = np.array([100 - sum(h.weight for h in c.holdings if h.group in ("Fixed income", "Cash")) for c in clients],
                  dtype=float)
    rows = []
    for size in (50, 500):
        vals = eq[:size]
        guess = float(np.abs(vals - vals.mean()).mean())  # attacker who just guesses the cohort average
        for eps in (None, 0.1, 0.5, 1.0, 2.0, 5.0):
            rows.append(dict(cohort_size=size, epsilon="exact" if eps is None else eps,
                             differencing_attack_error=differencing_attack(vals, eps),
                             blind_guess_error=guess,
                             cohort_mean_error=0.0 if eps is None else utility_error(vals, eps)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "dp_cohorts.csv", index=False)
    return df


def eval_rehydration(clients, n=300, seed=5):
    rng = random.Random(seed)
    kinds = ["intact", "lowercase", "truncated", "no_brackets", "hallucinated"]
    rows = []
    for kind in kinds:
        ok = flagged = wrong = 0
        for c in clients[:n]:
            s = Session()
            ans = simulated_answer("meeting_prep", c, TASKS["meeting_prep"]["default"], s)
            tok = s.token("PERSON", c.name)
            if kind == "lowercase":
                bad = tok.lower()
            elif kind == "truncated":
                bad = tok[:-2] + ">"
            elif kind == "no_brackets":
                bad = tok[1:-1]
            elif kind == "hallucinated":
                bad = f"<PERSON_{rng.randrange(16 ** 6):06x}>"
            else:
                bad = tok
            r = rehydrate(ans.replace(tok, bad), s)
            if c.name in r.text and not r.unresolved:
                ok += 1
            elif r.unresolved:
                flagged += 1
            else:
                wrong += 1
        rows.append(dict(corruption=kind, restored=ok / n, flagged_for_review=flagged / n, silently_wrong=wrong / n))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "rehydration.csv", index=False)
    return df


def eval_latency(clients, ctx_ids, storage, engine):
    pipe = Pipeline(storage, engine, SimulatedLLM(), K_MIN)
    t = []
    for c, ctx in list(zip(clients, ctx_ids))[:300]:
        for task in CORE_TASKS:
            tr = pipe.run(c, ctx, task, mode="adaptive")
            t.append(tr.timings)
    df = pd.DataFrame(t).describe(percentiles=[.5, .95]).T[["mean", "50%", "95%"]]
    df.to_csv(RES / "latency_ms.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=1000)
    ap.add_argument("--population", type=int, default=50_000)
    ap.add_argument("--no-presidio", action="store_true")
    ap.add_argument("--gliner", nargs="*", default=[],
                    help="GLiNER model ids to add, e.g. nvidia/gliner-PII knowledgator/gliner-pii-base-v1.0")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    log("generating synthetic environment")
    pop, clients = load_environment(args.population, args.clients)
    engine = KEngine(pop)
    summary = {"population": len(pop), "clients": len(clients), "k_min": K_MIN}

    log("detection")
    det = eval_detection(clients, not args.no_presidio, args.gliner)
    summary["detection"] = det.to_dict(orient="records")

    log("split storage and key attacks")
    storage = SplitStorage(EnsembleDetector([RegexDetector(), CRMDictionaryDetector()]))
    ctx_ids = [storage.ingest(c) for c in clients]
    ka = key_attacks(storage, clients, ctx_ids)
    ka.to_csv(RES / "key_attacks.csv", index=False)
    summary["key_attacks"] = ka.to_dict(orient="records")

    log("single-request re-identification")
    summary["single_request"] = eval_single_request(clients, engine).to_dict(orient="records")

    log("privacy-utility frontier")
    eval_frontier(clients, engine)

    log("cumulative exposure across requests")
    cum = eval_cumulative(clients, engine)
    summary["cumulative_final"] = cum[cum.request == cum.request.max()].to_dict(orient="records")

    log("attribute inference")
    summary["attribute_inference"] = eval_inference(pop, clients).to_dict(orient="records")

    log("strategy extraction")
    curve, honest = eval_strategy(pop)
    summary["strategy_at_2000_queries"] = curve[curve.queries == curve.queries.max()].to_dict(orient="records")
    summary["strategy_honest_user_error"] = honest

    log("differential privacy cohorts")
    summary["dp"] = eval_dp(clients).to_dict(orient="records")

    log("rehydration robustness")
    summary["rehydration"] = eval_rehydration(clients).to_dict(orient="records")

    log("latency")
    lat = eval_latency(clients, ctx_ids, storage, engine)
    summary["latency_ms_mean"] = lat["mean"].round(3).to_dict()

    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done: results in {RES.relative_to(ROOT)}, figures in {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
