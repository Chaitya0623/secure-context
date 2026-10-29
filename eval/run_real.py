"""Run every metric on the public datasets.

    python3 scripts/download_data.py        # once
    python3 eval/run_real.py                # about 5 to 10 minutes
    python3 eval/run_real.py --no-presidio  # faster, skips the spaCy detector
    python3 eval/run_real.py --gliner nvidia/gliner-PII   # add a GLiNER detector (downloads the model)
    NVIDIA_API_KEY=... python3 eval/run_real.py --gliner nvidia-api --detect-docs 400 --detect-notes 200
                                            # hosted nvidia/gliner-pii, no download

Writes eval/results_real/*.csv, eval/results_real/summary.json and eval/figures_real/*.png.
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
sys.path.insert(0, str(ROOT / "eval"))

import run_all  # noqa: E402  reuse the experiment functions
from secure_context import exposure  # noqa: E402
from secure_context.attacks import key_attacks  # noqa: E402
from secure_context.detect import CRMDictionaryDetector, EnsembleDetector, RegexDetector, score_detection  # noqa: E402
from secure_context.policy import ALL_EXACT, TASKS  # noqa: E402
from secure_context.privacy import ExposureLedger, KEngine, choose_policy  # noqa: E402
from secure_context.real_data import (GRETEL_DIRECT, INTENT_TASK, load_advisory, load_bitext, load_gretel,  # noqa: E402
                                      load_real_environment)
from secure_context.strategy import extraction_curve, honest_user_error  # noqa: E402
from secure_context.tasks import CORE_TASKS, utility  # noqa: E402
from secure_context.tokens import Session  # noqa: E402
from secure_context.vault import SplitStorage  # noqa: E402

RES = ROOT / "eval" / "results_real"
FIG = ROOT / "eval" / "figures_real"
K_MIN = 10
_API_DETECTORS = []
log = run_all.log


def gretel_detection(docs, use_presidio, gliner_models):
    from secure_context.detect import Detected
    dets = {"regex": [RegexDetector()]}
    if use_presidio:
        from secure_context.detect import PresidioDetector
        p = PresidioDetector()
        dets["presidio"] = [p]
        dets["regex + presidio"] = [RegexDetector(), p]
    for mid in gliner_models:
        from secure_context.detect import model_detector
        g = model_detector(mid)
        _API_DETECTORS.append(g)
        dets[mid] = [g]
        dets[f"regex + {mid}"] = [RegexDetector(), g]
    rows, per = [], []
    for name, ds in dets.items():
        det = EnsembleDetector(ds)
        caught, labels, tp, npred, leaky, t = [], [], 0, 0, 0, 0.0
        for d in docs:
            gold = [Detected(s["start"], s["end"], s["label"], d["text"][s["start"]:s["end"]], "") for s in d["spans"]
                    if s["label"] in GRETEL_DIRECT]
            allgold = [Detected(s["start"], s["end"], s["label"], "", "") for s in d["spans"]]
            t0 = time.perf_counter()
            pred = det.detect(d["text"])
            t += time.perf_counter() - t0
            cg, _, _ = score_detection(gold, pred, len(d["text"]))
            _, tpp, npp = score_detection(allgold, pred, len(d["text"]))
            caught += cg
            labels += [g.label for g in gold]
            tp += tpp
            npred += npp
            leaky += not all(cg)
        caught = np.array(caught)
        rows.append(dict(detector=name, direct_id_recall=caught.mean(), precision=tp / max(npred, 1),
                         docs_with_direct_leak=leaky / len(docs), ms_per_doc=t / len(docs) * 1e3))
        for lab in sorted(set(labels)):
            m = np.array(labels) == lab
            per.append(dict(detector=name, label=lab, recall=caught[m].mean(), n=int(m.sum())))
    df, pl = pd.DataFrame(rows), pd.DataFrame(per)
    df.to_csv(RES / "gretel_detection.csv", index=False)
    pl.to_csv(RES / "gretel_detection_by_label.csv", index=False)
    piv = pl.pivot(index="label", columns="detector", values="recall")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    piv.plot.bar(ax=ax, width=.8)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_title(f"Direct-identifier recall on {len(docs):,} Gretel financial documents")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "gretel_detection.png")
    plt.close(fig)
    return df


def single_request_all(clients, engine):
    rows = []
    for task in TASKS:
        for mode in ("masked", "default", "adaptive"):
            ks, us, ref = [], [], 0
            for c in clients:
                if mode == "adaptive":
                    best = choose_policy(task, c, engine, K_MIN)
                    if best is None:
                        ref += 1
                        ks.append(engine.total)
                        us.append(0.0)
                        continue
                    lv, u, k = best
                else:
                    lv = ALL_EXACT if mode == "masked" else TASKS[task]["default"]
                    k, u = engine.k(c, lv), utility(task, c, lv)
                ks.append(k)
                us.append(u)
            ks = np.array(ks)
            rows.append(dict(task=task, mode=mode, pct_unique=(ks == 1).mean(), pct_k_lt_10=(ks < K_MIN).mean(),
                             median_k=float(np.median(ks)), mean_utility=float(np.mean(us)), refused=ref / len(clients)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "single_request.csv", index=False)
    return df


def request_mix(bitext):
    mix = bitext.task.value_counts(normalize=True).rename("share").reset_index()
    mix.to_csv(RES / "request_mix.csv", index=False)
    by_intent = bitext.groupby(["intent", "task"]).size().rename("n").reset_index()
    by_intent.to_csv(RES / "request_intents.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(mix["task"], mix.share * 100, color="#1E5B86")
    ax.set_xlabel("Share of 23,806 Bitext wealth-management requests (%)")
    ax.invert_yaxis()
    ax.set_title("Which client context real request types need")
    fig.tight_layout()
    fig.savefig(FIG / "request_mix.png")
    plt.close(fig)
    return mix


def cumulative_real_mix(clients, engine, bitext, n_requests=12, seed=3):
    rng = random.Random(seed)
    tasks = bitext.task.tolist()
    seqs = {c.client_id: [rng.choice(tasks) for _ in range(n_requests)] for c in clients}
    rows = []
    for mode in ("default", "adaptive"):
        ledger = ExposureLedger(engine)
        for step in range(n_requests):
            ks, us, ref = [], [], 0
            for c in clients:
                task = seqs[c.client_id][step]
                if mode == "default":
                    lv = TASKS[task]["default"]
                    ledger.record(c, lv, task)
                    us.append(utility(task, c, lv))
                else:
                    best = choose_policy(task, c, engine, K_MIN, ledger)
                    if best is None:
                        ref += 1
                        us.append(0.0)
                    else:
                        ledger.record(c, best[0], task)
                        us.append(best[1])
                ks.append(ledger.k_now(c))
            ks = np.array(ks)
            rows.append(dict(mode=mode, request=step + 1, pct_k_lt_10=(ks < K_MIN).mean(), pct_unique=(ks == 1).mean(),
                             median_k=float(np.median(ks)), mean_utility=float(np.mean(us)),
                             refusal_rate=ref / len(clients)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "cumulative_real_mix.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    for mode, col, lab in (("default", "#A6620F", "fixed task policy"), ("adaptive", "#22704C", "adaptive + ledger")):
        d = df[df["mode"] == mode]
        ax.plot(d.request, d.pct_k_lt_10 * 100, marker="o", ms=3, color=col, label=lab)
    ax.set_xlabel("Request number (task mix from Bitext intents)")
    ax.set_ylabel("Clients with cumulative k < 10 (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "cumulative_real_mix.png")
    plt.close(fig)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=1000)
    ap.add_argument("--no-presidio", action="store_true")
    ap.add_argument("--gliner", nargs="*", default=[])
    ap.add_argument("--reuse-detection", action="store_true", help="reuse detection CSVs from a previous run")
    ap.add_argument("--detect-docs", type=int, default=None, help="limit Gretel documents (API rate limits)")
    ap.add_argument("--detect-notes", type=int, default=None, help="limit client notes for detection")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    run_all.RES, run_all.FIG = RES, FIG  # reused experiment functions write here

    log("loading public datasets")
    directory, clients = load_real_environment(args.clients)
    adv, bitext, docs = load_advisory(), load_bitext(), load_gretel()
    docs = random.Random(0).sample(docs, len(docs))  # so --detect-docs takes a random subset
    summary = {"datasets": {
        "Census ACS PUMS 2023 adults (directory)": len(directory),
        "Fed SCF 2022 records (wealth donors)": int(pd.read_csv(ROOT / "data/processed/scf2022.csv.gz").shape[0]),
        "Financial-Advisory-Clients (profiles)": len(adv), "clients evaluated": len(clients),
        "Bitext wealth-management requests": len(bitext), "Gretel English financial documents": len(docs)},
        "k_min": K_MIN}
    engine = KEngine(directory)

    if args.reuse_detection and (RES / "gretel_detection.csv").exists() and (RES / "detection.csv").exists():
        log("reusing detection results")
        summary["gretel_detection"] = pd.read_csv(RES / "gretel_detection.csv").to_dict(orient="records")
        summary["note_detection"] = pd.read_csv(RES / "detection.csv").to_dict(orient="records")
    else:
        log("PII detection on Gretel financial documents")
        summary["gretel_detection"] = gretel_detection(docs[:args.detect_docs], not args.no_presidio,
                                                       args.gliner).to_dict(orient="records")
        log("PII detection on client notes (template and stress)")
        summary["note_detection"] = run_all.eval_detection(clients[:args.detect_notes], not args.no_presidio,
                                                           args.gliner).to_dict(orient="records")

    for d in _API_DETECTORS + run_all._API_DETECTORS:
        if hasattr(d, "flush"):
            d.flush()

    log("split storage and key attacks")
    storage = SplitStorage(EnsembleDetector([RegexDetector(), CRMDictionaryDetector()]))
    ctx_ids = [storage.ingest(c) for c in clients]
    ka = key_attacks(storage, clients, ctx_ids)
    ka.to_csv(RES / "key_attacks.csv", index=False)
    summary["key_attacks"] = ka.to_dict(orient="records")

    log("single request, every task (builds per-client match histograms on the first pass)")
    summary["single_request"] = single_request_all(clients, engine).to_dict(orient="records")

    log("privacy-utility frontier")
    run_all.eval_frontier(clients, engine)

    log("request mix and cumulative exposure under real request types")
    summary["request_mix"] = request_mix(bitext).to_dict(orient="records")
    cum = cumulative_real_mix(clients, engine, bitext)
    summary["cumulative_final"] = cum[cum.request == cum.request.max()].to_dict(orient="records")
    cum3 = run_all.eval_cumulative(clients, engine)
    summary["cumulative_core_tasks_final"] = cum3[cum3.request == cum3.request.max()].to_dict(orient="records")

    log("attribute inference")
    summary["attribute_inference"] = run_all.eval_inference(directory, clients).to_dict(orient="records")

    log("transaction unicity")
    tu = exposure.transaction_unicity(exposure.transaction_streams(clients))
    tu.to_csv(RES / "transaction_unicity.csv", index=False)
    summary["transaction_unicity"] = tu.to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(6, 3))
    for rep, g in tu.groupby("representation", sort=False):
        ax.plot(g.known_points, g.pct_unique * 100, marker="o", ms=3, label=rep)
    ax.set_xlabel("Transactions the attacker knows")
    ax.set_ylabel("Clients singled out (%)")
    ax.legend(fontsize=7)
    ax.set_title("Transaction unicity in the client book")
    fig.tight_layout()
    fig.savefig(FIG / "transaction_unicity.png")
    plt.close(fig)

    log("advisor query exposure")
    queries = exposure.make_advisor_queries(clients, bitext.instruction.tolist())
    aq, example = exposure.advisor_query_exposure(queries, engine, storage.detector, Session)
    aq_sum = dict(queries=len(aq), pct_with_direct_identifier=float(aq.has_direct.mean()),
                  pct_direct_left_after_sanitizing=float(aq.direct_left.mean()),
                  pct_unique_from_query_qis=float((aq.k_raw[aq.qi_count > 0] == 1).mean()),
                  pct_k_lt_10_from_query_qis=float((aq.k_raw[aq.qi_count > 0] < K_MIN).mean()),
                  pct_k_lt_10_after_sanitizing=float((aq.k_clean[aq.qi_count > 0] < K_MIN).mean()),
                  example_sanitized=example)
    aq.to_csv(RES / "advisor_queries.csv", index=False)
    summary["advisor_queries"] = aq_sum

    log("reference population")
    rp = exposure.reference_population_k(directory, clients, {t: TASKS[t]["default"] for t in CORE_TASKS})
    rp.to_csv(RES / "reference_population.csv", index=False)
    summary["reference_population"] = rp.to_dict(orient="records")

    log("policy side channel")
    sc = [exposure.policy_side_channel(clients, engine, t) for t in CORE_TASKS]
    pd.DataFrame(sc).to_csv(RES / "policy_side_channel.csv", index=False)
    summary["policy_side_channel"] = sc

    log("exact amounts as fingerprints")
    af = exposure.amount_fingerprints(adv)
    af.to_csv(RES / "amount_fingerprints.csv", index=False)
    summary["amount_fingerprints"] = af.to_dict(orient="records")

    log("strategy extraction (advisor books drawn from the real client profiles)")
    book = pd.DataFrame(dict(age=[c.age for c in clients], risk=[c.risk for c in clients], aum=[max(c.aum, .01) for c in clients],
                             objective=["Retirement income" if c.objective == "Retirement" else c.objective for c in clients],
                             top_single=[c.holdings[0].single for c in clients],
                             top_weight=[c.holdings[0].weight for c in clients]))
    curve = extraction_curve(book)
    curve.to_csv(RES / "strategy_extraction.csv", index=False)
    summary["strategy_at_2000_queries"] = curve[curve.queries == curve.queries.max()].to_dict(orient="records")
    summary["strategy_honest_user_error"] = honest_user_error()

    log("differential privacy, rehydration, latency")
    summary["dp"] = run_all.eval_dp(clients).to_dict(orient="records")
    summary["rehydration"] = run_all.eval_rehydration(clients).to_dict(orient="records")
    summary["latency_ms_mean"] = run_all.eval_latency(clients, ctx_ids, storage, engine)["mean"].round(3).to_dict()

    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done: results in {RES.relative_to(ROOT)}, figures in {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
