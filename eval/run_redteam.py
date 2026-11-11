"""Red-team stress test: an attacker on the model side tries to extract client PII through the full pipeline.

    python3 eval/run_redteam.py                              # scripted agents, offline, seconds
    python3 eval/run_redteam.py --presidio                   # add Presidio to the query sanitizer
    export GROQ_API_KEY=...
    python3 eval/run_redteam.py --backend groq --clients 10  # real models for advisor, assistant and attacker

Writes eval/results_redteam/<backend>/:
    sessions.jsonl     every conversation, turn by turn: what the advisor typed, what crossed, the reply, the advisor view
    sessions.csv       one row per session with its scores
    summary.csv        per mode: leaks, sanitizer catch rate, exfiltration with and without the link guard, attacker hits
    leaks_by_item.csv  per mode and client value: how often it was typed, crossed, and which step let it through
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context import redteam as R  # noqa: E402
from secure_context.detect import default_detector  # noqa: E402
from secure_context.llm import get_backend  # noqa: E402
from secure_context.privacy import KEngine  # noqa: E402
from secure_context.real_data import load_real_environment  # noqa: E402
from secure_context.vault import SplitStorage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="simulated", help="model for the assistant and attacker (simulated = scripted)")
    ap.add_argument("--advisor-backend", default=None, help="model playing the advisor (default: same as --backend)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--modes", default=",".join(R.MODES))
    ap.add_argument("--k-min", type=int, default=10)
    ap.add_argument("--presidio", action="store_true", help="add Presidio NER to the query sanitizer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    real = args.backend != "simulated"
    assistant = get_backend(args.backend, args.model) if real else None
    adv_name = args.advisor_backend or args.backend
    advisor = get_backend(adv_name, args.model) if adv_name != "simulated" else None
    out = ROOT / "eval" / "results_redteam" / (args.backend + ("-presidio" if args.presidio else ""))
    out.mkdir(parents=True, exist_ok=True)

    directory, clients = load_real_environment(max(args.clients, 200))
    clients = clients[:args.clients]
    engine = KEngine(directory)
    detector = default_detector(use_presidio=args.presidio)
    storage = SplitStorage(detector)

    rows, item_rows = [], []
    with open(out / "sessions.jsonl", "w") as fh:
        for c in clients:
            ctx_id = storage.ingest(c)
            for mode in args.modes.split(","):
                s = R.run_session(c, lambda sess: storage.session_notes(ctx_id, sess), engine, detector, mode,
                                  args.turns, args.k_min, assistant, advisor, args.seed)
                sc = R.score_session(s, c)
                fh.write(json.dumps(dict(**asdict(s), score=sc), default=str) + "\n")
                rows.append(sc)
                for item in R.secrets(c):
                    where = sc["origin"].get(item, (None, None))[0]
                    item_rows.append(dict(mode=mode, item=item.split("_")[0] if item.startswith("child") else item,
                                          typed=item in sc["typed"], crossed=item in sc["crossed"], step=where))
            print(f"{c.client_id}: done", flush=True)

    df = pd.DataFrame(rows)
    att = pd.json_normalize(df.attacker).add_prefix("attacker_")
    flat = pd.concat([df.drop(columns=["attacker", "origin"]), att], axis=1)
    flat.to_csv(out / "sessions.csv", index=False)

    summary = flat.groupby("mode").agg(
        sessions=("client_id", "size"),
        pct_sessions_with_crossing=("crossed", lambda s: 100 * (s.str.len() > 0).mean()),
        mean_values_typed=("typed", lambda s: s.str.len().mean()),
        mean_values_crossed=("crossed", lambda s: s.str.len().mean()),
        sanitizer_catch_rate=("sanitizer_catch_rate", "mean"),
        pct_sessions_exfil=("exfil_links", lambda s: 100 * (s > 0).mean()),
        pct_sessions_exfil_guarded=("exfil_links_guarded", lambda s: 100 * (s > 0).mean()),
        refused_turns=("refused_turns", "sum"),
        median_k_final=("k_final", "median"),
        **{f"attacker_{k}": (f"attacker_{k}", "mean") for k in att.columns.str.replace("attacker_", "")},
    ).round(3)
    summary.to_csv(out / "summary.csv")

    items = pd.DataFrame(item_rows)
    by_item = items.groupby(["mode", "item"]).agg(typed=("typed", "mean"), crossed=("crossed", "mean")).round(3)
    steps = items[items.crossed].groupby(["mode", "item"]).step.agg(lambda s: Counter(s).most_common(1)[0][0])
    by_item = by_item.join(steps.rename("main_step")).reset_index()
    by_item.to_csv(out / "leaks_by_item.csv", index=False)

    pd.set_option("display.width", 220)
    print(summary.T.to_string())
    print(by_item[by_item.crossed > 0].to_string(index=False))
    print(f"\nWritten to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
