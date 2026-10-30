"""Exposure channels beyond the client profile.

transaction_unicity     how few known transactions single out a client (de Montjoye-style unicity)
advisor_query_exposure  identifiers and quasi-identifiers typed into the advisor's own request
reference_population_k  crowd size against the firm's client book instead of the general population
policy_side_channel     whether the adaptive policy's choice itself reveals that a client is rare
amount_fingerprints     how unique exact dollar amounts are, and how much rounding helps
"""
from __future__ import annotations

import random
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from .data import Client, REGION
from .detect import resolve_coreference
from .policy import FIELDS, QI_FIELDS, TASKS, age_band, aum_bucket, rank
from .privacy import KEngine, choose_policy


# Transactions

def transaction_streams(clients: list[Client], months: int = 12, seed: int = 0):
    """Synthetic account activity driven by each client's income, assets and goal."""
    rng = random.Random(seed)
    start = date(2025, 1, 1)
    streams = {}
    for c in clients:
        inc = getattr(c, "extra", {}).get("income", 80_000 + c.aum * 50_000)
        tx = []
        pay_day = rng.choice([1, 15, 28])
        contrib_day = rng.randint(2, 27)
        for m in range(months):
            d0 = date(start.year + (start.month - 1 + m) // 12, (start.month - 1 + m) % 12 + 1, 1)
            if inc > 0:
                tx.append((d0 + timedelta(days=pay_day - 1), round(inc / 12 * rng.uniform(.97, 1.03), 2), "deposit"))
            tx.append((d0 + timedelta(days=contrib_day - 1), round(max(50, inc * .01) * rng.choice([1, 1, 1.5, 2]), 2),
                       "contribution"))
            for _ in range(rng.randint(0, 3)):
                tx.append((d0 + timedelta(days=rng.randint(0, 27)), round(rng.lognormvariate(6.5, 1.0), 2), "trade"))
            if rng.random() < .3:
                tx.append((d0 + timedelta(days=rng.randint(0, 27)), round(rng.lognormvariate(7.5, 1.0), 2), "withdrawal"))
        streams[c.client_id] = tx
    return streams


def _amount_bucket(a):
    return int(np.floor(np.log10(max(a, 1)) * 4))  # quarter-decade buckets


REPRESENTATIONS = {
    "exact date and amount": lambda d, a, k: (d.isoformat(), round(a, 2)),
    "week and amount to $100": lambda d, a, k: (d.isocalendar()[1], round(a, -2)),
    "month and amount band": lambda d, a, k: (d.month, _amount_bucket(a)),
    "month, type and amount band": lambda d, a, k: (d.month, k, _amount_bucket(a)),
}


def transaction_unicity(streams: dict, points=(1, 2, 3, 4, 5), trials: int = 600, seed: int = 0):
    rng = random.Random(seed)
    ids = list(streams)
    rows = []
    for rep, fn in REPRESENTATIONS.items():
        sets = {cid: frozenset(fn(d, a, k) for d, a, k in streams[cid]) for cid in ids}
        all_sets = list(sets.values())
        for p in points:
            unique = 0
            for _ in range(trials):
                cid = rng.choice(ids)
                s = list(sets[cid])
                known = set(rng.sample(s, min(p, len(s))))
                matches = sum(1 for other in all_sets if known <= other)
                unique += matches == 1
            rows.append(dict(representation=rep, known_points=p, pct_unique=unique / trials))
    return pd.DataFrame(rows)


# Advisor queries

QUERY_TEMPLATES = [
    ("{instr} This is for {name}.", ()),
    ("For my client {first} ({age}, {occ} in {state}): {instr}", ("age", "occupation", "location")),
    ("{instr} Client is {title} {last}, the {age}-year-old in {state} with about ${aumk}K invested.",
     ("age", "location", "aum")),
    ("Prep for {name}, account {account}: {instr}", ()),
    ("{instr} ({first} {last}, {occ}, {state}, roughly ${aumk}K)", ("occupation", "location", "aum")),
]
STATE_NAMES = {"NY": "New York", "NJ": "New Jersey", "CA": "California", "TX": "Texas", "FL": "Florida",
               "IL": "Illinois"}


def make_advisor_queries(clients: list[Client], instructions: list[str], seed: int = 0):
    rng = random.Random(seed)
    out = []
    for c in clients:
        tpl, qis = rng.choice(QUERY_TEMPLATES)
        aumk = max(1, int(round(c.aum * 1000)))
        text = tpl.format(instr=rng.choice(instructions).strip().rstrip("?.") + ".", name=c.name, first=c.first,
                          last=c.last, title=c.title, age=c.age, occ=c.occ_cat.lower(),
                          state=STATE_NAMES.get(c.state, c.state), aumk=f"{aumk:,}", account=c.account)
        out.append(dict(client=c, text=text, qis=qis))
    return out


def sanitize_query(text: str, c: Client, detector, session):
    """Tokenize identifiers, then generalize the quasi-identifiers the advisor typed."""
    from .tokens import dehydrate
    spans = resolve_coreference(detector.detect(text, c), c)
    t = dehydrate(text, spans, session)
    t = re.sub(r"\b(\d{2})(-year-old)?\b(?=[,)\s])", lambda m: age_band(int(m.group(1))) if 18 <= int(m.group(1)) <= 99
               else m.group(0), t)
    t = re.sub(r"\$([\d,]+)K", lambda m: aum_bucket(int(m.group(1).replace(",", "")) / 1000), t)
    full = STATE_NAMES.get(c.state, c.state)
    t = t.replace(full, f"the {REGION[c.state]} US")
    return t, spans


def advisor_query_exposure(queries, engine: KEngine, detector, session_factory):
    rows = []
    for q in queries:
        c = q["client"]
        direct = [c.name, c.first, c.last, c.account]
        s = session_factory()
        clean, spans = sanitize_query(q["text"], c, detector, s)
        raw_levels = {f: "drop" for f in FIELDS}
        clean_levels = dict(raw_levels)
        for f in q["qis"]:
            raw_levels[f] = {"age": "exact", "location": "state", "occupation": "category", "aum": "exact"}[f]
            clean_levels[f] = {"age": "band", "location": "region", "occupation": "category", "aum": "bucket"}[f]
        rows.append(dict(
            has_direct=any(v in q["text"] for v in direct),
            direct_left=any(v in clean for v in direct if len(v) > 2),
            qi_count=len(q["qis"]),
            k_raw=engine.k(c, raw_levels, effective=False),
            k_clean=engine.k(c, clean_levels, effective=False)))
    return pd.DataFrame(rows), clean


# Reference population

def reference_population_k(directory: pd.DataFrame, clients: list[Client], levels_by_task: dict,
                           book_sizes=(150,), seed: int = 0):
    """k against the general directory vs. against the firm's own client book (and one advisor's book)."""
    from dataclasses import replace
    general = KEngine(directory)
    idx = [c.person_idx for c in clients]
    book = directory.loc[idx].reset_index(drop=True)
    book_engine = KEngine(book)
    remap = [replace(c, person_idx=i) for i, c in enumerate(clients)]
    rng = np.random.default_rng(seed)
    rows = []
    for task, lv in levels_by_task.items():
        kg = np.array([general.k(c, lv) for c in clients])
        kb = np.array([book_engine.k(c, lv) for c in remap])
        rows.append(dict(task=task, reference="general population directory", pct_k_lt_10=(kg < 10).mean(),
                         median_k=float(np.median(kg))))
        rows.append(dict(task=task, reference=f"firm client book ({len(clients):,})", pct_k_lt_10=(kb < 10).mean(),
                         median_k=float(np.median(kb))))
        for size in book_sizes:
            sub = rng.choice(len(clients), size=size, replace=False)
            e = KEngine(book.loc[sub].reset_index(drop=True))
            ks = np.array([e.k(replace(remap[j], person_idx=i), lv) for i, j in enumerate(sub)])
            rows.append(dict(task=task, reference=f"one advisor's book ({size})", pct_k_lt_10=(ks < 10).mean(),
                             median_k=float(np.median(ks))))
    return pd.DataFrame(rows)


# Policy side channel

def policy_side_channel(clients: list[Client], engine: KEngine, task: str = "income_plan", k_min: int = 10):
    """Does the policy the minimizer picks (or a refusal) reveal that the client is rare?"""
    X, rare, utils = [], [], []
    default = TASKS[task]["default"]
    for c in clients:
        best = choose_policy(task, c, engine, k_min)
        lv = best[0] if best else {f: "drop" for f in FIELDS}
        X.append([rank(f, lv[f]) for f in QI_FIELDS] + [best is None])
        rare.append(engine.k(c, default) < 100)
        utils.append(best[1] if best else 0.0)
    X, y = np.array(X, dtype=float), np.array(rare)
    if y.all() or not y.any():
        auc = float("nan")
    else:
        p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, p))
    # mitigation: one policy for everyone, the most generalized one any client needed
    worst = {f: FIELDS[f].levels[int(X[:, i].max())] if i < len(QI_FIELDS) else None for i, f in enumerate(QI_FIELDS)}
    uniform = dict(default)
    uniform.update({f: v for f, v in worst.items() if v})
    from .tasks import utility
    u_uniform = float(np.mean([utility(task, c, uniform) for c in clients]))
    ku = np.array([engine.k(c, uniform) for c in clients])
    return dict(task=task, rare_share=float(y.mean()), auc_policy_predicts_rare=auc,
                adaptive_utility=float(np.mean(utils)), uniform_policy=uniform, uniform_utility=u_uniform,
                uniform_pct_k_lt_10=float((ku < k_min).mean()))


# Exact amounts

def amount_fingerprints(adv: pd.DataFrame):
    rows = []
    cols = {"investment_portfolio_usd": "portfolio value", "target_amount_usd": "goal target",
            "annual_income_usd": "annual income"}
    for col, label in cols.items():
        v = adv[col].astype(float)
        for name, f in [("exact", lambda x: x.round(2)), ("to $1K", lambda x: (x / 1e3).round()),
                        ("to $10K", lambda x: (x / 1e4).round()), ("to $100K", lambda x: (x / 1e5).round()),
                        ("bucket", lambda x: x.map(lambda a: aum_bucket(a / 1e6)))]:
            g = f(v)
            rows.append(dict(amount=label, precision=name, pct_unique=float((g.map(g.value_counts()) == 1).mean())))
    combo = adv.target_amount_usd.round(-4).astype(str) + "|" + adv.time_horizon_years.astype(str) + "|" + \
        adv.primary_goal.astype(str)
    rows.append(dict(amount="goal target to $10K + horizon + goal", precision="combined",
                     pct_unique=float((combo.map(combo.value_counts()) == 1).mean())))
    return pd.DataFrame(rows)
