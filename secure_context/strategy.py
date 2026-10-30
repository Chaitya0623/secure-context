"""A planted proprietary strategy and an extraction attack against it.

The firm's model-portfolio rule (artificial, for testing) maps a client profile
to an allocation. Its secrets are the glide-path knee and slope, the asset level
that unlocks alternatives, and the concentration level that triggers a collar.
An attacker queries the assistant with chosen profiles and fits a surrogate
model; fidelity of that surrogate is the extraction score.

Defenses compared:
  none          exact allocation returned
  coarse        allocation rounded to 10 points; alternatives and overlay hidden
  noise         Laplace noise on each answer, fresh per query
  noise_cached  same noise, but the same profile always gets the same answer
  book_bound    exact answers, but only for clients in the caller's own book
  book_bound+coarse   both of the above

Against the two noise defenses the attacker asks each profile 5 times and
averages; caching makes that averaging useless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

SECRET = dict(knee=52, slope=1.1, alt_aum=5.0, collar_conc=25)
DEFENSES = ("none", "coarse", "noise", "noise_cached", "book_bound", "book_bound+coarse")


def proprietary(age, risk, aum, income, conc):
    """risk: 0 conservative, 1 moderate, 2 aggressive. aum in $M. conc = largest single-stock %."""
    age, risk, aum, income, conc = map(np.asarray, (age, risk, aum, income, conc))
    equity = np.choose(risk.astype(int), [45, 65, 85]) - np.maximum(0, age - SECRET["knee"]) * SECRET["slope"]
    alt = np.where(aum >= SECRET["alt_aum"], 7, 0)
    equity = equity - alt * .5 - income * 5
    equity = np.clip(equity, 20, 90)
    collar = (conc >= SECRET["collar_conc"]).astype(int)
    return equity, alt, collar


def random_profiles(n, rng):
    return dict(age=rng.integers(25, 90, n), risk=rng.integers(0, 3, n), aum=np.exp(rng.uniform(np.log(.2), np.log(60), n)),
                income=rng.integers(0, 2, n), conc=rng.uniform(0, 60, n))


def book_profiles(n_book, rng, pop: pd.DataFrame):
    rows = pop.sample(n=n_book, random_state=int(rng.integers(1e9)))
    risk = rows.risk.map({"Conservative": 0, "Moderate": 1, "Aggressive": 2}).values
    income = rows.objective.str.contains("ncome").astype(int).values
    conc = np.where(rows.top_single.values, rows.top_weight.values, 0)
    return dict(age=rows.age.values, risk=risk, aum=rows.aum.values, income=income, conc=conc)


def answer(defense, prof, rng, noise_scale=4.0, cache=None):
    eq, alt, collar = proprietary(**prof)
    eq = eq.astype(float)
    if defense == "coarse":
        return np.round(eq / 10) * 10, np.full_like(alt, -1), np.full_like(collar, -1)
    if defense == "noise":
        return eq + rng.laplace(0, noise_scale, len(eq)), alt, collar
    if defense == "noise_cached":
        keys = list(zip(prof["age"], prof["risk"], np.round(prof["aum"], 2), prof["income"], np.round(prof["conc"], 1)))
        out = []
        for k, e in zip(keys, eq):
            if k not in cache:
                cache[k] = e + rng.laplace(0, noise_scale)
            out.append(cache[k])
        return np.array(out), alt, collar
    return eq, alt, collar


def _X(p):
    # Public SCF records can have zero financial assets. Use a $1 floor (AUM
    # is in millions) for the log feature, without changing the strategy inputs.
    log_aum = np.log(np.maximum(np.asarray(p["aum"], dtype=float), 1e-6))
    return np.column_stack([p["age"], p["risk"], log_aum, p["income"], p["conc"]])


def extraction_curve(pop, query_counts=(25, 50, 100, 200, 500, 1000, 2000), n_test=3000, book_size=150, seed=0):
    rng = np.random.default_rng(seed)
    test = random_profiles(n_test, rng)
    t_eq, t_alt, t_col = proprietary(**test)
    rows = []
    reps = 5  # against noisy answers the attacker repeats each profile and averages
    for defense in DEFENSES:
        for q in query_counts:
            rr = np.random.default_rng(seed + q)
            if defense.startswith("book_bound"):
                book = book_profiles(book_size, rr, pop)
                idx = rr.integers(0, book_size, q)  # repeat queries can only revisit the same clients
                prof = {k: v[idx] for k, v in book.items()}
                eq, alt, col = answer("coarse" if "coarse" in defense else "none", prof, rr, cache={})
            elif defense in ("noise", "noise_cached"):
                base = random_profiles(max(q // reps, 2), rr)
                prof = {k: np.repeat(v, reps) for k, v in base.items()}
                eq, alt, col = answer(defense, prof, rr, cache={})
                eq = eq.reshape(-1, reps).mean(axis=1)
                alt, col = alt[::reps], col[::reps]
                prof = base
            else:
                prof = random_profiles(q, rr)
                eq, alt, col = answer(defense, prof, rr, cache={})
            reg = GradientBoostingRegressor(random_state=0).fit(_X(prof), eq)
            mae = float(np.abs(reg.predict(_X(test)) - t_eq).mean())
            if (alt >= 0).all() and len(set(alt)) > 1:
                alt_acc = float((DecisionTreeClassifier(max_depth=4).fit(_X(prof), alt).predict(_X(test)) == t_alt).mean())
            else:
                share = (t_alt > 0).mean()
                alt_acc = float(max(share, 1 - share))
            if (col >= 0).all() and len(set(col)) > 1:
                col_acc = float((DecisionTreeClassifier(max_depth=4).fit(_X(prof), col).predict(_X(test)) == t_col).mean())
            else:
                col_acc = float(max(t_col.mean(), 1 - t_col.mean()))
            rows.append(dict(defense=defense, queries=q, equity_mae=mae, alt_rule_acc=alt_acc, collar_rule_acc=col_acc))
    return pd.DataFrame(rows)


def honest_user_error(n=2000, seed=1):
    """How much each defense degrades answers for a legitimate advisor (equity MAE, points)."""
    rng = np.random.default_rng(seed)
    prof = random_profiles(n, rng)
    true, _, _ = proprietary(**prof)
    out = {}
    for d in DEFENSES:
        mode = "coarse" if "coarse" in d else "none" if d.startswith("book_bound") else d
        eq, _, _ = answer(mode, prof, rng, cache={})
        out[d] = float(np.abs(eq - true).mean())
    return out
