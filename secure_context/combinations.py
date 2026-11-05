"""Which fields are harmless alone but identifying together.

The crowd-size engine answers "is this one policy safe for this client?". This module
maps the space around it, so policy authors can see which combinations to avoid:

ATTRS              quasi-identifiers at the coarse level task policies send, and at the fine level
combination_k      k for every client under every subset of attributes (one groupby per subset)
combination_risk   share of clients with k < k_min / unique, per subset
pair_synergy       joint risk of each pair vs. the riskier field alone (the "network" edges)
minimal_sets       per client, the smallest attribute sets that already push k below k_min
inference_links    how much knowing one attribute reveals about another (uncertainty coefficient)
crowd_homogeneity  k is large enough, but the crowd shares the client's hidden value
                   (the gap l-diversity / t-closeness address)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .policy import aum_bucket, conc_bucket


def _sectors(p):
    return p.largest_group.astype(str) + "|" + p.largest_group_weight.map(conc_bucket).astype(str)


def _anonymized(p):
    return (p.top_group.astype(str) + "|" + p.top_single.astype(str) + "|" + (p.top_weight // 5).astype(str)
            + "|" + _sectors(p))


def _income_band(p):
    return pd.cut(p.income, [-1, 50e3, 100e3, 200e3, 500e3, 1e12], labels=False).astype(str)


# name -> (label, coarse key, fine key). Coarse matches what the task policies send.
ATTRS = {
    "age": ("Age", lambda p: (p.age // 10).astype(str), lambda p: p.age.astype(str)),
    "location": ("Location", lambda p: p.region.astype(str), lambda p: p.state + "|" + p.city),
    "occupation": ("Occupation", lambda p: p.occ_cat.astype(str), lambda p: p.occupation.astype(str)),
    "aum": ("Assets", lambda p: p.aum.map(aum_bucket), lambda p: p.aum.map(aum_bucket) + "|" + p.aum.round(1).astype(str)),
    "holdings": ("Holdings", _sectors, _anonymized),
    "marital": ("Marital status", lambda p: p.marital.astype(str), lambda p: p.marital.astype(str)),
}

# Attributes an attacker would like to learn when the policy withholds them:
# name -> (attribute that would reveal it if sent, key)
HIDDEN = {
    "state": ("location", lambda p: p.state.astype(str)),
    "occupation group": ("occupation", lambda p: p.occ_cat.astype(str)),
    "income band": (None, _income_band),
    "asset bucket": ("aum", lambda p: p.aum.map(aum_bucket)),
    "marital status": ("marital", lambda p: p.marital.astype(str)),
}


def _code(cols: list[pd.Series]) -> np.ndarray:
    if len(cols) == 1:
        return pd.factorize(cols[0])[0]
    df = pd.concat([c.reset_index(drop=True) for c in cols], axis=1, ignore_index=True)
    return df.groupby(list(df.columns), sort=False).ngroup().values


def keys(pop: pd.DataFrame, level: str = "coarse", attrs: dict = ATTRS) -> dict:
    i = 1 if level == "coarse" else 2
    return {a: spec[i](pop).reset_index(drop=True) for a, spec in attrs.items()}


def combination_k(pop: pd.DataFrame, pidx, level: str = "coarse", attrs: dict = ATTRS) -> dict:
    """{subset (tuple of attribute names): k per client}, for every non-empty subset."""
    kc = keys(pop, level, attrs)
    pidx = np.asarray(pidx)
    out = {}
    for r in range(1, len(attrs) + 1):
        for sub in itertools.combinations(attrs, r):
            code = _code([kc[a] for a in sub])
            out[sub] = np.bincount(code)[code[pidx]]
    return out


def combination_risk(ks: dict, k_min: int = 10) -> pd.DataFrame:
    rows = [dict(fields=" + ".join(sub), size=len(sub), pct_k_below_min=100 * float((k < k_min).mean()),
                 pct_unique=100 * float((k == 1).mean()), median_k=float(np.median(k))) for sub, k in ks.items()]
    return pd.DataFrame(rows).sort_values(["size", "pct_k_below_min"], ascending=[True, False]).reset_index(drop=True)


def pair_synergy(ks: dict, k_min: int = 10) -> pd.DataFrame:
    """Edges of the combination network: how much riskier a pair is than its riskier member."""
    single = {sub[0]: 100 * float((k < k_min).mean()) for sub, k in ks.items() if len(sub) == 1}
    rows = []
    for sub, k in ks.items():
        if len(sub) != 2:
            continue
        joint = 100 * float((k < k_min).mean())
        a, b = sub
        rows.append(dict(a=a, b=b, risk_a=single[a], risk_b=single[b], joint=joint,
                         synergy=joint - max(single[a], single[b])))
    return pd.DataFrame(rows).sort_values("joint", ascending=False).reset_index(drop=True)


def minimal_sets(ks: dict, k_min: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per client, the smallest attribute sets with k < k_min (no risky proper subset).

    Returns (per-client sizes, how often each minimal set occurs)."""
    subs = sorted(ks, key=len)
    n = len(next(iter(ks.values())))
    risky = {sub: ks[sub] < k_min for sub in subs}
    found = [[] for _ in range(n)]
    for sub in subs:
        for i in np.flatnonzero(risky[sub]):
            if not any(set(m) <= set(sub) for m in found[i]):
                found[i].append(sub)
    per = pd.DataFrame(dict(smallest=[min((len(m) for m in f), default=0) for f in found],
                            n_minimal=[len(f) for f in found]))
    counts = pd.Series([" + ".join(m) for f in found for m in f]).value_counts()
    freq = pd.DataFrame(dict(fields=counts.index, pct_clients=100 * counts.values / n,
                             size=[s.count("+") + 1 for s in counts.index]))
    return per, freq


def _entropy(code: np.ndarray) -> float:
    p = np.bincount(code) / len(code)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def inference_links(pop: pd.DataFrame, attrs: dict | None = None) -> pd.DataFrame:
    """U(b | a) = I(a; b) / H(b): share of b's uncertainty removed by knowing a (0 none, 1 fully)."""
    attrs = attrs or {**{k: v[1] for k, v in ATTRS.items()}, **{f"hidden: {k}": v[1] for k, v in HIDDEN.items()}}
    codes = {a: pd.factorize(fn(pop).reset_index(drop=True))[0] for a, fn in attrs.items()}
    h = {a: _entropy(c) for a, c in codes.items()}
    rows = []
    for a, b in itertools.permutations(codes, 2):
        if h[b] == 0:
            continue
        mi = h[a] + h[b] - _entropy(_code([pd.Series(codes[a]), pd.Series(codes[b])]))
        rows.append(dict(known=a, inferred=b, uncertainty_removed=mi / h[b]))
    return pd.DataFrame(rows).sort_values("uncertainty_removed", ascending=False).reset_index(drop=True)


def crowd_homogeneity(pop: pd.DataFrame, pidx, sent: list[str], level: str = "coarse", k_min: int = 10,
                      confident: float = 0.8) -> pd.DataFrame:
    """For clients whose crowd is large enough (k >= k_min), how often does the crowd still
    reveal a withheld attribute? Attacker guesses the crowd's most common value."""
    kc = keys(pop, level)
    pidx = np.asarray(pidx)
    sent_code = _code([kc[a] for a in sent])
    k = np.bincount(sent_code)[sent_code[pidx]]
    safe = k >= k_min
    rows = []
    for name, (revealed_by, fn) in HIDDEN.items():
        if revealed_by in sent:
            continue
        hid = pd.factorize(fn(pop).reset_index(drop=True))[0]
        joint = sent_code.astype(np.int64) * (hid.max() + 1) + hid
        jc = pd.Series(joint).value_counts()
        top = pd.DataFrame(dict(s=sent_code, h=hid)).groupby(["s", "h"]).size().groupby(level=0).max()
        share_true = jc.reindex(joint[pidx]).values / k          # posterior of the client's real value
        share_top = top.reindex(sent_code[pidx]).values / k      # attacker's confidence in their guess
        guess_right = share_true == share_top
        base = (hid[pidx] == np.bincount(hid).argmax())[safe].mean()  # always guess the directory's most common value
        rows.append(dict(hidden=name, clients_k_ok=int(safe.sum()),
                         pct_disclosed=100 * float((safe & guess_right & (share_true >= confident)).sum() / max(safe.sum(), 1)),
                         attacker_accuracy=100 * float(guess_right[safe].mean()), majority_baseline=100 * float(base)))
    return pd.DataFrame(rows)


def book_frame(clients) -> pd.DataFrame:
    """The firm's own client book as a population, with fields only the firm knows (dependents)."""
    from .data import holdings_summary
    rows = []
    for c in clients:
        s = holdings_summary(c.holdings)
        mar, dep = [x.strip() for x in c.household.split(",")]
        rows.append(dict(age=c.age, region=c.state and _region(c.state), state=c.state, city=c.city,
                         occ_cat=c.occ_cat, occupation=c.occupation, aum=c.aum, income=c.extra.get("income", 0),
                         marital=mar, dependents=dep.split()[0], **s))
    return pd.DataFrame(rows)


def _region(state):
    from .data import REGION
    return REGION.get(state, "Other")
