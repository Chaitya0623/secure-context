"""Attacks used in the evaluation.

key_attacks          lookups, joins, cross-session linkage and hash brute force against the split storage
attribute_inference  a model trained on public data guesses attributes the context left out
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Client
from .policy import aum_bucket
from .tokens import Session
from .vault import AccessDenied, SplitStorage


def key_attacks(storage: SplitStorage, clients: list[Client], ctx_ids: list[str], n_check: int = 200):
    rows = []
    sess_a, sess_b = Session(), Session()
    sample = list(zip(clients, ctx_ids))[:n_check]

    # 1. AI-log token used as a lookup key
    hits = 0
    for c, ctx in sample:
        tok = sess_a.token("PERSON", c.name)
        hits += len(storage.store.lookup_any(tok, role="context_service"))
    rows.append(("Look up context store by AI-log token", f"{hits} rows found in {len(sample)} attempts",
                 "blocked" if hits == 0 else "exposed"))

    # 2. Search the context store dump for direct identifiers
    hits = 0
    for c, ctx in sample:
        for v in (c.name, c.ssn, c.email, c.account, c.phone):
            hits += len(storage.store.lookup_any(v, role="context_service"))
    rows.append(("Search context dump for name, SSN, email, account, phone",
                 f"{hits} matches across {len(sample) * 5} values", "blocked" if hits == 0 else "exposed"))

    # 3. Join context to identity without the vault
    cols = storage.store.columns()
    ident_cols = [c for c in cols if c in ("name", "ssn", "email", "vault_id", "account")]
    rows.append(("Join client_context to identity on a shared key",
                 f"context columns: {', '.join(cols)}; shared identity keys: {len(ident_cols)}",
                 "blocked" if not ident_cols else "exposed"))

    # 4. Read the vault with the LLM service role
    try:
        storage.vault.identity(ctx_ids[0], role="llm_service")
        res = ("Read identity vault as llm_service", "read succeeded", "exposed")
    except AccessDenied as e:
        res = ("Read identity vault as llm_service", str(e), "blocked")
    rows.append(res)

    # 5. Link two sessions by token
    same = sum(sess_a.token("PERSON", c.name) == sess_b.token("PERSON", c.name) for c, _ in sample)
    rows.append(("Link the same client across two sessions by token", f"{same} of {len(sample)} tokens identical",
                 "blocked" if same == 0 else "exposed"))

    # 6. Brute force a plain SHA-256(SSN) key versus an HMAC token
    target = clients[0].ssn.replace("-", "")
    plain = hashlib.sha256(target.encode()).hexdigest()
    prefix = target[:5]
    t0 = time.perf_counter()
    found = None
    for i in range(10_000):
        cand = f"{prefix}{i:04d}"
        if hashlib.sha256(cand.encode()).hexdigest() == plain:
            found = cand
    dt = time.perf_counter() - t0
    per_hash = dt / 10_000
    rows.append(("Brute-force a plain SHA-256(SSN) key",
                 f"recovered={'yes' if found == target else 'no'}; full 10^9 SSN space in about "
                 f"{per_hash * 1e9 / 60:.0f} CPU-minutes (much less on a GPU)", "exposed"))
    key = os.urandom(32)
    tok = hmac.new(key, f"SSN|{clients[0].ssn}".encode(), hashlib.sha256).hexdigest()[:6]
    guess_key = os.urandom(32)
    found = sum(hmac.new(guess_key, f"SSN|{prefix[:3]}-{prefix[3:5]}-{i:04d}".encode(), hashlib.sha256)
                .hexdigest()[:6] == tok for i in range(10_000))
    rows.append(("Brute-force an HMAC token without the key", f"{found} chance collisions in 10,000 guesses "
                 "(with a wrong key nothing is learned; the key space is 2^256)", "blocked"))
    return pd.DataFrame(rows, columns=["attack", "detail", "result"])


# Attribute inference

TARGETS = {
    "occupation category": ("occupation", lambda p: p.occ_cat),
    "region": ("location", lambda p: p.region),
    "age band": ("age", lambda p: (p.age // 10).astype(str)),
    "asset bucket": ("aum", lambda p: p.aum.map(aum_bucket)),
}


def _features(p: pd.DataFrame, levels: dict, profile_fields: bool = True) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    if levels["age"] == "exact":
        f["age"] = p.age
    elif levels["age"] == "band":
        f["age_band"] = p.age // 10
    if levels["aum"] == "exact":
        f["aum"] = p.aum
    elif levels["aum"] == "bucket":
        f["aum_b"] = p.aum.map(aum_bucket)
    if levels["location"] == "city":
        f["city"] = p.state + "|" + p.city
    elif levels["location"] == "state":
        f["state"] = p.state
    elif levels["location"] == "region":
        f["region"] = p.region
    if levels["occupation"] == "exact":
        f["occ"] = p.occupation
    elif levels["occupation"] == "category":
        f["occ_cat"] = p.occ_cat
    if levels["holdings"] == "exact":
        f["ticker"] = p.top_ticker
        f["tw"] = p.top_weight
    elif levels["holdings"] == "anonymized":
        f["tgroup"] = p.top_group
        f["tsingle"] = p.top_single.astype(int)
        f["tw"] = p.top_weight
    elif levels["holdings"] == "sectors":
        f["lgroup"] = p.largest_group
        f["lgw"] = p.largest_group_weight
    for k in ("risk", "objective", "horizon") if profile_fields else ():
        if levels[k] != "drop":
            f[k] = p[k]
    if f.shape[1] == 0:
        f["const"] = 0
    return f


def attribute_inference(pop: pd.DataFrame, clients: list[Client], levels: dict, n_train: int = 20_000, seed: int = 0,
                        profile_fields: bool = True):
    """For each attribute the policy hides, train on public data and guess it for the clients.

    profile_fields=False drops risk, objective and horizon from the attacker's features; use it
    when the directory does not hold real values for them (the Census directory does not)."""
    client_idx = [c.person_idx for c in clients]
    rest = pop.drop(index=client_idx)
    train = rest.sample(n=min(n_train, len(rest)), random_state=seed)
    test = pop.loc[client_idx]
    out = []
    for target, (field, fn) in TARGETS.items():
        if levels[field] != "drop":
            continue  # only attack attributes the policy withheld
        xtr = _features(train, levels, profile_fields)
        xte = _features(test, levels, profile_fields)
        allx = pd.get_dummies(pd.concat([xtr, xte]), dtype=float)
        xtr, xte = allx.iloc[:len(xtr)], allx.iloc[len(xtr):]
        ytr, yte = fn(train).astype(str), fn(test).astype(str)
        model = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(xtr, ytr)
        acc = float((model.predict(xte) == yte.values).mean())
        base = float((yte == ytr.value_counts().idxmax()).mean())
        out.append(dict(target=target, attacker_accuracy=acc, majority_baseline=base, lift=acc - base,
                        n_classes=int(ytr.nunique())))
    return pd.DataFrame(out)
