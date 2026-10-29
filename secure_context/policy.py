"""Context fields, generalization levels, and task policies.

Each field has levels ordered from most to least detailed. A task policy picks
one level per field. "Quasi-identifier" (qi=True) fields are the ones an
outsider could plausibly know and use for linkage.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Client, REGION


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    levels: tuple
    qi: bool = False


FIELDS = {f.key: f for f in [
    Field("age", "Age", ("exact", "band", "drop"), qi=True),
    Field("location", "Location", ("city", "state", "region", "drop"), qi=True),
    Field("occupation", "Occupation", ("exact", "category", "drop"), qi=True),
    Field("aum", "Assets", ("exact", "bucket", "drop"), qi=True),
    Field("holdings", "Holdings", ("exact", "anonymized", "sectors", "drop"), qi=True),
    Field("household", "Household", ("exact", "drop")),
    Field("objective", "Objective", ("exact", "drop")),
    Field("risk", "Risk tolerance", ("exact", "drop")),
    Field("horizon", "Time horizon", ("exact", "drop")),
    Field("liquidity", "Liquidity need", ("exact", "bucket", "drop")),
    Field("notes", "Advisor notes", ("tokenized", "summary", "drop")),
]}
QI_FIELDS = [k for k, f in FIELDS.items() if f.qi]


def rank(field: str, level: str) -> int:
    return FIELDS[field].levels.index(level)


def at_least(field: str, level: str, need: str) -> bool:
    """True when `level` is as detailed as `need` or more."""
    return rank(field, level) <= rank(field, need)


def more_detailed(field: str, a: str, b: str) -> str:
    return a if rank(field, a) <= rank(field, b) else b


ALL_EXACT = {k: f.levels[0] for k, f in FIELDS.items()}
ALL_DROP = {k: "drop" for k in FIELDS}

TASKS = {
    "meeting_prep": dict(
        name="Prepare meeting brief",
        needs=dict(objective="exact", risk="exact", horizon="exact", liquidity="bucket", age="band",
                   aum="bucket", holdings="sectors", household="exact", notes="summary"),
        default=dict(age="band", location="drop", occupation="drop", aum="bucket", holdings="sectors",
                     household="exact", objective="exact", risk="exact", horizon="exact", liquidity="bucket",
                     notes="summary")),
    "concentration": dict(
        name="Portfolio concentration check",
        needs=dict(holdings="anonymized", aum="bucket", risk="exact", liquidity="bucket"),
        default=dict(age="drop", location="drop", occupation="drop", aum="bucket", holdings="anonymized",
                     household="drop", objective="drop", risk="exact", horizon="drop", liquidity="bucket",
                     notes="drop")),
    "income_plan": dict(
        name="Retirement income estimate",
        needs=dict(age="exact", aum="exact", risk="exact", horizon="exact", liquidity="bucket"),
        default=dict(age="exact", location="drop", occupation="drop", aum="exact", holdings="drop",
                     household="drop", objective="exact", risk="exact", horizon="exact", liquidity="bucket",
                     notes="drop")),
}


# Generalization helpers (shared with the population-side key builder)

def age_band(a: int) -> str:
    lo = (a // 10) * 10
    return f"{lo}-{lo + 9}"


AUM_EDGES = [(1, "Under $1M"), (5, "$1M-$5M"), (10, "$5M-$10M"), (25, "$10M-$25M"), (1e9, "$25M+")]


def aum_bucket(m: float) -> str:
    for edge, lab in AUM_EDGES:
        if m < edge:
            return lab
    return AUM_EDGES[-1][1]


AUM_MID = {"Under $1M": .6, "$1M-$5M": 2.6, "$5M-$10M": 7.2, "$10M-$25M": 15.0, "$25M+": 40.0}


def conc_bucket(w: float) -> int:
    return 0 if w < 20 else 1 if w < 35 else 2 if w < 50 else 3


CONC_LABEL = ["under 20%", "20-35%", "35-50%", "over 50%"]


def liquidity_bucket(amount: int, text: str) -> str:
    amt = "under $50K" if amount < 50_000 else "$50K-$150K" if amount < 150_000 else \
        "$150K-$500K" if amount < 500_000 else "$500K+"
    months = int(text.split("within ")[1].split(" ")[0]) if "within " in text else 12
    when = "within a year" if months <= 12 else "in 1 to 2 years"
    return f"{amt} {when}"


def holdings_text(c: Client, level: str) -> str:
    hs = c.holdings
    if level == "exact":
        return "; ".join("Cash {}%".format(h.weight) if h.ticker == "CASH" else f"{h.ticker} {h.weight}%" for h in hs)
    if level == "anonymized":
        out, i = [], 0
        for h in hs:
            if h.ticker == "CASH":
                out.append(f"Cash {h.weight}%")
            else:
                out.append(f"Position {chr(65 + i)} ({'single stock' if h.single else 'fund'}, {h.group}) {h.weight}%")
                i += 1
        return "; ".join(out)
    if level == "sectors":
        g = {}
        for h in hs:
            g[h.group] = g.get(h.group, 0) + h.weight
        return "; ".join(f"{k} {v}%" for k, v in sorted(g.items(), key=lambda kv: -kv[1]))
    return ""


def value(c: Client, field: str, level: str, tokenized_notes: str = "") -> str | None:
    if level == "drop":
        return None
    if field == "age":
        return str(c.age) if level == "exact" else age_band(c.age)
    if field == "location":
        return {"city": f"{c.city}, {c.state}", "state": c.state, "region": f"{REGION[c.state]} US"}[level]
    if field == "occupation":
        return c.occupation if level == "exact" else c.occ_cat
    if field == "aum":
        return f"${c.aum:.1f}M" if level == "exact" else aum_bucket(c.aum)
    if field == "holdings":
        return holdings_text(c, level)
    if field == "liquidity":
        return c.liquidity if level == "exact" else liquidity_bucket(c.liquidity_amount, c.liquidity)
    if field == "notes":
        return tokenized_notes if level == "tokenized" else c.summary
    return getattr(c, field)


def render_context(c: Client, levels: dict, tokenized_notes: str = "") -> dict:
    out = {}
    for k in FIELDS:
        v = value(c, k, levels[k], tokenized_notes)
        if v:
            out[FIELDS[k].label] = v
    return out


def coverage(levels: dict, task: str) -> tuple[float, list]:
    """Share of the task's needs met at the required level (partial credit 0.5 if present but coarser)."""
    needs = TASKS[task]["needs"]
    score, missing = 0.0, []
    for f, need in needs.items():
        lv = levels[f]
        if at_least(f, lv, need):
            score += 1
        elif lv != "drop":
            score += .5
            missing.append(f)
        else:
            missing.append(f)
    return score / len(needs), missing
