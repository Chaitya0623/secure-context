"""Advisor tasks: prompts, a deterministic reference engine, and utility scoring.

The reference engine plays an ideal assistant that uses exactly what is in the
context it receives. Scoring its output against the full-context output gives
a task-quality number that does not depend on LLM randomness. The optional LLM
judge in eval/ replaces this with real model answers when an API key is set.
"""
from __future__ import annotations

from .data import Client
from .policy import FIELDS, TASKS, AUM_MID, at_least, render_context


# Meeting prep: agenda items and the context each one needs

def _meeting_items(c: Client):
    items = [("Liquidity plan", {"liquidity": "bucket"}),
             ("Objective review", {"objective": "exact"}),
             ("Risk and horizon check", {"risk": "exact", "horizon": "exact"}),
             ("Life-stage planning", {"age": "band"}),
             ("Service tier and planning scope", {"aum": "bucket"}),
             ("Household and beneficiaries", {"household": "exact"}),
             ("Portfolio mix", {"holdings": "sectors"}),
             ("Client concerns from notes", {"notes": "summary"})]
    top = next(h for h in c.holdings if h.ticker != "CASH")
    if top.single and top.weight >= 20:
        items.append(("Concentrated position", {"holdings": "anonymized", "notes": "summary"}))
    return items


def _item_score(levels, needs, any_of=False):
    states = []
    for f, need in needs.items():
        lv = levels[f]
        states.append(2 if at_least(f, lv, need) else 1 if lv != "drop" else 0)
    if any_of:  # concentration item: either holdings detail or the notes summary is enough
        return 1.0 if 2 in states else .5 if 1 in states else 0.0
    return 1.0 if min(states) == 2 else .5 if min(states) == 1 else 0.0


def _conc_metrics(weights_items):
    ws = sorted((w for _, w, _ in weights_items), reverse=True)
    largest_single = max((w for _, w, s in weights_items if s), default=0)
    return dict(largest=ws[0], top3=sum(ws[:3]), hhi=sum((w / 100) ** 2 for w in ws),
                flag=largest_single > 20)


def _withdrawal_rate(age: float, risk: str | None) -> float:
    r = min(max(.03 + .001 * (age - 55), .03), .06)
    return r + {"Aggressive": .002, "Conservative": -.003}.get(risk or "", 0)


def utility(task: str, c: Client, levels: dict) -> float:
    if task == "meeting_prep":
        scores = []
        for name, needs in _meeting_items(c):
            scores.append(_item_score(levels, needs, any_of=name == "Concentrated position"))
        return sum(scores) / len(scores)

    if task == "concentration":
        true = _conc_metrics([(h.ticker, h.weight, h.single) for h in c.holdings if h.ticker != "CASH"])
        hl = levels["holdings"]
        items = []
        if hl in ("exact", "anonymized"):
            items += [1, 1, 1, 1]
        elif hl == "sectors":
            groups = {}
            for h in c.holdings:
                if h.ticker != "CASH":
                    groups[h.group] = groups.get(h.group, 0) + h.weight
            est = _conc_metrics([(g, w, g not in ("Broad equity", "Fixed income")) for g, w in groups.items()])
            items += [abs(est["largest"] - true["largest"]) <= 2, abs(est["top3"] - true["top3"]) <= 3,
                      abs(est["hhi"] - true["hhi"]) <= .02, est["flag"] == true["flag"]]
        else:
            items += [0, 0, 0, 0]
        items.append(levels["risk"] != "drop")
        items.append(at_least("liquidity", levels["liquidity"], "bucket"))
        items.append(levels["aum"] != "drop")
        return sum(float(x) for x in items) / len(items)

    if task == "income_plan":
        if levels["aum"] == "drop":
            return 0.0
        age = c.age if levels["age"] == "exact" else (c.age // 10) * 10 + 4.5 if levels["age"] == "band" else 65
        aum = c.aum if levels["aum"] == "exact" else AUM_MID[_bucket(c.aum)]
        risk = c.risk if levels["risk"] != "drop" else None
        est = aum * _withdrawal_rate(age, risk)
        true = c.aum * _withdrawal_rate(c.age, c.risk)
        acc = max(0.0, 1 - abs(est - true) / max(true, 1e-4) / .3)  # $100 floor avoids dividing by zero
        rest = (levels["risk"] != "drop") * .5 + (levels["horizon"] != "drop") * .25 + \
            at_least("liquidity", levels["liquidity"], "bucket") * .25
        return .6 * acc + .4 * rest
    if task in TASKS:  # portfolio_review, transaction, general: share of needs met
        from .policy import coverage
        return coverage(levels, task)[0]
    raise KeyError(task)

CORE_TASKS = ("meeting_prep", "concentration", "income_plan")


def _bucket(m):
    from .policy import aum_bucket
    return aum_bucket(m)


# Prompt and simulated model answer

SYSTEM_PROMPT = ("You are an assistant for a financial advisor. Client identifiers appear as reference tokens "
                 "like <PERSON_1a2b3c>. Copy tokens exactly as written. Never guess values that are not in the "
                 "context; say what is missing instead.")


def build_prompt(task: str, c: Client, levels: dict, session, tokenized_notes: str = "") -> str:
    ctx = render_context(c, levels, tokenized_notes)
    lines = [f"Task: {TASKS[task]['name']}.", f"Client ref: {session.token('PERSON', c.name)}",
             f"Contact ref: {session.token('EMAIL', c.email)}"]
    lines += [f"{k}: {v}" for k, v in ctx.items()]
    return "\n".join(lines)


def simulated_answer(task: str, c: Client, levels: dict, session) -> str:
    """Template answer that uses only what the context contains (offline stand-in for an LLM)."""
    name, email = session.token("PERSON", c.name), session.token("EMAIL", c.email)
    ctx = render_context(c, levels)
    out = []
    if task == "meeting_prep":
        out.append(f"Meeting brief for {name}")
        for i, (item, needs) in enumerate(_meeting_items(c), 1):
            s = _item_score(levels, needs, any_of=item == "Concentrated position")
            if s == 0:
                miss = ", ".join(FIELDS[f].label.lower() for f in needs if levels[f] == "drop")
                out.append(f"{i}. {item}: skipped, context did not include {miss}.")
            else:
                detail = "; ".join(f"{FIELDS[f].label}: {ctx[FIELDS[f].label]}" for f in needs
                                   if FIELDS[f].label in ctx and f != "notes")
                out.append(f"{i}. {item}{' (' + detail + ')' if detail else ''}.")
        out.append(f"Send the agenda to {email} before the meeting.")
    elif task == "concentration":
        out.append(f"Concentration review for {name}")
        if levels["holdings"] in ("exact", "anonymized"):
            m = _conc_metrics([(h.ticker, h.weight, h.single) for h in c.holdings if h.ticker != "CASH"])
            out.append(f"Largest position {m['largest']}%, top three {m['top3']}%, HHI {m['hhi']:.3f}.")
            out.append("Single-stock exposure is above a 20% guideline." if m["flag"] else "No single stock above 20%.")
        elif levels["holdings"] == "sectors":
            out.append(f"Only sector weights were shared ({ctx['Holdings']}); position-level concentration is unknown.")
        else:
            out.append("No holdings were shared, so concentration cannot be assessed.")
        out.append(f"Risk tolerance: {ctx.get('Risk tolerance', 'not provided')}.")
        out.append(f"Liquidity: {ctx.get('Liquidity need', 'not provided')}.")
        out.append(f"Share the trim schedule with {email}.")
    elif task in ("portfolio_review", "transaction", "general"):
        out.append(f"{TASKS[task]['name']} for {name}")
        for label, v in ctx.items():
            out.append(f"{label}: {v}")
        if not ctx:
            out.append("Answered from general product information; no client data was needed.")
        out.append(f"Confirmation goes to {email}.")
    else:
        if levels["aum"] == "drop":
            out.append(f"Income estimate for {name}: assets were not shared, so no estimate is possible.")
        else:
            age = c.age if levels["age"] == "exact" else (c.age // 10) * 10 + 4.5 if levels["age"] == "band" else 65
            aum = c.aum if levels["aum"] == "exact" else AUM_MID[_bucket(c.aum)]
            r = _withdrawal_rate(age, c.risk if levels["risk"] != "drop" else None)
            out.append(f"Income estimate for {name}: about ${aum * r * 1000:,.0f}K per year "
                       f"({r * 100:.1f}% of ${aum:.1f}M{', assumed from bucket' if levels['aum'] != 'exact' else ''}).")
            out.append(f"Horizon: {ctx.get('Time horizon', 'not provided')}. Liquidity: {ctx.get('Liquidity need', 'not provided')}.")
        out.append(f"Send the projection to {email}.")
    return "\n".join(out)
