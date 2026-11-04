"""Counterfactual necessity test: which client fields does a task's answer actually depend on?

The task policies in policy.py declare what each task needs. This module measures it.
For each client and task the model answers once with the full (tokenized) context, then
again with exactly one field changed and everything else held fixed:

  level   the field sent at a coarser level (age band instead of age, ..., drop)
  swap    the field's value replaced by another client's value, same level
  null    nothing changed except the session tokens; any "change" here is noise

A comparator decides whether a variant's answer would lead the advisor to the same
actions as the baseline answer. A field's measured need for a task is the coarsest
level whose answers stay the same for all but `tolerance` (plus the null-control rate)
of clients. "drop" means the task does not need the field at all.

Swap sensitivity separates two kinds of dependence: an answer that changes when the
field disappears but not when its value changes depends on the field's presence only
(for example, an agenda item that exists whenever risk tolerance is known).

Comparators
  echo_masked  deterministic. Masks reference tokens and any context value the answer
               merely restates, then compares text. Used for the offline simulator,
               whose answers restate their inputs verbatim.
  judge        an LLM decides whether an advisor would act differently. Used for real models.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

import pandas as pd

from .data import Client
from .policy import ALL_EXACT, FIELDS, TASKS, render_context
from .tasks import SYSTEM_PROMPT, build_prompt
from .tokens import LOOSE_TOKEN_RE, Session

# Full context with identifiers tokenized. Notes go as the summary, since raw notes
# carry tokens whose detection is measured elsewhere.
BASELINE = {**ALL_EXACT, "notes": "summary"}

# Client attributes that together make up each field's value.
FIELD_ATTRS = {
    "age": ("age",), "location": ("city", "state"), "occupation": ("occupation", "occ_cat"), "aum": ("aum",),
    "holdings": ("holdings",), "household": ("household",), "objective": ("objective",), "risk": ("risk",),
    "horizon": ("horizon",), "liquidity": ("liquidity", "liquidity_amount"), "notes": ("summary",),
}


@dataclass
class Variant:
    kind: str          # baseline, null, level, swap
    field: str
    level: str
    client: Client
    levels: dict
    session: Session


def coarser_levels(field: str, start: str) -> list:
    lv = FIELDS[field].levels
    return list(lv[lv.index(start) + 1:])


def swap_field(c: Client, field: str, pool: list, rng: random.Random) -> Optional[Client]:
    """The same client with one field's value taken from another client in the book."""
    def key(x):
        return tuple(repr(getattr(x, a)) for a in FIELD_ATTRS[field])
    options = [d for d in pool if key(d) != key(c)]
    if not options:
        return None
    donor = rng.choice(options)
    return replace(c, **{a: getattr(donor, a) for a in FIELD_ATTRS[field]})


def variants(c: Client, pool: list, rng: random.Random, session_for: Callable[[str], Session],
             fields: Optional[Iterable[str]] = None) -> list:
    base = session_for("base")
    out = [Variant("baseline", "", "", c, dict(BASELINE), base),
           Variant("null", "", "", c, dict(BASELINE), session_for("null"))]
    for f in fields or FIELDS:
        for lv in coarser_levels(f, BASELINE[f]):
            out.append(Variant("level", f, lv, c, {**BASELINE, f: lv}, base))
        donor = swap_field(c, f, pool, rng)
        if donor is not None:
            out.append(Variant("swap", f, BASELINE[f], donor, dict(BASELINE), base))
    return out


def ask(llm, task: str, v: Variant) -> str:
    prompt = build_prompt(task, v.client, v.levels, v.session)
    return llm.complete(SYSTEM_PROMPT, prompt, task=task, client=v.client, levels=v.levels, session=v.session)


# Comparators

def _mask(text: str, values: Iterable[str]) -> str:
    text = LOOSE_TOKEN_RE.sub("<REF>", text)
    for v in sorted({v for v in values if v}, key=len, reverse=True):
        # whole values only: "47" must not match inside "$1,470K" or "4.7%"
        text = re.sub(r"(?<![\w$])(?<!\d[.,])" + re.escape(v) + r"(?![\w%])(?![.,]\d)", "<V>", text)
    return " ".join(text.split())


def echo_masked(base_answer: str, base: Variant, answer: str, v: Variant) -> bool:
    """True when the answers differ once restated inputs and tokens are masked."""
    a = _mask(base_answer, render_context(base.client, base.levels).values())
    b = _mask(answer, render_context(v.client, v.levels).values())
    return a != b


JUDGE_PROMPT = """Two answers to the same financial-advisor task follow. Answer A was written with the full
client file. Answer B was written with a changed file. Ignore wording, ordering, formatting and reference
tokens such as <PERSON_1a2b3c>.

Task: {task}

Answer A:
{a}

Answer B:
{b}

Would an advisor reading B take materially different actions, give different advice, or reach different
numbers or conclusions than an advisor reading A? Reply with exactly one word: SAME or DIFFERENT."""


class Judge:
    """LLM comparator. Returns True (changed), False (same) or None (unparseable)."""

    def __init__(self, llm):
        self.llm = llm
        self.name = f"judge:{llm.name}"

    def __call__(self, base_answer: str, base: Variant, answer: str, v: Variant, task: str = "") -> Optional[bool]:
        if " ".join(base_answer.split()) == " ".join(answer.split()):
            return False
        reply = self.llm.complete("You compare answers strictly and consistently.",
                                  JUDGE_PROMPT.format(task=TASKS[task]["name"] if task else "", a=base_answer,
                                                      b=answer)).upper()
        if "DIFFERENT" in reply:
            return True
        if "SAME" in reply:
            return False
        return None


# Running and summarizing

def run(llm, clients: list, tasks: Iterable[str], comparator=None, pool: Optional[list] = None,
        fields: Optional[Iterable[str]] = None, seed: int = 0,
        session_factory: Optional[Callable[[str], Session]] = None, progress: Optional[Callable] = None,
        keep_answers: bool = False) -> pd.DataFrame:
    """One row per client, task and variant: did the answer change against the baseline?"""
    pool = pool or clients
    session_factory = session_factory or (lambda name: Session())
    tasks = list(tasks)
    rows = []
    for i, c in enumerate(clients):
        for task in tasks:
            rng = random.Random(f"{seed}:{c.client_id}:{task}")
            vs = variants(c, pool, rng, lambda name: session_factory(f"cf:{name}:{c.client_id}:{task}"), fields)
            base = vs[0]
            base_answer = ask(llm, task, base)
            for v in vs[1:]:
                ans = ask(llm, task, v)
                if isinstance(comparator, Judge):
                    changed = comparator(base_answer, base, ans, v, task)
                else:
                    changed = (comparator or echo_masked)(base_answer, base, ans, v)
                row = dict(client=c.client_id, task=task, kind=v.kind, field=v.field, level=v.level, changed=changed)
                if keep_answers:
                    row.update(baseline_answer=base_answer, answer=ans)
                rows.append(row)
        if progress:
            progress(i + 1, len(clients))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Change rate per task, kind, field and level. Unparseable judge replies are excluded and counted."""
    r = raw.copy()
    r["unparsed"] = r.changed.isna()
    r["changed"] = r.changed.astype("float")
    return (r.groupby(["task", "kind", "field", "level"], sort=False)
             .agg(change_rate=("changed", "mean"), n=("changed", "count"), unparsed=("unparsed", "sum"))
             .reset_index())


def measured_needs(raw: pd.DataFrame, tolerance: float = 0.10) -> pd.DataFrame:
    """Coarsest sufficient level per task and field, next to the hand-coded need in policy.py."""
    s = summarize(raw)
    out = []
    for task in s.task.unique():
        t = s[s.task == task]
        null = t[t.kind == "null"].change_rate
        null_rate = float(null.iloc[0]) if len(null) and pd.notna(null.iloc[0]) else 0.0
        tol = tolerance + null_rate
        for f in t[t.kind == "level"].field.unique():
            rates = t[(t.kind == "level") & (t.field == f)].set_index("level").change_rate
            need = BASELINE[f]
            for lv in coarser_levels(f, BASELINE[f]):
                if lv not in rates or pd.isna(rates[lv]) or rates[lv] > tol:
                    break
                need = lv
            swap = t[(t.kind == "swap") & (t.field == f)].change_rate
            hand = TASKS[task]["needs"].get(f, "drop")
            out.append(dict(task=task, field=f, measured_need=need, hand_coded_need=hand, agree=need == hand,
                            drop_change_rate=rates.get("drop"), swap_change_rate=float(swap.iloc[0]) if len(swap) else None,
                            null_rate=null_rate, tolerance_used=tol))
    return pd.DataFrame(out)


def measured_policy(needs: pd.DataFrame, task: str) -> dict:
    """Levels that send each field at its measured need and drop the rest."""
    lv = {f: "drop" for f in FIELDS}
    for r in needs[needs.task == task].itertuples():
        lv[r.field] = r.measured_need
    return lv
