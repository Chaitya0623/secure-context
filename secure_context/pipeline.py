"""End-to-end request: detect, split storage, dehydrate, minimize, call, rehydrate, account for exposure."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .data import Client
from .detect import resolve_coreference
from .policy import TASKS, coverage
from .privacy import ExposureLedger, KEngine, choose_policy
from .tasks import SYSTEM_PROMPT, build_prompt, utility
from .tokens import Session, rehydrate
from .vault import SplitStorage


@dataclass
class Trace:
    task: str
    mode: str
    levels: dict
    spans: list
    tokenized_notes: str
    prompt: str
    response: str
    rehydrated: str
    unresolved: list
    k_single: int
    k_cumulative: int
    utility: float
    coverage: float
    refused: bool = False
    timings: dict = field(default_factory=dict)


class Pipeline:
    """mode: 'full' (no protection), 'masked' (tokens, all fields exact), 'default' (task policy),
    'adaptive' (per-client policy search with cumulative k budget)."""

    def __init__(self, storage: SplitStorage, engine: KEngine, llm, k_min: int = 10):
        self.storage = storage
        self.engine = engine
        self.llm = llm
        self.k_min = k_min
        self.ledger = ExposureLedger(engine)

    def run(self, c: Client, ctx_id: str, task: str, mode: str = "adaptive", session: Session | None = None,
            levels: dict | None = None) -> Trace:
        from .policy import ALL_EXACT
        t = {}
        session = session or Session()
        t0 = time.perf_counter()
        spans = resolve_coreference(self.storage.detector.detect(c.notes, c), c)
        t["detect_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        notes_tok = self.storage.session_notes(ctx_id, session)
        t["rekey_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        refused = False
        if levels is None:
            if mode in ("full", "masked"):
                levels = dict(ALL_EXACT)
            elif mode == "default":
                levels = dict(TASKS[task]["default"])
            else:
                best = choose_policy(task, c, self.engine, self.k_min, self.ledger)
                if best is None:
                    refused = True
                    levels = {k: "drop" for k in ALL_EXACT}
                else:
                    levels = best[0]
        t["policy_ms"] = (time.perf_counter() - t0) * 1e3

        if mode == "full":
            prompt = f"Task: {TASKS[task]['name']}.\nClient: {c.name}, DOB {c.dob}, SSN {c.ssn}, {c.email}, " \
                     f"{c.phone}, {c.address}\n" + build_prompt(task, c, levels, session, c.notes).split("\n", 3)[-1]
        else:
            prompt = build_prompt(task, c, levels, session, notes_tok)

        t0 = time.perf_counter()
        response = "" if refused else self.llm.complete(SYSTEM_PROMPT, prompt, task=task, client=c, levels=levels,
                                                         session=session)
        t["llm_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        r = rehydrate(response, session)
        t["rehydrate_ms"] = (time.perf_counter() - t0) * 1e3

        k_single = 1 if mode == "full" else self.engine.k(c, levels)
        if not refused and mode != "full":
            self.ledger.record(c, levels, task)
        k_cum = 1 if mode == "full" else self.ledger.k_now(c)
        return Trace(task, mode, levels, spans, notes_tok, prompt, response, r.text, r.unresolved, k_single, k_cum,
                     0.0 if refused else utility(task, c, levels), coverage(levels, task)[0], refused, t)
