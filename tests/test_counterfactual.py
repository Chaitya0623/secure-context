"""The counterfactual necessity test must recover dependencies we know in advance."""
import pandas as pd
import pytest

from secure_context import counterfactual as cf
from secure_context.data import load_environment
from secure_context.llm import SimulatedLLM
from secure_context.policy import FIELDS, TASKS
from secure_context.tasks import CORE_TASKS


class ProbeLLM:
    """Reads only the prompt, like a real model. Depends on the age decade and the risk value, nothing else."""
    name = "probe"
    EQUITY = {"Conservative": "40% equity", "Moderate": "60% equity", "Aggressive": "80% equity"}

    def complete(self, system, prompt, **_):
        ctx = dict(line.split(": ", 1) for line in prompt.splitlines()[1:] if ": " in line)
        age = ctx.get("Age")
        decade = f"{int(age.split('-')[0]) // 10 * 10}s" if age else "unknown age"
        return f"For {ctx['Client ref']}: someone in their {decade}, {self.EQUITY.get(ctx.get('Risk tolerance'), 'no equity view')}."


@pytest.fixture(scope="module")
def book():
    _, clients = load_environment(n_pop=5000, n_clients=80)
    return clients


@pytest.fixture(scope="module")
def probe_needs(book):
    raw = cf.run(ProbeLLM(), book[:30], ["meeting_prep"], pool=book)
    return raw, cf.measured_needs(raw).set_index("field")


def test_probe_dependencies_recovered(probe_needs):
    _, needs = probe_needs
    assert needs.loc["age", "measured_need"] == "band"     # the decade is enough, exact age is not needed
    assert needs.loc["risk", "measured_need"] == "exact"
    others = set(FIELDS) - {"age", "risk"}
    assert (needs.loc[sorted(others), "measured_need"] == "drop").all()   # negative controls


def test_probe_swap_separates_value_from_presence(probe_needs):
    _, needs = probe_needs
    assert needs.loc["risk", "swap_change_rate"] > 0.3
    assert needs.loc["occupation", "swap_change_rate"] == 0
    assert needs.loc["location", "swap_change_rate"] == 0


def test_null_control_is_clean(probe_needs, book):
    raw, _ = probe_needs
    assert not raw[raw.kind == "null"].changed.any()
    sim = cf.run(SimulatedLLM(), book[:10], CORE_TASKS, pool=book)
    assert not sim[sim.kind == "null"].changed.any()


def test_simulator_matches_what_its_code_reads(book):
    raw = cf.run(SimulatedLLM(), book[:30], CORE_TASKS, pool=book)
    needs = cf.measured_needs(raw)
    for task in ("meeting_prep", "income_plan"):
        t = needs[needs.task == task]
        assert t.agree.all(), t[~t.agree]
    c = needs[needs.task == "concentration"].set_index("field")
    # the simulated concentration answer never mentions assets, though policy.py lists them as needed
    assert c.loc["aum", "measured_need"] == "drop" and TASKS["concentration"]["needs"]["aum"] == "bucket"
    assert c.drop("aum").agree.all()


def test_level_logic_exact_vs_band(book):
    raw = cf.run(SimulatedLLM(), book[:30], CORE_TASKS, pool=book, fields=["age"])
    needs = cf.measured_needs(raw).set_index("task")
    assert needs.loc["meeting_prep", "measured_need"] == "band"   # only the band is used
    assert needs.loc["income_plan", "measured_need"] == "exact"   # the estimate uses exact age


def test_deterministic(book):
    a = cf.run(SimulatedLLM(), book[:8], ["income_plan"], pool=book, seed=3)
    b = cf.run(SimulatedLLM(), book[:8], ["income_plan"], pool=book, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_swap_changes_only_one_field(book):
    import random
    c = book[0]
    for f, attrs in cf.FIELD_ATTRS.items():
        d = cf.swap_field(c, f, book, random.Random(0))
        changed = {k for k in vars(c) if getattr(c, k) != getattr(d, k)}
        assert changed and changed <= set(attrs), (f, changed)


def test_tolerance_rises_with_null_noise():
    rows = [dict(client=i, task="general", kind="null", field="", level="", changed=i < 2) for i in range(10)]
    rows += [dict(client=i, task="general", kind="level", field="age", level=lv, changed=i < 2)
             for i in range(10) for lv in ("band", "drop")]
    needs = cf.measured_needs(pd.DataFrame(rows), tolerance=0.1).set_index("field")
    # 20% change on drop equals the 20% null noise, so age is not needed
    assert needs.loc["age", "measured_need"] == "drop"
    assert abs(needs.loc["age", "tolerance_used"] - 0.3) < 1e-9


class FakeJudgeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, system, prompt, **_):
        self.calls += 1
        return self.reply


def test_judge_parsing_and_shortcut(book):
    v = cf.Variant("level", "age", "band", book[0], dict(cf.BASELINE), None)
    same = FakeJudgeLLM("SAME")
    assert cf.Judge(same)("Answer one.", v, "Answer   one.", v, "income_plan") is False
    assert same.calls == 0                                   # identical text needs no judge call
    assert cf.Judge(FakeJudgeLLM("Different."))("a", v, "b", v, "income_plan") is True
    assert cf.Judge(FakeJudgeLLM("unclear"))("a", v, "b", v, "income_plan") is None
