import pytest

from secure_context import real_data
from secure_context.exposure import amount_fingerprints, transaction_streams, transaction_unicity
from secure_context.policy import ALL_EXACT, TASKS
from secure_context.privacy import KEngine

pytestmark = pytest.mark.skipif(not real_data.available(), reason="public data sample not present")


@pytest.fixture(scope="module")
def real():
    directory, clients = real_data.load_real_environment(n_clients=40, max_directory=30_000)
    return directory, clients, KEngine(directory)


def test_clients_live_in_directory(real):
    directory, clients, _ = real
    for c in clients:
        row = directory.loc[c.person_idx]
        assert row.age == c.age and row.state == c.state and abs(row.aum - c.aum) < 1e-3


def test_masking_leaves_most_clients_unique(real):
    _, clients, engine = real
    assert sum(engine.k(c, ALL_EXACT) == 1 for c in clients) / len(clients) > 0.8


def test_general_requests_reveal_nothing(real):
    _, clients, engine = real
    assert all(engine.k(c, TASKS["general"]["default"]) == engine.n for c in clients)


def test_intent_map_covers_bitext_intents():
    assert len(real_data.INTENT_TASK) == 24


def test_transactions_more_unique_when_exact(real):
    _, clients, _ = real
    tu = transaction_unicity(transaction_streams(clients), points=(2,), trials=100)
    exact = tu[tu.representation.str.startswith("exact")].pct_unique.iloc[0]
    coarse = tu[tu.representation.str.startswith("month and")].pct_unique.iloc[0]
    assert exact >= coarse


def test_amount_rounding_reduces_uniqueness():
    df = amount_fingerprints(real_data.load_advisory())
    t = df[df.amount == "goal target"].set_index("precision").pct_unique
    assert t["exact"] >= t["to $10K"] >= t["bucket"]
