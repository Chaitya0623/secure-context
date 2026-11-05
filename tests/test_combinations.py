import numpy as np
import pandas as pd

from secure_context import combinations as C


def _pop():
    # 12 people. Age band and region each split them 6/6; together 3/3/3/3.
    # Person 0 is the only Engineer, and the only married Engineer in the 40s.
    n = 12
    return pd.DataFrame({
        'age': [45] * 6 + [65] * 6,
        'region': (['Northeast'] * 3 + ['West'] * 3) * 2,
        'state': (['NY'] * 3 + ['CA'] * 3) * 2,
        'city': ['PUMA 1'] * n,
        'occ_cat': ['Engineer'] + ['Teacher'] * (n - 1),
        'occupation': ['1 Engineer'] + ['2 Teacher'] * (n - 1),
        'aum': [2.0] * n,
        'income': [80e3] * 6 + [300e3] * 6,
        'marital': ['Married'] * 6 + ['Single'] * 6,
        'largest_group': ['Broad equity'] * n,
        'largest_group_weight': [60] * n,
        'top_group': ['Broad equity'] * n, 'top_single': [False] * n, 'top_weight': [60] * n,
    })


def test_combination_k_counts_matching_people():
    ks = C.combination_k(_pop(), [0, 5, 11], 'coarse')
    assert list(ks[('age',)]) == [6, 6, 6]
    assert list(ks[('age', 'location')]) == [3, 3, 3]
    assert list(ks[('occupation',)]) == [1, 11, 11]
    assert len(ks) == 2 ** len(C.ATTRS) - 1


def test_pairs_can_be_riskier_than_either_field():
    ks = C.combination_k(_pop(), range(12), 'coarse')
    pairs = C.pair_synergy(ks, k_min=4).set_index(['a', 'b'])
    e = pairs.loc[('age', 'location')]
    assert e.risk_a == 0 and e.risk_b == 0 and e.joint == 100 and e.synergy == 100


def test_minimal_sets_skip_supersets():
    ks = C.combination_k(_pop(), [0], 'coarse')
    per, freq = C.minimal_sets(ks, k_min=2)
    # occupation alone identifies person 0, so no superset of it is listed
    assert per.smallest[0] == 1
    assert not any('occupation' in f and f != 'occupation' for f in freq.fields)


def test_crowd_homogeneity_flags_shared_hidden_value():
    pop = _pop()
    # Sending age band: the 40s crowd (6 people) all have the same income band and marital status.
    h = C.crowd_homogeneity(pop, [1, 7], ['age'], k_min=5).set_index('hidden')
    assert h.loc['income band', 'pct_disclosed'] == 100
    assert h.loc['marital status', 'pct_disclosed'] == 100
    assert h.loc['state', 'pct_disclosed'] == 0  # crowd is split 3/3 across states


def test_inference_links_are_bounded():
    inf = C.inference_links(_pop())
    assert inf.uncertainty_removed.between(0, 1 + 1e-9).all()
    row = inf[(inf.known == 'age') & (inf.inferred == 'marital')].uncertainty_removed.iloc[0]
    assert np.isclose(row, 1.0)  # age band determines marital status in this population
