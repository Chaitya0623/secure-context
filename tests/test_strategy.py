import numpy as np
import pandas as pd

from secure_context.strategy import _X, extraction_curve


def test_zero_asset_profiles_have_finite_features():
    profile = dict(age=np.array([40, 60]), risk=np.array([1, 0]),
                   aum=np.array([0., 5.]), income=np.array([0, 1]), conc=np.array([10., 20.]))
    features = _X(profile)
    assert np.isfinite(features).all()
    assert features[1, 2] == np.log(5.)
    assert profile['aum'][0] == 0


def test_extraction_with_zero_asset_population():
    pop = pd.DataFrame({'age': [30, 50, 70], 'risk': ['Aggressive', 'Moderate', 'Conservative'],
                        'aum': [0., 1., 10.], 'objective': ['Growth', 'Income', 'Retirement'],
                        'top_single': [True, False, True], 'top_weight': [30., 10., 40.]})
    curve = extraction_curve(pop, query_counts=(25,), n_test=30, book_size=3)
    assert len(curve) == 6
    assert np.isfinite(curve.select_dtypes('number').to_numpy()).all()
