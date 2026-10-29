"""Differentially private cohort statistics (NIST SP 800-226 style Laplace mechanism).

Use case: "how does this client's equity share compare with similar clients?"
An exact cohort average lets an attacker run a differencing attack: ask for
the average with and without one client, and solve for that client's value.
"""
from __future__ import annotations

import numpy as np


def laplace_mean(values: np.ndarray, epsilon: float, lo: float = 0, hi: float = 100, rng=None) -> float:
    rng = rng or np.random.default_rng()
    v = np.clip(values, lo, hi)
    n = len(v)
    sens = (hi - lo) / n
    return float(v.mean() + rng.laplace(0, sens / epsilon))


def differencing_attack(values: np.ndarray, epsilon: float | None, trials: int = 500, seed: int = 0) -> float:
    """Mean absolute error (points) when recovering one member's value from two cohort means.

    With epsilon=None the means are exact and the error is zero. Each query
    spends epsilon, so the pair costs 2*epsilon of privacy budget.
    """
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        i = rng.integers(len(values))
        others = np.delete(values, i)
        n = len(values)
        if epsilon is None:
            m_all, m_rest = values.mean(), others.mean()
        else:
            m_all, m_rest = laplace_mean(values, epsilon, rng=rng), laplace_mean(others, epsilon, rng=rng)
        est = min(max(n * m_all - (n - 1) * m_rest, 0), 100)  # attacker clips to the valid range
        errs.append(abs(est - values[i]))
    return float(np.mean(errs))


def utility_error(values: np.ndarray, epsilon: float, trials: int = 500, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([abs(laplace_mean(values, epsilon, rng=rng) - values.mean()) for _ in range(trials)]))
