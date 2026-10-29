"""Re-identification risk, cumulative exposure, and task-aware minimization.

KEngine          crowd size k: how many people in the population directory match
                 everything a context reveals (k = 1 means unique).
ExposureLedger   remembers the most detailed level of every quasi-identifier
                 already sent about a client and computes cumulative k, since a
                 model provider could link requests by content even when tokens rotate.
choose_policy    "intelligent context sharing": per client and task, pick the levels
                 with the best task utility whose (cumulative) k stays at or above k_min.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .data import Client
from .policy import FIELDS, QI_FIELDS, TASKS, conc_bucket, more_detailed, rank
from .tasks import utility


def _codes(series_list):
    if len(series_list) == 1:
        return pd.factorize(series_list[0])[0].astype(np.int32)
    df = pd.DataFrame({i: s.values for i, s in enumerate(series_list)})
    return df.groupby(list(df.columns), sort=False, dropna=False).ngroup().values.astype(np.int32)


class KEngine:
    """Counts directory entries that match a client on every disclosed quasi-identifier.

    Levels are nested (exact age implies the band, city implies state and region,
    exact holdings imply the sector mix), so for each client every directory row
    gets a small match signature: the most detailed level it agrees on, per field.
    A histogram of signatures (at most 432 cells) answers any policy instantly,
    which keeps the per-client policy search fast on a million-row directory.
    Pass `weights` (e.g. census person weights) to estimate population counts.
    """

    def __init__(self, pop: pd.DataFrame, weights=None):
        self.n = len(pop)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.total = float(self.weights.sum()) if self.weights is not None else self.n
        p = pop
        bucket = pd.cut(p.aum, [-1, 1, 5, 10, 25, 1e12], right=False).astype(str)
        sect = [p.largest_group, p.largest_group_weight.map(conc_bucket)]
        self.codes = {
            ("age", "exact"): _codes([p.age]),
            ("age", "band"): _codes([p.age // 10]),
            ("location", "city"): _codes([p.state, p.city]),
            ("location", "state"): _codes([p.state]),
            ("location", "region"): _codes([p.region]),
            ("occupation", "exact"): _codes([p.occupation]),
            ("occupation", "category"): _codes([p.occ_cat]),
            ("aum", "exact"): _codes([bucket, p.aum.round(1)]),
            ("aum", "bucket"): _codes([bucket]),
            ("holdings", "exact"): _codes([p.top_ticker, p.top_weight // 5] + sect),
            ("holdings", "anonymized"): _codes([p.top_group, p.top_single, p.top_weight // 5] + sect),
            ("holdings", "sectors"): _codes(sect),
        }
        self.fields = QI_FIELDS
        self.nlev = [len(FIELDS[f].levels) for f in self.fields]
        self.mult = np.cumprod([1] + self.nlev[:-1])
        grid = np.array(list(itertools.product(*[range(n) for n in reversed(self.nlev)])))[:, ::-1]
        self._sig_levels = grid[np.argsort(grid @ self.mult)]  # row s holds the level index per field
        self._hist = {}

    @staticmethod
    def effective(levels: dict, c: Client) -> dict:
        """Tokenized notes can still mention occupation, city or a ticker in plain words."""
        eff = dict(levels)
        if levels.get("notes") == "tokenized":
            for f, lv in c.qi_mentions.items():
                eff[f] = more_detailed(f, lv, eff[f])
        return eff

    def _histogram(self, pidx: int):
        h = self._hist.get(pidx)
        if h is None:
            sig = np.zeros(self.n, dtype=np.int32)
            for f, n, m in zip(self.fields, self.nlev, self.mult):
                best = np.full(self.n, n - 1, dtype=np.int32)  # "drop" always matches
                for i in range(n - 2, -1, -1):                 # coarse to detailed
                    col = self.codes[(f, FIELDS[f].levels[i])]
                    best[col == col[pidx]] = i
                sig += best * m
            h = np.bincount(sig, weights=self.weights, minlength=int(np.prod(self.nlev)))
            if len(self._hist) > 50_000:
                self._hist.clear()
            self._hist[pidx] = h
        return h

    def k(self, c: Client, levels: dict, effective: bool = True) -> int:
        eff = self.effective(levels, c) if effective else levels
        req = np.array([rank(f, eff[f]) for f in self.fields])
        h = self._histogram(c.person_idx)
        return int(round(h[(self._sig_levels <= req).all(axis=1)].sum()))

    def k_many(self, clients: list[Client], levels: dict) -> np.ndarray:
        return np.array([self.k(c, levels) for c in clients])


class ExposureLedger:
    def __init__(self, engine: KEngine):
        self.engine = engine
        self.disclosed: dict[str, dict] = {}
        self.history: dict[str, list] = {}

    def merged(self, c: Client, levels: dict) -> dict:
        eff = KEngine.effective(levels, c)
        prev = self.disclosed.get(c.client_id)
        if prev:
            for f in QI_FIELDS:
                eff[f] = more_detailed(f, eff[f], prev[f])
        return eff

    def k_if_sent(self, c: Client, levels: dict) -> int:
        return self.engine.k(c, self.merged(c, levels), effective=False)

    def record(self, c: Client, levels: dict, task: str = ""):
        self.disclosed[c.client_id] = self.merged(c, levels)
        self.history.setdefault(c.client_id, []).append((task, dict(levels)))

    def k_now(self, c: Client) -> int:
        d = self.disclosed.get(c.client_id)
        return self.engine.n if d is None else self.engine.k(c, d, effective=False)


def candidate_policies(task: str):
    """Unneeded fields are dropped. Needed non-identifying fields are sent at the
    needed level (they cost nothing in k). Needed quasi-identifiers and the notes
    try every level, since those trade utility for privacy."""
    needs = TASKS[task]["needs"]
    choices = []
    for f, fld in FIELDS.items():
        if f not in needs:
            choices.append([(f, "drop")])
        elif fld.qi or f == "notes":
            choices.append([(f, lv) for lv in fld.levels])
        else:
            choices.append([(f, needs[f])])
    for combo in itertools.product(*choices):
        yield dict(combo)


def choose_policy(task: str, c: Client, engine: KEngine, k_min: int = 10, ledger: ExposureLedger | None = None,
                  min_utility: float = 0.0):
    """Best utility subject to k >= k_min; ties go to the larger k. Returns (levels, utility, k) or None."""
    best = None
    for lv in candidate_policies(task):
        u = utility(task, c, lv)
        if u < min_utility:
            continue
        if best is not None and u < best[1]:
            continue
        k = ledger.k_if_sent(c, lv) if ledger else engine.k(c, lv)
        if k < k_min:
            continue
        if best is None or u > best[1] or (u == best[1] and k > best[2]):
            best = (lv, u, k)
    return best
