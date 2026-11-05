"""Combination risk: which fields are harmless alone but identifying together.

    python3 eval/run_combinations.py              # about 15 seconds on the committed sample

Uses the same public-data environment as run_real.py (full data if downloaded, else data/sample).
Writes eval/results_real/combinations_*.csv, household_gap.csv, combinations_summary.json
and eval/figures_real/combinations_*.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context import combinations as C  # noqa: E402
from secure_context.policy import TASKS  # noqa: E402
from secure_context.privacy import KEngine  # noqa: E402
from secure_context.real_data import DATA, load_real_environment  # noqa: E402

RES = ROOT / "eval" / "results_real"
FIG = ROOT / "eval" / "figures_real"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order (validated palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# The firm knows dependents; the Census person file does not.
BOOK_ATTRS = {**C.ATTRS,
              "dependents": ("Dependents", lambda p: p.dependents.astype(str), lambda p: p.dependents.astype(str)),
              "has_dependents": ("Has dependents", lambda p: (p.dependents != "0").astype(str),
                                 lambda p: (p.dependents != "0").astype(str))}

# Meeting brief as the task policy sends it (age band, asset bucket, sector mix, household exact),
# and the household variants a policy could choose instead.
MEETING = ("age", "aum", "holdings")
VARIANTS = {
    "as sent: marital + dependents": MEETING + ("marital", "dependents"),
    "marital status only": MEETING + ("marital",),
    "'has dependents' flag only": MEETING + ("has_dependents",),
    "household dropped (current k engine)": MEETING,
}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y" if ax.name != "polar" else "both", color=GRID, lw=.6)
    ax.set_axisbelow(True)


def k_for(pop, attrs, fields):
    kc = C.keys(pop, "coarse", attrs)
    code = C._code([kc[a] for a in fields])
    return np.bincount(code)[code]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=1000)
    ap.add_argument("--advisor-book", type=int, default=150)
    ap.add_argument("--k-min", type=int, default=10)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    kmin = args.k_min

    directory, clients = load_real_environment(args.clients)
    pidx = np.array([c.person_idx for c in clients])
    book = C.book_frame(clients)
    sub = np.random.default_rng(0).choice(len(book), size=args.advisor_book, replace=False)  # same draw as run_real
    adv = book.iloc[sub].reset_index(drop=True)
    refs = {
        f"Census directory ({len(directory):,})": (directory, pidx, C.ATTRS),
        f"firm client book ({len(book):,})": (book, np.arange(len(book)), {k: v for k, v in BOOK_ATTRS.items() if k != "has_dependents"}),
        f"one advisor's book ({len(adv):,})": (adv, np.arange(len(adv)), {k: v for k, v in BOOK_ATTRS.items() if k != "has_dependents"}),
    }

    # Sanity check: on the meeting brief's fields this engine and KEngine agree exactly.
    eng = KEngine(directory)
    k_eng = np.array([eng.k(c, TASKS["meeting_prep"]["default"], effective=False) for c in clients])
    agree = float((k_eng == k_for(directory, C.ATTRS, MEETING)[pidx]).mean())

    risk_rows, pair_rows, min_rows, size_rows = [], [], [], []
    summary = dict(data=str(DATA.relative_to(ROOT)), directory_rows=len(directory), clients=len(clients), k_min=kmin,
                   agreement_with_kengine_meeting_prep=agree)
    for ref, (pop, idx, attrs) in refs.items():
        levels = ["coarse", "fine"] if ref.startswith("Census") else ["coarse"]
        for level in levels:
            ks = C.combination_k(pop, idx, level, attrs)
            r = C.combination_risk(ks, kmin).assign(reference=ref, level=level)
            risk_rows.append(r)
            pair_rows.append(C.pair_synergy(ks, kmin).assign(reference=ref, level=level))
            per, freq = C.minimal_sets(ks, kmin)
            min_rows.append(freq.assign(reference=ref, level=level))
            g = r.groupby("size").pct_k_below_min
            size_rows.append(pd.DataFrame(dict(size=g.max().index, worst=g.max().values, median=g.median().values,
                                               reference=ref, level=level)))
            summary[f"{ref} [{level}]"] = dict(
                max_single_field_pct_k_below_min=float(r[r["size"] == 1].pct_k_below_min.max()),
                all_fields_pct_k_below_min=float(r[r["size"] == r["size"].max()].pct_k_below_min.iloc[0]),
                clients_with_identifying_set_pct=float(100 * (per.smallest > 0).mean()),
                smallest_set_size_distribution=per.smallest.value_counts().sort_index().to_dict(),
                top_minimal_sets=freq.head(5).to_dict(orient="records"))

    risk = pd.concat(risk_rows)
    pairs = pd.concat(pair_rows)
    mins = pd.concat(min_rows)
    sizes = pd.concat(size_rows)
    risk.to_csv(RES / "combinations_risk.csv", index=False)
    pairs.to_csv(RES / "combinations_pairs.csv", index=False)
    mins.to_csv(RES / "combinations_minimal_sets.csv", index=False)
    sizes.to_csv(RES / "combinations_by_size.csv", index=False)

    # Household gap: the meeting brief sends household exact, but the k engine does not count it.
    gap = []
    for ref, (pop, idx, _) in refs.items():
        for name, fields in VARIANTS.items():
            if ref.startswith("Census") and ("dependents" in fields or "has_dependents" in fields):
                continue  # not in the Census person file; an outside attacker can't match on it
            k = k_for(pop, BOOK_ATTRS if not ref.startswith("Census") else C.ATTRS, fields)[idx]
            gap.append(dict(reference=ref, variant=name, pct_k_below_min=100 * float((k < kmin).mean()),
                            pct_unique=100 * float((k == 1).mean()), median_k=float(np.median(k))))
    gap = pd.DataFrame(gap)
    gap.to_csv(RES / "household_gap.csv", index=False)
    summary["household_gap"] = gap.to_dict(orient="records")

    # Inference links and crowd homogeneity (k is fine, but the crowd agrees on a hidden value)
    inf = C.inference_links(directory)
    inf.to_csv(RES / "combinations_inference.csv", index=False)
    hom = pd.concat([C.crowd_homogeneity(directory, pidx, list(MEETING), k_min=kmin).assign(sent="meeting brief"),
                     C.crowd_homogeneity(directory, pidx, ["aum", "holdings"], k_min=kmin).assign(sent="concentration check"),
                     C.crowd_homogeneity(directory, pidx, ["age", "location", "aum", "holdings", "marital"], k_min=kmin)
                     .assign(sent="five coarse fields")])
    hom.to_csv(RES / "combinations_homogeneity.csv", index=False)
    # sent field -> withheld attribute, excluding the trivial "a field reveals itself" pairs
    links = inf[inf.known.isin(list(C.ATTRS)) & inf.inferred.str.startswith("hidden: ") & (inf.uncertainty_removed < .99)]
    summary["inference_top"] = links.head(8).to_dict(orient="records")
    summary["crowd_homogeneity"] = hom.to_dict(orient="records")
    (RES / "combinations_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Figure 1: risk vs. number of fields combined (worst subset of each size), coarse levels
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    coarse = sizes[sizes.level == "coarse"]
    for col, (ref, d) in zip(SERIES, coarse.groupby("reference", sort=False)):
        ax.plot(d["size"], d.worst, color=col, lw=2, marker="o", ms=5, label=ref)
        ax.annotate(f"{d.worst.iloc[-1]:.0f}%", (d["size"].iloc[-1], d.worst.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    _style(ax)
    ax.set_xlabel("Number of coarse fields sent together", color=INK2, fontsize=9)
    ax.set_ylabel(f"Clients with k < {kmin} (%)\nworst combination", color=INK2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(.8, coarse["size"].max() + .7)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("Each field is safe alone; together they identify", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "combinations_by_size.png", dpi=160)
    plt.close(fig)

    # Figure 2: the combination network for the firm's book (node = field alone, edge = pair)
    ref = [r for r in refs if r.startswith("firm")][0]
    p = pairs[(pairs.reference == ref) & (pairs.level == "coarse")]
    single = risk[(risk.reference == ref) & (risk.level == "coarse") & (risk["size"] == 1)].set_index("fields").pct_k_below_min
    nodes = list(single.index)
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}
    labels = {k: v[0] for k, v in BOOK_ATTRS.items()}
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    blues = plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#2a78d6", "#0d366b"])
    for _, e in p.sort_values("joint").iterrows():
        if e.joint < 1:
            continue
        (x1, y1), (x2, y2) = pos[e.a], pos[e.b]
        ax.plot([x1, x2], [y1, y2], color=blues(min(e.joint / 40, 1)), lw=1 + e.joint / 5, zorder=1,
                solid_capstyle="round")
        if e.joint >= 15:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{e.joint:.0f}%", fontsize=8, ha="center", va="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none"), zorder=3)
    for n in nodes:
        x, y = pos[n]
        ax.scatter([x], [y], s=900, color="#fcfcfb", edgecolor=INK2, lw=1.2, zorder=2)
        ax.text(x * 1.42, y * 1.24, f"{labels[n]}\nalone: {single[n]:.1f}%", ha="center", va="center", fontsize=8.5,
                color=INK)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.set_title(f"Pairs that identify clients within the firm's book\n(edge = % of clients with k < {kmin} "
                 "when both fields are sent; labeled at 15% or more)", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "combinations_network.png", dpi=160)
    plt.close(fig)

    # Figure 3: household gap for the meeting brief
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    variants = list(VARIANTS)
    refs_l = list(refs)
    h = .26
    for j, (col, ref) in enumerate(zip(SERIES, refs_l)):
        d = gap[gap.reference == ref].set_index("variant").reindex(variants)
        y = np.arange(len(variants)) + (j - 1) * h
        ok = d.pct_k_below_min.notna()
        ax.barh(y[ok], d.pct_k_below_min[ok], height=h - .04, color=col, label=ref)
        for yy, v in zip(y[ok], d.pct_k_below_min[ok]):
            ax.text(v + 1, yy, f"{v:.1f}%" if v < 10 else f"{v:.0f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(np.arange(len(variants)), variants, fontsize=8)
    ax.invert_yaxis()
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_xlim(0, 112)
    ax.set_xlabel(f"Clients with k < {kmin} (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Meeting brief: counting the household it already sends", loc="left", fontsize=10, color=INK)
    fig.text(.01, .01, "Census has marital status but not dependents, so dependent variants are book-only.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(FIG / "combinations_household_gap.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("household_gap", "crowd_homogeneity", "inference_top")},
                     indent=1, default=float)[:3000])
    print(gap.to_string())
    print(hom.to_string())


if __name__ == "__main__":
    main()
