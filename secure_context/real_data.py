"""Build the experimental environment from public datasets.

Directory (attacker's reference data)
    Census ACS PUMS 2023 adults: real age, state, local area (PUMA), occupation and income.
    Each person gets a financial-asset level and portfolio mix drawn from a Federal
    Reserve SCF 2022 household in the same age, income and occupation cell
    (a statistical match, since no public file has both geography and wealth).

Client book
    Financial-Advisory-Clients profiles (age, income, marital status, dependents,
    portfolio value, goal, horizon, risk tolerance, topics discussed, recommended
    products). Each client is placed on a matching ACS person, so the client exists
    in the directory with real location and occupation. Names, IDs and contact
    details are generated with Faker, and the advisor note is written from the
    client's own topics and products.

Requests
    Bitext wealth-management intents, mapped to the context each one needs.

PII documents
    Gretel synthetic_pii_finance_multilingual, English test split.

Run scripts/download_data.py first; files are read from data/processed/.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .data import REGION, SINGLE_STOCKS, Client, Holding, _Builder, holdings_summary

_ROOT = Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed" if (_ROOT / "data" / "processed" / "acs_persons.csv.gz").exists() \
    else _ROOT / "data" / "sample"  # small committed sample (ACS, SCF, advisory clients) for the hosted app


def available() -> bool:
    return (DATA / "acs_persons.csv.gz").exists() and (DATA / "advisory_clients.csv").exists()

FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
        "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
        "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY", "72": "PR"}
ACS_DEFAULT_ORDER = [("NY", 152234), ("NJ", 69058), ("CA", 284713), ("TX", 212806), ("FL", 166376), ("IL", 93097)]
REGION.setdefault("DC", "South")
REGION.setdefault("PR", "South")

# ACS OCCP code ranges (2018 SOC-based) to major groups
OCC_RANGES = [(10, 440, "Management"), (500, 960, "Business and financial"), (1005, 1240, "Computer and math"),
              (1305, 1560, "Architecture and engineering"), (1600, 1980, "Science"),
              (2001, 2060, "Community and social service"), (2100, 2180, "Legal"), (2205, 2555, "Education"),
              (2600, 2970, "Arts and media"), (3000, 3550, "Healthcare practitioners"), (3601, 3655, "Healthcare support"),
              (3700, 3960, "Protective service"), (4000, 4160, "Food service"), (4200, 4255, "Building and grounds"),
              (4330, 4655, "Personal care"), (4700, 4965, "Sales"), (5000, 5940, "Office and admin"),
              (6005, 6130, "Farming"), (6200, 6950, "Construction"), (7000, 7640, "Maintenance and repair"),
              (7700, 8990, "Production"), (9005, 9760, "Transportation"), (9800, 9830, "Military")]
PROFESSIONAL = {"Management", "Business and financial", "Computer and math", "Architecture and engineering", "Science",
                "Community and social service", "Legal", "Education", "Arts and media", "Healthcare practitioners"}
SERVICE = {"Healthcare support", "Protective service", "Food service", "Building and grounds", "Personal care", "Sales",
           "Office and admin"}

# Bitext intent -> task policy (which client context the request needs)
INTENT_TASK = {
    "arrange_meeting": "meeting_prep", "get_manager": "general", "contact_manager": "general",
    "customer_service": "general", "human_agent": "general", "create_account": "general",
    "delete_account": "general", "recover_password": "general", "requirements_to_create_account": "general",
    "become_client": "general", "check_fees": "general", "check_services": "general",
    "minimum_amount_to_invest": "general", "search_for_stocks": "general", "set_price_alert": "general",
    "calculate_portfolio_risk": "concentration", "check_portfolio": "portfolio_review",
    "portfolio_performance": "portfolio_review", "portfolio_value": "portfolio_review",
    "check_balances": "portfolio_review", "run_simulator": "income_plan", "implement_own_plan": "income_plan",
    "transfer_money_to_account": "transaction", "withdraw_money_from_account": "transaction",
}

GRETEL_DIRECT = {"name", "first_name", "last_name", "ssn", "email", "phone_number", "street_address", "date_of_birth",
                 "bban", "iban", "credit_card_number", "credit_card_security_code", "bank_routing_number",
                 "customer_id", "passport_number", "driver_license_number", "account_pin", "password", "api_key",
                 "user_name", "employee_id"}


def occ_category(code) -> str:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Not working"
    for lo, hi, name in OCC_RANGES:
        if lo <= c <= hi:
            return name
    return "Not working"


def _occat2(cat: str) -> int:
    return 1 if cat in PROFESSIONAL else 2 if cat in SERVICE else 4 if cat == "Not working" else 3


def _age_cell(a):
    return np.clip((np.asarray(a) - 25) // 10, 0, 5)


def load_scf(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "scf2022.csv.gz")
    df.columns = [c.upper() for c in df.columns]
    for c in ("LIQ", "CDS", "NMMF", "STOCKS", "BOND", "RETQLIQ", "SAVBND", "CASHLI", "OTHMA", "OTHFIN", "FIN"):
        if c not in df:
            df[c] = 0.0
    fin = df[["LIQ", "CDS", "NMMF", "STOCKS", "BOND", "RETQLIQ", "SAVBND", "CASHLI", "OTHMA", "OTHFIN"]].clip(lower=0)
    tot = fin.sum(axis=1).replace(0, np.nan)
    df["sh_cash"] = ((fin.LIQ + fin.CDS + fin.CASHLI + fin.OTHFIN) / tot).fillna(1.0)
    df["sh_stock"] = (fin.STOCKS / tot).fillna(0.0)
    df["sh_eqfund"] = ((fin.NMMF * .7 + fin.RETQLIQ * .65 + fin.OTHMA * .6) / tot).fillna(0.0)
    df["sh_bond"] = ((fin.BOND + fin.SAVBND + fin.NMMF * .3 + fin.RETQLIQ * .35 + fin.OTHMA * .4) / tot).fillna(0.0)
    df["fin_m"] = fin.sum(axis=1) / 1e6
    df["age_cell"] = _age_cell(df.AGE)
    df["occ2"] = df.get("OCCAT2", pd.Series(3, index=df.index)).astype(int)
    df["w"] = df.get("WGT", pd.Series(1.0, index=df.index)).astype(float)
    return df


def load_acs(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "acs_persons.csv.gz", dtype={"ST": str, "PUMA": str, "OCCP": str})
    if df.ST.isna().all():
        # Early versions of the download script missed the 2023 STATE column; rows are in state order.
        order = (DATA / "acs_state_order.txt").read_text().split() if (DATA / "acs_state_order.txt").exists() \
            else [f"{s},{n}" for s, n in ACS_DEFAULT_ORDER]
        st = []
        for item in order:
            s, n = item.split(",")
            st += [s] * int(n)
        if len(st) != len(df):
            raise ValueError("ACS file has no state column and the row counts do not match; rerun download_data.py")
        df["state"] = st
    else:
        df["state"] = df.ST.str.zfill(2).map(FIPS)
    df = df.dropna(subset=["state"])
    df["age"] = df.AGEP.astype(int)
    df["occ_cat"] = df.OCCP.map(occ_category)
    df["occupation"] = df.OCCP.fillna("none").astype(str) + " " + df.occ_cat
    df["city"] = "PUMA " + df.PUMA.str.zfill(5)
    df["region"] = df.state.map(REGION)
    df["income"] = pd.to_numeric(df.PINCP, errors="coerce").fillna(0).clip(lower=0)
    df["marital"] = df.MAR.map({1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Single"}).fillna("Single")
    return df.reset_index(drop=True)


def _income_cells(scf_income, acs_income):
    edges = np.quantile(scf_income, [.2, .4, .6, .8])
    return np.digitize(scf_income, edges), np.digitize(acs_income, edges)


def build_directory(acs: pd.DataFrame, scf: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Attach an SCF household (wealth + portfolio mix) to every ACS adult by matching cell."""
    rng = np.random.default_rng(seed)
    scf = scf.copy()
    scf["inc_cell"], acs_inc = _income_cells(scf.INCOME.values, acs.income.values)
    acs = acs.copy()
    acs["inc_cell"] = acs_inc
    acs["age_cell"] = _age_cell(acs.age)
    acs["occ2"] = acs.occ_cat.map(_occat2)
    donor = np.empty(len(acs), dtype=np.int64)
    groups = {k: g.index.values for k, g in scf.groupby(["age_cell", "inc_cell", "occ2"])}
    looser = {k: g.index.values for k, g in scf.groupby(["age_cell", "inc_cell"])}
    for key, idx in acs.groupby(["age_cell", "inc_cell", "occ2"]).indices.items():
        pool = groups.get(key)
        if pool is None or len(pool) < 5:
            pool = looser.get(key[:2], scf.index.values)
        w = scf.w.values[pool]
        donor[idx] = rng.choice(pool, size=len(idx), p=w / w.sum())
    d = scf.iloc[donor].reset_index(drop=True)
    acs["aum"] = d.fin_m.values.round(3)
    acs["sh_cash"], acs["sh_stock"], acs["sh_eqfund"], acs["sh_bond"] = d.sh_cash.values, d.sh_stock.values, \
        d.sh_eqfund.values, d.sh_bond.values
    # single-stock position: half of direct stock holdings in one name, sector random
    sectors = list(SINGLE_STOCKS)
    sec = rng.choice(sectors, size=len(acs))
    tick = np.array([SINGLE_STOCKS[s][i % len(SINGLE_STOCKS[s])] for i, s in enumerate(sec)])
    w_single = np.round(acs.sh_stock.values * 50).astype(int)
    w_vti = np.round((acs.sh_eqfund.values * .6 + acs.sh_stock.values * .5) * 100).astype(int)
    w_vea = np.round(acs.sh_eqfund.values * 40).astype(int)
    w_bnd = np.round(acs.sh_bond.values * 100).astype(int)
    w_cash = 100 - w_single - w_vti - w_vea - w_bnd
    pos = np.stack([w_single, w_vti, w_vea, w_bnd, np.maximum(w_cash, 0)], axis=1)
    top = np.where(pos[:, :4].max(axis=1) > 0, pos[:, :4].argmax(axis=1), 4)  # all cash -> CASH
    names = np.array(["", "VTI", "VEA", "BND", "CASH"])
    acs["top_ticker"] = np.where(top == 0, tick, names[top])
    acs["top_weight"] = pos[np.arange(len(acs)), top]
    acs["top_group"] = np.where(top == 0, sec, np.array(["", "Broad equity", "Broad equity", "Fixed income", "Cash"])[top])
    acs["top_single"] = top == 0
    groups_w = {"Broad equity": w_vti + w_vea, "Fixed income": w_bnd, "Cash": np.maximum(w_cash, 0)}
    gmat = np.stack(list(groups_w.values()) + [w_single], axis=1)
    gi = gmat.argmax(axis=1)
    gnames = np.array(list(groups_w) + ["single"])
    acs["largest_group"] = np.where(gnames[gi] == "single", sec, gnames[gi])
    acs["largest_group_weight"] = gmat[np.arange(len(acs)), gi]
    acs["single_sector"], acs["single_ticker"] = sec, tick
    acs["risk"] = np.where(acs.age >= 65, "Conservative", np.where(acs.age >= 45, "Moderate", "Aggressive"))
    acs["objective"] = np.where(acs.age >= 65, "Retirement", np.where(acs.age >= 45, "Wealth Accumulation", "Home Purchase"))
    acs["horizon"] = np.where(acs.age >= 65, "5 years", np.where(acs.age >= 45, "15 years", "25 years"))
    acs["person_idx"] = np.arange(len(acs))
    return acs


def _holdings_for(row) -> list[Holding]:
    hs = []
    if row.top_single or row.sh_stock > .02:
        w = int(round(row.sh_stock * 50))
        if w > 0:
            hs.append(Holding(row.single_ticker, f"{row.single_ticker} common stock", row.single_sector, True, w))
    for t, name, grp, w in (("VTI", "Vanguard Total Stock Market ETF", "Broad equity",
                             round((row.sh_eqfund * .6 + row.sh_stock * .5) * 100)),
                            ("VEA", "Vanguard FTSE Developed Markets ETF", "Broad equity", round(row.sh_eqfund * 40)),
                            ("BND", "Vanguard Total Bond Market ETF", "Fixed income", round(row.sh_bond * 100))):
        if w > 0:
            hs.append(Holding(t, name, grp, False, int(w)))
    cash = 100 - sum(h.weight for h in hs)
    hs.append(Holding("CASH", "Cash", "Cash", False, max(cash, 0)))
    if len(hs) == 1:
        hs.insert(0, Holding("VTI", "Vanguard Total Stock Market ETF", "Broad equity", False, 0))
    return sorted(hs, key=lambda h: -h.weight)


def _risk_label(x: float) -> str:
    return "Conservative" if x < 4 else "Moderate" if x < 7 else "Aggressive"


def load_advisory(path=None) -> pd.DataFrame:
    """About 3.5% of rows miss income or portfolio value; they are dropped."""
    df = pd.read_csv(path or DATA / "advisory_clients.csv")
    return df.dropna(subset=["annual_income_usd", "investment_portfolio_usd"]).reset_index(drop=True)


def build_clients(directory: pd.DataFrame, adv: pd.DataFrame, n: int = 1000, seed: int = 11) -> list[Client]:
    """Place each advisory client on a matching directory person and generate identity + note."""
    rng = random.Random(seed)
    fk = Faker("en_US")
    fk.seed_instance(seed)
    adv = adv.sample(n=min(n, len(adv)), random_state=seed).reset_index(drop=True)
    by_age = {a: g for a, g in directory.groupby("age")}
    used = set()
    clients = []
    for k, r in adv.iterrows():
        age = int(min(max(r.age, 25), directory.age.max()))
        pool = by_age.get(age, directory)
        mar = str(r.marital_status)
        cand = pool[(pool.marital == mar) & (pool.income.between(r.annual_income_usd * .75, r.annual_income_usd * 1.25))]
        if len(cand) == 0:
            cand = pool[pool.income.between(r.annual_income_usd * .5, r.annual_income_usd * 1.5)]
        if len(cand) == 0:
            cand = pool
        cand = cand[~cand.person_idx.isin(used)]
        if len(cand) == 0:
            cand = pool[~pool.person_idx.isin(used)]
        row = cand.iloc[rng.randrange(len(cand))]
        used.add(int(row.person_idx))
        aum = float(r.investment_portfolio_usd) / 1e6
        directory.loc[row.person_idx, "aum"] = round(aum, 3)  # the client is this person
        holdings = _holdings_for(row)

        female = rng.random() < .5
        first = fk.first_name_female() if female else fk.first_name_male()
        last = fk.last_name()
        title = ("Mrs." if mar == "Married" else "Ms.") if female else "Mr."
        spouse = f"{fk.first_name_male() if female else fk.first_name_female()} {last}" if mar == "Married" else None
        kids = [f"{fk.first_name()} {last}" for _ in range(int(r.dependents_count))]
        birth_year = 2026 - age - (1 if rng.random() < .4 else 0)
        dob = f"{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        years = int(r.time_horizon_years)
        target = int(r.target_amount_usd)
        c = Client(
            client_id=str(r.client_id), person_idx=int(row.person_idx), name=f"{first} {last}", first=first, last=last,
            title=title, dob=dob, ssn=fk.ssn(),
            account=f"ACCT {rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
            routing=f"0{rng.randint(10000000, 99999999)}",
            email=f"{first[0].lower()}.{last.lower()}{rng.randint(1, 99)}@example{rng.choice(['mail', 'net'])}.com",
            phone=f"({rng.randint(201, 989)}) 555-{rng.randint(100, 199):04d}",
            address=f"{fk.building_number()} {fk.street_name()} {fk.street_suffix()}, {fk.city()}, {row.state} {fk.zipcode()}",
            employer=fk.company(), spouse=spouse, children=kids,
            age=age, city=row.city, state=row.state, occupation=row.occupation, occ_cat=row.occ_cat, aum=aum,
            household=f"{mar}, {int(r.dependents_count)} dependents", objective=str(r.primary_goal),
            risk=_risk_label(float(r.risk_tolerance_1_10)), horizon=f"{years} years",
            liquidity=f"Target ${target / 1000:,.0f}K within {years * 12} months ({str(r.primary_goal).lower()})",
            liquidity_amount=target, holdings=holdings)
        c.notes, c.gold_spans, c.qi_mentions = _note(rng, c, r)
        c.summary = (f"Goal: {c.objective.lower()} over {years} years. Discussed "
                     f"{', '.join(str(r.topics_discussed).split(';')[:3])}. Risk tolerance {c.risk.lower()}.")
        c.extra = dict(employment=str(r.employment_status), income=float(r.annual_income_usd),
                       topics=str(r.topics_discussed).split(";"), products=str(r.recommended_products).split(";"),
                       risk_score=float(r.risk_tolerance_1_10), target=target)
        clients.append(c)
    hs = directory.loc[[c.person_idx for c in clients]]
    for c, (_, row) in zip(clients, hs.iterrows()):
        s = holdings_summary(c.holdings)
        for key in ("top_ticker", "top_weight", "top_group", "top_single", "largest_group", "largest_group_weight"):
            directory.at[c.person_idx, key] = s[key]
        directory.at[c.person_idx, "risk"] = c.risk
        directory.at[c.person_idx, "objective"] = c.objective
        directory.at[c.person_idx, "horizon"] = c.horizon
    return clients


def _note(rng: random.Random, c: Client, r) -> tuple:
    b = _Builder()
    P = lambda v, canon=None: (v, "PERSON", canon or v)
    qi = {}
    b.add(rng.choice(["{name} (DOB {dob}) came in for a {ch} review.", "Annual review with {name}, born {dob}, via {ch}."]),
          dict(name=P(c.name), dob=(f"{c.dob[5:7]}/{c.dob[8:]}/{c.dob[:4]}", "DOB", c.dob),
               ch=(str(r.preferred_channel).lower(), None, None)))
    if c.occ_cat != "Not working" and rng.random() < .7:
        b.add("{ref} works at {emp} ({cat}).", dict(ref=rng.choice([P(c.first, c.name), P(f"{c.title} {c.last}", c.name)]),
                                                   emp=(c.employer, "ORG", c.employer), cat=(c.occ_cat.lower(), None, None)))
        qi["occupation"] = "category"
    elif str(r.employment_status) == "Retired":
        b.add("{ref} is retired.", dict(ref=P(f"{c.title} {c.last}", c.name)))
    if c.spouse and rng.random() < .6:
        b.add("{sp} joined the call.", dict(sp=P(c.spouse.split()[0], c.spouse)))
    b.add("Discussed " + "; ".join(str(r.topics_discussed).split(";")) + ".", {})
    b.add("Recommended " + ", ".join(str(r.recommended_products).split(";")) + ".", {})
    b.add(f"Goal is {c.objective.lower()} with a target of ${c.liquidity_amount:,} in {c.horizon}.", {})
    top = next(h for h in c.holdings if h.ticker != "CASH")
    if top.single and top.weight >= 15 and rng.random() < .7:
        b.add(f"Wants to reduce the {top.ticker} position ({top.weight}%).", {})
        qi["holdings"] = "exact"
    b.add(rng.choice(["SSN {ssn} verified.", "Confirmed last four of SSN {ssn4}."]),
          dict(ssn=(c.ssn, "SSN", c.ssn), ssn4=(c.ssn[-4:], "SSN", c.ssn)))
    b.add("Funds move from {acct}.", dict(acct=(c.account, "ACCOUNT", c.account)))
    if rng.random() < .6:
        b.add("Mail to {addr}.", dict(addr=(c.address, "ADDRESS", c.address)))
    b.add("Reach at {ph} or {em}.", dict(ph=(c.phone, "PHONE", c.phone), em=(c.email, "EMAIL", c.email)))
    if c.children and rng.random() < .4:
        b.add("Add {kid} as beneficiary.", dict(kid=P(c.children[0])))
    return b.text.strip(), b.spans, qi


def load_bitext(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "bitext_wealth.csv")
    df["task"] = df.intent.map(INTENT_TASK).fillna("general")
    return df


def load_gretel(path=None) -> list[dict]:
    df = pd.read_parquet(path or DATA / "gretel_en_test.parquet")
    if "language" in df:
        df = df[df.language.str.lower().str.startswith("english")]
    out = []
    for r in df.itertuples():
        spans = json.loads(r.pii_spans) if isinstance(r.pii_spans, str) else list(r.pii_spans)
        out.append(dict(text=r.generated_text, spans=spans, doc_type=r.document_type))
    return out


def load_real_environment(n_clients: int = 1000, seed: int = 7, max_directory: int | None = None):
    acs = load_acs()
    if max_directory and len(acs) > max_directory:
        acs = acs.sample(n=max_directory, random_state=seed).reset_index(drop=True)
    directory = build_directory(acs, load_scf(), seed)
    clients = build_clients(directory, load_advisory(), n_clients, seed + 4)
    return directory, clients
