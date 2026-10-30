"""Synthetic data for the experimental environment.

Three things are generated, all from one seed so every run is reproducible:

1. A population directory (default 50,000 people). This plays the attacker's
   outside knowledge: the quasi-identifiers someone could look up or buy
   (age, city, occupation, rough wealth, a notable holding).
2. A client book sampled from that population (default 1,000 clients). Each
   client gets an identity record, a financial profile, holdings, and a
   free-text advisor note with ground-truth PII spans.
3. Ground truth for every note: which characters are direct identifiers
   (gold spans) and which quasi-identifiers the text mentions in plain words.

Nothing here is real. SSNs use Faker's generator, phone numbers use the 555
exchange, and emails use example domains.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

# Reference tables

STATES = [("CA", 12), ("TX", 9), ("FL", 6.7), ("NY", 5.9), ("PA", 3.9), ("IL", 3.8), ("OH", 3.5),
          ("GA", 3.2), ("NC", 3.2), ("MI", 3), ("NJ", 2.8), ("VA", 2.6), ("WA", 2.3), ("AZ", 2.2),
          ("MA", 2.1), ("TN", 2.1), ("IN", 2), ("MD", 1.8), ("MO", 1.8), ("WI", 1.8), ("CO", 1.8),
          ("MN", 1.7), ("SC", 1.6), ("AL", 1.5), ("LA", 1.4), ("KY", 1.3), ("OR", 1.3), ("OK", 1.2),
          ("CT", 1.1), ("UT", 1), ("IA", 1), ("NV", 1), ("AR", .9), ("MS", .9), ("KS", .9), ("NM", .6),
          ("NE", .6), ("ID", .6), ("WV", .5), ("HI", .4), ("NH", .4), ("ME", .4), ("MT", .3), ("RI", .3),
          ("DE", .3), ("SD", .3), ("ND", .2), ("AK", .2), ("VT", .2), ("WY", .2)]
REGION = {}
for _s in "CT ME MA NH RI VT NJ NY PA".split():
    REGION[_s] = "Northeast"
for _s in "IL IN MI OH WI IA KS MN MO NE ND SD".split():
    REGION[_s] = "Midwest"
for _s in "DE FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX".split():
    REGION[_s] = "South"
for _s, _ in STATES:
    REGION.setdefault(_s, "West")

OCCUPATIONS = {
    "Healthcare": ["Physician", "Cardiothoracic surgeon", "Dentist", "Nurse practitioner", "Pharmacist", "Anesthesiologist"],
    "Technology": ["Software engineer", "Engineering director", "Product manager", "Data scientist", "IT consultant"],
    "Education": ["Teacher", "School principal", "Professor", "School administrator"],
    "Finance": ["Accountant", "Financial analyst", "Banker", "Insurance agent", "Hedge fund partner"],
    "Legal": ["Attorney", "Paralegal", "Judge"],
    "Business owner": ["Restaurant owner", "Retail owner", "Construction company owner", "Consulting firm owner"],
    "Engineering": ["Civil engineer", "Mechanical engineer", "Electrical engineer"],
    "Government": ["Federal employee", "Military officer", "City planner"],
    "Other": ["Real estate agent", "Pilot", "Sales executive", "Homemaker"],
}
CAT_WEIGHT = {"Healthcare": 12, "Technology": 14, "Education": 9, "Finance": 11, "Legal": 6,
              "Business owner": 16, "Engineering": 9, "Government": 8, "Other": 15}
# wealth multiplier and chance of holding a concentrated single stock, by category
CAT_WEALTH = {"Healthcare": 1.7, "Technology": 1.6, "Education": .6, "Finance": 1.8, "Legal": 1.5,
              "Business owner": 1.9, "Engineering": 1.0, "Government": .7, "Other": .9}
CAT_CONC = {"Healthcare": .25, "Technology": .45, "Education": .08, "Finance": .35, "Legal": .2,
            "Business owner": .4, "Engineering": .2, "Government": .05, "Other": .15}

SINGLE_STOCKS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "CRM", "ADBE"],
    "Healthcare": ["VRTX", "JNJ", "UNH", "LLY", "PFE", "ISRG"],
    "Financials": ["JPM", "BRK.B", "GS", "MS", "V"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer": ["AMZN", "PG", "COST", "HD", "NKE"],
}
FUNDS = {  # ticker: (name, group)
    "VTI": ("Vanguard Total Stock Market ETF", "Broad equity"),
    "VOO": ("Vanguard S&P 500 ETF", "Broad equity"),
    "VEA": ("Vanguard FTSE Developed Markets ETF", "Broad equity"),
    "SCHD": ("Schwab US Dividend Equity ETF", "Broad equity"),
    "QQQ": ("Invesco QQQ Trust", "Technology"),
    "BND": ("Vanguard Total Bond Market ETF", "Fixed income"),
    "AGG": ("iShares Core US Aggregate Bond ETF", "Fixed income"),
    "MUB": ("iShares National Muni Bond ETF", "Fixed income"),
    "TIP": ("iShares TIPS Bond ETF", "Fixed income"),
}
CAT_TO_SECTOR = {"Healthcare": "Healthcare", "Technology": "Technology", "Finance": "Financials"}

OBJECTIVES = ["Growth", "Growth with college savings", "Balanced growth and income",
              "Income with legacy planning", "Capital preservation", "Retirement income"]
RISKS = ["Conservative", "Moderate", "Aggressive"]

LIQUIDITY_REASONS = ["home renovation", "vacation home", "child's tuition", "business investment",
                     "wedding", "new car", "property purchase", "medical expenses reserve"]


@dataclass
class Holding:
    ticker: str
    name: str
    group: str
    single: bool
    weight: int  # percent


@dataclass
class Span:
    start: int
    end: int
    label: str        # PERSON, SSN, ACCOUNT, ROUTING, EMAIL, PHONE, DOB, ADDRESS, ORG
    text: str
    canonical: str    # value this mention refers to (for coreference)


@dataclass
class Client:
    client_id: str
    person_idx: int                 # row in the population directory
    # identity (lives only in the vault)
    name: str
    first: str
    last: str
    title: str
    dob: str
    ssn: str
    account: str
    routing: str
    email: str
    phone: str
    address: str
    employer: str
    spouse: Optional[str]
    children: list
    # profile (context store)
    age: int
    city: str
    state: str
    occupation: str
    occ_cat: str
    aum: float                      # millions of dollars
    household: str
    objective: str
    risk: str
    horizon: str
    liquidity: str
    liquidity_amount: int           # dollars
    holdings: list = field(default_factory=list)
    notes: str = ""
    gold_spans: list = field(default_factory=list)
    qi_mentions: dict = field(default_factory=dict)  # field -> level the raw note text reveals
    summary: str = ""

    def to_dict(self):
        d = asdict(self)
        return d


def _weighted(rng: random.Random, items):
    vals, ws = zip(*items)
    return rng.choices(vals, weights=ws, k=1)[0]


def _city_names(seed: int):
    fk = Faker("en_US")
    fk.seed_instance(seed)
    names = {}
    for s, _ in STATES:
        seen = []
        while len(seen) < 20:
            c = fk.city()
            if c not in seen:
                seen.append(c)
        names[s] = seen
    return names


def _make_holdings(rng: random.Random, occ_cat: str, age: int, risk: str):
    holdings = []
    remaining = 100
    if rng.random() < CAT_CONC[occ_cat]:
        sector = CAT_TO_SECTOR.get(occ_cat) or rng.choice(list(SINGLE_STOCKS))
        if rng.random() < .3:
            sector = rng.choice(list(SINGLE_STOCKS))
        t = rng.choice(SINGLE_STOCKS[sector])
        w = rng.randint(12, 55)
        holdings.append(Holding(t, f"{t} common stock", sector, True, w))
        remaining -= w
    cash = rng.randint(2, 12)
    remaining -= cash
    bond_share = {"Conservative": .6, "Moderate": .35, "Aggressive": .12}[risk] + max(0, age - 60) * .008
    bond_share = min(bond_share, .8)
    bond_w = int(round(remaining * bond_share))
    eq_w = remaining - bond_w
    eq_funds = rng.sample(["VTI", "VOO", "VEA", "SCHD", "QQQ"], k=2 if eq_w > 25 else 1)
    bd_funds = rng.sample(["BND", "AGG", "MUB", "TIP"], k=2 if bond_w > 25 else 1)
    for funds, total in ((eq_funds, eq_w), (bd_funds, bond_w)):
        if total <= 0:
            continue
        if len(funds) == 1:
            parts = [total]
        else:
            a = rng.randint(int(total * .4), int(total * .7))
            parts = [a, total - a]
        for t, w in zip(funds, parts):
            if w > 0:
                holdings.append(Holding(t, FUNDS[t][0], FUNDS[t][1], False, w))
    holdings.append(Holding("CASH", "Cash", "Cash", False, cash))
    holdings.sort(key=lambda h: -h.weight)
    return holdings


def holdings_summary(holdings):
    invested = [h for h in holdings if h.ticker != "CASH" and h.weight > 0]
    top = max(invested, key=lambda h: h.weight) if invested else next(h for h in holdings if h.ticker == "CASH")
    groups = {}
    for h in holdings:
        groups[h.group] = groups.get(h.group, 0) + h.weight
    gname, gw = max(groups.items(), key=lambda kv: kv[1])
    return dict(top_ticker=top.ticker, top_weight=top.weight, top_group=top.group, top_single=top.single,
                largest_group=gname, largest_group_weight=gw)


def make_population(n: int = 50_000, seed: int = 7) -> pd.DataFrame:
    """The attacker's reference directory: quasi-identifiers plus profile fields."""
    rng = random.Random(seed)
    cities = _city_names(seed)
    occ_items = []
    for cat, occs in OCCUPATIONS.items():
        for o in occs:
            occ_items.append(((o, cat), CAT_WEIGHT[cat] / len(occs)))
    rows = []
    for i in range(n):
        age = rng.randint(28, 88)
        state = _weighted(rng, STATES)
        city_idx = min(int(rng.paretovariate(1.2)) - 1, 19)
        occ, cat = _weighted(rng, occ_items)
        aum = float(np.exp(np.log(1.2 * CAT_WEALTH[cat]) + 1.0 * rng.gauss(0, 1) + (age - 50) * .012))
        risk = rng.choices(RISKS, weights=[.15 + max(0, age - 45) * .012, .5, .35 - max(0, age - 45) * .006])[0]
        if age >= 65:
            objective = rng.choice(["Retirement income", "Income with legacy planning", "Capital preservation"])
        elif age >= 50:
            objective = rng.choice(["Balanced growth and income", "Income with legacy planning", "Growth"])
        else:
            objective = rng.choice(["Growth", "Growth with college savings", "Balanced growth and income"])
        horizon = "20+ years" if age < 45 else "15+ years" if age < 60 else "About 10 years" if age < 75 else "5 to 10 years"
        holdings = _make_holdings(rng, cat, age, risk)
        hs = holdings_summary(holdings)
        rows.append(dict(person_idx=i, age=age, state=state, region=REGION[state], city=cities[state][city_idx],
                         occupation=occ, occ_cat=cat, aum=round(aum, 2), risk=risk, objective=objective,
                         horizon=horizon, holdings=holdings, **hs))
    return pd.DataFrame(rows)


# Note generation with ground-truth spans

class _Builder:
    def __init__(self):
        self.text = ""
        self.spans: list[Span] = []

    def add(self, template: str, slots: dict):
        """template uses {slot}; slots maps slot -> (value, label or None, canonical)."""
        i = 0
        while i < len(template):
            j = template.find("{", i)
            if j == -1:
                self.text += template[i:]
                break
            self.text += template[i:j]
            k = template.index("}", j)
            key = template[j + 1:k]
            value, label, canon = slots[key]
            start = len(self.text)
            self.text += value
            if label:
                self.spans.append(Span(start, start + len(value), label, value, canon))
            i = k + 1
        self.text += " "


def _phone_variant(rng, digits):
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return rng.choice([f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"{a}.{b}.{c}"])


def _dob_variant(rng, dob):
    y, m, d = dob.split("-")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    return rng.choice([f"{m}/{d}/{y}", f"{months[int(m) - 1]} {int(d)}, {y}"])


def _write_note(rng: random.Random, c: Client):
    b = _Builder()
    P = lambda v, canon=None: (v, "PERSON", canon or v)
    name_ref = lambda: rng.choice([P(c.name), P(f"{c.title} {c.last}", c.name), P(c.first, c.name)])
    qi = {}
    phone_digits = "".join(ch for ch in c.phone if ch.isdigit())

    b.add(rng.choice([
        "Met with {name} (DOB {dob}) at their home.",
        "Annual review with {name}, born {dob}.",
        "{name} (DOB {dob}) called to schedule a review.",
    ]), dict(name=P(c.name), dob=(_dob_variant(rng, c.dob), "DOB", c.dob)))

    if rng.random() < .7:
        b.add(rng.choice([
            "{ref} works at {employer} as a {occ}.",
            "{ref} recently retired from {employer} after {years} years as a {occ}.",
        ]), dict(ref=name_ref(), employer=(c.employer, "ORG", c.employer), occ=(c.occupation.lower(), None, None),
                 years=(str(rng.randint(12, 35)), None, None)))
        qi["occupation"] = "exact"

    if c.spouse and rng.random() < .7:
        sfirst = c.spouse.split()[0]
        b.add(rng.choice(["Spouse {sp} joined for the second half.", "{sp} asked about survivor benefits."]),
              dict(sp=rng.choice([P(c.spouse), P(sfirst, c.spouse)])))

    b.add("Goal: {obj}.", dict(obj=(c.objective.lower(), None, None)))
    b.add(f"Needs about ${c.liquidity_amount // 1000}K for a {c.liquidity.split('(')[-1].rstrip(')')}.", {})

    top = c.holdings[0] if c.holdings[0].ticker != "CASH" else c.holdings[1]
    if top.single and rng.random() < .75:
        b.add("Concerned about the {t} position at {w}% of the portfolio.",
              dict(t=(top.ticker, None, None), w=(str(top.weight), None, None)))
        qi["holdings"] = "exact"

    if rng.random() < .5:
        b.add("They are looking at property near {city}.", dict(city=(c.city, None, None)))
        qi["location"] = "city"

    ssn_digits = c.ssn.replace("-", "")
    b.add(rng.choice([
        "SSN {ssn} verified for the beneficiary form.",
        "Identity confirmed with SSN {ssn}.",
        "Confirmed last four of SSN {ssn4}.",
    ]), dict(ssn=(rng.choice([c.ssn, ssn_digits]), "SSN", c.ssn), ssn4=(ssn_digits[-4:], "SSN", c.ssn)))

    if rng.random() < .6:
        b.add("Wire instructions on file: routing {routing}, {acct}.",
              dict(routing=(c.routing, "ROUTING", c.routing), acct=(c.account, "ACCOUNT", c.account)))
    else:
        b.add("Transfer from {acct} is pending.", dict(acct=(c.account, "ACCOUNT", c.account)))

    if rng.random() < .6:
        b.add("Statements go to {addr}.", dict(addr=(c.address, "ADDRESS", c.address)))

    b.add(rng.choice(["Best reach is {phone} or {email}.", "Email {email}; cell {phone}."]),
          dict(phone=(_phone_variant(rng, phone_digits), "PHONE", c.phone), email=(c.email, "EMAIL", c.email)))

    if c.children and rng.random() < .4:
        kid = c.children[0]
        b.add("Add {kid} as contingent beneficiary.", dict(kid=P(kid)))
    return b.text.strip(), b.spans, qi


def make_clients(pop: pd.DataFrame, n: int = 1_000, seed: int = 11) -> list[Client]:
    """Sample a wealth-management book from the population (richer people more likely)."""
    rng = random.Random(seed)
    fk = Faker("en_US")
    fk.seed_instance(seed)
    w = np.sqrt(pop["aum"].values)
    idx = np.random.default_rng(seed).choice(len(pop), size=n, replace=False, p=w / w.sum())
    clients = []
    for k, pi in enumerate(idx):
        r = pop.iloc[int(pi)]
        female = rng.random() < .5
        first = fk.first_name_female() if female else fk.first_name_male()
        last = fk.last_name()
        title = rng.choice(["Ms.", "Mrs."]) if female else "Mr."
        married = r.age > 30 and rng.random() < .65
        spouse = f"{fk.first_name_male() if female else fk.first_name_female()} {last}" if married else None
        n_kids = rng.choice([0, 1, 2, 2, 3]) if r.age > 30 else 0
        children = [f"{fk.first_name()} {last}" for _ in range(n_kids)]
        birth_year = 2026 - int(r.age) - (1 if rng.random() < .4 else 0)
        dob = f"{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        area = {"CA": "415", "TX": "512", "FL": "305", "NY": "212"}.get(r.state, str(rng.randint(201, 989)))
        phone = f"({area}) 555-{rng.randint(100, 199):04d}"
        email = f"{first[0].lower()}.{last.lower()}{rng.randint(1, 99)}@example{rng.choice(['mail', 'net', 'post'])}.com"
        amount = rng.choice([25, 40, 60, 80, 120, 150, 250, 400, 600]) * 1000
        reason = rng.choice(LIQUIDITY_REASONS)
        months = rng.choice([3, 6, 9, 12, 18, 24])
        household = ("Married" if spouse else "Single") + (f", {n_kids} children" if n_kids else ", no children")
        if r.age >= 70 and not spouse:
            household = "Widowed" + (f", {n_kids} adult children" if n_kids else "")
        c = Client(
            client_id=f"C{k:04d}", person_idx=int(pi), name=f"{first} {last}", first=first, last=last, title=title,
            dob=dob, ssn=fk.ssn(), account=f"ACCT {rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
            routing=f"0{rng.randint(10000000, 99999999)}", email=email, phone=phone,
            address=f"{fk.building_number()} {fk.street_name()} {fk.street_suffix()}, {r.city}, {r.state} {fk.zipcode()}",
            employer=fk.company(), spouse=spouse, children=children,
            age=int(r.age), city=r.city, state=r.state, occupation=r.occupation, occ_cat=r.occ_cat, aum=float(r.aum),
            household=household, objective=r.objective, risk=r.risk, horizon=r.horizon,
            liquidity=f"About ${amount // 1000}K within {months} months ({reason})", liquidity_amount=amount,
            holdings=list(r.holdings),
        )
        c.notes, c.gold_spans, c.qi_mentions = _write_note(rng, c)
        c.summary = _summary(c)
        clients.append(c)
    return clients


def _summary(c: Client) -> str:
    """What a local summarizer inside the protected environment would produce."""
    top = c.holdings[0] if c.holdings[0].ticker != "CASH" else c.holdings[1]
    parts = [f"Objective is {c.objective.lower()}.",
             f"Needs about ${c.liquidity_amount // 1000}K of liquidity in the near term."]
    if top.single and top.weight >= 20:
        parts.append("Holds a concentrated single-stock position and is concerned about taxes on trimming it.")
    if c.spouse:
        parts.append("Spouse is involved in planning.")
    return " ".join(parts)


def make_stress_notes(clients: list[Client], seed: int = 99):
    """Harder notes written in formats the regexes and CRM dictionary were not built around.

    Typos and lowercase names, relatives the CRM does not know, spaced or prefixed
    SSNs, +1 phone formats, bare account digits, obfuscated emails, addresses
    without a ZIP, and short-form DOBs. Returns a list of (text, gold_spans).
    """
    rng = random.Random(seed)
    fk = Faker("en_US")
    fk.seed_instance(seed)
    out = []
    for c in clients:
        b = _Builder()
        P = lambda v, canon: (v, "PERSON", canon)
        variant = rng.choice(["lower", "typo", "nick"])
        if variant == "lower":
            nm = c.name.lower()
        elif variant == "typo":
            f = list(c.first)
            if len(f) > 3:
                i = rng.randrange(1, len(f) - 1)
                f[i], f[i + 1] = f[i + 1], f[i]
            nm = "".join(f) + " " + c.last
        else:
            nm = c.first[:3] + " " + c.last
        y, m, d = c.dob.split("-")
        b.add("Call with {nm}, dob {dob}.", dict(nm=P(nm, c.name), dob=(f"{int(m)}-{int(d)}-{y[2:]}", "DOB", c.dob)))
        rel = fk.first_name() + " " + fk.last_name()
        b.add("Her brother {rel} may become a co-trustee." if c.title != "Mr." else "His brother {rel} may become a co-trustee.",
              dict(rel=P(rel, rel)))
        sd = c.ssn.replace("-", "")
        b.add("SSN on file {ssn}.", dict(ssn=(rng.choice([f"{sd[:3]} {sd[3:5]} {sd[5:]}", f"#{sd}"]), "SSN", c.ssn)))
        pd_ = "".join(ch for ch in c.phone if ch.isdigit())
        b.add("Reach at {ph}.", dict(ph=(rng.choice([f"+1 {pd_[:3]} {pd_[3:6]} {pd_[6:]}", f"{pd_[:3]} {pd_[3:6]} {pd_[6:]}"]),
                                         "PHONE", c.phone)))
        acct_digits = "".join(ch for ch in c.account if ch.isdigit())
        b.add("Move funds out of acct {ac} next week.", dict(ac=(acct_digits, "ACCOUNT", c.account)))
        user, dom = c.email.split("@")
        b.add("Email is {em}.", dict(em=(f"{user} at {dom.replace('.', ' dot ')}", "EMAIL", c.email)))
        street = c.address.split(",")[0]
        b.add("Mail checks to {addr} in {city}.", dict(addr=(street, "ADDRESS", c.address), city=(c.city, None, None)))
        out.append((b.text.strip(), b.spans))
    return out


def load_environment(n_pop: int = 50_000, n_clients: int = 1_000, seed: int = 7):
    """Convenience: build the population and the client book together."""
    pop = make_population(n_pop, seed)
    clients = make_clients(pop, n_clients, seed + 4)
    return pop, clients
