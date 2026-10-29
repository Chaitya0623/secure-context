"""Client Context Shield: interactive demo of the secure-context pipeline.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from secure_context.data import load_environment  # noqa: E402
from secure_context.detect import CRMDictionaryDetector, EnsembleDetector, RegexDetector, resolve_coreference  # noqa: E402
from secure_context.llm import get_backend  # noqa: E402
from secure_context.pipeline import Pipeline  # noqa: E402
from secure_context.policy import ALL_EXACT, FIELDS, QI_FIELDS, TASKS, coverage, render_context  # noqa: E402
from secure_context.privacy import ExposureLedger, KEngine, choose_policy  # noqa: E402
from secure_context.strategy import DEFENSES, extraction_curve, honest_user_error  # noqa: E402
from secure_context.tasks import SYSTEM_PROMPT, build_prompt, utility  # noqa: E402
from secure_context.tokens import Session, rehydrate  # noqa: E402
from secure_context.vault import SplitStorage  # noqa: E402

st.set_page_config(page_title="Client Context Shield", layout="wide")

COLORS = {"PERSON": "#B5541B", "SSN": "#B23A30", "ACCOUNT": "#B23A30", "ROUTING": "#B23A30", "EMAIL": "#7A4BA0",
          "PHONE": "#7A4BA0", "DOB": "#9A6B00", "ADDRESS": "#1E7A86", "ORG": "#5B6F1E"}
st.markdown("""<style>
.box{background:rgba(127,127,127,.07);border:1px solid rgba(127,127,127,.25);border-radius:8px;padding:12px 14px;
     line-height:1.75;white-space:pre-wrap;font-size:14px}
.hl{border-radius:3px;padding:0 3px;font-weight:600}
.tok{font-family:monospace;font-size:12.5px;color:#22704C;background:rgba(34,112,76,.12);border-radius:3px;padding:0 3px}
.bad{color:#B23A30;background:rgba(178,58,48,.12)}
.rest{border-bottom:2px solid #1E5B86;font-weight:600}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Generating synthetic clients and a 50,000-person directory...")
def environment(n_pop=50_000, n_clients=300):
    pop, clients = load_environment(n_pop, n_clients)
    storage = SplitStorage(EnsembleDetector([RegexDetector(), CRMDictionaryDetector()]))
    ids = [storage.ingest(c) for c in clients]
    return pop, clients, storage, ids, KEngine(pop)


pop, clients, storage, ctx_ids, engine = environment()
by_label = {f"{c.client_id} · {c.name} ({c.occupation}, {c.city}, {c.state})": i for i, c in enumerate(clients)}

if "session" not in st.session_state:
    st.session_state.session = Session()
    st.session_state.sessions = 1
if "ledgers" not in st.session_state:
    st.session_state.ledgers = {"default": ExposureLedger(engine), "adaptive": ExposureLedger(engine)}
    st.session_state.log = []


def esc(s):
    return html.escape(str(s))


def mark_spans(text, spans):
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        out.append(esc(text[pos:s.start]))
        out.append(f'<span class="hl" style="color:{COLORS.get(s.label, "#555")};background:{COLORS.get(s.label, "#555")}22" '
                   f'title="{s.label} via {s.detector}">{esc(text[s.start:s.end])}</span>')
        pos = s.end
    out.append(esc(text[pos:]))
    return "".join(out)


TOK_HTML = re.compile(r"&lt;([A-Z]+_[0-9a-f]{3,6})&gt;")


def mark_tokens(text, session):
    def rep(m):
        raw = f"<{m.group(1)}>"
        return f'<span class="tok{"" if raw in session.vault else " bad"}">{esc(raw)}</span>'
    return TOK_HTML.sub(rep, esc(text))


def box(inner):
    st.markdown(f'<div class="box">{inner}</div>', unsafe_allow_html=True)


# Sidebar

st.sidebar.title("Client Context Shield")
st.sidebar.caption("All clients, notes and the population directory are synthetic.")
choice = st.sidebar.selectbox("Client", list(by_label))
c = clients[by_label[choice]]
ctx_id = ctx_ids[by_label[choice]]
task = st.sidebar.selectbox("Advisor task", list(TASKS), format_func=lambda t: TASKS[t]["name"])
mode = st.sidebar.radio("Context sharing", ["adaptive", "default", "masked", "custom"],
                        format_func={"adaptive": "Adaptive (per client, k budget)", "default": "Task policy",
                                     "masked": "Masking only", "custom": "Custom levels"}.get)
k_min = st.sidebar.slider("Minimum crowd size k", 2, 100, 10)
if st.sidebar.button("New session key"):
    st.session_state.session = Session()
    st.session_state.sessions += 1
session: Session = st.session_state.session
st.sidebar.caption(f"Session #{st.session_state.sessions} · key fingerprint `{session.fingerprint()}`")

def _secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return None


backend_name = "simulated"
if _secret("ANTHROPIC_API_KEY") and _secret("LLM_MODEL"):
    os.environ["ANTHROPIC_API_KEY"] = _secret("ANTHROPIC_API_KEY")
    os.environ["LLM_MODEL"] = _secret("LLM_MODEL")
    backend_name = st.sidebar.selectbox("Model", ["simulated", "anthropic"])
try:
    llm = get_backend(backend_name)
except Exception as e:  # missing key or package
    st.sidebar.warning(f"Using the simulated model: {e}")
    llm = get_backend("simulated")

# Levels for this request
best = None
if mode == "masked":
    levels = dict(ALL_EXACT)
elif mode == "default":
    levels = dict(TASKS[task]["default"])
elif mode == "adaptive":
    best = choose_policy(task, c, engine, k_min)
    levels = best[0] if best else {k: "drop" for k in FIELDS}
else:
    levels = {}
    st.sidebar.markdown("**Custom levels**")
    for k, f in FIELDS.items():
        d = TASKS[task]["default"][k]
        levels[k] = st.sidebar.select_slider(f.label, options=list(f.levels), value=d, key=f"lv_{k}")

tab_pipe, tab_ledger, tab_eval, tab_strat = st.tabs(["Pipeline", "Exposure over time", "Evaluation", "Strategy extraction"])

# Pipeline tab

STEPS = {
    "1 Raw record": "The client as the CRM holds it. Everything here would reach the AI provider with no protection.",
    "2 Detect PII": "Regex finds structured IDs; the CRM dictionary finds names, relatives, employer and address it already "
                    "knows; coreference maps every mention to one canonical value. Production adds GLiNER for text the CRM "
                    "does not know.",
    "3 Split storage": "Identity sits in an encrypted vault, context in a separate store. Both use random UUID keys and only "
                       "the vault holds the link, so a context dump cannot be joined back to a person.",
    "4 Dehydrate": "Each value becomes HMAC(session key, type|value). Same value, same token inside a session; a new session "
                   "changes every token, so logs cannot be linked across sessions.",
    "5 Minimize": "Pick how much detail each field carries for this task. Adaptive mode searches the levels for the best "
                  "task utility whose crowd size k stays above your minimum.",
    "6 LLM call": "Only this prompt crosses the boundary.",
    "7 Rehydrate": "Tokens are swapped back for the advisor. Unknown or mangled tokens are repaired only when the match is "
                   "unique, otherwise flagged.",
}
with tab_pipe:
    step = st.radio("Pipeline step", list(STEPS), horizontal=True, label_visibility="collapsed")
    st.info(STEPS[step])
    spans = resolve_coreference(storage.detector.detect(c.notes, c), c)
    notes_tok = storage.session_notes(ctx_id, session)
    left, right = st.columns(2)

    if step.startswith("1"):
        with left:
            st.subheader("Protected: client record")
            st.json({k: getattr(c, k) for k in ("name", "dob", "ssn", "account", "routing", "email", "phone", "address",
                                                "employer", "spouse", "children")}, expanded=False)
            st.dataframe(pd.DataFrame([{**render_context(c, ALL_EXACT)}]).T.rename(columns={0: "value"}),
                         width="stretch")
            box(esc(c.notes))
        with right:
            st.subheader("If sent as-is")
            k1 = engine.k(c, ALL_EXACT)
            st.metric("Crowd size k with masking only", k1, help="People in the 50,000 directory who match")
            box(esc(f"Client: {c.name}, DOB {c.dob}, SSN {c.ssn}\n{c.email}, {c.phone}\n{c.address}\n\n{c.notes}"))
    elif step.startswith("2"):
        with left:
            st.subheader("Protected: detections")
            box(mark_spans(c.notes, spans))
        with right:
            st.subheader("Detected spans")
            st.dataframe(pd.DataFrame([dict(text=s.text, type=s.label, detector=s.detector, resolves_to=s.canonical)
                                       for s in spans]), width="stretch", hide_index=True)
    elif step.startswith("3"):
        with left:
            st.subheader("Identity vault (rehydrator role only)")
            st.code(f"ctx_link: {ctx_id} -> vault_id (random UUID)\nidentity: name, address, email, phone (plain)\n"
                    f"          ssn, account, routing, dob (AES-GCM sealed)", language="text")
            st.dataframe(pd.DataFrame(storage.vault.audit_log()[-8:], columns=["ts", "role", "action", "ctx_id"]),
                         width="stretch", hide_index=True)
        with right:
            st.subheader("Context store (no identifiers)")
            rec = storage.store.get(ctx_id, role="context_service")
            st.json({"ctx_id": rec["ctx_id"], "profile": rec["profile"]}, expanded=False)
            box(esc(rec["notes_storage_tokens"]))
            st.caption("Notes are stored with long-lived storage tokens and re-keyed per session before any prompt.")
    elif step.startswith("4"):
        with left:
            st.subheader("Token map (vault only)")
            st.dataframe(pd.DataFrame([dict(token=t, type=v[0], value=v[1]) for t, v in session.vault.items()]),
                         width="stretch", hide_index=True)
        with right:
            st.subheader("Session-keyed note")
            box(mark_tokens(notes_tok, session))
    elif step.startswith("5"):
        k = engine.k(c, levels)
        u = utility(task, c, levels)
        cov, missing = coverage(levels, task)
        with left:
            st.subheader("Policy")
            st.dataframe(pd.DataFrame([dict(field=f.label, level=levels[k_], quasi_identifier="yes" if f.qi else "",
                                            task_needs=TASKS[task]["needs"].get(k_, ""))
                                       for k_, f in FIELDS.items()]), width="stretch", hide_index=True)
            if mode == "adaptive" and not best:
                st.error("No policy reaches this k with any utility; the request would be refused.")
        with right:
            st.subheader("Context the AI receives")
            a, b, cc = st.columns(3)
            a.metric("Crowd size k", f"{k:,}")
            b.metric("Task utility", f"{u:.2f}")
            cc.metric("Needs covered", f"{cov:.0%}")
            st.json(render_context(c, levels, notes_tok))
    else:
        prompt = build_prompt(task, c, levels, session, notes_tok)
        answer = llm.complete(SYSTEM_PROMPT, prompt, task=task, client=c, levels=levels, session=session)
        if step.startswith("6"):
            with left:
                st.subheader("Prompt sent")
                box(mark_tokens(prompt, session))
            with right:
                st.subheader(f"Model answer ({llm.name})")
                box(mark_tokens(answer, session))
        else:
            r = rehydrate(answer, session)
            with left:
                st.subheader("Answer with tokens")
                box(mark_tokens(answer, session))
            with right:
                st.subheader("Advisor view")
                st.caption(f"{r.resolved} tokens restored · {len(r.repaired)} repaired · {len(r.unresolved)} flagged")
                box(esc(r.text))

# Exposure ledger tab

with tab_ledger:
    st.write("Send several requests about the same client. A model provider can link requests by their content even "
             "when tokens rotate, so what matters is everything revealed so far. The fixed task policy and the adaptive "
             "policy with an exposure budget run side by side.")
    cols = st.columns(len(TASKS) + 1)
    for i, t in enumerate(TASKS):
        if cols[i].button(f"Send: {TASKS[t]['name']}"):
            L = st.session_state.ledgers
            lv_d = TASKS[t]["default"]
            L["default"].record(c, lv_d, t)
            best_a = choose_policy(t, c, engine, k_min, L["adaptive"])
            if best_a:
                L["adaptive"].record(c, best_a[0], t)
            st.session_state.log.append(dict(client=c.client_id, task=TASKS[t]["name"],
                                             fixed_k=L["default"].k_now(c), fixed_utility=round(utility(t, c, lv_d), 2),
                                             adaptive_k=L["adaptive"].k_now(c),
                                             adaptive_utility=round(best_a[1], 2) if best_a else 0.0,
                                             adaptive_refused=best_a is None))
    if cols[-1].button("Reset"):
        st.session_state.ledgers = {"default": ExposureLedger(engine), "adaptive": ExposureLedger(engine)}
        st.session_state.log = []
    log = pd.DataFrame([r for r in st.session_state.log if r["client"] == c.client_id])
    if len(log):
        log.index = np.arange(1, len(log) + 1)
        st.line_chart(log[["fixed_k", "adaptive_k"]].clip(lower=1), y_label="cumulative crowd size k")
        st.dataframe(log, width="stretch")
    else:
        st.caption("No requests yet for this client.")

# Evaluation tab

RES, FIG = ROOT / "eval" / "results", ROOT / "eval" / "figures"
with tab_eval:
    if not (RES / "summary.json").exists():
        st.warning("Run `python eval/run_all.py` to generate results.")
    else:
        s = json.loads((RES / "summary.json").read_text())
        st.caption(f"{s['clients']:,} synthetic clients · {s['population']:,}-person directory · k_min = {s['k_min']}")
        sr = pd.DataFrame(s["single_request"])
        st.subheader("One request: masking vs. task policy vs. adaptive")
        st.dataframe(sr.pivot(index="mode", columns="task", values="pct_k_lt_10").style.format("{:.0%}"),
                     width="stretch")
        st.caption("Share of clients whose context matches fewer than 10 people.")
        for name, title in [("frontier.png", "Privacy-utility frontier"), ("cumulative.png", "Cumulative exposure"),
                            ("detection_recall.png", "PII detection"), ("strategy_extraction.png", "Strategy extraction")]:
            if (FIG / name).exists():
                st.subheader(title)
                st.image(str(FIG / name))
        st.subheader("Key-based attacks against split storage")
        st.dataframe(pd.read_csv(RES / "key_attacks.csv"), width="stretch", hide_index=True)
        st.subheader("Attribute inference from what was shared")
        st.dataframe(pd.read_csv(RES / "attribute_inference.csv").round(3), width="stretch", hide_index=True)

# Strategy tab

with tab_strat:
    st.write("A planted model-portfolio rule (glide-path knee, an asset level that unlocks alternatives, a concentration "
             "level that triggers a collar). An attacker queries the assistant with chosen profiles and trains a "
             "surrogate. Lower surrogate error means more of the strategy leaked.")
    defense = st.selectbox("Defense", DEFENSES)
    q = st.select_slider("Attacker queries", options=[25, 50, 100, 200, 500, 1000, 2000], value=500)

    @st.cache_data
    def curve():
        return extraction_curve(pop)
    cv = curve()
    row = cv[(cv.defense == defense) & (cv.queries == q)].iloc[0]
    a, b, cc, d = st.columns(4)
    a.metric("Surrogate equity error", f"{row.equity_mae:.1f} pts")
    b.metric("'Alternatives' rule recovered", f"{row.alt_rule_acc:.0%}")
    cc.metric("'Collar' rule recovered", f"{row.collar_rule_acc:.0%}")
    d.metric("Cost to honest advisors", f"{honest_user_error()[defense]:.1f} pts")
    st.line_chart(cv.pivot(index="queries", columns="defense", values="equity_mae"), y_label="surrogate error (pts)")
