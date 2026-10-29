# Evaluation

Reproduce with `python eval/run_all.py` (about 2 minutes; add `--no-presidio` to skip spaCy). Every number below comes from `eval/results/` for seed 7, 1,000 clients and a 50,000-person directory. Crowd size k is the number of directory entries that match everything a context reveals; k = 1 means the client is unique. The privacy floor used throughout is k of 10 or more.

## Setup

**Population directory.** 50,000 synthetic people with age, state, city (20 per state, skewed), occupation (36 in 9 categories), assets, risk tolerance, objective, horizon and a portfolio. Occupation drives wealth and the chance of holding a concentrated single stock; age drives risk, objective and the bond share. The directory stands in for what an outsider could know or buy.

**Client book.** 1,000 people sampled from the directory (wealthier people more likely), each with a synthetic identity, household, liquidity need and an advisor note. Notes are assembled from templates and record ground-truth spans for every identifier, plus which quasi-identifiers the note mentions in plain words (occupation, a ticker, a city).

**Stress notes.** A second note per client written in formats the detectors were not built around: typos and lowercase names, nicknames, a relative the CRM does not know, spaced or prefixed SSNs, +1 phone formats, bare account digits, emails written as "name at domain dot com", addresses without a ZIP, short DOBs.

**Tasks and utility.** Three advisor tasks (meeting brief, concentration check, retirement income estimate). A deterministic reference engine answers each task from whatever context it receives, and utility is scored against its full-context answer (agenda items covered, concentration metrics within tolerance, income estimate error). `eval/llm_eval.py` adds an LLM judge with real model answers when an API key is available.

**Sharing modes.** Full context (identifiers included), masking only (identifiers tokenized, every other field exact), task policy (a fixed generalization per task), and adaptive (per-client search for the best utility with k of 10 or more, optionally against the cumulative ledger).

## 1. PII detection

| Note set | Detector | Span recall | Precision | Notes with a leak | ms/note |
|---|---|---|---|---|---|
| template | regex | 66.2% | 100% | 100% | 0.1 |
| template | CRM dictionary | 54.2% | 100% | 100% | 0.7 |
| template | regex + CRM dictionary | 100% | 100% | 0% | 0.7 |
| template | Presidio (spaCy sm) | 69.4% | 87.2% | 100% | 21 |
| template | regex + CRM + Presidio | 100% | 87.1% | 0% | 22 |
| stress | regex + CRM dictionary | 6.5% | 99.6% | 100% | 0.6 |
| stress | Presidio | 57.0% | 64.7% | 100% | 18 |
| stress | regex + CRM + Presidio | 57.1% | 64.8% | 100% | 18 |

The template result is optimistic by construction: the regexes were written for the same formats the generator uses. The stress set is the honest number. Obfuscated emails (0%), short DOBs (1%) and addresses without a ZIP (9%) were missed by every detector; Presidio caught most names, phones, SSNs and bare account digits. A missed span is a leak, so the next step is running `--gliner nvidia/gliner-PII` (and GLiNER2-PII) on a machine with Hugging Face access and reporting the same table.

## 2. Split storage and key-based attacks

| Attack | Result |
|---|---|
| Look up the context store by an AI-log token | blocked (0 rows in 200 attempts) |
| Search a context dump for name, SSN, email, account, phone | blocked (0 matches across 1,000 values) |
| Join context to identity on a shared key | blocked (no identity column in the context store) |
| Read the vault as the LLM service role | blocked (no grant) |
| Link one client across two sessions by token | blocked (0 of 200 tokens identical) |
| Brute-force a plain SHA-256(SSN) key | **exposed**: recovered; the full SSN space takes about 19 CPU-minutes |
| Brute-force an HMAC token without the key | blocked (2^256 key space) |

Primary keys are random UUIDs and the only link sits in the vault, so a stolen context store cannot be joined back. A hashed SSN is not a safe key.

## 3. Re-identification from one request

Share of clients with k < 10 (share unique in brackets):

| Task | Masking only | Task policy | Adaptive (k of 10 or more) |
|---|---|---|---|
| Meeting brief | 100% (100%) | 3.4% (0.7%) | 0% (0%), utility 0.996 |
| Concentration check | 100% (100%) | 1.7% (0.1%) | 0% (0%), utility 0.998 |
| Retirement income | 100% (100%) | 47.4% (12.0%) | 0% (0%), utility 0.944 |

Masking alone fails completely. Fixed policies work for tasks that tolerate generalization, but the income task needs exact age and assets and leaves almost half the book identifiable. The adaptive search coarsens those fields only for the clients who need it and gives up 6% utility on that task. `eval/figures/frontier.png` shows every candidate policy.

## 4. Cumulative exposure across requests

Each client receives 12 requests with randomly chosen tasks. The ledger assumes the provider can link all requests about a client by content.

| After request | Task policy: k < 10 | Task policy: unique | Adaptive + ledger: k < 10 | Adaptive utility |
|---|---|---|---|---|
| 1 | 17.2% | 3.6% | 0% | 0.98 |
| 2 | 49.6% | 23.6% | 0% | 0.89 |
| 4 | 78.7% | 45.5% | 0% | 0.89 |
| 12 | 98.9% | 60.9% | 0% | 0.89 |

This is the central finding. Requests that are each safe compose into a unique fingerprint (exact age from the income task plus sector weights from the meeting brief plus position sizes from the concentration check). With the ledger, the minimizer plans each request against what has already been revealed: no refusals were needed, and utility settled at 0.89. Median cumulative k fell from 302 to 27, so the floor is doing the work.

## 5. Attribute inference

An attacker trains a gradient-boosted classifier on 20,000 directory entries and guesses attributes the task policy withheld.

| Task policy | Hidden attribute | Attacker accuracy | Majority guess |
|---|---|---|---|
| Concentration check | age band | 31.2% | 19.5% |
| Concentration check | occupation category | 24.8% | 17.7% |
| Meeting brief | occupation category | 22.5% | 17.7% |
| any | region | 36-37% | 36.7% |

Dropping a field does not make it unknowable: position sizes and risk tolerance carry age and occupation signal. The gains here are modest because the synthetic correlations are modest; real books are likely more correlated, so this should be rerun on LPL's sample data and with the LLM attacker in `eval/llm_eval.py`.

## 6. Proprietary strategy extraction

A planted model-portfolio rule has four secrets: glide-path knee (age 52), slope (1.1 points per year), the asset level that unlocks alternatives ($5M) and the concentration that triggers a collar (25%). The attacker queries with chosen profiles and fits a surrogate. At 2,000 queries:

| Defense | Surrogate equity error (pts) | Alternatives rule recovered | Collar rule recovered | Cost to honest advisors (pts) |
|---|---|---|---|---|
| none | 0.51 | 100% | 100% | 0 |
| noise (fresh each query) | 1.33 | 100% | 100% | 4.0 |
| noise, cached per profile | 2.32 | 100% | 100% | 4.0 |
| book-bound (own 150 clients only) | 1.20 | 99.6% | 81.1% | 0 |
| coarse (rounded, rule outputs hidden) | 1.68 | 56% (chance) | 60% (chance) | 2.2 |
| book-bound + coarse | 2.64 | 56% (chance) | 60% (chance) | 2.2 |

Numeric noise only slows extraction, and averaging repeated queries defeats it unless answers are cached. Hiding the discrete rule outputs is what protects the thresholds. Binding queries to the caller's real clients removes the attacker's ability to probe edge cases and is free for honest users.

## 7. Differentially private cohort statistics

"How does this client compare with similar clients?" leaks through a differencing attack (average with and without one client).

| Cohort | epsilon | Differencing attack error (pts) | Blind guess error | Cohort mean error |
|---|---|---|---|---|
| 50 | exact | 0.0 | 13.3 | 0.0 |
| 50 | 1.0 | 46.1 | 13.3 | 1.95 |
| 500 | 1.0 | 44.1 | 16.2 | 0.20 |
| 500 | 5.0 | 24.6 | 16.2 | 0.04 |

Exact cohort averages reveal a member's value exactly. With epsilon of 1 the attack is worse than guessing while the average itself stays within 2 points for a cohort of 50.

## 8. Rehydration robustness

| Token corruption by the model | Restored | Flagged for review | Silently wrong |
|---|---|---|---|
| none | 100% | 0% | 0% |
| lowercase | 100% | 0% | 0% |
| truncated by 2 hex chars | 100% | 0% | 0% |
| brackets dropped | 100% | 0% | 0% |
| invented token | 0% | 100% | 0% |

## 9. Overhead

Mean per request with regex + CRM detection: detection 0.5 ms, session re-keying 0.14 ms, adaptive policy search 0.44 ms, rehydration 0.02 ms. Presidio adds about 20 ms per note. All of it is small next to an LLM call.

## Limitations and next steps

* Synthetic data: correlations, note styles and the directory are assumptions. Rerun on LPL's approved samples.
* The ledger assumes full linkability, which is conservative. Measure how often requests are actually linkable by content.
* Utility is a deterministic proxy. Run `eval/llm_eval.py` for LLM-judged quality and an LLM attacker.
* GLiNER detectors were not run here (no Hugging Face access in the build environment).
* Free-text liquidity and objective strings were treated as non-identifying; they can still act as linkage fingerprints.
