# Evaluation

Two environments run the same experiments:

* **Public data (primary):** `python3 eval/run_real.py`, results in `eval/results_real/`. Census ACS PUMS 2023 (978,284 adults in NY, NJ, CA, TX, FL, IL) is the attacker's directory; each person gets wealth and a portfolio mix from a Fed SCF 2022 household in the same age, income and occupation cell; 1,000 clients come from Financial-Advisory-Clients (4,825 usable rows) and are placed on matching Census people; request types come from Bitext (23,806 requests, 24 intents); PII detection is also benchmarked on 2,891 Gretel financial documents.
* **Synthetic (secondary):** `python3 eval/run_all.py`, results in `eval/results/`, a fully generated 50,000-person world kept for tests.

Crowd size **k** is the number of directory people who match everything a context reveals; k = 1 means the client is unique. The privacy floor is k of 10 or more. Names, SSNs, accounts and contact details are always generated.

## Results on public data

### 1. Masking alone does not protect clients

Share of clients with k < 10 after one request (share unique in brackets), against the 978K-person Census directory:

| Task (share of real requests) | Masking only | Task policy | Adaptive |
|---|---|---|---|
| General question (58.4%) | 99.9% (96.4%) | 0% | 0% |
| Portfolio review (16.5%) | 99.9% (96.4%) | 5.6% (3.5%) | 0%, utility 0.99 |
| Transfer or withdrawal (8.4%) | 99.9% (96.4%) | 0% | 0% |
| Retirement income (8.4%) | 99.9% (96.4%) | 0.3% (0%) | 0%, utility 1.00 |
| Meeting brief (4.2%) | 99.9% (96.4%) | 0.6% (0.4%) | 0%, utility 0.999 |
| Concentration check (4.2%) | 99.9% (96.4%) | 0.8% (0.6%) | 0%, utility 0.997 |

With names and accounts tokenized but everything else exact, 96.4% of clients are still unique among almost a million real people. **58% of real wealth-management requests need no client data at all** (Bitext intents such as fees, services, account setup), so the biggest single win is not sending context for them.

### 2. Exposure accumulates across requests

Twelve requests per client drawn from the real Bitext request mix; the ledger assumes the provider can link requests by content.

| After request | Task policy: k < 10 | Task policy: unique | Adaptive + ledger: k < 10 | Adaptive utility |
|---|---|---|---|---|
| 1 | 1.2% | 0.7% | 0% | 0.997 |
| 4 | 5.6% | 3.2% | 0% | 0.996 |
| 12 | 15.9% | 8.9% | 0% | 0.994 |

With only the three data-heavy tasks (meeting brief, concentration, income plan) the fixed policies reach 22.4% with k < 10 after 12 requests; the ledger holds 0% at utility 0.988. No request was refused in either case.

### 3. Who the attacker compares against matters most

Same task policies, three reference populations:

| Task | General population (978K) | Firm's client book (1,000) | One advisor's book (150) |
|---|---|---|---|
| Meeting brief: k < 10 | 0.6% | 6.4% | 36.0% |
| Concentration check: k < 10 | 0.8% | 12.4% | 31.3% |
| Retirement income: k < 10 | 0.3% | 80.7% | 82.7% |

A policy that looks safe against the general population can identify most clients to anyone who already knows they are clients of a given advisor (the advisor's assistant, a vendor serving one branch, an insider). The k floor should be set against the firm's own client base. This is the strongest argument for running minimization inside LPL, not at the model provider.

### 4. PII detection on third-party documents

2,891 Gretel financial documents (direct identifiers only: names, SSNs, accounts, contact details, IDs):

| Detector | Direct-identifier recall | Precision | Documents with a leak | ms/doc |
|---|---|---|---|---|
| Regex | 11.3% | 46.0% | 74.5% | 0.4 |
| Presidio (spaCy small) | 43.4% | 37.5% | 68.0% | 43 |
| Regex + Presidio | 46.1% | 37.3% | 66.7% | 41 |
| nvidia/gliner-PII (hosted) | pending: `scripts/run_with_nvidia.sh` | | | |

By type (regex + Presidio): emails 92%, SSNs 91%, credit cards 98%, IBANs 100%, but person names 62% and street addresses only 9%. On our own client notes, regex plus the CRM dictionary catch 100% of identifiers in the expected formats, but only 6% in the stress notes (typos, nicknames, unknown relatives, obfuscated emails); adding Presidio raises that to 56%. Rule-based detection is not enough on its own; the CRM dictionary covers known clients, and a model such as GLiNER is needed for everything else.

### 5. Other exposure channels

**Transactions are the most identifying data.** In the client book, one exact transaction (date and amount) singles out 99% of clients. Coarsening helps a lot: with month and an amount band, 5 known transactions single out 12%; adding the transaction type raises that to 55%. Transaction tasks should receive monthly aggregates, not line items.

**Exact amounts are fingerprints.** Among 4,825 Advisory clients, 89% have a unique exact portfolio value and 99% a unique exact goal target; rounding to $10K brings these to 0.9% and 5.7%, and buckets to 0%. Goal target to $10K plus horizon plus goal type is still unique for 39%.

**Advisor queries.** All 1,000 simulated advisor requests contained a direct identifier (name or account) typed by the advisor; after the query sanitizer none remained, and the share with k < 10 from typed details (age, state, occupation, rough assets) fell from 1.7% to 0%.

**Policy side channel.** The adaptive policy's choices predict which clients are rare (AUC 0.66 to 1.00, though rare clients are only 0.6% to 8% of the book). Using one uniform policy per task (the most generalized level any client needed) removes the signal but costs utility: 0.875 for meeting briefs, 0.43 for concentration, 0.59 for income estimates. In practice the generalization level should be chosen per cohort, not per client.

**Attribute inference.** Trained on 20,000 Census people, an attacker guesses the withheld occupation group 19 to 23% of the time against a 13% majority baseline (income and assets carry the signal). Region and age band were not inferable above chance.

### 6. Split storage and keys

Every key-based attack was blocked: AI-log tokens find nothing in the context store, a context dump contains no identity column or value, the LLM role cannot read the vault, and the same client's tokens never repeat across sessions. A plain SHA-256 of the SSN used as a key was brute-forced (the full SSN space takes about 14 CPU-minutes); HMAC tokens without the key yield nothing.

### 7. Proprietary strategy extraction

A planted allocation rule (glide-path knee, an asset level that unlocks alternatives, a concentration level that triggers a collar), attacked with 2,000 queries:

| Defense | Surrogate error (pts) | Alternatives rule recovered | Collar rule recovered | Cost to honest advisors (pts) |
|---|---|---|---|---|
| none | 0.5 | 100% | 100% | 0 |
| noise, fresh per query | 1.3 | 100% | 100% | 4.0 |
| noise, cached per profile | 2.3 | 100% | 100% | 4.0 |
| coarse answers | 1.7 | 56% (chance) | 60% (chance) | 2.2 |
| book-bound (own 150 real clients only) | 6.5 | 56% (chance) | 100% | 0 |
| book-bound + coarse | 6.9 | 56% (chance) | 60% (chance) | 2.2 |

With real client books, binding queries to the advisor's own clients is the strongest free defense: real books do not cover the edge cases an attacker needs. Combined with coarse answers it hides every rule tested.

### 8. Differential privacy, rehydration, overhead

* Exact cohort averages let a differencing attack recover a member's equity share exactly; with epsilon of 1 the attack error (41 points) is worse than a blind guess (25 points) while the cohort average moves by 2 points for 50 clients and 0.2 for 500.
* Rehydration restored 100% of lowercase, truncated and bracket-less tokens and flagged 100% of invented tokens; none were silently wrong.
* Per request: detection 0.6 ms, session re-keying 0.15 ms, policy search 0.6 ms, rehydration 0.02 ms (Presidio adds about 40 ms per document).

### 9. Real-model answers and the LLM attacker

pending: `scripts/run_with_nvidia.sh` (hosted Llama 3.1 70B answers 15 clients x 3 tasks x 4 sharing modes, grades each against the full-context answer, and tries to guess hidden attributes).

## Results on synthetic data

See `eval/results/`. The synthetic world was built to stress the system (wealthier clients, more concentrated positions, a 50,000-person directory), so its numbers are harsher, but the conclusions match: masking alone leaves every client unique, fixed task policies compound to 68% unique after 12 requests, and the adaptive policy with a ledger holds k of 10 or more at about 0.88 utility.

## Limitations

* No LPL data: notes, holdings, transactions, request logs and strategies are public or generated stand-ins. See [FUTURE_WORK.md](FUTURE_WORK.md).
* The directory joins two real sources statistically (Census geography and occupation, SCF wealth), so wealth is realistic in distribution but not tied to real individuals.
* Census income is personal income; SCF income is household income. Cells use the same cut points for both.
* The ledger assumes every request about a client is linkable; real linkability is likely lower.
* Advisor queries are built by inserting client details into Bitext requests.
* Gretel's labels were partly produced with GLiNER, which may favor GLiNER-family detectors on that benchmark.

### Groq evaluation

Set `GROQ_API_KEY` in your shell, then run `bash scripts/run_with_groq.sh`.
The runner uses `openai/gpt-oss-20b` for prompted PII extraction and
`openai/gpt-oss-120b` for answers, grading and attribute inference. Override these
with `GROQ_DETECT_MODEL` and `GROQ_LLM_MODEL`. Model access and quotas depend on
your Groq account; this runner does not guarantee free usage.

The `groq-llm` detector is a prompted LLM, not GLiNER. It extracts exact text
matches into the existing span schema. This benchmark sends public dataset
text to Groq. Results should be attributed to the selected models, not to NVIDIA.

Successful responses are checkpointed in ignored `data/cache/groq/`, keyed by
model, prompts and generation settings. Evaluation-only session keys persist
in `data/cache/eval_sessions/` so pseudonymized prompts remain stable on resume.
Production sessions continue to use fresh random keys. Interrupted runs can
resume by running the same command. Keep the cache private: responses and
benchmark session keys are stored locally. Delete these cache directories to
force fresh requests. Timing measurements on resumed runs include cache hits
and should not be interpreted as uncached API latency.

The runner checks access to both models before evaluating, and backs up previous
outputs under `data/cache/run_backups/`. Tests use mocks and do not write to the
real evaluation result folders.
