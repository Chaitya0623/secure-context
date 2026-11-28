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
| General question (58.4%) | 100% (98.1%) | 0% | 0% |
| Portfolio review (16.5%) | 100% (98.1%) | 5.6% (3.5%) | 0%, utility 0.99 |
| Transfer or withdrawal (8.4%) | 100% (98.1%) | 0% | 0% |
| Retirement income (8.4%) | 100% (98.1%) | 0.3% (0%) | 0%, utility 1.00 |
| Meeting brief (4.2%) | 100% (98.1%) | 0.7% (0.6%) | 0%, utility 0.999 |
| Concentration check (4.2%) | 100% (98.1%) | 0.8% (0.6%) | 0%, utility 0.997 |

With names and accounts tokenized but everything else exact, 98.1% of clients are still unique among almost a million real people. **58% of real wealth-management requests need no client data at all** (Bitext intents such as fees, services, account setup), so the biggest single win is not sending context for them.

Household (marital status and dependents) counts toward k as of this version. Census records marital status, so it narrows the crowd for the general population; only the firm's own book also knows dependents. Before this change the meeting brief's household line was sent but not counted, which understated identifiability (see [COMBINATIONS.md](COMBINATIONS.md), section 2).

### 2. Exposure accumulates across requests

Twelve requests per client drawn from the real Bitext request mix; the ledger assumes the provider can link requests by content.

| After request | Task policy: k < 10 | Task policy: unique | Adaptive + ledger: k < 10 | Adaptive utility |
|---|---|---|---|---|
| 1 | 1.2% | 0.7% | 0% | 0.997 |
| 4 | 6.8% | 3.4% | 0% | 0.996 |
| 12 | 20.8% | 10.6% | 0% | 0.993 |

With only the three data-heavy tasks (meeting brief, concentration, income plan) the fixed policies reach 38.6% with k < 10 after 12 requests; the ledger holds 0% at utility 0.981. No request was refused in either case.

### 3. Who the attacker compares against matters most

Same task policies, three reference populations:

| Task | General population (978K) | Firm's client book (1,000) | One advisor's book (150) |
|---|---|---|---|
| Meeting brief: k < 10 | 0.7% | 50.4% | 100% |
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

Run on Gemini on Vertex AI (2026-11-20): 30 clients x 3 tasks x 5 sharing modes. `gemini-3.8-flash` answers and plays the attacker; a separate model, `gemini-3.1-pro-preview`, grades each answer 1 to 5 against the full-context answer, so no model grades its own output. Command: `GOOGLE_CLOUD_PROJECT=<project> python3 eval/llm_eval.py --backend gemini --n 30 --judge-model gemini-3.1-pro-preview`. Files: `eval/results_real/llm_*.csv`.

**Answer quality vs. the full-context answer** (5 = same advice; 95% bootstrap intervals over 90 client-task pairs):

| Sharing mode | Meeting brief | Concentration | Income plan | All tasks |
|---|---|---|---|---|
| Full context (reference) | 5 | 5 | 5 | 5 |
| Masked (tokens, all fields exact) | 4.70 | 4.80 | 4.80 | 4.77 (4.64 to 4.88) |
| Task policy | 2.83 | 2.67 | 3.73 | 3.08 (2.91 to 3.26) |
| Adaptive policy | 2.83 | 2.53 | 3.87 | 3.08 (2.88 to 3.29) |
| No client data (baseline) | 1.00 | 1.00 | 2.40 | 1.47 (1.29 to 1.66) |

* Masking names, accounts and identifiers costs little: 73 of 90 answers were graded identical, and none scored below 2.
* Minimizing the fields costs about 1.7 points. Most minimized answers score 3 ("same direction, missing specifics").
* The minimized context is doing real work: the task policy beats the no-data baseline by 1.61 points on matched pairs (95% interval 1.47 to 1.74; better in 84 of 90, never worse). Without client data the model can only give generic advice, which the judge scores 1 on meeting briefs and concentration checks.
* Adaptive vs. task policy: +0.00 on matched pairs (95% interval -0.12 to +0.12; better in 10, same in 69, worse in 11). With a separate judge and 30 clients, the adaptive policy's extra detail does not measurably improve answers; its value is the larger crowd it keeps when requests add up (section 2).
* No requests were refused, no tokens failed to rehydrate, and the client's name reached the model only in full-context mode.

**LLM attacker** (sees the task-policy context, guesses hidden attributes; majority-guess baseline in brackets):

| Task | Occupation group [0.29] | Region [0.39] | Age band [0.19] |
|---|---|---|---|
| Meeting brief | 0.37 | 0.40 | 1.00 |
| Concentration | 0.13 | 0.13 | 0.07 |
| Income plan | 0.23 | 0.23 | 1.00 |

Age band is recovered every time where the policy sends age (band or exact), so that hit is by design. Meeting-brief notes let the model edge past the baseline on occupation (0.37 vs 0.29) and match it on region. The concentration and income-plan contexts leak nothing beyond the baseline.

Changes from the earlier 15-client run (same model as answerer and grader): minimized scores fell by about 0.1 to 0.3, and the small adaptive advantage (+0.13) disappeared. A stricter, separate judge and the larger sample are the likely causes; we did not run the two setups side by side to separate them.

Caveats: full context is the reference by construction; the judge is from the same model family as the answerer; 30 clients is still modest; the grader has not been calibrated against advisors (see FUTURE_WORK.md item 6).

### 10. Can the model tell where a client works from their holdings?

PII detection does not cover investment patterns. A client with a large stake in one mid-cap stock is often an executive or early employee there, and insiders' holdings are public in SEC Form 4 filings. We planted the answer: 40 "insider" clients from the public-data book were given a 15 to 35% position in a mid-cap company, that company's state and a management occupation. 40 "outside investors" got the same kind of position but kept their own job and state, and 40 ordinary clients were left as they are. No name, employer or notes were sent. `gemini-3.8-flash` was asked which public company each client works for (2026-11-20). Command: `python3 eval/pattern_attack.py --n-insiders 40 --n-investors 40 --n-controls 40`. Files: `eval/results_real/pattern_attack*.csv`.

**Share of clients for whom the model named the employer correctly / named a wrong company with confidence of 50 or more:**

| Context sent | Holdings level | Insiders: correct | Outside investors: wrong guess | Ordinary clients: wrong guess |
|---|---|---|---|---|
| Masked (everything exact but identifiers) | exact tickers | 100% | 60% | 15% |
| Masked | anonymized (single stock + sector) | 2.5% | 12.5% | 0% |
| Masked | sectors | 0% | 2.5% | 0% |
| Concentration task policy | exact tickers | 100% | 100% | 18% |
| Concentration task policy | anonymized (the current default) | 0% | 0% | 0% |
| Concentration task policy | sectors | 0% | 0% | 0% |

* **The ticker is the leak.** Whenever the ticker of a concentrated mid-cap position is sent, the model names that company as the employer, for every insider, even with no occupation or location in the context.
* **It cannot tell insiders from outside investors by the position alone.** With only the concentration context it named the held company for 100% of both groups. Occupation and state help it separate them a little (it named the company for 62.5% of outside investors in masked mode). In real books, how often such a guess is right depends on how many concentrated holders are employees, which we cannot measure here.
* **The current task policies already block it.** No task sends exact tickers by default; the anonymized level ("single stock, Healthcare, 22%") brought correct guesses to 0 to 2.5%. The risk is in masked or full-context modes, and in any policy that follows the counterfactual result that some answers improve with exact holdings.

Caveats: the ground truth is planted, the mid-cap list is ours, and one model was tested. The test shows what the context makes possible, not how often real clients would be identified.

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

The `groq-llm` detector is a prompted LLM, not GLiNER. It uses strict structured
outputs to require an items array of exact text and allowed entity labels, then
matches these against the source text to produce the existing span schema.
`GROQ_DETECT_MODEL` must support Groq strict structured outputs. Plain JSON mode
can fail with HTTP 400 `json_validate_failed`, even when the key is valid.
Live testing also observed intermittent validation errors in strict mode, so the
client retries this specific generation error up to twice. Persistent failures
stop with the provider error code and message; failed answers are not cached. This benchmark sends public dataset
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
