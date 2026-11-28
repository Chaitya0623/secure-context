# Counterfactual necessity test

**Question:** which client fields does each task's answer actually depend on, and at what level of detail?

The task policies in `secure_context/policy.py` declare what each task needs. This test measures it instead.

## Method

For each client and task, the model answers once with the full tokenized context (the baseline). It then answers again with exactly one thing changed:

| Counterfactual | What changes | What it shows |
|---|---|---|
| level | one field sent coarser (age band instead of age, ..., dropped) | the least detail the task needs |
| swap | one field's value replaced by another client's, same level | whether the answer uses the value or only its presence |
| null | only the session tokens | noise floor; should not change the answer |

A comparator decides whether each answer would lead the advisor to act differently from the baseline.

* **Real models:** an LLM judge, asked whether an advisor would take different actions or reach different numbers.
* **Offline simulator:** a deterministic comparison that masks tokens and restated inputs.

A field's **measured need** is the coarsest level whose answers stay the same for all but 10% of clients, plus the null-control rate. "drop" means not needed.

Swapped values come from other clients in the same book, so each counterfactual value is realistic. Changing one field at a time can create unusual combinations (a 30-year-old with a retirement goal). That is a known limit of one-at-a-time tests.

## Run it

```bash
python3 eval/run_counterfactual.py                         # simulator, 50 clients, seconds, no key
export GROQ_API_KEY=...
python3 eval/run_counterfactual.py --backend groq --n 10   # real model, about 1,900 calls, resumable
export GOOGLE_CLOUD_PROJECT=secure-context-llm              # Gemini on Vertex AI (gcloud ADC login)
python3 eval/run_counterfactual.py --backend gemini --n 10 # default model gemini-3.8-flash, cached in data/cache/gemini/
```

Options:

* `--tasks all` runs every task, not just the three core tasks.
* `--fields age,aum,risk` limits the test to a subset of fields.
* `--judge-model` picks a different judge model.
* `--keep-answers` stores the answer text.

Outputs go to `eval/results_counterfactual/<backend>/`:

| File | Contents |
|---|---|
| `needs.csv` | measured vs. hand-coded need, with drop and swap change rates |
| `summary.csv` | change rate per task, field and level |
| `privacy.csv` | crowd size k of the measured policy vs. the hand-coded policy |
| `raw.csv` | one row per client, task and counterfactual |
| `change_rates.png` | heatmap of change rates |

## What the simulator run shows (and does not)

The simulator's dependencies are known from its code, so the simulator run checks the test itself. It is not evidence about real models.

* **Agreement:** measured needs match `policy.py` on 32 of 33 task-field pairs.
* **The one disagreement is real:** the simulated concentration answer never uses assets, but the policy lists assets as needed.
* **Presence vs. value:** for the meeting brief, every needed field has 0% swap sensitivity. The simulated agenda depends on whether a field is present, not on its value. A real model will likely differ.
* **Age for income estimates:** age band changes only 30% of answers. The simulator's withdrawal rate floors at 3% below age 55, so exact age only matters for older clients.

`tests/test_counterfactual.py` also checks the test against a probe model with planted dependencies (age decade and risk only). The test must recover "age: band, risk: exact, everything else: drop."

## What the Gemini run shows

`gemini-3.8-flash` on Vertex AI answers and `gemini-3.1-pro-preview` judges, 30 clients, three core tasks, 2026-11-20 (`eval/results_counterfactual/gemini/`). Command: `python3 eval/run_counterfactual.py --backend gemini --n 30 --judge-model gemini-3.1-pro-preview`. All 2,790 judgments parsed.

| | 10 clients, Flash judges itself (2026-11-06) | 30 clients, Pro judges (2026-11-20) |
|---|---|---|
| Null control (meeting brief / concentration / income plan) | 0% / 0% / 20% | 0% / 3% / 17% |
| Measured needs that match `policy.py` | 18 of 33 | 13 of 33 |
| Income plan median k, hand-coded vs. measured policy | 909 vs. 18 | 909 vs. 58 |

* **Null control:** the income plan answer still moves 17% of the time when only session tokens change, so its tolerance rises to 27% and its measured needs are the least reliable.
* **The model uses more exact data than the policy sends, in every task.** Assets and liquidity are needed exactly, not as buckets (removal changed 73 to 100% of answers). Exact holdings matter for the meeting brief and concentration check (sectors or anonymized holdings changed 10 to 43%). This held with the stronger judge.
* **Removing a field and changing its value still behave differently.** For the meeting brief, removing objective, risk or horizon changed 0% of answers, yet swapping in another client's value changed 97 to 100%. Concentration risk shows the same pattern (removal 7%, swap 97%). The model fills in a default when a field is missing, so those "drop" verdicts should not be trusted: keep objective, risk and horizon.
* **Concentration is the clean case.** Age, location, occupation and household are unused (removal 0%, swap 3 to 20%), which matches the policy.
* **Privacy cost, largest for the meeting brief.** Sending what the model measurably uses drops the meeting brief median k from 42,986 to 2, with 76% of clients below k = 10 and 49% unique. Concentration falls from 113,934 to 4,565 (7% below 10); income plan from 909 to 58 (20.5% below 10). The meeting brief is where answer quality and privacy conflict most.

The quality eval (EVALUATION.md section 9) is the other half of this trade: it measures how much worse answers get under the task policy, while this test measures which fields the answers lean on.

## Before trusting real-model results

1. **Check the null control.** A high null rate means the model or judge is noisy, and the tolerance rises with it.
2. **Calibrate the judge.** Grade about 20 answer pairs by hand and compare them with its verdicts.
3. **Read some swapped answers** (`--keep-answers`). Make sure "changed" means a different recommendation, not just different restated numbers.
