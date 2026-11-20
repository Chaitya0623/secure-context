# Can sessions be linked by the client's pattern?

The model keeps no memory between API calls. The provider's logs do, and so would an agent with a memory feature. Session tokens change every time, so the question is whether the rest of the context is enough to tell that two sessions are about the same client.

## Setup

`eval/session_linking.py`, public-data client book, 2026-11-20.

* **Two quarters per client.** "Last quarter" is the client as is. "This quarter" has new session tokens, holdings weights moved by up to 3 points, assets moved by up to 8%, age up a year for half the clients, and the goal three months closer. Notes are dropped everywhere, since real notes differ between meetings.
* **Five contexts:** everything exact except identifiers (masked mode); the concentration-check policy with exact tickers, with anonymized holdings (the current default) and with sectors; and the meeting-brief policy.
* **Two attackers:**
  * **Model:** `gemini-3.8-flash` sees last quarter's context and a lineup of 10 current contexts (the same client plus 9 random others) and picks the match or says NONE. 50 clients, chance is 10%.
  * **Script:** no AI. It picks the candidate with the most identical lines, then the closest numbers. This is what anyone holding the logs could run, so we also run it against every client in the book.

Command: `GOOGLE_CLOUD_PROJECT=<project> python3 eval/session_linking.py --n 50 --lineup 10`. Files: `eval/results_real/session_linking*.csv`.

## Results

**Lineup of 10 (chance 10%):**

| Context | Model picks the right client | Script picks the right client |
|---|---|---|
| Masked, all exact | 100% | 100% |
| Concentration, exact tickers | 88% | 88% |
| Concentration policy (anonymized, current default) | 88% | 88% |
| Concentration, sectors | 88% | 94% |
| Meeting brief policy | 100% | 100% |

When the model missed, it mostly answered NONE (5 or 6 of the 6 misses in each concentration context), not a wrong client.

**Script against the whole book:**

| Context | Book of 200 (chance 0.5%) | Book of 1,000 (chance 0.1%) |
|---|---|---|
| Masked, all exact | 100% | 100% |
| Concentration, exact tickers | 35% | 16.5% |
| Concentration policy (anonymized) | 35% | 16.5% |
| Concentration, sectors | 51% | 35.5% |
| Meeting brief policy | 98% | 95% |

**Meeting brief, book of 1,000, one field removed:** none removes the link. Without household 85%, without horizon 88%, without liquidity 91%, without holdings 92.5%, without any other single field 93 to 94.5%.

## What it means

* **Minimizing each request does not stop linking.** The meeting-brief policy keeps every client in a crowd of 10 or more against the Census, yet the same client's sessions can be matched 95% of the time among 1,000 clients. The crowd that matters for linking is the firm's own clients, the same lesson as the reference-population result in EVALUATION.md section 3.
* **No single field is to blame.** The link comes from the combination of stable answers (household, horizon, goal, risk, age band, asset bucket). Removing any one field still leaves 85% or more linkable.
* **Coarser holdings are not a defense here.** Hiding tickers changes nothing (35% either way), and sector totals link more easily than position-level weights (51% vs 35%). We also tried rounding holdings weights to 5, 10 and 25 points before the whole-book run; linking stayed the same or rose (rounded values match exactly more often), so that check is not in the script.
* **An AI is not needed.** A few lines of string matching do as well as the model.

## What would help (not built yet)

Linking is a property of what gets logged over time, so the fixes are mostly architectural:

* Send the stable profile only when the task needs it, and send only the fields that changed when it does not.
* Use a provider agreement with zero data retention, so there are no logs to link.
* Count linkability in the exposure ledger: treat all sessions for one client as one release and measure k against the firm's book, not the Census.
* Add random noise to stable numeric answers per session (for example horizon plus or minus a year) and measure the cost to answer quality.

## Caveats

The drift between quarters is our assumption: larger real changes would make linking harder, while unchanged profiles (common, see investor inertia) would make it easier. The lineup uses random clients, not lookalikes. One model was tested. The whole-book numbers use the script only.
