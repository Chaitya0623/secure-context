# Does an agent with memory learn who its clients are?

`docs/SESSION_LINKING.md` showed two sessions to the model in one prompt. This test gives an agent real notes that persist between sessions.

## Setup

`eval/memory_agent.py`, `gemini-3.8-flash`, public-data client book, 30 clients, 4 quarters, 2026-11-23.

* **Each quarter, each client has one session:** a different task (meeting brief, concentration check, income plan, portfolio review), a new session token, drifted values, and only the fields that task's policy allows. Advisor notes are dropped.
* **The agent** sees its whole memory, says which entry is this client (or NEW), answers, and rewrites that entry in at most 80 words. It is never told whether it picked correctly.
* **Two memory rules:**
  * **Helpful:** keep whatever will help future sessions.
  * **No client facts:** keep only general lessons, never facts about a client.
* **What the notes retained:** a separate call reads only the stored note and extracts age, assets and portfolio shape, scored against the true values. We then compute crowd size k against the Census-based directory.

Command: `GOOGLE_CLOUD_PROJECT=<project> python3 eval/memory_agent.py --n 30 --rounds 4`. Files: `eval/results_real/memory_agent_sessions.csv` (includes the stored notes) and `memory_agent_summary.csv`. All replies parsed.

## Results

**Which entry the agent picked for a returning client (30 sessions per quarter, chance about 3%):**

| Rule | Quarter | Own entry | Another client's entry | Said NEW |
|---|---|---|---|---|
| Helpful | 2 | 17% | 57% | 27% |
| Helpful | 3 | 57% | 33% | 10% |
| Helpful | 4 | 37% | 53% | 10% |
| **Helpful, all** | | **37%** | **48%** | **16%** |
| No client facts | 2 | 0% | 97% | 3% |
| No client facts | 3 | 3% | 87% | 10% |
| No client facts | 4 | 0% | 87% | 13% |
| **No client facts, all** | | **1%** | **90%** | **9%** |

Under the helpful rule, recognition was 54% for the 16 clients with invested portfolios and 17% for the 14 clients whose holdings are all cash, who look alike.

**How identifiable each client is by quarter 4 (helpful rule):**

| What an observer holds | Median crowd k | Clients below k = 10 |
|---|---|---|
| One session | 29,491 | 3% |
| The agent's stored note | 15,025 | 10% |
| Everything disclosed across all 4 sessions | 77 | 23% |

By quarter 4 the helpful notes held exact assets for 60% of clients, exact age for 33%, and portfolio shape for 73%.

## What it means

* **The agent does learn clients, well above chance but unreliably.** It found the right client 37% of the time against about 3% by chance.
* **The more common failure is a mix-up.** In 48% of returning sessions it attached the client to someone else's note and then rewrote that note. That is a confidentiality and advice-quality problem even without an attacker: one client's facts can shape another client's answer.
* **Memory raises exposure, but less than the logs do.** A note is more identifying than one session, because it keeps facts from earlier tasks. It is much less identifying than the full session history, because 80 words cannot hold everything. The logs remain the larger risk (see `docs/SESSION_LINKING.md`).
* **Banning client facts works, with a leak.** Recognition fell to chance and no client fell below k = 10. But 21 of 120 "general lessons" still described the client's situation (for example, a conservative client holding only cash), so a written rule is not a guarantee.

## Caveats

30 clients, one model, one seed, and one memory design. The agent sees all entries; a retrieval-based memory could behave differently. A wrong match overwrites the other client's note, so early mix-ups can lower later recognition. A match counts as right if it picks the entry the client created, even if another client has since overwritten it. 47% of these 30 clients hold only cash, which makes them hard to tell apart, so a book with more varied portfolios would likely be easier to recognize. We asked the agent to match clients; a product that never does so would have fewer mix-ups but might still accumulate facts. Retained facts are read by an LLM extractor, which was not hand-checked.
