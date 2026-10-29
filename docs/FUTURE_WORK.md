# Future work

What this project could not measure without LPL data or production access, and what we would do with it. Items are ordered by how much they could change the conclusions.

## Needs LPL data

### 1. Real advisor notes and client documents
**Gap.** Detection was tested on generated notes and on Gretel's synthetic financial documents. Real advisor notes have their own shorthand, nicknames, misspellings and pasted email threads, and statements arrive as PDFs and scans.
**With LPL data.** Measure span recall and the share of notes with at least one leak on a labeled sample of real notes, statements and emails; add OCR for scanned documents; fine-tune a GLiNER model on LPL-style text and compare.

### 2. Real holdings and transaction histories
**Gap.** Holdings come from a statistical match to the Fed SCF, and transaction streams are generated. Our unicity results show transactions are the most identifying data in the system, so the real number matters most here.
**With LPL data.** Re-run transaction unicity and the holdings linkage attack on real positions and activity; test which aggregations (monthly totals, category bands) keep advisor tasks working.

### 3. The firm's real client base as the reference population
**Gap.** Crowd size k is measured against 978K Census adults, against our 1,000-client book, and against a simulated 150-client advisor book. The book-level results already show k drops sharply when the attacker knows someone is a client, but the real branch and advisor structure is unknown.
**With LPL data.** Measure k against the actual client base, by branch and by advisor, to set the k floor per task.

### 4. Real request logs from advisors
**Gap.** The task mix comes from Bitext's customer-facing intents, and advisor queries were built by inserting client details into those requests.
**With LPL data.** Measure how often advisors type names, ages, places and amounts into requests; calibrate the task router on real request types; measure how linkable a client's requests are by content in practice (the ledger currently assumes everything is linkable).

### 5. Real proprietary strategies and business information
**Gap.** Extraction was tested on one planted allocation rule. Institutional IP also includes research views, model portfolios, product shelf decisions, pricing and business patterns (AUM by region, client counts, flows).
**With LPL data.** Plant real (or realistic) model portfolios and research, run extraction attacks through the assistant, and test aggregate questions that could reveal business patterns.

### 6. Usefulness judged by advisors and compliance
**Gap.** Task utility is a deterministic proxy; `eval/llm_eval.py` adds an LLM judge but has not been calibrated against people.
**With LPL data.** Blind review by advisors of full-context vs. minimized answers on real tasks; compliance review of what each task policy discloses.

## Needs production access, not data

* **Key management and audit:** session keys in a KMS or HSM, vault access audited end to end, and incident-response timing under the amended SEC Regulation S-P.
* **Model provider retention:** confirm what the provider logs, caches or trains on, and whether zero-retention options change the threat model.
* **Latency at scale:** detection and policy search are under 1 ms per request here; measure with a real CRM, real document sizes and concurrent advisors.

## Threats identified but not yet tested

* **Prompt injection on the way back:** a document in the context instructs the model to place tokens in a link or email; rehydration then fills in real values. Control: rehydrated text never triggers actions automatically, and outbound links are stripped or reviewed.
* **Embedding inversion:** if client documents are stored as embeddings for retrieval, embeddings can be partly inverted. Control: embed dehydrated text only.
* **Household and relationship structure:** the shape of a family (widowed, two children, son with power of attorney) can identify a client even when every name is tokenized.
* **Token collisions:** tokens use 24 bits; very large sessions should check uniqueness or use longer tokens.
* **Insider misuse:** an advisor querying clients outside their book; the book-bound control in the strategy experiment also limits this.

## Ready to run now (needs only compute or an API key)

* GLiNER detectors: `python3 eval/run_real.py --gliner nvidia/gliner-PII`
* LLM-judged quality and the LLM attacker: `python3 eval/llm_eval.py --backend anthropic --n 30`
