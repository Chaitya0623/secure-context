# secure-context

**p040: Giving AI Client Context Without Exposing Client Data**
Columbia Data Science Institute industry project with LPL Financial.

Code, prototype and evaluation live on the [`rishika`](https://github.com/Chaitya0623/secure-context/tree/rishika) branch.

## Industry context

AI in wealth management is only as useful as its understanding of context. An advisor's AI assistant becomes much more helpful when it knows a client's investment objectives, financial circumstances, portfolio characteristics, preferences and history. That same personalization creates privacy, confidentiality and data-protection risk.

Masking, tokenization and pseudonymization help, but they are not enough on their own. Even with names and account numbers removed, combinations of financial attributes, or repeated AI interactions, can reveal information that was meant to stay confidential.

The challenge also goes beyond client identity. Financial institutions hold valuable intellectual property: proprietary research, investment methodologies, business strategies and commercially sensitive information. An AI system can expose this indirectly through repeated interactions, even when no single response looks like a disclosure.

## The challenge

How can AI systems be context-aware and deliver personalized wealth-management insights without unnecessarily exposing client information or allowing sensitive institutional knowledge to be reconstructed?

Can we determine the minimum information an AI system needs to complete a task accurately, while measuring and controlling what could still be inferred?

## Key areas of exploration

1. **Privacy-preserving client context.** Represent sensitive client information so it stays useful without exposing the underlying records: tokenization, pseudonymization, derived financial attributes, abstraction. The core question is which details are essential for accurate analysis and which can stay protected.
2. **Intelligent context sharing.** Different tasks need different information. A meeting agenda needs objectives and preferences; a concentration calculation needs position-level detail. The system should decide what each task needs and share an appropriate representation instead of the full profile.
3. **Protection against information reconstruction.** Limited information can become sensitive when combined with other data or repeated responses. Two risks matter most:
   * **Client confidentiality:** reconstructing identities, financial circumstances or hidden portfolio details.
   * **Institutional intellectual property:** inferring proprietary strategies, research, business patterns or confidential organizational knowledge.

   A key difficulty is understanding how exposure accumulates across many interactions.

## Envisioned product

A working privacy-preserving client context system for wealth-management AI, with:

* **Protected client-information environment:** representative client profiles, financial details and a separately controlled identity-reference mapping.
* **Context transformation:** detailed information converted into task-relevant representations that reduce unnecessary disclosure.
* **Controlled information sharing:** AI applications receive what an authorized task needs.
* **Context-aware AI application:** personalized insights or advisor-support materials.
* **Privacy testing:** whether sensitive information can be inferred through single or repeated interactions.
* **Comparative evaluation:** conventional masking vs. selective context sharing.
* **Interactive dashboard:** AI usefulness vs. amount disclosed vs. observed privacy risk.

## Illustrative scenario

An AI assistant helps advisors prepare for client meetings. It needs the client's objectives, portfolio characteristics, risk preferences and liquidity needs, but not their full identity, account numbers, detailed holdings and transaction history. Three approaches are compared:

| Approach | What the AI receives |
|---|---|
| Full client context | the complete simulated client profile |
| Masked client context | direct identifiers removed or replaced; most other financial information kept |
| Intelligent context sharing | only what the task needs, possibly summarized or derived |

The system should also test whether repeated questions can reveal hidden portfolio characteristics, or whether repeated queries about recommendations can reveal a proprietary investment strategy.

## Success measures

* Accuracy and usefulness of AI-generated financial analysis
* Amount and sensitivity of information disclosed
* Ability to infer hidden client attributes or reconstruct client identity
* Ability to reconstruct protected institutional information across repeated queries
* Impact of context reduction on task completion and answer quality
* Latency and computational overhead of privacy-preserving approaches
* Effectiveness of safeguards under different financial scenarios

No single transformation should be assumed to make information completely anonymous or immune to reconstruction.

## Data and experimental environment

Public financial-industry materials and synthetic client information: simulated portfolios, objectives, preferences, transaction histories and artificial proprietary strategies. Test cases carry known ground truth so both usefulness and leakage can be measured. Actual client records, confidential research and production access are not required.

## Expected outcomes (December 2026)

* A simulated client-information repository
* A task-aware context-sharing mechanism
* A working advisor-assistance application
* A comparison of information exposure and task quality against conventional masking
* Reusable source code, test scenarios, architecture documentation and findings

## References

1. NIST IR 8053, [De-Identification of Personal Information](https://csrc.nist.gov/pubs/ir/8053/final)
2. [MINIM: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization](https://proceedings.mlr.press/v306/yu26bq.html)
3. [RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in RAG](https://ojs.aaai.org/index.php/AAAI/article/view/40432) (AAAI)
4. NIST SP 800-226, [Guidelines for Evaluating Differential Privacy Guarantees](https://csrc.nist.gov/pubs/sp/800/226/final)
5. [Operationalizing Data Minimization for Privacy-Preserving LLM Prompting](https://arxiv.org/abs/2510.03662) (arXiv 2510.03662)
6. SEC, [Regulation S-P: Privacy of Consumer Financial Information and Safeguarding Customer Information](https://www.sec.gov/rules-regulations/2024/06/s7-05-23)
