# AI_USE.md — AI Usage Disclosure Log

> **Policy:** AI tools may be used only after a first-attempt commit, for supportive purposes (Socratic hints, debugging, counterexamples, quizzes). Never for producing first-attempt code, drills, or blog posts.  
> **Protocol:** FRAME → ATTEMPT → INQUIRE → VERIFY → RECONSTRUCT → TRANSFER → TEACH-BACK → DELAYED RETRIEVAL

---

## Log Entries

### Template (copy for each AI-assisted episode)

| Field | Entry |
|-------|-------|
| **Week/date** | W__ / YYYY-MM-DD |
| **Learning question** | [The precise concept/problem being investigated] |
| **Pre-AI evidence** | [Commit hash or handwritten first-attempt artifact] |
| **AI tool** | [Tool/model used, e.g., ChatGPT-4, Claude, Copilot] |
| **Prompt purpose** | [Socratic hint / counterexample / debugging question / quiz / etc.] |
| **Hint/question received** | [Concise summary — do NOT paste full generated solution] |
| **Verification source** | [CO3117 note / textbook section / NPTEL lecture / repository reference] |
| **What changed** | [Specific misconception, derivation, test, or code decision corrected] |
| **Closed-book reproduction** | [Yes / Not yet — link delayed-retrieval artifact when available] |

---

<!-- AI usage entries will be added below as the semester progresses -->

### Episode 1: W05 — Perceptron & Delta Rule Formulation

| Field | Entry |
|-------|-------|
| **Week/date** | W05 / 2026-09-29 |
| **Learning question** | Formulating multi-class One-vs-Rest aggregation for single-layer perceptrons, verifying Perceptron vs Delta rule mathematical distinctions, and reviewing handwritten proof for XOR limitation. |
| **Pre-AI evidence** | Commit `443faae` ([`exercises/w05-first-attempt.pdf`](exercises/w05-first-attempt.pdf)) and commit `48b97e7` ([`src/from_scratch/perceptron.py`](src/from_scratch/perceptron.py)) |
| **AI tool** | Antigravity AI Assistant |
| **Prompt purpose** | Socratic review of mathematical derivations, verifying OvR competitive decision function, and checking 8-column metrics schema. |
| **Hint/question received** | Verified that raw signed distance $X w + b$ must be passed to $\arg\max$ across class detectors, rather than discrete $\text{sign}(z)$, to resolve multi-class ties; checked standard 8-column CSV format. |
| **Verification source** | Tom Mitchell (1997) *Machine Learning*, Chapter 4, Sections 4.4–4.5; Bishop (2006) *Pattern Recognition and Machine Learning*, Chapter 4. |
| **What changed** | Standardized `results/metrics.csv` to match exact repository schema; verified proof-by-contradiction for XOR; confirmed AND/XOR sanity checks. |
| **Closed-book reproduction** | Yes — handwritten drill completed closed-book on paper (`exercises/w05-first-attempt.pdf`). |
