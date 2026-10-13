# AI_USE

One row per AI-assisted learning episode (assignment Section 12.5). Weeks with a row here must
also have section H (Inquiry trail) in that week's post. Pre-AI evidence is a commit hash from W05 on;
for the W01-W04 catch-up cite the release-day baseline file.

| Week/date | Learning question | Pre-AI evidence | AI tool | Prompt purpose | Hint/question received | Verification source | What changed | Closed-book reproduction |
|---|---|---|---|---|---|---|---|---|
| W05 / 2026-09-29 | Formulating multi-class OvR Perceptron, Perceptron vs Delta distinctions, XOR limitation proof | 443faae | Antigravity AI Assistant | Socratic review of derivations, OvR decision score, metrics schema | Verified raw signed distance for argmax ties; checked standard 8-column CSV format | Tom Mitchell (1997) ML Ch 4.4-4.5; Bishop (2006) PRML Ch 4 | Verified XOR contradiction proof; confirmed AND/XOR sanity checks | Yes, see exercises/w05-first-attempt.pdf |
| W06 / 2026-10-06 | Replicating sklearn GaussianNB log-posterior normalization and variance smoothing | b60f5c5 | Antigravity AI Assistant | Implementation guidance for var_smoothing formula and logsumexp normalization | Smoothing uses var_smoothing × max(var) globally; logsumexp for numerical stability | sklearn source (GaussianNB); Tom Mitchell (1997) ML Ch 6 | Corrected epsilon scaling from fixed constant to data-relative | Yes, see exercises/w06-first-attempt.pdf |
