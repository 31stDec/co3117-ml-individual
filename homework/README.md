# Handwritten homework sets

Five sets, one per Part I topic. The problem sheets are posted on the course LMS; this folder holds
your submissions. Each set has five activities and is worth 10 points in total; the points of each activity are printed on the sheet.

| Set | Topic | Tag | Deadline (GitHub receives the tag) | Assessment Format |
| --- | --- | --- | --- | --- |
| HW1 | Foundations: workflow, evaluation, generalisation | `hw1` | 2026-10-07 07:00 | Formative handwritten + self-check |
| HW2 | Decision trees | `hw2` | 2026-10-07 07:00 | Formative handwritten + self-check |
| HW3 | Artificial neural networks | `hw3` | 2026-10-07 07:00 | Formative handwritten + self-check |
| HW4 | Bayesian learning and Bayesian networks | `hw4` | 2026-10-14 23:59 | Formative handwritten + self-check |
| HW5 | Genetic algorithms | `hw5` | 2026-10-14 23:59 | Formative handwritten + self-check |

## How to submit

1. Write by hand, on paper or with a stylus. Show your working. Put your full name, student ID and
   the set number at the top of the first page.
2. Work on your own. Unlike the weekly drill, this is open-notes: lecture notes and course slides are allowed; AI tools, solution manuals and other
   students' work are not, until your submission is tagged. Record any later AI use in `AI_USE.md`.
3. Scan all pages, in order, into ONE PDF of at most 5 MB. A phone scanner app in grey scale at
   about 150 to 200 dpi is enough. Name it `homework/hwN-submission.pdf` (N = 1 to 5).
4. Commit, check, tag and push:

```bash
git add homework/hwN-submission.pdf
git commit -m "[W05][homework] HWN submission"
python tools/check.py --checkpoint hwN
git tag hwN
git push origin main hwN
```

Push one tag at a time, never with `--tags`.

## Self-checking and corrections

No official worked solutions are published. After completing your handwritten submission, you are
encouraged to verify your calculations using course slides, discussion with peers/instructors, and AI
assistants (logging prompt strategies under `AI_USE.md`).

If you discover misconceptions or calculation errors, write `homework/hwN-corrections.md`: what was
wrong, why, and the corrected reasoning, citing the reference used. The corrections file must be
committed after the submission. Never replace, delete, or move the original submission or its tag; a
moved or deleted tag does not change what is marked, and all history is preserved.

## How the sets relate to the assignment

The five sets are an addition to the assignment text: they are handwritten practice on the Part I
topics and sit beside the weekly work, not on top of the roughly 3 to 4 hours a week the assignment
plans. The last activity of each set (the Assignment Bridge) uses your own project dataset. Put its
page(s) in your submission PDF; you may reuse the result in that week's post or in `protocol.yaml`
(for example the Protocol Card of HW1), but the PDF is what the homework tag records.
