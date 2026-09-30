# CO3117 Machine Learning: individual longitudinal repository

This repository is your single, continuous record for the whole semester (assignment Section 5).
It lives in the course organisation; you have write access and the instructor has admin access.
Nobody else in the class can see it.

> **New here? Read [START_HERE.md](START_HERE.md) first.** The full assignment text is in
> [docs/assignment/ASSIGNMENT.pdf](docs/assignment/ASSIGNMENT.pdf) (also as
> [HTML](docs/assignment/ASSIGNMENT.html)); deadlines are in `START_HERE.md` and `tools/course.yaml`.

## 1. First setup

1. Accept the invitation e-mailed by GitHub (it expires after 7 days; ask the instructor to re-invite).
2. `git clone https://github.com/HCMUT-CO3117-261/ml-<student-id>.git` and `cd` into it.
3. `python -m venv .venv`, activate it, `pip install -r requirements.txt`.
4. `python tools/check.py` must run (failures are expected before you start).
5. Set your Git name and e-mail once: `git config user.name "..."` and `git config user.email "..."`.

**You already started a repository in your own account?** Do not copy files over. Move the whole
history so that your earlier commits keep their real dates:

```
cd <your old repository>
git remote add course https://github.com/HCMUT-CO3117-261/ml-<student-id>.git
git fetch course
git merge --allow-unrelated-histories course/main    # brings in the course skeleton
git push course HEAD:main
```

If your old repository already has tags, push them one at a time (`git push course <tag>`), never
with `--tags`: GitHub does not run the checkpoint check when more than three tags arrive in one push.

Then work only in the course repository and archive the old one (Settings, Archive).

## 2. Weekly routine (from W03)

1. Written drill, closed book: scan it to `exercises/wNN-first-attempt.pdf` (or .jpg/.png) and commit it
   immediately: `[W05][drill] first attempt`.
2. First attempt at the code for Depth-A components in `src/from_scratch/`: commit it before you open any
   reference repository or AI tool: `[W05][code] own Gaussian NB, first attempt`.
3. Study, fix, benchmark, write `exercises/wNN-corrections.md` and the weekly post
   `docs/weekly/wNN-<topic>.md` from `docs/weekly/_TEMPLATE.md`.
4. Fill the week's row in PROGRESS.md (post and drill links, first-attempt hash, revision hash).
5. `python tools/check.py --checkpoint wNN` until it passes, then tag and push:

```
git tag wNN
git push origin main wNN
```

Pushing the tag runs the same check in the Actions tab, and GitHub records the time it received the
tag: that time, not the commit date, is your submission time. Push checkpoint tags one at a time,
never with `--tags`; GitHub does not run the check when more than three tags arrive in one push.
Deadlines for each tag are in `tools/course.yaml`.

**Dating your work honestly.** The repositories were issued on 2 October, so your first W03 and W04
commits are dated 2 October or later. That is expected: never back-date a commit and never push old
work as if it were new. Commit dates are set by your computer; the instructor uses the time GitHub
received the push.

Special tags (push each one on its own, after `python tools/check.py --checkpoint <tag>` passes):

| Tag | What it marks | Deadline |
| --- | --- | --- |
| `release-baseline` | R0 setup gate: skeleton, catch-up post, release-day baseline scan `exercises/release-baseline-w01-w02.pdf`, frozen `protocol.yaml` | `tools/course.yaml` |
| `w08-midterm` | Midterm week: compact entry (suggested `docs/weekly/w08-midterm.md`), timed rehearsal scan (suggested `exercises/w08-rehearsal.pdf`), then `exam/midterm-reflection.md` after the exam | `tools/course.yaml` |
| `part1-final` | Graded Part I package (`SUBMISSION_PART1.md`, `report/part1_summary.pdf`, `exam/a4-notes-part1-midterm.pdf`) | 14 Oct 2026 23:59 |
| `part2-final` | Graded Part II package (adds the mock final and `exam/a4-notes-part2-final.pdf`) | two calendar days before the final exam; date not yet published |
| `hw1` to `hw5` | One handwritten homework set each (section 2b) | `tools/course.yaml` |

Also required by the assignment: update the two-A4 exam sheet and `MODEL_LOG.md` every week, and follow
the AI-use protocol of assignment Section 12 (see `AI_USE.md`). `START_HERE.md` lists the rest.

## 2b. Handwritten homework sets (Part I)

Five homework sets, one per Part I topic. The problem sheets are in `homework/` and on the course LMS.
They are formative handwritten practice, separate from the weekly drill, and they are reviewed under
the Written-exam portfolio criterion; no conversion of their scores to points is published. Deadlines
are listed under `homework` in `tools/course.yaml` and in `homework/README.md`.

1. Write your answers by hand (paper or stylus), with your name, student ID and set number on page 1.
2. Scan all pages, in order, into one PDF of at most 5 MB: `homework/hwN-submission.pdf`.
3. Commit it (`[W05][homework] HW3 submission`; use the week of the set), run `python tools/check.py --checkpoint hwN`, then:

```
git tag hwN
git push origin main hwN
```

The time GitHub receives the tag is your submission time. No worked solutions are published. After you
have checked your work with the course materials and AI tools, you may add `homework/hwN-corrections.md`;
never replace, delete or re-tag the submission.

## 3. Rules the organisation enforces or records

- The history of `main` and every checkpoint tag are permanent. Do not force-push, rebase published
  commits, delete or move a tag. GitHub accepts such pushes, but it records force pushes, and the
  course keeps its own copy of every pushed state; the commit that is marked is the one a tag pointed
  to when GitHub first received it. A rewritten history caps the progress criterion at half marks.
- Commit dates are set by your computer; the instructor uses the time GitHub received the push.
  Late work stays visibly late. Never back-date commits.
- Do not edit `tools/`, `tests/contract/` or `.github/`. They are compared with the released version.
- Never commit raw data, archives, model binaries or secrets (see `.gitignore` and `data/README.md`).
- Do not enable GitHub Pages. Your dossier is Markdown under `docs/` and reads fine on GitHub.

## 4. What the machine checks, and what it does not

`tools/check.py` checks structure and process only: required files, PROGRESS.md rows and links,
first-attempt commit before revision commit, drill scan committed before its corrections, post length
and sections A-H, AI_USE.md rows, protocol.yaml filled, no forbidden or oversized files, and the
contract tests for your Depth-A components (Appendix A). Passing the check is necessary, not
sufficient: the quality of derivations, experiments, analysis and writing is marked by a person,
and any artifact can be the subject of an ownership check (assignment Section 10.1).

Contract tests: a component becomes required from the week listed under `components` in
`tools/course.yaml`. Before that week an unfinished stub is skipped.
