#!/usr/bin/env python3
"""Open a reminder issue in this repository and assign it to the student.

Run by .github/workflows/reminder.yml (Monday and Thursday). GitHub e-mails the assignee when an
issue is assigned, so the student is notified on GitHub and by e-mail. The previous reminder is
closed first, so there is never more than one open reminder.

It reads tools/course.yaml and the tags and commits of this repository, and sends:
  * every Monday: a weekly digest (deadlines in the next 8 days, missing tags, last commit);
  * on Thursday: only if a deadline is within 4 days and its tag is missing, or no commit for 4 days.
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
COURSE = yaml.safe_load((ROOT / "tools" / "course.yaml").read_text(encoding="utf-8"))
TZ = dt.timezone(dt.timedelta(hours=7))
END = dt.datetime(2027, 1, 31, tzinfo=TZ)               # no reminders after the semester
INSTRUCTOR_NAMES = {"Nguyễn An Khương", "Nguyen An Khuong"}
REPO = os.environ.get("GITHUB_REPOSITORY", "")
ASSIGNEE = os.environ.get("ASSIGNEE", "").strip()
NOW = dt.datetime.now(TZ)


def sh(*a: str) -> str:
    r = subprocess.run(a, cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def parse(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=TZ)


def deadlines() -> list[tuple[dt.datetime, str, str, str]]:
    out = []
    for k, v in (COURSE.get("gates") or {}).items():
        if v.get("due"):
            out.append((parse(v["due"]), k, k, "setup gate"))
    for k, v in COURSE["weeks"].items():
        out.append((parse(v["due"]), v["tag"], v["tag"], f"{k}: {v['topic']}"))
    for k, v in (COURSE.get("parts") or {}).items():
        if v.get("due"):
            out.append((parse(v["due"]), k, k, "graded part"))
    for k, v in (COURSE.get("homework") or {}).items():
        out.append((parse(v["due"]), k, k, f"homework: {v['topic']}"))
    return sorted(out)


def last_student_commit() -> tuple[dt.datetime | None, str]:
    for line in sh("git", "log", "-200", "--format=%cI|%an|%s").splitlines():
        when, author, subject = line.split("|", 2)
        if author not in INSTRUCTOR_NAMES:
            return dt.datetime.fromisoformat(when), subject
    return None, ""


def gh(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *a, "--repo", REPO], capture_output=True, text=True)


def main() -> int:
    if NOW > END:
        print("Semester over; no reminder.")
        return 0
    tags = set(sh("git", "tag").split())
    upcoming, overdue = [], []
    for due, tag, _, label in deadlines():
        if tag in tags:
            continue
        if due < NOW:
            overdue.append((due, tag, label))
        elif due - NOW <= dt.timedelta(days=8):
            upcoming.append((due, tag, label))
    last, subject = last_student_commit()
    idle_days = (NOW - last).days if last else None
    soon = any(d - NOW <= dt.timedelta(days=4) for d, _, _ in upcoming)
    idle = idle_days is None or idle_days >= 4
    monday = NOW.weekday() == 0
    if not (monday or soon or idle):
        print("Nothing to remind on this run.")
        return 0

    today = NOW.strftime("%Y-%m-%d")
    lines = [
        f"Hello{' @' + ASSIGNEE if ASSIGNEE else ''}, this is the automatic reminder of {today}. "
        "It closes itself when the next one arrives.",
        "",
        "**Commit rhythm.** Each week keeps two honest states: your first attempt (commit it before you "
        "open any reference code or AI tool) and the corrected version after study. Commit small and often, "
        "with messages such as `[W05][code] own Gaussian NB, first attempt`, and push. "
        "Push one checkpoint tag at a time, never `--tags`.",
        "",
    ]
    if last:
        lines.append(f"Your last commit: {last.astimezone(TZ):%Y-%m-%d %H:%M} ({idle_days} day(s) ago): {subject}")
    else:
        lines.append("There is no commit of yours yet. Start with `START_HERE.md`.")
    lines.append("")
    if upcoming:
        lines += ["**Due in the next 8 days (tag not pushed yet)**", "", "| Tag | Due (Asia/Ho_Chi_Minh) | What |", "|---|---|---|"]
        lines += [f"| `{t}` | {d:%a %d %b %H:%M} | {lab} |" for d, t, lab in upcoming]
        lines.append("")
    if overdue:
        lines += ["**Past due and still without a tag** (late work stays visibly late; push it when it is ready)",
                  "", "| Tag | Was due | What |", "|---|---|---|"]
        lines += [f"| `{t}` | {d:%a %d %b %H:%M} | {lab} |" for d, t, lab in overdue[-8:]]
        lines.append("")
    lines += ["Before tagging, run `python tools/check.py --checkpoint <tag>`. "
              "The assignment text is in `docs/assignment/ASSIGNMENT.pdf`; the deadlines are in `START_HERE.md`."]

    title = f"[Reminder] {today}: commit and push your work"
    if upcoming:
        title = f"[Reminder] {today}: {len(upcoming)} checkpoint(s) due within 8 days"
    body = "\n".join(lines)

    gh("label", "create", "reminder", "--color", "1f4e79", "--description", "Automatic course reminder", "--force")
    old = gh("issue", "list", "--label", "reminder", "--state", "open", "--json", "number", "--jq", ".[].number")
    cmd = ["issue", "create", "--title", title, "--body", body, "--label", "reminder"]
    r = gh(*cmd, *(["--assignee", ASSIGNEE] if ASSIGNEE else []))
    if r.returncode != 0 and ASSIGNEE:
        print("Assignment failed (invitation not accepted yet?):", r.stderr.strip())
        r = gh(*cmd)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1
    print(r.stdout.strip())
    for n in old.stdout.split():
        gh("issue", "close", n, "--comment", "Superseded by a newer reminder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
