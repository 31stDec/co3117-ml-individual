#!/usr/bin/env python3
"""CO3117 repository self-check.

    python tools/check.py                    # checkpoint inferred from today's date
    python tools/check.py --checkpoint w05   # or release-baseline, part1-final, part2-final
    python tools/check.py --checkpoint hw3   # a handwritten homework set (hw1..hw5)
    python tools/check.py --no-tests         # skip the contract tests
    python tools/check.py --json out.json    # machine-readable result

The same script runs in GitHub Actions when you push a checkpoint tag, and the instructor
runs it on the tagged commit. It reads only files and Git history; it never judges the
quality of your writing or experiments, which are marked by a person.

Exit code: 0 all checks pass, 1 at least one check failed, 2 usage error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import pathlib
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
COURSE = yaml.safe_load((ROOT / "tools" / "course.yaml").read_text(encoding="utf-8"))
HEX = re.compile(r"\b[0-9a-f]{7,40}\b")
POST_SECTIONS = ["A", "B", "C", "D", "E", "F", "G"]
PROTOCOL_KEYS = ["dataset", "dataset_version", "target", "use_case", "split_policy",
                 "primary_metric", "seeds"]


# ------------------------------------------------------------------ helpers
def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def commit_exists(sha: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT,
                          capture_output=True).returncode == 0


def is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT,
                          capture_output=True).returncode == 0


def wnum(week: str) -> int:
    return int(str(week).upper().lstrip("W"))


def week_of_checkpoint(cp: str) -> int:
    cp = cp.strip().lower()
    if cp in COURSE.get("homework", {}):
        return wnum(COURSE["homework"][cp]["week"])
    if cp in COURSE.get("gates", {}):
        return 4                      # setup gate (R0): W05 evidence belongs to the w05 tag, not to R0
    if cp in COURSE.get("parts", {}):
        return wnum(COURSE["parts"][cp]["week"])
    for w, info in COURSE["weeks"].items():
        if info["tag"] == cp:
            return wnum(w)
    m = re.fullmatch(r"w?(\d{1,2})", cp)
    if m:
        return int(m.group(1))
    raise SystemExit(f"Unknown checkpoint '{cp}'.")


def week_from_date(now: dt.datetime) -> int:
    """Latest week whose tag deadline has passed (at least W05)."""
    cur = 5
    for w, info in COURSE["weeks"].items():
        if dt.datetime.fromisoformat(info["due"]) <= now:
            cur = max(cur, wnum(w))
    return cur


def words(md: str) -> int:
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    md = re.sub(r"\$\$.*?\$\$", " ", md, flags=re.S)
    md = re.sub(r"<!--.*?-->", " ", md, flags=re.S)
    return len(re.findall(r"[A-Za-zÀ-ỹ0-9]+(?:['’-][A-Za-zÀ-ỹ0-9]+)*", md))


def md_links(cell: str) -> list[str]:
    return [t for t in re.findall(r"\]\(([^)\s]+)\)", cell) if not t.startswith(("http:", "https:", "#"))]


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, check: str, ok: bool, detail: str = "", level: str = "fail") -> None:
        self.items.append({"check": check, "ok": ok, "level": "ok" if ok else level, "detail": detail})

    @property
    def failed(self) -> list[dict]:
        return [i for i in self.items if i["level"] == "fail"]


# ------------------------------------------------------------------ checks
def check_files(rep: Report, cp: str) -> None:
    rf = COURSE["required_files"]
    req = list(rf["always"]) + list(rf.get(cp, []))
    if cp in ("part1-final", "part2-final"):
        req += rf["release-baseline"]         # the catch-up and the release-day baseline stay part of both parts
    if cp == "part2-final":
        req += rf["part1-final"]
    for f in req:
        rep.add(f"file {f}", (ROOT / f).is_file(), "missing")
    if cp in ("release-baseline", "part1-final", "part2-final"):
        has_baseline = (ROOT / "exercises/release-baseline-w01-w04.pdf").is_file() or (ROOT / "exercises/release-baseline-w01-w02.pdf").is_file()
        rep.add("file exercises/release-baseline-w01-w04.pdf (or -w02.pdf)", has_baseline, "missing exercises/release-baseline-w01-w04.pdf (or -w02.pdf)")


def check_tracked(rep: Report) -> None:
    tracked = [t for t in git("ls-files").splitlines() if t]
    lim = COURSE["limits"]
    bad = [t for t in tracked for g in lim["forbidden_globs"] if fnmatch.fnmatch(t, g)]
    rep.add("no forbidden files", not bad, ", ".join(sorted(set(bad))[:10]))
    big = [t for t in tracked if (ROOT / t).is_file() and (ROOT / t).stat().st_size > lim["max_file_mb"] * 2 ** 20]
    rep.add(f"no file over {lim['max_file_mb']} MB", not big, ", ".join(big[:10]))


def parse_progress() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    p = ROOT / "PROGRESS.md"
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or set(cells[0]) <= set("-: "):
            continue
        m = re.match(r"W(\d{2})", cells[0])
        if not m:
            continue
        rows[f"W{m.group(1)}"] = dict(zip(
            ["period", "topic", "post", "drill", "first", "revision", "tag", "status"], cells[:8]))
    return rows


def check_progress(rep: Report, week: int) -> None:
    rows = parse_progress()
    rep.add("PROGRESS.md has rows W01-W04 and W05-W15", "W01" in rows and all(
        f"W{w:02d}" in rows for w in range(5, 16)), "rows missing")
    for w in range(5, week + 1):
        key = f"W{w:02d}"
        r = rows.get(key)
        if r is None:
            continue
        for col in ("post", "drill"):
            links = md_links(r[col])
            missing = [l for l in links if not (ROOT / l).exists()]
            if w == 8 and not links:
                continue                               # midterm: link the compact entry / rehearsal if you have them
            rep.add(f"{key} {col} link", bool(links) and not missing,
                    "no link" if not links else "broken: " + ", ".join(missing))
        if w == 8:
            continue                                   # midterm exception: no two-state rule
        f, v = HEX.findall(r["first"]), HEX.findall(r["revision"])
        if not f or not v:
            rep.add(f"{key} first/revision commits", False, "hash missing in PROGRESS.md")
            continue
        f, v = f[0], v[0]
        ok_exist = commit_exists(f) and commit_exists(v)
        rep.add(f"{key} first/revision commits exist", ok_exist, f"{f} / {v}")
        if ok_exist:
            rep.add(f"{key} first attempt precedes revision",
                    is_ancestor(f, v) and git("rev-parse", f) != git("rev-parse", v), f"{f} -> {v}")


def check_posts(rep: Report, week: int) -> None:
    lim = COURSE["limits"]["post_words"]
    cu = ROOT / "docs" / "pre-release" / "PRE_RELEASE_CATCHUP.md"
    if cu.is_file():
        n = words(cu.read_text(encoding="utf-8"))
        rep.add("catch-up post length", lim["catchup"][0] <= n <= lim["catchup"][1], f"{n} words")
    ai_weeks = ai_use_weeks()
    for w in range(5, week + 1):
        if w == 8:
            continue
        posts = sorted((ROOT / "docs" / "weekly").glob(f"w{w:02d}-*.md"))
        if not posts:
            rep.add(f"W{w:02d} weekly post", False, f"no docs/weekly/w{w:02d}-*.md")
            continue
        text = posts[0].read_text(encoding="utf-8")
        n = words(text)
        rep.add(f"W{w:02d} post length", lim["weekly"][0] <= n <= lim["weekly"][1], f"{n} words")
        need = POST_SECTIONS + (["H"] if w in ai_weeks else [])
        miss = [s for s in need if not re.search(rf"^#{{2,3}}\s*{s}[.)]", text, re.M)]
        rep.add(f"W{w:02d} post sections", not miss, "missing " + ", ".join(miss))
        left = re.findall(r"\{\{[^}]*\}\}|TODO", text)
        rep.add(f"W{w:02d} post template text removed", not left, f"{len(left)} placeholder(s)")


def ai_use_weeks() -> set[int]:
    p = ROOT / "AI_USE.md"
    if not p.is_file():
        return set()
    return {int(m) for m in re.findall(r"^\|\s*W(\d{2})\b", p.read_text(encoding="utf-8"), re.M)}


def check_ai_use(rep: Report) -> None:
    p = ROOT / "AI_USE.md"
    if not p.is_file():
        return
    bad = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*W\d{2}\b", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or any(not c for c in cells[:8]):
            bad.append(cells[0] + " incomplete row")
            continue
        shas = HEX.findall(cells[2])
        if shas and not commit_exists(shas[0]):
            bad.append(f"{cells[0]} pre-AI commit {shas[0]} not found")
    rep.add("AI_USE.md rows complete", not bad, "; ".join(bad[:5]))


def check_protocol(rep: Report) -> None:
    p = ROOT / "protocol.yaml"
    if not p.is_file():
        return
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        rep.add("protocol.yaml parses", False, str(e)[:120])
        return
    empty = [k for k in PROTOCOL_KEYS if d.get(k) in (None, "", [], "TBD")]
    rep.add("protocol.yaml frozen fields filled", not empty, "empty: " + ", ".join(empty))


def added_at(path: str) -> str:
    """Commit that first added path (oldest one if it was deleted and re-added)."""
    out = git("log", "--diff-filter=A", "--format=%H", "--", path).splitlines()
    return out[-1] if out else ""


def check_drills(rep: Report, week: int) -> None:
    for w in range(5, week + 1):
        if w == 8:
            continue
        first = sorted((ROOT / "exercises").glob(f"w{w:02d}-first-attempt.*")) or sorted((ROOT / "exercises").glob(f"w{w:02d}-drill.*"))
        corr = ROOT / "exercises" / f"w{w:02d}-corrections.md"
        if not first:
            rep.add(f"W{w:02d} drill first attempt", False, f"no exercises/w{w:02d}-first-attempt.*")
            continue
        if not corr.is_file():
            rep.add(f"W{w:02d} drill corrections", False, f"no exercises/w{w:02d}-corrections.md")
            continue
        a = added_at(str(first[0].relative_to(ROOT)))
        b = added_at(str(corr.relative_to(ROOT)))
        rep.add(f"W{w:02d} drill committed before corrections",
                bool(a and b) and a != b and is_ancestor(a, b), "same commit or wrong order")


HW_SCAN_SUFFIXES = {".pdf"}


def check_homework(rep: Report, hw: str) -> None:
    """A homework checkpoint: one scanned PDF, committed before any corrections file."""
    scans = sorted((ROOT / "homework").glob(f"{hw}-submission.*"))
    rep.add(f"{hw} scan homework/{hw}-submission.pdf", bool(scans),
            f"no homework/{hw}-submission.pdf" if not scans else "")
    if not scans:
        return
    bad = [s.name for s in scans if s.suffix.lower() not in HW_SCAN_SUFFIXES]
    rep.add(f"{hw} scan is a single PDF", len(scans) == 1 and not bad,
            "merge all pages into one PDF: " + ", ".join(s.name for s in scans))
    rel = str(scans[0].relative_to(ROOT))
    added = added_at(rel)
    rep.add(f"{hw} scan committed", bool(added), "commit the scan before tagging")
    corr = ROOT / "homework" / f"{hw}-corrections.md"
    if corr.is_file() and added:
        b = added_at(str(corr.relative_to(ROOT)))
        rep.add(f"{hw} scan committed before corrections",
                bool(b) and b != added and is_ancestor(added, b), "same commit or wrong order")


def check_commit_messages(rep: Report) -> None:
    since = COURSE["release_date"]
    msgs = [m for m in git("log", f"--since={since}", "--format=%s").splitlines() if m]
    good = [m for m in msgs if re.match(r"^\[W\d{2}\]\[[a-z-]+\]", m)]
    share = len(good) / len(msgs) if msgs else 0.0
    rep.add("commit messages follow [Wnn][kind]", share >= 0.8,
            f"{len(good)}/{len(msgs)} match", level="warn")


def run_contract_tests(rep: Report, week: int) -> None:
    env = dict(os.environ, CO3117_WEEK=f"W{week:02d}", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfs", "tests/contract"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    tail = "\n".join(r.stdout.strip().splitlines()[-15:])
    rep.add("contract tests", r.returncode in (0, 5), tail)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="w05..w15, release-baseline, part1-final, part2-final, hw1..hw5")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    cp = (a.checkpoint or os.environ.get("CO3117_CHECKPOINT") or "").strip().lower()
    week = week_of_checkpoint(cp) if cp else week_from_date(dt.datetime.now())
    rep = Report()
    if cp in COURSE.get("homework", {}):
        # A homework tag checks only the homework scan and the repository-wide file rules; the
        # weekly items are checked at their own tags.
        check_tracked(rep)
        check_homework(rep, cp)
        a.no_tests = True
    else:
        check_files(rep, cp)
        check_tracked(rep)
        check_protocol(rep)
        check_progress(rep, week)
        check_posts(rep, week)
        check_drills(rep, week)
        check_ai_use(rep)
        check_commit_messages(rep)
    if not a.no_tests:
        run_contract_tests(rep, week)

    print(f"CO3117 self-check  checkpoint={cp or '(by date)'}  week=W{week:02d}")
    for i in rep.items:
        mark = {"ok": "PASS", "fail": "FAIL", "warn": "WARN"}[i["level"]]
        detail = "" if i["ok"] else f"  ({i['detail']})"
        print(f"  {mark}  {i['check']}{detail}" if "\n" not in detail else f"  {mark}  {i['check']}\n{i['detail']}")
    n_fail = len(rep.failed)
    print(f"\n{len(rep.items) - n_fail}/{len(rep.items)} checks pass" + ("" if not n_fail else f", {n_fail} to fix"))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"checkpoint": cp, "week": week, "items": rep.items},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
