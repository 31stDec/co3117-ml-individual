"""Shared logic for the contract tests.

The checkpoint week comes from the environment variable CO3117_WEEK (for example W07);
tools/check.py sets it from the tag being checked. Without it every component is treated
as optional, so an unfinished stub is skipped and never fails a local run.
"""
import os
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COURSE = yaml.safe_load((ROOT / "tools" / "course.yaml").read_text(encoding="utf-8"))


def _num(week):
    return int(str(week).upper().lstrip("W"))


def is_required(component):
    week = os.environ.get("CO3117_WEEK", "").strip()
    if not week:
        return False
    return _num(week) >= _num(COURSE["components"][component])


def call(component, fn, *args, **kwargs):
    """Run fn; turn NotImplementedError into skip (optional) or failure (required)."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        due = COURSE["components"][component]
        if is_required(component):
            pytest.fail(f"{component}: not implemented, required from {due}")
        pytest.skip(f"{component}: not implemented yet (required from {due})")


def implemented(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True
