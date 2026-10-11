"""Shared test configuration.

Puts the repository root on sys.path so tests import the packages under
test regardless of the working directory pytest was invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RESULTS = ROOT / "out" / "results"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def bc5cdr_dev():
    """Parsed BC5CDR development split, or skip if the corpus is absent."""
    path = DATA / "bc5cdr" / "CDR_DevelopmentSet.PubTator.txt"
    if not path.exists():
        pytest.skip("BC5CDR corpus not present")
    from pipeline.loaders import parse_pubtator_file
    return parse_pubtator_file(path)


@pytest.fixture(scope="session")
def reachability_rows():
    """Baseline reachability ledger, or skip if the run has not been made."""
    path = RESULTS / "reachability_dev.csv"
    if not path.exists():
        pytest.skip("reachability_dev.csv not present; run retrieval_eval")
    import csv
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
