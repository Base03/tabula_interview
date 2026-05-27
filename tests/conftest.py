"""Shared pytest fixtures for the test suite.

Currently provides one fixture, `paper_repo`, which locates the upstream
mdmparis/coli_phage_interactions_2023 clone at `<project_root>/paper-repo/`
and auto-clones it if missing. Session-scoped so a single fresh clone is
amortized across all tests that need the data.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_REPO_URL = "https://github.com/mdmparis/coli_phage_interactions_2023.git"
PAPER_REPO_PATH = PROJECT_ROOT / "paper-repo"


@pytest.fixture(scope="session")
def paper_repo() -> Path:
    """Locate the paper repo on disk; clone it to PAPER_REPO_PATH if missing.

    Skips dependent tests if cloning fails (e.g. no network).
    """
    if not PAPER_REPO_PATH.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", PAPER_REPO_URL, str(PAPER_REPO_PATH)],
                check=True, capture_output=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            pytest.skip(f"Cannot clone paper repo (no network?): {exc}")
    if not (PAPER_REPO_PATH / "data" / "interactions" / "interaction_matrix.csv").exists():
        pytest.skip(
            f"Paper repo at {PAPER_REPO_PATH} exists but is missing expected files. "
            f"Re-clone with `rm -rf {PAPER_REPO_PATH} && git clone {PAPER_REPO_URL} {PAPER_REPO_PATH}`."
        )
    return PAPER_REPO_PATH
