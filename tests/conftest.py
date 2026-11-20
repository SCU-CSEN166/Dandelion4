"""Shared pytest setup: make src/ and evaluation/ importable, plus common fixtures."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def messages(repo_root: Path) -> list[dict[str, str]]:
    """The records of examples/synthetic_messages.json."""
    path = repo_root / "examples" / "synthetic_messages.json"
    with path.open(encoding="utf-8") as handle:
        data: list[dict[str, str]] = json.load(handle)
    return data
