"""Top-level CLI entry point for running the Dandelion4 ASR + NLP pipeline.

Run ``python -m src.main --help`` for usage instructions and examples.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import pipeline  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Dandelion4 pipeline on any audio/text file, .zip archive, or folder."""
    return pipeline.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
