"""M3 - Baseline runner: run the unmodified text baseline without touching the repo.

``src/baseline.py`` runs all of its code at import time and always writes
``examples/baseline_output.json`` relative to the current working directory. This module
runs it as a subprocess inside a temporary working directory, so the committed
``examples/baseline_output.json`` is never overwritten, and returns the records it wrote.

The baseline script is never imported.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_SCRIPT = REPO_ROOT / "src" / "baseline.py"
TIMEOUT_SECONDS = 60
OUTPUT_RELATIVE_PATH = Path("examples") / "baseline_output.json"


class BaselineError(RuntimeError):
    """The baseline subprocess failed, timed out, or wrote no usable output.

    ``stderr`` holds the subprocess's standard error (empty if there was none).
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        self.stderr = stderr
        full = f"{message}\n--- baseline stderr ---\n{stderr}" if stderr else message
        super().__init__(full)


def _as_text(stream: str | bytes | None) -> str:
    """Return captured subprocess output as text (it may be bytes or None on timeout)."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def run_baseline(
    input_json: Path, baseline_script: Path = DEFAULT_BASELINE_SCRIPT
) -> list[dict[str, Any]]:
    """Run the unmodified text baseline on input_json and return its records."""
    # The child prints through pipes; force UTF-8 there so the decoding below matches.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            completed = subprocess.run(
                [sys.executable, str(baseline_script.resolve()), str(input_json.resolve())],
                cwd=tmp_dir,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise BaselineError(
                f"baseline exited with code {exc.returncode} on {input_json}",
                _as_text(exc.stderr),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BaselineError(
                f"baseline timed out after {exc.timeout} s on {input_json}",
                _as_text(exc.stderr),
            ) from exc

        output_path = tmp_dir / OUTPUT_RELATIVE_PATH
        try:
            with output_path.open(encoding="utf-8") as handle:
                records: object = json.load(handle)
        except FileNotFoundError as exc:
            raise BaselineError(
                f"baseline wrote no {OUTPUT_RELATIVE_PATH.as_posix()} for {input_json}",
                completed.stderr,
            ) from exc
        except json.JSONDecodeError as exc:
            raise BaselineError(
                f"baseline output for {input_json} is not valid JSON: {exc}",
                completed.stderr,
            ) from exc

    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise BaselineError(
            f"baseline output for {input_json} is not a list of objects",
            completed.stderr,
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Print the baseline records for one input file as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("input_json", type=Path, help="messages file in the §2.1 format")
    parser.add_argument("--baseline-script", type=Path, default=DEFAULT_BASELINE_SCRIPT)
    args = parser.parse_args(argv)
    try:
        records = run_baseline(args.input_json, args.baseline_script)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
