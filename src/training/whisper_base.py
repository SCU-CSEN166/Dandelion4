"""Shared helper for checking and downloading the base Whisper large-v3 checkpoint."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_WHISPER_LARGE_V3_MODEL_ID = "openai/whisper-large-v3"
BASE_WHISPER_LARGE_V3_DIR = REPO_ROOT / ".models" / "whisper-large-v3"

_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def is_local_whisper_model_ready(model_dir: Path) -> bool:
    """Return True if *model_dir* contains a Whisper config and model weights."""
    if not (model_dir / "config.json").is_file():
        return False
    return any((model_dir / name).is_file() for name in _WEIGHT_FILES)


def ensure_base_whisper_large_v3(
    model_dir: Path = BASE_WHISPER_LARGE_V3_DIR,
    *,
    model_id: str = BASE_WHISPER_LARGE_V3_MODEL_ID,
) -> Path:
    """Ensure base Whisper large-v3 is downloaded locally, returning its directory."""
    if is_local_whisper_model_ready(model_dir):
        print(f"Base Whisper large-v3 already present at {model_dir}", flush=True)
        return model_dir

    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} to {model_dir}...", flush=True)
    snapshot_download(repo_id=model_id, local_dir=str(model_dir))
    return model_dir


def resolve_training_base_model(
    base_model: str | Path,
    *,
    base_dir: Path = BASE_WHISPER_LARGE_V3_DIR,
    model_id: str = BASE_WHISPER_LARGE_V3_MODEL_ID,
) -> str:
    """Resolve a training base model, downloading base Whisper large-v3 if needed."""
    text = str(base_model)
    candidate = Path(text)
    if is_local_whisper_model_ready(candidate):
        return str(candidate)
    if text == model_id or candidate == base_dir:
        if is_local_whisper_model_ready(base_dir):
            return str(base_dir)
        return str(ensure_base_whisper_large_v3(base_dir, model_id=model_id))
    return text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the base Whisper large-v3 downloader."""
    parser = argparse.ArgumentParser(
        description="Check for local base Whisper large-v3 and download it if missing."
    )
    parser.add_argument("--model-id", default=BASE_WHISPER_LARGE_V3_MODEL_ID)
    parser.add_argument("--out-dir", type=Path, default=BASE_WHISPER_LARGE_V3_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)
    ensure_base_whisper_large_v3(args.out_dir, model_id=args.model_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
