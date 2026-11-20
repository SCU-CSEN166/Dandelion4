"""Download and prepare public-speech evaluation suites locally when missing.

Prepares:
- ``evaluation/datasets/test_300_seed1234.npz`` (300 clips from ``robello2/mini_afrispeech``)
- ``evaluation/datasets/test_330_balanced.npz`` (330 balanced clips across test shards)
- ``evaluation/datasets/kenyan_ea_test.npz`` (Kenyan/EA test shards)
- ``examples/fleurs_manifest.json`` + ``examples/fleurs_audio/`` (``google/fleurs`` ``en_us`` test)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_whisper_benchmarks as bench  # noqa: E402
from src.asr import transcribe  # noqa: E402
from src.training import fetch_fleurs, finetune_whisper  # noqa: E402

DATASETS_DIR = REPO_ROOT / "evaluation" / "datasets"
AFRISPEECH_DATASET = "robello2/mini_afrispeech"
EXTRA_AFRISPEECH_DATASET = "Professor/afrispeech-200-16khz"
PUBLIC_SUITES: tuple[str, ...] = (
    "test_300_seed1234",
    "test_330_balanced",
    "kenyan_ea_test",
    "fleurs_all",
)


def save_local_suite(
    name: str,
    items: Sequence[bench.BenchmarkItem],
    *,
    datasets_dir: Path = DATASETS_DIR,
) -> Path:
    """Save a sequence of ``BenchmarkItem`` clips to ``<datasets_dir>/<name>.npz``."""
    if not items:
        raise ValueError(f"cannot save empty benchmark suite: {name}")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    out_path = datasets_dir / f"{name}.npz"
    audio = np.concatenate([np.asarray(item.audio, dtype=np.float32) for item in items])
    lengths = [len(item.audio) for item in items]
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    np.savez(
        out_path,
        audio=audio,
        offsets=offsets,
        text=np.asarray([item.text for item in items]),
        domain=np.asarray([item.domain for item in items]),
        accent=np.asarray([item.accent for item in items]),
    )
    return out_path


def _prepare_test_300_seed1234(datasets_dir: Path) -> Path:  # pragma: no cover - network
    rows = finetune_whisper.stream_hf_afrispeech(AFRISPEECH_DATASET, "default", "test", seed=1234)
    items = bench._rows_to_items(rows, limit=300)
    return save_local_suite("test_300_seed1234", items, datasets_dir=datasets_dir)


def _prepare_test_330_balanced(datasets_dir: Path) -> Path:  # pragma: no cover - network
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(AFRISPEECH_DATASET, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("data/test-") and f.endswith(".parquet"))
    items: list[bench.BenchmarkItem] = []
    for shard in shards:
        rows = finetune_whisper.stream_hf_parquet_files(AFRISPEECH_DATASET, [shard])
        items.extend(bench._rows_to_items(rows, limit=30))
    return save_local_suite("test_330_balanced", items, datasets_dir=datasets_dir)


def _prepare_kenyan_ea_test(datasets_dir: Path) -> Path:  # pragma: no cover - network
    rows = finetune_whisper.stream_hf_parquet_files(
        EXTRA_AFRISPEECH_DATASET, list(bench.KENYAN_EA_TEST_SHARDS)
    )
    items = bench._rows_to_items(rows)
    return save_local_suite("kenyan_ea_test", items, datasets_dir=datasets_dir)


def is_fleurs_ready(
    manifest_path: Path = fetch_fleurs.DEFAULT_MANIFEST,
    root: Path = REPO_ROOT,
) -> bool:
    """Return True if *manifest_path* exists and all referenced WAV files are present."""
    if not manifest_path.is_file():
        return False
    manifest = transcribe.load_manifest(manifest_path)
    clips = manifest.get("clips", [])
    if not clips:
        return False
    return all(transcribe.clip_path(clip, root).is_file() for clip in clips)


def prepare_fleurs(
    manifest_path: Path = fetch_fleurs.DEFAULT_MANIFEST,
    *,
    out_dir: Path = fetch_fleurs.DEFAULT_OUT_DIR,
    messages_out: Path = fetch_fleurs.DEFAULT_MESSAGES_OUT,
    root: Path = REPO_ROOT,
) -> Path:  # pragma: no cover - network
    """Download the full ``google/fleurs`` ``en_us`` test split into ``examples/fleurs_audio/``."""
    if is_fleurs_ready(manifest_path, root=root):
        return manifest_path
    print("Downloading google/fleurs (en_us, test) into examples/fleurs_audio/...", flush=True)
    fetch_fleurs.fetch_fleurs(
        config=fetch_fleurs.DEFAULT_CONFIG,
        split=fetch_fleurs.DEFAULT_SPLIT,
        limit=10_000,
        out_dir=out_dir,
        manifest_path=manifest_path,
        messages_path=messages_out,
        root=root,
    )
    return manifest_path


def prepare_suite(name: str, *, datasets_dir: Path = DATASETS_DIR) -> Path:
    """Ensure public evaluation suite *name* is prepared locally, downloading if missing."""
    if name == "fleurs_all":
        return prepare_fleurs()
    target = datasets_dir / f"{name}.npz"
    if target.is_file():
        return target
    builders: dict[str, Any] = {
        "test_300_seed1234": _prepare_test_300_seed1234,
        "test_330_balanced": _prepare_test_330_balanced,
        "kenyan_ea_test": _prepare_kenyan_ea_test,
    }
    if name not in builders:
        raise ValueError(f"Unknown public evaluation suite: {name!r}")
    print(f"Downloading public evaluation suite {name!r} into {datasets_dir}...", flush=True)
    return Path(builders[name](datasets_dir))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for downloading public evaluation datasets."""
    parser = argparse.ArgumentParser(
        description="Download and prepare public-speech evaluation suites locally."
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(PUBLIC_SUITES),
        choices=list(PUBLIC_SUITES),
        help="Public evaluation suites to prepare locally.",
    )
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """CLI entry point for preparing public evaluation datasets."""
    args = parse_args(argv)
    for suite in args.suites:
        path = prepare_suite(suite, datasets_dir=args.datasets_dir)
        print(f"Ready: {suite} -> {path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
