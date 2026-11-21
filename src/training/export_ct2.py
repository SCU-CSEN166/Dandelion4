"""Export fine-tuned Hugging Face Whisper models to quantized CTranslate2 INT8 format."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def dir_size_mb(path: Path) -> float:
    """Return total size in MiB of all files under *path*."""
    total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def export_ct2_model(
    model_dir: Path,
    out_dir: Path,
    *,
    quantization: str = "int8",
    force: bool = True,
    verify: bool = True,
) -> Path:  # pragma: no cover - converts real Whisper weights
    """Convert a Hugging Face Whisper directory to CTranslate2 and verify offline CPU load."""
    import ctranslate2
    from faster_whisper import WhisperModel
    from transformers import WhisperFeatureExtractor

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json in {model_dir}")

    if force and out_dir.exists():
        shutil.rmtree(out_dir)

    copy_candidates = ["tokenizer.json", "preprocessor_config.json", "generation_config.json"]
    copy_files = [name for name in copy_candidates if (model_dir / name).is_file()]

    converter: Any = ctranslate2.converters.TransformersConverter(
        str(model_dir),
        copy_files=copy_files,
    )
    converter.convert(str(out_dir), quantization=quantization, force=force)

    if not (out_dir / "preprocessor_config.json").is_file():
        fe = WhisperFeatureExtractor.from_pretrained(str(model_dir))
        fe.save_pretrained(str(out_dir))

    if verify:
        model = WhisperModel(
            str(out_dir),
            device="cpu",
            compute_type=quantization,
            local_files_only=True,
        )
        silence = np.zeros(16_000, dtype=np.float32)
        segments, _ = model.transcribe(
            silence, language="en", task="transcribe", beam_size=1, condition_on_previous_text=False
        )
        _ = list(segments)

    print(
        f"Exported {model_dir} -> {out_dir} (quantization={quantization}, "
        f"size={dir_size_mb(out_dir):.1f} MB)",
        flush=True,
    )
    return out_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for CTranslate2 Whisper export."""
    parser = argparse.ArgumentParser(
        description="Convert a Whisper HF model directory to CTranslate2 format for CPU edge use."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
    )
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """Export a fine-tuned Whisper checkpoint to CTranslate2 format."""
    args = parse_args(argv)
    export_ct2_model(
        args.model,
        args.out_dir,
        quantization=args.quantization,
        verify=not args.no_verify,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
