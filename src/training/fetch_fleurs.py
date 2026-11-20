"""Fetch a small sample of real human speech from google/fleurs.

Writes 16 kHz mono PCM-16 WAV files, an ``audio_manifest.json``-compatible
manifest for ``src/transcribe.py``, and a ``Message``-format reference JSON
via ``contracts.write_json`` for ``evaluation/eval_wer.py``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
import tarfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import soundfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import contracts  # noqa: E402

DEFAULT_CONFIG = "en_us"
DEFAULT_SPLIT = "test"
DEFAULT_LIMIT = 20
DEFAULT_OUT_DIR = REPO_ROOT / "examples" / "fleurs_audio"
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "fleurs_manifest.json"
DEFAULT_MESSAGES_OUT = REPO_ROOT / "examples" / "fleurs_messages.json"
TARGET_SAMPLE_RATE = 16_000
DATASET_ID = "google/fleurs"
DEFAULT_VERSION = "clean"

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class FleursSample:
    """Normalized speech sample yielded by a FLEURS stream adapter."""

    sample_id: str
    speaker_id: str
    audio: FloatArray
    sample_rate: int
    text: str


StreamFleurs = Callable[[str, str, int], Iterable[FleursSample]]


def to_mono_16k(wave: npt.NDArray[Any], sr: int) -> FloatArray:
    """Average channels (shape ``(frames, channels)``) and resample to 16 kHz float32."""
    if sr <= 0:
        raise ValueError(f"sample_rate must be positive, got {sr}")
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    elif mono.ndim != 1:
        raise ValueError(f"expected a 1-D or 2-D wave, got shape {mono.shape}")
    if sr != TARGET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def repo_relative(path: Path, repo_root: Path) -> str:
    """Return ``path`` relative to ``repo_root`` with forward slashes."""
    resolved = path.resolve()
    root = repo_root.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def format_utc(moment: datetime) -> str:
    """Format a datetime as ``YYYY-MM-DDTHH:MM:SSZ`` in UTC."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_wav(path: Path, wave: FloatArray, sr: int = TARGET_SAMPLE_RATE) -> None:
    """Write a mono 16-bit PCM WAV file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, wave, sr, subtype="PCM_16")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write an audio manifest JSON file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _sample_from_hf_row(row: dict[str, Any], index: int) -> FleursSample:
    import io

    audio_info = row["audio"]
    if "array" in audio_info and "sampling_rate" in audio_info:
        array = np.asarray(audio_info["array"], dtype=np.float32)
        sr = int(audio_info["sampling_rate"])
    elif audio_info.get("bytes"):
        data, sr = soundfile.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        array = np.asarray(data, dtype=np.float32)
    elif audio_info.get("path"):
        data, sr = soundfile.read(audio_info["path"], dtype="float32")
        array = np.asarray(data, dtype=np.float32)
    else:
        raise ValueError(f"sample {index} has neither array nor audio bytes/path")
    text = str(row.get("transcription") or row.get("raw_transcription") or "").strip()
    sample_id = str(row.get("id") or index)
    speaker_id = f"fleurs_{sample_id}"
    return FleursSample(
        sample_id=sample_id,
        speaker_id=speaker_id,
        audio=array,
        sample_rate=sr,
        text=text,
    )


def _stream_via_datasets(config: str, split: str, limit: int) -> Iterable[FleursSample]:
    datasets_mod = importlib.import_module("datasets")
    ds = datasets_mod.load_dataset(DATASET_ID, config, split=split, streaming=True)
    if hasattr(ds, "cast_column"):
        ds = ds.cast_column("audio", datasets_mod.Audio(decode=False))
    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        yield _sample_from_hf_row(dict(row), idx)


def _stream_via_hub(config: str, split: str, limit: int) -> Iterable[FleursSample]:
    hub_mod = importlib.import_module("huggingface_hub")
    tsv_path = Path(
        hub_mod.hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"data/{config}/{split}.tsv",
        )
    )
    tar_path = Path(
        hub_mod.hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"data/{config}/audio/{split}.tar.gz",
        )
    )

    by_filename: dict[str, tuple[str, str]] = {}
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            sample_id, file_name, raw_text, norm_text = row[0], row[1], row[2], row[3]
            text = (norm_text or raw_text).strip()
            by_filename[Path(file_name).name] = (sample_id, text)

    count = 0
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in by_filename:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw_bytes = extracted.read()
            data, sr = soundfile.read(io.BytesIO(raw_bytes), dtype="float32")
            sample_id, text = by_filename[name]
            yield FleursSample(
                sample_id=sample_id,
                speaker_id=f"fleurs_{sample_id}",
                audio=np.asarray(data, dtype=np.float32),
                sample_rate=int(sr),
                text=text,
            )
            count += 1
            if count >= limit:
                break


def stream_fleurs_samples(config: str, split: str, limit: int) -> Iterable[FleursSample]:
    """Stream up to ``limit`` samples from ``google/fleurs``.

    Uses HuggingFace ``datasets`` when installed, otherwise falls back to
    ``huggingface_hub`` archive download.
    """
    try:
        importlib.import_module("datasets")
    except ModuleNotFoundError:
        yield from _stream_via_hub(config, split, limit)
    else:
        yield from _stream_via_datasets(config, split, limit)


def fetch_fleurs(
    *,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    limit: int = DEFAULT_LIMIT,
    out_dir: Path = DEFAULT_OUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    messages_path: Path = DEFAULT_MESSAGES_OUT,
    root: Path = REPO_ROOT,
    stream_fn: StreamFleurs = stream_fleurs_samples,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch FLEURS speech clips, write 16 kHz mono WAVs, manifest, and reference messages."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    samples: list[FleursSample] = []
    for sample in stream_fn(config, split, limit):
        samples.append(sample)
        if len(samples) >= limit:
            break

    if not samples:
        raise ValueError(f"no samples returned for {DATASET_ID} ({config}/{split})")

    width = max(4, len(str(len(samples))))
    clips: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples, start=1):
        message_id = f"fleurs_{idx:0{width}d}"
        sender_id = sample.speaker_id.strip() or f"fleurs_{config}"
        text = sample.text.strip()
        if not text:
            raise ValueError(f"sample {sample.sample_id!r} has empty reference text")

        wave = to_mono_16k(sample.audio, sample.sample_rate)
        n_samples = int(wave.shape[0])
        wav_path = out_dir / f"{message_id}_{DEFAULT_VERSION}.wav"
        write_wav(wav_path, wave, TARGET_SAMPLE_RATE)

        clips.append(
            {
                "clip_id": f"{message_id}_{DEFAULT_VERSION}",
                "message_id": message_id,
                "sender_id": sender_id,
                "version": DEFAULT_VERSION,
                "voice": f"fleurs_{config}",
                "path": repo_relative(wav_path, root),
                "sample_rate": TARGET_SAMPLE_RATE,
                "n_samples": n_samples,
                "duration_s": round(n_samples / TARGET_SAMPLE_RATE, 2),
                "reference_text": text,
                "snr_db_measured": None,
                "noise_seed": None,
            }
        )
        messages.append(
            {
                "id": message_id,
                "sender_id": sender_id,
                "source": "audio",
                "message": text,
            }
        )

    manifest: dict[str, Any] = {
        "generator": {
            "script": "src/training/fetch_fleurs.py",
            "created_utc": format_utc(now()),
            "dataset": DATASET_ID,
            "config": config,
            "split": split,
            "limit": len(clips),
            "sample_rate": TARGET_SAMPLE_RATE,
            "source_messages": repo_relative(messages_path, root),
        },
        "clips": clips,
    }
    write_manifest(manifest_path, manifest)
    messages_path.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_json(messages_path, messages)
    return manifest, messages


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for fetching FLEURS audio samples."""
    parser = argparse.ArgumentParser(
        description="Fetch real human speech samples from google/fleurs."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="FLEURS language config")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max clips to fetch")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--messages-out", type=Path, default=DEFAULT_MESSAGES_OUT)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    stream_fn: StreamFleurs = stream_fleurs_samples,
    root: Path = REPO_ROOT,
) -> int:
    """Command-line entry point. Returns process exit code."""
    args = parse_args(argv)
    try:
        manifest, messages = fetch_fleurs(
            config=args.config,
            split=args.split,
            limit=args.limit,
            out_dir=args.out_dir,
            manifest_path=args.manifest,
            messages_path=args.messages_out,
            root=root,
            stream_fn=stream_fn,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {len(manifest['clips'])} clips and {len(messages)} messages "
        f"-> {args.manifest}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
