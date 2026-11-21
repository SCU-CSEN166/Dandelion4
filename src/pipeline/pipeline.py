"""Flexible end-to-end runner for arbitrary audio, text, zip, or folder inputs.

Usage:
    python -m src.pipeline.run_pipeline path/to/file_or_folder
    python -m src.pipeline.run_pipeline path/to/folder --out-dir outputs/run1
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import clean_transcripts, transcribe  # noqa: E402
from src.core import contracts, setup_models  # noqa: E402
from src.nlp import gliner_system  # noqa: E402
from src.pipeline import hidden_test_pipeline  # noqa: E402

AUDIO_SUFFIXES = hidden_test_pipeline.SUPPORTED_AUDIO_SUFFIXES
TEXT_SUFFIXES = frozenset({".txt", ".md", ".json", ".jsonl", ".csv"})
DEFAULT_INPUT_DIR = REPO_ROOT / "demo" / "inputs"
DEFAULT_OUT_DIR = REPO_ROOT / "demo" / "outputs" / "custom_run"
DEFAULT_CHUNK_SIZE = 64

OUTPUT_CSV_COLUMNS: tuple[str, ...] = (
    "Message_id",
    "source_file",
    "source_type",
    "anonymized_text",
    "relevance",
    "activity_type",
    "participants",
    "duplicate_of",
)


def _is_ignored_path(path: Path) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in path.parts)


def _extract_zip_candidates(zip_path: Path, extract_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(zf.namelist()):
            rel = Path(member)
            if member.endswith("/") or _is_ignored_path(rel):
                continue
            extracted.append(Path(zf.extract(member, extract_dir)))
    return extracted


def discover_inputs(input_path: Path, extract_dir: Path) -> tuple[list[Path], list[Path]]:
    """Discover supported audio and text files from a file, .zip, or directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidates: list[Path] = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        candidates.extend(_extract_zip_candidates(input_path, extract_dir))
    elif input_path.is_dir():
        for idx, item in enumerate(sorted(input_path.rglob("*"))):
            if not item.is_file() or _is_ignored_path(item.relative_to(input_path)):
                continue
            if item.suffix.lower() == ".zip":
                candidates.extend(
                    _extract_zip_candidates(item, extract_dir / f"{item.stem}_{idx}")
                )
            else:
                candidates.append(item)
    elif input_path.is_file():
        candidates.append(input_path)
    else:
        raise ValueError(f"Unsupported input path: {input_path}")

    audio_files: list[Path] = []
    text_files: list[Path] = []
    for path in candidates:
        suffix = path.suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            audio_files.append(path)
        elif suffix in TEXT_SUFFIXES:
            text_files.append(path)

    if not audio_files and not text_files:
        raise ValueError(f"No supported audio or text files found in {input_path}")
    return audio_files, text_files


def _extract_text_from_obj(obj: Any) -> str | None:
    if isinstance(obj, str):
        text = obj.strip()
        return text or None
    if isinstance(obj, dict):
        for key in ("message", "text", "transcript", "asr_text", "body", "content"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def load_text_items(path: Path) -> list[tuple[str, str]]:
    """Return ``[(source_label, text), ...]`` from a .txt, .md, .json, .jsonl, or .csv file."""
    suffix = path.suffix.lower()
    items: list[tuple[str, str]] = []

    if suffix in {".txt", ".md"}:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            items.append((path.name, content))
        return items

    if suffix == ".jsonl":
        for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _extract_text_from_obj(obj)
            if text:
                items.append((f"{path.name}:{idx}", text))
        return items

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for idx, entry in enumerate(data, start=1):
                text = _extract_text_from_obj(entry)
                if text:
                    label = (
                        str(entry.get("id"))
                        if isinstance(entry, dict) and entry.get("id")
                        else f"{path.name}:{idx}"
                    )
                    items.append((label, text))
        else:
            text = _extract_text_from_obj(data)
            if text:
                items.append((path.name, text))
        return items

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader, start=1):
                text = _extract_text_from_obj(row)
                if text:
                    items.append((f"{path.name}:{idx}", text))
        return items

    return items


def _load_checkpoint(checkpoint_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not checkpoint_path.is_file():
        return items
    for raw in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            items.append(json.loads(line))
    return items


def _process_message_batch(
    batch: Sequence[tuple[int, str, str, str, str | None]],
    width: int,
    nlp: gliner_system.GLiNERSystem,
    checkpoint_fh: Any,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    meta: list[tuple[str, str]] = []
    for idx, label, src_type, text, asr_text in batch:
        msg: dict[str, Any] = {
            "id": f"msg_{idx:0{width}d}",
            "sender_id": f"worker_{idx:0{width}d}",
            "source": src_type,
            "message": text,
        }
        if asr_text is not None:
            msg["asr_text"] = asr_text
        messages.append(msg)
        meta.append((label, src_type))

    problems = contracts.validate_file("messages", messages)
    if problems:
        raise ValueError("Generated messages failed contract validation:\n" + "\n".join(problems))

    processed: list[dict[str, Any]] = []
    for msg, (label, src_type) in zip(messages, meta, strict=True):
        rec, loc_spans = nlp.process_message_detailed(msg)
        rec_problems = contracts.validate_file("records", [rec])
        if rec_problems:
            raise ValueError(
                "Extracted records failed contract validation:\n" + "\n".join(rec_problems)
            )
        anonymized = hidden_test_pipeline.anonymize_csv_text(
            str(msg["message"]), rec.get("redactions", []), loc_spans
        )
        entry = {
            "record": rec,
            "location_spans": loc_spans,
            "anonymized_text": anonymized,
            "source_file": label,
            "source_type": src_type,
        }
        checkpoint_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        processed.append(entry)
    checkpoint_fh.flush()
    return processed


def run_on_input(
    input_path: Path,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    device: str = "auto",
    model_id: str | None = None,
    num_beams: int = 1,
    initial_prompt: str | None = None,
    clean: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
    asr_backend: transcribe.ASRBackend | None = None,
    nlp_system: gliner_system.GLiNERSystem | None = None,
) -> dict[str, Any]:
    """Process any audio/text file, .zip, or directory in bounded chunks with JSONL resume."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "records.jsonl"
    if overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        audio_files, text_files = discover_inputs(input_path, Path(tmp))

        text_items: list[tuple[str, str]] = []
        for text_path in text_files:
            text_items.extend(load_text_items(text_path))

        total_items = len(audio_files) + len(text_items)
        if total_items == 0:
            raise ValueError(f"No non-empty audio or text content found in {input_path}")

        width = max(2, len(str(total_items)))
        entries = _load_checkpoint(checkpoint_path)
        done_sources = {str(item["source_file"]) for item in entries}

        pending_audio = [
            (idx, p)
            for idx, p in enumerate(audio_files, start=1)
            if p.name not in done_sources
        ]
        pending_text = [
            (idx, label, content)
            for idx, (label, content) in enumerate(text_items, start=len(audio_files) + 1)
            if label not in done_sources
        ]

        if pending_audio or pending_text:
            nlp = nlp_system or gliner_system.GLiNERSystem(device=device)
            asr: transcribe.ASRBackend | None = None
            if pending_audio:
                asr = asr_backend or transcribe.make_whisper(
                    transcribe.resolve_default_model(
                        Path(model_id) if model_id is not None else None,
                        device=device,
                    ),
                    device,
                    transcribe.DEFAULT_BATCH_SIZE,
                    initial_prompt=initial_prompt,
                    num_beams=num_beams,
                )
                asr.load()

            def _load_audio_chunk(
                chunk: Sequence[tuple[int, Path]],
            ) -> list[transcribe.FloatArray]:
                return [transcribe.load_for_asr(*transcribe.read_wav(path)) for _, path in chunk]

            try:
                tqdm_cls: Any = importlib.import_module("tqdm").tqdm
            except ImportError:
                tqdm_cls = None

            with checkpoint_path.open("a", encoding="utf-8", newline="\n") as ckpt_fh:
                if pending_audio:
                    audio_bar = (
                        tqdm_cls(total=len(pending_audio), desc="Processing audio", unit="clip")
                        if tqdm_cls is not None
                        else None
                    )
                    audio_chunks = [
                        pending_audio[start : start + chunk_size]
                        for start in range(0, len(pending_audio), chunk_size)
                    ]
                    try:
                        with ThreadPoolExecutor(max_workers=1) as prefetcher:
                            next_future = prefetcher.submit(_load_audio_chunk, audio_chunks[0])
                            for idx_chunk, chunk in enumerate(audio_chunks):
                                waves = next_future.result()
                                if idx_chunk + 1 < len(audio_chunks):
                                    next_future = prefetcher.submit(
                                        _load_audio_chunk, audio_chunks[idx_chunk + 1]
                                    )
                                assert asr is not None
                                transcripts = asr(waves)
                                del waves
                                batch: list[tuple[int, str, str, str, str | None]] = []
                                for (idx, path), raw_text in zip(chunk, transcripts, strict=True):
                                    cleaned = (
                                        clean_transcripts.clean_text(raw_text)
                                        if clean
                                        else raw_text.strip()
                                    )
                                    batch.append(
                                        (idx, path.name, "audio", cleaned, raw_text.strip())
                                    )
                                entries.extend(_process_message_batch(batch, width, nlp, ckpt_fh))
                                if audio_bar is not None:
                                    audio_bar.update(len(chunk))
                    finally:
                        if audio_bar is not None:
                            audio_bar.close()

                text_bar = (
                    tqdm_cls(total=len(pending_text), desc="Processing text", unit="item")
                    if tqdm_cls is not None and pending_text
                    else None
                )
                try:
                    for start in range(0, len(pending_text), chunk_size):
                        text_chunk = pending_text[start : start + chunk_size]
                        text_batch = [
                            (idx, label, "text", content, None)
                            for idx, label, content in text_chunk
                        ]
                        entries.extend(_process_message_batch(text_batch, width, nlp, ckpt_fh))
                        if text_bar is not None:
                            text_bar.update(len(text_chunk))
                finally:
                    if text_bar is not None:
                        text_bar.close()

        records = [entry["record"] for entry in entries]
        location_spans_per_msg = [entry["location_spans"] for entry in entries]
        activity_labels = [hidden_test_pipeline.format_activity_label(rec) for rec in records]
        redacted_msgs = [
            {"id": str(rec["id"]), "message": str(rec["text_redacted"])} for rec in records
        ]
        duplicates = hidden_test_pipeline.find_duplicate_events(
            redacted_msgs, records, location_spans_per_msg, activity_labels
        )

        rows: list[dict[str, str]] = []
        for entry, rec, act_label, dup in zip(
            entries, records, activity_labels, duplicates, strict=True
        ):
            relevant = bool(rec.get("relevant"))
            count = rec.get("reported_count")
            rows.append(
                {
                    "Message_id": str(rec["id"]),
                    "source_file": str(entry["source_file"]),
                    "source_type": str(entry["source_type"]),
                    "anonymized_text": str(entry["anonymized_text"]),
                    "relevance": "Event" if relevant else "Chit-chat",
                    "activity_type": act_label if relevant else "",
                    "participants": str(count) if (relevant and isinstance(count, int)) else "",
                    "duplicate_of": dup,
                }
            )

        records_path = out_dir / "records.json"
        csv_path = out_dir / "outputs.csv"
        transcribe.write_json(records_path, records)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

        return {
            "audio_count": len(audio_files),
            "text_count": len(text_items),
            "total_records": len(rows),
            "checkpoint_path": str(checkpoint_path),
            "records_path": str(records_path),
            "csv_path": str(csv_path),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the general-purpose pipeline runner."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Dandelion4 Audio & Text Pipeline\n"
            "--------------------------------\n"
            "Turns field voice notes and text reports into a clean, redacted CSV table\n"
            "and structured JSON files. Runs on CPU by default (no GPU required) and\n"
            "automatically downloads the pre-trained models on first run if missing.\n\n"
            "Supported inputs (default: demo/inputs when no path is given):\n"
            "  - A single audio file (.wav, .mp3, .mp4, .m4a, .ogg, .flac, .aac, ...)\n"
            "  - A single text or data file (.txt, .md, .json, .jsonl, .csv)\n"
            "  - A .zip archive containing audio and/or text files\n"
            "  - A folder (searched recursively for all supported audio & text files)\n\n"
            "Outputs written to --out-dir (default: demo/outputs/custom_run):\n"
            "  - outputs.csv  : Reviewable spreadsheet (redacted text, event, count, duplicates)\n"
            "  - records.json : Full pipeline-processed records (with text_redacted only)"
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.main\n"
            "  python -m src.main --in-dir path/to/field_reports_folder --out-dir outputs/week10\n"
            "  python -m src.main path/to/voice_note.mp4\n"
            "  python -m src.main demo/inputs --device cpu\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Optional file, .zip archive, or folder to process "
            f"(default: {DEFAULT_INPUT_DIR} when neither input nor --in-dir is given)."
        ),
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help=(
            "Folder, file, or .zip archive of audio/text reports to process "
            f"(default: {DEFAULT_INPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Folder where outputs.csv and records.json are saved "
            f"(default: {DEFAULT_OUT_DIR})."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Hardware to run on: "auto" (uses GPU if available, else CPU), "cpu", or "cuda:0".',
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional path or Hugging Face ID for a custom Whisper speech model.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help=(
            "Beam search width for speech recognition "
            "(1 = fast greedy default, 3 = higher accuracy)."
        ),
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional vocabulary hint passed to Whisper (e.g., local place names).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip post-ASR transcript normalization (keep raw Whisper text).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Items processed per memory-bounded chunk and checkpoint flush "
            f"(default: {DEFAULT_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing records.jsonl checkpoint in --out-dir and start fresh.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """CLI entry point: ensure runtime models exist, run the pipeline, and print a JSON summary."""
    args = parse_args(argv)
    if (
        args.model is None
        and os.environ.get("HF_HUB_OFFLINE") != "1"
        and not setup_models.are_models_ready()
    ):
        setup_models.ensure_runtime_models()

    input_source = args.in_dir or args.input or DEFAULT_INPUT_DIR
    summary = run_on_input(
        input_source,
        out_dir=args.out_dir,
        device=args.device,
        model_id=args.model,
        num_beams=args.num_beams,
        initial_prompt=args.initial_prompt,
        clean=not args.no_clean,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
