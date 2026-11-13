"""End-to-end Interface Spec v2 audio pipeline wrapping the fine-tuned Whisper model.

Flow for each requested manifest version (``clean``, ``noisy``, ``phone``):

1. Validate manifest clips and check that every referenced WAV exists before loading ASR weights.
2. Transcribe clips once-per-run with Whisper (defaulting to the local
   ``models/whisper-large-v3-turbo-v3`` checkpoint when present, else
   ``openai/whisper-large-v3-turbo``).
3. Emit Interface Spec v2 audio ``Message`` items (optionally applying deterministic transcript
   cleaning via ``clean_transcripts.clean_message``, preserving raw Whisper output in ``asr_text``)
   and validate them against ``contracts.validate_file("transcripts", ...)``.
4. Pass deep-copied ``Message`` items through either a v2 ``System`` (``process_messages``) or an
   explicit adapter around the legacy v1 keyword baseline (producing strict v2 ``Record`` objects
   via ``contracts.build_record`` with Kenyan ``PHONE`` redaction spans).
5. Validate all ``Record`` objects with ``contracts.validate_file("records", records, messages)``
   and matching input-ID order before writing any output files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asr import clean_transcripts, transcribe  # noqa: E402
from src.core import contracts  # noqa: E402

JsonDict = dict[str, Any]
ProcessMessagesFn = Callable[[list[JsonDict]], list[JsonDict]]
MakeASRFn = Callable[[str, str, int, str | None], transcribe.ASRBackend]
ReadWavFn = Callable[[Path], tuple[Any, int]]

_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?254|0)[\s-]?[17](?:[\s-]?\d){8}(?!\w)")


@dataclass(frozen=True)
class PipelineResult:
    """Validated outputs produced for a single manifest audio version."""

    version: str
    messages: list[JsonDict]
    records: list[JsonDict]
    transcripts_path: Path
    meta_path: Path
    records_path: Path


def _import_baseline_runner() -> Any:
    """Import evaluation/baseline_runner.py regardless of current sys.path layout."""
    from evaluation import baseline_runner

    return baseline_runner


def _display_model(model_id: str, repo_root: Path = transcribe.REPO_ROOT) -> str:
    """Represent repo-local model directories with portable POSIX relative paths."""
    candidate = Path(model_id)
    if candidate.exists():
        try:
            return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return model_id
    return model_id


def _write_json_lf(path: Path, data: Any) -> None:
    """Write strict finite UTF-8 JSON with LF line endings and a trailing newline."""
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def detect_phone_spans(text: str) -> list[JsonDict]:
    """Return Interface Spec v2 ``PHONE`` redaction spans matching the baseline phone regex."""
    return [
        {"type": "PHONE", "start": match.start(), "end": match.end()}
        for match in _PHONE_PATTERN.finditer(text)
    ]


def adapt_v1_baseline_records(
    messages: Sequence[JsonDict], v1_records: Sequence[JsonDict]
) -> list[JsonDict]:
    """Wrap 6-field v1 baseline dicts as strict Interface Spec v2 ``Record`` dicts.

    Fills ``sender_id`` and ``source`` from the input ``Message`` and builds ``redactions`` and
    ``text_redacted`` from Kenyan ``PHONE`` regex matches (without ``NAME`` NER redaction).
    Fields absent from v1 baseline outputs (``date_text``, ``event_name``, ``event_desc``,
    ``confidence``) are set to ``None``.
    """
    if len(v1_records) != len(messages):
        raise ValueError(
            f"baseline returned {len(v1_records)} records for {len(messages)} messages"
        )
    msg_ids = [str(m["id"]) for m in messages]
    rec_ids = [r.get("id") for r in v1_records]
    if rec_ids != msg_ids:
        raise ValueError("baseline record IDs do not match input message IDs in order")

    adapted: list[JsonDict] = []
    for msg, rec in zip(messages, v1_records, strict=True):
        spans = detect_phone_spans(str(msg["message"]))
        built: JsonDict = contracts.build_record(
            msg,
            relevant=bool(rec["relevant"]),
            activity_type=rec.get("activity_type"),
            location=rec.get("location"),
            reported_count=rec.get("reported_count"),
            date_text=None,
            event_name=None,
            event_desc=None,
            redactions=spans,
            confidence=None,
        )
        adapted.append(built)
    return adapted


def _utc_now() -> datetime:
    return datetime.now(UTC)


def baseline_v2_system(messages: list[JsonDict]) -> list[JsonDict]:
    """Run the repo's legacy keyword baseline via ``baseline_runner`` and emit v2 Records."""
    runner = _import_baseline_runner()
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "messages.json"
        _write_json_lf(input_path, messages)
        try:
            v1_records: list[JsonDict] = runner.run_baseline(input_path)
        except runner.BaselineError as exc:
            raise ValueError(f"baseline failed: {exc}") from exc
    return adapt_v1_baseline_records(messages, v1_records)


def load_system(spec: str) -> ProcessMessagesFn:
    """Resolve ``'baseline'`` or ``'module[:attr]'`` into a v2 ``process_messages`` callable."""
    if spec == "baseline":
        return baseline_v2_system
    module_name, sep, attr_name = spec.partition(":")
    attr = attr_name if sep else "process_messages"
    if not module_name or not attr:
        raise ValueError(f"invalid system specification: {spec!r}")
    module = import_module(module_name)
    target = getattr(module, attr, None)
    if not callable(target):
        raise ValueError(f"system target {module_name}:{attr} is not callable")
    resolved: ProcessMessagesFn = target
    return resolved


def transcribe_to_messages(
    manifest: dict[str, Any],
    version: str,
    asr: transcribe.ASRBackend,
    *,
    clean: bool = False,
    read_wav: Callable[[Path], tuple[NDArray[Any], int]] = transcribe.read_wav,
    root: Path = transcribe.REPO_ROOT,
) -> list[JsonDict]:
    """Transcribe one manifest version and return validated v2 audio ``Message`` dicts."""
    clips = transcribe.clips_for_version(manifest, version)
    missing = transcribe.missing_wavs(clips, root=root)
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} audio files, starting with {missing[0]}")

    raw_records = transcribe.transcribe_clips(clips, read_wav, asr, root=root)
    raw_messages: list[JsonDict] = [
        dict(item) for item in sorted(raw_records, key=lambda rec: str(rec["id"]))
    ]
    if clean:
        messages = [clean_transcripts.clean_message(item) for item in raw_messages]
    else:
        messages = raw_messages

    problems = [str(p) for p in contracts.validate_file("transcripts", messages)]
    if problems:
        raise ValueError("Invalid audio transcript messages:\n" + "\n".join(problems))
    return messages


def run_system_contract(
    messages: Sequence[JsonDict], system_fn: ProcessMessagesFn
) -> list[JsonDict]:
    """Execute a Stage C system on validated ``messages`` and validate returned v2 ``Record``s."""
    message_list = list(messages)
    input_problems = [str(p) for p in contracts.validate_file("messages", message_list)]
    if input_problems:
        raise ValueError("Invalid input messages:\n" + "\n".join(input_problems))

    records = system_fn(deepcopy(message_list))
    if not isinstance(records, list):
        raise ValueError("System output must be a list of Record dicts")

    record_problems = [str(p) for p in contracts.validate_file("records", records, message_list)]
    if record_problems:
        raise ValueError("Invalid system output records:\n" + "\n".join(record_problems))

    expected_ids = [m["id"] for m in message_list]
    actual_ids = [r["id"] for r in records]
    if actual_ids != expected_ids:
        raise ValueError("System output IDs must match input message IDs in the same order")
    return records


def run_pipeline(
    manifest: dict[str, Any],
    versions: Sequence[str],
    transcripts_dir: Path,
    make_asr: MakeASRFn,
    *,
    model: str | None = None,
    device: str = "auto",
    batch_size: int = transcribe.DEFAULT_BATCH_SIZE,
    initial_prompt: str | None = None,
    clean: bool = False,
    system_fn: ProcessMessagesFn = baseline_v2_system,
    records_dir: Path | None = None,
    read_wav: Callable[[Path], tuple[NDArray[Any], int]] = transcribe.read_wav,
    root: Path = transcribe.REPO_ROOT,
    now: Callable[[], datetime] | None = None,
) -> list[PipelineResult]:
    """Run Stage B (ASR) -> optional cleaning -> Stage C (system) under strict v2 contracts.

    Pre-flight checks run across all requested versions before loading the ASR model, and all
    transcript and record outputs are contract-validated in memory before any file is written.
    """
    if not versions:
        raise ValueError("at least one version must be specified")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    clips_by_version: dict[str, list[JsonDict]] = {}
    for version in versions:
        clips = transcribe.clips_for_version(manifest, version)
        missing = transcribe.missing_wavs(clips, root=root)
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} audio files for {version}, starting with {missing[0]}"
            )
        clips_by_version[version] = clips

    resolved_model = transcribe.resolve_default_model() if model is None else model
    resolved_device = transcribe.resolve_device(device)
    display_model = _display_model(resolved_model, repo_root=root)
    target_records_dir = transcripts_dir if records_dir is None else records_dir
    clock: Callable[[], datetime] = _utc_now if now is None else now

    asr = make_asr(resolved_model, resolved_device, batch_size, initial_prompt)

    staged: list[tuple[PipelineResult, JsonDict]] = []
    for version in versions:
        t0 = perf_counter()
        messages = transcribe_to_messages(
            manifest, version, asr, clean=clean, read_wav=read_wav, root=root
        )
        wall_seconds = perf_counter() - t0
        records = run_system_contract(messages, system_fn)
        asr_meta = dict(asr.metadata())
        asr_meta["model"] = display_model
        audio_seconds = sum(float(c.get("duration_s", 0.0)) for c in clips_by_version[version])
        meta = transcribe.build_metadata(
            version,
            asr_meta,
            len(clips_by_version[version]),
            audio_seconds,
            wall_seconds,
            clock(),
        )
        meta["cleaned"] = clean
        result = PipelineResult(
            version=version,
            messages=messages,
            records=records,
            transcripts_path=transcripts_dir / f"transcripts_{version}.json",
            meta_path=transcripts_dir / f"transcripts_{version}.meta.json",
            records_path=target_records_dir / f"records_{version}.json",
        )
        staged.append((result, meta))

    results: list[PipelineResult] = []
    for result, meta in staged:
        _write_json_lf(result.transcripts_path, result.messages)
        _write_json_lf(result.meta_path, meta)
        _write_json_lf(result.records_path, result.records)
        results.append(result)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the end-to-end contract-wrapped audio pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Whisper audio -> v2 Message -> v2 Record pipeline."
    )
    parser.add_argument("--manifest", type=Path, default=transcribe.DEFAULT_MANIFEST)
    parser.add_argument("--version", choices=[*transcribe.VERSIONS, "all"], default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", type=Path, default=transcribe.DEFAULT_OUT_DIR)
    parser.add_argument("--records-out-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda:0", ...')
    parser.add_argument("--batch-size", type=int, default=transcribe.DEFAULT_BATCH_SIZE)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Normalize transcripts with clean_transcripts while preserving raw ASR in asr_text.",
    )
    parser.add_argument(
        "--system",
        default="baseline",
        help='Stage C system: "baseline" or "module[:process_messages]".',
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    make_asr: MakeASRFn = transcribe.make_whisper,
    system_fn: ProcessMessagesFn | None = None,
    now: Callable[[], datetime] | None = None,
    root: Path = transcribe.REPO_ROOT,
) -> int:
    """CLI entry point for running the contract-wrapped audio pipeline."""
    args = parse_args(argv)
    versions = list(transcribe.VERSIONS) if args.version == "all" else [str(args.version)]
    try:
        resolved_system = load_system(args.system) if system_fn is None else system_fn
        manifest = transcribe.load_manifest(args.manifest)
        results = run_pipeline(
            manifest,
            versions,
            args.out_dir,
            make_asr,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
            initial_prompt=args.initial_prompt,
            clean=bool(args.clean),
            system_fn=resolved_system,
            records_dir=args.records_out_dir,
            root=root,
            now=now,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for item in results:
        relevant_count = sum(1 for r in item.records if r["relevant"])
        print(
            f"{item.version}: {len(item.messages)} messages -> {item.transcripts_path} | "
            f"{len(item.records)} records ({relevant_count} relevant) -> {item.records_path}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
