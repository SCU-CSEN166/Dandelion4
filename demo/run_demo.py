"""Live demo script: run the full Dandelion audio -> NLP -> CSV pipeline and print outputs.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe demo\\run_demo.py
    python demo/run_demo.py --show-raw
    python demo/run_demo.py --transcripts demo/outputs/predicted_transcripts.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe  # noqa: E402
from src.core import contracts, setup_models  # noqa: E402
from src.nlp import gliner_system  # noqa: E402
from src.pipeline import hidden_test_pipeline  # noqa: E402

DEMO_DIR = REPO_ROOT / "demo"
DEFAULT_INPUT = DEMO_DIR / "inputs" / "dandelion_audio.zip"
DEFAULT_EXPECTED_CSV = DEMO_DIR / "expected" / "hidden_test_outputs.csv"
DEFAULT_OUT_CSV = DEMO_DIR / "outputs" / "predicted_outputs.csv"
DEFAULT_ARTIFACTS_DIR = DEMO_DIR / "outputs"


def _ensure_utf8_stdout() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _print_banner(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def _wrap_labeled(label: str, text: str, width: int = 80) -> str:
    prefix = f"  {label:<13}: "
    indent = " " * len(prefix)
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=indent)


def format_demo_summary(
    rows: Sequence[dict[str, str]],
    messages: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None,
    *,
    elapsed_seconds: float,
    out_csv: Path,
    show_raw: bool = False,
) -> str:
    """Format the pipeline outputs into a readable terminal report for live demos."""
    lines: list[str] = []
    event_count = sum(1 for r in rows if r["relevance"] == "Event")
    chitchat_count = sum(1 for r in rows if r["relevance"] == "Chit-chat")
    dup_count = sum(1 for r in rows if r["duplicate_of"] != "none")
    n_name = sum(r["anonymized_text"].count("[NAME]") for r in rows)
    n_loc = sum(r["anonymized_text"].count("[LOCATION]") for r in rows)
    n_phone = sum(r["anonymized_text"].count("[PHONE]") for r in rows)

    lines.append("=" * 80)
    lines.append("DANDELION AUDIO-TO-INSIGHTS DEMO REPORT")
    lines.append("=" * 80)
    lines.append(f"  Output CSV     : {out_csv}")
    lines.append(
        f"  Total messages : {len(rows)} ({event_count} Event, {chitchat_count} Chit-chat)"
    )
    lines.append(f"  Duplicates     : {dup_count} duplicate report(s) linked")
    lines.append(f"  Anonymized PII : {n_name} [NAME], {n_loc} [LOCATION], {n_phone} [PHONE]")
    if meta is not None:
        lines.append(
            f"  Audio runtime  : {meta.get('audio_seconds_total', '?')} s audio in "
            f"{meta.get('wall_seconds', '?')} s ASR "
            f"({meta.get('device', '?')}, {meta.get('model', '?')})"
        )
    lines.append(f"  End-to-end     : {elapsed_seconds:.2f} s total")

    lines.append("\n" + "-" * 80)
    lines.append("SUMMARY TABLE")
    lines.append("-" * 80)
    lines.append(
        f"{'ID':>3} | {'Relevance':<9} | {'Activity Type':<40} | {'Count':>5} | {'Dup Of':<6}"
    )
    lines.append("-" * 80)
    for row in rows:
        act = row["activity_type"] or "-"
        if len(act) > 40:
            act = act[:37] + "..."
        cnt = row["participants"] or "-"
        lines.append(
            f"{row['Message_id']:>3} | {row['relevance']:<9} | {act:<40} | {cnt:>5} | "
            f"{row['duplicate_of']:<6}"
        )

    parse_num = hidden_test_pipeline.parse_numeric_message_id
    msg_by_id = {str(parse_num(str(m["id"]))): m for m in messages}
    rec_by_id = {str(parse_num(str(r["id"]))): r for r in records}

    lines.append("\n" + "-" * 80)
    lines.append("PER-MESSAGE DETAILS & ANONYMIZED TEXT")
    lines.append("-" * 80)
    for row in rows:
        mid = row["Message_id"]
        msg = msg_by_id.get(mid, {})
        rec = rec_by_id.get(mid, {})
        lines.append(
            f"\n[Message {mid}] Relevance: {row['relevance']} | "
            f"Activity: {row['activity_type'] or '-'} | "
            f"Participants: {row['participants'] or '-'} | "
            f"Duplicate of: {row['duplicate_of']}"
        )
        if show_raw and isinstance(msg.get("asr_text"), str):
            lines.append(_wrap_labeled("Raw ASR", msg["asr_text"]))
        if show_raw and isinstance(msg.get("message"), str):
            lines.append(_wrap_labeled("Cleaned", msg["message"]))
        lines.append(_wrap_labeled("Anonymized", row["anonymized_text"]))
        if rec.get("location") or rec.get("date_text"):
            lines.append(
                f"  {'Extracted':<13}: v2_activity={rec.get('activity_type')} | "
                f"location={rec.get('location')} | date={rec.get('date_text')} | "
                f"redactions={len(rec.get('redactions', []))}"
            )

    return "\n".join(lines)


def _format_metric_lines(metrics: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for key, val in metrics.items():
        if key.endswith("_pct"):
            stem, value_str = key[:-4], f"{val:.2f}%"
        elif key.endswith("_count"):
            stem, value_str = key[:-6], str(int(val))
        else:
            stem, value_str = key, str(val)
        label = (
            stem.replace("_", " ")
            .title()
            .replace("Ner", "NER")
            .replace("Asr", "ASR")
            .replace("Wer", "WER")
        )
        lines.append(f"  {label:<32}: {value_str}")
    return lines


def run_demo(
    *,
    input_audio: Path | None = DEFAULT_INPUT,
    transcripts_path: Path | None = None,
    out_csv: Path = DEFAULT_OUT_CSV,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    expected_csv: Path | None = None,
    clean: bool = True,
    device: str = "auto",
    model_id: str | None = None,
    batch_size: int = transcribe.DEFAULT_BATCH_SIZE,
    num_beams: int = 1,
    initial_prompt: str | None = None,
    show_raw: bool = False,
    pipeline_fn: Callable[..., list[dict[str, str]]] = (
        hidden_test_pipeline.run_hidden_test_pipeline
    ),
) -> int:
    """Execute the full hidden-test pipeline, verify contracts, and print demo results."""
    _ensure_utf8_stdout()
    _print_banner("RUNNING DANDELION END-TO-END PIPELINE DEMO")

    if transcripts_path is not None:
        print(f"Input mode     : Transcripts JSON ({transcripts_path})")
    else:
        print(f"Input source   : {input_audio}")
    print(f"Device request : {device}")
    print(f"ASR model      : {model_id or transcribe.resolve_default_model(device=device)}")
    gliner_src, local_only = gliner_system.resolve_gliner_model()
    print(f"GLiNER model   : {gliner_src} (local_only={local_only})")
    print(f"CSV destination: {out_csv}")
    print("Executing pipeline stages (ASR -> cleaning -> GLiNER -> dedup -> CSV)...")

    extra_kw: dict[str, Any] = {}
    if model_id is not None:
        extra_kw["model_id"] = model_id
    if batch_size != transcribe.DEFAULT_BATCH_SIZE:
        extra_kw["batch_size"] = batch_size
    if num_beams != 1:
        extra_kw["num_beams"] = num_beams
    if initial_prompt is not None:
        extra_kw["initial_prompt"] = initial_prompt

    t0 = time.perf_counter()
    rows = pipeline_fn(
        input_audio=None if transcripts_path is not None else input_audio,
        transcripts_path=transcripts_path,
        out_csv=out_csv,
        artifacts_dir=artifacts_dir,
        clean=clean,
        device=device,
        **extra_kw,
    )
    elapsed = time.perf_counter() - t0

    messages_file = artifacts_dir / "predicted_transcripts.json"
    records_file = artifacts_dir / "predicted_records.json"
    meta_file = artifacts_dir / "predicted_transcripts.meta.json"

    messages: list[dict[str, Any]] = json.loads(messages_file.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = json.loads(records_file.read_text(encoding="utf-8"))
    meta: dict[str, Any] | None = (
        json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else None
    )

    t_errs = contracts.validate_file("transcripts", messages)
    r_errs = contracts.validate_file("records", records, messages)
    if t_errs or r_errs:
        raise ValueError("Interface Spec v2 contract check failed:\n" + "\n".join(t_errs + r_errs))
    print(
        f"Contract checks: PASSED ({len(messages)} Message items, "
        f"{len(records)} Record items valid)"
    )

    print(
        format_demo_summary(
            rows,
            messages,
            records,
            meta,
            elapsed_seconds=elapsed,
            out_csv=out_csv,
            show_raw=show_raw,
        )
    )
    if expected_csv is not None:
        expected_rows = hidden_test_pipeline.load_csv_rows(expected_csv)
        metrics = hidden_test_pipeline.evaluate_csv_outputs(rows, expected_rows)
        metrics.update(hidden_test_pipeline.evaluate_transcript_wer(messages, expected_rows))
        metrics_file = artifacts_dir / "demo_metrics.json"
        metrics_file.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        _print_banner(f"VALIDATION METRICS vs. {expected_csv.name}")
        for line in _format_metric_lines(metrics):
            print(line)
        print(f"  Saved metrics  : {metrics_file}")

    _print_banner(f"RAW CSV WRITTEN TO {out_csv}")
    print(out_csv.read_text(encoding="utf-8").rstrip())
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Dandelion demo pipeline locally and print all outputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--transcripts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--expected", type=Path, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--kenyan-prompt",
        action="store_true",
        help="Use the built-in generic Kenyan CHW / Swahili ASR initial prompt.",
    )
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda:0", ...')
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also print raw ASR and cleaned text before anonymized text.",
    )
    return parser.parse_args(argv)


def _resolve_expected_csv(args: argparse.Namespace) -> Path | None:
    if args.no_eval:
        return None
    if args.expected is not None:
        return Path(args.expected)
    default_transcripts = DEFAULT_ARTIFACTS_DIR / "predicted_transcripts.json"
    is_default_hidden_run = (
        args.transcripts is None and args.input == DEFAULT_INPUT
    ) or args.transcripts == default_transcripts
    if is_default_hidden_run and DEFAULT_EXPECTED_CSV.is_file():
        return DEFAULT_EXPECTED_CSV
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.model is None
        and os.environ.get("HF_HUB_OFFLINE") != "1"
        and not setup_models.are_models_ready()
    ):
        setup_models.ensure_runtime_models()
    prompt = hidden_test_pipeline.resolve_initial_prompt(
        args.initial_prompt, kenyan_prompt=args.kenyan_prompt
    )
    return run_demo(
        input_audio=None if args.transcripts is not None else args.input,
        transcripts_path=args.transcripts,
        out_csv=args.out,
        artifacts_dir=args.artifacts_dir,
        expected_csv=_resolve_expected_csv(args),
        clean=not args.no_clean,
        device=args.device,
        model_id=args.model,
        num_beams=args.num_beams,
        initial_prompt=prompt,
        show_raw=args.show_raw,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
