"""Run synthetic, hidden and public-speech evaluations through one CLI."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import run_demo  # noqa: E402
from evaluation import eval_wer, eval_whisper_benchmarks  # noqa: E402
from src.asr import audio_pipeline, transcribe  # noqa: E402
from src.pipeline import hidden_test_pipeline  # noqa: E402

PUBLIC_SUITES = ("test_300_seed1234", "test_330_balanced", "kenyan_ea_test", "fleurs_all")
SUITES = ("synthetic", "hidden", *PUBLIC_SUITES)
BEAMS = {"greedy": 1, "beam3": 3, "beam5": 5}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper model on Dandelion audio and public speech benchmarks."
    )
    parser.add_argument("--model", default=None, help="Hugging Face checkpoint or CT2 directory.")
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU; otherwise use CUDA when available."
    )
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--modes", nargs="+", choices=tuple(BEAMS), default=["greedy", "beam3"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None, help="New or empty output directory.")
    parser.add_argument("--manifest", type=Path, default=transcribe.DEFAULT_MANIFEST)
    parser.add_argument(
        "--messages", type=Path, default=eval_wer.REPO_ROOT / "examples/synthetic_messages.json"
    )
    parser.add_argument(
        "--expected", type=Path, default=eval_wer.REPO_ROOT / "examples/expected_outputs.json"
    )
    parser.add_argument("--hidden-input", type=Path, default=run_demo.DEFAULT_INPUT)
    parser.add_argument("--hidden-expected", type=Path, default=run_demo.DEFAULT_EXPECTED_CSV)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--kenyan-prompt", action="store_true")
    prompts.add_argument("--initial-prompt", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if len(set(args.suites)) != len(args.suites):
        parser.error("--suites must not contain duplicates")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _release_cuda(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> Path:
    device = "cpu" if args.cpu else transcribe.resolve_device("auto")
    model = args.model or transcribe.resolve_default_model(device=device)
    prompt = hidden_test_pipeline.resolve_initial_prompt(
        args.initial_prompt, kenyan_prompt=args.kenyan_prompt
    )
    if args.out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / "evaluation" / "runs" / f"{stamp}-{uuid4().hex[:8]}"
    else:
        out_dir = args.out_dir
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"output directory must be new or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": model,
        "device": device,
        "initial_prompt": prompt,
        "batch_size": args.batch_size,
        "suites": list(args.suites),
        "modes": list(args.modes),
        "synthetic": {},
        "hidden": {},
    }
    print(f"Evaluation device: {device}; output: {out_dir}", flush=True)
    for mode in args.modes:
        beams = BEAMS[mode]
        if "synthetic" in args.suites:
            directory = out_dir / "synthetic" / mode
            manifest = transcribe.load_manifest(args.manifest)
            audio_pipeline.run_pipeline(
                manifest,
                transcribe.VERSIONS,
                directory,
                transcribe.make_whisper,
                model=model,
                device=device,
                batch_size=args.batch_size,
                initial_prompt=prompt,
                num_beams=beams,
                clean=True,
            )
            score_path = directory / "wer.json"
            eval_wer.main([
                "--manifest", str(args.manifest),
                "--messages", str(args.messages),
                "--expected", str(args.expected),
                "--transcripts-dir", str(directory),
                "--out", str(score_path),
            ])
            report["synthetic"][mode] = {
                "summary": _read_json(score_path)["summary"],
                "results_file": score_path.relative_to(out_dir).as_posix(),
            }
            _release_cuda(device)
        if "hidden" in args.suites:
            directory = out_dir / "hidden" / mode
            code = run_demo.run_demo(
                input_audio=args.hidden_input,
                out_csv=directory / "predicted.csv",
                artifacts_dir=directory,
                expected_csv=args.hidden_expected,
                device=device,
                model_id=model,
                batch_size=args.batch_size,
                num_beams=beams,
                initial_prompt=prompt,
            )
            if code != 0:
                raise RuntimeError(f"hidden evaluation failed for {mode}: exit {code}")
            report["hidden"][mode] = {
                "metrics": _read_json(directory / "demo_metrics.json"),
                "asr_metadata": _read_json(directory / "predicted_transcripts.meta.json"),
                "results_dir": directory.relative_to(out_dir).as_posix(),
            }
            _release_cuda(device)
    public = [name for name in args.suites if name in PUBLIC_SUITES]
    if public:
        results = eval_whisper_benchmarks.run_benchmarks(
            model,
            suites=public,
            modes=args.modes,
            batch_size=args.batch_size,
            device=device,
            initial_prompt=prompt,
        )
        public_path = out_dir / "public_speech.json"
        public_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        report["public_speech"] = results
    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation summary: {report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
