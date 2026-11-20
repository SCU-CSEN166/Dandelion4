"""Tests for the combined evaluation entry point."""

import json
from pathlib import Path
from typing import Any

import pytest

from demo import run_demo
from evaluation import eval_wer, eval_whisper_benchmarks, run_evaluation
from src.asr import audio_pipeline, transcribe


@pytest.mark.parametrize(
    ("cli", "detected", "expected"),
    [([], "cuda:0", "cuda:0"), ([], "cpu", "cpu"), (["--cpu"], "cuda:0", "cpu")],
)
def test_device_selection_and_public_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli: list[str],
    detected: str,
    expected: str,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(transcribe, "resolve_device", lambda _: detected)

    def fake_benchmarks(model: str, **kwargs: Any) -> dict[str, str]:
        calls.append({"model": model, **kwargs})
        return {"model": model}

    monkeypatch.setattr(
        eval_whisper_benchmarks,
        "run_benchmarks",
        fake_benchmarks,
    )
    out_dir = tmp_path / "output"
    args = run_evaluation.parse_args(
        ["--model", "test-model", "--suites", "fleurs_all", "--modes", "beam3",
         "--out-dir", str(out_dir), *cli]
    )
    result = run_evaluation.run(args)
    assert calls[0]["device"] == expected
    assert calls[0]["suites"] == ["fleurs_all"]
    assert calls[0]["modes"] == ["beam3"]
    assert json.loads(result.read_text(encoding="utf-8"))["device"] == expected
    assert (out_dir / "public_speech.json").is_file()


def test_synthetic_and_hidden_are_isolated_by_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int, str | None]] = []
    monkeypatch.setattr(transcribe, "load_manifest", lambda _: {})

    def fake_pipeline(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append(("synthetic", kwargs["num_beams"], kwargs["initial_prompt"]))
        return []

    def fake_score(argv: list[str]) -> int:
        path = Path(argv[argv.index("--out") + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": {"clean": {"wer_pct": 1}}}), encoding="utf-8")
        return 0

    def fake_demo(**kwargs: Any) -> int:
        calls.append(("hidden", kwargs["num_beams"], kwargs["initial_prompt"]))
        assert kwargs["batch_size"] == 1
        path = kwargs["artifacts_dir"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "demo_metrics.json").write_text('{"asr_raw_wer_pct": 2}', encoding="utf-8")
        (path / "predicted_transcripts.meta.json").write_text('{"device": "cpu"}', encoding="utf-8")
        return 0

    monkeypatch.setattr(audio_pipeline, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(eval_wer, "main", fake_score)
    monkeypatch.setattr(run_demo, "run_demo", fake_demo)
    monkeypatch.setattr(run_evaluation, "_release_cuda", lambda _: None)
    output = tmp_path / "output"
    args = run_evaluation.parse_args(
        ["--cpu", "--model", "model", "--suites", "synthetic", "hidden", "--kenyan-prompt",
         "--out-dir", str(output)]
    )
    report = json.loads(run_evaluation.run(args).read_text(encoding="utf-8"))
    assert [(name, beam) for name, beam, _ in calls] == [
        ("synthetic", 1), ("hidden", 1), ("synthetic", 3), ("hidden", 3)
    ]
    assert all(prompt == transcribe.KENYAN_CHW_INITIAL_PROMPT for _, _, prompt in calls)
    assert report["hidden"]["beam3"]["metrics"]["asr_raw_wer_pct"] == 2
    assert report["synthetic"]["greedy"]["results_file"] == "synthetic/greedy/wer.json"


def test_existing_output_is_protected(tmp_path: Path) -> None:
    (tmp_path / "existing.json").write_text("keep", encoding="utf-8")
    args = run_evaluation.parse_args(["--cpu", "--out-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="new or empty"):
        run_evaluation.run(args)
    assert (tmp_path / "existing.json").read_text(encoding="utf-8") == "keep"
