"""Offline unit tests for src/training/export_ct2.py and evaluation/eval_whisper_benchmarks.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation import eval_whisper_benchmarks
from evaluation.eval_whisper_benchmarks import (
    BenchmarkItem,
    compute_corpus_wer,
    fix_missing_apostrophes,
    regex_normalize,
    score_predictions,
)
from evaluation.eval_whisper_benchmarks import (
    parse_args as parse_bench_args,
)
from src.training.export_ct2 import dir_size_mb, export_ct2_model
from src.training.export_ct2 import parse_args as parse_ct2_args


def test_export_ct2_helpers_and_validation(tmp_path: Path) -> None:
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"a" * (1024 * 1024))
    assert dir_size_mb(model_dir) == pytest.approx(1.0, abs=0.01)

    with pytest.raises(FileNotFoundError, match=r"missing config\.json"):
        export_ct2_model(model_dir, tmp_path / "out", verify=False)

    args = parse_ct2_args(["--model", str(model_dir), "--out-dir", str(tmp_path / "ct2")])
    assert args.quantization == "int8"
    assert args.no_verify is False


def test_eval_whisper_benchmarks_pure_metrics() -> None:
    assert regex_normalize("Sister Mary's 42 women, Nakuru!") == "sister marys 42 women nakuru"
    assert fix_missing_apostrophes("They didnt say we shouldnt go") == (
        "They didn't say we shouldn't go"
    )
    assert compute_corpus_wer(
        ["We visited Narok.", ""],
        ["We visited Narok", "ignored"],
        regex_normalize,
    ) == 0.0

    items = [
        BenchmarkItem(
            audio=np.zeros(160, dtype=np.float32),
            text="They didnt visit Narok",
            domain="clinical",
            accent="swahili",
        ),
        BenchmarkItem(
            audio=np.zeros(160, dtype=np.float32),
            text="Screened 10 women",
            domain="general",
            accent="kikuyu",
        ),
    ]
    scored = score_predictions(
        items,
        ["They didn't visit Narok", "Screened 12 women"],
        whisper_norm=lambda t: t.lower().replace("'", ""),
    )
    assert scored["n"] == 2
    assert scored["domain"]["clinical"]["n"] == 1
    assert scored["domain"]["general"]["n"] == 1
    assert scored["accent"]["swahili"]["fair_wer"] < scored["accent"]["kikuyu"]["fair_wer"]

    args = parse_bench_args(["--model", "m", "--out", "o.json"])
    assert args.modes == ["greedy", "beam3"]
    assert args.suites == ["test_300_seed1234", "test_330_balanced", "kenyan_ea_test", "fleurs_all"]
    assert args.initial_prompt is None

    prompt_args = parse_bench_args(
        [
            "--model",
            "m",
            "--out",
            "o.json",
            "--modes",
            "greedy",
            "beam3",
            "beam5",
            "--initial-prompt",
            "Habari",
        ]
    )
    assert prompt_args.modes == ["greedy", "beam3", "beam5"]
    assert prompt_args.initial_prompt == "Habari"


def test_public_suite_loads_prepared_audio_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eval_whisper_benchmarks, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="offline benchmark dataset missing"):
        eval_whisper_benchmarks.load_test_300_seed1234()

    data_dir = tmp_path / "evaluation" / "datasets"
    data_dir.mkdir(parents=True)
    np.savez_compressed(
        data_dir / "test_300_seed1234.npz",
        audio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        offsets=np.array([0, 2, 3], dtype=np.int64),
        text=np.array(["one", "two"]),
        domain=np.array(["clinical", "general"]),
        accent=np.array(["swahili", "igbo"]),
    )
    items = eval_whisper_benchmarks.load_test_300_seed1234()
    assert [item.text for item in items] == ["one", "two"]
    assert items[0].accent == "swahili"
    assert items[0].audio.tolist() == pytest.approx([0.1, 0.2])
    assert items[1].audio.tolist() == pytest.approx([0.3])
