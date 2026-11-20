"""Tests for evaluation/fetch_public_datasets.py and auto-download hooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation import eval_whisper_benchmarks as bench
from evaluation import fetch_public_datasets


def test_save_and_load_local_suite_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets_dir = tmp_path / "evaluation" / "datasets"
    items = [
        bench.BenchmarkItem(
            audio=np.ones(1600, dtype=np.float32),
            text="first clip",
            domain="clinical",
            accent="swahili",
        ),
        bench.BenchmarkItem(
            audio=np.zeros(800, dtype=np.float32),
            text="second clip",
            domain="general",
            accent="luo",
        ),
    ]
    out_path = fetch_public_datasets.save_local_suite(
        "test_300_seed1234", items, datasets_dir=datasets_dir
    )
    assert out_path.is_file()

    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path)
    loaded = bench._load_local_suite("test_300_seed1234")
    assert len(loaded) == 2
    assert loaded[0].text == "first clip"
    assert loaded[1].accent == "luo"


def test_prepare_suite_skips_when_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets_dir = tmp_path / "evaluation" / "datasets"
    fetch_public_datasets.save_local_suite(
        "test_300_seed1234",
        [bench.BenchmarkItem(audio=np.zeros(800, dtype=np.float32), text="ok")],
        datasets_dir=datasets_dir,
    )
    called: list[str] = []

    def fake_prepare(d: Path) -> Path:
        called.append(str(d))
        return d

    monkeypatch.setattr(fetch_public_datasets, "_prepare_test_300_seed1234", fake_prepare)
    path = fetch_public_datasets.prepare_suite("test_300_seed1234", datasets_dir=datasets_dir)
    assert path.is_file()
    assert called == []


def test_load_local_suite_respects_offline_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline benchmark dataset missing"):
        bench.load_test_300_seed1234()
