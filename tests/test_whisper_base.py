"""Tests for src/training/whisper_base.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.training import whisper_base


def test_is_local_whisper_model_ready(tmp_path: Path) -> None:
    model_dir = tmp_path / "whisper-large-v3"
    assert not whisper_base.is_local_whisper_model_ready(model_dir)
    model_dir.mkdir()
    assert not whisper_base.is_local_whisper_model_ready(model_dir)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert not whisper_base.is_local_whisper_model_ready(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    assert whisper_base.is_local_whisper_model_ready(model_dir)


def test_ensure_base_whisper_large_v3_skips_download_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "whisper-large-v3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    called: list[Any] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: called.append(kwargs),
    )

    result = whisper_base.ensure_base_whisper_large_v3(model_dir)
    assert result == model_dir
    assert called == []


def test_ensure_base_whisper_large_v3_downloads_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "whisper-large-v3"
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = whisper_base.ensure_base_whisper_large_v3(model_dir)
    assert result == model_dir
    assert calls == [
        {
            "repo_id": whisper_base.BASE_WHISPER_LARGE_V3_MODEL_ID,
            "local_dir": str(model_dir),
        }
    ]
    assert whisper_base.is_local_whisper_model_ready(model_dir)


def test_resolve_training_base_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "whisper-large-v3"
    calls: list[Path] = []

    def fake_ensure(
        model_dir: Path = base_dir,
        *,
        model_id: str = whisper_base.BASE_WHISPER_LARGE_V3_MODEL_ID,
    ) -> Path:
        calls.append(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"weights")
        return model_dir

    monkeypatch.setattr(whisper_base, "ensure_base_whisper_large_v3", fake_ensure)

    assert (
        whisper_base.resolve_training_base_model(
            whisper_base.BASE_WHISPER_LARGE_V3_MODEL_ID,
            base_dir=base_dir,
        )
        == str(base_dir)
    )
    assert calls == [base_dir]

    # Second call sees the ready local directory directly without calling ensure again.
    assert (
        whisper_base.resolve_training_base_model(
            whisper_base.BASE_WHISPER_LARGE_V3_MODEL_ID,
            base_dir=base_dir,
        )
        == str(base_dir)
    )
    assert len(calls) == 1

    # Explicit non-v3-large model IDs pass through untouched.
    assert whisper_base.resolve_training_base_model("openai/whisper-small", base_dir=base_dir) == (
        "openai/whisper-small"
    )
