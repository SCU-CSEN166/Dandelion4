"""Tests for src/core/setup_models.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core import setup_models


def _populate_models(models_dir: Path) -> None:
    whisper = models_dir / "whisper-large-v3-tuned-int8"
    gliner = models_dir / "gliner_medium-v2.1"
    deberta = models_dir / "deberta-v3-base"
    whisper.mkdir(parents=True)
    gliner.mkdir(parents=True)
    deberta.mkdir(parents=True)
    (whisper / "model.bin").write_bytes(b"int8-weights")
    (gliner / "pytorch_model.bin").write_bytes(b"gliner-weights")
    (deberta / "config.json").write_text("{}", encoding="utf-8")


def test_are_models_ready(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    assert not setup_models.are_models_ready(models_dir)
    _populate_models(models_dir)
    assert setup_models.are_models_ready(models_dir)


def test_ensure_runtime_models_skips_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "models"
    _populate_models(models_dir)

    called: list[str] = []
    monkeypatch.setattr(
        setup_models,
        "_download_release_asset",
        lambda repo, tag, asset, dest: called.append(asset),
    )

    assert setup_models.ensure_runtime_models(models_dir) == models_dir
    assert called == []


def test_package_and_restore_runtime_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_models = tmp_path / "src_models"
    _populate_models(src_models)
    assets_dir = tmp_path / "assets"
    archives = setup_models.package_models(src_models, assets_dir)
    assert {p.name for p in archives} == {
        setup_models.WHISPER_ASSET_NAME,
        setup_models.NLP_ASSET_NAME,
    }

    dst_models = tmp_path / "dst_models"

    def fake_download(repo: str, tag: str, asset: str, dest: Path) -> None:
        dest.write_bytes((assets_dir / asset).read_bytes())

    monkeypatch.setattr(setup_models, "_download_release_asset", fake_download)
    setup_models.ensure_runtime_models(dst_models)
    assert setup_models.are_models_ready(dst_models)
