"""Tests for src/core/setup_models.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core import setup_models


def _populate_models(models_dir: Path, *, include_hf: bool = False) -> None:
    whisper = models_dir / "whisper-large-v3-tuned-int8"
    gliner = models_dir / "gliner_medium-v2.1"
    deberta = models_dir / "deberta-v3-base"
    whisper.mkdir(parents=True)
    gliner.mkdir(parents=True)
    deberta.mkdir(parents=True)
    (whisper / "model.bin").write_bytes(b"int8-weights")
    (gliner / "pytorch_model.bin").write_bytes(b"gliner-weights")
    (deberta / "config.json").write_text("{}", encoding="utf-8")
    if include_hf:
        hf = models_dir / "whisper-large-v3-tuned"
        hf.mkdir(parents=True)
        (hf / "config.json").write_text("{}", encoding="utf-8")
        (hf / "model.safetensors").write_bytes(b"hf-weights" * 50)


def test_are_models_ready(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    assert not setup_models.are_models_ready(models_dir, include_gpu_hf=False)
    _populate_models(models_dir, include_hf=False)
    assert setup_models.are_models_ready(models_dir, include_gpu_hf=False)
    assert not setup_models.are_models_ready(models_dir, include_gpu_hf=True)
    _populate_models(tmp_path / "models_hf", include_hf=True)
    assert setup_models.are_models_ready(tmp_path / "models_hf", include_gpu_hf=True)


def test_ensure_runtime_models_skips_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "models"
    _populate_models(models_dir, include_hf=False)

    called: list[str] = []
    monkeypatch.setattr(
        setup_models,
        "_download_release_asset",
        lambda repo, tag, asset, dest: called.append(asset),
    )

    assert setup_models.ensure_runtime_models(models_dir, include_gpu_hf=False) == models_dir
    assert called == []


def test_package_and_restore_runtime_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_models = tmp_path / "src_models"
    _populate_models(src_models, include_hf=True)
    assets_dir = tmp_path / "assets"
    archives = setup_models.package_models(src_models, assets_dir)
    assert {p.name for p in archives} == {
        setup_models.WHISPER_ASSET_NAME,
        setup_models.NLP_ASSET_NAME,
        *setup_models.WHISPER_HF_ASSET_PARTS,
    }

    dst_models = tmp_path / "dst_models"

    def fake_download(repo: str, tag: str, asset: str, dest: Path) -> None:
        dest.write_bytes((assets_dir / asset).read_bytes())

    monkeypatch.setattr(setup_models, "_download_release_asset", fake_download)
    setup_models.ensure_runtime_models(dst_models, include_gpu_hf=True)
    assert setup_models.are_models_ready(dst_models, include_gpu_hf=True)
