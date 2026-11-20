"""Offline unit tests for src/training/fetch_fleurs.py."""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile

from src.asr import transcribe
from src.core import contracts
from src.training import fetch_fleurs
from src.training.fetch_fleurs import FleursSample, main, to_mono_16k

FIXED_NOW = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)


def _tone(sr: int, seconds: float = 0.25, freq: float = 440.0) -> npt.NDArray[np.float32]:
    t = np.arange(int(sr * seconds)) / sr
    return (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _fake_stream(samples: list[FleursSample]) -> fetch_fleurs.StreamFleurs:
    def _stream(config: str, split: str, limit: int) -> list[FleursSample]:
        del config, split
        return samples[:limit]

    return _stream


def test_fetch_fleurs_writes_wavs_manifest_and_valid_messages(tmp_path: Path) -> None:
    stereo_24k = np.stack([_tone(24_000, 0.2), _tone(24_000, 0.2, 880.0)], axis=1)
    samples = [
        FleursSample(
            sample_id="101",
            speaker_id="spk_a",
            audio=_tone(16_000, 0.25),
            sample_rate=16_000,
            text="Community health workers visited ten households.",
        ),
        FleursSample(
            sample_id="102",
            speaker_id="spk_b",
            audio=stereo_24k,
            sample_rate=24_000,
            text="We screened thirty patients at the clinic.",
        ),
    ]
    out_dir = tmp_path / "examples" / "fleurs_audio"
    manifest_path = tmp_path / "examples" / "fleurs_manifest.json"
    messages_path = tmp_path / "examples" / "fleurs_messages.json"

    manifest, messages = fetch_fleurs.fetch_fleurs(
        config="en_us",
        split="test",
        limit=2,
        out_dir=out_dir,
        manifest_path=manifest_path,
        messages_path=messages_path,
        root=tmp_path,
        stream_fn=_fake_stream(samples),
        now=lambda: FIXED_NOW,
    )

    assert manifest["generator"]["script"] == "src/training/fetch_fleurs.py"
    assert manifest["generator"]["created_utc"] == "2026-11-13T09:00:00Z"
    assert manifest["generator"]["dataset"] == "google/fleurs"
    assert manifest["generator"]["config"] == "en_us"
    assert manifest["generator"]["split"] == "test"
    assert manifest["generator"]["limit"] == 2
    assert len(manifest["clips"]) == 2
    assert len(messages) == 2

    loaded_manifest = transcribe.load_manifest(manifest_path)
    clean_clips = transcribe.clips_for_version(loaded_manifest, "clean")
    assert len(clean_clips) == 2
    assert transcribe.missing_wavs(clean_clips, tmp_path) == []

    for clip in clean_clips:
        assert clip["sample_rate"] == 16_000
        assert clip["version"] == "clean"
        assert clip["n_samples"] > 0
        assert clip["reference_text"]
        wav_path = transcribe.clip_path(clip, tmp_path)
        data, sr = soundfile.read(wav_path, dtype="float32")
        assert sr == 16_000
        assert data.ndim == 1
        assert len(data) == clip["n_samples"]

    raw_messages = json.loads(messages_path.read_text(encoding="utf-8"))
    assert contracts.validate_file("messages", raw_messages) == []
    parsed_messages = contracts.load_messages(messages_path)
    assert len(parsed_messages) == 2
    assert [m["id"] for m in raw_messages] == ["fleurs_0001", "fleurs_0002"]


def test_fetch_fleurs_manifest_works_with_transcribe_versions(tmp_path: Path) -> None:
    samples = [
        FleursSample(
            sample_id="1",
            speaker_id="spk_1",
            audio=_tone(16_000, 0.2),
            sample_rate=16_000,
            text="First test utterance.",
        )
    ]
    manifest_path = tmp_path / "fleurs_manifest.json"
    fetch_fleurs.fetch_fleurs(
        out_dir=tmp_path / "audio",
        manifest_path=manifest_path,
        messages_path=tmp_path / "messages.json",
        root=tmp_path,
        stream_fn=_fake_stream(samples),
    )

    class _FakeASR:
        def load(self) -> None:
            return None

        def __call__(self, waves: list[npt.NDArray[np.float32]]) -> list[str]:
            return ["First test utterance." for _ in waves]

        def metadata(self) -> dict[str, Any]:
            base: dict[str, Any] = {
                "model": "fake-whisper",
                "language": "en",
                "task": "transcribe",
                "decoding": {"num_beams": 1, "do_sample": False},
                "initial_prompt": None,
                "device": "cpu",
                "dtype": "float32",
                "torch_version": "0.0",
                "transformers_version": "0.0",
            }
            return {k: base.get(k) for k in transcribe.ASR_METADATA_KEYS}

    manifest = transcribe.load_manifest(manifest_path)
    written = transcribe.transcribe_versions(
        manifest,
        ["clean"],
        tmp_path / "out",
        _FakeASR,
        root=tmp_path,
        now=lambda: FIXED_NOW,
    )
    assert len(written) == 2
    transcripts = json.loads(written[0].read_text(encoding="utf-8"))
    assert contracts.validate_file("transcripts", transcripts) == []
    assert transcripts[0]["id"] == "fleurs_0001"
    assert transcripts[0]["message"] == "First test utterance."


def test_stream_via_datasets_mocked_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "id": 7,
            "audio": {"array": _tone(16_000, 0.1), "sampling_rate": 16_000},
            "transcription": "Mocked HF row one.",
        },
        {
            "id": 8,
            "audio": {"array": _tone(16_000, 0.1), "sampling_rate": 16_000},
            "raw_transcription": "Mocked HF row two.",
        },
        {
            "id": 9,
            "audio": {"array": _tone(16_000, 0.1), "sampling_rate": 16_000},
            "transcription": "Should not be reached.",
        },
    ]

    fake_datasets = SimpleNamespace(load_dataset=lambda *args, **kwargs: iter(rows))
    real_import_module = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "datasets":
            return fake_datasets
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    got = list(fetch_fleurs.stream_fleurs_samples("en_us", "test", limit=2))
    assert len(got) == 2
    assert got[0].sample_id == "7"
    assert got[0].text == "Mocked HF row one."
    assert got[1].sample_id == "8"
    assert got[1].text == "Mocked HF row two."


def test_to_mono_16k_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        to_mono_16k(_tone(16_000, 0.1), 0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        to_mono_16k(np.zeros((2, 2, 2), dtype=np.float32), 16_000)


def test_main_cli_success_and_error(tmp_path: Path) -> None:
    samples = [
        FleursSample(
            sample_id="5",
            speaker_id="spk_5",
            audio=_tone(16_000, 0.1),
            sample_rate=16_000,
            text="CLI smoke sample.",
        )
    ]
    rc_ok = main(
        [
            "--config",
            "sw_ke",
            "--split",
            "test",
            "--limit",
            "1",
            "--out-dir",
            str(tmp_path / "audio"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--messages-out",
            str(tmp_path / "messages.json"),
        ],
        stream_fn=_fake_stream(samples),
        root=tmp_path,
    )
    assert rc_ok == 0
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "messages.json").is_file()

    rc_empty = main(
        [
            "--out-dir",
            str(tmp_path / "audio2"),
            "--manifest",
            str(tmp_path / "manifest2.json"),
            "--messages-out",
            str(tmp_path / "messages2.json"),
        ],
        stream_fn=_fake_stream([]),
        root=tmp_path,
    )
    assert rc_empty == 1

    rc_bad_limit = main(
        ["--limit", "0"],
        stream_fn=_fake_stream(samples),
        root=tmp_path,
    )
    assert rc_bad_limit == 1
