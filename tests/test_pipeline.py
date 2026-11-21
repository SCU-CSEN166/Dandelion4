"""Tests for src/pipeline/pipeline.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from src.pipeline import pipeline


class _FakeASR:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def __call__(self, waves: list[Any]) -> list[str]:
        return self._texts[: len(waves)]

    def metadata(self) -> dict[str, Any]:
        return {"model": "fake"}


class _FakeNLP:
    def process_message_detailed(
        self, msg: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        text = str(msg["message"])
        record = {
            "id": str(msg["id"]),
            "sender_id": str(msg["sender_id"]),
            "source": str(msg["source"]),
            "text_redacted": text,
            "relevant": True,
            "activity_type": "community_meeting",
            "location": None,
            "date_text": None,
            "event_name": None,
            "event_desc": None,
            "reported_count": 20,
            "redactions": [],
            "confidence": 0.9,
            "verification_status": "pending",
        }
        return record, []


def test_run_on_mixed_folder(tmp_path: Path) -> None:
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    soundfile.write(in_dir / "clip1.wav", np.zeros(16_000, dtype=np.float32), 16_000)
    (in_dir / "note.txt").write_text(
        "Held a family planning outreach for 20 women in Nakuru.", encoding="utf-8"
    )
    (in_dir / "items.json").write_text(
        json.dumps([{"text": "Visited 15 households in Narok."}]),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    summary = pipeline.run_on_input(
        in_dir,
        out_dir=out_dir,
        asr_backend=_FakeASR(["Met 20 mothers at Nakuru clinic"]),
        nlp_system=_FakeNLP(),  # type: ignore[arg-type]
    )

    assert summary["audio_count"] == 1
    assert summary["text_count"] == 2
    assert summary["total_records"] == 3
    assert not (out_dir / "messages.json").exists()
    assert (out_dir / "records.json").is_file()
    with (out_dir / "outputs.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["source_type"] == "audio"
    assert rows[1]["source_type"] == "text"
    assert rows[2]["source_type"] == "text"


def test_discover_inputs_extracts_zip_inside_directory(tmp_path: Path) -> None:
    import zipfile

    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    wav_path = tmp_path / "clip1.wav"
    soundfile.write(wav_path, np.zeros(16_000, dtype=np.float32), 16_000)
    with zipfile.ZipFile(in_dir / "bundle.zip", "w") as zf:
        zf.write(wav_path, arcname="clip1.wav")

    audio_files, text_files = pipeline.discover_inputs(in_dir, tmp_path / "extract")
    assert [p.name for p in audio_files] == ["clip1.wav"]
    assert text_files == []


def test_discover_inputs_rejects_empty_folder(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No supported audio or text files"):
        pipeline.discover_inputs(empty, tmp_path / "extract")
