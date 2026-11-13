"""Unit tests for src/pipeline/hidden_test_pipeline.py (MP4/zip, dedup, CSV schema, metrics)."""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe  # noqa: E402
from src.nlp import gliner_system  # noqa: E402
from src.pipeline import hidden_test_pipeline  # noqa: E402


class _FakeASR:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def __call__(self, waves: list[Any]) -> list[str]:
        assert len(waves) == len(self._texts)
        return self._texts

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "fake-whisper",
            "language": "en",
            "task": "transcribe",
            "decoding": {"num_beams": 1, "do_sample": False},
            "initial_prompt": None,
            "device": "cpu",
            "dtype": "float32",
            "torch_version": "2.6.0",
            "transformers_version": "5.0.0",
        }


class _StubNLP(gliner_system.GLiNERSystem):
    def _predict_entities(self, text: str) -> list[dict[str, Any]]:
        ents: list[dict[str, Any]] = []
        places = (
            "Kampi Yamoto dispensary",
            "Kampiamoto",
            "Camp Yamoto",
            "Rongai",
            "Solai",
            "Kabarak",
        )
        for place in places:
            idx = text.find(place)
            if idx != -1:
                ents.append(
                    {
                        "start": idx,
                        "end": idx + len(place),
                        "text": place,
                        "label": "location",
                        "score": 0.9,
                    }
                )
        return ents


def _write_test_mp4(path: Path, seconds: float = 0.5, sr: int = 24_000) -> None:
    """Create a valid mono AAC .mp4 audio file via PyAV for container reader tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(sr * seconds)
    t = np.arange(total, dtype=np.float32) / sr
    samples = (0.15 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=sr, layout="mono")
        frame_size = 1024
        for offset in range(0, total, frame_size):
            chunk = samples[offset : offset + frame_size]
            frame = av.AudioFrame.from_ndarray(chunk[None, :], format="fltp", layout="mono")
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_read_wav_decodes_mp4_via_pyav(tmp_path: Path) -> None:
    mp4_path = tmp_path / "msg01.mp4"
    _write_test_mp4(mp4_path, seconds=0.5, sr=24_000)
    wave, sr = transcribe.read_wav(mp4_path)
    assert sr == transcribe.ASR_SAMPLE_RATE
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert wave.size > 4_000


def test_collect_audio_files_sorts_zip_naturally(tmp_path: Path) -> None:
    zip_path = tmp_path / "clips.zip"
    mp4_a = tmp_path / "a.mp4"
    _write_test_mp4(mp4_a, seconds=0.25)
    raw = mp4_a.read_bytes()
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dandelion_audio/msg10.mp4", raw)
        zf.writestr("dandelion_audio/msg02.mp4", raw)
        zf.writestr("dandelion_audio/msg01.mp4", raw)
        zf.writestr("dandelion_audio/", "")

    extract_dir = tmp_path / "ext"
    entries = hidden_test_pipeline.collect_audio_files(zip_path, extract_dir)
    assert [num for num, _ in entries] == [1, 2, 10]


def test_end_to_end_pipeline_zip_dedup_and_csv_schema(tmp_path: Path) -> None:
    mp4 = tmp_path / "seed.mp4"
    _write_test_mp4(mp4, seconds=0.25)
    raw = mp4.read_bytes()
    zip_path = tmp_path / "dandelion.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(1, 5):
            zf.writestr(f"dandelion_audio/msg{i:02d}.mp4", raw)

    transcripts = [
        (
            "Hello team this is Grace Wanjiru from Rongai. Today we did family planning outreach "
            "at Kampi Yamoto dispensary. Forty-two women came and six women were referred. "
            "Call 0711-000482."
        ),
        (
            "Good evening its Faith Achieng. I was also at Kampiamoto outreach today with Grace. "
            "Just to confirm we served 42 women on family planning and six were referred. "
            "Call 0722-000917."
        ),
        (
            "Hello ladies its Rose Injeri here. The matatu from Solai broke down and we waited "
            "two hours. Who is bringing tea? Reach me on 0733-000156."
        ),
        (
            "Hello team this is Caroline Nyambura reporting from Camp Yamoto. Today we held "
            "kitchen garden training for women farmers. 18 women attended. Call 0712-000765."
        ),
    ]
    out_csv = tmp_path / "predicted_outputs.csv"
    artifacts_dir = tmp_path / "artifacts"

    rows = hidden_test_pipeline.run_hidden_test_pipeline(
        input_audio=zip_path,
        out_csv=out_csv,
        artifacts_dir=artifacts_dir,
        clean=True,
        make_asr=lambda *_args, **_kwargs: _FakeASR(transcripts),
        nlp_system=_StubNLP(),
    )
    assert len(rows) == 4
    assert (artifacts_dir / "predicted_transcripts.json").is_file()
    assert (artifacts_dir / "predicted_records.json").is_file()
    with out_csv.open(encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))
    assert list(reader[0].keys()) == list(hidden_test_pipeline.CSV_COLUMNS)

    # Event vs Chit-chat and numeric participants
    assert reader[0]["Message_id"] == "1"
    assert reader[0]["relevance"] == "Event"
    assert reader[0]["participants"] == "42"
    assert reader[0]["duplicate_of"] == "none"
    assert "[NAME]" in reader[0]["anonymized_text"]
    assert "[LOCATION]" in reader[0]["anonymized_text"]
    assert "[PHONE]" in reader[0]["anonymized_text"]

    # Row 2 duplicates Row 1
    assert reader[1]["duplicate_of"] == "1"

    # Row 3 is Chit-chat with blank activity_type & participants
    assert reader[2]["relevance"] == "Chit-chat"
    assert reader[2]["activity_type"] == ""
    assert reader[2]["participants"] == ""
    assert reader[2]["duplicate_of"] == "none"

    # Row 4 is at Camp Yamoto too, but kitchen garden (18) != family planning (42)
    assert reader[3]["relevance"] == "Event"
    assert reader[3]["participants"] == "18"
    assert reader[3]["duplicate_of"] == "none"

    # Evaluation helper sanity check
    metrics = hidden_test_pipeline.evaluate_csv_outputs(reader, reader)
    assert metrics == {
        "classification_accuracy_pct": 100.0,
        "event_precision_pct": 100.0,
        "event_recall_pct": 100.0,
        "event_f1_pct": 100.0,
        "ner_redaction_recall_pct": 100.0,
        "ner_over_redaction_count": 0.0,
        "attribute_exact_match_pct": 100.0,
        "activity_exact_match_pct": 100.0,
        "participants_exact_match_pct": 100.0,
        "duplication_f1_pct": 100.0,
        "duplication_accuracy_pct": 100.0,
    }


def test_parse_numeric_message_id_errors_on_missing_number() -> None:
    with pytest.raises(ValueError, match="cannot parse numeric message id"):
        hidden_test_pipeline.parse_numeric_message_id("no_digits_here")


def test_evaluate_expected_schema_and_transcript_wer(tmp_path: Path) -> None:
    expected_csv = tmp_path / "expected.csv"
    expected_csv.write_text(
        "\ufeffmessage_id,relevance,activity_type,participants,duplicate_of,"
        "expected_anonymized_text,transcript\n"
        'msg01.mp4,Event,Family planning,12,none,"Hi [NAME] at [LOCATION]",'
        '"Hi Grace at Nakuru"\n'
        'msg02.mp4,Chit-chat,,,"msg01.mp4","Thanks [NAME]",'
        '"Thanks Peter"\n',
        encoding="utf-8",
    )
    expected_rows = hidden_test_pipeline.load_csv_rows(expected_csv)
    pred_rows = [
        {
            "Message_id": "1",
            "relevance": "Event",
            "activity_type": "Family planning",
            "participants": "12",
            "duplicate_of": "none",
            "anonymized_text": "Hi [NAME] at [LOCATION] [PHONE]",
        },
        {
            "Message_id": "2",
            "relevance": "Event",
            "activity_type": "",
            "participants": "",
            "duplicate_of": "none",
            "anonymized_text": "Thanks Peter",
        },
    ]
    metrics = hidden_test_pipeline.evaluate_csv_outputs(pred_rows, expected_rows)
    assert metrics["classification_accuracy_pct"] == 50.0
    assert metrics["event_precision_pct"] == 50.0
    assert metrics["event_recall_pct"] == 100.0
    assert metrics["attribute_exact_match_pct"] == 100.0
    assert metrics["ner_redaction_recall_pct"] == 66.67
    assert metrics["ner_over_redaction_count"] == 1.0
    assert metrics["duplication_f1_pct"] == 0.0
    assert metrics["duplication_accuracy_pct"] == 50.0

    messages = [
        {"id": "msg01.mp4", "asr_text": "Hi Grace uh at Nakuru", "message": "Hi Grace at Nakuru"},
        {"id": "msg02.mp4", "asr_text": "Thanks Peter", "message": "Thanks Peter"},
    ]
    wer_metrics = hidden_test_pipeline.evaluate_transcript_wer(messages, expected_rows)
    assert wer_metrics["asr_raw_wer_pct"] > 0.0
    assert wer_metrics["asr_clean_wer_pct"] == 0.0
