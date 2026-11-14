"""Offline unit tests for demo/run_demo.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import run_demo  # noqa: E402
from src.core import contracts  # noqa: E402
from src.pipeline import hidden_test_pipeline  # noqa: E402


def test_parse_args_defaults() -> None:
    args = run_demo.parse_args([])
    assert args.input == run_demo.DEFAULT_INPUT
    assert args.out == run_demo.DEFAULT_OUT_CSV
    assert args.transcripts is None
    assert args.expected is None
    assert args.no_eval is False
    assert args.device == "auto"
    assert args.show_raw is False
    assert args.no_clean is False

    cpu_args = run_demo.parse_args(["--device", "cpu"])
    assert cpu_args.device == "cpu"


def _sample_demo_payload() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]
]:
    messages: list[dict[str, Any]] = [
        {
            "id": "msg_01",
            "sender_id": "worker_01",
            "source": "audio",
            "message": "Hello team Grace Wanjiru from Rongai. 42 women came. Call 0711000482.",
            "asr_text": (
                "Hello team Grace Wanjiru from Rongai. Forty-two women came. Call 0711000482."
            ),
        },
        {
            "id": "msg_02",
            "sender_id": "worker_02",
            "source": "audio",
            "message": "Matatu broke down in Solai. Call Rose Injeri on 0733000156.",
            "asr_text": "Matatu broke down in Solai. Call Rose Injeri on 0733000156.",
        },
    ]
    m1_text = messages[0]["message"]
    m2_text = messages[1]["message"]
    records: list[dict[str, Any]] = [
        contracts.build_record(
            messages[0],
            relevant=True,
            activity_type="community_meeting",
            location="Rongai",
            reported_count=42,
            date_text=None,
            event_name=None,
            event_desc=None,
            redactions=[
                {
                    "type": "NAME",
                    "start": m1_text.index("Grace Wanjiru"),
                    "end": m1_text.index("Grace Wanjiru") + len("Grace Wanjiru"),
                },
                {
                    "type": "PHONE",
                    "start": m1_text.index("0711000482"),
                    "end": m1_text.index("0711000482") + len("0711000482"),
                },
            ],
            confidence=0.9,
        ),
        contracts.build_record(
            messages[1],
            relevant=False,
            activity_type=None,
            location="Solai",
            reported_count=None,
            date_text=None,
            event_name=None,
            event_desc=None,
            redactions=[
                {
                    "type": "NAME",
                    "start": m2_text.index("Rose Injeri"),
                    "end": m2_text.index("Rose Injeri") + len("Rose Injeri"),
                },
                {
                    "type": "PHONE",
                    "start": m2_text.index("0733000156"),
                    "end": m2_text.index("0733000156") + len("0733000156"),
                },
            ],
            confidence=0.85,
        ),
    ]
    rows = [
        {
            "Message_id": "1",
            "anonymized_text": "Hello team [NAME] from [LOCATION]. 42 women came. Call [PHONE].",
            "relevance": "Event",
            "activity_type": "Family planning outreach",
            "participants": "42",
            "duplicate_of": "none",
        },
        {
            "Message_id": "2",
            "anonymized_text": "Matatu broke down in [LOCATION]. Call [NAME] on [PHONE].",
            "relevance": "Chit-chat",
            "activity_type": "",
            "participants": "",
            "duplicate_of": "none",
        },
    ]
    return messages, records, rows


def _make_fake_pipeline(
    messages: list[dict[str, Any]],
    records: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> Any:
    def fake_pipeline(**kwargs: Any) -> list[dict[str, str]]:
        art = Path(kwargs["artifacts_dir"])
        art.mkdir(parents=True, exist_ok=True)
        (art / "predicted_transcripts.json").write_text(json.dumps(messages), encoding="utf-8")
        (art / "predicted_records.json").write_text(json.dumps(records), encoding="utf-8")
        (art / "predicted_transcripts.meta.json").write_text(
            json.dumps({"audio_seconds_total": 75.0, "wall_seconds": 1.2, "device": "cuda:0"}),
            encoding="utf-8",
        )
        hidden_test_pipeline.write_outputs_csv(Path(kwargs["out_csv"]), rows)
        return rows

    return fake_pipeline


def test_run_demo_prints_summary_and_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_csv = tmp_path / "demo_outputs.csv"
    artifacts_dir = tmp_path / "artifacts"
    messages, records, rows = _sample_demo_payload()

    code = run_demo.run_demo(
        input_audio=tmp_path / "dummy.zip",
        out_csv=out_csv,
        artifacts_dir=artifacts_dir,
        device="cpu",
        show_raw=True,
        pipeline_fn=_make_fake_pipeline(messages, records, rows),
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "Device request : cpu" in stdout
    assert "DANDELION AUDIO-TO-INSIGHTS DEMO REPORT" in stdout
    assert "Contract checks: PASSED (2 Message items, 2 Record items valid)" in stdout
    assert "Family planning outreach" in stdout
    assert "Forty-two women came" in stdout
    assert "Message_id,anonymized_text,relevance,activity_type,participants,duplicate_of" in stdout


def test_run_demo_with_expected_csv_writes_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_csv = tmp_path / "demo_outputs.csv"
    artifacts_dir = tmp_path / "artifacts"
    expected_csv = tmp_path / "expected.csv"
    expected_csv.write_text(
        "message_id,relevance,activity_type,participants,duplicate_of,"
        "expected_anonymized_text,transcript\n"
        '1,Event,Family planning outreach,42,none,'
        '"Hello team [NAME] from [LOCATION]. 42 women came. Call [PHONE].",'
        '"Hello team Grace Wanjiru from Rongai. 42 women came. Call 0711000482."\n'
        '2,Chit-chat,,,"none","Matatu broke down in [LOCATION]. Call [NAME] on [PHONE].",'
        '"Matatu broke down in Solai. Call Rose Injeri on 0733000156."\n',
        encoding="utf-8",
    )
    messages, records, rows = _sample_demo_payload()

    code = run_demo.run_demo(
        input_audio=tmp_path / "dummy.zip",
        out_csv=out_csv,
        artifacts_dir=artifacts_dir,
        expected_csv=expected_csv,
        pipeline_fn=_make_fake_pipeline(messages, records, rows),
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "VALIDATION METRICS vs. expected.csv" in stdout
    metrics = json.loads((artifacts_dir / "demo_metrics.json").read_text(encoding="utf-8"))
    assert metrics["classification_accuracy_pct"] == 100.0
    assert metrics["asr_clean_wer_pct"] == 0.0
