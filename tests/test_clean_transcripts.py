"""Offline unit tests for src/clean_transcripts.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import clean_transcripts
import contracts


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("zero cases today", "0 cases today"),
        ("We referred six women.", "We referred 6 women."),
        ("There were nineteen people.", "There were 19 people."),
        ("We screened twenty patients.", "We screened 20 patients."),
        ("We reached twenty-five households.", "We reached 25 households."),
        ("We reached twenty five households.", "We reached 25 households."),
        ("Up to ninety-nine families attended.", "Up to 99 families attended."),
        ("About hundred people came.", "About 100 people came."),
        ("One hundred participants joined.", "100 participants joined."),
        ("We visited one hundred twenty homes.", "We visited 120 homes."),
        ("We visited one hundred and forty-two homes.", "We visited 142 homes."),
        ("Three hundred and five people.", "305 people."),
        ("Five and six women attended.", "5 and 6 women attended."),
        ("One, two, three clinics.", "1, 2, 3 clinics."),
        ("One hundred and counting.", "100 and counting."),
        ("Two three four.", "2 3 4."),
        ("Someone checked one's bag.", "Someone checked one's bag."),
        ("Already 48 women in Nakuru.", "Already 48 women in Nakuru."),
    ],
)
def test_convert_number_words(raw: str, expected: str) -> None:
    assert clean_transcripts.convert_number_words(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "We referred 33 women to the clinic in Narak.",
            "We referred 33 women to the clinic in Narok.",
        ),
        ("We visited 7 households in naraq today.", "We visited 7 households in Narok today."),
        ("Our team reached 22 families in Kajiadu.", "Our team reached 22 families in Kajiado."),
        ("Health screening in Nakurro.", "Health screening in Nakuru."),
        (
            "We referred 8 women to the clinic in Nakulu.",
            "We referred 8 women to the clinic in Nakuru.",
        ),
        ("Workshop in balingo today.", "Workshop in Baringo today."),
    ],
)
def test_normalize_place_names(raw: str, expected: str) -> None:
    assert clean_transcripts.normalize_place_names(raw) == expected


def test_clean_message_sets_asr_text_when_missing() -> None:
    raw_msg: dict[str, Any] = {
        "id": "msg_005",
        "sender_id": "worker_01",
        "message": "We referred thirty-three women to the clinic in Narak.",
    }
    cleaned = clean_transcripts.clean_message(raw_msg)
    assert cleaned == {
        "id": "msg_005",
        "sender_id": "worker_01",
        "source": "audio",
        "message": "We referred 33 women to the clinic in Narok.",
        "asr_text": "We referred thirty-three women to the clinic in Narak.",
    }


def test_clean_message_preserves_existing_asr_text() -> None:
    raw_msg: dict[str, Any] = {
        "id": "msg_005",
        "sender_id": "worker_01",
        "source": "audio",
        "message": "We referred thirty-three women to the clinic in Narok.",
        "asr_text": "We referred thirty three women to the clinic in Narak.",
    }
    cleaned = clean_transcripts.clean_message(raw_msg)
    assert cleaned["asr_text"] == "We referred thirty three women to the clinic in Narak."
    assert cleaned["message"] == "We referred 33 women to the clinic in Narok."
    assert cleaned["source"] == "audio"


def test_clean_transcripts_roundtrip_and_contract_validation(tmp_path: Path) -> None:
    input_path = tmp_path / "transcripts_in.json"
    output_path = tmp_path / "transcripts_out.json"
    payload = [
        {
            "id": "msg_001",
            "sender_id": "worker_01",
            "source": "audio",
            "message": "We referred six women in Nakurro.",
        },
        {
            "id": "msg_002",
            "sender_id": "worker_02",
            "source": "audio",
            "message": "We visited one hundred twenty homes in Kajiadu.",
        },
    ]
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = clean_transcripts.main([str(input_path), "--out", str(output_path)])
    assert exit_code == 0

    written: Any = json.loads(output_path.read_text(encoding="utf-8"))
    assert contracts.validate_file("transcripts", written) == []
    assert [item["id"] for item in written] == ["msg_001", "msg_002"]
    assert written[0]["message"] == "We referred 6 women in Nakuru."
    assert written[0]["asr_text"] == "We referred six women in Nakurro."
    assert written[1]["message"] == "We visited 120 homes in Kajiado."
    assert written[1]["asr_text"] == "We visited one hundred twenty homes in Kajiadu."


def test_cli_overwrites_in_place_without_out(tmp_path: Path) -> None:
    input_path = tmp_path / "transcripts.json"
    payload = [
        {
            "id": "msg_001",
            "sender_id": "worker_01",
            "source": "audio",
            "message": "Twenty-five families in Narak.",
        }
    ]
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = clean_transcripts.main([str(input_path)])
    assert exit_code == 0

    written: Any = json.loads(input_path.read_text(encoding="utf-8"))
    assert contracts.validate_file("transcripts", written) == []
    assert written[0]["message"] == "25 families in Narok."
    assert written[0]["asr_text"] == "Twenty-five families in Narak."


def test_cli_returns_nonzero_on_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does_not_exist.json"
    exit_code = clean_transcripts.main([str(missing)])
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err
