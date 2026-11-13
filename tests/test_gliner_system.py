"""Unit tests for src/nlp/gliner_system.py (fast offline tests without downloading weights)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import contracts  # noqa: E402
from src.nlp import gliner_system  # noqa: E402


class FakeGLiNERSystem(gliner_system.GLiNERSystem):
    """Deterministic stub overriding entity prediction for offline unit tests."""

    def __init__(
        self, fake_entities_by_substr: dict[str, list[dict[str, Any]]] | None = None
    ) -> None:
        super().__init__()
        self._fake_map = fake_entities_by_substr or {}

    def _predict_entities(self, text: str) -> list[dict[str, Any]]:
        for key, ents in self._fake_map.items():
            if key in text:
                return ents
        return []


def test_resolve_gliner_model_prefers_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "gliner_medium-v2.1"
    local_dir.mkdir()
    monkeypatch.setattr(gliner_system, "LOCAL_MODEL_DIR", local_dir)

    assert gliner_system.resolve_gliner_model() == (gliner_system.MODEL_ID, False)

    (local_dir / "gliner_config.json").write_text("{}", encoding="utf-8")
    (local_dir / "pytorch_model.bin").write_bytes(b"weights")
    assert gliner_system.resolve_gliner_model() == (str(local_dir), True)
    assert gliner_system.resolve_gliner_model("custom/repo") == ("custom/repo", False)


def test_trim_copied_value_respects_v2_rules() -> None:
    assert gliner_system.trim_copied_value("in the Nakuru.") == "Nakuru"
    assert gliner_system.trim_copied_value("at Rongai Health Center,") == "Rongai Health Center"
    assert gliner_system.trim_copied_value("Grace Wanjiru", "Hello [NAME]") is None


def test_honorific_after_preposition_is_name_not_location() -> None:
    text = (
        "Congratulations to Mama Puredi Jeptu on the new baby. "
        "Visit in Kabarak. Call Ruth on 0756-000819."
    )
    # Use clean stub entities aligned to exact offsets.
    kab_start = text.index("Kabarak")
    stub = FakeGLiNERSystem(
        {
            "Mama Puredi": [
                {
                    "start": kab_start,
                    "end": kab_start + len("Kabarak"),
                    "text": "Kabarak",
                    "label": "location",
                    "score": 0.92,
                },
                {
                    "start": text.index("Ruth"),
                    "end": text.index("Ruth") + 4,
                    "text": "Ruth",
                    "label": "person",
                    "score": 0.88,
                },
            ]
        }
    )
    msg = {"id": "msg_01", "sender_id": "worker_01", "source": "audio", "message": text}
    record, loc_spans = stub.process_message_detailed(msg)
    assert record["relevant"] is False
    assert "[NAME]" in record["text_redacted"]
    assert "Mama Puredi Jeptu" not in record["text_redacted"]
    assert "Ruth" not in record["text_redacted"]
    assert "[PHONE]" in record["text_redacted"]
    loc_texts = [text[s["start"] : s["end"]] for s in loc_spans]
    assert loc_texts == ["Kabarak"]
    assert not contracts.validate_file("records", [record], [msg])


def test_lowercase_intro_and_possessive_spacing_and_role_words() -> None:
    msg1 = {
        "id": "msg_01",
        "sender_id": "worker_01",
        "source": "audio",
        "message": (
            "Good evening everyone its faith aching. "
            "If you need the records Lucy s number is 0745-000638."
        ),
    }
    msg2 = {
        "id": "msg_02",
        "sender_id": "worker_02",
        "source": "audio",
        "message": (
            "Hello team Ditsmasiwambui I want to share about the community dialogue on "
            "gender-based violence we held this afternoon at the chiefs camp in Solai. "
            "35 women attended. Corporal Enmuchwa explained. Call 0701-000-593."
        ),
    }
    stub = FakeGLiNERSystem(
        {
            "faith aching": [],
            "Ditsmasiwambui": [
                {
                    "start": msg2["message"].index("Solai"),
                    "end": msg2["message"].index("Solai") + 5,
                    "text": "Solai",
                    "label": "location",
                    "score": 0.9,
                }
            ],
        }
    )
    records = stub.process_messages([msg1, msg2])
    assert "faith aching" not in records[0]["text_redacted"]
    assert "Lucy" not in records[0]["text_redacted"]
    assert "[PHONE]" in records[0]["text_redacted"]

    assert records[1]["relevant"] is True
    assert records[1]["activity_type"] == "community_meeting"
    assert records[1]["reported_count"] == 35
    assert "Ditsmasiwambui" not in records[1]["text_redacted"]
    assert "Corporal Enmuchwa" not in records[1]["text_redacted"]


def test_extract_reported_count_filters_time_class_and_money() -> None:
    text = (
        "Yesterday we went to Sarambiai Primary School for the menstrual hygiene session "
        "with the girls in class 6-8. 60 girls attended. Together they saved 18,000 shillings."
    )
    assert gliner_system.extract_reported_count(text) == 60
    assert gliner_system.extract_reported_count("We waited for almost 2 hours on the 15th.") is None
