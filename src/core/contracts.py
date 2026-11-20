"""Helpers for the Message, Record, Redaction, Gold, and results interfaces.

Validation covers structural rules and cross-file consistency; reviewers must
still check semantic labeling and whether every name and phone was redacted.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self

ACTIVITY_TYPES = frozenset(
    {"household_outreach", "health_screening", "community_meeting", "health_referral", "other"}
)
REDACTION_TYPES = frozenset({"NAME", "PHONE"})
VERIFICATION_STATUSES = frozenset({"pending", "not_applicable", "verified", "rejected"})
SOURCES = frozenset({"text", "audio"})
_ID = re.compile(r"([a-z0-9]+)_([0-9]+)")
_LEADING = re.compile(r"^(the|a|an|in|at|on|by)\s", re.IGNORECASE)
_TEXT_FIELDS = ("location", "date_text", "event_name", "event_desc")
_MESSAGE: dict[str, tuple[type, ...]] = {"id": (str,), "sender_id": (str,), "message": (str,)}
_MESSAGE_OPTIONAL: dict[str, tuple[type, ...]] = {"source": (str,), "asr_text": (str,)}
_RECORD = {
    "id": (str,),
    "sender_id": (str,),
    "source": (str,),
    "relevant": (bool,),
    "activity_type": (str, type(None)),
    "location": (str, type(None)),
    "reported_count": (int, type(None)),
    "date_text": (str, type(None)),
    "event_name": (str, type(None)),
    "event_desc": (str, type(None)),
    "text_redacted": (str,),
    "redactions": (list,),
    "confidence": (int, float, type(None)),
    "verification_status": (str,),
}
_GOLD = {"id": (str,), "relevant": (bool,), "activity_type": (str, type(None))}
_GOLD_OPTIONAL = {k: _RECORD[k] for k in (*_TEXT_FIELDS, "reported_count")}
_GOLD_OPTIONAL["pii"] = (list,)
_RESULT = {
    "n_messages": (int,),
    "classification_acc_pct": (int, float, type(None)),
    "filter_precision_pct": (int, float, type(None)),
    "filter_precision_n": (int,),
    "filter_recall_pct": (int, float, type(None)),
    "filter_recall_n": (int,),
    "attribute_f1_pct": (int, float, type(None)),
    "attribute_n": (int,),
    "pii_recall_pct": (int, float, type(None)),
    "pii_n": (int,),
    "pii_skipped_messages": (int,),
    "over_redactions": (int, type(None)),
}


def _type_ok(value: Any, types: tuple[type, ...]) -> bool:
    return type(value) in types


def _fields(
    obj: dict[str, Any],
    required: dict[str, tuple[type, ...]],
    optional: dict[str, tuple[type, ...]],
    where: str,
) -> list[str]:
    problems = [f"{where}: missing field {k}" for k in required if k not in obj]
    problems += [
        f"{where}: unknown field {k}" for k in obj if k not in required and k not in optional
    ]
    problems += [
        f"{where}: wrong type for {k}"
        for k, types in (required | optional).items()
        if k in obj and not _type_ok(obj[k], types)
    ]
    return problems


def _spans(spans: list[Any], where: str, gold: bool = False) -> list[str]:
    problems: list[str] = []
    last_end = 0
    gold_offsets: list[tuple[int, int]] = []
    required: dict[str, tuple[type, ...]] = {"type": (str,), "start": (int,), "end": (int,)}
    if gold:
        required["text"] = (str,)
    for i, span in enumerate(spans):
        label = f"{where}[{i}]"
        if not isinstance(span, dict):
            problems.append(f"{label}: not an object")
            continue
        errors = _fields(span, required, {}, label)
        problems.extend(errors)
        if errors:
            continue
        if span["type"] not in REDACTION_TYPES:
            problems.append(f"{label}: invalid redaction type")
        if not 0 <= span["start"] < span["end"]:
            problems.append(f"{label}: invalid offsets")
        elif gold:
            if len(span["text"]) != span["end"] - span["start"]:
                problems.append(f"{label}: text length does not match its offsets")
            gold_offsets.append((span["start"], span["end"]))
        elif span["start"] < last_end:
            problems.append(f"{label}: overlapping or unsorted spans")
        else:
            last_end = span["end"]
    for start, end in sorted(gold_offsets):
        if start < last_end:
            problems.append(f"{where}: overlapping gold spans")
        last_end = max(last_end, end)
    return problems


def _values(
    obj: dict[str, Any], text: str | None, where: str, *, defer_placeholders: bool = False
) -> list[str]:
    problems = []
    for key in _TEXT_FIELDS:
        value = obj.get(key)
        if not isinstance(value, str):
            continue
        if (
            not value
            or value != value.strip()
            or (
                unicodedata.category(value[-1]).startswith("P")
                and not value.endswith(("[NAME]", "[PHONE]"))
            )
            or _LEADING.match(value)
        ):
            problems.append(f"{where}: invalid copied value for {key}")
        is_placeholder_value = "[NAME]" in value or "[PHONE]" in value
        if (
            value
            and text is not None
            and not (defer_placeholders and is_placeholder_value)
            and value.casefold() not in text.casefold()
        ):
            problems.append(f"{where}: {key} is not in text")
    return problems


def apply_redactions(text: str, redactions: list[dict[str, Any]] | list[Redaction]) -> str:
    """Apply sorted, non-overlapping spans measured in the original text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(redactions, list):
        raise TypeError("redactions must be a list")
    spans = [s.to_dict() if isinstance(s, Redaction) else s for s in redactions]
    errors = _spans(spans, "redactions")
    if errors:
        raise ValueError("\n".join(errors))
    if any(s["end"] > len(text) for s in spans):
        raise ValueError("redaction extends past the text")
    for span in reversed(spans):
        text = text[: span["start"]] + f"[{span['type']}]" + text[span["end"] :]
    return text


def validate_file(kind: str, data: Any, messages: Any = None) -> list[str]:
    """Report structural problems; semantic labeling and PII still require review."""
    if not isinstance(kind, str) or kind not in {"messages", "transcripts", "records", "gold"}:
        raise ValueError(f"unknown kind: {kind}")
    if not isinstance(data, list):
        return ["top level must be an array"]
    problems: list[str] = []
    use_messages = messages is not None and kind in {"records", "gold"}
    if use_messages:
        reference_problems = validate_file("messages", messages)
        problems.extend(f"messages: {problem}" for problem in reference_problems)
        use_messages = not reference_problems
    by_id: dict[str, dict[str, Any]] = (
        {m["id"]: m for m in messages if isinstance(m, dict) and isinstance(m.get("id"), str)}
        if use_messages and isinstance(messages, list)
        else {}
    )
    ids: list[Any] = []
    for i, obj in enumerate(data):
        where = f"item {i}"
        if not isinstance(obj, dict):
            problems.append(f"{where}: not an object")
            continue
        id_ = obj.get("id")
        ids.append(id_)
        if not isinstance(id_, str) or _ID.fullmatch(id_) is None:
            problems.append(f"{where}: invalid id")
        message = by_id.get(id_) if isinstance(id_, str) else None
        if kind in {"messages", "transcripts"}:
            problems += _fields(obj, _MESSAGE, _MESSAGE_OPTIONAL, where)
            source = obj.get("source")
            if source is not None and (not isinstance(source, str) or source not in SOURCES):
                problems.append(f"{where}: invalid source")
            if kind == "transcripts" and source != "audio":
                problems.append(f"{where}: transcript must have audio source")
            if "asr_text" in obj and source != "audio":
                problems.append(f"{where}: asr_text requires a cleaned audio transcript")
            continue
        record = kind == "records"
        problems += _fields(
            obj, _RECORD if record else _GOLD, {} if record else _GOLD_OPTIONAL, where
        )
        relevant, activity = obj.get("relevant"), obj.get("activity_type")
        if relevant is True and (not isinstance(activity, str) or activity not in ACTIVITY_TYPES):
            problems.append(f"{where}: relevant activity type invalid")
        if relevant is False and activity is not None:
            problems.append(f"{where}: irrelevant activity type must be null")
        count = obj.get("reported_count")
        if type(count) is int and count < 0:
            problems.append(f"{where}: negative reported_count")
        if record:
            status = obj.get("verification_status")
            if (
                not isinstance(status, str)
                or status not in VERIFICATION_STATUSES
                or (relevant is False) != (status == "not_applicable")
            ):
                problems.append(f"{where}: invalid verification_status")
            if not isinstance(obj.get("source"), str) or obj["source"] not in SOURCES:
                problems.append(f"{where}: invalid source")
            confidence = obj.get("confidence")
            if (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and not 0 <= confidence <= 1
            ):
                problems.append(f"{where}: confidence out of range")
            text = obj.get("text_redacted")
            spans = obj.get("redactions")
            if isinstance(spans, list):
                problems += _spans(spans, where)
                if isinstance(text, str) and message is None:
                    for t in REDACTION_TYPES:
                        n = sum(1 for s in spans if isinstance(s, dict) and s.get("type") == t)
                        if text.count(f"[{t}]") < n:
                            problems.append(f"{where}: missing placeholder for {t}")
            problems += _values(obj, text if isinstance(text, str) else None, where)
            if message is not None:
                if obj.get("sender_id") != message.get("sender_id"):
                    problems.append(f"{where}: sender_id differs")
                if obj.get("source") != message.get("source", "text"):
                    problems.append(f"{where}: source differs")
                original = message.get("message")
                if (
                    isinstance(original, str)
                    and isinstance(spans, list)
                    and not _spans(spans, where)
                ):
                    try:
                        expected = apply_redactions(original, spans)
                    except ValueError as exc:
                        problems.append(f"{where}: {exc}")
                    else:
                        if text != expected:
                            problems.append(f"{where}: redacted text does not match message")
        else:
            original = message.get("message") if message else None
            spans = obj.get("pii")
            value_text = original if isinstance(original, str) else None
            defer_placeholders = True
            if (
                value_text is not None
                and isinstance(original, str)
                and isinstance(spans, list)
                and not _spans(spans, where, True)
                and all(
                    span["end"] <= len(original)
                    and original[span["start"] : span["end"]] == span["text"]
                    for span in spans
                )
            ):
                plain = sorted(
                    ({k: span[k] for k in ("type", "start", "end")} for span in spans),
                    key=lambda span: span["start"],
                )
                value_text = apply_redactions(value_text, plain)
                defer_placeholders = False
            problems += _values(obj, value_text, where, defer_placeholders=defer_placeholders)
            if isinstance(spans, list):
                problems += _spans(spans, where, gold=True)
                if isinstance(original, str):
                    for span in spans:
                        if (
                            isinstance(span, dict)
                            and type(span.get("start")) is int
                            and type(span.get("end")) is int
                        ):
                            if span["end"] > len(original):
                                problems.append(f"{where}: PII span exceeds message")
                            elif original[span["start"] : span["end"]] != span.get("text"):
                                problems.append(f"{where}: PII text does not match offsets")
    string_ids = [id_ for id_ in ids if isinstance(id_, str)]
    if len(set(string_ids)) != len(string_ids):
        problems.append("duplicate IDs")
    widths: dict[str, set[int]] = {}
    for id_ in string_ids:
        match = _ID.fullmatch(id_)
        if match:
            widths.setdefault(match.group(1), set()).add(len(match.group(2)))
    problems += [
        f"{prefix}: inconsistent id widths" for prefix, sizes in widths.items() if len(sizes) > 1
    ]
    if ids != sorted(ids, key=str):
        problems.append("IDs not sorted")
    if kind == "records" and use_messages:
        for id_ in sorted(set(by_id) - set(string_ids)):
            problems.append(f"no record for {id_}")
        for id_ in sorted(set(string_ids) - set(by_id)):
            problems.append(f"no message for {id_}")
    return problems


class _Model:
    required: ClassVar[dict[str, tuple[type, ...]]] = {}
    optional: ClassVar[dict[str, tuple[type, ...]]] = {}
    kind: ClassVar[str | None] = None

    def __init__(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        errors = _fields(data, self.required, self.optional, type(self).__name__)
        if self.kind:
            errors += validate_file(self.kind, [data])
        elif isinstance(self, (Redaction, GoldPiiSpan)):
            errors += _spans([data], type(self).__name__, isinstance(self, GoldPiiSpan))
        elif isinstance(self, RecordResults):
            for key, value in data.items():
                if not isinstance(key, str):
                    continue
                if key.endswith("_pct"):
                    if type(value) in (int, float) and not 0 <= value <= 100:
                        errors.append(f"{key}: percentage must be in 0..100")
                    elif type(value) in (int, float) and round(value, 2) != value:
                        errors.append(f"{key}: percentage must be rounded to two decimal places")
                elif type(value) is int and value < 0:
                    errors.append(f"{key}: count must be nonnegative")
            for metric, count in (
                ("classification_acc_pct", "n_messages"),
                ("filter_precision_pct", "filter_precision_n"),
                ("filter_recall_pct", "filter_recall_n"),
                ("attribute_f1_pct", "attribute_n"),
                ("pii_recall_pct", "pii_n"),
            ):
                if (
                    type(data.get(count)) is int
                    and data[count] == 0
                    and data.get(metric) is not None
                ):
                    errors.append(f"{metric}: a zero denominator requires null")
        if errors:
            raise ValueError("\n".join(errors))
        self._data = copy.deepcopy(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {key: self._data[key] for key in (*self.required, *self.optional) if key in self._data}
        )


class Message(_Model):
    required, optional, kind = _MESSAGE, _MESSAGE_OPTIONAL, "messages"


class Redaction(_Model):
    required: ClassVar[dict[str, tuple[type, ...]]] = {
        "type": (str,),
        "start": (int,),
        "end": (int,),
    }


class Record(_Model):
    required, kind = _RECORD, "records"


class GoldPiiSpan(_Model):
    required: ClassVar[dict[str, tuple[type, ...]]] = {
        "type": (str,),
        "start": (int,),
        "end": (int,),
        "text": (str,),
    }


class Gold(_Model):
    required, optional, kind = _GOLD, _GOLD_OPTIONAL, "gold"


class RecordResults(_Model):
    required = _RESULT


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> NoReturn:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _check_json_value(value: Any, active: set[int]) -> None:
    if type(value) in (str, int, float, bool, type(None)):
        return
    if type(value) not in (dict, list):
        raise ValueError("JSON values must use only dictionaries, lists, and JSON scalar types")
    identity = id(value)
    if identity in active:
        raise ValueError("JSON values cannot contain a cycle")
    active.add(identity)
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object field names must be strings")
        children = value.values()
    else:
        children = value
    for child in children:
        _check_json_value(child, active)
    active.remove(identity)


def _load[ModelT: _Model](
    path: str | Path, kind: str, cls: type[ModelT], messages: list[dict[str, Any]] | None = None
) -> list[ModelT]:
    with open(path, encoding="utf-8") as stream:
        data: Any = json.load(
            stream, object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite_constant
        )
    errors = validate_file(kind, data, messages)
    if errors:
        raise ValueError("\n".join(errors))
    return [cls.from_dict(item) for item in data]


def load_messages(path: str | Path) -> list[Message]:
    """Load a Message array, raising ValueError for invalid data."""
    return _load(path, "messages", Message)


def load_records(path: str | Path, messages: list[dict[str, Any]] | None = None) -> list[Record]:
    """Load Records and optionally compare them with their input Messages."""
    return _load(path, "records", Record, messages)


def load_gold(path: str | Path, messages: list[dict[str, Any]] | None = None) -> list[Gold]:
    """Load Gold entries and optionally compare them with their input Messages."""
    return _load(path, "gold", Gold, messages)


def write_json(path: str | Path, items: Sequence[_Model | dict[str, Any]]) -> None:
    """Write an id-bearing JSON array in sorted order without coercing values."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("items must be a sequence of models or dictionaries")
    if any(not isinstance(item, (_Model, dict)) for item in items):
        raise TypeError("each item must be a model or dictionary")
    values = [item.to_dict() if isinstance(item, _Model) else item for item in items]
    if any(not isinstance(item.get("id"), str) for item in values):
        raise ValueError("every item must have a string id")
    ids = [item["id"] for item in values]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    widths: dict[str, set[int]] = {}
    for ident in ids:
        match = _ID.fullmatch(ident)
        if match is None:
            raise ValueError(f"invalid id: {ident}")
        widths.setdefault(match.group(1), set()).add(len(match.group(2)))
    if any(len(sizes) > 1 for sizes in widths.values()):
        raise ValueError("ids for each dataset must use a consistent width")
    values.sort(key=lambda item: item["id"])
    _check_json_value(values, set())
    serialized = json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False)
    serialized.encode("utf-8")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(serialized)


def build_record(
    message: Message | dict[str, Any],
    *,
    relevant: bool,
    activity_type: str | None,
    location: str | None,
    reported_count: int | None,
    date_text: str | None,
    event_name: str | None,
    event_desc: str | None,
    redactions: list[dict[str, Any]],
    confidence: int | float | None,
) -> dict[str, Any]:
    """Build a Record from a Message and precomputed labeling and redaction decisions."""
    data = (
        message.to_dict() if isinstance(message, Message) else Message.from_dict(message).to_dict()
    )
    result = {
        "id": data["id"],
        "sender_id": data["sender_id"],
        "source": data.get("source", "text"),
        "relevant": relevant,
        "activity_type": activity_type,
        "location": location,
        "reported_count": reported_count,
        "date_text": date_text,
        "event_name": event_name,
        "event_desc": event_desc,
        "text_redacted": apply_redactions(data["message"], redactions),
        "redactions": redactions,
        "confidence": confidence,
        "verification_status": "pending" if relevant else "not_applicable",
    }
    problems = validate_file("records", [result], [data])
    if problems:
        raise ValueError("\n".join(problems))
    return result
