"""End-to-end hidden test pipeline: audio (.mp4/.wav/.zip) -> ASR -> GLiNER -> deduplicated CSV.

Produces:
- Validated Interface Spec v2 ``Message[]`` transcripts (raw ASR in ``asr_text``,
  normalized text in ``message``)
- Validated Interface Spec v2 ``Record[]`` extractions (with ``NAME``/``PHONE`` redactions and
  resolved ``[LOCATION]`` spans)
- ``demo/outputs/predicted_outputs.csv`` matching the schema required by
  ``Dandelion Hidden Test.pdf``:
  ``Message_id,anonymized_text,relevance,activity_type,participants,duplicate_of``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import jiwer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import clean_transcripts, transcribe  # noqa: E402
from src.core import contracts  # noqa: E402
from src.nlp import gliner_system  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "demo" / "inputs"
DEFAULT_OUT_CSV = REPO_ROOT / "demo" / "outputs" / "predicted_outputs.csv"
DEFAULT_EXPECTED_CSV = REPO_ROOT / "demo" / "expected" / "hidden_test_outputs.csv"

CSV_COLUMNS: tuple[str, ...] = (
    "Message_id",
    "anonymized_text",
    "relevance",
    "activity_type",
    "participants",
    "duplicate_of",
)

SUPPORTED_AUDIO_SUFFIXES = frozenset(
    transcribe.CONTAINER_AUDIO_SUFFIXES | {".wav", ".flac", ".ogg"}
)

_TRAILING_DIGITS_RE = re.compile(r"(\d+)$")
_PHONE_SCRUB_RE = re.compile(r"(?<!\w)(?:\+?254|0)[\s-]?[17](?:[\s-]?\d){8}(?!\w)")
_INT_TOKEN_RE = re.compile(r"\b(\d+)\b")
_DUPLICATE_CUE_RE = re.compile(
    r"\b(?:also\s+at|confirm(?:ing)?|adding\s+to|same\s+(?:event|outreach|camp)|helping\b)",
    re.IGNORECASE,
)
_VENUE_STOPWORDS = frozenset(
    {
        "dispensary", "health", "center", "centre", "primary", "school",
        "camp", "hospital", "level", "clinic", "the",
    }
)
_PERSON_TITLES = frozenset(
    {"sister", "mama", "mrs", "mr", "ms", "dr", "corporal", "pastor", "chief"}
)
_TOPIC_STOPWORDS = frozenset(
    {
        "a", "an", "and", "at", "for", "from", "held", "holding", "in", "monthly", "my", "of", "on",
        "our", "report", "reporting", "session", "today", "the", "this", "to", "was", "we", "went",
        "with", "yesterday", "day", "group", "members", "women", "girls", "people", "mothers",
    }
)


def _natural_key(path: Path) -> list[int | str]:
    """Return a sort key that orders filenames by embedded numbers (e.g., msg2 before msg10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_numeric_message_id(identifier: str) -> int:
    """Extract the trailing integer message ID from filenames like ``msg01.mp4`` or ``msg_01``."""
    stem = Path(identifier).stem if "." in identifier else identifier
    match = _TRAILING_DIGITS_RE.search(stem)
    if match is None:
        raise ValueError(f"cannot parse numeric message id from {identifier!r}")
    return int(match.group(1))


def collect_audio_files(input_path: Path, extract_dir: Path) -> list[tuple[int, Path]]:
    """Collect naturally sorted ``(numeric_id, audio_file)`` pairs from a zip archive or folder."""
    if not input_path.exists():
        raise FileNotFoundError(f"input audio source does not exist: {input_path}")

    files: list[Path] = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            members = [
                m
                for m in zf.namelist()
                if not m.endswith("/")
                and not Path(m).name.startswith(".")
                and "__MACOSX" not in Path(m).parts
                and Path(m).suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
            ]
            for member in sorted(members, key=lambda name: _natural_key(Path(name))):
                files.append(Path(zf.extract(member, extract_dir)))
    elif input_path.is_dir():
        files = sorted(
            (
                p
                for p in input_path.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
            ),
            key=_natural_key,
        )
    else:
        raise ValueError(f"expected a .zip archive or audio directory, got {input_path}")

    if not files:
        raise ValueError(f"no supported audio files found in {input_path}")

    entries = [(parse_numeric_message_id(path.name), path) for path in files]
    entries.sort(key=lambda item: item[0])
    ids = [num for num, _ in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate numeric message ids detected: {ids}")
    return entries


def transcribe_audio_entries(
    entries: Sequence[tuple[int, Path]],
    asr: transcribe.ASRBackend,
    *,
    clean: bool = True,
    read_wav: transcribe.ReadWav = transcribe.read_wav,
) -> tuple[list[dict[str, Any]], float, float]:
    """Transcribe ordered audio files into validated v2 ``Message[]`` dicts."""
    if not entries:
        raise ValueError("no audio entries to transcribe")

    width = max(2, len(str(max(num for num, _ in entries))))
    waves = []
    clips: list[dict[str, Any]] = []
    total_audio_seconds = 0.0

    for num, audio_path in entries:
        wave, sr = read_wav(audio_path)
        wave_16k = transcribe.load_for_asr(wave, sr)
        total_audio_seconds += wave_16k.shape[0] / transcribe.ASR_SAMPLE_RATE
        waves.append(wave_16k)
        clips.append({"message_id": f"msg_{num:0{width}d}", "sender_id": f"worker_{num:0{width}d}"})

    asr.load()
    t0 = time.perf_counter()
    texts = asr(waves)
    wall_seconds = time.perf_counter() - t0

    messages: list[dict[str, Any]] = [
        dict(rec) for rec in transcribe.to_transcript_records(clips, texts)
    ]
    if clean:
        messages = [clean_transcripts.clean_message(msg) for msg in messages]

    problems = contracts.validate_file("transcripts", messages)
    if problems:
        raise ValueError("transcribed messages failed contract validation:\n" + "\n".join(problems))
    return messages, total_audio_seconds, wall_seconds


def anonymize_csv_text(
    text: str,
    redactions: Sequence[dict[str, Any]],
    location_spans: Sequence[dict[str, Any]],
) -> str:
    """Apply ``[NAME]``, ``[PHONE]``, and ``[LOCATION]`` placeholders without span overlaps."""
    chosen: list[dict[str, Any]] = sorted(
        (dict(s) for s in redactions),
        key=lambda s: (int(s["start"]), int(s["end"])),
    )

    def overlaps_chosen(start: int, end: int) -> bool:
        """Return True if [start, end) overlaps any already-selected redaction span."""
        return any(start < int(s["end"]) and end > int(s["start"]) for s in chosen)

    for loc in sorted(location_spans, key=lambda s: (int(s["start"]), int(s["end"]))):
        start, end = int(loc["start"]), int(loc["end"])
        if 0 <= start < end <= len(text) and not overlaps_chosen(start, end):
            chosen.append({"type": "LOCATION", "start": start, "end": end})

    chosen.sort(key=lambda s: (int(s["start"]), int(s["end"])), reverse=True)
    result = text
    for span in chosen:
        start, end = int(span["start"]), int(span["end"])
        result = result[:start] + f"[{span['type']}]" + result[end:]
    return result


def format_activity_label(record: dict[str, Any]) -> str:
    """Format a concise human-readable activity label for the hidden test CSV."""
    if not record.get("relevant"):
        return ""
    event_name = record.get("event_name")
    if isinstance(event_name, str) and event_name.strip():
        cleaned = re.sub(r"\s+", " ", event_name).strip()
        if cleaned.lower().startswith("monthly table banking"):
            cleaned = "Table banking"
        return cleaned[0].upper() + cleaned[1:]
    v2_act = str(record.get("activity_type") or "other").replace("_", " ").strip()
    return v2_act[0].upper() + v2_act[1:]


def _strip_venue_suffixes(place: str) -> str:
    """Remove generic venue words (e.g., 'clinic', 'school') to compare core place names."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", place.lower()) if t and t not in _VENUE_STOPWORDS]
    return "".join(tokens)


def _extract_location_keys(
    record: dict[str, Any], text: str, loc_spans: Sequence[dict[str, Any]]
) -> set[str]:
    """Collect normalized location strings from the record and detected location spans."""
    keys: set[str] = set()
    if isinstance(record.get("location"), str) and record["location"].strip():
        keys.add(record["location"].strip().lower())
    for span in loc_spans:
        raw = text[int(span["start"]) : int(span["end"])].strip().lower()
        if raw:
            keys.add(raw)
    return keys


def _location_similarity(keys_a: set[str], keys_b: set[str]) -> float:
    """Return the highest similarity score (0.0 to 1.0) between two sets of place names."""
    if not keys_a or not keys_b:
        return 0.0
    best = 0.0
    for a in keys_a:
        core_a = _strip_venue_suffixes(a)
        for b in keys_b:
            core_b = _strip_venue_suffixes(b)
            if not core_a or not core_b:
                continue
            if core_a == core_b or core_a in core_b or core_b in core_a:
                best = max(best, 0.95)
            else:
                ratio = SequenceMatcher(None, core_a, core_b).ratio()
                best = max(best, ratio)
    return best


def _extract_secondary_numbers(text: str, participant_count: int | None) -> set[int]:
    """Extract non-participant numbers (e.g., kit or bed counts) as duplicate-matching clues."""
    scrubbed = _PHONE_SCRUB_RE.sub(" ", text)
    scrubbed = clean_transcripts.convert_number_words(scrubbed)
    scrubbed = re.sub(r"\b\d+(?:,\d{3})+\b", " ", scrubbed)  # ignore currency sums like 18,000
    scrubbed = re.sub(
        r"\bclass\s+\d+(?:\s*(?:-|to)\s*\d+)?\b", " ", scrubbed, flags=re.IGNORECASE
    )
    nums: set[int] = set()
    for match in _INT_TOKEN_RE.finditer(scrubbed):
        tail = scrubbed[match.end() : match.end() + 20].lower()
        if re.match(r"\s*(?:am|pm|shillings|hours?|minutes?|days?|th|st|nd|rd)\b", tail):
            continue
        val = int(match.group(1))
        if val <= 1 or val >= 1000 or val == participant_count:
            continue
        nums.add(val)
    return nums


def _extract_topic_tokens(activity_label: str, text: str) -> set[str]:
    """Extract high-signal domain topic words from the activity label and message text."""
    base = f"{activity_label} {text}".lower()
    raw_tokens = re.findall(r"[a-z]{3,}", base)
    # Keep high-signal topical stems from the message / event label.
    stems = {
        tok
        for tok in raw_tokens
        if tok not in _TOPIC_STOPWORDS
        and tok
        in {
            "family", "planning", "implant", "implants", "contraceptive",
            "cervical", "cancer", "screening", "camp",
            "table", "banking", "loans", "saved",
            "menstrual", "hygiene", "pads", "sanitary",
            "gender", "violence", "dialogue", "counselor",
            "kitchen", "garden", "gardens", "compost", "seedlings", "farmers",
            "nutrition", "malaria", "maternal", "antenatal", "vaccination",
        }
    }
    if not stems:
        stems = {
            tok
            for tok in re.findall(r"[a-z]{3,}", activity_label.lower())
            if tok not in _TOPIC_STOPWORDS
        }
    return stems


def _extract_person_tokens(text: str, redactions: Sequence[dict[str, Any]]) -> set[str]:
    """Extract non-title name tokens from ``NAME`` redaction spans for duplicate matching."""
    names: set[str] = set()
    for span in redactions:
        if span.get("type") != "NAME":
            continue
        raw = text[int(span["start"]) : int(span["end"])]
        for tok in re.findall(r"[a-z]+", raw.lower()):
            if len(tok) >= 4 and tok not in _PERSON_TITLES:
                names.add(tok)
    return names


def _build_duplicate_features(
    messages: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    location_spans_per_msg: Sequence[Sequence[dict[str, Any]]],
    activity_labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Build per-message feature dicts used to score potential duplicate event reports."""
    features: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        text = str(msg["message"])
        rec = records[i]
        participants = rec.get("reported_count")
        p_int = participants if isinstance(participants, int) else None
        features.append(
            {
                "relevant": bool(rec.get("relevant")),
                "locs": _extract_location_keys(rec, text, location_spans_per_msg[i]),
                "participants": p_int,
                "sec_nums": _extract_secondary_numbers(text, p_int),
                "topic_tokens": _extract_topic_tokens(activity_labels[i], text),
                "person_tokens": _extract_person_tokens(text, rec.get("redactions", [])),
                "has_cue": bool(_DUPLICATE_CUE_RE.search(text)),
            }
        )
    return features


def _score_duplicate_pair(fi: dict[str, Any], fj: dict[str, Any]) -> float:
    """Score how likely message ``fj`` is a duplicate report of earlier message ``fi``."""
    loc_sim = _location_similarity(fi["locs"], fj["locs"])
    union_t = fi["topic_tokens"] | fj["topic_tokens"]
    topic_jac = len(fi["topic_tokens"] & fj["topic_tokens"]) / len(union_t) if union_t else 0.0

    # Reward matching places, participant counts, and topics; penalize clear mismatches.
    score = 0.0
    if loc_sim >= 0.78:
        score += 2.5 * loc_sim
    elif fi["locs"] and fj["locs"] and loc_sim < 0.45:
        score -= 2.5

    pi, pj = fi["participants"], fj["participants"]
    if pi is not None and pj is not None:
        score += 3.0 if pi == pj else -3.0

    if topic_jac >= 0.30:
        score += 3.0 * topic_jac
    elif union_t and topic_jac == 0.0:
        score -= 3.0

    # Shared secondary numbers (e.g., item counts) or a follow-up cue with a shared name
    # provide extra evidence that two workers are reporting the same event.
    if fi["sec_nums"] & fj["sec_nums"]:
        score += 1.5

    if fj["has_cue"] and (fi["person_tokens"] & fj["person_tokens"]):
        score += 1.5

    return score


def find_duplicate_events(
    messages: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    location_spans_per_msg: Sequence[Sequence[dict[str, Any]]],
    activity_labels: Sequence[str],
) -> list[str]:
    """Identify earlier canonical ``Message_id`` values when an event report duplicates another."""
    n = len(messages)
    duplicates = ["none"] * n
    num_ids = [str(parse_numeric_message_id(str(m["id"]))) for m in messages]
    features = _build_duplicate_features(messages, records, location_spans_per_msg, activity_labels)

    for j in range(n):
        fj = features[j]
        if not fj["relevant"]:
            continue
        best_i: int | None = None
        best_score = 0.0
        for i in range(j):
            if not features[i]["relevant"] or duplicates[i] != "none":
                continue
            score = _score_duplicate_pair(features[i], fj)
            if score >= 4.5 and score > best_score:
                best_score = score
                best_i = i

        if best_i is not None:
            duplicates[j] = num_ids[best_i]

    return duplicates


def build_csv_rows(
    messages: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    location_spans_per_msg: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, str]]:
    """Build ordered CSV rows matching ``Dandelion Hidden Test.pdf``."""
    activity_labels = [format_activity_label(rec) for rec in records]
    duplicates = find_duplicate_events(messages, records, location_spans_per_msg, activity_labels)

    rows: list[dict[str, str]] = []
    for msg, rec, loc_spans, act_label, dup in zip(
        messages, records, location_spans_per_msg, activity_labels, duplicates, strict=True
    ):
        num_id = str(parse_numeric_message_id(str(msg["id"])))
        relevant = bool(rec.get("relevant"))
        anonymized = anonymize_csv_text(str(msg["message"]), rec.get("redactions", []), loc_spans)
        count = rec.get("reported_count")
        rows.append(
            {
                "Message_id": num_id,
                "anonymized_text": anonymized,
                "relevance": "Event" if relevant else "Chit-chat",
                "activity_type": act_label if relevant else "",
                "participants": str(count) if (relevant and isinstance(count, int)) else "",
                "duplicate_of": dup,
            }
        )
    return rows


def write_outputs_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write UTF-8 CSV using exact column names from ``Dandelion Hidden Test.pdf``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


_MESSAGE_ID_KEYS = ("Message_id", "message_id")
_ANONYMIZED_TEXT_KEYS = ("anonymized_text", "expected_anonymized_text")
_PLACEHOLDERS = ("[NAME]", "[LOCATION]", "[PHONE]")
_WER_APOSTROPHES = re.compile("['\u2019]")
_WER_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _csv_field(row: dict[str, str], keys: str | tuple[str, ...]) -> str:
    """Return the first matching column value from a CSV row, or an empty string."""
    candidates = (keys,) if isinstance(keys, str) else keys
    for key in candidates:
        if key in row:
            return row[key] or ""
    return ""


def _norm_message_id(raw: str) -> str:
    """Normalize a message ID to its numeric string when possible, else lowercase text."""
    stripped = raw.strip()
    try:
        return str(parse_numeric_message_id(stripped))
    except ValueError:
        return stripped.casefold()


def _norm_dup_value(raw: str) -> str:
    """Normalize ``duplicate_of`` values, mapping blank/NA/0 variants to ``'none'``."""
    val = raw.strip().casefold()
    if val in {"", "none", "na", "n/a", "null", "0"}:
        return "none"
    return _norm_message_id(val)


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score from true/false positive and negative counts."""
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _norm_wer_text(text: str) -> str:
    """Lowercase text and strip apostrophes/punctuation for word error rate (WER) scoring."""
    lowered = text.lower()
    without_apostrophes = _WER_APOSTROPHES.sub("", lowered)
    return " ".join(_WER_NON_ALNUM.sub(" ", without_apostrophes).split())


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string dictionaries, stripping any UTF-8 BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {str(k): (v or "") for k, v in row.items() if k is not None}
            for row in reader
        ]


@dataclass
class _EvalCounts:
    """Running tally of classification, attribute, duplicate, and redaction evaluation counts."""

    rel_ok: int = 0
    tp_ev: int = 0
    fp_ev: int = 0
    fn_ev: int = 0
    attr_total: int = 0
    attr_ok: int = 0
    act_ok: int = 0
    part_ok: int = 0
    dup_ok: int = 0
    tp_dup: int = 0
    fp_dup: int = 0
    fn_dup: int = 0
    placeholder_pred_count: int = 0
    placeholder_expected_count: int = 0
    over_redaction_count: int = 0


def _score_event_row(counts: _EvalCounts, pred: dict[str, str], expected: dict[str, str]) -> None:
    """Update relevance, event, activity, and participant-count tallies for one CSV row."""
    p_rel = _csv_field(pred, "relevance").strip()
    e_rel = _csv_field(expected, "relevance").strip()
    if p_rel.casefold() == e_rel.casefold():
        counts.rel_ok += 1

    p_is_ev = p_rel.casefold() == "event"
    e_is_ev = e_rel.casefold() == "event"
    if p_is_ev and e_is_ev:
        counts.tp_ev += 1
    elif p_is_ev and not e_is_ev:
        counts.fp_ev += 1
    elif not p_is_ev and e_is_ev:
        counts.fn_ev += 1

    if not e_is_ev:
        return
    counts.attr_total += 1
    same_act = (
        _csv_field(pred, "activity_type").strip().casefold()
        == _csv_field(expected, "activity_type").strip().casefold()
    )
    same_part = (
        _csv_field(pred, "participants").strip() == _csv_field(expected, "participants").strip()
    )
    if same_act:
        counts.act_ok += 1
    if same_part:
        counts.part_ok += 1
    if same_act and same_part:
        counts.attr_ok += 1


def _score_duplicate_row(
    counts: _EvalCounts, pred: dict[str, str], expected: dict[str, str]
) -> None:
    """Update duplicate-event accuracy and TP/FP/FN tallies for one CSV row."""
    p_dup = _norm_dup_value(_csv_field(pred, "duplicate_of"))
    e_dup = _norm_dup_value(_csv_field(expected, "duplicate_of"))
    if p_dup == e_dup:
        counts.dup_ok += 1
    if p_dup != "none" and p_dup == e_dup:
        counts.tp_dup += 1
    elif p_dup != "none" and p_dup != e_dup:
        counts.fp_dup += 1
    elif e_dup != "none" and p_dup == "none":
        counts.fn_dup += 1


def _score_redaction_row(
    counts: _EvalCounts, pred: dict[str, str], expected: dict[str, str]
) -> None:
    """Count matched and extra ``[NAME]``, ``[LOCATION]``, and ``[PHONE]`` placeholders."""
    e_text = _csv_field(expected, _ANONYMIZED_TEXT_KEYS)
    p_text = _csv_field(pred, _ANONYMIZED_TEXT_KEYS)
    for ph in _PLACEHOLDERS:
        e_cnt = e_text.count(ph)
        p_cnt = p_text.count(ph)
        counts.placeholder_expected_count += e_cnt
        counts.placeholder_pred_count += min(p_cnt, e_cnt)
        if p_cnt > e_cnt:
            counts.over_redaction_count += p_cnt - e_cnt


def _summarize_eval_counts(counts: _EvalCounts, total_rows: int) -> dict[str, float]:
    """Convert raw evaluation tallies into percentage metrics and F1 scores."""
    n = total_rows or 1
    ev_p, ev_r, ev_f1 = _prf1(counts.tp_ev, counts.fp_ev, counts.fn_ev)
    _, _, dup_f1 = _prf1(counts.tp_dup, counts.fp_dup, counts.fn_dup)
    attr_total = counts.attr_total
    return {
        "classification_accuracy_pct": round(100.0 * counts.rel_ok / n, 2),
        "event_precision_pct": round(100.0 * ev_p, 2),
        "event_recall_pct": round(100.0 * ev_r, 2),
        "event_f1_pct": round(100.0 * ev_f1, 2),
        "ner_redaction_recall_pct": round(
            100.0 * counts.placeholder_pred_count / counts.placeholder_expected_count, 2
        )
        if counts.placeholder_expected_count
        else 100.0,
        "ner_over_redaction_count": float(counts.over_redaction_count),
        "attribute_exact_match_pct": (
            round(100.0 * counts.attr_ok / attr_total, 2) if attr_total else 100.0
        ),
        "activity_exact_match_pct": (
            round(100.0 * counts.act_ok / attr_total, 2) if attr_total else 100.0
        ),
        "participants_exact_match_pct": (
            round(100.0 * counts.part_ok / attr_total, 2) if attr_total else 100.0
        ),
        "duplication_f1_pct": round(100.0 * dup_f1, 2),
        "duplication_accuracy_pct": round(100.0 * counts.dup_ok / n, 2),
    }


def evaluate_csv_outputs(
    pred_rows: Sequence[dict[str, str]],
    expected_rows: Sequence[dict[str, str]],
) -> dict[str, float]:
    """Compute hidden-test evaluation metrics comparing predicted vs expected CSV rows."""
    expected_by_id = {
        _norm_message_id(_csv_field(r, _MESSAGE_ID_KEYS)): r for r in expected_rows
    }
    if len(pred_rows) != len(expected_rows):
        raise ValueError(
            f"row count mismatch: {len(pred_rows)} pred vs {len(expected_rows)} expected"
        )

    counts = _EvalCounts()
    for pred in pred_rows:
        mid = _norm_message_id(_csv_field(pred, _MESSAGE_ID_KEYS))
        expected = expected_by_id[mid]
        _score_event_row(counts, pred, expected)
        _score_duplicate_row(counts, pred, expected)
        _score_redaction_row(counts, pred, expected)

    return _summarize_eval_counts(counts, len(expected_rows))


def evaluate_transcript_wer(
    messages: Sequence[dict[str, Any]],
    expected_rows: Sequence[dict[str, str]],
) -> dict[str, float]:
    """Compute corpus-level raw and cleaned ASR WER against expected ``transcript`` rows."""
    expected_by_id = {
        _norm_message_id(_csv_field(r, _MESSAGE_ID_KEYS)): r for r in expected_rows
    }
    refs: list[str] = []
    raw_hyps: list[str] = []
    clean_hyps: list[str] = []

    for msg in messages:
        mid = _norm_message_id(str(msg["id"]))
        expected = expected_by_id.get(mid)
        if expected is None:
            continue
        ref = _norm_wer_text(_csv_field(expected, "transcript"))
        if not ref:
            continue
        refs.append(ref)
        raw_text = msg.get("asr_text")
        clean_text = msg.get("message")
        raw_hyps.append(_norm_wer_text(str(raw_text if raw_text is not None else clean_text or "")))
        clean_hyps.append(_norm_wer_text(str(clean_text or "")))

    if not refs:
        return {}
    raw_out = jiwer.process_words(refs, raw_hyps)
    clean_out = jiwer.process_words(refs, clean_hyps)
    return {
        "asr_raw_wer_pct": round(100.0 * float(raw_out.wer), 2),
        "asr_clean_wer_pct": round(100.0 * float(clean_out.wer), 2),
    }


def resolve_initial_prompt(
    initial_prompt: str | None, *, kenyan_prompt: bool = False
) -> str | None:
    """Resolve an explicit ASR initial prompt or optional built-in Kenyan CHW prompt."""
    if initial_prompt is not None:
        return initial_prompt
    if kenyan_prompt:
        return transcribe.KENYAN_CHW_INITIAL_PROMPT
    return None


def _load_validated_transcripts(transcripts_path: Path) -> list[dict[str, Any]]:
    """Load a transcripts JSON file and verify it matches the Interface Spec v2 schema."""
    loaded = json.loads(transcripts_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"expected a list of messages in {transcripts_path}")
    problems = contracts.validate_file("transcripts", loaded)
    if problems:
        raise ValueError("input transcripts failed validation:\n" + "\n".join(problems))
    return loaded


def _transcribe_hidden_audio(
    input_audio: Path,
    *,
    make_asr: Callable[..., transcribe.ASRBackend],
    resolved_model: str,
    device: str,
    batch_size: int,
    num_beams: int,
    initial_prompt: str | None,
    clean: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract/collect audio files, run Whisper ASR, and return messages with run metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        entries = collect_audio_files(input_audio, Path(tmp))
        asr_kw: dict[str, Any] = {}
        if initial_prompt is not None:
            asr_kw["initial_prompt"] = initial_prompt
        if num_beams != 1:
            asr_kw["num_beams"] = num_beams
        asr = make_asr(resolved_model, device, batch_size, **asr_kw)
        messages, audio_s, wall_s = transcribe_audio_entries(entries, asr, clean=clean)
        meta = transcribe.build_metadata(
            "hidden_test",
            asr.metadata(),
            len(messages),
            audio_s,
            wall_s,
            datetime.now(UTC),
        )
        return messages, meta


def _run_nlp_on_messages(
    messages: Sequence[dict[str, Any]], nlp_system: Any
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Run GLiNER extraction on each message and validate the resulting v2 records."""
    system = nlp_system if nlp_system is not None else gliner_system.GLiNERSystem()
    records: list[dict[str, Any]] = []
    location_spans_per_msg: list[list[dict[str, Any]]] = []
    for msg in messages:
        rec, loc_spans = system.process_message_detailed(msg)
        records.append(rec)
        location_spans_per_msg.append(loc_spans)

    problems = contracts.validate_file("records", records, list(messages))
    if problems:
        raise ValueError("pipeline produced invalid v2 records:\n" + "\n".join(problems))
    return records, location_spans_per_msg


def _write_hidden_artifacts(
    artifacts_dir: Path,
    messages: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None,
) -> None:
    """Save intermediate transcripts, NLP records, and optional ASR metadata JSON files."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    contracts.write_json(artifacts_dir / "predicted_transcripts.json", list(messages))
    contracts.write_json(artifacts_dir / "predicted_records.json", list(records))
    if meta is not None:
        (artifacts_dir / "predicted_transcripts.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def run_hidden_test_pipeline(
    *,
    input_audio: Path | None = None,
    transcripts_path: Path | None = None,
    out_csv: Path = DEFAULT_OUT_CSV,
    artifacts_dir: Path | None = None,
    clean: bool = True,
    make_asr: Callable[..., transcribe.ASRBackend] = transcribe.make_whisper,
    model_id: str | None = None,
    device: str = "auto",
    batch_size: int = transcribe.DEFAULT_BATCH_SIZE,
    num_beams: int = 1,
    initial_prompt: str | None = None,
    nlp_system: Any = None,
) -> list[dict[str, str]]:
    """Run audio->ASR->GLiNER->dedup->CSV end-to-end (or from existing transcripts JSON)."""
    if num_beams < 1:
        raise ValueError(f"num_beams must be >= 1, got {num_beams}")
    resolved_model = model_id or transcribe.resolve_default_model(device=device)
    meta: dict[str, Any] | None = None

    if transcripts_path is not None:
        messages = _load_validated_transcripts(transcripts_path)
    elif input_audio is not None:
        messages, meta = _transcribe_hidden_audio(
            input_audio,
            make_asr=make_asr,
            resolved_model=resolved_model,
            device=device,
            batch_size=batch_size,
            num_beams=num_beams,
            initial_prompt=initial_prompt,
            clean=clean,
        )
    else:
        raise ValueError("provide either input_audio or transcripts_path")

    records, location_spans_per_msg = _run_nlp_on_messages(messages, nlp_system)
    rows = build_csv_rows(messages, records, location_spans_per_msg)
    write_outputs_csv(out_csv, rows)
    if artifacts_dir is not None:
        _write_hidden_artifacts(artifacts_dir, messages, records, meta)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the hidden-test inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Dandelion hidden-test pipeline (Whisper + GLiNER + dedup -> CSV)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--transcripts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Optional expected CSV path to compute validation metrics.",
    )
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=transcribe.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--kenyan-prompt",
        action="store_true",
        help="Use the built-in generic Kenyan CHW / Swahili ASR initial prompt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the hidden-test pipeline CLI and return an exit code."""
    args = parse_args(argv)
    prompt = resolve_initial_prompt(args.initial_prompt, kenyan_prompt=args.kenyan_prompt)
    rows = run_hidden_test_pipeline(
        input_audio=None if args.transcripts is not None else args.input,
        transcripts_path=args.transcripts,
        out_csv=args.out,
        artifacts_dir=args.artifacts_dir,
        clean=not args.no_clean,
        model_id=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        initial_prompt=prompt,
    )
    print(f"Wrote {len(rows)} predicted rows -> {args.out}")
    if args.expected is not None:
        metrics = evaluate_csv_outputs(rows, load_csv_rows(args.expected))
        print("Validation metrics:", json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
