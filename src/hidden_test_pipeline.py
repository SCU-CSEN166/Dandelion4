"""End-to-end hidden test pipeline: audio (.mp4/.wav/.zip) -> ASR -> GLiNER -> deduplicated CSV.

Produces:
- Validated Interface Spec v2 ``Message[]`` transcripts (raw ASR in ``asr_text``,
  normalized text in ``message``)
- Validated Interface Spec v2 ``Record[]`` extractions (with ``NAME``/``PHONE`` redactions and
  resolved ``[LOCATION]`` spans)
- ``hidden_test_outputs.csv`` matching the schema required by ``Dandelion Hidden Test.pdf``:
  ``Message_id,anonymized_text,relevance,activity_type,participants,duplicate_of``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from difflib import SequenceMatcher
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import clean_transcripts
    import contracts
    import gliner_system
    import transcribe
else:
    _pkg = f"{__package__}." if __package__ else ""
    clean_transcripts = import_module(f"{_pkg}clean_transcripts")
    contracts = import_module(f"{_pkg}contracts")
    gliner_system = import_module(f"{_pkg}gliner_system")
    transcribe = import_module(f"{_pkg}transcribe")

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
    tokens = [t for t in re.split(r"[^a-z0-9]+", place.lower()) if t and t not in _VENUE_STOPWORDS]
    return "".join(tokens)


def _extract_location_keys(
    record: dict[str, Any], text: str, loc_spans: Sequence[dict[str, Any]]
) -> set[str]:
    keys: set[str] = set()
    if isinstance(record.get("location"), str) and record["location"].strip():
        keys.add(record["location"].strip().lower())
    for span in loc_spans:
        raw = text[int(span["start"]) : int(span["end"])].strip().lower()
        if raw:
            keys.add(raw)
    return keys


def _location_similarity(keys_a: set[str], keys_b: set[str]) -> float:
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
    names: set[str] = set()
    for span in redactions:
        if span.get("type") != "NAME":
            continue
        raw = text[int(span["start"]) : int(span["end"])]
        for tok in re.findall(r"[a-z]+", raw.lower()):
            if len(tok) >= 4 and tok not in _PERSON_TITLES:
                names.add(tok)
    return names


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

    features: list[dict[str, Any]] = []
    for i in range(n):
        text = str(messages[i]["message"])
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

    for j in range(n):
        fj = features[j]
        if not fj["relevant"]:
            continue
        best_i: int | None = None
        best_score = 0.0
        for i in range(j):
            if not features[i]["relevant"] or duplicates[i] != "none":
                continue
            fi = features[i]
            loc_sim = _location_similarity(fi["locs"], fj["locs"])
            union_t = fi["topic_tokens"] | fj["topic_tokens"]
            topic_jac = (
                len(fi["topic_tokens"] & fj["topic_tokens"]) / len(union_t) if union_t else 0.0
            )

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

            if fi["sec_nums"] & fj["sec_nums"]:
                score += 1.5

            if fj["has_cue"] and (fi["person_tokens"] & fj["person_tokens"]):
                score += 1.5

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


def evaluate_csv_outputs(
    pred_rows: Sequence[dict[str, str]],
    gold_rows: Sequence[dict[str, str]],
) -> dict[str, float]:
    """Compute the four metrics recommended in ``Dandelion Hidden Test.pdf``."""
    gold_by_id = {str(r["Message_id"]): r for r in gold_rows}
    if len(pred_rows) != len(gold_rows):
        raise ValueError(f"row count mismatch: {len(pred_rows)} pred vs {len(gold_rows)} gold")

    rel_ok = 0
    attr_total = 0
    attr_ok = 0
    tp_dup = 0
    fp_dup = 0
    fn_dup = 0
    placeholder_pred_count = 0
    placeholder_gold_count = 0

    for pred in pred_rows:
        mid = str(pred["Message_id"])
        gold = gold_by_id[mid]
        if pred["relevance"] == gold["relevance"]:
            rel_ok += 1
        if gold["relevance"] == "Event":
            attr_total += 1
            if (
                pred["activity_type"].strip().casefold() == gold["activity_type"].strip().casefold()
                and pred["participants"].strip() == gold["participants"].strip()
            ):
                attr_ok += 1
        p_dup = pred["duplicate_of"].strip().lower()
        g_dup = gold["duplicate_of"].strip().lower()
        if p_dup != "none" and p_dup == g_dup:
            tp_dup += 1
        elif p_dup != "none" and p_dup != g_dup:
            fp_dup += 1
        elif g_dup != "none" and p_dup == "none":
            fn_dup += 1

        for ph in ("[NAME]", "[LOCATION]", "[PHONE]"):
            g_cnt = gold["anonymized_text"].count(ph)
            p_cnt = pred["anonymized_text"].count(ph)
            placeholder_gold_count += g_cnt
            placeholder_pred_count += min(p_cnt, g_cnt)

    n = len(gold_rows) or 1
    precision = tp_dup / (tp_dup + fp_dup) if (tp_dup + fp_dup) else 1.0
    recall = tp_dup / (tp_dup + fn_dup) if (tp_dup + fn_dup) else 1.0
    dup_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "classification_accuracy_pct": round(100.0 * rel_ok / n, 2),
        "ner_redaction_recall_pct": round(
            100.0 * placeholder_pred_count / placeholder_gold_count, 2
        )
        if placeholder_gold_count
        else 100.0,
        "attribute_exact_match_pct": (
            round(100.0 * attr_ok / attr_total, 2) if attr_total else 100.0
        ),
        "duplication_f1_pct": round(100.0 * dup_f1, 2),
    }


def run_hidden_test_pipeline(
    *,
    input_audio: Path | None = None,
    transcripts_path: Path | None = None,
    out_csv: Path = Path("hidden_test_outputs.csv"),
    artifacts_dir: Path | None = None,
    clean: bool = True,
    make_asr: Callable[..., transcribe.ASRBackend] = transcribe.make_whisper,
    model_id: str | None = None,
    device: str = "auto",
    batch_size: int = transcribe.DEFAULT_BATCH_SIZE,
    nlp_system: Any = None,
) -> list[dict[str, str]]:
    """Run audio->ASR->GLiNER->dedup->CSV end-to-end (or from existing transcripts JSON)."""
    resolved_model = model_id or transcribe.resolve_default_model()
    messages: list[dict[str, Any]]
    meta: dict[str, Any] | None = None

    if transcripts_path is not None:
        loaded = json.loads(transcripts_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"expected a list of messages in {transcripts_path}")
        problems = contracts.validate_file("transcripts", loaded)
        if problems:
            raise ValueError("input transcripts failed validation:\n" + "\n".join(problems))
        messages = loaded
    else:
        if input_audio is None:
            raise ValueError("provide either input_audio or transcripts_path")
        with tempfile.TemporaryDirectory() as tmp:
            entries = collect_audio_files(input_audio, Path(tmp))
            asr = make_asr(resolved_model, device, batch_size)
            messages, audio_s, wall_s = transcribe_audio_entries(entries, asr, clean=clean)
            meta = transcribe.build_metadata(
                "hidden_test",
                asr.metadata(),
                len(messages),
                audio_s,
                wall_s,
                datetime.now(UTC),
            )

    system = nlp_system if nlp_system is not None else gliner_system.GLiNERSystem()
    records: list[dict[str, Any]] = []
    location_spans_per_msg: list[list[dict[str, Any]]] = []
    for msg in messages:
        rec, loc_spans = system.process_message_detailed(msg)
        records.append(rec)
        location_spans_per_msg.append(loc_spans)

    problems = contracts.validate_file("records", records, messages)
    if problems:
        raise ValueError("pipeline produced invalid v2 records:\n" + "\n".join(problems))

    rows = build_csv_rows(messages, records, location_spans_per_msg)
    write_outputs_csv(out_csv, rows)

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        contracts.write_json(artifacts_dir / "hidden_test_transcripts.json", messages)
        contracts.write_json(artifacts_dir / "hidden_test_records.json", records)
        if meta is not None:
            (artifacts_dir / "hidden_test_transcripts.meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dandelion hidden-test pipeline (Whisper + GLiNER + dedup -> CSV)."
    )
    parser.add_argument("--input", type=Path, default=Path("dandelion_audio.zip"))
    parser.add_argument("--transcripts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("hidden_test_outputs.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=transcribe.DEFAULT_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_hidden_test_pipeline(
        input_audio=None if args.transcripts is not None else args.input,
        transcripts_path=args.transcripts,
        out_csv=args.out,
        artifacts_dir=args.artifacts_dir,
        clean=not args.no_clean,
        model_id=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Wrote {len(rows)} hidden test rows -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
