"""M4 - WER and downstream evaluator.

For each audio version (clean, noisy, phone) this module computes transcription accuracy
(WER), number and location accuracy at token level and field level, and the downstream
classification accuracy of the unmodified text baseline, compared with the baseline on the
original text of the same messages. Results go to ``evaluation/wer_results.json`` and the
raw baseline predictions to ``evaluation/baseline_predictions/<system>.json``.

Structure (design section 1.2):

* core functions (``normalize`` ... ``evaluate``) are pure apart from ``evaluate`` writing the
  baseline input files into the ``workdir`` it is given;
* ``evaluate`` receives the baseline as a ``run_baseline`` argument and never imports M3;
* ``main()`` is the CLI shell: it reads the files, wires in ``baseline_runner.run_baseline``
  (imported inside ``main``) and writes the outputs.

Rounding: ``evaluate`` returns unrounded values. ``round_results`` (which applies
``round_pct``) is the single place where percentages are rounded to 2 decimals; ``main``
calls it once, just before writing the file and printing the table.

Percentages whose denominator ``n`` is 0 are ``None`` (``null`` in JSON).
"""

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer

Record = dict[str, Any]
RunBaseline = Callable[[Path], list[Record]]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERSIONS: tuple[str, ...] = ("clean", "noisy", "phone")
SYSTEMS: tuple[str, ...] = ("original_text", *VERSIONS)
LOCATIONS: tuple[str, ...] = ("Nakuru", "Baringo", "Narok", "Kajiado")
NORMALIZATION_DESCRIPTION = "lowercase; delete apostrophes; other non-alphanumerics -> space"
PRED_FIELDS: tuple[str, ...] = ("relevant", "activity_type", "location", "reported_count")

WER_TARGET_PCT = 10.0
DROP_TARGET_PTS = 5.0

_APOSTROPHES = re.compile("['\N{RIGHT SINGLE QUOTATION MARK}]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


# --------------------------------------------------------------------------- core functions


def normalize(text: str) -> str:
    """Lowercase, delete apostrophes, turn other non-[a-z0-9] into spaces, collapse spaces."""
    lowered = text.lower()
    without_apostrophes = _APOSTROPHES.sub("", lowered)
    spaced = _NON_ALNUM.sub(" ", without_apostrophes)
    return " ".join(spaced.split())


def int_to_words(n: int) -> str:
    """Spell out 0-999 in English words without "and", e.g. 48 -> "forty eight"."""
    if not 0 <= n <= 999:
        raise ValueError(f"int_to_words supports 0-999, got {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"
    hundreds, rest = divmod(n, 100)
    head = f"{_ONES[hundreds]} hundred"
    return head if rest == 0 else f"{head} {int_to_words(rest)}"


def wer_stats(ref: str, hyp: str) -> dict[str, Any]:
    """Word-level error counts of hyp against ref, both normalized first.

    ``wer`` is a fraction (1.0 == 100 %). An empty (normalized) hypothesis is handled
    explicitly as all deletions. An empty reference raises ``ValueError``: WER is undefined.
    """
    ref_norm = normalize(ref)
    hyp_norm = normalize(hyp)
    ref_words = len(ref_norm.split())
    hyp_words = len(hyp_norm.split())
    if ref_words == 0:
        raise ValueError("reference is empty after normalization; WER is undefined")
    if hyp_words == 0:
        # Empty transcription counts every reference word as deleted (100% error).
        return {
            "wer": 1.0,
            "substitutions": 0,
            "deletions": ref_words,
            "insertions": 0,
            "hits": 0,
            "ref_words": ref_words,
            "hyp_words": 0,
        }
    out = jiwer.process_words(ref_norm, hyp_norm)
    return {
        "wer": float(out.wer),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
        "ref_words": ref_words,
        "hyp_words": hyp_words,
    }


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Corpus WER as a fraction: (S + D + I) / N summed over all pairs, on normalized text."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    if not refs:
        raise ValueError("corpus_wer needs at least one reference")
    refs_norm = [normalize(r) for r in refs]
    hyps_norm = [normalize(h) for h in hyps]
    if any(not r for r in refs_norm):
        raise ValueError("a reference is empty after normalization; WER is undefined")
    out = jiwer.process_words(refs_norm, hyps_norm)
    return float(out.wer)


def _contains_sequence(tokens: list[str], needle: list[str]) -> bool:
    """Return True if ``needle`` appears as a contiguous slice inside ``tokens``."""
    size = len(needle)
    return any(tokens[i : i + size] == needle for i in range(len(tokens) - size + 1))


def number_token_ok(ref: str, hyp: str) -> bool | None:
    """True if every integer in ref appears in hyp as the digit token or as English words.

    Returns ``None`` if the reference contains no number.
    """
    numbers = _DIGITS.findall(normalize(ref))
    if not numbers:
        return None
    hyp_tokens = normalize(hyp).split()
    for digits in numbers:
        value = int(digits)
        # Accept either digits ("48") or spoken words ("forty eight"), as ASR often spells numbers.
        if digits in hyp_tokens or str(value) in hyp_tokens:
            continue
        if value <= 999 and _contains_sequence(hyp_tokens, int_to_words(value).split()):
            continue
        return False
    return True


def reference_location(ref: str) -> str | None:
    """The first of LOCATIONS that appears as a token of the normalized reference."""
    tokens = normalize(ref).split()
    for location in LOCATIONS:
        if location.lower() in tokens:
            return location
    return None


def location_token_ok(
    ref: str, hyp: str, expected_location: str | None = None
) -> bool | None:
    """True if the expected location appears in the normalized hypothesis.

    Uses ``expected_location`` when provided; otherwise falls back to ``reference_location(ref)``.
    Returns ``None`` if neither provides a location.
    """
    location = reference_location(ref) if expected_location is None else expected_location
    if location is None:
        return None
    needle = normalize(location).split()
    if not needle:
        return None
    return _contains_sequence(normalize(hyp).split(), needle)


def classification_ok(pred: Record, gold: Record) -> bool:
    """The evaluation/evaluate.py rule: relevant and activity_type must both match."""
    return bool(
        pred["relevant"] == gold["relevant"] and pred["activity_type"] == gold["activity_type"]
    )


def round_pct(x: float) -> float:
    """Round a percentage to 2 decimal places (applied once, when results are written)."""
    return round(x, 2)


def pct(passed: int, n: int) -> float | None:
    """100 * passed / n, or None when n == 0."""
    return None if n == 0 else 100.0 * passed / n


def _ratio_pct(values: Sequence[bool | None]) -> tuple[float | None, int]:
    """Percentage of True among the non-None values, and that count n."""
    counted = [v for v in values if v is not None]
    return pct(sum(1 for v in counted if v), len(counted)), len(counted)


def _index_by_id(records: Sequence[Record], what: str) -> dict[str, Record]:
    """Build an ``id -> record`` lookup table, raising ValueError on duplicate IDs."""
    index: dict[str, Record] = {}
    for record in records:
        record_id = record["id"]
        if record_id in index:
            raise ValueError(f"{what}: duplicate id {record_id}")
        index[record_id] = record
    return index


def _check_id_set(found: set[str], wanted: set[str], what: str) -> None:
    """Verify that ``found`` IDs match ``wanted`` IDs exactly, or raise ValueError."""
    if found != wanted:
        missing = sorted(wanted - found)
        extra = sorted(found - wanted)
        raise ValueError(
            f"{what}: id set differs from the manifest's message ids "
            f"(missing: {missing}, extra: {extra})"
        )


def manifest_message_ids(manifest: Record) -> list[str]:
    """The sorted, unique message ids of the manifest's clips."""
    return sorted({clip["message_id"] for clip in manifest["clips"]})


def _write_json(path: Path, data: Any) -> None:
    """Write ``data`` as pretty-printed UTF-8 JSON to ``path``."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _system_summary(
    clips: list[Record], voices: list[str], original_acc: float | None
) -> dict[str, Any]:
    """Aggregate per-clip scores into overall and per-voice summary metrics for one version."""
    refs = [c["reference"] for c in clips]
    hyps = [c["hypothesis"] for c in clips]
    number_pct, number_n = _ratio_pct([c["number_token_ok"] for c in clips])
    location_pct, location_n = _ratio_pct([c["location_token_ok"] for c in clips])
    count_pct, count_n = _ratio_pct([c["count_field_ok"] for c in clips])
    loc_field_pct, loc_field_n = _ratio_pct([c["location_field_ok"] for c in clips])
    acc_pct, _ = _ratio_pct([c["classification_ok"] for c in clips])
    wer_pct = 100.0 * corpus_wer(refs, hyps)
    # Positive drop means downstream accuracy fell on ASR transcripts compared to original text.
    drop = None if original_acc is None or acc_pct is None else original_acc - acc_pct
    by_voice: dict[str, float | None] = {}
    for voice in voices:
        voice_clips = [c for c in clips if c["voice"] == voice]
        by_voice[voice] = (
            100.0
            * corpus_wer(
                [c["reference"] for c in voice_clips], [c["hypothesis"] for c in voice_clips]
            )
            if voice_clips
            else None
        )
    return {
        "wer_pct": wer_pct,
        "wer_mean_per_clip_pct": sum(c["wer_pct"] for c in clips) / len(clips),
        "number_token_acc_pct": number_pct,
        "number_token_n": number_n,
        "location_token_acc_pct": location_pct,
        "location_token_n": location_n,
        "count_field_acc_pct": count_pct,
        "count_field_n": count_n,
        "location_field_acc_pct": loc_field_pct,
        "location_field_n": loc_field_n,
        "baseline_acc_pct": acc_pct,
        "baseline_acc_drop_pts": drop,
        "false_positive_count": sum(
            1 for c in clips if not c["gold"]["relevant"] and c["pred"]["relevant"]
        ),
        "by_voice": by_voice,
        "targets": {
            "wer_lt_10": wer_pct < WER_TARGET_PCT,
            "drop_le_5": None if drop is None else drop <= DROP_TARGET_PTS,
        },
    }


def _index_and_validate_inputs(
    manifest: Record,
    transcripts: dict[str, list[Record]],
    messages: list[Record],
    expected: list[Record],
) -> tuple[
    list[str],
    dict[tuple[str, str], Record],
    dict[str, dict[str, Record]],
    dict[str, Record],
    dict[str, Record],
]:
    """Index all input files by ID and verify every message has all audio versions and labels."""
    ids = manifest_message_ids(manifest)
    id_set = set(ids)
    if not ids:
        raise ValueError("manifest has no clips")

    clip_index: dict[tuple[str, str], Record] = {}
    for clip in manifest["clips"]:
        clip_index[(clip["message_id"], clip["version"])] = clip
    for message_id in ids:
        for version in VERSIONS:
            if (message_id, version) not in clip_index:
                raise ValueError(f"manifest has no {version} clip for {message_id}")

    transcript_index: dict[str, dict[str, Record]] = {}
    for version in VERSIONS:
        if version not in transcripts:
            raise ValueError(f"no transcripts for version {version}")
        index = _index_by_id(transcripts[version], f"transcripts_{version}")
        _check_id_set(set(index), id_set, f"transcripts_{version}")
        transcript_index[version] = index

    message_index = _index_by_id(messages, "messages")
    missing_messages = sorted(id_set - set(message_index))
    if missing_messages:
        raise ValueError(f"messages: missing ids {missing_messages}")
    expected_index = _index_by_id(expected, "expected")
    missing_expected = sorted(id_set - set(expected_index))
    if missing_expected:
        raise ValueError(f"expected: missing ids {missing_expected}")

    return ids, clip_index, transcript_index, message_index, expected_index


def _run_baseline_systems(
    ids: list[str],
    message_index: dict[str, Record],
    transcript_index: dict[str, dict[str, Record]],
    run_baseline: RunBaseline,
    workdir: Path,
) -> dict[str, dict[str, Record]]:
    """Run the downstream baseline on original text and on each ASR version's transcripts."""
    id_set = set(ids)
    inputs: dict[str, list[Record]] = {
        "original_text": [
            {
                "id": i,
                "sender_id": message_index[i]["sender_id"],
                "message": message_index[i]["message"],
            }
            for i in ids
        ]
    }
    for version in VERSIONS:
        inputs[version] = [transcript_index[version][i] for i in ids]

    workdir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[str, Record]] = {}
    for system in SYSTEMS:
        input_path = workdir / f"{system}.json"
        _write_json(input_path, inputs[system])
        index = _index_by_id(run_baseline(input_path), f"baseline predictions ({system})")
        _check_id_set(set(index), id_set, f"baseline predictions ({system})")
        predictions[system] = index
    return predictions


def _gold_record(original_pred: Record, gold_entry: Record) -> Record:
    """Build the reference label record, falling back to original-text predictions if omitted."""
    # When expected outputs omit location or reported_count, use the original-text prediction
    # as the reference so we still measure how much ASR errors degrade field extraction.
    return {
        "relevant": gold_entry["relevant"],
        "activity_type": gold_entry["activity_type"],
        "location": (
            gold_entry["location"] if "location" in gold_entry else original_pred["location"]
        ),
        "reported_count": (
            gold_entry["reported_count"]
            if "reported_count" in gold_entry
            else original_pred["reported_count"]
        ),
    }


def _score_clip(
    message_id: str,
    version: str,
    clip: Record,
    transcript: Record,
    raw_pred: Record,
    gold: Record,
    gold_location: Any,
) -> Record:
    """Compute WER, token accuracy, and downstream field accuracy for a single audio clip."""
    reference = clip["reference_text"]
    asr_raw = transcript.get("asr_text")
    hypothesis = transcript["message"] if asr_raw is None else asr_raw
    stats = wer_stats(reference, hypothesis)
    pred = {field: raw_pred[field] for field in PRED_FIELDS}
    return {
        "clip_id": clip["clip_id"],
        "message_id": message_id,
        "version": version,
        "voice": clip["voice"],
        "reference": reference,
        "hypothesis": hypothesis,
        "reference_norm": normalize(reference),
        "hypothesis_norm": normalize(hypothesis),
        "wer_pct": 100.0 * stats["wer"],
        "substitutions": stats["substitutions"],
        "deletions": stats["deletions"],
        "insertions": stats["insertions"],
        "ref_words": stats["ref_words"],
        "hyp_words": stats["hyp_words"],
        "number_token_ok": number_token_ok(reference, hypothesis),
        "location_token_ok": location_token_ok(reference, hypothesis, gold_location),
        "pred": pred,
        "gold": dict(gold),
        "count_field_ok": (
            None
            if gold["reported_count"] is None
            else pred["reported_count"] == gold["reported_count"]
        ),
        "location_field_ok": (
            None if gold["location"] is None else pred["location"] == gold["location"]
        ),
        "classification_ok": classification_ok(pred, gold),
    }


def evaluate(
    manifest: Record,
    transcripts: dict[str, list[Record]],
    messages: list[Record],
    expected: list[Record],
    run_baseline: RunBaseline,
    workdir: Path,
    *,
    manifest_path: str = "examples/audio_manifest.json",
    asr_model: str | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Run the whole evaluation and return the section 6.7 structure, unrounded.

    The only I/O is writing ``<workdir>/<system>.json`` (the baseline inputs) and passing
    those paths to ``run_baseline``. Raises ``ValueError`` if any input's id set differs
    from the manifest's message ids, or a manifest clip is missing.
    """
    ids, clip_index, transcript_index, message_index, expected_index = (
        _index_and_validate_inputs(manifest, transcripts, messages, expected)
    )
    predictions = _run_baseline_systems(
        ids, message_index, transcript_index, run_baseline, workdir
    )

    original_acc, original_n = _ratio_pct(
        [
            classification_ok(
                predictions["original_text"][i],
                _gold_record(predictions["original_text"][i], expected_index[i]),
            )
            for i in ids
        ]
    )

    clips_out: list[Record] = []
    per_version: dict[str, list[Record]] = {v: [] for v in VERSIONS}
    for message_id in ids:
        gold = _gold_record(predictions["original_text"][message_id], expected_index[message_id])
        gold_location = expected_index[message_id].get("location")
        for version in VERSIONS:
            row = _score_clip(
                message_id,
                version,
                clip_index[(message_id, version)],
                transcript_index[version][message_id],
                predictions[version][message_id],
                gold,
                gold_location,
            )
            clips_out.append(row)
            per_version[version].append(row)

    voices = sorted({clip["voice"] for clip in manifest["clips"]})
    summary: dict[str, Any] = {"original_text": {"baseline_acc_pct": original_acc, "n": original_n}}
    for version in VERSIONS:
        summary[version] = _system_summary(per_version[version], voices, original_acc)

    return {
        "meta": {
            "created_utc": created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "manifest": manifest_path,
            "asr_model": asr_model,
            "normalization": NORMALIZATION_DESCRIPTION,
            "n_messages": len(ids),
        },
        "summary": summary,
        "clips": clips_out,
    }


def _round_value(key: str, value: Any) -> Any:
    """Recursively round percentage and point-difference fields to 2 decimal places."""
    if isinstance(value, dict):
        if key == "by_voice":
            return {k: None if v is None else round_pct(v) for k, v in value.items()}
        return {k: _round_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_value(key, v) for v in value]
    if isinstance(value, float) and key.endswith(("_pct", "_pts")):
        return round_pct(value)
    return value


def round_results(results: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``evaluate``'s output with every percentage rounded by round_pct.

    Rounded keys: every ``*_pct`` and ``*_pts`` float, and the ``by_voice`` values.
    ``None`` stays ``None``. Key order is kept. This is the only place rounding happens.
    """
    rounded: dict[str, Any] = {k: _round_value(k, v) for k, v in results.items()}
    return rounded


# --------------------------------------------------------------------------- CLI shell


def _fmt(value: Any) -> str:
    """Format a metric value for the summary table ('-' for None, 2 decimals for floats)."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_summary_table(summary: dict[str, Any]) -> str:
    """A plain-text table with one row per system, from the (already rounded) summary."""
    columns = (
        ("system", None),
        ("WER%", "wer_pct"),
        ("meanWER%", "wer_mean_per_clip_pct"),
        ("num_tok%", "number_token_acc_pct"),
        ("loc_tok%", "location_token_acc_pct"),
        ("count_fld%", "count_field_acc_pct"),
        ("loc_fld%", "location_field_acc_pct"),
        ("base_acc%", "baseline_acc_pct"),
        ("drop_pts", "baseline_acc_drop_pts"),
        ("FP", "false_positive_count"),
    )
    rows = [[title for title, _ in columns]]
    for system in SYSTEMS:
        values = summary[system]
        rows.append([system] + [_fmt(values.get(key)) for _, key in columns[1:] if key is not None])
    widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
    lines = [
        "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)
        )
        for row in rows
    ]
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _display_path(path: Path) -> str:
    """Repo-relative POSIX path when the path is inside the repo, else the path as given."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_asr_model(transcripts_dir: Path) -> str | None:
    """The ``model`` recorded in transcripts_<version>.meta.json files (None if absent).

    If the versions name different models, they are joined as "version: model" pairs.
    """
    models: dict[str, str] = {}
    for version in VERSIONS:
        meta_path = transcripts_dir / f"transcripts_{version}.meta.json"
        if meta_path.is_file():
            model = _load_json(meta_path).get("model")
            if isinstance(model, str):
                models[version] = model
    distinct = sorted(set(models.values()))
    if not distinct:
        return None
    if len(distinct) == 1:
        return distinct[0]
    return ", ".join(f"{version}: {model}" for version, model in models.items())


def load_transcripts(transcripts_dir: Path, message_ids: list[str]) -> dict[str, list[Record]]:
    """Read the three transcript files; ValueError if one is missing or its ids differ."""
    transcripts: dict[str, list[Record]] = {}
    for version in VERSIONS:
        path = transcripts_dir / f"transcripts_{version}.json"
        if not path.is_file():
            raise ValueError(f"transcript file not found: {path}")
        records: list[Record] = _load_json(path)
        found = [r["id"] for r in records]
        if len(found) != len(set(found)):
            raise ValueError(f"{path}: duplicate ids")
        _check_id_set(set(found), set(message_ids), str(path))
        transcripts[version] = records
    return transcripts


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for WER and downstream evaluation."""
    parser = argparse.ArgumentParser(description="Compute WER and downstream baseline metrics.")
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "examples" / "audio_manifest.json"
    )
    parser.add_argument("--transcripts-dir", type=Path, default=REPO_ROOT / "examples")
    parser.add_argument(
        "--messages", type=Path, default=REPO_ROOT / "examples" / "synthetic_messages.json"
    )
    parser.add_argument(
        "--expected", type=Path, default=REPO_ROOT / "examples" / "expected_outputs.json"
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="directory of precomputed Record files (original_text, clean, noisy, phone)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evaluation" / "wer_results.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Stops (SystemExit with a message) on any input error."""
    args = build_parser().parse_args(argv)
    from evaluation import baseline_runner  # CLI shell only: evaluate() itself never imports M3

    raw_predictions: dict[str, list[Record]] = {}

    def run_system(input_json: Path) -> list[Record]:
        """Load precomputed predictions or invoke the baseline runner for one system."""
        if args.predictions_dir is not None:
            pred_path = args.predictions_dir / f"{input_json.stem}.json"
            records: list[Record] = _load_json(pred_path)
            raw_predictions[input_json.stem] = records
            return records
        records = baseline_runner.run_baseline(input_json)
        raw_predictions[input_json.stem] = records
        return records

    try:
        manifest: Record = _load_json(args.manifest)
        transcripts = load_transcripts(args.transcripts_dir, manifest_message_ids(manifest))
        messages: list[Record] = _load_json(args.messages)
        expected: list[Record] = _load_json(args.expected)
        with tempfile.TemporaryDirectory() as tmp:
            results = evaluate(
                manifest,
                transcripts,
                messages,
                expected,
                run_system,
                Path(tmp),
                manifest_path=_display_path(args.manifest),
                asr_model=read_asr_model(args.transcripts_dir),
            )
    except (OSError, ValueError, KeyError, baseline_runner.BaselineError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    rounded = round_results(results)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, rounded)
    predictions_dir: Path | None = None
    if args.predictions_dir is None:
        predictions_dir = out_path.parent / "baseline_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for system in SYSTEMS:
            _write_json(predictions_dir / f"{system}.json", raw_predictions[system])

    print(format_summary_table(rounded["summary"]))
    print(f"\nResults written to {out_path}")
    if predictions_dir is not None:
        print(f"Baseline predictions written to {predictions_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
