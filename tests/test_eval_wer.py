"""Tests for evaluation/eval_wer.py (M4): pure, plus tmp_path files and one real baseline run."""

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import baseline_runner, eval_wer

Record = dict[str, Any]

VOICE_A = "en-KE-AsiliaNeural"
VOICE_B = "en-KE-ChilembaNeural"

# Design section 3.2 (defined here on purpose; tests do not import make_audio).
SELECTED_REPORT_IDS = [
    "msg_001", "msg_002", "msg_003", "msg_004", "msg_005", "msg_006",
    "msg_007", "msg_008", "msg_010", "msg_011", "msg_012", "msg_013",
    "msg_015", "msg_016", "msg_017", "msg_018", "msg_019", "msg_021",
    "msg_023", "msg_024", "msg_025", "msg_028", "msg_029", "msg_030",
]  # fmt: skip
SELECTED_CHAT_IDS = ["msg_031", "msg_032", "msg_033", "msg_034", "msg_036", "msg_038"]
SELECTED_IDS = SELECTED_REPORT_IDS + SELECTED_CHAT_IDS


# --------------------------------------------------------------------------- helpers


def make_manifest(messages: list[Record], voices: list[str] | None = None) -> Record:
    """A section 2.4-shaped manifest for these messages (voices alternate by default)."""
    clips: list[Record] = []
    for i, message in enumerate(sorted(messages, key=lambda m: m["id"])):
        voice = voices[i] if voices is not None else [VOICE_A, VOICE_B][i % 2]
        for version in eval_wer.VERSIONS:
            clips.append(
                {
                    "clip_id": f"{message['id']}_{version}",
                    "message_id": message["id"],
                    "sender_id": message["sender_id"],
                    "version": version,
                    "voice": voice,
                    "path": f"examples/audio/{message['id']}_{version}.wav",
                    "sample_rate": 8000 if version == "phone" else 16000,
                    "duration_s": 3.0,
                    "reference_text": message["message"],
                    "snr_db_measured": 10.0 if version == "noisy" else None,
                    "noise_seed": 1 if version == "noisy" else None,
                }
            )
    return {"generator": {"script": "src/training/make_audio.py"}, "clips": clips}


def records(messages: list[Record], texts: dict[str, str] | None = None) -> list[Record]:
    """Transcript records, optionally replacing the text of some ids."""
    texts = texts or {}
    return [
        {"id": m["id"], "sender_id": m["sender_id"], "message": texts.get(m["id"], m["message"])}
        for m in messages
    ]


def pred(
    message_id: str,
    relevant: bool,
    activity: str | None,
    location: str | None,
    count: int | None,
) -> Record:
    return {
        "id": message_id,
        "relevant": relevant,
        "activity_type": activity,
        "location": location,
        "reported_count": count,
        "verification_status": "pending" if activity else "not_applicable",
    }


class FakeBaseline:
    """Returns handwritten predictions chosen by the input file's stem (the system name)."""

    def __init__(self, predictions: dict[str, list[Record]]) -> None:
        self.predictions = predictions
        self.calls: list[Path] = []
        self.inputs: dict[str, list[Record]] = {}

    def __call__(self, input_json: Path) -> list[Record]:
        self.calls.append(input_json)
        with input_json.open(encoding="utf-8") as handle:
            self.inputs[input_json.stem] = json.load(handle)
        return self.predictions[input_json.stem]


def rule_baseline(input_json: Path) -> list[Record]:
    """A tiny text-rule stand-in for baseline.py, used by the main() tests."""
    with input_json.open(encoding="utf-8") as handle:
        items: list[Record] = json.load(handle)
    out: list[Record] = []
    for item in items:
        text = item["message"]
        lower = text.lower()
        activity = "health_screening" if "screened" in lower else None
        location = next((loc for loc in eval_wer.LOCATIONS if loc in text), None)
        digits = [int(tok) for tok in lower.replace(".", " ").split() if tok.isdigit()]
        out.append(
            pred(item["id"], activity is not None, activity, location, next(iter(digits), None))
        )
    return out


# Four-message scenario used by the hand-computed evaluate test.
MESSAGES: list[Record] = [
    {"id": "m1", "sender_id": "w1", "message": "We screened 48 women in Nakuru."},
    {"id": "m2", "sender_id": "w2", "message": "We visited 7 households in Narok today."},
    {"id": "m3", "sender_id": "w3", "message": "Our team made 22 health referrals in Baringo."},
    {"id": "m4", "sender_id": "w4", "message": "Thank you for the update."},
]
EXPECTED: list[Record] = [
    {"id": "m1", "relevant": True, "activity_type": "health_screening"},
    {"id": "m2", "relevant": True, "activity_type": "household_outreach"},
    {"id": "m3", "relevant": True, "activity_type": "health_referral"},
    {"id": "m4", "relevant": False, "activity_type": None},
]
CORRECT_PREDS: list[Record] = [
    pred("m1", True, "health_screening", "Nakuru", 48),
    pred("m2", True, "household_outreach", "Narok", 7),
    pred("m3", True, "health_referral", "Baringo", None),
    pred("m4", False, None, None, None),
]
VOICES = [VOICE_A, VOICE_B, VOICE_A, VOICE_B]


def scenario_inputs() -> tuple[Record, dict[str, list[Record]], FakeBaseline]:
    manifest = make_manifest(MESSAGES, VOICES)
    transcripts = {
        "clean": records(MESSAGES),
        "noisy": records(
            MESSAGES,
            {
                "m1": "We screened forty-eight women in Nakuru.",
                "m2": "We visited 7 households in Narrow today.",
            },
        ),
        "phone": records(MESSAGES, {"m4": ""}),
    }
    noisy_preds = [
        pred("m1", True, "health_screening", "Nakuru", None),
        pred("m2", True, "household_outreach", None, 7),
        pred("m3", True, "health_referral", "Baringo", None),
        pred("m4", True, "health_referral", None, None),
    ]
    fake = FakeBaseline(
        {
            "original_text": CORRECT_PREDS,
            "clean": CORRECT_PREDS,
            "noisy": noisy_preds,
            "phone": CORRECT_PREDS,
        }
    )
    return manifest, transcripts, fake


def run_scenario(tmp_path: Path) -> tuple[Record, FakeBaseline]:
    manifest, transcripts, fake = scenario_inputs()
    result = eval_wer.evaluate(
        manifest, transcripts, MESSAGES, EXPECTED, fake, tmp_path / "work", created_utc="T"
    )
    return result, fake


# --------------------------------------------------------------------------- normalize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Today's workshop", "todays workshop"),
        ("Today\N{RIGHT SINGLE QUOTATION MARK}s workshop", "todays workshop"),
        ("forty-eight.", "forty eight"),
        ("  We   screened\t48 \n women  ", "we screened 48 women"),
        ("Hello, World!", "hello world"),
        ("", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert eval_wer.normalize(text) == expected


# --------------------------------------------------------------------------- int_to_words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (6, "six"),
        (13, "thirteen"),
        (40, "forty"),
        (48, "forty eight"),
        (100, "one hundred"),
        (305, "three hundred five"),
        (999, "nine hundred ninety nine"),
    ],
)
def test_int_to_words(n: int, words: str) -> None:
    assert eval_wer.int_to_words(n) == words


@pytest.mark.parametrize("n", [-1, 1000])
def test_int_to_words_out_of_range(n: int) -> None:
    with pytest.raises(ValueError, match="0-999"):
        eval_wer.int_to_words(n)


# --------------------------------------------------------------------------- wer_stats / corpus


def test_wer_stats_substitution_deletion_insertion() -> None:
    # ref: a b c d ; hyp: a x c d e  -> 1 substitution, 1 insertion
    stats = eval_wer.wer_stats("A b, c d.", "a x c d e")
    assert stats == {
        "wer": pytest.approx(0.5),
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
        "hits": 3,
        "ref_words": 4,
        "hyp_words": 5,
    }
    deletion = eval_wer.wer_stats("we screened 48 women", "we screened women")
    assert (deletion["substitutions"], deletion["deletions"], deletion["insertions"]) == (0, 1, 0)
    assert deletion["wer"] == pytest.approx(0.25)


@pytest.mark.parametrize("hyp", ["", "   ", "?!"])
def test_wer_stats_empty_hypothesis_is_all_deletions(hyp: str) -> None:
    stats = eval_wer.wer_stats("We screened 48 women.", hyp)
    assert stats == {
        "wer": 1.0,
        "substitutions": 0,
        "deletions": 4,
        "insertions": 0,
        "hits": 0,
        "ref_words": 4,
        "hyp_words": 0,
    }


def test_wer_stats_empty_reference_raises() -> None:
    with pytest.raises(ValueError, match="reference is empty"):
        eval_wer.wer_stats("...", "words")


def test_corpus_wer_differs_from_mean_per_clip() -> None:
    refs = ["a", "b c d e f g h i j"]  # 1 word and 9 words
    hyps = ["x", "b c d e f g h i j"]  # 1 error on the short clip only
    corpus = eval_wer.corpus_wer(refs, hyps)
    mean = sum(eval_wer.wer_stats(r, h)["wer"] for r, h in zip(refs, hyps, strict=True)) / 2
    assert corpus == pytest.approx(0.1)
    assert mean == pytest.approx(0.5)
    assert corpus != pytest.approx(mean)


def test_corpus_wer_normalizes_and_handles_empty_hypothesis() -> None:
    assert eval_wer.corpus_wer(["Today's workshop."], ["todays WORKSHOP"]) == 0.0
    assert eval_wer.corpus_wer(["a b", "c"], ["", "c"]) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("refs", "hyps", "match"),
    [
        (["a"], [], "1 references but 0"),
        ([], [], "at least one"),
        (["a", "!"], ["a", "b"], "empty after normalization"),
    ],
)
def test_corpus_wer_errors(refs: list[str], hyps: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        eval_wer.corpus_wer(refs, hyps)


# --------------------------------------------------------------------------- token checks


@pytest.mark.parametrize(
    ("hyp", "ok"),
    [
        ("We screened 48 women in Nakuru.", True),
        ("We screened forty eight women in Nakuru.", True),
        ("We screened forty-eight women in Nakuru.", True),
        ("We screened 40 eight women in Nakuru.", False),
        ("We screened 84 women in Nakuru.", False),
        ("We screened women in Nakuru.", False),
    ],
)
def test_number_token_ok(hyp: str, ok: bool) -> None:
    assert eval_wer.number_token_ok("We screened 48 women in Nakuru.", hyp) is ok


def test_number_token_ok_needs_every_number_and_handles_large_numbers() -> None:
    ref = "We met 6 groups and 1200 people."
    assert eval_wer.number_token_ok(ref, "we met six groups and 1200 people") is True
    assert eval_wer.number_token_ok(ref, "we met six groups and twelve hundred people") is False
    assert eval_wer.number_token_ok("Visit 07 homes", "visit 7 homes") is True


def test_number_token_ok_none_without_number() -> None:
    assert eval_wer.number_token_ok("Good morning everyone!", "good morning 5") is None


@pytest.mark.parametrize(
    ("hyp", "ok"),
    [
        ("We screened 48 women in Nakuru", True),
        ("we screened 48 women in nakuru.", True),
        ("We screened 48 women in Nakurus", False),
        ("We screened 48 women in Na kuru", False),
    ],
)
def test_location_token_ok(hyp: str, ok: bool) -> None:
    assert eval_wer.location_token_ok("We screened 48 women in Nakuru.", hyp) is ok


def test_location_token_ok_none_without_location() -> None:
    assert eval_wer.location_token_ok("Thank you for the update.", "nakuru") is None
    assert eval_wer.reference_location("Kajiado and Narok") == "Narok"


# --------------------------------------------------------------------------- small helpers


def test_classification_ok() -> None:
    gold = {"relevant": True, "activity_type": "health_screening", "location": "Nakuru"}
    assert eval_wer.classification_ok(
        {"relevant": True, "activity_type": "health_screening", "location": None}, gold
    )
    assert not eval_wer.classification_ok({"relevant": True, "activity_type": None}, gold)
    assert not eval_wer.classification_ok(
        {"relevant": False, "activity_type": "health_screening"}, gold
    )


def test_round_pct_and_pct() -> None:
    assert eval_wer.round_pct(100 / 3) == 33.33
    assert eval_wer.round_pct(200 / 3) == 66.67
    assert eval_wer.pct(1, 3) == pytest.approx(100 / 3)
    assert eval_wer.pct(0, 0) is None


def test_manifest_message_ids_sorted_unique() -> None:
    manifest = make_manifest(list(reversed(MESSAGES)))
    assert eval_wer.manifest_message_ids(manifest) == ["m1", "m2", "m3", "m4"]


# --------------------------------------------------------------------------- evaluate


def test_evaluate_hand_computed_values(tmp_path: Path) -> None:
    result, fake = run_scenario(tmp_path)
    assert list(result) == ["meta", "summary", "clips"]
    assert result["meta"] == {
        "created_utc": "T",
        "manifest": "examples/audio_manifest.json",
        "asr_model": None,
        "normalization": eval_wer.NORMALIZATION_DESCRIPTION,
        "n_messages": 4,
    }
    summary = result["summary"]
    assert list(summary) == ["original_text", "clean", "noisy", "phone"]
    assert summary["original_text"] == {"baseline_acc_pct": 100.0, "n": 4}

    noisy = summary["noisy"]
    assert list(noisy) == [
        "wer_pct",
        "wer_mean_per_clip_pct",
        "number_token_acc_pct",
        "number_token_n",
        "location_token_acc_pct",
        "location_token_n",
        "count_field_acc_pct",
        "count_field_n",
        "location_field_acc_pct",
        "location_field_n",
        "baseline_acc_pct",
        "baseline_acc_drop_pts",
        "false_positive_count",
        "by_voice",
        "targets",
    ]
    # m1: "48" -> "forty eight" = 1 sub + 1 ins over 6; m2: 1 sub over 7; m3 8 words; m4 5.
    assert noisy["wer_pct"] == pytest.approx(100 * 3 / 26)
    assert noisy["wer_mean_per_clip_pct"] == pytest.approx(100 * (2 / 6 + 1 / 7) / 4)
    assert noisy["number_token_acc_pct"] == pytest.approx(100.0)
    assert noisy["number_token_n"] == 3
    assert noisy["location_token_acc_pct"] == pytest.approx(200 / 3)
    assert noisy["location_token_n"] == 3
    # Field gold comes from the original-text predictions: counts 48, 7 (m3 is null).
    assert noisy["count_field_acc_pct"] == pytest.approx(50.0)
    assert noisy["count_field_n"] == 2
    assert noisy["location_field_acc_pct"] == pytest.approx(200 / 3)
    assert noisy["location_field_n"] == 3
    assert noisy["baseline_acc_pct"] == pytest.approx(75.0)
    assert noisy["baseline_acc_drop_pts"] == pytest.approx(25.0)
    assert noisy["false_positive_count"] == 1
    assert noisy["by_voice"] == {
        VOICE_A: pytest.approx(100 * 2 / 14),
        VOICE_B: pytest.approx(100 * 1 / 12),
    }
    assert noisy["targets"] == {"wer_lt_10": False, "drop_le_5": False}

    clean = summary["clean"]
    assert clean["wer_pct"] == 0.0
    assert clean["count_field_acc_pct"] == 100.0
    assert clean["baseline_acc_drop_pts"] == 0.0
    assert clean["false_positive_count"] == 0
    assert clean["targets"] == {"wer_lt_10": True, "drop_le_5": True}

    phone = summary["phone"]
    assert phone["wer_pct"] == pytest.approx(100 * 5 / 26)
    assert phone["targets"] == {"wer_lt_10": False, "drop_le_5": True}

    # Baseline input files were written into workdir only, one per system.
    assert [p.name for p in fake.calls] == [f"{s}.json" for s in eval_wer.SYSTEMS]
    assert all(p.parent == tmp_path / "work" for p in fake.calls)
    assert fake.inputs["original_text"] == records(MESSAGES)
    assert fake.inputs["noisy"][1]["message"] == "We visited 7 households in Narrow today."


def test_evaluate_clip_rows(tmp_path: Path) -> None:
    result, _ = run_scenario(tmp_path)
    clips = result["clips"]
    assert len(clips) == 12
    assert [c["clip_id"] for c in clips[:3]] == ["m1_clean", "m1_noisy", "m1_phone"]
    m1_noisy = clips[1]
    assert list(m1_noisy) == [
        "clip_id",
        "message_id",
        "version",
        "voice",
        "reference",
        "hypothesis",
        "reference_norm",
        "hypothesis_norm",
        "wer_pct",
        "substitutions",
        "deletions",
        "insertions",
        "ref_words",
        "hyp_words",
        "number_token_ok",
        "location_token_ok",
        "pred",
        "gold",
        "count_field_ok",
        "location_field_ok",
        "classification_ok",
    ]
    assert m1_noisy["hypothesis_norm"] == "we screened forty eight women in nakuru"
    assert m1_noisy["wer_pct"] == pytest.approx(100 * 2 / 6)
    assert (m1_noisy["substitutions"], m1_noisy["insertions"]) == (1, 1)
    assert m1_noisy["number_token_ok"] is True
    assert m1_noisy["count_field_ok"] is False
    assert m1_noisy["gold"] == {
        "relevant": True,
        "activity_type": "health_screening",
        "location": "Nakuru",
        "reported_count": 48,
    }
    assert list(m1_noisy["pred"]) == ["relevant", "activity_type", "location", "reported_count"]
    m3_clean = clips[6]
    assert m3_clean["count_field_ok"] is None
    m4_phone = clips[11]
    assert m4_phone["hypothesis"] == ""
    assert (m4_phone["wer_pct"], m4_phone["deletions"], m4_phone["hyp_words"]) == (100.0, 5, 0)
    assert m4_phone["number_token_ok"] is None
    assert m4_phone["location_token_ok"] is None
    assert m4_phone["location_field_ok"] is None
    m4_noisy = clips[10]
    assert m4_noisy["classification_ok"] is False


def test_evaluate_n_zero_gives_none(tmp_path: Path) -> None:
    msgs = [MESSAGES[3]]
    manifest = make_manifest(msgs)
    preds = [CORRECT_PREDS[3]]
    fake = FakeBaseline({s: preds for s in eval_wer.SYSTEMS})
    result = eval_wer.evaluate(
        manifest,
        {v: records(msgs) for v in eval_wer.VERSIONS},
        msgs,
        [EXPECTED[3]],
        fake,
        tmp_path,
    )
    clean = result["summary"]["clean"]
    assert clean["number_token_acc_pct"] is None
    assert clean["number_token_n"] == 0
    assert clean["count_field_acc_pct"] is None
    assert clean["count_field_n"] == 0
    assert clean["by_voice"] == {VOICE_A: 0.0}
    assert result["meta"]["created_utc"].endswith("Z")
    rounded = eval_wer.round_results(result)
    assert rounded["summary"]["clean"]["location_field_acc_pct"] is None


def test_evaluate_by_voice_none_for_voice_without_clips() -> None:
    clips = [
        {
            "reference": "a b",
            "hypothesis": "a b",
            "voice": VOICE_A,
            "wer_pct": 0.0,
            "number_token_ok": None,
            "location_token_ok": None,
            "count_field_ok": None,
            "location_field_ok": None,
            "classification_ok": True,
            "gold": {"relevant": True},
            "pred": {"relevant": True},
        }
    ]
    summary = eval_wer._system_summary(clips, [VOICE_A, VOICE_B], None)
    assert summary["by_voice"] == {VOICE_A: 0.0, VOICE_B: None}
    assert summary["baseline_acc_drop_pts"] is None
    assert summary["targets"]["drop_le_5"] is None


def test_rounding_happens_only_at_output(tmp_path: Path) -> None:
    # original 2/3 correct, clean 1/3 correct: drop = 33.333...
    # Rounding each step would give 66.67 - 33.33 = 33.34; rounding once gives 33.33.
    msgs = MESSAGES[:3]
    wrong = [pred(m["id"], False, None, None, None) for m in msgs]
    fake = FakeBaseline(
        {
            "original_text": [*CORRECT_PREDS[:2], wrong[2]],
            "clean": [CORRECT_PREDS[0], *wrong[1:]],
            "noisy": CORRECT_PREDS[:3],
            "phone": CORRECT_PREDS[:3],
        }
    )
    result = eval_wer.evaluate(
        make_manifest(msgs),
        {v: records(msgs) for v in eval_wer.VERSIONS},
        msgs,
        EXPECTED[:3],
        fake,
        tmp_path,
    )
    raw = result["summary"]["clean"]["baseline_acc_drop_pts"]
    assert raw == pytest.approx(100 / 3)
    assert raw != 33.33  # evaluate() itself does not round
    naive = eval_wer.round_pct(200 / 3) - eval_wer.round_pct(100 / 3)
    assert naive == pytest.approx(33.34)
    rounded = eval_wer.round_results(result)
    assert rounded["summary"]["clean"]["baseline_acc_drop_pts"] == 33.33
    assert rounded["summary"]["original_text"]["baseline_acc_pct"] == 66.67
    # Negative drop (audio did better): 66.666... - 100 -> -33.33, not 66.67 - 100 = -33.33 by luck.
    assert rounded["summary"]["noisy"]["baseline_acc_drop_pts"] == -33.33


def test_round_results_rounds_every_percentage(tmp_path: Path) -> None:
    result, _ = run_scenario(tmp_path)
    rounded = eval_wer.round_results(result)
    noisy = rounded["summary"]["noisy"]
    assert noisy["wer_pct"] == 11.54
    assert noisy["wer_mean_per_clip_pct"] == 11.9
    assert noisy["location_field_acc_pct"] == 66.67
    assert noisy["by_voice"] == {VOICE_A: 14.29, VOICE_B: 8.33}
    assert noisy["count_field_n"] == 2
    assert noisy["targets"] == {"wer_lt_10": False, "drop_le_5": False}
    assert rounded["clips"][1]["wer_pct"] == 33.33
    assert rounded["clips"][1]["substitutions"] == 1
    assert list(rounded) == ["meta", "summary", "clips"]
    # The input structure is not modified.
    assert result["summary"]["noisy"]["wer_pct"] == pytest.approx(100 * 3 / 26)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("transcript_missing_id", r"transcripts_clean: id set differs.*missing: \['m4'\]"),
        ("transcript_extra_id", r"transcripts_noisy: id set differs.*extra: \['m9'\]"),
        ("transcript_duplicate_id", "transcripts_phone: duplicate id m1"),
        ("transcript_version_missing", "no transcripts for version phone"),
        ("manifest_clip_missing", "no noisy clip for m2"),
        ("manifest_empty", "manifest has no clips"),
        ("messages_missing_id", r"messages: missing ids \['m3'\]"),
        ("expected_missing_id", r"expected: missing ids \['m2'\]"),
        ("baseline_missing_id", r"baseline predictions \(clean\): id set differs"),
    ],
)
def test_evaluate_errors(tmp_path: Path, mutate: str, match: str) -> None:
    manifest, transcripts, fake = scenario_inputs()
    messages = list(MESSAGES)
    expected = list(EXPECTED)
    if mutate == "transcript_missing_id":
        transcripts["clean"] = transcripts["clean"][:3]
    elif mutate == "transcript_extra_id":
        transcripts["noisy"].append({"id": "m9", "sender_id": "w9", "message": "x"})
    elif mutate == "transcript_duplicate_id":
        transcripts["phone"].append(dict(transcripts["phone"][0]))
    elif mutate == "transcript_version_missing":
        del transcripts["phone"]
    elif mutate == "manifest_clip_missing":
        manifest["clips"] = [c for c in manifest["clips"] if c["clip_id"] != "m2_noisy"]
    elif mutate == "manifest_empty":
        manifest["clips"] = []
    elif mutate == "messages_missing_id":
        messages = [m for m in messages if m["id"] != "m3"]
    elif mutate == "expected_missing_id":
        expected = [e for e in expected if e["id"] != "m2"]
    elif mutate == "baseline_missing_id":
        fake.predictions["clean"] = CORRECT_PREDS[:3]
    with pytest.raises(ValueError, match=match):
        eval_wer.evaluate(manifest, transcripts, messages, expected, fake, tmp_path)


def test_original_text_scores_90_with_real_baseline(
    tmp_path: Path, messages: list[dict[str, str]], repo_root: Path
) -> None:
    """Design section 6.7: the current baseline scores 27/30 on the selected original text."""
    with (repo_root / "examples" / "expected_outputs.json").open(encoding="utf-8") as handle:
        expected: list[Record] = json.load(handle)
    by_id = {m["id"]: dict(m) for m in messages}
    selected: list[Record] = [by_id[i] for i in SELECTED_IDS]
    assert len(set(SELECTED_IDS)) == 30
    result = eval_wer.evaluate(
        make_manifest(selected),
        {v: records(selected) for v in eval_wer.VERSIONS},
        list(messages),
        expected,
        baseline_runner.run_baseline,
        tmp_path,
    )
    rounded = eval_wer.round_results(result)
    assert rounded["summary"]["original_text"] == {"baseline_acc_pct": 90.0, "n": 30}
    wrong = sorted({c["message_id"] for c in result["clips"] if not c["classification_ok"]})
    assert wrong == ["msg_008", "msg_012", "msg_025"]
    clean = rounded["summary"]["clean"]
    assert clean["wer_pct"] == 0.0
    assert clean["baseline_acc_pct"] == 90.0
    assert clean["baseline_acc_drop_pts"] == 0.0
    assert clean["number_token_n"] == 24
    assert clean["location_token_n"] == 24
    assert clean["count_field_n"] == 21  # 3 "health referrals" + 6 small talk have no count
    assert clean["location_field_n"] == 24
    assert clean["false_positive_count"] == 0


# --------------------------------------------------------------------------- read_asr_model


def test_read_asr_model(tmp_path: Path) -> None:
    assert eval_wer.read_asr_model(tmp_path) is None
    for version in eval_wer.VERSIONS:
        (tmp_path / f"transcripts_{version}.meta.json").write_text(
            json.dumps({"model": "openai/whisper-large-v3-turbo"}), encoding="utf-8"
        )
    assert eval_wer.read_asr_model(tmp_path) == "openai/whisper-large-v3-turbo"
    (tmp_path / "transcripts_phone.meta.json").write_text(
        json.dumps({"model": "openai/whisper-small"}), encoding="utf-8"
    )
    assert eval_wer.read_asr_model(tmp_path) == (
        "clean: openai/whisper-large-v3-turbo, noisy: openai/whisper-large-v3-turbo, "
        "phone: openai/whisper-small"
    )


# --------------------------------------------------------------------------- main()


def write_cli_inputs(tmp_path: Path, with_meta: bool = True) -> list[str]:
    def dump(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    msgs = MESSAGES[:2] + MESSAGES[3:]
    dump(tmp_path / "manifest.json", make_manifest(msgs))
    dump(tmp_path / "messages.json", MESSAGES)
    dump(tmp_path / "expected.json", EXPECTED)
    for version in eval_wer.VERSIONS:
        dump(tmp_path / f"transcripts_{version}.json", records(msgs))
        if with_meta:
            dump(tmp_path / f"transcripts_{version}.meta.json", {"model": "fake/asr"})
    return [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--transcripts-dir",
        str(tmp_path),
        "--messages",
        str(tmp_path / "messages.json"),
        "--expected",
        str(tmp_path / "expected.json"),
        "--out",
        str(tmp_path / "out" / "wer_results.json"),
    ]


def test_main_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(baseline_runner, "run_baseline", rule_baseline)
    argv = write_cli_inputs(tmp_path)
    assert eval_wer.main(argv) == 0

    out = tmp_path / "out" / "wer_results.json"
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert list(data) == ["meta", "summary", "clips"]
    assert data["meta"]["asr_model"] == "fake/asr"
    assert data["meta"]["manifest"] == str(tmp_path / "manifest.json")
    assert data["meta"]["n_messages"] == 3
    assert len(data["clips"]) == 9
    # rule_baseline: only m1 is classified correctly, m4 too (not relevant); m2 is wrong.
    assert data["summary"]["original_text"] == {"baseline_acc_pct": 66.67, "n": 3}
    assert data["summary"]["clean"]["wer_pct"] == 0.0

    pred_dir = tmp_path / "out" / "baseline_predictions"
    assert sorted(p.name for p in pred_dir.iterdir()) == sorted(
        f"{s}.json" for s in eval_wer.SYSTEMS
    )
    original = json.loads((pred_dir / "original_text.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in original] == ["m1", "m2", "m4"]
    assert "verification_status" in original[0]

    printed = capsys.readouterr().out
    assert "original_text" in printed
    assert "66.67" in printed
    assert "phone" in printed
    assert "Results written to" in printed


def test_main_without_meta_files_records_null_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline_runner, "run_baseline", rule_baseline)
    assert eval_wer.main(write_cli_inputs(tmp_path, with_meta=False)) == 0
    data = json.loads((tmp_path / "out" / "wer_results.json").read_text(encoding="utf-8"))
    assert data["meta"]["asr_model"] is None


def test_main_missing_transcript_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baseline_runner, "run_baseline", rule_baseline)
    argv = write_cli_inputs(tmp_path)
    (tmp_path / "transcripts_noisy.json").unlink()
    with pytest.raises(SystemExit, match="transcript file not found"):
        eval_wer.main(argv)
    assert not (tmp_path / "out").exists()


def test_main_id_set_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baseline_runner, "run_baseline", rule_baseline)
    argv = write_cli_inputs(tmp_path)
    (tmp_path / "transcripts_phone.json").write_text(
        json.dumps(records(MESSAGES[:2])), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match=r"id set differs.*missing: \['m4'\]"):
        eval_wer.main(argv)


def test_main_duplicate_transcript_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baseline_runner, "run_baseline", rule_baseline)
    argv = write_cli_inputs(tmp_path)
    recs = records(MESSAGES[:2] + MESSAGES[3:])
    (tmp_path / "transcripts_clean.json").write_text(json.dumps([*recs, recs[0]]), encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate ids"):
        eval_wer.main(argv)


def test_main_missing_manifest(tmp_path: Path) -> None:
    argv = write_cli_inputs(tmp_path)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="error:"):
        eval_wer.main(argv)


def test_main_baseline_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(input_json: Path) -> list[Record]:
        raise baseline_runner.BaselineError("baseline exited with code 1", "Traceback ...")

    monkeypatch.setattr(baseline_runner, "run_baseline", broken)
    with pytest.raises(SystemExit, match="baseline exited with code 1"):
        eval_wer.main(write_cli_inputs(tmp_path))


def test_display_path_inside_repo() -> None:
    path = eval_wer.REPO_ROOT / "examples" / "audio_manifest.json"
    assert eval_wer._display_path(path) == "examples/audio_manifest.json"


def test_format_summary_table_uses_two_decimals_and_dash_for_none() -> None:
    audio = {
        "wer_pct": 1.5,
        "wer_mean_per_clip_pct": 2.0,
        "number_token_acc_pct": None,
        "location_token_acc_pct": 100.0,
        "count_field_acc_pct": 33.33,
        "location_field_acc_pct": 0.0,
        "baseline_acc_pct": 90.0,
        "baseline_acc_drop_pts": -3.33,
        "false_positive_count": 2,
    }
    table = eval_wer.format_summary_table(
        {"original_text": {"baseline_acc_pct": 90.0, "n": 30}, "clean": audio, "noisy": audio,
         "phone": audio}
    )  # fmt: skip
    lines = table.splitlines()
    assert len(lines) == 5
    assert lines[1].split() == ["original_text", "-", "-", "-", "-", "-", "-", "90.00", "-", "-"]
    assert lines[2].split() == [
        "clean", "1.50", "2.00", "-", "100.00", "33.33", "0.00", "90.00", "-3.33", "2",
    ]  # fmt: skip


def test_location_token_ok_uses_expected_location_when_provided() -> None:
    assert eval_wer.location_token_ok(
        "Visited Grand Hotel today.", "visited grand hotel", "Grand Hotel"
    )
    assert not eval_wer.location_token_ok(
        "Visited Grand Hotel today.", "visited hotel", "Grand Hotel"
    )
    assert eval_wer.location_token_ok("Visited somewhere.", "visited somewhere", "") is None


def test_evaluate_uses_asr_text_and_gold_location(tmp_path: Path) -> None:
    messages = [
        {"id": "msg_001", "sender_id": "w1", "message": "We visited 7 households in Kisumu today."}
    ]
    manifest = make_manifest(messages, voices=["v1"])
    transcripts = {
        v: [
            {
                "id": "msg_001",
                "sender_id": "w1",
                "source": "audio",
                "message": "We visited 7 households in Kisumu today.",
                "asr_text": "We visited seven households in Kisumu today.",
            }
        ]
        for v in eval_wer.VERSIONS
    }
    expected = [
        {
            "id": "msg_001",
            "relevant": True,
            "activity_type": "household_outreach",
            "location": "Kisumu",
            "reported_count": 7,
        }
    ]
    res = eval_wer.evaluate(
        manifest, transcripts, messages, expected, rule_baseline, tmp_path / "work"
    )
    clean_clip = res["clips"][0]
    assert clean_clip["hypothesis"] == "We visited seven households in Kisumu today."
    assert clean_clip["wer_pct"] > 0.0
    assert clean_clip["location_token_ok"] is True
    assert clean_clip["gold"]["location"] == "Kisumu"
    assert clean_clip["gold"]["reported_count"] == 7


def test_main_with_predictions_dir(tmp_path: Path) -> None:
    messages = [{"id": "msg_001", "sender_id": "w1", "message": "We screened 5 women in Nakuru."}]
    manifest = make_manifest(messages, voices=["v1"])
    t_dir = tmp_path / "transcripts"
    t_dir.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "messages.json").write_text(json.dumps(messages), encoding="utf-8")
    (tmp_path / "expected.json").write_text(
        json.dumps([{"id": "msg_001", "relevant": True, "activity_type": "health_screening"}]),
        encoding="utf-8",
    )
    for v in eval_wer.VERSIONS:
        (t_dir / f"transcripts_{v}.json").write_text(
            json.dumps(records(messages)), encoding="utf-8"
        )
    pred_dir = tmp_path / "gliner_preds"
    pred_dir.mkdir()
    sample_pred = [pred("msg_001", True, "health_screening", "Nakuru", 5)]
    for system in eval_wer.SYSTEMS:
        (pred_dir / f"{system}.json").write_text(json.dumps(sample_pred), encoding="utf-8")

    out_file = tmp_path / "out" / "gliner_wer.json"
    rc = eval_wer.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--transcripts-dir",
            str(t_dir),
            "--messages",
            str(tmp_path / "messages.json"),
            "--expected",
            str(tmp_path / "expected.json"),
            "--predictions-dir",
            str(pred_dir),
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.is_file()
    assert not (out_file.parent / "baseline_predictions").exists()
