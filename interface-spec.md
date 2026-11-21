# Dandelion4 Interface Spec v2

**For:** Terence, Teresa, Eric
**Version:** v2, 2026-11-06 (status & module paths updated 2026-11-20)
**Status:** Implemented for v2 contracts, ASR, GLiNER pipeline, hidden-test evaluation, and demo (see §1, §7, and §12 for current module paths).

**Purpose.** Every module reads and writes the JSON formats defined here, so each of us can build and test our part without waiting for the others. Where this spec and the code disagree, the spec wins and the code gets changed. Section 12 tracks implementation status.

---

## 1. How the pieces fit

```
 A. Data / Inputs ─────────────► Message files  ─────────────┬──────────────► C. Systems ──► Record files ──► D. Evaluation
    synthetic / HF / hidden      (§3)                        │   legacy baseline (frozen, wrapped)            eval_wer.py /
    + gold label files (§6)                                  │   GLiNER v2 (`src/nlp/gliner_system.py`)       run_evaluation.py
                                                             │
                                  B. Audio → text ───────────┘
                                  TTS / MP4 / WAV → Whisper → transcripts (also Message files)
```

| Stage | Owner | Reads | Writes | Current Code |
|---|---|---|---|---|
| A. Data | Teresa | — | Message files (§3) and gold label files (§6) | `examples/synthetic_messages.json`, `examples/huggingface_messages.json`, `examples/expected_outputs.json` |
| B. Audio → text | Eric | Message files or audio/video inputs | WAV clips, `examples/audio_manifest.json`, transcript files (Message format, §3) | `src/training/make_audio.py`, `src/asr/transcribe.py`, `src/asr/clean_transcripts.py`, `src/asr/audio_pipeline.py` |
| C. Systems | baseline: Teresa; GLiNER: Terence | any Message file | a Record file (§4) | `src/legacy/baseline.py` (frozen v1, wrapped by `evaluation/baseline_runner.py`), `src/nlp/gliner_system.py`, `src/pipeline/pipeline.py`, `src/pipeline/hidden_test_pipeline.py` |
| D. Evaluation | Eric / Teresa | Record files + gold label files / hidden-test CSV | results JSON / summary | `evaluation/eval_wer.py`, `evaluation/run_evaluation.py`, `demo/run_demo.py` |

The only things that cross a stage boundary are the files in §3–§6. **No module may depend on another module's internal behavior,** only on these formats and on the public functions listed in §7.

---

## 2. Rules for every file

1. **Encoding.** Strict JSON, UTF-8: reject duplicate object keys, NaN and Infinity. Read with `open(path, encoding="utf-8")` and strict parsing. Write with `json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)`.
2. **Order.** Message, transcript, Record, and Gold files are arrays of objects sorted by `id`. Every script sorts these arrays by `id` before writing. Evaluation result files use their schema in §7.5; private stage-B files are outside this rule.
3. **Field names.** snake_case. (The Week 5 README example uses `activityName`; v2 uses `event_name`, see §4.)
4. **Required and optional fields.** Each format's table marks every field as required or optional.
   - A required field is always present. Use `null` for an unknown value only when the field type allows `null`; otherwise supply a valid value or stop with an error. An empty list is `[]`.
   - An optional field may be left out, and the table says what leaving it out means.
   - Every Record field is required.
   - No file may contain a field its table doesn't list. To add one, follow §10.
5. **IDs** look like `<prefix>_<number>`.
   - **The prefix names the dataset.** It contains one or more ASCII lowercase letters or digits, and no two datasets share one. The number contains one or more ASCII digits. Today `msg_` marks the template messages and `hf_` the SmolLM2 messages.
   - **Numbers within a dataset are zero-padded to one width.** For example, `msg_001` to `msg_040`; a dataset with 5,000 messages would use four digits. That keeps IDs in order when they're sorted as text.
   - **IDs are unique within a file and across datasets.** The one exception: a transcript keeps the ID of the text message it was made from. It is the same message in another medium, so it may share non-PII gold labels. Transcript PII labels must be created for the actual transcript text (§7.5).
6. **Text is stored as-is.** Preserve the supplied `message` when storing it and when constructing `text_redacted`, except for redaction replacements. Systems may normalize temporary copies for matching; copied values follow §4. For cleaned transcripts, `message` may be the cleaned text and `asr_text` contains the raw output (§7.3).
7. **Character offsets** are 0-based with `end` exclusive, so `text[start:end]` is the span. They refer to the `message` field of the Message being processed or labeled. For a transcript, that is the transcript text, not the text it was made from.
8. **Privacy.**
   - No private or personal data goes in the repo.
   - Data we create uses fake names, and phone numbers in the form `0700 000 1xx`.
   - Public datasets are used under their licenses and cited in the README.
   - **Nothing written to a Record file may contain a raw name or phone number** (§5).
9. **Scripts.**
   - Inputs and `--out` come from the command line, with the defaults listed in §7.
   - No script assumes a particular dataset: no fixed message count, file list or ID list. A built-in default is fine when it can be overridden from the command line.
   - Keep function and class definitions importable at module scope; put only command-line execution under `if __name__ == "__main__":`.
   - A model is loaded once per run, not once per message.
   - If a script cannot process a message, it stops with an error and does not create or modify the output file. It never writes a made-up Record.

---

## 3. Message: the input to every system

Any file in this format is a valid input to every system. **This spec fixes the format, not the data:** datasets can come from any source and be any size, and adding one needs no change here.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | See §2 rule 5 |
| `sender_id` | string | yes | Pseudonymous sender, e.g. `worker_05`. Never a real name. |
| `message` | string | yes | The text. For audio, the transcript. May be `""` if Whisper returned nothing. |
| `source` | `"text"` or `"audio"` | no | If missing, the message counts as `"text"`. Transcript files **must** set `"audio"`. |
| `asr_text` | string | no | Transcripts only: Whisper's raw output, present only when `message` holds a cleaned-up version (§7.3). If present, `source` must be `"audio"`. Cleaning may leave the text unchanged. |

```json
[
  {"id": "msg_016", "sender_id": "worker_04", "message": "We visited 7 households in Narok today."}
]
```

The same message as a transcript:

```json
[
  {"id": "msg_016", "sender_id": "worker_04", "source": "audio",
   "message": "We visited seven households in Narak today."}
]
```

**Dataset files**
- **Naming.** One file per dataset, named `<dataset>_messages.json`. A very large dataset may be split into parts named `<dataset>_messages_<part>.json` (`_001`, `_002`, …). IDs stay unique across the parts, and one gold file can cover all of them. Transcript files keep their `transcripts_<version>.json` names.
- **Location.** Data we create goes in `examples/`. Large or third-party datasets aren't committed; commit the script that downloads or converts them instead.
- **Language.** v2 assumes English. A message in another language is still processed like any other; gold labels for it are optional.
- **The list of datasets in use** belongs in the README's data section (Week 5 "Proposed data", Week 7 "Development data"), with each one's source, size and license. It doesn't belong in this spec.

How today's files measure up against the format is in §8.

---

## 4. Record: the output of every system

Both systems (baseline and GLiNER) write this exact format, so they can be scored the same way and compared directly. **All fields are required.**

| Field | Type | Rule |
|---|---|---|
| `id` | string | The Message's `id` |
| `sender_id` | string | Copied from the Message |
| `source` | `"text"` or `"audio"` | Copied from the Message (missing means `"text"`) |
| `relevant` | bool | See "What counts as relevant" below |
| `activity_type` | string or `null` | The type of the activity the Record describes: one of `household_outreach`, `health_screening`, `community_meeting`, `health_referral`, `other`. **`null` exactly when `relevant` is `false`.** Use `other` for a relevant activity that fits none of the listed types. The list can grow as new data needs it (§10). |
| `location` | string or `null` | The place where that activity happened, copied from the text. Evaluation ignores capitalization but not spelling. |
| `reported_count` | integer ≥ 0, or `null` | See "Counting rule" below |
| `date_text` | string or `null` | When that activity happened, copied from the text, e.g. `"today"` or `"this Friday"`. Not converted to a calendar date in v2. |
| `event_name` | string or `null` | The name of a named event, copied from the text, e.g. `"Annual Tech Gala"` |
| `event_desc` | string or `null` | A short phrase from the text describing what was done. Not scored in v2 (§7.5). |
| `text_redacted` | string | `message` with each redaction replaced by its placeholder. Nothing else is changed. |
| `redactions` | array | Redaction objects (§5), sorted by `start`; `[]` if there are none |
| `confidence` | number 0–1, or `null` | The system's confidence that the record is correct. `null` if the system has no estimate. Each system explains how it computes this in its module docstring. |
| `verification_status` | string | Systems write `"pending"` when `relevant` is `true` and `"not_applicable"` when it is `false`. A human reviewer later changes `pending` to `"verified"` or `"rejected"`. |

The same rules apply to gold labels (§6).

**What counts as relevant.**
- A message is relevant when it **reports** a field activity that the sender or their team has done or is doing: "We visited 22 households in Baringo today.", "We are screening women in Narok now."
- Plans, schedules, invitations, reminders, questions, greetings and thanks are **not** relevant, even when they mention an activity. "Our next health screening is scheduled for Friday" and "What time is our meeting?" are both `relevant: false`.

**One record per message.** A message gets exactly one Record, even when it describes several activities.
- **The first activity in reading order.** For a relevant message, select the first activity that reports work done or underway, ignoring plans, invitations, questions and other non-reports. The Record describes that activity: `activity_type`, `location`, `reported_count`, `date_text`, `event_name` and `event_desc` all come from it. If that activity has no value for a field, the field is `null`; don't borrow a value from another activity.
- **What one activity is.** One kind of work at one place and time. "We visited 10 households in Kajiado and 5 in Nakuru." is two activities, so the Record gets `location` `"Kajiado"` and `reported_count` `10`.
- **A value given twice.** If the activity gives a field twice, use the first: "this Saturday, July 8th" gives `date_text` `"this Saturday"`. For `reported_count`, the counting rule below decides instead.
- **Known limitation.** Messages with several activities lose information in v2 (the README's Week 5 challenge "Messages describing multiple events").

**Counting rule** for `reported_count`:

| Case | Rule | Example |
|---|---|---|
| What is counted | The number of people or units (households, families, participants, referrals) the activity reached | "We screened 48 women" → `48` |
| Parts of one group | Add them up | "20 women and 15 men" → `35` |
| One group given in two units | Use the first one given | "12 households (about 60 people)" → `12` |
| An estimate | Use its number | "about 30" → `30`; "over 40" → `40` |
| A range | Use its lower end | "20–25" → `20` |
| A number written as a word | It's the same number. A system that can't convert words writes `null`. | "seven households" → `7` |
| No number | `null` | "We visited some households" → `null` |

**Copying text values.** `location`, `date_text`, `event_name` and `event_desc` are copied from the message.
- **Only the words that name the value.** No leading "the", "a", "an", "in", "at", "on" or "by", and no trailing Unicode punctuation; the closing bracket of a `[NAME]` or `[PHONE]` placeholder is part of the placeholder: "…at the clinic in Nakuru." → `"Nakuru"`; "by this Friday" → `"this Friday"`; "the Annual Tech Gala" → `"Annual Tech Gala"`.
- **A place is named, not described.** `location` is the name of a place (a village, town, county or named facility), not the words around it.
- **The text's own spelling.** Keep it, even when it's wrong: a misheard "Narak" stays `"Narak"`, and evaluation counts it as wrong. Capitalization may differ; case-insensitive comparisons use Unicode casefolding.
- **Redacted spans become placeholders.** A value that would contain a redacted span uses the placeholder instead (§5), e.g. `"[NAME]'s house"`.
- **Never an empty string.** If there's no value, write `null`, not `""`.

**No guessing.** Every non-null value must be supported by the message text. When unsure, write `null`. This addresses the README's Week 5 risk of "hallucinated or unsupported information".

**Example 1: baseline on `msg_002`.** This is exactly what the updated baseline must write. The six pre-existing baseline fields match what it writes today for this message.

```json
{"id": "msg_002", "sender_id": "worker_05", "source": "text", "relevant": true,
 "activity_type": "health_screening", "location": "Nakuru", "reported_count": 48,
 "date_text": null, "event_name": null, "event_desc": null,
 "text_redacted": "We screened 48 women in Nakuru.", "redactions": [],
 "confidence": null, "verification_status": "pending"}
```

**Example 2: baseline on the clean transcript of `msg_016`.** Whisper heard "seven … Narak", so the count and location are lost.

```json
{"id": "msg_016", "sender_id": "worker_04", "source": "audio", "relevant": true,
 "activity_type": "household_outreach", "location": null, "reported_count": null,
 "date_text": null, "event_name": null, "event_desc": null,
 "text_redacted": "We visited seven households in Narak today.", "redactions": [],
 "confidence": null, "verification_status": "pending"}
```

**Example 3: a GLiNER record for a PII test message.** The values are illustrative, not real output. The message is `"Mary Wanjiku visited 12 households in Narok today. Call 0700 000 101 for details."`

```json
{"id": "pii_001", "sender_id": "worker_03", "source": "text", "relevant": true,
 "activity_type": "household_outreach", "location": "Narok", "reported_count": 12,
 "date_text": "today", "event_name": null, "event_desc": null,
 "text_redacted": "[NAME] visited 12 households in Narok today. Call [PHONE] for details.",
 "redactions": [{"type": "NAME", "start": 0, "end": 12}, {"type": "PHONE", "start": 56, "end": 68}],
 "confidence": 0.87, "verification_status": "pending"}
```

**Where each field comes from today:**

| v2 field | `baseline.py` output today | GLiNER notebook today | Week 5 README example |
|---|---|---|---|
| `id`, `relevant`, `activity_type`, `location`, `reported_count`, `verification_status` | same names, same meaning | — / `Location` label | `location` |
| `sender_id`, `source` | not written yet | — | — |
| `date_text` | — | `Date` label | `date` |
| `event_name` | — | `EventName` label | `activityName` |
| `event_desc` | — | `EventDesc` label | `activityDesc` |
| `text_redacted` | `anonymize()` result (computed but not written) | redacted text (printed) | — |
| `redactions` | — | `Name`/`Attendees` spans (used inside the notebook) | `"name": [NAME]` |
| `confidence` | — | entity `score` | "produces confidence level" |
| `reported_count` | (see first row) | — | `members` |

The notebook's `Attendees` and `EventActivities` labels have no v2 field.

---

## 5. Redaction

A Redaction object is `{"type": "NAME" | "PHONE", "start": int, "end": int}`, with offsets into the `message` being processed (§2 rule 7). Gold files also include `"text"` (§6).

| Type | What it covers | Placeholder |
|---|---|---|
| `NAME` | A person's name, including a title that is part of it ("Mary Wanjiku", "Dr. Achieng", "CEO Melon Usk"), or a role that points to one specific person ("VP of Sales") | `[NAME]` |
| `PHONE` | A phone number in any format or country ("0722 123 456", "+254 722 123 456", "+1 408 555 0100"). The baseline's regex finds only Kenyan formats; it is one implementation, not the definition. | `[PHONE]` |

**Span boundaries.** A span covers the name or number itself, without articles or a possessive "'s": "the VP of Sales" → "the [NAME]"; "Mary's house" → "[NAME]'s house".

**Not redacted:** groups and counts ("48 women", "all regional managers"), places, dates, and organisations, including those named after a person ("Kenyatta National Hospital").

**Not covered in v2:** email addresses, ID numbers, home addresses and ages are not redacted yet. Test data must not contain them. A new redaction type is added through §10.

Redactions never overlap. Every span satisfies `0 <= start < end <= len(message)` when the Message is available. Every redaction is replaced by exactly one typed placeholder. Literal placeholder text already in the Message remains unchanged; it is not a redaction.

**The rule applied to Terence's notebook example.** The notebook wrote `"[REDACTED], the [REDACTED], and all [REDACTED] will be there."` Under v2 the output is:

```
CEO Melon Usk, the VP of Sales, and all regional managers will be there.
→ [NAME], the [NAME], and all regional managers will be there.
```

Two changes from the notebook:
- **Typed placeholders.** `[NAME]` and `[PHONE]` replace `[REDACTED]`. They match the baseline's existing `[PHONE]`, and they tell a reviewer what kind of information was removed.
- **Groups stay.** "All regional managers" is a group, not a person. Groups and counts have to stay readable, or `reported_count` can't be checked against the text.

---

## 6. Gold labels

Gold files hold the correct answers that evaluation compares against. They follow the same rules as Records: relevance, one record per message, counting and copying text values (§4), and the redaction rules (§5). Gold `pii` spans may be in any order but must not overlap. Check copied gold values against the Message after replacing valid gold PII spans with their typed placeholders. When `pii` is missing, the checker defers substring checks for values containing placeholders; it cannot infer the spans.

| Field | Type | Required | Rule |
|---|---|---|---|
| `id` | string | yes | Matches the Message `id` |
| `relevant` | bool | yes | As in Record |
| `activity_type` | string or `null` | yes | As in Record |
| `location`, `reported_count`, `date_text`, `event_name`, `event_desc` | as in Record | no | **Missing means "not labeled"**, so the field isn't scored. `null` means "labeled, and there is none". |
| `pii` | array | no | The correct redactions: Redaction objects plus `"text"`, e.g. `{"type": "NAME", "start": 0, "end": 12, "text": "Mary Wanjiku"}`. Each span's `text` is exactly `message[start:end]`, and its length equals `end - start`. **Missing means "not labeled for PII"; `[]` means "labeled: no personal data".** |

```json
[
  {"id": "pii_001", "relevant": true, "activity_type": "household_outreach",
   "location": "Narok", "reported_count": 12,
   "pii": [{"type": "NAME", "start": 0, "end": 12, "text": "Mary Wanjiku"},
           {"type": "PHONE", "start": 56, "end": 68, "text": "0700 000 101"}]}
]
```

**Gold files**
- A dataset can be scored for any subset of its IDs; its gold file is named `<dataset>_expected.json`. Messages without a gold entry are not scored.
- A dataset without one can still be processed; it just isn't scored.
- Labels can cover only some fields and be added over time, since a missing field means "not labeled".
- The existing `examples/expected_outputs.json` holds non-PII labels for `synthetic_messages.json` and its transcripts. It keeps that name because current scripts read it. Transcript PII requires labels explicitly made for the exact transcript file (§7.5), for example `transcripts_clean_expected.json`.

---

## 7. Module interfaces

### 7.1 Systems (stage C)

Every system module exposes one function:

```python
def process_messages(messages: list[dict]) -> list[dict]:
    """Message v2 list in, Record v2 list out: same length, same ids, same order.
    Raises an error for a message it can't process."""
```

It also has a command line:

```
python src/<module>.py [INPUT] [--out OUTPUT]
```

- `INPUT` defaults to `examples/synthetic_messages.json`.
- `OUTPUT` defaults to `examples/<system>_output_<input>.json`, where `<input>` is the input file's name without `.json`; e.g. `examples/baseline_output_transcripts_clean.json`. The default distinguishes input basenames. Inputs with the same basename share a default output path; use `--out` to distinguish them.
- When the input is `synthetic_messages.json` (given or by default), the default output is `examples/<system>_output.json`, which keeps today's `baseline_output.json`.
- The path is relative, so it lands under whatever folder you run from.

| System | File | Owner | Output for the default input |
|---|---|---|---|
| Keyword baseline (frozen v1 comparison system) | `src/legacy/baseline.py` (wrapped by `evaluation/baseline_runner.py` & `src/asr/audio_pipeline.py`) | Teresa | `examples/baseline_output.json` |
| GLiNER | `src/nlp/gliner_system.py` | Terence | `examples/gliner_output.json` |

**Public functions in `src/legacy/baseline.py`.** Other modules may call these (or invoke `baseline.py` via `evaluation/baseline_runner.py`); their behavior only changes through §10.
- `anonymize(text)`
- `classify_message(text)`
- `extract_location(text)`
- `extract_count(text)`
- `process_message(message)`

The GLiNER system may reuse them for fields GLiNER doesn't produce. If it does, its docstring must say so.

**Colab / Unified CLI.** Run the end-to-end CLI via `python -m src.main` (or standalone GLiNER via `python -m src.nlp.gliner_system`). Notebooks in `notebooks/` are kept for reference/experiments, while production logic lives in `src/nlp/gliner_system.py`.

**Implementation & validation module.** `src/core/contracts.py` contains the standard-library model and validation helpers (`Message`, `Record`, `Gold`, `Redaction`, `GoldPiiSpan`, `RecordResults`, `build_record`, `validate_file`, `load_messages`, `load_records`, `load_gold`, `write_json`). The appendix below preserves the original standalone extractable template bundle from 2026-11-06 for historical reference.

**Minimal usage.** After extracting the companion files, save this as `src/empty_example.py`. It accepts only empty messages and rejects every other message before writing output; it does not attempt PII detection.

```python
from contracts import build_record
from system_template import run_cli


def process(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    if any(m["message"] for m in messages):
        raise ValueError("This example handles empty messages only")
    return [build_record(
        m, relevant=False, activity_type=None, location=None, reported_count=None, date_text=None,
        event_name=None, event_desc=None, redactions=[], confidence=None) for m in messages]


if __name__ == "__main__":
    run_cli(type("EmptySystem", (), {"name": "empty", "process_messages": staticmethod(process)})())
```

### 7.2 Data (stage A)

A data source is anything that produces Message files (§3) and, where labels exist, gold files (§6). That can be a generator script, a script that downloads or converts a public dataset, or a hand-made file. `src/generate_data.py` is one such source:

```
python src/generate_data.py [--out examples/huggingface_messages.json] [--prefix hf] [--seed 1234]
```

- Writes a Message file with IDs `<prefix>_001`… and sender IDs like `worker_01`.
- Removes list numbering and signature prefixes (`N. *Community Field Worker, Nairobi:`) from the generated lines.

### 7.3 Audio → text (stage B)

```
python -m src.training.make_audio
python -m src.asr.transcribe --version all
```

- **Input.** By default `src/training/make_audio.py` voices a fixed set of 30 messages from `synthetic_messages.json` (the Week 7 test set), and supports voicing other Message files or subsets via CLI flags.
- **Writes** `examples/transcripts_<version>.json` (Message format, `"source": "audio"`). Those are the only stage B files other modules read.
- **Private to stage B:** `audio_manifest.json`, `transcripts_*.meta.json` and the WAV clips belong to stage B and may change without a spec update.
- **Cleaned-up transcripts.** `src/asr/clean_transcripts.py` normalizes transcripts while preserving Whisper's raw output in `asr_text`, and WER is computed on `asr_text`. Systems copy their values from `message`, so a corrected place name reaches the Record.

### 7.4 Evaluation (stage D)

| Script | Owner | Command | Writes |
|---|---|---|---|
| `evaluation/run_evaluation.py` | Eric / Teresa | `python -m evaluation.run_evaluation [--mode synthetic\|hidden]` | evaluation summary JSON / CSV metrics |
| `evaluation/eval_wer.py` | Eric | `python -m evaluation.eval_wer` | `evaluation/wer_results.json` |
| `demo/run_demo.py` | Team | `python -m demo.run_demo` | `demo/outputs/demo_metrics.json`, `predicted_outputs.csv` |

- **`evaluate.py` arguments.** `M` is the Message file the Records in `P` were made from. The defaults are `examples/synthetic_messages.json`, `examples/baseline_output.json` and `examples/expected_outputs.json`, so the plain command keeps working.
- **Transcript PII provenance.** `--expected-transcript-pii` declares that PII spans in `E` were labeled for the exact input transcript file `M`. Without this flag, skip audio-source messages for PII, even if their offsets happen to match. Other metrics still use their gold fields. Use the flag only for a Gold file explicitly labeled against that exact transcript.
- **`eval_wer.py`** reads only `relevant`, `activity_type`, `location` and `reported_count` from Records, so it works with any v2 system.

### 7.5 Metric definitions

Every script that reports these metrics computes them this way:
- Express percentages to two decimal places, rounding exact ties away from zero. Counts are non-negative integers, including zero.
- A metric that can't be computed (for example, no gold `pii`) is `null`.
- **What gets scored.** Metrics are computed per input Message file, over its messages that have a gold entry.
  - A missing Record is incorrect for classification. For filter recall it is a false negative when gold is irrelevant; it does not enter filter precision. For attribute scoring, treat each missing prediction as null. For PII recall, it covers no gold spans and contributes no system spans to over-redactions.
  - Gold entries for messages that aren't in the input file are ignored. For example, a 30-message transcript file is scored on those 30 messages, not on all 40 gold entries.

| Metric (Week 5 target) | Definition |
|---|---|
| Classification accuracy | Share of scored messages where the Record's `relevant` **and** `activity_type` both equal gold |
| Filter precision (> 95%) | Of scored Records with `relevant: false` (the messages the system filters out), the share whose gold `relevant` is `false` |
| Filter recall | Of the gold-irrelevant messages, the share the system marked `relevant: false`. Reported alongside filter precision, because the Week 5 formula ("removed noise / flagged noise") reads like recall. |
| Attribute extraction F1 (> 95%) | Micro F1 over every (message, field) pair for the fields `location`, `reported_count`, `date_text` and `event_name` that are labeled in gold. `event_desc` isn't scored in v2: free text has no single right answer. Strings are compared after trimming and Unicode casefolding; integers exactly. True positive: gold is non-null and the prediction equals it. False positive: the prediction is non-null and differs from gold. False negative: gold is non-null and the prediction differs. A wrong value counts as both a false positive and a false negative. F1 = 2TP / (2TP + FP + FN). |
| PII scrubbing recall (> 95%) | Share of gold `pii` spans whose characters are all covered by the union of system redaction spans, whatever their types. Only messages whose gold has `pii` are scored (`[]` counts, missing does not). Skip a message if any gold span is out of bounds or its text differs from the input at those offsets. For `source: "audio"`, score PII only when `--expected-transcript-pii` declares that the Gold PII labels were explicitly created for the exact input file; otherwise skip that message for PII. `pii_skipped_messages` counts input messages with a gold entry skipped for missing `pii`, unavailable transcript-specific PII labels, or invalid/mismatched spans, counting each message once. **Over-redactions:** system spans on scored messages that overlap no gold span; use `null` if no messages were scored for PII. |
| WER (< 10%) | Aggregate (S + D + I) / N over all clips. Normalize by lowercasing, deleting straight and right-curly apostrophes, replacing each run of characters outside ASCII `[a-z0-9]` with a space, and collapsing whitespace. N is the total reference word count; an individual empty reference contributes its hypothesis words as insertions. If aggregate N is zero, WER is null. Use raw `asr_text` as the hypothesis when present. |

`evaluation/records_results.json` is organized by system name, then input file name:

```json
{"baseline": {"synthetic_messages": {
  "n_messages": 0,
  "classification_acc_pct": null,
  "filter_precision_pct": null, "filter_precision_n": 0,
  "filter_recall_pct": null, "filter_recall_n": 0,
  "attribute_f1_pct": null, "attribute_n": 0,
  "pii_recall_pct": null, "pii_n": 0, "pii_skipped_messages": 0,
  "over_redactions": null}}}
```

Counts describe the support below; they are always integers, including zero. The attribute F1 denominator is `2TP + FP + FN`, not `attribute_n`. Any metric with a zero denominator is `null`:

| Count | Support counted |
|---|---|
| `n_messages` | scored messages, for accuracy |
| `filter_precision_n` | scored Records with `relevant: false` |
| `filter_recall_n` | gold-irrelevant messages |
| `attribute_n` | labeled (message, field) pairs |
| `pii_n` | scored gold spans |

---

## 8. Checking your files

Extract the checker to `tests/check_interface.py` using the appendix instructions. It needs only the standard library. Run it on every file you produce before opening a PR:

```
python tests/check_interface.py messages    examples/synthetic_messages.json examples/huggingface_messages.json
python tests/check_interface.py transcripts examples/transcripts_clean.json
python tests/check_interface.py records     examples/baseline_output.json --messages examples/synthetic_messages.json
python tests/check_interface.py gold        examples/expected_outputs.json --messages examples/synthetic_messages.json
```

For a readable JSON file, structural problems produce exit status 1. Unreadable or malformed JSON produces an error and nonzero status. The checker prints the first 50 problems and their total; exit status 0 means its automated checks passed, not that semantic privacy or labeling has been verified.

**What it checks:**
- **Each file on its own:** required fields and types, allowed values, IDs, sort order, the relevance and status rules, and redaction spans. Record text values must appear in `text_redacted` without a leading "the", "in" and so on. Without the original Message, there must be at least as many occurrences of each placeholder as corresponding spans; additional occurrences may be literal text. With the Message, reconstruction is checked exactly. The checker does not reject phone-like digits by pattern because counts and phone numbers cannot be reliably distinguished that way.
- **With `--messages`, first validate the reference as a Message file.** If it is invalid, report its problems and skip dependent comparisons. Otherwise a Record file is compared with the Message file it came from: every message has exactly one Record, `sender_id` and `source` match, and the redaction offsets rebuild `text_redacted` exactly.
- **With `--messages`, a gold file is checked against its messages:** each `pii` span's `text` must be what the message says at those offsets.

**What it can't check:** whether a message is really relevant, whether the first activity was chosen, whether the count follows the counting rule, or whether every name or phone number was redacted. A passing check does not establish semantic privacy. Reviewers check those.

Check transcript files with `transcripts` on their own. They share IDs with their source messages on purpose, so checking both together in `messages` mode reports collisions.

| Files | Result today | Reason |
|---|---|---|
| `synthetic_messages.json` | 0 problems | — |
| `synthetic_messages.json` + `huggingface_messages.json` | 10 problems | 10 colliding IDs |
| `transcripts_clean.json` | 30 problems | `source` missing |
| `baseline_output.json` with `--messages synthetic_messages.json` | fails | Several required fields are missing and copied fields cannot yet be compared successfully |
| `expected_outputs.json` with `--messages synthetic_messages.json` | 0 problems | — |

**Working without the others.** The examples in §3–§6 pass the checker. Save them as test inputs; you don't have to wait for another module to be finished. Wrap the single Record examples in `[ ]`, because a file is always an array.

---

## 9. Branches and pull requests

1. Merge **PR #1** first. It brings the `synthetic-data` and audio code to `main`.
2. Branch from `main` for your own work, with one PR per module.
3. Don't edit a file someone else owns without that person reviewing the PR.
4. Run the checker on every file you produce before opening the PR, with `--messages` for Record and gold files.

---

## 10. Changing this spec

- **Adding a dataset:** no change to this spec. Give it its own ID prefix and list it in the README's data section.
- **Adding an optional field to Message or Gold, or a new activity type or redaction type:** open a PR that updates this file and the checker, and ask the owners of the affected formats to review.
- **Any other change:** all three of us agree, and the major version goes up (for example, v2 to v3). That covers renaming or removing a field or an allowed value, changing a field's type, adding a required field, and changing a labeling rule (relevance, one record per message, counting, copying text values, span boundaries), because that changes what a correct answer is.
- **Format owners** approve changes to their format:
  - Message and Gold: Teresa
  - transcript fields (`source`, `asr_text`): Eric
  - Record and Redaction: Terence

---

## 11. Decisions in v2

| Decision | Why |
|---|---|
| The spec fixes formats, not datasets | We expect more and bigger data sources. Any file that follows §3 works with every system without editing this spec. |
| Modules connect through JSON files plus `process_messages`. No web server, no orchestrator script. | Everything runs in batches, in one repo, locally or in Colab. Files work the same in a notebook as in a script. |
| Record v2 retains the six baseline field names and adds the GLiNER and Week 5 fields | Both systems produce the same format for comparison; values change where needed to follow the current rules |
| Keep the keyword baseline as the comparison system | The README's Week 7 section asks for "baseline versus current approach" |
| Relevant means a report of work done or underway; plans and questions aren't relevant | The goal is verifiable records of activities and outcomes, and a plan isn't an outcome yet |
| One Record per message, describing the first reported activity in reading order | A rule every system and labeler applies the same way; splitting multi-activity messages is left for a later version |
| Counting: parts add up, estimates count as their number, ranges as their lower end | One right answer per message, never more than the text supports |
| Text values are copied from the text without leading articles or prepositions; `event_desc` isn't scored | Exact-match scoring needs one right answer per field, and free text doesn't have one |
| `[NAME]`/`[PHONE]` only; `PHONE` means any phone number; groups and counts aren't redacted | Matches the baseline and the Week 5 example, doesn't depend on one regex, and keeps counts readable. Listing what isn't covered keeps test data honest. |
| Metrics are computed over the messages in the input file | Otherwise a 30-message transcript file scored against 40 gold entries would count 10 unvoiced messages as wrong |
| The default output name follows the input basename | It distinguishes different basenames; use `--out` when basenames collide |
| `activity_type` gains `other`, and the list can grow; `null` exactly when not relevant | Every relevant message gets a type, even outside the four (e.g. the SmolLM2 "community garden" message) |
| One ID prefix per dataset: keep `msg_`, rename the SmolLM2 IDs to `hf_`; transcripts keep the ID of their source message | The smallest change: nothing reads the SmolLM2 IDs, while the `msg_` IDs are used by the gold labels, the audio clips and the transcripts |
| In gold, a missing field means "not labeled" and `null` (or `[]` for `pii`) means "none" | Labels can be added gradually without breaking evaluation |
| Cleaned transcript in `message`, raw Whisper output in `asr_text` | Systems need no changes, and WER stays measured on what Whisper actually produced |

---

## 12. Checklist: implementation status

The `msg_` IDs and transcript texts stay. Recompute and report any changed results separately: relevance, counting, privacy, metrics and other corrected rules can change prior outputs.

**Data & Baseline (Teresa / Legacy)**
- [x] `examples/huggingface_messages.json` updated with `hf_` IDs and cleaned prefixes (`src/generate_data.py` was not added to this repo).
- [x] `src/legacy/baseline.py` kept frozen as the v1 comparison baseline and wrapped in-repo by `evaluation/baseline_runner.py` (isolated subprocess) and `src/asr/audio_pipeline.py` (adapts v1 output to v2 `Record` contracts).
- [x] `examples/expected_outputs.json` includes `location` and `reported_count` gold fields (`null` for irrelevant messages).
- [ ] `evaluation/evaluate.py` and `examples/pii_messages.json` / `examples/pii_expected.json`: superseded by `evaluation/eval_wer.py`, `evaluation/run_evaluation.py`, and hidden-test evaluation in `src/pipeline/hidden_test_pipeline.py` + `demo/run_demo.py`.

**GLiNER & Pipeline (Terence)**
- [x] Created `src/nlp/gliner_system.py` with `process_messages` and CLI, loading `gliner_medium-v2.1` once per run.
- [x] Mapped GLiNER labels + regex/rule extraction (`Location` → `location`, `Date` → `date_text`, `EventName` → `event_name`, `EventDesc` → `event_desc`, `NAME` & `PHONE` → `[NAME]`/`[PHONE]` redactions and `text_redacted`).
- [x] Integrated into `src/pipeline/pipeline.py`, `src/pipeline/hidden_test_pipeline.py`, and `src/main.py`.
- [x] Added `gliner` to `requirements.txt`.

**Audio, ASR, Evaluation & Contracts (Eric)**
- [x] Implemented `src/core/contracts.py` enforcing §2–§6 schemas and strict JSON validation.
- [x] `src/asr/transcribe.py` writes `"source": "audio"` across all transcript files and is covered by `tests/test_transcribe.py`.
- [x] `src/training/make_audio.py` supports voicing custom Message files and ID subsets while preserving the 30-message default.
- [x] `evaluation/eval_wer.py` scores both baseline and GLiNER Records, computes WER on `asr_text` when present, and reads gold locations.
- [x] `evaluation/baseline_runner.py` passes `--out` in an isolated temp directory and `tests/test_baseline_runner.py` verifies it.
- [x] Added `evaluation/run_evaluation.py` and `demo/run_demo.py` with unit tests (`tests/`, 291 fast tests passing).

---


## Appendix: Extractable Python files

The following fenced blocks contain the standalone checker and companion templates. `src/contracts.py` is a separate repository file. Save the following standard-library script as `extract_interface.py` and run it from the repository root to extract the five embedded files. It refuses to overwrite an existing target.

### Extracting the files

```python
import re
from pathlib import Path

spec = Path("interface-spec.md").read_text(encoding="utf-8")
targets = (
    "tests/check_interface.py",
    "src/system_template.py",
    "src/gliner_system.py",
    "evaluation/metrics_template.py",
    "tests/test_contracts.py",
)
extracted: dict[Path, str] = {}
for target in targets:
    pattern = rf"(?ms)^### `{re.escape(target)}`\n\n```python\n(.*?)^```[ \t]*$"
    matches = re.findall(pattern, spec)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one block for {target}, found {len(matches)}")
    path = Path(target)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    extracted[path] = matches[0]
for path, source in extracted.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
```


### Verifying the extracted files

Install the development tools pytest, mypy and Ruff in the environment before running these commands. The checks themselves need no network, model or GPU; GLiNER is imported lazily. These commands work without a `pyproject.toml`.

Windows PowerShell:

```powershell
$env:MYPYPATH = (Resolve-Path src).Path
python -m mypy --strict --explicit-package-bases extract_interface.py src/contracts.py src/system_template.py src/gliner_system.py evaluation/metrics_template.py tests/check_interface.py tests/test_contracts.py
python -m ruff check --isolated --config "line-length = 100" --select E,F,W,I,B,UP,N,SIM,RUF,PT extract_interface.py src/contracts.py src/system_template.py src/gliner_system.py evaluation/metrics_template.py tests/check_interface.py tests/test_contracts.py
python -m pytest -q tests/test_contracts.py
```

Colab or another Unix-like shell:

```sh
export MYPYPATH="$PWD/src"
python -m mypy --strict --explicit-package-bases extract_interface.py src/contracts.py src/system_template.py src/gliner_system.py evaluation/metrics_template.py tests/check_interface.py tests/test_contracts.py
python -m ruff check --isolated --config "line-length = 100" --select E,F,W,I,B,UP,N,SIM,RUF,PT extract_interface.py src/contracts.py src/system_template.py src/gliner_system.py evaluation/metrics_template.py tests/check_interface.py tests/test_contracts.py
python -m pytest -q tests/test_contracts.py
```

### `tests/check_interface.py`

```python
"""Check Dandelion4 Message, transcript, Record, and Gold JSON files.

    python check_interface.py messages FILE [FILE ...]
    python check_interface.py transcripts FILE [FILE ...]
    python check_interface.py records FILE [--messages INPUT]
    python check_interface.py gold FILE [--messages INPUT]

A passing check does not verify semantic privacy or labeling correctness.
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, NoReturn

ACTIVITY_TYPES = {
    "household_outreach",
    "health_screening",
    "community_meeting",
    "health_referral",
    "other",
}
REDACTION_TYPES = {"NAME", "PHONE"}
STATUSES = {"pending", "not_applicable", "verified", "rejected"}
ID = re.compile(r"([a-z0-9]+)_([0-9]+)")
LEADING_WORD = re.compile(r"^(the|a|an|in|at|on|by)\s", re.IGNORECASE)
TEXT_FIELDS = ("location", "date_text", "event_name", "event_desc")
MESSAGE: dict[str, tuple[type, ...]] = {"id": (str,), "sender_id": (str,), "message": (str,)}
MESSAGE_OPTIONAL: dict[str, tuple[type, ...]] = {"source": (str,), "asr_text": (str,)}
RECORD: dict[str, tuple[type, ...]] = {
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
GOLD: dict[str, tuple[type, ...]] = {
    "id": (str,),
    "relevant": (bool,),
    "activity_type": (str, type(None)),
}
GOLD_OPTIONAL = {k: RECORD[k] for k in (*TEXT_FIELDS, "reported_count")}
GOLD_OPTIONAL["pii"] = (list,)


def reject_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)


def check_fields(
    obj: dict[str, Any],
    required: dict[str, tuple[type, ...]],
    optional: dict[str, tuple[type, ...]],
    where: str,
) -> list[str]:
    problems = [f"{where}: missing field {key!r}" for key in required if key not in obj]
    problems += [
        f"{where}: unknown field {key!r}"
        for key in obj
        if key not in required and key not in optional
    ]
    problems += [
        f"{where}: wrong type for {key!r}"
        for key, kinds in (required | optional).items()
        if key in obj and type(obj[key]) not in kinds
    ]
    return problems


def check_spans(spans: list[Any], where: str, gold: bool = False) -> list[str]:
    problems: list[str] = []
    offsets: list[tuple[int, int]] = []
    expected = {"type", "start", "end", "text"} if gold else {"type", "start", "end"}
    for index, span in enumerate(spans):
        label = f"{where}[{index}]"
        if not isinstance(span, dict) or set(span) != expected:
            problems.append(f"{label}: invalid span fields")
            continue
        kind, start, end = span["type"], span["start"], span["end"]
        if (
            not isinstance(kind, str)
            or kind not in REDACTION_TYPES
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or (gold and (not isinstance(span["text"], str) or len(span["text"]) != end - start))
        ):
            problems.append(f"{label}: invalid span values")
            continue
        offsets.append((start, end))
    if not gold and offsets != sorted(offsets):
        problems.append(f"{where}: spans are not sorted by start")
    last_end = 0
    for start, end in sorted(offsets):
        if start < last_end:
            problems.append(f"{where}: spans overlap")
        last_end = max(last_end, end)
    return problems


def redact(text: str, spans: list[dict[str, Any]]) -> str:
    for span in sorted(spans, key=lambda item: item["start"], reverse=True):
        text = text[: span["start"]] + f"[{span['type']}]" + text[span["end"] :]
    return text


def check_values(
    obj: dict[str, Any], where: str, text: str | None, defer_placeholders: bool = False
) -> list[str]:
    problems = []
    for key in TEXT_FIELDS:
        value = obj.get(key)
        if not isinstance(value, str):
            continue
        if (
            not value
            or value != value.strip()
            or LEADING_WORD.match(value)
            or (
                unicodedata.category(value[-1]).startswith("P")
                and not value.endswith(("[NAME]", "[PHONE]"))
            )
        ):
            problems.append(f"{where}: invalid copied value for {key}")
        elif (
            text is not None
            and not (defer_placeholders and ("[NAME]" in value or "[PHONE]" in value))
            and value.casefold() not in text.casefold()
        ):
            problems.append(f"{where}: {key} does not occur in the source text")
    return problems


def validate_file(kind: str, data: Any, messages: Any = None) -> list[str]:
    if kind not in ("messages", "transcripts", "records", "gold"):
        raise ValueError(f"unknown kind {kind!r}")
    if not isinstance(data, list):
        return ["top level must be an array"]
    problems: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    if messages is not None and kind in ("records", "gold"):
        reference_problems = validate_file("messages", messages)
        problems += [f"reference messages: {problem}" for problem in reference_problems]
        if not reference_problems:
            by_id = {message["id"]: message for message in messages}
    reference_valid = messages is not None and not problems
    ids: list[str] = []
    for index, obj in enumerate(data):
        where = f"item {index}"
        if not isinstance(obj, dict):
            problems.append(f"{where}: not an object")
            continue
        ident = obj.get("id")
        if isinstance(ident, str):
            ids.append(ident)
        if not isinstance(ident, str) or not ID.fullmatch(ident):
            problems.append(f"{where}: invalid id")
        message = by_id.get(ident) if isinstance(ident, str) else None
        if kind in ("messages", "transcripts"):
            problems += check_fields(obj, MESSAGE, MESSAGE_OPTIONAL, where)
            source = obj.get("source")
            if source is not None and source not in ("text", "audio"):
                problems.append(f"{where}: invalid source")
            if kind == "transcripts" and source != "audio":
                problems.append(f"{where}: transcript must set source to audio")
            if "asr_text" in obj and source != "audio":
                problems.append(f"{where}: asr_text requires source audio")
            continue
        record = kind == "records"
        problems += check_fields(
            obj, RECORD if record else GOLD, {} if record else GOLD_OPTIONAL, where
        )
        relevant, activity = obj.get("relevant"), obj.get("activity_type")
        if relevant is True and (not isinstance(activity, str) or activity not in ACTIVITY_TYPES):
            problems.append(f"{where}: relevant activity_type is invalid")
        if relevant is False and activity is not None:
            problems.append(f"{where}: irrelevant activity_type must be null")
        count = obj.get("reported_count")
        if type(count) is int and count < 0:
            problems.append(f"{where}: negative reported_count")
        original = message.get("message") if message is not None else None
        if record:
            status, source = obj.get("verification_status"), obj.get("source")
            if (
                not isinstance(status, str)
                or status not in STATUSES
                or (relevant is False) != (status == "not_applicable")
            ):
                problems.append(f"{where}: invalid verification_status")
            if source not in ("text", "audio"):
                problems.append(f"{where}: invalid source")
            confidence = obj.get("confidence")
            if (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and not 0 <= confidence <= 1
            ):
                problems.append(f"{where}: confidence outside 0..1")
            redacted, spans = obj.get("text_redacted"), obj.get("redactions")
            span_problems = (
                check_spans(spans, f"{where}.redactions") if isinstance(spans, list) else []
            )
            problems += span_problems
            problems += check_values(obj, where, redacted if isinstance(redacted, str) else None)
            if message is not None:
                if obj.get("sender_id") != message.get("sender_id"):
                    problems.append(f"{where}: sender_id differs from Message")
                if source != message.get("source", "text"):
                    problems.append(f"{where}: source differs from Message")
                if isinstance(spans, list) and not span_problems and isinstance(original, str):
                    if any(span["end"] > len(original) for span in spans):
                        problems.append(f"{where}: redaction exceeds Message")
                    elif redacted != redact(original, spans):
                        problems.append(f"{where}: text_redacted does not reconstruct")
            elif isinstance(spans, list) and isinstance(redacted, str):
                for kind_name in REDACTION_TYPES:
                    needed = sum(
                        1
                        for span in spans
                        if isinstance(span, dict) and span.get("type") == kind_name
                    )
                    if redacted.count(f"[{kind_name}]") < needed:
                        problems.append(f"{where}: too few {kind_name} placeholders")
        else:
            spans = obj.get("pii")
            span_problems = (
                check_spans(spans, f"{where}.pii", gold=True) if isinstance(spans, list) else []
            )
            problems += span_problems
            usable = isinstance(original, str) and isinstance(spans, list) and not span_problems
            if isinstance(original, str) and isinstance(spans, list) and usable:
                for span in spans:
                    if (
                        span["end"] > len(original)
                        or original[span["start"] : span["end"]] != span["text"]
                    ):
                        problems.append(f"{where}: gold span exceeds or does not match Message")
                        usable = False
            value_text = (
                redact(original, spans)
                if usable and isinstance(original, str) and isinstance(spans, list)
                else original
            )
            problems += check_values(obj, where, value_text, defer_placeholders=not usable)
    counts = Counter(ids)
    problems += [f"duplicate id {ident}" for ident, count in sorted(counts.items()) if count > 1]
    widths: dict[str, set[int]] = {}
    for ident in ids:
        match = ID.fullmatch(ident)
        if match:
            widths.setdefault(match.group(1), set()).add(len(match.group(2)))
    problems += [
        f"{prefix}: inconsistent id widths"
        for prefix, sizes in sorted(widths.items())
        if len(sizes) > 1
    ]
    if ids != sorted(ids):
        problems.append("objects are not sorted by id")
    if kind == "records" and reference_valid:
        problems += [f"no Record for {ident}" for ident in sorted(set(by_id) - set(ids))]
        problems += [f"Record has no Message: {ident}" for ident in sorted(set(ids) - set(by_id))]
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("messages", "transcripts", "records", "gold"))
    parser.add_argument("files", nargs="+")
    parser.add_argument("--messages")
    args = parser.parse_args(argv)
    if args.messages and args.kind not in ("records", "gold"):
        parser.error("--messages is only supported for records and gold")
    try:
        messages = load(args.messages) if args.messages else None
        all_problems: list[str] = []
        first_seen: dict[str, str] = {}
        all_widths: dict[str, set[int]] = {}
        for path in args.files:
            data = load(path)
            all_problems += [
                f"{path}: {problem}" for problem in validate_file(args.kind, data, messages)
            ]
            if args.kind == "messages" and isinstance(data, list):
                for obj in data:
                    ident = obj.get("id") if isinstance(obj, dict) else None
                    if isinstance(ident, str):
                        if ident in first_seen and first_seen[ident] != path:
                            all_problems.append(
                                f"{path}: id {ident} also used in {first_seen[ident]}"
                            )
                        first_seen.setdefault(ident, path)
                        match = ID.fullmatch(ident)
                        if match:
                            all_widths.setdefault(match.group(1), set()).add(len(match.group(2)))
        all_problems += [
            f"{prefix}: inconsistent id widths across files"
            for prefix, widths in sorted(all_widths.items())
            if len(widths) > 1
        ]
    except (OSError, ValueError, RecursionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for problem in all_problems[:50]:
        print("FAIL", problem)
    print(
        f"{len(all_problems)} problem(s)" + (" (first 50 shown)" if len(all_problems) > 50 else "")
    )
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

### `src/system_template.py`

```python
"""Shared command-line entry point for systems implementing the v2 contract."""

from __future__ import annotations

import argparse
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from contracts import load_messages, validate_file, write_json
else:
    _contracts = import_module(f"{__package__}.contracts" if __package__ else "contracts")
    load_messages = _contracts.load_messages
    validate_file = _contracts.validate_file
    write_json = _contracts.write_json


class System(Protocol):
    name: str

    def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one record per message, with the same IDs in the same order."""
        ...


def run_cli(system: System, argv: list[str] | None = None) -> None:
    """Read messages, run the system, validate all results, then write them."""
    if type(getattr(system, "name", None)) is not str or not system.name:
        raise TypeError("system.name must be a nonempty string")
    if not callable(getattr(system, "process_messages", None)):
        raise TypeError("system.process_messages must be callable")
    if argv is not None and (type(argv) is not list or any(type(arg) is not str for arg in argv)):
        raise TypeError("argv must be a list of strings or None")
    parser = argparse.ArgumentParser(description=f"Run the {system.name} system")
    parser.add_argument("input", nargs="?", default="examples/synthetic_messages.json")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if args.out is not None:
        output_path = Path(args.out)
    elif input_path.name == "synthetic_messages.json":
        output_path = Path("examples") / f"{system.name}_output.json"
    else:
        output_path = Path("examples") / f"{system.name}_output_{input_path.stem}.json"

    messages = [message.to_dict() for message in load_messages(input_path)]
    records = system.process_messages(deepcopy(messages))
    problems = validate_file("records", records, messages)
    if problems:
        raise ValueError("Invalid system output:\n" + "\n".join(problems))
    message_ids = [message["id"] for message in messages]
    record_ids = [record["id"] for record in records]
    if record_ids != message_ids:
        raise ValueError("System output IDs must match input IDs in the same order")
    write_json(output_path, [record for record in records])
```

### `src/gliner_system.py`

```python
"""GLiNER system skeleton for Interface Spec v2.

The provisional labels are Name, Date, Attendees, Location, EventName,
EventDesc and EventActivities, with threshold 0.5. The spec requires these
choices to be documented but does not prescribe a complete set or threshold.
Relevance, activity type, count and confidence are not implemented yet.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from system_template import run_cli
else:
    _module_name = f"{__package__}.system_template" if __package__ else "system_template"
    _template = import_module(_module_name)
    run_cli = _template.run_cli

MODEL_ID = "urchade/gliner_medium-v2.1"
LABELS = ("Name", "Date", "Attendees", "Location", "EventName", "EventDesc", "EventActivities")
THRESHOLD = 0.5
LABEL_TO_FIELD = {
    "Location": "location",
    "Date": "date_text",
    "EventName": "event_name",
    "EventDesc": "event_desc",
    "Name": "NAME",
}


class GLiNERSystem:
    """An unimplemented system with one lazy-loaded model per instance."""

    name = "gliner"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        labels: tuple[str, ...] = LABELS,
        threshold: float = THRESHOLD,
    ) -> None:
        if type(model_id) is not str or not model_id:
            raise ValueError("model_id must be a nonempty string")
        if type(labels) is not tuple or not labels or any(
            type(label) is not str or not label for label in labels
        ):
            raise ValueError("labels must be a nonempty tuple of nonempty strings")
        if (
            type(threshold) not in (int, float)
            or not 0 <= threshold <= 1
            or not math.isfinite(threshold)
        ):
            raise ValueError("threshold must be a finite number in 0..1")
        self.model_id = model_id
        self.labels = labels
        self.threshold = threshold
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Import GLiNER and load a cached model only on first use."""
        if self._model is None:
            gliner = import_module("gliner")
            self._model = gliner.GLiNER.from_pretrained(self.model_id, local_files_only=True)
        return self._model

    def _first_activity(self, text: str, entities: list[dict[str, Any]]) -> object:
        """Select the first activity in reading order (§4, §12).

        Select the first report of work done or underway, ignoring plans,
        schedules, invitations, reminders, questions, greetings and thanks.
        An activity is one kind of work at one place and time. All fields must
        describe this activity; do not borrow a value from another activity. If
        a field occurs twice use the first, except that the counting rule
        decides reported_count.
        """
        raise NotImplementedError

    def _copy_text_values(
        self, text: str, entities: list[dict[str, Any]], activity: object
    ) -> dict[str, str | None]:
        """Map Location, Date, EventName and EventDesc to their §12 fields.

        Copy only values of the first activity from the message, preserving
        its spelling; capitalization may differ. Remove leading the, a, an,
        in, at, on and by and trailing punctuation. Location is the name of a
        place. Use a placeholder for a redacted span, never an empty string;
        use null if absent or uncertain (§4, §5, §12).
        """
        raise NotImplementedError

    def _redactions(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find NAME and PHONE spans in the original message (§2, §5, §12).

        Include titles belonging to names and roles identifying one person;
        exclude articles and possessive 's. Name entities identify people;
        Attendees entities are redacted only if they name a specific person.
        Include phone numbers in any country or format. Exclude groups,
        counts, places, dates and organizations. Return non-overlapping spans
        sorted by start, with zero-based, end-exclusive message offsets.
        """
        raise NotImplementedError

    def _classify_and_count(
        self, text: str, entities: list[dict[str, Any]], activity: object
    ) -> dict[str, Any]:
        """Determine relevance, activity type, count and confidence (§4, §12).

        Relevance requires reported work done or underway; plans and questions
        are not relevant. Use null activity type exactly when irrelevant and
        other for relevant work outside the listed types. Count reached units,
        adding parts of a group, using the first of two units, the stated
        estimate, or the lower end of a range. Number words may yield null if
        conversion is unavailable. Values must describe the first activity;
        use null when unsupported or uncertain. Document the final confidence
        calculation in the module docstring.
        """
        raise NotImplementedError

    def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one Record per Message with matching IDs and order, or raise."""
        raise NotImplementedError


def process_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose the required module-level system interface."""
    return GLiNERSystem().process_messages(messages)


if __name__ == "__main__":
    run_cli(GLiNERSystem())
```

### `evaluation/metrics_template.py`

```python
"""Metric signatures and definitions for Interface Spec section 7.5.

Percentages use two decimal places, with exact ties rounded away from zero.
"""

from collections.abc import Sequence

JsonObject = dict[str, object]


def classification_accuracy(
    messages: Sequence[JsonObject],
    records: Sequence[JsonObject],
    gold: Sequence[JsonObject],
) -> float | None:
    """Return the percentage of scored messages classified correctly.

    The denominator is input messages with a gold entry. Both relevant and
    activity_type must equal gold; a missing record is wrong. Gold entries
    outside the input file and messages without gold are skipped. Return None
    when there are no scored messages.
    """
    raise NotImplementedError


def filter_precision(
    messages: Sequence[JsonObject],
    records: Sequence[JsonObject],
    gold: Sequence[JsonObject],
) -> float | None:
    """Return the percentage of filtered records whose gold is irrelevant.

    The denominator is scored records with relevant false; the numerator is
    those whose gold relevant is false. Skip input messages with no gold and
    gold outside the input file. Missing records do not enter this denominator.
    Return None if the denominator is zero.
    """
    raise NotImplementedError


def filter_recall(
    messages: Sequence[JsonObject],
    records: Sequence[JsonObject],
    gold: Sequence[JsonObject],
) -> float | None:
    """Return the percentage of gold-irrelevant messages filtered correctly.

    The denominator is input messages with a gold entry whose relevant is
    false; the numerator is those with a record whose relevant is false. A
    missing record is wrong. Skip messages with no gold and gold entries
    outside the input file. Return None if the denominator is zero.
    """
    raise NotImplementedError


def attribute_extraction_f1(
    messages: Sequence[JsonObject],
    records: Sequence[JsonObject],
    gold: Sequence[JsonObject],
) -> float | None:
    """Return micro F1 as a percentage for the four scored attribute fields.

    Consider location, reported_count, date_text and event_name for each input
    message with gold. Skip fields missing from gold and gold outside the input
    file; do not score event_desc. Compare strings case-insensitively after
    trimming and integers exactly. TP: non-null gold equals prediction. FP:
    non-null prediction differs from gold. FN: non-null gold differs from
    prediction. Wrong non-null values can count as both FP and FN. F1 is
    2TP / (2TP + FP + FN); return None for a zero denominator. For a missing
    record, treat every predicted field as null.
    """
    raise NotImplementedError


def pii_scrubbing_recall(
    messages: Sequence[JsonObject],
    records: Sequence[JsonObject],
    gold: Sequence[JsonObject],
    *,
    expected_transcript_pii: bool = False,
) -> float | None:
    """Return the percentage of scored gold PII spans completely covered.

    The denominator is gold spans in input messages with a gold pii field.
    The numerator is spans whose every character falls inside system redaction
    spans, regardless of type. Skip messages without gold or with pii missing,
    and skip an entire message if any gold span's text does not match the
    input text at its offsets. Skip audio messages unless expected_transcript_pii
    asserts that gold PII was labeled for this exact transcript input file.
    An empty pii list participates but adds no spans. A missing record covers
    no spans. Ignore gold outside the input file. Return None if there are no
    scored spans.
    """
    raise NotImplementedError


def over_redactions(
    messages: Sequence[JsonObject],
    records: Sequence[JsonObject],
    gold: Sequence[JsonObject],
    *,
    expected_transcript_pii: bool = False,
) -> int | None:
    """Count system redaction spans overlapping no gold span on scored messages.

    A scored message is an input message with a gold pii field whose every
    gold span matches the input text at its offsets. Skip messages with no gold
    or no pii field, and gold outside the input file. Skip audio unless
    expected_transcript_pii asserts that gold PII was labeled for this exact
    transcript input file. The count has no denominator; an empty gold pii list
    is scored and makes every system span an over-redaction. A missing record
    has no system spans to count. Return None if no messages were PII-scored.
    """
    raise NotImplementedError


def pii_skipped_messages(
    messages: Sequence[JsonObject],
    gold: Sequence[JsonObject],
    *,
    expected_transcript_pii: bool = False,
) -> int:
    """Count input messages with gold skipped from PII scoring.

    Count each message once if its gold has no pii field, if it is audio and
    transcript-specific labels are not asserted, or if any gold span is
    invalid or does not match the input text. A labeled empty pii list is not
    skipped. Messages without a gold entry are outside the scored set.
    """
    raise NotImplementedError


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float | None:
    """Return aggregate WER as a percentage over all clips.

    Lowercase text, delete straight and right-curly apostrophes, replace each
    run of non-ASCII-letter-or-digit characters with a space, then collapse
    whitespace. The caller passes raw asr_text as the hypothesis when present.
    Compute (S + D + I) / N over all clips, where S, D and I are word-level
    substitutions, deletions and insertions, and N is reference word count.
    Empty individual references contribute hypothesis words as insertions.
    Return None if the aggregate N is zero.
    """
    raise NotImplementedError
```

### `tests/test_contracts.py`

```python
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from contracts import build_record, load_records, validate_file, write_json
    from gliner_system import GLiNERSystem
    from system_template import run_cli
else:
    from src.contracts import build_record, load_records, validate_file, write_json
    from src.gliner_system import GLiNERSystem
    from src.system_template import run_cli

CHECKER = Path(__file__).with_name("check_interface.py")
MESSAGE: dict[str, Any] = {
    "id": "pii_001",
    "sender_id": "worker_03",
    "source": "text",
    "message": "Mary Wanjiku visited 12 households in Narok today. Call 0700 000 101 for details.",
}
RECORD: dict[str, Any] = {
    "id": "pii_001",
    "sender_id": "worker_03",
    "source": "text",
    "relevant": True,
    "activity_type": "household_outreach",
    "location": "Narok",
    "reported_count": 12,
    "date_text": "today",
    "event_name": None,
    "event_desc": None,
    "text_redacted": "[NAME] visited 12 households in Narok today. Call [PHONE] for details.",
    "redactions": [{"type": "NAME", "start": 0, "end": 12},
                   {"type": "PHONE", "start": 56, "end": 68}],
    "confidence": 0.87,
    "verification_status": "pending",
}
GOLD: dict[str, Any] = {
    "id": "pii_001",
    "relevant": True,
    "activity_type": "household_outreach",
    "location": "Narok",
    "reported_count": 12,
    "pii": [{"type": "NAME", "start": 0, "end": 12, "text": "Mary Wanjiku"},
            {"type": "PHONE", "start": 56, "end": 68, "text": "0700 000 101"}],
}


def checker(
    tmp_path: Path, kind: str, data: Any, messages: list[dict[str, Any]] | None = None
) -> subprocess.CompletedProcess[str]:
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, str(CHECKER), kind, str(data_file)]
    if messages is not None:
        messages_file = tmp_path / "messages.json"
        messages_file.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        command.extend(["--messages", str(messages_file)])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def assert_both_reject(
    tmp_path: Path, kind: str, data: Any, messages: list[dict[str, Any]] | None = None
) -> None:
    assert validate_file(kind, data, messages)
    result = checker(tmp_path, kind, data, messages)
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("messages", [{"id": "msg_016", "sender_id": "worker_04",
                       "message": "We visited 7 households in Narok today."}]),
        ("transcripts", [{"id": "msg_016", "sender_id": "worker_04", "source": "audio",
                          "message": "We visited seven households in Narak today."}]),
        ("records", [{"id": "msg_002", "sender_id": "worker_05", "source": "text",
                       "relevant": True, "activity_type": "health_screening",
                       "location": "Nakuru", "reported_count": 48, "date_text": None,
                       "event_name": None, "event_desc": None,
                       "text_redacted": "We screened 48 women in Nakuru.", "redactions": [],
                       "confidence": None, "verification_status": "pending"}]),
        ("records", [{"id": "msg_016", "sender_id": "worker_04", "source": "audio",
                       "relevant": True, "activity_type": "household_outreach",
                       "location": None, "reported_count": None, "date_text": None,
                       "event_name": None, "event_desc": None,
                       "text_redacted": "We visited seven households in Narak today.",
                       "redactions": [], "confidence": None, "verification_status": "pending"}]),
        ("records", [RECORD]),
        ("gold", [GOLD]),
    ],
)
def test_spec_examples(tmp_path: Path, kind: str, data: list[dict[str, Any]]) -> None:
    assert validate_file(kind, data) == []
    result = checker(tmp_path, kind, data)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("reported_count", True),
        ("activity_type", None),
        ("verification_status", "not_applicable"),
        ("confidence", 1.5),
        ("source", "Audio"),
        ("extra", "unknown"),
        ("redactions", [{"type": "NAME", "start": 0, "end": 12},
                        {"type": "PHONE", "start": 10, "end": 15}]),
        ("text_redacted", "[NAME] visited 12 households in Narok today. Call for details."),
        ("location", "the Narok"),
        ("location", "Narok."),
        ("location", "Nairobi"),
    ],
)
def test_bad_record_fields(tmp_path: Path, key: str, value: Any) -> None:
    record = deepcopy(RECORD)
    record[key] = value
    if key == "location" and value in {"the Narok", "Narok."}:
        record["text_redacted"] = record["text_redacted"].replace("Narok", value)
    assert_both_reject(tmp_path, "records", [record])


def test_irrelevant_record_with_activity(tmp_path: Path) -> None:
    record = deepcopy(RECORD)
    record["relevant"] = False
    record["verification_status"] = "not_applicable"
    assert_both_reject(tmp_path, "records", [record])


@pytest.mark.parametrize("ids", [["msg_002", "msg_001"], ["msg_01", "msg_1"], ["Msg_001"]])
def test_bad_ids(tmp_path: Path, ids: list[str]) -> None:
    messages = [{"id": ident, "sender_id": "worker_01", "message": "Hello."} for ident in ids]
    assert_both_reject(tmp_path, "messages", messages)


def test_privacy_requires_semantic_review(tmp_path: Path) -> None:
    record = deepcopy(RECORD)
    record.update(
        location="Mary", text_redacted="Mary visited 12 households in Narok today.", redactions=[]
    )
    result = checker(tmp_path, "records", [record])
    assert result.returncode == 0
    assert validate_file("records", [record]) == []


def test_literal_placeholder_is_not_a_redaction(tmp_path: Path) -> None:
    record = deepcopy(RECORD)
    record.update(
        text_redacted="We visited 12 households in Narok today and wrote [NAME].", redactions=[]
    )
    assert validate_file("records", [record]) == []
    result = checker(tmp_path, "records", [record])
    assert result.returncode == 0, result.stdout + result.stderr


def test_numeric_pattern_alone_is_not_a_phone_error(tmp_path: Path) -> None:
    record = deepcopy(RECORD)
    record.update(
        text_redacted="We visited 0700 000 101 households in Narok today.",
        reported_count=700000101,
        redactions=[],
    )
    assert validate_file("records", [record]) == []
    result = checker(tmp_path, "records", [record])
    assert result.returncode == 0, result.stdout + result.stderr


def test_wrong_offsets(tmp_path: Path) -> None:
    record = deepcopy(RECORD)
    record["redactions"][1]["start"] = 55
    assert_both_reject(tmp_path, "records", [record], [MESSAGE])


def test_missing_record(tmp_path: Path) -> None:
    assert_both_reject(tmp_path, "records", [], [MESSAGE])


def test_extra_record(tmp_path: Path) -> None:
    assert_both_reject(tmp_path, "records", [RECORD], [])


def test_wrong_gold_span_text(tmp_path: Path) -> None:
    gold = deepcopy(GOLD)
    gold["pii"][0]["text"] = "Wrong"
    assert_both_reject(tmp_path, "gold", [gold], [MESSAGE])


def test_build_record_round_trip(tmp_path: Path) -> None:
    record = build_record(
        MESSAGE, relevant=True, activity_type="household_outreach", location="Narok",
        reported_count=12, date_text="today", event_name=None, event_desc=None,
        redactions=RECORD["redactions"], confidence=0.87,
    )
    assert record == RECORD
    path = tmp_path / "records.json"
    write_json(path, [record])
    assert [item.to_dict() for item in load_records(path)] == [record]


class ExampleSystem:
    name = "example"

    def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [build_record(
            message, relevant=False, activity_type=None, location=None, reported_count=None,
            date_text=None, event_name=None, event_desc=None, redactions=[], confidence=None,
        ) for message in messages]


@pytest.mark.parametrize(
    ("input_name", "output_name"),
    [("synthetic_messages.json", "example_output.json"),
     ("other_messages.json", "example_output_other_messages.json")],
)
def test_cli_default_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_name: str, output_name: str
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("examples").mkdir()
    input_path = Path(input_name)
    input_path.write_text(json.dumps([{"id": "msg_001", "sender_id": "worker_01",
                                      "message": "Hello."}]), encoding="utf-8")
    run_cli(ExampleSystem(), [str(input_path)])
    assert (Path("examples") / output_name).is_file()


def test_cli_failure_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSystem(ExampleSystem):
        def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            raise ValueError("Cannot process message")

    monkeypatch.chdir(tmp_path)
    Path("examples").mkdir()
    input_path = Path("synthetic_messages.json")
    input_path.write_text(json.dumps([{"id": "msg_001", "sender_id": "worker_01",
                                      "message": "Hello."}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot process message"):
        run_cli(FailingSystem(), [str(input_path)])
    assert not (Path("examples") / "example_output.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""},
        {"model_id": None},
        {"labels": ()},
        {"labels": ["Name"]},
        {"labels": ("Name", None)},
        {"threshold": True},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": -0.1},
        {"threshold": 1.1},
    ],
)
def test_gliner_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        GLiNERSystem(**kwargs)


def test_gliner_loads_once_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    fake_gliner = ModuleType("gliner")

    class FakeModel:
        @staticmethod
        def from_pretrained(model_id: str, *, local_files_only: bool) -> object:
            calls.append((model_id, local_files_only))
            return object()

    fake_gliner.__dict__["GLiNER"] = FakeModel
    monkeypatch.setitem(sys.modules, "gliner", fake_gliner)
    system = GLiNERSystem()
    assert calls == []
    first = system._load_model()
    assert system._load_model() is first
    assert calls == [("urchade/gliner_medium-v2.1", True)]


@pytest.mark.parametrize("argv", ["input.json", [1], 12])
def test_cli_rejects_invalid_arguments(argv: Any) -> None:
    with pytest.raises(TypeError):
        run_cli(ExampleSystem(), argv)


@pytest.mark.parametrize(
    "system",
    [SimpleNamespace(name="", process_messages=lambda messages: []), SimpleNamespace(name="bad")],
)
def test_cli_rejects_invalid_system(system: Any) -> None:
    with pytest.raises(TypeError):
        run_cli(system, [])
```
