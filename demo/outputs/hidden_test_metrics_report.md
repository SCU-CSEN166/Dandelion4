# Dandelion Hidden Test Evaluation Report

Evaluation of [`demo/outputs/predicted_outputs.csv`](predicted_outputs.csv) against [`demo/expected/hidden_test_outputs.csv`](../expected/hidden_test_outputs.csv), reported strictly according to [`demo/inputs/Dandelion Hidden Test.pdf`](../inputs/Dandelion%20Hidden%20Test.pdf).

> **Run Context:** End-to-end execution via [`demo/run_demo.py`](../run_demo.py) on [`demo/inputs/dandelion_audio.zip`](../inputs/dandelion_audio.zip) (`10` `.mp4` messages, `373.2 s` total audio; `6.64 s` Whisper v3-large-turbo LoRA ASR on `cuda:0`, `22.01 s` total pipeline runtime). Saved metrics JSON: [`demo/outputs/demo_metrics.json`](demo_metrics.json).

---

## 1. Relevance (`Event` as the positive class)

| Metric | Score | Exact Counts |
| :--- | :---: | :---: |
| **Accuracy** | **100.00%** | `10 / 10` |
| **Precision** | **100.00%** | `8 / 8` |
| **Recall** | **100.00%** | `8 / 8` |
| **F1 Score** | **100.00%** | — |

---

## 2. Activity Type & Participant Count (`8` Event messages)

| Metric | Score | Exact Counts | Notes |
| :--- | :---: | :---: | :--- |
| **Strict Exact Match (both `activity_type` & `participants`)** | **37.50%** | `3 / 8` | Messages `1`, `7`, `9` |
| **Participant Count Exact Match** | **100.00%** | `8 / 8` | `42, 42, 25, 60, 35, 28, 18, 28` |
| **Activity Type Exact Match** | **37.50%** | `3 / 8` | — |

### Per-Event `activity_type` Breakdown

| Msg ID | Predicted `activity_type` | Expected `activity_type` | Match Status |
| :---: | :--- | :--- | :--- |
| `1` | `Family planning outreach` | `Family planning outreach` | Exact match |
| `2` | `Family planning` | `Family planning outreach` | Paraphrase / shorter phrasing |
| `4` | `Table banking` | `Women's group table banking (savings) meeting` | Paraphrase / shorter phrasing |
| `5` | `Menstrual hygiene session` | `Menstrual hygiene session and sanitary pad distribution` | Paraphrase / shorter phrasing |
| `6` | `Community dialogue on gender-based violence` | `Gender-based violence awareness dialogue` | Paraphrase / different wording |
| `7` | `Cervical cancer screening camp` | `Cervical cancer screening camp` | Exact match |
| `9` | `Kitchen garden training` | `Kitchen garden training` | Exact match |
| `10` | `Health screening` | `Cervical cancer screening camp` | Coarser label |

---

## 3. Anonymization (`[NAME]`, `[LOCATION]`, `[PHONE]`)

### Share of Private Details Removed
- **Slot-level placeholder recall:** **97.83% (`45 / 46`)** across all private placeholder slots in `expected_anonymized_text`.
- **Unique `pii_items` recall:** **97.78% (`44 / 45`)** across unique `pii_items`.
- **Only miss (`1`):** In Msg `4`, the women's group name **`Tumaini`** was mistranscribed by Whisper ASR as **`"Too Many"`** (`"Too Many Women's group meeting"`), so NER did not redact that slot *(though the literal name `Tumaini` was never exposed)*.

### Non-Private Words Masked (Over-Masking)
- **False-positive non-private entity (`1` span / `2` words):**
  - Msg `6` masked **`"chiefs camp"`** as an extra `[LOCATION]` (`"at the [LOCATION] in [LOCATION]"`).
- **Boundary-spill words absorbed into valid PII spans (`14` words):** honorifics/context words adjacent to real names or locations that `expected_anonymized_text` left unmasked:
  - Msg `1`: `dispensary`, `nurse`, `Sister`
  - Msg `3`: `here`
  - Msg `4`: `Tim` *(ASR transcription of `team`)*, `Mama`
  - Msg `5`: `Mrs.`
  - Msg `6`: `Corporal` *(plus ASR fused `"it's Mercy Wambui"` into one token `Ditsmasiwambui`, redacted as `[NAME]`)*
  - Msg `7`: `Sister`, `Mrs`
  - Msg `8`: `Mama`
  - Msg `9`: `reporting`, `Mr`
  - Msg `10`: `Sister`
- **Total non-private words masked:** **`16` words** (`2` false-positive words + `14` adjacent boundary/title words, excluding the fused ASR token `Ditsmasiwambui`).

---

## 4. Duplicates

| Metric | Score | Exact Counts | Details |
| :--- | :---: | :---: | :--- |
| **Duplicate Status Accuracy** | **100.00%** | `10 / 10` | `Msg 2 -> 1`, `Msg 10 -> 7`, and remaining `8` messages -> `none` |
| **Duplication F1** | **100.00%** | — | Zero false-positive or false-negative duplicate links |

---

## 5. End-to-End ASR Context Metrics

From [`demo/outputs/demo_metrics.json`](demo_metrics.json) comparing [`demo/outputs/predicted_transcripts.json`](predicted_transcripts.json) to the `transcript` column in [`demo/expected/hidden_test_outputs.csv`](../expected/hidden_test_outputs.csv):
- **Raw ASR WER (`asr_raw_wer_pct`):** **`9.22%`**
- **Cleaned ASR WER (`asr_clean_wer_pct`):** **`10.60%`** *(higher after cleaning because [`clean_transcripts.py`](../../src/asr/clean_transcripts.py) normalizes spelled-out numbers like `"forty-two"` and `"six"` to digits `"42"` and `"6"`, while expected `transcript` spells them as words)*
