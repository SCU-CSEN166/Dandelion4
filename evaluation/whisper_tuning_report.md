# Tuned Whisper Model Performance & Test Datasets Report

**Default Pipeline ASR Model:** `models/whisper-large-v3-turbo-v3` (3-stage iterative LoRA fine-tune on `openai/whisper-large-v3-turbo` using real human African-accented English speech from [`robello2/mini_afrispeech`](https://huggingface.co/datasets/robello2/mini_afrispeech), paired with post-ASR cleanup in [`src/asr/clean_transcripts.py`](../src/asr/clean_transcripts.py)).

---

## 1. Fine-Tuning Progression (Training Lineage)

| Stage / Model ID | Base Model | Training Data (`robello2/mini_afrispeech`) | LoRA & Training Config | Validation Note |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `openai/whisper-large-v3-turbo` | Zero-shot (untuned) | — | Reference baseline |
| **`cv/fold_1`** | `openai/whisper-large-v3-turbo` | Fold 1 of 5 (`n_train=800`, `n_val=200`, `<=30s`) | `r=16`, `alpha=32`, targets: `q_proj, v_proj`, `2` epochs | Val WER: `23.57% -> 13.63%` (`-9.94` pts) |
| **`v2`** (`whisper-large-v3-turbo-v2`) | `models/whisper-large-v3-turbo-cv/fold_1` | `n_train=1,500` (`seed=999`, `<=30s`), `n_eval=300` (`eval_seed=1234`) | `r=16`, `alpha=32`, targets: `q_proj, v_proj`, `2` epochs (`376` steps), `lr=3e-4` cosine | `eval_loss=0.4342`; `test_300`: `12.05%` |
| **`v3`** (`whisper-large-v3-turbo-v3`) | `models/whisper-large-v3-turbo-v2` | `n_train=4,500` (`seed=2026`, `<=30s`, ALL-CAPS normalized via `--clean-transcripts`), `n_eval=300` (`eval_seed=1234`) | `r=32`, `alpha=64`, targets: `q_proj, k_proj, v_proj, out_proj, fc1, fc2`, `lr=1e-4` cosine, merged best Epoch-1 checkpoint (`checkpoint-563`) | `eval_loss=0.4161`; best overall balance across AfriSpeech, FLEURS, and Dandelion audio |

*(Note: Training was performed on `cuda:0` using `batch_size=1`, `grad_accum=8`, max clip duration `30.0 s` via [`src/training/finetune_whisper.py`](../src/training/finetune_whisper.py). Post-ASR seq2seq T5 typo correction with `willwade/t5-small-spoken-typo` was also ablated on `v3 beam-3` (`test_300_seed1234`) and rejected because it degraded WER from `11.58%` to `12.34%` regex / `10.78%` to `12.20%` `fair_norm`.)*

---

## 2. Summary Performance Table Across Benchmarks

### A. Real Human Speech Benchmarks (`robello2/mini_afrispeech` & `google/fleurs`)

All metrics are **corpus Word Error Rate (WER %)** — lower is better.

| Benchmark Split | Clips (`n`) | Baseline (`openai/whisper-large-v3-turbo`, greedy) | `fold_1` (greedy) | `v2` (greedy) | `v2` (beam-3) | `v3` Greedy (`regex_wer`) | `v3` Greedy (`fair_norm`) | `v3` Beam-3 (`regex_wer`) | `v3` Beam-3 (`fair_norm`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mini_afrispeech` `test_300_seed1234`** | `300` | `22.01%` | `12.55%` | `12.05%` | `11.88%` | **`11.90%`** | **`11.20%`** | **`11.58%`** | **`10.78%`** |
| **`mini_afrispeech` `test_330_balanced`** | `330` | `37.04%` | — | `16.48%` | — | **`15.73%`** | **`14.73%`** | **`15.30%`** | **`14.32%`** |
| **`google/fleurs` `en_us` (`100`-clip subset)** | `100` | `4.26%` | `4.21%` | `4.31%` | `4.35%` | — | — | — | — |
| **`google/fleurs` `en_us` (full test split)** | `647` | `4.97%` | — | — | — | **`5.14%`** | **`5.03%`** | **`5.00%`** | **`4.95%`** |

> **Normalization note:**
> - `regex_wer` matches [`evaluation/eval_wer.py`](../evaluation/eval_wer.py) (lowercase, strip apostrophes, replace non-alphanumerics with spaces).
> - `fair_norm` fixes missing-apostrophe contractions in AfriSpeech ground truth (e.g., `didnt` -> `didn't`) and applies Whisper's official `EnglishTextNormalizer` to both reference and hypothesis (`whisper_norm_wer` for `v3` was `11.38%` greedy / `10.96%` beam-3 on `test_300_seed1234`, `14.95%` / `14.54%` on `test_330_balanced`, and `5.29%` / `5.21%` on `fleurs_all`).
> - `eval_final.py` (`batch_size=8`) vs sequential `eval_v3.py` (`diag_eval.json`, `batch_size=1`) differed by <= `0.06` pts (`diag_eval.json` `regex_wer`: `11.94%` greedy / `11.61%` beam-3 on `test_300`, `15.73%` / `15.36%` on `test_330_balanced`, `5.13%` / `5.02%` on `fleurs_all`).

#### Domain & Accent Breakdown on `robello2/mini_afrispeech`

1. **`test_300_seed1234` (`300` clips: `138` clinical, `162` general; `290` Swahili, `7` Igbo, `3` Hausa)**

| Slice | `n` | Baseline (`regex`) | `fold_1` (`regex`) | `v2` Greedy (`regex`) | `v2` Beam-3 (`regex`) | `v3` Greedy (`fair_norm`) | `v3` Beam-3 (`fair_norm`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Domain: Clinical** | `138` | `25.10%` | `14.03%` | `13.27%` | `13.31%` | `12.64%` | **`11.93%`** |
| **Domain: General** | `162` | `19.97%` | `11.57%` | `11.25%` | `10.94%` | `10.24%` | **`10.02%`** |
| **Accent: Swahili (Kenyan/EA)** | `290` | `22.14%` | `12.66%` | `12.08%` | `11.88%` | `11.29%` | **`10.90%`** |
| **Accent: Igbo** | `7` | `15.84%` | `10.89%` | `13.86%` | `14.85%` | `6.93%` | **`6.93%`** |
| **Accent: Hausa** | `3` | `21.21%` | `0.00%` | `3.03%` | `3.03%` | `9.09%` | **`3.03%`** |

2. **`test_330_balanced` (`330` clips: `180` clinical, `150` general; `120` Yoruba, `92` Swahili, `60` Igbo, `58` Hausa)**

| Slice | `n` | Baseline (`regex`) | `v2` Greedy (`regex`) | `v3` Greedy (`fair_norm`) | `v3` Beam-3 (`fair_norm`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Domain: Clinical** | `180` | `50.85%` | `19.99%` | `17.31%` | **`16.79%`** |
| **Domain: General** | `150` | `24.10%` | `13.19%` | `12.31%` | **`12.00%`** |
| **Accent: Swahili** | `92` | `21.14%` | `11.80%` | `11.54%` | **`10.80%`** |
| **Accent: Igbo** | `60` | `26.15%` | `15.83%` | `14.20%` | **`13.47%`** |
| **Accent: Hausa** | `58` | `93.66%` | `19.70%` | **`15.52%`** | `15.66%` |
| **Accent: Yoruba** | `120` | `34.36%` | `19.79%` | `17.60%` | **`17.43%`** |

---

### B. Dandelion Kenyan-Accent Synthetic Benchmark (`examples/audio_manifest.json`, `90` clips)

Evaluated using [`evaluation/eval_wer.py`](../evaluation/eval_wer.py) with greedy decoding (`num_beams=1`, `do_sample=False`). The text-only extractor baseline on `original_text` (`30` messages) is **`90.00%`**. Target thresholds: **WER < 10%** and **downstream baseline accuracy drop <= 5.0 pts**.

| Model & Mode | Condition | WER % | Asilia Voice WER % | Chilemba Voice WER % | Count Field Acc % (`n=21`) | Location Field Acc % (`n=24`) | Downstream Baseline Acc % | Acc Drop (pts) | Targets Met? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned Baseline** (commit `34f944f`) | `clean` | `6.07%` | `4.59%` | `7.62%` | `76.19%` | `75.00%` | `83.33%` | `6.67` | WER ok / Drop fail |
| **Untuned Baseline** (commit `34f944f`) | `noisy` | `6.54%` | `4.59%` | `8.57%` | `76.19%` | `62.50%` | `90.00%` | `0.00` | Yes |
| **Untuned Baseline** (commit `34f944f`) | `phone` | `6.54%` | `4.59%` | `8.57%` | `76.19%` | `70.83%` | `83.33%` | `6.67` | WER ok / Drop fail |
| **`fold_1`** (raw ASR) | `clean` | `3.74%` | `2.75%` | `4.76%` | `90.48%` | `79.17%` | `86.67%` | `3.33` | Yes |
| **`fold_1`** (raw ASR) | `noisy` | `4.21%` | `4.59%` | `3.81%` | `90.48%` | `70.83%` | `90.00%` | `0.00` | Yes |
| **`fold_1`** (raw ASR) | `phone` | `3.27%` | `2.75%` | `3.81%` | `95.24%` | `79.17%` | `86.67%` | `3.33` | Yes |
| **`v2`** (raw ASR) | `clean` | `3.74%` | — | — | — | — | `86.67%` | `3.33` | Yes |
| **`v2`** (raw ASR) | `noisy` | `5.61%` | — | — | — | — | `86.67%` | `3.33` | Yes |
| **`v2`** (raw ASR) | `phone` | `3.27%` | — | — | — | — | `86.67%` | `3.33` | Yes |
| **`v3` (raw ASR, before cleaning)** | `clean` | `4.67%` | `3.67%` | `5.71%` | `80.95%` | `79.17%` | `86.67%` | `3.33` | Yes |
| **`v3` (raw ASR, before cleaning)** | `noisy` | `4.67%` | `3.67%` | `5.71%` | `76.19%` | `79.17%` | `90.00%` | `0.00` | Yes |
| **`v3` (raw ASR, before cleaning)** | `phone` | `4.21%` | `3.67%` | `4.76%` | `76.19%` | `83.33%` | `90.00%` | `0.00` | Yes |
| **`v3` + `clean_transcripts` (committed [`wer_results.json`](../evaluation/wer_results.json))** | `clean` | **`4.67%`** | `3.67%` | `5.71%` | **`100.00%`** | **`95.83%`** | `86.67%` | **`3.33`** | **Yes** |
| **`v3` + `clean_transcripts` (committed [`wer_results.json`](../evaluation/wer_results.json))** | `noisy` | **`4.67%`** | `3.67%` | `5.71%` | **`100.00%`** | **`95.83%`** | **`90.00%`** | **`0.00`** | **Yes** |
| **`v3` + `clean_transcripts` (committed [`wer_results.json`](../evaluation/wer_results.json))** | `phone` | **`4.21%`** | `3.67%` | `4.76%` | **`100.00%`** | **`100.00%`** | **`90.00%`** | **`0.00`** | **Yes** |

---

### C. Dandelion Hidden Test Audio (`demo/inputs/dandelion_audio.zip`, `10` clips)

Evaluated with `models/whisper-large-v3-turbo-v3` (greedy decoding, [`demo/outputs/demo_metrics.json`](../demo/outputs/demo_metrics.json)):
- **Raw Whisper ASR WER (`asr_text` vs expected `transcript`):** **`9.22%`**
- **Cleaned Transcript WER (`message` after [`clean_transcripts.py`](../src/asr/clean_transcripts.py) vs expected `transcript`):** **`10.60%`** *(higher only because `clean_transcripts` converts spelled-out number words such as `"forty-two"` and `"six"` to digits `"42"` and `"6"`, whereas the expected `transcript` column uses spelled-out words)*

---

## 3. Details of Test Datasets

| Dataset / Benchmark | Source & Config | Size & Duration | Selection / Sampling Method | Domains, Accents & Acoustic Conditions |
| :--- | :--- | :--- | :--- | :--- |
| **`robello2/mini_afrispeech` (`test_300_seed1234`)** | Hugging Face Parquet (`default`, `split="test"`), `16 kHz` mono | `300` real human speech clips | Shards deterministically shuffled with `seed=1234`; first `300` valid non-empty clips `<= 30.0 s` | **Domains:** `138` clinical, `162` general.<br>**Accents:** `290` Swahili (Kenyan/East African), `7` Igbo, `3` Hausa. |
| **`robello2/mini_afrispeech` (`test_330_balanced`)** | Hugging Face Parquet (`default`, `split="test"` across all `11` `data/test-*.parquet` shards), `16 kHz` mono | `330` real human speech clips | First `30` valid `<= 30.0 s` clips sampled from each of the `11` sorted test shards (`11 x 30 = 330`) | **Domains:** `180` clinical, `150` general.<br>**Accents:** `120` Yoruba, `92` Swahili, `60` Igbo, `58` Hausa. |
| **`robello2/mini_afrispeech` (CV Fold 1 val)** | Hugging Face Parquet (`default`, `split="train"`), `16 kHz` mono | `200` real human speech clips (`800` train) | 5-fold CV over `1,000` shuffled `<= 30.0 s` train clips | Mixed Pan-African clinical and general English speech. |
| **`google/fleurs` (`en_us`, `test`)** | Hugging Face (`en_us`, `split="test"`), saved locally as `16 kHz` WAV via [`src/training/fetch_fleurs.py`](../src/training/fetch_fleurs.py) & [`examples/fleurs_manifest.json`](../examples/fleurs_manifest.json) | `647` full test clips (`6,387.9 s` / `~1.77 h`); also `100`-clip subset (`953.9 s`) | Full US English test split (non-regression check for standard English) | Standard US English (`en_us`); real human speech across varied general topics. |
| **Dandelion Synthetic Benchmark** | [`examples/synthetic_messages.json`](../examples/synthetic_messages.json) rendered by [`src/training/make_audio.py`](../src/training/make_audio.py) (`edge-tts` `7.3.1`, [`examples/audio_manifest.json`](../examples/audio_manifest.json)) | `30` messages x `3` conditions = `90` clips (`486.12 s` total; `162.0 s` per condition) | Deterministic TTS generation (`base_seed=1234`, `1.0 s` silence pad) with regex extraction targets (`count` `n=21`, `location` `n=24`) | **Voices:** Kenyan English neural voices `en-KE-AsiliaNeural` & `en-KE-ChilembaNeural`.<br>**Conditions:** `clean` (`16 kHz`), `noisy` (`10.0 dB` SNR pink noise), `phone` (`8 kHz` downsampled telephony bandwidth). |
| **Dandelion Hidden Test Audio** | [`demo/inputs/dandelion_audio.zip`](../demo/inputs/dandelion_audio.zip) (`msg01.mp4`–`msg10.mp4`), evaluated against [`demo/expected/hidden_test_outputs.csv`](../demo/expected/hidden_test_outputs.csv) | `10` `.mp4` voice notes (`373.2 s` total audio; decoded via PyAV at `16 kHz`) | Complete hidden test submission set | Kenyan community health worker (CHW) event reports (`8` events) and group chit-chat (`2` non-events) containing Kenyan names, towns/dispensaries, phone numbers, and two duplicate pairs. |
