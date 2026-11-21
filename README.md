# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D4<br>
**Challenge:** Dandelion Africa<br>
**Team members:** Terence Zhou, Teresa Wei, Eric Wang<br>
**Repository:** https://github.com/SCU-CSEN166/Dandelion4<br>
**Last updated:** 10/30/2026

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
Since the Kenyan data is unavailable, our data must be out sourced. We will create synthetic data using AI. This data will consist of different audio messages and text messages in English. Sensitive info of personal lives will automatically be made anonymous. 

### Known limitations
Since it cannot process videos and images it may be inaccurate for real life conversations in Kenya. Audio recording turned to text may not also be 100% accurate. 

---

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A full readme describing our approach, our responsibilities, goals for each week and a plan for synthesizing data. 

### 1. Problem formulation and social impact
- **Problem:** We need to develop a way for WhatsApp group exports into to be made into structured records of  activities and outcomes.
- **Intended users:** Dandelion Africa program managers and volunteers
- **Inputs and outputs:** Input will be audio and transcribed audio as well as messages, output will consist of a structured record of events, plans and notes of the meetings. 

> **Input:** Hello, everyone! My name is Terence, and I worked with a group of 2 people on the TTP4SI project in Heafey 106 on 10/30/2026. This week, we worked on the README together.<br><br>
> **Output:** { "name":[NAME], "date":"10/30/2026", "members":3, "location":"Heafey 106", "activityName":"TTP4SI project", "activityDesc":"working on README" }

- **Success criteria:** finding relevant messages and structuring them in a readable manner 
- **Constraints and risks:** unable to use images or videos, keeping info anonymous, model is english only based

### 2. Proposed technical approach
- **Pipeline:** have transcribed data and put new data into JSON form
- **Baseline(s):** have audio/text, transcribe data, put data through model, recompile all data
- **Proposed models/tools:**
  - [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) — automatic speech recognition (ASR); processes audio messages
  - [`autotrust/JEV-35B`](https://huggingface.co/autotrust/JEV-35B) — text classification; detects important JSON entries & sensitive content
  - [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1) - text generation; produces JSON database
- **Alternative approach:** find more specialized models

### 3. Proposed data
| Dataset/source | Link or access instructions | Purpose | Size / languages | Privacy or usage considerations |
|---|---|---|---|---|
| HuggingFace | [espnet/yodas3](https://huggingface.co/datasets/espnet/yodas3) | Training | 55.7 TB / 100+ languages | asks for citations |
| HuggingFace | [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | Training | 110 MB / English | no resharing the validation or test set, prevents scraping |
| HuggingFace | [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) | Training | 4.5 TB / 6 languages | asks for citations |
| HuggingFace | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | Training | 878 GB / 100+ languages | asks for citations |

- **Data preparation:** compilation, anonymization, cleaning, recompilation
- **Data limitations:** audio-to-text may be noisy, data related to sensitive info may be removed

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| Word Error Rate (WER) | audio-to-text accuracy | (S + D + A) / N (substitutions, deletions, additions, total word count) | WER < 10% |
| PII Scrubbing Recall | sensitive info redaction | redactions / flagged content | PSR > 95% |
| Attribute Extraction F1 | database entry info accuracy | F1 score vs. JSON keys | F1 > 95% |
| Filter Precision | irrelevant messages removal | removed noise / flagged noise | FP > 95% |

- **Evaluation examples:** checks over database entry, produces confidence level and appends it to entry
- **Human verification:** team will check failure cases (eg. low confidence), ideally manually, or using another more specialized script
- **Failure cases to watch:** incorrect audio-to-text, subtle sensitive information in calls or text

### 5. Team responsibilities
| Member | Primary responsibility | Specific Weekly contribution | Planned next task |
|---|---|---|---|
| Terence Zhou | Model Structure | created the repository, worked on the README | programming, filtering, classification |
| Teresa Wei | Data (Re)Compliation | worked on the README | compile data from HuggingFace |
| Eric Wang | Postprocessing Data | N/A | anonymization, info extraction |
| Everyone | Core Tasks | N/A | evaluation, testing, integration |

### 6. Technical Challenges
- Informal language, spelling variations, and incomplete messages.
- Audio messages and different English accents.
- Messages describing multiple events.
- Multiple messages describing the same activity.
- Missing or ambiguous dates and locations.
- Preventing hallucinated or unsupported information.
- Balancing privacy protection with preserving useful activity data.

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status
- **What runs end-to-end now?**
1. text generation
    - generated 50+ WhatsApp messages related to community healthcare, activities, meetings, community activities, and miscellaneous conversations
    - produced JSON files that can be used to develop our baseline
2. text parsing
    - classified (most) prompted keywords in text
    - redacted sensitive information such as names<br><br>
- **How to run it:**
1. text generation
    - open [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model in Google Colab / VS Code
    - import Python dependencies (python3 -m pip install torch transformers)
    - run the model: generate WhatsApp messages, and save them to a JSON (python3 src/generate_data.py)
2. text parsing
    - open [`urchade/gliner_medium-v2.1`](https://huggingface.co/urchade/gliner_medium-v2.1) model in Google Colab / VS Code
    - import Python dependencies (!pip install gliner)
    - create input string (in "text" variable)
    - create text classes (in "labels" array)
    - input custom Python code for redaction process (see bottom of [`file`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week6_gliner_medium_v2_1.ipynb))
    - run the model: classify and redact text<br><br>
- **Model(s) and data actually used:**
1. [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct): pretrained language model
    - data used: generated from Python script
    - stored generated data in JSON (50 messages)
2. [`urchade/gliner_medium-v2.1]`(https://huggingface.co/urchade/gliner_medium-v2.1)
    - data used: custom-made strings, SCU emails
    - change of plans from using [`autotrust/JEV-35B`](https://huggingface.co/autotrust/JEV-35B), not a very specialized model<br><br>

- **Example 1: Text Generation**
> ### **<< Message Generation >>**<br>
> **Output:** We had a meeting with the clinic manager to discuss the upcoming health screening<br>

> ### **<< JSON Creation >>**<br>
> **Input:** We visited {n} households in {loc} today<br>
> **Output:**<br>
> {<br>
>   "id": "msg_001",<br>
>   "sender_id": "worker_01",<br>
>   "message": "We visited 25 households in Nakuru today."<br>
> }<br>

- **Example 2: Text Processing**
> **Full List of Inputs/Outputs:** [`here`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week6_gliner_medium_v2_1.ipynb)<br>

> ### **<< Message Redaction >>**<br>
> **Input:** Hi team, please RSVP for the Annual Tech Gala by this Friday. The event takes place on November 12th at the Grand Hotel Ballroom. CEO Melon Usk, the VP of Sales, and all regional managers will be there. We will be celebrating our yearly milestones and handing out leadership awards.<br><br>
> **Output:** Hi team, please RSVP for the Annual Tech Gala by this Friday. The event takes place on November 12th at the Grand Hotel Ballroom. [REDACTED], the [REDACTED], and all [REDACTED] will be there. We will be celebrating our yearly milestones and handing out leadership awards.<br>

> ### **<< Message Classification >>**<br>
> **Input:** Hi team, please RSVP for the Annual Tech Gala by this Friday. The event takes place on November 12th at the Grand Hotel Ballroom. CEO Melon Usk, the VP of Sales, and all regional managers will be there. We will be celebrating our yearly milestones and handing out leadership awards.<br><br>
> **Output:**<br>
> ----- REDACTED LABELS -----<br>
> EventName: Annual Tech Gala<br>
> Date: Friday<br>
> Date: November 12th<br>
> Location: Grand Hotel Ballroom<br>
> Name: [REDACTED]<br>
> Name: [REDACTED]<br>
> Attendees: [REDACTED]<br>

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| Python generates data | Yes | synthetic messages in JSON | successfully generated repeatable activity reports and other messages|
| generate test audio with edge-TTS | Yes | 90 audio clips | model can make clean/noisy/phone TTS audio |
| transcribing audio with Whisper | Yes | transcripts in JSON | model sometimes misheard names (eg. Narok, Kajiado) / improve accuracy |
| running baseline on transcripts | Yes | WER results in JSON | model sometimes makes transcription mistakes (eg. "screen" for "screened") / change results and improve accuracy |
| text classificaiton | Yes | model output (see [file](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week6_gliner_medium_v2_1.ipynb)) | model can classify text with arbitrary keywords / change keywords for better/more accurate classification |
| text redaction | Yes | model output (see [file](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week6_gliner_medium_v2_1.ipynb)) | redact actual names instead of general text |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| Terence Zhou | Text Classification/Redaction | Worked | [cbd51fc](https://github.com/SCU-CSEN166/Dandelion4/commit/cbd51fceaeb775d6a25548ede1eb8493ade829f0) | improve accuracy, store data in JSON |
| Teresa Wei | Message Generation & JSON Storage | Worked | [ff7c120](https://github.com/SCU-CSEN166/Dandelion4/commit/ff7c120231ad5e361d3402c4e98313612dbd48f9) | improve message variations and quality |
| Eric Wang | Audio Generation w/ TTS, Audio Transcription, WER measurements | Worked | [b76dd21](https://github.com/SCU-CSEN166/Dandelion4/commit/b76dd21452414fa495e688e2842ff85eb682b0f2) | improve handling of place names and numbers written as words |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:**
1. text parsing
    - source: pre-generated WhatsApp logs from Google Gemini
    - size: 5 prompts, each 1-3 large paragraphs
    - selection: related to field reports, announcements, urgent updates, summaries
    - annotations: specific NER categories, sensitive information redaction, storing data in JSONs
    - access: see [`file`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week7_gliner_medium_v2_1.ipynb)
2. audio and transcription
    - source: sythetic messages from audio generation script, not recordings from real community workers
    - size: 30 source messages with 3 audio conditions (clean, noisy, and phone simulation, 90 total clips)
    - selection: community activities, irrelevant chat messages, Kenyan and English text to speech voices, adds on noise for one condition, lowers sample rate for phone simulation
    - annotations: original written messages reference transcripts, audio associates each clip with its message and generation condition
    - access: see [`src/training/make_audio.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/training/make_audio.py), [`examples/audio_manifest.json`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/examples/audio_manifest.json), [`src/asr/transcribe.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/asr/transcribe.py), and [`evaluation/wer_results.json`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/evaluation/wer_results.json)<br><br>

- **Metrics and rationale:**
1. text parsing
    - PII Scrubbing Recall: checks for redactions / flagged content rate (must be > 95%)
    - Attribute Extraction F1: checks for database entry label accuracy (must be > 95%)
2. audio and transcription
    - WER: measures substitutions, deletions, and insertions against given transcripts (must be < 10%)
    - Whisper downstream impact: shows if transcription mistakes affect later relevance/activity predictions (compares original text to transcript predictions)<br><br>

- **Comparison:** [Baseline versus current approach, if different]
1. text parsing: N/A (same as baseline approach)
    - PII scrubbing recall: model successfully found all sensitive items, does not guarantee all future redactions are processed
    - Attribute Extraction F1: model failed to accurately attribute text, missing/unchecked attributes are still a major limitation

| System / version | PII Scrubbing Recall | Attribute Extraction F1 | Notes |
|---|---:|---:|---|
| Baseline | 100% | 68% | miss 6, mislabel 2, total 25 |
| Current | 100% | 64.4% | miss 36, mislabel 6, total 118 |

2. Whisper transcription: N/A (baseline results: [`evaluation/wer_results.json`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/evaluation/wer_results.json))
    - model tested 30 synthetic messages for each audio condition
    - WER: model successfully did not make many errors on synthetic speech, but place name errors continued to be an issue
    - correct location tokens: model successfully achieved 87.5% accuracy for 40 message rule based text
    - activity classification: model successfully classified 90% of the 30 source messages, but small transcription errors can affect later activity classifications

| System / version | Clips | WER | Correct Location Tokens | Activity Classification |
|---|---:|---:|---:|---:|
| Clean | 30 | 6.07% | 75.00% | 83.33% |
| Noisy | 30 | 6.54% | 62.50% | 90.00% |
| Phone | 30 | 6.54% | 70.83% | 83.33% |

3. text generation (for baseline data): N/A
    - model generated 40 fictional text messages and corresponding expected output
    - used [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) to generate messages not in the Python template

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| text parsing | improved F1 score | worse F1 score | Failure | message truncation, bad labelling |
| GLiNER model setup | download and run a pretrained model for item extraction | loaded [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1), tested custome NER labels in the notebook | Success | created a working baseline for text extraction |
| PII redaction | replace sensitive informations with [REDACTED] | recorded 100% recall on examples | Partial Success | redactions require accurate item descriptions, false positives still a concern| 
| JSON organization of extracted fields | comprehensive, structured JSON ouput | partial JSON conversion, not completely structured | Partial Success | field mapping, handling duplicate items need improvement | 
| Whisper speech transcription | convet audio to text under different recording conditions | WER of 6.07% (clean), 6.54% (noisy), 6.54% (phone) | Success (on synthetic audio) | pretrained speech model can handle simulated conditions, but not all real world accents |
| location transcription | preserve important Kenyan location names | location token correctness of 62.5% (noisy) | Failure | unfamiliar names and audio distortion produced unnecessary substitutions, rest of the sentence was readable |<br><br>

### Models Obtained
| Model/Tool| Model | Where it came from | Intended Role|
|---|---:|---:|---|
| GLiNER | [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1) | Pretrained model from Hugging face; | Identify event, location, date, participant and sensitive information |
| Whisper | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI pretrained speech to text, accessed through transformers | Converts audio into English transcriptions|
| SmolLM2 | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Hugging Face pretrained instruction model | Experiment with generating fictional whatsApp messages|
| Rule based baseline | Project Python Keyword | Team implementation | Provide a simple classification and extraction comparison |
### Model Setup and initial execution
1. GLiNER: The week 7 notebook installs the gliner package, imports GLiNER, and loads the model. We used GLiNER to identify important information is Whatsapp messages like names, dates, locations, event names, phone numbers and email addresses. We also tested removing sensitive information by replacing it with [REDACTED]. Finally, we started organizing the extracted info into a JSON file so that it could be stored easily. 
2. Whisper (Audio Transcription): We obtained the pretrained model openai/whisper-large-v3-turbo. We were able to use Whisper to convert audio recordings into written text, and the system successfully processed audio files and saved the transcriptions in a compatible format. This also allowed us to eventually process both written Whatsapp messages as well as recordings.
3. Synthetic Text generation: We used the pretrained Hugging Face model HuggingFaceTB/SmolLM2-360M-Instruct to generate fictional whatsapp messages. We generated 10 sample messages about community activities in Kenya and saved them in a JSON format. Following this we also experimented with Python generated messages to create testing data without using real personal information. This provided a starting point for week 8 synthetic data.
4. Combining the Models: Our goal is to connect the different components into one system. First, Whisper converts audio recordings into text. Next, GLiNER identifies the important details and removes sensitive information. During week 7, we focused on getting the pretrained models working individually and testing their initial capabilities. Fine tuning and further improvements will begin week 8.<br><br>

- **What improved?**
1. text parsing: data is more related to setting, more NER categories (labels), JSON conversion<br><br>
2. More NER labels: Tested specific categories for people, jobs, dates, times and locations in order to better address the reporting use case
3. Redaction: Demonstrated replacing identified sensitive spans with [redacted] and began converting extracted information into JSON records
4. Working audio baseline: Used whisper and evaluated 90 synthesized audio clips across clean and noisy phone conditions. Which we then measured the WER.
- **What did not improve?**
1. text parsing: still misses/mislabels some text, has some false positives<br><br>
2. Long message processing: Longer or information heavy texts may reduce the accuracy
3. Redaction: The recorded 100% recall does not tell us if the model removed information unnecessarily. This is important since we must preserve all useful information including attendance despite having to hide sensitive information
4. Incomplete formatting: JSON conversion was still partial, and the team could not yet validate a common output format across both text and audio
5. Speech field accuracy: Whisper's WER was under 10% on synthetic audio, but exact names and locations could be transcribed incorrectly
- **Next experiment:**
1. text parsing:
    - changes: use shorter messages (so no truncation), add specialized keywords to fish out false positives, improve current keywords to have more comprehensive searches
    - measure: manual checks, and metric checking: PII > 95% and F1 > 95% (redaction and label accuracy)<br><br>
2. Measure the outcome: compare PII recall, entity extraction precision/recall/F1, Whisper WER and accuracy of all necessary components in comparison to their verified labels. 
3. Improve final JSON output: Store redacted text and extracted attributes in a consistent structure, and test the pipeline of audio to transcript to extraction and final output. 
### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| Eric Wang | fine-tuned whisper on afrispeech and added transcript cleaning | Worked | [finetune_whisper.py](src/training/finetune_whisper.py), [wer_results.json](evaluation/wer_results.json) | connect audio pipeline with gliner |
| Terence Zhou | improved text parsing and redaction, added partial JSON conversion | Partly | [4181acb](https://github.com/SCU-CSEN166/Dandelion4/commit/4181acbf69f7993b08182da285774ebfbe401bc1) | parse data sourced from teammates, improve keyword searching, redact JSONs, store in actual JSON file |
|Teresa Wei|Generated fictional community worker messages, recorded expected labels and JSON data, tested a rule based classification baseline, experimented with hugging face SmoILM2 model | Worked as development support for synthetic examples and baselines |  [`src/generate_data.py`](src/generate_data.py), [`src/baseline.py`](src/baseline.py), [`evaluation/evaluate.py`](evaluation/evaluate.py) | Expan/review data and labels for fine tuning during week 8.|

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

### Baseline (Text)
- **Submitted system version / commit:** [`f0d2c3d`](https://github.com/SCU-CSEN166/Dandelion4/commit/f0d2c3da71ebd72b2d0e4f1b78b7a5a8c47a42ea)
- **Hidden-test metrics and leaderboard result:** [`results`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/hidden_test_outputs.csv)
- **Difference from development results:** redacted labels are vaguely labelled as [REDACTED] instead of: eg. [NAME], [LOCATION], [PERSON]; duplicate functionality unimplemented

### Audio & End-to-End Pipeline
- **Submitted system version / commit:** [9063d44](https://github.com/SCU-CSEN166/Dandelion4/commit/9063d44)
- **Hidden-test metrics and leaderboard result:** On the 10-clip hidden test ([`demo/outputs/demo_metrics.json`](demo/outputs/demo_metrics.json)):
  - Relevance: **100.00%**
  - Participant count exact match: **100.00%**
  - Strict activity_type and combined attribute exact match: **37.50%**
  - Redaction placeholder recall: **97.83%** (45/46; 1 resulted from audio pipeline word error)
  - Duplicate detection: **100.00%**
  - ASR WER: **9.22%** raw / **10.60%** cleaned
- **Difference from development results:** synthetic audio got 4.21–4.67% raw WER, while the 10 hidden clips have real speakers and kenyan names/places (9.22% raw WER)

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| keyword redaction | redacted labels are vaguely labelled as [REDACTED] | small code issue | little impact | insert keyword into redaction label |
| duplicate checking | duplicates were not marked | unimplemented in code | medium impact | implement it |
| kenyan name/group errors (e.g., Tumaini -> "Too Many") | unfamiliar kenyan group name got mistranscribed | smaller whisper turbo model struggled on kenyan names | high / missed 1 redaction slot | fine-tune larger full whisper-large-v3 and quantize to int8 |
| number formatting mismatch | cleaning turned spelled-out numbers into digits, raising WER | reference transcripts kept numbers as words | low / participant counts still matched 8/8 | keep raw ASR WER separate from cleaned text |

### Changes attempted after stress-testing
- **Submitted system version / commit:** [`28f0c8f`](https://github.com/SCU-CSEN166/Dandelion4/commit/28f0c8f639fa6e6c26caefbbfd5893bb4464d693)
- **Hidden-test metrics and leaderboard result:** [`results`](https://github.com/SCU-CSEN166/Dandelion4/blob/audio/demo/expected/hidden_test_outputs.csv)
- **Difference from development results:** csv formatting (very little to no effect)

| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| better keyword redaction | Yes | [REDACTED] made less vague, see [anonymize_csv_text](https://github.com/SCU-CSEN166/Dandelion4/blob/audio/src/pipeline/hidden_test_pipeline.py#L177) function | Keep |
| implemented duplicate checking | Yes | duplicates are now marked, see [find_duplicate_events](https://github.com/SCU-CSEN166/Dandelion4/blob/audio/src/pipeline/hidden_test_pipeline.py#L313) function | Keep |
| switched to larger whisper-large-v3 quantized to int8 | Yes | hidden-test WER dropped from 9.22% to 7.49%; AfriSpeech 11.58% -> 9.31% and FLEURS 5.00% -> 4.41% | Keep |

| Expanded synthetic dataset for model training | Generated 1,200 fictional WhatsApp health messages across 6 activity categories (200 per category). Added expected activity labels, relevance labels, and entity annotations.| Yes the dataset generation and preparation completed | 1,200 labeled synthetic messages| Dataset provides additional training and evaluation examples |


### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| Terence Zhou | processed baseline hidden test csv file | Partly | [f0d2c3d](https://github.com/SCU-CSEN166/Dandelion4/commit/f0d2c3da71ebd72b2d0e4f1b78b7a5a8c47a42ea) | tune model against teammate's generated code |
| Eric Wang | hooked up audio-to-records pipeline and tested on the 10 hidden clips | Worked | [9063d44](https://github.com/SCU-CSEN166/Dandelion4/commit/9063d44) | try larger whisper model quantized to int8 |

| Teresa Wei | Expanded the synthetic WhatsApp dataset to 1,200 messages and organized the data into training (840), validation (180), and test (180) for future fine tuning| Worked / Prepared | Synthetic dataset and generation scripts | Use the expanded dataset to evaluate and fine-tune activity classification and entity extraction. |


---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing
- **Team evaluated:** [`Dandelion 1`](https://github.com/SCU-CSEN166/Dandelion1)
- **One thing that worked well:** their fixed hidden-test pipeline (src/main.py) hit 10/10 on relevance, participant count, and duplicate detection, plus 44/46 PII item recall, and outputs clean messages.csv and follow_ups.csv tables
- **One meaningful failure:** src/main.py only takes pre-transcribed text/spreadsheets rather than raw audio files directly; also pii_count accuracy was 7/10 with 2 partial redactions
- **One actionable recommendation:** wire src/transcribe.py into src/main.py so raw voice notes run end-to-end in one command
- **Feedback received from another team:** they flagged a redaction miss on the hidden test, though it was actually an ASR mistranscription (Tumaini -> "Too Many") rather than a redaction leak, so the real name wasn't exposed; they also asked for clearer end-to-end setup and run instructions at the end of the README, which we are fixing going forward

### Responsible AI assessment
- **Potential harms or affected users:**
  - Who could be affected:
    - voice recordings and transcribed messages
    - speakers, field workers, and study participants
    - downstream analysts that rely on generated records
  - How?
    - Whisper transcription could mishear critical details due to phone audio compression or background noise
    - GLiNER could miss sensitive info, or could mislabel extracted info
- **Privacy and data handling:**
  - Sensitive PII (eg. names, phone numbers, and locations) are identified by [`gliner_system.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/28f0c8f639fa6e6c26caefbbfd5893bb4464d693/src/nlp/gliner_system.py), and replaced with the corresponding redaction keyword.
  - These redactions are then validated with [`src/core/contracts.py (line 166)`](https://github.com/SCU-CSEN166/Dandelion4/blob/28f0c8f639fa6e6c26caefbbfd5893bb4464d693/src/core/contracts.py#L166).
- **Reliability and unsupported claims:**
  - How outputs are verified:
    - Text parsing: verfified with [`src/core/contracts.py (line 166)`](https://github.com/SCU-CSEN166/Dandelion4/blob/28f0c8f639fa6e6c26caefbbfd5893bb4464d693/src/core/contracts.py#L166)
    - Audio processing of clean, noisy, and phone audio: evaluated and verified with [`evaluation/eval_wer.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/28f0c8f639fa6e6c26caefbbfd5893bb4464d693/evaluation/eval_wer.py) and [`evaluation/eval_whisper_benchmarks.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/audio/evaluation/eval_whisper_benchmarks.py)
  - However, because the pipeline relies on probabilistic ASR and zero-shot/fine-tuned GLiNER span predictions, low-confidence transcriptions or ambiguous entities can still produce misextractions without explicit uncertainty scores surfaced to end users.
- **Fairness, accessibility, and usability:**
  - Fairness:
    - Real-world recording conditions: simulated with clean, noisy, and phone audio variants ([`examples/transcripts_clean.json`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/examples/transcripts_clean.json), [`examples/transcripts_noisy.json`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/examples/transcripts_noisy.json), [`examples/transcripts_phone.json`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/examples/transcripts_phone.json)).
    - Speaker diversity: fetched, processed, and simulated with FLEURS ([`training/fetch_fleurs.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/training/fetch_fleurs.py))
  - Accessibility: supported through JSON/CSV outputs ([`predicted_outputs.csv`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/demo/outputs/predicted_outputs.csv))
  - Usability: supported through CLI demo ([`demo/run_demo.py`](https://github.com/SCU-CSEN166/Dandelion4/blob/main/demo/run_demo.py))
- **Human oversight:**
  - Where a human should review/correct:
    - WER of audio processing shoots up dramatically
    - GLiNER predicting borderline/overlapping values/keywords
    - before exporting (hopefully) fully redacted transcripts or structured CSV/JSONs
- **Remaining limitations:**
  - The system should not be trusted to run fully autonomously with little to no oversight.
  - Our model mostly processes casual chat messages or semi-formal announcements. A major increase in the confidentiality of transcripts is another major limitation.
  - Lesser limitations include:
    - higher ASR error rates in heavily degraded phone audio
    - potential false positives/negatives in GLiNER PII detection of uncommon names/locations/miscellaneous keyword phrasings
    - speakers that are not represented in the FLEURS or synthetic fine-tuning sets

### Final improvements and outcomes
| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence | Remaining limitation |
|---|---|---|---|---|
| setup and run instructions / large-batch speed | added unified cli (python -m src.main) with github release auto-download, chunked streaming, cpu multi-worker int8, gpu fp16, and cleaner output without warnings | Yes | [PR #2](https://github.com/SCU-CSEN166/Dandelion4/pull/2), 291 tests passing, 647 fleurs clips ran in 2m40s on gpu | still misses some balanced afrispeech wer targets so human review is needed |
| code readability, setup docs, and spec alignment | refactored long functions across src/evaluation/demo, added full docstrings and plain-english comments, fixed cpu tokenizer warning, and updated readme setup/reproducibility and interface-spec.md | Yes | 291 fast tests passing (98.70% coverage), ruff and mypy clean | frontend ui still pending teammate integration |
| Command line only workflow / limited usability for nontechnical users | Created a Streamlit frontend (app.py) with a Whatsapp style text frield, file uploads (audio, CSV, JSON, ZIP), a process button with a structured results table. The interface calls the existing src.pipeline.pipeline.run_on_input() backend rather than duplicating Whisper or GLiNER code | Yes, website launches and fictional mesages and audios were processed and passed through successfully | Requires audio testing, currently runs on a development Codespace |


### Individual contributions during that week
| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
|---|---|---|---|---|
| Eric Wang | re-tuned full whisper-large-v3, quantized to int8, added cli and github release model auto-download; cleaned up and documented codebase, fixed cpu tokenizer warning, and updated setup/reproducibility and interface-spec.md | Worked | [PR #2](https://github.com/SCU-CSEN166/Dandelion4/pull/2), [7f76928](https://github.com/SCU-CSEN166/Dandelion4/commit/7f76928), [746832f](https://github.com/SCU-CSEN166/Dandelion4/commit/746832f), [aa5c1b8](https://github.com/SCU-CSEN166/Dandelion4/commit/aa5c1b8), 291 tests passing, ruff and mypy clean | cpu int8 runs cleanly offline; still misses some balanced afrispeech wer targets so human review is needed |
| Terence Zhou | documented results, updated/cleaned README | Worked | [dc6200f](https://github.com/SCU-CSEN166/Dandelion4/commit/dc6200ffe1e491dc4ddadecaa18da78cb6f4f3c1), [dc566a4](https://github.com/SCU-CSEN166/Dandelion4/commit/dc566a4a9559d9ff9c04401a9317d6138c09a9ca) | final(?) check of README soon |
|Teresa Wei|Built a Streamlit web interface (app.py) for pasted text and uploaded files and connected the interface to the team's existing run_on_input() pipeline. Tested and developed in GitHub Codespace, pushed the frontend in separate branch.| Worked | [de65aa6d1dc15b2ed6509fcc8276aea62d6e64a8](https://github.com/SCU-CSEN166/Dandelion4/commit/de65aa6)| The interface allows for the pipeline to be more accessible without duplicating model code.|



### Final submission checklist
- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

---

## Setup and Reproducibility (keep updated)

### Requirements
- Python 3.12 and the pinned packages in requirements.txt
- Works on CPU out of the box on Windows, macOS, and Linux (no GPU required); automatically uses CUDA GPU when available
- Disk space: ~2.3 GB for CPU runtime models (whisper-large-v3-tuned-int8, gliner_medium-v2.1, deberta-v3-base), or ~5.2 GB if a CUDA GPU is present (also downloads the full FP16 whisper-large-v3-tuned model)
- Internet access once on first run to download runtime models from the models-v5-int8 GitHub Release into .models/; after that, inference runs completely offline (can be verified with HF_HUB_OFFLINE=1)

### Installation
```bash
git clone https://github.com/SCU-CSEN166/Dandelion4.git
cd Dandelion4

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: pre-download runtime models into .models/ ahead of time
python -m src.core.setup_models
```

### Run the system
```bash
# Run on the default demo/inputs folder (writes redacted outputs.csv, records.json, and resumable records.jsonl to demo/outputs/custom_run)
python -m src.main

# Run on any single audio file, text/json/csv file, .zip archive, or folder
python -m src.main --in-dir path/to/data --out-dir outputs/run1

# Common options: force CPU, change chunk size, overwrite previous checkpoint, or skip transcript cleanup
python -m src.main --device cpu --chunk-size 64 --overwrite
python -m src.main --help

# Run the hidden-test demo and compare against demo/expected/hidden_test_outputs.csv
python demo/run_demo.py
```
### Streamlit web interface (currently in Frontend branch)
The streamerlit app allows for a browser based alternative where users can paste whatsapp messages, upload audio or message files and run the existing integrated processing pipeline. Then the user will be allowed to download the outputted CSV or JSON.
To test the Frontend branch after completing the python 3.12 environment setup above:
### In the project root, with the virtual environment activated

```bash
python -m pip install streamlit
python -m src.core.setup_models
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType none
```

In Github Codespaces open the forwarded port 8501. In a local environment use http://localhost:8501. The frontend was successfully launched and used for text messages and audio uploads. 

Outputs are always redacted (no unredacted transcripts are written to disk):
- outputs.csv: reviewable spreadsheet with anonymized_text, relevance, activity_type, participants, and duplicate_of
- records.json: full structured Interface Spec v2 records (with text_redacted)
- records.jsonl: incremental checkpoint flushed per chunk so interrupted large runs can resume automatically

### Run evaluation and tests
```bash
# Run synthetic and hidden evaluation suites (CPU or CUDA automatically)
python -m evaluation.run_evaluation --suites synthetic hidden

# Run public speech suites (auto-downloads datasets into evaluation/datasets/ and examples/fleurs_audio/ when online)
python -m evaluation.run_evaluation --suites test_300_seed1234 test_330_balanced kenyan_ea_test fleurs_all

# Run the fast offline unit test suite (291 tests; excludes 2 @pytest.mark.slow tests that hit live Edge TTS / Hugging Face)
pytest -m "not slow" -q

# Optionally run the 2 slow network-dependent tests
pytest -m slow -q

# Run linter and type checker
ruff check --extend-exclude "*.ipynb" src evaluation tests demo
mypy
```

### Repository structure
```text
Dandelion4/
├── README.md
├── interface-spec.md
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── demo/           # demo runner, sample inputs (demo/inputs/), expected outputs (demo/expected/)
├── evaluation/     # WER scoring, public benchmark fetcher, unified evaluation CLI
├── examples/       # synthetic audio, manifests, transcripts
├── notebooks/      # Week 6-7 Colab/local notebooks
├── src/            # main CLI (src/main.py), asr/, nlp/, pipeline/, core/, training/, legacy/
└── tests/          # automated unit tests
```
