# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D4<br>
**Challenge:** Dandelion Africa<br>
**Team members:** Terence Zhou, Teresa Wei, Yaozheng Wang<br>
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
  - [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) — automatic speech recognition (ASR); processes audio messages
  - [autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B) — text classification; detects important JSON entries & sensitive content
  - [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) - text generation; produces JSON database
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
| Yaozheng Wang | Postprocessing Data | N/A | anonymization, info extraction |
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
    - open [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model in Google Colab / VS Code
    - import Python dependencies (python3 -m pip install torch transformers)
    - run the model: generate WhatsApp messages, and save them to a JSON (python3 src/generate_data.py)
2. text parsing
    - open [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1) model in Google Colab / VS Code
    - import Python dependencies (!pip install gliner)
    - create input string (in "text" variable)
    - create text classes (in "labels" array)
    - input custom Python code for redaction process (see bottom of [file](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week6_gliner_medium_v2_1.ipynb))
    - run the model: classify and redact text<br><br>
- **Model(s) and data actually used:**
1. [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct): pretrained language model
    - data used: generated from Python script
    - stored generated data in JSON (50 messages)
2. [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1)
    - data used: custom-made strings, SCU emails
    - change of plans from using [autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B), not a very specialized model<br><br>

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
> **Full List of Inputs/Outputs:** [here](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week6_gliner_medium_v2_1.ipynb)<br>

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
| Yaozheng Wang | Audio Generation w/ TTS, Audio Transcription, WER measurements | Worked | [b76dd21](https://github.com/SCU-CSEN166/Dandelion4/commit/b76dd21452414fa495e688e2842ff85eb682b0f2) | improve handling of place names and numbers written as words |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
1. text parsing
    - source: pre-generated WhatsApp logs from Google Gemini
    - size: 5 prompts, each 1-3 large paragraphs
    - selection: related to field reports, announcements, urgent updates, summaries
    - annotations: specific NER categories, sensitive information redaction, storing data in JSONs
    - access: see [file](https://github.com/SCU-CSEN166/Dandelion4/blob/main/src/week7_gliner_medium_v2_1.ipynb)<br><br>

- **Metrics and rationale:** [Why these metrics reflect the intended use]
1. text parsing
    - PII Scrubbing Recall: checks for redactions / flagged content rate (must be > 95%)
    - Attribute Extraction F1: checks for database entry label accuracy (must be > 95%)<br><br>
### Models Obtained
| Model/Tool| Model | Where it came from | Intended Role|
|---|---:|---:|---|
| GLiNER | https://huggingface.co/urchade/gliner_medium-v2.1 | Pretrained model from Hugging face; | Identify event, location, date, participant and sensitive information |
| Whisper | (https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI pretrained speech to text, accessed through transformers | Converts audio into English transcriptions|
|SmolLM2 | https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct | Hugging Face pretrained instruction model | Experiment with generating fictional whatsApp messages|
|Rule based baseline | Project Python Keyword | Team implementation | Provide a simple classification and extraction comparison |
### Model Setup and initial execution
1. GLiNER: The week 7 notebook installs the gliner package, imports GLiNER, and loads the model. We used GLiNER to identify important information is Whatsapp messages like names, dates, locations, event names, phone numbers and email addresses. We also tested removing sensitive information by replacing it with [REDACTED]. Finally, we started organizing the extracted info into a JSON file so that it could be stored easily. 
2. Whisper (Audio Transcription): We obtained the pretrained model openai/whisper-large-v3-turbo. We were able to use Whisper to convert audio recordings into written text, and the system successfully processed audio files and saved the transcriptions in a compatible format. This also allowed us to eventually process both written Whatsapp messages as well as recordings.
3. Synthetic Text generation: We used the pretrained Hugging Face model HuggingFaceTB/SmolLM2-360M-Instruct to generate fictional whatsapp messages. We generated 10 sample messages about community activities in Kenya and saved them in a JSON format. Following this we also experimented with Python generated messages to create testing data without using real personal information. This provided a starting point for week 8 synthetic data.
4. Combining the Models: Our goal is to connect the different components into one system. First, Whisper converts audio recordings into text. Next, GLiNER identifies the important details and removes sensitive information. During week 7, we focused on getting the pretrained models working individually and testing their initial capabilities. Fine tuning and further improvements will begin week 8.
### Development data - audio and transcription
- Source: sythetic messages from the team's audio generation script rather than recordings from real community workers
- Size: 30 source messages with 3 audio conditions resulting in 90 total clips: clean, noisy and simulated phone quality audio.
- Selection: The generator selects community activity and irrelevant chat examples, it used Kenyan and English text to speech voices, adds noise for one condition and lowers sample rate for phone quality simulation.
- Annotations: The original written messages are reference transcripts; the audio associates each clip with its message and generation condition.
- Access: - [`src/make_audio.py`](src/make_audio.py), [`examples/audio_manifest.json`](examples/audio_manifest.json), [`src/transcribe.py`](src/transcribe.py), and [`evaluation/wer_results.json`](evaluation/wer_results.json).
### Text generation and baseline data
- Used 40 fictional text messages and corresponding expected output. The team also tested a hugging face model SmolLM2 to generate messages outside of the python template.
### Metrics and Rationale
| Component | Metric | Why it matters | Week 7 interpretation |
|---|---:|---:|---|
| GLiNER redaction | PII Scrubbing recall | Missing a person name, phone number, email, or ID can expose private info | Team target was to get >95% |
| GliNER extraction | F1 | Balances missed certain entities and incorrect extraction from dataset | Target was >95%; extraction quality still requires improvement |
| Whisper transcription | WER | Measures substitutions, deletions and insertions against a given transcript | Lower is better, we aimed for <10%|
| Whisper downstream impact | Baseline classification for accuracy| Shows whether the transcription mistakes affect subsequent relevance/activity predictions | Compares predictions from original text with predictions from transcripts |

- **Comparison:** [Baseline versus current approach, if different]
1. text parsing: N/A (same as baseline approach)

| System / version | Metric 1 | Metric 2 | Notes |
|---|---:|---:|---|
| Baseline | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Notes] |

| System / version | PII Scrubbing Recall | Attribute Extraction F1 | Notes |
|---|---:|---:|---|
| Baseline | 100% | 68% | miss 6, mislabel 2, total 25 |
| Current | 100% | 64.4% | miss 36, mislabel 6, total 118 |
## PII scrubbing recall:
The reported 100% means the model found all sensitive items in the examples included in that evaluation. It does not establish that every future name, phone number, or ID will be be redacted.
## Attribute Extraction F1:
The larger test returned 64.4% lower than the initial 68%. The example sets have different sizes and difficulty, so this is not a valid before/after improvement measurement. Missing attributes remain a major limitation.
## Targets:
The team's stated goals are more than 95% PII recall and more than 95% extraction F1
# Whisper pretrained transcription
The teams earlier evaluated experiment seen in valuation/wer_results.json tested 30 synthetic messages in each audio condition, for 90 total clips.

| Audio Condition | Clips | WER | Correct Location Tokens | Activity Classification |
|---|---:|---:|---:|---:|
| Clean | 30 | 6.07% | 75.00% | 83.33% |
| Noisy | 30 | 6.54% | 62.50% | 90.00% |
| Phone Quality | 30 | 6.54% | 70.83% | 83.33% |

The original text baseline classified 90% of the 30 source messages correctly. The comparison with the transcribed versions shows that even a small transcription error can affect later activity classifications. While the team initially started with WER below 10% on synthetic speech, all three audio conditions met that initial goal, but place name errors continued to be an issue. The separate 40 message rule based text baseline achieved 35/40 accuracy. 

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |
| text parsing | improved F1 score | worse F1 score | Failure | message truncation, bad labelling |<br><br>

- **What improved?** [Evidence]
1. text parsing: data is more related to setting, more NER categories (labels), JSON conversion<br><br>
- **What did not improve?** [Evidence]
1. text parsing: still misses/mislabels some text, has some false positives<br><br>
- **Next experiment:** [Specific change and how you will measure it]
1. text parsing:
    - changes: use shorter messages (so no truncation), add specialized keywords to fish out false positives, improve current keywords to have more comprehensive searches
    - measure: manual checks, and metric checking: PII > 95% and F1 > 95% (redaction and label accuracy)<br><br>

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |
| Terence Zhou | improved text parsing and redaction, added partial JSON conversion | Partly | [commit]() | parse data sourced from teammates, improve keyword searching, redact JSONs, store in actual JSON file |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| [Case 1] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 2] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 3] | [Observation] | [Hypothesis] | [Impact] | [Fix] |

### Changes attempted after stress-testing
| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| [Change] | [Status] | [Result] | [Decision] |

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing
- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment
- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes
| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence | Remaining limitation |
|---|---|---|---|---|
| [Issue] | [Change] | [Status] | [Link or metric] | [Limitation] |

### Individual contributions during that week
| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Reflection] |



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
[Python version, libraries, compute requirements, model access]

### Installation
```bash
# Add installation commands here
```

### Run the system
```bash
# Add an example command here
```

### Run evaluation
```bash
# Add an example evaluation command here
```

### Repository structure
```text
project/
├── README.md
├── requirements.txt
├── src/
├── tests/
├── evaluation/
└── examples/
```
