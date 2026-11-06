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

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A full readme describing our approach, our responsibilities, goals for each week and a plan for synthesizing data. 

### 1. Problem formulation and social impact
- **Problem:** We need to develop a way for WhatsApp group exports into to be made into structured records of  activities and outcomes.
- **Intended users:** Dandelion Africa program managers and volunteers
- **Inputs and outputs:** Input will be audio and transcribed audio as well as messages, output will consist of a structured record of events, plans and notes of the meetings. 

> **Input:** Hello, everyone! My name is Terence, and I worked with a group of 2 people on the TTP4SI project in Heafey 106 on 10/30/2026. This week, we worked on the README together.<br>
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

Potential challenges include:
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
- We have implemented a synthetic text-data generation process through python and hugging face. We were able to create 40 whatsapp messages representing different community activities and conversations that may be viewed as unrelated. We used the pretrained SmoILM2-360M instruct model to generate 10 additional whatsapp messages. Both approaches were successful in producing JSON files that could be used for developing our baseline. The Hugging Face messages include examples related to community healthcare, activities and meetings. 
- **How to run it:**
- To generate the synthetic data we opened the project in VS code and installed python dependencies in order to run the hugging face model. We first ran python3 -m pip install torch transformers, then ran python3 src/generate_data.py
- This script was able to generate the whatsapp messages and saved them to our json
- **Model(s) and data actually used:**
- We used Hugging Face SmoILM2-360M-Instruct: pretrained language model
- Python for synthetic data and JSOn file
- JSON to store generates messages
- Dataset includes 50 total generated text examples
- Model reference: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
- **Example 1:**
- Input: We visited {n} households in {loc} today → Output: {
  "id": "msg_001",
  "sender_id": "worker_01",
  "message": "We visited 25 households in Nakuru today."
}
- **Example 2:** Ex of generated message: We had a meeting with the clinic manager to discuss the upcoming health screening 

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| [Attempt] | [Status] | [Evidence] | [Learning] |
| Python generated data| Yes| Synthetic messages in JSON| Successfully generated repeatable activity reports and other messages|
### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| Terence Zhou | Text to JSON | TBD | TBD | TBD |
| Teresa Wei | Audio Compilation | TBD | TBD | TBD |
| Yaozheng Wang | Text Compilation | TBD | TBD | TBD |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes |
|---|---:|---:|---|
| Baseline | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Notes] |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

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


