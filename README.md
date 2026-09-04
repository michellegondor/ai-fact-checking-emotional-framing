# Robustness of AI Fact-Checking Under Emotional Framing

**Does emotional framing influence Gemini's factual judgments, and can explicit evidence make them more robust?**

This project evaluates the robustness of Gemini's fact-checking judgments under emotional framing and examines whether providing explicit evidence improves factual accuracy or reduces sensitivity to framing.

## Experimental Design

The experiment used **1,500 claims from the FEVER dataset**.

Each claim was evaluated under:

- **5 framing conditions:** Neutral, Positive, Negative, Fear, and Anger
- **2 evidence conditions:** Evidence Present and Evidence Absent

Gemini's judgments were then compared across conditions using two main outcomes:

- **Fact-Check Reversal Rate (FCR):** sensitivity of factual judgments to changes in framing
- **Accuracy:** agreement with the ground-truth FEVER labels

## Main Findings

### 1. Emotional framing rarely changed Gemini's predictions

Framing-induced changes remained below **1% across all emotional conditions**, suggesting high stability to emotional framing.

### 2. Explicit evidence consistently improved factual accuracy

Providing evidence increased accuracy by approximately **2.2 percentage points across all framing conditions**.

### 3. Evidence did not significantly reduce framing sensitivity

The estimated difference in FCR between the no-evidence and evidence conditions was:

**ΔFCR = +0.117 percentage points**  
**95% CI: [-0.334, 0.567]**  
**p = .661**

The confidence interval includes zero, providing no evidence that explicit evidence significantly reduced framing sensitivity.

## Statistical Analysis

The analysis examined:

- framing-induced prediction changes across emotional conditions
- factual accuracy with and without explicit evidence
- the difference in framing sensitivity between evidence conditions
- confidence intervals and hypothesis tests for the primary effects

Analysis scripts and numerical outputs are included in this repository.

## Repository Contents

- `experiment_protocol.md` — experimental design and protocol
- `prepare_dataset_v2.py` — dataset preparation
- `extract_evidence.py` — evidence extraction
- `create_final_sample.py` — final sample construction
- `create_framed_v2.py` — emotional framing generation
- `prepare_final_batch.py` — preparation of Gemini evaluation requests
- `run_final_batch.py` — execution of the final Gemini evaluation
- `check_final_batch.py` — batch validation
- `parse_final_results.py` — parsing of model outputs
- `analyze_results.py` — statistical analysis
- `create_figures.py` — generation of the figures used in the study

## Reproducibility

The repository contains the analysis pipeline and final numerical outputs used for the study.

The original FEVER dataset is not redistributed here. Users wishing to reproduce the full experiment should obtain FEVER separately and configure their own Google Gemini API credentials.

No API credentials are included in this repository.

## Author

**Michelle Gondor**  
BSc Student in Electronics Engineering
