---
title: RankSense
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
python_version: "3.10"
---
# India Runs — V4Vin | Intelligent Candidate Ranking System

## Team: V4Vin
AI-powered hybrid candidate ranking system for the India Runs Hackathon (Redrob AI / Hack2Skill).

## Approach
- Semantic similarity using sentence-transformers (all-MiniLM-L6-v2)
- Behavioral signal scoring (23 Redrob signals)
- Rule-based scoring (experience, location, company type)

## How to Run
pip install sentence-transformers numpy pandas tqdm scikit-learn

python ranker_v4.py    # Generates rankings and submission.csv

python validate_submission.py submission.csv  # Validate your results

## Live Demo
You can view a live demo of the RankSense engine here:
https://huggingface.co/spaces/Vighneshnaik22/RankSense

**Note on Demo Mode:**
The live space is pre-loaded with `sample_candidates.json` to allow for an immediate demonstration of the ranking system upon page load. This demo is configured to run in a lightweight environment with pre-loaded data to ensure stability and meet performance constraints during evaluation.

## Output
submission.csv — Top 100 ranked candidates with score and reasoning
