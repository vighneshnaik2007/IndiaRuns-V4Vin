---
title: RankSense
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
---

# RankSense
**Fully local, AI-native candidate ranking** — built for the India Runs Hackathon (Data & AI Challenge: Intelligent Candidate Discovery)

---

## Overview

**RankSense** is an AI-native candidate intelligence platform engineered for absolute data privacy, massive throughput, and explainable recruitment decision-making. By running a 7-component hybrid scoring framework locally, our system removes the need for costly and data-leaking external API calls.

**Enterprise-Grade Scaling Capacity:** Built on top of optimized vector indexing and lightweight local tokenizers, the system is fully capable of parsing, auditing, and ranking **100,000+ candidates within 3 to 5 minutes** when deployed on standard localized compute clusters or multi-threaded hardware.

## How It Works

A two-stage pipeline trades unnecessary compute for speed without sacrificing precision:

1. **Stage 1 — Screen:** MiniLM + FAISS instantly screens all 100,000 candidates.
2. **Stage 2 — Score:** MPNet deep-scores only the top 2,000 candidates for precision.

Final ranking = a 7-component hybrid score: **Semantic Fit, Skill Graph, Behavioral Signals, Location Rules, Career Trajectory, Domain Alignment, and Alt. Credentials.**

## Project Structure

```
RankSense/
├── app.py                      # Hugging Face Spaces entry point (lightweight sandbox demo)
├── app_v4.py                   # Full local interactive dashboard (Gradio UI)
├── ranker_v4.py                # Batch CLI scoring engine — production / bulk runs (100k+ records)
├── validate_submission.py      # Validates output against the required submission schema
├── requirements.txt            # Python dependencies
├── sample_candidates.json      # Sample dataset preloaded in the HF sandbox
├── candidates.jsonl            # Input format expected by ranker_v4.py
└── submission.csv              # Generated output — top 100 ranked candidates
```

---

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

**Batch CLI** — `ranker_v4.py` (max speed, up to 100k+ records)
```bash
python ranker_v4.py --candidates candidates.jsonl --out submission.csv
```

**Validate Output** — `validate_submission.py`
```bash
python validate_submission.py submission.csv
```

**Interactive Dashboard** — `app_v4.py` (tune weights, NL search, lookalike search, outreach drafts)
```bash
python app_v4.py
```

**HF Sandbox (local preview)** — `app.py`
```bash
python app.py
```

## Live Demo

🔗 [huggingface.co/spaces/Vighneshnaik22/RankSense](https://huggingface.co/spaces/Vighneshnaik22/RankSense)

> Runs `app.py` on shared HF infrastructure with `sample_candidates.json` preloaded — a quick, no-setup preview, not a benchmark. To reproduce the full 100k-candidate pipeline, run `ranker_v4.py` / `app_v4.py` locally.

---

## Scoring Framework

| Component | Captures |
|---|---|
| **Semantic Fit** | Embedding similarity between candidate profile and JD |
| **Skill Graph** | Explicit + graph-inferred skill matches (50+ tech relationships) |
| **Behavioral Signals** | Platform engagement patterns + honeypot/fake-profile detection |
| **Location Rules** | Hard location/eligibility constraints |
| **Career Trajectory** | Progression, tenure, and growth patterns |
| **Domain Alignment** | Industry fit (FinTech, EdTech, SaaS, etc.) |
| **Alt. Credentials** | Bootcamps, certifications, non-traditional qualifications |

All 7 weights are recruiter-adjustable via live sliders in `app_v4.py`.

## Performance

- **100,000 candidates ranked against a JD in <4 minutes** (**<1 minute with cache**)
- Speed comes from the two-stage funnel: FAISS-based screening eliminates ~98% of candidates before the expensive MPNet stage runs on just the top 2,000
- Zero internet/API calls during ranking — fully private, fully local

## Main Deliverable

`submission.csv` — the top 100 ranked candidates, with normalized alignment scores and AI-generated alignment reasoning for each.
