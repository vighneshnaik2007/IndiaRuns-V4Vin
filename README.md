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

## Output
submission.csv — Top 100 ranked candidates with score and reasoning