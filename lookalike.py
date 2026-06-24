# ============================================================
# lookalike.py — Ideal Candidate Cloning ("Lookalike" Search)
# Input one great employee's profile, find candidates with
# highly similar career patterns. Uses our existing local
# embeddings — no external API needed.
# ============================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_lookalike_text(profile_dict):
    """
    Build searchable text from a manually-entered 'ideal employee' profile.
    profile_dict keys: title, company, skills (list), years_experience, summary
    """
    parts = [
        profile_dict.get("title", ""),
        profile_dict.get("summary", ""),
        f"Current: {profile_dict.get('title','')} at {profile_dict.get('company','')}",
        f"{profile_dict.get('years_experience', 0)} years experience",
        "Skills: " + ", ".join(profile_dict.get("skills", [])),
    ]
    return " ".join(filter(None, parts))


def find_lookalikes(seed_text, model, candidate_embeddings, candidates, top_n=20):
    """
    Embed the seed (ideal employee) text and find the most similar
    candidates by cosine similarity against precomputed embeddings.

    Returns list of (candidate, similarity_score) sorted descending.
    """
    seed_emb = model.encode([seed_text], show_progress_bar=False)
    norm = np.linalg.norm(seed_emb, axis=1, keepdims=True)
    seed_emb = seed_emb / np.maximum(norm, 1e-8)

    sims = cosine_similarity(seed_emb, candidate_embeddings)[0]

    ranked_idx = np.argsort(-sims)[:top_n]
    results = [(candidates[i], float(sims[i])) for i in ranked_idx]
    return results


def lookalike_explanation(seed_profile, candidate, similarity):
    """Generate a human-readable reason why this candidate resembles the seed."""
    p = candidate.get("profile", {})
    seed_skills = set(s.lower() for s in seed_profile.get("skills", []))
    cand_skills = set(s["name"].lower() for s in candidate.get("skills", []))
    shared = seed_skills & cand_skills

    reason = f"{similarity*100:.0f}% pattern match"
    if shared:
        reason += f" — shares {len(shared)} skills ({', '.join(list(shared)[:3])})"
    yoe_diff = abs(p.get("years_of_experience", 0) - seed_profile.get("years_experience", 0))
    if yoe_diff <= 1:
        reason += " — similar experience level"
    return reason
