# ============================================================
# ranker_v4.py — RankSense Ranking Engine v4 (Ultra-Optimized)
# Integrates:
# 1. FP16 GPU Acceleration (Half-precision)
# 2. Smart File Caching (MD5 Hashing with Collision Protection)
# 3. FAISS Vector Indexing for instant retrieval
# 4. Two-Stage Retrieve & Re-Rank (MiniLM -> MPNet)
# ============================================================

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import os
import torch
import hashlib
import faiss  # FAISS Vector Database
import argparse

from skill_graph import skill_graph_score, get_skill_matches, JD_SKILLS
from honeypot import filter_honeypots, get_honeypot_report
from career import analyze_trajectory
from jd_parser import parse_jd, get_jd_summary
from domain import detect_jd_domain, domain_alignment_score, domain_alignment_label
from credentials import alt_credentials_score

MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

DEFAULT_WEIGHTS = {
    "semantic": 0.30,
    "skill":    0.18,
    "signals":  0.17,
    "rules":    0.13,
    "career":   0.10,
    "domain":   0.07,
    "credentials": 0.05,
}

CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "mphasis", "hexaware", "tech mahindra", "hcl",
    "ibm global services"
]

def get_candidate_label(scores, profile):
    total = scores["final"]
    yoe = profile.get("years_of_experience", 0)
    if total >= 0.70 and 5 <= yoe <= 9: return "💎 Ideal Fit"
    elif total >= 0.60 and yoe <= 5:    return "🚀 Rising Star"
    elif total >= 0.55:                 return "✅ Strong Match"
    elif scores["rules"] < -0.1:        return "⚠️ Risky Pick"
    else:                               return "➡️ Potential"

def build_candidate_text(c):
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        f"Current: {p.get('current_title','')} at {p.get('current_company','')}",
        f"{p.get('years_of_experience', 0)} years experience",
        f"Location: {p.get('location','')}, {p.get('country','')}",
    ]
    for job in c.get("career_history", [])[:3]:
        parts.append(f"{job.get('title','')} at {job.get('company','')}: {job.get('description','')}")
    skills = [s["name"] for s in c.get("skills", [])]
    parts.append("Skills: " + ", ".join(skills))
    return " ".join(filter(None, parts))

def compute_signal_score(c):
    from datetime import date as _date
    s = c.get("redrob_signals", {})
    score = 0.0
    days = s.get("days_since_last_login", None)
    if days is None:
        last_active = s.get("last_active_date", "")
        if last_active:
            try:
                last_dt = _date.fromisoformat(str(last_active)[:10])
                days = (_date.today() - last_dt).days
            except Exception:
                days = 999
        else:
            days = 999
    days = int(days) if days is not None else 999

    if days <= 7:    score += 0.15
    elif days <= 30: score += 0.10
    elif days <= 90: score += 0.05
    else:            score -= 0.05

    score += s.get("recruiter_response_rate", 0) * 0.15
    pcs = s.get("profile_completeness_score", 0)
    score += (pcs / 100.0 if pcs > 1 else pcs) * 0.10
    score += min(s.get("saved_by_recruiters_30d", 0) / 20, 1.0) * 0.08
    score += s.get("interview_completion_rate", 0) * 0.07
    oar = s.get("offer_acceptance_rate", 0)
    score += (oar if oar >= 0 else 0) * 0.05
    github = s.get("github_activity_score", 0)
    score += (github / 100.0 if github > 1 else max(github, 0)) * 0.10
    score += min(s.get("search_appearance_30d", 0) / 100, 1.0) * 0.05

    if s.get("verified_email", False):     score += 0.03
    if s.get("verified_phone", False):     score += 0.02
    if s.get("linkedin_connected", False): score += 0.03
    return min(max(score, 0.0), 1.0)

def compute_rule_score(c, jd_profile):
    p = c.get("profile", {})
    score = 0.0
    exp_range = jd_profile["experience"]
    yoe = p.get("years_of_experience", 0)
    if exp_range["min"] <= yoe <= exp_range["max"]: score += 0.20
    elif exp_range["min"] - 1 <= yoe <= exp_range["max"] + 2: score += 0.10
    elif yoe < 2: score -= 0.15
    
    loc = (p.get("location", "") + " " + p.get("country", "")).lower()
    if any(city in loc for city in jd_profile["preferred_locations"]): score += 0.15
    
    signals = c.get("redrob_signals", {})
    if signals.get("willing_to_relocate", False): score += 0.05
    notice = signals.get("notice_period_days", 999)
    if notice <= 30: score += 0.05
    elif notice <= 60: score += 0.02
    
    current_company = p.get("current_company", "").lower()
    all_companies = [j.get("company", "").lower() for j in c.get("career_history", [])]
    all_companies.append(current_company)
    consulting_count = sum(1 for co in all_companies if any(f in co for f in CONSULTING_FIRMS))
    if "no_consulting" in jd_profile["negative_signals"]:
        if consulting_count >= 2: score -= 0.20
        elif consulting_count == 1: score -= 0.08
        
    title = p.get("current_title", "").lower()
    if jd_profile["is_ai_role"]:
        ai_keywords = ["ai", "ml", "machine learning", "nlp", "search", "ranking", "research", "data science"]
        if any(t in title for t in ai_keywords): score += 0.10
    return score

def compute_diversity_adjustment(top_candidates):
    if len(top_candidates) < 10: return top_candidates
    location_counts = {}
    for cid, score, c, _ in top_candidates:
        loc = c.get("profile", {}).get("location", "Unknown")
        location_counts[loc] = location_counts.get(loc, 0) + 1
    total = len(top_candidates)
    adjusted = []
    for cid, score, c, insights in top_candidates:
        loc = c.get("profile", {}).get("location", "Unknown")
        loc_ratio = location_counts.get(loc, 0) / total
        adjustment = -0.005 if loc_ratio > 0.4 else 0.0
        adjusted.append((cid, score + adjustment, c, insights))
    return adjusted


class RankSenseRanker:
    def __init__(self, jd_text=None):
        self.jd_profile = parse_jd(jd_text)
        self.jd_domain = detect_jd_domain(self.jd_profile["raw_text"])
        self.models = []
        self.weights = dict(DEFAULT_WEIGHTS)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def set_weights(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.weights: self.weights[k] = max(0.0, float(v))
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights: self.weights[k] = self.weights[k] / total

    def load_models(self):
        print(f"[*] Loading models onto {self.device.upper()}...")
        for model_name in MODELS:
            model = SentenceTransformer(model_name, device=self.device)
            # FP16 HALF PRECISION
            if self.device == 'cuda':
                model.half()
            self.models.append(model)

    def get_cache_path(self, filepath, model_name):
        # SMART CACHING VIA MD5 HASH
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        file_hash = h.hexdigest()[:12]
        safe_name = model_name.replace("/", "-")
        return f".cache/embeddings_{safe_name}_{file_hash}.npy"

    def load_candidates(self, filepath="candidates.jsonl"):
        candidates = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading JSON"):
                line = line.strip()
                if line: candidates.append(json.loads(line))
        return candidates

    def rank(self, candidates=None, filepath="candidates.jsonl", top_n=100,
             save_embeddings=True, weight_overrides=None, feedback_nudge=None):
        
        start = time.time()
        if weight_overrides: self.set_weights(**weight_overrides)
        if candidates is None: candidates = self.load_candidates(filepath)

        clean_candidates, removed, stats = filter_honeypots(candidates, verbose=False)
        texts = [build_candidate_text(c) for c in clean_candidates]
        
        mini_model = self.models[0]
        mpnet_model = self.models[1]

        # ==========================================
        # STAGE 1: FAISS + MiniLM Retrieval
        # ==========================================
        print(f"[*] Stage 1: Fast MiniLM FAISS screening on {len(texts)} candidates...")
        
        # Only attempt to cache if the file actually exists on disk
        cache_path = self.get_cache_path(filepath, MODELS[0]) if (filepath and os.path.exists(filepath)) else None
        
        load_from_cache = False
        if cache_path and os.path.exists(cache_path):
            mini_embs = np.load(cache_path)
            # CRITICAL FIX: Ensure the cache matches the candidate count!
            if mini_embs.shape[0] == len(texts):
                print(f"[CACHE HIT] Loaded {len(texts)} MiniLM embeddings instantly.")
                load_from_cache = True
            else:
                print(f"[CACHE MISMATCH] Cache has {mini_embs.shape[0]} items, but current file has {len(texts)}. Re-encoding...")
                
        if not load_from_cache:
            mini_embs = mini_model.encode(texts, batch_size=512, show_progress_bar=True, device=self.device, normalize_embeddings=True)
            if cache_path:
                os.makedirs(".cache", exist_ok=True)
                np.save(cache_path, mini_embs)
                print(f"[CACHED] Saved MiniLM embeddings to {cache_path}")

        # Ensure float32 for FAISS compatibility
        mini_embs = np.array(mini_embs, dtype=np.float32)

        # Build FAISS Index (Inner Product = Cosine Similarity for normalized vectors)
        dimension = mini_embs.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(mini_embs)

        # Generate JD embedding and search FAISS
        jd_emb_mini = mini_model.encode([self.jd_profile["raw_text"]], normalize_embeddings=True)
        jd_emb_mini = np.array(jd_emb_mini, dtype=np.float32)
        
        stage1_pool = min(2000, len(clean_candidates))
        distances, indices = index.search(jd_emb_mini, stage1_pool)
        
        top_indices = indices[0]
        stage1_scores = distances[0]
        
        # ==========================================
        # STAGE 2: MPNet Deep Precision Re-Rank
        # ==========================================
        print(f"[*] Stage 2: MPNet deep scoring on top {stage1_pool} candidates...")
        top_texts = [texts[i] for i in top_indices]
        top_candidates = [clean_candidates[i] for i in top_indices]
        
        mpnet_embs = mpnet_model.encode(top_texts, batch_size=128, show_progress_bar=False, device=self.device, normalize_embeddings=True)
        jd_emb_mpnet = mpnet_model.encode([self.jd_profile["raw_text"]], normalize_embeddings=True)[0]
        stage2_scores = cosine_similarity([jd_emb_mpnet], mpnet_embs)[0]

        w = self.weights
        nudge = feedback_nudge or {}
        final_scores = []

        # Run the heavy Python scoring loop ONLY on the top 2000 (or fewer if small file)
        for local_idx, original_idx in enumerate(top_indices):
            c = top_candidates[local_idx]
            p = c.get("profile", {})
            skills = [s["name"] for s in c.get("skills", [])]

            # Blend the models: 35% MiniLM (broad context) + 65% MPNet (deep precision)
            sem_score = float((0.35 * stage1_scores[local_idx]) + (0.65 * stage2_scores[local_idx]))
            
            skill_score = skill_graph_score(skills, self.jd_profile["required_skills"])
            signal_score = compute_signal_score(c)
            rule_score = compute_rule_score(c, self.jd_profile)
            career_score, career_insights = analyze_trajectory(c)
            dom_score = domain_alignment_score(c, self.jd_domain)
            cred_result = alt_credentials_score(c)
            cred_score = cred_result["score"]

            final = (
                sem_score    * (w["semantic"] + nudge.get("semantic", 0)) +
                skill_score  * (w["skill"] + nudge.get("skill", 0)) +
                signal_score * (w["signals"] + nudge.get("signals", 0)) +
                rule_score   * (w["rules"] + nudge.get("rules", 0)) +
                career_score * (w["career"] + nudge.get("career", 0)) +
                dom_score    * w["domain"] +
                cred_score   * w["credentials"]
            )

            scores = {
                "semantic": round(sem_score, 4), "skill": round(skill_score, 4),
                "signals": round(signal_score, 4), "rules": round(rule_score, 4),
                "career": round(career_score, 4), "domain": round(dom_score, 4),
                "credentials": round(cred_score, 4), "final": round(final, 4)
            }

            insights = {
                "career": career_insights,
                "skills": get_skill_matches(skills, self.jd_profile["required_skills"]),
                "label": get_candidate_label(scores, p),
                "domain_label": domain_alignment_label(dom_score),
                "credentials_detail": cred_result["detail"],
            }

            final_scores.append((c["candidate_id"], final, c, {"scores": scores, "insights": insights}))

        final_scores = compute_diversity_adjustment(final_scores)
        final_scores.sort(key=lambda x: (-round(x[1], 4), x[0]))
        top = final_scores[:top_n]

        rows = []
        for rank, (cid, score, c, meta) in enumerate(top, 1):
            p = c.get("profile", {})
            skills = [sk["name"] for sk in c.get("skills", [])[:5]]
            career_label = meta["insights"]["career"].get("overall", "Unknown")
            label = meta["insights"]["label"]
            domain_label = meta["insights"]["domain_label"]

            reasoning = (
                f"{label} | {p.get('years_of_experience',0)}yr "
                f"{p.get('current_title','')} at {p.get('current_company','')} "
                f"({p.get('location','')}, {p.get('country','')}); "
                f"skills: {', '.join(skills)}; career: {career_label}; "
                f"domain: {domain_label}; "
                f"scores: sem={meta['scores']['semantic']}, skill={meta['scores']['skill']}, "
                f"signals={meta['scores']['signals']}, rules={meta['scores']['rules']}, "
                f"career={meta['scores']['career']}, domain={meta['scores']['domain']}, "
                f"credentials={meta['scores']['credentials']}"
            )

            rows.append({
                "candidate_id": cid, "rank": rank, "score": round(score, 4),
                "reasoning": reasoning
            })

        df = pd.DataFrame(rows)
        print(f"[*] Finished processing in {round(time.time() - start, 2)} seconds.")
        
        # Return only the top candidates and their embeddings for downstream Gradio logic
        return df, removed, top_candidates, mini_embs[top_indices]

if __name__ == "__main__":
    # Set up command-line arguments for hackathon compatibility
    parser = argparse.ArgumentParser(description="RankSense Ranking Engine")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to input JSONL file")
    parser.add_argument("--out", type=str, default="submission.csv", help="Path to output CSV file")
    
    args = parser.parse_args()
    
    # Run the ranker
    ranker = RankSenseRanker()
    ranker.load_models()
    df, honeypots, clean, emb = ranker.rank(filepath=args.candidates, top_n=100, save_embeddings=True)
    
    # Save the output
    df.to_csv(args.out, index=False)
    print(f"✅ {args.out} saved! {len(honeypots)} honeypots removed.")