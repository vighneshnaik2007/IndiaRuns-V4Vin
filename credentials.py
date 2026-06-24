# ============================================================
# credentials.py — Alternative Credentials Recognition
# Values bootcamps, certifications, and project portfolios
# alongside traditional tier-1 university degrees.
# ============================================================

TIER1_INSTITUTIONS = [
    "iit", "indian institute of technology", "nit", "national institute of technology",
    "bits pilani", "iiit", "stanford", "mit", "harvard", "berkeley", "carnegie mellon",
    "cmu", "oxford", "cambridge", "iisc"
]

RECOGNIZED_BOOTCAMPS = [
    "scaler", "masai school", "newton school", "coding ninjas", "general assembly",
    "lambda school", "app academy", "le wagon", "flatiron school"
]

RECOGNIZED_CERTS = [
    "aws certified", "google cloud certified", "azure certified",
    "deeplearning.ai", "coursera", "fast.ai", "kaggle grandmaster",
    "kaggle master", "tensorflow developer certificate"
]


def classify_education(education_list):
    """
    Classify a candidate's education entries into tier1 / bootcamp / cert / standard.
    Returns dict with classification + a fairness-adjusted score.
    """
    if not education_list:
        return {"type": "none", "score": 0.0, "detail": "No education data"}

    classifications = []
    for edu in education_list:
        institution = (edu.get("institution", "") or "").lower()
        degree = (edu.get("degree", "") or "").lower()

        if any(t in institution for t in TIER1_INSTITUTIONS):
            classifications.append(("tier1", institution))
        elif any(b in institution for b in RECOGNIZED_BOOTCAMPS):
            classifications.append(("bootcamp", institution))
        else:
            classifications.append(("standard", institution))

    types_found = [c[0] for c in classifications]

    if "tier1" in types_found:
        return {"type": "tier1", "score": 0.10,
                "detail": "Tier-1 institution background"}
    elif "bootcamp" in types_found:
        return {"type": "bootcamp", "score": 0.08,
                "detail": "Bootcamp graduate — evaluated on project merit"}
    else:
        return {"type": "standard", "score": 0.05,
                "detail": "Standard institution — evaluated on project merit"}


def detect_certifications(skills_list, summary_text=""):
    """Look for recognized certifications mentioned in skills or summary."""
    text = (summary_text or "") + " " + " ".join(
        s.get("name", "") if isinstance(s, dict) else str(s) for s in (skills_list or [])
    )
    text = text.lower()
    found = [c for c in RECOGNIZED_CERTS if c in text]
    return found


def alt_credentials_score(candidate):
    """
    Combined alternative-credentials score (0.0 - 0.15).
    Designed to give bootcamp/self-taught/certified candidates
    a fair shot alongside tier-1 degree holders.
    """
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    summary = candidate.get("profile", {}).get("summary", "")

    edu_result = classify_education(education)
    certs = detect_certifications(skills, summary)

    score = edu_result["score"]
    if certs:
        score += min(len(certs) * 0.02, 0.05)

    detail = edu_result["detail"]
    if certs:
        detail += f"; certified in: {', '.join(certs[:3])}"

    return {"score": round(score, 3), "detail": detail, "education_type": edu_result["type"]}
