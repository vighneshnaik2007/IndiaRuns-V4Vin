# ============================================================
# career.py — Career Trajectory Analyzer
# Feature #3: Career Trajectory Analysis
# Team V4Vin | India Runs Hackathon 2026
# ============================================================

# Instead of just looking at current role, we analyze the
# candidate's entire career journey:
# - Are they moving UP in seniority?
# - Are they moving TOWARD AI roles?
# - Did they recently escape consulting?
# - Is their career accelerating or stagnating?

SENIORITY_LEVELS = {
    "intern": 0,
    "trainee": 0,
    "junior": 1,
    "associate": 1,
    "engineer": 2,
    "developer": 2,
    "analyst": 2,
    "senior": 3,
    "lead": 4,
    "staff": 4,
    "principal": 5,
    "architect": 5,
    "manager": 4,
    "head": 5,
    "director": 6,
    "vp": 7,
    "chief": 8,
    "cto": 8,
    "ceo": 8,
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "deep learning", "nlp",
    "computer vision", "data science", "artificial intelligence",
    "neural", "llm", "generative", "embedding", "search",
    "ranking", "recommendation", "research", "scientist"
]

CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "mphasis", "hexaware",
    "tech mahindra", "hcl", "ibm global", "deloitte",
    "pwc", "kpmg", "ey ", "ernst"
]

PRODUCT_COMPANIES = [
    "google", "microsoft", "amazon", "meta", "apple",
    "netflix", "uber", "airbnb", "flipkart", "zomato",
    "swiggy", "razorpay", "phonepe", "paytm", "meesho",
    "unacademy", "byju", "cred", "groww", "zepto",
    "openai", "anthropic", "redrob", "salesforce",
    "atlassian", "freshworks", "zoho", "browserstack"
]


def get_seniority_level(title):
    """Convert job title to numeric seniority level"""
    if not title:
        return 0
    title_lower = title.lower()
    max_level = 0
    for keyword, level in SENIORITY_LEVELS.items():
        if keyword in title_lower:
            max_level = max(max_level, level)
    return max_level


def is_ai_role(title, description=""):
    """Check if a role is AI/ML related"""
    text = (title + " " + description).lower()
    return any(kw in text for kw in AI_KEYWORDS)


def is_consulting_company(company):
    """Check if company is a consulting firm"""
    if not company:
        return False
    company_lower = company.lower()
    return any(firm in company_lower for firm in CONSULTING_FIRMS)


def is_product_company(company):
    """Check if company is a product company"""
    if not company:
        return False
    company_lower = company.lower()
    return any(prod in company_lower for prod in PRODUCT_COMPANIES)


def analyze_trajectory(candidate):
    """
    Analyze career trajectory and return a score + detailed insights.

    Returns:
        score: float between -0.3 and +0.4
        insights: dict with detailed trajectory analysis
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    if not career:
        return 0.0, {"status": "No career history available"}

    # Sort by most recent first (assume order is chronological)
    # career[0] = most recent, career[-1] = oldest

    score = 0.0
    insights = {}

    # ── 1. Seniority Growth ──────────────────────────────────
    # Are they moving up in their career?
    titles = [job.get("title", "") for job in career]
    seniority_levels = [get_seniority_level(t) for t in titles]

    if len(seniority_levels) >= 2:
        # Compare recent vs older roles
        recent_level = seniority_levels[0]
        oldest_level = seniority_levels[-1]

        if recent_level > oldest_level:
            score += 0.10
            insights["seniority_trend"] = f"📈 Growing — moved from level {oldest_level} to {recent_level}"
        elif recent_level == oldest_level:
            score += 0.02
            insights["seniority_trend"] = "➡️ Stable seniority"
        else:
            score -= 0.05
            insights["seniority_trend"] = "📉 Declining seniority"
    else:
        insights["seniority_trend"] = "Single role — no trend data"

    # ── 2. AI Role Progression ───────────────────────────────
    # Are they moving toward AI roles?
    ai_roles = [is_ai_role(job.get("title", ""),
                           job.get("description", ""))
                for job in career]
    ai_role_count = sum(ai_roles)
    total_roles = len(career)

    recent_is_ai = ai_roles[0] if ai_roles else False
    was_always_ai = all(ai_roles)
    recently_moved_to_ai = recent_is_ai and not all(ai_roles)

    if was_always_ai:
        score += 0.15
        insights["ai_progression"] = f"🎯 Consistent AI career — {ai_role_count}/{total_roles} roles in AI"
    elif recently_moved_to_ai:
        score += 0.10
        insights["ai_progression"] = "🔄 Recently transitioned to AI — positive signal"
    elif recent_is_ai:
        score += 0.08
        insights["ai_progression"] = "✅ Currently in AI role"
    else:
        score -= 0.05
        insights["ai_progression"] = "⚠️ Not currently in AI role"

    # ── 3. Consulting Escape Analysis ───────────────────────
    # Did they recently leave consulting for product?
    current_company = profile.get("current_company", "")
    past_companies = [job.get("company", "") for job in career[1:]]

    currently_consulting = is_consulting_company(current_company)
    was_in_consulting = any(is_consulting_company(c) for c in past_companies)
    currently_product = is_product_company(current_company)

    if currently_product and was_in_consulting:
        score += 0.10
        insights["consulting_analysis"] = "🚀 Escaped consulting → now at product company"
    elif currently_consulting:
        score -= 0.10
        insights["consulting_analysis"] = "⚠️ Currently at consulting firm"
    elif currently_product:
        score += 0.08
        insights["consulting_analysis"] = "✅ At product company"
    else:
        insights["consulting_analysis"] = "➡️ Neutral company type"

    # ── 4. Job Stability ─────────────────────────────────────
    # Are they job hopping every 6 months?
    durations = [job.get("duration_months", 0) for job in career]
    avg_duration = sum(durations) / max(len(durations), 1)

    if avg_duration >= 24:  # 2+ years average
        score += 0.05
        insights["stability"] = f"✅ Stable — avg {avg_duration:.0f} months per role"
    elif avg_duration >= 12:
        score += 0.02
        insights["stability"] = f"➡️ Moderate tenure — avg {avg_duration:.0f} months"
    else:
        score -= 0.05
        insights["stability"] = f"⚠️ Job hopping — avg only {avg_duration:.0f} months per role"

    # ── 5. Recent Role Quality ───────────────────────────────
    # Is their most recent role at a strong company?
    if career:
        recent_company = career[0].get("company", "")
        recent_title = career[0].get("title", "")
        recent_duration = career[0].get("duration_months", 0)

        if is_product_company(recent_company) and is_ai_role(recent_title):
            score += 0.10
            insights["recent_role"] = f"🌟 Strong recent role: {recent_title} at {recent_company}"
        elif is_product_company(recent_company):
            score += 0.05
            insights["recent_role"] = f"✅ Good recent role: {recent_title} at {recent_company}"
        else:
            insights["recent_role"] = f"➡️ Recent: {recent_title} at {recent_company}"

    # ── 6. Overall trajectory label ─────────────────────────
    if score >= 0.30:
        insights["overall"] = "🚀 Rising Star"
    elif score >= 0.15:
        insights["overall"] = "💎 Strong Trajectory"
    elif score >= 0.05:
        insights["overall"] = "✅ Positive Growth"
    elif score >= -0.05:
        insights["overall"] = "➡️ Stable"
    else:
        insights["overall"] = "⚠️ Concerning Trajectory"

    insights["raw_score"] = round(score, 3)
    return score, insights