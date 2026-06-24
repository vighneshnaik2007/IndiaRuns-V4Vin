# ============================================================
# explainer.py — Explainable AI (XAI) Module
# Feature #5: Explainable AI Dashboard
# Team V4Vin | India Runs Hackathon 2026
# ============================================================

# For every top candidate, this module generates:
# - Why they ranked where they did
# - Which signals boosted them
# - Which signals pulled them down
# - How they compare to JD requirements
# - A human-readable recruiter-style summary


def explain_candidate(candidate, scores, insights, rank, jd_profile):
    """
    Generate complete explanation for a candidate.

    Args:
        candidate: raw candidate dict
        scores: dict of individual scores
        insights: dict of career/skill insights
        rank: their final rank (1-100)
        jd_profile: parsed JD requirements

    Returns:
        explanation: dict with full explanation
    """
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    skills = [sk["name"] for sk in candidate.get("skills", [])]

    explanation = {
        "rank": rank,
        "candidate_id": candidate.get("candidate_id"),
        "label": insights.get("label", "Unknown"),
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "score_breakdown": scores,
        "skill_analysis": {},
        "signal_analysis": {},
        "career_analysis": {},
        "recruiter_note": ""
    }

    # ── Summary ──────────────────────────────────────────────
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "Unknown")
    company = p.get("current_company", "Unknown")
    location = f"{p.get('location','')}, {p.get('country','')}"

    explanation["summary"] = (
        f"Rank #{rank} — {yoe}yr {title} at {company} ({location}). "
        f"Overall score: {scores['final']:.4f}"
    )

    # ── Strengths ────────────────────────────────────────────
    strengths = []

    if scores["semantic"] >= 0.6:
        strengths.append(
            f"✅ Strong semantic fit (score: {scores['semantic']:.3f}) — "
            f"profile language closely matches JD requirements"
        )

    if scores["skill"] >= 0.5:
        skill_info = insights.get("skills", {})
        direct = skill_info.get("direct_matches", [])
        strengths.append(
            f"✅ Strong skill match (score: {scores['skill']:.3f}) — "
            f"direct matches: {', '.join(direct[:5]) if direct else 'via related skills'}"
        )

    if scores["signals"] >= 0.6:
        days = s.get("days_since_last_login", 999)
        rr = s.get("recruiter_response_rate", 0)
        strengths.append(
            f"✅ Highly active candidate — "
            f"last seen {days} days ago, "
            f"{int(rr*100)}% recruiter response rate"
        )

    if scores["career"] >= 0.15:
        career_info = insights.get("career", {})
        overall = career_info.get("overall", "")
        strengths.append(
            f"✅ Strong career trajectory: {overall}"
        )

    github = s.get("github_activity_score", 0)
    if github >= 0.7:
        strengths.append(
            f"✅ Active GitHub presence "
            f"(score: {github:.2f}) — real builder"
        )

    exp_range = jd_profile["experience"]
    if exp_range["min"] <= yoe <= exp_range["max"]:
        strengths.append(
            f"✅ Experience ({yoe} years) perfectly within "
            f"JD range ({exp_range['min']}-{exp_range['max']} years)"
        )

    loc = (p.get("location", "") + p.get("country", "")).lower()
    if any(city in loc for city in jd_profile["preferred_locations"]):
        strengths.append(
            f"✅ Located in preferred region: {p.get('location','')}"
        )

    explanation["strengths"] = strengths

    # ── Weaknesses ───────────────────────────────────────────
    weaknesses = []

    if scores["semantic"] < 0.4:
        weaknesses.append(
            f"⚠️ Weak semantic fit (score: {scores['semantic']:.3f}) — "
            f"profile doesn't strongly align with JD language"
        )

    if scores["skill"] < 0.3:
        skill_info = insights.get("skills", {})
        missing = skill_info.get("missing_skills", [])
        weaknesses.append(
            f"⚠️ Skill gaps detected — "
            f"missing: {', '.join(missing[:5]) if missing else 'several JD skills'}"
        )

    days = s.get("days_since_last_login", 999)
    if days > 90:
        weaknesses.append(
            f"⚠️ Inactive for {days} days — "
            f"may not be actively looking"
        )

    if scores["rules"] < -0.05:
        weaknesses.append(
            f"⚠️ Rule-based penalty (score: {scores['rules']:.3f}) — "
            f"possible consulting background or location mismatch"
        )

    if scores["career"] < 0:
        career_info = insights.get("career", {})
        trend = career_info.get("seniority_trend", "")
        weaknesses.append(
            f"⚠️ Career concern: {trend}"
        )

    notice = s.get("notice_period_days", 0)
    if notice > 90:
        weaknesses.append(
            f"⚠️ Long notice period: {notice} days"
        )

    explanation["weaknesses"] = weaknesses

    # ── Skill Analysis ───────────────────────────────────────
    skill_info = insights.get("skills", {})
    explanation["skill_analysis"] = {
        "direct_matches": skill_info.get("direct_matches", []),
        "related_matches": skill_info.get("related_matches", []),
        "missing_skills": skill_info.get("missing_skills", [])[:5],
        "match_rate": f"{skill_info.get('match_rate', 0)*100:.0f}%"
    }

    # ── Signal Analysis ──────────────────────────────────────
    explanation["signal_analysis"] = {
        "days_inactive": s.get("days_since_last_login", "N/A"),
        "response_rate": f"{int(s.get('recruiter_response_rate',0)*100)}%",
        "profile_completeness": f"{int(s.get('profile_completeness_score',0)*100)}%",
        "github_score": f"{s.get('github_activity_score',0):.2f}",
        "saved_by_recruiters": s.get("saved_by_recruiters_30d", 0),
        "interview_completion": f"{int(s.get('interview_completion_rate',0)*100)}%",
        "notice_period_days": s.get("notice_period_days", "N/A"),
        "willing_to_relocate": s.get("willing_to_relocate", False)
    }

    # ── Career Analysis ──────────────────────────────────────
    career_info = insights.get("career", {})
    explanation["career_analysis"] = {
        "overall": career_info.get("overall", "Unknown"),
        "seniority_trend": career_info.get("seniority_trend", "Unknown"),
        "ai_progression": career_info.get("ai_progression", "Unknown"),
        "consulting_status": career_info.get("consulting_analysis", "Unknown"),
        "stability": career_info.get("stability", "Unknown"),
        "recent_role": career_info.get("recent_role", "Unknown")
    }

    # ── Recruiter Note ───────────────────────────────────────
    # Generate a one-line recruiter-style note
    label = insights.get("label", "")
    if "Ideal" in label:
        note = (
            f"🌟 Top pick — {yoe}yr {title} with strong AI background, "
            f"India-based, highly active. Recommend immediate outreach."
        )
    elif "Rising" in label:
        note = (
            f"🚀 High potential — younger profile with strong growth "
            f"trajectory. Worth a screening call."
        )
    elif "Strong" in label:
        note = (
            f"✅ Solid candidate — good overall fit. "
            f"Review skill gaps before proceeding."
        )
    elif "Risky" in label:
        note = (
            f"⚠️ Proceed with caution — "
            f"possible consulting background or other concerns. "
            f"Verify product experience."
        )
    else:
        note = (
            f"➡️ Borderline candidate — "
            f"review manually before deciding."
        )

    explanation["recruiter_note"] = note

    return explanation


def generate_batch_explanations(top_candidates_data, jd_profile):
    """
    Generate explanations for all top candidates.

    Args:
        top_candidates_data: list of (cid, score, candidate, meta) tuples
        jd_profile: parsed JD

    Returns:
        list of explanation dicts
    """
    explanations = []
    for rank, (cid, score, candidate, meta) in enumerate(
        top_candidates_data, 1
    ):
        exp = explain_candidate(
            candidate=candidate,
            scores=meta["scores"],
            insights=meta["insights"],
            rank=rank,
            jd_profile=jd_profile
        )
        explanations.append(exp)
    return explanations


def format_explanation_text(explanation):
    """
    Format explanation as clean readable text.
    Used in Gradio UI to display candidate details.
    """
    lines = []
    lines.append(f"{'='*60}")
    lines.append(
        f"Rank #{explanation['rank']} | "
        f"{explanation['label']} | "
        f"Score: {explanation['score_breakdown']['final']}"
    )
    lines.append(f"{'='*60}")
    lines.append(f"📋 {explanation['summary']}")
    lines.append(f"\n💬 Recruiter Note: {explanation['recruiter_note']}")

    if explanation["strengths"]:
        lines.append("\n💪 Strengths:")
        for s in explanation["strengths"]:
            lines.append(f"  {s}")

    if explanation["weaknesses"]:
        lines.append("\n⚠️ Concerns:")
        for w in explanation["weaknesses"]:
            lines.append(f"  {w}")

    lines.append("\n📊 Score Breakdown:")
    scores = explanation["score_breakdown"]
    lines.append(f"  Semantic:  {scores['semantic']:.4f}")
    lines.append(f"  Skill:     {scores['skill']:.4f}")
    lines.append(f"  Signals:   {scores['signals']:.4f}")
    lines.append(f"  Rules:     {scores['rules']:.4f}")
    lines.append(f"  Career:    {scores['career']:.4f}")
    lines.append(f"  FINAL:     {scores['final']:.4f}")

    lines.append("\n🎯 Skill Analysis:")
    skill = explanation["skill_analysis"]
    lines.append(f"  Match rate: {skill['match_rate']}")
    if skill["direct_matches"]:
        lines.append(
            f"  Direct: {', '.join(skill['direct_matches'][:5])}"
        )
    if skill["related_matches"]:
        lines.append(
            f"  Related: {', '.join(skill['related_matches'][:3])}"
        )
    if skill["missing_skills"]:
        lines.append(
            f"  Missing: {', '.join(skill['missing_skills'][:3])}"
        )

    lines.append("\n📡 Activity Signals:")
    sig = explanation["signal_analysis"]
    lines.append(f"  Last active: {sig['days_inactive']} days ago")
    lines.append(f"  Response rate: {sig['response_rate']}")
    lines.append(f"  GitHub: {sig['github_score']}")
    lines.append(f"  Profile: {sig['profile_completeness']}")
    lines.append(
        f"  Notice period: {sig['notice_period_days']} days"
    )

    lines.append("\n📈 Career Trajectory:")
    career = explanation["career_analysis"]
    lines.append(f"  Overall: {career['overall']}")
    lines.append(f"  AI progression: {career['ai_progression']}")
    lines.append(f"  Consulting: {career['consulting_status']}")

    return "\n".join(lines)