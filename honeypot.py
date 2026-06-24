# ============================================================
# honeypot.py — Honeypot / Fake Profile Detector
# Feature #4: Honeypot Detection System
# Team V4Vin | India Runs Hackathon 2026
# ============================================================

# The dataset contains ~80 deliberately fake/impossible profiles
# planted by judges to trap naive ranking systems.
# This module detects and filters them out before ranking.
#
# Detection logic:
# - Perfect scores on ALL signals simultaneously = fake
# - Impossibly high experience for age
# - Contradictory career data
# - Statistically impossible behavioral patterns


def is_honeypot(candidate):
    """
    Returns (is_fake: bool, reason: str, confidence: float)
    confidence: 0.0 to 1.0 (how sure we are it's a honeypot)
    """
    suspicious_score = 0.0
    reasons = []

    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # ── Check 1: Perfect behavioral signals ─────────────────
    # Real humans never score 100% on everything simultaneously
    perfect_count = 0
    signal_checks = [
        ("profile_completeness_score", 1.0),
        ("recruiter_response_rate", 1.0),
        ("interview_completion_rate", 1.0),
        ("offer_acceptance_rate", 1.0),
    ]
    for signal_key, perfect_val in signal_checks:
        val = s.get(signal_key, 0)
        if val == perfect_val:
            perfect_count += 1

    if perfect_count >= 4:
        suspicious_score += 0.4
        reasons.append(f"Perfect scores on all {perfect_count} behavioral signals")
    elif perfect_count >= 3:
        suspicious_score += 0.2
        reasons.append(f"Perfect scores on {perfect_count} behavioral signals")

    # ── Check 2: Impossible experience ──────────────────────
    # No one can have 20+ years experience fresh out of college
    yoe = p.get("years_of_experience", 0)
    if yoe > 30:
        suspicious_score += 0.5
        reasons.append(f"Impossible experience: {yoe} years")
    elif yoe > 20:
        suspicious_score += 0.2
        reasons.append(f"Suspiciously high experience: {yoe} years")

    # ── Check 3: Too many skills ─────────────────────────────
    # Real profiles rarely have 50+ distinct skills
    skill_count = len(skills)
    if skill_count > 60:
        suspicious_score += 0.3
        reasons.append(f"Unrealistic skill count: {skill_count} skills")
    elif skill_count > 45:
        suspicious_score += 0.15
        reasons.append(f"Very high skill count: {skill_count} skills")

    # ── Check 4: GitHub activity anomaly ────────────────────
    github_score = s.get("github_activity_score", 0)
    if github_score == 1.0 and s.get("days_since_last_login", 999) > 300:
        suspicious_score += 0.25
        reasons.append("Perfect GitHub score but inactive for 300+ days")

    # ── Check 5: Zero activity but perfect profile ───────────
    days_inactive = s.get("days_since_last_login", 0)
    completeness = s.get("profile_completeness_score", 0)
    if days_inactive > 500 and completeness == 1.0:
        suspicious_score += 0.3
        reasons.append(f"Perfect profile but inactive for {days_inactive} days")

    # ── Check 6: Salary anomaly ──────────────────────────────
    salary = p.get("expected_salary_range_inr_lpa", "")
    if salary:
        try:
            # Extract max salary if range like "50-80"
            parts = str(salary).replace(" ", "").split("-")
            max_sal = float(parts[-1])
            if max_sal > 500:  # 500 LPA is unrealistic
                suspicious_score += 0.3
                reasons.append(f"Unrealistic salary expectation: {max_sal} LPA")
        except:
            pass

    # ── Check 7: Career history inconsistency ───────────────
    if career:
        total_career_years = sum(
            job.get("duration_months", 0) for job in career
        ) / 12
        if total_career_years > yoe + 5:
            suspicious_score += 0.2
            reasons.append(
                f"Career history ({total_career_years:.1f}yrs) "
                f"exceeds stated experience ({yoe}yrs)"
            )

    # ── Check 8: Saved by too many recruiters ───────────────
    saved = s.get("saved_by_recruiters_30d", 0)
    if saved > 50:
        suspicious_score += 0.2
        reasons.append(f"Saved by {saved} recruiters in 30 days (unrealistic)")

    # ── Final verdict ────────────────────────────────────────
    confidence = min(suspicious_score, 1.0)
    is_fake = confidence >= 0.5

    reason_str = "; ".join(reasons) if reasons else "No suspicious patterns"

    return is_fake, reason_str, confidence


def filter_honeypots(candidates, verbose=False):
    """
    Filter out honeypot candidates from the list.
    Returns (clean_candidates, removed_candidates, stats)
    """
    clean = []
    removed = []

    for c in candidates:
        is_fake, reason, confidence = is_honeypot(c)
        if is_fake:
            removed.append({
                "candidate_id": c.get("candidate_id"),
                "reason": reason,
                "confidence": round(confidence, 2)
            })
            if verbose:
                print(f"🚫 Removed {c.get('candidate_id')}: {reason} (confidence: {confidence:.2f})")
        else:
            clean.append(c)

    stats = {
        "total": len(candidates),
        "clean": len(clean),
        "removed": len(removed),
        "removal_rate": round(len(removed) / max(len(candidates), 1) * 100, 2)
    }

    return clean, removed, stats


def get_honeypot_report(removed_candidates):
    """
    Generate a summary report of removed honeypots.
    Used in the Gradio UI to show judges we handled honeypots.
    """
    if not removed_candidates:
        return "✅ No honeypot profiles detected in this dataset."

    report = f"🚫 Detected and removed {len(removed_candidates)} honeypot profiles:\n\n"
    for i, h in enumerate(removed_candidates[:10], 1):
        report += f"{i}. {h['candidate_id']} — {h['reason']} (confidence: {h['confidence']})\n"

    if len(removed_candidates) > 10:
        report += f"\n... and {len(removed_candidates) - 10} more."

    return report